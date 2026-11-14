"""Redaction engine: host identifiers, home paths, emails, IPs, phones, secrets.

Stdlib only. Contains NO literal identities: those come from a local-only file
(~/.config/git/redaction/identities.local) that is generated from ~/.ssh/config,
`tailscale status`, local account names and repository .env files.

The same file is copied into repos as a scrubber (lexsi_unlearn/scrub.py,
apart_epistemics/scrub.py), so it must stay free of personal values.

Placeholders (stable, so repeated redaction is idempotent):
  <HOST_SPARK> <HOST_GPU> <HOST> <HOST_LOCAL> <TAILNET_HOST> <TAILNET> <SSH_TARGET>
  <USER> <HOME> <EMAIL> <IP> <PHONE> and, for traces, <REDACTED:NAME>.
"""
from __future__ import annotations

import fnmatch
import ipaddress
import json
import os
import re
import shutil
import subprocess
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

REDACTION_DIR = Path.home() / ".config" / "git" / "redaction"
IDENTITIES_FILE = REDACTION_DIR / "identities.local"
MANUAL_FILE = REDACTION_DIR / "identities.manual"
GLOBAL_ALLOW_FILE = REDACTION_DIR / "allow.local"
QUARANTINE_DIR = REDACTION_DIR / "quarantine"

PLACEHOLDER = {
    "HOST_SPARK": "<HOST_SPARK>",
    "HOST_GPU": "<HOST_GPU>",
    "HOST": "<HOST>",
    "HOST_LOCAL": "<HOST_LOCAL>",
    "USER": "<USER>",
    "TAILNET": "<TAILNET_HOST>",
}
HOST_KINDS = set(PLACEHOLDER)

# ---------------------------------------------------------------- file classes
LOCKFILE_NAMES = {
    "package-lock.json", "yarn.lock", "pnpm-lock.yaml", "poetry.lock", "uv.lock", "pipfile.lock",
    "cargo.lock", "gemfile.lock", "composer.lock", "go.sum", "bun.lockb", "npm-shrinkwrap.json",
}
LOCKFILE_SUFFIXES = (".lock",)
MAX_TEXT = 8 * 1024 * 1024

SECRET_FILENAME_RES = [
    (re.compile(r"(^|/)\.env$"), "env-file"),
    (re.compile(r"(^|/)\.env\.(?!example$|sample$|template$|dist$)[^/]+$"), "env-file"),
    (re.compile(r"(^|/)[^/]*\.env$"), "env-file"),
    (re.compile(r"\.(pem|key|p12|pfx|jks|keystore|kdbx|ppk|gpg|asc)$", re.I), "key-file"),
    (re.compile(r"(^|/)id_(rsa|dsa|ecdsa|ed25519)[^/]*$"), "ssh-private-key"),
    (re.compile(r"(^|/)id_[A-Za-z0-9_-]+$"), "ssh-private-key"),
    (re.compile(r"(^|/)(credentials?|credentials?\.(json|ya?ml|toml|ini|txt)|\.git-credentials|\.netrc|\.pypirc|\.pgpass|\.htpasswd)$", re.I), "credentials-file"),
    (re.compile(r"(^|/)(secrets?\.(json|ya?ml|toml|ini|txt|env)|service[-_]account[^/]*\.json|kubeconfig|terraform\.tfstate[^/]*)$", re.I), "credentials-file"),
    (re.compile(r"(^|/)\.prepublish_identities$"), "identity-list"),
    (re.compile(r"(^|/)identities\.(local|manual)$"), "identity-list"),
]
ENV_TEMPLATE_RE = re.compile(r"(^|/)\.env\.(example|sample|template|dist)$")

SECRET_KEY_RE = re.compile(r"(?i)(secret|token|passw(or)?d|passwd|api[_-]?key|private[_-]?key|access[_-]?key|credential|auth|(^|_)key($|_)|(^|_)pat($|_))")
NOT_SECRET_KEY_RE = re.compile(r"(?i)((max|min|num|n|limit|count|budget|size|len|length|ttl|expir\w*|per|total|top|new|prompt|completion|input|output|eos|bos|pad|unk)[_-]?tokens?($|[_-])|tokens?[_-]?(limit|count|budget|max|min|size|len|length|ttl|per|id|ids|type)($|[_-]))")


def is_secret_env(key: str, value: str) -> bool:
    """A key/value pair that is plausibly a credential (not a numeric limit, flag or placeholder)."""
    v = value.strip()
    if not SECRET_KEY_RE.search(key) or NOT_SECRET_KEY_RE.search(key):
        return False
    if len(v) < 8 or _looks_placeholder(v) or re.fullmatch(r"[\d.,:_+-]+", v) or v.lower() in ("true", "false", "none", "null"):
        return False
    return True


PLACEHOLDER_VALUE_RE = re.compile(
    r"(?i)^(|\$\{?[A-Za-z_][A-Za-z0-9_]*\}?|<[^>]*>|\[[^\]]*\]|x{3,}|\*{3,}|\.{3,}|your[-_ ].*|changeme.*|replace[-_ ].*|paste[-_ ].*|"
    r"example.*|placeholder.*|none|null|false|true|todo.*|redacted.*|\[redacted.*|fill[-_ ]?me.*|dummy.*|test|secret|token|password)$")

# ------------------------------------------------------------- secret patterns
SECRET_PATTERNS = [
    ("private-key-block", re.compile(r"-----BEGIN (?:[A-Z0-9]+ )*PRIVATE KEY(?: BLOCK)?-----")),
    ("huggingface-token", re.compile(r"\bhf_[A-Za-z0-9]{20,}")),
    ("openrouter-key", re.compile(r"\bsk-or-(?:v1-)?[A-Za-z0-9]{16,}")),
    ("openai-style-key", re.compile(r"\bsk-(?:proj-|ant-)?[A-Za-z0-9_-]{24,}")),
    ("github-token", re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,})")),
    ("aws-access-key", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")),
    ("slack-token", re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}")),
    ("google-api-key", re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b")),
    ("tailscale-key", re.compile(r"\btskey-[A-Za-z0-9-]{16,}")),
    ("wandb-key", re.compile(r"(?i)\bwandb[_-]?(?:api[_-]?)?key\s*[:=]\s*['\"]?[0-9a-f]{40}\b")),
    ("bearer-token", re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]{20,}")),
    ("jwt", re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}")),
    ("url-credentials", re.compile(r"\b[a-z][a-z0-9+.-]*://[^\s/:@'\"]+:[^\s/@'\"]{4,}@[^\s/'\"]+")),
    ("npm-auth-token", re.compile(r"_authToken\s*=\s*\S{8,}")),
]
SECRET_KEYWORDS = {
    "private-key-block": ("PRIVATE KEY",), "huggingface-token": ("hf_",), "openrouter-key": ("sk-or-",), "openai-style-key": ("sk-",),
    "github-token": ("ghp_", "gho_", "ghu_", "ghs_", "ghr_", "github_pat_"), "aws-access-key": ("AKIA", "ASIA"), "slack-token": ("xox",),
    "google-api-key": ("AIza",), "tailscale-key": ("tskey-",), "wandb-key": ("wandb",), "bearer-token": ("earer", "EARER"),
    "jwt": ("eyJ",), "url-credentials": ("://",), "npm-auth-token": ("_authToken",),
}
ASSIGN_HINT_RE = re.compile(r"(?i)(api[_-]?key|secret|token|passw|private[_-]?key|access[_-]?key|auth[_-]?key|credential)")
ASSIGN_RE = re.compile(
    r"""(?ix)\b(?P<key>[A-Z0-9_.-]*(?:API[_-]?KEY|SECRET|TOKEN|PASSW(?:OR)?D|PASSWD|PRIVATE[_-]?KEY|ACCESS[_-]?KEY|AUTH[_-]?KEY|CREDENTIAL)[A-Z0-9_.-]*)
    ["']?[ \t]*[:=][ \t]*(?P<q>["']?)(?P<val>[A-Za-z0-9_+/=-]{16,})(?P=q)""")
HASHISH_KEY_RE = re.compile(r"(?i)(hash|sha\d*|digest|checksum|commit|fingerprint)")


def _looks_placeholder(value: str) -> bool:
    return bool(PLACEHOLDER_VALUE_RE.match(value.strip().strip("'\"")))


# ----------------------------------------------------------------- PII regexes
HOME_RE = re.compile(r"(?<![\w.~$-])/(?P<root>Users|home)/(?P<name>[^/\s\"'`<>:;,()\[\]{}\\|*?$]+)")
WIN_HOME_RE = re.compile(r"(?<![\w])(?P<drive>[A-Za-z]:)[\\/]+Users[\\/]+(?P<name>[^\\/\s\"'`<>:;,()|*?$]+)")
HOME_OK_NAMES = {
    "you", "your_user", "your-user", "yourname", "your_name", "your-name", "user", "username", "name", "me", "runner", "ubuntu",
    "shared", "guest", "example", "foo", "bar", "alice", "bob", "jovyan", "vscode", "node", "app", "<user>", "~",
    "xxx", "xxxx", "home", "localuser", "someone", "yourusername", "your_username", "your-username", "$user",
}
EMAIL_RE = re.compile(r"(?<![\w.+%-])[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}\b")
NOT_TLD = {
    "png", "jpg", "jpeg", "gif", "svg", "webp", "ico", "css", "js", "ts", "tsx", "jsx", "json", "py", "md", "txt", "yaml", "yml",
    "toml", "html", "htm", "pdf", "csv", "tsv", "npy", "npz", "pt", "bin", "sh", "tex", "bib", "log", "so", "dll", "whl", "gz", "zip", "tar",
}
PLACEHOLDER_LOCALS = {"user", "username", "you", "yourname", "your-name", "your.name", "name", "foo", "bar", "someone", "jane.doe", "john.doe", "first.last", "email", "your_email", "youremail"}
SAFE_EMAIL_DOMAINS = ("example.com", "example.org", "example.net", "example.edu", "users.noreply.github.com")
SAFE_EMAIL_RES = [
    re.compile(r"(?i)^(?:no-?reply|donotreply|do-not-reply)(?:\+[^@]*)?@"),
    re.compile(r"(?i)@users\.noreply\.github\.com$"),
    re.compile(r"(?i)^git@(?:github\.com|gitlab\.com|bitbucket\.org|ssh\.dev\.azure\.com)$"),
    re.compile(r"(?i)@(?:[A-Za-z0-9-]+\.)*(?:example\.(?:com|org|net|edu)|invalid|test|localhost)$"),
]
IPV4_RE = re.compile(r"(?<![\w.])(?<!\d\.)(?:\d{1,3}\.){3}\d{1,3}(?![\w]|\.\d)")
IPV6_RE = re.compile(r"(?<![\w:])(?:f[cd][0-9a-f]{2}|fe80)(?::[0-9a-fA-F]{0,4}){2,7}(?![\w:])", re.I)
SAFE_IPS = {"0.0.0.0", "255.255.255.255", "8.8.8.8", "8.8.4.4", "1.1.1.1", "1.0.0.1", "9.9.9.9"}
SAFE_NETS = [ipaddress.ip_network(n) for n in (
    "127.0.0.0/8", "192.0.2.0/24", "198.51.100.0/24", "203.0.113.0/24", "224.0.0.0/4", "169.254.0.0/16", "240.0.0.0/4")]
PHONE_RES = [
    re.compile(r"(?<![\w.+-])(?:\+?1[ .-])?(?:\(\d{3}\)[ .-]?|\d{3}[ .-])\d{3}[ .-]\d{4}(?![\w-])"),
    re.compile(r"(?<![\w.-])\+\d{1,3}[ .-]\(?\d{1,4}\)?[ .-]\d{2,4}[ .-]\d{2,4}(?:[ .-]\d{2,4})?(?![\w-])"),
]
FICTIONAL_PHONE_RE = re.compile(r"555[ .-](?:01\d\d|0100)\b|\b555-?0\d\d\d\b")
TS_NET_RE = re.compile(r"(?<![\w.-])(?:[A-Za-z0-9-]+\.)*[A-Za-z0-9-]+\.ts\.net(?![\w-])", re.I)
TS_NET_PLACEHOLDER_RE = re.compile(r"(?i)(xxx|example|your|tail1234|<|\{|\$|tailnet-name|tailnet\.ts)")
MDNS_RE = re.compile(r"(?<![\w.-])[A-Za-z0-9]+(?:-[A-Za-z0-9]+)+\.local(?![\w-])")  # hyphenated mDNS names only; never file stems like x.local
SSH_TARGET_RE = re.compile(r"(?<![\w.+-])([A-Za-z0-9._+-]+)@(<(?:TAILNET_HOST|HOST[A-Z_]*)>)")

PLACEHOLDER_TARGET_RE = re.compile(r"(?<![\w.+-])(<(?:HOST_SPARK|HOST_GPU|HOST|HOST_LOCAL)>)@(<(?:TAILNET_HOST|HOST[A-Z_]*)>)")
CONTEXT_PREFIX = r"(?:(?<=/Users/)|(?<=/home/)|(?<=[\w.+-]@)|(?<=ssh )|(?<=scp )|(?<=-l )|(?<=-u )|(?<=user=)|(?<=USER=)|(?<=user: )|(?<=User )|(?<=host=)|(?<=HOST=)|(?<=Host ))"


# ------------------------------------------------------------------- identities
@dataclass(frozen=True)
class Identity:
    kind: str       # key of PLACEHOLDER
    mode: str       # "bare" (anywhere) or "ctx" (only in path/ssh/user contexts)
    value: str


def _safe_name_key(value: str) -> str:
    return value.strip().lower()


GENERIC_WORDS = {
    "media", "server", "host", "localhost", "local", "gpu", "root", "admin", "user", "home", "mac", "macbook", "ubuntu", "debian",
    "pi", "box", "node", "main", "dev", "prod", "test", "spark", "git", "data", "work", "default", "ssh", "tailscale", "stream", "hub", "lab", "labs",
}


def default_mode(value: str) -> str:
    """bare = redact anywhere; ctx = only in path/ssh/user contexts (short or dictionary-like names)."""
    v = value.lower()
    if v in GENERIC_WORDS:
        return "ctx"
    if re.search(r"[-_0-9.]", v) and len(v) >= 5:
        return "bare"
    return "bare" if len(v) >= 9 else "ctx"


def parse_identities(text: str) -> list[Identity]:
    out, seen = [], set()
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t") if "\t" in line else line.split(None, 2)
        if len(parts) == 2:
            kind, value = parts
            mode = ""
        elif len(parts) >= 3:
            kind, mode, value = parts[0], parts[1], parts[2]
        else:
            continue
        kind = kind.upper()
        value = value.strip()
        if kind not in HOST_KINDS or not value or len(value) < 3:
            continue
        mode = mode if mode in ("bare", "ctx") else default_mode(value)
        key = (kind, value.lower())
        if key in seen:
            continue
        seen.add(key)
        out.append(Identity(kind, mode, value))
    return out


def load_identities(path: Path | None = None) -> list[Identity]:
    p = Path(path) if path else IDENTITIES_FILE
    try:
        return parse_identities(p.read_text(encoding="utf-8"))
    except OSError:
        return []


def _ssh_config_entries(path: Path):
    """Yield (alias, hostname, user) triples from an ssh_config, ignoring wildcards."""
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return
    block = None
    blocks = []
    for raw in lines:
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        m = re.match(r"(?i)^(\w+)\s*[= ]\s*(.+)$", line)
        if not m:
            continue
        key, val = m.group(1).lower(), m.group(2).strip()
        if key == "host":
            block = {"aliases": [a for a in val.split() if not any(c in a for c in "*?!")], "hostname": None, "user": None}
            blocks.append(block)
        elif key == "match":
            block = None
        elif block is not None and key == "hostname":
            block["hostname"] = val
        elif block is not None and key == "user":
            block["user"] = val
    for b in blocks:
        for alias in b["aliases"]:
            yield alias, b["hostname"], b["user"]


def _env_pairs(path: Path):
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return {}
    pairs = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[7:]
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip()
        if len(v) >= 2 and v[0] == v[-1] and v[0] in "'\"":
            v = v[1:-1]
        elif " #" in v:
            v = v.split(" #", 1)[0].strip()
        pairs[k] = v
    return pairs


def env_files(root: Path | None) -> list[Path]:
    if root is None:
        return []
    out = []
    try:
        for p in sorted(Path(root).glob(".env*")):
            if p.is_file() and not ENV_TEMPLATE_RE.search(p.name):
                out.append(p)
    except OSError:
        pass
    return out


def secret_values_from_env(root: Path | None) -> dict[str, str]:
    """NAME -> value for secret-looking keys in the repo's real .env files."""
    values = {}
    for f in env_files(root):
        for k, v in _env_pairs(f).items():
            if is_secret_env(k, v):
                values[k] = v
    return values


def _split_target(value: str):
    value = value.strip()
    if "@" in value and "://" not in value:
        user, host = value.split("@", 1)
        return user, host.split(":", 1)[0]
    return None, value.split(":", 1)[0]


def classify_rules(manual: Path | None = None) -> list:
    """`CLASSIFY<TAB>KIND<TAB>regex` lines in identities.manual decide which kind a host name gets."""
    rules = []
    try:
        text = Path(manual or MANUAL_FILE).read_text(encoding="utf-8")
    except OSError:
        return rules
    for raw in text.splitlines():
        parts = raw.rstrip("\n").split("\t")
        if len(parts) == 3 and parts[0].upper() == "CLASSIFY" and parts[1].upper() in HOST_KINDS:
            try:
                rules.append((parts[1].upper(), re.compile(parts[2], re.I)))
            except re.error:
                pass
    return rules


def classify_name(name: str, is_self: bool = False, rules=None) -> str:
    for kind, rx in (rules if rules is not None else classify_rules()):
        if rx.search(name):
            return kind
    if re.search(r"(?i)spark", name):
        return "HOST_SPARK"
    if re.search(r"(?i)gpu", name):
        return "HOST_GPU"
    return "HOST_LOCAL" if is_self else "HOST"


def _tailscale_entries(timeout: float = 5.0):
    exe = shutil.which("tailscale") or "/Applications/Tailscale.app/Contents/MacOS/Tailscale"
    if not exe or not os.path.exists(exe):
        return None
    try:
        res = subprocess.run([exe, "status", "--json"], capture_output=True, timeout=timeout, text=True)
        data = json.loads(res.stdout)
    except (OSError, subprocess.SubprocessError, ValueError):
        return None
    out = []
    nodes = [data.get("Self") or {}] + list((data.get("Peer") or {}).values())

    rules = classify_rules()

    def classify(name, is_self):
        return classify_name(name, is_self, rules)

    for n in nodes:
        is_self = n is nodes[0]
        dns = (n.get("DNSName") or "").rstrip(".")
        short = dns.split(".", 1)[0] if dns else ""
        if dns:
            out.append(Identity("TAILNET", "bare", dns))
        for name in {short, n.get("HostName") or ""}:
            if name and re.fullmatch(r"[A-Za-z0-9._-]{3,}", name):
                out.append(Identity(classify(name, is_self), default_mode(name), name))
    suffix = (data.get("MagicDNSSuffix") or "").strip(".")
    if suffix:
        out.append(Identity("TAILNET", "bare", suffix))
    return out


def build_identities(*, repos=(), ssh_config: Path | None = None, manual: Path | None = None,
                     use_tailscale: bool = True, extra_ips_from_tailscale: bool = True) -> list[Identity]:
    """Derive the denylist. Nothing here is written into any repo."""
    ids: list[Identity] = []

    def add(kind, value, mode=None):
        value = (value or "").strip()
        if len(value) < 3:
            return
        ids.append(Identity(kind, mode or default_mode(value), value))

    crules = [] if manual is False else classify_rules(manual)
    ssh = ssh_config or (Path.home() / ".ssh" / "config")
    for alias, hostname, user in _ssh_config_entries(ssh):
        kind = classify_name(alias, False, crules)
        if kind == "HOST_LOCAL":
            kind = "HOST"
        add(kind, alias)
        if hostname and hostname.lower() != alias.lower():
            add(kind, hostname)
        if user:
            add("USER", user)
    # local account names; the macOS account name doubles as a first name, so it is path/ssh-context only.
    for name in {os.environ.get("USER", ""), Path.home().name}:
        if name:
            add("USER", name, "ctx")
    for cmd in (["scutil", "--get", "LocalHostName"], ["hostname"]):
        try:
            out = subprocess.run(cmd, capture_output=True, timeout=3, text=True).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            out = ""
        if out:
            add("HOST_LOCAL", out)
            add("HOST_LOCAL", out.split(".")[0])
    for repo in repos:
        for f in env_files(Path(repo)):
            for k, v in _env_pairs(f).items():
                if not v or _looks_placeholder(v) or SECRET_KEY_RE.search(k):
                    continue
                ku = k.upper()
                is_spark = "SPARK" in ku
                if re.search(r"(HOST|SSH|REMOTE|SERVER|ADDR)", ku) and "DIR" not in ku and "URL" not in ku and "PORT" not in ku:
                    user, host = _split_target(v)
                    kind = "HOST_SPARK" if is_spark else ("HOST_GPU" if "REMOTE" in ku or "SERVER" in ku else "HOST")
                    if host and not re.fullmatch(r"(?:localhost|127\.0\.0\.1|0\.0\.0\.0)", host, re.I):
                        add(kind, host)
                    if user:
                        add("USER", user)
                elif re.search(r"(^|_)USER(NAME)?$", ku):
                    add("USER", v)
    if manual is not False:
        ids.extend(parse_identities(Path(manual or MANUAL_FILE).read_text(encoding="utf-8")) if Path(manual or MANUAL_FILE).exists() else [])
    if use_tailscale:
        ts = _tailscale_entries()
        if ts:
            ids.extend(ts)
        else:  # keep previous tailnet knowledge if the CLI is unavailable
            ids.extend(i for i in load_identities() if i.kind == "TAILNET")
    return parse_identities(serialize(ids))


def serialize(ids) -> str:
    return "".join(f"{i.kind}\t{i.mode}\t{i.value}\n" for i in ids)


def write_identities(ids, path: Path | None = None) -> Path:
    p = Path(path) if path else IDENTITIES_FILE
    p.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(p.parent, 0o700)
    except OSError:
        pass
    header = "# Local-only identity denylist (generated). Do not commit or copy into any repo.\n# kind<TAB>mode<TAB>value. Add your own lines to identities.manual.\n"
    tmp = p.with_suffix(".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as fh:
        fh.write(header + serialize(ids))
    os.replace(tmp, p)
    os.chmod(p, 0o600)
    return p


def identities_stale(repos=(), path: Path | None = None, max_age: float = 86400.0) -> bool:
    p = Path(path) if path else IDENTITIES_FILE
    try:
        built = p.stat().st_mtime
    except OSError:
        return True
    if time.time() - built > max_age:
        return True
    sources = [Path.home() / ".ssh" / "config", MANUAL_FILE]
    for repo in repos:
        sources.extend(env_files(Path(repo)))
    for s in sources:
        try:
            if s.stat().st_mtime > built:
                return True
        except OSError:
            continue
    return False


REPOS_FILE = REDACTION_DIR / "repos.list"


def known_repos(extra=()) -> list:
    """Repos whose .env files feed the denylist: every repo the hook has seen plus `extra`."""
    seen = []
    try:
        seen = [Path(l.strip()) for l in REPOS_FILE.read_text().splitlines() if l.strip()]
    except OSError:
        pass
    out = []
    for r in [*seen, *map(Path, extra)]:
        if r not in out:
            out.append(r)
    return out


def register_repo(repo: Path) -> None:
    repo = Path(repo)
    if repo in known_repos():
        return
    try:
        REDACTION_DIR.mkdir(parents=True, exist_ok=True)
        with open(REPOS_FILE, "a") as fh:
            fh.write(str(repo) + "\n")
        os.chmod(REPOS_FILE, 0o600)
    except OSError:
        pass


def ensure_identities(repos=(), path: Path | None = None) -> list[Identity]:
    """Load identities.local, rebuilding it when ssh config / any known repo's .env / manual list changed."""
    p = Path(path) if path else IDENTITIES_FILE
    allrepos = known_repos(repos)
    if path is None:
        for r in repos:
            register_repo(r)
    if identities_stale(allrepos, p):
        try:
            write_identities(build_identities(repos=allrepos), p)
        except OSError:
            pass
    return load_identities(p)


# ----------------------------------------------------------------- allowlists
@dataclass
class Allow:
    emails: set = field(default_factory=set)
    email_domains: set = field(default_factory=set)
    path_globs: list = field(default_factory=list)   # (kind, glob)

    def email_ok(self, addr: str) -> bool:
        a = addr.lower()
        if a in self.emails:
            return True
        return a.rsplit("@", 1)[-1] in self.email_domains

    def path_ok(self, kind: str, path: str) -> bool:
        return any(k == kind and fnmatch.fnmatch(path, g) for k, g in self.path_globs)


ALLOW_PATH_KINDS = {"home", "ip", "phone", "email", "binary"}


def parse_allow(text: str) -> Allow:
    """.redaction-allow lines:  email <addr> | email-domain <domain> | path <kind> <glob>.

    Host identifiers and secrets can never be allowed; unknown lines are ignored.
    """
    allow = Allow()
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        parts = line.split()
        if parts[0] == "email" and len(parts) == 2:
            allow.emails.add(parts[1].lower())
        elif parts[0] == "email-domain" and len(parts) == 2:
            allow.email_domains.add(parts[1].lower().lstrip("@"))
        elif parts[0] == "path" and len(parts) == 3 and parts[1] in ALLOW_PATH_KINDS:
            allow.path_globs.append((parts[1], parts[2]))
    return allow


def merge_allow(*allows: Allow) -> Allow:
    out = Allow()
    for a in allows:
        out.emails |= a.emails
        out.email_domains |= a.email_domains
        out.path_globs.extend(a.path_globs)
    return out


def load_global_allow() -> Allow:
    try:
        return parse_allow(GLOBAL_ALLOW_FILE.read_text(encoding="utf-8"))
    except OSError:
        return Allow()


# --------------------------------------------------------------------- engine
def is_lockfile(path: str) -> bool:
    name = path.rsplit("/", 1)[-1].lower()
    return name in LOCKFILE_NAMES or name.endswith(LOCKFILE_SUFFIXES)


def is_binary(data: bytes) -> bool:
    return b"\x00" in data[:8192]


def filename_block_reason(path: str, data: bytes | None = None) -> str | None:
    """Return a reason if the file name itself must never be committed."""
    if ENV_TEMPLATE_RE.search(path):
        if data is not None:
            for key, val in _env_pairs_from_text(data.decode("utf-8", "replace")).items():
                if SECRET_KEY_RE.search(key) and not NOT_SECRET_KEY_RE.search(key) and not _looks_placeholder(val) \
                        and not re.fullmatch(r"[\d.,:_+-]+", val.strip()):
                    return "env-template-with-value"
        return None
    for rx, reason in SECRET_FILENAME_RES:
        if rx.search(path):
            return reason
    return None


def _env_pairs_from_text(text: str) -> dict:
    pairs = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[7:]
        k, v = line.split("=", 1)
        v = v.strip()
        if len(v) >= 2 and v[0] == v[-1] and v[0] in "'\"":
            v = v[1:-1]
        pairs[k.strip()] = v
    return pairs


class Engine:
    def __init__(self, identities=None, secret_values=None, allow: Allow | None = None):
        self.identities = list(identities or [])
        self.secret_values = {k: v for k, v in (secret_values or {}).items() if v}
        self.allow = allow or Allow()
        self._build_identity_regexes()

    def _build_identity_regexes(self):
        bare, ctx = {}, {}
        for i in sorted(self.identities, key=lambda i: -len(i.value)):
            (bare if i.mode == "bare" else ctx).setdefault(i.kind, []).append(re.escape(i.value))
        self._bare, self._ctx, self._needles = [], [], {}
        for kind, vals in bare.items():
            self._bare.append((kind, re.compile(r"(?<![A-Za-z0-9_])(?:" + "|".join(vals) + r")(?![A-Za-z0-9_])", re.I)))
            self._needles[("b", kind)] = tuple(re.sub(r"\\(.)", r"\1", v).lower() for v in vals)
        for kind, vals in ctx.items():
            alt = "|".join(vals)
            self._ctx.append((kind, re.compile(
                r"(?:" + CONTEXT_PREFIX + r"(?:" + alt + r")(?![A-Za-z0-9_])|(?<![A-Za-z0-9_@/.~-])(?:" + alt + r")(?=@[A-Za-z0-9<]))", re.I)))
            self._needles[("c", kind)] = tuple(re.sub(r"\\(.)", r"\1", v).lower() for v in vals)
        order = {"TAILNET": 0, "HOST_SPARK": 1, "HOST_GPU": 2, "HOST_LOCAL": 3, "HOST": 4, "USER": 5}
        self._bare.sort(key=lambda kr: order.get(kr[0], 9))
        self._ctx.sort(key=lambda kr: order.get(kr[0], 9))

    # ---- detection -------------------------------------------------------
    def _present(self, tag: str, kind: str, lowered: str) -> bool:
        return any(n in lowered for n in self._needles[(tag, kind)])

    def host_hits(self, text: str) -> Counter:
        c = Counter()
        low = text.lower()
        for tag, group in (("b", self._bare), ("c", self._ctx)):
            for kind, rx in group:
                if self._present(tag, kind, low):
                    c[kind] += len(rx.findall(text))
        return +c

    def find_secrets(self, text: str) -> list[tuple[str, int]]:
        """(rule, 1-based line) for every secret-looking item. Never returns values."""
        found = []
        for name, value in self.secret_values.items():
            idx = text.find(value)
            if idx >= 0:
                found.append((f"env-value:{name}", text.count("\n", 0, idx) + 1))
        for rule, rx in SECRET_PATTERNS:
            if not any(k in text for k in SECRET_KEYWORDS.get(rule, ("",))):
                continue
            for m in rx.finditer(text):
                found.append((rule, text.count("\n", 0, m.start()) + 1))
        for m in (ASSIGN_RE.finditer(text) if ASSIGN_HINT_RE.search(text) else ()):
            val = m.group("val")
            if _looks_placeholder(val) or not (re.search(r"[A-Za-z]", val) and re.search(r"\d", val)):
                continue
            if HASHISH_KEY_RE.search(m.group("key")) and re.fullmatch(r"[0-9a-fA-F]{32,}", val):
                continue
            found.append(("secret-assignment", text.count("\n", 0, m.start()) + 1))
        return found

    # ---- redaction --------------------------------------------------------
    def redact(self, text: str, path: str = "", *, phones: bool = True, emails: bool = True) -> tuple[str, Counter]:
        counts: Counter = Counter()

        def sub(rx, repl, kind, src, guard=None):
            def _r(m):
                if guard and not guard(m):
                    return m.group(0)
                counts[kind] += 1
                return repl(m) if callable(repl) else repl
            return rx.sub(_r, src)

        out = text
        low = text.lower()
        if ".ts.net" in low:
            out = sub(TS_NET_RE, "<TAILNET_HOST>", "TAILNET", out, guard=lambda m: not TS_NET_PLACEHOLDER_RE.search(m.group(0)))
        if ("users" in low or "/home/" in low) and not self.allow.path_ok("home", path):
            out = sub(HOME_RE, lambda m: "<HOME>", "HOME", out, guard=lambda m: m.group("name").lower() not in HOME_OK_NAMES)
            out = sub(WIN_HOME_RE, lambda m: "<HOME>", "HOME", out, guard=lambda m: m.group("name").lower() not in HOME_OK_NAMES)
        for tag, group in (("b", self._bare), ("c", self._ctx)):
            for kind, rx in group:
                if self._present(tag, kind, low):
                    out = sub(rx, PLACEHOLDER[kind], kind, out)
        if ".local" in low:
            out = sub(MDNS_RE, "<HOST_LOCAL>", "HOST_LOCAL", out)
        if "<" in out and "@" in out:
            out = sub(PLACEHOLDER_TARGET_RE, lambda m: "<USER>@" + m.group(2), "USER", out)
        if "@" in out:
            out = sub(SSH_TARGET_RE, lambda m: (m.group(1) if m.group(1).startswith("<") else "<USER>") + "@" + m.group(2), "USER", out,
                      guard=lambda m: m.group(1).lower() not in ("git",))
            if emails and not self.allow.path_ok("email", path):
                out = sub(EMAIL_RE, "<EMAIL>", "EMAIL", out, guard=lambda m: not self.email_is_safe(m.group(0)))
        if not self.allow.path_ok("ip", path):
            out = sub(IPV4_RE, "<IP>", "IP", out, guard=lambda m: not self._ip_is_safe(m.group(0)))
            if ":" in out:
                out = sub(IPV6_RE, "<IP>", "IP", out)
        if phones and not self.allow.path_ok("phone", path):
            for rx in PHONE_RES:
                out = sub(rx, "<PHONE>", "PHONE", out, guard=lambda m: not FICTIONAL_PHONE_RE.search(m.group(0)))
        return out, counts

    def email_is_safe(self, addr: str) -> bool:
        if self.allow.email_ok(addr):
            return True
        local, _, domain = addr.lower().rpartition("@")
        if local in PLACEHOLDER_LOCALS or TS_NET_PLACEHOLDER_RE.search(domain):
            return True
        tld = addr.rsplit(".", 1)[-1].lower()
        if tld in NOT_TLD:
            return True
        return any(rx.search(addr) for rx in SAFE_EMAIL_RES)

    @staticmethod
    def _ip_is_safe(text: str) -> bool:
        if text in SAFE_IPS:
            return True
        try:
            ip = ipaddress.IPv4Address(text)
        except ValueError:
            return True  # not an address (e.g. 999.1.1.1 or a version string)
        if any(ip in net for net in SAFE_NETS):
            return True
        return False

    def scrub(self, text: str, path: str = "") -> str:
        """Trace/log scrub: env secrets -> <REDACTED:NAME>, token patterns -> <REDACTED:rule>, then PII."""
        out = text
        for name, value in sorted(self.secret_values.items(), key=lambda kv: -len(kv[1])):
            out = out.replace(value, f"<REDACTED:{name}>")
        for rule, rx in SECRET_PATTERNS:
            if any(k in out for k in SECRET_KEYWORDS.get(rule, ("",))):
                out = rx.sub(f"<REDACTED:{rule}>", out)

        def _assign(m):
            val = m.group("val")
            if _looks_placeholder(val) or not (re.search(r"[A-Za-z]", val) and re.search(r"\d", val)):
                return m.group(0)
            if HASHISH_KEY_RE.search(m.group("key")) and re.fullmatch(r"[0-9a-fA-F]{32,}", val):
                return m.group(0)
            return m.group(0).replace(val, "<REDACTED:secret-assignment>")
        if ASSIGN_HINT_RE.search(out):
            out = ASSIGN_RE.sub(_assign, out)
        out, _ = self.redact(out, path, phones=False)
        return out


def would_change_json(before: str, after: str, path: str) -> bool:
    """True when `after` is no longer valid JSON/JSONL although `before` was."""
    low = path.lower()
    try:
        if low.endswith((".json", ".ipynb")):
            json.loads(before)
            try:
                json.loads(after)
            except ValueError:
                return True
        elif low.endswith((".jsonl", ".ndjson")):
            for ln in before.splitlines():
                if ln.strip():
                    json.loads(ln)
            for ln in after.splitlines():
                if ln.strip():
                    try:
                        json.loads(ln)
                    except ValueError:
                        return True
    except ValueError:
        return False
    return False


def would_break_python(before: str, after: str, path: str) -> bool:
    if not path.lower().endswith(".py"):
        return False
    import ast
    try:
        ast.parse(before)
    except (SyntaxError, ValueError):
        return False
    try:
        ast.parse(after)
    except (SyntaxError, ValueError):
        return True
    return False


# -------------------------------------------------------------- repo-side API
_ENGINE_CACHE: dict = {}


def _default_root() -> Path | None:
    """Repo root = nearest parent of this file holding pyproject.toml (None when used from the hooks dir)."""
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").exists():
            return parent
    return None


def repo_engine(root: Path | None = None) -> Engine:
    """Engine for scripts inside a repo: local identities + identities/secrets from that repo's .env.

    Read-only: never rewrites identities.local (the git hook owns that file). Works on a machine that has
    no identities file (e.g. a GPU box) by deriving hosts from its own ~/.ssh/config and the repo .env.
    """
    root = Path(root) if root else _default_root()
    key = str(root)
    if key not in _ENGINE_CACHE:
        ids = load_identities()
        try:
            ids = parse_identities(serialize(ids + build_identities(repos=[root] if root else [], use_tailscale=False, manual=False)))
        except Exception:  # noqa: BLE001 - scrubbing must never crash a run
            pass
        _ENGINE_CACHE[key] = Engine(ids, {**secret_values_from_env(root), **_process_secret_values()}, load_global_allow())
    return _ENGINE_CACHE[key]


def _process_secret_values() -> dict:
    out = {}
    for k, v in os.environ.items():
        if (is_secret_env(k, v) and len(v) >= 12 and not v.startswith("/")
                and not re.search(r"(?i)(SOCK|PATH|_DIR|_FILE|URL|HOME|PWD|ASKPASS|SESSION|TTY|XPC)", k)):
            out[k] = v
    return out


def scrub_text(text: str, root: Path | None = None, path: str = "") -> str:
    """Scrub secrets (-> <REDACTED:NAME>) and host/PII identifiers. Falls back to token patterns only on error."""
    try:
        return repo_engine(root).scrub(text, path)
    except Exception:  # noqa: BLE001
        out = text
        for rule, rx in SECRET_PATTERNS:
            out = rx.sub(f"<REDACTED:{rule}>", out)
        return out


def scrub_obj(obj, root: Path | None = None):
    """Recursively scrub every string in a JSON-like structure (dict keys too). Structure is preserved."""
    if isinstance(obj, str):
        return scrub_text(obj, root)
    if isinstance(obj, dict):
        return {(scrub_text(k, root) if isinstance(k, str) else k): scrub_obj(v, root) for k, v in obj.items()}
    if isinstance(obj, list):
        return [scrub_obj(v, root) for v in obj]
    if isinstance(obj, tuple):
        return tuple(scrub_obj(v, root) for v in obj)
    return obj
