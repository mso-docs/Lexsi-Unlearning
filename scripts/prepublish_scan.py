#!/usr/bin/env python3
"""Build a redacted pre-publication secret and PII report.

Called by scripts/prepublish_scan.sh. Never prints secret values; every
snippet written to the report or stdout is redacted first.
"""
from __future__ import annotations

import argparse
import fnmatch
import ipaddress
import json
import os
import re
import socket
import stat
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterable
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")

SKIP_DIRS = {
    ".git",
    "__pycache__",
    ".venv",
    "venv",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".ipynb_checkpoints",
    "hf_cache",
    ".cache",
    "wandb",
    "lightning_logs",
    "checkpoints",
    ".idea",
    ".vscode",
    "node_modules",
    ".eggs",
}

# Destined for a public Hugging Face collection even when gitignored.
HF_DIR_PREFIXES = ("release/",)
HF_SUFFIXES = (".json", ".jsonl", ".md", ".csv", ".tsv", ".txt")

# Local-only secret / cache paths that must never be published.
LOCAL_ONLY_NAMES = {".env", ".env.local", ".prepublish_identities"}
LOCAL_ONLY_PREFIXES = (
    "hf_cache/",
    ".cache/",
    ".venv/",
    "venv/",
    "wandb/",
    "lightning_logs/",
    "checkpoints/",
    # Raw run outputs stay on the machine. Publish only the scrubbed trees
    # under release/ (Hugging Face) and docs/ (GitHub Pages).
    "artifacts/",
    "logs/",
    "traces/",
    "outputs/",
)

REQUIRED_IGNORED = [
    ".env",
    ".env.local",
    "artifacts/",
    "logs/",
    "traces/",
    "outputs/",
    "checkpoints/",
    "hf_cache/",
    ".cache/",
    "wandb/",
    ".prepublish_identities",
    "data/translations.json",
    "release/",
]

TEXT_EXT = {
    ".md", ".py", ".sh", ".bash", ".zsh", ".txt", ".toml", ".yaml", ".yml",
    ".json", ".jsonl", ".ipynb", ".tex", ".bib", ".sty", ".cls", ".rst",
    ".csv", ".tsv", ".html", ".htm", ".xml", ".ini", ".cfg", ".conf",
    ".example", ".in", ".r", ".jl", ".js", ".ts", ".css", ".gitignore",
    ".toml", ".log", ".lst", ".tex", ".beamer",
}
TEXT_NAMES = {
    ".gitignore", ".gitleaks.toml", ".gitleaksignore", ".prepublish_allowlist",
    ".prepublish_identities", ".prepublish_identities.example", ".env",
    ".env.example", ".env.local", "Dockerfile", "Makefile", "LICENSE",
    "LICENSE.md", "NOTICE",
}

PLACEHOLDER_PATH_USERS = {
    "you", "user", "username", "your-user", "your_user", "<user>", "<you>",
    "your-username", "somebody", "name",
}

GENERIC_IDENTITY_SKIP = {
    "root", "ubuntu", "admin", "user", "you", "nobody", "www-data",
    "local", "lan", "home", "users", "private", "var", "tmp", "host",
    "localhost", "spark", "tail", "com", "net", "org", "edu", "test",
    "example", "docker", "internal", "github", "gitlab", "bitbucket",
    "huggingface", "openai", "google", "macbook", "mac", "pro", "air",
    "documents", "document", "code", "desktop", "downloads", "library",
    "applications", "cache", ".cache", "artifacts", "workspace", "content",
    "opt", "usr", "etc", "bin", "src", "data", "logs", "traces", "outputs",
    "checkpoints", "presentation", "paper", "notebooks", "scripts", "configs",
}

EMAIL_RE = re.compile(r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b")
IPV4_RE = re.compile(
    r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)\b"
)
IPV6_RE = re.compile(r"\b(?:(?:[0-9a-fA-F]{1,4}:){2,7}[0-9a-fA-F]{1,4}|::1)\b")
USERS_PATH_RE = re.compile(r"/(?:Users|home)/([^/\s\"'`]+)")
ABS_USERS_RE = re.compile(r"/(?:Users|home)/[^/\s\"'`][^\s\"'`]*")
PHONE_US_RE = re.compile(
    r"(?<!\d)(?:\+1[-.\s]?)?(?:\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4})(?!\d)"
)
PHONE_INTL_RE = re.compile(
    r"(?<!\d)\+\d{1,3}[-.\s](?:\d{2,4}[-.\s]){1,3}\d{2,4}(?!\d)"
)
PRIVATE_KEY_RE = re.compile(r"-----BEGIN (?:[A-Z]+ )?PRIVATE KEY-----")
SSH_PUB_RE = re.compile(
    r"\b(?:ssh-ed25519|ssh-rsa|ecdsa-sha2-nistp256|ssh-dss) [A-Za-z0-9+/=]{20,}"
)
ENV_ASSIGN_RE = re.compile(
    r"(?i)\b(HF_TOKEN|OPENROUTER_API_KEY|WANDB_API_KEY|NGC_API_KEY|"
    r"AWS_SECRET_ACCESS_KEY|GITHUB_TOKEN|GH_TOKEN)\s*=\s*"
    r"(?P<val>(?:'[^']+'|\"[^\"]+\"|\S+))"
)
ENV_VAR_REF_RE = re.compile(r"^['\"]?\$\{?[A-Za-z_][A-Za-z0-9_]*\}?['\"]?$")

SECRET_PREFIX_RE = re.compile(
    r"(hf_[A-Za-z0-9]{8,}|sk-or-(?:v1-)?[A-Za-z0-9]{8,}|sk-[A-Za-z0-9]{8,}|"
    r"ghp_[A-Za-z0-9]{8,}|github_pat_[A-Za-z0-9_]{8,}|AKIA[A-Z0-9]{8,})"
)

BEARER_RE = re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._\-+=/]{8,}")
GEN_ID_RE = re.compile(r"\bgen-[A-Za-z0-9]{8,}\b")

MAX_FILE_BYTES = 10 * 1024 * 1024
SNIPPET_RADIUS = 80


@dataclass
class Finding:
    category: str
    path: str
    line: int
    raw: str
    snippet: str
    severity: str
    targets: list[str]
    rule: str = ""
    commit: str | None = None
    allowlisted: bool = False
    allow_reason: str = ""


@dataclass
class AllowRule:
    category: str
    path_glob: str
    regex: re.Pattern[str]
    reason: str


@dataclass
class GitignoreRow:
    path: str
    ignored: bool
    tracked: bool
    ok: bool
    detail: str


def relpath(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def redact_secret_token(text: str) -> str:
    def _sub(m: re.Match[str]) -> str:
        s = m.group(0)
        if s.startswith("hf_"):
            return "hf_[REDACTED]"
        if s.startswith("sk-or-"):
            return "sk-or-[REDACTED]"
        if s.startswith("sk-"):
            return "sk-[REDACTED]"
        if s.startswith("ghp_"):
            return "ghp_[REDACTED]"
        if s.startswith("github_pat_"):
            return "github_pat_[REDACTED]"
        if s.startswith("AKIA"):
            return "AKIA[REDACTED]"
        return "[SECRET]"

    return SECRET_PREFIX_RE.sub(_sub, text)


def redact_text(text: str, identities: list[str]) -> str:
    out = redact_secret_token(text)
    out = BEARER_RE.sub("Bearer [REDACTED]", out)
    out = GEN_ID_RE.sub("gen-[REDACTED]", out)
    out = PRIVATE_KEY_RE.sub("[PRIVATE_KEY]", out)
    out = SSH_PUB_RE.sub(lambda m: m.group(0).split()[0] + " [REDACTED]", out)
    out = EMAIL_RE.sub("[EMAIL]", out)
    out = PHONE_US_RE.sub("[PHONE]", out)
    out = PHONE_INTL_RE.sub("[PHONE]", out)

    def _path(m: re.Match[str]) -> str:
        body = m.group(0)
        return re.sub(r"/(Users|home)/[^/]+", r"/\1/[USER]", body)

    out = ABS_USERS_RE.sub(_path, out)

    def _ip(m: re.Match[str]) -> str:
        ip = m.group(0)
        if ip.startswith("127.") or ip in {"0.0.0.0", "255.255.255.255", "::1"}:
            return ip
        parts = ip.split(".")
        if len(parts) == 4:
            return f"{parts[0]}.{parts[1]}.[REDACTED]"
        return "[IP]"

    out = IPV4_RE.sub(_ip, out)
    # Identities last, longest first, so short names don't split longer ones.
    for ident in sorted(set(identities), key=len, reverse=True):
        if len(ident) < 3:
            continue
        out = re.sub(
            r"(?i)(?<![A-Za-z0-9_.-])" + re.escape(ident) + r"(?![A-Za-z0-9_.-])",
            "[IDENTITY]",
            out,
        )
    return out


def snippet_around(line: str, match: str, identities: list[str]) -> str:
    idx = line.find(match)
    if idx < 0:
        cut = line.strip()
    else:
        start = max(0, idx - SNIPPET_RADIUS)
        end = min(len(line), idx + len(match) + SNIPPET_RADIUS)
        cut = line[start:end].strip()
        if start > 0:
            cut = "…" + cut
        if end < len(line):
            cut = cut + "…"
    return redact_text(cut, identities)


def load_allowlist(path: Path) -> list[AllowRule]:
    rules: list[AllowRule] = []
    if not path.is_file():
        return rules
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = raw.split("\t")
        if len(parts) < 4:
            # also allow multiple spaces for convenience
            parts = re.split(r"\t+", raw.strip())
        if len(parts) < 4:
            continue
        cat, glob, regex, reason = parts[0].strip(), parts[1].strip(), parts[2], parts[3].strip()
        try:
            compiled = re.compile(regex)
        except re.error:
            compiled = re.compile(re.escape(regex))
        rules.append(AllowRule(cat, glob, compiled, reason))
    return rules


def is_allowlisted(finding: Finding, rules: list[AllowRule]) -> tuple[bool, str]:
    for rule in rules:
        if rule.category not in {"*", finding.category}:
            continue
        if rule.path_glob != "*" and not fnmatch.fnmatch(finding.path, rule.path_glob):
            continue
        if rule.regex.search(finding.raw):
            return True, rule.reason
    return False, ""



def _add_identity(bucket: list[str], seen: set[str], value: str, *, repo_name: str = "") -> None:
    val = (value or "").strip().strip("\"'")
    if not val:
        return
    # Never ingest absolute or relative paths as identities.
    if "/" in val or val.startswith(".") or val.startswith("$"):
        return
    if "XXXX" in val or val.upper().startswith("YOUR_"):
        return
    if len(val) < 3:
        return
    key = val.lower()
    if key in GENERIC_IDENTITY_SKIP or key in seen:
        return
    if repo_name and key == repo_name.lower():
        return
    if re.fullmatch(r"\d+", val):
        return
    seen.add(key)
    bucket.append(val)


def _users_from_path(path_val: str) -> list[str]:
    found = []
    for m in USERS_PATH_RE.finditer(path_val or ""):
        name = m.group(1)
        if name.lower() in PLACEHOLDER_PATH_USERS or name.upper() in {"YOU", "USER"}:
            continue
        if name.startswith("<") or name.startswith("[") or name.startswith("$"):
            continue
        found.append(name)
    return found


def load_identities(root: Path, identities_path: Path | None) -> list[str]:
    bucket: list[str] = []
    seen: set[str] = set()
    repo_name = root.name

    if identities_path and identities_path.is_file():
        for raw in identities_path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            _add_identity(bucket, seen, line, repo_name=repo_name)

    for name in (
        os.environ.get("USER"),
        os.environ.get("LOGNAME"),
        os.environ.get("USERNAME"),
    ):
        _add_identity(bucket, seen, name or "", repo_name=repo_name)
    try:
        import getpass
        _add_identity(bucket, seen, getpass.getuser(), repo_name=repo_name)
    except Exception:
        pass

    def add_host(val: str) -> None:
        host = (val or "").strip().strip("\"'").rstrip(".")
        if not host or "/" in host or "XXXX" in host:
            return
        _add_identity(bucket, seen, host, repo_name=repo_name)
        if "." in host:
            _add_identity(bucket, seen, host.split(".")[0], repo_name=repo_name)

    try:
        add_host(socket.gethostname())
    except Exception:
        pass

    if sys.platform == "darwin":
        for key in ("LocalHostName", "ComputerName", "HostName"):
            try:
                proc = subprocess.run(
                    ["scutil", "--get", key],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=2,
                )
                if proc.returncode == 0:
                    add_host(proc.stdout.strip())
            except Exception:
                pass

    for name in _users_from_path(str(root.resolve())):
        _add_identity(bucket, seen, name, repo_name=repo_name)

    env_file = root / ".env"
    if env_file.is_file():
        env = parse_env_file(env_file)
        add_host(env.get("SPARK_HOST", ""))
        _add_identity(bucket, seen, env.get("SPARK_USER", ""), repo_name=repo_name)
        for key in ("SPARK_PROJECT_DIR", "HF_HOME"):
            for name in _users_from_path(env.get(key, "")):
                _add_identity(bucket, seen, name, repo_name=repo_name)

    return bucket


def parse_env_file(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export "):]
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip()
        if len(val) >= 2 and val[0] == val[-1] and val[0] in {"'", '"'}:
            val = val[1:-1]
        if key:
            out[key] = val
    return out


def classify_targets(rel: str, ignored: bool) -> list[str]:
    name = Path(rel).name
    if name in LOCAL_ONLY_NAMES or any(rel == p or rel.startswith(p) for p in LOCAL_ONLY_PREFIXES):
        return ["local-only"]
    targets: list[str] = []
    if not ignored:
        targets.append("github")
    hf = any(rel == p.rstrip("/") or rel.startswith(p) for p in HF_DIR_PREFIXES)
    if hf and (rel.endswith(HF_SUFFIXES) or rel.endswith("/") or Path(rel).suffix == "" or hf):
        # Any file under artifacts/traces/logs/outputs/data is HF-destined.
        if any(rel.startswith(p) for p in HF_DIR_PREFIXES):
            if "huggingface" not in targets:
                targets.append("huggingface")
    if not targets:
        targets.append("ignored")
    return targets


def severity_for(targets: list[str]) -> str:
    if "github" in targets or "huggingface" in targets:
        return "blocking"
    if "local-only" in targets:
        return "local-only"
    return "ignored"


def is_ignored(root: Path, rel: str, cache: dict[str, bool]) -> bool:
    if rel in cache:
        return cache[rel]
    proc = subprocess.run(
        ["git", "check-ignore", "-q", "--", rel],
        cwd=root,
        check=False,
        capture_output=True,
    )
    # 0 = ignored, 1 = not ignored, 128 = error (treat as not ignored)
    cache[rel] = proc.returncode == 0
    return cache[rel]


def tracked_files(root: Path) -> set[str]:
    proc = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=root,
        check=False,
        capture_output=True,
    )
    if proc.returncode != 0 or not proc.stdout:
        return set()
    return {p.decode("utf-8", "replace") for p in proc.stdout.split(b"\0") if p}


def has_commits(root: Path) -> bool:
    proc = subprocess.run(
        ["git", "rev-parse", "--verify", "HEAD"],
        cwd=root,
        check=False,
        capture_output=True,
    )
    return proc.returncode == 0


def gitignore_check(root: Path) -> list[GitignoreRow]:
    tracked = tracked_files(root)
    rows: list[GitignoreRow] = []
    for p in REQUIRED_IGNORED:
        ignored_proc = subprocess.run(
            ["git", "check-ignore", "-q", "--", p],
            cwd=root,
            check=False,
            capture_output=True,
        )
        ignored = ignored_proc.returncode == 0
        is_tracked = p in tracked or p.rstrip("/") in tracked
        # also treat any tracked file under a required directory as tracked
        if p.endswith("/"):
            is_tracked = is_tracked or any(t.startswith(p) or t.startswith(p.rstrip("/")) for t in tracked)
        if is_tracked:
            ok = False
            detail = "TRACKED — must be untracked before going public"
        elif not ignored:
            ok = False
            detail = "not matched by .gitignore"
        else:
            ok = True
            detail = "gitignored and untracked"
        rows.append(GitignoreRow(p, ignored, is_tracked, ok, detail))
    return rows


def should_scan_file(path: Path) -> bool:
    if path.is_symlink() or not path.is_file():
        return False
    try:
        size = path.stat().st_size
    except OSError:
        return False
    if size == 0 or size > MAX_FILE_BYTES:
        return False
    if path.name in TEXT_NAMES or path.suffix.lower() in TEXT_EXT:
        return True
    # Names like .env.something
    if path.name.startswith(".env"):
        return True
    if path.suffix == "":
        try:
            chunk = path.read_bytes()[:512]
        except OSError:
            return False
        if b"\0" in chunk:
            return False
        return True
    return False


def iter_scan_files(root: Path) -> Iterable[Path]:
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        # Also skip hidden tooling dirs except those we explicitly care about
        # (.env files are in filenames at repo root).
        base = Path(dirpath)
        for name in filenames:
            if name.endswith(".pyc"):
                continue
            # Never re-scan the redacted report or raw gitleaks dumps.
            if name in {"prepublish_report.md"} or name.startswith("gitleaks-") and name.endswith(".json"):
                continue
            yield base / name


def identity_regex(identities: list[str]) -> re.Pattern[str] | None:
    parts = []
    for ident in sorted(set(identities), key=len, reverse=True):
        if len(ident) < 3:
            continue
        parts.append(re.escape(ident))
    if not parts:
        return None
    return re.compile(
        r"(?i)(?<![A-Za-z0-9_.-])(?:" + "|".join(parts) + r")(?![A-Za-z0-9_.-])"
    )


def looks_like_loopback_ip(ip: str) -> bool:
    try:
        obj = ipaddress.ip_address(ip)
    except ValueError:
        return False
    return bool(obj.is_loopback or obj.is_unspecified or obj.is_multicast or ip == "255.255.255.255")


def looks_like_clock_time(text: str) -> bool:
    """Avoid treating 17:00:00 / 07:30:00 as IPv6."""
    return re.fullmatch(r"\d{1,2}:\d{2}(?::\d{2})?(?:\.\d+)?", text) is not None


def env_value_is_empty_or_ref(val: str) -> bool:
    v = val.strip()
    if not v or v in {"''", '""'}:
        return True
    if ENV_VAR_REF_RE.match(v):
        return True
    core = v.strip("'\"")
    if len(core) < 8:
        return True
    low = core.lower()
    if low in {"", "changeme", "your_api_key", "your-api-key", "xxx", "todo", "none", "null"}:
        return True
    if any(tok in v for tok in ("startswith", "split(", "os.environ", "getenv", "line.")):
        return True
    return False


def scan_line(
    rel: str,
    line_no: int,
    line: str,
    identities_re: re.Pattern[str] | None,
    identities: list[str],
    ignored: bool,
) -> list[Finding]:
    found: list[Finding] = []
    targets = classify_targets(rel, ignored)
    severity = severity_for(targets)

    def add(category: str, raw: str, rule: str = "") -> None:
        found.append(
            Finding(
                category=category,
                path=rel,
                line=line_no,
                raw=raw,
                snippet=snippet_around(line.rstrip("\n"), raw, identities),
                severity=severity,
                targets=targets,
                rule=rule or category,
            )
        )

    for m in EMAIL_RE.finditer(line):
        add("email", m.group(0))
    if identities_re:
        for m in identities_re.finditer(line):
            raw = m.group(0)
            # Path findings already cover /Users/<name>/ and /home/<name>/.
            if re.search(r"/(?:Users|home)/" + re.escape(raw) + r"(?:/|$)", line):
                continue
            cat = "hostname" if ("." in raw or ":" in raw) else "username"
            add(cat, raw, rule="identity")
    for m in ABS_USERS_RE.finditer(line):
        user = USERS_PATH_RE.search(m.group(0))
        user_name = user.group(1) if user else ""
        if user_name.lower() in PLACEHOLDER_PATH_USERS:
            continue
        if user_name.upper() in {"YOU", "USER", "IDENTITY"}:
            continue
        if user_name.startswith(("<", "[", "$", "{")):
            continue
        add("path", m.group(0))
    for m in IPV4_RE.finditer(line):
        ip = m.group(0)
        if looks_like_loopback_ip(ip) or ip.startswith("127.") or ip in {"0.0.0.0", "255.255.255.255"}:
            continue
        add("ip", ip)
    for m in IPV6_RE.finditer(line):
        ip = m.group(0)
        if ip in {"::1"} or looks_like_loopback_ip(ip) or looks_like_clock_time(ip):
            continue
        add("ip", ip)
    for cre, cat in ((PHONE_US_RE, "phone"), (PHONE_INTL_RE, "phone")):
        for m in cre.finditer(line):
            add(cat, m.group(0))
    for m in PRIVATE_KEY_RE.finditer(line):
        add("ssh", m.group(0), rule="private-key")
    for m in SSH_PUB_RE.finditer(line):
        add("ssh", m.group(0), rule="ssh-public-key")
    for m in ENV_ASSIGN_RE.finditer(line):
        val = m.group("val")
        if env_value_is_empty_or_ref(val):
            continue
        add("env", m.group(0), rule=m.group(1))
    for m in SECRET_PREFIX_RE.finditer(line):
        add("secret", m.group(0), rule="token-prefix")
    for m in BEARER_RE.finditer(line):
        add("secret", m.group(0), rule="bearer")
    for m in GEN_ID_RE.finditer(line):
        add("secret", m.group(0), rule="openrouter-gen-id")
    return found


def scan_tree(
    root: Path,
    identities: list[str],
    ignore_cache: dict[str, bool],
) -> list[Finding]:
    ident_re = identity_regex(identities)
    findings: list[Finding] = []
    for path in iter_scan_files(root):
        if not should_scan_file(path):
            continue
        rel = relpath(root, path)
        ignored = is_ignored(root, rel, ignore_cache)
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for i, line in enumerate(text.splitlines(), start=1):
            findings.extend(scan_line(rel, i, line, ident_re, identities, ignored))
    return findings


def load_gitleaks_json(path: Path | None, identities: list[str], root: Path, ignore_cache: dict[str, bool], source: str) -> list[Finding]:
    if path is None or not path.is_file():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8", errors="replace") or "[]")
    except json.JSONDecodeError:
        return []
    if not isinstance(data, list):
        return []
    out: list[Finding] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        raw_file = str(item.get("File") or item.get("file") or "")
        rel = raw_file
        try:
            rel = relpath(root, Path(raw_file)) if raw_file else ""
        except Exception:
            rel = raw_file
        line_no = int(item.get("StartLine") or item.get("startLine") or 0)
        raw = str(item.get("Secret") or item.get("secret") or item.get("Match") or item.get("match") or "")
        rule = str(item.get("RuleID") or item.get("RuleId") or item.get("ruleID") or "gitleaks")
        desc = str(item.get("Description") or "")
        commit = item.get("Commit") or item.get("commit") or None
        if commit:
            commit = str(commit)
        ignored = is_ignored(root, rel, ignore_cache) if rel else False
        targets = classify_targets(rel, ignored)
        # History hits are always GitHub-relevant even if the current file is ignored.
        if source == "history" and "github" not in targets:
            targets = ["github"] + [t for t in targets if t != "ignored"]
        severity = "blocking" if source == "history" else severity_for(targets)
        snippet_src = str(item.get("Match") or item.get("Line") or raw)
        out.append(
            Finding(
                category="secret",
                path=rel or "(unknown)",
                line=line_no,
                raw=raw or snippet_src,
                snippet=redact_text(snippet_src.strip(), identities) + (f" — {desc}" if desc else ""),
                severity=severity,
                targets=targets,
                rule=rule,
                commit=commit,
            )
        )
    return out


def dedupe(findings: list[Finding]) -> list[Finding]:
    seen: set[tuple] = set()
    out: list[Finding] = []
    for f in findings:
        key = (f.category, f.path, f.line, f.rule, f.commit or "", f.raw)
        if key in seen:
            continue
        seen.add(key)
        out.append(f)
    return out


def md_escape(text: str) -> str:
    return text.replace("|", "\\|")


def write_report(
    report_path: Path,
    root: Path,
    findings: list[Finding],
    gi_rows: list[GitignoreRow],
    gitleaks_version: str,
    git_history_note: str,
    git_history_ok: bool,
    identities_count: int,
    allowlist_path: Path,
) -> int:
    now = datetime.now(ET).strftime("%Y-%m-%d %I:%M %p ET")
    blocking = [f for f in findings if f.severity == "blocking" and not f.allowlisted]
    local = [f for f in findings if f.severity == "local-only" and not f.allowlisted]
    ignored = [f for f in findings if f.severity == "ignored" and not f.allowlisted]
    allowed = [f for f in findings if f.allowlisted]
    gi_fail = [r for r in gi_rows if not r.ok]
    history = [f for f in blocking if f.commit]
    clean = not blocking and not gi_fail and git_history_ok
    status = "PASS" if clean else "FAIL"

    by_cat: dict[str, list[Finding]] = {}
    for f in findings:
        if f.allowlisted:
            continue
        by_cat.setdefault(f.category, []).append(f)

    lines: list[str] = []
    a = lines.append
    a("# Pre-publication scan report")
    a("")
    a(f"- **Result:** {status}")
    a(f"- **Generated:** {now}")
    a(f"- **Repo:** `{root}`".replace(str(root), repo_display(root)))
    a(f"- **gitleaks:** {gitleaks_version or 'not found'}")
    a(f"- **Allowlist:** `{allowlist_path.name}`")
    a(f"- **Identity patterns loaded:** {identities_count} (from the gitignored identities file, this machine, and non-secret `.env` keys). Values are not listed here.")
    a("")
    a("## Summary")
    a("")
    a(f"- Blocking findings (would be published to GitHub and/or Hugging Face): **{len(blocking)}**")
    a(f"- Secrets or identifiers in git history: **{len(history)}**")
    a(f"- Local-only findings (gitignored files such as `.env`; do not upload these): **{len(local)}**")
    a(f"- Allowlisted (reviewed false positives): **{len(allowed)}**")
    a(f"- Gitignore / tracking checks failed: **{len(gi_fail)}**")
    a(f"- Git history scan: {git_history_note}")
    a("")

    if clean:
        a("No publication-blocking secrets or PII were found. Re-run this scan after adding paper, slides, traces, or datasets, and once more immediately before going public.")
        a("")
    else:
        a("This report is redacted. Do not paste unredacted logs or `.env` contents into tickets, chats, or the public repo.")
        a("")

    a("## Gitignore and tracking")
    a("")
    a("These paths must stay gitignored and untracked so they are not pushed to GitHub.")
    a("")
    a("| Path | Status | Detail |")
    a("|---|---|---|")
    for row in gi_rows:
        flag = "ok" if row.ok else "FAIL"
        a(f"| `{row.path}` | {flag} | {md_escape(row.detail)} |")
    a("")

    a("## Git history (gitleaks)")
    a("")
    a(git_history_note)
    a("")
    if history:
        a("Secrets in git history are **not** fixed by deleting the working-tree file. The history has to be rewritten **after** every exposed credential is rotated, and only with an explicit go-ahead. Recommended approach (do not run until asked):")
        a("")
        a("1. Revoke and rotate every token or key that appeared in a commit (Hugging Face, OpenRouter, and any other hit).")
        a("2. Use [`git filter-repo`](https://github.com/newren/git-filter-repo) (preferred) or BFG to remove the secret from every commit. Do not use `git filter-branch`.")
        a("3. Force-push only if the remote has never been public; otherwise treat the old history as compromised and start a new root (or a fresh repository) after rotation.")
        a("4. Re-run `scripts/prepublish_scan.sh` and confirm this section is empty.")
        a("")
        _write_finding_table(a, history)
        a("")

    a("## Working-tree secrets (gitleaks dir / --no-git)")
    a("")
    secrets = [f for f in by_cat.get("secret", []) if not f.commit]
    if not secrets:
        a("No gitleaks hits in the working tree, other than any local-only files listed below.")
        a("")
    else:
        _write_finding_table(a, secrets)
        a("")

    a("## PII and private information")
    a("")
    pii_cats = ["email", "username", "hostname", "path", "ip", "phone", "ssh", "env"]
    any_pii = False
    for cat in pii_cats:
        rows = by_cat.get(cat, [])
        # username findings may include hostnames detected as identities; split host-like
        if not rows:
            continue
        any_pii = True
        a(f"### {cat}")
        a("")
        _write_finding_table(a, rows)
        a("")
    if not any_pii:
        a("No PII or private-info matches outside the allowlist.")
        a("")

    if local:
        a("## Local-only files (not for GitHub or Hugging Face)")
        a("")
        a("These hits are in gitignored local files (for example `.env`). They do not fail the scan by themselves, but they must not be uploaded to GitHub or Hugging Face.")
        a("")
        _write_finding_table(a, local)
        a("")

    if allowed:
        a("## Allowlisted findings")
        a("")
        a("| File | Line | Category | Snippet (redacted) | Reason |")
        a("|---|---:|---|---|---|")
        for f in allowed:
            a(
                f"| `{md_escape(f.path)}` | {f.line} | {f.category} | `{md_escape(f.snippet)}` | {md_escape(f.allow_reason)} |"
            )
        a("")

    a("## Suggested fixes")
    a("")
    suggestions = suggest_fixes(blocking, gi_fail, local, history)
    if suggestions:
        for s in suggestions:
            a(f"- {s}")
    else:
        a("- None right now. Re-run the scan after new artifacts, paper sources, or slides land, and once more before going public.")
    a("")
    a("## How to re-run")
    a("")
    a("```bash")
    a("./scripts/prepublish_scan.sh")
    a("```")
    a("")
    a("The script writes this redacted report to `artifacts/prepublish_report.md` and exits non-zero if anything publication-blocking was found. To record a reviewed false positive, add a tab-separated line to `.prepublish_allowlist`.")
    a("")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")
    try:
        os.chmod(report_path, stat.S_IRUSR | stat.S_IWUSR)
    except OSError:
        pass

    if not git_history_ok:
        return 2
    if blocking or gi_fail:
        return 1
    return 0


def repo_display(root: Path) -> str:
    # Redact the local user segment in the path printed in the report.
    text = str(root)
    return re.sub(r"/(Users|home)/[^/]+", r"/\1/[USER]", text)


def _write_finding_table(a, findings: list[Finding]) -> None:
    a("| File | Line | Rule | Targets | Snippet (redacted) |")
    a("|---|---:|---|---|---|")
    for f in findings:
        targets = ", ".join(f.targets)
        commit = f" commit `{f.commit[:12]}`" if f.commit else ""
        a(
            f"| `{md_escape(f.path)}` | {f.line} | `{md_escape(f.rule)}`{commit} | {targets} | `{md_escape(f.snippet)}` |"
        )


def suggest_fixes(
    blocking: list[Finding],
    gi_fail: list[GitignoreRow],
    local: list[Finding],
    history: list[Finding],
) -> list[str]:
    tips: list[str] = []
    if gi_fail:
        tips.append(
            "Update `.gitignore` for any failed path below and `git rm --cached` it if it is already tracked. Do not rewrite history unless a past commit also contained it."
        )
    if history:
        tips.append(
            "Rotate every credential found in git history, then rewrite history with `git filter-repo` (only after an explicit go-ahead). Deleting the file in a new commit is not enough."
        )
    cats = {f.category for f in blocking}
    if "path" in cats:
        tips.append(
            "Replace absolute local paths such as `/Users/[USER]/...` or `/home/[USER]/...` in docs and notebooks with placeholders (`$LEXSI_ROOT`, `/path/to/lexsi-unlearning`, or `/home/YOU/...`)."
        )
    if "username" in cats or "hostname" in cats:
        tips.append(
            "Remove or generalize personal usernames and machine hostnames in files that will be published. Keep real host values only in the gitignored `.env` or `.prepublish_identities`."
        )
    if "email" in cats:
        tips.append(
            "Remove personal email addresses from tracked files, or replace them with a public contact you are willing to publish."
        )
    if "secret" in cats or "env" in cats:
        tips.append(
            "Remove live tokens from any file that is not `.env`. Rotate the token if it may have been copied into traces, logs, notebooks, or a commit."
        )
    if "ssh" in cats:
        tips.append(
            "Remove SSH private or personal public keys from the repo and rotate the key if it may have been committed or uploaded."
        )
    if "ip" in cats:
        tips.append(
            "Replace non-loopback IP addresses with hostnames or placeholders before publishing."
        )
    if "phone" in cats:
        tips.append("Remove phone numbers from files that will be published.")
    if local:
        tips.append(
            "Keep `.env` and identity files gitignored. Do not include them in the Hugging Face collections or in `git add -f`."
        )
    # de-dupe while preserving order
    seen: set[str] = set()
    out: list[str] = []
    for t in tips:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def apply_allowlist(findings: list[Finding], rules: list[AllowRule]) -> None:
    for f in findings:
        ok, reason = is_allowlisted(f, rules)
        if ok:
            f.allowlisted = True
            f.allow_reason = reason


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Write a redacted prepublish scan report.")
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--report", type=Path, required=True)
    p.add_argument("--allowlist", type=Path, required=True)
    p.add_argument("--identities", type=Path, default=None)
    p.add_argument("--gitleaks-git", type=Path, default=None)
    p.add_argument("--gitleaks-dir", type=Path, default=None)
    p.add_argument("--gitleaks-version", default="")
    p.add_argument("--git-history-note", default="")
    p.add_argument("--git-history-ok", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    root = args.root.resolve()
    identities = load_identities(root, args.identities)
    allow_rules = load_allowlist(args.allowlist)
    ignore_cache: dict[str, bool] = {}

    gi_rows = gitignore_check(root)
    pii = scan_tree(root, identities, ignore_cache)
    git_findings = load_gitleaks_json(args.gitleaks_git, identities, root, ignore_cache, "history")
    dir_findings = load_gitleaks_json(args.gitleaks_dir, identities, root, ignore_cache, "workdir")

    findings = dedupe(git_findings + dir_findings + pii)
    apply_allowlist(findings, allow_rules)

    history_ok = bool(args.git_history_ok)
    note = args.git_history_note or ("Git history scan completed." if history_ok else "Git history scan did not run.")

    code = write_report(
        report_path=args.report if args.report.is_absolute() else root / args.report,
        root=root,
        findings=findings,
        gi_rows=gi_rows,
        gitleaks_version=args.gitleaks_version,
        git_history_note=note,
        git_history_ok=history_ok,
        identities_count=len(identities),
        allowlist_path=args.allowlist,
    )
    # A short redacted status line only — never dump findings to stdout.
    status = "PASS" if code == 0 else ("ERROR" if code == 2 else "FAIL")
    report = args.report if args.report.is_absolute() else root / args.report
    try:
        report_display = report.resolve().relative_to(root).as_posix()
    except ValueError:
        report_display = re.sub(r"/(Users|home)/[^/]+", r"/\1/[USER]", str(report))
    print(f"prepublish_scan: {status} (exit {code}). Redacted report: {report_display}")
    return code


if __name__ == "__main__":
    sys.exit(main())
