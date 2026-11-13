#!/usr/bin/env python3
"""Copy run outputs to a publication tree with paths rewritten and secrets/PII redacted.

Never edits the source tree in place. Writes a dry-run report by default when
`--dry-run` is set (no output files). After redaction, fails if any token-like
string (`hf_`, `sk-or-`, `sk-`, `ghp_`, Bearer, `gen-` ids) remains.

    python3 scripts/redact_release.py --dry-run --report artifacts/redact_report.md
    python3 scripts/redact_release.py --hf-out release/hf_dataset --pages-out release/pages_src

`real_authors` gold targets (public literary figures in TOFU) are listed in the
report and kept. Hostnames and usernames come from `.prepublish_identities`,
this machine, and non-secret `.env` keys (same loader as prepublish_scan).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import prepublish_scan as P  # noqa: E402


BEARER_RE = re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._\-+=/]{8,}")
GEN_ID_RE = re.compile(r"\bgen-[A-Za-z0-9]{8,}\b")
# Secret prefixes already defined in prepublish_scan.SECRET_PREFIX_RE
WORKSPACE_RE = re.compile(r"(?<![A-Za-z0-9_])\/workspace(?=/|\"|'|\s|$)")

TEXT_SUFFIXES = {
    ".json", ".jsonl", ".md", ".txt", ".log", ".csv", ".tsv", ".yml", ".yaml",
    ".tex", ".bib", ".html", ".css", ".js", ".py", ".sh", ".toml", ".example",
    ".lst",
}
TEXT_NAMES = {"README", "Makefile", "Dockerfile", "LICENSE", "NOTICE"}

# Publish these relative paths from the artifacts / logs / data inputs.
ARTIFACT_SUBDIRS = ("traces", "results", "interp", "circuitkit")
ARTIFACT_FILES = ("translations_review.md",)
SKIP_SOURCE_NAMES = {".env", ".prepublish_identities"}
WEIGHT_SUFFIXES = {".safetensors", ".bin", ".pt", ".pth", ".ckpt", ".gguf"}
WEIGHT_DIR_NAMES = {"rmu_unlearned", "inject_lora", "npo_unlearned"}

PLACEHOLDER_ROOT = "$LEXSI_ROOT"


@dataclass
class Stats:
    files_in: int = 0
    files_out: int = 0
    bytes_in: int = 0
    bytes_out: int = 0
    skipped_weights: int = 0
    skipped_other: int = 0
    replacements: Counter = field(default_factory=Counter)
    real_authors: Counter = field(default_factory=Counter)
    leftovers: list[tuple[str, str, str]] = field(default_factory=list)
    written: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def bearer_sub(_m: re.Match[str]) -> str:
    return "Bearer [REDACTED]"


def gen_sub(_m: re.Match[str]) -> str:
    return "gen-[REDACTED]"


def rewrite_home_path(match: re.Match[str]) -> str:
    """Turn /Users/<name>/.../lexsi-unlearning/<rest> or /home/<name>/... into $LEXSI_ROOT/<rest>."""
    body = match.group(0)
    # Prefer the repo folder as the cut point when present.
    for marker in ("/lexsi-unlearning/", "/lexsi-unlearning"):
        idx = body.find(marker)
        if idx >= 0:
            rest = body[idx + len("/lexsi-unlearning"):]
            if not rest or rest.startswith("/"):
                return PLACEHOLDER_ROOT + (rest if rest else "")
    # /home/<user> or /Users/<user> plus optional remainder: drop the identity segment.
    return re.sub(r"^/(?:Users|home)/[^/]+", PLACEHOLDER_ROOT, body)


def redact_string(text: str, identities: list[str], stats: Stats | None = None) -> str:
    """Apply token, path, hostname, email, and IP redaction to a text blob."""
    counts: Counter = Counter()

    def counted(kind: str, new: str, n: int) -> str:
        if n and stats is not None:
            stats.replacements[kind] += n
        counts[kind] += n
        return new

    def sub_count(pattern: re.Pattern[str], repl, kind: str, src: str) -> str:
        n = 0

        def _r(m: re.Match[str]) -> str:
            nonlocal n
            n += 1
            return repl(m) if callable(repl) else repl

        out = pattern.sub(_r, src)
        counted(kind, out, n)
        return out

    # Bearer first so a key inside an Authorization header is fully redacted
    # rather than leaving "Bearer sk-or-[REDACTED]".
    out = sub_count(BEARER_RE, bearer_sub, "bearer", text)
    out = sub_count(P.SECRET_PREFIX_RE, lambda m: _token_repl(m.group(0)), "token", out)
    out = sub_count(GEN_ID_RE, gen_sub, "gen_id", out)
    out = sub_count(P.ABS_USERS_RE, rewrite_home_path, "path", out)
    out = sub_count(WORKSPACE_RE, PLACEHOLDER_ROOT, "workspace_path", out)

    # Identities (usernames, hostnames) longest first, after paths so we don't
    # double-count a username already removed from /home/<user>/...
    for ident in sorted(set(identities), key=len, reverse=True):
        if len(ident) < 3:
            continue
        pattern = re.compile(
            r"(?i)(?<![A-Za-z0-9_.-])" + re.escape(ident) + r"(?![A-Za-z0-9_.-])"
        )
        out = sub_count(pattern, "[IDENTITY]", "identity", out)

    def email_repl(_m: re.Match[str]) -> str:
        return "[EMAIL]"

    out = sub_count(P.EMAIL_RE, email_repl, "email", out)

    def ip_repl(m: re.Match[str]) -> str:
        ip = m.group(0)
        if P.looks_like_loopback_ip(ip) or ip.startswith("127.") or ip in {
            "0.0.0.0", "255.255.255.255", "::1",
        }:
            return ip
        if P.looks_like_clock_time(ip):
            return ip
        parts = ip.split(".")
        if len(parts) == 4:
            return f"{parts[0]}.{parts[1]}.[REDACTED]"
        return "[IP]"

    # Count only non-loopback replacements.
    def ip_sub(pattern: re.Pattern[str], src: str, kind: str) -> str:
        n = 0

        def _r(m: re.Match[str]) -> str:
            nonlocal n
            repl = ip_repl(m)
            if repl != m.group(0):
                n += 1
            return repl

        out2 = pattern.sub(_r, src)
        counted(kind, out2, n)
        return out2

    out = ip_sub(P.IPV4_RE, out, "ip")
    out = ip_sub(P.IPV6_RE, out, "ip")
    return out


def _token_repl(s: str) -> str:
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


def leftover_tokens(text: str) -> list[tuple[str, str]]:
    """Return (rule, match) for any remaining token-like strings."""
    hits: list[tuple[str, str]] = []
    for m in P.SECRET_PREFIX_RE.finditer(text):
        hits.append(("token-prefix", m.group(0)))
    for m in BEARER_RE.finditer(text):
        hits.append(("bearer", m.group(0)))
    for m in GEN_ID_RE.finditer(text):
        hits.append(("openrouter-gen-id", m.group(0)))
    return hits


def collect_real_authors(obj: object, stats: Stats) -> None:
    if isinstance(obj, dict):
        if obj.get("split") == "real_authors":
            tgt = obj.get("target")
            if isinstance(tgt, str) and tgt.strip():
                stats.real_authors[tgt.strip()] += 1
        for v in obj.values():
            collect_real_authors(v, stats)
    elif isinstance(obj, list):
        for v in obj:
            collect_real_authors(v, stats)


def should_skip(path: Path, rel: str) -> str | None:
    if path.name in SKIP_SOURCE_NAMES or path.name.startswith(".env"):
        return "secret-file"
    if path.suffix.lower() in WEIGHT_SUFFIXES:
        return "weight"
    parts = Path(rel).parts
    if any(p.startswith("checkpoint-") for p in parts):
        return "checkpoint"
    if any(p in WEIGHT_DIR_NAMES for p in parts):
        return "weight-dir"
    return None


def is_text_file(path: Path) -> bool:
    if path.name in TEXT_NAMES or path.suffix.lower() in TEXT_SUFFIXES:
        return True
    if path.suffix == "":
        try:
            chunk = path.read_bytes()[:512]
        except OSError:
            return False
        return b"\0" not in chunk
    return False


def redact_file_text(text: str, path: Path, identities: list[str], stats: Stats) -> str:
    out = redact_string(text, identities, stats)
    if path.suffix.lower() in {".json", ".jsonl"}:
        # Validate and harvest real_authors names from structured traces.
        try:
            if path.suffix.lower() == ".jsonl":
                for line in out.splitlines():
                    if not line.strip():
                        continue
                    collect_real_authors(json.loads(line), stats)
            else:
                collect_real_authors(json.loads(out), stats)
        except json.JSONDecodeError as exc:
            stats.errors.append(f"{path}: redacted text is not valid JSON ({exc})")
    for rule, raw in leftover_tokens(out):
        stats.leftovers.append((str(path), rule, raw[:40]))
    return out


def iter_artifact_sources(artifacts: Path) -> list[tuple[Path, str]]:
    found: list[tuple[Path, str]] = []
    if not artifacts.is_dir():
        return found
    for sub in ARTIFACT_SUBDIRS:
        d = artifacts / sub
        if not d.is_dir():
            continue
        for p in d.rglob("*"):
            if p.is_file():
                found.append((p, f"{sub}/" + p.relative_to(d).as_posix()))
    for name in ARTIFACT_FILES:
        p = artifacts / name
        if p.is_file():
            found.append((p, name))
    return found


def copy_tree(
    sources: list[tuple[Path, str]],
    dest: Path,
    identities: list[str],
    stats: Stats,
    *,
    dry_run: bool,
) -> None:
    for src, rel in sources:
        skip = should_skip(src, rel)
        if skip:
            if skip in {"weight", "weight-dir", "checkpoint"}:
                stats.skipped_weights += 1
            else:
                stats.skipped_other += 1
            continue
        stats.files_in += 1
        try:
            size = src.stat().st_size
        except OSError as exc:
            stats.errors.append(f"{src}: {exc}")
            continue
        stats.bytes_in += size
        if not is_text_file(src):
            stats.skipped_other += 1
            stats.errors.append(f"{rel}: skipped non-text file ({src.suffix or 'no suffix'})")
            continue
        try:
            text = src.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            stats.errors.append(f"{src}: {exc}")
            continue
        out = redact_file_text(text, Path(rel), identities, stats)
        encoded = out.encode("utf-8")
        stats.files_out += 1
        stats.bytes_out += len(encoded)
        stats.written.append(rel)
        if not dry_run:
            target = dest / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(out, encoding="utf-8")


def write_hf_readme(dest: Path, *, dry_run: bool) -> None:
    text = (
        "# Lexsi Track 2b run records\n\n"
        "Scrubbed traces, logs, results, interpretability artifacts, and "
        "translations for the Lexsi unlearning vs. hiding audit. Absolute "
        "local paths have been rewritten to `$LEXSI_ROOT`. Secrets, emails, "
        "non-loopback IPs, and host identifiers have been redacted.\n\n"
        "Model weights are not in this dataset; they are published as a "
        "separate Hugging Face model collection.\n"
    )
    if not dry_run:
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "README.md").write_text(text, encoding="utf-8")


def write_report(path: Path, stats: Stats, *, dry_run: bool, dests: list[str], identities_n: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    a = lines.append
    a("# Redaction report")
    a("")
    a(f"- **Mode:** {'dry-run (no files written)' if dry_run else 'write'}")
    a(f"- **Identity patterns loaded:** {identities_n} (values are not listed here).")
    a(f"- **Destinations:** {', '.join(dests) if dests else '(none)'}")
    a("")
    a("## Counts")
    a("")
    a(f"- Files read: {stats.files_in}")
    a(f"- Files written (or would write): {stats.files_out}")
    a(f"- Bytes in / out: {stats.bytes_in} / {stats.bytes_out}")
    a(f"- Weight / checkpoint files skipped: {stats.skipped_weights}")
    a(f"- Other skips: {stats.skipped_other}")
    a("")
    a("## Replacements")
    a("")
    if stats.replacements:
        a("| Kind | Count |")
        a("|---|---:|")
        for kind, n in sorted(stats.replacements.items()):
            a(f"| {kind} | {n} |")
    else:
        a("No replacements were needed.")
    a("")
    a("## real_authors names (kept)")
    a("")
    a("These are gold targets from TOFU's public `real_authors` split (literary figures).")
    a("They were flagged and **not** redacted.")
    a("")
    if stats.real_authors:
        a("| Name | Occurrences |")
        a("|---|---:|")
        for name, n in stats.real_authors.most_common():
            a(f"| {name} | {n} |")
    else:
        a("No `real_authors` targets were found in the copied traces.")
    a("")
    a("## Leftover token-like strings")
    a("")
    if stats.leftovers:
        a("These would block a release. The redactor fails closed when any are present.")
        a("")
        a("| File | Rule | Match (truncated, not a full secret) |")
        a("|---|---|---|")
        for rel, rule, raw in stats.leftovers:
            shown = raw if len(raw) <= 12 else raw[:4] + "…"
            a(f"| `{rel}` | `{rule}` | `{shown}` |")
    else:
        a("None.")
    a("")
    if stats.errors:
        a("## Errors")
        a("")
        for e in stats.errors:
            a(f"- {e}")
        a("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    try:
        path.chmod(0o600)
    except OSError:
        pass


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, default=ROOT, help="Repo root (for identities and default paths)")
    p.add_argument("--artifacts", type=Path, default=None, help="Source artifacts/ directory")
    p.add_argument("--logs", type=Path, default=None, help="Source logs/ directory")
    p.add_argument("--translations", type=Path, default=None, help="Source data/translations.json")
    p.add_argument("--hf-out", type=Path, default=None, help="Full scrubbed tree for the HF dataset")
    p.add_argument("--pages-out", type=Path, default=None, help="Scrubbed tree used to build the Pages site")
    p.add_argument("--report", type=Path, default=None, help="Markdown report path")
    p.add_argument("--identities", type=Path, default=None)
    p.add_argument("--allowlist", type=Path, default=None, help="Reserved; replacements always run, leftovers still fail")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args(argv)


def resolve_defaults(args: argparse.Namespace) -> argparse.Namespace:
    root: Path = args.root
    if args.artifacts is None:
        args.artifacts = root / "artifacts"
    if args.logs is None:
        args.logs = root / "logs"
    if args.translations is None:
        cand = root / "data" / "translations.json"
        args.translations = cand if cand.is_file() else None
    if args.hf_out is None:
        args.hf_out = root / "release" / "hf_dataset"
    if args.pages_out is None:
        args.pages_out = root / "release" / "pages_src"
    if args.report is None:
        args.report = root / "artifacts" / "redact_report.md"
    if args.identities is None:
        cand = root / ".prepublish_identities"
        args.identities = cand if cand.is_file() else None
    if args.allowlist is None:
        cand = root / ".prepublish_allowlist"
        args.allowlist = cand if cand.is_file() else None
    return args


def main(argv: list[str] | None = None) -> int:
    args = resolve_defaults(parse_args(argv))
    root: Path = args.root
    identities = P.load_identities(root, args.identities)
    stats = Stats()
    sources: list[tuple[Path, str]] = []
    if args.artifacts and Path(args.artifacts).is_dir():
        sources.extend((p, rel) for p, rel in iter_artifact_sources(Path(args.artifacts)))
    if args.logs and Path(args.logs).is_dir():
        for p in Path(args.logs).rglob("*"):
            if p.is_file():
                sources.append((p, "logs/" + p.relative_to(args.logs).as_posix()))
    if args.translations and Path(args.translations).is_file():
        sources.append((Path(args.translations), "translations.json"))

    dests: list[Path] = []
    if args.hf_out:
        dests.append(Path(args.hf_out))
    if args.pages_out and Path(args.pages_out).resolve() != Path(args.hf_out).resolve():
        dests.append(Path(args.pages_out))
    elif args.pages_out and args.hf_out and Path(args.pages_out).resolve() == Path(args.hf_out).resolve():
        dests = [Path(args.hf_out)]

    if not args.dry_run:
        if not dests:
            print("redact_release: no destinations given", file=sys.stderr)
            return 2
        # Never write inside the source artifacts/logs/data tree.
        src_roots = [Path(args.artifacts).resolve() if args.artifacts else None,
                     Path(args.logs).resolve() if args.logs else None]
        for d in dests:
            d = d.resolve()
            for s in src_roots:
                if s and (d == s or s in d.parents or d in s.parents or d == s):
                    print(f"redact_release: refusing to write inside the source tree ({d})", file=sys.stderr)
                    return 2

    # Redact once into the first dest (or a throwaway when dry-run), then copy
    # the same text to any additional dest so counts are not doubled.
    primary = dests[0] if dests else Path("/dev/null")
    copy_tree(sources, primary, identities, stats, dry_run=args.dry_run)
    if not args.dry_run:
        write_hf_readme(Path(args.hf_out), dry_run=False)
        if len(dests) > 1:
            # Re-write the already-redacted files to the second tree.
            for rel in stats.written:
                src_f = dests[0] / rel
                if src_f.is_file():
                    target = dests[1] / rel
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_text(src_f.read_text(encoding="utf-8"), encoding="utf-8")
            if (dests[0] / "README.md").is_file() and dests[1] == Path(args.pages_out).resolve():
                pass  # Pages src does not need the HF README
            elif (dests[0] / "README.md").is_file():
                (dests[1] / "README.md").write_text((dests[0] / "README.md").read_text(encoding="utf-8"), encoding="utf-8")
    else:
        write_hf_readme(Path(args.hf_out), dry_run=True)

    def _rel(path: Path) -> str:
        path = Path(path)
        try:
            return path.resolve().relative_to(root.resolve()).as_posix()
        except ValueError:
            # Do not put absolute local paths in the report.
            return path.name

    dest_labels = [_rel(d) for d in (args.hf_out, args.pages_out) if d]
    write_report(Path(args.report), stats, dry_run=args.dry_run, dests=dest_labels, identities_n=len(identities))

    leftover_n = len(stats.leftovers)
    status = "FAIL" if leftover_n or stats.errors else "OK"
    print(
        f"redact_release: {status} ({'dry-run' if args.dry_run else 'write'}). "
        f"{stats.files_out} files, {sum(stats.replacements.values())} replacements, "
        f"{len(stats.real_authors)} real_authors names kept, {leftover_n} leftover tokens. "
        f"Report: {args.report}"
    )
    if leftover_n or stats.errors:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
