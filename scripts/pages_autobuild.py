#!/usr/bin/env python3
"""Auto-build GitHub Pages after each finished Lexsi Spark run, and commit locally.

Never pushes. Refuses any `git push`. See also:

  ~/Library/Application Support/lexsi-pages/README.txt

Launchd polls ~10 minutes (com.mackenzie.lexsi-pages-autobuild) via the
existing 127.0.0.1:8790 monitor tunnel, then on the Mac:

  1. sync --results (read-only on the Spark)
  2. a real pages build (redact, ingest, export) — not a dry run
  3. redaction report + prepublish_scan
  4. if both are clean: git add -f docs/ and a local commit
     ``pages: update after <run_id>``

Invalid or archived run_ids are listed in the state file's ``excluded``
map and are never ingested or committed.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import time
import traceback
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")

LIB = Path.home() / "Library" / "Application Support" / "lexsi-pages"
DEFAULT_REPO = Path.home() / "Documents" / "code" / "lexsi-unlearning"
EVALBOARD_APP_SUPPORT = Path.home() / "Library" / "Application Support" / "evalboard"
FRAMEWORK_PYTHON = Path(
    "/Library/Frameworks/Python.framework/Versions/3.14/bin/python3"
)
GIT_BIN = Path("/usr/bin/git")
MONITOR_URL = "http://127.0.0.1:8790/api/status"
LOCK_TIMEOUT_S = 60
POLL_CURL_TIMEOUT_S = 8

FINISHED_STATUSES = frozenset(
    {
        "done",
        "failed",
        "error",
        "stopped",
        "deadline_stopped",
        "aborted",
    }
)

# Seeded exclusion: no-op RMU main run (results archived on the Spark).
NOOP_RMU_RUN_ID = (
    "20261113T000819Z-inject+verify+unlearn+eval+probes+interp+relearn--631cb361"
)
DEFAULT_EXCLUSIONS: dict[str, dict[str, str]] = {
    NOOP_RMU_RUN_ID: {
        "reason": (
            "rmu-noop (5 steps); post-RMU results archived to "
            "<HOME>/lexsi-archive/rmu-noop-20261112/. "
            "Do not build or commit Pages from this run."
        )
    }
}

SKIP_DOC_NAMES = {".DS_Store", "Thumbs.db"}


class PushRefused(SystemExit):
    """Hard refusal: this tool must never push."""

    def __init__(self, detail: str = "pages autobuild refuses git push") -> None:
        super().__init__(detail)


class LockBusy(Exception):
    pass


class AutobuildError(Exception):
    pass


def now_et() -> datetime:
    return datetime.now(ET)


def iso_et(ts: datetime | None = None) -> str:
    return (ts or now_et()).isoformat()


def log(msg: str, log_path: Path | None = None) -> None:
    line = f"{now_et().strftime('%Y-%m-%d %H:%M:%S %Z')}  {msg}"
    print(line, flush=True)
    path = log_path or (LIB / "autobuild.log")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def is_push_token(token: str) -> bool:
    base = Path(str(token)).name
    return base == "push" or token == "push"


def refuse_push_in(argv: list[str]) -> None:
    # Allow our own self-test flag; still refuse if a real push was requested.
    for a in argv:
        if a in {"--self-test", "--once", "--install-seed", "--help", "-h"}:
            continue
        if is_push_token(a) or a in {"--push", "--force-push"} or "git-push" in a:
            raise PushRefused()


def git_argv(*args: str) -> list[str]:
    """Build a git argv, aborting if it would push."""
    if any(is_push_token(a) for a in args):
        raise PushRefused()
    return [str(GIT_BIN), *args]


# ---------------------------------------------------------------------------
# State, exclusions, monitor
# ---------------------------------------------------------------------------


def load_json(path: Path, default: Any) -> Any:
    if not path.is_file():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return default


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def merge_exclusions(state: dict[str, Any]) -> dict[str, Any]:
    excluded = dict(state.get("excluded") or {})
    for rid, meta in DEFAULT_EXCLUSIONS.items():
        excluded.setdefault(rid, dict(meta))
        excluded[rid].setdefault("reason", meta["reason"])
    extra = LIB / "exclusions.json"
    if extra.is_file():
        payload = load_json(extra, {})
        items = payload.get("excluded", payload) if isinstance(payload, dict) else {}
        if isinstance(items, dict):
            for rid, meta in items.items():
                if isinstance(meta, dict):
                    excluded.setdefault(rid, meta)
                else:
                    excluded.setdefault(rid, {"reason": str(meta)})
        elif isinstance(items, list):
            for rid in items:
                excluded.setdefault(str(rid), {"reason": "listed in exclusions.json"})
    state["excluded"] = excluded
    state.setdefault("processed", {})
    return state


def is_excluded(run_id: str, state: dict[str, Any]) -> bool:
    excluded = state.get("excluded") or {}
    if run_id in excluded:
        return True
    # Unique suffix used by the pipeline (e.g. --631cb361).
    for key in excluded:
        if run_id == key:
            return True
        if key.endswith(run_id) or run_id.endswith(key):
            return True
        short = key.rsplit("--", 1)[-1]
        if short and (run_id.endswith("--" + short) or run_id == short):
            return True
    return False


def iter_finished_runs(status: dict[str, Any]) -> Iterator[dict[str, Any]]:
    for run in status.get("runs") or []:
        if not isinstance(run, dict):
            continue
        rid = run.get("run_id")
        st = str(run.get("status") or "").lower()
        if rid and st in FINISHED_STATUSES:
            yield run


def new_runs_to_build(
    status: dict[str, Any], state: dict[str, Any]
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for run in iter_finished_runs(status):
        rid = str(run["run_id"])
        if is_excluded(rid, state):
            continue
        if rid in (state.get("processed") or {}):
            continue
        out.append(run)
    # Latest finished run first for the commit subject.
    def key(r: dict[str, Any]) -> str:
        return str(r.get("ended_at") or r.get("run_id") or "")

    return sorted(out, key=key)


def seed_held_at_install(status: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    """Record already-finished runs so install does not publish current traces."""
    state = merge_exclusions(state)
    processed = state.setdefault("processed", {})
    for run in iter_finished_runs(status):
        rid = str(run["run_id"])
        if is_excluded(rid, state):
            state.setdefault("excluded", {}).setdefault(
                rid, {"reason": "excluded at install"}
            )
            continue
        if rid not in processed:
            processed[rid] = {
                "status": "held_at_install",
                "ended_at": run.get("ended_at"),
                "slug": run.get("slug"),
                "note": (
                    "Present when autobuild was installed; first auto-commit waits "
                    "for a later finished run so current traces are not published."
                ),
            }
    # Always persist the built-in exclusion even if the monitor did not list it.
    state.setdefault("excluded", {}).setdefault(
        NOOP_RMU_RUN_ID, dict(DEFAULT_EXCLUSIONS[NOOP_RMU_RUN_ID])
    )
    return state


def fetch_monitor_status() -> dict[str, Any] | None:
    try:
        import urllib.request

        req = urllib.request.Request(MONITOR_URL, method="GET")
        with urllib.request.urlopen(req, timeout=POLL_CURL_TIMEOUT_S) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception as exc:  # noqa: BLE001 — tunnel may be down; retry later
        log(f"monitor unreachable at {MONITOR_URL}: {exc.__class__.__name__}")
        return None


# ---------------------------------------------------------------------------
# Lock, quiet hours, last_build.json
# ---------------------------------------------------------------------------


@contextmanager
def acquire_lock(lock_path: Path, timeout: int = LOCK_TIMEOUT_S) -> Iterator[None]:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR, 0o644)
    deadline = time.time() + timeout
    try:
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError as exc:
                if time.time() >= deadline:
                    raise LockBusy(f"could not acquire {lock_path} within {timeout}s") from exc
                time.sleep(0.25)
        # Advertise our pid so another worker can see who holds it.
        os.ftruncate(fd, 0)
        os.write(fd, f"{os.getpid()}\n".encode("ascii"))
        yield
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


def check_quiet_hours() -> None:
    hooks = Path.home() / ".config" / "git" / "hooks"
    if str(hooks) not in sys.path:
        sys.path.insert(0, str(hooks))
    import guard  # type: ignore

    guard.check_git_time()


def write_last_build(payload: dict[str, Any], path: Path | None = None) -> None:
    dest = path or (LIB / "last_build.json")
    payload = dict(payload)
    payload.setdefault("ts", iso_et())
    payload.setdefault("timezone", "America/New_York")
    payload.setdefault("pushed", False)
    payload.setdefault("ready_to_push", False)
    save_json(dest, payload)


# ---------------------------------------------------------------------------
# Redaction report + scan
# ---------------------------------------------------------------------------


def parse_redact_report(text: str) -> dict[str, Any]:
    files_read = None
    replacements: dict[str, int] = {}
    leftover_clean = False
    leftover_count = 0
    m = re.search(r"Files read:\s*(\d+)", text)
    if m:
        files_read = int(m.group(1))
    in_repl = False
    for line in text.splitlines():
        if line.startswith("## Replacements"):
            in_repl = True
            continue
        if in_repl:
            if line.startswith("## "):
                in_repl = False
            else:
                rm = re.match(r"\|\s*([A-Za-z0-9_]+)\s*\|\s*(\d+)\s*\|", line)
                if rm and rm.group(1).lower() != "kind":
                    replacements[rm.group(1)] = int(rm.group(2))
    leftover_section = re.search(
        r"## Leftover token-like strings\s*(.+?)(?:\n## |\Z)",
        text,
        re.S,
    )
    if leftover_section:
        body = leftover_section.group(1).strip()
        if re.fullmatch(r"None\.?", body, re.I):
            leftover_clean = True
            leftover_count = 0
        else:
            leftover_clean = False
            leftover_count = len(re.findall(r"^\|", body, re.M))
            # Header + separator rows are not leftovers.
            leftover_count = max(0, leftover_count - 2)
            if leftover_count == 0 and body and not re.fullmatch(r"None\.?", body, re.I):
                leftover_count = 1
    else:
        leftover_clean = False
    errors: list[str] = []
    err_section = re.search(r"## Errors\s*(.+?)(?:\n## |\Z)", text, re.S)
    if err_section:
        for raw in err_section.group(1).splitlines():
            line = raw.strip().lstrip("- ").strip()
            if line:
                errors.append(line)
    real_errors = [e for e in errors if "skipped non-text file" not in e]
    return {
        "clean": leftover_clean and leftover_count == 0 and not real_errors,
        "leftover": leftover_count,
        "replacements": replacements,
        "files_read": files_read,
        "replacement_total": sum(replacements.values()),
        "errors": errors,
        "real_errors": real_errors,
    }


def parse_prepublish_report(text: str) -> dict[str, Any]:
    def grab(label: str) -> int | None:
        m = re.search(rf"{label}[:\s]+(\d+)", text, re.I)
        return int(m.group(1)) if m else None

    blocking = grab("blocking")
    if blocking is None:
        m = re.search(r"publication-blocking[^\n]*?(\d+)", text, re.I)
        blocking = int(m.group(1)) if m else None
    return {
        "blocking": blocking,
        "history_secrets": grab("history") or grab("history secrets"),
        "local_only": grab("local-only") or grab("local only"),
        "allowlisted": grab("allowlisted") or grab("allowlist"),
        "pass": blocking == 0,
    }


# ---------------------------------------------------------------------------
# Git: Terminal (/usr/bin/git) vs launchd (dulwich). Never push.
# ---------------------------------------------------------------------------


def git_usable_on(repo: Path) -> bool:
    try:
        proc = subprocess.run(
            git_argv("-C", str(repo), "rev-parse", "--is-inside-work-tree"),
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError, PushRefused):
        return False
    return proc.returncode == 0 and "true" in (proc.stdout or "").lower()


def rsync_usable_on(path: Path) -> bool:
    probe = path / ".lexsi-pages-rsync-probe"
    # Do not leave a probe file in the user's repo if rsync works; use /tmp dest.
    dest = LIB / ".rsync-probe-dest"
    try:
        src = path / "README.md"
        if not src.is_file():
            src = path
        proc = subprocess.run(
            ["/usr/bin/rsync", "-a", str(src), str(dest)],
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
        if dest.exists():
            if dest.is_file():
                dest.unlink()
            elif dest.is_dir() and dest.name.startswith(".rsync"):
                shutil.rmtree(dest, ignore_errors=True)
        return proc.returncode == 0
    except OSError:
        return False


def _author_ident(repo: Path) -> tuple[str, str]:
    """Return (name, email) from git config. Callers must not log the email."""
    name, email = "Mackenzie", ""
    for cfg_path in (repo / ".git" / "config", Path.home() / ".gitconfig"):
        if not cfg_path.is_file():
            continue
        section = None
        for raw in cfg_path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = raw.strip()
            if line.startswith("[") and line.endswith("]"):
                section = line[1:-1].strip().strip('"')
                continue
            if section == "user" and "=" in line:
                k, _, v = line.partition("=")
                k, v = k.strip(), v.strip().strip('"')
                if k == "name" and v:
                    name = v
                elif k == "email" and v:
                    email = v
    return name, email


def _load_dulwich():
    try:
        from dulwich.objects import Blob, Commit, Tree  # noqa: F401
        from dulwich.repo import Repo  # noqa: F401

        return True
    except ImportError:
        venv_py = LIB / "venv" / "bin" / "python3"
        if venv_py.is_file() and Path(sys.executable).resolve() != venv_py.resolve():
            raise AutobuildError(
                f"dulwich is not importable in {sys.executable}; "
                f"re-run via {venv_py}"
            )
        raise AutobuildError("dulwich is required to write a local commit from launchd")


def build_tree_from_path(object_store: Any, root: Path) -> bytes:
    """Return the tree id (bytes hex) for files under root."""
    from dulwich.objects import Blob, Tree

    def build(dirpath: Path) -> bytes:
        tree = Tree()
        entries: list[tuple[str, int, bytes]] = []
        try:
            children = sorted(dirpath.iterdir(), key=lambda p: p.name)
        except FileNotFoundError:
            children = []
        for child in children:
            if child.name in SKIP_DOC_NAMES:
                continue
            if child.is_symlink():
                continue
            if child.is_dir():
                sha = build(child)
                entries.append((child.name, 0o040000, sha))
            elif child.is_file():
                data = child.read_bytes()
                blob = Blob.from_string(data)
                object_store.add_object(blob)
                mode = 0o100755 if os.access(child, os.X_OK) else 0o100644
                entries.append((child.name, mode, blob.id))
        for name, mode, sha in entries:
            tree.add(name.encode("utf-8"), mode, sha)
        object_store.add_object(tree)
        return tree.id

    return build(root)


def replace_subtree(object_store: Any, root_tree_id: bytes, name: str, new_id: bytes) -> bytes:
    from dulwich.objects import Tree

    old = object_store[root_tree_id]
    new = Tree()
    for entry in old.items():
        if entry.path == name.encode("utf-8"):
            continue
        new.add(entry.path, entry.mode, entry.sha)
    new.add(name.encode("utf-8"), 0o040000, new_id)
    object_store.add_object(new)
    return new.id


def update_index_docs(repo_obj: Any, repo: Path, docs: Path) -> None:
    from dulwich.index import IndexEntry

    index = repo_obj.open_index()
    # Drop existing docs/ entries only.
    to_delete = [p for p in list(index) if p == b"docs" or p.startswith(b"docs/")]
    for p in to_delete:
        del index[p]
    for path in sorted(docs.rglob("*")):
        if not path.is_file() or path.name in SKIP_DOC_NAMES or path.is_symlink():
            continue
        rel = path.relative_to(repo).as_posix().encode("utf-8")
        st = path.stat()
        data = path.read_bytes()
        from dulwich.objects import Blob

        blob = Blob.from_string(data)
        repo_obj.object_store.add_object(blob)
        mode = 0o100755 if (st.st_mode & stat.S_IXUSR) else 0o100644
        index[rel] = IndexEntry(
            ctime=(int(st.st_ctime), 0),
            mtime=(int(st.st_mtime), 0),
            dev=int(st.st_dev),
            ino=int(st.st_ino),
            mode=mode,
            uid=int(st.st_uid),
            gid=int(st.st_gid),
            size=int(st.st_size),
            sha=blob.id,
        )
    index.write()


def _append_reflog(gitdir: Path, new_sha: str, old_sha: str, ident: str, message: str) -> None:
    line = f"{old_sha} {new_sha} {ident} {message}\n"
    for rel in ("logs/HEAD", "logs/refs/heads/main"):
        path = gitdir / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(line)


def commit_docs_dulwich(repo: Path, subject: str, body: str) -> tuple[str | None, str]:
    """Create a local commit that only updates docs/. Returns (sha or None, note)."""
    _load_dulwich()
    from dulwich.objects import Commit
    from dulwich.repo import Repo

    repo_obj = Repo(str(repo))
    docs = repo / "docs"
    if not docs.is_dir():
        raise AutobuildError("docs/ is missing; build did not produce a Pages bundle")

    try:
        head = repo_obj.head()
    except KeyError as exc:
        raise AutobuildError("repository has no HEAD") from exc
    head_commit = repo_obj[head]
    new_docs = build_tree_from_path(repo_obj.object_store, docs)
    new_tree = replace_subtree(repo_obj.object_store, head_commit.tree, "docs", new_docs)
    if new_tree == head_commit.tree:
        return None, "docs/ unchanged; skipped commit"

    name, email = _author_ident(repo)
    ident = f"{name} <{email}>".encode("utf-8")
    now = int(time.time())
    # America/New_York offset at this instant (EST or EDT).
    tz = int(now_et().utcoffset().total_seconds()) if now_et().utcoffset() else -5 * 3600
    message = subject.strip() + "\n"
    if body.strip():
        message += "\n" + body.strip() + "\n"

    commit = Commit()
    commit.tree = new_tree
    commit.parents = [head]
    commit.author = ident
    commit.committer = ident
    commit.author_time = now
    commit.commit_time = now
    commit.author_timezone = tz
    commit.commit_timezone = tz
    commit.message = message.encode("utf-8")
    repo_obj.object_store.add_object(commit)

    ref = b"refs/heads/main"
    # Honor a detached or other-branch HEAD only if it is the current branch.
    head_file = (repo / ".git" / "HEAD").read_text(encoding="utf-8").strip()
    if head_file.startswith("ref: "):
        ref = head_file.split(" ", 1)[1].strip().encode("utf-8")
    else:
        raise AutobuildError("HEAD is detached; refusing to commit")

    if not repo_obj.refs.set_if_equals(ref, head, commit.id):
        raise AutobuildError("ref update failed (race); will retry next interval")

    update_index_docs(repo_obj, repo, docs)
    ident_s = ident.decode("utf-8")
    _append_reflog(
        repo / ".git",
        commit.id.decode("ascii"),
        head.decode("ascii"),
        f"{ident_s} {now} {now_et().strftime('%z')}",
        "\tcommit: " + subject.strip(),
    )
    (repo / ".git" / "COMMIT_EDITMSG").write_text(message, encoding="utf-8")
    return commit.id.decode("ascii"), "committed"


def commit_docs_git(repo: Path, subject: str, body: str) -> tuple[str | None, str]:
    """Terminal path: real git add -f docs/ && commit. Still refuses push."""
    status = subprocess.run(
        git_argv("-C", str(repo), "status", "--porcelain", "--", "docs"),
        check=True,
        capture_output=True,
        text=True,
    )
    if not status.stdout.strip():
        # Also consider untracked-but-ignored docs/data via force-add dry run.
        planned = subprocess.run(
            git_argv("-C", str(repo), "add", "-f", "-n", "--", "docs"),
            check=False,
            capture_output=True,
            text=True,
        )
        if not (planned.stdout or "").strip() and not status.stdout.strip():
            return None, "docs/ unchanged; skipped commit"

    add = subprocess.run(
        git_argv("-C", str(repo), "add", "-f", "--", "docs"),
        check=False,
        capture_output=True,
        text=True,
    )
    if add.returncode != 0:
        raise AutobuildError(f"git add -f docs/ failed: {add.stderr.strip()}")

    # Confirm nothing outside docs/ is staged.
    staged = subprocess.run(
        git_argv("-C", str(repo), "diff", "--cached", "--name-only"),
        check=True,
        capture_output=True,
        text=True,
    )
    bad = [p for p in staged.stdout.splitlines() if p and p != "docs" and not p.startswith("docs/")]
    if bad:
        subprocess.run(git_argv("-C", str(repo), "restore", "--staged", "--", "."), check=False)
        raise AutobuildError("refusing to commit: staged paths outside docs/")

    if not staged.stdout.strip():
        return None, "docs/ unchanged; skipped commit"

    msg_file = LIB / ".commit-msg"
    msg_file.write_text(subject.strip() + "\n\n" + body.strip() + "\n", encoding="utf-8")
    commit = subprocess.run(
        git_argv("-C", str(repo), "commit", "-F", str(msg_file), "--", "docs"),
        check=False,
        capture_output=True,
        text=True,
    )
    try:
        msg_file.unlink()
    except OSError:
        pass
    if commit.returncode != 0:
        raise AutobuildError(
            "git commit failed: " + (commit.stderr or commit.stdout or "").strip()
        )
    sha = subprocess.run(
        git_argv("-C", str(repo), "rev-parse", "HEAD"),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return sha, "committed"


def commit_docs(repo: Path, subject: str, body: str, prefer_git: bool | None = None) -> tuple[str | None, str]:
    if prefer_git is None:
        prefer_git = git_usable_on(repo)
    if prefer_git:
        return commit_docs_git(repo, subject, body)
    return commit_docs_dulwich(repo, subject, body)


# ---------------------------------------------------------------------------
# Sync + build (TCC-safe from launchd)
# ---------------------------------------------------------------------------


def read_spark_env(repo: Path) -> dict[str, str]:
    env: dict[str, str] = {}
    path = repo / ".env"
    if not path.is_file():
        raise AutobuildError("No .env in the repo")
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        k = k.strip()
        v = v.strip().strip("'").strip('"')
        env[k] = v
    for required in ("SPARK_HOST", "SPARK_USER", "SPARK_PROJECT_DIR"):
        if not env.get(required) or "XXXX" in env.get(required, ""):
            raise AutobuildError(f"{required} is missing or still a placeholder")
    return env


def ssh_base_cmd(spark_env: dict[str, str]) -> list[str]:
    port = spark_env.get("SPARK_SSH_PORT") or "22"
    identity = str(Path.home() / ".ssh" / "id_ed25519")
    # Prefer the launchd wrapper if it is on PATH or in LIB/bin.
    wrapper = LIB / "bin" / "ssh"
    ssh = str(wrapper) if wrapper.is_file() else "ssh"
    return [ssh, "-p", str(port), "-o", "BatchMode=yes"]


def sync_results(
    repo: Path, spark_env: dict[str, str], use_native: bool
) -> None:
    """Pull --results from the Spark. Native path uses the repo script."""
    if use_native:
        script = repo / "scripts" / "sync_from_spark.sh"
        proc = subprocess.run(
            ["/bin/bash", str(script), "--results"],
            cwd=str(repo),
            check=False,
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            raise AutobuildError(
                "sync_from_spark.sh --results failed: "
                + (proc.stderr or proc.stdout or "")[-2000:]
            )
        log("sync --results (native script) finished")
        return

    # TCC-safe: rsync into Application Support, then Python-copy into the repo.
    staging = LIB / "sync-staging"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    output_dir = spark_env.get("OUTPUT_DIR") or "artifacts"
    remote = f"{spark_env['SPARK_USER']}@{spark_env['SPARK_HOST']}"
    project = spark_env["SPARK_PROJECT_DIR"].rstrip("/")
    ssh = ssh_base_cmd(spark_env)
    # rsync -e needs a single string.
    rsh = " ".join(ssh)
    excludes = [
        "--exclude", ".env",
        "--exclude", "*.safetensors",
        "--exclude", "*.bin",
        "--exclude", "*.pt",
        "--exclude", "*.pth",
        "--exclude", "*.ckpt",
        "--exclude", "*.gguf",
        "--exclude", "checkpoint-*/",
        "--exclude", "rmu_unlearned/",
        "--exclude", "inject_lora/",
        "--exclude", "npo_unlearned/",
    ]

    def remote_test(kind: str, rel: str) -> bool:
        target = f"{project}/{rel}"
        flag = "-d" if kind == "dir" else "-f"
        proc = subprocess.run(
            [*ssh, remote, f"test {flag} '{target}'"],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
        return proc.returncode == 0

    def rsync_to_staging(remote_rel: str, dest: Path) -> None:
        dest.mkdir(parents=True, exist_ok=True)
        src = f"{remote}:{project}/{remote_rel}"
        if not remote_rel.endswith("/") and not Path(remote_rel).suffix:
            # directories we pass already include trailing semantics via test
            pass
        proc = subprocess.run(
            ["/usr/bin/rsync", "-az", "-e", rsh, *excludes, src, str(dest)],
            check=False,
            capture_output=True,
            text=True,
            timeout=60 * 30,
        )
        if proc.returncode != 0:
            raise AutobuildError(
                f"rsync to staging failed for {remote_rel}: "
                + (proc.stderr or proc.stdout or "")[-2000:]
            )

    for sub in ("traces", "results", "interp", "circuitkit"):
        rel = f"{output_dir}/{sub}"
        if remote_test("dir", rel):
            log(f"  staging {rel}/")
            rsync_to_staging(rel + "/", staging / output_dir / sub)
        else:
            log(f"  skip {rel}/ (not on the Spark yet)")

    review = f"{output_dir}/translations_review.md"
    if remote_test("file", review):
        rsync_to_staging(review, staging / output_dir)

    if remote_test("dir", "logs"):
        log("  staging logs/")
        rsync_to_staging("logs/", staging / "logs")

    if remote_test("file", "data/translations.json"):
        log("  staging data/translations.json")
        rsync_to_staging("data/translations.json", staging / "data")

    _python_merge(staging, repo)
    log("sync --results (staging + Python merge) finished")


def _python_merge(src_root: Path, dest_root: Path) -> None:
    if not src_root.is_dir():
        return
    for path in src_root.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(src_root)
        dest = dest_root / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        data = path.read_bytes()
        dest.write_bytes(data)
        try:
            shutil.copystat(path, dest, follow_symlinks=True)
        except OSError:
            pass


def detect_evalboard() -> Path:
    env = os.environ.get("EVALBOARD_ROOT")
    candidates = []
    if env:
        candidates.append(Path(env))
    candidates.extend(
        [
            EVALBOARD_APP_SUPPORT,
            Path.home() / "Documents" / "Code" / "modular_dashboard",
            Path.home() / "Documents" / "code" / "modular_dashboard",
        ]
    )
    for c in candidates:
        if (c / "evalboard" / "export_static.py").is_file():
            return c
    raise AutobuildError("Could not find Evalboard (set EVALBOARD_ROOT)")


def pipeline_python() -> str:
    if FRAMEWORK_PYTHON.is_file():
        return str(FRAMEWORK_PYTHON)
    return sys.executable


def run_pages_build(
    repo: Path,
    excluded_ids: set[str],
    allowed_ids: set[str] | None = None,
) -> dict[str, Any]:
    py = pipeline_python()
    report = repo / "artifacts" / "redact_report.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    redact = subprocess.run(
        [
            py,
            str(repo / "scripts" / "redact_release.py"),
            "--root",
            str(repo),
            "--hf-out",
            str(repo / "release" / "hf_dataset"),
            "--pages-out",
            str(repo / "release" / "pages_src"),
            "--report",
            str(report),
        ],
        cwd=str(repo),
        check=False,
        capture_output=True,
        text=True,
    )
    report_text = report.read_text(encoding="utf-8") if report.is_file() else ""
    redact_info = parse_redact_report(report_text)
    if redact.returncode != 0:
        # Older redact_release.py treated non-text skips as errors and exited 1
        # even with 0 leftover tokens. If the report itself is clean, continue.
        if redact_info.get("clean") and not redact_info.get("real_errors"):
            log(
                "redact exited non-zero but the report is clean "
                "(non-text skips only); continuing"
            )
        else:
            raise AutobuildError(
                "redact_release.py failed: " + (redact.stderr or redact.stdout or "")[-2000:]
            )
    if not redact_info["clean"]:
        raise AutobuildError(
            f"redaction report is not clean (leftover={redact_info.get('leftover')})"
        )

    removed = filter_excluded_release_traces(repo, excluded_ids, allowed_ids)
    if removed:
        log(f"filtered {removed} excluded or in-progress file(s) from release/")

    evalboard = detect_evalboard()
    env = os.environ.copy()
    env["PYTHONPATH"] = str(evalboard) + (
        (os.pathsep + env["PYTHONPATH"]) if env.get("PYTHONPATH") else ""
    )
    env["EVALBOARD_ROOT"] = str(evalboard)

    db = repo / "release" / "pages_workspace.sqlite3"
    if db.exists():
        db.unlink()

    traces_dir = repo / "release" / "pages_src" / "traces"
    traces = sorted(traces_dir.glob("*.jsonl")) if traces_dir.is_dir() else []
    if allowed_ids is not None:
        traces = [f for f in traces if f.stem in allowed_ids]
    if not traces:
        raise AutobuildError("No scrubbed traces left to export after exclusions")

    for f in traces:
        run_id = f.stem
        log(f"ingest {run_id}")
        ing = subprocess.run(
            [
                py,
                str(repo / "scripts" / "ingest_evalboard.py"),
                "--db",
                str(db),
                "--run",
                run_id,
                "--output-dir",
                str(repo / "release" / "pages_src"),
            ],
            cwd=str(repo),
            env=env,
            check=False,
            capture_output=True,
            text=True,
        )
        if ing.returncode != 0:
            raise AutobuildError(
                f"ingest {run_id} failed: " + (ing.stderr or ing.stdout or "")[-2000:]
            )

    docs = repo / "docs"
    docs.mkdir(parents=True, exist_ok=True)
    exp = subprocess.run(
        [
            py,
            "-m",
            "evalboard.export_static",
            "--db",
            str(db),
            "--out",
            str(docs),
            "--trim-bytes",
            os.environ.get("TRIM_BYTES", "250000"),
        ],
        cwd=str(repo),
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    if exp.returncode != 0:
        raise AutobuildError(
            "export_static failed: " + (exp.stderr or exp.stdout or "")[-2000:]
        )
    return redact_info


RUN_ID_RE = re.compile(r"20\d{6}T\d{6}Z-[A-Za-z0-9+_.:-]*--[0-9a-fA-F]{8,}")


def run_ids_in_path(rel: str) -> list[str]:
    return RUN_ID_RE.findall(rel.replace("\\", "/"))


def allowed_finished_run_ids(status: dict[str, Any], state: dict[str, Any]) -> set[str]:
    """Finished, non-excluded run_ids — including ones held at install (e.g. smoke)."""
    allowed: set[str] = set()
    for run in iter_finished_runs(status):
        rid = str(run["run_id"])
        if not is_excluded(rid, state):
            allowed.add(rid)
    return allowed


def filter_excluded_release_traces(
    repo: Path,
    excluded_ids: set[str],
    allowed_ids: set[str] | None = None,
) -> int:
    """Drop release files for excluded or not-yet-finished run_ids (path-wide)."""
    removed = 0
    roots = [
        repo / "release" / "pages_src",
        repo / "release" / "hf_dataset",
    ]
    fake_state = {"excluded": {i: {} for i in excluded_ids}}
    for root in roots:
        if not root.exists():
            continue
        for path in list(root.rglob("*")):
            if not path.is_file():
                continue
            rel = path.relative_to(root).as_posix()
            ids = run_ids_in_path(rel) or run_ids_in_path(path.stem)
            drop = False
            for rid in ids:
                if is_excluded(rid, fake_state) or rid in excluded_ids:
                    drop = True
                    break
                if allowed_ids is not None and rid not in allowed_ids:
                    drop = True
                    break
            if drop:
                path.unlink()
                removed += 1
        # Remove empty leftover dirs (e.g. checkpoints/, excluded run folders).
        for dirpath in sorted((d for d in root.rglob("*") if d.is_dir()), key=lambda d: len(d.parts), reverse=True):
            try:
                next(dirpath.iterdir())
            except StopIteration:
                dirpath.rmdir()
            except OSError:
                pass
    return removed


def excluded_id_set(state: dict[str, Any]) -> set[str]:
    return set((state.get("excluded") or {}).keys()) | set(DEFAULT_EXCLUSIONS)


def run_prepublish_scan(repo: Path, use_native: bool) -> dict[str, Any]:
    """Run the project prepublish scan. From launchd, copy targets to LIB first."""
    py = pipeline_python()
    report = repo / "artifacts" / "prepublish_report.md"
    if use_native:
        proc = subprocess.run(
            ["/bin/bash", str(repo / "scripts" / "prepublish_scan.sh"), "--report", str(report)],
            cwd=str(repo),
            check=False,
            capture_output=True,
            text=True,
        )
        text = report.read_text(encoding="utf-8") if report.is_file() else (proc.stdout or "")
        info = parse_prepublish_report(text)
        info["exit"] = proc.returncode
        info["pass"] = proc.returncode == 0 and (info.get("blocking") in (0, None))
        if proc.returncode == 0 and info.get("blocking") is None:
            info["pass"] = True
        if not info["pass"]:
            snippet = (proc.stderr or proc.stdout or "")[-2000:]
            info["error"] = snippet
        return info

    return _scan_tcc_safe(repo, py, report)


def _gitleaks_bin() -> str | None:
    for c in (
        Path.home() / ".local" / "bin" / "gitleaks",
        Path("/opt/homebrew/bin/gitleaks"),
        Path("/usr/local/bin/gitleaks"),
    ):
        if c.is_file() and os.access(c, os.X_OK):
            return str(c)
    found = shutil.which("gitleaks")
    return found


def _python_copy_tree(src: Path, dest: Path, ignore_names: set[str] | None = None) -> None:
    ignore_names = ignore_names or set()
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True, exist_ok=True)
    if not src.exists():
        return
    if src.is_file():
        dest.parent.mkdir(parents=True, exist_ok=True)
        (dest if dest.suffix else dest / src.name).write_bytes(src.read_bytes())
        return
    for path in src.rglob("*"):
        rel = path.relative_to(src)
        if any(p in ignore_names for p in rel.parts):
            continue
        if path.is_dir():
            (dest / rel).mkdir(parents=True, exist_ok=True)
        elif path.is_file() and not path.is_symlink():
            target = dest / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(path.read_bytes())


def _scan_tcc_safe(repo: Path, py: str, report: Path) -> dict[str, Any]:
    gitleaks = _gitleaks_bin()
    if not gitleaks:
        raise AutobuildError("gitleaks is not installed")
    scan_root = LIB / "scan-tree"
    if scan_root.exists():
        shutil.rmtree(scan_root)
    scan_root.mkdir(parents=True)
    # Copy the parts the publication scan cares about (Python can read Documents).
    for rel in (
        "docs",
        "paper",
        "presentation",
        "scripts",
        "src",
        "tests",
        "release",
        "README.md",
        ".gitleaks.toml",
        ".prepublish_allowlist",
        ".gitignore",
        ".prepublish_identities.example",
    ):
        src = repo / rel
        if src.exists():
            dest = scan_root / rel
            if src.is_dir():
                _python_copy_tree(src, dest, ignore_names={"__pycache__", ".git"})
            else:
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(src.read_bytes())

    config = scan_root / ".gitleaks.toml"
    if not config.is_file():
        # fall back to the repo file already copied; if missing, copy again
        raw = repo / ".gitleaks.toml"
        if raw.is_file():
            config.write_bytes(raw.read_bytes())

    tmp = LIB / "scan-tmp"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir()
    git_json = tmp / "gitleaks-git.json"
    dir_json = tmp / "gitleaks-dir.json"

    # History: mirror objects into the bare clone (Library) and scan that.
    bare = LIB / "lexsi-unlearning.git"
    _mirror_git_dir(repo / ".git", bare)
    git_ok = False
    git_note = "gitleaks history scan was skipped."
    if bare.is_dir():
        hist = subprocess.run(
            [
                gitleaks,
                "git",
                "--no-banner",
                "--no-color",
                "--redact",
                "--exit-code",
                "0",
                "-l",
                "warn",
                "-c",
                str(repo / ".gitleaks.toml") if (repo / ".gitleaks.toml").is_file() else str(config),
                "-f",
                "json",
                "-r",
                str(git_json),
                str(bare),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if hist.returncode == 0:
            git_ok = True
            git_note = "gitleaks scanned the mirrored git history."
        else:
            git_note = f"gitleaks history scan failed (exit {hist.returncode})."
        if not git_json.is_file() or git_json.stat().st_size == 0:
            git_json.write_text("[]\n", encoding="utf-8")
    else:
        git_json.write_text("[]\n", encoding="utf-8")

    dire = subprocess.run(
        [
            gitleaks,
            "dir",
            "--no-banner",
            "--no-color",
            "--redact",
            "--exit-code",
            "0",
            "-l",
            "warn",
            "-c",
            str(config),
            "-f",
            "json",
            "-r",
            str(dir_json),
            str(scan_root),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if dire.returncode != 0:
        raise AutobuildError(f"gitleaks dir scan failed (exit {dire.returncode})")
    if not dir_json.is_file() or dir_json.stat().st_size == 0:
        dir_json.write_text("[]\n", encoding="utf-8")

    version = subprocess.run(
        [gitleaks, "version"], check=False, capture_output=True, text=True
    ).stdout.strip().splitlines()
    version_s = version[0] if version else "unknown"

    args = [
        py,
        str(repo / "scripts" / "prepublish_scan.py"),
        "--root",
        str(repo),
        "--report",
        str(report),
        "--allowlist",
        str(repo / ".prepublish_allowlist"),
        "--gitleaks-git",
        str(git_json),
        "--gitleaks-dir",
        str(dir_json),
        "--gitleaks-version",
        version_s,
        "--git-history-note",
        git_note,
    ]
    if git_ok:
        args.append("--git-history-ok")
    identities = repo / ".prepublish_identities"
    if identities.is_file():
        args.extend(["--identities", str(identities)])
    proc = subprocess.run(args, cwd=str(repo), check=False, capture_output=True, text=True)
    text = report.read_text(encoding="utf-8") if report.is_file() else (proc.stdout or "")
    info = parse_prepublish_report(text)
    info["exit"] = proc.returncode
    info["pass"] = proc.returncode == 0 and (info.get("blocking") in (0, None))
    if proc.returncode == 0 and info.get("blocking") is None:
        info["pass"] = True
    if not info["pass"]:
        info["error"] = (proc.stderr or proc.stdout or "")[-2000:]
    return info


def _mirror_git_dir(src_git: Path, bare: Path) -> None:
    """Copy objects + refs from the Documents repo into a Library bare repo."""
    if not src_git.is_dir():
        return
    bare.mkdir(parents=True, exist_ok=True)
    for rel in ("objects", "refs", "packed-refs", "HEAD", "info", "config"):
        s = src_git / rel
        if not s.exists():
            continue
        d = bare / rel
        if s.is_dir():
            _python_copy_tree(s, d, ignore_names=set())
        else:
            d.parent.mkdir(parents=True, exist_ok=True)
            d.write_bytes(s.read_bytes())


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def cycle(
    repo: Path,
    state: dict[str, Any],
    *,
    once: bool = False,
) -> dict[str, Any]:
    state = merge_exclusions(state)
    if (LIB / "DISABLED").is_file():
        payload = {
            "ok": True,
            "action": "disabled",
            "message": "DISABLED flag present; no work done",
            "ready_to_push": False,
        }
        write_last_build(payload)
        log("DISABLED flag present; exiting")
        return payload

    status = fetch_monitor_status()
    if status is None:
        payload = {
            "ok": True,
            "action": "tunnel_down",
            "message": "monitor at 127.0.0.1:8790 is unreachable; will retry",
            "ready_to_push": False,
        }
        write_last_build(payload)
        return payload

    pending = new_runs_to_build(status, state)
    if not pending:
        payload = {
            "ok": True,
            "action": "idle",
            "run_ids": [],
            "message": "no new finished run_ids to build (excluded and already-processed skipped)",
            "ready_to_push": False,
            "excluded": sorted(excluded_id_set(state)),
        }
        write_last_build(payload)
        log("idle: no new finished runs")
        return payload

    run_ids = [str(r["run_id"]) for r in pending]
    latest = run_ids[-1]
    log(f"new finished runs: {', '.join(run_ids)}")

    try:
        check_quiet_hours()
    except Exception as exc:  # quiet-hours or import
        payload = {
            "ok": True,
            "action": "quiet_hours",
            "run_ids": run_ids,
            "message": str(exc),
            "ready_to_push": False,
        }
        write_last_build(payload)
        log(f"quiet hours (will retry): {exc}")
        return payload

    use_native_git = git_usable_on(repo)
    use_native_rsync = rsync_usable_on(repo) if use_native_git else False
    log(
        "tool path: "
        + ("native git/rsync (Terminal)" if use_native_git and use_native_rsync else "TCC-safe (Python/dulwich, staged rsync)")
    )

    spark_env = read_spark_env(repo)
    sync_results(repo, spark_env, use_native=use_native_git and use_native_rsync)

    try:
        allowed = allowed_finished_run_ids(status, state)
        log("pages ingest allowlist: " + (", ".join(sorted(allowed)) or "(none)"))
        redact_info = run_pages_build(repo, excluded_id_set(state), allowed)
    except AutobuildError as exc:
        payload = {
            "ok": False,
            "action": "redact_failed" if "redact" in str(exc).lower() or "leftover" in str(exc).lower() else "build_failed",
            "run_ids": run_ids,
            "message": str(exc),
            "ready_to_push": False,
            "scan": {"pass": False},
        }
        write_last_build(payload)
        log(f"build/redact failed (not marking processed): {exc}")
        return payload

    scan = run_prepublish_scan(repo, use_native=use_native_git)
    if not scan.get("pass"):
        payload = {
            "ok": False,
            "action": "scan_failed",
            "run_ids": run_ids,
            "message": scan.get("error") or "prepublish scan failed",
            "ready_to_push": False,
            "scan": scan,
            "redaction": redact_info,
        }
        write_last_build(payload)
        log("prepublish scan failed (not marking processed, not committing)")
        return payload

    subject = f"pages: update after {latest}"
    body_lines = [
        "Local-only Pages rebuild after finished Spark run(s):",
        "",
        *[f"- {r}" for r in run_ids],
        "",
        "This commit is local only. Do not push until Mackenzie confirms.",
    ]
    try:
        sha, note = commit_docs(repo, subject, "\n".join(body_lines), prefer_git=use_native_git)
    except AutobuildError as exc:
        payload = {
            "ok": False,
            "action": "commit_failed",
            "run_ids": run_ids,
            "message": str(exc),
            "ready_to_push": False,
            "scan": scan,
            "redaction": redact_info,
        }
        write_last_build(payload)
        log(f"commit failed (not marking processed): {exc}")
        return payload

    for run in pending:
        rid = str(run["run_id"])
        state.setdefault("processed", {})[rid] = {
            "status": run.get("status"),
            "ended_at": run.get("ended_at"),
            "slug": run.get("slug"),
            "commit": sha,
            "built_at": iso_et(),
            "note": note,
        }
    save_json(LIB / "state.json", state)

    ready = bool(sha)
    payload = {
        "ok": True,
        "action": "committed" if sha else "skipped_unchanged",
        "run_ids": run_ids,
        "commit": sha,
        "message": note,
        "ready_to_push": ready,
        "scan": scan,
        "redaction": redact_info,
        "subject": subject if sha else None,
    }
    write_last_build(payload)
    log(f"{note} commit={sha or 'none'} scan=pass leftover={redact_info.get('leftover')}")
    return payload


# ---------------------------------------------------------------------------
# Self-test (dry data; does not commit the real repo or ingest 631cb361)
# ---------------------------------------------------------------------------


def self_test() -> dict[str, Any]:
    log("starting self-test (dry data; will not touch the real docs/ commit)")
    results: dict[str, Any] = {"checks": {}}
    errors: list[str] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        results["checks"][name] = {"ok": ok, "detail": detail}
        if not ok:
            errors.append(f"{name}: {detail}")
        log(("PASS " if ok else "FAIL ") + name + (f" — {detail}" if detail else ""))

    # 1. Push refusal
    try:
        git_argv("push", "origin", "main")
        check("push_refusal", False, "git_argv allowed push")
    except PushRefused:
        check("push_refusal", True, "git_argv refused push")
    try:
        refuse_push_in(["--push"])
        check("push_flag_refusal", False, "--push was allowed")
    except PushRefused:
        check("push_flag_refusal", True)

    # 2. Exclusion of the no-op RMU run
    state = merge_exclusions({"processed": {}})
    check(
        "exclude_noop_rmu",
        is_excluded(NOOP_RMU_RUN_ID, state),
        "631cb361 must be excluded",
    )
    mock_status = {
        "runs": [
            {
                "slug": "full",
                "run_id": NOOP_RMU_RUN_ID,
                "status": "done",
                "ended_at": "2026-11-12T22:26:06-05:00",
            },
            {
                "slug": "smoke+translate",
                "run_id": "20261112T220000Z-smoke+translate--64148c72",
                "status": "done",
                "ended_at": "2026-11-12T17:14:04-05:00",
            },
            {"slug": "earth", "run_id": None, "status": "queued"},
        ]
    }
    held = seed_held_at_install(mock_status, {"processed": {}, "excluded": {}})
    pending = new_runs_to_build(mock_status, held)
    check(
        "hold_and_exclude_at_install",
        pending == [],
        f"pending={ [r.get('run_id') for r in pending] }",
    )
    # A later valid run should be selected, and 631cb361 still skipped.
    mock_status["runs"].append(
        {
            "slug": "full-fix2",
            "run_id": "dry-test-fix2--00c0ffee",
            "status": "done",
            "ended_at": "2026-11-13T04:00:00-05:00",
        }
    )
    pending2 = new_runs_to_build(mock_status, held)
    ids = [r["run_id"] for r in pending2]
    check(
        "selects_only_new_valid_run",
        ids == ["dry-test-fix2--00c0ffee"],
        f"ids={ids}",
    )

    # 3. Redaction parser
    clean_report = """# Redaction report

## Counts

- Files read: 4

## Replacements

| Kind | Count |
|---|---:|
| path | 2 |
| workspace_path | 3 |

## Leftover token-like strings

None.
"""
    dirty_report = clean_report.replace("None.", "| token | 1 |\n| --- | --- |\n| leftover | x |")
    # simpler dirty: not None
    dirty_report = """# Redaction report

## Leftover token-like strings

| Kind | Sample |
|---|---|
| token | [REDACTED] |
"""
    pr = parse_redact_report(clean_report)
    check("redact_parser_clean", pr["clean"] is True and pr["replacement_total"] == 5, str(pr))
    pr2 = parse_redact_report(dirty_report)
    check("redact_parser_dirty", pr2["clean"] is False and pr2["leftover"] >= 1, str(pr2))

    # 4. Lock
    lock_path = LIB / "selftest" / "test.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with acquire_lock(lock_path, timeout=2):
            busy = False
            try:
                with acquire_lock(lock_path, timeout=1):
                    pass
            except LockBusy:
                busy = True
            check("lock_blocks_second_holder", busy)
        # After release, a new holder can acquire.
        with acquire_lock(lock_path, timeout=2):
            check("lock_releases", True)
    except Exception as exc:  # noqa: BLE001
        check("lock_releases", False, str(exc))

    # 5. Dulwich local commit on a throwaway repo under Application Support
    #    (not the real lexsi repo; git works here even from launchd).
    fixture = LIB / "selftest" / "repo"
    if fixture.exists():
        shutil.rmtree(fixture)
    fixture.mkdir(parents=True)
    subprocess.run(
        git_argv("init", "-b", "main", str(fixture)),
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        git_argv("-C", str(fixture), "config", "user.name", "Mackenzie"),
        check=True,
        capture_output=True,
    )
    # Use a clearly fake local address so the fixture is not tied to a real inbox.
    subprocess.run(
        git_argv("-C", str(fixture), "config", "user.email", "pages-autobuild@selftest.invalid"),
        check=True,
        capture_output=True,
    )
    (fixture / "README.md").write_text("fixture\n", encoding="utf-8")
    (fixture / "docs").mkdir()
    (fixture / "docs" / ".nojekyll").write_text("", encoding="utf-8")
    (fixture / "docs" / "README.md").write_text("docs fixture v1\n", encoding="utf-8")
    subprocess.run(git_argv("-C", str(fixture), "add", "README.md", "docs"), check=True, capture_output=True)
    subprocess.run(
        git_argv("-C", str(fixture), "commit", "-m", "initial fixture"),
        check=True,
        capture_output=True,
        text=True,
    )
    (fixture / "docs" / "index.html").write_text("<!doctype html><title>dry</title>\n", encoding="utf-8")
    (fixture / "docs" / "data").mkdir()
    (fixture / "docs" / "data" / "workspace.json").write_text("{}\n", encoding="utf-8")
    sha, note = commit_docs_dulwich(
        fixture,
        "pages: update after dry-test-fix2--00c0ffee",
        "Dry self-test. Local only. Do not push.",
    )
    check("dulwich_commit", bool(sha) and note == "committed", f"sha={sha} note={note}")
    # Verify with real git (Library path is TCC-safe).
    if sha:
        shown = subprocess.run(
            git_argv("-C", str(fixture), "log", "-1", "--format=%H %s"),
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        check("git_log_sees_commit", shown.startswith(sha) and "pages: update after" in shown, shown)
        files = subprocess.run(
            git_argv("-C", str(fixture), "show", "--name-only", "--pretty=format:", sha),
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
        files = [f for f in files if f]
        only_docs = all(f == "docs" or f.startswith("docs/") for f in files)
        check("commit_only_docs", only_docs and "docs/index.html" in files, str(files))
        status = subprocess.run(
            git_argv("-C", str(fixture), "status", "--porcelain"),
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        check("worktree_clean_for_docs", "docs/" not in status, status or "clean")
        unchanged_sha, unchanged_note = commit_docs_dulwich(fixture, "pages: update after dry", "")
        check(
            "idempotent_skip_unchanged",
            unchanged_sha is None and "unchanged" in unchanged_note,
            unchanged_note,
        )

    # 6. Filter excluded traces (dry files, not 631cb361 content)
    rel = fixture / "release" / "pages_src" / "traces"
    rel.mkdir(parents=True)
    (rel / f"{NOOP_RMU_RUN_ID}.jsonl").write_text("{}\n", encoding="utf-8")
    (rel / "dry-test-fix2--00c0ffee.jsonl").write_text("{}\n", encoding="utf-8")
    n = filter_excluded_release_traces(fixture, {NOOP_RMU_RUN_ID})
    still_bad = (rel / f"{NOOP_RMU_RUN_ID}.jsonl").exists()
    still_good = (rel / "dry-test-fix2--00c0ffee.jsonl").exists()
    check("filters_excluded_traces", n == 1 and not still_bad and still_good, f"removed={n}")

    # 7. Quiet-hours helper is importable (should be allowed at this hour)
    try:
        check_quiet_hours()
        check("quiet_hours_import", True, "allowed now")
    except Exception as exc:  # noqa: BLE001
        # A block here is still "the check works"; record the message.
        if "08:00-18:00" in str(exc):
            check("quiet_hours_import", True, "blocked as policy (ok)")
        else:
            check("quiet_hours_import", False, str(exc))

    ok = not errors
    payload = {
        "ok": ok,
        "action": "self-test",
        "mode": "self-test",
        "ready_to_push": False,
        "pushed": False,
        "run_ids": ["dry-test-fix2--00c0ffee"],
        "commit": sha if ok else None,
        "scan": {"pass": None, "note": "self-test did not run prepublish on the real repo"},
        "redaction": {"clean": True, "note": "parser exercised on fixture reports only"},
        "message": "self-test passed" if ok else "self-test failed: " + "; ".join(errors),
        "checks": results["checks"],
        "real_repo_untouched": True,
    }
    write_last_build(payload)
    log("self-test " + ("PASSED" if ok else "FAILED"))
    if not ok:
        raise SystemExit("self-test failed:\n  " + "\n  ".join(errors))
    return payload


def usage() -> None:
    print(
        """Usage: pages_autobuild.py [--once | --self-test | --install-seed]

  --once          Poll the monitor once and build if a new valid run finished.
  --self-test     Dry-data checks (exclusion, lock, push refusal, local commit).
                  Does not build or commit the real repo.
  --install-seed  Record currently finished runs as held, and seed exclusions.
                  Does not build or commit.

Never accepts or performs git push.
"""
    )


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    refuse_push_in(argv)
    LIB.mkdir(parents=True, exist_ok=True)

    if not argv or argv[0] in {"-h", "--help"}:
        if argv and argv[0] in {"-h", "--help"}:
            usage()
            return 0
        argv = ["--once"]

    cmd = argv[0]
    repo = Path(os.environ.get("LEXSI_REPO") or DEFAULT_REPO)
    lock_path = LIB / "build.lock"

    if cmd == "--self-test":
        self_test()
        return 0

    if cmd == "--install-seed":
        status = fetch_monitor_status() or {"runs": []}
        state = merge_exclusions(load_json(LIB / "state.json", {"processed": {}, "excluded": {}}))
        state = seed_held_at_install(status, state)
        save_json(LIB / "state.json", state)
        log(
            "seeded state: excluded="
            + ",".join(sorted(state.get("excluded") or {}))
            + " processed="
            + ",".join(sorted(state.get("processed") or {}))
        )
        write_last_build(
            {
                "ok": True,
                "action": "install-seed",
                "ready_to_push": False,
                "excluded": sorted(state.get("excluded") or {}),
                "held": sorted(state.get("processed") or {}),
                "message": "Seeded exclusions and held already-finished runs. No build, no commit.",
            }
        )
        return 0

    if cmd not in {"--once"}:
        usage()
        return 2

    try:
        with acquire_lock(lock_path):
            state = merge_exclusions(load_json(LIB / "state.json", {"processed": {}, "excluded": {}}))
            cycle(repo, state, once=True)
        return 0
    except LockBusy as exc:
        payload = {
            "ok": True,
            "action": "lock_busy",
            "message": str(exc),
            "ready_to_push": False,
        }
        write_last_build(payload)
        log(str(exc))
        return 0
    except PushRefused:
        raise
    except Exception as exc:  # noqa: BLE001
        log("unhandled error: " + f"{exc.__class__.__name__}: {exc}")
        log(traceback.format_exc())
        write_last_build(
            {
                "ok": False,
                "action": "error",
                "message": f"{exc.__class__.__name__}: {exc}",
                "ready_to_push": False,
            }
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
