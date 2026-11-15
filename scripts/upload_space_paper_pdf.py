#!/usr/bin/env python3
"""Re-upload the Lexsi paper PDF to the private Hugging Face Space.

Use this after a paper rebuild:

    python3 scripts/upload_space_paper_pdf.py
    python3 scripts/upload_space_paper_pdf.py --wait
    python3 scripts/upload_space_paper_pdf.py --pdf paper/main.pdf --also-dataset

Never prints the HF token. Does not change Space or dataset visibility.
Does not commit or push to GitHub. Aborts if the PDF looks like it contains
PII (emails, home paths, private hostnames, public IPs, secret tokens)
or forbidden artifacts (Tiny Aya weight filenames, run 631cb361).
"""
from __future__ import annotations

import argparse
import ipaddress
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PDF = ROOT / "paper" / "main.pdf"
SPACE_ID = "mackenzietechdocs/lexsi-unlearning-demo"
SPACE_PATH = "data/paper/main.pdf"
DATASET_ID = "mackenzietechdocs/lexsi-unlearning-eval"
DATASET_PATH = "paper/main.pdf"
PAPER_URL = f"https://huggingface.co/spaces/{SPACE_ID}/resolve/main/{SPACE_PATH}"
LOCAL_SPACE_COPY = ROOT / "release" / "lexsi-unlearning-demo" / "data" / "paper" / "main.pdf"
ENV_PATH = ROOT / ".env"

EMAIL_RE = re.compile(r"\b[A-Z0-9._%+\-]+@[A-Z0-9.\-]+\.[A-Z]{2,}\b", re.I)
PATH_RE = re.compile(r"(/(?:Users|home)/[^\s/]+(?:/[^\s]*)?)")
def host_re() -> re.Pattern[str]:
    """Reject private hostnames if they appear in the PDF.

    Concrete names are loaded from the gitignored identities file at
    runtime so this script can be committed without embedding them.
    Only HOST_* and TAILNET values are used (not USER, which would
    match the paper's author line).
    """
    tokens: list[str] = [r"(?:[A-Za-z0-9-]+\.)*ts\.net"]
    ident = ROOT / ".prepublish_identities"
    if ident.is_file():
        for raw in ident.read_text(encoding="utf-8", errors="replace").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "\t" not in line:
                continue
            kind, _, val = line.partition("\t")
            val = val.strip().strip("'").strip('"').rstrip(".")
            if not (kind.startswith("HOST") or kind == "TAILNET"):
                continue
            if not val or "/" in val or "XXXX" in val:
                continue
            # Bare tokens shorter than 8 letters collide with ordinary
            # paper words (author affiliation, "stream", etc.).
            if "." not in val and len(val) < 8:
                continue
            tokens.append(re.escape(val))
    return re.compile(r"\b(?:%s)\b" % "|".join(tokens), re.I)


HOST_RE = host_re()
IP_RE = re.compile(r"\b(?:(?:\d{1,3}\.){3}\d{1,3})\b")
SECRET_RE = re.compile(
    r"\b(?:hf_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9]{20,}|ghp_[A-Za-z0-9]{20,}"
    r"|glpat-[A-Za-z0-9_\-]{20,}|AKIA[A-Z0-9]{16})\b"
)
RUN_RE = re.compile(r"631cb361")
WEIGHTS_RE = re.compile(r"(?:model\.safetensors|pytorch_model\.bin|tokenizer\.model)", re.I)
OK_IPS = {"127.0.0.1", "0.0.0.0"}


def load_token() -> str:
    import os

    for key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_HUB_TOKEN"):
        val = os.environ.get(key)
        if val:
            return val.strip()
    if ENV_PATH.exists():
        for line in ENV_PATH.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("HF_TOKEN="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise SystemExit("HF token not found in the environment or the repo .env file.")


def extract_text(pdf: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(pdf))
    parts: list[str] = []
    if reader.metadata:
        parts.extend(str(v) for v in dict(reader.metadata).values() if v)
    for page in reader.pages:
        parts.append(page.extract_text() or "")
    raw = pdf.read_bytes()
    strs = [s.decode("ascii") for s in re.findall(rb"[\x20-\x7e]{12,}", raw)]
    parts.append("\n".join(strs))
    return "\n".join(parts)


def scan_pdf(pdf: Path) -> None:
    if not pdf.is_file():
        raise SystemExit(f"PDF not found: {pdf}")
    if pdf.read_bytes()[:5] != b"%PDF-":
        raise SystemExit(f"Not a PDF: {pdf}")
    text = extract_text(pdf)
    hits: list[str] = []
    for rx, cat in (
        (EMAIL_RE, "email"),
        (PATH_RE, "home_path"),
        (HOST_RE, "hostname"),
        (SECRET_RE, "secret"),
        (RUN_RE, "forbidden_run"),
        (WEIGHTS_RE, "weights_filename"),
    ):
        if rx.search(text):
            hits.append(cat)
    for m in IP_RE.finditer(text):
        ip = m.group(0)
        try:
            obj = ipaddress.ip_address(ip)
        except ValueError:
            continue
        if ip in OK_IPS or obj.is_loopback:
            continue
        if not (obj.is_private or obj.is_link_local or obj.is_reserved or obj.is_multicast):
            hits.append("public_ip")
            break
    if hits:
        raise SystemExit(
            "PII/secrets scan failed; upload aborted. Categories: "
            + ", ".join(sorted(set(hits)))
            + ". Values are not printed."
        )
    print(f"PII/secrets scan clean for {pdf} ({pdf.stat().st_size} bytes)")


def session_with_token(token: str):
    from huggingface_hub import get_session

    session = get_session()
    session.headers.update({"Authorization": f"Bearer {token}"})
    return session


def check_url(session, url: str, auth: bool) -> tuple[int, str | None, int, bool]:
    # For logged-out checks, use a bare client if possible; otherwise curl-less get
    # without the Authorization header.
    if auth:
        r = session.get(url, timeout=60, follow_redirects=True)
    else:
        from huggingface_hub import get_session as bare

        r = bare().get(url, timeout=60, follow_redirects=True)
    ctype = r.headers.get("Content-Type")
    return r.status_code, ctype, len(r.content), r.content[:4] == b"%PDF"


def upload(pdf: Path, token: str, also_dataset: bool, message: str) -> None:
    from huggingface_hub import CommitOperationAdd, HfApi

    api = HfApi(token=token)
    ops = [
        CommitOperationAdd(path_in_repo=SPACE_PATH, path_or_fileobj=str(pdf)),
    ]
    api.create_commit(
        repo_id=SPACE_ID,
        repo_type="space",
        operations=ops,
        commit_message=message,
    )
    print(f"Uploaded {pdf.name} to space {SPACE_ID}:{SPACE_PATH}")
    if also_dataset:
        api.create_commit(
            repo_id=DATASET_ID,
            repo_type="dataset",
            operations=[CommitOperationAdd(path_in_repo=DATASET_PATH, path_or_fileobj=str(pdf))],
            commit_message=message,
        )
        print(f"Uploaded {pdf.name} to dataset {DATASET_ID}:{DATASET_PATH}")
    if LOCAL_SPACE_COPY.parent.exists():
        LOCAL_SPACE_COPY.parent.mkdir(parents=True, exist_ok=True)
        LOCAL_SPACE_COPY.write_bytes(pdf.read_bytes())
        print(f"Copied to local Space tree {LOCAL_SPACE_COPY}")


def wait_running(token: str, timeout: int = 300) -> str:
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        info = api.get_space_runtime(SPACE_ID)
        stage = getattr(info, "stage", None) or "UNKNOWN"
        if stage != last:
            print(f"Space runtime stage={stage}")
            last = stage
        if stage == "RUNNING":
            return stage
        if stage in {"RUNTIME_ERROR", "BUILD_ERROR", "CONFIG_ERROR", "NO_APP_FILE"}:
            raise SystemExit(f"Space entered error stage {stage}")
        time.sleep(5)
    raise SystemExit(f"Timed out waiting for Space to reach RUNNING (last stage={last})")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", type=Path, default=DEFAULT_PDF, help="Path to the compiled paper PDF")
    parser.add_argument("--also-dataset", action="store_true", help="Also replace paper/main.pdf in the private eval dataset")
    parser.add_argument("--wait", action="store_true", help="Wait for the Space to be RUNNING and re-check the resolve URL")
    parser.add_argument("--skip-upload", action="store_true", help="Only scan and (if --wait) verify; do not upload")
    parser.add_argument("--message", default="Update Space paper PDF (current compiled draft).")
    args = parser.parse_args()
    pdf = args.pdf.expanduser().resolve()
    scan_pdf(pdf)
    token = load_token()
    if not args.skip_upload:
        upload(pdf, token, args.also_dataset, args.message)
    if args.wait:
        wait_running(token)
    session = session_with_token(token)
    code, ctype, size, is_pdf = check_url(session, PAPER_URL, auth=True)
    print(f"AUTH resolve: http={code} ctype={ctype} size={size} is_pdf={is_pdf}")
    code2, ctype2, size2, is_pdf2 = check_url(session, PAPER_URL, auth=False)
    print(f"LOGGED-OUT resolve: http={code2} ctype={ctype2} size={size2} is_pdf={is_pdf2}")
    print(f"Working URL (auth required while the Space is private): {PAPER_URL}")
    if code != 200 or not is_pdf:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
