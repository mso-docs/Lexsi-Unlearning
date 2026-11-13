#!/usr/bin/env bash
# Pre-publication secret and PII scan for this repo.
# Scans git history and the working tree with gitleaks, then scans docs,
# notebooks, paper/, presentation/, and HF-destined artifacts / traces /
# logs / datasets for PII. Writes a redacted report and exits non-zero
# on publication-blocking findings.
#
# Usage:
#   scripts/prepublish_scan.sh
#   scripts/prepublish_scan.sh --report artifacts/prepublish_report.md
#
# Preferred gitleaks install:  brew install gitleaks
# This script also accepts a gitleaks binary already on PATH (for example
# ~/.local/bin/gitleaks).
#
# Does not commit, push, upload, or rewrite git history.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

REPORT="artifacts/prepublish_report.md"
ALLOWLIST=".prepublish_allowlist"
IDENTITIES=".prepublish_identities"
CONFIG=".gitleaks.toml"

usage() {
  cat <<'EOF'
Usage: scripts/prepublish_scan.sh [--report PATH] [--allowlist PATH] [--identities PATH]

  --report PATH      Redacted markdown report (default: artifacts/prepublish_report.md)
  --allowlist PATH   Reviewed false-positive list (default: .prepublish_allowlist)
  --identities PATH  Gitignored extra usernames/hostnames (default: .prepublish_identities)
  -h, --help         Show this help

Exit codes:  0 clean  |  1 publication-blocking findings  |  2 scan could not finish
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --report) REPORT="${2:?}"; shift 2 ;;
    --allowlist) ALLOWLIST="${2:?}"; shift 2 ;;
    --identities) IDENTITIES="${2:?}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

# Resolve gitleaks, preferring PATH, then Homebrew prefixes, then ~/.local/bin.
export PATH="$HOME/.local/bin:/opt/homebrew/bin:/usr/local/bin:$PATH"
GITLEAKS="$(command -v gitleaks || true)"
if [[ -z "$GITLEAKS" ]]; then
  cat >&2 <<'EOF'
gitleaks is not on PATH. Install it with Homebrew, then re-run this script:

  brew install gitleaks

EOF
  if ! command -v brew >/dev/null 2>&1 && [[ ! -x /opt/homebrew/bin/brew && ! -x /usr/local/bin/brew ]]; then
    cat >&2 <<'EOF'
Homebrew is not installed on this machine. Installing it needs your password
and a terminal prompt: https://brew.sh

If you already downloaded the official gitleaks release, put the binary on
PATH (for example ~/.local/bin/gitleaks) and re-run.
EOF
  fi
  exit 2
fi

if [[ ! -f "$CONFIG" ]]; then
  echo "Missing $CONFIG at the repo root." >&2
  exit 2
fi

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 is required to write the redacted report." >&2
  exit 2
fi

mkdir -p "$(dirname "$REPORT")" artifacts
TMP="$(mktemp -d "${TMPDIR:-/tmp}/prepublish-scan.XXXXXX")"
trap 'rm -rf "$TMP"' EXIT
chmod 700 "$TMP"

GIT_JSON="$TMP/gitleaks-git.json"
DIR_JSON="$TMP/gitleaks-dir.json"
VERSION="$("$GITLEAKS" version 2>/dev/null | head -n 1 || echo unknown)"

# Use the current commands (gitleaks git / gitleaks dir). Fall back to the
# older `gitleaks detect` and `gitleaks detect --no-git` if needed.
if "$GITLEAKS" dir --help >/dev/null 2>&1; then
  GIT_SUB=(git)
  DIR_SUB=(dir)
else
  GIT_SUB=(detect)
  DIR_SUB=(detect --no-git)
fi

GIT_NOTE=""
GIT_OK=0
if git rev-parse --verify HEAD >/dev/null 2>&1; then
  set +e
  "$GITLEAKS" "${GIT_SUB[@]}" \
    --no-banner --no-color --redact \
    --exit-code 0 \
    -l warn \
    -c "$CONFIG" \
    -f json -r "$GIT_JSON" \
    "$ROOT"
  git_ec=$?
  set -e
  if [[ $git_ec -ne 0 ]]; then
    GIT_NOTE="gitleaks history scan failed (exit $git_ec). Treat this as an incomplete scan."
    GIT_OK=0
  else
    if [[ ! -s "$GIT_JSON" ]]; then
      echo '[]' > "$GIT_JSON"
    fi
    GIT_NOTE="gitleaks scanned the full git history. Any hits are listed below and would require a history rewrite (after rotating credentials) before going public."
    GIT_OK=1
  fi
else
  echo '[]' > "$GIT_JSON"
  GIT_NOTE="This repository has no commits yet, so there is no git history for gitleaks to scan. The history is clean by definition. Re-run after the first commit and before the public push."
  GIT_OK=1
fi

set +e
"$GITLEAKS" "${DIR_SUB[@]}" \
  --no-banner --no-color --redact \
  --exit-code 0 \
  -l warn \
  -c "$CONFIG" \
  -f json -r "$DIR_JSON" \
  "$ROOT"
dir_ec=$?
set -e
if [[ $dir_ec -ne 0 ]]; then
  echo "gitleaks working-tree scan failed (exit $dir_ec)." >&2
  exit 2
fi
if [[ ! -s "$DIR_JSON" ]]; then
  echo '[]' > "$DIR_JSON"
fi
chmod 600 "$GIT_JSON" "$DIR_JSON" 2>/dev/null || true

PY_ARGS=(
  python3 "$ROOT/scripts/prepublish_scan.py"
  --root "$ROOT"
  --report "$REPORT"
  --allowlist "$ALLOWLIST"
  --gitleaks-git "$GIT_JSON"
  --gitleaks-dir "$DIR_JSON"
  --gitleaks-version "$VERSION"
  --git-history-note "$GIT_NOTE"
)
if [[ $GIT_OK -eq 1 ]]; then
  PY_ARGS+=(--git-history-ok)
fi
if [[ -f "$IDENTITIES" ]]; then
  PY_ARGS+=(--identities "$IDENTITIES")
fi

set +e
"${PY_ARGS[@]}"
ec=$?
set -e
exit "$ec"
