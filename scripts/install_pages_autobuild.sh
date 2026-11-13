#!/usr/bin/env bash
# Install or refresh the Pages auto-builder LaunchAgent.
# Does not push. Does not start, stop, or otherwise touch Spark units/containers.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LIB="${HOME}/Library/Application Support/lexsi-pages"
PLIST="${HOME}/Library/LaunchAgents/com.mackenzie.lexsi-pages-autobuild.plist"
PY_FRAMEWORK="/Library/Frameworks/Python.framework/Versions/3.14/bin/python3"

mkdir -p "$LIB/bin" "$HOME/Library/LaunchAgents" "$HOME/Library/Logs"

# Refresh the logic copy used by launchd (TCC-safe location).
cp "$ROOT/scripts/pages_autobuild.py" "$LIB/pages_autobuild.py"
chmod +x "$LIB/pages_autobuild.py"

if [[ ! -x "$LIB/venv/bin/python3" ]]; then
  "$PY_FRAMEWORK" -m venv "$LIB/venv"
fi
"$LIB/venv/bin/pip" install -q 'dulwich>=0.22'

# Seed exclusions + hold already-finished runs (no build, no commit).
"$LIB/venv/bin/python3" "$LIB/pages_autobuild.py" --install-seed

# Self-test on dry data.
"$LIB/venv/bin/python3" "$LIB/pages_autobuild.py" --self-test

LABEL=gui/$(id -u)/com.mackenzie.lexsi-pages-autobuild
launchctl bootout "$LABEL" 2>/dev/null || true
launchctl bootstrap "gui/$(id -u)" "$PLIST"
echo "Loaded $LABEL"
echo "Disable with:"
echo "  launchctl bootout $LABEL"
echo "  touch \"$LIB/DISABLED\""
