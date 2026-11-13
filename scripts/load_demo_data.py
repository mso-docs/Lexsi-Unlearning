#!/usr/bin/env python3
"""Load the SYNTHETIC demo written by make_demo_data.py into a throwaway local Evalboard.

    python3 scripts/make_demo_data.py --out /tmp/lexsi-demo
    python3 -m evalboard.server --db /tmp/lexsi-demo/workspace.sqlite3 --port 8766   # in the modular_dashboard repo
    python3 scripts/load_demo_data.py --dir /tmp/lexsi-demo --url http://127.0.0.1:8766

Safety: loopback only, refuses a workspace that already holds non-synthetic traces (so it can never
be pointed at the real project database by mistake), and only posts traces/artifacts marked synthetic.
Re-running is safe: Evalboard treats identical traces and artifacts as no-ops. Standard library only.
"""
import argparse
import json
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

# Overview and Improvements default to the primary metric; judge-agreement metrics stay visible but secondary.
LAYOUT = {
    "title": "DEMO — synthetic data (Lexsi unlearning)",
    "theme": "system",
    "components": ["overview", "scoreboard", "metrics", "traces", "improvements", "tools", "memory", "interp"],
    "visuals": True,
    "density": "comfortable",
    "corners": "rounded",
    "brand": {"name": "DEMO — synthetic data", "tagline": "Lexsi unlearning vs hiding (dummy data)", "badge": "SYNTHETIC"},
    "auto_structure": True,
    "metrics": [
        {"id": "forget-suppression", "label": "Forget suppression", "source": "score", "key": "forget_suppression", "aggregation": "mean", "format": "percent", "visible": True, "primary": True},
    ],
}


def call(url, path, body=None):
    data = None if body is None else json.dumps(body, allow_nan=False).encode()
    req = urllib.request.Request(url + path, data, {"Content-Type": "application/json"} if data else {})
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.load(resp)
    except urllib.error.HTTPError as err:
        raise SystemExit(f"{path} rejected ({err.code}): {err.read().decode()[:500]}")


def batches(items, max_bytes=6_000_000):
    batch, size = [], 0
    for item in items:
        n = len(json.dumps(item))
        if batch and size + n > max_bytes:
            yield batch
            batch, size = [], 0
        batch.append(item)
        size += n
    if batch:
        yield batch


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default="/tmp/lexsi-demo")
    ap.add_argument("--url", default="http://127.0.0.1:8766")
    args = ap.parse_args(argv)
    if urllib.parse.urlsplit(args.url).hostname not in ("127.0.0.1", "localhost", "::1"):
        raise SystemExit("Refusing non-loopback URL.")
    out = Path(args.dir)
    manifest = json.loads((out / "manifest.json").read_text())
    if not manifest.get("synthetic"):
        raise SystemExit("manifest.json is not marked synthetic; refusing.")
    existing = call(args.url, "/api/workspace")["traces"]
    if any(not t.get("synthetic") for t in existing):
        raise SystemExit("This Evalboard already holds non-synthetic traces; refusing to mix demo data into it.")
    traces, arts = [], []
    for run in manifest["runs"]:
        base = run["run_id"]
        traces += [json.loads(l) for l in (out / "traces" / f"{base}.jsonl").read_text().splitlines() if l.strip()]
        arts += [json.loads(p.read_text()) for p in sorted((out / "interp" / base).glob("*.json"))]
    if not all(a.get("synthetic") is True for a in arts) or not all(t.get("synthetic") is True for t in traces if "synthetic" in t):
        raise SystemExit("Found an artifact/trace not marked synthetic; refusing.")
    seen, unique = set(), []
    for t in traces:
        if t["id"] not in seen:
            seen.add(t["id"])
            unique.append(t)
    call(args.url, "/api/layout", LAYOUT)
    n = sum(call(args.url, "/api/traces", {"traces": b})["ingested"] for b in batches(unique))
    m = sum(call(args.url, "/api/interp", {"artifacts": b})["ingested"] for b in batches(arts))
    print(f"loaded {n} synthetic traces and {m} synthetic artifacts into {args.url}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
