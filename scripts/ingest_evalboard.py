#!/usr/bin/env python3
"""Push a finished run's traces and interpretability artifacts into a LOCAL Evalboard (Mac side).

    python3 scripts/ingest_evalboard.py --url http://127.0.0.1:8765 --run 20261112T220000Z-smoke--e191e41f
    python3 scripts/ingest_evalboard.py --db release/pages_workspace.sqlite3 --run <run_id>
    python3 scripts/ingest_evalboard.py --run <run_id> --dry-run        # validate/count only

Reads (after scripts/sync_from_spark.sh):
    ${OUTPUT_DIR:-artifacts}/traces/<run_id>.jsonl
    ${OUTPUT_DIR:-artifacts}/interp/<run_id>/*.json        (not index.jsonl)

After ingesting, the layout is topped up from configs/evalboard_layout.json (merged, never replacing
existing settings) so the Overview opens on the primary metric (forget_suppression). Use --no-layout to skip.

Only loopback URLs are accepted. Artifacts marked synthetic are refused. Re-running is safe:
Evalboard treats identical traces/artifacts as no-ops. Standard library only.
"""
import argparse
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def post(url, path, body):
    req = urllib.request.Request(url + path, json.dumps(body, allow_nan=False).encode(), {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.load(resp)
    except urllib.error.HTTPError as err:
        raise SystemExit(f"{path} rejected ({err.code}): {err.read().decode()[:500]}")


def get(url, path):
    with urllib.request.urlopen(url + path, timeout=60) as resp:
        return json.load(resp)


def load_evalboard():
    """Import evalboard.core from EVALBOARD_ROOT or a sibling modular_dashboard checkout."""
    import os
    import sys
    candidates = []
    env = (os.environ.get("EVALBOARD_ROOT") or "").strip()
    if env:
        candidates.append(Path(env))
    here = ROOT.parent
    candidates.extend(
        [
            here / "Code" / "modular_dashboard",
            here / "code" / "modular_dashboard",
            Path.home() / "Documents" / "Code" / "modular_dashboard",
            Path.home() / "Documents" / "code" / "modular_dashboard",
        ]
    )
    for c in candidates:
        if (c / "evalboard" / "core.py").is_file():
            sys.path.insert(0, str(c))
            from evalboard.core import Store  # noqa: WPS433
            return Store
    raise SystemExit(
        "Could not find the Evalboard toolkit. Set EVALBOARD_ROOT to the modular_dashboard checkout."
    )


def store_post_traces(store, traces):
    if not traces:
        return {"ingested": 0}
    n = store.ingest(traces)
    return {"ingested": n}


def store_post_interp(store, artifacts):
    if not artifacts:
        return {"ingested": 0}
    n = store.ingest_interp(artifacts)
    return {"ingested": n}


def merge_layout_metrics(layout, layout_file):
    """Merge metric definitions from layout_file into a layout dict. Returns (layout, changed)."""
    wanted = json.loads(Path(layout_file).read_text()).get("metrics") or []
    layout = dict(layout or {})
    metrics = list(layout.get("metrics") or [])
    have = {m.get("id"): i for i, m in enumerate(metrics)}
    changed = False
    for m in wanted:
        if m["id"] not in have:
            metrics.append(m)
            changed = True
        elif bool(m.get("primary")) != bool(metrics[have[m["id"]]].get("primary")):
            metrics[have[m["id"]]] = {**metrics[have[m["id"]]], "primary": bool(m.get("primary"))}
            changed = True
    layout = {**layout, "metrics": metrics}
    return layout, changed


def ensure_layout(url, layout_file):
    """Merge the metric definitions in layout_file into the workspace layout (by id). Returns a status string."""
    layout, changed = merge_layout_metrics(get(url, "/api/workspace").get("layout") or {}, layout_file)
    if not changed:
        return "layout already has the primary metric"
    saved = post(url, "/api/layout", layout)
    ok = any(m.get("primary") for m in saved.get("metrics", []))
    return "layout updated" + ("" if ok else " (server dropped the primary flag; restart it from the current modular_dashboard checkout)")


def ensure_layout_store(store, layout_file):
    snap = store.snapshot()
    layout, changed = merge_layout_metrics(snap.get("layout") or {}, layout_file)
    if not changed:
        return "layout already has the primary metric"
    saved = store.save_layout(layout)
    ok = any(m.get("primary") for m in saved.get("metrics", []))
    return "layout updated" + ("" if ok else " (primary flag not saved)")


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
    ap.add_argument("--run", required=True, help="base run id (file name of the traces JSONL)")
    ap.add_argument("--url", default="http://127.0.0.1:8765", help="loopback Evalboard URL (ignored when --db is set)")
    ap.add_argument("--db", default=None, help="Write to an Evalboard SQLite file instead of a running server")
    ap.add_argument("--output-dir", default=os.environ.get("OUTPUT_DIR") or "artifacts")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-layout", action="store_true", help="do not top up the Evalboard layout with the primary metric")
    ap.add_argument("--layout-file", default=str(ROOT / "configs" / "evalboard_layout.json"))
    args = ap.parse_args(argv)

    store = None
    if args.db:
        Store = load_evalboard()
        store = Store(Path(args.db))
    else:
        host = urllib.parse.urlsplit(args.url).hostname
        if host not in ("127.0.0.1", "localhost", "::1"):
            raise SystemExit("Refusing non-loopback URL: traces stay on this machine.")
    out = Path(args.output_dir)
    if not out.is_absolute():
        out = ROOT / out
    traces_file = out / "traces" / f"{args.run}.jsonl"
    interp_dir = out / "interp" / args.run
    traces = [json.loads(l) for l in traces_file.read_text().splitlines() if l.strip()] if traces_file.exists() else []
    seen, unique = set(), []
    for t in traces:
        if t["id"] not in seen:  # append-only JSONL may repeat a stage trace; identical repeats are dropped
            seen.add(t["id"])
            unique.append(t)
    arts = [json.loads(p.read_text()) for p in sorted(interp_dir.glob("*.json"))] if interp_dir.exists() else []
    if any(a.get("synthetic") for a in arts):
        raise SystemExit("Refusing to ingest synthetic artifacts into a real project.")
    print(f"{len(unique)} traces ({len(traces) - len(unique)} duplicate ids dropped), {len(arts)} interpretability artifacts")
    if args.dry_run:
        return 0
    if store is not None:
        for b in batches(unique):
            print("traces:", store_post_traces(store, b))
        for b in batches(arts):
            print("interp:", store_post_interp(store, b))
        if not args.no_layout and Path(args.layout_file).exists():
            try:
                print("layout:", ensure_layout_store(store, args.layout_file))
            except Exception as exc:
                print("layout: skipped:", type(exc).__name__, exc)
        return 0
    for b in batches(unique):
        print("traces:", post(args.url, "/api/traces", {"traces": b}))
    for b in batches(arts):
        print("interp:", post(args.url, "/api/interp", {"artifacts": b}))
    if not args.no_layout and Path(args.layout_file).exists():
        try:
            print("layout:", ensure_layout(args.url, args.layout_file))
        except Exception as exc:  # the layout is cosmetic; never fail an ingest over it
            print("layout: skipped:", type(exc).__name__, exc)
    return 0


if __name__ == "__main__":
    sys.exit(main())
