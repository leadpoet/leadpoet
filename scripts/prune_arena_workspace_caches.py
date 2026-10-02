#!/usr/bin/env python3
"""Dry-run or reclaim stale reproducible caches in Arena test workspaces."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lab_arena.cache_retention import CacheRetentionError, prune_workspace_root


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace-root", type=Path, required=True)
    parser.add_argument("--active-work-dir", type=Path, action="append", required=True)
    parser.add_argument("--older-than-days", type=int, default=7)
    parser.add_argument("--apply", action="store_true", help="remove eligible cache entries; default is dry-run")
    args = parser.parse_args(argv)
    if sys.version_info < (3, 11) or not shutil.rmtree.avoids_symlink_attacks:
        print("Arena cache retention requires Python 3.11 with symlink-safe removal", file=sys.stderr)
        return 1
    try:
        rows = prune_workspace_root(
            args.workspace_root,
            active_workspaces=tuple(args.active_work_dir),
            older_than_days=args.older_than_days,
            apply=args.apply,
        )
    except CacheRetentionError as exc:
        print("Arena cache retention failed: %s" % exc, file=sys.stderr)
        return 1
    except OSError as exc:
        print("Arena cache retention failed: %s errno=%s" % (type(exc).__name__, exc.errno), file=sys.stderr)
        return 1
    for row in rows:
        print(json.dumps(row, sort_keys=True))
    print(json.dumps({
        "mode": "apply" if args.apply else "dry-run",
        "eligible_or_removed": sum(row.get("status") in ("eligible", "removed") for row in rows),
        "allocated_bytes": sum(row.get("bytes", 0) for row in rows if row.get("status") in ("eligible", "removed")),
        "locked_workspaces": sum(row.get("status") == "locked" for row in rows),
        "active_workspaces": sum(row.get("status") == "active_process" for row in rows),
        "recent_workspaces": sum(row.get("status") == "recent_workspace" for row in rows),
        "mounted_entries": sum(row.get("status") == "mounted" for row in rows),
        "mounted_workspaces": sum(row.get("status") == "mounted_workspace" for row in rows),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
