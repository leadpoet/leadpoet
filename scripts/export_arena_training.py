#!/usr/bin/env python3
"""Export one Arena execution with explicit proof gaps, using read-only access."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from lab_arena.training_reader import _ledger, export_run


def write_private(path: Path, document: dict) -> None:
    """Create a new mode-0600 file; do not overwrite prior artifacts or links."""
    from lab_arena.contracts import canonical_json

    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(path, flags, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(canonical_json(document) + "\n")
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--environment-file", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if not re.fullmatch(r"[A-Za-z0-9._:-]{1,200}", args.run_id):
        parser.error("invalid run ID")
    store = None
    try:
        if args.environment_file:
            from scripts.run_lab_arena_service import load_scoped_environment
            load_scoped_environment(args.environment_file)
        from lab_arena.service import S3ObjectStore
        from lab_arena.store import ArenaStore, PostgrestTransport
        # No service construction: it can connect chain clients and start work.
        store = ArenaStore(PostgrestTransport(os.environ["LAB_ARENA_SUPABASE_URL"],
                                               service_key=os.environ["LAB_ARENA_SERVICE_KEY"]))
        objects = S3ObjectStore(os.environ["LAB_ARENA_BUCKET"], region_name=os.environ.get("AWS_REGION"),
                                prefix=os.environ.get("LAB_ARENA_OBJECT_PREFIX", ""))
        result = export_run(store, objects, args.run_id)
        write_private(args.output, result)
        print(json.dumps({"complete": result["completeness"]["complete"],
                          "dimensions": result["completeness"]["dimensions"],
                          "incomplete_reasons": result["completeness"]["incomplete_reasons"],
                          "artifact": str(args.output)}, sort_keys=True))
        return 0 if result["completeness"]["complete"] else 2
    except Exception as exc:
        # Never print a database/object-store exception: it can contain endpoints
        # or request data. Fixed error type is sufficient for this CLI boundary.
        print("Arena training export failed: " + type(exc).__name__, file=sys.stderr)
        return 1
    finally:
        if store is not None:
            store.close()


if __name__ == "__main__":
    raise SystemExit(main())
