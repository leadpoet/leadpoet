#!/usr/bin/env python3
"""Dry-run scaffold for Langfuse trace backfill from canonical records.

The first implementation intentionally reports what would be mirrored. Actual
backfill should only be enabled after staging redaction review.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research_lab.observability.langfuse_client import langfuse_enabled, langfuse_project  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--no-dry-run", action="store_true", help="Reserved; currently rejected")
    args = parser.parse_args()
    if args.no_dry_run:
        raise SystemExit("ERROR: Langfuse backfill writes are not enabled in v1; run staging trace export first")
    print(
        json.dumps(
            {
                "dry_run": True,
                "langfuse_enabled": langfuse_enabled(),
                "langfuse_project": langfuse_project(),
                "limit": max(1, args.limit),
                "status": "backfill_write_disabled",
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
