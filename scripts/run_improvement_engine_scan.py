#!/usr/bin/env python3
"""Run the Research Lab Improvement Engine scanner."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research_lab.improvement_engine.config import ImprovementEngineConfig  # noqa: E402
from research_lab.improvement_engine.scanner import scan_for_issues  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", default=True, help="Do not write Engine issues")
    parser.add_argument("--persist", action="store_true", help="Persist detected issues/events")
    parser.add_argument("--lookback-hours", type=int, default=None)
    parser.add_argument("--min-cluster-size", type=int, default=None)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    config = ImprovementEngineConfig.from_env()
    if args.lookback_hours is not None or args.min_cluster_size is not None:
        config = ImprovementEngineConfig(
            **{
                **config.__dict__,
                **({"lookback_hours": args.lookback_hours} if args.lookback_hours is not None else {}),
                **({"min_cluster_size": args.min_cluster_size} if args.min_cluster_size is not None else {}),
            }
        )
    result = asyncio.run(scan_for_issues(config=config, dry_run=args.dry_run, persist=args.persist))
    text = json.dumps(result, indent=2, sort_keys=True, default=str)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
