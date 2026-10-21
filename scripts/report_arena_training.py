#!/usr/bin/env python3
"""Report capture gaps and candidate yield from existing private Arena exports.

No database writes, provider calls, scoring, or model execution. The report
covers only supplied files. Default selection is published production rounds.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main(argv=None):
    from lab_arena import training_readiness
    from scripts.prepare_arena_training_dataset import _example_paths, _examples
    from scripts.export_arena_training import write_private

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--examples", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True, help="new private JSON report; never overwritten")
    parser.add_argument("--include-nonproduction-rounds", action="store_true")
    args = parser.parse_args(argv)
    try:
        paths = list(_example_paths(args.examples))
        if not paths or sum(path.stat().st_size for path in paths) > 256 * 1024 * 1024:
            raise ValueError("input_batch_empty_or_too_large")
        report = training_readiness.build(_examples(paths), production_rounds_only=not args.include_nonproduction_rounds)
        write_private(args.output, report)
    except Exception as exc:
        # Neither private source values nor exception text belongs in logs.
        print("Arena training report failed: " + type(exc).__name__, file=sys.stderr)
        return 1
    print(json.dumps({"report_hash": report["report_hash"],
                      "distinct_verified_executions": report["distinct_verified_executions"],
                      "qualification_candidates": report["qualification"]["candidate_records"],
                      "agent_step_candidates": report["agent_steps"]["candidate_records"],
                      "training_ready": False}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
