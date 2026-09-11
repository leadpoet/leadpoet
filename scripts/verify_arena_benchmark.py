#!/usr/bin/env python3
"""Verify saved Day 1 commitments against the corresponding Day 2 reveal."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lab_arena import benchmark_commitment as bc


def load(path: Path) -> dict:
    with path.open("rb") as source:
        payload = source.read(bc.MAX_ARTIFACT_BYTES + 1)
    if len(payload) > bc.MAX_ARTIFACT_BYTES:
        raise ValueError("document too large")
    value = json.loads(payload)
    if not isinstance(value, dict):
        raise ValueError("document must be an object")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("commitment", type=Path, help="saved Day 1 benchmark-commitment response")
    parser.add_argument("reveal", type=Path, help="Day 2 benchmark response")
    args = parser.parse_args()
    try:
        digest = bc.verify_reveal(load(args.commitment), load(args.reveal))
    except (OSError, ValueError, TypeError, RecursionError, bc.BenchmarkCommitmentError):
        print("Benchmark verification failed: invalid or mismatched documents.", file=sys.stderr)
        return 1
    print("Verified all 20 ICPs against " + digest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
