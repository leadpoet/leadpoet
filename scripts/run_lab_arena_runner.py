#!/usr/bin/env python3
"""Lab Arena runner entrypoint (labarena.md sections 9, 14.3).

A runner follows the Arena's current round (or one pinned with
``--round-id``), claims one assignment per free local slot
(``LAB_ARENA_MAX_PARALLEL_RUNS``, default 1), materializes the pinned image
from the Arena registry, executes it in a fresh gVisor sandbox, and submits
runner-signed receipts. Startup fails unless the host is Linux x86_64 with the
exact pinned runsc binary.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a Leadpoet Lab Arena runner")
    parser.add_argument("--api-base-url", required=False, default=os.environ.get("LAB_ARENA_API_BASE_URL", ""))
    parser.add_argument("--round-id", required=False, default=os.environ.get("LAB_ARENA_ROUND_ID", ""), help="pin one round; unset follows the Arena's current round across days")
    parser.add_argument("--wallet-name", default=os.environ.get("LAB_ARENA_WALLET_NAME", "default"))
    parser.add_argument("--hotkey-name", default=os.environ.get("LAB_ARENA_HOTKEY_NAME", "default"))
    parser.add_argument("--work-dir", default=os.environ.get("LAB_ARENA_RUNNER_WORK_DIR", "/var/lib/lab-arena/runner"))
    parser.add_argument("--runsc-path", default=os.environ.get("LAB_ARENA_RUNSC_PATH", "/usr/local/bin/runsc"))
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--once", action="store_true", help="claim until no work remains, then exit")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if not args.api_base_url:
        print("--api-base-url (or LAB_ARENA_API_BASE_URL) is required", file=sys.stderr)
        return 2
    from lab_arena.wiring import build_runner_from_environment  # lazy: bittensor wallet, runsc host checks

    runner = build_runner_from_environment(args)
    while True:
        taken = runner.run_once()
        if args.once and taken == 0:
            runner.close()
            return 0
        if taken == 0:
            time.sleep(max(5, int(args.poll_seconds)))


if __name__ == "__main__":
    raise SystemExit(main())
