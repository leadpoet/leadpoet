#!/usr/bin/env python3
"""Run the hosted Research Lab auto-research worker."""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gateway.research_lab.config import ResearchLabGatewayConfig  # noqa: E402
from gateway.research_lab.worker import ResearchLabHostedWorker  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the hosted Research Lab auto-research worker")
    parser.add_argument("--once", action="store_true", help="Process at most one queued run")
    parser.add_argument("--log-level", default="INFO", help="Python logging level")
    parser.add_argument("--worker-id", default="", help="Stable worker identifier for queue events")
    parser.add_argument("--worker-index", type=int, default=None, help="Zero-based worker index in the local fleet")
    parser.add_argument("--total-workers", type=int, default=None, help="Total worker processes in the local fleet")
    args = parser.parse_args()

    if args.worker_id:
        os.environ["RESEARCH_LAB_HOSTED_WORKER_ID"] = args.worker_id
    if args.worker_index is not None:
        os.environ["RESEARCH_LAB_HOSTED_WORKER_INDEX"] = str(args.worker_index)
    if args.total_workers is not None:
        os.environ["RESEARCH_LAB_HOSTED_WORKER_TOTAL_WORKERS"] = str(args.total_workers)

    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    worker = ResearchLabHostedWorker(ResearchLabGatewayConfig.from_env(), worker_ref=args.worker_id or None)
    if args.once:
        outcome = asyncio.run(worker.run_once())
        print(outcome.to_dict())
        return 0 if outcome.status not in {"failed"} else 1
    asyncio.run(worker.run_forever())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
