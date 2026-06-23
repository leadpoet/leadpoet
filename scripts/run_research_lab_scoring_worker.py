#!/usr/bin/env python3
"""Run one gateway-owned Research Lab qualification scoring worker."""

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
from gateway.research_lab.scoring_worker import ResearchLabGatewayScoringWorker  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a Research Lab gateway scoring worker")
    parser.add_argument("--once", action="store_true", help="Process at most one scoring pass")
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--worker-id", default="")
    parser.add_argument("--worker-index", type=int, default=None)
    parser.add_argument("--total-workers", type=int, default=None)
    args = parser.parse_args()

    if args.worker_id:
        os.environ["RESEARCH_LAB_SCORING_WORKER_ID"] = args.worker_id
    if args.worker_index is not None:
        os.environ["RESEARCH_LAB_SCORING_WORKER_INDEX"] = str(args.worker_index)
    if args.total_workers is not None:
        os.environ["RESEARCH_LAB_SCORING_WORKER_TOTAL_WORKERS"] = str(args.total_workers)

    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    worker = ResearchLabGatewayScoringWorker(ResearchLabGatewayConfig.from_env(), worker_ref=args.worker_id or None)
    if args.once:
        outcome = asyncio.run(worker.run_once())
        print(outcome)
        return 0 if outcome.get("status") not in {"scoring_worker_proxy_required"} else 1
    asyncio.run(worker.run_forever())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
