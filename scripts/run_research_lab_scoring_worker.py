#!/usr/bin/env python3
"""Run one gateway-owned Research Lab qualification scoring worker."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import logging
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gateway.research_lab.config import ResearchLabGatewayConfig  # noqa: E402
from gateway.research_lab.scoring_worker import ResearchLabGatewayScoringWorker  # noqa: E402


def _proxy_ref(proxy_url: str) -> str:
    if not proxy_url:
        return "none"
    return "sha256:" + hashlib.sha256(proxy_url.encode("utf-8")).hexdigest()[:16]


def _print_startup_banner(config: ResearchLabGatewayConfig, *, worker_id: str, once: bool) -> None:
    print("\n" + "=" * 70, flush=True)
    print("Research Lab Gateway Qualification Scoring Worker", flush=True)
    print("=" * 70, flush=True)
    print(f"Worker ID       : {worker_id or config.scoring_worker_id or 'auto'}", flush=True)
    print(f"Worker index    : {config.scoring_worker_index + 1}/{config.scoring_worker_total_workers}", flush=True)
    print(f"Poll seconds    : {config.scoring_worker_poll_seconds}", flush=True)
    print(f"Run mode        : {'once' if once else 'continuous'}", flush=True)
    print(f"Proxy required  : {config.scoring_worker_require_proxy}", flush=True)
    print(f"Proxy ref       : {_proxy_ref(config.scoring_worker_proxy_url)}", flush=True)
    print(f"Baseline daily  : {config.private_baseline_rebenchmark_enabled}", flush=True)
    print(f"Candidate batch : {config.scoring_worker_max_candidates}", flush=True)
    print(f"Model timeout   : {config.scoring_worker_model_timeout_seconds}s", flush=True)
    print("=" * 70 + "\n", flush=True)


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
    config = ResearchLabGatewayConfig.from_env()
    _print_startup_banner(config, worker_id=args.worker_id, once=args.once)
    worker = ResearchLabGatewayScoringWorker(config, worker_ref=args.worker_id or None)
    if args.once:
        outcome = asyncio.run(worker.run_once())
        print(outcome)
        return 0 if outcome.get("status") not in {"scoring_worker_proxy_required"} else 1
    asyncio.run(worker.run_forever())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
