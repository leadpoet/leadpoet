#!/usr/bin/env python3
"""Run the gateway-owned daily public-baseline worker."""

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

# Opt-in, fail-closed error monitoring (docs/sentry_error_monitoring.md).
# Complete no-op unless the LEADPOET_SENTRY_* environment gate is satisfied.
try:
    from leadpoet_observability import init_sentry  # noqa: E402

    init_sentry(component="research-lab-public-baseline-worker")
except Exception as _sentry_exc:  # must never break the worker
    print(
        "leadpoet_sentry_wiring_skipped error=%s" % type(_sentry_exc).__name__,
        flush=True,
    )

from gateway.research_lab.config import ResearchLabGatewayConfig  # noqa: E402
from gateway.research_lab.daily_worker import DailyPublicRebenchmarkWorker  # noqa: E402
from gateway.research_lab.worker_autostart import (  # noqa: E402
    build_research_lab_worker_environment,
)


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, str(level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    for logger_name in ("httpx", "httpcore", "hpack", "botocore", "boto3", "urllib3"):
        logging.getLogger(logger_name).setLevel(logging.WARNING)


def _print_startup_banner(config: ResearchLabGatewayConfig, *, worker_id: str, once: bool) -> None:
    print("\n" + "=" * 70, flush=True)
    print("Research Lab Daily Public Baseline Worker", flush=True)
    print("=" * 70, flush=True)
    print(f"Worker ID       : {worker_id or config.scoring_worker_id or 'auto'}", flush=True)
    print(f"Worker index    : {config.scoring_worker_index + 1}/{config.scoring_worker_total_workers}", flush=True)
    print(f"Poll seconds    : {config.scoring_worker_poll_seconds}", flush=True)
    print(f"Run mode        : {'once' if once else 'continuous'}", flush=True)
    print(f"Public baseline : {config.public_baseline_rebenchmark_enabled}", flush=True)
    print("=" * 70 + "\n", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the daily public baseline")
    parser.add_argument("--once", action="store_true", help="Process one daily pass")
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

    build_research_lab_worker_environment()
    _configure_logging(args.log_level)
    config = ResearchLabGatewayConfig.from_env()
    _print_startup_banner(config, worker_id=args.worker_id, once=args.once)
    worker = DailyPublicRebenchmarkWorker(config, worker_ref=args.worker_id or None)
    if args.once:
        outcome = asyncio.run(worker.run_once())
        print(outcome)
        return 1 if outcome.get("status") == "failed" else 0
    asyncio.run(worker.run_forever())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
