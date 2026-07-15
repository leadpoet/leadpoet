#!/usr/bin/env python3
"""Run the hosted Research Lab auto-research worker."""

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
from gateway.research_lab.git_tree_models import TreePolicy  # noqa: E402
from gateway.research_lab.worker import ResearchLabHostedWorker  # noqa: E402


def _proxy_ref(proxy_url: str) -> str:
    if not proxy_url:
        return "none"
    return "sha256:" + hashlib.sha256(proxy_url.encode("utf-8")).hexdigest()[:16]


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, str(level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    for logger_name in ("httpx", "httpcore", "hpack", "botocore", "boto3", "urllib3"):
        logging.getLogger(logger_name).setLevel(logging.WARNING)


def _print_startup_banner(config: ResearchLabGatewayConfig, *, worker_id: str, once: bool) -> None:
    tree_policy = TreePolicy.from_env()
    print("\n" + "=" * 70, flush=True)
    print("Research Lab Hosted Auto-Research Worker", flush=True)
    print("=" * 70, flush=True)
    print(f"Worker ID       : {worker_id or config.hosted_worker_id or 'auto'}", flush=True)
    print(f"Worker index    : {config.hosted_worker_index + 1}/{config.hosted_worker_total_workers}", flush=True)
    print(f"Poll seconds    : {config.hosted_worker_poll_seconds}", flush=True)
    print(f"Run mode        : {'once' if once else 'continuous'}", flush=True)
    print(f"Dry run         : {config.hosted_worker_dry_run}", flush=True)
    print(f"Proxy required  : {config.hosted_worker_require_proxy}", flush=True)
    print(f"Proxy ref       : {_proxy_ref(config.hosted_worker_proxy_url)}", flush=True)
    print(f"Runtime target  : {config.auto_research_min_seconds}s-{config.auto_research_max_seconds}s", flush=True)
    print(f"Iterations      : {config.auto_research_min_iterations}-{config.auto_research_max_iterations}", flush=True)
    print(f"Tree mode       : {tree_policy.mode}", flush=True)
    print(f"Tree nodes      : {tree_policy.max_nodes}", flush=True)
    print(
        f"Branch / beam   : {tree_policy.branch_factor}/{tree_policy.beam_width}",
        flush=True,
    )
    print("=" * 70 + "\n", flush=True)


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

    _configure_logging(args.log_level)
    config = ResearchLabGatewayConfig.from_env()
    _print_startup_banner(config, worker_id=args.worker_id, once=args.once)
    worker = ResearchLabHostedWorker(config, worker_ref=args.worker_id or None)
    if args.once:
        outcome = asyncio.run(worker.run_once())
        print(outcome.to_dict())
        return 0 if outcome.status not in {"failed", "worker_proxy_required"} else 1
    asyncio.run(worker.run_forever())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
