"""Entrypoint for gateway-supervised Research Lab worker processes."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import logging
import os
from pathlib import Path
import sys


PACKAGE_PARENT = Path(__file__).resolve().parents[2]
if str(PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PARENT))

from gateway.research_lab.config import ResearchLabGatewayConfig  # noqa: E402
from gateway.research_lab.scoring_worker import ResearchLabGatewayScoringWorker  # noqa: E402
from gateway.research_lab.worker import ResearchLabHostedWorker  # noqa: E402


HOSTED_PROXY_PREFIXES = (
    "RESEARCH_LAB_AUTO_RESEARCH_WEBSHARE_PROXY",
    "RESEARCH_LAB_WORKER_PROXY",
    "RESEARCH_LAB_WORKER_HTTPS_PROXY",
)
SCORING_PROXY_PREFIXES = (
    "RESEARCH_LAB_QUALIFICATION_WEBSHARE_PROXY",
    "QUALIFICATION_WEBSHARE_PROXY",
    "RESEARCH_LAB_SCORING_WORKER_PROXY",
)


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


def _proxy_for_worker(prefixes: tuple[str, ...], index: int) -> str:
    one_based = index + 1
    for prefix in prefixes:
        value = os.getenv(f"{prefix}_{one_based}", "").strip()
        if value:
            return value
    for prefix in prefixes:
        value = os.getenv(prefix, "").strip()
        if value:
            return value
    return ""


def _apply_proxy_env(proxy_url: str) -> None:
    if not proxy_url:
        return
    os.environ["HTTP_PROXY"] = proxy_url
    os.environ["HTTPS_PROXY"] = proxy_url
    os.environ["http_proxy"] = proxy_url
    os.environ["https_proxy"] = proxy_url


def _configure_hosted_worker(index: int, total_workers: int, worker_prefix: str) -> str:
    worker_id = f"{worker_prefix}-{index + 1}"
    proxy = os.getenv("RESEARCH_LAB_HOSTED_WORKER_PROXY", "").strip() or _proxy_for_worker(
        HOSTED_PROXY_PREFIXES,
        index,
    )
    os.environ.setdefault("RESEARCH_LAB_HOSTED_WORKER_ENABLED", "true")
    os.environ["RESEARCH_LAB_HOSTED_WORKER_ID"] = worker_id
    os.environ["RESEARCH_LAB_HOSTED_WORKER_INDEX"] = str(index)
    os.environ["RESEARCH_LAB_HOSTED_WORKER_TOTAL_WORKERS"] = str(total_workers)
    if proxy:
        os.environ["RESEARCH_LAB_HOSTED_WORKER_PROXY"] = proxy
        _apply_proxy_env(proxy)
    return worker_id


def _configure_scoring_worker(index: int, total_workers: int, worker_prefix: str) -> str:
    worker_id = f"{worker_prefix}-{index + 1}"
    proxy = os.getenv("RESEARCH_LAB_SCORING_WORKER_PROXY", "").strip() or _proxy_for_worker(
        SCORING_PROXY_PREFIXES,
        index,
    )
    os.environ.setdefault("RESEARCH_LAB_SCORING_WORKER_ENABLED", "true")
    os.environ["RESEARCH_LAB_SCORING_WORKER_ID"] = worker_id
    os.environ["RESEARCH_LAB_SCORING_WORKER_INDEX"] = str(index)
    os.environ["RESEARCH_LAB_SCORING_WORKER_TOTAL_WORKERS"] = str(total_workers)
    if proxy:
        os.environ["RESEARCH_LAB_SCORING_WORKER_PROXY"] = proxy
        _apply_proxy_env(proxy)
    return worker_id


def _print_hosted_banner(config: ResearchLabGatewayConfig, *, worker_id: str) -> None:
    print("\n" + "=" * 80, flush=True)
    print("Research Lab Auto-Research Worker", flush=True)
    print("=" * 80, flush=True)
    print(f"Worker ID       : {worker_id}", flush=True)
    print(f"Worker index    : {config.hosted_worker_index + 1}/{config.hosted_worker_total_workers}", flush=True)
    print(f"Poll seconds    : {config.hosted_worker_poll_seconds}", flush=True)
    print(f"Dry run         : {config.hosted_worker_dry_run}", flush=True)
    print(f"Proxy required  : {config.hosted_worker_require_proxy}", flush=True)
    print(f"Proxy ref       : {_proxy_ref(config.hosted_worker_proxy_url)}", flush=True)
    print(f"Runtime target  : {config.auto_research_min_seconds}s-{config.auto_research_max_seconds}s", flush=True)
    print(f"Iterations      : {config.auto_research_min_iterations}-{config.auto_research_max_iterations}", flush=True)
    print(f"Candidate limit : {config.hosted_worker_max_candidates}", flush=True)
    print("=" * 80 + "\n", flush=True)


def _print_scoring_banner(config: ResearchLabGatewayConfig, *, worker_id: str) -> None:
    baseline_owner = config.scoring_worker_index == 0
    print("\n" + "=" * 80, flush=True)
    print("Research Lab Qualification Scoring Worker", flush=True)
    print("=" * 80, flush=True)
    print(f"Worker ID       : {worker_id}", flush=True)
    print(f"Worker index    : {config.scoring_worker_index + 1}/{config.scoring_worker_total_workers}", flush=True)
    print(f"Poll seconds    : {config.scoring_worker_poll_seconds}", flush=True)
    print(f"Proxy required  : {config.scoring_worker_require_proxy}", flush=True)
    print(f"Proxy ref       : {_proxy_ref(config.scoring_worker_proxy_url)}", flush=True)
    print(f"Baseline daily  : {config.private_baseline_rebenchmark_enabled}", flush=True)
    print(f"Baseline owner  : {baseline_owner}", flush=True)
    print(f"Candidate batch : {config.scoring_worker_max_candidates}", flush=True)
    print(f"Model timeout   : {config.scoring_worker_model_timeout_seconds}s", flush=True)
    print("=" * 80 + "\n", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one gateway-supervised Research Lab worker")
    parser.add_argument("--kind", choices=("hosted", "scoring"), required=True)
    parser.add_argument("--worker-index", type=int, required=True)
    parser.add_argument("--total-workers", type=int, required=True)
    parser.add_argument("--worker-prefix", default="")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    _configure_logging(args.log_level)

    if args.kind == "hosted":
        worker_id = _configure_hosted_worker(
            args.worker_index,
            args.total_workers,
            args.worker_prefix or os.getenv("RESEARCH_LAB_HOSTED_WORKER_PREFIX", "research-lab-worker"),
        )
        config = ResearchLabGatewayConfig.from_env()
        _print_hosted_banner(config, worker_id=worker_id)
        asyncio.run(ResearchLabHostedWorker(config, worker_ref=worker_id).run_forever())
        return 0

    worker_id = _configure_scoring_worker(
        args.worker_index,
        args.total_workers,
        args.worker_prefix or os.getenv("RESEARCH_LAB_SCORING_WORKER_PREFIX", "research-lab-scorer"),
    )
    config = ResearchLabGatewayConfig.from_env()
    _print_scoring_banner(config, worker_id=worker_id)
    asyncio.run(ResearchLabGatewayScoringWorker(config, worker_ref=worker_id).run_forever())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
