"""Entrypoint for gateway-supervised Research Lab worker processes."""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
from pathlib import Path
import sys


GATEWAY_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_PARENT = GATEWAY_ROOT.parent
ATTESTED_RUNTIME = GATEWAY_ROOT / "_attested_runtime"
for path in (ATTESTED_RUNTIME, PACKAGE_PARENT):
    if not path.exists():
        continue
    while str(path) in sys.path:
        sys.path.remove(str(path))
    sys.path.insert(0, str(path))

# Opt-in, fail-closed error monitoring (docs/sentry_error_monitoring.md).
# Complete no-op unless the LEADPOET_SENTRY_* environment gate is satisfied.
try:
    from leadpoet_observability import init_sentry  # noqa: E402

    init_sentry(component="research-lab-worker")
except Exception as _sentry_exc:  # must never break the worker
    print(
        "leadpoet_sentry_wiring_skipped error=%s" % type(_sentry_exc).__name__,
        flush=True,
    )

from gateway.research_lab.config import ResearchLabGatewayConfig  # noqa: E402
from gateway.research_lab.logging_utils import format_worker_block  # noqa: E402
from gateway.research_lab.worker_autostart import (  # noqa: E402
    SCORING_PROXY_PREFIXES,
    build_research_lab_worker_environment,
)


WORKER_READY_FD_ENV = "RESEARCH_LAB_WORKER_READY_FD"


def _signal_parent_ready() -> None:
    value = os.environ.pop(WORKER_READY_FD_ENV, "").strip()
    if not value:
        return
    fd = int(value)
    try:
        os.write(fd, b"ready\n")
    finally:
        os.close(fd)


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
    return worker_id


def _print_scoring_banner(config: ResearchLabGatewayConfig, *, worker_id: str) -> None:
    baseline_owner = config.scoring_worker_index == 0
    print(
        format_worker_block(
            "RESEARCH LAB PUBLIC BASELINE WORKER",
            (
                ("Worker ID", worker_id),
                ("Worker index", f"{config.scoring_worker_index + 1}/{config.scoring_worker_total_workers}"),
                ("Poll seconds", config.scoring_worker_poll_seconds),
                ("Public baseline daily", config.public_baseline_rebenchmark_enabled),
                ("Baseline owner", baseline_owner),
            ),
        )
        + "\n",
        flush=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run one gateway-supervised daily public-baseline worker"
    )
    parser.add_argument("--kind", choices=("scoring",), required=True)
    parser.add_argument("--worker-index", type=int, required=True)
    parser.add_argument("--total-workers", type=int, required=True)
    parser.add_argument("--worker-prefix", default="")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    try:  # low-cardinality triage tag; safe no-op when Sentry is inactive
        from leadpoet_observability import set_sentry_tag

        set_sentry_tag("worker.kind", args.kind)
    except Exception:
        pass

    build_research_lab_worker_environment()
    _configure_logging(args.log_level)

    from gateway.research_lab.daily_worker import DailyPublicRebenchmarkWorker

    worker_id = _configure_scoring_worker(
        args.worker_index,
        args.total_workers,
        args.worker_prefix or os.getenv("RESEARCH_LAB_SCORING_WORKER_PREFIX", "research-lab-scorer"),
    )
    config = ResearchLabGatewayConfig.from_env()
    _print_scoring_banner(config, worker_id=worker_id)
    worker = DailyPublicRebenchmarkWorker(config, worker_ref=worker_id)
    _signal_parent_ready()
    asyncio.run(worker.run_forever())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
