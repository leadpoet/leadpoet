#!/usr/bin/env python3
"""Run a gateway-local fleet of hosted Research Lab workers."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
WORKER_SCRIPT = ROOT / "scripts" / "run_research_lab_hosted_worker.py"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gateway.research_lab.worker_autostart import (  # noqa: E402
    build_research_lab_worker_environment,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run multiple hosted Research Lab worker processes")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--worker-prefix", default=os.getenv("RESEARCH_LAB_HOSTED_WORKER_PREFIX", "research-lab-worker"))
    parser.add_argument("--log-level", default=os.getenv("RESEARCH_LAB_HOSTED_WORKER_LOG_LEVEL", "INFO"))
    parser.add_argument("--once", action="store_true", help="Start each worker for one run_once attempt, then exit")
    args = parser.parse_args()

    configured_count = int(os.getenv("RESEARCH_LAB_HOSTED_WORKER_PROCESS_COUNT", "0") or "0")
    worker_count = max(1, args.workers or configured_count or _configured_proxy_count() or 1)
    worker_environment = build_research_lab_worker_environment()
    children: dict[int, subprocess.Popen[bytes]] = {}
    stopping = False

    def start_worker(index: int) -> subprocess.Popen[bytes]:
        env = worker_environment.copy()
        env["RESEARCH_LAB_HOSTED_WORKER_TOTAL_WORKERS"] = str(worker_count)
        env["RESEARCH_LAB_HOSTED_WORKER_INDEX"] = str(index)
        env["RESEARCH_LAB_HOSTED_WORKER_ID"] = f"{args.worker_prefix}-{index + 1}"
        proxy = _proxy_for_worker(index)
        if proxy:
            env["RESEARCH_LAB_HOSTED_WORKER_PROXY"] = proxy
            env["HTTP_PROXY"] = proxy
            env["HTTPS_PROXY"] = proxy
            env["http_proxy"] = proxy
            env["https_proxy"] = proxy
        command = [
            sys.executable,
            str(WORKER_SCRIPT),
            "--log-level",
            args.log_level,
            "--worker-id",
            env["RESEARCH_LAB_HOSTED_WORKER_ID"],
            "--worker-index",
            str(index),
            "--total-workers",
            str(worker_count),
        ]
        if args.once:
            command.append("--once")
        return subprocess.Popen(command, cwd=str(ROOT), env=env)

    def stop_children(*_args: object) -> None:
        nonlocal stopping
        stopping = True
        for child in children.values():
            if child.poll() is None:
                child.terminate()

    signal.signal(signal.SIGTERM, stop_children)
    signal.signal(signal.SIGINT, stop_children)

    for index in range(worker_count):
        children[index] = start_worker(index)

    exit_code = 0
    try:
        while children:
            for index, child in list(children.items()):
                code = child.poll()
                if code is None:
                    continue
                if code != 0:
                    exit_code = code
                if args.once or stopping:
                    children.pop(index)
                    continue
                children[index] = start_worker(index)
            if stopping:
                for index, child in list(children.items()):
                    if child.poll() is not None:
                        children.pop(index)
                if children:
                    time.sleep(0.5)
                    continue
                break
            time.sleep(1)
    finally:
        stop_children()
        deadline = time.time() + 10
        for child in children.values():
            while child.poll() is None and time.time() < deadline:
                time.sleep(0.2)
            if child.poll() is None:
                child.kill()
    return exit_code


def _proxy_for_worker(index: int) -> str:
    one_based = index + 1
    return (
        os.getenv(f"RESEARCH_LAB_AUTO_RESEARCH_WEBSHARE_PROXY_{one_based}")
        or os.getenv(f"RESEARCH_LAB_WORKER_PROXY_{one_based}")
        or os.getenv(f"RESEARCH_LAB_WORKER_HTTPS_PROXY_{one_based}")
        or os.getenv("RESEARCH_LAB_AUTO_RESEARCH_WEBSHARE_PROXY")
        or os.getenv("RESEARCH_LAB_WORKER_PROXY")
        or ""
    ).strip()


def _configured_proxy_count() -> int:
    proxies = set()
    for index in range(1, 501):
        proxy = (
            os.getenv(f"RESEARCH_LAB_AUTO_RESEARCH_WEBSHARE_PROXY_{index}")
            or os.getenv(f"RESEARCH_LAB_WORKER_PROXY_{index}")
            or os.getenv(f"RESEARCH_LAB_WORKER_HTTPS_PROXY_{index}")
            or ""
        ).strip()
        if proxy:
            proxies.add(proxy)
    return len(proxies)


if __name__ == "__main__":
    raise SystemExit(main())
