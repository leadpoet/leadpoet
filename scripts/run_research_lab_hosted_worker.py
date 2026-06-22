#!/usr/bin/env python3
"""Run the hosted Research Lab auto-research worker."""

from __future__ import annotations

import argparse
import asyncio
import logging
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
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    worker = ResearchLabHostedWorker(ResearchLabGatewayConfig.from_env())
    if args.once:
        outcome = asyncio.run(worker.run_once())
        print(outcome.to_dict())
        return 0 if outcome.status not in {"failed"} else 1
    asyncio.run(worker.run_forever())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
