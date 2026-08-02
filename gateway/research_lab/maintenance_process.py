"""Short-lived entrypoint for memory-intensive Research Lab maintenance.

The hosted worker keeps the distributed maintenance lease and supervises this
process.  The process boundary is intentional: large PostgREST JSON decodes
are returned to the operating system after each sweep instead of remaining in
the long-lived worker allocator.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
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

from gateway.research_lab.config import ResearchLabGatewayConfig  # noqa: E402
from gateway.research_lab.public_activity import (  # noqa: E402
    reproject_stale_public_cards,
)


logger = logging.getLogger(__name__)


async def _run_public_reprojection() -> None:
    result = await reproject_stale_public_cards(
        config=ResearchLabGatewayConfig.from_env()
    )
    logger.info(
        "research_lab_public_reprojection_process_complete "
        "enabled=%s checked=%s stale=%s reprojected=%s deferred=%s failed=%s",
        result.get("enabled"),
        result.get("cards_checked"),
        result.get("stale_found"),
        result.get("reprojected"),
        result.get("deferred_to_next_sweep"),
        len(result.get("failed") or ()),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run one bounded Research Lab maintenance task"
    )
    parser.add_argument(
        "--task", choices=("public-reprojection",), required=True
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    if args.task == "public-reprojection":
        asyncio.run(_run_public_reprojection())
        return 0
    raise AssertionError("argparse accepted an unknown maintenance task")


if __name__ == "__main__":
    raise SystemExit(main())
