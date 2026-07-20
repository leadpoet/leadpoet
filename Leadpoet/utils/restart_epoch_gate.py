"""Fail-closed production restart gate for Subnet 71."""

from __future__ import annotations

import argparse
import json
from typing import Any, Optional, Sequence

from Leadpoet.utils.subnet_epoch import read_subnet_epoch_snapshot


MAXIMUM_RESTART_EPOCH_BLOCK = 300


class RestartEpochGateError(RuntimeError):
    """Raised when a production restart would begin too late in the epoch."""


def verify_restart_epoch_window(
    subtensor: Any,
    *,
    netuid: int = 71,
) -> dict[str, Any]:
    """Read the official scheduler and reject restart starts after block 300."""

    snapshot = read_subnet_epoch_snapshot(subtensor, netuid=netuid)
    result = {
        "schema_version": "leadpoet.restart_epoch_gate.v1",
        "network_genesis_hash": snapshot.network_genesis_hash,
        "netuid": snapshot.netuid,
        "current_block": snapshot.current_block,
        "subnet_epoch_index": snapshot.subnet_epoch_index,
        "epoch_block": snapshot.epoch_block,
        "maximum_restart_epoch_block": MAXIMUM_RESTART_EPOCH_BLOCK,
        "restart_allowed": snapshot.epoch_block <= MAXIMUM_RESTART_EPOCH_BLOCK,
        "snapshot": snapshot.to_dict(),
    }
    if not result["restart_allowed"]:
        raise RestartEpochGateError(
            "production restart may start only at official subnet epoch block "
            f"{MAXIMUM_RESTART_EPOCH_BLOCK} or earlier; observed "
            f"{snapshot.epoch_block}"
        )
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Verify that a gateway or validator restart may begin in the "
            "current official subnet epoch."
        )
    )
    parser.add_argument("--network", default="finney")
    parser.add_argument("--netuid", type=int, default=71)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)

    import bittensor as bt

    subtensor = bt.Subtensor(network=args.network)
    try:
        result = verify_restart_epoch_window(subtensor, netuid=args.netuid)
    finally:
        close = getattr(subtensor, "close", None)
        if callable(close):
            close()
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
