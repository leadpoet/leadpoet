"""Fail-closed production restart gate for Subnet 71."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Optional, Sequence

from Leadpoet.utils.subnet_epoch import (
    OFFICIAL_BITTENSOR_ARCHIVE_ENDPOINT,
    SubnetEpochError,
    SubnetEpochSnapshot,
    read_subnet_epoch_snapshot,
)


MAXIMUM_RESTART_EPOCH_BLOCK = 315
RESTART_START_SCHEMA_VERSION = "leadpoet.restart_epoch_start.v1"


class RestartEpochGateError(RuntimeError):
    """Raised when a production restart would begin too late in the epoch."""


def verify_restart_epoch_window(
    subtensor: Any,
    *,
    netuid: int = 71,
) -> dict[str, Any]:
    """Read the official scheduler and reject restart starts after block 315."""

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


def capture_restart_epoch_start(
    subtensor: Any,
    *,
    netuid: int = 71,
) -> dict[str, Any]:
    """Capture the single official restart-start decision for this invocation."""

    gate = verify_restart_epoch_window(subtensor, netuid=netuid)
    return {
        "schema_version": RESTART_START_SCHEMA_VERSION,
        "maximum_restart_epoch_block": MAXIMUM_RESTART_EPOCH_BLOCK,
        "restart_allowed": True,
        "snapshot": gate["snapshot"],
    }


def write_restart_epoch_start(path: Path, report: dict[str, Any]) -> None:
    """Atomically persist one private restart-start report."""

    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(report, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        temporary.chmod(0o600)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def load_restart_epoch_start(path: Path, *, netuid: int = 71) -> SubnetEpochSnapshot:
    """Load and strictly validate one captured restart-start snapshot."""

    try:
        report = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RestartEpochGateError("captured restart start is unavailable or invalid") from exc
    if not isinstance(report, dict):
        raise RestartEpochGateError("captured restart start must be one object")
    if report.get("schema_version") != RESTART_START_SCHEMA_VERSION:
        raise RestartEpochGateError("captured restart start schema is unsupported")
    captured_deadline = report.get("maximum_restart_epoch_block")
    if (
        not isinstance(captured_deadline, int)
        or isinstance(captured_deadline, bool)
        or not 0 <= captured_deadline <= MAXIMUM_RESTART_EPOCH_BLOCK
    ):
        raise RestartEpochGateError("captured restart start uses another deadline")
    if report.get("restart_allowed") is not True:
        raise RestartEpochGateError("captured restart start was not approved")
    snapshot_doc = report.get("snapshot")
    if not isinstance(snapshot_doc, dict):
        raise RestartEpochGateError("captured restart snapshot is missing")
    try:
        snapshot = SubnetEpochSnapshot.from_mapping(snapshot_doc)
    except Exception as exc:
        raise RestartEpochGateError("captured restart snapshot is invalid") from exc
    if snapshot.netuid != int(netuid):
        raise RestartEpochGateError("captured restart snapshot uses another netuid")
    if not 0 <= snapshot.epoch_block <= captured_deadline:
        raise RestartEpochGateError("captured restart start is outside its deadline")
    return snapshot


def _read_captured_snapshot(
    subtensor: Any,
    *,
    captured: SubnetEpochSnapshot,
) -> SubnetEpochSnapshot:
    try:
        return read_subnet_epoch_snapshot(
            subtensor,
            netuid=captured.netuid,
            block_hash=captured.block_hash,
        )
    except SubnetEpochError:
        pass

    import bittensor as bt

    archive = bt.Subtensor(network=OFFICIAL_BITTENSOR_ARCHIVE_ENDPOINT)
    try:
        return read_subnet_epoch_snapshot(
            archive,
            netuid=captured.netuid,
            block_hash=captured.block_hash,
        )
    except Exception as exc:
        raise RestartEpochGateError(
            "captured restart snapshot is unavailable from the official archive"
        ) from exc
    finally:
        close = getattr(archive, "close", None)
        if callable(close):
            close()


def verify_captured_restart_epoch_start(
    subtensor: Any,
    *,
    path: Path,
    netuid: int = 71,
) -> dict[str, Any]:
    """Verify one earlier valid start without reapplying the position deadline."""

    captured = load_restart_epoch_start(path, netuid=netuid)
    exact = _read_captured_snapshot(subtensor, captured=captured)
    authority_fields = (
        "network_genesis_hash",
        "netuid",
        "block_hash",
        "current_block",
        "last_epoch_block",
        "pending_epoch_at",
        "subnet_epoch_index",
        "tempo",
        "blocks_since_last_step",
    )
    if any(
        getattr(exact, field) != getattr(captured, field)
        for field in authority_fields
    ):
        raise RestartEpochGateError("captured restart snapshot differs from chain history")

    current = read_subnet_epoch_snapshot(subtensor, netuid=netuid)
    if (
        current.network_genesis_hash != captured.network_genesis_hash
        or current.netuid != captured.netuid
        or current.current_block < captured.current_block
    ):
        raise RestartEpochGateError(
            "captured restart start is not in the current official chain history"
        )
    return {
        "schema_version": RESTART_START_SCHEMA_VERSION,
        "maximum_restart_epoch_block": MAXIMUM_RESTART_EPOCH_BLOCK,
        "restart_allowed": True,
        "captured_epoch_block": captured.epoch_block,
        "current_epoch_block": current.epoch_block,
        "deadline_reapplied": False,
        "captured_snapshot": captured.to_dict(),
        "current_snapshot": current.to_dict(),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Verify that a gateway or validator restart may begin in the "
            "current official subnet epoch."
        )
    )
    parser.add_argument("--network", default="finney")
    parser.add_argument("--netuid", type=int, default=71)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--capture-output", type=Path)
    mode.add_argument("--captured-report", type=Path)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)

    import bittensor as bt

    subtensor = bt.Subtensor(network=args.network)
    try:
        if args.captured_report is not None:
            result = verify_captured_restart_epoch_start(
                subtensor,
                path=args.captured_report,
                netuid=args.netuid,
            )
        elif args.capture_output is not None:
            result = capture_restart_epoch_start(subtensor, netuid=args.netuid)
            write_restart_epoch_start(args.capture_output, result)
        else:
            result = verify_restart_epoch_window(subtensor, netuid=args.netuid)
    finally:
        close = getattr(subtensor, "close", None)
        if callable(close):
            close()
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
