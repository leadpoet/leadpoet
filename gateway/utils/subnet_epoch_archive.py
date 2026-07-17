"""Permanent archive authority for historical subnet-epoch anchors.

Live best/finalized scheduler observations belong on the normal Finney
endpoint.  Cutover and predecessor state are immutable historical reads and
must use Bittensor's official archive node after lite-node pruning begins.
"""

from __future__ import annotations

import threading
from typing import Any

from Leadpoet.utils.subnet_epoch import (
    OFFICIAL_BITTENSOR_ARCHIVE_ENDPOINT,
    SubnetEpochCutover,
    SubnetEpochSnapshot,
    assert_official_archive_subtensor,
    read_subnet_epoch_snapshot,
    validate_subnet_epoch_cutover_anchor,
)

_archive_subtensor: Any = None
_archive_subtensor_lock = threading.Lock()
_archive_query_lock = threading.Lock()


def get_official_archive_subtensor() -> Any:
    """Return a dedicated client pinned to the official archive endpoint."""

    global _archive_subtensor
    with _archive_subtensor_lock:
        if _archive_subtensor is None:
            import bittensor as bt

            _archive_subtensor = bt.Subtensor(
                network=OFFICIAL_BITTENSOR_ARCHIVE_ENDPOINT
            )
        return _archive_subtensor


def validate_cutover_anchor_from_archive(
    cutover: SubnetEpochCutover,
    *,
    archive_subtensor: Any = None,
) -> None:
    """Validate one immutable cutover exclusively through the archive node."""

    source = (
        get_official_archive_subtensor()
        if archive_subtensor is None
        else archive_subtensor
    )
    with _archive_query_lock:
        validate_subnet_epoch_cutover_anchor(source, cutover)


def read_exact_subnet_epoch_snapshot_from_archive(
    *,
    netuid: int,
    block_number: int | None = None,
    block_hash: str | None = None,
    archive_subtensor: Any = None,
) -> SubnetEpochSnapshot:
    """Read one historical exact-hash scheduler snapshot from the archive."""

    if (block_number is None) == (block_hash is None):
        raise ValueError("set exactly one historical block number or hash")
    source = (
        get_official_archive_subtensor()
        if archive_subtensor is None
        else archive_subtensor
    )
    with _archive_query_lock:
        assert_official_archive_subtensor(source)
        return read_subnet_epoch_snapshot(
            source,
            netuid=int(netuid),
            block_number=block_number,
            block_hash=block_hash,
        )


def read_finalized_subnet_epoch_snapshot_from_archive(
    *,
    netuid: int,
    archive_subtensor: Any = None,
) -> SubnetEpochSnapshot:
    """Read the current finalized scheduler state from the archive client.

    This is used only to discover the most recent immutable scheduler anchor.
    The anchor itself is subsequently re-read by exact block number before it
    is accepted as a cutover manifest input.
    """

    source = (
        get_official_archive_subtensor()
        if archive_subtensor is None
        else archive_subtensor
    )
    with _archive_query_lock:
        assert_official_archive_subtensor(source)
        return read_subnet_epoch_snapshot(
            source,
            netuid=int(netuid),
            finalized=True,
        )
