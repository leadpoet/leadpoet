"""Canonical first-frontier bootstrap from an attested allocation source.

The document is produced by the measured coordinator only after it has
verified the latest durable allocation result and the immutable reward
decisions represented by that result.  It lets a new release create the first
bounded settlement frontier without replaying the complete historical receipt
graph on the host.

This module is I/O-free and Python 3.7 compatible.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Mapping, Tuple

from leadpoet_canonical.allocation_settlement_frontier_v2 import (
    frontier_artifact_hashes_v2,
    validate_allocation_settlement_frontier_v2,
)
from leadpoet_canonical.attested_v2 import sha256_json


ALLOCATION_SETTLEMENT_FRONTIER_BOOTSTRAP_SCHEMA_VERSION = (
    "leadpoet.research_lab_allocation_settlement_frontier_bootstrap.v2"
)
ALLOCATION_SETTLEMENT_FRONTIER_BOOTSTRAP_REQUEST_SCHEMA_VERSION = (
    "leadpoet.research_lab_allocation_settlement_frontier_bootstrap_request.v2"
)
ALLOCATION_SETTLEMENT_FRONTIER_BOOTSTRAP_OPERATION = (
    "allocation_settlement_frontier_bootstrap_v2"
)
ALLOCATION_SETTLEMENT_FRONTIER_BOOTSTRAP_PURPOSE = (
    "research_lab.allocation_settlement_frontier_bootstrap.v2"
)

_HASH_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_BODY_FIELDS = frozenset(
    {
        "schema_version",
        "netuid",
        "bootstrap_epoch",
        "allocation_epoch",
        "allocation_source_receipt_hash",
        "source_state_hash",
        "frontier",
    }
)
_FIELDS = frozenset(set(_BODY_FIELDS) | {"bootstrap_hash"})


class AllocationSettlementFrontierBootstrapV2Error(ValueError):
    """The attested first-frontier bootstrap document is invalid."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AllocationSettlementFrontierBootstrapV2Error(message)


def _integer(value: Any, field: str, *, minimum: int = 0) -> int:
    _require(
        isinstance(value, int)
        and not isinstance(value, bool)
        and value >= minimum,
        "%s is invalid" % field,
    )
    return int(value)


def _hash(value: Any, field: str) -> str:
    raw = str(value or "")
    normalized = raw.strip().lower()
    _require(raw == normalized, "%s is not canonical" % field)
    _require(bool(_HASH_RE.fullmatch(normalized)), "%s is invalid" % field)
    return normalized


def build_allocation_settlement_frontier_bootstrap_v2(
    *,
    netuid: int,
    bootstrap_epoch: int,
    allocation_source_receipt_hash: str,
    source_state_hash: str,
    frontier: Mapping[str, Any],
) -> Dict[str, Any]:
    """Build one signed conversion of an allocation source into a frontier."""

    normalized_frontier = validate_allocation_settlement_frontier_v2(frontier)
    normalized_netuid = _integer(netuid, "bootstrap netuid", minimum=1)
    normalized_bootstrap_epoch = _integer(
        bootstrap_epoch,
        "bootstrap epoch",
        minimum=1,
    )
    allocation_epoch = int(normalized_frontier["allocation_epoch"])
    _require(
        normalized_frontier["mode"] == "legacy_full_history_bootstrap",
        "bootstrap frontier mode is invalid",
    )
    _require(
        normalized_frontier.get("predecessor_frontier_hash") is None,
        "bootstrap frontier declares a predecessor",
    )
    _require(
        int(normalized_frontier["netuid"]) == normalized_netuid,
        "bootstrap frontier netuid differs",
    )
    _require(
        allocation_epoch <= normalized_bootstrap_epoch,
        "bootstrap source is ahead of the execution epoch",
    )
    body = {
        "schema_version": (
            ALLOCATION_SETTLEMENT_FRONTIER_BOOTSTRAP_SCHEMA_VERSION
        ),
        "netuid": normalized_netuid,
        "bootstrap_epoch": normalized_bootstrap_epoch,
        "allocation_epoch": allocation_epoch,
        "allocation_source_receipt_hash": _hash(
            allocation_source_receipt_hash,
            "allocation source receipt hash",
        ),
        "source_state_hash": _hash(
            source_state_hash,
            "allocation source state hash",
        ),
        "frontier": normalized_frontier,
    }
    return {**body, "bootstrap_hash": sha256_json(body)}


def validate_allocation_settlement_frontier_bootstrap_v2(
    value: Any,
) -> Dict[str, Any]:
    _require(isinstance(value, Mapping), "frontier bootstrap must be an object")
    _require(
        set(value) == set(_FIELDS),
        "frontier bootstrap fields do not match schema",
    )
    _require(
        value.get("schema_version")
        == ALLOCATION_SETTLEMENT_FRONTIER_BOOTSTRAP_SCHEMA_VERSION,
        "frontier bootstrap schema is unsupported",
    )
    expected = build_allocation_settlement_frontier_bootstrap_v2(
        netuid=_integer(value.get("netuid"), "bootstrap netuid", minimum=1),
        bootstrap_epoch=_integer(
            value.get("bootstrap_epoch"),
            "bootstrap epoch",
            minimum=1,
        ),
        allocation_source_receipt_hash=_hash(
            value.get("allocation_source_receipt_hash"),
            "allocation source receipt hash",
        ),
        source_state_hash=_hash(
            value.get("source_state_hash"),
            "allocation source state hash",
        ),
        frontier=value.get("frontier"),
    )
    _require(
        value.get("allocation_epoch") == expected["allocation_epoch"],
        "frontier bootstrap allocation epoch differs",
    )
    _require(
        value.get("bootstrap_hash") == expected["bootstrap_hash"],
        "frontier bootstrap hash differs",
    )
    return expected


def frontier_bootstrap_artifact_hashes_v2(
    value: Mapping[str, Any],
) -> Tuple[str, ...]:
    bootstrap = validate_allocation_settlement_frontier_bootstrap_v2(value)
    return tuple(
        dict.fromkeys(
            (
                str(bootstrap["bootstrap_hash"]),
                str(bootstrap["source_state_hash"]),
                *frontier_artifact_hashes_v2(bootstrap["frontier"]),
            )
        )
    )
