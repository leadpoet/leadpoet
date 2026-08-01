"""Canonical bounded settlement frontier for Research Lab allocations.

The append-only settlement and receipt tables remain the historical source of
truth.  This document carries only cumulative balances for obligations that
are still active at one allocation boundary.  A signed allocation execution
receipt commits the frontier through its source-state artifact hash; the next
allocation verifies that receipt and applies only the newly settled epochs.

This module is I/O-free and Python 3.7 compatible.
"""

from __future__ import annotations

from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
import re
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

from leadpoet_canonical.attested_v2 import sha256_json


ALLOCATION_SETTLEMENT_FRONTIER_SCHEMA_VERSION = (
    "leadpoet.research_lab_allocation_settlement_frontier.v2"
)
REWARD_SETTLEMENT_CHECKPOINT_SCHEMA_VERSION = (
    "leadpoet.research_lab_reward_settlement_checkpoint.v2"
)
FRONTIER_MODES = frozenset({"legacy_full_history_bootstrap", "bounded_delta_v1"})
REWARD_KINDS = frozenset({"champion", "source_add"})
MAX_REWARD_CHECKPOINTS = 512
MAX_NETUID = 2 ** 31 - 1
MAX_EPOCH = 2 ** 63 - 1
MAX_ALPHA_PERCENT = Decimal("1000000000.000000")
ALPHA_QUANT = Decimal("0.000001")

_HASH_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:/-]{0,255}$")

_CHECKPOINT_BODY_FIELDS = frozenset(
    {
        "schema_version",
        "reward_kind",
        "source_id",
        "obligation_hash",
        "start_epoch",
        "epoch_count",
        "desired_alpha_percent",
        "total_due_alpha_percent",
        "applied_alpha_percent",
        "realized_alpha_percent",
        "excess_alpha_percent",
    }
)
_CHECKPOINT_FIELDS = frozenset(
    set(_CHECKPOINT_BODY_FIELDS) | {"checkpoint_hash"}
)
_FRONTIER_BODY_FIELDS = frozenset(
    {
        "schema_version",
        "mode",
        "netuid",
        "allocation_epoch",
        "settled_through_epoch",
        "predecessor_frontier_hash",
        "reward_checkpoint_count",
        "reward_checkpoint_hashes_root",
        "reward_checkpoints",
    }
)
_FRONTIER_FIELDS = frozenset(set(_FRONTIER_BODY_FIELDS) | {"frontier_hash"})


class AllocationSettlementFrontierV2Error(ValueError):
    """A settlement frontier is malformed, inconsistent, or unbounded."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AllocationSettlementFrontierV2Error(message)


def _integer(
    value: Any,
    field: str,
    *,
    minimum: int = 0,
    maximum: int = MAX_EPOCH,
) -> int:
    _require(
        isinstance(value, int)
        and not isinstance(value, bool)
        and minimum <= value <= maximum,
        "%s is invalid" % field,
    )
    return int(value)


def _identifier(value: Any, field: str) -> str:
    raw = str(value or "")
    normalized = raw.strip()
    _require(raw == normalized, "%s is not canonical" % field)
    _require(bool(_IDENTIFIER_RE.fullmatch(normalized)), "%s is invalid" % field)
    return normalized


def _hash(value: Any, field: str) -> str:
    raw = str(value or "")
    normalized = raw.strip().lower()
    _require(raw == normalized, "%s is not canonical" % field)
    _require(bool(_HASH_RE.fullmatch(normalized)), "%s is invalid" % field)
    return normalized


def _alpha_text(value: Any, field: str) -> str:
    try:
        amount = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise AllocationSettlementFrontierV2Error(
            "%s is invalid" % field
        ) from exc
    _require(amount.is_finite(), "%s is invalid" % field)
    normalized = amount.quantize(ALPHA_QUANT, rounding=ROUND_HALF_UP)
    _require(
        Decimal("0") <= normalized <= MAX_ALPHA_PERCENT,
        "%s is outside the canonical bound" % field,
    )
    return format(normalized, ".6f")


def _validated_alpha_text(value: Any, field: str) -> str:
    _require(isinstance(value, str), "%s must be a decimal string" % field)
    normalized = _alpha_text(value, field)
    _require(value == normalized, "%s is not canonical" % field)
    return normalized


def _decimal(value: str) -> Decimal:
    return Decimal(value)


def build_reward_settlement_checkpoint_v2(
    *,
    reward_kind: str,
    source_id: str,
    obligation_hash: str,
    start_epoch: int,
    epoch_count: int,
    desired_alpha_percent: Any,
    applied_alpha_percent: Any,
    realized_alpha_percent: Any,
    excess_alpha_percent: Any,
) -> Dict[str, Any]:
    """Build one cumulative, immutable-obligation-bound reward checkpoint."""

    normalized_kind = str(reward_kind or "")
    _require(normalized_kind in REWARD_KINDS, "reward kind is unsupported")
    normalized_source_id = _identifier(source_id, "reward source_id")
    normalized_obligation_hash = _hash(obligation_hash, "obligation_hash")
    normalized_start = _integer(start_epoch, "reward start_epoch")
    normalized_count = _integer(
        epoch_count,
        "reward epoch_count",
        minimum=1,
    )
    desired = _alpha_text(desired_alpha_percent, "desired_alpha_percent")
    total_due = _alpha_text(
        _decimal(desired) * Decimal(normalized_count),
        "total_due_alpha_percent",
    )
    applied = _alpha_text(applied_alpha_percent, "applied_alpha_percent")
    realized = _alpha_text(realized_alpha_percent, "realized_alpha_percent")
    excess = _alpha_text(excess_alpha_percent, "excess_alpha_percent")
    _require(
        _decimal(applied) <= _decimal(total_due),
        "applied reward exceeds lifetime entitlement",
    )
    _require(
        _decimal(realized) >= _decimal(applied),
        "realized reward is below applied reward",
    )
    _require(
        _decimal(applied) == min(_decimal(total_due), _decimal(realized)),
        "applied reward differs from realized lifetime credit",
    )
    _require(
        _decimal(excess) == _decimal(realized) - _decimal(applied),
        "reward excess differs from realized less applied",
    )
    body = {
        "schema_version": REWARD_SETTLEMENT_CHECKPOINT_SCHEMA_VERSION,
        "reward_kind": normalized_kind,
        "source_id": normalized_source_id,
        "obligation_hash": normalized_obligation_hash,
        "start_epoch": normalized_start,
        "epoch_count": normalized_count,
        "desired_alpha_percent": desired,
        "total_due_alpha_percent": total_due,
        "applied_alpha_percent": applied,
        "realized_alpha_percent": realized,
        "excess_alpha_percent": excess,
    }
    return {**body, "checkpoint_hash": sha256_json(body)}


def validate_reward_settlement_checkpoint_v2(
    value: Any,
) -> Dict[str, Any]:
    _require(isinstance(value, Mapping), "reward checkpoint must be an object")
    _require(
        set(value) == set(_CHECKPOINT_FIELDS),
        "reward checkpoint fields do not match schema",
    )
    _require(
        value.get("schema_version")
        == REWARD_SETTLEMENT_CHECKPOINT_SCHEMA_VERSION,
        "reward checkpoint schema is unsupported",
    )
    expected = build_reward_settlement_checkpoint_v2(
        reward_kind=str(value.get("reward_kind") or ""),
        source_id=str(value.get("source_id") or ""),
        obligation_hash=str(value.get("obligation_hash") or ""),
        start_epoch=_integer(value.get("start_epoch"), "reward start_epoch"),
        epoch_count=_integer(
            value.get("epoch_count"),
            "reward epoch_count",
            minimum=1,
        ),
        desired_alpha_percent=_validated_alpha_text(
            value.get("desired_alpha_percent"),
            "desired_alpha_percent",
        ),
        applied_alpha_percent=_validated_alpha_text(
            value.get("applied_alpha_percent"),
            "applied_alpha_percent",
        ),
        realized_alpha_percent=_validated_alpha_text(
            value.get("realized_alpha_percent"),
            "realized_alpha_percent",
        ),
        excess_alpha_percent=_validated_alpha_text(
            value.get("excess_alpha_percent"),
            "excess_alpha_percent",
        ),
    )
    _require(
        value.get("total_due_alpha_percent")
        == expected["total_due_alpha_percent"],
        "reward total due differs",
    )
    _require(
        value.get("checkpoint_hash") == expected["checkpoint_hash"],
        "reward checkpoint hash differs",
    )
    return expected


def reward_checkpoint_key_v2(value: Mapping[str, Any]) -> Tuple[str, str]:
    checkpoint = validate_reward_settlement_checkpoint_v2(value)
    return str(checkpoint["reward_kind"]), str(checkpoint["source_id"])


def reward_checkpoint_index_v2(
    values: Sequence[Mapping[str, Any]],
) -> Dict[Tuple[str, str], Dict[str, Any]]:
    _require(
        isinstance(values, Sequence)
        and not isinstance(values, (str, bytes, bytearray)),
        "reward checkpoints must be an array",
    )
    _require(
        len(values) <= MAX_REWARD_CHECKPOINTS,
        "reward checkpoint count exceeds the active-obligation bound",
    )
    indexed: Dict[Tuple[str, str], Dict[str, Any]] = {}
    normalized = [validate_reward_settlement_checkpoint_v2(item) for item in values]
    keys = [(str(item["reward_kind"]), str(item["source_id"])) for item in normalized]
    _require(keys == sorted(keys), "reward checkpoints are not canonically ordered")
    _require(len(keys) == len(set(keys)), "reward checkpoints are duplicated")
    for key, item in zip(keys, normalized):
        indexed[key] = item
    return indexed


def build_allocation_settlement_frontier_v2(
    *,
    mode: str,
    netuid: int,
    allocation_epoch: int,
    predecessor_frontier_hash: Optional[str],
    reward_checkpoints: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Build one bounded frontier through ``allocation_epoch - 1``."""

    normalized_mode = str(mode or "")
    _require(normalized_mode in FRONTIER_MODES, "frontier mode is unsupported")
    normalized_netuid = _integer(netuid, "frontier netuid", maximum=MAX_NETUID)
    normalized_epoch = _integer(
        allocation_epoch,
        "frontier allocation_epoch",
        minimum=1,
    )
    normalized_predecessor: Optional[str]
    if predecessor_frontier_hash is None:
        normalized_predecessor = None
    else:
        normalized_predecessor = _hash(
            predecessor_frontier_hash,
            "predecessor_frontier_hash",
        )
    if normalized_mode == "legacy_full_history_bootstrap":
        _require(
            normalized_predecessor is None,
            "bootstrap frontier cannot declare a predecessor",
        )
    else:
        _require(
            normalized_predecessor is not None,
            "bounded frontier requires a predecessor",
        )
    indexed = reward_checkpoint_index_v2(reward_checkpoints)
    normalized_checkpoints = [indexed[key] for key in sorted(indexed)]
    checkpoint_hashes = [str(item["checkpoint_hash"]) for item in normalized_checkpoints]
    body = {
        "schema_version": ALLOCATION_SETTLEMENT_FRONTIER_SCHEMA_VERSION,
        "mode": normalized_mode,
        "netuid": normalized_netuid,
        "allocation_epoch": normalized_epoch,
        "settled_through_epoch": normalized_epoch - 1,
        "predecessor_frontier_hash": normalized_predecessor,
        "reward_checkpoint_count": len(normalized_checkpoints),
        "reward_checkpoint_hashes_root": sha256_json(checkpoint_hashes),
        "reward_checkpoints": normalized_checkpoints,
    }
    return {**body, "frontier_hash": sha256_json(body)}


def validate_allocation_settlement_frontier_v2(value: Any) -> Dict[str, Any]:
    _require(isinstance(value, Mapping), "settlement frontier must be an object")
    _require(
        set(value) == set(_FRONTIER_FIELDS),
        "settlement frontier fields do not match schema",
    )
    _require(
        value.get("schema_version")
        == ALLOCATION_SETTLEMENT_FRONTIER_SCHEMA_VERSION,
        "settlement frontier schema is unsupported",
    )
    allocation_epoch = _integer(
        value.get("allocation_epoch"),
        "frontier allocation_epoch",
        minimum=1,
    )
    _require(
        value.get("settled_through_epoch") == allocation_epoch - 1,
        "settlement frontier boundary differs from allocation epoch",
    )
    checkpoints = value.get("reward_checkpoints")
    _require(isinstance(checkpoints, list), "reward checkpoints must be an array")
    predecessor = value.get("predecessor_frontier_hash")
    if predecessor is not None:
        predecessor = _hash(predecessor, "predecessor_frontier_hash")
    expected = build_allocation_settlement_frontier_v2(
        mode=str(value.get("mode") or ""),
        netuid=_integer(value.get("netuid"), "frontier netuid", maximum=MAX_NETUID),
        allocation_epoch=allocation_epoch,
        predecessor_frontier_hash=predecessor,
        reward_checkpoints=checkpoints,
    )
    _require(
        value.get("reward_checkpoint_count")
        == expected["reward_checkpoint_count"],
        "reward checkpoint count differs",
    )
    _require(
        value.get("reward_checkpoint_hashes_root")
        == expected["reward_checkpoint_hashes_root"],
        "reward checkpoint root differs",
    )
    _require(
        value.get("frontier_hash") == expected["frontier_hash"],
        "settlement frontier hash differs",
    )
    return expected


def validate_frontier_successor_v2(
    predecessor: Mapping[str, Any],
    successor: Mapping[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    previous = validate_allocation_settlement_frontier_v2(predecessor)
    current = validate_allocation_settlement_frontier_v2(successor)
    _require(
        current["mode"] == "bounded_delta_v1",
        "frontier successor mode is invalid",
    )
    _require(current["netuid"] == previous["netuid"], "frontier netuid changed")
    _require(
        current["allocation_epoch"] > previous["allocation_epoch"],
        "frontier allocation epoch did not advance",
    )
    _require(
        current["settled_through_epoch"] > previous["settled_through_epoch"],
        "frontier settlement boundary did not advance",
    )
    _require(
        current["predecessor_frontier_hash"] == previous["frontier_hash"],
        "frontier predecessor hash differs",
    )
    return previous, current


def frontier_paid_maps_v2(
    frontier: Mapping[str, Any],
) -> Dict[str, Dict[str, float]]:
    normalized = validate_allocation_settlement_frontier_v2(frontier)
    result: Dict[str, Dict[str, float]] = {
        "champion": {},
        "source_add": {},
    }
    for checkpoint in normalized["reward_checkpoints"]:
        result[str(checkpoint["reward_kind"])][str(checkpoint["source_id"])] = float(
            checkpoint["applied_alpha_percent"]
        )
    return result


def frontier_artifact_hashes_v2(frontier: Mapping[str, Any]) -> Tuple[str, ...]:
    normalized = validate_allocation_settlement_frontier_v2(frontier)
    return tuple(
        [str(normalized["frontier_hash"])]
        + [
            str(item["checkpoint_hash"])
            for item in normalized["reward_checkpoints"]
        ]
    )
