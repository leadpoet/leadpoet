"""Merge the lead-order lane into the Arena weight vector (live mode only).

Rules:

* No lane payout (off, staff, shadow, or an epoch with nothing paid) returns
  ``derive_arena_weights`` unchanged: the same object the Arena path returns.
* A lane payout takes weight only from the burn residual, never from the
  Arena champion or slot holders, and never more than the lane cap.
* Every lane hotkey must be registered at the finalized metagraph; the caller
  pays only registered hotkeys (``registered_entries``), so nothing recorded
  as paid is silently burned.

``leadpoet_canonical.arena_weights`` is attested and AST-pinned, so it is not
edited. ``_allocations`` and ``_sparse`` restate its allocation and
float32/u16 steps; a test pins that, with no lane, they reproduce
``derive_arena_weights`` exactly for every state shape.
"""

from __future__ import annotations

import struct
from fractions import Fraction
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from leadpoet_canonical import arena_weights
from leadpoet_canonical.arena_weights import (
    PARTS_PER_BILLION,
    REWARD_BASIS_V2_SCHEMA_VERSION,
    SLOT_ELIGIBILITY_UNTIL_REPLACED,
    U16_MAX,
    ArenaWeightError,
    champion_values,
    compare_weights_hash,
    slot_allocations,
)
from lab_arena.lead_orders.money import OwedEntry


def _allocations(state: Mapping[str, Any], hotkeys: Sequence[str]) -> Tuple[Dict[str, Fraction], Fraction, Fraction]:
    by_hotkey: Dict[str, Fraction] = {}
    burn_hotkey = str(state["burn_hotkey"])
    if state["reward_basis"]["schema_version"] == REWARD_BASIS_V2_SCHEMA_VERSION:
        by_hotkey = dict(slot_allocations(
            state["reward_basis"], int(state["epoch"]), list(hotkeys),
            until_replaced=state.get("slot_eligibility") == SLOT_ELIGIBILITY_UNTIL_REPLACED,
            weekly_slot_decay=state.get("weekly_slot_decay"),
        ))
        champion_share = sum(by_hotkey.values(), Fraction(0))
    else:
        champion = champion_values(state["reward_basis"], int(state["epoch"]), list(hotkeys))
        champion_share = Fraction(str(champion["champion_share"]))
        if champion["champion_uid"] is not None and champion_share:
            winner = hotkeys[int(champion["champion_uid"])]
            by_hotkey[winner] = by_hotkey.get(winner, Fraction(0)) + champion_share
        else:
            champion_share = Fraction(0)
    unused = Fraction(1) - champion_share
    if unused:
        by_hotkey[burn_hotkey] = by_hotkey.get(burn_hotkey, Fraction(0)) + unused
    return by_hotkey, champion_share, unused


def _sparse(state: Mapping[str, Any], hotkeys: Sequence[str], by_hotkey: Mapping[str, Fraction],
            champion_share: Fraction, unused: Fraction) -> Dict[str, Any]:
    registered = [(uid, by_hotkey[h]) for uid, h in enumerate(hotkeys) if by_hotkey.get(h, Fraction(0)) > 0]
    if not registered:
        raise ArenaWeightError("accepted state allocates no weight to finalized UIDs")
    float32 = lambda item: struct.unpack("!f", struct.pack("!f", float(item)))[0]
    float_weights = [(uid, float32(weight)) for uid, weight in registered]
    maximum = float(max(weight for _, weight in float_weights))
    sparse = [(uid, round((float(weight) / maximum) * U16_MAX)) for uid, weight in float_weights]
    sparse = [(uid, weight) for uid, weight in sparse if weight > 0]
    return {
        "state_hash": state["state_hash"], "netuid": state["netuid"], "epoch": state["epoch"],
        "sparse_uids": [uid for uid, _ in sparse], "sparse_weights_u16": [w for _, w in sparse],
        "weights_hash": compare_weights_hash(int(state["netuid"]), int(state["epoch"]), sparse),
        "champion_share_ppb": int(champion_share * PARTS_PER_BILLION),
        "burned_residual_ppb": int(unused * PARTS_PER_BILLION),
    }


def burn_residual(state: Mapping[str, Any], hotkeys: Sequence[str]) -> Fraction:
    """Share of emission the Arena leaves to the burn hotkey this epoch."""

    arena_weights.derive_arena_weights(state, hotkeys)  # full validation
    return _allocations(state, hotkeys)[2]


def registered_entries(entries: Iterable[OwedEntry], hotkeys: Sequence[str]) -> List[OwedEntry]:
    """Owed entries whose miner can receive weight now; the rest wait."""

    registered = set(hotkeys)
    return [entry for entry in entries if entry.miner_hotkey in registered]


def merge_lane_weights(
    state: Any,
    hotkeys: Sequence[str],
    lane_fractions: Optional[Mapping[str, Fraction]] = None,
    *,
    lane_cap: Fraction = Fraction(0),
) -> Dict[str, Any]:
    base = arena_weights.derive_arena_weights(state, hotkeys)
    lane = {k: Fraction(v) for k, v in (lane_fractions or {}).items() if Fraction(v) > 0}
    if not lane:
        return base
    total = sum(lane.values(), Fraction(0))
    if total > lane_cap:
        raise ArenaWeightError("lead-order lane exceeds its emission cap")
    by_hotkey, champion_share, unused = _allocations(state, hotkeys)
    if total > unused:
        raise ArenaWeightError("lead-order lane exceeds the burn residual")
    registered = set(hotkeys)
    for hotkey in lane:
        if hotkey not in registered:
            raise ArenaWeightError("lead-order lane pays an unregistered hotkey")
    burn_hotkey = str(state["burn_hotkey"])
    by_hotkey[burn_hotkey] = by_hotkey.get(burn_hotkey, Fraction(0)) - total
    for hotkey, share in lane.items():
        by_hotkey[hotkey] = by_hotkey.get(hotkey, Fraction(0)) + share
    out = _sparse(state, hotkeys, by_hotkey, champion_share, unused - total)
    out["lead_order_lane_ppb"] = int(total * PARTS_PER_BILLION)
    return out


__all__ = ["burn_residual", "merge_lane_weights", "registered_entries"]
