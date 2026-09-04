"""Lab Arena reward-basis builders (labarena.md section 13).

The kernel every side of the weight path runs, the champion triple from a
signed basis, lives in ``leadpoet_canonical.lab_arena_rewards`` and is
imported back here unchanged, so there is exactly one kernel. This module
keeps what only the Arena needs: the reward constants a round publishes, the
king start-epoch rule, and the reward-basis document builder.

Arithmetic: the king's pool is ``pool_percent`` of total emissions
(``pool_basis`` is ``total_emissions``: it does not depend on the Research Lab
or leaderboard allocations, which the weight computation shrinks to make
room). ``pool_percent`` defaults to ``contracts.LAB_ARENA_POOL_PERCENT`` and
is set per round from ``LAB_ARENA_POOL_PERCENT``; every published basis
carries the constants it was computed with, so a change reaches validators
through the next round's signed basis and never rewrites an old one.
"""

from __future__ import annotations

from fractions import Fraction
from typing import Any, Dict, Mapping, Optional, Sequence

from lab_arena.contracts import (
    ELIGIBILITY_MAX_EPOCHS,
    EPOCHS_PER_REWARD_WEEK,
    KING_POOL_SHARE_PERCENT_BY_WEEK,
    LAB_ARENA_POOL_BASIS,
    LAB_ARENA_POOL_PERCENT,
    REWARD_BASIS_SCHEMA_VERSION,
    finalize_reward_basis,
    validate_reward_basis,
)
from leadpoet_canonical import lab_arena_rewards as _kernel
from leadpoet_canonical.lab_arena_rewards import (  # noqa: F401  (re-exported: one kernel)
    LabArenaRewardError,
    PAYING_KING_OUTCOMES,
    champion_uid_for_hotkey,
    champion_uid_matches,
    champion_values,
    epoch_eligible,
    require_king_outcome,
    validate_reward_constants,
)

MAX_REWARD_WEEK_INDEX = len(KING_POOL_SHARE_PERCENT_BY_WEEK) - 1


def _require_epoch(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("%s must be an integer epoch ordinal" % name)
    if value < 0:
        raise ValueError("%s must not be negative" % name)
    return value


def _require_share(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("%s must be a number" % name)
    number = float(value)
    if number != number or number in (float("inf"), float("-inf")):
        raise ValueError("%s must be finite" % name)
    if number < 0.0:
        raise ValueError("%s must not be negative" % name)
    return number


def _exact(value: float) -> Fraction:
    return Fraction(repr(float(value)))


def reward_week_index(epoch_id: int, king_start_epoch: int, constants: Optional[Mapping[str, Any]] = None) -> int:
    """The kernel's week index with this Arena's constants unless a basis's constants are given."""

    return _kernel.reward_week_index(epoch_id, king_start_epoch, constants if constants is not None else reward_constants_document())


def champion_share_for_week(week_index: int, constants: Optional[Mapping[str, Any]] = None) -> float:
    """The kernel's exact weekly share with this Arena's constants unless a basis's constants are given."""

    return _kernel.champion_share_for_week(week_index, constants if constants is not None else reward_constants_document())


def governing_reward_basis(rows: Sequence[Any], epoch_id: int) -> Optional[Dict[str, Any]]:
    """The kernel's governing row after every row passes the Arena contract validator (ArenaContractError)."""

    return _kernel.governing_reward_basis([validate_reward_basis(row) for row in rows], epoch_id)


def king_pool_share_percent(week_index: int) -> int:
    if isinstance(week_index, bool) or not isinstance(week_index, int):
        raise ValueError("week_index must be an integer")
    if week_index < 0 or week_index > MAX_REWARD_WEEK_INDEX:
        raise ValueError("week_index must be within 0..%d" % MAX_REWARD_WEEK_INDEX)
    return int(KING_POOL_SHARE_PERCENT_BY_WEEK[week_index])


def derive_research_lab_share(allocation_doc: Any, fallback_share: float) -> float:
    """Exact re-implementation of ``weight_computation._doc_percent_share``."""

    fallback = _require_share(fallback_share, "fallback_share")
    if isinstance(allocation_doc, dict) and allocation_doc.get("lab_cap_percent") not in (None, ""):
        try:
            return max(0.0, min(1.0, float(allocation_doc.get("lab_cap_percent")) / 100.0))
        except (TypeError, ValueError):
            return fallback
    return fallback


def fulfillment_residual(research_lab_share: float, leaderboard_bonus_share: float) -> float:
    """``max(0, 1 - research_lab_share - leaderboard_bonus_share)``: what fulfillment keeps before the king."""

    lab = _exact(_require_share(research_lab_share, "research_lab_share"))
    leaderboard = _exact(_require_share(leaderboard_bonus_share, "leaderboard_bonus_share"))
    residual = Fraction(1) - lab - leaderboard
    return float(residual if residual > 0 else Fraction(0))


def reward_constants_document(pool_percent: Optional[int] = None) -> Dict[str, Any]:
    """The constants a round publishes and every basis carries.

    ``pool_percent`` is the one adjustable value (``LAB_ARENA_POOL_PERCENT``,
    default 25): the share of total emissions the king's pool takes in week
    one. The weekly decay, week length, and eligibility window are public
    constants.
    """

    percent = LAB_ARENA_POOL_PERCENT if pool_percent is None else pool_percent
    return validate_reward_constants({
        "pool_percent": percent,
        "pool_basis": str(LAB_ARENA_POOL_BASIS),
        "king_pool_share_percent_by_week": [int(v) for v in KING_POOL_SHARE_PERCENT_BY_WEEK],
        "epochs_per_reward_week": int(EPOCHS_PER_REWARD_WEEK),
        "eligibility_max_epochs": int(ELIGIBILITY_MAX_EPOCHS),
    })


def king_start_epoch_for_outcome(
    king_outcome: str,
    effective_reward_epoch: int,
    previous_king_start_epoch: Optional[int],
) -> int:
    """Start a new miner schedule on a crown; keep it on a defense."""

    outcome = require_king_outcome(king_outcome)
    effective = _require_epoch(effective_reward_epoch, "effective_reward_epoch")
    if outcome == "crowned":
        return effective
    if outcome == "defended":
        if previous_king_start_epoch is None:
            raise ValueError("a defended miner requires a previous reward basis")
        previous = _require_epoch(previous_king_start_epoch, "previous_king_start_epoch")
        if previous >= effective:
            raise ValueError("a defended king's start epoch must precede this round's effective epoch")
        return previous
    # retained_ineligible / no_king keep the previous start epoch, or 0 when none.
    if previous_king_start_epoch is None:
        return 0
    previous = _require_epoch(previous_king_start_epoch, "previous_king_start_epoch")
    if previous > effective:
        raise ValueError("previous_king_start_epoch cannot follow the effective epoch")
    return previous


def reward_basis_document(
    *,
    round_id: str,
    published_at: str,
    finalized_epoch: int,
    king_outcome: str,
    king_hotkey: str,
    previous_king_start_epoch: Optional[int] = None,
    reward_constants: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Build and hash (but do not sign) the immutable reward-basis document.

    ``effective_reward_epoch`` is the finalized epoch at publication plus one.
    ``king_start_epoch`` follows :func:`king_start_epoch_for_outcome`.
    ``reward_constants`` are the round configuration's (the constants the
    round was announced with); the defaults apply when none are given.
    """

    outcome = require_king_outcome(king_outcome)
    effective = _require_epoch(finalized_epoch, "finalized_epoch") + 1
    if outcome == "no_king":
        if king_hotkey not in ("", None):
            raise ValueError("no_king outcome cannot name a king")
        hotkey = ""
    else:
        if not isinstance(king_hotkey, str) or not king_hotkey:
            raise ValueError("%s outcome requires a king hotkey" % outcome)
        hotkey = king_hotkey
    constants = validate_reward_constants(reward_constants) if reward_constants is not None else reward_constants_document()
    document = {
        "schema_version": REWARD_BASIS_SCHEMA_VERSION,
        "round_id": str(round_id),
        "published_at": str(published_at),
        "effective_reward_epoch": effective,
        "king_hotkey": hotkey,
        "king_outcome": outcome,
        "king_start_epoch": king_start_epoch_for_outcome(outcome, effective, previous_king_start_epoch),
        "reward_constants": constants,
    }
    return finalize_reward_basis(document)
