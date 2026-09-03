"""Lab Arena reward kernel (labarena.md section 13).

Pure functions only: reward-week decay (13.2), epoch eligibility (13.3), the
champion triple the canonical weight computation consumes (13.1), and the
signed reward-basis document builder (13.4). Nothing here performs I/O, reads
the environment, or touches chain state.

Compatibility contract: the reward release relocates this file into
``leadpoet_canonical/`` unchanged, so it is written in Python 3.7 syntax
(``typing`` generics, ``# type:`` comments, no walrus, no ``match``, no PEP 604
unions, no builtin generics) and imports only the standard library plus the
public constants and the reward-basis schema helpers from
``lab_arena.contracts``. ``tests/lab_arena/test_lab_arena_rewards.py`` parses
this file and rejects post-3.7 syntax.

Arithmetic: the king's pool is ``LAB_ARENA_POOL_PERCENT`` of total emissions
(``pool_basis`` is ``total_emissions``: it does not depend on the Research Lab
or leaderboard allocations, which the reward adapter shrinks to make room).
The champion share is evaluated exactly with ``fractions.Fraction`` and
converted back to a float once, so the five weekly values are exactly the
floats ``0.25``, ``0.2``, ``0.15``, ``0.1`` and ``0.05`` and compare equal to
those literals on every side of the weight path. ``fulfillment_residual`` and
``derive_research_lab_share`` remain the byte-for-byte mirrors of the canonical
kernel the adapter uses to derive what fulfillment keeps.
"""

from __future__ import annotations

import math
from fractions import Fraction
from typing import Any, Dict, List, Mapping, Optional, Sequence

from lab_arena.contracts import (
    ELIGIBILITY_MAX_EPOCHS,
    EPOCHS_PER_REWARD_WEEK,
    KING_OUTCOMES,
    KING_POOL_SHARE_PERCENT_BY_WEEK,
    LAB_ARENA_POOL_BASIS,
    LAB_ARENA_POOL_PERCENT,
    REWARD_BASIS_SCHEMA_VERSION,
    finalize_reward_basis,
    validate_reward_basis,
)

# Outcomes that pay the king (section 13.3). Every other outcome returns the
# whole Arena amount to fulfillment.
PAYING_KING_OUTCOMES = ("crowned", "defended")
MAX_REWARD_WEEK_INDEX = len(KING_POOL_SHARE_PERCENT_BY_WEEK) - 1


# ---------------------------------------------------------------------------
# Input guards (every reader of a reward basis fails closed)
# ---------------------------------------------------------------------------


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
    if not math.isfinite(number):
        raise ValueError("%s must be finite" % name)
    if number < 0.0:
        raise ValueError("%s must not be negative" % name)
    return number


def _exact(value: float) -> Fraction:
    """Exact rational of a float's shortest decimal representation."""

    return Fraction(repr(float(value)))


def require_king_outcome(value: Any) -> str:
    """Return ``value`` when it is one of the four closed outcomes, else raise."""

    if not isinstance(value, str) or value not in KING_OUTCOMES:
        raise ValueError("king_outcome must be one of %s" % ", ".join(KING_OUTCOMES))
    return value


def _basis_fields(basis: Any) -> Dict[str, Any]:
    """Read the reward-basis fields the kernel needs, failing closed on shape."""

    if not isinstance(basis, Mapping):
        raise ValueError("reward basis must be an object")
    outcome = require_king_outcome(basis.get("king_outcome"))
    effective = _require_epoch(basis.get("effective_reward_epoch"), "effective_reward_epoch")
    start = _require_epoch(basis.get("king_start_epoch"), "king_start_epoch")
    if start > effective:
        raise ValueError("king_start_epoch cannot follow effective_reward_epoch")
    hotkey = basis.get("king_hotkey")
    if not isinstance(hotkey, str):
        raise ValueError("king_hotkey must be a string")
    if outcome == "no_king":
        if hotkey != "":
            raise ValueError("no_king outcome cannot name a king")
    elif not hotkey:
        raise ValueError("%s outcome requires a king hotkey" % outcome)
    return {
        "king_outcome": outcome,
        "effective_reward_epoch": effective,
        "king_start_epoch": start,
        "king_hotkey": hotkey,
    }


# ---------------------------------------------------------------------------
# Weekly decay (section 13.2)
# ---------------------------------------------------------------------------


def reward_week_index(epoch_id: int, king_start_epoch: int) -> int:
    """``min(floor((epoch_id - king_start_epoch) / EPOCHS_PER_REWARD_WEEK), 4)``."""

    epoch_id = _require_epoch(epoch_id, "epoch_id")
    king_start_epoch = _require_epoch(king_start_epoch, "king_start_epoch")
    if epoch_id < king_start_epoch:
        raise ValueError("epoch_id precedes king_start_epoch")
    return min((epoch_id - king_start_epoch) // EPOCHS_PER_REWARD_WEEK, MAX_REWARD_WEEK_INDEX)


def king_pool_share_percent(week_index: int) -> int:
    if isinstance(week_index, bool) or not isinstance(week_index, int):
        raise ValueError("week_index must be an integer")
    if week_index < 0 or week_index > MAX_REWARD_WEEK_INDEX:
        raise ValueError("week_index must be within 0..%d" % MAX_REWARD_WEEK_INDEX)
    return int(KING_POOL_SHARE_PERCENT_BY_WEEK[week_index])


# ---------------------------------------------------------------------------
# Shares (section 13.1)
# ---------------------------------------------------------------------------


def derive_research_lab_share(allocation_doc: Any, fallback_share: float) -> float:
    """Exact re-implementation of ``weight_computation._doc_percent_share``.

    A ``dict`` with a non-empty ``lab_cap_percent`` yields
    ``clamp(float(value) / 100, 0, 1)``; an unparsable value or a missing key
    yields ``fallback_share``. The clamp intentionally mirrors the canonical
    kernel byte for byte (including its treatment of non-finite values) so the
    validator host and the gateway coordinator derive the same number.
    """

    fallback = _require_share(fallback_share, "fallback_share")
    if isinstance(allocation_doc, dict) and allocation_doc.get("lab_cap_percent") not in (None, ""):
        try:
            return max(0.0, min(1.0, float(allocation_doc.get("lab_cap_percent")) / 100.0))
        except (TypeError, ValueError):
            return fallback
    return fallback


def _exact_residual(research_lab_share: float, leaderboard_bonus_share: float) -> Fraction:
    lab = _exact(_require_share(research_lab_share, "research_lab_share"))
    leaderboard = _exact(_require_share(leaderboard_bonus_share, "leaderboard_bonus_share"))
    residual = Fraction(1) - lab - leaderboard
    if residual < 0:
        return Fraction(0)
    return residual


def fulfillment_residual(research_lab_share: float, leaderboard_bonus_share: float) -> float:
    """``max(0, 1 - research_lab_share - leaderboard_bonus_share)``."""

    return float(_exact_residual(research_lab_share, leaderboard_bonus_share))


def champion_share_for_week(week_index: int) -> float:
    """``LAB_ARENA_POOL_PERCENT / 100 * week_share / 100`` of total emissions, exactly.

    The pool basis is total emissions: the king's share never shrinks when the
    Research Lab or leaderboard allocations grow.
    """

    pool = Fraction(LAB_ARENA_POOL_PERCENT, 100)
    week = Fraction(king_pool_share_percent(week_index), 100)
    return float(pool * week)


# ---------------------------------------------------------------------------
# Governing row and eligibility (section 13.3)
# ---------------------------------------------------------------------------


def governing_reward_basis(rows: Sequence[Any], epoch_id: int) -> Optional[Dict[str, Any]]:
    """The published basis with the greatest ``effective_reward_epoch <= epoch_id``.

    Every row is validated with ``contracts.validate_reward_basis``. Two rows
    sharing one effective epoch violate the write-once/unique publication rule
    and raise, whether or not either of them governs ``epoch_id``.
    """

    epoch_id = _require_epoch(epoch_id, "epoch_id")
    seen = set()  # type: set
    governing = None  # type: Optional[Dict[str, Any]]
    for row in rows:
        basis = validate_reward_basis(row)
        effective = basis["effective_reward_epoch"]
        if effective in seen:
            raise ValueError("duplicate effective_reward_epoch %d among reward bases" % effective)
        seen.add(effective)
        if effective > epoch_id:
            continue
        if governing is None or effective > governing["effective_reward_epoch"]:
            governing = basis
    return governing


def epoch_eligible(basis: Any, epoch_id: int) -> bool:
    """Eligible when the row is at most ``ELIGIBILITY_MAX_EPOCHS`` old and pays.

    Raises when the basis is not yet effective for ``epoch_id``: only the
    governing row may be passed here. An outcome outside the closed vocabulary
    raises as well, so an unknown outcome can never pay.
    """

    fields = _basis_fields(basis)
    epoch_id = _require_epoch(epoch_id, "epoch_id")
    effective = fields["effective_reward_epoch"]
    if epoch_id < effective:
        raise ValueError("reward basis is not effective at epoch %d" % epoch_id)
    if epoch_id - effective > ELIGIBILITY_MAX_EPOCHS:
        return False
    return fields["king_outcome"] in PAYING_KING_OUTCOMES


# ---------------------------------------------------------------------------
# Hotkey binding and champion triple (section 13.1)
# ---------------------------------------------------------------------------


def _require_hotkeys(metagraph_hotkeys: Any) -> List[str]:
    if isinstance(metagraph_hotkeys, (str, bytes)) or not isinstance(metagraph_hotkeys, Sequence):
        raise ValueError("metagraph_hotkeys must be a sequence of strings")
    out = []  # type: List[str]
    for item in metagraph_hotkeys:
        if not isinstance(item, str):
            raise ValueError("metagraph_hotkeys must be a sequence of strings")
        out.append(item)
    return out


def champion_uid_for_hotkey(metagraph_hotkeys: Sequence[str], king_hotkey: str) -> Optional[int]:
    """UID whose metagraph hotkey equals the king, or ``None`` when unregistered."""

    hotkeys = _require_hotkeys(metagraph_hotkeys)
    if not isinstance(king_hotkey, str) or not king_hotkey:
        return None
    matches = [uid for uid, hotkey in enumerate(hotkeys) if hotkey == king_hotkey]
    if len(matches) > 1:
        raise ValueError("king hotkey is registered at more than one UID")
    if not matches:
        return None
    return matches[0]


def champion_uid_matches(metagraph_hotkeys: Sequence[str], champion_uid: Any, king_hotkey: str) -> bool:
    """True only when ``metagraph_hotkeys[champion_uid]`` is the king hotkey."""

    hotkeys = _require_hotkeys(metagraph_hotkeys)
    if isinstance(champion_uid, bool) or not isinstance(champion_uid, int):
        return False
    if champion_uid < 0 or champion_uid >= len(hotkeys):
        return False
    return bool(king_hotkey) and hotkeys[champion_uid] == king_hotkey


def champion_values(
    basis: Any,
    epoch_id: int,
    metagraph_hotkeys: Sequence[str],
) -> Dict[str, Any]:
    """The champion triple for one weight epoch from the governing basis.

    ``champion_share`` is the week's share of total emissions.
    ``champion_share`` and ``effective_champion_share`` are always equal
    (section 13.1: the Arena never burns a gap). Both are ``0.0`` and
    ``champion_uid`` is ``None`` whenever the epoch is ineligible or the king
    hotkey is not registered on the supplied finalized metagraph.
    ``reward_week_index`` is reported whenever a king exists so the decay
    clock is visible even on an ineligible epoch; it is ``None`` for
    ``no_king``.
    """

    fields = _basis_fields(basis)
    epoch_id = _require_epoch(epoch_id, "epoch_id")
    hotkeys = _require_hotkeys(metagraph_hotkeys)
    eligible = epoch_eligible(fields, epoch_id)
    week_index = None  # type: Optional[int]
    if fields["king_outcome"] != "no_king":
        week_index = reward_week_index(epoch_id, fields["king_start_epoch"])
    uid = None  # type: Optional[int]
    share = 0.0
    if eligible:
        uid = champion_uid_for_hotkey(hotkeys, fields["king_hotkey"])
        if uid is not None:
            if week_index is None:
                raise ValueError("eligible basis without a king start epoch")
            share = champion_share_for_week(week_index)
    return {
        "champion_share": share,
        "effective_champion_share": share,
        "champion_uid": uid,
        "reward_week_index": week_index,
        "eligible": eligible,
    }


# ---------------------------------------------------------------------------
# Reward-basis document (section 13.4)
# ---------------------------------------------------------------------------


def reward_constants_document() -> Dict[str, Any]:
    return {
        "pool_percent": int(LAB_ARENA_POOL_PERCENT),
        "pool_basis": str(LAB_ARENA_POOL_BASIS),
        "king_pool_share_percent_by_week": [int(v) for v in KING_POOL_SHARE_PERCENT_BY_WEEK],
        "epochs_per_reward_week": int(EPOCHS_PER_REWARD_WEEK),
        "eligibility_max_epochs": int(ELIGIBILITY_MAX_EPOCHS),
    }


def king_start_epoch_for_outcome(
    king_outcome: str,
    effective_reward_epoch: int,
    previous_king_start_epoch: Optional[int],
) -> int:
    """Start epoch rule: a new king restarts the schedule, everything else keeps it."""

    outcome = require_king_outcome(king_outcome)
    effective = _require_epoch(effective_reward_epoch, "effective_reward_epoch")
    if outcome == "crowned":
        return effective
    if outcome == "defended":
        if previous_king_start_epoch is None:
            raise ValueError("a defended king must keep its previous start epoch")
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
    configuration_hash: str,
    commitment_hash: str,
    result_bundle_hash: str,
    published_at: str,
    finalized_epoch: int,
    king_outcome: str,
    king_hotkey: str,
    previous_king_start_epoch: Optional[int] = None,
) -> Dict[str, Any]:
    """Build and hash (but do not sign) the immutable reward-basis document.

    ``effective_reward_epoch`` is the finalized epoch at publication plus one.
    ``king_start_epoch`` follows :func:`king_start_epoch_for_outcome`.
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
    document = {
        "schema_version": REWARD_BASIS_SCHEMA_VERSION,
        "round_id": str(round_id),
        "configuration_hash": str(configuration_hash),
        "commitment_hash": str(commitment_hash),
        "result_bundle_hash": str(result_bundle_hash),
        "published_at": str(published_at),
        "effective_reward_epoch": effective,
        "king_hotkey": hotkey,
        "king_outcome": outcome,
        "king_start_epoch": king_start_epoch_for_outcome(outcome, effective, previous_king_start_epoch),
        "reward_constants": reward_constants_document(),
    }
    return finalize_reward_basis(document)
