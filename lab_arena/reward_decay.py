"""Prepare per-award clocks for the existing signed accepted weight state."""

from copy import deepcopy
from typing import Any, Callable, Mapping, Optional, Sequence

from leadpoet_canonical.lab_arena_rewards import validate_weekly_slot_decay


def weekly_slot_decay(
    basis: Mapping[str, Any], *, epoch: int,
    source_identity: Callable[[Mapping[str, Any]], str],
    history: Sequence[Mapping[str, Any]],
    previous_state: Optional[Mapping[str, Any]] = None,
    history_complete: bool = True,
) -> dict:
    """Bind each slot to its verified qualifying award, independent of source.

    A new award starts at its own effective epoch, even for the same owner
    and normalized model. Retaining or replaying the same award preserves its
    award epoch through daily, funding and policy changes. Reuse checkpoints
    only when they match that epoch; historical accepted states are unchanged.
    """
    previous = previous_state or {}
    old_clocks = (previous.get("weekly_slot_decay") or {}).get("slots", [None] * 3)
    old_slots = (previous.get("reward_basis") or {}).get("reward_slots", [None] * 3)
    clocks = []
    for index, achievement in enumerate(basis["reward_slots"]):
        if achievement is None:
            clocks.append(None)
            continue
        start = achievement.get("start_epoch")
        if start is None:
            if achievement["round_id"] == basis["round_id"]:
                start = basis["effective_reward_epoch"]
            else:
                # Pre-decay policies did not freeze a start on carried slots.
                # Resolve only this exact award's original signed basis; an
                # older award with identical source bytes is not this award.
                source = next((item for item in history
                               if item["round_id"] == achievement["round_id"]), None)
                if source is None:
                    if not history_complete:
                        raise ValueError("reward decay history is incomplete")
                    raise ValueError("reward decay source promotion is missing")
                start = source["effective_reward_epoch"]
        # Old issuers could carry a previous model's clock onto a genuine
        # re-win. Repair only a future state, after verifying this award's
        # frozen source, instead of propagating that incorrect checkpoint.
        prior = old_slots[index]
        prior_clock = old_clocks[index]
        same_owner = prior is not None and prior["miner_hotkey"] == achievement["miner_hotkey"]
        if (prior_clock is not None and same_owner
                and prior["round_id"] == achievement["round_id"]
                and prior["submission_id"] == achievement["submission_id"]
                and prior_clock["start_epoch"] == start):
            clocks.append(deepcopy(prior_clock))
            continue
        identity = source_identity(achievement)
        clocks.append({"start_epoch": start, "source_normalized_sha256": identity})
    return validate_weekly_slot_decay(
        {"epochs_per_halving": 140, "slots": clocks}, basis, epoch_id=epoch,
    )
