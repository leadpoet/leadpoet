"""Prepare per-model clocks for the existing signed accepted weight state."""

from copy import deepcopy
from typing import Any, Callable, Mapping, Optional, Sequence

from lab_arena import rewards
from leadpoet_canonical.lab_arena_rewards import validate_weekly_slot_decay


def weekly_slot_decay(
    basis: Mapping[str, Any], *, epoch: int,
    source_identity: Callable[[Mapping[str, Any]], str],
    history: Sequence[Mapping[str, Any]],
    previous_state: Optional[Mapping[str, Any]] = None,
    history_complete: bool = True,
) -> dict:
    """Use verified signed promotion history and exact source bytes.

    An unchanged owner and normalized model retain their clocks. A genuinely
    different promoted model starts at its achievement's effective epoch.
    A daily basis, funding change or a repackaged archive is not a new model.
    Previously signed clocks are durable checkpoints, not mutable counters.
    """
    previous = previous_state or {}
    old_clocks = (previous.get("weekly_slot_decay") or {}).get("slots", [None] * 3)
    old_slots = (previous.get("reward_basis") or {}).get("reward_slots", [None] * 3)
    clocks = []
    for index, achievement in enumerate(basis["reward_slots"]):
        if achievement is None:
            clocks.append(None)
            continue
        prior = old_slots[index]
        prior_clock = old_clocks[index]
        same_owner = prior is not None and prior["miner_hotkey"] == achievement["miner_hotkey"]
        if (prior_clock is not None and same_owner
                and prior["round_id"] == achievement["round_id"]
                and prior["submission_id"] == achievement["submission_id"]):
            clocks.append(deepcopy(prior_clock))
            continue
        identity = source_identity(achievement)
        start = achievement["start_epoch"]
        if start is None:
            start = basis["effective_reward_epoch"]
        # First deployment/recovery can have no signed clock yet. Walk only
        # consecutive prior occupants of this tier, never revive an older
        # model across an intervening different owner/model or empty slot.
        checkpoint = None
        while True:
            older = [item for item in history if item["effective_reward_epoch"] < start]
            if not older:
                if not history_complete:
                    raise ValueError("reward decay history is incomplete")
                break
            older_basis = max(older, key=lambda item: item["effective_reward_epoch"])
            if "slot_policy" not in older_basis:
                break
            # Allocation/decay rules do not establish model identity. Retain
            # the old decay shape while projecting only the tier thresholds.
            identity_policy = deepcopy(basis["slot_policy"])
            if "decay" in older_basis["slot_policy"]:
                identity_policy["decay"] = older_basis["slot_policy"]["decay"]
            else:
                identity_policy.pop("decay", None)
            projected = rewards.reward_basis_policy_projection_document(older_basis, identity_policy)
            earlier_slot = projected["reward_slots"][index]
            if earlier_slot is None or earlier_slot["miner_hotkey"] != achievement["miner_hotkey"]:
                break
            reaches_checkpoint = (
                prior_clock is not None and same_owner
                and earlier_slot["round_id"] == prior["round_id"]
                and earlier_slot["submission_id"] == prior["submission_id"]
            )
            earlier_identity = (prior_clock["source_normalized_sha256"]
                                if reaches_checkpoint else source_identity(earlier_slot))
            if earlier_identity != identity:
                break
            if reaches_checkpoint:
                checkpoint = deepcopy(prior_clock)
                break
            earlier = earlier_slot.get("start_epoch")
            if earlier is None:
                source = next((item for item in history
                               if item["round_id"] == earlier_slot["round_id"]), None)
                if source is None:
                    raise ValueError("reward decay source promotion is missing")
                earlier = source["effective_reward_epoch"]
            if earlier >= start:
                raise ValueError("reward decay source history is not ordered")
            start = earlier
        if checkpoint is not None:
            clocks.append(checkpoint)
            continue
        clocks.append({"start_epoch": start, "source_normalized_sha256": identity})
    return validate_weekly_slot_decay(
        {"epochs_per_halving": 140, "slots": clocks}, basis, epoch_id=epoch,
    )
