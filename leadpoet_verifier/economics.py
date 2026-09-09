"""Deterministic Research Lab economics for the open verifier."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal, ROUND_DOWN, ROUND_HALF_UP
import hashlib
import json
import re
from typing import Any, Dict, Mapping, Optional, Sequence


MICRO_USD = Decimal("1000000")
RATE_QUANT = Decimal("0.000001")

DEFAULT_RESEARCH_LAB_EMISSION_PERCENT = Decimal("30.0")
DEFAULT_RESEARCH_LAB_REWARD_EPOCHS = 20
DEFAULT_RESEARCH_LAB_CHAMPION_MIN_ALPHA_PERCENT = Decimal("7.0")
DEFAULT_RESEARCH_LAB_CHAMPION_EXTRA_ALPHA_PERCENT_PER_POINT = Decimal("0.3")
DEFAULT_RESEARCH_LAB_CHAMPION_MAX_ALPHA_PERCENT = Decimal("15.0")
DEFAULT_RESEARCH_LAB_CHAMPION_PLACEHOLDER_ALPHA_PERCENT = Decimal("0.0001")
DEFAULT_RESEARCH_LAB_CHAMPION_QUEUE_TRIGGER_RATIO = Decimal("0.50")
DEFAULT_RESEARCH_LAB_CHAMPION_THRESHOLD_POINTS = Decimal("1.0")
DEFAULT_USD_PER_0_1_PERCENT_EPOCH = Decimal("0.162")
DEFAULT_REIMBURSEMENT_MAX_COST_MULTIPLIER_WITH_CHAMPIONS = Decimal("2.0")
CHAMPION_CREDIT_POLICY_ACCELERATED_LIFETIME_CAP_V1 = (
    "accelerated_lifetime_cap_v1"
)


def canonical_json(data: Any) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_json(data: Any) -> str:
    return "sha256:" + hashlib.sha256(canonical_json(data).encode("utf-8")).hexdigest()


def usd_to_microusd(value: Any) -> int:
    return int((_decimal(value) * MICRO_USD).quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def microusd_to_usd(value: int) -> float:
    return round(int(value) / 1_000_000, 6)


def allocate_research_lab_epoch(
    epoch: int,
    policy: Mapping[str, Any],
    active_reimbursement_obligations: Sequence[Mapping[str, Any]],
    active_champion_obligations: Sequence[Mapping[str, Any]],
    *,
    fallback_reimbursement_obligations: Sequence[Mapping[str, Any]] = (),
) -> Dict[str, Any]:
    """Allocate the Research Lab emission slice for one epoch.

    Inputs are public/anchored obligation records. The output is deterministic
    and intended to be stored as the per-epoch Lab allocation snapshot.

    Arena champion and reimbursement obligations share the configured Lab cap.
    """
    epoch = int(epoch)
    lab_cap = _decimal(policy.get("research_lab_emission_percent", DEFAULT_RESEARCH_LAB_EMISSION_PERCENT))
    if lab_cap < 0 or lab_cap > 100:
        raise ValueError("research_lab_emission_percent must be between 0 and 100")

    reward_epochs = int(policy.get("reward_epochs", policy.get("reimbursement_epochs", DEFAULT_RESEARCH_LAB_REWARD_EPOCHS)))
    if reward_epochs <= 0:
        raise ValueError("reward_epochs must be positive")

    champion_credit_policy = _champion_credit_policy(policy)
    input_payload: Dict[str, Any] = {
        "epoch": epoch,
        "policy": _sorted_public(policy),
        "reimbursement_obligations": _sorted_public(active_reimbursement_obligations),
        "champion_obligations": _sorted_public(active_champion_obligations),
    }
    if champion_credit_policy is not None:
        input_payload["champion_credit_policy"] = champion_credit_policy
    if fallback_reimbursement_obligations:
        input_payload["fallback_reimbursement_obligations"] = _sorted_public(
            fallback_reimbursement_obligations
        )
    input_hash = sha256_json(input_payload)

    champions = [
        _normalize_champion_obligation(item, epoch=epoch, policy=policy)
        for item in active_champion_obligations
        if _champion_obligation_active(item, epoch, default_epoch_count=reward_epochs)
    ]
    champions = [item for item in champions if item["desired_alpha_percent"] > 0 and item["uid"] >= 0]
    champions.sort(key=lambda item: (item["start_epoch"], -item["improvement_points"], item["source_id"]))

    reimbursements = [
        _normalize_reimbursement_obligation(
            item,
            epoch=epoch,
            policy=policy,
            champions_active=bool(champions),
        )
        for item in active_reimbursement_obligations
        if _obligation_active(item, epoch, default_epoch_count=reward_epochs)
    ]
    reimbursements = [
        item
        for item in reimbursements
        if item["spend_microusd"] > 0 and item["uid"] >= 0
    ]
    fallback_reimbursements = [
        _normalize_fallback_reimbursement_obligation(item)
        for item in fallback_reimbursement_obligations
    ]
    fallback_reimbursements = [
        item
        for item in fallback_reimbursements
        if item["spend_microusd"] > 0 and item["uid"] >= 0
    ]
    fallback_sources = {
        (
            int(item["source_allocation_epoch"]),
            str(item["source_allocation_hash"]),
            int(item["fallback_window_start_epoch"]),
            int(item["fallback_window_end_epoch"]),
        )
        for item in fallback_reimbursements
    }
    if len(fallback_sources) > 1:
        raise ValueError(
            "historical compute fallback obligations use different authorities"
        )
    conservative = _conservative_enabled(policy)
    reimbursement_allocations: list[Dict[str, Any]] = []
    champion_allocations: list[Dict[str, Any]] = []
    queued_champion_allocations: list[Dict[str, Any]] = []

    if (
        not reimbursements
        and not champions
        and (conservative or not fallback_reimbursements)
    ):
        if not conservative and lab_cap > 0:
            raise ValueError(
                "non-conservative allocation requires historical compute fallback"
            )
        result = {
            "epoch": epoch,
            "lab_cap_percent": _rate_float(lab_cap),
            "reimbursement_allocations": [],
            "champion_allocations": [],
            "queued_champion_allocations": [],
            "reimbursement_alpha_percent": 0.0,
            "champion_alpha_percent": 0.0,
            "queued_champion_alpha_percent": 0.0,
            "unallocated_percent": _rate_float(lab_cap),
            "input_hash": input_hash,
        }
        if champion_credit_policy is not None:
            result["champion_credit_policy"] = champion_credit_policy
        _add_explicit_policy_modes(result, policy)
        return {**result, "allocation_hash": sha256_json(result)}

    reimbursement_paid = Decimal("0")
    if reimbursements:
        reimbursement_pool = (
            _reimbursement_pool_with_champions(champions, lab_cap, policy)
            if champions
            else lab_cap
        )
        reimbursement_allocations = _allocate_reimbursements_at_set_rate(
            reimbursements,
            reimbursement_pool,
        )
        _cap_allocation_sections_to_pool(
            (reimbursement_allocations,),
            reimbursement_pool,
        )
        reimbursement_paid = sum(_decimal(item["paid_alpha_percent"]) for item in reimbursement_allocations)

    remaining_for_champions = max(Decimal("0"), lab_cap - reimbursement_paid)
    if champions:
        champion_allocations, queued_champion_allocations = _allocate_champions(
            champions,
            remaining_for_champions,
            policy,
            reimbursement_paid=reimbursement_paid,
        )

    champion_paid = sum(
        (_decimal(item["paid_alpha_percent"]) for item in champion_allocations),
        Decimal("0"),
    )
    queued_paid = sum(
        (_decimal(item["paid_alpha_percent"]) for item in queued_champion_allocations),
        Decimal("0"),
    )
    reimbursement_surplus = max(
        Decimal("0"),
        lab_cap.quantize(RATE_QUANT, rounding=ROUND_HALF_UP)
        - reimbursement_paid
        - champion_paid
        - queued_paid,
    )
    if (
        reimbursement_surplus > 0
        and reimbursement_allocations
        and (not champions or conservative)
    ):
        _distribute_reimbursement_surplus(
            reimbursements,
            reimbursement_allocations,
            reimbursement_surplus,
        )
    elif (
        reimbursement_surplus > 0
        and not conservative
        and fallback_reimbursements
    ):
        reimbursement_allocations.extend(
            _allocate_fallback_reimbursements(
                fallback_reimbursements,
                reimbursement_surplus,
            )
        )

    _cap_allocation_sections_to_pool(
        (reimbursement_allocations, champion_allocations, queued_champion_allocations),
        lab_cap,
    )
    reimbursement_paid = sum((_decimal(item["paid_alpha_percent"]) for item in reimbursement_allocations), Decimal("0"))
    champion_paid = sum((_decimal(item["paid_alpha_percent"]) for item in champion_allocations), Decimal("0"))
    queued_paid = sum((_decimal(item["paid_alpha_percent"]) for item in queued_champion_allocations), Decimal("0"))
    total_paid = reimbursement_paid + champion_paid + queued_paid
    allocation_cap = (
        lab_cap.quantize(RATE_QUANT, rounding=ROUND_HALF_UP)
        if not conservative
        else lab_cap
    )
    unallocated = max(Decimal("0"), allocation_cap - total_paid)
    if not conservative and unallocated > 0:
        raise ValueError(
            "non-conservative allocation left Research Lab emissions unallocated"
        )
    result = {
        "epoch": epoch,
        "lab_cap_percent": _rate_float(lab_cap),
        "reimbursement_allocations": reimbursement_allocations,
        "champion_allocations": champion_allocations,
        "queued_champion_allocations": queued_champion_allocations,
        "reimbursement_alpha_percent": _rate_float(reimbursement_paid),
        "champion_alpha_percent": _rate_float(champion_paid),
        "queued_champion_alpha_percent": _rate_float(queued_paid),
        "unallocated_percent": _rate_float(unallocated),
        "input_hash": input_hash,
    }
    if champion_credit_policy is not None:
        result["champion_credit_policy"] = champion_credit_policy
    if any(
        item.get("reason") == "historical_compute_fallback_no_burn"
        for item in reimbursement_allocations
    ):
        (
            source_epoch,
            source_hash,
            window_start,
            window_end,
        ) = next(iter(fallback_sources))
        result.update(
            {
                "historical_compute_fallback_source_epoch": source_epoch,
                "historical_compute_fallback_source_allocation_hash": source_hash,
                "historical_compute_fallback_window_start_epoch": window_start,
                "historical_compute_fallback_window_end_epoch": window_end,
            }
        )
    _add_explicit_policy_modes(result, policy)
    return {**result, "allocation_hash": sha256_json(result)}


def _policy_boolean(
    policy: Mapping[str, Any],
    key: str,
    *,
    historical_default: bool,
) -> bool:
    if key not in policy:
        return historical_default
    value = policy[key]
    if not isinstance(value, bool):
        raise ValueError("%s must be a boolean" % key)
    return value


def _conservative_enabled(policy: Mapping[str, Any]) -> bool:
    # Old signed bundles predate this flag and must continue to replay with
    # their original burn-permitting behavior.
    return _policy_boolean(
        policy,
        "enable_conservative",
        historical_default=True,
    )


def _champ_cap_enabled(policy: Mapping[str, Any]) -> bool:
    # Historical allocations used accelerated lifetime retirement. New config
    # documents explicitly disable it so a champion remains eligible for its
    # full nominal window.
    return _policy_boolean(
        policy,
        "enable_champ_cap",
        historical_default=True,
    )


def _champion_credit_policy(policy: Mapping[str, Any]) -> Optional[str]:
    if _champ_cap_enabled(policy):
        return CHAMPION_CREDIT_POLICY_ACCELERATED_LIFETIME_CAP_V1
    return None


def _add_explicit_policy_modes(
    output: Dict[str, Any],
    policy: Mapping[str, Any],
) -> None:
    if "enable_conservative" in policy:
        output["conservative_mode"] = _conservative_enabled(policy)
    if "enable_champ_cap" in policy:
        output["champion_cap_enabled"] = _champ_cap_enabled(policy)


def _obligation_active(obligation: Mapping[str, Any], epoch: int, *, default_epoch_count: int) -> bool:
    status = str(obligation.get("status", obligation.get("schedule_status", "active")))
    if status in {"empty", "disabled", "ineligible", "blocked", "voided", "tombstoned", "completed"}:
        return False
    start_epoch = int(obligation.get("start_epoch", obligation.get("grant_start_epoch", epoch)))
    epoch_count = int(
        obligation.get(
            "epoch_count",
            obligation.get("reimbursement_epochs", obligation.get("reward_epochs", default_epoch_count)),
        )
    )
    return epoch_count > 0 and start_epoch <= int(epoch) < start_epoch + epoch_count


def _champion_obligation_active(obligation: Mapping[str, Any], epoch: int, *, default_epoch_count: int) -> bool:
    status = str(obligation.get("status", obligation.get("schedule_status", "active")))
    if status in {"empty", "disabled", "ineligible", "blocked", "voided", "tombstoned", "completed", "paid"}:
        return False
    start_epoch = int(obligation.get("start_epoch", obligation.get("grant_start_epoch", epoch)))
    if int(epoch) < start_epoch:
        return False
    replay_keys = {"remaining_alpha_percent", "paid_alpha_percent_to_date", "total_due_alpha_percent"}
    if replay_keys.intersection(obligation.keys()):
        epoch_count = int(
            obligation.get(
                "epoch_count",
                obligation.get("reward_epochs", default_epoch_count),
            )
        )
        if (
            obligation.get("champ_cap_enabled") is False
            and epoch_count > 0
            and int(epoch) < start_epoch + epoch_count
        ):
            return True
        remaining = _champion_remaining_alpha_percent(obligation, policy=None, default_epoch_count=default_epoch_count)
        return remaining > 0
    epoch_count = int(obligation.get("epoch_count", obligation.get("reward_epochs", default_epoch_count)))
    return epoch_count > 0 and int(epoch) < start_epoch + epoch_count


def _normalize_reimbursement_obligation(
    obligation: Mapping[str, Any],
    *,
    epoch: int,
    policy: Mapping[str, Any],
    champions_active: bool,
) -> Dict[str, Any]:
    reward_epochs = int(policy.get("reward_epochs", policy.get("reimbursement_epochs", DEFAULT_RESEARCH_LAB_REWARD_EPOCHS)))
    start_epoch = int(obligation.get("start_epoch", epoch))
    epoch_count = int(obligation.get("epoch_count", obligation.get("reimbursement_epochs", reward_epochs)))
    spend = _obligation_spend_microusd(obligation)
    eligible_compute_explicit = (
        "eligible_compute_microusd" in obligation
        or "eligible_cost_microusd" in obligation
    )
    eligible_compute = int(
        obligation.get("eligible_compute_microusd")
        or obligation.get("eligible_cost_microusd")
        or spend
    )
    if eligible_compute < 0:
        raise ValueError("eligible_compute_microusd must be non-negative")
    max_multiplier = _decimal(
        policy.get(
            "reimbursement_max_cost_multiplier_with_champions",
            "1.0",
        )
    )
    if max_multiplier < 0:
        raise ValueError("reimbursement_max_cost_multiplier_with_champions must be non-negative")
    intended = _alpha_percent_for_microusd(
        _round_microusd(
            Decimal(spend) / Decimal(max(1, epoch_count))
        ),
        policy,
    )
    applied_multiplier = (
        min(
            max_multiplier,
            DEFAULT_REIMBURSEMENT_MAX_COST_MULTIPLIER_WITH_CHAMPIONS,
        )
        if champions_active
        else Decimal("1.0")
    )
    compute_cap = _alpha_percent_for_microusd(
        _round_microusd(
            Decimal(eligible_compute) / Decimal(max(1, epoch_count))
        ),
        policy,
    ) * applied_multiplier
    capped_intended = min(intended * applied_multiplier, compute_cap)
    weight = Decimal(max(eligible_compute, 0)) * _decimal(
        obligation.get("island_weight", obligation.get("participation_weight", obligation.get("reimbursement_weight", 1)))
    )
    normalized = {
        "uid": int(obligation.get("uid", obligation.get("miner_uid", -1))),
        "miner_hotkey": str(obligation.get("miner_hotkey", "")),
        "source_id": str(
            obligation.get("source_id")
            or obligation.get("schedule_id")
            or obligation.get("award_id")
            or obligation.get("run_id")
            or ""
        ),
        "island": str(obligation.get("island", "generalist")),
        "start_epoch": start_epoch,
        "epoch_count": epoch_count,
        "spend_microusd": spend,
        "spend_usd": microusd_to_usd(spend),
        "island_weight": _rate_float(_decimal(obligation.get("island_weight", obligation.get("participation_weight", 1)))),
        "intended_alpha_percent": capped_intended,
        "pro_rata_weight": max(Decimal("0"), weight),
    }
    if eligible_compute_explicit:
        normalized.update(
            {
                "eligible_compute_microusd": eligible_compute,
                "eligible_compute_usd": microusd_to_usd(eligible_compute),
            }
        )
    return normalized


def _normalize_fallback_reimbursement_obligation(
    obligation: Mapping[str, Any],
) -> Dict[str, Any]:
    spend = _obligation_spend_microusd(obligation)
    return {
        "uid": int(obligation.get("uid", obligation.get("miner_uid", -1))),
        "miner_hotkey": str(obligation.get("miner_hotkey", "")),
        "source_id": str(obligation.get("source_id") or ""),
        "island": str(obligation.get("island", "historical_compute")),
        "spend_microusd": spend,
        "spend_usd": microusd_to_usd(spend),
        "island_weight": 1.0,
        "pro_rata_weight": Decimal(max(spend, 0)),
        "fallback_window_start_epoch": int(
            obligation.get("fallback_window_start_epoch", 0)
        ),
        "fallback_window_end_epoch": int(
            obligation.get("fallback_window_end_epoch", 0)
        ),
        "contribution_count": int(obligation.get("contribution_count", 0)),
        "contribution_hash": str(obligation.get("contribution_hash") or ""),
        "source_allocation_epoch": int(
            obligation.get("source_allocation_epoch", 0)
        ),
        "source_allocation_hash": str(
            obligation.get("source_allocation_hash") or ""
        ),
    }


def _obligation_spend_microusd(obligation: Mapping[str, Any]) -> int:
    for key in (
        "actual_openrouter_cost_microusd",
        "miner_openrouter_cost_microusd",
        "eligible_cost_microusd",
        "target_reimbursement_microusd",
        "total_microusd",
    ):
        if key in obligation:
            return max(0, int(obligation.get(key) or 0))
    for key in (
        "actual_openrouter_cost_usd",
        "miner_openrouter_cost_usd",
        "eligible_cost_usd",
        "target_reimbursement_usd",
        "total_usd",
    ):
        if key in obligation:
            return max(0, usd_to_microusd(obligation.get(key) or 0))
    return 0


def _normalize_champion_obligation(
    obligation: Mapping[str, Any],
    *,
    epoch: int,
    policy: Mapping[str, Any],
) -> Dict[str, Any]:
    reward_epochs = int(policy.get("reward_epochs", DEFAULT_RESEARCH_LAB_REWARD_EPOCHS))
    start_epoch = int(obligation.get("start_epoch", obligation.get("grant_start_epoch", epoch)))
    epoch_count = int(obligation.get("epoch_count", obligation.get("reward_epochs", reward_epochs)))
    improvement_points = _decimal(
        obligation.get(
            "improvement_points",
            obligation.get("score_delta", obligation.get("delta", obligation.get("mean_delta", 0))),
        )
    )
    base_desired = _champion_desired_alpha_percent(obligation, policy)
    remaining = _champion_remaining_alpha_percent(obligation, policy=policy, default_epoch_count=reward_epochs)
    champ_cap_enabled = _champ_cap_enabled(policy)
    nominal_end_epoch = start_epoch + epoch_count
    desired = (
        min(base_desired, remaining)
        if champ_cap_enabled
        else base_desired
    )
    total_due = _champion_total_due_alpha_percent(obligation, base_desired=base_desired, epoch_count=epoch_count)
    paid_to_date = _champion_paid_alpha_percent_to_date(obligation, total_due=total_due, remaining=remaining)
    expected_total_due = base_desired * Decimal(max(0, epoch_count))
    if total_due.quantize(RATE_QUANT, rounding=ROUND_HALF_UP) != (
        expected_total_due.quantize(RATE_QUANT, rounding=ROUND_HALF_UP)
    ):
        raise ValueError(
            "champion total_due_alpha_percent differs from lifetime entitlement"
        )
    if (
        paid_to_date > total_due
        or (
            paid_to_date + remaining
        ).quantize(RATE_QUANT, rounding=ROUND_HALF_UP)
        != total_due.quantize(RATE_QUANT, rounding=ROUND_HALF_UP)
    ):
        raise ValueError("champion lifetime balance is inconsistent")
    normalized: Dict[str, Any] = {
        "uid": int(obligation.get("uid", obligation.get("miner_uid", -1))),
        "miner_hotkey": str(obligation.get("miner_hotkey", "")),
        "source_id": str(
            obligation.get("source_id")
            or obligation.get("champion_reward_id")
            or obligation.get("grant_id")
            or obligation.get("candidate_id")
            or obligation.get("score_bundle_id")
            or ""
        ),
        "island": str(obligation.get("island", "generalist")),
        "start_epoch": start_epoch,
        "epoch_count": epoch_count,
        "improvement_points": improvement_points,
        "intended_alpha_percent": desired,
        "desired_alpha_percent": desired,
        "base_desired_alpha_percent": base_desired,
        "total_due_alpha_percent": total_due,
        "paid_alpha_percent_to_date": paid_to_date,
        "remaining_alpha_percent": remaining,
        "nominal_end_epoch": nominal_end_epoch,
        "champ_cap_enabled": champ_cap_enabled,
        "champ_cap_policy_explicit": "enable_champ_cap" in policy,
    }
    if obligation.get("replay_status") is not None:
        normalized["replay_status"] = str(obligation.get("replay_status") or "")
    return normalized






def _champion_total_due_alpha_percent(
    obligation: Mapping[str, Any],
    *,
    base_desired: Decimal,
    epoch_count: int,
) -> Decimal:
    if "total_due_alpha_percent" in obligation:
        return max(Decimal("0"), _decimal(obligation["total_due_alpha_percent"]))
    return max(Decimal("0"), base_desired) * Decimal(max(0, int(epoch_count)))


def _champion_paid_alpha_percent_to_date(
    obligation: Mapping[str, Any],
    *,
    total_due: Decimal,
    remaining: Decimal,
) -> Decimal:
    if "paid_alpha_percent_to_date" in obligation:
        return max(Decimal("0"), _decimal(obligation["paid_alpha_percent_to_date"]))
    if "remaining_alpha_percent" in obligation:
        return max(Decimal("0"), total_due - max(Decimal("0"), _decimal(obligation["remaining_alpha_percent"])))
    return Decimal("0")


def _champion_remaining_alpha_percent(
    obligation: Mapping[str, Any],
    *,
    policy: Mapping[str, Any] | None,
    default_epoch_count: int,
) -> Decimal:
    epoch_count = int(obligation.get("epoch_count", obligation.get("reward_epochs", default_epoch_count)))
    if policy is None:
        if "desired_alpha_percent" in obligation:
            base_desired = max(Decimal("0"), _decimal(obligation["desired_alpha_percent"]))
        else:
            base_desired = Decimal("0")
    else:
        base_desired = _champion_desired_alpha_percent(obligation, policy)
    total_due = _champion_total_due_alpha_percent(obligation, base_desired=base_desired, epoch_count=epoch_count)
    if "remaining_alpha_percent" in obligation:
        return _clamp(_decimal(obligation["remaining_alpha_percent"]), Decimal("0"), total_due)
    paid_to_date = max(Decimal("0"), _decimal(obligation.get("paid_alpha_percent_to_date", 0)))
    return max(Decimal("0"), total_due - paid_to_date)


def _champion_desired_alpha_percent(obligation: Mapping[str, Any], policy: Mapping[str, Any]) -> Decimal:
    if "desired_alpha_percent" in obligation:
        return max(Decimal("0"), _decimal(obligation["desired_alpha_percent"]))
    threshold = _decimal(policy.get("champion_threshold_points", DEFAULT_RESEARCH_LAB_CHAMPION_THRESHOLD_POINTS))
    minimum = _decimal(policy.get("champion_min_alpha_percent", DEFAULT_RESEARCH_LAB_CHAMPION_MIN_ALPHA_PERCENT))
    increment = _decimal(
        policy.get("champion_extra_alpha_percent_per_point", DEFAULT_RESEARCH_LAB_CHAMPION_EXTRA_ALPHA_PERCENT_PER_POINT)
    )
    maximum = _decimal(policy.get("champion_max_alpha_percent", DEFAULT_RESEARCH_LAB_CHAMPION_MAX_ALPHA_PERCENT))
    points = _decimal(
        obligation.get(
            "improvement_points",
            obligation.get("score_delta", obligation.get("delta", obligation.get("mean_delta", 0))),
        )
    )
    if points < threshold:
        return Decimal("0")
    return _clamp(minimum + (points - threshold) * increment, Decimal("0"), maximum)


def _alpha_percent_for_microusd(amount_microusd: int, policy: Mapping[str, Any]) -> Decimal:
    valuation = _decimal(
        policy.get(
            "usd_per_0_1_percent_epoch",
            policy.get("reimbursement_usd_per_0_1_percent_epoch", DEFAULT_USD_PER_0_1_PERCENT_EPOCH),
        )
    )
    if valuation <= 0:
        raise ValueError("usd_per_0_1_percent_epoch must be positive")
    amount_usd = Decimal(max(0, amount_microusd)) / MICRO_USD
    return (amount_usd / valuation) * Decimal("0.1")


def _minimum_champion_reserve(
    champions: Sequence[Mapping[str, Any]],
    lab_cap: Decimal,
    policy: Mapping[str, Any],
) -> Decimal:
    if not champions:
        return Decimal("0")
    placeholder = _decimal(
        policy.get("champion_placeholder_alpha_percent", DEFAULT_RESEARCH_LAB_CHAMPION_PLACEHOLDER_ALPHA_PERCENT)
    )
    oldest = champions[0]
    reserve = _decimal(oldest["desired_alpha_percent"])
    reserve += placeholder * Decimal(max(0, len(champions) - 1))
    return min(lab_cap, max(Decimal("0"), reserve))


def _reimbursement_pool_with_champions(
    champions: Sequence[Mapping[str, Any]],
    lab_cap: Decimal,
    policy: Mapping[str, Any],
) -> Decimal:
    champion_reserve = _minimum_champion_reserve(champions, lab_cap, policy)
    queue_trigger_ratio = _decimal(
        policy.get("champion_queue_trigger_ratio", DEFAULT_RESEARCH_LAB_CHAMPION_QUEUE_TRIGGER_RATIO)
    )
    if queue_trigger_ratio < 0:
        raise ValueError("champion_queue_trigger_ratio must be non-negative")
    reimbursement_share_cap = lab_cap * min(queue_trigger_ratio, Decimal("1"))
    return min(max(Decimal("0"), lab_cap - champion_reserve), reimbursement_share_cap)


def _allocate_reimbursements_at_set_rate(
    reimbursements: Sequence[Mapping[str, Any]],
    pool: Decimal,
) -> list[Dict[str, Any]]:
    caps = [_decimal(item["intended_alpha_percent"]) for item in reimbursements]
    weights = [_decimal(item["pro_rata_weight"]) for item in reimbursements]
    paid = _allocate_capped_pro_rata(pool, weights, caps)
    return [
        _reimbursement_allocation(
            item,
            amount,
            "full_reimbursement" if amount >= _decimal(item["intended_alpha_percent"]) else "scaled_by_lab_capacity",
        )
        for item, amount in zip(reimbursements, paid)
    ]


def _allocate_pro_rata_exact(
    pool: Decimal,
    weights: Sequence[Decimal],
) -> list[Decimal]:
    target = max(Decimal("0"), pool).quantize(
        RATE_QUANT,
        rounding=ROUND_HALF_UP,
    )
    normalized = [max(Decimal("0"), _decimal(weight)) for weight in weights]
    active_indices = [
        index for index, weight in enumerate(normalized) if weight > 0
    ]
    weight_sum = sum(
        (normalized[index] for index in active_indices),
        Decimal("0"),
    )
    paid = [Decimal("0") for _ in normalized]
    if target <= 0 or weight_sum <= 0:
        return paid

    raw_shares = [
        target * normalized[index] / weight_sum
        for index in active_indices
    ]
    shares = [
        share.quantize(RATE_QUANT, rounding=ROUND_DOWN)
        for share in raw_shares
    ]
    distributed = sum(shares, Decimal("0"))
    residual_units = int(
        ((target - distributed) / RATE_QUANT).to_integral_value(
            rounding=ROUND_HALF_UP
        )
    )
    remainder_order = sorted(
        range(len(active_indices)),
        key=lambda offset: (
            raw_shares[offset] - shares[offset],
            -active_indices[offset],
        ),
        reverse=True,
    )
    for offset in remainder_order[:residual_units]:
        shares[offset] += RATE_QUANT
    for index, share in zip(active_indices, shares):
        paid[index] = share
    if sum(paid, Decimal("0")) != target:
        raise ValueError("pro-rata allocation does not conserve its pool")
    return paid


def _allocate_fallback_reimbursements(
    reimbursements: Sequence[Mapping[str, Any]],
    pool: Decimal,
) -> list[Dict[str, Any]]:
    weights = [
        max(Decimal("0"), _decimal(item["pro_rata_weight"]))
        for item in reimbursements
    ]
    paid = _allocate_pro_rata_exact(pool, weights)
    allocations: list[Dict[str, Any]] = []
    for item, amount in zip(reimbursements, paid):
        if amount <= 0:
            continue
        allocation = _reimbursement_allocation(
            {**dict(item), "intended_alpha_percent": amount},
            amount,
            "historical_compute_fallback_no_burn",
        )
        allocation.update(
            {
                "fallback_window_start_epoch": int(
                    item["fallback_window_start_epoch"]
                ),
                "fallback_window_end_epoch": int(
                    item["fallback_window_end_epoch"]
                ),
                "contribution_count": int(item["contribution_count"]),
                "contribution_hash": str(item["contribution_hash"]),
                "source_allocation_epoch": int(
                    item["source_allocation_epoch"]
                ),
                "source_allocation_hash": str(
                    item["source_allocation_hash"]
                ),
            }
        )
        allocations.append(allocation)
    return allocations


def _distribute_reimbursement_surplus(
    reimbursements: Sequence[Mapping[str, Any]],
    allocations: list[Dict[str, Any]],
    pool: Decimal,
) -> Decimal:
    """Give otherwise-unused Lab capacity to active compute reimbursements."""
    if len(reimbursements) != len(allocations):
        raise ValueError("reimbursement inputs and allocations must align")

    weights = [
        max(Decimal("0"), _decimal(item["pro_rata_weight"]))
        for item in reimbursements
    ]
    shares = _allocate_pro_rata_exact(pool, weights)
    for index, extra in enumerate(shares):
        if extra <= 0:
            continue
        row = allocations[index]
        intended = _decimal(row["intended_alpha_percent"])
        paid = (
            _decimal(row["paid_alpha_percent"]) + extra
        ).quantize(RATE_QUANT, rounding=ROUND_HALF_UP)
        row["paid_alpha_percent"] = _rate_float(paid)
        row["deferred_alpha_percent"] = _rate_float(
            max(Decimal("0"), intended - paid)
        )
        row["overpaid_alpha_percent"] = _rate_float(
            max(Decimal("0"), paid - intended)
        )
        row["reason"] = "surplus_reimbursement_no_burn"

    return sum(shares, Decimal("0"))


def _cap_allocation_sections_to_pool(
    sections: Sequence[list[Dict[str, Any]]],
    pool: Decimal,
) -> None:
    cap = max(Decimal("0"), pool).quantize(RATE_QUANT, rounding=ROUND_HALF_UP)
    paid_total = sum(
        (_decimal(row.get("paid_alpha_percent", 0)) for rows in sections for row in rows),
        Decimal("0"),
    )
    overflow = paid_total - cap
    if overflow <= 0:
        return

    for rows in reversed(sections):
        for row in reversed(rows):
            paid = _decimal(row.get("paid_alpha_percent", 0))
            if paid <= 0:
                continue
            reduction = min(paid, overflow)
            new_paid = (paid - reduction).quantize(RATE_QUANT, rounding=ROUND_HALF_UP)
            row["paid_alpha_percent"] = _rate_float(new_paid)
            intended = _decimal(row.get("intended_alpha_percent", 0))
            if "deferred_alpha_percent" in row:
                row["deferred_alpha_percent"] = _rate_float(max(Decimal("0"), intended - new_paid))
            if "remaining_alpha_percent_before_epoch" in row:
                remaining_before = _decimal(
                    row["remaining_alpha_percent_before_epoch"]
                )
                row["remaining_alpha_percent_after_epoch"] = _rate_float(
                    max(Decimal("0"), remaining_before - new_paid)
                )
            if "overpaid_alpha_percent" in row:
                row["overpaid_alpha_percent"] = _rate_float(max(Decimal("0"), new_paid - intended))
            overflow -= reduction
            if overflow <= 0:
                return


def _allocate_champions(
    champions: Sequence[Mapping[str, Any]],
    pool: Decimal,
    policy: Mapping[str, Any],
    *,
    reimbursement_paid: Decimal = Decimal("0"),
) -> tuple[list[Dict[str, Any]], list[Dict[str, Any]]]:
    if not _champ_cap_enabled(policy):
        return _allocate_champions_minimum_window(champions, pool, policy)

    placeholder = _decimal(
        policy.get("champion_placeholder_alpha_percent", DEFAULT_RESEARCH_LAB_CHAMPION_PLACEHOLDER_ALPHA_PERCENT)
    )
    total_pool = max(Decimal("0"), pool)
    paid = [Decimal("0") for _ in champions]
    active_indices: list[int] = []
    queued_indices: list[int] = []
    remaining_pool = total_pool
    for index, champion in enumerate(champions):
        desired = _decimal(champion["desired_alpha_percent"])
        if desired <= 0:
            queued_indices.append(index)
            continue
        amount = min(desired, remaining_pool)
        paid[index] = amount
        remaining_pool -= amount
        if amount >= desired:
            active_indices.append(index)
        else:
            queued_indices.append(index)

    # Once chronological per-epoch dues are funded, genuine champions share
    # surplus by improvement points. Each share remains capped by the
    # champion's verified lifetime balance; capacity released by one capped
    # champion is redistributed among the other eligible champions.
    surplus_indices = active_indices
    if remaining_pool > 0 and surplus_indices:
        weights = [
            max(Decimal("0"), _decimal(champions[index].get("improvement_points", 0)))
            for index in surplus_indices
        ]
        weight_sum = sum(weights, Decimal("0"))
        if weight_sum <= 0:
            weights = [_decimal(champions[index]["desired_alpha_percent"]) for index in surplus_indices]
            weight_sum = sum(weights, Decimal("0"))
        if weight_sum <= 0:
            weights = [Decimal("1") for _ in surplus_indices]
        caps = [
            max(
                Decimal("0"),
                _decimal(champions[index]["remaining_alpha_percent"])
                - paid[index],
            )
            for index in surplus_indices
        ]
        surplus_paid = _allocate_capped_pro_rata(
            remaining_pool,
            weights,
            caps,
        )
        for index, amount in zip(surplus_indices, surplus_paid):
            paid[index] += amount
        remaining_pool -= sum(surplus_paid, Decimal("0"))

    active: list[Dict[str, Any]] = []
    queued: list[Dict[str, Any]] = []
    for champion, amount in zip(champions, paid):
        allocation = _champion_allocation(champion, amount)
        if amount >= _decimal(champion["desired_alpha_percent"]):
            active.append({**allocation, "reason": "active_champion_reward"})
        elif amount > 0:
            reason = "queued_with_placeholder" if amount <= placeholder else "queued_with_partial_capacity"
            queued.append({**allocation, "reason": reason})
        else:
            queued.append({**allocation, "reason": "queued_no_capacity"})
    return active, queued


def _allocate_champions_minimum_window(
    champions: Sequence[Mapping[str, Any]],
    pool: Decimal,
    policy: Mapping[str, Any],
) -> tuple[list[Dict[str, Any]], list[Dict[str, Any]]]:
    placeholder = _decimal(
        policy.get(
            "champion_placeholder_alpha_percent",
            DEFAULT_RESEARCH_LAB_CHAMPION_PLACEHOLDER_ALPHA_PERCENT,
        )
    )
    weights = [
        max(
            Decimal("0"),
            _decimal(
                champion.get(
                    "base_desired_alpha_percent",
                    champion.get("desired_alpha_percent", 0),
                )
            ),
        )
        for champion in champions
    ]
    paid = _allocate_pro_rata_exact(pool, weights)
    active: list[Dict[str, Any]] = []
    queued: list[Dict[str, Any]] = []
    for champion, amount in zip(champions, paid):
        allocation = _champion_allocation(champion, amount)
        desired = _decimal(champion["desired_alpha_percent"])
        if amount >= desired:
            active.append({**allocation, "reason": "active_champion_reward"})
        elif amount > 0:
            reason = (
                "queued_with_placeholder"
                if amount <= placeholder
                else "queued_with_partial_capacity"
            )
            queued.append({**allocation, "reason": reason})
        else:
            queued.append({**allocation, "reason": "queued_no_capacity"})
    return active, queued




def _allocate_capped_pro_rata(pool: Decimal, weights: Sequence[Decimal], caps: Sequence[Decimal]) -> list[Decimal]:
    paid = [Decimal("0") for _ in weights]
    remaining_indices = {idx for idx, cap in enumerate(caps) if cap > 0 and weights[idx] > 0}
    remaining_pool = max(Decimal("0"), pool)
    while remaining_indices and remaining_pool > 0:
        weight_sum = sum(weights[idx] for idx in remaining_indices)
        if weight_sum <= 0:
            break
        progressed = False
        for idx in list(sorted(remaining_indices)):
            share = remaining_pool * weights[idx] / weight_sum
            room = caps[idx] - paid[idx]
            if share >= room:
                paid[idx] += room
                remaining_pool -= room
                remaining_indices.remove(idx)
                progressed = True
        if not progressed:
            for idx in sorted(remaining_indices):
                share = remaining_pool * weights[idx] / weight_sum
                paid[idx] += share
            break
    return paid


def _reimbursement_allocation(item: Mapping[str, Any], paid: Decimal, reason: str) -> Dict[str, Any]:
    intended = _decimal(item["intended_alpha_percent"])
    deferred = max(Decimal("0"), intended - paid)
    overpaid = max(Decimal("0"), paid - intended)
    allocation = {
        "uid": int(item["uid"]),
        "miner_hotkey": str(item["miner_hotkey"]),
        "source_id": str(item["source_id"]),
        "island": str(item["island"]),
        "intended_alpha_percent": _rate_float(intended),
        "paid_alpha_percent": _rate_float(paid),
        "deferred_alpha_percent": _rate_float(deferred),
        "overpaid_alpha_percent": _rate_float(overpaid),
        "spend_microusd": int(item["spend_microusd"]),
        "spend_usd": microusd_to_usd(int(item["spend_microusd"])),
        "island_weight": item["island_weight"],
        "reason": reason,
    }
    if "eligible_compute_microusd" in item:
        eligible_compute = int(item["eligible_compute_microusd"])
        allocation.update(
            {
                "eligible_compute_microusd": eligible_compute,
                "eligible_compute_usd": microusd_to_usd(eligible_compute),
            }
        )
    return allocation


def _champion_allocation(item: Mapping[str, Any], paid: Decimal) -> Dict[str, Any]:
    intended = _decimal(item["desired_alpha_percent"]).quantize(
        RATE_QUANT,
        rounding=ROUND_HALF_UP,
    )
    paid = max(Decimal("0"), paid).quantize(
        RATE_QUANT,
        rounding=ROUND_HALF_UP,
    )
    allocation: Dict[str, Any] = {
        "uid": int(item["uid"]),
        "miner_hotkey": str(item["miner_hotkey"]),
        "source_id": str(item["source_id"]),
        "island": str(item["island"]),
        "intended_alpha_percent": float(intended),
        "paid_alpha_percent": float(paid),
        "deferred_alpha_percent": _rate_float(
            max(Decimal("0"), intended - paid)
        ),
        "improvement_points": _rate_float(_decimal(item["improvement_points"])),
    }
    if "base_desired_alpha_percent" in item:
        total_due = _decimal(item["total_due_alpha_percent"]).quantize(
            RATE_QUANT,
            rounding=ROUND_HALF_UP,
        )
        paid_to_date = _decimal(
            item["paid_alpha_percent_to_date"]
        ).quantize(
            RATE_QUANT,
            rounding=ROUND_HALF_UP,
        )
        remaining_before = max(
            Decimal("0"),
            total_due - paid_to_date,
        ).quantize(RATE_QUANT, rounding=ROUND_HALF_UP)
        credited = min(paid, remaining_before)
        if bool(item.get("champ_cap_enabled", True)):
            paid = credited
        remaining_after = (remaining_before - credited).quantize(
            RATE_QUANT,
            rounding=ROUND_HALF_UP,
        )
        allocation["paid_alpha_percent"] = float(paid)
        allocation["deferred_alpha_percent"] = _rate_float(
            max(Decimal("0"), intended - paid)
        )
        allocation.update(
            {
                "base_desired_alpha_percent": _rate_float(_decimal(item["base_desired_alpha_percent"])),
                "total_due_alpha_percent": float(total_due),
                "paid_alpha_percent_to_date": float(paid_to_date),
                "remaining_alpha_percent_before_epoch": float(remaining_before),
                "remaining_alpha_percent_after_epoch": float(remaining_after),
                "nominal_end_epoch": int(item.get("nominal_end_epoch", 0)),
            }
        )
        if item.get("champ_cap_policy_explicit"):
            allocation["champion_cap_enabled"] = bool(
                item.get("champ_cap_enabled")
            )
        if item.get("replay_status") is not None:
            allocation["replay_status"] = str(item.get("replay_status") or "")
    return allocation


def _decimal(value: Any) -> Decimal:
    return Decimal(str(value))


def _rate_float(value: Decimal) -> float:
    return float(value.quantize(RATE_QUANT, rounding=ROUND_HALF_UP))


def _money_float(value: Decimal) -> float:
    return float(value.quantize(Decimal("0.000001"), rounding=ROUND_HALF_UP))


def _round_microusd(value: Decimal) -> int:
    return int(value.quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def _clamp(value: Decimal, low: Decimal, high: Decimal) -> Decimal:
    return max(low, min(high, value))


def _sorted_public(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _sorted_public(nested) for key, nested in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple)):
        return [_sorted_public(item) for item in value]
    return value
