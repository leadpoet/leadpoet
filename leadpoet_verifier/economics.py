"""Deterministic Research Lab economics for the open verifier.

This module keeps validator-replayable arithmetic independent from private
champion code, sealed judges, gateway services, provider APIs, and production
weight submission. All inputs are anchored/public records or golden fixtures.
"""

from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP
import hashlib
import json
from typing import Any, Dict, Iterable, Mapping, Sequence

from .aggregation import u16_weights_from_scores


MICRO_USD = Decimal("1000000")
RATE_QUANT = Decimal("0.000001")

VALID_TRANSFER_CALLS = (
    "transfer",
    "transfer_keep_alive",
    "transfer_allow_death",
    "transfer_all",
)

DEFAULT_RESEARCH_LAB_EMISSION_PERCENT = Decimal("20.0")
DEFAULT_RESEARCH_LAB_REWARD_EPOCHS = 20
DEFAULT_RESEARCH_LAB_CHAMPION_MIN_ALPHA_PERCENT = Decimal("4.0")
DEFAULT_RESEARCH_LAB_CHAMPION_EXTRA_ALPHA_PERCENT_PER_POINT = Decimal("0.2")
DEFAULT_RESEARCH_LAB_CHAMPION_MAX_ALPHA_PERCENT = Decimal("10.0")
DEFAULT_RESEARCH_LAB_CHAMPION_PLACEHOLDER_ALPHA_PERCENT = Decimal("0.0001")
DEFAULT_RESEARCH_LAB_CHAMPION_THRESHOLD_POINTS = Decimal("1.0")
DEFAULT_RESEARCH_LAB_CHAMPION_EVAL_DAYS = 10
DEFAULT_RESEARCH_LAB_CHAMPION_ICPS_PER_DAY = 6
DEFAULT_USD_PER_0_1_PERCENT_EPOCH = Decimal("0.162")


def canonical_json(data: Any) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_json(data: Any) -> str:
    return "sha256:" + hashlib.sha256(canonical_json(data).encode("utf-8")).hexdigest()


def usd_to_microusd(value: Any) -> int:
    return int((_decimal(value) * MICRO_USD).quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def microusd_to_usd(value: int) -> float:
    return round(int(value) / 1_000_000, 6)


def compute_participation_score(snapshot: Mapping[str, Any], policy: Mapping[str, Any]) -> Decimal:
    return (
        _decimal(snapshot.get("distinct_funded_hotkeys", 0)) * _decimal(policy.get("distinct_funded_hotkey_weight", 1))
        + _decimal(snapshot.get("paid_loop_count", 0)) * _decimal(policy.get("paid_loop_weight", 1))
        + _decimal(snapshot.get("unique_brief_count", 0)) * _decimal(policy.get("unique_brief_weight", 1))
    )


def compute_rebate_rate(snapshot: Mapping[str, Any], policy: Mapping[str, Any]) -> Decimal:
    min_rate = _decimal(policy["min_rebate_rate"])
    max_rate = _decimal(policy["max_rebate_rate"])
    base_rate = _decimal(policy.get("base_rebate_rate", policy["max_rebate_rate"]))
    high_target = _decimal(policy["high_participation_target"])
    if min_rate < 0 or max_rate < 0 or base_rate < 0 or min_rate > max_rate or max_rate > 1:
        raise ValueError("rebate rates must satisfy 0 <= min <= max <= 1")
    if base_rate < min_rate or base_rate > max_rate:
        raise ValueError("base_rebate_rate must satisfy min <= base <= max")
    if high_target <= 0:
        raise ValueError("high_participation_target must be positive")

    participation = compute_participation_score(snapshot, policy)
    fraction = _clamp(participation / high_target, Decimal("0"), Decimal("1"))
    rate_ceiling = base_rate if str(snapshot.get("island")) == str(policy.get("default_island", "generalist")) else max_rate
    rate = rate_ceiling - fraction * (rate_ceiling - min_rate)
    return rate.quantize(RATE_QUANT, rounding=ROUND_HALF_UP)


def compute_reimbursement_award(
    run: Mapping[str, Any],
    snapshot: Mapping[str, Any],
    policy: Mapping[str, Any],
    cap_usage: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Compute a deterministic alpha reimbursement award from anchored inputs."""
    cap_usage = cap_usage or {}
    participation_score = compute_participation_score(snapshot, policy)
    participation_fraction = _clamp(
        participation_score / _decimal(policy["high_participation_target"]),
        Decimal("0"),
        Decimal("1"),
    )
    rebate_rate = compute_rebate_rate(snapshot, policy)
    reimbursement_epochs = int(policy["reimbursement_epochs"])
    if reimbursement_epochs <= 0:
        raise ValueError("reimbursement_epochs must be positive")

    eligible_cost = _eligible_reimbursement_cost(run, policy)
    include_loop_start_fee = bool(policy.get("include_loop_start_fee_in_base", False))
    if include_loop_start_fee:
        eligible_cost += usd_to_microusd(run.get("loop_start_tao_fee_usd", 0))

    input_hash = sha256_json(
        {
            "run": _sorted_public(run),
            "snapshot": _sorted_public(snapshot),
            "policy": _sorted_public(policy),
            "cap_usage": _sorted_public(cap_usage),
        }
    )

    if not bool(policy.get("enabled", False)):
        return _zero_reimbursement_award(
            run=run,
            policy=policy,
            status="disabled",
            reasons=("policy_disabled",),
            participation_score=participation_score,
            participation_fraction=participation_fraction,
            rebate_rate=rebate_rate,
            eligible_cost_microusd=eligible_cost,
            input_hash=input_hash,
            loop_start_fee_included=include_loop_start_fee,
        )

    reasons = _reimbursement_ineligibility_reasons(run)
    if eligible_cost <= 0:
        reasons.append("nonpositive_eligible_cost")
    if reasons:
        return _zero_reimbursement_award(
            run=run,
            policy=policy,
            status="ineligible",
            reasons=tuple(reasons),
            participation_score=participation_score,
            participation_fraction=participation_fraction,
            rebate_rate=rebate_rate,
            eligible_cost_microusd=eligible_cost,
            input_hash=input_hash,
            loop_start_fee_included=include_loop_start_fee,
        )

    caps_applied = []
    capped_cost_basis = min(eligible_cost, usd_to_microusd(policy["max_usd_per_run"]))
    if capped_cost_basis < eligible_cost:
        caps_applied.append("per_run")

    raw_target = _round_microusd(Decimal(capped_cost_basis) * rebate_rate)
    target = raw_target
    for cap_name, remaining in _remaining_reimbursement_caps(policy, cap_usage):
        if target > remaining:
            target = max(0, remaining)
            caps_applied.append(cap_name)

    status = "awarded" if target > 0 else "capped_to_zero"
    award_without_id = {
        "award_id": "",
        "run_id": str(run["run_id"]),
        "miner_hotkey": str(run["miner_hotkey"]),
        "island": str(run["island"]),
        "run_day": str(run["run_day"]),
        "status": status,
        "participation_score": _rate_float(participation_score),
        "participation_fraction": _rate_float(participation_fraction),
        "rebate_rate": _rate_float(rebate_rate),
        "eligible_cost_microusd": eligible_cost,
        "eligible_cost_usd": microusd_to_usd(eligible_cost),
        "capped_cost_basis_microusd": capped_cost_basis,
        "capped_cost_basis_usd": microusd_to_usd(capped_cost_basis),
        "raw_reimbursement_microusd": raw_target,
        "raw_reimbursement_usd": microusd_to_usd(raw_target),
        "target_reimbursement_microusd": target,
        "target_reimbursement_usd": microusd_to_usd(target),
        "reimbursement_epochs": reimbursement_epochs,
        "caps_applied": caps_applied,
        "ineligibility_reasons": [],
        "loop_start_fee_included": include_loop_start_fee,
        "input_hash": input_hash,
    }
    if "usd_per_0_1_percent_epoch" in policy:
        award_without_id["usd_per_0_1_percent_epoch"] = microusd_to_usd(
            usd_to_microusd(policy["usd_per_0_1_percent_epoch"])
        )
    return {**award_without_id, "award_id": "reimbursement_award:" + sha256_json(award_without_id)}


def build_reimbursement_schedule(award: Mapping[str, Any], *, start_epoch: int) -> Dict[str, Any]:
    if int(start_epoch) < 0:
        raise ValueError("start_epoch must be non-negative")
    total = int(award["target_reimbursement_microusd"])
    if award["status"] != "awarded" or total <= 0:
        return {
            "schedule_id": f"reimbursement_schedule:{award['award_id']}",
            "award_id": str(award["award_id"]),
            "status": "empty",
            "start_epoch": int(start_epoch),
            "epoch_count": 0,
            "total_microusd": 0,
            "total_usd": 0.0,
            "entries": [],
        }

    epoch_count = int(award["reimbursement_epochs"])
    if epoch_count <= 0:
        raise ValueError("reimbursement_epochs must be positive for an awarded reimbursement")
    base = total // epoch_count
    remainder = total % epoch_count
    entries = []
    for idx in range(epoch_count):
        amount = base + (1 if idx < remainder else 0)
        entry = {
            "epoch": int(start_epoch) + idx,
            "amount_microusd": amount,
            "amount_usd": microusd_to_usd(amount),
        }
        alpha_fields = _alpha_entry_fields(amount, award.get("usd_per_0_1_percent_epoch"))
        if alpha_fields:
            entry.update(alpha_fields)
        entries.append(entry)
    return {
        "schedule_id": f"reimbursement_schedule:{award['award_id']}",
        "award_id": str(award["award_id"]),
        "status": "scheduled",
        "start_epoch": int(start_epoch),
        "epoch_count": epoch_count,
        "total_microusd": total,
        "total_usd": microusd_to_usd(total),
        "entries": entries,
    }


def build_champion_reward_obligation(candidate: Mapping[str, Any], policy: Mapping[str, Any]) -> Dict[str, Any]:
    """Build a deterministic champion reward obligation from validator scoring output."""
    eval_days = int(policy.get("champion_eval_days", DEFAULT_RESEARCH_LAB_CHAMPION_EVAL_DAYS))
    icps_per_day = int(policy.get("champion_icps_per_day", DEFAULT_RESEARCH_LAB_CHAMPION_ICPS_PER_DAY))
    if eval_days <= 0 or icps_per_day <= 0:
        raise ValueError("champion evaluation window must be positive")

    daily_counts = _daily_icp_counts(candidate)
    window_mode = str(policy.get("champion_window_mode") or "legacy_rolling").strip().lower()
    if window_mode == "hybrid_fresh_retained":
        fresh_count = int(policy.get("champion_fresh_icp_count", 0) or 0)
        retained_count = int(policy.get("champion_retained_icp_count", 0) or 0)
        total_required = fresh_count + retained_count if fresh_count > 0 or retained_count > 0 else eval_days * icps_per_day
    else:
        total_required = eval_days * icps_per_day
    reasons: list[str] = []
    observed_total = sum(daily_counts.values())
    if window_mode == "hybrid_fresh_retained":
        # Hybrid windows intentionally mix retained historical ICPs with fresh
        # same-day ICPs, so the day distribution is not expected to be uniform.
        # Eligibility is based on the configured total window size.
        if observed_total != total_required:
            reasons.append("wrong_total_icp_count")
    else:
        if len(daily_counts) != eval_days:
            reasons.append("wrong_daily_icp_window")
        for day, count in sorted(daily_counts.items()):
            if count != icps_per_day:
                reasons.append(f"wrong_icp_count_for_day:{day}")
        if observed_total != total_required:
            reasons.append("wrong_total_icp_count")
    if total_required <= 0:
        reasons.append("wrong_total_icp_count")

    threshold = _decimal(policy.get("champion_threshold_points", DEFAULT_RESEARCH_LAB_CHAMPION_THRESHOLD_POINTS))
    improvement_points = _decimal(
        candidate.get(
            "improvement_points",
            candidate.get("score_delta", candidate.get("mean_delta", candidate.get("delta", 0))),
        )
    )
    if improvement_points < threshold:
        reasons.append("below_champion_threshold")

    start_epoch = int(candidate.get("start_epoch", int(candidate.get("evaluation_epoch", 0)) + 1))
    reward_epochs = int(policy.get("reward_epochs", policy.get("reimbursement_epochs", DEFAULT_RESEARCH_LAB_REWARD_EPOCHS)))
    if reward_epochs <= 0:
        raise ValueError("reward_epochs must be positive")

    desired_alpha = Decimal("0")
    status = "blocked" if reasons and reasons != ["below_champion_threshold"] else "not_eligible"
    if not reasons:
        desired_alpha = _champion_desired_alpha_percent(candidate, policy)
        status = "active" if desired_alpha > 0 else "not_eligible"

    payload_without_id = {
        "champion_reward_id": "",
        "status": status,
        "reasons": reasons,
        "uid": int(candidate.get("uid", candidate.get("miner_uid", -1))),
        "miner_hotkey": str(candidate.get("miner_hotkey", "")),
        "island": str(candidate.get("island", "generalist")),
        "source_id": str(
            candidate.get("candidate_id")
            or candidate.get("score_bundle_id")
            or candidate.get("run_id")
            or "unknown"
        ),
        "score_bundle_id": str(candidate.get("score_bundle_id", "")),
        "candidate_id": str(candidate.get("candidate_id", "")),
        "run_id": str(candidate.get("run_id", "")),
        "evaluation_epoch": int(candidate.get("evaluation_epoch", 0)),
        "start_epoch": start_epoch,
        "epoch_count": reward_epochs,
        "improvement_points": _rate_float(improvement_points),
        "threshold_points": _rate_float(threshold),
        "desired_alpha_percent": _rate_float(desired_alpha),
        "daily_icp_counts": dict(sorted(daily_counts.items())),
        "required_icp_count": total_required,
        "input_hash": sha256_json(_sorted_public({"candidate": candidate, "policy": policy})),
    }
    anchored_hash = sha256_json(payload_without_id)
    reward_id = "champion_reward:" + anchored_hash
    return {**payload_without_id, "champion_reward_id": reward_id, "anchored_hash": anchored_hash}


def allocate_research_lab_epoch(
    epoch: int,
    policy: Mapping[str, Any],
    active_reimbursement_obligations: Sequence[Mapping[str, Any]],
    active_champion_obligations: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Allocate the Research Lab emission slice for one epoch.

    Inputs are public/anchored obligation records. The output is deterministic
    and intended to be stored as the per-epoch Lab allocation snapshot.
    """
    epoch = int(epoch)
    lab_cap = _decimal(policy.get("research_lab_emission_percent", DEFAULT_RESEARCH_LAB_EMISSION_PERCENT))
    if lab_cap < 0 or lab_cap > 100:
        raise ValueError("research_lab_emission_percent must be between 0 and 100")

    reward_epochs = int(policy.get("reward_epochs", policy.get("reimbursement_epochs", DEFAULT_RESEARCH_LAB_REWARD_EPOCHS)))
    if reward_epochs <= 0:
        raise ValueError("reward_epochs must be positive")

    input_payload = {
        "epoch": epoch,
        "policy": _sorted_public(policy),
        "reimbursement_obligations": _sorted_public(active_reimbursement_obligations),
        "champion_obligations": _sorted_public(active_champion_obligations),
    }
    input_hash = sha256_json(input_payload)

    reimbursements = [
        _normalize_reimbursement_obligation(item, epoch=epoch, policy=policy)
        for item in active_reimbursement_obligations
        if _obligation_active(item, epoch, default_epoch_count=reward_epochs)
    ]
    reimbursements = [item for item in reimbursements if item["spend_microusd"] > 0 and item["uid"] >= 0]

    champions = [
        _normalize_champion_obligation(item, epoch=epoch, policy=policy)
        for item in active_champion_obligations
        if _obligation_active(item, epoch, default_epoch_count=reward_epochs)
    ]
    champions = [item for item in champions if item["desired_alpha_percent"] > 0 and item["uid"] >= 0]
    champions.sort(key=lambda item: (item["start_epoch"], -item["improvement_points"], item["source_id"]))

    reimbursement_allocations: list[Dict[str, Any]] = []
    champion_allocations: list[Dict[str, Any]] = []
    queued_champion_allocations: list[Dict[str, Any]] = []

    if not reimbursements and not champions:
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
        return {**result, "allocation_hash": sha256_json(result)}

    reimbursement_paid = Decimal("0")
    if reimbursements:
        if not champions and bool(policy.get("reimbursement_allow_overpay_without_champions", True)):
            reimbursement_allocations = _allocate_reimbursements_no_champions(reimbursements, lab_cap)
            _cap_allocation_sections_to_pool((reimbursement_allocations,), lab_cap)
        else:
            champion_reserve = _minimum_champion_reserve(champions, lab_cap, policy)
            reimbursement_pool = max(Decimal("0"), lab_cap - champion_reserve)
            reimbursement_allocations = _allocate_reimbursements_with_champions(reimbursements, reimbursement_pool)
            _cap_allocation_sections_to_pool((reimbursement_allocations,), reimbursement_pool)
        reimbursement_paid = sum(_decimal(item["paid_alpha_percent"]) for item in reimbursement_allocations)

    remaining_for_champions = max(Decimal("0"), lab_cap - reimbursement_paid)
    if champions:
        champion_allocations, queued_champion_allocations = _allocate_champions(
            champions,
            remaining_for_champions,
            policy,
            reimbursement_paid=reimbursement_paid,
        )

    _cap_allocation_sections_to_pool(
        (reimbursement_allocations, champion_allocations, queued_champion_allocations),
        lab_cap,
    )
    reimbursement_paid = sum((_decimal(item["paid_alpha_percent"]) for item in reimbursement_allocations), Decimal("0"))
    champion_paid = sum((_decimal(item["paid_alpha_percent"]) for item in champion_allocations), Decimal("0"))
    queued_paid = sum((_decimal(item["paid_alpha_percent"]) for item in queued_champion_allocations), Decimal("0"))
    total_paid = reimbursement_paid + champion_paid + queued_paid
    unallocated = max(Decimal("0"), lab_cap - total_paid)
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
    return {**result, "allocation_hash": sha256_json(result)}


def cap_reimbursement_schedules_by_epoch(
    schedules: Sequence[Mapping[str, Any]],
    *,
    max_alpha_percent_per_epoch: Any,
) -> list[Dict[str, Any]]:
    cap = _decimal(max_alpha_percent_per_epoch)
    if cap <= 0:
        return [dict(schedule) for schedule in schedules]

    outputs: list[Dict[str, Any]] = []
    obligations: list[tuple[int, str, int, int, Decimal, int]] = []
    for schedule_index, schedule in enumerate(schedules):
        output = dict(schedule)
        output["entries"] = []
        output["cap_applied"] = False
        output["roll_forward_microusd"] = 0
        outputs.append(output)
        schedule_id = str(schedule.get("schedule_id") or schedule.get("award_id") or schedule_index)
        for entry_index, entry in enumerate(schedule.get("entries", [])):
            amount = int(entry.get("amount_microusd", 0))
            if amount <= 0:
                continue
            alpha = _decimal(entry.get("alpha_percent", 0))
            if alpha <= 0:
                output["entries"].append(dict(entry))
                continue
            obligations.append((int(entry["epoch"]), schedule_id, entry_index, amount, alpha, schedule_index))

    usage_by_epoch: Dict[int, Decimal] = {}
    for original_epoch, _schedule_id, _entry_index, amount, alpha, schedule_index in sorted(obligations):
        unpaid_amount = int(amount)
        unpaid_alpha = alpha
        epoch = int(original_epoch)
        while unpaid_amount > 0 and unpaid_alpha > 0:
            used = usage_by_epoch.get(epoch, Decimal("0"))
            remaining = cap - used
            if remaining <= 0:
                outputs[schedule_index]["cap_applied"] = True
                epoch += 1
                continue
            if unpaid_alpha <= remaining:
                paid_amount = unpaid_amount
                paid_alpha = unpaid_alpha
            else:
                ratio = remaining / unpaid_alpha
                paid_amount = max(1, min(unpaid_amount, _round_microusd(Decimal(unpaid_amount) * ratio)))
                paid_alpha = unpaid_alpha * Decimal(paid_amount) / Decimal(unpaid_amount)
                outputs[schedule_index]["cap_applied"] = True

            entry = {
                "epoch": epoch,
                "amount_microusd": paid_amount,
                "amount_usd": microusd_to_usd(paid_amount),
                "alpha_percent": _rate_float(paid_alpha),
                "alpha_share_ppm": int((paid_alpha * Decimal("10000")).quantize(Decimal("1"), rounding=ROUND_HALF_UP)),
                "original_epoch": original_epoch,
            }
            if epoch != original_epoch:
                entry["roll_forward"] = True
            outputs[schedule_index]["entries"].append(entry)
            usage_by_epoch[epoch] = used + paid_alpha
            unpaid_amount -= paid_amount
            unpaid_alpha -= paid_alpha
            if unpaid_amount > 0:
                outputs[schedule_index]["roll_forward_microusd"] = int(outputs[schedule_index]["roll_forward_microusd"]) + unpaid_amount
                epoch += 1

    for output in outputs:
        output["entries"] = sorted(
            output["entries"],
            key=lambda item: (int(item.get("epoch", 0)), int(item.get("original_epoch", item.get("epoch", 0)))),
        )
        output["epoch_count"] = len({int(entry["epoch"]) for entry in output.get("entries", [])})
        output["total_microusd"] = sum(int(entry["amount_microusd"]) for entry in output.get("entries", []))
        output["total_usd"] = microusd_to_usd(int(output["total_microusd"]))
    return outputs


def verify_reimbursement_schedule(award: Mapping[str, Any], schedule: Mapping[str, Any]) -> Dict[str, Any]:
    errors = []
    total = sum(int(entry["amount_microusd"]) for entry in schedule.get("entries", []))
    if total != int(schedule["total_microusd"]):
        errors.append("entries_do_not_sum_to_schedule_total")
    if int(schedule["total_microusd"]) != int(award["target_reimbursement_microusd"]):
        errors.append("schedule_total_does_not_match_award_target")
    if award["status"] == "awarded" and int(schedule["epoch_count"]) != int(award["reimbursement_epochs"]):
        errors.append("epoch_count_does_not_match_award")
    if award["status"] != "awarded" and schedule.get("entries"):
        errors.append("non_awarded_schedule_must_be_empty")
    return {
        "passed": not errors,
        "errors": errors,
        "schedule_total_microusd": total,
        "award_target_microusd": int(award["target_reimbursement_microusd"]),
    }


def verify_loop_start_payment_state(
    payment: Mapping[str, Any],
    policy: Mapping[str, Any],
    *,
    used_payment_refs: Sequence[str] = (),
) -> Dict[str, Any]:
    payment_ref = str(payment.get("payment_ref") or f"{payment.get('block_hash', '')}:{int(payment.get('extrinsic_index', -1))}")
    amount_usd = _decimal(payment.get("amount_usd", microusd_to_usd(int(payment.get("amount_microusd", 0)))))
    required_usd = _decimal(policy.get("loop_start_fee_usd", 0))
    if not bool(policy.get("loop_start_fee_enabled", True)):
        return {
            "status": "fee_disabled",
            "payment_ref": payment_ref,
            "valid": True,
            "required_usd": 0.0,
            "amount_usd": _money_float(amount_usd),
            "reasons": [],
        }

    reasons = []
    if int(payment.get("extrinsic_index", -1)) < 0:
        reasons.append("invalid_extrinsic_index")
    if payment_ref in set(str(ref) for ref in used_payment_refs):
        reasons.append("duplicate_payment")
    if payment.get("call_function") not in VALID_TRANSFER_CALLS:
        reasons.append("not_a_valid_transfer_call")
    if payment.get("destination_wallet") != policy.get("leadpoet_payment_wallet"):
        reasons.append("wrong_destination_wallet")
    if payment.get("sender_coldkey") != payment.get("hotkey_owner_coldkey"):
        reasons.append("coldkey_hotkey_mismatch")
    required_with_buffer = required_usd * (Decimal("1") - _decimal(policy.get("amount_buffer_percent", 0)))
    if amount_usd < required_with_buffer:
        reasons.append("insufficient_payment")
    if not bool(payment.get("extrinsic_success", False)):
        reasons.append("failed_extrinsic")
    if int(payment.get("age_seconds", 0)) > int(policy.get("payment_max_age_seconds", 0)):
        reasons.append("stale_payment")
    if int(payment.get("age_seconds", 0)) < -300:
        reasons.append("future_payment_timestamp")
    if payment.get("payment_ref") and payment_ref != f"{payment.get('block_hash', '')}:{int(payment.get('extrinsic_index', -1))}":
        reasons.append("payment_ref_mismatch")

    return {
        "status": "rejected" if reasons else "verified",
        "payment_ref": payment_ref,
        "valid": not reasons,
        "required_usd": _money_float(required_usd),
        "amount_usd": _money_float(amount_usd),
        "reasons": reasons,
    }


def build_improvement_grant_schedule(crown: Mapping[str, Any], policy: Mapping[str, Any]) -> Dict[str, Any]:
    start_epoch = int(crown.get("grant_start_epoch", int(crown["crown_epoch"]) + 1))
    fixed_epochs = int(policy["fixed_term_epochs"])
    if fixed_epochs <= 0:
        raise ValueError("fixed_term_epochs must be positive")

    empty_reason = ""
    if not bool(crown.get("crowned", False)):
        empty_reason = "not_crowned"
    elif bool(crown.get("house_improvement", False)):
        empty_reason = "house_improvement_no_grant"
    elif _decimal(crown["delta"]) < _decimal(policy.get("delta_min", 0)):
        empty_reason = "below_delta_min"

    if empty_reason:
        return {
            "grant_id": f"improvement_grant:{sha256_json(_sorted_public(crown))}",
            "status": "empty",
            "reason": empty_reason,
            "miner_uid": int(crown.get("miner_uid", -1)),
            "miner_hotkey": str(crown.get("miner_hotkey", "")),
            "island": str(crown.get("island", "")),
            "area": str(crown.get("area", "")),
            "delta": _money_float(_decimal(crown.get("delta", 0))),
            "start_epoch": start_epoch,
            "epoch_count": 0,
            "per_epoch_share_ppm": 0,
            "total_share_ppm_epochs": 0,
            "entries": [],
        }

    per_epoch_share_ppm = interpolate_grant_share_ppm(_decimal(crown["delta"]), policy["grant_curve"])
    entries = [
        {
            "epoch": start_epoch + idx,
            "uid": int(crown["miner_uid"]),
            "share_ppm": per_epoch_share_ppm,
        }
        for idx in range(fixed_epochs)
    ]
    payload_without_id = {
        "grant_id": "",
        "status": "scheduled",
        "reason": "",
        "miner_uid": int(crown["miner_uid"]),
        "miner_hotkey": str(crown["miner_hotkey"]),
        "island": str(crown["island"]),
        "area": str(crown["area"]),
        "delta": _money_float(_decimal(crown["delta"])),
        "start_epoch": start_epoch,
        "epoch_count": fixed_epochs,
        "per_epoch_share_ppm": per_epoch_share_ppm,
        "total_share_ppm_epochs": per_epoch_share_ppm * fixed_epochs,
        "entries": entries,
    }
    return {**payload_without_id, "grant_id": "improvement_grant:" + sha256_json(payload_without_id)}


def verify_improvement_grant_schedule(
    crown: Mapping[str, Any],
    policy: Mapping[str, Any],
    schedule: Mapping[str, Any],
) -> Dict[str, Any]:
    expected = build_improvement_grant_schedule(crown, policy)
    errors = []
    if schedule != expected:
        errors.append("grant_schedule_does_not_match_recomputed_schedule")
    if schedule.get("status") == "scheduled":
        total = sum(int(entry["share_ppm"]) for entry in schedule.get("entries", []))
        if total != int(schedule["total_share_ppm_epochs"]):
            errors.append("grant_entries_do_not_sum_to_total")
    return {
        "passed": not errors,
        "errors": errors,
        "expected_schedule_hash": sha256_json(expected),
        "actual_schedule_hash": sha256_json(schedule),
    }


def interpolate_grant_share_ppm(delta: Decimal, points: Sequence[Mapping[str, Any]]) -> int:
    if not points:
        raise ValueError("grant_curve requires at least one point")
    sorted_points = sorted(
        ((_decimal(point["delta"]), _decimal(point["share_ppm"])) for point in points),
        key=lambda item: item[0],
    )
    if delta <= sorted_points[0][0]:
        return int(sorted_points[0][1].quantize(Decimal("1"), rounding=ROUND_HALF_UP))
    for (delta_a, share_a), (delta_b, share_b) in zip(sorted_points, sorted_points[1:]):
        if delta_b <= delta_a:
            raise ValueError("grant_curve deltas must be strictly increasing")
        if delta <= delta_b:
            fraction = (delta - delta_a) / (delta_b - delta_a)
            interpolated = share_a + fraction * (share_b - share_a)
            return int(interpolated.quantize(Decimal("1"), rounding=ROUND_HALF_UP))
    return int(sorted_points[-1][1].quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def compose_final_weight_vector(
    *,
    epoch: int,
    uids: Iterable[int],
    fulfillment_scores: Mapping[str, Any] | None = None,
    leaderboard_scores: Mapping[str, Any] | None = None,
    improvement_grant_schedules: Sequence[Mapping[str, Any]] = (),
    reimbursement_schedules: Sequence[Mapping[str, Any]] = (),
    research_lab_allocation: Mapping[str, Any] | None = None,
    active_researcher_floor_scores: Mapping[str, Any] | None = None,
    policy: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Compose Research Lab weight inputs and quantize to u16 deterministically."""
    policy = policy or {}
    if policy.get("max_reimbursement_alpha_percent_per_epoch") not in (None, "", 0, "0"):
        reimbursement_schedules = cap_reimbursement_schedules_by_epoch(
            reimbursement_schedules,
            max_alpha_percent_per_epoch=policy["max_reimbursement_alpha_percent_per_epoch"],
        )
    uid_set = {int(uid) for uid in uids}
    fulfillment_scores = fulfillment_scores or {}
    leaderboard_scores = leaderboard_scores or {}
    active_researcher_floor_scores = active_researcher_floor_scores or {}
    for mapping in (fulfillment_scores, leaderboard_scores, active_researcher_floor_scores):
        uid_set.update(int(uid) for uid in mapping)
    for schedule in improvement_grant_schedules:
        for entry in schedule.get("entries", []):
            if int(entry["epoch"]) == int(epoch):
                uid_set.add(int(entry["uid"]))
    for schedule in reimbursement_schedules:
        uid = schedule.get("uid")
        if uid is not None:
            uid_set.add(int(uid))
    if research_lab_allocation:
        for section in (
            "reimbursement_allocations",
            "champion_allocations",
            "queued_champion_allocations",
        ):
            for allocation in research_lab_allocation.get(section, []):
                uid_set.add(int(allocation["uid"]))

    grant_score_per_ppm = _decimal(policy.get("grant_score_per_ppm", "0.001"))
    reimbursement_score_per_microusd = _decimal(policy.get("reimbursement_score_per_microusd", "0.000001"))
    lab_score_per_alpha_percent = _decimal(policy.get("lab_score_per_alpha_percent", "1"))
    fulfillment_scale = _decimal(policy.get("fulfillment_scale", "1"))
    leaderboard_scale = _decimal(policy.get("leaderboard_scale", "1"))
    floor_scale = _decimal(policy.get("floor_scale", "1"))

    components: Dict[str, Dict[str, float]] = {}
    raw_scores: Dict[int, float] = {}
    for uid in sorted(uid_set):
        uid_key = str(uid)
        fulfillment = _decimal(fulfillment_scores.get(uid_key, fulfillment_scores.get(uid, 0))) * fulfillment_scale
        leaderboard = _decimal(leaderboard_scores.get(uid_key, leaderboard_scores.get(uid, 0))) * leaderboard_scale
        floor = _decimal(active_researcher_floor_scores.get(uid_key, active_researcher_floor_scores.get(uid, 0))) * floor_scale
        grant_score = Decimal("0")
        reimbursement_score = Decimal("0")
        queued_champion_score = Decimal("0")

        if research_lab_allocation:
            for allocation in research_lab_allocation.get("champion_allocations", []):
                if int(allocation["uid"]) == uid:
                    grant_score += _decimal(allocation["paid_alpha_percent"]) * lab_score_per_alpha_percent
            for allocation in research_lab_allocation.get("queued_champion_allocations", []):
                if int(allocation["uid"]) == uid:
                    queued_champion_score += _decimal(allocation["paid_alpha_percent"]) * lab_score_per_alpha_percent
            for allocation in research_lab_allocation.get("reimbursement_allocations", []):
                if int(allocation["uid"]) == uid:
                    reimbursement_score += _decimal(allocation["paid_alpha_percent"]) * lab_score_per_alpha_percent
        else:
            for schedule in improvement_grant_schedules:
                for entry in schedule.get("entries", []):
                    if int(entry["epoch"]) == int(epoch) and int(entry["uid"]) == uid:
                        grant_score += _decimal(entry["share_ppm"]) * grant_score_per_ppm

            for schedule in reimbursement_schedules:
                if int(schedule.get("uid", -1)) != uid:
                    continue
                for entry in schedule.get("entries", []):
                    if int(entry["epoch"]) == int(epoch):
                        reimbursement_score += _decimal(entry["amount_microusd"]) * reimbursement_score_per_microusd

        total = fulfillment + leaderboard + grant_score + queued_champion_score + reimbursement_score + floor
        component_doc = {
            "fulfillment": _money_float(fulfillment),
            "weekly_leaderboard": _money_float(leaderboard),
            "active_improvement_grant": _money_float(grant_score),
            "active_reimbursement": _money_float(reimbursement_score),
            "active_researcher_floor": _money_float(floor),
            "total": _money_float(total),
        }
        if research_lab_allocation:
            component_doc["queued_improvement_grant_placeholder"] = _money_float(queued_champion_score)
        components[uid_key] = component_doc
        raw_scores[uid] = float(total)

    u16 = u16_weights_from_scores(raw_scores, total_weight=int(policy.get("total_weight", 65535)))
    return {
        "epoch": int(epoch),
        "components_by_uid": components,
        "scores_by_uid": {str(uid): _money_float(Decimal(str(score))) for uid, score in sorted(raw_scores.items())},
        "u16_weights": {str(uid): weight for uid, weight in sorted(u16.items())},
        "weight_sum": sum(u16.values()),
    }


def _reimbursement_ineligibility_reasons(run: Mapping[str, Any]) -> list[str]:
    reasons: list[str] = []
    if not bool(run.get("paid_research_loop", True)):
        reasons.append("not_paid_research_loop")
    if not bool(run.get("valid_receipt", True)):
        reasons.append("missing_valid_receipt")
    if not (bool(run.get("verified_loop_start_payment", True)) or bool(run.get("preserved_loop_start_credit", False))):
        reasons.append("missing_loop_start_payment_or_credit")
    if not bool(run.get("miner_openrouter_key_present", True)):
        reasons.append("missing_miner_openrouter_key")
    if not bool(run.get("trusted_cost_ledger", True)):
        reasons.append("missing_trusted_cost_ledger")
    if not bool(run.get("passed_abuse_checks", True)):
        reasons.append("abuse_checks_failed")
    if bool(run.get("refunded", False)):
        reasons.append("refunded")
    if bool(run.get("voided", False)):
        reasons.append("voided")
    if bool(run.get("duplicate", False)):
        reasons.append("duplicate_run")
    if bool(run.get("novelty_rejected", False)):
        reasons.append("novelty_rejected")
    if bool(run.get("self_cancelled_before_minimum_work", False)):
        reasons.append("self_cancelled_before_minimum_work")
    if bool(run.get("banned_hotkey", False)):
        reasons.append("banned_hotkey")
    return reasons


def _remaining_reimbursement_caps(
    policy: Mapping[str, Any],
    usage: Mapping[str, Any],
) -> tuple[tuple[str, int], ...]:
    return (
        (
            "hotkey",
            max(0, usd_to_microusd(policy["max_usd_per_hotkey_day"]) - usd_to_microusd(usage.get("hotkey_day_awarded_usd", 0))),
        ),
        (
            "island",
            max(0, usd_to_microusd(policy["max_usd_per_island_day"]) - usd_to_microusd(usage.get("island_day_awarded_usd", 0))),
        ),
        (
            "global",
            max(0, usd_to_microusd(policy["global_budget_usd"]) - usd_to_microusd(usage.get("global_awarded_usd", 0))),
        ),
    )


def _eligible_reimbursement_cost(run: Mapping[str, Any], policy: Mapping[str, Any]) -> int:
    if "material_spend_ratio" not in policy:
        return usd_to_microusd(run.get("verified_compute_cost_usd", 0)) + usd_to_microusd(
            run.get("miner_openrouter_cost_usd", 0)
        )
    ratio = _decimal(policy["material_spend_ratio"])
    if ratio < 0 or ratio > 1:
        raise ValueError("material_spend_ratio must be between 0 and 1")
    funded = usd_to_microusd(run.get("funded_compute_budget_usd", run.get("verified_compute_cost_usd", 0)))
    actual = usd_to_microusd(run.get("actual_openrouter_cost_usd", run.get("miner_openrouter_cost_usd", 0)))
    if actual <= 0:
        return 0
    if funded <= 0:
        return actual
    material_threshold = _round_microusd(Decimal(funded) * ratio)
    if actual < material_threshold:
        return actual
    return funded


def _daily_icp_counts(candidate: Mapping[str, Any]) -> Dict[str, int]:
    for key in ("daily_icp_counts", "per_day_icp_counts", "icp_counts_by_day"):
        value = candidate.get(key)
        if isinstance(value, Mapping):
            return {str(day): int(count) for day, count in value.items()}
    counts: Dict[str, int] = {}
    for item in candidate.get("per_icp_results", []) or []:
        if not isinstance(item, Mapping):
            continue
        day = str(item.get("icp_day") or item.get("day") or item.get("date") or "")
        if day:
            counts[day] = counts.get(day, 0) + 1
    return counts


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


def _normalize_reimbursement_obligation(
    obligation: Mapping[str, Any],
    *,
    epoch: int,
    policy: Mapping[str, Any],
) -> Dict[str, Any]:
    reward_epochs = int(policy.get("reward_epochs", policy.get("reimbursement_epochs", DEFAULT_RESEARCH_LAB_REWARD_EPOCHS)))
    start_epoch = int(obligation.get("start_epoch", epoch))
    epoch_count = int(obligation.get("epoch_count", obligation.get("reimbursement_epochs", reward_epochs)))
    spend = _obligation_spend_microusd(obligation)
    max_multiplier = _decimal(policy.get("reimbursement_max_cost_multiplier_with_champions", "1.0"))
    if max_multiplier < 0:
        raise ValueError("reimbursement_max_cost_multiplier_with_champions must be non-negative")
    intended = _alpha_percent_for_microusd(_round_microusd(Decimal(spend) / Decimal(max(1, epoch_count))), policy)
    capped_intended = intended * min(max_multiplier, Decimal("1.0"))
    weight = Decimal(max(spend, 0)) * _decimal(
        obligation.get("island_weight", obligation.get("participation_weight", obligation.get("reimbursement_weight", 1)))
    )
    return {
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
    return {
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
        "intended_alpha_percent": _champion_desired_alpha_percent(obligation, policy),
        "desired_alpha_percent": _champion_desired_alpha_percent(obligation, policy),
    }


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


def _allocate_reimbursements_no_champions(
    reimbursements: Sequence[Mapping[str, Any]],
    lab_cap: Decimal,
) -> list[Dict[str, Any]]:
    weight_sum = sum(_decimal(item["pro_rata_weight"]) for item in reimbursements)
    if weight_sum <= 0:
        return [_reimbursement_allocation(item, Decimal("0"), "no_weight") for item in reimbursements]
    allocations = []
    for item in reimbursements:
        paid = lab_cap * _decimal(item["pro_rata_weight"]) / weight_sum
        allocations.append(_reimbursement_allocation(item, paid, "overpay_no_active_champions"))
    return allocations


def _allocate_reimbursements_with_champions(
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
    placeholder = _decimal(
        policy.get("champion_placeholder_alpha_percent", DEFAULT_RESEARCH_LAB_CHAMPION_PLACEHOLDER_ALPHA_PERCENT)
    )
    queue_trigger_ratio = _decimal(policy.get("champion_queue_trigger_ratio", Decimal("0.50")))
    if queue_trigger_ratio < 0:
        raise ValueError("champion_queue_trigger_ratio must be non-negative")

    total_pool = max(Decimal("0"), pool)
    active_indices: list[int] = []
    queued_indices: list[int] = []

    for index, champion in enumerate(champions):
        desired = _decimal(champion["desired_alpha_percent"])
        should_queue_for_reimbursement = (
            bool(active_indices)
            and queue_trigger_ratio > 0
            and reimbursement_paid >= desired * queue_trigger_ratio
        )
        if should_queue_for_reimbursement:
            queued_indices.append(index)
            continue

        remaining_undecided = len(champions) - index - 1
        active_desired = sum(_decimal(champions[idx]["desired_alpha_percent"]) for idx in active_indices)
        required_if_active = active_desired + desired
        required_if_active += placeholder * Decimal(len(queued_indices) + remaining_undecided)
        if required_if_active <= total_pool:
            active_indices.append(index)
        else:
            queued_indices.append(index)

    paid = [Decimal("0") for _ in champions]
    for index in active_indices:
        paid[index] = _decimal(champions[index]["desired_alpha_percent"])
    for index in queued_indices:
        paid[index] = min(placeholder, _decimal(champions[index]["desired_alpha_percent"]))

    spent = sum(paid, Decimal("0"))
    remaining = max(Decimal("0"), total_pool - spent)
    if remaining > 0 and active_indices:
        weights = [
            max(Decimal("0"), _decimal(champions[index].get("improvement_points", 0)))
            for index in active_indices
        ]
        weight_sum = sum(weights, Decimal("0"))
        if weight_sum <= 0:
            weights = [_decimal(champions[index]["desired_alpha_percent"]) for index in active_indices]
            weight_sum = sum(weights, Decimal("0"))
        if weight_sum <= 0:
            weights = [Decimal("1") for _ in active_indices]
            weight_sum = Decimal(len(active_indices))
        for index, weight in zip(active_indices, weights):
            paid[index] += remaining * weight / weight_sum
    elif remaining > 0 and paid:
        paid[0] += remaining

    active: list[Dict[str, Any]] = []
    queued: list[Dict[str, Any]] = []
    for champion, amount in zip(champions, paid):
        allocation = _champion_allocation(champion, amount)
        if amount >= _decimal(champion["desired_alpha_percent"]):
            active.append({**allocation, "reason": "active_champion_reward"})
        elif amount > 0:
            queued.append({**allocation, "reason": "queued_with_placeholder"})
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
    return {
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


def _champion_allocation(item: Mapping[str, Any], paid: Decimal) -> Dict[str, Any]:
    intended = _decimal(item["desired_alpha_percent"])
    return {
        "uid": int(item["uid"]),
        "miner_hotkey": str(item["miner_hotkey"]),
        "source_id": str(item["source_id"]),
        "island": str(item["island"]),
        "intended_alpha_percent": _rate_float(intended),
        "paid_alpha_percent": _rate_float(paid),
        "deferred_alpha_percent": _rate_float(max(Decimal("0"), intended - paid)),
        "improvement_points": _rate_float(_decimal(item["improvement_points"])),
    }


def _alpha_entry_fields(amount_microusd: int, usd_per_0_1_percent_epoch: Any) -> Dict[str, Any]:
    if usd_per_0_1_percent_epoch in (None, ""):
        return {}
    valuation = _decimal(usd_per_0_1_percent_epoch)
    if valuation <= 0:
        raise ValueError("usd_per_0_1_percent_epoch must be positive")
    amount_usd = Decimal(amount_microusd) / MICRO_USD
    alpha_percent = (amount_usd / valuation) * Decimal("0.1")
    alpha_share_ppm = int((alpha_percent * Decimal("10000")).quantize(Decimal("1"), rounding=ROUND_HALF_UP))
    return {
        "alpha_percent": _rate_float(alpha_percent),
        "alpha_share_ppm": alpha_share_ppm,
    }


def _zero_reimbursement_award(
    *,
    run: Mapping[str, Any],
    policy: Mapping[str, Any],
    status: str,
    reasons: Sequence[str],
    participation_score: Decimal,
    participation_fraction: Decimal,
    rebate_rate: Decimal,
    eligible_cost_microusd: int,
    input_hash: str,
    loop_start_fee_included: bool,
) -> Dict[str, Any]:
    award_without_id = {
        "award_id": "",
        "run_id": str(run["run_id"]),
        "miner_hotkey": str(run["miner_hotkey"]),
        "island": str(run["island"]),
        "run_day": str(run["run_day"]),
        "status": status,
        "participation_score": _rate_float(participation_score),
        "participation_fraction": _rate_float(participation_fraction),
        "rebate_rate": _rate_float(rebate_rate),
        "eligible_cost_microusd": max(0, eligible_cost_microusd),
        "eligible_cost_usd": microusd_to_usd(max(0, eligible_cost_microusd)),
        "capped_cost_basis_microusd": 0,
        "capped_cost_basis_usd": 0.0,
        "raw_reimbursement_microusd": 0,
        "raw_reimbursement_usd": 0.0,
        "target_reimbursement_microusd": 0,
        "target_reimbursement_usd": 0.0,
        "reimbursement_epochs": int(policy["reimbursement_epochs"]),
        "caps_applied": [],
        "ineligibility_reasons": list(reasons),
        "loop_start_fee_included": loop_start_fee_included,
        "input_hash": input_hash,
    }
    if "usd_per_0_1_percent_epoch" in policy:
        award_without_id["usd_per_0_1_percent_epoch"] = microusd_to_usd(
            usd_to_microusd(policy["usd_per_0_1_percent_epoch"])
        )
    return {**award_without_id, "award_id": "reimbursement_award:" + sha256_json(award_without_id)}


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
