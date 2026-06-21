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
    high_target = _decimal(policy["high_participation_target"])
    if min_rate < 0 or max_rate < 0 or min_rate > max_rate or max_rate > 1:
        raise ValueError("rebate rates must satisfy 0 <= min <= max <= 1")
    if high_target <= 0:
        raise ValueError("high_participation_target must be positive")

    participation = compute_participation_score(snapshot, policy)
    fraction = _clamp(participation / high_target, Decimal("0"), Decimal("1"))
    rate = max_rate - fraction * (max_rate - min_rate)
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

    eligible_cost = usd_to_microusd(run.get("verified_compute_cost_usd", 0)) + usd_to_microusd(
        run.get("miner_openrouter_cost_usd", 0)
    )
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
        entries.append(
            {
                "epoch": int(start_epoch) + idx,
                "amount_microusd": amount,
                "amount_usd": microusd_to_usd(amount),
            }
        )
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
    active_researcher_floor_scores: Mapping[str, Any] | None = None,
    policy: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Compose Research Lab weight inputs and quantize to u16 deterministically."""
    policy = policy or {}
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

    grant_score_per_ppm = _decimal(policy.get("grant_score_per_ppm", "0.001"))
    reimbursement_score_per_microusd = _decimal(policy.get("reimbursement_score_per_microusd", "0.000001"))
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

        total = fulfillment + leaderboard + grant_score + reimbursement_score + floor
        components[uid_key] = {
            "fulfillment": _money_float(fulfillment),
            "weekly_leaderboard": _money_float(leaderboard),
            "active_improvement_grant": _money_float(grant_score),
            "active_reimbursement": _money_float(reimbursement_score),
            "active_researcher_floor": _money_float(floor),
            "total": _money_float(total),
        }
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
