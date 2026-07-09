#!/usr/bin/env python3
"""Verify Research Lab local alpha reimbursement kernel."""

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from leadpoet_verifier.economics import (
    allocate_research_lab_epoch,
    build_champion_reward_obligation,
    cap_reimbursement_schedules_by_epoch,
    compose_final_weight_vector,
)
from research_lab.reimbursements import (
    build_reimbursement_schedule,
    compute_reimbursement_award,
    verify_research_lab_reimbursements,
)


def main() -> int:
    summary = verify_research_lab_reimbursements()
    _run_simulations()
    print(
        "Research Lab alpha reimbursement kernel verified: "
        f"low rate {summary['low_rebate_rate']:.6f}, "
        f"high rate {summary['high_rebate_rate']:.6f}, "
        f"low award {summary['low_award_microusd']} microusd, "
        f"high award {summary['high_award_microusd']} microusd, "
        f"schedule {summary['schedule_epochs']} epochs / "
        f"{summary['schedule_total_microusd']} microusd, "
        f"{summary['fixture_cases']} fixture runs."
    )
    return 0


def _run_simulations() -> None:
    policy = {
        "policy_id": "alpha-reimbursement-simulation-v1",
        "enabled": True,
        "min_rebate_rate": 1.0,
        "base_rebate_rate": 1.0,
        "max_rebate_rate": 1.0,
        "high_participation_target": 10,
        "reimbursement_epochs": 30,
        "max_usd_per_run": 100,
        "max_usd_per_hotkey_day": 1000,
        "max_usd_per_island_day": 1000,
        "global_budget_usd": 1000,
        "include_loop_start_fee_in_base": False,
        "material_spend_ratio": 0.8,
        "default_island": "generalist",
        "usd_per_0_1_percent_epoch": 0.162,
    }
    no_usage = {"hotkey_day_awarded_usd": 0, "island_day_awarded_usd": 0, "global_awarded_usd": 0}
    generalist = _snapshot("generalist", 0)
    crowded = _snapshot("generalist", 10)
    quiet_specialized = _snapshot("evidence-routing", 0)

    cases = [
        ("ten_generalist_material", _run("run-10", "generalist", 10, 8), generalist, 10_000_000),
        ("five_generalist_material", _run("run-5", "generalist", 5, 4), generalist, 5_000_000),
        ("ten_generalist_low_actual", _run("run-2", "generalist", 10, 2), generalist, 2_000_000),
        ("ten_generalist_zero_actual", _run("run-0", "generalist", 10, 0), generalist, 0),
        ("ten_specialized_quiet", _run("run-special", "evidence-routing", 10, 8), quiet_specialized, 10_000_000),
        ("ten_generalist_crowded", _run("run-crowded", "generalist", 10, 8), crowded, 10_000_000),
    ]
    schedules = []
    for name, run, snapshot, expected_target in cases:
        award = compute_reimbursement_award(run, snapshot, policy, no_usage)
        if int(award.target_reimbursement_microusd) != expected_target:
            raise AssertionError(f"{name}: expected {expected_target}, got {award.target_reimbursement_microusd}")
        schedule = build_reimbursement_schedule(award, start_epoch=2000).to_dict()
        if sum(int(entry["amount_microusd"]) for entry in schedule["entries"]) != expected_target:
            raise AssertionError(f"{name}: schedule does not sum to target")
        if expected_target > 0 and "alpha_percent" not in schedule["entries"][0]:
            raise AssertionError(f"{name}: schedule missing alpha conversion")
        schedules.append({**schedule, "uid": len(schedules) + 1})

    completed_award = compute_reimbursement_award(
        _run("run-completed", "generalist", 10, 8),
        generalist,
        policy,
        no_usage,
    )
    failed_same_spend = {
        **_run("run-failed", "generalist", 10, 8),
        "valid_receipt": True,
        "verified_loop_start_payment": True,
        "miner_openrouter_key_present": True,
        "trusted_cost_ledger": True,
        "passed_abuse_checks": True,
    }
    failed_award = compute_reimbursement_award(failed_same_spend, generalist, policy, no_usage)
    if failed_award.target_reimbursement_microusd != completed_award.target_reimbursement_microusd:
        raise AssertionError("failed run with same trusted spend must use the same reimbursement formula")

    failed_partial = compute_reimbursement_award(
        {
            **_run("run-failed-partial", "generalist", 10, 2),
            "valid_receipt": True,
            "verified_loop_start_payment": True,
            "miner_openrouter_key_present": True,
            "trusted_cost_ledger": True,
            "passed_abuse_checks": True,
        },
        generalist,
        policy,
        no_usage,
    )
    if int(failed_partial.target_reimbursement_microusd) != 2_000_000:
        raise AssertionError("failed partial-spend run should reimburse from actual spend below material threshold")

    failed_zero = compute_reimbursement_award(
        {
            **_run("run-failed-zero", "generalist", 10, 0),
            "valid_receipt": True,
            "verified_loop_start_payment": True,
            "miner_openrouter_key_present": True,
            "trusted_cost_ledger": True,
            "passed_abuse_checks": True,
        },
        generalist,
        policy,
        no_usage,
    )
    if int(failed_zero.target_reimbursement_microusd) != 0 or failed_zero.status != "ineligible":
        raise AssertionError("zero-cost failed run must not create a payable award")

    failed_untrusted = compute_reimbursement_award(
        {
            **_run("run-failed-untrusted", "generalist", 10, 8),
            "valid_receipt": True,
            "verified_loop_start_payment": True,
            "miner_openrouter_key_present": True,
            "trusted_cost_ledger": False,
            "passed_abuse_checks": True,
        },
        generalist,
        policy,
        no_usage,
    )
    if int(failed_untrusted.target_reimbursement_microusd) != 0 or failed_untrusted.status != "ineligible":
        raise AssertionError("untrusted failed-run ledger must remain non-payable")

    capped = cap_reimbursement_schedules_by_epoch(
        schedules[:2],
        max_alpha_percent_per_epoch=0.20,
    )
    original_total = sum(
        int(entry["amount_microusd"])
        for schedule in schedules[:2]
        for entry in schedule["entries"]
    )
    capped_total = sum(
        int(entry["amount_microusd"])
        for schedule in capped
        for entry in schedule["entries"]
    )
    if capped_total != original_total:
        raise AssertionError("capped schedules must preserve total reimbursement")
    for epoch in {int(entry["epoch"]) for schedule in capped for entry in schedule["entries"]}:
        epoch_alpha = sum(
            float(entry.get("alpha_percent") or 0)
            for schedule in capped
            for entry in schedule["entries"]
            if int(entry["epoch"]) == epoch
        )
        if epoch_alpha > 0.200001:
            raise AssertionError("capped schedules exceeded per-epoch alpha cap")
    _run_lab_allocator_simulations()


def _run_lab_allocator_simulations() -> None:
    policy = {
        "research_lab_emission_percent": 20.0,
        "fulfillment_emission_percent": 70.5,
        "fulfillment_leaderboard_emission_percent": 9.5,
        "reward_epochs": 20,
        "usd_per_0_1_percent_epoch": 0.162,
        "reimbursement_allow_overpay_without_champions": True,
        "reimbursement_max_cost_multiplier_with_champions": 1.0,
        "champion_min_alpha_percent": 4.0,
        "champion_extra_alpha_percent_per_point": 0.2,
        "champion_max_alpha_percent": 10.0,
        "champion_placeholder_alpha_percent": 0.0001,
        "champion_queue_trigger_ratio": 0.50,
        "champion_threshold_points": 2.0,
        "champion_eval_days": 10,
        "champion_icps_per_day": 6,
        "lab_score_per_alpha_percent": 1,
    }

    no_champion_reimbursements = [
        _reimbursement_obligation(uid, spend_usd=20 if uid <= 5 else 10)
        for uid in range(1, 11)
    ]
    allocation = allocate_research_lab_epoch(100, policy, no_champion_reimbursements, [])
    _assert_cap(allocation, 20.0)
    high = _paid_for_uid(allocation["reimbursement_allocations"], 1)
    low = _paid_for_uid(allocation["reimbursement_allocations"], 6)
    _assert_close(high, low * 2, "no-champion reimbursement should be spend-proportional")
    _assert_close(sum(item["paid_alpha_percent"] for item in allocation["reimbursement_allocations"]), 20.0, "no-champion reimbursement should use full lab cap")
    if all(item["overpaid_alpha_percent"] <= 0 for item in allocation["reimbursement_allocations"]):
        raise AssertionError("no-champion reimbursement should allow overpayment")

    two_champions = [
        _champion_obligation(101, start_epoch=100, improvement_points=4.0),
        _champion_obligation(102, start_epoch=100, improvement_points=7.0),
    ]
    three_reimbursements = [_reimbursement_obligation(uid, spend_usd=10) for uid in range(1, 4)]
    allocation = allocate_research_lab_epoch(100, policy, three_reimbursements, two_champions)
    _assert_cap(allocation, 20.0)
    _assert_close(allocation["unallocated_percent"], 0.0, "active winners should receive all non-reimbursed lab capacity")
    if len(allocation["champion_allocations"]) != 2:
        raise AssertionError("two active champions should both fit with three $10 reimbursements")
    if allocation["reimbursement_alpha_percent"] <= 0:
        raise AssertionError("reimbursements should be paid before champion remainder")
    if allocation["champion_alpha_percent"] <= allocation["reimbursement_alpha_percent"]:
        raise AssertionError("remaining lab capacity should go to champions")
    _assert_close(
        allocation["reimbursement_alpha_percent"] + allocation["champion_alpha_percent"],
        20.0,
        "reimbursements plus winners should exhaust lab cap",
    )

    crowded_reimbursements = [_reimbursement_obligation(uid, spend_usd=500) for uid in range(1, 31)]
    allocation = allocate_research_lab_epoch(100, policy, crowded_reimbursements, two_champions)
    _assert_cap(allocation, 20.0)
    _assert_close(allocation["unallocated_percent"], 0.0, "crowded reimbursements should not leave lab capacity unassigned")
    if allocation["reimbursement_alpha_percent"] >= 16.0:
        raise AssertionError("crowded reimbursements should scale down to reserve champion capacity")
    if not allocation["queued_champion_allocations"]:
        raise AssertionError("queue trigger should queue additional champions when reimbursements crowd champion capacity")
    if min(item["paid_alpha_percent"] for item in allocation["queued_champion_allocations"]) <= 0:
        raise AssertionError("queued champions should receive a positive placeholder")

    quiet_and_generalist = [
        _reimbursement_obligation(1, spend_usd=500, island="generalist", island_weight=1.0),
        _reimbursement_obligation(2, spend_usd=500, island="quiet", island_weight=2.0),
    ]
    allocation = allocate_research_lab_epoch(100, policy, quiet_and_generalist, two_champions)
    if _paid_for_uid(allocation["reimbursement_allocations"], 2) <= _paid_for_uid(allocation["reimbursement_allocations"], 1):
        raise AssertionError("low-participation island weight should increase reimbursement share")

    overflow_champions = [
        _champion_obligation(201, start_epoch=90, improvement_points=100.0),
        _champion_obligation(202, start_epoch=91, improvement_points=100.0),
        _champion_obligation(203, start_epoch=92, improvement_points=100.0),
    ]
    allocation = allocate_research_lab_epoch(100, policy, [], overflow_champions)
    _assert_cap(allocation, 20.0)
    _assert_close(allocation["unallocated_percent"], 0.0, "champion-only overflow should not leave lab capacity unassigned")
    if not allocation["queued_champion_allocations"]:
        raise AssertionError("champion overflow should queue later champions")
    if min(item["paid_alpha_percent"] for item in allocation["queued_champion_allocations"]) < 0:
        raise AssertionError("queued champion placeholder must be nonnegative")

    good_candidate = {
        "uid": 301,
        "miner_hotkey": "5Fchampion",
        "candidate_id": "candidate:abc",
        "run_id": "run-champion",
        "island": "generalist",
        "evaluation_epoch": 100,
        "improvement_points": 4.0,
        "daily_icp_counts": {f"2026-06-{day:02d}": 6 for day in range(1, 11)},
    }
    obligation = build_champion_reward_obligation(good_candidate, policy)
    _assert_close(obligation["desired_alpha_percent"], 4.4, "champion reward should be 4% + 0.2% per point over threshold")
    bad_candidate = {**good_candidate, "candidate_id": "candidate:bad", "daily_icp_counts": {"2026-06-01": 6}}
    blocked = build_champion_reward_obligation(bad_candidate, policy)
    if blocked["status"] != "blocked":
        raise AssertionError("missing daily ICP window should block champion obligation")

    hybrid_policy = {
        **policy,
        "champion_window_mode": "hybrid_fresh_retained",
        "champion_icps_per_day": 2,
        "champion_fresh_icp_count": 10,
        "champion_retained_icp_count": 10,
    }
    hybrid_candidate = {
        **good_candidate,
        "candidate_id": "candidate:hybrid",
        "daily_icp_counts": {
            "20260625": 1,
            "20260626": 1,
            "20260627": 1,
            "20260628": 1,
            "20260629": 1,
            "20260630": 1,
            "20260701": 1,
            "20260702": 2,
            "20260703": 1,
            "20260704": 10,
        },
    }
    hybrid_obligation = build_champion_reward_obligation(hybrid_candidate, hybrid_policy)
    if hybrid_obligation["status"] != "active":
        raise AssertionError(f"hybrid total-count champion obligation was not active: {hybrid_obligation}")
    hybrid_bad = {
        **hybrid_candidate,
        "candidate_id": "candidate:hybrid-bad",
        "daily_icp_counts": {"20260704": 19},
    }
    hybrid_blocked = build_champion_reward_obligation(hybrid_bad, hybrid_policy)
    if hybrid_blocked["status"] != "blocked" or "wrong_total_icp_count" not in hybrid_blocked["reasons"]:
        raise AssertionError("hybrid champion obligation must block wrong total ICP count")

    lab_allocation = allocate_research_lab_epoch(100, policy, three_reimbursements, two_champions)
    vector = compose_final_weight_vector(
        epoch=100,
        uids=[1, 2, 3, 101, 102],
        fulfillment_scores={"900": 70.5},
        leaderboard_scores={"901": 9.5},
        research_lab_allocation=lab_allocation,
        policy=policy,
    )
    if vector["weight_sum"] != 65535:
        raise AssertionError("composed weight vector should quantize to full u16 weight")

    for simulated_epoch in range(100, 125):
        active = list(no_champion_reimbursements)
        if simulated_epoch >= 101:
            active.extend(_reimbursement_obligation(uid, spend_usd=10, start_epoch=101) for uid in range(11, 13))
        champions = two_champions if simulated_epoch >= 105 else []
        allocation = allocate_research_lab_epoch(simulated_epoch, policy, active, champions)
        _assert_cap(allocation, 20.0)
        if simulated_epoch >= 121 and allocation["reimbursement_allocations"]:
            raise AssertionError("20-epoch reimbursement obligations should expire")


def _snapshot(island: str, score: int) -> dict[str, object]:
    return {
        "snapshot_id": f"participation:{island}:{score}",
        "island": island,
        "lookback_start": "2026-06-20T00:00:00Z",
        "lookback_end": "2026-06-21T00:00:00Z",
        "distinct_funded_hotkeys": score,
        "paid_loop_count": 0,
        "unique_brief_count": 0,
    }


def _run(run_id: str, island: str, funded_usd: float, actual_usd: float) -> dict[str, object]:
    return {
        "run_id": run_id,
        "miner_hotkey": f"5F{run_id}MinerHotkey",
        "island": island,
        "run_day": "2026-06-20",
        "funded_compute_budget_usd": funded_usd,
        "actual_openrouter_cost_usd": actual_usd,
        "loop_start_tao_fee_usd": 2.0,
        "paid_research_loop": True,
        "valid_receipt": True,
        "verified_loop_start_payment": True,
        "miner_openrouter_key_present": True,
        "trusted_cost_ledger": True,
        "passed_abuse_checks": True,
    }


def _reimbursement_obligation(
    uid: int,
    *,
    spend_usd: float,
    start_epoch: int = 100,
    island: str = "generalist",
    island_weight: float = 1.0,
) -> dict[str, object]:
    return {
        "uid": uid,
        "miner_hotkey": f"5Freimburse{uid}",
        "source_id": f"reimbursement:{uid}",
        "island": island,
        "start_epoch": start_epoch,
        "epoch_count": 20,
        "actual_openrouter_cost_usd": spend_usd,
        "island_weight": island_weight,
    }


def _champion_obligation(uid: int, *, start_epoch: int, improvement_points: float) -> dict[str, object]:
    return {
        "uid": uid,
        "miner_hotkey": f"5Fchampion{uid}",
        "source_id": f"champion:{uid}",
        "island": "generalist",
        "start_epoch": start_epoch,
        "epoch_count": 20,
        "improvement_points": improvement_points,
    }


def _paid_for_uid(allocations: list[dict[str, object]], uid: int) -> float:
    return sum(float(item["paid_alpha_percent"]) for item in allocations if int(item["uid"]) == uid)


def _assert_close(actual: float, expected: float, label: str, tolerance: float = 0.00001) -> None:
    if abs(float(actual) - float(expected)) > tolerance:
        raise AssertionError(f"{label}: expected {expected}, got {actual}")


def _assert_cap(allocation: dict[str, object], expected_cap: float) -> None:
    total = (
        float(allocation["reimbursement_alpha_percent"])
        + float(allocation["champion_alpha_percent"])
        + float(allocation["queued_champion_alpha_percent"])
        + float(allocation["unallocated_percent"])
    )
    _assert_close(total, expected_cap, "allocation must sum to lab cap", tolerance=0.0001)


if __name__ == "__main__":
    raise SystemExit(main())
