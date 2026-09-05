"""Scoring plan, policy binding, once-per-output execution, and bundles (labarena.md 12.1, 18.7)."""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from lab_arena import contracts, scoring, verify

ROUND = "arena-2026-09-02"


def make_icp(position: int) -> dict:
    return {
        "icp_id": "arena:%s:b1:%d" % (ROUND, position), "prompt": "p%d" % position, "industry": "Software", "sub_industry": "SaaS",
        "employee_count": ["11-50", "51-200", "201-500"], "company_stage": "Series A", "geography": "United States", "country": "United States",
        "product_service": "x", "intent_signals": ["Announced a funding round"], "intent_signal": "Announced a funding round",
        "max_companies": 5, "excluded_companies": ["excluded.example.com"],
    }


def company(index: int, bucket: str = "51-200") -> dict:
    return {"company_name": "Co %d" % index, "company_website": "https://co%d.example.com" % index, "industry": "Software", "employee_count": bucket, "country": "United States", "intent_signals": []}


def breakdown(score: float, reason: str = "") -> dict:
    row = {"final_score": score, "failure_reason": reason, "intent_signals_detail": [], "verifier_gate_receipts": [], "proof_quote": "secret evidence text"}
    return row


def fake_scorer(counter, delay=0.0, fail_first=0):
    calls = {"n": 0}

    def score(companies, icp, is_reference_model):
        assert is_reference_model is False
        with counter["lock"]:
            counter["executions"] += 1
        calls["n"] += 1
        if calls["n"] <= fail_first:
            return [breakdown(0.0, "intent verification unavailable: provider timeout")]
        if delay:
            time.sleep(delay)
        # The Lab scorer skips out-of-bucket companies without consuming a slot.
        scored, _ = verify.bucket_skip(icp, companies)
        return [breakdown(60.0 + position) for position, _ in enumerate(scored)]

    return score


def run_plan(plan, *, icps_by_position, outputs_by_run, scorer, workers=1, existing=None):
    """Score every work item once in the test (the Arena has no central scoring runner: validators judge)."""

    from concurrent.futures import ThreadPoolExecutor
    from types import SimpleNamespace

    validated = contracts.validate_scoring_plan(plan)
    results = {key: [dict(row) for row in value] for key, value in (existing or {}).items()}
    pending = [item for item in validated["work_items"] if item["scored_run_id"] not in results]

    def _score(item):
        return item["scored_run_id"], scoring.score_work_item(item, icp=icps_by_position[int(item["icp_position"])], companies=outputs_by_run[item["scored_run_id"]], scorer=scorer)

    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
        for key, breakdowns in pool.map(_score, pending):
            results[key] = breakdowns
    return SimpleNamespace(breakdowns_by_item=results, judge_executions=len(pending))


def runs_for(submissions, stage=1, *, outputs=None, causes=None):
    positions = contracts.stage_positions(stage)
    rows = []
    for submission in submissions:
        for position in positions:
            cause = (causes or {}).get((submission, position))
            run_id = "%s:%d:%d" % (submission, position, 0 if cause == "preflight_failed" else 1)
            if cause == "preflight_failed":
                rows.append({"run_id": run_id, "submission_id": submission, "icp_position": position, "stage": stage, "attempt": 0, "status": "failed", "terminal_cause": cause, "output_ref": None})
            elif cause:
                rows.append({"run_id": run_id, "submission_id": submission, "icp_position": position, "stage": stage, "attempt": 1, "status": "failed", "terminal_cause": cause, "output_ref": None})
            else:
                rows.append({"run_id": run_id, "submission_id": submission, "icp_position": position, "stage": stage, "attempt": 1, "status": "accepted", "terminal_cause": "accepted", "output_ref": "arena/outputs/%s.json" % run_id})
    return rows


_ICPS = {position: make_icp(position) for position in range(30)}


def test_policy_is_plain_and_binds_environment_fail_closed():
    policy = scoring.build_scorer_policy()
    assert policy == scoring.build_scorer_policy()
    assert policy["env_bindings"]["RESEARCH_LAB_EVAL_FP_PENALTY_POINTS"] == "10" and policy["max_scored_companies"] == 0
    environ = {}
    credentials = {name: "secret-" + name for name in scoring.CREDENTIAL_ENV_NAMES}
    applied = scoring.apply_policy_to_environment(
        policy, environ=environ, credentials=credentials
    )
    assert applied == policy["scoring_adapter_version"]
    assert environ["RESEARCH_LAB_EVAL_CAPPED_TOP5_SCORE"] == "0" and environ["OPENROUTER_API_KEY"] == credentials["OPENROUTER_API_KEY"]
    with pytest.raises(scoring.ScorerPolicyConflict):
        scoring.apply_policy_to_environment(policy, environ={"RESEARCH_LAB_EVAL_FP_PENALTY_POINTS": "25"}, credentials=credentials)
    with pytest.raises(scoring.ScorerPolicyConflict):
        scoring.apply_policy_to_environment(policy, environ={}, credentials=dict(credentials, EXA_API_KEY=""))


def test_plan_makes_one_work_item_per_accepted_assignment_and_synthesizes_zero_rows():
    """Identical outputs are never shared: each miner's output is judged on its own keys (no result cache)."""

    causes = {("c2", 1): "model_timeout", ("c3", 2): "budget_exhausted"}
    runs = runs_for(["king", "c1", "c2", "c3"], causes=causes)
    plan = scoring.build_scoring_plan(round_id=ROUND, stage=1, runs=runs)
    assert len({item["scored_run_id"] for item in plan["work_items"]}) == len(plan["work_items"])
    # Every accepted assignment is one item: four submissions over ten ICPs, minus the two zero rows.
    assert len(plan["work_items"]) == 40 - 2 and len({(item["submission_id"], item["icp_position"]) for item in plan["work_items"]}) == 38
    assert plan["zero_rows"] == [{"submission_id": "c2", "icp_position": 1, "cause": "model_timeout"}, {"submission_id": "c3", "icp_position": 2, "cause": "budget_exhausted"}]
    with pytest.raises(contracts.ArenaContractError, match="infrastructure reason"):
        scoring.build_scoring_plan(round_id=ROUND, stage=1, runs=runs_for(["c9"], causes={("c9", 3): "lease_expired"}))


def test_sixteen_identical_outputs_are_judged_sixteen_times_with_identical_breakdowns():
    """No result cache across miners: identical outputs cost one judge execution each and score the same."""

    submissions = ["king"] + ["c%d" % i for i in range(15)]
    runs = runs_for(submissions)
    policy = scoring.build_scorer_policy()
    plan = scoring.build_scoring_plan(round_id=ROUND, stage=1, runs=runs)
    assert len(plan["work_items"]) == 160 and len({item["submission_id"] for item in plan["work_items"]}) == 16
    counter = {"executions": 0, "lock": threading.Lock()}
    companies = [company(i) for i in range(5)]
    outputs_by_run = {item["scored_run_id"]: companies for item in plan["work_items"]}
    results = run_plan(plan, icps_by_position=_ICPS, outputs_by_run=outputs_by_run, scorer=fake_scorer(counter, delay=0.001), workers=8)
    assert results.judge_executions == 160 and counter["executions"] == 160
    bundle = scoring.build_stage_scores(plan=plan, policy=policy, icps_by_position=_ICPS, outputs_by_run=outputs_by_run, breakdowns_by_item=results.breakdowns_by_item)
    assert len(bundle["rows"]) == 160
    per_position = {}
    for row in bundle["rows"]:
        per_position.setdefault(row["icp_position"], set()).add(contracts.document_hash(row["breakdowns"]))
    assert all(len(hashes) == 1 for hashes in per_position.values())
    assert all("proof_quote" not in b for row in bundle["rows"] for b in row["breakdowns"])
    assert len(set(bundle["submission_scores"].values())) == 1
    # A restart resumes from durable results without re-executing the judge.
    resumed = run_plan(plan, icps_by_position=_ICPS, outputs_by_run=outputs_by_run, scorer=fake_scorer(counter), workers=4, existing=results.breakdowns_by_item)
    assert resumed.judge_executions == 0 and counter["executions"] == 160


def test_judge_infrastructure_failures_retry_then_raise_never_zero():
    counter = {"executions": 0, "lock": threading.Lock()}
    item = {"scored_run_id": "run-c1-0", "icp_position": 0, "output_ref": "arena/outputs/run-c1-0.json", "submission_id": "c1"}
    result = scoring.score_work_item(item, icp=_ICPS[0], companies=[company(1)], scorer=fake_scorer(counter, fail_first=2))
    assert result[0]["final_score"] == 60.0 and counter["executions"] == 3
    with pytest.raises(scoring.ScoringError):
        scoring.score_work_item(item, icp=_ICPS[0], companies=[company(1)], scorer=fake_scorer(counter, fail_first=10))

    def broken(companies, icp, is_reference_model):
        raise RuntimeError("judge exploded")

    with pytest.raises(scoring.ScoringError):
        scoring.score_work_item(item, icp=_ICPS[0], companies=[company(1)], scorer=broken)


def test_stage_cut_uses_ten_then_ten_and_final_mean_uses_all_twenty():
    policy = scoring.build_scorer_policy()
    counter = {"executions": 0, "lock": threading.Lock()}
    companies = [company(1), company(2, "10,001+"), company(3)]  # the second is outside the buckets and is skipped
    runs = runs_for(["king", "c1"], causes={("c1", 4): "invalid_output"})
    plan = scoring.build_scoring_plan(round_id=ROUND, stage=1, runs=runs)
    outputs_by_run = {item["scored_run_id"]: companies for item in plan["work_items"]}
    result = run_plan(plan, icps_by_position=_ICPS, outputs_by_run=outputs_by_run, scorer=fake_scorer(counter))
    bundle = scoring.build_stage_scores(plan=plan, policy=policy, icps_by_position=_ICPS, outputs_by_run=outputs_by_run, breakdowns_by_item=result.breakdowns_by_item)
    king_row = [row for row in bundle["rows"] if row["submission_id"] == "king"][0]
    assert king_row["scored_company_indexes"] == [0, 2] and king_row["skipped_company_indexes"] == [1]
    expected = verify.per_icp_score(_ICPS[0], king_row["breakdowns"], policy)["per_icp_score"]
    assert king_row["per_icp_score"] == expected == (60.0 + 61.0) / 5
    assert bundle["stage"] == 1
    assert bundle["submission_scores"]["king"] == verify.stage_score([expected] * 10, 10) == expected
    assert bundle["submission_scores"]["c1"] == verify.stage_score([expected] * 9 + [0.0], 10)
    assert len([row for row in bundle["rows"] if row["submission_id"] == "c1"]) == 10
    records = scoring.run_scores_for_store(bundle, runs)
    assert len(records) == 20 and {r["per_icp_score"] for r in records if r["run_id"] == "c1:4:1"} == {0.0}

    stage_2_runs = runs_for(["king", "c1"], stage=2)
    stage_2_plan = scoring.build_scoring_plan(round_id=ROUND, stage=2, runs=stage_2_runs)
    stage_2_outputs = {item["scored_run_id"]: companies for item in stage_2_plan["work_items"]}
    stage_2_result = run_plan(stage_2_plan, icps_by_position=_ICPS, outputs_by_run=stage_2_outputs, scorer=fake_scorer(counter))
    stage_2_bundle = scoring.build_stage_scores(
        plan=stage_2_plan,
        policy=policy,
        icps_by_position=_ICPS,
        outputs_by_run=stage_2_outputs,
        breakdowns_by_item=stage_2_result.breakdowns_by_item,
    )
    assert stage_2_bundle["stage"] == 2
    assert len([row for row in stage_2_bundle["rows"] if row["submission_id"] == "c1"]) == 10
    all_rows = bundle["rows"] + stage_2_bundle["rows"]
    king_scores = [row["per_icp_score"] for row in all_rows if row["submission_id"] == "king"]
    challenger_scores = [row["per_icp_score"] for row in all_rows if row["submission_id"] == "c1"]
    assert verify.stage_score(king_scores, 20) == expected
    assert verify.stage_score(challenger_scores, 20) == verify.stage_score([expected] * 19 + [0.0], 20)
    assert len(scoring.run_scores_for_store(stage_2_bundle, stage_2_runs)) == 20


def test_exact_final_tie_crowns_no_miner():
    king = {"submission_id": "king", "hotkey": "king-hotkey", "final_score": 75.0, "is_king": True}
    challenger = {"submission_id": "c1", "hotkey": "challenger-hotkey", "final_score": 75.0, "is_king": False}
    assert verify.final_ranking([challenger, king])[0]["submission_id"] == "king"
    assert verify.king_decision([challenger], king)["outcome"] == "no_king"
    challenger["final_score"] = 75.000001
    decision = verify.king_decision([challenger], king)
    assert (decision["outcome"], decision["winner_submission_id"]) == ("crowned", "c1")
