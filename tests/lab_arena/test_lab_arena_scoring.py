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


def runs_for(submissions, stage=1, *, outputs=None, causes=None):
    positions = range(0, 30)  # the one stage covers every benchmark slot
    rows = []
    for submission in submissions:
        for position in positions:
            cause = (causes or {}).get((submission, position))
            output_hash = (outputs or {}).get((submission, position), contracts.document_hash(["default-output"]))
            if cause == "preflight_failed":
                rows.append({"run_id": "%s:%d:0" % (submission, position), "submission_id": submission, "icp_position": position, "stage": stage, "attempt": 0, "status": "failed", "terminal_cause": cause, "icp_hash": icp_hashes()[position], "output_hash": None})
            elif cause:
                rows.append({"run_id": "%s:%d:1" % (submission, position), "submission_id": submission, "icp_position": position, "stage": stage, "attempt": 1, "status": "failed", "terminal_cause": cause, "icp_hash": icp_hashes()[position], "output_hash": None})
            else:
                rows.append({"run_id": "%s:%d:1" % (submission, position), "submission_id": submission, "icp_position": position, "stage": stage, "attempt": 1, "status": "accepted", "terminal_cause": "accepted", "icp_hash": icp_hashes()[position], "output_hash": output_hash})
    return rows


_ICPS = {position: make_icp(position) for position in range(30)}


def icp_hashes():
    return {position: contracts.document_hash(icp) for position, icp in _ICPS.items()}


def test_policy_is_signed_shaped_and_binds_environment_fail_closed():
    policy = scoring.build_scorer_policy()
    assert policy["policy_hash"].startswith("sha256:") and policy == scoring.build_scorer_policy()
    assert policy["env_bindings"]["RESEARCH_LAB_EVAL_FP_PENALTY_POINTS"] == "10" and policy["max_scored_companies"] == 0
    environ = {}
    credentials = {name: "secret-" + name for name in scoring.CREDENTIAL_ENV_NAMES}
    applied = scoring.apply_policy_to_environment(policy, environ=environ, cache_dir="/tmp/cache", credentials=credentials)
    assert applied == policy["policy_hash"] and environ["RESEARCH_LAB_SCORING_CACHE_DIR"] == "/tmp/cache"
    assert environ["RESEARCH_LAB_EVAL_CAPPED_TOP5_SCORE"] == "0" and environ["OPENROUTER_API_KEY"] == credentials["OPENROUTER_API_KEY"]
    with pytest.raises(scoring.ScorerPolicyConflict):
        scoring.apply_policy_to_environment(policy, environ={"RESEARCH_LAB_EVAL_FP_PENALTY_POINTS": "25"}, cache_dir="/tmp/cache", credentials=credentials)
    with pytest.raises(scoring.ScorerPolicyConflict):
        scoring.apply_policy_to_environment(policy, environ={}, cache_dir="", credentials=credentials)
    with pytest.raises(scoring.ScorerPolicyConflict):
        scoring.apply_policy_to_environment(policy, environ={}, cache_dir="/tmp/cache", credentials=dict(credentials, EXA_API_KEY=""))
    with pytest.raises(scoring.ScorerPolicyConflict):
        scoring.apply_policy_to_environment({k: v for k, v in policy.items() if k != "policy_hash"}, environ={}, cache_dir="/tmp/cache", credentials=credentials)


def test_plan_makes_one_work_item_per_accepted_assignment_and_synthesizes_zero_rows():
    """Identical outputs are never shared: each miner's output is judged on its own keys (no result cache)."""

    shared = contracts.document_hash(["king-output"])
    outputs = {("king", 0): shared, ("c1", 0): shared}
    causes = {("c2", 1): "model_timeout", ("c3", 2): "preflight_failed"}
    runs = runs_for(["king", "c1", "c2", "c3"], outputs=outputs, causes=causes)
    plan = scoring.build_scoring_plan(round_id=ROUND, stage=1, configuration_hash=contracts.document_hash("cfg"), commitment_hash=contracts.document_hash("cm"), scorer_policy_hash=scoring.build_scorer_policy()["policy_hash"], runs=runs, icp_hashes_by_position=icp_hashes())
    shared_items = [item for item in plan["work_items"] if item["output_hash"] == shared]
    assert sorted(item["submission_id"] for item in shared_items) == ["c1", "king"] and len({item["work_item_id"] for item in shared_items}) == 2
    # Every accepted assignment is one item: four submissions over 30 ICPs, minus the two zero rows.
    assert len(plan["work_items"]) == 120 - 2 and len({(item["submission_id"], item["icp_position"]) for item in plan["work_items"]}) == 118
    assert plan["zero_rows"] == [{"submission_id": "c2", "icp_position": 1, "cause": "model_timeout"}, {"submission_id": "c3", "icp_position": 2, "cause": "preflight_failed"}]
    with pytest.raises(contracts.ArenaContractError, match="infrastructure reason"):
        scoring.build_scoring_plan(round_id=ROUND, stage=1, configuration_hash=contracts.document_hash("cfg"), commitment_hash=contracts.document_hash("cm"), scorer_policy_hash=scoring.build_scorer_policy()["policy_hash"], runs=runs_for(["c9"], causes={("c9", 3): "lease_expired"}), icp_hashes_by_position=icp_hashes())


def test_sixteen_identical_outputs_are_judged_sixteen_times_with_identical_breakdowns():
    """No result cache across miners: identical outputs cost one judge execution each and score the same."""

    shared = contracts.document_hash(["shared"])
    submissions = ["king"] + ["c%d" % i for i in range(15)]
    outputs = {(submission, position): shared for submission in submissions for position in range(30)}
    runs = runs_for(submissions, outputs=outputs)
    policy = scoring.build_scorer_policy()
    plan = scoring.build_scoring_plan(round_id=ROUND, stage=1, configuration_hash=contracts.document_hash("cfg"), commitment_hash=contracts.document_hash("cm"), scorer_policy_hash=policy["policy_hash"], runs=runs, icp_hashes_by_position=icp_hashes())
    assert len(plan["work_items"]) == 480 and len({item["submission_id"] for item in plan["work_items"]}) == 16
    counter = {"executions": 0, "lock": threading.Lock()}
    companies = [company(i) for i in range(5)]
    results = scoring.run_scoring_plan(plan, icps_by_position=_ICPS, outputs_by_hash={shared: companies}, scorer=fake_scorer(counter, delay=0.001), workers=8)
    assert results.judge_executions == 480 and counter["executions"] == 480
    bundle = scoring.build_score_bundle(plan=plan, policy=policy, icps_by_position=_ICPS, outputs_by_hash={shared: companies}, breakdowns_by_item=results.breakdowns_by_item)
    assert len(bundle["rows"]) == 480
    per_position = {}
    for row in bundle["rows"]:
        per_position.setdefault(row["icp_position"], set()).add(contracts.document_hash(row["breakdowns"]))
    assert all(len(hashes) == 1 for hashes in per_position.values())
    assert all("proof_quote" not in b for row in bundle["rows"] for b in row["breakdowns"])
    assert len(set(bundle["submission_scores"].values())) == 1
    # A restart resumes from durable results without re-executing the judge.
    resumed = scoring.run_scoring_plan(plan, icps_by_position=_ICPS, outputs_by_hash={shared: companies}, scorer=fake_scorer(counter), workers=4, existing=results.breakdowns_by_item)
    assert resumed.judge_executions == 0 and counter["executions"] == 480


def test_judge_infrastructure_failures_retry_then_raise_never_zero():
    counter = {"executions": 0, "lock": threading.Lock()}
    item = {"work_item_id": contracts.document_hash("w"), "icp_position": 0, "icp_hash": icp_hashes()[0], "output_hash": contracts.document_hash("o"), "submission_id": "c1"}
    result = scoring.score_work_item(item, icp=_ICPS[0], companies=[company(1)], scorer=fake_scorer(counter, fail_first=2))
    assert result[0]["final_score"] == 60.0 and counter["executions"] == 3
    with pytest.raises(scoring.ScoringError):
        scoring.score_work_item(item, icp=_ICPS[0], companies=[company(1)], scorer=fake_scorer(counter, fail_first=10))

    def broken(companies, icp, is_reference_model):
        raise RuntimeError("judge exploded")

    with pytest.raises(scoring.ScoringError):
        scoring.score_work_item(item, icp=_ICPS[0], companies=[company(1)], scorer=broken)


def test_one_stage_bundle_carries_thirty_icp_scores_and_run_records():
    policy = scoring.build_scorer_policy()
    out1 = contracts.document_hash(["o1"])
    counter = {"executions": 0, "lock": threading.Lock()}
    companies = [company(1), company(2, "10,001+"), company(3)]  # the second is outside the buckets and is skipped
    runs = runs_for(["king", "c1"], outputs={(s, p): out1 for s in ("king", "c1") for p in range(30)}, causes={("c1", 4): "invalid_output"})
    plan = scoring.build_scoring_plan(round_id=ROUND, stage=1, configuration_hash=contracts.document_hash("cfg"), commitment_hash=contracts.document_hash("cm"), scorer_policy_hash=policy["policy_hash"], runs=runs, icp_hashes_by_position=icp_hashes())
    result = scoring.run_scoring_plan(plan, icps_by_position=_ICPS, outputs_by_hash={out1: companies}, scorer=fake_scorer(counter))
    bundle = scoring.build_score_bundle(plan=plan, policy=policy, icps_by_position=_ICPS, outputs_by_hash={out1: companies}, breakdowns_by_item=result.breakdowns_by_item)
    king_row = [row for row in bundle["rows"] if row["submission_id"] == "king"][0]
    assert king_row["scored_company_indexes"] == [0, 2] and king_row["skipped_company_indexes"] == [1]
    expected = verify.per_icp_score(_ICPS[0], king_row["breakdowns"], policy, icp_hash=icp_hashes()[0])["per_icp_score"]
    assert king_row["per_icp_score"] == expected == (60.0 + 61.0) / 5
    assert bundle["stage"] == 1 and "stage_1_bundle_hash" not in bundle
    assert bundle["submission_scores"]["king"] == verify.stage_score([expected] * 30, 30) == expected
    assert bundle["submission_scores"]["c1"] == verify.stage_score([expected] * 29 + [0.0], 30)
    assert len([row for row in bundle["rows"] if row["submission_id"] == "c1"]) == 30
    records = scoring.run_scores_for_store(bundle, runs, score_ref="ref")
    assert len(records) == 60 and {r["per_icp_score"] for r in records if r["run_id"] == "c1:4:1"} == {0.0}
    assert verify.validate_score_bundle(bundle)["bundle_hash"] == bundle["bundle_hash"]
    # The Arena has one stage: a plan for a second one is refused.
    with pytest.raises((scoring.ScoringError, contracts.ArenaContractError)):
        scoring.build_scoring_plan(round_id=ROUND, stage=2, configuration_hash=contracts.document_hash("cfg"), commitment_hash=contracts.document_hash("cm"), scorer_policy_hash=policy["policy_hash"], runs=runs, icp_hashes_by_position=icp_hashes())
