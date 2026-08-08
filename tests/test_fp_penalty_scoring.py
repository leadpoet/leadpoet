"""False-positive penalty scoring: junk companies subtract from the ICP score.

A company zeroed by a model-controllable gate (or carrying an unverified
primary intent) applies -X penalty points to the ICP's pre-normalization
sum, so per-ICP scores fall toward zero without leaving the published
non-negative score scale. The current main penalty defaults to and is capped
at 10 points; historical bundles remain replayable because the verifier uses
the penalty points and score floor recorded in each bundle.
"""

import hashlib

import pytest

from research_lab.eval import evaluator
from leadpoet_verifier.research_evaluation import (
    build_research_evaluation_score_bundle,
    compute_evaluation_aggregates,
    normalize_current_fp_penalty_points,
    verify_research_evaluation_score_bundle,
)


def _sha(label: str) -> str:
    return "sha256:" + hashlib.sha256(label.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Penalty arithmetic (capped mode)
# ---------------------------------------------------------------------------

def test_penalty_subtracts_before_normalization(monkeypatch):
    monkeypatch.setenv("RESEARCH_LAB_EVAL_CAPPED_TOP5_SCORE", "true")
    # (80 - 50) / 3
    assert evaluator.benchmark_icp_score_from_company_scores(
        [80.0], requested_count=3, fp_penalty_total=50.0
    ) == pytest.approx(10.0)
    # No penalty -> unchanged
    assert evaluator.benchmark_icp_score_from_company_scores(
        [80.0], requested_count=3
    ) == pytest.approx(80.0 / 3)


def test_penalty_cannot_push_icp_below_zero(monkeypatch):
    monkeypatch.setenv("RESEARCH_LAB_EVAL_CAPPED_TOP5_SCORE", "true")
    # Penalties still apply, but the persisted/public score scale starts at 0.
    assert evaluator.benchmark_icp_score_from_company_scores(
        [], requested_count=1, fp_penalty_total=100.0
    ) == pytest.approx(0.0)
    # Catastrophic penalties clamp at the floor
    assert evaluator.benchmark_icp_score_from_company_scores(
        [], requested_count=1, fp_penalty_total=100000.0
    ) == pytest.approx(0.0)


def test_zero_floored_icp_drags_the_benchmark_mean(monkeypatch):
    monkeypatch.setenv("RESEARCH_LAB_EVAL_CAPPED_TOP5_SCORE", "true")
    monkeypatch.setenv("RESEARCH_LAB_EVAL_FP_PENALTY_POINTS", "25")
    rows = [
        {"candidate_company_scores": [80.0, 80.0, 80.0], "icp_company_goal": 3},
        # 4 junk companies exhaust the score but cannot make it negative.
        {
            "candidate_company_scores": [],
            "icp_company_goal": 2,
            "candidate_fp_gate_count": 4,
        },
    ]
    score = evaluator._benchmark_style_score(rows, "candidate_company_scores")
    assert score == pytest.approx((80.0 + 0.0) / 2)


def test_current_penalty_defaults_to_ten_and_caps_stale_stricter_config(monkeypatch):
    monkeypatch.setenv("RESEARCH_LAB_EVAL_CAPPED_TOP5_SCORE", "true")
    monkeypatch.delenv("RESEARCH_LAB_EVAL_FP_PENALTY_POINTS", raising=False)
    monkeypatch.delenv(
        "RESEARCH_LAB_EVAL_FP_UNVERIFIED_PRIMARY_PENALTY", raising=False
    )
    rows = [
        {
            "candidate_company_scores": [60.0],
            "icp_company_goal": 5,
            "candidate_fp_gate_count": 1,
            "candidate_fp_unverified_primary_count": 0,
        }
    ]
    assert evaluator._benchmark_style_score(
        rows, "candidate_company_scores"
    ) == pytest.approx(10.0)
    assert evaluator._fp_penalty_points() == pytest.approx(10.0)

    monkeypatch.setenv("RESEARCH_LAB_EVAL_FP_PENALTY_POINTS", "25")
    assert evaluator._fp_penalty_points() == pytest.approx(10.0)
    monkeypatch.setenv("RESEARCH_LAB_EVAL_FP_PENALTY_POINTS", "4")
    assert evaluator._fp_penalty_points() == pytest.approx(4.0)


def test_current_penalty_normalizer_is_bounded_and_stable():
    assert normalize_current_fp_penalty_points(None) == pytest.approx(10.0)
    assert normalize_current_fp_penalty_points("25") == pytest.approx(10.0)
    assert normalize_current_fp_penalty_points("7.5") == pytest.approx(7.5)
    assert normalize_current_fp_penalty_points("-1") == pytest.approx(0.0)
    assert normalize_current_fp_penalty_points("invalid") == pytest.approx(10.0)


def test_scoring_configuration_commits_effective_penalty(monkeypatch):
    from gateway.tee import scoring_executor

    monkeypatch.setattr(
        scoring_executor,
        "_manifest_configuration_env_names",
        lambda: ("RESEARCH_LAB_EVAL_FP_PENALTY_POINTS",),
    )
    monkeypatch.setenv("RESEARCH_LAB_EVAL_FP_PENALTY_POINTS", "25")
    snapshot = scoring_executor.configuration_snapshot()
    assert snapshot["environment"]["RESEARCH_LAB_EVAL_FP_PENALTY_POINTS"] == "10"


def test_side_specific_counts(monkeypatch):
    monkeypatch.setenv("RESEARCH_LAB_EVAL_CAPPED_TOP5_SCORE", "true")
    monkeypatch.setenv("RESEARCH_LAB_EVAL_FP_PENALTY_POINTS", "10")
    row = {
        "base_company_scores": [50.0],
        "candidate_company_scores": [50.0],
        "icp_company_goal": 5,
        "base_fp_gate_count": 5,        # only the base side is penalized
        "candidate_fp_gate_count": 0,
    }
    base = evaluator._benchmark_style_score([row], "base_company_scores")
    cand = evaluator._benchmark_style_score([row], "candidate_company_scores")
    assert base == pytest.approx(0.0)   # (50 - 50)/5
    assert cand == pytest.approx(10.0)  # 50/5


# ---------------------------------------------------------------------------
# FP taxonomy
# ---------------------------------------------------------------------------

def _bd(reason=None, details=None):
    row = {"final_score": 0.0 if reason else 42.0, "failure_reason": reason}
    if details is not None:
        row["intent_signals_detail"] = details
    return row


def test_penalizable_gate_reasons_counted():
    breakdowns = [
        _bd("Company is on the ICP exclusion list: acme"),
        _bd("required_attribute validation did not pass"),
        _bd("Country mismatch: 'Germany' vs ICP 'United States'"),
        _bd("Duplicate company: 'Acme' already scored this evaluation"),
        _bd("Data quality issue: company_website is example/placeholder"),
        _bd("Company verification failed: stage differs"),
        _bd("Intent fabrication detected (hardcoded date or generic claim)"),
    ]
    gate, primary = evaluator.count_penalizable_false_positives(
        breakdowns, icp_has_intent_signals=True
    )
    assert gate == 7
    assert primary == 0


def test_infra_failures_never_penalized():
    breakdowns = [
        _bd("LLM scoring error: timeout talking to provider"),
        _bd("Scorer error: HTTP 429 from provider"),
        _bd("Company verification failed: website unreachable: ClientError"),
    ]
    gate, primary = evaluator.count_penalizable_false_positives(
        breakdowns, icp_has_intent_signals=True
    )
    assert gate == 0 and primary == 0
    assert all(
        evaluator.scorer_breakdown_has_retryable_infrastructure_failure(row)
        for row in breakdowns
    )


def test_content_rejection_is_not_misclassified_as_retryable():
    breakdown = _bd("Company verification failed: company identity differs")
    assert not evaluator.scorer_breakdown_has_retryable_infrastructure_failure(
        breakdown
    )


def test_unverified_primary_intent_counted_separately():
    verified = [{"matched_icp_signal": 0, "after_decay": 12.0}]
    bonus_only = [{"matched_icp_signal": 1, "after_decay": 30.0}]
    breakdowns = [
        _bd(details=verified),     # primary verified -> not an FP
        _bd(details=bonus_only),   # only bonus verified -> unverified primary
    ]
    gate, primary = evaluator.count_penalizable_false_positives(
        breakdowns, icp_has_intent_signals=True
    )
    assert gate == 0
    assert primary == 1
    # ICPs with no intent signals never produce primary FPs.
    gate2, primary2 = evaluator.count_penalizable_false_positives(
        breakdowns, icp_has_intent_signals=False
    )
    assert primary2 == 0


def test_verifier_infrastructure_error_fails_open():
    # A primary rejected because the three-stage verifier CRASHED (provider
    # outage) must not count as falsified intent — fail open per company.
    errored = [
        {
            "matched_icp_signal": -1,
            "after_decay": 0.0,
            "judge_verdict": {
                "decision": "rejected_verifier_error",
                "rejection_reason": "three_stage_exception",
                "error_class": "ReadTimeout",
            },
        }
    ]
    content_rejected = [
        {
            "matched_icp_signal": -1,
            "after_decay": 0.0,
            "judge_verdict": {
                "decision": "rejected_three_stage",
                "rejection_reason": "claim_not_supported_by_source",
            },
        }
    ]
    gate, primary = evaluator.count_penalizable_false_positives(
        [_bd(details=errored), _bd(details=content_rejected)],
        icp_has_intent_signals=True,
    )
    assert gate == 0
    assert primary == 1  # only the content rejection counts


def test_outer_fabrication_label_cannot_hide_verifier_outage():
    # Exact production trace shape: the legacy outer label said fabrication,
    # while the structured verdict proved that the verifier was unavailable.
    breakdown = _bd(
        "Intent fabrication detected (hardcoded date or generic claim)",
        details=[
            {
                "matched_icp_signal": 0,
                "after_decay": 0.0,
                "judge_verdict": {
                    "decision": "rejected_verifier_error",
                    "pipeline_decision": "unavailable",
                    "rejection_reason": "stage3_llm_error:no_openrouter_key",
                },
            }
        ],
    )
    assert evaluator.scorer_breakdown_has_retryable_infrastructure_failure(
        breakdown
    )
    gate, primary = evaluator.count_penalizable_false_positives(
        [breakdown], icp_has_intent_signals=True
    )
    assert (gate, primary) == (0, 0)


def test_valid_companies_remain_positive_with_genuine_fp_penalty(monkeypatch):
    monkeypatch.setenv("RESEARCH_LAB_EVAL_CAPPED_TOP5_SCORE", "true")
    # Five returned companies: three score positively and two are genuine
    # false positives. The current 10-point penalty reduces, but does not
    # erase, earned score: (180 - 20) / 5 = 32.
    assert evaluator.benchmark_icp_score_from_company_scores(
        [70.0, 60.0, 50.0, 0.0, 0.0],
        requested_count=5,
        fp_penalty_total=20.0,
    ) == pytest.approx(32.0)


def test_production_shaped_verifier_outages_do_not_erase_valid_score(monkeypatch):
    monkeypatch.setenv("RESEARCH_LAB_EVAL_CAPPED_TOP5_SCORE", "true")
    unavailable = _bd(
        "Intent fabrication detected (hardcoded date or generic claim)",
        details=[
            {
                "matched_icp_signal": 0,
                "after_decay": 0.0,
                "judge_verdict": {
                    "decision": "rejected_verifier_error",
                    "pipeline_decision": "unavailable",
                    "rejection_reason": "stage3_llm_error:no_openrouter_key",
                },
            }
        ],
    )
    breakdowns = [
        _bd(),
        _bd(),
        unavailable,
        unavailable,
        _bd("Company verification failed: company identity differs"),
    ]
    penalty = evaluator.fp_penalty_total_from_breakdowns(
        breakdowns,
        {"intent_signals": ["required intent"]},
    )
    assert penalty == pytest.approx(10.0)
    assert evaluator.benchmark_icp_score_from_company_scores(
        [70.0, 60.0, 0.0, 0.0, 0.0],
        requested_count=5,
        fp_penalty_total=penalty,
    ) == pytest.approx(24.0)


def test_fp_penalty_total_helper(monkeypatch):
    monkeypatch.setenv("RESEARCH_LAB_EVAL_FP_PENALTY_POINTS", "25")
    monkeypatch.setenv("RESEARCH_LAB_EVAL_FP_UNVERIFIED_PRIMARY_PENALTY", "10")
    breakdowns = [
        _bd("Country mismatch: 'Chile' vs ICP 'United States'"),
        _bd(details=[{"matched_icp_signal": 1, "after_decay": 5.0}]),
    ]
    icp = {"intent_signals": ["hiring engineers"]}
    total = evaluator.fp_penalty_total_from_breakdowns(breakdowns, icp)
    assert total == pytest.approx(10.0 + 10.0)


def test_fake_intent_inherits_main_penalty_by_default(monkeypatch):
    # Falsified intent is a false positive like any other: the ONE main knob
    # penalizes it at the same rate unless explicitly overridden.
    monkeypatch.setenv("RESEARCH_LAB_EVAL_FP_PENALTY_POINTS", "25")
    monkeypatch.delenv(
        "RESEARCH_LAB_EVAL_FP_UNVERIFIED_PRIMARY_PENALTY", raising=False
    )
    breakdowns = [
        _bd(details=[{"matched_icp_signal": 1, "after_decay": 5.0}]),  # bonus only
    ]
    icp = {"intent_signals": ["hiring engineers"]}
    total = evaluator.fp_penalty_total_from_breakdowns(breakdowns, icp)
    assert total == pytest.approx(10.0)
    # Explicit override can weight deception harder than ordinary non-fit.
    monkeypatch.setenv("RESEARCH_LAB_EVAL_FP_UNVERIFIED_PRIMARY_PENALTY", "50")
    assert evaluator.fp_penalty_total_from_breakdowns(
        breakdowns, icp
    ) == pytest.approx(50.0)


# ---------------------------------------------------------------------------
# Verifier parity
# ---------------------------------------------------------------------------

def test_verifier_recomputes_with_recorded_penalties():
    rows = [
        {
            "icp_ref": "icp:a",
            "icp_hash": _sha("1"),
            "icp_company_goal": 2,
            "base_company_scores": [],
            "candidate_company_scores": [80.0, 80.0],
            "candidate_fp_gate_count": 2,
        },
    ]
    aggregates = compute_evaluation_aggregates(
        rows, fp_penalty_points=25.0
    )
    row = aggregates["per_icp_results"][0]
    # (160 - 2*25)/2 = 55
    assert row["candidate_per_icp_score"] == pytest.approx(55.0)
    assert row["candidate_fp_gate_count"] == 2
    assert aggregates["fp_penalty_points"] == pytest.approx(25.0)


def test_verifier_zero_knobs_matches_legacy():
    rows = [
        {
            "icp_ref": "icp:a",
            "icp_hash": _sha("1"),
            "icp_company_goal": 2,
            "base_company_scores": [],
            "candidate_company_scores": [80.0, 80.0],
            "candidate_fp_gate_count": 9,
        },
    ]
    aggregates = compute_evaluation_aggregates(rows)
    assert aggregates["per_icp_results"][0]["candidate_per_icp_score"] == pytest.approx(80.0)


def test_verifier_legacy_floor_remains_replayable():
    rows = [
        {
            "icp_ref": "icp:a",
            "icp_hash": _sha("1"),
            "icp_company_goal": 1,
            "base_company_scores": [],
            "candidate_company_scores": [],
            "candidate_fp_gate_count": 50,
        },
    ]
    aggregates = compute_evaluation_aggregates(rows, fp_penalty_points=100.0)
    assert aggregates["per_icp_results"][0]["candidate_per_icp_score"] == pytest.approx(-100.0)
    assert "fp_penalty_icp_floor" not in aggregates


def test_verifier_recorded_nonnegative_floor_applies():
    rows = [
        {
            "icp_ref": "icp:a",
            "icp_hash": _sha("1"),
            "icp_company_goal": 1,
            "base_company_scores": [],
            "candidate_company_scores": [],
            "candidate_fp_gate_count": 50,
        },
    ]
    aggregates = compute_evaluation_aggregates(
        rows,
        fp_penalty_points=100.0,
        fp_penalty_icp_floor=0.0,
    )
    assert aggregates["per_icp_results"][0]["candidate_per_icp_score"] == pytest.approx(0.0)
    assert aggregates["fp_penalty_icp_floor"] == pytest.approx(0.0)


def test_penalized_bundle_records_floor_and_verifies():
    policy = {
        "min_delta": 0.0,
        "min_successful_icps": 1,
        "max_hard_failures": 0,
        "min_candidate_score": 0.0,
        "fp_penalty_points": 25.0,
    }
    bundle = build_research_evaluation_score_bundle(
        run_id="run-1",
        ticket_id="ticket-1",
        miner_hotkey="hotkey-1",
        island="core",
        evaluation_epoch=7,
        parent_artifact_hash=_sha("a"),
        candidate_artifact_hash=_sha("b"),
        private_model_manifest_hash=_sha("c"),
        candidate_patch_hash=_sha("d"),
        icp_set_hash=_sha("e"),
        scoring_version="scoring-v1",
        evaluator_version="evaluator-v1",
        per_icp_results=[
            {
                "icp_ref": "icp:a",
                "icp_hash": _sha("1"),
                "icp_company_goal": 1,
                "base_company_scores": [],
                "candidate_company_scores": [],
                "candidate_fp_gate_count": 4,
            }
        ],
        evidence_bundle_refs=("evidence:1",),
        execution_trace_ref="trace:1",
        cost_ledger_ref="ledger:1",
        benchmark_split_ref="split:1",
        policy=policy,
        signature_ref="sig:1",
    )
    assert bundle["aggregates"]["candidate_score"] == pytest.approx(0.0)
    assert bundle["aggregates"]["fp_penalty_icp_floor"] == pytest.approx(0.0)
    verification = verify_research_evaluation_score_bundle(bundle, policy=policy)
    assert verification["passed"], verification["errors"]
