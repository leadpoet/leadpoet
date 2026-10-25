from __future__ import annotations

import pytest

from lab_arena import contracts, public_dashboard


HOTKEY = "5" + "A" * 47


def _bucket(*, successful: bool) -> dict:
    counters = {
        "settled_microusd": 2_000,
        "reserved_or_uncertain_microusd": 1_000,
        "conservative_microusd": 3_000,
        "inflight_calls": 0,
        "uncertain_calls": 1,
        "refused_calls": 2,
        "call_count": 5,
    }
    if successful:
        counters.update(
            {
                "successful_microusd": 125,
                "successful_calls": 1,
                "success_unresolved_microusd": 5,
                "success_unresolved_calls": 1,
            }
        )
    return {
        **counters,
        "providers": [{"provider": "openrouter", **counters}],
    }


def _published_row(*, policy: bool) -> dict:
    execution = _bucket(successful=True)
    judge = _bucket(successful=True)
    ranking = {
        "rank": 1,
        "submission_id": "baseline-2026-09-13",
        "final_score": 60.0,
        "is_baseline": True,
        "eligible": False,
        "eligibility_reason": "provider_cost_uncertain",
        "cost_summary": {
            "returned_company_count": 5,
            "execution_cap_microusd": 80_000_000,
            "cost_per_company_cap_microusd": 800_000,
            "eligibility_cap_microusd": 4_000_000,
            "execution": execution,
            "judge": judge,
            "private_detail": "do-not-project",
        },
    }
    configuration = {"mode": "live"}
    if policy:
        configuration["sourcing_cost_eligibility_policy"] = (
            contracts.SUCCESSFUL_CALLS_COST_POLICY
        )
    return {
        "round_id": "arena-2026-09-13",
        "status": "published",
        "configuration_doc": configuration,
        "participants": [
            {
                "submission_id": ranking["submission_id"],
                "miner_hotkey": HOTKEY,
                "is_king": True,
            }
        ],
        "publication_doc": {
            "participants": [
                {
                    "submission_id": ranking["submission_id"],
                    "miner_hotkey": HOTKEY,
                    "is_baseline": True,
                }
            ],
            "final_ranking": [ranking],
            "king_decision": {
                "outcome": "no_king",
                "winner_submission_id": None,
                "king_submission_id": None,
                "king_hotkey": "",
            },
        },
    }


def test_new_policy_projects_competition_sourcing_beside_actual_billing():
    baseline = public_dashboard.round_summary(_published_row(policy=True))["baseline"]

    cost = baseline["cost_summary"]
    assert cost["sourcing_cost_eligibility_policy"] == "successful_calls_v1"
    assert cost["competition_sourcing_microusd"] == 130
    assert cost["execution"]["conservative_microusd"] == 3_000
    assert cost["execution"]["successful_microusd"] == 125
    assert cost["execution"]["success_unresolved_microusd"] == 5
    assert cost["judge"]["uncertain_calls"] == 1
    assert "private_detail" not in cost


def test_marker_absent_round_retains_historical_cost_projection_shape():
    baseline = public_dashboard.round_summary(_published_row(policy=False))["baseline"]

    cost = baseline["cost_summary"]
    assert "training_data_eligibility" not in baseline
    assert set(cost) == {
        "returned_company_count",
        "execution_cap_microusd",
        "cost_per_company_cap_microusd",
        "eligibility_cap_microusd",
        "execution",
        "judge",
    }
    expected_bucket_keys = {
        "settled_microusd",
        "reserved_or_uncertain_microusd",
        "conservative_microusd",
        "inflight_calls",
        "uncertain_calls",
        "refused_calls",
        "call_count",
        "providers",
    }
    assert set(cost["execution"]) == expected_bucket_keys
    assert set(cost["execution"]["providers"][0]) == {
        "provider",
        *(expected_bucket_keys - {"providers"}),
    }


def test_training_eligibility_projects_only_fixed_reasons_and_counts():
    row = _published_row(policy=True)
    ranking = row["publication_doc"]["final_ranking"][0]
    ranking.update(eligible=False, eligibility_reason="training_data_incomplete")
    ranking["training_data_eligibility"] = {
        "eligible": False,
        "reasons": ["full_thinking_call_missing", "source_review_not_passed"],
        "source_review": {
            "eligible": False,
            "reasons": ["source_review_not_passed"],
            "counts": {"enabled": 1, "reviewed": 0, "private_count": 999},
            "reviewer_excerpt": "private source text",
        },
        "capture": {
            "eligible": False,
            "reasons": ["full_thinking_call_missing"],
            "counts": {"enabled": 1, "selected_runs": 20, "checked_runs": 19,
                       "model_calls": 40, "full_thinking_calls": 0,
                       "tool_only_empty_calls": 1},
            "raw_trace": "private reasoning text",
        },
        "metadata": {"private": "do not publish"},
    }

    baseline = public_dashboard.round_summary(row)["baseline"]

    assert baseline["eligible"] is False
    assert baseline["eligibility_reason"] == "training_data_incomplete"
    assert baseline["training_data_eligibility"] == {
        "eligible": False,
        "reasons": ["full_thinking_call_missing", "source_review_not_passed"],
        "source_review": {
            "eligible": False,
            "reasons": ["source_review_not_passed"],
            "counts": {"enabled": 1, "reviewed": 0},
        },
        "capture": {
            "eligible": False,
            "reasons": ["full_thinking_call_missing"],
            "counts": {"enabled": 1, "selected_runs": 20, "checked_runs": 19,
                       "model_calls": 40, "full_thinking_calls": 0,
                       "tool_only_empty_calls": 1},
        },
    }
    assert "private source text" not in str(baseline)
    assert "private reasoning text" not in str(baseline)


def test_training_eligibility_drops_unrecognized_reason_text():
    row = _published_row(policy=True)
    ranking = row["publication_doc"]["final_ranking"][0]
    ranking.update(eligible=False, eligibility_reason="training_data_incomplete")
    ranking["training_data_eligibility"] = {
        "eligible": False,
        "reasons": ["raw secret: should never publish"],
    }

    baseline = public_dashboard.round_summary(row)["baseline"]

    assert baseline["eligibility_reason"] == "training_data_incomplete"
    assert "training_data_eligibility" not in baseline


def test_training_coverage_failure_retains_reason_and_public_counts():
    row = _published_row(policy=True)
    ranking = row["publication_doc"]["final_ranking"][0]
    ranking.update(eligible=False, eligibility_reason="training_data_incomplete")
    counts = {"enabled": 1, "selected_runs": 10, "checked_runs": 10,
              "complete_runs": 7, "incomplete_runs": 3,
              "required_complete_runs": 8, "minimum_coverage_percent": 80}
    ranking["training_data_eligibility"] = {
        "eligible": False, "reasons": ["training_coverage_incomplete"],
        "capture": {"eligible": False, "reasons": ["training_coverage_incomplete"],
                    "counts": {**counts, "private_detail": "must not publish"}},
    }
    baseline = public_dashboard.round_summary(row)["baseline"]
    gate = baseline["training_data_eligibility"]
    assert gate["reasons"] == ["training_coverage_incomplete"]
    assert gate["capture"]["counts"] == counts
    assert "must not publish" not in str(baseline)



def test_training_output_read_gap_remains_visible_at_historical_full_coverage():
    row = _published_row(policy=True)
    ranking = row["publication_doc"]["final_ranking"][0]
    ranking.update(eligible=False, eligibility_reason="training_data_incomplete")
    ranking["training_data_eligibility"] = {
        "eligible": False, "reasons": ["accepted_output_unavailable"],
        "capture": {"eligible": False, "reasons": ["accepted_output_unavailable"],
                    "counts": {"complete_runs": 9, "incomplete_runs": 1,
                               "minimum_coverage_percent": 100}},
    }
    result = public_dashboard.round_summary(row)["baseline"]["training_data_eligibility"]
    assert result["reasons"] == ["accepted_output_unavailable"]
    assert result["capture"]["reasons"] == ["accepted_output_unavailable"]


@pytest.mark.parametrize("reason", [
    "missing_reasoning_share_exceeded", "selected_artifact_temporarily_unavailable",
])
def test_training_token_coverage_and_storage_failures_remain_public_without_run_details(reason):
    row = _published_row(policy=True)
    ranking = row["publication_doc"]["final_ranking"][0]
    ranking.update(eligible=False, eligibility_reason="training_data_incomplete")
    counts = {
        "complete_runs": 7, "incomplete_runs": 3, "minimum_coverage_percent": 80,
        "completed_untruncated_verified_reasoning_tokens": 10000,
        "missing_reasoning_tokens": 101, "missing_reasoning_calls": 2,
        "unknown_reasoning_usage_calls": 1, "truncated_model_calls": 3,
        "verified_failed_model_attempts": 4, "maximum_missing_reasoning_percent": 1,
    }
    ranking["training_data_eligibility"] = {
        "eligible": False, "reasons": [reason],
        "capture": {"eligible": False, "reasons": [reason], "counts": {
            **counts, "reasoning_by_run": [{"run_id": "private-run-id"}],
            "private_detail": "private trace content",
        }},
    }

    baseline = public_dashboard.round_summary(row)["baseline"]
    result = baseline["training_data_eligibility"]
    assert result["reasons"] == [reason]
    assert result["capture"]["reasons"] == [reason]
    assert result["capture"]["counts"] == counts
    assert "private-run-id" not in str(baseline)
    assert "private trace content" not in str(baseline)


def test_new_policy_projection_rejects_incomplete_successful_call_counters():
    row = _published_row(policy=True)
    ranking = row["publication_doc"]["final_ranking"][0]
    del ranking["cost_summary"]["execution"]["successful_calls"]

    baseline = public_dashboard.round_summary(row)["baseline"]

    assert "cost_summary" not in baseline
    assert "eligible" not in baseline


@pytest.mark.parametrize('policy,company_cap,eligibility_cap,reason', [
    (contracts.PER_ICP_SUCCESSFUL_CALLS_COST_POLICY, 800_000, 1_600_000, 'cost_per_company_exceeded'),
    (contracts.FLAT_PER_ICP_SUCCESSFUL_CALLS_COST_POLICY, 4_000_000, 4_000_000, 'execution_cap_exceeded'),
])
def test_per_icp_policy_projects_its_cap_and_omits_legacy_aggregate_cap(policy, company_cap, eligibility_cap, reason):
    row = _published_row(policy=True)
    row["configuration_doc"]["sourcing_cost_eligibility_policy"] = (
        policy
    )
    ranking = row["publication_doc"]["final_ranking"][0]
    ranking["eligible"] = True
    ranking["eligibility_reason"] = "eligible"
    ranking["cost_summary"] = {
        "returned_company_count": 100,
        "qualified_company_count": 40,
        "eligible_icp_count": 19,
        "competition_sourcing_microusd": 9_000_000,
        "execution_icp_cap_microusd": 4_000_000,
        "cost_per_company_cap_microusd": company_cap,
        "execution_cap_microusd": 80_000_000,
        "per_icp": [
            {
                "icp_position": position,
                "returned_company_count": 5,
                "qualified_company_count": 2,
                "competition_sourcing_microusd": eligibility_cap + (position == 2),
                "eligibility_cap_microusd": eligibility_cap,
                "eligible": position != 2,
                "eligibility_reason": (
                    reason if position == 2 else "eligible"
                ),
            }
            for position in range(20)
        ],
        "execution": _bucket(successful=True),
        "judge": _bucket(successful=True),
    }

    cost = public_dashboard.round_summary(row)["baseline"]["cost_summary"]

    assert cost["sourcing_cost_eligibility_policy"] == policy
    assert cost["cost_per_company_cap_microusd"] == company_cap
    assert cost["per_icp"][2]["eligibility_cap_microusd"] == eligibility_cap
    assert cost["per_icp"][2]["eligibility_reason"] == reason
    assert cost["execution_icp_cap_microusd"] == 4_000_000
    assert cost["per_icp"][2]["eligible"] is False
    assert "execution_cap_microusd" not in cost
