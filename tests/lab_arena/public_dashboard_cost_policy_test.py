from __future__ import annotations

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


def test_new_policy_projection_rejects_incomplete_successful_call_counters():
    row = _published_row(policy=True)
    ranking = row["publication_doc"]["final_ranking"][0]
    del ranking["cost_summary"]["execution"]["successful_calls"]

    baseline = public_dashboard.round_summary(row)["baseline"]

    assert "cost_summary" not in baseline
    assert "eligible" not in baseline


def test_per_icp_policy_projects_its_cap_and_omits_legacy_aggregate_cap():
    row = _published_row(policy=True)
    row["configuration_doc"]["sourcing_cost_eligibility_policy"] = (
        contracts.PER_ICP_SUCCESSFUL_CALLS_COST_POLICY
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
        "cost_per_company_cap_microusd": 800_000,
        "execution_cap_microusd": 80_000_000,
        "per_icp": [
            {
                "icp_position": position,
                "returned_company_count": 5,
                "qualified_company_count": 2,
                "competition_sourcing_microusd": 1_600_000 + (position == 2),
                "eligibility_cap_microusd": 1_600_000,
                "eligible": position != 2,
                "eligibility_reason": (
                    "cost_per_company_exceeded" if position == 2 else "eligible"
                ),
            }
            for position in range(20)
        ],
        "execution": _bucket(successful=True),
        "judge": _bucket(successful=True),
    }

    cost = public_dashboard.round_summary(row)["baseline"]["cost_summary"]

    assert cost["sourcing_cost_eligibility_policy"] == (
        "successful_calls_per_icp_v1"
    )
    assert cost["execution_icp_cap_microusd"] == 4_000_000
    assert cost["per_icp"][2]["eligible"] is False
    assert "execution_cap_microusd" not in cost
