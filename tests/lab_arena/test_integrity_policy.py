"""Policy boundaries and anti-inflation properties independent of providers."""
from copy import deepcopy
import json

import pytest

from lab_arena import confirmation, contracts, integrity, scoring, verify
from tests.lab_arena.icp_fixtures import daily_icps


def fresh_icps():
    rows = deepcopy(daily_icps()[:5])
    for row in rows:
        row["prompt"] += " with a newly opened regional headquarters"
        row["intent_signal"] = "Opened a regional headquarters within six months"
        row["intent_signals"] = [row["intent_signal"]]
        row["intent_category"] = "EXPANSION"
    return rows


def test_icp_projection_removes_private_hints_and_keeps_buyer_requirements():
    original = daily_icps()[0]
    original.update({"excluded_companies": ["Acme"], "generation_notes": "SECRET",
                     "private": {"answer": "SECRET"}, "prompt": "Find suppliers that integrate with Acme"})
    original["intent_signals"] = [{"text": "Hiring", "intent_category": "HIRING",
        "max_age_days": 90, "verified_example_company": "SECRET", "generation_notes": "SECRET"}]
    projected = integrity.agent_visible_icp(original)
    assert "SECRET" not in json.dumps(projected)
    assert "verified_example_company" not in projected
    assert projected["employee_count"] == original["employee_count"]
    assert projected["excluded_companies"] == ["Acme"]
    assert projected["prompt"] == original["prompt"]
    assert projected["intent_signals"] == [{"text": "Hiring", "intent_category": "HIRING", "max_age_days": 90}]
    projected["employee_count"].append("1-10")
    assert "1-10" not in original["employee_count"]


@pytest.mark.parametrize("marker", [None, "", "future_version", True])
def test_unknown_policy_cannot_silently_fall_back(marker):
    with pytest.raises(ValueError):
        integrity.enabled({"integrity_policy": marker})
    assert not integrity.enabled({})


def test_confirmation_bank_is_distinct_committed_and_contains_no_answers():
    main = daily_icps()
    bank = confirmation.build_bank("arena-2026-09-12", fresh_icps(), main)
    raw = contracts.canonical_json(bank).encode()
    assert confirmation.read_bank(raw, round_id=bank["round_id"], digest=contracts.hash_bytes(raw)) == bank
    assert "verified_example_company" not in raw.decode()
    assert len(bank["icps"]) == 5
    assert len({row["icp_id"] for row in bank["icps"]}) == 5
    with pytest.raises(ValueError, match="hash mismatch"):
        confirmation.read_bank(raw + b" ", round_id=bank["round_id"], digest=contracts.hash_bytes(raw))
    with pytest.raises(ValueError, match="repeats"):
        confirmation.build_bank(bank["round_id"], main[:5], main)
    paraphrased = deepcopy(main[:5])
    for row in paraphrased:
        row["prompt"] = "Different wording of the same structured request"
    with pytest.raises(ValueError, match="repeats"):
        confirmation.build_bank(bank["round_id"], paraphrased, main)
    duplicate = fresh_icps()
    duplicate[-1] = {**duplicate[0], "icp_id": "changed-id"}
    with pytest.raises(ValueError, match="repeats"):
        confirmation.build_bank(bank["round_id"], duplicate, main)


def test_confirmation_cohort_is_fixed_top_three_eligible_main_improvements():
    entries = [{"submission_id": "baseline", "is_king": True, "final_score": 60}]
    entries += [{"submission_id": name, "is_king": False, "final_score": value}
                for name, value in (("low", 60.99), ("edge", 61), ("a", 70), ("b", 70), ("best", 80), ("cost-fail", 99))]
    eligibility = {row["submission_id"]: {"eligible": row["submission_id"] != "cost-fail"} for row in entries}
    cohort = confirmation.select_cohort(entries, eligibility)
    assert cohort["submission_ids"] == ["baseline", "best", "a", "b"]
    assert cohort["required"] is True
    edge = confirmation.select_cohort([entries[0], entries[2]], eligibility)
    assert edge["submission_ids"] == ["baseline", "edge"]
    none = confirmation.select_cohort(entries[:2], eligibility)
    assert none["required"] is False
    with pytest.raises(ValueError, match="baseline"):
        confirmation.select_cohort(entries[1:], eligibility)


def test_generation_draw_has_separate_identity_without_transmitting_main_bank(monkeypatch):
    from gateway.tasks import icp_generator
    requests = []
    async def generate(set_id, total_icps, *, generation_context):
        requests.append((set_id, total_icps, generation_context))
        return fresh_icps(), {}, "unused"
    monkeypatch.setattr(icp_generator, "generate_icps_with_openrouter", generate)
    for _ in range(2):
        result = confirmation.fresh_confirmation_icps(round_id="arena-2026-09-12", evaluation_date="2026-09-12", main_icps=daily_icps())
        assert len(result) == 5
    assert requests[0][:2] == (20260912, 20)
    assert requests[0][2] != requests[1][2]
    assert "verified_example_company" not in requests[0][2]
    assert daily_icps()[0]["prompt"] not in requests[0][2]


def test_integrity_retry_preserves_terminal_scores_and_original_sparse_positions():
    icp = daily_icps()[0]
    companies = [
        {"company_name": "Skip", "company_website": "https://skip.com", "employee_count": "1-10"},
        {"company_name": "Acme", "company_website": "https://acme.com", "employee_count": "51-200"},
        {"company_name": "Different alias", "company_website": "https://acme.com", "employee_count": "51-200"},
    ]
    calls = []
    def judge(batch, _icp, _reference):
        calls.append([row["company_name"] for row in batch])
        if len(calls) == 1:
            return [
                {"company_index": 1, "company_identity_key": "domain:acme.com|name:acme", "company_qualified": True, "duplicate_company": False, "final_score": 54, "verifier_gate_receipts": [{"gate": "company_fit", "decision": "match"}]},
                {"company_index": 2, "company_identity_key": "domain:acme.com|name:different alias", "company_qualified": False, "duplicate_company": False, "final_score": 0, "failure_reason": "infrastructure_error: provider timeout"},
            ]
        return [{"company_index": 0, "company_identity_key": "domain:acme.com|name:acme", "company_qualified": True, "duplicate_company": False, "final_score": 99, "verifier_gate_receipts": [{"gate": "company_fit", "decision": "match"}]}]
    judge.integrity_policy = True
    result = scoring.score_work_item({"scored_run_id": "run"}, icp=icp, companies=companies, scorer=judge)
    assert calls == [["Skip", "Acme", "Different alias"], ["Different alias"]]
    assert [row["company_index"] for row in result] == [1, 2]
    assert [row["final_score"] for row in result] == [54, 0]
    assert result[1]["duplicate_company"] and not result[1]["company_qualified"]


def test_integrity_receipts_reject_missing_fields_duplicate_credit_and_wrong_positions():
    icp = daily_icps()[0]
    companies = [{"employee_count": "51-200"}]
    valid = {"company_index": 0, "company_identity_key": "domain:a.com|name:a", "company_qualified": True, "duplicate_company": False, "final_score": 54}
    assert scoring.validate_breakdowns_for_item([valid], icp=icp, companies=companies, integrity_policy=True) == [valid]
    for bad in ({"final_score": 54}, {**valid, "company_index": 1}, {**valid, "duplicate_company": True}, {**valid, "final_score": 0}):
        with pytest.raises(scoring.ScoringError):
            scoring.validate_breakdowns_for_item([bad], icp=icp, companies=companies, integrity_policy=True)
