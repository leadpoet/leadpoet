"""A locally unrepresentable repair is not exhausted company-source evidence."""
import asyncio
from copy import deepcopy
import json
import os

import pytest

from lab_arena import scoring
from qualification.scoring import competition, intent_details as review, verification_helpers
from tests.test_intent_details_complete_protocol_correction import simple
from tests.test_intent_details_request_budget import compact, native_validate
from tests.lab_arena.test_lab_arena_scoring import make_icp, scored_company


CASES = [("citation", "user"), ("semantic", "user"),
         ("complete", "system"), ("citation", "system"), ("semantic", "system")]


def receipt_for(monkeypatch, repair, envelope, *, construction_error=False, invalid_reply=False):
    document, positive = simple()
    initial = deepcopy(positive)
    if repair == "complete":
        initial["facts_supported"] = False
    elif repair == "citation":
        initial["unit_grounding"][0]["evidence"][0]["quote"] = "unbound source text"
    else:
        initial["facts_supported"] = False
        initial["unit_grounding"][0]["status"] = "CONTRADICTED"
    if envelope == "user":
        document["padding"] = ""
        document["padding"] = "x" * (31_900 - len(compact(document)))
        assert len(compact(document)) == 31_900
    elif envelope == "system":
        if repair == "complete":
            monkeypatch.setattr(review, "_COMPLETE_RESPONSE_CORRECTION", "x" * 32_001)
        else:
            monkeypatch.setattr(review, "_" + repair + "_repair_prompt", lambda *args: "x" * 32_001)
    if construction_error:
        def broken(*args):
            raise ValueError("ordinary repair construction error")
        monkeypatch.setattr(review, "_citation_repair_user_prompt", broken)
    monkeypatch.setattr(review, "review_evidence", lambda *args, **kwargs: deepcopy(document))
    before = deepcopy(document)
    calls = []

    async def judge(prompt, **kwargs):
        for mode in (False, True):
            native_validate(prompt, system=kwargs["system_prompt"],
                            response_format=kwargs["response_format"], full_thinking=mode)
        calls.append(prompt)
        assert len(calls) <= 2
        if len(calls) == 1:
            return json.dumps(initial)
        assert invalid_reply
        return "not a valid correction"

    monkeypatch.setattr(verification_helpers, "openrouter_chat", judge)
    receipt = asyncio.run(review.review_intent_details(None, None, [], {}))
    assert document == before  # No repair-time clipping or reindexing.
    return receipt, calls


@pytest.mark.parametrize("repair,envelope", CASES)
def test_every_local_repair_bound_is_systemic_and_never_company_source_exhausted(monkeypatch, repair, envelope):
    receipt, calls = receipt_for(monkeypatch, repair, envelope)
    assert len(calls) == 1
    assert receipt["decision"] == "unavailable"
    assert receipt["failure_reason_code"] == "malformed_response"
    assert receipt["failure_class"] == "intent_details_evidence_unavailable"
    breakdown = {"company_qualified": False, "verifier_gate_receipts": [receipt]}
    assert not competition.scorer_breakdown_has_company_local_verification_failure(breakdown)
    for integrity in (False, True):
        assert competition.scorer_breakdown_has_retryable_infrastructure_failure(
            breakdown, integrity_policy=integrity)


@pytest.mark.parametrize("duplicate_names", [False, True])
@pytest.mark.parametrize("repair,envelope", CASES)
def test_native_exhaustion_raises_instead_of_materializing_company_zero(monkeypatch, repair, envelope, duplicate_names):
    receipt, calls = receipt_for(monkeypatch, repair, envelope)
    companies = [scored_company(0)] * (2 if duplicate_names else 1)
    row = {"company_qualified": False, "final_score": 0., "intent_signals_detail": [],
           "verifier_gate_receipts": [receipt]}
    attempts = []

    def scorer(selected, icp, is_reference_model):
        attempts.append(len(selected))
        return [deepcopy(row) for _ in selected]

    with pytest.raises(scoring.ScoringError, match="judge_exhausted attempts=2"):
        scoring.score_work_item({"scored_run_id": "local-repair-bound"}, icp=make_icp(0),
            companies=companies, scorer=scorer, max_retries=2)
    assert len(calls) == 1
    assert attempts == [len(companies), len(companies)]


@pytest.mark.parametrize("construction_error,invalid_reply", [(True, False), (False, True)])
def test_non_bound_construction_and_completed_citation_failure_keep_existing_semantics(monkeypatch, construction_error, invalid_reply):
    receipt, calls = receipt_for(monkeypatch, "citation", "none",
        construction_error=construction_error, invalid_reply=invalid_reply)
    assert len(calls) == (1 if construction_error else 2)
    assert receipt["failure_class"] == "intent_details_citation_unavailable"
    assert receipt["failure_reason_code"] == "malformed_response"
    assert competition.scorer_breakdown_has_company_local_verification_failure(
        {"verifier_gate_receipts": [receipt]})


@pytest.mark.parametrize("cached_company_policy", [False, True])
@pytest.mark.parametrize("repair,envelope", CASES)
def test_native_entrypoint_returns_judge_error_without_accepting_terminal_zero(monkeypatch, repair, envelope, cached_company_policy):
    """Real receipt/retry/entrypoint path; semantic scorer replies are fixtures."""
    from lab_arena import company_judgments, scorer_entrypoint
    from tests.lab_arena import company_judgments_test as cache_fixtures
    from tests.lab_arena.test_lab_arena_intent_details_real_judge import _scoring_input
    from tests.test_arena_score_integrity import _positive_breakdown
    receipt, calls = receipt_for(monkeypatch, repair, envelope)
    if cached_company_policy:
        document = cache_fixtures._input()
        document["companies"] = document["companies"][:1]
        document["company_judgment_cache"] = {
            "schema_version": company_judgments.LEASE_SCHEMA_VERSION,
            "hits": [], "misses": [cache_fixtures._miss(cache_fixtures._refs(document)[0])],
        }
    else:
        document = _scoring_input()
    attempts = []

    class FixtureScorer:
        integrity_policy = cached_company_policy
        company_quality = cached_company_policy
        contacts_required = False

        def __call__(self, companies, icp, is_reference_model):
            attempts.append(len(companies))
            rows = []
            for company in companies:
                row = _positive_breakdown(company["company_name"], company["company_website"].split("//", 1)[1].rstrip("/"))
                row.update(company_index=0, company_qualified=False, final_score=0.,
                    failure_reason="Intent Details verification unavailable: local input bound")
                row["verifier_gate_receipts"].append(deepcopy(receipt))
                rows.append(row)
            return rows

    # Preserve the caller's environment after the real entrypoint installs its
    # policy/placeholder provider settings. No credential or provider is used.
    monkeypatch.setattr(os, "environ", os.environ.copy())
    for name in scoring.CREDENTIAL_ENV_NAMES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(scoring, "lab_scorer", lambda *args, **kwargs: FixtureScorer())
    result = scorer_entrypoint.score_input(document)
    assert result["failure"] == "judge_error"
    assert result["reason"] == "malformed_response"
    assert not result.get("breakdowns") and not result.get("terminal_breakdowns")
    assert not result.get("company_judgments")
    assert len(attempts) == scoring.MAX_JUDGE_RETRIES
    assert len(calls) == 1
