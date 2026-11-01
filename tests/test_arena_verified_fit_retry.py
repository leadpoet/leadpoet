"""A later verifier retry must not replace a completed company-fit decision."""
import asyncio
from copy import deepcopy

import pytest

from lab_arena import scoring
from qualification.scoring import intent_details, lead_scorer
from qualification.scoring.company_fit_decision import (
    company_fit_match, company_fit_mismatch, company_fit_unavailable,
)
from qualification.scoring.evaluation_clock import use_evaluation_date
from qualification.scoring.competition import CompetitionCompanyScorer
from tests.test_arena_intent_retry_cache import (
    _company, _company_model, _icp, _icp_model, _identity,
    _record_positive, _scorer,
)


def test_paragraph_retry_keeps_fit_and_its_funding_sources(monkeypatch):
    fits, funding, paragraphs = [], [], []
    source = "https://acme.example/funding"

    async def fit(*args, **kwargs):
        text = "Acme completed Series B. " + str(len(fits))
        value = company_fit_match(details={"dimension_evidence": {
            "identity": {"decision": "match", "web_identity_receipt": _identity()},
            "stage": {"decision": "match", "web_evidence": {
                "url": source, "quote": "Acme completed Series B.",
            }},
        }})
        kwargs["matched_company_source_sink"][source] = {
            "final_url": source, "text": text,
        }
        fits.append(value)
        return value

    async def signal(value, *args, evidence_signals, verdict_out, **kwargs):
        if value.matched_icp_signal == 0:
            funding.append(deepcopy(kwargs["verified_funding_stage_context"]))
        return _record_positive(value, evidence_signals, verdict_out)

    async def paragraph(*args, **kwargs):
        paragraphs.append(deepcopy(args[3]))
        return {"gate": "intent_details",
                "decision": "unavailable" if len(paragraphs) % 2 else "match"}

    monkeypatch.setattr(lead_scorer, "_verify_company_fit", fit)
    monkeypatch.setattr(lead_scorer, "_score_single_intent_signal", signal)
    monkeypatch.setattr(intent_details, "review_intent_details", paragraph)
    policy_scorer = _scorer(intent_details=True)
    buyer = {**_icp(), "intent_category": "FUNDING"}
    company = _company(intent_details="Acme raised funding and is hiring staff.")
    with use_evaluation_date("2026-09-25"):
        for invocation in range(2):
            rows = scoring.score_work_item(
                {"scored_run_id": "fit-retry-" + str(invocation)}, icp=buyer,
                companies=[company], scorer=policy_scorer, max_retries=2,
            )
            assert rows[0]["final_score"] == 80
            # A new work item re-verifies; a retry of its paragraph does not.
            assert len(fits) == invocation + 1
    assert len(paragraphs) == 4
    assert paragraphs[0] == paragraphs[1]
    assert paragraphs[2] == paragraphs[3]
    assert len(funding) == 2
    assert funding[0] != funding[1]


@pytest.mark.parametrize("decision", ["unavailable", "mismatch"])
def test_incomplete_or_negative_fit_is_not_promoted_or_retained(monkeypatch, decision):
    calls = []
    factory = company_fit_unavailable if decision == "unavailable" else company_fit_mismatch

    async def fit(*args, **kwargs):
        calls.append(1)
        return factory("No verified fit")

    monkeypatch.setattr(lead_scorer, "_verify_company_fit", fit)
    cache = {}
    for _ in range(2):
        result = asyncio.run(lead_scorer.score_company_competition_intent(
            _company_model(), _icp_model(), 0, 0, set(), integrity_policy=True,
            verified_company_fit_retry_cache=cache,
        ))
        assert result.final_score == 0
    assert len(calls) == 2 and cache == {}


def test_forced_failure_precedes_verified_fit_reuse(monkeypatch):
    async def unexpected(*args, **kwargs):
        pytest.fail("Forced failure cannot invoke a verifier")
    monkeypatch.setattr(lead_scorer, "_verify_company_fit", unexpected)
    result = asyncio.run(lead_scorer.score_company_competition_intent(
        _company_model(), _icp_model(), 0, 0, set(), integrity_policy=True,
        force_fail_reason="forced", verified_company_fit_retry_cache={},
    ))
    assert result.final_score == 0 and result.failure_reason == "forced"


def test_verified_fit_is_private_to_its_complete_input_scope(monkeypatch):
    caches = []

    async def score_company(**kwargs):
        caches.append(kwargs["verified_company_fit_retry_cache"])
        return {"final_score": 0.0, "failure_reason": "fixture"}

    monkeypatch.setattr(lead_scorer, "score_company_competition_intent", score_company)
    scope = {}
    first = CompetitionCompanyScorer(integrity_policy=True, scorer_policy={"version": 1})
    second = CompetitionCompanyScorer(integrity_policy=True, scorer_policy={"version": 2})
    company, buyer = _company(), _icp()

    def run(scorer=first, company=company, buyer=buyer):
        asyncio.run(scorer.score_with_breakdowns(
            [company], buyer, False, retry_evidence_scope=scope,
        ))

    with use_evaluation_date("2026-09-25"):
        run()
        run()
        run(company={**company, "company_website": "https://other.example"})
        run(buyer={**buyer, "required_attribute": "Different requirement"})
        run(scorer=second)
    with use_evaluation_date("2026-09-26"):
        run()
    assert caches[0] is caches[1]
    assert len({id(value) for value in caches}) == 5


def test_verified_snapshot_is_not_mutated_by_later_receipt_consumers(monkeypatch):
    cache, receipts = {}, []
    fit = company_fit_match(details={"dimension_evidence": {
        "identity": {"web_identity_receipt": _identity()},
    }})

    async def verify(*args, **kwargs):
        return fit

    async def signal(*args, **kwargs):
        return 80, 80, 1, 100, False, []

    async def paragraph(*args, **kwargs):
        receipts.append(deepcopy(args[3]))
        args[3]["dimension_evidence"]["identity"]["web_identity_receipt"]["observed_name"] = "changed"
        return {"gate": "intent_details", "decision": "unavailable"}

    monkeypatch.setattr(lead_scorer, "_verify_company_fit", verify)
    monkeypatch.setattr(lead_scorer, "score_company_competition_intent_signal", signal)
    monkeypatch.setattr(intent_details, "review_intent_details", paragraph)
    company = _company_model().model_copy(update={"intent_details": "Acme is hiring."})
    for _ in range(2):
        asyncio.run(lead_scorer.score_company_competition_intent(
            company, _icp_model(), 0, 0, set(), integrity_policy=True,
            verified_company_fit_retry_cache=cache,
        ))
    assert len(receipts) == 2 and receipts[0] == receipts[1]


def test_fit_exception_does_not_create_verified_snapshot(monkeypatch):
    async def verify(*args, **kwargs):
        raise TimeoutError("offline transport failure")
    monkeypatch.setattr(lead_scorer, "_verify_company_fit", verify)
    cache = {}
    with pytest.raises(TimeoutError):
        asyncio.run(lead_scorer.score_company_competition_intent(
            _company_model(), _icp_model(), 0, 0, set(), integrity_policy=True,
            verified_company_fit_retry_cache=cache,
        ))
    assert cache == {}


def test_non_integrity_scoring_keeps_existing_verification_behavior(monkeypatch):
    calls = []
    async def fit(*args, **kwargs):
        calls.append(1)
        return company_fit_match()
    async def signal(*args, **kwargs):
        return 0, 0, 0, 0, True, []
    monkeypatch.setattr(lead_scorer, "_verify_company_fit", fit)
    monkeypatch.setattr(lead_scorer, "score_company_competition_intent_signal", signal)
    cache = {}
    for _ in range(2):
        asyncio.run(lead_scorer.score_company_competition_intent(
            _company_model(), _icp_model(), 0, 0, set(),
            verified_company_fit_retry_cache=cache,
        ))
    assert len(calls) == 2 and cache == {}
