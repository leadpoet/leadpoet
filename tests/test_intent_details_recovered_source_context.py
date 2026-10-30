"""A verified replacement source keeps its facts through paragraph review."""
from copy import deepcopy

import pytest

from qualification.scoring import intent_details
from tests.test_intent_details_fetched_first_party_context import _case, FUNDING


RECOVERED = "https://news.example/verified-funding-announcement"
BODY = FUNDING + " The agreement also includes a new customer operations program."


def recovered_case():
    company, icp, fit, signals = _case()
    signal = signals[0]
    signal["evidence_type"] = "PARTNERSHIP"
    trace = signal["judge_verdict"]["verification_trace"]
    evaluation = trace["intent_verdict"]["signal_evaluations"][0]
    evaluation.update(
        verification_mode="source_grounded",
        evidence_urls_used=[RECOVERED], source_urls_supplied=[RECOVERED],
    )
    trace["verified_source_context"] = [{
        "url": RECOVERED, "text": BODY, "source_title": "Funding announcement",
    }]
    trace["source_resolution"] = {"status": "verified", "selected_urls": [RECOVERED]}
    return company, icp, fit, signals


def context_entries(document):
    return [entry for entry in document["admitted_evidence"]
            if entry["evidence_kind"] == "verified_source_context"]


def test_same_event_replacement_retains_its_fetched_body_and_url():
    company, icp, fit, signals = recovered_case()
    before = deepcopy(signals)
    document = intent_details.review_evidence(company, icp, signals, fit.receipt("company_fit"))
    assert document["verified_signals"][0]["source_urls"] == [RECOVERED]
    assert any(entry["source_url"] == RECOVERED and BODY in entry["admitted_text"]
               for entry in context_entries(document))
    assert signals == before


@pytest.mark.parametrize("damage", [
    "not_supplied", "not_cited", "not_source_grounded", "unbound_quote",
    "different_context_url", "stage_role", "company_role", "unsafe_url",
    "no_recovery_receipt", "recovery_unverified", "not_selected",
])
def test_replacement_requires_same_evaluation_and_host_bound_source(damage):
    company, icp, fit, signals = recovered_case()
    trace = signals[0]["judge_verdict"]["verification_trace"]
    evaluation = trace["intent_verdict"]["signal_evaluations"][0]
    context = trace["verified_source_context"][0]
    if damage == "not_supplied":
        evaluation["source_urls_supplied"] = []
    elif damage == "not_cited":
        evaluation["evidence_urls_used"] = []
    elif damage == "not_source_grounded":
        evaluation["verification_mode"] = "search"
    elif damage == "unbound_quote":
        context["text"] = "An unrelated announcement with no funding information."
    elif damage == "different_context_url":
        context["url"] = "https://other.example/unrelated"
    elif damage in {"stage_role", "company_role"}:
        context["source_role"] = ("matched_company_stage" if damage == "stage_role"
                                  else "first_party_context")
    elif damage == "no_recovery_receipt":
        trace.pop("source_resolution")
    elif damage == "recovery_unverified":
        trace["source_resolution"]["status"] = "unavailable"
    elif damage == "not_selected":
        trace["source_resolution"]["selected_urls"] = []
    else:
        context["url"] = "http://127.0.0.1/private"
        evaluation["source_urls_supplied"] = evaluation["evidence_urls_used"] = [context["url"]]
        trace["source_resolution"]["selected_urls"] = [context["url"]]
    document = intent_details.review_evidence(company, icp, signals, fit.receipt("company_fit"))
    assert not context_entries(document)


def test_declared_source_projection_is_unchanged():
    company, icp, fit, signals = _case()
    document = intent_details.review_evidence(company, icp, signals, fit.receipt("company_fit"))
    assert context_entries(document)[0]["admitted_text"] == [FUNDING]
    assert document["verified_signals"][0]["source_urls"] == signals[0]["evidence_urls"]
