from __future__ import annotations

import json

import pytest

from qualification.scoring import company_evidence_investigator as investigator
from tests.test_company_evidence_judgment_reserve import (
    QUOTE, URL, _Clock, _finding, _response, _run_industry,
)


def _validate(finding):
    return investigator._validated_findings(
        {"findings": [finding]},
        targets=("industry",),
        fetched_pages={URL: QUOTE},
        first_party_domains={"acme.example"},
        identity_names={"acme"},
        identity_anchor={
            "submitted_domain": "acme.example",
            "verified_domain": "acme.example",
        },
    )["industry"]


@pytest.mark.parametrize("status", ["VERIFIED", "CONTRADICTED"])
@pytest.mark.parametrize("url,quote", [
    ("", ""), ("", QUOTE), (URL, ""), (URL, " \n\t"),
    ("not-a-url", QUOTE), ("http://127.0.0.1/private", QUOTE),
])
def test_missing_citation_does_not_claim_quote_was_absent(status, url, quote):
    submitted = _finding(status=status, url=url, quote=quote)
    finding = _validate(submitted)

    assert finding["status"] == "UNPROVEN"
    assert finding["evidence_url"] == finding["evidence_quote"] == ""
    assert investigator._submitted_finding_rejection(submitted, finding) == (
        investigator.MISSING_CITATION_REASON
    )
    assert "no quote was checked" in finding["reason"]


def test_real_quote_validation_and_explicit_negative_are_unchanged():
    exact = _finding()
    assert _validate(exact)["status"] == "VERIFIED"
    assert investigator._submitted_finding_rejection(exact, _validate(exact)) is None

    fabricated = _finding(quote="Acme manufactures aircraft.")
    rejected = _validate(fabricated)
    assert rejected["status"] == "UNPROVEN"
    assert rejected["reason"] == "submitted quote was not present in fetched source"

    negative = _finding(status="UNPROVEN")
    assert _validate(negative)["status"] == "UNPROVEN"
    assert investigator._submitted_finding_rejection(negative, _validate(negative)) is None


@pytest.mark.parametrize("corrected_status", ["VERIFIED", "UNPROVEN"])
def test_missing_citation_repairs_loaded_evidence_without_new_research(
    monkeypatch, corrected_status,
):
    requests = []
    searches = []

    async def post(_session, _url, *, headers, payload):
        requests.append(payload)
        if len(requests) == 1:
            return 200, _response("submit_findings", {
                "findings": [_finding(url="", quote="")],
            })
        if len(requests) == 3:
            assert corrected_status == "UNPROVEN"
            assert payload["tool_choice"]["function"]["name"] == "search_web"
            return 200, _response("search_web", {"query": "Acme enrollment software"}, call=3)
        if len(requests) == 4:
            return 200, _response("submit_findings", {
                "findings": [_finding(status="UNPROVEN")],
            }, call=4)
        assert len(requests) == 2
        feedback = next(
            json.loads(message["content"])
            for message in reversed(payload["messages"])
            if message.get("role") == "tool"
        )
        assert feedback["rejected_findings"][0]["reason"] == (
            investigator.MISSING_CITATION_REASON
        )
        assert "No passage was rejected for those findings" in feedback["instruction"]
        assert "missing fields do not establish that the fact is absent" in (
            feedback["instruction"]
        )
        assert "submitted quote was not present" not in json.dumps(feedback)
        return 200, _response("submit_findings", {
            "findings": [_finding(status=corrected_status)],
        }, call=2)

    async def unexpected_research(*_args, **_kwargs):
        raise AssertionError("A missing citation must not force another provider fetch")

    async def required_negative_search(_session, query, **_kwargs):
        assert corrected_status == "UNPROVEN"
        searches.append(query)
        return {"ok": True, "results": []}

    monkeypatch.setattr(investigator, "_fetch_page", unexpected_research)
    monkeypatch.setattr(investigator, "_search_web", required_negative_search)
    result = _run_industry(monkeypatch, post, _Clock(), requested_product_service="")

    assert len(requests) == (2 if corrected_status == "VERIFIED" else 4)
    assert result["_completed_submit"] is True
    assert result["claims"]["industry"]["status"] == corrected_status
    assert result["usage"]["fetch_calls"] == 0
    assert result["usage"]["search_calls"] == len(searches) == (
        0 if corrected_status == "VERIFIED" else 1
    )


def test_repeated_missing_citations_do_not_extend_existing_turn_limit(monkeypatch):
    requests = []

    async def post(_session, _url, *, headers, payload):
        requests.append(payload)
        return 200, _response("submit_findings", {
            "findings": [_finding(url="", quote="")],
        }, call=len(requests))

    async def unexpected_research(*_args, **_kwargs):
        raise AssertionError("Missing fields must not cause recovery research")

    monkeypatch.setattr(investigator, "_fetch_page", unexpected_research)
    monkeypatch.setattr(investigator, "_search_web", unexpected_research)
    result = _run_industry(monkeypatch, post, _Clock(), requested_product_service="")

    assert len(requests) == investigator.MAX_REASONING_TURNS + 1
    # Preserve the existing bounded negative decision; never manufacture proof
    # or retry indefinitely when the model still supplies no citation.
    assert result["claims"]["industry"]["status"] == "UNPROVEN"
    assert result["claims"]["industry"]["reason"] == investigator.MISSING_CITATION_REASON
    assert result["usage"]["fetch_calls"] == result["usage"]["search_calls"] == 0
