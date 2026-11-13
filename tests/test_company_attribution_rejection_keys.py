"""Repair diagnostics expose checked names without creating alias authority."""

import asyncio
import copy
import json

import pytest

from qualification.scoring import company_evidence_investigator as inv
from tests.test_company_evidence_judgment_reserve import _Clock, _finding, _response

URL = "https://vytexwindows.com/about-us/"
SHORT = "Vytex manufactures windows."
NAMED = "Vytex Corporation manufactures windows in its own plants."
TEXT = SHORT + " " + NAMED
KEYS = ["vytexcorporation", "vytexwindows"]


def _raw(quote=SHORT, **extra):
    return {**_finding(url=URL, quote=quote), "target": "required_attribute", **extra}


def _validate(raw, diagnostic):
    return inv._validated_findings(
        {"findings": [raw]}, targets=("required_attribute",),
        fetched_pages={URL: TEXT}, fetched_final_urls={URL: URL},
        first_party_domains={"vytexwindows.com"}, identity_names=set(KEYS),
        attribution_rejection_keys=diagnostic,
    )


def test_actual635_identity_keys_explain_short_name_rejection_without_accepting_alias():
    diagnostic = {}
    raw = _raw(company_identity_keys_checked=["vytex"])
    before = copy.deepcopy(raw)
    rejected = _validate(raw, diagnostic)["required_attribute"]
    assert rejected["reason"] == "source quote did not identify the investigated company"
    assert diagnostic == {"required_attribute": KEYS}
    assert "company_identity_keys_checked" not in rejected
    assert raw == before
    assert _validate(_raw(NAMED), diagnostic)["required_attribute"]["status"] == "VERIFIED"
    assert diagnostic == {}


def test_unrelated_citation_failures_clear_stale_attribution_diagnostics():
    diagnostic = {"required_attribute": ["stale"]}
    claim = _validate(_raw("Invented quote."), diagnostic)["required_attribute"]
    assert claim["reason"] == "submitted quote was not present in fetched source"
    assert diagnostic == {}


def test_explicit_unproven_does_not_receive_attribution_diagnostics():
    diagnostic = {"required_attribute": ["stale"]}
    raw = _raw(status="UNPROVEN", evidence_url="", evidence_quote="")
    assert _validate(raw, diagnostic)["required_attribute"]["status"] == "UNPROVEN"
    assert diagnostic == {}


@pytest.mark.parametrize("repair", ["named_positive", "honest_negative", "repeat_invalid"])
def test_support_slot_feedback_lists_actual_keys_and_preserves_native_outcomes(monkeypatch, repair):
    requests = []
    feedbacks = []
    monkeypatch.setenv("OPENROUTER_API_KEY", "offline-placeholder")
    monkeypatch.setenv("EXA_API_KEY", "offline-placeholder")
    monkeypatch.setattr(inv, "time", _Clock())
    monkeypatch.setattr(inv, "MAX_REASONING_TURNS", 1)

    async def post(_session, _url, *, headers, payload):
        requests.append(payload)
        assert len(requests) <= 2
        raw = _raw(NAMED, supporting_evidence_url_1=URL, supporting_evidence_quote_1=SHORT)
        if len(requests) == 2:
            feedback = next(json.loads(m["content"]) for m in reversed(payload["messages"]) if m["role"] == "tool")
            feedbacks.append(feedback)
            if repair == "named_positive":
                raw["supporting_evidence_quote_1"] = NAMED
            elif repair == "honest_negative":
                raw.update(status="UNPROVEN", evidence_url="", evidence_quote="")
        return 200, _response("submit_findings", {"findings": [raw]}, call=len(requests))

    async def forbidden(*_args, **_kwargs):
        pytest.fail("Attribution repair needs no new research")

    monkeypatch.setattr(inv, "_post_json", post)
    monkeypatch.setattr(inv, "_fetch_page", forbidden)
    monkeypatch.setattr(inv, "_search_web", forbidden)
    result = asyncio.run(inv.investigate_company_evidence(
        company_locator={"name": "Vytex Windows", "website": "https://vytexwindows.com"},
        targets=("required_attribute",), requested_attribute="Manufactures windows in its own plants",
        prior_observations={"observed_company_name": "Vytex Corporation", "submitted_source_urls": [URL]},
        verified_homepage_identity={"normalized_name": "vytexwindows", "registrable_dns_domain": "vytexwindows.com"},
        prefetched_pages={URL: {"text": TEXT, "final_url": URL}},
    ))
    assert len(requests) == 2
    rejected = feedbacks[0]["rejected_findings"][0]
    assert rejected.get("company_identity_keys_checked") == KEYS, rejected
    support = rejected["rejected_supporting_evidence"][0]
    assert support["supporting_evidence_slot"] == 1
    assert support["company_identity_keys_checked"] == KEYS
    assert support["reason"] == inv.REQUIRED_ATTRIBUTE_SUPPORT_REJECTION_REASON
    assert result["_completed_submit"] is (repair != "repeat_invalid")
    if repair == "named_positive":
        assert result["claims"]["required_attribute"]["status"] == "VERIFIED"
    elif repair == "repeat_invalid":
        assert result["failure_reason"] == "malformed_response"
