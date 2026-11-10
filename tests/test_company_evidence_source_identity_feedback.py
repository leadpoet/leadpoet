"""Nov8/Nov10 citation shapes; mocked replies make no accuracy claim."""

import asyncio
import json

import pytest

from qualification.scoring import company_evidence_investigator as inv
from tests.test_company_evidence_judgment_reserve import _Clock, _finding, _response


URL = "https://ventureglobal.com/about-us/"
UNBOUND_TEXT = (
    "Venture Global develops LNG export projects and operates pipelines. "
    "Its facilities supply fuel to commercial customers."
)
UNBOUND_QUOTE = "operates pipelines. Its facilities supply fuel to commercial customers."


def _attribute_finding(*, status="VERIFIED", url=URL, quote=UNBOUND_QUOTE):
    finding = _finding(status=status, url=url, quote=quote)
    finding.update(target="required_attribute", observed_value=(
        "Owns or operates energy infrastructure supplying fuel to customers"
        if status == "VERIFIED" else None
    ))
    return finding


def _run_unbound_feedback(monkeypatch, *, repair_status="UNPROVEN"):
    calls = []
    feedbacks = []

    async def post(_session, _url, *, headers, payload):
        calls.append(payload)
        assert len(calls) <= 2
        if len(calls) == 2:
            feedback = next(json.loads(message["content"])
                for message in reversed(payload["messages"])
                if message.get("role") == "tool")
            feedbacks.append(feedback)
            finding = (_attribute_finding(status="UNPROVEN", url="", quote="")
                       if repair_status == "UNPROVEN" else _attribute_finding())
        else:
            finding = _attribute_finding()
        return 200, _response("submit_findings", {"findings": [finding]}, call=len(calls))

    async def forbidden(*_args, **_kwargs):
        pytest.fail("The correction must use existing sources and calls")

    monkeypatch.setenv("OPENROUTER_API_KEY", "test")
    monkeypatch.setenv("EXA_API_KEY", "test")
    monkeypatch.setattr(inv, "_post_json", post)
    monkeypatch.setattr(inv, "_search_web", forbidden)
    monkeypatch.setattr(inv, "_fetch_page", forbidden)
    monkeypatch.setattr(inv, "time", _Clock())
    monkeypatch.setattr(inv, "MAX_REASONING_TURNS", 1)
    result = asyncio.run(inv.investigate_company_evidence(
        company_locator={"name": "Venture Global LNG", "website": "http://venturegloballng.com/"},
        targets=("required_attribute",),
        requested_attribute="Owns or operates energy infrastructure supplying fuel to customers",
        prior_observations={"observed_company_name": "Venture Global, Inc.",
                            "observed_company_website": "https://ventureglobal.com/",
                            "submitted_source_urls": [URL]},
        prefetched_pages={URL: {"final_url": URL, "text": UNBOUND_TEXT}},
    ))
    assert len(calls) == 2
    rejected = feedbacks[0]["rejected_findings"][0]
    assert rejected["target"] == "required_attribute"
    assert rejected["reason"] == "source quote did not identify the investigated company"
    assert rejected["source_url"] == URL
    assert rejected["source_lacks_admitted_company_name"] is True
    assert "different already fetched source" in feedbacks[0]["instruction"]
    assert "explicit complete UNPROVEN" in feedbacks[0]["instruction"]
    return result


def test_nov10_unbound_fetched_source_feedback_requires_distinct_proof_or_negative(monkeypatch):
    result = _run_unbound_feedback(monkeypatch)
    assert result["_completed_submit"] is True
    assert result["claims"]["required_attribute"]["status"] == "UNPROVEN"


def test_nov10_repeated_unsupported_positive_stays_incomplete(monkeypatch):
    result = _run_unbound_feedback(monkeypatch, repair_status="VERIFIED")
    assert result["_completed_submit"] is False
    assert result["failure_reason"] == "malformed_response"


def test_nov8_named_candidate_and_missing_source_controls():
    names = {"kratosdefenseandsecuritysolutions"}
    named = "Kratos Defense and Security Solutions makes military hardware."
    assert not inv._source_lacks_admitted_company_name(named, names)
    assert inv._source_lacks_admitted_company_name(UNBOUND_TEXT, {"venturegloballng"})
    assert not inv._source_lacks_admitted_company_name("", {"venturegloballng"})
    assert not inv._source_lacks_admitted_company_name(UNBOUND_TEXT, set())


def test_missing_requested_target_is_not_a_completed_negative():
    # The two Nov10 ICP1 final packets omitted required_attribute entirely.
    partial = {"findings": [_finding()]}
    assert inv._validated_findings(
        partial, targets=("industry", "required_attribute"),
        fetched_pages={"https://acme.example/platform":
                       "Acme supplies enrollment software to universities worldwide."},
        fetched_final_urls={}, first_party_domains={"acme.example"},
        identity_names={"acme"},
    ) is None
