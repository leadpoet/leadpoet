from __future__ import annotations

import asyncio
import copy
import json
from pathlib import Path

import pytest

from qualification.scoring import company_evidence_investigator as investigator
from tests.product_coverage_fixtures import coverage
from tests.test_company_evidence_judgment_reserve import (
    _Clock, _finding as _generic_finding, _response, _run_industry,
)


CASE = json.loads((
    Path(__file__).parent / "fixtures/company_evidence/"
    "element_missing_industry_observation.json"
).read_text())
FINDING = CASE["finding"]
URL = CASE["source"]["url"]
QUOTE = FINDING["evidence_quote"]
OBSERVATION = "DNA sequencing and multi-omics technology for research markets"
PRODUCT = "DNA sequencing platform for research markets"


def _validate(finding, *, source_text=None):
    url = finding["evidence_url"]
    return investigator._validated_findings(
        {"findings": [finding]},
        targets=(finding["target"],),
        fetched_pages={
            url: CASE["source"]["text"] if source_text is None else source_text,
        },
        fetched_final_urls={url: url},
        first_party_domains={"elementbiosciences.com"},
        identity_names={"elementbiosciences", "elementbiosciencesinc"},
        identity_anchor={
            "submitted_name": CASE["company_locator"]["name"],
            "submitted_domain": "elementbiosciences.com",
            "submitted_linkedin_slug": "",
            "observed_name": CASE["prior_observations"]["observed_company_name"],
            "observed_domain": "elementbiosciences.com",
            "observed_linkedin_slug": "elembio",
        },
    )[finding["target"]]


def test_actual_missing_observation_is_not_reported_as_rejected_activity_proof():
    assert FINDING["observed_industry"] == ""
    assert FINDING["activity_role"] == "supplier_operator"
    assert QUOTE in CASE["source"]["text"]
    finding = _validate(FINDING)
    assert finding["status"] == "UNPROVEN"
    assert finding["evidence_url"] == finding["evidence_quote"] == ""
    assert finding["reason"] == investigator.MISSING_INDUSTRY_OBSERVATION_REASON
    assert investigator._submitted_finding_rejection(FINDING, finding) == (
        investigator.MISSING_INDUSTRY_OBSERVATION_REASON
    )

    # Correct only the missing field, using an activity stated in the same quote.
    corrected = {**FINDING, "observed_industry": OBSERVATION}
    accepted = _validate(corrected)
    assert accepted["status"] == "VERIFIED"
    assert accepted["evidence_quote"] == QUOTE
    assert accepted["observed_industry"] == OBSERVATION


def test_contradicted_industry_requires_metadata_and_correct_activity_role():
    quote = (
        "Element Biosciences is a customer of another supplier's DNA sequencing "
        "platform and uses that platform for its research."
    )
    finding = {
        **FINDING,
        "status": "CONTRADICTED",
        "activity_role": "customer_user",
        "evidence_quote": quote,
    }
    missing = _validate(finding, source_text=quote)
    assert missing["status"] == "UNPROVEN"
    assert missing["reason"] == investigator.MISSING_INDUSTRY_OBSERVATION_REASON
    assert investigator._submitted_finding_rejection(finding, missing) == (
        investigator.MISSING_INDUSTRY_OBSERVATION_REASON
    )
    corrected = {**finding, "observed_industry": "DNA sequencing platform"}
    accepted = _validate(corrected, source_text=quote)
    assert accepted["status"] == "CONTRADICTED"
    assert accepted["evidence_quote"] == quote
    wrong_role = _validate(
        {**corrected, "activity_role": "supplier_operator"}, source_text=quote,
    )
    assert wrong_role["status"] == "UNPROVEN"
    assert wrong_role["reason"] == investigator.INDUSTRY_RELATIONSHIP_UNPROVEN_REASON


@pytest.mark.parametrize("changes,source_text,reason", [
    ({"activity_role": "customer_user"}, None,
     investigator.INDUSTRY_RELATIONSHIP_UNPROVEN_REASON),
    ({"activity_role": "unresolved"}, None,
     investigator.INDUSTRY_RELATIONSHIP_UNPROVEN_REASON),
    ({"evidence_url": "https://other.example/product"}, None,
     "industry evidence must use independently bound "
     "first-party requested and fetched final URLs"),
    ({"evidence_quote": "Other Company supplies sequencing products."},
     "Other Company supplies sequencing products.",
     "source quote did not identify the investigated company"),
    ({"evidence_quote": QUOTE + " Invented proof."}, None,
     "submitted quote was not present in fetched source"),
])
def test_missing_metadata_does_not_bypass_evidence_or_role_checks(
    changes, source_text, reason,
):
    rejected = _validate({**FINDING, **changes}, source_text=source_text)
    assert rejected["status"] == "UNPROVEN"
    assert rejected["reason"] == reason
    assert rejected["reason"] != investigator.MISSING_INDUSTRY_OBSERVATION_REASON


@pytest.mark.parametrize("repair,attribute_quote_rejected", [
    (True, False), (False, False), (True, True),
])
def test_metadata_feedback_reuses_quote_and_preserves_other_finding(
    monkeypatch, repair, attribute_quote_rejected,
):
    coverage(PRODUCT, URL, QUOTE)
    clock = _Clock()
    requests = []
    retained = {
        **FINDING,
        "target": "required_attribute",
        "reason": "The named company develops life-sciences platform technologies.",
    }

    async def post(_session, _url, *, headers, payload):
        requests.append(copy.deepcopy(payload))
        if len(requests) == 1:
            # Enter the existing submit-only correction, with no new retry path.
            clock.now = (
                investigator.ADMISSION_DEADLINE_SECONDS
                - investigator.JUDGMENT_ADMISSION_RESERVE_SECONDS
            )
            return 200, _response("submit_findings", {
                "findings": [{**FINDING, "product_coverage": coverage(PRODUCT, URL, QUOTE)}, {
                    **retained,
                    **({"evidence_quote": "An invented attribute quote."}
                       if attribute_quote_rejected else {}),
                }],
            })
        feedback = next(
            json.loads(message["content"])
            for message in reversed(payload["messages"])
            if message.get("role") == "tool"
            and "rejected_findings" in json.loads(message["content"])
        )
        assert feedback["rejected_findings"][0]["reason"] == (
            investigator.MISSING_INDUSTRY_OBSERVATION_REASON
        )
        if attribute_quote_rejected:
            assert feedback["rejected_findings"][1]["target"] == "required_attribute"
            assert feedback["rejected_findings"][1]["reason"] == (
                "submitted quote was not present in fetched source"
            )
            assert "Reassess the exact required attribute" in feedback["instruction"]
        instruction = feedback["instruction"]
        assert "exact quote, company binding, and activity role passed" in instruction
        assert "reuse the same fetched URL and exact quote" in instruction
        assert "Never repeat a rejected quote" not in instruction
        assert "relationship to the activity" not in json.dumps(feedback)
        submit = next(
            tool["function"] for tool in payload["tools"]
            if tool["function"]["name"] == "submit_findings"
        )
        properties = submit["parameters"]["properties"]["findings"]["items"]["properties"]
        assert properties["target"]["enum"] == (
            ["industry", "required_attribute"]
            if attribute_quote_rejected else ["industry"]
        )
        assert "nonempty source-grounded" in properties["observed_industry"]["description"]
        corrected = {**FINDING, "observed_industry": OBSERVATION} if repair else FINDING
        corrected = {**corrected, "product_coverage": coverage(PRODUCT, URL, QUOTE)}
        return 200, _response("submit_findings", {
            "findings": [corrected, retained] if attribute_quote_rejected else [corrected],
        }, call=len(requests))

    async def no_research(*_args, **_kwargs):
        raise AssertionError("A missing activity description needs no new research")

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")
    monkeypatch.setenv("EXA_API_KEY", "test-exa-key")
    monkeypatch.setattr(investigator, "time", clock)
    monkeypatch.setattr(investigator, "_post_json", post)
    monkeypatch.setattr(investigator, "_search_web", no_research)
    monkeypatch.setattr(investigator, "_fetch_page", no_research)
    result = asyncio.run(investigator.investigate_company_evidence(
        company_locator=CASE["company_locator"],
        targets=("industry", "required_attribute"),
        requested_industry="Biotechnology",
        requested_subindustry="Therapeutics and platform biotech",
        requested_product_service=PRODUCT,
        requested_attribute=(
            "Develops regulated biological products, therapeutic candidates, "
            "or platform technologies for the life sciences market."
        ),
        positive_semantic_review=True,
        prior_observations={
            **CASE["prior_observations"], "submitted_source_urls": [URL],
        },
        verified_homepage_identity=CASE["verified_homepage_identity"],
        prefetched_pages={URL: {
            "final_url": CASE["source"]["final_url"],
            "text": CASE["source"]["text"],
        }},
    ))

    assert result["claims"]["industry"]["status"] == (
        "VERIFIED" if repair else "UNPROVEN"
    )
    assert result["claims"]["required_attribute"]["status"] == "VERIFIED"
    assert result["claims"]["required_attribute"]["evidence_quote"] == QUOTE
    assert result["claims"]["required_attribute"]["reason"] == retained["reason"]
    assert result["claims"]["required_attribute"] == _validate(retained)
    assert result["usage"]["fetch_calls"] == result["usage"]["search_calls"] == 0
    if repair:
        assert len(requests) == 2
        assert result["claims"]["industry"]["evidence_quote"] == QUOTE
        assert result["claims"]["industry"]["observed_industry"] == OBSERVATION
    else:
        assert len(requests) <= investigator.MAX_REASONING_TURNS + 1


def test_other_quote_rejections_keep_existing_feedback(monkeypatch):
    clock = _Clock()
    requests = []

    async def post(_session, _url, *, headers, payload):
        requests.append(payload)
        if len(requests) == 1:
            clock.now = (
                investigator.ADMISSION_DEADLINE_SECONDS
                - investigator.JUDGMENT_ADMISSION_RESERVE_SECONDS
            )
            return 200, _response("submit_findings", {
                "findings": [_generic_finding(quote="An invented industry quote.")],
            })
        feedback = next(
            json.loads(message["content"])
            for message in reversed(payload["messages"])
            if message.get("role") == "tool"
            and "rejected_findings" in json.loads(message["content"])
        )
        assert feedback["rejected_findings"][0]["reason"] == (
            "submitted quote was not present in fetched source"
        )
        assert "Never repeat a rejected quote." in feedback["instruction"]
        assert "Do not repeat an unchanged rejected finding" not in feedback["instruction"]
        assert "You may reuse the same fetched URL" not in feedback["instruction"]
        return 200, _response("submit_findings", {
            "findings": [_generic_finding()],
        }, call=2)

    result = _run_industry(monkeypatch, post, clock)
    assert len(requests) == 2
    assert result["claims"]["industry"]["status"] == "VERIFIED"
