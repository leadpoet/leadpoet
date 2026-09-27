"""A positive activity review must retain its independently fetched proof."""

from __future__ import annotations

import asyncio

import pytest

from qualification.scoring import lead_scorer
from qualification.scoring.company_fit_decision import (
    COMPANY_FIT_MATCH,
    COMPANY_FIT_UNAVAILABLE,
    company_fit_match,
)
from tests.test_positive_semantic_fit_review import (
    _company, _finding, _icp, _verdict,
)


@pytest.mark.parametrize(
    "control",
    [
        "verified",
        "already_grounded",
        "unproven",
        "customer",
        "missing_page",
        "wrong_domain",
        "absent_quote",
        "unproven_stage",
    ],
)
def test_positive_review_reuses_only_complete_grounded_attribute_evidence(
    monkeypatch, control,
):
    company, icp = _company(), _icp()
    initial = _verdict()
    quote = (
        "PeerConnect supplies a student enrollment platform to universities."
    )
    source_url = (
        "https://other.example/platform" if control == "wrong_domain"
        else "https://peer.example/platform"
    )
    calls = {"provider": 0, "investigator": 0, "fetch": 0}

    async def provider(**_kwargs):
        calls["provider"] += 1
        assert calls["provider"] <= 2
        # The original attribute citation is ungrounded. A schema retry may
        # not fabricate a replacement for the independently fetched finding.
        return dict(initial), ""

    async def fetch(_session, url):
        calls["fetch"] += 1
        assert url == initial["required_attribute_evidence_url"]
        text = (
            initial["required_attribute_evidence_quote"]
            if control == "already_grounded" else "Contact our team."
        )
        return 200, url, f"<html><body>{text}</body></html>"

    async def keep_observation(verdict, *_args, **_kwargs):
        return verdict

    async def investigate(**kwargs):
        calls["investigator"] += 1
        assert kwargs["targets"] == ("stage", "industry")
        assert kwargs["positive_semantic_review"] is True
        assert kwargs["requested_attribute"] == (
            icp.required_attribute
        )
        industry = _finding(
            "industry",
            observed_value="Student enrollment platform",
            observed_industry="Education",
            observed_subindustry="Higher education services",
            activity_role="supplier_operator",
            evidence_url=source_url,
            evidence_quote=quote,
        )
        if control == "unproven":
            industry.update(
                status="UNPROVEN", activity_role="unresolved",
                evidence_url="", evidence_quote="",
            )
        elif control == "customer":
            industry.update(
                status="CONTRADICTED", activity_role="customer_user",
                evidence_quote="PeerConnect uses another vendor's platform.",
            )
        stage = _finding("stage")
        if control == "unproven_stage":
            stage.update(status="UNPROVEN", evidence_url="", evidence_quote="")
        pages = {} if control == "missing_page" else {
            source_url: {
                "final_url": source_url,
                "text": (
                    "Contact our team." if control == "absent_quote"
                    else industry["evidence_quote"]
                ),
            },
        }
        return {
            "claims": {"stage": stage, "industry": industry},
            "_validated_stage_finding": (
                stage if control != "unproven_stage" else None
            ),
            lead_scorer.PRIVATE_FETCHED_PAGES_KEY: pages,
            "failure_reason": "",
            "usage": {"reasoning_turns": 1, "search_calls": 1, "fetch_calls": 1},
        }

    homepage = company_fit_match(
        "homepage identity verified",
        details={
            "identity": {
                "decision": COMPANY_FIT_MATCH,
                "evidence_source": "company_homepage",
                "observed_name": "peerconnect",
                "observed_domain": "peer.example",
                "observed_linkedin_slug": "peerconnect",
            },
            "verified_homepage_transport_domain": "peer.example",
        },
    )
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(lead_scorer, "_request_company_reverify_json", provider)
    monkeypatch.setattr(lead_scorer, "_fetch_bounded_html", fetch)
    monkeypatch.setattr(
        lead_scorer, "_refresh_linkedin_employee_size_observation", keep_observation
    )
    monkeypatch.setattr(lead_scorer, "investigate_company_evidence", investigate)

    result = asyncio.run(lead_scorer._llm_reverify_company(
        company, icp, require_company_fit_dimensions=True,
        verified_homepage_identity=homepage, company_quality=True,
        evidence_investigator=True,
    ))

    assert calls["investigator"] == 1
    assert calls["fetch"] == 1
    if control in {"verified", "already_grounded"}:
        assert result.decision == COMPANY_FIT_MATCH
        assert result.details["required_attribute_decision"] == COMPANY_FIT_MATCH
        assert result.details["required_attribute_grounding"]["status"] == "grounded"
        assert calls["provider"] == 1
        evidence = result.details["dimension_evidence"]["required_attribute"]
        assert evidence["quote"] == (
            initial["required_attribute_evidence_quote"]
            if control == "already_grounded" else quote
        )
        if control == "verified":
            assert result.details["required_attribute_grounding"]["cache_hit"] is True
    else:
        assert result.decision != COMPANY_FIT_MATCH
        if control != "unproven_stage":
            assert result.details["required_attribute_decision"] != COMPANY_FIT_MATCH
        else:
            assert result.details["dimension_decisions"]["stage"] == (
                COMPANY_FIT_UNAVAILABLE
            )
