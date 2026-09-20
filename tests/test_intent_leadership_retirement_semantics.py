"""Leadership retirement semantics stay source-grounded and general."""

from __future__ import annotations

import pytest

from qualification.scoring import intent_verification_three_stage as intent


SOURCE_URL = (
    "https://www.prnewswire.com/news-releases/"
    "ptc-therapeutics-appoints-biotech-banking-pioneer-"
    "jessica-chutter-to-board-of-directors-302724358.html"
)
CLAIM = "Jessica Chutter recently retired after her Morgan Stanley career."
TARGET = (
    "Announced a leadership change in the last 12 months, or disclosed an "
    "acquisition or integration update with public evidence."
)
REAL_PARAGRAPH = (
    "PTC Therapeutics announced the appointment of Jessica Chutter to its "
    "Board of Directors. Ms. Chutter recently retired as Managing Director "
    "and Chair of Biotechnology Investment Banking of Morgan Stanley after "
    "a distinguished career spanning over 40 years."
)
REAL_QUOTE = (
    "Ms. Chutter recently retired as Managing Director and Chair of "
    "Biotechnology Investment Banking of Morgan Stanley after a distinguished "
    "career spanning over 40 years."
)


def _identity() -> dict:
    return {
        "decision": "match",
        "reason_code": "verifier_accepted",
        "submitted_name": "morganstanley",
        "submitted_domain": "morganstanley.com",
        "submitted_linkedin_slug": "morgan-stanley",
        "observed_name": "morganstanley",
        "observed_domain": "morganstanley.com",
        "observed_linkedin_slug": "morgan-stanley",
        "evidence_source": "company_web_reverification",
        "verified_legal_name_aliases": [],
    }


def _verdict(claim: str, status: str, quote: str) -> dict:
    supported = status == "supported"
    return {
        "answer": {
            "overall_verdict": "qualified" if supported else "disqualified",
            "overall_confidence": "high",
            "signal_evaluations": [{
                "claim": claim,
                "signal_status": status,
                "confidence": "high",
                "same_entity_check": "pass",
                "verification_mode": "source_grounded",
                "source_accessibility": "accessible",
                "evidence_urls_used": [SOURCE_URL],
                "supporting_quotes": [quote] if quote else [],
                "contradicting_quotes": [],
                "unsupported_parts": [] if supported else [
                    "The source does not prove the required recent leadership event."
                ],
                "risk_notes": ["source_publication_date:2026-03-25"],
                "claim_matches_miner_date": "no_date_in_content",
            }],
        },
        "model": "test-model",
        "usage": {},
    }


async def _run(
    monkeypatch,
    *,
    source_text: str,
    claim: str = CLAIM,
    target: str = TARGET,
    stage3_status: str,
    quote: str = "",
):
    prompts = []

    async def fetch(_urls):
        return {
            "results": [{
                "url": SOURCE_URL,
                "title": "PTC appoints Jessica Chutter to board",
                "text": source_text,
                "source_publication_date": "2026-03-25",
            }],
            "statuses": [{
                "url": SOURCE_URL,
                "source": "scrapingdog",
                "stage": "sd:baseline",
            }],
        }

    async def call_openrouter(_client, _model, prompt):
        prompts.append(prompt)
        if len(prompts) == 1:
            return _verdict(claim, "unable_to_verify", "")
        return _verdict(claim, stage3_status, quote)

    monkeypatch.setattr(intent, "_fetch_sd_then_exa", fetch)
    monkeypatch.setattr(intent, "_call_openrouter", call_openrouter)
    result = await intent.verify_three_stage(
        None,
        company_name="Morgan Stanley",
        company_linkedin="https://www.linkedin.com/company/morgan-stanley",
        company_website="https://www.morganstanley.com",
        source_url=SOURCE_URL,
        miner_claim=claim,
        target_signal_text=target,
        miner_signal_date="2026-03-01",
        evidence_type="LEADERSHIP_CHANGE",
        declared_source="news",
        stage1_soft_reject=True,
        integrity_policy=True,
        company_quality=True,
        verified_company_identity=_identity(),
    )
    return result, prompts


@pytest.mark.asyncio
async def test_senior_division_chair_retirement_reaches_supported_stage3(
    monkeypatch,
):
    result, prompts = await _run(
        monkeypatch,
        source_text=REAL_PARAGRAPH,
        stage3_status="supported",
        quote=REAL_QUOTE,
    )

    assert len(prompts) == 2
    assert REAL_PARAGRAPH in prompts[1]
    assert "retirement of a senior division or function leader" in prompts[1]
    assert "`Announced` does not by itself require" in prompts[1]
    assert result["client_ready"] is True
    assert result["decision"] == "approve"
    assert result["stage3"]["status"] == "supported"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("source_text", "claim", "target"),
    [
        (
            "PTC appointed Jane Doe. She worked as an analyst at Morgan Stanley.",
            "Morgan Stanley had a leadership change involving Jane Doe.",
            TARGET,
        ),
        (
            "John Doe retired as a Morgan Stanley managing director in 2010.",
            "John Doe recently retired from Morgan Stanley.",
            TARGET,
        ),
        (
            REAL_PARAGRAPH,
            CLAIM,
            "Morgan Stanley announced on its own website a leadership change "
            "in the last 12 months.",
        ),
    ],
)
async def test_name_biography_and_explicit_first_party_cases_do_not_autoapprove(
    monkeypatch,
    source_text,
    claim,
    target,
):
    result, prompts = await _run(
        monkeypatch,
        source_text=source_text,
        claim=claim,
        target=target,
        stage3_status="contradicted",
    )

    assert len(prompts) == 2
    assert result["client_ready"] is False
    assert result["decision"] == "reject"
    assert result["rejection_reason"] == "stage3_contradicted"
