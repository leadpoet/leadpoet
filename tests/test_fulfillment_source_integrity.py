from __future__ import annotations

import ast
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from gateway.qualification.models import ICPPrompt, IntentSignal
from qualification.scoring.intent_verification_three_stage import (
    verify_three_stage,
)
from qualification.scoring.lead_scorer import _score_single_intent_signal
from qualification.scoring.verification_helpers import (
    check_source_url_mismatch,
)

BEDROCK_URL = (
    "https://bedrocki.com/"
    "esri-chile-accelerates-digital-transformation-with-key-hires"
)


def test_arbitrary_article_cannot_claim_trusted_fulfillment_source_types() -> None:
    company = "https://www.esri.cl/"

    assert check_source_url_mismatch(
        "job_board",
        BEDROCK_URL,
        company,
        reject_unknown_third_party=True,
    )
    assert check_source_url_mismatch(
        "news",
        BEDROCK_URL,
        company,
        reject_unknown_third_party=True,
    )
    assert check_source_url_mismatch(
        "company_website",
        BEDROCK_URL,
        company,
        reject_unknown_third_party=True,
    )
    assert check_source_url_mismatch(
        "other",
        BEDROCK_URL,
        company,
        reject_unknown_third_party=True,
    )


@pytest.mark.parametrize(
    ("source", "url", "company"),
    [
        (
            "job_board",
            "https://jobs.lever.co/esri/1234",
            "https://www.esri.com/",
        ),
        (
            "job_board",
            "https://careers.esri.com/open-positions/engineer",
            "https://www.esri.com/",
        ),
        (
            "job_board",
            "https://www.esri.com/careers/open-positions/engineer",
            "https://www.esri.com/",
        ),
        (
            "company_website",
            "https://blog.esri.com/digital-transformation",
            "https://www.esri.com/",
        ),
        (
            "news",
            "https://www.reuters.com/technology/esri-expands-2026-07-01/",
            "https://www.esri.com/",
        ),
        (
            "other",
            "https://procurement.gov/contracts/esri-award",
            "https://www.esri.com/",
        ),
    ],
)
def test_identity_bound_or_recognized_sources_remain_valid(
    source: str,
    url: str,
    company: str,
) -> None:
    assert (
        check_source_url_mismatch(
            source,
            url,
            company,
            reject_unknown_third_party=True,
        )
        is None
    )


def test_first_party_job_category_defers_content_proof_to_exact_page_gate() -> None:
    assert (
        check_source_url_mismatch(
            "job_board",
            "https://www.esri.com/blog/key-hires",
            "https://www.esri.com/",
            reject_unknown_third_party=True,
        )
        is None
    )


def test_public_sector_exception_cannot_be_spoofed_by_a_subdomain_label() -> None:
    assert check_source_url_mismatch(
        "other",
        "https://registry.gov.attacker.com/esri-award",
        "https://www.esri.com/",
        reject_unknown_third_party=True,
    )


@pytest.mark.asyncio
async def test_active_fulfillment_gate_rejects_before_any_provider_call() -> None:
    signal = IntentSignal(
        source="job_board",
        description=(
            "Esri Chile is actively hiring for API integration and workflow "
            "automation roles supporting digital transformation."
        ),
        url=BEDROCK_URL,
        date="2026-06-21",
        snippet=(
            "Esri Chile is actively hiring for API integrations, workflow "
            "automation, and digital transformation roles."
        ),
        matched_icp_signal=0,
    )
    icp = ICPPrompt(
        icp_id="icp-source-integrity",
        industry="Software",
        sub_industry="Geospatial Software",
        employee_count="51-200",
        company_stage="growth",
        geography="Chile",
        product_service="workflow automation",
        intent_signals=["actively hiring integration engineers"],
    )
    verdicts: list[dict] = []

    with patch(
        "qualification.scoring.intent_verification_three_stage.verify_three_stage",
        new=AsyncMock(),
    ) as verifier:
        score, confidence, date_status, found_date, matched_idx = (
            await _score_single_intent_signal(
                signal,
                icp,
                None,
                "Esri Chile",
                company_website="https://www.esri.cl/",
                company_linkedin="https://www.linkedin.com/company/esri-chile/",
                enforce_source_integrity=True,
                stage1_soft_reject=True,
                verdict_out=verdicts,
            )
        )

    verifier.assert_not_awaited()
    assert (score, confidence, date_status, found_date, matched_idx) == (
        0.0,
        0,
        "source_mismatch",
        None,
        -1,
    )
    assert len(verdicts) == 1
    assert verdicts[0]["decision"] == "rejected_pregate"
    assert verdicts[0]["rejection_reason"] == "source_url_mismatch"
    assert "not a recognized job_board domain" in (
        verdicts[0]["source_integrity_error"]
    )


@pytest.mark.asyncio
async def test_declared_first_party_job_board_requires_job_body() -> None:
    url = "https://careers.acme.com/team"
    stage_one = {
        "answer": {
            "signal_evaluations": [
                {
                    "signal_status": "supported",
                    "confidence": "high",
                    "same_entity_check": "pass",
                    "verification_mode": "source_grounded",
                    "evidence_urls_used": [url],
                    "claim_matches_miner_date": "supported",
                }
            ],
        },
        "model": "test-model",
        "usage": {},
    }
    fetch = AsyncMock(
        return_value={
            "results": [
                {
                    "url": url,
                    "title": "Meet the Acme team",
                    "text": (
                        "Acme celebrates its people and describes employee benefits "
                        "and workplace culture."
                    ),
                }
            ],
            "statuses": [{"source": "scrapingdog", "stage": "ok"}],
        }
    )
    call = AsyncMock(return_value=stage_one)

    with (
        patch(
            "qualification.scoring.intent_verification_three_stage._call_openrouter",
            call,
        ),
        patch(
            "qualification.scoring.intent_verification_three_stage._fetch_sd_then_exa",
            fetch,
        ),
    ):
        result = await verify_three_stage(
            object(),
            company_name="Acme",
            company_linkedin="https://www.linkedin.com/company/acme",
            company_website="https://acme.com",
            source_url=url,
            miner_claim="Acme is actively hiring data engineers",
            target_signal_text="The company is actively hiring data engineers",
            miner_signal_date="2026-07-01",
            declared_source="job_board",
            stage1_soft_reject=True,
        )

    fetch.assert_awaited_once_with([url])
    assert call.await_count == 1
    assert result["client_ready"] is False
    assert result["rejection_reason"] == "job_body_not_in_fetched_content"


def test_gateway_fulfillment_wires_fail_closed_source_policy() -> None:
    root = Path(__file__).resolve().parents[1]
    tree = ast.parse((root / "gateway/fulfillment/scoring.py").read_text())
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and (
            isinstance(node.func, ast.Name)
            and node.func.id == "_score_single_intent_signal"
        )
    ]
    assert len(calls) == 1
    keywords = {keyword.arg: keyword.value for keyword in calls[0].keywords}
    for required in ("enforce_source_integrity", "stage1_soft_reject"):
        assert isinstance(keywords[required], ast.Constant)
        assert keywords[required].value is True
