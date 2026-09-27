from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from qualification.scoring import intent_details, verification_helpers


def _inputs(
    *,
    company_name: str,
    paragraph: str,
    signal_quote: str,
    source_text: str,
    source_publication_date: str = "",
    company_dimension: str = "",
    company_quote: str = "",
):
    website = f"https://{company_name.casefold()}.example"
    source_url = f"{website}/evidence"
    company = SimpleNamespace(
        company_name=company_name,
        company_website=website,
        intent_details=paragraph,
        intent_signals=[SimpleNamespace(
            matched_icp_signal=0,
            description=signal_quote,
            url=source_url,
        )],
    )
    icp = SimpleNamespace(
        prompt="Find relevant software companies with verified activity.",
        product_service="A workflow software platform",
        intent_signals=["Verified company activity from a current source."],
    )
    signal_results = [{
        "after_decay": 54,
        "matched_icp_signal": 0,
        "evidence_urls": [source_url],
        "judge_verdict": {
            "decision": "verified",
            "client_ready": True,
            "authoritative_date": source_publication_date or None,
            "authoritative_date_basis": (
                "publication_date"
                if source_publication_date else "missing_or_conflicting"
            ),
            "verification_trace": {
                "intent_verdict": {
                    "signal_evaluations": [{
                        "signal_status": "supported",
                        "verification_mode": "source_grounded",
                        "same_entity_check": "pass",
                        "supporting_quotes": [signal_quote],
                        "evidence_urls_used": [source_url],
                    }],
                },
                "verified_source_context": [{
                    "url": source_url,
                    "text": source_text,
                    "source_publication_date": source_publication_date,
                }],
            },
        },
    }]
    dimensions = {}
    if company_dimension:
        dimensions[company_dimension] = {
            "decision": "match",
            "web_evidence": {
                "url": f"{website}/{company_dimension}",
                "quote": company_quote,
            },
        }
    return company, icp, signal_results, {"dimension_evidence": dimensions}


def _binding(document: dict, quote: str) -> dict:
    for source in document["admitted_evidence"]:
        if any(
            quote in value
            for value in intent_details._admitted_evidence_values(source)
        ):
            return {"source_index": source["source_index"], "quote": quote}
    raise AssertionError(f"quote was not admitted: {quote}")


async def _review(
    monkeypatch,
    inputs,
    *,
    unit_specs: list[tuple[bool, str, list[str]]],
    expected_decision: str,
):
    # Mock only the semantic verdict. The production request construction,
    # evidence binding, response validation, and fail-closed gate all run.
    async def judge(prompt, **kwargs):
        system = kwargs["system_prompt"]
        assert "latest-known funding-stage wording" in system
        assert "source, report, or coverage" in system
        assert "present-status wording" in system
        assert 'such as "is hiring"' in system
        document = json.loads(prompt)
        assert len(document["intent_details_units"]) == len(unit_specs)
        grounding = []
        for unit, (contains_fact, status, quotes) in zip(
            document["intent_details_units"], unit_specs,
        ):
            grounding.append({
                "unit_id": unit["unit_id"],
                "contains_factual_claim": contains_fact,
                "status": status,
                "evidence": [_binding(document, quote) for quote in quotes],
            })
        facts_supported = all(
            status == "VERIFIED"
            for contains_fact, status, _quotes in unit_specs
            if contains_fact
        )
        return json.dumps({
            "unit_grounding": grounding,
            "signal_coverage": [{
                "matched_icp_signal": 0,
                "covered": True,
            }],
            "facts_supported": facts_supported,
            "verified_signals_covered": True,
            "relevance_grounded": True,
            "connects_icp": True,
            "natural_paragraph": True,
        })

    monkeypatch.setattr(verification_helpers, "openrouter_chat", judge)
    receipt = await intent_details.review_intent_details(*inputs)
    assert receipt["decision"] == expected_decision, receipt
    assert receipt["checks"]["facts_supported"] is (
        expected_decision == "match"
    )


@pytest.mark.asyncio
async def test_undated_recent_coverage_in_exact_paragraph_is_unproven(
    monkeypatch,
):
    job_quote = (
        "Unibuddy is recruiting a Software Engineer II to work across its "
        "platform and directly develop Chat, Assistant, and Intelligence "
        "products used by prospective students, university staff, and "
        "ambassadors."
    )
    stage_quote = (
        "Unibuddy is a series B company based in London (United Kingdom), "
        "founded in 2017 by Diego Fanara."
    )
    paragraph = (
        f"{job_quote} The canonical posting describes full-stack platform "
        "responsibilities and the company's role in higher-education "
        "recruitment, while recent coverage identifies its latest disclosed "
        "institutional round as Series B. This active platform hiring may "
        "indicate capacity-building around student engagement and enrollment "
        "workflows relevant to the offered platform."
    )
    inputs = _inputs(
        company_name="Unibuddy",
        paragraph=paragraph,
        signal_quote=job_quote,
        source_text=(
            job_quote + " The role has full-stack platform responsibilities "
            "and supports higher-education recruitment."
        ),
        company_dimension="stage",
        company_quote=stage_quote,
    )

    await _review(
        monkeypatch,
        inputs,
        unit_specs=[
            (True, "VERIFIED", [job_quote]),
            (True, "UNPROVEN", [stage_quote]),
            (False, "VERIFIED", []),
        ],
        expected_decision="mismatch",
    )


@pytest.mark.asyncio
async def test_undated_recent_industry_coverage_is_unproven(monkeypatch):
    job_quote = "HarborSoft is actively hiring a Platform Engineer."
    industry_quote = "HarborSoft is a workflow software provider."
    inputs = _inputs(
        company_name="HarborSoft",
        paragraph=(
            f"{job_quote} Recent industry coverage describes HarborSoft as a "
            "workflow software provider."
        ),
        signal_quote=job_quote,
        source_text=job_quote,
        company_dimension="industry",
        company_quote=industry_quote,
    )

    await _review(
        monkeypatch,
        inputs,
        unit_specs=[
            (True, "VERIFIED", [job_quote]),
            (True, "UNPROVEN", [industry_quote]),
        ],
        expected_decision="mismatch",
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "second_sentence",
    [
        "HarborSoft is currently described as a Series B company.",
        "HarborSoft's latest known funding stage is Series B.",
    ],
)
async def test_current_or_latest_known_stage_remains_supported(
    monkeypatch, second_sentence,
):
    job_quote = "HarborSoft is actively hiring a Platform Engineer."
    stage_quote = "HarborSoft is a Series B company."
    inputs = _inputs(
        company_name="HarborSoft",
        paragraph=f"{job_quote} {second_sentence}",
        signal_quote=job_quote,
        source_text=job_quote,
        company_dimension="stage",
        company_quote=stage_quote,
    )

    await _review(
        monkeypatch,
        inputs,
        unit_specs=[
            (True, "VERIFIED", [job_quote]),
            (True, "VERIFIED", [stage_quote]),
        ],
        expected_decision="match",
    )


@pytest.mark.asyncio
async def test_current_job_body_still_supports_is_hiring(monkeypatch):
    job_quote = "HarborSoft is actively hiring a Platform Engineer."
    inputs = _inputs(
        company_name="HarborSoft",
        paragraph=job_quote,
        signal_quote=job_quote,
        source_text=job_quote,
    )

    await _review(
        monkeypatch,
        inputs,
        unit_specs=[(True, "VERIFIED", [job_quote])],
        expected_decision="match",
    )


@pytest.mark.asyncio
async def test_recent_report_is_supported_when_publication_date_is_admitted(
    monkeypatch,
):
    source_text = (
        "Austin, TX, September 1, 2026 - HarborSoft published its Workflow "
        "Market Outlook for universities."
    )
    inputs = _inputs(
        company_name="HarborSoft",
        paragraph=(
            "Recent coverage dated 2026-09-01 reports that HarborSoft "
            "published its Workflow Market Outlook for universities."
        ),
        signal_quote=source_text,
        source_text=source_text,
        source_publication_date="2026-09-01",
    )

    await _review(
        monkeypatch,
        inputs,
        unit_specs=[(True, "VERIFIED", [source_text, "2026-09-01"])],
        expected_decision="match",
    )
