from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from qualification.scoring import intent_details, verification_helpers
from qualification.scoring.evaluation_clock import use_evaluation_date


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


def _response(
    document: dict,
    unit_specs: list[tuple[bool, str, list[str]]],
) -> dict:
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
    return {
        "unit_grounding": grounding,
        "signal_coverage": [{"matched_icp_signal": 0, "covered": True}],
        "facts_supported": all(
            status == "VERIFIED"
            for contains_fact, status, _quotes in unit_specs
            if contains_fact
        ),
        "verified_signals_covered": True,
        "relevance_grounded": True,
        "connects_icp": True,
        "natural_paragraph": True,
    }


async def _review(
    monkeypatch,
    inputs,
    *,
    unit_specs: list[tuple[bool, str, list[str]]],
    expected_decision: str,
):
    # Mock only the semantic verdict. The production request construction,
    # evidence binding, response validation, and fail-closed gate all run.
    calls = 0

    async def judge(prompt, **kwargs):
        nonlocal calls
        calls += 1
        system = kwargs["system_prompt"]
        assert "latest-known funding-stage wording" in system
        assert "recent coverage, a recently published" in system
        assert "For unit_grounding, treat a qualifier" in system
        assert "returned evidence includes a binding" in system
        assert "undated provider description" in system
        assert "latest-known-state claim directly supported" in system
        assert "present-status wording" in system
        assert 'such as "is hiring"' in system
        document = json.loads(prompt)
        assert len(document["intent_details_units"]) == len(unit_specs)
        return json.dumps(_response(document, unit_specs))

    monkeypatch.setattr(verification_helpers, "openrouter_chat", judge)
    receipt = await intent_details.review_intent_details(*inputs)
    assert calls == 1
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


@pytest.mark.asyncio
async def test_undated_relative_coverage_uses_one_bounded_semantic_recheck(
    monkeypatch,
):
    job_quote = "HarborSoft is actively hiring a Platform Engineer."
    stage_quote = "HarborSoft is a Series B company."
    inputs = _inputs(
        company_name="HarborSoft",
        paragraph=(
            f"{job_quote} Recent coverage identifies HarborSoft's latest "
            "known institutional stage as Series B."
        ),
        signal_quote=job_quote,
        source_text=job_quote,
        company_dimension="stage",
        company_quote=stage_quote,
    )
    calls = []

    async def judge(prompt, **kwargs):
        calls.append((prompt, kwargs))
        payload = json.loads(prompt)
        document = payload.get("review_document", payload)
        if len(calls) == 1:
            return json.dumps(_response(document, [
                (True, "VERIFIED", [job_quote]),
                (True, "VERIFIED", [stage_quote]),
            ]))
        control = payload["bounded_unit_repair_control"]
        assert control["units"] == [{
            "unit_id": 1,
            "contains_factual_claim": True,
            "status": "VERIFIED",
            "citation_errors": [],
            "semantic_recheck_allowed": True,
            "semantic_recheck_reason": "relative_time_grounding_review",
        }]
        assert "routing control, not evidence or a conclusion" in kwargs[
            "system_prompt"
        ]
        return json.dumps({"repairs": [{
            "unit_id": 1, "status": "UNPROVEN", "evidence": [],
        }]})

    monkeypatch.setattr(verification_helpers, "openrouter_chat", judge)
    receipt = await intent_details.review_intent_details(*inputs)

    assert len(calls) == 2
    assert receipt["decision"] == "mismatch"
    assert receipt["checks"]["facts_supported"] is False


@pytest.mark.asyncio
async def test_body_date_can_keep_relative_publication_claim_verified_on_recheck(
    monkeypatch,
):
    dated_excerpt = (
        "September 1, 2026. HarborSoft published its Workflow Market Outlook."
    )
    inputs = _inputs(
        company_name="HarborSoft",
        paragraph=(
            "HarborSoft's Workflow Market Outlook was recently published on "
            "September 1, 2026."
        ),
        signal_quote=dated_excerpt,
        source_text=dated_excerpt,
    )
    calls = []

    async def judge(prompt, **kwargs):
        calls.append((prompt, kwargs))
        payload = json.loads(prompt)
        document = payload.get("review_document", payload)
        if len(calls) == 1:
            return json.dumps(_response(
                document, [(True, "VERIFIED", [dated_excerpt])],
            ))
        assert "exact date-bearing source" in kwargs["system_prompt"]
        assert "excerpt" in kwargs["system_prompt"]
        return json.dumps({"repairs": [{
            "unit_id": 0,
            "status": "VERIFIED",
            "evidence": [_binding(document, dated_excerpt)],
        }]})

    monkeypatch.setattr(verification_helpers, "openrouter_chat", judge)
    receipt = await intent_details.review_intent_details(*inputs)

    assert len(calls) == 2
    assert receipt["decision"] == "match"


@pytest.mark.asyncio
async def test_typed_date_binding_avoids_relative_time_recheck(monkeypatch):
    dated_excerpt = "HarborSoft announced a funding round on 2026-09-01."
    inputs = _inputs(
        company_name="HarborSoft",
        paragraph="HarborSoft recently announced a funding round.",
        signal_quote=dated_excerpt,
        source_text=dated_excerpt,
        source_publication_date="2026-09-01",
    )
    calls = 0

    async def judge(prompt, **_kwargs):
        nonlocal calls
        calls += 1
        document = json.loads(prompt)
        return json.dumps(_response(
            document,
            [(True, "VERIFIED", [dated_excerpt, "2026-09-01"])],
        ))

    monkeypatch.setattr(verification_helpers, "openrouter_chat", judge)
    receipt = await intent_details.review_intent_details(*inputs)

    assert calls == 1
    assert receipt["decision"] == "match"


@pytest.mark.asyncio
async def test_authenticated_first_observed_date_is_not_publication_recency(
    monkeypatch,
):
    job_quote = "HarborSoft is actively hiring a Platform Engineer."
    observation_quote = (
        "The provider first observed this listing on 2026-09-20; this does "
        "not establish the posting, publication, or opening date."
    )
    inputs = _inputs(
        company_name="HarborSoft",
        paragraph=f"{job_quote} {observation_quote}",
        signal_quote=job_quote,
        source_text=job_quote,
    )
    inputs[3]["dimension_evidence"]["identity"] = {
        "decision": "match",
        "web_identity_receipt": {
            "decision": "match",
            "observed_domain": "harborsoft.example",
        },
    }
    calls = 0

    async def judge(prompt, **_kwargs):
        nonlocal calls
        calls += 1
        document = json.loads(prompt)
        provider_source = next(
            source for source in document["admitted_evidence"]
            if source["evidence_kind"] == "authenticated_provider_observation"
        )
        assert "observed_dates" not in provider_source
        return json.dumps(_response(document, [
            (True, "VERIFIED", [job_quote]),
            (True, "VERIFIED", [observation_quote]),
        ]))

    monkeypatch.setattr(verification_helpers, "openrouter_chat", judge)
    with use_evaluation_date("2026-09-27"):
        receipt = await intent_details.review_intent_details(
            *inputs,
            authenticated_provider_observation={
                "matched_icp_signal": 0,
                "source_url": inputs[0].intent_signals[0].url,
                "first_observed_date": "2026-09-20",
            },
        )

    assert calls == 1
    assert receipt["decision"] == "match"


@pytest.mark.asyncio
async def test_citation_and_temporal_findings_share_one_repair(monkeypatch):
    job_quote = "HarborSoft is actively hiring a Platform Engineer."
    stage_quote = "HarborSoft is a Series B company."
    inputs = _inputs(
        company_name="HarborSoft",
        paragraph=(
            f"{job_quote} Recent coverage identifies HarborSoft as a "
            "Series B company."
        ),
        signal_quote=job_quote,
        source_text=job_quote,
        company_dimension="stage",
        company_quote=stage_quote,
    )
    calls = []

    async def judge(prompt, **_kwargs):
        calls.append(prompt)
        payload = json.loads(prompt)
        document = payload.get("review_document", payload)
        if len(calls) == 1:
            response = _response(document, [
                (True, "VERIFIED", [job_quote]),
                (True, "VERIFIED", [stage_quote]),
            ])
            response["unit_grounding"][0]["evidence"][0]["quote"] = (
                "UNBOUND JOB QUOTE"
            )
            return json.dumps(response)
        assert payload["bounded_unit_repair_control"]["units"] == [
            {
                "unit_id": 0,
                "contains_factual_claim": True,
                "status": "VERIFIED",
                "citation_errors": ["nonexact_quote"],
                "semantic_recheck_allowed": False,
            },
            {
                "unit_id": 1,
                "contains_factual_claim": True,
                "status": "VERIFIED",
                "citation_errors": [],
                "semantic_recheck_allowed": True,
                "semantic_recheck_reason": "relative_time_grounding_review",
            },
        ]
        return json.dumps({"repairs": [
            {
                "unit_id": 0,
                "status": "VERIFIED",
                "evidence": [_binding(document, job_quote)],
            },
            {"unit_id": 1, "status": "UNPROVEN", "evidence": []},
        ]})

    monkeypatch.setattr(verification_helpers, "openrouter_chat", judge)
    receipt = await intent_details.review_intent_details(*inputs)

    assert len(calls) == 2
    assert receipt["decision"] == "mismatch"


@pytest.mark.asyncio
@pytest.mark.parametrize("repair_result", ["malformed", "provider_error"])
async def test_temporal_recheck_failure_remains_unavailable(
    monkeypatch, repair_result,
):
    quote = "HarborSoft announced a funding round."
    inputs = _inputs(
        company_name="HarborSoft",
        paragraph="HarborSoft recently announced a funding round.",
        signal_quote=quote,
        source_text=quote,
    )
    calls = 0

    async def judge(prompt, **_kwargs):
        nonlocal calls
        calls += 1
        document = json.loads(prompt).get(
            "review_document", json.loads(prompt),
        )
        if calls == 2:
            if repair_result == "provider_error":
                raise RuntimeError("test provider error")
            return "{malformed"
        return json.dumps(_response(
            document, [(True, "VERIFIED", [quote])],
        ))

    monkeypatch.setattr(verification_helpers, "openrouter_chat", judge)
    receipt = await intent_details.review_intent_details(*inputs)

    assert calls == 2
    assert receipt["decision"] == "unavailable"
    assert receipt["failure_reason_code"] == (
        "provider_error" if repair_result == "provider_error"
        else "malformed_response"
    )
