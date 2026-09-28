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
        source_publication_date="2026-04-21",
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

    calls = 0

    async def judge(prompt, **_kwargs):
        nonlocal calls
        calls += 1
        payload = json.loads(prompt)
        document = payload.get("review_document", payload)
        evidence = [
            _binding(document, source_text),
            _binding(document, "2026-09-01"),
        ]
        if calls == 2:
            return json.dumps({"repairs": [{
                "unit_id": 0, "status": "VERIFIED", "evidence": evidence,
            }]})
        return json.dumps(_response(
            document,
            [(True, "VERIFIED", [source_text, "2026-09-01"])],
        ))

    monkeypatch.setattr(verification_helpers, "openrouter_chat", judge)
    receipt = await intent_details.review_intent_details(*inputs)

    assert calls == 2
    assert receipt["decision"] == "match"


@pytest.mark.asyncio
async def test_undated_relative_coverage_uses_one_bounded_semantic_recheck(
    monkeypatch,
):
    job_quote = "HarborSoft is actively hiring a Platform Engineer."
    stage_quote = "HarborSoft is a Series B company."
    inputs = _inputs(
        company_name="HarborSoft",
        paragraph=(
            f"{job_quote} The canonical posting describes platform "
            "responsibilities, while recent coverage identifies HarborSoft's "
            "latest known institutional stage as Series B."
        ),
        signal_quote=job_quote,
        source_text=job_quote,
        source_publication_date="2026-04-21",
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
                (True, "VERIFIED", ["2026-04-21", stage_quote]),
            ]))
        control = payload["bounded_unit_repair_control"]
        assert len(control["units"]) == 1
        repair_unit = control["units"][0]
        target = repair_unit.pop("relative_time_target")
        assert repair_unit == {
            "unit_id": 1,
            "contains_factual_claim": True,
            "status": "VERIFIED",
            "citation_errors": [],
            "semantic_recheck_allowed": True,
            "semantic_recheck_reason": "relative_time_grounding_review",
        }
        unit_text = document["intent_details_units"][1]["text"]
        disputed_clause = (
            "recent coverage identifies HarborSoft's latest known "
            "institutional stage as Series B."
        )
        clause_start = unit_text.index(disputed_clause)
        qualifier_start = unit_text.index("recent coverage")
        assert target == {
            "qualifier_start": qualifier_start,
            "qualifier_end": qualifier_start + len("recent coverage"),
            "clause_start": clause_start,
            "clause_end": len(unit_text),
            "disputed_clause": disputed_clause,
            "offset_basis": "intent_details_units[unit_id].text",
            "untrusted_claim_text": True,
            "held_evidence_bindings": [
                {
                    "source_index": _binding(document, "2026-04-21")[
                        "source_index"
                    ],
                    "quote": "2026-04-21",
                    "source_url": "https://harborsoft.example/evidence",
                    "evidence_kind": "verified_source_context",
                },
                {
                    "source_index": _binding(document, stage_quote)[
                        "source_index"
                    ],
                    "quote": stage_quote,
                    "source_url": "https://harborsoft.example/stage",
                    "evidence_kind": "verified_company_fact",
                },
            ],
        }
        job_source = document["admitted_evidence"][
            target["held_evidence_bindings"][0]["source_index"]
        ]
        stage_source = document["admitted_evidence"][
            target["held_evidence_bindings"][1]["source_index"]
        ]
        assert job_source["observed_dates"] == [{
            "date": "2026-04-21", "basis": "source_publication_date",
        }]
        assert "observed_dates" not in stage_source
        assert "Trusted routing controls (not factual evidence)" in kwargs[
            "system_prompt"
        ]
        assert kwargs["system_prompt"].startswith(
            "Recheck the factual support of only the flagged paragraph units."
        )
        assert "Review a client-facing Intent Details" not in kwargs[
            "system_prompt"
        ]
        assert "facts_supported must equal" not in kwargs["system_prompt"]
        assert "events, sources, or clauses" in kwargs["system_prompt"]
        assert (
            "first classify the exact disputed clause" in
            kwargs["system_prompt"]
        )
        assert (
            "source neither supports nor contradicts that timing claim" in
            kwargs["system_prompt"]
        )
        assert (
            "CONTRADICTED applies only when evidence\n"
            "about that same timed subject proves incompatible timing"
            in kwargs["system_prompt"]
        )
        assert (
            "otherwise UNPROVEN if\nany clause is unproven; otherwise VERIFIED"
            in kwargs["system_prompt"]
        )
        assert (
            "Do not preserve unsupported other\nclauses" in
            kwargs["system_prompt"]
        )
        assert disputed_clause not in kwargs["system_prompt"]
        assert disputed_clause in prompt
        return json.dumps({"repairs": [{
            "unit_id": 1, "status": "UNPROVEN", "evidence": [],
        }]})

    monkeypatch.setattr(verification_helpers, "openrouter_chat", judge)
    receipt = await intent_details.review_intent_details(*inputs)

    assert len(calls) == 2
    assert receipt["decision"] == "mismatch"
    assert receipt["checks"]["facts_supported"] is False


def _relative_source(
    source_index: int,
    text: str,
    *,
    observed_date: str = "",
    evidence_kind: str = "verified_source_context",
) -> dict:
    return {
        "source_index": source_index,
        "evidence_kind": evidence_kind,
        "admitted_text": [text],
        **({
            "observed_dates": [{
                "date": observed_date,
                "basis": "source_publication_date",
            }],
        } if observed_date else {}),
    }


def _relative_repair_has_date(sources: list[dict], *bindings) -> bool:
    return intent_details._relative_time_repair_has_source_bound_date(
        {"unit_id": 0, "status": "VERIFIED", "evidence": list(bindings)},
        {"admitted_evidence": sources},
    )


def test_verified_relative_repair_rejects_missing_or_unbound_date_proof():
    stage = "HarborSoft is a Series B company."
    observed = "The provider first observed this listing on 2026-09-20."
    invalid = "On 2026-02-30 HarborSoft published a funding report."
    stage_source = _relative_source(
        5, stage, evidence_kind="verified_company_fact",
    )
    job_source = _relative_source(
        0, "HarborSoft is hiring an engineer.", observed_date="2026-04-21",
    )
    observation_source = _relative_source(
        2, observed, evidence_kind="authenticated_provider_observation",
    )

    assert not _relative_repair_has_date(
        [stage_source], {"source_index": 5, "quote": stage},
    )
    assert not _relative_repair_has_date(
        [job_source, stage_source],
        {"source_index": 0, "quote": "2026-04-21"},
        {"source_index": 5, "quote": stage},
    )
    assert not _relative_repair_has_date(
        [observation_source], {"source_index": 2, "quote": observed},
    )
    assert not _relative_repair_has_date(
        [_relative_source(3, invalid)],
        {"source_index": 3, "quote": invalid},
    )
    assert not _relative_repair_has_date(
        [_relative_source(4, "Publication date: 2026-09-01")],
        {"source_index": 4, "quote": "Publication date: 2026-09-01"},
    )
    for hidden_date in (
        "[HarborSoft report](https://example.com/2026-09-01)",
        "![HarborSoft report dated 2026-09-01](https://example.com/image.png)",
        "[HarborSoft report](not-a-valid-url/2026-09-01)",
        "Monday, Sep. 1, 2026 10:00 GMT",
    ):
        assert not _relative_repair_has_date(
            [_relative_source(6, hidden_date)],
            {"source_index": 6, "quote": hidden_date},
        )


def test_verified_relative_repair_accepts_same_source_date_proof():
    report = "HarborSoft published its market report."
    posting = "Platform Engineer role for HarborSoft's workflow product."

    for date_text in (
        "September 1, 2026",
        "Sep 1, 2026",
        "Sep. 1, 2026",
        "Sept. 1, 2026",
        "1 September 2026",
        "1 Sep 2026",
        "1 Sept. 2026",
    ):
        dated_report = f"{date_text}. {report}"
        assert _relative_repair_has_date(
            [_relative_source(0, dated_report)],
            {"source_index": 0, "quote": dated_report},
        )
    for text, observed_date in (
        (report, "2026-09-01"),
        (posting, "2026-09-25"),
    ):
        assert _relative_repair_has_date(
            [_relative_source(0, text, observed_date=observed_date)],
            {"source_index": 0, "quote": observed_date},
            {"source_index": 0, "quote": text},
        )


def test_exact_retained_unibuddy_repair_fails_closed_without_date_proof():
    stage_quote = (
        "You may have heard the news, we're proud to have raised a whopping "
        "$20 million USD in Series B funding and we're excited!"
    )
    document = {"admitted_evidence": [
        _relative_source(
            0,
            "Software Engineer II - Chat Systems. Unibuddy is hiring.",
            observed_date="2026-04-21",
        ),
        _relative_source(
            5, stage_quote, evidence_kind="verified_company_fact",
        ),
    ]}
    exact_action_21 = json.dumps({"repairs": [{
        "evidence": [{"quote": stage_quote, "source_index": 5}],
        "status": "VERIFIED",
        "unit_id": 1,
    }]})

    with pytest.raises(
        ValueError,
        match="verified relative-time repair lacks source-bound date proof",
    ):
        intent_details._validate_relative_time_repair_date_proof(
            exact_action_21,
            document,
            {1: {"disputed_clause": "recent coverage identifies Series B"}},
        )


@pytest.mark.asyncio
async def test_forged_date_bearing_relative_repair_remains_unavailable(
    monkeypatch,
):
    stage_quote = "HarborSoft is a Series B company."
    inputs = _inputs(
        company_name="HarborSoft",
        paragraph="Recent coverage identifies HarborSoft as Series B.",
        signal_quote="HarborSoft is actively hiring a Platform Engineer.",
        source_text="HarborSoft is actively hiring a Platform Engineer.",
        company_dimension="stage",
        company_quote=stage_quote,
    )
    calls = 0

    async def judge(prompt, **_kwargs):
        nonlocal calls
        calls += 1
        payload = json.loads(prompt)
        document = payload.get("review_document", payload)
        if calls == 1:
            return json.dumps(_response(
                document, [(True, "VERIFIED", [stage_quote])],
            ))
        stage_binding = _binding(document, stage_quote)
        stage_binding["quote"] = (
            "On 2026-09-01 HarborSoft was covered as a Series B company."
        )
        return json.dumps({"repairs": [{
            "unit_id": 0,
            "status": "VERIFIED",
            "evidence": [stage_binding],
        }]})

    monkeypatch.setattr(verification_helpers, "openrouter_chat", judge)
    receipt = await intent_details.review_intent_details(*inputs)

    assert calls == 2
    assert receipt["decision"] == "unavailable"
    assert receipt["failure_reason_code"] == "malformed_response"


@pytest.mark.asyncio
async def test_date_bearing_job_does_not_semantically_date_funding_coverage(
    monkeypatch,
):
    job_quote = (
        "On 2026-04-21 HarborSoft posted a Platform Engineer opening."
    )
    stage_quote = "HarborSoft is a Series B company."
    inputs = _inputs(
        company_name="HarborSoft",
        paragraph=(
            f"{job_quote} Recent coverage identifies HarborSoft's latest "
            "institutional stage as Series B."
        ),
        signal_quote=job_quote,
        source_text=job_quote,
        source_publication_date="2026-04-21",
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
        assert "substantive\nsupport from the same source_index" in kwargs[
            "system_prompt"
        ]
        return json.dumps({"repairs": [{
            "unit_id": 1,
            "status": "UNPROVEN",
            "evidence": [],
        }]})

    monkeypatch.setattr(verification_helpers, "openrouter_chat", judge)
    receipt = await intent_details.review_intent_details(*inputs)

    assert len(calls) == 2
    assert receipt["decision"] == "mismatch"
    assert receipt["checks"]["facts_supported"] is False


@pytest.mark.asyncio
async def test_dated_relative_claim_keeps_unsupported_other_clause_unproven(
    monkeypatch,
):
    report_quote = (
        "On 2026-09-01 HarborSoft published its Workflow Market Outlook."
    )
    inputs = _inputs(
        company_name="HarborSoft",
        paragraph=(
            "Recent coverage dated 2026-09-01 reports HarborSoft's Workflow "
            "Market Outlook and says it won a national university award."
        ),
        signal_quote=report_quote,
        source_text=report_quote,
        source_publication_date="2026-09-01",
    )
    calls = 0

    async def judge(prompt, **_kwargs):
        nonlocal calls
        calls += 1
        payload = json.loads(prompt)
        document = payload.get("review_document", payload)
        if calls == 1:
            return json.dumps(_response(
                document,
                [(True, "VERIFIED", [report_quote, "2026-09-01"])],
            ))
        return json.dumps({"repairs": [{
            "unit_id": 0,
            "status": "UNPROVEN",
            "evidence": [],
        }]})

    monkeypatch.setattr(verification_helpers, "openrouter_chat", judge)
    receipt = await intent_details.review_intent_details(*inputs)

    assert calls == 2
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
        target = payload["bounded_unit_repair_control"]["units"][0][
            "relative_time_target"
        ]
        assert target["disputed_clause"] == document[
            "intent_details_units"
        ][0]["text"]
        assert target["held_evidence_bindings"][0]["quote"] == dated_excerpt
        assert "date-bearing" in kwargs["system_prompt"]
        assert "admitted source excerpt" in kwargs["system_prompt"]
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
async def test_typed_date_binding_is_rechecked_and_can_remain_verified(
    monkeypatch,
):
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
        payload = json.loads(prompt)
        document = payload.get("review_document", payload)
        evidence = [
            _binding(document, dated_excerpt),
            _binding(document, "2026-09-01"),
        ]
        if calls == 2:
            return json.dumps({"repairs": [{
                "unit_id": 0, "status": "VERIFIED", "evidence": evidence,
            }]})
        return json.dumps(_response(
            document,
            [(True, "VERIFIED", [dated_excerpt, "2026-09-01"])],
        ))

    monkeypatch.setattr(verification_helpers, "openrouter_chat", judge)
    receipt = await intent_details.review_intent_details(*inputs)

    assert calls == 2
    assert receipt["decision"] == "match"


@pytest.mark.asyncio
async def test_relative_posting_repairs_three_quotes_to_two_complete_bindings(
    monkeypatch,
):
    context_quote = (
        "Platform Engineer role for HarborSoft's workflow product."
    )
    inputs = _inputs(
        company_name="HarborSoft",
        paragraph=(
            "HarborSoft's Platform Engineer role was recently posted for "
            "its workflow product."
        ),
        signal_quote=context_quote,
        source_text=context_quote,
        source_publication_date="2026-09-25",
    )
    calls = []

    async def judge(prompt, **kwargs):
        calls.append((prompt, kwargs))
        payload = json.loads(prompt)
        document = payload.get("review_document", payload)
        if len(calls) == 1:
            return json.dumps(_response(document, [(
                True,
                "VERIFIED",
                [
                    "2026-09-25",
                    "Platform Engineer role",
                    "HarborSoft's workflow product",
                ],
            )]))
        repair_unit = payload["bounded_unit_repair_control"]["units"][0]
        assert repair_unit["citation_errors"] == ["too_many_quotes"]
        assert repair_unit["semantic_recheck_allowed"] is True
        assert repair_unit["semantic_recheck_reason"] == (
            "relative_time_grounding_review"
        )
        assert "Treat each citation_errors value as a required output" in (
            kwargs["system_prompt"]
        )
        assert "For too_many_quotes, return at most two" in kwargs[
            "system_prompt"
        ]
        assert "never the original oversized list" in kwargs["system_prompt"]
        assert "Do not omit support for any factual clause" in kwargs[
            "system_prompt"
        ]
        return json.dumps({"repairs": [{
            "unit_id": 0,
            "status": "VERIFIED",
            "evidence": [
                _binding(document, "2026-09-25"),
                _binding(document, context_quote),
            ],
        }]})

    monkeypatch.setattr(verification_helpers, "openrouter_chat", judge)
    receipt = await intent_details.review_intent_details(*inputs)

    assert len(calls) == 2
    assert receipt["decision"] == "match"
    assert receipt["checks"]["facts_supported"] is True


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
        units = payload["bounded_unit_repair_control"]["units"]
        relative_target = units[1].pop("relative_time_target")
        assert units == [
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
        assert relative_target["disputed_clause"] == (
            "Recent coverage identifies HarborSoft as a Series B company."
        )
        assert [
            binding["quote"]
            for binding in relative_target["held_evidence_bindings"]
        ] == [stage_quote]
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
