"""Bounded unit grounding for the existing Intent Details review call."""

from __future__ import annotations

import asyncio
import hashlib
import json
from types import SimpleNamespace

import pytest

from qualification.scoring import intent_details, verification_helpers
from qualification.scoring.competition import (
    scorer_breakdown_has_retryable_infrastructure_failure,
)


@pytest.mark.parametrize("quote,source", [
    (
        "Since early 2024, Common Wealth's employer base has grown over 3x",
        "Since early 2024, Common Wealth’s employer base has grown over 3x, "
        "plan membership has grown over 3.5x.",
    ),
    ('Acme calls it "early access".', 'Acme calls it “early access”.'),
    ("Acme’s product is in ‘early access’.", "Acme's product is in 'early access'."),
])
def test_evidence_binding_accepts_quote_typography_only(quote, source):
    assert intent_details._quote_is_bound(quote, [source])


@pytest.mark.parametrize("quote,source", [
    ("Common Wealth's employer base grows three per day", "Common Wealth’s employer base has grown over 3x"),
    ("Acme's product is launched", "Acme’s product is not launched"),
    ("Acme's product is launched", "Acme’s product is planned. It has not launched"),
    ("Acme's product launched in 2026", "Acme’s product launched in 2025"),
    ("Acme raised $30M", "Acme raised $3.0M"),
    ("Acme's product has launched", "Acme’s product is planned; a competitor has launched"),
    ("Acme's product has launched", "Acme’s product ... has launched"),
    ("Acme's product launched", "ACME’s product launched"),
])
def test_evidence_binding_does_not_normalize_claim_content(quote, source):
    assert not intent_details._quote_is_bound(quote, [source])


def test_evidence_binding_does_not_stitch_across_sources():
    assert not intent_details._quote_is_bound(
        "Example launched its analytics product",
        ["Example launched its", "analytics product"],
    )


def test_private_response_schema_orders_grounding_before_aggregates():
    schema = intent_details._RESPONSE_FORMAT["json_schema"]["schema"]
    expected = ["unit_grounding", "signal_coverage", *intent_details._CHECKS]
    evidence_schema = schema["properties"]["unit_grounding"]["items"][
        "properties"
    ]["evidence"]

    assert list(schema["properties"]) == expected
    assert schema["required"] == expected
    assert evidence_schema["maxItems"] == 2
    assert evidence_schema["items"]["properties"]["quote"]["maxLength"] == 500
    assert "unsupported_factual_clause" not in schema["properties"]
    assert "unsupported_factual_reason" not in schema["properties"]
    repair_schema = intent_details._CITATION_REPAIR_RESPONSE_FORMAT[
        "json_schema"
    ]["schema"]
    assert set(repair_schema["properties"]) == {"repairs"}
    repair_item = repair_schema["properties"]["repairs"]["items"]
    assert set(repair_item["properties"]) == {"unit_id", "evidence"}
    assert repair_item["properties"]["evidence"]["maxItems"] == 2
    assert repair_item["properties"]["evidence"]["items"]["properties"][
        "quote"
    ]["maxLength"] == 500


def _inputs(
    paragraph: str,
    source_text: str,
    *,
    supporting_quote: str,
    authoritative_date: str = "2026-03-01",
):
    url = "https://example.test/news"
    company = SimpleNamespace(
        company_name="Example",
        company_website="https://example.test/",
        intent_details=paragraph,
        intent_signals=[SimpleNamespace(
            matched_icp_signal=0,
            description="Submitted claim text is not evidence.",
            url=url,
        )],
    )
    icp = SimpleNamespace(
        prompt="Find companies with a recent supported activity.",
        product_service="Business software",
        intent_signals=["Recent supported activity"],
    )
    results = [{
        "after_decay": 51,
        "matched_icp_signal": 0,
        "evidence_urls": [url],
        "judge_verdict": {
            "decision": "verified",
            "client_ready": True,
            "authoritative_date": authoritative_date,
            "authoritative_date_basis": "event",
            "verification_trace": {
                "intent_verdict": {"signal_evaluations": [{
                    "signal_status": "supported",
                    "same_entity_check": "pass",
                    "supporting_quotes": [supporting_quote],
                    "evidence_urls_used": [url],
                }]},
                "verified_source_context": [{
                    "url": url,
                    "text": source_text,
                    "source_publication_date": "2026-03-02",
                }],
            },
        },
    }]
    fit = {
        "gate": "company_fit",
        "decision": "match",
        "dimension_evidence": {
            "industry": {
                "decision": "match",
                "web_evidence": {
                    "url": "https://example.test/",
                    "quote": "Example provides business software.",
                },
            },
        },
    }
    return company, icp, results, fit


def _binding(document, text: str) -> dict:
    for source_index, values in intent_details._bound_evidence_sources(
        document
    ).items():
        if any(text in value for value in values):
            return {"source_index": source_index, "quote": text}
    raise AssertionError(f"No trusted evidence contains {text!r}")


def _response(document, unit_grounding, *, facts_supported):
    return {
        "unit_grounding": unit_grounding,
        "signal_coverage": [{"matched_icp_signal": 0, "covered": True}],
        **{name: True for name in intent_details._CHECKS},
        "facts_supported": facts_supported,
    }


def _repair_response(unit_id: int, evidence: list[dict]) -> dict:
    return {"repairs": [{"unit_id": unit_id, "evidence": evidence}]}


def _verified_unit(document, unit_id: int, quote: str) -> dict:
    return {
        "unit_id": unit_id,
        "contains_factual_claim": True,
        "status": "VERIFIED",
        "evidence": [_binding(document, quote)],
    }


def _unproven_unit(unit_id: int) -> dict:
    return {
        "unit_id": unit_id,
        "contains_factual_claim": True,
        "status": "UNPROVEN",
        "evidence": [],
    }


def _nonfactual_unit(unit_id: int) -> dict:
    return {
        "unit_id": unit_id,
        "contains_factual_claim": False,
        "status": "VERIFIED",
        "evidence": [],
    }


def _prompt_document(prompt: str) -> dict:
    payload = json.loads(prompt)
    return payload.get("review_document", payload)


def _review(monkeypatch, inputs, response_builder):
    async def judge(prompt, **kwargs):
        assert kwargs["max_tokens"] == 800
        document = _prompt_document(prompt)
        return json.dumps(response_builder(document), separators=(",", ":"))

    monkeypatch.setattr(verification_helpers, "openrouter_chat", judge)
    return asyncio.run(intent_details.review_intent_details(*inputs))


def test_typesafe_api_claim_is_an_unproven_complete_unit(monkeypatch):
    paragraph = (
        "Dealroom's September 15, 2026 coverage reports TypeSafe's seed "
        "financing and says Jev is in early access for select developers. "
        "Its API reference documents Jev-powered evaluation of application "
        "state into structured answers."
    )
    supported = (
        "TypeSafe raised seed financing and Jev is in early access for select "
        "developers."
    )
    inputs = _inputs(paragraph, supported, supporting_quote=supported)

    def response(document):
        assert len(document["intent_details_units"]) == 2
        assert "API reference" not in " ".join(
            value
            for values in intent_details._bound_evidence_sources(document).values()
            for value in values
        )
        return _response(
            document,
            [
                _verified_unit(document, 0, supported[:100]),
                _unproven_unit(1),
            ],
            facts_supported=False,
        )

    receipt = _review(monkeypatch, inputs, response)
    assert receipt["decision"] == "mismatch"
    assert receipt["checks"]["facts_supported"] is False


def test_common_wealth_daily_growth_cannot_be_omitted(monkeypatch):
    paragraph = (
        "Common Wealth closed a C$12M Series A in March 2026 after bridge "
        "financing supported product innovation and customer growth. In a 2026 "
        "update, the company reports adding about three employers every business "
        "day, with 80% offering retirement benefits for the first time. Flow "
        "Capital reports that Common Wealth was selected to administer Canada's "
        "$30M Personal Support Worker Retirement Savings Innovation Program."
    )
    funding = "Common Wealth closed a C$12M Series A in March 2026."
    first_plans = "Over 80% of employers offer a retirement plan for the first time."
    program = (
        "Common Wealth was selected to administer Canada's $30M Personal "
        "Support Worker Retirement Savings Innovation Program."
    )
    source = f"{funding} {first_plans} {program}"
    inputs = _inputs(paragraph, source, supporting_quote=funding)

    def response(document):
        assert len(document["intent_details_units"]) == 3
        admitted = " ".join(
            value
            for values in intent_details._bound_evidence_sources(document).values()
            for value in values
        )
        assert "three employers every business day" not in admitted
        return _response(
            document,
            [
                _verified_unit(document, 0, funding),
                _unproven_unit(1),
                _verified_unit(document, 2, program[:100]),
            ],
            facts_supported=False,
        )

    receipt = _review(monkeypatch, inputs, response)
    assert receipt["decision"] == "mismatch"
    assert receipt["checks"]["facts_supported"] is False


def test_supported_paraphrase_uses_exact_source_span_without_literal_match(
    monkeypatch,
):
    paragraph = "Example secured growth financing to expand its operations."
    quote = "Example raised a $25 million Series A round."
    inputs = _inputs(paragraph, quote, supporting_quote=quote)

    def response(document):
        assert "secured growth financing" not in quote
        return _response(
            document,
            [_verified_unit(document, 0, quote)],
            facts_supported=True,
        )

    receipt = _review(monkeypatch, inputs, response)
    assert receipt["decision"] == "match"


def test_compound_unit_can_bind_two_existing_sources(monkeypatch):
    paragraph = "Example raised a Series A and launched its analytics product."
    financing = "Example raised a Series A."
    launch = "Example launched its analytics product."
    inputs = _inputs(
        paragraph,
        launch,
        supporting_quote=financing,
    )

    def response(document):
        signal_source = _binding(document, financing)
        context_source = _binding(document, launch)["source_index"]
        unit = _verified_unit(document, 0, financing)
        unit["evidence"] = [
            signal_source,
            {"source_index": context_source, "quote": launch},
        ]
        return _response(document, [unit], facts_supported=True)

    receipt = _review(monkeypatch, inputs, response)
    assert receipt["decision"] == "match"


def test_nonfactual_unit_prompt_requests_no_binding_but_parser_accepts_one(
    monkeypatch,
):
    paragraph = (
        "Example raised a $25 million Series A. This may support future product "
        "delivery."
    )
    quote = "Example raised a $25 million Series A."
    inputs = _inputs(paragraph, quote, supporting_quote=quote)

    def response(document):
        conditional = _nonfactual_unit(1)
        # Preserve backward compatibility for an otherwise valid response from
        # a provider that still binds the factual premise.
        conditional["evidence"] = [_binding(document, quote)]
        return _response(
            document,
            [
                _verified_unit(document, 0, quote),
                conditional,
            ],
            facts_supported=True,
        )

    async def judge(prompt, **kwargs):
        assert (
            "A unit with no factual claim must be\nVERIFIED and return an empty "
            "evidence list. Still assess its premise under the\naggregate "
            "commercial relevance and ICP checks."
        ) in kwargs["system_prompt"]
        return json.dumps(response(_prompt_document(prompt)), separators=(",", ":"))

    monkeypatch.setattr(verification_helpers, "openrouter_chat", judge)
    receipt = asyncio.run(intent_details.review_intent_details(*inputs))
    assert receipt["decision"] == "match"
    assert receipt["checks"]["relevance_grounded"] is True


def test_unproven_compound_unit_can_bind_its_supported_part(monkeypatch):
    paragraph = (
        "Example raised a Series A and added three customers every business day."
    )
    quote = "Example raised a Series A."
    inputs = _inputs(paragraph, quote, supporting_quote=quote)

    def response(document):
        unit = _unproven_unit(0)
        unit["evidence"] = [_binding(document, quote)]
        return _response(
            document,
            [unit],
            facts_supported=False,
        )

    receipt = _review(monkeypatch, inputs, response)
    assert receipt["decision"] == "mismatch"
    assert receipt["checks"]["facts_supported"] is False


def test_all_nonfactual_units_conflict_with_complete_signal_coverage(monkeypatch):
    paragraph = "This may support future delivery. This could create demand."
    quote = "Example raised a Series A."
    inputs = _inputs(paragraph, quote, supporting_quote=quote)

    def response(document):
        units = [
            _nonfactual_unit(unit["unit_id"])
            for unit in document["intent_details_units"]
        ]
        return _response(document, units, facts_supported=True)

    receipt = _review(monkeypatch, inputs, response)
    assert receipt["decision"] == "unavailable"
    assert receipt["failure_reason_code"] == "malformed_response"


@pytest.mark.parametrize("mutation", ["omitted", "duplicate"])
def test_incomplete_or_duplicate_unit_ids_remain_retryable(monkeypatch, mutation):
    paragraph = "Example raised a Series A. Example launched a product."
    quote = "Example raised a Series A and launched a product."
    inputs = _inputs(paragraph, quote, supporting_quote=quote)

    def response(document):
        units = [
            _verified_unit(document, unit["unit_id"], quote)
            for unit in document["intent_details_units"]
        ]
        if mutation == "omitted":
            units.pop()
        else:
            units[-1]["unit_id"] = units[0]["unit_id"]
        return _response(document, units, facts_supported=True)

    receipt = _review(monkeypatch, inputs, response)
    assert receipt["decision"] == "unavailable"
    assert receipt["failure_reason_code"] == "malformed_response"
    assert scorer_breakdown_has_retryable_infrastructure_failure({
        "verifier_gate_receipts": [receipt],
    })


@pytest.mark.parametrize(
    "invalid", ["paragraph_self_quote", "source_index", "source_url"]
)
def test_unbound_or_invalid_source_evidence_is_retryable(monkeypatch, invalid):
    paragraph = "Example claims an undocumented API capability."
    quote = "Example raised a Series A."
    inputs = _inputs(paragraph, quote, supporting_quote=quote)

    def response(document):
        unit = _verified_unit(document, 0, quote)
        if invalid == "paragraph_self_quote":
            unit["evidence"][0]["quote"] = paragraph
        elif invalid == "source_index":
            unit["evidence"][0]["source_index"] = 999
        else:
            unit["evidence"][0]["quote"] = "https://example.test/news"
        return _response(document, [unit], facts_supported=True)

    receipt = _review(monkeypatch, inputs, response)
    assert receipt["decision"] == "unavailable"
    assert receipt["failure_reason_code"] == "malformed_response"


def test_authoritative_date_scalar_is_an_allowed_bound_source(monkeypatch):
    paragraph = "Example completed the launch on March 1, 2026."
    quote = "Example completed the launch."
    inputs = _inputs(
        paragraph,
        quote,
        supporting_quote=quote,
        authoritative_date="2026-03-01",
    )

    def response(document):
        return _response(
            document,
            [_verified_unit(document, 0, "2026-03-01")],
            facts_supported=True,
        )

    receipt = _review(monkeypatch, inputs, response)
    assert receipt["decision"] == "match"


def test_verified_company_fact_quote_is_an_allowed_bound_source(monkeypatch):
    paragraph = "Example provides business software."
    signal_quote = "Example raised a Series A."
    inputs = _inputs(paragraph, signal_quote, supporting_quote=signal_quote)
    company_quote = "Example provides business software."

    def response(document):
        return _response(
            document,
            [_verified_unit(document, 0, company_quote)],
            facts_supported=True,
        )

    receipt = _review(monkeypatch, inputs, response)
    assert receipt["decision"] == "match"


def test_projection_exposes_only_flat_admitted_evidence_as_bindable():
    paragraph = "Example launched its analytics product on March 1, 2026."
    source = "Example launched its analytics product on March 1, 2026."
    inputs = _inputs(
        paragraph,
        source,
        supporting_quote=source,
        authoritative_date="2026-03-01",
    )
    company, icp, results, fit = inputs
    company.intent_signals[0].description = (
        "SUBMITTED TITLE: Example added one hundred customers each day."
    )
    document = intent_details.review_evidence(company, icp, results, fit)
    sources = intent_details._bound_evidence_sources(document)
    bound_text = " ".join(value for values in sources.values() for value in values)

    assert document["admitted_evidence"] == [
        {
            "source_index": 0,
            "evidence_kind": "verified_source_context",
            "matched_icp_signal": 0,
            "source_url": "https://example.test/news",
            "admitted_text": [source],
            "observed_dates": [{
                "date": "2026-03-02", "basis": "source_publication_date",
            }],
        },
        {
            "source_index": 1,
            "evidence_kind": "verified_signal_observation",
            "matched_icp_signal": 0,
            "source_url": "https://example.test/news",
            "observed_dates": [{"date": "2026-03-01", "basis": "event"}],
        },
        {
            "source_index": 2,
            "evidence_kind": "verified_company_fact",
            "company_dimension": "industry",
            "source_url": "https://example.test/",
            "admitted_text": ["Example provides business software."],
        },
    ]
    assert document["verified_signals"][0]["evidence_source_indexes"] == [0, 1]
    assert document["verified_signals"][0][
        "untrusted_submitted_claim_context"
    ] == ["SUBMITTED TITLE: Example added one hundred customers each day."]
    assert "SUBMITTED TITLE" not in bound_text
    assert source in bound_text
    assert sources[0] == [source, "2026-03-02"]
    assert sources[1] == ["2026-03-01", "event"]

    def source_index_paths(value, path=()):
        paths = []
        if isinstance(value, dict):
            for key, item in value.items():
                if key == "source_index":
                    paths.append(path + (key,))
                paths.extend(source_index_paths(item, path + (key,)))
        elif isinstance(value, list):
            for index, item in enumerate(value):
                paths.extend(source_index_paths(item, path + (index,)))
        return paths

    assert all(path[0] == "admitted_evidence" for path in source_index_paths(document))


def test_prior_notes_are_omitted_and_submitted_title_is_not_bindable():
    paragraph = "Example raised a Series A."
    source = "Example raised a Series A."
    company, icp, results, fit = _inputs(
        paragraph, source, supporting_quote=source,
    )
    company.intent_signals[0].description = "SUBMITTED ALL-CAPS RESEARCH TITLE"
    rejected = json.loads(json.dumps(results[0]))
    rejected["after_decay"] = 0
    rejected["judge_verdict"].update(
        decision="rejected_three_stage", client_ready=False,
    )
    evaluation = rejected["judge_verdict"]["verification_trace"][
        "intent_verdict"
    ]["signal_evaluations"][0]
    evaluation.update({
        "verification_mode": "source_grounded",
        "signal_status": "contradicted",
        "supporting_quotes": [],
        "contradicting_quotes": ["Example announced a planned Series A."],
        "unsupported_parts": ["PRIOR NOTE: The launch claim is unsupported."],
    })
    document = intent_details.review_evidence(
        company, icp, results + [rejected], fit,
    )
    bound_text = " ".join(
        value
        for values in intent_details._bound_evidence_sources(document).values()
        for value in values
    )

    assert "SUBMITTED ALL-CAPS RESEARCH TITLE" in json.dumps(document)
    assert "PRIOR NOTE: The launch claim is unsupported." not in json.dumps(document)
    assert "SUBMITTED ALL-CAPS RESEARCH TITLE" not in bound_text
    assert "Example announced a planned Series A." in bound_text


def test_unpaired_quotes_do_not_claim_one_url_from_a_multi_source_signal():
    paragraph = "Example raised a Series A."
    source_quote = "Example raised a Series A."
    company, icp, results, fit = _inputs(
        paragraph,
        "A fetched page with related context.",
        supporting_quote=source_quote,
    )
    second_url = "https://second.example/news"
    results[0]["evidence_urls"].append(second_url)
    results[0]["judge_verdict"]["verification_trace"]["intent_verdict"][
        "signal_evaluations"
    ][0]["evidence_urls_used"].append(second_url)
    document = intent_details.review_evidence(company, icp, results, fit)
    observation = next(
        source for source in document["admitted_evidence"]
        if source["evidence_kind"] == "verified_signal_observation"
    )

    assert observation["admitted_text"] == [source_quote]
    assert "source_url" not in observation
    assert document["verified_signals"][0]["source_urls"] == [
        "https://example.test/news", second_url,
    ]


def test_company_fact_quotes_keep_their_paired_locators():
    company, icp, results, fit = _inputs(
        "Example raised a Series A.",
        "Example raised a Series A.",
        supporting_quote="Example raised a Series A.",
    )
    fit["dimension_evidence"]["industry"]["web_evidence"].update({
        "evidence_url": "https://profile.example/company",
        "evidence_quote": "Example is listed as a software company.",
    })
    document = intent_details.review_evidence(company, icp, results, fit)
    company_sources = [
        document["admitted_evidence"][source_index]
        for source_index in document["verified_company_evidence"]["industry"][
            "evidence_source_indexes"
        ]
    ]

    assert [(source["source_url"], source["admitted_text"]) for source in company_sources] == [
        ("https://example.test/", ["Example provides business software."]),
        (
            "https://profile.example/company",
            ["Example is listed as a software company."],
        ),
    ]


def test_existing_but_wrong_source_index_is_retryable(monkeypatch):
    paragraph = "Example launched its analytics product."
    body_quote = "Example launched its analytics product."
    inputs = _inputs(
        paragraph,
        body_quote,
        supporting_quote="Example raised a Series A.",
    )

    def response(document):
        body_binding = _binding(document, body_quote)
        other_indexes = [
            source_index
            for source_index in intent_details._bound_evidence_sources(document)
            if source_index != body_binding["source_index"]
        ]
        assert other_indexes
        body_binding["source_index"] = other_indexes[0]
        unit = {
            "unit_id": 0,
            "contains_factual_claim": True,
            "status": "VERIFIED",
            "evidence": [body_binding],
        }
        return _response(document, [unit], facts_supported=True)

    receipt = _review(monkeypatch, inputs, response)
    assert receipt["decision"] == "unavailable"
    assert receipt["failure_reason_code"] == "malformed_response"


@pytest.mark.parametrize(
    "length,expected",
    [(266, "match"), (292, "match"), (500, "match"), (501, "unavailable")],
)
def test_exact_evidence_quote_uses_500_character_hard_cap(
    monkeypatch, length, expected,
):
    quote = "Q" * length
    inputs = _inputs("Example published a detailed report.", quote, supporting_quote=quote)

    def response(document):
        unit = {
            "unit_id": 0,
            "contains_factual_claim": True,
            "status": "VERIFIED",
            "evidence": [_binding(document, quote)],
        }
        return _response(document, [unit], facts_supported=True)

    receipt = _review(monkeypatch, inputs, response)
    assert receipt["decision"] == expected
    if expected == "unavailable":
        assert receipt["failure_reason_code"] == "malformed_response"


def test_more_than_two_exact_evidence_bindings_is_retryable(monkeypatch):
    paragraph = "Example published a detailed report."
    quotes = [
        "Example published the report.",
        "The report covers business analytics.",
        "The report includes a market summary.",
    ]
    source = " ".join(quotes)
    inputs = _inputs(paragraph, source, supporting_quote=quotes[0])

    def response(document):
        source_index = _binding(document, source)["source_index"]
        unit = {
            "unit_id": 0,
            "contains_factual_claim": True,
            "status": "VERIFIED",
            "evidence": [
                {"source_index": source_index, "quote": quote}
                for quote in quotes
            ],
        }
        return _response(document, [unit], facts_supported=True)

    receipt = _review(monkeypatch, inputs, response)

    assert receipt["decision"] == "unavailable"
    assert receipt["failure_reason_code"] == "malformed_response"


@pytest.mark.parametrize(
    "defect,category",
    [
        ("missing", "missing_evidence"),
        ("invalid_index", "invalid_source_index"),
        ("empty", "empty_quote"),
        ("duplicate", "duplicate_quote"),
        ("overcap", "quote_over_cap"),
        ("too_many", "too_many_quotes"),
        ("nonexact", "nonexact_quote"),
    ],
)
def test_one_local_repair_can_replace_a_citation_only_defect(
    monkeypatch, defect, category,
):
    valid_quote = "Example raised a Series A."
    extra_quotes = [
        "The financing supports product delivery.",
        "The round was announced this year.",
    ]
    overcap_quote = "L" * 501
    source = " ".join([valid_quote, *extra_quotes, overcap_quote])
    inputs = _inputs(
        "Example raised a Series A.", source, supporting_quote=valid_quote,
    )
    calls = []

    async def judge(prompt, **kwargs):
        calls.append((prompt, kwargs))
        document = _prompt_document(prompt)
        if len(calls) == 2:
            assert kwargs["max_retries"] == 0
            assert kwargs["max_tokens"] == 800
            assert '"unit_id":0' in kwargs["system_prompt"]
            assert category in kwargs["system_prompt"]
            assert "OLD INVENTED CITATION TOKEN" not in kwargs["system_prompt"]
            assert "OLD INVENTED CITATION TOKEN" not in prompt
            assert "up to 500 characters" in kwargs["system_prompt"]
            assert "Never use ellipses, remove words, or stitch" in kwargs[
                "system_prompt"
            ]
            assert "at most 100 characters" not in kwargs["system_prompt"]
            assert kwargs["response_format"] == (
                intent_details._CITATION_REPAIR_RESPONSE_FORMAT
            )
            return json.dumps(_repair_response(
                0, [_binding(document, valid_quote)],
            ))

        binding = _binding(document, valid_quote)
        evidence = [binding]
        if defect == "missing":
            evidence = []
        elif defect == "invalid_index":
            evidence[0]["source_index"] = 999
        elif defect == "empty":
            evidence[0]["quote"] = "   "
        elif defect == "duplicate":
            evidence.append(dict(binding))
        elif defect == "overcap":
            evidence = [_binding(document, overcap_quote)]
        elif defect == "too_many":
            evidence = [_binding(document, quote) for quote in [
                valid_quote, *extra_quotes,
            ]]
        else:
            evidence[0]["quote"] = "OLD INVENTED CITATION TOKEN"
        return json.dumps(_response(
            document,
            [{
                "unit_id": 0,
                "contains_factual_claim": True,
                "status": "VERIFIED",
                "evidence": evidence,
            }],
            facts_supported=True,
        ))

    monkeypatch.setattr(verification_helpers, "openrouter_chat", judge)
    receipt = asyncio.run(intent_details.review_intent_details(*inputs))

    assert receipt["decision"] == "match"
    assert len(calls) == 2
    assert receipt["input_hash"] == (
        "sha256:" + hashlib.sha256(calls[0][0].encode("utf-8")).hexdigest()
    )
    original_document = json.loads(calls[0][0])
    repair_payload = json.loads(calls[1][0])
    assert repair_payload["review_document"] == original_document
    assert repair_payload["citation_repair_control"]["non_evidentiary"] is True
    assert repair_payload["citation_repair_control"]["units"] == [{
        "unit_id": 0,
        "contains_factual_claim": True,
        "status": "VERIFIED",
        "citation_errors": [category],
    }]
    assert len(calls[1][0]) <= intent_details._MAX_REVIEW_DOCUMENT_CHARACTERS
    assert calls[0][1]["response_format"] == intent_details._RESPONSE_FORMAT
    assert calls[1][1]["response_format"] == (
        intent_details._CITATION_REPAIR_RESPONSE_FORMAT
    )


def test_local_repair_can_supply_missing_contradiction_evidence(monkeypatch):
    quote = "Example has not launched an API."
    inputs = _inputs(
        "Example launched an API.", quote, supporting_quote=quote,
    )
    calls = 0

    async def judge(prompt, **kwargs):
        nonlocal calls
        calls += 1
        document = _prompt_document(prompt)
        evidence = [] if calls == 1 else [_binding(document, quote)]
        if calls == 2:
            assert "missing_evidence" in kwargs["system_prompt"]
            return json.dumps(_repair_response(0, evidence))
        return json.dumps(_response(
            document,
            [{
                "unit_id": 0,
                "contains_factual_claim": True,
                "status": "CONTRADICTED",
                "evidence": evidence,
            }],
            facts_supported=False,
        ))

    monkeypatch.setattr(verification_helpers, "openrouter_chat", judge)
    receipt = asyncio.run(intent_details.review_intent_details(*inputs))

    assert calls == 2
    assert receipt["decision"] == "mismatch"
    assert receipt["checks"]["facts_supported"] is False


def test_literal_citation_repair_preserves_an_unproven_unit(monkeypatch):
    financing = "Example raised a Series A."
    company_fact = "Example provides business software."
    paragraph = (
        "Example raised a Series A. Example claims an undocumented API. "
        "Example provides business software."
    )
    inputs = _inputs(paragraph, financing, supporting_quote=financing)
    calls = []
    validated_responses = []
    original_validate = intent_details._validate_review_response

    def capture_validation(response, document):
        validated_responses.append(json.loads(response))
        return original_validate(response, document)

    async def judge(prompt, **kwargs):
        calls.append((prompt, kwargs))
        document = _prompt_document(prompt)
        unit_zero = _verified_unit(document, 0, financing)
        unit_two = _verified_unit(document, 2, company_fact)
        unit_two["evidence"].append(_binding(document, "business software"))
        if len(calls) == 1:
            unit_zero["evidence"][0]["quote"] = "UNBOUND OLD CITATION"
        else:
            repair_control = json.loads(prompt)["citation_repair_control"]
            assert repair_control["units"] == [{
                "unit_id": 0,
                "contains_factual_claim": True,
                "status": "VERIFIED",
                "citation_errors": ["nonexact_quote"],
            }]
            assert "UNBOUND OLD CITATION" not in prompt
            assert "UNBOUND OLD CITATION" not in kwargs["system_prompt"]
            assert company_fact not in kwargs["system_prompt"]
            return json.dumps(_repair_response(0, unit_zero["evidence"]))
        return json.dumps(_response(
            document,
            [unit_zero, _unproven_unit(1), unit_two],
            facts_supported=False,
        ))

    monkeypatch.setattr(
        intent_details, "_validate_review_response", capture_validation,
    )
    monkeypatch.setattr(verification_helpers, "openrouter_chat", judge)
    receipt = asyncio.run(intent_details.review_intent_details(*inputs))

    assert len(calls) == 2
    assert receipt["decision"] == "mismatch"
    assert receipt["checks"]["facts_supported"] is False
    initial_units = validated_responses[0]["unit_grounding"]
    merged_units = validated_responses[1]["unit_grounding"]
    assert merged_units[1] == initial_units[1]
    assert merged_units[2] == initial_units[2]
    assert merged_units[2]["evidence"] == [
        _binding(_prompt_document(calls[0][0]), company_fact),
        _binding(_prompt_document(calls[0][0]), "business software"),
    ]


def test_unproven_bad_optional_quote_repairs_to_empty_and_terminal_mismatch(
    monkeypatch,
):
    contradiction = "Example has not launched an API."
    financing = "Example raised a Series A."
    bad_title = "SUBMITTED ALL-CAPS RESEARCH TITLE"
    inputs = _inputs(
        (
            "Example launched an API. Example raised a Series A. "
            "Example published a research report."
        ),
        f"{contradiction} {financing}",
        supporting_quote=financing,
    )
    calls = 0
    repair_payloads = []
    original_merge = intent_details._merge_citation_repairs

    def capture_merge(response, held_response, expected_unit_ids):
        repair_payloads.append(json.loads(response))
        return original_merge(response, held_response, expected_unit_ids)

    async def judge(prompt, **kwargs):
        nonlocal calls
        calls += 1
        document = _prompt_document(prompt)
        if calls == 2:
            assert (
                "For a flagged UNPROVEN or non-factual unit, return evidence:[];"
            ) in kwargs["system_prompt"]
            assert (
                "A factual VERIFIED or CONTRADICTED unit still\nrequires "
                "continuous exact bound evidence."
            ) in kwargs["system_prompt"]
            assert json.loads(prompt)["citation_repair_control"]["units"] == [{
                "unit_id": 2,
                "contains_factual_claim": True,
                "status": "UNPROVEN",
                "citation_errors": ["nonexact_quote"],
            }]
            return json.dumps(_repair_response(2, []))
        units = [
            {
                "unit_id": 0,
                "contains_factual_claim": True,
                "status": "CONTRADICTED",
                "evidence": [_binding(document, contradiction)],
            },
            _verified_unit(document, 1, financing),
            {
                "unit_id": 2,
                "contains_factual_claim": True,
                "status": "UNPROVEN",
                "evidence": [{"source_index": 0, "quote": bad_title}],
            },
        ]
        return json.dumps(_response(
            document, units, facts_supported=False,
        ))

    monkeypatch.setattr(intent_details, "_merge_citation_repairs", capture_merge)
    monkeypatch.setattr(verification_helpers, "openrouter_chat", judge)
    receipt = asyncio.run(intent_details.review_intent_details(*inputs))

    assert calls == 2
    assert repair_payloads == [_repair_response(2, [])]
    assert receipt["decision"] == "mismatch"
    assert receipt["checks"]["facts_supported"] is False


def test_unproven_bad_optional_quote_is_not_automatically_cleared(monkeypatch):
    source = "Example raised a Series A."
    bad_title = "SUBMITTED ALL-CAPS RESEARCH TITLE"
    inputs = _inputs(
        "Example published a research report.", source, supporting_quote=source,
    )
    calls = 0

    async def judge(prompt, **_kwargs):
        nonlocal calls
        calls += 1
        document = _prompt_document(prompt)
        evidence = [{"source_index": 0, "quote": bad_title}]
        if calls == 2:
            return json.dumps(_repair_response(0, evidence))
        return json.dumps(_response(
            document,
            [{
                "unit_id": 0,
                "contains_factual_claim": True,
                "status": "UNPROVEN",
                "evidence": evidence,
            }],
            facts_supported=False,
        ))

    monkeypatch.setattr(verification_helpers, "openrouter_chat", judge)
    receipt = asyncio.run(intent_details.review_intent_details(*inputs))

    assert calls == 2
    assert receipt["decision"] == "unavailable"
    assert receipt["failure_reason_code"] == "malformed_response"


@pytest.mark.parametrize(
    "change",
    [
        "upgrade_unproven", "factual_flag", "coverage", "aggregate_boolean",
        "unflagged_evidence",
    ],
)
def test_typed_citation_repair_rejects_semantic_fields_or_unflagged_units(
    monkeypatch, change,
):
    financing = "Example raised a Series A."
    company_fact = "Example provides business software."
    paragraph = (
        "Example raised a Series A. Example claims an undocumented API. "
        "Example provides business software."
    )
    inputs = _inputs(paragraph, financing, supporting_quote=financing)
    calls = 0

    async def judge(prompt, **_kwargs):
        nonlocal calls
        calls += 1
        document = _prompt_document(prompt)
        unit_zero = _verified_unit(document, 0, financing)
        unit_one = _unproven_unit(1)
        unit_two = _verified_unit(document, 2, company_fact)
        response = _response(
            document, [unit_zero, unit_one, unit_two], facts_supported=False,
        )
        if calls == 1:
            unit_zero["evidence"][0]["quote"] = "UNBOUND OLD CITATION"
            return json.dumps(response)
        repair = _repair_response(0, [_binding(document, financing)])
        if change == "upgrade_unproven":
            repair["repairs"][0]["status"] = "VERIFIED"
        elif change == "factual_flag":
            repair["repairs"][0]["contains_factual_claim"] = True
        elif change == "coverage":
            repair["signal_coverage"] = [
                {"matched_icp_signal": 0, "covered": False}
            ]
        elif change == "aggregate_boolean":
            repair["facts_supported"] = True
        else:
            repair["repairs"].append({
                "unit_id": 2,
                "evidence": unit_two["evidence"],
            })
        return json.dumps(repair)

    monkeypatch.setattr(verification_helpers, "openrouter_chat", judge)
    receipt = asyncio.run(intent_details.review_intent_details(*inputs))

    assert calls == 2
    assert receipt["decision"] == "unavailable"
    assert receipt["failure_reason_code"] == "malformed_response"


@pytest.mark.parametrize("id_defect", ["missing", "extra", "duplicate"])
def test_typed_citation_repair_requires_the_exact_flagged_unit_ids(
    monkeypatch, id_defect,
):
    quote = "Example raised a Series A and launched a product."
    inputs = _inputs(
        "Example raised a Series A. Example launched a product.",
        quote,
        supporting_quote=quote,
    )
    calls = 0

    async def judge(prompt, **_kwargs):
        nonlocal calls
        calls += 1
        document = _prompt_document(prompt)
        if calls == 1:
            units = [
                _verified_unit(document, unit_id, quote)
                for unit_id in (0, 1)
            ]
            for unit in units:
                unit["evidence"][0]["quote"] = (
                    f"UNBOUND CITATION {unit['unit_id']}"
                )
            return json.dumps(_response(
                document, units, facts_supported=True,
            ))
        binding = _binding(document, quote)
        repairs = [
            {"unit_id": 0, "evidence": [binding]},
            {"unit_id": 1, "evidence": [binding]},
        ]
        if id_defect == "missing":
            repairs.pop()
        elif id_defect == "extra":
            repairs.append({"unit_id": 2, "evidence": [binding]})
        else:
            repairs[1]["unit_id"] = 0
        return json.dumps({"repairs": repairs})

    monkeypatch.setattr(verification_helpers, "openrouter_chat", judge)
    receipt = asyncio.run(intent_details.review_intent_details(*inputs))

    assert calls == 2
    assert receipt["decision"] == "unavailable"
    assert receipt["failure_reason_code"] == "malformed_response"


def test_frozen_verified_unit_with_empty_repair_evidence_is_unavailable(
    monkeypatch,
):
    quote = "Example raised a Series A."
    inputs = _inputs(
        "Example raised a Series A.", quote, supporting_quote=quote,
    )
    calls = 0

    async def judge(prompt, **_kwargs):
        nonlocal calls
        calls += 1
        document = _prompt_document(prompt)
        unit = _verified_unit(document, 0, quote)
        if calls == 2:
            return json.dumps(_repair_response(0, []))
        unit["evidence"] = [
            {"source_index": 0, "quote": "UNBOUND OLD CITATION"}
        ]
        return json.dumps(_response(document, [unit], facts_supported=True))

    monkeypatch.setattr(verification_helpers, "openrouter_chat", judge)
    receipt = asyncio.run(intent_details.review_intent_details(*inputs))

    assert calls == 2
    assert receipt["decision"] == "unavailable"
    assert receipt["failure_reason_code"] == "malformed_response"


@pytest.mark.parametrize("second_result", ["malformed", "provider_error"])
def test_failed_local_citation_repair_remains_unavailable(
    monkeypatch, second_result,
):
    quote = "Example raised a Series A."
    inputs = _inputs(
        "Example raised a Series A.", quote, supporting_quote=quote,
    )
    calls = 0

    async def judge(prompt, **_kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            if second_result == "provider_error":
                raise RuntimeError("test provider failure")
            return "{not json"
        document = _prompt_document(prompt)
        unit = _verified_unit(document, 0, quote)
        unit["evidence"][0]["quote"] = "UNBOUND FIRST CITATION"
        return json.dumps(_response(document, [unit], facts_supported=True))

    monkeypatch.setattr(verification_helpers, "openrouter_chat", judge)
    receipt = asyncio.run(intent_details.review_intent_details(*inputs))

    assert calls == 2
    assert receipt["decision"] == "unavailable"
    assert receipt["failure_reason_code"] == (
        "provider_error" if second_result == "provider_error"
        else "malformed_response"
    )


def test_citation_repair_shares_the_original_timeout_budget(monkeypatch):
    quote = "Example raised a Series A."
    inputs = _inputs(
        "Example raised a Series A.", quote, supporting_quote=quote,
    )
    calls = 0
    timeouts = []
    original_wait_for = asyncio.wait_for

    async def observed_wait_for(awaitable, *, timeout):
        timeouts.append(timeout)
        return await original_wait_for(awaitable, timeout=timeout)

    async def judge(prompt, **_kwargs):
        nonlocal calls
        calls += 1
        document = _prompt_document(prompt)
        if calls == 1:
            await asyncio.sleep(0.01)
            unit = _verified_unit(document, 0, quote)
            unit["evidence"][0]["quote"] = "UNBOUND CITATION"
        else:
            return json.dumps(_repair_response(
                0, [_binding(document, quote)],
            ))
        return json.dumps(_response(document, [unit], facts_supported=True))

    monkeypatch.setattr(intent_details.asyncio, "wait_for", observed_wait_for)
    monkeypatch.setattr(verification_helpers, "openrouter_chat", judge)
    receipt = asyncio.run(intent_details.review_intent_details(*inputs))

    assert receipt["decision"] == "match"
    assert calls == 2
    assert len(timeouts) == 2
    assert 0 < timeouts[1] < timeouts[0] <= intent_details.REVIEW_TIMEOUT_SECONDS


def test_citation_repair_timeout_remains_provider_unavailable(monkeypatch):
    quote = "Example raised a Series A."
    inputs = _inputs(
        "Example raised a Series A.", quote, supporting_quote=quote,
    )
    calls = 0
    waits = 0

    async def controlled_wait_for(awaitable, *, timeout):
        nonlocal waits
        waits += 1
        if waits == 2:
            awaitable.close()
            raise asyncio.TimeoutError
        return await awaitable

    async def judge(prompt, **_kwargs):
        nonlocal calls
        calls += 1
        document = _prompt_document(prompt)
        unit = _verified_unit(document, 0, quote)
        unit["evidence"][0]["quote"] = "UNBOUND CITATION"
        return json.dumps(_response(document, [unit], facts_supported=True))

    monkeypatch.setattr(intent_details.asyncio, "wait_for", controlled_wait_for)
    monkeypatch.setattr(verification_helpers, "openrouter_chat", judge)
    receipt = asyncio.run(intent_details.review_intent_details(*inputs))

    assert waits == 2
    assert calls == 1
    assert receipt["decision"] == "unavailable"
    assert receipt["failure_reason_code"] == "provider_error"


def test_repair_metadata_preserves_evidence_and_the_existing_input_bound():
    admitted = [{
        "source_index": 0,
        "evidence_kind": "verified_source_context",
        "admitted_text": ["Example raised a Series A."],
    }]
    document = {"admitted_evidence": admitted}
    held_response = {
        "unit_grounding": [{
            "unit_id": 0,
            "contains_factual_claim": True,
            "status": "VERIFIED",
            "evidence": [{
                "source_index": 0,
                "quote": "UNBOUND OLD CITATION",
            }],
        }],
        "signal_coverage": [{"matched_icp_signal": 0, "covered": True}],
        **{name: True for name in intent_details._CHECKS},
    }
    issues = {0: ("nonexact_quote",)}

    prompt = intent_details._citation_repair_user_prompt(
        document, issues, held_response,
    )
    payload = json.loads(prompt)

    assert len(prompt) <= intent_details._MAX_REVIEW_DOCUMENT_CHARACTERS
    assert payload["review_document"]["admitted_evidence"] == admitted
    assert payload["citation_repair_control"]["units"] == [{
        "unit_id": 0,
        "contains_factual_claim": True,
        "status": "VERIFIED",
        "citation_errors": ["nonexact_quote"],
    }]
    assert "UNBOUND OLD CITATION" not in prompt

    oversized = {**document, "padding": "x" * 48_000}
    with pytest.raises(ValueError, match="exceeds its input bound"):
        intent_details._citation_repair_user_prompt(
            oversized, issues, held_response,
        )
    assert oversized["admitted_evidence"] == admitted


@pytest.mark.parametrize(
    "defect",
    [
        "invalid_json", "incomplete_units", "duplicate_units", "invalid_status",
        "aggregate_conflict", "invalid_coverage", "invalid_boolean",
        "nonscalar_index",
    ],
)
def test_non_citation_contract_defects_do_not_trigger_local_repair(
    monkeypatch, defect,
):
    paragraph = "Example raised a Series A. Example launched a product."
    quote = "Example raised a Series A and launched a product."
    inputs = _inputs(paragraph, quote, supporting_quote=quote)
    calls = 0

    async def judge(prompt, **_kwargs):
        nonlocal calls
        calls += 1
        if defect == "invalid_json":
            return "{not json"
        document = _prompt_document(prompt)
        units = [
            _verified_unit(document, unit["unit_id"], quote)
            for unit in document["intent_details_units"]
        ]
        response = _response(document, units, facts_supported=True)
        if defect == "incomplete_units":
            response["unit_grounding"].pop()
        elif defect == "duplicate_units":
            response["unit_grounding"][-1]["unit_id"] = 0
        elif defect == "invalid_status":
            response["unit_grounding"][0]["status"] = "MAYBE"
        elif defect == "aggregate_conflict":
            response["facts_supported"] = False
        elif defect == "invalid_coverage":
            response["signal_coverage"] = []
        elif defect == "nonscalar_index":
            response["unit_grounding"][0]["evidence"][0]["source_index"] = []
        else:
            response["natural_paragraph"] = "true"
        return json.dumps(response)

    monkeypatch.setattr(verification_helpers, "openrouter_chat", judge)
    receipt = asyncio.run(intent_details.review_intent_details(*inputs))

    assert calls == 1
    assert receipt["decision"] == "unavailable"
    assert receipt["failure_reason_code"] == "malformed_response"


def test_provider_failure_does_not_trigger_local_repair(monkeypatch):
    inputs = _inputs(
        "Example raised a Series A.",
        "Example raised a Series A.",
        supporting_quote="Example raised a Series A.",
    )
    calls = 0

    async def judge(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        raise RuntimeError("test provider failure")

    monkeypatch.setattr(verification_helpers, "openrouter_chat", judge)
    receipt = asyncio.run(intent_details.review_intent_details(*inputs))

    assert calls == 1
    assert receipt["decision"] == "unavailable"
    assert receipt["failure_reason_code"] == "provider_error"


def test_citation_failure_with_coverage_aggregate_conflict_does_not_repair(
    monkeypatch,
):
    quote = "Example raised a Series A."
    inputs = _inputs(
        "Example raised a Series A.", quote, supporting_quote=quote,
    )
    calls = 0

    async def judge(prompt, **_kwargs):
        nonlocal calls
        calls += 1
        document = _prompt_document(prompt)
        unit = _verified_unit(document, 0, quote)
        unit["evidence"][0]["quote"] = "UNBOUND CITATION"
        response = _response(document, [unit], facts_supported=True)
        response["verified_signals_covered"] = False
        return json.dumps(response)

    monkeypatch.setattr(verification_helpers, "openrouter_chat", judge)
    receipt = asyncio.run(intent_details.review_intent_details(*inputs))

    assert calls == 1
    assert receipt["decision"] == "unavailable"
    assert receipt["failure_reason_code"] == "malformed_response"


def test_valid_semantic_mismatch_does_not_trigger_local_repair(monkeypatch):
    quote = "Example raised a Series A."
    inputs = _inputs(
        "Example claims an undocumented API capability.",
        quote,
        supporting_quote=quote,
    )
    calls = 0

    async def judge(prompt, **_kwargs):
        nonlocal calls
        calls += 1
        document = _prompt_document(prompt)
        return json.dumps(_response(
            document, [_unproven_unit(0)], facts_supported=False,
        ))

    monkeypatch.setattr(verification_helpers, "openrouter_chat", judge)
    receipt = asyncio.run(intent_details.review_intent_details(*inputs))

    assert calls == 1
    assert receipt["decision"] == "mismatch"


def test_two_unbound_positive_citations_are_never_accepted(monkeypatch):
    quote = "Example raised a Series A."
    inputs = _inputs(
        "Example raised a Series A.", quote, supporting_quote=quote,
    )
    calls = 0

    async def judge(prompt, **_kwargs):
        nonlocal calls
        calls += 1
        document = _prompt_document(prompt)
        unit = _verified_unit(document, 0, quote)
        unit["evidence"][0]["quote"] = f"UNBOUND CITATION {calls}"
        if calls == 2:
            return json.dumps(_repair_response(0, unit["evidence"]))
        return json.dumps(_response(document, [unit], facts_supported=True))

    monkeypatch.setattr(verification_helpers, "openrouter_chat", judge)
    receipt = asyncio.run(intent_details.review_intent_details(*inputs))

    assert calls == 2
    assert receipt["decision"] == "unavailable"
    assert receipt["failure_reason_code"] == "malformed_response"


def test_flat_boolean_cannot_override_an_unproven_unit(monkeypatch):
    paragraph = "Example claims an undocumented API capability."
    quote = "Example raised a Series A."
    inputs = _inputs(paragraph, quote, supporting_quote=quote)

    def response(document):
        return _response(
            document,
            [_unproven_unit(0)],
            facts_supported=True,
        )

    receipt = _review(monkeypatch, inputs, response)
    assert receipt["decision"] == "unavailable"
    assert receipt["failure_reason_code"] == "malformed_response"


def test_six_unit_two_binding_response_fits_existing_output_bound():
    natural_quote_one = (
        "Common Wealth added employers through its retirement platform every "
        "business day in Canada."
    )[:100]
    natural_quote_two = (
        "TypeSafe evaluates application state and returns structured results "
        "through its documented API."
    )[:100]
    response = {
        "unit_grounding": [
            {
                "unit_id": unit_id,
                "contains_factual_claim": True,
                "status": "VERIFIED",
                "evidence": [
                    {"source_index": 0, "quote": natural_quote_one},
                    {"source_index": 1, "quote": natural_quote_two},
                ],
            }
            for unit_id in range(intent_details._MAX_STATEMENT_UNITS)
        ],
        "signal_coverage": [{"matched_icp_signal": 0, "covered": True}],
        **{name: True for name in intent_details._CHECKS},
    }
    compact = json.dumps(response, separators=(",", ":"), ensure_ascii=False)

    # This is a realistic upper-shape check, not a provider-tokenizer proof.
    assert len(compact.encode("utf-8")) < 2_400
