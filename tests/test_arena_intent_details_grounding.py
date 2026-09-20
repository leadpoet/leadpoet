"""The client paragraph is grounded in saved source evidence, not model prose."""

import asyncio
from copy import deepcopy
import hashlib
import json
from types import SimpleNamespace

import pytest

from qualification.intent_details import validate_intent_details_text
from qualification.scoring import intent_details, verification_helpers, lead_scorer
from qualification.scoring.company_fit_decision import company_fit_match
from qualification.scoring.competition import scorer_breakdown_has_retryable_infrastructure_failure


PARAGRAPH = (
    "Acme launched a reporting platform on September 1, 2026, which could expand "
    "the workflows its customers manage. It also opened a Berlin office on "
    "September 3, which may support regional delivery. Together, these activities "
    "make Acme relevant to the ICP for reporting software companies expanding "
    "their product and geographic reach."
)
FISERV_PARAGRAPH = (
    "On October 29, 2025, Fiserv announced several executive changes, including "
    "Paul Todd’s appointment as Chief Financial Officer effective October 31 and "
    "the appointment of Takis Georgakopoulos and Dhivya Suryadevara as "
    "Co-Presidents effective December 1. These changes placed new leadership over "
    "finance and major operating areas including Financial Solutions, Sales and "
    "Operations at a provider of account processing, digital banking, card "
    "processing, and payment infrastructure; the transition may create a timely "
    "need to coordinate and optimize the core banking and money-movement workflows "
    "Fiserv delivers."
)
FISERV_REVIEW = {
    "facts_supported": True,
    "connects_icp": True,
    "natural_paragraph": True,
    "relevance_grounded": True,
    "signal_coverage": [{
        "matched_icp_signal": 0,
        "covered": True,
    }],
    "verified_signals_covered": True,
}

INTEGRATED_CONNECTION_PARAGRAPH = (
    "Relevant to the ICP's focus on reporting software expansion, Acme launched "
    "a reporting platform on September 1, 2026, which could expand the workflows "
    "its customers manage. It also opened a Berlin office on September 3, which "
    "may support regional delivery."
)
NO_CONNECTION_PARAGRAPH = (
    "Acme launched a reporting platform on September 1, 2026. It also opened a "
    "Berlin office on September 3, 2026."
)


def inputs():
    signals = [
        SimpleNamespace(matched_icp_signal=0, description="Acme launched its reporting platform.",
                        date="2026-01-01", url="https://acme.example/platform"),
        SimpleNamespace(matched_icp_signal=1, description="Acme opened a Berlin office.",
                        date="2026-01-01", url="https://acme.example/berlin"),
    ]
    company = SimpleNamespace(company_name="Acme", company_website="https://acme.example/",
                              intent_signals=signals, intent_details=PARAGRAPH)
    icp = SimpleNamespace(prompt="Find reporting software companies with product and geographic expansion.",
                          product_service="Reporting software", intent_signals=["Product launch", "New office"])
    results = []
    for index, (signal, quote, date) in enumerate(zip(signals, [
        "Acme launched its reporting platform on September 1, 2026.",
        "Acme opened its Berlin office on September 3, 2026.",
    ], ["2026-09-01", "2026-09-03"])):
        results.append({
            "raw": 50.0, "after_decay": 50.0, "matched_icp_signal": index,
            "evidence_urls": [signal.url],
            "judge_verdict": {
                "decision": "verified", "client_ready": True,
                "authoritative_date": date, "authoritative_date_basis": "event",
                "verification_trace": {"intent_verdict": {"signal_evaluations": [{
                    "signal_status": "supported", "same_entity_check": "pass",
                    "supporting_quotes": [quote], "evidence_urls_used": [signal.url],
                }]}},
            },
        })
    fit = {"gate": "company_fit", "decision": "match", "dimension_evidence": {
        "industry": {"decision": "match", "web_evidence": {
            "url": "https://acme.example/", "quote": "Acme provides reporting software."}}
    }}
    return company, icp, results, fit


@pytest.mark.parametrize("value", [None, "", "  ", "One paragraph.\n\nAnother paragraph.",
                                  "- A signal\n- Another signal", "```prose```", "x" * 2001,
                                  "Hidden\u200bwords", "Bad\x00text"])
def test_paragraph_shape_rejects_non_prose(value):
    with pytest.raises(ValueError):
        validate_intent_details_text(value)


def test_paragraph_preserves_words_and_normalizes_line_wrapping():
    assert validate_intent_details_text("  An event occurred.\nIts relevance is clear.  ") == (
        "An event occurred. Its relevance is clear."
    )
    assert validate_intent_details_text(PARAGRAPH) == PARAGRAPH


@pytest.mark.parametrize("model,expected", [
    ("gpt-4o-mini", "openai/gpt-4o-mini"),
    ("anthropic/claude-sonnet-4.5", "anthropic/claude-sonnet-4.5"),
])
def test_review_transport_preserves_legacy_and_pinned_model_names(monkeypatch, model, expected):
    class Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def post(self, url, *, json, **kwargs):
            assert json["model"] == expected
            return SimpleNamespace(raise_for_status=lambda: None,
                                   json=lambda: {"choices": [{"message": {"content": "{}"}}]})

    monkeypatch.setattr(verification_helpers.httpx, "AsyncClient", Client)
    assert asyncio.run(verification_helpers.openrouter_chat(
        "review", model=model, api_key="test-only-placeholder", max_retries=0,
    )) == "{}"


def test_review_uses_authoritative_dates_and_excludes_failed_signals():
    company, icp, results, fit = inputs()
    failed = deepcopy(results[1])
    failed.update(after_decay=0, matched_icp_signal=2)
    failed["judge_verdict"]["verification_trace"]["intent_verdict"]["signal_evaluations"][0]["supporting_quotes"] = ["UNVERIFIED CLAIM"]
    result = intent_details.review_evidence(company, icp, results + [failed], fit)
    assert len(result["verified_signals"]) == 2
    assert [row["authoritative_date"] for row in result["verified_signals"]] == ["2026-09-01", "2026-09-03"]
    assert "2026-01-01" not in json.dumps(result)
    assert "UNVERIFIED CLAIM" not in json.dumps(result)
    assert result["verified_company_evidence"]["industry"]["quote"] == "Acme provides reporting software."


def test_review_keeps_fetched_context_missing_from_selected_signal_quotes():
    company, icp, results, fit = inputs()
    source = {
        "url": company.intent_signals[0].url,
        "text": "September 1, 2026. Acme launched its reporting platform today. The platform integrates reporting and analytics.",
        "source_publication_date": "2026-09-01",
    }
    trace = results[0]["judge_verdict"]["verification_trace"]
    trace["verified_source_context"] = [source, {
        "url": "https://unrelated.example/", "text": "UNRELATED SOURCE",
    }]
    result = intent_details.review_evidence(company, icp, results, fit)
    assert result["verified_signals"][0]["source_context"] == [source]
    assert "UNRELATED SOURCE" not in json.dumps(result)
    assert "platform integrates" not in str(result["verified_signals"][0]["supporting_quotes"])


def test_source_context_preserves_review_size_and_utf8_bounds():
    company, icp, results, fit = inputs()
    for result, signal in zip(results, company.intent_signals):
        result["judge_verdict"]["verification_trace"]["verified_source_context"] = [{
            "url": signal.url, "text": "é" * 20_000,
        }]
    document = intent_details.review_evidence(company, icp, results, fit)
    contexts = [c for v in document["verified_signals"] for c in v["source_context"]]
    assert sum(len(c["text"].encode("utf-8")) for c in contexts) <= 12_000
    assert all(len(c["text"].encode("utf-8")) <= 6_000 for c in contexts)
    assert len(json.dumps(document, ensure_ascii=False)) <= 48_000
    assert json.loads(json.dumps(document)) == document


@pytest.mark.parametrize("count", [2, 3])
def test_early_multi_source_signal_cannot_starve_later_context(count):
    company, icp, results, fit = inputs()
    if count == 3:
        results.append(deepcopy(results[1]))
        results[-1]["matched_icp_signal"] = 2
    for result in results:
        url = result["evidence_urls"][0]
        result["judge_verdict"]["verification_trace"]["verified_source_context"] = [
            {"url": url, "text": "é" * 3_000},
            {"url": url, "text": "extra" * 2_000},
        ]
    document = intent_details.review_evidence(company, icp, results, fit)
    sizes = [sum(len(c["text"].encode("utf-8")) for c in v["source_context"])
             for v in document["verified_signals"]]
    assert sizes == [12_000 // count] * count
    assert sum(sizes) == 12_000


@pytest.mark.parametrize("missing", ["supporting_quotes", "same_entity_check"])
def test_claims_cannot_replace_missing_source_support(missing):
    company, icp, results, fit = inputs()
    del results[0]["judge_verdict"]["verification_trace"]["intent_verdict"]["signal_evaluations"][0][missing]
    with pytest.raises(ValueError, match="source quotes"):
        intent_details.review_evidence(company, icp, results, fit)


@pytest.mark.parametrize("failed_check", [None, *intent_details._CHECKS])
def test_all_grounding_and_writing_checks_must_pass(monkeypatch, failed_check):
    company, icp, results, fit = inputs()
    checks = {name: name != failed_check for name in intent_details._CHECKS}

    async def judge(prompt, **kwargs):
        document = json.loads(prompt)
        assert document["intent_details"] == PARAGRAPH
        assert len(document["verified_signals"]) == 2
        assert "untrusted JSON data" in kwargs["system_prompt"]
        assert kwargs["max_retries"] == 0
        assert kwargs["model"] == "anthropic/claude-sonnet-4.5"
        return json.dumps({**checks, "signal_coverage": [
            {"matched_icp_signal": index, "covered": True}
            for index in (0, 1)
        ]})

    monkeypatch.setattr(verification_helpers, "openrouter_chat", judge)
    result = asyncio.run(intent_details.review_intent_details(company, icp, results, fit))
    assert result["decision"] == ("match" if failed_check is None else "mismatch")
    assert result["checks"] == checks
    assert result["input_hash"].startswith("sha256:")


def test_review_assesses_original_paragraph_without_requiring_a_copy(monkeypatch):
    company, icp, results, fit = inputs()
    company.intent_details = FISERV_PARAGRAPH
    prompts = []

    async def judge(prompt, **kwargs):
        schema = kwargs["response_format"]["json_schema"]["schema"]
        coverage_schema = schema["properties"]["signal_coverage"]["items"]
        assert coverage_schema["properties"]["covered"] == {"type": "boolean"}
        assert "paragraph_quote" not in coverage_schema["properties"]
        assert "Read the original paragraph directly" in kwargs["system_prompt"]
        prompts.append(prompt)
        return json.dumps(FISERV_REVIEW)

    monkeypatch.setattr(verification_helpers, "openrouter_chat", judge)
    receipt = asyncio.run(intent_details.review_intent_details(
        company, icp, results[:1], fit
    ))

    assert receipt["decision"] == "match"
    assert receipt["checks"] == {
        name: True for name in intent_details._CHECKS
    }
    assert json.loads(prompts[0])["intent_details"] == FISERV_PARAGRAPH
    assert "Paul Todd’s" in prompts[0]
    assert receipt["input_hash"] == (
        "sha256:" + hashlib.sha256(prompts[0].encode("utf-8")).hexdigest()
    )


def test_grounded_icp_connection_does_not_require_a_separate_final_sentence(
    monkeypatch,
):
    company, icp, results, fit = inputs()
    company.intent_details = INTEGRATED_CONNECTION_PARAGRAPH

    async def judge(*_args, **_kwargs):
        return json.dumps({
            **{name: True for name in intent_details._CHECKS},
            "signal_coverage": [
                {
                    "matched_icp_signal": 0,
                    "covered": True,
                },
                {
                    "matched_icp_signal": 1,
                    "covered": True,
                },
            ],
        })

    monkeypatch.setattr(verification_helpers, "openrouter_chat", judge)
    receipt = asyncio.run(intent_details.review_intent_details(
        company, icp, results, fit
    ))

    assert receipt["decision"] == "match"
    assert receipt["contract_id"] == "intent-details:v2"
    assert receipt["checks"]["connects_icp"] is True


@pytest.mark.parametrize(
    ("paragraph", "failed_check", "expected_true"),
    [
        (
            PARAGRAPH,
            "facts_supported",
            {
                "verified_signals_covered", "relevance_grounded",
                "connects_icp", "natural_paragraph",
            },
        ),
        (
            NO_CONNECTION_PARAGRAPH,
            "connects_icp",
            {
                "facts_supported", "verified_signals_covered",
                "relevance_grounded", "natural_paragraph",
            },
        ),
    ],
)
def test_review_projects_factual_and_icp_connection_checks_independently(
    monkeypatch, paragraph, failed_check, expected_true
):
    company, icp, results, fit = inputs()
    company.intent_details = paragraph
    checks = {name: name != failed_check for name in intent_details._CHECKS}

    async def judge(*_args, **_kwargs):
        return json.dumps({
            **checks,
            "signal_coverage": [
                {"matched_icp_signal": index, "covered": True}
                for index in (0, 1)
            ],
        })

    monkeypatch.setattr(verification_helpers, "openrouter_chat", judge)
    receipt = asyncio.run(intent_details.review_intent_details(
        company, icp, results, fit
    ))

    assert receipt["decision"] == "mismatch"
    assert receipt["checks"][failed_check] is False
    assert {name for name, passed in receipt["checks"].items() if passed} == expected_true


@pytest.mark.parametrize("response", ["{}", "not json", "[]", json.dumps({name: "true" for name in intent_details._CHECKS})])
def test_malformed_review_is_retryable_not_a_terminal_zero(monkeypatch, response):
    async def judge(*args, **kwargs):
        return response
    monkeypatch.setattr(verification_helpers, "openrouter_chat", judge)
    receipt = asyncio.run(intent_details.review_intent_details(*inputs()))
    assert receipt["decision"] == "unavailable"
    assert receipt["failure_reason_code"] == "malformed_response"
    assert scorer_breakdown_has_retryable_infrastructure_failure({"verifier_gate_receipts": [receipt]})


def test_provider_error_retains_retry_without_leaking_exception(monkeypatch):
    async def judge(*args, **kwargs):
        raise RuntimeError("private provider diagnostic")
    monkeypatch.setattr(verification_helpers, "openrouter_chat", judge)
    receipt = asyncio.run(intent_details.review_intent_details(*inputs()))
    assert receipt["decision"] == "unavailable"
    assert receipt["failure_reason_code"] == "provider_error"
    assert "private provider diagnostic" not in json.dumps(receipt)


@pytest.mark.parametrize("coverage", [
    [{"matched_icp_signal": 0, "covered": True}],
    [{"matched_icp_signal": 0, "covered": True},
     {"matched_icp_signal": 2, "covered": True}],
    [{"matched_icp_signal": 0, "covered": True},
     {"matched_icp_signal": 0, "covered": True}],
    [{"matched_icp_signal": 0, "covered": True},
     {"matched_icp_signal": True, "covered": True}],
    [{"matched_icp_signal": 0, "covered": True},
     {"matched_icp_signal": 1, "covered": "true"}],
    [{"matched_icp_signal": 0, "covered": True},
     {"matched_icp_signal": 1, "covered": 1}],
    [{"matched_icp_signal": 0, "covered": True},
     {"matched_icp_signal": 1, "paragraph_quote": PARAGRAPH}],
])
def test_incomplete_or_malformed_coverage_is_retryable(monkeypatch, coverage):
    async def judge(*args, **kwargs):
        return json.dumps({**{name: True for name in intent_details._CHECKS},
                           "signal_coverage": coverage})
    monkeypatch.setattr(verification_helpers, "openrouter_chat", judge)
    receipt = asyncio.run(intent_details.review_intent_details(*inputs()))
    assert receipt["decision"] == "unavailable"
    assert receipt["failure_reason_code"] == "malformed_response"
    assert scorer_breakdown_has_retryable_infrastructure_failure({
        "verifier_gate_receipts": [receipt],
    })


def test_a_missing_activity_remains_a_terminal_mismatch(monkeypatch):
    async def judge(*args, **kwargs):
        return json.dumps({**{name: True for name in intent_details._CHECKS},
                           "signal_coverage": [
                               {"matched_icp_signal": 0, "covered": True},
                               {"matched_icp_signal": 1, "covered": False},
                           ]})
    monkeypatch.setattr(verification_helpers, "openrouter_chat", judge)
    receipt = asyncio.run(intent_details.review_intent_details(*inputs()))
    assert receipt["decision"] == "mismatch"
    assert receipt["checks"]["verified_signals_covered"] is False
    assert not scorer_breakdown_has_retryable_infrastructure_failure({
        "verifier_gate_receipts": [receipt],
    })


@pytest.mark.parametrize("decision", ["match", "mismatch", "unavailable"])
def test_scorer_checks_paragraph_after_signals_and_preserves_arithmetic(monkeypatch, decision):
    company, icp, results, fit = inputs()

    async def verify_company(*args, **kwargs):
        return company_fit_match("verified", details={"dimension_evidence": fit["dimension_evidence"]})

    async def verify_signals(*args, **kwargs):
        return 100.0, 100.0, 1.0, 100, False, results

    async def review(*args):
        assert args[2] is results
        return {"gate": "intent_details", "decision": decision,
                **({"failure_class": "intent_details_provider_unavailable"} if decision == "unavailable" else {})}

    monkeypatch.setattr(lead_scorer, "_verify_company_fit", verify_company)
    monkeypatch.setattr(lead_scorer, "score_company_competition_intent_signal", verify_signals)
    monkeypatch.setattr(intent_details, "review_intent_details", review)
    result = asyncio.run(lead_scorer.score_company_competition_intent(
        company, icp, 0, 0, set(), integrity_policy=True,
    )).model_dump(mode="json")
    assert result["final_score"] == (100.0 if decision == "match" else 0.0)
    assert result["verifier_gate_receipts"][-1]["decision"] == decision
    assert scorer_breakdown_has_retryable_infrastructure_failure(result, integrity_policy=True) == (decision == "unavailable")


def test_unverified_primary_does_not_spend_on_prose(monkeypatch):
    company, icp, results, fit = inputs()

    async def verify_company(*args, **kwargs):
        return company_fit_match("verified")

    async def verify_signals(*args, **kwargs):
        return 50.0, 0.0, 1.0, 100, False, results

    async def review(*args):
        raise AssertionError("should not review prose for an unqualified company")

    monkeypatch.setattr(lead_scorer, "_verify_company_fit", verify_company)
    monkeypatch.setattr(lead_scorer, "score_company_competition_intent_signal", verify_signals)
    monkeypatch.setattr(intent_details, "review_intent_details", review)
    result = asyncio.run(lead_scorer.score_company_competition_intent(company, icp, 0, 0, set(), integrity_policy=True))
    assert result.final_score == 0
