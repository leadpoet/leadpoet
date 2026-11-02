"""Keep paragraph evidence inside the unchanged native request contract.

The CACI inputs reconstruct retained evidence. The rejected original request
was not persisted, so this fixture does not claim byte-exact request replay.
"""
import asyncio
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from gateway.qualification.models import CompanyOutput, ICPPrompt
from lab_arena import operations
from qualification.scoring import intent_details as review, verification_helpers
from qualification.scoring.competition import _normalized_company, _normalized_icp


FIXTURES = Path(__file__).with_name("fixtures")


def compact(document):
    return json.dumps(document, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def caci():
    saved = json.loads((FIXTURES / "intent_details_caci_budget_reconstruction.json").read_text())
    return (CompanyOutput(**saved["company"]), ICPPrompt(**saved["icp"]),
            saved["signal_results"], saved["company_fit_receipt"]), saved["company_source_contexts"]


def healthy():
    saved = json.loads((Path(__file__).parent / "lab_arena/fixtures/oct01_koho_optional_bonus.json").read_text())
    return (CompanyOutput(**_normalized_company(saved["company"], integrity_policy=True, company_quality=True)),
            ICPPrompt(**_normalized_icp(saved["icp"])), saved["signal_results"], saved["company_fit"])


def native_validate(prompt, *, system=None, response_format=None, full_thinking=False):
    return operations.validate_operation_request("openrouter.chat", {
        "model": review.REVIEW_MODEL,
        "messages": [{"role": "system", "content": system or review._SYSTEM},
                     {"role": "user", "content": prompt}],
        "max_tokens": 800,
        "response_format": response_format or review._RESPONSE_FORMAT,
    }, allow_full_thinking=full_thinking)


@pytest.mark.parametrize("full_thinking", [False, True])
def test_retained_caci_shape_fits_native_request_without_dropping_mandatory_evidence(monkeypatch, full_thinking):
    monkeypatch.setenv("ARENA_INDUSTRY_EVIDENCE_MODEL", "reviewer")
    inputs, contexts = caci()
    document = review.review_evidence(*inputs, company_source_contexts=contexts)
    prompt = compact(document)
    assert len(prompt) <= 32_000
    native_validate(prompt, full_thinking=full_thinking)
    text = "\n".join(t for source in document["admitted_evidence"] for t in source.get("admitted_text", []))
    industry = inputs[3]["dimension_evidence"]["industry"]["web_evidence"]
    for pair in [industry, *industry["supporting_evidence"]]:
        assert pair["quote"] in text
    for result in inputs[2]:
        verdict = result["judge_verdict"]
        assert verdict["authoritative_date"] in prompt
        for evaluation in verdict["verification_trace"]["intent_verdict"]["signal_evaluations"]:
            for quote in evaluation["supporting_quotes"]:
                assert quote in text
    assert document["company"]["name"] == inputs[0].company_name
    assert document["icp"]["prompt"] == inputs[1].prompt
    assert document["intent_details_units"] == review._statement_units(inputs[0].intent_details)


def test_healthy_document_and_input_hash_are_byte_identical():
    prompt = compact(review.review_evidence(*healthy()))
    assert hashlib.sha256(prompt.encode()).hexdigest() == "41bcafacf4c3d02e6b667988bcdbdf1a3a80a3d67a6bcc51d4b6854c83b6b59a"


def test_rebuild_is_from_unchanged_raw_observations_and_keeps_fair_context(monkeypatch):
    monkeypatch.setenv("ARENA_INDUSTRY_EVIDENCE_MODEL", "reviewer")
    inputs, contexts = caci()
    snapshot = deepcopy((inputs, contexts))
    budgets = []
    builder = review._review_evidence_with_context_budget

    def assemble(*args, **kwargs):
        budgets.append(kwargs["context_budget"])
        return builder(*args, **kwargs)

    monkeypatch.setattr(review, "_review_evidence_with_context_budget", assemble)
    document = review.review_evidence(*inputs, company_source_contexts=contexts)
    assert (inputs, contexts) == snapshot
    assert budgets[0] == 12_000 and len(budgets) > 1
    assert all(b < a for a, b in zip(budgets, budgets[1:]))
    lengths = [sum(len(t.encode()) for t in source.get("admitted_text", []))
               for source in document["admitted_evidence"]
               if source["evidence_kind"] == "verified_source_context"]
    assert len(lengths) == 2 and min(lengths) > 2_000
    assert abs(lengths[0] - lengths[1]) < 100
    # Every reference still targets the final, once-selected indexed document.
    for item in [*document["verified_signals"], *document["verified_company_evidence"].values()]:
        assert all(0 <= index < len(document["admitted_evidence"])
                   for index in item.get("evidence_source_indexes", []))


def test_unmodified_first_trial_matches_retained_before_fix_document(monkeypatch):
    monkeypatch.setenv("ARENA_INDUSTRY_EVIDENCE_MODEL", "reviewer")
    inputs, contexts = caci()
    document = review._review_evidence_with_context_budget(
        *inputs, company_source_contexts=contexts,
        authenticated_provider_observation=None, context_budget=12_000,
    )
    prompt = compact(document)
    assert len(prompt) == 34_681
    assert hashlib.sha256(prompt.encode()).hexdigest() == "f9f549e3d487cbe63b90b6d23459fc7f0e8c5fdd53d07272078e0db2b34f3c3a"
    for full_thinking in (False, True):
        with pytest.raises(operations.OperationRequestError, match="invalid_field"):
            native_validate(prompt, full_thinking=full_thinking)


def test_small_shared_budget_preserves_company_and_each_intent_share(monkeypatch):
    monkeypatch.setenv("ARENA_INDUSTRY_EVIDENCE_MODEL", "reviewer")
    inputs, contexts = caci()
    document = review._review_evidence_with_context_budget(
        *inputs, company_source_contexts=contexts,
        authenticated_provider_observation=None, context_budget=3_000,
    )
    intent = [row for row in document["admitted_evidence"]
              if row["evidence_kind"] == "verified_source_context"]
    assert {row["matched_icp_signal"] for row in intent} == {0, 1}
    assert all(800 < sum(len(t.encode()) for t in row["admitted_text"]) <= 1_000
               for row in intent)
    company_context = [row for row in document["admitted_evidence"]
                       if row["evidence_kind"] == "verified_company_source_context"]
    assert company_context
    assert 0 < sum(len(t.encode()) for row in company_context for t in row["admitted_text"]) <= 1_050


def test_rebuild_count_is_bounded_even_when_escaping_gives_no_size_progress(monkeypatch):
    inputs = healthy()
    builder = review._review_evidence_with_context_budget
    budgets = []

    def stalled(*args, **kwargs):
        budgets.append(kwargs["context_budget"])
        document = builder(*args, **kwargs)
        if kwargs["context_budget"]:
            document["test_optional_context"] = ""
            document["test_optional_context"] = "x" * (32_001 - len(compact(document)))
        return document

    monkeypatch.setattr(review, "_review_evidence_with_context_budget", stalled)
    assert len(compact(review.review_evidence(*inputs))) <= 32_000
    assert budgets[0] == 12_000 and budgets[-1] == 0
    assert len(budgets) == 48


@pytest.mark.parametrize("text", ["ordinary optional context. ", 'é中😀\\"\n optional context. '])
@pytest.mark.parametrize("full_thinking", [False, True])
def test_large_multisource_context_uses_serialized_characters_and_preserves_quotes(monkeypatch, text, full_thinking):
    monkeypatch.setenv("ARENA_INDUSTRY_EVIDENCE_MODEL", "reviewer")
    inputs, contexts = caci()
    for signal in inputs[2]:
        for context in signal["judge_verdict"]["verification_trace"]["verified_source_context"]:
            context["text"] = text * 500 + context["text"] + text * 500
            context["source_publication_date"] = "2026-01-29"
    document = review.review_evidence(*inputs, company_source_contexts=contexts)
    prompt = compact(document)
    assert len(prompt) <= 32_000
    native_validate(prompt, full_thinking=full_thinking)
    texts = "\n".join(t for row in document["admitted_evidence"] for t in row.get("admitted_text", []))
    for signal in inputs[2]:
        for evaluation in signal["judge_verdict"]["verification_trace"]["intent_verdict"]["signal_evaluations"]:
            for quote in evaluation["supporting_quotes"]:
                assert quote in texts
    assert "2026-01-29" in prompt


def test_context_free_floor_retains_observed_source_dates(monkeypatch):
    # Date metadata is authoritative even when optional surrounding prose has
    # no remaining allowance. It must not disappear with its context window.
    inputs = healthy()
    for signal in inputs[2]:
        trace = signal["judge_verdict"]["verification_trace"]
        for context in trace.get("verified_source_context", []):
            context["source_publication_date"] = "2026-01-29"
    document = review._review_evidence_with_context_budget(
        *inputs, company_source_contexts=None,
        authenticated_provider_observation=None, context_budget=0,
    )
    assert "2026-01-29" in compact(document)


def test_mandatory_oversize_is_local_without_provider_dispatch(monkeypatch):
    inputs = healthy()
    inputs[1].prompt = "Mandatory criterion " * 2_000
    calls = []

    async def judge(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("oversized evidence must not reach a provider")

    monkeypatch.setattr(verification_helpers, "openrouter_chat", judge)
    receipt = asyncio.run(review.review_intent_details(*inputs))
    assert calls == []
    assert receipt["decision"] == "unavailable"
    assert receipt["failure_class"] == "intent_details_evidence_unavailable"
    assert receipt["failure_reason_code"] == "malformed_response"


@pytest.mark.parametrize("full_thinking", [False, True])
@pytest.mark.parametrize("repair", ["complete", "citation", "semantic"])
def test_initial_and_repair_requests_validate_natively_without_changing_document(monkeypatch, full_thinking, repair):
    from tests.test_intent_details_complete_protocol_correction import simple
    document, answer = simple()
    initial = deepcopy(answer)
    if repair == "complete":
        initial["facts_supported"] = False
    elif repair == "citation":
        initial["unit_grounding"][0]["evidence"][0]["quote"] = "unbound source text"
    else:
        initial["facts_supported"] = False
        initial["unit_grounding"][0]["status"] = "CONTRADICTED"
    monkeypatch.setattr(review, "review_evidence", lambda *args, **kwargs: deepcopy(document))
    calls = []

    async def judge(prompt, **kwargs):
        native_validate(prompt, system=kwargs["system_prompt"],
                        response_format=kwargs["response_format"], full_thinking=full_thinking)
        request_document = json.loads(prompt).get("review_document", json.loads(prompt))
        assert request_document == document
        assert kwargs["max_tokens"] == 800 and kwargs["max_retries"] == 0
        assert kwargs["model"] == review.REVIEW_MODEL
        calls.append(prompt)
        if len(calls) == 1:
            return json.dumps(initial)
        assert len(calls) == 2
        if repair == "complete":
            return json.dumps(answer)
        unit = answer["unit_grounding"][0]
        return json.dumps({"repairs": [{"unit_id": 0, "evidence": unit["evidence"],
                                       **({"status": "VERIFIED"} if repair == "semantic" else {})}]})

    monkeypatch.setattr(verification_helpers, "openrouter_chat", judge)
    receipt = asyncio.run(review.review_intent_details(None, None, [], {}))
    assert len(calls) == 2
    assert receipt["decision"] == "match"


@pytest.mark.parametrize("overflow", ["initial", "repair_user", "repair_system"])
def test_oversize_dispatch_guard_is_local_and_never_clips_projected_evidence(monkeypatch, overflow):
    from tests.test_intent_details_complete_protocol_correction import simple
    document, initial = simple()
    initial["facts_supported"] = False
    document["padding"] = "x" * (32_000 if overflow == "initial" else 30_000)
    monkeypatch.setattr(review, "review_evidence", lambda *args, **kwargs: deepcopy(document))
    if overflow == "repair_user":
        # Repair builders include controls, even when the original user input
        # fits. Do not re-project or clip that already-issued source contract.
        initial["facts_supported"] = True
        initial["unit_grounding"][0]["evidence"][0]["quote"] = "unbound"
        document["padding"] += "x" * (31_900 - len(compact(document)))
    elif overflow == "repair_system":
        monkeypatch.setattr(review, "_COMPLETE_RESPONSE_CORRECTION", "x" * 32_001)
    calls = []

    async def judge(prompt, **kwargs):
        calls.append((prompt, kwargs))
        assert len(prompt) <= 32_000 and len(kwargs["system_prompt"]) <= 32_000
        return json.dumps(initial)

    monkeypatch.setattr(verification_helpers, "openrouter_chat", judge)
    receipt = asyncio.run(review.review_intent_details(None, None, [], {}))
    assert len(calls) == (0 if overflow == "initial" else 1)
    assert receipt["decision"] == "unavailable"
    assert receipt["failure_reason_code"] == "malformed_response"
    assert receipt["failure_class"] != "intent_details_provider_unavailable"


@pytest.mark.parametrize("repair", ["citation", "semantic"])
def test_repair_envelope_must_fit_without_reindexing_document(repair):
    document = {"intent_details_units": [{"unit_id": 0, "text": "Example grew."}],
                "admitted_evidence": [], "padding": "x" * 32_000}
    held = {"unit_grounding": [{"unit_id": 0, "contains_factual_claim": True,
                               "status": "VERIFIED", "evidence": []}]}
    snapshot = deepcopy(document)
    with pytest.raises(ValueError, match="bound"):
        if repair == "citation":
            review._citation_repair_user_prompt(document, {0: ["missing_evidence"]}, held)
        else:
            review._semantic_repair_user_prompt(document, {0: ["missing_evidence"]}, held, {0})
    assert document == snapshot
