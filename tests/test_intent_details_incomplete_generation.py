"""Incomplete paragraph generation spends the existing correction, never a third call."""
import asyncio
from copy import deepcopy
import json
from types import SimpleNamespace

import httpx
import pytest

from lab_arena import scoring
from qualification.scoring import competition, intent_details as review
from qualification.scoring import openrouter_options as options, verification_helpers as helpers
from tests.test_intent_details_complete_protocol_correction import simple
from tests.lab_arena.test_lab_arena_scoring import make_icp, scored_company

MODEL = "deepseek/deepseek-v4.1-flash-1017"
# Retain the Nov 8 HTTP-200 / length / invalid-JSON / 5,502-character shape.
# This synthetic text contains no private production prose.
TRUNCATED = '{"unit_grounding":[' + " " * (5502 - len('{"unit_grounding":['))


def chat(answer, finish="stop"):
    return {"choices": [{"finish_reason": finish, "message": {
        "content": answer if isinstance(answer, str) else json.dumps(answer),
        "reasoning": "Private native thinking remains in the HTTP capture.",
    }}]}


def setup(monkeypatch, answers, *, document=None, elapsed=0):
    document = deepcopy(document if document is not None else simple()[0])
    before = deepcopy(document)
    monkeypatch.setattr(review, "review_evidence", lambda *a, **k: deepcopy(document))
    monkeypatch.setattr(review, "REVIEW_MODEL", MODEL)
    monkeypatch.setattr(helpers, "OPENROUTER_API_KEY", "test-not-secret")
    monkeypatch.setenv("ARENA_JUDGE_REASONING_CAPTURE_POLICY", options.JUDGE_REASONING_POLICY)
    monkeypatch.setenv("ARENA_JUDGE_PROVIDER_ROUTES", json.dumps({MODEL: "example/verified"}))
    monkeypatch.setenv("ARENA_JUDGE_REQUEST_OPTIONS", json.dumps({MODEL: {
        "reasoning": {"effort": "low"}, "max_output_tokens": 8192}}))
    calls, timeouts, clock = [], [], [100.0]

    class Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def post(self, url, **kwargs):
            calls.append(deepcopy(kwargs))
            assert len(calls) <= len(answers), "unexpected extra paragraph call"
            answer = answers[len(calls) - 1]
            if len(calls) == 1:
                clock[0] += elapsed
            if isinstance(answer, Exception):
                raise answer
            return SimpleNamespace(raise_for_status=lambda: None, json=lambda: answer)

    async def wait_for(awaitable, *, timeout):
        timeouts.append(timeout)
        return await awaitable

    monkeypatch.setattr(helpers.httpx, "AsyncClient", Client)
    monkeypatch.setattr(review, "asyncio", SimpleNamespace(
        get_running_loop=lambda: SimpleNamespace(time=lambda: clock[0]),
        wait_for=wait_for, TimeoutError=asyncio.TimeoutError))
    result = asyncio.run(review.review_intent_details(None, None, [], {}))
    assert document == before
    return result, calls, timeouts


def assert_systemic(receipt):
    assert receipt["decision"] == "unavailable"
    assert receipt["failure_class"] == "intent_details_review_unavailable"
    assert receipt["failure_reason_code"] == "malformed_response"
    row = {"verifier_gate_receipts": [receipt]}
    assert not competition.scorer_breakdown_has_company_local_verification_failure(row)
    for integrity in (False, True):
        assert competition.scorer_breakdown_has_retryable_infrastructure_failure(row, integrity_policy=integrity)
    assert scoring._retryable_breakdown_failure_reason(row) == "malformed_response"
    assert "thinking" not in json.dumps(receipt).casefold()


@pytest.mark.parametrize("negative", [False, True])
def test_nov8_retained_truncation_shape_uses_one_fresh_full_review(monkeypatch, negative):
    _, fresh = simple()
    if negative:
        fresh["facts_supported"] = False
        fresh["unit_grounding"][0].update(status="UNPROVEN", evidence=[])
    receipt, calls, timeouts = setup(monkeypatch, [chat(TRUNCATED, "length"), chat(fresh)], elapsed=31.7)
    assert len(calls) == 2
    assert timeouts == [120, pytest.approx(88.3)]
    assert receipt["decision"] == ("mismatch" if negative else "match")
    assert calls[0]["json"]["messages"][1] == calls[1]["json"]["messages"][1]
    assert "incomplete" in calls[1]["json"]["messages"][0]["content"].casefold()
    for call in calls:
        body = call["json"]
        assert body["max_tokens"] == 8192
        assert body["reasoning"] == {"effort": "low", "exclude": False}
        assert body["provider"] == {"only": ["example/verified"]}
        assert body["response_format"] == review._RESPONSE_FORMAT
        assert TRUNCATED not in json.dumps(body)


@pytest.mark.parametrize("second", [chat(TRUNCATED, "length"), chat("not JSON")])
def test_repeated_or_malformed_generation_is_systemic_without_third_call(monkeypatch, second):
    receipt, calls, _ = setup(monkeypatch, [chat(TRUNCATED, "length"), second])
    assert len(calls) == 2
    assert_systemic(receipt)


def test_complete_but_nonstop_content_is_never_accepted(monkeypatch):
    _, answer = simple()
    receipt, calls, _ = setup(monkeypatch, [chat(answer, "length"), chat(TRUNCATED, "length")])
    assert len(calls) == 2
    assert_systemic(receipt)


@pytest.mark.parametrize("elapsed", [120, 130])
def test_incomplete_generation_cannot_extend_shared_deadline(monkeypatch, elapsed):
    receipt, calls, timeouts = setup(monkeypatch, [chat(TRUNCATED, "length")], elapsed=elapsed)
    assert len(calls) == 1 and timeouts == [120]
    assert_systemic(receipt)


@pytest.mark.parametrize("initial_defect", ["aggregate", "citation", "semantic"])
def test_incomplete_existing_repair_does_not_dispatch_third_or_become_source_exhausted(monkeypatch, initial_defect):
    _, initial = simple()
    if initial_defect == "aggregate":
        initial["facts_supported"] = False
    elif initial_defect == "citation":
        initial["unit_grounding"][0]["evidence"][0]["quote"] = "unbound quote"
    else:
        initial["facts_supported"] = False
        initial["unit_grounding"][0]["status"] = "CONTRADICTED"
    receipt, calls, _ = setup(monkeypatch, [chat(initial), chat(TRUNCATED, "length")])
    assert len(calls) == 2
    assert_systemic(receipt)


@pytest.mark.parametrize("first_incomplete", [False, True])
@pytest.mark.parametrize("transport_error", [httpx.ConnectError("private diagnostic"), httpx.ReadTimeout("private diagnostic")])
def test_actual_transport_failure_keeps_provider_label(monkeypatch, first_incomplete, transport_error):
    answers = ([chat(TRUNCATED, "length")] if first_incomplete else []) + [transport_error]
    receipt, calls, _ = setup(monkeypatch, answers)
    assert len(calls) == len(answers)
    assert receipt["failure_class"] == "intent_details_provider_unavailable"
    assert receipt["failure_reason_code"] == "provider_error"
    assert "private diagnostic" not in json.dumps(receipt)


@pytest.mark.parametrize("negative", [False, True])
def test_healthy_response_uses_one_call(monkeypatch, negative):
    _, answer = simple()
    if negative:
        answer["facts_supported"] = False
        answer["unit_grounding"][0].update(status="UNPROVEN", evidence=[])
    receipt, calls, _ = setup(monkeypatch, [chat(answer)])
    assert len(calls) == 1
    assert receipt["decision"] == ("mismatch" if negative else "match")


def test_exhausted_incomplete_generation_raises_instead_of_company_zero(monkeypatch):
    receipt, _, _ = setup(monkeypatch, [chat(TRUNCATED, "length"), chat(TRUNCATED, "length")])
    row = {"company_qualified": False, "final_score": 0., "intent_signals_detail": [],
           "verifier_gate_receipts": [receipt]}
    attempts = []

    def scorer(selected, icp, is_reference_model):
        attempts.append(len(selected))
        return [deepcopy(row) for _ in selected]

    with pytest.raises(scoring.ScoringError, match="judge_exhausted attempts=2 reason=malformed_response"):
        scoring.score_work_item({"scored_run_id": "incomplete-generation"}, icp=make_icp(0),
            companies=[scored_company(0)], scorer=scorer, max_retries=2)
    assert attempts == [1, 1]


@pytest.mark.parametrize("defect", ["citation", "missing_unit", "aggregate", "wrong_boolean"])
def test_fresh_complete_answer_still_requires_entire_native_contract(monkeypatch, defect):
    _, fresh = simple()
    if defect == "citation":
        fresh["unit_grounding"][0]["evidence"][0]["quote"] = "unbound quote"
    elif defect == "missing_unit":
        fresh["unit_grounding"] = []
    elif defect == "aggregate":
        fresh["facts_supported"] = False
    else:
        fresh["natural_paragraph"] = "true"
    receipt, calls, _ = setup(monkeypatch, [chat(TRUNCATED, "length"), chat(fresh)])
    assert len(calls) == 2
    assert_systemic(receipt)


def test_error_message_cannot_impersonate_typed_incomplete_generation(monkeypatch):
    receipt, calls, _ = setup(monkeypatch, [ValueError("judge_response_incomplete")])
    assert len(calls) == 1
    assert receipt["failure_reason_code"] == "provider_error"


def test_shared_helper_retains_static_valueerror_contract_and_does_not_retry(monkeypatch):
    setup(monkeypatch, [chat(simple()[1])])
    calls = []

    class Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def post(self, *args, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(raise_for_status=lambda: None,
                                   json=lambda: chat(TRUNCATED, "length"))

    monkeypatch.setattr(helpers.httpx, "AsyncClient", Client)
    with pytest.raises(helpers.IncompleteJudgeResponse, match="^judge_response_incomplete$") as exc:
        asyncio.run(helpers.openrouter_chat("Check", MODEL, max_retries=2))
    assert isinstance(exc.value, ValueError)
    assert exc.value.args == ("judge_response_incomplete",)
    assert len(calls) == 1


@pytest.mark.parametrize("cached_company_policy", [False, True])
def test_native_entrypoint_keeps_incomplete_generation_retryable_and_unsaved(monkeypatch, cached_company_policy):
    import os
    from lab_arena import company_judgments, scorer_entrypoint
    from tests.lab_arena import company_judgments_test as cache_fixtures
    from tests.lab_arena.test_lab_arena_intent_details_real_judge import _scoring_input
    from tests.test_arena_score_integrity import _positive_breakdown

    receipt, calls, _ = setup(monkeypatch, [chat(TRUNCATED, "length"), chat(TRUNCATED, "length")])
    if cached_company_policy:
        document = cache_fixtures._input()
        document["companies"] = document["companies"][:1]
        document["company_judgment_cache"] = {
            "schema_version": company_judgments.LEASE_SCHEMA_VERSION,
            "hits": [], "misses": [cache_fixtures._miss(cache_fixtures._refs(document)[0])],
        }
    else:
        document = _scoring_input()
    attempts = []

    class FixtureScorer:
        integrity_policy = cached_company_policy
        company_quality = cached_company_policy
        contacts_required = False

        def __call__(self, companies, icp, is_reference_model):
            attempts.append(len(companies))
            rows = []
            for company in companies:
                row = _positive_breakdown(company["company_name"], company["company_website"].split("//", 1)[1].rstrip("/"))
                row.update(company_index=0, company_qualified=False, final_score=0.,
                           failure_reason="Intent Details verification unavailable: incomplete review")
                row["verifier_gate_receipts"].append(deepcopy(receipt))
                rows.append(row)
            return rows

    monkeypatch.setattr(os, "environ", os.environ.copy())
    for name in scoring.CREDENTIAL_ENV_NAMES:
        monkeypatch.delenv(name, raising=False)
    for name in ("ARENA_JUDGE_REASONING_CAPTURE_POLICY", "ARENA_JUDGE_PROVIDER_ROUTES",
                 "ARENA_JUDGE_REQUEST_OPTIONS"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(scoring, "lab_scorer", lambda *args, **kwargs: FixtureScorer())
    result = scorer_entrypoint.score_input(document)
    assert result["failure"] == "judge_error"
    assert result["reason"] == "malformed_response"
    assert not result.get("breakdowns") and not result.get("terminal_breakdowns")
    assert not result.get("company_judgments")
    assert len(attempts) == scoring.MAX_JUDGE_RETRIES
    assert len(calls) == 2
