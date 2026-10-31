"""Private outcome metadata must not become review or public authority."""

import copy
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from lab_arena import code_review, code_review_policy, public_dashboard
from lab_arena.service import ArenaService
from tests.lab_arena.test_lab_arena_code_review_runtime import (
    MINER_KEY, ReviewTransport, _archive, _review_body, _reviewer, _source,
)


def _response(parameters):
    body = _review_body(parameters)
    body.update(id="gen-123-abc", provider="Anthropic")
    body["choices"][0]["native_finish_reason"] = "end_turn"
    body["usage"].update(prompt_tokens=200, completion_tokens=33,
                         completion_tokens_details={"reasoning_tokens": 9})
    body["openrouter_metadata"] = {
        "requested": parameters["model"], "strategy": "direct", "attempt": 1,
        "endpoints": {"total": 1, "available": [{
            "selected": True, "model": parameters["model"], "provider": "Anthropic",
        }]},
    }
    return body


def _run(change=lambda body: None):
    def respond(parameters):
        body = _response(parameters)
        change(body)
        return {"body": body}
    reviewer, row, store, _objects = _reviewer(
        _archive(_source()), ReviewTransport(respond),
    )
    outcome = reviewer.review(row)
    return outcome, store.finish_calls[0][4], store.finish_calls[0][5]


@pytest.mark.parametrize("finish,status,reason", [
    ("stop", "passed", None), ("length", "error", "not_finished"),
    ("content_filter", "error", "not_finished"), (None, "error", "not_finished"),
])
def test_runtime_retains_private_provider_outcome_without_changing_decision(finish, status, reason):
    outcome, document, cost = _run(
        lambda body: body["choices"][0].update(finish_reason=finish),
    )
    assert outcome["status"] == status
    assert cost == 1234
    assert document.get("error_reason") == reason
    assert document["provider_diagnostic"] == {
        "finish_reason": finish if finish is not None else "missing",
        "native_finish_reason": "end_turn", "max_output_tokens": 16384,
        "prompt_tokens": 200, "completion_tokens": 33, "reasoning_tokens": 9,
        "context_integrity": "not_required",
        "response_model": code_review.DEFAULT_REVIEW_MODEL,
        "selected_model": code_review.DEFAULT_REVIEW_MODEL,
        "selected_provider": "Anthropic", "generation_id": "gen-123-abc",
    }


def test_context_failure_is_private_and_keeps_existing_first_failure(monkeypatch):
    prepare = code_review.prepare_request
    monkeypatch.setattr(code_review, "prepare_request", lambda *a, **k: replace(
        prepare(*a, **k), requires_context_integrity=True,
    ))
    def change(body):
        body["choices"][0]["finish_reason"] = "length"
        body["openrouter_metadata"]["pipeline"] = [{"private_source": "never retain"}]
    outcome, document, cost = _run(change)
    assert outcome["status"] == "error" and cost == 1234
    assert document["error_reason"] == "context_integrity"
    assert document["provider_diagnostic"]["context_integrity"] == "pipeline"
    assert document["provider_diagnostic"]["finish_reason"] == "length"
    assert "never retain" not in json.dumps(document)


@pytest.mark.parametrize("change", [
    lambda body: body["choices"][0].update(native_finish_reason="private prose" * 10000),
    lambda body: body["usage"].update(prompt_tokens=True, completion_tokens=-1,
                                      completion_tokens_details={"reasoning_tokens": 10**100}),
    lambda body: body["openrouter_metadata"]["endpoints"].update(available=[{}] * 10000),
    lambda body: body.update(id="gen-" + "x" * 10000),
    lambda body: body["openrouter_metadata"]["endpoints"]["available"][0].update(provider="private prose"),
])
def test_malformed_or_large_diagnostic_values_do_not_change_review(change):
    outcome, document, cost = _run(change)
    assert outcome["status"] == "passed" and cost == 1234
    diagnostic = document["provider_diagnostic"]
    assert len(json.dumps(diagnostic)) < 1024
    assert "private prose" not in json.dumps(diagnostic)
    assert all(type(diagnostic[k]) is int and 0 <= diagnostic[k] <= 10**9
               for k in ("prompt_tokens", "completion_tokens", "reasoning_tokens") if k in diagnostic)


def test_credential_echo_never_enters_private_diagnostics():
    outcome, document, _cost = _run(lambda body: body.update(id="gen-123-" + MINER_KEY))
    assert outcome["status"] == "error"
    assert document["error_code"] == "code_review_credential_echo"
    assert "provider_diagnostic" not in document
    assert MINER_KEY not in json.dumps(document)


@pytest.mark.parametrize("error", [ValueError, RuntimeError, MemoryError])
def test_diagnostic_projection_failure_cannot_replace_verdict_or_cost(monkeypatch, error):
    def fail(*_args, **_kwargs):
        raise error("private diagnostic failure")
    monkeypatch.setattr(code_review_policy, "private_response_diagnostic", fail)
    outcome, document, cost = _run()
    assert outcome["status"] == "passed" and cost == 1234
    assert "provider_diagnostic" not in document


def test_public_review_views_exclude_diagnostics_without_changing_existing_fields():
    _outcome, document, _cost = _run()
    row = {"code_review_status": "passed", "code_review_attempts": 1,
           "code_review_doc": document}
    baseline = copy.deepcopy(row)
    baseline["code_review_doc"].pop("provider_diagnostic")
    for project in (public_dashboard.code_review_summary, ArenaService._public_code_review):
        assert project(row) == project(baseline)
        assert "gen-123-abc" not in json.dumps(project(row))


@pytest.mark.parametrize("change,reason", [
    (lambda b: b.pop("openrouter_metadata"), "metadata_missing"),
    (lambda b: b["openrouter_metadata"].update(requested="other"), "requested_model"),
    (lambda b: b["openrouter_metadata"].update(strategy="cached"), "strategy"),
    (lambda b: b["openrouter_metadata"].update(attempt=True), "attempt"),
    (lambda b: b["openrouter_metadata"].update(pipeline=None), "pipeline"),
    (lambda b: b["openrouter_metadata"].pop("endpoints"), "endpoints_missing"),
    (lambda b: b["openrouter_metadata"]["endpoints"].update(available=[]), "endpoints_available"),
    (lambda b: b["openrouter_metadata"]["endpoints"].update(total=False), "endpoints_total"),
    (lambda b: b["openrouter_metadata"]["endpoints"]["available"][0].update(selected="yes"), "endpoint_shape"),
    (lambda b: b["openrouter_metadata"]["endpoints"]["available"][0].update(selected=False), "selected_count"),
    (lambda b: b["openrouter_metadata"]["endpoints"]["available"][0].update(model="other"), "selected_model"),
    (lambda b: b.pop("usage"), "usage_missing"),
    (lambda b: b["usage"].update(prompt_tokens=True), "prompt_tokens"),
    (lambda b: b["usage"].update(prompt_tokens=1_000_000), "context_budget"),
    (lambda b: None, None),
])
def test_context_branch_observation_preserves_existing_parse_order(change, reason):
    prepared = replace(code_review.prepare_request(_archive(_source())), requires_context_integrity=True)
    body = _response(prepared.parameters)
    body["choices"][0]["finish_reason"] = "length"
    change(body)
    assert code_review.context_integrity_failure(body, prepared) == reason
    assert code_review._context_integrity_verified(body, prepared) is (reason is None)
    with pytest.raises(code_review.CodeReviewError) as caught:
        code_review.parse_response(body, prepared)
    assert caught.value.response_reason == ("context_integrity" if reason else "not_finished")
    diagnostic = code_review_policy.private_response_diagnostic(body, prepared)
    assert diagnostic["context_integrity"] == (reason or "verified")
    assert diagnostic["context_integrity"] in code_review_policy.REVIEW_CONTEXT_REASONS


@pytest.mark.parametrize("response,reason", [([], "envelope"), ({"choices": [[]]}, "model_mismatch")])
def test_malformed_provider_envelope_still_finishes_as_the_existing_error(response, reason):
    reviewer, row, store, _objects = _reviewer(
        _archive(_source()), ReviewTransport(lambda _parameters: {"body": response}),
    )
    assert reviewer.review(row)["status"] == "error"
    doc = store.finish_calls[0][4]
    assert doc["error_reason"] == reason
    assert doc["provider_diagnostic"]["finish_reason"] == "invalid"


def test_public_submission_and_round_paths_exclude_private_diagnostics():
    from tests.lab_arena.test_lab_arena_public_dashboard import DashboardService, _published_round, _now, MINER_HOTKEY

    _outcome, document, _cost = _run()
    round_row = _published_round()
    row = {
        "submission_id": "sub-private-review", "round_id": round_row["round_id"],
        "miner_hotkey": MINER_HOTKEY, "status": "rejected", "is_king": False,
        "accepted_at": "2026-09-08T17:50:08Z", "rejection_rule": "code_review_incomplete",
        "replaced_by_submission_id": None, "code_review_status": "error",
        "code_review_attempts": 1, "code_review_doc": document,
    }
    service = DashboardService(_round=lambda _: round_row, now=_now,
        _store=SimpleNamespace(list_submissions=lambda *_a, **_k: [row],
                              list_runs=lambda *_a, **_k: [], list_runtime_starts=lambda *_a, **_k: []))
    assert "gen-123-abc" not in json.dumps(public_dashboard.submissions_snapshot(service, round_row["round_id"]))
    status_service = SimpleNamespace(_require_identifier=lambda *_: None,
        _store=SimpleNamespace(get_submission=lambda _: row), _round=lambda _: round_row,
        now=_now, _public_code_review=ArenaService._public_code_review)
    status = ArenaService.submission_status(status_service, row["submission_id"])
    assert "provider_diagnostic" not in json.dumps(status)
    assert "gen-123-abc" not in json.dumps(status)
