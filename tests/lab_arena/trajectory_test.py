"""Bounded Arena trajectory validation and provider projections."""

from __future__ import annotations

import base64
import json
import math
from types import SimpleNamespace

import pytest

from lab_arena import trajectory
from lab_arena.broker import BrokerResult
from lab_arena.service import ArenaService, ServiceError
from lab_arena.store import ArenaStoreError, hash_lease_token


def test_event_redacts_credentials_nul_nonfinite_and_hidden_reasoning():
    document = trajectory.event(
        "runtime.stderr",
        {
            "text": "before\x00 authorization=Bearer-secret after",
            "api_key": "should-not-survive",
            "prompt_tokens": 42,
            "run_id": "spoofed-run",
            "ratio": math.inf,
            "nested": {"reasoning": "private chain", "ok": True},
        },
        occurred_at="2026-09-25T12:00:00-07:00",
    )

    assert document["occurred_at"] == "2026-09-25T19:00:00.000Z"
    assert document["content"]["api_key"] == "[REDACTED]"
    assert document["content"]["nested"]["reasoning"] == "[REDACTED]"
    assert document["content"]["prompt_tokens"] == 42
    assert document["content"]["run_id"] == "[DERIVED_BY_GATEWAY]"
    assert document["content"]["ratio"] is None
    assert "\x00" not in document["content"]["text"]
    assert "Bearer-secret" not in document["content"]["text"]
    json.dumps(document, allow_nan=False)


def test_redact_text_replaces_exact_secret_before_chunking():
    secret = "lease-secret-crossing-a-future-chunk-boundary"
    redacted = trajectory.redact_text("prefix " + secret + " suffix", (secret,))
    assert secret not in redacted
    assert redacted == "prefix [REDACTED] suffix"


def test_redact_text_scrubs_structured_and_url_credentials_but_keeps_models():
    text = (
        'json={"api_key":"actual-secret-value","Authorization":'
        '"Bearer bearer-secret-value"} '
        'url=https://example.test/path?api_key=url-secret-value&mode=one '
        'keys=sk-or-v1-abcdefghijklmnopqrst sk-abcdefghijklmnopqrst '
        'model=openai/gpt-6-luna'
    )
    redacted = trajectory.redact_text(text)
    for secret in (
        "actual-secret-value",
        "bearer-secret-value",
        "url-secret-value",
        "sk-or-v1-abcdefghijklmnopqrst",
        "sk-abcdefghijklmnopqrst",
    ):
        assert secret not in redacted
    assert "openai/gpt-6-luna" in redacted


@pytest.mark.parametrize(
    "mutation",
    [
        {"run_id": "spoof"},
        {"lease_token": "spoof"},
        {"runner_hotkey": "5" * 48},
    ],
)
def test_validate_event_rejects_caller_identity(mutation):
    document = trajectory.event("runtime.started", {})
    document.update(mutation)
    with pytest.raises(trajectory.TrajectoryError, match="caller identity"):
        trajectory.validate_event(document)


def test_batch_and_event_caps_are_enforced():
    events = [trajectory.event("runtime.stdout", {"sequence": index}) for index in range(33)]
    with pytest.raises(trajectory.TrajectoryError, match="count"):
        trajectory.validate_batch({"events": events})
    with pytest.raises(trajectory.TrajectoryError, match="too large"):
        trajectory.event(
            "runtime.stdout",
            {"text": "x" * 9000, "other": "y" * 9000},
        )


def test_provider_projection_keeps_useful_shape_without_large_or_hidden_content():
    request = trajectory.provider_request_content(
        {
            "operation_id": "openrouter.responses",
            "action_sequence": 7,
            "timeout_ms": 300_000,
            "parameters": {
                "model": "openai/gpt-6-luna",
                "input": [{"role": "user", "content": "a" * 20_000}],
                "reasoning": "secret",
            },
        }
    )
    assert request["model"] == "openai/gpt-6-luna"
    assert request["parameters"]["reasoning"] == "[REDACTED]"
    assert request["parameters"]["input"]["count"] == 1
    assert "[TRUNCATED]" in request["parameters"]["input"]["latest"][0]["content"]

    response_body = json.dumps(
        {"output": [{"content": "answer"}], "reasoning": "private"}
    ).encode()
    response = trajectory.provider_response_content(
        {
            "status": 200,
            "headers": {"authorization": "never-project"},
            "body_b64": base64.b64encode(response_body).decode(),
            "call": {
                "call_identity": "sha256:" + "a" * 64,
                "provider": "openrouter",
                "provider_status": 200,
                "provider_attempt": 1,
            },
        },
        elapsed_ms=123,
    )
    assert response["response"]["output"]["latest"][0]["content"] == "answer"
    assert "reasoning" not in response["response"]
    assert "headers" not in response
    assert len(json.dumps(response)) < trajectory.MAX_EVENT_BYTES


def test_provider_projection_keeps_latest_action_and_survives_unicode_bounds():
    messages = [
        {"role": "user", "content": "old-%d" % index}
        for index in range(20)
    ]
    messages[-1]["content"] = "latest-ICP-action " + "\U0001f642" * 4000
    content = trajectory.provider_request_content(
        {
            "operation_id": "openrouter.chat",
            "action_sequence": 20,
            "timeout_ms": 120_000,
            "parameters": {
                "model": "openai/gpt-6-luna",
                "messages": messages,
                ("\U0001f642" * 80): "value",
            },
        }
    )
    event = trajectory.event("provider.request", content)
    assert content["parameters"]["messages"]["count"] == 20
    assert "latest-ICP-action" in json.dumps(
        content["parameters"]["messages"]["latest"]
    )
    assert len(json.dumps(event)) < trajectory.MAX_EVENT_BYTES


def test_large_provider_response_keeps_latest_output_and_usage_metadata():
    body = {
        "id": "response-1",
        "status": "completed",
        "model": "openai/gpt-6-luna",
        "usage": {"input_tokens": 100, "output_tokens": 50},
        "output": [
            {"type": "message", "content": "old"},
            {"type": "message", "content": "latest " + "x" * 20_000},
        ],
        "padding": "z" * 20_000,
        "reasoning": "must not appear",
    }
    response = trajectory.provider_response_content(
        {
            "status": 200,
            "body_b64": base64.b64encode(json.dumps(body).encode()).decode(),
            "call": {"provider": "openrouter", "provider_status": 200},
        },
        elapsed_ms=99,
    )
    assert response["response"]["usage"]["input_tokens"] == 100
    assert response["response"]["output"]["count"] == 2
    assert "latest" in json.dumps(response["response"]["output"]["latest"])
    assert "must not appear" not in json.dumps(response)
    trajectory.event("provider.response", response)


def _service_for_trajectory(store, broker=None):
    service = object.__new__(ArenaService)
    service._store = store
    service._require_round_ownership = lambda round_id: None
    if broker is not None:
        service._broker_for = lambda round_id: broker
    return service


def _run():
    return {
        "run_id": "run-1",
        "assignment_id": "assignment-1",
        "round_id": "round-1",
        "submission_id": "submission-1",
        "miner_hotkey": "5" * 48,
        "runner_hotkey": "6" * 48,
        "stage": 1,
        "icp_position": 2,
        "attempt": 1,
        "kind": "execute",
    }


def test_service_trajectory_passes_only_lease_hash_and_sanitized_events():
    calls = []

    class Store:
        get_run = staticmethod(lambda run_id: _run() if run_id == "run-1" else None)

        @staticmethod
        def append_trajectory_events(run_id, lease_hash, events):
            calls.append((run_id, lease_hash, events))
            return {
                "status": "accepted", "accepted": len(events),
                "inserted": len(events), "existing": 0,
            }

    service = _service_for_trajectory(Store())
    document = {"events": [trajectory.event("runtime.started", {"ok": True})]}
    result = service.handle_trajectory("run-1", "lease-token", document)
    assert result["accepted"] == 1
    assert calls[0][0:2] == ("run-1", hash_lease_token("lease-token"))
    assert "run_id" not in calls[0][2][0]

    with_secret = {
        "events": [
            trajectory.event(
                "runtime.stderr", {"text": "leaked lease-token value"}
            )
        ]
    }
    service.handle_trajectory("run-1", "lease-token", with_secret)
    assert "lease-token" not in calls[-1][2][0]["content"]["text"]

    Store.append_trajectory_events = staticmethod(
        lambda *_args: {"status": "stale"}
    )
    with pytest.raises(ServiceError, match="lease_stale"):
        service.handle_trajectory("run-1", "wrong-token", document)
    with pytest.raises(ServiceError, match="runtime events required"):
        service.handle_trajectory(
            "run-1",
            "lease-token",
            {"events": [trajectory.event("provider.request", {})]},
        )


def test_provider_trajectory_failures_never_change_provider_result():
    class Store:
        get_run = staticmethod(lambda run_id: _run())

        @staticmethod
        def append_trajectory_events(*_args):
            raise ArenaStoreError("diagnostics unavailable")

    expected = BrokerResult(
        200,
        {"content-type": "application/json"},
        b'{"answer":"ok"}',
        {
            "call_identity": "sha256:" + "a" * 64,
            "provider": "openrouter",
            "provider_status": 200,
            "provider_attempt": 1,
        },
    )
    broker = SimpleNamespace(execute=lambda *_args, **_kwargs: expected)
    service = _service_for_trajectory(Store(), broker)
    frame = {
        "operation_id": "openrouter.chat",
        "parameters": {
            "model": "openai/gpt-4o-mini",
            "messages": [{"role": "user", "content": "hello"}],
        },
        "timeout_ms": 120_000,
        "action_sequence": 3,
    }

    assert service.handle_provider("run-1", "lease-token", frame) == expected.to_document()


@pytest.mark.parametrize("http_status", [200, 403, 502])
def test_provider_trajectory_records_request_and_bounded_terminal_status(http_status):
    persisted = []

    class Store:
        get_run = staticmethod(lambda run_id: _run())

        @staticmethod
        def append_trajectory_events(_run_id, _lease_hash, events):
            persisted.extend(events)
            return {
                "status": "accepted", "accepted": len(events),
                "inserted": len(events), "existing": 0,
            }

    result = BrokerResult(
        http_status,
        {},
        json.dumps({"answer": "ok"}).encode(),
        {
            "call_identity": "sha256:" + "a" * 64,
            "operation_id": "openrouter.chat",
            "provider": "openrouter",
            "provider_status": http_status,
            "provider_attempt": 2,
            "error_code": None if http_status == 200 else "provider_unavailable",
        },
    )
    service = _service_for_trajectory(
        Store(), SimpleNamespace(execute=lambda *_args, **_kwargs: result)
    )
    frame = {
        "operation_id": "openrouter.chat",
        "parameters": {
            "model": "openai/gpt-4o-mini",
            "messages": [{"role": "user", "content": "private prompt"}],
        },
        "timeout_ms": 120_000,
        "action_sequence": 9,
    }

    returned = service.handle_provider("run-1", "lease-token", frame)
    assert returned == result.to_document()
    assert [item["kind"] for item in persisted] == [
        "provider.request", "provider.response"
    ]
    assert persisted[0]["content"]["model"] == "openai/gpt-4o-mini"
    assert persisted[1]["content"]["call"]["call_identity"].startswith("sha256:")
    assert persisted[1]["content"]["call"]["provider_status"] == http_status
