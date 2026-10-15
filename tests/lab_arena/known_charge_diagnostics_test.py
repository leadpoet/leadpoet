"""Safe diagnostics for the existing known-charge exception settlement."""

import json
from types import SimpleNamespace

import pytest

from lab_arena import broker as br, operations, telemetry
from lab_arena.service import _provider_telemetry_fields
from tests.lab_arena.provider_trajectory_audit_test import FRAME
from tests.lab_arena.test_lab_arena_broker import (
    CHAT, CONTEXT, FakeTransport, make_broker,
)
from tests.lab_arena.trajectory_test import _run, _service_for_trajectory


SECRET = "private-message-that-must-not-be-retained"


@pytest.mark.parametrize("stage", ["response_adaptation", "response_sanitization"])
@pytest.mark.parametrize("exception_kind", ["known", "unknown"])
def test_known_charge_diagnostic_reaches_trajectory_and_telemetry_without_policy_change(
    monkeypatch, stage, exception_kind,
):
    error = ValueError(SECRET) if exception_kind == "known" else type(
        "PrivateExceptionNameMustNotBeRetained", (Exception,), {},
    )(SECRET)

    def refuse(*_args, **_kwargs):
        raise error

    if stage == "response_adaptation":
        monkeypatch.setattr(br, "_provider_call_succeeded", refuse)
    else:
        monkeypatch.setattr(operations, "sanitize_response", refuse)
    provider, ledger, transport = make_broker(transport=FakeTransport([(200, {
        "model": "openai/gpt-4o-mini", "choices": [],
        "usage": {"cost": "0.0000123"}, "private_payload": SECRET,
    })]))

    class Store:
        persisted = []
        get_run = staticmethod(lambda _run_id: _run())

        @classmethod
        def append_trajectory_events(cls, _run_id, _lease_hash, events):
            cls.persisted.extend(events)
            return {"status": "accepted", "accepted": len(events),
                    "inserted": len(events), "existing": 0}

    recorded = []
    monkeypatch.setattr(telemetry, "_recorder", SimpleNamespace(
        record_provider=lambda *args, **kwargs: recorded.append((args, kwargs)),
    ))
    service = _service_for_trajectory(Store(), provider)
    document = service.handle_provider("run-1", "lease-token", FRAME)
    result = br.parse_broker_document(document)
    expected_class = "ValueError" if exception_kind == "known" else "Exception"
    assert result.status == 502
    assert json.loads(result.body) == {"error": {"code": "provider_unavailable"}}
    assert result.call["error_code"] == "provider_unavailable"
    assert result.call["outcome"] == "settled"
    assert result.call["actual_microusd"] == 13
    assert "provider_status" not in result.call
    assert result.call["observed_provider_http_status"] == 200
    assert result.call["failure_stage"] == stage
    assert result.call["error_class"] == expected_class
    assert len(transport.sent) == 1
    assert ledger.log == ["reserve", "dispatch", "settle"]
    saved = next(iter(ledger.calls.values()))
    assert saved["actual"] == 13
    assert saved["terminal"]["call_succeeded"] is False
    assert saved["terminal"]["status"] == 502
    assert saved["terminal"]["body_b64"] == document["body_b64"]
    for field in ("failure_stage", "error_class", "observed_provider_http_status"):
        assert field not in saved["terminal"]
    event = Store.persisted[-1]
    assert event["kind"] == "provider.response"
    assert event["content"]["call"]["error_class"] == expected_class
    assert event["content"]["call"]["failure_stage"] == stage
    assert event["content"]["call"]["observed_provider_http_status"] == 200
    assert recorded[-1][1]["error_type"] == expected_class
    assert recorded[-1][1]["provider_status"] == 200
    assert recorded[-1][1]["http_status"] == 502
    assert recorded[-1][0][2] == "failed"
    serialized = json.dumps([document, Store.persisted, recorded, saved["terminal"]])
    assert SECRET not in serialized
    assert "PrivateExceptionNameMustNotBeRetained" not in serialized


@pytest.mark.parametrize("invalid", [True, -1, 0, 99, 600, "200", None])
def test_telemetry_rejects_invalid_observed_http_status_and_exception_labels(invalid):
    call = {
        "operation_id": "openrouter.chat", "error_code": "provider_unavailable",
        "observed_provider_http_status": invalid, "error_class": SECRET,
    }
    fields = _provider_telemetry_fields(call, 502, 1)
    assert fields["provider_status"] == 0
    assert "error_type" not in fields
    call["provider_status"] = 429
    assert _provider_telemetry_fields(call, 502, 1)["provider_status"] == 429


def test_normal_known_charge_success_has_no_exception_diagnostics():
    provider, ledger, transport = make_broker(transport=FakeTransport([(200, {
        "id": "generation", "model": "openai/gpt-4o-mini",
        "choices": [{"message": {"role": "assistant", "content": "ok"}}],
        "usage": {"cost": "0.0000123"},
    })]))
    result = provider.execute(CONTEXT, operation_id="openrouter.chat", parameters=CHAT,
                              action_sequence=0, timeout_ms=30000)
    assert result.status == 200
    assert result.call["actual_microusd"] == 13
    assert len(transport.sent) == 1
    assert ledger.log == ["reserve", "dispatch", "settle"]
    assert not {"error_class", "failure_stage", "observed_provider_http_status"}.intersection(result.call)
