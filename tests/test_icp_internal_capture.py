"""Internal evidence follows the real ICP HTTP and bank persistence paths."""

import asyncio
import gzip
import json
from datetime import datetime, timezone

import httpx
import pytest

from gateway.tasks import icp_generator as generator
from lab_arena.service import LocalObjectStore
from tests.test_icp_generator_dynamic_count import _provider_rows


class CaptureSpy:
    def __init__(self, core):
        self.core = core
        self.calls = []

    def start(self, **kwargs):
        call = self.core.start(**kwargs)
        record = {"request": kwargs, "call_id": call.call_id, "prefix": call.prefix,
                  "responses": [], "finishes": []}
        self.calls.append(record)

        class Call:
            call_id = call.call_id

            def response(self, **response):
                record["responses"].append(response)
                call.response(**response)

            def finish(self, **outcome):
                record["finishes"].append(outcome)
                call.finish(**outcome)

        return Call()


def _capture(monkeypatch, tmp_path):
    from lab_arena import internal_capture

    spy = CaptureSpy(internal_capture.InternalCapture(LocalObjectStore(tmp_path)))
    monkeypatch.setattr(internal_capture, "from_environment", lambda: spy)
    return spy


def _transport(monkeypatch, *, generation_status=200, generation_body=None, exclusion_body=None):
    requests = []
    dumps = json.dumps

    class Client:
        def __init__(self, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def post(self, url, *, headers, json):
            requests.append(json)
            slots = json["messages"][0]["content"].split(
                "ORDERED INDUSTRY SLOTS (one ICP per slot, in this order; repeated names are intentional):\n", 1,
            )[1].split("\n", 1)[0].split(", ")
            body = generation_body
            if body is None:
                body = {"id": "gen-1-capture", "usage": {"cost": "0.0000123"},
                        "choices": [{"finish_reason": "stop", "message": {
                            "content": dumps({"icps": _provider_rows(slots)}),
                            "role": "assistant",
                            "reasoning": "Trace with test-provider-secret that must stay private",
                        }}]}
            return httpx.Response(generation_status, json=body)

    def post(url, *, headers, json, timeout):
        requests.append(json)
        body = exclusion_body if exclusion_body is not None else {
            "status": "completed", "usage": {"cost": 0}, "output": [
                {"type": "reasoning", "content": [{"type": "reasoning_text", "text": "Checked exclusion source"}]},
                {"type": "message", "role": "assistant", "status": "completed", "content": [{
                    "type": "output_text", "text": '[{"name":"Excluded","domain":"https://www.excluded.example/path"}]',
                }]},
            ],
        }
        return httpx.Response(200, json=body)

    monkeypatch.setattr(generator.httpx, "AsyncClient", Client)
    monkeypatch.setattr(generator.httpx, "post", post)
    monkeypatch.setattr(generator, "OPENROUTER_API_KEY", "test-provider-secret")
    return requests


@pytest.mark.parametrize("stored,activated,expected", [
    (True, True, "activated"),
    (False, True, "storage_failed"),
    (True, False, "stored_activation_failed"),
])
def test_real_bank_path_records_exact_calls_normalization_and_commit_outcome(
    monkeypatch, tmp_path, stored, activated, expected,
):
    spy = _capture(monkeypatch, tmp_path)
    requests = _transport(monkeypatch)
    persisted = {}

    async def active(*, strict=False):
        return None

    async def store(**kwargs):
        assert not spy.calls[0]["finishes"]
        persisted.update(kwargs)
        return stored

    async def activate(set_id):
        assert stored
        return activated

    monkeypatch.setattr(generator, "get_active_icp_set", active)
    monkeypatch.setattr(generator, "frozen_benchmark_icp_count", lambda set_id: None)
    monkeypatch.setattr(generator, "store_icp_set", store)
    monkeypatch.setattr(generator, "activate_icp_set", activate)
    result = asyncio.run(generator.generate_and_activate_icp_set(
        datetime(2026, 11, 5, tzinfo=timezone.utc), total_icps=10,
    ))
    assert result == (20261105 if stored and activated else None)
    assert len(spy.calls) == len(requests) == 11
    generation = spy.calls[0]
    finish = generation["finishes"]
    assert len(finish) == 1
    assert finish[0]["cost_microusd"] == 13
    outcome = finish[0]["outcome"]
    assert outcome["committed_outcome"] == expected
    assert outcome["normalized_icps"] == persisted["icps"]
    assert outcome["icp_set_hash"] == persisted["icp_set_hash"]
    assert generator.compute_icp_set_hash(persisted["icps"]) == persisted["icp_set_hash"]
    assert outcome["exclusion_call_ids"] == [call["call_id"] for call in spy.calls[1:]]
    for position, call in enumerate(spy.calls):
        assert call["request"]["request"] == requests[position]
        assert len(call["responses"]) == len(call["finishes"]) == 1
        if position:
            context = call["request"]["context"]
            assert context["parent_generation_call_id"] == generation["call_id"]
            assert context["icp_position"] == position - 1
            assert context["call_order"] == position
            assert call["finishes"][0]["cost_microusd"] == 0
            assert call["finishes"][0]["outcome"]["normalized_output"] == ["excluded.example"]
    # Private stored objects must retain readable reasoning but remove credentials.
    private_bytes = b"\n".join(
        gzip.decompress(path.read_bytes()) if path.suffix == ".gz" else path.read_bytes()
        for path in tmp_path.rglob("*") if path.is_file()
    )
    assert b"test-provider-secret" not in private_bytes
    assert b"Trace with" in private_bytes
    assert b"Checked exclusion source" in private_bytes
    assert not any("capture" in key for icp in persisted["icps"] for key in icp)
    from lab_arena.internal_capture import read_call
    raw, candidate = read_call(spy.core.objects, generation["prefix"] + "request.json")
    assert raw["outcome"]["committed_outcome"] == expected
    assert raw["outcome"]["icp_set_hash"] == persisted["icp_set_hash"]
    assert raw["request"] == requests[0]
    assert raw["response"]["choices"][0]["message"]["content"]
    assert "privacy_redactions" in raw["completeness"]["gaps"]
    assert candidate is None
    exclusion, candidate = read_call(spy.core.objects, spy.calls[1]["prefix"] + "request.json")
    assert exclusion["context"]["parent_generation_call_id"] == generation["call_id"]
    assert exclusion["outcome"]["normalized_output"] == ["excluded.example"]
    assert candidate is not None
    assert candidate["quality"]["status"] == "unreviewed"


@pytest.mark.parametrize("status,body", [
    (403, {"error": {"message": "refused"}}),
    (200, {"choices": [{"message": {"content": "not JSON"}}]}),
])
def test_failed_generation_still_has_raw_response_unknown_cost_and_one_finish(
    monkeypatch, tmp_path, status, body,
):
    spy = _capture(monkeypatch, tmp_path)
    _transport(monkeypatch, generation_status=status, generation_body=body)
    assert asyncio.run(generator.generate_icps_with_openrouter(20261105, total_icps=10)) is None
    call, = spy.calls
    assert json.loads(call["responses"][0]["body"]) == body
    assert call["responses"][0]["status"] == status
    assert len(call["finishes"]) == 1
    assert call["finishes"][0]["cost_microusd"] is None
    assert call["finishes"][0]["outcome"]["business_outcome"] == "failed"


def test_capture_storage_failure_does_not_change_generation(monkeypatch):
    from lab_arena import internal_capture

    class Unavailable:
        def put(self, *args):
            raise OSError("storage unavailable")

    monkeypatch.setattr(internal_capture, "from_environment",
                        lambda: internal_capture.InternalCapture(Unavailable()))
    _transport(monkeypatch)
    result = asyncio.run(generator.generate_icps_with_openrouter(20261105, total_icps=10))
    assert len(result[0]) == 10
    assert result[2] == generator.compute_icp_set_hash(result[0])


def test_exclusion_parse_failure_captures_reply_and_logs_only_error_type(monkeypatch, tmp_path, caplog):
    spy = _capture(monkeypatch, tmp_path)
    _transport(monkeypatch, exclusion_body={"status": "completed", "output": [{
        "type": "message", "role": "assistant", "content": [{
            "type": "output_text", "text": '[test-provider-secret]',
        }],
    }]})
    assert generator.generate_exclusions_for_icp({"industry": "Software"}) == []
    call, = spy.calls
    assert len(call["responses"]) == len(call["finishes"]) == 1
    assert call["finishes"][0]["cost_microusd"] is None
    assert "test-provider-secret" not in caplog.text


def test_exclusion_transport_failure_finishes_unknown_without_provider_reply(monkeypatch, tmp_path, caplog):
    spy = _capture(monkeypatch, tmp_path)
    monkeypatch.setattr(generator, "OPENROUTER_API_KEY", "test-provider-secret")

    def unavailable(*args, **kwargs):
        raise httpx.ReadTimeout("test-provider-secret")

    monkeypatch.setattr(generator.httpx, "post", unavailable)
    assert generator.generate_exclusions_for_icp({"industry": "Software"}) == []
    call, = spy.calls
    assert call["responses"] == []
    assert len(call["finishes"]) == 1
    assert call["finishes"][0]["cost_microusd"] is None
    assert "test-provider-secret" not in caplog.text


@pytest.mark.parametrize("cost", [None, True, "NaN", "Infinity", -1, "not-a-cost"])
def test_invalid_provider_charge_never_becomes_zero(monkeypatch, tmp_path, cost):
    spy = _capture(monkeypatch, tmp_path)
    _transport(monkeypatch, exclusion_body={"usage": {"cost": cost}, "choices": [{
        "message": {"content": "[]"},
    }]})
    assert generator.generate_exclusions_for_icp({"industry": "Software"}) == []
    assert spy.calls[0]["finishes"][0]["cost_microusd"] is None
