"""Frozen lease policy reaches both worker request ingress protocols."""

import json

import pytest

from lab_arena import contracts, operations, runner, shim
from tests.lab_arena.test_lab_arena_runner import FakeApi, lease


CONTINUATION = {
    "model": "openai/gpt-4o-mini", "max_tokens": 20,
    "messages": [
        {"role": "user", "content": "Find evidence."},
        {"role": "assistant", "content": None, "reasoning": "Need the cited source.",
         "reasoning_details": [{"type": "reasoning.text", "text": "Check source.",
                                "id": "r1", "signature": None, "format": "unknown", "index": 0}],
         "tool_calls": [{"id": "call-1", "index": 0, "type": "function",
                         "function": {"name": "search", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "call-1", "content": "Found one source."},
    ],
}


def _worker(tmp_path, *, full, kind="execute"):
    issued = lease()
    issued["kind"] = kind
    if full:
        issued["reasoning_capture_policy"] = shim.FULL_THINKING_POLICY
    api = FakeApi([])
    state = runner.RunState(lease=issued, lease_token="host-lease-token")
    return runner.WorkerSocketServer(tmp_path / "worker.sock", api, state), api


@pytest.mark.parametrize("full,kind,allowed", [
    (False, "execute", False), (True, "execute", True),
    (True, "score", False),
])
def test_worker_frame_and_http_share_frozen_policy(tmp_path, full, kind, allowed):
    worker, api = _worker(tmp_path, full=full, kind=kind)
    frame = shim.build_operation_frame("openrouter.chat", CONTINUATION, 1000)
    answer = json.loads(worker.handle_frame(frame))
    status, _headers, _body = worker.handle_http(
        "POST", "https://openrouter.ai/api/v1/chat/completions",
        json.dumps(CONTINUATION).encode(), {},
    )
    if allowed:
        assert answer["status"] == status == 200
        assert len(api.provider_frames) == 2
        for sent in api.provider_frames:
            message = sent["parameters"]["messages"][1]
            assert message["content"] is None
            assert message["reasoning"] == "Need the cited source."
            assert message["reasoning_details"][0]["signature"] is None
            assert message["tool_calls"][0]["index"] == 0
    else:
        assert answer == {"error": "invalid_request"}
        assert status == 400
        assert api.provider_frames == []


def test_sandbox_hint_cannot_override_worker_lease(tmp_path, monkeypatch):
    monkeypatch.setenv(shim.REASONING_POLICY_ENV, shim.FULL_THINKING_POLICY)
    monkeypatch.setattr(shim, "_dispatch_with_response_url",
                        lambda *args: (200, {}, b"ok", ""))
    status, _headers, body, _url = shim._execute_with_response_url(
        method="POST", url="https://openrouter.ai/api/v1/chat/completions",
        headers={}, body=json.dumps(CONTINUATION).encode(), timeout_ms=1000,
    )
    assert (status, body) == (200, b"ok")

    worker, api = _worker(tmp_path, full=False)
    forged = json.loads(shim.build_operation_frame("openrouter.chat", CONTINUATION, 1000))
    forged["allow_full_thinking"] = True
    assert json.loads(worker.handle_frame(json.dumps(forged).encode())) == {
        "error": "invalid_frame",
    }
    assert json.loads(worker.handle_frame(
        shim.build_operation_frame("openrouter.chat", CONTINUATION, 1000)
    )) == {"error": "invalid_request"}
    assert api.provider_frames == []


def test_policy_off_normalized_request_hash_is_unchanged():
    simple = {"model": "openai/gpt-4o-mini", "max_tokens": 20,
              "messages": [{"role": "user", "content": "Find evidence."}]}
    old = operations.validate_operation_request("openrouter.chat", simple)
    matched_operation, matched = operations.match_request(
        "POST", "https://openrouter.ai/api/v1/chat/completions",
        json.dumps(simple).encode(), {},
    )
    assert matched_operation == "openrouter.chat"
    assert contracts.document_hash(matched) == contracts.document_hash(old)
    assert "reasoning_capture_policy" not in matched


@pytest.mark.parametrize("full,kind,expected", [
    (False, "execute", False), (True, "execute", True),
    (True, "score", False),
])
def test_runner_passes_host_policy_hint_to_shim_only_for_active_lease(
    tmp_path, full, kind, expected,
):
    from lab_arena import scoring
    from tests.lab_arena.test_lab_arena_runner import (
        BridgingRuntime, TrajectoryApi, make_config, scoring_lease, valid_company,
    )

    issued = scoring_lease() if kind == "score" else lease()
    issued["kind"] = kind
    if full:
        issued["reasoning_capture_policy"] = shim.FULL_THINKING_POLICY
    api = TrajectoryApi([issued])
    output = (scoring.build_scoring_output("r1", [
        {"final_score": 71.0, "failure_reason": ""},
        {"final_score": 44.5, "failure_reason": ""},
    ]) if kind == "score" else {"companies": [valid_company(1)]})
    sandbox = BridgingRuntime(output=output, calls=0)
    (tmp_path / "work").mkdir()
    client = runner.Runner(make_config(tmp_path, api, sandbox))
    assert client.run_once() == 1
    environment = sandbox.specs[0].extra_environment
    if expected:
        assert environment[shim.REASONING_POLICY_ENV] == shim.FULL_THINKING_POLICY
    else:
        assert shim.REASONING_POLICY_ENV not in environment
