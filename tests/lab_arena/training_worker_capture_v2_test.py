"""Focused proof of cooperative training capture and worker socket receipts."""

import copy
import hashlib
import json
import shutil
import socket
import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from lab_arena import lab_arena_checkpoint, runner, shim, training_observations
from lab_arena.api import create_app
from tests.lab_arena.test_lab_arena_runner import FakeApi, lease
from tests.lab_arena.training_input_test import CLAIM_HASH, TOKEN, service_for
from tests.lab_arena.training_observations_test import Objects, response


def test_worker_socket_receipt_web_text_state_and_lease_bound_snapshot(tmp_path, monkeypatch):
    body = json.dumps({"answer": "actual paid response"}).encode()
    document = response(200, json.loads(body), outcome="settled")
    state = runner.RunState(lease=lease(), lease_token="private-lease-token")
    api = FakeApi([], broker_documents=[document])
    socket_path = Path(tempfile.mkdtemp(prefix="arena-v2-", dir="/tmp")) / "w.sock"
    server = runner.WorkerSocketServer(socket_path, api, state)
    server.start()
    monkeypatch.setenv(shim.WORKER_SOCKET_ENV, str(socket_path))
    try:
        status, headers, returned = shim.dispatch(
            "openrouter.chat", {"model": "openai/gpt-4o-mini",
                                "messages": [{"role": "user", "content": "check this"}], "max_tokens": 20}, 1000,
        )
        assert (status, returned) == (200, body)
        receipt = state.worker_observations.response_receipts[0]
        assert receipt["write_state"] == "socket_written" and receipt["socket_written_at"]
        assert receipt["body_sha256"] == "sha256:" + hashlib.sha256(body).hexdigest()
        assert receipt["body_bytes"] == len(body)
        assert not state.worker_observations.observations  # no duplicate paid body
        identity = receipt["call_identity"]
        assert headers[shim.CALL_IDENTITY_HEADER] == identity

        acknowledged = lab_arena_checkpoint.record_response_received(
            call_identity=identity, status=status, body_sha256=receipt["body_sha256"],
            body_bytes=len(returned),
        )
        assert acknowledged == 0
        text = "Evidence from https://example.com/page says hello."
        first = lab_arena_checkpoint.record_web_fetch(
            url="https://example.com/page", final_url="https://example.com/page",
            status=200, content_type="text/html", body=text,
            depends_on=["agent:0"],
        )
        second = lab_arena_checkpoint.record_web_fetch(
            url="https://example.com/page", final_url="https://example.com/page",
            status=200, content_type="text/html", body=text,
            depends_on=["agent:1"],
        )
        selected = lab_arena_checkpoint.record_state_snapshot(
            "selected_sources", {"source": "https://example.com/page"},
            depends_on=["agent:1"],
        )
        assert (first, second, selected) == (1, 2, 3)
    finally:
        server.stop()
        shutil.rmtree(socket_path.parent)

    payload = state.worker_observations.document()
    assert payload["complete"] and payload["response_receipt_count"] == 1
    assert payload["agent_observation_count"] == 4
    assert payload["agent_observations"][1]["data"]["body"] == text
    assert payload["agent_observations"][2]["data"]["body_ref"] == 1
    assert payload["agent_observations"][2]["data"]["body"] is None
    assert payload["agent_observations"][3]["depends_on"] == ["agent:1"]
    assert state.action_sequence == len(state.calls) == len(api.provider_frames) == 1
    snapshot, compressed = training_observations.build_snapshot(
        "execute:1", "execute", payload, lease_generation=1,
        claim_request_hash=CLAIM_HASH, secrets=(state.lease_token,),
    )
    assert snapshot["schema_version"] == training_observations.SCHEMA_VERSION
    objects = Objects()
    objects.put(training_observations.object_ref("execute:1", 1, CLAIM_HASH), compressed)
    assert training_observations.read_snapshot(objects, "execute:1", "execute", 1, CLAIM_HASH) == snapshot


def test_unwritten_response_and_web_capture_loss_are_explicit(tmp_path, monkeypatch):
    state = runner.RunState(lease=lease(), lease_token="private-lease-token")
    socket_path = Path(tempfile.mkdtemp(prefix="arena-v2-", dir="/tmp")) / "w.sock"
    server = runner.WorkerSocketServer(socket_path, FakeApi([], broker_documents=[
        response(200, {"answer": "available"}, outcome="settled")
    ]), state)
    # Direct invocation prepares a response but cannot prove a socket write.
    server.handle_frame(shim.build_operation_frame(
        "openrouter.chat", {"model": "openai/gpt-4o-mini",
                            "messages": [{"role": "user", "content": "test"}], "max_tokens": 20}, 1000,
    ))
    assert state.worker_observations.document()["complete"] is False
    assert state.worker_observations.response_receipts[0]["write_state"] == "prepared"

    server.start()
    monkeypatch.setenv(shim.WORKER_SOCKET_ENV, str(socket_path))
    try:
        assert lab_arena_checkpoint.record_web_fetch(
            url="https://example.com/large", final_url="https://example.com/large",
            status=200, content_type="text/plain",
            body="x" * (lab_arena_checkpoint.MAX_TRAINING_WEB_BODY_BYTES + 1),
        ) is None
        assert lab_arena_checkpoint.record_web_fetch(
            url="https://example.com/escaped", final_url="https://example.com/escaped",
            status=200, content_type="text/plain", body="\x00" * 2_000_000,
        ) is None
    finally:
        server.stop()
        shutil.rmtree(socket_path.parent)
    rows = state.worker_observations.agent_observations
    assert len(rows) == 2 and all(row["kind"] == "capture_failure" and row["data"]["reason"] == "size_limit" for row in rows)
    assert not state.worker_observations.document()["complete"]


def test_old_worker_artifact_remains_readable():
    legacy = {"observations": [], "observation_count": 0, "local_refusal_count": 0,
              "complete": True, "failures": []}
    snapshot, compressed = training_observations.build_snapshot(
        "execute:1", "execute", legacy, lease_generation=1, claim_request_hash=CLAIM_HASH,
    )
    assert snapshot["schema_version"] == training_observations.LEGACY_SCHEMA_VERSION
    objects = Objects()
    objects.put(training_observations.object_ref("execute:1", 1, CLAIM_HASH), compressed)
    assert training_observations.read_snapshot(objects, "execute:1", "execute", 1, CLAIM_HASH) == snapshot


@pytest.mark.parametrize("body", [
    r'{\"url\":\"https://example.com/?api_key=secret,\"next\":\"data\",\"last\":\"more\"}',
    r'{\"url\":\"https://example.com/?api_key=[REDACTED],\"last\":\"more\"}',
    "https://example.com/?api_key=secret&keep=visible#;fragment",
])
def test_escaped_json_web_body_keeps_worker_hashes_through_upload_and_read(tmp_path, body):
    state = runner.RunState(lease=lease(), lease_token="private-lease-token")
    server = runner.WorkerSocketServer(tmp_path / "unused.sock", FakeApi([]), state)
    capture = json.dumps({
        "schema_version": lab_arena_checkpoint.TRAINING_SCHEMA_VERSION,
        "control": lab_arena_checkpoint.TRAINING_CONTROL, "kind": "web_fetch", "depends_on": [],
        "data": {"url": "https://example.com/", "final_url": "https://example.com/",
                 "retrieved_at": "2026-10-19T00:00:00Z", "status": 200,
                 "content_type": "text/html", "body": body},
    }).encode()
    assert json.loads(server._handle_training_control(capture))["observation_id"] == 0
    assert json.loads(server._handle_training_control(capture))["observation_id"] == 1
    payload = state.worker_observations.document()
    assert payload["complete"] and not state.calls
    stored = payload["agent_observations"][0]["data"]
    assert stored["body"] == body.replace("=secret", "=%5BREDACTED%5D")
    assert stored["stored_body_sha256"] == "sha256:" + hashlib.sha256(stored["body"].encode()).hexdigest()
    assert payload["agent_observations"][1]["data"]["body_ref"] == 0

    snapshot, compressed = training_observations.build_snapshot(
        "execute:1", "execute", payload, lease_generation=1, claim_request_hash=CLAIM_HASH)
    assert snapshot["payload"] == payload
    objects = Objects()
    objects.put(training_observations.object_ref("execute:1", 1, CLAIM_HASH), compressed)
    assert training_observations.read_snapshot(objects, "execute:1", "execute", 1, CLAIM_HASH) == snapshot

    for field, invalid in (("stored_body_sha256", "sha256:" + "f" * 64),
                           ("stored_body_bytes", stored["stored_body_bytes"] + 1),
                           ("body", stored["body"] + "tampered")):
        tampered = copy.deepcopy(payload)
        tampered["agent_observations"][0]["data"][field] = invalid
        with pytest.raises(training_observations.ObservationError, match="hash_invalid"):
            training_observations.build_snapshot("execute:1", "execute", tampered,
                lease_generation=1, claim_request_hash=CLAIM_HASH)


def test_web_privacy_and_false_agent_ack_cannot_prove_receipt(tmp_path, monkeypatch):
    state = runner.RunState(lease=lease(), lease_token="private-lease-token")
    socket_path = Path(tempfile.mkdtemp(prefix="arena-v2-", dir="/tmp")) / "w.sock"
    server = runner.WorkerSocketServer(socket_path, FakeApi([]), state)
    server.start()
    monkeypatch.setenv(shim.WORKER_SOCKET_ENV, str(socket_path))
    try:
        assert lab_arena_checkpoint.record_web_fetch(
            url="https://example.com/page?token=private-lease-token",
            final_url="https://example.com/page?token=private-lease-token",
            status=200, content_type="text/html",
            body="Bearer private-lease-token page text",
        ) == 0
        assert lab_arena_checkpoint.record_response_received(
            call_identity="sha256:" + "1" * 64, status=200, body=b"invented",
        ) is None
    finally:
        server.stop()
        shutil.rmtree(socket_path.parent)
    rows = state.worker_observations.agent_observations
    assert rows[0]["privacy"] == {"redacted": True, "redactions": ["credential"]}
    assert "private-lease-token" not in json.dumps(rows)
    assert not state.worker_observations.document()["complete"]


def test_v2_upload_is_bound_to_existing_lease_and_private_store(tmp_path):
    service, objects = service_for(tmp_path)
    client = TestClient(create_app(service))
    payload = training_observations.Buffer().document()
    route = "/arena/v1/runs/r1/training-observations"
    assert client.post(route, json=payload).status_code == 401
    uploaded = client.post(route, json=payload, headers={"x-lab-arena-lease": TOKEN})
    assert uploaded.status_code == 200
    assert uploaded.json()["schema_version"] == training_observations.SCHEMA_VERSION
    saved = training_observations.read_snapshot(objects, "r1", "execute", 1, CLAIM_HASH)
    assert saved["payload"] == payload


def _exchange_frame(path, raw, *, extended):
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
        connection.settimeout(3)
        connection.connect(str(path))
        declared = len(raw) | (0x80000000 if extended else 0)
        connection.sendall(declared.to_bytes(4, "big") + raw)

        def receive(size):
            result = b""
            while len(result) < size:
                part = connection.recv(size - len(result))
                assert part, "worker closed before returning its response"
                result += part
            return result

        return json.loads(receive(int.from_bytes(receive(4), "big")))


def test_capture_prefix_cannot_dispatch_other_work_or_bypass_capture_budget(tmp_path, monkeypatch):
    state = runner.RunState(lease=lease(), lease_token="private-lease-token")
    api = FakeApi([], broker_documents=[response(200, {"answer": "available"}, outcome="settled")])
    socket_path = Path(tempfile.mkdtemp(prefix="arena-v2-", dir="/tmp")) / "w.sock"
    server = runner.WorkerSocketServer(socket_path, api, state)
    server.start()
    try:
        operation = shim.build_operation_frame("openrouter.chat", {
            "model": "openai/gpt-4o-mini", "messages": [{"role": "user", "content": "test"}],
            "max_tokens": 20,
        }, 1000)
        assert _exchange_frame(socket_path, operation, extended=True) == {"recorded": False}
        quota = json.dumps(lab_arena_checkpoint.QUOTA_CONTROL_FRAME).encode()
        assert _exchange_frame(socket_path, quota, extended=True) == {"recorded": False}
        assert state.action_sequence == len(api.provider_frames) == 0
        assert _exchange_frame(socket_path, operation, extended=False)["status"] == 200
        assert state.action_sequence == len(api.provider_frames) == 1

        monkeypatch.setattr(runner, "MAX_TRAINING_CONTROL_FRAMES", 0)
        capture = json.dumps({
            "schema_version": lab_arena_checkpoint.TRAINING_SCHEMA_VERSION,
            "control": lab_arena_checkpoint.TRAINING_CONTROL,
            "kind": "state_snapshot", "depends_on": [], "data": {"name": "safe", "value": {}},
        }).encode()
        assert _exchange_frame(socket_path, capture, extended=False) == {"error": "invalid_frame"}
        assert _exchange_frame(socket_path, capture, extended=True) == {"recorded": False}
        assert not state.worker_observations.agent_observations
        assert state.training_control_frames == 2
    finally:
        server.stop()
        shutil.rmtree(socket_path.parent)

    # Large capture frames do not enlarge the ordinary operation contract.
    with pytest.raises(shim.OperationFrameError, match="frame_too_large"):
        shim.decode_operation_frame(b" " * (shim.MAX_FRAME_BYTES + 1))


@pytest.mark.parametrize("field", ["body_bytes", "stored_body_bytes"])
def test_web_body_ref_requires_matching_original_and_stored_sizes(tmp_path, field):
    state = runner.RunState(lease=lease(), lease_token="private-lease-token")
    server = runner.WorkerSocketServer(tmp_path / "unused.sock", FakeApi([]), state)
    capture = json.dumps({
        "schema_version": lab_arena_checkpoint.TRAINING_SCHEMA_VERSION,
        "control": lab_arena_checkpoint.TRAINING_CONTROL, "kind": "web_fetch", "depends_on": [],
        "data": {"url": "https://example.com/", "final_url": "https://example.com/",
                 "retrieved_at": "2026-10-19T00:00:00Z", "status": 200,
                 "content_type": "text/plain", "body": "same body"},
    }).encode()
    assert json.loads(server._handle_training_control(capture))["observation_id"] == 0
    assert json.loads(server._handle_training_control(capture))["observation_id"] == 1
    payload = state.worker_observations.document()
    assert payload["agent_observations"][1]["data"]["body_ref"] == 0
    payload["agent_observations"][1]["data"][field] += 1
    with pytest.raises(training_observations.ObservationError, match="web_body_ref_invalid"):
        training_observations.build_snapshot("execute:1", "execute", payload,
            lease_generation=1, claim_request_hash=CLAIM_HASH)


def test_invalid_sdk_values_and_deep_extended_json_leave_explicit_capture_gaps(tmp_path, monkeypatch):
    state = runner.RunState(lease=lease(), lease_token="private-lease-token")
    socket_path = Path(tempfile.mkdtemp(prefix="arena-v2-", dir="/tmp")) / "w.sock"
    server = runner.WorkerSocketServer(socket_path, FakeApi([]), state)
    server.start()
    monkeypatch.setenv(shim.WORKER_SOCKET_ENV, str(socket_path))
    try:
        assert lab_arena_checkpoint.record_state_snapshot("safe", {}, depends_on=1) is None
        nested = []
        for _ in range(1100):
            nested = [nested]
        assert lab_arena_checkpoint.record_state_snapshot("deep", nested) is None
        malformed = b"[" * 1100 + b"0" + b"]" * 1100
        assert _exchange_frame(socket_path, malformed, extended=True) == {"recorded": False}
    finally:
        server.stop()
        shutil.rmtree(socket_path.parent)
    payload = state.worker_observations.document()
    assert not payload["complete"]
    assert [row["data"] for row in payload["agent_observations"]] == [
        {"category": "state_snapshot", "reason": "invalid_value"},
        {"category": "state_snapshot", "reason": "invalid_value"},
    ]
    assert all(row["depends_on"] == [] for row in payload["agent_observations"])
