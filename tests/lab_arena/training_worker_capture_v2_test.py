"""Focused proof of cooperative training capture and worker socket receipts."""

import hashlib
import json
import shutil
import tempfile
from pathlib import Path

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
