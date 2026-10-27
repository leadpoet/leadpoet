"""Header-only capture remains private, incomplete and outside paid read receipts."""
import json
import shutil
import tempfile
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from lab_arena import automatic_capture, lab_arena_checkpoint as checkpoint, runner, shim
from lab_arena import training_observations as obs
from lab_arena.api import create_app
from tests.lab_arena.test_lab_arena_runner import FakeApi, lease
from tests.lab_arena.training_input_test import CLAIM_HASH, TOKEN, service_for
from tests.lab_arena.training_observation_journal_test import _journal


def metadata():
    return {"url": "https://example.com/start", "status": 302,
            "content_type": "text/html", "redirect_url": "https://example.com/next",
            "retrieved_at": "2026-10-23T12:00:00Z", "source": "runtime_http_v1",
            "body_unconsumed": True}


def row():
    return {"id": 0, "kind": "http_response_metadata", "data": metadata(),
            "observed_at": "2026-10-23T12:00:00Z", "after_action_sequence": None,
            "depends_on": [], "privacy": {"redacted": False, "redactions": []}}


@pytest.mark.parametrize("field,value", [
    ("body_unconsumed", False), ("body_unconsumed", 1), ("source", "trusted"),
    ("status", True), ("status", 99), ("content_type", "a" * 257),
    ("retrieved_at", "2026-10-23"), ("url", "file:///private"),
    ("url", "https://example.com/\nprivate"), ("url", "https://example.com/" + "a" * 2000),
    ("redirect_url", "https://name:password@example.com/next"),
    ("redirect_url", "https://example.com/next?token=secret"),
    ("redirect_url", "https://example.com/next#secret"), ("redirect_url", "/next"),
])
def test_metadata_rejects_body_claims_unsafe_redirects_and_unbounded_fields(field, value):
    data = metadata()
    data[field] = value
    frame = {"schema_version": checkpoint.TRAINING_SCHEMA_VERSION,
             "control": checkpoint.TRAINING_CONTROL, "kind": "http_response_metadata",
             "depends_on": [], "data": data}
    with pytest.raises(ValueError):
        checkpoint.validate_training_frame(frame)
    with pytest.raises(obs.ObservationError, match="http_response_metadata_invalid"):
        obs.clean_agent_observation({**row(), "data": data})


@pytest.mark.parametrize("location,target", [
    ("../next?private=token#private", "https://example.com/next"),
    ("//name:password@other.example/next?token=private", "https://other.example/next"),
    ("javascript:alert(1)", None), ("file:///private", None),
    ("https://example.com/next\nprivate", None), ("a" * 2001, None),
])
def test_redirect_target_is_resolved_and_private_components_removed(location, target):
    assert automatic_capture._redirect_url("https://example.com/start/page", location) == target


def test_metadata_stays_incomplete_through_snapshot_and_journal(tmp_path):
    service, objects = service_for(tmp_path)
    client = TestClient(create_app(service))
    state, journal = _journal(client)
    buffer = state.worker_observations
    assert buffer.append_agent_observation(row()) == 0
    document = buffer.document()
    assert document["complete"] is False
    assert document["failures"] == [obs.OPTIONAL_CAPTURE_INCOMPLETE]
    assert document["agent_observation_count"] == 1
    assert document["response_receipt_count"] == document["observation_count"] == 0
    assert journal._prepare() and journal._send_pending()
    segments, prefix, failure = obs.read_journal_prefix(objects, "r1", "execute", 1, CLAIM_HASH)
    assert len(segments) == 1 and failure is None
    assert segments[0]["payload"]["failures"] == [obs.OPTIONAL_CAPTURE_INCOMPLETE]
    # A journal prefix still lacks the final snapshot and keeps a hard gap.
    assert prefix == {**document, "failures": ["capture_incomplete"]}
    assert not obs.optional_capture_only(prefix)
    service.handle_training_observations("r1", TOKEN, document)
    assert obs.read_snapshot(objects, "r1", "execute", 1, CLAIM_HASH)["payload"] == document
    forged = {**document, "complete": True, "failures": []}
    with pytest.raises(obs.ObservationError, match="completeness_invalid"):
        obs.build_snapshot("r1", "execute", forged, lease_generation=1, claim_request_hash=CLAIM_HASH)


def test_actual_unread_capture_crosses_worker_socket_without_receipt_or_extra_read(monkeypatch):
    state = runner.RunState(lease=lease(), lease_token="private-lease-token")
    api = FakeApi([])
    directory = Path(tempfile.mkdtemp(prefix="arena-metadata-", dir="/tmp"))
    server = runner.WorkerSocketServer(directory / "w.sock", api, state)
    server.start()
    monkeypatch.setenv(shim.WORKER_SOCKET_ENV, str(directory / "w.sock"))
    read_count = 0

    class Unread(httpx.SyncByteStream):
        def __iter__(self):
            nonlocal read_count
            read_count += 1
            raise AssertionError("capture must not consume a response")
            yield b""

    def handler(request):
        return httpx.Response(302, headers={"content-type": "text/html", "location":
            "https://name:password@other.example/next?token=private#fragment"}, stream=Unread())

    recorder = automatic_capture.install(checkpoint)
    try:
        with httpx.Client(transport=httpx.MockTransport(handler)) as client:
            with client.stream("GET", "https://name:password@example.com/start?api_key=private") as response:
                assert response.status_code == 302
                assert response.headers["location"].endswith("?token=private#fragment")
                assert not response.is_stream_consumed
        summary = recorder.close()
    finally:
        recorder.close()
        server.stop()
        shutil.rmtree(directory)
    assert read_count == 0
    assert not summary["complete"] and summary["counts"]["response_metadata_written"] == 1
    document = state.worker_observations.document()
    assert len(document["agent_observations"]) == 1
    saved = document["agent_observations"][0]
    assert saved["kind"] == "http_response_metadata"
    assert saved["data"]["redirect_url"] == "https://other.example/next"
    serialized = json.dumps(saved)
    assert "password" not in serialized and "name:" not in serialized and "private" not in serialized
    assert saved["privacy"] == {"redacted": True, "redactions": ["credential"]}
    assert document["complete"] is False
    assert state.action_sequence == len(state.calls) == len(api.provider_frames) == 0
    assert document["response_receipts"] == []
    assert not any(item["kind"] == "response_received" for item in document["agent_observations"])


def test_previous_observation_snapshots_remain_valid(tmp_path):
    _, objects = service_for(tmp_path)
    # v1 predates agent observations; v2 without the new kind stays unchanged.
    for generation, document in enumerate(({"observations": [], "observation_count": 0, "local_refusal_count": 0,
                      "complete": True, "failures": []}, obs.Buffer().document()), start=1):
        snapshot, raw = obs.build_snapshot("r1", "execute", document, lease_generation=generation,
                                          claim_request_hash=CLAIM_HASH)
        objects.put(obs.object_ref("r1", generation, CLAIM_HASH), raw)
        assert obs.read_snapshot(objects, "r1", "execute", generation, CLAIM_HASH) == snapshot
