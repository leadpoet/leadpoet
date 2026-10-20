"""Run-wide capture capacity preserves late evidence without widening events."""

import copy
import time

import pytest
from fastapi.testclient import TestClient

from lab_arena import api as api_module, contracts, runner, training_observations as obs
from lab_arena.api import create_app
from tests.lab_arena.training_input_test import CLAIM_HASH, TOKEN, service_for
from tests.lab_arena.training_observations_test import Objects, observation, payload


STAMP = "2026-10-20T10:00:00+00:00"


def agent_row(index, kind, data):
    return {"id": index, "kind": kind, "observed_at": STAMP,
            "after_action_sequence": None, "depends_on": [], "data": data,
            "privacy": {"redacted": False, "redactions": []}}


def web_row(index, body):
    raw = body.encode()
    return agent_row(index, "web_fetch", {
        "url": "https://example.com/page/%d" % index,
        "final_url": "https://example.com/page/%d" % index,
        "retrieved_at": STAMP, "status": 200, "content_type": "text/html",
        "body_sha256": contracts.sha256_bytes(raw), "body_bytes": len(raw),
        "stored_body_sha256": None, "stored_body_bytes": None,
        "body": body, "body_ref": None})


def ordinary_full_run():
    buffer = obs.Buffer()
    # Distinct HTML pages prevent body deduplication from hiding the old limit.
    page = '<html><a href="https://example.com/about">Company evidence</a></html>\n'
    for index in range(100):
        body = str(index) + page * 1600
        if index == 0:
            body += " Bearer " + TOKEN
        assert buffer.append_agent_observation(web_row(index, body), secrets=(TOKEN,)) == index
    # These responses and all task boundaries arrive after more than 8 MiB.
    for index in range(246):
        digest = contracts.sha256_bytes(("response%d" % index).encode())
        identity = contracts.sha256_bytes(("call%d" % index).encode())
        assert buffer.append_response_receipt({
            "ordinal": index, "action_sequence": index, "operation_id": "openrouter.chat",
            "call_identity": identity, "protocol": "http", "status": 200,
            "body_sha256": digest, "body_bytes": 20, "write_state": "socket_written",
            "socket_written_at": STAMP})
        index = len(buffer.agent_observations)
        assert buffer.append_agent_observation(agent_row(index, "response_received", {
            "call_identity": identity, "status": 200, "body_sha256": digest,
            "body_bytes": 20})) == index
    for name in ("after_discovery", "after_verification", "after_admission", "final", "training_capture_summary"):
        index = len(buffer.agent_observations)
        assert buffer.append_agent_observation(agent_row(index, "state_snapshot", {
            "name": name, "value": {"rows": [{"company": "Example", "source": "https://example.com"}]}})) == index
    for index in range(10):
        buffer.append(observation(action=246 + index, ordinal=246 + index))
    return buffer.document()


def test_full_ordinary_run_over_old_limit_uploads_all_late_evidence(tmp_path):
    document = ordinary_full_run()
    raw = contracts.canonical_json(document).encode()
    assert 8 * 1024 * 1024 < len(raw) < obs.MAX_OBJECT_BYTES
    assert document["complete"] and not document["failures"]
    assert document["observation_count"] == len(document["observations"]) == 10
    assert document["response_receipt_count"] == len(document["response_receipts"]) == 246
    assert document["agent_observation_count"] == len(document["agent_observations"]) == 351
    assert TOKEN.encode() not in raw
    first = document["agent_observations"][0]
    assert first["privacy"]["redacted"]
    assert first["data"]["stored_body_sha256"] != first["data"]["body_sha256"]
    service, objects = service_for(tmp_path)
    client = TestClient(create_app(service))
    api = runner.HttpArenaApiClient("http://localhost", client=client)
    start = time.monotonic()
    result = api.training_observations("r1", TOKEN, document)
    elapsed = time.monotonic() - start
    assert result["status"] == "stored"
    saved = obs.read_snapshot(objects, "r1", "execute", 1, CLAIM_HASH)
    assert saved["payload"] == document
    assert saved["payload_sha256"] == result["payload_sha256"]
    print("ordinary capture bytes=%d upload_seconds=%.3f" % (len(raw), elapsed))


def test_aggregate_boundary_and_overflow_keep_honest_counts(monkeypatch):
    row = agent_row(0, "state_snapshot", {"name": "boundary", "value": {"answer": "x" * 100}})
    size = len(obs._canonical(obs.clean_agent_observation(row))) + 1
    monkeypatch.setattr(obs, "MAX_OBJECT_BYTES", size + 65_536)
    buffer = obs.Buffer()
    assert buffer.append_agent_observation(row) == 0
    assert buffer.size_bytes == size and buffer.document()["complete"]
    later = copy.deepcopy(row)
    later["id"] = 1
    assert buffer.append_agent_observation(later) is None
    document = buffer.document()
    assert document["agent_observation_count"] == 2
    assert len(document["agent_observations"]) == 1
    assert not document["complete"] and document["failures"] == ["capture_incomplete"]
    snapshot, _ = obs.build_snapshot("r1", "execute", document, lease_generation=1, claim_request_hash=CLAIM_HASH)
    assert not snapshot["payload"]["complete"]


def test_event_web_and_snapshot_limits_still_fail_closed(monkeypatch):
    assert obs.MAX_EVENT_BYTES == 8 * 1024 * 1024
    assert obs.MAX_OBJECT_BYTES == 256 * 1024 * 1024
    assert obs.MAX_WEB_BODY_BYTES == 6 * 1024 * 1024
    row = observation()
    row["request"]["admitted_parameters"] = {"text": "x" * obs.MAX_EVENT_BYTES}
    with pytest.raises(obs.ObservationError, match="size_limit"):
        obs.clean_observation(row)
    with pytest.raises(obs.ObservationError, match="web_fetch_invalid"):
        obs.clean_agent_observation(web_row(0, "x" * (obs.MAX_WEB_BODY_BYTES + 1)))
    document = obs.Buffer().document()
    snapshot, compressed = obs.build_snapshot("r1", "execute", document, lease_generation=1, claim_request_hash=CLAIM_HASH)
    monkeypatch.setattr(obs, "MAX_OBJECT_BYTES", len(obs._canonical(snapshot)) - 1)
    with pytest.raises(obs.ObservationError, match="size_limit"):
        obs.build_snapshot("r1", "execute", document, lease_generation=1, claim_request_hash=CLAIM_HASH)
    objects = Objects()
    objects.put(obs.object_ref("r1", 1, CLAIM_HASH), compressed)
    with pytest.raises(obs.ObservationError, match="size_limit"):
        obs.read_snapshot(objects, "r1", "execute", 1, CLAIM_HASH)



def test_compressed_object_limit_stays_bounded(monkeypatch):
    document = obs.Buffer().document()
    snapshot, compressed = obs.build_snapshot("r1", "execute", document,
        lease_generation=1, claim_request_hash=CLAIM_HASH)
    monkeypatch.setattr(obs, "MAX_COMPRESSED_BYTES", len(compressed) - 1)
    with pytest.raises(obs.ObservationError, match="size_limit"):
        obs.build_snapshot("r1", "execute", document,
            lease_generation=1, claim_request_hash=CLAIM_HASH)
    class BoundedObjects:
        def get_bounded(self, ref, max_bytes):
            assert max_bytes == len(compressed) - 1
            raise ValueError("object exceeds read bound")
    with pytest.raises(obs.ObservationError, match="observations_unavailable"):
        obs.read_snapshot(BoundedObjects(), "r1", "execute", 1, CLAIM_HASH)


def test_legacy_and_prior_v2_snapshots_still_read_exactly():
    for document in (payload(observation()), obs.Buffer().document()):
        snapshot, compressed = obs.build_snapshot("r1", "execute", document, lease_generation=1, claim_request_hash=CLAIM_HASH)
        objects = Objects()
        objects.put(obs.object_ref("r1", 1, CLAIM_HASH), compressed)
        assert obs.read_snapshot(objects, "r1", "execute", 1, CLAIM_HASH) == snapshot


@pytest.mark.parametrize("changes,token,status", [({}, "b" * 64, 401), ({"status": "accepted"}, TOKEN, 409), ({"lease_expires_at": "2020-01-01T00:00:00Z"}, TOKEN, 409)])
def test_large_body_is_not_read_before_active_lease_authority(tmp_path, monkeypatch, changes, token, status):
    service, _ = service_for(tmp_path, **changes)
    async def unexpected_read(*args, **kwargs):
        pytest.fail("an unauthorized upload body was read")
    monkeypatch.setattr(api_module, "_read_json", unexpected_read)
    response = TestClient(create_app(service)).post(
        "/arena/v1/runs/r1/training-observations", content=b"not JSON",
        headers={"x-lab-arena-lease": token})
    assert response.status_code == status


def test_upload_authority_is_rechecked_after_body_read(tmp_path, monkeypatch):
    service, _ = service_for(tmp_path)
    read_json = api_module._read_json
    async def revoke_after_read(*args, **kwargs):
        document = await read_json(*args, **kwargs)
        service._store.get_run("r1")["status"] = "accepted"
        return document
    monkeypatch.setattr(api_module, "_read_json", revoke_after_read)
    response = TestClient(create_app(service)).post(
        "/arena/v1/runs/r1/training-observations", json=obs.Buffer().document(),
        headers={"x-lab-arena-lease": TOKEN})
    assert response.status_code == 409 and response.json()["code"] == "lease_inactive"
