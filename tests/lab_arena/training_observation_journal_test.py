"""Crash recovery retains only a verified, partial worker-observation prefix."""

import base64
import copy
import hashlib
import time
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from fastapi.testclient import TestClient
import pytest

from lab_arena import contracts, runner, training_observations as obs
from lab_arena.api import create_app
from lab_arena.service import S3ObjectStore
from lab_arena.training_export import assemble_example
from tests.lab_arena.training_export_test import fixture
from tests.lab_arena.training_input_test import CLAIM_HASH, NOW, TOKEN, service_for
from tests.lab_arena.training_observations_test import IDENTITY, observation
from tests.lab_arena.test_lab_arena_runner import BridgingRuntime, FakeApi, lease, make_config, valid_company
from tests.lab_arena.claim_recovery_test import claim_service, original_claim


LEASE = {"run_id": "r1", "kind": "execute", "lease_generation": 1,
         "claim_request_hash": CLAIM_HASH}
ROUTE = "/arena/v1/runs/r1/training-observation-segments"
HEADERS = {"x-lab-arena-lease": TOKEN}


def _api(client):
    def send(run_id, token, document):
        response = client.post(ROUTE, headers={"x-lab-arena-lease": token}, json=document)
        return response.json()
    return SimpleNamespace(training_observation_segment=send)


def _journal(client):
    state = runner.RunState(lease=LEASE, lease_token=TOKEN)
    journal = runner._ObservationJournal(_api(client), LEASE, TOKEN, state)
    return state, journal


def test_crashed_runner_recovers_only_persisted_prefix_and_keeps_final_authority(tmp_path):
    service, objects = service_for(tmp_path)
    client = TestClient(create_app(service))
    state, journal = _journal(client)
    with state.lock:
        state.worker_observations.append(observation())
    assert journal._prepare() and journal._send_pending()
    # The worker dies before final upload; a later in-memory event is lost.
    with state.lock:
        state.worker_observations.append(observation(ordinal=1, action=1))
    segments, partial, failure = obs.read_journal_prefix(objects, "r1", "execute", 1, CLAIM_HASH)
    assert failure is None and len(segments) == 1
    assert partial["complete"] is False and len(partial["observations"]) == 1
    assert partial["failures"] == ["capture_incomplete"]
    final = state.worker_observations.document()
    service.handle_training_observations("r1", TOKEN, final)
    assert obs.read_snapshot(objects, "r1", "execute", 1, CLAIM_HASH)["payload"] == final


def test_ambiguous_write_retries_identical_segment_without_duplicate(tmp_path):
    service, objects = service_for(tmp_path)
    calls = []
    def ambiguous(run_id, token, document):
        calls.append(copy.deepcopy(document))
        result = service.handle_training_observation_segment(run_id, token, document)
        if len(calls) == 1:
            raise TimeoutError("reply lost after object commit")
        return result
    state = runner.RunState(lease=LEASE, lease_token=TOKEN)
    journal = runner._ObservationJournal(SimpleNamespace(training_observation_segment=ambiguous), LEASE, TOKEN, state)
    with state.lock:
        state.worker_observations.append(observation())
    assert journal._prepare()
    assert not journal._send_pending()
    assert journal.number == 0 and journal.pending is not None
    assert journal._send_pending()
    assert calls[0] == calls[1] and journal.number == 1
    assert len(list((tmp_path / "objects").rglob("segment-*.gz"))) == 1
    assert obs.read_journal_prefix(objects, "r1", "execute", 1, CLAIM_HASH)[1]["observation_count"] == 1


def test_background_uploader_keeps_capture_failures_out_of_run_result(tmp_path, monkeypatch):
    service, objects = service_for(tmp_path)
    client = TestClient(create_app(service))
    state, journal = _journal(client)
    monkeypatch.setattr(runner._ObservationJournal, "FLUSH_SECONDS", 0.01)
    with state.lock:
        state.worker_observations.append(observation())
    journal.start()
    deadline = time.monotonic() + 2
    while journal.number == 0 and time.monotonic() < deadline:
        time.sleep(0.01)
    journal.stop()
    assert journal.number == 1
    assert obs.read_journal_prefix(objects, "r1", "execute", 1, CLAIM_HASH)[1] is not None
    assert state.worker_observations.document()["complete"]

    failing = runner._ObservationJournal(
        SimpleNamespace(training_observation_segment=lambda *_: (_ for _ in ()).throw(OSError("store down"))),
        LEASE, TOKEN, state)
    assert failing._prepare() and not failing._send_pending()
    assert state.worker_observations.document()["complete"]
    assert failing.pending is not None and failing.number == 0


def test_journal_thread_start_failure_does_not_change_accepted_output(tmp_path, monkeypatch):
    claimed = lease()
    api = FakeApi([claimed])
    (tmp_path / "work").mkdir()
    monkeypatch.setattr(runner._ObservationJournal, "start",
                        lambda self: (_ for _ in ()).throw(RuntimeError("thread unavailable")))
    result = runner.Runner(make_config(
        tmp_path, api, BridgingRuntime(output={"companies": [valid_company(1)]}, calls=0))).run_once()
    assert result == 1
    assert api.completions[0]["body"]["result"]["terminal_status"] == "accepted"
    assert len(api.completions[0]["body"]["output"]["companies"]) == 1


def test_runner_lease_carries_exact_persisted_claim_hash():
    now = datetime(2026, 9, 2, 1, tzinfo=timezone.utc)
    service, store = claim_service(lambda: now)
    signed = original_claim(now)
    claimed = service.handle_claim(signed)
    assert claimed["claim_request_hash"] == store.run["claim_request_hash"]
    assert claimed["claim_request_hash"] == contracts.request_bytes_hash(signed)


def test_object_store_distinguishes_missing_key_from_missing_bucket():
    class Missing(Exception):
        def __init__(self, code):
            self.response = {"Error": {"Code": code},
                             "ResponseMetadata": {"HTTPStatusCode": 404}}

    class Client:
        code = "NoSuchKey"
        def get_object(self, **_kwargs):
            raise Missing(self.code)

    client = Client()
    objects = S3ObjectStore("private", client=client)
    assert objects.get_optional_bounded("ref", 1) is None
    client.code = "NoSuchBucket"
    with pytest.raises(Missing):
        objects.get_optional_bounded("ref", 1)


def test_large_receipt_count_is_not_cut_off_at_old_eight_row_batches(tmp_path):
    service, objects = service_for(tmp_path)
    state = runner.RunState(lease=LEASE, lease_token=TOKEN)
    journal = runner._ObservationJournal(
        SimpleNamespace(training_observation_segment=service.handle_training_observation_segment),
        LEASE, TOKEN, state)
    with state.lock:
        for ordinal in range(4200):
            assert state.worker_observations.append_response_receipt({
                "ordinal": ordinal, "action_sequence": ordinal,
                "operation_id": "openrouter.chat", "call_identity": IDENTITY,
                "protocol": "frame", "status": 200,
                "body_sha256": "sha256:" + "0" * 64, "body_bytes": 0,
                "write_state": "socket_written", "socket_written_at": NOW.isoformat()})
    while journal.receipt_cursor < 4200:
        assert journal._prepare() and journal._send_pending()
    segments, partial, failure = obs.read_journal_prefix(objects, "r1", "execute", 1, CLAIM_HASH)
    assert failure is None and len(segments) < obs.MAX_SEGMENTS
    assert len(partial["response_receipts"]) == 4200


def test_conflict_wrong_lease_expiry_and_capacity_are_rejected(tmp_path):
    service, objects = service_for(tmp_path)
    client = TestClient(create_app(service))
    state, journal = _journal(client)
    with state.lock:
        state.worker_observations.append(observation())
    assert journal._prepare()
    document = copy.deepcopy(journal.pending[0])
    assert client.post(ROUTE, headers=HEADERS, json=document).status_code == 200
    changed = copy.deepcopy(document)
    changed["payload"]["observation_count"] = 2
    changed["cumulative_bytes"] = len(contracts.canonical_json(changed["payload"]).encode())
    assert client.post(ROUTE, headers=HEADERS, json=changed).status_code == 409
    regressed = {"observation_from": 1, "observations": [], "receipt_from": 0,
                 "response_receipts": [], "receipt_updates": [], "agent_from": 0,
                 "agent_observations": [], "observation_count": 0,
                 "local_refusal_count": 0, "response_receipt_count": 0,
                 "agent_observation_count": 0, "failures": []}
    second = {"number": 1, "previous_sha256": service.handle_training_observation_segment(
        "r1", TOKEN, document)["segment_sha256"], "payload": regressed,
        "cumulative_bytes": document["cumulative_bytes"]
        + len(contracts.canonical_json(regressed).encode())}
    assert client.post(ROUTE, headers=HEADERS, json=second).status_code == 409
    assert client.post(ROUTE, headers={"x-lab-arena-lease": "b" * 64}, json=document).status_code == 401
    service._store.get_run("r1")["lease_expires_at"] = (NOW - timedelta(seconds=181)).isoformat()
    assert client.post(ROUTE, headers=HEADERS, json=document).status_code == 409
    too_large = copy.deepcopy(document)
    too_large["cumulative_bytes"] = obs.MAX_OBJECT_BYTES + 1
    service._store.get_run("r1")["lease_expires_at"] = (NOW + timedelta(minutes=2)).isoformat()
    assert client.post(ROUTE, headers=HEADERS, json=too_large).status_code == 400
    assert len(list((tmp_path / "objects").rglob("segment-*.gz"))) == 1


def test_receipt_transition_and_missing_or_corrupt_segment_remain_incomplete(tmp_path):
    service, objects = service_for(tmp_path)
    client = TestClient(create_app(service))
    state, journal = _journal(client)
    row = observation(kind="nonstandard_response", identity=IDENTITY)
    raw = base64.b64decode(row["outcome"]["response"]["body_b64"])
    with state.lock:
        state.worker_observations.append(row)
        state.worker_observations.append_response_receipt({
            "ordinal": 0, "action_sequence": 0, "operation_id": "openrouter.chat",
            "call_identity": IDENTITY, "protocol": "frame", "status": 502,
            "body_sha256": "sha256:" + hashlib.sha256(raw).hexdigest(),
            "body_bytes": len(raw), "write_state": "prepared", "socket_written_at": None})
    assert journal._prepare() and journal._send_pending()
    with state.lock:
        state.worker_observations.mark_socket_write(0, written_at=NOW.isoformat())
    assert journal._prepare() and journal._send_pending()
    segments, partial, failure = obs.read_journal_prefix(objects, "r1", "execute", 1, CLAIM_HASH)
    assert failure is None and len(segments) == 2
    assert partial["response_receipts"][0]["write_state"] == "socket_written"
    objects._path(obs.segment_ref("r1", 1, CLAIM_HASH, 1)).unlink()
    segments, partial, failure = obs.read_journal_prefix(objects, "r1", "execute", 1, CLAIM_HASH)
    assert len(segments) == 1 and failure is None
    assert partial["response_receipts"][0]["write_state"] == "prepared"
    updated = copy.deepcopy(partial["response_receipts"][0])
    updated.update(write_state="socket_written", socket_written_at=NOW.isoformat())
    invalid = {"observation_from": 1, "observations": [], "receipt_from": 1,
               "response_receipts": [], "receipt_updates": [updated, updated],
               "agent_from": 0, "agent_observations": [], "observation_count": 1,
               "local_refusal_count": 0, "response_receipt_count": 1,
               "agent_observation_count": 0, "failures": []}
    _, bad_bytes = obs.build_segment(
        "r1", "execute", invalid, lease_generation=1, claim_request_hash=CLAIM_HASH,
        number=1, previous_sha256=segments[0]["segment_sha256"],
        cumulative_bytes=segments[0]["cumulative_bytes"]
        + len(contracts.canonical_json(invalid).encode()))
    objects.put(obs.segment_ref("r1", 1, CLAIM_HASH, 1), bad_bytes)
    segments, partial, failure = obs.read_journal_prefix(objects, "r1", "execute", 1, CLAIM_HASH)
    assert len(segments) == 1 and failure == "journal_transition_invalid"
    assert partial["response_receipts"][0]["write_state"] == "prepared"
    # A corrupt present segment is not mistaken for a normal end of journal.
    objects._path(obs.segment_ref("r1", 1, CLAIM_HASH, 1)).unlink()
    objects.put(obs.segment_ref("r1", 1, CLAIM_HASH, 1), b"corrupt")
    segments, partial, failure = obs.read_journal_prefix(objects, "r1", "execute", 1, CLAIM_HASH)
    assert len(segments) == 1 and failure == "journal_unavailable"


def test_gateway_redacts_segment_and_export_keeps_partial_separate(tmp_path):
    service, objects = service_for(tmp_path)
    client = TestClient(create_app(service))
    state, journal = _journal(client)
    row = observation()
    row["request"]["admitted_parameters"]["api_key"] = TOKEN
    with state.lock:
        state.worker_observations.append(row, secrets=(TOKEN,))
    assert journal._prepare() and journal._send_pending()
    partial = obs.read_journal_prefix(objects, "r1", "execute", 1, CLAIM_HASH)[1]
    assert TOKEN not in contracts.canonical_json(partial)
    assert partial["observations"][0]["privacy"]["redacted"]

    values = fixture()
    old = values["objects"]
    old.get_optional_bounded = lambda ref, maximum: old.get_bounded(ref, maximum) if ref in old.values else None
    run_id = "execute:1"
    payload = {"observation_from": 0, "observations": partial["observations"],
               "receipt_from": 0, "response_receipts": [], "receipt_updates": [],
               "agent_from": 0, "agent_observations": [], "observation_count": 1,
               "local_refusal_count": 0, "response_receipt_count": 0,
               "agent_observation_count": 0, "failures": []}
    segment, data = obs.build_segment(run_id, "execute", payload, lease_generation=1,
                                      claim_request_hash=CLAIM_HASH, number=0,
                                      previous_sha256=None,
                                      cumulative_bytes=len(contracts.canonical_json(payload).encode()))
    old.put(obs.segment_ref(run_id, 1, CLAIM_HASH, 0), data)
    result = assemble_example(**values)
    assert not result["completeness"]["complete"]
    assert "worker_observations_incomplete" in result["completeness"]["incomplete_reasons"]
    assert run_id in result["partial_worker_observation_artifacts"]
    assert run_id not in result["worker_observation_artifacts"]
    assert result["judgment"]["per_icp_score"] == 60
