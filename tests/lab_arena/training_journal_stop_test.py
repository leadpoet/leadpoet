"""Orderly shutdown preserves a bounded journal tail without final authority."""

import copy
import threading
import time
from types import SimpleNamespace

import pytest

from lab_arena import contracts, runner, training_observations as obs
from tests.lab_arena.training_input_test import CLAIM_HASH, NOW, TOKEN, service_for
from tests.lab_arena.training_observations_test import IDENTITY, observation


LEASE = {"run_id": "r1", "kind": "execute", "lease_generation": 1,
         "claim_request_hash": CLAIM_HASH}


def journal_for(api):
    state = runner.RunState(lease=LEASE, lease_token=TOKEN)
    journal = runner._ObservationJournal(api, LEASE, TOKEN, state)
    return state, journal


def prefix(objects):
    segments, partial, error = obs.read_journal_prefix(objects, "r1", "execute", 1, CLAIM_HASH)
    assert error is None
    return segments, partial


def test_normal_stop_preserves_tail_when_both_final_snapshot_uploads_fail(tmp_path):
    service, objects = service_for(tmp_path)
    posts = []

    def unavailable(*args):
        posts.append(args)
        raise OSError("synthetic final endpoint failure")

    api = SimpleNamespace(training_observation_segment=service.handle_training_observation_segment,
                          training_observations=unavailable)
    state, journal = journal_for(api)
    state.worker_observations.append(observation())
    assert journal._prepare() and journal._send_pending()
    state.worker_observations.append(observation(ordinal=1, action=1))
    journal.start()
    journal.stop()
    result = runner._record_training_observations(SimpleNamespace(api=api), LEASE, TOKEN, state)
    assert result["status"] == "unavailable" and len(posts) == 2
    segments, partial = prefix(objects)
    assert len(segments) == 2 and len(partial["observations"]) == 2
    # A persisted tail is still not a terminal snapshot or complete ICP proof.
    assert partial["complete"] is False and partial["failures"] == ["capture_incomplete"]
    with pytest.raises(obs.ObservationError):
        obs.read_snapshot(objects, "r1", "execute", 1, CLAIM_HASH)


def test_normal_stop_preserves_prepared_to_written_receipt_transition(tmp_path):
    service, objects = service_for(tmp_path)
    state, journal = journal_for(SimpleNamespace(
        training_observation_segment=service.handle_training_observation_segment))
    assert state.worker_observations.append_response_receipt({
        "ordinal": 0, "action_sequence": 0, "operation_id": "openrouter.chat",
        "call_identity": IDENTITY, "protocol": "frame", "status": 200,
        "body_sha256": contracts.hash_bytes(b"saved response"), "body_bytes": 14,
        "write_state": "prepared", "socket_written_at": None})
    assert journal._prepare() and journal._send_pending()
    state.worker_observations.mark_socket_write(0, written_at=NOW.isoformat())
    journal.start()
    journal.stop()
    segments, partial = prefix(objects)
    assert len(segments) == 2
    assert partial["response_receipts"][0]["write_state"] == "socket_written"


def test_stop_retries_ambiguous_pending_bytes_without_duplicate_sequence(tmp_path):
    service, objects = service_for(tmp_path)
    requests = []

    def send(run, token, document):
        requests.append(copy.deepcopy(document))
        result = service.handle_training_observation_segment(run, token, document)
        if len(requests) == 1:
            raise TimeoutError("reply lost after object commit")
        return result

    state, journal = journal_for(SimpleNamespace(training_observation_segment=send))
    state.worker_observations.append(observation())
    journal.start()
    journal.stop()
    assert requests[0] == requests[1] and len(requests) == 2
    assert journal.number == 1 and journal.pending is None
    assert len(prefix(objects)[0]) == 1


def test_stop_hard_failure_keeps_two_attempt_bound_and_exact_pending_bytes():
    requests = []

    def fail(*args):
        requests.append(copy.deepcopy(args[2]))
        raise OSError("synthetic unavailable segment endpoint")

    state, journal = journal_for(SimpleNamespace(training_observation_segment=fail))
    state.worker_observations.append(observation())
    journal.start()
    journal.stop()
    assert len(requests) == 2 and requests[0] == requests[1]
    assert journal.number == 0 and journal.pending is not None
    assert state.worker_observations.document()["complete"] is True


def test_stop_waits_for_single_uploader_before_tail_send(tmp_path, monkeypatch):
    service, objects = service_for(tmp_path)
    entered, release = threading.Event(), threading.Event()
    requests, active = [], []
    maximum = [0]

    def send(run, token, document):
        active.append(threading.get_ident())
        maximum[0] = max(maximum[0], len(active))
        requests.append((threading.get_ident(), document["number"]))
        try:
            if len(requests) == 1:
                entered.set()
                assert release.wait(2)
            return service.handle_training_observation_segment(run, token, document)
        finally:
            active.pop()

    state, journal = journal_for(SimpleNamespace(training_observation_segment=send))
    monkeypatch.setattr(journal, "FLUSH_SECONDS", 0.001)
    state.worker_observations.append(observation())
    journal.start()
    assert entered.wait(2)
    state.worker_observations.append(observation(ordinal=1, action=1))
    stopper = threading.Thread(target=journal.stop)
    stopper.start()
    try:
        assert stopper.is_alive()
        release.set()
        stopper.join(2)
        assert not stopper.is_alive() and maximum[0] == 1
        assert requests == [(journal.thread.ident, 0), (journal.thread.ident, 1)]
        assert len(prefix(objects)[1]["observations"]) == 2
    finally:
        release.set()
        stopper.join(2)


def test_stop_attempt_bound_leaves_large_tail_explicitly_partial(tmp_path):
    service, objects = service_for(tmp_path)
    sent = []

    def send(run, token, document):
        sent.append(document["number"])
        return service.handle_training_observation_segment(run, token, document)

    state, journal = journal_for(SimpleNamespace(training_observation_segment=send))
    for number in range(193):
        state.worker_observations.append(observation(ordinal=number, action=number))
    journal.start()
    journal.stop()
    assert sent == [0, 1]
    assert len(prefix(objects)[1]["observations"]) == 128
    assert prefix(objects)[1]["complete"] is False
    deadline = journal.stop_deadline
    journal.stop()
    assert journal.stop_deadline == deadline and sent == [0, 1]


def test_stop_fixed_deadline_does_not_wait_for_misbehaving_background_api(monkeypatch):
    entered, release = threading.Event(), threading.Event()
    requests = []

    def send(*args):
        requests.append(args)
        entered.set()
        release.wait(2)
        raise OSError("synthetic blocked transport")

    state, journal = journal_for(SimpleNamespace(training_observation_segment=send))
    monkeypatch.setattr(journal, "FLUSH_SECONDS", 0.001)
    monkeypatch.setattr(journal, "STOP_TIMEOUT_SECONDS", 0.03)
    state.worker_observations.append(observation())
    journal.start()
    assert entered.wait(2)
    before = time.monotonic()
    journal.stop()
    try:
        assert time.monotonic() - before < 0.3
        with state.lock:
            state.worker_observations_frozen = True
        release.set()
        journal.thread.join(2)
        assert not journal.thread.is_alive() and len(requests) == 1
        assert journal.number == 0 and journal.pending is not None
    finally:
        release.set()
        journal.thread.join(2)


def test_periodic_prepare_cannot_send_after_stop_deadline_and_snapshot_freeze(monkeypatch):
    prepared, release = threading.Event(), threading.Event()
    sent = []
    state, journal = journal_for(SimpleNamespace(
        training_observation_segment=lambda *_: sent.append(state.worker_observations_frozen)))
    original = journal._prepare

    def prepare():
        result = original()
        prepared.set()
        assert release.wait(2)
        return result

    monkeypatch.setattr(journal, "_prepare", prepare)
    monkeypatch.setattr(journal, "FLUSH_SECONDS", 0.001)
    monkeypatch.setattr(journal, "STOP_TIMEOUT_SECONDS", 0.03)
    state.worker_observations.append(observation())
    journal.start()
    assert prepared.wait(2)
    journal.stop()
    try:
        with state.lock:
            state.worker_observations_frozen = True
        release.set()
        journal.thread.join(2)
        assert not journal.thread.is_alive() and not sent
        assert journal.number == 0 and journal.pending is not None
    finally:
        release.set()
        journal.thread.join(2)


@pytest.mark.parametrize("mode", ["unsupported", "not_started", "already_dead", "frozen"])
def test_stop_never_starts_a_replacement_or_sends_after_freeze(mode):
    requests = []
    api = SimpleNamespace(training_observation_segment=lambda *args: requests.append(args))
    if mode == "unsupported":
        api = SimpleNamespace()
    state, journal = journal_for(api)
    state.worker_observations.append(observation())
    if mode == "already_dead":
        journal.thread = threading.Thread(target=lambda: None)
        journal.thread.start()
        journal.thread.join()
    elif mode == "frozen":
        state.worker_observations_frozen = True
        journal.start()
    elif mode == "unsupported":
        journal.start()
    journal.stop()
    assert not requests and journal.number == 0
    assert journal.thread.ident is None or not journal.thread.is_alive()


def test_final_snapshot_remains_complete_authority_after_tail_flush(tmp_path):
    service, objects = service_for(tmp_path)
    api = SimpleNamespace(training_observation_segment=service.handle_training_observation_segment,
                          training_observations=service.handle_training_observations)
    state, journal = journal_for(api)
    state.worker_observations.append(observation())
    journal.start()
    journal.stop()
    summary = runner._record_training_observations(SimpleNamespace(api=api), LEASE, TOKEN, state)
    assert summary["status"] == "stored"
    assert obs.read_snapshot(objects, "r1", "execute", 1, CLAIM_HASH)["payload"]["complete"] is True
    assert prefix(objects)[1]["complete"] is False


def test_first_final_upload_failure_uses_existing_second_identical_post(tmp_path):
    service, objects = service_for(tmp_path)
    uploads = []

    def final(run, token, document):
        uploads.append(copy.deepcopy(document))
        if len(uploads) == 1:
            raise OSError("synthetic first POST failure")
        return service.handle_training_observations(run, token, document)

    api = SimpleNamespace(training_observation_segment=service.handle_training_observation_segment,
                          training_observations=final)
    state, journal = journal_for(api)
    state.worker_observations.append(observation())
    journal.start()
    journal.stop()
    summary = runner._record_training_observations(SimpleNamespace(api=api), LEASE, TOKEN, state)
    assert summary["status"] == "stored" and uploads[0] == uploads[1]
    assert len(uploads) == 2 and state.calls == [] and state.action_sequence == 0
    assert obs.read_snapshot(objects, "r1", "execute", 1, CLAIM_HASH)["payload"]["complete"] is True
