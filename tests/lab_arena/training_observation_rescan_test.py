"""The gateway scans each streamed training row for secrets once, not again."""

from fastapi.testclient import TestClient

from lab_arena import contracts, training_observations as obs
from lab_arena.api import create_app
from tests.lab_arena.training_input_test import CLAIM_HASH, TOKEN, service_for
from tests.lab_arena.training_observation_journal_test import _journal
from tests.lab_arena.training_observations_test import observation


def _count(monkeypatch, name):
    calls = []
    original = getattr(obs, name)

    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(obs, name, counted)
    return calls


def test_final_upload_scans_only_rows_not_already_stored(tmp_path, monkeypatch):
    service, objects = service_for(tmp_path)
    state, journal = _journal(TestClient(create_app(service)))
    with state.lock:
        state.worker_observations.append(observation())
    assert journal._prepare() and journal._send_pending()

    with state.lock:
        state.worker_observations.append(observation(ordinal=1, action=1))
    final = state.worker_observations.document()
    # A row the segments never held still gets the full scan with the lease secret.
    final["observations"][1]["request"]["admitted_parameters"]["api_key"] = TOKEN
    expected = obs._validate_payload(final, (TOKEN,))

    scans = _count(monkeypatch, "clean_observation")
    service.handle_training_observations("r1", TOKEN, final)
    upload_scans = len(scans)

    stored = obs.read_snapshot(objects, "r1", "execute", 1, CLAIM_HASH)["payload"]
    assert stored == expected
    assert TOKEN not in contracts.canonical_json(stored)
    assert stored["observations"][1]["privacy"]["redacted"]
    assert upload_scans == 1


def test_segment_write_does_not_rescan_the_previous_segment(tmp_path, monkeypatch):
    service, objects = service_for(tmp_path)
    state, journal = _journal(TestClient(create_app(service)))
    with state.lock:
        state.worker_observations.append(observation())
    assert journal._prepare() and journal._send_pending()

    with state.lock:
        state.worker_observations.append(observation(ordinal=1, action=1))
    assert journal._prepare()
    scans = _count(monkeypatch, "_clean_segment_payload")
    assert journal._send_pending()

    # The runner scanned in _prepare; the gateway scans the new segment only.
    assert len(scans) == 1
    segments, partial, failure = obs.read_journal_prefix(objects, "r1", "execute", 1, CLAIM_HASH)
    assert failure is None and len(segments) == 2
    assert len(partial["observations"]) == 2
