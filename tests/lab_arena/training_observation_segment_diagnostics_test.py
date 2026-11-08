"""Segment failures name the storage phase without exposing captured content."""
import copy

from fastapi.testclient import TestClient
import pytest

from lab_arena import service as arena_service, training_observations as obs
from lab_arena.api import create_app
from lab_arena.contracts import ArenaContractError
from tests.lab_arena.training_input_test import TOKEN, service_for
from tests.lab_arena.training_observation_journal_test import HEADERS, ROUTE, _journal
from tests.lab_arena.training_observations_test import observation


PRIVATE = "synthetic-private-exception-payload-header"
UNAVAILABLE = {"status": "rejected", "code": "training_observation_segment_unavailable"}


def prepared(tmp_path, *, second=False):
    service, objects = service_for(tmp_path)
    client = TestClient(create_app(service))
    state, journal = _journal(client)
    state.worker_observations.append(observation())
    assert journal._prepare()
    if second:
        assert journal._send_pending()
        state.worker_observations.append(observation(ordinal=1, action=1))
        assert journal._prepare()
    return service, objects, client, copy.deepcopy(journal.pending[0])


def diagnostic(capsys):
    messages = capsys.readouterr().err.splitlines()
    assert len(messages) == 1
    message = messages[0]
    assert "run_id=r1" in message and PRIVATE not in message and TOKEN not in message
    return message


@pytest.mark.parametrize("stage,exception_class,cause_class", [
    ("store_missing", "NoneType", "none"),
    ("previous_read", "ObservationError", "OSError"),
    ("put", "OSError", "none"),
])
def test_native_storage_failure_keeps_503_and_logs_only_safe_identity(
        tmp_path, monkeypatch, capsys, stage, exception_class, cause_class):
    service, objects, client, document = prepared(tmp_path, second=stage == "previous_read")
    calls = []

    def fail(*args, **kwargs):
        calls.append(True)
        raise OSError(PRIVATE + " " + TOKEN)

    if stage == "store_missing":
        service._config.training_object_store = None
    else:
        monkeypatch.setattr(objects, "get_optional_bounded" if stage == "previous_read" else "put", fail)
    capsys.readouterr()
    response = client.post(ROUTE, headers=HEADERS, json=document)
    assert response.status_code == 503 and response.json() == UNAVAILABLE
    assert len(calls) == (0 if stage == "store_missing" else 1)  # No new retry.
    message = diagnostic(capsys)
    assert "segment=" + str(document["number"]) in message
    assert "stage=" + stage in message
    assert "error_class=" + exception_class in message
    assert "cause_class=" + cause_class in message
    assert "observation_reason=" + ("journal_unavailable" if stage == "previous_read" else "none") in message


@pytest.mark.parametrize("reason", [
    "journal_store_unavailable", "segment_size_limit", "journal_invalid",
    "journal_unavailable", PRIVATE,
])
def test_only_fixed_observation_reasons_are_logged(tmp_path, monkeypatch, capsys, reason):
    _service, _objects, client, document = prepared(tmp_path, second=True)

    def refuse(*args, **kwargs):
        raise obs.ObservationError(reason)

    monkeypatch.setattr(obs, "read_segment", refuse)
    capsys.readouterr()
    response = client.post(ROUTE, headers=HEADERS, json=document)
    assert response.status_code == 503 and response.json() == UNAVAILABLE
    message = diagnostic(capsys)
    assert "observation_reason=" + ("none" if reason == PRIVATE else reason) in message


def test_diagnostic_stream_failure_does_not_change_the_existing_503(tmp_path, monkeypatch):
    service, _objects, client, document = prepared(tmp_path)
    service._config.training_object_store = None

    def fail(*args, **kwargs):
        raise RuntimeError(PRIVATE)

    monkeypatch.setattr(arena_service.sys.stderr, "write", fail)
    response = client.post(ROUTE, headers=HEADERS, json=document)
    assert response.status_code == 503 and response.json() == UNAVAILABLE


def test_read_logs_only_the_immediate_known_wrapper_context(tmp_path, monkeypatch, capsys):
    _service, objects, client, document = prepared(tmp_path, second=True)

    def fail(*args, **kwargs):
        try:
            raise LookupError(PRIVATE)
        except LookupError as deep:
            raise OSError(PRIVATE + TOKEN) from deep

    monkeypatch.setattr(objects, "get_optional_bounded", fail)
    capsys.readouterr()
    response = client.post(ROUTE, headers=HEADERS, json=document)
    assert response.status_code == 503 and response.json() == UNAVAILABLE
    message = diagnostic(capsys)
    assert "cause_class=OSError" in message and "LookupError" not in message


@pytest.mark.parametrize("stage", ["previous_read", "put"])
def test_exception_class_is_bounded(tmp_path, monkeypatch, capsys, stage):
    _service, objects, client, document = prepared(tmp_path, second=stage == "previous_read")
    error = type("X" * 100, (OSError,), {})

    def fail(*args, **kwargs):
        raise error(PRIVATE)

    monkeypatch.setattr(objects, "get_optional_bounded" if stage == "previous_read" else "put", fail)
    capsys.readouterr()
    response = client.post(ROUTE, headers=HEADERS, json=document)
    assert response.status_code == 503 and response.json() == UNAVAILABLE
    message = diagnostic(capsys)
    field = "cause_class=" if stage == "previous_read" else "error_class="
    assert field + "X" * 64 + " " in message


def test_success_and_identical_replay_remain_silent(tmp_path, capsys):
    _service, _objects, client, document = prepared(tmp_path)
    first = client.post(ROUTE, headers=HEADERS, json=document)
    second = client.post(ROUTE, headers=HEADERS, json=document)
    assert first.status_code == second.status_code == 200 and first.json() == second.json()
    assert len(list((tmp_path / "objects").rglob("segment-*.gz"))) == 1
    assert "arena training observation segment unavailable" not in capsys.readouterr().err


def test_immutable_put_conflict_keeps_409_without_an_unavailable_warning(tmp_path, monkeypatch, capsys):
    _service, objects, client, document = prepared(tmp_path)

    def conflict(*args, **kwargs):
        raise ArenaContractError(PRIVATE)

    monkeypatch.setattr(objects, "put", conflict)
    response = client.post(ROUTE, headers=HEADERS, json=document)
    assert response.status_code == 409
    assert response.json() == {"status": "rejected", "code": "training_observation_segment_conflict"}
    assert "arena training observation segment unavailable" not in capsys.readouterr().err


def test_missing_previous_segment_keeps_409_without_an_unavailable_warning(tmp_path, monkeypatch, capsys):
    _service, objects, client, document = prepared(tmp_path, second=True)
    monkeypatch.setattr(objects, "get_optional_bounded", lambda *_args: None)
    capsys.readouterr()
    response = client.post(ROUTE, headers=HEADERS, json=document)
    assert response.status_code == 409
    assert response.json() == {"status": "rejected", "code": "training_observation_segment_conflict"}
    assert "arena training observation segment unavailable" not in capsys.readouterr().err


@pytest.mark.parametrize("invalid", ["lease", "document"])
def test_authentication_and_validation_fail_before_diagnostics(tmp_path, capsys, invalid):
    _service, _objects, client, document = prepared(tmp_path)
    headers = {"x-lab-arena-lease": "b" * 64} if invalid == "lease" else HEADERS
    if invalid == "document":
        document["number"] = PRIVATE
    response = client.post(ROUTE, headers=headers, json=document)
    assert response.status_code == (401 if invalid == "lease" else 400)
    assert "arena training observation segment unavailable" not in capsys.readouterr().err
