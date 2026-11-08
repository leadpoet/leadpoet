"""Capture diagnostics survive the chain library disabling application loggers."""

import logging
from types import SimpleNamespace

import pytest

from lab_arena import contracts, service


@pytest.fixture(autouse=True)
def chain_logging_state(monkeypatch):
    # These are the actual changes made by Bittensor's lazy import.
    monkeypatch.setattr(service.logger, "handlers", [])
    monkeypatch.setattr(service.logger, "level", logging.CRITICAL)
    monkeypatch.setattr(service, "_TRAJECTORY_WARNING_LAST", {})


def test_segment_warning_survives_disabled_logger(capsys):
    service._warn_observation_segment_failure("run-1", 2, "put", OSError("private-secret"))
    message = capsys.readouterr().err
    assert "run_id=run-1 segment=2 stage=put error_class=OSError" in message
    assert "private-secret" not in message


def test_trajectory_warning_survives_with_existing_throttle(monkeypatch, capsys):
    clock = iter((100.0, 110.0, 130.0))
    monkeypatch.setattr(service.time, "monotonic", lambda: next(clock))
    for _ in range(3):
        service._warn_trajectory_failure("store", OSError("private-secret"))
    lines = capsys.readouterr().err.splitlines()
    assert len(lines) == 2
    assert all("phase=store error_class=OSError" in line for line in lines)
    assert "private-secret" not in "".join(lines)


def test_eligibility_failure_keeps_false_and_safe_diagnostic(capsys):
    def fail(_identity):
        raise OSError("private-secret")
    instance = SimpleNamespace(_store=SimpleNamespace(get_submission=fail))
    outcome = service.ArenaService._submission_training_eligibility(
        instance, {"round_id": "private-round"}, "private-submission", [])
    assert outcome == {"eligible": False, "reasons": ["training_evidence_check_failed"]}
    message = capsys.readouterr().err
    assert "arena_training_evidence_check_failed" in message
    assert contracts.hash_bytes(b"private-round") in message
    assert contracts.hash_bytes(b"private-submission") in message
    assert "error_class=OSError" in message
    assert all(value not in message for value in ("private-secret", "private-round", "private-submission"))


def test_broken_stderr_does_not_escape_capture_diagnostics(monkeypatch):
    class BrokenStream:
        def write(self, _value):
            raise OSError("closed diagnostic stream")
    monkeypatch.setattr(service.sys, "stderr", BrokenStream())
    service._warn_observation_segment_failure("run-1", 2, "put", OSError("private-secret"))
    service._warn_trajectory_failure("store", OSError("private-secret"))
    def fail(_identity):
        raise OSError("private-secret")
    instance = SimpleNamespace(_store=SimpleNamespace(get_submission=fail))
    outcome = service.ArenaService._submission_training_eligibility(
        instance, {"round_id": "private-round"}, "private-submission", [])
    assert outcome == {"eligible": False, "reasons": ["training_evidence_check_failed"]}
