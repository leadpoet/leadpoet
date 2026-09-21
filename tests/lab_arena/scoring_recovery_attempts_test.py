"""Recovered scoring supersedes older failures without discarding evidence."""
from types import SimpleNamespace

from lab_arena.service import ArenaService


def _run(generation, attempt, status, cause):
    return {"scored_run_id": "accepted-execution", "stage_generation": generation,
            "attempt": attempt, "status": status, "terminal_cause": cause}


def _service(runs):
    service = object.__new__(ArenaService)
    service._load_scoring_plan = lambda _round, _stage: {
        "work_items": [{"scored_run_id": "accepted-execution"}]}
    service._store = SimpleNamespace(list_runs=lambda *_args, **_kwargs: runs)
    return service


def test_recovered_credential_failure_does_not_revive_old_exhausted_judge_error():
    old = _run(4, 2, "failed", "judge_error")
    recovered = _run(5, 1, "failed", "credential_error")
    service = _service([old, recovered])
    assert service._scoring_outputs("round", 1)["accepted-execution"] == recovered


def test_newest_recovery_failure_is_selected_without_reviving_old_attempt():
    runs = [_run(4, 2, "failed", "judge_error"),
            _run(5, 1, "failed", "judge_error"),
            _run(5, 2, "failed", "judge_error")]
    assert _service(runs)._scoring_outputs("round", 1)["accepted-execution"] == runs[-1]


def test_recovered_pending_attempt_keeps_round_running():
    runs = [_run(4, 2, "failed", "judge_error"), _run(5, 1, "pending", None)]
    assert _service(runs)._scoring_outputs("round", 1)["accepted-execution"] == runs[-1]


def test_accepted_work_wins_over_failures_in_every_generation():
    accepted = _run(4, 1, "accepted", "accepted")
    runs = [accepted, _run(5, 2, "failed", "judge_error")]
    service = _service(runs)
    assert service._scoring_outputs("round", 1)["accepted-execution"] == accepted
