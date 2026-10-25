"""The paid E2E driver follows only score retries admitted by the service."""

from types import SimpleNamespace

import pytest

from scripts.verify_arena_full_thinking_e2e import FullThinkingHarness


def _driver(*, admit_retry):
    state = {"status": "stage1_judged", "runs": [{"status": "failed"}], "claims": 0}

    def advance(_round_id):
        if admit_retry and state["claims"] == 0:
            state["runs"] = [{"status": "failed"}, {"status": "pending"}]
            return {"status": "retry", "reason": "judge_incomplete"}
        if not admit_retry:
            return {"status": "retry", "reason": "judge_incomplete"}
        state["status"] = "stage1_scored"
        return {"status": "ok"}

    harness = object.__new__(FullThinkingHarness)
    harness.round_id = "disposable-round"
    harness.service = SimpleNamespace(
        advance_round=advance,
        store=SimpleNamespace(list_runs=lambda *_args, **_kwargs: state["runs"]),
    )
    harness.status = lambda: state["status"]
    harness.schedule = lambda: {}
    harness.run_stage_with_runners = lambda _count: state.update(
        claims=state["claims"] + 1, runs=[{"status": "accepted"}]
    )
    return harness, state


def test_driver_claims_ordinary_retry_while_round_is_judged(monkeypatch):
    monkeypatch.setattr("scripts.verify_arena_full_thinking_e2e.time.sleep", lambda _seconds: None)
    harness, state = _driver(admit_retry=True)
    assert harness.advance_until("stage1_scored", runners=1, max_steps=4) == {"status": "ok"}
    assert state["claims"] == 1


def test_driver_rejects_incomplete_judgment_without_queued_retry(monkeypatch):
    monkeypatch.setattr("scripts.verify_arena_full_thinking_e2e.time.sleep", lambda _seconds: None)
    harness, state = _driver(admit_retry=False)
    with pytest.raises(AssertionError, match="no queued score retry"):
        harness.advance_until("stage1_scored", runners=1, max_steps=4)
    assert state["claims"] == 0
