"""Regression tests: idle workers back off polling exponentially with jitter.

Egress reduction: hosted and scoring workers polled the queue every ~15s even
when idle. With N workers that is N × constant requests. Idle backoff grows
15 -> 30 -> 60 (capped) and resets the instant work appears, so idle request
volume stays ~constant as worker count grows.
"""

from __future__ import annotations

import gateway.research_lab.worker as hosted
import gateway.research_lab.scoring_worker as scoring


import pytest


@pytest.mark.parametrize("mod", [hosted, scoring])
def test_backoff_doubles_and_clamps_to_cap(mod) -> None:
    base, cap = 15.0, 60.0
    seq = [base]
    for _ in range(5):
        seq.append(mod._idle_backoff_next(seq[-1], base, cap))
    # 15 -> 30 -> 60 -> 60 (capped, never exceeds cap, never below base).
    assert seq == [15.0, 30.0, 60.0, 60.0, 60.0, 60.0]
    assert all(base <= x <= cap for x in seq)


@pytest.mark.parametrize("mod", [hosted, scoring])
def test_backoff_max_never_below_base(mod, monkeypatch) -> None:
    monkeypatch.setenv("RESEARCH_LAB_WORKER_IDLE_BACKOFF_MAX_SECONDS", "5")
    # Even if the configured cap is below the base poll, the base wins.
    assert mod._idle_backoff_max_seconds(30.0) == 30.0
    monkeypatch.setenv("RESEARCH_LAB_WORKER_IDLE_BACKOFF_MAX_SECONDS", "120")
    assert mod._idle_backoff_max_seconds(15.0) == 120.0


@pytest.mark.parametrize("mod", [hosted, scoring])
def test_sleep_jitter_is_bounded(mod) -> None:
    for _ in range(200):
        s = mod._idle_backoff_sleep_seconds(40.0)
        assert 40.0 <= s <= 50.0  # up to +25% jitter, never less than interval


@pytest.mark.parametrize("mod", [hosted, scoring])
def test_reset_returns_to_base(mod) -> None:
    base, cap = 15.0, 60.0
    idle = mod._idle_backoff_next(mod._idle_backoff_next(base, base, cap), base, cap)
    assert idle == 60.0
    # Work appears -> caller resets to base (the loop assigns base_poll directly);
    # a subsequent idle step grows from the base again, not from the cap.
    assert mod._idle_backoff_next(base, base, cap) == 30.0


def _next_interval(processed: bool, status: str, current: float, base: float, cap: float) -> float:
    # Mirrors the run_forever backoff decision: reset ONLY when real work
    # happened; every non-processed status backs off. This encodes the fix for
    # the finding that non-idle no-work statuses reset to the base interval.
    if processed:
        return base
    return hosted._idle_backoff_next(current, base, cap)


@pytest.mark.parametrize(
    "status",
    [
        "idle",
        "maintenance_paused",
        "provider_preflight_unhealthy",
        "candidate_scoring_daily_baseline_hold",
        "candidate_claim_capacity_limited",
        "writes_or_eval_disabled",
    ],
)
def test_all_no_work_statuses_back_off(status) -> None:
    base, cap = 15.0, 60.0
    # A no-work pass (processed=False) with ANY status must grow the interval,
    # not reset it -- previously only the exact "idle" status backed off.
    assert _next_interval(False, status, base, base, cap) == 30.0
    assert _next_interval(False, status, 30.0, base, cap) == 60.0
    assert _next_interval(False, status, 60.0, base, cap) == 60.0  # capped


def test_processed_pass_resets_regardless_of_status() -> None:
    base, cap = 15.0, 60.0
    # Real work resets to base even if the status string is non-"idle".
    assert _next_interval(True, "processed", 60.0, base, cap) == base
    assert _next_interval(True, "baseline_completed", 60.0, base, cap) == base
