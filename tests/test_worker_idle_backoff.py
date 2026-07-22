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
