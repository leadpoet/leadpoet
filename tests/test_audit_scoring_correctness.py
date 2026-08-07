"""Regression tests for audit-found scoring defects.

C4: champion margin was computed with an inline `/ champion_score` that crashed
    (ZeroDivisionError) when a champion re-benchmarked to 0.0 on the current set,
    aborting champion promotion. Fix routes through the zero-guarded
    `calculate_margin`.
C7: `check_evidence_freshness` accepted future-dated evidence (negative age never
    exceeds the cap), defeating the buyer recency cap; the time-decay path also
    treated future dates as "fresher than today". Fix rejects future dates at the
    gate (when a cap exists) and clamps negative ages in the decay path.
"""

import inspect
from datetime import date, datetime, timedelta, timezone

from qualification.scoring import champion as champion_mod
from qualification.scoring.intent_signal_gate import check_evidence_freshness
from qualification.scoring.lead_scorer import (
    calculate_age_months,
    calculate_time_decay_multiplier,
)


# ---- C4: champion margin never divides by a zero champion score ----

def test_calculate_margin_handles_zero_champion_score():
    assert champion_mod.calculate_margin(25.0, 0.0) == float("inf")
    assert champion_mod.calculate_margin(0.0, 0.0) == 0.0
    assert champion_mod.calculate_margin(30.0, 20.0) == 50.0


def test_no_inline_unguarded_champion_division_remains():
    src = inspect.getsource(champion_mod)
    # The three previously-crashing inline call-site expressions must be gone;
    # the only division by a champion score now lives inside the guarded
    # calculate_margin (which this deliberately does not match).
    assert "new_champion_model.total_score - champion_score) / champion_score" not in src
    assert "best_challenger.total_score - champion_score) / champion_score" not in src
    assert "champion.score - previous_champion.score) / previous_champion.score" not in src


# ---- C7: future-dated evidence ----

def _iso(days_from_now):
    return (datetime.now(timezone.utc) + timedelta(days=days_from_now)).strftime("%Y-%m-%d")


def test_future_dated_evidence_rejected_when_cap_exists():
    reason = check_evidence_freshness("", _iso(+400), buyer_cap_days=90)
    assert reason is not None and "future" in reason.lower()


def test_recent_evidence_passes():
    assert check_evidence_freshness("", _iso(-10), buyer_cap_days=90) is None


def test_stale_evidence_rejected():
    reason = check_evidence_freshness("", _iso(-400), buyer_cap_days=90)
    assert reason is not None and "old" in reason.lower()


def test_minor_clock_skew_tolerated():
    # A signal dated "tomorrow" (within tolerance) is not rejected as future.
    assert check_evidence_freshness("", _iso(+1), buyer_cap_days=90) is None


# ---- C7 decay path: future dates do not max out the multiplier ----

def test_future_date_decay_clamped_to_zero_age():
    future = date.today() + timedelta(days=400)
    assert calculate_age_months(future) == 0.0
    # Old evidence still decays.
    old = date.today() - timedelta(days=400)
    assert calculate_age_months(old) > 12
    # A clamped future age gets the same (non-boosted) multiplier as "today".
    assert calculate_time_decay_multiplier(calculate_age_months(future)) == \
        calculate_time_decay_multiplier(0.0)
