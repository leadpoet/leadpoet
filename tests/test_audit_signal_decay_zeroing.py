"""W4-3: a signal the three-stage verifier accepted (date_status="verified")
but that has no usable date and comes from a date-required source is zeroed.
That behavior is intentional and preserved — but it used to be silent. These
tests pin the behavior AND assert the new WARNING fires, so the fix can't
regress into either a scoring change or back into a silent sentinel."""
import logging
from datetime import date, timedelta

from qualification.scoring.lead_scorer import _apply_signal_time_decay


def test_verified_undated_date_required_source_zeroes_with_warning(caplog):
    with caplog.at_level(logging.WARNING):
        score, mult = _apply_signal_time_decay(54.0, None, "verified", "news")
    # Behavior preserved: the verified-but-undated signal is still zeroed.
    assert (score, mult) == (0.0, 0.0)
    # No longer silent: the zeroing is now observable with a unique tag.
    assert "intent_signal_zeroed_missing_date" in caplog.text


def test_verified_undated_date_not_required_source_is_kept(caplog):
    # A source that doesn't require a date (tech stack / company info) must NOT
    # be zeroed and must NOT emit the warning.
    with caplog.at_level(logging.WARNING):
        score, mult = _apply_signal_time_decay(54.0, None, "verified", "company_website")
    assert (score, mult) == (54.0, 1.0)
    assert "intent_signal_zeroed_missing_date" not in caplog.text


def test_verified_recent_dated_signal_is_not_zeroed():
    # A dated, recent signal keeps full weight — the fix only touches the
    # undated fall-through, not the normal decay path. Compute relative to
    # today so the "recent" assertion doesn't drift with the wall clock.
    recent = (date.today() - timedelta(days=5)).isoformat()
    score, mult = _apply_signal_time_decay(54.0, recent, "verified", "news")
    assert score > 0.0
    assert mult == 1.0


def test_no_date_soft_path_still_halves():
    # The softer NO_DATE_DECAY path (its real caller passes date_status="no_date")
    # is unchanged: 0.5x, not a hard zero.
    score, mult = _apply_signal_time_decay(54.0, None, "no_date", "news")
    assert (score, mult) == (27.0, 0.5)
