"""Fixture dates follow cutoff authority through UTC day/month boundaries."""

from datetime import datetime, timedelta

import pytest

from tests.lab_arena.round_timing_fixture import commit_dates


@pytest.mark.parametrize("now,evaluation,bank", [
    ("2026-10-28T23:29:00+00:00", "2026-10-28", "2026-10-27"),
    ("2026-10-28T23:31:00+00:00", "2026-10-29", "2026-10-28"),
    ("2026-10-31T23:29:00+00:00", "2026-10-31", "2026-10-30"),
    ("2026-10-31T23:31:00+00:00", "2026-11-01", "2026-10-31"),
    ("2026-12-31T23:31:00+00:00", "2027-01-01", "2026-12-31"),
    ("2028-02-29T23:31:00+00:00", "2028-03-01", "2028-02-29"),
    ("2026-11-01T00:10:00+02:00", "2026-10-31", "2026-10-30"),
])
def test_commit_fixture_uses_cutoff_utc_day(now, evaluation, bank):
    cutoff = datetime.fromisoformat(now) + timedelta(minutes=30)
    actual = commit_dates({"schedule": {"submission_cutoff": cutoff.isoformat()}})
    assert tuple(day.isoformat() for day in actual) == (evaluation, bank)
