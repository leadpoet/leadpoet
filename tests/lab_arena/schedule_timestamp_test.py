"""Schedule comparisons use aware instants for both native UTC serializations."""
from datetime import datetime, timezone

import pytest
from lab_arena.service import _parse_iso


@pytest.mark.parametrize('value,expected', [
    ('2026-10-21T23:00:00Z', datetime(2026, 10, 21, 23, tzinfo=timezone.utc)),
    ('2026-10-21T23:00:00.123456Z', datetime(2026, 10, 21, 23, microsecond=123456, tzinfo=timezone.utc)),
    ('2026-10-21T23:00:00+00:00', datetime(2026, 10, 21, 23, tzinfo=timezone.utc)),
    ('2026-10-21T23:59:00+00:00', datetime(2026, 10, 21, 23, 59, tzinfo=timezone.utc)),
    ('2026-10-21T23:59:59+00:00', datetime(2026, 10, 21, 23, 59, 59, tzinfo=timezone.utc)),
    ('2026-10-22T01:30:00.123456+02:30', datetime(2026, 10, 21, 23, microsecond=123456, tzinfo=timezone.utc)),
    ('2026-10-21T20:00:00-03:00', datetime(2026, 10, 21, 23, tzinfo=timezone.utc)),
])
def test_aware_schedule_timestamps_normalize_to_utc(value, expected):
    actual = _parse_iso(value)
    assert actual == expected and actual.tzinfo is timezone.utc


@pytest.mark.parametrize('value', [
    '', 'not-a-date', '2026-10-21', '2026-10-21T23:00:00',
    '2026-10-21T23:00:00.123456', '2026-02-30T23:00:00+00:00',
    '2026-10-21T23:00:00+25:00',
])
def test_naive_or_invalid_schedule_timestamps_are_rejected(value):
    with pytest.raises(ValueError):
        _parse_iso(value)


@pytest.mark.parametrize('fraction', ['1', '12', '123', '1234', '12345', '123456'])
@pytest.mark.parametrize('suffix', ['Z', '+00:00'])
def test_all_existing_fraction_precisions_remain_supported(fraction, suffix):
    actual = _parse_iso('2026-10-21T23:00:00.' + fraction + suffix)
    assert actual.microsecond == int(fraction.ljust(6, '0'))
    assert actual.tzinfo is timezone.utc
