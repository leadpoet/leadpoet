"""All-participant daily capacity includes the baseline and both attempts."""

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from lab_arena import capacity, contracts
from lab_arena.service import ArenaService, DEFAULT_STAGE_MINUTES


def configuration(*, runners=1, minutes=None):
    service = object.__new__(ArenaService)
    service._config = SimpleNamespace(defaults=SimpleNamespace(stage_minutes=minutes or DEFAULT_STAGE_MINUTES))
    return {
        "schedule": service.build_schedule(datetime(2026, 9, 10, tzinfo=timezone.utc)),
        "runner_slot_ceiling": contracts.RUNNER_SLOT_CEILING,
        "runner_hotkeys": ["runner-%d" % index for index in range(runners)],
        "max_attempts_per_assignment": contracts.MAX_ATTEMPTS_PER_ASSIGNMENT,
        "stage_1_icp_count": contracts.STAGE_1_ICP_COUNT,
        "stage_2_icp_count": contracts.STAGE_2_ICP_COUNT,
        "icp_wall_clock_seconds": contracts.ICP_WALL_CLOCK_SECONDS,
        "scoring_wall_clock_seconds": contracts.SCORING_WALL_CLOCK_SECONDS,
    }


def test_one_real_eight_slot_runner_supports_eight_challengers_with_retries():
    config = configuration()
    assert capacity.daily_challenger_capacity(config) == 8
    assert config["schedule"]["final_scoring_close"] == "2026-09-10T20:30:02Z"


def test_second_configured_runner_can_support_requested_sixteen():
    assert capacity.daily_challenger_capacity(configuration(runners=2)) >= 16


def test_old_schedule_cannot_support_sixteen_even_before_shared_failures():
    old = {**DEFAULT_STAGE_MINUTES, "stage_1_scoring": 360, "final_scoring": 240}
    assert capacity.daily_challenger_capacity(configuration(minutes=old)) == 5


def test_duplicate_runner_does_not_invent_capacity():
    config = configuration()
    config["runner_hotkeys"] *= 2
    assert capacity.daily_challenger_capacity(config) == 8


def test_zero_workers_and_overlapping_daily_budget_fail_closed():
    with pytest.raises(ValueError, match="runner capacity"):
        capacity.daily_challenger_capacity(configuration(runners=0))
    config = configuration(minutes={**DEFAULT_STAGE_MINUTES, "final_scoring": 650})
    with pytest.raises(ValueError, match="next cutoff"):
        capacity.daily_challenger_capacity(config)


def test_rounding_counts_full_retry_waves_and_baseline():
    import math

    config = configuration()
    allowed = capacity.daily_challenger_capacity(config)
    waves = math.ceil((allowed + 1) * 10 * 2 / 8)
    assert waves * (900 + capacity.ATTEMPT_OVERHEAD_SECONDS) <= 390 * 60
    too_many_waves = math.ceil((allowed + 2) * 10 * 2 / 8)
    assert too_many_waves * (900 + capacity.ATTEMPT_OVERHEAD_SECONDS) > 390 * 60
