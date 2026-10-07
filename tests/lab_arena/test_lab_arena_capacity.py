"""All-participant daily capacity includes the baseline and both attempts."""

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from lab_arena import capacity, contracts, scoring
from lab_arena.service import ArenaService, DEFAULT_STAGE_MINUTES, RoundDefaults


def configuration(*, runners=1, minutes=None, slots=8):
    service = object.__new__(ArenaService)
    service._config = SimpleNamespace(defaults=SimpleNamespace(stage_minutes=minutes or DEFAULT_STAGE_MINUTES))
    return {
        "schedule": service.build_schedule(datetime(2026, 9, 10, tzinfo=timezone.utc)),
        "runner_slot_ceiling": slots,
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


def test_measured_runner_slots_bound_the_frozen_ceiling_and_ignore_offline_runners():
    config = configuration(runners=3, slots=251)
    assert capacity.daily_challenger_capacity(config, runner_parallelism={
        "runner-0": 8, "runner-1": 0,
    }) == capacity.daily_challenger_capacity(configuration(slots=8))
    assert capacity.daily_challenger_capacity(config, runner_parallelism={
        "runner-0": 8, "runner-1": 8, "unregistered": 251,
    }) == capacity.daily_challenger_capacity(configuration(runners=2, slots=8))
    assert capacity.daily_challenger_capacity(config, runner_parallelism={
        "runner-0": 1000,
    }) == capacity.daily_challenger_capacity(configuration(slots=251))
    assert capacity.daily_challenger_capacity(config, runner_parallelism={}) == 0
    with pytest.raises(ValueError, match="parallelism"):
        capacity.daily_challenger_capacity(config, runner_parallelism={"runner-0": True})


def test_baseline_first_60_minute_round_respects_current_proxy_capacity():
    config = configuration(slots=251)
    config["stage_1_icp_count"] = config["stage_2_icp_count"] = 5
    config["execution_sequence_policy"] = contracts.BASELINE_SCORED_FIRST_POLICY
    config["icp_wall_clock_seconds"] = 60 * 60
    # The verified proxy inventory has one native slot in addition to its
    # proxy slots. The stage-two window must carry every miner execution.
    for proxies, expected in ((9, 1), (19, 2), (29, 3), (250, 25)):
        assert capacity.daily_challenger_capacity(
            config, runner_parallelism={"runner-0": proxies + 1}
        ) == expected
    assert capacity.daily_challenger_capacity(config) == 25


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


def parallel_45_minute_configuration(**minute_overrides):
    # Six participants (five challengers plus the baseline) need two full
    # attempts across twenty executions and both ten-ICP scoring phases.
    minutes = {
        **DEFAULT_STAGE_MINUTES,
        "stage_1": 1012,          # 22 waves at 45 minutes + one minute.
        "stage_1_scoring": 192,   # 12 waves at 15 minutes + one minute.
        "stage_2": 1,             # Activation of preexecuted stage-two runs.
        "final_scoring": 192,
    }
    minutes.update(minute_overrides)
    config = configuration(minutes=minutes, slots=11)
    config["parallel_twenty_icp_execution"] = True
    config["icp_wall_clock_seconds"] = 2700
    return config


def test_parallel_twenty_supports_five_challengers_on_eleven_slots_without_stage_two_reruns():
    config = parallel_45_minute_configuration()
    assert config["schedule"]["final_scoring_close"] == "2026-09-10T23:47:02Z"
    assert capacity.daily_challenger_capacity(config) == 5
    # The same one-minute stage-two execution window cannot support a
    # sequential round, where ten more executions are genuinely required.
    assert capacity.daily_challenger_capacity({
        key: value for key, value in config.items()
        if key != "parallel_twenty_icp_execution"
    }) == 0


def test_parallel_twenty_preserves_all_execution_and_both_scoring_retry_budgets():
    for phase, shorter_minutes in (
        ("stage_1", 966),
        ("stage_1_scoring", 160),
        ("final_scoring", 160),
    ):
        shorter = parallel_45_minute_configuration(**{phase: shorter_minutes})
        assert capacity.daily_challenger_capacity(shorter) == 4, phase


@pytest.mark.parametrize("max_challengers", [7, 256])
@pytest.mark.parametrize("slots", [1, 10, 251])
def test_new_live_round_keeps_configured_admission_limit(max_challengers, slots, monkeypatch):
    service = object.__new__(ArenaService)
    digest = "sha256:" + "a" * 64
    service._config = SimpleNamespace(
        mode="live", network_name="finney", netuid=71,
        defaults=RoundDefaults(
            runner_hotkeys=("5" * 48,), baseline_hotkey="5" * 48,
            max_challengers=max_challengers, runner_slot_ceiling=slots,
            execution_sequence_from="2026-01-01T00:00:00Z",
            scorer_image_digest=digest,
            scorer_image_reference="registry.example/scorer@" + digest,
        ),
    )
    service._scorer_policy = scoring.build_scorer_policy()
    service.runner_settings = lambda: (["5" * 48], [])
    service._require_round_ownership = lambda _round_id: None
    stored = []
    service._store = SimpleNamespace(
        create_round=lambda _id, config: stored.append(config) or {"status": "created"},
    )
    monkeypatch.setattr(
        capacity, "daily_challenger_capacity",
        lambda *_args, **_kwargs: pytest.fail("admission used a workload estimate"),
    )

    config = service.create_round(datetime(2026, 10, 8, tzinfo=timezone.utc))

    assert config["max_challengers"] == max_challengers
    assert stored == [config]
    assert config["runner_slot_ceiling"] == slots
    assert config["max_attempts_per_assignment"] == 2
    assert config["execution_sequence_policy"] == contracts.BASELINE_SCORED_FIRST_POLICY
