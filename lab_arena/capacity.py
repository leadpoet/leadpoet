"""Small daily intake bound from the published schedule and worker limits."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Mapping


# Leaves time for dispatch, result delivery, and normal sandbox setup in each
# attempt. Retries consume a full second attempt, not an assumed cheap retry.
ATTEMPT_OVERHEAD_SECONDS = 60


def daily_challenger_capacity(configuration: Mapping[str, Any]) -> int:
    """Maximum challengers whose baseline plus all 20 ICPs fit every phase.

    This bounds configured workload. It cannot promise provider uptime or
    replace monitoring that the configured runners are actually available.
    """

    schedule = configuration["schedule"]
    slots = int(configuration["runner_slot_ceiling"]) * len(set(configuration["runner_hotkeys"]))
    attempts = int(configuration["max_attempts_per_assignment"])
    if slots < 1 or attempts < 1:
        raise ValueError("daily competition requires runner capacity")

    def seconds(start: str, end: str) -> float:
        return (
            datetime.fromisoformat(schedule[end].replace("Z", "+00:00"))
            - datetime.fromisoformat(schedule[start].replace("Z", "+00:00"))
        ).total_seconds()

    if seconds("submission_cutoff", "publication_deadline") >= 24 * 3600:
        raise ValueError("daily evaluation must finish before the next cutoff")
    limits = []
    for stage in (1, 2):
        count = int(configuration["stage_%d_icp_count" % stage])
        if count < 1:
            raise ValueError("daily competition requires ICPs")
        close = "stage_%d_close" % stage
        for start, end, duration in (
            ("stage_%d_start" % stage, close, "icp_wall_clock_seconds"),
            (close, "stage_1_scoring_close" if stage == 1 else "final_scoring_close", "scoring_wall_clock_seconds"),
        ):
            wave_seconds = int(configuration[duration]) + ATTEMPT_OVERHEAD_SECONDS
            if wave_seconds <= ATTEMPT_OVERHEAD_SECONDS:
                raise ValueError("daily competition requires a positive run limit")
            waves = max(0, int(seconds(start, end) // wave_seconds))
            limits.append((waves * slots) // (attempts * count) - 1)
    if configuration.get("integrity_policy") == "arena_integrity_v1":
        # A fixed cohort of three challengers plus the baseline must fit even
        # when every assignment takes its full retry budget.
        for start, end, duration in (
            ("stage_3_start", "stage_3_close", "icp_wall_clock_seconds"),
            ("stage_3_close", "stage_3_scoring_close", "scoring_wall_clock_seconds"),
        ):
            waves = max(0, int(seconds(start, end) // (int(configuration[duration]) + ATTEMPT_OVERHEAD_SECONDS)))
            if waves * slots < attempts * 5 * 4:
                return 0
    return max(0, min(limits))
