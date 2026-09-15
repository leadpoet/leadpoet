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
    parallel_twenty = configuration.get("parallel_twenty_icp_execution") is True
    limits = []
    for stage in (1, 2):
        count = int(configuration["stage_%d_icp_count" % stage])
        if count < 1:
            raise ValueError("daily competition requires ICPs")
        close = "stage_%d_close" % stage
        phases = []
        if stage == 1:
            # Parallel rounds execute both ICP stages before stage-one scoring.
            execution_count = count
            if parallel_twenty:
                execution_count += int(configuration["stage_2_icp_count"])
            phases.append(("stage_1_start", close, "icp_wall_clock_seconds", execution_count))
        elif not parallel_twenty:
            phases.append(("stage_2_start", close, "icp_wall_clock_seconds", count))
        phases.append((
            close,
            "stage_1_scoring_close" if stage == 1 else "final_scoring_close",
            "scoring_wall_clock_seconds",
            count,
        ))
        for start, end, duration, assignments_per_participant in phases:
            wave_seconds = int(configuration[duration]) + ATTEMPT_OVERHEAD_SECONDS
            if wave_seconds <= ATTEMPT_OVERHEAD_SECONDS:
                raise ValueError("daily competition requires a positive run limit")
            waves = max(0, int(seconds(start, end) // wave_seconds))
            limits.append((waves * slots) // (attempts * assignments_per_participant) - 1)
    return max(0, min(limits))
