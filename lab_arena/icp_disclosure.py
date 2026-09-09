"""A simple, stable 7/3 public split from the completed daily baseline."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from lab_arena import contracts


def baseline_disclosure(round_row: Mapping[str, Any], runs: Sequence[Mapping[str, Any]]) -> dict | None:
    """Only a complete set of persisted baseline scores can disclose ICPs.

    Lowest seven + highest three are public. The complement has three of
    the lower half and seven of the upper half. Ties use original position.
    No miner result, RNG, or current timestamp affects the split.
    """
    baselines = [p for p in round_row.get("participants") or [] if p.get("is_king") is True]
    if len(baselines) != 1:
        return None
    baseline_id = baselines[0].get("submission_id")
    selected = {}
    for run in runs:
        if run.get("submission_id") != baseline_id or run.get("per_icp_score") is None:
            continue
        position = run.get("icp_position")
        if isinstance(position, bool) or not isinstance(position, int) or position not in range(contracts.BENCHMARK_ICP_COUNT):
            continue
        if run.get("kind", "execute") != "execute":
            continue
        current = selected.get(position)
        if current is None or int(run.get("attempt") or 0) > int(current.get("attempt") or 0):
            selected[position] = run
    if len(selected) != contracts.BENCHMARK_ICP_COUNT:
        return None
    scores = {}
    for position, run in selected.items():
        value = run["per_icp_score"]
        if isinstance(value, bool):
            return None
        try:
            score = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(score) or not 0 <= score <= 100:
            return None
        scores[position] = score
    # Match the existing final-baseline eligibility rule: an all-failed model
    # is not a completed rebenchmark with an invented zero.
    if not any(run.get("terminal_cause") == "accepted" for run in selected.values()):
        return None
    ranked = sorted(scores, key=lambda position: (scores[position], position))
    public = sorted(ranked[:7] + ranked[-3:])
    return {
        "baseline_submission_id": baseline_id,
        "public_positions": public,
        "private_positions": sorted(set(scores) - set(public)),
        "baseline_scores": scores,
    }
