"""Shadow-round gate metrics (labarena.md sections 16, 18.8, 20 as revised).

A shadow round is a full round whose publication is marked shadow and whose
reward basis never governs. This module derives, from the published material
only, the per-ICP execution timings, how much of the public stage window the
executions and the scoring needed, and the scoring statistics the shadow
rounds must measure before the stage window and the scoring worker count are
fixed for the paid pilot.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Optional, Sequence

SHADOW_REPORT_SCHEMA_VERSION = "leadpoet.lab_arena.shadow_report.v2"
STAGE = 1  # one stage: every participant runs the same 30 ICPs


def _parse(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


def percentile(values: Sequence[float], fraction: float) -> Optional[float]:
    """Nearest-rank percentile; None for an empty sequence."""

    ordered = sorted(float(v) for v in values)
    if not ordered:
        return None
    rank = max(1, int(round(fraction * len(ordered) + 0.5)))
    return ordered[min(len(ordered), rank) - 1]


def execution_timings(receipts: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Per-ICP wall-clock statistics from the execution receipts."""

    values: List[float] = []
    for receipt in receipts:
        if int(receipt.get("stage") or 0) != STAGE or str(receipt.get("kind") or "execute") != "execute":
            continue
        try:
            seconds = (_parse(receipt["finished_at"]) - _parse(receipt["started_at"])).total_seconds()
        except (KeyError, ValueError, TypeError):
            continue
        values.append(max(0.0, seconds))
    return {"count": len(values), "p50_seconds": percentile(values, 0.5), "p95_seconds": percentile(values, 0.95), "max_seconds": max(values) if values else None}


def stage_completion(*, stage_open: str, stage_close: str, last_receipt_finished: Optional[str], window_start: str, window_end: str) -> Dict[str, Any]:
    """Fraction of the public stage window the stage actually needed."""

    start = _parse(window_start)
    end = _parse(window_end)
    window = (end - start).total_seconds()
    finished = _parse(last_receipt_finished) if last_receipt_finished else _parse(stage_close)
    used = (finished - _parse(stage_open)).total_seconds()
    return {"window_seconds": window, "used_seconds": used, "fraction_of_window": (used / window) if window > 0 else None, "inside_eighty_percent": (used <= 0.8 * window) if window > 0 else None}


def shadow_report(*, round_row: Mapping[str, Any], public_bundle: Mapping[str, Any], scoring_timings: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Assemble the shadow gate report for one published shadow round."""

    configuration = round_row["configuration_doc"]
    participants = public_bundle.get("participants") or []
    receipts = public_bundle.get("receipts") or []
    schedule = configuration["schedule"]
    finished = [receipt.get("finished_at") or "" for receipt in receipts if int(receipt.get("stage") or 0) == STAGE and str(receipt.get("kind") or "execute") == "execute"]
    finished = [value for value in finished if value]
    completion = stage_completion(
        stage_open=schedule["stage_1_start"], stage_close=schedule["stage_1_close"],
        last_receipt_finished=max(finished) if finished else None,
        window_start=schedule["stage_1_start"], window_end=schedule["stage_1_close"],
    )
    scoring: Dict[str, Any] = {}
    for entry in scoring_timings:
        if int(entry.get("stage") or 0) != STAGE:
            continue
        scoring = {
            "seconds": (_parse(entry["finished_at"]) - _parse(entry["started_at"])).total_seconds(),
            "judge_executions": int(entry.get("judge_executions") or 0),
            # Validator scoring: every work item is judged by a validator on the
            # scored miner's keys and replayed by the Arena from the recorded responses.
            "work_items": int(entry.get("work_items") or 0),
            "key_refused_items": len(entry.get("key_refused_items") or []),
            "replay_mismatches": sum(1 for item in (entry.get("replays") or []) if isinstance(item, dict) and item.get("outcome") == "mismatch"),
        }
        window = (_parse(schedule["stage_1_scoring_close"]) - _parse(schedule["stage_1_close"])).total_seconds()
        scoring["fraction_of_window"] = (scoring["seconds"] / window) if window > 0 else None
    report = {
        "schema_version": SHADOW_REPORT_SCHEMA_VERSION,
        "round_id": round_row["round_id"],
        "participants": len(participants),
        "king_submission_id": (public_bundle.get("king_decision") or {}).get("king_submission_id"),
        "winner_submission_id": (public_bundle.get("king_decision") or {}).get("winner_submission_id"),
        "execution_timings": execution_timings(receipts),
        "stage_completion": completion,
        "scoring": scoring,
        "runner_fractions": public_bundle.get("runner_fractions"),
    }
    # The capacity gate: the executions fit inside 80 percent of the public window.
    report["passes_capacity_gate"] = bool(completion.get("inside_eighty_percent"))
    return report
