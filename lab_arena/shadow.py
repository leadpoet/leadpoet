"""Shadow-round gate metrics (labarena.md sections 16, 18.8, 20 stage 1).

A shadow round makes every participant run all fifty ICPs. This module
derives, from the published material only, whether the simulated Stage 1
top ten contains the actual 50-ICP winning challenger, and the per-ICP
execution and scoring timings the shadow rounds must measure before the
stage windows and the scoring worker count are fixed.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Optional, Sequence

from lab_arena import contracts, verify

SHADOW_REPORT_SCHEMA_VERSION = "leadpoet.lab_arena.shadow_report.v1"


def _parse(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


def percentile(values: Sequence[float], fraction: float) -> Optional[float]:
    """Nearest-rank percentile; None for an empty sequence."""

    ordered = sorted(float(v) for v in values)
    if not ordered:
        return None
    rank = max(1, int(round(fraction * len(ordered) + 0.5)))
    return ordered[min(len(ordered), rank) - 1]


def simulated_top_ten_contains_winner(*, stage1_ranking: Sequence[Mapping[str, Any]], final_scores: Mapping[str, float], king_submission_id: Optional[str]) -> Dict[str, Any]:
    """The section 20 stage-1 gate: does the Stage 1 top ten contain the 50-ICP winner?"""

    simulated = verify.select_finalists(stage1_ranking)
    challengers = {sid: score for sid, score in final_scores.items() if sid != king_submission_id}
    if not challengers:
        return {"simulated_finalists": simulated, "actual_winner": None, "contains_winner": None}
    winner = max(sorted(challengers), key=lambda sid: challengers[sid])
    return {"simulated_finalists": simulated, "actual_winner": winner, "winner_final_score": challengers[winner], "contains_winner": winner in simulated}


def execution_timings(receipts: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Per-ICP wall-clock statistics from runner receipts, per stage."""

    per_stage: Dict[int, List[float]] = {1: [], 2: []}
    for receipt in receipts:
        try:
            seconds = (_parse(receipt["finished_at"]) - _parse(receipt["started_at"])).total_seconds()
        except (KeyError, ValueError, TypeError):
            continue
        per_stage.setdefault(int(receipt.get("stage") or 0), []).append(max(0.0, seconds))
    return {
        "stage_%d" % stage: {"count": len(values), "p50_seconds": percentile(values, 0.5), "p95_seconds": percentile(values, 0.95), "max_seconds": max(values) if values else None}
        for stage, values in per_stage.items() if stage in (1, 2)
    }


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
    if not configuration.get("all_participants_run_stage_2"):
        raise contracts.ArenaContractError("shadow report requires a round where every participant ran Stage 2")
    king_submission_id = (public_bundle.get("king_decision") or {}).get("king_submission_id")
    participants = public_bundle.get("participants") or []
    king_ids = {p["submission_id"] for p in participants if p.get("is_king")}
    final_scores = dict(public_bundle["score_bundles"]["final"]["submission_scores"])
    gate = simulated_top_ten_contains_winner(stage1_ranking=public_bundle["stage1_ranking"], final_scores=final_scores, king_submission_id=next(iter(king_ids), None))
    receipts = public_bundle.get("receipts") or []
    timings = execution_timings(receipts)
    schedule = configuration["schedule"]
    by_stage: Dict[int, List[str]] = {1: [], 2: []}
    for receipt in receipts:
        by_stage.setdefault(int(receipt.get("stage") or 0), []).append(receipt.get("finished_at") or "")
    completion = {}
    for stage in (1, 2):
        finished = [value for value in by_stage.get(stage, []) if value]
        completion["stage_%d" % stage] = stage_completion(
            stage_open=schedule["stage_%d_start" % stage], stage_close=schedule["stage_%d_close" % stage],
            last_receipt_finished=max(finished) if finished else None,
            window_start=schedule["stage_%d_start" % stage], window_end=schedule["stage_%d_close" % stage],
        )
    scoring = {}
    for entry in scoring_timings:
        scoring["stage_%d" % int(entry["stage"])] = {
            "seconds": (_parse(entry["finished_at"]) - _parse(entry["started_at"])).total_seconds(),
            "judge_executions": int(entry.get("judge_executions") or 0),
            "workers": int(entry.get("workers") or 1),
        }
    report = {
        "schema_version": SHADOW_REPORT_SCHEMA_VERSION,
        "round_id": round_row["round_id"],
        "participants": len(participants),
        "king_submission_id": king_submission_id,
        "finalist_gate": gate,
        "execution_timings": timings,
        "stage_completion": completion,
        "scoring": scoring,
        "runner_fractions": public_bundle.get("runner_fractions"),
    }
    report["passes_stage_1_gate"] = bool(gate.get("contains_winner")) and all(item.get("inside_eighty_percent") for item in completion.values())
    return report
