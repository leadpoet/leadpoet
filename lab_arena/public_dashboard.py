"""Small, allow-listed public projections for the Arena dashboard.

The durable Arena rows contain private source and runtime fields.  This module
is the only place where the dashboard shapes are assembled; it never returns a
raw database row or document.
"""

from __future__ import annotations

from datetime import datetime, timezone
import math
from typing import Any, Dict, Mapping, Optional, Sequence

from lab_arena import contracts, icp_disclosure, source_disclosure, verify


PUBLIC_BASELINE_REPOSITORY = "https://github.com/leadpoet/pydantic-harness/tree/lab"
DEFAULT_RECENT_ROUND_LIMIT = 30
MAX_RECENT_ROUND_LIMIT = 100
_ROUND_COLUMNS = (
    "round_id,status,created_at,configuration_doc,participants,"
    "publication_doc,published_at,cancel_reason,promotion_required,"
    "baseline_promoted_at,icp_set_date,evaluation_date"
)


def _timestamp(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, datetime):
        moment = value
    elif isinstance(value, str):
        try:
            moment = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    else:
        return None
    if moment.tzinfo is None or moment.utcoffset() is None:
        return None
    return moment.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _configuration_scope(row: Mapping[str, Any]) -> tuple[str, int, str]:
    configuration = row.get("configuration_doc")
    configuration = configuration if isinstance(configuration, Mapping) else {}
    return (
        str(configuration.get("network_name") or "finney"),
        int(configuration.get("netuid") or 71),
        str(configuration.get("mode") or ""),
    )


def _publication(row: Mapping[str, Any]) -> Mapping[str, Any]:
    value = row.get("publication_doc")
    return value if isinstance(value, Mapping) else {}


def _score(value: Any) -> Optional[float]:
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) and 0.0 <= number <= 100.0 else None


def _participants(row: Mapping[str, Any]) -> Sequence[Mapping[str, Any]]:
    publication = _publication(row)
    raw = publication.get("participants") if row.get("status") == "published" else row.get("participants")
    if not isinstance(raw, list):
        return ()
    return tuple(item for item in raw if isinstance(item, Mapping))


def _rankings(row: Mapping[str, Any], key: str) -> Dict[str, Mapping[str, Any]]:
    raw = _publication(row).get(key)
    if not isinstance(raw, list):
        return {}
    return {
        str(item.get("submission_id")): item
        for item in raw
        if isinstance(item, Mapping) and item.get("submission_id")
    }


def _champion_submission_id(row: Mapping[str, Any]) -> Optional[str]:
    decision = _publication(row).get("king_decision")
    if not isinstance(decision, Mapping):
        return None
    outcome = str(decision.get("outcome") or "")
    if outcome == "crowned":
        value = decision.get("winner_submission_id")
    elif outcome in ("defended", "retained_ineligible"):
        value = decision.get("king_submission_id")
    else:
        return None
    return str(value) if value else None


def _baseline_and_champion(row: Mapping[str, Any]) -> tuple[Optional[dict], Optional[dict]]:
    published = row.get("status") == "published"
    final = _rankings(row, "final_ranking") if published else {}
    baseline = None
    champion = None
    champion_id = _champion_submission_id(row) if published else None
    for participant in _participants(row):
        submission_id = str(participant.get("submission_id") or "")
        if not submission_id:
            continue
        is_baseline = bool(
            participant.get("is_baseline", participant.get("is_king", False))
        )
        ranking = final.get(submission_id) or {}
        projected = {
            "submission_id": submission_id,
            "miner_hotkey": str(participant.get("miner_hotkey") or ""),
            "final_score": _score(ranking.get("final_score")),
        }
        if is_baseline:
            baseline = projected
        elif submission_id == champion_id:
            champion = projected
    return baseline, champion


def round_summary(row: Mapping[str, Any]) -> dict:
    network_name, netuid, mode = _configuration_scope(row)
    configuration = row.get("configuration_doc")
    configuration = configuration if isinstance(configuration, Mapping) else {}
    schedule = configuration.get("schedule")
    schedule = schedule if isinstance(schedule, Mapping) else {}
    disclosure = icp_disclosure.disclosure_metadata(row) or {}
    baseline, champion = _baseline_and_champion(row)
    if champion is None or row.get("promotion_required") is not True:
        promotion_status = "not_required"
    elif row.get("baseline_promoted_at"):
        promotion_status = "promoted"
    else:
        promotion_status = "pending"
    return {
        "round_id": str(row.get("round_id") or ""),
        "status": str(row.get("status") or ""),
        "mode": mode,
        "network_name": network_name,
        "netuid": netuid,
        "created_at": _timestamp(row.get("created_at")),
        "submission_open": _timestamp(schedule.get("submission_open")),
        "submission_cutoff": _timestamp(schedule.get("submission_cutoff")),
        "icp_set_date": disclosure.get("icp_set_date"),
        "evaluation_date": disclosure.get("evaluation_date"),
        "public_at": disclosure.get("public_at"),
        "published_at": _timestamp(row.get("published_at")),
        "cancel_reason": str(row.get("cancel_reason")) if row.get("cancel_reason") else None,
        "participant_count": len(_participants(row)),
        "baseline": baseline,
        "champion": champion,
        "promotion_status": promotion_status,
    }


def competition_snapshot(service: Any, *, limit: int = DEFAULT_RECENT_ROUND_LIMIT) -> dict:
    bounded_limit = max(1, min(int(limit), MAX_RECENT_ROUND_LIMIT))
    network_name, netuid = service._chain_scope()
    pinned_round_id = getattr(service._config, "pinned_round_id", None)
    if pinned_round_id is not None:
        rows = [service._round(str(pinned_round_id))]
    else:
        rows = service._store.list_rounds(
            mode=service._config.mode,
            network_name=network_name,
            netuid=netuid,
            limit=bounded_limit,
            columns=_ROUND_COLUMNS,
        )
    summaries = [round_summary(row) for row in rows]
    open_round = next((row for row in summaries if row["status"] == "open"), None)
    latest_round = next(
        (row for row in summaries if row["status"] != "open"),
        None,
    )
    latest_completed = next(
        (row for row in summaries if row["status"] == "published"), None
    )
    if latest_completed is None and pinned_round_id is None:
        published = service._store.list_rounds(
            status="published",
            mode=service._config.mode,
            network_name=network_name,
            netuid=netuid,
            limit=1,
            columns=_ROUND_COLUMNS,
        )
        latest_completed = round_summary(published[0]) if published else None
    return {
        "mode": service._config.mode,
        "network_name": network_name,
        "netuid": netuid,
        "repo_url": PUBLIC_BASELINE_REPOSITORY,
        "open_round": open_round,
        "latest_round": latest_round,
        "latest_completed_round": latest_completed,
        "rounds": summaries,
    }


def _stage1_scores(service: Any, row: Mapping[str, Any]) -> Dict[str, float]:
    # Intermediate scores must not escape before evaluation is published.
    if row.get("status") != "published":
        return {}
    selected: Dict[tuple[str, int], Mapping[str, Any]] = {}
    for run in service._store.list_runs(str(row["round_id"]), stage=1, kind="execute"):
        if run.get("per_icp_score") is None:
            continue
        submission_id = str(run.get("submission_id") or "")
        position = int(run.get("icp_position") or 0)
        key = (submission_id, position)
        current = selected.get(key)
        if current is None or int(run.get("attempt") or 0) > int(current.get("attempt") or 0):
            selected[key] = run
    result: Dict[str, float] = {}
    positions = contracts.stage_positions(1)
    for participant in _participants(row):
        submission_id = str(participant.get("submission_id") or "")
        runs = [selected.get((submission_id, position)) for position in positions]
        if submission_id and all(run is not None for run in runs):
            result[submission_id] = verify.stage_score(
                [float(run["per_icp_score"]) for run in runs if run is not None],
                len(positions),
            )
    return result


def _submission_lifecycle(
    *, raw_status: str, round_status: str, is_champion: bool,
    final_score: Optional[float],
) -> str:
    if round_status == "open" and raw_status == "accepted":
        return "queued"
    if round_status == "cancelled":
        return "cancelled"
    if round_status == "published":
        if final_score is None:
            return "scoring_failed"
        return "champion" if is_champion else "scored"
    if round_status == "scored":
        return "scored"
    return "scoring"


def _submitted_at(submission: Mapping[str, Any]) -> Optional[str]:
    accepted_at = _timestamp(submission.get("accepted_at"))
    if accepted_at is not None:
        return accepted_at
    if submission.get("status") == "frozen":
        return _timestamp(submission.get("frozen_at"))
    if submission.get("status") == "accepted":
        return _timestamp(submission.get("updated_at"))
    return None


def submissions_snapshot(service: Any, round_id: str) -> dict:
    row = service._round(round_id)
    round_status = str(row.get("status") or "")
    stage1_scores = _stage1_scores(service, row)
    final_scores = _rankings(row, "final_ranking") if round_status == "published" else {}
    champion_id = _champion_submission_id(row)
    participants = {
        str(item.get("submission_id")): item for item in _participants(row)
        if item.get("submission_id")
    }
    records = service._store.list_submissions(
        round_id,
        columns=(
            "submission_id,round_id,miner_hotkey,status,is_king,updated_at,"
            "accepted_at,frozen_at,source_ref,consent"
        ),
    )
    submissions = []
    for submission in records:
        raw_status = str(submission.get("status") or "")
        if raw_status not in ("accepted", "frozen"):
            continue
        submission_id = str(submission.get("submission_id") or "")
        participant = participants.get(submission_id)
        # Open accepted intake is public by explicit dashboard policy.  After
        # cutoff, only the immutable frozen participant set can be enumerated.
        if round_status != "open" and participant is None:
            continue
        is_baseline = bool(
            (participant or {}).get(
                "is_baseline",
                (participant or {}).get("is_king", submission.get("is_king", False)),
            )
        )
        is_champion = bool(
            round_status == "published"
            and not is_baseline
            and champion_id == submission_id
        )
        final = final_scores.get(submission_id) or {}
        final_score = _score(final.get("final_score"))
        submissions.append(
            {
                "submission_id": submission_id,
                "miner_hotkey": str(submission.get("miner_hotkey") or ""),
                "is_baseline": is_baseline,
                "status": _submission_lifecycle(
                    raw_status=raw_status,
                    round_status=round_status,
                    is_champion=is_champion,
                    final_score=final_score,
                ),
                "submitted_at": _submitted_at(submission),
                "stage1_score": stage1_scores.get(submission_id),
                "final_score": final_score,
                "is_champion": is_champion,
                "code": source_disclosure.disclosure_status(
                    submission, service.now(), round_row=row
                ),
            }
        )
    return {"round_id": round_id, "submissions": submissions}


__all__ = [
    "competition_snapshot",
    "round_summary",
    "submissions_snapshot",
]
