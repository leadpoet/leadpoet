"""UTC next-day disclosure for a round's complete 20-ICP bank."""

from __future__ import annotations

import math
from datetime import date, datetime, time, timedelta, timezone
from typing import Any, Mapping, Sequence

from lab_arena import contracts


DISCLOSURE_POLICY = "all_20_next_day"


def _date(value: Any) -> date | None:
    try:
        raw = str(value or "")
        parsed = date.fromisoformat(raw)
    except (TypeError, ValueError):
        return None
    return parsed if raw == parsed.isoformat() else None


def _instant(value: Any) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(str(value or "").replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(timezone.utc)


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def disclosure_metadata(round_row: Mapping[str, Any]) -> dict | None:
    """Return stable bank and publication dates without inspecting private ICPs.

    New rounds persist ``icp_set_date`` at commitment and become public at
    their exact submission cutoff. Legacy rounds have no such marker: their
    actual bank date was ``evaluation_date``, so waiting until the next UTC
    day avoids interpreting their old cutoff as an early publication time.
    """

    schedule = (round_row.get("configuration_doc") or {}).get("schedule") or {}
    submission_open = _instant(schedule.get("submission_open"))
    cutoff = _instant(schedule.get("submission_cutoff"))
    explicit_bank_date = _date(round_row.get("icp_set_date"))
    evaluation_date = _date(round_row.get("evaluation_date"))
    planned = (
        explicit_bank_date is None
        and evaluation_date is None
        and round_row.get("status") == "open"
    )
    bank_date = (
        explicit_bank_date
        or evaluation_date
        or (submission_open.date() if planned and submission_open else None)
    )
    if bank_date is None:
        return None
    if explicit_bank_date is not None or planned:
        next_day = bank_date + timedelta(days=1)
        if (
            cutoff is None
            or submission_open is None
            or submission_open.date() != bank_date
            or cutoff.date() != next_day
            or cutoff <= submission_open
            or (
                explicit_bank_date is not None
                and evaluation_date != cutoff.date()
            )
        ):
            return None
        public_at = max(
            cutoff,
            datetime.combine(next_day, time.min, tzinfo=timezone.utc),
        )
    else:
        public_at = datetime.combine(
            bank_date + timedelta(days=1), time.min, tzinfo=timezone.utc
        )
    return {
        "icp_set_date": bank_date.isoformat(),
        "public_at": _iso(public_at),
        "disclosure_policy": DISCLOSURE_POLICY,
        "evaluation_date": (
            evaluation_date.isoformat()
            if evaluation_date
            else (cutoff.date().isoformat() if planned and cutoff else None)
        ),
        "planned": planned,
    }


def _baseline_scores(
    round_row: Mapping[str, Any], runs: Sequence[Mapping[str, Any]]
) -> dict[int, float]:
    baselines = [
        participant
        for participant in round_row.get("participants") or []
        if participant.get("is_king") is True
    ]
    if len(baselines) != 1:
        return {}
    baseline_id = baselines[0].get("submission_id")
    selected: dict[int, Mapping[str, Any]] = {}
    for run in runs:
        if run.get("submission_id") != baseline_id or run.get("per_icp_score") is None:
            continue
        position = run.get("icp_position")
        if (
            isinstance(position, bool)
            or not isinstance(position, int)
            or position not in range(contracts.BENCHMARK_ICP_COUNT)
            or run.get("kind", "execute") != "execute"
        ):
            continue
        current = selected.get(position)
        if current is None or int(run.get("attempt") or 0) > int(current.get("attempt") or 0):
            selected[position] = run
    scores: dict[int, float] = {}
    for position, run in selected.items():
        value = run["per_icp_score"]
        if isinstance(value, bool):
            continue
        try:
            score = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(score) and 0 <= score <= 100:
            scores[position] = score
    return scores


def baseline_disclosure(
    round_row: Mapping[str, Any],
    runs: Sequence[Mapping[str, Any]],
    now: datetime | None = None,
) -> dict | None:
    """Disclose all 20 positions only after the bank's next-day freeze."""

    metadata = disclosure_metadata(round_row)
    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None:
        return None
    current = current.astimezone(timezone.utc)
    public_at = _instant(metadata.get("public_at")) if metadata else None
    if (
        metadata is None
        or public_at is None
        or current < public_at
        or round_row.get("status") == "open"
    ):
        return None
    return {
        **metadata,
        "baseline_submission_id": next(
            (
                participant.get("submission_id")
                for participant in round_row.get("participants") or []
                if participant.get("is_king") is True
            ),
            None,
        ),
        "public_positions": list(range(contracts.BENCHMARK_ICP_COUNT)),
        "private_positions": [],
        "baseline_scores": (
            _baseline_scores(round_row, runs)
            if round_row.get("status") == "published"
            else {}
        ),
    }
