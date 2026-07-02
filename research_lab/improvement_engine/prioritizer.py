"""Priority scoring for deterministic Engine issue clusters."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Sequence

from .models import EngineTraceEvent


CRITICAL_COMPONENTS = {"scoring_worker", "score_bundle", "sourcing_model", "code_build", "promotion"}


def _parse_ts(value: str) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def severity_score(events: Sequence[EngineTraceEvent]) -> float:
    if not events:
        return 0.0
    occurrence_weight = min(len(events) / 10.0, 3.0)
    losses = [abs(float(event.score_delta)) for event in events if event.score_delta is not None and event.score_delta < 0]
    quality_impact_weight = max(0.1, max(losses) if losses else max(event.severity_hint for event in events))
    affected_component_weight = 1.5 if any(event.component in CRITICAL_COMPONENTS for event in events) else 1.0
    parsed = [_parse_ts(event.event_at) for event in events]
    recent = [value for value in parsed if value is not None]
    if recent:
        hours = (datetime.now(timezone.utc) - max(recent)).total_seconds() / 3600.0
        recency_weight = 1.0 if hours < 24 else 0.6
    else:
        recency_weight = 0.6
    return round(occurrence_weight * quality_impact_weight * affected_component_weight * recency_weight, 4)


def priority_for_score(score: float, *, category: str) -> str:
    if "secret" in category or score >= 8.0:
        return "critical"
    if score >= 4.0:
        return "high"
    if score >= 1.5:
        return "medium"
    return "low"
