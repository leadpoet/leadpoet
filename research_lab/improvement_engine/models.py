"""Typed payloads for Improvement Engine issue detection."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping


ENGINE_VERSION = "leadpoet-improvement-engine:v1"


@dataclass(frozen=True)
class EngineTraceEvent:
    failure_category: str
    runtime_stage: str
    normalized_failure_reason: str
    component: str = "unknown"
    run_id: str = ""
    ticket_id: str = ""
    candidate_id: str = ""
    score_bundle_hash: str = ""
    trace_id: str = ""
    execution_trace_ref: str = ""
    event_type: str = ""
    event_at: str = ""
    severity_hint: float = 1.0
    score_delta: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EngineIssue:
    issue_key: str
    title: str
    status: str
    priority: str
    category: str
    fingerprint: str
    first_seen_at: str
    last_seen_at: str
    occurrence_count: int
    severity_score: float
    confidence: float
    root_cause_doc: Mapping[str, Any]
    suggested_fix_doc: Mapping[str, Any]
    evaluator_spec_doc: Mapping[str, Any]
    dataset_spec_doc: Mapping[str, Any]
    linked_trace_ids: tuple[str, ...] = ()
    linked_score_bundle_hashes: tuple[str, ...] = ()
    linked_run_ids: tuple[str, ...] = ()
    linked_ticket_ids: tuple[str, ...] = ()
    created_by_engine_version: str = ENGINE_VERSION

    def to_row(self) -> dict[str, Any]:
        return {
            "issue_key": self.issue_key,
            "title": self.title,
            "status": self.status,
            "priority": self.priority,
            "category": self.category,
            "fingerprint": self.fingerprint,
            "first_seen_at": self.first_seen_at,
            "last_seen_at": self.last_seen_at,
            "occurrence_count": self.occurrence_count,
            "severity_score": self.severity_score,
            "confidence": self.confidence,
            "root_cause_doc": dict(self.root_cause_doc),
            "suggested_fix_doc": dict(self.suggested_fix_doc),
            "evaluator_spec_doc": dict(self.evaluator_spec_doc),
            "dataset_spec_doc": dict(self.dataset_spec_doc),
            "linked_trace_ids": list(self.linked_trace_ids),
            "linked_score_bundle_hashes": list(self.linked_score_bundle_hashes),
            "linked_run_ids": list(self.linked_run_ids),
            "linked_ticket_ids": list(self.linked_ticket_ids),
            "created_by_engine_version": self.created_by_engine_version,
        }


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
