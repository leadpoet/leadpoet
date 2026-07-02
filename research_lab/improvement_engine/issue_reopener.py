"""Issue reopen policy helpers."""

from __future__ import annotations

from .models import EngineIssue


def should_reopen(issue: EngineIssue, *, recurrence_count: int) -> bool:
    return issue.status in {"fixed", "ignored"} and recurrence_count >= 1 and issue.priority in {"high", "critical"}
