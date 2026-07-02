"""Notification helpers for Engine issues."""

from __future__ import annotations

import json
import logging
from typing import Any
from urllib import request

from .config import ImprovementEngineConfig
from .models import EngineIssue


logger = logging.getLogger(__name__)


def notify_issue(issue: EngineIssue, config: ImprovementEngineConfig) -> None:
    if not config.notify_webhook_url:
        return
    payload: dict[str, Any] = {
        "type": "engine.issue.opened",
        "priority": issue.priority,
        "title": issue.title,
        "category": issue.category,
        "occurrence_count": issue.occurrence_count,
        "last_seen_at": issue.last_seen_at,
    }
    try:
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        req = request.Request(config.notify_webhook_url, data=body, method="POST", headers={"Content-Type": "application/json"})
        with request.urlopen(req, timeout=10):
            return
    except Exception as exc:
        logger.warning("improvement_engine_notification_failed issue_key=%s error=%s", issue.issue_key, str(exc)[:200])
