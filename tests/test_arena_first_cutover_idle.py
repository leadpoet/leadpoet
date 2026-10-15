"""The one-time legacy worker cutover requires a provably idle review window."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from scripts.arena_first_cutover_idle import require_idle


def test_first_cutover_allows_only_distant_open_cutoff_without_claims():
    now = datetime(2026, 10, 15, tzinfo=timezone.utc)
    round_row = {"round_id": "r", "status": "open"}
    claims = []
    service = SimpleNamespace(
        active_rounds=lambda: [{"round_id": "r"}],
        _round=lambda _id: round_row,
        _submission_replacement_cutoff=lambda _round: now + timedelta(hours=3),
        _is_daily_baseline=lambda _row, _round: False,
        _store=SimpleNamespace(list_submissions=lambda _id, *, status: claims if status == "accepted" else []),
    )
    require_idle(service, now=now)
    claims.append({"code_review_claim": "in-flight"})
    with pytest.raises(ValueError, match="claimed"):
        require_idle(service, now=now)
    claims.clear()
    round_row["status"] = "stage2"
    claims.append({"code_review_claim": None, "code_review_status": "error"})
    with pytest.raises(ValueError, match="can still start"):
        require_idle(service, now=now)
    claims[0]["code_review_status"] = "passed"
    require_idle(service, now=now)


def test_first_cutover_rejects_near_review_cutoff():
    now = datetime(2026, 10, 15, tzinfo=timezone.utc)
    service = SimpleNamespace(
        active_rounds=lambda: [{"round_id": "r"}],
        _round=lambda _id: {"round_id": "r", "status": "open"},
        _submission_replacement_cutoff=lambda _round: now + timedelta(minutes=9),
    )
    with pytest.raises(ValueError, match="too near"):
        require_idle(service, now=now)
