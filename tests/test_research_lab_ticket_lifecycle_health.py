from __future__ import annotations

import pytest

from gateway.research_lab import maintenance


@pytest.mark.asyncio
async def test_ticket_lifecycle_health_reports_open_ticket_with_terminal_queues(monkeypatch) -> None:
    async def fake_select_all(table: str, **kwargs):
        if table == "research_loop_run_queue_current":
            return [
                {
                    "run_id": "run-1",
                    "ticket_id": "ticket-1",
                    "current_queue_status": "completed",
                    "current_event_hash": "queue-hash-1",
                    "current_status_at": "2026-07-06T10:10:00+00:00",
                },
                {
                    "run_id": "run-2",
                    "ticket_id": "ticket-1",
                    "current_queue_status": "failed",
                    "current_event_hash": "queue-hash-2",
                    "current_status_at": "2026-07-06T10:15:00+00:00",
                },
            ]
        assert table == "research_loop_ticket_current"
        return [
            {
                "ticket_id": "ticket-1",
                "miner_hotkey": "hotkey-1",
                "requested_loop_count": 2,
                "current_ticket_status": "running",
                "current_event_hash": "ticket-event-hash",
                "current_status_at": "2026-07-06T10:00:00+00:00",
                "created_at": "2026-07-06T09:00:00+00:00",
            }
        ]

    async def fake_select_many(table: str, **kwargs):
        raise AssertionError(f"unexpected select_many call: {table}")

    monkeypatch.setattr(maintenance, "select_all", fake_select_all)
    monkeypatch.setattr(maintenance, "select_many", fake_select_many)

    health = await maintenance.ticket_lifecycle_health(sample_limit=10)

    assert health["ok"] is False
    assert health["open_ticket_count"] == 1
    assert health["terminal_queue_open_ticket_count"] == 1
    assert health["samples"][0]["target_ticket_status"] == "completed"
    assert health["samples"][0]["queue_status_counts"] == {"completed": 1, "failed": 1}


@pytest.mark.asyncio
async def test_reconcile_terminal_ticket_statuses_writes_append_only_event(monkeypatch) -> None:
    events: list[dict] = []
    projections: list[tuple[str, str]] = []

    async def fake_select_all(table: str, **kwargs):
        if table == "research_loop_run_queue_current":
            return [
                {
                    "run_id": "run-3",
                    "ticket_id": "ticket-2",
                    "current_queue_status": "failed",
                    "current_event_hash": "queue-hash-3",
                    "current_status_at": "2026-07-06T10:15:00+00:00",
                }
            ]
        assert table == "research_loop_ticket_current"
        return [
            {
                "ticket_id": "ticket-2",
                "miner_hotkey": "hotkey-2",
                "requested_loop_count": 1,
                "current_ticket_status": "running",
                "current_event_hash": "ticket-event-hash",
                "current_status_at": "2026-07-06T10:00:00+00:00",
                "created_at": "2026-07-06T09:00:00+00:00",
            }
        ]

    async def fake_select_many(table: str, **kwargs):
        raise AssertionError(f"unexpected select_many call: {table}")

    async def fake_select_one(table: str, **kwargs):
        assert table == "research_loop_ticket_current"
        return {
            "ticket_id": "ticket-2",
            "current_ticket_status": "running",
            "current_event_hash": "ticket-event-hash",
            "current_status_at": "2026-07-06T10:00:00+00:00",
        }

    async def fake_create_ticket_event(**kwargs):
        events.append(kwargs)
        return {"seq": 7, "anchored_hash": "new-ticket-event-hash"}

    async def fake_project_public_loop_activity(ticket_id: str, **kwargs):
        projections.append((ticket_id, str(kwargs.get("source_ref") or "")))

    monkeypatch.setattr(maintenance, "select_all", fake_select_all)
    monkeypatch.setattr(maintenance, "select_many", fake_select_many)
    monkeypatch.setattr(maintenance, "select_one", fake_select_one)
    monkeypatch.setattr(maintenance, "create_ticket_event", fake_create_ticket_event)
    monkeypatch.setattr(maintenance, "safe_project_public_loop_activity", fake_project_public_loop_activity)

    result = await maintenance.reconcile_terminal_ticket_statuses(dry_run=False, actor_ref="test-operator")

    assert result["ok"] is True
    assert result["repaired_count"] == 1
    assert events[0]["ticket_id"] == "ticket-2"
    assert events[0]["event_type"] == "failed"
    assert events[0]["reason"] == "terminal_ticket_status_reconciler"
    assert events[0]["event_doc"]["actor_ref"] == "test-operator"
    assert projections == [("ticket-2", "terminal_ticket_status_reconciler:ticket-2")]


@pytest.mark.asyncio
async def test_reconcile_terminal_ticket_statuses_skips_if_ticket_already_terminal(monkeypatch) -> None:
    events: list[dict] = []

    async def fake_select_all(table: str, **kwargs):
        if table == "research_loop_run_queue_current":
            return [
                {
                    "run_id": "run-5",
                    "ticket_id": "ticket-5",
                    "current_queue_status": "completed",
                    "current_event_hash": "queue-hash-5",
                    "current_status_at": "2026-07-06T10:15:00+00:00",
                }
            ]
        assert table == "research_loop_ticket_current"
        return [
            {
                "ticket_id": "ticket-5",
                "miner_hotkey": "hotkey-5",
                "requested_loop_count": 1,
                "current_ticket_status": "running",
                "current_event_hash": "ticket-event-hash",
                "current_status_at": "2026-07-06T10:00:00+00:00",
                "created_at": "2026-07-06T09:00:00+00:00",
            }
        ]

    async def fake_select_many(table: str, **kwargs):
        raise AssertionError(f"unexpected select_many call: {table}")

    async def fake_select_one(table: str, **kwargs):
        assert table == "research_loop_ticket_current"
        return {
            "ticket_id": "ticket-5",
            "current_ticket_status": "completed",
            "current_event_hash": "newer-ticket-event-hash",
            "current_status_at": "2026-07-06T10:20:00+00:00",
        }

    async def fake_create_ticket_event(**kwargs):
        events.append(kwargs)
        return {"seq": 8, "anchored_hash": "should-not-write"}

    monkeypatch.setattr(maintenance, "select_all", fake_select_all)
    monkeypatch.setattr(maintenance, "select_many", fake_select_many)
    monkeypatch.setattr(maintenance, "select_one", fake_select_one)
    monkeypatch.setattr(maintenance, "create_ticket_event", fake_create_ticket_event)

    result = await maintenance.reconcile_terminal_ticket_statuses(dry_run=False, actor_ref="test-operator")

    assert result["ok"] is True
    assert result["planned_count"] == 1
    assert result["repaired_count"] == 0
    assert result["skipped_count"] == 1
    assert result["skipped"][0]["reason"] == "ticket_already_terminal"
    assert events == []


@pytest.mark.asyncio
async def test_reconcile_terminal_ticket_statuses_skips_missing_expected_runs(monkeypatch) -> None:
    async def fake_select_all(table: str, **kwargs):
        if table == "research_loop_run_queue_current":
            return [{"run_id": "run-4", "ticket_id": "ticket-3", "current_queue_status": "completed"}]
        assert table == "research_loop_ticket_current"
        return [
            {
                "ticket_id": "ticket-3",
                "requested_loop_count": 2,
                "current_ticket_status": "running",
            }
        ]

    async def fake_select_many(table: str, **kwargs):
        raise AssertionError(f"unexpected select_many call: {table}")

    monkeypatch.setattr(maintenance, "select_all", fake_select_all)
    monkeypatch.setattr(maintenance, "select_many", fake_select_many)

    result = await maintenance.reconcile_terminal_ticket_statuses(dry_run=True)

    assert result["ok"] is True
    assert result["planned_count"] == 0
