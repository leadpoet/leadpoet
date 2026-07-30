from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from gateway.research_lab import maintenance as maintenance_mod
from gateway.research_lab import store as store_mod
from gateway.research_lab import worker as worker_mod
from gateway.research_lab.config import ResearchLabGatewayConfig


RUN_ID = "11111111-1111-4111-8111-111111111111"
TICKET_ID = "22222222-2222-4222-8222-222222222222"
QUEUE_HASH = "sha256:" + "a" * 64


@pytest.mark.asyncio
async def test_append_event_with_seq_preserves_supplied_event_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows: list[dict[str, Any]] = []
    event_id = "33333333-3333-4333-8333-333333333333"

    async def fake_next_seq(table: str, key_field: str, key_value: Any) -> int:
        return 4

    async def fake_insert(table: str, row: dict[str, Any]) -> dict[str, Any]:
        rows.append(dict(row))
        return dict(row)

    monkeypatch.setattr(store_mod, "next_event_seq", fake_next_seq)
    monkeypatch.setattr(store_mod, "insert_row", fake_insert)

    result = await store_mod.append_event_with_seq(
        "events",
        "run_id",
        RUN_ID,
        lambda seq: {"run_id": RUN_ID, "seq": seq},
        event_id=event_id,
    )

    assert result["event_id"] == event_id
    assert rows[0]["event_id"] == event_id


def _queue_row(status: str, event_hash: str = QUEUE_HASH) -> dict[str, Any]:
    return {
        "run_id": RUN_ID,
        "current_queue_status": status,
        "current_event_hash": event_hash,
        "current_event_seq": 4,
        "current_reason": "blocked_for_credit",
        "worker_ref": "worker-a",
    }


def _loop_row(status: str) -> dict[str, Any]:
    return {
        "run_id": RUN_ID,
        "current_loop_status": status,
        "current_event_type": "checkpoint_saved",
        "current_event_seq": 7,
        "current_event_hash": "sha256:" + "b" * 64,
    }


@pytest.mark.asyncio
async def test_loop_transition_recovers_committed_insert_after_response_loss(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inserted: dict[str, dict[str, Any]] = {}
    insert_calls = 0

    async def fake_select_one(table: str, **kwargs: Any) -> dict[str, Any] | None:
        if table == "research_lab_auto_research_loop_events":
            event_id = str(kwargs["filters"][0][1])
            return inserted.get(event_id)
        if table == "research_loop_run_queue_current":
            return _queue_row("paused")
        if table == "research_lab_auto_research_loop_current":
            return _loop_row("running")
        raise AssertionError(table)

    async def fake_create_loop_event(**kwargs: Any) -> dict[str, Any]:
        nonlocal insert_calls
        insert_calls += 1
        row = {
            **kwargs,
            "provider_usage": kwargs.get("provider_usage") or [],
            "cost_ledger": kwargs.get("cost_ledger") or {},
            "seq": 8,
            "anchored_hash": "sha256:" + "c" * 64,
        }
        inserted[str(kwargs["event_id"])] = row
        raise TimeoutError("response was lost after commit")

    monkeypatch.setattr(store_mod, "select_one", fake_select_one)
    monkeypatch.setattr(
        store_mod, "create_auto_research_loop_event", fake_create_loop_event
    )

    kwargs = {
        "run_id": RUN_ID,
        "ticket_id": TICKET_ID,
        "receipt_id": "33333333-3333-4333-8333-333333333333",
        "event_type": "loop_paused",
        "loop_status": "paused",
        "worker_ref": "worker-a",
        "expected_queue_status": "paused",
        "queue_event_hash": QUEUE_HASH,
        "event_doc": {"reason": "blocked_for_credit"},
    }
    first = await store_mod.ensure_auto_research_loop_transition_event(**kwargs)
    second = await store_mod.ensure_auto_research_loop_transition_event(
        **{
            **kwargs,
            "worker_ref": "operator-reconciler",
            "event_doc": {"source": "paused_queue_reconciler"},
        }
    )

    assert insert_calls == 1
    assert first == second
    assert first["event_doc"]["queue_event_hash"] == QUEUE_HASH
    assert first["event_doc"]["queue_status"] == "paused"
    assert first["event_doc"]["loop_transition_id"] == first["event_id"]


@pytest.mark.asyncio
async def test_loop_transition_rejects_superseded_queue_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_select_one(table: str, **kwargs: Any) -> dict[str, Any] | None:
        if table == "research_lab_auto_research_loop_events":
            return None
        if table == "research_loop_run_queue_current":
            return _queue_row("queued", "sha256:" + "d" * 64)
        raise AssertionError(table)

    async def forbidden_create(**kwargs: Any) -> dict[str, Any]:
        raise AssertionError("superseded transition must not append")

    monkeypatch.setattr(store_mod, "select_one", fake_select_one)
    monkeypatch.setattr(store_mod, "create_auto_research_loop_event", forbidden_create)

    with pytest.raises(RuntimeError, match="queue authority was superseded"):
        await store_mod.ensure_auto_research_loop_transition_event(
            run_id=RUN_ID,
            ticket_id=TICKET_ID,
            event_type="loop_paused",
            loop_status="paused",
            worker_ref="worker-a",
            expected_queue_status="paused",
            queue_event_hash=QUEUE_HASH,
        )


@pytest.mark.asyncio
async def test_pause_transition_coalesces_an_already_paused_projection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_select_one(table: str, **kwargs: Any) -> dict[str, Any] | None:
        if table == "research_lab_auto_research_loop_events":
            return None
        if table == "research_loop_run_queue_current":
            return _queue_row("paused")
        if table == "research_lab_auto_research_loop_current":
            return _loop_row("paused")
        raise AssertionError(table)

    async def forbidden_create(**kwargs: Any) -> dict[str, Any]:
        raise AssertionError("an already-paused projection must not duplicate")

    monkeypatch.setattr(store_mod, "select_one", fake_select_one)
    monkeypatch.setattr(store_mod, "create_auto_research_loop_event", forbidden_create)

    result = await store_mod.ensure_auto_research_loop_transition_event(
        run_id=RUN_ID,
        ticket_id=TICKET_ID,
        event_type="loop_paused",
        loop_status="paused",
        worker_ref="worker-a",
        expected_queue_status="paused",
        queue_event_hash=QUEUE_HASH,
        coalesce_current_status=True,
    )

    assert result["already_projected"] is True
    assert result["loop_status"] == "paused"


@pytest.mark.asyncio
async def test_resume_transition_records_execution_even_if_legacy_status_is_running(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    writes: list[dict[str, Any]] = []

    async def fake_select_one(table: str, **kwargs: Any) -> dict[str, Any] | None:
        if table == "research_lab_auto_research_loop_events":
            return None
        if table == "research_loop_run_queue_current":
            return _queue_row("started")
        if table == "research_lab_auto_research_loop_current":
            return _loop_row("running")
        raise AssertionError(table)

    async def fake_create_loop_event(**kwargs: Any) -> dict[str, Any]:
        writes.append(dict(kwargs))
        return {
            **kwargs,
            "seq": 8,
            "anchored_hash": "sha256:" + "e" * 64,
        }

    monkeypatch.setattr(store_mod, "select_one", fake_select_one)
    monkeypatch.setattr(
        store_mod, "create_auto_research_loop_event", fake_create_loop_event
    )

    result = await store_mod.ensure_auto_research_loop_transition_event(
        run_id=RUN_ID,
        ticket_id=TICKET_ID,
        event_type="loop_resumed",
        loop_status="running",
        worker_ref="worker-a",
        expected_queue_status="started",
        queue_event_hash=QUEUE_HASH,
        event_doc={"checkpoint_hash": "sha256:" + "f" * 64},
    )

    assert result["event_type"] == "loop_resumed"
    assert len(writes) == 1


@pytest.mark.asyncio
async def test_nonterminal_transition_cannot_resurrect_terminal_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_select_one(table: str, **kwargs: Any) -> dict[str, Any] | None:
        if table == "research_lab_auto_research_loop_events":
            return None
        if table == "research_loop_run_queue_current":
            return _queue_row("paused")
        if table == "research_lab_auto_research_loop_current":
            return _loop_row("completed")
        raise AssertionError(table)

    monkeypatch.setattr(store_mod, "select_one", fake_select_one)

    with pytest.raises(RuntimeError, match="after terminal loop status completed"):
        await store_mod.ensure_auto_research_loop_transition_event(
            run_id=RUN_ID,
            ticket_id=TICKET_ID,
            event_type="loop_paused",
            loop_status="paused",
            worker_ref="worker-a",
            expected_queue_status="paused",
            queue_event_hash=QUEUE_HASH,
        )


def _hosted_context() -> worker_mod.HostedRunContext:
    return worker_mod.HostedRunContext(
        queue_row={
            "run_id": RUN_ID,
            "ticket_id": TICKET_ID,
            "queue_priority": 2,
        },
        ticket={
            "miner_hotkey": "5F3sa2TJAWMqDhXG6jhV4N8ko9SxwGy8TpaNS1repo5EYjQX",
            "ticket_doc": {},
        },
        payment=None,
        receipt_id="33333333-3333-4333-8333-333333333333",
    )


@pytest.mark.asyncio
async def test_credit_block_transition_projects_loop_before_reporting_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worker = worker_mod.ResearchLabHostedWorker(
        ResearchLabGatewayConfig(),
        worker_ref="worker-a",
    )
    context = _hosted_context()
    calls: list[tuple[str, dict[str, Any]]] = []

    async def fake_checkpoint(run_id: str) -> dict[str, Any]:
        assert run_id == RUN_ID
        return {"checkpoint_hash": "sha256:" + "f" * 64}

    async def record(name: str, **kwargs: Any) -> dict[str, Any]:
        calls.append((name, dict(kwargs)))
        if name == "queue":
            return {"anchored_hash": QUEUE_HASH, "seq": 5}
        return {}

    async def fake_receipt(**kwargs: Any) -> dict[str, Any]:
        return await record("receipt", **kwargs)

    async def fake_queue(**kwargs: Any) -> dict[str, Any]:
        return await record("queue", **kwargs)

    async def fake_loop(**kwargs: Any) -> dict[str, Any]:
        return await record("loop", **kwargs)

    async def fake_ticket(**kwargs: Any) -> dict[str, Any]:
        return await record("ticket", **kwargs)

    async def fake_public(*args: Any, **kwargs: Any) -> None:
        calls.append(("public", dict(kwargs)))

    monkeypatch.setattr(worker_mod, "latest_auto_research_checkpoint", fake_checkpoint)
    monkeypatch.setattr(worker_mod, "create_receipt_event", fake_receipt)
    monkeypatch.setattr(worker_mod, "create_queue_event", fake_queue)
    monkeypatch.setattr(
        worker_mod, "ensure_auto_research_loop_transition_event", fake_loop
    )
    monkeypatch.setattr(worker_mod, "create_ticket_event", fake_ticket)
    monkeypatch.setattr(worker_mod, "safe_project_public_loop_activity", fake_public)

    outcome = await worker._mark_blocked_for_credit(
        context,
        "OpenRouter returned HTTP 402",
    )

    assert outcome.status == "blocked_for_credit"
    assert [name for name, _kwargs in calls] == [
        "receipt",
        "queue",
        "loop",
        "ticket",
        "public",
    ]
    loop_call = calls[2][1]
    assert loop_call["event_type"] == "loop_paused"
    assert loop_call["loop_status"] == "paused"
    assert loop_call["expected_queue_status"] == "paused"
    assert loop_call["queue_event_hash"] == QUEUE_HASH
    assert loop_call["coalesce_current_status"] is True


@pytest.mark.asyncio
async def test_resume_uses_current_heartbeat_head_as_queue_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worker = worker_mod.ResearchLabHostedWorker(
        ResearchLabGatewayConfig(),
        worker_ref="worker-a",
    )
    context = _hosted_context()
    heartbeat_hash = "sha256:" + "9" * 64

    async def fake_select_one(table: str, **kwargs: Any) -> dict[str, Any]:
        assert table == "research_loop_run_queue_current"
        return {
            "run_id": RUN_ID,
            "current_queue_status": "started",
            "worker_ref": "worker-a",
            "current_event_hash": heartbeat_hash,
            "current_event_seq": 6,
        }

    monkeypatch.setattr(worker_mod, "select_one", fake_select_one)

    queue_event_hash = await worker._current_started_queue_event_hash(context)

    assert queue_event_hash == heartbeat_hash


@pytest.mark.asyncio
async def test_resumed_loop_event_binds_to_heartbeat_after_claim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worker = worker_mod.ResearchLabHostedWorker(
        ResearchLabGatewayConfig(),
        worker_ref="worker-a",
    )
    context = _hosted_context()
    heartbeat_hash = "sha256:" + "9" * 64
    writes: list[dict[str, Any]] = []

    async def fake_select_one(table: str, **kwargs: Any) -> dict[str, Any]:
        assert table == "research_loop_run_queue_current"
        return {
            "run_id": RUN_ID,
            "current_queue_status": "started",
            "worker_ref": "worker-a",
            "current_event_hash": heartbeat_hash,
            "current_event_seq": 6,
        }

    async def fake_transition(**kwargs: Any) -> dict[str, Any]:
        writes.append(dict(kwargs))
        return {
            "event_type": "loop_resumed",
            "seq": 8,
            "anchored_hash": "sha256:" + "e" * 64,
        }

    monkeypatch.setattr(worker_mod, "select_one", fake_select_one)
    monkeypatch.setattr(
        worker_mod,
        "ensure_auto_research_loop_transition_event",
        fake_transition,
    )

    event = SimpleNamespace(
        event_type="loop_resumed",
        loop_status="running",
        node_id=None,
        elapsed_seconds=12.5,
        candidate_artifact_hash=None,
        candidate_patch_hash=None,
        cost_ledger={"spent_microusd": 10},
    )
    result = await worker._persist_loop_event(
        context=context,
        event=event,
        event_doc={"checkpoint_hash": "sha256:" + "f" * 64},
        event_provider_usage=[],
    )

    assert result["event_type"] == "loop_resumed"
    assert len(writes) == 1
    assert writes[0]["queue_event_hash"] == heartbeat_hash
    assert writes[0]["expected_queue_status"] == "started"


@pytest.mark.asyncio
async def test_resume_rejects_heartbeat_head_owned_by_another_worker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worker = worker_mod.ResearchLabHostedWorker(
        ResearchLabGatewayConfig(),
        worker_ref="worker-a",
    )
    context = _hosted_context()

    async def fake_select_one(table: str, **kwargs: Any) -> dict[str, Any]:
        assert table == "research_loop_run_queue_current"
        return {
            "run_id": RUN_ID,
            "current_queue_status": "started",
            "worker_ref": "worker-b",
            "current_event_hash": "sha256:" + "9" * 64,
            "current_event_seq": 6,
        }

    monkeypatch.setattr(worker_mod, "select_one", fake_select_one)

    with pytest.raises(
        worker_mod.HostedResearchLabClaimLost,
        match="superseded before loop resume",
    ):
        await worker._current_started_queue_event_hash(context)

    assert context.claim_lost is True


@pytest.mark.asyncio
async def test_periodic_worker_reconciles_paused_queue_projection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worker = worker_mod.ResearchLabHostedWorker(
        ResearchLabGatewayConfig(),
        worker_ref="worker-a",
    )
    writes: list[dict[str, Any]] = []

    async def fake_select_many(table: str, **kwargs: Any) -> list[dict[str, Any]]:
        assert table == "research_lab_auto_research_loop_current"
        return [
            {
                "run_id": RUN_ID,
                "ticket_id": TICKET_ID,
                "receipt_id": "33333333-3333-4333-8333-333333333333",
                "current_loop_status": "running",
            }
        ]

    async def fake_select_one(table: str, **kwargs: Any) -> dict[str, Any]:
        assert table == "research_loop_run_queue_current"
        return {
            **_queue_row("paused"),
            "ticket_id": TICKET_ID,
        }

    async def fake_loop_transition(**kwargs: Any) -> dict[str, Any]:
        writes.append(dict(kwargs))
        return {"anchored_hash": "sha256:" + "c" * 64, "seq": 8}

    monkeypatch.setattr(worker_mod, "select_many", fake_select_many)
    monkeypatch.setattr(worker_mod, "select_one", fake_select_one)
    monkeypatch.setattr(
        worker_mod,
        "ensure_auto_research_loop_transition_event",
        fake_loop_transition,
    )

    reconciled = await worker._reconcile_stale_loop_projections()

    assert reconciled == 1
    assert len(writes) == 1
    assert writes[0]["event_type"] == "loop_paused"
    assert writes[0]["queue_event_hash"] == QUEUE_HASH


@pytest.mark.asyncio
async def test_operator_reconciler_uses_same_idempotent_transition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    writes: list[dict[str, Any]] = []

    async def fake_select_all(table: str, **kwargs: Any) -> list[dict[str, Any]]:
        assert table == "research_loop_run_queue_current"
        return [
            {
                **_queue_row("paused"),
                "ticket_id": TICKET_ID,
                "current_status_at": "2026-07-29T00:00:00+00:00",
            }
        ]

    async def fake_select_one(table: str, **kwargs: Any) -> dict[str, Any]:
        assert table == "research_lab_auto_research_loop_current"
        return {
            **_loop_row("running"),
            "ticket_id": TICKET_ID,
            "receipt_id": "33333333-3333-4333-8333-333333333333",
            "current_status_at": "2026-07-29T00:00:00+00:00",
        }

    async def fake_transition(**kwargs: Any) -> dict[str, Any]:
        writes.append(dict(kwargs))
        return {"seq": 8, "anchored_hash": "sha256:" + "c" * 64}

    async def noop_public(*args: Any, **kwargs: Any) -> None:
        return None

    monkeypatch.setattr(maintenance_mod, "select_all", fake_select_all)
    monkeypatch.setattr(maintenance_mod, "select_one", fake_select_one)
    monkeypatch.setattr(
        maintenance_mod,
        "ensure_auto_research_loop_transition_event",
        fake_transition,
    )
    monkeypatch.setattr(
        maintenance_mod, "safe_project_public_loop_activity", noop_public
    )

    result = await maintenance_mod.reconcile_paused_loop_projections(
        dry_run=False,
        actor_ref="operator-a",
    )

    assert result["ok"] is True
    assert result["repaired_count"] == 1
    assert writes[0]["queue_event_hash"] == QUEUE_HASH
    assert writes[0]["expected_queue_status"] == "paused"
    assert writes[0]["coalesce_current_status"] is True
