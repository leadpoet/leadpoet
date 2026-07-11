from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from fastapi import HTTPException

from gateway.research_lab import admin, api, maintenance, public_activity, worker as worker_mod
from gateway.research_lab.config import ResearchLabGatewayConfig
from gateway.research_lab.models import (
    ResearchLabLoopStartRequest,
    ResearchLabLoopTopUpRequest,
    ResearchLabProbeRequest,
    ResearchLabTicketResponse,
)
from gateway.research_lab.ticket_lifecycle import (
    UNPAID_TICKET_TTL_SECONDS,
    is_ticket_expiry_conflict,
    ticket_is_house_arm,
    unpaid_ticket_deadline_passed,
    unpaid_ticket_expires_at,
)


ROOT = Path(__file__).resolve().parents[1]
MIGRATION = ROOT / "scripts" / "85-research-lab-unpaid-ticket-expiry.sql"
TICKET_ID = "11111111-1111-4111-8111-111111111111"
HOTKEY = "5F3sa2TJAWMqDhXG6jhV4N8ko9SxwGy8TpaNS1repo5EYjQX"


def _ticket(*, hours_old: float = 25, status: str = "opened", house: bool = False) -> dict:
    created = datetime.now(timezone.utc) - timedelta(hours=hours_old)
    return {
        "ticket_id": TICKET_ID,
        "miner_hotkey": HOTKEY,
        "ticket_status": "opened",
        "current_ticket_status": status,
        "ticket_doc": {"arm": "house"} if house else {},
        "created_at": created.isoformat(),
        "unpaid_expires_at": (created + timedelta(seconds=UNPAID_TICKET_TTL_SECONDS)).isoformat(),
    }


def test_unpaid_deadline_uses_exact_24_hour_boundary() -> None:
    now = datetime(2026, 7, 10, 12, 0, tzinfo=timezone.utc)
    before = {"created_at": (now - timedelta(hours=24) + timedelta(microseconds=1)).isoformat()}
    at = {"created_at": (now - timedelta(hours=24)).isoformat()}
    assert unpaid_ticket_deadline_passed(before, now=now) is False
    assert unpaid_ticket_deadline_passed(at, now=now) is True
    assert unpaid_ticket_expires_at(at) == now


def test_timestamp_parser_accepts_postgrest_nanoseconds_and_house_marker() -> None:
    row = {
        "created_at": "2026-07-09T12:00:00.123456789+00:00",
        "ticket_doc": {"arm": "house"},
    }
    assert unpaid_ticket_expires_at(row) == datetime(2026, 7, 10, 12, 0, 0, 123456, tzinfo=timezone.utc)
    assert ticket_is_house_arm(row) is True


def test_ticket_response_round_trips_with_optional_expiry() -> None:
    response = ResearchLabTicketResponse(
        ticket_id=TICKET_ID,
        status="opened",
        event_id="event-1",
        event_seq=0,
        ticket_hash="sha256:" + "a" * 64,
        unpaid_expires_at="2026-07-10T12:00:00+00:00",
    )
    assert ResearchLabTicketResponse(**response.model_dump(mode="json")) == response


def test_expiry_config_and_admin_command_default_fail_closed(monkeypatch) -> None:
    for name in (
        "RESEARCH_LAB_UNPAID_TICKET_EXPIRY_ENABLED",
        "RESEARCH_LAB_UNPAID_TICKET_EXPIRY_INTERVAL_SECONDS",
        "RESEARCH_LAB_UNPAID_TICKET_EXPIRY_LIMIT",
        "RESEARCH_LAB_UNPAID_TICKET_EXPIRY_WORKER_INDEX",
    ):
        monkeypatch.delenv(name, raising=False)
    config = ResearchLabGatewayConfig.from_env()
    assert config.unpaid_ticket_expiry_enabled is False
    assert config.unpaid_ticket_expiry_interval_seconds == 60
    assert config.unpaid_ticket_expiry_limit == 100
    args = admin.build_parser().parse_args(
        ["expire-unpaid-tickets", "--ticket-id", TICKET_ID]
    )
    assert args.ticket_ids == [TICKET_ID]
    assert args.apply is False


def test_admin_wrapper_uses_live_gateway_layout_and_hydrated_env() -> None:
    wrapper = (ROOT / "scripts" / "install_research_lab_admin_wrapper.sh").read_text(encoding="utf-8")
    assert 'LEADPOET_REPO:-/home/ec2-user}' in wrapper
    assert '/home/ec2-user/.config/leadpoet/gateway.env' in wrapper
    assert '/home/ec2-user/leadpoet_repo' not in wrapper
    assert '/home/ec2-user/gw.environ' not in wrapper


@pytest.mark.asyncio
async def test_admin_expiry_command_dispatches_dry_run_by_default(monkeypatch) -> None:
    captured: dict = {}

    async def fake_expire(**kwargs):
        captured.update(kwargs)
        return {"ok": True, "dry_run": kwargs["dry_run"]}

    monkeypatch.setattr(admin, "expire_unpaid_tickets", fake_expire)
    args = admin.build_parser().parse_args(
        ["expire-unpaid-tickets", "--ticket-id", TICKET_ID, "--limit", "5"]
    )
    result = await admin._run(args)
    assert result == {"ok": True, "dry_run": True}
    assert captured["ticket_ids"] == [TICKET_ID]
    assert captured["limit"] == 5


@pytest.mark.asyncio
async def test_get_ticket_for_miner_reads_current_projection(monkeypatch) -> None:
    calls: list[tuple[str, dict]] = []

    async def fake_select_one(table: str, **kwargs):
        calls.append((table, kwargs))
        return _ticket(status="expired")

    monkeypatch.setattr(api, "select_one", fake_select_one)
    row = await api._get_ticket_for_miner(TICKET_ID, HOTKEY)
    assert row["current_ticket_status"] == "expired"
    assert calls[0][0] == "research_loop_ticket_current"


@pytest.mark.asyncio
async def test_expired_ticket_returns_410_without_database_precheck(monkeypatch) -> None:
    async def unexpected_select(*args, **kwargs):
        raise AssertionError("explicit expiry should not query the candidate view")

    monkeypatch.setattr(api, "select_one", unexpected_select)
    with pytest.raises(HTTPException) as caught:
        await api._require_ticket_mutable(_ticket(status="expired"), enforce_unpaid_deadline=True)
    assert caught.value.status_code == 410
    assert caught.value.detail["code"] == "research_lab_ticket_expired"


@pytest.mark.asyncio
@pytest.mark.parametrize("route_name", ["probe", "loop_start", "top_up"])
async def test_expired_mutation_routes_return_structured_410(monkeypatch, route_name: str) -> None:
    config = ResearchLabGatewayConfig(
        api_enabled=True,
        production_writes_enabled=True,
        miner_submissions_enabled=True,
        paid_loops_enabled=True,
        loop_topups_enabled=True,
        probes_enabled=True,
        hosted_runs_enabled=True,
        miner_openrouter_key_required=False,
    )
    monkeypatch.setattr(api.ResearchLabGatewayConfig, "from_env", classmethod(lambda cls: config))

    async def noop(*args, **kwargs):
        return None

    async def expired_ticket(*args, **kwargs):
        return _ticket(status="expired")

    monkeypatch.setattr(api, "_verify_signed_miner", noop)
    monkeypatch.setattr(api, "_require_autoresearch_not_paused", noop)
    monkeypatch.setattr(api, "_enforce_research_lab_submission_rate_limit", noop)
    monkeypatch.setattr(api, "_get_ticket_for_miner", expired_ticket)

    common = {
        "miner_hotkey": HOTKEY,
        "signature": "s" * 16,
        "timestamp": int(datetime.now(timezone.utc).timestamp()),
        "idempotency_key": "expiry-route-test",
        "ticket_id": TICKET_ID,
    }
    if route_name == "probe":
        call = api.create_research_lab_probe(
            ResearchLabProbeRequest(**common, probe_ref="probe:expiry-test")
        )
    elif route_name == "loop_start":
        call = api.start_research_lab_paid_loop(
            ResearchLabLoopStartRequest(
                **common,
                payment_block_hash="0x12345678",
                payment_extrinsic_index=1,
                miner_openrouter_key_ref="key-ref-expiry",
                miner_openrouter_key_handling="encrypted_ref",
                miner_openrouter_preflight_status="passed",
            )
        )
    else:
        call = api.top_up_research_lab_paid_loop(
            ResearchLabLoopTopUpRequest(
                **common,
                continue_from_run_id="22222222-2222-4222-8222-222222222222",
                payment_block_hash="0x12345678",
                payment_extrinsic_index=1,
                additional_compute_budget_usd=5.0,
                miner_openrouter_key_ref="key-ref-expiry",
                miner_openrouter_key_handling="encrypted_ref",
                miner_openrouter_preflight_status="passed",
            )
        )
    with pytest.raises(HTTPException) as caught:
        await call
    assert caught.value.status_code == 410
    assert caught.value.detail["code"] == "research_lab_ticket_expired"


@pytest.mark.asyncio
async def test_deadline_precheck_only_rejects_database_confirmed_unpaid_ticket(monkeypatch) -> None:
    async def eligible(*args, **kwargs):
        return {"ticket_id": TICKET_ID}

    monkeypatch.setattr(api, "select_one", eligible)
    with pytest.raises(HTTPException) as caught:
        await api._require_ticket_mutable(_ticket(), enforce_unpaid_deadline=True)
    assert caught.value.status_code == 410

    async def protected_by_evidence(*args, **kwargs):
        return None

    monkeypatch.setattr(api, "select_one", protected_by_evidence)
    await api._require_ticket_mutable(_ticket(), enforce_unpaid_deadline=True)
    await api._require_ticket_mutable(_ticket(), enforce_unpaid_deadline=False)


def test_storage_expiry_conflict_maps_to_410() -> None:
    error = RuntimeError("research_lab_ticket_expired: ticket passed its unpaid deadline")
    assert is_ticket_expiry_conflict(error) is True
    with pytest.raises(HTTPException) as caught:
        api._raise_storage_error(error)
    assert caught.value.status_code == 410
    assert caught.value.detail["code"] == "research_lab_ticket_expired"


@pytest.mark.asyncio
async def test_expired_tickets_do_not_count_toward_open_cap(monkeypatch) -> None:
    async def fake_select_all(table: str, **kwargs):
        if table == "research_loop_ticket_current":
            return [
                {"ticket_id": f"expired-{index}", "current_ticket_status": "expired"}
                for index in range(3)
            ] + [{"ticket_id": "open-1", "current_ticket_status": "opened"}]
        assert table == "research_lab_public_loop_card_current"
        return []

    monkeypatch.setattr(api, "select_all", fake_select_all)
    await api._enforce_open_ticket_cap(ResearchLabGatewayConfig(max_open_tickets_per_hotkey=3), HOTKEY)


def test_public_projection_surfaces_expired_as_neutral_terminal_state() -> None:
    outcome = public_activity.derive_public_loop_outcome(
        ticket=_ticket(status="expired"),
        queue_rows=(),
        receipt_rows=(),
        candidate_rows=(),
        score_bundle_rows=(),
        promotion_event_rows=(),
    )
    assert (outcome.event_type, outcome.outcome_label, outcome.outcome_band) == (
        "expired",
        "expired",
        "expired",
    )
    assert outcome.event_doc["payment_state"] == "expired"
    assert public_activity.public_loop_outcome_closes_ticket(
        {"current_outcome_label": "expired", "current_outcome_band": "expired"}
    ) is True


@pytest.mark.asyncio
async def test_find_expirable_tickets_is_paginated_and_bounded(monkeypatch) -> None:
    captured: dict = {}

    async def fake_select_all(table: str, **kwargs):
        captured.update({"table": table, **kwargs})
        return [{"ticket_id": str(index)} for index in range(1205)]

    monkeypatch.setattr(maintenance, "select_all", fake_select_all)
    rows = await maintenance.find_expirable_unpaid_tickets(
        ticket_ids=[TICKET_ID, TICKET_ID],
        limit=1205,
    )
    assert len(rows) == 1205
    assert captured["table"] == maintenance.UNPAID_TICKET_EXPIRY_CANDIDATE_VIEW
    assert captured["batch_size"] == 1000
    assert captured["max_rows"] == 1205
    assert captured["allow_partial"] is True
    assert captured["filters"] == (("ticket_id", "in", (TICKET_ID,)),)


@pytest.mark.asyncio
async def test_expiry_dry_run_is_read_only(monkeypatch) -> None:
    plan = _ticket()

    async def fake_find(**kwargs):
        return [plan]

    async def unexpected_write(**kwargs):
        raise AssertionError("dry run must not write")

    monkeypatch.setattr(maintenance, "find_expirable_unpaid_tickets", fake_find)
    monkeypatch.setattr(maintenance, "create_ticket_event", unexpected_write)
    result = await maintenance.expire_unpaid_tickets(ticket_ids=[TICKET_ID], dry_run=True)
    assert result["ok"] is True
    assert result["planned_count"] == 1


@pytest.mark.asyncio
async def test_expiry_revalidates_appends_once_and_projects(monkeypatch) -> None:
    plan = {
        **_ticket(),
        "current_event_seq": 0,
        "current_event_hash": "sha256:" + "b" * 64,
    }
    events: list[dict] = []
    projections: list[str] = []

    async def fake_find(**kwargs):
        return [plan]

    async def fake_select_one(table: str, **kwargs):
        assert table == maintenance.UNPAID_TICKET_EXPIRY_CANDIDATE_VIEW
        return plan

    async def fake_create_ticket_event(**kwargs):
        events.append(kwargs)
        return {"seq": 1, "anchored_hash": "sha256:" + "c" * 64}

    async def fake_project(ticket_id: str, **kwargs):
        projections.append(ticket_id)
        return {"event": {"event_type": "expired"}}

    monkeypatch.setattr(maintenance, "find_expirable_unpaid_tickets", fake_find)
    monkeypatch.setattr(maintenance, "select_one", fake_select_one)
    monkeypatch.setattr(maintenance, "create_ticket_event", fake_create_ticket_event)
    monkeypatch.setattr(maintenance, "safe_project_public_loop_activity", fake_project)

    result = await maintenance.expire_unpaid_tickets(dry_run=False, actor_ref="test-actor")
    assert result["ok"] is True
    assert result["expired_count"] == 1
    assert events[0]["event_type"] == "expired"
    assert events[0]["event_doc"]["ttl_seconds"] == 86400
    assert events[0]["event_doc"]["actor_ref"] == "test-actor"
    assert projections == [TICKET_ID]


@pytest.mark.asyncio
async def test_expiry_scan_and_projection_failures_are_non_blocking(monkeypatch) -> None:
    async def failed_find(**kwargs):
        raise RuntimeError("temporary database outage")

    monkeypatch.setattr(maintenance, "find_expirable_unpaid_tickets", failed_find)
    failed = await maintenance.expire_unpaid_tickets(dry_run=False)
    assert failed["ok"] is False
    assert failed["error"] == "expiry_candidate_scan_unavailable"

    plan = {**_ticket(), "current_event_hash": "sha256:" + "d" * 64}

    async def good_find(**kwargs):
        return [plan]

    async def current(*args, **kwargs):
        return plan

    async def event(**kwargs):
        return {"seq": 1, "anchored_hash": "sha256:" + "e" * 64}

    async def projection_pending(*args, **kwargs):
        return None

    monkeypatch.setattr(maintenance, "find_expirable_unpaid_tickets", good_find)
    monkeypatch.setattr(maintenance, "select_one", current)
    monkeypatch.setattr(maintenance, "create_ticket_event", event)
    monkeypatch.setattr(maintenance, "safe_project_public_loop_activity", projection_pending)
    result = await maintenance.expire_unpaid_tickets(dry_run=False)
    assert result["expired_count"] == 1
    assert result["projection_pending_count"] == 1


@pytest.mark.asyncio
async def test_worker_expiry_sweep_is_sharded_and_interval_limited(monkeypatch) -> None:
    calls: list[dict] = []

    async def fake_expire(**kwargs):
        calls.append(kwargs)
        return {"ok": True, "planned_count": 1, "expired_count": 1, "skipped_count": 0}

    monkeypatch.setattr(worker_mod, "expire_unpaid_tickets", fake_expire)
    config = ResearchLabGatewayConfig(
        unpaid_ticket_expiry_enabled=True,
        unpaid_ticket_expiry_worker_index=0,
        unpaid_ticket_expiry_interval_seconds=60,
        unpaid_ticket_expiry_limit=7,
        hosted_worker_index=0,
        hosted_worker_total_workers=2,
    )
    worker = worker_mod.ResearchLabHostedWorker(config, worker_ref="expiry-worker")
    await worker._maybe_expire_unpaid_tickets()
    await worker._maybe_expire_unpaid_tickets()
    assert len(calls) == 1
    assert calls[0]["limit"] == 7
    assert calls[0]["dry_run"] is False

    other = worker_mod.ResearchLabHostedWorker(
        ResearchLabGatewayConfig(
            unpaid_ticket_expiry_enabled=True,
            unpaid_ticket_expiry_worker_index=0,
            hosted_worker_index=1,
            hosted_worker_total_workers=2,
        )
    )
    await other._maybe_expire_unpaid_tickets()
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_operator_pause_prevents_expiry_sweep(monkeypatch) -> None:
    config = ResearchLabGatewayConfig(
        unpaid_ticket_expiry_enabled=True,
        hosted_worker_dry_run=False,
    )
    worker = worker_mod.ResearchLabHostedWorker(config)
    worker._require_enabled = lambda: None

    async def noop():
        return None

    called = 0

    async def expiry():
        nonlocal called
        called += 1

    worker._recover_stale_started_runs = noop
    worker._reconcile_stale_loop_projections = noop
    worker._maybe_reconcile_terminal_tickets = noop
    worker._maybe_refresh_allocator_priors = noop
    worker._maybe_expire_unpaid_tickets = expiry

    async def paused_state():
        return {"paused": True, "reason": "gateway_restart"}

    monkeypatch.setattr(worker_mod, "get_autoresearch_maintenance_state", paused_state)
    outcome = await worker.run_once()
    assert outcome.status == "maintenance_paused"
    assert called == 0


def test_sql_contract_has_atomic_lock_guards_and_all_disqualifying_evidence() -> None:
    sql = MIGRATION.read_text(encoding="utf-8")
    assert "'expired'" in sql
    assert "INTERVAL '24 hours'" in sql
    assert sql.count("pg_catalog.clock_timestamp()") == 2
    assert sql.count("pg_catalog.hashtext('research_lab_ticket_lifecycle')") >= 3
    assert "guard_research_lab_ticket_lifecycle_insert" in sql
    assert "guard_research_lab_loop_start_payment_expiry_insert" in sql
    assert "guard_research_loop_start_credit_consume_insert" in sql
    for index_name in (
        "idx_research_loop_start_payments_ticket",
        "idx_research_loop_start_credits_ticket",
        "idx_research_loop_start_credit_events_ticket",
        "idx_research_loop_balance_ledger_ticket",
    ):
        assert index_name in sql
    for table in (
        "research_loop_start_payments",
        "research_loop_start_credits",
        "research_loop_start_credit_events",
        "research_loop_balance_ledger",
        "research_loop_run_queue_events",
        "research_loop_receipts",
        "research_lab_auto_research_loop_events",
        "research_lab_candidate_artifacts",
    ):
        assert table in sql
    assert "ticket_doc->>'arm', '') = 'house'" in sql
    assert "payment_kind = 'loop_start'" in sql
    protected_alters = (
        "research_lab_candidate_evaluation_events",
        "research_lab_private_model_benchmark",
        "research_reimbursement_awards",
        "research_lab_epoch_allocations",
    )
    for table in protected_alters:
        assert f"ALTER TABLE public.{table}" not in sql
