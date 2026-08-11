"""Later-run credential and credit-resume continuity regressions."""

from __future__ import annotations

import asyncio
from pathlib import Path
import time
from typing import Any

import pytest

from gateway.research_lab import api, maintenance, recovery, store, worker
from gateway.research_lab.models import ResearchLabResumeCreditBlockedRequest


KEY_A = "encrypted_ref:openrouter:" + "a" * 32
KEY_B = "encrypted_ref:openrouter:" + "b" * 32


def test_resume_preflight_sequence_binds_complete_signed_attempt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[int] = []

    async def fake_verify(_payload: Any) -> None:
        return None

    async def fake_not_paused(_config: Any) -> None:
        return None

    async def fake_resume(
        _miner_hotkey: str,
        *,
        run_ids: list[str] | None,
        preflight_sequence: int,
    ) -> dict[str, Any]:
        assert run_ids == ["run-1"]
        observed.append(preflight_sequence)
        return {"requeued": 0, "still_blocked": 0, "results": []}

    monkeypatch.setattr(api, "_verify_signed_miner", fake_verify)
    monkeypatch.setattr(api, "_require_autoresearch_not_paused", fake_not_paused)
    monkeypatch.setattr(
        recovery,
        "resume_credit_blocked_runs_for_miner",
        fake_resume,
    )
    monkeypatch.setattr(
        api.ResearchLabGatewayConfig,
        "from_env",
        classmethod(
            lambda _cls: type(
                "Config",
                (),
                {
                    "api_enabled": True,
                    "production_writes_enabled": True,
                    "miner_submissions_enabled": True,
                },
            )()
        ),
    )

    def payload(*, timestamp: int, idempotency_key: str) -> Any:
        return ResearchLabResumeCreditBlockedRequest(
            miner_hotkey="miner-0000000001",
            signature="0" * 128,
            timestamp=timestamp,
            idempotency_key=idempotency_key,
            run_ids=["run-1"],
        )

    now = int(time.time())
    first = payload(timestamp=now, idempotency_key="resume-a")
    replay = payload(timestamp=now, idempotency_key="resume-a")
    same_second_new_attempt = payload(
        timestamp=now,
        idempotency_key="resume-b",
    )
    asyncio.run(api.resume_research_lab_credit_blocked(first))
    asyncio.run(api.resume_research_lab_credit_blocked(replay))
    asyncio.run(api.resume_research_lab_credit_blocked(same_second_new_attempt))

    assert observed[0] == observed[1]
    assert observed[0] != observed[2]
    assert all(0 <= value < 2**60 for value in observed)


def test_reimbursement_context_prefers_latest_run_event_key() -> None:
    ticket = {"miner_openrouter_key_ref": KEY_A}
    events = [
        {
            "seq": 3,
            "event_doc": {"miner_openrouter_key_ref": KEY_B},
        }
    ]

    assert recovery._openrouter_key_ref_from_ticket_or_events(ticket, events) == KEY_B


@pytest.mark.asyncio
async def test_attested_credit_preflight_uses_later_run_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    async def fake_select_one(table: str, **_kwargs: Any) -> dict[str, Any] | None:
        if table == "research_loop_ticket_current":
            return {
                "ticket_id": "ticket-1",
                "miner_hotkey": "miner-1",
                "miner_openrouter_key_ref": KEY_A,
                "miner_openrouter_key_handling": "encrypted_ref_only",
            }
        if table == "research_lab_openrouter_key_refs":
            return {"key_ref": KEY_B, "miner_hotkey": "miner-1"}
        raise AssertionError(table)

    async def fake_select_all(table: str, **_kwargs: Any) -> list[dict[str, Any]]:
        assert table == "research_loop_run_queue_events"
        return [
            {
                "run_id": "run-2",
                "ticket_id": "ticket-1",
                "event_type": "queued",
                "seq": 4,
                "event_doc": {"miner_openrouter_key_ref": KEY_B},
            }
        ]

    async def fake_preflight(**kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return {
            "result": {
                "preflight_doc": {
                    "key_hash": "sha256:" + "c" * 64,
                    "limit": 1.0,
                    "limit_remaining": 0.5,
                    "limit_reset": None,
                }
            }
        }

    async def fake_epoch(_configured: int) -> tuple[int, int, str]:
        return 7, 100, "test"

    monkeypatch.setattr(maintenance, "select_one", fake_select_one)
    monkeypatch.setattr(maintenance, "select_all", fake_select_all)
    monkeypatch.setattr(maintenance, "legacy_v1_enabled", lambda: False)
    monkeypatch.setattr(
        "gateway.research_lab.attested_coordinator_v2.preflight_openrouter_key_ref_v2",
        fake_preflight,
    )
    monkeypatch.setattr(
        "gateway.research_lab.chain.resolve_research_lab_evaluation_epoch",
        fake_epoch,
    )

    result = await maintenance._preflight_openrouter_key_for_run(
        "ticket-1",
        run_id="run-2",
        preflight_sequence=1234,
    )

    assert result["ok"] is True
    assert captured["key_ref"] == KEY_B
    assert captured["miner_hotkey"] == "miner-1"
    assert captured["sequence"] == 1234
    assert result["preflight_sequence"] == 1234


@pytest.mark.asyncio
async def test_run_credential_resolution_is_complete_and_run_scoped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, Any] = {}
    ticket = {
        "ticket_id": "ticket-1",
        "miner_hotkey": "miner-1",
        "miner_openrouter_key_ref": KEY_A,
    }

    async def fake_select_all(table: str, **kwargs: Any) -> list[dict[str, Any]]:
        assert table == "research_loop_run_queue_events"
        observed.update(kwargs)
        return [
            {
                "run_id": "run-b",
                "ticket_id": "ticket-1",
                "event_type": "queued",
                "seq": 0,
                "event_doc": {"miner_openrouter_key_ref": KEY_B},
            },
            {
                "run_id": "run-b",
                "ticket_id": "ticket-1",
                "event_type": "queued",
                "seq": 44,
                "event_doc": {"miner_openrouter_key_ref": KEY_B},
            },
        ]

    async def fake_select_one(table: str, **_kwargs: Any) -> dict[str, Any]:
        assert table == "research_lab_openrouter_key_refs"
        return {"key_ref": KEY_B, "miner_hotkey": "miner-1"}

    monkeypatch.setattr(maintenance, "select_all", fake_select_all)
    monkeypatch.setattr(maintenance, "select_one", fake_select_one)
    monkeypatch.setattr(maintenance, "legacy_v1_enabled", lambda: False)

    resolved = await maintenance.resolve_openrouter_key_ref_for_run(
        ticket,
        run_id="run-b",
    )

    assert resolved == KEY_B
    assert observed["filters"] == (("run_id", "run-b"), ("event_type", "queued"))
    assert observed["max_rows"] == 10000


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("events", "key_row", "message"),
    [
        (
            [
                {
                    "run_id": "run-b",
                    "ticket_id": "ticket-1",
                    "event_type": "queued",
                    "event_doc": {"miner_openrouter_key_ref": KEY_A},
                },
                {
                    "run_id": "run-b",
                    "ticket_id": "ticket-1",
                    "event_type": "queued",
                    "event_doc": {"miner_openrouter_key_ref": KEY_B},
                },
            ],
            {"key_ref": KEY_B, "miner_hotkey": "miner-1"},
            "conflicting credential refs",
        ),
        (
            [
                {
                    "run_id": "run-b",
                    "ticket_id": "ticket-other",
                    "event_type": "queued",
                    "event_doc": {"miner_openrouter_key_ref": KEY_B},
                }
            ],
            {"key_ref": KEY_B, "miner_hotkey": "miner-1"},
            "credential ticket differs",
        ),
        (
            [
                {
                    "run_id": "run-b",
                    "ticket_id": "ticket-1",
                    "event_type": "queued",
                    "event_doc": {"miner_openrouter_key_ref": KEY_B},
                }
            ],
            {"key_ref": KEY_B, "miner_hotkey": "miner-other"},
            "does not belong to ticket miner",
        ),
    ],
)
async def test_run_credential_resolution_fails_closed_on_identity_drift(
    monkeypatch: pytest.MonkeyPatch,
    events: list[dict[str, Any]],
    key_row: dict[str, Any],
    message: str,
) -> None:
    async def fake_select_all(*_args: Any, **_kwargs: Any) -> list[dict[str, Any]]:
        return events

    async def fake_select_one(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        return key_row

    monkeypatch.setattr(maintenance, "select_all", fake_select_all)
    monkeypatch.setattr(maintenance, "select_one", fake_select_one)
    monkeypatch.setattr(maintenance, "legacy_v1_enabled", lambda: False)

    with pytest.raises(RuntimeError, match=message):
        await maintenance.resolve_openrouter_key_ref_for_run(
            {
                "ticket_id": "ticket-1",
                "miner_hotkey": "miner-1",
                "miner_openrouter_key_ref": KEY_A,
            },
            run_id="run-b",
        )


def test_worker_uses_only_prevalidated_run_credential() -> None:
    context = worker.HostedRunContext(
        queue_row={"run_id": "run-b", "ticket_id": "ticket-1"},
        ticket={"miner_hotkey": "miner-1", "miner_openrouter_key_ref": KEY_A},
        payment=None,
        ticket_events=(
            {"event_doc": {"miner_openrouter_key_ref": "newer-run-key"}},
        ),
        openrouter_key_ref=KEY_B,
    )

    assert worker._miner_openrouter_key_ref(context) == KEY_B


@pytest.mark.asyncio
async def test_credit_resume_event_is_deterministic_and_rpc_atomic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []

    async def fake_rpc(_name: str, params: dict[str, Any]) -> list[dict[str, Any]]:
        calls.append(dict(params))
        event_doc = dict(params["p_event_doc"])
        payload = {
            "run_id": params["p_run_id"],
            "ticket_id": params["p_ticket_id"],
            "seq": params["p_expected_event_seq"] + 1,
            "event_type": "queued",
            "queue_priority": params["p_queue_priority"],
            "worker_ref": params["p_worker_ref"],
            "reason": params["p_reason"],
            "event_doc": event_doc,
        }
        return [
            {
                "event_id": params["p_event_id"],
                **payload,
                "anchored_hash": store.canonical_hash(payload),
            }
        ]

    monkeypatch.setattr(store, "call_rpc", fake_rpc)
    kwargs = {
        "run_id": "10000000-0000-0000-0000-000000000001",
        "ticket_id": "20000000-0000-0000-0000-000000000002",
        "expected_event_seq": 4,
        "expected_event_hash": "sha256:" + "d" * 64,
        "queue_priority": 7,
        "worker_ref": "miner:test",
        "reason": "credit_topup_resume",
        "event_doc": {
            "schema_version": "1.0",
            "resume_source": "miner_credit_topup_resume",
        },
    }

    first, second = await asyncio.gather(
        store.resume_credit_blocked_queue_event(**kwargs),
        store.resume_credit_blocked_queue_event(**kwargs),
    )

    assert first["event_id"] == second["event_id"]
    assert first["anchored_hash"] == second["anchored_hash"]
    assert calls[0]["p_event_id"] == calls[1]["p_event_id"]
    assert calls[0]["p_anchored_hash"] == calls[1]["p_anchored_hash"]


def test_atomic_credit_resume_migration_is_expected_head_and_idempotent() -> None:
    sql = (
        Path(__file__).parents[1]
        / "scripts"
        / "148-research-lab-atomic-credit-resume.sql"
    ).read_text(encoding="utf-8")

    assert "resume_research_lab_credit_blocked_run_v1" in sql
    assert "pg_advisory_xact_lock" in sql
    assert "head.anchored_hash IS DISTINCT FROM p_expected_event_hash" in sql
    assert "head.event_type IS DISTINCT FROM 'paused'" in sql
    assert "head.reason IS DISTINCT FROM 'blocked_for_credit'" in sql
    assert "WHERE e.event_id = p_event_id" in sql
    assert "NOTIFY pgrst, 'reload schema'" in sql

    from gateway.tee.supabase_schema_preflight_v2 import (
        REQUIRED_SUPABASE_V2_RPCS,
    )
    from tests.restart_rehearsal.postgres_v2_contract_probe import (
        ATOMIC_CREDIT_RESUME_MIGRATION,
        EXPECTED_APPLIED_MIGRATIONS,
    )

    assert (
        "scripts/148-research-lab-atomic-credit-resume.sql",
        "resume_research_lab_credit_blocked_run_v1",
    ) in REQUIRED_SUPABASE_V2_RPCS
    assert ATOMIC_CREDIT_RESUME_MIGRATION == EXPECTED_APPLIED_MIGRATIONS[-1]


@pytest.mark.asyncio
async def test_admin_credit_resume_uses_exact_nonzero_head_and_canonical_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, Any] = {}

    async def fake_select_one(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        return {
            "run_id": "run-b",
            "ticket_id": "ticket-1",
            "current_queue_status": "paused",
            "current_reason": "blocked_for_credit",
            "current_event_seq": 37,
            "current_event_hash": "sha256:" + "d" * 64,
            "queue_priority": 2,
            "current_status_at": "2026-08-09T00:00:00Z",
        }

    async def fake_preflight(*_args: Any, **kwargs: Any) -> dict[str, Any]:
        observed["preflight"] = kwargs
        return {"ok": True, "limit_remaining": 2, "preflight_sequence": 7788}

    async def fake_resume(**kwargs: Any) -> dict[str, Any]:
        observed["resume"] = kwargs
        return {"seq": 38, "anchored_hash": "sha256:" + "e" * 64}

    monkeypatch.setattr(maintenance, "select_one", fake_select_one)
    monkeypatch.setattr(maintenance, "_preflight_openrouter_key_for_run", fake_preflight)
    monkeypatch.setattr(maintenance, "resume_credit_blocked_queue_event", fake_resume)

    result = await maintenance.resume_credit_blocked_run(
        run_id="run-b",
        dry_run=False,
        actor_ref="operator:test",
        preflight_sequence=7788,
    )

    assert result["event_seq"] == 38
    assert observed["preflight"]["preflight_sequence"] == 7788
    assert observed["resume"]["expected_event_seq"] == 37
    assert observed["resume"]["reason"] == "credit_topup_resume"
    assert (
        observed["resume"]["event_doc"]["resume_source"]
        == "miner_credit_topup_resume"
    )


@pytest.mark.asyncio
async def test_miner_resume_preflights_and_requeues_the_exact_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_id = "10000000-0000-0000-0000-000000000001"
    ticket_id = "20000000-0000-0000-0000-000000000002"
    expected_hash = "sha256:" + "e" * 64
    observed: dict[str, Any] = {}

    async def fake_select_all(*_args: Any, **_kwargs: Any) -> list[dict[str, Any]]:
        return [
            {
                "run_id": run_id,
                "ticket_id": ticket_id,
                "current_queue_status": "paused",
                "current_reason": "blocked_for_credit",
                "current_event_seq": 4,
                "current_event_hash": expected_hash,
                "queue_priority": 3,
            }
        ]

    async def fake_select_one(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        return {"ticket_id": ticket_id, "miner_hotkey": "miner-1"}

    async def fake_ready(_config: Any, **kwargs: Any) -> tuple[bool, str]:
        observed["preflight"] = kwargs
        return True, "limit_remaining=1"

    async def fake_resume(**kwargs: Any) -> dict[str, Any]:
        observed["resume"] = kwargs
        return {"event_id": "30000000-0000-0000-0000-000000000003"}

    async def fake_projection(*_args: Any, **_kwargs: Any) -> None:
        return None

    monkeypatch.setattr(recovery, "select_all", fake_select_all)
    monkeypatch.setattr(recovery, "select_one", fake_select_one)
    monkeypatch.setattr(recovery, "_openrouter_credit_ready", fake_ready)
    monkeypatch.setattr(recovery, "resume_credit_blocked_queue_event", fake_resume)
    monkeypatch.setattr(recovery, "_project_after_recovery", fake_projection)

    result = await recovery.resume_credit_blocked_runs_for_miner(
        "miner-1",
        run_ids=[run_id],
        preflight_sequence=9001,
    )

    assert result["requeued"] == 1
    assert observed["preflight"] == {
        "ticket_id": ticket_id,
        "run_id": run_id,
        "preflight_sequence": 9001,
    }
    assert observed["resume"]["expected_event_seq"] == 4
    assert observed["resume"]["expected_event_hash"] == expected_hash
