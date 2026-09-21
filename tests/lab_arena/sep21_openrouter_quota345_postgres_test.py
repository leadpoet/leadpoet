"""Real-PostgreSQL proof for the bounded Sep21 quota repair."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from lab_arena import contracts
from lab_arena.store import ArenaStore, PsycopgTransport
from tests.lab_arena.lab_arena_pg_harness import (
    CURRENT_SERVICE_MIGRATIONS,
    database_with_lab_arena_migration,
)
from tests.lab_arena.test_lab_arena_migration_postgres import hotkey, sha


ROOT = Path(__file__).parents[2]
MIGRATION = ROOT / "scripts/345-arena-2026-09-21-openrouter-quota-2000.sql"
ROUND = "arena-2026-09-21"
HISTORY = "arena-2026-09-20"
FUTURE = "arena-2026-09-22"
SUBMISSION = "baseline-2026-09-21"
RUN = "sep21-active-leased-run"
LEASE_TOKEN_HASH = "sha256:" + "a" * 64
BASELINE = hotkey("sep21-quota-baseline")

PRODUCTION_ADMISSION_MIGRATIONS = (
    "264-lab-arena-codex-cost-reconciliation.sql",
    "289-lab-arena-per-icp-cost-policy.sql",
    "311-lab-arena-per-icp-closed-billing-reconciliation.sql",
    "312-lab-arena-temporary-hold-admission.sql",
    "314-lab-arena-openrouter-web-search-reservation.sql",
    "319-lab-arena-quota-sourcing-cost.sql",
    "321-lab-arena-confirmed-cost-admission.sql",
)


@pytest.fixture(scope="module")
def database():
    yield from database_with_lab_arena_migration(
        CURRENT_SERVICE_MIGRATIONS + PRODUCTION_ADMISSION_MIGRATIONS
    )


def _configuration(round_id: str, openrouter_quota: int) -> dict:
    date = round_id.removeprefix("arena-")
    return {
        "schema_version": contracts.ROUND_CONFIGURATION_SCHEMA_VERSION,
        "round_id": round_id,
        "mode": "live",
        "network_name": "finney",
        "netuid": 71,
        "rewards_enabled": True,
        "schedule": {"submission_cutoff": f"{date}T00:00:00Z"},
        "stage_1_icp_count": 10,
        "stage_2_icp_count": 10,
        "max_attempts_per_assignment": 2,
        "call_quotas": {
            "scrapingdog": 200,
            "deepline": 200,
            "openrouter": openrouter_quota,
        },
        "scoring_call_quotas": {
            "scrapingdog": 150,
            "deepline": 40,
            "openrouter": 120,
        },
        "icp_wall_clock_seconds": 2700,
        "scoring_wall_clock_seconds": 900,
        "execution_cap_microusd": 80_000_000,
        "execution_icp_cap_microusd": 4_000_000,
        "cost_per_company_microusd": 800_000,
        "sourcing_cost_eligibility_policy": "successful_calls_per_icp_v1",
        "integrity_policy": "arena_integrity_v1",
        "scorer_policy": {
            "scoring_adapter_version": "qualification_integrity_v2"
        },
        "checkpoint_deadline_policy": "atomic_checkpoint_45m_v1",
        "parallel_twenty_icp_execution": True,
        "scoring_cap_microusd": 50_000_000,
        "baseline_hotkey": BASELINE,
        "baseline_source_url": (
            "https://github.com/leadpoet/champion_model/archive/refs/heads/lab.tar.gz"
        ),
        "reward_constants": {"pool_percent": 30, "preserve": True},
    }


def _seed(connection) -> None:
    participants = [
        {"submission_id": SUBMISSION, "miner_hotkey": BASELINE, "is_king": True}
    ]
    with connection.cursor() as cursor:
        cursor.execute("SET session_replication_role=replica")
        cursor.execute(
            "TRUNCATE public.lab_arena_ledger,public.lab_arena_runs,"
            "public.lab_arena_submissions,public.lab_arena_rounds "
            "RESTART IDENTITY CASCADE"
        )
        cursor.execute(
            "INSERT INTO public.lab_arena_rounds("
            "round_id,status,status_generation,stage_generation,configuration_doc,"
            "rewards_enabled,participants,benchmark_ref,evaluation_date,icp_set_date,"
            "confirmation_bank_ref,confirmation_bank_hash,confirmation_cohort) VALUES "
            "(%s,'stage1',9,7,%s::jsonb,TRUE,%s::jsonb,%s,'2026-09-21',"
            "'2026-09-20',%s,%s,%s::jsonb),"
            "(%s,'published',20,20,%s::jsonb,TRUE,'[]'::jsonb,"
            "NULL,NULL,NULL,NULL,NULL,NULL),"
            "(%s,'open',0,0,%s::jsonb,TRUE,'[]'::jsonb,"
            "NULL,NULL,NULL,NULL,NULL,NULL)",
            (
                ROUND,
                json.dumps(_configuration(ROUND, 200)),
                json.dumps(participants),
                f"arena/{ROUND}/benchmark.json",
                f"arena/{ROUND}/confirmation-bank.json",
                "sha256:" + "b" * 64,
                json.dumps({"source_round_id": HISTORY, "preserve": True}),
                HISTORY,
                json.dumps(_configuration(HISTORY, 200)),
                FUTURE,
                json.dumps(_configuration(FUTURE, 2000)),
            ),
        )
        source_ref = f"arena/{ROUND}/sources/{SUBMISSION}.tar.gz"
        cursor.execute(
            "INSERT INTO public.lab_arena_submissions("
            "submission_id,round_id,miner_hotkey,status,is_king,source_ref,"
            "source_size_bytes,submission_doc,frozen_at) VALUES "
            "(%s,%s,%s,'frozen',TRUE,%s,12345,%s::jsonb,"
            "'2026-09-21T00:00:30Z')",
            (
                SUBMISSION,
                ROUND,
                BASELINE,
                source_ref,
                json.dumps(
                    {
                        "source_ref": source_ref,
                        "source_sha256": "c" * 64,
                        "source_commit": "d" * 40,
                        "consent": {"public_rerun": True},
                    }
                ),
            ),
        )
        cursor.execute(
            "INSERT INTO public.lab_arena_runs("
            "run_id,assignment_id,round_id,submission_id,miner_hotkey,stage,"
            "icp_position,attempt,kind,status,stage_generation,terminal_cause,"
            "terminal_doc) VALUES "
            "('sep21-preserved-failed-run','sep21-failed-assignment',%s,%s,%s,"
            "1,1,1,'execute','failed',7,'budget_exhausted',%s::jsonb)",
            (
                ROUND,
                SUBMISSION,
                BASELINE,
                json.dumps({"reason": "provider_budget", "preserve": True}),
            ),
        )
        cursor.execute(
            "INSERT INTO public.lab_arena_runs("
            "run_id,assignment_id,round_id,submission_id,miner_hotkey,stage,"
            "icp_position,attempt,kind,status,runner_hotkey,lease_token_hash,"
            "lease_generation,stage_generation,lease_expires_at) VALUES "
            "(%s,'sep21-active-assignment',%s,%s,%s,1,2,1,'execute','leased',"
            "%s,%s,3,7,clock_timestamp()+interval '1 hour')",
            (RUN, ROUND, SUBMISSION, BASELINE, BASELINE, LEASE_TOKEN_HASH),
        )

        # The active ICP has exhausted the old OpenRouter count while all
        # reservation and dispatch holds are zero. Confirmed cost is $3.90.
        cursor.execute(
            "WITH calls AS ("
            " SELECT provider,ordinal,'sha256:'||encode(extensions.digest("
            "   provider||':'||ordinal::text,'sha256'),'hex') AS identity"
            " FROM (SELECT 'openrouter'::text provider,generate_series(1,200) ordinal"
            "       UNION ALL SELECT 'deepline',generate_series(1,25)) source"
            ") INSERT INTO public.lab_arena_ledger("
            "entry_kind,miner_hotkey,round_id,submission_id,run_id,stage,"
            "call_identity,provider,operation_id,funding_source,amount_microusd,"
            "entry_doc) SELECT 'reservation',%s,%s,%s,%s,1,identity,provider,"
            "provider||'.execute','host',0,'{\"seeded\":true}'::jsonb FROM calls",
            (BASELINE, ROUND, SUBMISSION, RUN),
        )
        cursor.execute(
            "INSERT INTO public.lab_arena_ledger("
            "entry_kind,miner_hotkey,round_id,submission_id,run_id,stage,"
            "call_identity,provider,operation_id,funding_source,amount_microusd,"
            "entry_doc) SELECT 'dispatch',miner_hotkey,round_id,submission_id,"
            "run_id,stage,call_identity,provider,operation_id,funding_source,0,"
            "'{\"seeded\":true}'::jsonb FROM public.lab_arena_ledger "
            "WHERE run_id=%s AND entry_kind='reservation' ORDER BY entry_id",
            (RUN,),
        )
        cursor.execute(
            "INSERT INTO public.lab_arena_ledger("
            "entry_kind,miner_hotkey,round_id,submission_id,run_id,stage,"
            "call_identity,provider,operation_id,funding_source,amount_microusd,"
            "entry_doc,terminal_response) SELECT 'settlement',miner_hotkey,"
            "round_id,submission_id,run_id,stage,call_identity,provider,"
            "operation_id,funding_source,CASE provider WHEN 'openrouter' THEN "
            "19500 ELSE 0 END,'{\"seeded\":true}'::jsonb,"
            "'{\"status\":200,\"call_succeeded\":true}'::jsonb "
            "FROM public.lab_arena_ledger WHERE run_id=%s "
            "AND entry_kind='reservation' ORDER BY entry_id",
            (RUN,),
        )
        cursor.execute("SET session_replication_role=origin")
    connection.commit()


def _snapshot(cursor) -> dict:
    result = {}
    for name, order in (
        ("lab_arena_rounds", "round_id"),
        ("lab_arena_submissions", "submission_id"),
        ("lab_arena_runs", "run_id"),
        ("lab_arena_ledger", "entry_id"),
    ):
        cursor.execute(
            "SELECT COALESCE(jsonb_agg(to_jsonb(row_data) ORDER BY "
            f"{order}),'[]'::jsonb) FROM public.{name} row_data"
        )
        result[name] = cursor.fetchone()[0]
    return result


def _execute(connection) -> None:
    with connection.cursor() as cursor:
        cursor.execute(MIGRATION.read_text(encoding="utf-8"))
    connection.commit()


def _reserve(store: ArenaStore, label: str) -> tuple[str, dict]:
    identity = contracts.provider_call_identity(
        attempt=1,
        assignment_id="sep21-active-assignment",
        icp_position=2,
        action_sequence=int(label.rsplit("-", 1)[-1]),
        operation_id="openrouter.responses",
        request_hash=sha(label),
    )
    return identity, store.reserve_call(
        run_id=RUN,
        lease_token_hash=LEASE_TOKEN_HASH,
        call_identity=identity,
        operation_id="openrouter.responses",
        provider="openrouter",
        funding_source="host",
        amount_microusd=500_000,
        call_doc={"request_hash": sha(label)},
    )


def _trigger_state(cursor) -> str:
    cursor.execute(
        "SELECT tgenabled FROM pg_catalog.pg_trigger WHERE tgrelid="
        "'public.lab_arena_rounds'::regclass AND "
        "tgname='lab_arena_rounds_write_once' AND NOT tgisinternal"
    )
    return cursor.fetchone()[0]


def test_active_leased_run_crosses_old_quota_without_rewriting_state(database):
    psycopg2, dsn = database
    connection = psycopg2.connect(**dsn)
    store = ArenaStore(PsycopgTransport(lambda: psycopg2.connect(**dsn)))
    try:
        _seed(connection)
        before_quota = store.run_quota_snapshot(RUN, LEASE_TOKEN_HASH)
        assert before_quota["providers"]["openrouter"] == {
            "limit": 200,
            "used": 200,
            "remaining": 0,
            "inflight": 0,
        }
        assert before_quota["providers"]["deepline"]["used"] == 25
        _old_identity, refused = _reserve(store, "pre-repair-201")
        assert (refused["status"], refused["reason"]) == (
            "refused",
            "per_icp_quota",
        )
        costs_before = store.submission_costs(SUBMISSION)
        openrouter_before = next(
            row for row in costs_before["providers"]
            if row["kind"] == "execute" and row["provider"] == "openrouter"
        )
        assert openrouter_before["settled_microusd"] == 3_900_000
        assert openrouter_before["reserved_or_uncertain_microusd"] == 0

        with connection.cursor() as cursor:
            state_before = _snapshot(cursor)
        _execute(connection)
        with connection.cursor() as cursor:
            state_after = _snapshot(cursor)
            assert _trigger_state(cursor) == "O"

        expected_rounds = copy.deepcopy(state_before["lab_arena_rounds"])
        target = next(row for row in expected_rounds if row["round_id"] == ROUND)
        target["configuration_doc"]["call_quotas"]["openrouter"] = 2000
        target["updated_at"] = next(
            row for row in state_after["lab_arena_rounds"]
            if row["round_id"] == ROUND
        )["updated_at"]
        assert state_after["lab_arena_rounds"] == expected_rounds
        for table in ("lab_arena_submissions", "lab_arena_runs", "lab_arena_ledger"):
            assert state_after[table] == state_before[table]
        assert store.submission_costs(SUBMISSION) == costs_before

        repaired = store.run_quota_snapshot(RUN, LEASE_TOKEN_HASH)
        assert repaired["providers"]["openrouter"] == {
            "limit": 2000,
            "used": 200,
            "remaining": 1800,
            "inflight": 0,
        }

        existing_identity = "sha256:" + hashlib.sha256(
            b"openrouter:1"
        ).hexdigest()
        existing_rows = store.list_ledger(call_identity=existing_identity)
        existing_replay = store.reserve_call(
            run_id=RUN,
            lease_token_hash=LEASE_TOKEN_HASH,
            call_identity=existing_identity,
            operation_id="openrouter.execute",
            provider="openrouter",
            funding_source="host",
            amount_microusd=500_000,
            call_doc={"request_hash": sha("existing-replay")},
        )
        assert (existing_replay["status"], existing_replay["idempotent"]) == (
            "settled",
            True,
        )
        existing_settlement = store.settle_call(
            run_id=RUN,
            lease_token_hash=LEASE_TOKEN_HASH,
            call_identity=existing_identity,
            actual_microusd=19_500,
            terminal_response={"status": 200, "call_succeeded": True},
        )
        assert (existing_settlement["status"], existing_settlement["idempotent"]) == (
            "settled",
            True,
        )
        assert store.list_ledger(call_identity=existing_identity) == existing_rows

        admitted_identity, admitted = _reserve(store, "post-repair-202")
        assert (admitted["status"], admitted["amount_microusd"]) == (
            "reserved",
            0,
        )
        assert store.mark_dispatched(
            run_id=RUN,
            lease_token_hash=LEASE_TOKEN_HASH,
            call_identity=admitted_identity,
        )["status"] == "dispatched"
        settled = store.settle_call(
            run_id=RUN,
            lease_token_hash=LEASE_TOKEN_HASH,
            call_identity=admitted_identity,
            actual_microusd=100_000,
            terminal_response={"status": 200, "call_succeeded": True},
        )
        assert (settled["status"], settled["idempotent"]) == ("settled", False)
        replay = store.settle_call(
            run_id=RUN,
            lease_token_hash=LEASE_TOKEN_HASH,
            call_identity=admitted_identity,
            actual_microusd=100_000,
            terminal_response={"status": 200, "call_succeeded": True},
        )
        assert (replay["status"], replay["idempotent"]) == ("settled", True)
        assert len(store.list_ledger(call_identity=admitted_identity)) == 3

        _over_identity, over = _reserve(store, "post-repair-203")
        assert (over["status"], over["reason"]) == ("refused", "money_cap")
        costs_after = store.submission_costs(SUBMISSION)
        openrouter_after = next(
            row for row in costs_after["providers"]
            if row["kind"] == "execute" and row["provider"] == "openrouter"
        )
        assert openrouter_after["settled_microusd"] == 4_000_000
        assert openrouter_after["reserved_or_uncertain_microusd"] == 0

        with connection.cursor() as cursor:
            replay_before = _snapshot(cursor)
        _execute(connection)
        with connection.cursor() as cursor:
            assert _snapshot(cursor) == replay_before
            assert _trigger_state(cursor) == "O"
    finally:
        store.close()
        connection.close()


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (
            "UPDATE public.lab_arena_rounds SET configuration_doc=jsonb_set("
            "configuration_doc,'{icp_wall_clock_seconds}','1800'::jsonb,FALSE) "
            f"WHERE round_id='{ROUND}'",
            "profile differs",
        ),
        (
            "UPDATE public.lab_arena_rounds SET status='cancelled',"
            "cancel_reason='test-cancelled' "
            f"WHERE round_id='{ROUND}'",
            "active unpublished",
        ),
        (
            "UPDATE public.lab_arena_rounds SET status='published',"
            "publication_doc='{\"preserve\":true}'::jsonb,"
            "published_at='2026-09-21T12:00:00Z' "
            f"WHERE round_id='{ROUND}'",
            "active unpublished",
        ),
    ),
)
def test_wrong_profile_and_terminal_rounds_fail_without_mutation(
    database, mutation, message
):
    psycopg2, dsn = database
    connection = psycopg2.connect(**dsn)
    try:
        _seed(connection)
        with connection.cursor() as cursor:
            cursor.execute("SET session_replication_role=replica")
            cursor.execute(mutation)
            cursor.execute("SET session_replication_role=origin")
        connection.commit()
        with connection.cursor() as cursor:
            before = _snapshot(cursor)

        with pytest.raises(psycopg2.Error, match=message):
            _execute(connection)
        connection.rollback()
        with connection.cursor() as cursor:
            assert _snapshot(cursor) == before
            assert _trigger_state(cursor) == "O"
    finally:
        connection.close()


def test_migration_is_exactly_bounded_to_the_sep21_round():
    body = MIGRATION.read_text(encoding="utf-8")
    assert MIGRATION.name.startswith("345-")
    assert MIGRATION.name not in CURRENT_SERVICE_MIGRATIONS
    assert "arena-2026-09-21" in body
    assert "arena-2026-09-22" not in body
    assert "DISABLE TRIGGER lab_arena_rounds_write_once" in body
    assert "DISABLE TRIGGER USER" not in body
    assert "DELETE FROM" not in body and "TRUNCATE " not in body
