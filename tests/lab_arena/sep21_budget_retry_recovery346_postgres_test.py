"""Real-PostgreSQL proof for the bounded Sep21 budget retry recovery."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from lab_arena import contracts, scoring
from lab_arena.store import ArenaStore, PsycopgTransport
from tests.lab_arena.lab_arena_pg_harness import (
    CURRENT_SERVICE_MIGRATIONS,
    database_with_lab_arena_migration,
)
from tests.lab_arena.test_lab_arena_migration_postgres import hotkey, sha


ROOT = Path(__file__).parents[2]
MIGRATION = ROOT / "scripts/346-arena-2026-09-21-budget-retry-recovery.sql"
ROUND = "arena-2026-09-21"
BASELINE = "baseline-2026-09-21"
SOURCE_REF = f"arena/{ROUND}/sources/{BASELINE}.tar.gz"
SOURCE_SIZE = 858_906
SOURCE_SHA256 = "82a444e0282bac6820a61a4dcb7f8aba33423318f64ebd3aaf94c0b35415adc3"
SOURCE_COMMIT = "8e467397527cba8cde839a3f86302e35eb5a2edd"
BASELINE_HOTKEY = hotkey("sep21-retry-baseline")
RUNNER_A = hotkey("sep21-retry-runner-a")
RUNNER_B = hotkey("sep21-retry-runner-b")
LEASE_HASH = "sha256:" + "a" * 64
TARGETS = tuple(range(10))

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


def _configuration() -> dict:
    return {
        "schema_version": contracts.ROUND_CONFIGURATION_SCHEMA_VERSION,
        "round_id": ROUND,
        "mode": "live",
        "network_name": "finney",
        "netuid": 71,
        "rewards_enabled": True,
        "schedule": {"submission_cutoff": "2026-09-21T00:00:00Z"},
        "stage_1_icp_count": 10,
        "stage_2_icp_count": 10,
        "max_attempts_per_assignment": 2,
        "call_quotas": {
            "scrapingdog": 200,
            "deepline": 200,
            "openrouter": 2000,
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
        "checkpoint_deadline_policy": "atomic_checkpoint_45m_v1",
        "parallel_twenty_icp_execution": True,
        "runner_slot_ceiling": 20,
        "scoring_cap_microusd": 50_000_000,
        "baseline_hotkey": BASELINE_HOTKEY,
        "baseline_source_url": "https://example.invalid/baseline.tar.gz",
        "scorer_policy": {"scoring_adapter_version": "qualification_integrity_v2"},
        "reward_constants": {"pool_percent": 30},
    }


def _assignment(submission_id: str, position: int) -> str:
    stage = 1 if position < 10 else 2
    return f"{ROUND}:{submission_id}:{stage}:{position}"


def _seed(connection) -> None:
    participants = []
    submissions = []
    for index in range(6):
        submission_id = BASELINE if index == 0 else f"sep21-miner-{index}"
        miner_hotkey = BASELINE_HOTKEY if index == 0 else hotkey(f"sep21-miner-{index}")
        source_ref = SOURCE_REF if index == 0 else f"arena/{ROUND}/sources/{submission_id}.tar.gz"
        source_size = SOURCE_SIZE if index == 0 else 100_000 + index
        participants.append({
            "submission_id": submission_id,
            "miner_hotkey": miner_hotkey,
            "is_king": index == 0,
            "source_ref": source_ref,
            "source_size_bytes": source_size,
        })
        document = {
            "consent": {"public_rerun": True},
            "is_king": index == 0,
            "source_ref": source_ref,
            "source_size_bytes": source_size,
        }
        submissions.append((submission_id, miner_hotkey, index == 0, source_ref, source_size, document))

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
            "rewards_enabled,participants,benchmark_ref,evaluation_date,icp_set_date) "
            "VALUES (%s,'stage1',9,7,%s::jsonb,TRUE,%s::jsonb,%s,%s,%s)",
            (
                ROUND,
                json.dumps(_configuration()),
                json.dumps(participants),
                f"arena/{ROUND}/benchmark.json",
                "2026-09-21",
                "2026-09-20",
            ),
        )
        cursor.executemany(
            "INSERT INTO public.lab_arena_submissions("
            "submission_id,round_id,miner_hotkey,status,is_king,source_ref,"
            "source_size_bytes,submission_doc,frozen_at) VALUES "
            "(%s,%s,%s,'frozen',%s,%s,%s,%s::jsonb,'2026-09-21T00:00:32Z')",
            [
                (submission_id, ROUND, miner_hotkey, is_king, source_ref, source_size, json.dumps(document))
                for submission_id, miner_hotkey, is_king, source_ref, source_size, document in submissions
            ],
        )

        run_rows = []
        for submission_id, miner_hotkey, is_king, _ref, _size, _doc in submissions:
            for position in range(20):
                assignment = _assignment(submission_id, position)
                stage = 1 if position < 10 else 2
                if is_king and position < 10:
                    status = "failed"
                    cause = "budget_exhausted"
                    result_doc = {"terminal_status": cause, "preserve": position}
                    output_ref = None
                    lease_hash = None
                    expires = None
                elif is_king:
                    status = "leased"
                    cause = None
                    result_doc = None
                    output_ref = None
                    lease_hash = "sha256:" + hashlib.sha256(str(position).encode()).hexdigest()
                    expires = "2026-09-21T02:00:00Z"
                else:
                    status = "accepted"
                    cause = "accepted"
                    result_doc = {"terminal_status": cause, "preserve": submission_id}
                    output_ref = f"arena/{ROUND}/outputs/{assignment}:1.json"
                    lease_hash = None
                    expires = None
                run_rows.append((
                    assignment + ":1", assignment, ROUND, submission_id,
                    miner_hotkey, stage, position, 1, "execute", status,
                    RUNNER_A, lease_hash, 2, 7, expires, cause,
                    json.dumps(result_doc) if result_doc is not None else None,
                    output_ref,
                ))
        cursor.executemany(
            "INSERT INTO public.lab_arena_runs("
            "run_id,assignment_id,round_id,submission_id,miner_hotkey,stage,"
            "icp_position,attempt,kind,status,runner_hotkey,lease_token_hash,"
            "lease_generation,stage_generation,lease_expires_at,terminal_cause,"
            "result_doc,output_ref) VALUES ("
            "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb,%s)",
            run_rows,
        )

        # Each failed target has exactly 200 settled OpenRouter calls costing
        # $3.90 total, followed by its terminal quota refusal.
        cursor.execute(
            "WITH calls AS ("
            " SELECT position,ordinal,'sha256:'||encode(extensions.digest("
            "   'sep21:'||position::text||':'||ordinal::text,'sha256'),'hex') identity"
            " FROM generate_series(0,9) position CROSS JOIN generate_series(1,200) ordinal"
            ") INSERT INTO public.lab_arena_ledger("
            "entry_kind,miner_hotkey,round_id,submission_id,run_id,stage,"
            "call_identity,provider,operation_id,funding_source,amount_microusd,"
            "entry_doc,terminal_response) SELECT 'settlement',%s,%s,%s,"
            "%s||position::text||':1',1,identity,'openrouter',"
            "'openrouter.responses','host',19500,'{\"seeded\":true}'::jsonb,"
            "'{\"status\":200,\"call_succeeded\":true}'::jsonb FROM calls",
            (BASELINE_HOTKEY, ROUND, BASELINE, f"{ROUND}:{BASELINE}:1:"),
        )
        cursor.execute(
            "WITH positions AS (SELECT generate_series(0,9) position) "
            "INSERT INTO public.lab_arena_ledger("
            "entry_kind,miner_hotkey,round_id,submission_id,run_id,stage,"
            "call_identity,provider,operation_id,funding_source,amount_microusd,"
            "entry_doc) SELECT 'refusal',%s,%s,%s,%s||position::text||':1',1,"
            "'sha256:'||encode(extensions.digest('refusal:'||position::text,'sha256'),'hex'),"
            "'openrouter','openrouter.responses','host',0,"
            "'{\"reason\":\"per_icp_quota\"}'::jsonb FROM positions",
            (BASELINE_HOTKEY, ROUND, BASELINE, f"{ROUND}:{BASELINE}:1:"),
        )
        cursor.execute("SET session_replication_role=origin")
    connection.commit()


def _snapshot(cursor) -> dict:
    result = {}
    for table, order in (
        ("lab_arena_rounds", "round_id"),
        ("lab_arena_submissions", "submission_id"),
        ("lab_arena_runs", "run_id"),
        ("lab_arena_ledger", "entry_id"),
    ):
        cursor.execute(
            "SELECT COALESCE(jsonb_agg(to_jsonb(rows) ORDER BY %s),'[]'::jsonb) "
            "FROM public.%s AS rows" % (order, table)
        )
        result[table] = cursor.fetchone()[0]
    return result


def _apply(connection) -> None:
    with connection.cursor() as cursor:
        cursor.execute(MIGRATION.read_text(encoding="utf-8"))
    connection.commit()


def _run_id(position: int, attempt: int) -> str:
    return _assignment(BASELINE, position) + f":{attempt}"


def test_recovery_adds_only_retries_and_preserves_progress_on_replay(database):
    psycopg2, dsn = database
    connection = psycopg2.connect(**dsn)
    store = ArenaStore(PsycopgTransport(lambda: psycopg2.connect(**dsn)))
    try:
        _seed(connection)
        with connection.cursor() as cursor:
            before = _snapshot(cursor)
        _apply(connection)
        with connection.cursor() as cursor:
            after = _snapshot(cursor)

        assert after["lab_arena_rounds"] == before["lab_arena_rounds"]
        assert after["lab_arena_submissions"] == before["lab_arena_submissions"]
        assert after["lab_arena_ledger"] == before["lab_arena_ledger"]
        old_runs = {row["run_id"]: row for row in before["lab_arena_runs"]}
        new_runs = {row["run_id"]: row for row in after["lab_arena_runs"]}
        assert {key: new_runs[key] for key in old_runs} == old_runs
        retries = [new_runs[_run_id(position, 2)] for position in TARGETS]
        assert all(row["status"] == "pending" for row in retries)
        assert all(row["attempt"] == 2 and row["stage_generation"] == 7 for row in retries)
        assert all(row["previous_runner_hotkey"] == RUNNER_A for row in retries)
        assert all(new_runs[_run_id(position, 1)]["status"] == "leased" for position in range(10, 20))
        assert len([row for row in new_runs.values() if row["status"] == "accepted" and row["submission_id"] != BASELINE]) == 100

        # Progress two retries. Position 0 is accepted and must replace its
        # earlier budget zero in the scoring plan. Position 1 proves that the
        # original $3.90 remains charged and only $0.10 is left.
        with connection.cursor() as cursor:
            cursor.execute("SET session_replication_role=replica")
            cursor.execute(
                "UPDATE public.lab_arena_runs SET status='accepted',"
                "terminal_cause='accepted',result_doc=%s::jsonb,output_ref=%s "
                "WHERE run_id=%s",
                (
                    json.dumps({"terminal_status": "accepted"}),
                    f"arena/{ROUND}/outputs/{_run_id(0, 2)}.json",
                    _run_id(0, 2),
                ),
            )
            cursor.execute(
                "UPDATE public.lab_arena_runs SET status='leased',runner_hotkey=%s,"
                "lease_token_hash=%s,lease_generation=lease_generation+1,"
                "lease_expires_at=clock_timestamp()+interval '1 hour' WHERE run_id=%s",
                (RUNNER_B, LEASE_HASH, _run_id(1, 2)),
            )
            cursor.execute("SET session_replication_role=origin")
        connection.commit()

        plan = scoring.build_scoring_plan(
            round_id=ROUND,
            stage=1,
            runs=store.list_runs(ROUND, stage=1, kind="execute"),
        )
        selected = next(
            item for item in plan["work_items"]
            if item["submission_id"] == BASELINE and item["icp_position"] == 0
        )
        assert selected["scored_run_id"] == _run_id(0, 2)
        assert not any(
            row["submission_id"] == BASELINE and row["icp_position"] == 0
            for row in plan["zero_rows"]
        )

        identity = contracts.provider_call_identity(
            attempt=2,
            assignment_id=_assignment(BASELINE, 1),
            icp_position=1,
            action_sequence=201,
            operation_id="openrouter.responses",
            request_hash=sha("sep21-retry-cost-201"),
        )
        reserved = store.reserve_call(
            run_id=_run_id(1, 2),
            lease_token_hash=LEASE_HASH,
            call_identity=identity,
            operation_id="openrouter.responses",
            provider="openrouter",
            funding_source="host",
            amount_microusd=500_000,
            call_doc={"request_hash": sha("sep21-retry-cost-201")},
        )
        assert (reserved["status"], reserved["amount_microusd"]) == ("reserved", 0)
        assert store.mark_dispatched(
            run_id=_run_id(1, 2), lease_token_hash=LEASE_HASH,
            call_identity=identity,
        )["status"] == "dispatched"
        assert store.settle_call(
            run_id=_run_id(1, 2), lease_token_hash=LEASE_HASH,
            call_identity=identity, actual_microusd=100_000,
            terminal_response={"status": 200, "call_succeeded": True},
        )["status"] == "settled"
        next_identity = contracts.provider_call_identity(
            attempt=2,
            assignment_id=_assignment(BASELINE, 1),
            icp_position=1,
            action_sequence=202,
            operation_id="openrouter.responses",
            request_hash=sha("sep21-retry-cost-202"),
        )
        refused = store.reserve_call(
            run_id=_run_id(1, 2), lease_token_hash=LEASE_HASH,
            call_identity=next_identity,
            operation_id="openrouter.responses", provider="openrouter",
            funding_source="host", amount_microusd=1,
            call_doc={"request_hash": sha("sep21-retry-cost-202")},
        )
        assert (refused["status"], refused["reason"]) == ("refused", "money_cap")

        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT public.lab_arena__successful_icp_cost_state(%s,%s,1) "
                "->>'settled_microusd'",
                (ROUND, BASELINE),
            )
            assert int(cursor.fetchone()[0]) == 4_000_000
            progressed = _snapshot(cursor)
        _apply(connection)
        with connection.cursor() as cursor:
            assert _snapshot(cursor) == progressed
    finally:
        store.close()
        connection.close()


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (
            "UPDATE public.lab_arena_submissions SET submission_doc="
            "jsonb_set(submission_doc,'{source_ref}','\"bad\"'::jsonb,FALSE) "
            f"WHERE submission_id='{BASELINE}'",
            "source or profile differs",
        ),
        (
            "UPDATE public.lab_arena_rounds SET configuration_doc="
            "jsonb_set(configuration_doc,'{call_quotas,openrouter}','200'::jsonb,FALSE) "
            f"WHERE round_id='{ROUND}'",
            "source or profile differs",
        ),
        (
            "UPDATE public.lab_arena_runs SET terminal_cause='model_error',"
            "result_doc='{\"terminal_status\":\"model_error\"}'::jsonb "
            f"WHERE run_id='{_run_id(0, 1)}'",
            "target failures differ",
        ),
        (
            "INSERT INTO public.lab_arena_runs("
            "run_id,assignment_id,round_id,submission_id,miner_hotkey,stage,"
            "icp_position,attempt,kind,status,stage_generation) VALUES ("
            f"'{_run_id(0, 2)}','{_assignment(BASELINE, 0)}','{ROUND}',"
            f"'{BASELINE}','{BASELINE_HOTKEY}',1,0,2,'execute','pending',7)",
            "conflicting second attempts",
        ),
        (
            "INSERT INTO public.lab_arena_ledger("
            "entry_kind,miner_hotkey,round_id,submission_id,run_id,stage,"
            "call_identity,provider,operation_id,funding_source,amount_microusd,entry_doc) "
            f"VALUES ('dispatch','{BASELINE_HOTKEY}','{ROUND}','{BASELINE}',"
            f"'{_run_id(0, 1)}',1,'sha256:{'f' * 64}','openrouter',"
            "'openrouter.responses','host',0,'{}'::jsonb)",
            "ledger differs or is inflight",
        ),
    ),
)
def test_mismatched_or_inflight_targets_fail_atomically(database, mutation, message):
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
            _apply(connection)
        connection.rollback()
        with connection.cursor() as cursor:
            assert _snapshot(cursor) == before
    finally:
        connection.close()


def test_migration_is_exactly_bounded():
    body = MIGRATION.read_text(encoding="utf-8")
    assert MIGRATION.name.startswith("346-")
    assert MIGRATION.name not in CURRENT_SERVICE_MIGRATIONS
    assert SOURCE_SHA256 in body and SOURCE_COMMIT in body
    assert "DELETE FROM" not in body and "TRUNCATE " not in body
    assert "UPDATE public.lab_arena_rounds" not in body
    assert "ALTER TABLE public.lab_arena_rounds" not in body
