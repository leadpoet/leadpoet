"""Guarded natural expiry retains closed provider accounting and retries."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.lab_arena import restart_expired_zero_call402_postgres_test as prior


MIGRATION = Path(__file__).parents[2] / "scripts/404-lab-arena-restart-closed-ledger-expiry.sql"
PREIMAGE = prior.POSTIMAGE
POSTIMAGE = (
    "b9cfb9b59ef428d8e43b052258cac72f1fff3d141a736080c8012d644211497b",
    "94c8c23deeccd6b89bb575b0d2782709b0c030810e00bef5691b254b5bb84d35",
)
database = prior.database
migration_sql = prior.migration_sql
installed = prior.installed
prior_upgraded = prior.upgraded
source = prior.source
guard = prior.prior
RUN_ID = source.ASSIGNMENT + ":1"


def _entry(cursor, kind: str, letter: str, amount: int = 1) -> None:
    cursor.execute(
        "INSERT INTO public.lab_arena_ledger("
        "entry_kind,miner_hotkey,round_id,submission_id,run_id,stage,"
        "call_identity,provider,operation_id,funding_source,amount_microusd) "
        "SELECT %s,r.miner_hotkey,r.round_id,r.submission_id,r.run_id,r.stage,"
        "%s,'openrouter','openrouter.chat','host',%s "
        "FROM public.lab_arena_runs r WHERE r.run_id=%s",
        (kind, "sha256:" + letter * 64, amount, RUN_ID),
    )


def _closed_ledger(cursor) -> None:
    _entry(cursor, "reservation", "a", 10)
    _entry(cursor, "dispatch", "a", 10)
    _entry(cursor, "settlement", "a", 6)
    _entry(cursor, "reservation", "b", 10)
    _entry(cursor, "uncertain", "b", 10)
    _entry(cursor, "refusal", "c", 0)


def _ledger_bytes(cursor):
    cursor.execute(
        "SELECT entry_id,pg_catalog.to_jsonb(l)::TEXT "
        "FROM public.lab_arena_ledger l WHERE run_id=%s ORDER BY entry_id",
        (RUN_ID,),
    )
    return cursor.fetchall()


def _execution_spend(cursor) -> int:
    cursor.execute(
        "SELECT public.lab_arena__submission_kind_spend(%s,'execute')",
        (source.SUBMISSION,),
    )
    return cursor.fetchone()[0]


def _acquire(connection) -> int:
    with connection.cursor() as cursor:
        cursor.execute("SELECT guard_generation FROM public.lab_arena_restart_claim_control")
        generation = cursor.fetchone()[0]
    return guard._acquire(connection, generation)["guard_generation"]


def _abort(connection, generation: int) -> None:
    guard._rpc(connection, "lab_arena_abort_restart_guard_v1",
               guard.GUARD, guard.OWNER, generation, "test-abort")


@pytest.fixture(scope="module")
def upgraded(prior_upgraded):
    psycopg, dsn = prior_upgraded
    with psycopg.connect(**dsn) as connection:
        connection.autocommit = True
        with connection.cursor() as cursor:
            before = guard._function_state(cursor)
            assert tuple(row[0] for row in before) == PREIMAGE
            sql = MIGRATION.read_text()
            cursor.execute("BEGIN")
            with pytest.raises(psycopg.Error, match="preimage differs"):
                cursor.execute(sql.replace(PREIMAGE[0], "f" * 64, 1))
            cursor.execute("ROLLBACK")
            assert guard._function_state(cursor) == before
        prior._execute_lease(connection)
        with connection.cursor() as cursor:
            _closed_ledger(cursor)
            ledger_before = _ledger_bytes(cursor)
            assert _execution_spend(cursor) == 16
        generation = _acquire(connection)
        waiting = guard._quiescence(connection, generation=generation)
        assert waiting["preserved"] is False
        assert waiting["still_leased_count"] == 1
        with connection.cursor() as cursor:
            cursor.execute(sql)
            after = guard._function_state(cursor)
            assert tuple(row[0] for row in after) == POSTIMAGE
            cursor.execute(sql)
            assert guard._function_state(cursor) == after
            assert all(row[1:] == old[1:] for row, old in zip(after, before))
        drained = guard._quiescence(connection, generation=generation)
        assert drained["preserved"] is True
        assert (drained["expired_receipt_count"], drained["pending_retry_count"]) == (1, 1)
        with connection.cursor() as cursor:
            assert _ledger_bytes(cursor) == ledger_before
            assert _execution_spend(cursor) == 16
            cursor.execute(
                "SELECT status,terminal_cause,result_doc,output_ref "
                "FROM public.lab_arena_runs WHERE run_id=%s", (RUN_ID,),
            )
            assert cursor.fetchone() == ("failed", "lease_expired", None, None)
        _abort(connection, generation)
    yield psycopg, dsn


def test_paid_closed_heads_expire_without_receipt_change(upgraded):
    psycopg, dsn = upgraded
    with psycopg.connect(**dsn) as connection:
        connection.autocommit = True
        prior._execute_lease(connection, stage=2)
        with connection.cursor() as cursor:
            _closed_ledger(cursor)
            before = _ledger_bytes(cursor)
            assert _execution_spend(cursor) == 16
        generation = _acquire(connection)
        drained = guard._quiescence(connection, generation=generation)
        assert drained["preserved"] is True
        assert (drained["expired_receipt_count"], drained["pending_retry_count"]) == (1, 1)
        with connection.cursor() as cursor:
            assert _ledger_bytes(cursor) == before
            assert _execution_spend(cursor) == 16
            cursor.execute(
                "SELECT status,kind,stage,terminal_cause,result_doc,output_ref "
                "FROM public.lab_arena_runs WHERE run_id=%s", (RUN_ID,),
            )
            assert cursor.fetchone() == ("failed", "execute", 2, "lease_expired", None, None)
            cursor.execute(
                "SELECT status,attempt FROM public.lab_arena_runs WHERE run_id=%s",
                (source.ASSIGNMENT + ":2",),
            )
            assert cursor.fetchone() == ("pending", 2)
        _abort(connection, generation)


@pytest.mark.parametrize("head", ["reservation", "dispatch"])
def test_open_head_stays_leased(upgraded, head):
    psycopg, dsn = upgraded
    with psycopg.connect(**dsn) as connection:
        connection.autocommit = True
        prior._execute_lease(connection)
        with connection.cursor() as cursor:
            _closed_ledger(cursor)
            _entry(cursor, "reservation", "d", 10)
            if head == "dispatch":
                _entry(cursor, "dispatch", "d", 10)
            before = _ledger_bytes(cursor)
        generation = _acquire(connection)
        waiting = guard._quiescence(connection, generation=generation)
        assert waiting["preserved"] is False
        assert waiting["still_leased_count"] == 1
        with connection.cursor() as cursor:
            assert _ledger_bytes(cursor) == before
            cursor.execute("SELECT status FROM public.lab_arena_runs WHERE run_id=%s", (RUN_ID,))
            assert cursor.fetchone()[0] == "leased"
        _abort(connection, generation)


def test_unidentified_ledger_cannot_prove_closed(upgraded):
    psycopg, dsn = upgraded
    with psycopg.connect(**dsn) as connection:
        connection.autocommit = True
        prior._execute_lease(connection)
        with connection.cursor() as cursor:
            _closed_ledger(cursor)
            cursor.execute(
                "INSERT INTO public.lab_arena_ledger("
                "entry_kind,miner_hotkey,round_id,submission_id,run_id,stage,"
                "provider,operation_id,funding_source,amount_microusd) "
                "SELECT 'reservation',miner_hotkey,round_id,submission_id,run_id,stage,"
                "'openrouter','openrouter.chat','host',10 "
                "FROM public.lab_arena_runs WHERE run_id=%s",
                (RUN_ID,),
            )
            before = _ledger_bytes(cursor)
        generation = _acquire(connection)
        waiting = guard._quiescence(connection, generation=generation)
        assert waiting["preserved"] is False
        assert waiting["still_leased_count"] == 1
        with connection.cursor() as cursor:
            assert _ledger_bytes(cursor) == before
            cursor.execute("SELECT status FROM public.lab_arena_runs WHERE run_id=%s", (RUN_ID,))
            assert cursor.fetchone()[0] == "leased"
        _abort(connection, generation)


def test_future_paid_lease_is_not_expired_early(upgraded):
    psycopg, dsn = upgraded
    with psycopg.connect(**dsn) as connection:
        connection.autocommit = True
        prior._execute_lease(connection)
        with connection.cursor() as cursor:
            _closed_ledger(cursor)
            cursor.execute("SET session_replication_role=replica")
            cursor.execute(
                "UPDATE public.lab_arena_runs SET lease_expires_at=now()+interval '5 minutes' "
                "WHERE run_id=%s", (RUN_ID,),
            )
            cursor.execute("SET session_replication_role=origin")
        generation = _acquire(connection)
        waiting = guard._quiescence(connection, generation=generation)
        assert waiting["preserved"] is False
        assert waiting["still_leased_count"] == 1
        _abort(connection, generation)


@pytest.mark.parametrize("kind", ["execute", "score"])
def test_zero_ledger_expiry_still_uses_ordinary_retry(upgraded, kind):
    psycopg, dsn = upgraded
    with psycopg.connect(**dsn) as connection:
        connection.autocommit = True
        if kind == "execute":
            prior._execute_lease(connection)
        else:
            guard._lease(connection, overdue=True)
        generation = _acquire(connection)
        drained = guard._quiescence(connection, generation=generation)
        assert drained["preserved"] is True
        assert (drained["expired_receipt_count"], drained["pending_retry_count"]) == (1, 1)
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT status,kind,terminal_cause FROM public.lab_arena_runs WHERE run_id=%s",
                (RUN_ID,),
            )
            assert cursor.fetchone() == ("failed", kind, "lease_expired")
            cursor.execute(
                "SELECT status,kind FROM public.lab_arena_runs WHERE run_id=%s",
                (source.ASSIGNMENT + ":2",),
            )
            assert cursor.fetchone() == ("pending", kind)
            assert _ledger_bytes(cursor) == []
        _abort(connection, generation)


def test_future_open_head_in_same_round_holds_expiry(upgraded):
    psycopg, dsn = upgraded
    with psycopg.connect(**dsn) as connection:
        connection.autocommit = True
        prior._execute_lease(connection)
        second_assignment = source.ASSIGNMENT + "-future-open"
        second_run = second_assignment + ":1"
        with connection.cursor() as cursor:
            _closed_ledger(cursor)
            before = _ledger_bytes(cursor)
            cursor.execute("SET session_replication_role=replica")
            cursor.execute(
                "INSERT INTO public.lab_arena_runs("
                "run_id,assignment_id,round_id,submission_id,miner_hotkey,stage,"
                "icp_position,attempt,kind,status,lease_generation,stage_generation,"
                "runner_hotkey,lease_token_hash,lease_expires_at) "
                "SELECT %s,%s,round_id,submission_id,miner_hotkey,stage,"
                "1,1,kind,'leased',lease_generation,stage_generation,"
                "runner_hotkey,lease_token_hash,now()+interval '5 minutes' "
                "FROM public.lab_arena_runs WHERE run_id=%s",
                (second_run, second_assignment, RUN_ID),
            )
            cursor.execute("SET session_replication_role=origin")
            for kind in ("reservation", "dispatch"):
                cursor.execute(
                    "INSERT INTO public.lab_arena_ledger("
                    "entry_kind,miner_hotkey,round_id,submission_id,run_id,stage,"
                    "call_identity,provider,operation_id,funding_source,amount_microusd) "
                    "SELECT %s,miner_hotkey,round_id,submission_id,run_id,stage,"
                    "%s,'openrouter','openrouter.chat','host',10 "
                    "FROM public.lab_arena_runs WHERE run_id=%s",
                    (kind, "sha256:" + "e" * 64, second_run),
                )
        generation = _acquire(connection)
        waiting = guard._quiescence(connection, generation=generation)
        assert waiting["preserved"] is False
        assert waiting["still_leased_count"] == 2
        with connection.cursor() as cursor:
            assert _ledger_bytes(cursor) == before
            cursor.execute(
                "SELECT run_id,status FROM public.lab_arena_runs WHERE run_id IN (%s,%s)",
                (RUN_ID, second_run),
            )
            assert set(cursor.fetchall()) == {(RUN_ID, "leased"), (second_run, "leased")}
        _abort(connection, generation)


@pytest.mark.parametrize("change", ["generation", "stage", "output"])
def test_changed_captured_paid_lease_fails_closed(upgraded, change):
    psycopg, dsn = upgraded
    with psycopg.connect(**dsn) as connection:
        connection.autocommit = True
        prior._execute_lease(connection)
        with connection.cursor() as cursor:
            _closed_ledger(cursor)
        generation = _acquire(connection)
        with connection.cursor() as cursor:
            cursor.execute("SET session_replication_role=replica")
            if change == "generation":
                cursor.execute("UPDATE public.lab_arena_runs SET lease_generation=lease_generation+1 WHERE run_id=%s", (RUN_ID,))
            elif change == "stage":
                cursor.execute("UPDATE public.lab_arena_rounds SET status='stage2' WHERE round_id=%s", (source.ROUND,))
            else:
                cursor.execute("UPDATE public.lab_arena_runs SET output_ref='unexpected' WHERE run_id=%s", (RUN_ID,))
            cursor.execute("SET session_replication_role=origin")
        try:
            observed = guard._quiescence(connection, generation=generation)
        except psycopg.Error:
            connection.rollback()
        else:
            assert observed["preserved"] is False
        with connection.cursor() as cursor:
            cursor.execute("SELECT status FROM public.lab_arena_runs WHERE run_id=%s", (RUN_ID,))
            assert cursor.fetchone()[0] == "leased"
        # A changed captured write set also forbids guard abort; the next
        # disposable fixture seed restores the canonical rows.


def test_uncaptured_overdue_lease_fails_closed(upgraded):
    psycopg, dsn = upgraded
    with psycopg.connect(**dsn) as connection:
        connection.autocommit = True
        prior._execute_lease(connection)
        with connection.cursor() as cursor:
            _closed_ledger(cursor)
            before = _ledger_bytes(cursor)
        generation = _acquire(connection)
        second_assignment = source.ASSIGNMENT + "-uncaptured"
        second_run = second_assignment + ":1"
        with connection.cursor() as cursor:
            cursor.execute("SET session_replication_role=replica")
            cursor.execute(
                "INSERT INTO public.lab_arena_runs("
                "run_id,assignment_id,round_id,submission_id,miner_hotkey,stage,"
                "icp_position,attempt,kind,status,lease_generation,stage_generation,"
                "runner_hotkey,lease_token_hash,lease_expires_at) "
                "SELECT %s,%s,round_id,submission_id,miner_hotkey,stage,"
                "1,1,kind,'leased',lease_generation,stage_generation,"
                "runner_hotkey,lease_token_hash,now()-interval '1 second' "
                "FROM public.lab_arena_runs WHERE run_id=%s",
                (second_run, second_assignment, RUN_ID),
            )
            cursor.execute("SET session_replication_role=origin")
        connection.commit()
        with pytest.raises(psycopg.Error, match="captured expiry write set differs"):
            guard._quiescence(connection, generation=generation)
        connection.rollback()
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT run_id,status FROM public.lab_arena_runs WHERE run_id IN (%s,%s) ORDER BY run_id",
                (RUN_ID, second_run),
            )
            assert set(cursor.fetchall()) == {(RUN_ID, "leased"), (second_run, "leased")}
            assert _ledger_bytes(cursor) == before
