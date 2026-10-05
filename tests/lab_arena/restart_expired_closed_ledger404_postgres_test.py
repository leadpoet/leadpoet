"""Guarded natural expiry keeps terminal provider accounting unchanged."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.lab_arena import restart_expired_zero_call402_postgres_test as prior


database = prior.database
migration_sql = prior.migration_sql
installed = prior.installed
upgraded = prior.upgraded
MIGRATION = Path(__file__).parents[2] / "scripts/404-lab-arena-restart-expired-closed-ledger-drain.sql"
SCORE_RETRY = Path(__file__).parents[2] / "scripts/398-lab-arena-zero-call-score-host-retry.sql"
TRAJECTORIES = Path(__file__).parents[2] / "scripts/365-lab-arena-trajectories.sql"
RUN_ID = prior.source.ASSIGNMENT + ":1"
PREIMAGE = prior.POSTIMAGE
POSTIMAGE = (
    "d36a3c7ca0ca92b03f50ee7ca97fba3580ff7a50ab20b8c2e5e44f443a1bdbdb",
    "dbb9f5270a190b6604ca8ed77abf5577bff2a58b11ec39820cadba3e662e214c",
)
ACCOUNTING_FUNCTIONS = (
    "public.lab_arena_expire_leases(text)",
    "public.lab_arena__terminate_open_calls(text,text)",
)


def _accounting_state(cursor):
    state = []
    for signature in ACCOUNTING_FUNCTIONS:
        cursor.execute(
            "SELECT encode(extensions.digest(pg_get_functiondef(p.oid),'sha256'),'hex'),"
            "owner.rolname,p.proacl::text,p.prosecdef,p.provolatile,p.proconfig "
            "FROM pg_catalog.pg_proc AS p "
            "JOIN pg_catalog.pg_roles AS owner ON owner.oid=p.proowner "
            "WHERE p.oid=%s::regprocedure", (signature,),
        )
        state.append(cursor.fetchone())
    return state


@pytest.fixture(scope="module")
def migrated(upgraded):
    psycopg, dsn = upgraded
    with psycopg.connect(**dsn) as connection:
        connection.autocommit = True
        with connection.cursor() as cursor:
            cursor.execute(TRAJECTORIES.read_text())
            cursor.execute(SCORE_RETRY.read_text())
            accounting_before = _accounting_state(cursor)
            assert tuple(row[0] for row in accounting_before) == (
                "21634a503b8526bd51dbe102bb89b0bbcbdef582057070ae1d66324b0ebc0532",
                "6d59ebe2f7dc254bc190ddbb6b5b1f3cc0df0f46807b0a029c44f123d9dcbca1",
            )
            before = prior.prior._function_state(cursor)
            assert tuple(row[0] for row in before) == PREIMAGE
            sql = MIGRATION.read_text()
            cursor.execute("BEGIN")
            with pytest.raises(psycopg.Error, match="preimage differs"):
                cursor.execute(sql.replace(PREIMAGE[0], "0" * 64, 1))
            cursor.execute("ROLLBACK")
            assert prior.prior._function_state(cursor) == before
            cursor.execute(sql)
            after = prior.prior._function_state(cursor)
            assert tuple(row[0] for row in after) == POSTIMAGE
            cursor.execute(sql)
            assert prior.prior._function_state(cursor) == after
            assert all(row[1:] == old[1:] for row, old in zip(after, before))
            assert _accounting_state(cursor) == accounting_before
    return upgraded


def _connection(migrated):
    psycopg, dsn = migrated
    connection = psycopg.connect(**dsn)
    connection.autocommit = True
    return connection


def _seed_lease(connection, *, kind="execute", stage=1, overdue=True):
    if kind == "execute":
        prior._execute_lease(connection, stage=stage)
    else:
        prior.prior._lease(connection, overdue=True)
    if not overdue:
        with connection.cursor() as cursor:
            cursor.execute("SET session_replication_role=replica")
            cursor.execute(
                "UPDATE public.lab_arena_runs SET lease_expires_at="
                "clock_timestamp()+interval '10 minutes' WHERE run_id=%s", (RUN_ID,)
            )
            cursor.execute("SET session_replication_role=origin")
        connection.commit()


def _generation(connection):
    with connection.cursor() as cursor:
        cursor.execute("SELECT guard_generation FROM public.lab_arena_restart_claim_control")
        return cursor.fetchone()[0]


def _guard(connection):
    state = prior.prior._acquire(connection, _generation(connection))
    assert state["drain"]["captured_count"] == 1
    connection.commit()
    return state["guard_generation"]


def _insert_call(connection, *, head, identity="e", run_id=RUN_ID):
    sequence = ["reservation"]
    if head in ("settlement", "uncertain", "dispatch"):
        sequence.append("dispatch")
    if head not in ("reservation", "dispatch"):
        sequence.append(head)
    with connection.cursor() as cursor:
        for entry_kind in sequence:
            cursor.execute(
                "INSERT INTO public.lab_arena_ledger("
                "entry_kind,miner_hotkey,round_id,submission_id,run_id,stage,"
                "call_identity,provider,operation_id,funding_source,amount_microusd) "
                "VALUES (%s,%s,%s,%s,%s,1,%s,'openrouter','openrouter.chat','host',7)",
                (entry_kind, prior.source.MINER, prior.source.ROUND,
                 prior.source.SUBMISSION, run_id, "sha256:" + identity * 64),
            )
    connection.commit()


def _ledger(connection, run_id=RUN_ID):
    with connection.cursor() as cursor:
        cursor.execute(
            "SELECT * FROM public.lab_arena_ledger WHERE run_id=%s ORDER BY entry_id",
            (run_id,),
        )
        return cursor.fetchall()


@pytest.mark.parametrize("kind,head", [
    ("execute", "settlement"), ("execute", "recovery"),
    ("execute", "uncertain"), ("score", "settlement"),
    ("score", "recovery"), ("score", "uncertain"),
])
def test_closed_paid_heads_expire_without_ledger_change(migrated, kind, head):
    with _connection(migrated) as connection:
        _seed_lease(connection, kind=kind)
        _insert_call(connection, head=head)
        before = _ledger(connection)
        generation = _guard(connection)
        drained = prior.prior._quiescence(connection, generation=generation)
        assert (drained["preserved"], drained["expired_receipt_count"],
                drained["lost_or_mutated_count"], drained["pending_retry_count"]) == (
            True, 1, 0, 1,
        )
        assert _ledger(connection) == before
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT status,terminal_cause,result_doc,output_ref,"
                "terminal_doc->>'expired_at' FROM public.lab_arena_runs WHERE run_id=%s",
                (RUN_ID,),
            )
            status, cause, result, output, expired_at = cursor.fetchone()
            assert (status, cause, result, output) == (
                "failed", "lease_expired", None, None,
            )
            assert expired_at is not None
            cursor.execute(
                "SELECT status,attempt,kind,result_doc,output_ref FROM public.lab_arena_runs "
                "WHERE run_id=%s", (prior.source.ASSIGNMENT + ":2",),
            )
            assert cursor.fetchone() == ("pending", 2, kind, None, None)
        prior.prior._rpc(
            connection, "lab_arena_abort_restart_guard_v1",
            prior.prior.GUARD, prior.prior.OWNER, generation, "test-abort",
        )


@pytest.mark.parametrize("head", ["reservation", "dispatch"])
def test_open_paid_head_remains_leased(migrated, head):
    with _connection(migrated) as connection:
        _seed_lease(connection)
        _insert_call(connection, head=head)
        before = _ledger(connection)
        generation = _guard(connection)
        waiting = prior.prior._quiescence(connection, generation=generation)
        assert waiting["still_leased_count"] == 1
        assert waiting["preserved"] is False
        assert _ledger(connection) == before
        prior.prior._rpc(
            connection, "lab_arena_abort_restart_guard_v1",
            prior.prior.GUARD, prior.prior.OWNER, generation, "test-abort",
        )


def test_live_paid_lease_waits_for_real_expiry(migrated):
    with _connection(migrated) as connection:
        _seed_lease(connection, overdue=False)
        _insert_call(connection, head="settlement")
        before = _ledger(connection)
        generation = _guard(connection)
        assert prior.source._claim(
            connection, prior.source.RUNNER_A, "f"
        ) == {"status": "paused"}
        waiting = prior.prior._quiescence(connection, generation=generation)
        assert waiting["still_leased_count"] == 1
        assert waiting["preserved"] is False
        assert _ledger(connection) == before
        prior.prior._rpc(
            connection, "lab_arena_abort_restart_guard_v1",
            prior.prior.GUARD, prior.prior.OWNER, generation, "test-abort",
        )


def test_score_claims_stay_paused_under_guard(migrated):
    with _connection(migrated) as connection:
        _seed_lease(connection, kind="score", overdue=False)
        _insert_call(connection, head="uncertain")
        generation = _guard(connection)
        assert prior.source._claim(
            connection, prior.source.RUNNER_A, "a"
        ) == {"status": "paused"}
        prior.prior._rpc(
            connection, "lab_arena_abort_restart_guard_v1",
            prior.prior.GUARD, prior.prior.OWNER, generation, "test-abort",
        )


@pytest.mark.parametrize("change", [
    "owner", "generation", "phase", "stage", "stage_generation",
    "output", "result", "foreign_overdue",
])
def test_guard_rejects_changed_authority_or_expiry_write_set(migrated, change):
    with _connection(migrated) as connection:
        _seed_lease(connection)
        _insert_call(connection, head="settlement")
        before = _ledger(connection)
        generation = _guard(connection)
        if change in ("phase", "stage", "stage_generation", "output", "result", "foreign_overdue"):
            with connection.cursor() as cursor:
                cursor.execute("SET session_replication_role=replica")
                if change == "phase":
                    cursor.execute(
                        "UPDATE public.lab_arena_restart_claim_control "
                        "SET restart_phase='gateway_destructive' WHERE singleton"
                    )
                elif change == "stage":
                    cursor.execute(
                        "UPDATE public.lab_arena_rounds SET status='stage1_scoring' "
                        "WHERE round_id=%s", (prior.source.ROUND,)
                    )
                elif change == "stage_generation":
                    cursor.execute(
                        "UPDATE public.lab_arena_rounds SET stage_generation=2 "
                        "WHERE round_id=%s", (prior.source.ROUND,)
                    )
                elif change in ("output", "result"):
                    cursor.execute(
                        "UPDATE public.lab_arena_runs SET "
                        + ("output_ref='saved-output'" if change == "output"
                           else "result_doc='{}'::jsonb")
                        + " WHERE run_id=%s", (RUN_ID,)
                    )
                else:
                    cursor.execute(
                        "INSERT INTO public.lab_arena_runs("
                        "run_id,assignment_id,round_id,submission_id,miner_hotkey,"
                        "stage,icp_position,attempt,status,lease_generation,"
                        "stage_generation,kind,runner_hotkey,lease_expires_at,"
                        "lease_token_hash) "
                        "SELECT run_id||'-foreign',assignment_id||'-foreign',"
                        "round_id,submission_id,miner_hotkey,stage,icp_position+1,"
                        "attempt,status,lease_generation,stage_generation,kind,"
                        "runner_hotkey,lease_expires_at,lease_token_hash "
                        "FROM public.lab_arena_runs WHERE run_id=%s", (RUN_ID,)
                    )
                cursor.execute("SET session_replication_role=origin")
            connection.commit()
        if change in ("owner", "generation", "stage", "stage_generation", "output", "result", "foreign_overdue"):
            with pytest.raises(Exception, match=(
                "owner_or_generation_differs" if change in ("owner", "generation")
                else "captured expiry write set differs"
            )):
                prior.prior._quiescence(
                    connection,
                    owner=("lab_arena_restart_owner:" + "f" * 64)
                    if change == "owner" else prior.prior.OWNER,
                    generation=generation + (1 if change == "generation" else 0),
                )
            connection.rollback()
        else:
            waiting = prior.prior._quiescence(connection, generation=generation)
            assert waiting["still_leased_count"] == 1
            assert waiting["preserved"] is False
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT status FROM public.lab_arena_runs WHERE run_id=%s",
                (RUN_ID,),
            )
            assert cursor.fetchone()[0] == "leased"
        assert _ledger(connection) == before
        if change == "phase":
            with connection.cursor() as cursor:
                cursor.execute("SET session_replication_role=replica")
                cursor.execute(
                    "UPDATE public.lab_arena_restart_claim_control "
                    "SET restart_phase='draining' WHERE singleton"
                )
                cursor.execute("SET session_replication_role=origin")
        prior.prior._rpc(
            connection, "lab_arena_abort_restart_guard_v1",
            prior.prior.GUARD, prior.prior.OWNER, generation, "test-abort",
        )


def test_accepted_paid_receipt_is_not_expired(migrated):
    with _connection(migrated) as connection:
        _seed_lease(connection)
        _insert_call(connection, head="settlement")
        before = _ledger(connection)
        generation = _guard(connection)
        with connection.cursor() as cursor:
            cursor.execute("SET session_replication_role=replica")
            cursor.execute(
                "UPDATE public.lab_arena_runs SET status='accepted',"
                "terminal_cause='accepted',result_doc='{"
                "\"terminal_status\":\"accepted\"}'::jsonb,"
                "output_ref='arena/test/accepted.json' WHERE run_id=%s",
                (RUN_ID,),
            )
            cursor.execute("SET session_replication_role=origin")
        connection.commit()
        drained = prior.prior._quiescence(connection, generation=generation)
        assert drained["accepted_receipt_count"] == 1
        assert drained["expired_receipt_count"] == 0
        assert drained["preserved"] is True, drained
        assert _ledger(connection) == before
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT status FROM public.lab_arena_runs WHERE run_id=%s",
                (RUN_ID,),
            )
            assert cursor.fetchone()[0] == "accepted"
        prior.prior._rpc(
            connection, "lab_arena_abort_restart_guard_v1",
            prior.prior.GUARD, prior.prior.OWNER, generation, "test-abort",
        )


def test_paid_exhausted_score_promotes_group_follower_without_new_attempt(migrated):
    with _connection(migrated) as connection:
        _seed_lease(connection, kind="score")
        exhausted = prior.source.ASSIGNMENT + ":2"
        follower = prior.source.ASSIGNMENT + "-follower:1"
        cache_key = "sha256:" + "d" * 64
        with connection.cursor() as cursor:
            cursor.execute("SET session_replication_role=replica")
            cursor.execute(
                "UPDATE public.lab_arena_runs SET run_id=%s,attempt=2,"
                "judgment_cache_key=%s,judgment_input_hash=%s,"
                "judgment_scope_doc='{}'::jsonb,judgment_group_leader=TRUE "
                "WHERE run_id=%s",
                (exhausted, cache_key, "sha256:" + "c" * 64, RUN_ID),
            )
            cursor.execute(
                "INSERT INTO public.lab_arena_runs("
                "run_id,assignment_id,round_id,submission_id,miner_hotkey,"
                "stage,icp_position,attempt,kind,status,stage_generation,"
                "judgment_cache_key,judgment_input_hash,judgment_scope_doc,"
                "judgment_group_leader) "
                "SELECT %s,assignment_id||'-follower',round_id,submission_id,"
                "miner_hotkey,stage,icp_position+1,1,'score','pending',"
                "stage_generation,judgment_cache_key,judgment_input_hash,"
                "judgment_scope_doc,FALSE FROM public.lab_arena_runs WHERE run_id=%s",
                (follower, exhausted),
            )
            cursor.execute("SET session_replication_role=origin")
        connection.commit()
        _insert_call(connection, head="uncertain", run_id=exhausted)
        before = _ledger(connection, exhausted)
        generation = _guard(connection)
        drained = prior.prior._quiescence(connection, generation=generation)
        assert drained["preserved"] is True, drained
        assert (drained["expired_receipt_count"], drained["pending_retry_count"]) == (1, 0)
        assert _ledger(connection, exhausted) == before
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT status,terminal_cause FROM public.lab_arena_runs WHERE run_id=%s",
                (exhausted,),
            )
            assert cursor.fetchone() == ("failed", "lease_expired")
            cursor.execute(
                "SELECT status,judgment_group_leader FROM public.lab_arena_runs "
                "WHERE run_id=%s", (follower,),
            )
            assert cursor.fetchone() == ("pending", True)
            cursor.execute(
                "SELECT count(*) FROM public.lab_arena_runs WHERE assignment_id=%s",
                (prior.source.ASSIGNMENT,),
            )
            assert cursor.fetchone()[0] == 1
        prior.prior._rpc(
            connection, "lab_arena_abort_restart_guard_v1",
            prior.prior.GUARD, prior.prior.OWNER, generation, "test-abort",
        )
