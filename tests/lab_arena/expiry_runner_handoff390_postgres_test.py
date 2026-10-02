"""Expiry retry keeps validator identity and hands zero-call score failures off."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from tests.lab_arena import zero_setup_runner_handoff360_postgres_test as prior


database = prior.database
MIGRATION = Path(__file__).parents[2] / 'scripts/390-lab-arena-expired-retry-runner-handoff.sql'
CLAIM_HASH = '0252fa3efafe895a7cade9169a36493dc8be70a9148a5ca037d0d0dde870a7c0'
LIVE_PREIMAGE_ENV = 'LAB_ARENA_CLAIM390_LIVE_PREIMAGE_JSON'


def _hash(cursor, signature):
    cursor.execute("SELECT encode(extensions.digest(pg_get_functiondef(%s::regprocedure),"
                   "'sha256'),'hex')", (signature,))
    return cursor.fetchone()[0]


@pytest.fixture(scope='module')
def migration_sql(database):
    psycopg, dsn = database
    with psycopg.connect(**dsn) as conn:
        conn.autocommit = True
        with conn.cursor() as cursor:
            sql = MIGRATION.read_text()
            assert CLAIM_HASH in sql
            cursor.execute("SELECT owner.rolname,p.proacl,p.prosecdef,p.provolatile,p.proconfig "
                           "FROM pg_proc p JOIN pg_namespace n ON n.oid=p.pronamespace "
                           "JOIN pg_roles owner ON owner.oid=p.proowner "
                           "WHERE n.nspname='public' AND p.proname='lab_arena_claim_assignment' "
                           "AND p.pronargs=9")
            security_before = cursor.fetchone()
            local_hash = _hash(cursor,
                'public.lab_arena_claim_assignment(text,text,integer,integer,text[],text,text,text,integer)')
            live_path = os.environ.get(LIVE_PREIMAGE_ENV)
            if live_path:
                live = json.loads(Path(live_path).read_text())
                assert live['sha256'] == CLAIM_HASH
                cursor.execute(live['definition'])
                assert _hash(cursor,
                    'public.lab_arena_claim_assignment(text,text,integer,integer,text[],text,text,text,integer)') == CLAIM_HASH
                conn.commit()
            else:
                sql = sql.replace(CLAIM_HASH, local_hash)
            cursor.execute('BEGIN')
            with pytest.raises(psycopg.Error, match='preimage differs'):
                cursor.execute(sql.replace(CLAIM_HASH if live_path else local_hash,
                                           '0' * 64, 1))
            cursor.execute('ROLLBACK')
            assert _hash(cursor,
                'public.lab_arena_claim_assignment(text,text,integer,integer,text[],text,text,text,integer)') == (CLAIM_HASH if live_path else local_hash)
            cursor.execute(sql)
            cursor.execute(sql)
            cursor.execute("SELECT owner.rolname,p.proacl,p.prosecdef,p.provolatile,p.proconfig "
                           "FROM pg_proc p JOIN pg_namespace n ON n.oid=p.pronamespace "
                           "JOIN pg_roles owner ON owner.oid=p.proowner "
                           "WHERE n.nspname='public' AND p.proname='lab_arena_claim_assignment' "
                           "AND p.pronargs=9")
            assert cursor.fetchone() == security_before
            yield sql


def _leased_prior(connection, *, kind='score'):
    prior._seed(connection, {'terminal_status':'judge_error'}, 'judge_error', kind=kind)
    with connection.cursor() as cursor:
        cursor.execute('SET session_replication_role=replica')
        cursor.execute("DELETE FROM public.lab_arena_runs WHERE run_id=%s", (prior.ASSIGNMENT+':2',))
        cursor.execute("UPDATE public.lab_arena_runs SET status='leased',terminal_cause=NULL,"
                       "result_doc=NULL,lease_token_hash=%s,lease_expires_at=now()-interval '1 second' "
                       "WHERE run_id=%s", ('sha256:'+'a'*64, prior.ASSIGNMENT+':1'))
        cursor.execute('SET session_replication_role=origin')
    connection.commit()


def _cooldown_seed(connection, *, failure_positions=(0, 1, 2), fresh_kind='score',
                   age_seconds=1, generation=1, with_ledger=False,
                   stage=1, round_status='stage1_scoring'):
    prior._seed(connection, {'terminal_status':'lease_expired'}, 'lease_expired', kind='score')
    with connection.cursor() as cursor:
        cursor.execute('SET session_replication_role=replica')
        cursor.execute('DELETE FROM public.lab_arena_runs WHERE attempt=2')
        cursor.execute('UPDATE public.lab_arena_rounds SET status=%s WHERE round_id=%s',
                       (round_status, prior.ROUND))
        cursor.execute('DELETE FROM public.lab_arena_runs WHERE attempt=1')
        for index, position in enumerate(failure_positions):
            attempt = 1 + sum(p == position for p in failure_positions[:index])
            assignment = f'{prior.ROUND}:{prior.SUBMISSION}:{stage}:{position}'
            cursor.execute(
                "INSERT INTO public.lab_arena_runs("
                "run_id,assignment_id,round_id,submission_id,miner_hotkey,stage,"
                "icp_position,attempt,kind,status,lease_generation,stage_generation,"
                "runner_hotkey,terminal_cause,lease_expires_at) VALUES "
                "(%s,%s,%s,%s,%s,%s,%s,%s,'score','failed',1,%s,%s,"
                "'lease_expired',now()-(%s*interval '1 second'))",
                (f'{assignment}:{attempt}', assignment, prior.ROUND, prior.SUBMISSION,
                 prior.MINER, stage, position, attempt, generation,
                 prior.RUNNER_A, age_seconds))
        fresh_stage = 2 if round_status.startswith('stage2') else 1
        fresh_position = 19 if fresh_stage == 2 else 9
        fresh = f'{prior.ROUND}:{prior.SUBMISSION}:{fresh_stage}:{fresh_position}'
        cursor.execute(
            "INSERT INTO public.lab_arena_runs("
            "run_id,assignment_id,round_id,submission_id,miner_hotkey,stage,"
            "icp_position,attempt,kind,status,lease_generation,stage_generation) VALUES "
            "(%s,%s,%s,%s,%s,%s,%s,1,%s,'pending',0,1)",
            (fresh+':1', fresh, prior.ROUND, prior.SUBMISSION, prior.MINER,
             fresh_stage, fresh_position, fresh_kind))
        if with_ledger:
            first = f'{prior.ROUND}:{prior.SUBMISSION}:{stage}:{failure_positions[0]}:1'
            cursor.execute(
                "INSERT INTO public.lab_arena_ledger("
                "entry_kind,miner_hotkey,round_id,submission_id,run_id,stage,"
                "call_identity,provider,operation_id,funding_source,amount_microusd) "
                "VALUES ('reservation',%s,%s,%s,%s,%s,%s,'deepline',"
                "'deepline.execute','host',1)",
                (prior.MINER, prior.ROUND, prior.SUBMISSION, first, stage,
                 'sha256:'+'c'*64))
        cursor.execute('SET session_replication_role=origin')
    connection.commit()
    return fresh+':1'


@pytest.mark.parametrize('positions,blocked', [
    ((0,), False),
    ((0, 1), False),
    ((0, 1, 2), True),
    ((0, 0, 1), False),
], ids=['one_transient','two_transients','three_distinct','three_rows_two_assignments'])
def test_cooldown_threshold_distinct_and_healthy_alternate(database, migration_sql,
                                                            positions, blocked):
    psycopg, dsn = database
    with psycopg.connect(**dsn) as conn:
        fresh = _cooldown_seed(conn, failure_positions=positions)
        response = prior._claim(conn, prior.RUNNER_A, 'a')
        assert (response == {'status':'no_pending'}) is blocked
        if blocked:
            assert prior._claim(conn, prior.RUNNER_B, 'b')['run_id'] == fresh
        else:
            assert response['run_id'] == fresh


@pytest.mark.parametrize('change', [
    'old_generation', 'older_than_frozen_ttl', 'provider_ledger',
    'other_runner', 'old_stage', 'execute_phase',
], ids=['generation_reset','finite_ttl','provider_exemption','runner_scoped',
        'old_stage','execute_unchanged'])
def test_cooldown_scope_and_expiry(database, migration_sql, change):
    psycopg, dsn = database
    with psycopg.connect(**dsn) as conn:
        opts = {}
        if change == 'old_generation':
            opts['generation'] = 0
        elif change == 'older_than_frozen_ttl':
            opts['age_seconds'] = 421  # fixture freezes lease_ttl_seconds=420
        elif change == 'provider_ledger':
            opts['with_ledger'] = True
        elif change == 'old_stage':
            opts['stage'] = 1
            opts['round_status'] = 'stage2_scoring'
        elif change == 'execute_phase':
            opts['fresh_kind'] = 'execute'
        fresh = _cooldown_seed(conn, **opts)
        runner = prior.RUNNER_B if change == 'other_runner' else prior.RUNNER_A
        assert prior._claim(conn, runner, 'd')['run_id'] == fresh


def test_three_real_expiries_start_finite_cooldown(database, migration_sql):
    psycopg, dsn = database
    with psycopg.connect(**dsn) as conn:
        _cooldown_seed(conn)
        with conn.cursor() as cursor:
            cursor.execute('SET session_replication_role=replica')
            cursor.execute("UPDATE public.lab_arena_runs SET status='leased',"
                           "terminal_cause=NULL,lease_token_hash=%s,"
                           "lease_expires_at=now()-interval '1 second' "
                           "WHERE status='failed'", ('sha256:'+'e'*64,))
            cursor.execute('SET session_replication_role=origin')
        conn.commit()
        with conn.cursor() as cursor:
            cursor.execute('SELECT public.lab_arena_expire_leases(%s)', (prior.ROUND,))
            assert cursor.fetchone()[0]['retried'] == 3
        conn.commit()
        assert prior._claim(conn, prior.RUNNER_A, 'e') == {'status':'no_pending'}
        alternate = prior._claim(conn, prior.RUNNER_B, 'f')
        assert alternate['status'] == 'leased' and alternate['kind'] == 'score'


def test_expiry_retains_prior_runner_and_prevents_zero_call_score_reclaim(database, migration_sql):
    psycopg, dsn = database
    with psycopg.connect(**dsn) as conn:
        _leased_prior(conn)
        with conn.cursor() as cursor:
            cursor.execute('SELECT public.lab_arena_expire_leases(%s)', (prior.ROUND,))
            assert cursor.fetchone()[0]['retried'] == 1
            cursor.execute("SELECT previous_runner_hotkey,status,kind FROM public.lab_arena_runs "
                           "WHERE run_id=%s", (prior.ASSIGNMENT+':2',))
            assert cursor.fetchone() == (prior.RUNNER_A, 'pending', 'score')
        conn.commit()
        assert prior._claim(conn, prior.RUNNER_A, '1') == {'status':'no_pending'}
        claim = prior._claim(conn, prior.RUNNER_B, '2')
        assert claim['status']=='leased' and claim['run_id']==prior.ASSIGNMENT+':2'


@pytest.mark.parametrize('cause,with_ledger,expect_block', [
    ('lease_expired', False, True),
    ('lease_expired', True, False),
    ('judge_error', False, False),
], ids=['zero_call_expiry','provider_spend','ordinary_judge_failure'])
def test_only_exact_zero_call_expiry_blocks_same_runner(
    database, migration_sql, cause, with_ledger, expect_block
):
    psycopg, dsn = database
    with psycopg.connect(**dsn) as conn:
        prior._seed(conn, {'terminal_status':cause}, cause,
                    kind='score', prior_has_ledger=with_ledger)
        with conn.cursor() as cursor:
            cursor.execute('SET session_replication_role=replica')
            cursor.execute("UPDATE public.lab_arena_runs SET result_doc=NULL,"
                           "lease_expires_at=now()-interval '1 second' "
                           "WHERE run_id=%s", (prior.ASSIGNMENT+':1',))
            cursor.execute('SET session_replication_role=origin')
        conn.commit()
        response = prior._claim(conn, prior.RUNNER_A, '3')
        assert (response == {'status':'no_pending'}) is expect_block
        if expect_block:
            assert prior._claim(conn, prior.RUNNER_B, '4')['status']=='leased'
        else:
            assert response['run_id']==prior.ASSIGNMENT+':2'


@pytest.mark.parametrize('mutation', [
    "UPDATE public.lab_arena_runs SET result_doc='{}'::jsonb WHERE attempt=1",
    "UPDATE public.lab_arena_runs SET status='accepted',terminal_cause='accepted' WHERE attempt=1",
    "UPDATE public.lab_arena_runs SET lease_expires_at=now()+interval '1 hour' WHERE attempt=1",
    "UPDATE public.lab_arena_runs SET attempt=3 WHERE attempt=2",
], ids=['has_result','not_failed','not_expired','not_immediate'])
def test_unproved_predecessor_keeps_existing_lone_runner_fallback(
    database, migration_sql, mutation
):
    psycopg, dsn = database
    with psycopg.connect(**dsn) as conn:
        prior._seed(conn, {'terminal_status':'lease_expired'}, 'lease_expired', kind='score')
        with conn.cursor() as cursor:
            cursor.execute('SET session_replication_role=replica')
            cursor.execute("UPDATE public.lab_arena_runs SET result_doc=NULL,"
                           "lease_expires_at=now()-interval '1 second' WHERE attempt=1")
            cursor.execute(mutation)
            cursor.execute('SET session_replication_role=origin')
        conn.commit()
        assert prior._claim(conn, prior.RUNNER_A, '7')['status']=='leased'


def test_execute_expiry_keeps_existing_claim_fallback(database, migration_sql):
    psycopg, dsn = database
    with psycopg.connect(**dsn) as conn:
        _leased_prior(conn, kind='execute')
        with conn.cursor() as cursor:
            cursor.execute('SELECT public.lab_arena_expire_leases(%s)', (prior.ROUND,))
            assert cursor.fetchone()[0]['retried']==1
            cursor.execute('SELECT previous_runner_hotkey FROM public.lab_arena_runs '
                           'WHERE run_id=%s', (prior.ASSIGNMENT+':2',))
            assert cursor.fetchone()[0]==prior.RUNNER_A
        conn.commit()
        assert prior._claim(conn, prior.RUNNER_A, '5')['status']=='leased'
