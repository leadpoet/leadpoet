"""A paid final provider error must not strand a stopped score assignment."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import uuid

import pytest

from tests.lab_arena import closed_failed_execute_drain476_postgres_test as prior
from tests.lab_arena import general_judge_continuation472_postgres_test as judge
from tests.lab_arena.host_interruption_migration_authority_postgres_test import snapshot


SQL = Path(__file__).parents[2] / 'scripts/477-lab-arena-settled-score-call-continuation.sql'
PREIMAGE = 'bc1e769a81daacdcb6fac4c6d1e094b0ad288a075ca5a1e1a65433fd38c3e550'
POSTIMAGE = 'aa8250cbfe7b62e0ba7802eb1b8c2a38c5b4a012a3e0044a04cb09692318bae7'
SIGNATURE = 'public.lab_arena__judge_calls_clear_v1(text)'
database = prior.database
service_database = prior.service_database


@pytest.fixture(scope='module')
def migrated(database):
    prior.upgraded.__wrapped__(database)
    psycopg, dsn = database
    with psycopg.connect(**dsn) as conn:
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute('CREATE ROLE arena_settled_score_migrator NOLOGIN NOSUPERUSER INHERIT')
            cur.execute('GRANT lab_arena_owner TO arena_settled_score_migrator')
            cur.execute('GRANT USAGE,CREATE ON SCHEMA public TO arena_settled_score_migrator WITH GRANT OPTION')
            cur.execute('ALTER DEFAULT PRIVILEGES FOR ROLE arena_settled_score_migrator IN SCHEMA public GRANT EXECUTE ON FUNCTIONS TO anon,authenticated,service_role')
            cur.execute("SELECT nspacl::text FROM pg_namespace WHERE nspname='public'")
            schema_acl = cur.fetchone()[0]
            before = snapshot(cur, (SIGNATURE,))
            assert hashlib.sha256(before[SIGNATURE][0].encode()).hexdigest() == PREIMAGE
            with psycopg.connect(**dsn) as probe:
                blocked_assignment, _, _ = judge.prepare(probe)
                _ledger_head(probe, blocked_assignment + ':1',
                             terminal={'status': 502, 'headers': {}, 'body_b64': 'e30='},
                             amount=1582)
                assert _clear(probe, blocked_assignment) is False
            cur.execute('SET ROLE arena_settled_score_migrator')
            cur.execute("SELECT rolsuper,has_schema_privilege(current_user,'public','CREATE'),has_schema_privilege('lab_arena_owner','public','CREATE'),pg_has_role(current_user,'lab_arena_owner','MEMBER') FROM pg_roles WHERE rolname=current_user")
            assert cur.fetchone() == (False, True, False, True)
            cur.execute('COMMIT')
            with pytest.raises(psycopg.Error, match='preimage differs'):
                cur.execute(SQL.read_text().replace(PREIMAGE, '0' * 64, 1))
            cur.execute('ROLLBACK')
            assert snapshot(cur, (SIGNATURE,)) == before
            for _ in range(2):
                cur.execute(SQL.read_text())
                cur.execute("SELECT nspacl::text FROM pg_namespace WHERE nspname='public'")
                assert cur.fetchone()[0] == schema_acl
                cur.execute("SELECT has_schema_privilege('lab_arena_owner','public','CREATE'),has_schema_privilege('lab_arena_service','public','CREATE')")
                assert cur.fetchone() == (False, False)
                after = snapshot(cur, (SIGNATURE,))
                assert hashlib.sha256(after[SIGNATURE][0].encode()).hexdigest() == POSTIMAGE
                assert after[SIGNATURE][1:] == before[SIGNATURE][1:]
            cur.execute('RESET ROLE')
            with psycopg.connect(**dsn) as probe:
                assert _clear(probe, blocked_assignment) is True
    return True


def _ledger_head(conn, run_id, *, kind='settlement', call='a', terminal=None, entry=None,
                 amount=1582):
    with conn.cursor() as cur:
        cur.execute("""INSERT INTO public.lab_arena_ledger
          (entry_kind,miner_hotkey,round_id,submission_id,run_id,stage,
           call_identity,provider,operation_id,funding_source,amount_microusd,
           entry_doc,terminal_response)
          VALUES (%s,%s,%s,%s,%s,1,%s,'openrouter','openrouter.chat',
            'host',%s,%s::jsonb,%s::jsonb)""", (
            kind, judge.previous.MINER, judge.previous.ROUND, judge.previous.SUB,
            run_id, 'sha256:' + call * 64, amount,
            json.dumps(entry or {}), json.dumps(terminal) if terminal is not None else None))
    conn.commit()


def _clear(conn, assignment):
    with conn.cursor() as cur:
        cur.execute('SELECT public.lab_arena__judge_calls_clear_v1(%s)', (assignment,))
        return cur.fetchone()[0]


def _event(conn, lease, kind, content):
    event = [{'event_id': str(uuid.uuid4()), 'kind': kind,
              'occurred_at': datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
              'content': content}]
    return judge.previous.rpc(conn, 'lab_arena_append_trajectory_events_v1(%s,%s,%s::jsonb)',
                              (lease['run_id'], judge.previous.TOKEN, json.dumps(event)))


def test_signed_second_score_loss_with_paid_502_rejoins_judge_queue(database, migrated):
    with database[0].connect(**database[1]) as conn:
        assignment, source_id, cfg = judge.prepare(conn)
        # The first attempt has a final, charged HTTP 502 with no success flag.
        _ledger_head(conn, assignment + ':1', terminal={
            'status': 502, 'headers': {}, 'body_b64': 'eyJlcnJvciI6e319',
            'provider_cost': {'basis': 'provider_reported', 'units': 1}}, amount=1582)
        with conn.cursor() as cur:
            cur.execute('SET LOCAL session_replication_role=replica')
            cur.execute('DELETE FROM public.lab_arena_runs WHERE run_id=%s', (assignment + ':2',))
            cur.execute('SET LOCAL session_replication_role=origin')
        conn.commit()
        assert _clear(conn, assignment) is True
        assert judge.previous.rpc(conn, 'lab_arena_expire_leases(%s)',
                                  (judge.previous.ROUND,))['retried'] == 1
        with conn.cursor() as cur:
            cur.execute('SET LOCAL session_replication_role=replica')
            cur.execute("UPDATE public.lab_arena_runs SET created_at=clock_timestamp()-interval '20 minutes' WHERE run_id=%s", (assignment + ':2',))
            cur.execute('SET LOCAL session_replication_role=origin')
        conn.commit()
        lease = judge.previous.claim(conn)
        assert (lease['status'], lease['attempt'], lease['scored_run_id']) == (
            'leased', 2, source_id), lease
        assert _event(conn, lease, 'runtime.started', {
            'status': 'starting', 'runtime': 'runsc',
            'lease_generation': lease['lease_generation']})['status'] == 'accepted'
        call = 'sha256:' + 'b' * 64
        token = judge.previous.TOKEN
        assert judge.previous.rpc(conn,
            "lab_arena_reserve_call(%s,%s,%s,'openrouter.chat','openrouter','host',100,'{}'::jsonb,4500)",
            (lease['run_id'], token, call))['status'] == 'reserved'
        assert judge.previous.rpc(conn, 'lab_arena_mark_dispatched(%s,%s,%s)',
                                  (lease['run_id'], token, call))['status'] == 'dispatched'
        assert judge.previous.rpc(conn, 'lab_arena_settle_call(%s,%s,%s,1582,%s::jsonb,4500)',
            (lease['run_id'], token, call,
             json.dumps({'status': 502, 'headers': {}, 'body_b64': 'eyJlcnJvciI6e319',
                         'provider_cost': {'basis': 'provider_reported', 'units': 1}})))['status'] == 'settled'
        assert _clear(conn, assignment) is True
        assert _event(conn, lease, 'runtime.error', {
            'status': 'abandoned', 'failure_stage': 'runtime',
            'error_class': 'RuntimeHostError', 'runtime_host_reason': 'sandbox_launcher_signaled',
            'launch_exit_code': -2, 'launch_timed_out': False})['status'] == 'accepted'
        # Exercise the ordinary timeout path with the signed host event kept.
        with conn.cursor() as cur:
            cur.execute('SET LOCAL session_replication_role=replica')
            cur.execute("UPDATE public.lab_arena_runs SET lease_expires_at=clock_timestamp()-interval '1 second' WHERE run_id=%s", (lease['run_id'],))
            cur.execute('SET LOCAL session_replication_role=origin')
        conn.commit()
        before = _paid_heads(conn, assignment)
        assert len(before) == 2 and sum(row[3] for row in before) == 3164
        assert all('call_succeeded' not in row[4] for row in before)
        expired = judge.previous.rpc(conn, 'lab_arena_expire_leases(%s)',
                                     (judge.previous.ROUND,))
        assert (expired['expired'], expired['retried']) == (1, 1), expired
        assert _paid_heads(conn, assignment) == before
        with conn.cursor() as cur:
            cur.execute('SELECT status,terminal_cause FROM public.lab_arena_runs WHERE run_id=%s', (lease['run_id'],))
            status, cause = cur.fetchone()
            assert (status, cause) == ('failed', 'lease_expired')
        assert judge.previous.rpc(conn, 'lab_arena_expire_leases(%s)',
                                  (judge.previous.ROUND,))['retried'] == 0
        with conn.cursor() as cur:
            cur.execute('SELECT count(*),max(attempt) FROM public.lab_arena_runs WHERE assignment_id=%s', (assignment,))
            assert cur.fetchone() == (3, 3)
            cur.execute('SET LOCAL session_replication_role=replica')
            cur.execute("UPDATE public.lab_arena_runs SET created_at=clock_timestamp()-interval '20 minutes' WHERE run_id=%s", (assignment + ':3',))
            cur.execute('SET LOCAL session_replication_role=origin')
        conn.commit()
        third = judge.previous.claim(conn, suffix='c')
        assert (third['status'], third['attempt'], third['scored_run_id']) == (
            'leased', 3, source_id), third
        done = judge.previous.rpc(conn, 'lab_arena_complete_attempt(%s,%s,%s::jsonb,%s,%s)',
            (third['run_id'], token, json.dumps({'terminal_status': 'accepted'}),
             'accepted', 'valid-judge-output'))
        assert done['status'] == 'accepted', done
        assert _paid_heads(conn, assignment) == before
        scored = judge.previous.rpc(conn, 'lab_arena_record_run_scores(%s,1::smallint,%s::jsonb)',
            (judge.previous.ROUND, json.dumps([{'run_id': source_id, 'per_icp_score': 0,
              'qualification_doc': {'companies': []}}])))
        assert (scored['status'], scored['recorded']) == ('ok', 1), scored
        with conn.cursor() as cur:
            cur.execute('SELECT public.lab_arena__all_judgments_ready_v1(%s)', (judge.previous.ROUND,))
            assert cur.fetchone()[0] is True
            cur.execute('SELECT configuration_doc,stage1_scoring_plan_doc FROM public.lab_arena_rounds WHERE round_id=%s', (judge.previous.ROUND,))
            assert cur.fetchone() == (cfg, {'work_items': [{'scored_run_id': source_id,
                'submission_id': judge.previous.SUB, 'icp_position': 0,
                'output_ref': 'source-output'}]})


def _paid_heads(conn, assignment):
    with conn.cursor() as cur:
        cur.execute("""SELECT run_id,call_identity,entry_kind,amount_microusd,terminal_response
          FROM public.lab_arena_ledger WHERE run_id LIKE %s AND entry_kind='settlement'
          ORDER BY run_id,call_identity,entry_id""", (assignment + ':%',))
        return cur.fetchall()


@pytest.mark.parametrize('kind,entry', [
    ('reservation', {}), ('dispatch', {}), ('uncertain', {}),
    ('uncertain', {'call': {'call_succeeded': True}}),
])
def test_open_or_ambiguous_head_still_blocks_continuation(database, migrated, kind, entry):
    with database[0].connect(**database[1]) as conn:
        assignment, _, _ = judge.prepare(conn)
        _ledger_head(conn, assignment + ':1', terminal={'status': 502}, amount=1582)
        _ledger_head(conn, assignment + ':2', kind=kind, call='c', entry=entry,
                     terminal=None, amount=100)
        assert _clear(conn, assignment) is False
        assert judge.previous.rpc(conn, 'lab_arena_expire_leases(%s)',
                                  (judge.previous.ROUND,))['retried'] == 0


def test_prior_day_remains_outside_continuation(database, migrated):
    with database[0].connect(**database[1]) as conn:
        assignment, _, _ = judge.prepare(conn)
        _ledger_head(conn, assignment + ':1', terminal={'status': 502}, amount=1582)
        with conn.cursor() as cur:
            cur.execute('SET LOCAL session_replication_role=replica')
            cur.execute("UPDATE public.lab_arena_rounds SET evaluation_date='2026-10-13' WHERE round_id=%s", (judge.previous.ROUND,))
            cur.execute('SET LOCAL session_replication_role=origin')
        conn.commit()
        assert judge.previous.rpc(conn, 'lab_arena_expire_leases(%s)',
                                  (judge.previous.ROUND,))['retried'] == 0
        with conn.cursor() as cur:
            cur.execute('SELECT count(*) FROM public.lab_arena_runs WHERE assignment_id=%s AND attempt=3', (assignment,))
            assert cur.fetchone()[0] == 0


def test_existing_publication_and_reward_controls(service_database, tmp_path):
    prior.test_existing_acceptance_publication_and_rewards_still_work(service_database, tmp_path)
