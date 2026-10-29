"""A charged final error does not strand baseline execute recovery."""
import hashlib
import json
from pathlib import Path

import pytest

from tests.lab_arena import settled_score_call_continuation477_postgres_test as prior
from tests.lab_arena import bounded_baseline_recovery470_postgres_test as baseline
from tests.lab_arena.host_interruption_migration_authority_postgres_test import snapshot


SQL = Path(__file__).parents[2] / 'scripts/478-lab-arena-baseline-closed-settlement-recovery.sql'
PREIMAGE = 'cc87055b6200b9560a488b1d73f8681ae6dd09bac2e98deb29690ffa8e5dc046'
POSTIMAGE = 'fc708bc7e5727bf5fd049ba8381d3a23be07c595a4c36de8ed9dfc0099fef147'
SIGNATURE = 'public.lab_arena_recover_baseline_v1(text,jsonb)'
database = prior.database


def _head(conn, run_id, *, kind='settlement', call='a', terminal=None, entry=None,
          amount=1582):
    with conn.cursor() as cur:
        cur.execute("""INSERT INTO public.lab_arena_ledger
          (entry_kind,miner_hotkey,round_id,submission_id,run_id,stage,
           call_identity,provider,operation_id,funding_source,amount_microusd,
           entry_doc,terminal_response)
          VALUES (%s,%s,%s,%s,%s,1,%s,'openrouter','openrouter.chat',
            'host',%s,%s::jsonb,%s::jsonb) RETURNING entry_id""", (
            kind, baseline.MINER, baseline.ROUND, baseline.SUB, run_id,
            'sha256:' + call * 64, amount, json.dumps(entry or {}),
            json.dumps(terminal) if terminal is not None else None))
        entry_id = cur.fetchone()[0]
    conn.commit()
    return entry_id


def _closed_reconciled(conn, run_id, *, call='a'):
    # Migration 225's final paid head follows a bound worker-reported uncertain
    # call. Migration 411 derives the failed outcome from that linked head.
    _head(conn, run_id, kind='reservation', call=call, amount=100)
    _head(conn, run_id, kind='dispatch', call=call, amount=100)
    prior_id = _head(conn, run_id, kind='uncertain', call=call, amount=100,
                     entry={'reason': 'worker_reported', 'call': {
                         'reason': 'missing_provider_cost', 'call_succeeded': False}})
    _head(conn, run_id, call=call, terminal={
        'status': 502, 'headers': {}, 'body_b64': 'eyJlcnJvciI6e319',
        'provider_cost': {'basis': 'openrouter_generation_cost', 'units': '0.001582'}},
        entry={'late_reconciliation': True,
               'openrouter_delayed_reconciliation': True,
               'reconciled_uncertainty_entry_id': prior_id,
               'reconciled_uncertainty_reason': 'worker_reported'})


def _state(conn):
    with conn.cursor() as cur:
        cur.execute('SELECT public.lab_arena__successful_icp_cost_state(%s,%s,0)',
                    (baseline.ROUND, baseline.SUB))
        return cur.fetchone()[0]


@pytest.fixture(scope='module')
def migrated(database):
    prior.migrated.__wrapped__(database)
    psycopg, dsn = database
    with psycopg.connect(**dsn) as probe:
        blocked, _ = baseline.seed(probe, kind='execute', cause='provider_error')
        _closed_reconciled(probe, blocked + ':1')
        state = _state(probe)
        assert (state['settled_microusd'], state['success_unresolved_calls'],
                state['successful_calls'], state['inflight_calls']) == (1582, 0, 0, 0)
        assert baseline.recover(probe) == {'status': 'ok', 'queued': 0}
    with psycopg.connect(**dsn) as conn:
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute('CREATE ROLE arena_baseline_closed_migrator NOLOGIN NOSUPERUSER INHERIT')
            cur.execute('GRANT lab_arena_owner TO arena_baseline_closed_migrator')
            cur.execute('GRANT USAGE,CREATE ON SCHEMA public TO arena_baseline_closed_migrator WITH GRANT OPTION')
            cur.execute('ALTER DEFAULT PRIVILEGES FOR ROLE arena_baseline_closed_migrator IN SCHEMA public GRANT EXECUTE ON FUNCTIONS TO anon,authenticated,service_role')
            cur.execute("SELECT nspacl::text FROM pg_namespace WHERE nspname='public'")
            schema_acl = cur.fetchone()[0]
            before = snapshot(cur, (SIGNATURE,))
            assert hashlib.sha256(before[SIGNATURE][0].encode()).hexdigest() == PREIMAGE
            cur.execute('SET ROLE arena_baseline_closed_migrator')
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
        assert baseline.recover(probe) == {'status': 'ok', 'queued': 1}
        assert baseline.recover(probe) == {'status': 'ok', 'queued': 0}
    return True


def test_charged_502_baseline_execute_rejoins_claim_and_accepts(database, migrated):
    with database[0].connect(**database[1]) as conn:
        assignment, cfg = baseline.seed(conn, kind='execute', cause='provider_error')
        _closed_reconciled(conn, assignment + ':1')
        assert _state(conn)['success_unresolved_calls'] == 0
        eligible = baseline.rpc(conn,
            'lab_arena_icp_cost_eligibility(%s,%s,0,1)',
            (baseline.ROUND, baseline.SUB))
        assert (eligible['eligible'], eligible['competition_sourcing_microusd'],
                eligible['execution']['settled_microusd']) == (True, 0, 1582)
        with conn.cursor() as cur:
            cur.execute('SELECT entry_id,amount_microusd,terminal_response FROM public.lab_arena_ledger WHERE run_id=%s', (assignment + ':1',))
            old_cost = cur.fetchall()
        assert baseline.recover(conn) == {'status': 'ok', 'queued': 1}
        assert baseline.recover(conn) == {'status': 'ok', 'queued': 0}
        lease = baseline.claim(conn)
        assert (lease['status'], lease['attempt'], lease['kind']) == ('leased', 3, 'execute'), lease
        token = baseline.TOKEN
        call = 'sha256:' + 'b' * 64
        assert baseline.rpc(conn,
            "lab_arena_reserve_call(%s,%s,%s,'openrouter.chat','openrouter','host',100,'{}'::jsonb,4500)",
            (lease['run_id'], token, call))['status'] == 'reserved'
        assert baseline.rpc(conn, 'lab_arena_mark_dispatched(%s,%s,%s)',
                            (lease['run_id'], token, call))['status'] == 'dispatched'
        assert baseline.rpc(conn, 'lab_arena_settle_call(%s,%s,%s,100,%s::jsonb,4500)',
            (lease['run_id'], token, call,
             json.dumps({'status': 200, 'call_succeeded': True})))['status'] == 'settled'
        done = baseline.rpc(conn, 'lab_arena_complete_attempt(%s,%s,%s::jsonb,%s,%s)',
            (lease['run_id'], token, json.dumps({'terminal_status': 'accepted'}),
             'accepted', 'recovered-output'))
        assert done['status'] == 'accepted', done
        with conn.cursor() as cur:
            cur.execute('SELECT entry_id,amount_microusd,terminal_response FROM public.lab_arena_ledger WHERE run_id=%s', (assignment + ':1',))
            assert cur.fetchall() == old_cost
            cur.execute('SELECT status,output_ref FROM public.lab_arena_runs WHERE run_id=%s', (lease['run_id'],))
            assert cur.fetchone() == ('accepted', 'recovered-output')
            cur.execute('SELECT count(*),max(attempt) FROM public.lab_arena_runs WHERE assignment_id=%s', (assignment,))
            assert cur.fetchone() == (3, 3)
            cur.execute('SELECT configuration_doc FROM public.lab_arena_rounds WHERE round_id=%s', (baseline.ROUND,))
            assert cur.fetchone()[0] == cfg


@pytest.mark.parametrize('kind,entry', [
    ('reservation', {}), ('dispatch', {}), ('uncertain', {}),
    ('uncertain', {'call': {'call_succeeded': True}}),
])
def test_other_open_or_unresolved_head_still_blocks(database, migrated, kind, entry):
    with database[0].connect(**database[1]) as conn:
        assignment, _ = baseline.seed(conn, kind='execute', cause='provider_error')
        _closed_reconciled(conn, assignment + ':1')
        _head(conn, assignment + ':2', kind=kind, call='c', entry=entry)
        assert baseline.recover(conn) == {'status': 'ok', 'queued': 0}


def test_flagless_settlement_without_linked_failed_call_stays_blocked(database, migrated):
    with database[0].connect(**database[1]) as conn:
        assignment, _ = baseline.seed(conn, kind='execute', cause='provider_error')
        _head(conn, assignment + ':1', terminal={'status': 502, 'headers': {},
              'body_b64': 'eyJlcnJvciI6e319'})
        assert _state(conn)['success_unresolved_calls'] == 1
        assert baseline.recover(conn) == {'status': 'ok', 'queued': 0}
        eligible = baseline.rpc(conn,
            'lab_arena_icp_cost_eligibility(%s,%s,0,1)',
            (baseline.ROUND, baseline.SUB))
        assert (eligible['eligible'], eligible['eligibility_reason']) == (
            False, 'provider_cost_uncertain')


def test_prior_day_execute_is_unchanged(database, migrated):
    with database[0].connect(**database[1]) as conn:
        assignment, _ = baseline.seed(conn, kind='execute', cause='provider_error')
        _closed_reconciled(conn, assignment + ':1')
        with conn.cursor() as cur:
            cur.execute('SET LOCAL session_replication_role=replica')
            cur.execute("UPDATE public.lab_arena_rounds SET evaluation_date='2026-10-13' WHERE round_id=%s", (baseline.ROUND,))
            cur.execute('SET LOCAL session_replication_role=origin')
        conn.commit()
        assert baseline.recover(conn) == {'status': 'stale', 'queued': 0}
        with conn.cursor() as cur:
            cur.execute('SELECT count(*) FROM public.lab_arena_runs WHERE assignment_id=%s AND attempt=3', (assignment,))
            assert cur.fetchone()[0] == 0


@pytest.fixture()
def service_database():
    from tests.lab_arena.lab_arena_pg_harness import current_service_database
    yield from current_service_database()


def test_existing_scoring_publication_and_reward_controls(database, migrated, service_database, tmp_path):
    # The historical fixture above still proves the exact 478 application.
    # Its current service/publication control needs the current read schema.
    prior.prior.test_existing_acceptance_publication_and_rewards_still_work(service_database, tmp_path)
