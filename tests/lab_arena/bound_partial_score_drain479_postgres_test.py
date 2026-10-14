"""A bound failed score artifact is a durable restart completion receipt."""
import hashlib
import json
from pathlib import Path

import pytest

from tests.lab_arena import baseline_closed_settlement478_postgres_test as prior
from tests.lab_arena import general_judge_continuation472_postgres_test as judge
from tests.lab_arena import owner_stopped_score_drain454_postgres_test as stopped
from tests.lab_arena.host_interruption_migration_authority_postgres_test import snapshot


SQL = Path(__file__).parents[2] / 'scripts/479-lab-arena-bound-partial-score-drain-receipt.sql'
PREIMAGE = '5c1d6335fd6160b4240b0cd60b003f2892505850df6b510a4e275057ccf5980c'
POSTIMAGE = '65384dfe99fbbac1f484139cfcf2a213079601b8d3af458a7153ec0ef7b2438a'
SIGNATURE = 'public.lab_arena__restart_drain_state_v1()'
HELPER = 'public.lab_arena__legacy_partial_bound(public.lab_arena_runs,jsonb,text)'
database = prior.database


def _reset_control(conn):
    with conn.cursor() as cur:
        cur.execute("""UPDATE public.lab_arena_restart_claim_control SET
          operator_paused=FALSE,pause_reason='',actor_ref='',guard_commitment='',
          owner_commitment='',guard_expires_at=NULL,candidate_commit='',
          restart_scope='',restart_phase='',captured_leases='[]' WHERE singleton""")
    conn.commit()


def _seed(conn):
    _reset_control(conn)
    assignment, source_id, cfg = judge.prepare(conn)
    with conn.cursor() as cur:
        cur.execute('SET LOCAL session_replication_role=replica')
        cur.execute('DELETE FROM public.lab_arena_runs WHERE run_id=%s',
                    (assignment + ':2',))
        cur.execute('SET LOCAL session_replication_role=origin')
    conn.commit()
    assert judge.previous.rpc(conn, 'lab_arena_expire_leases(%s)',
                              (judge.previous.ROUND,))['retried'] == 1
    with conn.cursor() as cur:
        cur.execute('SET LOCAL session_replication_role=replica')
        cur.execute("UPDATE public.lab_arena_runs SET created_at=clock_timestamp()-interval '20 minutes' WHERE run_id=%s", (assignment + ':2',))
        cur.execute('SET LOCAL session_replication_role=origin')
    conn.commit()
    lease = judge.previous.claim(conn)
    assert (lease['status'], lease['attempt']) == ('leased', 2), lease
    with conn.cursor() as cur:
        cur.execute('SELECT to_jsonb(r) FROM public.lab_arena_runs r WHERE run_id=%s',
                    (lease['run_id'],))
        run = cur.fetchone()[0]
        cur.execute('SELECT evaluation_date,configuration_doc FROM public.lab_arena_rounds WHERE round_id=%s',
                    (judge.previous.ROUND,))
        date, config = cur.fetchone()
    output_ref = f"arena/{run['round_id']}/scores/items/{run['run_id']}.json"
    receipt = {
        'schema_version': 'leadpoet.lab_arena.legacy_partial.v1',
        **{key: run[key] for key in ('round_id', 'submission_id', 'miner_hotkey',
                                    'stage', 'icp_position', 'scored_run_id')},
        'source_score_run_id': run['run_id'],
        'source_runner_hotkey': run['runner_hotkey'],
        'source_claim_request_id': run['claim_request_id'],
        'source_claim_request_hash': run['claim_request_hash'],
        'source_lease_generation': run['lease_generation'],
        'runner_authority_exclusions': run['claim_response']['runner_authority_exclusions'],
        'source_output_ref': output_ref,
        'source_output_hash': 'sha256:' + 'b' * 64,
        'scoring_input_hash': 'sha256:' + 'c' * 64,
        'execution_output_hash': 'sha256:' + 'd' * 64,
        'source_completion_request_hash': 'sha256:' + 'e' * 64,
        'scorer_image_digest': config['scorer_image_digest'],
        'scorer_image_reference': config['scorer_image_reference'],
        'scorer_policy': config['scorer_policy'],
        'evaluation_date': str(date),
        'terminal_count': 1,
    }
    with conn.cursor() as cur:
        cur.execute('SELECT public.lab_arena__legacy_partial_bound(r,%s::jsonb,%s) FROM public.lab_arena_runs r WHERE run_id=%s',
                    (json.dumps(receipt), output_ref, lease['run_id']))
        assert cur.fetchone()[0] is True
    return lease, output_ref, receipt, source_id, cfg


def _complete(conn, lease, *, cause='judge_error', output_ref='', receipt=None):
    result = {'terminal_status': cause}
    if receipt is not None:
        result['legacy_partial_checkpoint'] = receipt
    return judge.previous.rpc(conn,
        'lab_arena_complete_attempt(%s,%s,%s::jsonb,%s,%s)',
        (lease['run_id'], judge.previous.TOKEN, json.dumps(result), cause, output_ref))


def _drain(conn):
    with conn.cursor() as cur:
        cur.execute('SELECT public.lab_arena__restart_drain_state_v1()')
        return cur.fetchone()[0]


@pytest.fixture(scope='module')
def migrated(database):
    prior.migrated.__wrapped__(database)
    psycopg, dsn = database
    with psycopg.connect(**dsn) as probe:
        lease, output_ref, receipt, _, _ = _seed(probe)
        generation = stopped._hold_capture(probe)
        assert _complete(probe, lease, output_ref=output_ref, receipt=receipt)['status'] == 'failed'
        before = _drain(probe)
        assert (before['captured_count'], before['lost_or_mutated_count'],
                before['preserved']) == (1, 1, False)
    with psycopg.connect(**dsn) as conn:
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute('CREATE ROLE arena_partial_drain_migrator NOLOGIN NOSUPERUSER INHERIT')
            cur.execute('GRANT lab_arena_owner TO arena_partial_drain_migrator')
            cur.execute('GRANT USAGE,CREATE ON SCHEMA public TO arena_partial_drain_migrator WITH GRANT OPTION')
            cur.execute('ALTER DEFAULT PRIVILEGES FOR ROLE arena_partial_drain_migrator IN SCHEMA public GRANT EXECUTE ON FUNCTIONS TO anon,authenticated,service_role')
            cur.execute("SELECT nspacl::text FROM pg_namespace WHERE nspname='public'")
            schema_acl = cur.fetchone()[0]
            before = snapshot(cur, (SIGNATURE, HELPER))
            assert hashlib.sha256(before[SIGNATURE][0].encode()).hexdigest() == PREIMAGE
            assert hashlib.sha256(before[HELPER][0].encode()).hexdigest() == (
                '5e25500320b1d60a8b26bccf5964ada24630dad639917d5cd72d77d58544b3df')
            cur.execute('SET ROLE arena_partial_drain_migrator')
            cur.execute("SELECT rolsuper,has_schema_privilege(current_user,'public','CREATE'),has_schema_privilege('lab_arena_owner','public','CREATE'),pg_has_role(current_user,'lab_arena_owner','MEMBER') FROM pg_roles WHERE rolname=current_user")
            assert cur.fetchone() == (False, True, False, True)
            cur.execute('COMMIT')
            with pytest.raises(psycopg.Error, match='preimage differs'):
                cur.execute(SQL.read_text().replace(PREIMAGE, '0' * 64, 1))
            cur.execute('ROLLBACK')
            assert snapshot(cur, (SIGNATURE, HELPER)) == before
            for _ in range(2):
                cur.execute(SQL.read_text())
                cur.execute("SELECT nspacl::text FROM pg_namespace WHERE nspname='public'")
                assert cur.fetchone()[0] == schema_acl
                cur.execute("SELECT has_schema_privilege('lab_arena_owner','public','CREATE'),has_schema_privilege('lab_arena_service','public','CREATE')")
                assert cur.fetchone() == (False, False)
                after = snapshot(cur, (SIGNATURE, HELPER))
                assert hashlib.sha256(after[SIGNATURE][0].encode()).hexdigest() == POSTIMAGE
                assert after[SIGNATURE][1:] == before[SIGNATURE][1:]
                assert after[HELPER] == before[HELPER]
            cur.execute('RESET ROLE')
    with psycopg.connect(**dsn) as probe:
        first = stopped.guard._quiescence(probe, generation=generation)
        again = stopped.guard._quiescence(probe, generation=generation)
        assert (first['reported_terminal_receipt_count'],
                first['lost_or_mutated_count'], first['preserved']) == (1, 0, True)
        assert again['outcome_commitment'] == first['outcome_commitment']
        authorized = stopped.guard._rpc(probe, 'lab_arena_authorize_restart_phase_v1',
                                      stopped.guard.GUARD, stopped.guard.OWNER,
                                      generation, 'gateway_destructive')
        assert authorized['restart_phase'] == 'gateway_destructive'
        ready = stopped.guard._rpc(probe, 'lab_arena_mark_restart_ready_v1',
                                 stopped.guard.GUARD, stopped.guard.OWNER,
                                 generation, 'gateway_ready')
        assert ready['restart_phase'] == 'gateway_ready'
        released = stopped.guard._rpc(probe, 'lab_arena_release_restart_guard_v1',
                                    stopped.guard.GUARD, stopped.guard.OWNER,
                                    generation, 'partial-score-drain-test')
        assert released['guard_present'] is False
    return True


def test_bound_failed_score_receipt_retains_partial_and_paid_calls(database, migrated):
    with database[0].connect(**database[1]) as conn:
        lease, output_ref, receipt, source_id, _ = _seed(conn)
        call = 'sha256:' + 'f' * 64
        token = judge.previous.TOKEN
        assert judge.previous.rpc(conn,
            "lab_arena_reserve_call(%s,%s,%s,'openrouter.chat','openrouter','host',100,'{}'::jsonb,4500)",
            (lease['run_id'], token, call))['status'] == 'reserved'
        assert judge.previous.rpc(conn, 'lab_arena_mark_dispatched(%s,%s,%s)',
                                  (lease['run_id'], token, call))['status'] == 'dispatched'
        assert judge.previous.rpc(conn, 'lab_arena_settle_call(%s,%s,%s,100,%s::jsonb,4500)',
            (lease['run_id'], token, call,
             json.dumps({'status': 200, 'call_succeeded': True})))['status'] == 'settled'
        generation = stopped._hold_capture(conn)
        assert _complete(conn, lease, output_ref=output_ref, receipt=receipt)['status'] == 'failed'
        with conn.cursor() as cur:
            cur.execute('SELECT result_doc,output_ref,company_judgment_refs FROM public.lab_arena_runs WHERE run_id=%s',
                        (lease['run_id'],))
            saved = cur.fetchone()
            cur.execute('SELECT entry_kind,amount_microusd,terminal_response FROM public.lab_arena_ledger WHERE run_id=%s ORDER BY entry_id',
                        (lease['run_id'],))
            costs = cur.fetchall()
        drain = stopped.guard._quiescence(conn, generation=generation)
        assert (drain['captured_count'], drain['reported_terminal_receipt_count'],
                drain['lost_or_mutated_count'], drain['preserved']) == (1, 1, 0, True)
        assert drain['pending_retry_count'] == 1
        with conn.cursor() as cur:
            cur.execute('SELECT result_doc,output_ref,company_judgment_refs FROM public.lab_arena_runs WHERE run_id=%s',
                        (lease['run_id'],))
            assert cur.fetchone() == saved
            cur.execute('SELECT entry_kind,amount_microusd,terminal_response FROM public.lab_arena_ledger WHERE run_id=%s ORDER BY entry_id',
                        (lease['run_id'],))
            assert cur.fetchall() == costs
            cur.execute('SELECT status,scored_run_id FROM public.lab_arena_runs WHERE run_id=%s',
                        (lease['assignment_id'] + ':3',))
            assert cur.fetchone() == ('pending', source_id)


@pytest.mark.parametrize('fault', [
    'missing_checkpoint', 'wrong_ref', 'wrong_generation',
    'open_dispatch', 'execute', 'prior_day',
])
def test_invalid_partial_receipt_remains_lost(database, migrated, fault):
    with database[0].connect(**database[1]) as conn:
        lease, output_ref, receipt, _, _ = _seed(conn)
        stopped._hold_capture(conn)
        assert _complete(conn, lease, output_ref=output_ref, receipt=receipt)['status'] == 'failed'
        with conn.cursor() as cur:
            cur.execute('SET LOCAL session_replication_role=replica')
            if fault == 'missing_checkpoint':
                cur.execute("UPDATE public.lab_arena_runs SET result_doc=result_doc-'legacy_partial_checkpoint' WHERE run_id=%s", (lease['run_id'],))
            elif fault == 'wrong_ref':
                cur.execute("UPDATE public.lab_arena_runs SET output_ref='arena/other.json' WHERE run_id=%s", (lease['run_id'],))
            elif fault == 'wrong_generation':
                cur.execute('UPDATE public.lab_arena_runs SET lease_generation=lease_generation+1 WHERE run_id=%s', (lease['run_id'],))
            elif fault == 'open_dispatch':
                cur.execute("""INSERT INTO public.lab_arena_ledger
                  (entry_kind,miner_hotkey,round_id,submission_id,run_id,stage,
                   call_identity,provider,operation_id,funding_source,amount_microusd)
                  VALUES ('dispatch',%s,%s,%s,%s,1,%s,'openrouter','openrouter.chat','host',100)""",
                  (judge.previous.MINER, judge.previous.ROUND, judge.previous.SUB,
                   lease['run_id'], 'sha256:' + 'e' * 64))
            elif fault == 'execute':
                cur.execute("""UPDATE public.lab_arena_runs SET kind='execute',
                  terminal_cause='provider_error',
                  result_doc=jsonb_set(result_doc,'{terminal_status}','\"provider_error\"'::jsonb)
                  WHERE run_id=%s""", (lease['run_id'],))
            elif fault == 'prior_day':
                cur.execute("UPDATE public.lab_arena_rounds SET evaluation_date='2026-10-13' WHERE round_id=%s", (judge.previous.ROUND,))
                cur.execute("""UPDATE public.lab_arena_runs SET result_doc=jsonb_set(
                  result_doc,'{legacy_partial_checkpoint,evaluation_date}',
                  '\"2026-10-13\"'::jsonb) WHERE run_id=%s""", (lease['run_id'],))
            cur.execute('SET LOCAL session_replication_role=origin')
        conn.commit()
        if fault == 'prior_day':
            with conn.cursor() as cur:
                cur.execute('SELECT public.lab_arena__legacy_partial_bound(r,r.result_doc->\'legacy_partial_checkpoint\',r.output_ref) FROM public.lab_arena_runs r WHERE run_id=%s',
                            (lease['run_id'],))
                assert cur.fetchone()[0] is True
        drain = _drain(conn)
        assert (drain['captured_count'], drain['lost_or_mutated_count'],
                drain['preserved']) == (1, 1, False), (fault, drain)


@pytest.mark.parametrize('cause,output_ref,expected', [
    ('judge_error', '', 'reported_terminal_receipt_count'),
    ('accepted', 'arena/accepted.json', 'accepted_receipt_count'),
])
def test_existing_failed_and_accepted_receipts(database, migrated, cause,
                                               output_ref, expected):
    with database[0].connect(**database[1]) as conn:
        lease, _, _, _, _ = _seed(conn)
        stopped._hold_capture(conn)
        assert _complete(conn, lease, cause=cause, output_ref=output_ref)['status'] == (
            'accepted' if cause == 'accepted' else 'failed')
        drain = _drain(conn)
        assert (drain[expected], drain['lost_or_mutated_count'],
                drain['preserved']) == (1, 0, True), drain
