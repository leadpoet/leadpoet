"""Canonical restart drains preserve already-authorized late baseline work."""
import hashlib
import json
from pathlib import Path
import uuid
from datetime import datetime, timezone

import pytest

from tests.lab_arena import baseline_infrastructure_continuation485_postgres_test as prior
from tests.lab_arena import restart_expired_zero_call394_postgres_test as guard

ROOT = Path(__file__).parents[2]
SQL = ROOT / 'scripts/487-lab-arena-baseline-drain-lineage.sql'
BASE = prior.base
PRIORITY = prior.priority
HELPER = 'public.lab_arena__baseline_drain_phase_v1(public.lab_arena_rounds,public.lab_arena_runs)'
EXPIRY = 'lab_arena_expire_leases(text)'
DRAIN = 'lab_arena__restart_drain_state_v1()'
QUIESCENCE = 'lab_arena_restart_quiescence_v1(text,text,bigint)'
PRE = ('c8e1723044ab74469c9ed8f808cfa50f608149cc1bc86d9c19612474f0393d72', 'db4e1a3b9d51299e00d2b2a350f47700da1a8e2fbf13795bfc9fb54f1bab65f1')
POST = ('35cbc97d9d8390f2d208e36617e7684cfcad4e1d1a8296ca4fbb461cc625f594', 'f00bdd46ba168268a5541fa3f47baa5edbb9aa0db0e41b03e6420259b0e2c469')


def capture(conn):
    with conn.cursor() as cur:
        cur.execute('SELECT guard_generation FROM public.lab_arena_restart_claim_control')
        generation = cur.fetchone()[0]
    result = guard._acquire(conn, generation)
    conn.commit()
    assert result['drain']['captured_count'] == 1
    return result['guard_generation']


def quiescence(conn, generation):
    result = guard._quiescence(conn, generation=generation)
    conn.commit()
    return result


def release(conn, generation, *, abort=False):
    if not abort:
        guard._rpc(conn, 'lab_arena_authorize_restart_phase_v1',
                   guard.GUARD, guard.OWNER, generation, 'gateway_destructive')
        guard._rpc(conn, 'lab_arena_mark_restart_ready_v1',
                   guard.GUARD, guard.OWNER, generation, 'gateway_ready')
    result = guard._rpc(conn, 'lab_arena_abort_restart_guard_v1' if abort else 'lab_arena_release_restart_guard_v1',
                        guard.GUARD, guard.OWNER, generation, 'fixture487-complete')
    conn.commit()
    return result


def seed(conn, phase='stage2_scoring'):
    assignment = prior.seed(conn, phase)
    with conn.cursor() as cur:
        cur.execute('SET LOCAL session_replication_role=replica')
        cur.execute('UPDATE public.lab_arena_rounds SET stage_generation=7 WHERE round_id=%s', (BASE.ROUND,))
    conn.commit()
    assert BASE.recover(conn)['queued'] == 1
    prior.cool(conn)
    lease = PRIORITY.claim(conn)
    assert (lease['attempt'], lease['stage'], lease['stage_generation'], lease['run_id']) == (4, 1, 7, assignment+'4')
    return lease


def audit(conn):
    with conn.cursor() as cur:
        result = []
        for table, order in [('lab_arena_rounds', 'round_id'), ('lab_arena_submissions', 'submission_id'),
                             ('lab_arena_runs', 'run_id'), ('lab_arena_ledger', 'entry_id'),
                             ('lab_arena_trajectory_events', 'event_id')]:
            cur.execute('SELECT to_jsonb(r) FROM public.'+table+' r ORDER BY '+order)
            result.append(cur.fetchall())
        return result


def call(conn, lease, *, settle=True):
    identity = 'sha256:'+hashlib.sha256((lease['run_id']+'487').encode()).hexdigest()
    assert BASE.rpc(conn, "lab_arena_reserve_call(%s,%s,%s,'openrouter.chat','openrouter','host',100,'{}'::jsonb,4500)",
                    (lease['run_id'], BASE.TOKEN, identity))['status'] == 'reserved'
    assert BASE.rpc(conn, 'lab_arena_mark_dispatched(%s,%s,%s)',
                    (lease['run_id'], BASE.TOKEN, identity))['status'] == 'dispatched'
    if settle:
        assert BASE.rpc(conn, 'lab_arena_settle_call(%s,%s,%s,100,%s::jsonb,4500)',
                        (lease['run_id'], BASE.TOKEN, identity,
                         json.dumps({'status': 200, 'call_succeeded': True})))['status'] == 'settled'
    return identity


def host_error(conn, lease, *, cleanup=False, wrong_runner=False):
    event = dict(event_id=str(uuid.uuid4()), kind='runtime.cleanup_error' if cleanup else 'runtime.error',
                 occurred_at=datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
                 content=dict(status='abandoned', failure_stage='runtime', error_class='RuntimeHostError',
                              runtime_host_reason='sandbox_launcher_signaled', launch_exit_code=-9,
                              launch_timed_out=False))
    started = dict(event_id=str(uuid.uuid4()), kind='runtime.started', occurred_at=event['occurred_at'],
                   content=dict(status='starting', lease_generation=lease['lease_generation']))
    assert BASE.rpc(conn, 'lab_arena_append_trajectory_events_v1(%s,%s,%s::jsonb)',
                    (lease['run_id'], BASE.TOKEN, json.dumps([started, event])))['status'] == 'accepted'
    if wrong_runner:
        with conn.cursor() as cur:
            cur.execute('SET LOCAL session_replication_role=replica')
            cur.execute("UPDATE public.lab_arena_trajectory_events SET runner_hotkey='wrong-runner' WHERE event_id=%s", (event['event_id'],))
        conn.commit()


def judge(conn, lease, attempt=1):
    assert prior.complete(conn, lease)['status'] == 'accepted'
    assert BASE.recover(conn, [BASE.scope_item(conn, lease['run_id'])])['queued'] == 1
    result = PRIORITY.claim(conn)
    for _ in range(1, attempt):
        assert prior.complete(conn, result, 'judge_error')['status'] == 'failed'
        prior.cool(conn)
        result = PRIORITY.claim(conn)
    assert result['attempt'] == attempt and result['scored_run_id'] == lease['run_id']
    return result


@pytest.fixture(scope='module')
def database():
    generator = prior.database.__wrapped__()
    db = next(generator)
    try:
        with db[0].connect(**db[1]) as conn, conn.cursor() as cur:
            cur.execute((ROOT / 'scripts/486-lab-arena-restart-baseline-continuation-phase.sql').read_text())
            conn.commit()
            # Prove both residual defects against the verbatim merged486.
            bad_judge = judge(conn, seed(conn), 3)
            call(conn, bad_judge)
            host_error(conn, bad_judge)
            generation = capture(conn)
            before = audit(conn)
            with pytest.raises(db[0].Error, match='captured write set differs') as error:
                quiescence(conn, generation)
            assert error.value.pgcode == '55000'
            conn.rollback()
            assert audit(conn) == before
            expired = seed(conn)
            call(conn, expired)
            host_error(conn, expired)
            generation = capture(conn)
            assert quiescence(conn, generation)['preserved']
            assert BASE.recover(conn)['queued'] == 1
            assert not quiescence(conn, generation)['preserved']
            lease = seed(conn)
            generation = capture(conn)
            before = audit(conn)
            state = prior.functions(cur)
            old = {r[0]: r[1] for r in state}
            assert tuple(hashlib.sha256(old[s].encode()).hexdigest() for s in (EXPIRY, DRAIN)) == PRE
            text = SQL.read_text()
            for original, changed, error in [(PRE[0], '0'*64, 'preimage differs'),
                                               (POST[1], '0'*64, 'postimage differs'),
                                               ('4acbf6eba8b497455270d1f7b1ea4e407e3dbf175d136005e38fe81af90a9428', '0'*64, 'authority differs')]:
                with pytest.raises(db[0].Error, match=error):
                    cur.execute(text.replace(original, changed, 1))
                conn.rollback()
                assert prior.functions(cur) == state
            cur.execute('GRANT EXECUTE ON FUNCTION public.lab_arena__restart_drain_state_v1() TO lab_arena_service')
            with pytest.raises(db[0].Error, match='security shape differs'):
                cur.execute(text)
            conn.rollback()
            assert prior.functions(cur) == state
            cur.execute('CREATE ROLE baseline487_migrator NOLOGIN NOSUPERUSER INHERIT')
            cur.execute('GRANT lab_arena_owner TO baseline487_migrator')
            cur.execute('GRANT USAGE,CREATE ON SCHEMA public TO baseline487_migrator WITH GRANT OPTION')
            cur.execute('SET ROLE baseline487_migrator')
            cur.execute('SELECT rolsuper FROM pg_roles WHERE rolname=current_user')
            assert cur.fetchone() == (False,)
            cur.execute(text)
            after = prior.functions(cur)
            cur.execute(text)
            assert prior.functions(cur) == after
            cur.execute('RESET ROLE')
            new = {r[0]: r[1] for r in after}
            assert tuple(hashlib.sha256(new[s].encode()).hexdigest() for s in (EXPIRY, DRAIN)) == POST
            assert {s for s in old if old[s] != new[s]} == {EXPIRY, DRAIN}
            # The merged486 natural-expiry fallback already accepts both score phases.
            assert old[QUIESCENCE] == new[QUIESCENCE]
            assert [(r[0], r[2:]) for r in state] == [(r[0], r[2:]) for r in after if r[0] in old]
            # The ordinary expiry/retry/accounting tail is byte-for-byte intact.
            tail = "  FOR v_run IN\n    SELECT * FROM public.lab_arena_runs\n    WHERE round_id = p_round_id AND status = 'leased'"
            assert old[EXPIRY].split(tail)[1] == new[EXPIRY].split(tail)[1]
            cur.execute('SELECT has_function_privilege(\'lab_arena_service\',%s,\'EXECUTE\'),'
                        'has_function_privilege(\'service_role\',%s,\'EXECUTE\'),'
                        'has_schema_privilege(\'lab_arena_owner\',\'public\',\'CREATE\')', (HELPER, HELPER))
            assert cur.fetchone() == (False, False, False)
            conn.commit()
            assert audit(conn) == before
            assert quiescence(conn, generation)['still_leased_count'] == 1
            assert prior.complete(conn, lease)['status'] == 'accepted'
            assert quiescence(conn, generation)['preserved']
            release(conn, generation)
        yield db
    finally:
        generator.close()


@pytest.mark.parametrize('phase', ['stage2', 'stage2_closed', 'stage2_scoring'])
def test_exact_healthy_attempt4_waits_then_completes_without_mutation(database, phase):
    with database[0].connect(**database[1]) as conn:
        lease = seed(conn, phase)
        call(conn, lease)
        generation = capture(conn)
        before = audit(conn)
        waiting = quiescence(conn, generation)
        assert (waiting['still_leased_count'], waiting['expired_receipt_count'], waiting['preserved']) == (1, 0, False)
        assert audit(conn) == before
        assert PRIORITY.claim(conn)['status'] == 'paused'
        assert prior.complete(conn, lease)['status'] == 'accepted'
        drained = quiescence(conn, generation)
        assert drained['accepted_receipt_count'] == 1 and drained['preserved']
        assert quiescence(conn, generation)['outcome_commitment'] == drained['outcome_commitment']
        release(conn, generation)


@pytest.mark.parametrize('attempt', [1, 3])
def test_execute_to_dependent_judge_capture_settled_host_and_receipt(database, attempt):
    with database[0].connect(**database[1]) as conn:
        lease = judge(conn, seed(conn), attempt)
        call(conn, lease)
        host_error(conn, lease)
        generation = capture(conn)
        before = audit(conn)
        drained = quiescence(conn, generation)
        assert drained['preserved'] and drained['expired_receipt_count'] == 1
        after = audit(conn)
        assert before[:2] == after[:2] and before[3:] == after[3:]
        assert quiescence(conn, generation)['outcome_commitment'] == drained['outcome_commitment']
        release(conn, generation)
        prior.cool(conn)
        retry = PRIORITY.claim(conn)
        assert retry['attempt'] == attempt+1 and retry['scored_run_id'] == lease['scored_run_id']
        assert prior.complete(conn, retry)['status'] == 'accepted'
        assert BASE.rpc(conn, 'lab_arena_close_scoring(%s,2::smallint)', (BASE.ROUND,))['status'] == 'closed'
        scores = [dict(run_id=lease['scored_run_id'], per_icp_score=0, qualification_doc={'companies': []})]
        assert BASE.rpc(conn, 'lab_arena_record_run_scores(%s,1::smallint,%s::jsonb)',
                        (BASE.ROUND, json.dumps(scores)))['recorded'] == 1
        assert prior.final_ready(conn)


def test_stopped_execute_receipt_survives_queued_successor_and_release(database):
    with database[0].connect(**database[1]) as conn:
        lease = seed(conn)
        call(conn, lease)
        host_error(conn, lease)
        generation = capture(conn)
        before = audit(conn)
        drained = quiescence(conn, generation)
        assert drained['preserved'] and drained['expired_receipt_count'] == 1
        assert BASE.recover(conn)['queued'] == 1
        assert quiescence(conn, generation)['preserved']
        assert PRIORITY.claim(conn)['status'] == 'paused'
        after = audit(conn)
        assert before[:2] == after[:2] and before[3:] == after[3:]
        release(conn, generation)
        prior.cool(conn)
        assert PRIORITY.claim(conn)['attempt'] == 5


@pytest.mark.parametrize('kind', ['execute', 'score'])
def test_zero_call_host_recovery_and_captured_natural_expiry(database, kind):
    with database[0].connect(**database[1]) as conn:
        lease = seed(conn)
        if kind == 'score':
            lease = judge(conn, lease, 3)
        host_error(conn, lease)
        before = audit(conn)
        # Canonical capture continues to hold zero-call early recovery.
        generation = capture(conn)
        assert quiescence(conn, generation)['still_leased_count'] == 1
        release(conn, generation, abort=True)
        assert BASE.rpc(conn, 'lab_arena_expire_leases(%s)', (BASE.ROUND,))['expired'] == 1
        after = audit(conn)
        assert before[:2] == after[:2] and before[3:] == after[3:]


@pytest.mark.parametrize('head', ['none', 'settlement', 'dispatch'])
def test_captured_natural_deadline_preserves_accounting_and_no_new_work_after_close(database, head):
    with database[0].connect(**database[1]) as conn:
        lease = seed(conn)
        if head != 'none':
            call(conn, lease, settle=head == 'settlement')
        with conn.cursor() as cur:
            cur.execute('SET LOCAL session_replication_role=replica')
            if head == 'dispatch':
                cur.execute('UPDATE public.lab_arena_ledger SET amount_microusd=100 WHERE run_id=%s', (lease['run_id'],))
            cur.execute("UPDATE public.lab_arena_runs SET lease_expires_at=clock_timestamp()-interval '1 second' WHERE run_id=%s", (lease['run_id'],))
            cur.execute("UPDATE public.lab_arena_rounds SET configuration_doc=jsonb_set(configuration_doc,'{schedule,final_scoring_close}',to_jsonb((clock_timestamp()-interval '1 second')::text))")
        conn.commit()
        generation = capture(conn)
        before = audit(conn)
        drained = quiescence(conn, generation)
        assert drained['preserved'] and drained['expired_receipt_count'] == 1
        assert BASE.recover(conn)['queued'] == 0
        after = audit(conn)
        assert before[:2] == after[:2] and before[4] == after[4]
        if head == 'dispatch':
            assert after[3][:-1] == before[3]
            assert after[3][-1][0]['entry_kind'] == 'uncertain'
            assert after[3][-1][0]['amount_microusd'] == 100
        else:
            assert before[3] == after[3]
        release(conn, generation)
        assert PRIORITY.claim(conn)['status'] == 'no_pending'


@pytest.mark.parametrize('fault', ['miner', 'old_baseline', 'parent', 'stage_generation', 'capture_generation',
                                 'uncaptured', 'foreign', 'result', 'output', 'wrong_source', 'judge_scope'])
def test_invalid_phase_scope_or_capture_cannot_reach_mutation(database, fault):
    with database[0].connect(**database[1]) as conn:
        lease = seed(conn)
        if fault in ('wrong_source', 'judge_scope'):
            lease = judge(conn, lease, 3)
        call(conn, lease)
        host_error(conn, lease)
        generation = capture(conn)
        with conn.cursor() as cur:
            cur.execute('SET LOCAL session_replication_role=replica')
            if fault == 'miner':
                cur.execute('UPDATE public.lab_arena_submissions SET is_king=false WHERE submission_id=%s', (BASE.SUB,))
            elif fault == 'old_baseline':
                cur.execute("UPDATE public.lab_arena_rounds SET evaluation_date='2026-10-13'")
            elif fault == 'parent':
                cur.execute("UPDATE public.lab_arena_runs SET terminal_cause='model_error' WHERE run_id=%s", (lease['run_id'][:-1]+'3',))
            elif fault == 'stage_generation':
                cur.execute('UPDATE public.lab_arena_runs SET stage_generation=6 WHERE run_id=%s', (lease['run_id'],))
            elif fault == 'capture_generation':
                cur.execute("UPDATE public.lab_arena_restart_claim_control SET captured_leases=jsonb_set(captured_leases,'{0,lease_generation}','999')")
            elif fault == 'uncaptured':
                cur.execute("UPDATE public.lab_arena_restart_claim_control SET captured_leases='[]'")
            elif fault == 'foreign':
                cur.execute("UPDATE public.lab_arena_runs SET status='leased',lease_expires_at=clock_timestamp()+interval '1 hour',lease_generation=1 WHERE kind='execute' AND stage=2 AND icp_position=0 AND submission_id=%s", (PRIORITY.MINERS[0],))
            elif fault == 'result':
                cur.execute("UPDATE public.lab_arena_runs SET result_doc='{}' WHERE run_id=%s", (lease['run_id'],))
            elif fault == 'output':
                cur.execute("UPDATE public.lab_arena_runs SET output_ref='retained-output' WHERE run_id=%s", (lease['run_id'],))
            elif fault == 'wrong_source':
                cur.execute("UPDATE public.lab_arena_runs SET scored_run_id='other-source' WHERE run_id=%s", (lease['run_id'],))
            elif fault == 'judge_scope':
                cur.execute("UPDATE public.lab_arena_runs SET judgment_scope_doc='{}' WHERE run_id=%s", (lease['run_id'],))
        conn.commit()
        before = audit(conn)
        if fault in ('capture_generation', 'uncaptured'):
            assert not quiescence(conn, generation)['preserved']
        else:
            with pytest.raises(database[0].Error, match='captured write set differs'):
                quiescence(conn, generation)
            conn.rollback()
        assert audit(conn) == before


@pytest.mark.parametrize('fault', ['owner', 'generation', 'expired_guard', 'cleanup', 'runner', 'dispatch', 'unbound'])
def test_owner_host_cleanup_and_provider_controls_hold_late_baseline(database, fault):
    with database[0].connect(**database[1]) as conn:
        lease = seed(conn)
        call(conn, lease, settle=fault != 'dispatch')
        host_error(conn, lease, wrong_runner=fault == 'runner')
        if fault == 'cleanup':
            host_error(conn, lease, cleanup=True)
        generation = capture(conn)
        with conn.cursor() as cur:
            cur.execute('SET LOCAL session_replication_role=replica')
            if fault == 'expired_guard':
                cur.execute("UPDATE public.lab_arena_restart_claim_control SET guard_expires_at=clock_timestamp()-interval '1 second'")
            elif fault == 'unbound':
                cur.execute("UPDATE public.lab_arena_ledger SET call_identity=NULL WHERE entry_kind='settlement'")
        conn.commit()
        before = audit(conn)
        if fault in ('owner', 'generation'):
            with pytest.raises(database[0].Error, match='owner_or_generation_differs'):
                guard._quiescence(conn, owner='lab_arena_restart_owner:'+'f'*64 if fault == 'owner' else guard.OWNER,
                                  generation=generation+1 if fault == 'generation' else generation)
            conn.rollback()
        else:
            drained = quiescence(conn, generation)
            assert not drained['preserved'] and drained['expired_receipt_count'] == 0
        assert audit(conn) == before


@pytest.mark.parametrize('fault', ['successor_scope', 'successor_parent', 'successor_generation', 'captured_generation',
                                 'claim_generation', 'host_proof', 'receipt_time'])
def test_later_witness_cannot_replace_expired_receipt_proof(database, fault):
    with database[0].connect(**database[1]) as conn:
        lease = seed(conn)
        call(conn, lease)
        host_error(conn, lease)
        generation = capture(conn)
        assert quiescence(conn, generation)['preserved']
        assert BASE.recover(conn)['queued'] == 1
        assert quiescence(conn, generation)['preserved']
        with conn.cursor() as cur:
            cur.execute('SET LOCAL session_replication_role=replica')
            if fault == 'successor_scope':
                cur.execute("UPDATE public.lab_arena_runs SET icp_position=1 WHERE run_id=%s", (lease['run_id'][:-1]+'5',))
            elif fault == 'successor_generation':
                cur.execute("UPDATE public.lab_arena_runs SET stage_generation=8 WHERE run_id=%s", (lease['run_id'][:-1]+'5',))
            elif fault == 'successor_parent':
                cur.execute("UPDATE public.lab_arena_runs SET terminal_cause='model_error' WHERE run_id=%s", (lease['run_id'][:-1]+'3',))
            elif fault == 'captured_generation':
                cur.execute("UPDATE public.lab_arena_restart_claim_control SET captured_leases=jsonb_set(captured_leases,'{0,lease_generation}','999')")
            elif fault == 'claim_generation':
                cur.execute("UPDATE public.lab_arena_runs SET claim_response=jsonb_set(claim_response,'{lease_generation}','999') WHERE run_id=%s", (lease['run_id'],))
            elif fault == 'host_proof':
                cur.execute("DELETE FROM public.lab_arena_trajectory_events WHERE event_kind='runtime.started'")
            elif fault == 'receipt_time':
                cur.execute("UPDATE public.lab_arena_runs SET terminal_doc=jsonb_set(terminal_doc,'{expired_at}',to_jsonb((clock_timestamp()+interval '1 day')::text)) WHERE run_id=%s", (lease['run_id'],))
        conn.commit()
        before = audit(conn)
        assert not quiescence(conn, generation)['preserved']
        assert audit(conn) == before


def test_cutoff_holds_queued_successor_and_retains_operator_pause(database):
    with database[0].connect(**database[1]) as conn:
        lease = seed(conn)
        call(conn, lease)
        host_error(conn, lease)
        with conn.cursor() as cur:
            cur.execute("UPDATE public.lab_arena_restart_claim_control SET operator_paused=true,pause_reason='operator487'")
        conn.commit()
        generation = capture(conn)
        assert quiescence(conn, generation)['preserved']
        assert BASE.recover(conn)['queued'] == 0
        with conn.cursor() as cur:
            cur.execute('UPDATE public.lab_arena_restart_claim_control SET operator_paused=false')
        conn.commit()
        assert BASE.recover(conn)['queued'] == 1
        with conn.cursor() as cur:
            cur.execute('UPDATE public.lab_arena_restart_claim_control SET operator_paused=true')
            cur.execute('SET LOCAL session_replication_role=replica')
            cur.execute("UPDATE public.lab_arena_rounds SET configuration_doc=jsonb_set(configuration_doc,'{schedule,final_scoring_close}',to_jsonb((clock_timestamp()-interval '1 second')::text))")
        conn.commit()
        assert quiescence(conn, generation)['preserved']
        assert release(conn, generation)['operator_paused']
        assert PRIORITY.claim(conn)['status'] == 'paused'
        with conn.cursor() as cur:
            cur.execute('UPDATE public.lab_arena_restart_claim_control SET operator_paused=false')
        conn.commit()
        prior.cool(conn)
        assert PRIORITY.claim(conn)['status'] == 'no_pending'


@pytest.mark.parametrize('fault', ['helper_acl', 'helper_body', 'dependency_acl'])
def test_migration_replay_rejects_authority_drift(database, fault):
    with database[0].connect(**database[1]) as conn, conn.cursor() as cur:
        before = prior.functions(cur)
        if fault == 'helper_acl':
            cur.execute('GRANT EXECUTE ON FUNCTION '+HELPER+' TO lab_arena_service')
        elif fault == 'helper_body':
            cur.execute('SELECT pg_get_functiondef(%s::regprocedure)', (HELPER,))
            cur.execute(cur.fetchone()[0].replace('-- Recovery admission', '-- Drifted recovery admission'))
        elif fault == 'dependency_acl':
            cur.execute('GRANT EXECUTE ON FUNCTION public.lab_arena__baseline_recovery_run_v1(public.lab_arena_runs) TO lab_arena_service')
        with pytest.raises(database[0].Error, match='helper differs|authority differs'):
            cur.execute(SQL.read_text())
        conn.rollback()
        assert prior.functions(cur) == before
