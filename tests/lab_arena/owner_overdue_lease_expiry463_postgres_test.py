"""Captured overdue work closes by ordinary expiry without losing liability."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import threading
import time
import json

import pytest
from tests.lab_arena import owner_stopped_execute_drain461_postgres_test as prior

ROOT = Path(__file__).parents[2]
SQL = ROOT / 'scripts/463-lab-arena-owner-overdue-lease-expiry.sql'
PREHASH = '6fb3d102cf53f32ec2cdc653d3a43af398897019a16db83885136defa96b17bb'
database = prior.database
paid = prior.paid
guard = prior.guard
fresh_control = prior.fresh_control


def _due(conn, lease, *, future=False):
    with conn.cursor() as cur:
        cur.execute('SET LOCAL session_replication_role=replica')
        cur.execute("UPDATE public.lab_arena_runs SET lease_expires_at=clock_timestamp()+%s::interval WHERE run_id=%s",
                    ('5 minutes' if future else '-1 second', lease['run_id']))
        cur.execute('SET LOCAL session_replication_role=origin')
    conn.commit()


def _seed(conn, head='dispatch', kind='execute'):
    lease = paid._seed(conn, kind=kind, abandoned=False, settled=False)
    if head == 'reservation':
        with conn.cursor() as cur:
            cur.execute('SET LOCAL session_replication_role=replica')
            cur.execute("DELETE FROM public.lab_arena_ledger WHERE run_id=%s AND entry_kind='dispatch'", (lease['run_id'],))
            cur.execute('SET LOCAL session_replication_role=origin')
        conn.commit()
    with conn.cursor() as cur:
        cur.execute('SET LOCAL session_replication_role=replica')
        cur.execute('UPDATE public.lab_arena_ledger SET amount_microusd=1000 WHERE run_id=%s', (lease['run_id'],))
        cur.execute('SET LOCAL session_replication_role=origin')
    conn.commit()
    _due(conn, lease)
    return lease


@pytest.fixture(scope='module')
def migrated(database):
    prior.migrated.__wrapped__(database)
    psycopg, dsn = database
    with psycopg.connect(**dsn) as conn:
        conn.autocommit = True
        prior._reset(conn)
        lease = _seed(conn)
        generation = prior.prior._hold_capture(conn)
        before = paid._audit(conn, lease)
        blocked = guard._quiescence(conn, generation=generation)
        assert blocked['still_leased_count'] == 1 and blocked['expired_receipt_count'] == 0
        assert paid._audit(conn, lease) == before  # Real 65f preimage deadlock.
        with conn.cursor() as cur:
            state = prior.prior._state(cur)
            cur.execute('BEGIN')
            with pytest.raises(psycopg.Error, match='preimage differs'):
                cur.execute(SQL.read_text().replace(PREHASH, '0' * 64, 1))
            cur.execute('ROLLBACK')
            assert prior.prior._state(cur) == state
            cur.execute(SQL.read_text())
            after = prior.prior._state(cur)
            assert [x[2:] for x in state] == [x[2:] for x in after]
            old, new = ({x[0]: x[1] for x in rows} for rows in (state, after))
            expiry = 'lab_arena_expire_leases(text)'
            tail = '  -- lab_arena_settled_execute_host_recovery_v1:'
            assert old[expiry].split(tail)[1] == new[expiry].split(tail)[1]
            assert old[expiry].split('    -- Preserve404:')[0] == new[expiry].split('    -- lab_arena_guard_owned_overdue_expiry_v1:')[0]
            assert old['lab_arena_restart_quiescence_v1(text,text,bigint)'] == new['lab_arena_restart_quiescence_v1(text,text,bigint)']
            cur.execute(SQL.read_text())
            assert prior.prior._state(cur) == after
        fixed = guard._quiescence(conn, generation=generation)
        assert fixed['preserved'] and fixed['expired_receipt_count'] == 1
    return True


@pytest.mark.parametrize('kind', ['execute', 'score'])
@pytest.mark.parametrize('head', ['reservation', 'dispatch'])
def test_expired_captured_lease_preserves_evidence_liability_and_dormant_retry(database, migrated, head, kind):
    psycopg, dsn = database
    with psycopg.connect(**dsn) as conn:
        lease = _seed(conn, head, kind)
        generation = prior.prior._hold_capture(conn)
        before, frozen = paid._audit(conn, lease), prior._round(conn)
        drained = guard._quiescence(conn, generation=generation)
        assert (drained['expired_receipt_count'], drained['lost_or_mutated_count'], drained['pending_retry_count'], drained['preserved']) == (1, 0, 1, True)
        after = paid._audit(conn, lease)
        assert after['lab_arena_trajectory_events'] == before['lab_arena_trajectory_events']
        assert after['lab_arena_ledger'][:-1] == before['lab_arena_ledger']
        entry = after['lab_arena_ledger'][-1]
        reservation = before['lab_arena_ledger'][0]
        assert entry['entry_kind'] == ('uncertain' if head == 'dispatch' else 'recovery')
        assert entry['amount_microusd'] == (reservation['amount_microusd'] if head == 'dispatch' else 0)
        assert entry['entry_doc']['reason'] == 'lease_expired'
        assert after['run'][0:2] == ('failed', 'lease_expired')
        assert after['run'][3:] == before['run'][3:]
        assert prior._round(conn) == frozen
        assert guard._quiescence(conn, generation=generation)['outcome_commitment'] == drained['outcome_commitment']
        assert paid.recovery._claim(conn, paid.recovery.prior.RUNNER_B, 'b')['status'] == 'paused'


@pytest.mark.parametrize('fault', ['owner', 'generation', 'uncaptured', 'capture_generation', 'stage', 'result', 'output', 'unidentified', 'unbound', 'future_unbound', 'future'])
def test_ownership_identity_accounting_and_actual_deadline_fences(database, migrated, fault):
    psycopg, dsn = database
    with psycopg.connect(**dsn) as conn:
        lease = _seed(conn)
        if fault in ('future', 'future_unbound'):
            _due(conn, lease, future=True)
        generation = prior.prior._hold_capture(conn)
        with conn.cursor() as cur:
            cur.execute('SET LOCAL session_replication_role=replica')
            if fault == 'uncaptured':
                cur.execute("UPDATE public.lab_arena_restart_claim_control SET captured_leases='[]'")
            elif fault == 'capture_generation':
                cur.execute("UPDATE public.lab_arena_restart_claim_control SET captured_leases=jsonb_set(captured_leases,'{0,lease_generation}','999')")
            elif fault == 'stage':
                cur.execute('UPDATE public.lab_arena_runs SET stage_generation=stage_generation+1 WHERE run_id=%s', (lease['run_id'],))
            elif fault in ('result', 'output'):
                cur.execute(('UPDATE public.lab_arena_runs SET result_doc=\'{}\' WHERE run_id=%s' if fault == 'result' else "UPDATE public.lab_arena_runs SET output_ref='arena/preserved.json' WHERE run_id=%s"), (lease['run_id'],))
            elif fault == 'unidentified':
                cur.execute('UPDATE public.lab_arena_ledger SET call_identity=NULL WHERE run_id=%s', (lease['run_id'],))
            elif fault in ('unbound', 'future_unbound'):
                cur.execute("UPDATE public.lab_arena_ledger SET operation_id='wrong' WHERE run_id=%s AND entry_kind='dispatch'", (lease['run_id'],))
            cur.execute('SET LOCAL session_replication_role=origin')
        conn.commit()
        before = paid._audit(conn, lease)
        if fault in ('owner', 'generation', 'stage', 'result', 'output'):
            with pytest.raises(psycopg.Error, match='owner_or_generation_differs|captured write set differs'):
                guard._quiescence(conn, owner='lab_arena_restart_owner:'+'f'*64 if fault == 'owner' else guard.OWNER,
                                  generation=generation+1 if fault == 'generation' else generation)
            conn.rollback()
        else:
            waiting = guard._quiescence(conn, generation=generation)
            assert not waiting['preserved'] and waiting['expired_receipt_count'] == 0
        assert paid._audit(conn, lease) == before


@pytest.mark.parametrize('kind', ['execute', 'score'])
def test_unexpired_paid_sibling_does_not_block_overdue_work_or_get_expired(database, migrated, kind):
    psycopg, dsn = database
    with psycopg.connect(**dsn) as conn:
        expired = _seed(conn, kind=kind)
        live = paid.recovery._claim(conn, paid.recovery.prior.RUNNER_A, 'b')
        assert live['status'] == 'leased'
        with conn.cursor() as cur:
            for kind in ('reservation', 'dispatch'):
                cur.execute("INSERT INTO public.lab_arena_ledger(entry_kind,miner_hotkey,round_id,submission_id,run_id,stage,call_identity,provider,operation_id,funding_source,amount_microusd) SELECT %s,miner_hotkey,round_id,submission_id,run_id,stage,%s,'deepline','exa.contents','host',1000 FROM public.lab_arena_runs WHERE run_id=%s", (kind, paid.SECOND_CALL, live['run_id']))
        conn.commit()
        generation = prior.prior._hold_capture(conn)
        before = paid._audit(conn, live)
        drain = guard._quiescence(conn, generation=generation)
        assert (drain['expired_receipt_count'], drain['still_leased_count'], drain['preserved']) == (1, 1, False)
        assert paid._audit(conn, live) == before
        with pytest.raises(psycopg.Error, match='drain_not_preserved'):
            guard._rpc(conn, 'lab_arena_authorize_restart_phase_v1', guard.GUARD, guard.OWNER, generation, 'gateway_destructive')
        conn.rollback()


@pytest.mark.parametrize('winner', ['renewal', 'completion', 'generation'])
def test_completion_and_renewal_lock_is_rechecked_before_expiry(database, migrated, winner):
    psycopg, dsn = database
    with psycopg.connect(**dsn) as conn:
        lease = _seed(conn)
        generation = prior.prior._hold_capture(conn)
        locked = psycopg.connect(**dsn)
        try:
            with locked.cursor() as cur:
                cur.execute('SELECT 1 FROM public.lab_arena_rounds WHERE round_id=%s FOR UPDATE', (paid.recovery.prior.ROUND,))
                cur.execute('SELECT 1 FROM public.lab_arena_runs WHERE run_id=%s FOR UPDATE', (lease['run_id'],))
            started = threading.Event()
            waiter_pid = []
            def poll():
                with psycopg.connect(**dsn) as racing:
                    with racing.cursor() as cur:
                        cur.execute("SELECT pg_backend_pid()")
                        waiter_pid.append(cur.fetchone()[0])
                    started.set()
                    return guard._quiescence(racing, generation=generation)
            with ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(poll)
                assert started.wait(5)
                deadline = time.monotonic() + 5
                while True:
                    with conn.cursor() as cur:
                        cur.execute('SELECT cardinality(pg_blocking_pids(%s))', (waiter_pid[0],))
                        blocked = cur.fetchone()[0] > 0
                    if blocked:
                        break
                    assert time.monotonic() < deadline, 'quiescence did not wait on the live lease lock'
                    time.sleep(0.01)
                with locked.cursor() as cur:
                    cur.execute('SET LOCAL session_replication_role=replica')
                    cur.execute("UPDATE public.lab_arena_runs SET lease_expires_at=clock_timestamp()+interval '5 minutes' WHERE run_id=%s", (lease['run_id'],))
                    cur.execute('SET LOCAL session_replication_role=origin')
                if winner == 'completion':
                    assert paid._settle(locked, lease, commit=False)['status'] == 'settled'
                    assert paid._rpc(locked, 'lab_arena_complete_attempt(%s,%s,%s::jsonb,%s,%s)',
                        (lease['run_id'], paid.TOKEN_HASH, json.dumps({'terminal_status':'accepted'}),
                         'accepted', 'arena/preserved/output.json'), commit=False)['status'] == 'accepted'
                elif winner == 'generation':
                    with locked.cursor() as cur:
                        cur.execute('SET LOCAL session_replication_role=replica')
                        cur.execute('UPDATE public.lab_arena_runs SET lease_generation=lease_generation+1 WHERE run_id=%s', (lease['run_id'],))
                        cur.execute('SET LOCAL session_replication_role=origin')
                locked.commit()
                if winner == 'generation':
                    with pytest.raises(psycopg.Error, match='captured write set differs'):
                        future.result(timeout=10)
                    assert paid._audit(conn, lease)['run'][0] == 'leased'
                    return
                drain = future.result(timeout=10)
            assert drain['expired_receipt_count'] == 0
            assert drain['still_leased_count'] == (1 if winner == 'renewal' else 0)
            assert paid._audit(conn, lease)['run'][0] == ('leased' if winner == 'renewal' else 'accepted')
        finally:
            locked.close()


def test_multiple_rounds_expire_only_due_captured_cohorts(database, migrated):
    psycopg, dsn = database
    with psycopg.connect(**dsn) as conn:
        original = _seed(conn)
        other_round = 'arena-2099-01-02-other463'
        other_run = original['run_id'] + '-other463'
        with conn.cursor() as cur:
            cur.execute('SET LOCAL session_replication_role=replica')
            cur.execute("SELECT attname FROM pg_attribute WHERE attrelid='public.lab_arena_rounds'::regclass AND attnum>0 AND NOT attisdropped AND attgenerated='' ORDER BY attnum")
            columns = ','.join('"'+row[0]+'"' for row in cur.fetchall())
            cur.execute('INSERT INTO public.lab_arena_rounds('+columns+') SELECT '+columns+" FROM jsonb_populate_record(NULL::public.lab_arena_rounds,(SELECT to_jsonb(r)||jsonb_build_object('round_id',%s,'reward_basis_hash',NULL,'effective_reward_epoch',NULL) FROM public.lab_arena_rounds r WHERE round_id=%s))",
                        (other_round, paid.recovery.prior.ROUND))
            cur.execute("INSERT INTO public.lab_arena_runs SELECT (jsonb_populate_record(NULL::public.lab_arena_runs,to_jsonb(r)||jsonb_build_object('run_id',%s,'assignment_id',%s,'round_id',%s,'claim_request_id',NULL))).* FROM public.lab_arena_runs r WHERE run_id=%s",
                        (other_run, other_run, other_round, original['run_id']))
            cur.execute("INSERT INTO public.lab_arena_ledger(entry_kind,miner_hotkey,round_id,submission_id,run_id,stage,call_identity,provider,operation_id,funding_source,amount_microusd) SELECT entry_kind,miner_hotkey,%s,submission_id,%s,stage,%s,provider,operation_id,funding_source,amount_microusd FROM public.lab_arena_ledger WHERE run_id=%s",
                        (other_round, other_run, 'sha256:'+'c'*64, original['run_id']))
            cur.execute('SET LOCAL session_replication_role=origin')
        conn.commit()
        generation = prior.prior._hold_capture(conn)
        drained = guard._quiescence(conn, generation=generation)
        assert (drained['captured_count'], drained['expired_receipt_count'], drained['lost_or_mutated_count'], drained['preserved']) == (2, 2, 0, True)
        with conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM public.lab_arena_ledger WHERE run_id IN (%s,%s) AND entry_kind='uncertain' AND amount_microusd=1000", (original['run_id'], other_run))
            assert cur.fetchone()[0] == 2


def test_expired_dispatch_can_reconcile_exact_future_cost_once_without_run_edits(database, migrated):
    psycopg, dsn = database
    with psycopg.connect(**dsn) as conn:
        lease = _seed(conn)
        request = 'ctx-tool-' + paid.CALL[7:39]
        fingerprint = 'sha256:' + 'c'*64
        with conn.cursor() as cur:
            cur.execute('SET LOCAL session_replication_role=replica')
            cur.execute("UPDATE public.lab_arena_ledger SET entry_doc=entry_doc||%s::jsonb WHERE run_id=%s AND entry_kind='reservation'", (json.dumps({'deepline_request_id':request,'tool':'exa_contents','credential_fingerprint':fingerprint}), lease['run_id']))
            cur.execute('SET LOCAL session_replication_role=origin')
        conn.commit()
        generation = prior.prior._hold_capture(conn)
        assert guard._quiescence(conn, generation=generation)['preserved']
        before = paid._audit(conn, lease)
        uncertainty = before['lab_arena_ledger'][-1]
        args = (paid.recovery.prior.ROUND, lease['run_id'], paid.CALL, uncertainty['entry_id'], request, 'exa_contents', fingerprint, 2000, '0.02')
        expression = 'lab_arena_reconcile_deepline_cost_v1(%s,%s,%s,%s,%s,%s,%s,%s,%s)'
        denied = args[:6] + ('sha256:'+'d'*64,) + args[7:]
        assert paid._rpc(conn, expression, denied)['status'] == 'stale'
        assert paid._audit(conn, lease) == before
        first = paid._rpc(conn, expression, args)
        assert (first['status'], first['idempotent']) == ('settled', False)
        replay = paid._rpc(conn, expression, args)
        assert (replay['status'], replay['idempotent']) == ('settled', True)
        after = paid._audit(conn, lease)
        assert after['run'] == before['run']
        assert after['lab_arena_trajectory_events'] == before['lab_arena_trajectory_events']
        assert after['lab_arena_ledger'][:-1] == before['lab_arena_ledger']
        assert after['lab_arena_ledger'][-1]['amount_microusd'] == 2000
        assert after['lab_arena_ledger'][-1]['entry_kind'] == 'settlement'


def test_execute_attempt_two_expiry_does_not_create_attempt_three(database, migrated):
    psycopg, dsn = database
    with psycopg.connect(**dsn) as conn:
        lease = _seed(conn)
        with conn.cursor() as cur:
            cur.execute('SET LOCAL session_replication_role=replica')
            cur.execute('UPDATE public.lab_arena_runs SET attempt=2 WHERE run_id=%s', (lease['run_id'],))
            cur.execute('SET LOCAL session_replication_role=origin')
        conn.commit()
        generation = prior.prior._hold_capture(conn)
        result = guard._quiescence(conn, generation=generation)
        assert result['preserved'] and result['expired_receipt_count'] == 1
        assert result['pending_retry_count'] == 0
        with conn.cursor() as cur:
            cur.execute('SELECT max(attempt) FROM public.lab_arena_runs WHERE assignment_id=%s', (lease['assignment_id'],))
            assert cur.fetchone()[0] == 2
