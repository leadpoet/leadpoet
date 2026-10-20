"""Proved launcher losses cannot exhaust a miner's two actual executions."""
import json
from pathlib import Path

import pytest

from tests.lab_arena import signal_judgments503_postgres_test as current
from tests.lab_arena import infrastructure_retry_cost478_postgres_test as costs
from tests.lab_arena import host_interruption_retry474_postgres_test as execution
from tests.lab_arena import paid_host_quarantine480_postgres_test as quarantine
from tests.lab_arena.test_lab_arena_migration_postgres import hotkey

SQL = Path(__file__).parents[2] / 'scripts/504-lab-arena-zero-call-host-attempt-credit.sql'
SIGNATURE = 'public.lab_arena_execution_attempt_count(public.lab_arena_runs)'
RUNNERS = [hotkey('zero504-' + label) for label in ('bad-a', 'bad-b', 'healthy-c', 'healthy-d')]


def snapshot(cur):
    cur.execute("SELECT oid::regprocedure::text,pg_get_functiondef(oid),proowner,proacl::text,"
                "prosecdef,provolatile,proconfig FROM pg_proc WHERE pronamespace='public'::regnamespace "
                "AND proname LIKE 'lab_arena_%%' AND prokind='f' ORDER BY oid")
    functions = cur.fetchall()
    rows = []
    for table, key in (('lab_arena_rounds', 'round_id'), ('lab_arena_runs', 'run_id'),
                       ('lab_arena_ledger', 'entry_id'), ('lab_arena_trajectory_events', 'trajectory_id')):
        cur.execute(f'SELECT jsonb_agg(to_jsonb(t) ORDER BY {key}) FROM public.{table} t')
        rows.append(cur.fetchone()[0])
    return functions, rows


@pytest.fixture(scope='module')
def database():
    gen = current.database.__wrapped__()
    db = next(gen)
    try:
        with db[0].connect(**db[1]) as conn, conn.cursor() as cur:
            costs.seed(conn)
            accepted = execution.claim(conn)
            assert accepted['status'] == 'leased'
            assert execution.pay(conn, accepted, amount=123)['status'] == 'settled'
            assert execution.finish(conn, accepted, 'accepted', token='a')['status'] == 'accepted'
            before = snapshot(cur)
            text = SQL.read_text()
            for name, message in (('v_preimage', 'preimage differs'), ('v_postimage', 'postimage differs')):
                digest = text.split(name + " CONSTANT TEXT := '")[1].split("'")[0]
                with pytest.raises(db[0].Error, match=message):
                    cur.execute(text.replace(digest, '0' * 64, 1))
                conn.rollback()
                assert snapshot(cur) == before
            cur.execute(f'GRANT EXECUTE ON FUNCTION {SIGNATURE} TO anon')
            with pytest.raises(db[0].Error, match='security shape differs'):
                cur.execute(text)
            conn.rollback()
            assert snapshot(cur) == before
            cur.execute('CREATE ROLE zero504_migrator NOLOGIN NOSUPERUSER INHERIT')
            cur.execute('GRANT lab_arena_owner TO zero504_migrator')
            cur.execute('GRANT USAGE,CREATE ON SCHEMA public TO zero504_migrator')
            cur.execute('ALTER DEFAULT PRIVILEGES FOR ROLE zero504_migrator IN SCHEMA public '
                        'GRANT EXECUTE ON FUNCTIONS TO anon,authenticated,service_role')
            cur.execute('SET ROLE zero504_migrator')
            cur.execute(text)
            after = snapshot(cur)
            assert before[1] == after[1]
            assert len(before[0]) == len(after[0])
            changed = []
            for old, new in zip(before[0], after[0]):
                assert old[0] == new[0] and old[2:] == new[2:]
                if old[1] != new[1]:
                    changed.append(old[0])
            assert changed == ['lab_arena_execution_attempt_count(lab_arena_runs)']
            cur.execute(text)
            assert snapshot(cur) == after
            cur.execute('RESET ROLE')
        yield db
    finally:
        gen.close()


def seed(conn, *, enabled=True, date='2099-10-14'):
    cfg = costs.seed(conn, enabled=enabled)
    cfg['runner_hotkeys'] = RUNNERS
    with conn.cursor() as cur:
        cur.execute('SET LOCAL session_replication_role=replica')
        cur.execute('UPDATE public.lab_arena_rounds SET configuration_doc=%s::jsonb,evaluation_date=%s',
                    (json.dumps(cfg), date))
        cur.execute('SET LOCAL session_replication_role=origin')
    conn.commit()
    return cfg


def claim(conn, runner, token):
    return execution.rpc(conn, 'lab_arena_claim_assignment(%s,%s,20,20,ARRAY[]::text[],%s,%s,%s,4500)',
        (execution.ROUND, runner, token * 32, 'sha256:' + token * 64, 'sha256:' + token * 64))


def zero_loss(conn, runner, token, *, started=True):
    lease = claim(conn, runner, token)
    assert lease['status'] == 'leased', lease
    if started:
        assert execution.event(conn, lease, 'runtime.started',
            {'status': 'starting', 'lease_generation': lease['lease_generation']}, token=token)['status'] == 'accepted'
    # This is the exact October 20 startup failure shape, including exit 128.
    assert execution.event(conn, lease, 'runtime.error', {
        'status': 'abandoned', 'failure_stage': 'runtime', 'error_class': 'RuntimeHostError',
        'runtime_host_reason': 'sandbox_launch_failed', 'launch_exit_code': 128,
        'launch_timed_out': False}, token=token)['status'] == 'accepted'
    assert execution.expire(conn)['retried'] == 1
    rows = execution.projected(conn)
    assert rows[-2]['terminal_doc']['recovery_reason'] == 'authenticated_zero_call_runtime_host_error'
    return lease


def counts(conn):
    return [row['execution_attempt'] for row in execution.projected(conn)]


def test_two_bad_hosts_then_healthy_execution_preserves_two_model_attempts(database):
    with database[0].connect(**database[1]) as conn:
        seed(conn)
        zero_loss(conn, RUNNERS[0], 'a')
        zero_loss(conn, RUNNERS[1], 'b')
        assert counts(conn) == [0, 0, 1]
        third = claim(conn, RUNNERS[2], 'c')
        assert third['attempt'] == 3
        assert execution.pay(conn, third, token='c', call='e', amount=123)['status'] == 'settled'
        # A real model/provider failure consumes its first actual attempt.
        assert execution.finish(conn, third, token='c')['confirmation_attempt'] == 4
        fourth = claim(conn, RUNNERS[3], 'd')
        assert fourth['attempt'] == 4
        assert execution.finish(conn, fourth, 'accepted', token='d')['status'] == 'accepted'
        before = execution.projected(conn)
        assert execution.expire(conn)['retried'] == 0
        assert execution.projected(conn) == before
        assert counts(conn) == [0, 0, 1, 2]
        with conn.cursor() as cur:
            cur.execute("SELECT count(*),sum(amount_microusd) FROM public.lab_arena_ledger WHERE entry_kind='settlement'")
            assert cur.fetchone() == (1, 123)
            cur.execute("SELECT count(*) FROM public.lab_arena_ledger WHERE entry_kind='dispatch'")
            assert cur.fetchone()[0] == 1


def test_two_actual_failures_still_exhaust_after_host_losses(database):
    with database[0].connect(**database[1]) as conn:
        seed(conn)
        zero_loss(conn, RUNNERS[0], 'a')
        zero_loss(conn, RUNNERS[1], 'b')
        third = claim(conn, RUNNERS[2], 'c')
        assert execution.finish(conn, third, token='c')['confirmation_attempt'] == 4
        fourth = claim(conn, RUNNERS[3], 'd')
        assert 'confirmation_attempt' not in execution.finish(conn, fourth, token='d')
        assert execution.expire(conn)['retried'] == 0
        assert counts(conn) == [0, 0, 1, 2]


@pytest.mark.parametrize('mutation', [
    'no_receipt', 'no_error', 'setup_receipt', 'wrong_original_expiry', 'wrong_runner', 'wrong_assignment',
    'stale_occurred', 'late_created', 'wrong_class', 'setup_stage', 'provider_event',
    'finished', 'cleanup', 'ledger', 'result', 'output', 'retired', 'wrong_generation',
    'policy_disabled', 'pre_oct14',
])
def test_unproved_or_contradictory_zero_call_failure_is_counted(database, mutation):
    with database[0].connect(**database[1]) as conn:
        seed(conn)
        first = zero_loss(conn, RUNNERS[0], 'a')
        with conn.cursor() as cur:
            cur.execute('SET LOCAL session_replication_role=replica')
            run = first['run_id']
            if mutation == 'no_receipt':
                cur.execute('UPDATE public.lab_arena_runs SET terminal_doc=NULL WHERE run_id=%s', (run,))
            elif mutation == 'no_error':
                cur.execute("DELETE FROM public.lab_arena_trajectory_events WHERE run_id=%s AND event_kind='runtime.error'", (run,))
            elif mutation == 'setup_receipt':
                cur.execute("UPDATE public.lab_arena_runs SET terminal_doc=jsonb_set(terminal_doc,'{recovery_reason}',"
                            "'\"authenticated_zero_call_setup_host_error\"') WHERE run_id=%s", (run,))
            elif mutation == 'wrong_original_expiry':
                cur.execute("UPDATE public.lab_arena_runs SET terminal_doc=jsonb_set(terminal_doc,'{original_lease_expires_at}',"
                            "to_jsonb((clock_timestamp()+interval '2 hours')::text)) WHERE run_id=%s", (run,))
            elif mutation in ('wrong_runner', 'wrong_assignment'):
                column, value = ('runner_hotkey', RUNNERS[1]) if mutation == 'wrong_runner' else ('assignment_id', 'foreign-assignment')
                cur.execute(f"UPDATE public.lab_arena_trajectory_events SET {column}=%s WHERE run_id=%s AND event_kind='runtime.error'", (value, run))
            elif mutation in ('stale_occurred', 'late_created'):
                column, value = ('occurred_at', 'clock_timestamp()-interval \'2 hours\'') if mutation == 'stale_occurred' else ('created_at', 'clock_timestamp()+interval \'2 hours\'')
                cur.execute(f"UPDATE public.lab_arena_trajectory_events SET {column}={value} WHERE run_id=%s AND event_kind='runtime.error'", (run,))
            elif mutation in ('wrong_class', 'setup_stage'):
                key, value = ('error_class', 'AgentDependencyError') if mutation == 'wrong_class' else ('failure_stage', 'setup')
                cur.execute("UPDATE public.lab_arena_trajectory_events SET content=jsonb_set(content,%s,to_jsonb(%s::text)) "
                            "WHERE run_id=%s AND event_kind='runtime.error'", ([key], value, run))
            elif mutation in ('provider_event', 'finished', 'cleanup'):
                kind = {'provider_event': 'provider.request', 'finished': 'runtime.finished', 'cleanup': 'runtime.cleanup_error'}[mutation]
                cur.execute("UPDATE public.lab_arena_trajectory_events SET event_kind=%s WHERE run_id=%s AND event_kind='runtime.started'", (kind, run))
            elif mutation == 'ledger':
                cur.execute("INSERT INTO public.lab_arena_ledger(entry_kind,miner_hotkey,round_id,submission_id,run_id,stage,"
                            "call_identity,provider,operation_id,funding_source,amount_microusd) "
                            "SELECT 'reservation',miner_hotkey,round_id,submission_id,run_id,stage,'sha256:'||repeat('e',64),"
                            "'deepline','deepline.execute','miner_key',1 FROM public.lab_arena_runs WHERE run_id=%s", (run,))
            elif mutation == 'result':
                cur.execute("UPDATE public.lab_arena_runs SET result_doc='{}' WHERE run_id=%s", (run,))
            elif mutation == 'output':
                cur.execute("UPDATE public.lab_arena_runs SET output_ref='existing-output' WHERE run_id=%s", (run,))
            elif mutation == 'retired':
                cur.execute('UPDATE public.lab_arena_runs SET champion_restart_required=true WHERE run_id=%s', (run,))
            elif mutation == 'wrong_generation':
                cur.execute('UPDATE public.lab_arena_runs SET lease_generation=lease_generation+1 WHERE run_id=%s', (run,))
            elif mutation == 'policy_disabled':
                cur.execute("UPDATE public.lab_arena_rounds SET configuration_doc=configuration_doc-'infrastructure_retry_cost_policy'")
            elif mutation == 'pre_oct14':
                cur.execute("UPDATE public.lab_arena_rounds SET evaluation_date='2026-10-13'")
            cur.execute('SET LOCAL session_replication_role=origin')
        conn.commit()
        assert counts(conn) == [1, 2]


def test_existing_receipt_does_not_require_a_new_start_event(database):
    with database[0].connect(**database[1]) as conn:
        seed(conn)
        zero_loss(conn, RUNNERS[0], 'a', started=False)
        assert counts(conn) == [0, 1]


@pytest.mark.parametrize('cause', ['model_error', 'invalid_output', 'credential_error', 'budget_exhausted'])
def test_real_miner_failure_remains_terminal(database, cause):
    with database[0].connect(**database[1]) as conn:
        seed(conn)
        zero_loss(conn, RUNNERS[0], 'a')
        second = claim(conn, RUNNERS[1], 'b')
        assert 'confirmation_attempt' not in execution.finish(conn, second, cause, token='b')
        assert execution.expire(conn)['retried'] == 0
        assert counts(conn) == [0, 1]


@pytest.mark.parametrize('gate', ['cutoff', 'hold'])
def test_existing_cutoff_and_operator_hold_still_block_new_claims(database, gate):
    with database[0].connect(**database[1]) as conn:
        seed(conn)
        zero_loss(conn, RUNNERS[0], 'a')
        with conn.cursor() as cur:
            cur.execute('SET LOCAL session_replication_role=replica')
            if gate == 'cutoff':
                cur.execute("UPDATE public.lab_arena_rounds SET configuration_doc=jsonb_set(configuration_doc,"
                            "'{schedule,stage_2_close}',to_jsonb((clock_timestamp()-interval '1 second')::text))")
            else:
                cur.execute('UPDATE public.lab_arena_restart_claim_control SET operator_paused=true')
            cur.execute('SET LOCAL session_replication_role=origin')
        conn.commit()
        assert claim(conn, RUNNERS[1], 'b')['status'] != 'leased'


def test_existing_paid_and_zero_call_round_quarantine_still_blocks(database):
    with database[0].connect(**database[1]) as conn:
        seed(conn)  # Restore operator controls after the hold negative control.
    quarantine.test_three_paid_interruptions_block_runner_but_keep_healthy_claim(database, True, True)
    quarantine.test_existing_zero_call_quarantine_unchanged(database, True)


def test_paid_host_retry_and_actual_billing_are_unchanged(database):
    costs.test_host_loss_retains_actual_bill_and_releases_all_competition_cost(database, 238181)


@pytest.mark.parametrize('name', [
    'test_extra_attempt_does_not_reset_cumulative_provider_quota',
    'test_extra_attempt_enforces_exact_original_cost_boundary_before_dispatch',
    'test_host_credit_cannot_extend_frozen_execution_deadline',
    'test_budget_only_deepline_keeps_original_zero_quota_semantics',
    'test_missing_authenticated_start_does_not_create_host_credit',
])
def test_original_host_retry_controls_on_current_migrated_schema(database, name):
    getattr(execution, name)(database)


@pytest.mark.parametrize('name', [
    'test_cost_credit_does_not_extend_cumulative_provider_call_cap',
    'test_active_retry_still_stops_at_its_own_fixed_money_cap',
    'test_absent_opt_in_preserves_old_published_cost_policy',
])
def test_original_cost_controls_on_current_migrated_schema(database, name):
    getattr(costs, name)(database)
