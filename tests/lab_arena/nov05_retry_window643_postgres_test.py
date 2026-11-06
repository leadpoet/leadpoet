"""Second schedule extension preserves work and ordinary post-expiry retries."""
from copy import deepcopy
import json
import re

import pytest

from tests.lab_arena import nov05_recovery_window641_postgres_test as window

prior = window.prior
database = window.database
ROUND = prior.ROUND
SQL = prior.source.SCRIPTS / '643-arena-2026-11-05-retry-window-extension.sql'
scalar = prior.scalar
snapshot = window.snapshot


def render(p):
    text = SQL.read_text().replace('2026-11-06T', '2099-11-06T')
    manifest = json.loads(re.search(r'\$manifest643\$(.*?)\$manifest643\$', text, re.S)[1])
    with p.conn.cursor() as q:
        cfg = scalar(q, 'SELECT configuration_doc FROM public.lab_arena_rounds WHERE round_id=%s', (ROUND,))
        manifest['configuration_sha256'] = prior.source.hash_doc(q, cfg)
        new = deepcopy(cfg)
        new['schedule'].update(manifest['schedule'])
        manifest['new_configuration_sha256'] = prior.source.hash_doc(q, new)
    return re.sub(r'\$manifest643\$.*?\$manifest643\$', lambda _: '$manifest643$' + json.dumps(manifest) + '$manifest643$', text, flags=re.S)


def apply(p, sql=None):
    prior.apply(p, sql or p.retry_window_sql)


@pytest.fixture
def prepared(database):
    p = prior.seed(database)
    with p.conn.cursor() as q:
        original_history = scalar(q, "SELECT pg_get_functiondef('public.lab_arena_scoring_policy_history_v1(text)'::regprocedure)")
        cfg = scalar(q, 'SELECT configuration_doc FROM public.lab_arena_rounds WHERE round_id=%s', (ROUND,))
        cfg.update(checkpoint_deadline_policy='atomic_checkpoint_60m_v1',
                   icp_wall_clock_seconds=3600, lease_ttl_seconds=4500)
        q.execute('SET LOCAL session_replication_role=replica')
        q.execute('UPDATE public.lab_arena_rounds SET configuration_doc=%s::jsonb WHERE round_id=%s', (json.dumps(cfg), ROUND))
        q.execute('SET LOCAL session_replication_role=origin')
    p.conn.commit()
    try:
        p.sql = prior.render(p)
        prior.apply(p)
        p.window_sql = window.render(p)
        window.apply(p)
        p.retry_window_sql = render(p)
        yield p
    finally:
        p.conn.rollback()
        with p.conn.cursor() as q:
            q.execute(original_history)
        p.conn.commit()
        p.store.close()
        p.conn.close()


@pytest.mark.parametrize('apply_at', [None, '2099-11-06T06:45:00Z'])
def test_exact_extension_preserves_paid_lease_all_work_and_two_typed_receipts(prepared, apply_at):
    p = prepared
    lease = prior.claim(p)
    prior.bill(p, lease)
    before = snapshot(p)
    sql = p.retry_window_sql
    if apply_at:
        sql = sql.replace('clock_timestamp()', "TIMESTAMPTZ '" + apply_at + "'")
    apply(p, sql)
    after = snapshot(p)
    expected = deepcopy(before)
    expected['rounds'][ROUND]['configuration_doc']['schedule'].update(
        stage_2_close='2099-11-06T07:30:00Z', final_scoring_close='2099-11-06T09:30:00Z',
        publication_deadline='2099-11-06T09:35:00Z')
    expected['rounds'][ROUND]['updated_at'] = after['rounds'][ROUND]['updated_at']
    expected['history'] = after['history']
    assert after == expected
    assert len(after['history']) == len(before['history']) + 1
    assert prior.complete(p, lease)['status'] == 'accepted'
    progressed = snapshot(p)
    apply(p)
    assert snapshot(p) == progressed
    history = p.store.scoring_policy_history(ROUND)
    windows = [r for r in history['revisions'] if (r.get('recovery') or {}).get('operation') == 'nov05_recovery_window641_v1']
    assert len(windows) == 2
    assert all(r['generation_floor'] == r['previous_generation_floor'] == 0
               and r['old_scorer_policy'] == r['new_scorer_policy'] for r in windows)
    assert windows[-1]['receipt_generation'] == after['history'][-1]['generation_floor']
    assert after['history'][-1]['request_id'] == 'nov05-recovery-window643-v1'
    assert after['history_function'] == before['history_function']


@pytest.mark.parametrize('fault', ['configuration', 'phase', 'generation', 'published', 'hold',
                                  'guard_disabled', 'clock', 'postimage', 'prior_receipt', 'plan'])
def test_unsafe_state_rolls_back_without_changes(prepared, fault):
    p = prepared
    with p.conn.cursor() as q:
        q.execute('SET LOCAL session_replication_role=replica')
        if fault == 'configuration':
            q.execute("UPDATE public.lab_arena_rounds SET configuration_doc=jsonb_set(configuration_doc,'{max_attempts_per_assignment}','3') WHERE round_id=%s", (ROUND,))
        elif fault == 'phase':
            q.execute("UPDATE public.lab_arena_rounds SET status='stage2_closed' WHERE round_id=%s", (ROUND,))
        elif fault == 'generation':
            q.execute('UPDATE public.lab_arena_rounds SET scoring_generation_floor=1 WHERE round_id=%s', (ROUND,))
        elif fault == 'published':
            q.execute('UPDATE public.lab_arena_rounds SET published_at=now() WHERE round_id=%s', (ROUND,))
        elif fault == 'hold':
            q.execute('UPDATE public.lab_arena_restart_claim_control SET operator_paused=true WHERE singleton')
        elif fault == 'guard_disabled':
            q.execute('ALTER TABLE public.lab_arena_rounds DISABLE TRIGGER lab_arena_rounds_write_once')
        elif fault == 'prior_receipt':
            q.execute("DELETE FROM public.lab_arena_scoring_supersessions WHERE request_id='nov05-recovery-window641-v1'")
        elif fault == 'plan':
            q.execute("UPDATE public.lab_arena_rounds SET stage2_scoring_plan_doc='{}' WHERE round_id=%s", (ROUND,))
        q.execute('SET LOCAL session_replication_role=origin')
    p.conn.commit()
    sql = p.retry_window_sql
    if fault == 'clock':
        sql = sql.replace("clock_timestamp()>='2099-11-06T07:19:25Z'", "clock_timestamp()>='2000-11-06T07:19:25Z'")
    elif fault == 'postimage':
        text = re.search(r'\$manifest643\$(.*?)\$manifest643\$', sql, re.S)[1]
        bad = json.loads(text)
        bad['new_configuration_sha256'] = '0' * 64
        sql = sql.replace(text, json.dumps(bad))
    before = snapshot(p)
    try:
        with pytest.raises(p.pg.Error, match='Window643'):
            apply(p, sql)
        p.conn.rollback()
        assert snapshot(p) == before
    finally:
        if fault == 'guard_disabled':
            with p.conn.cursor() as q:
                q.execute('ALTER TABLE public.lab_arena_rounds ENABLE TRIGGER lab_arena_rounds_write_once')
            p.conn.commit()


@pytest.mark.parametrize('extended', [False, True])
def test_native_expiry_after_old_cutoff_creates_second_attempt_only_with_extension(prepared, extended):
    p = prepared
    first = next(t['run_id'] for t in p.targets if t['mode'] == 'restore_unused' and t['run_id'].endswith(':1'))
    with p.conn.cursor() as q:
        q.execute('SET LOCAL session_replication_role=replica')
        q.execute("UPDATE public.lab_arena_runs SET created_at='2000-01-01' WHERE run_id=%s", (first,))
        q.execute('SET LOCAL session_replication_role=origin')
    p.conn.commit()
    for _ in range(50):
        lease = prior.claim(p)
        assert lease['status'] == 'leased'
        if lease['run_id'] == first:
            break
        assert prior.complete(p, lease)['status'] == 'accepted'
    assert lease['run_id'] == first and lease['attempt'] == 1
    prior.bill(p, lease)
    if extended:
        apply(p)
    with p.conn.cursor() as q:
        original = scalar(q, "SELECT pg_get_functiondef('public.lab_arena_expire_leases(text)'::regprocedure)")
        # Freeze only this disposable function's clock after old06:30 and before
        # new07:30. All actual expiry, retry and accounting logic stays native.
        frozen = re.sub(r'(?:pg_catalog\.)?clock_timestamp\(\)', "(TIMESTAMPTZ '2099-11-06T07:19:26Z')", original)
        q.execute(frozen)
    p.conn.commit()
    try:
        p.store.expire_leases(ROUND)
        with p.conn.cursor() as q:
            successor = scalar(q, 'SELECT (SELECT to_jsonb(a) FROM public.lab_arena_runs a WHERE run_id=%s)', (first.rsplit(':', 1)[0] + ':2',))
        assert (successor is not None) is extended
        if extended:
            assert successor['status'] == 'pending' and successor['attempt'] == 2
        assert p.store.get_run(first)['status'] == 'failed'
    finally:
        p.conn.rollback()
        with p.conn.cursor() as q:
            q.execute(original)
        p.conn.commit()


def test_early_completion_still_closes_opens_native_plan_and_preserves_existing_judgments(prepared):
    p = prepared
    apply(p)
    old = {k: v for k, v in snapshot(p)['runs'].items() if v['kind'] == 'score'}
    for _ in range(50):
        lease = prior.claim(p)
        assert lease['status'] == 'leased' and lease['kind'] == 'execute'
        prior.bill(p, lease)
        assert prior.complete(p, lease)['status'] == 'accepted'
    assert p.service.close_stage(ROUND, 2)['status'] == 'ok'
    row = p.store.get_round(ROUND)
    assert row['status'] == 'stage2_closed' and len(row['stage2_scoring_plan_doc']['work_items']) == 215
    assert p.service.open_scoring(ROUND, 2)['status'] == 'ok'
    assert {k: p.store.get_run(k) for k in old} == old
    progressed = snapshot(p)
    apply(p)
    assert snapshot(p) == progressed


def test_native_post_cutoff_drain_then_extension_preserves_paid_live_lease(prepared):
    p = prepared
    lease = prior.claim(p)
    prior.bill(p, lease)
    with p.conn.cursor() as q:
        q.execute('SET LOCAL session_replication_role=replica')
        q.execute("UPDATE public.lab_arena_runs SET lease_expires_at='2099-11-06T07:19:25Z' WHERE run_id=%s", (lease['run_id'],))
        q.execute('SET LOCAL session_replication_role=origin')
        original = scalar(q, "SELECT pg_get_functiondef('public.lab_arena_close_stage(text,smallint)'::regprocedure)")
        frozen = re.sub(r'(?:pg_catalog\.)?clock_timestamp\(\)', "(TIMESTAMPTZ '2099-11-06T06:45:00Z')", original)
        q.execute(frozen)
    p.conn.commit()
    before = snapshot(p)
    try:
        assert p.store.close_stage(ROUND, 2)['status'] == 'draining'
        assert snapshot(p) == before
        row = p.store.get_round(ROUND)
        assert (row['status'], row['status_generation'], row['stage_generation']) == ('stage2', 9, 7)
    finally:
        p.conn.rollback()
        with p.conn.cursor() as q:
            q.execute(original)
        p.conn.commit()
    apply(p, p.retry_window_sql.replace('clock_timestamp()', "TIMESTAMPTZ '2099-11-06T06:45:00Z'"))
    after = snapshot(p)
    assert after['runs'] == before['runs'] and after['ledger'] == before['ledger']
    row = p.store.get_round(ROUND)
    assert (row['status'], row['status_generation'], row['stage_generation']) == ('stage2', 9, 7)
    assert row['configuration_doc']['schedule']['stage_2_close'] == '2099-11-06T07:30:00Z'


def test_post_update_clock_failure_rolls_back_receipt_and_schedule(prepared):
    p = prepared
    before = snapshot(p)
    head, tail = p.retry_window_sql.rsplit("clock_timestamp()>='2099-11-06T07:19:25Z'", 1)
    sql = head + "clock_timestamp()>='2000-11-06T07:19:25Z'" + tail
    with pytest.raises(p.pg.Error, match='Window643 exact readback differs'):
        apply(p, sql)
    p.conn.rollback()
    assert snapshot(p) == before
