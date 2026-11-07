"""Exact schedule-only641 preserves all accepted work and native recovery paths."""
from copy import deepcopy
import json
import re

import pytest

from tests.lab_arena import nov05_infrastructure_recovery633_postgres_test as prior

database = prior.database
ROUND = prior.ROUND
SQL = prior.source.SCRIPTS / '641-arena-2026-11-05-recovery-window-extension.sql'
scalar = prior.scalar


def render(p):
    text = SQL.read_text().replace('2026-11-06T', '2099-11-06T')
    m = json.loads(re.search(r'\$manifest641\$(.*?)\$manifest641\$', text, re.S)[1])
    with p.conn.cursor() as q:
        cfg = scalar(q, 'SELECT configuration_doc FROM public.lab_arena_rounds WHERE round_id=%s', (ROUND,))
        m['configuration_sha256'] = prior.source.hash_doc(q, cfg)
        new = deepcopy(cfg)
        new['schedule'].update(m['schedule'])
        m['new_configuration_sha256'] = prior.source.hash_doc(q, new)
    return re.sub(r'\$manifest641\$.*?\$manifest641\$', lambda _: '$manifest641$' + json.dumps(m) + '$manifest641$', text, flags=re.S)


@pytest.fixture
def prepared(database):
    p = prior.seed(database)
    with p.conn.cursor() as q:
        original_history = scalar(q, "SELECT pg_get_functiondef('public.lab_arena_scoring_policy_history_v1(text)'::regprocedure)")
    try:
        prior.apply(p)  # Existing633 is applied only in this disposable fixture.
        p.window_sql = render(p)
        yield p
    finally:
        p.conn.rollback()
        with p.conn.cursor() as q:
            q.execute(original_history)
        p.conn.commit()
        p.store.close()
        p.conn.close()


def apply(p, sql=None):
    prior.apply(p, sql or p.window_sql)


def snapshot(p):
    result = prior.snapshot(p)
    with p.conn.cursor() as q:
        result['submissions'] = scalar(q, 'SELECT jsonb_object_agg(submission_id,to_jsonb(a)) FROM public.lab_arena_submissions a')
        result['history'] = scalar(q, 'SELECT jsonb_agg(to_jsonb(a) ORDER BY generation_floor) FROM public.lab_arena_scoring_supersessions a')
        result['history_function'] = scalar(q, "SELECT jsonb_build_object('definition',pg_get_functiondef(p.oid),'metadata',to_jsonb(p)-'prosrc') FROM pg_proc p WHERE p.oid='public.lab_arena_scoring_policy_history_v1(text)'::regprocedure")
    return result


def test_exact_extension_preserves_work_and_native_claim_completion(prepared):
    p = prepared
    lease = prior.claim(p)
    assert lease['status'] == 'leased'
    # An active paid lease is valid;641 neither cancels nor resets it.
    prior.bill(p, lease)
    before = snapshot(p)
    apply(p)
    after = snapshot(p)
    expected = deepcopy(before)
    expected['rounds'][ROUND]['configuration_doc']['schedule'].update(
        stage_2_close='2099-11-06T06:30:00Z',
        final_scoring_close='2099-11-06T08:30:00Z',
        publication_deadline='2099-11-06T08:35:00Z')
    expected['rounds'][ROUND]['updated_at'] = after['rounds'][ROUND]['updated_at']
    expected['history'] = after['history']
    expected['history_function'] = after['history_function']
    assert after['history_function']['metadata'] == before['history_function']['metadata']
    assert after == expected
    assert len(after['history']) == len(before['history']) + 1
    receipt = after['history'][-1]
    assert receipt['old_round_doc'] == {k: v for k, v in before['rounds'][ROUND].items() if k != 'updated_at'}
    assert receipt['new_round_doc'] == {k: v for k, v in after['rounds'][ROUND].items() if k != 'updated_at'}
    assert prior.complete(p, lease)['status'] == 'accepted'
    progressed = snapshot(p)
    apply(p)
    assert snapshot(p) == progressed
    next_lease = prior.claim(p)
    assert next_lease['status'] == 'leased' and next_lease['run_id'] != lease['run_id']


@pytest.mark.parametrize('fault', ['configuration', 'phase', 'generation', 'published', 'hold', 'guard_disabled', 'clock', 'postimage'])
def test_unsafe_precondition_rolls_back_without_receipt(prepared, fault):
    p = prepared
    with p.conn.cursor() as q:
        q.execute('SET LOCAL session_replication_role=replica')
        if fault == 'configuration':
            q.execute("UPDATE public.lab_arena_rounds SET configuration_doc=jsonb_set(configuration_doc,'{max_attempts_per_assignment}','3') WHERE round_id=%s", (ROUND,))
        elif fault == 'phase':
            q.execute("UPDATE public.lab_arena_rounds SET status='stage2_judged' WHERE round_id=%s", (ROUND,))
        elif fault == 'generation':
            q.execute('UPDATE public.lab_arena_rounds SET scoring_generation_floor=1 WHERE round_id=%s', (ROUND,))
        elif fault == 'published':
            q.execute('UPDATE public.lab_arena_rounds SET published_at=now() WHERE round_id=%s', (ROUND,))
        elif fault == 'hold':
            q.execute('UPDATE public.lab_arena_restart_claim_control SET operator_paused=true WHERE singleton')
        elif fault == 'guard_disabled':
            q.execute('ALTER TABLE public.lab_arena_rounds DISABLE TRIGGER lab_arena_rounds_write_once')
        q.execute('SET LOCAL session_replication_role=origin')
    p.conn.commit()
    sql = p.window_sql
    if fault == 'clock':
        sql = sql.replace("clock_timestamp()>='2099-11-06T05:30:00Z'", "clock_timestamp()>='2000-11-06T05:30:00Z'")
    elif fault == 'postimage':
        m = re.search(r'\$manifest641\$(.*?)\$manifest641\$', sql, re.S)[1]
        altered = json.loads(m)
        altered['new_configuration_sha256'] = '0' * 64
        sql = sql.replace(m, json.dumps(altered))
    before = snapshot(p)
    try:
        with pytest.raises(p.pg.Error, match='Window641'):
            apply(p, sql)
        p.conn.rollback()
        assert snapshot(p) == before
    finally:
        if fault == 'guard_disabled':
            with p.conn.cursor() as q:
                q.execute('ALTER TABLE public.lab_arena_rounds ENABLE TRIGGER lab_arena_rounds_write_once')
            p.conn.commit()


def test_post_update_failure_rolls_back_receipt_and_configuration(prepared):
    p = prepared
    before = snapshot(p)
    sql = p.window_sql
    head, tail = sql.rsplit("clock_timestamp()>='2099-11-06T05:30:00Z'", 1)
    sql = head + "clock_timestamp()>='2000-11-06T05:30:00Z'" + tail
    with pytest.raises(p.pg.Error, match='Window641 exact readback differs'):
        apply(p, sql)
    p.conn.rollback()
    assert snapshot(p) == before


def test_changed_configuration_after_application_fails_replay(prepared):
    p = prepared
    apply(p)
    with p.conn.cursor() as q:
        q.execute('SET LOCAL session_replication_role=replica')
        q.execute("UPDATE public.lab_arena_rounds SET configuration_doc=jsonb_set(configuration_doc,'{max_attempts_per_assignment}','3') WHERE round_id=%s", (ROUND,))
        q.execute('SET LOCAL session_replication_role=origin')
    p.conn.commit()
    before = snapshot(p)
    with pytest.raises(p.pg.Error, match='Window641 replay differs'):
        apply(p)
    p.conn.rollback()
    assert snapshot(p) == before


@pytest.mark.parametrize('late635', [False, True])
def test_existing_history_reports_unchanged_floor_and_policy(prepared, late635):
    p = prepared
    if late635:
        text = (prior.source.SCRIPTS / '635-arena-2026-11-06-late-review-recovery.sql').read_text()
        block = 'DO $patch635_history$' + text.split('DO $patch635_history$')[1].split('$patch635_history$;')[0] + '$patch635_history$;'
        with p.conn.cursor() as q:
            q.execute(block)
        p.conn.commit()
    old = p.store.scoring_policy_history(ROUND)
    policy = p.store.get_round(ROUND)['configuration_doc']['scorer_policy']
    apply(p)
    history = p.store.scoring_policy_history(ROUND)
    assert history['revisions'][:-1] == old['revisions']
    revision = history['revisions'][-1]
    assert revision['previous_generation_floor'] == revision['generation_floor'] == 0
    assert revision['old_scorer_policy'] == revision['new_scorer_policy'] == policy
    assert revision['recovery']['operation'] == 'nov05_recovery_window641_v1'
    assert revision['receipt_generation'] > history['revisions'][-2]['receipt_generation']
    # This regression fails if641 adds a receipt without the existing mapping.
    before = snapshot(p)
    apply(p)
    assert snapshot(p) == before


def test_normal_early_execution_close_preserves_existing_judgments(prepared):
    p = prepared
    apply(p)
    old_judgments = {key: value for key, value in snapshot(p)['runs'].items() if value['kind'] == 'score'}
    for _ in range(50):
        lease = prior.claim(p)
        assert lease['status'] == 'leased' and lease['kind'] == 'execute'
        prior.bill(p, lease)
        assert prior.complete(p, lease)['status'] == 'accepted'
    assert p.service.close_stage(ROUND, 2)['status'] == 'ok'
    row = p.store.get_round(ROUND)
    assert row['status'] == 'stage2_closed'
    assert len(row['stage2_scoring_plan_doc']['work_items']) == 215
    assert {key: p.store.get_run(key) for key in old_judgments} == old_judgments
    progressed = snapshot(p)
    apply(p)
    assert snapshot(p) == progressed



def test_promotion_lock_timeout_precedes_history_or_data_changes(prepared):
    p = prepared
    before = snapshot(p)
    holder = p.pg.connect(**p.dsn)
    try:
        with holder.cursor() as q:
            q.execute("SELECT pg_advisory_xact_lock(hashtextextended('lab_arena.baseline_promotion',0))")
        with pytest.raises(p.pg.errors.LockNotAvailable):
            apply(p, p.window_sql.replace("lock_timeout='5s'", "lock_timeout='100ms'"))
        p.conn.rollback()
        assert snapshot(p) == before
    finally:
        holder.rollback()
        holder.close()
