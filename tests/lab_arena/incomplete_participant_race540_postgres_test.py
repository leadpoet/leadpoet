"""Real transaction races at the canonical publication540 lock boundary.

Only disposable fixture setup bypasses triggers. Competing publication and
completion calls execute the installed service RPCs, without provider traffic.
"""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
import time

import pytest

from lab_arena import contracts
from tests.lab_arena import incomplete_participant_publication540_postgres_test as isolation

# Reuse the exact migrated database and actual saved-output fixture.
database = isolation.database
completed = isolation.completed
TOKEN = '7' * 64


class _CapturedPublication(Exception):
    pass


def _omitted_patch(h):
    """Keep the real service's pre-race proof and immutable publication draft."""
    original = h.service.store.transition_round
    saved = {}
    def capture(rid, expected, next_status, patch):
        assert expected == 'scored' and next_status == 'published'
        saved.update(deepcopy(patch))
        raise _CapturedPublication()
    h.service.store.transition_round = capture
    try:
        with pytest.raises(_CapturedPublication):
            h.service.publish(h.round_id)
    finally:
        h.service.store.transition_round = original
    return saved


def _publish(database, rid, patch):
    try:
        with database[0].connect(**database[1]) as conn, conn.cursor() as cur:
            cur.execute("SET LOCAL statement_timeout='8s'")
            cur.execute('SET LOCAL ROLE lab_arena_service')
            cur.execute("SELECT public.lab_arena_transition_round(%s,'scored','published',%s::jsonb)", (rid, json.dumps(patch)))
            return cur.fetchone()[0]
    except database[0].Error as exc:
        return {'status': 'rejected', 'sqlstate': exc.pgcode, 'message': str(exc).splitlines()[0]}


def _wait_for_round_lock(database, rid, future):
    """Observe a real ungranted lock; timing alone does not prove serialization."""
    until = time.monotonic() + 5
    with database[0].connect(**database[1]) as conn, conn.cursor() as cur:
        while time.monotonic() < until:
            assert not future.done(), future.result()
            cur.execute("SELECT EXISTS(SELECT 1 FROM pg_stat_activity a JOIN pg_locks l ON l.pid=a.pid WHERE a.query LIKE 'SELECT public.lab_arena_transition_round(%' AND NOT l.granted)")
            if cur.fetchone()[0]:
                return
            time.sleep(.01)
    pytest.fail('publication did not reach its blocked database lock')


def _complete(cur, judge):
    cur.execute('SET LOCAL ROLE lab_arena_service')
    cur.execute("SELECT public.lab_arena_complete_attempt_v3(%s,%s,%s::jsonb,'accepted',%s,%s,'[]'::jsonb,%s)",
        (judge['run_id'], TOKEN, json.dumps(judge['result_doc']),
         judge['output_ref'], judge['_output_hash'], 'sha256:' + '9' * 64))
    result = cur.fetchone()[0]
    cur.execute('RESET ROLE')
    return result


def test_committing_valid_late_judgment_invalidates_precomputed_omission(database, completed):
    h = isolation.reset(database, completed)
    run = isolation.selected(completed, 'MissingJudge')
    judge = dict(next(r for r in completed[2] if r['scored_run_id'] == run['run_id']))
    judge['_output_hash'] = contracts.document_hash(json.loads(h.objects.get(judge['output_ref'])))
    assert judge['company_judgment_refs'] == []
    with database[0].connect(**database[1]) as conn, conn.cursor() as cur:
        isolation.gap(cur, run)
    patch = _omitted_patch(h)
    with database[0].connect(**database[1]) as completing, completing.cursor() as cur, ThreadPoolExecutor(max_workers=1) as pool:
        # Retain the real claim authority and saved output, but reopen only this
        # disposable attempt to exercise the real late completion RPC.
        cur.execute('SET LOCAL session_replication_role=replica')
        cur.execute("UPDATE public.lab_arena_runs SET status='leased',terminal_cause=NULL,lease_token_hash=%s,lease_expires_at=clock_timestamp()+interval '1 hour' WHERE run_id=%s", (TOKEN, judge['run_id']))
        cur.execute('SET LOCAL session_replication_role=origin')
        assert _complete(cur, judge)['status'] == 'accepted'
        waiting = pool.submit(_publish, database, h.round_id, patch)
        _wait_for_round_lock(database, h.round_id, waiting)
        completing.commit()
        result = waiting.result(timeout=8)
        assert result['status'] == 'rejected', result
        assert result['sqlstate'] == '22023', result
    assert h.status() == 'scored'
    # It is now complete-but-unmaterialized, never valid omission evidence.
    with database[0].connect(**database[1]) as conn, conn.cursor() as cur:
        assert isolation.reasons(cur, h.round_id)[run['submission_id']] is None
        cur.execute('SELECT status FROM public.lab_arena_runs WHERE run_id=%s', (judge['run_id'],))
        assert cur.fetchone() == ('accepted',)
    # A fresh canonical pass must materialize this real saved output rather
    # than lose the completed participant after rejecting the stale draft.
    assert h.service.publish(h.round_id)['status'] == 'ok'
    ranking = h.service.store.get_round(h.round_id)['publication_doc']['final_ranking']
    assert next(r for r in ranking if r['submission_id'] == run['submission_id'])['final_score'] == 0.0


@pytest.mark.parametrize('active', ['leased', 'reservation', 'dispatch'])
def test_uncommitted_admission_is_seen_after_publication_waits(database, completed, active):
    h = isolation.reset(database, completed)
    run = isolation.selected(completed, 'MissingJudge')
    with database[0].connect(**database[1]) as conn, conn.cursor() as cur:
        isolation.gap(cur, run)
    patch = _omitted_patch(h)
    with database[0].connect(**database[1]) as admission, admission.cursor() as cur, ThreadPoolExecutor(max_workers=1) as pool:
        # Canonical reserve/dispatch lease validation holds this round SHARE.
        # Isolate that exact boundary with no network/provider side effects.
        cur.execute('SELECT 1 FROM public.lab_arena_rounds WHERE round_id=%s FOR SHARE', (h.round_id,))
        cur.execute('SET LOCAL session_replication_role=replica')
        if active == 'leased':
            cur.execute("UPDATE public.lab_arena_runs SET status='leased' WHERE scored_run_id=%s", (run['run_id'],))
        else:
            for kind in (['reservation'] if active == 'reservation' else ['reservation', 'dispatch']):
                cur.execute("INSERT INTO public.lab_arena_ledger(entry_kind,miner_hotkey,round_id,submission_id,run_id,stage,call_identity,provider,operation_id,funding_source,amount_microusd) SELECT %s,miner_hotkey,round_id,submission_id,run_id,stage,%s,'openrouter','openrouter.chat','miner_key',0 FROM public.lab_arena_runs WHERE run_id=%s", (kind, 'sha256:' + '6' * 64, run['run_id']))
        cur.execute('SET LOCAL session_replication_role=origin')
        waiting = pool.submit(_publish, database, h.round_id, patch)
        _wait_for_round_lock(database, h.round_id, waiting)
        admission.commit()
        assert waiting.result(timeout=8) == {'status': 'retry', 'reason': 'evaluation_draining'}
    assert h.status() == 'scored'


def test_publication_winner_closes_stale_completion_without_mutating_history(database, completed):
    h = isolation.reset(database, completed)
    run = isolation.selected(completed, 'MissingJudge')
    judge = dict(next(r for r in completed[2] if r['scored_run_id'] == run['run_id']))
    judge['_output_hash'] = contracts.document_hash(json.loads(h.objects.get(judge['output_ref'])))
    with database[0].connect(**database[1]) as conn, conn.cursor() as cur:
        isolation.gap(cur, run)
    patch = _omitted_patch(h)
    with database[0].connect(**database[1]) as conn, conn.cursor() as cur:
        before = isolation.state(cur, h.round_id)
    with database[0].connect(**database[1]) as publishing, publishing.cursor() as cur, ThreadPoolExecutor(max_workers=1) as pool:
        cur.execute('SELECT pg_advisory_xact_lock(hashtextextended(\'lab-arena-claim-control\',0))')
        cur.execute('SELECT 1 FROM public.lab_arena_rounds WHERE round_id=%s FOR UPDATE', (h.round_id,))
        cur.execute("SELECT public.lab_arena_transition_round(%s,'scored','published',%s::jsonb)", (h.round_id, json.dumps(patch)))
        assert cur.fetchone()[0]['status'] == 'ok'
        def late():
            with database[0].connect(**database[1]) as conn, conn.cursor() as q:
                q.execute("SET LOCAL statement_timeout='8s'")
                return _complete(q, judge)
        waiting = pool.submit(late)
        # The stale attempt must at least be unable to overwrite publication;
        # an already terminal lease may reject before waiting on the round.
        publishing.commit()
        # The already failed attempt replays its durable terminal status.
        # It must not accept this delayed success or mutate its old result.
        assert waiting.result(timeout=8)['status'] == 'failed'
    row = h.service.store.get_round(h.round_id)
    assert row['status'] == 'published' and row['publication_doc'] == patch['publication_doc']
    with database[0].connect(**database[1]) as conn, conn.cursor() as cur:
        assert isolation.state(cur, h.round_id) == before
        cur.execute('SELECT status,per_icp_score FROM public.lab_arena_runs WHERE run_id=%s', (run['run_id'],))
        assert cur.fetchone() == ('accepted', None)
