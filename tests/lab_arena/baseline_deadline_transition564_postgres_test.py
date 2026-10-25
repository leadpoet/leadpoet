"""A live baseline retry delays final authority, not post-deadline miner work."""
import copy
import json
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from tests.lab_arena import company_judgment_validation_projection548_postgres_test as prior


ROOT = Path(__file__).parents[2]
SQL = ROOT / 'scripts/564-lab-arena-baseline-deadline-transition.sql'
HELPER = 'public.lab_arena__baseline_transition_ready_v1'
PROOF = 'public.lab_arena__publication_execution_incomplete_v1'


@pytest.fixture(scope='module')
def scenario():
    database_gen = prior.database.__wrapped__()
    db = next(database_gen)
    with db[0].connect(**db[1]) as conn, conn.cursor() as cur:
        for migration in ('551-lab-arena-expire-leases-shared-round-lock.sql',
                          '552-lab-arena-expiry-recovery-candidate-locks.sql'):
            cur.execute((ROOT / 'scripts' / migration).read_text())
        # Exercise the exact combined deployed SQL, not only the historical
        # migration 424 and 472 bodies in isolation.
        for signature, expected in {
            'lab_arena_expire_leases(text)': '5025e77ac2a602941e7166d0a201ba49efb03d0da8af1a0fda1e1889056de40d',
            'lab_arena__judge_continuation_v1(public.lab_arena_rounds,public.lab_arena_runs)': '3a0ff0eb9209d66f766daaee4eebab92a81f5f1d05e5d189130d7ec1dc69b0af',
            'lab_arena_close_scoring(text,smallint)': 'a07b809f89df827f3e0034e71f01c587b49bf1718715c7b60320e1521f322a6c',
            'lab_arena__publication_execution_incomplete_v1(text,text)': '7e75295c727e3c62243f9de6c2dcfb422718fe3bec539871d1bb8ebc62075c2e',
        }.items():
            assert scalar(cur, "encode(extensions.digest(pg_get_functiondef(%s::regprocedure),'sha256'),'hex')",
                          ('public.' + signature,)) == expected
    # Harness objects share their parent path. Keep this fixture's parent
    # private even when another PostgreSQL test is running concurrently.
    class Paths:
        def __init__(self, root):
            self.root = root

        def mktemp(self, name):
            path = self.root / name
            path.mkdir()
            return path

    with tempfile.TemporaryDirectory(prefix='arena-transition564-') as tmp:
        completed_gen = prior.completed.__wrapped__(db, Paths(Path(tmp)))
        try:
            harness, row = next(completed_gen)
            with db[0].connect(**db[1]) as conn, conn.cursor() as cur:
                saved = {}
                for table in ('lab_arena_rounds','lab_arena_runs','lab_arena_ledger',
                              'lab_arena_judgment_cache'):
                    cur.execute(f'SELECT to_jsonb(t) FROM public.{table} t')
                    saved[table] = [item[0] for item in cur.fetchall()]
            yield db, harness, row, saved
        finally:
            completed_gen.close()
            database_gen.close()


def scalar(cur, expression, params=()):
    cur.execute('SELECT ' + expression, params)
    return cur.fetchone()[0]


def prepare(scenario, *, before=False, attempt=12):
    db, h, original, saved = scenario
    rid = original['round_id']
    baseline = next(p['submission_id'] for p in original['participants'] if p['is_king'])
    cfg = copy.deepcopy(original['configuration_doc'])
    now = datetime.now(timezone.utc)
    close = now + timedelta(hours=1) if before else now - timedelta(seconds=5)
    cfg['schedule'].update({
        'submission_cutoff': (now - timedelta(hours=12)).isoformat(),
        'stage_1_start': (now - timedelta(hours=12)).isoformat(),
        'stage_1_close': (now - timedelta(hours=2)).isoformat(),
        'stage_1_scoring_close': close.isoformat(),
        'stage_2_start': close.isoformat(),
        'stage_2_close': (now + timedelta(hours=4)).isoformat(),
        'final_scoring_close': (now + timedelta(hours=5)).isoformat(),
        'publication_deadline': (now + timedelta(hours=6)).isoformat(),
    })
    with db[0].connect(**db[1]) as conn, conn.cursor() as cur:
        cur.execute('SET LOCAL session_replication_role=replica')
        for table in reversed(saved):
            cur.execute(f'DELETE FROM public.{table}')
        for table, rows in saved.items():
            cur.execute("SELECT string_agg(quote_ident(attname),',' ORDER BY attnum) "
                        "FROM pg_attribute WHERE attrelid=%s::regclass AND attnum>0 "
                        "AND NOT attisdropped AND attgenerated=''", ('public.' + table,))
            columns = cur.fetchone()[0]
            for row in rows:
                cur.execute(f'INSERT INTO public.{table} ({columns}) SELECT {columns} '
                            f'FROM jsonb_populate_record(NULL::public.{table},%s::jsonb)',
                            (json.dumps(row),))
        cur.execute('DELETE FROM public.lab_arena_runs WHERE round_id=%s AND submission_id<>%s',
                    (rid, baseline))
        cur.execute("UPDATE public.lab_arena_rounds SET status='stage1_judged',"
                    "configuration_doc=%s::jsonb,stage_generation=3,"
                    'stage2_scoring_plan_doc=NULL,finalists=NULL,publication_doc=NULL WHERE round_id=%s',
                    (json.dumps(cfg), rid))
        cur.execute("SELECT to_jsonb(r) FROM public.lab_arena_runs r WHERE round_id=%s "
                    "AND submission_id=%s AND kind='score' AND icp_position=9 AND status='accepted'",
                    (rid, baseline))
        judge = cur.fetchone()[0]
        for number in range(1, attempt + 1):
            doc = copy.deepcopy(judge)
            doc.update(run_id=judge['assignment_id'] + ':' + str(number), attempt=number,
                status='failed', terminal_cause='judge_error', result_doc={'terminal_status':'judge_error'},
                output_ref=None, claim_request_id=None, claim_response=None,
                lease_token_hash=None, lease_expires_at=None)
            if number == 1:
                cur.execute('DELETE FROM public.lab_arena_runs WHERE run_id=%s', (judge['run_id'],))
            cur.execute('INSERT INTO public.lab_arena_runs SELECT '
                        '(jsonb_populate_record(NULL::public.lab_arena_runs,%s::jsonb)).*',
                        (json.dumps(doc),))
        cur.execute('DELETE FROM public.lab_arena_judgment_cache WHERE cache_key=%s',
                    (judge['judgment_cache_key'],))
        cur.execute('UPDATE public.lab_arena_runs SET per_icp_score=NULL,qualification_doc=NULL '
                    'WHERE run_id=%s', (judge['scored_run_id'],))
        cur.execute('SET LOCAL session_replication_role=origin')
    h.clock.now = now
    h.service._invalidate_hot_round()
    return rid, baseline, judge


def fingerprint(cur, rid):
    return tuple(scalar(cur, f"md5(COALESCE(jsonb_agg(to_jsonb(t) ORDER BY to_jsonb(t)::text)::text,'null')) "
        f'FROM public.{table} t WHERE round_id=%s' + where, (rid,))
        for table, where in [('lab_arena_ledger',''),
            ('lab_arena_runs'," AND kind='score' AND status='accepted'")])


def test_real_advance_fails_before_migration_and_preserves_live_retry_after(scenario):
    db, h, _, _ = scenario
    rid, baseline, judge = prepare(scenario)
    # This is the production order: expire materializes ordinary attempt 13
    # before score_stage calls the SQL transition. It failed before this fix.
    with pytest.raises(Exception, match='lab_arena_stage1_baseline_scores_incomplete'):
        h.service.advance_round(rid)
    with db[0].connect(**db[1]) as conn, conn.cursor() as cur:
        before = fingerprint(cur, rid)
        security = scalar(cur, "jsonb_build_array(proowner,proacl,prosecdef,provolatile,proconfig) "
            "FROM pg_proc WHERE oid='public.lab_arena_transition_round(text,text,text,jsonb)'::regprocedure")
        schema_acl = scalar(cur, "nspacl FROM pg_namespace WHERE nspname='public'")
        publication_source = scalar(cur, "pg_get_functiondef((%s)::regprocedure)",
            (PROOF + '(text,text)',))
        cur.execute(SQL.read_text())
        cur.execute(SQL.read_text())
        assert scalar(cur, "jsonb_build_array(proowner,proacl,prosecdef,provolatile,proconfig) "
            "FROM pg_proc WHERE oid='public.lab_arena_transition_round(text,text,text,jsonb)'::regprocedure") == security
        assert scalar(cur, "nspacl FROM pg_namespace WHERE nspname='public'") == schema_acl
        assert scalar(cur, "jsonb_build_array(pg_get_userbyid(proowner),proacl::text,prosecdef,provolatile,proconfig) "
            'FROM pg_proc WHERE oid=%s::regprocedure', (HELPER + '(text,text)',)) == [
                'lab_arena_owner','{lab_arena_owner=X/lab_arena_owner}',True,'s',['search_path=pg_catalog, public']]
        assert scalar(cur, "pg_get_functiondef((%s)::regprocedure)",
            (PROOF + '(text,text)',)) == publication_source
        assert scalar(cur, HELPER + '(%s,%s)', (rid, baseline)) is True
        assert scalar(cur, PROOF + '(%s,%s)', (rid, baseline)) is False
        assert scalar(cur, 'public.lab_arena__baseline_final_ready_v1(%s)', (rid,)) is False
        assert fingerprint(cur, rid) == before
    h.service._invalidate_hot_round()
    assert h.service.advance_round(rid)['status'] == 'ok'
    assert h.status() == 'stage1_scored'
    assert h.service.advance_round(rid)['status'] == 'ok'
    assert h.status() == 'stage2'
    with db[0].connect(**db[1]) as conn, conn.cursor() as cur:
        assert fingerprint(cur, rid) == before
        cur.execute('SELECT status,attempt FROM public.lab_arena_runs WHERE assignment_id=%s ORDER BY attempt DESC LIMIT 1',
                    (judge['assignment_id'],))
        assert cur.fetchone() == ('pending', 13)
    # Keep the normal retry backoff. Miners can finish without taking or
    # terminating that baseline assignment, but cannot publish or promote.
    h.advance_until('stage2_scoring', runners=1)
    h.run_stage_with_runners(1)
    assert h.service.advance_round(rid)['status'] == 'waiting'
    assert h.status() == 'stage2_scoring'
    assert h.service.store.get_run(judge['assignment_id'] + ':13')['status'] == 'pending'
    assert h.service.store.get_run(judge['scored_run_id'])['per_icp_score'] is None
    # Current baseline judges now participate in stage-two completion, so the
    # unchanged final guard is reached only after that same retry finishes.
    assert h.service.publish(rid)['status'] == 'stale'
    assert not h.service._baseline_fully_succeeded(h.service.store.get_round(rid))
    assert h.service.promote_baseline(rid)['status'] == 'not_required'
    row = h.service.store.get_round(rid)
    assert row['publication_doc'] is None and row['promotion_doc'] is None
    with db[0].connect(**db[1]) as conn, conn.cursor() as cur:
        cur.execute('SET LOCAL session_replication_role=replica')
        cur.execute("UPDATE public.lab_arena_runs SET created_at=now()-interval '30 minutes' WHERE run_id=%s",
                    (judge['assignment_id'] + ':13',))
        cur.execute('SET LOCAL session_replication_role=origin')
    # Existing runners complete the preserved attempt with their normal signed
    # claim/completion and in-memory provider transport, after miners finish.
    h.run_stage_with_runners(1)
    accepted = h.service.store.get_run(judge['assignment_id'] + ':13')
    assert accepted['status'] == 'accepted'
    assert h.service.store.get_run(judge['scored_run_id'])['per_icp_score'] is None
    # Normal final-stage scoring materializes the real accepted attempt 13
    # against the original source before publication. No score is fabricated.
    h.service._invalidate_hot_round()
    h.advance_until('scored', runners=1)
    assert h.service.publish(rid)['status'] == 'ok'
    assert h.status() == 'published'
    with db[0].connect(**db[1]) as conn, conn.cursor() as cur:
        assert scalar(cur, 'public.lab_arena__baseline_final_ready_v1(%s)', (rid,)) is True
        assert scalar(cur, 'count(*) FROM public.lab_arena_runs WHERE round_id=%s AND submission_id=%s '
            "AND kind='execute' AND per_icp_score IS NOT NULL", (rid, baseline)) == 10


def test_normally_claimed_lease_survives_stage_change_and_recovers_baseline(scenario):
    db, h, _, _ = scenario
    rid, _, judge = prepare(scenario)
    with db[0].connect(**db[1]) as conn, conn.cursor() as cur:
        cur.execute(SQL.read_text())
        conn.commit()
        scalar(cur, 'public.lab_arena_expire_leases(%s)', (rid,))
        cur.execute('SET LOCAL session_replication_role=replica')
        cur.execute("UPDATE public.lab_arena_runs SET created_at=now()-interval '30 minutes' WHERE run_id=%s",
                    (judge['assignment_id'] + ':13',))
        cur.execute('SET LOCAL session_replication_role=origin')
    runner = h.runner(0)
    try:
        lease = runner.claim_one()
        assert lease['status']=='leased' and lease['run_id']==judge['assignment_id'] + ':13'
        preserved = h.service.store.get_run(lease['run_id'])
        h.service._invalidate_hot_round()
        assert h.service.advance_round(rid)['status']=='ok'
        assert h.service.advance_round(rid)['status']=='ok'
        assert h.status()=='stage2'
        assert h.service.store.get_run(lease['run_id'])==preserved
        # Match run_once's local slot ownership while we deliberately defer
        # dispatch across the stage change.
        assert runner._slots.acquire(blocking=False)
        runner._run_lease(lease)
        assert h.service.store.get_run(lease['run_id'])['status']=='accepted'
    finally:
        runner.close()
    # The normal stage2 score path invokes _score_recovered_baseline while its
    # accepted source and attempt13 retain their frozen original membership.
    h.advance_until('scored', runners=1)
    assert h.service.store.get_run(judge['scored_run_id'])['per_icp_score'] is not None


@pytest.mark.parametrize('variant', ['pending','leased','before','accepted','stale',
    'bad_lineage','missing_parent','nonbaseline','accepted_cache','healthy','old_stage',
    'missing_source','bad_source_id'])
def test_transition_proof_bounds_and_final_authority(scenario, variant):
    db, h, _, _ = scenario
    rid, baseline, judge = prepare(scenario, before=variant in ('before','healthy'))
    with db[0].connect(**db[1]) as conn, conn.cursor() as cur:
        cur.execute(SQL.read_text())
        conn.commit()
        assert scalar(cur, 'public.lab_arena_expire_leases(%s)', (rid,))['retried'] == 1
        assert scalar(cur, 'public.lab_arena_expire_leases(%s)', (rid,))['retried'] == 0
        retry = judge['assignment_id'] + ':13'
        cur.execute('SET LOCAL session_replication_role=replica')
        if variant == 'leased':
            cur.execute("UPDATE public.lab_arena_runs SET status='leased',lease_expires_at=now()+interval '30 minutes' WHERE run_id=%s", (retry,))
        elif variant in ('accepted','accepted_cache','healthy'):
            cur.execute("UPDATE public.lab_arena_runs SET status='accepted',terminal_cause='accepted',"
                        "output_ref=%s,result_doc=%s::jsonb,claim_response=%s::jsonb WHERE run_id=%s",
                        (judge['output_ref'], json.dumps(judge['result_doc']), json.dumps(judge['claim_response']), retry))
            if variant == 'accepted_cache':
                # Company-cache references are the preserved accepted authority
                # for this frozen policy, including the valid empty output case.
                cur.execute('UPDATE public.lab_arena_runs SET company_judgment_refs=%s::jsonb WHERE run_id=%s',
                            (json.dumps(judge['company_judgment_refs']), retry))
            if variant == 'healthy':
                source = next(r for r in scenario[3]['lab_arena_runs'] if r['run_id']==judge['scored_run_id'])
                cur.execute('UPDATE public.lab_arena_runs SET per_icp_score=%s,qualification_doc=%s::jsonb WHERE run_id=%s',
                            (source['per_icp_score'], json.dumps(source['qualification_doc']), source['run_id']))
        elif variant == 'stale':
            # The generation floor is immutable even with replica mode.
            cur.execute('SELECT to_jsonb(r) FROM public.lab_arena_runs r WHERE run_id=%s', (retry,))
            stale = cur.fetchone()[0]
            stale['scoring_generation_floor'] = 1
            cur.execute('DELETE FROM public.lab_arena_runs WHERE run_id=%s', (retry,))
            cur.execute('INSERT INTO public.lab_arena_runs SELECT '
                        '(jsonb_populate_record(NULL::public.lab_arena_runs,%s::jsonb)).*', (json.dumps(stale),))
        elif variant == 'bad_lineage':
            cur.execute("UPDATE public.lab_arena_runs SET judgment_input_hash='sha256:'||repeat('f',64) WHERE run_id=%s", (retry,))
        elif variant == 'missing_parent':
            cur.execute('DELETE FROM public.lab_arena_runs WHERE run_id=%s', (judge['assignment_id'] + ':12',))
        elif variant == 'missing_source':
            cur.execute('DELETE FROM public.lab_arena_runs WHERE run_id=%s', (judge['scored_run_id'],))
        elif variant == 'bad_source_id':
            cur.execute("UPDATE public.lab_arena_runs SET assignment_id='malformed' WHERE run_id=%s", (judge['scored_run_id'],))
        elif variant == 'nonbaseline':
            cur.execute("UPDATE public.lab_arena_rounds SET configuration_doc=jsonb_set(configuration_doc,'{execution_sequence_policy}','\"other\"') WHERE round_id=%s", (rid,))
        elif variant == 'old_stage':
            # Stage generations advance during the same frozen score policy;
            # existing continuation authority deliberately spans that boundary.
            cur.execute('UPDATE public.lab_arena_rounds SET stage_generation=4 WHERE round_id=%s', (rid,))
        cur.execute('SET LOCAL session_replication_role=origin')
        before = fingerprint(cur, rid)
        assert scalar(cur, HELPER + '(%s,%s)', (rid, baseline)) is (variant in ('pending','leased','old_stage'))
        # The unchanged legacy proof ignores a superseded-floor pending row;
        # its failed current-floor parent retains the pre-existing omission proof.
        assert scalar(cur, PROOF + '(%s,%s)', (rid, baseline)) is (variant=='stale')
        assert scalar(cur, 'public.lab_arena__baseline_final_ready_v1(%s)', (rid,)) is (variant=='healthy')
        assert fingerprint(cur, rid) == before
        if variant in ('pending','leased'):
            for column, value in [('status', 'published'), ('promotion_doc', '{}')]:
                cur.execute('SAVEPOINT final_authority')
                with pytest.raises(db[0].Error, match='lab_arena_final_baseline_incomplete'):
                    cur.execute(f'UPDATE public.lab_arena_rounds SET {column}=%s WHERE round_id=%s',
                                (value, rid))
                cur.execute('ROLLBACK TO SAVEPOINT final_authority')
                assert fingerprint(cur, rid) == before
    if variant in ('pending','leased','healthy','old_stage','accepted','accepted_cache'):
        preserved = h.service.store.get_run(retry)
        h.service._invalidate_hot_round()
        assert h.service.advance_round(rid)['status'] == 'ok'
        assert h.status() == 'stage1_scored'
        assert h.service.store.get_run(retry) == preserved
        if variant in ('accepted','accepted_cache'):
            assert h.service.store.get_run(judge['scored_run_id'])['per_icp_score'] is not None
        assert h.service.store.transition_round(rid,'stage1_judged','stage1_scored',{})['status'] == 'stale'
    if variant == 'before':
        h.service._invalidate_hot_round()
        assert h.service.advance_round(rid)['reason'] == 'partial_baseline_deadline_pending'
        assert h.status() == 'stage1_judged'


@pytest.mark.parametrize('variant', ['helper_acl','helper_source','rpc_acl','rpc_source'])
def test_migration_rejects_source_or_security_drift(scenario, variant):
    db, _, _, _ = scenario
    with db[0].connect(**db[1]) as conn:
        with conn.cursor() as cur:
            cur.execute(SQL.read_text())
            conn.commit()
            signature = (HELPER + '(text,text)' if variant.startswith('helper') else
                         'public.lab_arena_transition_round(text,text,text,jsonb)')
            source = scalar(cur, 'pg_get_functiondef(%s::regprocedure)', (signature,))
            if variant.endswith('acl'):
                cur.execute('GRANT EXECUTE ON FUNCTION ' + signature + ' TO PUBLIC')
            else:
                cur.execute(source.replace('BEGIN', '-- unexpected source drift\nBEGIN', 1))
            with pytest.raises(db[0].Error, match='source/security differs|security differs|preimage differs'):
                cur.execute(SQL.read_text())
        conn.rollback()
