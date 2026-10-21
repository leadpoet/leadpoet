"""Frozen judge admission ends while existing work and accounting drain safely."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import hashlib
import json
import re
import time

import pytest
from lab_arena.store import ArenaStore, PsycopgTransport
from tests.lab_arena import partial_judge_host_fault527_postgres_test as prior
from tests.lab_arena import ordered_claim_admission520_postgres_test as queue

ROOT = Path(__file__).parents[2]
SQL = ROOT / 'scripts/530-lab-arena-judge-final-deadline.sql'
SIGNATURES = {'claim': prior.SIG, 'cancel': 'public.lab_arena_cancel_round(text,text)'}


@pytest.fixture(scope='module')
def database():
    gen = prior.database.__wrapped__()
    db = next(gen)
    try:
        with db[0].connect(**db[1]) as conn, conn.cursor() as cur:
            cur.execute(SQL.read_text())
            cur.execute(SQL.read_text())
        yield db
    finally:
        gen.close()


def definition(cur, key):
    cur.execute('SELECT pg_get_functiondef(%s::regprocedure)', (SIGNATURES[key],))
    return cur.fetchone()[0]


def preimage(key, post):
    text = SQL.read_text()
    old = re.search(r'\$old_'+key+r'\$(.*?)\$old_'+key+r'\$', text, re.S)[1]
    new = re.search(r'\$new_'+key+r'\$(.*?)\$new_'+key+r'\$', text, re.S)[1]
    assert post.count(new) == 1
    post = post.replace(new, old)
    if key == 'claim':
        for tag in ('claim_decl', 'claim_read'):
            old = re.search(r'\$old_'+tag+r'\$(.*?)\$old_'+tag+r'\$', text, re.S)[1]
            new = re.search(r'\$new_'+tag+r'\$(.*?)\$new_'+tag+r'\$', text, re.S)[1]
            assert post.count(new) == 1
            post = post.replace(new, old)
    return post


def prepared(db, offset=3600, *, wide=False):
    # Set the fixture's frozen deadline BEFORE the real policy revision records
    # its protected hash. A post-revision config edit would invalidate authority.
    store, row, old, new = queue.revision.seed(db)
    try:
        with db[0].connect(**db[1]) as conn, conn.cursor() as cur:
            deadline(cur, row['round_id'], offset)
            if wide:
                cur.execute('SET LOCAL session_replication_role=replica')
                cur.execute("UPDATE public.lab_arena_rounds SET configuration_doc=configuration_doc||jsonb_build_object('wide_deadline_control',%s::jsonb) WHERE round_id=%s", (json.dumps([hashlib.sha256(str(i).encode()).hexdigest() for i in range(26000)]), row['round_id']))
                cur.execute('SET LOCAL session_replication_role=origin')
            cur.execute('SELECT to_jsonb(r) FROM public.lab_arena_rounds r WHERE round_id=%s', (row['round_id'],))
            queue.revision.invoke(cur, cur.fetchone()[0], new)
            cur.execute('SET LOCAL session_replication_role=replica')
            cur.execute("UPDATE public.lab_arena_runs SET created_at=clock_timestamp()-interval '20 minutes' WHERE round_id=%s AND status='pending'", (row['round_id'],))
            cur.execute('SET LOCAL session_replication_role=origin')
            cur.execute('UPDATE public.lab_arena_restart_claim_control SET operator_paused=false WHERE singleton')
            cur.execute("SELECT run_id FROM public.lab_arena_runs WHERE round_id=%s AND kind='score' AND status='pending' ORDER BY icp_position,created_at,assignment_id", (row['round_id'],))
            ids = [r[0] for r in cur.fetchall()]
        return row['round_id'], ids
    finally:
        store.close()


def deadline(cur, rid, offset):
    # Only fixture clock control bypasses immutable frozen configuration.
    cur.execute('SET LOCAL session_replication_role=replica')
    cur.execute("UPDATE public.lab_arena_rounds SET configuration_doc=jsonb_set(configuration_doc,'{schedule,publication_deadline}',to_jsonb(clock_timestamp()+make_interval(secs=>%s))) WHERE round_id=%s", (offset, rid))
    cur.execute('SET LOCAL session_replication_role=origin')


def snapshot(cur, rid):
    cur.execute('SELECT jsonb_agg(to_jsonb(r)ORDER BY run_id) FROM public.lab_arena_runs r WHERE round_id=%s', (rid,))
    runs = cur.fetchone()[0]
    cur.execute('SELECT jsonb_agg(to_jsonb(l)ORDER BY entry_id) FROM public.lab_arena_ledger l WHERE round_id=%s', (rid,))
    return runs, cur.fetchone()[0]


def cancel(cur, rid, reason='scoring_incomplete'):
    cur.execute('SELECT public.lab_arena_cancel_round(%s,%s)', (rid, reason))
    return cur.fetchone()[0]


@pytest.mark.parametrize('offset,expected', [(3600, 'leased'), (0, 'no_pending'), (-1, 'no_pending')])
def test_real_score_claim_frozen_deadline(database, offset, expected):
    rid, _ = prepared(database, offset=offset)
    with database[0].connect(**database[1]) as conn, conn.cursor() as cur:
        before = snapshot(cur, rid)
        result = queue.claim(cur, rid)
        assert result['status'] == expected
        if expected == 'no_pending':
            assert snapshot(cur, rid) == before


def test_missing_deadline_fails_closed_for_score(database):
    rid, _ = prepared(database)
    with database[0].connect(**database[1]) as conn, conn.cursor() as cur:
        cur.execute('SET LOCAL session_replication_role=replica')
        cur.execute("UPDATE public.lab_arena_rounds SET configuration_doc=configuration_doc#-'{schedule,publication_deadline}' WHERE round_id=%s", (rid,))
        cur.execute('SET LOCAL session_replication_role=origin')
        assert queue.claim(cur, rid)['status'] == 'no_pending'


def test_pending_claim_waiting_on_control_lock_rechecks_deadline(database):
    rid, _ = prepared(database, offset=1)
    with database[0].connect(**database[1]) as blocker, blocker.cursor() as lock, ThreadPoolExecutor(max_workers=1) as pool:
        lock.execute("SELECT pg_advisory_xact_lock(hashtextextended('lab-arena-claim-control',0))")
        def call():
            with database[0].connect(**database[1]) as conn, conn.cursor() as cur:
                cur.execute("SET LOCAL statement_timeout='5s'")
                return queue.claim(cur, rid)
        waiting = pool.submit(call)
        # Cross the actual frozen boundary while the claim cannot yet read it.
        time.sleep(1.1)
        blocker.commit()
        assert waiting.result(timeout=5)['status'] == 'no_pending'


def add_head(cur, rid, kind, *, cid='c', terminal=None):
    cur.execute('SET LOCAL session_replication_role=replica')
    cur.execute("INSERT INTO public.lab_arena_ledger(entry_kind,miner_hotkey,round_id,submission_id,run_id,stage,call_identity,provider,operation_id,funding_source,amount_microusd,entry_doc) SELECT %s,miner_hotkey,round_id,submission_id,run_id,stage,%s,'deepline','exa.contents','host',123,%s::jsonb FROM public.lab_arena_runs WHERE round_id=%s AND kind='execute' ORDER BY run_id LIMIT 1", (kind, 'sha256:'+cid*64, json.dumps(terminal or {}), rid))
    cur.execute('SET LOCAL session_replication_role=origin')


@pytest.mark.parametrize('active', ['leased', 'submitted', 'reservation', 'dispatch'])
def test_atomic_deadline_cancel_does_not_terminate_active_work(database, active):
    rid, ids = prepared(database)
    with database[0].connect(**database[1]) as conn, conn.cursor() as cur:
        if active == 'leased':
            assert queue.claim(cur, rid)['status'] == 'leased'
            cur.execute('RESET ROLE')
        elif active == 'submitted':
            cur.execute('SET LOCAL session_replication_role=replica')
            cur.execute("UPDATE public.lab_arena_runs SET status='submitted' WHERE run_id=%s", (ids[0],))
            cur.execute('SET LOCAL session_replication_role=origin')
        else:
            add_head(cur, rid, 'reservation')
            if active == 'dispatch':
                add_head(cur, rid, 'dispatch')
        deadline(cur, rid, -1)
        before = snapshot(cur, rid)
        assert cancel(cur, rid)['reason'] == 'evaluation_draining'
        assert snapshot(cur, rid) == before
        cur.execute('SELECT status FROM public.lab_arena_rounds WHERE round_id=%s', (rid,))
        assert cur.fetchone()[0] != 'cancelled'


@pytest.mark.parametrize('kind', ['settlement', 'uncertain', 'recovery', 'refusal'])
def test_closed_or_unknown_heads_do_not_block_safe_failure_or_change_costs(database, kind):
    rid, _ = prepared(database)
    with database[0].connect(**database[1]) as conn, conn.cursor() as cur:
        add_head(cur, rid, 'reservation')
        add_head(cur, rid, kind, terminal={'success': False, 'finality': 'unknown'})
        deadline(cur, rid, -1)
        _, ledger = snapshot(cur, rid)
        cur.execute("SELECT jsonb_agg(to_jsonb(r)ORDER BY run_id) FROM public.lab_arena_runs r WHERE round_id=%s AND status='accepted'", (rid,))
        accepted = cur.fetchone()[0]
        assert cancel(cur, rid)['status'] == 'cancelled'
        assert snapshot(cur, rid)[1] == ledger
        cur.execute("SELECT jsonb_agg(to_jsonb(r)ORDER BY run_id) FROM public.lab_arena_runs r WHERE round_id=%s AND status='accepted'", (rid,))
        assert cur.fetchone()[0] == accepted
        cur.execute('SELECT publication_doc,reward_basis_doc,baseline_promoted_at FROM public.lab_arena_rounds WHERE round_id=%s', (rid,))
        assert cur.fetchone() == (None, None, None)
        assert cancel(cur, rid)['status'] == 'existing'


def test_predeadline_claim_and_cancel_serialize_without_losing_lease(database):
    rid, _ = prepared(database)
    with database[0].connect(**database[1]) as claimed, claimed.cursor() as cur, ThreadPoolExecutor(max_workers=1) as pool:
        lease = queue.claim(cur, rid)
        assert lease['status'] == 'leased'
        def fail():
            with database[0].connect(**database[1]) as conn, conn.cursor() as q:
                q.execute("SET LOCAL statement_timeout='5s'")
                return cancel(q, rid)
        waiting = pool.submit(fail)
        time.sleep(.05)
        assert not waiting.done()
        claimed.commit()
        assert waiting.result(timeout=5)['reason'] == 'evaluation_draining'
        with database[0].connect(**database[1]) as conn, conn.cursor() as q:
            q.execute('SELECT status,lease_generation,claim_response FROM public.lab_arena_runs WHERE run_id=%s', (lease['run_id'],))
            state, generation, response = q.fetchone()
            assert state == 'leased' and generation == lease['lease_generation'] and response == lease


def test_real_late_valid_company_completion_and_billing_remain_authoritative(database):
    rid, _ = prepared(database, offset=2)
    with database[0].connect(**database[1]) as conn, conn.cursor() as cur:
        cur.execute('SET LOCAL session_replication_role=replica')
        cur.execute("UPDATE public.lab_arena_rounds SET status='scored' WHERE round_id=%s", (rid,))
        cur.execute('SET LOCAL session_replication_role=origin')
    store = ArenaStore(PsycopgTransport(lambda: database[0].connect(**database[1])))
    try:
        company = queue.revision.company
        runner = company.original.hotkey('deadline530-late')
        lease, token, _, _ = company.original.claim(store, rid, runner, excluded=[runner])
        assert lease['status'] == 'leased'
        run = store.get_run(lease['run_id'])
        time.sleep(2.1)
        with database[0].connect(**database[1]) as conn, conn.cursor() as cur:
            assert cancel(cur, rid)['reason'] == 'evaluation_draining'
        result = company.complete_subset(store, lease, token, run['company_judgment_refs'])
        assert result['status'] == 'accepted'
        accepted = store.get_run(lease['run_id'])
        assert accepted['status'] == 'accepted' and accepted['output_ref']
        # Completion durably preserves actual signed company authorities.
        # Full service verification/projection/publication is exercised below;
        # this control never fabricates a qualification receipt or numeric score.
        assert store.get_run(accepted['scored_run_id'])['per_icp_score'] is None
        with database[0].connect(**database[1]) as conn, conn.cursor() as cur:
            assert cancel(cur, rid)['status'] == 'cancelled'
        assert store.get_run(lease['run_id']) == accepted
    finally:
        store.close()


@pytest.mark.parametrize('key', ['claim', 'cancel'])
@pytest.mark.parametrize('installed', [False, True])
@pytest.mark.parametrize('drift', ['source', 'owner', 'acl', 'secdef', 'volatility', 'path'])
def test_exact_source_security_and_replay_fail_closed(database, key, installed, drift):
    with database[0].connect(**database[1]) as conn, conn.cursor() as cur:
        saved = {k: definition(cur, k) for k in SIGNATURES}
        cur.execute(saved[key] if installed else preimage(key, saved[key]))
        cur.execute('SAVEPOINT hostile')
        sig = SIGNATURES[key]
        if drift == 'source':
            cur.execute(definition(cur, key).replace('BEGIN\n', 'BEGIN\n  -- unknown source\n', 1))
        else:
            cur.execute({'owner': 'ALTER FUNCTION '+sig+' OWNER TO postgres',
                'acl': 'GRANT EXECUTE ON FUNCTION '+sig+' TO anon',
                'secdef': 'ALTER FUNCTION '+sig+' SECURITY INVOKER',
                'volatility': 'ALTER FUNCTION '+sig+' STABLE',
                'path': 'ALTER FUNCTION '+sig+' SET search_path=public,pg_catalog'}[drift])
        body = SQL.read_text().split('DO $judge_final_deadline$')[1].split('$judge_final_deadline$;')[0]
        with pytest.raises(database[0].Error):
            cur.execute('DO $judge_final_deadline$'+body+'$judge_final_deadline$;')
        cur.execute('ROLLBACK TO SAVEPOINT hostile')
        for source in saved.values():
            cur.execute(source)


def test_full_service_late_publication_preserves_verified_real_scores(database, tmp_path, monkeypatch):
    from dataclasses import replace
    from datetime import timedelta
    from lab_arena import service
    from tests.lab_arena.test_integrity_round import IntegrityHarness
    from tests.lab_arena import test_lab_arena_service_round as flows
    from tests.lab_arena.baseline_scored_first_postgres_test import _baseline_first_judge, _enable_verified_proxy_runtime
    from tests.lab_arena.icp_fixtures import daily_icps
    with database[0].connect(**database[1]) as conn, conn.cursor() as cur:
        cur.execute('SET LOCAL session_replication_role=replica')
        cur.execute('TRUNCATE public.lab_arena_rounds,public.lab_arena_ledger,public.lab_arena_trajectory_events RESTART IDENTITY CASCADE')
        cur.execute("UPDATE public.lab_arena_restart_claim_control SET operator_paused=false,pause_reason='',actor_ref='' WHERE singleton")
        cur.execute('SET LOCAL session_replication_role=origin')
    harness = IntegrityHarness(lambda: database[0].connect(**database[1]), tmp_path,
        challengers=['Control'], runners=['alpha'])
    harness.service.config.defaults = replace(harness.service.config.defaults,
        execution_sequence_from='2000-01-01T00:00:00Z', per_icp_cost_policy=True,
        rewards_enabled=False, benchmark_icp_count=2)
    harness.service.config.daily_icp_source = lambda **kwargs: {
        'status': 'ready', 'set_id': kwargs['set_id'], 'icps': daily_icps()[:2]}
    _enable_verified_proxy_runtime(harness)
    monkeypatch.setattr(flows, 'deterministic_scorer', _baseline_first_judge)
    try:
        assert flows._start_round(harness, day=23, epoch=62530) == 2
        harness.clock.advance_to(harness.schedule()['stage_1_start'])
        harness.advance_until('scored', runners=1)
        before = harness.service.store.list_runs(harness.round_id)
        assert all(r['status'] == 'accepted' for r in before if r['kind'] == 'score')
        harness.clock.advance_to(service._iso(service._parse_iso(harness.schedule()['publication_deadline']) + timedelta(seconds=1)))
        assert harness.service.publish(harness.round_id)['status'] == 'ok'
        row = harness.service.store.get_round(harness.round_id)
        assert row['status'] == 'published'
        assert row['publication_doc']['king_decision']
        assert harness.service.store.list_runs(harness.round_id) == before
        for participant in row['participants']:
            public = harness.service.public_results(harness.round_id, participant['submission_id'])
            assert public['submission_scores']['final'] is not None
    finally:
        harness.service.store.close()


def test_other_round_open_head_does_not_block_this_round(database):
    rid, _ = prepared(database)
    with database[0].connect(**database[1]) as conn, conn.cursor() as cur:
        add_head(cur, rid, 'reservation')
        cur.execute('SET LOCAL session_replication_role=replica')
        cur.execute("UPDATE public.lab_arena_ledger SET round_id='another-round' WHERE round_id=%s", (rid,))
        cur.execute('SET LOCAL session_replication_role=origin')
        assert cancel(cur, rid)['status'] == 'cancelled'


def test_existing_non_deadline_cancellation_reason_retains_its_policy(database):
    rid, _ = prepared(database)
    with database[0].connect(**database[1]) as conn, conn.cursor() as cur:
        assert queue.claim(cur, rid)['status'] == 'leased'
        assert cancel(cur, rid, 'operator')['status'] == 'cancelled'


def test_completion_holding_round_share_finishes_before_safe_cancel(database):
    rid, _ = prepared(database)
    company = queue.revision.company
    store = ArenaStore(PsycopgTransport(lambda: database[0].connect(**database[1])))
    sig = 'public.lab_arena_complete_attempt_v3(text,text,jsonb,text,text,text,jsonb,text)'
    try:
        lease, token, _, _ = company.original.claim(store, rid, company.original.hotkey('deadline530-race'), excluded=[company.original.hotkey('deadline530-race')])
        refs = store.get_run(lease['run_id'])['company_judgment_refs']
        with database[0].connect(**database[1]) as conn, conn.cursor() as cur:
            cur.execute('SELECT pg_get_functiondef(%s::regprocedure)', (sig,))
            source = cur.fetchone()[0]
            marker = '    WHERE round_row.round_id=v_source_round_id FOR SHARE;'
            assert source.count(marker) == 1
            cur.execute(source.replace(marker, marker+'\n  PERFORM pg_advisory_xact_lock(7530530);'))
        with database[0].connect(**database[1]) as blocker, blocker.cursor() as lock, ThreadPoolExecutor(max_workers=2) as pool:
            lock.execute('SELECT pg_advisory_xact_lock(7530530)')
            completed = pool.submit(company.complete_subset, store, lease, token, refs)
            until = time.monotonic()+5
            with database[0].connect(**database[1]) as conn, conn.cursor() as cur:
                while True:
                    if completed.done():
                        assert completed.result()['status'] == 'accepted', 'completion ended before lock boundary'
                        pytest.fail('completion bypassed its instrumented v3 boundary')
                    cur.execute("SELECT EXISTS(SELECT 1 FROM pg_locks WHERE locktype='advisory' AND objid=7530530 AND NOT granted)")
                    if cur.fetchone()[0]:
                        break
                    assert time.monotonic() < until
                    time.sleep(.01)
            def fail():
                with database[0].connect(**database[1]) as conn, conn.cursor() as cur:
                    cur.execute("SET LOCAL statement_timeout='5s'")
                    return cancel(cur, rid)
            cancelled = pool.submit(fail)
            time.sleep(.05)
            assert not cancelled.done()
            blocker.commit()
            assert completed.result(timeout=5)['status'] == 'accepted'
            assert cancelled.result(timeout=5)['status'] == 'cancelled'
        assert store.get_run(lease['run_id'])['status'] == 'accepted'
    finally:
        with database[0].connect(**database[1]) as conn, conn.cursor() as cur:
            cur.execute(source)
        store.close()


def test_uncommitted_open_call_under_round_share_is_not_missed(database):
    rid, _ = prepared(database)
    with database[0].connect(**database[1]) as reserving, reserving.cursor() as cur, ThreadPoolExecutor(max_workers=1) as pool:
        # The canonical lease validator holds round SHARE for admission. Isolate
        # that exact boundary with a direct ledger fixture, without a provider.
        cur.execute('SELECT 1 FROM public.lab_arena_rounds WHERE round_id=%s FOR SHARE', (rid,))
        add_head(cur, rid, 'reservation')
        def fail():
            with database[0].connect(**database[1]) as conn, conn.cursor() as q:
                q.execute("SET LOCAL statement_timeout='5s'")
                return cancel(q, rid)
        waiting = pool.submit(fail)
        time.sleep(.05)
        assert not waiting.done()
        reserving.commit()
        assert waiting.result(timeout=5)['reason'] == 'evaluation_draining'


def test_non_superuser_apply_replay_preserves_rows_metadata_and_shared_helpers(database):
    rid, _ = prepared(database)
    with database[0].connect(**database[1]) as conn, conn.cursor() as cur:
        originals = {key: definition(cur, key) for key in SIGNATURES}
        for key, post in originals.items():
            cur.execute(preimage(key, post))
        cur.execute("SELECT to_jsonb(p)-'prosrc' FROM pg_proc p WHERE oid IN (%s::regprocedure,%s::regprocedure) ORDER BY oid", tuple(SIGNATURES.values()))
        security = cur.fetchall()
        cur.execute('SELECT pg_get_functiondef(%s::regprocedure),pg_get_functiondef(%s::regprocedure)', (prior.EXPIRY, prior.fast.CONTINUATION))
        shared = cur.fetchone()
        before = snapshot(cur, rid)
        cur.execute('CREATE ROLE deadline530_migrator NOSUPERUSER INHERIT')
        cur.execute('GRANT lab_arena_owner TO deadline530_migrator')
        cur.execute('GRANT USAGE,CREATE ON SCHEMA public TO deadline530_migrator')
        cur.execute('SET ROLE deadline530_migrator')
        cur.execute(SQL.read_text())
        cur.execute(SQL.read_text())
        cur.execute('RESET ROLE')
        assert snapshot(cur, rid) == before
        assert {key: definition(cur, key) for key in SIGNATURES} == originals
        cur.execute("SELECT to_jsonb(p)-'prosrc' FROM pg_proc p WHERE oid IN (%s::regprocedure,%s::regprocedure) ORDER BY oid", tuple(SIGNATURES.values()))
        assert cur.fetchall() == security
        cur.execute('SELECT pg_get_functiondef(%s::regprocedure),pg_get_functiondef(%s::regprocedure)', (prior.EXPIRY, prior.fast.CONTINUATION))
        assert cur.fetchone() == shared


def test_wide_expired_queue_extracts_deadline_once_without_changing_outcome(database):
    rid, _ = prepared(database, offset=-1, wide=True)
    with database[0].connect(**database[1]) as conn, conn.cursor() as cur:
        # Clone representative frozen pending/source identities as in the
        # existing 519 large-queue fixture; no payload is a production body.
        cur.execute("SELECT to_jsonb(r) FROM public.lab_arena_runs r WHERE round_id=%s AND kind='score' AND status='pending' ORDER BY icp_position DESC LIMIT 1", (rid,))
        template = cur.fetchone()[0]
        cur.execute('SET LOCAL session_replication_role=replica')
        for index in range(158):
            clone = dict(template)
            clone.update(run_id=rid+':wide-score-'+str(index)+':2', assignment_id=rid+':wide-score-'+str(index))
            cur.execute('INSERT INTO public.lab_arena_runs SELECT (jsonb_populate_record(NULL::public.lab_arena_runs,%s::jsonb)).*', (json.dumps(clone),))
        cur.execute('SET LOCAL session_replication_role=origin')
        cur.execute("SELECT count(*) FROM public.lab_arena_runs WHERE round_id=%s AND status='pending'", (rid,))
        assert cur.fetchone()[0] == 160
        cur.execute('SELECT octet_length(configuration_doc::text) FROM public.lab_arena_rounds WHERE round_id=%s', (rid,))
        width = cur.fetchone()[0]
        assert width > 1_700_000
        source = definition(cur, 'claim')
        scalar = 'v_publication_deadline::TIMESTAMPTZ'
        repeated = "(v_round.configuration_doc->'schedule'->>'publication_deadline')::TIMESTAMPTZ"
        assert source.count(scalar) == 1
        before = snapshot(cur, rid)
        timings = {}
        try:
            for version in ('repeated', 'scalar'):
                cur.execute(source.replace(scalar, repeated) if version == 'repeated' else source)
                cur.execute("SET LOCAL statement_timeout='8s'")
                started = time.perf_counter()
                assert queue.claim(cur, rid)['status'] == 'no_pending'
                timings[version] = time.perf_counter()-started
                cur.execute('RESET ROLE')
                assert snapshot(cur, rid) == before
            print('DEADLINE530_WIDE_QUEUE', {'pending': 160, 'config_bytes': width, 'seconds': timings}, flush=True)
            # A generous resource bound catches wide-JSON work accidentally
            # returned to every rejected candidate, without a speed-ratio gate.
            assert timings['scalar'] < 2
        finally:
            cur.execute(source)


def test_existing_request_replay_after_deadline_never_allocates_another_lease(database):
    rid, _ = prepared(database, offset=1)
    company = queue.revision.company
    store = ArenaStore(PsycopgTransport(lambda: database[0].connect(**database[1])))
    try:
        runner = company.original.hotkey('deadline530-replay')
        lease, token, request_id, _ = company.original.claim(store, rid, runner, excluded=[runner])
        assert lease['status'] == 'leased'
        before = store.get_run(lease['run_id'])
        time.sleep(1.1)
        replay, _, _, _ = company.original.claim(store, rid, runner,
            excluded=[runner], token=token, request_id=request_id)
        assert replay == lease and store.get_run(lease['run_id']) == before
        fresh, _, _, _ = company.original.claim(store, rid, runner,
            parallelism=7, excluded=[runner])
        assert fresh['status'] == 'no_pending'
    finally:
        store.close()


def test_natural_expiry_can_close_after_deadline_without_retrying_new_work(database):
    rid, _ = prepared(database, offset=1)
    store = ArenaStore(PsycopgTransport(lambda: database[0].connect(**database[1])))
    try:
        with database[0].connect(**database[1]) as conn, conn.cursor() as cur:
            lease = queue.claim(cur, rid)
            assert lease['status'] == 'leased'
        time.sleep(1.1)
        with database[0].connect(**database[1]) as conn, conn.cursor() as cur:
            cur.execute('SET LOCAL session_replication_role=replica')
            cur.execute("UPDATE public.lab_arena_runs SET lease_expires_at=clock_timestamp()-interval '1 second' WHERE run_id=%s", (lease['run_id'],))
            cur.execute('SET LOCAL session_replication_role=origin')
        assert store.expire_leases(rid)['expired'] == 1
        assert store.get_run(lease['run_id'])['terminal_cause'] == 'lease_expired'
        with database[0].connect(**database[1]) as conn, conn.cursor() as cur:
            runner = queue.revision.company.original.hotkey('ordered520')
            cur.execute('SELECT public.lab_arena_claim_assignment(%s,%s,7,8,%s,%s,%s,%s,4500)',
                (rid, runner, [], 'b'*32, 'sha256:'+'b'*64, 'sha256:'+'c'*64))
            assert cur.fetchone()[0]['status'] == 'no_pending'
            assert cancel(cur, rid)['status'] == 'cancelled'
    finally:
        store.close()
