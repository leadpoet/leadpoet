"""The original recovered execution remains the source of ordinary judge retries."""
import json
from pathlib import Path

import pytest

from tests.lab_arena import bounded_baseline_recovery470_postgres_test as original
from tests.lab_arena import general_judge_continuation472_postgres_test as deployed

SQL = Path(__file__).parents[2] / 'scripts/473-lab-arena-recovered-execution-judge-continuation.sql'
SIGNATURE = 'public.lab_arena__judge_continuation_v1(public.lab_arena_rounds,public.lab_arena_runs)'


@pytest.fixture(scope='module')
def database():
    generator = deployed.database.__wrapped__()
    db = next(generator)
    try:
        with db[0].connect(**db[1]) as conn, conn.cursor() as cursor:
            signatures = (SIGNATURE, 'public.lab_arena_record_run_scores(text,smallint,jsonb)')
            cursor.execute("SELECT oid,md5(pg_get_functiondef(oid)) FROM pg_proc WHERE pronamespace='public'::regnamespace AND proname LIKE 'lab_arena_%%' AND prokind='f' AND oid<>ALL(%s::regprocedure[]) ORDER BY oid", (list(signatures),))
            unchanged_functions = cursor.fetchall()
            cursor.execute('SELECT oid,proowner,proacl,prosecdef,provolatile,proconfig FROM pg_proc WHERE oid=ANY(%s::regprocedure[]) ORDER BY oid', (list(signatures),))
            authority = cursor.fetchall()
            cursor.execute(SQL.read_text())
            cursor.execute('SELECT pg_get_functiondef(oid) FROM pg_proc WHERE oid=ANY(%s::regprocedure[]) ORDER BY oid', (list(signatures),))
            first = cursor.fetchall()
            cursor.execute(SQL.read_text())
            cursor.execute('SELECT pg_get_functiondef(oid) FROM pg_proc WHERE oid=ANY(%s::regprocedure[]) ORDER BY oid', (list(signatures),))
            assert cursor.fetchall() == first
            cursor.execute('SELECT oid,proowner,proacl,prosecdef,provolatile,proconfig FROM pg_proc WHERE oid=ANY(%s::regprocedure[]) ORDER BY oid', (list(signatures),))
            assert cursor.fetchall() == authority
            cursor.execute("SELECT oid,md5(pg_get_functiondef(oid)) FROM pg_proc WHERE pronamespace='public'::regnamespace AND proname LIKE 'lab_arena_%%' AND prokind='f' AND oid<>ALL(%s::regprocedure[]) ORDER BY oid", (list(signatures),))
            assert cursor.fetchall() == unchanged_functions
            for role in ('anon', 'authenticated', 'service_role', 'lab_arena_service'):
                cursor.execute('SELECT has_function_privilege(%s,%s,\'EXECUTE\')', (role, SIGNATURE))
                assert cursor.fetchone() == (False,)
        yield db
    finally:
        generator.close()


def recovered_judge(conn):
    assignment, cfg, source = original.accepted_recovery(conn)
    assert original.recover(conn, [original.scope_item(conn, source['run_id'])])['queued'] == 1
    with conn.cursor() as cursor:
        cursor.execute('SET LOCAL session_replication_role=replica')
        cursor.execute("UPDATE public.lab_arena_rounds SET status='stage2_scoring',stage_generation=7")
        cursor.execute('SET LOCAL session_replication_role=origin')
    conn.commit()
    assert original.recover(conn)['queued'] == 0
    lease = original.claim(conn, suffix='b')
    assert (lease['kind'], lease['attempt'], lease['scored_run_id']) == ('score', 1, source['run_id'])
    return assignment, cfg, source, lease


def cool(conn):
    with conn.cursor() as cursor:
        cursor.execute('SET LOCAL session_replication_role=replica')
        cursor.execute("UPDATE public.lab_arena_runs SET created_at=clock_timestamp()-interval '20 minutes' WHERE status='pending'")
        cursor.execute('SET LOCAL session_replication_role=origin')
    conn.commit()


def complete(conn, lease, *, accepted=False):
    cause = 'accepted' if accepted else 'judge_error'
    return original.rpc(conn, 'lab_arena_complete_attempt(%s,%s,%s::jsonb,%s,%s)',
        (lease['run_id'], original.TOKEN, json.dumps({'terminal_status': cause}), cause,
         'accepted-judge-output' if accepted else ''))


def test_original_plan_absent_source_three_continues_through_four(database):
    with database[0].connect(**database[1]) as conn:
        assignment, cfg, source, lease = recovered_judge(conn)
        with conn.cursor() as cursor:
            cursor.execute('SELECT to_jsonb(r) FROM public.lab_arena_runs r WHERE kind=\'execute\' ORDER BY run_id')
            accepted_work = cursor.fetchall()
        for attempt in (1, 2, 3):
            assert lease['attempt'] == attempt
            result = complete(conn, lease)
            assert result['confirmation_attempt'] == attempt + 1
            cool(conn)
            lease = original.claim(conn, suffix=chr(ord('b') + attempt))
            assert (lease['status'], lease['scored_run_id']) == ('leased', source['run_id'])
        assert lease['attempt'] == 4
        assert complete(conn, lease, accepted=True)['status'] == 'accepted'
        assert complete(conn, lease, accepted=True)['idempotent'] is True
        assert original.rpc(conn, 'lab_arena_expire_leases(%s)', (original.ROUND,))['retried'] == 0
        with conn.cursor() as cursor:
            cursor.execute('SELECT to_jsonb(r) FROM public.lab_arena_runs r WHERE kind=\'execute\' ORDER BY run_id')
            assert cursor.fetchall() == accepted_work
            cursor.execute('SELECT configuration_doc,stage1_scoring_plan_doc FROM public.lab_arena_rounds')
            assert cursor.fetchone() == (cfg, {'unchanged': 'frozen-plan'})
            cursor.execute("SELECT count(*),max(attempt) FROM public.lab_arena_runs WHERE assignment_id=%s", (assignment + ':score',))
            assert cursor.fetchone() == (4, 4)


@pytest.mark.parametrize('fault', [
    'foreign_source', 'source_one', 'forged_baseline', 'miner', 'noncontiguous',
    'credential_parent', 'source_unaccepted', 'participant', 'wrong_image',
    'wrong_input', 'old_rollout',
])
def test_alternate_witness_does_not_authorize_other_work(database, fault):
    with database[0].connect(**database[1]) as conn:
        _, _, source, lease = recovered_judge(conn)
        with conn.cursor() as cursor:
            cursor.execute('SET LOCAL session_replication_role=replica')
            if fault == 'foreign_source':
                cursor.execute("UPDATE public.lab_arena_runs SET scored_run_id='foreign' WHERE kind='score'")
            elif fault == 'source_one':
                cursor.execute("UPDATE public.lab_arena_runs SET scored_run_id=replace(scored_run_id,':3',':1') WHERE kind='score'")
            elif fault in ('forged_baseline', 'miner'):
                cursor.execute("UPDATE public.lab_arena_submissions SET is_king=false")
                if fault == 'miner':
                    cursor.execute("UPDATE public.lab_arena_rounds SET participants=jsonb_set(participants,'{0,is_king}','false')")
            elif fault in ('noncontiguous', 'credential_parent'):
                cursor.execute("UPDATE public.lab_arena_runs SET terminal_cause='credential_error',result_doc='{\"terminal_status\":\"credential_error\"}' WHERE kind='execute' AND attempt=2")
                if fault == 'noncontiguous':
                    cursor.execute("DELETE FROM public.lab_arena_runs WHERE kind='execute' AND attempt=1")
            elif fault == 'source_unaccepted':
                cursor.execute("UPDATE public.lab_arena_runs SET status='failed' WHERE run_id=%s", (source['run_id'],))
            elif fault == 'participant':
                cursor.execute("UPDATE public.lab_arena_rounds SET participants='[]'")
            elif fault == 'wrong_image':
                cursor.execute("UPDATE public.lab_arena_runs SET judgment_scope_doc=jsonb_set(judgment_scope_doc,'{scorer_image_digest}','\"wrong\"') WHERE kind='score'")
            elif fault == 'wrong_input':
                cursor.execute("UPDATE public.lab_arena_runs SET judgment_input_hash='sha256:'||repeat('f',64) WHERE kind='score'")
            else:
                cursor.execute("UPDATE public.lab_arena_rounds SET evaluation_date='2026-10-13'")
            cursor.execute('SET LOCAL session_replication_role=origin')
            cursor.execute('SELECT public.lab_arena__judge_continuation_v1(round_row,run) FROM public.lab_arena_rounds round_row JOIN public.lab_arena_runs run USING(round_id) WHERE run.run_id=%s', (lease['run_id'],))
            assert cursor.fetchone() == (False,)
        conn.commit()
        result = complete(conn, lease)
        assert 'confirmation_attempt' not in result
        assert original.rpc(conn, 'lab_arena_expire_leases(%s)', (original.ROUND,))['retried'] == 0
        with conn.cursor() as cursor:
            cursor.execute("SELECT count(*) FROM public.lab_arena_runs WHERE kind='score'")
            assert cursor.fetchone() == (1,)


@pytest.mark.parametrize('fault', ['wrong_image', 'wrong_cache', 'wrong_input',
                                  'wrong_source', 'wrong_witness', 'old_rollout'])
def test_accepted_later_judge_cannot_project_unbound_score(database, fault):
    with database[0].connect(**database[1]) as conn:
        _, _, source, lease = recovered_judge(conn)
        for attempt in (1, 2, 3):
            assert complete(conn, lease)['confirmation_attempt'] == attempt + 1
            cool(conn)
            lease = original.claim(conn, suffix=chr(ord('b') + attempt))
        assert complete(conn, lease, accepted=True)['status'] == 'accepted'
        with conn.cursor() as cursor:
            cursor.execute('SET LOCAL session_replication_role=replica')
            cursor.execute("UPDATE public.lab_arena_rounds SET status='stage2_judged'")
            if fault == 'wrong_image':
                cursor.execute("UPDATE public.lab_arena_runs SET judgment_scope_doc=jsonb_set(judgment_scope_doc,'{scorer_image_digest}','\"wrong\"') WHERE run_id=%s", (lease['run_id'],))
            elif fault in ('wrong_cache', 'wrong_input'):
                column = 'judgment_cache_key' if fault == 'wrong_cache' else 'judgment_input_hash'
                cursor.execute("UPDATE public.lab_arena_runs SET " + column + "='sha256:'||repeat('f',64) WHERE run_id=%s", (lease['run_id'],))
            elif fault == 'wrong_source':
                cursor.execute("UPDATE public.lab_arena_runs SET scored_run_id=replace(scored_run_id,':3',':1') WHERE run_id=%s", (lease['run_id'],))
            elif fault == 'wrong_witness':
                cursor.execute("UPDATE public.lab_arena_submissions SET is_king=false")
            else:
                cursor.execute("UPDATE public.lab_arena_rounds SET evaluation_date='2026-10-13'")
            cursor.execute('SET LOCAL session_replication_role=origin')
            cursor.execute('SELECT to_jsonb(r) FROM public.lab_arena_runs r ORDER BY run_id')
            before = cursor.fetchall()
        conn.commit()
        scores = [{'run_id': source['run_id'], 'per_icp_score': 0,
                   'qualification_doc': {'companies': []}}]
        with pytest.raises(database[0].Error, match='late_baseline_score_invalid'):
            original.rpc(conn, 'lab_arena_record_run_scores(%s,1::smallint,%s::jsonb)',
                         (original.ROUND, json.dumps(scores)))
        conn.rollback()
        with conn.cursor() as cursor:
            cursor.execute('SELECT to_jsonb(r) FROM public.lab_arena_runs r ORDER BY run_id')
            assert cursor.fetchall() == before


def test_original_source_late_score_requires_unchanged_plan_membership(database):
    with database[0].connect(**database[1]) as conn:
        _, source_id, _ = deployed.prepare(conn)
        assert original.rpc(conn, 'lab_arena_expire_leases(%s)', (original.ROUND,))['retried'] == 1
        cool(conn)
        third = original.claim(conn, suffix='b')
        assert complete(conn, third)['confirmation_attempt'] == 4
        cool(conn)
        fourth = original.claim(conn, suffix='c')
        assert complete(conn, fourth, accepted=True)['status'] == 'accepted'
        with conn.cursor() as cursor:
            cursor.execute('SET LOCAL session_replication_role=replica')
            cursor.execute("UPDATE public.lab_arena_rounds SET status='stage2_judged',stage1_scoring_plan_doc='{\"work_items\":[]}'")
            cursor.execute('SET LOCAL session_replication_role=origin')
        conn.commit()
        with pytest.raises(database[0].Error, match='late_baseline_score_invalid'):
            original.rpc(conn, 'lab_arena_record_run_scores(%s,1::smallint,%s::jsonb)',
                (original.ROUND, json.dumps([{'run_id': source_id, 'per_icp_score': 0,
                    'qualification_doc': {'companies': []}}])))


@pytest.mark.parametrize('late_completion', [False, True])
def test_signed_http_source_three_judge_four_persists_and_publishes(database, tmp_path, monkeypatch, late_completion):
    """Use the original mixed recovery fixture, real HTTP Runner and score assembly."""
    from fastapi.testclient import TestClient
    from lab_arena import runtime, scoring
    from lab_arena.api import create_app
    from lab_arena.runner import HttpArenaApiClient
    from lab_arena.store import ArenaStore
    from tests.lab_arena import test_lab_arena_service_round as flows
    from tests.lab_arena.test_integrity_round import IntegrityHarness

    clients = []
    build = IntegrityHarness.build_service
    sandbox_run = flows.ModelSandbox.run_icp
    store_claim = ArenaStore.claim_assignment
    advance = IntegrityHarness.advance_until
    failed_reviews = []
    original_source_failures = []
    assembled = []

    def build_http(self):
        service = build(self)
        # The real creation clock is October 13; use 472's existing isolated
        # reward-disabled shadow rollout rather than changing its date fence.
        service.config.mode = 'shadow'
        client = TestClient(create_app(service))
        clients.append(client)
        self.api_factory = lambda: HttpArenaApiClient('http://localhost', client=client)
        return service

    def temporary_provider_error(self, spec, **kwargs):
        document = json.loads((spec.input_dir / runtime.INPUT_FILE_NAME).read_text())
        source_id = str(document.get('scored_run_id') or '')
        if (document.get('schema_version') == scoring.SCORING_INPUT_SCHEMA_VERSION
            and ':1:0:3' in source_id and len(failed_reviews) < 3):
            # The model/provider bridge still runs and records its normal cost.
            sandbox_run(self, spec, **kwargs)
            failed_reviews.append(source_id)
            failure = scoring.build_scoring_failure(source_id, 'judge_error', detail='provider error 503')
            return runtime.fake_result(exit_code=0, output_bytes=json.dumps(failure).encode())
        if (document.get('schema_version') == scoring.SCORING_INPUT_SCHEMA_VERSION
            and ':1:3:1' in source_id and not original_source_failures):
            # The initial accepted review predates the seeded recovery boundary.
            # Fail only the resumed third review, then accept its fourth review.
            with database[0].connect(**database[1]) as conn, conn.cursor() as cursor:
                cursor.execute("SELECT count(*) FROM public.lab_arena_runs WHERE scored_run_id=%s AND kind='score' AND attempt=3", (source_id,))
                resumed = cursor.fetchone()[0] == 1
            if resumed:
                sandbox_run(self, spec, **kwargs)
                original_source_failures.append(source_id)
                failure = scoring.build_scoring_failure(source_id, 'judge_error', detail='provider error 503')
                return runtime.fake_result(exit_code=0, output_bytes=json.dumps(failure).encode())
        return sandbox_run(self, spec, **kwargs)

    def elapsed_backoff(self, *args, **kwargs):
        # Disposable DB clock adapter: avoid wall-clock sleeps for 472 backoff.
        with database[0].connect(**database[1]) as conn:
            cool(conn)
        if late_completion:
            row = self.get_round(kwargs['round_id'])
            if row['status'] != 'scored':
                # Leave fourth reviews in the real backoff period until the
                # normal forward transition reaches scored.
                with database[0].connect(**database[1]) as conn, conn.cursor() as cursor:
                    cursor.execute('SET LOCAL session_replication_role=replica')
                    cursor.execute("UPDATE public.lab_arena_runs SET created_at=clock_timestamp() WHERE round_id=%s AND kind='score' AND attempt=4 AND status='pending'", (kwargs['round_id'],))
        return store_claim(self, *args, **kwargs)

    def assemble_at_scored(self, target, **kwargs):
        run_stage = self.run_stage_with_runners

        def finish_window_with_pending_reviews(count):
            run_stage(count)
            pending = [run for run in self.service.store.list_runs(self.round_id, kind='score')
                       if run['status'] == 'pending' and run['attempt'] == 4]
            if len(pending) == 2:
                # Current continuation waits while retry work remains. Reach
                # the real scheduled phase boundary before the late accepts.
                assert self.status() == 'stage2_scoring'
                self.clock.advance_to(self.schedule()['final_scoring_close'])

        if late_completion and target == 'scored':
            self.run_stage_with_runners = finish_window_with_pending_reviews
        try:
            result = advance(self, target, **kwargs)
        finally:
            self.run_stage_with_runners = run_stage
        if target == 'scored':
            row = self.service.store.get_round(self.round_id)
            source = next(r for r in self.service.store.list_runs(self.round_id, kind='execute')
                          if r['attempt'] == 3 and r['icp_position'] == 0)
            if late_completion:
                assert source['per_icp_score'] is None
                assert not self.service._baseline_final_ready(row)
                self.run_stage_with_runners(1)
            judge = next(r for r in self.service.store.list_runs(self.round_id, kind='score')
                         if r['scored_run_id'] == source['run_id'] and r['status'] == 'accepted')
            assert judge['attempt'] == 4
            original_judge = next(r for r in self.service.store.list_runs(self.round_id, kind='score')
                if r['icp_position'] == 3 and r['submission_id'] == source['submission_id'] and r['status'] == 'accepted')
            assert original_judge['attempt'] == 4
            self.service._record_recovered_judgment_scores(row)
            refreshed = self.service.store.get_run(source['run_id'])
            assert refreshed['per_icp_score'] == 80
            assembled.append(judge['run_id'])
        return result

    monkeypatch.setattr(IntegrityHarness, 'build_service', build_http)
    monkeypatch.setattr(IntegrityHarness, 'advance_until', assemble_at_scored)
    monkeypatch.setattr(flows.ModelSandbox, 'run_icp', temporary_provider_error)
    monkeypatch.setattr(ArenaStore, 'claim_assignment', elapsed_backoff)
    try:
        original.test_normal_signed_runner_recovers_only_missing_baseline_work_and_preserves_miners(
            database, tmp_path, monkeypatch, True)
        assert len(failed_reviews) == 3 and len(set(failed_reviews)) == 1
        assert len(original_source_failures) == 1
        assert len(assembled) == 1
    finally:
        for client in clients:
            client.close()
