"""Hermetic migration transition: real claims, cached retry, round publication."""
from pathlib import Path

import pytest

from lab_arena.store import ArenaStore,PsycopgTransport
from tests.lab_arena import score_cooldown_claim_index510_postgres_test as prior
from tests.lab_arena import test_lab_arena_service_round as flows


@pytest.fixture(scope='module')
def database():
    gen=prior.database.__wrapped__();db=next(gen)
    try:
        conn=db[0].connect(**db[1]);conn.autocommit=True
        try:
            with conn.cursor() as cur:
                cur.execute((Path(__file__).parents[2]/'scripts/510-lab-arena-score-cooldown-claim-index.sql').read_text())
                cur.execute("SELECT indisvalid AND indisready FROM pg_index WHERE indexrelid='public.lab_arena_runs_score_cooldown_claim_idx'::regclass")
                assert cur.fetchone()==(True,)
        finally:conn.close()
        yield db
    finally:gen.close()


def test_company_failure_retry_preserves_completed_authorities_and_finishes(database):
    store=ArenaStore(PsycopgTransport(lambda:database[0].connect(**database[1])))
    store._test_database=database
    try:
        rid='arena-2099-10-20-transition509'
        _,items=prior.company.open_new(store,round_id=rid,markers_by_submission=[['a','b','c']])
        refs=items[0]['company_judgment_refs']
        runner1=prior.original.hotkey('transition-one')
        first,token,_,_=prior.original.claim(store,rid,runner1,excluded=[runner1])
        assert first['status']=='leased' and len(first['company_judgment_cache']['misses'])==3
        assert prior.company.complete_subset(store,first,token,refs,limit=2,terminal='judge_error')['company_judgments_stored']==2
        failed=store.get_run(first['run_id']);assert failed['status']=='failed'
        saved=[store.get_company_judgment(ref['cache_key'],0) for ref in refs[:2]]
        assert all(saved) and store.get_company_judgment(refs[2]['cache_key'],0) is None
        with database[0].connect(**database[1]) as conn,conn.cursor() as cur:
            cur.execute('SET LOCAL session_replication_role=replica')
            cur.execute("UPDATE public.lab_arena_runs SET created_at=clock_timestamp()-interval '20 minutes' WHERE round_id=%s AND status='pending'",(rid,))
            cur.execute('SET LOCAL session_replication_role=origin')
        runner2=prior.original.hotkey('transition-two')
        retry,token2,_,_=prior.original.claim(store,rid,runner2,excluded=[runner2])
        assert retry['status']=='leased' and retry['attempt']==2
        assert len(retry['company_judgment_cache']['hits'])==2
        assert len(retry['company_judgment_cache']['misses'])==1
        assert prior.company.complete_subset(store,retry,token2,refs)['company_judgments_stored']==1
        assert store.get_run(retry['run_id'])['status']=='accepted'
        assert store.get_run(first['run_id'])==failed
        assert [store.get_company_judgment(ref['cache_key'],0) for ref in refs[:2]]==saved
        assert store.get_company_judgment(refs[2]['cache_key'],0) is not None
        with database[0].connect(**database[1]) as conn,conn.cursor() as cur:
            cur.execute('SELECT count(*) FROM public.lab_arena_company_judgment_reservations WHERE run_id=ANY(%s)',([first['run_id'],retry['run_id']],))
            assert cur.fetchone()==(0,)
    finally:store.close()


def test_latest_schema_retains_complete_normal_round_aggregate_and_publication(database,tmp_path,monkeypatch):
    # Use the current complete-baseline fixture, rather than an older partial
    # baseline rehearsal that predates the final-baseline authority guard.
    from dataclasses import replace
    from tests.lab_arena.test_integrity_round import IntegrityHarness
    from tests.lab_arena.baseline_scored_first_postgres_test import _baseline_first_judge,_enable_verified_proxy_runtime
    from tests.lab_arena.icp_fixtures import daily_icps
    with database[0].connect(**database[1]) as conn,conn.cursor() as cur:
        cur.execute('SET LOCAL session_replication_role=replica')
        cur.execute('TRUNCATE public.lab_arena_rounds RESTART IDENTITY CASCADE')
        cur.execute('SET LOCAL session_replication_role=origin')
    harness=IntegrityHarness(lambda:database[0].connect(**database[1]),tmp_path,challengers=['Control'],runners=['alpha'])
    harness.service.config.defaults=replace(harness.service.config.defaults,
        execution_sequence_from='2000-01-01T00:00:00Z',per_icp_cost_policy=True,
        rewards_enabled=False,benchmark_icp_count=2)
    harness.service.config.daily_icp_source=lambda **kwargs:{'status':'ready','set_id':kwargs['set_id'],'icps':daily_icps()[:2]}
    _enable_verified_proxy_runtime(harness)
    monkeypatch.setattr(flows,'deterministic_scorer',_baseline_first_judge)
    try:
        assert flows._start_round(harness,day=20,epoch=62420)==2
        harness.clock.advance_to(harness.schedule()['stage_1_start'])
        harness.advance_until('published',runners=1)
        row=harness.service.store.get_round(harness.round_id)
        assert row['status']=='published' and len(row['publication_doc']['final_ranking'])==2
        for participant in row['participants']:
            public=harness.service.public_results(harness.round_id,participant['submission_id'])
            assert public['submission_scores']['final'] is not None
            assert len(public['scores']['stage_1'])+len(public['scores']['stage_2'])==2
        assert all(r['status']=='accepted' for r in harness.service.store.list_runs(harness.round_id,kind='score'))
        flows.assert_canary_absent(harness,lambda:database[0].connect(**database[1]))
    finally:harness.service.store.close()
