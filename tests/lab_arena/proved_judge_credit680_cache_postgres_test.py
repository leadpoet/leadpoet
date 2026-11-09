"""The native successor retains completed company evidence without paid rejudging."""
from pathlib import Path

import pytest

from lab_arena import scorer_entrypoint, scoring
from tests.lab_arena import operator_scorer_bug_recovery666_postgres_test as base
from tests.lab_arena import retained_transient_provider_proof673_postgres_test as proof
from tests.lab_arena.company_judgments_test import _company, _input

ROOT = Path(__file__).parents[2]
database = base.database
prepared = base.prepared
prepared_company = base.prepared_company
case = base.case


@pytest.fixture
def preserved_case(case):
    p = case
    with p.c.cursor() as q:
        original = base.old.scalar(q, 'SELECT pg_get_functiondef(%s::regprocedure)', ('public.lab_arena__score_attempt_count_v1(public.lab_arena_runs)',))
    p.c.commit()
    try:
        yield p
    finally:
        p.c.rollback()
        with p.c.cursor() as q:
            q.execute(original)
        p.c.commit()


def test_native_expiry_successor_reuses_completed_company_judgments(preserved_case, monkeypatch):
    p = preserved_case
    child = p.parent['assignment_id'] + ':' + str(p.parent['attempt'] + 1)
    with p.c.cursor() as q:
        # Seed a historical interrupted claim only in this disposable DB.
        q.execute('ALTER TABLE public.lab_arena_runs DISABLE TRIGGER lab_arena_runs_terminal')
        q.execute('ALTER TABLE public.lab_arena_runs DISABLE TRIGGER lab_arena_runs_reviewer_runtime')
        q.execute('SET LOCAL session_replication_role=replica')
        q.execute("UPDATE public.lab_arena_runs SET status='leased',terminal_cause=NULL,result_doc=NULL,output_ref=NULL,runner_hotkey='runner',claim_request_id='request',claim_request_hash='sha256:'||repeat('c',64),lease_generation=1,lease_token_hash=%s,lease_expires_at=clock_timestamp()-interval '1 second',claim_response=jsonb_build_object('status','leased','lease_generation',1,'lease_expires_at',clock_timestamp()-interval '2 seconds') WHERE run_id=%s", (base.old.TOKEN, p.parent['run_id']))
        q.execute('SET LOCAL session_replication_role=origin')
        q.execute('ALTER TABLE public.lab_arena_runs ENABLE ALWAYS TRIGGER lab_arena_runs_terminal')
        q.execute('ALTER TABLE public.lab_arena_runs ENABLE ALWAYS TRIGGER lab_arena_runs_reviewer_runtime')
        source = proof.SQL.read_text()
        helper = source.split('CREATE OR REPLACE FUNCTION ', 1)[1].split('DO $patch673$', 1)[0]
        q.execute('GRANT CREATE ON SCHEMA public TO lab_arena_owner')
        q.execute('CREATE OR REPLACE FUNCTION ' + helper)
        q.execute('REVOKE CREATE ON SCHEMA public FROM lab_arena_owner')
    p.c.commit()
    proof.receipt(p)
    base.old.rpc(p, 'lab_arena_expire_leases(%s)', (base.old.ROUND,))
    with p.c.cursor() as q:
        assert base.old.scalar(q, 'SELECT count(*) FROM public.lab_arena_runs WHERE run_id=%s', (child,)) == 0
    before = base.snapshot(p)
    with p.c.cursor() as q:
        q.execute((ROOT / 'scripts/680-lab-arena-proved-infrastructure-judge-attempt-credit.sql').read_text())
        q.execute("SELECT public.lab_arena__score_attempt_count_v1(r) FROM public.lab_arena_runs r WHERE run_id=%s", (p.parent['run_id'],))
        assert q.fetchone()[0] == (5 if p.organizer else 4)
    p.c.commit()
    retried = base.old.rpc(p, 'lab_arena_expire_leases(%s)', (base.old.ROUND,))['retried']
    if p.organizer:
        # Existing explicitly finite organizer jobs do not gain recursion.
        assert retried == 0
        return
    assert retried >= 1  # The existing whole-round sweep can also progress other fixture work.
    with p.c.cursor() as q:
        q.execute("UPDATE public.lab_arena_runs SET created_at=clock_timestamp()-interval '1 hour' WHERE run_id=%s", (child,))
        cfg = base.old.scalar(q, 'SELECT configuration_doc FROM public.lab_arena_rounds WHERE round_id=%s', (base.old.ROUND,))
    p.c.commit()
    base.old.release_guard(p)
    lease = base.old.claim(p)
    assert lease['run_id'] == child
    assert len(lease['company_judgment_cache']['hits']) == 2
    assert lease['company_judgment_cache']['misses'] == []
    calls = []

    def scorer(companies, icp):
        calls.append(companies)
        raise AssertionError('a completed company must not trigger paid rejudging')

    scorer.integrity_policy = True
    scorer.company_quality = False
    monkeypatch.setattr(scoring, 'lab_scorer', lambda *args, **kwargs: scorer)
    document = scoring.build_scoring_input(
        scored_run_id=lease['scored_run_id'], icp=_input()['icp'],
        companies=[_company('A'), _company('B')], policy=cfg['scorer_policy'],
        evaluation_date='2026-11-06')
    document['company_judgment_cache'] = lease['company_judgment_cache']
    # Match the isolated scorer process rather than inheriting CI provider keys.
    # Keep the real frozen-policy and placeholder-credential checks active.
    with monkeypatch.context() as process:
        process.setattr(scorer_entrypoint.os, 'environ', {})
        output = scorer_entrypoint.score_input(document)
    assert 'failure' not in output
    assert len(output['breakdowns']) == 2
    assert calls == []
    after = base.snapshot(p)
    assert after['lab_arena_company_judgments'] == before['lab_arena_company_judgments']
    assert after['lab_arena_signal_judgments'] == before['lab_arena_signal_judgments']
    assert after['lab_arena_ledger'] == before['lab_arena_ledger']
