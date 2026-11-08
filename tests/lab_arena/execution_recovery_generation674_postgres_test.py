"""Recover real old-phase executions without crossing a grading generation."""
from pathlib import Path

import pytest

from tests.lab_arena.operator_execution_recovery672_postgres_test import *
from tests.lab_arena import operator_execution_recovery672_postgres_test as original

CORRECTION = Path(__file__).parents[2] / 'scripts/674-lab-arena-execution-recovery-phase-generation.sql'
SIGNATURE = 'public.lab_arena_grant_execution_infrastructure_recovery_v1(text,text,text,text,text,boolean)'
OLD_GRANT = None


@pytest.fixture(scope='module')
def database():
    global OLD_GRANT
    gen = original.database.__wrapped__()
    pg, dsn = next(gen)
    previous = original.SQL
    try:
        with pg.connect(**dsn) as c, c.cursor() as q:
            OLD_GRANT = original.scalar(q, 'SELECT pg_get_functiondef(%s::regprocedure)', (SIGNATURE,))
            q.execute(CORRECTION.read_text())
        # The inherited replay controls exercise the current additive migration.
        # Historical672 stays immutable and must not be reapplied over674.
        original.SQL = CORRECTION
        yield pg, dsn
    finally:
        original.SQL = previous
        gen.close()


@pytest.fixture
def case(database):
    gen = original.case.__wrapped__(database)
    p = next(gen)
    try:
        yield p
    finally:
        gen.close()


@pytest.fixture
def old_phase_case(case):
    p = case
    try:
        with p.c.cursor() as q:
            q.execute('SET LOCAL session_replication_role=replica')
            q.execute("UPDATE public.lab_arena_runs SET stage_generation=4 WHERE round_id=%s AND kind='execute'AND status='failed'", (original.ROUND,))
        p.c.commit()
        yield p
    finally:
        p.c.rollback()


def test_old_phase_fails_before_fix_and_current_child_identity_matches_native_row(old_phase_case):
    p = old_phase_case
    parent = p.parents[0]
    with p.c.cursor() as q:
        before = original.scalar(q, 'SELECT to_jsonb(a)FROM public.lab_arena_runs a WHERE run_id=%s', (parent,))
        assert before['stage_generation'] == 4
        assert original.scalar(q, 'SELECT stage_generation FROM public.lab_arena_rounds WHERE round_id=%s', (original.ROUND,)) == 7
        digest = 'sha256:' + original.scalar(q, "SELECT encode(extensions.digest(to_jsonb(a)::text,'sha256'),'hex')FROM public.lab_arena_runs a WHERE run_id=%s", (parent,))
        q.execute('SAVEPOINT before674')
        q.execute(OLD_GRANT)
        with pytest.raises(p.pg.Error, match='expired latest parent'):
            q.execute('SELECT ' + SIGNATURE.split('(')[0] + '(%s,%s,%s,%s,%s,true)',
                (parent, digest, 'failing-before-phase-regression', '674-owner', 'verified-source'))
        q.execute('ROLLBACK TO SAVEPOINT before674')
    p.c.commit()
    receipt = original.grant(p)
    child_id = receipt['child_identity']['run_id']
    assert receipt['child_identity']['stage_generation'] == 7
    with p.c.cursor() as q:
        assert original.scalar(q, 'SELECT stage_generation FROM public.lab_arena_runs WHERE run_id=%s', (child_id,)) == 7
        assert original.scalar(q, 'SELECT to_jsonb(a)FROM public.lab_arena_runs a WHERE run_id=%s', (parent,)) == before
        assert original.scalar(q, 'SELECT public.lab_arena__operator_execute_recovery_valid_v1(r,a)FROM public.lab_arena_rounds r JOIN public.lab_arena_runs a USING(round_id)WHERE a.run_id=%s', (child_id,)) is True
    assert original.grant(p)['idempotent'] is True
    lease = original.claim(p)
    assert lease['status'] == 'leased' and lease['run_id'] == child_id


@pytest.mark.parametrize('fault', ['future_phase', 'different_floor', 'claim_generation', 'claim_status', 'missing_token'])
def test_old_phase_does_not_weaken_floor_or_expired_claim_identity(old_phase_case, fault):
    p = old_phase_case
    with p.c.cursor() as q:
        q.execute('SET LOCAL session_replication_role=replica')
        if fault == 'future_phase':
            q.execute('UPDATE public.lab_arena_runs SET stage_generation=8 WHERE run_id=%s', (p.parents[0],))
        elif fault == 'different_floor':
            q.execute('UPDATE public.lab_arena_rounds SET scoring_generation_floor=1 WHERE round_id=%s', (original.ROUND,))
        elif fault == 'claim_generation':
            q.execute("UPDATE public.lab_arena_runs SET claim_response=jsonb_set(claim_response,'{lease_generation}','1'::jsonb)WHERE run_id=%s", (p.parents[0],))
        elif fault == 'claim_status':
            q.execute("UPDATE public.lab_arena_runs SET claim_response=jsonb_set(claim_response,'{status}','\"pending\"'::jsonb)WHERE run_id=%s", (p.parents[0],))
        else:
            q.execute('UPDATE public.lab_arena_runs SET lease_token_hash=NULL WHERE run_id=%s', (p.parents[0],))
    p.c.commit()
    before = original.snapshot(p)
    with pytest.raises(p.pg.Error, match='expired latest parent'):
        original.grant(p)
    p.c.rollback()
    assert original.snapshot(p) == before


def test_all_nine_real_old_phase_parents_preserve_all_accepted_work(old_phase_case):
    original.test_nine_observed_states_dry_run_idempotence_and_preservation(old_phase_case)


@pytest.mark.parametrize('outcome', ['accepted', 'provider_error', 'model_error', 'expired', 'host'])
def test_old_phase_successors_use_native_lifecycle_and_remain_finite(old_phase_case, outcome):
    original.test_real_child_lifecycle_never_recurses(old_phase_case, outcome)


def test_all_nine_old_phase_successors_complete_then_open_native_judgments(old_phase_case):
    original.test_real_completion_opens_only_new_judgments_and_preserves_old_bindings(old_phase_case)
