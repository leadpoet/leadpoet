"""Native sandbox timeouts use the existing finite owner recovery authority.

Only disposable PostgreSQL fixture setup changes a historical terminal row.
Claims and completions use the real service RPC; no provider I/O occurs.
"""
import hashlib
import json
from pathlib import Path

import pytest
from tests.lab_arena import provider_terminal_scorer_correction691_postgres_test as previous

old, base, finite, prior = previous.old, previous.base, previous.finite, previous.prior
COMMON = previous.COMMON
SQL = Path(__file__).parents[2] / 'scripts/694-lab-arena-native-judge-timeout-recovery.sql'
RUNTIME = previous.RUNTIME
database = previous.database
prepared = previous.prepared
prepared_company = previous.prepared_company
case = previous.case
base_extended = previous.base_extended
extended = previous.extended
provider_case = previous.provider_case
plain_finite = previous.plain_finite
before = previous.before
post = previous.post
installed = previous.installed


@pytest.fixture
def current(previous_current):
    p = previous_current
    previous.lookup.upgrade(p)
    previous.upgrade(p)
    return p


previous_current = previous.current


def seed_timeout(p, terminal='judge_timeout', result_terminal=None, stage='sandbox', error='judge_timeout'):
    """Seed the exact runner timeout envelope on a disposable historical parent."""
    result_terminal = terminal if result_terminal is None else result_terminal
    with p.c.cursor() as q:
        q.execute('ALTER TABLE lab_arena_runs DISABLE TRIGGER lab_arena_runs_terminal')
        q.execute('SET LOCAL session_replication_role=replica')
        q.execute('UPDATE lab_arena_runs SET terminal_cause=%s,result_doc=%s::jsonb WHERE run_id=%s', (
            terminal, json.dumps({'terminal_status': result_terminal,
                                 'failure_diagnostic': {'stage': stage, 'error_class': error}}), p.parent['run_id']))
        q.execute('SET LOCAL session_replication_role=origin')
        q.execute('ALTER TABLE lab_arena_runs ENABLE ALWAYS TRIGGER lab_arena_runs_terminal')
    p.c.commit()
    prior.saved_parent(p, p.parent['run_id'])


def upgrade(p):
    with p.c.cursor() as q:
        q.execute(SQL.read_text())
    p.c.commit()


def grant(p, kind='budget', **changes):
    if kind == 'budget':
        return finite.grant(p, **changes)
    return previous.grant(p, **changes)


@pytest.mark.parametrize('kind', ['budget', 'scorer'])
def test_current_rejects_native_timeout_despite_exact_diagnostic_and_clear_calls(current, kind):
    p = current
    seed_timeout(p)
    if kind == 'scorer':
        prior.mapping(p, RUNTIME)
    rows = base.snapshot(p)
    with pytest.raises(p.pg.Error, match='failed parent'):
        grant(p, kind, dry=True)
    p.c.rollback()
    assert base.snapshot(p) == rows


@pytest.mark.parametrize('kind', ['budget', 'scorer'])
@pytest.mark.parametrize('terminal', ['judge_timeout', 'judge_error'])
def test_exact_current_and_legacy_timeouts_preserve_work_and_complete_native_claim(current, kind, terminal):
    p = current
    seed_timeout(p, terminal)
    if kind == 'scorer':
        prior.mapping(p, RUNTIME)
    rows = base.snapshot(p)
    upgrade(p)
    upgrade(p)
    assert base.snapshot(p) == rows
    preview = grant(p, kind, dry=True)
    assert base.snapshot(p) == rows
    result = grant(p, kind)
    assert result['child_identity'] == preview['child_identity']
    assert result['checkpoints'] == p.checkpoints
    assert result['signal_checkpoints'] == p.signal_checkpoints
    after = base.snapshot(p)
    originals = {r['run_id']: r for r in rows['lab_arena_runs']}
    assert {r['run_id']: r for r in after['lab_arena_runs'] if r['run_id'] in originals} == originals
    for table in set(rows) - {'lab_arena_runs', 'lab_arena_scoring_supersessions'}:
        assert after[table] == rows[table]
    assert len(after['lab_arena_runs']) == len(rows['lab_arena_runs']) + 1
    assert len(after['lab_arena_scoring_supersessions']) == len(rows['lab_arena_scoring_supersessions']) + 1
    assert grant(p, kind)['idempotent'] is True
    child = result['child_identity']['run_id']
    assert old.funding(p, child)['funding_source'] == ('host' if p.organizer else 'miner_key')
    assert old.funding(p, child, 'deepline')['funding_source'] == 'miner_key'
    lease = prior.claim_child(p, child)
    assert lease['company_judgment_cache']['misses'] == []
    old.rpc(p, 'lab_arena_complete_attempt(%s,%s,%s::jsonb,%s,%s)', (
        child, old.TOKEN, json.dumps({'terminal_status': 'accepted'}), 'accepted', child + '-output'))
    old.rpc(p, 'lab_arena_expire_leases(%s)', (old.ROUND,))
    with p.c.cursor() as q:
        assert old.scalar(q, 'SELECT status FROM lab_arena_runs WHERE run_id=%s', (child,)) == 'accepted'
        assert old.scalar(q, 'SELECT count(*) FROM lab_arena_runs WHERE assignment_id=%s AND attempt>%s',
                          (lease['assignment_id'], lease['attempt'])) == 0
    p.c.commit()


@pytest.mark.parametrize('kind', ['budget', 'scorer'])
@pytest.mark.parametrize('fault', ['result_terminal', 'terminal', 'stage', 'error', 'missing_error'])
def test_mismatched_or_spoofed_timeout_is_refused_without_writes(current, kind, fault):
    p = current
    options = {'result_terminal': {'result_terminal': 'judge_error'},
               'terminal': {'terminal': 'judge_error', 'result_terminal': 'judge_timeout'},
               'stage': {'stage': 'scorer'}, 'error': {'error': 'judge_error'},
               'missing_error': {'error': None}}[fault]
    seed_timeout(p, **options)
    if kind == 'scorer':
        prior.mapping(p, RUNTIME)
    upgrade(p)
    rows = base.snapshot(p)
    with pytest.raises(p.pg.Error, match='failed parent'):
        grant(p, kind, dry=True)
    p.c.rollback()
    assert base.snapshot(p) == rows


def test_real_native_timeout_after_budget_child_allows_one_changed_runtime_fix(current):
    p = current
    child = finite.grant(p)['child_identity']['run_id']
    prior.claim_child(p, child)
    old.rpc(p, 'lab_arena_complete_attempt(%s,%s,%s::jsonb,%s,%s)', (
        child, old.TOKEN, json.dumps({'terminal_status': 'judge_timeout',
                                    'failure_diagnostic': {'stage': 'sandbox', 'error_class': 'judge_timeout'}}),
        'judge_timeout', None))
    prior.saved_parent(p, child)
    prior.mapping(p, RUNTIME)
    rows = base.snapshot(p)
    with pytest.raises(p.pg.Error, match='failed parent'):
        grant(p, 'scorer', dry=True)
    p.c.rollback()
    upgrade(p)
    assert base.snapshot(p) == rows
    with pytest.raises(p.pg.Error):
        grant(p, runtime=RUNTIME, request_id='no-second-budget-exception')
    p.c.rollback()
    result = grant(p, 'scorer')
    assert result['checkpoints'] == p.checkpoints
    prior.claim_child(p, result['child_identity']['run_id'])
    prior.complete_failed(p, result['child_identity']['run_id'])
    prior.saved_parent(p, result['child_identity']['run_id'])
    prior.mapping(p, 'sha256:' + 'e' * 64)
    rows = base.snapshot(p)
    with pytest.raises(p.pg.Error):
        grant(p, 'scorer', runtime='sha256:' + 'e' * 64, request_id='no-second-scorer-fix')
    p.c.rollback()
    assert base.snapshot(p) == rows


@pytest.mark.parametrize('kind', ['budget', 'scorer'])
def test_proved_provider_terminal_paths_are_preserved(current, kind):
    p = current
    previous.budget.diagnostic(p)
    entry = previous.previous.transport(p)
    assert previous.previous.proved(p, entry) is True
    if kind == 'scorer':
        prior.mapping(p, RUNTIME)
    upgrade(p)
    result = grant(p, kind)
    prior.claim_child(p, result['child_identity']['run_id'])


@pytest.mark.parametrize('terminal', ['judge_timeout', 'judge_error'])
def test_timeout_does_not_authorize_provider_fault_recovery(current, terminal):
    p = current
    seed_timeout(p, terminal)
    upgrade(p)
    rows = base.snapshot(p)
    with pytest.raises(p.pg.Error, match='failed parent'):
        prior.provider.grant(p)
    p.c.rollback()
    assert base.snapshot(p) == rows


@pytest.mark.parametrize('fault', ['stale_hash', 'approval', 'checkpoints', 'signal_checkpoints',
                                 'accepted', 'later', 'live', 'calls_pending', 'expired',
                                 'published', 'generation', 'frozen_source'])
def test_native_timeout_keeps_existing_recovery_guards(current, fault):
    p = current
    seed_timeout(p)
    p.original_parent = dict(p.parent)
    upgrade(p)
    finite.test_refusals_preserve_every_row(p, fault)


@pytest.mark.parametrize('role', ['lab_arena_service', 'anon', 'authenticated', 'service_role'])
def test_timeout_does_not_add_invocation_authority(current, role):
    seed_timeout(current)
    upgrade(current)
    finite.test_owner_only(current, role)


@pytest.mark.parametrize('check', [base.test_prior_costs_still_exhaust_the_same_cumulative_budget,
                                 base.test_existing_stage_call_quota_is_not_reset_by_the_grant])
def test_native_timeout_keeps_cumulative_money_and_call_caps(current, monkeypatch, check):
    seed_timeout(current)
    upgrade(current)
    finite.test_existing_money_and_provider_quotas(current, monkeypatch, check)


def test_scorer_timeout_still_requires_a_different_reviewed_runtime(current):
    p = current
    seed_timeout(p)
    upgrade(p)
    rows = base.snapshot(p)
    with pytest.raises(p.pg.Error):
        grant(p, 'scorer', runtime=p.parent['reviewer_runtime_doc']['runtime_image_digest'])
    p.c.rollback()
    assert base.snapshot(p) == rows


def test_function_only_patch_and_replay_keep_all_metadata(current):
    p = current
    with p.c.cursor() as q:
        before = old.scalar(q, "SELECT jsonb_agg(to_jsonb(p) ORDER BY oid) FROM pg_proc p WHERE pronamespace='public'::regnamespace")
        source = old.scalar(q, 'SELECT prosrc FROM pg_proc WHERE oid=%s::regprocedure', (COMMON,))
    p.c.commit()
    assert hashlib.sha256(source.encode()).hexdigest() == previous.POST
    rows = base.snapshot(p)
    upgrade(p)
    upgrade(p)
    with p.c.cursor() as q:
        after = old.scalar(q, "SELECT jsonb_agg(to_jsonb(p) ORDER BY oid) FROM pg_proc p WHERE pronamespace='public'::regnamespace")
    p.c.commit()
    changed = [(a, b) for a, b in zip(before, after) if a != b]
    assert len(before) == len(after) and len(changed) == 1
    original, updated = changed[0]
    assert original['proname'] == 'lab_arena__grant_failed_judge_recovery_v1'
    assert {k: v for k, v in original.items() if k != 'prosrc'} == {k: v for k, v in updated.items() if k != 'prosrc'}
    assert hashlib.sha256(updated['prosrc'].encode()).hexdigest() == '2c064da4290e3095821fc3cbbc3f79a4c3cadc0da1b2fe496d5f0907bd09b743'
    assert base.snapshot(p) == rows


@pytest.mark.parametrize('drift', ['source', 'acl', 'owner', 'security', 'search_path'])
def test_unknown_source_or_security_fails_closed(current, drift):
    p = current
    with p.c.cursor() as q:
        before = old.scalar(q, 'SELECT to_jsonb(p) FROM pg_proc p WHERE oid=%s::regprocedure', (COMMON,))
        if drift == 'source':
            definition = old.scalar(q, 'SELECT pg_get_functiondef(%s::regprocedure)', (COMMON,))
            q.execute(definition.replace(before['prosrc'], before['prosrc'] + '\n-- unknown source', 1))
        elif drift == 'acl':
            q.execute('GRANT EXECUTE ON FUNCTION ' + COMMON + ' TO anon')
        elif drift == 'owner':
            q.execute('ALTER FUNCTION ' + COMMON + ' OWNER TO anon')
        elif drift == 'security':
            q.execute('ALTER FUNCTION ' + COMMON + ' SECURITY INVOKER')
        else:
            q.execute('ALTER FUNCTION ' + COMMON + ' SET search_path=public')
        with pytest.raises(p.pg.Error, match='preimage differs|security differs'):
            q.execute(SQL.read_text())
    p.c.rollback()
    with p.c.cursor() as q:
        assert old.scalar(q, 'SELECT to_jsonb(p) FROM pg_proc p WHERE oid=%s::regprocedure', (COMMON,)) == before
    p.c.commit()
