"""Reviewed code repair after a finite owner exception uses native SQL only."""
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest
from tests.lab_arena import postpublication_judgments678_postgres_test as postpub

f = postpub.f
finite = f.old
prior = finite.prior
base = finite.base
old = finite.old
SQL = Path(__file__).parents[2] / 'scripts/679-lab-arena-finite-child-scorer-correction.sql'
COMMON = prior.COMMON
HOST = prior.provider.HELPERS[1]
RUNTIME = 'sha256:' + '5' * 64
legacy_grant = base.grant
database = f.database
prepared = f.prepared
prepared_company = f.prepared_company
case = f.case
base_extended = f.base_extended
extended = f.extended
provider_case = f.provider_case
plain_finite = f.plain_finite
before = f.compatible_plain
post = postpub.post


@pytest.fixture
def prior_bug(request):
    return getattr(request, 'param', False)


@pytest.fixture
def budget_parent(post, prior_bug):
    p = post
    p.original_parent = dict(p.parent)
    if prior_bug:
        prior.mapping(p, prior.RUNTIME)
        child = prior.grant(p)['child_identity']['run_id']
        prior.claim_child(p, child)
        prior.complete_failed(p, child)
        prior.saved_parent(p, child)
        p.exception_runtime = prior.RUNTIME
    child = finite.grant(p)['child_identity']['run_id']
    prior.claim_child(p, child)
    prior.complete_failed(p, child)
    prior.saved_parent(p, child)
    prior.mapping(p, RUNTIME)
    return p


def grant(p, **changes):
    return legacy_grant(p, runtime=changes.pop('runtime', RUNTIME),
                        correction='git:independently-reviewed-code-defect', **changes)


@pytest.fixture
def corrected(budget_parent):
    p = budget_parent
    with p.c.cursor() as q:
        originals = [old.scalar(q, 'SELECT pg_get_functiondef(%s::regprocedure)', (sig,))
                     for sig in [COMMON, HOST]]
        q.execute(SQL.read_text())
    p.c.commit()
    try:
        yield p
    finally:
        p.c.rollback()
        with p.c.cursor() as q:
            for definition in originals:
                q.execute(definition)
        p.c.commit()


def test_before_migration_refuses_proved_bug_after_budget_child(budget_parent):
    p = budget_parent
    rows = base.snapshot(p)
    with pytest.raises(p.pg.Error, match='failed parent'):
        grant(p)
    p.c.rollback()
    assert base.snapshot(p) == rows


@pytest.mark.parametrize('outcome', ['accepted', 'judge_error', 'expired'])
def test_one_reviewed_correction_retains_work_payer_and_native_finality(corrected, outcome):
    p = corrected
    rows = base.snapshot(p)
    preview = grant(p, dry=True)
    assert base.snapshot(p) == rows
    result = grant(p)
    assert result['child_identity'] == preview['child_identity']
    assert len(result['checkpoints']) == 2
    assert result['organizer_openrouter_authorized'] is p.organizer
    after = base.snapshot(p)
    assert len(after['lab_arena_runs']) == len(rows['lab_arena_runs']) + 1
    assert len(after['lab_arena_scoring_supersessions']) == len(rows['lab_arena_scoring_supersessions']) + 1
    originals = {r['run_id']: r for r in rows['lab_arena_runs']}
    assert {r['run_id']: r for r in after['lab_arena_runs'] if r['run_id'] in originals} == originals
    for table in set(rows) - {'lab_arena_runs', 'lab_arena_scoring_supersessions'}:
        assert rows[table] == after[table]
    assert grant(p)['idempotent'] is True and base.snapshot(p) == after
    child = result['child_identity']['run_id']
    assert old.funding(p, child)['funding_source'] == ('host' if p.organizer else 'miner_key')
    assert old.funding(p, child, 'deepline')['funding_source'] == 'miner_key'
    prior.claim_child(p, child)
    if outcome == 'expired':
        with p.c.cursor() as q:
            q.execute("UPDATE lab_arena_runs SET lease_expires_at=clock_timestamp()-interval '1 second' WHERE run_id=%s", (child,))
        p.c.commit()
    else:
        old.rpc(p, 'lab_arena_complete_attempt(%s,%s,%s::jsonb,%s,%s)', (
            child, old.TOKEN, json.dumps({'terminal_status': outcome, 'failure_diagnostic': {
                'stage': 'scorer', 'error_class': 'judge_error', 'reason': 'malformed_response'}}),
            outcome, child + '-output' if outcome == 'accepted' else None))
    old.rpc(p, 'lab_arena_expire_leases(%s)', (old.ROUND,))
    prior.saved_parent(p, child)
    prior.mapping(p, 'sha256:' + '6' * 64)
    rows = base.snapshot(p)
    for attempt in [lambda: grant(p, runtime='sha256:' + '6' * 64, request_id='no-second-bug'),
                    lambda: finite.grant(p, runtime='sha256:' + '6' * 64, request_id='no-second-budget')]:
        with pytest.raises(p.pg.Error):
            attempt()
        p.c.rollback()
        assert base.snapshot(p) == rows
    assert old.funding(p, child)['funding_source'] == ('host' if p.organizer else 'miner_key')
    with p.c.cursor() as q:
        assert old.scalar(q, 'SELECT count(*) FROM lab_arena_runs WHERE assignment_id=%s AND attempt>%s',
                          (p.parent['assignment_id'], p.parent['attempt'])) == 0
    p.c.commit()


@pytest.mark.parametrize('fault', ['same_runtime', 'stale_hash', 'accepted', 'later', 'calls_pending',
                                  'expired', 'published', 'generation', 'journal_identity',
                                  'journal_runtime', 'journal_parent_hash', 'journal_source',
                                  'journal_floor', 'journal_organizer', 'journal_operation',
                                  'journal_kind', 'journal_parent_id', 'journal_child_id',
                                  'journal_request_hash', 'prior_correction'])
def test_invalid_continuation_is_atomic(corrected, fault):
    # Existing controls corrupt only disposable state and compare every row.
    # Rehash forged journals below, so semantic proof checks are exercised too.
    if not fault.startswith('journal_'):
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(prior, 'grant', grant)
            return prior.test_invalid_continuation_fails_closed_without_writes(corrected, fault)
    p = corrected
    field, value = {
        'journal_identity': ('{child_identity,judgment_input_hash}', 'tampered'),
        'journal_operation': ('{operation}', 'wrong'),
        'journal_kind': ('{recovery_kind}', 'wrong'),
        'journal_parent_id': ('{parent_run_id}', 'missing-parent'),
        'journal_child_id': ('{child_identity,run_id}', 'missing-child'),
        'journal_request_hash': ('{approval_ref}', 'changed-proof'),
        'journal_source': ('{child_identity,scored_run_id}', 'cross-source'),
        'journal_floor': ('{child_identity,scoring_generation_floor}', 123),
        'journal_organizer': ('{organizer_openrouter_authorized}', not p.organizer),
        'journal_runtime': ('{runtime_image_digest}', 'sha256:' + '0' * 64),
        'journal_parent_hash': ('{parent_sha256}', 'sha256:' + '0' * 64),
    }[fault]
    with p.c.cursor() as q:
        q.execute('SET LOCAL session_replication_role=replica')
        q.execute("UPDATE lab_arena_scoring_supersessions SET recovery_doc=jsonb_set(recovery_doc,%s,%s::jsonb) WHERE recovery_doc#>>'{child_identity,run_id}'=%s", (field, json.dumps(value), p.parent['run_id']))
        if fault != 'journal_request_hash':
            q.execute("UPDATE lab_arena_scoring_supersessions SET request_hash='sha256:'||encode(extensions.digest(recovery_doc::text,'sha256'),'hex') WHERE recovery_doc->>'recovery_kind'='operator_judge_budget_exception_v1' OR request_id='finite-owner-exception'")
        q.execute('SET LOCAL session_replication_role=origin')
    p.c.commit()
    rows = base.snapshot(p)
    with pytest.raises(p.pg.Error):
        grant(p)
    p.c.rollback()
    assert base.snapshot(p) == rows


@pytest.mark.parametrize('prior_bug', [True], indirect=True)
def test_bug_then_budget_cannot_obtain_another_bug_correction(corrected):
    p = corrected
    rows = base.snapshot(p)
    with pytest.raises(p.pg.Error, match='failed parent'):
        grant(p, request_id='no-bug-after-intervening-budget')
    p.c.rollback()
    assert base.snapshot(p) == rows


@pytest.mark.parametrize('fault', ['active', 'frozen_source', 'missing_quota'])
def test_other_native_guards_remain_closed(corrected, fault):
    p = corrected
    changes = {}
    with p.c.cursor() as q:
        q.execute('ALTER TABLE lab_arena_runs DISABLE TRIGGER lab_arena_runs_terminal')
        q.execute('SET LOCAL session_replication_role=replica')
        if fault == 'active':
            q.execute("UPDATE lab_arena_runs SET status='pending' WHERE assignment_id=%s AND attempt=1", (p.parent['assignment_id'],))
        elif fault == 'frozen_source':
            q.execute("UPDATE lab_arena_runs SET judgment_input_hash='sha256:'||repeat('0',64) WHERE run_id=%s", (p.parent['run_id'],))
            changes['parent_hash'] = 'sha256:' + old.scalar(q, "SELECT encode(extensions.digest(to_jsonb(r)::text,'sha256'),'hex') FROM lab_arena_runs r WHERE run_id=%s", (p.parent['run_id'],))
        else:
            q.execute("UPDATE lab_arena_rounds SET configuration_doc=configuration_doc-'scoring_call_quotas' WHERE round_id=%s", (old.ROUND,))
        q.execute('SET LOCAL session_replication_role=origin')
        q.execute('ALTER TABLE lab_arena_runs ENABLE ALWAYS TRIGGER lab_arena_runs_terminal')
    p.c.commit()
    rows = base.snapshot(p)
    if fault == 'missing_quota':
        child = grant(p)['child_identity']['run_id']
        with p.c.cursor() as q:
            q.execute("UPDATE lab_arena_runs SET created_at=clock_timestamp()-interval '1 hour' WHERE run_id=%s", (child,))
        p.c.commit()
        old.release_guard(p)
        lease = old.claim(p)
        assert lease['run_id'] == child
        request_hash = 'sha256:' + '7' * 64
        call = base.contracts.provider_call_identity(
            assignment_id=lease['assignment_id'], attempt=lease['attempt'],
            icp_position=lease['icp_position'], action_sequence=1,
            operation_id='openrouter.chat', request_hash=request_hash)
        rows = base.snapshot(p)
        with pytest.raises(p.pg.Error, match='quota_missing'):
            old.rpc(p, 'lab_arena_reserve_call(%s,%s,%s,%s,%s,%s,%s,%s::jsonb,%s)', (
                child, old.TOKEN, call, 'openrouter.chat', 'openrouter',
                'host' if p.organizer else 'miner_key', 500,
                json.dumps(dict(request_hash=request_hash, base_call_identity=call,
                                action_sequence=1, provider_attempt=1, max_output_tokens=8)), 1500))
        assert base.snapshot(p) == rows
    else:
        with pytest.raises(p.pg.Error):
            grant(p, **changes)
        p.c.rollback()
        assert base.snapshot(p) == rows


@pytest.mark.parametrize('check', [base.test_prior_costs_still_exhaust_the_same_cumulative_budget,
                                 base.test_existing_stage_call_quota_is_not_reset_by_the_grant])
def test_cumulative_costs_and_quota_still_apply(corrected, monkeypatch, check):
    monkeypatch.setattr(base, 'grant', grant)
    check(corrected)


def test_upgrade_and_replay_change_only_two_function_bodies(budget_parent):
    # Reuse the full catalog/security and data equality check against679.
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(prior, 'SQL', SQL)
        prior.test_first_migration_changes_only_two_function_bodies_and_no_data(budget_parent)


@pytest.mark.parametrize('target', ['common', 'host'])
@pytest.mark.parametrize('fault', ['source', 'volatility', 'language', 'strict', 'leakproof', 'parallel', 'acl'])
def test_migration_drift_refusal_is_atomic(corrected, target, fault):
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(prior, 'SQL', SQL)
        prior.test_migration_replay_rejects_changed_source_or_metadata(corrected, target, fault)


def test_migration_replay_retains_rows(corrected):
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(prior, 'SQL', SQL)
        prior.test_migration_replay_preserves_state(corrected)


def test_second_function_drift_rolls_back_first_function_upgrade(budget_parent):
    p = budget_parent
    with p.c.cursor() as q:
        original = old.scalar(q, 'SELECT pg_get_functiondef(%s::regprocedure)', (HOST,))
        q.execute(original.replace('SELECT coalesce', 'SELECT /* drift */ coalesce', 1))
        sources = old.scalar(q, 'SELECT jsonb_agg(to_jsonb(p) ORDER BY oid) FROM pg_proc p WHERE oid=ANY(%s::regprocedure[])', ([COMMON, HOST],))
        acl = old.scalar(q, "SELECT nspacl::text FROM pg_namespace WHERE nspname='public'")
    p.c.commit()
    rows = base.snapshot(p)
    try:
        with pytest.raises(p.pg.Error, match='preimage differs'):
            with p.c.cursor() as q:
                q.execute(SQL.read_text())
        p.c.rollback()
        with p.c.cursor() as q:
            assert old.scalar(q, 'SELECT jsonb_agg(to_jsonb(p) ORDER BY oid) FROM pg_proc p WHERE oid=ANY(%s::regprocedure[])', ([COMMON, HOST],)) == sources
            assert old.scalar(q, "SELECT nspacl::text FROM pg_namespace WHERE nspname='public'") == acl
        assert base.snapshot(p) == rows
    finally:
        p.c.rollback()
        with p.c.cursor() as q:
            q.execute(original)
        p.c.commit()


def test_service_cannot_grant_reviewed_correction(corrected):
    prior.test_service_cannot_use_correction_authority(corrected)


def test_competing_owner_requests_append_only_one_child(corrected):
    p = corrected
    def attempt(request):
        c = p.pg.connect(**p.dsn)
        other = SimpleNamespace(c=c, parent=p.parent, parent_hash=p.parent_hash)
        try:
            return grant(other, request_id=request)
        except p.pg.Error:
            c.rollback()
            return None
        finally:
            c.close()
    rows = base.snapshot(p)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(attempt, ['owner-race-a', 'owner-race-b']))
    assert sum(r is not None for r in results) == 1
    after = base.snapshot(p)
    assert len(after['lab_arena_runs']) == len(rows['lab_arena_runs']) + 1
    assert len(after['lab_arena_scoring_supersessions']) == len(rows['lab_arena_scoring_supersessions']) + 1
