"""A proved provider terminal can receive one reviewed, different-runtime fix.

Use the exact689/690 helper and native owner budget replacement ancestry.
All writes occur in disposable PostgreSQL; no provider or production I/O.
"""
import hashlib
import json
from pathlib import Path

import pytest
from tests.lab_arena import retained_proof_call_lookup690_postgres_test as lookup

previous = lookup.previous
old, base, finite = previous.old, previous.base, previous.finite
prior, budget = previous.prior, previous.budget
ROOT = Path(__file__).parents[2]
SQL = ROOT / 'scripts/691-lab-arena-provider-terminal-scorer-correction.sql'
COMMON = 'public.lab_arena__grant_failed_judge_recovery_v1(text,text,text,text,text,text,boolean,text,bigint)'
PRE = '5befa705e76c54757dea017657c7a60a8cd184b7f27419153c5157671740c77e'
POST = '1f2b05d010a26698a2aa9b7c402868582884376efc6533857178d813e8b3fc85'
RUNTIME = 'sha256:' + 'd' * 64
NATIVE_GRANT = base.grant

database = lookup.database
prepared = lookup.prepared
prepared_company = lookup.prepared_company
case = lookup.case
base_extended = lookup.base_extended
extended = lookup.extended
provider_case = lookup.provider_case
plain_finite = lookup.plain_finite
before = lookup.before
post = lookup.post
installed = lookup.installed
current = lookup.current


def grant(p, **changes):
    return NATIVE_GRANT(p, runtime=changes.pop('runtime', RUNTIME),
                      correction='git:reviewed-legacy-company-progress-fix', **changes)


def upgrade(p):
    with p.c.cursor() as q:
        q.execute(SQL.read_text())
    p.c.commit()


def prepare_provider_parent(p):
    lookup.upgrade(p)
    p.root_failed_parent = dict(p.parent)
    previous.make_failed_budget_child(p)
    child = finite.grant(p, request_id='replacement', approval='approval:replacement')['child_identity']['run_id']
    prior.claim_child(p, child)
    prior.complete_failed(p, child)
    prior.saved_parent(p, child)
    budget.diagnostic(p)
    p.transient_entry = previous.transport(p)
    p.old_runtime = p.parent['reviewer_runtime_doc']['runtime_image_digest']
    prior.mapping(p, RUNTIME)
    with p.c.cursor() as q:
        assert old.scalar(q, "SELECT count(*) FROM lab_arena_scoring_supersessions WHERE recovery_doc->>'recovery_kind'='operator_judge_budget_exception_v1' AND recovery_doc#>>'{child_identity,assignment_id}'=%s", (p.parent['assignment_id'],)) == 2
        source = old.scalar(q, 'SELECT prosrc FROM pg_proc WHERE oid=%s::regprocedure', (COMMON,))
        assert hashlib.sha256(source.encode()).hexdigest() == PRE
    p.c.commit()
    return p


@pytest.fixture
def provider_parent(current):
    return prepare_provider_parent(current)


def test_current689_rejects_provider_terminal_despite_reviewed_corrected_runtime(provider_parent):
    p = provider_parent
    assert previous.proved(p, p.transient_entry) is True
    rows = base.snapshot(p)
    with pytest.raises(p.pg.Error, match='failed parent'):
        grant(p, dry=True)
    p.c.rollback()
    assert base.snapshot(p) == rows


def test_exact_function_only_upgrade_replay_and_native_correction_preserve_all_work(provider_parent):
    p = provider_parent
    with p.c.cursor() as q:
        source = old.scalar(q, 'SELECT prosrc FROM pg_proc WHERE oid=%s::regprocedure', (COMMON,))
        metadata = old.scalar(q, "SELECT to_jsonb(p)-'prosrc' FROM pg_proc p WHERE oid=%s::regprocedure", (COMMON,))
    p.c.commit()
    rows = base.snapshot(p)
    upgrade(p)
    upgrade(p)
    assert base.snapshot(p) == rows
    with p.c.cursor() as q:
        changed = old.scalar(q, 'SELECT prosrc FROM pg_proc WHERE oid=%s::regprocedure', (COMMON,))
        assert old.scalar(q, "SELECT to_jsonb(p)-'prosrc' FROM pg_proc p WHERE oid=%s::regprocedure", (COMMON,)) == metadata
    p.c.commit()
    assert changed == source.replace("   OR ((p_recovery_kind='operator_judge_budget_exception_v1' AND", "   OR ((p_recovery_kind IN ('operator_judge_budget_exception_v1','operator_scorer_bug_recovery_v1') AND", 1)
    assert hashlib.sha256(changed.encode()).hexdigest() == POST
    preview = grant(p, dry=True)
    assert base.snapshot(p) == rows
    result = grant(p)
    assert result['child_identity'] == preview['child_identity']
    assert result['checkpoints'] == p.checkpoints
    assert result['signal_checkpoints'] == p.signal_checkpoints
    after = base.snapshot(p)
    assert len(after['lab_arena_runs']) == len(rows['lab_arena_runs']) + 1
    assert len(after['lab_arena_scoring_supersessions']) == len(rows['lab_arena_scoring_supersessions']) + 1
    originals = {r['run_id']: r for r in rows['lab_arena_runs']}
    assert {r['run_id']: r for r in after['lab_arena_runs'] if r['run_id'] in originals} == originals
    for table in set(rows) - {'lab_arena_runs', 'lab_arena_scoring_supersessions'}:
        assert after[table] == rows[table]
    assert grant(p)['idempotent'] is True
    assert base.snapshot(p) == after
    child = result['child_identity']['run_id']
    assert result['child_identity']['attempt'] == p.parent['attempt'] + 1
    assert old.funding(p, child)['funding_source'] == ('host' if p.organizer else 'miner_key')
    assert old.funding(p, child, 'deepline')['funding_source'] == 'miner_key'
    lease = prior.claim_child(p, child)
    assert lease['company_judgment_cache']['misses'] == []
    prior.complete_failed(p, child)
    prior.saved_parent(p, child)
    budget.diagnostic(p)
    previous.transport(p)
    prior.mapping(p, 'sha256:' + 'e' * 64)
    saved = base.snapshot(p)
    with pytest.raises(p.pg.Error, match='failed parent'):
        grant(p, runtime='sha256:' + 'e' * 64, request_id='no-second-scorer-correction')
    p.c.rollback()
    assert base.snapshot(p) == saved


@pytest.mark.parametrize('fault', ['same_runtime', 'no_transient', 'no_unresolved', 'healed',
                                  'wrong_parent', 'stale_hash', 'calls_pending', 'receipt_hash',
                                  'receipt_identity', 'receipt_runtime', 'receipt_parent_hash'])
def test_invalid_provider_correction_is_atomic(provider_parent, fault):
    p = provider_parent
    changes = {}
    if fault == 'same_runtime':
        prior.mapping(p, p.old_runtime)
        changes['runtime'] = p.old_runtime
    elif fault == 'no_unresolved':
        budget.diagnostic(p, unresolved=0)
    elif fault == 'healed':
        budget.receipt(p, status=200, success=True)
    elif fault == 'wrong_parent':
        changes['parent'] = p.root_failed_parent['run_id']
        with p.c.cursor() as q:
            changes['parent_hash'] = 'sha256:' + old.scalar(q, "SELECT encode(extensions.digest(to_jsonb(r)::text,'sha256'),'hex') FROM lab_arena_runs r WHERE run_id=%s", (changes['parent'],))
        p.c.commit()
    elif fault == 'stale_hash':
        changes['parent_hash'] = 'sha256:' + '0' * 64
    else:
        with p.c.cursor() as q:
            q.execute('SET LOCAL session_replication_role=replica')
            if fault == 'no_transient':
                q.execute("UPDATE lab_arena_ledger SET entry_doc=jsonb_set(entry_doc,'{call,observed_provider_status}','401') WHERE entry_id=%s", (p.transient_entry,))
            elif fault == 'calls_pending':
                # A different run in the same assignment leaves the current
                # retained failure unhealed, isolating the clear-call guard.
                r = p.root_failed_parent
                old.insert(q, 'lab_arena_ledger', dict(entry_kind='reservation', run_id=r['run_id'],
                    round_id=r['round_id'], submission_id=r['submission_id'], miner_hotkey=r['miner_hotkey'],
                    stage=r['stage'], call_identity='sha256:' + '7' * 64, provider='openrouter',
                    operation_id='openrouter.chat', funding_source='host' if p.organizer else 'miner_key',
                    amount_microusd=500, entry_doc={}))
            else:
                field, value = {'receipt_hash': ('{approval_ref}', 'changed'),
                                'receipt_identity': ('{child_identity,scored_run_id}', 'cross-source'),
                                'receipt_runtime': ('{runtime_image_digest}', 'sha256:' + '0' * 64),
                                'receipt_parent_hash': ('{parent_sha256}', 'sha256:' + '0' * 64)}[fault]
                q.execute("UPDATE lab_arena_scoring_supersessions SET recovery_doc=jsonb_set(recovery_doc,%s,%s::jsonb) WHERE recovery_doc#>>'{child_identity,run_id}'=%s", (field, json.dumps(value), p.parent['run_id']))
                if fault != 'receipt_hash':
                    q.execute("UPDATE lab_arena_scoring_supersessions SET request_hash='sha256:'||encode(extensions.digest(recovery_doc::text,'sha256'),'hex') WHERE recovery_doc#>>'{child_identity,run_id}'=%s", (p.parent['run_id'],))
            q.execute('SET LOCAL session_replication_role=origin')
        p.c.commit()
    upgrade(p)
    rows = base.snapshot(p)
    with pytest.raises(p.pg.Error):
        grant(p, **changes)
    p.c.rollback()
    assert base.snapshot(p) == rows


def test_budget_and_provider_recovery_caps_are_not_expanded(provider_parent):
    p = provider_parent
    upgrade(p)
    rows = base.snapshot(p)
    for action in [lambda: finite.grant(p, request_id='no-third-budget', approval='approval:no-third'),
                   lambda: prior.provider.grant(p, request='no-provider-only-chain', entry=p.transient_entry)]:
        with pytest.raises(p.pg.Error):
            action()
        p.c.rollback()
        assert base.snapshot(p) == rows


@pytest.mark.parametrize('drift', ['source', 'acl', 'path', 'proof_source', 'proof_acl'])
def test_unknown_function_or_security_refuses_atomically(provider_parent, drift):
    p = provider_parent
    with p.c.cursor() as q:
        target = lookup.SIG if drift.startswith('proof_') else COMMON
        original = old.scalar(q, 'SELECT pg_get_functiondef(%s::regprocedure)', (target,))
        if drift == 'proof_source':
            q.execute(original.replace('DECLARE call_doc ', '-- changed\nDECLARE call_doc ', 1))
        elif drift == 'proof_acl':
            q.execute('GRANT EXECUTE ON FUNCTION ' + target + ' TO anon')
        elif drift == 'source':
            q.execute(original.replace('DECLARE parent ', '-- changed\nDECLARE parent ', 1))
        elif drift == 'acl':
            q.execute('GRANT EXECUTE ON FUNCTION ' + COMMON + ' TO anon')
        else:
            q.execute('ALTER FUNCTION ' + COMMON + ' SET search_path=public,pg_catalog')
    p.c.commit()
    rows = base.snapshot(p)
    try:
        with pytest.raises(p.pg.Error, match='(preimage|security|retained proof) differs'):
            upgrade(p)
        p.c.rollback()
        assert base.snapshot(p) == rows
    finally:
        with p.c.cursor() as q:
            q.execute(original)
            if drift in ('acl', 'proof_acl'):
                q.execute('REVOKE EXECUTE ON FUNCTION ' + target + ' FROM anon')
        p.c.commit()


def test_baseline_stage1_provider_terminal_uses_same_native_correction_and_host_payer(current):
    p = current
    if p.organizer:
        pytest.skip('Construct the baseline once from the ordinary payer fixture.')
    previous.previous.baseline(p)
    p = prepare_provider_parent(p)
    rows = base.snapshot(p)
    with pytest.raises(p.pg.Error, match='failed parent'):
        grant(p, dry=True)
    p.c.rollback()
    assert base.snapshot(p) == rows
    upgrade(p)
    result = grant(p)
    child = result['child_identity']['run_id']
    assert result['child_identity']['stage'] == 1
    assert result['child_identity']['submission_id'].startswith('baseline-')
    assert result['checkpoints'] == p.checkpoints
    assert result['signal_checkpoints'] == p.signal_checkpoints
    assert old.funding(p, child)['funding_source'] == 'host'
    lease = prior.claim_child(p, child)
    assert lease['run_id'] == child
    assert lease['company_judgment_cache']['misses'] == []


@pytest.mark.parametrize('check', [base.test_prior_costs_still_exhaust_the_same_cumulative_budget,
                                  base.test_existing_stage_call_quota_is_not_reset_by_the_grant])
def test_native_cumulative_money_and_stage_quota_remain_enforced(provider_parent, monkeypatch, check):
    p = provider_parent
    upgrade(p)
    parent, parent_hash = p.parent, p.parent_hash
    # Seed the historical spend in an older run of this exact assignment.
    # The final failed call in the current parent remains genuinely unhealed.
    p.parent = p.root_failed_parent

    def grant_current(candidate, **changes):
        candidate.parent, candidate.parent_hash = parent, parent_hash
        return grant(candidate, **changes)

    monkeypatch.setattr(base, 'grant', grant_current)
    check(p)


def test_original_provider_terminal_also_uses_one_distinct_runtime_correction(current):
    p = current
    lookup.upgrade(p)
    budget.diagnostic(p)
    entry = previous.transport(p)
    assert previous.proved(p, entry) is True
    prior.mapping(p, RUNTIME)
    rows = base.snapshot(p)
    with pytest.raises(p.pg.Error, match='failed parent'):
        grant(p, dry=True)
    p.c.rollback()
    assert base.snapshot(p) == rows
    upgrade(p)
    preview = grant(p, dry=True)
    assert base.snapshot(p) == rows
    result = grant(p)
    assert result['child_identity'] == preview['child_identity']
    assert result['checkpoints'] == p.checkpoints
    child = result['child_identity']['run_id']
    assert old.funding(p, child)['funding_source'] == ('host' if p.organizer else 'miner_key')
    prior.claim_child(p, child)
