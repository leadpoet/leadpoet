"""One distinct scorer correction after an exact finite provider recovery."""
import json
from pathlib import Path

import pytest
from tests.lab_arena import retained_transient_provider_proof673_postgres_test as transient
from tests.lab_arena import operator_provider_fault_recovery671_postgres_test as provider

base = provider.base
old = base.old
SQL = Path(__file__).parents[2] / 'scripts/675-lab-arena-provider-child-scorer-correction.sql'
COMMON = transient.COMMON
RUNTIME = 'sha256:' + '3' * 64
database = provider.database
prepared = provider.prepared
prepared_company = provider.prepared_company
case = provider.case
base_extended = provider.extended
extended = transient.extended
provider_case = provider.provider_case


def saved_parent(p, run_id):
    with p.c.cursor() as q:
        p.parent = old.scalar(q, 'SELECT to_jsonb(r)FROM lab_arena_runs r WHERE run_id=%s', (run_id,))
        p.parent_hash = 'sha256:' + old.scalar(q, "SELECT encode(extensions.digest(to_jsonb(r)::text,'sha256'),'hex')FROM lab_arena_runs r WHERE run_id=%s", (run_id,))
    p.c.commit()


def mapping(p, runtime):
    with p.c.cursor() as q:
        cfg = old.scalar(q, 'SELECT configuration_doc FROM lab_arena_rounds WHERE round_id=%s', (old.ROUND,))
        q.execute('SELECT public.lab_arena_register_reviewer_runtime_v1(%s,%s,%s,%s)', (
            cfg['scorer_image_digest'], cfg['scorer_image_reference'], runtime, base.REPOSITORY + '@' + runtime))
    p.c.commit()


def claim_child(p, child):
    with p.c.cursor() as q:
        # Only disposable tests age retry backoff. No live timestamp changes.
        q.execute("UPDATE lab_arena_runs SET created_at=clock_timestamp()-interval '1 hour'WHERE run_id=%s", (child,))
    p.c.commit()
    old.release_guard(p)
    lease = old.claim(p)
    assert lease['run_id'] == child
    assert len(lease['company_judgment_cache']['hits']) == 2
    return lease


def complete_failed(p, child):
    old.rpc(p, 'lab_arena_complete_attempt(%s,%s,%s::jsonb,%s,%s)', (
        child, old.TOKEN, json.dumps({'terminal_status': 'judge_error', 'failure_diagnostic': {
            'stage': 'scorer', 'reason': 'provider_error', 'error_class': 'judge_error'}}), 'judge_error', None))


@pytest.fixture
def correctable(provider_case):
    p = provider_case
    p.original_parent = p.parent
    child = provider.grant(p)['child_identity']['run_id']
    claim_child(p, child)
    complete_failed(p, child)
    saved_parent(p, child)
    mapping(p, RUNTIME)
    yield p


@pytest.fixture
def corrected(correctable):
    p = correctable
    with p.c.cursor() as q:
        originals = [old.scalar(q, 'SELECT pg_get_functiondef(%s::regprocedure)', (sig,))
                     for sig in [COMMON, provider.HELPERS[1]]]
        q.execute(SQL.read_text())
    p.c.commit()
    try:
        yield p
    finally:
        p.c.rollback()
        with p.c.cursor() as q:
            for original in originals: q.execute(original)
        p.c.commit()


legacy_grant = base.grant


def grant(p, **changes):
    return legacy_grant(p, runtime=changes.pop('runtime', RUNTIME), **changes)


def test_before_migration_rejects_provider_child_even_after_runtime_correction(correctable):
    p = correctable
    before = base.snapshot(p)
    with pytest.raises(p.pg.Error, match='failed parent'):
        grant(p)
    p.c.rollback()
    assert base.snapshot(p) == before


def test_distinct_runtime_correction_is_one_append_with_checkpoints_and_same_payer(corrected):
    p = corrected
    before = base.snapshot(p)
    preview = grant(p, dry=True)
    assert base.snapshot(p) == before
    result = grant(p)
    assert result['child_identity'] == preview['child_identity']
    assert len(result['checkpoints']) == 2
    assert result['organizer_openrouter_authorized'] is p.organizer
    after = base.snapshot(p)
    assert len(after['lab_arena_runs']) == len(before['lab_arena_runs']) + 1
    assert len(after['lab_arena_scoring_supersessions']) == len(before['lab_arena_scoring_supersessions']) + 1
    originals = {r['run_id']: r for r in before['lab_arena_runs']}
    assert {r['run_id']: r for r in after['lab_arena_runs'] if r['run_id'] in originals} == originals
    for table in set(before) - {'lab_arena_runs', 'lab_arena_scoring_supersessions'}:
        assert before[table] == after[table]
    assert grant(p)['idempotent'] is True
    assert base.snapshot(p) == after
    child = result['child_identity']['run_id']
    assert old.funding(p, child)['funding_source'] == ('host' if p.organizer else 'miner_key')
    assert old.funding(p, child, 'deepline')['funding_source'] == 'miner_key'
    lease = claim_child(p, child)
    with p.c.cursor() as q:
        assert old.scalar(q, "SELECT reviewer_runtime_doc->>'runtime_image_digest'FROM lab_arena_runs WHERE run_id=%s", (child,)) == RUNTIME
    p.c.commit()
    complete_failed(p, child)
    old.rpc(p, 'lab_arena_expire_leases(%s)', (old.ROUND,))
    saved_parent(p, child)
    mapping(p, 'sha256:' + '4' * 64)
    before = base.snapshot(p)
    for call in [lambda: grant(p, runtime='sha256:' + '4' * 64, request_id='no-third-operator-grant'),
                 lambda: provider.grant(p, runtime=RUNTIME, request='no-provider-after-correction')]:
        with pytest.raises(p.pg.Error):
            call()
        p.c.rollback()
        assert base.snapshot(p) == before


@pytest.mark.parametrize('fault', ['same_runtime', 'provider_retry', 'stale_hash', 'accepted', 'later', 'calls_pending',
                                  'expired', 'published', 'generation', 'journal_identity', 'journal_runtime',
                                  'journal_parent_hash', 'journal_source', 'journal_floor', 'journal_organizer', 'journal_operation', 'journal_kind', 'journal_parent_id', 'journal_child_id', 'journal_request_hash', 'prior_correction'])
def test_invalid_continuation_fails_closed_without_writes(corrected, fault):
    p = corrected
    changes = {}
    with p.c.cursor() as q:
        q.execute('ALTER TABLE lab_arena_runs DISABLE TRIGGER lab_arena_runs_terminal')
        q.execute('SET LOCAL session_replication_role=replica')
        if fault == 'same_runtime':
            pass
        elif fault == 'stale_hash': changes['parent_hash'] = 'sha256:' + '0' * 64
        elif fault == 'accepted': q.execute("UPDATE lab_arena_runs SET status='accepted',terminal_cause='accepted',output_ref='fixture'WHERE run_id=%s", (p.original_parent['run_id'],))
        elif fault == 'later':
            row = dict(p.parent);row.update(run_id=row['assignment_id'] + ':99', attempt=99, claim_request_id=None)
            for key in ['created_at', 'updated_at', 'reviewer_runtime_doc']: row.pop(key, None)
            old.insert(q, 'lab_arena_runs', row)
        elif fault == 'calls_pending': old.insert(q, 'lab_arena_ledger', dict(entry_kind='reservation', run_id=p.parent['run_id'], round_id=old.ROUND, submission_id=p.parent['submission_id'], miner_hotkey=p.parent['miner_hotkey'], stage=2, call_identity='sha256:'+'7'*64, provider='openrouter', operation_id='openrouter.chat', funding_source='host' if p.organizer else 'miner_key', amount_microusd=500, entry_doc={}))
        elif fault == 'expired': q.execute("UPDATE lab_arena_rounds SET configuration_doc=jsonb_set(configuration_doc,'{schedule,final_scoring_close}',to_jsonb('2000-01-01T00:00:00Z'::text))WHERE round_id=%s", (old.ROUND,))
        elif fault == 'published': q.execute("UPDATE lab_arena_rounds SET status='published',published_at=clock_timestamp(),publication_doc='{}'WHERE round_id=%s", (old.ROUND,))
        elif fault == 'generation': q.execute('UPDATE lab_arena_rounds SET scoring_generation_floor=1 WHERE round_id=%s', (old.ROUND,))
        elif fault.startswith('journal_'):
            field, value = {'journal_identity': ('{child_identity,judgment_input_hash}', json.dumps('tampered')),
                            'journal_operation': ('{operation}', json.dumps('damaged-operation')),
                            'journal_kind': ('{recovery_kind}', json.dumps('damaged-kind')),
                            'journal_parent_id': ('{parent_run_id}', json.dumps('missing-parent')),
                            'journal_child_id': ('{child_identity,run_id}', json.dumps('missing-child')),
                            'journal_request_hash': ('{correction_ref}', json.dumps('changed-proof')),
                            'journal_source': ('{child_identity,scored_run_id}', json.dumps('cross-source')),
                            'journal_floor': ('{child_identity,scoring_generation_floor}', '123'),
                            'journal_organizer': ('{organizer_openrouter_authorized}', 'false' if p.organizer else 'true'),
                            'journal_runtime': ('{runtime_image_digest}', json.dumps('sha256:'+'0'*64)),
                            'journal_parent_hash': ('{parent_sha256}', json.dumps('sha256:'+'0'*64))}[fault]
            q.execute('UPDATE lab_arena_scoring_supersessions SET recovery_doc=jsonb_set(recovery_doc,%s,%s::jsonb)WHERE recovery_doc#>>\'{child_identity,run_id}\'=%s', (field, value, p.parent['run_id']))
        elif fault == 'prior_correction':
            row = old.scalar(q, 'SELECT to_jsonb(h)FROM lab_arena_scoring_supersessions h WHERE recovery_doc#>>\'{child_identity,run_id}\'=%s', (p.parent['run_id'],))
            row.update(generation_floor=row['generation_floor']+1000, request_id='prior-scorer-correction')
            row['recovery_doc']['recovery_kind']='operator_scorer_bug_recovery_v1'
            old.insert(q,'lab_arena_scoring_supersessions',row)
        q.execute('SET LOCAL session_replication_role=origin')
        q.execute('ALTER TABLE lab_arena_runs ENABLE ALWAYS TRIGGER lab_arena_runs_terminal')
    p.c.commit()
    if fault == 'same_runtime': mapping(p, base.RUNTIME);changes['runtime'] = base.RUNTIME
    before = base.snapshot(p)
    with pytest.raises(p.pg.Error):
        if fault == 'provider_retry': provider.grant(p, request='no-provider-provider')
        else: grant(p, **changes)
    p.c.rollback()
    assert base.snapshot(p) == before


def test_migration_replay_preserves_state(corrected):
    p = corrected
    before = base.snapshot(p)
    with p.c.cursor() as q: q.execute(SQL.read_text())
    p.c.commit()
    assert base.snapshot(p) == before


@pytest.mark.parametrize('check', [base.test_prior_costs_still_exhaust_the_same_cumulative_budget,
                                  base.test_existing_stage_call_quota_is_not_reset_by_the_grant])
def test_correction_retains_original_cost_and_quota_caps(corrected, monkeypatch, check):
    monkeypatch.setattr(base, 'grant', grant)
    check(corrected)


@pytest.mark.parametrize('target', ['common', 'host'])
@pytest.mark.parametrize('fault', ['source', 'volatility', 'language', 'strict', 'leakproof', 'parallel', 'acl'])
def test_migration_replay_rejects_changed_source_or_metadata(corrected, target, fault):
    p = corrected
    sig = COMMON if target == 'common' else provider.HELPERS[1]
    with p.c.cursor() as q:
        original = old.scalar(q, 'SELECT pg_get_functiondef(%s::regprocedure)', (sig,))
        if fault == 'source':
            q.execute(original.replace('BEGIN', 'BEGIN\n-- tamper', 1) if target == 'common'
                      else original.replace('SELECT coalesce', 'SELECT /* tamper */ coalesce', 1))
        elif fault == 'volatility': q.execute('ALTER FUNCTION ' + sig + ' IMMUTABLE')
        elif fault == 'language':
            q.execute('SET check_function_bodies=off')
            q.execute(original.replace('LANGUAGE plpgsql', 'LANGUAGE sql') if target == 'common'
                      else original.replace('LANGUAGE sql', 'LANGUAGE plpgsql'))
            q.execute('SET check_function_bodies=on')
        elif fault == 'strict': q.execute('ALTER FUNCTION ' + sig + ' STRICT')
        elif fault == 'leakproof': q.execute('ALTER FUNCTION ' + sig + ' LEAKPROOF')
        elif fault == 'parallel': q.execute('ALTER FUNCTION ' + sig + ' PARALLEL SAFE')
        elif fault == 'acl': q.execute('GRANT EXECUTE ON FUNCTION ' + sig + ' TO lab_arena_service')
    p.c.commit()
    before = base.snapshot(p)
    with pytest.raises(p.pg.Error):
        with p.c.cursor() as q: q.execute(SQL.read_text())
    p.c.rollback()
    assert base.snapshot(p) == before
    with p.c.cursor() as q:
        q.execute(original)
        q.execute('REVOKE ALL ON FUNCTION ' + sig + ' FROM lab_arena_service')
    p.c.commit()


def test_first_migration_changes_only_two_function_bodies_and_no_data(correctable):
    p = correctable
    def catalog():
        with p.c.cursor() as q:
            return old.scalar(q, "SELECT jsonb_object_agg(p.oid::regprocedure::text,to_jsonb(p))FROM pg_proc p WHERE p.pronamespace='public'::regnamespace")
    before_data, before = base.snapshot(p), catalog()
    with p.c.cursor() as q: q.execute(SQL.read_text())
    p.c.commit()
    after = catalog()
    assert before_data == base.snapshot(p)
    changed = {sig for sig in before if before[sig] != after[sig]}
    with p.c.cursor() as q:
        expected = {old.scalar(q, 'SELECT %s::regprocedure::text', (sig,)) for sig in [COMMON, provider.HELPERS[1]]}
    assert set(before) == set(after)
    assert changed == expected
    for sig in changed:
        assert {k:v for k,v in before[sig].items() if k != 'prosrc'} == {k:v for k,v in after[sig].items() if k != 'prosrc'}
    # Return fixture functions to the installed673 state for outer teardown.
    with p.c.cursor() as q:
        for sig in changed:
            old_source = before[sig]['prosrc']
            definition = old.scalar(q, 'SELECT pg_get_functiondef(%s::regprocedure)', (sig,))
            q.execute(definition.replace(after[sig]['prosrc'], old_source))
    p.c.commit()


def test_corrective_mapping_change_prevents_new_claim(corrected):
    p = corrected
    child = grant(p)['child_identity']['run_id']
    mapping(p, 'sha256:' + '4' * 64)
    with p.c.cursor() as q:
        q.execute("UPDATE lab_arena_runs SET created_at=clock_timestamp()-interval '1 hour'WHERE run_id=%s", (child,))
    p.c.commit()
    assert old.claim(p)['status'] != 'leased'


def test_service_cannot_use_correction_authority(corrected):
    p = corrected
    with pytest.raises(p.pg.errors.InsufficientPrivilege):
        old.rpc(p, base.GRANT, (p.parent['run_id'], p.parent_hash, RUNTIME,
            'service-correction-denied', 'service', 'reviewed-fix', False))
