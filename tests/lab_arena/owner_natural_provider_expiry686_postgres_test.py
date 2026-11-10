"""A finite provider lease can receive one explicit owner grant after natural expiry.

Disposable fixtures age the whole captured claim window before invoking native
expiry. They never fabricate terminal receipts or shorten only a lease deadline.
No live lease, provider, credential, or production database is used.
"""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from tests.lab_arena import owner_provider_judge_budget685_postgres_test as budget

f = budget.f
finite = budget.finite
prior = budget.prior
base = budget.base
old = budget.old
ROOT = Path(__file__).parents[2]
SQL = ROOT / 'scripts/686-lab-arena-owner-natural-provider-expiry.sql'
COMMON = budget.COMMON
database = budget.database
prepared = budget.prepared
prepared_company = budget.prepared_company
case = budget.case
base_extended = budget.base_extended
extended = budget.extended
provider_case = budget.provider_case
before = f.compatible_plain
post = budget.post
installed = budget.installed


@pytest.fixture
def plain_finite(provider_case):
    # Construct each payer once; the new provider parent is authored natively.
    yield from finite.plain_finite.__wrapped__(provider_case, SimpleNamespace(param='ordinary'))


def baseline(p):
    submission_id = 'baseline-' + old.ROUND.removeprefix('arena-')
    assignment = old.ROUND + ':' + submission_id + ':1:0'
    source_id = assignment + ':1'
    score_assignment = assignment + ':score'
    with p.c.cursor() as q:
        q.execute('SET LOCAL session_replication_role=replica')
        cfg = old.scalar(q, 'SELECT configuration_doc FROM lab_arena_rounds WHERE round_id=%s', (old.ROUND,))
        submission = old.scalar(q, 'SELECT to_jsonb(s) FROM lab_arena_submissions s WHERE submission_id=%s', (p.parent['submission_id'],))
        submission.update(submission_id=submission_id, miner_hotkey=cfg['baseline_hotkey'], is_king=True)
        old.insert(q, 'lab_arena_submissions', submission)
        source = old.scalar(q, 'SELECT to_jsonb(r) FROM lab_arena_runs r WHERE run_id=%s', (p.parent['scored_run_id'],))
        source.update(run_id=source_id, assignment_id=assignment, submission_id=submission_id,
                      miner_hotkey=cfg['baseline_hotkey'], stage=1)
        old.insert(q, 'lab_arena_runs', source)
        q.execute('ALTER TABLE lab_arena_runs DISABLE TRIGGER lab_arena_runs_reviewer_runtime')
        scores = old.scalar(q, 'SELECT jsonb_agg(to_jsonb(r) ORDER BY attempt) FROM lab_arena_runs r WHERE assignment_id=%s', (p.parent['assignment_id'],))
        for row in scores:
            row.update(run_id=score_assignment + ':' + str(row['attempt']), assignment_id=score_assignment,
                       submission_id=submission_id, miner_hotkey=cfg['baseline_hotkey'], stage=1, scored_run_id=source_id)
            old.insert(q, 'lab_arena_runs', row)
        q.execute('ALTER TABLE lab_arena_runs ENABLE ALWAYS TRIGGER lab_arena_runs_reviewer_runtime')
        plan = {'work_items': [{'submission_id': submission_id, 'icp_position': 0,
                               'scored_run_id': source_id, 'output_ref': source['output_ref']}]}
        q.execute("UPDATE lab_arena_rounds SET status='stage1_scoring',stage1_scoring_plan_doc=%s::jsonb,participants=participants||%s::jsonb WHERE round_id=%s", (
            json.dumps(plan), json.dumps([{'submission_id': submission_id, 'miner_hotkey': cfg['baseline_hotkey'], 'is_king': True}]), old.ROUND))
        q.execute('SET LOCAL session_replication_role=origin')
    p.c.commit()
    prior.saved_parent(p, score_assignment + ':' + str(p.parent['attempt']))
    p.funding_source = 'host'


@pytest.fixture(params=['miner', 'baseline'])
def provider_parent(installed, request):
    p = installed
    if request.param == 'baseline':
        if p.organizer:
            pytest.skip('Deduplicate baseline construction across inherited payer cases.')
        baseline(p)
    budget.upgrade(p)
    budget.diagnostic(p, stage='scorer', error='judge_error', reason='provider_error')
    p.proof = budget.receipt(p, status=429)
    p.original_parent = dict(p.parent)
    child = prior.provider.grant(p, request='natural-expiry-provider-parent')['child_identity']['run_id']
    p.provider_lease = prior.claim_child(p, child)
    prior.saved_parent(p, child)
    p.expected_funding = getattr(p, 'funding_source', 'host' if p.organizer else 'miner_key')
    return p


def expire(p):
    with p.c.cursor() as q:
        # Advance the entire disposable claim window, retaining its equality.
        # Native expiry alone writes the final server expired_at receipt.
        q.execute("""WITH elapsed AS MATERIALIZED (
          SELECT clock_timestamp()-interval '1 second' AS ended)
          UPDATE lab_arena_runs SET created_at=created_at-interval '1 hour',
            lease_expires_at=elapsed.ended,
            claim_response=jsonb_set(claim_response,'{lease_expires_at}',to_jsonb(elapsed.ended))
          FROM elapsed WHERE run_id=%s""", (p.parent['run_id'],))
    p.c.commit()
    response = old.rpc(p, 'lab_arena_expire_leases(%s)', (old.ROUND,))
    with p.c.cursor() as q:
        assert old.scalar(q, 'SELECT count(*) FROM lab_arena_runs WHERE assignment_id=%s AND attempt>%s', (p.parent['assignment_id'], p.parent['attempt'])) == 0
    p.c.commit()
    prior.saved_parent(p, p.parent['run_id'])
    assert p.parent['terminal_cause'] == 'lease_expired'
    assert p.parent['result_doc'] is None and p.parent['output_ref'] is None
    assert list(p.parent['terminal_doc']) == ['expired_at']
    return p


@pytest.fixture
def expired(provider_parent):
    return expire(provider_parent)


def upgrade(p):
    with p.c.cursor() as q:
        q.execute(SQL.read_text())
    p.c.commit()


def test_before_migration_refuses_authentic_expired_provider_child(expired):
    p = expired
    rows = base.snapshot(p)
    with pytest.raises(p.pg.Error, match='failed parent'):
        finite.grant(p)
    p.c.rollback()
    assert base.snapshot(p) == rows


@pytest.mark.parametrize('outcome', ['accepted', 'judge_error', 'expired'])
def test_preview_append_claim_terminal_and_replay_preserve_native_state(expired, outcome):
    p = expired
    rows = base.snapshot(p)
    with p.c.cursor() as q:
        metadata = old.scalar(q, "SELECT to_jsonb(p)-'prosrc' FROM pg_proc p WHERE oid=%s::regprocedure", (COMMON,))
        unchanged = [old.scalar(q, 'SELECT pg_get_functiondef(%s::regprocedure)', (sig,)) for sig in prior.provider.HELPERS + old.SIGS + ['public.lab_arena__score_attempt_count_v1(public.lab_arena_runs)']]
    p.c.commit()
    upgrade(p)
    upgrade(p)
    assert base.snapshot(p) == rows
    with p.c.cursor() as q:
        assert old.scalar(q, "SELECT to_jsonb(p)-'prosrc' FROM pg_proc p WHERE oid=%s::regprocedure", (COMMON,)) == metadata
        assert [old.scalar(q, 'SELECT pg_get_functiondef(%s::regprocedure)', (sig,)) for sig in prior.provider.HELPERS + old.SIGS + ['public.lab_arena__score_attempt_count_v1(public.lab_arena_runs)']] == unchanged
    p.c.commit()
    preview = finite.grant(p, dry=True)
    assert base.snapshot(p) == rows
    result = finite.grant(p)
    assert result['child_identity'] == preview['child_identity']
    assert result['checkpoints'] == p.checkpoints and len(result['checkpoints']) == 2
    assert result['signal_checkpoints'] == p.signal_checkpoints
    after = base.snapshot(p)
    originals = {r['run_id']: r for r in rows['lab_arena_runs']}
    assert {r['run_id']: r for r in after['lab_arena_runs'] if r['run_id'] in originals} == originals
    assert len(after['lab_arena_runs']) == len(rows['lab_arena_runs']) + 1
    assert len(after['lab_arena_scoring_supersessions']) == len(rows['lab_arena_scoring_supersessions']) + 1
    for table in set(rows) - {'lab_arena_runs', 'lab_arena_scoring_supersessions'}:
        assert after[table] == rows[table]
    assert finite.grant(p)['idempotent'] is True and base.snapshot(p) == after
    child = result['child_identity']['run_id']
    assert old.funding(p, child)['funding_source'] == p.expected_funding
    assert old.funding(p, child, 'deepline')['funding_source'] == ('host' if p.parent['stage'] == 1 else 'miner_key')
    lease = prior.claim_child(p, child)
    assert lease['company_judgment_cache']['misses'] == []
    with p.c.cursor() as q:
        assert old.scalar(q, "SELECT reviewer_runtime_doc->>'runtime_image_digest' FROM lab_arena_runs WHERE run_id=%s", (child,)) == p.exception_runtime
    p.c.commit()
    if outcome == 'expired':
        saved = (p.parent, p.parent_hash)
        prior.saved_parent(p, child)
        expire(p)
        p.parent, p.parent_hash = saved
    else:
        old.rpc(p, 'lab_arena_complete_attempt(%s,%s,%s::jsonb,%s,%s)', (
            child, old.TOKEN, json.dumps({'terminal_status': outcome, 'failure_diagnostic': {
                'stage': 'scorer', 'error_class': 'judge_error', 'reason': 'malformed_response'}}),
            outcome, child + '-output' if outcome == 'accepted' else None))
    old.rpc(p, 'lab_arena_expire_leases(%s)', (old.ROUND,))
    assert finite.grant(p)['idempotent'] is True
    prior.saved_parent(p, child)
    after = base.snapshot(p)
    with pytest.raises(p.pg.Error):
        finite.grant(p, request_id='no-second-expiry-budget')
    p.c.rollback()
    assert base.snapshot(p) == after
    with p.c.cursor() as q:
        assert old.scalar(q, 'SELECT count(*) FROM lab_arena_runs WHERE assignment_id=%s AND attempt>%s', (lease['assignment_id'], lease['attempt'])) == 0
    p.c.commit()

@pytest.mark.parametrize('fault', [
    'live', 'shortened', 'future_expiry', 'early_receipt', 'missing_receipt', 'extra_receipt', 'bad_timestamp',
    'result', 'output', 'claim_missing', 'claim_run', 'claim_assignment', 'claim_source', 'claim_request',
    'claim_generation', 'claim_stage_generation', 'claim_runtime', 'claim_stage', 'claim_icp', 'claim_submission',
    'claim_miner', 'claim_kind', 'claim_attempt', 'token', 'request_hash', 'runner', 'other_runner',
    'exclusions', 'journal_hash', 'journal_kind', 'journal_child', 'journal_parent_hash', 'journal_parent',
    'journal_runtime', 'journal_reference', 'journal_floor', 'journal_organizer', 'journal_operation',
    'original_changed', 'later', 'dispatch', 'unknown_success', 'deadline', 'publication_deadline',
    'generation', 'frozen_source', 'stale_hash', 'checkpoints', 'signal_checkpoints', 'runtime'])
def test_invalid_expiry_authority_refuses_without_writes(expired, fault):
    p = expired
    upgrade(p)
    changes = {}
    claim_fields = {
        'claim_run': ('run_id', 'cross-run'), 'claim_assignment': ('assignment_id', 'cross-assignment'),
        'claim_source': ('scored_run_id', 'cross-source'), 'claim_request': ('request_id', '0' * 32),
        'claim_generation': ('lease_generation', 999), 'claim_stage_generation': ('stage_generation', 999),
        'claim_stage': ('stage', 99), 'claim_icp': ('icp_position', 99),
        'claim_submission': ('submission_id', 'cross-submission'), 'claim_miner': ('miner_hotkey', 'cross-miner'),
        'claim_kind': ('kind', 'execute'), 'claim_attempt': ('attempt', 999),
    }
    with p.c.cursor() as q:
        q.execute('ALTER TABLE lab_arena_runs DISABLE TRIGGER lab_arena_runs_terminal')
        q.execute('SET LOCAL session_replication_role=replica')
        if fault == 'live':
            q.execute("UPDATE lab_arena_runs SET status='leased' WHERE run_id=%s", (p.parent['run_id'],))
        elif fault == 'shortened':
            q.execute("UPDATE lab_arena_runs SET lease_expires_at=(claim_response->>'lease_expires_at')::timestamptz-interval '1 second' WHERE run_id=%s", (p.parent['run_id'],))
        elif fault == 'future_expiry':
            q.execute("UPDATE lab_arena_runs SET terminal_doc=jsonb_build_object('expired_at',clock_timestamp()+interval '1 hour') WHERE run_id=%s", (p.parent['run_id'],))
        elif fault == 'early_receipt':
            q.execute("UPDATE lab_arena_runs SET terminal_doc=jsonb_build_object('expired_at',lease_expires_at-interval '1 second') WHERE run_id=%s", (p.parent['run_id'],))
        elif fault == 'missing_receipt':
            q.execute('UPDATE lab_arena_runs SET terminal_doc=NULL WHERE run_id=%s', (p.parent['run_id'],))
        elif fault == 'extra_receipt':
            q.execute("UPDATE lab_arena_runs SET terminal_doc=terminal_doc||'{\"recovery_reason\":\"early_host\"}'::jsonb WHERE run_id=%s", (p.parent['run_id'],))
        elif fault == 'bad_timestamp':
            q.execute("UPDATE lab_arena_runs SET terminal_doc='{\"expired_at\":\"invalid\"}' WHERE run_id=%s", (p.parent['run_id'],))
        elif fault in ('result', 'output'):
            q.execute('UPDATE lab_arena_runs SET ' + ('result_doc=\'{}\'::jsonb' if fault == 'result' else "output_ref='saved-output'") + ' WHERE run_id=%s', (p.parent['run_id'],))
        elif fault == 'claim_missing':
            q.execute('UPDATE lab_arena_runs SET claim_response=NULL WHERE run_id=%s', (p.parent['run_id'],))
        elif fault in claim_fields:
            field, value = claim_fields[fault]
            q.execute('UPDATE lab_arena_runs SET claim_response=jsonb_set(claim_response,%s,%s::jsonb) WHERE run_id=%s', ('{' + field + '}', json.dumps(value), p.parent['run_id']))
        elif fault in ('token', 'request_hash', 'runner'):
            column = {'token': 'lease_token_hash', 'request_hash': 'claim_request_hash', 'runner': 'runner_hotkey'}[fault]
            q.execute('UPDATE lab_arena_runs SET ' + column + '=NULL WHERE run_id=%s', (p.parent['run_id'],))
        elif fault == 'other_runner':
            q.execute('UPDATE lab_arena_runs SET runner_hotkey=%s WHERE run_id=%s', (p.parent['miner_hotkey'], p.parent['run_id']))
        elif fault == 'claim_runtime':
            q.execute("UPDATE lab_arena_runs SET claim_response=jsonb_set(claim_response,'{reviewer_runtime_doc}','{}') WHERE run_id=%s", (p.parent['run_id'],))
        elif fault == 'exclusions':
            q.execute("UPDATE lab_arena_runs SET claim_response=jsonb_set(claim_response,'{runner_authority_exclusions}','[]') WHERE run_id=%s", (p.parent['run_id'],))
        elif fault.startswith('journal_'):
            field, value = {
                'journal_hash': ('approval_ref', 'rehashed-no-authority'),
                'journal_kind': ('recovery_kind', 'operator_scorer_bug_recovery_v1'),
                'journal_child': ('child_identity,scored_run_id', 'cross-source'),
                'journal_parent_hash': ('parent_sha256', 'sha256:' + '0' * 64),
                'journal_parent': ('parent_run_id', 'missing-parent'),
                'journal_runtime': ('runtime_image_digest', 'sha256:' + '0' * 64),
                'journal_reference': ('runtime_image_reference', 'registry.example/wrong@sha256:' + '0' * 64),
                'journal_floor': ('scoring_generation_floor', 999),
                'journal_organizer': ('organizer_openrouter_authorized', not p.organizer),
                'journal_operation': ('operation', 'wrong-operation'),
            }[fault]
            q.execute('UPDATE lab_arena_scoring_supersessions SET recovery_doc=jsonb_set(recovery_doc,%s,%s::jsonb) WHERE recovery_doc#>>\'{child_identity,run_id}\'=%s', ('{' + field + '}', json.dumps(value), p.parent['run_id']))
            if fault != 'journal_hash':
                q.execute("UPDATE lab_arena_scoring_supersessions SET request_hash='sha256:'||encode(extensions.digest(recovery_doc::text,'sha256'),'hex') WHERE recovery_doc#>>'{child_identity,run_id}'=%s", (p.parent['run_id'],))
        elif fault == 'original_changed':
            q.execute("UPDATE lab_arena_runs SET output_ref='changed-original' WHERE run_id=%s", (p.original_parent['run_id'],))
        elif fault == 'later':
            row = dict(p.parent)
            row.update(run_id=row['assignment_id'] + ':99', attempt=99, status='pending', claim_request_id=None, reviewer_runtime_doc=None)
            old.insert(q, 'lab_arena_runs', row)
        elif fault in ('deadline', 'publication_deadline'):
            path = '{schedule,final_scoring_close}' if fault == 'deadline' else '{schedule,publication_deadline}'
            q.execute("UPDATE lab_arena_rounds SET configuration_doc=jsonb_set(configuration_doc,%s,to_jsonb('2000-01-01T00:00:00Z'::text)) WHERE round_id=%s", (path, old.ROUND))
        elif fault == 'generation':
            q.execute('UPDATE lab_arena_rounds SET scoring_generation_floor=999 WHERE round_id=%s', (old.ROUND,))
        elif fault == 'frozen_source':
            q.execute("UPDATE lab_arena_runs SET judgment_input_hash='sha256:'||repeat('0',64) WHERE run_id=%s", (p.parent['run_id'],))
        elif fault == 'dispatch':
            row = dict(entry_kind='dispatch', run_id=p.parent['run_id'], round_id=old.ROUND,
                       submission_id=p.parent['submission_id'], miner_hotkey=p.parent['miner_hotkey'], stage=p.parent['stage'],
                       call_identity='sha256:' + '7' * 64, provider='openrouter', operation_id='openrouter.chat',
                       funding_source=p.expected_funding, amount_microusd=0, entry_doc={})
            old.insert(q, 'lab_arena_ledger', row)
        q.execute('SET LOCAL session_replication_role=origin')
        q.execute('ALTER TABLE lab_arena_runs ENABLE ALWAYS TRIGGER lab_arena_runs_terminal')
    p.c.commit()
    if fault == 'unknown_success':
        budget.receipt(p, status=200, success=True, kind='uncertain')
    prior.saved_parent(p, p.parent['run_id'])
    if fault == 'stale_hash':
        changes['parent_hash'] = 'sha256:' + '0' * 64
    elif fault in ('checkpoints', 'signal_checkpoints'):
        changes[fault] = {'unknown': 'hash'}
    elif fault == 'runtime':
        changes['runtime'] = prior.RUNTIME
    rows = base.snapshot(p)
    with pytest.raises(p.pg.Error):
        finite.grant(p, **changes)
    p.c.rollback()
    assert base.snapshot(p) == rows


@pytest.mark.parametrize('role', ['lab_arena_service', 'anon', 'authenticated', 'service_role'])
def test_expiry_budget_remains_owner_only(expired, role):
    upgrade(expired)
    finite.test_owner_only(expired, role)


@pytest.mark.parametrize('kind', ['provider', 'scorer'])
def test_expiry_shape_does_not_expand_legacy_grants(expired, kind):
    p = expired
    upgrade(p)
    rows = base.snapshot(p)
    with pytest.raises(p.pg.Error):
        if kind == 'provider':
            prior.provider.grant(p, request='no-recursive-expiry-provider')
        else:
            prior.grant(p, request_id='no-unaudited-expiry-scorer')
    p.c.rollback()
    assert base.snapshot(p) == rows


@pytest.mark.parametrize('shape', ['scorer', 'sandbox', 'provider'])
def test_previous_judge_error_parent_shapes_stay_usable(installed, shape):
    p = installed
    budget.upgrade(p)
    upgrade(p)
    budget.diagnostic(p, stage={'scorer': 'scorer', 'sandbox': 'sandbox', 'provider': 'provider_call'}[shape],
                      error={'scorer': 'judge_error', 'sandbox': 'judge_timeout', 'provider': 'provider_unavailable'}[shape])
    if shape == 'provider':
        budget.receipt(p, status=429)
    child = finite.grant(p)['child_identity']['run_id']
    prior.claim_child(p, child)

@pytest.mark.parametrize('fault', ['money', 'quota'])
def test_expiry_grant_preserves_cumulative_money_and_stage_quota(expired, fault):
    p = expired
    with p.c.cursor() as q:
        cfg = old.scalar(q, 'SELECT configuration_doc FROM lab_arena_rounds WHERE round_id=%s', (old.ROUND,))
        count = 1 if fault == 'money' else cfg['scoring_call_quotas']['openrouter'] * cfg['stage_' + str(p.parent['stage']) + '_icp_count'] * cfg['max_attempts_per_assignment']
        q.execute('SET LOCAL session_replication_role=replica')
        q.execute("""INSERT INTO lab_arena_ledger(entry_kind,run_id,round_id,submission_id,miner_hotkey,
          stage,call_identity,provider,operation_id,funding_source,amount_microusd,entry_doc,terminal_response)
          SELECT kind,%s,%s,%s,%s,%s,'sha256:'||encode(extensions.digest('quota686-'||n::text,'sha256'),'hex'),
            'openrouter','openrouter.chat',%s,CASE WHEN kind='settlement' THEN %s ELSE 0 END,'{}',
            CASE WHEN kind='settlement' THEN '{"status":200,"call_succeeded":true}'::jsonb ELSE NULL END
          FROM generate_series(1,%s)n CROSS JOIN(VALUES('reservation',1),('settlement',2))k(kind,ordering)
          ORDER BY n,ordering""", (p.parent['run_id'], old.ROUND, p.parent['submission_id'], p.parent['miner_hotkey'],
            p.parent['stage'], p.expected_funding, cfg['scoring_cap_microusd'] if fault == 'money' else 0, count))
        q.execute('SET LOCAL session_replication_role=origin')
    p.c.commit()
    upgrade(p)
    ledger_before = base.snapshot(p)['lab_arena_ledger']
    child = finite.grant(p)['child_identity']['run_id']
    lease = prior.claim_child(p, child)
    request_hash = 'sha256:' + '6' * 64
    call = base.contracts.provider_call_identity(assignment_id=lease['assignment_id'], attempt=lease['attempt'],
        icp_position=lease['icp_position'], action_sequence=1, operation_id='openrouter.chat', request_hash=request_hash)
    result = old.rpc(p, 'lab_arena_reserve_call(%s,%s,%s,%s,%s,%s,%s,%s::jsonb,%s)', (
        child, old.TOKEN, call, 'openrouter.chat', 'openrouter', p.expected_funding,
        500, json.dumps(dict(request_hash=request_hash, base_call_identity=call, action_sequence=1,
                            provider_attempt=1, max_output_tokens=8)), 1500))
    assert result['status'] == 'refused'
    assert result['reason'] == ('money_cap' if fault == 'money' else 'stage_quota')
    assert [r for r in base.snapshot(p)['lab_arena_ledger'] if r['call_identity'] != call] == ledger_before


def test_failed_uncertain_liability_and_attempt_counter_remain_intact(provider_parent):
    p = provider_parent
    entry = budget.receipt(p, status=429)
    with p.c.cursor() as q:
        q.execute('SET LOCAL session_replication_role=replica')
        episode = old.scalar(q, 'SELECT jsonb_agg(to_jsonb(l) ORDER BY entry_id) FROM lab_arena_ledger l WHERE call_identity=(SELECT call_identity FROM lab_arena_ledger WHERE entry_id=%s)', (entry,))
        for row in episode:
            row.pop('entry_id')
            row['call_identity'] = 'sha256:' + '8' * 64
            row['amount_microusd'] = 500 if row['entry_kind'] in ('reservation', 'uncertain') else 0
            if row['entry_kind'] == 'uncertain':
                row['entry_doc']['reserved_microusd'] = 500
                row['entry_doc']['call']['upstream_response_receipt']['call_identity'] = row['call_identity']
            old.insert(q, 'lab_arena_ledger', row)
        q.execute('SET LOCAL session_replication_role=origin')
    p.c.commit()
    expire(p)
    with p.c.cursor() as q:
        assert old.scalar(q, 'SELECT public.lab_arena__judge_calls_clear_v1(%s)', (p.parent['assignment_id'],)) is True
        count_before = old.scalar(q, 'SELECT public.lab_arena__score_attempt_count_v1(r) FROM lab_arena_runs r WHERE run_id=%s', (p.parent['run_id'],))
    p.c.commit()
    ledger_before = base.snapshot(p)['lab_arena_ledger']
    upgrade(p)
    child = finite.grant(p)['child_identity']['run_id']
    assert base.snapshot(p)['lab_arena_ledger'] == ledger_before
    with p.c.cursor() as q:
        assert old.scalar(q, 'SELECT public.lab_arena__score_attempt_count_v1(r) FROM lab_arena_runs r WHERE run_id=%s', (p.parent['run_id'],)) == count_before
        assert old.scalar(q, 'SELECT public.lab_arena__score_attempt_count_v1(r) FROM lab_arena_runs r WHERE run_id=%s', (child,)) == count_before + 1
    p.c.commit()
    prior.claim_child(p, child)
    assert old.funding(p, child)['funding_source'] == p.expected_funding


@pytest.mark.parametrize('drift', ['source', 'acl', 'owner', 'child_source', 'child_acl', 'retained_source'])
def test_unknown_function_source_or_metadata_refuses_atomically(expired, drift):
    p = expired
    target = prior.provider.HELPERS[0] if drift.startswith('child') else budget.PROOF if drift.startswith('retained') else COMMON
    with p.c.cursor() as q:
        definition = old.scalar(q, 'SELECT pg_get_functiondef(%s::regprocedure)', (target,))
        metadata = old.scalar(q, 'SELECT to_jsonb(p) FROM pg_proc p WHERE oid=%s::regprocedure', (target,))
        if drift in ('source', 'child_source', 'retained_source'):
            q.execute(definition.replace('AS $function$', 'AS $function$\n-- unreviewed source\n', 1))
        elif drift in ('acl', 'child_acl'):
            q.execute('GRANT EXECUTE ON FUNCTION ' + target + ' TO anon')
        else:
            q.execute('ALTER FUNCTION ' + target + ' OWNER TO anon')
        with pytest.raises(p.pg.Error, match='differs'):
            q.execute(SQL.read_text())
    p.c.rollback()
    with p.c.cursor() as q:
        assert old.scalar(q, 'SELECT to_jsonb(p) FROM pg_proc p WHERE oid=%s::regprocedure', (target,)) == metadata
    p.c.commit()
