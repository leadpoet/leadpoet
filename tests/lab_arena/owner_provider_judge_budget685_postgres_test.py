"""A proved final provider failure can use the existing one-time owner grant."""
import base64
import hashlib
import json
import uuid
from pathlib import Path

import pytest
from tests.lab_arena import finite_child_scorer_correction679_postgres_test as previous

f = previous.f
finite = f.old
prior = finite.prior
base = finite.base
old = finite.old
ROOT = Path(__file__).parents[2]
SQL = ROOT / 'scripts/685-lab-arena-owner-provider-judge-budget-proof.sql'
COMMON = previous.COMMON
PROOF = 'public.lab_arena__retained_provider_transient_v1(public.lab_arena_ledger)'
database = f.database
prepared = f.prepared
prepared_company = f.prepared_company
case = f.case
base_extended = f.base_extended
extended = f.extended
provider_case = f.provider_case
plain_finite = f.plain_finite
before = f.compatible_plain
post = previous.post


@pytest.fixture
def installed(post):
    p = post
    with p.c.cursor() as q:
        originals = [old.scalar(q, 'SELECT pg_get_functiondef(%s::regprocedure)', (sig,))
                     for sig in [COMMON, previous.HOST, 'public.lab_arena__score_attempt_count_v1(public.lab_arena_runs)']]
        q.execute(previous.SQL.read_text())
        q.execute((ROOT / 'scripts/680-lab-arena-proved-infrastructure-judge-attempt-credit.sql').read_text())
    p.c.commit()
    #671's ordinary fixture has an older valid429. A final non-transient
    #synthetic episode makes it stale; each test appends its own final proof.
    receipt(p, status=200, success=False)
    try:
        yield p
    finally:
        p.c.rollback()
        with p.c.cursor() as q:
            for definition in originals:
                q.execute(definition)
        p.c.commit()


def diagnostic(p, *, stage='provider_call', error='provider_unavailable', reason='provider_error', unresolved=1):
    with p.c.cursor() as q:
        q.execute('ALTER TABLE lab_arena_runs DISABLE TRIGGER lab_arena_runs_terminal')
        q.execute('SET LOCAL session_replication_role=replica')
        q.execute("UPDATE lab_arena_runs SET result_doc=%s::jsonb WHERE run_id=%s", (json.dumps({
            'terminal_status': 'judge_error', 'failure_diagnostic': {
                'stage': stage, 'error_class': error, 'reason': reason},
            'resource_summary': {'provider_unresolved_failure_count': unresolved}}), p.parent['run_id']))
        q.execute('SET LOCAL session_replication_role=origin')
        q.execute('ALTER TABLE lab_arena_runs ENABLE ALWAYS TRIGGER lab_arena_runs_terminal')
    p.c.commit()
    prior.saved_parent(p, p.parent['run_id'])


def receipt(p, *, status=502, success=False, kind=None, corrupt=False, mismatch=None):
    #429 is the existing retained shared-provider-pool uncertain receipt proof.
    kind = kind or ('uncertain' if status == 429 else 'settlement')
    body = {'error': {'code': status, 'metadata': {'is_byok': False,
            'limit_source': 'upstream_provider_shared_pool'}}}
    raw = json.dumps(body).encode()
    cid = 'sha256:' + hashlib.sha256(raw + str(success).encode() + uuid.uuid4().bytes).hexdigest()
    request_hash = 'sha256:' + '4' * 64
    call = {'call_succeeded': success, 'status': status, 'upstream_response_receipt': {
        'status': status, 'call_succeeded': success, 'call_identity': cid,
        'operation_id': 'openrouter.chat', 'request_hash': request_hash,
        'body_b64': base64.b64encode(raw).decode(),
        'body_sha256': 'sha256:' + ('0' * 64 if corrupt else hashlib.sha256(raw).hexdigest())}}
    with p.c.cursor() as q:
        q.execute('SET LOCAL session_replication_role=replica')
        for entry_kind in ['reservation', 'dispatch', kind]:
            row = dict(entry_kind=entry_kind, run_id=p.parent['run_id'], round_id=old.ROUND,
                       submission_id=p.parent['submission_id'], miner_hotkey=p.parent['miner_hotkey'],
                       stage=p.parent['stage'], call_identity=cid, provider='openrouter', operation_id='openrouter.chat',
                       funding_source=getattr(p, 'funding_source', 'host' if p.organizer else 'miner_key'), amount_microusd=0,
                       entry_doc={'request_hash': request_hash} if entry_kind == 'reservation' else
                                 {'call': call} if entry_kind == 'uncertain' else {},
                       terminal_response=call if entry_kind == 'settlement' else None)
            if mismatch:
                row.update(mismatch)
            old.insert(q, 'lab_arena_ledger', row)
        q.execute('SET LOCAL session_replication_role=origin')
        entry = old.scalar(q, 'SELECT max(entry_id) FROM lab_arena_ledger WHERE call_identity=%s', (cid,))
    p.c.commit()
    return entry


def upgrade(p):
    with p.c.cursor() as q:
        q.execute(SQL.read_text())
    p.c.commit()


def test_reproduces_rejected_proved_final_provider_parent(installed):
    p = installed
    diagnostic(p)
    entry = receipt(p)
    with p.c.cursor() as q:
        assert old.scalar(q, 'SELECT public.lab_arena__retained_provider_transient_v1(l) FROM lab_arena_ledger l WHERE entry_id=%s', (entry,)) is True
        assert old.scalar(q, 'SELECT public.lab_arena__judge_calls_clear_v1(%s)', (p.parent['assignment_id'],)) is True
    p.c.commit()
    rows = base.snapshot(p)
    with pytest.raises(p.pg.Error, match='failed parent'):
        finite.grant(p)
    p.c.rollback()
    assert base.snapshot(p) == rows


@pytest.mark.parametrize('status', [429, 500, 502, 503, 504])
def test_final_proved_provider_failure_preview_append_claim_complete_and_replay(installed, status):
    p = installed
    diagnostic(p)
    receipt(p, status=status)
    rows = base.snapshot(p)
    with p.c.cursor() as q:
        metadata = old.scalar(q, "SELECT to_jsonb(p)-'prosrc' FROM pg_proc p WHERE oid=%s::regprocedure", (COMMON,))
        helpers = [old.scalar(q, 'SELECT pg_get_functiondef(%s::regprocedure)', (sig,)) for sig in prior.provider.HELPERS]
    p.c.commit()
    upgrade(p)
    upgrade(p)
    assert base.snapshot(p) == rows
    with p.c.cursor() as q:
        assert old.scalar(q, "SELECT to_jsonb(p)-'prosrc' FROM pg_proc p WHERE oid=%s::regprocedure", (COMMON,)) == metadata
        assert [old.scalar(q, 'SELECT pg_get_functiondef(%s::regprocedure)', (sig,)) for sig in prior.provider.HELPERS] == helpers
    p.c.commit()
    preview = finite.grant(p, dry=True)
    assert base.snapshot(p) == rows
    result = finite.grant(p)
    assert result['child_identity'] == preview['child_identity']
    assert result['checkpoints'] == p.checkpoints and len(result['checkpoints']) == 2
    assert result['signal_checkpoints'] == p.signal_checkpoints
    after = base.snapshot(p)
    assert len(after['lab_arena_runs']) == len(rows['lab_arena_runs']) + 1
    assert len(after['lab_arena_scoring_supersessions']) == len(rows['lab_arena_scoring_supersessions']) + 1
    originals = {r['run_id']: r for r in rows['lab_arena_runs']}
    assert {r['run_id']: r for r in after['lab_arena_runs'] if r['run_id'] in originals} == originals
    for table in set(rows) - {'lab_arena_runs', 'lab_arena_scoring_supersessions'}:
        assert after[table] == rows[table]
    assert finite.grant(p)['idempotent'] is True and base.snapshot(p) == after
    child = result['child_identity']['run_id']
    assert old.funding(p, child)['funding_source'] == ('host' if p.organizer else 'miner_key')
    assert old.funding(p, child, 'deepline')['funding_source'] == 'miner_key'
    lease = prior.claim_child(p, child)
    assert lease['company_judgment_cache']['misses'] == []
    old.rpc(p, 'lab_arena_complete_attempt(%s,%s,%s::jsonb,%s,%s)', (
        child, old.TOKEN, json.dumps({'terminal_status': 'accepted'}), 'accepted', child + '-output'))
    old.rpc(p, 'lab_arena_expire_leases(%s)', (old.ROUND,))
    assert finite.grant(p)['idempotent'] is True
    with p.c.cursor() as q:
        assert old.scalar(q, 'SELECT count(*) FROM lab_arena_runs WHERE assignment_id=%s AND attempt>%s',
                          (lease['assignment_id'], lease['attempt'])) == 0
        assert old.scalar(q, 'SELECT status FROM lab_arena_runs WHERE run_id=%s', (child,)) == 'accepted'
    p.c.commit()


@pytest.mark.parametrize('fault', ['no_proof', 'forged_body', 'healed', 'later_attempt', 'cross_run',
                                  'cross_round', 'cross_submission', 'cross_miner', 'cross_stage',
                                  'auth401', 'funding402', 'auth403', 'unsupported501', 'success',
                                  'stage', 'error_class', 'reason', 'missing_stage', 'missing_error',
                                  'missing_reason', 'no_unresolved', 'string_unresolved', 'null_unresolved'])
def test_unproved_or_nonfinal_provider_parent_is_refused_without_changes(installed, fault):
    p = installed
    changes = {}
    if fault in ('stage', 'missing_stage'):
        changes['stage'] = None if fault.startswith('missing') else 'scorer'
    elif fault in ('error_class', 'missing_error'):
        changes['error'] = None if fault.startswith('missing') else 'judge_error'
    elif fault in ('reason', 'missing_reason'):
        changes['reason'] = None if fault.startswith('missing') else 'malformed_response'
    elif fault in ('no_unresolved', 'string_unresolved', 'null_unresolved'):
        changes['unresolved'] = {'no_unresolved': 0, 'string_unresolved': '1', 'null_unresolved': None}[fault]
    diagnostic(p, **changes)
    if fault != 'no_proof':
        mismatch = {'cross_run': {'run_id': p.parent['scored_run_id']},
                    'cross_round': {'round_id': 'arena-2026-11-07'},
                    'cross_submission': {'submission_id': 'cross-submission'},
                    'cross_miner': {'miner_hotkey': 'cross-miner'}, 'cross_stage': {'stage': 1}}.get(fault)
        receipt(p, status={'auth401': 401, 'funding402': 402, 'auth403': 403,
                           'unsupported501': 501, 'success': 200}.get(fault, 502),
                success=fault == 'success', corrupt=fault == 'forged_body', mismatch=mismatch)
    if fault in ('healed', 'later_attempt'):
        receipt(p, status=200, success=fault == 'healed')
    upgrade(p)
    rows = base.snapshot(p)
    with pytest.raises(p.pg.Error):
        finite.grant(p)
    p.c.rollback()
    assert base.snapshot(p) == rows


@pytest.mark.parametrize('outcome', ['judge_error', 'expired'])
def test_provider_budget_child_remains_finite(installed, outcome):
    p = installed
    diagnostic(p)
    receipt(p)
    upgrade(p)
    child = finite.grant(p)['child_identity']['run_id']
    lease = prior.claim_child(p, child)
    if outcome == 'expired':
        with p.c.cursor() as q:
            q.execute("UPDATE lab_arena_runs SET lease_expires_at=clock_timestamp()-interval '1 second' WHERE run_id=%s", (child,))
        p.c.commit()
    else:
        prior.complete_failed(p, child)
    old.rpc(p, 'lab_arena_expire_leases(%s)', (old.ROUND,))
    assert finite.grant(p)['idempotent'] is True
    prior.saved_parent(p, child)
    rows = base.snapshot(p)
    with pytest.raises(p.pg.Error):
        finite.grant(p, request_id='no-recursive-provider-budget')
    p.c.rollback()
    assert base.snapshot(p) == rows
    with p.c.cursor() as q:
        assert old.scalar(q, 'SELECT count(*) FROM lab_arena_runs WHERE assignment_id=%s AND attempt>%s',
                          (lease['assignment_id'], lease['attempt'])) == 0
    p.c.commit()


@pytest.mark.parametrize('role', ['lab_arena_service', 'anon', 'authenticated', 'service_role'])
def test_provider_budget_grant_stays_owner_only(installed, role):
    p = installed
    diagnostic(p)
    receipt(p)
    upgrade(p)
    finite.test_owner_only(p, role)


@pytest.mark.parametrize('fault', ['stale_hash', 'runtime', 'approval', 'checkpoints', 'signal_checkpoints',
                                  'later', 'live', 'calls_pending', 'expired', 'published', 'generation', 'frozen_source'])
def test_provider_parent_preserves_existing_authority_controls(installed, fault):
    p = installed
    p.original_parent = dict(p.parent)
    diagnostic(p)
    receipt(p)
    upgrade(p)
    if fault == 'runtime':
        rows = base.snapshot(p)
        with pytest.raises(p.pg.Error):
            finite.grant(p, runtime=prior.RUNTIME)
        p.c.rollback()
        assert base.snapshot(p) == rows
    else:
        finite.test_refusals_preserve_every_row(p, fault)


@pytest.mark.parametrize('fault', ['money', 'quota'])
def test_provider_parent_cumulative_money_and_quota_stay_closed(installed, fault):
    p = installed
    diagnostic(p)
    with p.c.cursor() as q:
        cfg = old.scalar(q, 'SELECT configuration_doc FROM lab_arena_rounds WHERE round_id=%s', (old.ROUND,))
        count = 1 if fault == 'money' else cfg['scoring_call_quotas']['openrouter'] * cfg['stage_2_icp_count'] * cfg['max_attempts_per_assignment']
        q.execute('SET LOCAL session_replication_role=replica')
        q.execute("""INSERT INTO lab_arena_ledger(entry_kind,run_id,round_id,submission_id,miner_hotkey,
          stage,call_identity,provider,operation_id,funding_source,amount_microusd,entry_doc,terminal_response)
          SELECT kind,%s,%s,%s,%s,2,'sha256:'||encode(extensions.digest('quota685-'||n::text,'sha256'),'hex'),
            'openrouter','openrouter.chat',%s,CASE WHEN kind='settlement' THEN %s ELSE 0 END,'{}',
            CASE WHEN kind='settlement' THEN '{"status":200,"call_succeeded":true}'::jsonb ELSE NULL END
          FROM generate_series(1,%s)n CROSS JOIN(VALUES('reservation',1),('settlement',2))k(kind,ordering)
          ORDER BY n,ordering""", (p.parent['run_id'], old.ROUND, p.parent['submission_id'], p.parent['miner_hotkey'],
            'host' if p.organizer else 'miner_key', cfg['scoring_cap_microusd'] if fault == 'money' else 0, count))
        q.execute('SET LOCAL session_replication_role=origin')
    p.c.commit()
    receipt(p, status=503)
    upgrade(p)
    ledger_before = base.snapshot(p)['lab_arena_ledger']
    child = finite.grant(p)['child_identity']['run_id']
    lease = prior.claim_child(p, child)
    request_hash = 'sha256:' + '6' * 64
    call = base.contracts.provider_call_identity(assignment_id=lease['assignment_id'], attempt=lease['attempt'],
        icp_position=lease['icp_position'], action_sequence=1, operation_id='openrouter.chat', request_hash=request_hash)
    result = old.rpc(p, 'lab_arena_reserve_call(%s,%s,%s,%s,%s,%s,%s,%s::jsonb,%s)', (
        child, old.TOKEN, call, 'openrouter.chat', 'openrouter', 'host' if p.organizer else 'miner_key',
        500, json.dumps(dict(request_hash=request_hash, base_call_identity=call, action_sequence=1,
                            provider_attempt=1, max_output_tokens=8)), 1500))
    assert result['status'] == 'refused'
    assert result['reason'] == ('money_cap' if fault == 'money' else 'stage_quota')
    assert [r for r in base.snapshot(p)['lab_arena_ledger'] if r['call_identity'] != call] == ledger_before


@pytest.mark.parametrize('kind', ['scorer', 'provider'])
def test_new_diagnostic_does_not_expand_legacy_grants(installed, kind):
    p = installed
    diagnostic(p)
    receipt(p)
    upgrade(p)
    rows = base.snapshot(p)
    with pytest.raises(p.pg.Error):
        if kind == 'provider':
            prior.provider.grant(p, request='no-provider-diagnostic-expansion')
        else:
            prior.grant(p, request_id='no-scorer-diagnostic-expansion')
    p.c.rollback()
    assert base.snapshot(p) == rows


@pytest.mark.parametrize('drift', ['source', 'acl', 'owner', 'security', 'search_path', 'proof_source', 'proof_acl'])
def test_migration_refuses_unknown_source_and_security_atomically(installed, drift):
    p = installed
    sig = PROOF if drift.startswith('proof_') else COMMON
    with p.c.cursor() as q:
        definition = old.scalar(q, 'SELECT pg_get_functiondef(%s::regprocedure)', (sig,))
        if drift in ('source', 'proof_source'):
            q.execute(definition.replace('AS $function$', 'AS $function$\n-- unreviewed drift\n', 1))
        elif drift in ('acl', 'proof_acl'):
            q.execute('GRANT EXECUTE ON FUNCTION ' + sig + ' TO anon')
        elif drift == 'owner':
            q.execute('ALTER FUNCTION ' + sig + ' OWNER TO anon')
        elif drift == 'security':
            q.execute('ALTER FUNCTION ' + sig + ' SECURITY INVOKER')
        else:
            q.execute('ALTER FUNCTION ' + sig + ' SET search_path=public')
        # Keep drift in this transaction. Migration error must roll it all back.
        with pytest.raises(p.pg.Error, match='differs'):
            q.execute(SQL.read_text())
    p.c.rollback()
    with p.c.cursor() as q:
        assert old.scalar(q, 'SELECT pg_get_functiondef(%s::regprocedure)', (sig,)) == definition
    p.c.commit()

@pytest.mark.parametrize('ancestry', ['ordinary', 'provider'])
@pytest.mark.parametrize('outcome', ['accepted', 'judge_error', 'expired'])
def test_baseline_stage1_full_native_lifecycle_preserves_host_payer_and_checkpoints(installed, outcome, ancestry):
    p = installed
    if p.parent_category != 'ordinary' or p.organizer:
        pytest.skip('Baseline fixture uses the ordinary, non-exception ancestry once.')
    baseline = 'baseline-' + old.ROUND.removeprefix('arena-')
    assignment = old.ROUND + ':' + baseline + ':1:0'
    source_id = assignment + ':1'
    score_assignment = assignment + ':score'
    with p.c.cursor() as q:
        q.execute('SET LOCAL session_replication_role=replica')
        cfg = old.scalar(q, 'SELECT configuration_doc FROM lab_arena_rounds WHERE round_id=%s', (old.ROUND,))
        submission = old.scalar(q, 'SELECT to_jsonb(s) FROM lab_arena_submissions s WHERE submission_id=%s', (p.parent['submission_id'],))
        submission.update(submission_id=baseline, miner_hotkey=cfg['baseline_hotkey'], is_king=True)
        old.insert(q, 'lab_arena_submissions', submission)
        source = old.scalar(q, 'SELECT to_jsonb(r) FROM lab_arena_runs r WHERE run_id=%s', (p.parent['scored_run_id'],))
        source.update(run_id=source_id, assignment_id=assignment, submission_id=baseline,
                      miner_hotkey=cfg['baseline_hotkey'], stage=1)
        old.insert(q, 'lab_arena_runs', source)
        q.execute('ALTER TABLE lab_arena_runs DISABLE TRIGGER lab_arena_runs_reviewer_runtime')
        scores = old.scalar(q, 'SELECT jsonb_agg(to_jsonb(r) ORDER BY attempt) FROM lab_arena_runs r WHERE assignment_id=%s', (p.parent['assignment_id'],))
        for row in scores:
            row.update(run_id=score_assignment + ':' + str(row['attempt']), assignment_id=score_assignment,
                       submission_id=baseline, miner_hotkey=cfg['baseline_hotkey'], stage=1, scored_run_id=source_id)
            old.insert(q, 'lab_arena_runs', row)
        q.execute('ALTER TABLE lab_arena_runs ENABLE ALWAYS TRIGGER lab_arena_runs_reviewer_runtime')
        plan = {'work_items': [{'submission_id': baseline, 'icp_position': 0,
                               'scored_run_id': source_id, 'output_ref': source['output_ref']}]}
        q.execute("UPDATE lab_arena_rounds SET status='stage1_scoring',stage1_scoring_plan_doc=%s::jsonb,participants=participants||%s::jsonb WHERE round_id=%s", (
            json.dumps(plan), json.dumps([{'submission_id': baseline, 'miner_hotkey': cfg['baseline_hotkey'], 'is_king': True}]), old.ROUND))
        q.execute('SET LOCAL session_replication_role=origin')
    p.c.commit()
    prior.saved_parent(p, score_assignment + ':' + str(p.parent['attempt']))
    p.funding_source = 'host'
    if ancestry == 'provider':
        # Author the baseline's finite provider parent through the unchanged
        # native671 path, including its protected journal and runtime pin.
        diagnostic(p, stage='scorer', error='judge_error', reason='provider_error')
        p.proof = receipt(p, status=429)
        provider_child = prior.provider.grant(p, request='baseline-finite-provider-parent')['child_identity']['run_id']
        prior.claim_child(p, provider_child)
        prior.complete_failed(p, provider_child)
        prior.saved_parent(p, provider_child)
        with p.c.cursor() as q:
            assert old.scalar(q, 'SELECT public.lab_arena__operator_scorer_bug_child_v1(r) FROM lab_arena_runs r WHERE run_id=%s', (provider_child,)) is True
            assert old.scalar(q, "SELECT reviewer_runtime_doc->>'runtime_image_digest' FROM lab_arena_runs WHERE run_id=%s", (provider_child,)) == p.exception_runtime
        p.c.commit()
    diagnostic(p)
    receipt(p, status=429)
    upgrade(p)
    rows = base.snapshot(p)
    preview = finite.grant(p, dry=True)
    assert base.snapshot(p) == rows
    result = finite.grant(p)
    assert result['child_identity'] == preview['child_identity']
    assert result['checkpoints'] == p.checkpoints
    assert result['signal_checkpoints'] == p.signal_checkpoints
    child = result['child_identity']['run_id']
    assert old.funding(p, child)['funding_source'] == 'host'
    assert old.funding(p, child, 'deepline')['funding_source'] == 'host'
    lease = prior.claim_child(p, child)
    assert lease['stage'] == 1 and lease['submission_id'] == baseline
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
    assert finite.grant(p)['idempotent'] is True
    prior.saved_parent(p, child)
    rows = base.snapshot(p)
    with pytest.raises(p.pg.Error):
        finite.grant(p, request_id='no-second-baseline-budget')
    p.c.rollback()
    assert base.snapshot(p) == rows
    with p.c.cursor() as q:
        assert old.scalar(q, 'SELECT count(*) FROM lab_arena_runs WHERE assignment_id=%s AND attempt>%s',
                          (lease['assignment_id'], lease['attempt'])) == 0
    p.c.commit()

@pytest.mark.parametrize('shape', ['scorer', 'sandbox'])
def test_existing_scorer_and_timeout_parent_shapes_still_claim(installed, shape):
    p = installed
    diagnostic(p, stage='scorer' if shape == 'scorer' else 'sandbox',
               error='judge_error' if shape == 'scorer' else 'judge_timeout',
               reason='malformed_response' if shape == 'scorer' else 'sandbox_judge_timeout', unresolved=0)
    upgrade(p)
    child = finite.grant(p)['child_identity']['run_id']
    prior.claim_child(p, child)
    assert old.funding(p, child)['funding_source'] == ('host' if p.organizer else 'miner_key')
