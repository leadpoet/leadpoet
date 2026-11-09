"""Proved infrastructure losses use the native queue, not new retry authority."""
import base64
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import uuid

import pytest

from lab_arena.store import hash_lease_token
from tests.lab_arena import zero_call_judge_attempt597_postgres_test as old

ROOT = Path(__file__).parents[2]
SQL = ROOT / 'scripts/680-lab-arena-proved-infrastructure-judge-attempt-credit.sql'
PROOF = 'public.lab_arena__retained_provider_transient_v1(public.lab_arena_ledger)'
scenario = old.scenario


@pytest.fixture(autouse=True)
def reset(scenario):
    old.reset_functions.__wrapped__(scenario)
    db = scenario[0][0]
    with db[0].connect(**db[1]) as conn, conn.cursor() as cur:
        cur.execute('DROP FUNCTION IF EXISTS ' + PROOF)


def prepare(scenario, *, attempt=1):
    context, rid, pending = old.prepare(scenario, attempt)
    old.upgrade(context)
    # Install the exact existing673 helper, without its unrelated finite grant.
    source = (ROOT / 'scripts/673-lab-arena-retained-transient-provider-proof.sql').read_text()
    helper = source.split('CREATE OR REPLACE FUNCTION ', 1)[1].split('DO $patch673$', 1)[0]
    db = context[0]
    with db[0].connect(**db[1]) as conn, conn.cursor() as cur:
        cur.execute('GRANT CREATE ON SCHEMA public TO lab_arena_owner')
        cur.execute('CREATE OR REPLACE FUNCTION ' + helper)
        cur.execute('REVOKE CREATE ON SCHEMA public FROM lab_arena_owner')
    return context, rid, pending


def upgrade(context):
    db = context[0]
    before = old.prior.snapshots(context)
    with db[0].connect(**db[1]) as conn, conn.cursor() as cur:
        metadata = old.prior.scalar(cur, "to_jsonb(p)-'prosrc' FROM pg_proc p WHERE oid=%s::regprocedure", (old.HELPER,))
        cur.execute(SQL.read_text())
        cur.execute(SQL.read_text())
        assert old.prior.scalar(cur, "to_jsonb(p)-'prosrc' FROM pg_proc p WHERE oid=%s::regprocedure", (old.HELPER,)) == metadata
    assert old.prior.snapshots(context) == before


def ledger(context, lease, *, success=False, status=502, nonce=None, corrupt=False, amount=1000):
    """Seed private provider evidence only in the disposable fixture database."""
    cid = 'sha256:' + hashlib.sha256((nonce or uuid.uuid4().hex).encode()).hexdigest()
    request = 'sha256:' + '4' * 64
    raw = json.dumps({'error': {'code': status}}).encode()
    receipt = dict(status=status, call_succeeded=success, call_identity=cid,
                   operation_id='openrouter.chat', request_hash=request,
                   body_sha256='sha256:' + hashlib.sha256(raw).hexdigest(),
                   body_b64=base64.b64encode(raw).decode())
    if corrupt:
        receipt['body_sha256'] = 'sha256:' + '0' * 64
    call = dict(status=status, call_succeeded=success, upstream_response_receipt=receipt)
    db = context[0]
    funding = context[1].service.store.provider_funding(lease['run_id'], 'openrouter')['funding_source']
    with db[0].connect(**db[1]) as conn, conn.cursor() as cur:
        cur.execute('SET LOCAL session_replication_role=replica')
        for kind in ('reservation', 'dispatch', 'settlement'):
            cur.execute('''INSERT INTO public.lab_arena_ledger
                (entry_kind,run_id,round_id,submission_id,miner_hotkey,stage,call_identity,
                 provider,operation_id,funding_source,amount_microusd,entry_doc,terminal_response)
                SELECT %s,run_id,round_id,submission_id,miner_hotkey,stage,%s,
                 'openrouter','openrouter.chat',%s,%s,%s::jsonb,%s::jsonb
                FROM public.lab_arena_runs WHERE run_id=%s''',
                (kind, cid, funding, amount if kind == 'settlement' else 0,
                 json.dumps({'request_hash': request}) if kind == 'reservation' else '{}',
                 json.dumps(call) if kind == 'settlement' else None, lease['run_id']))
    return cid


def fail_provider(context, lease, *, reason='provider_error', stage='provider_call', unresolved=1):
    return context[1].service.store.complete_attempt(
        run_id=lease['run_id'], lease_token_hash=hash_lease_token(lease['lease_token']),
        result={'terminal_status': 'judge_error', 'failure_diagnostic': {
            'stage': stage, 'error_class': 'provider_unavailable' if stage == 'provider_call' else 'judge_error',
            'reason': reason}, 'resource_summary': {'provider_unresolved_failure_count': unresolved}},
        terminal_cause='judge_error', output_ref=None)


def paid_loss(context, pending):
    lease = old.claim(context, pending=pending)
    assert lease['status'] == 'leased'
    ledger(context, lease, success=True, status=200)
    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    events = [dict(event_id=str(uuid.uuid4()), kind='runtime.started', occurred_at=now,
                   content={'status': 'starting', 'runtime': 'runsc', 'lease_generation': lease['lease_generation']}),
              dict(event_id=str(uuid.uuid4()), kind='runtime.error', occurred_at=now,
                   content={'status': 'abandoned', 'failure_stage': 'runtime', 'error_class': 'RuntimeHostError',
                            'runtime_host_reason': 'sandbox_launcher_signaled', 'launch_exit_code': -15,
                            'launch_timed_out': False})]
    assert context[1].service.store.append_trajectory_events(
        lease['run_id'], hash_lease_token(lease['lease_token']), events)['status'] == 'accepted'
    expiry = context[1].service.store.expire_leases(context[1].round_id)
    run = context[1].service.store.get_run(lease['run_id'])
    assert run['terminal_doc']['recovery_reason'] == 'authenticated_settled_score_runtime_host_error'
    return lease, expiry


def test_paid_host_fifth_attempt_reproduces_then_native_expiry_recovers_once(scenario):
    context, rid, pending = prepare(scenario, attempt=5)
    accepted = [r for r in context[1].service.store.list_runs(rid) if r['status'] == 'accepted']
    lease, expiry = paid_loss(context, pending)
    assert expiry['retried'] == 0 and old.count(context, lease['run_id']) == 5
    before = old.prior.snapshots(context)
    upgrade(context)
    assert old.count(context, lease['run_id']) == 4
    with ThreadPoolExecutor(max_workers=2) as pool:
        receipts = list(pool.map(lambda _: context[1].service.store.expire_leases(rid), range(2)))
    assert sum(r['retried'] for r in receipts) == 1
    assert old.prior.run_ids(context, pending).count(pending['assignment_id'] + ':6') == 1
    assert context[1].service.store.get_run(lease['run_id']) in before['lab_arena_runs']
    assert [r for r in context[1].service.store.list_runs(rid) if r['status'] == 'accepted'] == accepted
    retry = old.claim(context, pending=pending)
    assert retry['attempt'] == 6
    stale = context[1].service.store.complete_attempt(
        run_id=lease['run_id'], lease_token_hash=hash_lease_token(lease['lease_token']),
        result={'terminal_status': 'accepted'}, terminal_cause='accepted', output_ref='forged-late-output')
    assert stale['status'] == 'failed'
    assert context[1].service.store.get_run(lease['run_id']) in before['lab_arena_runs']


def test_final_proved_provider_failure_reproduces_then_existing_completion_continues(scenario):
    context, rid, pending = prepare(scenario, attempt=5)
    lease = old.claim(context, pending=pending)
    ledger(context, lease)
    fail_provider(context, lease)
    assert old.count(context, lease['run_id']) == 5
    upgrade(context)
    assert old.count(context, lease['run_id']) == 4
    assert context[1].service.store.expire_leases(rid)['retried'] == 1
    retry = old.claim(context, pending=pending)
    assert retry['attempt'] == 6
    ledger(context, retry)
    response = fail_provider(context, retry)
    assert 'confirmation_attempt' in response
    assert old.count(context, retry['run_id']) == 4


@pytest.mark.parametrize('fault', ['malformed', 'scorer_provider', 'healed', 'no_receipt', 'forged_receipt',
                                  'no_unresolved', 'success', 'auth', 'unsupported', 'natural_expiry'])
def test_unproved_healed_or_judge_failures_keep_five_attempt_bound(scenario, fault):
    context, rid, pending = prepare(scenario, attempt=5)
    upgrade(context)
    lease = old.claim(context, pending=pending)
    if fault != 'no_receipt':
        ledger(context, lease, corrupt=fault == 'forged_receipt', success=fault == 'success',
               status=401 if fault == 'auth' else 501 if fault == 'unsupported' else 200 if fault == 'success' else 502)
    if fault == 'healed':
        ledger(context, lease, success=True, status=200)
    if fault == 'natural_expiry':
        db = context[0]
        with db[0].connect(**db[1]) as conn, conn.cursor() as cur:
            cur.execute("UPDATE public.lab_arena_runs SET lease_expires_at=clock_timestamp()-interval '1 second' WHERE run_id=%s", (lease['run_id'],))
        context[1].service.store.expire_leases(rid)
    else:
        fail_provider(context, lease, reason='malformed_response' if fault == 'malformed' else 'provider_error',
                      stage='scorer' if fault in ('malformed', 'scorer_provider') else 'provider_call',
                      unresolved=0 if fault == 'no_unresolved' else 1)
    assert old.count(context, lease['run_id']) == 5
    assert context[1].service.store.expire_leases(rid)['retried'] == 0
    assert old.claim(context, pending=pending) == {'status': 'no_pending'}


@pytest.mark.parametrize('drift', ['counter_body', 'counter_acl', 'proof_body', 'proof_acl', 'clear_body', 'clear_acl'])
def test_migration_rejects_schema_or_security_drift_atomically(scenario, drift):
    context, _, _ = prepare(scenario)
    db = context[0]
    with db[0].connect(**db[1]) as conn, conn.cursor() as cur:
        target = PROOF if drift.startswith('proof') else 'public.lab_arena__judge_calls_clear_v1(text)' if drift.startswith('clear') else old.HELPER
        if drift.endswith('acl'):
            cur.execute('GRANT EXECUTE ON FUNCTION ' + target + ' TO authenticated')
        else:
            definition = old.prior.scalar(cur, 'pg_get_functiondef(%s::regprocedure)', (target,))
            cur.execute(definition.replace('AS $function$', 'AS $function$\n-- unexpected drift\n', 1))
        conn.commit()
        with pytest.raises(db[0].Error, match='differs'):
            cur.execute(SQL.read_text())
        conn.rollback()
        cur.execute('SELECT has_function_privilege(%s,%s,%s)', ('lab_arena_service', old.HELPER, 'EXECUTE'))
        assert cur.fetchone()[0] is False


@pytest.mark.parametrize('bound', ['cost', 'scoring_deadline', 'publication_deadline'])
def test_repeated_distinct_transport_losses_preserve_cost_and_time_admission(scenario, bound):
    context, rid, pending = prepare(scenario, attempt=5)
    upgrade(context)
    for attempt in range(5, 9):
        lease = old.claim(context, pending=pending)
        assert lease['attempt'] == attempt
        ledger(context, lease)
        response = fail_provider(context, lease)
        assert 'confirmation_attempt' in response
        assert old.count(context, lease['run_id']) == 4
        assert context[1].service.store.expire_leases(rid)['retried'] == 0
        # One physical failure is credited once, even after reconciliation sweeps.
        assert old.count(context, lease['run_id']) == 4
    if bound == 'cost':
        retry = old.claim(context, pending=pending)
        db = context[0]
        with db[0].connect(**db[1]) as conn, conn.cursor() as cur:
            cap = old.prior.scalar(cur, "(configuration_doc->>'scoring_cap_microusd')::bigint FROM public.lab_arena_rounds WHERE round_id=%s", (rid,))
        ledger(context, retry, success=True, status=200, amount=cap)
        refused = context[1].service.store.reserve_call(
            run_id=retry['run_id'], lease_token_hash=hash_lease_token(retry['lease_token']),
            call_identity='sha256:' + uuid.uuid4().hex * 2, operation_id='openrouter.chat',
            provider='openrouter', funding_source=context[1].service.store.provider_funding(retry['run_id'], 'openrouter')['funding_source'], amount_microusd=1,
            call_doc={'request_hash': 'sha256:' + 'f' * 64})
        assert (refused['status'], refused['reason']) == ('refused', 'money_cap')
    else:
        db = context[0]
        with db[0].connect(**db[1]) as conn, conn.cursor() as cur:
            cur.execute('SET LOCAL session_replication_role=replica')
            key = 'stage_1_scoring_close' if bound == 'scoring_deadline' else 'publication_deadline'
            cur.execute("UPDATE public.lab_arena_rounds SET configuration_doc=jsonb_set(configuration_doc,%s::text[],to_jsonb((clock_timestamp()-interval '1 second')::text)) WHERE round_id=%s", (['schedule', key], rid))
        context[1].service._invalidate_hot_round()
        receipt = old.claim(context, pending=pending)
        if bound == 'scoring_deadline':
            assert receipt['status'] == 'leased'  # existing continuation policy
        else:
            assert receipt == {'status': 'no_pending'}


@pytest.mark.parametrize('fault', ['receipt', 'claim_generation', 'original_expiry', 'error_identity', 'error_stage', 'policy'])
def test_paid_host_credit_requires_unchanged_authenticated_proof(scenario, fault):
    context, rid, pending = prepare(scenario, attempt=5)
    lease, _ = paid_loss(context, pending)
    db = context[0]
    with db[0].connect(**db[1]) as conn, conn.cursor() as cur:
        cur.execute('ALTER TABLE public.lab_arena_runs DISABLE TRIGGER lab_arena_runs_terminal')
        cur.execute('SET LOCAL session_replication_role=replica')
        if fault == 'receipt':
            cur.execute("UPDATE public.lab_arena_runs SET terminal_doc=terminal_doc-'recovery_reason' WHERE run_id=%s", (lease['run_id'],))
        elif fault == 'claim_generation':
            cur.execute("UPDATE public.lab_arena_runs SET claim_response=jsonb_set(claim_response,'{lease_generation}','0') WHERE run_id=%s", (lease['run_id'],))
        elif fault == 'original_expiry':
            cur.execute("UPDATE public.lab_arena_runs SET terminal_doc=jsonb_set(terminal_doc,'{original_lease_expires_at}',to_jsonb('2000-01-01T00:00:00Z'::text)) WHERE run_id=%s", (lease['run_id'],))
        elif fault == 'error_identity':
            cur.execute("UPDATE public.lab_arena_trajectory_events SET assignment_id='wrong-assignment' WHERE run_id=%s AND event_kind='runtime.error'", (lease['run_id'],))
        elif fault == 'error_stage':
            cur.execute("UPDATE public.lab_arena_trajectory_events SET content=jsonb_set(content,'{failure_stage}',to_jsonb('setup'::text)) WHERE run_id=%s AND event_kind='runtime.error'", (lease['run_id'],))
        else:
            cur.execute("UPDATE public.lab_arena_rounds SET configuration_doc=configuration_doc-'infrastructure_retry_cost_policy' WHERE round_id=%s", (rid,))
        cur.execute('SET LOCAL session_replication_role=origin')
        cur.execute('ALTER TABLE public.lab_arena_runs ENABLE TRIGGER lab_arena_runs_terminal')
    upgrade(context)
    assert old.count(context, lease['run_id']) == 5
    assert context[1].service.store.expire_leases(rid)['retried'] == 0


def natural_loss(context, pending, *, liability=None, shortened=False, output=False, paid=True):
    lease = old.claim(context, pending=pending)
    assert lease['status'] == 'leased'
    if paid:
        ledger(context, lease, success=True, status=200)
    db = context[0]
    with db[0].connect(**db[1]) as conn, conn.cursor() as cur:
        cur.execute('SET LOCAL session_replication_role=replica')
        # Shift both ends of the disposable claim window. The real expiry RPC
        # authors expired_at; no host event or recovery receipt is supplied.
        cur.execute("UPDATE public.lab_arena_runs SET lease_expires_at=clock_timestamp()-interval '1 second',claim_response=jsonb_set(claim_response,'{lease_expires_at}',to_jsonb(clock_timestamp()-interval '2 seconds')) WHERE run_id=%s", (lease['run_id'],))
        if shortened:
            cur.execute("UPDATE public.lab_arena_runs SET claim_response=jsonb_set(claim_response,'{lease_expires_at}',to_jsonb(clock_timestamp()+interval '1 hour')) WHERE run_id=%s", (lease['run_id'],))
        if output:
            cur.execute("UPDATE public.lab_arena_runs SET output_ref='saved-score-output' WHERE run_id=%s", (lease['run_id'],))
        if liability:
            cid = 'sha256:' + uuid.uuid4().hex * 2
            stages = ('reservation',) if liability == 'reservation' else ('reservation', 'dispatch') if liability == 'dispatch' else ('reservation', 'dispatch', 'uncertain')
            for kind in stages:
                cur.execute("INSERT INTO public.lab_arena_ledger(entry_kind,run_id,round_id,submission_id,miner_hotkey,stage,call_identity,provider,operation_id,funding_source,amount_microusd,entry_doc) SELECT %s,run_id,round_id,submission_id,miner_hotkey,stage,%s,'openrouter','openrouter.chat','host',1000,%s::jsonb FROM public.lab_arena_runs WHERE run_id=%s", (
                    kind, cid, json.dumps({'call': {'call_succeeded': False}}) if kind == 'uncertain' else '{}', lease['run_id']))
    expiry = context[1].service.store.expire_leases(context[1].round_id)
    return lease, expiry


@pytest.mark.parametrize('paid', [True, False], ids=['paid-work', 'zero-calls'])
def test_natural_fifth_expiry_reproduces_then_resumes_without_host_receipt(scenario, paid):
    context, rid, pending = prepare(scenario, attempt=5)
    lease, expiry = natural_loss(context, pending, paid=paid)
    run = context[1].service.store.get_run(lease['run_id'])
    assert run['status'] == 'failed' and run['terminal_cause'] == 'lease_expired'
    assert set(run['terminal_doc']) == {'expired_at'}
    assert expiry['retried'] == 0 and old.count(context, lease['run_id']) == 5
    upgrade(context)
    assert old.count(context, lease['run_id']) == 4
    assert context[1].service.store.expire_leases(rid)['retried'] == 1
    assert old.claim(context, age=False) == {'status': 'no_pending'}  # physical backoff
    old.prior.age(context, pending['assignment_id'] + ':6')
    runner = context[1].runner(0)
    retry = runner.claim_one()
    assert retry['attempt'] == 6
    # An old worker cannot reserve new paid work or replace the failed receipt.
    refused = context[1].service.store.reserve_call(
        run_id=lease['run_id'], lease_token_hash=hash_lease_token(lease['lease_token']),
        call_identity='sha256:' + uuid.uuid4().hex * 2, operation_id='openrouter.chat',
        provider='openrouter', funding_source='host', amount_microusd=1, call_doc={})
    assert refused['status'] == 'stale'
    assert context[1].service.store.get_run(lease['run_id']) == run
    try:
        assert runner._slots.acquire(blocking=False)
        runner._run_lease(retry)
        assert context[1].service.store.get_run(retry['run_id'])['status'] == 'accepted'
    finally:
        runner.close()
    assert context[1].service.store.expire_leases(rid)['retried'] == 0
    assert old.claim(context, pending=pending) == {'status': 'no_pending'}
    context[1].advance_until('scored', runners=1)
    assert context[1].service.publish(rid)['status'] == 'ok'
    assert context[1].status() == 'published'


@pytest.mark.parametrize('fault', ['reservation', 'dispatch', 'uncertain', 'shortened', 'saved_output'])
def test_natural_expiry_credit_never_bypasses_unclosed_work_or_saved_output(scenario, fault):
    context, rid, pending = prepare(scenario, attempt=5)
    upgrade(context)
    lease, _ = natural_loss(context, pending, liability=fault if fault in ('reservation', 'dispatch', 'uncertain') else None,
                            shortened=fault == 'shortened', output=fault == 'saved_output')
    assert old.count(context, lease['run_id']) == 5
    assert context[1].service.store.expire_leases(rid)['retried'] == 0
    assert old.claim(context, pending=pending) == {'status': 'no_pending'}
