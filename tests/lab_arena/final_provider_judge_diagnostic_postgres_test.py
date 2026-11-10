"""Actual runner/Broker/PG completion retains SQL680's exact economic proof."""
from copy import deepcopy
import json
import os

import pytest

from lab_arena import broker, contracts, runtime, scoring, shim
from tests.lab_arena import proved_infrastructure_judge_attempt680_postgres_test as base

scenario = base.scenario
reset = base.reset


class FailedProvider:
    def __init__(self): self.calls = []
    def send(self, **kwargs):
        self.calls.append(kwargs)
        return broker.ProviderResponse(502, {'content-type': 'application/json'},
            b'{"error":{"code":502,"message":"Provider unavailable"}}')


class FailedJudge:
    def __init__(self, model, *, reason, healed):
        self.model, self.reason, self.healed = model, reason, healed
    def run_icp(self, spec, **_kwargs):
        data = json.loads((spec.input_dir / runtime.INPUT_FILE_NAME).read_text())
        previous = {key: os.environ.get(key) for key in (shim.WORKER_SOCKET_ENV, shim.TRUSTED_SCORER_ENV,
                                                        'ARENA_JUDGE_REASONING_CAPTURE_POLICY')}
        os.environ[shim.WORKER_SOCKET_ENV] = str(spec.socket_path)
        os.environ[shim.TRUSTED_SCORER_ENV] = '1'
        os.environ['ARENA_JUDGE_REASONING_CAPTURE_POLICY'] = shim.FULL_THINKING_POLICY
        try:
            for _ in range(2 if self.healed else 1):
                shim.dispatch('openrouter.chat', {'model': self.model,
                    'messages': [{'role': 'user', 'content': 'Verify retained evidence.'}], 'max_tokens': 128}, 5000)
        finally:
            for key, value in previous.items():
                if value is None: os.environ.pop(key, None)
                else: os.environ[key] = value
        output = scoring.build_scoring_failure(data['scored_run_id'], 'judge_error', reason=self.reason)
        return runtime.fake_result(exit_code=0, output_bytes=json.dumps(output).encode())


def miner_pending(context, baseline_pending):
    db, harness, original, saved = context
    # This historical fixture finalized empty miner output without a score job.
    # Construct only its missing synthetic score history from the native
    # baseline shape, bound to that miner's own accepted execution and payer.
    source = next(row for row in saved['lab_arena_runs'] if row['kind'] == 'execute'
        and row['submission_id'] != baseline_pending['submission_id']
        and row['status'] == 'accepted')
    original_score = deepcopy(baseline_pending)
    original_score.update({key: source[key] for key in ('submission_id', 'miner_hotkey', 'stage', 'icp_position')})
    original_score.update(assignment_id=source['assignment_id'] + ':score', scored_run_id=source['run_id'],
                          company_judgment_refs=[], judgment_cache_key=None, judgment_scope_doc=None)
    with db[0].connect(**db[1]) as conn, conn.cursor() as cur:
        cur.execute('SET LOCAL session_replication_role=replica')
        cur.execute("UPDATE public.lab_arena_rounds SET status='stage2_scoring',finalists=%s::jsonb,stage2_scoring_plan_doc=%s::jsonb WHERE round_id=%s",
            (json.dumps(original['finalists']), json.dumps(original['stage2_scoring_plan_doc']), original['round_id']))
        cur.execute("UPDATE public.lab_arena_runs SET status='failed',terminal_cause='judge_error',result_doc='{" +
                    '"terminal_status":"judge_error"' + "}'::jsonb WHERE run_id=%s", (baseline_pending['run_id'],))
        cur.execute('INSERT INTO public.lab_arena_runs SELECT (jsonb_populate_record(NULL::public.lab_arena_runs,%s::jsonb)).*', (json.dumps(source),))
        for attempt in range(1, 6):
            row = deepcopy(original_score)
            row.update(run_id=row['assignment_id'] + ':' + str(attempt), attempt=attempt,
                status='pending' if attempt == 5 else 'failed',
                terminal_cause=None if attempt == 5 else 'judge_error',
                result_doc=None if attempt == 5 else {'terminal_status': 'judge_error'},
                output_ref=None, terminal_doc=None, claim_request_id=None, claim_response=None,
                lease_token_hash=None, lease_expires_at=None)
            cur.execute('INSERT INTO public.lab_arena_runs SELECT (jsonb_populate_record(NULL::public.lab_arena_runs,%s::jsonb)).*', (json.dumps(row),))
        cur.execute("UPDATE public.lab_arena_runs SET created_at=clock_timestamp()-interval '1 hour' WHERE run_id=%s", (row['run_id'],))
    harness.service._invalidate_hot_round()
    return row


@pytest.mark.parametrize('actor', ['baseline', 'miner'])
@pytest.mark.parametrize('fault', ['provider', 'malformed', 'healed'])
def test_runner_native_accounting_completion_and_finite_successor(scenario, monkeypatch, actor, fault):
    context, rid, pending = base.prepare(scenario, attempt=5)
    if actor == 'miner': pending = miner_pending(context, pending)
    base.upgrade(context)
    db, harness, *_ = context
    transport = FailedProvider()
    if fault == 'healed':
        native_send = transport.send
        def send(**kwargs):
            if not transport.calls: return native_send(**kwargs)
            transport.calls.append(kwargs)
            return broker.ProviderResponse(200, {'content-type': 'application/json'},
                json.dumps({'model': json.loads(kwargs['body'])['model'], 'choices': [
                    {'finish_reason': 'stop', 'message': {'role': 'assistant', 'content': 'negative'}}],
                    'usage': {'cost': '0.0001'}}).encode())
        transport.send = send
    native_factory = harness.service._config.broker_factory
    def factory(*args):
        result = native_factory(*args)
        result._transport = transport
        return result
    monkeypatch.setattr(harness.service._config, 'broker_factory', factory)
    monkeypatch.setattr(harness.service, '_brokers', {})
    worker = harness.runner(0)
    leased = worker.claim_one()
    assert leased['run_id'] == pending['run_id'] and leased['attempt'] == 5
    # Exercise the current worker's existing trusted acknowledgment channel
    # while retaining this fixture's historical grading/cache policy unchanged.
    leased['judge_reasoning_capture_policy'] = shim.FULL_THINKING_POLICY
    model = next(iter(leased['scorer_policy']['judge_models'].values()))
    worker._config.sandbox_runtime = FailedJudge(model,
        reason='malformed_response' if fault == 'malformed' else 'provider_error', healed=fault == 'healed')
    before = {row['run_id']: row for row in harness.service.store.list_runs(rid)
              if row['status'] in ('accepted', 'failed')}
    try:
        assert worker._slots.acquire(blocking=False)
        worker._run_lease(leased)
    finally: worker.close()
    failed = harness.service.store.get_run(leased['run_id'])
    assert failed['status'] == 'failed' and failed['terminal_cause'] == 'judge_error'
    expected_stage = 'provider_call' if fault == 'provider' else 'scorer'
    assert failed['result_doc']['failure_diagnostic']['stage'] == expected_stage
    assert base.old.count(context, leased['run_id']) == (4 if fault == 'provider' else 5)
    with db[0].connect(**db[1]) as conn, conn.cursor() as cur:
        cur.execute("SELECT entry_kind,funding_source,amount_microusd,terminal_response,entry_doc FROM public.lab_arena_ledger WHERE run_id=%s ORDER BY entry_id", (leased['run_id'],))
        ledger = cur.fetchall()
    assert ledger and any(row[0] == 'dispatch' for row in ledger)
    assert any(row[0] == 'uncertain' for row in ledger)
    assert len(transport.calls) == (2 if fault == 'healed' else 1)
    assert {row[1] for row in ledger} == {'host' if actor == 'baseline' else 'miner_key'}
    for row in harness.service.store.list_runs(rid):
        if row['run_id'] in before: assert row == before[row['run_id']]
    successor = harness.service.store.get_run(pending['assignment_id'] + ':6')
    assert (successor is not None) is (fault == 'provider')
    if successor is not None:
        assert successor['status'] == 'pending'
        assert successor['scored_run_id'] == failed['scored_run_id']
    # Native expiry must not clear liabilities or create another successor.
    harness.service.store.expire_leases(rid)
    with db[0].connect(**db[1]) as conn, conn.cursor() as cur:
        cur.execute("SELECT entry_kind,funding_source,amount_microusd,terminal_response,entry_doc FROM public.lab_arena_ledger WHERE run_id=%s ORDER BY entry_id", (leased['run_id'],))
        assert cur.fetchall() == ledger
    assert harness.service.store.get_run(pending['assignment_id'] + ':7') is None
