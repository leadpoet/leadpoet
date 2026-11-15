"""The normal queue recovers a broker-proved lost reply without resetting spend."""
import json
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from lab_arena.store import hash_lease_token
from tests.lab_arena import proved_infrastructure_judge_attempt680_postgres_test as native

scenario = native.scenario
reset = native.reset
old = native.old
SQL = Path(__file__).parents[2] / 'scripts/689-lab-arena-interrupted-judge-response.sql'


def upgrade(context):
    # This fixture installs the real queue. Owner-only grant coverage is in
    # interrupted_judge_response689_postgres_test, using the complete migration.
    source = SQL.read_text().split('DO $patch689_1$', 1)[0] + 'COMMIT;'
    db = context[0]
    before = old.prior.snapshots(context)
    with db[0].connect(**db[1]) as c, c.cursor() as q:
        q.execute(source)
        q.execute(source)
    assert old.prior.snapshots(context) == before


def interrupted(context, lease, amount=1000):
    cid = 'sha256:' + uuid.uuid4().hex * 2
    call = dict(reason='missing_provider_cost', call_succeeded=False,
                transport_failure=True, transport_error_class='ReadTimeout',
                provider_status=0, observed_provider_status=200,
                openrouter_generation_id='gen-1794458785-fixture',
                credential_fingerprint='sha256:' + 'c' * 64)
    db = context[0]
    funding = context[1].service.store.provider_funding(lease['run_id'], 'openrouter')['funding_source']
    with db[0].connect(**db[1]) as c, c.cursor() as q:
        q.execute('SET LOCAL session_replication_role=replica')
        for kind in ('reservation', 'dispatch', 'uncertain'):
            doc = {'request_hash': 'sha256:' + '4' * 64} if kind == 'reservation' else {'reason': 'worker_reported', 'call': call} if kind == 'uncertain' else {}
            q.execute('''INSERT INTO public.lab_arena_ledger
                (entry_kind,run_id,round_id,submission_id,miner_hotkey,stage,call_identity,
                 provider,operation_id,funding_source,amount_microusd,entry_doc)
                SELECT %s,run_id,round_id,submission_id,miner_hotkey,stage,%s,
                 'openrouter','openrouter.chat',%s,%s,%s::jsonb
                FROM public.lab_arena_runs WHERE run_id=%s''',
                (kind, cid, funding, amount, json.dumps(doc), lease['run_id']))


def test_normal_queue_recovers_proved_failure_once_without_owner_grant(scenario):
    context, rid, pending = native.prepare(scenario, attempt=5)
    native.upgrade(context)
    lease = old.claim(context, pending=pending)
    interrupted(context, lease)
    native.fail_provider(context, lease)
    assert old.count(context, lease['run_id']) == 5
    assert context[1].service.store.expire_leases(rid)['retried'] == 0
    before = old.prior.snapshots(context)
    upgrade(context)
    assert old.count(context, lease['run_id']) == 4
    with ThreadPoolExecutor(max_workers=2) as pool:
        receipts = list(pool.map(lambda _: context[1].service.store.expire_leases(rid), range(2)))
    assert sum(r['retried'] for r in receipts) == 1
    retry = old.claim(context, pending=pending)
    assert retry['attempt'] == 6
    assert old.prior.snapshots(context)['lab_arena_ledger'] == before['lab_arena_ledger']
    # A genuine bad judgment still uses the remaining normal attempt.
    context[1].service.store.complete_attempt(
        run_id=retry['run_id'], lease_token_hash=hash_lease_token(retry['lease_token']),
        result={'terminal_status': 'judge_error', 'failure_diagnostic': {
            'stage': 'scorer', 'error_class': 'judge_error', 'reason': 'invalid_citation'}},
        terminal_cause='judge_error', output_ref=None)
    assert old.count(context, retry['run_id']) == 5
    assert context[1].service.store.expire_leases(rid)['retried'] == 0
    after = old.prior.snapshots(context)
    for row in before['lab_arena_runs']:
        assert row in after['lab_arena_runs']
    assert after['lab_arena_judgment_cache'] == before['lab_arena_judgment_cache']


@pytest.mark.parametrize('bound', ['cost', 'deadline'])
def test_transport_credit_keeps_existing_money_and_publication_limits(scenario, bound):
    context, rid, pending = native.prepare(scenario, attempt=5)
    native.upgrade(context)
    upgrade(context)
    db = context[0]
    with db[0].connect(**db[1]) as c, c.cursor() as q:
        cap = old.prior.scalar(q, "(configuration_doc->>'scoring_cap_microusd')::bigint FROM public.lab_arena_rounds WHERE round_id=%s", (rid,))
    lease = old.claim(context, pending=pending)
    interrupted(context, lease, amount=cap if bound == 'cost' else 1000)
    native.fail_provider(context, lease)
    assert old.count(context, lease['run_id']) == 4
    if bound == 'deadline':
        with db[0].connect(**db[1]) as c, c.cursor() as q:
            q.execute('SET LOCAL session_replication_role=replica')
            q.execute("UPDATE public.lab_arena_rounds SET configuration_doc=jsonb_set(configuration_doc,'{schedule,publication_deadline}',to_jsonb((clock_timestamp()-interval '1 second')::text)) WHERE round_id=%s", (rid,))
        context[1].service._invalidate_hot_round()
        assert old.claim(context, pending=pending) == {'status': 'no_pending'}
    else:
        retry = old.claim(context, pending=pending)
        # Frozen confirmed-cost admission does not charge an unknown cost as
        # settled. Its raw uncertainty stays intact; a confirmed settlement
        # still consumes the same submission-wide cap across physical attempts.
        before = old.prior.snapshots(context)['lab_arena_ledger']
        native.ledger(context, retry, success=True, status=200, amount=cap)
        response = context[1].service.store.reserve_call(
            run_id=retry['run_id'], lease_token_hash=hash_lease_token(retry['lease_token']),
            call_identity='sha256:' + uuid.uuid4().hex * 2, operation_id='openrouter.chat',
            provider='openrouter', funding_source=context[1].service.store.provider_funding(retry['run_id'], 'openrouter')['funding_source'],
            amount_microusd=1, call_doc={'request_hash': 'sha256:' + 'f' * 64})
        assert (response['status'], response.get('reason')) == ('refused', 'money_cap'), response
        after = old.prior.snapshots(context)['lab_arena_ledger']
        assert all(row in after for row in before)


@pytest.mark.parametrize('actor', ['baseline', 'miner'])
def test_real_http_broker_lost_body_recovers_without_replaying_uncertain_call(scenario, monkeypatch, actor):
    from fastapi.testclient import TestClient
    from lab_arena.api import create_app
    from lab_arena.runner import HttpArenaApiClient
    from tests.lab_arena.transport_failure_body_capture_test import transport_fixture
    from tests.lab_arena.final_provider_judge_diagnostic_postgres_test import miner_pending

    context, rid, pending = native.prepare(scenario, attempt=5)
    if actor == 'miner':
        pending = miner_pending(context, pending)
    native.upgrade(context)
    old.prior.age(context, pending['run_id'])
    # Use the normal signed lease; a direct SQL fixture token is not API auth.
    lease = old.prior.claim(context)
    assert lease['run_id'] == pending['run_id']
    service = context[1].service
    broker = service._broker_for(rid)
    transport, stream, sent = transport_fixture(b'', generation=True)
    monkeypatch.setattr(broker, '_transport', transport)
    model = next(iter(service.store.get_round(rid)['configuration_doc']['scorer_policy']['judge_models'].values()))
    frame = dict(operation_id='openrouter.chat', action_sequence=0, timeout_ms=5000,
                 parameters={'model': model, 'messages': [{'role': 'user', 'content': 'Check the saved company evidence.'}],
                             'max_tokens': 32})
    try:
        # Exercise HTTP handlers without starting the unrelated round driver.
        client = TestClient(create_app(service))
        api = HttpArenaApiClient('http://localhost', client=client)
        result = api.provider(lease['run_id'], lease['lease_token'], frame)
        assert result['status'] == 502, result
        assert result['call']['outcome'] == 'uncertain'
        assert api.provider(lease['run_id'], lease['lease_token'], frame)['status'] == 409
        assert len(sent) == 1 and stream.closed
        native.fail_provider(context, lease)
        assert old.count(context, lease['run_id']) == 5
        before = old.prior.snapshots(context)
        upgrade(context)
        assert old.count(context, lease['run_id']) == 4
        assert service.store.expire_leases(rid)['retried'] == 1
        assert old.claim(context, pending=pending)['attempt'] == 6
        assert old.prior.snapshots(context)['lab_arena_ledger'] == before['lab_arena_ledger']
        assert len(sent) == 1
    finally:
        client.close()
        transport.close()
