"""Late original inputs have bounded capture authority, never execution authority."""
import copy
from datetime import timedelta

from fastapi.testclient import TestClient
import pytest

from lab_arena import contracts, training_input
from lab_arena.api import TRAINING_INPUT_WORKER_LIMIT, create_app
from lab_arena.service import ServiceError
from lab_arena.store import hash_lease_token
from tests.lab_arena.training_input_test import CLAIM_HASH, NOW, TOKEN, service_for

ROUTE = '/arena/v1/runs/r1/training-input'
HEADERS = {'x-lab-arena-lease': TOKEN}


def terminal_service(tmp_path, *, kind='execute', cause='accepted', seconds=1, expiry_seconds=60):
    service, objects = service_for(tmp_path, kind=kind)
    row = service._store.get_run('r1')
    row.update(status='accepted' if cause == 'accepted' else 'failed', terminal_cause=cause,
        lease_expires_at=(NOW + timedelta(seconds=expiry_seconds)).isoformat().replace("+00:00", "Z"),
        claim_request_id='1' * 32, claim_response={'status': 'leased', 'request_id': '1' * 32,
            'run_id': 'r1', 'lease_generation': 1, 'kind': kind},
        result_doc={'schema_version': 'leadpoet.lab_arena.run_result.v1',
            'started_at': (NOW - timedelta(seconds=300)).isoformat().replace("+00:00", "Z"),
            'finished_at': (NOW - timedelta(seconds=seconds)).isoformat().replace("+00:00", "Z"), 'terminal_status': cause,
            'resource_summary': {'wall_seconds': 1.0, 'cpu_seconds': 0.0, 'max_rss_bytes': 0,
                'stdout_bytes': 0, 'stderr_bytes': 0, 'provider_call_count': 0}})
    row['runner_hotkey'] = 'validator'
    if cause == 'accepted':
        row['participation_accepted_at'] = NOW.isoformat()
    finish_event = {'run_id': 'r1', 'round_id': row['round_id'], 'runner_hotkey': row['runner_hotkey'],
        'run_kind': kind, 'event_kind': 'runtime.finished', 'created_at': NOW.isoformat(),
        'status': cause, 'started_at': row['result_doc']['started_at'],
        'finished_at': row['result_doc']['finished_at'], 'resource_summary': copy.deepcopy(row['result_doc']['resource_summary'])}
    service._store.get_input_finish_event = lambda _: finish_event
    round_state = {'round_id': row['round_id'], 'status': 'stage1'}
    service._store.get_round_capture_state = lambda _: round_state
    return service, objects, row, round_state


@pytest.mark.parametrize('kind,cause', [('execute', 'accepted'), ('execute', 'credential_error'),
    ('execute', 'model_error'), ('score', 'accepted'), ('score', 'judge_error')])
@pytest.mark.parametrize('seconds', [0, .2, 179.999])
def test_original_input_can_arrive_after_ordinary_terminal_without_other_authority(tmp_path, kind, cause, seconds):
    service, objects, row, _state = terminal_service(tmp_path, kind=kind, cause=cause, seconds=seconds)
    original = copy.deepcopy(row)
    document = {'original': 'exact runtime input'}
    client = TestClient(create_app(service))
    result = client.post(ROUTE, headers=HEADERS, json=document)
    assert result.status_code == 200, result.text
    assert client.post(ROUTE, headers=HEADERS, json=document).json() == result.json()
    assert client.post(ROUTE, headers=HEADERS, json={'different': True}).status_code == 409
    assert training_input.read_input(objects, 'r1', kind, 1, CLAIM_HASH)['input'] == document
    assert row == original
    with pytest.raises(ServiceError):
        service._training_capture_run('r1', TOKEN)


@pytest.mark.parametrize('anchor', ['finished', 'expiry', 'accepted'])
@pytest.mark.parametrize('seconds,allowed', [(179.999, True), (180, False), (181, False)])
def test_grace_ends_at_the_earliest_reliable_anchor(tmp_path, anchor, seconds, allowed):
    service, objects, row, _state = terminal_service(tmp_path, seconds=1)
    if anchor == 'finished': row['result_doc']['finished_at'] = (NOW - timedelta(seconds=seconds)).isoformat().replace("+00:00", "Z")
    elif anchor == 'expiry': row['lease_expires_at'] = (NOW - timedelta(seconds=seconds)).isoformat().replace("+00:00", "Z")
    else: row['participation_accepted_at'] = (NOW - timedelta(seconds=seconds)).isoformat().replace("+00:00", "Z")
    result = TestClient(create_app(service)).post(ROUTE, headers=HEADERS, json={'original': True})
    assert result.status_code == (200 if allowed else 409)
    assert bool(list((tmp_path / 'objects').rglob('*.gz'))) is allowed


@pytest.mark.parametrize('damage', ['wrong_token', 'generation', 'claim_id', 'claim_hash', 'claim_run', 'claim_kind',
    'claim_status', 'pending', 'superseded', 'recovered', 'cancelled', 'round_scope', 'missing_result', 'invalid_result',
    'future_finish', 'naive_finish', 'missing_finish', 'finish_before_start', 'future_acceptance', 'naive_expiry',
    'terminal_cause', 'changed_claim'])
def test_terminal_capture_rejects_reclaimed_or_unreliable_authority_before_storage(tmp_path, damage):
    service, objects, row, state = terminal_service(tmp_path)
    token = TOKEN
    if damage == 'wrong_token': token = 'b' * 64
    elif damage == 'generation': row['lease_generation'] = 2
    elif damage == 'claim_id': row['claim_request_id'] = '2' * 32
    elif damage == 'claim_hash': row['claim_request_hash'] = 'invalid'
    elif damage == 'claim_run': row['claim_response']['run_id'] = 'other'
    elif damage == 'claim_kind': row['claim_response']['kind'] = 'score'
    elif damage == 'claim_status': row['claim_response']['status'] = 'pending'
    elif damage == 'pending': row['status'] = 'pending'
    elif damage == 'superseded': row['terminal_doc'] = {'recovery': 'superseded'}
    elif damage == 'recovered': row['terminal_cause'] = 'stage_closed'
    elif damage == 'cancelled': state['status'] = 'cancelled'
    elif damage == 'round_scope':
        def reject(_): raise ServiceError('round_scope_mismatch', 409)
        service._require_round_ownership = reject
    elif damage == 'missing_result': row['result_doc'] = None
    elif damage == 'invalid_result': row['result_doc']['schema_version'] = 'cached-or-derived'
    elif damage == 'future_finish': row['result_doc']['finished_at'] = (NOW + timedelta(seconds=1)).isoformat().replace("+00:00", "Z")
    elif damage == 'naive_finish': row['result_doc']['finished_at'] = NOW.replace(tzinfo=None).isoformat().replace("+00:00", "Z")
    elif damage == 'missing_finish': row['result_doc'].pop('finished_at')
    elif damage == 'finish_before_start': row['result_doc']['finished_at'] = (NOW - timedelta(days=1)).isoformat().replace("+00:00", "Z")
    elif damage == 'future_acceptance': row['participation_accepted_at'] = (NOW + timedelta(seconds=1)).isoformat().replace("+00:00", "Z")
    elif damage == 'naive_expiry': row['lease_expires_at'] = NOW.replace(tzinfo=None).isoformat().replace("+00:00", "Z")
    elif damage == 'terminal_cause': row['terminal_cause'] = 'model_error'
    elif damage == 'changed_claim':
        row.update(lease_generation=2, claim_request_hash='sha256:' + 'd' * 64,
            lease_token_hash=hash_lease_token('b' * 64))
    before = copy.deepcopy(row)
    response = TestClient(create_app(service)).post(ROUTE, headers={'x-lab-arena-lease': token}, json={'original': True})
    assert response.status_code in {401, 409}
    assert row == before
    assert not list((tmp_path / 'objects').rglob('*.gz'))


def test_mutable_updated_at_cannot_extend_terminal_capture_grace(tmp_path):
    service, objects, row, _state = terminal_service(tmp_path, seconds=181)
    row['updated_at'] = NOW.isoformat().replace("+00:00", "Z")
    assert TestClient(create_app(service)).post(ROUTE, headers=HEADERS, json={}).status_code == 409
    assert not list((tmp_path / 'objects').rglob('*.gz'))


@pytest.mark.parametrize('field,value', [('lease_generation', 2), ('claim_request_hash', 'sha256:' + 'd' * 64),
    ('kind', 'score'), ('round_id', 'different-round'), ('lease_token_hash', hash_lease_token('b' * 64))])
def test_identity_rotation_during_snapshot_build_fails_before_put(tmp_path, field, value, monkeypatch):
    service, objects = service_for(tmp_path)
    row = service._store.get_run('r1')
    original = training_input.build_snapshot
    def build(*args, **kwargs):
        snapshot = original(*args, **kwargs)
        row[field] = value
        return snapshot
    monkeypatch.setattr(training_input, 'build_snapshot', build)
    response = TestClient(create_app(service)).post(ROUTE, headers=HEADERS, json={'original': True})
    assert response.status_code in {401, 409}
    assert not list((tmp_path / 'objects').rglob('*.gz'))


def test_late_input_keeps_privacy_and_size_limits(tmp_path):
    service, objects, _row, _state = terminal_service(tmp_path)
    client = TestClient(create_app(service))
    response = client.post(ROUTE, headers=HEADERS, json={'api_key': 'synthetic-private-value'})
    assert response.status_code == 200
    snapshot = training_input.read_input(objects, 'r1', 'execute', 1, CLAIM_HASH)
    assert snapshot['privacy'] == {'redacted': True, 'redactions': ['credential']}
    assert snapshot['input']['api_key'] == '[REDACTED]'
    assert client.post(ROUTE, headers=HEADERS, json={'too_large': 'x' * training_input.MAX_INPUT_BYTES}).status_code == 413


def test_cancellation_read_is_a_tiny_existing_table_projection():
    from lab_arena.store import ArenaStore
    calls = []
    class Transport:
        def select(self, table, **query):
            calls.append((table, query))
            return [{'round_id': 'round', 'status': 'cancelled'}]
    assert ArenaStore(Transport()).get_round_capture_state('round')['status'] == 'cancelled'
    assert calls == [('lab_arena_rounds', {'filters': {'round_id': 'round'}, 'limit': 1, 'columns': 'round_id,status'})]


def test_queued_capture_rechecks_grace_when_an_input_worker_becomes_available(tmp_path, monkeypatch):
    import asyncio
    import threading
    import httpx
    from tests.lab_arena.api_provider_capacity_test import wait_until
    service, objects, _row, _state = terminal_service(tmp_path, seconds=179)
    now = [NOW]
    service._clock = lambda: now[0]
    original = training_input.build_snapshot
    entered, release = [], threading.Event()
    def blocked(*args, **kwargs):
        entered.append(args[2])
        assert release.wait(2)
        return original(*args, **kwargs)
    monkeypatch.setattr(training_input, 'build_snapshot', blocked)
    async def scenario():
        app = create_app(service)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as client:
            active = [asyncio.create_task(client.post(ROUTE, headers=HEADERS, json={'active': i}))
                      for i in range(TRAINING_INPUT_WORKER_LIMIT)]
            await wait_until(lambda: len(entered) == TRAINING_INPUT_WORKER_LIMIT)
            queued = asyncio.create_task(client.post(ROUTE, headers=HEADERS, json={'queued': True}))
            try:
                await asyncio.sleep(.02)
                now[0] += timedelta(seconds=2)
            finally:
                release.set()
            replies = await asyncio.gather(*active, queued)
            assert [r.status_code for r in replies] == [409] * (TRAINING_INPUT_WORKER_LIMIT + 1)
    asyncio.run(scenario())
    assert all('queued' not in row for row in entered)
    assert not list((tmp_path / 'objects').rglob('*.gz'))


def test_failed_terminal_server_finish_clamps_a_later_worker_finish_claim(tmp_path):
    service, objects, row, _state = terminal_service(tmp_path, cause='credential_error', seconds=60, expiry_seconds=300)
    event = service._store.get_input_finish_event('r1')
    event['created_at'] = (NOW - timedelta(seconds=240)).isoformat()
    row['updated_at'] = (NOW - timedelta(seconds=240)).isoformat()
    response = TestClient(create_app(service)).post(ROUTE, headers=HEADERS, json={'original': True})
    assert response.status_code == 409
    assert not list((tmp_path / 'objects').rglob('*.gz'))


@pytest.mark.parametrize('damage', ['missing', 'wrong_run', 'wrong_round', 'wrong_validator', 'wrong_kind',
    'wrong_event', 'wrong_status', 'wrong_start', 'wrong_summary', 'wrong_summary_type', 'wrong_finish', 'future_server', 'naive_server', 'missing_server'])
def test_failed_capture_requires_matching_authenticated_server_finish_anchor(tmp_path, damage):
    service, objects, row, _state = terminal_service(tmp_path, cause='credential_error')
    event = service._store.get_input_finish_event('r1')
    if damage == 'missing': service._store.get_input_finish_event = lambda _: None
    elif damage == 'wrong_run': event['run_id'] = 'other'
    elif damage == 'wrong_round': event['round_id'] = 'other'
    elif damage == 'wrong_validator': event['runner_hotkey'] = 'other'
    elif damage == 'wrong_kind': event['run_kind'] = 'score'
    elif damage == 'wrong_event': event['event_kind'] = 'runtime.error'
    elif damage == 'wrong_status': event['status'] = 'accepted'
    elif damage == 'wrong_start': event['started_at'] = (NOW - timedelta(seconds=301)).isoformat()
    elif damage == 'wrong_summary': event['resource_summary']['provider_call_count'] = 1
    elif damage == 'wrong_summary_type': event['resource_summary']['provider_call_count'] = False
    elif damage == 'wrong_finish': event['finished_at'] = (NOW - timedelta(seconds=2)).isoformat()
    elif damage == 'future_server': event['created_at'] = (NOW + timedelta(seconds=1)).isoformat()
    elif damage == 'naive_server': event['created_at'] = NOW.replace(tzinfo=None).isoformat()
    elif damage == 'missing_server': event.pop('created_at')
    row['updated_at'] = NOW.isoformat()
    assert TestClient(create_app(service)).post(ROUTE, headers=HEADERS, json={'original': True}).status_code == 409
    assert not list((tmp_path / 'objects').rglob('*.gz'))


def test_missing_accepted_server_timestamp_keeps_capture_closed(tmp_path):
    service, objects, row, _state = terminal_service(tmp_path)
    row.pop('participation_accepted_at')
    assert TestClient(create_app(service)).post(ROUTE, headers=HEADERS, json={}).status_code == 409
    assert not list((tmp_path / 'objects').rglob('*.gz'))


def test_finish_anchor_changes_after_build_are_rechecked(tmp_path, monkeypatch):
    service, objects, _row, _state = terminal_service(tmp_path, cause='credential_error')
    event = service._store.get_input_finish_event('r1')
    original = training_input.build_snapshot
    def build(*args, **kwargs):
        result = original(*args, **kwargs)
        event['runner_hotkey'] = 'different-validator'
        return result
    monkeypatch.setattr(training_input, 'build_snapshot', build)
    assert TestClient(create_app(service)).post(ROUTE, headers=HEADERS, json={}).status_code == 409
    assert not list((tmp_path / 'objects').rglob('*.gz'))


def test_finish_projection_is_bounded_in_postgrest_and_direct_sql():
    import httpx
    from lab_arena.store import ArenaStore, PostgrestTransport, PsycopgTransport
    requests = []
    with httpx.Client(transport=httpx.MockTransport(lambda request: requests.append(request) or httpx.Response(200, json=[]))) as client:
        store = ArenaStore(PostgrestTransport('https://project.example', service_key='sb_secret_test', http_client=client))
        assert store.get_input_finish_event('r1') is None
    assert len(requests) == 1
    params = requests[0].url.params
    assert params['run_id'] == 'eq.r1' and params['event_kind'] == 'eq.runtime.finished'
    assert params['limit'] == '1' and params['order'] == 'created_at.asc'
    assert 'created_at' in params['select'] and 'finished_at:content->>finished_at' in params['select']
    assert ',content,' not in params['select']
    queries = []
    class Cursor:
        def __enter__(self): return self
        def __exit__(self, *_args): return None
        def execute(self, sql, values): queries.append((sql, values))
        def fetchall(self): return []
    direct = object.__new__(PsycopgTransport)
    direct._acquire = lambda: type('Connection', (), {'cursor': lambda self: Cursor()})()
    direct._release = lambda _connection: None
    assert ArenaStore(direct).get_input_finish_event('r1') is None
    assert len(queries) == 1
    sql, values = queries[0]
    assert 'ORDER BY created_at ASC LIMIT 1' in sql
    assert "content ->> 'finished_at' AS finished_at" in sql
    assert values == ['r1', 'runtime.finished']
