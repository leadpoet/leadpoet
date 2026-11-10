"""Terminal final capture must pass the public pre-buffer authorization path."""
import asyncio
import copy
from contextlib import contextmanager

from fastapi import FastAPI
from fastapi.testclient import TestClient
import httpx
import pytest
from starlette.requests import Request

from gateway.api import arena_proxy
from lab_arena import training_observations as observations
from lab_arena.api import create_app
from tests.lab_arena.training_input_grace_test import terminal_service
from tests.lab_arena.training_input_test import CLAIM_HASH, TOKEN
from tests.lab_arena.training_observations_test import observation, payload


@contextmanager
def public_gateway(service, monkeypatch):
    """Use actual gateway routes and actual sidecar handlers, without sockets."""
    monkeypatch.setenv('LAB_ARENA_MODE', 'live')
    monkeypatch.setenv('LAB_ARENA_TESTNET_ENABLED', 'true')
    sidecar = create_app(service)
    calls = []
    async def forward(method, path, *, query, body, headers, **routing):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=sidecar), base_url='http://sidecar') as client:
            result = await client.request(method, '/arena/' + path, params=query, content=body, headers=headers)
        calls.append((method, result.status_code, dict(headers)))
        return result
    monkeypatch.setattr(arena_proxy, '_request_sidecar', forward)
    app = FastAPI()
    app.include_router(arena_proxy.router)
    app.include_router(arena_proxy.testnet_router)
    with TestClient(app) as client:
        yield client, calls


TERMINALS = [('execute', 'accepted', False), ('execute', 'model_error', False),
             ('score', 'accepted', False), ('score', 'judge_error', False),
             ('score', 'accepted', True)]


@pytest.mark.parametrize('testnet', [False, True])
@pytest.mark.parametrize('kind,cause,all_hit', TERMINALS)
@pytest.mark.parametrize('seconds,wrong_token,allowed', [(179, False, True), (180, False, False),
                                                       (181, False, False), (179, True, False)])
def test_public_terminal_final_uses_same_strict_proof(tmp_path, monkeypatch, testnet, kind, cause, all_hit,
                                                    seconds, wrong_token, allowed):
    service, objects, row, _ = terminal_service(tmp_path, kind=kind, cause=cause, seconds=seconds)
    if all_hit:
        row['participation_accepted_at'] = None
    before = copy.deepcopy(row)
    token = 'b' * 64 if wrong_token else TOKEN
    path = ('/testnet' if testnet else '') + '/arena/v1/runs/r1/training-observations'
    document = payload(observation())
    with public_gateway(service, monkeypatch) as (client, calls):
        response = client.post(path, json=document, headers={'x-lab-arena-lease': token})
        if allowed:
            assert response.status_code == 200, response.text
            assert [(method, status) for method, status, _ in calls] == [('HEAD', 204), ('POST', 200)]
            assert calls[0][2]['x-lab-arena-observation-intent'] == 'final-write'
            assert 'x-lab-arena-observation-intent' not in calls[1][2]
            digest = response.json()['payload_sha256']
            assert client.head(path, headers={'x-lab-arena-lease': token, 'if-match': '"' + digest + '"'}).status_code == 204
            assert observations.read_snapshot(objects, 'r1', kind, 1, CLAIM_HASH)['payload'] == document
        else:
            assert response.status_code == (401 if wrong_token else 409), response.text
            assert [(method, status) for method, status, _ in calls] == [('HEAD', response.status_code)]
            assert not list((tmp_path / 'objects').rglob('*.gz'))
        # Clients cannot extend ordinary HEAD authorization by injecting the
        # internal preauthorization intent. The gateway does not forward it.
        assert client.head(path, headers={'x-lab-arena-lease': TOKEN,
            'x-lab-arena-observation-intent': 'final-write'}).status_code == 409
        assert 'x-lab-arena-observation-intent' not in calls[-1][2]
    assert row == before


@pytest.mark.parametrize('damage', ['boundary', 'wrong_token', 'wrong_claim', 'missing_finish_event',
                                  'wrong_finish_event', 'cancelled_round'])
def test_public_terminal_denial_does_not_read_private_body(tmp_path, monkeypatch, damage):
    service, _objects, row, round_state = terminal_service(tmp_path, kind='score', seconds=180 if damage == 'boundary' else 179)
    row['participation_accepted_at'] = None
    if damage == 'missing_finish_event':
        service._store.get_input_finish_event = lambda _: None
    elif damage == 'wrong_finish_event':
        event = service._store.get_input_finish_event('r1')
        event['run_id'] = 'another-run'
    elif damage == 'wrong_claim':
        row['claim_response']['lease_generation'] += 1
    elif damage == 'cancelled_round':
        round_state['status'] = 'cancelled'
    token = 'b' * 64 if damage == 'wrong_token' else TOKEN
    reads = []
    async def unread():
        reads.append(True)
        raise AssertionError('denied private body was read')
    request = Request({'type':'http', 'method':'POST', 'scheme':'http',
        'path':'/arena/v1/runs/r1/training-observations', 'query_string':b'',
        'server':('test',80), 'headers':[(b'x-lab-arena-lease', token.encode())]}, unread)
    with public_gateway(service, monkeypatch):
        response = asyncio.run(arena_proxy._proxy_request('v1/runs/r1/training-observations', request))
    assert response.status_code == (401 if damage == 'wrong_token' else 409)
    assert reads == []


@pytest.mark.parametrize('intent,digest,status', [(None, False, 409), ('wrong', False, 400),
                                               ('final-write', False, 204), ('final-write', True, 400)])
def test_internal_intent_does_not_change_plain_or_digest_head(tmp_path, intent, digest, status):
    service, _objects, _row, _ = terminal_service(tmp_path, seconds=179)
    headers = {'x-lab-arena-lease': TOKEN}
    if intent is not None:
        headers['x-lab-arena-observation-intent'] = intent
    if digest:
        headers['if-match'] = '"sha256:' + '1' * 64 + '"'
    with TestClient(create_app(service)) as client:
        response = client.head('/arena/v1/runs/r1/training-observations', headers=headers)
    assert response.status_code == status


def test_worker_entry_rechecks_final_authority_after_public_preflight(tmp_path, monkeypatch):
    service, objects, row, _ = terminal_service(tmp_path, kind='score', seconds=179)
    row['participation_accepted_at'] = None
    original = service.handle_training_observations
    entered = []
    def expired_after_preflight(*args):
        entered.append(True)
        service._store.get_input_finish_event = lambda _: None
        return original(*args)
    monkeypatch.setattr(service, 'handle_training_observations', expired_after_preflight)
    with public_gateway(service, monkeypatch) as (client, calls):
        response = client.post('/arena/v1/runs/r1/training-observations', json=payload(observation()),
            headers={'x-lab-arena-lease': TOKEN})
    assert response.status_code == 409
    assert [(method, status) for method, status, _ in calls] == [('HEAD', 204), ('POST', 409)]
    assert entered == [True] and not list((tmp_path / 'objects').rglob('*.gz'))
