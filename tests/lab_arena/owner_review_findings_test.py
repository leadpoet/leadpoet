"""Owner excerpts stay authenticated, bounded, redacted, and out of public data."""
from datetime import datetime, timezone
import json
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient
import httpx
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
import pytest

from lab_arena import contracts, miner_cli, miner_submit, owner_review, public_dashboard
from lab_arena.api import create_app
from lab_arena.service import ArenaService
from gateway.api import arena_proxy
from gateway.middleware.priority import classify_path
from tests.lab_arena.test_lab_arena_code_review_runtime import (
    ReviewTransport, _archive, _review_body, _reviewer, _source,
)

NOW = datetime(2026, 11, 10, tzinfo=timezone.utc)
RID = 'arena-2026-11-11'
SID = 'sub-owner'
HOTKEY = '5' + 'A' * 47
OTHER = '5' + 'B' * 47
EVIDENCE = 'private_owner_source_call()'
CATEGORY = 'fabricated_evidence'


def findings(evidence=EVIDENCE):
    return [{'category': CATEGORY, 'file': 'harness.py', 'evidence': evidence,
             'explanation': 'OTHER_MINER_SOURCE and PRIVATE_MODEL_PROSE'}]


def row():
    return {'submission_id': SID, 'round_id': RID, 'miner_hotkey': HOTKEY,
            'owner_coldkey': 'same-coldkey', 'status': 'rejected',
            'code_review_status': 'rejected', 'rejection_rule': 'code_review_rejected',
            'code_review_doc': {'categories': [CATEGORY], 'owner_findings':
                                owner_review.from_validated_findings(findings())}}


def test_excerpt_utf8_redaction_bounds_and_never_prose():
    values = findings('api_key="sk-or-v1-' + 'a' * 48 + '"\n' + 'é' * 1000) * 9
    result = owner_review.from_validated_findings(values)
    assert len(result['findings']) == 8 and result['truncated'] and result['redacted']
    serialized = json.dumps(result)
    assert 'aaaaa' not in serialized and 'OTHER_MINER_SOURCE' not in serialized
    assert set(result['findings'][0]) == {'category', 'file', 'evidence'}
    assert all(len(item['evidence'].encode()) <= 1000 for item in result['findings'])
    known = owner_review.from_validated_findings(findings('private-key-value'), secrets=('private-key-value',))
    assert known['findings'][0]['evidence'] == '[REDACTED]'
    controls = owner_review.from_validated_findings(findings('\x01' * 1000) * 8)
    assert len(contracts.canonical_json(controls).encode()) <= 16000 and controls['truncated']
    assert owner_review.from_validated_findings(findings('\x00'))['findings'] == []


@pytest.mark.parametrize('bad', [None, {}, {'findings': [None], 'truncated': False, 'redacted': False},
    {'findings': [{'category': [], 'file': 'h.py', 'evidence': 'x'}], 'truncated': False, 'redacted': False},
    {'findings': [{'category': CATEGORY, 'file': 'h.py', 'evidence': 'x' * 1001}], 'truncated': False, 'redacted': False}])
def test_historical_and_malformed_saved_findings_fail_closed(bad):
    saved = row()
    saved['code_review_doc']['owner_findings'] = bad
    assert owner_review.private_projection(saved)['available'] is False
    saved['code_review_doc']['owner_findings'] = owner_review.from_validated_findings(findings())
    for status in ('pending', 'reviewing', 'passed', 'error'):
        saved['code_review_status'] = status
        assert owner_review.private_projection(saved)['available'] is False


@pytest.fixture
def signed_client():
    keys = {HOTKEY: Ed25519PrivateKey.generate(), OTHER: Ed25519PrivateKey.generate()}
    saved = row()
    round_row = {'round_id': RID, 'status': 'published', 'configuration_doc': {'mode': 'live'}}
    reads = []
    def verify(hotkey, signature, message):
        try:
            keys[hotkey].public_key().verify(bytes.fromhex(signature.removeprefix('0x')), message.encode())
            return True
        except (InvalidSignature, ValueError, KeyError):
            return False
    service = object.__new__(ArenaService)
    service.now = lambda: NOW
    service._config = SimpleNamespace(mode='live', network_name='finney', netuid=71,
                                      pinned_round_id=None, verify_signature=verify)
    service._store = SimpleNamespace(
        get_round=lambda rid: {**round_row, 'round_id': rid},
        get_submission=lambda sid: (reads.append(sid) or saved) if sid == SID else None,
    )
    service._round = lambda rid: round_row
    def envelope(*, hotkey=HOTKEY, scope=contracts.SCOPE_SUBMISSION_REVIEW,
                 round_id=RID, body=None, timestamp=None):
        return contracts.build_signed_request(
            scope=scope, round_id=round_id, hotkey=hotkey,
            timestamp=int(NOW.timestamp()) if timestamp is None else timestamp,
            body={'submission_id': SID} if body is None else body,
            sign_message=lambda message: keys[hotkey].sign(message.encode()).hex(),
        )
    with TestClient(create_app(service)) as client:
        yield client, service, saved, round_row, envelope, reads


def post(client, envelope, sid=SID):
    response = client.post('/arena/v1/submissions/' + sid + '/review', json=envelope)
    assert response.headers['cache-control'] == 'private, no-store'
    assert response.headers['x-content-type-options'] == 'nosniff'
    return response


def test_real_signed_owner_read_and_public_status_nonleak(signed_client):
    client, service, saved, round_row, envelope, reads = signed_client
    response = post(client, envelope())
    assert response.status_code == 200
    assert response.json()['owner_findings']['findings'][0]['evidence'] == EVIDENCE
    assert EVIDENCE not in client.get('/arena/v1/submissions/' + SID).text
    assert EVIDENCE not in json.dumps(public_dashboard.code_review_summary(saved))
    assert 'PRIVATE_MODEL_PROSE' not in response.text and 'OTHER_MINER_SOURCE' not in response.text
    del saved['code_review_doc']['owner_findings']
    assert post(client, envelope()).json()['owner_findings']['available'] is False


@pytest.mark.parametrize('variant,expected', [
    ('sibling',403), ('scope',400), ('stale',400), ('signature',401),
    ('round',403), ('path',400), ('payload',400), ('mode',409), ('network',409), ('banned',403),
])
def test_owner_read_rejects_invalid_signed_access(signed_client, variant, expected):
    client, service, saved, round_row, envelope, reads = signed_client
    kwargs = {}
    sid = SID
    if variant == 'sibling': kwargs['hotkey'] = OTHER
    if variant == 'scope': kwargs['scope'] = contracts.SCOPE_SUBMISSION_FINALIZE
    if variant == 'stale': kwargs['timestamp'] = int(NOW.timestamp()) - 10000
    if variant == 'round': kwargs['round_id'] = 'arena-2026-11-10'
    if variant == 'path': sid = 'sub-other'
    if variant == 'payload': kwargs['body'] = {'submission_id': SID, 'extra': EVIDENCE}
    if variant == 'mode': round_row['configuration_doc']['mode'] = 'shadow'
    if variant == 'network': round_row['configuration_doc']['network_name'] = 'test'
    if variant == 'banned': round_row['configuration_doc']['banned_hotkeys'] = [HOTKEY]
    request = envelope(**kwargs)
    if variant == 'signature': request['signature'] = '0' * 128
    response = post(client, request, sid)
    assert response.status_code == expected, response.text
    assert EVIDENCE not in response.text
    if variant not in ('sibling','round'):
        assert reads == []


def test_bad_json_remains_private_no_store(signed_client):
    client, *_ = signed_client
    response = client.post('/arena/v1/submissions/' + SID + '/review', content='{')
    assert response.status_code == 400 and response.headers['cache-control'] == 'private, no-store'


def test_gateway_forwards_signed_private_read_and_headers(signed_client, monkeypatch):
    sidecar, service, saved, round_row, envelope, reads = signed_client
    forwarded = []
    async def forward(method, path, *, query, body, headers):
        forwarded.append((method, path))
        response = sidecar.request(method, '/arena/' + path, content=body, headers=headers)
        return httpx.Response(response.status_code, content=response.content, headers=response.headers)
    monkeypatch.setenv('LAB_ARENA_MODE', 'live')
    monkeypatch.setattr(arena_proxy, '_request_sidecar', forward)
    app = FastAPI()
    app.include_router(arena_proxy.router)
    with TestClient(app) as gateway:
        assert post(gateway, envelope()).json()['owner_findings']['available']
        assert post(gateway, envelope(hotkey=OTHER)).status_code == 403
    assert forwarded == [('POST','v1/submissions/' + SID + '/review')] * 2
    assert classify_path('/arena/v1/submissions/' + SID + '/review') == 'miner'




@pytest.mark.parametrize('cost', ['0.001234', None])
def test_runtime_rejected_excerpt_persistence_preserves_unknown_cost(cost):
    source = _source()
    evidence = source['src/agent.py'].decode().strip()
    def responder(parameters):
        items = [{'category': CATEGORY, 'file': 'src/agent.py', 'evidence': evidence,
                  'explanation': 'private explanation'}]
        return {'status': 200, 'body': _review_body(parameters, verdict='reject', findings=items, cost=cost)}
    reviewer, submitted, store, _ = _reviewer(_archive(source), ReviewTransport(responder))
    reviewer.review(submitted)
    assert store.finish_calls[0][3] == 'rejected'
    assert store.finish_calls[0][4]['owner_findings']['findings'][0]['evidence'] == evidence
    assert 'explanation' not in repr(store.finish_calls[0][4])


def test_optional_findings_failure_preserves_verdict(monkeypatch):
    def unavailable(*args, **kwargs): raise ValueError('private')
    monkeypatch.setattr(owner_review, 'from_validated_findings', unavailable)
    source = _source()
    def responder(parameters):
        return {'status': 200, 'body': _review_body(parameters, verdict='reject', findings=[{
            'category': CATEGORY, 'file': 'src/agent.py',
            'evidence': source['src/agent.py'].decode().strip(), 'explanation': 'private'}])}
    reviewer, submitted, store, _ = _reviewer(_archive(source), ReviewTransport(responder))
    reviewer.review(submitted)
    assert store.finish_calls[0][3] == 'rejected' and 'owner_findings' not in store.finish_calls[0][4]


def test_runtime_redacts_source_credential_before_persistence():
    source = _source()
    source['prompts/evidence.txt'] = b'api_key="sk-or-v1-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"\n'
    def responder(parameters):
        return {'status': 200, 'body': _review_body(parameters, verdict='reject', findings=[{
            'category': CATEGORY, 'file': 'prompts/evidence.txt',
            'evidence': source['prompts/evidence.txt'].decode().strip(),
            'explanation': 'private'}])}
    reviewer, submitted, store, _ = _reviewer(_archive(source), ReviewTransport(responder))
    reviewer.review(submitted)
    assert store.finish_calls[0][3] == 'rejected'
    document = store.finish_calls[0][4]
    assert document['owner_findings']['redacted'] is True
    assert 'aaaaaaaaaa' not in repr(document)
    assert '[REDACTED]' in repr(document['owner_findings'])


def test_signed_cli_read_and_parser():
    key = Ed25519PrivateKey.generate()
    class Keypair:
        ss58_address = HOTKEY
        def sign(self, payload): return key.sign(payload)
    calls = []
    class Session:
        def post(self, url, **kwargs):
            calls.append((url, kwargs))
            request = kwargs['json']
            key.public_key().verify(bytes.fromhex(request['signature'].removeprefix('0x')), contracts.signed_request_message(request).encode())
            return SimpleNamespace(status_code=200, json=lambda: {
                'round_id': RID, 'submission_id': SID, 'owner_findings': {'available': False}})
    result = miner_submit.read_submission_review(round_id=RID, submission_id=SID,
        api_base_url='https://arena.example', keypair=Keypair(), session=Session(), now=NOW.timestamp)
    assert result['submission_id'] == SID
    assert calls[0][0].endswith('/' + SID + '/review')
    assert calls[0][1]['json']['scope'] == contracts.SCOPE_SUBMISSION_REVIEW
    assert calls[0][1]['allow_redirects'] is False
    args = miner_cli.build_parser().parse_args(['review','--round-id',RID,'--submission-id',SID])
    assert args.command == 'review'
