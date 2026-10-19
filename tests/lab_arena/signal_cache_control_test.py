"""Exercise the real scorer socket and authenticated gateway transport."""
import asyncio
from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace

import httpx
import pytest

from lab_arena import contracts, runner, scoring, shim, signal_cache_control as control
from lab_arena.service import ArenaService, RoundDefaults
from qualification.scoring.signal_judgment_cache import SignalJudgmentCacheError
from tests.lab_arena.test_benchmark_disclosure_policy import _RoundStore


def request():
    return {'operation': 'claim', 'company_index': 0, 'signal_index': 0,
            'verified_company_identity': {'decision': 'match'},
            'verified_funding_stage_context': []}


def state(*, kind='score', enabled=True):
    return runner.RunState(lease={'run_id': 'score-one', 'kind': kind,
        'scorer_policy': {'signal_judgment_policy': control.POLICY} if enabled else {}},
        lease_token='lease-test-only')


def test_socket_to_gateway_has_no_provider_debit(monkeypatch):
    requests = []
    def gateway(req):
        assert req.url.path == '/arena/v1/runs/score-one/signal-judgment'
        assert req.headers['x-lab-arena-lease'] == 'lease-test-only'
        requests.append(json.loads(req.content))
        return httpx.Response(200, json={'status': 'miss', 'reference': {'key': 'example'}})
    run = state()
    with httpx.Client(transport=httpx.MockTransport(gateway)) as http:
        api = runner.HttpArenaApiClient('https://arena.example', client=http)
        with tempfile.TemporaryDirectory(prefix='signal-', dir='/tmp') as directory:
            path = Path(directory) / 'worker.sock'
            worker = runner.WorkerSocketServer(path, api, run)
            worker.start()
            monkeypatch.setenv(shim.WORKER_SOCKET_ENV, str(path))
            monkeypatch.setenv(shim.TRUSTED_SCORER_ENV, '1')
            try:
                result = asyncio.run(control.exchange(request()))
            finally:
                worker.stop()
    assert result['status'] == 'miss' and requests == [request()]
    assert run.action_sequence == 0 and run.calls == [] and run.refusals == 0


@pytest.mark.parametrize('kind,enabled', [('execute', True), ('score', False)])
def test_workers_deny_miner_or_unfrozen_access(tmp_path, kind, enabled):
    api = SimpleNamespace(signal_judgment=lambda *_: pytest.fail('unauthorized gateway call'))
    worker = runner.WorkerSocketServer(tmp_path/'s', api, state(kind=kind, enabled=enabled))
    raw = contracts.canonical_json({'schema_version': control.SCHEMA, 'request': request()}).encode()
    assert json.loads(worker.handle_frame(raw)) == {'status': 'unavailable'}


@pytest.mark.parametrize('field,value', [('company_index', True), ('signal_index', -1),
    ('run_id', 'foreign'), ('lease_token', 'foreign'), ('cache_key', 'foreign')])
def test_control_rejects_context_override(field, value):
    with pytest.raises(ValueError):
        control.validate_request({**request(), field: value})


def test_busy_wait_replays_only_the_control_request(monkeypatch):
    seen = []
    def exchange(value):
        seen.append(deepcopy(value))
        return {'status': 'busy'} if len(seen) == 1 else {'status': 'miss', 'reference': {}}
    async def no_sleep(_):
        pass
    monkeypatch.setattr(control, '_exchange_once', exchange)
    monkeypatch.setattr(control.asyncio, 'sleep', no_sleep)
    assert asyncio.run(control.exchange(request()))['status'] == 'miss'
    assert seen == [request(), request()]


def test_no_miner_shim_access(monkeypatch):
    monkeypatch.delenv(shim.TRUSTED_SCORER_ENV, raising=False)
    with pytest.raises(SignalJudgmentCacheError, match='scorer mode'):
        control._exchange_once(request())


def test_policy_roundtrip_and_invalid_combination():
    policy = scoring.build_scorer_policy(scoring_adapter_version='qualification_integrity_v2', company_judgment_cache=True)
    policy['signal_judgment_policy'] = control.POLICY
    assert contracts.validate_scorer_policy(json.loads(contracts.canonical_json(policy))) == policy
    policy.pop('company_judgment_policy')
    with pytest.raises(contracts.ArenaContractError, match='company judgment'):
        contracts.validate_scorer_policy(policy)


@pytest.mark.parametrize("quality", [False, True])
def test_activation_preserves_existing_round_and_only_changes_future_policy(quality):
    defaults = RoundDefaults(
        baseline_hotkey='5'+'B'*47, baseline_source_url='https://example.com/baseline.tar.gz',
        scorer_image_digest='sha256:'+'1'*64,
        scorer_image_reference='registry.example/scorer@sha256:'+'1'*64,
        integrity_from='2026-10-01T00:00:00Z', company_judgment_from='2026-10-01T00:00:00Z',
        signal_judgment_from='2026-10-20T00:00:00+00:00',
        company_quality_from='2026-10-01T00:00:00Z' if quality else None,
    )
    service = object.__new__(ArenaService)
    service._config = SimpleNamespace(defaults=defaults, mode='shadow', network_name='test', netuid=401, pinned_round_id=None)
    service._store = _RoundStore()
    service._scorer_policy = scoring.build_scorer_policy()
    service.runner_settings = lambda: (['5'+'C'*47], [])
    service._require_integrity_schema = lambda: None
    service._require_company_quality_schema = lambda: None
    cutoff = datetime(2026, 10, 20, tzinfo=timezone.utc)
    old = service.create_round(cutoff, round_id='arena-2026-10-20-test')
    assert 'signal_judgment_policy' not in old['scorer_policy']
    service._config.defaults = replace(defaults, signal_judgment_from='2026-10-19T00:00:00Z')
    assert service.create_round(cutoff, round_id='arena-2026-10-20-test') == old
    service._config.defaults = defaults
    future = service.create_round(cutoff+timedelta(days=1), round_id='arena-2026-10-21-test')
    if quality:
        assert future['scorer_policy']['company_quality_policy'] == 'company_quality_v1'
        assert 'signal_judgment_policy' not in future['scorer_policy']
        assert 'company_judgment_policy' not in future['scorer_policy']
    else:
        assert future['scorer_policy']['signal_judgment_policy'] == control.POLICY
    assert {k:v for k,v in future['scorer_policy'].items() if k!='signal_judgment_policy'} == old['scorer_policy']


def test_gateway_route_uses_live_lease_contract_and_no_store():
    from fastapi.testclient import TestClient
    from lab_arena.api import create_app
    from tests.lab_arena.test_lab_arena_api import StubService
    service = StubService()
    calls = []
    def handle(run_id, lease_token, body):
        calls.append((run_id, lease_token, body))
        return {"status": "miss", "reference": {}}
    service.handle_signal_judgment = handle
    with TestClient(create_app(service)) as client:
        response = client.post('/arena/v1/runs/score-one/signal-judgment',
            headers={'x-lab-arena-lease': 'a'*64}, json=request())
        assert response.status_code == 200
        assert 'no-store' in response.headers['cache-control']
        assert calls == [('score-one', 'a'*64, request())]
        denied = client.post('/arena/v1/runs/score-one/signal-judgment', json=request())
        assert denied.status_code == 401
        assert len(calls) == 1


def test_signal_activation_cannot_precede_company_activation():
    from lab_arena.service import ServiceConfig, ServiceError
    base = dict(mode='off', store=None, object_store=None, signer=None, chain=None,
        verify_signature=None, daily_icp_source=None, banned_hotkeys_source=None, broker_factory=None)
    with pytest.raises(ServiceError, match='signal_judgment_activation_invalid'):
        ServiceConfig(**base, defaults=RoundDefaults(integrity_from='2026-10-01T00:00:00Z',
            company_judgment_from='2026-10-20T00:00:00Z', signal_judgment_from='2026-10-19T00:00:00Z'))
