"""A lost claim response must not allocate another assignment on the next poll."""
from datetime import timedelta
import json
from types import SimpleNamespace

import httpx
import pytest

from lab_arena import contracts, runner as rn
from tests.lab_arena.test_lab_arena_runner import (
    BridgingRuntime, FakeApi, ROUND, lease, make_config, verify,
)


@pytest.mark.parametrize('failure', ('http_503', 'read_timeout'))
def test_late_committed_claim_survives_short_retry_exhaustion_and_runs_once(
    tmp_path, failure,
):
    requests = []
    allocations = {}

    def respond(request):
        requests.append(request.content)
        document = json.loads(request.content)
        contracts.validate_signed_request(
            document, expected_scope=contracts.SCOPE_CLAIM,
            expected_round_id=ROUND, now=int(config.clock().timestamp()),
            verify_signature=verify,
        )
        allocations.setdefault(document['request_id'], lease())
        if len(requests) <= 3:
            if failure == 'read_timeout':
                raise httpx.ReadTimeout('lost response', request=request)
            return httpx.Response(503, json={'code': 'arena_store_unavailable'})
        return httpx.Response(200, json=allocations[document['request_id']])

    client = httpx.Client(transport=httpx.MockTransport(respond))

    class ClaimApi(FakeApi):
        def claim(self, envelope):
            return http_api.claim(envelope)

    http_api = rn.HttpArenaApiClient('http://localhost', client=client)
    api = ClaimApi([])
    sandbox = BridgingRuntime(output={'companies': []}, calls=0)
    config = make_config(tmp_path, api, sandbox)
    (tmp_path / 'work').mkdir()
    runner = rn.Runner(config)
    now = config.clock()
    try:
        assert runner.run_once(max_claims=1) == 0
        assert len(allocations) == 1
        assert sandbox.specs == [] and api.completions == []
        config.clock = lambda: now + timedelta(seconds=30)
        assert runner.run_once(max_claims=1) == 1
        assert len(requests) == 4 and len(set(requests)) == 1
        assert len(allocations) == 1
        assert len(sandbox.specs) == 1 and len(api.completions) == 1
        assert runner.completed == [{'run_id': 'r1', 'result': {'status': 'accepted'}}]
        assert runner._pending_claims == {}
    finally:
        runner.close()
        http_api.close()


@pytest.mark.parametrize('status', (
    'no_pending', 'no_open_round', 'stage_closed', 'paused', 'no_free_slot',
    'cache_busy', 'rejected', 'request_id_reused', 'no_work',
))
def test_definite_nonlease_answer_allows_a_new_request(tmp_path, status):
    class Api(FakeApi):
        def claim(self, envelope):
            self.claims.append(envelope)
            return {'status': status}

    api = Api([])
    runner = rn.Runner(make_config(tmp_path, api, BridgingRuntime(calls=0)))
    try:
        assert runner.claim_one()['status'] == status
        assert runner.claim_one()['status'] == status
        assert api.claims[0]['request_id'] != api.claims[1]['request_id']
        assert runner._pending_claims == {}
    finally:
        runner.close()


@pytest.mark.parametrize('response', ({}, {'status': []}, {'status': {}}, {'status': 'unknown_status'}))
def test_untyped_or_unknown_answer_retains_identity_without_dispatch(tmp_path, response):
    class Api(FakeApi):
        def claim(self, envelope):
            self.claims.append(envelope)
            return response if len(self.claims) == 1 else {'status': 'no_pending'}

    api = Api([])
    sandbox = BridgingRuntime(calls=0)
    runner = rn.Runner(make_config(tmp_path, api, sandbox))
    try:
        assert runner.run_once() == 0
        assert runner.run_once() == 0
        assert api.claims[0] is api.claims[1]
        assert sandbox.specs == [] and api.completions == []
        assert runner._pending_claims == {}
    finally:
        runner.close()


@pytest.mark.parametrize('response', (None, [], 12))
def test_nonobject_answer_stays_ambiguous_across_invocations(tmp_path, response):
    class Api(FakeApi):
        def claim(self, envelope):
            self.claims.append(envelope)
            return response if len(self.claims) <= 3 else {'status': 'no_pending'}

    api = Api([])
    runner = rn.Runner(make_config(tmp_path, api, BridgingRuntime(calls=0)))
    try:
        with pytest.raises(rn.RunnerError, match='non-object'):
            runner.claim_one()
        assert runner.claim_one() == {'status': 'no_pending'}
        assert len(api.claims) == 4
        assert all(x is api.claims[0] for x in api.claims)
    finally:
        runner.close()


def test_expired_ambiguous_claim_blocks_without_resigning_or_new_request(tmp_path):
    class Api(FakeApi):
        def claim(self, envelope):
            self.claims.append(envelope)
            if envelope['round_id'] == ROUND:
                raise rn.RunnerError('availability failure', http_status=503)
            return {'status': 'no_pending'}

    api = Api([])
    config = make_config(tmp_path, api, BridgingRuntime(calls=0))
    runner = rn.Runner(config)
    now = config.clock()
    try:
        with pytest.raises(rn.RunnerError):
            runner.claim_one()
        pending = runner._pending_claims[ROUND]
        config.clock = lambda: now + timedelta(seconds=301)
        with pytest.raises(rn.RunnerError, match='pending claim timestamp expired'):
            runner.claim_one()
        assert len(api.claims) == 3
        assert runner._pending_claims[ROUND] is pending
        # One uncertain round does not prevent independent work in another.
        assert runner.claim_one('arena-2026-09-03') == {'status': 'no_pending'}
        assert api.claims[-1]['request_id'] != pending['request_id']
        assert runner._pending_claims[ROUND] is pending
    finally:
        runner.close()


def test_timestamp_rejection_after_window_does_not_discard_ambiguous_claim(tmp_path):
    class Api(FakeApi):
        def claim(self, envelope):
            self.claims.append(envelope)
            if len(self.claims) <= 3:
                raise rn.RunnerError('availability failure', http_status=503)
            config.clock = lambda: now + timedelta(seconds=301)
            return {'status': 'rejected'}

    api = Api([])
    config = make_config(tmp_path, api, BridgingRuntime(calls=0))
    now = config.clock()
    runner = rn.Runner(config)
    try:
        with pytest.raises(rn.RunnerError):
            runner.claim_one()
        config.clock = lambda: now + timedelta(seconds=299)
        with pytest.raises(rn.RunnerError, match='pending claim timestamp expired'):
            runner.claim_one()
        assert len(api.claims) == 4
        assert runner._pending_claims[ROUND] is api.claims[0]
        assert all(x is api.claims[0] for x in api.claims)
    finally:
        runner.close()


def test_capacity_change_does_not_change_pending_signed_body(tmp_path):
    class Api(FakeApi):
        def claim(self, envelope):
            self.claims.append(envelope)
            if len(self.claims) <= 3:
                raise rn.RunnerError('availability failure', http_status=503)
            return {'status': 'no_pending'}

    api = Api([])
    config = make_config(tmp_path, api, BridgingRuntime(calls=0), parallel=2)
    runner = rn.Runner(config)
    try:
        with pytest.raises(rn.RunnerError):
            runner.claim_one()
        pending = runner._pending_claims[ROUND]
        original_hash = contracts.request_bytes_hash(pending)
        config.proxy_worker_pool = SimpleNamespace(capacity=2, quarantined=2)
        assert runner.claim_one() == {'status': 'no_pending'}
        assert len(api.claims) == 3 and runner._pending_claims[ROUND] is pending
        config.proxy_worker_pool.quarantined = 1
        assert runner.claim_one() == {'status': 'no_pending'}
        assert api.claims[-1] is pending
        assert pending['body']['declared_parallelism'] == 2
        assert contracts.request_bytes_hash(pending) == original_hash
    finally:
        runner.close()
