"""Definite transaction rollback is temporary without amplifying RPC retries."""
import httpx
import pytest

from lab_arena.store import (
    ArenaStoreError, ArenaStoreUnavailable, DEADLOCK_RETRIES, PostgrestTransport,
)


@pytest.mark.parametrize('function', ['lab_arena_reserve_call', 'lab_arena_reserve_judgment_call', 'lab_arena_transition_round', 'lab_arena_whoami'])
@pytest.mark.parametrize('code', ['55P03', '40P01'])
def test_exhausted_proved_rollback_is_typed_availability(function, code):
    requests = []
    with httpx.Client(transport=httpx.MockTransport(lambda request:
        requests.append(request) or httpx.Response(500, json={'code': code, 'message': 'private SQL detail'}))) as client:
        transport = PostgrestTransport('https://db.example', service_key='sb_secret_test', http_client=client)
        with pytest.raises(ArenaStoreUnavailable) as failure:
            transport.rpc(function, {})
        assert failure.value.transaction_rolled_back is True
        expected_retries = DEADLOCK_RETRIES if code == '40P01' or 'reserve' in function else 0
        assert transport.deadlock_retries == expected_retries
        assert len(requests) == expected_retries + 1


@pytest.mark.parametrize('status,body', [
    (401, {'code': '55P03'}), (403, {'code': '42501'}), (503, {'code': '55P03'}),
    (500, {'code': '42501'}), (500, {'code': '22023'}), (500, {'code': 'P0001'}),
    (500, {'message': 'lock timeout'}), (500, {'code': ['55P03']}), (500, ['55P03']),
])
def test_auth_contract_unknown_or_malformed_failure_is_not_typed_availability(status, body):
    with httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(status, json=body))) as client:
        transport = PostgrestTransport('https://db.example', service_key='sb_secret_test', http_client=client)
        with pytest.raises((ArenaStoreError, AttributeError)) as failure:
            transport.rpc('lab_arena_whoami', {})
        assert not isinstance(failure.value, ArenaStoreUnavailable)
        assert transport.deadlock_retries == 0
