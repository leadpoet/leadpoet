"""Store failures retain a bounded recovery class without secret text."""
import pytest

from lab_arena.broker import _safe_exception_class
from lab_arena.store import ArenaStoreError, ArenaStoreUnavailable


class PrivateStoreSubclass(ArenaStoreUnavailable):
    pass


@pytest.mark.parametrize('error', [ArenaStoreError, ArenaStoreUnavailable, PrivateStoreSubclass])
def test_store_family_retains_existing_bounded_class(error):
    assert _safe_exception_class(error('private credential and transport text')) == 'ArenaStoreError'


def test_unrecognized_exception_stays_generic():
    assert _safe_exception_class(RuntimeError('private credential')) == 'Exception'
    assert _safe_exception_class(ValueError('private credential')) == 'ValueError'


def test_unavailable_settlement_stores_recoverable_family_and_exact_success(monkeypatch):
    from tests.lab_arena.broker_accounting_outage_test import clock_fixture
    from tests.lab_arena.test_lab_arena_broker import CONTEXT, CHAT, FakeTransport, make_broker
    clock = clock_fixture(monkeypatch)
    broker, store, transport = make_broker(transport=FakeTransport([(200, {
        'model': 'openai/gpt-4o-mini', 'choices': [], 'usage': {'cost': '0.0000123'},
    })]))

    def unavailable(**kwargs):
        raise ArenaStoreUnavailable('private transport detail')

    store.settle_call = unavailable
    result = broker.execute(CONTEXT, operation_id='openrouter.chat', parameters=CHAT,
                            action_sequence=0, timeout_ms=5000)
    assert clock.now == 120
    assert result.call['outcome'] == 'uncertain'
    witness = store.calls[result.call['call_identity']]['uncertain_doc']
    assert witness['reason'] == 'settle_failure'
    assert witness['failure_stage'] == 'settlement'
    assert witness['error_class'] == 'ArenaStoreError'
    assert witness['call_succeeded'] is True
    assert witness['known_actual_microusd'] == 13
    assert len(transport.sent) == 1
    assert 'private transport detail' not in str(witness)
