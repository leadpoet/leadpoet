"""Frozen source checks and broker holds do not replay paid provider work."""
from types import SimpleNamespace

import pytest

from lab_arena import service as arena_service
from tests.lab_arena.full_thinking_round_test import OPEN, opted_in
from tests.lab_arena.broker_reliability_test import ReleaseStore, execute
from tests.lab_arena.test_lab_arena_broker import FakeTransport, make_broker


@pytest.mark.parametrize('status', ['existing', 'promotion_pending'])
def test_frozen_baseline_rechecks_sql_without_source_fetch(status):
    service = opted_in()
    configuration = service.create_round(OPEN)
    calls = []
    service._store = SimpleNamespace(freeze_full_thinking_baseline=lambda rid, **params:
        calls.append((rid, params)) or {'status': status})
    service._config.baseline_source_fetcher = lambda *_args: pytest.fail('refetched a frozen source')
    row = dict(round_id=configuration['round_id'], champion_funding_frozen=True,
               configuration_doc=configuration)
    if status == 'promotion_pending':
        with pytest.raises(arena_service.ServiceError, match='baseline_promotion_pending'):
            service._freeze_full_thinking_baseline(row['round_id'], row)
    else:
        assert service._freeze_full_thinking_baseline(row['round_id'], row) == {'status': status}
    assert len(calls) == 1
    assert calls[0][1]['expected_commit'] is None
    assert calls[0][1]['source_url'] == configuration['baseline_source_url']


def test_publication_checks_source_before_comparing_or_reading_outputs():
    service = opted_in()
    configuration = service.create_round(OPEN)
    row = dict(round_id=configuration['round_id'], status='scored', champion_funding_frozen=True,
               configuration_doc=configuration)
    service._round = lambda _: row
    service._store = SimpleNamespace(freeze_full_thinking_baseline=lambda *_args, **_kwargs:
        {'status': 'promotion_pending'})
    with pytest.raises(arena_service.ServiceError, match='baseline_promotion_pending'):
        service._publish_locked(row['round_id'])


@pytest.mark.parametrize('entrypoint', ['reserve_call', 'mark_dispatched'])
def test_broker_promotion_hold_never_reaches_transport(entrypoint):
    store = ReleaseStore()
    setattr(store, entrypoint, lambda **_kwargs: {'status': 'promotion_pending'})
    broker, _, transport = make_broker(store=store, transport=FakeTransport([]))
    result = execute(broker)
    assert transport.sent == []
    assert result.call['outcome'] == 'not_dispatched'
    assert result.call['reason'] == 'baseline_promotion_pending'
    assert not any(name == 'settle' for name in store.log)
