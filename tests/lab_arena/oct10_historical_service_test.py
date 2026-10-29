"""The Oct10 algorithm boundary cannot replace current or foreign authority."""
from copy import deepcopy
from types import SimpleNamespace

import pytest

from lab_arena.store import ArenaStore, PsycopgTransport
from tests.lab_arena import oct10_historical_service as historical
from tests.lab_arena import oct10_uniform_pe_rejudge445_postgres_test as migration
from tests.lab_arena.lab_arena_pg_harness import current_service_database

# Reuse the exact migration fixture, not a handwritten old schema.
database = migration.database
migrated = migration.migrated
seeded = migration.seeded
ROUND = 'arena-2026-10-10'


@pytest.fixture()
def old_service(seeded):
    pg, dsn = seeded
    connect = lambda: pg.connect(**dsn)
    store = ArenaStore(PsycopgTransport(connect))
    try:
        yield SimpleNamespace(_store=store), connect
    finally:
        store.close()


def test_exact_algorithms_and_profile_accept_the_historical_schema(old_service):
    service, connect = old_service
    historical.validate_oct10_algorithms()
    historical.pin_oct10_service(service, connect, round_id=ROUND)
    assert callable(service.commit_scoring_plan) and callable(service.publish)
    with pytest.raises(AssertionError, match='foreign Oct10 round'):
        service.publish('arena-2026-10-11')


def test_current_database_cannot_use_the_historical_boundary():
    generator = current_service_database()
    pg, dsn = next(generator)
    try:
        with pytest.raises(AssertionError, match='cannot replace current authority'):
            historical.pin_oct10_service(SimpleNamespace(), lambda: pg.connect(**dsn), round_id=ROUND)
    finally:
        generator.close()


@pytest.mark.parametrize('change', ['mode', 'adapter', 'benchmark', 'sequence', 'cost', 'date'])
def test_foreign_frozen_profiles_fail_before_binding(old_service, monkeypatch, change):
    service, connect = old_service
    row = deepcopy(service._store.get_round(ROUND))
    config = row['configuration_doc']
    if change == 'mode': config['mode'] = 'shadow'
    elif change == 'adapter': config['scorer_policy']['scoring_adapter_version'] = 'qualification_contacts_v3'
    elif change == 'benchmark': config.update(stage_1_icp_count=10, stage_2_icp_count=10)
    elif change == 'sequence': config.pop('execution_sequence_policy')
    elif change == 'cost': config['sourcing_cost_eligibility_policy'] = 'successful_calls_per_icp_flat_v1'
    else: row['evaluation_date'] = '2026-10-11'
    monkeypatch.setattr(service._store, 'get_round', lambda _: row)
    with pytest.raises(AssertionError):
        historical.pin_oct10_service(service, connect, round_id=ROUND)
    assert not hasattr(service, 'commit_scoring_plan')


@pytest.mark.parametrize('method', ['commit_scoring_plan', 'publish'])
def test_bound_methods_reject_configuration_drift(old_service, monkeypatch, method):
    service, connect = old_service
    historical.pin_oct10_service(service, connect, round_id=ROUND)
    row = deepcopy(service._store.get_round(ROUND))
    row['configuration_doc']['promotion_margin'] = 99
    monkeypatch.setattr(service._store, 'get_round', lambda _: row)
    with pytest.raises(AssertionError, match='frozen configuration drift'):
        getattr(service, method)(ROUND, *([1] if method == 'commit_scoring_plan' else []))


@pytest.mark.parametrize('change', ['manifest', 'ast'])
def test_algorithm_hash_and_actual_ast_drift_fail_closed(monkeypatch, change):
    if change == 'manifest':
        monkeypatch.setitem(historical.OCT10_AST_SHA256, 'publish', '0' * 64)
    else:
        original = historical.inspect.getsource
        def changed(function):
            source = original(function)
            return source.replace('"provider_error"', '"model_error"', 1) if function is historical._oct10_build_scoring_plan else source
        monkeypatch.setattr(historical.inspect, 'getsource', changed)
    with pytest.raises(AssertionError, match='historical algorithm drift'):
        historical.validate_oct10_algorithms()


@pytest.mark.parametrize('change', ['source', 'acl'])
def test_sql_protocol_or_permissions_drift_fail_before_binding(old_service, change):
    service, connect = old_service
    signature = 'public.lab_arena_record_run_scores(text,smallint,jsonb)'
    with connect() as connection, connection.cursor() as cursor:
        cursor.execute('SELECT pg_get_functiondef(%s::regprocedure)', (signature,))
        original = cursor.fetchone()[0]
        if change == 'source': cursor.execute(original.replace('BEGIN', 'BEGIN -- historical drift', 1))
        else: cursor.execute('GRANT EXECUTE ON FUNCTION ' + signature + ' TO anon')
    try:
        with pytest.raises(AssertionError, match='Oct10 SQL (protocol|permissions) drift'):
            historical.pin_oct10_service(service, connect, round_id=ROUND)
        assert not hasattr(service, 'commit_scoring_plan')
    finally:
        with connect() as connection, connection.cursor() as cursor:
            cursor.execute(original)
            cursor.execute('REVOKE EXECUTE ON FUNCTION ' + signature + ' FROM anon')
