"""Historical recovery boundaries cannot disable current service authority."""
from types import SimpleNamespace

import pytest

from tests.lab_arena.historical_service import pin_pre470_baseline_service, pin_pre484_credit_terminal, seed_pre428_initial_reward
from tests.lab_arena.lab_arena_pg_harness import (
    CURRENT_SERVICE_MIGRATIONS, current_service_database,
    database_with_lab_arena_migration,
)


def test_pre470_boundary_rejects_current_database():
    generator = current_service_database()
    pg, dsn = next(generator)
    try:
        with pytest.raises(AssertionError, match='current recovery authority'):
            pin_pre470_baseline_service(SimpleNamespace(), lambda: pg.connect(**dsn), round_id='arena-2026-10-10')
        with pytest.raises(AssertionError, match='current reward authority'):
            seed_pre428_initial_reward(SimpleNamespace(), lambda: pg.connect(**dsn), round_id='arena-2026-09-20')
        with pytest.raises(AssertionError, match='terminal-credit authority'):
            pin_pre484_credit_terminal(SimpleNamespace(), lambda: pg.connect(**dsn), round_id='arena-2026-11-23')
    finally:
        generator.close()


def test_pre470_boundary_keeps_frozen_round_scope_and_rejects_modern_dates():
    generator = database_with_lab_arena_migration(CURRENT_SERVICE_MIGRATIONS)
    pg, dsn = next(generator)
    row = {'round_id': 'arena-2026-10-10', 'evaluation_date': '2026-10-29',
           'configuration_doc': {'mode': 'live'}}
    service = SimpleNamespace(_store=SimpleNamespace(get_round=lambda _: row,
        get_run=lambda source: {'round_id': row['round_id'] if source == 'bound-source' else 'arena-2026-10-11'}))
    try:
        pin_pre470_baseline_service(service, lambda: pg.connect(**dsn), round_id=row['round_id'])
        assert service._recover_baseline(row) is None
        assert service._terminal_miner_credit_judgment('bound-source') is False
        with pytest.raises(AssertionError):
            service._terminal_miner_credit_judgment('other-source')
        configuration = row['configuration_doc']
        row['configuration_doc'] = {**configuration, 'changed': True}
        with pytest.raises(AssertionError):
            service._terminal_miner_credit_judgment('bound-source')
        row['configuration_doc'] = configuration
        with pytest.raises(AssertionError):
            service._recover_baseline({**row, 'round_id': 'arena-2026-10-11'})
        with pytest.raises(AssertionError):
            service._recover_baseline({**row, 'evaluation_date': '2026-10-11'})
        with pytest.raises(AssertionError):
            pin_pre470_baseline_service(
                SimpleNamespace(_store=SimpleNamespace(get_round=lambda _: {**row, 'round_id': 'arena-2026-10-14', 'evaluation_date': '2026-10-14'})),
                lambda: pg.connect(**dsn), round_id='arena-2026-10-14',
            )
        with pg.connect(**dsn) as connection, connection.cursor() as cursor:
            cursor.execute("CREATE FUNCTION public.lab_arena__miner_credit_judgment_terminal_v1(text) "
                           "RETURNS boolean LANGUAGE sql AS 'SELECT true'")
        with pytest.raises(AssertionError, match='terminal-credit authority'):
            pin_pre470_baseline_service(service, lambda: pg.connect(**dsn), round_id=row['round_id'])
    finally:
        generator.close()
