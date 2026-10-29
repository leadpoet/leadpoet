"""Historical recovery boundaries cannot disable current service authority."""
from types import SimpleNamespace

import pytest

from tests.lab_arena.historical_service import (pin_pre470_baseline_service, pin_pre484_credit_terminal, seed_pre428_initial_reward, pin_sep20_publication_service, pin_sep20_round_reads)
from tests.lab_arena.lab_arena_pg_harness import (
    CURRENT_SERVICE_MIGRATIONS, current_service_database,
    database_with_lab_arena_migration,
)


def test_pre470_boundary_rejects_current_database():
    generator = current_service_database()
    pg, dsn = next(generator)
    try:
        with pytest.raises(AssertionError, match='cannot replace current authority'):
            pin_sep20_publication_service(SimpleNamespace(), lambda: pg.connect(**dsn), round_id='arena-2026-09-20')
        with pytest.raises(AssertionError, match='actual historical schema'):
            pin_sep20_round_reads(SimpleNamespace(), lambda: pg.connect(**dsn))
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


def test_sep20_algorithms_match_original_source_ast():
    """Indentation and local names aside, preserve all four original algorithms."""
    import ast
    import hashlib
    from pathlib import Path
    from tests.lab_arena import historical_service
    expected = {'build_scoring_plan': '1c3c3c97744e4b81dba1301b1743cc974507655aa22cf020b8384ddd85811a47', 'commit_scoring_plan': '1f72bf17b2641b946e4582267f74d568cde6477eb153e7fca8d29f21f24e3972', '_score_entries_from_runs': '00a019835ffd37ba76f5749e33905ac707a494a871cf681f7dcf6f95a4eaeadf', 'publish': '81c9a6b8e454b2538b8216d6d2677e36077f67afbdad87ea54ffdace15c78db3'}
    tree = ast.parse(Path(historical_service.__file__).read_text())
    for name, digest in expected.items():
        local_name = '_sep20_build_scoring_plan' if name == 'build_scoring_plan' else name
        node = next(x for x in ast.walk(tree) if isinstance(x, ast.FunctionDef) and x.name == local_name)
        node.name = name
        if name == 'commit_scoring_plan':
            for call in ast.walk(node):
                if isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id == '_sep20_build_scoring_plan':
                    call.func = ast.Attribute(value=ast.Name(id='scoring', ctx=ast.Load()), attr='build_scoring_plan', ctx=ast.Load())
        assert hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest() == digest


def test_sep20_boundary_rejects_other_round_profile_and_configuration_drift():
    from tests.lab_arena import sep20_authority_preserving_rejudge332_postgres_test as old
    from lab_arena import contracts
    generator = old.database.__wrapped__()
    pg, dsn = next(generator)
    configuration = {'mode': 'live', 'sourcing_cost_eligibility_policy': contracts.PER_ICP_SUCCESSFUL_CALLS_COST_POLICY}
    row = {'round_id': old.ROUND, 'status': 'stage1', 'configuration_doc': configuration}
    service = SimpleNamespace(_store=SimpleNamespace(get_round=lambda _: row), _round=lambda _: row)
    try:
        with pytest.raises(AssertionError):
            pin_sep20_publication_service(service, lambda: pg.connect(**dsn), round_id='arena-2026-09-21')
        row['configuration_doc'] = {**configuration, 'stage_1_icp_count': 5, 'stage_2_icp_count': 5}
        with pytest.raises(AssertionError):
            pin_sep20_publication_service(service, lambda: pg.connect(**dsn), round_id=old.ROUND)
        row['configuration_doc'] = configuration
        pin_sep20_publication_service(service, lambda: pg.connect(**dsn), round_id=old.ROUND)
        assert service.publish(old.ROUND) == {'status': 'stale', 'round_status': 'stage1'}
        row['configuration_doc'] = {**configuration, 'changed': True}
        with pytest.raises(AssertionError):
            service.publish(old.ROUND)
        row['configuration_doc'] = configuration
        row['round_id'] = 'arena-2026-09-21'
        with pytest.raises(AssertionError):
            service.publish(old.ROUND)
    finally:
        generator.close()
