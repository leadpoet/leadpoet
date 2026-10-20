"""508 adds only a concurrent lookup index; billing authorities stay unchanged."""
from pathlib import Path
import re

import pytest

from tests.lab_arena import closed_provider_round_membership507_postgres_test as prior


ROOT = Path(__file__).parents[2]
MIGRATION = ROOT / 'scripts/508-lab-arena-openrouter-uncertain-run-index.sql'
INDEX = 'public.lab_arena_ledger_openrouter_uncertain_run_entry_idx'


def test_migration_is_one_unwrapped_concurrent_index_statement():
    text = re.sub(r'--[^\n]*', '', MIGRATION.read_text()).strip()
    assert text.count(';') == 1 and text.endswith(';')
    assert text.startswith('CREATE INDEX CONCURRENTLY ')
    assert not re.search(r'\b(BEGIN|COMMIT|IF\s+NOT\s+EXISTS|FUNCTION)\b', text)


@pytest.fixture(scope='module')
def database():
    yield from prior.database.__wrapped__()


def test_concurrent_index_shape_preserves_real_mixed_provider_selection(database):
    router, rid, _, _, candidate = prior.router._call(database, '508router')
    deepline, deepline_rid, deepline_candidate = prior.prior._closed_call(database, '508deepline')
    connection = database[0].connect(**database[1])
    connection.autocommit = True  # CONCURRENTLY must not run inside a transaction.
    try:
        with connection.cursor() as cur:
            before_functions, before_rows = prior.catalog(cur), prior.rows(cur)
            cases = [
                {}, {'rid': rid}, {'rid': deepline_rid},
                {'cursor': candidate['uncertain_entry_id']},
                {'cursor': deepline_candidate['uncertain_entry_id']},
                {'cursor': 9223372036854775807}, {'cursor': None},
                {'rid': 'missing-round'}, {'mode': 'shadow'},
                {'network': 'test'}, {'netuid': 1},
            ]
            before = [prior.prior._equivalent(cur, **args) for args in cases]
            assert before[0][1]['status'] == 'ok'
            assert before[1][1]['provider'] == 'openrouter'
            assert before[2][1]['provider'] == 'deepline'
            cur.execute(MIGRATION.read_text())
            assert [prior.prior._equivalent(cur, **args) for args in cases] == before
            assert prior.catalog(cur) == before_functions
            assert prior.rows(cur) == before_rows
            cur.execute(
                'SELECT pg_catalog.pg_get_indexdef(i.indexrelid), '
                'i.indisvalid,i.indisready,i.indisunique,am.amname '
                'FROM pg_catalog.pg_index i '
                'JOIN pg_catalog.pg_class c ON c.oid=i.indexrelid '
                'JOIN pg_catalog.pg_am am ON am.oid=c.relam '
                'WHERE i.indexrelid=%s::regclass', (INDEX,),
            )
            definition, valid, ready, unique, method = cur.fetchone()
            assert definition == (
                'CREATE INDEX lab_arena_ledger_openrouter_uncertain_run_entry_idx '
                'ON public.lab_arena_ledger USING btree (run_id, entry_id) '
                "WHERE ((provider = 'openrouter'::text) AND (entry_kind = 'uncertain'::text))"
            )
            assert (valid, ready, unique, method) == (True, True, False, 'btree')
            # A name collision must fail instead of hiding a pre-existing index.
            with pytest.raises(database[0].errors.DuplicateTable):
                cur.execute(MIGRATION.read_text())
            assert prior.catalog(cur) == before_functions
            assert prior.rows(cur) == before_rows
    finally:
        connection.close()
        router.close()
        deepline.close()
