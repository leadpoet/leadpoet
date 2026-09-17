"""Verbatim rerun286 SQL against the protected published production snapshot."""
import json
import os
from pathlib import Path

import pytest

from tests.lab_arena.lab_arena_pg_harness import CURRENT_SERVICE_MIGRATIONS, database_with_lab_arena_migration
from tests.lab_arena.sep17_baseline_recovery278_postgres_test import _insert_rows
from tests.lab_arena.sep17_published_rerun286_postgres_test import (
    BASELINE, ROUND, ROOT, SOURCE_285_COMMIT, SOURCE_285_SHA, SOURCE_285_SIZE,
    _future_schedule, _render_285, captured_284_terminal, restore_284_terminal,
)

SNAPSHOT = Path(os.environ.get('RERUN286_SEALED_TERMINAL', '/nonexistent-rerun286-snapshot'))
pytestmark = pytest.mark.skipif(not SNAPSHOT.is_file(), reason='protected terminal snapshot required')


@pytest.fixture
def database():
    yield from database_with_lab_arena_migration(CURRENT_SERVICE_MIGRATIONS)


def test_verbatim_committed_migration_preserves_exact_published_terminal(database):
    psycopg2, dsn = database
    snapshot = json.loads(SNAPSHOT.read_text())
    sql = (ROOT / 'scripts/286-arena-2026-09-17-published-baseline-rerun.sql').read_text()
    import re
    schedule = json.loads(re.search(r'\$forward_schedule\$(.*?)\$forward_schedule\$', sql).group(1))
    source_size = int(re.search(r'p_source_size_bytes IS DISTINCT FROM (\d+)', sql).group(1))
    source_sha = re.search(r"p_source_sha256 IS DISTINCT FROM '([0-9a-f]{64})'", sql).group(1)
    source_commit = re.search(r"p_source_commit IS DISTINCT FROM '([0-9a-f]{40})'", sql).group(1)
    connection = psycopg2.connect(**dsn)
    try:
        old = restore_284_terminal(connection)
        prior_schedule = _future_schedule(old['proof']['round']['schedule'])
        with connection.cursor() as cursor:
            cursor.execute(_render_285(captured_284_terminal(), prior_schedule))
            connection.commit()
            cursor.execute('SELECT public.lab_arena_prepare_sep17_baseline_recovery285_v1(%s,%s,%s,%s,%s::jsonb)',
                           (SOURCE_285_SIZE, SOURCE_285_SHA, SOURCE_285_COMMIT,
                            '7023eca8a6518434d5010d31481c1d156aa96abb3e5565c30e033073b088c871', json.dumps(prior_schedule)))
            assert cursor.fetchone()[0]['status'] == 'prepared'
            cursor.execute('SET LOCAL session_replication_role=replica')
            for table in ('lab_arena_rounds', 'lab_arena_submissions', 'lab_arena_runs', 'lab_arena_ledger'):
                cursor.execute('ALTER TABLE public.' + table + ' DISABLE TRIGGER USER')
            for table in ('lab_arena_ledger', 'lab_arena_runs', 'lab_arena_submissions'):
                cursor.execute('DELETE FROM public.' + table + ' WHERE round_id=%s AND submission_id=%s', (ROUND, BASELINE))
            cursor.execute('DELETE FROM public.lab_arena_rounds WHERE round_id=%s', (ROUND,))
            for table, rows in (
                ('lab_arena_rounds', [snapshot['round']]),
                ('lab_arena_submissions', [snapshot['baseline']]),
                ('lab_arena_runs', snapshot['baseline_runs']),
                ('lab_arena_ledger', snapshot['baseline_ledger']),
            ):
                for start in range(0, len(rows), 250):
                    _insert_rows(cursor, table, rows[start:start+250])
            cursor.execute('SET LOCAL session_replication_role=origin')
            for table in ('lab_arena_rounds', 'lab_arena_submissions', 'lab_arena_runs', 'lab_arena_ledger'):
                cursor.execute('ALTER TABLE public.' + table + ' ENABLE TRIGGER USER')
        connection.commit()
        with connection.cursor() as cursor:
            cursor.execute(sql)
            cursor.execute(sql)
            parameters = (source_size, source_sha, source_commit,
                          '7023eca8a6518434d5010d31481c1d156aa96abb3e5565c30e033073b088c871', json.dumps(schedule))
            cursor.execute('SELECT public.lab_arena_prepare_sep17_published_rerun286_v1(%s,%s,%s,%s,%s::jsonb)', parameters)
            assert cursor.fetchone()[0]['status'] == 'prepared'
            cursor.execute('SELECT public.lab_arena_prepare_sep17_published_rerun286_v1(%s,%s,%s,%s,%s::jsonb)', parameters)
            assert cursor.fetchone()[0]['status'] == 'existing'
            cursor.execute('SELECT count(*) FROM public.lab_arena_runs WHERE round_id=%s AND submission_id=%s AND status=%s', (ROUND, BASELINE, 'pending'))
            assert cursor.fetchone()[0] == 20
            cursor.execute('SELECT reward_basis_hash,reward_basis_doc,signing_key_doc,effective_reward_epoch,reward_activated_at::text FROM public.lab_arena_rounds WHERE round_id=%s', (ROUND,))
            row = cursor.fetchone()
            assert row[:4] == tuple(snapshot['round'][key] for key in ('reward_basis_hash', 'reward_basis_doc', 'signing_key_doc', 'effective_reward_epoch'))
            cursor.execute('SELECT public.lab_arena_sep17_rerun285_archive_valid286_v1(),public.lab_arena_sep17_rerun286_nonbaseline_valid_v1()')
            assert cursor.fetchone() == (True, True)
        connection.commit()
    finally:
        connection.close()
