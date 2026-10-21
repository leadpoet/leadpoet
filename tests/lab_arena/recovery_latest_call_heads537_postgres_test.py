"""Recovery keeps exact drain authority while pruning historical call heads."""
import re
from pathlib import Path

import pytest
from tests.lab_arena import evaluation_recovery536_postgres_test as prior

SQL = Path(__file__).parents[2] / 'scripts/537-lab-arena-recovery-latest-call-heads.sql'


@pytest.fixture(scope='module')
def database():
    gen = prior.database.__wrapped__()
    db = next(gen)
    try:
        with db[0].connect(**db[1]) as conn:
            with conn.cursor() as cur:
                # The disposable database has no concurrent application traffic.
                cur.execute((SQL.parent / '532-lab-arena-round-call-head-index.sql').read_text().replace('CREATE INDEX CONCURRENTLY', 'CREATE INDEX'))
            with conn.cursor() as cur:
                cur.execute('SELECT to_jsonb(p)-\'prosrc\' FROM pg_proc p WHERE p.oid=%s::regprocedure', (prior.SIG,))
                before = cur.fetchone()[0]
                cur.execute(SQL.read_text())
                cur.execute(SQL.read_text())
                cur.execute('SELECT to_jsonb(p)-\'prosrc\' FROM pg_proc p WHERE p.oid=%s::regprocedure', (prior.SIG,))
                assert cur.fetchone()[0] == before
        yield db
    finally:
        gen.close()


def test_recovery_and_unresolved_provider_guard_still_use_real_rpc(database):
    prior.test_recovery_preserves_source_history_fees_and_queues_ordinary_attempts(database)
    prior.test_unsafe_recovery_is_atomic(database, 'open_provider')


@pytest.mark.parametrize('rows,expected', [
    ([], False),
    ([('target', 'a', 1, 'reservation')], True),
    ([('target', 'a', 1, 'reservation'), ('target', 'a', 2, 'dispatch')], True),
    ([('target', 'a', 1, 'reservation'), ('target', 'a', 2, 'settlement')], False),
    ([('target', 'a', 1, 'reservation'), ('target', 'a', 2, 'uncertain')], False),
    ([('target', 'a', 1, 'reservation'), ('other', 'a', 2, 'settlement')], False),
    ([('other', 'a', 1, 'settlement'), ('target', 'a', 2, 'reservation')], True),
    ([('target', 'a', 1, 'reservation'), ('other', 'b', 2, 'settlement')], True),
])
def test_latest_heads_preserve_old_predicate_including_global_later_rows(database, rows, expected):
    text = SQL.read_text()
    old = re.search(r'\$old537\$(.*?)\$old537\$', text, re.S)[1]
    new = re.search(r'\$new537\$(.*?)\$new537\$', text, re.S)[1]
    with database[0].connect(**database[1]) as conn, conn.cursor() as cur:
        cur.execute('CREATE TEMP TABLE ledger537(round_id text,call_identity text,entry_id bigint,entry_kind text)')
        if rows:
            cur.executemany('INSERT INTO ledger537 VALUES(%s,%s,%s,%s)', rows)
        for predicate in (old, new):
            predicate = predicate.strip()[3:].replace('public.lab_arena_ledger', 'ledger537').replace('p_round_id', "'target'")
            cur.execute('SELECT ' + predicate)
            assert cur.fetchone() == (expected,)


@pytest.mark.parametrize('fault', ['grant', 'source', 'index'])
def test_patch_rejects_catalog_drift_without_changing_data(database, fault):
    with database[0].connect(**database[1]) as conn, conn.cursor() as cur:
        if fault == 'grant':
            cur.execute('GRANT EXECUTE ON FUNCTION ' + prior.SIG + ' TO anon')
        elif fault == 'index':
            cur.execute('ALTER INDEX public.lab_arena_ledger_round_call_head_idx RENAME TO drifted537')
        else:
            cur.execute('SELECT pg_get_functiondef(%s::regprocedure)', (prior.SIG,))
            cur.execute(cur.fetchone()[0].replace('recovery input invalid', 'unreviewed recovery input'))
        conn.commit()
        try:
            with pytest.raises(database[0].Error):
                cur.execute(SQL.read_text())
        finally:
            conn.rollback()
            if fault == 'grant':
                cur.execute('REVOKE EXECUTE ON FUNCTION ' + prior.SIG + ' FROM anon')
            elif fault == 'index':
                cur.execute('ALTER INDEX public.drifted537 RENAME TO lab_arena_ledger_round_call_head_idx')
            else:
                # Restore only this disposable test's own drifted definition.
                cur.execute('SELECT pg_get_functiondef(%s::regprocedure)', (prior.SIG,))
                cur.execute(cur.fetchone()[0].replace('unreviewed recovery input', 'recovery input invalid'))
