"""Score billing lane is a guarded subset of the current V1 candidates."""
from pathlib import Path

import pytest

from lab_arena.store import hash_lease_token
from tests.lab_arena import adaptation_cost_reconciliation453_postgres_test as adaptation_fixture
from tests.lab_arena import deepline_cost_priority_postgres_test as priority
from tests.lab_arena.deepline_budget_only_exact_recovery_postgres_test import run
from tests.lab_arena.deepline_delayed_cost_reconciliation_postgres_test import (
    _open_scoring, _store,
)
from tests.lab_arena.test_lab_arena_migration_postgres import claim, complete

database = adaptation_fixture.database
ROOT = Path(__file__).parents[2]
SQL = ROOT / 'scripts/456-lab-arena-score-deepline-cost-priority.sql'
V3 = 'public.lab_arena_list_deepline_cost_reconciliations_v3(text,text,bigint,integer)'


@pytest.fixture(scope='module')
def migrated(database):
    adaptation_fixture.migrated.__wrapped__(database)
    psycopg, dsn = database
    with psycopg.connect(**dsn) as conn:
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute('SELECT pg_get_functiondef(%s::regprocedure), pg_get_functiondef(%s::regprocedure)',
                        (priority.V1, priority.V2))
            before = cur.fetchone()
            cur.execute(SQL.read_text())
            cur.execute(SQL.read_text())
            cur.execute('SELECT pg_get_functiondef(%s::regprocedure), pg_get_functiondef(%s::regprocedure)',
                        (priority.V1, priority.V2))
            assert cur.fetchone() == before
            cur.execute('SELECT pg_get_functiondef(%s::regprocedure)', (V3,))
            restored = cur.fetchone()[0].replace(
                'FUNCTION public.lab_arena_list_deepline_cost_reconciliations_v3(',
                'FUNCTION public.lab_arena_list_deepline_cost_reconciliations_v1(',
                1).replace("\n              AND scope.kind = 'score'", '', 1)
            assert restored == before[0]
            cur.execute('SELECT pg_get_userbyid(proowner), prosecdef, provolatile, proconfig '
                        'FROM pg_proc WHERE oid=%s::regprocedure', (V3,))
            owner, definer, volatility, settings = cur.fetchone()
            assert owner == 'lab_arena_owner' and definer and volatility == 's'
            assert 'search_path=pg_catalog, public' in settings
            for role in ('anon', 'authenticated', 'service_role', 'lab_arena_service'):
                cur.execute('SELECT has_function_privilege(%s,%s,\'EXECUTE\')', (role, V3))
                assert cur.fetchone()[0] is (role == 'lab_arena_service')
    return True


def test_score_lane_preserves_bindings_failed_calls_and_cursor(database, migrated, tmp_path):
    h, execute, execute_token, connect = run(database, tmp_path, '66')
    execute_id = priority.uncertain(h, execute, execute_token, 'execute-first', success=False)[0]
    assert complete(h.service.store, execute['run_id'], hash_lease_token(execute_token),
                    'accepted', output_ref='arena/' + h.round_id + '/execute.json')['status'] == 'accepted'
    assert h.service.store.list_deepline_cost_reconciliations(h.round_id, score_only=True) == []
    assert [row['call_identity'] for row in h.service.store.list_deepline_cost_reconciliations(h.round_id)] == [execute_id]

    store = _store(database)
    rid = 'arena-2026-09-14-scorelane456'
    runners, _ = _open_scoring(store, rid, participants=1, runners=1)
    score, token = claim(store, rid, runners[0])[:2]
    assert score['kind'] == 'score'
    good = priority.uncertain(h, score, token, 'score-success', success=True)[0]
    failed = priority.uncertain(h, score, token, 'score-failed', success=False)[0]
    adaptation = priority.uncertain(h, score, token, 'score-adaptation',
                                    reason='settle_failure', success=False)[0]
    corrupt = priority.uncertain(h, score, token, 'score-corrupt', success=True)[0]
    with connect() as conn, conn.cursor() as cur:
        cur.execute('ALTER TABLE public.lab_arena_ledger DISABLE TRIGGER lab_arena_ledger_append_only')
        cur.execute("UPDATE public.lab_arena_ledger SET entry_doc=jsonb_set("
                    "jsonb_set(entry_doc,'{call,failure_stage}','\"response_adaptation\"'::jsonb),"
                    "'{call,error_class}','\"CompatibilityResponseError\"'::jsonb) "
                    "WHERE call_identity=%s AND entry_kind='uncertain'", (adaptation,))
        cur.execute("UPDATE public.lab_arena_ledger SET entry_doc=jsonb_set("
                    "entry_doc,'{call,credential_fingerprint}',"
                    "'\"sha256:0000000000000000000000000000000000000000000000000000000000000000\"'::jsonb) "
                    "WHERE call_identity=%s AND entry_kind='uncertain'", (corrupt,))
        cur.execute('ALTER TABLE public.lab_arena_ledger ENABLE TRIGGER lab_arena_ledger_append_only')
    ordinary = store.list_deepline_cost_reconciliations(rid, limit=20)
    selected = store.list_deepline_cost_reconciliations(rid, limit=20, score_only=True)
    assert selected == ordinary
    assert {row['call_identity'] for row in selected} == {good, failed, adaptation}
    assert all(row['kind'] == 'score' for row in selected)
    first = store.list_deepline_cost_reconciliations(rid, score_only=True)[0]
    second = store.list_deepline_cost_reconciliations(
        rid, after_entry_id=first['uncertain_entry_id'], score_only=True)[0]
    third = store.list_deepline_cost_reconciliations(
        rid, after_entry_id=second['uncertain_entry_id'], score_only=True)[0]
    wrapped = store.list_deepline_cost_reconciliations(
        rid, after_entry_id=third['uncertain_entry_id'], score_only=True)[0]
    assert len({row['call_identity'] for row in (first, second, third)}) == 3
    assert wrapped['call_identity'] == first['call_identity']
    assert store.list_deepline_cost_reconciliations('arena-missing', score_only=True) == []
    store.close()
    h.service.store.close()
