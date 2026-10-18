"""V3 head pruning and exact upgrade controls on disposable PostgreSQL."""

from pathlib import Path

import pytest

from tests.lab_arena import deepline_candidate_head_scope483_postgres_test as prior
from tests.lab_arena.closed_scoring_billing_postgres_test import _closed_call

ROOT = Path(__file__).parents[2]
SQL = ROOT / 'scripts/497-lab-arena-score-deepline-head-pruning.sql'
V3 = 'public.lab_arena_list_deepline_cost_reconciliations_v3(text,text,bigint,integer)'
OLD = 'public.test_before_lab_arena_list_deepline_cost_reconciliations_v3'
database = prior.database


@pytest.fixture(scope='module')
def migrated(database):
    with database[0].connect(**database[1]) as conn, conn.cursor() as cur:
        cur.execute('SELECT pg_get_functiondef(%s::regprocedure)', (V3,))
        definition = cur.fetchone()[0]
        cur.execute(definition.replace(
            'FUNCTION public.lab_arena_list_deepline_cost_reconciliations_v3(',
            'FUNCTION ' + OLD + '(', 1))
        before = prior.prior._catalog(cur)
        rows = prior._rows(cur)
        cur.execute(SQL.read_text())
        after = prior.prior._catalog(cur)
        assert before[1] == after[1] and len(before[0]) == len(after[0])
        for old, new in zip(before[0], after[0]):
            if old[1] == 'lab_arena_list_deepline_cost_reconciliations_v3':
                assert old[:2] == new[:2] and old[3:] == new[3:]
                assert old[2] != new[2]
            else:
                assert old == new
        assert prior._rows(cur) == rows
        cur.execute(SQL.read_text())
        assert prior.prior._catalog(cur) == after and prior._rows(cur) == rows
        cur.execute('SELECT pg_get_functiondef(%s::regprocedure)', (V3,))
        changed = cur.fetchone()[0]
        assert changed.count('early_later.call_identity = uncertainty.call_identity') == 1
        assert changed.count('later.call_identity = uncertainty.call_identity') == 2
        assert 'AND scope.kind = \'score\'' in changed
    return True


def _pair(cur, rid, run_id='', after=0, limit=20):
    cur.execute('SELECT ' + OLD + '(%s,%s,%s,%s), '
                'public.lab_arena_list_deepline_cost_reconciliations_v3(%s,%s,%s,%s)',
                (rid, run_id, after, limit, rid, run_id, after, limit))
    old, new = cur.fetchone()
    assert old == new
    return new['items']


def test_existing_heads_parity_and_full_eligibility_matrix(database, migrated):
    store, rid, candidate = _closed_call(database, 'head497matrix')
    try:
        with database[0].connect(**database[1]) as conn, conn.cursor() as cur:
            cur.execute('SET LOCAL session_replication_role=replica')
            prior._duplicate_calls(cur, candidate, 25, start=497000)
            cur.execute("SELECT entry_id FROM public.lab_arena_ledger WHERE round_id=%s "
                        "AND entry_kind='uncertain' ORDER BY entry_id", (rid,))
            ids = [row[0] for row in cur.fetchall()]
            assert len(ids) == 26
            # One superseded head, one malformed binding, and one failed
            # worker-call marker. The old policy remains the eligibility oracle.
            cur.execute("INSERT INTO public.lab_arena_ledger "
                        "(entry_kind,miner_hotkey,round_id,submission_id,run_id,stage,call_identity,"
                        "provider,operation_id,funding_source,amount_microusd,entry_doc) "
                        "SELECT 'settlement',miner_hotkey,round_id,submission_id,run_id,stage,"
                        "call_identity,provider,operation_id,funding_source,0,'{}'::jsonb "
                        "FROM public.lab_arena_ledger WHERE entry_id=%s", (ids[0],))
            cur.execute("UPDATE public.lab_arena_ledger SET entry_doc=jsonb_set("
                        "entry_doc,'{call,credential_fingerprint}',to_jsonb(%s::text)) "
                        "WHERE entry_id=%s", ('sha256:' + 'b' * 64, ids[1]))
            cur.execute("UPDATE public.lab_arena_ledger SET entry_doc=jsonb_set("
                        "jsonb_set(entry_doc,'{call,call_succeeded}','false'::jsonb),"
                        "'{call,failure_stage}',to_jsonb('provider'::text)) "
                        "WHERE entry_id=%s", (ids[2],))
            cur.execute("UPDATE public.lab_arena_rounds SET status='stage1_scoring' "
                        "WHERE round_id=%s", (rid,))
            active_ids = {item['uncertain_entry_id'] for item in _pair(cur, rid)}
            assert ids[0] not in active_ids  # Superseded by settlement.
            assert ids[1] not in active_ids  # Malformed credential binding.
            assert ids[2] in active_ids and ids[3] in active_ids
            cur.execute("UPDATE public.lab_arena_rounds SET status='published',"
                        "configuration_doc=configuration_doc || "
                        "jsonb_build_object('sourcing_cost_eligibility_policy','successful_calls_v1') "
                        "WHERE round_id=%s", (rid,))
            published_ids = {item['uncertain_entry_id'] for item in _pair(cur, rid)}
            assert ids[2] in published_ids and ids[3] in published_ids
            for status in ('stage1_scoring', 'published', 'cancelled', 'open'):
                for policy in ('successful_calls_v1', 'legacy'):
                    cur.execute("UPDATE public.lab_arena_rounds SET status=%s,"
                                "configuration_doc=configuration_doc || "
                                "jsonb_build_object('sourcing_cost_eligibility_policy',%s::text) "
                                "WHERE round_id=%s", (status, policy, rid))
                    for run_id in ('', candidate['run_id'], 'missing-run'):
                        for after in (0, ids[10], 9223372036854775807):
                            for limit in (0, 1, 20):
                                items = _pair(cur, rid, run_id, after, limit)
                                if status == 'open' or run_id == 'missing-run' or limit == 0:
                                    assert items == []
                                if (status == 'published' and policy == 'legacy'
                                        and run_id == '' and after == 0):
                                    assert items == []
            assert _pair(cur, 'arena-missing') == []
            conn.rollback()  # Remove only disposable test fixtures.
    finally:
        store.close()


def test_superseded_score_heads_skip_full_helpers(database, migrated, record_property):
    store, rid, candidate = _closed_call(database, 'head497count')
    try:
        with database[0].connect(**database[1]) as conn, conn.cursor() as cur:
            cur.execute('SET LOCAL session_replication_role=replica')
            prior._duplicate_calls(cur, candidate, 63, start=498000)
            cur.execute("UPDATE public.lab_arena_rounds SET status='published' WHERE round_id=%s", (rid,))
            prior._helper_counter(cur)
            old, old_calls = _counted_pair(cur, rid, before=True)
            new, new_calls = _counted_pair(cur, rid)
            assert old == new and len(new['items']) == 20
            assert old_calls > 0 and new_calls > 0
            cur.execute("INSERT INTO public.lab_arena_ledger "
                        "(entry_kind,miner_hotkey,round_id,submission_id,run_id,stage,call_identity,"
                        "provider,operation_id,funding_source,amount_microusd,entry_doc) "
                        "SELECT 'settlement',miner_hotkey,round_id,submission_id,run_id,stage,"
                        "call_identity,provider,operation_id,funding_source,0,'{}'::jsonb "
                        "FROM public.lab_arena_ledger WHERE round_id=%s AND entry_kind='uncertain'", (rid,))
            old, wasted = _counted_pair(cur, rid, before=True)
            new, remaining = _counted_pair(cur, rid)
            assert old == new == {'status': 'ok', 'items': []}
            assert wasted == 64 and remaining == 0
            record_property('superseded_v3_helper_calls_before', wasted)
            record_property('superseded_v3_helper_calls_after', remaining)
            conn.rollback()
    finally:
        store.close()


def _counted_pair(cur, rid, *, before=False):
    cur.execute('ALTER SEQUENCE public.test483_helper_calls RESTART WITH 1')
    name = OLD if before else 'public.lab_arena_list_deepline_cost_reconciliations_v3'
    cur.execute('SELECT ' + name + '(%s,\'\',0,20)', (rid,))
    result = cur.fetchone()[0]
    cur.execute('SELECT CASE WHEN is_called THEN last_value ELSE 0 END '
                'FROM public.test483_helper_calls')
    return result, cur.fetchone()[0]


@pytest.mark.parametrize('applied', [False, True])
@pytest.mark.parametrize('mutation', [
    'ALTER FUNCTION ' + V3 + " SET statement_timeout='1s'",
    'ALTER FUNCTION ' + V3 + ' OWNER TO postgres',
    'GRANT EXECUTE ON FUNCTION ' + V3 + ' TO anon',
    "CREATE OR REPLACE FUNCTION public.lab_arena_list_deepline_cost_reconciliations_v3("
    "p_round_id text,p_run_id text,p_after_entry_id bigint,p_limit integer) RETURNS jsonb "
    "LANGUAGE sql STABLE SECURITY DEFINER SET search_path=pg_catalog, public AS 'SELECT null::jsonb'",
])
def test_migration_fails_closed_on_changed_function_or_security(database, migrated,
                                                               mutation, applied):
    with database[0].connect(**database[1]) as conn:
        with conn.cursor() as cur:
            if not applied:
                cur.execute('SELECT pg_get_functiondef(%s::regprocedure)', (OLD + '(text,text,bigint,integer)',))
                definition = cur.fetchone()[0]
                cur.execute(definition.replace('FUNCTION ' + OLD + '(',
                    'FUNCTION public.lab_arena_list_deepline_cost_reconciliations_v3(', 1))
            cur.execute(mutation)
            with pytest.raises(Exception, match='(security shape changed|preimage changed)'):
                cur.execute(SQL.read_text())
        conn.rollback()
