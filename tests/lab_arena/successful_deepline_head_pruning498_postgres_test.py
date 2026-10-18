"""V2 latest-head pruning on disposable PostgreSQL; no provider calls."""

from pathlib import Path

import pytest

from tests.lab_arena import deepline_candidate_head_scope483_postgres_test as prior
from tests.lab_arena.closed_scoring_billing_postgres_test import _closed_call

ROOT = Path(__file__).parents[2]
SQL = ROOT / 'scripts/498-lab-arena-successful-deepline-head-pruning.sql'
V2 = 'public.lab_arena_list_deepline_cost_reconciliations_v2'
SIGNATURE = V2 + '(text,text,bigint,integer,boolean)'
OLD = 'public.test_before_lab_arena_list_deepline_cost_reconciliations_v2'
database = prior.database


@pytest.fixture(scope='module')
def migrated(database):
    with database[0].connect(**database[1]) as conn, conn.cursor() as cur:
        cur.execute('SELECT pg_get_functiondef(%s::regprocedure), '
                    'md5(pg_get_functiondef(%s::regprocedure))', (SIGNATURE, SIGNATURE))
        definition, fingerprint = cur.fetchone()
        assert fingerprint == '82531576cffccfea2937eb215d47c89a'
        cur.execute(definition.replace('FUNCTION ' + V2 + '(', 'FUNCTION ' + OLD + '(', 1))
        before = prior.prior._catalog(cur)
        rows = prior._rows(cur)
        cur.execute(SQL.read_text())
        after = prior.prior._catalog(cur)
        assert before[1] == after[1] and len(before[0]) == len(after[0])
        for old, new in zip(before[0], after[0]):
            if old[1] == 'lab_arena_list_deepline_cost_reconciliations_v2':
                assert old[:2] == new[:2] and old[3:] == new[3:]
                assert old[2] != new[2]
            else:
                assert old == new
        assert prior._rows(cur) == rows
        cur.execute('SELECT md5(pg_get_functiondef(%s::regprocedure))', (SIGNATURE,))
        assert cur.fetchone() == ('ea4cd7bc3fd18d3861c6a05addfcb69e',)
        cur.execute(SQL.read_text())
        assert prior.prior._catalog(cur) == after and prior._rows(cur) == rows
        cur.execute('SELECT pg_get_functiondef(%s::regprocedure)', (SIGNATURE,))
        changed = cur.fetchone()[0]
        assert changed.count('early_later.call_identity = early.call_identity') == 1
        assert changed.count('later.call_identity = uncertainty.call_identity') == 1
        assert 'p_successful_execute_only IS TRUE' in changed
    return True


def _pair(cur, rid, run_id='', after=0, limit=20, successful=True):
    cur.execute('SELECT ' + OLD + '(%s,%s,%s,%s,%s), ' + V2 + '(%s,%s,%s,%s,%s)',
                (rid, run_id, after, limit, successful,
                 rid, run_id, after, limit, successful))
    old, new = cur.fetchone()
    assert old == new
    return new['items']


def test_full_json_parity_for_policy_status_run_flag_and_paging(database, migrated):
    store, rid, candidate = _closed_call(database, 'head498matrix')
    try:
        with database[0].connect(**database[1]) as conn, conn.cursor() as cur:
            cur.execute('SET LOCAL session_replication_role=replica')
            prior._duplicate_calls(cur, candidate, 25, start=498000)
            cur.execute("SELECT entry_id FROM public.lab_arena_ledger WHERE round_id=%s "
                        "AND entry_kind='uncertain' ORDER BY entry_id", (rid,))
            ids = [row[0] for row in cur.fetchall()]
            assert len(ids) == 26
            cur.execute("UPDATE public.lab_arena_ledger SET entry_doc="
                        "jsonb_set(jsonb_set(entry_doc,'{call,reason}',"
                        "to_jsonb('missing_provider_cost'::text)),"
                        "'{call,call_succeeded}','true'::jsonb) "
                        "WHERE round_id=%s AND entry_kind='uncertain'", (rid,))
            cur.execute("INSERT INTO public.lab_arena_ledger "
                        "(entry_kind,miner_hotkey,round_id,submission_id,run_id,stage,"
                        "call_identity,provider,operation_id,funding_source,amount_microusd,entry_doc) "
                        "SELECT 'settlement',miner_hotkey,round_id,submission_id,run_id,stage,"
                        "call_identity,provider,operation_id,funding_source,0,'{}'::jsonb "
                        "FROM public.lab_arena_ledger WHERE entry_id=%s", (ids[0],))
            cur.execute("UPDATE public.lab_arena_ledger SET entry_doc=jsonb_set("
                        "entry_doc,'{call,credential_fingerprint}',to_jsonb(%s::text)) "
                        "WHERE entry_id=%s", ('sha256:' + 'b' * 64, ids[1]))
            cur.execute("UPDATE public.lab_arena_ledger SET entry_doc=jsonb_set("
                        "entry_doc,'{call,call_succeeded}','false'::jsonb) "
                        "WHERE entry_id=%s", (ids[2],))
            for status in ('stage1_scoring', 'published', 'cancelled', 'open'):
                for policy in ('successful_calls_v1', 'successful_calls_per_icp_v1',
                               'legacy', None):
                    cur.execute("UPDATE public.lab_arena_rounds SET status=%s,"
                                "configuration_doc=configuration_doc || "
                                "jsonb_build_object('sourcing_cost_eligibility_policy',%s::text) "
                                "WHERE round_id=%s", (status, policy, rid))
                    for kind in ('execute', 'score'):
                        cur.execute('UPDATE public.lab_arena_runs SET kind=%s WHERE run_id=%s',
                                    (kind, candidate['run_id']))
                        for flag in (True, False, None):
                            items = _pair(cur, rid, successful=flag)
                            assert ids[0] not in {item['uncertain_entry_id'] for item in items}
            cur.execute("UPDATE public.lab_arena_rounds SET status='cancelled' "
                        "WHERE round_id=%s", (rid,))
            cur.execute("UPDATE public.lab_arena_runs SET kind='execute' WHERE run_id=%s",
                        (candidate['run_id'],))
            successful_ids = {item['uncertain_entry_id'] for item in
                              _pair(cur, rid, successful=True)}
            ordinary_ids = {item['uncertain_entry_id'] for item in
                            _pair(cur, rid, successful=False)}
            assert ids[1] not in ordinary_ids  # Damaged binding.
            assert ids[2] not in successful_ids and ids[2] in ordinary_ids
            assert ids[3] in successful_ids and ids[3] in ordinary_ids
            for run_id in ('', candidate['run_id'], 'missing-run'):
                for after in (0, ids[12], 9223372036854775807):
                    for limit in (0, 1, 20):
                        for flag in (True, False, None):
                            items = _pair(cur, rid, run_id, after, limit, flag)
                            if run_id == 'missing-run' or limit == 0 or flag is None:
                                assert items == []
            assert _pair(cur, 'arena-missing') == []
            conn.rollback()
    finally:
        store.close()


def _counted(cur, rid, *, before=False):
    cur.execute('ALTER SEQUENCE public.test483_helper_calls RESTART WITH 1')
    cur.execute('SELECT ' + (OLD if before else V2) + '(%s,\'\',0,20,true)', (rid,))
    result = cur.fetchone()[0]
    cur.execute('SELECT CASE WHEN is_called THEN last_value ELSE 0 END '
                'FROM public.test483_helper_calls')
    return result, cur.fetchone()[0]


def test_superseded_published_execute_heads_skip_full_helpers(database, migrated,
                                                               record_property):
    store, rid, candidate = _closed_call(database, 'head498count')
    try:
        with database[0].connect(**database[1]) as conn, conn.cursor() as cur:
            cur.execute('SET LOCAL session_replication_role=replica')
            prior._duplicate_calls(cur, candidate, 63, start=499000)
            cur.execute("UPDATE public.lab_arena_ledger SET entry_doc="
                        "jsonb_set(jsonb_set(entry_doc,'{call,reason}',"
                        "to_jsonb('missing_provider_cost'::text)),"
                        "'{call,call_succeeded}','true'::jsonb) "
                        "WHERE round_id=%s AND entry_kind='uncertain'", (rid,))
            cur.execute("UPDATE public.lab_arena_runs SET kind='execute' WHERE run_id=%s",
                        (candidate['run_id'],))
            cur.execute("UPDATE public.lab_arena_rounds SET status='cancelled' WHERE round_id=%s",
                        (rid,))
            prior._helper_counter(cur)
            old, old_calls = _counted(cur, rid, before=True)
            new, new_calls = _counted(cur, rid)
            assert old == new and len(new['items']) == 20 and old_calls == new_calls
            cur.execute("UPDATE public.lab_arena_rounds SET status='published' WHERE round_id=%s",
                        (rid,))
            cur.execute("INSERT INTO public.lab_arena_ledger "
                        "(entry_kind,miner_hotkey,round_id,submission_id,run_id,stage,"
                        "call_identity,provider,operation_id,funding_source,amount_microusd,entry_doc) "
                        "SELECT 'settlement',miner_hotkey,round_id,submission_id,run_id,stage,"
                        "call_identity,provider,operation_id,funding_source,0,'{}'::jsonb "
                        "FROM public.lab_arena_ledger WHERE round_id=%s "
                        "AND entry_kind='uncertain'", (rid,))
            old, wasted = _counted(cur, rid, before=True)
            new, remaining = _counted(cur, rid)
            assert old == new == {'status': 'ok', 'items': []}
            assert wasted > 0 and remaining == 0
            record_property('superseded_v2_helper_calls_before', wasted)
            record_property('superseded_v2_helper_calls_after', remaining)
            conn.rollback()
    finally:
        store.close()


@pytest.mark.parametrize('applied', [False, True])
@pytest.mark.parametrize('mutation', [
    'ALTER FUNCTION ' + SIGNATURE + " SET statement_timeout='1s'",
    'ALTER FUNCTION ' + SIGNATURE + ' OWNER TO postgres',
    'GRANT EXECUTE ON FUNCTION ' + SIGNATURE + ' TO anon',
    "CREATE OR REPLACE FUNCTION " + V2 + "(p_round_id text,p_run_id text,"
    "p_after_entry_id bigint,p_limit integer,p_successful_execute_only boolean) "
    "RETURNS jsonb LANGUAGE sql STABLE SECURITY DEFINER "
    "SET search_path=pg_catalog, public AS 'SELECT null::jsonb'",
])
def test_migration_fails_closed_on_changed_function_or_security(database, migrated,
                                                               mutation, applied):
    with database[0].connect(**database[1]) as conn:
        with conn.cursor() as cur:
            if not applied:
                cur.execute('SELECT pg_get_functiondef(%s::regprocedure)',
                            (OLD + '(text,text,bigint,integer,boolean)',))
                definition = cur.fetchone()[0]
                cur.execute(definition.replace('FUNCTION ' + OLD + '(',
                                               'FUNCTION ' + V2 + '(', 1))
            cur.execute(mutation)
            with pytest.raises(Exception, match='(security shape changed|preimage changed)'):
                cur.execute(SQL.read_text())
        conn.rollback()
