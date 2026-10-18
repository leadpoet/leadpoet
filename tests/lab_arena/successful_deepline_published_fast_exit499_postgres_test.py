"""V2 published execute fast exit and exact upgrade on disposable PostgreSQL."""

from pathlib import Path

import pytest

from tests.lab_arena import successful_deepline_head_pruning498_postgres_test as prior
from tests.lab_arena import deepline_candidate_head_scope483_postgres_test as heads
from tests.lab_arena.closed_scoring_billing_postgres_test import _closed_call

ROOT = Path(__file__).parents[2]
SQL = ROOT / 'scripts/499-lab-arena-successful-deepline-published-fast-exit.sql'
V3_UPGRADE = ROOT / 'scripts/497-lab-arena-score-deepline-head-pruning.sql'
V2 = prior.V2
SIGNATURE = prior.SIGNATURE
OLD = 'public.test_before_lab_arena_list_deepline_cost_reconciliations_v2_499'
HELPERS = (
    ('lab_arena__closed_score_dynamic_uncertainty_v1', '7312e9ca80018f634cd3799cd4a90093'),
    ('lab_arena__closed_host_score_success_uncertainty_v1', 'b096b9440dd66037b9f286632c5b27a9'),
    ('lab_arena__closed_miner_score_success_uncertainty_v1', '2827d938ea977794ef3427aa9d7517dc'),
)
database = prior.database
pre498 = prior.migrated


@pytest.fixture(scope='module')
def migrated(database, pre498):
    with database[0].connect(**database[1]) as conn, conn.cursor() as cur:
        # Mirror the latest-main family: V1/483, V3/497, V2/498, then V2/499.
        cur.execute(V3_UPGRADE.read_text())
        cur.execute('SELECT pg_get_functiondef(%s::regprocedure), '
                    'md5(pg_get_functiondef(%s::regprocedure))', (SIGNATURE, SIGNATURE))
        definition, fingerprint = cur.fetchone()
        assert fingerprint == 'ea4cd7bc3fd18d3861c6a05addfcb69e'
        cur.execute(definition.replace('FUNCTION ' + V2 + '(', 'FUNCTION ' + OLD + '(', 1))
        before = heads.prior._catalog(cur)
        rows = heads._rows(cur)
        cur.execute(SQL.read_text())
        after = heads.prior._catalog(cur)
        assert before[1] == after[1] and len(before[0]) == len(after[0])
        for old, new in zip(before[0], after[0]):
            if old[1] == 'lab_arena_list_deepline_cost_reconciliations_v2':
                assert old[:2] == new[:2] and old[3:] == new[3:]
                assert old[2] != new[2]
            else:
                assert old == new
        assert heads._rows(cur) == rows
        cur.execute('SELECT md5(pg_get_functiondef(%s::regprocedure))', (SIGNATURE,))
        assert cur.fetchone() == ('6cc6a84e068ce445dc25769a070075ae',)
        cur.execute(SQL.read_text())
        assert heads.prior._catalog(cur) == after and heads._rows(cur) == rows
        cur.execute('SELECT pg_get_functiondef(%s::regprocedure)', (SIGNATURE,))
        changed = cur.fetchone()[0]
        assert changed.count("early_round.status <> 'published'") == 1
        assert changed.count('later.call_identity = uncertainty.call_identity') == 1
        for helper, hash_value in HELPERS:
            cur.execute('SELECT md5(pg_get_functiondef(%s::regprocedure))',
                        ('public.' + helper + '(bigint)',))
            assert cur.fetchone() == (hash_value,)
    return True


def _pair(cur, rid, run_id='', after=0, limit=20, successful=True):
    cur.execute('SELECT ' + OLD + '(%s,%s,%s,%s,%s), ' + V2 + '(%s,%s,%s,%s,%s)',
                (rid, run_id, after, limit, successful,
                 rid, run_id, after, limit, successful))
    old, new = cur.fetchone()
    assert old == new
    return new['items']


def test_full_json_parity_and_positive_active_execute_controls(database, migrated):
    store, rid, candidate = _closed_call(database, 'p499matrix')
    try:
        with database[0].connect(**database[1]) as conn, conn.cursor() as cur:
            cur.execute('SET LOCAL session_replication_role=replica')
            heads._duplicate_calls(cur, candidate, 31, start=499000)
            cur.execute("UPDATE public.lab_arena_ledger SET entry_doc="
                        "jsonb_set(jsonb_set(entry_doc,'{call,reason}',"
                        "to_jsonb('missing_provider_cost'::text)),"
                        "'{call,call_succeeded}','true'::jsonb) "
                        "WHERE round_id=%s AND entry_kind='uncertain'", (rid,))
            cur.execute("SELECT entry_id FROM public.lab_arena_ledger WHERE round_id=%s "
                        "AND entry_kind='uncertain' ORDER BY entry_id", (rid,))
            ids = [row[0] for row in cur.fetchall()]
            assert len(ids) == 32
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
            for status in ('stage2', 'published', 'cancelled', 'open'):
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
                            found = {item['uncertain_entry_id'] for item in items}
                            assert ids[0] not in found and ids[1] not in found
                            if flag is True and kind == 'execute' and status in ('stage2', 'cancelled'):
                                assert ids[3] in found and ids[2] not in found
                            if flag is True and status == 'published':
                                assert items == []
                            if flag is None or status == 'open':
                                assert items == []
            cur.execute("UPDATE public.lab_arena_rounds SET status='cancelled' "
                        "WHERE round_id=%s", (rid,))
            cur.execute("UPDATE public.lab_arena_runs SET kind='execute' WHERE run_id=%s",
                        (candidate['run_id'],))
            for run_id in ('', candidate['run_id'], 'missing-run'):
                for cursor in (0, ids[15], 9223372036854775807):
                    for limit in (0, 1, 20):
                        for flag in (True, False, None):
                            items = _pair(cur, rid, run_id, cursor, limit, flag)
                            if run_id == 'missing-run' or limit == 0 or flag is None:
                                assert items == []
            assert _pair(cur, 'arena-missing') == []
            conn.rollback()
    finally:
        store.close()


def _shared_hits(cur, function, rid):
    cur.execute('EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) SELECT ' + function +
                '(%s,\'\',0,20,true)', (rid,))
    return cur.fetchone()[0][0]['Plan'].get('Shared Hit Blocks', 0)


def test_published_large_execute_page_short_circuits_before_ledger(database, migrated,
                                                                    record_property):
    store, rid, candidate = _closed_call(database, 'p499large')
    try:
        with database[0].connect(**database[1]) as conn, conn.cursor() as cur:
            cur.execute('SET LOCAL session_replication_role=replica')
            heads._duplicate_calls(cur, candidate, 255, start=500000)
            cur.execute("UPDATE public.lab_arena_runs SET kind='execute' WHERE run_id=%s",
                        (candidate['run_id'],))
            cur.execute("UPDATE public.lab_arena_rounds SET status='published' WHERE round_id=%s",
                        (rid,))
            assert _pair(cur, rid) == []
            old_hits = _shared_hits(cur, OLD, rid)
            new_hits = _shared_hits(cur, V2, rid)
            assert old_hits > new_hits * 10, (old_hits, new_hits)
            record_property('published_v2_shared_hits_before', old_hits)
            record_property('published_v2_shared_hits_after', new_hits)
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
] + ["ALTER FUNCTION public." + name + "(bigint) SET statement_timeout='1s'"
     for name, _ in HELPERS])
def test_migration_fails_closed_on_changed_function_or_helper(database, migrated,
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
            with pytest.raises(Exception, match='(helper prerequisite changed|security shape changed|preimage changed)'):
                cur.execute(SQL.read_text())
        conn.rollback()
