"""Exact-current V1 candidate fencing on disposable PostgreSQL; no paid work."""
from pathlib import Path

import pytest

from tests.lab_arena import closed_adaptation_score_billing457_postgres_test as adaptation
from tests.lab_arena import deepline_candidate_query_postgres_test as prior
from tests.lab_arena import deepline_cost_priority_postgres_test as priority
from tests.lab_arena import score_deepline_cost_priority456_postgres_test as score_lane
from tests.lab_arena.closed_scoring_billing_postgres_test import _closed_call
from tests.lab_arena.lab_arena_pg_harness import (
    CURRENT_PROVIDER_SERVICE_MIGRATIONS, database_with_lab_arena_migration,
)

ROOT = Path(__file__).parents[2]
SQL = ROOT / 'scripts/483-lab-arena-deepline-candidate-head-scope.sql'
V1 = priority.V1
HELPERS = (
    'lab_arena__closed_score_dynamic_uncertainty_v1',
    'lab_arena__closed_host_score_success_uncertainty_v1',
    'lab_arena__closed_miner_score_success_uncertainty_v1',
    'lab_arena__closed_score_adaptation_uncertainty_v1',
)
TAIL = (
    '435-lab-arena-deepline-candidate-query.sql',
    '436-lab-arena-closed-provider-candidate-query.sql',
    '438-lab-arena-accepted-host-score-billing.sql',
    '439-lab-arena-closed-miner-success-score-billing.sql',
    '261-lab-arena-settlement-failure-deepline-cost-reconciliation.sql',
    '453-lab-arena-deepline-adaptation-cost-reconciliation.sql',
    '456-lab-arena-score-deepline-cost-priority.sql',
    '457-lab-arena-closed-adaptation-score-billing.sql',
)


def _rows(cur):
    cur.execute("SELECT tablename FROM pg_tables WHERE schemaname='public' ORDER BY tablename")
    tables = [row[0] for row in cur.fetchall()]
    result = {}
    for table in tables:
        # Names come from pg_catalog, and quote_ident prevents SQL name injection.
        cur.execute('SELECT quote_ident(%s)', (table,))
        name = cur.fetchone()[0]
        cur.execute('SELECT count(*), md5(coalesce(string_agg(doc,\'\' ORDER BY doc),\'\')) '
                    'FROM (SELECT to_jsonb(t)::text doc FROM public.' + name + ' t) s')
        result[table] = cur.fetchone()
    return result


@pytest.fixture(scope='module')
def database():
    for db in database_with_lab_arena_migration(CURRENT_PROVIDER_SERVICE_MIGRATIONS + TAIL):
        # Old rows exist before483, so transition tests also cover stored liabilities.
        store, rid, candidate = _closed_call(db, 'scope483pre')
        store.close()
        with db[0].connect(**db[1]) as conn, conn.cursor() as cur:
            for name, signature in zip(prior.NAMES, prior.SIGNATURES):
                cur.execute('SELECT pg_get_functiondef(%s::regprocedure)', (signature,))
                definition = cur.fetchone()[0]
                cur.execute(definition.replace('FUNCTION public.' + name + '(',
                    'FUNCTION public.test_before_' + name + '(', 1))
            cur.execute('SELECT md5(pg_get_functiondef(%s::regprocedure))', (V1,))
            assert cur.fetchone() == ('b1b891991482cd99c8c06a22c71178f3',)
            cur.execute("INSERT INTO public.qualification_private_icp_sets "
                "(set_id,icps,is_active) VALUES (483,'[{\"name\":\"preserved483\"}]'::jsonb,true)")
            before = prior._catalog(cur)
            rows = _rows(cur)
            cur.execute(SQL.read_text())  # Exact verbatim release artifact.
            after = prior._catalog(cur)
            assert before[1] == after[1]
            assert len(before[0]) == len(after[0])
            for old, new in zip(before[0], after[0]):
                if old[1] == prior.NAMES[0]:
                    assert old[:2] == new[:2] and old[3:] == new[3:]
                    assert old[2] != new[2]
                else:
                    assert old == new  # Includes V2/V3, settlement and helpers.
            assert _rows(cur) == rows
            cur.execute(SQL.read_text())
            assert prior._catalog(cur) == after and _rows(cur) == rows
            assert prior._list(cur, rid)[0][0]['call_identity'] == candidate['call_identity']
        yield db


def test_original_identity_binding_and_active_cursor_controls(database, tmp_path):
    prior.test_all_guards_and_cursor_json_are_preserved(database, tmp_path)


def test_saved_success_failed_uncertainty_and_settlement_controls(database, tmp_path):
    priority.test_priority_is_exact_success_and_recovered_success_with_fair_wrap(database, tmp_path)
    priority.test_raw_failure_and_cancelled_liabilities_stay_in_ordinary_lane(database, tmp_path)
    priority.test_new_list_is_service_only(database)


def _duplicate_calls(cur, candidate, count, start=483000):
    # Production-shaped three-entry call histories; only disposable fixtures may
    # bypass ledger append-only and round/run write-once triggers.
    for number in range(count):
        identity = 'sha256:' + format(number + start, '064x')
        request = 'ctx-tool-' + identity[7:39]
        cur.execute("""
            INSERT INTO public.lab_arena_ledger
              (entry_kind,miner_hotkey,round_id,submission_id,run_id,stage,
               call_identity,provider,operation_id,funding_source,amount_microusd,entry_doc)
            SELECT entry_kind,miner_hotkey,round_id,submission_id,run_id,stage,
              %s,provider,operation_id,funding_source,amount_microusd,
              CASE WHEN entry_kind='reservation'
                THEN jsonb_set(entry_doc,'{deepline_request_id}',to_jsonb(%s::text))
                WHEN entry_kind='uncertain'
                THEN jsonb_set(entry_doc,'{call,deepline_request_id}',to_jsonb(%s::text))
                ELSE entry_doc END
            FROM public.lab_arena_ledger WHERE call_identity=%s
            ORDER BY entry_id
        """, (identity, request, request, candidate['call_identity']))


def test_published_cancelled_execute_score_run_scope_and_wrapped_pages(database):
    store, rid, candidate = _closed_call(database, 'scope483matrix')
    try:
        with database[0].connect(**database[1]) as conn, conn.cursor() as cur:
            cur.execute('SET LOCAL session_replication_role=replica')
            _duplicate_calls(cur, candidate, 25)
            # A settlement suppresses one head. A later non-settlement suppresses
            # another; the original final latest-head guard remains intact.
            cur.execute("INSERT INTO public.lab_arena_ledger "
                "(entry_kind,miner_hotkey,round_id,submission_id,run_id,stage,call_identity,"
                "provider,operation_id,funding_source,amount_microusd,entry_doc) "
                "SELECT 'settlement',miner_hotkey,round_id,submission_id,run_id,stage,"
                "call_identity,provider,operation_id,funding_source,0,'{}'::jsonb "
                "FROM public.lab_arena_ledger WHERE entry_id=%s", (candidate['uncertain_entry_id'],))
            cur.execute("UPDATE public.lab_arena_ledger SET entry_id=-entry_id "
                "WHERE call_identity=%s AND entry_kind='uncertain'",
                ('sha256:' + format(483000, '064x'),))
            cur.execute("SELECT entry_id FROM public.lab_arena_ledger WHERE round_id=%s "
                "AND entry_kind='uncertain' AND entry_id>0 ORDER BY entry_id", (rid,))
            ids = [row[0] for row in cur.fetchall() if row[0] != candidate['uncertain_entry_id']]
            assert len(ids) == 24
            checks = 0
            for status in ('stage1_scoring', 'published', 'cancelled', 'open'):
                for kind in ('execute', 'score'):
                    for policy in ('successful_calls_v1', 'successful_calls_per_icp_v1', 'legacy', None):
                        cur.execute("UPDATE public.lab_arena_rounds SET status=%s,configuration_doc="
                            "configuration_doc || jsonb_build_object('sourcing_cost_eligibility_policy',%s::text) "
                            "WHERE round_id=%s", (status, policy, rid))
                        cur.execute('UPDATE public.lab_arena_runs SET kind=%s WHERE run_id=%s',
                                    (kind, candidate['run_id']))
                        eligible = status not in ('open', 'published') or (
                            status == 'published' and kind == 'score' and policy in (
                                'successful_calls_v1', 'successful_calls_per_icp_v1'))
                        for run_id in ('', None, candidate['run_id'], 'missing-run'):
                            for cursor in (0, ids[19], 9223372036854775807):
                                for limit in (1, 20):
                                    items = prior._list(cur, rid, run_id, cursor, limit)[0]
                                    expected = eligible and run_id != 'missing-run'
                                    assert len(items) == (limit if expected else 0)
                                    if expected:
                                        ordered = [n for n in ids if n > cursor] + [n for n in ids if n <= cursor]
                                        assert [x['uncertain_entry_id'] for x in items] == sorted(ordered[:limit])
                                    checks += 1
            assert checks == 768
    finally:
        store.close()


def test_real_candidate_exact_cost_get_settlement_and_frozen_publication(database, tmp_path):
    adaptation.test_closed_accepted_adaptation_exact_bill_preserves_published_result(database, tmp_path)


def test_failed_and_damaged_receipts_cannot_settle(database, tmp_path):
    adaptation.test_closed_adaptation_rejects_unrelated_and_damaged_binding(database, tmp_path)


def test_existing_score_lane_controls_use_current_schema(database, tmp_path):
    # The historical 456 fixture replays an obsolete 441/420 upgrade preimage.
    # Exercise its unchanged behavioral controls on the exact-current schema.
    score_lane.test_score_lane_preserves_bindings_failed_calls_and_cursor(
        database, True, tmp_path)


@pytest.mark.parametrize('applied', [False, True])
@pytest.mark.parametrize('mutation', [
    'ALTER FUNCTION ' + V1 + " SET statement_timeout='1s'",
    'ALTER FUNCTION ' + V1 + ' OWNER TO postgres',
    'GRANT EXECUTE ON FUNCTION ' + V1 + ' TO anon',
    "CREATE OR REPLACE FUNCTION public.lab_arena_list_deepline_cost_reconciliations_v1("
    "p_round_id text,p_run_id text,p_after_entry_id bigint,p_limit integer) RETURNS jsonb "
    "LANGUAGE sql STABLE SECURITY DEFINER SET search_path=pg_catalog, public AS 'SELECT null::jsonb'",
] + ["CREATE OR REPLACE FUNCTION public." + name +
     "(p_uncertain_entry_id bigint) RETURNS boolean LANGUAGE sql STABLE SECURITY DEFINER "
     "SET search_path=pg_catalog, public AS 'SELECT true'" for name in HELPERS])
def test_migration_fails_closed_for_changed_preimage_security_or_helper(database, mutation, applied):
    with database[0].connect(**database[1]) as conn:
        with conn.cursor() as cur:
            if not applied:
                cur.execute('SELECT pg_get_functiondef(\'public.test_before_' +
                    prior.NAMES[0] + '(text,text,bigint,integer)\'::regprocedure)')
                cur.execute(cur.fetchone()[0].replace('FUNCTION public.test_before_' + prior.NAMES[0],
                                                     'FUNCTION public.' + prior.NAMES[0], 1))
            cur.execute(mutation)
            with pytest.raises(Exception, match='(security shape changed|preimage changed|helper prerequisite changed)'):
                cur.execute(SQL.read_text())
        conn.rollback()


def _helper_counter(cur):
    """Count the unchanged full helper path only inside the disposable DB."""
    cur.execute('CREATE SEQUENCE public.test483_helper_calls')
    cur.execute('GRANT USAGE, SELECT ON SEQUENCE public.test483_helper_calls TO lab_arena_owner')
    for number, name in enumerate(HELPERS):
        cur.execute('SELECT pg_get_functiondef(%s::regprocedure)', ('public.' + name + '(bigint)',))
        full = cur.fetchone()[0].replace('FUNCTION public.' + name + '(',
            'FUNCTION public.test483_full_' + str(number) + '(', 1)
        cur.execute(full)
        cur.execute('ALTER FUNCTION public.test483_full_' + str(number) + '(bigint) OWNER TO lab_arena_owner')
        cur.execute('GRANT EXECUTE ON FUNCTION public.test483_full_' + str(number) + '(bigint) TO lab_arena_owner')
        cur.execute('CREATE OR REPLACE FUNCTION public.' + name +
            '(p_uncertain_entry_id bigint) RETURNS boolean LANGUAGE plpgsql STABLE SECURITY DEFINER '
            'SET search_path=pg_catalog, public AS $counter$ BEGIN '
            "PERFORM pg_catalog.nextval('public.test483_helper_calls'); "
            'RETURN public.test483_full_' + str(number) + '(p_uncertain_entry_id); END $counter$')


def _counted_list(cur, rid, *, before=False, run_id=''):
    cur.execute('ALTER SEQUENCE public.test483_helper_calls RESTART WITH 1')
    cur.execute('SELECT public.' + ('test_before_' if before else '') + prior.NAMES[0] +
        '(%s,%s,0,20)', (rid, run_id))
    result = cur.fetchone()[0]
    cur.execute('SELECT CASE WHEN is_called THEN last_value ELSE 0 END FROM public.test483_helper_calls')
    return result, cur.fetchone()[0]


def test_production_shape_dead_candidates_never_enter_full_helpers(database, record_property):
    store, rid, candidate = _closed_call(database, 'scope483count')
    try:
        with database[0].connect(**database[1]) as conn:
            with conn.cursor() as cur:
                cur.execute('SET LOCAL session_replication_role=replica')
                _duplicate_calls(cur, candidate, 255, start=484000)
                cur.execute("UPDATE public.lab_arena_rounds SET status='published' WHERE round_id=%s", (rid,))
                cur.execute("UPDATE public.lab_arena_runs SET kind='execute' WHERE run_id=%s", (candidate['run_id'],))
                _helper_counter(cur)
                before, wasted = _counted_list(cur, rid, before=True)
                after, remaining = _counted_list(cur, rid)
                assert before == after == {'status': 'ok', 'items': []}
                assert wasted == 256 * 4 and remaining == 0
                record_property('published_execute_helper_calls_before', wasted)
                record_property('published_execute_helper_calls_after', remaining)
                # A published score head remains eligible and exercises the full
                # exact helper; pinning kind alone cannot invent a charge.
                cur.execute("UPDATE public.lab_arena_runs SET kind='score' WHERE run_id=%s", (candidate['run_id'],))
                before, old_work = _counted_list(cur, rid, before=True)
                after, new_work = _counted_list(cur, rid)
                assert before == after and len(after['items']) == 20
                assert old_work > 0 and new_work > 0
                record_property('eligible_score_helper_calls_after', new_work)
                # Every uncertain row is now superseded by an immutable
                # settlement. No amount of older uncertainty can enter a helper.
                cur.execute("INSERT INTO public.lab_arena_ledger "
                    "(entry_kind,miner_hotkey,round_id,submission_id,run_id,stage,call_identity,"
                    "provider,operation_id,funding_source,amount_microusd,entry_doc) "
                    "SELECT 'settlement',miner_hotkey,round_id,submission_id,run_id,stage,"
                    "call_identity,provider,operation_id,funding_source,0,'{}'::jsonb "
                    "FROM public.lab_arena_ledger WHERE round_id=%s AND entry_kind='uncertain'", (rid,))
                before, old_heads = _counted_list(cur, rid, before=True)
                after, new_heads = _counted_list(cur, rid)
                assert before == after == {'status': 'ok', 'items': []}
                assert old_heads == 256 and new_heads == 0
                record_property('superseded_score_helper_calls_before', old_heads)
                record_property('superseded_score_helper_calls_after', new_heads)
                for old in (True, False):
                    result, calls = _counted_list(cur, rid, before=old, run_id='missing-run')
                    assert result == {'status': 'ok', 'items': []} and calls == 0
            # Restore exact helper definitions, sequences and synthetic rows.
            conn.rollback()
    finally:
        store.close()
