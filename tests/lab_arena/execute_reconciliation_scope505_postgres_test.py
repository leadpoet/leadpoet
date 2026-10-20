"""Exact V2 JSON and authority parity after existing-index run scoping."""
import json
from pathlib import Path

import pytest

from tests.lab_arena import zero_call_host_attempt504_postgres_test as current
from tests.lab_arena import flat_per_icp_cost500_postgres_test as flat
from tests.lab_arena import deepline_candidate_head_scope483_postgres_test as heads
from tests.lab_arena import host_interruption_retry474_postgres_test as execution

ROOT = Path(__file__).parents[2]
SQL = ROOT / 'scripts/505-lab-arena-execute-reconciliation-run-scope.sql'
V2 = 'public.lab_arena_list_deepline_cost_reconciliations_v2'
SIGNATURE = V2 + '(text,text,bigint,integer,boolean)'
OLD = 'public.test_before_execute_scope505'


def snapshot(cur):
    cur.execute("SELECT indexname,indexdef FROM pg_indexes WHERE schemaname='public' ORDER BY indexname")
    indexes = cur.fetchall()
    return flat.catalog(cur), heads._rows(cur), indexes


@pytest.fixture(scope='module')
def database():
    gen = current.database.__wrapped__()
    db = next(gen)
    try:
        with db[0].connect(**db[1]) as conn, conn.cursor() as cur:
            cur.execute('SELECT pg_get_functiondef(%s::regprocedure)', (SIGNATURE,))
            old = cur.fetchone()[0]
            cur.execute(old.replace('FUNCTION ' + V2 + '(', 'FUNCTION ' + OLD + '(', 1))
            conn.commit()
            before = snapshot(cur)
            text = SQL.read_text()
            for name, message in (('v_preimage', 'preimage differs'), ('v_postimage', 'postimage differs')):
                digest = text.split(name + " CONSTANT TEXT := '")[1].split("'")[0]
                with pytest.raises(db[0].Error, match=message):
                    cur.execute(text.replace(digest, '0' * 64, 1))
                conn.rollback()
                assert snapshot(cur) == before
            cur.execute(f'GRANT EXECUTE ON FUNCTION {SIGNATURE} TO anon')
            with pytest.raises(db[0].Error, match='security shape differs'):
                cur.execute(text)
            conn.rollback()
            assert snapshot(cur) == before
            cur.execute('CREATE ROLE execute505_migrator NOLOGIN NOSUPERUSER INHERIT')
            cur.execute('GRANT lab_arena_owner TO execute505_migrator')
            cur.execute('GRANT USAGE,CREATE ON SCHEMA public TO execute505_migrator')
            cur.execute('ALTER DEFAULT PRIVILEGES FOR ROLE execute505_migrator IN SCHEMA public '
                        'GRANT EXECUTE ON FUNCTIONS TO anon,authenticated,service_role')
            cur.execute('SET ROLE execute505_migrator')
            cur.execute(text)
            after = snapshot(cur)
            assert before[1:] == after[1:]
            changed = []
            for a, b in zip(before[0], after[0]):
                assert a[0] == b[0] and a[2:] == b[2:]
                if a[1] != b[1]:
                    changed.append(a[0])
            assert changed == ['lab_arena_list_deepline_cost_reconciliations_v2(text,text,bigint,integer,boolean)']
            cur.execute(text)
            assert snapshot(cur) == after
            cur.execute('RESET ROLE')
        yield db
    finally:
        gen.close()


def pair(cur, rid, run_id='', after=0, limit=20, flag=True):
    # Disable disposable-database JIT compilation; both real SQL bodies and
    # every authority helper still execute unchanged.
    cur.execute('SET LOCAL jit=off')
    args = (rid, run_id, after, limit, flag)
    cur.execute('SELECT ' + OLD + '(%s,%s,%s,%s,%s),' + V2 + '(%s,%s,%s,%s,%s)', args + args)
    old, new = cur.fetchone()
    assert old == new
    return new['items']


def seed(database, suffix):
    with database[0].connect(**database[1]) as conn:
        execution.seed(conn)
        lease = execution.claim(conn)
        assert lease['status'] == 'leased', lease
        identity = 'sha256:' + 'e' * 64
        metadata = {'reserve_remaining_budget': True, 'tool': 'firecrawl_scrape',
                    'deepline_request_id': 'ctx-tool-' + 'e' * 32,
                    'credential_fingerprint': 'sha256:' + 'a' * 64}
        assert execution.rpc(conn, 'lab_arena_reserve_call(%s,%s,%s,%s,%s,%s,0,%s::jsonb,4500)',
            (lease['run_id'], 'sha256:' + 'a' * 64, identity, 'deepline.execute',
             'deepline', 'miner_key', json.dumps(metadata)))['status'] == 'reserved'
        assert execution.rpc(conn, 'lab_arena_mark_dispatched(%s,%s,%s)',
            (lease['run_id'], 'sha256:' + 'a' * 64, identity))['status'] == 'dispatched'
        call = dict(metadata, reason='missing_provider_cost', call_succeeded=True,
                    deepline_operation='firecrawl_scrape')
        assert execution.rpc(conn, 'lab_arena_mark_uncertain(%s,%s,%s,%s::jsonb,4500)',
            (lease['run_id'], 'sha256:' + 'a' * 64, identity, json.dumps(call)))['status'] == 'uncertain'
        with conn.cursor() as cur:
            items = pair(cur, execution.ROUND)
            assert len(items) == 1
            candidate = items[0]
        return execution.ROUND, candidate


def test_full_json_parity_for_round_policy_kind_flag_and_cursor(database):
    rid, candidate = seed(database, 'scope505matrix')
    with database[0].connect(**database[1]) as conn, conn.cursor() as cur:
        cur.execute('SET LOCAL session_replication_role=replica')
        heads._duplicate_calls(cur, candidate, 24, start=505000)
        cur.execute("UPDATE public.lab_arena_ledger SET entry_doc=jsonb_set(jsonb_set("
                    "entry_doc,'{call,reason}',to_jsonb('missing_provider_cost'::text)),"
                    "'{call,call_succeeded}','true'::jsonb) WHERE round_id=%s AND entry_kind='uncertain'", (rid,))
        cur.execute("SELECT entry_id FROM public.lab_arena_ledger WHERE round_id=%s "
                    "AND entry_kind='uncertain' ORDER BY entry_id", (rid,))
        ids = [r[0] for r in cur.fetchall()]
        assert len(ids) == 25
        assert [x['uncertain_entry_id'] for x in pair(cur, rid)] == ids[:20]
        assert pair(cur, rid, after=ids[12], limit=1)[0]['uncertain_entry_id'] == ids[13]
        assert pair(cur, rid, after=ids[-1], limit=1)[0]['uncertain_entry_id'] == ids[0]
        for status in ('stage1_scoring', 'open', 'published', 'cancelled'):
            for policy in ('successful_calls_v1', 'successful_calls_per_icp_v1',
                           'successful_calls_flat_per_icp_v1', 'legacy', None):
                cur.execute("UPDATE public.lab_arena_rounds SET status=%s,configuration_doc="
                            "configuration_doc || jsonb_build_object('sourcing_cost_eligibility_policy',%s::text) "
                            "WHERE round_id=%s", (status, policy, rid))
                for kind in ('execute', 'score'):
                    cur.execute('UPDATE public.lab_arena_runs SET kind=%s WHERE run_id=%s', (kind, candidate['run_id']))
                    for flag in (True, False, None):
                        items = pair(cur, rid, flag=flag)
                        if status == 'open' or flag is None or flag is True and kind != 'execute':
                            assert items == []
        cur.execute("UPDATE public.lab_arena_rounds SET status='cancelled' WHERE round_id=%s", (rid,))
        cur.execute("UPDATE public.lab_arena_runs SET kind='execute' WHERE run_id=%s", (candidate['run_id'],))
        for run_id in ('', None, candidate['run_id'], 'missing-run'):
            for after in (0, ids[12], 9223372036854775807):
                for limit in (0, 1, 20):
                    for flag in (True, False, None):
                        items = pair(cur, rid, run_id=run_id, after=after, limit=limit, flag=flag)
                        if run_id == 'missing-run' or limit == 0 or flag is None:
                            assert items == []
        assert pair(cur, 'missing-round') == pair(cur, None) == []


@pytest.mark.parametrize('damage', ['later_settlement', 'later_nonsettlement', 'wrong_run_round',
                                    'wrong_reservation_binding', 'failed_worker_result'])
def test_latest_heads_and_all_existing_identity_guards_remain_authoritative(database, damage):
    rid, candidate = seed(database, 'scope505-' + damage)
    with database[0].connect(**database[1]) as conn, conn.cursor() as cur:
        cur.execute('SET LOCAL session_replication_role=replica')
        cur.execute("UPDATE public.lab_arena_rounds SET status='cancelled' WHERE round_id=%s", (rid,))
        cur.execute("UPDATE public.lab_arena_runs SET kind='execute' WHERE run_id=%s", (candidate['run_id'],))
        cur.execute("UPDATE public.lab_arena_ledger SET entry_doc=jsonb_set(jsonb_set(entry_doc,"
                    "'{call,reason}',to_jsonb('missing_provider_cost'::text)),"
                    "'{call,call_succeeded}','true'::jsonb) WHERE entry_id=%s", (candidate['uncertain_entry_id'],))
        assert len(pair(cur, rid)) == 1
        if damage == 'later_settlement':
            cur.execute("INSERT INTO public.lab_arena_ledger(entry_kind,miner_hotkey,round_id,submission_id,"
                        "run_id,stage,call_identity,provider,operation_id,funding_source,amount_microusd,entry_doc) "
                        "SELECT 'settlement',miner_hotkey,round_id,submission_id,run_id,stage,call_identity,"
                        "provider,operation_id,funding_source,0,'{}'::jsonb FROM public.lab_arena_ledger WHERE entry_id=%s", (candidate['uncertain_entry_id'],))
        elif damage == 'later_nonsettlement':
            cur.execute('UPDATE public.lab_arena_ledger SET entry_id=-entry_id WHERE entry_id=%s', (candidate['uncertain_entry_id'],))
        elif damage == 'wrong_run_round':
            cur.execute("UPDATE public.lab_arena_runs SET round_id='missing-round' WHERE run_id=%s", (candidate['run_id'],))
        elif damage == 'wrong_reservation_binding':
            cur.execute("UPDATE public.lab_arena_ledger SET entry_doc=jsonb_set(entry_doc,"
                        "'{credential_fingerprint}',to_jsonb(%s::text)) WHERE call_identity=%s AND entry_kind='reservation'",
                        ('sha256:' + 'b' * 64, candidate['call_identity']))
        else:
            cur.execute("UPDATE public.lab_arena_ledger SET entry_doc=jsonb_set(entry_doc,"
                        "'{call,call_succeeded}','false'::jsonb) WHERE entry_id=%s", (candidate['uncertain_entry_id'],))
        assert pair(cur, rid) == []


@pytest.mark.parametrize('response_success', [True, False])
def test_saved_response_success_and_unsuccessful_worker_json_parity(database, response_success):
    rid, candidate = seed(database, 'saved')
    with database[0].connect(**database[1]) as conn, conn.cursor() as cur:
        cur.execute('SET LOCAL session_replication_role=replica')
        cur.execute("UPDATE public.lab_arena_ledger SET entry_doc=jsonb_set(entry_doc,"
                    "'{call,call_succeeded}','false'::jsonb) WHERE entry_id=%s", (candidate['uncertain_entry_id'],))
        assert pair(cur, rid) == []
        assert len(pair(cur, rid, flag=False)) == 1
        cur.execute("INSERT INTO public.lab_arena_deepline_call_responses(call_identity,"
                    "reservation_entry_id,run_id,request_id,terminal_response) "
                    "SELECT call_identity,entry_id,run_id,'native-recovered',%s::jsonb "
                    "FROM public.lab_arena_ledger WHERE call_identity=%s AND entry_kind='reservation'",
                    (json.dumps({'call_succeeded': response_success, 'status': 200}), candidate['call_identity']))
        assert len(pair(cur, rid)) == int(response_success)


@pytest.mark.parametrize('row_kind', ['reservation', 'dispatch'])
@pytest.mark.parametrize('column,value', [('run_id','foreign-run'), ('round_id','foreign-round'),
    ('submission_id','foreign-submission'), ('miner_hotkey','foreign-miner'),
    ('stage',1), ('provider','openrouter'), ('operation_id','foreign-operation'), ('funding_source','host')])
def test_every_final_ledger_identity_join_still_fails_closed(database, row_kind, column, value):
    rid, candidate = seed(database, 'binding')
    with database[0].connect(**database[1]) as conn, conn.cursor() as cur:
        cur.execute('SET LOCAL session_replication_role=replica')
        assert len(pair(cur, rid)) == 1
        cur.execute(f'UPDATE public.lab_arena_ledger SET {column}=%s WHERE call_identity=%s AND entry_kind=%s',
                    (value, candidate['call_identity'], row_kind))
        assert pair(cur, rid, flag=True) == pair(cur, rid, flag=False) == []
