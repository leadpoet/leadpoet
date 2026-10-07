"""A native upstream status cannot relabel a later confirmed-budget refusal."""
import json
from dataclasses import replace
from pathlib import Path

import pytest

from lab_arena import contracts, deepline_catalog
from lab_arena.store import hash_lease_token
from tests.lab_arena import oct07_cutoff_recovery419_postgres_test as current_schema
from tests.lab_arena.oct07_cutoff_recovery419_postgres_test import base_database
from tests.lab_arena.parallel_twenty_icp_execution_postgres_test import _start_parallel_round
from tests.lab_arena.per_icp_cost_admission_postgres_test import _settle
from tests.lab_arena.test_deepline_catalog import row
from tests.lab_arena.test_lab_arena_migration_postgres import claim, sha
from tests.lab_arena.test_lab_arena_service_round import Harness, keypair

SQL = Path(__file__).parents[2] / 'scripts/421-lab-arena-bound-account-refusal.sql'


@pytest.fixture(scope='module')
def database(base_database):
    current_schema.database.__wrapped__(base_database)
    with base_database[0].connect(**base_database[1]) as conn, conn.cursor() as cur:
        cur.execute(SQL.read_text())
        cur.execute(SQL.read_text())
    return base_database


def test_migration_rejects_unreviewed_preimage_without_changing_function(database):
    with database[0].connect(**database[1]) as conn, conn.cursor() as cur:
        signature = 'public.lab_arena_reserve_call(text,text,text,text,text,text,bigint,jsonb,integer)'
        cur.execute('SELECT pg_get_functiondef(%s::regprocedure)', (signature,))
        before = cur.fetchone()[0]
        wrong = SQL.read_text().replace(
            'f41d91bec7f3182accf9aad86541d8abe6380ed6ff000bac1aeca83c78dc222d', '0' * 64)
        with pytest.raises(Exception, match='preimage differs'):
            cur.execute(wrong)
        conn.rollback()
        cur.execute('SELECT pg_get_functiondef(%s::regprocedure)', (signature,))
        assert cur.fetchone()[0] == before


@pytest.mark.parametrize('proof,expected', [
    (None, False),
    ({'error_class':'account_credential_failure','provider_status':402}, True),
    ({'error_class':'account_credential_failure','provider_status':401}, False),
    ({'error_class':'account_credential_failure','provider_status':'402'}, False),
    ({'error_class':'upstream_error','provider_status':402}, False),
])
def test_budget_refusal_requires_existing_bound_account_proof(database,tmp_path,proof,expected):
    label='21-'+sha(json.dumps(proof))[-8:]
    connect = lambda: database[0].connect(**database[1])
    h = Harness(connect,tmp_path,challengers=['AccountControl'],runners=['alpha'])
    miner = keypair('svc-miner-AccountControl').ss58_address
    h.chain.owned[miner] = [miner]
    h.service.config.defaults = replace(h.service.config.defaults,
        per_icp_cost_policy=True,integrity_from='2000-01-01T00:00:00Z')
    catalog = deepline_catalog.freeze_catalog({'tools':[row('vendor_company_search')]})
    h.service.config.deepline_catalog_source = lambda **_: catalog
    participants = _start_parallel_round(h,'arena-2099-08-'+label,slot_ceiling=2)
    lease,token = claim(h.service.store,h.round_id,h.runner_keys[0],parallelism=2,ceiling=2,
        excluded=[p['miner_hotkey'] for p in participants if p['is_king']])[:2]
    assert h.service.store.provider_funding(lease['run_id'],'deepline')['funding_source']=='miner_key'
    store=h.service.store
    def reserve(sequence):
        identity=contracts.provider_call_identity(attempt=lease['attempt'],
            assignment_id=lease['assignment_id'],icp_position=lease['icp_position'],
            action_sequence=sequence,operation_id='deepline.execute',request_hash=sha(str(sequence)))
        result=store.reserve_call(run_id=lease['run_id'],lease_token_hash=hash_lease_token(token),
            call_identity=identity,operation_id='deepline.execute',provider='deepline',
            funding_source='miner_key',amount_microusd=1,
            call_doc={'request_hash':sha(str(sequence))})
        return identity,result
    first,result=reserve(0)
    assert result['status']=='reserved'
    assert store.mark_dispatched(run_id=lease['run_id'],lease_token_hash=hash_lease_token(token),call_identity=first)['status']=='dispatched'
    call={'reason':'missing_provider_cost','provider_status':402,'call_succeeded':False}
    if proof is not None:call['account_failure_evidence']=proof
    assert store.mark_uncertain(run_id=lease['run_id'],lease_token_hash=hash_lease_token(token),call_identity=first,call_doc=call)['status']=='uncertain'
    billed,result=reserve(1)
    assert result['status']=='reserved'  # Unknown cost still creates no hold.
    _settle(store,lease,token,billed,4_000_000)
    denied,result=reserve(2)
    assert result['status']=='refused' and result['reason']=='money_cap'
    assert result['prior_miner_credential_refusal'] is expected
    assert reserve(2)[1]['prior_miner_credential_refusal'] is expected
    with connect() as conn,conn.cursor() as cur:
        cur.execute('SELECT entry_kind,amount_microusd,entry_doc FROM public.lab_arena_ledger WHERE call_identity=%s ORDER BY entry_id DESC LIMIT 1',(first,))
        last=cur.fetchone()
        assert last[0:2]==('uncertain',0) and last[2]['call']==call
        cur.execute('SELECT max(amount_microusd) FROM public.lab_arena_ledger WHERE run_id=%s AND entry_kind=%s',(lease['run_id'],'reservation'))
        assert cur.fetchone()[0]==0
        cur.execute(SQL.read_text())
