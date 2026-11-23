"""Paid executions that expire without final hand-back use the normal capacity cooldown."""
from pathlib import Path

import pytest

from tests.lab_arena import installer_capacity710_postgres_test as previous
from tests.lab_arena import stage_cutoff_drain420_postgres_test as jobs

SQL = Path(__file__).resolve().parents[2] / 'scripts/713-lab-arena-paid-execution-expiry-cooldown.sql'


@pytest.fixture(scope='module')
def database():
    for pg, dsn in previous.database.__wrapped__():
        with pg.connect(**dsn) as conn, conn.cursor() as q:
            q.execute(SQL.read_text())
            q.execute(SQL.read_text())
        yield pg, dsn


@pytest.mark.parametrize('scenario,blocked', [
    ('paid', True), ('zero', False), ('two', False), ('finished', False),
    ('unexpired', False), ('old', False), ('shortened', False),
])
def test_native_claim_after_paid_batch_loss(database, scenario, blocked):
    pg, dsn = database
    with pg.connect(**dsn) as conn:
        with conn.cursor() as q:
            q.execute("TRUNCATE public.lab_arena_ledger RESTART IDENTITY CASCADE")
        conn.commit()
        jobs._seed(conn, positions=8, current_schema=True)
        leases = [previous.claim(conn, jobs.RUNNER, t) for t in 'abcd']
        assert all(x['status'] == 'leased' for x in leases)
        ids = [x['run_id'] for x in leases[:3]]
        if scenario == 'finished':
            assert previous.event(conn, leases[2], 'runtime.finished', {'status':'provider_error'}, token='c')['status']=='accepted'
        with conn.cursor() as q:
            # Move the clock of three real leases in this disposable database.
            # Retain the original issued lease, grant and run binding.
            q.execute("SET LOCAL session_replication_role=replica")
            q.execute("""UPDATE lab_arena_runs SET lease_expires_at=now()-interval '5 seconds',
                claim_response=jsonb_set(claim_response,'{lease_expires_at}',to_jsonb((now()-interval '8 seconds')::text))
                WHERE run_id=ANY(%s)""", (ids,))
            if scenario == 'unexpired':
                q.execute("UPDATE lab_arena_runs SET lease_expires_at=now()+interval '15 minutes' WHERE run_id=%s",(ids[-1],))
            if scenario == 'shortened':
                q.execute("""UPDATE lab_arena_runs SET claim_response=jsonb_set(claim_response,
                    '{lease_expires_at}',to_jsonb((now()+interval '5 minutes')::text))
                    WHERE run_id=%s""",(ids[-1],))
            if scenario == 'old':
                q.execute("""UPDATE lab_arena_runs SET lease_expires_at=now()-interval '76 minutes',
                    claim_response=jsonb_set(claim_response,'{lease_expires_at}',to_jsonb((now()-interval '77 minutes')::text))
                    WHERE run_id=ANY(%s)""",(ids,))
            q.execute("SET LOCAL session_replication_role=origin")
            for i, run in enumerate(ids):
                fee = 0 if scenario == 'zero' or (scenario == 'two' and i == 2) else 100
                q.execute("""INSERT INTO lab_arena_ledger(entry_kind,miner_hotkey,round_id,submission_id,
                    run_id,stage,call_identity,provider,operation_id,funding_source,amount_microusd,entry_doc,created_at)
                    SELECT 'settlement',miner_hotkey,round_id,submission_id,run_id,stage,
                    'sha256:'||encode(extensions.digest(run_id,'sha256'),'hex'),'openrouter','openrouter.chat','miner_key',%s,'{}',
                    lease_expires_at-interval '1 minute' FROM lab_arena_runs WHERE run_id=%s""",(fee,run))
        conn.commit()
        jobs._rpc(conn, "SELECT lab_arena_expire_leases(%s)", (jobs.ROUND,))
        assert previous.claim(conn, jobs.RUNNER, 'd') == leases[3]
        assert previous.claim(conn, jobs.RUNNER, 'e')['status'] == ('no_pending' if blocked else 'leased')
        assert previous.claim(conn, previous.HEALTHY, 'f')['status'] == 'leased'
        assert jobs._complete(conn, leases[3], token='d')['status'] == 'accepted'
        with conn.cursor() as q:
            q.execute("SELECT count(*),sum(amount_microusd) FROM lab_arena_ledger WHERE run_id=ANY(%s)",(ids,))
            assert q.fetchone() == (3, 0 if scenario=='zero' else 200 if scenario=='two' else 300)

        if blocked:
            # Scoring uses the released scorer image and retains fresh capacity.
            with conn.cursor() as q:
                q.execute("SET LOCAL session_replication_role=replica")
                q.execute("""UPDATE lab_arena_rounds SET status='stage2_scoring',
                    configuration_doc=jsonb_set(configuration_doc,'{schedule}',configuration_doc->'schedule' ||
                        jsonb_build_object('final_scoring_close',(now()+interval '2 hours')::text,
                                           'publication_deadline',(now()+interval '3 hours')::text))
                    WHERE round_id=%s""",(jobs.ROUND,))
                q.execute("""INSERT INTO lab_arena_runs(run_id,assignment_id,round_id,submission_id,miner_hotkey,
                    stage,icp_position,attempt,kind,status,stage_generation,scoring_generation_floor,scored_run_id,
                    judgment_group_leader,judgment_group_miner_hotkeys)
                    SELECT assignment_id||':score:1',assignment_id||':score',round_id,submission_id,miner_hotkey,
                    stage,icp_position,1,'score','pending',stage_generation,0,run_id,true,ARRAY[miner_hotkey]
                    FROM lab_arena_runs WHERE run_id=%s""",(leases[3]['run_id'],))
            conn.commit()
            judge = previous.claim(conn, jobs.RUNNER, '7')
            assert judge['status'] == 'leased', judge
            assert (judge['kind'], judge['scored_run_id']) == ('score', leases[3]['run_id'])
