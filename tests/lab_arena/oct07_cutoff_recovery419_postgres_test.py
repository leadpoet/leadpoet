"""Disposable-PG controls for the exact October7 cancelled-round recovery."""
from __future__ import annotations

import copy
import json
import os
from datetime import datetime, timedelta, timezone
from dataclasses import replace
from pathlib import Path

import pytest

from lab_arena import contracts
from tests.lab_arena import abandoned_host_recovery412_postgres_test as recovery
from tests.lab_arena.test_lab_arena_migration_postgres import hotkey
from tests.lab_arena.sep18_schedule_capacity288_postgres_test import (
    _configuration as fixture_configuration,
)
from tests.lab_arena.company_quality_round_test import QualityHarness
from tests.lab_arena.company_only_capacity_e2e_test import _install_company_only_sandbox
from tests.lab_arena.icp_fixtures import daily_icps
from tests.lab_arena.parallel_twenty_icp_execution_postgres_test import (
    _verified_test_pool,
)

ROOT = Path(__file__).resolve().parents[2]
MIGRATION = ROOT / "scripts/419-arena-2026-10-07-cutoff-recovery.sql"
ROUND = "arena-2026-10-07"
OTHER = "arena-2026-10-08"
HASHES = {
    "round": "917c007fcc697dbc52ecfb80650af90ae528b97c8c0df888c81024fbac290cd1",
    "targets": "4358942ab1c09fb04fd3d0a57d56e3f7eaf0be723fa481479a2b8703a9fa87ed",
    "config": "bcc605522feb7e6d4be5ed94fd120b15a79f624db0191c2492e7b265d28878a3",
    "post": "12e3f0e058bb3d0c88847ff459b762e7f17b79c02214b3e143033cd8bc57d0ba",
}

SCHEDULE = {
    "submission_open": "2026-10-06T00:00:00Z",
    "submission_cutoff": "2026-10-07T00:00:00Z",
    "benchmark_deadline": "2026-10-07T00:30:00Z",
    "stage_1_start": "2026-10-07T00:30:01Z",
    "stage_1_close": "2026-10-07T04:30:01Z",
    "stage_1_scoring_close": "2026-10-07T11:00:01Z",
    "stage_2_start": "2026-10-07T11:00:02Z",
    "stage_2_close": "2026-10-07T14:00:02Z",
    "final_scoring_close": "2026-10-07T20:30:02Z",
    "publication_deadline": "2026-10-07T20:30:03Z",
}
NEW_DATES = {
    "stage_2_close": "2026-10-07T17:00:02Z",
    "final_scoring_close": "2026-10-07T23:30:02Z",
    "publication_deadline": "2026-10-07T23:30:03Z",
}
base_database = recovery.database


@pytest.fixture(scope="module")
def database(base_database):
    # Reuse the exact historical401 -> current412 setup, then install the
    # subsequent current migrations. Do not weaken their preimage guards.
    recovery.migrated.__wrapped__(base_database)
    with base_database[0].connect(**base_database[1]) as connection, connection.cursor() as cur:
        for number in (404, 413, 414, 415, 416, 417):
            paths = list((ROOT / "scripts").glob(f"{number}-*.sql"))
            assert len(paths) == 1
            cur.execute(paths[0].read_text())
    return base_database


def _snapshot(cur):
    value = {}
    for table, order in (
        ("lab_arena_rounds", "round_id"),
        ("lab_arena_submissions", "submission_id"),
        ("lab_arena_runs", "run_id"),
        ("lab_arena_ledger", "entry_id"),
    ):
        cur.execute(
            f"SELECT coalesce(jsonb_agg(to_jsonb(x) ORDER BY {order}),'[]'::jsonb) "
            f"FROM public.{table} x"
        )
        value[table] = cur.fetchone()[0]
    cur.execute(
        "SELECT pg_get_triggerdef(t.oid),t.tgenabled,pg_get_functiondef(t.tgfoid) "
        "FROM pg_trigger t WHERE t.tgrelid='public.lab_arena_rounds'::regclass "
        "AND NOT t.tgisinternal ORDER BY t.tgname"
    )
    value["round_triggers"] = cur.fetchall()
    return value


def _hash(cur, expression, table, order, where=""):
    cur.execute(f"SELECT encode(extensions.digest(coalesce(jsonb_agg({expression} ORDER BY {order}),'[]'::jsonb)::text,'sha256'),'hex') FROM {table} x {where}")
    return cur.fetchone()[0]


def _values(cur):
    cur.execute("SELECT encode(extensions.digest(to_jsonb(r)::text,'sha256'),'hex'), encode(extensions.digest(configuration_doc::text,'sha256'),'hex'), encode(extensions.digest(jsonb_set(jsonb_set(jsonb_set(configuration_doc,'{schedule,stage_2_close}','\"2026-10-07T17:00:02Z\"',false),'{schedule,final_scoring_close}','\"2026-10-07T23:30:02Z\"',false),'{schedule,publication_deadline}','\"2026-10-07T23:30:03Z\"',false)::text,'sha256'),'hex') FROM public.lab_arena_rounds r WHERE round_id=%s", (ROUND,))
    row,config,post=cur.fetchone()
    where=f"WHERE round_id='{ROUND}'"
    return dict(round=row, config=config, post=post,
        runs=_hash(cur,"to_jsonb(x)","public.lab_arena_runs","run_id",where),
        submissions=_hash(cur,"to_jsonb(x)","public.lab_arena_submissions","submission_id",where),
        targets=_hash(cur,"to_jsonb(x)","public.lab_arena_runs","run_id",where+" AND kind='execute' AND stage=2 AND attempt=1 AND status='failed' AND terminal_cause='stage_closed' AND stage_generation=5"))


def _seed(connection):
    config = fixture_configuration(ROUND)
    config["schedule"] = copy.deepcopy(SCHEDULE)
    config["max_attempts_per_assignment"] = 2
    with connection.cursor() as cur:
        cur.execute("SET session_replication_role=replica")
        cur.execute("TRUNCATE public.lab_arena_ledger,public.lab_arena_runs,public.lab_arena_submissions,public.lab_arena_rounds RESTART IDENTITY CASCADE")
        cur.execute("UPDATE public.lab_arena_restart_claim_control SET operator_paused=false,guard_commitment='',owner_commitment='',candidate_commit='',restart_scope='',restart_phase='',guard_expires_at=NULL,captured_leases='[]'")
        for name in (ROUND, OTHER):
            document=dict(config,round_id=name)
            cur.execute("INSERT INTO public.lab_arena_rounds(round_id,status,status_generation,stage_generation,cancel_reason,configuration_doc,participants,benchmark_ref,evaluation_date,icp_set_date,rewards_enabled) VALUES (%s,'cancelled',8,6,'execution_incomplete:stage2:80',%s::jsonb,'[{\"submission_id\":\"preserve\"}]',%s,'2026-10-07','2026-10-06',true)",(name,json.dumps(document),f"arena/{name}/benchmark.json"))
            sub=name+":miner"
            cur.execute("INSERT INTO public.lab_arena_submissions(submission_id,round_id,miner_hotkey,status,is_king,source_ref,submission_doc) VALUES (%s,%s,%s,'frozen',false,%s,'{\"source_sha256\":\"preserve\"}')",(sub,name,hotkey(name),f"arena/{name}/sources/exact.tar.gz"))
            for index in range(81 if name==ROUND else 1):
                accepted=index==80
                assignment=name+":"+sub+":2:"+str(index)
                cur.execute("INSERT INTO public.lab_arena_runs(run_id,assignment_id,round_id,submission_id,miner_hotkey,stage,icp_position,attempt,kind,status,stage_generation,terminal_cause,terminal_doc,lease_generation,lease_token_hash,runner_hotkey,lease_expires_at,output_ref,result_doc,per_icp_score) VALUES (%s,%s,%s,%s,%s,2,%s,1,'execute',%s,5,%s,%s::jsonb,%s,%s,%s,%s,%s,%s::jsonb,%s)",
                    (assignment+':1',assignment,name,sub,hotkey(name),10+index%10,'accepted' if accepted else 'failed',None if accepted else 'stage_closed',json.dumps({'closed_at':'2026-10-07T14:00:04Z','previous_status':'leased' if index<10 else 'pending'}),1 if index<10 else 0,'sha256:'+'a'*64 if index<10 else None,hotkey('runner') if index<10 else None,'2026-10-07T15:15:00Z' if index<10 else None,f'arena/{name}/outputs/exact.json' if accepted else None,json.dumps({'keep':'receipt'}) if accepted else None,26.25 if accepted else None))
            cur.execute("INSERT INTO public.lab_arena_ledger(entry_kind,round_id,submission_id,run_id,miner_hotkey,stage,call_identity,provider,operation_id,funding_source,amount_microusd,entry_doc) VALUES ('uncertain',%s,%s,%s,%s,2,%s,'openrouter','openrouter.chat','miner_key',1234,'{\"keep\":\"billing-provenance\"}')",(name,sub,name+':'+sub+':2:0:1',hotkey(name),'sha256:'+('a' if name==ROUND else 'b')*64))
        cur.execute("SET session_replication_role=origin")
        values=_values(cur)
    connection.commit()
    return values


def _sql(values, at="2026-10-07T14:30:00Z"):
    sql=MIGRATION.read_text()
    for key,production in HASHES.items():
        assert sql.count(production)==1
        sql=sql.replace(production,values[key])
    sql=sql.replace('690::BIGINT','1::BIGINT').replace('775::BIGINT','81::BIGINT')
    return sql.replace("pg_catalog.clock_timestamp() >", f"'{at}'::TIMESTAMPTZ >").replace("pg_catalog.clock_timestamp() <", f"'{at}'::TIMESTAMPTZ <")


def _execute(connection, sql):
    with connection.cursor() as cur:
        cur.execute(sql)
    connection.commit()



def test_append_only_retry_preserves_all_old_rows_and_replays_after_progress(database):
    psycopg,dsn=database
    with psycopg.connect(**dsn) as conn:
        values=_seed(conn)
        with conn.cursor() as cur: before=_snapshot(cur)
        _execute(conn,_sql(values))
        with conn.cursor() as cur: after=_snapshot(cur)
        original_ids={r['run_id'] for r in before['lab_arena_runs']}
        assert [r for r in after['lab_arena_runs'] if r['run_id'] in original_ids] == before['lab_arena_runs']
        for table in ('lab_arena_submissions','lab_arena_ledger','round_triggers'):
            assert after[table]==before[table]
        retries=[r for r in after['lab_arena_runs'] if r['run_id'] not in original_ids]
        assert len(retries)==80
        assert all(r['attempt']==2 and r['stage_generation']==7 and r['status']=='pending' and r['lease_token_hash'] is None and r['claim_response'] is None for r in retries)
        target=next(r for r in after['lab_arena_rounds'] if r['round_id']==ROUND)
        assert (target['status'],target['status_generation'],target['stage_generation'],target['cancel_reason'])==('stage2',9,7,None)
        assert target['configuration_doc']['schedule']==dict(SCHEDULE,**NEW_DATES)
        with conn.cursor() as cur:
            cur.execute('SET LOCAL session_replication_role=replica')
            cur.execute("UPDATE public.lab_arena_runs SET status='accepted',output_ref='preserve-new-output',result_doc='{\"preserve\":\"new-receipt\"}' WHERE run_id=%s",(retries[0]['run_id'],))
        conn.commit()
        with conn.cursor() as cur: progressed=_snapshot(cur)
        _execute(conn,_sql(values,'2026-10-07T18:00:00Z'))
        with conn.cursor() as cur: assert _snapshot(cur)==progressed
        with pytest.raises(Exception,match='write-once'):
            _execute(conn,"UPDATE public.lab_arena_rounds SET configuration_doc=jsonb_set(configuration_doc,'{schedule,stage_2_close}','\"2026-10-07T18:00:02Z\"') WHERE round_id='arena-2026-10-07'")
        conn.rollback()


def _failure(database,mutation=None,*,at='2026-10-07T14:30:00Z',expected='differ',inject=None):
    psycopg,dsn=database
    with psycopg.connect(**dsn) as conn:
        values=_seed(conn)
        if mutation:
            with conn.cursor() as cur:
                cur.execute('SET LOCAL session_replication_role=replica')
                cur.execute(mutation)
            conn.commit()
        with conn.cursor() as cur: before=_snapshot(cur)
        sql=_sql(values,at)
        if inject: sql=sql.replace(inject,"RAISE EXCEPTION 'injected rollback';\n  "+inject)
        with pytest.raises(Exception,match=expected): _execute(conn,sql)
        conn.rollback()
        with conn.cursor() as cur: assert _snapshot(cur)==before


@pytest.mark.parametrize('at',['2026-10-07T14:00:03Z','2026-10-07T15:00:01Z'])
def test_first_apply_rejects_outside_time_window(database,at):
    _failure(database,at=at,expected='outside bounded')


@pytest.mark.parametrize('change',[
    "status='stage2'", "status='published'", "cancel_reason='other'",
    'stage_generation=7','status_generation=9',"participants='[]'::jsonb",
    "stage2_scoring_plan_doc='{}'::jsonb", "icp_set_date='2026-10-05'",
    "configuration_doc=jsonb_set(configuration_doc,'{max_attempts_per_assignment}','3')",
    "configuration_doc=jsonb_set(configuration_doc,'{scorer_image_digest}','\"sha256:changed\"')",
    "configuration_doc=jsonb_set(configuration_doc,'{schedule,submission_cutoff}','\"2026-10-07T00:00:01Z\"')",
])
def test_frozen_round_drift_fails_closed(database,change):
    _failure(database,f"UPDATE public.lab_arena_rounds SET {change} WHERE round_id='{ROUND}'")


@pytest.mark.parametrize('change',[
    "terminal_cause='model_error'",'attempt=2',"stage_generation=6",
    "result_doc='{}'::jsonb", "output_ref='foreign'",'per_icp_score=0',
    "status='accepted'", "round_id='arena-2026-10-08'",
])
def test_exact_target_changes_fails_closed(database,change):
    _failure(database,f"UPDATE public.lab_arena_runs SET {change} WHERE round_id='{ROUND}' AND run_id=(SELECT min(run_id) FROM public.lab_arena_runs WHERE round_id='arena-2026-10-07')")


def test_operator_guard_rejects_recovery(database):
    _failure(database,"UPDATE public.lab_arena_restart_claim_control SET operator_paused=true",expected='guard is active')


def test_valid_restart_guard_rejects_recovery(database):
    _failure(database,"UPDATE public.lab_arena_restart_claim_control SET guard_commitment='sha256:'||repeat('a',64),owner_commitment='sha256:'||repeat('b',64),guard_generation=1,guard_expires_at=now()+interval '1 hour',candidate_commit=repeat('c',40),restart_scope='gateway',restart_phase='draining',captured_leases='[]'",expected='guard is active')


@pytest.mark.parametrize('inject',[
    'ALTER TABLE public.lab_arena_rounds DISABLE TRIGGER lab_arena_rounds_write_once;',
    "UPDATE public.lab_arena_rounds SET status='stage2',status_generation=9,",
    'ALTER TABLE public.lab_arena_rounds ENABLE TRIGGER lab_arena_rounds_write_once;',
])
def test_error_rolls_back_insert_update_and_trigger_state(database,inject):
    _failure(database,expected='injected rollback',inject=inject)


def test_partial_replay_rejected_without_reset(database):
    psycopg,dsn=database
    with psycopg.connect(**dsn) as conn:
        values=_seed(conn)
        _execute(conn,_sql(values))
        with conn.cursor() as cur:
            cur.execute('SET LOCAL session_replication_role=replica')
            cur.execute("DELETE FROM public.lab_arena_runs WHERE run_id=(SELECT min(run_id) FROM public.lab_arena_runs WHERE attempt=2)")
        conn.commit()
        with conn.cursor() as cur: before=_snapshot(cur)
        with pytest.raises(Exception,match='replay state differs'): _execute(conn,_sql(values))
        conn.rollback()
        with conn.cursor() as cur: assert _snapshot(cur)==before


def _insert_json_record(cur,table,document):
    cur.execute("SELECT column_name FROM information_schema.columns WHERE table_schema='public' AND table_name=%s AND is_generated='NEVER' ORDER BY ordinal_position",(table,))
    columns=','.join('"'+r[0]+'"' for r in cur.fetchall())
    cur.execute(f'INSERT INTO public.{table} ({columns}) SELECT {columns} FROM jsonb_populate_record(NULL::public.{table},%s::jsonb)',(json.dumps(document),))


def test_verbatim_candidate_exact_protected_round_and_80_targets(database):
    """Verbatim SQL; actual round/targets, synthetic old accepted/sibling rows.

    The protected production fixture stays outside Git. Old payloads, source
    credentials and full ledger were deliberately not exported.
    """
    fixture_path=os.environ.get('ARENA_OCT07_RECOVERY_FIXTURE')
    if not fixture_path:
        pytest.skip('operator-only verbatim fixture; generic recovery controls run normally')
    fixture=Path(fixture_path)
    assert fixture.is_file(), 'required protected exact-round fixture missing'
    assert fixture.stat().st_mode & 0o077 == 0
    document=json.loads(fixture.read_text())
    round_row=json.loads(document['round_sql_json_text'])
    targets=json.loads(document['target_rows_sql_json_text'])
    psycopg,dsn=database
    with psycopg.connect(**dsn) as conn:
        with conn.cursor() as cur:
            cur.execute("SET TIME ZONE 'UTC'")
            cur.execute('SET LOCAL session_replication_role=replica')
            cur.execute('TRUNCATE public.lab_arena_rounds CASCADE')
            cur.execute("UPDATE public.lab_arena_restart_claim_control SET operator_paused=false,guard_commitment='',owner_commitment='',candidate_commit='',restart_scope='',restart_phase='',guard_expires_at=NULL,captured_leases='[]'")
            _insert_json_record(cur,'lab_arena_rounds',round_row)
            seen=set()
            for row in targets:
                if row['submission_id'] not in seen:
                    seen.add(row['submission_id'])
                    cur.execute("INSERT INTO public.lab_arena_submissions(submission_id,round_id,miner_hotkey,status,is_king,source_ref) VALUES (%s,%s,%s,'frozen',false,'arena/arena-2026-10-07/sources/synthetic.tar.gz')",(row['submission_id'],ROUND,row['miner_hotkey']))
                _insert_json_record(cur,'lab_arena_runs',row)
            sample=targets[0]
            for index in range(695):
                cur.execute("INSERT INTO public.lab_arena_runs(run_id,assignment_id,round_id,submission_id,miner_hotkey,stage,icp_position,attempt,kind,status,stage_generation,output_ref,result_doc) VALUES (%s,%s,%s,%s,%s,2,0,1,'execute',%s,5,%s,%s::jsonb)",
                    (f'synthetic-preserved-{index}:1',f'synthetic-preserved-{index}',ROUND,sample['submission_id'],sample['miner_hotkey'],'accepted' if index<690 else 'failed','arena/arena-2026-10-07/outputs/synthetic.json' if index<690 else None,json.dumps({'synthetic':'accepted-preservation'}) if index<690 else None))
        conn.commit()
        with conn.cursor() as cur:
            values=_values(cur)
            assert values['round']==HASHES['round']
            assert values['targets']==HASHES['targets']
            assert values['config']==HASHES['config']
            before=_snapshot(cur)
        # Real clock is in the bounded first-apply window. No SQL substitutions.
        _execute(conn,MIGRATION.read_text())
        with conn.cursor() as cur: after=_snapshot(cur)
        old_ids={r['run_id'] for r in before['lab_arena_runs']}
        assert [r for r in after['lab_arena_runs'] if r['run_id'] in old_ids]==before['lab_arena_runs']
        assert len(after['lab_arena_runs'])==855
        assert after['lab_arena_submissions']==before['lab_arena_submissions']
        assert after['lab_arena_ledger']==before['lab_arena_ledger']
        assert after['round_triggers']==before['round_triggers']
        _execute(conn,MIGRATION.read_text())
        with conn.cursor() as cur: assert _snapshot(cur)==after



def test_real_company_only_cutoff_recovery_reuses_old_receipts_and_publishes(database,tmp_path):
    """Real service/signed runner/SQL transitions with fake provider + judge.

    A smaller 10-ICP/2-participant round keeps this focused gate bounded.
    Only fixture hashes/counts/dates are projected; candidate guards stay intact.
    """
    from lab_arena.store import hash_lease_token
    from tests.lab_arena.confirmed_cost_admission_postgres_test import _reserve,_dispatch,_settle
    import re
    psycopg,dsn=database
    connect=lambda:psycopg.connect(**dsn)
    with connect() as conn,conn.cursor() as cur:
        cur.execute('SET LOCAL session_replication_role=replica')
        cur.execute('TRUNCATE public.lab_arena_rounds CASCADE')
    h=QualityHarness(connect,tmp_path,challengers=['PublicBaselineAlternate'],runners=['alpha','beta'])
    h.service.config.defaults=replace(h.service.config.defaults,benchmark_icp_count=10,runner_slot_ceiling=4,company_quality_from=None,intent_details_from='2026-01-01T00:00:00Z',contacts_from=None,execution_sequence_from='2026-01-01T00:00:00Z',per_icp_cost_policy=True,rewards_enabled=True)
    h.service.config.daily_icp_source=lambda **kwargs:{'status':'ready','set_id':int(kwargs['set_id']),'icps':copy.deepcopy(daily_icps()[:10])}
    observed=_install_company_only_sandbox(h)
    h.round_id=ROUND
    h.service.create_round(datetime.now(timezone.utc)+timedelta(minutes=30),round_id=ROUND)
    h.submit('PublicBaselineAlternate',ROUND)
    h.clock.advance_to(h.schedule()['submission_cutoff'])
    assert h.service.advance_round(ROUND)['status']=='ok'
    participants=h.service.store.get_round(ROUND)['participants']
    for participant in participants: h.flavors.setdefault(participant['submission_id'],'PublicBaseline')
    h.clock.advance_to(h.schedule()['stage_1_start'])
    assert h.service.advance_round(ROUND)['status']=='ok'
    original_runner=h.runner
    def verified_runner(index,parallel=4):
        runner=original_runner(index,parallel)
        runner._config.proxy_worker_pool=_verified_test_pool(parallel)
        return runner
    h.runner=verified_runner
    h.advance_until('stage2',runners=1,max_steps=40)
    scheduled_now=h.clock.now
    h.clock.now=datetime.now(timezone.utc)
    retained_runner=h.runner(0)
    old_lease=retained_runner.claim_one()
    other=retained_runner.claim_one()
    assert old_lease['status']==other['status']=='leased'
    assert retained_runner._slots.acquire(blocking=False)
    retained_runner._run_lease(other)
    assert retained_runner.completed[-1]['result']['status']=='accepted'
    store=h.service.store
    old_token=old_lease['lease_token']
    settled_identity,reserved=_reserve(store,old_lease,old_token,'original-paid',200000)
    assert reserved['status']=='reserved'
    assert _settle(store,old_lease,old_token,settled_identity,200000)['status']=='settled'
    from tests.lab_arena.openrouter_delayed_cost_reconciliation_postgres_test import _uncertain_call
    fingerprint='sha256:'+'d'*64
    late_identity=_uncertain_call(store,old_lease,hash_lease_token(old_token),label='original-late',sequence=987,amount=300000,generation_id='gen-cutoff-old',credential_fingerprint=fingerprint)
    late_candidate=next(row for row in store.list_openrouter_cost_reconciliations(ROUND,run_id=old_lease['run_id'],limit=20) if row['call_identity']==late_identity)
    h.clock.advance_to(h.schedule()['stage_2_close'])
    closed=h.service.advance_round(ROUND)
    assert closed['status']=='cancelled'
    old_round=store.get_round(ROUND)
    assert (old_round['status_generation'],old_round['stage_generation'])==(8,6)
    old_config=old_round['configuration_doc']
    assert old_config['scorer_policy']['scoring_adapter_version']=='qualification_integrity_v2'
    assert old_config['execution_sequence_policy']==contracts.BASELINE_SCORED_FIRST_POLICY
    new_config=copy.deepcopy(old_config)
    dates={field:(datetime.strptime(old_config['schedule'][field],'%Y-%m-%dT%H:%M:%SZ').replace(tzinfo=timezone.utc)+timedelta(hours=3)).strftime('%Y-%m-%dT%H:%M:%SZ') for field in NEW_DATES}
    new_config['schedule'].update(dates)
    with connect() as conn,conn.cursor() as cur:
        values=_values(cur)
        cur.execute("SELECT encode(extensions.digest(%s::jsonb::text,'sha256'),'hex')",(json.dumps(new_config),))
        values['post']=cur.fetchone()[0]
        before=_snapshot(cur)
        targets=[r for r in before['lab_arena_runs'] if r['terminal_cause']=='stage_closed']
        accepted=[r for r in before['lab_arena_runs'] if r['status']=='accepted']
        assert len(targets)==9
        assert sum(r['terminal_doc']['previous_status']=='leased' for r in targets)==1
        assert any(r['stage']==2 and r['kind']=='execute' for r in accepted)
        objects={p.relative_to(h.objects_root):p.read_bytes() for p in h.objects_root.rglob('*') if p.is_file()}
        sql=_sql(values)
        sql=re.sub(r'\b80\b',str(len(targets)),sql)
        sql=re.sub(r'\b81::BIGINT\b',f"{len(before['lab_arena_runs'])}::BIGINT",sql)
        sql=re.sub(r'\b1::BIGINT\b',f'{len(accepted)}::BIGINT',sql)
        for field,exact in NEW_DATES.items(): sql=sql.replace(exact,dates[field])
        _execute(conn,sql)
        after=_snapshot(cur)
    old_ids={r['run_id'] for r in before['lab_arena_runs']}
    assert [r for r in after['lab_arena_runs'] if r['run_id'] in old_ids]==before['lab_arena_runs']
    assert after['lab_arena_submissions']==before['lab_arena_submissions']
    assert after['lab_arena_ledger']==before['lab_arena_ledger']
    assert after['round_triggers']==before['round_triggers']
    # Closed old workers cannot submit new output; only exact late costs remain.
    h.clock.now=datetime.now(timezone.utc)
    stale=store.complete_attempt(run_id=old_lease['run_id'],lease_token_hash=hash_lease_token(old_token),result={'terminal_status':'accepted'},terminal_cause='accepted',output_ref=f'arena/{ROUND}/outputs/forbidden-stale.json')
    assert stale['status']!='accepted'
    successor_runner=h.runner(1)
    successor=successor_runner.claim_one()
    assert successor['status']=='leased' and successor['attempt']==2
    assert successor['assignment_id']==old_lease['assignment_id']
    arguments=dict(round_id=ROUND,run_id=old_lease['run_id'],call_identity=late_identity,uncertain_entry_id=late_candidate['uncertain_entry_id'],generation_id='gen-cutoff-old',credential_fingerprint=fingerprint,actual_microusd=300000,cost_units='0.3')
    wrong_key=store.reconcile_openrouter_cost(**dict(arguments,credential_fingerprint='sha256:'+'e'*64))
    assert wrong_key['status']!='settled'
    late=store.reconcile_openrouter_cost(**arguments)
    assert late['status']=='settled'
    retry_late=store.reconcile_openrouter_cost(**arguments)
    assert retry_late['status']=='settled' and retry_late['idempotent'] is True
    assert successor_runner._slots.acquire(blocking=False)
    successor_runner._run_lease(successor)
    assert successor_runner.completed[-1]['result']['status']=='accepted'
    successor_runner.close()
    retained_runner.close()
    h.clock.now=scheduled_now
    h.advance_until('published',runners=1,max_steps=60)
    with connect() as conn,conn.cursor() as cur: final=_snapshot(cur)
    for old in accepted:
        current=next(r for r in final['lab_arena_runs'] if r['run_id']==old['run_id'])
        assert {k:v for k,v in current.items() if k not in ('per_icp_score','qualification_doc','updated_at')}=={k:v for k,v in old.items() if k not in ('per_icp_score','qualification_doc','updated_at')}
        if old['qualification_doc'] is not None: assert current['qualification_doc']==old['qualification_doc']
        if old['per_icp_score'] is not None: assert current['per_icp_score']==old['per_icp_score']
    for old in targets:
        assert next(r for r in final['lab_arena_runs'] if r['run_id']==old['run_id'])==old
    for path,payload in objects.items(): assert (h.objects_root/path).read_bytes()==payload
    published=store.get_round(ROUND)
    assert published['configuration_doc']==new_config
    assert contracts.document_hash(old_config)!=contracts.document_hash(new_config)
    assert contracts.document_hash(old_config['scorer_policy'])==contracts.document_hash(new_config['scorer_policy'])
    assert published['publication_doc'] and published['publication_doc']['final_ranking']
    assert h.service.activate_reward(ROUND)['status']=='activated'
    assert observed=={'execute':20,'score':20}
    assert all(r['status']=='accepted' for r in final['lab_arena_runs'] if r['run_id'] not in {r['run_id'] for r in targets})
    costs=store.submission_costs(old_lease['submission_id'])
    assert sum(p['settled_microusd'] for p in costs['providers'])>=500000
    for participant in participants:
        public=h.service.public_results(ROUND,participant['submission_id'])
        assert public['score_status']=='published'
        assert len(public['scores']['stage_1']+public['scores']['stage_2'])==10
    with connect() as conn,conn.cursor() as cur: before_replay=_snapshot(cur)
    with connect() as conn: _execute(conn,sql)
    with connect() as conn,conn.cursor() as cur: assert _snapshot(cur)==before_replay
    (tmp_path/'recovery-transition-proof.json').write_text(json.dumps({'old_configuration_hash':contracts.document_hash(old_config),'new_configuration_hash':contracts.document_hash(new_config),'scorer_policy_hash':contracts.document_hash(old_config['scorer_policy']),'scoring_adapter':'qualification_integrity_v2','execution_sequence_policy':old_config['execution_sequence_policy'],'configured_icps':10,'participants':len(participants),'old_accepted_preserved':len(accepted),'old_failed_preserved':len(targets),'fresh_attempt2':len(targets),'new_stage_generation':7,'old_paid_microusd':200000,'late_paid_microusd':300000,'late_settlement_idempotent':True,'old_object_bytes_preserved':len(objects),'executed':observed,'publication':published['status'],'reward_activated':True,'limitation':'Fake provider transport and deterministic judge; frozen production image/provider not executed.'},indent=2)+'\n')


def test_current_sql_counts_failed_billed_cost_across_attempts_without_cap_reset(database,tmp_path):
    from tests.lab_arena.confirmed_cost_admission_postgres_test import test_confirmed_billed_failure_reaches_cap_across_attempts
    test_confirmed_billed_failure_reaches_cap_across_attempts(database,tmp_path)


@pytest.mark.parametrize('attempt,status',[(2,'accepted'),(3,'pending')])
def test_accepted_assignment_or_excess_attempt_rejected(database,attempt,status):
    mutation=f"INSERT INTO public.lab_arena_runs(run_id,assignment_id,round_id,submission_id,miner_hotkey,stage,icp_position,attempt,kind,status,stage_generation) SELECT assignment_id||':{attempt}',assignment_id,round_id,submission_id,miner_hotkey,stage,icp_position,{attempt},kind,'{status}',7 FROM public.lab_arena_runs WHERE run_id=(SELECT min(run_id) FROM public.lab_arena_runs WHERE round_id='{ROUND}')"
    _failure(database,mutation)


@pytest.mark.parametrize('trigger',['lab_arena_rounds_write_once','lab_arena_rounds_operator_hold_transition_guard'])
def test_disabled_required_guard_rejects_without_mutation(database,trigger):
    psycopg,dsn=database
    with psycopg.connect(**dsn) as conn:
        values=_seed(conn)
        with conn.cursor() as cur: cur.execute(f'ALTER TABLE public.lab_arena_rounds DISABLE TRIGGER {trigger}')
        conn.commit()
        try:
            with conn.cursor() as cur: before=_snapshot(cur)
            with pytest.raises(Exception,match='guard differs'): _execute(conn,_sql(values))
            conn.rollback()
            with conn.cursor() as cur: assert _snapshot(cur)==before
        finally:
            with conn.cursor() as cur: cur.execute(f'ALTER TABLE public.lab_arena_rounds ENABLE TRIGGER {trigger}')
            conn.commit()


def test_other_session_cannot_use_disabled_write_once_exception(database):
    psycopg,dsn=database
    with psycopg.connect(**dsn) as owner,psycopg.connect(**dsn) as other:
        _seed(owner)
        with owner.cursor() as cur:
            cur.execute('BEGIN')
            cur.execute('LOCK TABLE public.lab_arena_rounds IN ACCESS EXCLUSIVE MODE')
            cur.execute('ALTER TABLE public.lab_arena_rounds DISABLE TRIGGER lab_arena_rounds_write_once')
        with other.cursor() as cur:
            cur.execute("SET LOCAL lock_timeout='100ms'")
            with pytest.raises(Exception,match='lock timeout'):
                cur.execute("UPDATE public.lab_arena_rounds SET configuration_doc=jsonb_set(configuration_doc,'{schedule,stage_2_close}','\"2026-10-07T17:00:02Z\"') WHERE round_id=%s",(OTHER,))
        other.rollback()
        owner.rollback()
        with other.cursor() as cur:
            with pytest.raises(Exception,match='write-once'):
                cur.execute("UPDATE public.lab_arena_rounds SET configuration_doc=jsonb_set(configuration_doc,'{schedule,stage_2_close}','\"2026-10-07T17:00:02Z\"') WHERE round_id=%s",(OTHER,))
        other.rollback()



@pytest.mark.parametrize('dml,expected',[
    ("INSERT INTO public.lab_arena_ledger(entry_kind,round_id,amount_microusd,entry_doc,miner_hotkey) SELECT 'uncertain',v_round.round_id,7,'{}',miner_hotkey FROM public.lab_arena_submissions WHERE round_id=v_round.round_id LIMIT 1;",'preservation'),
    ('UPDATE public.lab_arena_ledger SET amount_microusd=amount_microusd+1;', 'append-only'),
    ('DELETE FROM public.lab_arena_ledger;', 'append-only'),
])
def test_accidental_ledger_writes_abort_entire_recovery(database,dml,expected):
    psycopg,dsn=database
    with psycopg.connect(**dsn) as conn:
        values=_seed(conn)
        with conn.cursor() as cur: before=_snapshot(cur)
        anchor='  SELECT * INTO v_after FROM public.lab_arena_rounds WHERE round_id=v_round.round_id;'
        sql=_sql(values)
        assert sql.count(anchor)==1
        sql=sql.replace(anchor,'  '+dml+'\n'+anchor)
        with pytest.raises(Exception,match=expected): _execute(conn,sql)
        conn.rollback()
        with conn.cursor() as cur: assert _snapshot(cur)==before


@pytest.mark.parametrize('mutation',[
    'ALTER TABLE public.lab_arena_ledger DISABLE TRIGGER lab_arena_ledger_append_only',
    'ALTER FUNCTION public.lab_arena_append_only_v1() SET search_path=public',
])
def test_changed_ledger_guard_rejects_without_mutation(database,mutation):
    psycopg,dsn=database
    with psycopg.connect(**dsn) as conn:
        values=_seed(conn)
        with conn.cursor() as cur:
            cur.execute("SELECT pg_get_functiondef('public.lab_arena_append_only_v1()'::regprocedure)")
            original=cur.fetchone()[0]
            cur.execute(mutation)
        conn.commit()
        try:
            with conn.cursor() as cur: before=_snapshot(cur)
            with pytest.raises(Exception,match='ledger append-only guard differs'): _execute(conn,_sql(values))
            conn.rollback()
            with conn.cursor() as cur: assert _snapshot(cur)==before
        finally:
            with conn.cursor() as cur:
                cur.execute(original)
                cur.execute('ALTER TABLE public.lab_arena_ledger ENABLE TRIGGER lab_arena_ledger_append_only')
            conn.commit()


def test_replica_session_rejects_before_exception(database):
    psycopg,dsn=database
    with psycopg.connect(**dsn) as conn:
        values=_seed(conn)
        with conn.cursor() as cur:
            before=_snapshot(cur)
            cur.execute('SET LOCAL session_replication_role=replica')
        with pytest.raises(Exception,match='origin trigger execution is required'): _execute(conn,_sql(values))
        conn.rollback()
        with conn.cursor() as cur: assert _snapshot(cur)==before


def test_200000_large_payload_ledger_rows_are_not_serialized_under_recovery_lock(database,tmp_path):
    import time
    psycopg,dsn=database
    with psycopg.connect(**dsn) as conn:
        values=_seed(conn)
        with conn.cursor() as cur:
            cur.execute("INSERT INTO public.lab_arena_ledger(entry_kind,round_id,amount_microusd,entry_doc,terminal_response,miner_hotkey) SELECT 'uncertain',%s,1,jsonb_build_object('synthetic_payload',repeat('p',32768)),jsonb_build_object('synthetic_response',repeat('r',32768)),%s FROM generate_series(1,200000)",(ROUND,hotkey(ROUND)))
        conn.commit()
        started=time.monotonic()
        _execute(conn,_sql(values))
        elapsed=time.monotonic()-started
        with conn.cursor() as cur:
            cur.execute('SELECT count(*) FROM public.lab_arena_ledger')
            assert cur.fetchone()[0]==200002
        assert elapsed<5, 'count/max recovery should not decode ledger payloads'
        (tmp_path/'ledger-volume-proof.json').write_text(json.dumps({'synthetic_rows':200000,'logical_json_payload_bytes_per_row':65536,'logical_payload_bytes_total':13107200000,'recovery_wall_seconds':elapsed,'preservation':'exact append-only guard+global write lock+global count/max before/after','limitation':'Compressible synthetic provider text; proves no JSON serialization, not exact production index/cache/I/O timing.'},indent=2)+'\n')
