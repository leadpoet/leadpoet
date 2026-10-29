"""Exact recovery preserves old rows and resumes only frozen score work."""
import hashlib
import json
from pathlib import Path
import re
from datetime import datetime, timedelta, timezone

import pytest
from tests.lab_arena import cancel_latest_call_heads533_postgres_test as previous
from tests.lab_arena import partial_judge_host_fault527_postgres_test as shape

SQL = Path(__file__).parents[2] / 'scripts/535-arena-2026-10-21-judge-continuation-recovery.sql'
RID = 'arena-2026-10-21'
SUB = 'sub-8394c3b3fceb6bf6527e4711b8ea4de3'
TARGET = RID + ':' + SUB + ':2:9:score'
database = previous.database


def digest(cur, expression):
    cur.execute("SELECT encode(extensions.digest((" + expression + ")::TEXT,'sha256'),'hex')")
    return cur.fetchone()[0]


def prepare(conn):
    cfg = shape.score_seed(conn, positions=1)
    cfg['round_id'] = RID
    cfg['mode'] = 'live'
    cfg['schedule']['stage_2_close'] = '2026-10-21T14:00:02Z'
    cfg['schedule']['final_scoring_close'] = '2026-10-21T20:30:02Z'
    cfg['schedule']['publication_deadline'] = '2026-10-21T20:30:03Z'
    runner = shape.fast.seed.RUNNER
    member = shape.fast.seed.hotkey(SUB)
    source = 'arena/' + RID + '/sources/' + SUB + '.tar.gz'
    items = []
    with conn.cursor() as cur:
        cur.execute('SET LOCAL session_replication_role=replica')
        cur.execute('TRUNCATE public.lab_arena_rounds CASCADE')
        cur.execute('DROP TABLE IF EXISTS public.lab_arena_oct21_judge_recovery535')
        cur.execute("UPDATE public.lab_arena_restart_claim_control SET operator_paused=false,pause_reason='',actor_ref='',guard_commitment='',owner_commitment='',guard_expires_at=NULL,candidate_commit='',restart_scope='',restart_phase='',captured_leases='[]'")
        cur.execute("INSERT INTO public.lab_arena_rounds(round_id,status,status_generation,stage_generation,scoring_generation_floor,configuration_doc,participants,benchmark_ref,evaluation_date,rewards_enabled,cancel_reason) VALUES(%s,'cancelled',11,9,0,%s::jsonb,%s::jsonb,%s,'2026-10-21',false,'scoring_incomplete')", (RID,json.dumps(cfg),json.dumps([dict(submission_id=SUB,miner_hotkey=member,is_king=False,source_ref=source)]),'arena/'+RID+'/benchmark.json'))
        cur.execute("INSERT INTO public.lab_arena_submissions(submission_id,round_id,miner_hotkey,status,is_king,source_ref,source_size_bytes,submission_doc,code_review_status,code_review_doc,code_review_claim,code_review_started_at,code_review_attempts) VALUES(%s,%s,%s,'frozen',false,%s,123,'{}','passed','{\"decision\":\"pass\"}','sha256:'||repeat('1',64),now(),1)",(SUB,RID,member,source))
        # Full production cardinality; fixture-only extra assignments use the
        # same admitted identity. The target retains its exact run identity.
        for i in range(168):
            assignment = TARGET.removesuffix(':score') if i == 0 else f'{RID}:{SUB}:2:{i+9}'
            run_id = assignment + ':1'
            output = 'fixture/' + run_id
            cur.execute("INSERT INTO public.lab_arena_runs(run_id,assignment_id,round_id,submission_id,miner_hotkey,stage,icp_position,attempt,status,kind,stage_generation,scoring_generation_floor,terminal_cause,output_ref,per_icp_score) VALUES(%s,%s,%s,%s,%s,2,9,1,'accepted','execute',8,0,'accepted',%s,%s)", (run_id,assignment,RID,SUB,member,output,None if i == 0 else 50))
            scope = dict(round_id=RID,scorer_image_digest=cfg['scorer_image_digest'],scorer_image_reference=cfg['scorer_image_reference'],scoring_input_hash='sha256:'+hashlib.sha256(run_id.encode()).hexdigest(),cache_key='sha256:'+hashlib.sha256(assignment.encode()).hexdigest())
            judge_assignment = assignment + ':score'
            attempts = range(1,14) if i == 0 else range(1,3) if i <= 81 else [1]
            for attempt in attempts:
                accepted = i != 0 and attempt == max(attempts)
                cause = 'accepted' if accepted else 'stage_closed' if i == 0 and attempt >= 12 else 'judge_error'
                cur.execute("INSERT INTO public.lab_arena_runs(run_id,assignment_id,round_id,submission_id,miner_hotkey,stage,icp_position,attempt,status,kind,stage_generation,scoring_generation_floor,terminal_cause,output_ref,scored_run_id,judgment_cache_key,judgment_input_hash,judgment_scope_doc,judgment_group_leader,judgment_group_miner_hotkeys,previous_runner_hotkey,created_at) VALUES(%s,%s,%s,%s,%s,2,9,%s,%s,'score',8,0,%s,%s,%s,%s,%s,%s::jsonb,true,%s,%s,clock_timestamp()-interval '20 minutes')",(judge_assignment+':'+str(attempt),judge_assignment,RID,SUB,member,attempt,'accepted' if accepted else 'failed',cause,output+':score' if accepted else None,run_id,scope['cache_key'],scope['scoring_input_hash'],json.dumps(scope),[member],runner))
            items.append(dict(scored_run_id=run_id,submission_id=SUB,icp_position=9,output_ref=output))
        for i in range(69):
            assignment = f'{RID}:{SUB}:1:failed{i}'
            cur.execute("INSERT INTO public.lab_arena_runs(run_id,assignment_id,round_id,submission_id,miner_hotkey,stage,icp_position,attempt,status,kind,stage_generation,scoring_generation_floor,terminal_cause) VALUES(%s,%s,%s,%s,%s,1,9,1,'failed','execute',7,0,'model_error')",(assignment+':1',assignment,RID,SUB,member))
        cur.execute('UPDATE public.lab_arena_rounds SET stage2_scoring_plan_doc=%s::jsonb WHERE round_id=%s',(json.dumps(dict(work_items=items,zero_rows=[],incomplete_rows=[])),RID))
        cur.execute('SET LOCAL session_replication_role=origin')
        replacements = {
            'expected_round':digest(cur,"(SELECT to_jsonb(r) FROM public.lab_arena_rounds r WHERE round_id='"+RID+"')"),
            'expected_config':digest(cur,"(SELECT configuration_doc FROM public.lab_arena_rounds WHERE round_id='"+RID+"')"),
            'expected_participants':digest(cur,"(SELECT participants FROM public.lab_arena_rounds WHERE round_id='"+RID+"')"),
        }
        state = dict(runs=digest(cur,"(SELECT jsonb_agg(to_jsonb(r) ORDER BY run_id) FROM public.lab_arena_runs r WHERE round_id='"+RID+"')"),submissions=digest(cur,"(SELECT jsonb_agg(to_jsonb(r) ORDER BY submission_id) FROM public.lab_arena_submissions r WHERE round_id='"+RID+"')"),ledger_count=0,ledger_max=None,trajectory_count=0,trajectory_max=None)
    conn.commit()
    text = SQL.read_text()
    for key,value in replacements.items():
        text = re.sub(r"("+key+r" CONSTANT TEXT:=')[^']+(';)",lambda m:m[1]+value+m[2],text)
    text = re.sub(r"(expected_state CONSTANT JSONB:=')[^']+('::JSONB;)",lambda m:m[1]+json.dumps(state)+m[2],text)
    # Only the test clock changes. The production migration keeps exact UTC bounds.
    text = text.replace("clock_timestamp()>='2026-10-21T23:59:58Z'::TIMESTAMPTZ", "'2026-10-21T22:00:00Z'::TIMESTAMPTZ>='2026-10-21T23:59:58Z'::TIMESTAMPTZ")
    # The real claim RPC uses its own clock. Project only the recovered fixture
    # admission deadline into that clock; keep the sealed initial state intact.
    target = "'\"2026-10-21T23:59:59Z\"'"
    assert text.count(target) == 1
    deadline = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
    text = text.replace(target, "'" + json.dumps(deadline) + "'")
    return text


def rows(cur):
    cur.execute("SELECT jsonb_agg(to_jsonb(r) ORDER BY run_id) FROM public.lab_arena_runs r WHERE round_id=%s",(RID,))
    return cur.fetchone()[0]


def test_preserves_all_rows_and_normal_continuation_only_adds_score(database):
    with database[0].connect(**database[1]) as conn:
        text = prepare(conn)
        with conn.cursor() as cur:
            before = rows(cur)
            cur.execute(text)
            assert rows(cur) == before
            cur.execute("SELECT status,stage_generation,status_generation,scoring_generation_floor,configuration_doc#>>'{schedule,stage_2_close}' FROM public.lab_arena_rounds WHERE round_id=%s",(RID,))
            assert cur.fetchone() == ('stage2_judged',10,12,0,'2026-10-21T14:00:02Z')
            cur.execute(text)
            assert rows(cur) == before
            cur.execute('SELECT public.lab_arena_expire_leases(%s)',(RID,))
            assert cur.fetchone()[0]['retried'] == 1
            after = rows(cur)
            new = [r for r in after if r['run_id'] not in {r['run_id'] for r in before}]
            assert len(new) == 1 and new[0]['run_id'] == TARGET+':14'
            assert new[0]['kind'] == 'score' and new[0]['status'] == 'pending'
            assert [r for r in after if r['run_id'] != TARGET+':14'] == before
            parent = next(r for r in before if r['run_id'] == TARGET+':13')
            for key in ('scored_run_id','judgment_input_hash','judgment_scope_doc','judgment_cache_key'):
                assert new[0][key] == parent[key]
            # Fixture-only clock control crosses the ordinary retry backoff.
            # The production migration does not backdate or alter any attempt.
            cur.execute('BEGIN')
            cur.execute('SET LOCAL session_replication_role=replica')
            cur.execute("UPDATE public.lab_arena_runs SET created_at=clock_timestamp()-interval '20 minutes' WHERE run_id=%s",(TARGET+':14',))
            cur.execute('SET LOCAL session_replication_role=origin')
            cur.execute('SAVEPOINT expired_fixture_admission')
            cur.execute('SET LOCAL session_replication_role=replica')
            cur.execute("UPDATE public.lab_arena_rounds SET configuration_doc=jsonb_set(configuration_doc,'{schedule,publication_deadline}',to_jsonb((clock_timestamp()-interval '1 second')::text)) WHERE round_id=%s", (RID,))
            cur.execute('SET LOCAL session_replication_role=origin')
            denied_before = rows(cur)
            assert previous.prior.queue.claim(cur,RID,runner=shape.fast.seed.RUNNER) == {'status': 'no_pending'}
            cur.execute('RESET ROLE')
            assert rows(cur) == denied_before
            cur.execute('ROLLBACK TO SAVEPOINT expired_fixture_admission')
            cur.execute("SELECT (configuration_doc#>>'{schedule,publication_deadline}')::timestamptz>clock_timestamp() FROM public.lab_arena_rounds WHERE round_id=%s", (RID,))
            assert cur.fetchone()[0] is True
            lease = previous.prior.queue.claim(cur,RID,runner=shape.fast.seed.RUNNER)
            assert lease['status'] == 'leased' and lease['run_id'] == TARGET+':14', lease
            cur.execute('RESET ROLE')
            claimed = rows(cur)
            assert [r for r in claimed if r['kind']=='execute'] == [r for r in before if r['kind']=='execute']
            assert [r for r in claimed if r['run_id'] != TARGET+':14'] == before
            cur.execute("SELECT has_table_privilege('lab_arena_service','public.lab_arena_oct21_judge_recovery535','SELECT'),has_table_privilege('anon','public.lab_arena_oct21_judge_recovery535','SELECT')")
            assert cur.fetchone() == (False, False)
            cur.execute("SELECT bool_or(public.lab_arena__baseline_recovery_phase_v1(r,e)) FROM public.lab_arena_rounds r JOIN public.lab_arena_runs e USING(round_id) WHERE r.round_id=%s AND e.kind='execute'",(RID,))
            assert cur.fetchone()[0] is False


@pytest.mark.parametrize('mutation',[
    "UPDATE public.lab_arena_rounds SET stage_generation=10 WHERE round_id='"+RID+"'",
    "UPDATE public.lab_arena_rounds SET scoring_generation_floor=1 WHERE round_id='"+RID+"'",
    "UPDATE public.lab_arena_runs SET status='leased',terminal_cause=NULL WHERE run_id='"+TARGET+":13'",
    "UPDATE public.lab_arena_runs SET terminal_cause='credential_error' WHERE run_id='"+TARGET+":13'",
])
def test_changed_preimage_refuses_without_partial_recovery(database,mutation):
    with database[0].connect(**database[1]) as conn:
        text = prepare(conn)
        with conn.cursor() as cur:
            cur.execute('SET LOCAL session_replication_role=replica')
            cur.execute(mutation)
            cur.execute('SET LOCAL session_replication_role=origin')
        conn.commit()
        with conn.cursor() as cur:
            before = rows(cur)
            with pytest.raises(database[0].Error):
                cur.execute(text)
            conn.rollback()
            assert rows(cur) == before
            cur.execute("SELECT status FROM public.lab_arena_rounds WHERE round_id=%s",(RID,))
            assert cur.fetchone()[0] == 'cancelled'
            cur.execute("SELECT tgenabled FROM pg_trigger WHERE tgrelid='public.lab_arena_rounds'::regclass AND tgname='lab_arena_rounds_write_once'")
            assert cur.fetchone()[0] == 'O'


def test_expired_recovery_window_refuses(database):
    with database[0].connect(**database[1]) as conn:
        text = prepare(conn).replace("'2026-10-21T22:00:00Z'::TIMESTAMPTZ>=", "'2026-10-22T01:00:00Z'::TIMESTAMPTZ>=")
        with conn.cursor() as cur:
            before = rows(cur)
            with pytest.raises(database[0].Error, match='exact state or window differs'):
                cur.execute(text)
            conn.rollback()
            assert rows(cur) == before


def test_open_call_prevents_recovery(database):
    with database[0].connect(**database[1]) as conn:
        text = prepare(conn)
        with conn.cursor() as cur:
            previous.prior.add_head(cur,RID,'reservation')
        conn.commit()
        with conn.cursor() as cur:
            with pytest.raises(database[0].Error, match='not drained or is superseded'):
                cur.execute(text)
            conn.rollback()


def test_newer_published_day_prevents_recovery(database):
    with database[0].connect(**database[1]) as conn:
        text = prepare(conn)
        with conn.cursor() as cur:
            cur.execute('SET LOCAL session_replication_role=replica')
            cur.execute("INSERT INTO public.lab_arena_rounds(round_id,status,configuration_doc,evaluation_date,rewards_enabled) SELECT 'arena-2026-10-22','published',jsonb_set(configuration_doc,'{round_id}','\"arena-2026-10-22\"'),'2026-10-22',false FROM public.lab_arena_rounds WHERE round_id=%s",(RID,))
            cur.execute('SET LOCAL session_replication_role=origin')
        conn.commit()
        with conn.cursor() as cur:
            with pytest.raises(database[0].Error, match='not drained or is superseded'):
                cur.execute(text)
            conn.rollback()


def test_audit_is_append_only_and_guard_security_drift_refuses(database):
    with database[0].connect(**database[1]) as conn:
        text = prepare(conn)
        with conn.cursor() as cur:
            cur.execute('ALTER FUNCTION public.lab_arena_rounds_write_once_v1() SECURITY DEFINER')
            with pytest.raises(database[0].Error, match='round guard differs'):
                cur.execute(text)
            conn.rollback()
            cur.execute(text)
            cur.execute('BEGIN')
            cur.execute('SAVEPOINT audit_write')
            with pytest.raises(database[0].Error):
                cur.execute("UPDATE public.lab_arena_oct21_judge_recovery535 SET before_round='{}'")
            cur.execute('ROLLBACK TO SAVEPOINT audit_write')
            cur.execute("SELECT count(*) FROM public.lab_arena_oct21_judge_recovery535")
            assert cur.fetchone()[0] == 1


def test_disabled_ledger_append_only_guard_prevents_recovery(database):
    with database[0].connect(**database[1]) as conn:
        text = prepare(conn)
        with conn.cursor() as cur:
            cur.execute('ALTER TABLE public.lab_arena_ledger DISABLE TRIGGER lab_arena_ledger_append_only')
            with pytest.raises(database[0].Error, match='ledger append-only guard differs'):
                cur.execute(text)
            conn.rollback()
            cur.execute("SELECT tgenabled FROM pg_trigger WHERE tgrelid='public.lab_arena_ledger'::regclass AND tgname='lab_arena_ledger_append_only'")
            assert cur.fetchone()[0] == 'O'
