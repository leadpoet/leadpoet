"""A proved judge gap can omit an unmeasured miner without resolving its bill."""
from pathlib import Path
import json

import pytest

from lab_arena.store import ArenaStoreError
from tests.lab_arena.lab_arena_pg_harness import current_service_database, SCRIPTS
from tests.lab_arena import incomplete_participant_publication540_postgres_test as prior
from tests.lab_arena.nonempty_unjudged_publication_postgres_test import nonempty_gap, uncertain

SQL = SCRIPTS / '664-lab-arena-unscored-uncertain-publication.sql'
REASON = 'public.lab_arena__participant_incomplete_reason_v1(text,text)'
STRICT = 'public.lab_arena__execution_privacy_route_unavailable_v1(text)'
PRE = '4066cdc359bc23fedcf1efc7a002410231bb9f79080bad3e7f7b2c45cdd1124f'


@pytest.fixture(scope='module')
def database():
    gen = current_service_database()
    db = next(gen)
    try:
        with db[0].connect(**db[1]) as conn, conn.cursor() as cur:
            # Install the committed 604 helper/function upgrades, without its
            # November3 historical data operator. This is a fixture boundary.
            text = (SCRIPTS / '604-arena-2026-11-03-forward-only-verifier.sql').read_text()
            cur.execute(text[:text.index('DO $nov03_forward604$')] + text[text.index('DO $restore604$'):])
            cur.execute('SELECT pg_get_functiondef(%s::regprocedure),encode(extensions.digest(prosrc,\'sha256\'),\'hex\') FROM pg_proc WHERE oid=%s::regprocedure', (REASON, REASON))
            definition, digest = cur.fetchone()
            assert digest == PRE
            cur.execute(SQL.read_text())
            cur.execute(SQL.read_text())
        yield (*db, definition)
    finally:
        gen.close()


@pytest.fixture(scope='module')
def completed(database, tmp_path_factory):
    gen = prior.completed.__wrapped__(database[:2], tmp_path_factory)
    try:
        h, _, _ = next(gen)
        # Current native publication materializes legitimate baseline projections.
        # Finish that existing step before freezing the fixture's immutable rows.
        h.service._record_recovered_judgment_scores(h.service.store.get_round(h.round_id))
        with database[0].connect(**database[1]) as conn, conn.cursor() as cur:
            cur.execute('SELECT to_jsonb(r) FROM public.lab_arena_rounds r WHERE round_id=%s', (h.round_id,))
            row = cur.fetchone()[0]
            cur.execute('SELECT to_jsonb(r) FROM public.lab_arena_runs r WHERE round_id=%s ORDER BY run_id', (h.round_id,))
            runs = [x[0] for x in cur.fetchall()]
        yield h, row, runs
    finally:
        gen.close()


@pytest.fixture(autouse=True)
def isolated_native_receipts(database):
    # Tamper controls can deliberately move a row outside its old round key.
    # This disposable database has no production rows; remove its synthetic
    # OpenRouter receipts between cases without changing any live guard.
    with database[0].connect(**database[1]) as conn, conn.cursor() as cur:
        cur.execute('SET LOCAL session_replication_role=replica')
        cur.execute("DELETE FROM public.lab_arena_ledger WHERE provider='openrouter'")
        cur.execute('SET LOCAL session_replication_role=origin')


def seed(database, completed, *, baseline=False, empty=False, deadline=-1):
    h = prior.reset(database[:2], completed, deadline=deadline)
    run = (next(r for r in completed[2] if r['kind']=='execute' and r['stage']==1)
           if baseline else prior.selected(completed, 'MissingJudge', position=1))
    with database[0].connect(**database[1]) as conn, conn.cursor() as cur:
        if empty:
            prior.gap(cur, run)
        else:
            nonempty_gap(h, cur, run, 'uncertain664')
        uncertain(cur, run)
        before = prior.state(cur, h.round_id)
    if not baseline:
        assert run['stage'] == 2
    return h, run, before


def reason(cur, h, run):
    cur.execute('SELECT public.lab_arena__participant_incomplete_reason_v1(%s,%s)',
                (h.round_id, run['submission_id']))
    return cur.fetchone()[0]


def test_old_participant_helper_rejects_the_exact_valid_unknown_bill(database, completed):
    h, run, _ = seed(database, completed)
    with database[0].connect(**database[1]) as conn, conn.cursor() as cur:
        cur.execute(database[2])
        with pytest.raises(database[0].Error, match='Arena execution receipt binding invalid'):
            reason(cur, h, run)
        conn.rollback()  # Restore664, including its security, without changing data.


def test_valid_unknown_bill_remains_null_ineligible_and_unchanged(database, completed):
    h, run, before = seed(database, completed)
    cost = h.service.store.submission_costs(run['submission_id'])
    assert any(p['reserved_or_uncertain_microusd'] >= 1234 for p in cost['providers'])
    with database[0].connect(**database[1]) as conn, conn.cursor() as cur:
        assert reason(cur, h, run) == 'judge_error'
        # The strict privacy proof has not been weakened for any caller.
        with pytest.raises(database[0].Error, match='Arena execution receipt binding invalid'):
            cur.execute('SELECT public.lab_arena__execution_privacy_route_unavailable_v1(%s)', (run['run_id'],))
    assert h.service.publish(h.round_id)['status'] == 'ok'
    row = h.service.store.get_round(h.round_id)
    entry = next(x for x in row['publication_doc']['final_ranking'] if x['submission_id']==run['submission_id'])
    assert entry['final_score'] is None and entry['cost_summary'] is None
    assert entry['eligible'] is False and entry['evaluation']=='not_evaluated'
    assert entry['evaluation_reason']=='judge_error'
    assert row['reward_activated_at'] is None
    assert h.service.store.submission_costs(run['submission_id']) == cost
    with database[0].connect(**database[1]) as conn, conn.cursor() as cur:
        assert prior.state(cur, h.round_id) == before


@pytest.mark.parametrize('fault', [
    'missing_dispatch','duplicate_uncertain','wrong_order','wrong_operation','wrong_action',
    'wrong_hash','all_wrong_round','all_wrong_submission','all_wrong_miner','all_wrong_stage',
    'dispatch_wrong_run','dispatch_wrong_funding','latest_wrong_head','malformed_native',
])
def test_malformed_unknown_binding_is_never_ignored(database, completed, fault):
    h, run, _ = seed(database, completed)
    with database[0].connect(**database[1]) as conn, conn.cursor() as cur:
        cur.execute('SET LOCAL session_replication_role=replica')
        args=(run['run_id'],)
        if fault=='missing_dispatch':
            cur.execute("DELETE FROM public.lab_arena_ledger WHERE run_id=%s AND entry_kind='dispatch'",args)
        elif fault=='duplicate_uncertain':
            with pytest.raises(database[0].errors.UniqueViolation):
                cur.execute("INSERT INTO public.lab_arena_ledger(entry_kind,miner_hotkey,round_id,submission_id,run_id,stage,call_identity,provider,operation_id,funding_source,amount_microusd,entry_doc) SELECT entry_kind,miner_hotkey,round_id,submission_id,run_id,stage,call_identity,provider,operation_id,funding_source,amount_microusd,entry_doc FROM public.lab_arena_ledger WHERE run_id=%s AND entry_kind='uncertain'",args)
            conn.rollback()
            return  # Native uniqueness rejects this impossible lifecycle.
        elif fault=='wrong_order':
            cur.execute("UPDATE public.lab_arena_ledger SET entry_id=entry_id+1000000 WHERE run_id=%s AND entry_kind='dispatch'",args)
        elif fault=='wrong_operation':
            cur.execute("UPDATE public.lab_arena_ledger SET operation_id='deepline.execute' WHERE run_id=%s",args)
        elif fault=='wrong_action':
            cur.execute("UPDATE public.lab_arena_ledger SET entry_doc=jsonb_set(entry_doc,'{action_sequence}','\"3\"') WHERE run_id=%s AND entry_kind='reservation'",args)
        elif fault=='wrong_hash':
            cur.execute("UPDATE public.lab_arena_ledger SET entry_doc=jsonb_set(entry_doc,'{request_hash}','\"bad\"') WHERE run_id=%s AND entry_kind='reservation'",args)
        elif fault.startswith('all_wrong_'):
            field,value={'all_wrong_round':('round_id','wrong-round'),'all_wrong_submission':('submission_id','wrong-submission'),'all_wrong_miner':('miner_hotkey','wrong-miner'),'all_wrong_stage':('stage',1)}[fault]
            cur.execute(f'UPDATE public.lab_arena_ledger SET {field}=%s WHERE run_id=%s',(value,run['run_id']))
        elif fault=='dispatch_wrong_run':
            cur.execute("UPDATE public.lab_arena_ledger SET run_id='wrong-run' WHERE run_id=%s AND entry_kind='dispatch'",args)
        elif fault=='dispatch_wrong_funding':
            cur.execute("UPDATE public.lab_arena_ledger SET funding_source='host' WHERE run_id=%s AND entry_kind='dispatch'",args)
        elif fault=='latest_wrong_head':
            cur.execute("UPDATE public.lab_arena_ledger SET entry_kind='recovery' WHERE run_id=%s AND entry_kind='uncertain'",args)
        else:
            cur.execute("UPDATE public.lab_arena_ledger SET terminal_response='{}' WHERE run_id=%s AND entry_kind='uncertain'",args)
        cur.execute('SET LOCAL session_replication_role=origin')
        before = prior.state(cur,h.round_id)
    if fault == 'wrong_order':
        with database[0].connect(**database[1]) as conn, conn.cursor() as cur:
            with pytest.raises(database[0].Error, match='receipt'):
                reason(cur, h, run)
        assert h.service.publish(h.round_id) == {'status':'retry','round_status':'scored','reason':'judge_incomplete'}
    else:
        with pytest.raises(ArenaStoreError, match='receipt|issuance'):
            h.service.publish(h.round_id)
    assert h.service.store.get_round(h.round_id)['status']=='scored'
    with database[0].connect(**database[1]) as conn, conn.cursor() as cur:
        assert prior.state(cur,h.round_id)==before


@pytest.mark.parametrize('status,message,expected',[
 (404,'No endpoints found matching your data policy (Zero data retention).','openrouter_zdr_route_unavailable'),
 (404,'Requested model does not exist','judge_error'),(403,'Forbidden','judge_error')])
def test_genuine_native_privacy_reason_is_preserved(database,completed,status,message,expected):
    h=prior.reset(database[:2],completed)
    run=prior.selected(completed,'MissingJudge',position=1)
    with database[0].connect(**database[1])as conn,conn.cursor()as cur:
        prior.gap(cur,run);prior.native(cur,run,status=status,message=message)
        assert reason(cur,h,run)==expected
    assert h.service.publish(h.round_id)['status']=='ok'
    entry=next(x for x in h.service.store.get_round(h.round_id)['publication_doc']['final_ranking']if x['submission_id']==run['submission_id'])
    assert entry['evaluation_reason']==expected and entry['final_score'] is None


def test_baseline_and_early_deadline_are_not_omission_authority(database,completed):
    h,run,before=seed(database,completed,baseline=True,deadline=60)
    with database[0].connect(**database[1])as conn,conn.cursor()as cur:
        assert reason(cur,h,run)is None
    assert h.service.publish(h.round_id)['reason']=='baseline_incomplete'
    with database[0].connect(**database[1])as conn,conn.cursor()as cur:
        assert prior.state(cur,h.round_id)==before
    h,run,_=seed(database,completed,deadline=60)
    with database[0].connect(**database[1])as conn,conn.cursor()as cur:
        assert reason(cur,h,run)is None
    assert h.service.publish(h.round_id)['reason']=='judge_incomplete'


def test_scored_and_accepted_judge_never_gain_omission_authority(database,completed):
    h=prior.reset(database[:2],completed)
    run=prior.selected(completed,'HistoricalRoute',position=1)
    with database[0].connect(**database[1])as conn,conn.cursor()as cur:
        uncertain(cur,run)
        with pytest.raises(database[0].Error,match='receipt binding invalid'):
            reason(cur,h,run)
    h=prior.reset(database[:2],completed)
    with database[0].connect(**database[1])as conn,conn.cursor()as cur:
        cur.execute('SET LOCAL session_replication_role=replica')
        cur.execute('UPDATE public.lab_arena_runs SET per_icp_score=NULL,qualification_doc=NULL WHERE run_id=%s',(run['run_id'],))
        cur.execute('SET LOCAL session_replication_role=origin')
        uncertain(cur,run)
        assert reason(cur,h,run) is None  # Accepted judge requires score recovery.


def test_stage_one_miner_does_not_enter_unknown_branch(database,completed):
    h,run,_=seed(database,completed)
    with database[0].connect(**database[1])as conn,conn.cursor()as cur:
        cur.execute('SET LOCAL session_replication_role=replica')
        cur.execute('UPDATE public.lab_arena_runs SET stage=1 WHERE run_id=%s',(run['run_id'],))
        cur.execute('UPDATE public.lab_arena_ledger SET stage=1 WHERE run_id=%s',(run['run_id'],))
        cur.execute('SET LOCAL session_replication_role=origin')
        with pytest.raises(database[0].Error,match='receipt binding invalid'):
            reason(cur,h,run)


def test_migration_replay_security_and_wrong_preimage(database,completed):
    with database[0].connect(**database[1])as conn,conn.cursor()as cur:
        cur.execute('SELECT to_jsonb(p)FROM pg_proc p WHERE oid=ANY(ARRAY[%s::regprocedure,%s::regprocedure])ORDER BY oid',(REASON,STRICT));before=cur.fetchall()
        cur.execute(SQL.read_text())
        cur.execute('SELECT to_jsonb(p)FROM pg_proc p WHERE oid=ANY(ARRAY[%s::regprocedure,%s::regprocedure])ORDER BY oid',(REASON,STRICT));assert cur.fetchall()==before
        cur.execute('SELECT pg_get_functiondef(%s::regprocedure)',(REASON,));definition=cur.fetchone()[0]
        cur.execute(definition.replace('route_unknown:=FALSE;','route_unknown:=FALSE; --foreign'))
        with pytest.raises(database[0].Error,match='preimage differs'):
            cur.execute(SQL.read_text())
        conn.rollback()
