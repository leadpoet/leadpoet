"""Real publication and funding gates accept the frozen company-cache protocol."""
from __future__ import annotations

import copy
import json
from dataclasses import replace
from datetime import datetime,timezone
from pathlib import Path

import pytest

from lab_arena import contracts
from lab_arena.store import ArenaStore,ArenaStoreError,PsycopgTransport
from tests.lab_arena import legacy_company_cache501_postgres_test as prior
from tests.lab_arena import miner_credit_terminal484_postgres_test as credit
from tests.lab_arena import test_lab_arena_service_round as flows
from tests.lab_arena.baseline_scored_first_postgres_test import _baseline_first_judge,_enable_verified_proxy_runtime
from tests.lab_arena.icp_fixtures import daily_icps
from tests.lab_arena.test_integrity_round import IntegrityHarness

SQL=Path(__file__).parents[2]/'scripts/502-lab-arena-company-judgment-finality.sql'
SIGNATURES=('public.lab_arena__all_judgments_ready_v1(text)','public.lab_arena_miner_credit_failure_v1(text)')


@pytest.fixture(scope='module')
def database():
    gen=prior.database.__wrapped__();db=next(gen)
    try:yield db
    finally:gen.close()


@pytest.fixture(scope='module')
def original_functions(database):
    with database[0].connect(**database[1]) as conn,conn.cursor() as cur:
        cur.execute('SELECT pg_get_functiondef(oid) FROM pg_proc WHERE oid=ANY(%s::regprocedure[]) ORDER BY oid',(list(SIGNATURES),))
        return [r[0] for r in cur.fetchall()]


@pytest.fixture(autouse=True)
def restore_functions(database,original_functions):
    with database[0].connect(**database[1]) as conn,conn.cursor() as cur:
        for definition in original_functions:cur.execute(definition)


def apply(conn):
    with conn.cursor() as cur:cur.execute(SQL.read_text())
    conn.commit()


def catalog(cur):
    return prior.latest.catalog(cur)


def ready(conn,rid):
    with conn.cursor() as cur:
        cur.execute('SELECT public.lab_arena__baseline_final_ready_v1(%s),public.lab_arena__all_judgments_ready_v1(%s)',(rid,rid))
        return cur.fetchone()


@pytest.fixture(scope='module')
def publication_seed(database,tmp_path_factory):
    pg,dsn=database
    harness=IntegrityHarness(lambda:pg.connect(**dsn),tmp_path_factory.mktemp('company-finality502'),challengers=[],runners=['alpha'])
    harness.service.config.defaults=replace(harness.service.config.defaults,
        execution_sequence_from='2000-01-01T00:00:00Z',per_icp_cost_policy=True,
        rewards_enabled=False,benchmark_icp_count=2)
    harness.service.config.daily_icp_source=lambda **kwargs:{'status':'ready','set_id':kwargs['set_id'],'icps':daily_icps()[:2]}
    _enable_verified_proxy_runtime(harness)
    monkey=pytest.MonkeyPatch()
    monkey.setattr(flows,'deterministic_scorer',_baseline_first_judge)
    try:
        assert flows._start_round(harness,day=29,epoch=62429)==1
        harness.clock.advance_to(harness.schedule()['stage_1_start'])
        harness.advance_until('scored',runners=1)
        captured={}
        transition=harness.service.store.transition_round
        def capture(*args):captured['args']=copy.deepcopy(args);return {'status':'captured'}
        monkey.setattr(harness.service.store,'transition_round',capture)
        assert harness.service.publish(harness.round_id)['status']=='captured'
        monkey.setattr(harness.service.store,'transition_round',transition)
        with pg.connect(**dsn) as conn,conn.cursor() as cur:
            tables={}
            for table in ('lab_arena_rounds','lab_arena_runs'):
                cur.execute(f'SELECT to_jsonb(t) FROM public.{table} t WHERE round_id=%s ORDER BY 1::text',(harness.round_id,))
                tables[table]=[r[0] for r in cur.fetchall()]
        return dict(round_id=harness.round_id,patch=captured['args'][3],tables=tables)
    finally:monkey.undo()


def reset_publication(conn,seed,*,company=True,empty=False):
    with conn.cursor() as cur:
        cur.execute('SET LOCAL session_replication_role=replica')
        for table,rows in seed['tables'].items():
            key='round_id' if table=='lab_arena_rounds' else 'run_id'
            for row in rows:
                cur.execute('SELECT attname FROM pg_attribute WHERE attrelid=%s::regclass AND attgenerated<>%s',(table,''))
                generated={r[0] for r in cur.fetchall()}
                names=[name for name in row if name not in generated]
                cur.execute(f"UPDATE public.{table} SET ({','.join(names)})=(SELECT {','.join(names)} FROM jsonb_populate_record(NULL::public.{table},%s::jsonb)) WHERE {key}=%s",(json.dumps(row),row[key]))
        cur.execute('SELECT configuration_doc,evaluation_date,arena_network_name,arena_netuid FROM public.lab_arena_rounds WHERE round_id=%s',(seed['round_id'],))
        cfg,date,network,netuid=cur.fetchone()
        if company:
            cfg['company_judgment_policy']=prior.POLICY
            cfg['scorer_policy']['company_judgment_policy']=prior.POLICY
            cur.execute('UPDATE public.lab_arena_rounds SET configuration_doc=%s::jsonb WHERE round_id=%s',(json.dumps(cfg),seed['round_id']))
            for run in seed['tables']['lab_arena_runs']:
                if run['kind']!='score':continue
                ref=prior.scope_original_safe(seed['round_id'],run['run_id'],0)
                doc=ref['scope_doc'];doc.pop('cache_key')
                doc.update(evaluation_date=date,network_name=network,netuid=netuid,
                    scorer_image_digest=cfg['scorer_image_digest'],scorer_image_reference=cfg['scorer_image_reference'])
                ref['cache_key']=doc['cache_key']=contracts.document_hash(doc)
                cur.execute('UPDATE public.lab_arena_runs SET company_judgment_refs=%s::jsonb,judgment_cache_key=NULL,judgment_input_hash=NULL,judgment_scope_doc=NULL WHERE run_id=%s',
                    (json.dumps([] if empty else [ref]),run['run_id']))
        cur.execute('SET LOCAL session_replication_role=origin')
    conn.commit()


def publish(conn,seed,patch=None):
    return prior.baseline.rpc(conn,'lab_arena_transition_round(%s,%s,%s,%s::jsonb)',
        (seed['round_id'],'scored','published',json.dumps(seed['patch'] if patch is None else patch)))


@pytest.mark.parametrize('empty',[False,True])
def test_company_mode_publication_reproduces_old_guard_failure_then_passes_actual_transition(database,publication_seed,empty):
    with database[0].connect(**database[1]) as conn:
        reset_publication(conn,publication_seed,empty=empty)
        assert ready(conn,publication_seed['round_id'])==(True,False)
        with pytest.raises(database[0].Error,match='lab_arena_final_baseline_incomplete'):
            publish(conn,publication_seed)
        conn.rollback()
        with conn.cursor() as cur:before=prior.latest.rows(cur)
        apply(conn)
        assert ready(conn,publication_seed['round_id'])==(True,True)
        with conn.cursor() as cur:assert prior.latest.rows(cur)==before
        assert publish(conn,publication_seed)['status']=='ok'
        with conn.cursor() as cur:
            cur.execute('SELECT status,publication_doc FROM public.lab_arena_rounds WHERE round_id=%s',(publication_seed['round_id'],))
            assert cur.fetchone()==('published',publication_seed['patch']['publication_doc'])


@pytest.mark.parametrize('fault',['bad_scope','image','reference','whole_key','whole_input','whole_scope','missing_judge','unaccepted_judge','wrong_member','wrong_position','no_score','no_output','unknown_policy','mismatched_policy'])
def test_finality_rejects_incomplete_unbound_or_different_frozen_scope(database,publication_seed,fault):
    with database[0].connect(**database[1]) as conn:
        reset_publication(conn,publication_seed)
        apply(conn)
        with conn.cursor() as cur:
            cur.execute('SET LOCAL session_replication_role=replica')
            where="WHERE run_id=(SELECT run_id FROM public.lab_arena_runs WHERE kind='score' AND status='accepted' ORDER BY run_id LIMIT 1)"
            if fault in ('bad_scope','image','reference'):
                field={'bad_scope':'company_input_hash','image':'scorer_image_digest','reference':'scorer_image_reference'}[fault]
                cur.execute(f"UPDATE public.lab_arena_runs SET company_judgment_refs=jsonb_set(company_judgment_refs,'{{0,scope_doc,{field}}}','\"wrong\"'::jsonb) "+where)
            elif fault.startswith('whole_'):
                field={'whole_key':'judgment_cache_key','whole_input':'judgment_input_hash','whole_scope':'judgment_scope_doc'}[fault]
                value="'{}'::jsonb" if field=='judgment_scope_doc' else "'sha256:'||repeat('f',64)"
                cur.execute(f'UPDATE public.lab_arena_runs SET {field}={value} '+where)
            elif fault in ('missing_judge','unaccepted_judge'):
                cur.execute("UPDATE public.lab_arena_runs SET status='failed',terminal_cause='judge_error' "+where)
            elif fault=='wrong_member':cur.execute("UPDATE public.lab_arena_runs SET miner_hotkey=%s "+where,(prior.original.hotkey('foreign-finality502'),))
            elif fault=='wrong_position':cur.execute('UPDATE public.lab_arena_runs SET icp_position=icp_position+1 '+where)
            elif fault=='no_score':cur.execute("UPDATE public.lab_arena_runs SET per_icp_score=NULL WHERE kind='execute'")
            elif fault=='no_output':cur.execute('UPDATE public.lab_arena_runs SET output_ref=NULL '+where)
            else:
                path='{company_judgment_policy}' if fault=='unknown_policy' else '{scorer_policy,company_judgment_policy}'
                cur.execute(f"UPDATE public.lab_arena_rounds SET configuration_doc=jsonb_set(configuration_doc,'{path}','\"unknown\"'::jsonb)")
                # A forged legacy image scope cannot become a fallback witness.
                cur.execute("UPDATE public.lab_arena_runs r SET judgment_scope_doc=jsonb_build_object('scorer_image_digest',a.configuration_doc->>'scorer_image_digest','scorer_image_reference',a.configuration_doc->>'scorer_image_reference') FROM public.lab_arena_rounds a WHERE a.round_id=r.round_id AND r.kind='score'")
            cur.execute('SET LOCAL session_replication_role=origin')
            before=prior.latest.rows(cur)
        conn.commit()
        assert ready(conn,publication_seed['round_id'])[1] is False
        with pytest.raises(database[0].Error,match='lab_arena_final_baseline_incomplete'):
            publish(conn,publication_seed)
        conn.rollback()
        with conn.cursor() as cur:assert prior.latest.rows(cur)==before


def test_legacy_scope_publication_still_passes(database,publication_seed):
    with database[0].connect(**database[1]) as conn:
        reset_publication(conn,publication_seed,company=False)
        apply(conn)
        assert ready(conn,publication_seed['round_id'])==(True,True)
        assert publish(conn,publication_seed)['status']=='ok'



def test_service_role_cannot_change_accepted_company_refs(database,publication_seed):
    with database[0].connect(**database[1]) as conn:
        reset_publication(conn,publication_seed)
        with conn.cursor() as cur:
            before=prior.latest.rows(cur)
            cur.execute('SET ROLE lab_arena_service')
            with pytest.raises(database[0].Error) as error:
                cur.execute("UPDATE public.lab_arena_runs SET company_judgment_refs='[]'::jsonb WHERE round_id=%s AND kind='score'",(publication_seed['round_id'],))
            assert error.value.pgcode=='42501'
        conn.rollback()
        with conn.cursor() as cur:assert prior.latest.rows(cur)==before


def test_existing_publication_score_guard_still_rejects_forgery(database,publication_seed):
    with database[0].connect(**database[1]) as conn:
        reset_publication(conn,publication_seed)
        apply(conn)
        wrong=copy.deepcopy(publication_seed['patch'])
        wrong['publication_doc']['final_ranking'][0]['final_score']+=1
        with conn.cursor() as cur:before=prior.latest.rows(cur)
        with pytest.raises(database[0].Error):publish(conn,publication_seed,wrong)
        conn.rollback()
        with conn.cursor() as cur:assert prior.latest.rows(cur)==before


def company_credit_seed(conn,*,funding='miner_key',proof=True,baseline=False):
    assignment,source=credit.seed(conn,funding=funding,proof=proof,baseline=baseline)
    with conn.cursor() as cur:
        cur.execute('SET LOCAL session_replication_role=replica')
        cur.execute('SELECT configuration_doc,evaluation_date,arena_network_name,arena_netuid FROM public.lab_arena_rounds')
        cfg,date,network,netuid=cur.fetchone()
        cfg.pop('contact_policy',None)
        cfg['scorer_policy']['scoring_adapter_version']='qualification_integrity_v2'
        cfg['company_judgment_policy']=prior.POLICY
        cfg['scorer_policy']['company_judgment_policy']=prior.POLICY
        cur.execute('UPDATE public.lab_arena_rounds SET configuration_doc=%s::jsonb',(json.dumps(cfg),))
        ref=prior.scope_original_safe(credit.base.ROUND,'credit502',0)
        doc=ref['scope_doc'];doc.pop('cache_key')
        doc.update(evaluation_date=date,network_name=network,netuid=netuid,
            scorer_image_digest=cfg['scorer_image_digest'],scorer_image_reference=cfg['scorer_image_reference'])
        ref['cache_key']=doc['cache_key']=contracts.document_hash(doc)
        cur.execute("UPDATE public.lab_arena_runs SET company_judgment_refs=%s::jsonb,judgment_cache_key=NULL,judgment_input_hash=NULL,judgment_scope_doc=NULL WHERE kind='score'",(json.dumps([ref]),))
        cur.execute('SET LOCAL session_replication_role=origin')
    conn.commit()
    return assignment,source,[ref]


@pytest.mark.parametrize('fault',[None,'host','unproved','baseline','image','scope','unknown_policy','wrong_owner','unbound_receipt','inflight'])
def test_miner_credit_scope_switch_preserves_original_funding_proof(database,fault):
    with database[0].connect(**database[1]) as conn:
        assignment,source,refs=company_credit_seed(conn,funding='host' if fault=='host' else 'miner_key',proof=fault!='unproved',baseline=fault=='baseline')
        assert not credit.proved(conn,assignment+':2')
        apply(conn)
        with conn.cursor() as cur:
            cur.execute('SET LOCAL session_replication_role=replica')
            if fault in ('image','scope'):
                field='scorer_image_digest' if fault=='image' else 'company_input_hash'
                cur.execute(f"UPDATE public.lab_arena_runs SET company_judgment_refs=jsonb_set(company_judgment_refs,'{{0,scope_doc,{field}}}','\"wrong\"'::jsonb) WHERE run_id=%s",(assignment+':2',))
            elif fault=='unknown_policy':
                cur.execute("UPDATE public.lab_arena_rounds SET configuration_doc=jsonb_set(configuration_doc,'{company_judgment_policy}','\"unknown\"'::jsonb)")
                # Even otherwise valid legacy scope cannot bypass a bad marker.
                item=credit.base.scope_item(conn,source)
                cur.execute('UPDATE public.lab_arena_runs SET judgment_scope_doc=%s::jsonb,judgment_cache_key=%s,judgment_input_hash=%s WHERE run_id=%s',
                    (json.dumps(item['judgment_scope_doc']),item['judgment_cache_key'],item['judgment_input_hash'],assignment+':2'))
            elif fault=='wrong_owner':cur.execute('UPDATE public.lab_arena_ledger SET miner_hotkey=%s WHERE run_id=%s',(prior.original.hotkey('credit502-foreign'),assignment+':2'))
            elif fault=='unbound_receipt':cur.execute('UPDATE public.lab_arena_ledger SET call_identity=NULL WHERE run_id=%s',(assignment+':2',))
            elif fault=='inflight':cur.execute("DELETE FROM public.lab_arena_ledger WHERE run_id=%s AND entry_kind='settlement'",(assignment+':2',))
            cur.execute('SET LOCAL session_replication_role=origin')
            before=prior.latest.rows(cur)
        conn.commit()
        assert credit.proved(conn,assignment+':2') is (fault is None)
        assert credit.terminal(conn,source) is (fault is None)
        with conn.cursor() as cur:assert prior.latest.rows(cur)==before
        if fault is None:
            with conn.cursor() as cur:
                cur.execute('SELECT public.lab_arena__all_judgments_ready_v1(%s)',(credit.base.ROUND,))
                assert cur.fetchone()==(True,)


@pytest.mark.parametrize('fault',[None,'refs','image','dispatched','ledger','infrastructure'])
def test_pending_credit_successor_requires_exact_frozen_company_refs_and_no_new_work(database,fault):
    with database[0].connect(**database[1]) as conn:
        assignment,source,refs=company_credit_seed(conn)
        apply(conn)
        credit.pending(conn,assignment,3)
        with conn.cursor() as cur:
            cur.execute('SET LOCAL session_replication_role=replica')
            changed=copy.deepcopy(refs)
            if fault=='refs':
                # A different but internally valid scope must not inherit the
                # preceding failure merely because all whole-item fields NULL.
                changed[0]['scope_doc']['company_input_hash']='sha256:'+'f'*64
                changed[0]['company_input_hash']='sha256:'+'f'*64
                changed[0]['scope_doc'].pop('cache_key')
                changed[0]['cache_key']=changed[0]['scope_doc']['cache_key']=contracts.document_hash(changed[0]['scope_doc'])
            elif fault=='image':changed[0]['scope_doc']['scorer_image_digest']='wrong'
            cur.execute('UPDATE public.lab_arena_runs SET company_judgment_refs=%s::jsonb WHERE run_id=%s',(json.dumps(changed),assignment+':3'))
            if fault=='dispatched':cur.execute('UPDATE public.lab_arena_runs SET runner_hotkey=%s WHERE run_id=%s',(credit.base.RUNNER,assignment+':3'))
            elif fault=='infrastructure':cur.execute("UPDATE public.lab_arena_runs SET status='failed',terminal_cause='judge_error',result_doc='{\"terminal_status\":\"judge_error\"}' WHERE run_id=%s",(assignment+':3',))
            cur.execute('SET LOCAL session_replication_role=origin')
        if fault=='ledger':credit.credit(conn,assignment+':3')
        conn.commit()
        assert credit.proved(conn,assignment+':3') is (fault is None)


def test_migration_is_idempotent_restricted_and_leaves_rows_and_other_functions_unchanged(database):
    with database[0].connect(**database[1]) as conn,conn.cursor() as cur:
        original=catalog(cur);rows=prior.latest.rows(cur)
        cur.execute('CREATE ROLE company502_migrator NOLOGIN NOSUPERUSER INHERIT')
        cur.execute('GRANT lab_arena_owner TO company502_migrator')
        cur.execute('GRANT USAGE,CREATE ON SCHEMA public TO company502_migrator')
        cur.execute('ALTER DEFAULT PRIVILEGES FOR ROLE company502_migrator IN SCHEMA public GRANT EXECUTE ON FUNCTIONS TO anon,authenticated,service_role')
        conn.commit()
        bad=SQL.read_text().replace("$old$AND judge.judgment_scope_doc", "$old$AND wrong.judgment_scope_doc",1)
        with pytest.raises(database[0].Error,match='anchor differs'):cur.execute(bad)
        conn.rollback()
        assert catalog(cur)==original and prior.latest.rows(cur)==rows
        cur.execute('GRANT EXECUTE ON FUNCTION public.lab_arena__all_judgments_ready_v1(text) TO anon')
        with pytest.raises(database[0].Error,match='security shape differs'):cur.execute(SQL.read_text())
        conn.rollback()
        assert catalog(cur)==original and prior.latest.rows(cur)==rows
        cur.execute('SET ROLE company502_migrator')
        cur.execute(SQL.read_text());after=catalog(cur)
        cur.execute(SQL.read_text());assert catalog(cur)==after
        cur.execute('RESET ROLE')
        assert prior.latest.rows(cur)==rows
        assert len(after)==len(original)
        changes=0
        for before,later in zip(original,after):
            assert before[0]==later[0] and before[2:]==later[2:]
            if before[1]!=later[1]:
                assert 'public.'+before[0] in SIGNATURES
                changes+=1
        assert changes==2
