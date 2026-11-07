"""Data-only extension preserves an exact owned hold and ordinary retry authority."""
from copy import deepcopy
import json
import re
import pytest
from tests.lab_arena import nov05_retry_window644_postgres_test as previous

prior=previous.prior
database=previous.database
ROUND=prior.ROUND
SQL=prior.source.SCRIPTS/'646-arena-2026-11-05-held-retry-window-extension.sql'
ACTOR='nov05-primary-route-ci-hold'
REASON='Temporary Nov5 primary routing pending protected CI; new claims only.'
snapshot=previous.snapshot


def render(p):
    text=SQL.read_text().replace('2026-11-06T','2099-11-06T')
    manifest=json.loads(re.search(r'\$manifest646\$(.*?)\$manifest646\$',text,re.S)[1])
    with p.conn.cursor()as q:
        cfg=prior.scalar(q,'SELECT configuration_doc FROM public.lab_arena_rounds WHERE round_id=%s',(ROUND,))
        manifest['configuration_sha256']=prior.source.hash_doc(q,cfg)
        new=deepcopy(cfg);new['schedule'].update(manifest['schedule'])
        manifest['new_configuration_sha256']=prior.source.hash_doc(q,new)
    return re.sub(r'\$manifest646\$.*?\$manifest646\$',lambda _:'$manifest646$'+json.dumps(manifest)+'$manifest646$',text,flags=re.S)


def apply(p,sql=None):prior.apply(p,sql or p.held_window_sql)


def hold(p):
    with p.conn.cursor()as q:
        r=prior.scalar(q,'SELECT to_jsonb(r) FROM public.lab_arena_rounds r WHERE round_id=%s',(ROUND,))
        guard=prior.scalar(q,'SELECT guard_generation FROM public.lab_arena_restart_claim_control WHERE singleton')
        q.execute('SELECT public.lab_arena_scoring_operator_hold_v1(%s,%s::jsonb,%s,%s,%s,%s,%s,true)',(ROUND,json.dumps(r['configuration_doc']),r['status_generation'],r['stage_generation'],guard,ACTOR,REASON))
        assert q.fetchone()[0]['operator_paused']is True
    p.conn.commit()


@pytest.fixture
def prepared(database):
    gen=previous.prepared.__wrapped__(database);p=next(gen)
    try:
        previous.apply(p);p.held_window_sql=render(p)
        yield p
    finally:gen.close()


@pytest.mark.parametrize('held',[False,True])
def test_exact_extension_after_old_cutoff_preserves_every_row_and_control(prepared,held):
    p=prepared;lease=prior.claim(p);prior.bill(p,lease)
    if held:hold(p)
    before=snapshot(p)
    with p.conn.cursor()as q:control=prior.scalar(q,'SELECT to_jsonb(c) FROM public.lab_arena_restart_claim_control c WHERE singleton')
    sql=p.held_window_sql.replace('clock_timestamp()',"TIMESTAMPTZ '2099-11-06T07:40:00Z'")
    apply(p,sql);after=snapshot(p)
    expected=deepcopy(before);expected['rounds'][ROUND]['configuration_doc']['schedule']['stage_2_close']='2099-11-06T08:30:00Z'
    expected['rounds'][ROUND]['updated_at']=after['rounds'][ROUND]['updated_at'];expected['history']=after['history']
    assert after==expected
    with p.conn.cursor()as q:assert prior.scalar(q,'SELECT to_jsonb(c) FROM public.lab_arena_restart_claim_control c WHERE singleton')==control
    assert len(after['history'])==len(before['history'])+1
    progressed=snapshot(p);apply(p);assert snapshot(p)==progressed
    assert prior.complete(p,lease)['status']=='accepted'


@pytest.mark.parametrize('fault',['foreign_actor','foreign_reason','guard','phase','config','clock','postimage','trigger'])
def test_wrong_scope_or_control_rolls_back(prepared,fault):
    p=prepared
    if fault in('foreign_actor','foreign_reason'):hold(p)
    with p.conn.cursor()as q:
        q.execute('SET LOCAL session_replication_role=replica')
        if fault=='foreign_actor':q.execute("UPDATE public.lab_arena_restart_claim_control SET actor_ref='foreign' WHERE singleton")
        elif fault=='foreign_reason':q.execute("UPDATE public.lab_arena_restart_claim_control SET pause_reason='foreign' WHERE singleton")
        elif fault=='guard':q.execute("UPDATE public.lab_arena_restart_claim_control SET operator_paused=true,actor_ref='foreign',pause_reason='foreign' WHERE singleton")
        elif fault=='phase':q.execute("UPDATE public.lab_arena_rounds SET status='stage2_closed' WHERE round_id=%s",(ROUND,))
        elif fault=='config':q.execute("UPDATE public.lab_arena_rounds SET configuration_doc=jsonb_set(configuration_doc,'{max_attempts_per_assignment}','3') WHERE round_id=%s",(ROUND,))
        elif fault=='trigger':q.execute('ALTER TABLE public.lab_arena_rounds DISABLE TRIGGER lab_arena_rounds_write_once')
        q.execute('SET LOCAL session_replication_role=origin')
    p.conn.commit();sql=p.held_window_sql
    if fault=='clock':sql=sql.replace("clock_timestamp()>='2099-11-06T08:30:00Z'","clock_timestamp()>='2000-11-06T08:30:00Z'")
    elif fault=='postimage':
        raw=re.search(r'\$manifest646\$(.*?)\$manifest646\$',sql,re.S)[1];bad=json.loads(raw);bad['new_configuration_sha256']='0'*64;sql=sql.replace(raw,json.dumps(bad))
    before=snapshot(p)
    try:
        with pytest.raises(p.pg.Error,match='Window646'):apply(p,sql)
        p.conn.rollback();assert snapshot(p)==before
    finally:
        if fault=='trigger':
            with p.conn.cursor()as q:q.execute('ALTER TABLE public.lab_arena_rounds ENABLE TRIGGER lab_arena_rounds_write_once')
            p.conn.commit()


def test_own_hold_survives_post_update_failure(prepared):
    p=prepared;hold(p);before=snapshot(p)
    with p.conn.cursor()as q:control=prior.scalar(q,'SELECT to_jsonb(c) FROM public.lab_arena_restart_claim_control c WHERE singleton')
    head,tail=p.held_window_sql.rsplit("clock_timestamp()>='2099-11-06T08:30:00Z'",1)
    with pytest.raises(p.pg.Error,match='Window646 exact readback differs'):
        apply(p,head+"clock_timestamp()>='2000-11-06T08:30:00Z'"+tail)
    p.conn.rollback();assert snapshot(p)==before
    with p.conn.cursor()as q:assert prior.scalar(q,'SELECT to_jsonb(c) FROM public.lab_arena_restart_claim_control c WHERE singleton')==control


@pytest.mark.parametrize('extended',[False,True])
def test_native_retry_after0730_under_hold_needs_extended_window(prepared,extended):
    p=prepared
    first=next(t['run_id']for t in p.targets if t['mode']=='restore_unused'and t['run_id'].endswith(':1'))
    for _ in range(50):
        lease=prior.claim(p)
        if lease['run_id']==first:break
        assert prior.complete(p,lease)['status']=='accepted'
    assert lease['run_id']==first
    prior.bill(p,lease);hold(p)
    if extended:apply(p)
    with p.conn.cursor()as q:
        definition=prior.scalar(q,"SELECT pg_get_functiondef('public.lab_arena_expire_leases(text)'::regprocedure)")
        q.execute(re.sub(r'(?:pg_catalog\.)?clock_timestamp\(\)',"(TIMESTAMPTZ '2099-11-06T07:40:00Z')",definition))
    p.conn.commit()
    try:
        p.store.expire_leases(ROUND)
        with p.conn.cursor()as q:
            exists=prior.scalar(q,'SELECT EXISTS(SELECT 1 FROM public.lab_arena_runs WHERE run_id=%s)',(first.rsplit(':',1)[0]+':2',))
            assert prior.scalar(q,'SELECT operator_paused FROM public.lab_arena_restart_claim_control WHERE singleton')is True
        assert exists is extended
    finally:
        p.conn.rollback()
        with p.conn.cursor()as q:q.execute(definition)
        p.conn.commit()
