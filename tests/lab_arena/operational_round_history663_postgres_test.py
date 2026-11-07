"""Operational receipts retain the real floor without reinterpreting old history."""
import hashlib
import json
from pathlib import Path
import re

import pytest
from tests.lab_arena import nov05_infrastructure_recovery633_postgres_test as source

SCRIPTS=Path(__file__).resolve().parents[2]/'scripts'
SQL=(SCRIPTS/'663-lab-arena-operational-round-history.sql').read_text()
SIGNATURE='public.lab_arena_scoring_policy_history_v1(text)'
OPERATION='round_operational_update_v1'


def block(text,tag):
    return 'DO $'+tag+'$'+text.split('DO $'+tag+'$')[1].split('$'+tag+'$;')[0]+'$'+tag+'$;'


@pytest.fixture(scope='module')
def database():
    gen=source.database.__wrapped__();pg,dsn=next(gen)
    try:
        with pg.connect(**dsn)as c,c.cursor()as q:
            q.execute(block(next(SCRIPTS.glob('635-*.sql')).read_text(),'patch635_history'));c.commit()
            q.execute(source.SQL.read_text().split('-- Exact one-time data correction.')[0]+'COMMIT;');c.commit()
            q.execute(block(next(SCRIPTS.glob('641-*.sql')).read_text(),'history641'));c.commit()
            q.execute('SELECT pg_get_functiondef(%s::regprocedure)',(SIGNATURE,));d=q.fetchone()[0]
            assert hashlib.sha256(d.encode()).hexdigest()=='5fdfee1a65c755a862b17998607514941a222c5912d60852a00bf21e027777bd'
            text=next(SCRIPTS.glob('639-*.sql')).read_text()
            pair=next((old,new)for old,new in re.findall(r'\$old639\$(.*?)\$old639\$,\$new639\$(.*?)\$new639\$',text,re.S)if 'nov05_recovery_window641' in old)
            assert d.count(pair[0])==1;q.execute(d.replace(*pair));c.commit()
            q.execute('SELECT pg_get_functiondef(%s::regprocedure)',(SIGNATURE,));d=q.fetchone()[0]
            assert hashlib.sha256(d.encode()).hexdigest()=='34123c2c2c2e68d4056a94c2f9f5bba4889d158ee8ac38869f6e68da88a2eb84'
        yield pg,dsn
    finally:gen.close()


@pytest.fixture
def prepared(database):
    p=source.seed(database)
    with p.conn.cursor()as q:
        q.execute('SELECT pg_get_functiondef(%s::regprocedure)',(SIGNATURE,));original=q.fetchone()[0]
    p.conn.commit()
    try:yield p
    finally:
        p.conn.rollback()
        with p.conn.cursor()as q:q.execute(original)
        p.conn.commit();p.store.close();p.conn.close()


def history(q):
    q.execute('SELECT public.lab_arena_scoring_policy_history_v1(%s)',(source.ROUND,));return q.fetchone()[0]


def insert(q,revision,operation,*,saved_floor=0):
    q.execute("SELECT to_jsonb(r)-'updated_at'FROM lab_arena_rounds r WHERE round_id=%s",(source.ROUND,));old=q.fetchone()[0]
    new=dict(old);new['scoring_generation_floor']=saved_floor
    recovery=None if operation is None else {'operation':operation}
    if operation=='restore_recorded_result_v1':recovery.update(selected_generation_floor=saved_floor,restored_from_generation_floor=1)
    q.execute("INSERT INTO lab_arena_scoring_supersessions(round_id,generation_floor,request_id,request_hash,actor_ref,transaction_id,backend_pid,old_round_doc,new_round_doc,old_score_projections,recovery_doc)VALUES(%s,%s,%s,%s,'fixture663',txid_current(),pg_backend_pid(),%s::jsonb,%s::jsonb,'{}',%s::jsonb)",(source.ROUND,revision,'fixture663-'+str(revision),'sha256:'+'a'*64,json.dumps(old),json.dumps(new),json.dumps(recovery)))


def metadata(q):
    q.execute("SELECT to_jsonb(p)-'prosrc'FROM pg_proc p WHERE oid=%s::regprocedure",(SIGNATURE,));return q.fetchone()[0]


def test_all_legacy_history_is_byte_equal_and_generic_tag_keeps_floor(prepared):
    p=prepared
    with p.conn.cursor()as q:
        q.execute('SELECT prosrc FROM pg_proc WHERE oid=%s::regprocedure',(SIGNATURE,));d=q.fetchone()[0]
        operations=re.search(r"h.recovery_doc->>'operation' IN\((.*?)\) THEN",d,re.S)[1]
        old_ops=re.findall(r"'([^']+)'",operations)+[None,'restore_recorded_result_v1','unknown_legacy_operation']
        for n,op in enumerate(old_ops,1):insert(q,n,op)
        before=history(q);before_metadata=metadata(q);before_data=source.snapshot(p)
    p.conn.commit()
    for _ in range(2):
        with p.conn.cursor()as q:q.execute(SQL)
        p.conn.commit()
        with p.conn.cursor()as q:
            assert history(q)==before
            assert metadata(q)==before_metadata
            assert source.snapshot(p)==before_data
        p.conn.commit()
    with p.conn.cursor()as q:
        insert(q,100,OPERATION)
        after=history(q)
        assert after['revisions'][:-1]==before['revisions']
        r=after['revisions'][-1]
        assert r['generation_floor']==0 and r['receipt_generation']==100
        assert r['recovery']['operation']==OPERATION
        # A universal saved-floor rewrite would reinterpret this legacy row.
        restored=next(x for x in before['revisions']if x['restoration']is not None)
        assert restored['generation_floor']!=restored['restoration']['selected_generation_floor']
    p.conn.commit()


@pytest.mark.parametrize('fault',['source','acl'])
def test_unknown_source_or_acl_refused_without_partial_change(prepared,fault):
    p=prepared
    with p.conn.cursor()as q:
        q.execute('SELECT pg_get_functiondef(%s::regprocedure)',(SIGNATURE,));d=q.fetchone()[0]
        if fault=='source':
            changed=d.replace("'receipt_generation'","'receipt_generation' /* unrelated663 */",1)
            assert changed!=d;q.execute(changed)
        else:q.execute('GRANT EXECUTE ON FUNCTION '+SIGNATURE+' TO authenticated')
        before=metadata(q);before_data=source.snapshot(p)
    p.conn.commit()
    try:
        with pytest.raises(p.pg.Error,match='Operational history663'):
            with p.conn.cursor()as q:q.execute(SQL)
        p.conn.rollback()
        with p.conn.cursor()as q:
            assert metadata(q)==before
            assert source.snapshot(p)==before_data
    finally:
        p.conn.rollback()
        if fault=='acl':
            with p.conn.cursor()as q:q.execute('REVOKE EXECUTE ON FUNCTION '+SIGNATURE+' FROM authenticated')
            p.conn.commit()
