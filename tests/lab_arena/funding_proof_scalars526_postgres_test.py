"""Funding proof semantics stay strict with wide retained provider responses."""
from pathlib import Path
import time
import pytest
from tests.lab_arena import proven_funding_recovery524_postgres_test as prior
from tests.lab_arena.quiescence_recovery_gate522_postgres_test import reset

SQL=Path(__file__).parents[2]/'scripts/526-lab-arena-funding-proof-scalars.sql'

@pytest.fixture(scope='module')
def database():
    gen=prior.database.__wrapped__();db=next(gen)
    try:
        with db[0].connect(**db[1]) as c,c.cursor() as q:
            q.execute(SQL.read_text());q.execute(SQL.read_text())
        yield db
    finally:gen.close()

# Reuse the exact contract controls against the new installed definition.
test_paid_new_dispatch_retires_stop_without_mutating_history=prior.test_paid_new_dispatch_retires_stop_without_mutating_history
test_untrusted_or_stale_paid_proof_never_retires_credit=prior.test_untrusted_or_stale_paid_proof_never_retires_credit
test_existing_single_dispatch_fence_still_prevents_duplicate_paid_identity=prior.test_existing_single_dispatch_fence_still_prevents_duplicate_paid_identity
test_authorized_inherited_root_uses_current_paid_proof_but_preserves_old_proof=prior.test_authorized_inherited_root_uses_current_paid_proof_but_preserves_old_proof
test_new_denial_consumes_old_watermark_and_stops_normal_retry_again=prior.test_new_denial_consumes_old_watermark_and_stops_normal_retry_again
test_every_exhausted_key_requires_its_own_new_paid_dispatch=prior.test_every_exhausted_key_requires_its_own_new_paid_dispatch


def test_wide_native_responses_parse_once_without_changing_proof(database):
    run,denial=prior.seed(database)
    with database[0].connect(**database[1]) as c,c.cursor() as q:
        funded=prior.clone(c,run)
        for index in range(500):
            prior.call(c,funded,'historical-'+str(index),at=denial-prior.timedelta(minutes=5))
        prior.call(c,funded,'new-funded')
        q.execute('SET LOCAL session_replication_role=replica')
        # Retain irrelevant response bodies as production does (up to 1.48 MB).
        q.execute("UPDATE public.lab_arena_ledger SET terminal_response=terminal_response||jsonb_build_object('retained_body',repeat('wide-json-body-',5000))WHERE run_id=%s AND entry_kind='settlement'",(funded,))
        q.execute('SET LOCAL session_replication_role=origin')
        before=prior.snapshot(q)
        q.execute('SELECT pg_get_functiondef(%s::regprocedure)',(prior.SIG,));new=q.fetchone()[0]
        import re
        text=SQL.read_text();old=re.search(r'\$old_funding_scalar526\$(.*?)\$old_funding_scalar526\$',text,re.S).group(1);replacement=re.search(r'\$new_funding_scalar526\$(.*?)\$new_funding_scalar526\$',text,re.S).group(1)
        assert new.count(replacement)==1
        q.execute(new.replace(replacement,old))
        started=time.monotonic();assert prior.proof(q,run['run_id']) is False;old_seconds=time.monotonic()-started
        q.execute(new)
        started=time.monotonic();assert prior.proof(q,run['run_id']) is False;new_seconds=time.monotonic()-started
        assert prior.snapshot(q)==before
        print('526 wide responses=501 old_seconds=%.6f new_seconds=%.6f'%(old_seconds,new_seconds))
        assert new_seconds<old_seconds*.8
        assert new_seconds<1.0


@pytest.mark.parametrize('installed',[False,True])
@pytest.mark.parametrize('drift',['owner','acl','secdef','volatility','path','source'])
def test_source_security_and_replay_guards(database,installed,drift):
    import re
    with database[0].connect(**database[1]) as c,c.cursor() as q:
        q.execute('SELECT pg_get_functiondef(%s::regprocedure)',(prior.SIG,));post=q.fetchone()[0]
        text=SQL.read_text();old=re.search(r'\$old_funding_scalar526\$(.*?)\$old_funding_scalar526\$',text,re.S).group(1);new=re.search(r'\$new_funding_scalar526\$(.*?)\$new_funding_scalar526\$',text,re.S).group(1)
        before=post if installed else post.replace(new,old)
        q.execute(before);q.execute('SAVEPOINT hostile')
        if drift=='source':q.execute(before.replace('BEGIN\n','BEGIN\n  -- unknown drift\n',1))
        else:q.execute({'owner':'ALTER FUNCTION '+prior.SIG+' OWNER TO postgres','acl':'GRANT EXECUTE ON FUNCTION '+prior.SIG+' TO anon','secdef':'ALTER FUNCTION '+prior.SIG+' SECURITY INVOKER','volatility':'ALTER FUNCTION '+prior.SIG+' VOLATILE','path':'ALTER FUNCTION '+prior.SIG+' SET search_path=public,pg_catalog'}[drift])
        body=text.split("SET LOCAL statement_timeout='30s';",1)[1].removesuffix('COMMIT;\n')
        with pytest.raises(database[0].Error):q.execute(body)
        q.execute('ROLLBACK TO SAVEPOINT hostile');q.execute(post)


@pytest.mark.parametrize('fault',['response_scalar','response_array','reservation_scalar','reservation_array','success_string','success_number','native_cost_scalar','native_cost_array','cache_null'])
def test_record_projection_preserves_malformed_and_exact_json_guards(database,fault):
    run,_=prior.seed(database)
    with database[0].connect(**database[1]) as c,c.cursor() as q:
        funded=prior.clone(c,run);cid=prior.call(c,funded,'fresh')
        q.execute('SET LOCAL session_replication_role=replica')
        if fault.startswith('reservation_'):
            value='[]' if fault.endswith('array') else '1'
            q.execute("UPDATE public.lab_arena_ledger SET entry_doc=%s::jsonb WHERE run_id=%s AND call_identity=%s AND entry_kind='reservation'",(value,funded,cid))
        else:
            if fault=='response_scalar':value="'1'::jsonb"
            elif fault=='response_array':value="'[]'::jsonb"
            else:
                value={'success_string':'jsonb_set(terminal_response,\'{call_succeeded}\',\'"true"\'::jsonb)','success_number':"jsonb_set(terminal_response,'{call_succeeded}','1'::jsonb)",'native_cost_scalar':"jsonb_set(terminal_response,'{provider_cost}','1'::jsonb)",'native_cost_array':"jsonb_set(terminal_response,'{provider_cost}','[]'::jsonb)",'cache_null':"terminal_response||jsonb_build_object('judgment_cache_source_run_id',NULL)"}[fault]
            q.execute("UPDATE public.lab_arena_ledger SET terminal_response="+value+" WHERE run_id=%s AND call_identity=%s AND entry_kind='settlement'",(funded,cid))
        q.execute('SET LOCAL session_replication_role=origin')
        assert prior.proof(q,run['run_id']) is True


@pytest.mark.parametrize('installed',[False,True])
def test_accounting_projection_refuses_unknown_matcher_source(database,installed):
    import re
    with database[0].connect(**database[1]) as c,c.cursor() as q:
        q.execute('SELECT pg_get_functiondef(%s::regprocedure)',(prior.SIG,));post=q.fetchone()[0]
        text=SQL.read_text();old=re.search(r'\$old_funding_scalar526\$(.*?)\$old_funding_scalar526\$',text,re.S).group(1);new=re.search(r'\$new_funding_scalar526\$(.*?)\$new_funding_scalar526\$',text,re.S).group(1)
        q.execute(post if installed else post.replace(new,old))
        q.execute("SELECT pg_get_functiondef('public.lab_arena__champion_account_failure_matches(jsonb,jsonb)'::regprocedure)");matcher=q.fetchone()[0]
        q.execute('SAVEPOINT hostile_matcher')
        q.execute(matcher.replace('SELECT COALESCE(','SELECT /* unknown matcher */ COALESCE(',1))
        body=text.split("SET LOCAL statement_timeout='30s';",1)[1].removesuffix('COMMIT;\n')
        with pytest.raises(database[0].Error):q.execute(body)
        q.execute('ROLLBACK TO SAVEPOINT hostile_matcher');q.execute(post)
