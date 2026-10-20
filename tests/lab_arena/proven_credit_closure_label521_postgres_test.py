"""A cheap closure label is written only by the existing native-credit branch."""
from pathlib import Path
import pytest
from tests.lab_arena import ordered_claim_admission520_postgres_test as admission
from tests.lab_arena import model_policy_supersession517_postgres_test as revision

SQL=Path(__file__).parents[2]/'scripts/521-lab-arena-proven-credit-closure-label.sql'
SIG='public.lab_arena_expire_leases(text)'

@pytest.fixture(scope='module')
def database():
    yield from admission.database.__wrapped__()

@pytest.fixture(scope='module')
def original(database):
    with database[0].connect(**database[1]) as conn,conn.cursor() as cur:
        cur.execute('SELECT pg_get_functiondef(%s::regprocedure)',(SIG,));return cur.fetchone()[0]

@pytest.fixture(autouse=True)
def restore(database,original):
    with database[0].connect(**database[1]) as conn,conn.cursor() as cur:cur.execute(original)
    yield
    with database[0].connect(**database[1]) as conn,conn.cursor() as cur:cur.execute(original)


def apply(db):
    with db[0].connect(**db[1]) as conn,conn.cursor() as cur:
        cur.execute(SQL.read_text());cur.execute(SQL.read_text())


@pytest.mark.parametrize('proof,funding,expected',[(True,'miner_key',True),(False,'miner_key',False),(True,'host',False)])
def test_only_exact_native_same_payer_credit_closure_gets_label(database,proof,funding,expected):
    apply(database)
    # This existing control checks real proof, model-only inheritance, terminal
    # state, continuation admission and byte-identical historical charges.
    revision.test_only_native_same_payer_credit_proof_survives_model_revision(database,proof,funding,expected)
    with database[0].connect(**database[1]) as conn,conn.cursor() as cur:
        cur.execute("SELECT run_id,status,terminal_cause,terminal_doc FROM public.lab_arena_runs WHERE round_id='arena-2099-10-20-supersession' AND kind='score' AND submission_id LIKE '%%sub-1' AND attempt=2")
        rid,status,cause,doc=cur.fetchone()
        assert (doc or {}).get('failure_reason')==('provider_credit_exhausted' if expected else None)
        if expected:
            assert status=='failed' and cause=='stage_closed' and doc['previous_status']=='pending'
            cur.execute('SELECT to_jsonb(r) FROM public.lab_arena_runs r WHERE run_id=%s',(rid,));before=cur.fetchone()[0]
            cur.execute("SELECT public.lab_arena_expire_leases('arena-2099-10-20-supersession')")
            cur.execute('SELECT to_jsonb(r) FROM public.lab_arena_runs r WHERE run_id=%s',(rid,));assert cur.fetchone()[0]==before


def test_install_and_later_expiry_do_not_backfill_historical_terminal_row(database):
    revision.test_only_native_same_payer_credit_proof_survives_model_revision(database,True,'miner_key',True)
    with database[0].connect(**database[1]) as conn,conn.cursor() as cur:
        cur.execute("SELECT jsonb_agg(to_jsonb(r) ORDER BY run_id) FROM public.lab_arena_runs r WHERE status IN('failed','accepted')")
        before=cur.fetchone()[0]
        assert any(r.get('terminal_doc',{}).get('previous_status')=='pending' and 'failure_reason' not in r['terminal_doc'] for r in before if r.get('terminal_doc'))
    apply(database)
    with database[0].connect(**database[1]) as conn,conn.cursor() as cur:
        cur.execute("SELECT public.lab_arena_expire_leases('arena-2099-10-20-supersession')")
        cur.execute("SELECT jsonb_agg(to_jsonb(r) ORDER BY run_id) FROM public.lab_arena_runs r WHERE status IN('failed','accepted')")
        assert cur.fetchone()[0]==before


@pytest.mark.parametrize('installed',[False,True])
@pytest.mark.parametrize('drift',['owner','acl','secdef','volatility','path','source'])
def test_migration_source_and_security_drift_fail_closed(database,installed,drift):
    if installed:apply(database)
    with database[0].connect(**database[1]) as conn,conn.cursor() as cur:
        cur.execute('SELECT pg_get_functiondef(%s::regprocedure)',(SIG,));saved=cur.fetchone()[0]
        cur.execute('SAVEPOINT hostile')
        if drift=='source':cur.execute(saved.replace('BEGIN\n','BEGIN\n  -- unknown source drift\n',1))
        else:
            statements={
                'owner':'ALTER FUNCTION '+SIG+' OWNER TO postgres',
                'acl':'GRANT EXECUTE ON FUNCTION '+SIG+' TO anon',
                'secdef':'ALTER FUNCTION '+SIG+' SECURITY INVOKER',
                'volatility':'ALTER FUNCTION '+SIG+' STABLE',
                'path':'ALTER FUNCTION '+SIG+' SET search_path=public,pg_catalog',
            }
            cur.execute(statements[drift])
        body=SQL.read_text().split('DO $proven_credit_closure_label$')[1].split('$proven_credit_closure_label$;')[0]
        with pytest.raises(database[0].Error):cur.execute('DO $proven_credit_closure_label$'+body+'$proven_credit_closure_label$;')
        cur.execute('ROLLBACK TO SAVEPOINT hostile')


def test_authentication_failure_does_not_gain_credit_label(database,monkeypatch):
    from tests.lab_arena import miner_credit_terminal484_postgres_test as credit
    apply(database)
    original_credit=credit.credit
    def authentication_failure(conn,run_id,**kwargs):
        return original_credit(conn,run_id,reason='invalid_api_key',uncertain=True,**kwargs)
    monkeypatch.setattr(credit,'credit',authentication_failure)
    revision.test_only_native_same_payer_credit_proof_survives_model_revision(database,True,'miner_key',False)
    with database[0].connect(**database[1]) as conn,conn.cursor() as cur:
        cur.execute("SELECT status,terminal_doc FROM public.lab_arena_runs WHERE round_id='arena-2099-10-20-supersession' AND kind='score' AND submission_id LIKE '%%sub-1' AND attempt=2")
        status,doc=cur.fetchone();assert status=='pending' and not (doc or {}).get('failure_reason')
