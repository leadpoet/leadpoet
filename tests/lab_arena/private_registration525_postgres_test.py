"""Keep the internal admission helper private without changing service wrappers."""
import json
from datetime import timedelta
from pathlib import Path

import pytest

from lab_arena.owner_admission import OwnerAdmission
from lab_arena.store import ArenaStore, PsycopgTransport
from tests.lab_arena.lab_arena_pg_harness import CURRENT_SERVICE_MIGRATIONS, database_with_lab_arena_migration
from tests.lab_arena.queued_submission_replacement_postgres_test import _config, _doc
from tests.lab_arena.test_lab_arena_source_migration_postgres import _hotkey

SQL = Path(__file__).parents[2] / 'scripts/525-lab-arena-private-registration-helper.sql'
TARGET = 'public.lab_arena__register_submission_v2(text,text,text,jsonb,text,bigint,text)'
WRAPPER = 'public.lab_arena_register_submission_v2(text,text,text,jsonb,text,bigint,text)'
BLOCK_HASH = '0x' + 'a' * 64


@pytest.fixture(scope='module')
def database():
    yield from database_with_lab_arena_migration(CURRENT_SERVICE_MIGRATIONS)


def catalog(q):
    q.execute("SELECT jsonb_agg(to_jsonb(p) ORDER BY p.oid) FROM pg_proc p WHERE p.oid IN (%s::regprocedure,%s::regprocedure,'public.lab_arena_register_submission(text,text,text,jsonb)'::regprocedure)", (TARGET, WRAPPER))
    functions = q.fetchone()[0]
    q.execute('SELECT coalesce(jsonb_agg(to_jsonb(a) ORDER BY a.oid),\'[]\') FROM pg_default_acl a')
    defaults = q.fetchone()[0]
    q.execute('SELECT coalesce(jsonb_agg(to_jsonb(s) ORDER BY submission_id),\'[]\') FROM public.lab_arena_submissions s')
    return functions, defaults, q.fetchone()[0]


def live_acl(q):
    # Match the actual Supabase grants left after PUBLIC was already revoked.
    q.execute('SET LOCAL ROLE lab_arena_owner')
    q.execute('REVOKE ALL ON FUNCTION ' + TARGET + ' FROM PUBLIC,anon,authenticated,service_role,lab_arena_service')
    q.execute('GRANT EXECUTE ON FUNCTION ' + TARGET + ' TO anon,authenticated,service_role')
    q.execute('RESET ROLE')


def apply(database):
    with database[0].connect(**database[1]) as c, c.cursor() as q:
        q.execute('SET ROLE lab_arena_owner')
        q.execute("SELECT rolsuper FROM pg_roles WHERE rolname=current_user")
        assert q.fetchone() == (False,)
        q.execute(SQL.read_text())


def make_round(database, suffix):
    rid = 'arena-2098-08-21-' + suffix.replace('_', '')
    config = _config(rid, cutoff_after=timedelta(hours=3))
    config.update(integrity_policy='arena_integrity_v1', stage_1_icp_count=10, stage_2_icp_count=10, max_challengers=20, cost_per_company_microusd=1_000_000, scorer_policy={'scoring_adapter_version':'qualification_integrity_v2'})
    transport = PsycopgTransport(lambda: database[0].connect(**database[1]))
    store = ArenaStore(transport)
    assert store.create_round(rid, config)['status'] == 'created'
    return store, transport, rid, config


def invoke(q, rid, sid, miner, owner, *, baseline=False):
    doc = _doc(rid, sid, 123)
    if baseline:
        doc['is_king'] = True
    q.execute('SELECT public.lab_arena__register_submission_v2(%s,%s,%s,%s::jsonb,%s,%s,%s)',
              (rid, sid, miner, json.dumps(doc), None if baseline else owner,
               None if baseline else 123, None if baseline else BLOCK_HASH))
    return q.fetchone()[0]


def test_actual_default_acl_gap_then_non_superuser_apply_and_exact_replay(database):
    store, transport, rid, config = make_round(database, 'gap')
    try:
        with database[0].connect(**database[1]) as c, c.cursor() as q:
            q.execute('ALTER DEFAULT PRIVILEGES FOR ROLE postgres IN SCHEMA public GRANT EXECUTE ON FUNCTIONS TO anon,authenticated,service_role')
            live_acl(q)
            q.execute('SET LOCAL ROLE anon')
            # No signed API request or owner lookup occurred. The old grant is sufficient.
            assert invoke(q, rid, 'anonymous-reservation', _hotkey('unverified-miner'), _hotkey('unverified-owner'))['status'] == 'registered'
            q.execute('RESET ROLE')
            before = catalog(q)
        apply(database)
        with database[0].connect(**database[1]) as c, c.cursor() as q:
            after = catalog(q)
            assert after[1:] == before[1:]
            old = {x['oid']: x for x in before[0]}
            for function in after[0]:
                prior = old[function['oid']]
                if function['proname'] == 'lab_arena__register_submission_v2':
                    assert function['proacl'] == ['lab_arena_owner=X/lab_arena_owner']
                    for role in ('anon','authenticated','service_role'):
                        q.execute("SELECT has_function_privilege(%s,%s,'EXECUTE')", (role,TARGET))
                        assert q.fetchone() == (False,)
                    assert {k:v for k,v in function.items() if k != 'proacl'} == {k:v for k,v in prior.items() if k != 'proacl'}
                else:
                    assert function == prior
        apply(database)
        with database[0].connect(**database[1]) as c, c.cursor() as q:
            assert catalog(q) == after
    finally:
        transport.close()


@pytest.mark.parametrize('role', ['anon', 'authenticated', 'service_role'])
def test_external_roles_cannot_reserve_normal_or_baseline(database, role):
    store, transport, rid, config = make_round(database, role)
    try:
        with database[0].connect(**database[1]) as c, c.cursor() as q:
            live_acl(q)
        apply(database)
        for baseline in (False, True):
            with database[0].connect(**database[1]) as c, c.cursor() as q:
                before = catalog(q)
                q.execute('SAVEPOINT denial')
                q.execute('SET LOCAL ROLE ' + role)
                with pytest.raises(database[0].Error) as error:
                    invoke(q, rid, 'baseline-' + rid.removeprefix('arena-') if baseline else 'denied-' + role,
                           config['baseline_hotkey'] if baseline else _hotkey(role), _hotkey('unverified-owner'), baseline=baseline)
                assert error.value.pgcode == '42501'
                q.execute('ROLLBACK TO SAVEPOINT denial')
                assert catalog(q) == before
    finally:
        transport.close()


def test_service_normal_and_baseline_wrappers_still_register_and_retry(database):
    store, transport, rid, config = make_round(database, 'wrappers')
    try:
        with database[0].connect(**database[1]) as c, c.cursor() as q:
            live_acl(q)
        apply(database)
        def service_connection():
            c = database[0].connect(**database[1])
            c.autocommit = True
            with c.cursor() as q:
                q.execute('SET ROLE lab_arena_service')
            return c
        service_transport = PsycopgTransport(service_connection)
        service = ArenaStore(service_transport)
        try:
            admission = OwnerAdmission(_hotkey('real-owner'), 123, BLOCK_HASH)
            args = (rid, 'normal-service', _hotkey('real-miner'), _doc(rid, 'normal-service', 123))
            assert service.register_submission(*args, owner_admission=admission)['status'] == 'registered'
            assert service.register_submission(*args, owner_admission=admission)['status'] == 'existing'
            baseline_id = 'baseline-' + rid.removeprefix('arena-')
            doc = {**_doc(rid, baseline_id, 123), 'is_king': True}
            assert service.register_submission(rid, baseline_id, config['baseline_hotkey'], doc)['status'] == 'registered'
            assert service.register_submission(rid, baseline_id, config['baseline_hotkey'], doc)['status'] == 'existing'
        finally:
            service_transport.close()
    finally:
        transport.close()


@pytest.mark.parametrize('installed', [False, True])
@pytest.mark.parametrize('drift', ['owner', 'secdef', 'path', 'source', 'acl', 'wrapper_acl', 'wrapper_source'])
def test_security_and_source_drift_fail_closed_even_after_install(database, installed, drift):
    with database[0].connect(**database[1]) as c, c.cursor() as q:
        live_acl(q)
    if installed:
        apply(database)
    with database[0].connect(**database[1]) as c, c.cursor() as q:
        before = catalog(q)
        q.execute('SAVEPOINT drift')
        if drift in ('source', 'wrapper_source'):
            sig = WRAPPER if drift == 'wrapper_source' else TARGET
            q.execute('SELECT pg_get_functiondef(%s::regprocedure)', (sig,))
            d = q.fetchone()[0]
            q.execute(d.replace('AS $function$\n', 'AS $function$\n-- unreviewed source\n', 1))
        else:
            q.execute({'owner':'ALTER FUNCTION '+TARGET+' OWNER TO postgres',
                       'secdef':'ALTER FUNCTION '+TARGET+' SECURITY INVOKER',
                       'path':'ALTER FUNCTION '+TARGET+' SET search_path=public,pg_catalog',
                       'acl':'GRANT EXECUTE ON FUNCTION '+TARGET+' TO PUBLIC',
                       'wrapper_acl':'GRANT EXECUTE ON FUNCTION '+WRAPPER+' TO anon'}[drift])
        body = 'DO $private_registration_helper$' + SQL.read_text().split('DO $private_registration_helper$', 1)[1].rsplit('COMMIT;', 1)[0]
        with pytest.raises(database[0].Error):
            q.execute(body)
        q.execute('ROLLBACK TO SAVEPOINT drift')
        assert catalog(q) == before
