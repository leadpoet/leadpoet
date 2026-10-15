"""Production-shaped Supabase defaults cannot expose the two weight write RPCs."""
from pathlib import Path

import pytest

from tests.lab_arena import chain_outcome_retry488_postgres_test as prior
from tests.lab_arena.lab_arena_pg_harness import DEFAULT_MIGRATIONS, database_with_lab_arena_migration

ROOT = Path(__file__).parents[2]
SQL = ROOT / 'scripts/489-lab-arena-weight-write-rpc-privileges.sql'
RETRY_SQL = ROOT / 'scripts/488-lab-arena-chain-outcome-semantic-retry.sql'
SIGNATURES = (prior.SIGNATURE, 'public.lab_arena_publish_weight_state_v1(text,integer,bigint,text,jsonb)')
LEGACY = '{lab_arena_owner=X/lab_arena_owner,anon=X/lab_arena_owner,authenticated=X/lab_arena_owner,service_role=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner}'
STRICT = '{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner}'
PUBLIC_ROLES = ('anon', 'authenticated', 'service_role')


def apply(conn, path=SQL):
    with conn.cursor() as cur:
        cur.execute(path.read_text())
    conn.commit()


def state(cur):
    cur.execute("SELECT p.oid::regprocedure::text,pg_get_functiondef(p.oid),owner.rolname,p.proacl::text,p.prosecdef,p.provolatile,p.proconfig FROM pg_proc p JOIN pg_roles owner ON owner.oid=p.proowner WHERE p.pronamespace='public'::regnamespace AND p.prokind='f' ORDER BY p.oid")
    functions = cur.fetchall()
    cur.execute("SELECT c.relname,c.relacl::text,c.relowner,c.relrowsecurity,c.relforcerowsecurity FROM pg_class c WHERE c.relnamespace='public'::regnamespace ORDER BY c.oid")
    tables = cur.fetchall()
    cur.execute('SELECT defaclrole,defaclnamespace,defaclobjtype,defaclacl::text FROM pg_default_acl ORDER BY oid')
    defaults = cur.fetchall()
    cur.execute('SELECT roleid,member,grantor,admin_option FROM pg_auth_members ORDER BY roleid,member')
    return functions, tables, defaults, cur.fetchall()


def assert_hardened(cur):
    for signature in SIGNATURES:
        cur.execute('SELECT proacl::text FROM pg_proc WHERE oid=%s::regprocedure', (signature,))
        assert cur.fetchone() == (STRICT,)
        for role in PUBLIC_ROLES:
            cur.execute('SELECT has_function_privilege(%s,%s,\'EXECUTE\'),has_schema_privilege(%s,\'public\',\'USAGE\')', (role, signature, role))
            assert cur.fetchone() == (False, True)
        cur.execute('SELECT has_function_privilege(\'lab_arena_service\',%s,\'EXECUTE\')', (signature,))
        assert cur.fetchone() == (True,)


def data_rows(cur):
    result = []
    for table in ('lab_arena_accepted_weight_states', 'lab_arena_chain_outcomes'):
        cur.execute('SELECT to_jsonb(r) FROM public.'+table+' r ORDER BY network,netuid,epoch,created_at')
        result.append(cur.fetchall())
    return result


@pytest.fixture(scope='module')
def modeled_database():
    generator = database_with_lab_arena_migration(())
    db = next(generator)
    try:
        assert db[1]['host'] == '127.0.0.1'
        with db[0].connect(**db[1]) as conn, conn.cursor() as cur:
            # These are the observed production postgres defaults.202 revokes
            # PUBLIC after creation but leaves these explicit role grants.
            cur.execute('ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT EXECUTE ON FUNCTIONS TO anon,authenticated,service_role')
            cur.execute('ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON TABLES TO anon,service_role')
            cur.execute('ALTER ROLE service_role BYPASSRLS')
            for migration in DEFAULT_MIGRATIONS:
                cur.execute((ROOT / 'scripts' / migration).read_text())
            cur.execute('CREATE ROLE authenticator NOLOGIN NOINHERIT')
            cur.execute('GRANT anon,authenticated,service_role,lab_arena_service TO authenticator')
            cur.execute('GRANT USAGE ON SCHEMA public TO anon,authenticated,service_role,lab_arena_service,authenticator')
            definitions = []
            for signature in SIGNATURES:
                cur.execute('SELECT pg_get_functiondef(oid),proacl::text FROM pg_proc WHERE oid=%s::regprocedure', (signature,))
                definition, acl = cur.fetchone()
                assert acl == LEGACY
                definitions.append(definition)
            cur.execute('REVOKE EXECUTE ON FUNCTION '+SIGNATURES[0]+' FROM anon,authenticated,service_role')
            cur.execute(RETRY_SQL.read_text())
            cur.execute('SELECT pg_get_functiondef(%s::regprocedure)', (SIGNATURES[0],))
            post = cur.fetchone()[0]
        yield db, definitions, post
    finally:
        generator.close()


@pytest.fixture
def database(modeled_database):
    db, definitions, _ = modeled_database
    with db[0].connect(**db[1]) as conn, conn.cursor() as cur:
        for signature, definition in zip(SIGNATURES, definitions):
            cur.execute(definition)
            cur.execute('REVOKE ALL ON FUNCTION '+signature+' FROM PUBLIC,anon,authenticated,service_role,lab_arena_service')
            cur.execute('GRANT EXECUTE ON FUNCTION '+signature+' TO anon,authenticated,service_role,lab_arena_service')
            cur.execute('SELECT proacl::text FROM pg_proc WHERE oid=%s::regprocedure', (signature,))
            assert cur.fetchone() == (LEGACY,)
        cur.execute('TRUNCATE public.lab_arena_chain_outcomes,public.lab_arena_accepted_weight_states')
    return db


def rpc_as(conn, role, expression, args):
    try:
        with conn.cursor() as cur:
            cur.execute('SET LOCAL ROLE '+role)
            cur.execute('SELECT public.'+expression, args)
            result = cur.fetchone()[0]
        conn.commit()
        return result
    except Exception:
        conn.rollback()
        raise


PUBLISH = 'lab_arena_publish_weight_state_v1(%s::text,%s::integer,%s::bigint,%s::text,%s::jsonb)'
RECORD = 'lab_arena_record_chain_outcome_v1(%s::text,%s::integer,%s::bigint,%s::text,%s::text,%s::text,%s::jsonb)'


def arguments(document):
    import json
    return (document['network'], document['netuid'], document['epoch'], document['validator_hotkey'],
            document['request_id'], document['extrinsic_hash'], json.dumps(document))


def test_exact_production_acl_blocks488_then489_hardens_without_function_or_data_changes(database):
    import json
    with prior.service(database) as app, database[0].connect(**database[1]) as conn, conn.cursor() as cur:
        # Prove all three public roles can execute both owner functions before
        # the correction, exclusively inside this disposable database.
        for index, role in enumerate(PUBLIC_ROLES):
            accepted = prior.accepted(app, 25786+index)
            publish_args = ('finney', 71, accepted['epoch'], accepted['state_hash'], json.dumps(accepted))
            assert rpc_as(conn, role, PUBLISH, publish_args)['status'] == 'accepted'
            report = prior.report(accepted)
            assert rpc_as(conn, role, RECORD, arguments(report)) == {'status': 'recorded'}
        canonical = data_rows(cur)
        before = state(cur)
        with pytest.raises(database[0].Error, match='security shape differs'):
            apply(conn, RETRY_SQL)
        conn.rollback()
        assert state(cur) == before and data_rows(cur) == canonical
        cur.execute('CREATE ROLE privilege489_migrator NOLOGIN NOSUPERUSER INHERIT')
        cur.execute('GRANT lab_arena_owner TO privilege489_migrator')
        conn.commit()
        before = state(cur)
        cur.execute('SET ROLE privilege489_migrator')
        cur.execute('SELECT rolsuper FROM pg_roles WHERE rolname=current_user')
        assert cur.fetchone() == (False,)
        apply(conn)
        cur.execute('RESET ROLE')
        after = state(cur)
        changed = [(a, b) for a, b in zip(before[0], after[0]) if a != b]
        assert {a[0] for a, _ in changed} == {s.removeprefix('public.') for s in SIGNATURES}
        assert all(a[:3] == b[:3] and a[4:] == b[4:] and b[3] == STRICT for a, b in changed)
        assert before[1:] == after[1:]
        assert_hardened(cur)
        assert data_rows(cur) == canonical
        apply(conn)
        assert state(cur) == after
        # The unchanged488 migration now applies normally after489.
        apply(conn, RETRY_SQL)
        post = state(cur)
        apply(conn)
        assert state(cur) == post
        assert_hardened(cur)
        assert data_rows(cur) == canonical


@pytest.mark.parametrize('role', PUBLIC_ROLES)
def test_public_roles_cannot_execute_either_rpc_after_hardening(database, role):
    with database[0].connect(**database[1]) as conn, conn.cursor() as cur:
        apply(conn)
        assert_hardened(cur)
        for expression, args in [(PUBLISH, (None,)*5), (RECORD, (None,)*7)]:
            with pytest.raises(database[0].errors.InsufficientPrivilege, match='permission denied for function'):
                rpc_as(conn, role, expression, args)
        cur.execute('SELECT count(*) FROM public.lab_arena_chain_outcomes')
        assert cur.fetchone() == (0,)


def test_dedicated_service_publishes_and_authenticates_semantic_reports_after_both_migrations(database):
    with database[0].connect(**database[1]) as conn:
        apply(conn)
        apply(conn, RETRY_SQL)
    with prior.service(database) as app:
        accepted = prior.accepted(app)
        first = prior.report(accepted)
        assert app.record_chain_outcome(first) == {'status': 'recorded'}
        canonical = prior.rows(database)
        retry = prior.report(accepted, observed_at='2026-10-15T16:08:19Z')
        assert app.record_chain_outcome(retry) == {'status': 'recorded'}
        retry['signature'] = '0x'+'00'*64
        with pytest.raises(prior.ServiceError, match='signature_invalid'):
            app.record_chain_outcome(retry)
        assert prior.rows(database) == canonical


@pytest.mark.parametrize('postimage', [False, True])
@pytest.mark.parametrize('strict_record,strict_publish', [(False, False), (True, False), (False, True), (True, True)])
def test_reviewed_legacy_and_strict_shapes_are_idempotent_before_or_after488(database, modeled_database, postimage, strict_record, strict_publish):
    with database[0].connect(**database[1]) as conn, conn.cursor() as cur:
        if postimage:
            cur.execute(modeled_database[2])
        for signature, strict in zip(SIGNATURES, (strict_record, strict_publish)):
            if strict:
                cur.execute('REVOKE EXECUTE ON FUNCTION '+signature+' FROM anon,authenticated,service_role')
        conn.commit()
        before = state(cur)
        apply(conn)
        after = state(cur)
        assert [r[:3]+r[4:] for r in before[0]] == [r[:3]+r[4:] for r in after[0]]
        assert before[1:] == after[1:]
        assert_hardened(cur)
        apply(conn)
        assert state(cur) == after


@pytest.mark.parametrize('target', [0, 1])
@pytest.mark.parametrize('fault', ['public', 'extra_role', 'grant_option', 'partial_acl', 'missing_service',
                                 'owner', 'definer', 'search_path', 'volatility', 'code', 'missing'])
def test_unknown_authority_fails_before_any_revoke(database, target, fault):
    with database[0].connect(**database[1]) as conn, conn.cursor() as cur:
        original = state(cur)
        signature = SIGNATURES[target]
        if fault == 'public': cur.execute('GRANT EXECUTE ON FUNCTION '+signature+' TO PUBLIC')
        elif fault == 'extra_role':
            cur.execute('CREATE ROLE unexpected489 NOLOGIN')
            cur.execute('GRANT EXECUTE ON FUNCTION '+signature+' TO unexpected489')
        elif fault == 'grant_option': cur.execute('GRANT EXECUTE ON FUNCTION '+signature+' TO anon WITH GRANT OPTION')
        elif fault == 'partial_acl': cur.execute('REVOKE EXECUTE ON FUNCTION '+signature+' FROM anon')
        elif fault == 'missing_service': cur.execute('REVOKE EXECUTE ON FUNCTION '+signature+' FROM lab_arena_service')
        elif fault == 'owner': cur.execute('ALTER FUNCTION '+signature+' OWNER TO postgres')
        elif fault == 'definer': cur.execute('ALTER FUNCTION '+signature+' SECURITY INVOKER')
        elif fault == 'search_path': cur.execute('ALTER FUNCTION '+signature+' SET search_path=public,pg_catalog')
        elif fault == 'volatility': cur.execute('ALTER FUNCTION '+signature+' STABLE')
        elif fault == 'code':
            cur.execute('SELECT pg_get_functiondef(%s::regprocedure)', (signature,))
            cur.execute(cur.fetchone()[0].replace('DECLARE existing', '-- unexpected drift\nDECLARE existing'))
        elif fault == 'missing': cur.execute('DROP FUNCTION '+signature)
        with pytest.raises(database[0].Error, match='authority differs'):
            apply(conn)
        conn.rollback()
        assert state(cur) == original


@pytest.mark.parametrize('role', PUBLIC_ROLES)
def test_effective_inherited_execute_is_not_hidden_by_strict_direct_acl(database, role):
    with database[0].connect(**database[1]) as conn, conn.cursor() as cur:
        original = state(cur)
        cur.execute('GRANT lab_arena_service TO '+role)
        with pytest.raises(database[0].Error, match='privilege readback differs'):
            apply(conn)
        conn.rollback()
        assert state(cur) == original


def test_gateway_schema_usage_is_required(database):
    with database[0].connect(**database[1]) as conn, conn.cursor() as cur:
        original = state(cur)
        cur.execute('REVOKE USAGE ON SCHEMA public FROM PUBLIC,lab_arena_service')
        with pytest.raises(database[0].Error, match='privilege readback differs'):
            apply(conn)
        conn.rollback()
        assert state(cur) == original
