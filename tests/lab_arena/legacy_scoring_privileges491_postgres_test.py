"""The retained scoring backup cannot bypass authenticated gateway admission."""

from pathlib import Path

import pytest

from tests.lab_arena import progression_lock_order490_postgres_test as locks


ROOT = Path(__file__).resolve().parents[2]
SQL = ROOT / "scripts/491-lab-arena-legacy-scoring-rpc-privileges.sql"
SIGNATURE = "public.lab_arena_open_scoring_sep15_baseline_only_v1(text,smallint,jsonb)"
LEGACY = "{lab_arena_owner=X/lab_arena_owner,anon=X/lab_arena_owner,authenticated=X/lab_arena_owner,service_role=X/lab_arena_owner}"
STRICT = "{lab_arena_owner=X/lab_arena_owner}"
ROLES = ("anon", "authenticated", "service_role", "lab_arena_service")
database = locks.database


def apply(conn, path=SQL):
    with conn.cursor() as cur:
        cur.execute(path.read_text())
    conn.commit()


def metadata(cur):
    cur.execute("SELECT p.oid::regprocedure::text,to_jsonb(p) FROM pg_proc p "
                "WHERE p.pronamespace='public'::regnamespace ORDER BY p.oid")
    return dict(cur.fetchall())


@pytest.fixture(scope="module")
def original(database):
    with database[0].connect(**database[1]) as conn, conn.cursor() as cur:
        cur.execute("SELECT pg_get_functiondef(%s::regprocedure)", (SIGNATURE,))
        return cur.fetchone()[0]


@pytest.fixture
def restored(database, original):
    with database[0].connect(**database[1]) as conn, conn.cursor() as cur:
        cur.execute(original)
        cur.execute("REVOKE ALL ON FUNCTION " + SIGNATURE + " FROM PUBLIC,anon,authenticated,service_role,lab_arena_service")
        cur.execute("GRANT EXECUTE ON FUNCTION " + SIGNATURE + " TO anon,authenticated,service_role")
        cur.execute("GRANT USAGE ON SCHEMA public TO anon,authenticated,service_role,lab_arena_service")
        cur.execute("SELECT proacl::text FROM pg_proc WHERE oid=%s::regprocedure", (SIGNATURE,))
        assert cur.fetchone() == (LEGACY,)
    return database


@pytest.mark.parametrize("lock_first", [False, True])
def test_real_anonymous_mutation_is_blocked_and_both_migrations_replay(restored, lock_first):
    psycopg, dsn = restored
    round_id = "arena-2040-01-0" + ("2" if lock_first else "1")
    with psycopg.connect(**dsn) as conn, conn.cursor() as cur:
        if lock_first:
            apply(conn, locks.SQL)
        cur.execute("INSERT INTO public.lab_arena_rounds "
                    "(round_id,status,status_generation,stage_generation,configuration_doc,participants,benchmark_ref,rewards_enabled) "
                    "VALUES (%s,'stage1_closed',1,1,'{\"integrity_policy\":\"arena_integrity_v1\","
                    "\"cost_per_company_microusd\":800000,\"scorer_policy\":{\"scoring_adapter_version\":\"qualification_integrity_v2\"}}', '[]',%s,false)",
                    (round_id, "arena/" + round_id + "/benchmark.json"))
        conn.commit()
        # This is the exact unguarded state mutation, in a disposable database.
        cur.execute("SET LOCAL ROLE anon")
        cur.execute("SELECT public.lab_arena_open_scoring_sep15_baseline_only_v1(%s,1::smallint,'[]'::jsonb)", (round_id,))
        assert cur.fetchone()[0]["round_status"] == "stage1_scoring"
        conn.rollback()
        before = metadata(cur)
        apply(conn)
        after = metadata(cur)
        changed = [key for key in before if before[key] != after[key]]
        assert changed == [SIGNATURE.removeprefix("public.")]
        old = before[changed[0]].copy()
        new = after[changed[0]].copy()
        assert old.pop("proacl") != new.pop("proacl")
        assert old == new
        cur.execute("SELECT proacl::text FROM pg_proc WHERE oid=%s::regprocedure", (SIGNATURE,))
        assert cur.fetchone() == (STRICT,)
        for role in ROLES:
            cur.execute("SELECT has_schema_privilege(%s,'public','USAGE'),has_function_privilege(%s,%s,'EXECUTE')", (role, role, SIGNATURE))
            assert cur.fetchone() == (True, False)
            cur.execute("SET LOCAL ROLE " + role)
            with pytest.raises(psycopg.errors.InsufficientPrivilege, match="permission denied for function"):
                cur.execute("SELECT public.lab_arena_open_scoring_sep15_baseline_only_v1(%s,1::smallint,'[]'::jsonb)", (round_id,))
            conn.rollback()
        apply(conn, locks.SQL)
        apply(conn)
        stable = metadata(cur)
        apply(conn, locks.SQL)
        apply(conn)
        assert metadata(cur) == stable
        cur.execute("SELECT status,status_generation,stage_generation FROM public.lab_arena_rounds WHERE round_id=%s", (round_id,))
        assert cur.fetchone() == ("stage1_closed", 1, 1)
        cur.execute("SELECT has_function_privilege('lab_arena_owner',%s,'EXECUTE'),"
                    "has_function_privilege('lab_arena_service','public.lab_arena_open_scoring_v3(text,smallint,jsonb)','EXECUTE')", (SIGNATURE,))
        assert cur.fetchone() == (True, True)


@pytest.mark.parametrize("fault", ["public", "service", "owner", "invoker", "path", "body", "missing"])
def test_unreviewed_authority_rolls_back_without_other_changes(restored, fault):
    psycopg, dsn = restored
    with psycopg.connect(**dsn) as conn, conn.cursor() as cur:
        before = metadata(cur)
        if fault == "public":
            cur.execute("GRANT EXECUTE ON FUNCTION " + SIGNATURE + " TO PUBLIC")
        elif fault == "service":
            cur.execute("GRANT EXECUTE ON FUNCTION " + SIGNATURE + " TO lab_arena_service")
        elif fault == "owner":
            cur.execute("ALTER FUNCTION " + SIGNATURE + " OWNER TO postgres")
        elif fault == "invoker":
            cur.execute("ALTER FUNCTION " + SIGNATURE + " SECURITY INVOKER")
        elif fault == "path":
            cur.execute("ALTER FUNCTION " + SIGNATURE + " SET search_path=public,pg_catalog")
        elif fault == "body":
            cur.execute("SELECT pg_get_functiondef(%s::regprocedure)", (SIGNATURE,))
            cur.execute(cur.fetchone()[0].replace("DECLARE", "-- unreviewed drift\nDECLARE", 1))
        elif fault == "missing":
            cur.execute("DROP FUNCTION " + SIGNATURE)
        with pytest.raises(psycopg.Error, match="legacy scoring491 authority differs"):
            apply(conn)
        conn.rollback()
        assert metadata(cur) == before


def test_owner_member_can_harden_without_superuser(restored):
    psycopg, dsn = restored
    with psycopg.connect(**dsn) as conn, conn.cursor() as cur:
        cur.execute("CREATE ROLE legacy491_migrator NOLOGIN NOSUPERUSER INHERIT")
        cur.execute("GRANT lab_arena_owner TO legacy491_migrator")
        conn.commit()
        cur.execute("SET ROLE legacy491_migrator")
        cur.execute("SELECT rolsuper FROM pg_roles WHERE rolname=current_user")
        assert cur.fetchone() == (False,)
        apply(conn)
        cur.execute("RESET ROLE")
        cur.execute("SELECT proacl::text FROM pg_proc WHERE oid=%s::regprocedure", (SIGNATURE,))
        assert cur.fetchone() == (STRICT,)


def test_inherited_execute_fails_readback_and_rolls_back(restored):
    psycopg, dsn = restored
    with psycopg.connect(**dsn) as conn, conn.cursor() as cur:
        before = metadata(cur)
        cur.execute("GRANT lab_arena_owner TO anon")
        with pytest.raises(psycopg.Error, match="legacy scoring491 privilege readback differs"):
            apply(conn)
        conn.rollback()
        assert metadata(cur) == before
        cur.execute("SELECT pg_has_role('anon','lab_arena_owner','MEMBER')")
        assert cur.fetchone() == (False,)
