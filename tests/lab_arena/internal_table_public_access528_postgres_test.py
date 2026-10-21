"""Public expired reads survive while API writes and internal access are denied."""
from pathlib import Path

import pytest

from tests.postgres_migration_harness import _database_with_migrations

SQL = Path(__file__).resolve().parents[2] / "scripts/528-lab-arena-internal-table-public-access.sql"
TABLES = (
    "lab_arena_sep15_baseline_rerun_audit",
    "lab_arena_sep15_rerun_release_authority",
    "lab_arena_sep16_open_config_audit",
)
VIEW = "qualification_expired_icp_sets"


@pytest.fixture(scope="module")
def database():
    yield from _database_with_migrations(())


def query_as(conn, role, query):
    with conn.cursor() as cur:
        cur.execute("SET ROLE " + role)
        try:
            cur.execute(query)
            return cur.fetchall() if cur.description else None
        finally:
            cur.execute("RESET ROLE")


def test_production_permissions_and_data_survive_idempotent_migration(database):
    pg, dsn = database
    conn = pg.connect(**dsn)
    conn.autocommit = True
    try:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE ROLE lab_arena_owner NOLOGIN;
                CREATE ROLE leadpoet_parity_reader NOLOGIN BYPASSRLS;
                ALTER ROLE service_role BYPASSRLS;
                GRANT USAGE ON SCHEMA public TO anon, authenticated,
                    service_role, lab_arena_owner, leadpoet_parity_reader;
                ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON TABLES TO anon, service_role;
                CREATE TABLE public.qualification_private_icp_sets (
                    set_id int PRIMARY KEY, is_active boolean, icps jsonb
                );
                INSERT INTO public.qualification_private_icp_sets VALUES
                    (1, false, '["expired"]'), (2, true, '["private"]');
                ALTER TABLE public.qualification_private_icp_sets ENABLE ROW LEVEL SECURITY;
                REVOKE ALL ON public.qualification_private_icp_sets FROM anon;
                CREATE VIEW public.qualification_expired_icp_sets
                    WITH (security_invoker=false) AS
                    SELECT set_id, icps FROM public.qualification_private_icp_sets
                    WHERE is_active=false ORDER BY set_id DESC;
                GRANT SELECT ON public.qualification_expired_icp_sets TO authenticated, leadpoet_parity_reader;
            """)
            for name in TABLES:
                cur.execute(f"CREATE TABLE public.{name} (id int PRIMARY KEY)")
                cur.execute(f"ALTER TABLE public.{name} OWNER TO lab_arena_owner")
                cur.execute(f"GRANT SELECT ON public.{name} TO leadpoet_parity_reader")
                cur.execute(f"INSERT INTO public.{name} VALUES (1)")
            cur.execute("SELECT pg_get_viewdef(%s::regclass),reloptions,relowner FROM pg_class WHERE oid=%s::regclass", (VIEW, VIEW))
            view_before = cur.fetchone()
            cur.execute("SELECT defaclacl::text FROM pg_default_acl ORDER BY oid")
            defaults_before = cur.fetchall()

        # Reproduce the actual vulnerability using a no-op write in a fixture.
        assert query_as(conn, "anon", f"UPDATE public.{VIEW} SET icps=icps RETURNING set_id") == [(1,)]
        assert query_as(conn, "anon", f"SELECT id FROM public.{TABLES[0]}") == [(1,)]

        for _ in range(2):
            with conn.cursor() as cur:
                cur.execute(SQL.read_text())
            for role in ("anon", "authenticated", "service_role", "leadpoet_parity_reader"):
                assert query_as(conn, role, f"SELECT set_id FROM public.{VIEW}") == [(1,)]
            for role in ("anon", "authenticated", "service_role"):
                for statement in (
                    f"INSERT INTO public.{VIEW} VALUES (3, '[]')",
                    f"UPDATE public.{VIEW} SET icps='[]'",
                    f"DELETE FROM public.{VIEW}",
                ):
                    with pytest.raises(pg.errors.InsufficientPrivilege):
                        query_as(conn, role, statement)
            for role in ("anon", "authenticated"):
                for name in (*TABLES, "qualification_private_icp_sets"):
                    for statement in (
                        f"SELECT * FROM public.{name}",
                        f"DELETE FROM public.{name}",
                    ):
                        with pytest.raises(pg.errors.InsufficientPrivilege):
                            query_as(conn, role, statement)
            for role in ("service_role", "lab_arena_owner", "leadpoet_parity_reader"):
                for name in TABLES:
                    assert query_as(conn, role, f"SELECT id FROM public.{name}") == [(1,)]
            for role in ("service_role", "lab_arena_owner"):
                for name in TABLES:
                    assert query_as(conn, role, f"UPDATE public.{name} SET id=id RETURNING id") == [(1,)]
            with conn.cursor() as cur:
                cur.execute("SELECT pg_get_viewdef(%s::regclass),reloptions,relowner FROM pg_class WHERE oid=%s::regclass", (VIEW, VIEW))
                assert cur.fetchone() == view_before
                cur.execute("SELECT defaclacl::text FROM pg_default_acl ORDER BY oid")
                assert cur.fetchall() == defaults_before
                cur.execute("SELECT set_id,is_active,icps FROM public.qualification_private_icp_sets ORDER BY set_id")
                assert cur.fetchall() == [(1, False, ["expired"]), (2, True, ["private"])]
                for name in TABLES:
                    cur.execute("SELECT relrowsecurity,pg_get_userbyid(relowner) FROM pg_class WHERE oid=%s::regclass", (name,))
                    assert cur.fetchone() == (True, "lab_arena_owner")
                    for role in ("anon", "authenticated"):
                        for privilege in ("SELECT", "INSERT", "UPDATE", "DELETE", "TRUNCATE", "REFERENCES", "TRIGGER"):
                            cur.execute("SELECT has_table_privilege(%s,%s,%s)", (role, name, privilege))
                            assert cur.fetchone() == (False,)
    finally:
        conn.close()


def test_fresh_install_without_legacy_objects():
    db = _database_with_migrations((SQL.name,))
    try:
        pg, dsn = next(db)
        with pg.connect(**dsn) as conn, conn.cursor() as cur:
            cur.execute("SELECT to_regclass('public.qualification_expired_icp_sets')")
            assert cur.fetchone() == (None,)
    finally:
        db.close()
