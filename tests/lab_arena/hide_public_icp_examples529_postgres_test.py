"""Public expired banks retain their requirements but omit generator answers."""
import json
from pathlib import Path

import pytest

from tests.postgres_migration_harness import _database_with_migrations

SQL = Path(__file__).resolve().parents[2] / "scripts/529-lab-arena-hide-public-icp-examples.sql"
SETUP = """
CREATE TABLE public.qualification_private_icp_sets (
    set_id int PRIMARY KEY, created_at timestamptz DEFAULT now(),
    active_from timestamptz, active_until timestamptz, icps jsonb, is_active boolean
);
ALTER TABLE public.qualification_private_icp_sets ENABLE ROW LEVEL SECURITY;
GRANT ALL ON public.qualification_private_icp_sets TO service_role;
ALTER ROLE service_role BYPASSRLS;
CREATE VIEW public.qualification_expired_icp_sets WITH (security_invoker=false) AS
SELECT set_id,created_at,active_from,active_until,icps
FROM public.qualification_private_icp_sets WHERE is_active=false ORDER BY set_id DESC;
GRANT ALL ON public.qualification_expired_icp_sets TO anon,service_role;
GRANT SELECT ON public.qualification_expired_icp_sets TO authenticated;
"""


def test_public_view_strips_examples_without_changing_private_data_or_permissions():
    db = _database_with_migrations(("528-lab-arena-internal-table-public-access.sql",), setup_sql=SETUP)
    try:
        pg, dsn = next(db)
        conn = pg.connect(**dsn)
        conn.autocommit = True
        try:
            with conn.cursor() as q:
                icps = [
                    {"icp_id": "first", "prompt": "Find workflow software", "verified_example_company": "PRIVATE EXAMPLE",
                     "intent_signals": ["Announced a product"], "excluded_companies": ["excluded.example"]},
                    {"icp_id": "second", "verified_example_company": ""},
                    {"icp_id": "third", "country": "Canada"},
                ]
                q.execute("INSERT INTO public.qualification_private_icp_sets(set_id,icps,is_active) VALUES (1,%s::jsonb,false),(2,'[]',false),(3,NULL,false),(4,%s::jsonb,true)", (json.dumps(icps),json.dumps(icps)))
                q.execute("SELECT to_jsonb(t) FROM public.qualification_private_icp_sets t ORDER BY set_id")
                private_before = q.fetchall()
                q.execute("SELECT relowner,relacl::text,reloptions FROM pg_class WHERE oid='public.qualification_expired_icp_sets'::regclass")
                permissions_before = q.fetchone()
                for _ in range(2):
                    q.execute(SQL.read_text())
                    q.execute("SELECT relowner,relacl::text,reloptions FROM pg_class WHERE oid='public.qualification_expired_icp_sets'::regclass")
                    assert q.fetchone() == permissions_before
                    q.execute("SELECT to_jsonb(t) FROM public.qualification_private_icp_sets t ORDER BY set_id")
                    assert q.fetchall() == private_before
                    for role in ("anon", "authenticated", "service_role"):
                        q.execute("SET ROLE " + role)
                        q.execute("SELECT set_id,icps FROM public.qualification_expired_icp_sets")
                        public_rows = q.fetchall()
                        assert public_rows == [(3,None),(2,[]),(1,[{k:v for k,v in i.items() if k!='verified_example_company'} for i in icps])]
                        assert "PRIVATE EXAMPLE" not in json.dumps(public_rows)
                        for statement in (
                            "INSERT INTO public.qualification_expired_icp_sets(set_id) VALUES (5)",
                            "UPDATE public.qualification_expired_icp_sets SET set_id=set_id",
                            "DELETE FROM public.qualification_expired_icp_sets",
                        ):
                            with pytest.raises(pg.errors.InsufficientPrivilege):
                                q.execute(statement)
                        q.execute("RESET ROLE")
                    q.execute("SET ROLE anon")
                    with pytest.raises(pg.errors.InsufficientPrivilege):
                        q.execute("SELECT * FROM public.qualification_private_icp_sets")
                    q.execute("RESET ROLE")
        finally:
            conn.close()
    finally:
        db.close()


def test_missing_legacy_view_is_safe():
    db = _database_with_migrations((SQL.name,))
    try:
        pg, dsn = next(db)
        with pg.connect(**dsn) as c, c.cursor() as q:
            q.execute("SELECT to_regclass('public.qualification_expired_icp_sets')")
            assert q.fetchone() == (None,)
    finally:
        db.close()
