"""The real governed PostgreSQL allocator consults the owner-controlled sandbox admission."""

import os
from pathlib import Path
from psycopg2.extras import Json
import pytest
from lab_arena.collection_programs import _hash, sandbox
from tests.lab_arena.collection_pool_reward_rate_pg_test import migrate
from tests.lab_arena.collection_standing_pool_pg_test import db, pool


def test_774_catalog_fails_closed_and_preserves_allocator_permissions(pool):
    migrate(pool)
    scripts = Path(__file__).resolve().parents[2] / "scripts"
    with pool.driver.connect(**pool.dsn) as conn, conn.cursor() as c:
        c.execute("SET ROLE rate_test_migrator")
        c.execute(
            Path(
                os.getenv(
                    "COLLECTION_TEST_773_SQL",
                    scripts / "773-governed-standing-workers.sql",
                )
            ).read_text()
        )
        c.execute((scripts / "774-registered-sales-sandboxes.sql").read_text())
        for f in ("P7", "P8", "P9", "P10"):
            c.execute("SELECT public.collection_registered_sandbox_v1(%s)", (f,))
            assert c.fetchone()[0] == {
                "kind": "train",
                "registry_sha256": _hash(sandbox(f)),
            }
        for sid, allowed in [
            ("P7-s10000000", True),
            ("P8-s20000002", True),
            ("P9-s30000004", True),
            ("P9-s81000", True),
            ("P10-s40000007", False),
            ("P7-s199", False),
            ("P77-s10000000", False),
            ("C06-s3001", False),
        ]:
            c.execute("SELECT public.collection_trainable_scenario_v1(%s)", (sid,))
            assert c.fetchone()[0] is allowed
        c.execute(
            "INSERT INTO public.collection_registered_sandboxes VALUES ('P77','evaluation',%s),('P88','train',%s)",
            ("sha256:" + "b" * 64, "sha256:" + "a" * 64),
        )
        c.execute(
            "SELECT public.collection_trainable_scenario_v1('P77-s10000000'), public.collection_trainable_scenario_v1('P88-s10000000')"
        )
        assert c.fetchone() == (False, True)
        c.execute(
            "SELECT pg_get_functiondef('public.collection_governed_claim_v1(uuid,text,text,text,text,jsonb,jsonb,jsonb,jsonb,text,text,bigint,jsonb)'::regprocedure)"
        )
        assert c.fetchone()[0].count("collection_trainable_scenario_v1(s.id)") == 2
        c.execute(
            "SELECT has_table_privilege('lab_arena_service','public.collection_registered_sandboxes','INSERT'),has_function_privilege('lab_arena_service','public.collection_registered_sandbox_v1(text)','EXECUTE')"
        )
        assert c.fetchone() == (False, True)
