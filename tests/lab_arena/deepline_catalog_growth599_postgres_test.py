"""Both real catalog commit RPCs admit growth and retain their existing guards."""
from copy import deepcopy
import hashlib
from pathlib import Path

import pytest

from lab_arena import deepline_catalog as catalog
from lab_arena.store import ArenaStore, ArenaStoreError, PsycopgTransport
from tests.lab_arena import full_thinking_catalog_commit568_postgres_test as prior
from tests.lab_arena.deepline_catalog_growth_test import growth_rows


MIGRATION = Path(__file__).parents[2] / "scripts/599-lab-arena-deepline-catalog-growth.sql"
SIGNATURES = (
    "public.lab_arena_commit_round_v3(text,jsonb,text,text,date,text,text,jsonb)",
    "public.lab_arena_commit_round_v4(text,jsonb,text,text,date,text,text,jsonb,text)",
)


@pytest.fixture(scope="module")
def database():
    fixture = prior.database.__wrapped__()
    db = next(fixture)
    try:
        with db[0].connect(**db[1]) as connection, connection.cursor() as cursor:
            cursor.execute(prior.MIGRATION.read_text())
        yield db
    finally:
        fixture.close()


def _functions(cursor):
    cursor.execute(
        "SELECT p.oid::regprocedure::text, pg_get_functiondef(p.oid), "
        "pg_get_userbyid(p.proowner), p.proacl::text, p.prosecdef, "
        "p.provolatile, p.proconfig, l.lanname "
        "FROM pg_proc p JOIN pg_language l ON l.oid=p.prolang "
        "WHERE p.oid=ANY(%s::regprocedure[]) ORDER BY p.oid::regprocedure::text",
        (list(SIGNATURES),),
    )
    return cursor.fetchall()


def test_real_commits_accept_complete_grown_catalog_with_unchanged_guards(database):
    psycopg, dsn = database
    store = ArenaStore(PsycopgTransport(lambda: psycopg.connect(**dsn)))
    snapshot = catalog.fetch_catalog(read_json=lambda *_: {"tools": growth_rows()})
    pending = []
    # Reproduce both database failures independently of the fixed Python path.
    for suffix, v4 in ((1, False), (2, True)):
        round_id, args = prior._open_catalog_round(
            store, database, suffix=suffix, cap=6_000_000, full=True, v4=v4)
        args["deepline_catalog"] = snapshot
        with pytest.raises(ArenaStoreError, match="lab_arena_deepline_catalog_commit_invalid"):
            store.commit_round_v2(round_id, **args)
        assert store.get_round(round_id)["status"] == "open"
        pending.append((round_id, args, store.get_round(round_id)))

    with psycopg.connect(**dsn) as connection, connection.cursor() as cursor:
        before = _functions(cursor)
        cursor.execute(MIGRATION.read_text())
        cursor.execute(MIGRATION.read_text())  # Idempotent exact postimage.
        after = _functions(cursor)
    old = "pg_catalog.jsonb_array_length(p_deepline_catalog -> 'tools') NOT BETWEEN 1 AND 4096"
    new = old.replace("4096", str(catalog.MAX_CATALOG_TOOLS))
    assert len(before) == len(after) == 2
    for original, updated in zip(before, after):
        assert original[0] == updated[0]
        assert original[1].count(old) == 1
        assert updated[1] == original[1].replace(old, new)
        assert updated[2:] == original[2:]  # Owner, ACL, definer and settings.
        assert hashlib.sha256(updated[1].encode()).hexdigest() in MIGRATION.read_text()
    for round_id, args, original in pending:
        assert store.get_round(round_id) == original  # Migration edits no rows.
        assert store.commit_round_v2(round_id, **args)["status"] == "ok"
        saved = store.get_round(round_id)
        assert saved["status"] == "committed"
        assert saved["configuration_doc"]["deepline_catalog"] == snapshot

    # Frozen authority and bounds still fail closed, in each RPC variant.
    cases = []
    for kind in ("count", "bytes", "policy", "people", "tools"):
        for v4 in (False, True):
            cases.append((kind, v4))
    for suffix, (kind, v4) in enumerate(cases, 3):
        round_id, args = prior._open_catalog_round(
            store, database, suffix=suffix, cap=6_000_000, full=True, v4=v4)
        broken = deepcopy(prior.CATALOG)
        if kind == "count":
            broken["tools"] *= catalog.MAX_CATALOG_TOOLS + 1
        elif kind == "bytes":
            broken["padding"] = "x" * 4_194_304
        elif kind == "policy":
            broken["policy_version"] = "untrusted-policy"
        elif kind == "people":
            broken["allow_people"] = True
        else:
            broken["tools"] = {}
        args["deepline_catalog"] = broken
        with pytest.raises(ArenaStoreError, match="lab_arena_deepline_catalog_commit_invalid"):
            store.commit_round_v2(round_id, **args)
        assert store.get_round(round_id)["status"] == "open"

    # A valid count at the precise new bound remains possible under the
    # independent byte limit. The database keeps its existing shallow shape
    # checks; the host validates the complete schema and policy above.
    for suffix, v4 in ((13, False), (14, True)):
        round_id, args = prior._open_catalog_round(
            store, database, suffix=suffix, cap=4_000_000, full=False, v4=v4)
        bounded = deepcopy(prior.CATALOG)
        bounded["tools"] = [{"tool_id": "company_%d" % i}
                            for i in range(catalog.MAX_CATALOG_TOOLS)]
        args["deepline_catalog"] = bounded
        assert store.commit_round_v2(round_id, **args)["status"] == "ok"


def test_migration_rejects_changed_authority_or_body(database):
    psycopg, dsn = database
    # Each transaction rolls back the deliberate disposable-database change.
    for mutation in (
        "ALTER FUNCTION " + SIGNATURES[0] + " SET search_path=public",
        None,
    ):
        connection = psycopg.connect(**dsn)
        try:
            with connection.cursor() as cursor:
                if mutation:
                    cursor.execute(mutation)
                else:
                    cursor.execute("SELECT pg_get_functiondef(%s::regprocedure)", (SIGNATURES[0],))
                    body = cursor.fetchone()[0]
                    assert "16384" in body
                    cursor.execute(body.replace("16384", "16385"))
                with pytest.raises(psycopg.Error, match="Deepline catalog growth (authority|preimage) differs"):
                    cursor.execute(MIGRATION.read_text())
        finally:
            connection.rollback()
            connection.close()
