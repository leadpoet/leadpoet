"""Historical fixture order must preserve the real SQL420 migration guards."""

import hashlib
from pathlib import Path
import re

import pytest

from tests.lab_arena import oct07_cutoff_recovery419_postgres_test as historical


SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
CUTOFF_SQL = (SCRIPTS / "420-lab-arena-bounded-stage-cutoff-drain.sql").read_text()
PARTIAL_SQL = (SCRIPTS / "465-lab-arena-partial-company-judgments.sql").read_text()
PATCHES = re.findall(r"\('([^']+)', '([0-9a-f]{64})', '([0-9a-f]{64})'\)", CUTOFF_SQL)
assert len(PATCHES) == 4


@pytest.fixture()
def database():
    generator = historical.base_database.__wrapped__()
    value = next(generator)
    try:
        historical.database.__wrapped__(value)
        yield value
    finally:
        generator.close()


def _state(cursor):
    result = {}
    for signature, _, _ in PATCHES:
        cursor.execute(
            "SELECT pg_get_functiondef(p.oid), owner.rolname, p.proacl::text, "
            "p.prosecdef, p.provolatile, p.proconfig "
            "FROM pg_proc p JOIN pg_roles owner ON owner.oid=p.proowner "
            "WHERE p.oid=%s::regprocedure", (signature,),
        )
        result[signature] = cursor.fetchone()
    return result


def test_historical_bootstrap_reaches_exact_cutoff_preimages_and_keeps_security(database):
    psycopg, dsn = database
    with psycopg.connect(**dsn) as connection, connection.cursor() as cursor:
        before = _state(cursor)
        for signature, preimage, _ in PATCHES:
            assert hashlib.sha256(before[signature][0].encode()).hexdigest() == preimage
        cursor.execute(CUTOFF_SQL)
        after = _state(cursor)
        for signature, _, postimage in PATCHES:
            assert hashlib.sha256(after[signature][0].encode()).hexdigest() == postimage
            assert after[signature][1:] == before[signature][1:]
        cursor.execute(CUTOFF_SQL)
        assert _state(cursor) == after


def test_out_of_order_partial_judgment_migration_still_rejects_cutoff_without_changes(database):
    psycopg, dsn = database
    with psycopg.connect(**dsn) as connection, connection.cursor() as cursor:
        before = _state(cursor)
        cursor.execute(PARTIAL_SQL)
        changed = _state(cursor)
        completion = "public.lab_arena_complete_attempt(text,text,jsonb,text,text)"
        assert changed[completion][0] != before[completion][0]
        assert changed[completion][1:] == before[completion][1:]
        assert all(changed[key] == value for key, value in before.items() if key != completion)
        with pytest.raises(psycopg.errors.RaiseException, match="stage-cutoff drain preimage differs"):
            cursor.execute(CUTOFF_SQL)
        connection.rollback()
        assert _state(cursor) == changed
