"""Additive v3 count of calls with reasoning on the actual PostgreSQL ledger/reader path."""
from pathlib import Path

import pytest

from tests.lab_arena.capture_summary593_postgres_test import READER, ROUND, _catalog, database, migrated
from tests.lab_arena.capture_summary598_postgres_test import SQL as SQL598, SQL595, _seed_bodies

SQL = Path(__file__).resolve().parents[2] / "scripts/683-lab-arena-capture-summary-v3.sql"
V1 = "public.lab_arena_capture_summary_v1(text,text)"
V2 = "public.lab_arena_capture_summary_v2(text,text)"
V3 = "public.lab_arena_capture_summary_v3(text,text)"


@pytest.fixture(scope="module")
def upgraded(database, migrated):
    psycopg2, dsn = database
    with psycopg2.connect(**dsn) as connection, connection.cursor() as cur:
        cur.execute(SQL595.read_text())
        cur.execute(SQL598.read_text())
        before = _catalog(cur)
        cur.execute(SQL.read_text())
        first = _catalog(cur)
        cur.execute(SQL.read_text())
        second = _catalog(cur)
    return before, first, second


def _read(database, version, submission, role=READER):
    psycopg2, dsn = database
    with psycopg2.connect(**dsn) as connection, connection.cursor() as cur:
        cur.execute("SET ROLE " + role)
        cur.execute(f"SELECT to_jsonb(s) FROM public.lab_arena_capture_summary_{version}(%s,%s) s", (ROUND, submission))
        rows = cur.fetchall()
    assert len(rows) == 1
    return rows[0][0]


def _reasoning(tokens, field="reasoning_tokens"):
    if field == "completion_details":
        usage = {"completion_tokens_details": {"reasoning_tokens": tokens}}
    elif field == "output_details":
        usage = {"output_tokens_details": {"reasoning_tokens": tokens}}
    else:
        usage = {field: tokens}
    return {"model": "test/model", "usage": usage}


def test_additive_idempotent_and_leaves_earlier_readers_alone(upgraded):
    before, first, second = upgraded
    assert first == second
    assert set(first[0]) - set(before[0]) == {"lab_arena_capture_summary_v3(text,text)"}
    assert all(first[0][name] == function for name, function in before[0].items())
    assert first[1:] == before[1:]


def test_counts_only_calls_with_positive_reported_reasoning(database, upgraded):
    submission = _seed_bodies(database, [
        _reasoning(7), _reasoning(9, "completion_details"), _reasoning(11, "output_details"),
        _reasoning(0), _reasoning(0, "completion_details"), _reasoning(-1), _reasoning("12"), _reasoning(1.5),
        {"model": "test/model"}, b"not json",
    ])
    summary = _read(database, "v3", submission)
    assert (summary["model_calls"], summary["reasoning_calls"]) == (10, 3)
    assert summary["reasoning_tokens"] == 27


def test_no_call_submission_reports_zero(database, upgraded):
    summary = _read(database, "v3", _seed_bodies(database, []))
    assert (summary["model_calls"], summary["reasoning_calls"]) == (0, 0)


def test_every_v2_column_is_unchanged(database, upgraded):
    submission = _seed_bodies(database, [
        {"model": "a/b", "usage": {"prompt_tokens": 10, "completion_tokens": 8, "reasoning_tokens": 3}},
        {"model": "c/d", "usage": {"prompt_tokens": 4, "completion_tokens": 2}},
    ])
    v2, v3 = _read(database, "v2", submission), _read(database, "v3", submission)
    assert v3.pop("reasoning_calls") == 1
    assert v3 == v2


def test_reader_acl_is_the_three_aggregate_functions(database, upgraded):
    psycopg2, dsn = database
    with psycopg2.connect(**dsn) as connection, connection.cursor() as cur:
        cur.execute("SELECT p.oid::regprocedure::text FROM pg_proc p WHERE p.pronamespace='public'::regnamespace "
                    "AND EXISTS (SELECT 1 FROM aclexplode(p.proacl) a WHERE a.grantee=%s::regrole) ORDER BY 1",
                    (READER,))
        assert [name for (name,) in cur.fetchall()] == [
            "lab_arena_capture_summary_v1(text,text)",
            "lab_arena_capture_summary_v2(text,text)",
            "lab_arena_capture_summary_v3(text,text)",
        ]


@pytest.mark.parametrize("role", ["anon", "authenticated", "service_role", "lab_arena_service"])
def test_other_roles_cannot_execute_v3(database, upgraded, role):
    psycopg2, _ = database
    with pytest.raises(psycopg2.errors.InsufficientPrivilege):
        _read(database, "v3", "any", role=role)


@pytest.mark.parametrize("drift", ["v2_body", "v2_acl", "table_grant"])
def test_migration_fails_closed_on_drift(database, upgraded, drift):
    psycopg2, dsn = database
    with psycopg2.connect(**dsn) as connection, connection.cursor() as cur:
        if drift == "v2_body":
            cur.execute("SELECT pg_get_functiondef(%s::regprocedure)", (V2,))
            cur.execute(cur.fetchone()[0].replace("BEGIN", "BEGIN\n -- unexpected", 1))
        elif drift == "v2_acl":
            cur.execute(f"GRANT EXECUTE ON FUNCTION {V2} TO authenticated")
        else:
            cur.execute("GRANT SELECT ON public.lab_arena_ledger TO leadpoet_capture_reader")
        with pytest.raises(psycopg2.Error, match="(preimage|security) differs|holds other privileges"):
            cur.execute(SQL.read_text())
        connection.rollback()
