"""Capture summary reader counts and privileges on disposable PostgreSQL."""
import base64
import hashlib
import json
from datetime import datetime, timedelta, timezone
from functools import partial
from pathlib import Path
from uuid import uuid4

import pytest

from lab_arena.broker import _has_successful_terminal_response
from lab_arena.training_export import _response_value
from lab_arena.training_thinking import attempt_coverage
from tests import postgres_migration_harness as pg_harness
from tests.lab_arena import lab_arena_pg_harness as arena_pg
from tests.lab_arena.lab_arena_pg_harness import current_service_database

SQL = Path(__file__).resolve().parents[2] / "scripts/593-lab-arena-capture-summary-reader.sql"
POSTGRES_IMAGE = "postgres:17.9@sha256:2a0d0fe14825b0939f78a8cad5cd4e6aa68bf94d0e5dd96e24b6d23af4315545"
FUNCTION = "public.lab_arena_capture_summary_v1(text,text)"
READER = "leadpoet_capture_reader"
ROUND, SUBMISSION, OTHER = "round-capture593", "sub-capture593", "sub-other593"
START = datetime(2026, 10, 30, 12, 0, tzinfo=timezone.utc)
RUNS = (  # run_id, submission, kind, icp_position, attempt
    ("run593-a", SUBMISSION, "execute", 1, 1),
    ("run593-b", SUBMISSION, "execute", 3, 1),
    ("run593-c", SUBMISSION, "execute", 3, 2),
    ("run593-score", SUBMISSION, "score", 1, 1),
    ("run593-other", OTHER, "execute", 2, 1),
)


def _body(document):
    raw = document if isinstance(document, bytes) else json.dumps(document).encode()
    return base64.b64encode(raw).decode()


def _ok(body, status=200):
    return {"call_succeeded": True, "status": status, "body_b64": body}


def _usage(**usage):
    return _body({"id": "gen", "choices": [{"message": {"content": "x"}}], "usage": usage})


FAILED = {"call_succeeded": False, "status": 502}
# name: (run, terminals as (entry_kind, terminal_response), provider, operation, reserved).
# The ledger admits one uncertain/recovery/refusal and one later settlement.
CASES = {
    "retried": ("run593-a", [("uncertain", FAILED),
                             ("settlement", _ok(_usage(completion_tokens_details={"reasoning_tokens": 100})))]),
    "latest_failed": ("run593-a", [("uncertain", _ok(_usage(reasoning_tokens=999))),
                                   ("settlement", _ok(_usage(reasoning_tokens=999), status=502))]),
    "failed": ("run593-a", [("settlement", {"call_succeeded": False, "status": 400,
                                            "body_b64": _usage(reasoning_tokens=50)})]),
    "no_reasoning": ("run593-a", [("settlement", _ok(_usage(prompt_tokens=5, completion_tokens=9)))]),
    "zero": ("run593-b", [("settlement", _ok(_usage(reasoning_tokens=0)))]),
    "invalid_json": ("run593-b", [("settlement", _ok(_body(b'{"usage":{"reasoning_tokens":4}')))]),
    "invalid_base64": ("run593-b", [("settlement", _ok("@@not base64@@"))]),
    "invalid_utf8": ("run593-b", [("settlement", _ok(_body(b'{"usage":{"reasoning_tokens":5},"x":"\xff"}')))]),
    "body_not_string": ("run593-b", [("settlement", _ok({"usage": {"reasoning_tokens": 6}}))]),
    "string_status": ("run593-b", [("settlement", {**_ok(_usage(reasoning_tokens=7)), "status": "200"})]),
    "escapes_and_bom": ("run593-c", [("settlement", _ok(_body(
        b'\xef\xbb\xbf{"usage":{"reasoning_tokens":7},"a":"\\u0000","b":"\\ud800","c":"\\\\ud800"}')))]),
    "responses_api": ("run593-c", [("settlement", _ok(_usage(output_tokens_details={"reasoning_tokens": 20})))],
                      "openrouter", "openrouter.responses"),
    "float": ("run593-c", [("settlement", _ok(_usage(reasoning_tokens=5.0)))]),
    "conflicting": ("run593-c", [("settlement", _ok(_usage(
        reasoning_tokens=0, completion_tokens_details={"reasoning_tokens": 30, "x": True})))]),
    "negative_and_bool": ("run593-c", [("settlement", _ok(_usage(
        reasoning_tokens=-4, completion_tokens_details={"reasoning_tokens": True})))]),
    "deepline": ("run593-c", [("settlement", _ok(_usage(reasoning_tokens=11)))], "deepline", "deepline.search"),
    "unreserved": ("run593-c", [("settlement", _ok(_usage(reasoning_tokens=12)))], "openrouter",
                   "openrouter.chat", False),
    "in_flight": ("run593-c", []),
    "refused": ("run593-c", [("refusal", None)]),
    "score_run": ("run593-score", [("settlement", _ok(_usage(reasoning_tokens=1000)))]),
    "other_submission": ("run593-other", [("settlement", _ok(_usage(reasoning_tokens=2000)))]),
}


def _spec(case):
    run_id, terminals, *rest = case
    provider, operation, reserved = (*rest, *("openrouter", "openrouter.chat", True)[len(rest):])
    return run_id, terminals, provider, operation, reserved


def _identity(name):
    return "sha256:" + hashlib.sha256(name.encode()).hexdigest()


def _python_tokens(terminal, operation):
    try:
        response, _ = _response_value(terminal, "openrouter")
    except Exception:
        response = None
    return attempt_coverage({"response": response, "operation_id": operation}, "miner")["tokens_reported"]


@pytest.fixture(scope="module")
def database():
    # Role membership options (WITH INHERIT FALSE, SET TRUE) need PostgreSQL 16; build on the deployed PG17.
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(arena_pg, "_database_with_migrations",
                      partial(pg_harness._database_with_migrations, image=POSTGRES_IMAGE))
        generator = current_service_database()
        database = next(generator)
    try:
        yield database
    finally:
        generator.close()


def _catalog(cur):
    cur.execute("SELECT p.oid::regprocedure::text, to_jsonb(p) - 'oid' FROM pg_proc p "
                "WHERE p.pronamespace = 'public'::regnamespace ORDER BY 1")
    functions = dict(cur.fetchall())
    cur.execute("SELECT c.oid::regclass::text, c.relacl::text FROM pg_class c "
                "WHERE c.relnamespace = 'public'::regnamespace ORDER BY 1")
    relations = cur.fetchall()
    cur.execute("SELECT nspacl::text FROM pg_namespace WHERE nspname = 'public'")
    schema = cur.fetchone()
    cur.execute("SELECT r.rolname, to_jsonb(r) - 'oid' FROM pg_roles r WHERE r.rolname <> %s ORDER BY 1", (READER,))
    roles = cur.fetchall()
    cur.execute("SELECT roleid::regrole::text, member::regrole::text, inherit_option, set_option "
                "FROM pg_auth_members ORDER BY 1, 2")
    return functions, relations, schema, roles, cur.fetchall()


@pytest.fixture(scope="module")
def migrated(database):
    psycopg2, dsn = database
    connection = psycopg2.connect(**dsn)
    connection.autocommit = True
    with connection, connection.cursor() as cur:
        cur.execute("CREATE ROLE authenticator NOINHERIT NOLOGIN")
        before = _catalog(cur)
        cur.execute(SQL.read_text(encoding="utf-8"))
        first = _catalog(cur)
        cur.execute(SQL.read_text(encoding="utf-8"))
        second = _catalog(cur)
        cur.execute("SET session_replication_role = replica")
        for run_id, submission, kind, position, attempt in RUNS:
            cur.execute(
                "INSERT INTO public.lab_arena_runs (run_id, assignment_id, round_id, submission_id, "
                "miner_hotkey, stage, icp_position, attempt, kind, scoring_generation_floor) "
                "VALUES (%s,%s,%s,%s,'hk593',1,%s,%s,%s,0)",
                (run_id, run_id, ROUND, submission, position, attempt, kind))
        clock, times = iter(range(10_000)), {}
        for name, case in CASES.items():
            run_id, terminals, provider, operation, reserved = _spec(case)
            submission = next(row[1] for row in RUNS if row[0] == run_id)
            entries = ([("reservation", None)] if reserved else []) + terminals
            times[name] = []
            for kind, terminal in entries:
                times[name].append(START + timedelta(seconds=next(clock)))
                cur.execute(
                    "INSERT INTO public.lab_arena_ledger (entry_kind, miner_hotkey, round_id, submission_id, "
                    "run_id, stage, call_identity, provider, operation_id, amount_microusd, terminal_response, "
                    "created_at) VALUES (%s,'hk593',%s,%s,%s,1,%s,%s,%s,0,%s,%s)",
                    (kind, ROUND, submission, run_id, _identity(name), provider, operation,
                     None if terminal is None else json.dumps(terminal), times[name][-1]))
        cur.execute("SET session_replication_role = origin")
    connection.close()
    return before, first, second, times


def _summary(database, round_id=ROUND, submission_id=SUBMISSION, role=READER):
    psycopg2, dsn = database
    with psycopg2.connect(**dsn) as connection, connection.cursor() as cur:
        cur.execute("SET ROLE " + role)
        cur.execute("SELECT to_jsonb(s) FROM public.lab_arena_capture_summary_v1(%s, %s) AS s",
                    (round_id, submission_id))
        rows = cur.fetchall()
    assert len(rows) == 1
    return rows[0][0]


def test_apply_is_idempotent_and_only_adds_the_reader_function(migrated):
    before, first, second, _ = migrated
    assert first == second
    assert set(first[0]) - set(before[0]) == {"lab_arena_capture_summary_v1(text,text)"}
    assert {name: doc for name, doc in first[0].items() if name in before[0]} == before[0]
    assert (first[1], first[3]) == (before[1], before[3])
    assert first[2][0] == before[2][0][:-1] + "," + READER + "=U/pg_database_owner}"
    added = set(first[4]) - set(before[4])
    assert added == {(READER, "authenticator", False, True)}
    assert set(before[4]) <= set(first[4])


def test_counts_follow_latest_terminal_and_python_usage_rules(database, migrated):
    calls = usage_calls = tokens = 0
    started, finished = [], []
    for name, case in CASES.items():
        run_id, terminals, provider, operation, reserved = _spec(case)
        scoped = run_id in ("run593-a", "run593-b", "run593-c") and provider == "openrouter"
        if not (scoped and reserved and terminals and _has_successful_terminal_response(terminals[-1][1])):
            continue
        calls += 1
        started.append(migrated[3][name][0])
        finished.append(migrated[3][name][-1])
        reported = _python_tokens(terminals[-1][1], operation)
        if reported is not None:
            usage_calls, tokens = usage_calls + 1, tokens + reported
    assert (calls, usage_calls, tokens) == (12, 5, 157)
    summary = _summary(database)
    assert {key: summary[key] for key in summary if not key.endswith("_at")} == {
        "round_id": ROUND, "submission_id": SUBMISSION, "execute_runs": 3, "icp_positions": [1, 3],
        "model_calls": calls, "usage_calls": usage_calls, "reasoning_tokens": tokens}
    # A call starts at its reservation and ends at its latest terminal.
    assert summary["first_call_at"] == min(started).isoformat()
    assert summary["last_call_at"] == max(finished).isoformat()


def test_unmatched_scope_returns_one_empty_row(database, migrated):
    empty = {"execute_runs": 0, "icp_positions": [], "model_calls": 0, "usage_calls": 0,
             "reasoning_tokens": None, "first_call_at": None, "last_call_at": None}
    assert _summary(database, "round-missing", SUBMISSION) == {
        "round_id": "round-missing", "submission_id": SUBMISSION, **empty}
    assert _summary(database, ROUND, "sub-missing") == {"round_id": ROUND, "submission_id": "sub-missing", **empty}
    summary = _summary(database, ROUND, OTHER)
    assert (summary["execute_runs"], summary["model_calls"], summary["reasoning_tokens"]) == (1, 1, 2000)


def _summary_for_bodies(database, bodies):
    """Seed a separate submission through the same ledger path as the main cases."""
    psycopg2, dsn = database
    submission = "sub593-" + uuid4().hex
    run_id = "run593-" + uuid4().hex
    with psycopg2.connect(**dsn) as connection, connection.cursor() as cur:
        cur.execute("SET session_replication_role = replica")
        cur.execute(
            "INSERT INTO public.lab_arena_runs (run_id, assignment_id, round_id, submission_id, "
            "miner_hotkey, stage, icp_position, attempt, kind, scoring_generation_floor) "
            "VALUES (%s,%s,%s,%s,'hk593',1,1,1,'execute',0)",
            (run_id, run_id, ROUND, submission))
        for index, body in enumerate(bodies):
            identity = _identity(submission + str(index))
            for kind, terminal in (("reservation", None), ("settlement", _ok(_body(body)))):
                cur.execute(
                    "INSERT INTO public.lab_arena_ledger (entry_kind, miner_hotkey, round_id, submission_id, "
                    "run_id, stage, call_identity, provider, operation_id, amount_microusd, terminal_response) "
                    "VALUES (%s,'hk593',%s,%s,%s,1,%s,'openrouter','openrouter.chat',0,%s)",
                    (kind, ROUND, submission, run_id, identity,
                     None if terminal is None else json.dumps(terminal)))
        cur.execute("SET session_replication_role = origin")
    return _summary(database, submission_id=submission)


@pytest.mark.parametrize("body,reported", [
    (b'{"usage":{"reasoning_tokens":123},"x":"\\udc00"}', 123),
    (b'{"usage":{"reasoning_tokens":123},"x":"\\u0000","y":"\\ud83d\\ude00"}', 123),
    # Literal backslash escapes remain strings, never token counts or token keys.
    (json.dumps({"usage": {"reasoning_tokens": r"\udc00"}, "x": "\0"}).encode(), None),
    (json.dumps({"usage": {r"reasoning_tokens\udc00": 123}, "x": "\0"}).encode(), None),
    (json.dumps({"usage": {"reasoning_tokens": 123}, "x": r"\udc00", "y": "\0"}).encode(), 123),
])
def test_usage_with_surrogate_and_literal_escapes(database, migrated, body, reported):
    assert _python_tokens(_ok(_body(body)), "openrouter.chat") == reported
    summary = _summary_for_bodies(database, [body])
    assert summary["model_calls"] == 1
    assert summary["usage_calls"] == (0 if reported is None else 1)
    assert summary["reasoning_tokens"] == reported


@pytest.mark.parametrize("values,expected", [
    ([0], 0),
    ([0, 1, 123], 124),
    ([1000000000000000000], 1000000000000000000),
    ([9223372036854775807], 9223372036854775807),
    ([9223372036854775808], None),
    ([999999999999999999] * 9 + [223372036854775816], 9223372036854775807),
    ([999999999999999999] * 9 + [223372036854775816, 1], None),
    ([999999999999999999] * 10, None),
])
def test_token_sum_keeps_bigint_contract_without_overflow(database, migrated, values, expected):
    summary = _summary_for_bodies(database, [{"usage": {"reasoning_tokens": value}} for value in values])
    assert summary["model_calls"] == summary["usage_calls"] == len(values)
    assert summary["reasoning_tokens"] == expected


def test_reader_reaches_arena_data_only_through_the_summary(database, migrated):
    psycopg2, dsn = database
    with psycopg2.connect(**dsn) as connection, connection.cursor() as cur:
        cur.execute("SELECT c.oid::regclass::text FROM pg_class c WHERE c.relnamespace = 'public'::regnamespace "
                    "AND c.relkind IN ('r','p','v','m','f') AND has_table_privilege(%s, c.oid, "
                    "'SELECT,INSERT,UPDATE,DELETE,TRUNCATE,REFERENCES,TRIGGER')", (READER,))
        assert cur.fetchall() == []
        cur.execute("SELECT p.oid::regprocedure::text FROM pg_proc p WHERE p.pronamespace = 'public'::regnamespace "
                    "AND has_function_privilege(%s, p.oid, 'EXECUTE') AND NOT EXISTS ("
                    "SELECT 1 FROM aclexplode(coalesce(p.proacl, acldefault('f', p.proowner))) a "
                    "WHERE a.grantee = 0) ORDER BY 1", (READER,))
        assert cur.fetchall() == [("lab_arena_capture_summary_v1(text,text)",)]
        cur.execute("SELECT r.rolcanlogin, r.rolinherit, r.rolsuper, r.rolbypassrls, s.setconfig "
                    "FROM pg_roles r LEFT JOIN pg_db_role_setting s ON s.setrole = r.oid AND s.setdatabase = 0 "
                    "WHERE r.rolname = %s", (READER,))
        assert cur.fetchone() == (False, False, False, False, ["statement_timeout=60s"])
    for statement in ("SELECT count(*) FROM public.lab_arena_ledger",
                      "SELECT count(*) FROM public.lab_arena_runs",
                      "SELECT count(*) FROM public.lab_arena_submissions",
                      "SELECT public.lab_arena_create_round('round-reader593', '{}'::jsonb)",
                      "UPDATE public.lab_arena_runs SET status = 'failed'"):
        with psycopg2.connect(**dsn) as connection, connection.cursor() as cur:
            cur.execute("SET ROLE " + READER)
            with pytest.raises(psycopg2.errors.InsufficientPrivilege):
                cur.execute(statement)


@pytest.mark.parametrize("role", ["anon", "authenticated", "service_role", "lab_arena_service"])
def test_other_roles_cannot_execute_the_summary(database, migrated, role):
    psycopg2, _ = database
    with pytest.raises(psycopg2.errors.InsufficientPrivilege):
        _summary(database, role=role)
