"""Additive v2 capture aggregates on the actual PostgreSQL ledger/reader path."""
import json
from pathlib import Path
from uuid import uuid4

import pytest

from tests.lab_arena.capture_summary593_postgres_test import (
    READER, ROUND, SUBMISSION, _body, _catalog, _identity, _ok,
    _summary as _v1_summary, database, migrated,
)
from tests.lab_arena.capture_called_icps595_postgres_test import _seed as _seed595

ROOT = Path(__file__).resolve().parents[2]
SQL595 = ROOT / "scripts/595-lab-arena-capture-called-icps.sql"
SQL = ROOT / "scripts/598-lab-arena-capture-summary-v2.sql"
V1 = "lab_arena_capture_summary_v1(text,text)"
V2 = "lab_arena_capture_summary_v2(text,text)"
MAX_BIGINT = 9223372036854775807
NEW_COLUMNS = {"input_tokens", "output_tokens", "cached_input_tokens", "token_calls", "models"}


def _security_catalog(cur):
    cur.execute("SELECT rolname, to_jsonb(r)-'oid' FROM pg_roles r ORDER BY rolname")
    roles = cur.fetchall()
    cur.execute("SELECT setdatabase, setrole::regrole::text, setconfig FROM pg_db_role_setting ORDER BY 1,2")
    settings = cur.fetchall()
    cur.execute("SELECT attrelid::regclass::text, attname, attacl::text FROM pg_attribute "
                "WHERE attacl IS NOT NULL ORDER BY 1,2")
    return roles, settings, cur.fetchall()


@pytest.fixture(scope="module")
def upgraded(database, migrated):
    psycopg2, dsn = database
    with psycopg2.connect(**dsn) as connection, connection.cursor() as cur:
        cur.execute(SQL595.read_text())
        before = _catalog(cur), _security_catalog(cur)
        cur.execute(SQL.read_text())
        first = _catalog(cur), _security_catalog(cur)
        cur.execute(SQL.read_text())
        second = _catalog(cur), _security_catalog(cur)
    return before, first, second


def _summary(database, submission_id=SUBMISSION, round_id=ROUND, role=READER):
    psycopg2, dsn = database
    with psycopg2.connect(**dsn) as connection, connection.cursor() as cur:
        cur.execute("SET ROLE " + role)
        cur.execute("SELECT to_jsonb(s) FROM public.lab_arena_capture_summary_v2(%s,%s) s",
                    (round_id, submission_id))
        rows = cur.fetchall()
    assert len(rows) == 1
    summary = rows[0][0]
    if role == READER:
        assert {key: value for key, value in summary.items() if key not in NEW_COLUMNS} == _v1_summary(
            database, round_id=round_id, submission_id=submission_id)
    return summary


def _seed_bodies(database, bodies=None, *, terminals=None):
    """Keep raw bodies private; return only submission identity for the reader RPC."""
    psycopg2, dsn = database
    submission, run = "sub598-" + uuid4().hex, "run598-" + uuid4().hex
    if terminals is None:
        terminals = [_ok(_body(body)) for body in bodies]
    with psycopg2.connect(**dsn) as connection, connection.cursor() as cur:
        cur.execute("SET session_replication_role=replica")
        cur.execute("INSERT INTO public.lab_arena_runs (run_id,assignment_id,round_id,submission_id,"
                    "miner_hotkey,stage,icp_position,attempt,kind,scoring_generation_floor) "
                    "VALUES (%s,%s,%s,%s,'hk598',1,2,1,'execute',0)",
                    (run, run, ROUND, submission))
        for index, terminal in enumerate(terminals):
            for kind, value in (("reservation", None), ("settlement", terminal)):
                cur.execute("INSERT INTO public.lab_arena_ledger (entry_kind,miner_hotkey,round_id,submission_id,"
                            "run_id,stage,call_identity,provider,operation_id,amount_microusd,terminal_response) "
                            "VALUES (%s,'hk598',%s,%s,%s,1,%s,'openrouter','openrouter.chat',0,%s)",
                            (kind, ROUND, submission, run, _identity(submission + str(index)),
                             None if value is None else json.dumps(value)))
        cur.execute("SET session_replication_role=origin")
    return submission


def _for_bodies(database, bodies):
    return _summary(database, _seed_bodies(database, bodies))


def _model(name, calls=1, input_tokens=None, output_tokens=None, reasoning_tokens=None):
    return {"model": name, "calls": calls, "input_tokens": input_tokens,
            "output_tokens": output_tokens, "reasoning_tokens": reasoning_tokens}


def test_additive_idempotent_catalog_and_unchanged_security(upgraded):
    (before, security_before), (first, security_first), second = upgraded
    assert (first, security_first) == second
    assert set(first[0]) - set(before[0]) == {V2}
    assert {name: doc for name, doc in first[0].items() if name != V2} == before[0]
    assert first[1:] == before[1:]
    assert security_first == security_before
    function = first[0][V2]
    assert function["prosecdef"] is True and function["provolatile"] == "s"
    assert function["proconfig"] == ["search_path=pg_catalog, public"]
    assert function["proargnames"] == ["p_round_id", "p_submission_id", "round_id", "submission_id",
        "execute_runs", "icp_positions", "model_calls", "usage_calls", "reasoning_tokens",
        "first_call_at", "last_call_at", "input_tokens", "output_tokens", "cached_input_tokens",
        "token_calls", "models"]


def test_v1_fixture_parity_includes_latest_heads_scope_timestamps_and_invalid_bodies(database, upgraded):
    summary = _summary(database)
    assert summary["model_calls"] == 12
    assert summary["reasoning_tokens"] == 157
    assert summary["input_tokens"] == 5 and summary["output_tokens"] == 9
    assert summary["cached_input_tokens"] is None and summary["token_calls"] == 1
    assert summary["models"] == [_model("unknown", 12, 5, 9, 157)]
    empty = _summary(database, round_id="missing-round598")
    assert empty["model_calls"] == empty["token_calls"] == 0
    assert all(empty[key] is None for key in ("input_tokens", "output_tokens", "cached_input_tokens"))
    assert empty["models"] == []


def test_called_icps_retain595_semantics_across_providers_and_no_call_runs(database, upgraded):
    psycopg2, dsn = database
    with psycopg2.connect(**dsn) as connection, connection.cursor() as cur:
        cur.execute("SET session_replication_role=replica")
        _seed595(cur, provider="deepline", kind="refusal", position=1)
        _seed595(cur, kind=None, position=2)
        _seed595(cur, call=False, position=3)
        _seed595(cur, run_kind="score", position=4)
        _seed595(cur, round_match=False, position=5)
        _seed595(cur, submission_match=False, position=6)
        cur.execute("SET session_replication_role=origin")
        for version in (1, 2):
            cur.execute(f"SELECT to_jsonb(s) FROM public.lab_arena_capture_summary_v{version}('round595','sub595') s")
            result = cur.fetchone()[0]
            assert result["icp_positions"] == [1] and result["execute_runs"] == 5
            assert result["model_calls"] == 0
        connection.rollback()


def test_alias_precedence_fallback_and_reasoning_are_independent(database, upgraded):
    summary = _for_bodies(database, [
        {"model": "vendor/chat-v1", "usage": {"prompt_tokens": 10, "input_tokens": 99,
            "completion_tokens": 20, "output_tokens": 88, "reasoning_tokens": 30,
            "completion_tokens_details": {"reasoning_tokens": 35},
            "prompt_tokens_details": {"cached_tokens": 4}, "input_tokens_details": {"cached_tokens": 77}}},
        {"model": "vendor/responses", "usage": {"prompt_tokens": "bad", "input_tokens": 3,
            "completion_tokens": -1, "output_tokens": 5, "prompt_tokens_details": {"cached_tokens": True},
            "input_tokens_details": {"cached_tokens": 2}, "output_tokens_details": {"reasoning_tokens": 1}}},
    ])
    assert [summary[key] for key in ("input_tokens", "output_tokens", "cached_input_tokens", "token_calls")] == [13, 25, 6, 2]
    assert summary["reasoning_tokens"] == 36  # Output remains 25: no reasoning subtraction.
    assert summary["models"] == [_model("vendor/chat-v1", 1, 10, 20, 35),
                                 _model("vendor/responses", 1, 3, 5, 1)]


@pytest.mark.parametrize("usage,expected", [
    ({}, [None, None, None, 0]),
    ({"prompt_tokens": None, "completion_tokens": None}, [None, None, None, 0]),
    ({"prompt_tokens": 0, "completion_tokens": 0, "prompt_tokens_details": {"cached_tokens": 0}}, [0, 0, 0, 1]),
    ({"input_tokens": 7}, [7, None, None, 0]),
    ({"output_tokens": 8}, [None, 8, None, 0]),
    ({"input_tokens_details": {"cached_tokens": 9}}, [None, None, 9, 0]),
])
def test_missing_null_zero_and_partial_usage(database, upgraded, usage, expected):
    summary = _for_bodies(database, [{"model": "m", "usage": usage}])
    assert [summary[key] for key in ("input_tokens", "output_tokens", "cached_input_tokens", "token_calls")] == expected
    assert summary["models"] == [_model("m", 1, expected[0], expected[1])]


@pytest.mark.parametrize("value", ['"12"', 'true', 'false', '-1', '-0', '1.0', '1e2',
    'null', '[]', '{}', '10000000000000000000', '"1; SELECT pg_sleep(1)"', '"\\\\u0031"'])
def test_new_tokens_reject_non_whole_number_spellings_and_malicious_strings(database, upgraded, value):
    body = ('{"model":"m","usage":{"prompt_tokens":VALUE,"completion_tokens":VALUE,'
            '"prompt_tokens_details":{"cached_tokens":VALUE},"reasoning_tokens":7}}').replace("VALUE", value).encode()
    summary = _for_bodies(database, [body])
    assert all(summary[key] is None for key in ("input_tokens", "output_tokens", "cached_input_tokens"))
    assert summary["token_calls"] == 0 and summary["reasoning_tokens"] == 7
    assert summary["models"] == [_model("m", reasoning_tokens=7)]


@pytest.mark.parametrize("values,expected", [
    ([0], 0), ([MAX_BIGINT], MAX_BIGINT), ([MAX_BIGINT + 1], None),
    ([9999999999999999999], None), ([MAX_BIGINT - 1, 1], MAX_BIGINT),
    ([MAX_BIGINT, 1], None), ([999999999999999999] * 10, None),
])
def test_aggregate_and_model_bigint_bounds_never_raise(database, upgraded, values, expected):
    summary = _for_bodies(database, [{"model": "m", "usage": {"input_tokens": value,
        "output_tokens": value, "input_tokens_details": {"cached_tokens": value}, "reasoning_tokens": value}}
        for value in values])
    assert summary["model_calls"] == summary["token_calls"] == len(values)
    assert [summary[key] for key in ("input_tokens", "output_tokens", "cached_input_tokens", "reasoning_tokens")] == [expected] * 4
    assert summary["models"] == [_model("m", len(values), expected, expected, expected)]


@pytest.mark.parametrize("model,expected", [
    (None, "unknown"), (123, "unknown"), ({"private": "body"}, "unknown"),
    ("", "unknown"), ("a" * 201, "unknown"), ("private prompt with spaces", "unknown"),
    ("x\nsecret", "unknown"), ("x;SELECT-secret", "unknown"), ("x\\secret", "unknown"),
    ("é", "unknown"), ("a" * 200, "a" * 200), ("org/model.v1_test:@+x-y", "org/model.v1_test:@+x-y"),
])
def test_model_identifier_privacy_boundary(database, upgraded, model, expected):
    summary = _for_bodies(database, [{"model": model, "usage": {"prompt_tokens": 1, "completion_tokens": 2},
                                      "choices": [{"message": {"content": "private-output598"}}]}])
    assert summary["models"] == [_model(expected, 1, 1, 2)]
    assert "private-output598" not in json.dumps(summary)


def test_no_model_no_usage_and_invalid_body_still_count_unknown_calls(database, upgraded):
    terminals = [_ok(_body({"usage": None})), _ok(_body({"model": "m"})), _ok("@@bad@@"),
                 _ok(_body(b"not-json")), _ok(_body(b"\xff")), _ok({"not": "a string"}),
                 _ok(_body([{"model": "private"}])), _ok(_body({"usage": [], "model": "m"}))]
    summary = _summary(database, _seed_bodies(database, terminals=terminals))
    assert summary["model_calls"] == 8 and summary["token_calls"] == 0
    assert summary["models"] == [_model("unknown", 6), _model("m", 2)]


@pytest.mark.parametrize("body", [
    b'\xef\xbb\xbf{"model":"m","usage":{"input_tokens":1,"output_tokens":2},"x":"\\u0000"}',
    b'{"model":"m","usage":{"input_tokens":1,"output_tokens":2},"x":"\\ud800"}',
])
def test_bom_and_unrelated_surrogate_preserve_valid_tokens(database, upgraded, body):
    summary = _for_bodies(database, [body])
    assert summary["models"] == [_model("m", 1, 1, 2)]


def test_model_group_order_and_real_other_below_limit(database, upgraded):
    summary = _for_bodies(database, [{"model": name, "usage": {"input_tokens": 1}} for name in
        ["z", "a", "other", "z", "a", "top", "top", "top"]])
    assert summary["models"] == [_model("top", 3, 3), _model("a", 2, 2),
                                  _model("z", 2, 2), _model("other", 1, 1)]


@pytest.mark.parametrize("include_other", [False, True])
def test_model_groups_cap_at50_and_fold_remainder_without_duplicate_other(database, upgraded, include_other):
    names = [f"m{index:02}" for index in range(52)] + (["other"] * 5 if include_other else [])
    summary = _for_bodies(database, [{"model": name, "usage": {"input_tokens": 1,
        "output_tokens": 2, "reasoning_tokens": 3}} for name in names])
    models = summary["models"]
    remainder = 3 + (5 if include_other else 0)
    assert len(models) == 50 and len({item["model"] for item in models}) == 50
    assert models == [_model("other", remainder, remainder, remainder * 2, remainder * 3)] + [
        _model(f"m{index:02}", 1, 1, 2, 3) for index in range(49)]
    assert sum(item["calls"] for item in models) == len(names)


def test_fold_uses_numeric_sums_before_bigint_projection(database, upgraded):
    bodies = [{"model": f"m{index:02}"} for index in range(51)]
    bodies[-1]["usage"] = {"input_tokens": MAX_BIGINT}
    bodies[-2]["usage"] = {"input_tokens": 1}
    summary = _for_bodies(database, bodies)
    assert summary["input_tokens"] is None
    assert summary["models"][0] == _model("other", 2)


def test_reader_acl_allows_only_both_aggregate_functions_and_no_raw_data(database, upgraded):
    psycopg2, dsn = database
    with psycopg2.connect(**dsn) as connection, connection.cursor() as cur:
        cur.execute("SELECT p.oid::regprocedure::text FROM pg_proc p WHERE p.pronamespace='public'::regnamespace "
                    "AND EXISTS (SELECT 1 FROM aclexplode(p.proacl) a WHERE a.grantee=%s::regrole) ORDER BY 1",
                    (READER,))
        assert cur.fetchall() == [(V1,), (V2,)]
        cur.execute("SELECT has_schema_privilege(%s,'public','CREATE')", (READER,))
        assert cur.fetchone() == (False,)
    for statement in ("SELECT * FROM public.lab_arena_ledger", "SELECT * FROM public.lab_arena_runs",
                      "SELECT * FROM public.lab_arena_submissions", "UPDATE public.lab_arena_runs SET status='failed'",
                      "SELECT public.lab_arena_create_round('round598', '{}'::jsonb)"):
        with psycopg2.connect(**dsn) as connection, connection.cursor() as cur:
            cur.execute("SET ROLE " + READER)
            with pytest.raises(psycopg2.errors.InsufficientPrivilege):
                cur.execute(statement)


@pytest.mark.parametrize("role", ["anon", "authenticated", "service_role", "lab_arena_service"])
def test_other_roles_cannot_execute_v2(database, upgraded, role):
    psycopg2, _ = database
    with pytest.raises(psycopg2.errors.InsufficientPrivilege):
        _summary(database, role=role)


@pytest.mark.parametrize("drift", ["v1_body", "v1_acl", "reader_role", "table_grant", "column_grant", "other_function"])
def test_migration_fails_closed_on_security_or_v1_drift(database, upgraded, drift):
    psycopg2, dsn = database
    with psycopg2.connect(**dsn) as connection, connection.cursor() as cur:
        if drift == "v1_body":
            cur.execute("SELECT pg_get_functiondef(%s::regprocedure)", (V1,))
            cur.execute(cur.fetchone()[0].replace("BEGIN", "BEGIN\n -- unexpected", 1))
        elif drift == "v1_acl":
            cur.execute("GRANT EXECUTE ON FUNCTION public.lab_arena_capture_summary_v1(text,text) TO authenticated")
        elif drift == "reader_role":
            cur.execute("ALTER ROLE leadpoet_capture_reader INHERIT")
        elif drift == "table_grant":
            cur.execute("GRANT SELECT ON public.lab_arena_ledger TO leadpoet_capture_reader")
        elif drift == "column_grant":
            cur.execute("GRANT SELECT(terminal_response) ON public.lab_arena_ledger TO leadpoet_capture_reader")
        else:
            cur.execute("GRANT EXECUTE ON FUNCTION public.lab_arena_create_round(text,jsonb) TO leadpoet_capture_reader")
        with pytest.raises(psycopg2.Error, match="(preimage|security|attributes) differ|holds other privileges"):
            cur.execute(SQL.read_text())
        connection.rollback()


def test_model_groups_at_exact50_preserve_real_other(database, upgraded):
    names = [f"m{index:02}" for index in range(49)] + ["other"]
    summary = _for_bodies(database, [{"model": name} for name in names])
    assert summary["models"] == [_model(name) for name in names]


def test_per_model_overflow_does_not_erase_other_models(database, upgraded):
    summary = _for_bodies(database, [
        {"model": "a", "usage": {"input_tokens": MAX_BIGINT}},
        {"model": "a", "usage": {"input_tokens": 1}},
        {"model": "b", "usage": {"input_tokens": 2}},
    ])
    assert summary["input_tokens"] is None
    assert summary["models"] == [_model("a", 2), _model("b", 1, 2)]


def test_invalid_surrogate_model_does_not_erase_valid_usage(database, upgraded):
    summary = _for_bodies(database, [
        b'{"model":"\\ud800","usage":{"input_tokens":1,"output_tokens":2,"reasoning_tokens":3}}',
    ])
    assert summary["models"] == [_model("unknown", 1, 1, 2, 3)]


def test_body_size_boundary_remains_same_as_v1(database, upgraded):
    body = b'{"model":"m","usage":{"input_tokens":1,"output_tokens":2,"reasoning_tokens":3}}'
    limit = 16777216
    summary = _for_bodies(database, [body + b" " * (limit - len(body)),
                                    body + b" " * (limit + 1 - len(body))])
    assert summary["model_calls"] == 2 and summary["token_calls"] == 1
    assert summary["models"] == [_model("m", 1, 1, 2, 3), _model("unknown")]
