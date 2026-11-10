"""Additive v3 count of calls with reasoning on the actual PostgreSQL ledger/reader path."""
from pathlib import Path

import pytest

from tests.lab_arena.capture_summary593_postgres_test import READER, ROUND, _catalog, database, migrated
from tests.lab_arena.capture_summary598_postgres_test import SQL as SQL598, SQL595, _seed_bodies

SQL = Path(__file__).resolve().parents[2] / "scripts/684-lab-arena-capture-summary-v3.sql"
V1 = "public.lab_arena_capture_summary_v1(text,text)"
V2 = "public.lab_arena_capture_summary_v2(text,text)"
V3 = "public.lab_arena_capture_summary_v3(text,text)"


@pytest.fixture(scope="module")
def upgraded(database, migrated):
    psycopg2, dsn = database
    with psycopg2.connect(**dsn) as connection, connection.cursor() as cur:
        cur.execute(SQL595.read_text())
        cur.execute(SQL598.read_text())
        cur.execute((SQL.parent / "620-lab-arena-retained-success-capture.sql").read_text())
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
    result = rows[0][0]
    if version == "v3" and role == READER:
        assert {key: value for key, value in result.items()
                if key not in {"reasoning_calls", "reasoning_capture"}} == _read(database, "v2", submission)
    return result


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
    _assert_capture(summary, {})


def test_every_v2_column_is_unchanged(database, upgraded):
    submission = _seed_bodies(database, [
        {"model": "a/b", "usage": {"prompt_tokens": 10, "completion_tokens": 8, "reasoning_tokens": 3}},
        {"model": "c/d", "usage": {"prompt_tokens": 4, "completion_tokens": 2}},
    ])
    v2, v3 = _read(database, "v2", submission), _read(database, "v3", submission)
    assert v3.pop("reasoning_calls") == 1
    assert sum(v3.pop("reasoning_capture").values()) == v3["model_calls"]
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


CAPTURE_CLASSES = {"complete_readable", "partial_readable", "tokens_only", "no_readable", "unparseable"}


def _assert_capture(summary, expected):
    counts = summary["reasoning_capture"]
    assert set(counts) == CAPTURE_CLASSES
    assert all(type(value) is int and value >= 0 for value in counts.values())
    assert counts == {name: expected.get(name, 0) for name in CAPTURE_CLASSES}
    assert sum(counts.values()) == summary["model_calls"]


def _chat(message, finish="stop", usage=None):
    return {"choices": [{"message": message, "finish_reason": finish}],
            **({"usage": usage} if usage is not None else {})}


@pytest.mark.parametrize("body,expected", [
    (_chat({"reasoning": "readable"}), "complete_readable"),
    (_chat({"reasoning_content": "readable"}, "tool_calls"), "complete_readable"),
    (_chat({"reasoning": "readable"}, "length"), "partial_readable"),
    (_chat({"reasoning": "readable"}, None), "partial_readable"),
    (_chat({"reasoning": "readable"}, "error"), "partial_readable"),
    (_chat({"reasoning": "readable"}, "stop", {"reasoning_tokens": 0}), "complete_readable"),
    (_chat({"reasoning_details": [{"type": "reasoning.text", "text": "thinking"}]}), "complete_readable"),
    (_chat({"reasoning_details": [{"type": "thinking", "thinking": "thinking"}]}), "complete_readable"),
    (_chat({"reasoning_details": [{"type": "reasoning.text", "text": [{"text": "thinking"}]}]}), "complete_readable"),
    (_chat({"content": "answer"}, usage={"reasoning_tokens": 7}), "tokens_only"),
    (_chat({"reasoning": " \t\n\r"}, usage={"reasoning_tokens": 7}), "tokens_only"),
    (_chat({"reasoning": "\u00a0\u2003\u3000"}), "no_readable"),
    (_chat({"reasoning": ""}, usage={"reasoning_tokens": 0}), "no_readable"),
    (_chat({"content": "answer"}), "no_readable"),
    (_chat({"reasoning_details": [{"type": "reasoning.summary", "text": "summary", "summary": "summary"}]},
           usage={"reasoning_tokens": 7}), "tokens_only"),
    (_chat({"reasoning_details": [{"type": "summary_text", "text": "summary"}]}), "no_readable"),
    (_chat({"reasoning_details": [{"type": "reasoning.encrypted", "text": "ciphertext"}]},
           usage={"reasoning_tokens": 7}), "tokens_only"),
    (_chat({"reasoning_details": [{"type": "redacted_thinking", "text": "opaque"}]}), "no_readable"),
    (_chat({"encrypted_content": "ciphertext", "hidden_reasoning": "opaque"}), "no_readable"),
    (_chat({"reasoning": {"text": "opaque shape"}}), "no_readable"),
    ({"choices": {}}, "no_readable"),
    ({"choices": [None, 1, "bad", {"message": ["bad"]}]}, "no_readable"),
    (_chat({"reasoning_details": {"text": "bad shape"}}), "no_readable"),
    (_chat({"reasoning_details": [None, 1, "bad", {"text": [None, 1]}]}), "no_readable"),
    (None, "no_readable"), ([], "no_readable"), ("json string", "no_readable"),
    (b"not json", "unparseable"), (b"\xff", "unparseable"),
])
def test_saved_chat_text_categories(database, upgraded, body, expected):
    summary = _read(database, "v3", _seed_bodies(database, [body]))
    _assert_capture(summary, {expected: 1})


def _seed_response(database, body):
    submission = _seed_bodies(database, [body])
    pg, dsn = database
    with pg.connect(**dsn) as connection, connection.cursor() as cur:
        # Disposable fixture only: preserve the production append-only trigger.
        cur.execute("SET LOCAL session_replication_role=replica")
        cur.execute("UPDATE public.lab_arena_ledger SET operation_id='openrouter.responses' "
                    "WHERE round_id=%s AND submission_id=%s", (ROUND, submission))
    return submission


@pytest.mark.parametrize("body,expected", [
    ({"status": "completed", "output": [{"type": "reasoning", "content": [{"type": "reasoning_text", "text": "thinking"}]}]}, "complete_readable"),
    ({"status": "incomplete", "output": [{"type": "reasoning", "text": "thinking"}]}, "partial_readable"),
    ({"output": [{"type": "reasoning", "thinking": "thinking"}]}, "partial_readable"),
    ({"status": "completed", "incomplete_details": {"reason": "max_output_tokens"},
      "output": [{"type": "reasoning", "text": "thinking"}]}, "partial_readable"),
    ({"status": "completed", "output": [{"type": "reasoning", "summary": [{"type": "summary_text", "text": "summary"}]}],
      "usage": {"output_tokens_details": {"reasoning_tokens": 5}}}, "tokens_only"),
    ({"status": "completed", "output": [{"type": "reasoning", "content": [{"type": "summary_text", "text": "summary"}]}]}, "no_readable"),
    ({"status": "completed", "output": [{"type": "reasoning", "encrypted_content": "opaque"}]}, "no_readable"),
    ({"status": "completed", "output": [{"type": "message", "content": [{"type": "output_text", "text": "answer"}]}]}, "no_readable"),
    ({"status": "completed", "output": {}}, "no_readable"),
    ({"status": "completed", "output": [None, 1, "bad", {"type": "reasoning", "content": [None, 2]}]}, "no_readable"),
])
def test_saved_responses_text_categories(database, upgraded, body, expected):
    _assert_capture(_read(database, "v3", _seed_response(database, body)), {expected: 1})


@pytest.mark.parametrize("terminal", [
    {"call_succeeded": True, "status": 200},
    {"call_succeeded": True, "status": 200, "body_b64": "@@invalid@@"},
    {"call_succeeded": True, "status": 200, "body_b64": {}},
])
def test_missing_or_invalid_body_is_unparseable(database, upgraded, terminal):
    _assert_capture(_read(database, "v3", _seed_bodies(database, terminals=[terminal])), {"unparseable": 1})


def test_aliases_count_one_call_and_no_trace_text_is_returned(database, upgraded):
    text = "private-readable-reasoning"
    body = _chat({"reasoning": text, "reasoning_content": text,
                  "reasoning_details": [{"type": "reasoning.text", "text": text}]})
    summary = _read(database, "v3", _seed_bodies(database, [body]))
    _assert_capture(summary, {"complete_readable": 1})
    assert text not in str(summary)


@pytest.mark.parametrize("settled", [False, True])
def test_retained_success_matches_post620_v2_and_counts_identity_once(database, upgraded, settled):
    from tests.lab_arena.retained_success_capture620_postgres_test import seed
    submission, *_ = seed(database, settled=settled)
    summary = _read(database, "v3", submission)
    _assert_capture(summary, {"tokens_only": 1})
    assert summary.pop("reasoning_calls") == 1
    summary.pop("reasoning_capture")
    assert summary == _read(database, "v2", submission)


@pytest.mark.parametrize("fault", ["fingerprint", "status", "private_only", "stale", "membership"])
def test_retained_success_rejects_invalid_binding(database, upgraded, fault):
    from tests.lab_arena.retained_success_capture620_postgres_test import seed
    submission, *_ = seed(database, settled=True, fault=fault)
    summary = _read(database, "v3", submission)
    _assert_capture(summary, {})
    assert summary["model_calls"] == summary["reasoning_calls"] == 0


def test_execute_run_completion_not_required_and_score_runs_excluded(database, upgraded):
    submission = _seed_bodies(database, [_chat({"reasoning": "readable"})])
    pg, dsn = database
    with pg.connect(**dsn) as connection, connection.cursor() as cur:
        cur.execute("SELECT status FROM public.lab_arena_runs WHERE round_id=%s AND submission_id=%s", (ROUND, submission))
        assert cur.fetchone()[0] != "succeeded"
    _assert_capture(_read(database, "v3", submission), {"complete_readable": 1})
    with pg.connect(**dsn) as connection, connection.cursor() as cur:
        cur.execute("UPDATE public.lab_arena_runs SET kind='score' WHERE round_id=%s AND submission_id=%s", (ROUND, submission))
    summary = _read(database, "v3", submission)
    _assert_capture(summary, {})
    assert summary["execute_runs"] == 0


@pytest.mark.parametrize("operation,body", [
    ("openrouter.chat", _chat({"reasoning": "readable"})),
    ("openrouter.chat", _chat({"reasoning_content": "readable"})),
    ("openrouter.chat", _chat({"reasoning_details": [{"type": "reasoning.text", "text": "readable"}]})),
    ("openrouter.chat", _chat({"reasoning_details": [{"type": "reasoning.summary", "summary": "summary"}]})),
    ("openrouter.chat", _chat({"reasoning_details": [{"type": "reasoning.encrypted", "data": "opaque"}]})),
    ("openrouter.responses", {"status": "completed", "output": [{"type": "reasoning", "content": [{"type": "reasoning_text", "text": "readable"}]}]}),
    ("openrouter.responses", {"status": "completed", "output": [{"type": "reasoning", "content": [{"type": "summary_text", "text": "summary"}]}]}),
])
def test_readable_presence_matches_existing_python_segment_parser(database, upgraded, operation, body):
    from lab_arena.training_thinking import _segments
    segments, _, _ = _segments(body, operation)
    readable = any("summary" not in segment["field"] for segment in segments)
    submission = (_seed_response(database, body) if operation == "openrouter.responses"
                  else _seed_bodies(database, [body]))
    summary = _read(database, "v3", submission)
    assert summary["reasoning_capture"]["complete_readable"] == int(readable)


def test_equal_text_on_distinct_call_identities_is_not_deduplicated(database, upgraded):
    body = _chat({"reasoning": "same text on two calls"})
    summary = _read(database, "v3", _seed_bodies(database, [body, body]))
    _assert_capture(summary, {"complete_readable": 2})


def test_inflight_and_failed_calls_are_not_successful_response_capture(database, upgraded):
    from tests.lab_arena.capture_summary593_postgres_test import _body, _ok
    body = _body(_chat({"reasoning": "readable"}, usage={"reasoning_tokens": 9}))
    submission = _seed_bodies(database, terminals=[_ok(body, status=502)])
    pg, dsn = database
    with pg.connect(**dsn) as connection, connection.cursor() as cur:
        cur.execute("INSERT INTO public.lab_arena_ledger (entry_kind,miner_hotkey,round_id,submission_id,"
                    "run_id,stage,call_identity,provider,operation_id,amount_microusd) "
                    "SELECT 'reservation',miner_hotkey,round_id,submission_id,run_id,stage,"
                    "'sha256:' || repeat('a',64),'openrouter','openrouter.chat',0 "
                    "FROM public.lab_arena_runs WHERE round_id=%s AND submission_id=%s", (ROUND, submission))
    summary = _read(database, "v3", submission)
    _assert_capture(summary, {})
    assert summary["execute_runs"] == 1 and summary["icp_positions"] == [2]
