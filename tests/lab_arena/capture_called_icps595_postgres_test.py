"""595 retains reader permissions/totals while counting only called execute ICPs."""
import json
from pathlib import Path
from uuid import uuid4

import pytest

from tests.lab_arena.capture_summary593_postgres_test import (
    SQL as SQL593, READER, _catalog, _summary, database, migrated,
)

SQL = Path(__file__).resolve().parents[2] / "scripts/595-lab-arena-capture-called-icps.sql"


@pytest.fixture(scope="module")
def upgraded(database, migrated):
    psycopg2, dsn = database
    with psycopg2.connect(**dsn) as connection, connection.cursor() as cur:
        before = _catalog(cur)
        baseline = _summary(database)
        cur.execute(SQL.read_text())
        first = _catalog(cur)
        cur.execute(SQL.read_text())
        assert _catalog(cur) == first
    assert _summary(database) == baseline
    # No function metadata changes except the exact existing function body.
    name = "lab_arena_capture_summary_v1(text,text)"
    old = before[0][name].pop("prosrc")
    new = first[0][name].pop("prosrc")
    assert old != new
    assert before == first


def _seed(cur, *, provider="openrouter", kind="reservation", call=True, run_kind="execute",
          round_match=True, submission_match=True, position=0, attempt=1):
    token = uuid4().hex
    run = "run595-" + token
    cur.execute("INSERT INTO public.lab_arena_runs (run_id,assignment_id,round_id,submission_id,"
                "miner_hotkey,stage,icp_position,attempt,kind,scoring_generation_floor) "
                "VALUES (%s,%s,'round595','sub595','hk595',1,%s,%s,%s,0)",
                (run, run, position, attempt, run_kind))
    if kind:
        cur.execute("INSERT INTO public.lab_arena_ledger (entry_kind,miner_hotkey,round_id,submission_id,"
                    "run_id,stage,call_identity,provider,operation_id,amount_microusd) "
                    "VALUES (%s,'hk595',%s,%s,%s,1,%s,%s,'test.operation',0)",
                    (kind, "round595" if round_match else "other-round595",
                     "sub595" if submission_match else "other-sub595", run,
                     "sha256:" + token * 2 if call else None, provider))


@pytest.mark.parametrize("provider,kind", [
    ("openrouter", "reservation"), ("deepline", "reservation"),
    ("deepline", "refusal"), ("openrouter", "dispatch"),
    ("openrouter", "uncertain"), ("deepline", "settlement"),
])
def test_trace_positions_match_actual_cost_call_activity(database, upgraded, provider, kind):
    psycopg2, dsn = database
    with psycopg2.connect(**dsn) as connection, connection.cursor() as cur:
        cur.execute("SET session_replication_role=replica")
        _seed(cur, provider=provider, kind=kind, position=1)
        _seed(cur, provider=provider, kind=kind, position=1, attempt=2)
        _seed(cur, kind=None, position=2)  # assigned but never called
        _seed(cur, call=False, position=3)  # non-call ledger event
        _seed(cur, run_kind="score", position=4)
        _seed(cur, round_match=False, position=5)
        _seed(cur, submission_match=False, position=6)
        cur.execute("SET session_replication_role=origin")
        cur.execute("SELECT public.lab_arena__successful_icp_cost_state('round595','sub595',1)")
        assert cur.fetchone()[0]["call_count"] == 2
        cur.execute("SET ROLE " + READER)
        cur.execute("SELECT to_jsonb(s) FROM public.lab_arena_capture_summary_v1('round595','sub595') s")
        summary = cur.fetchone()[0]
        assert summary["execute_runs"] == 6
        assert summary["icp_positions"] == [1]
        assert (summary["model_calls"], summary["usage_calls"], summary["reasoning_tokens"]) == (0, 0, None)
        connection.rollback()


def test_uncalled_positions_fail_before_upgrade_and_totals_stay_unchanged(database, upgraded):
    psycopg2, dsn = database
    with psycopg2.connect(**dsn) as connection, connection.cursor() as cur:
        cur.execute("SET session_replication_role=replica")
        _seed(cur, kind=None, position=7)
        cur.execute("SET session_replication_role=origin")
        cur.execute(SQL593.read_text())
        cur.execute("SELECT to_jsonb(s) FROM public.lab_arena_capture_summary_v1('round595','sub595') s")
        old = cur.fetchone()[0]
        assert old["icp_positions"] == [7]
        cur.execute(SQL.read_text())
        cur.execute("SELECT to_jsonb(s) FROM public.lab_arena_capture_summary_v1('round595','sub595') s")
        new = cur.fetchone()[0]
        assert new["icp_positions"] == []
        assert {k: v for k, v in new.items() if k != "icp_positions"} == {
            k: v for k, v in old.items() if k != "icp_positions"}


@pytest.mark.parametrize("drift", ["body", "acl"])
def test_upgrade_rejects_unexpected_preimage_or_permissions(database, upgraded, drift):
    psycopg2, dsn = database
    with psycopg2.connect(**dsn) as connection, connection.cursor() as cur:
        cur.execute("SELECT pg_get_functiondef('public.lab_arena_capture_summary_v1(text,text)'::regprocedure)")
        original = cur.fetchone()[0]
        if drift == "body":
            cur.execute(original.replace("BEGIN", "BEGIN\n  -- unexpected body", 1))
        else:
            cur.execute("GRANT EXECUTE ON FUNCTION public.lab_arena_capture_summary_v1(text,text) TO authenticated")
        with pytest.raises(psycopg2.Error, match="(preimage|security) differs"):
            cur.execute(SQL.read_text())
        connection.rollback()
