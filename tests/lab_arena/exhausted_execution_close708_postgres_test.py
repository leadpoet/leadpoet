"""Keep failed provider attempts incomplete through both execution close paths."""

from pathlib import Path

import pytest

from lab_arena import scoring
from tests.lab_arena.lab_arena_pg_harness import current_service_database
from tests.lab_arena.parallel_twenty_icp_execution_postgres_test import (
    _seed_deadline_close_guard,
)

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
MIGRATION = SCRIPTS / "708-lab-arena-exhausted-execution-close.sql"


@pytest.fixture(scope="module")
def db():
    for pg, dsn in current_service_database():
        with pg.connect(**dsn) as conn, conn.cursor() as cur:
            old = (
                SCRIPTS / "605-arena-2026-11-03-admission-review-recovery.sql"
            ).read_text()
            # Install only the deployed close function and its historical predicate.
            cur.execute(
                old[
                    old.index(
                        "CREATE OR REPLACE FUNCTION public.lab_arena__nov03_review605"
                    ) : old.index("-- Read-only authority")
                ]
            )
            cur.execute(old[old.index("DO $close605$") : old.index("DO $recover605$")])
            cur.execute(MIGRATION.read_text())
            cur.execute(MIGRATION.read_text())
        yield pg, dsn


@pytest.mark.parametrize("parallel", [False, True])
@pytest.mark.parametrize(
    "prior,cause,marked",
    [
        ("provider_error", "provider_error", True),
        ("provider_error", "deadline", True),
        ("model_error", "provider_error", False),
        ("provider_error", "accepted", False),
    ],
)
def test_close_agrees_with_scoring_and_preserves_paid_work(
    db, parallel, prior, cause, marked
):
    pg, dsn = db
    rid = f"arena-2099-11-22-p{int(parallel)}{prior[0]}{cause[0]}"
    with pg.connect(**dsn) as conn, conn.cursor() as cur:
        # This historical seed predates the generation column. Supply its initial value.
        cur.execute(
            "ALTER TABLE lab_arena_runs ALTER COLUMN scoring_generation_floor SET DEFAULT 0"
        )
        _seed_deadline_close_guard(
            conn,
            rid,
            prior_cause=prior,
            retry_status="leased",
            ledger_kinds=("reservation", "dispatch"),
        )
        cur.execute(
            "ALTER TABLE lab_arena_runs ALTER COLUMN scoring_generation_floor DROP DEFAULT"
        )
        cur.execute("SET LOCAL session_replication_role=replica")
        cur.execute(
            """
            INSERT INTO public.lab_arena_submissions(
                submission_id, round_id, miner_hotkey, status, is_king, source_ref,
                source_size_bytes, submission_doc, code_review_status, code_review_doc,
                code_review_claim, code_review_started_at, code_review_attempts)
            SELECT DISTINCT submission_id, round_id, miner_hotkey, 'frozen', false,
                'arena/'||round_id||'/sources/'||submission_id||'.tar.gz', 123,
                '{}'::jsonb, 'passed', jsonb_build_object('decision','pass'),
                'sha256:'||repeat('1',64), now(), 1
            FROM public.lab_arena_runs WHERE round_id=%s
        """,
            (rid,),
        )
        if not parallel:
            cur.execute(
                """UPDATE public.lab_arena_rounds SET configuration_doc = configuration_doc ||
                jsonb_build_object('stage_1_icp_count',2,'parallel_twenty_icp_execution',false) WHERE round_id=%s""",
                (rid,),
            )
            cur.execute(
                "UPDATE public.lab_arena_runs SET stage=1 WHERE round_id=%s", (rid,)
            )
        if cause != "deadline":
            cur.execute(
                "UPDATE public.lab_arena_runs SET status=%s,terminal_cause=%s,result_doc=jsonb_build_object('terminal_status',%s::text),output_ref=%s WHERE round_id=%s AND attempt=2",
                (
                    "accepted" if cause == "accepted" else "failed",
                    cause,
                    cause,
                    "arena/test/output" if cause == "accepted" else None,
                    rid,
                ),
            )
        cur.execute("SET LOCAL session_replication_role=origin")
        cur.execute(
            "SELECT to_jsonb(r) FROM public.lab_arena_runs r WHERE round_id=%s ORDER BY run_id",
            (rid,),
        )
        before = cur.fetchall()
        cur.execute(
            "SELECT to_jsonb(l) FROM lab_arena_ledger l WHERE round_id=%s ORDER BY entry_id",
            (rid,),
        )
        ledger_before = cur.fetchall()
        close = (
            "public.lab_arena_close_parallel_execution_v1(%s)"
            if parallel
            else "public.lab_arena_close_stage(%s,1::smallint)"
        )
        cur.execute("SELECT " + close, (rid,))
        assert cur.fetchone()[0]["incomplete_assignments"] == int(marked)
        cur.execute(
            "SELECT to_jsonb(r) FROM public.lab_arena_runs r WHERE round_id=%s ORDER BY run_id",
            (rid,),
        )
        after = cur.fetchall()
        assert len(after) == len(before)
        for (original,), (result,) in zip(before, after):
            if original["attempt"] != 2 or not marked:
                assert original == result
                continue
            assert result["per_icp_score"] is None
            assert result["terminal_doc"]["infrastructure_incomplete"] is True
            assert result["terminal_cause"] == (
                "stage_closed" if cause == "deadline" else cause
            )
            if cause != "deadline":
                assert {
                    **original,
                    "terminal_doc": result["terminal_doc"],
                    "updated_at": result["updated_at"],
                } == result
        cur.execute(
            "SELECT to_jsonb(l) FROM lab_arena_ledger l WHERE round_id=%s ORDER BY entry_id",
            (rid,),
        )
        ledger_after = cur.fetchall()
        # A live dispatch closed at its deadline retains its existing billing finalization.
        assert ledger_after[: len(ledger_before)] == ledger_before
        if cause != "deadline":
            assert ledger_after == ledger_before
        rows = [r for (r,) in after if r["assignment_id"].endswith("assignment-1")]
        # Use the same explicit scope and positions for both fixture layouts.
        plan = scoring.build_scoring_plan(
            round_id=rid,
            stage=1,
            runs=[{**r, "stage": 1, "icp_position": 0} for r in rows],
        )
        assert bool(plan.get("incomplete_rows")) is marked
        assert bool(plan["zero_rows"]) is (prior == "model_error")
        cur.execute("SELECT " + close, (rid,))
        assert cur.fetchone()[0]["status"] == "existing"


@pytest.mark.parametrize("success", [True, False])
def test_signed_baseline_recovery_and_publication(db, tmp_path, monkeypatch, success):
    from tests.lab_arena.bounded_baseline_recovery470_postgres_test import (
        exercise_signed_baseline_recovery,
    )

    exercise_signed_baseline_recovery(db, tmp_path, monkeypatch, success)
