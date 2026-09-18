"""Focused PostgreSQL and static guards for the one-time Sep18 rerun302."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.lab_arena.lab_arena_pg_harness import (
    CURRENT_SERVICE_MIGRATIONS,
    database_with_lab_arena_migration,
)

ROOT = Path(__file__).parents[2]
SQL_PATH = ROOT / "scripts/302-arena-2026-09-18-cancelled-baseline-newjudge-rerun.sql"
ROUND = "arena-2026-09-18"
BASELINE = "baseline-2026-09-18"
MINER = "sep18-miner-score-reset-proof"


@pytest.fixture(scope="module")
def database():
    assert CURRENT_SERVICE_MIGRATIONS[-2:] == (
        "294-lab-arena-retire-open-cost-backfill.sql",
        "301-lab-arena-score-payer-boundary.sql",
    )
    yield from database_with_lab_arena_migration(
        CURRENT_SERVICE_MIGRATIONS[:-2]
        + (
            "289-lab-arena-per-icp-cost-policy.sql",
            "292-lab-arena-null-final-score-publication.sql",
        )
        + CURRENT_SERVICE_MIGRATIONS[-2:]
    )


def test_literal_sql_has_fresh_all_participant_and_cost_boundaries():
    sql = SQL_PATH.read_text()
    import re
    assert re.search(r"__[A-Z0-9_]+__", sql) is None
    assert "THEN ':rerun302'" in sql
    assert "NEW.assignment_id IS DISTINCT FROM" in sql
    assert "':score:rerun302'" in sql
    assert "archived_execution_judgments" in sql
    assert "SET per_icp_score = NULL,\n      qualification_doc = NULL" in sql
    assert "- 'per_icp_score' - 'qualification_doc' - 'updated_at'" in sql
    assert "active_execute.kind = 'execute'" in sql
    assert "v_orphan_ledger_hash" in sql
    assert "rerun300" + "cancelledarchive" not in sql
    assert "company_identity" + "_key" not in sql
    assert "linked" + "in.com" not in sql.lower()
    assert "5e136618c577ccded99c10f0cb691242cdb3b397959a98b6a06b5a43e188cb72" in sql
    assert "b83c135347d6ea80fb3e21ebbc5bbf992afc87a5e04ed6404a4831707ef3d03d" in sql
    assert "sha256:6af423ff5de5b4acfa83ef4d0a2e6842d142453ceaac2c17d1f40ecca7cdf67f" in sql
    assert sql.count("SELECT public.lab_arena_prepare_sep18_cancelled_rerun302_v1(") == 1


def test_real_score_function_accepts_changed_judgment_only_after_exact_reset(database):
    psycopg2, dsn = database
    with psycopg2.connect(**dsn) as connection:
        with connection.cursor() as cursor:
            cursor.execute("SET session_replication_role=replica")
            cursor.execute(
                "INSERT INTO public.lab_arena_rounds("
                "round_id,status,configuration_doc,rewards_enabled,participants,"
                "benchmark_ref,evaluation_date,icp_set_date,promotion_required,"
                "champion_funding_frozen) VALUES ("
                "%s,'stage1_closed','{}'::jsonb,FALSE,'[]'::jsonb,%s,"
                "'2026-09-18','2026-09-17',FALSE,TRUE)",
                (ROUND, f"arena/{ROUND}/benchmark.json"),
            )
            cursor.execute(
                "INSERT INTO public.lab_arena_submissions("
                "submission_id,round_id,miner_hotkey,status,is_king,submission_doc,"
                "source_ref,source_size_bytes,consent,frozen_at,code_review_status,"
                "code_review_attempts) VALUES (%s,%s,%s,'frozen',FALSE,'{}'::jsonb,"
                "%s,1,'{}'::jsonb,clock_timestamp(),'pending',0)",
                (MINER, ROUND, "5" + "A" * 47, f"arena/{ROUND}/sources/{MINER}.tar.gz"),
            )
            run_id = f"{ROUND}:{MINER}:1:0:1"
            cursor.execute(
                "INSERT INTO public.lab_arena_runs("
                "run_id,assignment_id,round_id,submission_id,miner_hotkey,stage,"
                "icp_position,attempt,kind,status,stage_generation,terminal_cause,"
                "output_ref,per_icp_score) VALUES (%s,%s,%s,%s,%s,1,0,1,'execute',"
                "'accepted',1,'accepted','arena/proof.json',1.000000)",
                (run_id, run_id[:-2], ROUND, MINER, "5" + "A" * 47),
            )
            cursor.execute("SET session_replication_role=origin")
            cursor.execute("SAVEPOINT old_score")
            with pytest.raises(psycopg2.Error) as error:
                cursor.execute(
                    "SELECT public.lab_arena_record_run_scores(%s,1::smallint,%s::jsonb)",
                    (ROUND, json.dumps([{"run_id": run_id, "per_icp_score": 7}])),
                )
            assert error.value.pgcode == "42501"
            cursor.execute("ROLLBACK TO SAVEPOINT old_score")
            cursor.execute("ALTER TABLE public.lab_arena_runs DISABLE TRIGGER USER")
            cursor.execute(
                "UPDATE public.lab_arena_runs SET per_icp_score=NULL,qualification_doc=NULL "
                "WHERE round_id=%s AND submission_id<>%s AND kind='execute'",
                (ROUND, BASELINE),
            )
            assert cursor.rowcount == 1
            cursor.execute("ALTER TABLE public.lab_arena_runs ENABLE TRIGGER USER")
            cursor.execute(
                "SELECT public.lab_arena_record_run_scores(%s,1::smallint,%s::jsonb)",
                (ROUND, json.dumps([{"run_id": run_id, "per_icp_score": 7}])),
            )
            assert cursor.fetchone()[0] == {
                "status": "ok", "recorded": 1, "existing": 0,
            }
            cursor.execute(
                "SELECT per_icp_score,qualification_doc FROM public.lab_arena_runs "
                "WHERE run_id=%s", (run_id,),
            )
            assert cursor.fetchone() == (7, None)
