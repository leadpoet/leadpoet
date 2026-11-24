"""A cancelled, never-dispatched round retains its original paid attempt budget."""

from datetime import datetime, timezone
from pathlib import Path
import json

import pytest

from tests.lab_arena import accepted_execution_correction703_postgres_test as prior
from tests.lab_arena.stage_cutoff_drain420_postgres_test import fixture_configuration
from tests.lab_arena.test_lab_arena_migration_postgres import hotkey

database = prior.database
SQL = Path(__file__).parents[2] / "scripts/736-arena-nov23-unclaimed-dispatch.sql"
ROUND = "arena-2026-11-23"
REQUEST = "nov23-restore-unclaimed-736-v1"


def test_migration_replay_stage_claim_and_exact_archive(database):
    pg, dsn = database
    insert = prior.old.base.insert
    with pg.connect(**dsn) as conn, conn.cursor() as cur:
        cur.execute(SQL.read_text())
        cur.execute(SQL.read_text())
        cur.execute(
            "SELECT has_function_privilege('lab_arena_service', "
            "'public.lab_arena__reopen_unclaimed_nov23_v1()', 'EXECUTE')"
        )
        assert not cur.fetchone()[0]
        cur.execute("TRUNCATE lab_arena_rounds CASCADE")
        config = fixture_configuration(ROUND)
        config.pop("parallel_twenty_icp_execution", None)
        config.update(
            stage_1_icp_count=5,
            stage_2_icp_count=5,
            execution_sequence_policy="baseline_scored_first_v1",
        )
        baseline = dict(
            submission_id="baseline-2026-11-23",
            miner_hotkey=config["baseline_hotkey"],
            is_king=True,
        )
        miners = [
            dict(
                submission_id=f"sub-test-{i}",
                miner_hotkey=hotkey(f"736-{i}"),
                is_king=False,
            )
            for i in range(6)
        ]
        cur.execute("BEGIN; SET LOCAL session_replication_role=replica")
        insert(
            cur,
            "lab_arena_rounds",
            dict(
                round_id=ROUND,
                status="committed",
                stage_generation=9,
                configuration_doc=config,
                participants=[baseline, *miners],
                evaluation_date="2026-11-23",
                icp_set_date="2026-11-22",
                benchmark_ref="arena/benchmark.json",
            ),
        )
        for s in [baseline, *miners]:
            insert(
                cur,
                "lab_arena_submissions",
                dict(
                    **s,
                    round_id=ROUND,
                    status="frozen",
                    source_ref=f"arena/{ROUND}/sources/{s['submission_id']}.tar.gz",
                    source_size_bytes=123,
                    code_review_status="passed",
                    code_review_doc={"decision": "pass"},
                    code_review_started_at=datetime.now(timezone.utc),
                    code_review_claim="sha256:" + "c" * 64,
                    code_review_attempts=1,
                    consent={"public_rerun": True},
                ),
            )
            stage = 1 if s["is_king"] else 2
            for pos in range(10):
                assignment = f"{ROUND}:{s['submission_id']}:{stage}:{pos}"
                insert(
                    cur,
                    "lab_arena_runs",
                    dict(
                        round_id=ROUND,
                        submission_id=s["submission_id"],
                        miner_hotkey=s["miner_hotkey"],
                        assignment_id=assignment,
                        run_id=assignment + ":1",
                        attempt=1,
                        stage=stage,
                        icp_position=pos,
                        kind="execute",
                        stage_generation=1 if stage == 1 else 5,
                        scoring_generation_floor=0,
                        status="failed",
                        terminal_cause="stage_closed",
                        terminal_doc={
                            "previous_status": "pending",
                            "infrastructure_incomplete": True,
                        },
                    ),
                )
        cur.execute("SET LOCAL session_replication_role=origin")
        cur.execute(
            "UPDATE lab_arena_restart_claim_control SET operator_paused=false,guard_commitment='',owner_commitment='',guard_expires_at=NULL,captured_leases='[]',candidate_commit='',restart_scope='',restart_phase=''"
        )

        def open_stage(stage):
            selected = [
                {"submission_id": p["submission_id"], "miner_hotkey": p["miner_hotkey"]}
                for p in ([baseline] if stage == 1 else miners)
            ]
            cur.execute(
                "SELECT lab_arena_open_stage(%s,%s::smallint,%s::jsonb,%s::int[])",
                (ROUND, stage, json.dumps(selected), list(range(10))),
            )
            return cur.fetchone()[0]

        cur.execute("SAVEPOINT absent")
        with pytest.raises(pg.errors.UniqueViolation):
            open_stage(1)
        cur.execute("ROLLBACK TO SAVEPOINT absent")
        cur.execute(
            """INSERT INTO lab_arena_scoring_supersessions(round_id,generation_floor,request_id,request_hash,
            actor_ref,transaction_id,backend_pid,old_round_doc,new_round_doc,old_score_projections,recovery_doc)
            SELECT %s,1,%s,'sha256:'||repeat('a',64),'test',txid_current(),pg_backend_pid(),'{}','{}','{}',
            jsonb_build_object('unclaimed_runs',jsonb_object_agg(run_id,to_jsonb(a))) FROM lab_arena_runs a WHERE round_id=%s""",
            (ROUND, REQUEST, ROUND),
        )
        cur.execute("SET LOCAL timezone='Asia/Kolkata'")
        assert open_stage(1)["assignments"] == 10
        assert open_stage(1)["status"] == "stale"
        assert open_stage(2)["status"] == "stale"
        cur.execute(
            "SELECT count(*) FROM lab_arena_runs WHERE round_id=%s AND status='pending' AND attempt=1",
            (ROUND,),
        )
        assert cur.fetchone()[0] == 10
        # Isolate the next ordinary stage-open transaction after the baseline gate.
        cur.execute(
            "UPDATE lab_arena_rounds SET status='stage1_scored' WHERE round_id=%s",
            (ROUND,),
        )
        assert open_stage(2)["assignments"] == 60
        assert open_stage(2)["status"] == "stale"
        cur.execute(
            "SELECT count(*) FROM lab_arena_runs WHERE round_id=%s AND status='pending' AND attempt=1 AND lease_generation=0",
            (ROUND,),
        )
        assert cur.fetchone()[0] == 70
