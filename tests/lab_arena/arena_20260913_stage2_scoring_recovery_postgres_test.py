"""Guarded recovery of the September 13 stage-2 scoring cancellation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.lab_arena.arena_20260912_recovery_postgres_test import _row_hash
from tests.lab_arena.arena_20260913_scoring_recovery_postgres_test import (
    PARTICIPANT_IDS,
    ROUND_ID,
    _scope,
    _seed,
)
from tests.lab_arena.lab_arena_pg_harness import (
    POSTGREST_MIGRATIONS,
    database_with_lab_arena_migration,
)


ROOT = Path(__file__).resolve().parents[2]
MIGRATION = ROOT / "scripts" / "240-recover-arena-2026-09-13-stage2-scoring.sql"
OLD_DIGEST = "sha256:188fe7f79e0233c4213bd6262f0d47e07f83d14b3374ee502f7268433111ba4e"
NEW_DIGEST = "sha256:" + "9" * 64
NEW_REFERENCE = "registry.example/scorer@" + NEW_DIGEST
BASELINE = "baseline-2026-09-13"
CAF = "sub-caf0e1ef30c9712e6385afe24a75375e"
COST_TARGET = "sub-5dffdbaa2b96e8dc78160aea8f80a7b9"
JUDGE_TARGET = "sub-6211e8d46819c34df3418ded36f788ef"
PREFIX = f"{ROUND_ID}:score-recovery240:"
RECONCILIATIONS = (
    (315826, CAF, f"{ROUND_ID}:{CAF}:1:0:1", 1,
     "sha256:223335eda1ccf9fcf8b0d4d3a00f85d957cf896611b8c845fea331be83c8c14d"),
    (331371, COST_TARGET, f"{ROUND_ID}:{COST_TARGET}:2:12:1", 2,
     "sha256:6bc891aeb6faf7f07d1fe97e3f9cc12e3cac686211097473456b77613a524f2c"),
    (331374, COST_TARGET, f"{ROUND_ID}:{COST_TARGET}:2:12:1", 2,
     "sha256:bbeb448fa9f273c0b58b6a82ad8ae7d7154e20a8789d39b596f0f5dcce7bbb76"),
    (331383, COST_TARGET, f"{ROUND_ID}:{COST_TARGET}:2:12:1", 2,
     "sha256:070df2985bf72f35fdc6d01c93786769f6fb8f7d976294ddf1fc14fe756d3496"),
    (331388, COST_TARGET, f"{ROUND_ID}:{COST_TARGET}:2:12:1", 2,
     "sha256:231f8c7f751019140a87611f800b4cc1fc6256b440aad576ba3d31899310a7da"),
    (331398, COST_TARGET, f"{ROUND_ID}:{COST_TARGET}:2:12:1", 2,
     "sha256:83b62e720fd19d4bafa5bab2d2dd09345b591f28e576f4d87c7bab19198894f1"),
    (331411, COST_TARGET, f"{ROUND_ID}:{COST_TARGET}:2:12:1", 2,
     "sha256:bf2cd93f1511916c9c2c5967605d058d830c4797f0cdc055d60d29f339749fb1"),
    (336996, COST_TARGET, f"{ROUND_ID}:{COST_TARGET}:2:10:score:1", 2,
     "sha256:94484384d8b1dfcb5de6ec34755d9c55abb4ef2cf8fc826e624de541d5d29a93"),
)


@pytest.fixture(scope="module")
def database():
    yield from database_with_lab_arena_migration(POSTGREST_MIGRATIONS)


def _migration() -> str:
    return (
        MIGRATION.read_text(encoding="utf-8")
        .replace("__FIXED_SCORER_IMAGE_DIGEST__", NEW_DIGEST)
        .replace("__FIXED_SCORER_IMAGE_REFERENCE__", NEW_REFERENCE)
    )


def _prepare(database, *, reconciliation_count: int = 8):
    connection, store, transport = _seed(database)
    miners = {
        row["submission_id"]: row["miner_hotkey"]
        for row in store.get_round(ROUND_ID)["participants"]
    }
    finalists = [sid for sid in PARTICIPANT_IDS if sid != BASELINE][:10]
    work_items = []
    zero_rows = []
    with connection.cursor() as cursor:
        cursor.execute("ALTER TABLE public.lab_arena_runs DISABLE TRIGGER lab_arena_runs_terminal")
        cursor.execute("ALTER TABLE public.lab_arena_runs DISABLE TRIGGER lab_arena_integrity_run_guard")
        for sid in PARTICIPANT_IDS:
            for position in range(10, 20):
                assignment = f"{ROUND_ID}:{sid}:2:{position}"
                execute_id = assignment + ":1"
                accepted_execute = sid != CAF
                cursor.execute(
                    "INSERT INTO public.lab_arena_runs (run_id,assignment_id,round_id,"
                    "submission_id,miner_hotkey,stage,icp_position,attempt,status,"
                    "stage_generation,kind,output_ref,terminal_cause) VALUES "
                    "(%s,%s,%s,%s,%s,2,%s,1,%s,11,'execute',%s,%s)",
                    (
                        execute_id, assignment, ROUND_ID, sid, miners[sid], position,
                        "accepted" if accepted_execute else "failed",
                        f"arena/{ROUND_ID}/outputs/{execute_id}.json" if accepted_execute else None,
                        "accepted" if accepted_execute else "credential_error",
                    ),
                )
                if not accepted_execute:
                    zero_rows.append({"submission_id": sid, "icp_position": position,
                                      "cause": "credential_error"})
                    continue
                work_items.append({"submission_id": sid, "icp_position": position,
                                   "scored_run_id": execute_id,
                                   "output_ref": f"arena/{ROUND_ID}/outputs/{execute_id}.json"})
                score_assignment = assignment + ":score"
                scope = _scope(position, OLD_DIGEST,
                               "registry.example/scorer@" + OLD_DIGEST,
                               marker=f"stage2-{sid}-{position}")
                accepted = position in (10, 11)
                cause = "accepted" if accepted else "stage_closed"
                if sid == COST_TARGET and position in (18, 19):
                    accepted, cause = False, "credential_error"
                if sid == JUDGE_TARGET and position == 17:
                    accepted, cause = False, "judge_error"
                cursor.execute(
                    "INSERT INTO public.lab_arena_runs (run_id,assignment_id,round_id,"
                    "submission_id,miner_hotkey,stage,icp_position,attempt,status,"
                    "stage_generation,kind,scored_run_id,runner_hotkey,terminal_cause,"
                    "judgment_cache_key,judgment_input_hash,judgment_scope_doc,"
                    "judgment_group_leader,judgment_group_miner_hotkeys) VALUES "
                    "(%s,%s,%s,%s,%s,2,%s,1,%s,12,'score',%s,%s,%s,%s,%s,%s::jsonb,true,%s)",
                    (
                        score_assignment + ":1", score_assignment, ROUND_ID, sid,
                        miners[sid], position, "accepted" if accepted else "failed",
                        execute_id, "judge-one", cause, scope["cache_key"],
                        scope["scoring_input_hash"], json.dumps(scope), [miners[sid]],
                    ),
                )
                if sid == JUDGE_TARGET and position == 17:
                    cursor.execute(
                        "INSERT INTO public.lab_arena_runs (run_id,assignment_id,round_id,"
                        "submission_id,miner_hotkey,stage,icp_position,attempt,status,"
                        "stage_generation,kind,scored_run_id,runner_hotkey,terminal_cause,"
                        "judgment_cache_key,judgment_input_hash,judgment_scope_doc,"
                        "judgment_group_leader,judgment_group_miner_hotkeys) VALUES "
                        "(%s,%s,%s,%s,%s,2,%s,2,'failed',12,'score',%s,'judge-two',"
                        "'judge_error',%s,%s,%s::jsonb,true,%s)",
                        (
                            score_assignment + ":2", score_assignment, ROUND_ID, sid,
                            miners[sid], position, execute_id, scope["cache_key"],
                            scope["scoring_input_hash"], json.dumps(scope), [miners[sid]],
                        ),
                    )
        cursor.execute("ALTER TABLE public.lab_arena_runs ENABLE TRIGGER lab_arena_integrity_run_guard")
        cursor.execute("ALTER TABLE public.lab_arena_runs ENABLE TRIGGER lab_arena_runs_terminal")
        for uncertain_id, sid, run_id, stage, identity in RECONCILIATIONS[
            :reconciliation_count
        ]:
            cursor.execute(
                "INSERT INTO public.lab_arena_ledger (entry_kind,miner_hotkey,"
                "round_id,submission_id,run_id,stage,call_identity,provider,"
                "operation_id,funding_source,amount_microusd,entry_doc,"
                "terminal_response) VALUES ('settlement',%s,%s,%s,%s,%s,%s,"
                "'deepline','deepline.execute','miner_key',0,%s::jsonb,%s::jsonb)",
                (
                    miners[sid], ROUND_ID, sid, run_id, stage, identity,
                    json.dumps({
                        "deepline_402_history_reconciliation": True,
                        "reconciled_uncertainty_entry_id": uncertain_id,
                    }),
                    json.dumps({
                        "status": 502,
                        "call_succeeded": False,
                        "provider_cost": {"basis": "deepline_authenticated_history_no_charge",
                                          "units": "0", "unit_name": "credits",
                                          "operation": "deepline.execute"},
                    }),
                ),
            )
        plan = {"schema_version": "leadpoet.lab_arena.scoring_plan.v1",
                "round_id": ROUND_ID, "stage": 2,
                "work_items": sorted(work_items, key=lambda x: x["scored_run_id"]),
                "zero_rows": sorted(zero_rows,
                                    key=lambda x: (x["submission_id"], x["icp_position"]))}
        cursor.execute("ALTER TABLE public.lab_arena_rounds DISABLE TRIGGER lab_arena_rounds_write_once")
        cursor.execute(
            "UPDATE public.lab_arena_rounds SET status='cancelled',cancel_reason='scoring_incomplete',"
            "status_generation=14,stage_generation=12,icp_set_date='2026-09-12',finalists=%s::jsonb,"
            "stage2_scoring_plan_doc=%s::jsonb,configuration_doc=jsonb_set(jsonb_set("
            "jsonb_set(configuration_doc,'{scorer_image_digest}',to_jsonb(%s::text),false),"
            "'{scorer_image_reference}',to_jsonb(%s::text),false),"
            "'{sourcing_cost_eligibility_policy}','\"successful_calls_v1\"'::jsonb,true) WHERE round_id=%s",
            (json.dumps(finalists), json.dumps(plan), OLD_DIGEST,
             "registry.example/scorer@" + OLD_DIGEST, ROUND_ID),
        )
        cursor.execute("ALTER TABLE public.lab_arena_rounds ENABLE TRIGGER lab_arena_rounds_write_once")
    return connection, store, transport


def test_recovery_preserves_history_and_advances_with_fresh_run_ids(database):
    connection, store, transport = _prepare(database)
    try:
        with connection.cursor() as cursor:
            old_runs = _row_hash(cursor, "lab_arena_runs", "round_id=%s", (ROUND_ID,))
            ledger = _row_hash(cursor, "lab_arena_ledger", "round_id=%s", (ROUND_ID,))
            cursor.execute("SELECT count(*) FILTER (WHERE status='frozen'),"
                           "(SELECT count(*) FROM public.lab_arena_submission_credentials "
                           "WHERE submission_id=ANY(%s)) FROM public.lab_arena_submissions "
                           "WHERE round_id=%s", (list(PARTICIPANT_IDS), ROUND_ID))
            assert cursor.fetchone() == (12, 24)
            cursor.execute(_migration())
            assert _row_hash(cursor, "lab_arena_runs",
                             "round_id=%s AND run_id NOT LIKE %s",
                             (ROUND_ID, PREFIX + "%")) == old_runs
            assert _row_hash(cursor, "lab_arena_ledger", "round_id=%s", (ROUND_ID,)) == ledger
            cursor.execute(
                "SELECT count(*),count(*) FILTER (WHERE attempt=2),"
                "count(*) FILTER (WHERE attempt=1 AND assignment_id LIKE '%%:recovery240') "
                "FROM public.lab_arena_runs WHERE run_id LIKE %s", (PREFIX + "%",),
            )
            assert cursor.fetchone() == (88, 87, 1)
            before_replay = _row_hash(cursor, "lab_arena_runs", "round_id=%s", (ROUND_ID,))
            cursor.execute(_migration())
            assert _row_hash(cursor, "lab_arena_runs", "round_id=%s", (ROUND_ID,)) == before_replay
            cursor.execute("ALTER TABLE public.lab_arena_runs DISABLE TRIGGER lab_arena_runs_terminal")
            cursor.execute(
                "UPDATE public.lab_arena_runs SET status='accepted',terminal_cause='accepted',"
                "runner_hotkey='judge-recovery',output_ref='arena/recovered/'||run_id||'.json' "
                "WHERE run_id LIKE %s", (PREFIX + "%",),
            )
            cursor.execute("ALTER TABLE public.lab_arena_runs ENABLE TRIGGER lab_arena_runs_terminal")
        closed = store.close_scoring(ROUND_ID, 2)
        assert closed["round_status"] == "stage2_judged"
        assert closed["incomplete_assignments"] == 1
    finally:
        connection.close()
        transport.close()


def test_recovery_rejects_unreconciled_target_cost(database):
    connection, _store, transport = _prepare(database)
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                "INSERT INTO public.lab_arena_ledger (entry_kind,miner_hotkey,round_id,"
                "submission_id,run_id,stage,call_identity,provider,operation_id,"
                "funding_source,amount_microusd,entry_doc) SELECT 'reservation',miner_hotkey,"
                "round_id,submission_id,run_id,stage,'sha256:'||repeat('a',64),"
                "'openrouter','openrouter.chat','miner_key',5000,'{}'::jsonb "
                "FROM public.lab_arena_runs WHERE round_id=%s AND submission_id=%s "
                "AND stage=2 AND kind='score' LIMIT 1", (ROUND_ID, COST_TARGET),
            )
        with pytest.raises(connection.Error, match="stage2 recovery state differs"):
            with connection.cursor() as cursor:
                cursor.execute(_migration())
        connection.rollback()
    finally:
        connection.close()
        transport.close()


def test_recovery_rejects_one_missing_exact_reconciliation(database):
    connection, _store, transport = _prepare(database, reconciliation_count=7)
    try:
        before = None
        with connection.cursor() as cursor:
            before = _row_hash(cursor, "lab_arena_runs", "round_id=%s", (ROUND_ID,))
        with pytest.raises(connection.Error, match="exact cost reconciliation incomplete"):
            with connection.cursor() as cursor:
                cursor.execute(_migration())
        with connection.cursor() as cursor:
            cursor.execute("ROLLBACK")
            assert _row_hash(cursor, "lab_arena_runs", "round_id=%s", (ROUND_ID,)) == before
            cursor.execute("SELECT count(*) FROM public.lab_arena_runs WHERE run_id LIKE %s",
                           (PREFIX + "%",))
            assert cursor.fetchone()[0] == 0
    finally:
        connection.close()
        transport.close()
