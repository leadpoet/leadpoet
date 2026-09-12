"""Production-shaped recovery of the third arena-2026-09-12 cancellation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lab_arena import contracts
from lab_arena.store import hash_lease_token
from tests.lab_arena.arena_20260912_recovery_postgres_test import (
    PARTICIPANT_IDS,
    PRESERVED_SUBMISSION,
    ROUND_ID,
    RUNNER,
    _row_hash,
)
from tests.lab_arena.arena_20260912_scoring_recovery_postgres_test import (
    FINALISTS,
    PRODUCTION_PLAN_MD5,
    PRODUCTION_SCORER,
    _apply_220,
    _insert_run,
    _seed_scoring_failure,
)
from tests.lab_arena.lab_arena_pg_harness import (
    POSTGREST_MIGRATIONS,
    database_with_lab_arena_migration,
)
from tests.lab_arena.test_lab_arena_migration_postgres import claim, complete, sha


ROOT = Path(__file__).resolve().parents[2]
MIGRATION_224 = ROOT / "scripts" / "224-recover-arena-2026-09-12-scoring.sql"
PRODUCTION_PLAN = json.loads(
    (Path(__file__).parent / "fixtures" / "arena_20260912_stage2_scoring_plan.json")
    .read_text(encoding="utf-8")
)
BOUNDED_SUBMISSION = "sub-1460c4e33f19f923b969ef2fd941072f"
FAILED_SUBMISSION = "sub-d39d12f74f5f6fb56837d22890c7dd94"
OPENROUTER_CALLS = {
    ("sub-64750217950f089a3bec21ee6fe72d6e", 15): (
        302116,
        302117,
        302118,
        "sha256:6a2453c7617a47c4454838996928101fbd6e91ae0f5fde25ba2a304cb63564be",
        20254,
    ),
    ("sub-eb8a764bc501efb7d47f9c6c74d49756", 14): (
        302110,
        302111,
        302119,
        "sha256:a2d13165186dadb801e4f57486468d12af682bd372a66cb742870b7c709eb916",
        139629,
    ),
}
OLD_LEASE_TOKEN = "224-pre-recovery-lease"
REPLAY_REQUEST_HASH = sha("224-replay-request")
REPLAY_SUBMISSION = "sub-64750217950f089a3bec21ee6fe72d6e"
REPLAY_POSITION = 15
OLD_REPLAY_ASSIGNMENT = (
    f"{ROUND_ID}:{REPLAY_SUBMISSION}:2:{REPLAY_POSITION}:score:recovery220"
)
OLD_REPLAY_IDENTITY = contracts.provider_call_identity(
    assignment_id=OLD_REPLAY_ASSIGNMENT,
    attempt=1,
    icp_position=REPLAY_POSITION,
    action_sequence=7,
    operation_id="openrouter.chat",
    request_hash=REPLAY_REQUEST_HASH,
)
RECOVERY220_ACCEPTED = {
    ("baseline-2026-09-12", 14),
    ("baseline-2026-09-12", 15),
    (BOUNDED_SUBMISSION, 14),
    ("sub-64750217950f089a3bec21ee6fe72d6e", 14),
    ("sub-67fdb5d43bc8f9ea4cb6e74df4797037", 14),
    ("sub-c2ab2d55f187807c2793433701485825", 14),
    (PRESERVED_SUBMISSION, 12),
    (PRESERVED_SUBMISSION, 13),
    (PRESERVED_SUBMISSION, 14),
    ("sub-eb8a764bc501efb7d47f9c6c74d49756", 13),
}
ORIGINAL_ACCEPTED = {
    ("baseline-2026-09-12", position) for position in range(10, 14)
} | {
    (BOUNDED_SUBMISSION, position) for position in range(10, 13)
} | {
    ("sub-64750217950f089a3bec21ee6fe72d6e", position)
    for position in range(11, 14)
} | {
    ("sub-67fdb5d43bc8f9ea4cb6e74df4797037", position)
    for position in range(10, 14)
} | {
    ("sub-c2ab2d55f187807c2793433701485825", position)
    for position in range(10, 14)
} | {
    (PRESERVED_SUBMISSION, 10),
    (PRESERVED_SUBMISSION, 11),
    ("sub-eb8a764bc501efb7d47f9c6c74d49756", 10),
    ("sub-eb8a764bc501efb7d47f9c6c74d49756", 11),
    ("sub-eb8a764bc501efb7d47f9c6c74d49756", 12),
}
PROOF_SETTLEMENTS = (
    (301857, "sub-1460c4e33f19f923b969ef2fd941072f", 11,
     "iad1::rxx97-1789180998144-d8d38ab2f371"),
    (301865, "sub-67fdb5d43bc8f9ea4cb6e74df4797037", 11,
     "iad1::f65sc-1789181027940-6ca64c78ab6d"),
    (301868, "sub-c2ab2d55f187807c2793433701485825", 11,
     "iad1::8kb7l-1789181028173-1d3c04d66e18"),
    (301908, "sub-c2ab2d55f187807c2793433701485825", 11,
     "iad1::9dnkl-1789181047530-d15b4d25be2e"),
    (301914, "sub-67fdb5d43bc8f9ea4cb6e74df4797037", 11,
     "iad1::7bfxj-1789181054267-d3bcb4315d95"),
    (301963, BOUNDED_SUBMISSION, 13,
     "iad1::ngv9f-1789181111082-cc2be8cfc3eb"),
)


@pytest.fixture(scope="module")
def database():
    yield from database_with_lab_arena_migration(POSTGREST_MIGRATIONS)


def _apply_224(connection) -> None:
    with connection.cursor() as cursor:
        cursor.execute(MIGRATION_224.read_text(encoding="utf-8"))


def _replace_stage2_with_production_plan(cursor, miners) -> None:
    cursor.execute(
        "ALTER TABLE public.lab_arena_ledger DISABLE TRIGGER "
        "lab_arena_ledger_append_only"
    )
    cursor.execute(
        "DELETE FROM public.lab_arena_ledger WHERE round_id=%s AND stage=2",
        (ROUND_ID,),
    )
    cursor.execute(
        "ALTER TABLE public.lab_arena_ledger ENABLE TRIGGER "
        "lab_arena_ledger_append_only"
    )
    cursor.execute(
        "ALTER TABLE public.lab_arena_runs DISABLE TRIGGER lab_arena_runs_terminal"
    )
    cursor.execute(
        "DELETE FROM public.lab_arena_runs WHERE round_id=%s AND stage=2",
        (ROUND_ID,),
    )
    work = {
        (item["submission_id"], item["icp_position"]): item
        for item in PRODUCTION_PLAN["work_items"]
    }
    zero = {
        (item["submission_id"], item["icp_position"]): item["cause"]
        for item in PRODUCTION_PLAN["zero_rows"]
    }
    for submission_id in PARTICIPANT_IDS:
        for position in range(10, 20):
            assignment = f"{ROUND_ID}:{submission_id}:2:{position}"
            item = work.get((submission_id, position))
            if item is not None:
                if item["scored_run_id"].endswith(":2"):
                    _insert_run(
                        cursor, run_id=f"{assignment}:1", assignment_id=assignment,
                        submission_id=submission_id, miner_hotkey=miners[submission_id],
                        stage=2, position=position, attempt=1, kind="execute",
                        status="failed", cause="model_error",
                    )
                    attempt = 2
                else:
                    attempt = 1
                _insert_run(
                    cursor, run_id=f"{assignment}:{attempt}",
                    assignment_id=assignment, submission_id=submission_id,
                    miner_hotkey=miners[submission_id], stage=2, position=position,
                    attempt=attempt, kind="execute", status="accepted",
                    cause="accepted", previous_runner=RUNNER if attempt == 2 else None,
                )
            else:
                cause = zero[(submission_id, position)]
                _insert_run(
                    cursor, run_id=f"{assignment}:1", assignment_id=assignment,
                    submission_id=submission_id, miner_hotkey=miners[submission_id],
                    stage=2, position=position, attempt=1, kind="execute",
                    status="failed", cause=cause,
                )
                if (submission_id, position) in {
                    ("sub-64750217950f089a3bec21ee6fe72d6e", 10),
                    ("sub-64750217950f089a3bec21ee6fe72d6e", 17),
                }:
                    retry_cause = "provider_error" if position == 10 else "model_error"
                    _insert_run(
                        cursor, run_id=f"{assignment}:2", assignment_id=assignment,
                        submission_id=submission_id, miner_hotkey=miners[submission_id],
                        stage=2, position=position, attempt=2, kind="execute",
                        status="failed", cause=retry_cause, previous_runner=RUNNER,
                    )

    for item in PRODUCTION_PLAN["work_items"]:
        submission_id = item["submission_id"]
        position = item["icp_position"]
        assignment = f"{ROUND_ID}:{submission_id}:2:{position}:score"
        if (submission_id, position) in ORIGINAL_ACCEPTED:
            _insert_run(
                cursor, run_id=f"{assignment}:1", assignment_id=assignment,
                submission_id=submission_id, miner_hotkey=miners[submission_id],
                stage=2, position=position, attempt=1, kind="score",
                status="accepted", cause="accepted",
                scored_run_id=item["scored_run_id"],
            )
            continue
        cause = "judge_error" if (
            submission_id == PRESERVED_SUBMISSION and position in (12, 13)
        ) else "stage_closed"
        _insert_run(
            cursor, run_id=f"{assignment}:1", assignment_id=assignment,
            submission_id=submission_id, miner_hotkey=miners[submission_id],
            stage=2, position=position, attempt=1, kind="score",
            status="failed", cause=cause, scored_run_id=item["scored_run_id"],
        )
        if submission_id == PRESERVED_SUBMISSION and position in (12, 13):
            retry_cause = "judge_error" if position == 12 else "stage_closed"
            _insert_run(
                cursor, run_id=f"{assignment}:2", assignment_id=assignment,
                submission_id=submission_id, miner_hotkey=miners[submission_id],
                stage=2, position=position, attempt=2, kind="score",
                status="failed", cause=retry_cause,
                scored_run_id=item["scored_run_id"], previous_runner=RUNNER,
            )
    cursor.execute(
        "ALTER TABLE public.lab_arena_runs ENABLE TRIGGER lab_arena_runs_terminal"
    )
    cursor.execute(
        "ALTER TABLE public.lab_arena_rounds DISABLE TRIGGER "
        "lab_arena_rounds_write_once"
    )
    cursor.execute(
        "UPDATE public.lab_arena_rounds SET stage2_scoring_plan_doc=%s::jsonb, "
        "configuration_doc=jsonb_set(configuration_doc,'{scorer_image_digest}',"
        "%s::jsonb) "
        "WHERE round_id=%s",
        (json.dumps(PRODUCTION_PLAN), json.dumps(PRODUCTION_SCORER), ROUND_ID),
    )
    cursor.execute(
        "ALTER TABLE public.lab_arena_rounds ENABLE TRIGGER "
        "lab_arena_rounds_write_once"
    )


def _insert_billing_evidence(cursor, miners) -> None:
    target_run_11 = f"{ROUND_ID}:{PRESERVED_SUBMISSION}:2:11:score:1"
    target_run_12 = f"{ROUND_ID}:{PRESERVED_SUBMISSION}:2:12:score:1"
    oversized_doc = json.dumps({"reason": "worker_reported", "call": {
        "reason": "missing_provider_cost", "provider_status": 502,
        "response_provenance": "response_too_large",
    }})
    cursor.execute(
        "INSERT INTO public.lab_arena_ledger "
        "(entry_id,entry_kind,miner_hotkey,round_id,submission_id,run_id,stage,"
        "call_identity,provider,operation_id,funding_source,amount_microusd,"
        "entry_doc,terminal_response,created_at) VALUES "
        "(301884,'settlement',%s,%s,%s,%s,2,%s,'deepline',"
        "'scrapingdog.scrape','miner_key',2000,'{}'::jsonb,%s::jsonb,"
        "'2026-09-12T02:44:00Z'),"
        "(301934,'uncertain',%s,%s,%s,%s,2,%s,'deepline',"
        "'scrapingdog.scrape','miner_key',49329183,%s::jsonb,NULL,"
        "'2026-09-12T02:44:33Z'),"
        "(301941,'uncertain',%s,%s,%s,%s,2,%s,'deepline',"
        "'scrapingdog.scrape','miner_key',4217,%s::jsonb,NULL,"
        "'2026-09-12T02:44:39Z')",
        (
            miners[PRESERVED_SUBMISSION], ROUND_ID, PRESERVED_SUBMISSION,
            target_run_11,
            "sha256:21d55103563b01712ff2033ddcc10c55e23e8712e99c485e94b912760a66d880",
            json.dumps({"status": 200, "provider_cost": {
                "request_id": "iad1::nrll6-1789181035861-44985ee12c9a",
            }}),
            miners[PRESERVED_SUBMISSION], ROUND_ID, PRESERVED_SUBMISSION,
            target_run_12,
            "sha256:6acff63503cff507fd39853485d8bf58a8b8bca6c50ae0a64be1afa774149b74",
            oversized_doc,
            miners[PRESERVED_SUBMISSION], ROUND_ID, PRESERVED_SUBMISSION,
            target_run_12,
            "sha256:ba4e76b407cf02a5786b35a5ef03c040f3810b60f1c81070c73c6f654a213a92",
            oversized_doc,
        ),
    )
    for entry_id, submission_id, position, request_id in PROOF_SETTLEMENTS:
        run_id = f"{ROUND_ID}:{submission_id}:2:{position}:score:1"
        cursor.execute(
            "INSERT INTO public.lab_arena_ledger "
            "(entry_id,entry_kind,miner_hotkey,round_id,submission_id,run_id,stage,"
            "call_identity,provider,operation_id,funding_source,amount_microusd,"
            "entry_doc,terminal_response,created_at) VALUES "
            "(%s,'settlement',%s,%s,%s,%s,2,%s,'deepline',"
            "'scrapingdog.scrape','miner_key',2000,'{}'::jsonb,%s::jsonb,"
            "'2026-09-12T02:44:00Z')",
            (
                entry_id, miners[submission_id], ROUND_ID, submission_id, run_id,
                sha(f"224-proof-{entry_id}"),
                json.dumps({"status": 200, "provider_cost": {
                    "basis": "deepline_billing_credits_charged_x_0.10_usd",
                    "units": "0.02", "operation": "firecrawl_scrape",
                    "unit_name": "credits", "request_id": request_id,
                }}),
            ),
        )
    call_identity = (
        "sha256:d1c0ba74a77451bf0a65c950df16186cd537fd146389c0291adbe543699fe742"
    )
    run_id = f"{ROUND_ID}:{BOUNDED_SUBMISSION}:2:13:score:1"
    for entry_id, kind in ((301970, "reservation"), (301971, "dispatch")):
        cursor.execute(
            "INSERT INTO public.lab_arena_ledger "
            "(entry_id,entry_kind,miner_hotkey,round_id,submission_id,run_id,stage,"
            "call_identity,provider,operation_id,funding_source,amount_microusd,"
            "entry_doc,created_at) VALUES "
            "(%s,%s,%s,%s,%s,%s,2,%s,'deepline','scrapingdog.scrape',"
            "'miner_key',49294310,'{}'::jsonb,'2026-09-12T02:45:26Z')",
            (entry_id, kind, miners[BOUNDED_SUBMISSION], ROUND_ID,
             BOUNDED_SUBMISSION, run_id, call_identity),
        )
    cursor.execute(
        "INSERT INTO public.lab_arena_ledger "
        "(entry_id,entry_kind,miner_hotkey,round_id,submission_id,run_id,stage,"
        "call_identity,provider,operation_id,funding_source,amount_microusd,"
        "entry_doc,terminal_response,created_at) VALUES "
        "(301984,'uncertain',%s,%s,%s,%s,2,%s,'deepline',"
        "'scrapingdog.scrape','miner_key',49294310,%s::jsonb,NULL,"
        "'2026-09-12T02:45:46Z')",
        (miners[BOUNDED_SUBMISSION], ROUND_ID, BOUNDED_SUBMISSION, run_id,
         call_identity, json.dumps({"reason": "round_cancelled"})),
    )


def _insert_current_cost_guards(cursor, miners) -> None:
    cursor.execute(
        "UPDATE public.lab_arena_ledger SET entry_id=289258, provider='deepline', "
        "operation_id='scrapingdog.scrape', amount_microusd=50000000, "
        "run_id=%s, call_identity=%s, entry_doc=%s::jsonb WHERE entry_kind='uncertain' "
        "AND submission_id=%s AND stage=1",
        (
            f"{ROUND_ID}:{FAILED_SUBMISSION}:1:0:score:1",
            "sha256:ff08e5a119653f7931bc6e3b6f574e6ccf35109d9acd8dd8028282d574ab64e3",
            json.dumps({"reason": "worker_reported", "call": {
                "reason": "missing_provider_cost", "provider_status": 402,
                "billing_present": True,
            }}),
            FAILED_SUBMISSION,
        ),
    )
    assert cursor.rowcount == 1
    max_run = f"{ROUND_ID}:baseline-2026-09-12:2:10:score:1"
    max_call = sha("224-max-openrouter-reservation")
    cursor.execute(
        "INSERT INTO public.lab_arena_ledger "
        "(entry_id,entry_kind,miner_hotkey,round_id,submission_id,run_id,stage,"
        "call_identity,provider,operation_id,funding_source,amount_microusd,entry_doc) "
        "VALUES (300000,'reservation',%s,%s,'baseline-2026-09-12',%s,2,%s,"
        "'openrouter','openrouter.chat','miner_key',156957,'{}'::jsonb),"
        "(300001,'settlement',%s,%s,'baseline-2026-09-12',%s,2,%s,"
        "'openrouter','openrouter.chat','miner_key',1000,'{}'::jsonb)",
        (miners["baseline-2026-09-12"], ROUND_ID, max_run, max_call,
         miners["baseline-2026-09-12"], ROUND_ID, max_run, max_call),
    )
    for (submission_id, position), (
        reservation_id, dispatch_id, uncertain_id, call_identity, amount
    ) in OPENROUTER_CALLS.items():
        run_id = f"{ROUND_ID}:{submission_id}:2:{position}:score:1"
        cursor.execute(
            "INSERT INTO public.lab_arena_ledger "
            "(entry_id,entry_kind,miner_hotkey,round_id,submission_id,run_id,stage,"
            "call_identity,provider,operation_id,funding_source,amount_microusd,"
            "entry_doc,terminal_response,created_at) VALUES "
            "(%s,'reservation',%s,%s,%s,%s,2,%s,'openrouter','openrouter.chat',"
            "'miner_key',%s,'{}'::jsonb,NULL,'2026-09-12T03:32:30Z'),"
            "(%s,'dispatch',%s,%s,%s,%s,2,%s,'openrouter','openrouter.chat',"
            "'miner_key',%s,'{}'::jsonb,NULL,'2026-09-12T03:32:31Z'),"
            "(%s,'uncertain',%s,%s,%s,%s,2,%s,'openrouter','openrouter.chat',"
            "'miner_key',%s,%s::jsonb,NULL,'2026-09-12T03:32:37Z')",
            (
                reservation_id, miners[submission_id], ROUND_ID, submission_id,
                run_id, call_identity, amount,
                dispatch_id, miners[submission_id], ROUND_ID, submission_id,
                run_id, call_identity, amount,
                uncertain_id, miners[submission_id], ROUND_ID, submission_id,
                run_id, call_identity, amount,
                json.dumps({"reason": "round_cancelled"}),
            ),
        )
    cursor.execute(
        "INSERT INTO public.lab_arena_ledger "
        "(entry_id,entry_kind,miner_hotkey,round_id,submission_id,run_id,stage,"
        "call_identity,provider,operation_id,funding_source,amount_microusd,"
        "entry_doc,created_at) VALUES "
        "(302050,'refusal',%s,%s,%s,%s,2,%s,'openrouter','openrouter.chat',"
        "'miner_key',0,%s::jsonb,'2026-09-12T03:32:38Z')",
        (
            miners[REPLAY_SUBMISSION], ROUND_ID, REPLAY_SUBMISSION,
            f"{ROUND_ID}:{REPLAY_SUBMISSION}:2:{REPLAY_POSITION}:score:1",
            OLD_REPLAY_IDENTITY,
            json.dumps({"reason": "provider_cost_uncertain"}),
        ),
    )


def _seed_third_cancellation(database):
    connection, store, transport, _plan_md5, _scorer_digest = _seed_scoring_failure(
        database
    )
    with connection.cursor() as cursor:
        cursor.execute(
            "SELECT submission_id,miner_hotkey FROM public.lab_arena_submissions "
            "WHERE round_id=%s", (ROUND_ID,),
        )
        miners = dict(cursor.fetchall())
        _replace_stage2_with_production_plan(cursor, miners)
        _insert_billing_evidence(cursor, miners)
        cursor.execute(
            "SELECT md5(stage2_scoring_plan_doc::text) FROM public.lab_arena_rounds "
            "WHERE round_id=%s", (ROUND_ID,),
        )
        assert cursor.fetchone()[0] == PRODUCTION_PLAN_MD5
        cursor.execute(
            "SELECT count(*) FROM jsonb_array_elements(%s::jsonb->'work_items') item "
            "JOIN public.lab_arena_runs run ON run.run_id=item->>'scored_run_id' "
            "WHERE run.round_id=%s AND run.stage=2 AND run.kind='execute' "
            "AND run.status='accepted'",
            (json.dumps(PRODUCTION_PLAN), ROUND_ID),
        )
        assert cursor.fetchone()[0] == 68
    _apply_220(connection, PRODUCTION_PLAN_MD5, PRODUCTION_SCORER)

    with connection.cursor() as cursor:
        cursor.execute(
            "ALTER TABLE public.lab_arena_runs DISABLE TRIGGER lab_arena_runs_terminal"
        )
        cursor.execute(
            "UPDATE public.lab_arena_runs SET status='failed',terminal_cause='stage_closed',"
            "terminal_doc='{}'::jsonb WHERE round_id=%s AND stage=2 AND kind='score' "
            "AND status='pending'",
            (ROUND_ID,),
        )
        for submission_id, position in RECOVERY220_ACCEPTED:
            assignment = f"{ROUND_ID}:{submission_id}:2:{position}:score:recovery220"
            cursor.execute(
                "UPDATE public.lab_arena_runs SET status='accepted',terminal_cause='accepted',"
                "result_doc='{}'::jsonb,output_ref=%s,terminal_doc='{}'::jsonb "
                "WHERE assignment_id=%s AND attempt=(SELECT max(attempt) FROM "
                "public.lab_arena_runs WHERE assignment_id=%s)",
                (f"arena/{ROUND_ID}/recovered/{assignment}.json", assignment, assignment),
            )
            assert cursor.rowcount == 1
        for position, retry_cause in ((13, "judge_error"), (15, "stage_closed")):
            assignment = f"{ROUND_ID}:{BOUNDED_SUBMISSION}:2:{position}:score:recovery220"
            cursor.execute(
                "UPDATE public.lab_arena_runs SET terminal_cause='judge_error' "
                "WHERE assignment_id=%s AND attempt=1",
                (assignment,),
            )
            assert cursor.rowcount == 1
            scored_run_id = next(
                item["scored_run_id"] for item in PRODUCTION_PLAN["work_items"]
                if item["submission_id"] == BOUNDED_SUBMISSION
                and item["icp_position"] == position
            )
            _insert_run(
                cursor, run_id=f"{assignment}:2", assignment_id=assignment,
                submission_id=BOUNDED_SUBMISSION,
                miner_hotkey=miners[BOUNDED_SUBMISSION], stage=2,
                position=position, attempt=2, kind="score", status="failed",
                cause=retry_cause, scored_run_id=scored_run_id,
                previous_runner=RUNNER,
            )
        cursor.execute(
            "UPDATE public.lab_arena_runs SET runner_hotkey=%s,lease_token_hash=%s,"
            "claim_request_id=%s,claim_request_hash=%s,claim_response='{}'::jsonb,"
            "lease_expires_at=clock_timestamp()+interval '5 minutes' "
            "WHERE run_id=%s",
            (
                RUNNER, hash_lease_token(OLD_LEASE_TOKEN), "2" * 32,
                sha("224-old-claim"),
                f"{ROUND_ID}:{REPLAY_SUBMISSION}:2:{REPLAY_POSITION}:score:1",
            ),
        )
        assert cursor.rowcount == 1
        cursor.execute(
            "ALTER TABLE public.lab_arena_runs ENABLE TRIGGER lab_arena_runs_terminal"
        )
        cursor.execute(
            "ALTER TABLE public.lab_arena_rounds DISABLE TRIGGER "
            "lab_arena_rounds_write_once"
        )
        cursor.execute(
            "UPDATE public.lab_arena_rounds SET status='cancelled',status_generation=13,"
            "stage_generation=11,cancel_reason='scoring_incomplete' WHERE round_id=%s",
            (ROUND_ID,),
        )
        cursor.execute(
            "ALTER TABLE public.lab_arena_rounds ENABLE TRIGGER "
            "lab_arena_rounds_write_once"
        )
        cursor.execute(
            "ALTER TABLE public.lab_arena_ledger DISABLE TRIGGER "
            "lab_arena_ledger_append_only"
        )
        _insert_current_cost_guards(cursor, miners)
        cursor.execute(
            "ALTER TABLE public.lab_arena_ledger ENABLE TRIGGER "
            "lab_arena_ledger_append_only"
        )
    return connection, store, transport


def test_third_recovery_preserves_work_costs_and_completes(database):
    connection, store, transport = _seed_third_cancellation(database)
    try:
        with connection.cursor() as cursor:
            before = {
                "accepted": _row_hash(cursor, "lab_arena_runs",
                                      "round_id=%s AND status='accepted'", (ROUND_ID,)),
                "submissions": _row_hash(cursor, "lab_arena_submissions",
                                          "round_id=%s", (ROUND_ID,)),
                "credentials": _row_hash(
                    cursor, "lab_arena_submission_credentials",
                    "submission_id=ANY(%s)", (list(PARTICIPANT_IDS),),
                ),
                "other_ledger": _row_hash(
                    cursor, "lab_arena_ledger",
                    "round_id=%s AND entry_id<>301984", (ROUND_ID,),
                ),
            }
            cursor.execute(
                "SELECT assignment_id,run_id FROM (SELECT DISTINCT ON (assignment_id) "
                "assignment_id,run_id,status FROM public.lab_arena_runs WHERE "
                "round_id=%s AND stage=2 AND kind='score' ORDER BY assignment_id,"
                "(status='accepted') DESC,attempt DESC) latest WHERE status<>'accepted' "
                "ORDER BY run_id", (ROUND_ID,),
            )
            old_incomplete_rows = cursor.fetchall()
            old_incomplete_assignments = [row[0] for row in old_incomplete_rows]
            old_incomplete_run_ids = {row[1] for row in old_incomplete_rows}
            assert len(old_incomplete_run_ids) == 35
            cursor.execute(
                "SELECT run_id FROM public.lab_arena_runs WHERE round_id=%s "
                "AND stage=2 AND kind='score' AND assignment_id=ANY(%s) "
                "ORDER BY run_id",
                (ROUND_ID, old_incomplete_assignments),
            )
            old_incomplete_lineage_run_ids = {row[0] for row in cursor.fetchall()}
            assert len(old_incomplete_lineage_run_ids) == 37
            cursor.execute(
                "SELECT lease_generation FROM public.lab_arena_runs WHERE run_id=%s",
                (f"{ROUND_ID}:{REPLAY_SUBMISSION}:2:{REPLAY_POSITION}:score:1",),
            )
            replay_old_lease_generation = cursor.fetchone()[0]
            cursor.execute(
                "SELECT count(*),count(distinct assignment_id),"
                "count(*) filter(where status='accepted'),"
                "count(*) filter(where terminal_cause='judge_error'),"
                "count(*) filter(where terminal_cause='stage_closed') "
                "FROM public.lab_arena_runs WHERE round_id=%s AND stage=2 "
                "AND kind='score'", (ROUND_ID,),
            )
            assert cursor.fetchone() == (72, 68, 33, 5, 34)
        _apply_224(connection)
        recovered_round = store.get_round(ROUND_ID)
        assert (recovered_round["status"], recovered_round["status_generation"],
                recovered_round["stage_generation"]) == (
            "stage2_scoring", 14, 12,
        )
        with connection.cursor() as cursor:
            assert _row_hash(cursor, "lab_arena_runs",
                             "round_id=%s AND status='accepted'", (ROUND_ID,)) == before["accepted"]
            assert _row_hash(cursor, "lab_arena_submissions",
                             "round_id=%s", (ROUND_ID,)) == before["submissions"]
            assert _row_hash(cursor, "lab_arena_submission_credentials",
                             "submission_id=ANY(%s)", (list(PARTICIPANT_IDS),)) == before["credentials"]
            assert _row_hash(cursor, "lab_arena_ledger",
                             "round_id=%s AND entry_id<>301984", (ROUND_ID,)) == before["other_ledger"]
            cursor.execute(
                "SELECT amount_microusd,entry_kind,terminal_response,"
                "entry_doc#>>'{recovery224,original_amount_microusd}',"
                "entry_doc#>>'{recovery224,unresolved_group_upper_bound_microusd}' "
                "FROM public.lab_arena_ledger WHERE entry_id=301984"
            )
            assert cursor.fetchone() == (2000, "uncertain", None, "49294310", "2000")
            cursor.execute(
                "SELECT count(*),count(distinct assignment_id),"
                "count(*) filter(where status='pending') FROM public.lab_arena_runs "
                "WHERE round_id=%s AND stage=2 AND kind='score' AND "
                "assignment_id LIKE '%%:recovery224'", (ROUND_ID,),
            )
            assert cursor.fetchone() == (37, 35, 35)
            cursor.execute(
                "SELECT count(*) FROM public.lab_arena_ledger WHERE entry_id IN "
                "(301934,301941,302118,302119) AND amount_microusd IN "
                "(4000,20254,139629)"
            )
            assert cursor.fetchone()[0] == 4
            cursor.execute(
                "SELECT runner_hotkey,lease_token_hash,lease_expires_at,"
                "claim_request_id,claim_request_hash,claim_response,lease_generation "
                "FROM public.lab_arena_runs WHERE run_id=%s",
                (f"{ROUND_ID}:{REPLAY_SUBMISSION}:2:{REPLAY_POSITION}:score:1",),
            )
            assert cursor.fetchone() == (
                None, None, None, None, None, None, replay_old_lease_generation + 1,
            )

        preserved_run_ids = {
            run["run_id"] for run in store.list_runs(ROUND_ID, stage=2, kind="score")
            if run["assignment_id"].endswith(":recovery224")
        }
        assert preserved_run_ids == old_incomplete_lineage_run_ids
        pending_run_ids = {
            run["run_id"] for run in store.list_runs(ROUND_ID, stage=2, kind="score")
            if run["status"] == "pending"
        }
        assert pending_run_ids == old_incomplete_run_ids
        before_immediate_replay = store.get_round(ROUND_ID)
        _apply_224(connection)
        assert store.get_round(ROUND_ID) == before_immediate_replay

        completed = 0
        fresh_identity_checked = False
        while True:
            response, token, _, _ = claim(
                store, ROUND_ID, RUNNER, parallelism=100, ceiling=100
            )
            if response["status"] != "leased":
                break
            assert response["run_id"] in preserved_run_ids
            assert response["assignment_id"].endswith(":recovery220:recovery224")
            token_hash = hash_lease_token(token)
            if (
                response["submission_id"] == REPLAY_SUBMISSION
                and response["icp_position"] == REPLAY_POSITION
            ):
                stale_identity = sha("224-stale-token-call")
                assert store.reserve_call(
                    run_id=response["run_id"],
                    lease_token_hash=hash_lease_token(OLD_LEASE_TOKEN),
                    call_identity=stale_identity,
                    operation_id="openrouter.chat", provider="openrouter",
                    funding_source="miner_key", amount_microusd=1000,
                    call_doc={},
                )["status"] == "stale"
                assert store.settle_call(
                    run_id=response["run_id"],
                    lease_token_hash=hash_lease_token(OLD_LEASE_TOKEN),
                    call_identity=stale_identity, actual_microusd=500,
                    terminal_response={"status": 200},
                )["status"] == "stale"
                old = store.reserve_call(
                    run_id=response["run_id"], lease_token_hash=token_hash,
                    call_identity=OLD_REPLAY_IDENTITY,
                    operation_id="openrouter.chat", provider="openrouter",
                    funding_source="miner_key", amount_microusd=1000,
                    call_doc={},
                )
                assert old["status"] == "refused" and old["idempotent"] is True
                new_identity = contracts.provider_call_identity(
                    assignment_id=response["assignment_id"], attempt=1,
                    icp_position=REPLAY_POSITION, action_sequence=7,
                    operation_id="openrouter.chat",
                    request_hash=REPLAY_REQUEST_HASH,
                )
                assert new_identity != OLD_REPLAY_IDENTITY
                assert store.reserve_call(
                    run_id=response["run_id"], lease_token_hash=token_hash,
                    call_identity=new_identity, operation_id="openrouter.chat",
                    provider="openrouter", funding_source="miner_key",
                    amount_microusd=1000, call_doc={},
                )["status"] == "reserved"
                assert store.mark_dispatched(
                    run_id=response["run_id"], lease_token_hash=token_hash,
                    call_identity=new_identity,
                )["status"] == "dispatched"
                assert store.settle_call(
                    run_id=response["run_id"], lease_token_hash=token_hash,
                    call_identity=new_identity, actual_microusd=500,
                    terminal_response={"status": 200},
                )["status"] == "settled"
                fresh_identity_checked = True
            assert complete(
                store, response["run_id"], token_hash, "accepted",
                output_ref=f"arena/{ROUND_ID}/recovered/{response['run_id']}.json",
            )["status"] == "accepted"
            completed += 1
        assert completed == 35
        assert fresh_identity_checked
        closed = store.close_scoring(ROUND_ID, 2)
        assert closed["round_status"] == "stage2_judged"
        assert closed["incomplete_assignments"] == 0
        before_replay = store.get_round(ROUND_ID)
        _apply_224(connection)
        assert store.get_round(ROUND_ID) == before_replay
    finally:
        connection.close()
        transport.close()


def test_recovery_rolls_back_and_rejects_a_new_cancellation(database):
    connection, store, transport = _seed_third_cancellation(database)
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                "CREATE OR REPLACE FUNCTION public.test_224_forced_failure() "
                "RETURNS trigger LANGUAGE plpgsql AS $$BEGIN "
                "IF NEW.assignment_id LIKE '%:recovery224' THEN "
                "RAISE EXCEPTION 'test_224_forced_failure'; END IF; RETURN NEW; END$$"
            )
            cursor.execute(
                "CREATE TRIGGER test_224_forced_failure BEFORE UPDATE ON "
                "public.lab_arena_runs FOR EACH ROW EXECUTE FUNCTION "
                "public.test_224_forced_failure()"
            )
        with pytest.raises(Exception, match="test_224_forced_failure"):
            _apply_224(connection)
        with connection.cursor() as cursor:
            cursor.execute("ROLLBACK")
            cursor.execute("DROP TRIGGER test_224_forced_failure ON public.lab_arena_runs")
            cursor.execute("DROP FUNCTION public.test_224_forced_failure()")
        connection.commit()
        assert store.get_round(ROUND_ID)["status"] == "cancelled"
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT amount_microusd,entry_doc ? 'recovery224' "
                "FROM public.lab_arena_ledger WHERE entry_id=301984"
            )
            assert cursor.fetchone() == (49294310, False)
            cursor.execute(
                "SELECT bool_and(tgenabled='O') FROM pg_catalog.pg_trigger WHERE "
                "(tgrelid='public.lab_arena_runs'::regclass AND tgname='lab_arena_runs_terminal') OR "
                "(tgrelid='public.lab_arena_rounds'::regclass AND tgname='lab_arena_rounds_write_once') OR "
                "(tgrelid='public.lab_arena_ledger'::regclass AND tgname='lab_arena_ledger_append_only')"
            )
            assert cursor.fetchone()[0] is True

        _apply_224(connection)
        first_assignment = None
        retry_id = None
        exhausted_checked = False
        retry_checked = False
        while True:
            claimed, token, _, _ = claim(
                store, ROUND_ID, RUNNER, parallelism=100, ceiling=100
            )
            if claimed["status"] != "leased":
                break
            token_hash = hash_lease_token(token)
            if (
                claimed["submission_id"] == BOUNDED_SUBMISSION
                and claimed["icp_position"] == 13
                and claimed["attempt"] == 2
            ):
                exhausted = complete(
                    store, claimed["run_id"], token_hash, "judge_error"
                )
                assert "confirmation_attempt" not in exhausted
                assert store.get_run(claimed["assignment_id"] + ":3") is None
                exhausted_checked = True
            elif first_assignment is None and claimed["attempt"] == 1:
                first_assignment = claimed["assignment_id"]
                failed = complete(
                    store, claimed["run_id"], token_hash, "judge_error"
                )
                assert failed["confirmation_attempt"] == 2
                retry_id = first_assignment + ":2"
                assert store.get_run(retry_id)["status"] == "pending"
            elif retry_id is not None and claimed["run_id"] == retry_id:
                failed_retry = complete(
                    store, claimed["run_id"], token_hash, "judge_error"
                )
                assert "confirmation_attempt" not in failed_retry
                assert store.get_run(first_assignment + ":3") is None
                retry_checked = True
            # Other claims remain leased so close_scoring exercises the
            # canonical stale-lease and incomplete-round transition.
        assert exhausted_checked and retry_checked
        cancelled = store.close_scoring(ROUND_ID, 2)
        assert cancelled["round_status"] == "cancelled"
        with pytest.raises(Exception, match="recovery replay state differs"):
            _apply_224(connection)
        with connection.cursor() as cursor:
            cursor.execute("ROLLBACK")
    finally:
        connection.close()
        transport.close()


def test_recovery_rejects_an_unrelated_scoring_uncertainty(database):
    connection, _store, transport = _seed_third_cancellation(database)
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                "INSERT INTO public.lab_arena_ledger "
                "(entry_kind,miner_hotkey,round_id,submission_id,run_id,stage,"
                "call_identity,provider,operation_id,funding_source,amount_microusd,"
                "entry_doc) SELECT 'uncertain',miner_hotkey,round_id,submission_id,"
                "run_id,stage,%s,'openrouter','openrouter.chat','miner_key',"
                "50000000,%s::jsonb FROM public.lab_arena_runs WHERE run_id=%s",
                (
                    sha("224-unrelated-uncertainty"),
                    json.dumps({"reason": "round_cancelled"}),
                    f"{ROUND_ID}:baseline-2026-09-12:2:16:score:1",
                ),
            )
            assert cursor.rowcount == 1
        with pytest.raises(Exception, match="scoring cost guard differs"):
            _apply_224(connection)
        with connection.cursor() as cursor:
            cursor.execute("ROLLBACK")
    finally:
        connection.close()
        transport.close()
