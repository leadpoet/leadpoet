"""Second guarded recovery of arena-2026-09-13 stage-1 scoring."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from lab_arena import contracts, judgment_cache
from lab_arena.store import hash_lease_token
from tests.lab_arena.arena_20260912_recovery_postgres_test import _row_hash
from tests.lab_arena.arena_20260913_scoring_recovery_postgres_test import (
    ASPIRE_SUBMISSION,
    JUDGE_RUNNER,
    MIGRATION as FIRST_RECOVERY,
    NEW_DIGEST as CURRENT_DIGEST,
    NEW_REFERENCE as CURRENT_REFERENCE,
    PARTICIPANT_IDS,
    ROUND_ID,
    UNRELATED_ROUND_ID,
    _scope,
    _seed,
)
from tests.lab_arena.lab_arena_pg_harness import (
    POSTGREST_MIGRATIONS,
    database_with_lab_arena_migration,
)
from tests.lab_arena.test_lab_arena_migration_postgres import claim, sha


ROOT = Path(__file__).resolve().parents[2]
MIGRATION = ROOT / "scripts" / "235-recover-arena-2026-09-13-scoring-retry.sql"
NEWLY_ACCEPTED = (
    (
        f"{ROUND_ID}:{ASPIRE_SUBMISSION}:1:1:score:recovery234",
        f"{ROUND_ID}:{ASPIRE_SUBMISSION}:1:1:score:2",
        1,
    ),
    (
        f"{ROUND_ID}:sub-028c4e5c655e855f1c343fa39274cd2c:1:1:score:recovery234",
        f"{ROUND_ID}:sub-028c4e5c655e855f1c343fa39274cd2c:1:1:score:2",
        1,
    ),
    (
        f"{ROUND_ID}:sub-5dffdbaa2b96e8dc78160aea8f80a7b9:1:2:score:recovery234",
        f"{ROUND_ID}:sub-5dffdbaa2b96e8dc78160aea8f80a7b9:1:2:score:1",
        2,
    ),
    (
        f"{ROUND_ID}:sub-6211e8d46819c34df3418ded36f788ef:1:2:score:recovery234",
        f"{ROUND_ID}:sub-6211e8d46819c34df3418ded36f788ef:1:2:score:1",
        2,
    ),
)
NEW_JUDGE_FAILURES = (
    f"{ROUND_ID}:{ASPIRE_SUBMISSION}:1:0:score:recovery234",
    f"{ROUND_ID}:sub-5c1f20eb379bcd4e11c29912b5251b0d:1:1:score:recovery234",
)


@pytest.fixture(scope="module")
def database():
    yield from database_with_lab_arena_migration(POSTGREST_MIGRATIONS)


def _migration_text() -> tuple[str, str, str]:
    text = MIGRATION.read_text(encoding="utf-8")
    digest = re.search(
        r"v_new_digest CONSTANT TEXT :=\s*'([^']+)';", text
    ).group(1)
    reference = re.search(
        r"v_new_reference CONSTANT TEXT :=\s*'([^']+)';", text
    ).group(1)
    assert re.fullmatch(r"sha256:[0-9a-f]{64}", digest)
    assert reference.endswith("@" + digest)
    return text, digest, reference


def _prepare_second_cancellation(database):
    connection, store, transport = _seed(database)
    with connection.cursor() as cursor:
        cursor.execute(FIRST_RECOVERY.read_text(encoding="utf-8"))
        cursor.execute(
            "ALTER TABLE public.lab_arena_runs DISABLE TRIGGER "
            "lab_arena_runs_terminal"
        )
        cursor.execute(
            "ALTER TABLE public.lab_arena_runs DISABLE TRIGGER "
            "lab_arena_integrity_run_guard"
        )
        for index, (assignment_id, run_id, position) in enumerate(NEWLY_ACCEPTED):
            scope = _scope(
                position,
                CURRENT_DIGEST,
                CURRENT_REFERENCE,
                marker=f"recovery234-accepted-{index}",
            )
            cursor.execute(
                "UPDATE public.lab_arena_runs SET status='accepted',"
                "runner_hotkey=%s,result_doc=%s::jsonb,output_ref=%s,"
                "terminal_cause='accepted',per_icp_score=1.0,"
                "judgment_cache_key=%s,judgment_input_hash=%s,"
                "judgment_scope_doc=%s::jsonb,"
                "judgment_cache_source_run_id=run_id "
                "WHERE assignment_id=%s AND run_id=%s AND status='pending'",
                (
                    JUDGE_RUNNER,
                    json.dumps({"terminal_status": "accepted"}),
                    f"arena/{ROUND_ID}/scores/items/{run_id}.json",
                    scope["cache_key"],
                    scope["scoring_input_hash"],
                    json.dumps(scope),
                    assignment_id,
                    run_id,
                ),
            )
            assert cursor.rowcount == 1
            cursor.execute(
                "SELECT scored_run_id FROM public.lab_arena_runs WHERE run_id=%s",
                (run_id,),
            )
            scored_run_id = cursor.fetchone()[0]
            evidence = {
                "cache_key": scope["cache_key"],
                "scoring_input_hash": scope["scoring_input_hash"],
                "source_score_run_id": run_id,
                "source_scored_run_id": scored_run_id,
                "source_runner_hotkey": JUDGE_RUNNER,
                "runner_authority_exclusions": [JUDGE_RUNNER],
            }
            cursor.execute(
                "INSERT INTO public.lab_arena_judgment_cache (cache_key,"
                "scope_doc,scoring_input_hash,evidence_hash,evidence_doc,"
                "source_score_run_id,source_scored_run_id,source_runner_hotkey) "
                "VALUES (%s,%s::jsonb,%s,%s,%s::jsonb,%s,%s,%s)",
                (
                    scope["cache_key"],
                    json.dumps(scope),
                    scope["scoring_input_hash"],
                    contracts.document_hash(evidence),
                    json.dumps(evidence),
                    run_id,
                    scored_run_id,
                    JUDGE_RUNNER,
                ),
            )
            cursor.execute(
                "INSERT INTO public.lab_arena_ledger (entry_kind,miner_hotkey,"
                "round_id,submission_id,run_id,stage,call_identity,provider,"
                "operation_id,funding_source,amount_microusd,entry_doc,"
                "terminal_response) SELECT 'settlement',miner_hotkey,round_id,"
                "submission_id,run_id,stage,%s,'openrouter','openrouter.chat',"
                "'miner_key',%s,'{}'::jsonb,'{\"status\":200}'::jsonb "
                "FROM public.lab_arena_runs WHERE run_id=%s",
                (sha(f"recovery234-cost-{index}"), 400 + index, run_id),
            )

        accepted_ids = [item[1] for item in NEWLY_ACCEPTED]
        cursor.execute(
            "UPDATE public.lab_arena_runs SET status='failed',"
            "terminal_cause='stage_closed',"
            "result_doc='{\"terminal_status\":\"stage_closed\"}'::jsonb "
            "WHERE round_id=%s AND stage=1 AND kind='score' "
            "AND status='pending' AND NOT (run_id=ANY(%s))",
            (ROUND_ID, accepted_ids),
        )
        assert cursor.rowcount == 84
        cursor.execute(
                "UPDATE public.lab_arena_runs SET terminal_cause='judge_error',"
                "result_doc='{\"terminal_status\":\"judge_error\"}'::jsonb,"
                "lease_token_hash=%s WHERE assignment_id=ANY(%s) AND attempt=2",
            (hash_lease_token("recovery234-old-lease"), list(NEW_JUDGE_FAILURES)),
        )
        assert cursor.rowcount == 2
        cursor.execute(
            "ALTER TABLE public.lab_arena_runs ENABLE TRIGGER "
            "lab_arena_integrity_run_guard"
        )
        cursor.execute(
            "ALTER TABLE public.lab_arena_runs ENABLE TRIGGER "
            "lab_arena_runs_terminal"
        )
        cursor.execute(
            "ALTER TABLE public.lab_arena_rounds DISABLE TRIGGER "
            "lab_arena_rounds_write_once"
        )
        cursor.execute(
            "UPDATE public.lab_arena_rounds SET status='cancelled',"
            "status_generation=7,stage_generation=6,"
            "cancel_reason='scoring_incomplete' WHERE round_id=%s",
            (ROUND_ID,),
        )
        cursor.execute(
            "ALTER TABLE public.lab_arena_rounds ENABLE TRIGGER "
            "lab_arena_rounds_write_once"
        )
    return connection, store, transport


def _preserved_hashes(cursor) -> dict[str, str]:
    return {
        "accepted": _row_hash(
            cursor, "lab_arena_runs",
            "round_id=%s AND status='accepted'", (ROUND_ID,),
        ),
        "non_score": _row_hash(
            cursor, "lab_arena_runs",
            "round_id=%s AND kind<>'score'", (ROUND_ID,),
        ),
        "submissions": _row_hash(
            cursor, "lab_arena_submissions", "round_id=%s", (ROUND_ID,),
        ),
        "credentials": _row_hash(
            cursor, "lab_arena_submission_credentials",
            "submission_id=ANY(%s)", (list(PARTICIPANT_IDS),),
        ),
        "ledger": _row_hash(
            cursor, "lab_arena_ledger", "round_id=%s", (ROUND_ID,),
        ),
        "cache": _row_hash(
            cursor, "lab_arena_judgment_cache",
            "scope_doc->>'round_id'=%s", (ROUND_ID,),
        ),
        "unrelated_round": _row_hash(
            cursor, "lab_arena_rounds", "round_id=%s", (UNRELATED_ROUND_ID,),
        ),
    }


def _extend_test_deadline(cursor) -> None:
    cursor.execute(
        "ALTER TABLE public.lab_arena_rounds DISABLE TRIGGER "
        "lab_arena_rounds_write_once"
    )
    cursor.execute(
        "UPDATE public.lab_arena_rounds SET configuration_doc=jsonb_set("
        "configuration_doc,'{schedule,stage_1_scoring_close}',"
        "'\"2100-01-01T00:00:00Z\"'::jsonb,FALSE) WHERE round_id=%s",
        (ROUND_ID,),
    )
    cursor.execute(
        "ALTER TABLE public.lab_arena_rounds ENABLE TRIGGER "
        "lab_arena_rounds_write_once"
    )


def test_second_recovery_preserves_new_work_costs_and_completes(database):
    connection, store, transport = _prepare_second_cancellation(database)
    migration, new_digest, new_reference = _migration_text()
    try:
        with connection.cursor() as cursor:
            before = _preserved_hashes(cursor)
            cursor.execute(migration)

        recovered = store.get_round(ROUND_ID)
        assert (
            recovered["status"], recovered["status_generation"],
            recovered["stage_generation"], recovered["cancel_reason"],
        ) == ("stage1_scoring", 8, 7, None)
        assert recovered["configuration_doc"]["scorer_image_digest"] == new_digest
        assert recovered["configuration_doc"]["scorer_image_reference"] == new_reference
        assert recovered["configuration_doc"]["schedule"][
            "stage_1_scoring_close"
        ] == "2026-09-13T11:00:01Z"

        pending = [
            run for run in store.list_runs(ROUND_ID, stage=1, kind="score")
            if run["status"] == "pending"
        ]
        assert len(pending) == 84
        assert all(
            run["assignment_id"].endswith(":recovery234:recovery235")
            for run in pending
        )
        for run in pending:
            scope = dict(run["judgment_scope_doc"])
            cache_key = scope.pop("cache_key")
            assert cache_key == run["judgment_cache_key"]
            assert cache_key == contracts.document_hash(scope)
            assert scope["scorer_image_digest"] == new_digest
            assert scope["scorer_image_reference"] == new_reference

        with connection.cursor() as cursor:
            assert _preserved_hashes(cursor) == before
            _extend_test_deadline(cursor)

        stale_checked = False
        completed_leaders = 0
        while True:
            response, token, _, _ = claim(
                store, ROUND_ID, JUDGE_RUNNER, parallelism=100, ceiling=100,
                excluded=[JUDGE_RUNNER],
            )
            if response["status"] != "leased":
                assert response["status"] == "no_pending"
                break
            run = store.get_run(response["run_id"])
            if not stale_checked:
                stale = store.complete_attempt(
                    run_id=run["run_id"],
                    lease_token_hash=hash_lease_token("recovery234-old-lease"),
                    result={"terminal_status": "accepted"},
                    terminal_cause="accepted",
                    output_ref="stale",
                )
                assert stale["status"] == "stale"
                stale_checked = True
            output = {
                "schema_version": "leadpoet.lab_arena.scoring_output.v1",
                "scored_run_id": run["scored_run_id"],
                "breakdowns": [],
            }
            evidence = judgment_cache.build_evidence_snapshot(
                output=output,
                cache_scope=run["judgment_scope_doc"],
                source_score_run_id=run["run_id"],
                source_scored_run_id=run["scored_run_id"],
                source_output_ref=f"arena/{ROUND_ID}/retry/{run['run_id']}.json",
                source_runner_hotkey=JUDGE_RUNNER,
                runner_authority_exclusions=response[
                    "runner_authority_exclusions"
                ],
            )
            result = store.complete_attempt(
                run_id=run["run_id"],
                lease_token_hash=hash_lease_token(token),
                result={"terminal_status": "accepted"},
                terminal_cause="accepted",
                output_ref=evidence["source_output_ref"],
                judgment_evidence=evidence,
                judgment_evidence_hash=contracts.document_hash(evidence),
            )
            assert result["status"] == "accepted"
            completed_leaders += 1
        assert stale_checked
        assert completed_leaders == 10
        scores = store.list_runs(ROUND_ID, stage=1, kind="score")
        assert sum(run["status"] == "accepted" for run in scores) == 110
        closed = store.close_scoring(ROUND_ID, 1)
        assert closed["round_status"] == "stage1_judged"
        assert closed["incomplete_assignments"] == 0

        before_replay = store.get_round(ROUND_ID)
        with connection.cursor() as cursor:
            cursor.execute(migration)
        assert store.get_round(ROUND_ID) == before_replay
    finally:
        connection.close()
        transport.close()


def test_second_recovery_rejects_scope_drift_atomically(database):
    connection, _store, transport = _prepare_second_cancellation(database)
    migration, _digest, _reference = _migration_text()
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                "UPDATE public.lab_arena_runs SET judgment_scope_doc="
                "judgment_scope_doc || '{\"unexpected\":true}'::jsonb "
                "WHERE run_id=(SELECT run_id FROM public.lab_arena_runs "
                "WHERE round_id=%s AND stage=1 AND kind='score' "
                "AND status='failed' ORDER BY run_id LIMIT 1)",
                (ROUND_ID,),
            )
            before = _row_hash(
                cursor, "lab_arena_runs", "round_id=%s", (ROUND_ID,)
            )
        with pytest.raises(database[0].Error, match="judgment scope differs"):
            with connection.cursor() as cursor:
                cursor.execute(migration)
        with connection.cursor() as cursor:
            cursor.execute("ROLLBACK")
            assert _row_hash(
                cursor, "lab_arena_runs", "round_id=%s", (ROUND_ID,)
            ) == before
            cursor.execute(
                "SELECT status,status_generation,stage_generation "
                "FROM public.lab_arena_rounds WHERE round_id=%s",
                (ROUND_ID,),
            )
            assert cursor.fetchone() == ("cancelled", 7, 6)
    finally:
        connection.close()
        transport.close()
