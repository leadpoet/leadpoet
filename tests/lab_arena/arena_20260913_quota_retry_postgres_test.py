"""Guarded retry for one September 13 score with retained quota history."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lab_arena import contracts, judgment_cache
from lab_arena.store import hash_lease_token
from tests.lab_arena.arena_20260912_recovery_postgres_test import _row_hash
from tests.lab_arena.arena_20260913_scoring_recovery_postgres_test import (
    JUDGE_RUNNER,
    PARTICIPANT_IDS,
    ROUND_ID,
    RUNNER,
    UNRELATED_ROUND_ID,
)
from tests.lab_arena.arena_20260913_second_scoring_recovery_postgres_test import (
    MIGRATION as SECOND_RECOVERY,
    _extend_test_deadline,
    _prepare_second_cancellation,
)
from tests.lab_arena.lab_arena_pg_harness import (
    POSTGREST_MIGRATIONS,
    database_with_lab_arena_migration,
)
from tests.lab_arena.test_lab_arena_migration_postgres import claim, sha


ROOT = Path(__file__).resolve().parents[2]
MIGRATION = ROOT / "scripts" / "237-recover-arena-2026-09-13-quota-retry.sql"
SUBMISSION_ID = "sub-5c1f20eb379bcd4e11c29912b5251b0d"
MINER_HOTKEY = "5H956XR9zjVxgzgPAbamfNrWboXTMp9iLDu7LKrBW4VYKF8r"
ASSIGNMENT_ID = (
    f"{ROUND_ID}:{SUBMISSION_ID}:1:0:score:recovery234:recovery235"
)
FIRST_RUN_ID = f"{ROUND_ID}:{SUBMISSION_ID}:1:0:score:1"
RETRY_RUN_ID = f"{ASSIGNMENT_ID}:2"
SCORED_RUN_ID = f"{ROUND_ID}:{SUBMISSION_ID}:1:0:1"
OLD_LEASE_TOKEN = "quota-artifact-old-lease"


@pytest.fixture(scope="module")
def database():
    yield from database_with_lab_arena_migration(POSTGREST_MIGRATIONS)


def _prepare_quota_failure(database):
    connection, store, transport = _prepare_second_cancellation(database)
    with connection.cursor() as cursor:
        cursor.execute(SECOND_RECOVERY.read_text(encoding="utf-8"))
        _extend_test_deadline(cursor)
        cursor.execute(
            "SELECT judgment_scope_doc,judgment_cache_key "
            "FROM public.lab_arena_runs WHERE run_id=%s",
            (FIRST_RUN_ID,),
        )
        old_scope, old_cache_key = cursor.fetchone()
        scope_body = {
            **{key: value for key, value in old_scope.items() if key != "cache_key"},
            "scoring_input_hash": sha("quota-target-unique"),
        }
        scope = {**scope_body, "cache_key": contracts.document_hash(scope_body)}

        cursor.execute(
            "ALTER TABLE public.lab_arena_runs DISABLE TRIGGER "
            "lab_arena_runs_terminal"
        )
        cursor.execute(
            "ALTER TABLE public.lab_arena_runs DISABLE TRIGGER "
            "lab_arena_integrity_run_guard"
        )
        cursor.execute(
            "UPDATE public.lab_arena_runs SET status='failed',"
            "runner_hotkey=%s,previous_runner_hotkey=NULL,"
            "lease_token_hash=%s,lease_generation=5,"
            "result_doc=%s::jsonb,output_ref=NULL,"
            "terminal_cause='credential_error',terminal_doc=NULL,"
            "per_icp_score=NULL,claim_response=%s::jsonb,"
            "judgment_cache_key=%s,judgment_input_hash=%s,"
            "judgment_scope_doc=%s::jsonb,judgment_group_leader=TRUE,"
            "judgment_group_miner_hotkeys=%s,"
            "judgment_cache_source_run_id=NULL,company_judgment_refs=NULL "
            "WHERE run_id=%s AND status='pending'",
            (
                JUDGE_RUNNER,
                hash_lease_token(OLD_LEASE_TOKEN),
                json.dumps({
                    "schema_version": "leadpoet.lab_arena.run_result.v1",
                    "terminal_status": "credential_error",
                    "resource_summary": {"provider_call_count": 36},
                }),
                json.dumps({
                    "run_id": FIRST_RUN_ID,
                    "assignment_id": ASSIGNMENT_ID,
                    "runner_authority_exclusions": [JUDGE_RUNNER],
                }),
                scope["cache_key"],
                scope["scoring_input_hash"],
                json.dumps(scope),
                [MINER_HOTKEY],
                FIRST_RUN_ID,
            ),
        )
        assert cursor.rowcount == 1
        cursor.execute(
            "WITH remaining AS (SELECT pg_catalog.min(run_id) AS leader "
            "FROM public.lab_arena_runs WHERE round_id=%s AND stage=1 "
            "AND kind='score' AND judgment_cache_key=%s AND run_id<>%s) "
            "UPDATE public.lab_arena_runs AS run SET "
            "judgment_group_leader=(run.run_id=remaining.leader),"
            "judgment_group_miner_hotkeys="
            "pg_catalog.array_remove(run.judgment_group_miner_hotkeys,%s) "
            "FROM remaining WHERE run.round_id=%s AND run.stage=1 "
            "AND run.kind='score' AND run.judgment_cache_key=%s "
            "AND run.run_id<>%s",
            (
                ROUND_ID, old_cache_key, FIRST_RUN_ID, MINER_HOTKEY,
                ROUND_ID, old_cache_key, FIRST_RUN_ID,
            ),
        )
        cursor.execute(
            "ALTER TABLE public.lab_arena_runs ENABLE TRIGGER "
            "lab_arena_integrity_run_guard"
        )
        cursor.execute(
            "ALTER TABLE public.lab_arena_runs ENABLE TRIGGER "
            "lab_arena_runs_terminal"
        )

        cursor.execute(
            "SELECT pg_catalog.count(*) FROM public.lab_arena_ledger "
            "WHERE run_id=%s", (FIRST_RUN_ID,),
        )
        assert cursor.fetchone()[0] == 0
        for index in range(40):
            _insert_terminal_call(
                cursor, index, "deepline", "settlement", "exa.contents"
            )
        for index in range(41):
            _insert_terminal_call(
                cursor, 100 + index, "openrouter", "settlement",
                "openrouter.chat",
            )
        _insert_terminal_call(
            cursor, 200, "deepline", "refusal", "exa.contents",
            refusal=True,
        )
        _insert_terminal_call(
            cursor, 201, "deepline", "refusal", "scrapingdog.scrape",
            refusal=True,
        )
    return connection, store, transport


def _insert_terminal_call(cursor, index, provider, kind, operation, *, refusal=False):
    cursor.execute(
        "INSERT INTO public.lab_arena_ledger (entry_kind,miner_hotkey,round_id,"
        "submission_id,run_id,stage,call_identity,provider,operation_id,"
        "funding_source,amount_microusd,entry_doc,terminal_response) VALUES "
        "(%s,%s,%s,%s,%s,1,%s,%s,%s,'miner_key',%s,%s::jsonb,%s::jsonb)",
        (
            kind, MINER_HOTKEY, ROUND_ID, SUBMISSION_ID, FIRST_RUN_ID,
            sha(f"quota-ledger-{index}"), provider, operation,
            0 if refusal else 100 + index,
            json.dumps({
                "reason": "per_icp_quota",
                "prior_miner_credential_refusal": False,
            }) if refusal else "{}",
            "null" if refusal else json.dumps({
                "status": 200, "call_succeeded": True,
            }),
        ),
    )


def _preserved_hashes(cursor):
    return {
        "existing_runs": _row_hash(
            cursor, "lab_arena_runs",
            "round_id=%s AND run_id<>%s", (ROUND_ID, RETRY_RUN_ID),
        ),
        "round": _row_hash(
            cursor, "lab_arena_rounds", "round_id=%s", (ROUND_ID,),
        ),
        "unrelated_round": _row_hash(
            cursor, "lab_arena_rounds", "round_id=%s", (UNRELATED_ROUND_ID,),
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
    }


def _complete_score(store, response, token, runner):
    run = store.get_run(response["run_id"])
    output_ref = f"arena/{ROUND_ID}/quota-retry/{run['run_id']}.json"
    evidence = judgment_cache.build_evidence_snapshot(
        output={
            "schema_version": "leadpoet.lab_arena.scoring_output.v1",
            "scored_run_id": run["scored_run_id"],
            "breakdowns": [],
        },
        cache_scope=run["judgment_scope_doc"],
        source_score_run_id=run["run_id"],
        source_scored_run_id=run["scored_run_id"],
        source_output_ref=output_ref,
        source_runner_hotkey=runner,
        runner_authority_exclusions=response["runner_authority_exclusions"],
    )
    result = store.complete_attempt(
        run_id=run["run_id"],
        lease_token_hash=hash_lease_token(token),
        result={"terminal_status": "accepted"},
        terminal_cause="accepted",
        output_ref=output_ref,
        judgment_evidence=evidence,
        judgment_evidence_hash=contracts.document_hash(evidence),
    )
    assert result["status"] == "accepted"


def test_quota_retry_preserves_history_and_completes_all_assignments(database):
    connection, store, transport = _prepare_quota_failure(database)
    migration = MIGRATION.read_text(encoding="utf-8")
    try:
        active, active_token, _, _ = claim(
            store, ROUND_ID, JUDGE_RUNNER, parallelism=100, ceiling=100,
            excluded=[JUDGE_RUNNER],
        )
        assert active["status"] == "leased"
        assert active["run_id"] != FIRST_RUN_ID

        with connection.cursor() as cursor:
            before = _preserved_hashes(cursor)
            cursor.execute(migration)
            assert _preserved_hashes(cursor) == before

        retry = store.get_run(RETRY_RUN_ID)
        first = store.get_run(FIRST_RUN_ID)
        assert retry["status"] == "pending"
        assert retry["assignment_id"] == ASSIGNMENT_ID
        assert retry["attempt"] == 2
        assert retry["scored_run_id"] == SCORED_RUN_ID
        assert retry["previous_runner_hotkey"] == JUDGE_RUNNER
        assert retry["judgment_scope_doc"] == first["judgment_scope_doc"]
        assert retry["judgment_cache_key"] == first["judgment_cache_key"]

        assert store.get_run(active["run_id"])["status"] == "leased"
        _complete_score(store, active, active_token, JUDGE_RUNNER)

        response, token, _, _ = claim(
            store, ROUND_ID, RUNNER, parallelism=100, ceiling=100,
            excluded=[RUNNER],
        )
        assert response["status"] == "leased"
        assert response["run_id"] == RETRY_RUN_ID
        stale = store.complete_attempt(
            run_id=RETRY_RUN_ID,
            lease_token_hash=hash_lease_token(OLD_LEASE_TOKEN),
            result={"terminal_status": "accepted"},
            terminal_cause="accepted",
            output_ref="stale",
        )
        assert stale["status"] == "stale"
        _complete_score(store, response, token, RUNNER)

        completed_leaders = 2
        while True:
            response, token, _, _ = claim(
                store, ROUND_ID, JUDGE_RUNNER, parallelism=100, ceiling=100,
                excluded=[JUDGE_RUNNER],
            )
            if response["status"] == "no_pending":
                break
            assert response["status"] == "leased"
            _complete_score(store, response, token, JUDGE_RUNNER)
            completed_leaders += 1
        assert completed_leaders == 11

        scores = store.list_runs(ROUND_ID, stage=1, kind="score")
        accepted_assignments = {
            run["assignment_id"] for run in scores if run["status"] == "accepted"
        }
        assert len(accepted_assignments) == 110
        assert store.get_run(FIRST_RUN_ID)["terminal_cause"] == "credential_error"
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


def test_quota_retry_rejects_scope_drift_atomically(database):
    connection, _store, transport = _prepare_quota_failure(database)
    migration = MIGRATION.read_text(encoding="utf-8")
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                "ALTER TABLE public.lab_arena_runs DISABLE TRIGGER "
                "lab_arena_integrity_run_guard"
            )
            cursor.execute(
                "UPDATE public.lab_arena_runs SET judgment_scope_doc="
                "judgment_scope_doc || '{\"unexpected\":true}'::jsonb "
                "WHERE run_id=%s", (FIRST_RUN_ID,),
            )
            cursor.execute(
                "ALTER TABLE public.lab_arena_runs ENABLE TRIGGER "
                "lab_arena_integrity_run_guard"
            )
            before = _preserved_hashes(cursor)
        with pytest.raises(database[0].Error, match="quota retry state differs"):
            with connection.cursor() as cursor:
                cursor.execute(migration)
        with connection.cursor() as cursor:
            cursor.execute("ROLLBACK")
            assert _preserved_hashes(cursor) == before
            cursor.execute(
                "SELECT pg_catalog.count(*) FROM public.lab_arena_runs "
                "WHERE run_id=%s", (RETRY_RUN_ID,),
            )
            assert cursor.fetchone()[0] == 0
    finally:
        connection.close()
        transport.close()
