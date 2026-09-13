"""Bind a corrected scorer only after September 13 stage 1 is quiescent."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from lab_arena import contracts
from lab_arena.service import ArenaService
from tests.lab_arena.arena_20260912_recovery_postgres_test import _row_hash
from tests.lab_arena.arena_20260913_quota_retry_postgres_test import (
    MIGRATION as QUOTA_RETRY,
    RETRY_RUN_ID,
    _complete_score,
    _prepare_quota_failure,
)
from tests.lab_arena.arena_20260913_scoring_recovery_postgres_test import (
    JUDGE_RUNNER,
    NEW_DIGEST as FIRST_RECOVERY_DIGEST,
    OLD_DIGEST as INITIAL_DIGEST,
    PARTICIPANT_IDS,
    ROUND_ID,
    RUNNER,
    UNRELATED_ROUND_ID,
)
from tests.lab_arena.arena_20260913_second_scoring_recovery_postgres_test import (
    _preserved_hashes,
)
from tests.lab_arena.cost_eligibility_test import _company
from tests.lab_arena.judgment_cache_test import _icp
from tests.lab_arena.lab_arena_pg_harness import (
    POSTGREST_MIGRATIONS,
    database_with_lab_arena_migration,
)
from tests.lab_arena.test_lab_arena_migration_postgres import claim


ROOT = Path(__file__).resolve().parents[2]
MIGRATION = (
    ROOT / "scripts" / "238-bind-arena-2026-09-13-future-stage-scorer.sql"
)
CURRENT_DIGEST = (
    "sha256:67b3ed3c8691a321007da8cc8353a4bb89b517813c7e017fcb93f55e468dd536"
)
NEW_DIGEST = (
    "sha256:188fe7f79e0233c4213bd6262f0d47e07f83d14b3374ee502f7268433111ba4e"
)
NEW_REFERENCE = (
    "493765492819.dkr.ecr.us-east-1.amazonaws.com/leadpoet/"
    f"sourcing-model@{NEW_DIGEST}"
)


@pytest.fixture(scope="module")
def database():
    yield from database_with_lab_arena_migration(POSTGREST_MIGRATIONS)


def _migration_text() -> str:
    text = MIGRATION.read_text(encoding="utf-8")
    assert NEW_DIGEST in text
    assert NEW_REFERENCE in text
    return text


def _complete_and_close_stage1(database):
    connection, store, transport = _prepare_quota_failure(database)
    with connection.cursor() as cursor:
        cursor.execute(QUOTA_RETRY.read_text(encoding="utf-8"))

    # The target retry excludes the first judge. Complete one normal group
    # first so per-submission serialization cannot hide the target retry.
    response, token, _, _ = claim(
        store, ROUND_ID, JUDGE_RUNNER, parallelism=100, ceiling=100,
        excluded=[JUDGE_RUNNER],
    )
    assert response["status"] == "leased"
    _complete_score(store, response, token, JUDGE_RUNNER)

    response, token, _, _ = claim(
        store, ROUND_ID, RUNNER, parallelism=100, ceiling=100,
        excluded=[RUNNER],
    )
    assert response["run_id"] == RETRY_RUN_ID
    _complete_score(store, response, token, RUNNER)

    leaders = 2
    while True:
        response, token, _, _ = claim(
            store, ROUND_ID, JUDGE_RUNNER, parallelism=100, ceiling=100,
            excluded=[JUDGE_RUNNER],
        )
        if response["status"] == "no_pending":
            break
        assert response["status"] == "leased"
        _complete_score(store, response, token, JUDGE_RUNNER)
        leaders += 1
    assert leaders == 11
    # The older recovery fixture intentionally populated only a sample of its
    # already-accepted cache rows. Production has every accepted cache object;
    # complete that shape before testing the boundary migration.
    with connection.cursor() as cursor:
        cursor.execute(
            "SELECT score.run_id,score.scored_run_id,score.runner_hotkey,"
            "score.judgment_cache_key,score.judgment_input_hash,"
            "score.judgment_scope_doc FROM public.lab_arena_runs AS score "
            "LEFT JOIN public.lab_arena_judgment_cache AS cache "
            "ON cache.cache_key=score.judgment_cache_key "
            "WHERE score.round_id=%s AND score.stage=1 AND score.kind='score' "
            "AND score.status='accepted' AND cache.cache_key IS NULL",
            (ROUND_ID,),
        )
        for run_id, scored_run_id, runner, key, input_hash, scope in cursor.fetchall():
            evidence = {
                "cache_key": key,
                "scoring_input_hash": input_hash,
                "source_score_run_id": run_id,
                "source_scored_run_id": scored_run_id,
                "source_runner_hotkey": runner,
                "runner_authority_exclusions": [runner],
            }
            cursor.execute(
                "INSERT INTO public.lab_arena_judgment_cache (cache_key,"
                "scope_doc,scoring_input_hash,evidence_hash,evidence_doc,"
                "source_score_run_id,source_scored_run_id,"
                "source_runner_hotkey) VALUES "
                "(%s,%s::jsonb,%s,%s,%s::jsonb,%s,%s,%s)",
                (
                    key, json.dumps(scope), input_hash,
                    contracts.document_hash(evidence), json.dumps(evidence),
                    run_id, scored_run_id, runner,
                ),
            )
    closed = store.close_scoring(ROUND_ID, 1)
    assert closed["round_status"] == "stage1_judged"
    assert closed["incomplete_assignments"] == 0
    return connection, store, transport


def _all_preserved(cursor):
    return {
        **_preserved_hashes(cursor),
        "all_runs": _row_hash(
            cursor, "lab_arena_runs", "round_id=%s", (ROUND_ID,),
        ),
    }


def test_future_image_update_preserves_stage1_and_binds_stage2_scope(
    database,
):
    connection, store, transport = _complete_and_close_stage1(database)
    migration = _migration_text()
    try:
        with connection.cursor() as cursor:
            before = _all_preserved(cursor)
            cursor.execute(migration)
            assert _all_preserved(cursor) == before

            cursor.execute(
                "SELECT judgment_scope_doc->>'scorer_image_digest',count(*) "
                "FROM public.lab_arena_runs WHERE round_id=%s AND stage=1 "
                "AND kind='score' AND status='accepted' GROUP BY 1",
                (ROUND_ID,),
            )
            assert dict(cursor.fetchall()) == {
                INITIAL_DIGEST: 22,
                FIRST_RECOVERY_DIGEST: 4,
                CURRENT_DIGEST: 84,
            }

        round_row = store.get_round(ROUND_ID)
        assert round_row["status"] == "stage1_judged"
        assert round_row["configuration_doc"]["scorer_image_digest"] == NEW_DIGEST
        assert (
            round_row["configuration_doc"]["scorer_image_reference"]
            == NEW_REFERENCE
        )

        captured = {}
        execution = {
            "run_id": f"{ROUND_ID}:stage2-execution",
            "submission_id": PARTICIPANT_IDS[0],
            "miner_hotkey": RUNNER,
            "icp_position": 10,
            "output_ref": f"arena/{ROUND_ID}/stage2-output.json",
            "status": "accepted",
            "kind": "execute",
        }
        plan = {
            "schema_version": "leadpoet.lab_arena.scoring_plan.v1",
            "round_id": ROUND_ID,
            "stage": 2,
            "zero_rows": [],
            "work_items": [{
                "submission_id": execution["submission_id"],
                "icp_position": 10,
                "scored_run_id": execution["run_id"],
                "output_ref": execution["output_ref"],
            }],
        }

        class ScoringStore:
            def list_runs(self, *_args, **_kwargs):
                return [execution]

            def get_judgment_cache(self, _key):
                return None

            def open_scoring(self, _round_id, _stage, items, **kwargs):
                captured["items"] = items
                captured["kwargs"] = kwargs
                return {"status": "ok", "round_status": "stage2_scoring",
                        "assignments": len(items)}

        service = object.__new__(ArenaService)
        service._store = ScoringStore()
        service._objects = SimpleNamespace(
            get_bounded=lambda _ref, _limit: contracts.canonical_json({
                "schema_version": contracts.OUTPUT_DOCUMENT_SCHEMA_VERSION,
                "companies": [_company("stage2.example", "Stage Two")],
            }).encode("utf-8")
        )
        stage2_round = {
            **round_row,
            "status": "stage2_closed",
            "arena_network_name": "finney",
            "arena_netuid": 71,
            "participants": [{
                "submission_id": execution["submission_id"],
                "miner_hotkey": RUNNER,
                "is_king": True,
            }],
        }
        service._round = lambda _round_id: stage2_round
        service._load_scoring_plan = lambda _round, _stage: plan
        service._require_code_review = lambda *_args: None
        service.evaluation_icps = lambda _round_id: [_icp()] * 30
        result = service.open_scoring(ROUND_ID, 2)
        assert result["status"] == "ok"
        scope = captured["items"][0]["judgment_scope_doc"]
        assert scope["scorer_image_digest"] == NEW_DIGEST
        assert scope["scorer_image_reference"] == NEW_REFERENCE
        assert captured["kwargs"] == {"integrity_cache": True}

        before_replay = store.get_round(ROUND_ID)
        with connection.cursor() as cursor:
            cursor.execute(migration)
        assert store.get_round(ROUND_ID) == before_replay
    finally:
        connection.close()
        transport.close()


def test_future_image_update_rejects_stage1_evidence_drift(database):
    connection, store, transport = _complete_and_close_stage1(database)
    migration = _migration_text()
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                "ALTER TABLE public.lab_arena_runs DISABLE TRIGGER "
                "lab_arena_integrity_run_guard"
            )
            cursor.execute(
                "UPDATE public.lab_arena_runs SET judgment_scope_doc="
                "judgment_scope_doc || '{\"unexpected\":true}'::jsonb "
                "WHERE run_id=(SELECT run_id FROM public.lab_arena_runs "
                "WHERE round_id=%s AND stage=1 AND kind='score' "
                "AND status='accepted' ORDER BY run_id LIMIT 1)",
                (ROUND_ID,),
            )
            cursor.execute(
                "ALTER TABLE public.lab_arena_runs ENABLE TRIGGER "
                "lab_arena_integrity_run_guard"
            )
            before = _row_hash(
                cursor, "lab_arena_rounds", "round_id=%s", (ROUND_ID,),
            )
        with pytest.raises(database[0].Error, match="future scorer state differs"):
            with connection.cursor() as cursor:
                cursor.execute(migration)
        with connection.cursor() as cursor:
            cursor.execute("ROLLBACK")
            assert _row_hash(
                cursor, "lab_arena_rounds", "round_id=%s", (ROUND_ID,),
            ) == before
            assert store.get_round(ROUND_ID)["configuration_doc"][
                "scorer_image_digest"
            ] == CURRENT_DIGEST
    finally:
        connection.close()
        transport.close()


def test_future_image_update_accepts_stage1_scored_boundary(database):
    connection, store, transport = _complete_and_close_stage1(database)
    migration = _migration_text()
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT participants FROM public.lab_arena_rounds "
                "WHERE round_id=%s", (ROUND_ID,),
            )
            participants = cursor.fetchone()[0]
            finalists = [
                row["submission_id"] for row in participants
                if not row.get("is_king")
            ][:10]
            assert len(finalists) == 10
            cursor.execute(
                "ALTER TABLE public.lab_arena_runs DISABLE TRIGGER "
                "lab_arena_integrity_run_guard"
            )
            cursor.execute(
                "UPDATE public.lab_arena_runs SET per_icp_score=1 "
                "WHERE round_id=%s AND stage=1 AND kind='execute' "
                "AND status='accepted'", (ROUND_ID,),
            )
            assert cursor.rowcount == 110
            cursor.execute(
                "ALTER TABLE public.lab_arena_runs ENABLE TRIGGER "
                "lab_arena_integrity_run_guard"
            )
            cursor.execute(
                "ALTER TABLE public.lab_arena_rounds DISABLE TRIGGER "
                "lab_arena_rounds_write_once"
            )
            cursor.execute(
                "UPDATE public.lab_arena_rounds SET status='stage1_scored',"
                "status_generation=10,finalists=%s::jsonb WHERE round_id=%s",
                (json.dumps(finalists), ROUND_ID),
            )
            cursor.execute(
                "ALTER TABLE public.lab_arena_rounds ENABLE TRIGGER "
                "lab_arena_rounds_write_once"
            )
            before_runs = _row_hash(
                cursor, "lab_arena_runs", "round_id=%s", (ROUND_ID,),
            )
            cursor.execute(migration)
            assert _row_hash(
                cursor, "lab_arena_runs", "round_id=%s", (ROUND_ID,),
            ) == before_runs
        round_row = store.get_round(ROUND_ID)
        assert round_row["status"] == "stage1_scored"
        assert round_row["status_generation"] == 10
        assert round_row["configuration_doc"]["scorer_image_digest"] == NEW_DIGEST
    finally:
        connection.close()
        transport.close()
