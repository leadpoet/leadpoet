"""Production-shaped recovery of arena-2026-09-13 stage-1 scoring."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lab_arena import contracts, judgment_cache, scoring
from lab_arena.owner_admission import OwnerAdmission
from lab_arena.store import ArenaStore, PsycopgTransport, hash_lease_token
from tests.lab_arena.arena_20260912_recovery_postgres_test import _row_hash
from tests.lab_arena.lab_arena_pg_harness import (
    POSTGREST_MIGRATIONS,
    database_with_lab_arena_migration,
)
from tests.lab_arena.test_lab_arena_migration_postgres import (
    claim,
    encrypted_runtime_credentials,
    hotkey,
    pass_code_review,
    round_config,
    sha,
    source_submission_doc,
)


ROOT = Path(__file__).resolve().parents[2]
MIGRATION = ROOT / "scripts" / "234-recover-arena-2026-09-13-scoring.sql"
ROUND_ID = "arena-2026-09-13"
UNRELATED_ROUND_ID = "arena-2026-09-14"
RUNNER = "5FNVgRnrxMibhcBGEAaajGrYjsaCn441a5HuGUBUNnxEBLo9"
JUDGE_RUNNER = hotkey("arena-2026-09-13-recovery-judge")
OLD_DIGEST = "sha256:b6e58855c4962faa345bd785e0d9aff3210457f5cf2d1dd3b4ee8e29f0b69757"
NEW_DIGEST = "sha256:f1812fbedf8700fd491fea94c576917018cc6fc1970a124784e22359eabaee4d"
OLD_REFERENCE = (
    "493765492819.dkr.ecr.us-east-1.amazonaws.com/leadpoet/"
    f"sourcing-model@{OLD_DIGEST}"
)
NEW_REFERENCE = (
    "493765492819.dkr.ecr.us-east-1.amazonaws.com/leadpoet/"
    f"sourcing-model@{NEW_DIGEST}"
)
FAILED_CREDENTIAL_SUBMISSION = "sub-caf0e1ef30c9712e6385afe24a75375e"
ASPIRE_SUBMISSION = "sub-00ee4222188195476cdcf63fdb94171f"
PARTICIPANTS = (
    ("sub-caf0e1ef30c9712e6385afe24a75375e", "5CDbB7NHrppEHM4QRtoYfRpzKURVKD2mc1F1qqTYU7TTXv4M"),
    ("sub-028c4e5c655e855f1c343fa39274cd2c", "5EvjcxLMtDMuAHv9gFYXtqZ9Qj69km7N7HuhheUEvLjchRa6"),
    ("sub-6211e8d46819c34df3418ded36f788ef", "5DZrBs1WQzpopHuMjtP74vbYDCKQhDUVtLWLAuYAyGnM3AFh"),
    ("sub-cb17cc2f2f351c1e92a59b85e35333c3", "5GuQa8NMGoNv4ATVqKw3Si9Lhvbm7YeELARJH49WWua7xpbT"),
    ("sub-5d27bd75ca8c5999ae22f28a0a244660", "5HjDZmJfnKRwQ4X7K8EFZAheyEWq88Wo5NCwetqNbqTPB9fc"),
    ("sub-1728be05a36000fcdb781a11d35594cd", "5FtFPnXCEnc1VQuRz3NCsAjcE3LMe3p63Wpxtr27JFgx78Fh"),
    ("sub-ca0c9c5da2d2b203f246258213c27263", "5FdxJvnjZCpBTPSoFyKE7BYPq9Z1JVm6SQQ1aYcVwHD9JeNa"),
    ("sub-5dffdbaa2b96e8dc78160aea8f80a7b9", "5Hb5Sxe46cp1XjwG23FX4BcqWSmgrewZLATBEX3LwreEFfC5"),
    ("sub-5c1f20eb379bcd4e11c29912b5251b0d", "5H956XR9zjVxgzgPAbamfNrWboXTMp9iLDu7LKrBW4VYKF8r"),
    (ASPIRE_SUBMISSION, "5GsukxFS2cpNr913xjqkAADDAFNC3FhyQBpSqF3yxkjzkSok"),
    ("sub-5ff557c97e83dcddd7b266226e16239c", "5Hp6fhYwgD1HEZJbL2CVbD4GQ8Y3SncFWWLDNQVShxaZNjWk"),
    ("baseline-2026-09-13", RUNNER),
)
PARTICIPANT_IDS = tuple(sorted(item[0] for item in PARTICIPANTS))
ACCEPTED_SCORES = {
    ("baseline-2026-09-13", 0),
    ("baseline-2026-09-13", 1),
    ("baseline-2026-09-13", 2),
    ("sub-028c4e5c655e855f1c343fa39274cd2c", 0),
    ("sub-028c4e5c655e855f1c343fa39274cd2c", 2),
    ("sub-1728be05a36000fcdb781a11d35594cd", 0),
    ("sub-1728be05a36000fcdb781a11d35594cd", 1),
    ("sub-1728be05a36000fcdb781a11d35594cd", 2),
    ("sub-5d27bd75ca8c5999ae22f28a0a244660", 0),
    ("sub-5d27bd75ca8c5999ae22f28a0a244660", 1),
    ("sub-5d27bd75ca8c5999ae22f28a0a244660", 2),
    ("sub-5dffdbaa2b96e8dc78160aea8f80a7b9", 0),
    ("sub-5dffdbaa2b96e8dc78160aea8f80a7b9", 1),
    ("sub-5ff557c97e83dcddd7b266226e16239c", 0),
    ("sub-5ff557c97e83dcddd7b266226e16239c", 1),
    ("sub-5ff557c97e83dcddd7b266226e16239c", 2),
    ("sub-6211e8d46819c34df3418ded36f788ef", 0),
    ("sub-6211e8d46819c34df3418ded36f788ef", 1),
    ("sub-ca0c9c5da2d2b203f246258213c27263", 0),
    ("sub-ca0c9c5da2d2b203f246258213c27263", 1),
    ("sub-cb17cc2f2f351c1e92a59b85e35333c3", 0),
    ("sub-cb17cc2f2f351c1e92a59b85e35333c3", 1),
}
RETRIED_SCORES = {
    (ASPIRE_SUBMISSION, 0): ("judge_error", "stage_closed"),
    (ASPIRE_SUBMISSION, 1): ("judge_error", "judge_error"),
    ("sub-028c4e5c655e855f1c343fa39274cd2c", 1): (
        "judge_error", "stage_closed"
    ),
    ("sub-5c1f20eb379bcd4e11c29912b5251b0d", 1): (
        "judge_error", "stage_closed"
    ),
    ("sub-1728be05a36000fcdb781a11d35594cd", 1): (
        "judge_error", "accepted"
    ),
    ("sub-5dffdbaa2b96e8dc78160aea8f80a7b9", 0): (
        "judge_error", "accepted"
    ),
}
EXECUTION_RETRIES = {
    ("sub-5c1f20eb379bcd4e11c29912b5251b0d", 9),
    ("sub-5d27bd75ca8c5999ae22f28a0a244660", 9),
}


@pytest.fixture(scope="module")
def database():
    yield from database_with_lab_arena_migration(POSTGREST_MIGRATIONS)


def _connect(database):
    psycopg2, dsn = database
    return psycopg2.connect(**dsn)


def _scope(position: int, digest: str, reference: str, *, marker: str | None = None):
    body = {
        "schema_version": "leadpoet.lab_arena.judgment_cache_scope.v1",
        "integrity_policy": "arena_integrity_v1",
        "round_id": ROUND_ID,
        "network_name": "finney",
        "netuid": 71,
        "scorer_image_digest": digest,
        "scorer_image_reference": reference,
        "evaluation_date": "2026-09-13",
        "scoring_input_hash": sha(f"scoring-input-{marker or position}"),
    }
    return {**body, "cache_key": contracts.document_hash(body)}


def _insert_score_run(cursor, submission_id, miner, position, attempt, cause,
                      scored_run_id, scope):
    assignment = f"{ROUND_ID}:{submission_id}:1:{position}:score"
    status = "accepted" if cause == "accepted" else "failed"
    cursor.execute(
        "INSERT INTO public.lab_arena_runs (run_id,assignment_id,round_id,"
        "submission_id,miner_hotkey,stage,icp_position,attempt,kind,"
        "scored_run_id,status,runner_hotkey,previous_runner_hotkey,"
        "stage_generation,lease_generation,lease_token_hash,result_doc,"
        "output_ref,terminal_cause,terminal_doc,judgment_cache_key,"
        "judgment_input_hash,judgment_scope_doc,judgment_group_leader,"
        "judgment_group_miner_hotkeys,judgment_cache_source_run_id) VALUES "
        "(%s,%s,%s,%s,%s,1,%s,%s,'score',%s,%s,%s,%s,3,%s,%s,%s::jsonb,"
        "%s,%s,%s::jsonb,%s,%s,%s::jsonb,TRUE,%s,%s)",
        (
            f"{assignment}:{attempt}", assignment, ROUND_ID, submission_id,
            miner, position, attempt, scored_run_id, status,
            RUNNER if cause != "stage_closed" else None,
            RUNNER if attempt == 2 else None, attempt,
            hash_lease_token("old-aspire-lease")
            if submission_id == ASPIRE_SUBMISSION and position == 1
            and attempt == 2 else None,
            json.dumps({"terminal_status": cause, "fixture": assignment}),
            f"arena/{ROUND_ID}/scores/items/{assignment}:{attempt}.json"
            if status == "accepted" else None,
            cause,
            json.dumps({"failure": cause}), scope["cache_key"],
            scope["scoring_input_hash"], json.dumps(scope), [miner],
            f"{assignment}:{attempt}" if status == "accepted" else None,
        ),
    )


def _seed(database):
    connection = _connect(database)
    connection.autocommit = True
    transport = PsycopgTransport(lambda: _connect(database))
    store = ArenaStore(transport, lease_ttl_seconds=120)
    with connection.cursor() as cursor:
        cursor.execute(
            "TRUNCATE public.lab_arena_ledger,public.lab_arena_rounds "
            "RESTART IDENTITY CASCADE"
        )
    assert store.create_round(
        UNRELATED_ROUND_ID,
        round_config(UNRELATED_ROUND_ID, [JUDGE_RUNNER]),
    )["status"] == "created"
    config = round_config(
        ROUND_ID, [RUNNER, JUDGE_RUNNER], stage_1_icps=10, max_attempts=2,
        execution_cap_microusd=80_000_000,
        scoring_cap_microusd=50_000_000, rewards_enabled=True,
        cost_per_company_microusd=800_000,
    )
    config.update({
        "integrity_policy": "arena_integrity_v1",
        "scorer_policy": scoring.build_scorer_policy(
            scoring_adapter_version="qualification_integrity_v2"
        ),
        "scorer_image_digest": OLD_DIGEST,
        "scorer_image_reference": OLD_REFERENCE,
        "baseline_hotkey": RUNNER,
    })
    assert store.create_round(ROUND_ID, config)["status"] == "created"
    assert store.prepare_confirmation_bank(
        ROUND_ID,
        f"arena/{ROUND_ID}/confirmation/{'c' * 64}.json",
        "sha256:" + "c" * 64,
    )["status"] == "ok"
    participants = []
    for submission_id, miner in PARTICIPANTS:
        is_king = submission_id == "baseline-2026-09-13"
        owner_admission = None
        if not is_king:
            owner_admission = OwnerAdmission(
                hotkey(f"owner-{submission_id}"),
                123,
                "0x" + "d" * 64,
            )
        assert store.register_submission(
            ROUND_ID, submission_id, miner,
            source_submission_doc(ROUND_ID, submission_id, is_king=is_king),
            owner_admission=owner_admission,
        )["status"] == "registered"
        if is_king:
            assert store.update_submission(
                ROUND_ID, submission_id, "uploading", "accepted",
                {"is_king": True},
            )["status"] == "ok"
        else:
            credentials = encrypted_runtime_credentials(submission_id)
            if submission_id in {PARTICIPANTS[0][0], PARTICIPANTS[1][0]}:
                credentials["scrapingdog"] = credentials["deepline"]
            assert store.accept_submission_with_credentials(
                ROUND_ID, submission_id, miner, credentials,
            )["status"] == "ok"
        if not is_king:
            pass_code_review(store, submission_id, miner)
        assert store.update_submission(
            ROUND_ID, submission_id, "accepted", "frozen",
            {"is_king": is_king},
        )["status"] == "ok"
        participants.append({
            "submission_id": submission_id,
            "miner_hotkey": miner,
            "is_king": is_king,
        })
    historical_schedule = {
        "submission_open": "2026-09-12T00:00:00Z",
        "submission_cutoff": "2026-09-13T00:00:00Z",
        "stage_1_scoring_close": "2026-09-13T11:00:01Z",
        "stage_2_start": "2026-09-13T11:00:02Z",
    }
    with connection.cursor() as cursor:
        cursor.execute(
            "ALTER TABLE public.lab_arena_rounds DISABLE TRIGGER "
            "lab_arena_rounds_write_once"
        )
        cursor.execute(
            "UPDATE public.lab_arena_rounds SET configuration_doc="
            "jsonb_set(configuration_doc,'{schedule}',%s::jsonb,FALSE) "
            "WHERE round_id=%s",
            (json.dumps(historical_schedule), ROUND_ID),
        )
        cursor.execute(
            "ALTER TABLE public.lab_arena_rounds ENABLE TRIGGER "
            "lab_arena_rounds_write_once"
        )
    assert store.transition_round(
        ROUND_ID, "open", "committed", {
            "participants": participants,
            "benchmark_ref": f"arena/{ROUND_ID}/benchmark.json",
            "evaluation_date": "2026-09-13",
        },
    )["status"] == "ok"
    assert store.open_stage(ROUND_ID, 1, participants, range(10))[
        "assignments"
    ] == 120

    miners = dict(PARTICIPANTS)
    work_items = []
    zero_rows = []
    with connection.cursor() as cursor:
        cursor.execute(
            "ALTER TABLE public.lab_arena_runs DISABLE TRIGGER "
            "lab_arena_runs_terminal"
        )
        for submission_id, miner in PARTICIPANTS:
            for position in range(10):
                assignment = f"{ROUND_ID}:{submission_id}:1:{position}"
                run_id = f"{assignment}:1"
                if submission_id == FAILED_CREDENTIAL_SUBMISSION:
                    cursor.execute(
                        "UPDATE public.lab_arena_runs SET status='failed',"
                        "terminal_cause='credential_error',result_doc=%s::jsonb "
                        "WHERE run_id=%s",
                        (json.dumps({"terminal_status": "credential_error"}),
                         run_id),
                    )
                    zero_rows.append({
                        "submission_id": submission_id,
                        "icp_position": position,
                        "cause": "credential_error",
                    })
                    continue
                if (submission_id, position) in EXECUTION_RETRIES:
                    cursor.execute(
                        "UPDATE public.lab_arena_runs SET status='failed',"
                        "terminal_cause='lease_expired',result_doc=%s::jsonb "
                        "WHERE run_id=%s",
                        (json.dumps({"terminal_status": "lease_expired"}),
                         run_id),
                    )
                    run_id = f"{assignment}:2"
                    cursor.execute(
                        "INSERT INTO public.lab_arena_runs (run_id,assignment_id,"
                        "round_id,submission_id,miner_hotkey,stage,icp_position,"
                        "attempt,kind,status,runner_hotkey,previous_runner_hotkey,"
                        "stage_generation,lease_generation,result_doc,output_ref,"
                        "terminal_cause) VALUES (%s,%s,%s,%s,%s,1,%s,2,'execute',"
                        "'accepted',%s,%s,1,1,%s::jsonb,%s,'accepted')",
                        (run_id, assignment, ROUND_ID, submission_id, miner,
                         position, RUNNER, RUNNER,
                         json.dumps({"terminal_status": "accepted"}),
                         f"arena/{ROUND_ID}/outputs/{run_id}.json"),
                    )
                else:
                    cursor.execute(
                        "UPDATE public.lab_arena_runs SET status='accepted',"
                        "runner_hotkey=%s,lease_generation=1,result_doc=%s::jsonb,"
                        "output_ref=%s,terminal_cause='accepted' WHERE run_id=%s",
                        (RUNNER, json.dumps({"terminal_status": "accepted"}),
                         f"arena/{ROUND_ID}/outputs/{run_id}.json", run_id),
                    )
                work_items.append({
                    "output_ref": f"arena/{ROUND_ID}/outputs/{run_id}.json",
                    "icp_position": position,
                    "scored_run_id": run_id,
                    "submission_id": submission_id,
                })
        cursor.execute(
            "ALTER TABLE public.lab_arena_runs ENABLE TRIGGER "
            "lab_arena_runs_terminal"
        )

        cursor.execute(
            "ALTER TABLE public.lab_arena_rounds DISABLE TRIGGER "
            "lab_arena_rounds_write_once"
        )
        cursor.execute(
            "ALTER TABLE public.lab_arena_rounds DISABLE TRIGGER "
            "lab_arena_icp_set_date_write_once"
        )
        cursor.execute(
            "UPDATE public.lab_arena_rounds SET status='stage1_scoring',"
            "status_generation=4,stage_generation=3,evaluation_date='2026-09-13',"
            "icp_set_date='2026-09-12' WHERE round_id=%s", (ROUND_ID,),
        )
        cursor.execute(
            "ALTER TABLE public.lab_arena_rounds ENABLE TRIGGER "
            "lab_arena_icp_set_date_write_once"
        )
        cursor.execute(
            "ALTER TABLE public.lab_arena_rounds ENABLE TRIGGER "
            "lab_arena_rounds_write_once"
        )

        cursor.execute(
            "ALTER TABLE public.lab_arena_runs DISABLE TRIGGER "
            "lab_arena_runs_terminal"
        )
        for item in work_items:
            key = (item["submission_id"], item["icp_position"])
            marker = None
            if key in ACCEPTED_SCORES:
                marker = f"{item['submission_id']}-{item['icp_position']}"
            scope = _scope(
                item["icp_position"], OLD_DIGEST, OLD_REFERENCE, marker=marker
            )
            causes = RETRIED_SCORES.get(key)
            if causes is None:
                causes = (("accepted",) if key in ACCEPTED_SCORES
                          else ("stage_closed",))
            for attempt, cause in enumerate(causes, start=1):
                _insert_score_run(
                    cursor, item["submission_id"], miners[item["submission_id"]],
                    item["icp_position"], attempt, cause,
                    item["scored_run_id"], scope,
                )
        cursor.execute(
            "ALTER TABLE public.lab_arena_runs ENABLE TRIGGER "
            "lab_arena_runs_terminal"
        )

        cursor.execute(
            "SELECT run_id,scored_run_id,runner_hotkey,judgment_scope_doc,"
            "judgment_cache_key,judgment_input_hash FROM public.lab_arena_runs "
            "WHERE round_id=%s AND kind='score' AND status='accepted' "
            "ORDER BY run_id LIMIT 11", (ROUND_ID,),
        )
        for row in cursor.fetchall():
            run_id, scored_run_id, runner, scope, cache_key, input_hash = row
            evidence = {
                "cache_key": cache_key,
                "scoring_input_hash": input_hash,
                "source_score_run_id": run_id,
                "source_scored_run_id": scored_run_id,
                "source_runner_hotkey": runner,
                "runner_authority_exclusions": [runner],
            }
            cursor.execute(
                "INSERT INTO public.lab_arena_judgment_cache (cache_key,scope_doc,"
                "scoring_input_hash,evidence_hash,evidence_doc,source_score_run_id,"
                "source_scored_run_id,source_runner_hotkey) VALUES "
                "(%s,%s::jsonb,%s,%s,%s::jsonb,%s,%s,%s)",
                (cache_key, json.dumps(scope), input_hash,
                 contracts.document_hash(evidence), json.dumps(evidence),
                 run_id, scored_run_id, runner),
            )
        plan = {
            "stage": 1,
            "round_id": ROUND_ID,
            "zero_rows": zero_rows,
            "work_items": sorted(
                work_items,
                key=lambda item: (item["submission_id"], item["icp_position"]),
            ),
            "schema_version": "leadpoet.lab_arena.scoring_plan.v1",
        }
        cursor.execute(
            "INSERT INTO public.lab_arena_ledger (entry_kind,miner_hotkey,"
            "round_id,submission_id,run_id,stage,call_identity,provider,"
            "operation_id,funding_source,amount_microusd,entry_doc,"
            "terminal_response) SELECT 'settlement',miner_hotkey,round_id,"
            "submission_id,run_id,stage,%s,'openrouter','openrouter.chat',"
            "'miner_key',321,'{}'::jsonb,'{\"status\":200}'::jsonb FROM "
            "public.lab_arena_runs WHERE round_id=%s AND kind='score' "
            "AND status='accepted' ORDER BY run_id LIMIT 1",
            (sha("preserved-score-cost"), ROUND_ID),
        )
        cursor.execute(
            "ALTER TABLE public.lab_arena_rounds DISABLE TRIGGER "
            "lab_arena_rounds_write_once"
        )
        cursor.execute(
            "UPDATE public.lab_arena_rounds SET status='cancelled',"
            "status_generation=5,stage_generation=4,cancel_reason='scoring_incomplete',"
            "stage1_scoring_plan_doc=%s::jsonb WHERE round_id=%s",
            (json.dumps(plan), ROUND_ID),
        )
        cursor.execute(
            "ALTER TABLE public.lab_arena_rounds ENABLE TRIGGER "
            "lab_arena_rounds_write_once"
        )
    return connection, store, transport


def test_recovery_preserves_work_and_costs_then_completes(database):
    connection, store, transport = _seed(database)
    try:
        with connection.cursor() as cursor:
            before = {
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
                    cursor, "lab_arena_rounds", "round_id=%s",
                    (UNRELATED_ROUND_ID,),
                ),
            }
        with connection.cursor() as cursor:
            cursor.execute(MIGRATION.read_text(encoding="utf-8"))

        recovered = store.get_round(ROUND_ID)
        assert (
            recovered["status"], recovered["status_generation"],
            recovered["stage_generation"], recovered["cancel_reason"],
        ) == ("stage1_scoring", 6, 5, None)
        assert recovered["configuration_doc"]["scorer_image_digest"] == NEW_DIGEST
        assert recovered["configuration_doc"]["scorer_image_reference"] == NEW_REFERENCE
        assert recovered["configuration_doc"]["schedule"] == {
            "submission_open": "2026-09-12T00:00:00Z",
            "submission_cutoff": "2026-09-13T00:00:00Z",
            "stage_1_scoring_close": "2026-09-13T11:00:01Z",
            "stage_2_start": "2026-09-13T11:00:02Z",
        }

        pending = [
            run for run in store.list_runs(ROUND_ID, stage=1, kind="score")
            if run["status"] == "pending"
        ]
        assert len(pending) == 88
        assert all(run["assignment_id"].endswith(":recovery234") for run in pending)
        for run in pending:
            scope = dict(run["judgment_scope_doc"])
            cache_key = scope.pop("cache_key")
            assert cache_key == run["judgment_cache_key"]
            assert cache_key == contracts.document_hash(scope)
            assert scope["scorer_image_digest"] == NEW_DIGEST
            assert scope["scorer_image_reference"] == NEW_REFERENCE

        with connection.cursor() as cursor:
            assert _row_hash(cursor, "lab_arena_runs", "round_id=%s AND status='accepted'", (ROUND_ID,)) == before["accepted"]
            assert _row_hash(cursor, "lab_arena_runs", "round_id=%s AND kind<>'score'", (ROUND_ID,)) == before["non_score"]
            assert _row_hash(cursor, "lab_arena_submissions", "round_id=%s", (ROUND_ID,)) == before["submissions"]
            assert _row_hash(cursor, "lab_arena_submission_credentials", "submission_id=ANY(%s)", (list(PARTICIPANT_IDS),)) == before["credentials"]
            assert _row_hash(cursor, "lab_arena_ledger", "round_id=%s", (ROUND_ID,)) == before["ledger"]
            assert _row_hash(cursor, "lab_arena_judgment_cache", "scope_doc->>'round_id'=%s", (ROUND_ID,)) == before["cache"]
            assert _row_hash(cursor, "lab_arena_rounds", "round_id=%s", (UNRELATED_ROUND_ID,)) == before["unrelated_round"]

            # Keep this historical recovery regression deterministic after its
            # real 2026 deadline. The assertion above proves the migration did
            # not change the production schedule.
            cursor.execute(
                "ALTER TABLE public.lab_arena_rounds DISABLE TRIGGER "
                "lab_arena_rounds_write_once"
            )
            cursor.execute(
                "UPDATE public.lab_arena_rounds SET configuration_doc="
                "jsonb_set(configuration_doc,"
                "'{schedule,stage_1_scoring_close}',"
                "'\"2100-01-01T00:00:00Z\"'::jsonb,FALSE) "
                "WHERE round_id=%s",
                (ROUND_ID,),
            )
            cursor.execute(
                "ALTER TABLE public.lab_arena_rounds ENABLE TRIGGER "
                "lab_arena_rounds_write_once"
            )

        old_lease_hash = hash_lease_token("old-aspire-lease")
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
            assert run is not None
            if not stale_checked:
                stale = store.complete_attempt(
                    run_id=run["run_id"], lease_token_hash=old_lease_hash,
                    result={"terminal_status": "accepted"},
                    terminal_cause="accepted", output_ref="stale",
                )
                assert stale["status"] == "stale"
                stale_checked = True
            output = {
                "schema_version": "leadpoet.lab_arena.scoring_output.v1",
                "scored_run_id": run["scored_run_id"],
                "breakdowns": [],
            }
            scope = dict(run["judgment_scope_doc"])
            evidence = judgment_cache.build_evidence_snapshot(
                output=output,
                cache_scope=scope,
                source_score_run_id=run["run_id"],
                source_scored_run_id=run["scored_run_id"],
                source_output_ref=f"arena/{ROUND_ID}/recovered/{run['run_id']}.json",
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
        assert len([
            run for run in store.list_runs(ROUND_ID, stage=1, kind="score")
            if run["status"] == "accepted"
        ]) == 110
        closed = store.close_scoring(ROUND_ID, 1)
        assert closed["round_status"] == "stage1_judged"
        assert closed["incomplete_assignments"] == 0

        before_replay = store.get_round(ROUND_ID)
        with connection.cursor() as cursor:
            cursor.execute(MIGRATION.read_text(encoding="utf-8"))
        assert store.get_round(ROUND_ID) == before_replay
    finally:
        connection.close()
        transport.close()


def test_recovery_rejects_unexpected_judgment_scope_without_partial_writes(database):
    connection, _store, transport = _seed(database)
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
            runs_before = _row_hash(
                cursor, "lab_arena_runs", "round_id=%s", (ROUND_ID,)
            )
            round_before = _row_hash(
                cursor, "lab_arena_rounds", "round_id=%s", (ROUND_ID,)
            )
        with pytest.raises(database[0].Error, match="judgment scope differs"):
            with connection.cursor() as cursor:
                cursor.execute(MIGRATION.read_text(encoding="utf-8"))
        with connection.cursor() as cursor:
            cursor.execute("ROLLBACK")
            assert _row_hash(
                cursor, "lab_arena_runs", "round_id=%s", (ROUND_ID,)
            ) == runs_before
            assert _row_hash(
                cursor, "lab_arena_rounds", "round_id=%s", (ROUND_ID,)
            ) == round_before
    finally:
        connection.close()
        transport.close()
