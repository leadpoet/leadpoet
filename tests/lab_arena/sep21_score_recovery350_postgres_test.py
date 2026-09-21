"""Real-PostgreSQL proof for the append-only Sep21 score recovery."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from lab_arena import contracts, judgment_cache, scoring
from lab_arena.store import ArenaStore, PsycopgTransport, hash_lease_token
from tests.lab_arena.lab_arena_pg_harness import (
    CURRENT_SERVICE_MIGRATIONS,
    database_with_lab_arena_migration,
)
from tests.lab_arena.test_lab_arena_migration_postgres import claim, hotkey


ROOT = Path(__file__).parents[2]
MIGRATION = ROOT / "scripts/350-arena-2026-09-21-unresolved-score-recovery.sql"
ROUND = "arena-2026-09-21"
OLD_DIGEST = "sha256:088d77919300b6cb210003862ebd5b25608369e8a22478e16bae8e69f6adc1af"
NEW_DIGEST = "sha256:33012bf556b6fe46263ecb80183a8d344017b8232f51e95e582ac1ccf67ef1c6"
RUNNER = hotkey("sep21-score-recovery-runner")


def _payload() -> list[dict]:
    match = re.search(
        r"v_targets CONSTANT JSONB := \$targets\$(.*?)\$targets\$::JSONB;",
        MIGRATION.read_text(encoding="utf-8"),
        re.DOTALL,
    )
    assert match is not None
    return json.loads(match.group(1))


TARGETS = _payload()


def _invalidations() -> list[dict]:
    match = re.search(
        r"v_invalidations CONSTANT JSONB := \$invalidations\$(.*?)\$invalidations\$::JSONB;",
        MIGRATION.read_text(encoding="utf-8"),
        re.DOTALL,
    )
    assert match is not None
    return json.loads(match.group(1))


INVALIDATIONS = _invalidations()
EXPECTED_INVALIDATION_HASHES = {
    "arena-2026-09-21:baseline-2026-09-21:1:7:score:2":
        "sha256:9ed7b1bd916e712535aecc3c2b2e83c77a5ae7b83204bce1d8b000246b08d098",
    "arena-2026-09-21:sub-3c31076fbb4d3e914495fdd5dc0f4799:1:2:score:1":
        "sha256:eec6cc9f586bde491ce9ce5f29202941560d4c5636504c0e328ac520b5390a5c",
    "arena-2026-09-21:sub-3c31076fbb4d3e914495fdd5dc0f4799:1:5:score:1":
        "sha256:35f846aab31d734b2043ec549ffba6af4f0eb2886a34b90a57c6df2eb07f2b74",
    "arena-2026-09-21:sub-3c31076fbb4d3e914495fdd5dc0f4799:1:8:score:1":
        "sha256:2c5b8305b38a256653b6e239b4b33406cce0e900ba3143c9f9827dae62fac84a",
}
assert {item["run_id"]: item["old_output_hash"] for item in INVALIDATIONS} == EXPECTED_INVALIDATION_HASHES
INVALIDATION_BY_SCORED = {item["scored_run_id"]: item for item in INVALIDATIONS}

PATRONUS_ACCEPTED = {
    "arena-2026-09-21:sub-335b9e187d44c6b5905538aaef21f4ef:1:2:score:1": (
        "arena/arena-2026-09-21/scores/items/arena-2026-09-21:sub-335b9e187d44c6b5905538aaef21f4ef:1:2:score:1.json",
        "sha256:67944c2c5ca5fb1a20c03257bd0a109b7d2a0aac0eaf487d924e3390c6f51ee6",
    ),
    "arena-2026-09-21:sub-a6c72a590f2014449fb6ca7f66e420f8:1:2:score:1": (
        "arena/arena-2026-09-21/scores/items/arena-2026-09-21:sub-a6c72a590f2014449fb6ca7f66e420f8:1:2:score:1.json",
        "sha256:9f7616b93be4ae92e41d95f367483a47da15d9665bcb6f6440f11297c9706526",
    ),
    "arena-2026-09-21:sub-c99d1357151933192f533043283d8e27:1:2:score:1": (
        "arena/arena-2026-09-21/scores/items/arena-2026-09-21:sub-a6c72a590f2014449fb6ca7f66e420f8:1:2:score:1.json",
        "sha256:9f7616b93be4ae92e41d95f367483a47da15d9665bcb6f6440f11297c9706526",
    ),
}


@pytest.fixture(scope="module")
def database():
    yield from database_with_lab_arena_migration(
        CURRENT_SERVICE_MIGRATIONS
        + ("349-lab-arena-scoring-effective-outcome.sql",)
    )


def _participants() -> list[dict]:
    identities = {}
    for item in TARGETS:
        identities[item["submission_id"]] = item["miner_hotkey"]
    assert len(identities) == 6
    return [
        {
            "submission_id": submission_id,
            "miner_hotkey": miner,
            "is_king": submission_id == "baseline-2026-09-21",
            "source_ref": f"arena/{ROUND}/sources/{submission_id}.tar.gz",
            "source_size_bytes": 100_000 + index,
        }
        for index, (submission_id, miner) in enumerate(sorted(identities.items()))
    ]


def _configuration() -> dict:
    old_reference = (
        "493765492819.dkr.ecr.us-east-1.amazonaws.com/leadpoet/"
        "sourcing-model@" + OLD_DIGEST
    )
    return {
        "schema_version": contracts.ROUND_CONFIGURATION_SCHEMA_VERSION,
        "round_id": ROUND,
        "mode": "live",
        "network_name": "finney",
        "netuid": 71,
        "baseline_hotkey": next(
            item["miner_hotkey"]
            for item in TARGETS
            if item["submission_id"] == "baseline-2026-09-21"
        ),
        "integrity_policy": "arena_integrity_v1",
        "scorer_image_digest": OLD_DIGEST,
        "scorer_image_reference": old_reference,
        "schedule": {"stage_1_scoring_close": "2026-09-21T11:00:01Z"},
        "scoring_call_quotas": {
            "deepline": 40,
            "openrouter": 120,
            "scrapingdog": 150,
        },
        "scoring_cap_microusd": 50_000_000,
        "cost_per_company_microusd": 800_000,
        "scoring_wall_clock_seconds": 900,
        "runner_slot_ceiling": 20,
        "lease_ttl_seconds": 3600,
        "scorer_policy": {"scoring_adapter_version": "qualification_integrity_v2"},
    }


def _old_scope(item: dict) -> dict:
    scope = dict(item["judgment_scope_doc"])
    scope["scorer_image_digest"] = OLD_DIGEST
    scope["scorer_image_reference"] = (
        "493765492819.dkr.ecr.us-east-1.amazonaws.com/leadpoet/"
        "sourcing-model@" + OLD_DIGEST
    )
    scope["cache_key"] = item["old_judgment_cache_key"]
    without_key = dict(scope)
    key = without_key.pop("cache_key")
    assert contracts.document_hash(without_key) == key
    return scope


def _assert_new_scope(item: dict) -> None:
    scope = dict(item["judgment_scope_doc"])
    key = scope.pop("cache_key")
    assert contracts.document_hash(scope) == key
    assert key == item["judgment_cache_key"]
    assert scope["scoring_input_hash"] == item["judgment_input_hash"]


def _seed(connection) -> None:
    participants = _participants()
    participant_by_id = {
        item["submission_id"]: item for item in participants
    }
    accepted_plan = []
    filler_submission = next(
        item["submission_id"]
        for item in participants
        if not item["is_king"]
    )
    patronus_runs = list(PATRONUS_ACCEPTED)
    for index in range(40):
        position = index % 10
        if index < len(patronus_runs):
            scored_run_id = patronus_runs[index]
            submission_id = scored_run_id.split(":")[1]
            output_ref = PATRONUS_ACCEPTED[scored_run_id][0]
        else:
            scored_run_id = f"{ROUND}:{filler_submission}:1:{position}:accepted-{index}"
            submission_id = filler_submission
            output_ref = f"arena/{ROUND}/outputs/accepted-{index}.json"
        accepted_plan.append({
            "scored_run_id": scored_run_id,
            "submission_id": submission_id,
            "icp_position": position,
            "output_ref": output_ref,
        })
    plan = [
        {
            "scored_run_id": item["scored_run_id"],
            "submission_id": item["submission_id"],
            "icp_position": item["icp_position"],
            "output_ref": INVALIDATION_BY_SCORED.get(item["scored_run_id"], {}).get(
                "old_output_ref", f"arena/{ROUND}/outputs/{item['scored_run_id']}.json"),
        }
        for item in TARGETS
    ] + accepted_plan
    scoring_plan = {
        "schema_version": contracts.SCORING_PLAN_SCHEMA_VERSION,
        "round_id": ROUND,
        "stage": 1,
        "work_items": plan,
        "zero_rows": [{
            "cause": "provider_error",
            "icp_position": 2,
            "submission_id": "baseline-2026-09-21",
        }],
    }
    with connection.cursor() as cursor:
        cursor.execute("SET session_replication_role=replica")
        cursor.execute(
            "TRUNCATE public.lab_arena_company_judgment_reservations,"
            "public.lab_arena_company_judgments,public.lab_arena_judgment_cache,"
            "public.lab_arena_ledger,public.lab_arena_runs,"
            "public.lab_arena_submissions,public.lab_arena_rounds "
            "RESTART IDENTITY CASCADE"
        )
        cursor.execute(
            "INSERT INTO public.lab_arena_rounds("
            "round_id,status,status_generation,stage_generation,configuration_doc,"
            "rewards_enabled,participants,benchmark_ref,evaluation_date,icp_set_date,"
            "stage1_scoring_plan_doc,cancel_reason,confirmation_bank_ref,"
            "confirmation_bank_hash) VALUES ("
            "%s,'cancelled',5,4,%s::jsonb,TRUE,%s::jsonb,%s,'2026-09-21',"
            "'2026-09-20',%s::jsonb,'scoring_incomplete',%s,%s)",
            (
                ROUND,
                json.dumps(_configuration()),
                json.dumps(participants),
                f"arena/{ROUND}/benchmark.json",
                json.dumps(scoring_plan),
                f"arena/{ROUND}/confirmation-bank.json",
                "sha256:" + "b" * 64,
            ),
        )
        cursor.executemany(
            "INSERT INTO public.lab_arena_submissions("
            "submission_id,round_id,miner_hotkey,status,is_king,source_ref,"
            "source_size_bytes,submission_doc,code_review_status,code_review_doc,"
            "code_review_claim,code_review_started_at,code_review_attempts) VALUES ("
            "%s,%s,%s,'frozen',%s,%s,%s,%s::jsonb,%s,%s::jsonb,%s,%s,%s)",
            [
                (
                    item["submission_id"], ROUND, item["miner_hotkey"],
                    item["is_king"], item["source_ref"],
                    item["source_size_bytes"], json.dumps(item),
                    "pending" if item["is_king"] else "passed",
                    None if item["is_king"] else json.dumps({"status": "passed"}),
                    None if item["is_king"] else "sha256:" + "c" * 64,
                    None if item["is_king"] else "2026-09-20T23:00:00Z",
                    0 if item["is_king"] else 1,
                )
                for item in participants
            ],
        )

        execute_rows = []
        for item in plan:
            miner = participant_by_id[item["submission_id"]]["miner_hotkey"]
            execute_rows.append((
                item["scored_run_id"], f"execute-plan:{item['scored_run_id']}",
                item["submission_id"], miner, 1, item["icp_position"],
                "accepted", "accepted", item["output_ref"],
            ))
        for index in range(60):
            participant = participants[index % len(participants)]
            execute_rows.append((
                f"execute-stage2-{index}", f"execute-stage2-assignment-{index}",
                participant["submission_id"], participant["miner_hotkey"],
                2, 10 + index % 10, "accepted", "accepted",
                f"arena/{ROUND}/outputs/stage2-{index}.json",
            ))
        for index in range(11):
            participant = participants[index % len(participants)]
            execute_rows.append((
                f"execute-failed-{index}", f"execute-failed-assignment-{index}",
                participant["submission_id"], participant["miner_hotkey"],
                1, index % 10, "failed", "model_error", None,
            ))
        assert len(execute_rows) == 130
        cursor.executemany(
            "INSERT INTO public.lab_arena_runs("
            "run_id,assignment_id,round_id,submission_id,miner_hotkey,stage,"
            "icp_position,attempt,kind,status,terminal_cause,output_ref,"
            "stage_generation) VALUES (%s,%s,%s,%s,%s,%s,%s,1,'execute',"
            "%s,%s,%s,3)",
            [row[:2] + (ROUND,) + row[2:] for row in execute_rows],
        )

        # Exact unresolved production identities and their frozen old-image
        # judgment bindings.
        for item in TARGETS:
            assignment = item["latest_run_id"].rsplit(":", 1)[0]
            old_scope = _old_scope(item)
            invalidation = INVALIDATION_BY_SCORED.get(item["scored_run_id"])
            if invalidation:
                cursor.execute(
                    "INSERT INTO public.lab_arena_runs("
                    "run_id,assignment_id,round_id,submission_id,miner_hotkey,stage,"
                    "icp_position,attempt,kind,status,terminal_cause,scored_run_id,"
                    "stage_generation,output_ref,result_doc,judgment_cache_key,"
                    "judgment_input_hash,judgment_scope_doc,judgment_group_leader,"
                    "judgment_group_miner_hotkeys) VALUES ("
                    "%s,%s,%s,%s,%s,1,%s,%s,'score','accepted','accepted',%s,3,%s,%s::jsonb,%s,%s,%s::jsonb,%s,%s)",
                    (
                        item["latest_run_id"], assignment, ROUND,
                        item["submission_id"], item["miner_hotkey"],
                        item["icp_position"], item["latest_attempt"],
                        item["scored_run_id"], invalidation["old_output_ref"],
                        json.dumps({"terminal_status": "accepted", "fixture_output_hash": invalidation["old_output_hash"]}),
                        item["old_judgment_cache_key"], item["judgment_input_hash"],
                        json.dumps(old_scope), item["judgment_group_leader"],
                        item["judgment_group_miner_hotkeys"],
                    ),
                )
                continue
            if item["latest_attempt"] == 2:
                cursor.execute(
                    "INSERT INTO public.lab_arena_runs("
                    "run_id,assignment_id,round_id,submission_id,miner_hotkey,stage,"
                    "icp_position,attempt,kind,status,terminal_cause,scored_run_id,"
                    "stage_generation,judgment_cache_key,judgment_input_hash,"
                    "judgment_scope_doc,judgment_group_leader,"
                    "judgment_group_miner_hotkeys) VALUES ("
                    "%s,%s,%s,%s,%s,1,%s,1,'score','failed','judge_error',%s,3,"
                    "%s,%s,%s::jsonb,%s,%s)",
                    (
                        assignment + ":1", assignment, ROUND,
                        item["submission_id"], item["miner_hotkey"],
                        item["icp_position"], item["scored_run_id"],
                        item["old_judgment_cache_key"],
                        item["judgment_input_hash"], json.dumps(old_scope),
                        item["judgment_group_leader"],
                        item["judgment_group_miner_hotkeys"],
                    ),
                )
            cursor.execute(
                "INSERT INTO public.lab_arena_runs("
                "run_id,assignment_id,round_id,submission_id,miner_hotkey,stage,"
                "icp_position,attempt,kind,status,terminal_cause,scored_run_id,"
                "stage_generation,judgment_cache_key,judgment_input_hash,"
                "judgment_scope_doc,judgment_group_leader,"
                "judgment_group_miner_hotkeys) VALUES ("
                "%s,%s,%s,%s,%s,1,%s,%s,'score','failed',%s,%s,3,"
                "%s,%s,%s::jsonb,%s,%s)",
                (
                    item["latest_run_id"], assignment, ROUND,
                    item["submission_id"], item["miner_hotkey"],
                    item["icp_position"], item["latest_attempt"],
                    item["latest_terminal_cause"], item["scored_run_id"],
                    item["old_judgment_cache_key"],
                    item["judgment_input_hash"], json.dumps(old_scope),
                    item["judgment_group_leader"],
                    item["judgment_group_miner_hotkeys"],
                ),
            )

        for index, item in enumerate(accepted_plan):
            assignment = f"score-accepted-{index}"
            if index == 0:
                cursor.execute(
                    "INSERT INTO public.lab_arena_runs("
                    "run_id,assignment_id,round_id,submission_id,miner_hotkey,stage,"
                    "icp_position,attempt,kind,status,terminal_cause,scored_run_id,"
                    "stage_generation) VALUES (%s,%s,%s,%s,%s,1,%s,1,'score',"
                    "'failed','judge_error',%s,3)",
                    (
                        assignment + ":1", assignment, ROUND,
                        item["submission_id"],
                        participant_by_id[item["submission_id"]]["miner_hotkey"],
                        item["icp_position"], item["scored_run_id"],
                    ),
                )
                attempt = 2
            else:
                attempt = 1
            cursor.execute(
                "INSERT INTO public.lab_arena_runs("
                "run_id,assignment_id,round_id,submission_id,miner_hotkey,stage,"
                "icp_position,attempt,kind,status,terminal_cause,scored_run_id,"
                "stage_generation,output_ref) VALUES (%s,%s,%s,%s,%s,1,%s,%s,"
                "'score','accepted','accepted',%s,3,%s)",
                (
                    assignment + f":{attempt}", assignment, ROUND,
                    item["submission_id"],
                    participant_by_id[item["submission_id"]]["miner_hotkey"],
                    item["icp_position"], attempt, item["scored_run_id"],
                    item["output_ref"],
                ),
            )
        cursor.execute(
            "INSERT INTO public.lab_arena_ledger("
            "entry_kind,miner_hotkey,round_id,submission_id,run_id,stage,"
            "call_identity,provider,operation_id,funding_source,amount_microusd,"
            "entry_doc,terminal_response) VALUES ("
            "'settlement',%s,%s,%s,%s,1,%s,'openrouter','openrouter.chat',"
            "'host',12345,'{}'::jsonb,'{\"status\":200,\"call_succeeded\":true}'::jsonb)",
            (
                TARGETS[0]["miner_hotkey"], ROUND,
                TARGETS[0]["submission_id"], TARGETS[0]["latest_run_id"],
                "sha256:" + "1" * 64,
            ),
        )
        cursor.execute("SET session_replication_role=origin")
    connection.commit()


def _snapshot(cursor, where: str, params=()):
    cursor.execute(where, params)
    return cursor.fetchone()[0]


def _apply(connection) -> None:
    with connection.cursor() as cursor:
        cursor.execute(MIGRATION.read_text(encoding="utf-8"))
    connection.commit()


def test_recovery_adds_only_new_image_pending_rows_and_preserves_old_evidence(database):
    for item in TARGETS:
        _assert_new_scope(item)
    psycopg2, dsn = database
    with psycopg2.connect(**dsn) as connection:
        _seed(connection)
        with connection.cursor() as cursor:
            round_before = _snapshot(
                cursor,
                "SELECT to_jsonb(r) FROM public.lab_arena_rounds r WHERE round_id=%s",
                (ROUND,),
            )
            old_scores = _snapshot(
                cursor,
                "SELECT jsonb_agg(to_jsonb(r) ORDER BY run_id) FROM "
                "public.lab_arena_runs r WHERE round_id=%s AND kind='score'",
                (ROUND,),
            )
            execute = _snapshot(
                cursor,
                "SELECT jsonb_agg(to_jsonb(r) ORDER BY run_id) FROM "
                "public.lab_arena_runs r WHERE round_id=%s AND kind='execute'",
                (ROUND,),
            )
            ledger = _snapshot(
                cursor,
                "SELECT jsonb_agg(to_jsonb(l) ORDER BY entry_id) FROM "
                "public.lab_arena_ledger l WHERE round_id=%s",
                (ROUND,),
            )
            submissions = _snapshot(
                cursor,
                "SELECT jsonb_agg(to_jsonb(s) ORDER BY submission_id) FROM "
                "public.lab_arena_submissions s WHERE round_id=%s",
                (ROUND,),
            )
        _apply(connection)
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT status,status_generation,stage_generation,cancel_reason,"
                "configuration_doc->>'scorer_image_digest' FROM "
                "public.lab_arena_rounds WHERE round_id=%s",
                (ROUND,),
            )
            assert cursor.fetchone() == (
                "stage1_scoring", 6, 5, None, NEW_DIGEST
            )
            round_after = _snapshot(
                cursor,
                "SELECT to_jsonb(r) FROM public.lab_arena_rounds r WHERE round_id=%s",
                (ROUND,),
            )
            for key in ("participants", "benchmark_ref", "evaluation_date",
                        "icp_set_date", "stage1_scoring_plan_doc",
                        "confirmation_bank_ref", "confirmation_bank_hash",
                        "rewards_enabled"):
                assert round_after[key] == round_before[key]
            assert round_after["configuration_doc"]["scorer_image_digest"] == NEW_DIGEST
            assert round_after["configuration_doc"]["scorer_image_reference"].endswith(NEW_DIGEST)
            assert {key: value for key, value in round_after.items()
                    if key not in {"status", "status_generation", "stage_generation",
                                   "cancel_reason", "configuration_doc", "updated_at"}} == {
                key: value for key, value in round_before.items()
                if key not in {"status", "status_generation", "stage_generation",
                               "cancel_reason", "configuration_doc", "updated_at"}
            }
            cursor.execute(
                "SELECT count(*),bool_and(status='pending'),"
                "bool_and(attempt=1),bool_and(stage_generation=5),"
                "bool_and(judgment_scope_doc->>'scorer_image_digest'=%s),"
                "bool_and(judgment_scope_doc->>'scoring_input_hash'="
                "judgment_input_hash) FROM public.lab_arena_runs WHERE "
                "round_id=%s AND assignment_id LIKE '%%:score:recovery350'",
                (NEW_DIGEST, ROUND),
            )
            assert cursor.fetchone() == (19, True, True, True, True, True)
            cursor.execute(
                "SELECT count(*),count(*) FILTER (WHERE status='accepted'),"
                "count(*) FILTER (WHERE status='failed'),"
                "count(*) FILTER (WHERE status='failed' AND terminal_cause='judge_error') "
                "FROM public.lab_arena_runs WHERE round_id=%s AND kind='score' "
                "AND assignment_id NOT LIKE '%%:score:recovery350'",
                (ROUND,),
            )
            assert cursor.fetchone() == (62, 40, 22, 8)
            cursor.execute(
                "SELECT jsonb_agg(to_jsonb(r) ORDER BY run_id) FROM "
                "public.lab_arena_runs r WHERE round_id=%s AND kind='score' "
                "AND assignment_id NOT LIKE '%%:score:recovery350'",
                (ROUND,),
            )
            after_scores = cursor.fetchone()[0]
            assert len(after_scores) == len(old_scores) == 62
            old_by_id = {row["run_id"]: row for row in old_scores}
            after_by_id = {row["run_id"]: row for row in after_scores}
            assert set(old_by_id) == set(after_by_id)
            for run_id, before in old_by_id.items():
                after = after_by_id[run_id]
                allowed = {"status", "terminal_cause", "terminal_doc", "updated_at"}
                if run_id not in {item["run_id"] for item in INVALIDATIONS}:
                    assert after == before
                else:
                    assert {k: v for k, v in after.items() if k not in allowed} == {
                        k: v for k, v in before.items() if k not in allowed
                    }
                    assert after["status"] == "failed"
                    assert after["terminal_cause"] == "judge_error"
                    invalidation = next(item for item in INVALIDATIONS if item["run_id"] == run_id)
                    assert after["output_ref"] == invalidation["old_output_ref"]
                    assert after["result_doc"]["fixture_output_hash"] == invalidation["old_output_hash"]
                    assert after["terminal_doc"]["previous_output_hash"] == invalidation["old_output_hash"]
                    assert after["terminal_doc"]["previous_output_ref"] == invalidation["old_output_ref"]
                    assert after["terminal_doc"]["migration"] == "350"
                if run_id in PATRONUS_ACCEPTED:
                    assert after == before
            assert _snapshot(
                cursor,
                "SELECT jsonb_agg(to_jsonb(r) ORDER BY run_id) FROM "
                "public.lab_arena_runs r WHERE round_id=%s AND kind='execute'",
                (ROUND,),
            ) == execute
            assert _snapshot(
                cursor,
                "SELECT jsonb_agg(to_jsonb(l) ORDER BY entry_id) FROM "
                "public.lab_arena_ledger l WHERE round_id=%s",
                (ROUND,),
            ) == ledger
            assert _snapshot(
                cursor,
                "SELECT jsonb_agg(to_jsonb(s) ORDER BY submission_id) FROM "
                "public.lab_arena_submissions s WHERE round_id=%s",
                (ROUND,),
            ) == submissions
            cursor.execute(
                "SELECT tgname,tgenabled FROM pg_trigger "
                "WHERE tgrelid='public.lab_arena_runs'::regclass "
                "AND NOT tgisinternal ORDER BY tgname"
            )
            assert all(enabled == "O" for _, enabled in cursor.fetchall())


def test_recovery_rows_claim_complete_and_replay_without_reset(database):
    psycopg2, dsn = database
    with psycopg2.connect(**dsn) as connection:
        _seed(connection)
        _apply(connection)
    store = ArenaStore(PsycopgTransport(lambda: psycopg2.connect(**dsn)))
    completed = 0
    while True:
        leased, token, _, _ = claim(
            store, ROUND, RUNNER, parallelism=20, ceiling=20,
            excluded=[RUNNER],
        )
        if leased["status"] == "no_pending":
            break
        assert leased["kind"] == "score"
        source = store.get_run(leased["run_id"])
        output = scoring.build_scoring_output(
            leased["scored_run_id"],
            [{"company_name": "fixture", "final_score": 1.0}],
        )
        evidence = judgment_cache.build_evidence_snapshot(
            output=output,
            cache_scope=source["judgment_scope_doc"],
            source_score_run_id=leased["run_id"],
            source_scored_run_id=leased["scored_run_id"],
            source_output_ref=f"arena/{ROUND}/scores/{leased['run_id']}.json",
            source_runner_hotkey=RUNNER,
            runner_authority_exclusions=leased["runner_authority_exclusions"],
        )
        result = store.complete_attempt(
            run_id=leased["run_id"],
            lease_token_hash=hash_lease_token(token),
            result={"terminal_status": "accepted"},
            terminal_cause="accepted",
            output_ref=evidence["source_output_ref"],
            judgment_evidence=evidence,
            judgment_evidence_hash=contracts.document_hash(evidence),
        )
        assert result["status"] == "accepted"
        completed += 1
    assert completed == 15  # eleven unresolved groups plus four reviewed repairs
    rows = [
        row for row in store.list_runs(ROUND, stage=1, kind="score")
        if row["assignment_id"].endswith(":score:recovery350")
    ]
    assert len(rows) == 19
    assert all(row["status"] == "accepted" for row in rows)

    before = {row["run_id"]: row for row in rows}
    with psycopg2.connect(**dsn) as connection:
        _apply(connection)
    after = {
        row["run_id"]: row
        for row in store.list_runs(ROUND, stage=1, kind="score")
        if row["assignment_id"].endswith(":score:recovery350")
    }
    assert after == before

    with psycopg2.connect(**dsn) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT public.lab_arena_close_scoring(%s::text,1::smallint)",
                (ROUND,),
            )
            assert cursor.fetchone()[0] == {
                "status": "closed",
                "round_status": "stage1_judged",
                "incomplete_assignments": 0,
                "stage_generation": 6,
            }
            cursor.execute(
                "SELECT status,status_generation,stage_generation,cancel_reason "
                "FROM public.lab_arena_rounds WHERE round_id=%s",
                (ROUND,),
            )
            assert cursor.fetchone() == ("stage1_judged", 7, 6, None)


@pytest.mark.parametrize("field,value", [
    ("status", "published"),
    ("cancel_reason", "manual_cancel"),
])
def test_recovery_rejects_wrong_terminal_round(database, field, value):
    psycopg2, dsn = database
    with psycopg2.connect(**dsn) as connection:
        _seed(connection)
        with connection.cursor() as cursor:
            cursor.execute("SET LOCAL session_replication_role=replica")
            cursor.execute(
                f"UPDATE public.lab_arena_rounds SET {field}=%s WHERE round_id=%s",
                (value, ROUND),
            )
            cursor.execute("SET LOCAL session_replication_role=origin")
            cursor.execute("SAVEPOINT refused")
            with pytest.raises(psycopg2.Error, match="terminal round differs"):
                cursor.execute(MIGRATION.read_text(encoding="utf-8"))
            cursor.execute("ROLLBACK TO SAVEPOINT refused")
