"""PostgreSQL proof for the inert Sep16 native baseline rerun."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from bittensor_wallet import Keypair

from lab_arena import contact_policy, contracts, rewards, scoring
from tests.lab_arena.lab_arena_pg_harness import (
    CURRENT_SERVICE_MIGRATIONS, database_with_lab_arena_migration,
)

MIGRATIONS = CURRENT_SERVICE_MIGRATIONS + (
    "264-lab-arena-codex-cost-reconciliation.sql",
    "265-arena-2026-09-16-native-baseline-rerun.sql",
    "267-arena-2026-09-16-failed-cost-publication.sql",
)
ROUND = "arena-2026-09-16"
BASELINE = "baseline-2026-09-16"
BASIS_HASH = "sha256:b19a147f1c8cc8365e3cf7dcc96b827ffffdd6c1e8fad65ad9fc25c8895e493f"
BANK_HASH = "42b417604acf15e612570687e8fbccef82fe7b0808364a6491b4a2a4bcb07390"
OLD_IMAGE = "sha256:ee84f274ba24b07fa204c03535b21aac3c030c72aff0fcf7918d88d3c2c8ddee"
NEW_IMAGE = "sha256:" + "f" * 64


@pytest.fixture
def connect():
    generator = database_with_lab_arena_migration(MIGRATIONS)
    psycopg2, dsn = next(generator)
    try:
        yield lambda: psycopg2.connect(**dsn)
    finally:
        generator.close()


def _schedule():
    now = datetime.now(timezone.utc)

    def future(hours=0, minutes=0):
        return (now + timedelta(hours=hours, minutes=minutes)).strftime("%Y-%m-%dT%H:%M:%SZ")

    return {
        "submission_open": "2026-09-15T00:00:00Z",
        "submission_cutoff": "2026-09-16T00:00:00Z",
        "benchmark_deadline": "2026-09-16T00:30:00Z",
        "stage_1_start": "2026-09-16T00:31:00Z",
        "stage_1_close": future(hours=2),
        "stage_1_scoring_close": future(hours=3),
        "stage_2_start": future(hours=3, minutes=1),
        "stage_2_close": future(hours=4),
        "final_scoring_close": future(hours=5),
        "publication_deadline": future(hours=5, minutes=1),
    }



def _seed_observed_sep16(connection, objects=None):
    hotkeys = [Keypair.create_from_uri("//Sep16Proof%d" % n).ss58_address
               for n in range(9)]
    ids = [BASELINE] + ["miner-2026-09-16-%d" % n for n in range(1, 5)]
    schedule = _schedule()
    config = {
        "schema_version": contracts.ROUND_CONFIGURATION_SCHEMA_VERSION,
        "round_id": ROUND, "mode": "live", "rewards_enabled": True,
        "network_name": "finney", "netuid": 71,
        "schedule": schedule,
        "integrity_policy": "arena_integrity_v1",
        "contact_policy": "contacts_v1",
        "intent_details_policy": "intent_details_v1",
        "sourcing_cost_eligibility_policy": contracts.SUCCESSFUL_CALLS_COST_POLICY,
        "stage_1_icp_count": 10, "stage_2_icp_count": 10,
        "finalist_count": 10, "max_challengers": 20,
        "runner_slot_ceiling": 20, "max_attempts_per_assignment": 2,
        "lease_ttl_seconds": 3600, "companies_per_icp": 5,
        "providers": list(contracts.PROVIDERS),
        "call_quotas": dict(contracts.CALL_QUOTAS_PER_ICP),
        "scoring_call_quotas": dict(contracts.SCORING_CALL_QUOTAS_PER_WORK_ITEM),
        "icp_wall_clock_seconds": 2700, "scoring_wall_clock_seconds": 900,
        "scorer_policy": scoring.build_scorer_policy(
            scoring_adapter_version=contact_policy.SCORING_ADAPTER,
            intent_details=True,
        ),
        "execution_cap_microusd": 80_000_000,
        "cost_per_company_microusd": 800_000,
        "scoring_cap_microusd": 50_000_000,
        "scorer_image_digest": OLD_IMAGE,
        "scorer_image_reference": "registry.example/scorer@" + OLD_IMAGE,
        "baseline_hotkey": hotkeys[0],
        "baseline_source_url": "https://github.com/leadpoet/pydantic-harness/archive/refs/heads/lab.tar.gz",
        "runner_hotkeys": hotkeys[5:7], "banned_hotkeys": [],
        "reward_constants": rewards.reward_constants_document(),
        "parallel_twenty_icp_execution": True,
        "checkpoint_deadline_policy": contracts.CHECKPOINT_DEADLINE_POLICY,
    }
    contracts.validate_round_configuration(config)
    icps = daily_icps() if objects is not None else None
    if objects is not None:
        objects.put("arena/arena-2026-09-16/benchmark.json", json.dumps({
            "schema_version": "leadpoet.lab_arena.benchmark.v1",
            "round_id": ROUND, "icps": icps,
        }).encode())
    participants = [
        {"submission_id": submission_id, "miner_hotkey": hotkeys[index],
         "is_king": index == 0,
         "source_ref": "arena/%s/sources/%s.tar.gz" % (ROUND, submission_id),
         "source_size_bytes": 123204 if index == 0 else 4096}
        for index, submission_id in enumerate(ids)
    ]
    publication = {
        "king_decision": {"outcome": "no_king", "king_hotkey": ""},
        "final_ranking": [
            {"submission_id": submission_id, "is_baseline": index == 0,
             "eligible": False,
             "eligibility_reason": "cost_per_company_exceeded",
             "final_score": 0.0,
             "cost_summary": {
                 "sourcing_cost_eligibility_policy":
                     contracts.SUCCESSFUL_CALLS_COST_POLICY,
                 "competition_sourcing_microusd": 4463935 if index == 0 else 20000000,
                 "eligibility_cap_microusd": 0 if index == 0 else 16000000,
             }}
            for index, submission_id in enumerate(ids)
        ],
    }
    with connection.cursor() as cursor:
        for table in ("lab_arena_rounds", "lab_arena_submissions",
                      "lab_arena_runs", "lab_arena_ledger"):
            cursor.execute("ALTER TABLE public.%s DISABLE TRIGGER USER" % table)
        cursor.execute(
            "INSERT INTO public.lab_arena_rounds (round_id,status,status_generation,"
            "stage_generation,configuration_doc,rewards_enabled,participants,"
            "benchmark_ref,evaluation_date,icp_set_date,finalists,publication_doc,"
            "king_outcome,effective_reward_epoch,reward_basis_hash,reward_basis_doc,"
            "signing_key_doc,reward_activated_at,published_at) "
            "VALUES (%s,'published',12,8,%s::jsonb,true,%s::jsonb,"
            "%s,%s,%s,%s::jsonb,%s::jsonb,'no_king',25201,%s,%s::jsonb,%s::jsonb,"
            "%s,%s)",
            (ROUND, json.dumps(config), json.dumps(participants),
             "arena/arena-2026-09-16/benchmark.json", "2026-09-16",
             "2026-09-15", json.dumps(ids[1:]), json.dumps(publication),
             BASIS_HASH, json.dumps({"round_id": ROUND,
                                    "king_outcome": "no_king",
                                    "effective_reward_epoch": 25201,
                                    "published_at": "2026-09-16T00:44:44Z"}),
             json.dumps({}), "2026-09-16T00:44:46Z",
             "2026-09-16T00:44:44Z"),
        )
        for index, submission_id in enumerate(ids):
            cursor.execute(
                "INSERT INTO public.lab_arena_submissions (submission_id,round_id,"
                "miner_hotkey,status,is_king,source_ref,source_size_bytes,consent,"
                "submission_doc,frozen_at,code_review_status,code_review_attempts,"
                "code_review_doc,code_review_claim,code_review_started_at) VALUES "
                "(%s,%s,%s,'frozen',%s,%s,%s,"
                "'{\"public_rerun\":true}'::jsonb,'{}'::jsonb,now(),"
                "%s,%s,%s::jsonb,%s,CASE WHEN %s THEN now() ELSE NULL END)",
                (submission_id, ROUND, hotkeys[index], index == 0,
                 participants[index]["source_ref"],
                 participants[index]["source_size_bytes"],
                 "pending" if index == 0 else "passed",
                 0 if index == 0 else 1,
                 None if index == 0 else json.dumps({"decision": "passed"}),
                 None if index == 0 else "sha256:" + ("%064x" % (index + 7000)),
                 index != 0),
            )
        for n in range(4):
            cursor.execute(
                "INSERT INTO public.lab_arena_submissions (submission_id,round_id,"
                "miner_hotkey,status,submission_doc) VALUES (%s,%s,%s,'rejected','{}'::jsonb)",
                ("rejected-2026-09-16-%d" % n, ROUND, hotkeys[5 + n]),
            )
        accepted_challenger = []
        for index, submission_id in enumerate(ids):
            for position in range(20):
                stage = 1 if position < 10 else 2
                execute_id = "old-execute-%d-%d" % (index, position)
                failed = False
                cursor.execute(
                    "INSERT INTO public.lab_arena_runs (run_id,assignment_id,round_id,"
                    "submission_id,miner_hotkey,stage,icp_position,attempt,kind,status,"
                    "output_ref,terminal_cause) VALUES (%s,%s,%s,%s,%s,%s,%s,1,'execute',"
                    "%s,%s,%s)",
                    (execute_id, execute_id, ROUND, submission_id,
                     hotkeys[index], stage, position,
                     "failed" if failed else "accepted",
                     None if failed else "arena/output/%s.json" % execute_id,
                     "model_error" if failed else "accepted"),
                )
                if failed:
                    if objects is not None:
                        cursor.execute(
                            "UPDATE public.lab_arena_runs SET per_icp_score=0, "
                            "qualification_doc='{\"companies\":[]}'::jsonb "
                            "WHERE run_id=%s", (execute_id,),
                        )
                    continue
                if index > 0:
                    accepted_challenger.append((execute_id, submission_id,
                                                hotkeys[index], stage, position))
                score_assignment = (
                    "old-score-%d-%d" % (index, position) if index == 0 else
                    "%s:%s:%d:%d:score" % (ROUND, submission_id, stage, position)
                )
                score_id = (
                    score_assignment if index == 0 else score_assignment + ":1"
                )
                # The protected original has ten stage-two miner-account
                # failures and no accepted replacement for those executions.
                score_failed = index == 1 and stage == 2
                value = qualification = None
                if objects is not None:
                    value, qualification = _proof_execution(
                        objects, icps[position], index, position, execute_id,
                        None if score_failed else score_id,
                    )
                cursor.execute(
                    "INSERT INTO public.lab_arena_runs (run_id,assignment_id,round_id,"
                    "submission_id,miner_hotkey,stage,icp_position,attempt,kind,status,"
                    "scored_run_id,terminal_cause,output_ref,runner_hotkey,"
                    "judgment_cache_source_run_id) VALUES (%s,%s,%s,%s,%s,%s,%s,1,'score',"
                    "%s,%s,%s,%s,%s,%s)",
                    (score_id, score_assignment, ROUND, submission_id, hotkeys[index],
                     stage, position, "failed" if score_failed else "accepted",
                    execute_id, "credential_error" if score_failed else "accepted",
                     "arena/score/%s.json" % score_id, hotkeys[5],
                     "old-score-0-0" if 0 < len(accepted_challenger) <= 79 else None),
                )
                if objects is not None and not score_failed:
                    cursor.execute(
                        "UPDATE public.lab_arena_runs SET per_icp_score=%s, "
                        "qualification_doc=%s::jsonb WHERE run_id=%s",
                        (value, json.dumps(qualification), execute_id),
                    )
            if index > 0:
                cursor.execute(
                    "INSERT INTO public.lab_arena_ledger "
                    "(entry_kind,miner_hotkey,round_id,submission_id,run_id,"
                    "stage,call_identity,provider,operation_id,amount_microusd,"
                    "terminal_response) VALUES ('settlement',%s,%s,%s,%s,2,%s,"
                    "'deepline','proof-call',20000000,'{\"call_succeeded\":true}'::jsonb)",
                    (hotkeys[index], ROUND, submission_id,
                     "old-execute-%d-10" % index,
                    "sha256:" + ("%064x" % (index + 5000))),
                )
                if objects is not None and index == 2:
                    cursor.execute(
                        "INSERT INTO public.lab_arena_ledger "
                        "(entry_kind,miner_hotkey,round_id,submission_id,run_id,"
                        "stage,call_identity,provider,operation_id,amount_microusd,"
                        "entry_doc) VALUES ('uncertain',%s,%s,%s,%s,1,%s,"
                        "'deepline','proof-uncertain',500000,%s::jsonb)",
                        (hotkeys[index], ROUND, submission_id,
                         "old-execute-%d-0" % index,
                         "sha256:" + ("%064x" % (index + 6000)),
                         json.dumps({"reason": "worker_reported",
                                     "call": {"call_succeeded": False}})),
                    )
        assert len(accepted_challenger) == 80
        cursor.execute(
            "INSERT INTO public.lab_arena_ledger (entry_kind,miner_hotkey,round_id,"
            "submission_id,run_id,amount_microusd) VALUES ('settlement',%s,%s,%s,%s,4463935)",
            (hotkeys[0], ROUND, BASELINE, "old-execute-0-0"),
        )
        cache_key = "sha256:" + "c" * 64
        input_hash = "sha256:" + "d" * 64
        cursor.execute(
            "INSERT INTO public.lab_arena_judgment_cache (cache_key,scope_doc,"
            "scoring_input_hash,evidence_hash,evidence_doc,source_score_run_id,"
            "source_scored_run_id,source_runner_hotkey) VALUES (%s,%s::jsonb,%s,%s,"
            "%s::jsonb,%s,%s,%s)",
            (cache_key, json.dumps({"cache_key": cache_key,
                                    "scoring_input_hash": input_hash}),
             input_hash, "sha256:" + "e" * 64,
             json.dumps({"cache_key": cache_key,
                         "scoring_input_hash": input_hash,
                         "source_score_run_id": "old-score-0-0",
                         "source_scored_run_id": "old-execute-0-0",
                         "source_runner_hotkey": hotkeys[5],
                         "runner_authority_exclusions": [hotkeys[5]]}),
             "old-score-0-0", "old-execute-0-0", hotkeys[5]),
        )
        for table in ("lab_arena_ledger", "lab_arena_runs",
                      "lab_arena_submissions", "lab_arena_rounds"):
            cursor.execute("ALTER TABLE public.%s ENABLE TRIGGER USER" % table)
    connection.commit()
    return schedule, hotkeys, ids


def _state_seal(connection):
    with connection.cursor() as cursor:
        cursor.execute(
            "SELECT 'sha256:' || encode(extensions.digest(to_jsonb(round_row)::text,"
            "'sha256'),'hex') FROM public.lab_arena_rounds AS round_row "
            "WHERE round_id=%s", (ROUND,),
        )
        round_hash = cursor.fetchone()[0]
        cursor.execute(
            "SELECT 'sha256:' || encode(extensions.digest(to_jsonb(submission_row)::text,"
            "'sha256'),'hex') FROM public.lab_arena_submissions AS submission_row "
            "WHERE submission_id=%s", (BASELINE,),
        )
        baseline_submission_hash = cursor.fetchone()[0]

        def rows_hash(table, predicate, *, order_by, stripped=False, empty=False):
            expression = "to_jsonb(row_value)"
            if stripped:
                expression += " - 'round_id' - 'submission_id'"
            aggregate = (
                "coalesce(string_agg(encode(extensions.digest((%s)::text,'sha256'),"
                "'hex'),'' ORDER BY %s),'')" if empty else
                "string_agg(encode(extensions.digest((%s)::text,'sha256'),"
                "'hex'),'' ORDER BY %s)"
            ) % (expression, order_by)
            cursor.execute(
                "SELECT 'sha256:' || encode(extensions.digest((%s),'sha256'),'hex') "
                "FROM public.%s AS row_value WHERE %s" % (aggregate, table, predicate)
            )
            return cursor.fetchone()[0]

        baseline_runs_hash = rows_hash(
            "lab_arena_runs", "round_id='%s' AND submission_id='%s'" % (ROUND, BASELINE),
            stripped=True, order_by="run_id",
        )
        baseline_ledger_hash = rows_hash(
            "lab_arena_ledger", "round_id='%s' AND submission_id='%s'" % (ROUND, BASELINE),
            stripped=True, order_by="entry_id",
        )
        challenger_runs_hash = rows_hash(
            "lab_arena_runs", "round_id='%s' AND submission_id<>'%s'" % (ROUND, BASELINE),
            order_by="run_id",
        )
        challenger_submissions_hash = rows_hash(
            "lab_arena_submissions",
            "round_id='%s' AND submission_id<>'%s'" % (ROUND, BASELINE),
            order_by="submission_id",
        )
        challenger_ledger_hash = rows_hash(
            "lab_arena_ledger", "round_id='%s' AND submission_id<>'%s'" % (ROUND, BASELINE),
            empty=True, order_by="entry_id",
        )
        cursor.execute(
            "SELECT coalesce(max(entry_id),0) FROM public.lab_arena_ledger "
            "WHERE round_id=%s AND submission_id<>%s", (ROUND, BASELINE),
        )
        challenger_ledger_max = cursor.fetchone()[0]
        cursor.execute(
            "SELECT coalesce(sum(amount_microusd),0) FROM public.lab_arena_ledger "
            "WHERE round_id=%s AND submission_id=%s "
            "AND entry_kind IN ('settlement','uncertain')", (ROUND, BASELINE),
        )
        baseline_actual = cursor.fetchone()[0]
    return {
        "round": round_hash,
        "baseline_submission": baseline_submission_hash,
        "baseline_runs": baseline_runs_hash,
        "baseline_ledger": baseline_ledger_hash,
        "challenger_runs": challenger_runs_hash,
        "challenger_submissions": challenger_submissions_hash,
        "challenger_ledger": challenger_ledger_hash,
        "challenger_ledger_max": challenger_ledger_max,
        "baseline_actual": baseline_actual,
    }


def _insert_authority(connection, schedule):
    seal = _state_seal(connection)
    with connection.cursor() as cursor:
        cursor.execute(
            "INSERT INTO public.lab_arena_sep16_rerun_release_authority ("
            "round_id,source_ref,source_size_bytes,source_sha256,source_commit,"
            "champion_model_main_commit,champion_model_lab_commit,bank_sha256,"
            "old_round_hash,old_baseline_submission_hash,old_baseline_runs_hash,"
            "old_baseline_ledger_hash,old_challenger_runs_hash,"
            "old_challenger_submissions_hash,old_challenger_ledger_hash,"
            "old_challenger_ledger_max_entry_id,old_baseline_actual_microusd,"
            "old_scorer_image_digest,new_scorer_image_reference,"
            "new_scorer_image_digest,scoring_tree_hash,native_runtime_commit,"
            "verified_parallel_runner_slots,forward_schedule) VALUES ("
            "%s,%s,4096,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,"
            "%s,%s,%s,10,%s::jsonb)",
            (
                ROUND,
                "arena/arena-2026-09-16/sources/"
                "baseline-2026-09-16-native-rerun265.tar.gz",
                "a" * 64, "b" * 40, "b" * 40, "b" * 40, BANK_HASH,
                seal["round"], seal["baseline_submission"], seal["baseline_runs"],
                seal["baseline_ledger"], seal["challenger_runs"],
                seal["challenger_submissions"], seal["challenger_ledger"],
                seal["challenger_ledger_max"], seal["baseline_actual"], OLD_IMAGE,
                "registry.example/native@" + NEW_IMAGE, NEW_IMAGE,
                "f8452314cc8b1529fbfa8b7fc9345143c1b7d731",
                "9d97e2ae295f715209b8fdfef2b6ddebdc46622e",
                json.dumps(schedule),
            ),
        )
    connection.commit()
    return seal


def _render_owner_seal(schedule, seal, *, source_sha="a" * 64):
    template = (
        Path(__file__).parents[2]
        / "scripts/266-arena-2026-09-16-native-rerun-owner-seal.sql.template"
    ).read_text(encoding="utf-8")
    values = {
        "__SEALED_SOURCE_SIZE_BYTES__": "4096",
        "__SEALED_SOURCE_SHA256__": source_sha,
        "__SEALED_CHAMPION_MODEL_COMMIT__": "b" * 40,
        "__SEALED_NATIVE_IMAGE_REFERENCE__": "registry.example/native@" + NEW_IMAGE,
        "__SEALED_NATIVE_IMAGE_DIGEST__": NEW_IMAGE,
        "__SEALED_OLD_ROUND_HASH__": seal["round"],
        "__SEALED_OLD_BASELINE_SUBMISSION_HASH__": seal["baseline_submission"],
        "__SEALED_OLD_BASELINE_RUNS_HASH__": seal["baseline_runs"],
        "__SEALED_OLD_BASELINE_LEDGER_HASH__": seal["baseline_ledger"],
        "__SEALED_OLD_CHALLENGER_RUNS_HASH__": seal["challenger_runs"],
        "__SEALED_OLD_CHALLENGER_SUBMISSIONS_HASH__": seal["challenger_submissions"],
        "__SEALED_OLD_CHALLENGER_LEDGER_HASH__": seal["challenger_ledger"],
        "__SEALED_OLD_CHALLENGER_LEDGER_MAX_ENTRY_ID__": str(
            seal["challenger_ledger_max"]
        ),
        "__SEALED_OLD_BASELINE_ACTUAL_MICROUSD__": str(seal["baseline_actual"]),
        "__SEALED_FORWARD_SCHEDULE_JSON__": json.dumps(schedule),
    }
    for marker, value in values.items():
        template = template.replace(marker, value)
        assert marker not in template
    return template


def _prepare(connection, schedule):
    _insert_authority(connection, schedule)
    with connection.cursor() as cursor:
        cursor.execute(
            "SELECT public.lab_arena_prepare_sep16_baseline_rerun_v1("
            "4096,%s,%s,%s,%s::jsonb)",
            ("a" * 64, "b" * 40, BANK_HASH, json.dumps(schedule)),
        )
        result = cursor.fetchone()[0]
    connection.commit()
    return result


def _stage1_scoring_items(
    connection, hotkeys, finalists, *, failed_position=None, failed_positions=(),
):
    with connection.cursor() as cursor:
        cursor.execute("ALTER TABLE public.lab_arena_runs DISABLE TRIGGER USER")
        cursor.execute("ALTER TABLE public.lab_arena_rounds DISABLE TRIGGER USER")
        cursor.execute(
            "UPDATE public.lab_arena_runs SET status='accepted',per_icp_score=1.0,"
            "terminal_cause='accepted',output_ref='arena/output/' || run_id || '.json' "
            "WHERE round_id=%s AND submission_id=%s AND kind='execute' "
            "AND status='pending'",
            (ROUND, BASELINE),
        )
        failed = set(failed_positions)
        if failed_position is not None:
            failed.add(failed_position)
        if failed:
            cursor.execute(
                "UPDATE public.lab_arena_runs SET status='failed',"
                "terminal_cause='model_error',output_ref=NULL,per_icp_score=NULL "
                "WHERE round_id=%s AND submission_id=%s AND kind='execute' "
                "AND icp_position=ANY(%s)", (ROUND, BASELINE, sorted(failed)),
            )
        cursor.execute(
            "SELECT run_id,submission_id,stage,icp_position,attempt,kind,status,"
            "output_ref,terminal_cause FROM public.lab_arena_runs "
            "WHERE round_id=%s AND stage=1 AND kind='execute'", (ROUND,),
        )
        names = (
            "run_id", "submission_id", "stage", "icp_position", "attempt",
            "kind", "status", "output_ref", "terminal_cause",
        )
        plan = scoring.build_scoring_plan(
            round_id=ROUND, stage=1,
            runs=[dict(zip(names, record)) for record in cursor.fetchall()],
        )
        cursor.execute(
            "UPDATE public.lab_arena_rounds SET status='stage1_closed',"
            "status_generation=14,stage_generation=10,finalists=%s::jsonb,"
            "stage1_scoring_plan_doc=%s::jsonb WHERE round_id=%s",
            (json.dumps(finalists), json.dumps(plan), ROUND),
        )
        cursor.execute("ALTER TABLE public.lab_arena_rounds ENABLE TRIGGER USER")
        cursor.execute("ALTER TABLE public.lab_arena_runs ENABLE TRIGGER USER")
    items = []
    for index, planned in enumerate(plan["work_items"]):
        item = dict(planned)
        if item["submission_id"] == BASELINE:
            cache_key = "sha256:" + ("%064x" % (index + 100))
            input_hash = "sha256:" + ("%064x" % (index + 200))
            item.update(
                judgment_cache_key=cache_key,
                judgment_input_hash=input_hash,
                judgment_scope_doc={
                    "cache_key": cache_key,
                    "scoring_input_hash": input_hash,
                    "round_id": ROUND,
                    "network_name": "finney",
                    "netuid": 71,
                    "integrity_policy": "arena_integrity_v1",
                    "evaluation_date": "2026-09-16",
                    "scorer_image_digest": NEW_IMAGE,
                    "scorer_image_reference": "registry.example/native@" + NEW_IMAGE,
                },
                judgment_group_leader=True,
                judgment_group_miner_hotkeys=[hotkeys[0]],
            )
        items.append(item)
    connection.commit()
    return items


def _ready_for_publication(connection, hotkeys, baseline_score, *, failed_position=None):
    with connection.cursor() as cursor:
        cursor.execute("ALTER TABLE public.lab_arena_runs DISABLE TRIGGER USER")
        cursor.execute("ALTER TABLE public.lab_arena_rounds DISABLE TRIGGER USER")
        cursor.execute(
            "UPDATE public.lab_arena_runs SET status='accepted',per_icp_score=1.0,"
            "terminal_cause='accepted',output_ref='arena/output/' || run_id || '.json' "
            "WHERE round_id=%s AND submission_id=%s AND kind='execute' "
            "AND status='pending'",
            (ROUND, BASELINE),
        )
        if failed_position is not None:
            cursor.execute(
                "UPDATE public.lab_arena_runs SET status='failed',"
                "terminal_cause='model_error',output_ref=NULL,per_icp_score=0 "
                "WHERE round_id=%s AND submission_id=%s AND kind='execute' "
                "AND icp_position=%s", (ROUND, BASELINE, failed_position),
            )
        cursor.execute(
            "SELECT run_id,stage,icp_position FROM public.lab_arena_runs "
            "WHERE round_id=%s AND submission_id=%s AND kind='execute' "
            "AND status='accepted' "
            "ORDER BY icp_position", (ROUND, BASELINE),
        )
        for execute_id, stage, position in cursor.fetchall():
            assignment = "%s:%s:%d:%d:score:rerun265" % (
                ROUND, BASELINE, stage, position,
            )
            cursor.execute(
                "INSERT INTO public.lab_arena_runs (run_id,assignment_id,round_id,"
                "submission_id,miner_hotkey,stage,icp_position,attempt,kind,status,"
                "scored_run_id,terminal_cause,output_ref,runner_hotkey,stage_generation) "
                "VALUES (%s,%s,%s,%s,%s,%s,%s,1,'score','accepted',%s,'accepted',"
                "%s,%s,12)",
                (assignment + ":1", assignment, ROUND, BASELINE, hotkeys[0], stage,
                 position, execute_id, "arena/score/" + assignment + ".json", hotkeys[5]),
            )
        cursor.execute(
            "SELECT old_round_doc -> 'publication_doc' "
            "FROM public.lab_arena_sep16_baseline_rerun_audit WHERE round_id=%s",
            (ROUND,),
        )
        publication = cursor.fetchone()[0]
        baseline = next(
            row for row in publication["final_ranking"]
            if row["submission_id"] == BASELINE
        )
        baseline["final_score"] = baseline_score
        cursor.execute(
            "UPDATE public.lab_arena_rounds SET status='scored',status_generation=18,"
            "stage_generation=12,publication_doc=%s::jsonb WHERE round_id=%s",
            (json.dumps(publication), ROUND),
        )
        cursor.execute("ALTER TABLE public.lab_arena_rounds ENABLE TRIGGER USER")
        cursor.execute("ALTER TABLE public.lab_arena_runs ENABLE TRIGGER USER")
    connection.commit()
    return publication


def _insert_baseline_cost_head(
    connection, *, kind, entry_kind, identity_number, call_succeeded=None
):
    with connection.cursor() as cursor:
        cursor.execute(
            "SELECT run_id,miner_hotkey,stage FROM public.lab_arena_runs "
            "WHERE round_id=%s AND submission_id=%s AND kind=%s "
            "AND status='accepted' ORDER BY icp_position,attempt LIMIT 1",
            (ROUND, BASELINE, kind),
        )
        run_id, miner_hotkey, stage = cursor.fetchone()
        entry_doc = {}
        if entry_kind == "uncertain":
            entry_doc = {
                "reason": "worker_reported",
                "call": {"call_succeeded": call_succeeded},
            }
        cursor.execute(
            "INSERT INTO public.lab_arena_ledger (entry_kind,miner_hotkey,"
            "round_id,submission_id,run_id,stage,call_identity,provider,"
            "operation_id,amount_microusd,entry_doc) VALUES "
            "(%s,%s,%s,%s,%s,%s,%s,'openrouter','openrouter.responses',"
            "1000,%s::jsonb)",
            (
                entry_kind,
                miner_hotkey,
                ROUND,
                BASELINE,
                run_id,
                stage,
                "sha256:" + ("%064x" % identity_number),
                json.dumps(entry_doc),
            ),
        )


def _publication_preservation_state(connection):
    with connection.cursor() as cursor:
        cursor.execute(
            "SELECT jsonb_build_object("
            "'reward_basis_hash',reward_basis_hash,"
            "'reward_basis_doc',reward_basis_doc,"
            "'signing_key_doc',signing_key_doc,"
            "'reward_activated_at',reward_activated_at,"
            "'effective_reward_epoch',effective_reward_epoch,"
            "'baseline_submission',(SELECT to_jsonb(s) "
            "FROM public.lab_arena_submissions s "
            "WHERE s.round_id=r.round_id AND s.submission_id=%s),"
            "'challenger_submissions',(SELECT jsonb_agg(to_jsonb(s) "
            "ORDER BY submission_id) FROM public.lab_arena_submissions s "
            "WHERE s.round_id=r.round_id AND s.submission_id<>%s),"
            "'challenger_runs',(SELECT jsonb_agg(to_jsonb(x) ORDER BY run_id) "
            "FROM public.lab_arena_runs x WHERE x.round_id=r.round_id "
            "AND x.submission_id<>%s)) "
            "FROM public.lab_arena_rounds r WHERE round_id=%s",
            (BASELINE, BASELINE, BASELINE, ROUND),
        )
        return cursor.fetchone()[0]


def _baseline_ledger_state(connection):
    with connection.cursor() as cursor:
        cursor.execute(
            "SELECT jsonb_agg(to_jsonb(l) ORDER BY entry_id) "
            "FROM public.lab_arena_ledger l WHERE submission_id=%s",
            (BASELINE,),
        )
        return cursor.fetchone()[0]


def test_prepare_is_sealed_and_preserves_miners_rewards_and_history(connect):
    connection = connect()
    try:
        schedule, _hotkeys, ids = _seed_observed_sep16(connection)
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT relname,relrowsecurity FROM pg_catalog.pg_class WHERE relname IN ("
                "'lab_arena_sep16_baseline_rerun_audit',"
                "'lab_arena_sep16_rerun_release_authority') ORDER BY relname"
            )
            assert cursor.fetchall() == [
                ("lab_arena_sep16_baseline_rerun_audit", True),
                ("lab_arena_sep16_rerun_release_authority", True),
            ]
            for role in ("anon", "authenticated", "service_role", "lab_arena_service"):
                for table in (
                    "public.lab_arena_sep16_baseline_rerun_audit",
                    "public.lab_arena_sep16_rerun_release_authority",
                ):
                    cursor.execute(
                        "SELECT has_table_privilege(%s,%s,'SELECT,INSERT,UPDATE,DELETE')",
                        (role, table),
                    )
                    assert cursor.fetchone()[0] is False
            private_functions = (
                "public.lab_arena_open_scoring_sep16_baseline_only_v1(text,smallint,jsonb)",
                "public.lab_arena_sep16_rerun_score_namespace_guard_v1()",
                "public.lab_arena_sep16_challenger_seals_valid_v1()",
                "public.lab_arena_sep16_rerun_publication_guard_v1()",
            )
            service_functions = (
                "public.lab_arena_prepare_sep16_baseline_rerun_v1(bigint,text,text,text,jsonb)",
                "public.lab_arena_open_sep16_baseline_scoring_v1(text,smallint,jsonb)",
            )
            for role in ("anon", "authenticated", "service_role"):
                for function in private_functions + service_functions:
                    cursor.execute(
                        "SELECT has_function_privilege(%s,%s,'EXECUTE')",
                        (role, function),
                    )
                    assert cursor.fetchone()[0] is False
            for function in private_functions:
                cursor.execute(
                    "SELECT has_function_privilege('lab_arena_service',%s,'EXECUTE')",
                    (function,),
                )
                assert cursor.fetchone()[0] is False
            for function in service_functions:
                cursor.execute(
                    "SELECT has_function_privilege('lab_arena_service',%s,'EXECUTE')",
                    (function,),
                )
                assert cursor.fetchone()[0] is True
        with connection.cursor() as cursor:
            with pytest.raises(Exception):
                cursor.execute(
                    "SELECT public.lab_arena_prepare_sep16_baseline_rerun_v1("
                    "4096,%s,%s,%s,%s::jsonb)",
                    ("a" * 64, "b" * 40, BANK_HASH, json.dumps(schedule)),
                )
        connection.rollback()
        old_seal = _insert_authority(connection, schedule)
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT public.lab_arena_prepare_sep16_baseline_rerun_v1("
                "4096,%s,%s,%s,%s::jsonb)",
                ("a" * 64, "b" * 40, BANK_HASH, json.dumps(schedule)),
            )
            assert cursor.fetchone()[0]["status"] == "prepared"
            cursor.execute(
                "SELECT status,reward_basis_hash,king_outcome,promotion_required,"
                "promotion_doc,baseline_promoted_at,champion_submission_id,"
                "configuration_doc,participants FROM public.lab_arena_rounds "
                "WHERE round_id=%s", (ROUND,),
            )
            row = cursor.fetchone()
            assert row[:7] == ("stage1", BASIS_HASH, "no_king", True, None, None, None)
            configuration, participants = row[7:]
            assert configuration["baseline_source_url"] == (
                "https://github.com/leadpoet/champion_model/archive/refs/heads/lab.tar.gz"
            )
            assert configuration["scorer_image_digest"] == NEW_IMAGE
            assert configuration["intent_details_policy"] == "intent_details_v1"
            assert configuration.get("quality_policy") is None
            assert configuration["runner_slot_ceiling"] == 10
            assert len(participants) == 5
            assert [item["submission_id"] for item in participants[1:]] == ids[1:]
            cursor.execute(
                "SELECT count(*) FROM public.lab_arena_runs WHERE round_id=%s "
                "AND submission_id=%s AND status='pending'", (ROUND, BASELINE),
            )
            assert cursor.fetchone()[0] == 20
            cursor.execute(
                "SELECT count(*) FROM public.lab_arena_runs WHERE round_id=%s "
                "AND submission_id=%s",
                ("arena-2026-09-16-archive",
                 "baseline-2026-09-16-archive"),
            )
            assert cursor.fetchone()[0] == 40
            cursor.execute(
                "SELECT public.lab_arena_sep16_challenger_seals_valid_v1()"
            )
            assert cursor.fetchone()[0] is True
            cursor.execute(
                "SELECT public.lab_arena_prepare_sep16_baseline_rerun_v1("
                "4096,%s,%s,%s,%s::jsonb)",
                ("a" * 64, "b" * 40, BANK_HASH, json.dumps(schedule)),
            )
            assert cursor.fetchone()[0]["status"] == "existing"
            cursor.execute(
                "SELECT old_round_hash,old_challenger_runs_hash,old_actual_microusd "
                "FROM public.lab_arena_sep16_baseline_rerun_audit WHERE round_id=%s",
                (ROUND,),
            )
            assert cursor.fetchone() == (
                old_seal["round"], old_seal["challenger_runs"],
                old_seal["baseline_actual"],
            )
        connection.commit()
    finally:
        connection.close()


def test_owner266_seal_is_exact_idempotent_and_does_not_prepare(connect):
    connection = connect()
    try:
        schedule, _hotkeys, _ids = _seed_observed_sep16(connection)
        seal = _state_seal(connection)
        sql = _render_owner_seal(schedule, seal)
        with connection.cursor() as cursor:
            cursor.execute(sql)
            cursor.execute(sql)
            cursor.execute(
                "SELECT source_commit,champion_model_main_commit,"
                "champion_model_lab_commit,verified_parallel_runner_slots "
                "FROM public.lab_arena_sep16_rerun_release_authority"
            )
            assert cursor.fetchone() == ("b" * 40, "b" * 40, "b" * 40, 10)
            cursor.execute(
                "SELECT status FROM public.lab_arena_rounds WHERE round_id=%s", (ROUND,),
            )
            assert cursor.fetchone() == ("published",)
            cursor.execute(
                "SELECT count(*) FROM public.lab_arena_sep16_baseline_rerun_audit"
            )
            assert cursor.fetchone()[0] == 0
        connection.commit()
        changed = _render_owner_seal(schedule, seal, source_sha="c" * 64)
        with connection.cursor() as cursor:
            with pytest.raises(Exception):
                cursor.execute(changed)
        connection.rollback()
    finally:
        connection.close()


def test_stale_release_seal_cannot_partially_prepare(connect):
    connection = connect()
    try:
        schedule, _hotkeys, _ids = _seed_observed_sep16(connection)
        _insert_authority(connection, schedule)
        with connection.cursor() as cursor:
            cursor.execute(
                "UPDATE public.lab_arena_sep16_rerun_release_authority "
                "SET old_round_hash=%s WHERE round_id=%s",
                ("sha256:" + "0" * 64, ROUND),
            )
        connection.commit()
        with connection.cursor() as cursor:
            with pytest.raises(Exception):
                cursor.execute(
                    "SELECT public.lab_arena_prepare_sep16_baseline_rerun_v1("
                    "4096,%s,%s,%s,%s::jsonb)",
                    ("a" * 64, "b" * 40, BANK_HASH, json.dumps(schedule)),
                )
        connection.rollback()
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT status FROM public.lab_arena_rounds WHERE round_id=%s", (ROUND,),
            )
            assert cursor.fetchone() == ("published",)
            cursor.execute(
                "SELECT count(*) FROM public.lab_arena_rounds "
                "WHERE round_id='arena-2026-09-16-archive'"
            )
            assert cursor.fetchone()[0] == 0
            cursor.execute(
                "SELECT count(*) FROM public.lab_arena_sep16_baseline_rerun_audit"
            )
            assert cursor.fetchone()[0] == 0
    finally:
        connection.close()


def test_concurrent_prepare_serializes_to_one_archive(connect):
    seed = connect()
    try:
        schedule, _hotkeys, _ids = _seed_observed_sep16(seed)
        _insert_authority(seed, schedule)
    finally:
        seed.close()

    def prepare_once():
        connection = connect()
        try:
            with connection.cursor() as cursor:
                cursor.execute(
                    "SELECT public.lab_arena_prepare_sep16_baseline_rerun_v1("
                    "4096,%s,%s,%s,%s::jsonb)",
                    ("a" * 64, "b" * 40, BANK_HASH, json.dumps(schedule)),
                )
                status = cursor.fetchone()[0]["status"]
            connection.commit()
            return status
        finally:
            connection.close()

    with ThreadPoolExecutor(max_workers=2) as pool:
        statuses = sorted(pool.map(lambda _index: prepare_once(), range(2)))
    assert statuses == ["existing", "prepared"]
    verify = connect()
    try:
        with verify.cursor() as cursor:
            cursor.execute(
                "SELECT count(*) FROM public.lab_arena_rounds "
                "WHERE round_id='arena-2026-09-16-archive'"
            )
            assert cursor.fetchone()[0] == 1
            cursor.execute(
                "SELECT count(*) FROM public.lab_arena_runs WHERE round_id=%s "
                "AND submission_id=%s AND kind='execute'", (ROUND, BASELINE),
            )
            assert cursor.fetchone()[0] == 20
    finally:
        verify.close()


def test_scoring_accepts_recomputed_finalists_and_reuses_challenger_rows(connect):
    connection = connect()
    try:
        schedule, hotkeys, ids = _seed_observed_sep16(connection)
        assert _prepare(connection, schedule)["status"] == "prepared"
        recomputed_finalists = [ids[2], ids[4]]
        items = _stage1_scoring_items(connection, hotkeys, recomputed_finalists)
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT public.lab_arena_open_sep16_baseline_scoring_v1("
                "%s,1::smallint,%s::jsonb)", (ROUND, json.dumps(items)),
            )
            opened = cursor.fetchone()[0]
            assert opened["status"] == "ok"
            assert opened["round_status"] == "stage1_scoring"
            assert opened["assignments"] == 10
            cursor.execute(
                "SELECT finalists FROM public.lab_arena_rounds WHERE round_id=%s", (ROUND,),
            )
            assert cursor.fetchone()[0] == recomputed_finalists
            cursor.execute(
                "SELECT count(*) FROM public.lab_arena_runs WHERE round_id=%s "
                "AND submission_id=%s AND kind='score' "
                "AND assignment_id LIKE '%%:score:rerun265'", (ROUND, BASELINE),
            )
            assert cursor.fetchone()[0] == 10
            cursor.execute(
                "SELECT count(*) FROM public.lab_arena_runs WHERE round_id=%s "
                "AND submission_id<>%s", (ROUND, BASELINE),
            )
            assert cursor.fetchone()[0] == 160
        connection.commit()
    finally:
        connection.close()


def test_generic_scoring_route_is_blocked_after_prepare(connect):
    connection = connect()
    try:
        schedule, hotkeys, ids = _seed_observed_sep16(connection)
        assert _prepare(connection, schedule)["status"] == "prepared"
        items = _stage1_scoring_items(connection, hotkeys, [ids[1]])
        with connection.cursor() as cursor:
            with pytest.raises(Exception):
                cursor.execute(
                    "SELECT public.lab_arena_open_scoring_v2("
                    "%s,1::smallint,%s::jsonb)", (ROUND, json.dumps(items)),
                )
        connection.rollback()
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT count(*) FROM public.lab_arena_runs WHERE round_id=%s "
                "AND kind='score' AND assignment_id NOT LIKE '%%:score:rerun265' "
                "AND stage_generation=10", (ROUND,),
            )
            assert cursor.fetchone()[0] == 0
            cursor.execute(
                "SELECT public.lab_arena_open_sep16_baseline_scoring_v1("
                "%s,1::smallint,%s::jsonb)", (ROUND, json.dumps(items)),
            )
            assert cursor.fetchone()[0]["assignments"] == 10
        connection.commit()
    finally:
        connection.close()


def test_prepare_and_scoring_replay_count_logical_assignments_after_retries(connect):
    connection = connect()
    try:
        schedule, hotkeys, ids = _seed_observed_sep16(connection)
        assert _prepare(connection, schedule)["status"] == "prepared"
        with connection.cursor() as cursor:
            cursor.execute("ALTER TABLE public.lab_arena_runs DISABLE TRIGGER USER")
            cursor.execute(
                "SELECT run_id,assignment_id,stage,icp_position FROM public.lab_arena_runs "
                "WHERE round_id=%s AND submission_id=%s AND kind='execute' "
                "AND icp_position=0", (ROUND, BASELINE),
            )
            run_id, assignment, stage, position = cursor.fetchone()
            cursor.execute(
                "UPDATE public.lab_arena_runs SET status='failed',"
                "terminal_cause='model_error',per_icp_score=NULL WHERE run_id=%s",
                (run_id,),
            )
            cursor.execute(
                "INSERT INTO public.lab_arena_runs (run_id,assignment_id,round_id,"
                "submission_id,miner_hotkey,stage,icp_position,attempt,kind,status,"
                "terminal_cause,output_ref,per_icp_score,stage_generation) "
                "VALUES (%s,%s,%s,%s,%s,%s,%s,2,'execute','accepted','accepted',"
                "%s,1.0,9)",
                (assignment + ":2", assignment, ROUND, BASELINE, hotkeys[0], stage,
                 position, "arena/output/" + assignment + ":2.json"),
            )
            cursor.execute("ALTER TABLE public.lab_arena_runs ENABLE TRIGGER USER")
            cursor.execute(
                "SELECT public.lab_arena_prepare_sep16_baseline_rerun_v1("
                "4096,%s,%s,%s,%s::jsonb)",
                ("a" * 64, "b" * 40, BANK_HASH, json.dumps(schedule)),
            )
            assert cursor.fetchone()[0]["status"] == "existing"
        connection.commit()
        items = _stage1_scoring_items(connection, hotkeys, [ids[1]])
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT public.lab_arena_open_sep16_baseline_scoring_v1("
                "%s,1::smallint,%s::jsonb)", (ROUND, json.dumps(items)),
            )
            assert cursor.fetchone()[0]["assignments"] == 10
            cursor.execute("ALTER TABLE public.lab_arena_runs DISABLE TRIGGER USER")
            cursor.execute(
                "SELECT run_id,assignment_id,scored_run_id,stage,icp_position "
                "FROM public.lab_arena_runs WHERE round_id=%s AND submission_id=%s "
                "AND kind='score' AND icp_position=0 AND attempt=1",
                (ROUND, BASELINE),
            )
            score_id, score_assignment, scored_id, stage, position = cursor.fetchone()
            cursor.execute(
                "UPDATE public.lab_arena_runs SET status='failed',"
                "terminal_cause='judge_error' WHERE run_id=%s", (score_id,),
            )
            cursor.execute(
                "INSERT INTO public.lab_arena_runs (run_id,assignment_id,round_id,"
                "submission_id,miner_hotkey,stage,icp_position,attempt,kind,status,"
                "scored_run_id,terminal_cause,output_ref,runner_hotkey,stage_generation) "
                "VALUES (%s,%s,%s,%s,%s,%s,%s,2,'score','accepted',%s,'accepted',"
                "%s,%s,11)",
                (score_assignment + ":2", score_assignment, ROUND, BASELINE,
                 hotkeys[0], stage, position, scored_id,
                 "arena/score/" + score_assignment + ":2.json", hotkeys[5]),
            )
            cursor.execute("ALTER TABLE public.lab_arena_runs ENABLE TRIGGER USER")
            cursor.execute(
                "SELECT public.lab_arena_open_sep16_baseline_scoring_v1("
                "%s,1::smallint,%s::jsonb)", (ROUND, json.dumps(items)),
            )
            replay = cursor.fetchone()[0]
            assert replay["status"] == "existing"
            assert replay["assignments"] == 10
        connection.commit()
    finally:
        connection.close()


def test_scoring_refuses_a_plan_without_historical_challenger_judgment(connect):
    connection = connect()
    try:
        schedule, hotkeys, ids = _seed_observed_sep16(connection)
        with connection.cursor() as cursor:
            cursor.execute("ALTER TABLE public.lab_arena_runs DISABLE TRIGGER USER")
            cursor.execute(
                "DELETE FROM public.lab_arena_runs WHERE run_id=%s",
                (ROUND + ":" + ids[1] + ":1:0:score:1",),
            )
            cursor.execute("ALTER TABLE public.lab_arena_runs ENABLE TRIGGER USER")
        connection.commit()
        assert _prepare(connection, schedule)["status"] == "prepared"
        items = _stage1_scoring_items(connection, hotkeys, [ids[1]])
        with connection.cursor() as cursor:
            with pytest.raises(Exception):
                cursor.execute(
                    "SELECT public.lab_arena_open_sep16_baseline_scoring_v1("
                    "%s,1::smallint,%s::jsonb)", (ROUND, json.dumps(items)),
                )
        connection.rollback()
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT count(*) FROM public.lab_arena_runs WHERE round_id=%s "
                "AND submission_id=%s AND kind='score' "
                "AND assignment_id LIKE '%%:score:rerun265'", (ROUND, BASELINE),
            )
            assert cursor.fetchone()[0] == 0
    finally:
        connection.close()


def test_scoring_uses_normal_zero_row_for_one_failed_baseline_position(connect):
    connection = connect()
    try:
        schedule, hotkeys, ids = _seed_observed_sep16(connection)
        assert _prepare(connection, schedule)["status"] == "prepared"
        items = _stage1_scoring_items(
            connection, hotkeys, [ids[1]], failed_position=7
        )
        assert sum(item["submission_id"] == BASELINE for item in items) == 9
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT stage1_scoring_plan_doc -> 'zero_rows' "
                "FROM public.lab_arena_rounds WHERE round_id=%s", (ROUND,),
            )
            zero_rows = cursor.fetchone()[0]
            assert any(
                row["submission_id"] == BASELINE and row["icp_position"] == 7
                for row in zero_rows
            )
            cursor.execute(
                "SELECT public.lab_arena_open_sep16_baseline_scoring_v1("
                "%s,1::smallint,%s::jsonb)", (ROUND, json.dumps(items)),
            )
            opened = cursor.fetchone()[0]
            assert opened["status"] == "ok"
            assert opened["assignments"] == 9
    finally:
        connection.close()


def test_scoring_allows_all_baseline_positions_to_be_normal_zero_rows(connect):
    connection = connect()
    try:
        schedule, hotkeys, ids = _seed_observed_sep16(connection)
        assert _prepare(connection, schedule)["status"] == "prepared"
        items = _stage1_scoring_items(
            connection, hotkeys, [ids[1]], failed_positions=range(10)
        )
        assert all(item["submission_id"] != BASELINE for item in items)
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT public.lab_arena_open_sep16_baseline_scoring_v1("
                "%s,1::smallint,%s::jsonb)", (ROUND, json.dumps(items)),
            )
            opened = cursor.fetchone()[0]
            assert opened["status"] == "ok"
            assert opened["assignments"] == 0
    finally:
        connection.close()


@pytest.mark.parametrize("baseline_score", [0.0, 35.0])
def test_publication_guard_does_not_force_the_native_baseline_score(
    connect, baseline_score
):
    connection = connect()
    try:
        schedule, hotkeys, _ids = _seed_observed_sep16(connection)
        assert _prepare(connection, schedule)["status"] == "prepared"
        publication = _ready_for_publication(connection, hotkeys, baseline_score)
        with connection.cursor() as cursor:
            cursor.execute(
                "ALTER TABLE public.lab_arena_rounds DISABLE TRIGGER "
                "lab_arena_integrity_publication_guard"
            )
            cursor.execute(
                "UPDATE public.lab_arena_rounds SET status='published',"
                "status_generation=status_generation+1,published_at=now(),"
                "publication_doc=%s::jsonb WHERE round_id=%s RETURNING status",
                (json.dumps(publication), ROUND),
            )
            assert cursor.fetchone() == ("published",)
            cursor.execute(
                "ALTER TABLE public.lab_arena_rounds ENABLE TRIGGER "
                "lab_arena_integrity_publication_guard"
            )
        connection.commit()
    finally:
        connection.close()


def test_publication_guard_accepts_one_terminal_execution_zero_row(connect):
    connection = connect()
    try:
        schedule, hotkeys, _ids = _seed_observed_sep16(connection)
        assert _prepare(connection, schedule)["status"] == "prepared"
        publication = _ready_for_publication(
            connection, hotkeys, 35.0, failed_position=7
        )
        with connection.cursor() as cursor:
            cursor.execute(
                "ALTER TABLE public.lab_arena_rounds DISABLE TRIGGER "
                "lab_arena_integrity_publication_guard"
            )
            cursor.execute(
                "UPDATE public.lab_arena_rounds SET status='published',"
                "status_generation=status_generation+1,published_at=now(),"
                "publication_doc=%s::jsonb WHERE round_id=%s RETURNING status",
                (json.dumps(publication), ROUND),
            )
            assert cursor.fetchone() == ("published",)
            cursor.execute(
                "SELECT count(*) FILTER (WHERE status='accepted'),"
                "count(*) FILTER (WHERE status='failed') "
                "FROM public.lab_arena_runs WHERE round_id=%s AND submission_id=%s "
                "AND kind='execute' AND assignment_id LIKE '%%:rerun265'",
                (ROUND, BASELINE),
            )
            assert cursor.fetchone() == (19, 1)
            cursor.execute(
                "SELECT count(*) FROM public.lab_arena_runs WHERE round_id=%s "
                "AND submission_id=%s AND kind='score' AND status='accepted' "
                "AND assignment_id LIKE '%%:score:rerun265'", (ROUND, BASELINE),
            )
            assert cursor.fetchone()[0] == 19
            cursor.execute(
                "ALTER TABLE public.lab_arena_rounds ENABLE TRIGGER "
                "lab_arena_integrity_publication_guard"
            )
        connection.commit()
    finally:
        connection.close()


def test_publication_guard_rejects_terminal_execution_without_persisted_zero(connect):
    connection = connect()
    try:
        schedule, hotkeys, _ids = _seed_observed_sep16(connection)
        assert _prepare(connection, schedule)["status"] == "prepared"
        publication = _ready_for_publication(
            connection, hotkeys, 35.0, failed_position=7
        )
        with connection.cursor() as cursor:
            cursor.execute("ALTER TABLE public.lab_arena_runs DISABLE TRIGGER USER")
            cursor.execute(
                "UPDATE public.lab_arena_runs SET per_icp_score=NULL "
                "WHERE round_id=%s AND submission_id=%s AND kind='execute' "
                "AND icp_position=7", (ROUND, BASELINE),
            )
            cursor.execute("ALTER TABLE public.lab_arena_runs ENABLE TRIGGER USER")
            cursor.execute(
                "ALTER TABLE public.lab_arena_rounds DISABLE TRIGGER "
                "lab_arena_integrity_publication_guard"
            )
            with pytest.raises(Exception):
                cursor.execute(
                    "UPDATE public.lab_arena_rounds SET status='published',"
                    "status_generation=status_generation+1,published_at=now(),"
                    "publication_doc=%s::jsonb WHERE round_id=%s",
                    (json.dumps(publication), ROUND),
                )
        connection.rollback()
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT status FROM public.lab_arena_rounds WHERE round_id=%s", (ROUND,),
            )
            assert cursor.fetchone() == ("scored",)
    finally:
        connection.close()


def test_publication_guard_rejects_changed_miner_result(connect):
    connection = connect()
    try:
        schedule, hotkeys, _ids = _seed_observed_sep16(connection)
        assert _prepare(connection, schedule)["status"] == "prepared"
        publication = _ready_for_publication(connection, hotkeys, 35.0)
        challenger = next(
            row for row in publication["final_ranking"]
            if row["submission_id"] != BASELINE
        )
        challenger["final_score"] = 99.0
        with connection.cursor() as cursor:
            cursor.execute(
                "ALTER TABLE public.lab_arena_rounds DISABLE TRIGGER "
                "lab_arena_integrity_publication_guard"
            )
            with pytest.raises(Exception):
                cursor.execute(
                    "UPDATE public.lab_arena_rounds SET status='published',"
                    "status_generation=status_generation+1,published_at=now(),"
                    "publication_doc=%s::jsonb WHERE round_id=%s",
                    (json.dumps(publication), ROUND),
                )
        connection.rollback()
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT status FROM public.lab_arena_rounds WHERE round_id=%s", (ROUND,),
            )
            assert cursor.fetchone() == ("scored",)
    finally:
        connection.close()


def test_failed_cost_uncertainty_preserves_full_positive_publication(connect):
    connection = connect()
    try:
        migration = (
            Path(__file__).parents[2]
            / "scripts/267-arena-2026-09-16-failed-cost-publication.sql"
        )
        connection.autocommit = True
        with connection.cursor() as cursor:
            cursor.execute(migration.read_text(encoding="utf-8"))
            cursor.execute(migration.read_text(encoding="utf-8"))
            cursor.execute(
                "SELECT pg_get_functiondef(to_regprocedure("
                "'public.lab_arena_sep16_rerun_publication_guard_v1()'))"
            )
            definition = cursor.fetchone()[0]
        connection.autocommit = False
        assert "v_execute_cost ->> 'uncertain_calls'" not in definition
        assert "v_score_cost ->> 'uncertain_calls'" not in definition
        for counter in (
            "v_execute_cost ->> 'inflight_calls'",
            "v_execute_cost ->> 'success_unresolved_calls'",
            "v_score_cost ->> 'inflight_calls'",
            "v_score_cost ->> 'success_unresolved_calls'",
        ):
            assert counter in definition

        schedule, hotkeys, _ids = _seed_observed_sep16(connection)
        assert _prepare(connection, schedule)["status"] == "prepared"
        publication = _ready_for_publication(connection, hotkeys, 1.0)
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT count(DISTINCT icp_position),avg(per_icp_score),"
                "count(*) FILTER (WHERE status='accepted') "
                "FROM public.lab_arena_runs WHERE round_id=%s "
                "AND submission_id=%s AND kind='execute' "
                "AND assignment_id LIKE '%%:rerun265'",
                (ROUND, BASELINE),
            )
            position_count, aggregate, accepted_executions = cursor.fetchone()
            assert position_count == 20
            assert accepted_executions == 20
            assert float(aggregate) == 1.0
            cursor.execute(
                "SELECT count(*) FROM public.lab_arena_runs WHERE round_id=%s "
                "AND submission_id=%s AND kind='score' AND status='accepted' "
                "AND assignment_id LIKE '%%:score:rerun265'",
                (ROUND, BASELINE),
            )
            assert cursor.fetchone() == (20,)
            assert next(
                row["final_score"] for row in publication["final_ranking"]
                if row["submission_id"] == BASELINE
            ) == float(aggregate)
            cursor.execute(
                "ALTER TABLE public.lab_arena_rounds DISABLE TRIGGER "
                "lab_arena_integrity_publication_guard"
            )
        connection.commit()

        for index, (kind, entry_kind, succeeded) in enumerate((
            ("execute", "uncertain", True),
            ("score", "uncertain", True),
            ("execute", "reservation", None),
            ("score", "reservation", None),
        )):
            _insert_baseline_cost_head(
                connection,
                kind=kind,
                entry_kind=entry_kind,
                identity_number=91000 + index,
                call_succeeded=succeeded,
            )
            with connection.cursor() as cursor:
                with pytest.raises(Exception, match="conflicts with sealed"):
                    cursor.execute(
                        "UPDATE public.lab_arena_rounds SET status='published',"
                        "status_generation=status_generation+1,published_at=now(),"
                        "publication_doc=%s::jsonb WHERE round_id=%s",
                        (json.dumps(publication), ROUND),
                    )
            connection.rollback()
            with connection.cursor() as cursor:
                cursor.execute(
                    "SELECT status FROM public.lab_arena_rounds WHERE round_id=%s",
                    (ROUND,),
                )
                assert cursor.fetchone() == ("scored",)

        _insert_baseline_cost_head(
            connection,
            kind="execute",
            entry_kind="uncertain",
            identity_number=92001,
            call_succeeded=False,
        )
        _insert_baseline_cost_head(
            connection,
            kind="score",
            entry_kind="uncertain",
            identity_number=92002,
            call_succeeded=False,
        )
        connection.commit()
        with connection.cursor() as cursor:
            for kind in ("execute", "score"):
                cursor.execute(
                    "SELECT public.lab_arena__successful_call_cost_state(%s,%s,NULL)",
                    (BASELINE, kind),
                )
                state = cursor.fetchone()[0]
                assert state["uncertain_calls"] == 1
                assert state["success_unresolved_calls"] == 0
                assert state["inflight_calls"] == 0

        ledger_before = _baseline_ledger_state(connection)
        protected_before = _publication_preservation_state(connection)
        with connection.cursor() as cursor:
            cursor.execute(
                "UPDATE public.lab_arena_rounds SET status='published',"
                "status_generation=status_generation+1,published_at=now(),"
                "publication_doc=%s::jsonb WHERE round_id=%s RETURNING status",
                (json.dumps(publication), ROUND),
            )
            assert cursor.fetchone() == ("published",)
            cursor.execute(
                "ALTER TABLE public.lab_arena_rounds ENABLE TRIGGER "
                "lab_arena_integrity_publication_guard"
            )
        connection.commit()
        assert _baseline_ledger_state(connection) == ledger_before
        assert _publication_preservation_state(connection) == protected_before
    finally:
        connection.close()
