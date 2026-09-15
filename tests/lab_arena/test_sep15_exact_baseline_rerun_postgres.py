"""Disposable PostgreSQL proof for the exact Sep15 state override."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest
from bittensor_wallet import Keypair

from lab_arena import capacity, contact_policy, contracts, rewards, scoring
from tests.lab_arena.lab_arena_pg_harness import (
    CURRENT_SERVICE_MIGRATIONS, database_with_lab_arena_migration,
)

MIGRATIONS = CURRENT_SERVICE_MIGRATIONS + (
    "256-arena-2026-09-15-baseline-rerun.sql",
    "257-arena-2026-09-16-open-round-45m-adoption.sql",
)
ROUND = "arena-2026-09-15"
BASELINE = "baseline-2026-09-15"
BASIS_HASH = "sha256:502cd6a5234e5680cfdd761a49b5d825d2064ff3fb063fc8fe7d7f1755e96a18"
BANK_HASH = "4e11123c9098bbba977fdae9e531419a6dea40cf76dacde1c4c83a82df0a95c2"


@pytest.fixture(scope="module")
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
        "submission_open": "2026-09-14T00:00:00Z",
        "submission_cutoff": "2026-09-15T00:00:00Z",
        "benchmark_deadline": "2026-09-15T00:30:00Z",
        "stage_1_start": "2026-09-15T00:31:00Z",
        "stage_1_close": future(hours=2),
        "stage_1_scoring_close": future(hours=3),
        "stage_2_start": future(hours=3, minutes=1),
        "stage_2_close": future(hours=4),
        "final_scoring_close": future(hours=5),
        "publication_deadline": future(hours=5, minutes=1),
    }


def _seed_observed_sep15(connection):
    hotkeys = [Keypair.create_from_uri("//Sep15Proof%d" % n).ss58_address
               for n in range(19)]
    ids = [BASELINE] + ["miner-2026-09-15-%d" % n for n in range(1, 10)]
    schedule = _schedule()
    config = {
        "schema_version": contracts.ROUND_CONFIGURATION_SCHEMA_VERSION,
        "round_id": ROUND, "mode": "live", "rewards_enabled": True,
        "network_name": "finney", "netuid": 71,
        "schedule": schedule,
        "integrity_policy": "arena_integrity_v1",
        "contact_policy": "contacts_v1",
        "stage_1_icp_count": 10, "stage_2_icp_count": 10,
        "finalist_count": 10, "max_challengers": 15,
        "runner_slot_ceiling": 8, "max_attempts_per_assignment": 2,
        "lease_ttl_seconds": 1200, "companies_per_icp": 5,
        "providers": list(contracts.PROVIDERS),
        "call_quotas": dict(contracts.CALL_QUOTAS_PER_ICP),
        "scoring_call_quotas": dict(contracts.SCORING_CALL_QUOTAS_PER_WORK_ITEM),
        "icp_wall_clock_seconds": 300, "scoring_wall_clock_seconds": 900,
        "scorer_policy": scoring.build_scorer_policy(
            scoring_adapter_version=contact_policy.SCORING_ADAPTER
        ),
        "execution_cap_microusd": 80_000_000,
        "cost_per_company_microusd": 800_000,
        "scoring_cap_microusd": 50_000_000,
        "scorer_image_digest": "sha256:" + "a" * 64,
        "scorer_image_reference": "registry.example/scorer@sha256:" + "a" * 64,
        "baseline_hotkey": hotkeys[0],
        "baseline_source_url": "https://github.com/leadpoet/pydantic-harness/archive/refs/heads/lab.tar.gz",
        "runner_hotkeys": hotkeys[10:12], "banned_hotkeys": [],
        "reward_constants": rewards.reward_constants_document(),
    }
    contracts.validate_round_configuration(config)
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
             "final_score": 0.0}
            for index, submission_id in enumerate(ids[:7])
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
            "%s,%s,%s,%s::jsonb,%s::jsonb,'no_king',25183,%s,%s::jsonb,%s::jsonb,"
            "%s,%s)",
            (ROUND, json.dumps(config), json.dumps(participants),
             "arena/arena-2026-09-15/benchmark.json", "2026-09-15",
             "2026-09-14", json.dumps(ids[1:]), json.dumps(publication),
             BASIS_HASH, json.dumps({"round_id": ROUND,
                                    "king_outcome": "no_king",
                                    "effective_reward_epoch": 25183,
                                    "published_at": "2026-09-15T02:52:46Z"}),
             json.dumps({}), "2026-09-15T02:52:49Z",
             "2026-09-15T02:52:46Z"),
        )
        for index, submission_id in enumerate(ids):
            cursor.execute(
                "INSERT INTO public.lab_arena_submissions (submission_id,round_id,"
                "miner_hotkey,status,is_king,source_ref,source_size_bytes,consent,"
                "submission_doc,frozen_at) VALUES (%s,%s,%s,'frozen',%s,%s,%s,"
                "'{\"public_rerun\":true}'::jsonb,'{}'::jsonb,now())",
                (submission_id, ROUND, hotkeys[index], index == 0,
                 participants[index]["source_ref"],
                 participants[index]["source_size_bytes"]),
            )
        for n in range(9):
            cursor.execute(
                "INSERT INTO public.lab_arena_submissions (submission_id,round_id,"
                "miner_hotkey,status,submission_doc) VALUES (%s,%s,%s,'rejected','{}'::jsonb)",
                ("rejected-2026-09-15-%d" % n, ROUND, hotkeys[10 + n]),
            )
        accepted_challenger = []
        for index, submission_id in enumerate(ids):
            for position in range(20):
                stage = 1 if position < 10 else 2
                execute_id = "old-execute-%d-%d" % (index, position)
                failed = index == 9 and position < 6
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
                    continue
                if index > 0:
                    accepted_challenger.append((execute_id, submission_id,
                                                hotkeys[index], stage, position))
                score_id = "old-score-%d-%d" % (index, position)
                score_failed = index > 0 and len(accepted_challenger) <= 10
                cursor.execute(
                    "INSERT INTO public.lab_arena_runs (run_id,assignment_id,round_id,"
                    "submission_id,miner_hotkey,stage,icp_position,attempt,kind,status,"
                    "scored_run_id,terminal_cause,output_ref,runner_hotkey,"
                    "judgment_cache_source_run_id) VALUES (%s,%s,%s,%s,%s,%s,%s,1,'score',"
                    "%s,%s,%s,%s,%s,%s)",
                    (score_id, score_id, ROUND, submission_id, hotkeys[index],
                     stage, position, "failed" if score_failed else "accepted",
                     execute_id, "judge_error" if score_failed else "accepted",
                     "arena/score/%s.json" % score_id, hotkeys[10],
                     "old-score-0-0" if 0 < len(accepted_challenger) <= 79 else None),
                )
        assert len(accepted_challenger) == 174
        cursor.execute(
            "INSERT INTO public.lab_arena_ledger (entry_kind,miner_hotkey,round_id,"
            "submission_id,run_id,amount_microusd) VALUES ('settlement',%s,%s,%s,%s,14071603)",
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
                         "source_runner_hotkey": hotkeys[10],
                         "runner_authority_exclusions": [hotkeys[10]]}),
             "old-score-0-0", "old-execute-0-0", hotkeys[10]),
        )
        for table in ("lab_arena_ledger", "lab_arena_runs",
                      "lab_arena_submissions", "lab_arena_rounds"):
            cursor.execute("ALTER TABLE public.%s ENABLE TRIGGER USER" % table)
    connection.commit()
    return schedule, hotkeys, ids


def test_exact_sep15_prepare_archives_old_evidence_and_reuses_challengers(connect):
    connection = connect()
    try:
        schedule, hotkeys, ids = _seed_observed_sep15(connection)
        with connection.cursor() as cursor:
            with pytest.raises(Exception):
                cursor.execute(
                    "SELECT public.lab_arena_prepare_sep15_baseline_rerun_v1("
                    "4096,%s,%s,%s,%s::jsonb)",
                    ("a" * 64, "b" * 40, BANK_HASH, json.dumps(schedule)),
                )
        connection.rollback()
        with connection.cursor() as cursor:
            cursor.execute(
                "INSERT INTO public.lab_arena_sep15_rerun_release_authority "
                "(round_id,source_ref,source_size_bytes,source_sha256,"
                "source_commit,bank_sha256,verified_parallel_runner_slots,"
                "forward_schedule) "
                "VALUES (%s,%s,4096,%s,%s,%s,11,%s::jsonb)",
                (ROUND,
                 "arena/arena-2026-09-15/sources/baseline-2026-09-15-rerun256.tar.gz",
                 "a" * 64, "b" * 40, BANK_HASH, json.dumps(schedule)),
            )
        connection.commit()
        with connection.cursor() as cursor:
            with pytest.raises(Exception):
                cursor.execute(
                    "SELECT public.lab_arena_prepare_sep15_baseline_rerun_v1("
                    "4096,%s,%s,%s,%s::jsonb)",
                    ("a" * 64, "b" * 40, "0" * 64, json.dumps(schedule)),
                )
        connection.rollback()
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT public.lab_arena_prepare_sep15_baseline_rerun_v1("
                "4096,%s,%s,%s,%s::jsonb)",
                ("a" * 64, "b" * 40, BANK_HASH, json.dumps(schedule)),
            )
            assert cursor.fetchone()[0]["status"] == "prepared"
            cursor.execute(
                "SELECT round_id,status,reward_basis_hash,king_outcome "
                "FROM public.lab_arena_rounds WHERE round_id=%s", (ROUND,)
            )
            assert cursor.fetchone() == (ROUND, "stage1", BASIS_HASH, "no_king")
            cursor.execute(
                "SELECT configuration_doc ->> 'checkpoint_deadline_policy' "
                "FROM public.lab_arena_rounds WHERE round_id=%s", (ROUND,)
            )
            assert cursor.fetchone()[0] == contracts.CHECKPOINT_DEADLINE_POLICY
            cursor.execute(
                "SELECT round_id FROM public.lab_arena_reward_basis_v1 WHERE round_id=%s",
                (ROUND,),
            )
            assert cursor.fetchone() == (ROUND,)
            cursor.execute(
                "SELECT count(*) FROM public.lab_arena_runs "
                "WHERE round_id=%s AND submission_id=%s",
                ("arena-2026-09-15-archive", "baseline-2026-09-15-archive"),
            )
            assert cursor.fetchone()[0] == 40
            cursor.execute(
                "SELECT sum(amount_microusd) FROM public.lab_arena_ledger "
                "WHERE round_id=%s AND submission_id=%s",
                ("arena-2026-09-15-archive", "baseline-2026-09-15-archive"),
            )
            assert cursor.fetchone()[0] == 14071603
            cursor.execute(
                "SELECT count(*) FROM public.lab_arena_runs "
                "WHERE round_id=%s AND submission_id=%s AND status='pending'",
                (ROUND, BASELINE),
            )
            assert cursor.fetchone()[0] == 20
            cursor.execute(
                "SELECT count(*) FROM public.lab_arena_runs "
                "WHERE round_id=%s AND submission_id<>%s",
                (ROUND, BASELINE),
            )
            assert cursor.fetchone()[0] == 354
            cursor.execute(
                "SELECT rounds.round_id FROM public.lab_arena_judgment_cache AS cache "
                "JOIN public.lab_arena_runs AS rounds "
                "ON rounds.run_id=cache.source_score_run_id"
            )
            assert cursor.fetchone()[0] == "arena-2026-09-15-archive"
            cursor.execute(
                "SELECT public.lab_arena_prepare_sep15_baseline_rerun_v1("
                "4096,%s,%s,%s,%s::jsonb)",
                ("a" * 64, "b" * 40, BANK_HASH, json.dumps(schedule)),
            )
            assert cursor.fetchone()[0]["status"] == "existing"
            # Simulate ten accepted fresh baseline executions. The existing
            # service will derive this full plan from the same immutable runs.
            cursor.execute("ALTER TABLE public.lab_arena_runs DISABLE TRIGGER USER")
            cursor.execute("ALTER TABLE public.lab_arena_rounds DISABLE TRIGGER USER")
            cursor.execute(
                "UPDATE public.lab_arena_runs SET status='accepted', "
                "terminal_cause='accepted', output_ref='arena/output/' || run_id || '.json' "
                "WHERE round_id=%s AND submission_id=%s AND kind='execute'",
                (ROUND, BASELINE),
            )
            cursor.execute(
                "SELECT run_id,submission_id,stage,icp_position,attempt,kind,status,"
                "output_ref,terminal_cause FROM public.lab_arena_runs "
                "WHERE round_id=%s AND stage=1 AND kind='execute'",
                (ROUND,),
            )
            names = ("run_id", "submission_id", "stage", "icp_position",
                     "attempt", "kind", "status", "output_ref", "terminal_cause")
            plan = scoring.build_scoring_plan(
                round_id=ROUND, stage=1,
                runs=[dict(zip(names, record)) for record in cursor.fetchall()],
            )
            cursor.execute(
                "UPDATE public.lab_arena_rounds SET status='stage1_closed', "
                "status_generation=14,stage_generation=10,stage1_scoring_plan_doc=%s::jsonb "
                "WHERE round_id=%s",
                (json.dumps(plan), ROUND),
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
                            "network_name": "finney", "netuid": 71,
                            "integrity_policy": "arena_integrity_v1",
                            "evaluation_date": "2026-09-15",
                            "scorer_image_digest": "sha256:" + "a" * 64,
                            "scorer_image_reference":
                                "registry.example/scorer@sha256:" + "a" * 64,
                        },
                        judgment_group_leader=True,
                        judgment_group_miner_hotkeys=[hotkeys[0]],
                    )
                items.append(item)
            cursor.execute(
                "SELECT public.lab_arena_open_sep15_baseline_scoring_v1(%s,1::smallint,%s::jsonb)",
                (ROUND, json.dumps(items)),
            )
            opened = cursor.fetchone()[0]
            assert opened["status"] == "ok" and opened["assignments"] == 10
            cursor.execute(
                "SELECT count(*) FROM public.lab_arena_runs WHERE round_id=%s "
                "AND submission_id=%s AND kind='score' AND status='pending'",
                (ROUND, BASELINE),
            )
            assert cursor.fetchone()[0] == 10
            cursor.execute(
                "SELECT count(*) FROM public.lab_arena_runs WHERE round_id=%s "
                "AND submission_id<>%s AND kind='score'",
                (ROUND, BASELINE),
            )
            assert cursor.fetchone()[0] == 174
    finally:
        connection.close()


def test_sep16_adoption_changes_only_open_execution_limits(connect):
    connection = connect()
    try:
        hotkeys = [Keypair.create_from_uri("//Sep16Proof%d" % n).ss58_address
                   for n in range(5)]
        now = datetime(2026, 9, 16, 1, tzinfo=timezone.utc)

        def at(hours=0):
            return (now + timedelta(hours=hours)).strftime("%Y-%m-%dT%H:%M:%SZ")

        schedule = {
            "submission_open": "2026-09-15T00:00:00Z",
            "submission_cutoff": "2026-09-16T00:00:00Z",
            "benchmark_deadline": at(), "stage_1_start": at(1),
            "stage_1_close": at(7), "stage_1_scoring_close": at(9),
            "stage_2_start": at(10), "stage_2_close": at(16),
            "final_scoring_close": at(18), "publication_deadline": at(19),
        }
        config = {
            "schema_version": contracts.ROUND_CONFIGURATION_SCHEMA_VERSION,
            "round_id": "arena-2026-09-16", "mode": "live", "rewards_enabled": True,
            "network_name": "finney", "netuid": 71, "schedule": schedule,
            "integrity_policy": "arena_integrity_v1",
            "contact_policy": "contacts_v1",
            "intent_details_policy": "intent_details_v1",
            "stage_1_icp_count": 10, "stage_2_icp_count": 10,
            "finalist_count": 10, "max_challengers": 15,
            "runner_slot_ceiling": 8, "max_attempts_per_assignment": 2,
            "lease_ttl_seconds": 1200, "companies_per_icp": 5,
            "providers": list(contracts.PROVIDERS),
            "call_quotas": dict(contracts.CALL_QUOTAS_PER_ICP),
            "scoring_call_quotas": dict(contracts.SCORING_CALL_QUOTAS_PER_WORK_ITEM),
            "icp_wall_clock_seconds": 300, "scoring_wall_clock_seconds": 900,
            "scorer_policy": scoring.build_scorer_policy(
                scoring_adapter_version=contact_policy.SCORING_ADAPTER,
                intent_details=True,
            ),
            "execution_cap_microusd": 80_000_000,
            "cost_per_company_microusd": 800_000,
            "scoring_cap_microusd": 50_000_000,
            "scorer_image_digest": "sha256:" + "a" * 64,
            "scorer_image_reference": "registry.example/scorer@sha256:" + "a" * 64,
            "baseline_hotkey": hotkeys[0],
            "baseline_source_url": "https://github.com/leadpoet/pydantic-harness/archive/refs/heads/lab.tar.gz",
            "runner_hotkeys": hotkeys[1:3], "banned_hotkeys": [],
            "reward_constants": rewards.reward_constants_document(),
        }
        contracts.validate_round_configuration(config)
        proposed = dict(config, icp_wall_clock_seconds=2700,
                        lease_ttl_seconds=3600, runner_slot_ceiling=20,
                        parallel_twenty_icp_execution=True,
                        checkpoint_deadline_policy=contracts.CHECKPOINT_DEADLINE_POLICY)
        contracts.validate_round_configuration(proposed)
        supported = capacity.daily_challenger_capacity(proposed)
        assert supported >= 1
        proof = {
            "round_id": "arena-2026-09-16",
            "validated_by": "arena_service_capacity_v1",
            "parallel_twenty_icp_execution": True,
            "checkpoint_deadline_policy": contracts.CHECKPOINT_DEADLINE_POLICY,
            "icp_wall_clock_seconds": 2700,
            "runner_slot_ceiling": 20,
            "configured_challenger_capacity": supported,
            "runner_hotkeys": proposed["runner_hotkeys"],
            "schedule": proposed["schedule"],
            "configuration_doc": proposed,
        }
        with connection.cursor() as cursor:
            cursor.execute(
                "INSERT INTO public.lab_arena_rounds "
                "(round_id,configuration_doc,rewards_enabled) "
                "VALUES (%s,%s::jsonb,true)",
                ("arena-2026-09-16", json.dumps(config)),
            )
            cursor.execute("ALTER TABLE public.lab_arena_submissions DISABLE TRIGGER USER")
            for index in range(3):
                cursor.execute(
                    "INSERT INTO public.lab_arena_submissions "
                    "(submission_id,round_id,miner_hotkey,status,source_ref,"
                    "source_size_bytes) VALUES (%s,%s,%s,'accepted',%s,4096)",
                    ("sep16-miner-%d" % index, "arena-2026-09-16",
                     hotkeys[index + 1],
                     "arena/arena-2026-09-16/sources/sep16-miner-%d.tar.gz" % index),
                )
            cursor.execute("ALTER TABLE public.lab_arena_submissions ENABLE TRIGGER USER")
        connection.commit()
        with connection.cursor() as cursor:
            with pytest.raises(Exception):
                bad = dict(proof, configuration_doc=dict(proposed,
                    scorer_policy={"tampered": True}))
                cursor.execute(
                    "SELECT public.lab_arena_adopt_sep16_open_config_v1(%s::jsonb)",
                    (json.dumps(bad),),
                )
        connection.rollback()
        with connection.cursor() as cursor:
            with pytest.raises(Exception):
                forged = dict(proof, configured_challenger_capacity=supported + 1)
                cursor.execute(
                    "SELECT public.lab_arena_adopt_sep16_open_config_v1(%s::jsonb)",
                    (json.dumps(forged),),
                )
        connection.rollback()
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT public.lab_arena_adopt_sep16_open_config_v1(%s::jsonb)",
                (json.dumps(proof),),
            )
            assert cursor.fetchone()[0]["status"] == "adopted"
            cursor.execute(
                "SELECT configuration_doc,status,status_generation,stage_generation "
                "FROM public.lab_arena_rounds WHERE round_id=%s",
                ("arena-2026-09-16",),
            )
            adopted, status, status_generation, stage_generation = cursor.fetchone()
            assert adopted == proposed
            assert (status, status_generation, stage_generation) == ("open", 0, 0)
            cursor.execute(
                "SELECT count(*) FROM public.lab_arena_submissions WHERE round_id=%s",
                ("arena-2026-09-16",),
            )
            assert cursor.fetchone()[0] == 3
            cursor.execute(
                "SELECT public.lab_arena_adopt_sep16_open_config_v1(%s::jsonb)",
                (json.dumps(proof),),
            )
            assert cursor.fetchone()[0]["status"] == "existing"
    finally:
        connection.close()
