"""Disposable PostgreSQL proof for the exact measured Sep16 adoption RPC."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from bittensor_wallet import Keypair

from lab_arena import capacity, contact_policy, contracts, rewards, scoring
from lab_arena.service import ArenaService, DEFAULT_STAGE_MINUTES
from scripts import arena_sep15_exact_rerun as operator
from tests.lab_arena.lab_arena_pg_harness import (
    CURRENT_SERVICE_MIGRATIONS, database_with_lab_arena_migration,
)


ROUND = "arena-2026-09-16"
MIGRATIONS = CURRENT_SERVICE_MIGRATIONS + (
    "256-arena-2026-09-15-baseline-rerun.sql",
    "257-arena-2026-09-16-open-round-45m-adoption.sql",
    "259-arena-2026-09-16-measured-open-round-adoption.sql",
)


@pytest.fixture(scope="module")
def connect():
    generator = database_with_lab_arena_migration(MIGRATIONS)
    psycopg2, dsn = next(generator)
    try:
        yield lambda: psycopg2.connect(**dsn)
    finally:
        generator.close()


def _schedule(cutoff: datetime, minutes: dict[str, int]) -> dict[str, str]:
    service = object.__new__(ArenaService)
    service._config = SimpleNamespace(defaults=SimpleNamespace(stage_minutes=minutes))
    return service.build_schedule(cutoff)


def _round_configuration(hotkeys: list[str], schedule: dict[str, str]) -> dict:
    config = {
        "schema_version": contracts.ROUND_CONFIGURATION_SCHEMA_VERSION,
        "round_id": ROUND, "mode": "live", "rewards_enabled": True,
        "network_name": "finney", "netuid": 71, "schedule": schedule,
        "integrity_policy": "arena_integrity_v1",
        "contact_policy": "contacts_v1", "intent_details_policy": "intent_details_v1",
        "stage_1_icp_count": 10, "stage_2_icp_count": 10,
        "finalist_count": 10, "max_challengers": 20,
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
        "runner_hotkeys": [hotkeys[1]], "banned_hotkeys": [],
        "reward_constants": rewards.reward_constants_document(),
    }
    contracts.validate_round_configuration(config)
    return config


def _proof(config: dict, forward: dict[str, str], slots: int = 11) -> dict:
    proposed = dict(config,
        schedule=forward,
        icp_wall_clock_seconds=2700,
        lease_ttl_seconds=3600,
        runner_slot_ceiling=20,
        parallel_twenty_icp_execution=True,
        checkpoint_deadline_policy=contracts.CHECKPOINT_DEADLINE_POLICY,
    )
    contracts.validate_round_configuration(proposed)
    modeled = dict(proposed, runner_slot_ceiling=slots)
    return {
        "round_id": ROUND,
        "validated_by": "arena_service_capacity_v1",
        "parallel_twenty_icp_execution": True,
        "checkpoint_deadline_policy": contracts.CHECKPOINT_DEADLINE_POLICY,
        "icp_wall_clock_seconds": 2700,
        "runner_slot_ceiling": 20,
        "verified_parallel_runner_slots": slots,
        "configured_challenger_capacity": capacity.daily_challenger_capacity(modeled),
        "already_accepted_challengers": 5,
        "runner_hotkeys": proposed["runner_hotkeys"],
        "schedule": forward,
        "configuration_doc": proposed,
    }


def _insert_admission(cursor, submission_id: str, hotkey: str, index: int) -> None:
    claim = "sha256:" + format(index + 1, "064x")
    cursor.execute(
        "INSERT INTO public.lab_arena_submissions "
        "(submission_id,round_id,miner_hotkey,status,source_ref,source_size_bytes,"
        "code_review_status,code_review_attempts,code_review_doc,"
        "code_review_claim,code_review_started_at) "
        "VALUES (%s,%s,%s,'accepted',%s,4096,'passed',1,'{}'::jsonb,%s,now())",
        (submission_id, ROUND, hotkey,
         "arena/arena-2026-09-16/sources/%s.tar.gz" % submission_id, claim),
    )
    for kind in ("reservation", "dispatch", "settlement"):
        cursor.execute(
            "INSERT INTO public.lab_arena_ledger "
            "(entry_kind,miner_hotkey,round_id,submission_id,call_identity,"
            "provider,operation_id,funding_source,amount_microusd,entry_doc) "
            "VALUES (%s,%s,%s,%s,%s,'openrouter','openrouter.code_review',"
            "'miner_key',0,%s::jsonb)",
            (kind, hotkey, ROUND, submission_id, claim,
             json.dumps({"review_status": "passed"} if kind == "settlement" else {})),
        )


def _rpc(cursor, proof: dict) -> dict:
    cursor.execute(
        "SELECT public.lab_arena_adopt_sep16_open_config_v1(%s::jsonb)",
        (json.dumps(proof),),
    )
    return cursor.fetchone()[0]


def test_measured_sep16_rpc_preserves_completed_admissions_and_rejects_bad_state(connect):
    connection = connect()
    try:
        hotkeys = [Keypair.create_from_uri("//Sep16Measured%d" % n).ss58_address
                   for n in range(9)]
        now = datetime.now(timezone.utc)
        cutoff = datetime.combine((now + timedelta(days=1)).date(),
                                  datetime.min.time(), tzinfo=timezone.utc)
        original = _schedule(cutoff, dict(DEFAULT_STAGE_MINUTES))
        forward = _schedule(cutoff, {
            **DEFAULT_STAGE_MINUTES,
            "stage_1": 1012, "stage_1_scoring": 192,
            "stage_2": 1, "final_scoring": 192,
        })
        forward["publication_deadline"] = (
            cutoff + timedelta(hours=23, minutes=55)
        ).strftime("%Y-%m-%dT%H:%M:%SZ")
        config = _round_configuration(hotkeys, original)
        proof = _proof(config, forward)
        assert proof["configured_challenger_capacity"] == 5
        with connection.cursor() as cursor:
            cursor.execute("ALTER TABLE public.lab_arena_submissions DISABLE TRIGGER USER")
            cursor.execute("ALTER TABLE public.lab_arena_ledger DISABLE TRIGGER USER")
            cursor.execute(
                "INSERT INTO public.lab_arena_rounds "
                "(round_id,configuration_doc,rewards_enabled) VALUES (%s,%s::jsonb,true)",
                (ROUND, json.dumps(config)),
            )
            sibling = dict(config, round_id="arena-2026-09-14")
            cursor.execute(
                "INSERT INTO public.lab_arena_rounds "
                "(round_id,status,configuration_doc,rewards_enabled,"
                "stage1_scoring_plan_doc,promotion_doc) "
                "VALUES ('arena-2026-09-14','published',%s::jsonb,true,"
                "'{\"marker\":\"scoring\"}'::jsonb,'{\"marker\":\"promotion\"}'::jsonb)",
                (json.dumps(sibling),),
            )
            for index in range(5):
                _insert_admission(cursor, "sep16-miner-%d" % index,
                                  hotkeys[index + 2], index)
            cursor.execute("ALTER TABLE public.lab_arena_submissions ENABLE TRIGGER USER")
            cursor.execute("ALTER TABLE public.lab_arena_ledger ENABLE TRIGGER USER")
            cursor.execute(
                "SELECT to_jsonb(round) FROM public.lab_arena_rounds AS round "
                "WHERE round_id=%s", (ROUND,),
            )
            original_round = cursor.fetchone()[0]
            cursor.execute(
                "SELECT to_jsonb(round) FROM public.lab_arena_rounds AS round "
                "WHERE round_id='arena-2026-09-14'",
            )
            sibling_round = cursor.fetchone()[0]

            def rejected(mutator, expected: str, candidate: dict = proof) -> None:
                cursor.execute("SAVEPOINT bad_adoption")
                try:
                    mutator()
                    with pytest.raises(Exception, match=expected):
                        _rpc(cursor, candidate)
                finally:
                    cursor.execute("ROLLBACK TO SAVEPOINT bad_adoption")

            def missing_settlement():
                cursor.execute("ALTER TABLE public.lab_arena_ledger DISABLE TRIGGER USER")
                cursor.execute(
                    "DELETE FROM public.lab_arena_ledger WHERE round_id=%s "
                    "AND submission_id='sep16-miner-0' AND entry_kind='settlement'",
                    (ROUND,),
                )

            rejected(missing_settlement, "completed triples")

            def extra_ledger():
                cursor.execute(
                    "INSERT INTO public.lab_arena_ledger "
                    "(entry_kind,miner_hotkey,round_id,submission_id,call_identity,"
                    "provider,operation_id,funding_source,amount_microusd) "
                    "VALUES ('refusal',%s,%s,'sep16-miner-0',%s,'openrouter',"
                    "'openrouter.code_review','miner_key',0)",
                    (hotkeys[2], ROUND, "sha256:" + format(1, "064x")),
                )

            rejected(extra_ledger, "completed triples")

            def uncertain_ledger():
                cursor.execute("ALTER TABLE public.lab_arena_ledger DISABLE TRIGGER USER")
                cursor.execute(
                    "UPDATE public.lab_arena_ledger SET entry_kind='uncertain' "
                    "WHERE round_id=%s AND submission_id='sep16-miner-0' "
                    "AND entry_kind='settlement'", (ROUND,),
                )

            rejected(uncertain_ledger, "completed triples")

            def non_admission_ledger():
                cursor.execute("ALTER TABLE public.lab_arena_ledger DISABLE TRIGGER USER")
                cursor.execute(
                    "UPDATE public.lab_arena_ledger "
                    "SET operation_id='openrouter.evaluation' WHERE round_id=%s "
                    "AND submission_id='sep16-miner-0'", (ROUND,),
                )

            rejected(non_admission_ledger, "completed triples")
            insufficient = _proof(config, forward, slots=10)
            rejected(lambda: None, "accepted workload exceeds measured", insufficient)
            for field in (
                "submission_open", "submission_cutoff",
                "benchmark_deadline", "stage_1_start",
            ):
                changed_intake = dict(proof, schedule=dict(forward,
                    **{field: "2026-09-16T00:01:00Z"}))
                rejected(lambda: None, "immutable intake or start", changed_intake)
            changed_count = dict(proof, already_accepted_challengers=4)
            rejected(lambda: None, "accepted submissions changed", changed_count)
            changed_scoring = dict(proof, configuration_doc=dict(
                proof["configuration_doc"], scorer_policy={"tampered": True},
            ))
            rejected(lambda: None, "proof config differs", changed_scoring)

            def later_completed_admission():
                cursor.execute(
                    "ALTER TABLE public.lab_arena_submissions "
                    "DISABLE TRIGGER lab_arena_submissions_owner_admission"
                )
                _insert_admission(cursor, "sep16-miner-later", hotkeys[7], 7)

            rejected(later_completed_admission, "accepted workload exceeds measured",
                     dict(proof, already_accepted_challengers=6))

            def progressed_round():
                cursor.execute("ALTER TABLE public.lab_arena_rounds DISABLE TRIGGER USER")
                cursor.execute(
                    "UPDATE public.lab_arena_rounds SET status_generation=1 "
                    "WHERE round_id=%s", (ROUND,),
                )

            rejected(progressed_round, "open-only state differs")

            result = _rpc(cursor, proof)
            assert result["status"] == "adopted"
            assert result["admission_ledger_rows_preserved"] == 15
            assert result["verified_parallel_runner_slots"] == 11
            assert result["already_accepted_challengers"] == 5
            assert result["future_admissions_may_exceed_measured_capacity"] is True
            cursor.execute(
                "SELECT to_jsonb(round) FROM public.lab_arena_rounds AS round "
                "WHERE round_id=%s", (ROUND,),
            )
            adopted_round = cursor.fetchone()[0]
            assert adopted_round["configuration_doc"] == proof["configuration_doc"]
            assert adopted_round["configuration_doc"]["runner_slot_ceiling"] == 20
            assert adopted_round["configuration_doc"]["max_challengers"] == 20
            adopted_round.pop("configuration_doc")
            adopted_round.pop("updated_at")
            original_round.pop("configuration_doc")
            original_round.pop("updated_at")
            assert adopted_round == original_round
            cursor.execute(
                "SELECT old_round_doc,new_configuration_doc,capacity_doc "
                "FROM public.lab_arena_sep16_open_config_audit WHERE round_id=%s",
                (ROUND,),
            )
            old_audit, new_audit, proof_audit = cursor.fetchone()
            assert old_audit["configuration_doc"] == config
            assert new_audit == proof["configuration_doc"]
            assert proof_audit == proof
            cursor.execute(
                "SELECT to_jsonb(round) FROM public.lab_arena_rounds AS round "
                "WHERE round_id='arena-2026-09-14'",
            )
            assert cursor.fetchone()[0] == sibling_round
            cursor.execute("SELECT count(*) FROM public.lab_arena_ledger WHERE round_id=%s", (ROUND,))
            assert cursor.fetchone()[0] == 15
            assert _rpc(cursor, proof)["status"] == "existing"
            rejected(lambda: None, "replay differs",
                     dict(proof, verified_parallel_runner_slots=12))

            # Bypass only the unrelated owner-authorization fixture guard;
            # leave the submission-capacity trigger active for this check.
            cursor.execute(
                "ALTER TABLE public.lab_arena_submissions "
                "DISABLE TRIGGER lab_arena_submissions_owner_admission"
            )
            cursor.execute(
                "INSERT INTO public.lab_arena_submissions "
                "(submission_id,round_id,miner_hotkey,status,source_ref,source_size_bytes) "
                "VALUES ('sep16-later-miner',%s,%s,'accepted',"
                "'arena/arena-2026-09-16/sources/later.tar.gz',4096)",
                (ROUND, hotkeys[7]),
            )
            cursor.execute(
                "ALTER TABLE public.lab_arena_submissions "
                "ENABLE TRIGGER lab_arena_submissions_owner_admission"
            )
            cursor.execute(
                "SELECT count(*) FROM public.lab_arena_submissions "
                "WHERE round_id=%s AND status='accepted'", (ROUND,),
            )
            assert cursor.fetchone()[0] == 6
            assert _rpc(cursor, proof)["status"] == "existing"
    finally:
        connection.rollback()
        connection.close()


def test_operator_measures_physical_slots_and_preserves_intake_file(tmp_path):
    hotkeys = [Keypair.create_from_uri("//Sep16Operator%d" % n).ss58_address
               for n in range(8)]
    cutoff = datetime.combine((datetime.now(timezone.utc) + timedelta(days=1)).date(),
                              datetime.min.time(), tzinfo=timezone.utc)
    original = _schedule(cutoff, dict(DEFAULT_STAGE_MINUTES))
    forward = _schedule(cutoff, {
        **DEFAULT_STAGE_MINUTES,
        "stage_1": 1012, "stage_1_scoring": 192,
        "stage_2": 1, "final_scoring": 192,
    })
    forward["publication_deadline"] = (
        cutoff + timedelta(hours=23, minutes=55)
    ).strftime("%Y-%m-%dT%H:%M:%SZ")
    path = tmp_path / "sep16-forward.json"
    path.write_text(json.dumps(forward), encoding="utf-8")
    config = _round_configuration(hotkeys, original)
    fake_store = SimpleNamespace(
        get_round=lambda _round: {"status": "open", "configuration_doc": config},
        list_submissions=lambda _round, **_kw: [
            {"submission_id": "miner-%d" % n} for n in range(5)
        ],
    )
    result = operator._adopt_sep16(SimpleNamespace(store=fake_store), SimpleNamespace(
        forward_schedule_file=path, verified_parallel_runner_slots=11,
        dry_run=True,
    ))
    assert result["configured_challenger_capacity"] == 5
    assert result["first_phase_retry_reserved_waves_at_verified_slots"] == 22
    assert result["scoring_retry_reserved_waves_per_stage"] == 12
    assert result["max_challengers"] == 20
    assert result["future_admissions_may_exceed_measured_capacity"] is True
    with pytest.raises(operator.ExactRerunRefused,
                       match="accepted parallel workload exceeds the frozen stage1 window"):
        operator._adopt_sep16(SimpleNamespace(store=fake_store), SimpleNamespace(
            forward_schedule_file=path, verified_parallel_runner_slots=10,
            dry_run=True,
        ))
