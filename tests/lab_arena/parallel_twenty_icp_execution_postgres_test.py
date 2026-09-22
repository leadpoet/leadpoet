"""Opt-in twenty-ICP execution scheduling on the current PostgreSQL schema."""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from lab_arena import contact_policy, contracts
from lab_arena.service import ServiceError
from lab_arena.proxy_workers import (
    ProxyWorkerPool,
    VerifiedProxyWorker,
    VerifiedProxyWorkerInventory,
)
from lab_arena.store import ArenaStoreError, hash_lease_token
from tests.lab_arena.test_lab_arena_migration_postgres import claim, complete, sha
from tests.lab_arena.test_lab_arena_service_round import (
    Harness,
    connect,
    database,
    keypair,
)

MIGRATION = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "255-lab-arena-parallel-twenty-icp-execution.sql"
)
ISOLATION_MIGRATION = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "356-lab-arena-deadline-provider-retry-isolation.sql"
)


def _verified_test_pool(process_capacity: int) -> ProxyWorkerPool:
    workers = tuple(
        VerifiedProxyWorker(
            slot_index=index + 1,
            source_name="test_proxy_%d" % (index + 1),
            proxy_url="http://worker:secret@proxy-%d.example:8080" % index,
            proxy_fingerprint="proxy%011d" % index,
            exit_ip="8.8.4.%d" % (index + 1),
            exit_ip_fingerprint="exit%012d" % index,
        )
        for index in range(process_capacity - 1)
    )
    return ProxyWorkerPool(
        VerifiedProxyWorkerInventory(
            workers=workers,
            native_exit_ip="1.1.1.1",
            native_exit_ip_fingerprint="nativeexit000000",
        )
    )


def _start_parallel_round(
    harness: Harness, round_id: str, *, slot_ceiling: int
) -> list[dict]:
    harness.service.config.defaults = replace(
        harness.service.config.defaults,
        runner_slot_ceiling=slot_ceiling,
        parallel_twenty_icp_execution=True,
    )
    harness.chain.epoch += int(round_id[-2:], 16)
    # Enter the review window after replacement freezes, as production does.
    cutoff = datetime.now(timezone.utc) + timedelta(minutes=30)
    configuration = harness.service.create_round(cutoff, round_id=round_id)
    assert configuration["runner_slot_ceiling"] == slot_ceiling
    assert configuration["parallel_twenty_icp_execution"] is True
    harness.round_id = round_id
    for flavor in harness.challengers:
        harness.submit(flavor, round_id)
    harness.clock.advance_to(harness.schedule()["submission_cutoff"])
    assert harness.service.advance_round(round_id)["status"] == "ok"
    participants = harness.service.store.get_round(round_id)["participants"]
    for participant in participants:
        harness.flavors.setdefault(participant["submission_id"], "PublicBaseline")
    harness.clock.advance_to(harness.schedule()["stage_1_start"])
    opened = harness.service.advance_round(round_id)
    assert opened["status"] == "ok"
    assert opened["assignments"] == len(participants) * contracts.BENCHMARK_ICP_COUNT
    return participants


def test_parallel_execution_migration_is_idempotent_and_keeps_score_guard(
    connect,
):
    connection = connect()
    connection.autocommit = True
    try:
        with connection.cursor() as cursor:
            signature = (
                "public.lab_arena_claim_assignment(text,text,integer,integer,"
                "text[],text,text,text,integer)"
            )
            cursor.execute("SELECT pg_get_functiondef(%s::regprocedure)", (signature,))
            before = cursor.fetchone()[0]
            cursor.execute(MIGRATION.read_text(encoding="utf-8"))
            cursor.execute("SELECT pg_get_functiondef(%s::regprocedure)", (signature,))
            after = cursor.fetchone()[0]
            cursor.execute(
                "SELECT public.lab_arena_parallel_execution_schema_v1()"
            )
            capability = cursor.fetchone()[0]
            cursor.execute(
                "SELECT p.provolatile, p.prosecdef "
                "FROM pg_catalog.pg_proc AS p "
                "WHERE p.oid='public.lab_arena_parallel_execution_schema_v1()'"
                "::pg_catalog.regprocedure"
            )
            volatility, security_definer = cursor.fetchone()
            privileges = {}
            for role in (
                "lab_arena_service",
                "anon",
                "authenticated",
                "service_role",
            ):
                cursor.execute(
                    "SELECT has_function_privilege(%s, "
                    "'public.lab_arena_parallel_execution_schema_v1()', "
                    "'EXECUTE')",
                    (role,),
                )
                privileges[role] = cursor.fetchone()[0]
        assert after == before
        for marker in (
            "lab_arena_parallel_twenty_icp_execution",
            "lab_arena_score_submission_serialization",
            "lab_arena_closed_scoring_reservation_claim",
            "company_judgment_cache",
            "champion_funding_sources",
        ):
            assert marker in after
        assert capability == {
            "schema_version": "leadpoet.lab_arena.parallel_execution_schema.v1",
            "version": 255,
            "max_parallel_icps": 20,
        }
        assert (volatility, security_definer) == ("s", True)
        assert privileges == {
            "lab_arena_service": True,
            "anon": False,
            "authenticated": False,
            "service_role": False,
        }
        # Migration 255 intentionally proves its own replay. Restore the
        # current close contract for the shared module database.
        with connection.cursor() as cursor:
            cursor.execute(ISOLATION_MIGRATION.read_text(encoding="utf-8"))
    finally:
        connection.close()


def test_parallel_default_startup_requires_the_255_capability(connect, tmp_path):
    harness = Harness(connect, tmp_path, challengers=[], runners=["alpha"])
    observed = []
    real_capability = harness.service.store.parallel_execution_schema

    def checked_capability():
        observed.append("checked")
        return real_capability()

    harness.service.store.parallel_execution_schema = checked_capability
    harness.service.startup_checks()
    assert observed == []

    harness.service.config.defaults = replace(
        harness.service.config.defaults,
        parallel_twenty_icp_execution=True,
    )
    harness.service.startup_checks()
    assert observed == ["checked"]

    def unavailable_capability():
        raise ArenaStoreError("function is unavailable")

    harness.service.store.parallel_execution_schema = unavailable_capability
    with pytest.raises(ServiceError) as caught:
        harness.service.startup_checks()
    assert caught.value.code == "parallel_execution_schema_unavailable"


def test_ten_slot_round_has_a_strict_ten_plus_ten_barrier_and_retry_recovery(
    connect, tmp_path
):
    harness = Harness(connect, tmp_path, challengers=[], runners=["alpha", "beta"])
    participants = _start_parallel_round(
        harness, "arena-2099-01-01-d10", slot_ceiling=10
    )
    assert len(participants) == 1
    store = harness.service.store
    first_runner, second_runner = harness.runner_keys

    runner_key = keypair("svc-runner-alpha")
    old_claim = contracts.build_signed_request(
        scope=contracts.SCOPE_CLAIM, round_id=harness.round_id,
        hotkey=runner_key.ss58_address, body={"declared_parallelism": 8},
        timestamp=int(harness.clock().timestamp()),
        sign_message=lambda message: runner_key.sign(message.encode()).hex(),
    )
    with pytest.raises(ServiceError) as rejected:
        harness.service.handle_claim(old_claim)
    assert rejected.value.code == "validator_proxy_execution_upgrade_required"
    assert all(run["status"] == "pending" for run in store.list_runs(harness.round_id))
    first_claim = contracts.build_signed_request(
        scope=contracts.SCOPE_CLAIM,
        round_id=harness.round_id,
        hotkey=runner_key.ss58_address,
        body={"declared_parallelism": 10, "proxy_execution_version": contracts.PROXY_EXECUTION_VERSION},
        timestamp=int(harness.clock().timestamp()),
        sign_message=lambda message: runner_key.sign(message.encode()).hex(),
    )
    first_lease = harness.service.handle_claim(first_claim)
    assert first_lease["parallel_twenty_icp_execution"] is True
    first_batch = [(first_lease, first_lease["lease_token"])] + [
        claim(
            store,
            harness.round_id,
            first_runner,
            parallelism=10,
            ceiling=10,
        )[:2]
        for _ in range(9)
    ]
    assert [leased["icp_position"] for leased, _token in first_batch] == list(range(10))
    blocked, *_ = claim(
        store,
        harness.round_id,
        second_runner,
        parallelism=20,
        ceiling=10,
    )
    assert blocked["status"] == "no_pending"

    for leased, token in first_batch[:-1]:
        assert (
            complete(
                store,
                leased["run_id"],
                hash_lease_token(token),
                "accepted",
                output_ref="arena/test/outputs/%s.json" % leased["run_id"],
            )["status"]
            == "accepted"
        )
    failed, failed_token = first_batch[-1]
    failure = complete(
        store,
        failed["run_id"],
        hash_lease_token(failed_token),
        "provider_error",
    )
    assert failure["status"] == "failed"
    assert failure["confirmation_attempt"] == 2

    retry, retry_token, *_ = claim(
        store,
        harness.round_id,
        second_runner,
        parallelism=20,
        ceiling=10,
    )
    assert (retry["stage"], retry["icp_position"], retry["attempt"]) == (1, 9, 2)
    assert store.get_run(retry["run_id"])["runner_hotkey"] == second_runner
    assert (
        complete(
            store,
            retry["run_id"],
            hash_lease_token(retry_token),
            "accepted",
            output_ref="arena/test/outputs/%s.json" % retry["run_id"],
        )["status"]
        == "accepted"
    )

    second_batch, *_ = claim(
        store,
        harness.round_id,
        first_runner,
        parallelism=10,
        ceiling=10,
    )
    assert (second_batch["stage"], second_batch["icp_position"]) == (2, 10)
    with pytest.raises(Exception, match="parallel_execution_capacity_invalid"):
        claim(
            store,
            harness.round_id,
            first_runner,
            parallelism=20,
            ceiling=20,
        )


def test_twenty_slot_round_allows_two_validator_identities_independently(
    connect, tmp_path
):
    harness = Harness(connect, tmp_path, challengers=[], runners=["alpha", "beta"])
    _start_parallel_round(harness, "arena-2099-01-01-d20", slot_ceiling=20)
    store = harness.service.store
    first_runner, second_runner = harness.runner_keys

    first_identity = [
        claim(
            store,
            harness.round_id,
            first_runner,
            parallelism=10,
            ceiling=20,
        )[0]
        for _ in range(10)
    ]
    full, *_ = claim(
        store,
        harness.round_id,
        first_runner,
        parallelism=10,
        ceiling=20,
    )
    assert full == {"status": "no_free_slot", "active_leases": 10, "slot_limit": 10}

    second_identity = [
        claim(
            store,
            harness.round_id,
            second_runner,
            parallelism=20,
            ceiling=20,
        )[0]
        for _ in range(10)
    ]
    assert {run["icp_position"] for run in first_identity + second_identity} == set(
        range(20)
    )
    assert {
        store.get_run(run["run_id"])["runner_hotkey"] for run in first_identity
    } == {first_runner}
    assert {
        store.get_run(run["run_id"])["runner_hotkey"] for run in second_identity
    } == {second_runner}
    drained, *_ = claim(
        store,
        harness.round_id,
        second_runner,
        parallelism=20,
        ceiling=20,
    )
    assert drained["status"] == "no_pending"


def test_parallel_execution_preserves_standard_twenty_icp_scoring_and_costs(
    connect, tmp_path
):
    harness = Harness(
        connect, tmp_path, challengers=["ParallelMiner"], runners=["alpha", "beta"]
    )
    participants = _start_parallel_round(
        harness, "arena-2099-01-01-e20", slot_ceiling=20
    )
    pending = harness.service.store.list_runs(harness.round_id, kind="execute")
    assert len(pending) == len(participants) * contracts.BENCHMARK_ICP_COUNT
    assert all(run["status"] == "pending" for run in pending)
    assert {run["stage"] for run in pending} == {1, 2}

    original_runner = harness.runner

    def runner_with_verified_pool(index, parallel=4):
        runner = original_runner(index, parallel)
        runner._config.proxy_worker_pool = _verified_test_pool(parallel)
        return runner

    harness.runner = runner_with_verified_pool

    harness.advance_until("published", runners=2, max_steps=100)
    published = harness.service.store.get_round(harness.round_id)
    assert published["status"] == "published"
    execute_runs = harness.service.store.list_runs(harness.round_id, kind="execute")
    score_runs = harness.service.store.list_runs(harness.round_id, kind="score")
    assert all(run["status"] == "accepted" for run in execute_runs + score_runs)
    assert len(execute_runs) == len(participants) * 20
    assert len(score_runs) == len(participants) * 20

    for participant in participants:
        public = harness.service.public_results(
            harness.round_id, participant["submission_id"]
        )
        rows = public["scores"]["stage_1"] + public["scores"]["stage_2"]
        assert len(rows) == 20
        assert {row["icp_position"] for row in rows} == set(range(20))
        costs = harness.service.store.submission_costs(participant["submission_id"])
        assert costs["submission_id"] == participant["submission_id"]
        assert sum(row["inflight_calls"] for row in costs["providers"]) == 0
        assert sum(row["uncertain_calls"] for row in costs["providers"]) == 0


def test_exhausted_provider_error_in_stage_two_scores_zero_and_publishes(
    connect, tmp_path
):
    failed_position = 10
    harness = Harness(connect, tmp_path, challengers=[], runners=["alpha"])
    participants = _start_parallel_round(
        harness, "arena-2099-01-01-e21", slot_ceiling=20
    )
    assert len(participants) == 1
    baseline_id = participants[0]["submission_id"]
    original_runner = harness.runner
    store = harness.service.store
    runner_hotkey = harness.runner_keys[0]
    icps = harness.service.evaluation_icps(harness.round_id)
    output_schema_version = contact_policy.output_schema(
        store.get_round(harness.round_id)["configuration_doc"]
    )
    for expected_position in range(10):
        lease, token, *_ = claim(
            store,
            harness.round_id,
            runner_hotkey,
            parallelism=1,
            ceiling=20,
        )
        assert lease["icp_position"] == expected_position
        icp = icps[expected_position]
        flavor = "PublicBaseline"
        companies = [
            {
                "company_name": "%s Company %d" % (flavor, index),
                "company_website": "https://%s-%d.example.com"
                % (flavor.lower(), index),
                "company_linkedin": "",
                "industry": icp["industry"],
                "employee_count": icp["employee_count"][0],
                "company_stage": str(icp.get("company_stage") or ""),
                "country": icp.get("country") or "United States",
                "state": "",
                "fit_summary": "The company matches the ICP.",
                "fit_evidence_urls": [
                    "https://%s-%d.example.com/about" % (flavor.lower(), index)
                ],
                "intent_signals": [
                    {
                        "description": "Raised a round",
                        "url": "https://news.example.com/%s/%d" % (flavor, index),
                        "date": "2026-08-01",
                        "why_now": "The funding makes outreach timely.",
                        "snippet": "Funding announced",
                        "matched_icp_signal": 0,
                    }
                ],
            }
            for index in range(5)
        ]
        output_ref = "arena/test/outputs/%s.json" % lease["run_id"]
        harness.objects.put(
            output_ref,
            json.dumps(
                {
                    "schema_version": output_schema_version,
                    "companies": companies,
                }
            ).encode("utf-8"),
        )
        assert complete(
            store,
            lease["run_id"],
            hash_lease_token(token),
            "accepted",
            output_ref=output_ref,
        )["status"] == "accepted"

    first_failure, first_token, *_ = claim(
        store,
        harness.round_id,
        runner_hotkey,
        parallelism=1,
        ceiling=20,
    )
    assert (first_failure["icp_position"], first_failure["attempt"]) == (
        failed_position,
        1,
    )
    assert complete(
        store,
        first_failure["run_id"],
        hash_lease_token(first_token),
        "provider_error",
    )["confirmation_attempt"] == 2
    second_failure, second_token, *_ = claim(
        store,
        harness.round_id,
        runner_hotkey,
        parallelism=1,
        ceiling=20,
    )
    assert (second_failure["icp_position"], second_failure["attempt"]) == (
        failed_position,
        2,
    )
    assert complete(
        store,
        second_failure["run_id"],
        hash_lease_token(second_token),
        "provider_error",
    )["status"] == "failed"

    def runner_with_verified_pool(index, parallel=4):
        runner = original_runner(index, parallel)
        runner._config.proxy_worker_pool = _verified_test_pool(parallel)
        return runner

    harness.runner = runner_with_verified_pool
    harness.advance_until("published", runners=1, max_steps=100)

    published = harness.service.store.get_round(harness.round_id)
    assert published["status"] == "published"
    assert published["stage2_scoring_plan_doc"]["zero_rows"] == [
        {
            "submission_id": baseline_id,
            "icp_position": failed_position,
            "cause": "provider_error",
        }
    ]
    execute_runs = harness.service.store.list_runs(
        harness.round_id, submission_id=baseline_id, kind="execute"
    )
    failed_runs = [
        run for run in execute_runs if run["icp_position"] == failed_position
    ]
    accepted_runs = [run for run in execute_runs if run["status"] == "accepted"]
    assert len(execute_runs) == 21
    assert len(accepted_runs) == 19
    assert [run["attempt"] for run in failed_runs] == [1, 2]
    assert all(
        run["status"] == "failed"
        and run["terminal_cause"] == "provider_error"
        and run["output_ref"] is None
        for run in failed_runs
    )
    assert failed_runs[0]["per_icp_score"] is None
    assert failed_runs[1]["per_icp_score"] == 0
    costed_accepted = [
        run for run in accepted_runs if run["icp_position"] > failed_position
    ]
    assert len(costed_accepted) == 9
    assert all(
        harness.service.store.list_ledger(run_id=run["run_id"])
        for run in costed_accepted
    )

    score_runs = harness.service.store.list_runs(
        harness.round_id, submission_id=baseline_id, kind="score"
    )
    assert len(score_runs) == 19
    assert all(run["status"] == "accepted" for run in score_runs)
    costs = harness.service.store.submission_costs(baseline_id)
    assert sum(row["inflight_calls"] for row in costs["providers"]) == 0
    assert sum(row["uncertain_calls"] for row in costs["providers"]) == 0
    result = published["publication_doc"]["final_ranking"][0]
    selected_scores = [
        float(run["per_icp_score"])
        for run in accepted_runs + [failed_runs[1]]
    ]
    assert result["final_score"] == pytest.approx(sum(selected_scores) / 20)
    assert result["final_score"] > 0

    before_round = harness.service.store.get_round(harness.round_id)
    before_runs = harness.service.store.list_runs(harness.round_id)
    before_ledger = harness.service.store.list_ledger(submission_id=baseline_id)
    connection = connect()
    connection.autocommit = True
    try:
        with connection.cursor() as cursor:
            migration = ISOLATION_MIGRATION.read_text(encoding="utf-8")
            cursor.execute(migration)
            cursor.execute(migration)
            cursor.execute(
                "SELECT pg_catalog.pg_get_functiondef("
                "'public.lab_arena_close_stage(text,smallint)'::regprocedure),"
                "pg_catalog.pg_get_functiondef("
                "'public.lab_arena_close_parallel_execution_v1(text)'::regprocedure),"
                "has_function_privilege('lab_arena_service',"
                "'public.lab_arena_close_stage(text,smallint)','EXECUTE'),"
                "has_function_privilege('anon',"
                "'public.lab_arena_close_stage(text,smallint)','EXECUTE')"
            )
            sequential, parallel, service_execute, anon_execute = cursor.fetchone()
    finally:
        connection.close()
    assert "latest_status = 'failed'" in sequential
    assert "latest_cause = 'provider_error'" in sequential
    assert "latest_status = 'failed'" in parallel
    assert "latest_cause = 'provider_error'" in parallel
    assert (service_execute, anon_execute) == (True, False)
    assert harness.service.store.get_round(harness.round_id) == before_round
    assert harness.service.store.list_runs(harness.round_id) == before_runs
    assert harness.service.store.list_ledger(submission_id=baseline_id) == before_ledger


def test_dispatched_provider_retry_closed_by_deadline_scores_zero_and_publishes(
    connect, tmp_path
):
    """Use an accelerated test clock; production stage budgets stay unchanged."""

    failed_position = 10
    harness = Harness(connect, tmp_path, challengers=[], runners=["alpha", "beta"])
    participants = _start_parallel_round(
        harness,
        "arena-2099-01-01-e22",
        slot_ceiling=20,
    )
    baseline_id = participants[0]["submission_id"]
    store = harness.service.store
    runner_hotkey = harness.runner_keys[0]
    icps = harness.service.evaluation_icps(harness.round_id)
    output_schema_version = contact_policy.output_schema(
        store.get_round(harness.round_id)["configuration_doc"]
    )

    def accept_position(expected_position, worker_hotkey, *, parallelism=1):
        lease, token, *_ = claim(
            store,
            harness.round_id,
            worker_hotkey,
            parallelism=parallelism,
            ceiling=20,
        )
        assert lease["icp_position"] == expected_position
        companies = [
            {
                "company_name": "Deadline Company %d" % index,
                "company_website": "https://deadline-%d.example.com" % index,
                "company_linkedin": "",
                "industry": icps[expected_position]["industry"],
                "employee_count": icps[expected_position]["employee_count"][0],
                "company_stage": str(
                    icps[expected_position].get("company_stage") or ""
                ),
                "country": icps[expected_position].get("country")
                or "United States",
                "state": "",
                "fit_summary": "The company matches the ICP.",
                "fit_evidence_urls": [
                    "https://deadline-%d.example.com/about" % index
                ],
                "intent_signals": [
                    {
                        "description": "Raised a round",
                        "url": "https://news.example.com/deadline/%d" % index,
                        "date": "2026-08-01",
                        "why_now": "The funding makes outreach timely.",
                        "snippet": "Funding announced",
                        "matched_icp_signal": 0,
                    }
                ],
            }
            for index in range(5)
        ]
        output_ref = "arena/test/outputs/%s.json" % lease["run_id"]
        harness.objects.put(
            output_ref,
            json.dumps(
                {
                    "schema_version": output_schema_version,
                    "companies": companies,
                }
            ).encode("utf-8"),
        )
        assert complete(
            store,
            lease["run_id"],
            hash_lease_token(token),
            "accepted",
            output_ref=output_ref,
        )["status"] == "accepted"

    for expected_position in range(10):
        accept_position(expected_position, runner_hotkey)

    first_failure, first_token, *_ = claim(
        store,
        harness.round_id,
        runner_hotkey,
        parallelism=1,
        ceiling=20,
    )
    assert (first_failure["icp_position"], first_failure["attempt"]) == (
        failed_position,
        1,
    )
    assert complete(
        store,
        first_failure["run_id"],
        hash_lease_token(first_token),
        "provider_error",
    )["confirmation_attempt"] == 2

    retry, retry_token, *_ = claim(
        store,
        harness.round_id,
        runner_hotkey,
        parallelism=1,
        ceiling=20,
    )
    assert (retry["icp_position"], retry["attempt"]) == (failed_position, 2)
    retry_token_hash = hash_lease_token(retry_token)
    call_identity = contracts.provider_call_identity(
        attempt=2,
        assignment_id=retry["assignment_id"],
        icp_position=failed_position,
        action_sequence=0,
        operation_id="openrouter.responses",
        request_hash=sha("deadline-provider-retry"),
    )
    assert store.reserve_call(
        run_id=retry["run_id"],
        lease_token_hash=retry_token_hash,
        call_identity=call_identity,
        operation_id="openrouter.responses",
        provider="openrouter",
        funding_source="host",
        amount_microusd=1_000,
        call_doc={"request_hash": sha("deadline-provider-retry")},
    )["status"] == "reserved"
    assert store.mark_dispatched(
        run_id=retry["run_id"],
        lease_token_hash=retry_token_hash,
        call_identity=call_identity,
    )["status"] == "dispatched"

    # Finish all other assignments, then advance the fake clock to the close.
    # The dispatched retry remains leased and is closed by PostgreSQL.
    for expected_position in range(11, 20):
        accept_position(expected_position, runner_hotkey, parallelism=20)
    harness.clock.advance_to(harness.schedule()["stage_1_close"])
    closed = harness.service.advance_round(harness.round_id)
    assert closed["status"] == "ok"
    assert harness.status() == "stage1_closed"

    original_runner = harness.runner

    def runner_with_verified_pool(index, parallel=4):
        runner = original_runner(index, parallel)
        runner._config.proxy_worker_pool = _verified_test_pool(parallel)
        return runner

    harness.runner = runner_with_verified_pool
    harness.advance_until("published", runners=2, max_steps=100)

    published = store.get_round(harness.round_id)
    assert published["status"] == "published"
    assert published["stage2_scoring_plan_doc"]["zero_rows"] == [
        {
            "submission_id": baseline_id,
            "icp_position": failed_position,
            "cause": "provider_error",
        }
    ]
    failed_runs = [
        run
        for run in store.list_runs(
            harness.round_id, submission_id=baseline_id, kind="execute"
        )
        if run["icp_position"] == failed_position
    ]
    assert [run["attempt"] for run in failed_runs] == [1, 2]
    assert failed_runs[0]["terminal_cause"] == "provider_error"
    assert failed_runs[1]["terminal_cause"] == "stage_closed"
    assert failed_runs[1]["terminal_doc"]["previous_status"] == "leased"
    assert (
        failed_runs[1]["terminal_doc"]["deadline_provider_retry_exhausted"]
        is True
    )
    assert failed_runs[0]["per_icp_score"] is None
    assert failed_runs[1]["per_icp_score"] == 0
    assert [
        row["entry_kind"] for row in store.list_ledger(run_id=retry["run_id"])
    ] == ["reservation", "dispatch", "uncertain"]

    accepted_runs = [
        run
        for run in store.list_runs(
            harness.round_id, submission_id=baseline_id, kind="execute"
        )
        if run["status"] == "accepted"
    ]
    assert len(accepted_runs) == 19
    result = published["publication_doc"]["final_ranking"][0]
    selected_scores = [
        float(run["per_icp_score"])
        for run in accepted_runs + [failed_runs[1]]
    ]
    assert result["final_score"] == pytest.approx(sum(selected_scores) / 20)
    assert result["final_score"] > 0


def _seed_deadline_close_guard(
    connection,
    round_id: str,
    *,
    retry_status: str,
    prior_cause: str,
    ledger_kinds: tuple[str, ...],
    include_second_assignment: bool = True,
    round_status: str = "stage1",
) -> None:
    submission_id = round_id + "-submission"
    miner_hotkey = keypair(round_id + "-miner").ss58_address
    runner_hotkey = keypair(round_id + "-runner").ss58_address
    configuration = {
        "parallel_twenty_icp_execution": True,
        "stage_1_icp_count": 1,
        "stage_2_icp_count": 1,
    }
    participants = [
        {"submission_id": submission_id, "miner_hotkey": miner_hotkey}
    ]
    with connection.cursor() as cursor:
        cursor.execute("SET session_replication_role=replica")
        cursor.execute(
            "INSERT INTO public.lab_arena_rounds("
            "round_id,status,status_generation,stage_generation,configuration_doc,"
            "rewards_enabled,participants,evaluation_date,icp_set_date) VALUES "
            "(%s,%s,1,1,%s::jsonb,FALSE,%s::jsonb,'2026-09-22','2026-09-21')",
            (
                round_id,
                round_status,
                json.dumps(configuration),
                json.dumps(participants),
            ),
        )
        cursor.execute(
            "INSERT INTO public.lab_arena_runs("
            "run_id,assignment_id,round_id,submission_id,miner_hotkey,stage,"
            "icp_position,attempt,kind,status,terminal_cause,result_doc,output_ref,"
            "stage_generation) VALUES ("
            "%s,%s,%s,%s,%s,1,0,1,'execute','accepted','accepted',"
            "'{\"terminal_status\":\"accepted\"}'::jsonb,%s,1)",
            (
                round_id + ":accepted",
                round_id + ":assignment-0",
                round_id,
                submission_id,
                miner_hotkey,
                "arena/test/" + round_id + "/accepted.json",
            ),
        )
        if include_second_assignment:
            assignment_id = round_id + ":assignment-1"
            first_run_id = assignment_id + ":1"
            retry_run_id = assignment_id + ":2"
            cursor.execute(
                "INSERT INTO public.lab_arena_runs("
                "run_id,assignment_id,round_id,submission_id,miner_hotkey,stage,"
                "icp_position,attempt,kind,status,terminal_cause,result_doc,"
                "stage_generation) VALUES ("
                "%s,%s,%s,%s,%s,2,1,1,'execute','failed',%s,"
                "jsonb_build_object('terminal_status',%s::text),1)",
                (
                    first_run_id,
                    assignment_id,
                    round_id,
                    submission_id,
                    miner_hotkey,
                    prior_cause,
                    prior_cause,
                ),
            )
            cursor.execute(
                "INSERT INTO public.lab_arena_runs("
                "run_id,assignment_id,round_id,submission_id,miner_hotkey,stage,"
                "icp_position,attempt,kind,status,runner_hotkey,lease_token_hash,"
                "lease_generation,lease_expires_at,stage_generation) VALUES ("
                "%s,%s,%s,%s,%s,2,1,2,'execute',%s,%s,%s,1,"
                "clock_timestamp()+interval '1 hour',1)",
                (
                    retry_run_id,
                    assignment_id,
                    round_id,
                    submission_id,
                    miner_hotkey,
                    retry_status,
                    runner_hotkey if retry_status == "leased" else None,
                    "sha256:" + "a" * 64 if retry_status == "leased" else None,
                ),
            )
            call_identity = sha(round_id)
            for entry_kind in ledger_kinds:
                cursor.execute(
                    "INSERT INTO public.lab_arena_ledger("
                    "entry_kind,miner_hotkey,round_id,submission_id,run_id,stage,"
                    "call_identity,provider,operation_id,funding_source,"
                    "amount_microusd,entry_doc) VALUES ("
                    "%s,%s,%s,%s,%s,2,%s,'openrouter','openrouter.responses',"
                    "'host',100,'{}'::jsonb)",
                    (
                        entry_kind,
                        miner_hotkey,
                        round_id,
                        submission_id,
                        retry_run_id,
                        call_identity,
                    ),
                )
        cursor.execute("SET session_replication_role=origin")
    connection.commit()


@pytest.mark.parametrize(
    ("suffix", "retry_status", "prior_cause", "ledger_kinds"),
    [
        ("gp", "pending", "provider_error", ()),
        ("gr", "leased", "provider_error", ("reservation",)),
        (
            "gw",
            "leased",
            "worker_lost",
            ("reservation", "dispatch"),
        ),
        (
            "gj",
            "leased",
            "result_rejected",
            ("reservation", "dispatch"),
        ),
    ],
)
def test_deadline_provider_retry_guards_stay_fail_closed(
    connect, suffix, retry_status, prior_cause, ledger_kinds
):
    connection = connect()
    connection.autocommit = False
    round_id = "arena-2099-01-02-" + suffix
    try:
        _seed_deadline_close_guard(
            connection,
            round_id,
            retry_status=retry_status,
            prior_cause=prior_cause,
            ledger_kinds=ledger_kinds,
        )
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT public.lab_arena_close_parallel_execution_v1(%s)",
                (round_id,),
            )
            result = cursor.fetchone()[0]
            cursor.execute(
                "SELECT terminal_doc FROM public.lab_arena_runs "
                "WHERE round_id=%s AND attempt=2",
                (round_id,),
            )
            terminal_doc = cursor.fetchone()[0]
        connection.commit()
        assert result["status"] == "cancelled"
        assert result["incomplete_assignments"] == 1
        assert terminal_doc.get("deadline_provider_retry_exhausted") is None
    finally:
        connection.close()


def test_sequential_close_marks_a_dispatched_deadline_retry_complete(connect):
    connection = connect()
    connection.autocommit = False
    round_id = "arena-2099-01-02-gs"
    try:
        _seed_deadline_close_guard(
            connection,
            round_id,
            retry_status="leased",
            prior_cause="provider_error",
            ledger_kinds=("reservation", "dispatch"),
            round_status="stage2",
        )
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT public.lab_arena_close_stage(%s,2::smallint)",
                (round_id,),
            )
            result = cursor.fetchone()[0]
            cursor.execute(
                "SELECT terminal_cause,terminal_doc FROM public.lab_arena_runs "
                "WHERE round_id=%s AND attempt=2",
                (round_id,),
            )
            cause, terminal_doc = cursor.fetchone()
        connection.commit()
        assert result["status"] == "closed"
        assert result["incomplete_assignments"] == 0
        assert cause == "stage_closed"
        assert terminal_doc["previous_status"] == "leased"
        assert terminal_doc["deadline_provider_retry_exhausted"] is True
    finally:
        connection.close()


def test_parallel_close_keeps_incomplete_assignment_cardinality_fail_closed(connect):
    connection = connect()
    connection.autocommit = False
    round_id = "arena-2099-01-02-gc"
    try:
        _seed_deadline_close_guard(
            connection,
            round_id,
            retry_status="pending",
            prior_cause="provider_error",
            ledger_kinds=(),
            include_second_assignment=False,
        )
        with connection.cursor() as cursor:
            with pytest.raises(Exception, match="assignments_invalid"):
                cursor.execute(
                    "SELECT public.lab_arena_close_parallel_execution_v1(%s)",
                    (round_id,),
                )
        connection.rollback()
    finally:
        connection.close()
