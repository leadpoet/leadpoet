"""Real runner/API/socket coverage for the twenty-ICP execution window.

The provider and sandbox process boundary stay controlled.  Claims, leases,
source delivery, the worker socket, provider accounting, retries, scoring, and
publication use the production service and PostgreSQL functions.
"""

from __future__ import annotations

import json
import os
import socketserver
import threading
import time
from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from lab_arena import contracts, lab_arena_checkpoint, runner as rn, runtime, scoring, shim
from lab_arena.api import create_app
from lab_arena.proxy_workers import (
    ProxyWorkerPool,
    VerifiedProxyWorker,
    VerifiedProxyWorkerInventory,
)
from tests.lab_arena.lab_arena_pg_harness import (
    CURRENT_SERVICE_MIGRATIONS,
    database_with_lab_arena_migration,
)
from tests.lab_arena.successful_call_cost_round_test import CostHarness
from tests.lab_arena.test_lab_arena_service_round import _start_round, keypair


@pytest.fixture(autouse=True)
def bounded_socket_shutdown_poll(monkeypatch):
    """Keep real socket cleanup without a half-second delay per fixture run."""

    serve_forever = socketserver.BaseServer.serve_forever

    def serve_with_short_poll(server, poll_interval=0.5):
        return serve_forever(server, poll_interval=min(poll_interval, 0.005))

    monkeypatch.setattr(socketserver.BaseServer, "serve_forever", serve_with_short_poll)


@pytest.fixture()
def database():
    yield from database_with_lab_arena_migration(CURRENT_SERVICE_MIGRATIONS)


def _proxy_pool(process_capacity: int) -> ProxyWorkerPool:
    """Build N verified proxy exits for the N+1 process-capacity contract."""

    assert process_capacity in (10, 20)
    workers = tuple(
        VerifiedProxyWorker(
            slot_index=index + 1,
            source_name="LAB_ARENA_WEBSHARE_PROXY_%d" % (index + 1),
            proxy_url="http://worker:secret@proxy-%d.example:8080" % index,
            proxy_fingerprint=("proxy%011d" % index),
            exit_ip="8.8.4.%d" % (index + 1),
            exit_ip_fingerprint=("exit%012d" % index),
        )
        for index in range(process_capacity - 1)
    )
    inventory = VerifiedProxyWorkerInventory(
        workers=workers,
        native_exit_ip="1.1.1.1",
        native_exit_ip_fingerprint="nativeexit000000",
    )
    assert inventory.total_process_capacity == process_capacity
    return ProxyWorkerPool(inventory)


def _register_external_validator(harness: CostHarness, label: str):
    key = keypair(label)
    harness.chain.runners.append(key.ss58_address)
    harness.chain.permits[key.ss58_address] = True
    harness.chain.stakes[key.ss58_address] = 100_000.0
    harness.chain.active[key.ss58_address] = False
    return key


def _http_runner(
    harness: CostHarness,
    http: TestClient,
    root: Path,
    *,
    key_label: str,
    local_capacity: int,
) -> rn.Runner:
    key = keypair(key_label)
    cache_key = key.ss58_address[:12]
    api = rn.HttpArenaApiClient("http://localhost", client=http)
    image_cache = rn.ImageCache(
        root / ("images-" + cache_key),
        lambda _reference, _digest, target: (target / "rootfs").mkdir(),
    )
    source_cache = rn.SourceCache(
        root / ("sources-" + cache_key),
        api.source,
        dependency_installer=lambda _requirements, _target: None,
    )
    work_dir = root / ("work-" + cache_key)
    work_dir.mkdir(parents=True, exist_ok=True)
    return rn.Runner(
        rn.RunnerConfig(
            round_id=harness.round_id,
            identity=rn.RunnerIdentity(
                hotkey=key.ss58_address,
                sign=lambda message: key.sign(message.encode("utf-8")).hex(),
            ),
            api=api,
            sandbox_runtime=harness.sandbox,
            image_cache=image_cache,
            source_cache=source_cache,
            work_dir=work_dir,
            max_parallel_runs=local_capacity,
            evaluation_date="2026-09-14",
            clock=harness.clock,
            proxy_worker_pool=_proxy_pool(local_capacity),
            completion_retry_seconds=(0.0, 0.0),
            claim_retry_seconds=(),
        )
    )


class _ExecutionEvidence:
    def __init__(self, harness: CostHarness, *, expected_high_water: int) -> None:
        self.harness = harness
        self.original = harness.sandbox.run_icp
        self.lock = threading.Lock()
        self.condition = threading.Condition(self.lock)
        self.expected_high_water = expected_high_water
        self.active = 0
        self.high_water = 0
        self.execution_specs = []
        self.successful_charged_calls = 0

    def run_icp(self, spec: runtime.SandboxSpec, **kwargs):
        document = json.loads(
            (spec.input_dir / runtime.INPUT_FILE_NAME).read_text(encoding="utf-8")
        )
        if document.get("schema_version") == scoring.SCORING_INPUT_SCHEMA_VERSION:
            assert spec.web_bridge_path is None
            return self.original(spec, **kwargs)

        assert spec.web_bridge_path is not None
        assert (spec.socket_path.parent / runtime.SANDBOX_WEB_SOCKET_NAME).is_socket()
        with self.condition:
            self.execution_specs.append(spec)
            self.active += 1
            self.high_water = max(self.high_water, self.active)
            self.condition.notify_all()
            deadline = time.monotonic() + 2.0
            while self.high_water < self.expected_high_water:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self.condition.wait(remaining)
        try:
            # Keep assignments overlapped long enough for the test to observe
            # the actual Runner worker count.  The charged request then crosses
            # the real worker Unix socket and service HTTP provider endpoint.
            time.sleep(0.005)
            with self.harness.sandbox.lock:
                os.environ[shim.WORKER_SOCKET_ENV] = str(spec.socket_path)
                try:
                    quota = lab_arena_checkpoint.quota_usage()
                    assert quota["providers"]["openrouter"] == {
                        "limit": contracts.CALL_QUOTAS_PER_ICP["openrouter"],
                        "used": 0,
                        "remaining": contracts.CALL_QUOTAS_PER_ICP["openrouter"],
                        "inflight": 0,
                    }
                    status, _headers, _body = shim.dispatch(
                        "openrouter.chat",
                        {
                            "model": "openai/gpt-4o-mini",
                            "messages": [
                                {
                                    "role": "user",
                                    "content": "successful empty completion",
                                }
                            ],
                            "max_tokens": 100,
                        },
                        5000,
                    )
                finally:
                    os.environ.pop(shim.WORKER_SOCKET_ENV, None)
            assert status == 200
            with self.lock:
                self.successful_charged_calls += 1
            return self.original(spec, **kwargs)
        finally:
            with self.condition:
                self.active -= 1
                self.condition.notify_all()


@contextmanager
def _postgres_lease_clock(harness: CostHarness):
    scheduled = harness.clock.now
    harness.clock.now = datetime.now(timezone.utc)
    try:
        yield
    finally:
        harness.clock.now = scheduled


def _drain(runner: rn.Runner, harness: CostHarness, *, sequential: bool) -> int:
    total = 0
    with _postgres_lease_clock(harness):
        while True:
            taken = runner.run_once(max_claims=1 if sequential else 1000)
            total += taken
            if taken == 0:
                return total


def _expire_one_unstarted_attempt(
    harness: CostHarness,
    http: TestClient,
    root: Path,
    connect,
) -> str:
    dead_label = "parallel-runtime-dead-" + harness.round_id
    _register_external_validator(harness, dead_label)
    dead = _http_runner(
        harness,
        http,
        root / "dead",
        key_label=dead_label,
        local_capacity=10,
    )
    try:
        with _postgres_lease_clock(harness):
            lease = dead.claim_one()
        assert lease["status"] == "leased"
        assert lease["parallel_twenty_icp_execution"] is True
    finally:
        dead.close()
    with connect() as connection, connection.cursor() as cursor:
        cursor.execute(
            "UPDATE public.lab_arena_runs "
            "SET lease_expires_at=clock_timestamp()-interval '1 second' "
            "WHERE run_id=%s",
            (lease["run_id"],),
        )
    expired = harness.service.store.expire_leases(harness.round_id)
    assert expired["expired"] == 1 and expired["retried"] == 1
    return str(lease["assignment_id"])


def _published_signature(harness: CostHarness) -> dict:
    round_row = harness.service.store.get_round(harness.round_id)
    final_by_submission = {
        row["submission_id"]: row
        for row in round_row["publication_doc"]["final_ranking"]
    }
    participants = []
    for participant in round_row["participants"]:
        submission_id = participant["submission_id"]
        public = harness.service.public_results(harness.round_id, submission_id)
        scores = sorted(
            (
                int(row["icp_position"]),
                float(row["per_icp_score"]),
            )
            for row in public["scores"]["stage_1"] + public["scores"]["stage_2"]
        )
        entry = final_by_submission[submission_id]
        output_by_position = sorted(
            (
                int(score["icp_position"]),
                public["outputs"][score["run_id"]],
            )
            for score in public["scores"]["stage_1"] + public["scores"]["stage_2"]
        )
        participants.append(
            {
                "flavor": harness.flavors[submission_id],
                "is_baseline": bool(participant["is_king"]),
                "scores": scores,
                "stage_1": public["submission_scores"]["stage_1"],
                "final": public["submission_scores"]["final"],
                "eligible": entry["eligible"],
                "cost_summary": entry["cost_summary"],
                "outputs": output_by_position,
                "company_counts": sorted(
                    len(output["companies"])
                    for output in public["outputs"].values()
                ),
            }
        )
    decision = round_row["publication_doc"]["king_decision"]
    winner_id = decision.get("winner_submission_id")
    return {
        "king_outcome": round_row["king_outcome"],
        "winner_flavor": None if not winner_id else harness.flavors[winner_id],
        "participants": sorted(participants, key=lambda row: row["flavor"]),
    }


def test_baseline_first_lease_keeps_verified_web_and_codex_runtime(
    database, tmp_path
):
    """The sequence policy changes scheduling, not the trusted execute runtime."""

    psycopg2, dsn = database
    connect = lambda: psycopg2.connect(**dsn)
    harness = CostHarness(
        connect,
        tmp_path / "baseline-first-runtime",
        challengers=["BaselineFirstRuntimeCandidate"],
        runners=["alpha"],
    )
    harness.service.config.defaults = replace(
        harness.service.config.defaults,
        execution_sequence_from="2000-01-01T00:00:00Z",
        parallel_twenty_icp_execution=False,
    )
    _start_round(harness, day=27, epoch=62_027)
    harness.clock.advance_to(harness.schedule()["stage_1_start"])
    assert harness.service.advance_round(harness.round_id)["assignments"] == 20

    evidence = _ExecutionEvidence(harness, expected_high_water=1)
    harness.sandbox.run_icp = evidence.run_icp
    runner_label = "baseline-first-runtime-runner"
    _register_external_validator(harness, runner_label)
    with TestClient(create_app(harness.service)) as http:
        runner = _http_runner(
            harness,
            http,
            tmp_path / "baseline-first-runtime-runner",
            key_label=runner_label,
            local_capacity=10,
        )
        try:
            with _postgres_lease_clock(harness):
                lease = runner.claim_one()
                assert lease["execution_sequence_policy"] == (
                    contracts.BASELINE_SCORED_FIRST_POLICY
                )
                assert "parallel_twenty_icp_execution" not in lease
                runner._executor.execute(
                    lease, str(lease["lease_token"]), lease["icp"]
                )
        finally:
            runner.close()

    assert len(evidence.execution_specs) == 1
    spec = evidence.execution_specs[0]
    assert spec.web_bridge_path is not None
    assert spec.codex_module_path is not None
    assert spec.web_bridge_path.name == "web-egress-bridge.py"
    assert spec.codex_module_path.name == "lab_arena_codex.py"


def _run_published_case(
    connect,
    root: Path,
    *,
    day: int,
    epoch: int,
    local_capacity: int,
    sequential: bool,
) -> tuple[dict, int, int, str, list[dict]]:
    root.mkdir(parents=True, exist_ok=True)
    harness = CostHarness(
        connect,
        root,
        challengers=["ParallelCandidate"],
        runners=["alpha"],
    )
    harness.service.config.defaults = replace(
        harness.service.config.defaults,
        runner_slot_ceiling=local_capacity,
        parallel_twenty_icp_execution=True,
    )
    participants = _start_round(harness, day=day, epoch=epoch)
    assert participants == 2
    configuration = harness.service.store.get_round(harness.round_id)[
        "configuration_doc"
    ]
    assert configuration["runner_slot_ceiling"] == local_capacity
    assert configuration["parallel_twenty_icp_execution"] is True

    evidence = _ExecutionEvidence(
        harness,
        expected_high_water=1 if sequential else local_capacity,
    )
    harness.sandbox.run_icp = evidence.run_icp
    runner_label = "parallel-runtime-live-%d-%d" % (day, local_capacity)
    _register_external_validator(harness, runner_label)

    harness.clock.advance_to(harness.schedule()["stage_1_start"])
    opened = harness.service.advance_round(harness.round_id)
    assert opened["assignments"] == participants * contracts.BENCHMARK_ICP_COUNT

    with TestClient(create_app(harness.service)) as http:
        expired_assignment = _expire_one_unstarted_attempt(
            harness, http, root, connect
        )
        runner = _http_runner(
            harness,
            http,
            root,
            key_label=runner_label,
            local_capacity=local_capacity,
        )
        try:
            _drain(runner, harness, sequential=sequential)
        finally:
            runner.close()

    assert harness.status() == "stage1"
    closed = harness.service.advance_round(harness.round_id)
    assert closed["status"] == "ok" and harness.status() == "stage1_closed"
    frozen_configuration = harness.service.store.get_round(harness.round_id)[
        "configuration_doc"
    ]
    harness.service = harness.build_service()
    assert harness.service.store.get_round(harness.round_id)[
        "configuration_doc"
    ] == frozen_configuration

    with TestClient(create_app(harness.service)) as http:
        runner = _http_runner(
            harness,
            http,
            root / "after-restart",
            key_label=runner_label,
            local_capacity=local_capacity,
        )
        try:
            for _step in range(30):
                status = harness.status()
                if status == "published":
                    break
                if status in (
                    "stage1",
                    "stage1_scoring",
                    "stage2",
                    "stage2_scoring",
                ):
                    _drain(runner, harness, sequential=sequential)
                if status == "stage1_scored":
                    harness.clock.advance_to(harness.schedule()["stage_2_start"])
                result = harness.service.advance_round(harness.round_id)
                assert result.get("status") not in (
                    "cancelled",
                    "terminal",
                    "retry",
                    "stale",
                ), (status, result)
            else:
                raise AssertionError("round did not publish")
        finally:
            runner.close()

    execute_runs = harness.service.store.list_runs(
        harness.round_id, kind="execute"
    )
    attempts = [
        row for row in execute_runs if row["assignment_id"] == expired_assignment
    ]
    assert [(row["attempt"], row["status"]) for row in attempts] == [
        (1, "failed"),
        (2, "accepted"),
    ]
    assert attempts[0]["terminal_cause"] == "lease_expired"
    assert attempts[0]["runner_hotkey"] != attempts[1]["runner_hotkey"]
    assert all(
        row["status"] == "accepted"
        for row in execute_runs
        if row["assignment_id"] != expired_assignment
    )
    assert evidence.successful_charged_calls == (
        participants * contracts.BENCHMARK_ICP_COUNT
    )
    return (
        _published_signature(harness),
        evidence.high_water,
        evidence.successful_charged_calls,
        expired_assignment,
        execute_runs,
    )


def test_real_parallel_runtime_matches_sequential_score_publication_and_cost(
    database, tmp_path
):
    """Ten and twenty workers preserve all outputs, scores, costs, and retry isolation."""

    psycopg2, dsn = database
    connect = lambda: psycopg2.connect(**dsn)
    sequential, sequential_high_water, calls, _expired, _runs = _run_published_case(
        connect,
        tmp_path / "sequential",
        day=26,
        epoch=61_026,
        local_capacity=10,
        sequential=True,
    )
    assert sequential_high_water == 1
    assert calls == 40

    for day, epoch, capacity in ((27, 61_027, 10), (28, 61_028, 20)):
        parallel, high_water, calls, _expired, execute_runs = _run_published_case(
            connect,
            tmp_path / ("parallel-%d" % capacity),
            day=day,
            epoch=epoch,
            local_capacity=capacity,
            sequential=False,
        )
        assert high_water == capacity
        assert calls == 40
        assert parallel == sequential
        assert len(execute_runs) == 41
        for participant in parallel["participants"]:
            assert len(participant["scores"]) == contracts.BENCHMARK_ICP_COUNT
            assert participant["company_counts"] == [5] * contracts.BENCHMARK_ICP_COUNT
            cost = participant["cost_summary"]
            assert cost["execution"]["settled_microusd"] == 200
            assert cost["execution"]["successful_microusd"] == 200
            assert cost["competition_sourcing_microusd"] == 200
            assert cost["execution"]["successful_calls"] == 40


@pytest.mark.parametrize(
    ("round_cap", "small_claims", "large_claims"),
    ((10, 10, 10), (20, 10, 20)),
)
def test_mixed_external_validator_claims_use_each_local_capacity_below_round_cap(
    database,
    tmp_path,
    round_cap,
    small_claims,
    large_claims,
):
    """A larger eligible validator exposes work but cannot raise the frozen cap."""

    psycopg2, dsn = database
    connect = lambda: psycopg2.connect(**dsn)
    harness = CostHarness(
        connect,
        tmp_path / ("claims-%d" % round_cap),
        challengers=["MixedCapacityCandidate"],
        runners=["alpha"],
    )
    harness.service.config.defaults = replace(
        harness.service.config.defaults,
        runner_slot_ceiling=round_cap,
        parallel_twenty_icp_execution=True,
    )
    _start_round(harness, day=round_cap, epoch=62_000 + round_cap)
    harness.clock.advance_to(harness.schedule()["stage_1_start"])
    assert harness.service.advance_round(harness.round_id)["assignments"] == 40

    small_label = "mixed-small-%d" % round_cap
    large_label = "mixed-large-%d" % round_cap
    small_key = _register_external_validator(harness, small_label)
    large_key = _register_external_validator(harness, large_label)
    with TestClient(create_app(harness.service)) as http:
        small = _http_runner(
            harness,
            http,
            tmp_path / ("small-%d" % round_cap),
            key_label=small_label,
            local_capacity=10,
        )
        large = _http_runner(
            harness,
            http,
            tmp_path / ("large-%d" % round_cap),
            key_label=large_label,
            local_capacity=20,
        )
        try:
            with _postgres_lease_clock(harness):
                small_leases = [small.claim_one() for _ in range(small_claims)]
                large_leases = [large.claim_one() for _ in range(large_claims)]
                assert small.claim_one()["status"] in {"no_capacity", "no_free_slot"}
                assert large.claim_one()["status"] in {"no_capacity", "no_free_slot"}
        finally:
            small.close()
            large.close()

    assert {row["status"] for row in small_leases + large_leases} == {"leased"}
    assert all(row["parallel_twenty_icp_execution"] is True for row in small_leases + large_leases)
    runs = harness.service.store.list_runs(harness.round_id, kind="execute")
    by_runner = {
        small_key.ss58_address: sum(
            row["status"] == "leased" and row["runner_hotkey"] == small_key.ss58_address
            for row in runs
        ),
        large_key.ss58_address: sum(
            row["status"] == "leased" and row["runner_hotkey"] == large_key.ss58_address
            for row in runs
        ),
    }
    assert by_runner == {
        small_key.ss58_address: small_claims,
        large_key.ss58_address: large_claims,
    }
    assert by_runner[small_key.ss58_address] <= min(10, round_cap)
    assert by_runner[large_key.ss58_address] <= min(20, round_cap)
