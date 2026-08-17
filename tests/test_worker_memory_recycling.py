"""Scoring-worker memory recycling, supervisor RSS backstop, and trace caps.

Production incident: scorer workers retain the interpreter's high-water RSS
after heavy scoring passes (3+ GiB each; the baseline-owner worker reached
24.6 GiB), exhausting host memory until claims were refused and API calls
returned 500s. These tests cover the three mitigations: worker between-pass
self-recycling, the supervisor's RSS telemetry/hard backstop, and the
host-side byte budget on decoded in-container trace entries.
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
from types import SimpleNamespace

import pytest

import gateway.research_lab.worker as hosted_worker_module
from gateway.research_lab import worker_autostart
from gateway.research_lab.scoring_worker import (
    BaselineCheckpointRecycle,
    ResearchLabGatewayScoringWorker,
    _baseline_checkpoint_recycle_pressure,
    _read_own_rss_mb,
    _worker_recycle_max_jobs,
    _worker_recycle_rss_mb,
)
from gateway.research_lab.worker_autostart import (
    ResearchLabWorkerAutoStartPlan,
    ResearchLabWorkerFleetPlan,
    ResearchLabWorkerStartupError,
    ResearchLabWorkerSupervisor,
    _child_rss_mb,
    _hard_rss_limit_mb,
    _vmrss_mb,
    start_worker_supervisor_without_blocking_event_loop,
)
from gateway.research_lab.worker import HostedWorkerOutcome, ResearchLabHostedWorker
from research_lab.eval.private_runtime import (
    INCONTAINER_TRACE_MARKER,
    parse_incontainer_trace_lines,
)


def _write_status(tmp_path, rss_kb: int) -> str:
    path = tmp_path / "status"
    path.write_text(
        "Name:\tpython3\nVmPeak:\t  999999 kB\n"
        f"VmRSS:\t  {rss_kb} kB\nThreads:\t20\n",
        encoding="utf-8",
    )
    return str(path)


def test_vmrss_parses_proc_status(tmp_path):
    assert _vmrss_mb(_write_status(tmp_path, 3_355_648)) == 3277
    assert _read_own_rss_mb(_write_status(tmp_path, 61_440)) == 60


def test_vmrss_missing_file_and_malformed_are_none(tmp_path):
    assert _vmrss_mb(str(tmp_path / "absent")) is None
    bad = tmp_path / "bad"
    bad.write_text("VmRSS:\n", encoding="utf-8")
    assert _vmrss_mb(str(bad)) is None
    no_field = tmp_path / "nofield"
    no_field.write_text("Name:\tpython3\n", encoding="utf-8")
    assert _vmrss_mb(str(no_field)) is None


@pytest.mark.asyncio
async def test_worker_supervisor_startup_does_not_block_gateway_event_loop():
    main_thread = threading.get_ident()
    started = threading.Event()
    release = threading.Event()
    worker_threads: list[int] = []

    class BlockingSupervisor:
        def start(self) -> None:
            worker_threads.append(threading.get_ident())
            started.set()
            if not release.wait(timeout=2):
                raise AssertionError("worker supervisor test release timed out")

        def health(self) -> dict[str, bool]:
            worker_threads.append(threading.get_ident())
            return {"ready": True}

    startup = asyncio.create_task(
        start_worker_supervisor_without_blocking_event_loop(
            BlockingSupervisor()  # type: ignore[arg-type]
        )
    )
    assert await asyncio.to_thread(started.wait, 1)

    loop_tick = asyncio.Event()
    asyncio.get_running_loop().call_soon(loop_tick.set)
    await asyncio.wait_for(loop_tick.wait(), timeout=0.2)
    assert not startup.done()

    release.set()
    assert await asyncio.wait_for(startup, timeout=1) == {"ready": True}
    assert len(worker_threads) == 2
    assert worker_threads[0] == worker_threads[1]
    assert worker_threads[0] != main_thread


@pytest.mark.asyncio
async def test_worker_supervisor_cancellation_waits_before_stop() -> None:
    started = threading.Event()
    release = threading.Event()
    finished = threading.Event()
    stop_raced_start = False

    class BlockingSupervisor:
        def start(self) -> None:
            started.set()
            if not release.wait(timeout=2):
                raise AssertionError("worker supervisor test release timed out")
            finished.set()

        def health(self) -> dict[str, bool]:
            return {"ready": True}

        def stop(self) -> None:
            nonlocal stop_raced_start
            stop_raced_start = not finished.is_set()

    supervisor = BlockingSupervisor()
    startup = asyncio.create_task(
        start_worker_supervisor_without_blocking_event_loop(
            supervisor  # type: ignore[arg-type]
        )
    )
    assert await asyncio.to_thread(started.wait, 1)
    startup.cancel()
    await asyncio.sleep(0)
    assert not startup.done()

    release.set()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(startup, timeout=1)
    supervisor.stop()
    assert finished.is_set()
    assert stop_raced_start is False


@pytest.mark.asyncio
async def test_real_supervisor_spawns_on_event_loop_thread_and_waits_off_thread(
    monkeypatch,
) -> None:
    caller_thread = threading.get_ident()
    popen_threads: list[int] = []
    wait_threads: list[int] = []

    class ReadyChild:
        def __init__(self, _command, **kwargs):
            popen_threads.append(threading.get_ident())
            os.write(int(kwargs["env"]["RESEARCH_LAB_WORKER_READY_FD"]), b"ready\n")

        def poll(self):
            return None

        def terminate(self):
            raise AssertionError("ready worker was unexpectedly terminated")

    plan = ResearchLabWorkerAutoStartPlan(
        auto_start_enabled=True,
        hosted=_fleet("hosted", 1),
        scoring=_fleet("scoring", 1),
    )
    supervisor = ResearchLabWorkerSupervisor(plan)
    original_wait = supervisor._wait_for_child_ready

    def observed_wait(*args, **kwargs):
        wait_threads.append(threading.get_ident())
        return original_wait(*args, **kwargs)

    monkeypatch.setenv("GATEWAY_TEE_TOPOLOGY_MODE", "full")
    monkeypatch.setattr(
        worker_autostart,
        "build_research_lab_worker_environment",
        lambda: {},
    )
    monkeypatch.setattr(worker_autostart.subprocess, "Popen", ReadyChild)
    monkeypatch.setattr(supervisor, "_wait_for_child_ready", observed_wait)
    monkeypatch.setattr(supervisor, "_monitor_children", lambda: None)

    health = await start_worker_supervisor_without_blocking_event_loop(supervisor)

    assert health["status"] == "ready"
    assert popen_threads == [caller_thread, caller_thread]
    assert len(wait_threads) == 2
    assert all(thread_id != caller_thread for thread_id in wait_threads)


def test_child_rss_for_bogus_pid_is_none():
    assert _child_rss_mb(2**31 - 1) is None


def test_recycle_thresholds_env_defaults_and_overrides(monkeypatch):
    monkeypatch.delenv("RESEARCH_LAB_SCORING_WORKER_RECYCLE_RSS_MB", raising=False)
    monkeypatch.delenv("RESEARCH_LAB_SCORING_WORKER_RECYCLE_JOBS", raising=False)
    assert _worker_recycle_rss_mb() == 3072
    assert _worker_recycle_max_jobs() == 16
    monkeypatch.setenv("RESEARCH_LAB_SCORING_WORKER_RECYCLE_RSS_MB", "512")
    monkeypatch.setenv("RESEARCH_LAB_SCORING_WORKER_RECYCLE_JOBS", "0")
    assert _worker_recycle_rss_mb() == 512
    assert _worker_recycle_max_jobs() == 0
    monkeypatch.setenv("RESEARCH_LAB_SCORING_WORKER_RECYCLE_RSS_MB", "junk")
    assert _worker_recycle_rss_mb() == 3072


def test_hosted_recycle_threshold_and_current_rss(monkeypatch, tmp_path):
    monkeypatch.delenv("RESEARCH_LAB_HOSTED_WORKER_RECYCLE_RSS_MB", raising=False)
    assert hosted_worker_module._hosted_worker_recycle_rss_mb() == 3072
    monkeypatch.setenv("RESEARCH_LAB_HOSTED_WORKER_RECYCLE_RSS_MB", "4096")
    assert hosted_worker_module._hosted_worker_recycle_rss_mb() == 4096
    monkeypatch.setenv("RESEARCH_LAB_HOSTED_WORKER_RECYCLE_RSS_MB", "junk")
    assert hosted_worker_module._hosted_worker_recycle_rss_mb() == 3072

    status_path = _write_status(tmp_path, 3_355_648)
    assert hosted_worker_module._read_own_rss_mb(status_path) == 3277
    assert hosted_worker_module._read_own_rss_mb(str(tmp_path / "missing")) is None


def _hosted_worker_for_recycle_test(outcome):
    worker = object.__new__(ResearchLabHostedWorker)
    worker.worker_ref = "hosted-memory-test"
    worker.config = SimpleNamespace(
        hosted_worker_poll_seconds=1,
        hosted_worker_max_runs=0,
    )

    async def run_once():
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    worker.run_once = run_once
    return worker


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "outcome, expected_status",
    (
        (
            HostedWorkerOutcome(
                processed=False,
                dry_run=False,
                status="tree_preflight_deferred",
            ),
            "tree_preflight_deferred",
        ),
        (RuntimeError("pass failed after releasing resources"), "failed"),
    ),
)
async def test_hosted_worker_recycles_between_passes_above_rss_limit(
    monkeypatch,
    caplog,
    outcome,
    expected_status,
):
    worker = _hosted_worker_for_recycle_test(outcome)
    monkeypatch.setattr(
        "gateway.research_lab.capture_health.enforce_capture_health",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        hosted_worker_module,
        "_hosted_worker_recycle_required",
        lambda **_kwargs: 4096,
    )

    with caplog.at_level(
        logging.WARNING,
        logger="gateway.research_lab.worker",
    ):
        await worker.run_forever()

    assert "research_lab_hosted_worker_recycle" in caplog.text
    assert f"pass_status={expected_status}" in caplog.text


@pytest.mark.asyncio
async def test_hosted_worker_below_rss_limit_continues_polling(monkeypatch):
    worker = _hosted_worker_for_recycle_test(
        HostedWorkerOutcome(processed=False, dry_run=False, status="idle")
    )
    monkeypatch.setattr(
        "gateway.research_lab.capture_health.enforce_capture_health",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        hosted_worker_module,
        "_hosted_worker_recycle_required",
        lambda **_kwargs: None,
    )

    class PollContinued(RuntimeError):
        pass

    async def stop_after_sleep(_seconds):
        raise PollContinued

    monkeypatch.setattr(hosted_worker_module.asyncio, "sleep", stop_after_sleep)

    with pytest.raises(PollContinued):
        await worker.run_forever()


def test_baseline_checkpoint_recycle_pressure_uses_existing_limits(monkeypatch):
    monkeypatch.setenv("RESEARCH_LAB_SCORING_WORKER_RECYCLE_RSS_MB", "3072")
    config = SimpleNamespace(scoring_worker_min_available_memory_mb=4096)

    assert (
        _baseline_checkpoint_recycle_pressure(
            config,
            rss_mb=1024,
            available_memory_mb=8192,
        )
        is None
    )
    assert _baseline_checkpoint_recycle_pressure(
        config,
        rss_mb=4096,
        available_memory_mb=8192,
    )["reason"] == "worker_rss_limit"
    assert _baseline_checkpoint_recycle_pressure(
        config,
        rss_mb=1024,
        available_memory_mb=2048,
    )["reason"] == "host_memory_pressure"


@pytest.mark.asyncio
async def test_parallel_baseline_recycles_only_after_checkpointed_wave(monkeypatch):
    worker = object.__new__(ResearchLabGatewayScoringWorker)
    worker.worker_ref = "baseline-worker"
    worker.config = SimpleNamespace(
        private_baseline_concurrency=2,
        private_baseline_retry_concurrency=1,
        private_baseline_provider_retry_rounds=0,
        scoring_worker_min_available_memory_mb=4096,
    )
    window = SimpleNamespace(
        benchmark_items=[
            {"icp_ref": f"icp-{index}", "icp_hash": f"hash-{index}"}
            for index in range(1, 6)
        ]
    )
    called: list[int] = []
    checkpointed: list[int] = []

    async def run_icp(self, *, item_index, item, **_kwargs):  # noqa: ANN001
        called.append(item_index)
        return {
            "icp_ref": item["icp_ref"],
            "icp_hash": item["icp_hash"],
            "score": float(item_index),
            "company_count": 1,
            "sourced_count": 1,
            "diagnostics": {},
            "_item_index": item_index,
            "_retryable": False,
            "_nonempty": True,
            "_runtime_error": "",
            "_retry_backoff_seconds": 0.0,
        }

    async def checkpoint(row):  # noqa: ANN001
        checkpointed.append(int(row["_item_index"]))
        return True

    async def maintenance_state():
        return {"paused": False}

    monkeypatch.setattr(
        "gateway.research_lab.scoring_worker.get_scoring_maintenance_state",
        maintenance_state,
    )
    monkeypatch.setattr(ResearchLabGatewayScoringWorker, "_run_baseline_icp", run_icp)
    monkeypatch.setattr(
        "gateway.research_lab.scoring_worker._baseline_checkpoint_recycle_pressure",
        lambda _config: {
            "reason": "worker_rss_limit",
            "rss_mb": 4096,
            "recycle_rss_mb": 3072,
            "available_memory_mb": 8192,
            "min_available_memory_mb": 4096,
        },
    )

    with pytest.raises(BaselineCheckpointRecycle) as raised:
        await worker._run_baseline_batch_inner(
            runner=object(),
            retry_runner=object(),
            scorer=object(),
            window=window,
            run_start=time.time(),
            icp_checkpoint=checkpoint,
            checkpoint_recycle_enabled=True,
        )

    assert sorted(called) == [1, 2]
    assert sorted(checkpointed) == [1, 2]
    assert raised.value.completed_icps == 2
    assert raised.value.total_icps == 5


@pytest.mark.asyncio
async def test_parallel_baseline_resumes_across_recycles_without_duplicate_icps(
    monkeypatch,
):
    worker = object.__new__(ResearchLabGatewayScoringWorker)
    worker.worker_ref = "baseline-worker"
    worker.config = SimpleNamespace(
        private_baseline_concurrency=2,
        private_baseline_retry_concurrency=1,
        private_baseline_provider_retry_rounds=0,
        scoring_worker_min_available_memory_mb=4096,
    )
    window = SimpleNamespace(
        benchmark_items=[
            {"icp_ref": f"icp-{index}", "icp_hash": f"hash-{index}"}
            for index in range(1, 6)
        ]
    )
    called: list[int] = []
    persisted: dict[str, dict] = {}
    pressure_checks = iter(
        (
            {"reason": "worker_rss_limit", "rss_mb": 4096},
            {"reason": "worker_rss_limit", "rss_mb": 4096},
            None,
        )
    )

    async def run_icp(self, *, item_index, item, **_kwargs):  # noqa: ANN001
        called.append(item_index)
        return {
            "icp_ref": item["icp_ref"],
            "icp_hash": item["icp_hash"],
            "score": float(item_index),
            "company_count": 1,
            "sourced_count": 1,
            "diagnostics": {},
            "_item_index": item_index,
            "_retryable": False,
            "_nonempty": True,
            "_runtime_error": "",
            "_retry_backoff_seconds": 0.0,
        }

    async def checkpoint(row):  # noqa: ANN001
        persisted[str(row["icp_ref"])] = {
            key: value for key, value in row.items() if not key.startswith("_")
        }
        return True

    async def maintenance_state():
        return {"paused": False}

    monkeypatch.setattr(
        "gateway.research_lab.scoring_worker.get_scoring_maintenance_state",
        maintenance_state,
    )
    monkeypatch.setattr(ResearchLabGatewayScoringWorker, "_run_baseline_icp", run_icp)
    monkeypatch.setattr(
        "gateway.research_lab.scoring_worker._baseline_checkpoint_recycle_pressure",
        lambda _config: next(pressure_checks),
    )

    for expected_completed in (2, 4):
        with pytest.raises(BaselineCheckpointRecycle) as raised:
            await worker._run_baseline_batch_inner(
                runner=object(),
                retry_runner=object(),
                scorer=object(),
                window=window,
                run_start=time.time(),
                resume_results=list(persisted.values()),
                icp_checkpoint=checkpoint,
                checkpoint_recycle_enabled=True,
            )
        assert raised.value.completed_icps == expected_completed

    rows, stats = await worker._run_baseline_batch_inner(
        runner=object(),
        retry_runner=object(),
        scorer=object(),
        window=window,
        run_start=time.time(),
        resume_results=list(persisted.values()),
        icp_checkpoint=checkpoint,
        checkpoint_recycle_enabled=True,
    )

    assert called == [1, 2, 3, 4, 5]
    assert [row["icp_ref"] for row in rows] == [
        "icp-1",
        "icp-2",
        "icp-3",
        "icp-4",
        "icp-5",
    ]
    assert [row["score"] for row in rows] == [1.0, 2.0, 3.0, 4.0, 5.0]
    assert stats == {"retried": 0, "recovered": 0, "unresolved": 0}


def test_hard_rss_limit_env(monkeypatch):
    monkeypatch.delenv("RESEARCH_LAB_WORKER_HARD_RSS_LIMIT_MB", raising=False)
    assert _hard_rss_limit_mb() == 16384
    monkeypatch.setenv("RESEARCH_LAB_WORKER_HARD_RSS_LIMIT_MB", "8192")
    assert _hard_rss_limit_mb() == 8192


class _StubChild:
    def __init__(self, pid: int = 4242):
        self.pid = pid
        self.terminated = threading.Event()

    def poll(self):
        return None

    def terminate(self):
        self.terminated.set()


class _StubFleet:
    kind = "scoring"


def _run_monitor_briefly(supervisor: ResearchLabWorkerSupervisor, seconds: float) -> None:
    thread = threading.Thread(target=supervisor._monitor_children, daemon=True)
    thread.start()
    time.sleep(seconds)
    supervisor._stop_event.set()
    thread.join(timeout=2)


@pytest.fixture
def supervisor_with_stub(monkeypatch):
    supervisor = ResearchLabWorkerSupervisor.__new__(ResearchLabWorkerSupervisor)
    supervisor._stop_event = threading.Event()
    child = _StubChild()
    supervisor.children = {"scoring:0": child}
    supervisor._child_specs = {"scoring:0": (_StubFleet(), 0)}
    monkeypatch.setenv("RESEARCH_LAB_WORKER_SUPERVISOR_POLL_SECONDS", "0.05")
    return supervisor, child


def test_supervisor_terminates_child_over_hard_limit(monkeypatch, supervisor_with_stub):
    supervisor, child = supervisor_with_stub
    monkeypatch.setenv("RESEARCH_LAB_WORKER_HARD_RSS_LIMIT_MB", "1024")
    monkeypatch.setattr(worker_autostart, "_child_rss_mb", lambda pid: 2048)
    _run_monitor_briefly(supervisor, 0.3)
    assert child.terminated.is_set()


def test_supervisor_leaves_child_under_hard_limit(monkeypatch, supervisor_with_stub):
    supervisor, child = supervisor_with_stub
    monkeypatch.setenv("RESEARCH_LAB_WORKER_HARD_RSS_LIMIT_MB", "1024")
    monkeypatch.setattr(worker_autostart, "_child_rss_mb", lambda pid: 512)
    _run_monitor_briefly(supervisor, 0.3)
    assert not child.terminated.is_set()


def test_supervisor_hard_limit_zero_disables_backstop(monkeypatch, supervisor_with_stub):
    supervisor, child = supervisor_with_stub
    monkeypatch.setenv("RESEARCH_LAB_WORKER_HARD_RSS_LIMIT_MB", "0")
    monkeypatch.setattr(worker_autostart, "_child_rss_mb", lambda pid: 999_999)
    _run_monitor_briefly(supervisor, 0.3)
    assert not child.terminated.is_set()


def test_supervisor_rss_telemetry_line(monkeypatch, supervisor_with_stub, capsys):
    supervisor, _child = supervisor_with_stub
    monkeypatch.setenv("RESEARCH_LAB_WORKER_RSS_TELEMETRY_SECONDS", "0.01")
    monkeypatch.setattr(worker_autostart, "_child_rss_mb", lambda pid: 777)
    _run_monitor_briefly(supervisor, 0.3)
    out = capsys.readouterr().out
    assert "research_lab_worker_rss" in out
    assert "scoring:0=777MB" in out


def test_supervisor_restarts_worker_after_clean_recycle_exit(monkeypatch):
    supervisor = ResearchLabWorkerSupervisor.__new__(ResearchLabWorkerSupervisor)
    supervisor._stop_event = threading.Event()
    exited = _StubChild()
    exited.poll = lambda: 0
    replacement = _StubChild(pid=4343)
    fleet = _StubFleet()
    supervisor.children = {"scoring:0": exited}
    supervisor._child_specs = {"scoring:0": (fleet, 0)}
    supervisor._ready_children = {"scoring:0"}
    started: list[tuple[object, int]] = []

    def restart(selected_fleet, index):  # noqa: ANN001
        started.append((selected_fleet, index))
        supervisor._stop_event.set()
        return replacement

    monkeypatch.setenv("RESEARCH_LAB_WORKER_SUPERVISOR_POLL_SECONDS", "0.01")
    monkeypatch.setattr(supervisor, "_start_child", restart)

    supervisor._monitor_children()

    assert started == [(fleet, 0)]
    assert supervisor.children == {"scoring:0": replacement}
    assert supervisor._ready_children == {"scoring:0"}


def _fleet(kind: str, count: int) -> ResearchLabWorkerFleetPlan:
    return ResearchLabWorkerFleetPlan(
        kind=kind,
        worker_count=count,
        worker_prefix=kind,
        log_level="INFO",
        proxy_refs=tuple("proxy-%d" % index for index in range(count)),
        enabled=True,
    )


def test_full_topology_worker_health_uses_configured_counts(monkeypatch):
    monkeypatch.setenv("GATEWAY_TEE_TOPOLOGY_MODE", "full")
    hosted_workers = 3
    scoring_workers = 7
    plan = ResearchLabWorkerAutoStartPlan(
        auto_start_enabled=True,
        hosted=_fleet("hosted", hosted_workers),
        scoring=_fleet("scoring", scoring_workers),
    )
    supervisor = ResearchLabWorkerSupervisor(plan)
    supervisor.children = {
        **{"hosted:%d" % index: _StubChild(1000 + index) for index in range(hosted_workers)},
        **{"scoring:%d" % index: _StubChild(2000 + index) for index in range(scoring_workers)},
    }
    supervisor._ready_children = set(supervisor.children)
    health = supervisor.health()
    assert health["hosted_running"] == hosted_workers
    assert health["scoring_running"] == scoring_workers


def test_full_topology_worker_health_rejects_empty_fleet(monkeypatch):
    monkeypatch.setenv("GATEWAY_TEE_TOPOLOGY_MODE", "full")
    plan = ResearchLabWorkerAutoStartPlan(
        auto_start_enabled=True,
        hosted=_fleet("hosted", 0),
        scoring=_fleet("scoring", 7),
    )
    with pytest.raises(ResearchLabWorkerStartupError, match="configured enabled"):
        ResearchLabWorkerSupervisor(plan).health()


def test_explicit_worker_fleet_deferral_starts_no_host_workers(monkeypatch):
    monkeypatch.setenv("GATEWAY_TEE_TOPOLOGY_MODE", "full")
    plan = ResearchLabWorkerAutoStartPlan(
        auto_start_enabled=True,
        hosted=_fleet("hosted", 3),
        scoring=_fleet("scoring", 7),
    )
    supervisor = ResearchLabWorkerSupervisor(
        plan,
        environment={"GATEWAY_V2_DEFER_WORKER_FLEETS": "all"},
    )
    monkeypatch.setattr(
        supervisor,
        "_start_child",
        lambda *_args, **_kwargs: pytest.fail(
            "deferred host worker was started"
        ),
    )

    supervisor.start()
    health = supervisor.health()

    assert supervisor.children == {}
    assert health["deferred_worker_fleet_roles"] == [
        "gateway_autoresearch",
        "gateway_scoring",
    ]
    assert health["hosted_configured"] == 3
    assert health["hosted_expected_running"] == 0
    assert health["hosted_running"] == 0
    assert health["scoring_configured"] == 7
    assert health["scoring_expected_running"] == 0
    assert health["scoring_running"] == 0


def test_invalid_worker_fleet_deferral_fails_closed():
    plan = ResearchLabWorkerAutoStartPlan(
        auto_start_enabled=True,
        hosted=_fleet("hosted", 3),
        scoring=_fleet("scoring", 7),
    )
    with pytest.raises(
        ResearchLabWorkerStartupError,
        match="invalid deferred V2 worker fleet role",
    ):
        ResearchLabWorkerSupervisor(
            plan,
            environment={"GATEWAY_V2_DEFER_WORKER_FLEETS": "unknown"},
        )


def _trace_line(seq: int, payload_pad: str = "") -> str:
    return (
        f"{INCONTAINER_TRACE_MARKER} "
        f'{{"seq": {seq}, "outcome": "success", "pad": "{payload_pad}"}}'
    )


def test_trace_parse_unbudgeted_keeps_all(monkeypatch):
    monkeypatch.delenv("RESEARCH_LAB_INCONTAINER_TRACE_MAX_BYTES", raising=False)
    stderr = "\n".join(_trace_line(i) for i in range(5))
    entries = parse_incontainer_trace_lines(stderr)
    assert [e["seq"] for e in entries] == [0, 1, 2, 3, 4]


def test_trace_parse_drops_entries_past_byte_budget(monkeypatch, caplog):
    monkeypatch.setenv("RESEARCH_LAB_INCONTAINER_TRACE_MAX_BYTES", "200")
    trace_logger = logging.getLogger("research_lab.eval.private_runtime")
    monkeypatch.setattr(trace_logger, "disabled", False)
    monkeypatch.setattr(trace_logger, "propagate", True)
    stderr = "\n".join(_trace_line(i, payload_pad="x" * 60) for i in range(10))
    with caplog.at_level("WARNING", logger=trace_logger.name):
        entries = parse_incontainer_trace_lines(stderr)
    assert 0 < len(entries) < 10
    assert entries[0]["seq"] == 0
    assert "incontainer_trace_capture_truncated" in caplog.text


def test_trace_parse_zero_budget_disables_cap(monkeypatch):
    monkeypatch.setenv("RESEARCH_LAB_INCONTAINER_TRACE_MAX_BYTES", "0")
    stderr = "\n".join(_trace_line(i, payload_pad="x" * 60) for i in range(10))
    assert len(parse_incontainer_trace_lines(stderr)) == 10
