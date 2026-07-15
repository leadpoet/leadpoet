"""Scoring-worker memory recycling, supervisor RSS backstop, and trace caps.

Production incident: scorer workers retain the interpreter's high-water RSS
after heavy scoring passes (3+ GiB each; the baseline-owner worker reached
24.6 GiB), exhausting host memory until claims were refused and API calls
returned 500s. These tests cover the three mitigations: worker between-pass
self-recycling, the supervisor's RSS telemetry/hard backstop, and the
host-side byte budget on decoded in-container trace entries.
"""

from __future__ import annotations

import logging
import threading
import time

import pytest

from gateway.research_lab import worker_autostart
from gateway.research_lab.scoring_worker import (
    _read_own_rss_mb,
    _worker_recycle_max_jobs,
    _worker_recycle_rss_mb,
)
from gateway.research_lab.worker_autostart import (
    EXPECTED_HOSTED_WORKERS,
    EXPECTED_SCORING_WORKERS,
    ResearchLabWorkerAutoStartPlan,
    ResearchLabWorkerFleetPlan,
    ResearchLabWorkerStartupError,
    ResearchLabWorkerSupervisor,
    _child_rss_mb,
    _hard_rss_limit_mb,
    _vmrss_mb,
)
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


def _fleet(kind: str, count: int) -> ResearchLabWorkerFleetPlan:
    return ResearchLabWorkerFleetPlan(
        kind=kind,
        worker_count=count,
        worker_prefix=kind,
        log_level="INFO",
        proxy_refs=tuple("proxy-%d" % index for index in range(count)),
        enabled=True,
    )


def test_full_topology_worker_health_requires_exact_unchanged_counts(monkeypatch):
    monkeypatch.setenv("GATEWAY_TEE_TOPOLOGY_MODE", "full")
    plan = ResearchLabWorkerAutoStartPlan(
        auto_start_enabled=True,
        hosted=_fleet("hosted", EXPECTED_HOSTED_WORKERS),
        scoring=_fleet("scoring", EXPECTED_SCORING_WORKERS),
    )
    supervisor = ResearchLabWorkerSupervisor(plan)
    supervisor.children = {
        **{"hosted:%d" % index: _StubChild(1000 + index) for index in range(EXPECTED_HOSTED_WORKERS)},
        **{"scoring:%d" % index: _StubChild(2000 + index) for index in range(EXPECTED_SCORING_WORKERS)},
    }
    supervisor._ready_children = set(supervisor.children)
    health = supervisor.health()
    assert health["hosted_running"] == 10
    assert health["scoring_running"] == 25


def test_full_topology_worker_health_rejects_reduced_fleet(monkeypatch):
    monkeypatch.setenv("GATEWAY_TEE_TOPOLOGY_MODE", "full")
    plan = ResearchLabWorkerAutoStartPlan(
        auto_start_enabled=True,
        hosted=_fleet("hosted", EXPECTED_HOSTED_WORKERS - 1),
        scoring=_fleet("scoring", EXPECTED_SCORING_WORKERS),
    )
    with pytest.raises(ResearchLabWorkerStartupError, match="exactly 10"):
        ResearchLabWorkerSupervisor(plan).health()


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
