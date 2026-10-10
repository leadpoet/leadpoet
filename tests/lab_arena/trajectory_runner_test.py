"""Runtime log delivery preserves provider, completion and secret boundaries."""

import json
import os
from dataclasses import replace
from datetime import datetime, timezone

import httpx
import pytest

from lab_arena import runner, runtime, shim, trajectory
from tests.lab_arena.test_lab_arena_runner import (
    BridgingRuntime, FakeApi, lease, make_config, scoring_lease, valid_company,
)


class LoggingApi(FakeApi):
    def __init__(self, *, failures=0):
        super().__init__([lease()])
        self.batches = []
        self.failures = failures
        self.order = []

    def trajectory(self, run_id, lease_token, events):
        self.batches.append({"run_id": run_id, "events": list(events)})
        self.order.append("trajectory")
        trajectory.validate_batch({"events": list(events)})
        if self.failures:
            self.failures -= 1
            raise runner.RunnerError("transport failure")
        return {"accepted": len(events)}

    def complete(self, envelope):
        self.order.append("complete")
        return super().complete(envelope)


class ProviderFailureApi(LoggingApi):
    def __init__(self, *, provider_failures=1, logging_failures=0):
        super().__init__(failures=logging_failures)
        self.provider_failures = provider_failures
        self.provider_attempts = []

    def provider(self, run_id, lease_token, frame):
        self.provider_attempts.append(dict(frame))
        if self.provider_failures:
            self.provider_failures -= 1
            raise runner.RunnerError(
                "private transport failure detail",
                http_status=503,
                denial_code="round_unknown",
            )
        return super().provider(run_id, lease_token, frame)


class RecoveringProviderRuntime:
    def __init__(self, *, calls=1, raise_after=False):
        self.calls = calls
        self.raise_after = raise_after
        self.failures = 0

    def run_icp(self, spec, **_):
        os.environ[shim.WORKER_SOCKET_ENV] = str(spec.socket_path)
        try:
            for _ in range(self.calls):
                try:
                    shim.dispatch(
                        "deepline.execute",
                        {"tool": "exa_search", "payload": {"query": "fintech"}},
                        5000,
                    )
                except (shim.ShimProviderError, shim.ShimTransportError):
                    self.failures += 1
        finally:
            os.environ.pop(shim.WORKER_SOCKET_ENV, None)
        if self.raise_after:
            raise ValueError("private runtime failure detail")
        output = json.dumps({"companies": [valid_company(1)]}).encode()
        return runtime.fake_result(output_bytes=output)


def run_model(tmp_path, api, sandbox):
    (tmp_path / "work").mkdir()
    worker = runner.Runner(make_config(tmp_path, api, sandbox))
    try:
        assert worker.run_once() == 1
    finally:
        worker.close()
    return worker


def test_runtime_events_precede_accepted_completion_and_retry_same_ids(tmp_path):
    api = LoggingApi(failures=1)
    run_model(tmp_path, api, BridgingRuntime(output={"companies": [valid_company(1)]}))
    assert api.batches[0] == api.batches[1]
    events = [event for batch in api.batches[1:] for event in batch["events"]]
    assert [event["kind"] for event in events] == [
        "runtime.started", "runtime.decision_capture", "runtime.finished",
        "runtime.stdout",
    ]
    assert events[1]["content"] == {
        "status": "not_provided", "recorded": 0, "omitted": 0,
    }
    finished = events[2]["content"]
    assert finished["status"] == "accepted"
    assert finished["company_count"] == 1
    assert finished["resource_summary"]["provider_call_count"] == 1
    assert events[3]["content"]["text"] == "model log line\n"
    assert api.order[-1] == "complete"
    assert api.completions[0]["body"]["result"]["terminal_status"] == "accepted"
    assert all(set(event) == {"kind", "event_id", "occurred_at", "content"} for event in events)


def test_pre_gateway_provider_failure_is_logged_before_recovered_completion(tmp_path):
    api = ProviderFailureApi()
    sandbox = RecoveringProviderRuntime()
    before = datetime.now(timezone.utc)
    run_model(tmp_path, api, sandbox)
    after = datetime.now(timezone.utc)

    events = [event for batch in api.batches for event in batch["events"]]
    provider_error = next(
        event for event in events if event["kind"] == "runtime.provider_error"
    )
    assert provider_error["content"] == {
        "operation_id": "deepline.execute",
        "action_sequence": 0,
        "outcome": "unknown",
        "error_class": "RunnerError",
        "error_code": "broker_unavailable",
        "provider": "deepline",
        "http_status": 503,
        "denial_code": "round_unknown",
    }
    occurred_at = datetime.fromisoformat(
        provider_error["occurred_at"].replace("Z", "+00:00")
    )
    assert before <= occurred_at <= after
    assert [event["kind"] for event in events] == [
        "runtime.started", "runtime.provider_error",
        "runtime.decision_capture", "runtime.finished",
    ]
    assert sandbox.failures == 1
    assert api.order[-1] == "complete"
    result = api.completions[0]["body"]["result"]
    assert result["terminal_status"] == "accepted"
    assert result["resource_summary"]["provider_call_count"] == 1
    assert "private transport failure detail" not in json.dumps(events)


def test_provider_error_event_construction_failure_is_fail_open(
    tmp_path, capsys, monkeypatch,
):
    def fail_observation(*_args):
        raise ValueError("private observation failure detail")

    monkeypatch.setattr(runner, "_provider_error_event", fail_observation)
    api = ProviderFailureApi()
    worker = run_model(tmp_path, api, RecoveringProviderRuntime())

    events = [event for batch in api.batches for event in batch["events"]]
    assert worker.abandoned == 0
    assert api.completions[0]["body"]["result"]["terminal_status"] == "accepted"
    assert api.completions[0]["body"]["result"]["resource_summary"][
        "provider_call_count"
    ] == 1
    assert all(event["kind"] != "runtime.provider_error" for event in events)
    diagnostic = capsys.readouterr().err
    assert "Arena provider error trajectory capture failed: ValueError" in diagnostic
    assert "private observation failure detail" not in diagnostic


def test_provider_error_buffer_is_capped_and_logging_failure_is_fail_open(
    tmp_path, capsys,
):
    api = ProviderFailureApi(provider_failures=35, logging_failures=100)
    sandbox = RecoveringProviderRuntime(calls=35)
    worker = run_model(tmp_path, api, sandbox)

    assert worker.abandoned == 0
    assert len(api.completions) == 1
    assert api.completions[0]["body"]["result"]["terminal_status"] == "accepted"
    assert api.completions[0]["body"]["result"]["resource_summary"][
        "provider_call_count"
    ] == 35
    attempted = api.batches[-1]["events"]
    provider_errors = [
        event for event in attempted if event["kind"] == "runtime.provider_error"
    ]
    assert len(provider_errors) == runner.MAX_BUFFERED_PROVIDER_ERROR_EVENTS
    assert provider_errors[-1]["content"]["dropped"] is True
    assert provider_errors[-1]["content"]["dropped_count"] == 3
    assert api.batches[-2] == api.batches[-1]
    assert "Arena trajectory upload failed: RunnerError" in capsys.readouterr().err


def test_provider_error_buffer_flushes_on_abandon(tmp_path):
    api = ProviderFailureApi()
    worker = run_model(tmp_path, api, RecoveringProviderRuntime(raise_after=True))
    events = [event for batch in api.batches for event in batch["events"]]
    assert worker.abandoned == 1
    assert [event["kind"] for event in events] == [
        "runtime.started", "runtime.provider_error",
        "runtime.decision_capture", "runtime.error",
    ]
    assert events[-1]["content"] == {
        "status": "abandoned", "failure_stage": "runtime",
        "error_class": "ValueError",
    }
    assert "private runtime failure detail" not in json.dumps(events)


def test_logging_outage_does_not_lose_model_completion(tmp_path, capsys):
    api = LoggingApi(failures=100)
    worker = run_model(tmp_path, api, BridgingRuntime(output={"companies": []}))
    assert worker.abandoned == 0
    assert len(api.completions) == 1
    assert "Arena trajectory upload failed: RunnerError" in capsys.readouterr().err


def test_runtime_crash_retains_error_without_exposing_exception_text(tmp_path):
    class BrokenRuntime:
        def run_icp(self, spec):
            raise ValueError("secret-private-exception-payload")

    api = LoggingApi()
    worker = run_model(tmp_path, api, BrokenRuntime())
    assert worker.abandoned == 1
    events = [event for batch in api.batches for event in batch["events"]]
    assert events[-1]["kind"] == "runtime.error"
    assert events[-1]["content"] == {
        "status": "abandoned", "failure_stage": "runtime",
        "error_class": "ValueError",
    }
    assert "secret-private-exception-payload" not in json.dumps(events)


def test_preflight_failure_records_setup_boundary_and_removes_directories(tmp_path):
    bad_lease = lease()
    bad_lease["checkpoint_deadline_policy"] = "unsupported-policy"
    api = LoggingApi()
    api.leases = [bad_lease]
    (tmp_path / "work").mkdir()
    config = make_config(
        tmp_path, api, BridgingRuntime(output={"companies": []})
    )
    config.socket_root = tmp_path / "sockets"
    config.socket_root.mkdir()
    worker = runner.Runner(config)
    try:
        assert worker.run_once() == 1
    finally:
        worker.close()

    events = [event for batch in api.batches for event in batch["events"]]
    assert [event["kind"] for event in events] == [
        "runtime.started", "runtime.decision_capture", "runtime.error",
    ]
    assert events[-1]["content"] == {
        "status": "abandoned", "failure_stage": "setup",
        "error_class": "RunnerError",
    }
    assert worker.abandoned == 1 and api.completions == []
    assert list((tmp_path / "work").iterdir()) == []
    assert list(config.socket_root.iterdir()) == []


def test_worker_stop_failure_records_cleanup_boundary_and_runtime_logs(
    tmp_path, monkeypatch,
):
    original_stop = runner.WorkerSocketServer.stop

    def fail_after_stop(server):
        original_stop(server)
        raise OSError("private cleanup detail")

    monkeypatch.setattr(runner.WorkerSocketServer, "stop", fail_after_stop)
    api = LoggingApi()
    worker = run_model(
        tmp_path, api, BridgingRuntime(output={"companies": [valid_company(1)]})
    )
    events = [event for batch in api.batches for event in batch["events"]]
    assert [event["kind"] for event in events] == [
        "runtime.started", "runtime.decision_capture", "runtime.error",
        "runtime.stdout",
    ]
    assert events[2]["content"] == {
        "status": "abandoned", "failure_stage": "cleanup",
        "error_class": "OSError",
        "resource_summary": {
            "wall_seconds": 1.0,
            "cpu_seconds": 0.5,
            "max_rss_bytes": 1024 * 1024,
            "stdout_bytes": len(b"model log line\n"),
            "stderr_bytes": 0,
            "provider_call_count": 1,
        },
        "exit_code": 0,
        "timed_out": False,
    }
    assert events[3]["content"]["text"] == "model log line\n"
    assert worker.abandoned == 1 and api.completions == []
    assert "private cleanup detail" not in json.dumps(events)


def test_worker_stop_failure_does_not_replace_runtime_failure(
    tmp_path, monkeypatch,
):
    class FailingRuntime:
        @staticmethod
        def run_icp(_spec):
            raise ValueError("primary detail")

    original_stop = runner.WorkerSocketServer.stop

    def fail_after_stop(server):
        original_stop(server)
        raise OSError("private cleanup detail")

    monkeypatch.setattr(runner.WorkerSocketServer, "stop", fail_after_stop)
    api = LoggingApi()
    worker = run_model(tmp_path, api, FailingRuntime())
    events = [event for batch in api.batches for event in batch["events"]]
    assert [event["kind"] for event in events] == [
        "runtime.started", "runtime.decision_capture", "runtime.error",
        "runtime.cleanup_error",
    ]
    assert events[2]["content"] == {
        "status": "abandoned", "failure_stage": "runtime",
        "error_class": "ValueError",
    }
    assert events[3]["content"] == {
        "failure_stage": "cleanup", "error_class": "OSError",
    }
    assert worker.abandoned == 1 and api.completions == []
    assert "primary detail" not in json.dumps(events)
    assert "private cleanup detail" not in json.dumps(events)


@pytest.mark.parametrize(
    ("failure", "error_class"),
    (
        (
            runner.DependencyInstallInfrastructureError("network_error"),
            "DependencyInstallInfrastructureError",
        ),
        (
            runner.AgentDependencyError("private dependency detail"),
            "AgentDependencyError",
        ),
    ),
)
def test_scoring_dependency_failure_records_setup_boundary(
    tmp_path, failure, error_class,
):
    class FailingImageCache:
        @staticmethod
        def acquire(*_args, **_kwargs):
            raise failure

    api = LoggingApi()
    api.leases = [scoring_lease()]
    (tmp_path / "work").mkdir()
    config = make_config(tmp_path, api, BridgingRuntime())
    config.image_cache = FailingImageCache()
    worker = runner.Runner(config)
    try:
        assert worker.run_once() == 1
    finally:
        worker.close()

    events = [event for batch in api.batches for event in batch["events"]]
    assert [event["kind"] for event in events] == [
        "runtime.started", "runtime.error",
    ]
    assert events[1]["content"] == {
        "status": "abandoned", "failure_stage": "setup",
        "error_class": error_class,
    }
    assert worker.abandoned == 1 and api.completions == []
    assert "private dependency detail" not in json.dumps(events)


def test_runsc_cleanup_failure_retains_captured_result_for_trajectory(tmp_path):
    class CleanupFailureRuntime:
        def run_icp(self, _spec):
            result = runtime.fake_result(
                stdout=b"captured stdout", stderr=b"captured stderr"
            )
            raise runtime.SandboxCleanupError(
                "private cleanup detail", result=result
            )

    api = LoggingApi()
    worker = run_model(tmp_path, api, CleanupFailureRuntime())
    events = [event for batch in api.batches for event in batch["events"]]
    assert [event["kind"] for event in events] == [
        "runtime.started", "runtime.decision_capture", "runtime.error",
        "runtime.stdout", "runtime.stderr",
    ]
    assert events[2]["content"] == {
        "status": "abandoned", "failure_stage": "cleanup",
        "error_class": "SandboxCleanupError",
        "resource_summary": {
            "wall_seconds": 1.0,
            "cpu_seconds": 0.5,
            "max_rss_bytes": 1024 * 1024,
            "stdout_bytes": len(b"captured stdout"),
            "stderr_bytes": len(b"captured stderr"),
            "provider_call_count": 0,
        },
        "exit_code": 0,
        "timed_out": False,
    }
    assert events[3]["content"]["text"] == "captured stdout"
    assert events[4]["content"]["text"] == "captured stderr"
    assert worker.abandoned == 1 and api.completions == []
    assert "private cleanup detail" not in json.dumps(events)


@pytest.mark.parametrize("text", ["\x00" * 65536, "\U0001f600" * 16384, "line\n" * 13108], ids=["control", "unicode", "lines"])
def test_full_bounded_stream_survives_event_and_batch_limits(text, tmp_path):
    raw = text.encode()[:runtime.MAX_LOG_BYTES]
    result = replace(runtime.fake_result(stdout=raw, stderr=raw), stdout_truncated=True)
    events = runner._runtime_log_events(result, "private-lease-token", "2026-09-25T12:00:00Z")
    for stream in ("stdout", "stderr"):
        selected = [event for event in events if event["kind"] == "runtime." + stream]
        expected = trajectory.redact_text(raw.decode()).encode()[:runtime.MAX_LOG_BYTES].decode(errors="replace")
        assert "".join(event["content"]["text"] for event in selected) == expected
        assert "\x00" not in json.dumps(selected)
        assert [event["content"]["sequence"] for event in selected] == list(range(len(selected)))
    api = LoggingApi()
    config = make_config(tmp_path, api, None)
    runner._record_trajectory(config, lease(), "private-lease-token", events)
    assert sum(len(batch["events"]) for batch in api.batches) == len(events)


def test_redaction_happens_before_runtime_chunking():
    secret = "lease-secret-" + "x" * 80
    text = "a" * 1000 + secret + "\nAuthorization: Bearer abcdefghijklmnop\n"
    events = runner._runtime_log_events(
        runtime.fake_result(stdout=text.encode()), secret, "2026-09-25T12:00:00Z",
    )
    encoded = json.dumps(events)
    assert secret not in encoded and "abcdefghijklmnop" not in encoded
    assert "[REDACTED]" in encoded


def test_external_client_upload_uses_only_existing_gateway_lease(monkeypatch):
    for key in ("SUPABASE_URL", "SUPABASE_KEY", "SUPABASE_SERVICE_ROLE_KEY", "SUPABASE_ANON_KEY"):
        monkeypatch.delenv(key, raising=False)
    event = trajectory.event("runtime.started", {"status": "starting"})

    def respond(request):
        assert request.url.path == "/arena/v1/runs/run-123/trajectory"
        assert request.headers["x-lab-arena-lease"] == "existing-lease"
        assert "authorization" not in request.headers and "apikey" not in request.headers
        assert json.loads(request.content) == {"events": [event]}
        return httpx.Response(200, json={"accepted": 1})

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        api = runner.HttpArenaApiClient("https://gateway.example", client=client)
        assert api.trajectory("run-123", "existing-lease", [event]) == {"accepted": 1}


@pytest.mark.parametrize("job_lease", [lease, scoring_lease])
@pytest.mark.parametrize("reason", ["sandbox_launch_failed", "sandbox_launcher_signaled", "sandbox_startup_timeout"])
def test_host_launch_failure_retains_private_diagnostic_and_no_completion(
    tmp_path, capsys, job_lease, reason,
):
    request = job_lease()
    token = request["lease_token"]

    class BrokenHost:
        def run_icp(self, _spec, **_kwargs):
            raise runtime.RuntimeHostError(
                "untrusted exception secret-text", reason=reason,
                launch_exit_code=-2 if reason == "sandbox_launcher_signaled" else 128,
                launch_timed_out=reason.endswith("timeout"),
                launch_stderr=("mount namespace: operation not permitted\nlease=" + token).encode(),
            )

    api = LoggingApi()
    api.leases = [request]
    worker = run_model(tmp_path, api, BrokenHost())
    assert worker.abandoned == 1
    assert api.completions == []
    assert api.provider_frames == []
    errors = [event for batch in api.batches for event in batch["events"]
              if event["kind"] == "runtime.error"]
    assert len(errors) == 1
    content = errors[0]["content"]
    assert content["status"] == "abandoned"
    assert content["failure_stage"] == "runtime"
    assert content["error_class"] == "RuntimeHostError"
    assert content["runtime_host_reason"] == reason
    assert content["launch_exit_code"] == (-2 if reason == "sandbox_launcher_signaled" else 128)
    assert content["launch_timed_out"] == reason.endswith("timeout")
    assert "operation not permitted" in content["launch_stderr"]
    logs = capsys.readouterr().err
    assert "operation not permitted" in logs
    for private in (token, "untrusted exception secret-text"):
        assert private not in json.dumps(errors)
        assert private not in logs
    assert "operation not permitted" not in worker.completed[0]["detail"]


def test_installed_probe_reports_private_launch_failure_without_traceback(monkeypatch, capsys):
    from scripts import _lab_arena_runsc_probe_ci as probe

    def fail(**_kwargs):
        raise runtime.RuntimeHostError(
            "private exception text", reason="sandbox_launch_failed", launch_exit_code=128,
            launch_stderr=b"mount failed: permission denied\napi_key=secret-value",
        )

    monkeypatch.setattr(probe, "run_probe", fail)
    assert probe.main(["--runsc-path", "/usr/bin/runsc"]) == 1
    logs = capsys.readouterr().err
    assert "LAB_ARENA_RUNSC_PROBE_FAILED" in logs
    assert "permission denied" in logs
    assert "launch_exit_code=128" in logs
    assert "secret-value" not in logs
    assert "private exception text" not in logs
