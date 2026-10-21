"""Runtime log delivery preserves provider, completion and secret boundaries."""

import json
import os
from dataclasses import replace
from datetime import datetime, timezone

import httpx
import pytest

from lab_arena import contracts, runner, runtime, scoring, shim, trajectory
from lab_arena.service import ArenaService
from lab_arena.store import hash_lease_token
from tests.lab_arena.test_lab_arena_runner import (
    BridgingRuntime, FakeApi, lease, make_config, scoring_lease, valid_company,
    verify,
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


def _run_judge_with_detail(tmp_path, detail=None):
    tmp_path.mkdir(exist_ok=True)
    api = LoggingApi()
    api.leases = [scoring_lease()]
    output = scoring.build_scoring_failure(
        "r1", "judge_error", detail or "", reason="malformed_response",
    )
    run_model(tmp_path, api, BridgingRuntime(output=output, calls=0))
    finished = next(event for batch in api.batches for event in batch["events"]
                    if event["kind"] == "runtime.finished")
    envelope = api.completions[0]
    assert verify(envelope["hotkey"], envelope["signature"],
                  contracts.signed_request_message(envelope))
    return finished["content"], envelope["body"]


def test_judge_exhaustion_summary_crosses_existing_trajectory_only(tmp_path):
    summary = (
        "judge_exhausted attempts=3 reason=malformed_response "
        "fit_dims=employee_size,required_attribute fit_class=insufficient_fit_evidence "
        "fit_reason=malformed_response inv_complete=no "
        "inv_reason=malformed_response fetch=ok:1,fetch_failed:2"
    )
    finished, completion = _run_judge_with_detail(tmp_path / "valid", summary)
    assert finished["judge_failure_summary"] == summary
    assert finished["failure_diagnostic"] == {
        "stage": "[DERIVED_BY_GATEWAY]", "error_class": "judge_error",
        "reason": "malformed_response",
    }
    assert "judge_failure_summary" not in json.dumps(completion)
    assert contracts.validate_run_result(completion["result"])[
        "failure_diagnostic"
    ] == {"stage": "scorer", "error_class": "judge_error",
          "reason": "malformed_response"}

    invalid_finished, invalid_completion = _run_judge_with_detail(
        tmp_path / "invalid", "private arbitrary detail",
    )
    assert "judge_failure_summary" not in invalid_finished
    assert completion == invalid_completion


def test_judge_summary_projector_failure_does_not_change_completion_or_cleanup(
    tmp_path, monkeypatch,
):
    detail = "judge_exhausted attempts=3 reason=malformed_response fit=absent"
    normal, expected_completion = _run_judge_with_detail(tmp_path / "normal", detail)
    assert normal["judge_failure_summary"] == detail

    def broken_projector(_detail):
        raise ValueError("private telemetry failure")

    monkeypatch.setattr(runner, "_judge_exhausted_summary", broken_projector)
    failed, actual_completion = _run_judge_with_detail(tmp_path / "failed", detail)
    assert "judge_failure_summary" not in failed
    assert actual_completion == expected_completion
    for directory in (tmp_path / "normal" / "work", tmp_path / "failed" / "work"):
        assert [path.name for path in directory.iterdir()] == ["pending-completions"]
        assert list((directory / "pending-completions").iterdir()) == []


def test_runner_judge_summary_passes_gateway_validation_and_append(tmp_path):
    summary = "judge_exhausted attempts=3 reason=malformed_response fit=absent"
    claimed = scoring_lease()
    persisted = []

    class RecordingStore:
        @staticmethod
        def get_run(run_id):
            return {
                **claimed, "runner_hotkey": "host-runner",
            } if run_id == claimed["run_id"] else None

        @staticmethod
        def append_trajectory_events(run_id, lease_hash, events):
            assert run_id == claimed["run_id"]
            assert lease_hash == hash_lease_token(claimed["lease_token"])
            persisted.extend(events)
            return {"status": "accepted", "accepted": len(events),
                    "inserted": len(events), "existing": 0}

    gateway = object.__new__(ArenaService)
    gateway._store = RecordingStore()
    gateway._require_round_ownership = lambda _round_id: None

    class GatewayApi(LoggingApi):
        def trajectory(self, run_id, lease_token, events):
            result = gateway.handle_trajectory(run_id, lease_token, {"events": list(events)})
            super().trajectory(run_id, lease_token, events)
            return result

    api = GatewayApi()
    api.leases = [claimed]
    output = scoring.build_scoring_failure(
        "r1", "judge_error", summary, reason="malformed_response",
    )
    run_model(tmp_path, api, BridgingRuntime(output=output, calls=0))
    finished = next(event for event in persisted if event["kind"] == "runtime.finished")
    assert finished["content"]["judge_failure_summary"] == summary
    envelope = api.completions[0]
    assert verify(envelope["hotkey"], envelope["signature"],
                  contracts.signed_request_message(envelope))
    assert "judge_failure_summary" not in json.dumps(envelope["body"])


@pytest.mark.parametrize("detail", [
    "private arbitrary detail",
    "judge_exhausted attempts=3 reason=malformed_response fit=absent\nsecret",
    "judge_exhausted attempts=3 reason=malformed_response fit=absent https://x.example/key",
    "judge_exhausted attempts=3 reason=malformed_response fit=absent api_key=secret",
    "judge_exhausted attempts=3 reason=malformed_response fit=absent" + "x" * 300,
    "judge_exhausted attempts=3 reason=malformed_response fit_dims=identity "
    "fit_class=secret fit_reason=malformed_response inv_complete=no "
    "inv_reason=none fetch=none",
    "judge_exhausted attempts=3 reason=malformed_response fit_dims=identity "
    "fit_class=missing fit_reason=missing inv_complete=no inv_reason=none "
    "fetch=fetch_failed:100",
])
def test_judge_summary_rejects_untrusted_or_truncated_detail(detail):
    assert runner._judge_exhausted_summary(detail) is None


@pytest.mark.parametrize("detail", ["", "judge_exhausted attempts=2 reason=unknown fit=absent",
                                    "judge_exhausted attempts=2 reason=unknown origin=exception"])
def test_judge_summary_missing_or_fixed_fallback_detail(tmp_path, detail):
    finished, _ = _run_judge_with_detail(tmp_path, detail)
    if detail:
        assert finished["judge_failure_summary"] == detail
    else:
        assert "judge_failure_summary" not in finished


def test_accepted_scoring_run_has_no_judge_failure_summary(tmp_path):
    api = LoggingApi()
    api.leases = [scoring_lease()]
    output = scoring.build_scoring_output(
        "r1", [{"final_score": 71.0, "failure_reason": ""},
               {"final_score": 44.5, "failure_reason": ""}],
    )
    run_model(tmp_path, api, BridgingRuntime(output=output, calls=0))
    finished = next(event for batch in api.batches for event in batch["events"]
                    if event["kind"] == "runtime.finished")
    assert finished["content"]["status"] == "accepted"
    assert "judge_failure_summary" not in finished["content"]


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
    assert "judge_failure_summary" not in finished
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


class ExpiringCaptureApi(LoggingApi):
    """An authenticated abandon lets a concurrent recovery expire the lease."""

    def __init__(self, request, *, capture_fails=False):
        super().__init__()
        self.leases = [request]
        self.active = True
        self.capture_fails = capture_fails
        self.capture_attempts = []
        self.capture_documents = []
        self.expiry_observed = []

    def trajectory(self, run_id, lease_token, events):
        result = super().trajectory(run_id, lease_token, events)
        if any(event["kind"] == "runtime.error"
               and event["content"].get("status") == "abandoned"
               for event in events):
            # Wait for the competing recovery before the worker can continue.
            # This forces the old capture-after-abandon race on every run.
            from concurrent.futures import ThreadPoolExecutor
            def expire():
                self.active = False
                self.expiry_observed.append(True)
            with ThreadPoolExecutor(max_workers=1) as pool:
                pool.submit(expire).result(timeout=2)
        return result

    def training_observations(self, run_id, lease_token, document):
        self.capture_attempts.append(self.active)
        self.capture_documents.append(document)
        self.order.append("observations")
        if not self.active:
            raise runner.RunnerError("lease_inactive", http_status=409)
        if self.capture_fails:
            raise runner.RunnerError("capture unavailable", http_status=503)
        return {"status": "stored", "payload_sha256": contracts.document_hash(document)}


@pytest.mark.parametrize("job_lease", [lease, scoring_lease])
@pytest.mark.parametrize("capture_fails", [False, True])
def test_capture_precedes_authenticated_abandon_even_when_capture_fails(
    tmp_path, job_lease, capture_fails,
):
    failure = runtime.RuntimeHostError(
        "controlled interruption", reason="sandbox_launcher_signaled",
        launch_exit_code=-2, launch_timed_out=False,
    )
    class Interrupted:
        def run_icp(self, _spec, **_kwargs):
            raise failure

    api = ExpiringCaptureApi(job_lease(), capture_fails=capture_fails)
    worker = run_model(tmp_path, api, Interrupted())
    assert api.capture_attempts == ([True, True] if capture_fails else [True])
    assert api.expiry_observed == [True]
    assert worker.abandoned == 1 and api.completions == []
    errors = [event for batch in api.batches for event in batch["events"]
              if event["kind"] == "runtime.error"]
    assert len(errors) == 1
    assert errors[0]["content"]["error_class"] == "RuntimeHostError"
    assert errors[0]["content"]["launch_exit_code"] == -2
    assert api.order.index("observations") < max(
        i for i, name in enumerate(api.order) if name == "trajectory"
    )
    assert not list((tmp_path / "work").iterdir())


def test_initial_setup_failure_captures_before_abandon(tmp_path):
    request = lease()
    request["icp_wall_clock_seconds"] = True
    api = ExpiringCaptureApi(request)
    worker = run_model(tmp_path, api, BridgingRuntime(calls=0))
    assert api.capture_attempts == [True]
    assert api.expiry_observed == [True]
    assert worker.abandoned == 1 and api.completions == []


def test_normal_completion_captures_once_without_abandon(tmp_path):
    api = ExpiringCaptureApi(lease())
    worker = run_model(tmp_path, api, BridgingRuntime(calls=0))
    assert api.capture_attempts == [True]
    assert api.expiry_observed == []
    assert worker.abandoned == 0 and len(api.completions) == 1
    assert api.order.index("observations") < api.order.index("complete")


@pytest.mark.parametrize("stop_fails", [False, True])
def test_capture_failure_does_not_replace_original_exception(
    tmp_path, monkeypatch, stop_fails,
):
    original = runtime.RuntimeHostError(
        "controlled interruption", reason="sandbox_launcher_signaled",
        launch_exit_code=-2,
    )
    class Interrupted:
        def run_icp(self, _spec, **_kwargs):
            raise original

    stopped = []
    stop = runner.WorkerSocketServer.stop
    def stop_then_fail(server):
        stop(server)
        stopped.append(True)
        if stop_fails:
            raise OSError("controlled cleanup failure")
    monkeypatch.setattr(runner.WorkerSocketServer, "stop", stop_then_fail)
    request = lease()
    api = ExpiringCaptureApi(request, capture_fails=True)
    (tmp_path / "work").mkdir()
    executor = runner.AssignmentExecutor(make_config(tmp_path, api, Interrupted()))
    with pytest.raises(runtime.RuntimeHostError) as caught:
        executor.execute(request, request["lease_token"], request["icp"])
    assert caught.value is original
    assert stopped == [True]
    assert api.capture_attempts == [True, True]
    assert api.expiry_observed == [True] and api.completions == []
    assert not list((tmp_path / "work").iterdir())


def test_cleanup_only_failure_still_captures_before_abandon(tmp_path, monkeypatch):
    stop = runner.WorkerSocketServer.stop
    def fail_after_stop(server):
        stop(server)
        raise OSError("controlled cleanup failure")
    monkeypatch.setattr(runner.WorkerSocketServer, "stop", fail_after_stop)
    api = ExpiringCaptureApi(lease())
    worker = run_model(tmp_path, api, BridgingRuntime(calls=0))
    assert api.capture_attempts == [True]
    assert api.expiry_observed == [True]
    assert worker.abandoned == 1 and api.completions == []


def test_interrupted_provider_work_captures_without_replaying_calls(tmp_path):
    class Interrupted(BridgingRuntime):
        def run_icp(self, spec, **kwargs):
            super().run_icp(spec, **kwargs)
            raise runtime.RuntimeHostError(
                "controlled interruption", reason="sandbox_launcher_signaled",
                launch_exit_code=-2,
            )

    api = ExpiringCaptureApi(lease())
    worker = run_model(tmp_path, api, Interrupted(calls=1))
    assert len(api.provider_frames) == 1
    assert api.capture_attempts == [True]
    assert len(api.capture_documents) == 1
    events = [event for batch in api.batches for event in batch["events"]]
    assert sum(event["kind"] == "runtime.error" for event in events) == 1
    assert worker.abandoned == 1 and api.completions == []


@pytest.mark.parametrize("complete_mode", ["success", "transport_failure", "rejected"])
def test_valid_interrupted_judge_subset_completes_before_lease_expiry(
    tmp_path, complete_mode,
):
    from tests.lab_arena.company_judgments_test import _input, _new, _refs
    from tests.lab_arena.partial_company_judgments_test import _lease

    document = _input()
    request = scoring_lease()
    request.update(
        scorer_policy=document["scorer_policy"], company_quality_policy="company_quality_v1",
        company_judgment_cache=_lease(document),
        scored_output={"companies": document["companies"]},
    )
    rows = [_new(_refs(document)[0])]
    checkpoint = scoring.build_scoring_failure(
        "r1", "judge_error", company_judgments=rows,
    )
    class Interrupted:
        def run_icp(self, _spec, **_kwargs):
            raise runtime.RuntimeHostError(
                "controlled interruption", reason="sandbox_launcher_signaled",
                launch_exit_code=-2, checkpoint_bytes=json.dumps(checkpoint).encode(),
            )
    class Api(ExpiringCaptureApi):
        def complete(self, envelope):
            self.delivery_attempts.append(envelope)
            if not self.active:
                raise runner.RunnerError("lease_inactive", http_status=409)
            if complete_mode == "transport_failure":
                raise runner.RunnerError("controlled availability failure", http_status=503)
            if complete_mode == "rejected":
                return {"status": "rejected"}
            return super().complete(envelope)
    api = Api(request)
    api.delivery_attempts = []
    worker = run_model(tmp_path, api, Interrupted())
    assert api.capture_attempts == [True]
    assert api.expiry_observed == []
    for delivered in api.delivery_attempts:
        assert delivered["body"]["output"]["company_judgments"] == rows
        assert delivered["body"]["result"]["terminal_status"] == "judge_error"
    assert worker.abandoned == (complete_mode == "transport_failure")
    if complete_mode == "success":
        assert len(api.completions) == 1
    else:
        assert api.completions == []
        retained = list((tmp_path / "work" / "pending-completions").glob("*.json"))
        assert len(retained) == 1
        saved = json.loads(retained[0].read_text())
        assert saved["body"]["output"]["company_judgments"] == rows
