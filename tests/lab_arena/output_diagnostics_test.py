"""Execution output diagnostics preserve checkpoint and scoring behavior."""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from lab_arena import contracts, runner, runtime, shim
from lab_arena.service import ArenaService, ServiceError
from tests.lab_arena.test_lab_arena_runner import (
    FakeApi,
    RefusingApi,
    lease,
    make_config,
)
from tests.lab_arena.test_lab_arena_runtime import (
    FakeClock,
    FakeProcess,
    FakeRunner,
    _checkpoint_spec,
    make_config as make_runtime_config,
)


VALID_EMPTY = b'{"companies":[]}'
INVALID_NONEMPTY = b'{"companies":[{"private_payload":"do-not-persist"}]}'


class CheckpointSequenceRuntime:
    def __init__(
        self,
        work_dir: Path,
        *,
        mode: str,
        repaired: bool = False,
        initial_valid: bool = True,
        stderr: bytes = b"executionOutputInvalid: do-not-persist\n",
    ) -> None:
        self.work_dir = work_dir
        self.mode = mode
        self.repaired = repaired
        self.initial_valid = initial_valid
        self.stderr = stderr
        self.result = None

    def run_icp(self, spec, **_kwargs):
        clock = FakeClock()
        emitted = set()
        events = (
            [(1000.04, VALID_EMPTY), (1000.08, INVALID_NONEMPTY)]
            if self.initial_valid
            else [(1000.04, INVALID_NONEMPTY)]
        )
        if self.repaired:
            events.append(
                (
                    1000.12,
                    b'{"schema_version":"leadpoet.lab_arena.output.v1",'
                    b'"companies":[]}',
                )
            )
        sleeps = 0

        def sleep(seconds):
            nonlocal sleeps
            sleeps += 1
            clock.sleep(seconds)
            for index, (at, payload) in enumerate(events):
                if clock() >= at and index not in emitted:
                    spec.output_path.write_bytes(payload)
                    emitted.add(index)
            if self.mode == "deadline" and sleeps >= 3:
                clock.now += float(spec.wall_clock_seconds)

        finish_at = None if self.mode == "deadline" else (
            1000.20 if self.repaired else 1000.15
        )
        process_runner = FakeRunner(
            clock,
            run_process=lambda argv: FakeProcess(
                argv,
                clock=clock,
                finish_at=finish_at,
                returncode=0,
                stderr=self.stderr,
            ),
        )
        self.result = runtime.run_sandbox(
            runtime.RuntimeConfig(
                runsc_path=Path("/usr/local/bin/runsc"),
                work_dir=self.work_dir,
            ),
            spec,
            process_runner=process_runner,
            clock=clock,
            sleep=sleep,
            rusage=lambda: (0.0, 0),
        )
        return self.result

    def read_output(self, spec):
        return runtime.read_output(spec)


def checkpoint_lease():
    document = lease()
    document.update(
        {
            "checkpoint_deadline_policy": contracts.CHECKPOINT_DEADLINE_POLICY,
            "icp_wall_clock_seconds": contracts.CHECKPOINT_WALL_CLOCK_SECONDS,
            "lease_ttl_seconds": contracts.CHECKPOINT_LEASE_TTL_SECONDS,
        }
    )
    return document


def execution_diagnostic_line(document, *, ending=b"\n"):
    payload = json.dumps(
        document, sort_keys=True, separators=(",", ":")
    ).encode("ascii")
    line = runner._EXECUTION_DIAGNOSTIC_PREFIX + payload + ending
    assert len(line) <= runner._EXECUTION_DIAGNOSTIC_MAX_BYTES
    return line


def supervisor_diagnostic(
    *, failure_class="runtime_error", reason="two_failed_codex_exits"
):
    return {
        "schema_version": 1,
        "event": "supervisor_failure",
        "failure_class": failure_class,
        "reason": reason,
    }


def test_execution_diagnostic_parser_keeps_last_valid_supervisor_record():
    failure = supervisor_diagnostic()
    last_failure = supervisor_diagnostic(failure_class="other", reason="unexpected")
    secret = "private-secret-do-not-log"
    stderr = b"".join(
        (
            runner._EXECUTION_DIAGNOSTIC_PREFIX
            + json.dumps(
                {**failure, "unknown": secret},
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
            + b"\n",
            runner._EXECUTION_DIAGNOSTIC_PREFIX + b"x" * 300 + b"\n",
            execution_diagnostic_line(failure, ending=b"\r\n"),
            execution_diagnostic_line(failure),
            execution_diagnostic_line(last_failure)[:-1],
            runner._EXECUTION_DIAGNOSTIC_PREFIX
            + b'{"event":"supervisor_failure","failure_class":"'
            + secret.encode()
            + b'","reason":"unexpected","schema_version":1}\n',
            execution_diagnostic_line(last_failure),
        )
    )

    assert runner._execution_diagnostic_from_stderr(stderr) == last_failure


def test_execution_diagnostic_parser_rejects_noncanonical_or_wrong_types():
    wrong_class = supervisor_diagnostic()
    wrong_class["failure_class"] = ["runtime_error"]
    wrong_reason = supervisor_diagnostic()
    wrong_reason["reason"] = {"secret": "do-not-log"}
    wrong_version = supervisor_diagnostic()
    wrong_version["schema_version"] = True
    duplicate = (
        runner._EXECUTION_DIAGNOSTIC_PREFIX
        + b'{"event":"supervisor_failure","failure_class":"runtime_error",'
        b'"reason":"unexpected","schema_version":1,"schema_version":1}\n'
    )
    noncanonical = (
        runner._EXECUTION_DIAGNOSTIC_PREFIX
        + b'{"schema_version": 1, "event": "tyche_finish"}\n'
    )

    assert runner._execution_diagnostic_from_stderr(
        execution_diagnostic_line(wrong_class)
        + execution_diagnostic_line(wrong_reason)
        + execution_diagnostic_line(wrong_version)
        + duplicate
        + noncanonical
    ) is None


@pytest.mark.parametrize("mode, timed_out", [("fast", False), ("deadline", True)])
def test_invalid_final_checkpoint_keeps_accepted_output_and_adds_safe_diagnostic(
    tmp_path, capsys, mode, timed_out
):
    api = FakeApi([checkpoint_lease()])
    sandbox = CheckpointSequenceRuntime(
        tmp_path / "runtime-work", mode=mode
    )
    sandbox.work_dir.mkdir()
    (tmp_path / "work").mkdir()

    runner.Runner(make_config(tmp_path, api, sandbox)).run_once()

    completion = api.completions[0]["body"]
    result = contracts.validate_run_result(completion["result"])
    assert sandbox.result.timed_out is timed_out
    assert sandbox.result.output_bytes == VALID_EMPTY
    assert sandbox.result.checkpoint_output_invalid is True
    assert result["terminal_status"] == "accepted"
    assert completion["output"] == {
        "schema_version": "leadpoet.lab_arena.output.v1",
        "companies": [],
    }
    assert result["failure_diagnostic"] == {
        "stage": "sandbox_output",
        "error_class": "execution_output_invalid",
        "reason": "invalid_final_checkpoint",
    }
    assert "do-not-persist" not in json.dumps(completion)
    journal = capsys.readouterr().err
    if timed_out:
        assert "trust=host_observed" in journal
        assert "event=sandbox_outcome" in journal
        assert "host_timed_out=true" in journal
    else:
        assert "Lab Arena execution diagnostic:" not in journal


def test_execution_diagnostic_crosses_runtime_capture_on_accepted_fallback(
    tmp_path, capsys
):
    api = FakeApi([checkpoint_lease()])
    failure = supervisor_diagnostic(
        failure_class="timeout", reason="deadline_or_idle_timeout"
    )
    sandbox = CheckpointSequenceRuntime(
        tmp_path / "runtime-work",
        mode="deadline",
        stderr=execution_diagnostic_line(failure),
    )
    sandbox.work_dir.mkdir()
    (tmp_path / "work").mkdir()

    runner.Runner(make_config(tmp_path, api, sandbox)).run_once()

    result = contracts.validate_run_result(api.completions[0]["body"]["result"])
    assert result["terminal_status"] == "accepted"
    assert result["failure_diagnostic"]["reason"] == "invalid_final_checkpoint"
    captured = capsys.readouterr().err
    assert captured.count("Lab Arena execution diagnostic:") == 1
    assert "trust=untrusted" in captured
    assert "event=supervisor_failure" in captured
    assert "failure_class=timeout" in captured
    assert "host_exit_code=-" in captured
    assert "host_timed_out=true" in captured
    assert "LAB_ARENA_EXECUTION_DIAGNOSTIC" not in captured


def test_malformed_execution_diagnostic_never_reaches_journal(tmp_path, capsys):
    secret = "private-secret-class"
    stderr = (
        runner._EXECUTION_DIAGNOSTIC_PREFIX
        + b'{"event":"supervisor_failure","failure_class":"'
        + secret.encode()
        + b'","reason":"unexpected","schema_version":1}\n'
    )
    api = FakeApi([lease()])
    sandbox = runtime.FakeRuntime(
        [runtime.fake_result(output_bytes=VALID_EMPTY, stderr=stderr)]
    )
    (tmp_path / "work").mkdir()

    runner.Runner(make_config(tmp_path, api, sandbox)).run_once()

    assert api.completions[0]["body"]["result"]["terminal_status"] == "accepted"
    captured = capsys.readouterr().err
    assert "Lab Arena execution diagnostic:" not in captured
    assert secret not in captured


def test_provider_override_keeps_terminal_behavior_and_logs_telemetry(
    tmp_path, capsys
):
    class ProviderOverrideApi(RefusingApi):
        def provider(self, run_id, lease_token, frame):
            document = super().provider(run_id, lease_token, frame)
            document["call"]["error_code"] = "provider_unavailable"
            return document

    class ProviderFailureRuntime:
        @staticmethod
        def run_icp(spec, **_kwargs):
            os.environ[shim.WORKER_SOCKET_ENV] = str(spec.socket_path)
            try:
                status, _headers, _body = shim.dispatch(
                    "deepline.execute",
                    {"tool": "exa_search", "payload": {"query": "fintech"}},
                    5000,
                )
            finally:
                os.environ.pop(shim.WORKER_SOCKET_ENV, None)
            assert status == 402
            return runtime.fake_result(
                exit_code=3,
                stderr=execution_diagnostic_line(supervisor_diagnostic()),
            )

        @staticmethod
        def read_output(spec):
            return runtime.read_output(spec)

    api = ProviderOverrideApi([lease()])
    (tmp_path / "work").mkdir()

    runner.Runner(make_config(tmp_path, api, ProviderFailureRuntime())).run_once()

    result = contracts.validate_run_result(api.completions[0]["body"]["result"])
    assert result["terminal_status"] == "provider_error"
    assert "failure_diagnostic" not in result
    captured = capsys.readouterr().err
    assert "event=supervisor_failure" in captured
    assert "host_exit_code=3" in captured
    assert "host_timed_out=false" in captured


def test_execution_diagnostic_precedes_stale_completion(tmp_path, capsys):
    class StaleApi(FakeApi):
        def complete(self, envelope):
            self.completions.append(envelope)
            return {"status": "stale"}

    api = StaleApi([lease()])
    sandbox = runtime.FakeRuntime(
        [
            runtime.fake_result(
                exit_code=3,
                output_bytes=VALID_EMPTY,
            )
        ]
    )
    (tmp_path / "work").mkdir()

    runner_ = runner.Runner(make_config(tmp_path, api, sandbox))
    runner_.run_once()

    assert runner_.completed == [{"run_id": "r1", "result": {"status": "stale"}}]
    assert api.completions[0]["body"]["result"]["terminal_status"] == "accepted"
    captured = capsys.readouterr()
    assert "trust=host_observed" in captured.err
    assert "event=sandbox_outcome" in captured.err
    assert "host_exit_code=3" in captured.err
    assert "Lab Arena completion status: stale" in captured.out


def test_closed_diagnostic_journal_does_not_block_completion(tmp_path, monkeypatch):
    class ClosedStream:
        @staticmethod
        def write(_value):
            raise ValueError("closed stream")

        @staticmethod
        def flush():
            raise ValueError("closed stream")

    api = FakeApi([lease()])
    sandbox = runtime.FakeRuntime(
        [runtime.fake_result(exit_code=3, output_bytes=VALID_EMPTY)]
    )
    (tmp_path / "work").mkdir()
    monkeypatch.setattr(
        runner,
        "sys",
        SimpleNamespace(stderr=ClosedStream(), executable=runner.sys.executable),
    )

    runner.Runner(make_config(tmp_path, api, sandbox)).run_once()

    assert api.completions[0]["body"]["result"]["terminal_status"] == "accepted"


def test_valid_latest_checkpoint_clears_stale_invalid_observation(tmp_path):
    api = FakeApi([checkpoint_lease()])
    sandbox = CheckpointSequenceRuntime(
        tmp_path / "runtime-work", mode="fast", repaired=True
    )
    sandbox.work_dir.mkdir()
    (tmp_path / "work").mkdir()

    runner.Runner(make_config(tmp_path, api, sandbox)).run_once()

    result = api.completions[0]["body"]["result"]
    assert sandbox.result.checkpoint_output_invalid is False
    assert result["terminal_status"] == "accepted"
    assert "failure_diagnostic" not in result


@pytest.mark.parametrize(
    "mode, expected_terminal",
    [("fast", "model_error"), ("deadline", "model_timeout")],
)
def test_only_invalid_checkpoint_keeps_terminal_and_adds_safe_diagnostic(
    tmp_path, mode, expected_terminal
):
    api = FakeApi([checkpoint_lease()])
    sandbox = CheckpointSequenceRuntime(
        tmp_path / "runtime-work", mode=mode, initial_valid=False
    )
    sandbox.work_dir.mkdir()
    (tmp_path / "work").mkdir()

    runner.Runner(make_config(tmp_path, api, sandbox)).run_once()

    completion = api.completions[0]["body"]
    result = contracts.validate_run_result(completion["result"])
    assert sandbox.result.output_bytes is None
    assert sandbox.result.checkpoint_output_invalid is True
    assert result["terminal_status"] == expected_terminal
    assert completion["output"] is None
    assert result["failure_diagnostic"] == {
        "stage": "sandbox_output",
        "error_class": "execution_output_invalid",
        "reason": "no_valid_checkpoint",
    }
    assert "do-not-persist" not in json.dumps(completion)


def test_output_read_error_finishing_at_deadline_does_not_mark_fallback(
    tmp_path, monkeypatch
):
    config = make_runtime_config(tmp_path)
    spec = _checkpoint_spec(tmp_path, seconds=1)
    clock = FakeClock()
    actual_read_output = runtime.read_output
    reads = 0

    def delayed_read(candidate_spec):
        nonlocal reads
        reads += 1
        if reads == 3:
            clock.now = 1001.0
            raise runtime.SandboxOutputError("late unreadable output")
        return actual_read_output(candidate_spec)

    def sleep(seconds):
        clock.sleep(seconds)
        if clock() >= 1000.04 and not spec.output_path.exists():
            spec.output_path.write_bytes(VALID_EMPTY)

    monkeypatch.setattr(runtime, "read_output", delayed_read)
    result = runtime.run_sandbox(
        config,
        spec,
        process_runner=FakeRunner(
            clock,
            run_process=lambda argv: FakeProcess(
                argv, clock=clock, finish_at=None
            ),
        ),
        clock=clock,
        sleep=sleep,
        rusage=lambda: (0.0, 0),
    )

    assert result.timed_out
    assert result.output_bytes == VALID_EMPTY
    assert result.checkpoint_output_invalid is False


@pytest.mark.parametrize(
    "payload, reason",
    [
        (b'{not-json private_payload=do-not-persist', "invalid_json"),
        (
            b'{"schema_version":"private-do-not-persist","companies":[]}',
            "output_schema_mismatch",
        ),
        (
            b'{"companies":[],"private_payload":"do-not-persist"}',
            "output_contract_violation",
        ),
    ],
)
def test_invalid_execution_output_has_only_a_safe_classified_diagnostic(
    tmp_path, payload, reason
):
    api = FakeApi([lease()])
    sandbox = runtime.FakeRuntime(
        [runtime.fake_result(output_bytes=payload)]
    )
    (tmp_path / "work").mkdir()

    runner.Runner(make_config(tmp_path, api, sandbox)).run_once()

    completion = api.completions[0]["body"]
    result = contracts.validate_run_result(completion["result"])
    assert result["terminal_status"] == "invalid_output"
    assert completion["output"] is None
    assert result["failure_diagnostic"] == {
        "stage": "sandbox_output",
        "error_class": "execution_output_invalid",
        "reason": reason,
    }
    assert "do-not-persist" not in json.dumps(completion)


def test_bounded_output_read_error_has_safe_diagnostic(tmp_path):
    api = FakeApi([lease()])
    sandbox = runtime.FakeRuntime(
        [runtime.fake_result(output_error="output exceeds private 999999 bytes")]
    )
    (tmp_path / "work").mkdir()

    runner.Runner(make_config(tmp_path, api, sandbox)).run_once()

    completion = api.completions[0]["body"]
    result = contracts.validate_run_result(completion["result"])
    assert result["terminal_status"] == "invalid_output"
    assert result["failure_diagnostic"] == {
        "stage": "sandbox_output",
        "error_class": "sandbox_output_error",
    }
    assert "999999" not in json.dumps(completion)


@pytest.mark.parametrize(
    "override, expected_terminal",
    [
        ("credential", "credential_error"),
        ("provider", "provider_error"),
    ],
)
def test_infrastructure_override_does_not_keep_output_diagnostic(
    tmp_path, override, expected_terminal
):
    class OverrideApi(RefusingApi):
        def provider(self, run_id, lease_token, frame):
            document = super().provider(run_id, lease_token, frame)
            if override == "credential":
                document["call"].update(
                    funding_source="miner_key",
                    error_code="miner_credentials_unavailable",
                )
            else:
                document["call"]["error_code"] = "provider_unavailable"
            return document

    class InvalidAfterCredentialFailure:
        @staticmethod
        def run_icp(spec, **_kwargs):
            os.environ[shim.WORKER_SOCKET_ENV] = str(spec.socket_path)
            try:
                status, _headers, _body = shim.dispatch(
                    "deepline.execute",
                    {"tool": "exa_search", "payload": {"query": "fintech"}},
                    5000,
                )
            finally:
                os.environ.pop(shim.WORKER_SOCKET_ENV, None)
            assert status == 402
            return runtime.fake_result(output_bytes=b"{not-json")

        @staticmethod
        def read_output(spec):
            return runtime.read_output(spec)

    api = OverrideApi([lease()])
    (tmp_path / "work").mkdir()

    runner.Runner(
        make_config(tmp_path, api, InvalidAfterCredentialFailure())
    ).run_once()

    result = contracts.validate_run_result(api.completions[0]["body"]["result"])
    assert result["terminal_status"] == expected_terminal
    assert "failure_diagnostic" not in result


@pytest.mark.parametrize("funding_source", ["host", "miner_key"])
@pytest.mark.parametrize(
    "mode, expected_terminal",
    [
        ("accepted", "accepted"),
        ("invalid", "invalid_output"),
        ("model_error", "model_error"),
    ],
)
def test_provider_request_refusal_keeps_the_models_own_terminal_result(
    tmp_path, funding_source, mode, expected_terminal
):
    """An ordinary provider rejection must not become infrastructure failure."""

    class RequestRefusedApi(FakeApi):
        def provider(self, run_id, lease_token, frame):
            self.provider_frames.append(dict(frame))
            body = b'{"error":{"code":"provider_request_refused"}}'
            return {
                "status": 403,
                "headers": {
                    "content-type": "application/json",
                    "content-length": str(len(body)),
                },
                "body_b64": base64.b64encode(body).decode(),
                "call": {
                    "call_identity": contracts.document_hash(
                        ["request-refused", frame["action_sequence"]]
                    ),
                    "operation_id": frame["operation_id"],
                    "provider": "deepline",
                    "funding_source": funding_source,
                    "reserved_microusd": 0,
                    "actual_microusd": 0,
                    "outcome": "settled",
                    "status": 403,
                    "provider_status": 403,
                    "error_code": "provider_request_refused",
                },
            }

    class RequestRefusedRuntime:
        @staticmethod
        def run_icp(spec, **_kwargs):
            os.environ[shim.WORKER_SOCKET_ENV] = str(spec.socket_path)
            try:
                status, _headers, _body = shim.dispatch(
                    "deepline.execute",
                    {
                        "tool": "generic_http_request",
                        "payload": {
                            "url": "https://public.example/status/403",
                            "method": "GET",
                        },
                    },
                    5000,
                )
            finally:
                os.environ.pop(shim.WORKER_SOCKET_ENV, None)
            assert status == 403
            if mode == "accepted":
                return runtime.fake_result(output_bytes=VALID_EMPTY)
            if mode == "invalid":
                return runtime.fake_result(output_bytes=b"{not-json")
            return runtime.fake_result(exit_code=1)

        @staticmethod
        def read_output(spec):
            return runtime.read_output(spec)

    api = RequestRefusedApi([lease()])
    (tmp_path / "work").mkdir()

    runner.Runner(make_config(tmp_path, api, RequestRefusedRuntime())).run_once()

    completion = api.completions[0]["body"]
    result = contracts.validate_run_result(completion["result"])
    assert result["terminal_status"] == expected_terminal
    assert len(api.provider_frames) == 1
    if mode == "invalid":
        assert result["failure_diagnostic"] == {
            "stage": "sandbox_output",
            "error_class": "execution_output_invalid",
            "reason": "invalid_json",
        }
    else:
        assert "failure_diagnostic" not in result


def test_safe_diagnostic_is_private_until_existing_public_results_release():
    diagnostic = {
        "stage": "sandbox_output",
        "error_class": "execution_output_invalid",
        "reason": "invalid_final_checkpoint",
    }
    run_result = contracts.validate_run_result(
        {
            "schema_version": contracts.RUN_RESULT_SCHEMA_VERSION,
            "resource_summary": {
                "wall_seconds": 1.0,
                "cpu_seconds": 0.5,
                "max_rss_bytes": 1024,
                "stdout_bytes": 0,
                "stderr_bytes": 0,
                "provider_call_count": 0,
            },
            "started_at": "2026-09-16T01:00:00Z",
            "finished_at": "2026-09-16T01:00:01Z",
            "terminal_status": "accepted",
            "failure_diagnostic": diagnostic,
        }
    )
    round_row = {
        "round_id": "arena-2026-09-16",
        "status": "stage1",
        "configuration_doc": {},
        "publication_doc": {
            "participants": [
                {
                    "submission_id": "submission-1",
                    "miner_hotkey": "5" * 48,
                    "is_baseline": False,
                }
            ],
            "stage1_ranking": [],
            "final_ranking": [],
        },
    }
    service = object.__new__(ArenaService)
    service._round = lambda _round_id: round_row
    service._public_icp_disclosure = lambda _row: {
        "public_positions": [0]
    }
    service._store = SimpleNamespace(
        list_runs=lambda *_args, **_kwargs: [
            {
                "run_id": "run-1",
                "icp_position": 0,
                "stage": 1,
                "per_icp_score": None,
                "output_ref": "",
                "result_doc": run_result,
            }
        ]
    )

    with pytest.raises(ServiceError, match="results_not_public"):
        service.public_results("arena-2026-09-16", "submission-1")

    round_row["status"] = "published"
    public = service.public_results("arena-2026-09-16", "submission-1")
    assert public["run_results"][0]["failure_diagnostic"] == diagnostic
