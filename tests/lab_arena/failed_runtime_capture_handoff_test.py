"""Runtime failures retain passive web evidence without a run result."""

import json
import signal
from types import SimpleNamespace

import pytest

from lab_arena import automatic_capture, contracts, lab_arena_checkpoint, runner, runtime
from lab_arena.runtime_host import runtime_host_private_diagnostic
from tests.lab_arena.automatic_capture_integration_test import capture
from tests.lab_arena.test_lab_arena_runner import BridgingRuntime, FakeApi, lease, make_config
from tests.lab_arena.test_lab_arena_runtime import (
    FakeClock, FakeProcess, FakeRunner, make_config as runtime_config, make_spec,
)


TEXT = "Exact page consumed before host signal"


def _frame():
    return json.dumps(lab_arena_checkpoint.validate_training_frame({
        "schema_version": lab_arena_checkpoint.TRAINING_SCHEMA_VERSION,
        "control": lab_arena_checkpoint.TRAINING_CONTROL,
        "kind": "web_fetch", "depends_on": [], "retry_token": "3" * 32,
        "data": {"url": "https://example.com/page", "final_url": "https://example.com/page",
                 "status": 200, "content_type": "text/plain", "body": TEXT,
                 "retrieved_at": "2026-11-10T10:00:00+00:00", "capture": capture(TEXT)},
    }), sort_keys=True, separators=(",", ":")).encode()


def _failed_error(tmp_path, *, kind="signal", cleanup=None, monkeypatch=None, raw=None,
                  capture_enabled=True):
    checkpoint = tmp_path / "trusted-checkpoint.py"
    automatic = tmp_path / "trusted-capture.py"
    checkpoint.write_bytes(runner.CHECKPOINT_MODULE_PATH.read_bytes())
    automatic.write_bytes(runner.AUTOMATIC_CAPTURE_PATH.read_bytes())
    checkpoint.chmod(0o444)
    automatic.chmod(0o444)
    spec = make_spec(tmp_path, checkpoint_module_path=checkpoint,
                     automatic_capture_path=automatic if capture_enabled else None)
    clock = FakeClock()

    def launch(argv):
        # No successful execution output is needed for this consumed evidence.
        (spec.output_dir / (automatic_capture.HANDOFF_PREFIX + "00000000")).write_bytes(_frame() if raw is None else raw)
        if kind == "launch_oserror":
            raise OSError("simulated launcher failure")
        return FakeProcess(argv, clock=clock, finish_at=clock() + 1,
                           returncode=-signal.SIGINT if kind == "signal" else 0)

    if kind == "pipe":
        assert monkeypatch is not None
        def failed_drain(stream, capture, stop):
            capture.failed = True
        monkeypatch.setattr(runtime, "_drain", failed_drain)
    processes = FakeRunner(clock, run_process=launch, fail={cleanup} if cleanup else ())
    expected = (runtime.SandboxCleanupError if cleanup else
                runtime.RuntimeHostError if kind == "signal" else
                OSError if kind == "launch_oserror" else runtime.ArenaRuntimeError)
    with pytest.raises(expected) as failure:
        runtime.run_sandbox(runtime_config(tmp_path), spec, process_runner=processes,
                            clock=clock, sleep=clock.sleep, rusage=lambda: (0.0, 0))
    assert processes.kinds() == ["mount", "run", "delete", "umount"]
    assert not spec.output_dir.exists()
    return failure.value


def test_signaled_launcher_retains_bounded_handoff_before_cleanup(tmp_path):
    error = _failed_error(tmp_path)
    assert error.reason == "sandbox_launcher_signaled"
    assert error.launch_exit_code == -signal.SIGINT
    assert error.checkpoint_bytes is None
    assert error.automatic_capture_handoff == (_frame(),)
    assert error.automatic_capture_handoff_invalid is False
    assert error.automatic_capture_summary == {
        "source": "runtime_http_v1", "complete": False, "failure": "summary_unavailable"}
    assert TEXT not in repr(error)
    assert TEXT not in runtime_host_private_diagnostic(error)


@pytest.mark.parametrize("kind,cleanup", [
    ("signal", None), ("signal", "delete"), ("signal", "umount"),
    ("pipe", None), ("pipe", "delete"), ("pipe", "umount"),
])
def test_runner_imports_failed_handoff_and_still_abandons(tmp_path, monkeypatch, kind, cleanup):
    monkeypatch.setattr(runner.shutil, "disk_usage", lambda _: SimpleNamespace(free=10**12))
    failed = tmp_path / "failed"
    failed.mkdir()
    error = _failed_error(failed, kind=kind, cleanup=cleanup, monkeypatch=monkeypatch)

    assert error.automatic_capture_handoff == (_frame(),)
    if cleanup:
        assert error.result is None
        assert isinstance(error, runtime.SandboxCleanupError)
    elif kind == "pipe":
        assert str(error) == "sandbox pipe read failed"

    class CapturedApi(FakeApi):
        def training_observations(self, run_id, token, document):
            self.saved = document
            return {"status": "stored", "payload_sha256": contracts.document_hash(document)}

    class SignaledRuntime(BridgingRuntime):
        def run_icp(self, spec, **kwargs):
            super().run_icp(spec, **kwargs)
            raise error

    api = CapturedApi([lease()])
    (tmp_path / "work").mkdir()
    worker = runner.Runner(make_config(tmp_path, api, SignaledRuntime(output={"companies": []}, calls=1)))
    try:
        assert worker.run_once() == 1
    finally:
        worker.close()
    assert worker.abandoned == 1
    assert api.completions == []
    assert len(api.provider_frames) == 1
    rows = [row for row in api.saved["agent_observations"] if row["kind"] == "web_fetch"]
    assert len(rows) == 1 and rows[0]["data"]["body"] == TEXT
    assert rows[0]["data"]["capture"]["received_body_sha256"] == contracts.hash_bytes(TEXT.encode())
    assert api.saved["complete"] is False
    assert "capture_incomplete" in api.saved["failures"]


@pytest.mark.parametrize("reader", ["handoff", "summary"])
def test_optional_capture_read_failure_preserves_original_runtime_error(tmp_path, monkeypatch, reader):
    def failed_read(_spec):
        raise ValueError("optional read failed")
    monkeypatch.setattr(runtime, "read_automatic_capture_" + reader, failed_read)
    error = _failed_error(tmp_path)
    assert error.reason == "sandbox_launcher_signaled"
    assert error.launch_exit_code == -signal.SIGINT
    if reader == "handoff":
        assert error.automatic_capture_handoff == ()
        assert error.automatic_capture_handoff_invalid is True
    else:
        assert error.automatic_capture_handoff == (_frame(),)
        assert error.automatic_capture_summary["complete"] is False


def test_invalid_handoff_on_failed_run_stays_incomplete_and_cannot_add_web_evidence(tmp_path, monkeypatch):
    monkeypatch.setattr(runner.shutil, "disk_usage", lambda _: SimpleNamespace(free=10**12))
    failed = tmp_path / "failed"
    failed.mkdir()
    error = _failed_error(failed, cleanup="delete", raw=b"not a training frame")

    class CapturedApi(FakeApi):
        def training_observations(self, run_id, token, document):
            self.saved = document
            return {"status": "stored", "payload_sha256": contracts.document_hash(document)}

    class FailedRuntime(BridgingRuntime):
        def run_icp(self, spec, **kwargs):
            super().run_icp(spec, **kwargs)
            raise error

    api = CapturedApi([lease()])
    (tmp_path / "work").mkdir()
    worker = runner.Runner(make_config(tmp_path, api, FailedRuntime(output={"companies": []}, calls=1)))
    try:
        assert worker.run_once() == 1
    finally:
        worker.close()
    assert worker.abandoned == 1 and api.completions == []
    assert len(api.provider_frames) == 1
    assert not [row for row in api.saved["agent_observations"] if row["kind"] == "web_fetch"]
    assert api.saved["complete"] is False
    assert "capture_incomplete" in api.saved["failures"]


def test_failed_runtime_without_capture_mount_has_no_passive_evidence(tmp_path):
    error = _failed_error(tmp_path, capture_enabled=False)
    assert error.reason == "sandbox_launcher_signaled"
    assert error.automatic_capture_handoff == ()
    assert error.automatic_capture_handoff_invalid is False
    assert error.automatic_capture_summary is None


def test_failed_launch_without_process_does_not_import_staged_bytes(tmp_path):
    error = _failed_error(tmp_path, kind="launch_oserror")
    assert str(error) == "simulated launcher failure"
    assert not getattr(error, "automatic_capture_handoff", ())
    assert getattr(error, "automatic_capture_summary", None) is None


def test_rejected_output_setup_does_not_import_stale_handoff(tmp_path):
    checkpoint = tmp_path / "trusted-checkpoint.py"
    automatic = tmp_path / "trusted-capture.py"
    checkpoint.write_bytes(runner.CHECKPOINT_MODULE_PATH.read_bytes())
    automatic.write_bytes(runner.AUTOMATIC_CAPTURE_PATH.read_bytes())
    checkpoint.chmod(0o444)
    automatic.chmod(0o444)
    spec = make_spec(tmp_path, checkpoint_module_path=checkpoint, automatic_capture_path=automatic)
    spec.output_dir.mkdir()
    (spec.output_dir / (automatic_capture.HANDOFF_PREFIX + "00000000")).write_bytes(_frame())
    clock = FakeClock()
    processes = FakeRunner(clock)
    with pytest.raises(runtime.SandboxSpecError) as failure:
        runtime.run_sandbox(runtime_config(tmp_path), spec, process_runner=processes,
                            clock=clock, sleep=clock.sleep, rusage=lambda: (0.0, 0))
    assert processes.kinds() == ["delete"]
    assert str(failure.value) == "output directory must be empty"
    assert not getattr(failure.value, "automatic_capture_handoff", ())
    assert getattr(failure.value, "automatic_capture_summary", None) is None
