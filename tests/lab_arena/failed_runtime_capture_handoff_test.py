"""A failed launcher can retain passive web evidence without a run result."""

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


def _signaled_error(tmp_path):
    checkpoint = tmp_path / "trusted-checkpoint.py"
    automatic = tmp_path / "trusted-capture.py"
    checkpoint.write_bytes(runner.CHECKPOINT_MODULE_PATH.read_bytes())
    automatic.write_bytes(runner.AUTOMATIC_CAPTURE_PATH.read_bytes())
    checkpoint.chmod(0o444)
    automatic.chmod(0o444)
    spec = make_spec(tmp_path, checkpoint_module_path=checkpoint, automatic_capture_path=automatic)
    clock = FakeClock()

    def launch(argv):
        # No successful execution output is needed for this consumed evidence.
        (spec.output_dir / (automatic_capture.HANDOFF_PREFIX + "00000000")).write_bytes(_frame())
        return FakeProcess(argv, clock=clock, finish_at=clock() + 1, returncode=-signal.SIGINT)

    processes = FakeRunner(clock, run_process=launch)
    with pytest.raises(runtime.RuntimeHostError) as failure:
        runtime.run_sandbox(runtime_config(tmp_path), spec, process_runner=processes,
                            clock=clock, sleep=clock.sleep, rusage=lambda: (0.0, 0))
    assert processes.kinds() == ["mount", "run", "delete", "umount"]
    assert not spec.output_dir.exists()
    return failure.value


def test_signaled_launcher_retains_bounded_handoff_before_cleanup(tmp_path):
    error = _signaled_error(tmp_path)
    assert error.reason == "sandbox_launcher_signaled"
    assert error.launch_exit_code == -signal.SIGINT
    assert error.checkpoint_bytes is None
    assert error.automatic_capture_handoff == (_frame(),)
    assert error.automatic_capture_handoff_invalid is False
    assert error.automatic_capture_summary == {
        "source": "runtime_http_v1", "complete": False, "failure": "summary_unavailable"}
    assert TEXT not in repr(error)
    assert TEXT not in runtime_host_private_diagnostic(error)


def test_runner_imports_signaled_handoff_and_still_abandons(tmp_path, monkeypatch):
    monkeypatch.setattr(runner.shutil, "disk_usage", lambda _: SimpleNamespace(free=10**12))
    failed = tmp_path / "failed"
    failed.mkdir()
    error = _signaled_error(failed)

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
