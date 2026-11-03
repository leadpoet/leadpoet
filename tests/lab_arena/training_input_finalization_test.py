"""Retry only failed input evidence before ordinary terminal finalization."""

import copy
import json
from types import SimpleNamespace

import httpx
import pytest

from lab_arena import contracts, runner, runtime, scoring, training_input
from lab_arena.service import ServiceError
from lab_arena.store import hash_lease_token
from tests.lab_arena.test_lab_arena_runner import (
    BridgingRuntime, FakeApi, lease, make_config, scoring_lease, valid_company,
)
from tests.lab_arena.training_input_test import CLAIM_HASH, NOW, TOKEN, service_for


def setup_run(tmp_path, kind, mode, monkeypatch, *, runtime_fails=False, stop_fails=False):
    active = scoring_lease(run_id="r1") if kind == "score" else lease()
    active["lease_token"] = TOKEN
    active["icp"]["prompt"] = "exact é\ninput"
    service, objects = service_for(tmp_path, kind=kind)
    timeline, attempts, identities, runtime_bytes = [], [], [], []

    class CaptureApi(FakeApi):
        def training_input(self, run_id, token, document):
            timeline.append("input")
            identities.append(id(document))
            attempts.append(copy.deepcopy(document))
            if mode == "ambiguous" and len(attempts) == 1:
                service.handle_training_input(run_id, token, document)
                self.initial_object = objects.get(training_input.object_ref(run_id, 1))
            if mode == "always_fail" or (mode in {"recover", "ambiguous"} and len(attempts) <= 2):
                raise httpx.ReadTimeout("synthetic private failure")
            return service.handle_training_input(run_id, token, document)

        def training_observations(self, *_args):
            timeline.append("observations")
            # This test exercises input storage, not the separate observation API.
            raise RuntimeError("synthetic unavailable observation API")

        def trajectory(self, _run_id, _token, events):
            if any(event["kind"] == "runtime.error" for event in events):
                timeline.append("runtime.error")
            return {"accepted": len(events)}

    class Journal:
        def __init__(self, *_args):
            pass

        def start(self):
            pass

        def stop(self):
            timeline.append("journal.stop")

    class Sandbox(BridgingRuntime):
        def run_icp(self, spec, **kwargs):
            timeline.append("runtime")
            runtime_bytes.append((spec.input_dir / runtime.INPUT_FILE_NAME).read_bytes())
            result = super().run_icp(spec, **kwargs)
            if runtime_fails:
                raise ValueError("synthetic primary failure")
            return result

    monkeypatch.setattr(runner, "_ObservationJournal", Journal)
    if stop_fails:
        original_stop = runner.WorkerSocketServer.stop

        def fail_stop(server):
            original_stop(server)
            raise OSError("synthetic cleanup failure")

        monkeypatch.setattr(runner.WorkerSocketServer, "stop", fail_stop)
    output = (scoring.build_scoring_output("r1", [
        {"final_score": 71.0, "failure_reason": ""},
        {"final_score": 44.5, "failure_reason": ""},
    ]) if kind == "score" else {"companies": [valid_company(1)]})
    api = CaptureApi([])
    sandbox = Sandbox(output=output, calls=1)
    (tmp_path / "work").mkdir()
    config = make_config(tmp_path, api, sandbox)
    execute = lambda: runner.AssignmentExecutor(config).execute(active, TOKEN, active["icp"])
    return execute, api, sandbox, service, objects, timeline, attempts, identities, runtime_bytes


@pytest.mark.parametrize("kind", ["execute", "score"])
@pytest.mark.parametrize("mode", ["recover", "ambiguous"])
def test_failed_initial_input_retries_exact_document_before_finalization(tmp_path, kind, mode, monkeypatch):
    execute, api, sandbox, _service, objects, timeline, attempts, identities, raw = setup_run(
        tmp_path, kind, mode, monkeypatch,
    )
    original_wait = runner.WorkerSocketServer.wait_for_handlers

    def wait_for_handlers(server):
        original_wait(server)
        timeline.append("handlers.drained")

    monkeypatch.setattr(runner.WorkerSocketServer, "wait_for_handlers", wait_for_handlers)
    envelope = execute()
    assert timeline == ["input", "input", "runtime", "handlers.drained", "input", "journal.stop", "observations", "observations"]
    assert len(attempts) == 3 and len(set(identities)) == 1
    document = json.loads(raw[0])
    assert all(attempt == document for attempt in attempts)
    assert document == sandbox.input_documents[0]
    assert raw[0] == json.dumps(attempts[-1], sort_keys=True).encode("utf-8")
    saved = training_input.read_input(objects, "r1", kind, 1, CLAIM_HASH)
    assert saved["input"] == document
    assert saved["input_sha256"] == contracts.document_hash(document)
    assert saved["privacy"] == {"redacted": False, "redactions": []}
    assert len(list((tmp_path / "objects").rglob("*.gz"))) == 1
    if mode == "ambiguous":
        assert objects.get(training_input.object_ref("r1", 1)) == api.initial_object
    assert envelope["body"]["result"]["terminal_status"] == "accepted"
    assert envelope["body"]["result"]["resource_summary"]["provider_call_count"] == 1
    assert len(api.provider_frames) == 1
    assert not list((tmp_path / "work").iterdir())


@pytest.mark.parametrize("kind", ["execute", "score"])
@pytest.mark.parametrize("available", [False, True])
def test_successful_or_unavailable_input_has_no_terminal_retry(tmp_path, kind, available, monkeypatch):
    execute, api, _sandbox, _service, _objects, timeline, attempts, *_ = setup_run(
        tmp_path, kind, "stored", monkeypatch,
    )
    if not available:
        api.training_input = None
    assert execute()["body"]["result"]["terminal_status"] == "accepted"
    assert len(attempts) == int(available)
    assert timeline.index("journal.stop") > timeline.index("runtime")


@pytest.mark.parametrize("kind", ["execute", "score"])
def test_recovered_input_preserves_output_calls_cost_and_result(tmp_path, kind, monkeypatch):
    bodies, frames = [], []
    for mode in ("stored", "recover"):
        directory = tmp_path / mode
        directory.mkdir()
        execute, api, *_ = setup_run(directory, kind, mode, monkeypatch)
        bodies.append(execute()["body"])
        frames.append(api.provider_frames)
    assert bodies[0] == bodies[1]
    assert frames[0] == frames[1]


@pytest.mark.parametrize("kind", ["execute", "score"])
@pytest.mark.parametrize("runtime_fails,stop_fails", [(False, False), (True, False), (False, True), (True, True)])
def test_final_upload_failure_is_bounded_and_preserves_primary_result(
    tmp_path, kind, runtime_fails, stop_fails, monkeypatch,
):
    execute, api, _sandbox, _service, _objects, timeline, attempts, *_ = setup_run(
        tmp_path, kind, "always_fail", monkeypatch,
        runtime_fails=runtime_fails, stop_fails=stop_fails,
    )
    if runtime_fails or stop_fails:
        with pytest.raises(ValueError if runtime_fails else OSError):
            execute()
        assert timeline.index("input", 2) < timeline.index("runtime.error")
    else:
        assert execute()["body"]["result"]["terminal_status"] == "accepted"
    assert len(attempts) == 3
    assert len(api.provider_frames) == 1
    assert timeline.index("input", 2) < timeline.index("journal.stop")
    assert not list((tmp_path / "work").iterdir())


@pytest.mark.parametrize("kind", ["execute", "score"])
def test_failure_before_input_write_has_no_upload_or_retry(tmp_path, kind, monkeypatch):
    execute, _api, _sandbox, _service, _objects, _timeline, attempts, *_ = setup_run(
        tmp_path, kind, "recover", monkeypatch,
    )
    original_write = runner.Path.write_text

    def fail_input_write(path, *args, **kwargs):
        if path.name == runtime.INPUT_FILE_NAME:
            raise OSError("synthetic input write failure")
        return original_write(path, *args, **kwargs)

    monkeypatch.setattr(runner.Path, "write_text", fail_input_write)
    with pytest.raises(OSError):
        execute()
    assert attempts == []
    assert not list((tmp_path / "work").iterdir())


@pytest.mark.parametrize("kind", ["execute", "score"])
def test_setup_failure_after_input_write_retries_before_failure_reporting(tmp_path, kind, monkeypatch):
    execute, _api, _sandbox, _service, objects, timeline, attempts, *_ = setup_run(
        tmp_path, kind, "recover", monkeypatch,
    )

    def fail_image_check(*_args):
        raise ValueError("synthetic setup failure")

    monkeypatch.setattr(runner, "_check_runtime_image", fail_image_check)
    with pytest.raises(ValueError, match="synthetic setup failure"):
        execute()
    assert "runtime" not in timeline and len(attempts) == 3
    assert timeline.index("input", 2) < timeline.index("runtime.error")
    assert training_input.read_input(objects, "r1", kind, 1, CLAIM_HASH)["input"] == attempts[0]
    assert not list((tmp_path / "work").iterdir())


@pytest.mark.parametrize("mode,expected,count", [
    ("absent", None, 0), ("stored", True, 1),
    ("timeout", False, 2), ("rejected", False, 1),
])
def test_input_helper_reports_outcome_without_changing_initial_attempts(mode, expected, count):
    attempts = []

    def send(*args):
        attempts.append(args)
        if mode == "timeout":
            raise httpx.ReadTimeout("synthetic failure")
        return {"status": "stored" if mode == "stored" else "unavailable"}

    config = SimpleNamespace(api=SimpleNamespace(training_input=None if mode == "absent" else send))
    assert runner._record_training_input(config, {"run_id": "r1"}, TOKEN, {}) is expected
    assert len(attempts) == count


@pytest.mark.parametrize("change,expected", [
    ({"lease_expires_at": NOW.isoformat()}, "lease_expired"),
    ({"status": "accepted"}, "lease_inactive"),
    ({"lease_token_hash": hash_lease_token("b" * 64)}, "lease_invalid"),
    ({"lease_generation": 0}, "training_input_invalid"),
    ({"lease_generation": 2, "lease_token_hash": hash_lease_token("b" * 64)}, "lease_invalid"),
    ({}, "training_input_conflict"),
])
def test_terminal_retry_keeps_lease_and_immutable_storage_checks(tmp_path, change, expected, monkeypatch):
    execute, api, sandbox, service, objects, _timeline, attempts, *_ = setup_run(
        tmp_path, "execute", "recover", monkeypatch,
    )
    rejections, original_run = [], sandbox.run_icp
    original_send = api.training_input
    prior = None
    if not change:
        service.handle_training_input("r1", TOKEN, {"different": "original input"})
        prior = objects.get(training_input.object_ref("r1", 1))

    def run_and_change_lease(spec, **kwargs):
        result = original_run(spec, **kwargs)
        service._store.get_run("r1").update(change)
        return result

    def send(*args):
        try:
            return original_send(*args)
        except ServiceError as exc:
            rejections.append(exc.code)
            raise

    monkeypatch.setattr(sandbox, "run_icp", run_and_change_lease)
    monkeypatch.setattr(api, "training_input", send)
    assert execute()["body"]["result"]["terminal_status"] == "accepted"
    assert len(attempts) == 3 and rejections == [expected]
    if prior is not None:
        assert objects.get(training_input.object_ref("r1", 1)) == prior
    else:
        assert not list((tmp_path / "objects").rglob("*.gz"))
