"""Runner: claims, local slots, socket bridge, and per-ICP run results."""

from __future__ import annotations

import base64
from concurrent.futures import ThreadPoolExecutor
import gzip
import io
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import tarfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import pytest
import httpx
from bittensor_wallet import Keypair

from lab_arena import contracts, runner as rn, runtime, shim, source_bundle
from lab_arena.output import output_document_from_bytes

RUNNER = Keypair.create_from_uri("//Runner")
MINER = Keypair.create_from_uri("//Miner").ss58_address
ROUND = "arena-2026-09-02"
IMAGE = "sha256:" + "a" * 64
ARENA_REPOSITORY = "arena.example/lab-arena/models"
IMAGE_REFERENCE = ARENA_REPOSITORY + "@" + IMAGE


def _source_archive() -> bytes:
    raw = io.BytesIO()
    with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as compressed:
        with tarfile.open(fileobj=compressed, mode="w") as archive:
            data = b"def run_icp(icp):\n    return []\n"
            info = tarfile.TarInfo("harness.py")
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
    return raw.getvalue()


def _source_archive_with_requirements(requirement: str) -> bytes:
    raw = io.BytesIO()
    with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as compressed:
        with tarfile.open(fileobj=compressed, mode="w") as archive:
            for name, data in (
                ("harness.py", b"def run_icp(icp):\n    return []\n"),
                ("requirements.txt", requirement.encode("utf-8")),
            ):
                info = tarfile.TarInfo(name)
                info.size = len(data)
                archive.addfile(info, io.BytesIO(data))
    return raw.getvalue()


SOURCE_BYTES = _source_archive()
SOURCE_REF = "arena/%s/sources/s1.tar.gz" % ROUND


def sign(message: str) -> str:
    return RUNNER.sign(message.encode("utf-8")).hex()


def verify(hotkey: str, signature: str, message: str) -> bool:
    raw = bytes.fromhex(signature[2:] if signature.startswith("0x") else signature)
    return bool(Keypair(ss58_address=hotkey).verify(message.encode("utf-8"), raw))


def valid_company(index: int) -> Dict[str, Any]:
    return {
        "company_name": "Co %d" % index,
        "company_website": "https://co%d.example.com" % index,
        "company_linkedin": "",
        "industry": "Software",
        "employee_count": "51-200",
        "company_stage": "Series A",
        "country": "United States",
        "state": "",
        "fit_summary": "The company matches the ICP.",
        "fit_evidence_urls": ["https://co%d.example.com/about" % index],
        "intent_signals": [{"description": "Raised a round", "url": "https://news.example.com/%d" % index, "date": "2026-08-01", "why_now": "The funding makes outreach timely.", "snippet": "Co raised money", "matched_icp_signal": 0}],
    }


class FakeApi:
    """In-process stand-in for the service's runner endpoints."""

    def __init__(self, leases: List[Dict[str, Any]], *, broker_documents=None):
        self.leases = list(leases)
        self.claims: List[Dict[str, Any]] = []
        self.provider_frames: List[Dict[str, Any]] = []
        self.completions: List[Dict[str, Any]] = []
        self.source_requests: List[tuple[str, str]] = []
        self.source_payload = SOURCE_BYTES
        self.broker_documents = list(broker_documents or [])
        self.lock = threading.Lock()

    def claim(self, envelope):
        self.claims.append(envelope)
        assert envelope["scope"] == contracts.SCOPE_CLAIM and verify(envelope["hotkey"], envelope["signature"], contracts.signed_request_message(envelope))
        if not self.leases:
            return {"status": "no_pending"}
        lease = self.leases.pop(0)
        return lease

    def provider(self, run_id, lease_token, frame):
        self.provider_frames.append(dict(frame))
        call = {"call_identity": contracts.document_hash(["call", frame["action_sequence"]]), "operation_id": frame["operation_id"], "reserved_microusd": 5000, "actual_microusd": 5000, "outcome": "settled", "status": 200, "request_hash": contracts.document_hash(frame["parameters"]), "response_hash": contracts.document_hash("resp")}
        body = json.dumps({"results": [{"url": "https://co1.example.com"}]}).encode()
        return {"status": 200, "headers": {"content-type": "application/json", "content-length": str(len(body))}, "body_b64": base64.b64encode(body).decode(), "call": call}

    def complete(self, envelope):
        self.completions.append(envelope)
        return {"status": "accepted"}

    def source(self, run_id, lease_token):
        self.source_requests.append((run_id, lease_token))
        return self.source_payload


def lease(run_id="r1", position=0):
    return {
        "status": "leased", "run_id": run_id, "assignment_id": "%s:s1:1:%d" % (ROUND, position), "submission_id": "s1", "miner_hotkey": MINER,
        "image_digest": IMAGE, "stage": 1, "icp_position": position, "attempt": 1,
        "lease_generation": 1, "stage_generation": 1, "lease_expires_at": "2026-09-02T01:07:00+00:00",
        "lease_token": "tok-" + run_id, "icp": {"icp_id": "arena:x", "prompt": "p", "max_companies": 5},
        "image_reference": IMAGE_REFERENCE,
        "source_ref": SOURCE_REF,
        "source_size_bytes": len(SOURCE_BYTES),
        "round_id": ROUND, "evaluation_date": "2026-09-02",
    }


class BridgingRuntime:
    """A fake sandbox that behaves like a model: talks to the worker socket, writes output."""

    def __init__(self, *, output: Any = None, timed_out=False, exit_code=0, calls=1):
        self.output = output
        self.timed_out = timed_out
        self.exit_code = exit_code
        self.calls = calls
        self.specs: List[runtime.SandboxSpec] = []

    def run_icp(self, spec, **_):
        self.specs.append(spec)
        os.environ[shim.WORKER_SOCKET_ENV] = str(spec.socket_path)
        try:
            for _ in range(self.calls):
                status, headers, body = shim.dispatch("deepline.execute", {"tool": "exa_search", "payload": {"query": "fintech"}}, 5000)
                assert status == 200 and json.loads(body)["results"]
        finally:
            os.environ.pop(shim.WORKER_SOCKET_ENV, None)
        if self.output is not None:
            spec.output_path.write_bytes(self.output if isinstance(self.output, bytes) else json.dumps(self.output).encode())
        return runtime.fake_result(exit_code=self.exit_code, timed_out=self.timed_out, output_bytes=runtime.read_output(spec), stdout=b"model log line\n", stderr=b"")


def fake_exporter(reference, digest, target):
    (target / "rootfs").mkdir()


def make_config(tmp_path, api, sandbox_runtime, *, parallel=1):
    cache = rn.ImageCache(tmp_path / "images", fake_exporter)
    source_cache = rn.SourceCache(
        tmp_path / "sources", api.source, dependency_installer=lambda _requirements, _target: None
    )
    return rn.RunnerConfig(
        round_id=ROUND, identity=rn.RunnerIdentity(hotkey=RUNNER.ss58_address, sign=sign), api=api, sandbox_runtime=sandbox_runtime,
        image_cache=cache, source_cache=source_cache, work_dir=tmp_path / "work", max_parallel_runs=parallel,
        evaluation_date="2026-09-02", clock=lambda: datetime(2026, 9, 2, 1, 0, tzinfo=timezone.utc), completion_retry_seconds=(0.0, 0.0),
    )


def test_accepted_run_bridges_provider_calls_and_returns_a_small_result(tmp_path):
    api = FakeApi([lease()])
    sandbox = BridgingRuntime(output={"companies": [valid_company(1), valid_company(2)]}, calls=2)
    (tmp_path / "work").mkdir()
    runner_ = rn.Runner(make_config(tmp_path, api, sandbox))
    assert runner_.run_once() == 1
    envelope = api.completions[0]
    assert envelope["scope"] == contracts.SCOPE_COMPLETE and verify(envelope["hotkey"], envelope["signature"], contracts.signed_request_message(envelope))
    validated = contracts.validate_run_result(envelope["body"]["result"])
    assert envelope["body"]["run_id"] == "r1" and validated["terminal_status"] == "accepted"
    assert validated["resource_summary"]["provider_call_count"] == 2
    frames = api.provider_frames
    assert [f["action_sequence"] for f in frames] == [0, 1] and all(set(f) == {"operation_id", "parameters", "timeout_ms", "action_sequence"} for f in frames)
    output = envelope["body"]["output"]
    assert len(output["companies"]) == 2
    spec = sandbox.specs[0]
    assert spec.rootfs_path == tmp_path / "images" / IMAGE.replace(":", "-") / "rootfs" and spec.wall_clock_seconds == contracts.ICP_WALL_CLOCK_SECONDS
    assert spec.entry_command == runtime.AGENT_ENTRY_COMMAND and spec.working_dir == runtime.AGENT_WORKING_DIR
    assert spec.source_dir is not None and (spec.source_dir / "harness.py").is_file()
    assert spec.dependency_dir is not None
    assert spec.agent_entrypoint_path == rn.AGENT_ENTRYPOINT_PATH
    assert not spec.input_dir.exists()  # run directory cleaned
    assert runner_.abandoned == 0


@pytest.mark.parametrize("kind,expected", [
    ("timeout", "model_timeout"),
    ("invalid_json", "invalid_output"),
    ("unknown_field", "invalid_output"),
    ("no_output", "model_error"),
])
def test_model_failures_map_to_terminal_causes_with_no_output_hash(tmp_path, kind, expected):
    api = FakeApi([lease()])
    if kind == "timeout":
        sandbox = BridgingRuntime(output=None, timed_out=True, exit_code=None)
    elif kind == "invalid_json":
        sandbox = BridgingRuntime(output=b"{not json")
    elif kind == "unknown_field":
        sandbox = BridgingRuntime(output={"companies": [dict(valid_company(1), extra_field="x")]})
    else:
        sandbox = BridgingRuntime(output=None, exit_code=3)
    (tmp_path / "work").mkdir()
    rn.Runner(make_config(tmp_path, api, sandbox)).run_once()
    result = api.completions[0]["body"]["result"]
    assert result["terminal_status"] == expected
    assert api.completions[0]["body"]["output"] is None


def test_local_slots_bound_claims_and_images_export_once(tmp_path):
    leases = [lease("r%d" % i, i) for i in range(5)]
    api = FakeApi(leases)
    exports = []
    cache = rn.ImageCache(tmp_path / "images", lambda reference, digest, target: exports.append(digest) or (target / "rootfs").mkdir())
    (tmp_path / "work").mkdir()
    config = make_config(tmp_path, api, BridgingRuntime(output={"companies": [valid_company(1)]}), parallel=3)
    config.image_cache = cache
    runner_ = rn.Runner(config)
    assert runner_.run_once() == 3  # three local slots, then the loop waits for completions
    assert exports == [IMAGE]
    assert len(api.source_requests) == 1
    assert runner_.run_once() == 2 and len(api.completions) == 5
    assert runner_.run_once() == 0
    assert all(c["body"]["declared_parallelism"] == 3 for c in api.claims)


def test_image_cache_evicts_only_idle_images_and_cleans_rejected_exports(tmp_path):
    first = "sha256:" + "1" * 64
    second = "sha256:" + "2" * 64

    def exporter(_reference, digest, target):
        rootfs = target / "rootfs"
        rootfs.mkdir()
        (rootfs / "digest").write_text(digest, encoding="utf-8")

    cache = rn.ImageCache(tmp_path / "images", exporter, max_entries=1, max_bytes=1024 * 1024)
    with cache.acquire(first) as first_rootfs:
        assert first_rootfs.is_dir()
        with pytest.raises(rn.RunnerError, match="capacity is in use"):
            with cache.acquire(second):
                pass
        assert first_rootfs.is_dir()
        assert not (tmp_path / "images" / second.replace(":", "-")).exists()
    with cache.acquire(second) as second_rootfs:
        assert second_rootfs.is_dir()
        assert not (tmp_path / "images" / first.replace(":", "-")).exists()

    oversized = rn.ImageCache(tmp_path / "small", exporter, max_entries=1, max_bytes=1)
    with pytest.raises(rn.RunnerError, match="exceeds runner cache capacity"):
        oversized.rootfs_for(first)
    assert list((tmp_path / "small").iterdir()) == []


def test_source_cache_extracts_once_and_installs_optional_binary_dependencies(tmp_path):
    payload = _source_archive_with_requirements("pydantic-ai==1.0.0\n")
    submission_id = "sub-first"
    fetches = []
    installs = []

    def fetch(run_id, lease_token):
        fetches.append((run_id, lease_token))
        return payload

    def install(requirements, target):
        installs.append(requirements.read_text(encoding="utf-8"))
        (target / "installed.py").write_text("VALUE = 1\n", encoding="utf-8")

    cache = rn.SourceCache(tmp_path / "sources", fetch, dependency_installer=install)
    arguments = ("run-1", "a" * 64, SOURCE_REF, submission_id, len(payload))
    with cache.acquire(*arguments) as (source, dependencies):
        assert (source / "harness.py").is_file()
        assert (dependencies / "installed.py").is_file()
    cache = rn.SourceCache(tmp_path / "sources", fetch, dependency_installer=install)
    with cache.acquire("run-2", "b" * 64, SOURCE_REF, submission_id, len(payload)):
        pass
    second_submission_id = "sub-second"
    with cache.acquire("run-3", "c" * 64, SOURCE_REF, second_submission_id, len(payload)):
        pass
    assert fetches == [("run-1", "a" * 64), ("run-3", "c" * 64)]
    assert installs == ["pydantic-ai==1.0.0\n"] * 2
    assert (tmp_path / "sources" / ("submission-" + submission_id)).is_dir()
    assert (tmp_path / "sources" / ("submission-" + second_submission_id)).is_dir()


def test_source_cache_prepares_different_bundles_in_parallel(tmp_path):
    payloads = {
        "run-1": _source_archive_with_requirements("package-one==1.0.0\n"),
        "run-2": _source_archive_with_requirements("package-two==1.0.0\n"),
    }
    barrier = threading.Barrier(2)

    def fetch(run_id, _lease_token):
        return payloads[run_id]

    def install(_requirements, target):
        barrier.wait(timeout=2)
        (target / "installed.py").write_text("VALUE = 1\n", encoding="utf-8")

    cache = rn.SourceCache(
        tmp_path / "sources", fetch, dependency_installer=install
    )

    def prepare(run_id):
        payload = payloads[run_id]
        with cache.acquire(
            run_id, "a" * 64, SOURCE_REF, "submission-" + run_id, len(payload)
        ) as paths:
            assert (paths[0] / "harness.py").is_file()

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(prepare, run_id) for run_id in sorted(payloads)]
        for future in futures:
            future.result(timeout=3)


def test_binary_requirement_install_rejects_urls_options_and_source_builds(
    tmp_path, monkeypatch
):
    requirements = tmp_path / "requirements.txt"
    target = tmp_path / "deps"
    target.mkdir()
    for unsafe in ("-r other.txt\n", "agent @ https://example.test/a.whl\n", "../agent\n"):
        requirements.write_text(unsafe, encoding="utf-8")
        with pytest.raises(rn.RunnerError, match="package names and version constraints"):
            rn.install_binary_requirements(requirements, target)

    calls = []
    requirements.write_text("pydantic-ai==1.0.0\n", encoding="utf-8")
    monkeypatch.setattr(
        rn.subprocess,
        "run",
        lambda command, **kwargs: calls.append((command, kwargs))
        or type("Result", (), {"returncode": 0})(),
    )
    rn.install_binary_requirements(requirements, target)
    mount_command, mount_kwargs = calls[0]
    command, kwargs = next(call for call in calls if "pip" in call[0])
    assert mount_command[:3] == ["mount", "-t", "tmpfs"]
    assert "size=%d" % rn.MAX_DEPENDENCY_BYTES in mount_command[4]
    assert mount_kwargs["timeout"] == rn.DEPENDENCY_MOUNT_TIMEOUT_SECONDS
    assert "--only-binary=:all:" in command
    assert "--no-deps" not in command and "--no-compile" in command
    assert kwargs["timeout"] == rn.DEPENDENCY_INSTALL_TIMEOUT_SECONDS
    assert Path(kwargs["env"]["HOME"]).parent.name.startswith("lab-arena-deps-")
    assert Path(kwargs["env"]["TMPDIR"]).parent == Path(kwargs["env"]["HOME"]).parent
    assert calls[-1][0][0] == "umount"


def test_submitted_dependency_failure_is_a_model_error_not_an_abandoned_lease(
    tmp_path,
):
    payload = _source_archive_with_requirements("missing-package==1.0.0\n")
    run_lease = lease()
    run_lease.update(source_size_bytes=len(payload))
    api = FakeApi([run_lease])
    api.source_payload = payload

    def reject_dependency(_requirements, _target):
        raise rn.AgentDependencyError("binary dependency installation failed")

    sandbox = BridgingRuntime(output={"companies": [valid_company(1)]})
    (tmp_path / "work").mkdir()
    config = make_config(tmp_path, api, sandbox)
    config.source_cache = rn.SourceCache(
        tmp_path / "failing-sources",
        api.source,
        dependency_installer=reject_dependency,
    )
    runner_ = rn.Runner(config)

    assert runner_.run_once() == 1
    assert runner_.abandoned == 0
    assert sandbox.specs == []
    assert api.completions[0]["body"]["result"]["terminal_status"] == "model_error"
    assert api.completions[0]["body"]["output"] is None


def test_http_source_download_uses_the_existing_lease_header_and_a_byte_cap():
    token = "a" * 64

    def handler(request):
        assert request.headers["x-lab-arena-lease"] == token
        return httpx.Response(200, content=SOURCE_BYTES)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    api = rn.HttpArenaApiClient("http://localhost", client=client)
    assert api.source("run-1", token) == SOURCE_BYTES

    oversized = httpx.Client(
        transport=httpx.MockTransport(
            lambda _request: httpx.Response(
                200,
                headers={
                    "content-length": str(source_bundle.MAX_SOURCE_ARCHIVE_BYTES + 1)
                },
                content=b"x",
            )
        )
    )
    api = rn.HttpArenaApiClient("http://localhost", client=oversized)
    with pytest.raises(rn.RunnerError, match="exceeds the archive limit"):
        api.source("run-1", token)


def test_frame_validation_rejects_identity_fields_and_unknown_operations(tmp_path):
    api = FakeApi([])
    state = rn.RunState(lease=lease(), lease_token="tok")
    server = rn.WorkerSocketServer(tmp_path / "worker.sock", api, state)
    good = shim.build_operation_frame("deepline.execute", {"tool": "exa_search", "payload": {"query": "x"}}, 1000)
    response = json.loads(server.handle_frame(good))
    assert response["status"] == 200 and state.action_sequence == 1
    for hostile in (
        json.dumps({"schema_version": shim.OPERATION_FRAME_SCHEMA_VERSION, "operation_id": "deepline.execute", "parameters": {"tool": "exa_search", "payload": {"query": "x"}}, "timeout_ms": 1000, "lease_token": "steal"}),
        json.dumps({"schema_version": shim.OPERATION_FRAME_SCHEMA_VERSION, "operation_id": "deepline.play", "parameters": {}, "timeout_ms": 1000}),
        json.dumps({"schema_version": shim.OPERATION_FRAME_SCHEMA_VERSION, "operation_id": "deepline.execute", "parameters": {"tool": "exa_search", "payload": {"query": "x"}, "headers": {"x-api-key": "k"}}, "timeout_ms": 1000}),
        b"not json",
    ):
        error = json.loads(server.handle_frame(hostile.encode() if isinstance(hostile, str) else hostile))
        assert set(error) == {"error"}
    assert state.action_sequence == 1 and len(api.provider_frames) == 1


def test_parallelism_env_and_http_boundary():
    assert rn.max_parallel_runs_from_environment({}) == 8 and rn.max_parallel_runs_from_environment({rn.MAX_PARALLEL_ENV: "4"}) == 4
    with pytest.raises(rn.RunnerError):
        rn.max_parallel_runs_from_environment({rn.MAX_PARALLEL_ENV: "0"})
    with pytest.raises(rn.RunnerError):
        rn.max_parallel_runs_from_environment({rn.MAX_PARALLEL_ENV: "9"})
    with pytest.raises(rn.RunnerError):
        rn.HttpArenaApiClient("http://arena.example.com")
    with pytest.raises(rn.RunnerError):
        rn.HttpArenaApiClient("http://localhost.evil.example")
    document = output_document_from_bytes(json.dumps([valid_company(1)]).encode())
    assert document["schema_version"] == contracts.OUTPUT_DOCUMENT_SCHEMA_VERSION and len(document["companies"]) == 1


def test_provider_http_timeout_covers_the_requested_provider_window():
    class Response:
        status_code = 200

        @staticmethod
        def json():
            return {"status": "ok"}

    class RecordingClient:
        def __init__(self):
            self.timeouts = []

        def post(self, _url, **kwargs):
            self.timeouts.append(kwargs["timeout"])
            return Response()

        def close(self):
            return None

    client = RecordingClient()
    api = rn.HttpArenaApiClient("http://localhost", client=client)
    api.provider(
        "run-1",
        "a" * 64,
        {
            "operation_id": "openrouter.chat",
            "parameters": {},
            "timeout_ms": 120_000,
            "action_sequence": 0,
        },
    )
    assert client.timeouts[-1].read == rn.MAX_PROVIDER_API_TIMEOUT_SECONDS
    api.claim({"request_id": "claim-1"})
    assert client.timeouts[-1].read == rn.API_TIMEOUT_SECONDS


def test_worker_socket_refuses_connections_above_its_fixed_thread_bound(tmp_path):
    socket_dir = Path(tempfile.mkdtemp(prefix="la", dir="/tmp"))
    server = rn.WorkerSocketServer(
        socket_dir / "worker.sock",
        FakeApi([]),
        rn.RunState(lease=lease(), lease_token="tok"),
        max_connections=1,
        read_timeout_seconds=1,
    )
    server.start()
    first = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    second = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        first.connect(str(socket_dir / "worker.sock"))
        first.sendall(b"\x00")  # one incomplete frame holds the only handler
        time.sleep(0.05)
        second.settimeout(1)
        second.connect(str(socket_dir / "worker.sock"))
        assert second.recv(1) == b""
    finally:
        first.close()
        second.close()
        server.stop()
        shutil.rmtree(socket_dir, ignore_errors=True)


def test_invalid_submission_id_and_runtime_image_mismatches_are_refused(tmp_path):
    wrong_source = lease("r1")
    wrong_source["submission_id"] = "../unsafe"
    mismatched = lease("r2")
    mismatched["image_reference"] = ARENA_REPOSITORY + "@sha256:" + "b" * 64
    wrong_judge = scoring_lease("r3")
    wrong_judge["image_reference"] = "arena.example/lab-arena/other@sha256:" + "b" * 64
    api = FakeApi([wrong_source, mismatched, wrong_judge])
    (tmp_path / "work").mkdir()
    config = make_config(tmp_path, api, BridgingRuntime(output={"companies": [valid_company(1)]}), parallel=3)
    runner_ = rn.Runner(config)
    runner_.run_once()
    assert runner_.abandoned == 3 and len(api.completions) == 0
    assert all(c["error"] == "RunnerError" for c in runner_.completed if "error" in c)


class FollowApi(FakeApi):
    """A fake Arena whose current round can change between polls."""

    def __init__(self, leases, *, rounds):
        super().__init__(leases)
        self.rounds = rounds  # round id -> configuration
        self.current_round = None
        self.current_status = "stage1"
        self.polls = 0

    def current(self):
        self.polls += 1
        return {"round": {"round_id": self.current_round, "status": self.current_status} if self.current_round else None}

    def round(self, round_id):
        return {"round_id": round_id, "configuration": self.rounds[round_id]}


def test_a_runner_without_a_pinned_round_follows_each_current_round_without_code_identity_gates(tmp_path):
    configuration = {"schedule": {}}
    api = FollowApi([lease("r1")], rounds={ROUND: configuration, "arena-2026-09-03": configuration, "arena-2026-09-04": configuration})
    (tmp_path / "work").mkdir()
    config = make_config(tmp_path, api, BridgingRuntime(output={"companies": [valid_company(1)]}))
    config.round_id = None
    runner_ = rn.Runner(config)
    # No round yet: nothing to claim, nothing signed.
    assert runner_.run_once() == 0 and runner_.round_id is None and api.claims == []
    api.current_round = ROUND
    assert runner_.run_once() == 1 and runner_.round_id == ROUND
    assert api.claims[-1]["round_id"] == ROUND and api.completions[0]["round_id"] == ROUND and api.completions[0]["body"]["run_id"] == "r1"
    api.current_round = "arena-2026-09-03"
    assert runner_.run_once() == 0 and runner_.round_id == "arena-2026-09-03"
    api.current_round = "arena-2026-09-04"
    api.leases.append(dict(lease("r2"), round_id="arena-2026-09-04", assignment_id="arena-2026-09-04:s1:1:0"))
    assert runner_.run_once() == 1 and runner_.round_id == "arena-2026-09-04"
    assert api.claims[-1]["round_id"] == "arena-2026-09-04" and api.completions[-1]["round_id"] == "arena-2026-09-04"


@pytest.mark.parametrize("status", rn.WORKING_STATUSES)
def test_an_unpinned_runner_follows_every_execution_and_scoring_window(tmp_path, status):
    api = FollowApi([], rounds={ROUND: {"schedule": {}}})
    api.current_round = ROUND
    api.current_status = status
    (tmp_path / "work").mkdir()
    config = make_config(tmp_path, api, BridgingRuntime(output={"companies": [valid_company(1)]}))
    config.round_id = None

    runner_ = rn.Runner(config)
    assert runner_.run_once() == 0
    assert runner_.round_ids == [ROUND]


def test_a_model_speaks_plain_http_over_the_worker_socket(tmp_path):
    """The runtime-neutral contract: the provider's own request, sent to the socket without a credential."""

    import http.client
    import socket as socket_module

    api = FakeApi([])
    state = rn.RunState(lease=lease("r1"), lease_token="tok-r1")
    socket_dir = Path(tempfile.mkdtemp(prefix="la", dir="/tmp"))
    server = rn.WorkerSocketServer(socket_dir / runtime.SANDBOX_SOCKET_NAME, api, state)
    server.start()

    def request(method, host, path, body=None, headers=None):
        connection = http.client.HTTPConnection(host)
        connection.sock = socket_module.socket(socket_module.AF_UNIX, socket_module.SOCK_STREAM)
        connection.sock.connect(str(socket_dir / runtime.SANDBOX_SOCKET_NAME))
        try:
            connection.request(method, path, body=body, headers=headers or {})
            response = connection.getresponse()
            return response.status, dict(response.getheaders()), response.read()
        finally:
            connection.close()

    try:
        status, headers, body = request("GET", "api.scrapingdog.com", "/scrape?url=https%3A%2F%2Fexample.com%2F")
        assert status == 200 and json.loads(body)["results"] and headers["Connection"] == "close"
        assert api.provider_frames[-1]["operation_id"] == "scrapingdog.scrape" and state.calls
        # A credential header is refused, an unknown host has no operation, and the frame path still works.
        status, _headers, body = request("GET", "api.scrapingdog.com", "/scrape?url=https%3A%2F%2Fexample.com%2F", headers={"Authorization": "Bearer leaked"})
        assert status == 400 and json.loads(body)["error"]["code"] == "forbidden_header"
        status, _headers, body = request("GET", "evil.example", "/anything")
        assert status == 400 and json.loads(body)["error"]["code"] == "no_matching_operation"
        os.environ[shim.WORKER_SOCKET_ENV] = str(socket_dir / runtime.SANDBOX_SOCKET_NAME)
        try:
            frame_status, _frame_headers, frame_body = shim.dispatch("deepline.execute", {"tool": "exa_search", "payload": {"query": "fintech"}}, 5000)
        finally:
            os.environ.pop(shim.WORKER_SOCKET_ENV, None)
        assert frame_status == 200 and json.loads(frame_body)["results"]
        assert len(state.calls) == 2  # only the two matched requests reached the Arena
    finally:
        server.stop()
        shutil.rmtree(socket_dir, ignore_errors=True)


def test_agent_entrypoint_loads_the_trusted_shim_for_a_standard_http_client(tmp_path):
    """The OCI Python path installs the broker shim before a miner harness imports urllib."""

    api = FakeApi([])
    state = rn.RunState(lease=lease("r1"), lease_token="tok-r1")
    socket_dir = Path(tempfile.mkdtemp(prefix="la", dir="/tmp"))
    server = rn.WorkerSocketServer(
        socket_dir / runtime.SANDBOX_SOCKET_NAME,
        api,
        state,
    )
    source = tmp_path / "source"
    source.mkdir()
    (source / "harness.py").write_text(
        "import json\n"
        "import urllib.request\n"
        "def run_icp(icp):\n"
        "    url = 'https://api.scrapingdog.com/scrape?url=https%3A%2F%2Fexample.com%2F'\n"
        "    with urllib.request.urlopen(url, timeout=5) as response:\n"
        "        result = json.load(response)\n"
        "    return [{'company_name': result['results'][0]['url']}]\n",
        encoding="utf-8",
    )
    input_path = tmp_path / "icp.json"
    input_path.write_text(
        json.dumps({"icp": {"icp_id": "arena:test", "prompt": "find leads"}}),
        encoding="utf-8",
    )
    output_path = tmp_path / "companies.json"
    trusted_model = tmp_path / "model"
    trusted_model.mkdir()
    (trusted_model / "sitecustomize.py").write_text(
        shim.SITECUSTOMIZE_SOURCE,
        encoding="utf-8",
    )
    (trusted_model / "lab_arena").symlink_to(
        Path(rn.__file__).resolve().parent,
        target_is_directory=True,
    )
    script = (
        "import sys\n"
        "from pathlib import Path\n"
        "from lab_arena.agent_entrypoint import run\n"
        "run(source_dir=Path(sys.argv[1]), input_path=Path(sys.argv[2]), "
        "output_path=Path(sys.argv[3]))\n"
    )
    environment = {
        "HOME": str(tmp_path),
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PATH": runtime.PROCESS_ENV["PATH"],
        "PYTHONPATH": str(trusted_model),
        shim.WORKER_SOCKET_ENV: str(socket_dir / runtime.SANDBOX_SOCKET_NAME),
    }
    server.start()
    try:
        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                script,
                str(source),
                str(input_path),
                str(output_path),
            ],
            cwd=source,
            env=environment,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            timeout=15,
            check=False,
        )
        assert completed.returncode == 0, completed.stderr.decode("utf-8", "replace")
        assert json.loads(output_path.read_text(encoding="utf-8")) == {
            "companies": [{"company_name": "https://co1.example.com"}]
        }
        assert api.provider_frames[-1]["operation_id"] == "scrapingdog.scrape"
    finally:
        server.stop()
        shutil.rmtree(socket_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Scoring assignments: the validator runs the pinned judge image on one output
# ---------------------------------------------------------------------------

SCORER_IMAGE = "sha256:" + "5" * 64
SCORER_IMAGE_REFERENCE = "arena.example/lab-arena/judge@" + SCORER_IMAGE
def scoring_lease(run_id="r9", position=3):
    from lab_arena import scoring

    base = lease(run_id, position)
    base.update({
        "assignment_id": "%s:s1:1:%d:score" % (ROUND, position), "kind": "score",
        "scored_run_id": "r1", "image_digest": SCORER_IMAGE, "image_reference": SCORER_IMAGE_REFERENCE,
        "scored_output": {"companies": [valid_company(1), valid_company(2)]}, "scorer_policy": scoring.build_scorer_policy(),
    })
    return base


def test_scoring_lease_runs_the_judge_image_in_trusted_mode(tmp_path):
    from lab_arena import scoring

    breakdowns = [{"final_score": 71.0, "failure_reason": ""}, {"final_score": 44.5, "failure_reason": ""}]
    output = scoring.build_scoring_output("r1", breakdowns)
    api = FakeApi([scoring_lease()])
    sandbox = BridgingRuntime(output=output, calls=1)
    (tmp_path / "work").mkdir()
    runner_ = rn.Runner(make_config(tmp_path, api, sandbox))
    assert runner_.run_once() == 1 and runner_.abandoned == 0
    envelope = api.completions[0]
    validated = contracts.validate_run_result(envelope["body"]["result"])
    assert validated["terminal_status"] == "accepted"
    assert envelope["body"]["output"] == output
    spec = sandbox.specs[0]
    assert spec.entry_command == runtime.SCORER_ENTRY_COMMAND and spec.working_dir == runtime.SCORER_WORKING_DIR
    assert spec.extra_environment[shim.TRUSTED_SCORER_ENV] == "1"
    assert spec.rootfs_path == tmp_path / "images" / SCORER_IMAGE.replace(":", "-") / "rootfs"
    # The judge sandbox read a scoring input, not an ICP input.
    assert sandbox.specs[0].input_dir.name == "input"


@pytest.mark.parametrize("output, timed_out, expected", [
    ({"schema_version": "leadpoet.lab_arena.scoring_output.v1", "scored_run_id": "r1", "failure": "judge_error"}, False, "judge_error"),
    (None, True, "judge_timeout"),
    (b"not json", False, "judge_error"),
    ({"companies": [valid_company(1)]}, False, "judge_error"),
])
def test_judge_failures_map_to_judge_causes_with_no_output(tmp_path, output, timed_out, expected):
    api = FakeApi([scoring_lease()])
    sandbox = BridgingRuntime(output=output, timed_out=timed_out, calls=0)
    (tmp_path / "work").mkdir()
    runner_ = rn.Runner(make_config(tmp_path, api, sandbox))
    assert runner_.run_once() == 1 and runner_.abandoned == 0
    validated = contracts.validate_run_result(api.completions[0]["body"]["result"])
    assert validated["terminal_status"] == expected
    assert api.completions[0]["body"].get("output") in (None, {})


class RefusingApi(FakeApi):
    """The broker refuses a scoring call under its external quota."""

    def __init__(self, leases, *, code="budget_refused"):
        super().__init__(leases)
        self.code = code

    def provider(self, run_id, lease_token, frame):
        self.provider_frames.append(dict(frame))
        call = {"call_identity": contracts.document_hash(["call", frame["action_sequence"]]), "operation_id": frame["operation_id"], "reserved_microusd": 0, "actual_microusd": 0, "outcome": "refused", "error_code": self.code}
        body = b'{"error":{"code":"%s"}}' % self.code.encode()
        return {"status": 402, "headers": {"content-type": "application/json", "content-length": str(len(body))}, "body_b64": base64.b64encode(body).decode(), "call": call}


def test_valid_fallback_output_survives_a_refused_provider_call(tmp_path):
    """A miner can handle an unavailable provider and return a valid result."""

    class FallbackRuntime(BridgingRuntime):
        def run_icp(self, spec, **_):
            self.specs.append(spec)
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
            spec.output_path.write_bytes(json.dumps(self.output).encode())
            return runtime.fake_result(exit_code=0, output_bytes=runtime.read_output(spec))

    api = RefusingApi([lease()])
    sandbox = FallbackRuntime(output={"companies": [valid_company(1)]})
    (tmp_path / "work").mkdir()
    runner_ = rn.Runner(make_config(tmp_path, api, sandbox))

    assert runner_.run_once() == 1
    completion = api.completions[0]["body"]
    assert contracts.validate_run_result(completion["result"])["terminal_status"] == "accepted"
    assert [item["company_name"] for item in completion["output"]["companies"]] == ["Co 1"]


class RefusedJudgeRuntime(BridgingRuntime):
    """The real judge folds a refused provider call into its own failure document."""

    def run_icp(self, spec, **_):
        self.specs.append(spec)
        os.environ[shim.WORKER_SOCKET_ENV] = str(spec.socket_path)
        try:
            status, _headers, _body = shim.dispatch("openrouter.chat", {"model": "openai/gpt-4o-mini", "messages": [{"role": "user", "content": "judge"}]}, 5000)
        finally:
            os.environ.pop(shim.WORKER_SOCKET_ENV, None)
        assert status == 402
        spec.output_path.write_bytes(json.dumps(self.output).encode())
        return runtime.fake_result(exit_code=0, output_bytes=runtime.read_output(spec))


@pytest.mark.parametrize("code", ["budget_refused", "budget_exhausted", "call_refused", "provider_unavailable"])
def test_a_refused_scoring_call_is_an_infrastructure_judge_error(tmp_path, code):

    from lab_arena import scoring

    failure = scoring.build_scoring_failure("r1", "judge_error", detail="provider error")
    api = RefusingApi([scoring_lease()], code=code)
    (tmp_path / "work").mkdir()
    runner_ = rn.Runner(make_config(tmp_path, api, RefusedJudgeRuntime(output=failure)))
    assert runner_.run_once() == 1 and runner_.abandoned == 0
    validated = contracts.validate_run_result(api.completions[0]["body"]["result"])
    assert validated["terminal_status"] == "judge_error"
    assert api.provider_frames[0]["operation_id"] == "openrouter.chat"


@pytest.mark.parametrize("scoring_run", [False, True])
def test_miner_key_failure_does_not_become_an_infrastructure_failure(tmp_path, scoring_run):
    from lab_arena import scoring

    class MinerKeyApi(RefusingApi):
        def provider(self, run_id, lease_token, frame):
            document = super().provider(run_id, lease_token, frame)
            document["call"].update(funding_source="miner_key", provider_status=401)
            return document

    failure = scoring.build_scoring_failure("r1", "judge_error", detail="provider error")
    api = MinerKeyApi([scoring_lease() if scoring_run else lease()], code="miner_credentials_unavailable")
    (tmp_path / "work").mkdir()
    runner_ = rn.Runner(make_config(tmp_path, api, RefusedJudgeRuntime(output=failure)))
    assert runner_.run_once() == 1 and runner_.abandoned == 0
    result = contracts.validate_run_result(api.completions[0]["body"]["result"])
    assert result["terminal_status"] == "credential_error"
    assert api.completions[0]["body"].get("output") in (None, {})


class FlakyCompletionApi(FakeApi):
    """The completion call fails a given number of times before the Arena accepts it."""

    def __init__(self, leases, *, failures):
        super().__init__(leases)
        self.failures = failures
        self.attempts = 0

    def complete(self, envelope):
        self.attempts += 1
        if self.attempts <= self.failures:
            raise rn.RunnerError("Arena API failed: HTTP 503")
        return super().complete(envelope)


class CompletionDocumentApi(FakeApi):
    """Return fixed completion documents and retain only envelope identities."""

    def __init__(self, leases, responses):
        super().__init__(leases)
        self.responses = list(responses)
        self.completion_attempts = []

    def complete(self, envelope):
        self.completion_attempts.append(envelope)
        if not self.responses:
            raise AssertionError("unexpected completion attempt")
        return self.responses.pop(0)


@pytest.mark.parametrize("failures, expect_abandoned", [(1, 0), (2, 0), (3, 1)])
def test_a_transient_completion_failure_is_retried_before_the_run_is_abandoned(tmp_path, failures, expect_abandoned):
    """Two retries cover a lost response or a transient Arena failure; a third failure fails closed."""

    api = FlakyCompletionApi([lease()], failures=failures)
    (tmp_path / "work").mkdir()
    runner_ = rn.Runner(make_config(tmp_path, api, BridgingRuntime(output={"companies": [valid_company(1)]}, calls=1)))
    assert runner_.run_once() == 1
    assert runner_.abandoned == expect_abandoned
    assert api.attempts == min(failures + 1, 3)
    assert len(api.completions) == (0 if expect_abandoned else 1)
    if expect_abandoned:
        assert runner_.completed[0]["error"] == "RunnerError"


def test_accounting_open_retries_the_original_completion_until_accepted(
    tmp_path, capsys
):
    api = CompletionDocumentApi(
        [lease()],
        [
            {"status": "accounting_open", "open_calls": 1},
            {"status": "accepted"},
        ],
    )
    (tmp_path / "work").mkdir()
    config = make_config(
        tmp_path,
        api,
        BridgingRuntime(output={"companies": [valid_company(1)]}, calls=0),
    )
    assert sum(config.accounting_open_retry_seconds) >= rn.MAX_PROVIDER_API_TIMEOUT_SECONDS
    assert (
        sum(config.accounting_open_retry_seconds)
        < contracts.REQUEST_TIMESTAMP_WINDOW_SECONDS
    )
    assert sum(config.accounting_open_retry_seconds) < contracts.LEASE_TTL_SECONDS
    config.accounting_open_retry_seconds = (0.0,)
    runner_ = rn.Runner(config)

    assert runner_.run_once() == 1
    assert runner_.abandoned == 0
    assert runner_.completed == [
        {"run_id": "r1", "result": {"status": "accepted"}}
    ]
    assert len(api.completion_attempts) == 2
    assert api.completion_attempts[0] is api.completion_attempts[1]
    captured = capsys.readouterr()
    assert captured.out.splitlines() == [
        "Lab Arena completion status: accounting_open",
        "Lab Arena completion status: accepted",
    ]
    assert captured.err == ""


def test_accounting_open_retry_exhaustion_abandons_without_logging_the_envelope(
    tmp_path, capsys
):
    api = CompletionDocumentApi(
        [lease()],
        [{"status": "accounting_open", "open_calls": 1}] * 3,
    )
    (tmp_path / "work").mkdir()
    config = make_config(
        tmp_path,
        api,
        BridgingRuntime(output={"companies": [valid_company(1)]}, calls=0),
    )
    config.accounting_open_retry_seconds = (0.0, 0.0)
    runner_ = rn.Runner(config)

    assert runner_.run_once() == 1
    assert runner_.abandoned == 1
    assert len(api.completion_attempts) == 3
    assert all(
        envelope is api.completion_attempts[0]
        for envelope in api.completion_attempts
    )
    captured = capsys.readouterr()
    assert captured.out.splitlines() == [
        "Lab Arena completion status: accounting_open",
    ] * 3
    assert captured.err == "Lab Arena run abandoned: RunnerError\n"
    assert "tok-r1" not in captured.out + captured.err
    assert "signature" not in captured.out + captured.err


def test_completion_rejection_is_logged_but_not_retried(tmp_path, capsys):
    api = CompletionDocumentApi(
        [lease()],
        [{"status": "rejected", "detail": "must-not-log"}],
    )
    (tmp_path / "work").mkdir()
    config = make_config(
        tmp_path,
        api,
        BridgingRuntime(output={"companies": [valid_company(1)]}, calls=0),
    )
    config.accounting_open_retry_seconds = (0.0, 0.0)
    runner_ = rn.Runner(config)

    assert runner_.run_once() == 1
    assert runner_.abandoned == 0
    assert len(api.completion_attempts) == 1
    captured = capsys.readouterr()
    assert captured.out == "Lab Arena completion status: rejected\n"
    assert captured.err == ""
    assert "must-not-log" not in captured.out



class RefusingEveryFrameApi(RefusingApi):
    """Every frame is refused by the Arena: the quota is gone."""

    def __init__(self, leases):
        super().__init__(leases, code="budget_exhausted")


class LoopingRuntime(BridgingRuntime):
    """A model that keeps calling after its quota is exhausted."""

    def __init__(self, frames):
        super().__init__(output={"companies": [valid_company(1)]}, calls=0)
        self.frames = frames
        self.statuses = []

    def run_icp(self, spec, **_):
        self.specs.append(spec)
        os.environ[shim.WORKER_SOCKET_ENV] = str(spec.socket_path)
        try:
            for _ in range(self.frames):
                try:
                    status, _headers, _body = shim.dispatch("deepline.execute", {"tool": "exa_search", "payload": {"query": "fintech"}}, 5000)
                    self.statuses.append(status)
                except shim.ShimError as exc:
                    self.statuses.append(str(exc))
        finally:
            os.environ.pop(shim.WORKER_SOCKET_ENV, None)
        spec.output_path.write_bytes(json.dumps(self.output).encode())
        return runtime.fake_result(exit_code=0, output_bytes=runtime.read_output(spec))


def test_after_repeated_refusals_the_worker_answers_frames_locally(tmp_path):
    """A model looping on a refused quota costs the Arena at most MAX_REFUSED_FRAMES round trips."""

    api = RefusingEveryFrameApi([lease()])
    runtime_ = LoopingRuntime(frames=rn.MAX_REFUSED_FRAMES + 40)
    (tmp_path / "work").mkdir()
    runner_ = rn.Runner(make_config(tmp_path, api, runtime_))
    assert runner_.run_once() == 1 and runner_.abandoned == 0
    assert len(api.provider_frames) == rn.MAX_REFUSED_FRAMES
    assert len(runtime_.statuses) == rn.MAX_REFUSED_FRAMES + 40
    assert all(status == 402 for status in runtime_.statuses[:rn.MAX_REFUSED_FRAMES])
    assert all("budget_exhausted" in str(status) for status in runtime_.statuses[rn.MAX_REFUSED_FRAMES:])
    assert api.completions[0]["body"]["result"]["resource_summary"]["provider_call_count"] == rn.MAX_REFUSED_FRAMES
