"""Arena runner (labarena.md sections 9, 10, 14.3, 15.1; image-by-digest plan).

A runner follows the Arena's current round, claims one pending assignment per
free local slot, materializes the pinned image's root filesystem from the
Arena registry, executes the image's own entry command in a fresh gVisor
sandbox for that single ICP, bridges the sandbox's provider requests (plain
HTTP over the worker socket, or the judge shim's operation frames) to the
service's provider endpoint, appends the private event log, and submits one
runner-signed receipt. It never reports a score, never holds a database
credential, and never chooses a miner or ICP.
"""

from __future__ import annotations

import base64
import http.server
import json
import os
import shutil
import socket
import socketserver
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Protocol, Sequence, Tuple

import httpx

from lab_arena import contracts, images, operations, runtime, scoring, shim
from lab_arena.contracts import ArenaContractError
from lab_arena.output import OutputInvalid, output_document_from_bytes, output_document_hash

WORKER_RELEASE_SCHEMA_VERSION = "leadpoet.lab_arena.worker_release.v1"
DEFAULT_MAX_PARALLEL_RUNS = 1
MAX_PARALLEL_ENV = "LAB_ARENA_MAX_PARALLEL_RUNS"
LOG_EVENT_CHUNK_BYTES = 8192
DEFAULT_SOCKET_ROOT = "/tmp"
MAX_REFUSED_FRAMES = 25  # after this many refused calls the worker answers a run's frames locally
# A request on the worker socket is either a length-prefixed operation frame
# (first byte 0x00: the judge shim) or an HTTP request (an ASCII method).
HTTP_FIRST_BYTES = b"GPHDO"
HTTP_ERROR_STATUS = {"budget_exhausted": 402, "worker_unavailable": 503, "request_too_large": 413}
IMAGE_DIGEST_RE = __import__("re").compile(r"^(?:[a-z0-9][a-z0-9._/-]{0,200}@)?sha256:[0-9a-f]{64}$")
MAX_SOCKET_PATH_BYTES = 100
EVENT_APPEND_RETRIES = 3
API_TIMEOUT_SECONDS = 30.0


class RunnerError(RuntimeError):
    """A runner-side failure; the attempt fails closed."""


class SignatureFn(Protocol):
    def __call__(self, message: str) -> str: ...


# ---------------------------------------------------------------------------
# Service client
# ---------------------------------------------------------------------------


class ArenaApiClient(Protocol):
    def claim(self, envelope: Mapping[str, Any]) -> Dict[str, Any]: ...

    def provider(self, run_id: str, lease_token: str, frame: Mapping[str, Any]) -> Dict[str, Any]: ...

    def append_events(self, run_id: str, lease_token: str, events: Sequence[Mapping[str, Any]]) -> Dict[str, Any]: ...

    def complete(self, envelope: Mapping[str, Any]) -> Dict[str, Any]: ...

    def current(self) -> Dict[str, Any]: ...

    def round(self, round_id: str) -> Dict[str, Any]: ...


class HttpArenaApiClient:
    """HTTPS client for ``/arena/v1`` runner endpoints (section 14.3)."""

    def __init__(self, base_url: str, *, client: Optional[httpx.Client] = None) -> None:
        if not base_url.startswith("https://") and not base_url.startswith("http://127.0.0.1") and not base_url.startswith("http://localhost"):
            raise RunnerError("Arena API base URL must be https (or loopback for tests)")
        self._base_url = base_url.rstrip("/")
        self._client = client or httpx.Client(http1=True, http2=False, follow_redirects=False, timeout=httpx.Timeout(API_TIMEOUT_SECONDS))

    def _post(self, path: str, document: Mapping[str, Any], *, headers: Optional[Mapping[str, str]] = None) -> Dict[str, Any]:
        try:
            response = self._client.post(self._base_url + path, content=contracts.canonical_json(document).encode("utf-8"), headers={"content-type": "application/json", **(headers or {})})
        except httpx.HTTPError as exc:
            raise RunnerError("Arena API transport failure: %s" % type(exc).__name__) from exc
        if response.status_code >= 500:
            raise RunnerError("Arena API failed: HTTP %d" % response.status_code)
        try:
            payload = response.json()
        except ValueError as exc:
            raise RunnerError("Arena API returned non-JSON") from exc
        if not isinstance(payload, dict):
            raise RunnerError("Arena API returned a non-object")
        if response.status_code >= 400 and "status" not in payload:
            payload = {"status": "rejected", "http_status": response.status_code, "detail": payload.get("detail")}
        return payload

    def claim(self, envelope: Mapping[str, Any]) -> Dict[str, Any]:
        return self._post("/arena/v1/runs/claim", envelope)

    def provider(self, run_id: str, lease_token: str, frame: Mapping[str, Any]) -> Dict[str, Any]:
        return self._post("/arena/v1/runs/%s/provider" % run_id, frame, headers={"x-lab-arena-lease": lease_token})

    def append_events(self, run_id: str, lease_token: str, events: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
        return self._post("/arena/v1/runs/%s/events" % run_id, {"events": list(events)}, headers={"x-lab-arena-lease": lease_token})

    def complete(self, envelope: Mapping[str, Any]) -> Dict[str, Any]:
        return self._post("/arena/v1/runs/%s/complete" % envelope["body"]["receipt"]["run_id"], envelope)

    def round(self, round_id: str) -> Dict[str, Any]:
        return self._get("/arena/v1/rounds/%s" % round_id, "round %s" % round_id)

    def current(self) -> Dict[str, Any]:
        return self._get("/arena/v1/current", "current round")

    def _get(self, path: str, what: str) -> Dict[str, Any]:
        try:
            response = self._client.get(self._base_url + path)
        except httpx.HTTPError as exc:
            raise RunnerError("Arena API transport failure: %s" % type(exc).__name__) from exc
        if response.status_code != 200:
            raise RunnerError("%s is unavailable: HTTP %d" % (what, response.status_code))
        payload = response.json()
        if not isinstance(payload, dict):
            raise RunnerError("Arena API returned a non-object")
        return payload

    def close(self) -> None:
        self._client.close()


def verify_release_against_round(configuration: Mapping[str, Any], *, worker_release_hash: str, runtime_lock_hash: str, operation_table_hash: str = operations.OPERATION_TABLE_HASH) -> Dict[str, str]:
    """Section 16: a runner fails startup when its worker, runtime, or runsc identity differs from the round."""

    release = configuration.get("release") if isinstance(configuration, Mapping) else None
    if not isinstance(release, Mapping):
        raise RunnerError("round configuration carries no release identity")
    expected = {
        "worker_release_hash": (release.get("worker_release_hash"), worker_release_hash),
        "runsc_lock_hash": (release.get("runsc_lock_hash"), runtime_lock_hash),
        "shim_hash": (release.get("shim_hash"), shim.shim_source_hash()),
        "operation_table_hash": (configuration.get("operation_table_hash"), operation_table_hash),
    }
    for name, (pinned, ours) in expected.items():
        if not pinned or pinned != ours:
            raise RunnerError("runner %s differs from the signed round configuration" % name)
    return {name: ours for name, (_pinned, ours) in expected.items()}


# ---------------------------------------------------------------------------
# Runner identity and image cache
# ---------------------------------------------------------------------------


def worker_release_identity(*, repository_commit: str, runtime_lock_hash: str) -> Dict[str, Any]:
    """The worker release the runner reports and the round configuration pins."""

    return contracts.hashed_document(
        {
            "schema_version": WORKER_RELEASE_SCHEMA_VERSION,
            "repository_commit": repository_commit,
            "runtime_lock_hash": runtime_lock_hash,
            "shim_hash": shim.shim_source_hash(),
            "operation_table_hash": operations.OPERATION_TABLE_HASH,
        },
        "worker_release_hash",
    )


@dataclass
class RunnerIdentity:
    hotkey: str
    sign: SignatureFn
    coldkey_owned_hotkeys: Sequence[str] = ()

    def __post_init__(self) -> None:
        contracts.require_hotkey(self.hotkey)


class ImageExporter(Protocol):
    """Populate ``target_dir/rootfs`` with the image named by reference and pinned by digest."""

    def __call__(self, image_reference: str, image_digest: str, target_dir: Path) -> None: ...


class ImageCache:
    """Pinned images by digest: each digest is materialized at most once per runner."""

    def __init__(self, root: Path, exporter: ImageExporter) -> None:
        self._root = Path(root)
        self._exporter = exporter
        self._lock = threading.Lock()
        self._ready: Dict[str, Path] = {}

    def rootfs_for(self, image_digest: str, image_reference: str = "") -> Path:
        if not isinstance(image_digest, str) or not IMAGE_DIGEST_RE.match(image_digest):
            raise RunnerError("image digest is invalid")
        with self._lock:
            path = self._ready.get(image_digest)
            if path is not None:
                return path
            # Cache directories are keyed by the content digest alone.
            target = self._root / ("sha256-" + image_digest.rsplit("sha256:", 1)[1])
            if not (target / ".exported").exists():
                if target.exists():
                    shutil.rmtree(target)
                target.mkdir(parents=True)
                self._exporter(image_reference, image_digest, target)
                if not (target / "rootfs").is_dir():
                    shutil.rmtree(target, ignore_errors=True)
                    raise RunnerError("image exporter produced no root filesystem")
                (target / ".exported").write_text(image_digest, encoding="utf-8")
            rootfs = target / "rootfs"
            self._ready[image_digest] = rootfs
            return rootfs


def registry_image_exporter(client: images.RegistryClient, *, rules: Optional[images.ImageRules] = None) -> ImageExporter:
    """Materialize a pinned image from the Arena registry with the hardened extractor (no Docker daemon)."""

    image_rules = rules or images.ImageRules()

    def export(image_reference: str, image_digest: str, target_dir: Path) -> None:
        try:
            reference = images.parse_reference(image_reference)
            if reference.digest != image_digest:
                raise RunnerError("lease image reference does not name the lease digest")
            images.materialize_rootfs(client, reference, target_dir, rules=image_rules)
        except images.ImageError as exc:
            raise RunnerError("image %s could not be materialized: %s" % (image_digest[:19], exc.rule_id)) from exc

    return export


# ---------------------------------------------------------------------------
# Per-run state: events, calls, and the worker socket
# ---------------------------------------------------------------------------


@dataclass
class RunState:
    lease: Dict[str, Any]
    lease_token: str
    event_cursor: int
    event_head_hash: str
    event_hashes: List[str] = field(default_factory=list)
    calls: List[Dict[str, Any]] = field(default_factory=list)
    action_sequence: int = 0
    refusals: int = 0  # refused calls answered by the Arena for this run
    failed: Optional[str] = None
    lock: threading.Lock = field(default_factory=threading.Lock)


def _timestamp(clock: Callable[[], datetime]) -> str:
    return clock().astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class EventWriter:
    """Appends worker events in order; a failed append fails the run closed."""

    def __init__(self, api: ArenaApiClient, state: RunState, clock: Callable[[], datetime]) -> None:
        self._api = api
        self._state = state
        self._clock = clock

    def append(self, event_type: str, payload: Mapping[str, Any]) -> None:
        state = self._state
        with state.lock:
            if state.failed:
                raise RunnerError("run already failed: %s" % state.failed)
            event = contracts.build_private_event(event_type=event_type, sequence=state.event_cursor, prev_hash=state.event_head_hash, timestamp=_timestamp(self._clock), payload=payload)
            last_error: Optional[BaseException] = None
            for _attempt in range(EVENT_APPEND_RETRIES):
                try:
                    result = self._api.append_events(state.lease["run_id"], state.lease_token, [event])
                except RunnerError as exc:
                    last_error = exc
                    continue
                if result.get("status") in ("appended", "existing"):
                    state.event_cursor = int(result["event_cursor"])
                    state.event_head_hash = str(result["event_head_hash"])
                    state.event_hashes.append(event["event_hash"])
                    return
                state.failed = "event_append_%s" % result.get("status")
                raise RunnerError("event append rejected: %s" % result.get("status"))
            state.failed = "event_append_unavailable"
            raise RunnerError("event append failed after retries: %s" % (last_error,))

    def sync_from_call(self, call: Mapping[str, Any]) -> None:
        """Adopt the cursor the broker advanced with its provider_call event."""

        state = self._state
        with state.lock:
            cursor = call.get("event_cursor")
            head = call.get("event_head_hash")
            if cursor is None or head is None:
                return
            if int(cursor) > state.event_cursor:
                # Exactly one provider_call event was appended by the broker.
                if int(cursor) != state.event_cursor + 1:
                    state.failed = "event_cursor_gap"
                    raise RunnerError("broker advanced the event cursor by more than one")
                state.event_cursor = int(cursor)
                state.event_head_hash = str(head)
                state.event_hashes.append(str(head))


class WorkerSocketServer:
    """Unix-socket bridge: operation frames in, provider responses out.

    Frames carry only the operation id, validated parameters, and a timeout;
    round, miner, stage, run, account, and lease identity come from the lease
    the worker holds, never from the sandbox.
    """

    def __init__(self, socket_path: Path, api: ArenaApiClient, state: RunState, events: EventWriter) -> None:
        self._path = Path(socket_path)
        self._api = api
        self._state = state
        self._events = events
        self._server: Optional[socketserver.ThreadingUnixStreamServer] = None
        self._thread: Optional[threading.Thread] = None

    def _dispatch(self, operation_id: str, parameters: Mapping[str, Any], timeout_ms: int) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
        """Bridge one validated operation to the Arena: ``(error_code, None)`` or ``(None, document)``."""

        state = self._state
        with state.lock:
            if state.failed:
                return "worker_unavailable", None
            sequence = state.action_sequence
            state.action_sequence += 1
            refused = state.refusals >= MAX_REFUSED_FRAMES
        if refused:
            # The run's quota or key keeps refusing: answer locally instead of
            # spending an Arena round trip and a ledger row on every request.
            return "budget_exhausted", None
        frame = {"operation_id": operation_id, "parameters": dict(parameters), "timeout_ms": int(timeout_ms), "action_sequence": sequence}
        try:
            document = self._api.provider(state.lease["run_id"], state.lease_token, frame)
        except RunnerError:
            return "worker_unavailable", None
        if "call" not in document or "body_b64" not in document:
            return "worker_unavailable", None
        call = dict(document["call"])
        with state.lock:
            state.calls.append(call)
            if call.get("error_code") in ("budget_refused", "budget_exhausted") or call.get("outcome") == "refused":
                state.refusals += 1
        try:
            self._events.sync_from_call(call)
        except RunnerError:
            return "worker_unavailable", None
        return None, document

    def handle_frame(self, raw: bytes) -> bytes:
        """The judge shim's transport: one length-prefixed operation frame."""

        try:
            operation_id, parameters, timeout_ms = shim.decode_operation_frame(raw)
        except shim.OperationFrameError as exc:
            return shim.encode_worker_error(str(exc) if str(exc) in shim.FRAME_ERROR_CODES else "invalid_frame")
        except operations.OperationError as exc:
            code = getattr(exc, "code", "invalid_request")
            return shim.encode_worker_error(code if code in shim.FRAME_ERROR_CODES else "invalid_request")
        error, document = self._dispatch(operation_id, parameters, timeout_ms)
        if error:
            return shim.encode_worker_error(error)
        return contracts.canonical_json({"status": document["status"], "headers": document["headers"], "body_b64": document["body_b64"]}).encode("utf-8")

    def handle_http(self, method: str, url: str, body: bytes, headers: Mapping[str, str]) -> Tuple[int, Dict[str, str], bytes]:
        """The miner contract: a provider's own HTTP request, sent over the socket without a credential."""

        try:
            operation_id, parameters = operations.match_request(method, url, body, headers)
        except operations.OperationError as exc:
            code = getattr(exc, "code", "invalid_request")
            return HTTP_ERROR_STATUS.get(code, 400), {}, _http_error_body(code)
        error, document = self._dispatch(operation_id, parameters, shim.DEFAULT_TIMEOUT_MS)
        if error:
            return HTTP_ERROR_STATUS.get(error, 400), {}, _http_error_body(error)
        response_headers = {str(name): str(value) for name, value in dict(document.get("headers") or {}).items()}
        try:
            payload = base64.b64decode(str(document["body_b64"]), validate=True)
        except (ValueError, TypeError):
            return 503, {}, _http_error_body("worker_unavailable")
        return int(document["status"]), response_headers, payload

    def start(self) -> None:
        server_self = self

        class HttpBridge(http.server.BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"
            server_version = "LabArenaWorker/1"
            sys_version = ""

            def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 - stdlib signature
                return

            def _bridge(self) -> None:
                self.close_connection = True
                if self.headers.get("Transfer-Encoding"):
                    self._answer(400, {}, _http_error_body("invalid_request"))
                    return
                try:
                    length = int(self.headers.get("Content-Length") or "0")
                except ValueError:
                    self._answer(400, {}, _http_error_body("invalid_request"))
                    return
                if length < 0 or length > shim.MAX_FRAME_BYTES:
                    self._answer(413, {}, _http_error_body("request_too_large"))
                    return
                body = self.rfile.read(length) if length else b""
                host = (self.headers.get("Host") or "").strip()
                if not host or "/" in host or any(ch.isspace() for ch in host):
                    self._answer(400, {}, _http_error_body("invalid_request"))
                    return
                headers = {name: value for name, value in self.headers.items()}
                status, response_headers, payload = server_self.handle_http(self.command, "https://" + host + self.path, body, headers)
                self._answer(status, response_headers, payload)

            def _answer(self, status: int, headers: Mapping[str, str], payload: bytes) -> None:
                self.send_response(status)
                for name, value in headers.items():
                    if name.lower() in ("content-length", "transfer-encoding", "connection"):
                        continue
                    self.send_header(name, value)
                self.send_header("Content-Length", str(len(payload)))
                self.send_header("Connection", "close")
                self.end_headers()
                if self.command != "HEAD":
                    self.wfile.write(payload)

            do_GET = do_POST = do_PUT = do_PATCH = do_DELETE = do_HEAD = do_OPTIONS = _bridge

        class Handler(socketserver.BaseRequestHandler):
            def handle(self) -> None:
                connection = self.request
                connection.settimeout(300)
                try:
                    first = connection.recv(1, socket.MSG_PEEK)
                except OSError:
                    return
                if first and first[0] in HTTP_FIRST_BYTES:
                    try:
                        HttpBridge(connection, self.client_address, self.server)
                    except (OSError, ValueError):
                        return
                    return
                try:
                    header = _recv_exact(connection, 4)
                    size = int.from_bytes(header, "big")
                    if size < 2 or size > shim.MAX_FRAME_BYTES:
                        payload = shim.encode_worker_error("frame_too_large")
                    else:
                        payload = server_self.handle_frame(_recv_exact(connection, size))
                    connection.sendall(len(payload).to_bytes(4, "big") + payload)
                except OSError:
                    return

        if self._path.exists():
            self._path.unlink()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._server = socketserver.ThreadingUnixStreamServer(str(self._path), Handler)
        self._server.daemon_threads = True
        os.chmod(self._path, 0o666)
        self._thread = threading.Thread(target=self._server.serve_forever, name="lab-arena-worker-socket", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        if self._path.exists():
            self._path.unlink()


def _http_error_body(code: str) -> bytes:
    return json.dumps({"error": {"code": str(code)}}, sort_keys=True).encode("utf-8")


def _recv_exact(connection: socket.socket, size: int) -> bytes:
    output = bytearray()
    while len(output) < size:
        chunk = connection.recv(min(65536, size - len(output)))
        if not chunk:
            raise OSError("connection closed")
        output.extend(chunk)
    return bytes(output)


# ---------------------------------------------------------------------------
# Executing one assignment
# ---------------------------------------------------------------------------


class SandboxRuntime(Protocol):
    def run_icp(self, spec: runtime.SandboxSpec, **kwargs: Any) -> runtime.SandboxResult: ...


@dataclass
class RunnerConfig:
    # None follows the Arena's current round (production); a value pins one round.
    round_id: Optional[str]
    identity: RunnerIdentity
    api: ArenaApiClient
    sandbox_runtime: SandboxRuntime
    image_cache: ImageCache
    worker_release_hash: str
    work_dir: Path
    max_parallel_runs: int = DEFAULT_MAX_PARALLEL_RUNS
    slot_ceiling: int = contracts.RUNNER_SLOT_CEILING
    wall_clock_seconds: int = contracts.ICP_WALL_CLOCK_SECONDS
    # Waits between completion retries after a transport or server failure.
    completion_retry_seconds: Tuple[float, ...] = (2.0, 5.0)
    evaluation_date: str = ""  # fallback only; every lease names the round's evaluation date
    clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc)
    socket_root: Path = Path(DEFAULT_SOCKET_ROOT)
    # Release identity checked against every round the runner follows.
    runtime_lock_hash: str = ""
    # Where leases may point: the round's Arena repository for miner images and
    # the exact judge reference for scoring assignments (empty: not checked).
    registry_repository: str = ""
    scorer_image_reference: str = ""

    def __post_init__(self) -> None:
        if isinstance(self.max_parallel_runs, bool) or not isinstance(self.max_parallel_runs, int) or self.max_parallel_runs < 1:
            raise RunnerError("max_parallel_runs must be a positive integer")
        self.work_dir = Path(self.work_dir)
        self.socket_root = Path(self.socket_root)
        self.socket_root.mkdir(parents=True, exist_ok=True)

    def adopt_round(self, configuration: Mapping[str, Any]) -> None:
        """Pin the image sources a round's signed configuration names."""

        self.registry_repository = str(configuration.get("registry_repository") or "")
        self.scorer_image_reference = str((configuration.get("release") or {}).get("scorer_image_reference") or "")


def max_parallel_runs_from_environment(environ: Mapping[str, str] = os.environ) -> int:
    raw = str(environ.get(MAX_PARALLEL_ENV) or "").strip()
    if not raw:
        return DEFAULT_MAX_PARALLEL_RUNS
    try:
        value = int(raw)
    except ValueError as exc:
        raise RunnerError("%s must be a positive integer" % MAX_PARALLEL_ENV) from exc
    if value < 1:
        raise RunnerError("%s must be a positive integer" % MAX_PARALLEL_ENV)
    return value


def _chunk_log(data: bytes) -> List[str]:
    text = bytes(data).decode("utf-8", errors="replace")
    return [text[index : index + LOG_EVENT_CHUNK_BYTES] for index in range(0, len(text), LOG_EVENT_CHUNK_BYTES)] or [""]


def cost_record(call: Mapping[str, Any]) -> Dict[str, Any]:
    return {"call_identity": call.get("call_identity"), "reserved_microusd": call.get("reserved_microusd"), "actual_microusd": call.get("actual_microusd"), "outcome": call.get("outcome")}


def provider_call_record(call: Mapping[str, Any]) -> Dict[str, Any]:
    return {"call_identity": call.get("call_identity"), "operation_id": call.get("operation_id"), "request_hash": call.get("request_hash"), "outcome": call.get("outcome"), "status": call.get("status"), "response_hash": call.get("response_hash")}


def _ordered_calls(calls: Sequence[Mapping[str, Any]]) -> List[Mapping[str, Any]]:
    # Roots are order-independent (sorted by call identity) so the service can
    # rebuild them from the ledger without knowing the model's concurrency.
    unique: Dict[str, Mapping[str, Any]] = {}
    for call in calls:
        identity = str(call.get("call_identity") or "")
        if identity:
            unique[identity] = call
    return [unique[key] for key in sorted(unique)]


def cost_root(calls: Sequence[Mapping[str, Any]]) -> str:
    return contracts.ordered_root([contracts.document_hash(cost_record(c)) for c in _ordered_calls(calls)])


def provider_call_root(calls: Sequence[Mapping[str, Any]]) -> str:
    return contracts.ordered_root([contracts.document_hash(provider_call_record(c)) for c in _ordered_calls(calls)])


class AssignmentExecutor:
    def __init__(self, config: RunnerConfig) -> None:
        self._config = config

    def execute(self, lease: Mapping[str, Any], lease_token: str, icp: Mapping[str, Any]) -> Dict[str, Any]:
        """Run one leased ICP end to end and return the completion envelope."""

        config = self._config
        state = RunState(lease=dict(lease), lease_token=lease_token, event_cursor=int(lease["event_cursor"]), event_head_hash=str(lease["event_head_hash"] or ""))
        events = EventWriter(config.api, state, config.clock)
        run_dir = Path(tempfile.mkdtemp(prefix="run-", dir=str(config.work_dir)))
        input_dir = run_dir / "input"
        output_dir = run_dir / "output"
        input_dir.mkdir()
        output_dir.mkdir()
        # Unix socket paths are limited to about 100 bytes, so the worker
        # socket lives in a short directory of its own, never under the run dir.
        socket_dir = Path(tempfile.mkdtemp(prefix="la", dir=str(config.socket_root)))
        socket_path = socket_dir / runtime.SANDBOX_SOCKET_NAME
        if len(str(socket_path).encode("utf-8")) > MAX_SOCKET_PATH_BYTES:
            shutil.rmtree(socket_dir, ignore_errors=True)
            shutil.rmtree(run_dir, ignore_errors=True)
            raise RunnerError("worker socket path exceeds %d bytes; set a shorter socket_root" % MAX_SOCKET_PATH_BYTES)
        server = WorkerSocketServer(socket_path, config.api, state, events)
        started_at = _timestamp(config.clock)
        kind = str(lease.get("kind") or "execute")
        scoring_run = kind == "score"
        terminal = "judge_error" if scoring_run else "model_error"
        output_document: Optional[Dict[str, Any]] = None
        result: Optional[runtime.SandboxResult] = None
        evaluation_date = str(lease.get("evaluation_date") or config.evaluation_date)
        try:
            if scoring_run:
                # A scoring assignment runs the Arena judge image on one accepted
                # output; the judge's provider calls cross the same socket.
                input_document = scoring.build_scoring_input(
                    work_item_id=str(lease["work_item_id"]), icp=icp, companies=list((lease.get("scored_output") or {}).get("companies") or []),
                    policy=lease["scorer_policy"], evaluation_date=evaluation_date,
                )
                extra_environment = {shim.TRUSTED_SCORER_ENV: "1"}
            else:
                input_document = {
                    "schema_version": "leadpoet.lab_arena.icp_input.v1",
                    "icp": dict(icp),
                    "evaluation_date": evaluation_date,
                    "company_limit": int(icp.get("max_companies") or 5),
                    "provider_operations": sorted(operations.OPERATIONS),
                }
                extra_environment = {}
            (input_dir / runtime.INPUT_FILE_NAME).write_text(json.dumps(input_document, sort_keys=True), encoding="utf-8")
            # The image and its process come from the lease, pinned by the Arena
            # at admission; the runner checks them against the round's sources.
            image_reference = str(lease.get("image_reference") or "")
            _check_lease_image(config, kind, image_reference, str(lease["image_digest"]))
            rootfs = config.image_cache.rootfs_for(str(lease["image_digest"]), image_reference)
            spec = runtime.SandboxSpec(
                sandbox_id="arena-%s" % contracts.document_hash(lease["run_id"])[7:39],
                rootfs_path=rootfs,
                input_dir=input_dir,
                output_dir=output_dir,
                socket_path=socket_path,
                entry_command=tuple(str(item) for item in (lease.get("entry_command") or ())),
                image_environment={str(name): str(value) for name, value in dict(lease.get("image_environment") or {}).items()},
                working_dir=str(lease.get("working_dir") or ""),
                evaluation_date=evaluation_date,
                random_seed=int(contracts.document_hash(lease["assignment_id"])[7:15], 16) % (2 ** 32),
                wall_clock_seconds=contracts.SCORING_WALL_CLOCK_SECONDS if scoring_run else config.wall_clock_seconds,
                extra_environment=extra_environment,
            )
            server.start()
            events.append("process_started", {"run_id": lease["run_id"], "assignment_id": lease["assignment_id"], "icp_position": lease["icp_position"], "image_digest": lease["image_digest"], "attempt": lease["attempt"]})
            result = config.sandbox_runtime.run_icp(spec)
            for stream_name, data, truncated in (("stdout", result.stdout, result.stdout_truncated), ("stderr", result.stderr, result.stderr_truncated)):
                for index, chunk in enumerate(_chunk_log(data)):
                    events.append(stream_name, {"chunk": index, "text": chunk, "truncated": bool(truncated)})
            if result.timed_out:
                terminal = "judge_timeout" if scoring_run else "model_timeout"
                events.append("process_finished", {"exit_code": result.exit_code, "timed_out": True, "wall_seconds": result.wall_seconds})
            else:
                events.append("process_finished", {"exit_code": result.exit_code, "timed_out": False, "wall_seconds": result.wall_seconds})
                if result.output_error or result.output_bytes is None:
                    terminal = "judge_error" if scoring_run else ("invalid_output" if result.output_error else "model_error")
                    events.append("output_rejected", {"reason": result.output_error or "no_output"})
                elif scoring_run:
                    try:
                        output_document = scoring.scoring_output_from_bytes(result.output_bytes)
                    except scoring.ScoringError as exc:
                        terminal = "judge_error"
                        events.append("output_rejected", {"reason": str(exc)[:200]})
                    else:
                        if "failure" in output_document:
                            terminal = str(output_document["failure"])
                            events.append("output_rejected", {"reason": terminal})
                            output_document = None
                        else:
                            terminal = "accepted"
                            events.append("output_validated", {"breakdown_count": len(output_document["breakdowns"]), "output_hash": contracts.document_hash(output_document)})
                else:
                    try:
                        output_document = output_document_from_bytes(result.output_bytes)
                    except OutputInvalid as exc:
                        terminal = "invalid_output"
                        events.append("output_rejected", {"reason": str(exc)[:200]})
                    else:
                        terminal = "accepted"
                        events.append("output_validated", {"company_count": len(output_document["companies"]), "output_hash": output_document_hash(output_document)})
            if scoring_run and terminal in ("judge_error", "judge_timeout"):
                # The judge folds provider errors into its own failure handling, so
                # the worker decides from the ledger it kept: a key or quota refusal
                # on the scored miner's keys is that miner's outcome, not the judge's.
                refusals = [str(call.get("error_code")) for call in state.calls if call.get("error_code") in scoring.KEY_REFUSAL_CODES]
                if refusals:
                    terminal = "judge_key_refused"
                    events.append("output_rejected", {"reason": "judge_key_refused:" + refusals[0]})
        finally:
            server.stop()
            shutil.rmtree(run_dir, ignore_errors=True)
            shutil.rmtree(socket_dir, ignore_errors=True)
        finished_at = _timestamp(config.clock)
        round_id = str(lease.get("round_id") or config.round_id or "")
        receipt_body = {
            "schema_version": contracts.ICP_RECEIPT_SCHEMA_VERSION,
            "round_id": round_id,
            "submission_id": lease["submission_id"],
            "assignment_id": lease["assignment_id"],
            "attempt": int(lease["attempt"]),
            "stage": int(lease["stage"]),
            "icp_position": int(lease["icp_position"]),
            "lease_generation": int(lease["lease_generation"]),
            "runner_hotkey": config.identity.hotkey,
            "miner_hotkey": lease["miner_hotkey"],
            "worker_release_hash": config.worker_release_hash,
            "image_digest": lease["image_digest"],
            "icp_hash": lease["icp_hash"],
            "kind": kind,
            "provider_call_root": provider_call_root(state.calls),
            "private_event_root": contracts.ordered_root(state.event_hashes),
            "output_hash": (contracts.document_hash(output_document) if scoring_run else output_document_hash(output_document)) if output_document is not None else contracts.document_hash(None),
            "cost_root": cost_root(state.calls),
            "resource_summary": {
                "wall_seconds": float(result.wall_seconds) if result else 0.0,
                "cpu_seconds": float(result.cpu_seconds) if result else 0.0,
                "max_rss_bytes": int(result.max_rss_bytes) if result else 0,
                "stdout_bytes": len(result.stdout) if result else 0,
                "stderr_bytes": len(result.stderr) if result else 0,
                "provider_call_count": len(state.calls),
            },
            "started_at": started_at,
            "finished_at": finished_at,
            "terminal_status": terminal,
        }
        receipt = contracts.finalize_icp_receipt(receipt_body)
        receipt["runner_signature"] = config.identity.sign(receipt["receipt_hash"])
        receipt["run_id"] = lease["run_id"]
        body = {"receipt": receipt, "output": output_document, "calls": state.calls, "event_hashes": state.event_hashes, "lease_token": lease_token}
        return contracts.build_signed_request(
            scope=contracts.SCOPE_COMPLETE,
            round_id=round_id,
            hotkey=config.identity.hotkey,
            body=body,
            timestamp=int(config.clock().timestamp()),
            sign_message=config.identity.sign,
        )


def _check_lease_image(config: RunnerConfig, kind: str, image_reference: str, image_digest: str) -> None:
    """A lease may only point at the round's Arena repository or the round's exact judge image."""

    if not image_reference:
        raise RunnerError("lease carries no image reference")
    reference = images.parse_reference(image_reference)
    if reference.digest != image_digest:
        raise RunnerError("lease image reference does not name the lease digest")
    if kind == "score":
        if config.scorer_image_reference and image_reference != config.scorer_image_reference:
            raise RunnerError("scoring lease names an image other than the round's judge")
    elif config.registry_repository and reference.name != config.registry_repository:
        raise RunnerError("lease image is outside the round's Arena repository")


# ---------------------------------------------------------------------------
# Claim loop
# ---------------------------------------------------------------------------


class Runner:
    def __init__(self, config: RunnerConfig) -> None:
        self._config = config
        self._executor = AssignmentExecutor(config)
        self._slots = threading.BoundedSemaphore(config.max_parallel_runs)
        self._pool = ThreadPoolExecutor(max_workers=config.max_parallel_runs, thread_name_prefix="lab-arena-slot")
        self.completed: List[Dict[str, Any]] = []
        self.abandoned = 0
        self._pinned = config.round_id is not None
        self._round_id: Optional[str] = config.round_id

    @property
    def round_id(self) -> Optional[str]:
        return self._round_id

    def refresh_round(self) -> Optional[str]:
        """Follow the Arena's current round; a changed round is re-verified before any claim.

        A runner started without ``--round-id`` asks ``/arena/v1/current`` at
        every idle poll, so the daily rounds roll over without a restart.
        """

        if self._pinned:
            return self._round_id
        config = self._config
        current = (config.api.current().get("round") or {}) if True else {}
        round_id = current.get("round_id") if isinstance(current, Mapping) else None
        if not round_id:
            self._round_id = None
            return None
        if round_id != self._round_id:
            configuration = config.api.round(str(round_id)).get("configuration") or {}
            verify_release_against_round(configuration, worker_release_hash=config.worker_release_hash, runtime_lock_hash=config.runtime_lock_hash)
            config.adopt_round(configuration)
            self._round_id = str(round_id)
        return self._round_id

    def claim_one(self) -> Dict[str, Any]:
        config = self._config
        if self._round_id is None:
            return {"status": "no_open_round"}
        envelope = contracts.build_signed_request(
            scope=contracts.SCOPE_CLAIM,
            round_id=self._round_id,
            hotkey=config.identity.hotkey,
            body={"declared_parallelism": config.max_parallel_runs, "worker_release_hash": config.worker_release_hash},
            timestamp=int(config.clock().timestamp()),
            sign_message=config.identity.sign,
        )
        return config.api.claim(envelope)

    def _run_lease(self, lease: Mapping[str, Any]) -> None:
        try:
            envelope = self._executor.execute(lease, str(lease["lease_token"]), lease["icp"])
            result = self._complete_with_retries(envelope)
            self.completed.append({"run_id": lease["run_id"], "result": result})
        except Exception as exc:  # the attempt fails closed; the service expires the lease
            self.abandoned += 1
            self.completed.append({"run_id": lease.get("run_id"), "error": type(exc).__name__})
        finally:
            self._slots.release()

    def _complete_with_retries(self, envelope: Mapping[str, Any]) -> Dict[str, Any]:
        """Deliver the signed completion; a transport or server failure is retried briefly.

        The envelope is idempotent, so a lost response or a transient object
        store failure on the Arena costs a retry, not the whole sandbox run.
        A rejection the Arena returns as a document is never retried.
        """

        delays = tuple(self._config.completion_retry_seconds)
        for attempt in range(len(delays) + 1):
            try:
                return self._config.api.complete(envelope)
            except Exception:
                if attempt >= len(delays):
                    raise
                time.sleep(max(0.0, float(delays[attempt])))
        raise RunnerError("completion retries exhausted")  # pragma: no cover - the loop returns or raises

    def run_once(self, *, max_claims: int = 1000) -> int:
        """Claim while a local slot is free; return the number of leases taken."""

        if not self._pinned:
            try:
                self.refresh_round()
            except RunnerError:
                return 0  # the Arena or the round is unavailable: poll again later
        taken = 0
        futures = []
        for _ in range(max_claims):
            if not self._slots.acquire(blocking=False):
                break
            try:
                response = self.claim_one()
            except RunnerError:
                self._slots.release()
                break
            if response.get("status") != "leased":
                self._slots.release()
                break
            taken += 1
            futures.append(self._pool.submit(self._run_lease, response))
        for future in futures:
            future.result()
        return taken

    def close(self) -> None:
        self._pool.shutdown(wait=True)
