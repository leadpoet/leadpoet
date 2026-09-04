"""Arena runner: sandbox one bundle on one benchmark input.

A runner follows every running Arena round, claims one pending assignment per
free local slot, materializes the pinned image's root filesystem from the
Arena registry, executes the fixed ``/agent/run`` entrypoint in a fresh gVisor
sandbox for that single ICP, bridges the sandbox's provider requests (plain
HTTP over the worker socket, or the judge shim's operation frames) to the
service's provider endpoint, appends an operational event log, and submits one
small run result through its authenticated API request. It never reports a score, never holds a database
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
from collections import OrderedDict
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Mapping, Optional, Protocol, Sequence, Tuple
from urllib.parse import urlsplit

import httpx

from lab_arena import contracts, images, operations, runtime, scoring, shim
from lab_arena.contracts import ArenaContractError
from lab_arena.output import OutputInvalid, output_document_from_bytes

DEFAULT_MAX_PARALLEL_RUNS = contracts.RUNNER_SLOT_CEILING
MAX_PARALLEL_ENV = "LAB_ARENA_MAX_PARALLEL_RUNS"
DEFAULT_SOCKET_ROOT = "/tmp"
MAX_REFUSED_FRAMES = 25  # after this many refused calls the worker answers a run's frames locally
# A request on the worker socket is either a length-prefixed operation frame
# (first byte 0x00: the judge shim) or an HTTP request (an ASCII method).
HTTP_FIRST_BYTES = b"GPHDO"
HTTP_ERROR_STATUS = {"budget_exhausted": 402, "worker_unavailable": 503, "request_too_large": 413}
IMAGE_DIGEST_RE = __import__("re").compile(r"^(?:[a-z0-9][a-z0-9._/-]{0,200}@)?sha256:[0-9a-f]{64}$")
MAX_SOCKET_PATH_BYTES = 100
API_TIMEOUT_SECONDS = 30.0
PROVIDER_API_TIMEOUT_GRACE_SECONDS = 15.0
MAX_PROVIDER_API_TIMEOUT_SECONDS = 135.0
DEFAULT_IMAGE_CACHE_MAX_BYTES = 16 * 1024 * 1024 * 1024
DEFAULT_IMAGE_CACHE_MAX_ENTRIES = 32
MAX_WORKER_CONNECTIONS = 8
WORKER_SOCKET_READ_TIMEOUT_SECONDS = 10.0


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

    def complete(self, envelope: Mapping[str, Any]) -> Dict[str, Any]: ...

    def current(self) -> Dict[str, Any]: ...

    def round(self, round_id: str) -> Dict[str, Any]: ...


class HttpArenaApiClient:
    """HTTPS client for ``/arena/v1`` runner endpoints (section 14.3)."""

    def __init__(self, base_url: str, *, client: Optional[httpx.Client] = None) -> None:
        try:
            parsed = urlsplit(base_url)
            hostname = parsed.hostname
            parsed.port  # force validation of a malformed port
        except ValueError:
            hostname = None
            parsed = urlsplit("")
        secure = parsed.scheme == "https" and bool(hostname)
        loopback = parsed.scheme == "http" and hostname in ("localhost", "127.0.0.1", "::1")
        if (not secure and not loopback) or parsed.username is not None or parsed.password is not None or parsed.fragment:
            raise RunnerError("Arena API base URL must be https (or loopback for tests)")
        self._base_url = base_url.rstrip("/")
        self._client = client or httpx.Client(http1=True, http2=False, follow_redirects=False, timeout=httpx.Timeout(API_TIMEOUT_SECONDS), trust_env=False)

    def _post(
        self,
        path: str,
        document: Mapping[str, Any],
        *,
        headers: Optional[Mapping[str, str]] = None,
        timeout_seconds: float = API_TIMEOUT_SECONDS,
    ) -> Dict[str, Any]:
        try:
            response = self._client.post(
                self._base_url + path,
                content=contracts.canonical_json(document).encode("utf-8"),
                headers={"content-type": "application/json", **(headers or {})},
                timeout=httpx.Timeout(float(timeout_seconds)),
            )
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
        requested_timeout = frame.get("timeout_ms")
        timeout_seconds = API_TIMEOUT_SECONDS
        if isinstance(requested_timeout, int) and not isinstance(requested_timeout, bool):
            timeout_seconds = max(
                API_TIMEOUT_SECONDS,
                min(
                    MAX_PROVIDER_API_TIMEOUT_SECONDS,
                    requested_timeout / 1000.0 + PROVIDER_API_TIMEOUT_GRACE_SECONDS,
                ),
            )
        return self._post(
            "/arena/v1/runs/%s/provider" % run_id,
            frame,
            headers={"x-lab-arena-lease": lease_token},
            timeout_seconds=timeout_seconds,
        )

    def complete(self, envelope: Mapping[str, Any]) -> Dict[str, Any]:
        return self._post("/arena/v1/runs/%s/complete" % envelope["body"]["run_id"], envelope)

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


# ---------------------------------------------------------------------------
# Runner identity and image cache
# ---------------------------------------------------------------------------


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


def _cache_path_bytes(path: Path) -> int:
    """Return allocated bytes once per inode without following symlinks."""

    total = 0
    seen = set()
    for directory, names, files in os.walk(path, followlinks=False):
        for name in list(names) + list(files):
            candidate = Path(directory) / name
            try:
                stat_result = candidate.lstat()
            except OSError:
                continue
            identity = (int(stat_result.st_dev), int(stat_result.st_ino))
            if identity in seen:
                continue
            seen.add(identity)
            blocks = int(getattr(stat_result, "st_blocks", 0)) * 512
            total += blocks if blocks > 0 else int(stat_result.st_size)
    return total


def _remove_cache_path(path: Path) -> None:
    if path.is_symlink() or not path.is_dir():
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        return
    shutil.rmtree(path, ignore_errors=True)


class ImageCache:
    """A small LRU of materialized images; active root filesystems stay pinned."""

    def __init__(
        self,
        root: Path,
        exporter: ImageExporter,
        *,
        max_bytes: int = DEFAULT_IMAGE_CACHE_MAX_BYTES,
        max_entries: int = DEFAULT_IMAGE_CACHE_MAX_ENTRIES,
    ) -> None:
        if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes < 1:
            raise RunnerError("image cache byte limit is invalid")
        if isinstance(max_entries, bool) or not isinstance(max_entries, int) or max_entries < 1:
            raise RunnerError("image cache entry limit is invalid")
        self._root = Path(root)
        self._exporter = exporter
        self._max_bytes = max_bytes
        self._max_entries = max_entries
        self._lock = threading.Lock()
        self._ready: "OrderedDict[str, Path]" = OrderedDict()
        self._sizes: Dict[str, int] = {}
        self._in_use: Dict[str, int] = {}
        self._root.mkdir(parents=True, exist_ok=True)
        with self._lock:
            self._load_existing_locked()

    def _load_existing_locked(self) -> None:
        existing = []
        for target in sorted(self._root.glob("sha256-*"), key=lambda item: item.name):
            marker = target / ".exported"
            try:
                digest = marker.read_text(encoding="utf-8").strip()
                marker_time = marker.stat().st_mtime_ns
            except (OSError, UnicodeError):
                _remove_cache_path(target)
                continue
            expected_name = "sha256-" + digest.rsplit("sha256:", 1)[-1]
            rootfs = target / "rootfs"
            if not IMAGE_DIGEST_RE.match(digest) or target.name != expected_name or rootfs.is_symlink() or not rootfs.is_dir():
                _remove_cache_path(target)
                continue
            existing.append((marker_time, target.name, digest, rootfs, _cache_path_bytes(target)))
        for _time, _name, digest, rootfs, size in sorted(existing):
            self._ready[digest] = rootfs
            self._sizes[digest] = size
        self._evict_locked()

    def _within_limits_locked(self) -> bool:
        return len(self._ready) <= self._max_entries and sum(self._sizes.values()) <= self._max_bytes

    def _evict_locked(self, *, protected: Sequence[str] = ()) -> None:
        protected_set = set(protected)
        while not self._within_limits_locked():
            victim = next(
                (digest for digest in self._ready if digest not in protected_set and self._in_use.get(digest, 0) == 0),
                None,
            )
            if victim is None:
                return
            rootfs = self._ready.pop(victim)
            self._sizes.pop(victim, None)
            self._in_use.pop(victim, None)
            _remove_cache_path(rootfs.parent)

    def _rootfs_for_locked(self, image_digest: str, image_reference: str) -> Path:
        path = self._ready.get(image_digest)
        if path is not None and path.is_dir() and not path.is_symlink():
            self._ready.move_to_end(image_digest)
            return path
        if path is not None:
            self._ready.pop(image_digest, None)
            self._sizes.pop(image_digest, None)
            self._in_use.pop(image_digest, None)
        # Cache directories are keyed by the content digest alone.
        target = self._root / ("sha256-" + image_digest.rsplit("sha256:", 1)[1])
        if target.exists() or target.is_symlink():
            _remove_cache_path(target)
        target.mkdir(parents=True)
        try:
            self._exporter(image_reference, image_digest, target)
            rootfs = target / "rootfs"
            if rootfs.is_symlink() or not rootfs.is_dir():
                raise RunnerError("image exporter produced no root filesystem")
            (target / ".exported").write_text(image_digest, encoding="utf-8")
            size = _cache_path_bytes(target)
            if size > self._max_bytes:
                raise RunnerError("image exceeds runner cache capacity")
        except Exception:
            _remove_cache_path(target)
            raise
        self._ready[image_digest] = rootfs
        self._sizes[image_digest] = size
        self._evict_locked(protected=(image_digest,))
        if not self._within_limits_locked():
            self._ready.pop(image_digest, None)
            self._sizes.pop(image_digest, None)
            _remove_cache_path(target)
            raise RunnerError("image cache capacity is in use")
        return rootfs

    def rootfs_for(self, image_digest: str, image_reference: str = "") -> Path:
        if not isinstance(image_digest, str) or not IMAGE_DIGEST_RE.match(image_digest):
            raise RunnerError("image digest is invalid")
        with self._lock:
            return self._rootfs_for_locked(image_digest, image_reference)

    @contextmanager
    def acquire(self, image_digest: str, image_reference: str = "") -> Iterator[Path]:
        """Pin one cached rootfs until the caller's sandbox has stopped."""

        if not isinstance(image_digest, str) or not IMAGE_DIGEST_RE.match(image_digest):
            raise RunnerError("image digest is invalid")
        with self._lock:
            rootfs = self._rootfs_for_locked(image_digest, image_reference)
            self._in_use[image_digest] = self._in_use.get(image_digest, 0) + 1
        try:
            yield rootfs
        finally:
            with self._lock:
                remaining = self._in_use.get(image_digest, 0) - 1
                if remaining > 0:
                    self._in_use[image_digest] = remaining
                else:
                    self._in_use.pop(image_digest, None)
                self._evict_locked()


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
# Per-run state and the worker socket
# ---------------------------------------------------------------------------


@dataclass
class RunState:
    lease: Dict[str, Any]
    lease_token: str
    calls: List[Dict[str, Any]] = field(default_factory=list)
    action_sequence: int = 0
    refusals: int = 0  # refused calls answered by the Arena for this run
    lock: threading.Lock = field(default_factory=threading.Lock)


def _timestamp(clock: Callable[[], datetime]) -> str:
    return clock().astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class WorkerSocketServer:
    """Unix-socket bridge: operation frames in, provider responses out.

    Frames carry only the operation id, validated parameters, and a timeout;
    round, miner, stage, run, account, and lease identity come from the lease
    the worker holds, never from the sandbox.
    """

    def __init__(
        self,
        socket_path: Path,
        api: ArenaApiClient,
        state: RunState,
        *,
        max_connections: int = MAX_WORKER_CONNECTIONS,
        read_timeout_seconds: float = WORKER_SOCKET_READ_TIMEOUT_SECONDS,
    ) -> None:
        if isinstance(max_connections, bool) or not isinstance(max_connections, int) or max_connections < 1:
            raise RunnerError("worker socket connection limit is invalid")
        if read_timeout_seconds <= 0:
            raise RunnerError("worker socket read timeout is invalid")
        self._path = Path(socket_path)
        self._api = api
        self._state = state
        self._max_connections = max_connections
        self._read_timeout_seconds = float(read_timeout_seconds)
        self._server: Optional[socketserver.ThreadingUnixStreamServer] = None
        self._thread: Optional[threading.Thread] = None

    def _dispatch(self, operation_id: str, parameters: Mapping[str, Any], timeout_ms: int) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
        """Bridge one validated operation to the Arena: ``(error_code, None)`` or ``(None, document)``."""

        state = self._state
        with state.lock:
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
        slots = threading.BoundedSemaphore(self._max_connections)

        class BoundedServer(socketserver.ThreadingUnixStreamServer):
            daemon_threads = True

            def verify_request(self, request: Any, client_address: Any) -> bool:
                return slots.acquire(blocking=False)

            def process_request(self, request: Any, client_address: Any) -> None:
                try:
                    super().process_request(request, client_address)
                except Exception:
                    slots.release()
                    raise

            def process_request_thread(self, request: Any, client_address: Any) -> None:
                try:
                    super().process_request_thread(request, client_address)
                finally:
                    slots.release()

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
                connection.settimeout(server_self._read_timeout_seconds)
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
        self._server = BoundedServer(str(self._path), Handler)
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
    work_dir: Path
    max_parallel_runs: int = DEFAULT_MAX_PARALLEL_RUNS
    slot_ceiling: int = contracts.RUNNER_SLOT_CEILING
    wall_clock_seconds: int = contracts.ICP_WALL_CLOCK_SECONDS
    # Waits between completion retries after a transport or server failure.
    completion_retry_seconds: Tuple[float, ...] = (2.0, 5.0)
    evaluation_date: str = ""  # fallback only; every lease names the round's evaluation date
    clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc)
    socket_root: Path = Path(DEFAULT_SOCKET_ROOT)
    # Where miner leases may point (empty: not checked).
    registry_repository: str = ""
    # Rounds overlap, so retain each followed round's repository.
    round_sources: Dict[str, Dict[str, str]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if (
            isinstance(self.max_parallel_runs, bool)
            or not isinstance(self.max_parallel_runs, int)
            or not 1 <= self.max_parallel_runs <= contracts.RUNNER_SLOT_CEILING
        ):
            raise RunnerError(
                "max_parallel_runs must be between 1 and %d"
                % contracts.RUNNER_SLOT_CEILING
            )
        self.work_dir = Path(self.work_dir)
        self.socket_root = Path(self.socket_root)
        self.socket_root.mkdir(parents=True, exist_ok=True)

    def adopt_round(self, configuration: Mapping[str, Any]) -> None:
        """Use the miner image repository named by a round configuration.

        The sources are kept per round because a runner follows every running
        round at once; the flat fields hold the last adopted round's sources
        for a runner pinned to one round.
        """

        sources = {
            "registry_repository": str(configuration.get("registry_repository") or ""),
        }
        round_id = str(configuration.get("round_id") or "")
        if round_id:
            self.round_sources[round_id] = sources
        self.registry_repository = sources["registry_repository"]

    def sources_for(self, round_id: str) -> Dict[str, str]:
        """The image sources pinned for ``round_id`` (the flat fields when the round was not adopted by id)."""

        return dict(self.round_sources.get(str(round_id)) or {"registry_repository": self.registry_repository})


def max_parallel_runs_from_environment(environ: Mapping[str, str] = os.environ) -> int:
    raw = str(environ.get(MAX_PARALLEL_ENV) or "").strip()
    if not raw:
        return DEFAULT_MAX_PARALLEL_RUNS
    try:
        value = int(raw)
    except ValueError as exc:
        raise RunnerError("%s must be a positive integer" % MAX_PARALLEL_ENV) from exc
    if not 1 <= value <= contracts.RUNNER_SLOT_CEILING:
        raise RunnerError(
            "%s must be between 1 and %d"
            % (MAX_PARALLEL_ENV, contracts.RUNNER_SLOT_CEILING)
        )
    return value


class AssignmentExecutor:
    def __init__(self, config: RunnerConfig) -> None:
        self._config = config

    def execute(self, lease: Mapping[str, Any], lease_token: str, icp: Mapping[str, Any]) -> Dict[str, Any]:
        """Run one leased ICP end to end and return the completion envelope."""

        config = self._config
        state = RunState(lease=dict(lease), lease_token=lease_token)
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
        server = WorkerSocketServer(socket_path, config.api, state)
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
                    scored_run_id=str(lease["scored_run_id"]), icp=icp, companies=list((lease.get("scored_output") or {}).get("companies") or []),
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
            # The lease names immutable image bytes. The host owns both process
            # commands; miner OCI ENTRYPOINT, CMD, ENV, and WORKDIR are ignored.
            image_reference = str(lease.get("image_reference") or "")
            _check_lease_image(config, kind, image_reference, str(lease["image_digest"]), round_id=str(lease.get("round_id") or config.round_id or ""))
            with config.image_cache.acquire(str(lease["image_digest"]), image_reference) as rootfs:
                spec = runtime.SandboxSpec(
                    sandbox_id="arena-%s" % contracts.document_hash(lease["run_id"])[7:39],
                    rootfs_path=rootfs,
                    input_dir=input_dir,
                    output_dir=output_dir,
                    socket_path=socket_path,
                    entry_command=runtime.SCORER_ENTRY_COMMAND if scoring_run else runtime.AGENT_ENTRY_COMMAND,
                    working_dir=runtime.SCORER_WORKING_DIR if scoring_run else runtime.AGENT_WORKING_DIR,
                    evaluation_date=evaluation_date,
                    random_seed=int(contracts.document_hash(lease["assignment_id"])[7:15], 16) % (2 ** 32),
                    wall_clock_seconds=contracts.SCORING_WALL_CLOCK_SECONDS if scoring_run else config.wall_clock_seconds,
                    extra_environment=extra_environment,
                )
                server.start()
                result = config.sandbox_runtime.run_icp(spec)
            if result.timed_out:
                terminal = "judge_timeout" if scoring_run else "model_timeout"
            else:
                if result.output_error or result.output_bytes is None:
                    terminal = "judge_error" if scoring_run else ("invalid_output" if result.output_error else "model_error")
                elif scoring_run:
                    try:
                        output_document = scoring.scoring_output_from_bytes(result.output_bytes)
                    except scoring.ScoringError as exc:
                        terminal = "judge_error"
                    else:
                        if "failure" in output_document:
                            terminal = str(output_document["failure"])
                            output_document = None
                        else:
                            terminal = "accepted"
                else:
                    try:
                        output_document = output_document_from_bytes(result.output_bytes)
                    except OutputInvalid as exc:
                        terminal = "invalid_output"
                    else:
                        terminal = "accepted"
            # A shared host key or account failure is infrastructure when it
            # prevents an output. An agent that handles the failure and still
            # returns a valid output has completed the assignment.
            provider_infrastructure_failed = any(operations.provider_status_is_infrastructure(call.get("provider_status")) for call in state.calls)
            if provider_infrastructure_failed and terminal != "accepted":
                terminal = "judge_error" if scoring_run else "provider_error"
                output_document = None
        finally:
            server.stop()
            shutil.rmtree(run_dir, ignore_errors=True)
            shutil.rmtree(socket_dir, ignore_errors=True)
        finished_at = _timestamp(config.clock)
        round_id = str(lease.get("round_id") or config.round_id or "")
        run_result = {
            "schema_version": contracts.RUN_RESULT_SCHEMA_VERSION,
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
        body = {"run_id": lease["run_id"], "result": run_result, "output": output_document, "lease_token": lease_token}
        return contracts.build_signed_request(
            scope=contracts.SCOPE_COMPLETE,
            round_id=round_id,
            hotkey=config.identity.hotkey,
            body=body,
            timestamp=int(config.clock().timestamp()),
            sign_message=config.identity.sign,
        )


def _check_lease_image(config: RunnerConfig, kind: str, image_reference: str, image_digest: str, *, round_id: str = "") -> None:
    """A lease names pinned bytes; miner images stay in the Arena repository."""

    if not image_reference:
        raise RunnerError("lease carries no image reference")
    reference = images.parse_reference(image_reference)
    if reference.digest != image_digest:
        raise RunnerError("lease image reference does not name the lease digest")
    sources = config.sources_for(round_id)
    if kind != "score" and sources["registry_repository"] and reference.name != sources["registry_repository"]:
        raise RunnerError("lease image is outside the round's Arena repository")


# ---------------------------------------------------------------------------
# Claim loop
# ---------------------------------------------------------------------------

# The round statuses in which assignments can be leased: both execution and
# scoring windows in the two-stage competition.
WORKING_STATUSES = ("stage1", "stage1_scoring", "stage2", "stage2_scoring")


class Runner:
    def __init__(self, config: RunnerConfig) -> None:
        self._config = config
        self._executor = AssignmentExecutor(config)
        self._slots = threading.BoundedSemaphore(config.max_parallel_runs)
        self._pool = ThreadPoolExecutor(max_workers=config.max_parallel_runs, thread_name_prefix="lab-arena-slot")
        self.completed: List[Dict[str, Any]] = []
        self.abandoned = 0
        self._pinned = config.round_id is not None
        self._round_ids: List[str] = [config.round_id] if config.round_id else []

    @property
    def round_id(self) -> Optional[str]:
        """The first followed round (the pinned round, or the oldest running round)."""

        return self._round_ids[0] if self._round_ids else None

    @property
    def round_ids(self) -> List[str]:
        return list(self._round_ids)

    def refresh_round(self) -> Optional[str]:
        """Follow every running round the Arena reports; a new round is verified before any claim.

        Rounds overlap, so a runner started without ``--round-id`` asks
        ``/arena/v1/current`` at every idle poll for the rounds with work
        (executing or scoring), adopts each new round's configuration once,
        and drops rounds that ended. The daily rounds roll over without a
        restart.
        """

        if self._pinned:
            return self.round_id
        config = self._config
        current = config.api.current()
        rows = current.get("running_rounds") if isinstance(current, Mapping) else None
        if not isinstance(rows, list):
            row = current.get("round") if isinstance(current, Mapping) else None
            rows = [row] if isinstance(row, Mapping) else []
        wanted = []
        for row in rows:
            if isinstance(row, Mapping) and row.get("round_id") and str(row.get("status") or "") in WORKING_STATUSES:
                wanted.append(str(row["round_id"]))
        for round_id in wanted:
            if round_id not in self._round_ids:
                configuration = dict(config.api.round(round_id).get("configuration") or {})
                config.adopt_round(dict(configuration, round_id=round_id))
        self._round_ids = wanted
        return self.round_id

    def claim_one(self, round_id: Optional[str] = None) -> Dict[str, Any]:
        config = self._config
        round_id = round_id or self.round_id
        if round_id is None:
            return {"status": "no_open_round"}
        envelope = contracts.build_signed_request(
            scope=contracts.SCOPE_CLAIM,
            round_id=round_id,
            hotkey=config.identity.hotkey,
            body={"declared_parallelism": config.max_parallel_runs},
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
        for round_id in list(self._round_ids):
            # Oldest round first: its deadline is nearer. Each round is claimed
            # until it has nothing to lease or the local slots are full.
            while taken < max_claims:
                if not self._slots.acquire(blocking=False):
                    break
                try:
                    response = self.claim_one(round_id)
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
