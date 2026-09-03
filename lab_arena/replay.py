"""Replay recomputation of a scoring assignment from its recorded judge responses.

Validators do the judge's work; the Arena keeps score authority by re-running
the same judge entrypoint against the responses the broker recorded for that
scoring run. Every judge call crossed the broker, which stored the sanitized
reply per call identity together with the request hash it was made under, so
the replay answers each judge request from that record and never calls a
provider. A breakdown list that differs from the validator's means the
validator did not report what the recorded responses imply, and the replayed
list is the one the Arena scores.

The replay runs the entrypoint as a subprocess with the shim installed
through ``sitecustomize`` and its worker socket pointed at a replay server,
so the service process itself is never patched.
"""

from __future__ import annotations

import base64
import json
import os
import shutil
import socketserver
import subprocess
import sys
import tempfile
import threading
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from lab_arena import broker as broker_module
from lab_arena import contracts, operations, runtime, shim

REPLAY_TIMEOUT_SECONDS = 900
REPLAY_ERROR_CODE = "call_refused"  # a request with no recorded response looks like a refused call to the judge
_SITECUSTOMIZE = "from lab_arena import shim\nshim.install()\n"


class ReplayError(RuntimeError):
    """The replay could not run; it says nothing about the validator's result."""


def recorded_responses(ledger_entries: Sequence[Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Map each call's request hash to its recorded outcome: a settled reply or the refusal it met."""

    heads: Dict[str, Mapping[str, Any]] = {}
    reservations: Dict[str, Mapping[str, Any]] = {}
    for entry in ledger_entries:
        identity = entry.get("call_identity")
        if not identity:
            continue
        heads[identity] = entry
        if entry.get("entry_kind") in ("reservation", "refusal"):
            reservations.setdefault(identity, entry)
    responses: Dict[str, Dict[str, Any]] = {}
    for identity, head in heads.items():
        reservation = reservations.get(identity, head)
        request_hash = ((reservation.get("entry_doc") or {}).get("request_hash")) or ((reservation.get("entry_doc") or {}).get("call") or {}).get("request_hash")
        if not request_hash:
            continue
        kind = head.get("entry_kind")
        if kind == "settlement" and isinstance(head.get("terminal_response"), Mapping):
            responses[request_hash] = {"kind": "response", "terminal": dict(head["terminal_response"])}
        elif kind == "refusal":
            responses[request_hash] = {"kind": "error", "code": "budget_refused"}
        else:
            responses[request_hash] = {"kind": "error", "code": "provider_unavailable"}
    return responses


class ReplayServer:
    """Answers judge frames from recorded responses over the worker socket protocol."""

    def __init__(self, socket_path: Path, responses: Mapping[str, Mapping[str, Any]]) -> None:
        self._path = Path(socket_path)
        self._responses = dict(responses)
        self._server: Optional[socketserver.ThreadingUnixStreamServer] = None
        self._thread: Optional[threading.Thread] = None
        self.lock = threading.Lock()
        self.served: list = []
        self.misses: list = []

    def handle_frame(self, raw: bytes) -> bytes:
        try:
            operation_id, parameters, _timeout = shim.decode_operation_frame(raw)
            normalized = broker_module.normalized_request(operation_id, parameters)
        except (shim.OperationFrameError, operations.OperationError, broker_module.BrokerError):
            return shim.encode_worker_error("invalid_request")
        request_hash = contracts.document_hash(normalized)
        record = self._responses.get(request_hash)
        with self.lock:
            if record is None:
                self.misses.append({"operation_id": operation_id, "request_hash": request_hash})
            else:
                self.served.append(request_hash)
        if record is None:
            return shim.encode_worker_error(REPLAY_ERROR_CODE)
        if record["kind"] == "error":
            return shim.encode_worker_error(str(record["code"]))
        terminal = record["terminal"]
        try:
            status, headers, body = int(terminal["status"]), dict(terminal.get("headers") or {}), base64.b64decode(str(terminal["body_b64"]), validate=True)
        except (KeyError, TypeError, ValueError):
            return shim.encode_worker_error("provider_unavailable")
        return shim.encode_worker_response(status, headers, body)

    def start(self) -> None:
        server_self = self

        class Handler(socketserver.BaseRequestHandler):
            def handle(self) -> None:
                connection = self.request
                connection.settimeout(120)
                try:
                    size = int.from_bytes(_recv_exact(connection, 4), "big")
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
        self._thread = threading.Thread(target=self._server.serve_forever, name="lab-arena-replay-socket", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        if self._path.exists():
            self._path.unlink()


def _recv_exact(connection: Any, size: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < size:
        chunk = connection.recv(size - len(chunks))
        if not chunk:
            raise OSError("connection closed")
        chunks.extend(chunk)
    return bytes(chunks)


def default_entry_command() -> list:
    return [sys.executable, "-m", "lab_arena.scorer_entrypoint"]


def replay_work_item(
    *,
    input_document: Mapping[str, Any],
    ledger_entries: Sequence[Mapping[str, Any]],
    work_dir: Path,
    entry_command: Optional[Sequence[str]] = None,
    timeout_seconds: int = REPLAY_TIMEOUT_SECONDS,
    socket_root: str = "/tmp",
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Run the judge entrypoint against recorded responses; return (scoring output, replay report)."""

    responses = recorded_responses(ledger_entries)
    run_dir = Path(tempfile.mkdtemp(prefix="replay-", dir=str(work_dir)))
    socket_dir = Path(tempfile.mkdtemp(prefix="lr", dir=socket_root))
    socket_path = socket_dir / runtime.SANDBOX_SOCKET_NAME
    server = ReplayServer(socket_path, responses)
    try:
        (run_dir / "sitecustomize.py").write_text(_SITECUSTOMIZE, encoding="utf-8")
        input_path = run_dir / "input.json"
        output_path = run_dir / "output.json"
        input_path.write_text(json.dumps(dict(input_document), sort_keys=True), encoding="utf-8")
        repo_root = str(Path(__file__).resolve().parent.parent)
        environment = {
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "HOME": str(run_dir),
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
            "TZ": "UTC",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONHASHSEED": "0",
            "PYTHONPATH": str(run_dir) + os.pathsep + repo_root,
            "LAB_ARENA_INPUT_PATH": str(input_path),
            "LAB_ARENA_OUTPUT_PATH": str(output_path),
            shim.WORKER_SOCKET_ENV: str(socket_path),
            shim.TRUSTED_SCORER_ENV: "1",
            "LAB_ARENA_EVALUATION_DATE": str(input_document.get("evaluation_date") or ""),
        }
        environment.update(runtime.PROVIDER_BASE_URLS)
        server.start()
        try:
            completed = subprocess.run(list(entry_command or default_entry_command()), cwd=str(run_dir), env=environment, capture_output=True, timeout=timeout_seconds, check=False)
        except subprocess.TimeoutExpired:
            raise ReplayError("replay timed out after %d seconds" % timeout_seconds) from None
        if completed.returncode != 0 or not output_path.exists():
            raise ReplayError("replay entrypoint failed with exit code %s" % completed.returncode)
        from lab_arena import scoring

        output = scoring.scoring_output_from_bytes(output_path.read_bytes())
        report = {"served": len(server.served), "misses": list(server.misses), "recorded": len(responses)}
        return output, report
    finally:
        server.stop()
        shutil.rmtree(run_dir, ignore_errors=True)
        shutil.rmtree(socket_dir, ignore_errors=True)


__all__ = ["ReplayError", "ReplayServer", "recorded_responses", "replay_work_item", "default_entry_command", "REPLAY_ERROR_CODE"]
