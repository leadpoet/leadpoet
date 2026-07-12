"""Measured AF_UNIX bridge from a runsc model sandbox to the provider broker."""

from __future__ import annotations

import base64
import json
from pathlib import Path
import socket
import threading
from typing import Any, Dict, Mapping

from gateway.tee.provider_client_v2 import BrokeredProviderTransportV2
from leadpoet_canonical.attested_v2 import canonical_json


SANDBOX_PROVIDER_SCHEMA_VERSION = "leadpoet.sandbox_provider_rpc.v2"
MAX_SANDBOX_PROVIDER_FRAME_BYTES = 64 * 1024 * 1024


class SandboxProviderSocketV2Error(RuntimeError):
    """A sandbox provider request is malformed or outside its measured job."""


def _recv_exact(connection: Any, size: int) -> bytes:
    output = bytearray()
    while len(output) < size:
        chunk = connection.recv(size - len(output))
        if not chunk:
            raise SandboxProviderSocketV2Error("sandbox provider frame is incomplete")
        output.extend(chunk)
    return bytes(output)


def _read_frame(connection: Any) -> Dict[str, Any]:
    size = int.from_bytes(_recv_exact(connection, 4), "big")
    if size < 2 or size > MAX_SANDBOX_PROVIDER_FRAME_BYTES:
        raise SandboxProviderSocketV2Error("sandbox provider frame is outside limit")
    try:
        value = json.loads(_recv_exact(connection, size).decode("utf-8"))
    except Exception as exc:
        raise SandboxProviderSocketV2Error("sandbox provider frame is invalid JSON") from exc
    if not isinstance(value, dict):
        raise SandboxProviderSocketV2Error("sandbox provider request must be an object")
    return value


def _write_frame(connection: Any, value: Mapping[str, Any]) -> None:
    encoded = canonical_json(dict(value)).encode("utf-8")
    if len(encoded) < 2 or len(encoded) > MAX_SANDBOX_PROVIDER_FRAME_BYTES:
        raise SandboxProviderSocketV2Error("sandbox provider response is outside limit")
    connection.sendall(len(encoded).to_bytes(4, "big") + encoded)


def _normalize_request(value: Mapping[str, Any]) -> Dict[str, Any]:
    fields = {
        "schema_version",
        "method",
        "url",
        "headers",
        "body_b64",
        "timeout_ms",
    }
    if set(value) != fields or value.get("schema_version") != SANDBOX_PROVIDER_SCHEMA_VERSION:
        raise SandboxProviderSocketV2Error("sandbox provider request fields are invalid")
    headers = value.get("headers")
    if not isinstance(headers, Mapping):
        raise SandboxProviderSocketV2Error("sandbox provider headers are invalid")
    try:
        body = base64.b64decode(str(value.get("body_b64") or ""), validate=True)
    except Exception as exc:
        raise SandboxProviderSocketV2Error("sandbox provider body is invalid") from exc
    timeout_ms = value.get("timeout_ms")
    if not isinstance(timeout_ms, int) or timeout_ms <= 0:
        raise SandboxProviderSocketV2Error("sandbox provider timeout is invalid")
    return {
        "method": str(value.get("method") or "").upper(),
        "url": str(value.get("url") or ""),
        "headers": {str(name): str(item) for name, item in headers.items()},
        "body": body,
        "timeout_ms": timeout_ms,
    }


class SandboxProviderSocketServerV2:
    """Serve one job-scoped provider RPC socket with shared retry ordinals."""

    def __init__(
        self,
        *,
        socket_path: Path,
        transport: BrokeredProviderTransportV2,
        execution_scope: Any,
    ) -> None:
        self.socket_path = Path(socket_path)
        self._transport = transport
        self._execution_scope = execution_scope
        self._listener = None
        self._thread = None
        self._stop = threading.Event()

    @property
    def running(self) -> bool:
        return bool(self._thread and self._thread.is_alive() and self._listener)

    def start(self) -> None:
        if self.running:
            return
        self.socket_path.parent.mkdir(parents=True, exist_ok=True)
        self.socket_path.unlink(missing_ok=True)
        listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        listener.bind(str(self.socket_path))
        self.socket_path.chmod(0o600)
        listener.listen(32)
        self._listener = listener
        self._thread = threading.Thread(
            target=self._accept_loop,
            name="leadpoet-sandbox-provider-v2",
            daemon=True,
        )
        self._thread.start()

    def close(self) -> None:
        self._stop.set()
        if self._listener is not None:
            try:
                self._listener.close()
            except Exception:
                pass
        self._listener = None
        self.socket_path.unlink(missing_ok=True)

    def _accept_loop(self) -> None:
        while not self._stop.is_set():
            try:
                connection, _ = self._listener.accept()
            except Exception:
                return
            threading.Thread(
                target=self._handle,
                args=(connection,),
                name="leadpoet-sandbox-provider-request-v2",
                daemon=True,
            ).start()

    def _handle(self, connection: Any) -> None:
        try:
            request = _normalize_request(_read_frame(connection))
            with self._transport.activate_scope(self._execution_scope):
                result = self._transport.execute_http(**request)
            _write_frame(connection, {"result": result})
        except Exception as exc:
            try:
                _write_frame(
                    connection,
                    {
                        "status": "error",
                        "error_code": "sandbox_provider_%s"
                        % type(exc).__name__.lower()[:80],
                    },
                )
            except Exception:
                pass
        finally:
            try:
                connection.close()
            except Exception:
                pass
