"""Atomic, complete Arena output checkpoints available to submitted harnesses.

The model chooses when to call ``write``. The Arena host only observes complete
output bytes that it can validate before the signed execution deadline.
"""

from __future__ import annotations

import json
import os
import socket
import tempfile
from pathlib import Path
from typing import Any, Mapping


OUTPUT_PATH = Path("/output/companies.json")
MAX_OUTPUT_BYTES = 512 * 1024
WORKER_SOCKET_ENV = "LAB_ARENA_WORKER_SOCKET"
QUOTA_CONTROL_SCHEMA_VERSION = "leadpoet.lab_arena.control_frame.v1"
QUOTA_SNAPSHOT_SCHEMA_VERSION = "leadpoet.lab_arena.quota_snapshot.v1"
QUOTA_CONTROL_FRAME = {
    "schema_version": QUOTA_CONTROL_SCHEMA_VERSION,
    "control": "quota_usage",
}
QUOTA_PROVIDERS = ("scrapingdog", "deepline", "openrouter")
MAX_QUOTA_RESPONSE_BYTES = 4096
QUOTA_SOCKET_TIMEOUT_SECONDS = 35.0


class QuotaUnavailable(RuntimeError):
    """The passive quota snapshot could not be read safely."""


def _recv_exact(connection: socket.socket, size: int) -> bytes:
    output = bytearray()
    while len(output) < size:
        chunk = connection.recv(min(4096, size - len(output)))
        if not chunk:
            raise QuotaUnavailable("quota unavailable")
        output.extend(chunk)
    return bytes(output)


def validate_quota_snapshot(value: Any) -> dict[str, Any]:
    """Validate the exact counter-only response shared by host and harness."""

    if not isinstance(value, Mapping) or set(value) != {
        "schema_version",
        "providers",
    }:
        raise QuotaUnavailable("quota unavailable")
    if value.get("schema_version") != QUOTA_SNAPSHOT_SCHEMA_VERSION:
        raise QuotaUnavailable("quota unavailable")
    providers = value.get("providers")
    if not isinstance(providers, Mapping) or set(providers) != set(
        QUOTA_PROVIDERS
    ):
        raise QuotaUnavailable("quota unavailable")
    normalized: dict[str, Any] = {
        "schema_version": QUOTA_SNAPSHOT_SCHEMA_VERSION,
        "providers": {},
    }
    for provider in QUOTA_PROVIDERS:
        counters = providers.get(provider)
        if not isinstance(counters, Mapping) or set(counters) != {
            "limit",
            "used",
            "remaining",
            "inflight",
        }:
            raise QuotaUnavailable("quota unavailable")
        limit = counters.get("limit")
        used = counters.get("used")
        remaining = counters.get("remaining")
        inflight = counters.get("inflight")
        if (
            any(
                isinstance(item, bool) or not isinstance(item, int)
                for item in (limit, used, remaining, inflight)
            )
            or limit < 1
            or used < 0
            or used > limit
            or remaining != limit - used
            or inflight < 0
            or inflight > used
        ):
            raise QuotaUnavailable("quota unavailable")
        normalized["providers"][provider] = {
            "limit": limit,
            "used": used,
            "remaining": remaining,
            "inflight": inflight,
        }
    return normalized


def quota_usage() -> dict[str, Any]:
    """Read one cached point-in-time quota snapshot from the run's worker."""

    path = str(os.environ.get(WORKER_SOCKET_ENV) or "").strip()
    if not path.startswith("/"):
        raise QuotaUnavailable("quota unavailable")
    encoded = json.dumps(
        QUOTA_CONTROL_FRAME, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        connection.settimeout(QUOTA_SOCKET_TIMEOUT_SECONDS)
        try:
            connection.connect(path)
            connection.sendall(len(encoded).to_bytes(4, "big") + encoded)
            size = int.from_bytes(_recv_exact(connection, 4), "big")
            if size < 2 or size > MAX_QUOTA_RESPONSE_BYTES:
                raise QuotaUnavailable("quota unavailable")
            raw = _recv_exact(connection, size)
        except OSError:
            raise QuotaUnavailable("quota unavailable") from None
    finally:
        try:
            connection.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        connection.close()
    try:
        response = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        raise QuotaUnavailable("quota unavailable") from None
    if response == {"error": "quota_unavailable"}:
        raise QuotaUnavailable("quota unavailable")
    return validate_quota_snapshot(response)


def write(companies: list[dict[str, Any]], *, output_path: Path = OUTPUT_PATH) -> None:
    """Publish one complete ``{"companies": [...]}`` document atomically."""

    if not isinstance(companies, list) or any(not isinstance(row, dict) for row in companies):
        raise ValueError("companies must be a list of company objects")
    payload = json.dumps(
        {"companies": companies}, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    if len(payload) > MAX_OUTPUT_BYTES:
        raise ValueError("checkpoint exceeds Arena output limit")
    output_path = Path(output_path)
    descriptor, name = tempfile.mkstemp(prefix=".arena-checkpoint-", dir=output_path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
        os.replace(name, output_path)
    finally:
        try:
            os.unlink(name)
        except FileNotFoundError:
            pass
