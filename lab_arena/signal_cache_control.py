"""Bounded scorer-only control over the existing worker socket.

This is judgment coordination, not a provider call. It has no provider action
sequence, quota debit, or synthetic provider receipt.
"""
from __future__ import annotations

import asyncio
import json
import socket
import time
from typing import Any, Mapping

SCHEMA = "leadpoet.lab_arena.signal_judgment_control.v1"
POLICY = "signal_judgment_v1"
MAX_BYTES = 1_048_576
REQUEST_FIELDS = {"operation", "company_index", "signal_index", "verified_company_identity", "verified_funding_stage_context"}


def validate_request(value: Any) -> dict:
    if not isinstance(value, Mapping):
        raise ValueError("invalid signal cache request")
    expected = REQUEST_FIELDS | ({"reference", "judgment"} if value.get("operation") == "complete" else set())
    if set(value) != expected or value.get("operation") not in {"claim", "complete"}:
        raise ValueError("invalid signal cache operation")
    if any(type(value.get(key)) is not int or not 0 <= value[key] < 100 for key in ("company_index", "signal_index")):
        raise ValueError("invalid signal cache position")
    if not isinstance(value.get("verified_company_identity"), Mapping) or not isinstance(value.get("verified_funding_stage_context"), list):
        raise ValueError("invalid signal cache context")
    from lab_arena import contracts
    encoded = contracts.canonical_json(value).encode("utf-8")
    if len(encoded) > MAX_BYTES:
        raise ValueError("signal cache request too large")
    return json.loads(encoded)


def _exchange_once(request: Mapping[str, Any]) -> dict:
    from lab_arena import contracts, shim
    from qualification.scoring.signal_judgment_cache import SignalJudgmentCacheError
    if not shim.trusted_scorer_mode():
        raise SignalJudgmentCacheError("signal cache requires scorer mode")
    encoded = contracts.canonical_json({"schema_version": SCHEMA, "request": validate_request(request)}).encode("utf-8")
    if len(encoded) > MAX_BYTES:
        raise SignalJudgmentCacheError("signal cache frame too large")
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
        connection.settimeout(35.0)
        connection.connect(shim.worker_socket_path())
        connection.sendall(len(encoded).to_bytes(4, "big") + encoded)
        size = int.from_bytes(shim._recv_exact(connection, 4), "big")
        if not 2 <= size <= MAX_BYTES:
            raise SignalJudgmentCacheError("signal cache response too large")
        result = json.loads(shim._recv_exact(connection, size))
    if not isinstance(result, dict):
        raise SignalJudgmentCacheError("invalid signal cache response")
    return result


async def exchange(request: Mapping[str, Any]) -> dict:
    """Retry the same control request only; never replay a provider operation."""
    from qualification.scoring.signal_judgment_cache import SignalJudgmentCacheError
    deadline = time.monotonic() + 120.0
    while True:
        try:
            result = await asyncio.to_thread(_exchange_once, request)
        except (OSError, ValueError) as exc:
            raise SignalJudgmentCacheError("signal cache transport unavailable") from exc
        if result.get("status") not in {"busy", "pending"}:
            if result.get("status") not in {"hit", "miss", "stored", "existing", "released"}:
                raise SignalJudgmentCacheError("signal cache request unavailable")
            return result
        if time.monotonic() >= deadline:
            raise SignalJudgmentCacheError("signal cache wait incomplete")
        await asyncio.sleep(2.0)
