"""Atomic, complete Arena output checkpoints available to submitted harnesses.

The model chooses when to call ``write``. The Arena host only observes complete
output bytes that it can validate before the signed execution deadline.
"""

from __future__ import annotations

import json
import os
import socket
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping


OUTPUT_PATH = Path("/output/companies.json")
MAX_OUTPUT_BYTES = 512 * 1024
WORKER_SOCKET_ENV = "LAB_ARENA_WORKER_SOCKET"
QUOTA_CONTROL_SCHEMA_VERSION = "leadpoet.lab_arena.control_frame.v1"
QUOTA_SNAPSHOT_SCHEMA_VERSION = "leadpoet.lab_arena.quota_snapshot.v1"
QUOTA_COST_SNAPSHOT_SCHEMA_VERSION = "leadpoet.lab_arena.quota_snapshot.v2"
QUOTA_CONTROL_FRAME = {
    "schema_version": QUOTA_CONTROL_SCHEMA_VERSION,
    "control": "quota_usage",
}
QUOTA_COST_CONTROL_FRAME = {
    "schema_version": QUOTA_CONTROL_SCHEMA_VERSION,
    "control": "quota_usage_v2",
}
QUOTA_PROVIDERS = ("scrapingdog", "deepline", "openrouter")
MAX_QUOTA_RESPONSE_BYTES = 4096
QUOTA_SOCKET_TIMEOUT_SECONDS = 35.0
DECISION_SCHEMA_VERSION = "leadpoet.lab_arena.decision.v1"
DECISION_CONTROL = "decision"
DECISION_CHOICES = frozenset(
    {"investigate", "accept", "reject", "defer", "finish"}
)
MAX_DECISION_TEXT_CHARS = 500
MAX_DECISION_EVIDENCE_ITEMS = 5
MAX_DECISION_FRAME_BYTES = 4096
DECISION_SOCKET_TIMEOUT_SECONDS = 2.0


class QuotaUnavailable(RuntimeError):
    """The passive quota snapshot could not be read safely."""


def validate_decision_frame(value: Any) -> dict[str, Any]:
    """Validate one closed, bounded model-reported decision control frame."""

    required = {
        "schema_version",
        "control",
        "objective",
        "evidence",
        "rationale",
        "next_action",
        "decision",
    }
    if not isinstance(value, Mapping) or not required <= set(value) <= (
        required | {"candidate"}
    ):
        raise ValueError("decision frame is invalid")
    if (
        value.get("schema_version") != DECISION_SCHEMA_VERSION
        or value.get("control") != DECISION_CONTROL
        or value.get("decision") not in DECISION_CHOICES
    ):
        raise ValueError("decision frame is invalid")

    normalized = dict(value)
    for field in ("objective", "rationale", "next_action"):
        text = normalized.get(field)
        if (
            not isinstance(text, str)
            or not text.strip()
            or len(text) > MAX_DECISION_TEXT_CHARS
        ):
            raise ValueError("decision frame is invalid")
    candidate = normalized.get("candidate")
    if "candidate" in normalized and (
        not isinstance(candidate, str)
        or not candidate.strip()
        or len(candidate) > MAX_DECISION_TEXT_CHARS
    ):
        raise ValueError("decision frame is invalid")
    evidence = normalized.get("evidence")
    if (
        not isinstance(evidence, list)
        or len(evidence) > MAX_DECISION_EVIDENCE_ITEMS
        or any(
            not isinstance(item, str)
            or not item.strip()
            or len(item) > MAX_DECISION_TEXT_CHARS
            for item in evidence
        )
    ):
        raise ValueError("decision frame is invalid")
    encoded = json.dumps(
        normalized,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    if len(encoded) > MAX_DECISION_FRAME_BYTES:
        raise ValueError("decision frame is invalid")
    return normalized


def _recv_decision_response(
    connection: socket.socket, size: int, deadline: float,
) -> bytes:
    output = bytearray()
    while len(output) < size:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("decision response deadline")
        connection.settimeout(remaining)
        chunk = connection.recv(min(4096, size - len(output)))
        if not chunk:
            raise OSError("decision response closed")
        output.extend(chunk)
    return bytes(output)


def log_decision(
    *,
    objective: str,
    evidence: list[str],
    rationale: str,
    next_action: str,
    decision: str,
    candidate: str | None = None,
) -> bool:
    """Best-effort capture of one explicit model decision for private audit."""

    frame: dict[str, Any] = {
        "schema_version": DECISION_SCHEMA_VERSION,
        "control": DECISION_CONTROL,
        "objective": objective,
        "evidence": evidence,
        "rationale": rationale,
        "next_action": next_action,
        "decision": decision,
    }
    if candidate is not None:
        frame["candidate"] = candidate
    try:
        normalized = validate_decision_frame(frame)
        encoded = json.dumps(
            normalized,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
        path = str(os.environ.get(WORKER_SOCKET_ENV) or "").strip()
        if not path.startswith("/"):
            return False
        connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            deadline = time.monotonic() + DECISION_SOCKET_TIMEOUT_SECONDS
            connection.settimeout(DECISION_SOCKET_TIMEOUT_SECONDS)
            connection.connect(path)
            connection.sendall(len(encoded).to_bytes(4, "big") + encoded)
            size = int.from_bytes(
                _recv_decision_response(connection, 4, deadline), "big"
            )
            if size < 2 or size > MAX_QUOTA_RESPONSE_BYTES:
                return False
            response = json.loads(
                _recv_decision_response(connection, size, deadline).decode("utf-8")
            )
            return response == {"recorded": True}
        finally:
            try:
                connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            connection.close()
    except Exception:
        return False


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


def validate_quota_cost_snapshot(value: Any) -> dict[str, Any]:
    """Validate opt-in per-ICP execute costs, including model LLM usage."""

    if (
        not isinstance(value, Mapping)
        or set(value) != {"schema_version", "providers", "sourcing_cost"}
        or value.get("schema_version") != QUOTA_COST_SNAPSHOT_SCHEMA_VERSION
    ):
        raise QuotaUnavailable("quota unavailable")
    normalized = validate_quota_snapshot(
        {
            "schema_version": QUOTA_SNAPSHOT_SCHEMA_VERSION,
            "providers": value["providers"],
        }
    )
    cost = value["sourcing_cost"]
    fields = {
        "successful_microusd",
        "success_unresolved_microusd",
        "settled_microusd",
        "reserved_or_uncertain_microusd",
        "inflight_calls",
        "success_unresolved_calls",
        "admission_cap_microusd",
        "per_qualified_pair_cap_microusd",
    }
    if (
        not isinstance(cost, Mapping)
        or set(cost) != fields
        or any(type(cost[key]) is not int or cost[key] < 0 for key in fields)
        or cost["admission_cap_microusd"] < 1
        or cost["per_qualified_pair_cap_microusd"] < 1
        or cost["successful_microusd"] > cost["settled_microusd"]
        or cost["inflight_calls"] > cost["success_unresolved_calls"]
    ):
        raise QuotaUnavailable("quota unavailable")
    normalized["schema_version"] = QUOTA_COST_SNAPSHOT_SCHEMA_VERSION
    normalized["sourcing_cost"] = dict(cost)
    return normalized


def quota_usage(*, include_sourcing_cost: bool = False) -> dict[str, Any]:
    """Read one cached point-in-time quota snapshot from the run's worker."""

    path = str(os.environ.get(WORKER_SOCKET_ENV) or "").strip()
    if not path.startswith("/"):
        raise QuotaUnavailable("quota unavailable")
    if type(include_sourcing_cost) is not bool:
        raise QuotaUnavailable("quota unavailable")
    encoded = json.dumps(
        QUOTA_COST_CONTROL_FRAME if include_sourcing_cost else QUOTA_CONTROL_FRAME,
        sort_keys=True,
        separators=(",", ":"),
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
    return (
        validate_quota_cost_snapshot(response)
        if include_sourcing_cost
        else validate_quota_snapshot(response)
    )


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
