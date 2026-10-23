"""Atomic, complete Arena output checkpoints available to submitted harnesses.

The model chooses when to call ``write``. The Arena host only observes complete
output bytes that it can validate before the signed execution deadline.
"""

from __future__ import annotations

import json
import base64
import hashlib
import codecs
import re
import os
import socket
import tempfile
import time
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlsplit


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
TRAINING_SCHEMA_VERSION = "leadpoet.lab_arena.agent_observation.v1"
TRAINING_CONTROL = "training_observation"
MAX_TRAINING_FRAME_BYTES = 8 * 1024 * 1024
MAX_TRAINING_WEB_BODY_BYTES = 6 * 1024 * 1024
MAX_TRAINING_BINARY_BODY_BYTES = 5 * 1024 * 1024
MAX_TRAINING_STATE_BYTES = 65_536
TRAINING_SOCKET_TIMEOUT_SECONDS = 1.0
_TRAINING_FAILURE = threading.local()


def _training_failure_reason() -> str | None:
    """Return only the bounded reason for this thread's latest capture call."""
    return getattr(_TRAINING_FAILURE, "reason", None)


class QuotaUnavailable(RuntimeError):
    """The passive quota snapshot could not be read safely."""


def validate_web_capture_metadata(value: Any, body: str | None = None) -> dict[str, Any]:
    """Validate reported runtime context; it is not an origin attestation."""
    if (not isinstance(value, dict) or set(value) != {
            "source", "complete", "encoding", "received_body_sha256", "received_body_bytes"}
            or value["source"] != "runtime_http_v1"
            or type(value["complete"]) is not bool
            or not isinstance(value["encoding"], str) or not 1 <= len(value["encoding"]) <= 64
            or any(not (c.isascii() and (c.isalnum() or c in "-_.")) for c in value["encoding"])
            or not isinstance(value["received_body_sha256"], str)
            or len(value["received_body_sha256"]) != 71
            or not value["received_body_sha256"].startswith("sha256:")
            or any(c not in "0123456789abcdef" for c in value["received_body_sha256"][7:])
            or type(value["received_body_bytes"]) is not int
            or not 0 <= value["received_body_bytes"] <= MAX_TRAINING_WEB_BODY_BYTES):
        raise ValueError("web capture metadata is invalid")
    codec = codecs.lookup(value["encoding"])
    if not getattr(codec, "_is_text_encoding", False):
        raise ValueError("web capture encoding is invalid")
    if body is not None:
        received = body.encode(value["encoding"], errors="strict")
        if (len(received) != value["received_body_bytes"]
                or "sha256:" + hashlib.sha256(received).hexdigest() != value["received_body_sha256"]):
            raise ValueError("web capture bytes do not match")
    return dict(value)


def validate_binary_capture_metadata(value: Any, body: bytes | None = None) -> dict[str, Any]:
    """Validate consumed binary bytes without treating them as trainable text."""
    if (not isinstance(value, dict) or set(value) != {
            "source", "complete", "received_body_sha256", "received_body_bytes"}
            or value["source"] != "runtime_http_v1"
            or type(value["complete"]) is not bool
            or not isinstance(value["received_body_sha256"], str)
            or not re.fullmatch(r"sha256:[0-9a-f]{64}", value["received_body_sha256"])
            or type(value["received_body_bytes"]) is not int
            or not 0 <= value["received_body_bytes"] <= MAX_TRAINING_BINARY_BODY_BYTES):
        raise ValueError("binary capture metadata is invalid")
    if body is not None and (len(body) != value["received_body_bytes"]
            or "sha256:" + hashlib.sha256(body).hexdigest() != value["received_body_sha256"]):
        raise ValueError("binary capture bytes do not match")
    return dict(value)


def validate_http_response_metadata(data: Any) -> dict[str, Any]:
    """Validate header-only evidence without inventing a body read receipt."""
    if (not isinstance(data, dict) or set(data) != {
            "url", "status", "content_type", "redirect_url", "retrieved_at", "source", "body_unconsumed"}
            or data["source"] != "runtime_http_v1" or data["body_unconsumed"] is not True
            or type(data["status"]) is not int or not 100 <= data["status"] <= 599
            or not isinstance(data["content_type"], str) or len(data["content_type"]) > 256
            or not isinstance(data["retrieved_at"], str) or len(data["retrieved_at"]) > 40):
        raise ValueError("http response metadata is invalid")
    try:
        if datetime.fromisoformat(data["retrieved_at"].replace("Z", "+00:00")).tzinfo is None:
            raise ValueError("timestamp missing timezone")
        for key in ("url", "redirect_url"):
            value = data[key]
            if key == "redirect_url" and value is None:
                continue
            if (not isinstance(value, str) or len(value) > 2000
                    or any(ord(character) < 32 or ord(character) == 127 for character in value)):
                raise ValueError("url invalid")
            parts = urlsplit(value)
            if parts.scheme not in {"http", "https"} or not parts.hostname:
                raise ValueError("url invalid")
            if key == "redirect_url" and (parts.username is not None or parts.password is not None
                                          or parts.query or parts.fragment):
                raise ValueError("redirect target contains private components")
    except (ValueError, TypeError):
        raise ValueError("http response metadata is invalid") from None
    return dict(data)


def validate_training_frame(value: Any) -> dict[str, Any]:
    """Validate capture reports; run identity comes from the host."""
    if (not isinstance(value, Mapping) or set(value) not in ({
            "schema_version", "control", "kind", "depends_on", "data"}, {
            "schema_version", "control", "kind", "depends_on", "data", "retry_token"})
            or value.get("schema_version") != TRAINING_SCHEMA_VERSION
            or value.get("control") != TRAINING_CONTROL
            or value.get("kind") not in {"web_fetch", "web_fetch_binary", "http_response_metadata", "response_received", "state_snapshot", "capture_failure"}
            or not isinstance(value.get("depends_on"), list)
            or len(value["depends_on"]) > 32
            or any(not isinstance(item, str) or len(item) > 96 for item in value["depends_on"])
            or len(set(value["depends_on"])) != len(value["depends_on"])
            or not isinstance(value.get("data"), dict)):
        raise ValueError("training frame is invalid")
    if "retry_token" in value and (not isinstance(value["retry_token"], str)
            or not re.fullmatch(r"[0-9a-f]{32}", value["retry_token"])):
        raise ValueError("training frame is invalid")
    data = value["data"]
    kind = value["kind"]
    if kind == "web_fetch":
        fields = {"url", "final_url", "retrieved_at", "status", "content_type", "body"}
        if (set(data) not in (fields, fields | {"capture"})
                or any(not isinstance(data[key], str) for key in ("url", "final_url", "retrieved_at", "content_type", "body"))
                or type(data["status"]) is not int or not 100 <= data["status"] <= 599
                or len(data["body"].encode("utf-8")) > MAX_TRAINING_WEB_BODY_BYTES):
            raise ValueError("training frame is invalid")
        if "capture" in data:
            validate_web_capture_metadata(data["capture"], data["body"])
    elif kind == "web_fetch_binary":
        fields = {"url", "final_url", "retrieved_at", "status", "content_type", "body_b64", "capture"}
        if (set(data) != fields or any(not isinstance(data[key], str)
                for key in ("url", "final_url", "retrieved_at", "content_type", "body_b64"))
                or type(data["status"]) is not int or not 100 <= data["status"] <= 599
                or len(data["body_b64"]) > 4 * ((MAX_TRAINING_BINARY_BODY_BYTES + 2) // 3)):
            raise ValueError("training frame is invalid")
        try:
            binary = base64.b64decode(data["body_b64"], validate=True)
            if base64.b64encode(binary).decode("ascii") != data["body_b64"]:
                raise ValueError("noncanonical base64")
            validate_binary_capture_metadata(data["capture"], binary)
        except (ValueError, UnicodeError, TypeError):
            raise ValueError("training frame is invalid") from None
    elif kind == "http_response_metadata":
        validate_http_response_metadata(data)
    elif kind == "response_received":
        if (set(data) != {"call_identity", "status", "body_sha256", "body_bytes"}
                or not isinstance(data["call_identity"], str) or not isinstance(data["body_sha256"], str)
                or type(data["status"]) is not int or not 100 <= data["status"] <= 599
                or type(data["body_bytes"]) is not int or data["body_bytes"] < 0):
            raise ValueError("training frame is invalid")
    elif kind == "state_snapshot" and (set(data) != {"name", "value"} or not isinstance(data["name"], str)
          or not 1 <= len(data["name"]) <= 80
          or len(json.dumps(data["value"], ensure_ascii=False, allow_nan=False).encode("utf-8")) > MAX_TRAINING_STATE_BYTES):
        raise ValueError("training frame is invalid")
    elif kind == "capture_failure" and (set(data) != {"category", "reason"}
          or data["category"] not in {"web_fetch", "web_fetch_binary", "http_response_metadata", "response_received", "state_snapshot"}
          or data["reason"] not in {"size_limit", "invalid_value"}):
        raise ValueError("training frame is invalid")
    if len(json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8")) > MAX_TRAINING_FRAME_BYTES:
        raise ValueError("training frame is invalid")
    return dict(value)


def _record_training(kind: str, data: dict[str, Any], depends_on: list[str] | None,
                     *, retry_ambiguous: bool = False, retry_token: str | None = None) -> int | None:
    _TRAINING_FAILURE.reason = None
    frame = {"schema_version": TRAINING_SCHEMA_VERSION, "control": TRAINING_CONTROL,
             "kind": kind, "depends_on": depends_on, "data": data}
    if retry_ambiguous or retry_token is not None:
        frame["retry_token"] = retry_token or uuid.uuid4().hex
    try:
        frame["depends_on"] = list(depends_on or [])
        normalized = validate_training_frame(frame)
        encoded = json.dumps(normalized, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    except Exception:
        if kind != "capture_failure":
            reason = "invalid_value"
            try:
                if len(json.dumps(frame, ensure_ascii=False, allow_nan=False).encode("utf-8")) > MAX_TRAINING_FRAME_BYTES:
                    reason = "size_limit"
            except Exception:
                pass
            # Invalid dependencies must not make the failure report fail too.
            _record_training("capture_failure", {"category": kind, "reason": reason}, None)
        _TRAINING_FAILURE.reason = "invalid_frame"
        return None
    try:
        path = str(os.environ.get(WORKER_SOCKET_ENV) or "").strip()
        if not path.startswith("/"):
            _TRAINING_FAILURE.reason = "socket_unavailable"
            return None
        connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            deadline = time.monotonic() + TRAINING_SOCKET_TIMEOUT_SECONDS
            connection.settimeout(TRAINING_SOCKET_TIMEOUT_SECONDS)
            connection.connect(path)
            # The high bit reserves an extended frame for private capture.
            # Ordinary operation frames keep their existing 1 MiB early cap.
            connection.sendall((len(encoded) | 0x80000000).to_bytes(4, "big") + encoded)
            size = int.from_bytes(_recv_decision_response(connection, 4, deadline), "big")
            if size < 2 or size > MAX_QUOTA_RESPONSE_BYTES:
                _TRAINING_FAILURE.reason = "invalid_response"
                return None
            try:
                response = json.loads(_recv_decision_response(connection, size, deadline))
            except (ValueError, UnicodeError):
                _TRAINING_FAILURE.reason = "invalid_response"
                return None
            if isinstance(response, dict) and response.get("recorded") is False:
                _TRAINING_FAILURE.reason = "host_rejected"
                return None
            result = response.get("observation_id") if isinstance(response, dict) and response.get("recorded") is True else None
            if type(result) is int and result >= 0:
                return result
            _TRAINING_FAILURE.reason = "invalid_response"
            return None
        finally:
            connection.close()
    except TimeoutError:
        _TRAINING_FAILURE.reason = "timeout"
        if retry_ambiguous:
            return _record_training(kind, data, depends_on, retry_token=frame["retry_token"])
        return None
    except Exception:
        _TRAINING_FAILURE.reason = "transport_error"
        if retry_ambiguous:
            return _record_training(kind, data, depends_on, retry_token=frame["retry_token"])
        return None


def record_web_fetch(*, url: str, final_url: str, status: int, content_type: str,
                     body: str, retrieved_at: str | None = None,
                     depends_on: list[str] | None = None,
                     capture: dict[str, Any] | None = None,
                     _retry_ambiguous: bool = False) -> int | None:
    """Report exact text returned by a cooperative direct-web fetch."""
    try:
        body_size = len(body.encode("utf-8")) if isinstance(body, str) else -1
    except UnicodeError:
        body_size = -1
    if body_size < 0 or body_size > MAX_TRAINING_WEB_BODY_BYTES:
        reason = "size_limit" if body_size > MAX_TRAINING_WEB_BODY_BYTES else "invalid_value"
        _record_training("capture_failure", {"category": "web_fetch", "reason": reason}, depends_on)
        _TRAINING_FAILURE.reason = "invalid_frame"
        return None
    data = {"url": url, "final_url": final_url, "status": status,
        "content_type": content_type, "body": body,
        "retrieved_at": retrieved_at or datetime.now(timezone.utc).isoformat()}
    if capture is not None:
        data["capture"] = capture
    return _record_training("web_fetch", data, depends_on, retry_ambiguous=_retry_ambiguous)


def record_web_fetch_binary(*, url: str, final_url: str, status: int, content_type: str,
                            body: bytes, retrieved_at: str | None = None,
                            capture: dict[str, Any], depends_on: list[str] | None = None,
                            _retry_ambiguous: bool = False) -> int | None:
    """Report already consumed bytes to the private run artifact."""
    if not isinstance(body, bytes) or len(body) > MAX_TRAINING_BINARY_BODY_BYTES:
        _record_training("capture_failure", {"category": "web_fetch_binary", "reason": "size_limit"}, depends_on)
        _TRAINING_FAILURE.reason = "invalid_frame"
        return None
    return _record_training("web_fetch_binary", {
        "url": url, "final_url": final_url, "status": status, "content_type": content_type,
        "body_b64": base64.b64encode(body).decode("ascii"),
        "retrieved_at": retrieved_at or datetime.now(timezone.utc).isoformat(),
        "capture": capture}, depends_on, retry_ambiguous=_retry_ambiguous)


def record_http_response_metadata(*, url: str, status: int, content_type: str,
                                  redirect_url: str | None, retrieved_at: str,
                                  source: str, body_unconsumed: bool,
                                  depends_on: list[str] | None = None,
                                  _retry_ambiguous: bool = False) -> int | None:
    """Report already available response headers; do not claim consumed bytes."""
    return _record_training("http_response_metadata", {
        "url": url, "status": status, "content_type": content_type,
        "redirect_url": redirect_url, "retrieved_at": retrieved_at,
        "source": source, "body_unconsumed": body_unconsumed}, depends_on,
        retry_ambiguous=_retry_ambiguous)


def record_response_received(*, call_identity: str, status: int, body: bytes | None = None,
                             body_sha256: str | None = None, body_bytes: int | None = None,
                             depends_on: list[str] | None = None,
                             _retry_ambiguous: bool = False) -> int | None:
    """Report that agent code obtained a broker response; the host checks its digest."""
    if body is not None:
        if body_sha256 is not None or body_bytes is not None:
            _record_training("capture_failure", {"category": "response_received", "reason": "invalid_value"}, depends_on)
            _TRAINING_FAILURE.reason = "invalid_frame"
            return None
        try:
            raw = bytes(body)
        except (TypeError, ValueError, RecursionError):
            _record_training("capture_failure", {"category": "response_received", "reason": "invalid_value"}, depends_on)
            _TRAINING_FAILURE.reason = "invalid_frame"
            return None
        body_sha256 = "sha256:" + hashlib.sha256(raw).hexdigest()
        body_bytes = len(raw)
    elif body_sha256 is None or body_bytes is None:
        _record_training("capture_failure", {"category": "response_received", "reason": "invalid_value"}, depends_on)
        _TRAINING_FAILURE.reason = "invalid_frame"
        return None
    return _record_training("response_received", {"call_identity": call_identity, "status": status,
        "body_sha256": body_sha256, "body_bytes": body_bytes}, depends_on,
        retry_ambiguous=_retry_ambiguous)


def record_state_snapshot(name: str, value: Any, *, depends_on: list[str] | None = None) -> int | None:
    """Report only state the agent explicitly selected for training capture."""
    try:
        if len(json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8")) > MAX_TRAINING_STATE_BYTES:
            _record_training("capture_failure", {"category": "state_snapshot", "reason": "size_limit"}, depends_on)
            _TRAINING_FAILURE.reason = "invalid_frame"
            return None
    except (TypeError, ValueError, UnicodeError, RecursionError):
        _record_training("capture_failure", {"category": "state_snapshot", "reason": "invalid_value"}, depends_on)
        _TRAINING_FAILURE.reason = "invalid_frame"
        return None
    return _record_training("state_snapshot", {"name": name, "value": value}, depends_on)


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
                for item in (limit, used, inflight)
            )
            or used < 0
            or not (
                provider == "deepline" and limit == 0 and remaining is None
                or limit >= 1 and type(remaining) is int
                and used <= limit and remaining == limit - used
            )
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
