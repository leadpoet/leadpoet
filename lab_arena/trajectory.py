"""Bounded, redacted Arena runtime and provider trajectory events.

Events are private diagnostics.  The caller supplies only an id, kind, time,
and content.  Run identity is attached by the service and database from the
active lease.
"""

from __future__ import annotations

import base64
import json
import math
import re
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Mapping, Optional, Sequence


MAX_EVENTS_PER_REQUEST = 32
MAX_REQUEST_BYTES = 64 * 1024
MAX_EVENT_BYTES = 8 * 1024
MAX_STRING_CHARS = 4096
MAX_DEPTH = 6
MAX_KEYS = 256
MAX_EVENTS_PER_RUN = 10_000
MAX_RUNTIME_EVENTS_PER_RUN = 512

_KIND_RE = re.compile(r"^[a-z][a-z0-9]*(?:[._-][a-z0-9]+){0,7}$")
_SENSITIVE_KEY_RE = re.compile(
    r"^(?:authorization|proxy_authorization|api_?key|x_api_key|lease(?:_token)?|"
    r"password|passwd|secret|client_secret|access_token|refresh_token|cookie|set_cookie)$",
    re.IGNORECASE,
)
_BEARER_RE = re.compile(r"(?i)\b(?:bearer|basic)\s+[A-Za-z0-9._~+/=-]{8,}")
_KEY_VALUE_RE = re.compile(
    r"(?i)\b(api[-_ ]?key|authorization|password|secret|access[-_ ]?token|"
    r"refresh[-_ ]?token|lease[-_ ]?token)\s*[:=]\s*([^\s,;]{4,})"
)
_QUOTED_KEY_VALUE_RE = re.compile(
    r"(?i)([\"'](?:api[-_ ]?key|authorization|password|secret|access[-_ ]?token|"
    r"refresh[-_ ]?token|lease[-_ ]?token)[\"']\s*:\s*)([\"'])([^\"']{4,})([\"'])"
)
_BARE_API_KEY_RE = re.compile(
    r"\b(?:sk-or-v1-[A-Za-z0-9_-]{12,}|sk-[A-Za-z0-9_-]{16,})\b"
)
_HIDDEN_REASONING_KEYS = frozenset(
    {"reasoning", "reasoning_content", "hidden_reasoning", "encrypted_content"}
)
_TOP_LEVEL_IDENTITY_FIELDS = frozenset(
    {
        "run_id", "round_id", "submission_id", "icp_position", "stage",
        "attempt", "kind_of_run", "miner_hotkey", "runner_hotkey",
        "validator_hotkey", "hotkey", "lease_token", "authorization",
    }
)
_CONTENT_IDENTITY_KEYS = frozenset(
    {
        "run_id", "round_id", "submission_id", "icp_position", "stage",
        "attempt", "miner_hotkey", "runner_hotkey", "validator_hotkey",
        "model_role", "run_kind", "icp_identifier",
    }
)


class TrajectoryError(ValueError):
    """One trajectory event violated the bounded private contract."""


def redact_text(value: str, secrets: Sequence[str] = ()) -> str:
    """Redact exact known secrets and common credential forms from one stream."""

    text = str(value).replace("\x00", "\\u0000")
    for secret in sorted(
        {item for item in secrets if isinstance(item, str) and len(item) >= 4},
        key=len,
        reverse=True,
    ):
        text = text.replace(secret, "[REDACTED]")
    text = _QUOTED_KEY_VALUE_RE.sub(
        lambda match: "%s%s[REDACTED]%s" % (
            match.group(1), match.group(2), match.group(4)
        ),
        text,
    )
    text = _BEARER_RE.sub("[REDACTED]", text)
    text = _KEY_VALUE_RE.sub(lambda match: "%s=[REDACTED]" % match.group(1), text)
    text = _BARE_API_KEY_RE.sub("[REDACTED]", text)
    return text


def _sanitize(
    value: Any,
    *,
    depth: int,
    state: Dict[str, int],
    secrets: Sequence[str] = (),
) -> Any:
    if depth > MAX_DEPTH:
        return "[TRUNCATED_DEPTH]"
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, str):
        text = redact_text(value, secrets)
        if len(text) > MAX_STRING_CHARS:
            return text[:MAX_STRING_CHARS] + "[TRUNCATED]"
        return text
    if isinstance(value, Mapping):
        result: Dict[str, Any] = {}
        for raw_key, item in value.items():
            if state["keys"] >= MAX_KEYS:
                result["_truncated_keys"] = True
                break
            key = str(raw_key).replace("\x00", "\\u0000")[:128]
            state["keys"] += 1
            normalized = key.lower().replace("-", "_")
            if normalized in _CONTENT_IDENTITY_KEYS:
                result[key] = "[DERIVED_BY_GATEWAY]"
            elif _SENSITIVE_KEY_RE.fullmatch(normalized) or normalized in _HIDDEN_REASONING_KEYS:
                result[key] = "[REDACTED]"
            else:
                result[key] = _sanitize(
                    item, depth=depth + 1, state=state, secrets=secrets
                )
        return result
    if isinstance(value, (list, tuple)):
        items = list(value[:128])
        result = [
            _sanitize(item, depth=depth + 1, state=state, secrets=secrets)
            for item in items
        ]
        if len(value) > len(items):
            result.append("[TRUNCATED_ITEMS]")
        return result
    return redact_text(str(value), secrets)[:MAX_STRING_CHARS]


def sanitize_content(
    value: Any, *, secrets: Sequence[str] = ()
) -> Dict[str, Any]:
    """Return a bounded JSON object with credentials and hidden reasoning removed."""

    if not isinstance(value, Mapping):
        raise TrajectoryError("trajectory content must be an object")
    sanitized = _sanitize(value, depth=0, state={"keys": 0}, secrets=secrets)
    if not isinstance(sanitized, dict):  # pragma: no cover - type guard
        raise TrajectoryError("trajectory content must be an object")
    return sanitized


def _utc_timestamp(value: Optional[Any] = None) -> str:
    if value is None:
        parsed = datetime.now(timezone.utc)
    elif isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise TrajectoryError("trajectory occurred_at is invalid") from exc
    else:
        raise TrajectoryError("trajectory occurred_at is invalid")
    if parsed.tzinfo is None:
        raise TrajectoryError("trajectory occurred_at must include a timezone")
    return parsed.astimezone(timezone.utc).isoformat(timespec="milliseconds").replace(
        "+00:00", "Z"
    )


def event(
    kind: str,
    content: Mapping[str, Any],
    *,
    occurred_at: Optional[Any] = None,
) -> Dict[str, Any]:
    """Create one client event with a UUID and normalized UTC timestamp."""

    document = {
        "event_id": str(uuid.uuid4()),
        "kind": kind,
        "occurred_at": _utc_timestamp(occurred_at),
        "content": sanitize_content(content),
    }
    return validate_event(document)


def validate_event(
    value: Any, *, secrets: Sequence[str] = ()
) -> Dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TrajectoryError("trajectory event must be an object")
    forbidden = set(value) & _TOP_LEVEL_IDENTITY_FIELDS
    if forbidden:
        raise TrajectoryError("trajectory event contains caller identity")
    if set(value) != {"event_id", "kind", "occurred_at", "content"}:
        raise TrajectoryError("trajectory event fields are invalid")
    try:
        event_id = str(uuid.UUID(str(value["event_id"])))
    except (AttributeError, TypeError, ValueError) as exc:
        raise TrajectoryError("trajectory event_id is invalid") from exc
    kind = value["kind"]
    if (
        not isinstance(kind, str)
        or _KIND_RE.fullmatch(kind) is None
        or not kind.startswith(("runtime.", "provider."))
    ):
        raise TrajectoryError("trajectory kind is invalid")
    document = {
        "event_id": event_id,
        "kind": kind,
        "occurred_at": _utc_timestamp(value["occurred_at"]),
        "content": sanitize_content(value["content"], secrets=secrets),
    }
    encoded = json.dumps(document, sort_keys=True, separators=(",", ":")).encode("utf-8")
    if len(encoded) > MAX_EVENT_BYTES:
        raise TrajectoryError("trajectory event is too large")
    return document


def validate_batch(
    value: Any, *, secrets: Sequence[str] = ()
) -> Sequence[Dict[str, Any]]:
    if not isinstance(value, Mapping) or set(value) != {"events"}:
        raise TrajectoryError("trajectory body fields are invalid")
    events = value["events"]
    if not isinstance(events, list) or not 1 <= len(events) <= MAX_EVENTS_PER_REQUEST:
        raise TrajectoryError("trajectory event count is invalid")
    normalized = [validate_event(item, secrets=secrets) for item in events]
    encoded = json.dumps(
        {"events": normalized}, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    if len(encoded) > MAX_REQUEST_BYTES:
        raise TrajectoryError("trajectory request is too large")
    return normalized


def provider_request_content(frame: Mapping[str, Any]) -> Dict[str, Any]:
    operation_id = str(frame.get("operation_id") or "")
    parameters = frame.get("parameters")
    projected: Dict[str, Any] = {
        "operation_id": operation_id,
        "action_sequence": frame.get("action_sequence"),
        "timeout_ms": frame.get("timeout_ms"),
    }
    if isinstance(parameters, Mapping):
        projected["parameter_keys"] = sorted(str(key)[:64] for key in parameters)[:64]
        projected["parameters"] = _provider_request_projection(parameters)
        model = parameters.get("model")
        if isinstance(model, str):
            projected["model"] = model
    return _fit_content(
        projected,
        {
            "operation_id": operation_id,
            "action_sequence": frame.get("action_sequence"),
            "timeout_ms": frame.get("timeout_ms"),
            "parameter_count": len(parameters) if isinstance(parameters, Mapping) else 0,
        },
    )


def provider_response_content(
    result: Mapping[str, Any], *, elapsed_ms: int
) -> Dict[str, Any]:
    call = result.get("call") if isinstance(result.get("call"), Mapping) else {}
    body_b64 = result.get("body_b64")
    projected: Dict[str, Any] = {
        "http_status": result.get("status"),
        "elapsed_ms": max(0, int(elapsed_ms)),
        "response_bytes_b64": len(body_b64) if isinstance(body_b64, str) else 0,
        "call": call,
    }
    # Small JSON responses retain a bounded, redacted view of useful result
    # fields.  Large provider bodies expose only size metadata; decoding them
    # here would duplicate unbounded model output in gateway memory.
    if isinstance(body_b64, str) and len(body_b64) <= 350_000:
        try:
            raw = base64.b64decode(body_b64, validate=True)
            if len(raw) > 256 * 1024:
                raise ValueError("provider response projection limit")
            decoded = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, ValueError):
            decoded = None
        if decoded is not None:
            projected["response"] = _provider_response_projection(decoded)
    return _fit_content(
        projected,
        {
            "http_status": result.get("status"),
            "elapsed_ms": max(0, int(elapsed_ms)),
            "response_bytes_b64": len(body_b64) if isinstance(body_b64, str) else 0,
            "call": call,
        },
    )


def _provider_request_projection(parameters: Mapping[str, Any]) -> Dict[str, Any]:
    projected: Dict[str, Any] = {}
    for raw_key, value in list(parameters.items())[:48]:
        key = str(raw_key)
        if key in {"messages", "input"} and isinstance(value, list):
            projected[key] = {
                "count": len(value),
                "latest": value[-3:],
            }
        elif key == "tools" and isinstance(value, list):
            projected[key] = {
                "count": len(value),
                "items": value[:12],
            }
        elif isinstance(value, list):
            projected[key] = {
                "count": len(value),
                "items": value[:8],
            }
        else:
            projected[key] = value
    if len(parameters) > len(projected):
        projected["_truncated_keys"] = True
    return _bounded_projection(projected)


def _provider_response_projection(value: Any) -> Any:
    if not isinstance(value, Mapping):
        return _bounded_projection(value)
    projected: Dict[str, Any] = {"keys": sorted(str(key)[:64] for key in value)[:64]}
    for key in (
        "id", "object", "status", "model", "error", "usage",
        "output_text", "result", "answer",
    ):
        if key in value:
            projected[key] = value[key]
    for key in ("output", "choices", "results", "data", "messages"):
        items = value.get(key)
        if isinstance(items, list):
            projected[key] = {
                "count": len(items),
                "latest": items[-3:],
            }
    return _bounded_projection(projected)


def _fit_content(
    projected: Mapping[str, Any], fallback: Mapping[str, Any]
) -> Dict[str, Any]:
    content = sanitize_content(projected)
    encoded = json.dumps(
        content, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return content if len(encoded) <= 6000 else sanitize_content(fallback)


def _bounded_projection(value: Any) -> Any:
    """Keep useful provider shape and excerpts within one event's budget."""

    remaining = {"characters": 2048, "keys": 0}

    def project(item: Any, depth: int = 0) -> Any:
        if depth > 4:
            return "[TRUNCATED_DEPTH]"
        if item is None or isinstance(item, (bool, int)):
            return item
        if isinstance(item, float):
            return item if math.isfinite(item) else None
        if isinstance(item, str):
            allowed = min(512, remaining["characters"])
            text = redact_text(item)
            projected_text = _truncate_for_json(text, allowed)
            remaining["characters"] -= min(
                allowed, len(json.dumps(projected_text, ensure_ascii=True))
            )
            return projected_text
        if isinstance(item, Mapping):
            result: Dict[str, Any] = {}
            for raw_key, child in item.items():
                if remaining["keys"] >= 64 or len(result) >= 24:
                    result["_truncated_keys"] = True
                    break
                key = _truncate_for_json(
                    str(raw_key).replace("\x00", "\\u0000"), 64
                )
                normalized = key.lower().replace("-", "_")
                remaining["keys"] += 1
                if normalized in _CONTENT_IDENTITY_KEYS:
                    result[key] = "[DERIVED_BY_GATEWAY]"
                elif (
                    _SENSITIVE_KEY_RE.fullmatch(normalized)
                    or normalized in _HIDDEN_REASONING_KEYS
                ):
                    result[key] = "[REDACTED]"
                else:
                    result[key] = project(child, depth + 1)
            return result
        if isinstance(item, (list, tuple)):
            result = [project(child, depth + 1) for child in list(item)[:12]]
            if len(item) > len(result):
                result.append("[TRUNCATED_ITEMS]")
            return result
        return project(str(item), depth)

    return project(value)


def _truncate_for_json(value: str, budget: int) -> str:
    if budget <= 2:
        return ""
    if len(json.dumps(value, ensure_ascii=True)) <= budget:
        return value
    suffix = "[TRUNCATED]"
    low, high = 0, len(value)
    while low < high:
        middle = (low + high + 1) // 2
        candidate = value[:middle] + suffix
        if len(json.dumps(candidate, ensure_ascii=True)) <= budget:
            low = middle
        else:
            high = middle - 1
    return value[:low] + suffix if low else suffix[: max(0, budget - 2)]


__all__ = [
    "MAX_EVENTS_PER_REQUEST",
    "MAX_REQUEST_BYTES",
    "MAX_EVENTS_PER_RUN",
    "MAX_RUNTIME_EVENTS_PER_RUN",
    "TrajectoryError",
    "event",
    "provider_request_content",
    "provider_response_content",
    "redact_text",
    "sanitize_content",
    "validate_batch",
    "validate_event",
]
