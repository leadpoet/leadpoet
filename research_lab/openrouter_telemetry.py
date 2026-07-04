"""Shared OpenRouter telemetry client (trajectoryimprovements.md P1).

Every production OpenRouter call must be captured or explicitly classified as
intentionally uncaptured. This module is the one wrapper the direct legacy
call sites migrate onto (or instrument through), providing:

* request body construction with reasoning requested by default (P8) and a
  retry-without-reasoning fallback recorded as ``reasoning_request_dropped``;
* HTTP / transport error capture — failed attempts are training signal;
* 402 credit-block classification (``OpenRouterCreditBlockedError``);
* cost/token extraction into a ``provider_usage`` doc;
* encrypted (SSE-KMS) S3 raw request/response trace upload with secret
  redaction, sharing the ``RESEARCH_LAB_RAW_TRACE_S3_PREFIX`` destination;
* axis-provenance stamping (P11) via ``research_lab.axis_provenance``;
* an explicit non-streaming posture (P19: streamed completions are refused
  loudly, never silently degraded);
* an embeddings capture variant recording input-hash + model + output-hash
  (P19: embeddings inputs/outputs must be reproducible for the novelty gate).

Two integration styles:

* ``call_openrouter_chat`` / ``call_openrouter_chat_async`` — the full
  transport for new/migrated sites (operator repair path first);
* ``record_openrouter_trace`` — a capture-only hook for legacy sites that
  keep their own httpx/requests transport but must stop being invisible.

Local/dev inertness matches the worker recorder: no S3 prefix → one log then
silent; the P5 capture-health preflight is what makes that loud in prod.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import threading
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence
from urllib import request as urlrequest
from urllib.error import HTTPError, URLError

from research_lab.axis_provenance import provenance_for_stage

logger = logging.getLogger(__name__)

OPENROUTER_CHAT_COMPLETIONS_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_EMBEDDINGS_URL = "https://openrouter.ai/api/v1/embeddings"

RAW_TRACE_S3_PREFIX_ENV = "RESEARCH_LAB_RAW_TRACE_S3_PREFIX"
RAW_TRACE_KMS_KEY_ENV = "RESEARCH_LAB_TRACE_KMS_KEY_ID"
INCLUDE_REASONING_ENV = "RESEARCH_LAB_LLM_INCLUDE_REASONING"

_SECRET_PATTERNS = (
    re.compile(r"sk-or-[A-Za-z0-9\-_]+"),
    re.compile(r"sk-[A-Za-z0-9]{20,}"),
    re.compile(r"Bearer\s+[A-Za-z0-9\-._~+/]{16,}=*"),
)
_SECRET_KEY_NAMES = {"authorization", "api_key", "apikey", "x-api-key"}


def sha256_text(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_doc(doc: Any) -> str:
    return sha256_text(
        json.dumps(doc, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)
    )


def redact_secrets(value: Any) -> Any:
    """Recursively scrub credential-shaped material from a trace doc."""
    if isinstance(value, Mapping):
        return {
            key: ("[redacted-secret]" if str(key).lower() in _SECRET_KEY_NAMES else redact_secrets(item))
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [redact_secrets(item) for item in value]
    if isinstance(value, str):
        redacted = value
        for pattern in _SECRET_PATTERNS:
            redacted = pattern.sub("[redacted-secret]", redacted)
        return redacted
    return value


class OpenRouterTelemetryError(RuntimeError):
    """Chat call failed; carries the captured provider usage and trace ref."""

    def __init__(
        self,
        message: str,
        *,
        http_status: int | None = None,
        provider_usage: dict[str, Any] | None = None,
    ):
        super().__init__(message)
        self.http_status = http_status
        self.provider_usage = provider_usage or {}


class OpenRouterCreditBlockedError(OpenRouterTelemetryError):
    """402 / insufficient credits — resumable, not a generic failure."""


class OpenRouterStreamingUnsupportedError(ValueError):
    """P19 streaming posture: this transport is non-streaming by contract."""


@dataclass
class OpenRouterTelemetryResult:
    content: str
    decoded: dict[str, Any]
    provider_usage: dict[str, Any]
    raw_trace_ref: dict[str, str] | None = None
    reasoning_request_dropped: bool = False
    cost_microusd: int = 0
    finish_reason: str = ""
    reasoning_returned: bool = False
    extras: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# raw trace uploader (shared by the transport and the record-only hook)
# ---------------------------------------------------------------------------


class _TelemetryTraceUploader:
    """Best-effort SSE-KMS S3 uploader for telemetry raw traces.

    Keys: ``{prefix}/telemetry/{channel}/{date}/{uuid}-{purpose}.json.enc``.
    Never raises; missing prefix logs once per process then stays inert.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._executor: ThreadPoolExecutor | None = None
        self._pending: set[Future[None]] = set()
        self._warned = False
        self._disabled = False

    def capture(
        self,
        *,
        channel: str,
        purpose: str,
        payload: Mapping[str, Any],
    ) -> dict[str, str] | None:
        try:
            if self._disabled:
                return None
            prefix = str(os.getenv(RAW_TRACE_S3_PREFIX_ENV, "")).strip().rstrip("/")
            if not prefix.startswith("s3://"):
                self._warn_once(
                    f"set {RAW_TRACE_S3_PREFIX_ENV}=s3://bucket/prefix to capture "
                    "legacy-channel OpenRouter traces"
                )
                return None
            bucket, _sep, key_prefix = prefix[5:].partition("/")
            if not bucket:
                return None
            kms_key_id = str(os.getenv(RAW_TRACE_KMS_KEY_ENV, "")).strip()
            if not kms_key_id:
                self._warn_once(
                    f"set {RAW_TRACE_KMS_KEY_ENV}=alias/leadpoet-research-lab-trace-encryption "
                    "to capture legacy-channel OpenRouter traces"
                )
                return None
            safe_channel = _path_segment(channel, "channel")
            safe_purpose = _path_segment(purpose, "call")
            date_segment = datetime.now(timezone.utc).strftime("%Y%m%d")
            object_key = "/".join(
                segment
                for segment in (
                    key_prefix.strip("/"),
                    "telemetry",
                    safe_channel,
                    date_segment,
                    f"{uuid.uuid4().hex}-{safe_purpose}.json.enc",
                )
                if segment
            )
            body = json.dumps(
                redact_secrets(dict(payload)),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                default=str,
            ).encode("utf-8")
            digest = "sha256:" + hashlib.sha256(body).hexdigest()
            self._submit(bucket=bucket, object_key=object_key, body=body, kms_key_id=kms_key_id)
            return {"s3_ref": f"s3://{bucket}/{object_key}", "sha256": digest}
        except Exception as exc:  # noqa: BLE001 - capture never fails the call
            logger.warning(
                "openrouter_telemetry_capture_failed channel=%s error=%s",
                channel,
                str(exc)[:200],
            )
            return None

    def flush(self, timeout_seconds: float = 10.0) -> None:
        import time as _time

        deadline = _time.monotonic() + max(0.0, timeout_seconds)
        while True:
            with self._lock:
                pending = tuple(self._pending)
            if not pending or _time.monotonic() >= deadline:
                return
            for future in pending:
                remaining = deadline - _time.monotonic()
                if remaining <= 0:
                    return
                try:
                    future.exception(timeout=remaining)
                except Exception:  # noqa: BLE001
                    return

    def _warn_once(self, hint: str) -> None:
        with self._lock:
            if self._warned:
                return
            self._warned = True
        logger.warning("openrouter_telemetry_capture_unavailable hint=%s", hint)

    def _submit(self, *, bucket: str, object_key: str, body: bytes, kms_key_id: str) -> None:
        with self._lock:
            if self._executor is None:
                self._executor = ThreadPoolExecutor(
                    max_workers=2, thread_name_prefix="openrouter-telemetry-trace"
                )
            executor = self._executor
        future = executor.submit(self._put_object, bucket, object_key, body, kms_key_id)
        with self._lock:
            self._pending.add(future)
        future.add_done_callback(self._consume)

    def _put_object(self, bucket: str, object_key: str, body: bytes, kms_key_id: str) -> None:
        import boto3  # type: ignore

        put_kwargs: dict[str, Any] = {
            "Bucket": bucket,
            "Key": object_key,
            "Body": body,
            "ContentType": "application/json",
            "ServerSideEncryption": "aws:kms",
            "SSEKMSKeyId": kms_key_id,
        }
        boto3.client("s3").put_object(**put_kwargs)

    def _consume(self, future: "Future[None]") -> None:
        with self._lock:
            self._pending.discard(future)
        if future.cancelled():
            return
        exc = future.exception()
        if exc is None:
            return
        if isinstance(exc, ImportError):
            with self._lock:
                self._disabled = True
        logger.warning(
            "openrouter_telemetry_trace_upload_failed error=%s",
            str(exc)[:200],
        )


def _path_segment(value: str, fallback: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", str(value or "").strip()).strip("-.")
    return cleaned[:80] or fallback


_uploader = _TelemetryTraceUploader()


def _safe_capture(*, channel: str, purpose: str, payload: Mapping[str, Any]) -> dict[str, str] | None:
    """Capture-or-None: an exploding uploader must never fail the LLM call."""
    try:
        return _uploader.capture(channel=channel, purpose=purpose, payload=payload)
    except Exception:  # noqa: BLE001 - absolute safety rail
        logger.warning("openrouter_telemetry_capture_raised channel=%s", channel, exc_info=True)
        return None


def flush_telemetry_traces(timeout_seconds: float = 10.0) -> None:
    """Wait for in-flight uploads (tests / orderly teardown only)."""
    _uploader.flush(timeout_seconds)


# ---------------------------------------------------------------------------
# capture-only hook for legacy transports
# ---------------------------------------------------------------------------


def record_openrouter_trace(
    *,
    channel: str,
    purpose: str,
    request_body: Mapping[str, Any],
    response_doc: Any,
    outcome: str = "response",
    model_id: str = "",
    context_ref: str = "",
    stage: str = "",
) -> dict[str, Any]:
    """Capture one legacy-site OpenRouter exchange without changing transport.

    Returns a ``provider_usage`` doc (with ``raw_trace_ref`` when S3 capture is
    configured) the call site can attach to its own records. Never raises.
    """
    provenance = provenance_for_stage(stage or purpose)
    usage = (
        response_doc.get("usage")
        if isinstance(response_doc, Mapping) and isinstance(response_doc.get("usage"), Mapping)
        else {}
    )
    reasoning_requested = bool(
        dict(request_body).get("include_reasoning") or dict(request_body).get("reasoning")
    )
    reasoning_text = _extract_reasoning_text(response_doc)
    payload = {
        "schema_version": "1.1",
        "artifact_type": "research_lab_raw_llm_trace",
        "channel": str(channel),
        "purpose": str(purpose),
        "stage": str(stage or purpose),
        "context_ref": str(context_ref or ""),
        "outcome": str(outcome),
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "call_emitter": provenance["call_emitter"],
        "component": provenance["component"],
        "teacher_model_flag": provenance["teacher_model_flag"],
        "request": dict(request_body),
        "response": response_doc,
    }
    raw_trace_ref = _safe_capture(channel=channel, purpose=purpose, payload=payload)
    provider_usage: dict[str, Any] = {
        "provider": "openrouter",
        "channel": str(channel),
        "purpose": str(purpose),
        "call_emitter": provenance["call_emitter"],
        "teacher_model_flag": provenance["teacher_model_flag"],
        "model": str(model_id or dict(request_body).get("model") or ""),
        "outcome": str(outcome),
        "prompt_tokens": _i(usage.get("prompt_tokens")),
        "completion_tokens": _i(usage.get("completion_tokens")),
        "total_tokens": _i(usage.get("total_tokens")),
        "cost_usd": _f(usage.get("cost")),
        "reasoning_capture": {
            "requested": reasoning_requested,
            "returned": bool(reasoning_text),
            "reasoning_byte_count": len(reasoning_text.encode("utf-8")) if reasoning_text else 0,
            "reasoning_hashes": [sha256_text(reasoning_text)] if reasoning_text else [],
            "storage_state": "raw_trace_ref" if raw_trace_ref else "metadata_only",
        },
    }
    if raw_trace_ref:
        provider_usage["raw_trace_ref"] = raw_trace_ref
    return provider_usage


def record_openrouter_embeddings(
    *,
    channel: str,
    purpose: str,
    model_id: str,
    input_texts: Sequence[str],
    embeddings: Sequence[Sequence[float]] | None,
    context_ref: str = "",
) -> dict[str, Any]:
    """P19 embeddings capture decision: persist input-hash + model +
    output-hash (never the raw vectors in Supabase; the full doc goes to S3
    when capture is configured) so novelty-gate/reranker runs are reproducible.
    Never raises."""
    input_hashes = [sha256_text(str(text)) for text in input_texts]
    output_hash = (
        sha256_doc([[round(float(v), 8) for v in vec] for vec in embeddings])
        if embeddings is not None
        else ""
    )
    payload = {
        "schema_version": "1.0",
        "artifact_type": "research_lab_embeddings_trace",
        "channel": str(channel),
        "purpose": str(purpose),
        "context_ref": str(context_ref or ""),
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "model": str(model_id),
        "input_hashes": input_hashes,
        "output_hash": output_hash,
        "input_count": len(input_hashes),
    }
    raw_trace_ref = _safe_capture(
        channel=channel, purpose=f"{purpose}-embeddings", payload=payload
    )
    doc = {
        "provider": "openrouter",
        "kind": "embeddings",
        "channel": str(channel),
        "purpose": str(purpose),
        "model": str(model_id),
        "input_hashes": input_hashes,
        "output_hash": output_hash,
        "input_count": len(input_hashes),
    }
    if raw_trace_ref:
        doc["raw_trace_ref"] = raw_trace_ref
    return doc


# ---------------------------------------------------------------------------
# full transport
# ---------------------------------------------------------------------------


def include_reasoning_default() -> bool:
    """P8: production OpenRouter calls request reasoning by default."""
    return str(os.getenv(INCLUDE_REASONING_ENV, "true")).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def reasoning_request_unsupported(status: int, message: str) -> bool:
    """True when an HTTP rejection is the model refusing the reasoning field —
    the caller should retry without reasoning and record the drop (P8)."""
    if status not in {400, 404, 422}:
        return False
    lowered = (message or "").lower()
    return "reasoning" in lowered or "include_reasoning" in lowered


# Backwards-compatible private aliases (internal call sites below).
_include_reasoning_default = include_reasoning_default
_reasoning_request_unsupported = reasoning_request_unsupported


def _is_credit_block(status: int | None, message: str) -> bool:
    if status == 402:
        return True
    lowered = (message or "").lower()
    return "insufficient credits" in lowered or "payment required" in lowered


def _extract_reasoning_text(decoded: Any) -> str:
    if not isinstance(decoded, Mapping):
        return ""
    choices = decoded.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    first = choices[0] if isinstance(choices[0], Mapping) else {}
    message = first.get("message") if isinstance(first.get("message"), Mapping) else {}
    parts: list[str] = []
    for key in ("reasoning", "reasoning_content"):
        value = message.get(key)
        if isinstance(value, str) and value.strip():
            parts.append(value)
    details = message.get("reasoning_details")
    if isinstance(details, list):
        for item in details:
            if isinstance(item, Mapping) and isinstance(item.get("text"), str):
                parts.append(item["text"])
    return "\n".join(parts)


def _i(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _f(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def call_openrouter_chat(
    *,
    api_key: str,
    model_id: str,
    messages: Sequence[Mapping[str, Any]],
    channel: str,
    purpose: str,
    stage: str = "",
    context_ref: str = "",
    max_tokens: int = 1800,
    temperature: float | None = None,
    timeout_seconds: int = 90,
    response_format: Mapping[str, Any] | None = None,
    include_reasoning: bool | None = None,
    reasoning_effort: str = "",
    zdr: bool = True,
    extra_body: Mapping[str, Any] | None = None,
    opener: Any = None,
) -> OpenRouterTelemetryResult:
    """One captured, non-streaming OpenRouter chat completion.

    Reasoning is requested by default (P8); models that reject it are retried
    without and the drop is recorded (``reasoning_request_dropped=true`` +
    ``drop_error_hash``). 402s raise ``OpenRouterCreditBlockedError``. Every
    attempt — success, HTTP error, transport error — is captured to S3 when
    configured.
    """
    if not api_key:
        raise OpenRouterTelemetryError("OpenRouter api key is required")
    if not model_id:
        raise OpenRouterTelemetryError("OpenRouter model id is required")
    extra = dict(extra_body or {})
    if extra.get("stream"):
        # P19 streaming posture: this transport is non-streaming by contract;
        # the raw-trace recorder assumes complete JSON bodies.
        raise OpenRouterStreamingUnsupportedError(
            "streamed completions are not supported by the telemetry transport"
        )
    request_reasoning = (
        _include_reasoning_default() if include_reasoning is None else bool(include_reasoning)
    )
    open_fn = opener or urlrequest.urlopen

    def _body(with_reasoning: bool) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": model_id,
            "messages": [dict(message) for message in messages],
            "max_tokens": max(1, int(max_tokens or 0)),
            **extra,
        }
        if temperature is not None:
            body["temperature"] = min(2.0, max(0.0, float(temperature)))
        if response_format is not None:
            body["response_format"] = dict(response_format)
        if zdr:
            body.setdefault("provider", {"data_collection": "deny", "zdr": True})
        if with_reasoning:
            body["include_reasoning"] = True
            if reasoning_effort:
                body["reasoning"] = {"effort": str(reasoning_effort)}
        return body

    def _record(body: Mapping[str, Any], response_doc: Any, outcome: str) -> dict[str, str] | None:
        payload = {
            "schema_version": "1.1",
            "artifact_type": "research_lab_raw_llm_trace",
            "channel": str(channel),
            "purpose": str(purpose),
            "stage": str(stage or purpose),
            "context_ref": str(context_ref or ""),
            "outcome": outcome,
            "captured_at": datetime.now(timezone.utc).isoformat(),
            **{
                key: provenance[key]
                for key in ("call_emitter", "component", "teacher_model_flag")
            },
            "request": {
                "url": OPENROUTER_CHAT_COMPLETIONS_URL,
                "method": "POST",
                "body": dict(body),
            },
            "response": response_doc,
        }
        return _safe_capture(channel=channel, purpose=purpose, payload=payload)

    provenance = provenance_for_stage(stage or purpose)

    def _call_once(with_reasoning: bool) -> OpenRouterTelemetryResult:
        body = _body(with_reasoning)
        req = urlrequest.Request(
            OPENROUTER_CHAT_COMPLETIONS_URL,
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            method="POST",
        )
        try:
            with open_fn(req, timeout=max(1, int(timeout_seconds))) as response:
                decoded = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            message = exc.read().decode("utf-8", errors="replace")[:500]
            raw_ref = _record(
                body,
                {"http_status": int(exc.code), "error_excerpt": redact_secrets(message)},
                "http_error",
            )
            usage_doc = _usage_doc(
                body,
                {},
                raw_ref,
                outcome="http_error",
                reasoning_requested=with_reasoning,
            )
            error = f"OpenRouter call failed: HTTP {exc.code}: {message}"
            if with_reasoning and _reasoning_request_unsupported(int(exc.code), message):
                raise _ReasoningUnsupported(error) from exc
            if _is_credit_block(int(exc.code), message):
                raise OpenRouterCreditBlockedError(
                    error, http_status=int(exc.code), provider_usage=usage_doc
                ) from exc
            raise OpenRouterTelemetryError(
                error, http_status=int(exc.code), provider_usage=usage_doc
            ) from exc
        except URLError as exc:
            raw_ref = _record(
                body, {"error_excerpt": redact_secrets(str(exc)[:500])}, "transport_error"
            )
            raise OpenRouterTelemetryError(
                f"OpenRouter call failed: {exc}",
                provider_usage=_usage_doc(
                    body,
                    {},
                    raw_ref,
                    outcome="transport_error",
                    reasoning_requested=with_reasoning,
                ),
            ) from exc
        raw_ref = _record(body, decoded, "response")
        choices = decoded.get("choices") if isinstance(decoded, Mapping) else None
        if not isinstance(choices, list) or not choices:
            raise OpenRouterTelemetryError(
                "OpenRouter returned no choices",
                provider_usage=_usage_doc(
                    body,
                    decoded,
                    raw_ref,
                    outcome="no_choices",
                    reasoning_requested=with_reasoning,
                ),
            )
        first = choices[0] if isinstance(choices[0], Mapping) else {}
        message = first.get("message") if isinstance(first.get("message"), Mapping) else {}
        content = str(message.get("content") or "")
        usage_doc = _usage_doc(
            body,
            decoded,
            raw_ref,
            outcome="response",
            reasoning_requested=with_reasoning,
        )
        reasoning_text = _extract_reasoning_text(decoded)
        return OpenRouterTelemetryResult(
            content=content,
            decoded=decoded if isinstance(decoded, Mapping) else {},
            provider_usage=usage_doc,
            raw_trace_ref=raw_ref,
            cost_microusd=int(round(_f((decoded.get("usage") or {}).get("cost")) * 1_000_000))
            if isinstance(decoded, Mapping) and isinstance(decoded.get("usage"), Mapping)
            else 0,
            finish_reason=str(first.get("finish_reason") or ""),
            reasoning_returned=bool(reasoning_text),
        )

    def _usage_doc(
        body: Mapping[str, Any],
        decoded: Any,
        raw_ref: dict[str, str] | None,
        *,
        outcome: str,
        reasoning_requested: bool,
    ) -> dict[str, Any]:
        usage = (
            decoded.get("usage")
            if isinstance(decoded, Mapping) and isinstance(decoded.get("usage"), Mapping)
            else {}
        )
        reasoning_text = _extract_reasoning_text(decoded)
        doc: dict[str, Any] = {
            "provider": "openrouter",
            "channel": str(channel),
            "purpose": str(purpose),
            "call_emitter": provenance["call_emitter"],
            "teacher_model_flag": provenance["teacher_model_flag"],
            "model": str(model_id),
            "outcome": outcome,
            "response_id": str(decoded.get("id") or "") if isinstance(decoded, Mapping) else "",
            "prompt_tokens": _i(usage.get("prompt_tokens")),
            "completion_tokens": _i(usage.get("completion_tokens")),
            "total_tokens": _i(usage.get("total_tokens")),
            "cost_usd": _f(usage.get("cost")),
            "reasoning_capture": {
                "requested": reasoning_requested,
                "returned": bool(reasoning_text),
                "reasoning_byte_count": (
                    len(reasoning_text.encode("utf-8")) if reasoning_text else 0
                ),
                "reasoning_hashes": [sha256_text(reasoning_text)] if reasoning_text else [],
                "storage_state": "raw_trace_ref" if raw_ref else "metadata_only",
            },
        }
        if raw_ref:
            doc["raw_trace_ref"] = raw_ref
        return doc

    class _ReasoningUnsupported(Exception):
        pass

    try:
        return _call_once(request_reasoning)
    except _ReasoningUnsupported as exc:
        drop_error_hash = sha256_text(str(exc))
        logger.warning(
            "openrouter_telemetry_reasoning_unsupported channel=%s model=%s drop_error_hash=%s; "
            "retrying without reasoning",
            channel,
            model_id,
            drop_error_hash,
        )
        result = _call_once(False)
        result.reasoning_request_dropped = True
        result.provider_usage["reasoning_capture"]["request_dropped"] = True
        result.provider_usage["reasoning_capture"]["drop_error_hash"] = drop_error_hash
        return result


async def call_openrouter_chat_async(**kwargs: Any) -> OpenRouterTelemetryResult:
    import asyncio

    return await asyncio.to_thread(lambda: call_openrouter_chat(**kwargs))
