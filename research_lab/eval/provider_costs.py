"""Research Lab provider cost accounting helpers.

This module is intentionally provider-boundary focused: it never stores raw
prompts, raw responses, API keys, or sensitive query text. Callers pass only
provider responses at the host proxy boundary, and exported events contain
redacted request fingerprints plus numeric cost metadata.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
import base64
import json
import os
import threading
import urllib.parse
from typing import Any, Mapping, Sequence

from research_lab.canonical import sha256_json


DEFAULT_PROVIDER_COST_CAP_USD_PER_ICP = Decimal("0.50")
DEFAULT_SCRAPINGDOG_COST_PER_CREDIT_USD = Decimal("0.00005")
DEFAULT_UNKNOWN_ENDPOINT_POLICY = "fail_closed"

PROVIDER_COST_HEADER_PREFIX = "X-Research-Lab-Provider-Cost-"
PROVIDER_COST_SCOPE_HEADER = "X-Research-Lab-Cost-Scope"
PROVIDER_COST_CAP_HEADER = "X-Research-Lab-Cost-Cap-Usd"

SECRET_MARKERS = (
    "sk-or-",
    "sb_secret_",
    "aws_secret_access_key",
    "openrouter_api_key",
    "openrouter_management_key",
    "scrapingdog_api_key",
    "exa_api_key",
    "service_role",
    "raw_secret",
    "api_key=",
    "authorization:",
)

SCRAPINGDOG_ENDPOINT_CREDITS: dict[str, int] = {
    "/profile": 100,
    "/profile/post": 5,
    "/google/ai": 10,
    "/google/ai_mode": 10,
    "/youtube/transcripts": 5,
    "/youtube": 5,
    "/tiktok/profile": 5,
    "/linkedinjobs": 5,
    "/instagram/profile": 15,
}


def decimal_from_env(name: str, default: Decimal) -> Decimal:
    raw = str(os.getenv(name) or "").strip()
    if not raw:
        return default
    try:
        value = Decimal(raw)
    except InvalidOperation:
        return default
    if value < 0:
        return default
    return value


def decimal_to_float(value: Decimal) -> float:
    return float(value.quantize(Decimal("0.00000001"), rounding=ROUND_HALF_UP))


def safe_decimal(value: Any) -> Decimal | None:
    if value is None:
        return None
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None
    if parsed < 0:
        return None
    return parsed


def contains_secret_material(value: Any) -> bool:
    try:
        text = json.dumps(value, sort_keys=True, default=str)
    except Exception:
        text = str(value)
    lowered = text.lower()
    return any(marker in lowered for marker in SECRET_MARKERS)


def redacted_endpoint(provider: str, upstream_url: str) -> str:
    try:
        split = urllib.parse.urlsplit(str(upstream_url or ""))
    except Exception:
        return ""
    path = split.path or "/"
    if provider == "sd":
        return _scrapingdog_endpoint_key(path)
    if provider == "or":
        return path[:160]
    if provider == "exa":
        return path[:160]
    return path[:160]


def _json_body(body: bytes | str | None) -> Any:
    if body in (None, b"", ""):
        return None
    try:
        if isinstance(body, bytes):
            text = body.decode("utf-8", "replace")
        else:
            text = str(body)
        return json.loads(text)
    except Exception:
        return None


def _iter_json_values(value: Any, keys: set[str]) -> Sequence[Any]:
    found: list[Any] = []
    wanted = {str(key) for key in keys}

    def walk(node: Any) -> None:
        if isinstance(node, Mapping):
            for key, item in node.items():
                if str(key) in wanted:
                    found.append(item)
                walk(item)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(value)
    return found


def extract_exa_cost_dollars(body: bytes | str | None) -> Decimal | None:
    parsed = _json_body(body)
    if parsed is None:
        return None
    for value in _iter_json_values(parsed, {"costDollars", "cost_dollars", "cost"}):
        parsed_cost = safe_decimal(value)
        if parsed_cost is not None:
            return parsed_cost
        if isinstance(value, Mapping):
            for nested_key in ("total", "usd", "amount"):
                parsed_cost = safe_decimal(value.get(nested_key))
                if parsed_cost is not None:
                    return parsed_cost
    return None


def extract_openrouter_cost_dollars(body: bytes | str | None) -> tuple[Decimal | None, dict[str, Any]]:
    parsed = _json_body(body)
    metadata: dict[str, Any] = {}
    if not isinstance(parsed, Mapping):
        return None, metadata
    usage = parsed.get("usage") if isinstance(parsed.get("usage"), Mapping) else {}
    if isinstance(usage, Mapping):
        for source_key, target_key in (
            ("prompt_tokens", "prompt_tokens"),
            ("completion_tokens", "completion_tokens"),
            ("total_tokens", "total_tokens"),
        ):
            try:
                if usage.get(source_key) is not None:
                    metadata[target_key] = int(usage[source_key])
            except Exception:
                pass
        for key in ("cost", "cost_usd", "cost_dollars"):
            cost = safe_decimal(usage.get(key))
            if cost is not None:
                return cost, metadata
    for key in ("cost", "cost_usd", "cost_dollars"):
        cost = safe_decimal(parsed.get(key))
        if cost is not None:
            return cost, metadata
    for source_key, target_key in (
        ("tokens_prompt", "prompt_tokens"),
        ("native_tokens_prompt", "prompt_tokens"),
        ("tokens_completion", "completion_tokens"),
        ("native_tokens_completion", "completion_tokens"),
    ):
        for value in _iter_json_values(parsed, {source_key}):
            try:
                metadata.setdefault(target_key, int(value))
            except Exception:
                pass
            break
    for value in _iter_json_values(parsed, {"cost", "cost_usd", "cost_dollars", "total_cost", "total_cost_usd"}):
        cost = safe_decimal(value)
        if cost is not None:
            return cost, metadata
    return None, metadata


def openrouter_generation_id(body: bytes | str | None) -> str:
    parsed = _json_body(body)
    if not isinstance(parsed, Mapping):
        return ""
    for key in ("id", "generation_id"):
        value = str(parsed.get(key) or "").strip()
        if value:
            return value[:160]
    return ""


def _scrapingdog_endpoint_key(path: str) -> str:
    normalized = "/" + str(path or "").strip().lstrip("/")
    normalized = normalized.rstrip("/") or "/"
    # API docs use /google/ai, but candidate code may spell the mode route a
    # few equivalent ways. Normalize only the allowlisted Google AI forms.
    if normalized in {"/google/ai", "/google/ai_mode", "/google-ai"}:
        return "/google/ai"
    return normalized


def scrapingdog_credits_for_path(path: str) -> int | None:
    return SCRAPINGDOG_ENDPOINT_CREDITS.get(_scrapingdog_endpoint_key(path))


@dataclass
class ProviderCostEstimate:
    provider: str
    endpoint: str
    model: str = ""
    billable: bool = False
    cost_usd: Decimal = Decimal("0")
    cost_source: str = "not_billable"
    credits: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    generation_id: str = ""
    tracking_failed: bool = False
    tracking_reason: str = ""

    def to_doc(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "endpoint": self.endpoint,
            "model": self.model,
            "billable": self.billable,
            "cost_usd": decimal_to_float(self.cost_usd),
            "cost_source": self.cost_source,
            "credits": self.credits,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "generation_id": self.generation_id,
            "tracking_failed": self.tracking_failed,
            "tracking_reason": self.tracking_reason,
        }


def estimate_provider_cost(
    *,
    provider: str,
    upstream_url: str,
    status: int,
    response_body: bytes | None,
    request_body: bytes | None,
    scrapingdog_credit_price_usd: Decimal,
) -> ProviderCostEstimate:
    endpoint = redacted_endpoint(provider, upstream_url)
    if status >= 400 or status <= 0:
        return ProviderCostEstimate(
            provider=provider,
            endpoint=endpoint,
            billable=False,
            cost_source="provider_failure_zero_cost",
        )
    if provider == "sd":
        try:
            path = urllib.parse.urlsplit(upstream_url).path
        except Exception:
            path = endpoint
        credits = scrapingdog_credits_for_path(path)
        if credits is None:
            return ProviderCostEstimate(
                provider=provider,
                endpoint=endpoint,
                tracking_failed=True,
                tracking_reason="unknown_scrapingdog_endpoint",
            )
        return ProviderCostEstimate(
            provider=provider,
            endpoint=endpoint,
            billable=True,
            cost_usd=scrapingdog_credit_price_usd * Decimal(credits),
            cost_source="scrapingdog_credit_map",
            credits=credits,
        )
    if provider == "exa":
        cost = extract_exa_cost_dollars(response_body)
        if cost is None:
            return ProviderCostEstimate(
                provider=provider,
                endpoint=endpoint,
                tracking_failed=True,
                tracking_reason="missing_exa_cost_dollars",
            )
        return ProviderCostEstimate(
            provider=provider,
            endpoint=endpoint,
            billable=True,
            cost_usd=cost,
            cost_source="exa_cost_dollars",
        )
    if provider == "or":
        cost, metadata = extract_openrouter_cost_dollars(response_body)
        parsed_request = _json_body(request_body)
        model = ""
        if isinstance(parsed_request, Mapping):
            model = str(parsed_request.get("model") or "")[:160]
        if cost is None:
            return ProviderCostEstimate(
                provider=provider,
                endpoint=endpoint,
                model=model,
                generation_id=openrouter_generation_id(response_body),
                tracking_failed=True,
                tracking_reason="missing_openrouter_cost",
            )
        return ProviderCostEstimate(
            provider=provider,
            endpoint=endpoint,
            model=model,
            billable=True,
            cost_usd=cost,
            cost_source="openrouter_response_usage",
            prompt_tokens=int(metadata.get("prompt_tokens") or 0),
            completion_tokens=int(metadata.get("completion_tokens") or 0),
            generation_id=openrouter_generation_id(response_body),
        )
    return ProviderCostEstimate(
        provider=provider,
        endpoint=endpoint,
        tracking_failed=True,
        tracking_reason="unknown_provider",
    )


@dataclass
class ProviderCostEvent:
    scope: str
    provider: str
    endpoint: str
    request_fingerprint: str
    evidence: str
    status_code: int = 0
    billable: bool = False
    cost_usd: Decimal = Decimal("0")
    cost_source: str = "not_billable"
    credits: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    model: str = ""
    cap_usd: Decimal = DEFAULT_PROVIDER_COST_CAP_USD_PER_ICP
    spent_before_usd: Decimal = Decimal("0")
    spent_after_usd: Decimal = Decimal("0")
    cap_blocked: bool = False
    cap_exceeded_after_success: bool = False
    tracking_failed: bool = False
    tracking_reason: str = ""
    generation_id: str = ""
    extra_doc: dict[str, Any] = field(default_factory=dict)

    def to_headers(self) -> dict[str, str]:
        doc = self.to_doc()
        return {
            "X-Research-Lab-Provider-Cost-Event": base64.b64encode(
                json.dumps(doc, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).decode("ascii")
        }

    def to_doc(self) -> dict[str, Any]:
        doc = {
            "schema_version": "1.0",
            "scope": self.scope,
            "provider": self.provider,
            "endpoint": self.endpoint,
            "model": self.model,
            "request_fingerprint": self.request_fingerprint,
            "evidence": self.evidence,
            "status_code": int(self.status_code),
            "billable": bool(self.billable),
            "cost_usd": decimal_to_float(self.cost_usd),
            "cost_source": self.cost_source,
            "credits": int(self.credits),
            "prompt_tokens": int(self.prompt_tokens),
            "completion_tokens": int(self.completion_tokens),
            "cap_usd": decimal_to_float(self.cap_usd),
            "spent_before_usd": decimal_to_float(self.spent_before_usd),
            "spent_after_usd": decimal_to_float(self.spent_after_usd),
            "cap_blocked": bool(self.cap_blocked),
            "cap_exceeded_after_success": bool(self.cap_exceeded_after_success),
            "tracking_failed": bool(self.tracking_failed),
            "tracking_reason": self.tracking_reason,
            "generation_id": self.generation_id,
        }
        for key, value in self.extra_doc.items():
            if key not in doc:
                doc[key] = value
        if contains_secret_material(doc):
            raise ValueError("provider cost event contains secret-looking material")
        doc["event_hash"] = sha256_json({k: v for k, v in doc.items() if k != "event_hash"})
        return doc


class ProviderCostLedger:
    def __init__(self, *, scope: str, cap_usd: Decimal) -> None:
        self.scope = scope or "unscoped"
        self.cap_usd = cap_usd
        self._lock = threading.Lock()
        self._spent = Decimal("0")
        self._events: list[dict[str, Any]] = []
        self._blocked = False

    @property
    def spent_usd(self) -> Decimal:
        with self._lock:
            return self._spent

    def already_at_cap(self) -> bool:
        with self._lock:
            return self._spent >= self.cap_usd

    def should_block_paid_call(self) -> bool:
        with self._lock:
            return self._blocked or self._spent >= self.cap_usd

    def cache_hit_event(self, *, provider: str, endpoint: str, request_fingerprint: str, status_code: int) -> ProviderCostEvent:
        with self._lock:
            event = ProviderCostEvent(
                scope=self.scope,
                provider=provider,
                endpoint=endpoint,
                request_fingerprint=request_fingerprint,
                evidence="hit",
                status_code=status_code,
                billable=False,
                cost_source="cache_hit_zero_cost",
                cap_usd=self.cap_usd,
                spent_before_usd=self._spent,
                spent_after_usd=self._spent,
            )
            self._events.append(event.to_doc())
            return event

    def block_event(
        self,
        *,
        provider: str,
        endpoint: str,
        request_fingerprint: str,
        reason: str,
    ) -> ProviderCostEvent:
        with self._lock:
            self._blocked = True
            event = ProviderCostEvent(
                scope=self.scope,
                provider=provider,
                endpoint=endpoint,
                request_fingerprint=request_fingerprint,
                evidence="blocked",
                billable=False,
                cost_source="blocked_before_paid_call",
                cap_usd=self.cap_usd,
                spent_before_usd=self._spent,
                spent_after_usd=self._spent,
                cap_blocked=True,
                tracking_failed=reason != "cost_cap_reached",
                tracking_reason=reason,
            )
            self._events.append(event.to_doc())
            return event

    def record_live_event(
        self,
        *,
        provider: str,
        request_fingerprint: str,
        status_code: int,
        estimate: ProviderCostEstimate,
    ) -> ProviderCostEvent:
        with self._lock:
            before = self._spent
            after = before
            if estimate.billable and not estimate.tracking_failed:
                after = before + estimate.cost_usd
                self._spent = after
            event = ProviderCostEvent(
                scope=self.scope,
                provider=provider,
                endpoint=estimate.endpoint,
                model=estimate.model,
                request_fingerprint=request_fingerprint,
                evidence="recorded",
                status_code=status_code,
                billable=estimate.billable and not estimate.tracking_failed,
                cost_usd=estimate.cost_usd if not estimate.tracking_failed else Decimal("0"),
                cost_source=estimate.cost_source,
                credits=estimate.credits,
                prompt_tokens=estimate.prompt_tokens,
                completion_tokens=estimate.completion_tokens,
                generation_id=estimate.generation_id,
                cap_usd=self.cap_usd,
                spent_before_usd=before,
                spent_after_usd=after,
                cap_exceeded_after_success=bool(estimate.billable and after > self.cap_usd),
                tracking_failed=estimate.tracking_failed,
                tracking_reason=estimate.tracking_reason,
            )
            if estimate.tracking_failed:
                self._blocked = True
            self._events.append(event.to_doc())
            return event

    def summary(self) -> dict[str, Any]:
        with self._lock:
            return summarize_provider_cost_events(self._events, cap_usd=self.cap_usd)


def decode_cost_event_header(value: Any) -> dict[str, Any] | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        decoded = base64.b64decode(text).decode("utf-8")
        doc = json.loads(decoded)
    except Exception:
        return None
    if not isinstance(doc, Mapping) or contains_secret_material(doc):
        return None
    return dict(doc)


def cost_event_from_trace_entry(entry: Mapping[str, Any]) -> dict[str, Any] | None:
    doc = entry.get("provider_cost_event")
    if isinstance(doc, Mapping) and not contains_secret_material(doc):
        return dict(doc)
    return None


def summarize_provider_cost_events(events: Sequence[Mapping[str, Any]], *, cap_usd: Decimal | None = None) -> dict[str, Any]:
    total = Decimal("0")
    provider_breakdown: dict[str, dict[str, Any]] = {}
    paid_call_count = 0
    failed_call_count = 0
    cache_hit_count = 0
    blocked_count = 0
    tracking_failed_count = 0
    cap_blocked = False
    cap_exceeded_after_success = False
    effective_cap = cap_usd
    for raw in events:
        if not isinstance(raw, Mapping) or contains_secret_material(raw):
            continue
        provider = str(raw.get("provider") or "unknown")
        cost = safe_decimal(raw.get("cost_usd")) or Decimal("0")
        billable = bool(raw.get("billable"))
        total += cost if billable else Decimal("0")
        if raw.get("evidence") == "hit":
            cache_hit_count += 1
        if int(raw.get("status_code") or 0) >= 400:
            failed_call_count += 1
        if billable:
            paid_call_count += 1
            bucket = provider_breakdown.setdefault(
                provider,
                {"cost_usd": 0.0, "paid_call_count": 0, "credits": 0},
            )
            bucket["cost_usd"] = round(float(bucket["cost_usd"]) + decimal_to_float(cost), 8)
            bucket["paid_call_count"] = int(bucket["paid_call_count"]) + 1
            bucket["credits"] = int(bucket["credits"]) + int(raw.get("credits") or 0)
        if raw.get("cap_blocked"):
            cap_blocked = True
            blocked_count += 1
        if raw.get("cap_exceeded_after_success"):
            cap_exceeded_after_success = True
        if raw.get("tracking_failed"):
            tracking_failed_count += 1
        parsed_cap = safe_decimal(raw.get("cap_usd"))
        if effective_cap is None and parsed_cap is not None:
            effective_cap = parsed_cap
    return {
        "schema_version": "1.0",
        "total_cost_usd": decimal_to_float(total),
        "cap_usd": decimal_to_float(effective_cap or DEFAULT_PROVIDER_COST_CAP_USD_PER_ICP),
        "paid_call_count": paid_call_count,
        "failed_call_count": failed_call_count,
        "cache_hit_count": cache_hit_count,
        "blocked_call_count": blocked_count,
        "tracking_failed_count": tracking_failed_count,
        "cap_blocked": cap_blocked,
        "cap_exceeded_after_success": cap_exceeded_after_success,
        "provider_breakdown": dict(sorted(provider_breakdown.items())),
    }


def summarize_provider_cost_trace_entries(entries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    events = [event for entry in entries if (event := cost_event_from_trace_entry(entry))]
    return summarize_provider_cost_events(events)
