"""Bounded provider operations with reservation, dispatch, and settlement
(labarena.md sections 7.3, 7.4, 7.5).

The broker is the only Arena component that holds provider credentials. It
receives one validated operation frame per model action, reserves the
maximum possible cost through the ledger functions, commits the dispatch
marker, sends the request it builds itself from the closed operation table,
settles the actual cost, and returns a sanitized response. Errors returned to
the model are generic codes that never carry provider account, quota,
credential, or transport detail.
"""

from __future__ import annotations

import asyncio
import base64
from collections import deque
import contextvars
import hashlib
import hmac
import json
import logging
import math
import re
import threading
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from decimal import Decimal, ROUND_CEILING
from email.utils import parsedate_to_datetime
from typing import Any, Callable, Deque, Dict, Mapping, Optional, Protocol, Sequence, Tuple
from urllib import request as urlrequest
from urllib.error import HTTPError, URLError
from urllib.parse import quote, unquote_to_bytes, urlsplit

import httpx

from lab_arena import contracts, deepline_catalog, operations, provider_costs, scoring_provider_compat
from lab_arena import telemetry
from lab_arena.contracts import ArenaContractError
from lab_arena.store import ArenaStoreError, ArenaStoreUnavailable

PRICE_TABLE_SCHEMA_VERSION = "leadpoet.lab_arena.openrouter_price_table.v1"
JUDGMENT_CACHE_SCHEMA_VERSION = "leadpoet.lab_arena.verifier_provider_request.v1"
JUDGMENT_CACHE_BUSY_POLL_SECONDS = 1.0
OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"
OPENROUTER_GENERATION_URL = "https://openrouter.ai/api/v1/generation?id="
OPENROUTER_LUNA_RESPONSES_MODEL = "openai/gpt-5.6-luna"
OPENROUTER_GPT6_LUNA_RESPONSES_MODEL = "openai/gpt-6-luna"
OPENROUTER_REGIONAL_LUNA_RESPONSES_MODELS = frozenset(
    (OPENROUTER_LUNA_RESPONSES_MODEL, OPENROUTER_GPT6_LUNA_RESPONSES_MODEL)
)
# Luna's price ceilings use the regional endpoint price at 1.10x the model
# catalog row. Prompt-cache writes are 1.25x its ordinary prompt price. The
# bounded host policy reserves that larger input-token ceiling because Responses
# requests can carry a prompt_cache_key and OpenRouter bills cache writes in usage.cost.
OPENROUTER_LUNA_REGIONAL_PRICE_MULTIPLIER = Decimal("1.10")
OPENROUTER_LUNA_CACHE_WRITE_PRICE_MULTIPLIER = Decimal("1.25")
# The public Luna catalog applies these prices at 272k prompt tokens.  The
# runtime catalog price table contains the base row, so keep this one model's
# published tier beside its source-bound route until the catalog schema carries
# tiers.
OPENROUTER_LUNA_LONG_CONTEXT_MIN_PROMPT_TOKENS = 272_000
OPENROUTER_LUNA_LONG_CONTEXT_PROMPT_PRICE = Decimal("0.0000004")
OPENROUTER_LUNA_LONG_CONTEXT_COMPLETION_PRICE = Decimal("0.0000018")
# GPT-6 Luna has its own published 272k-token tier. These catalog prices are
# converted to the verified Azure US/EU endpoint ceiling below.
OPENROUTER_GPT6_LUNA_LONG_CONTEXT_PROMPT_PRICE = Decimal("0.0000002")
OPENROUTER_GPT6_LUNA_LONG_CONTEXT_COMPLETION_PRICE = Decimal("0.00000075")
OPENROUTER_PRICE_PER_MILLION = Decimal("1000000")
OPENROUTER_NATIVE_WEB_SEARCH_RESERVATION_USD_PER_CALL = Decimal("0.01")
DEEPLINE_BILLING_LEDGER_URL = "https://code.deepline.com/api/v2/billing/ledger"
DEEPLINE_EXACT_BILLING_URL = "https://code.deepline.com/api/v2/billing/usage?request_id="
DEEPLINE_EXECUTION_BY_KEY_URL = "https://code.deepline.com/api/v2/executions/by-key/"
DEEPLINE_BILLING_HISTORY_URL = (
    "https://code.deepline.com/api/v2/billing/usage?recent_limit=50"
)
PRICED_COMPONENTS = ("prompt", "completion", "request", "image", "web_search", "internal_reasoning")
MAX_PRICE_TABLE_BYTES = 8 * 1024 * 1024
MICROUSD = Decimal(1_000_000)
# Conservative input-token bound: no production tokenizer emits more tokens
# than characters, and every message carries a small fixed overhead.
TOKENS_PER_CHAR_BOUND = 1
REQUEST_TOKEN_OVERHEAD = 16
_DIRECT_URLOPEN = urlrequest.build_opener(urlrequest.ProxyHandler({})).open

GENERIC_ERRORS = {
    "invalid_request": 400,
    "model_not_allowed": 400,
    "budget_refused": 402,
    "lease_stale": 409,
    "call_uncertain": 409,
    "call_refused": 402,
    "provider_unavailable": 502,
    "provider_request_refused": 403,
    "broker_unavailable": 503,
    "miner_credentials_unavailable": 402,
    "miner_provider_not_configured": 400,
}

_DECIMAL_RE = re.compile(r"^-?[0-9]+(?:\.[0-9]+)?(?:[eE]-?[0-9]+)?$")
_SAFE_EXCEPTION_CLASSES = frozenset(
    {
        "ArenaContractError",
        "ArenaStoreError",
        "CompatibilityResponseError",
        "JSONDecodeError",
        "KeyError",
        "OperationError",
        "OperationResponseError",
        "OverflowError",
        "TypeError",
        "UnicodeDecodeError",
        "ValueError",
    }
)
_DEEPLINE_JOB_STATUSES = frozenset(
    {"cancelled", "completed", "failed", "in_progress", "pending", "queued", "running"}
)
_DEEPLINE_BILLING_MAX_ATTEMPTS = 24
_DEEPLINE_BILLING_POLL_SECONDS = 2.0
_OPENROUTER_BILLING_MAX_ATTEMPTS = 6
_OPENROUTER_BILLING_POLL_SECONDS = 2.0
_OPENROUTER_RESPONSE_POLICY_ERROR_CODES = {
    "content_policy_violation": "image_content_policy_violation",
    "refusal": "invalid_prompt",
}
_OPENROUTER_RESPONSE_ERROR_TYPE_STATUSES = {
    "context_length_exceeded": 400,
    "max_tokens_exceeded": 400,
    "token_limit_exceeded": 400,
    "string_too_long": 400,
    "authentication": 401,
    "permission_denied": 403,
    "payment_required": 402,
    "rate_limit_exceeded": 429,
    "provider_overloaded": 503,
    "provider_unavailable": 502,
    "invalid_request": 400,
    "invalid_prompt": 400,
    "not_found": 404,
    "precondition_failed": 412,
    "payload_too_large": 413,
    "unprocessable": 422,
    "content_policy_violation": 403,
    "refusal": 403,
    "invalid_image": 400,
    "image_too_large": 400,
    "image_too_small": 400,
    "unsupported_image_format": 400,
    "image_not_found": 404,
    "image_download_failed": 400,
    "server": 500,
    "timeout": 504,
    "unmapped": 500,
}
OPENROUTER_DELAYED_RECONCILIATION_TIMEOUT_SECONDS = 5.0
_SETTLEMENT_STORE_MAX_ATTEMPTS = 3
DEEPLINE_DELAYED_RECONCILIATION_TIMEOUT_SECONDS = 5.0
_DEEPLINE_REQUEST_ID_RE = re.compile(r"^ctx-tool-[0-9a-f]{32}$")
_DEEPLINE_EXECUTION_KEY_RE = re.compile(r"^arena:[0-9a-f]{64}$")
_DEEPLINE_NATIVE_REQUEST_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,199}$")
# This batch submission does not honor Deepline execution keys. It needs its
# own submission receipt; never treat a missing key lookup as a free request.
_DEEPLINE_UNKEYED_TOOLS = frozenset({"firecrawl_batch_scrape"})
_DEEPLINE_BILLING_REQUEST_ID_RE = re.compile(
    r"^(?:ctx-tool-[0-9a-f]{32}|[a-z0-9]{3,8}::[a-z0-9]{1,16}-[0-9]{13}-[a-f0-9]{12,64})$"
)
_TRANSPORT_ERROR_CLASSES = frozenset({
    "ConnectError", "ConnectTimeout", "ReadError", "ReadTimeout",
    "WriteError", "WriteTimeout", "PoolTimeout", "RemoteProtocolError",
    "LocalProtocolError", "ProxyError", "UnsupportedProtocol",
})
CHAMPION_CREDENTIAL_PROVIDER_ATTEMPTS = 4
_OPENROUTER_GENERATION_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,199}$")
_CREDENTIAL_FINGERPRINT_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
# Broker-internal only: the observed Firecrawl envelope was 5,271,155 bytes.
# The requested Scrapingdog-compatible response keeps its existing 2 MiB cap.
_DEEPLINE_FIRECRAWL_ENVELOPE_MAX_BYTES = 16 * 1024 * 1024
# Documented post-mortem billing identity:
# https://openrouter.ai/docs/guides/features/router-metadata#error-responses
_OPENROUTER_GENERATION_HEADER = "x-generation-id"
_RETRY_AFTER_ABSENT = object()
_MAX_RETRY_AFTER_SECONDS = 3600
OPENROUTER_SHARED_GATE_DEFAULT_MAX = 2
OPENROUTER_SHARED_GATE_MAX = 10
OPENROUTER_SHARED_GATE_MIN_PROVIDER_SECONDS = 30.0
OPENROUTER_SHARED_GATE_IDLE_SECONDS = 300.0
OPENROUTER_SHARED_GATE_MAX_IDLE_STATES = 256
OPENROUTER_SHARED_GATE_MAX_STATES = 1024
_OPENROUTER_SHARED_GATE_FALLBACK_SECONDS = (20.0, 40.0)


@dataclass
class _OpenRouterGateState:
    active: int = 0
    waiters: Deque[int] = field(default_factory=deque)
    cooldown_until: float = 0.0
    cooldown_generation: int = 0
    throttle_count: int = 0
    last_used: float = 0.0


class OpenRouterGateLease:
    """One admitted request. Release is idempotent for fail-closed paths."""

    def __init__(
        self,
        gate: "OpenRouterSharedGate",
        credential_fingerprint: str,
        ticket: int,
        cooldown_generation: int,
    ) -> None:
        self._gate = gate
        self.credential_fingerprint = credential_fingerprint
        self.ticket = ticket
        self.cooldown_generation = cooldown_generation
        self._released = False

    def release(self) -> None:
        if not self._released:
            self._released = True
            self._gate._release(self)


class OpenRouterGateCancelled(RuntimeError):
    """The queued caller disconnected or its cancellation probe failed."""


class OpenRouterSharedGate:
    """Process-shared FIFO admission and cooldown per credential fingerprint."""

    def __init__(
        self,
        max_concurrency: int = OPENROUTER_SHARED_GATE_DEFAULT_MAX,
        *,
        idle_seconds: float = OPENROUTER_SHARED_GATE_IDLE_SECONDS,
        max_idle_states: int = OPENROUTER_SHARED_GATE_MAX_IDLE_STATES,
        max_states: int = OPENROUTER_SHARED_GATE_MAX_STATES,
        fallback_seconds: Sequence[float] = _OPENROUTER_SHARED_GATE_FALLBACK_SECONDS,
    ) -> None:
        if (
            isinstance(max_concurrency, bool)
            or not isinstance(max_concurrency, int)
            or not 1 <= max_concurrency <= OPENROUTER_SHARED_GATE_MAX
        ):
            raise ValueError("OpenRouter shared concurrency must be an integer from 1 to 10")
        if (
            idle_seconds < 0
            or max_idle_states < 1
            or max_states < max_idle_states
        ):
            raise ValueError("OpenRouter shared gate cleanup bounds are invalid")
        if len(fallback_seconds) != 2 or any(float(value) < 0 for value in fallback_seconds):
            raise ValueError("OpenRouter shared gate fallback is invalid")
        self.max_concurrency = max_concurrency
        self._idle_seconds = float(idle_seconds)
        self._max_idle_states = int(max_idle_states)
        self._max_states = int(max_states)
        self._fallback_seconds = tuple(float(value) for value in fallback_seconds)
        self._condition = threading.Condition()
        self._states: Dict[str, _OpenRouterGateState] = {}
        self._next_ticket = 0

    @staticmethod
    def _cancelled(cancel_requested: Optional[Callable[[], bool]]) -> bool:
        if cancel_requested is None:
            return False
        try:
            return bool(cancel_requested())
        except Exception:
            # A broken disconnect probe must fail closed before reservation.
            return True

    def _cleanup_locked(self, now: float) -> None:
        idle = [
            (fingerprint, state)
            for fingerprint, state in self._states.items()
            if state.active == 0
            and not state.waiters
            and state.cooldown_until <= now
        ]
        for fingerprint, state in idle:
            if now - state.last_used >= self._idle_seconds:
                self._states.pop(fingerprint, None)
        idle = sorted(
            (
                (state.last_used, fingerprint)
                for fingerprint, state in self._states.items()
                if state.active == 0
                and not state.waiters
                and state.cooldown_until <= now
            )
        )
        excess = max(0, len(idle) - self._max_idle_states)
        for _, fingerprint in idle[:excess]:
            self._states.pop(fingerprint, None)

    def acquire(
        self,
        credential_fingerprint: str,
        *,
        deadline: float,
        cancel_requested: Optional[Callable[[], bool]] = None,
    ) -> Optional[OpenRouterGateLease]:
        if _CREDENTIAL_FINGERPRINT_RE.fullmatch(credential_fingerprint) is None:
            raise ValueError("OpenRouter credential fingerprint is invalid")
        # Contention telemetry only. An uncontended admission records nothing,
        # so gate span volume tracks the problem rather than the traffic.
        gate_entered_ns = time.time_ns()
        gate_entered = time.monotonic()

        def _record_gate(outcome: str) -> None:
            telemetry.record_gate(
                outcome,
                duration_ms=(time.monotonic() - gate_entered) * 1000.0,
                start_ns=gate_entered_ns,
            )

        with self._condition:
            now = time.monotonic()
            self._cleanup_locked(now)
            state = self._states.get(credential_fingerprint)
            if state is None:
                if len(self._states) >= self._max_states:
                    _record_gate("no_capacity")
                    return None
                state = _OpenRouterGateState(last_used=now)
                self._states[credential_fingerprint] = state
            self._next_ticket += 1
            ticket = self._next_ticket
            state.waiters.append(ticket)
        first_check = True
        while True:
            # The disconnect probe can cross the async/thread boundary. Never
            # run it while holding the process-wide condition shared by other
            # credentials.
            cancelled = self._cancelled(cancel_requested)
            with self._condition:
                now = time.monotonic()
                can_admit = (
                    state.waiters
                    and state.waiters[0] == ticket
                    and state.active < self.max_concurrency
                    and now >= state.cooldown_until
                )
                if cancelled:
                    try:
                        state.waiters.remove(ticket)
                    except ValueError:
                        pass
                    state.last_used = now
                    self._condition.notify_all()
                    self._cleanup_locked(now)
                    _record_gate("cancelled")
                    raise OpenRouterGateCancelled
                # A short caller timeout is valid when the gate is idle. Its
                # queue deadline can equal entry time, so allow this first
                # uncontended admission before applying the queue bound.
                if can_admit and (first_check or now < deadline):
                    state.waiters.popleft()
                    state.active += 1
                    state.last_used = now
                    lease = OpenRouterGateLease(
                        self,
                        credential_fingerprint,
                        ticket,
                        state.cooldown_generation,
                    )
                    self._condition.notify_all()
                    if not first_check:
                        _record_gate("admitted_after_wait")
                    return lease
                if now >= deadline:
                    try:
                        state.waiters.remove(ticket)
                    except ValueError:
                        pass
                    state.last_used = now
                    self._condition.notify_all()
                    self._cleanup_locked(now)
                    _record_gate("timed_out")
                    return None
                wake_at = deadline
                if (
                    state.waiters
                    and state.waiters[0] == ticket
                    and state.active < self.max_concurrency
                    and state.cooldown_until > now
                ):
                    wake_at = min(wake_at, state.cooldown_until)
                # Poll a real disconnect callback while queued.
                if cancel_requested is not None:
                    wake_at = min(wake_at, now + 0.1)
                first_check = False
                self._condition.wait(max(0.001, wake_at - now))

    def observe_throttle(
        self,
        lease: OpenRouterGateLease,
        retry_after_seconds: object = _RETRY_AFTER_ABSENT,
    ) -> bool:
        if retry_after_seconds is _RETRY_AFTER_ABSENT:
            use_fallback = True
        elif (
            type(retry_after_seconds) is int
            and 0 <= retry_after_seconds <= _MAX_RETRY_AFTER_SECONDS
        ):
            use_fallback = False
        else:
            return False
        with self._condition:
            state = self._states.get(lease.credential_fingerprint)
            if state is None:
                return False
            if use_fallback:
                index = min(state.throttle_count, len(self._fallback_seconds) - 1)
                delay = self._fallback_seconds[index]
            else:
                delay = float(retry_after_seconds)
            state.throttle_count = min(state.throttle_count + 1, 2)
            state.cooldown_generation += 1
            now = time.monotonic()
            state.cooldown_until = max(state.cooldown_until, now + delay)
            state.last_used = now
            self._condition.notify_all()
            return True

    def observe_success(self, lease: OpenRouterGateLease) -> None:
        with self._condition:
            state = self._states.get(lease.credential_fingerprint)
            if state is None:
                return
            # A request admitted before a newer throttle is stale evidence. It
            # must not reset that newer cooldown or its fallback progression.
            if lease.cooldown_generation == state.cooldown_generation:
                state.throttle_count = 0
            state.last_used = time.monotonic()

    def _release(self, lease: OpenRouterGateLease) -> None:
        with self._condition:
            state = self._states.get(lease.credential_fingerprint)
            if state is None or state.active < 1:
                raise RuntimeError("OpenRouter shared gate lease is inconsistent")
            state.active -= 1
            state.last_used = time.monotonic()
            self._condition.notify_all()


def _retry_after_seconds(headers: Mapping[str, str]) -> object:
    """Parse one bounded trusted Retry-After value for the worker only."""

    if "retry-after" not in headers:
        return _RETRY_AFTER_ABSENT
    value = headers.get("retry-after")
    if not isinstance(value, str) or not value or value != value.strip():
        return None
    if value.isdecimal():
        if len(value) > 4:
            return None
        seconds = int(value)
    else:
        try:
            retry_at = parsedate_to_datetime(value)
            if retry_at.tzinfo is None:
                return None
            seconds = max(0, math.ceil(retry_at.timestamp() - time.time()))
        except (OverflowError, TypeError, ValueError):
            return None
    return seconds if seconds <= _MAX_RETRY_AFTER_SECONDS else None


def _safe_exception_class(exc: BaseException) -> str:
    """Return only a bounded Python class label, never exception text."""

    name = type(exc).__name__
    return name if name in _SAFE_EXCEPTION_CLASSES else "Exception"


class BrokerError(RuntimeError):
    """A generic broker failure; ``code`` is one of ``GENERIC_ERRORS``."""

    def __init__(self, code: str) -> None:
        if code not in GENERIC_ERRORS:
            raise ValueError("unknown broker error code")
        super().__init__(code)
        self.code = code

    @property
    def status(self) -> int:
        return GENERIC_ERRORS[self.code]


class ProviderTransportError(RuntimeError):
    """The provider request failed at the transport layer (outcome unknown)."""

    def __init__(
        self,
        message: str,
        *,
        openrouter_generation_id: Optional[str] = None,
        deepline_job_id: Optional[str] = None,
        observed_status: Optional[int] = None,
    ) -> None:
        super().__init__(message)
        self.openrouter_generation_id = (
            openrouter_generation_id.strip()
            if isinstance(openrouter_generation_id, str)
            and _OPENROUTER_GENERATION_ID_RE.fullmatch(
                openrouter_generation_id.strip()
            )
            else None
        )
        self.deepline_job_id = (
            deepline_job_id if isinstance(deepline_job_id, str)
            and _DEEPLINE_NATIVE_REQUEST_ID_RE.fullmatch(deepline_job_id)
            else None
        )
        self.observed_status = (
            observed_status
            if isinstance(observed_status, int)
            and not isinstance(observed_status, bool)
            and 100 <= observed_status <= 599
            else None
        )


# ---------------------------------------------------------------------------
# OpenRouter price table (section 7.3)
# ---------------------------------------------------------------------------


def _price_string(value: Any, component: str) -> str:
    if value is None:
        return "0"
    text = str(value).strip()
    if not _DECIMAL_RE.match(text):
        raise ArenaContractError("price table component %s is not decimal" % component)
    amount = Decimal(text)
    if amount < 0 or not amount.is_finite():
        raise ArenaContractError("price table component %s is invalid" % component)
    return format(amount.normalize(), "f")


def validate_price_table(document: Any) -> Dict[str, Any]:
    if not isinstance(document, Mapping):
        raise ArenaContractError("price table must be an object")
    contracts.require_only_keys(document, ("schema_version", "fetched_at", "source", "models"))
    contracts.require_keys(document, ("schema_version", "fetched_at", "source", "models"))
    if document["schema_version"] != PRICE_TABLE_SCHEMA_VERSION:
        raise ArenaContractError("unsupported price table schema")
    if not isinstance(document["models"], Mapping) or not document["models"]:
        raise ArenaContractError("price table must list at least one model")
    models: Dict[str, Dict[str, str]] = {}
    for model_id, pricing in document["models"].items():
        if not isinstance(model_id, str) or not operations._MODEL_ID_RE.match(model_id):
            raise ArenaContractError("price table model id is invalid")
        if not isinstance(pricing, Mapping) or set(pricing) != set(PRICED_COMPONENTS):
            raise ArenaContractError("price table model %s must price every component" % model_id)
        models[model_id] = {component: _price_string(pricing[component], component) for component in PRICED_COMPONENTS}
    table = {
        "schema_version": PRICE_TABLE_SCHEMA_VERSION,
        "fetched_at": str(document["fetched_at"]),
        "source": str(document["source"]),
        "models": models,
    }
    return table


def price_table_from_models_response(response: Mapping[str, Any], model_ids: Optional[Sequence[str]] = None, *, fetched_at: str) -> Dict[str, Any]:
    """Build a catalog of models with usable prompt and completion prices."""

    if not isinstance(response, Mapping) or not isinstance(response.get("data"), list):
        raise ArenaContractError("models endpoint response is malformed")
    wanted = None if model_ids is None else {str(model) for model in model_ids}
    if wanted is not None and not wanted:
        raise ArenaContractError("at least one model is required")
    found: Dict[str, Dict[str, Any]] = {}
    for item in response["data"]:
        if not isinstance(item, Mapping):
            continue
        model_id = item.get("id")
        if (
            not isinstance(model_id, str)
            or operations._MODEL_ID_RE.fullmatch(model_id) is None
            or (wanted is not None and model_id not in wanted)
        ):
            continue
        pricing = item.get("pricing")
        if not isinstance(pricing, Mapping) or pricing.get("prompt") is None or pricing.get("completion") is None:
            continue
        candidate = {component: pricing.get(component, "0") for component in PRICED_COMPONENTS}
        try:
            found[model_id] = {component: _price_string(candidate[component], component) for component in PRICED_COMPONENTS}
        except ArenaContractError:
            continue
    missing = sorted((wanted or set()) - set(found))
    if wanted is not None and missing:
        raise ArenaContractError("models endpoint lacks allowed models: %s" % ", ".join(missing))
    if not found:
        raise ArenaContractError("models endpoint has no models with usable pricing")
    return validate_price_table({
        "schema_version": PRICE_TABLE_SCHEMA_VERSION,
        "fetched_at": fetched_at,
        "source": OPENROUTER_MODELS_URL,
        "models": found,
    })


def fetch_openrouter_price_table(
    model_ids: Optional[Sequence[str]] = None,
    *,
    urlopen: Optional[Callable[..., Any]] = None,
    timeout_seconds: int = 20,
    now: Optional[Callable[[], datetime]] = None,
) -> Dict[str, Any]:
    urlopen = urlopen or _DIRECT_URLOPEN
    request = urlrequest.Request(OPENROUTER_MODELS_URL, headers={"Accept": "application/json"}, method="GET")
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            raw = response.read(MAX_PRICE_TABLE_BYTES + 1)
    except HTTPError as exc:
        raise ArenaContractError("models endpoint returned HTTP %d" % exc.code) from exc
    except URLError as exc:
        raise ArenaContractError("models endpoint unreachable") from exc
    if len(raw) > MAX_PRICE_TABLE_BYTES:
        raise ArenaContractError("models endpoint response exceeds the size cap")
    try:
        decoded = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise ArenaContractError("models endpoint returned invalid JSON") from exc
    moment = (now or (lambda: datetime.now(timezone.utc)))()
    return price_table_from_models_response(decoded, model_ids, fetched_at=moment.strftime("%Y-%m-%dT%H:%M:%SZ"))


def _microusd_ceiling(usd: Decimal) -> int:
    return int((usd * MICROUSD).to_integral_value(rounding=ROUND_CEILING))


def bounded_input_tokens(parameters: Mapping[str, Any]) -> int:
    """A conservative token ceiling for messages, tools, and reasoning input."""

    # Canonical JSON includes every caller-controlled input field. One token
    # per serialized UTF-8 byte is a conservative bound for supported models.
    return REQUEST_TOKEN_OVERHEAD + TOKENS_PER_CHAR_BOUND * len(
        contracts.canonical_json(dict(parameters)).encode("utf-8")
    )


def max_openrouter_cost_microusd(price_table: Mapping[str, Any], model: str, parameters: Mapping[str, Any], *, max_output_tokens: int) -> int:
    """Maximum possible cost from the bounded input, the capped output, and
    every other priced component; rounded up to micro-USD."""

    pricing = price_table["models"].get(model)
    if pricing is None:
        raise BrokerError("model_not_allowed")
    return _max_openrouter_cost_for_pricing(
        pricing, parameters, max_output_tokens=max_output_tokens
    )


def _max_openrouter_cost_for_pricing(
    pricing: Mapping[str, Any],
    parameters: Mapping[str, Any],
    *,
    max_output_tokens: int,
) -> int:
    input_tokens = bounded_input_tokens(parameters)
    output_tokens = int(max_output_tokens)
    web_search_enabled = _openrouter_web_search_enabled(parameters)
    web_search_calls = (
        int(parameters.get("max_tool_calls", 0))
        if web_search_enabled else 0
    )
    web_search_price = max(
        Decimal(pricing["web_search"]),
        OPENROUTER_NATIVE_WEB_SEARCH_RESERVATION_USD_PER_CALL,
    ) if web_search_enabled else Decimal("0")
    usd = (
        Decimal(pricing["prompt"]) * input_tokens
        + Decimal(pricing["completion"]) * output_tokens
        + Decimal(pricing["internal_reasoning"]) * output_tokens
        + Decimal(pricing["request"])
        + web_search_price * web_search_calls
    )
    return _microusd_ceiling(usd)


def _openrouter_web_search_enabled(parameters: Mapping[str, Any]) -> bool:
    tools = parameters.get("tools", [])
    return isinstance(tools, list) and any(
        isinstance(tool, Mapping)
        and tool.get("type") == "openrouter:web_search"
        for tool in tools
    )


@dataclass(frozen=True)
class OpenRouterHostRoute:
    provider_policy: Mapping[str, Any]
    reservation_pricing: Mapping[str, str]


def _json_decimal_number(value: Decimal) -> Any:
    """Return a plain JSON number without exponent notation."""

    text = format(value.normalize(), "f")
    return int(text) if "." not in text else float(text)


def _openrouter_host_route(
    *,
    kind: str,
    operation_id: str,
    model: str,
    pricing: Mapping[str, Any],
    parameters: Mapping[str, Any],
) -> Optional[OpenRouterHostRoute]:
    """Return the one source-controlled execution route that differs by model."""

    if (
        kind != "execute"
        or operation_id != "openrouter.responses"
        or model not in OPENROUTER_REGIONAL_LUNA_RESPONSES_MODELS
    ):
        return None
    regional = {
        component: Decimal(str(pricing[component]))
        * OPENROUTER_LUNA_REGIONAL_PRICE_MULTIPLIER
        for component in PRICED_COMPONENTS
    }
    if bounded_input_tokens(parameters) >= OPENROUTER_LUNA_LONG_CONTEXT_MIN_PROMPT_TOKENS:
        if model == OPENROUTER_GPT6_LUNA_RESPONSES_MODEL:
            long_prompt_price = OPENROUTER_GPT6_LUNA_LONG_CONTEXT_PROMPT_PRICE
            long_completion_price = (
                OPENROUTER_GPT6_LUNA_LONG_CONTEXT_COMPLETION_PRICE
            )
        else:
            long_prompt_price = OPENROUTER_LUNA_LONG_CONTEXT_PROMPT_PRICE
            long_completion_price = OPENROUTER_LUNA_LONG_CONTEXT_COMPLETION_PRICE
        regional["prompt"] = max(
            regional["prompt"],
            long_prompt_price * OPENROUTER_LUNA_REGIONAL_PRICE_MULTIPLIER,
        )
        regional["completion"] = max(
            regional["completion"],
            long_completion_price * OPENROUTER_LUNA_REGIONAL_PRICE_MULTIPLIER,
        )
    regional["prompt"] *= OPENROUTER_LUNA_CACHE_WRITE_PRICE_MULTIPLIER
    max_completion_price = (
        regional["completion"] + regional["internal_reasoning"]
    )
    provider_policy = {
        "data_collection": "deny",
        "zdr": True,
        # Exclude the generic Azure route while allowing OpenRouter to select
        # and fail over between the two independently verified regional routes.
        "only": ["azure/us", "azure/eu"],
        "allow_fallbacks": True,
        "max_price": {
            "prompt": _json_decimal_number(
                regional["prompt"] * OPENROUTER_PRICE_PER_MILLION
            ),
            "completion": _json_decimal_number(
                max_completion_price * OPENROUTER_PRICE_PER_MILLION
            ),
            "request": _json_decimal_number(regional["request"]),
        },
    }
    return OpenRouterHostRoute(
        provider_policy=provider_policy,
        reservation_pricing={
            component: format(value.normalize(), "f")
            for component, value in regional.items()
        },
    )


def actual_openrouter_cost_microusd(price_table: Mapping[str, Any], model: str, response_json: Any) -> Optional[int]:
    """Return the provider's actual charge, including cached usage or aliases."""

    if not isinstance(response_json, Mapping):
        return None
    pricing = price_table["models"].get(model)
    if pricing is None:
        return None
    cost = provider_costs.openrouter_cost(response_json)
    return None if cost is None else cost.microusd


# ---------------------------------------------------------------------------
# Credentials and transport
# ---------------------------------------------------------------------------



def openrouter_normalized(parameters: Mapping[str, Any]) -> Dict[str, Any]:
    """The OpenRouter body the broker hashes and sends: the output cap is always explicit."""

    token_field = "max_output_tokens" if "input" in parameters else "max_tokens"
    requested = parameters.get(token_field)
    cap = (operations.OPENROUTER_RESPONSES_MAX_OUTPUT_TOKENS
           if token_field == "max_output_tokens" else operations.OPENROUTER_MAX_OUTPUT_TOKENS)
    max_tokens = operations.OPENROUTER_MAX_OUTPUT_TOKENS if requested is None else min(int(requested), cap)
    if max_tokens < 1:
        raise BrokerError("invalid_request")
    normalized = dict(parameters)
    normalized[token_field] = max_tokens
    return normalized


def normalized_request(operation_id: str, parameters: Mapping[str, Any]) -> Dict[str, Any]:
    """Validate and normalize a request exactly as ``Broker.execute`` does."""

    normalized = operations.validate_operation_request(operation_id, parameters)
    if operations.OPERATIONS[operation_id].provider == "openrouter":
        normalized = openrouter_normalized(normalized)
    return normalized


def deepline_cost_microusd(body: bytes) -> Optional[int]:
    """Return Deepline credits charged at $0.10 each, or ``None``."""

    try:
        document = json.loads(bytes(body).decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        return None
    cost = provider_costs.deepline_cost(document)
    return None if cost is None else cost.microusd


def _openrouter_failed_cost_structure(document: Any, model: str) -> Dict[str, Any]:
    """Keep bounded billing structure, never response content or billing authority."""

    if not isinstance(document, Mapping) or document.get("status") != "failed":
        return {}

    def kind(value: Any) -> str:
        if value is None:
            return "null"
        if isinstance(value, Mapping):
            return "object"
        if isinstance(value, list):
            return "array"
        return "other"

    def bounded_integer(value: Any, maximum: int = 1_000_000_000) -> Optional[int]:
        return value if type(value) is int and 0 <= value <= maximum else None

    usage = document.get("usage")
    metadata = document.get("openrouter_metadata")
    output = document.get("output")
    error = document.get("error")
    code = error.get("code") if isinstance(error, Mapping) else None
    result: Dict[str, Any] = {
        "schema_version": 1,
        "usage_kind": kind(usage),
        "metadata_kind": kind(metadata),
        "output_kind": kind(output),
        "output_count": bounded_integer(len(output), 4096) if isinstance(output, list) else None,
        "rate_limit_error": isinstance(code, str) and code == "rate_limit_exceeded",
    }
    if isinstance(usage, Mapping):
        result["usage_cost_present"] = "cost" in usage
        result["usage_cost_null"] = usage.get("cost") is None
        for name in ("input_tokens", "output_tokens", "total_tokens"):
            result[name] = bounded_integer(usage.get(name))
        details = usage.get("output_tokens_details")
        result["reasoning_tokens"] = bounded_integer(details.get("reasoning_tokens")) if isinstance(details, Mapping) else None
    if isinstance(metadata, Mapping):
        result["requested_model_matches"] = metadata.get("requested") == model
        result["is_byok"] = metadata.get("is_byok") if type(metadata.get("is_byok")) is bool else None
        result["attempt"] = bounded_integer(metadata.get("attempt"), 128)
        for name in ("pipeline", "attempts"):
            value = metadata.get(name)
            result[name + "_present"] = name in metadata
            result[name + "_kind"] = kind(value)
            result[name + "_count"] = bounded_integer(len(value), 128) if isinstance(value, list) else None
        attempts = metadata.get("attempts")
        result["all_attempts_failed_http"] = (
            isinstance(attempts, list) and 0 < len(attempts) <= 128
            and all(isinstance(attempt, Mapping) and type(attempt.get("status")) is int
                    and 400 <= attempt["status"] <= 599 for attempt in attempts)
        )
    return result


def _openrouter_completed_rate_limit_retryable(
    document: Any,
    *,
    model: str,
    generation_id: Optional[str],
    credential_fingerprint: Optional[str],
) -> bool:
    """Recognize one complete failed response whose price is still unknown.

    This is execution evidence only. It never proves a zero charge: the
    original call stays uncertain and its generation remains reconcilable.
    """

    if (
        not isinstance(document, Mapping)
        or document.get("status") != "failed"
        or document.get("error_type") != "rate_limit_exceeded"
        or document.get("output") != []
        or "usage" not in document
        or document.get("usage") is not None
        or not isinstance(generation_id, str)
        or _OPENROUTER_GENERATION_ID_RE.fullmatch(generation_id) is None
        or not isinstance(credential_fingerprint, str)
        or _CREDENTIAL_FINGERPRINT_RE.fullmatch(credential_fingerprint) is None
    ):
        return False
    try:
        if _openrouter_responses_error_status(
            document, document.get("error")
        ) != 429:
            return False
    except operations.OperationResponseError:
        return False
    structure = _openrouter_failed_cost_structure(document, model)
    metadata_kind = structure.get("metadata_kind")
    metadata_valid = metadata_kind == "null" or (
        metadata_kind == "object"
        and structure.get("requested_model_matches") is True
        and structure.get("is_byok") is False
        and type(structure.get("attempt")) is int
        and structure.get("pipeline_present") is True
        and structure.get("pipeline_kind") == "array"
        and type(structure.get("pipeline_count")) is int
        and structure.get("attempts_present") is True
        and structure.get("attempts_kind") == "array"
        and type(structure.get("attempts_count")) is int
        and 0 < structure["attempts_count"] <= 128
        and structure.get("all_attempts_failed_http") is True
    )
    return (
        structure.get("usage_kind") == "null"
        and structure.get("output_kind") == "array"
        and structure.get("output_count") == 0
        and structure.get("rate_limit_error") is True
        and metadata_valid
    )


def _missing_provider_cost_call_doc(
    response: ProviderResponse,
    document: Any,
    *,
    call_succeeded: bool,
    deepline_request_id: Optional[str] = None,
    deepline_response_request_id: Optional[str] = None,
    deepline_execution_key: Optional[str] = None,
    deepline_operation: Optional[str] = None,
    openrouter_generation_id: Optional[str] = None,
    openrouter_model: Optional[str] = None,
    credential_fingerprint: Optional[str] = None,
) -> Dict[str, Any]:
    """Return bounded structural diagnostics without provider content."""

    is_mapping = isinstance(document, Mapping)
    status = response.status
    provider_status = (
        status if isinstance(status, int) and not isinstance(status, bool) else 0
    )
    body_bytes = len(response.body) if isinstance(response.body, bytes) else 0
    diagnostics: Dict[str, Any] = {
        "reason": "missing_provider_cost",
        "call_succeeded": bool(call_succeeded),
        "provider_status": provider_status,
        "body_bytes": body_bytes,
        "body_is_mapping": is_mapping,
        "usage_present": is_mapping and "usage" in document,
        "billing_present": is_mapping and "billing" in document,
    }
    top_status = document.get("status") if is_mapping else None
    if isinstance(top_status, str) and top_status in _DEEPLINE_JOB_STATUSES:
        diagnostics["top_level_job_status"] = top_status
    if openrouter_model is not None:
        structure = _openrouter_failed_cost_structure(document, openrouter_model)
        if structure:
            diagnostics["openrouter_failed_response_structure"] = structure
    if deepline_request_id is not None and deepline_operation is not None:
        diagnostics.update(
            {
                "deepline_request_id": deepline_request_id,
                "deepline_operation": deepline_operation,
            }
        )
    if deepline_response_request_id is not None:
        diagnostics["deepline_job_id"] = deepline_response_request_id
    if deepline_execution_key is not None:
        diagnostics["deepline_execution_key"] = deepline_execution_key
    if (
        deepline_request_id is not None
        and credential_fingerprint is not None
        and _DEEPLINE_REQUEST_ID_RE.fullmatch(deepline_request_id)
        and _CREDENTIAL_FINGERPRINT_RE.fullmatch(credential_fingerprint)
    ):
        diagnostics["credential_fingerprint"] = credential_fingerprint
    if (
        openrouter_generation_id is not None
        and credential_fingerprint is not None
        and _OPENROUTER_GENERATION_ID_RE.fullmatch(openrouter_generation_id)
        and _CREDENTIAL_FINGERPRINT_RE.fullmatch(credential_fingerprint)
    ):
        # These fields stay in the service-only ledger. They bind a later
        # exact generation lookup to the credential that sent the paid call.
        diagnostics.update(
            {
                "openrouter_generation_id": openrouter_generation_id,
                "credential_fingerprint": credential_fingerprint,
            }
        )
    if response.internal_provenance in (
        "credential_echo",
        "redirect_rejected",
        "response_too_large",
    ):
        diagnostics["response_provenance"] = response.internal_provenance
    return diagnostics


def inject_credential(outbound: operations.OutboundRequest, secret: str) -> Tuple[str, Dict[str, str]]:
    """Place the credential exactly where the operation table says."""

    placement = outbound.credential
    headers: Dict[str, str] = {"accept": "application/json, text/html;q=0.9, */*;q=0.1", "user-agent": "leadpoet-lab-arena-broker/1"}
    for name, value in getattr(outbound, "headers", {}).items():
        headers[str(name).lower()] = str(value)
    if outbound.content_type:
        headers["content-type"] = outbound.content_type
    if placement.location == "header":
        value = ("%s %s" % (placement.scheme, secret)) if getattr(placement, "scheme", None) else secret
        headers[placement.name] = value
        return outbound.url, headers
    if placement.location == "query":
        separator = "&" if "?" in outbound.url else "?"
        return outbound.url + separator + "%s=%s" % (placement.name, urlrequest.quote(secret, safe="")), headers
    raise BrokerError("broker_unavailable")


def _credential_fingerprint(secret: str) -> str:
    """Return an internal stable binding without retaining the credential."""

    return "sha256:" + hashlib.sha256(secret.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ProviderResponse:
    status: int
    headers: Mapping[str, str]
    body: bytes
    # Broker-internal provenance for a synthetic generic response. Never
    # forwarded as a provider header or model-visible response field.
    internal_provenance: Optional[str] = None


def _response_contains_credential(response: ProviderResponse, secret: str) -> bool:
    """Catch literal, URL-encoded, and JSON-escaped echoes before persistence."""

    secret_bytes = secret.encode("utf-8")

    def contains_secret(value: bytes) -> bool:
        return secret_bytes in value or secret_bytes in unquote_to_bytes(value)

    if contains_secret(response.body) or any(
        contains_secret(name.encode("utf-8", errors="surrogatepass"))
        or contains_secret(value.encode("utf-8", errors="surrogatepass"))
        for name, value in response.headers.items()
    ):
        return True
    try:
        # Keep duplicate object members, too: a client can inspect the raw
        # response even when its usual JSON decoder would discard a member.
        parsed = json.loads(response.body, object_pairs_hook=list)
    except (UnicodeDecodeError, ValueError):
        # Non-JSON text still has the literal check above. The operation's
        # existing sanitizer decides whether the response format is allowed.
        return False
    except RecursionError:
        return True  # Fail closed if a structured response cannot be inspected.
    pending = [parsed]
    while pending:
        value = pending.pop()
        if isinstance(value, str) and contains_secret(
            value.encode("utf-8", errors="surrogatepass")
        ):
            return True
        if isinstance(value, (list, tuple)):
            pending.extend(value)
    return False


class ProviderTransport(Protocol):
    def send(
        self,
        *,
        method: str,
        url: str,
        headers: Mapping[str, str],
        body: bytes,
        timeout_seconds: float,
        max_response_bytes: Optional[int] = None,
    ) -> ProviderResponse: ...


_PROVIDER_HTTP_IN_FLIGHT = contextvars.ContextVar("arena_provider_http_in_flight", default=False)


class _ProviderHTTPLogFilter(logging.Filter):
    """Keep vendor request URLs and response headers out of broker logs.

    Scrapingdog authenticates in its query string. The broker's redacted call
    records remain the operational log; unrelated HTTP traffic is unaffected.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        return not _PROVIDER_HTTP_IN_FLIGHT.get()


_PROVIDER_HTTP_LOG_FILTER = _ProviderHTTPLogFilter()


def _deepline_response_header_request_id(headers: Mapping[str, str]) -> Optional[str]:
    """Retain the provider's identity before reading a possibly broken body."""

    explicit = headers.get("x-deepline-request-id")
    vercel = headers.get("x-vercel-id", "")
    # Public API responses prepend the edge region to the native job identity.
    if vercel.count("::") == 2:
        edge, vercel = vercel.split("::", 1)
        if re.fullmatch(r"[a-z0-9]{3,8}", edge) is None:
            return None
    valid_vercel = vercel if _DEEPLINE_BILLING_REQUEST_ID_RE.fullmatch(vercel) else None
    if explicit is not None:
        if not _DEEPLINE_NATIVE_REQUEST_ID_RE.fullmatch(explicit):
            return None
        if valid_vercel is not None and explicit != valid_vercel:
            return None
        return explicit
    return valid_vercel


class HttpxProviderTransport:
    """HTTPS to the constant provider hosts: HTTP/1.1, no redirects, bounded."""

    def __init__(
        self,
        *,
        client_factory: Optional[Callable[[], httpx.AsyncClient]] = None,
        max_response_bytes: int = 4 * 1024 * 1024,
    ) -> None:
        self._client_factory = client_factory or (
            lambda: httpx.AsyncClient(
                http1=True,
                http2=False,
                follow_redirects=False,
                timeout=httpx.Timeout(30.0),
                trust_env=False,
            )
        )
        self._max_response_bytes = max_response_bytes
        self._closed = False
        # These are the loggers used by the pinned asynchronous HTTP/1.1 path.
        # The context-local filter does not silence concurrent unrelated work.
        for name in ("httpx", "httpcore.connection", "httpcore.http11", "httpcore.proxy"):
            logging.getLogger(name).addFilter(_PROVIDER_HTTP_LOG_FILTER)

    def send(
        self,
        *,
        method: str,
        url: str,
        headers: Mapping[str, str],
        body: bytes,
        timeout_seconds: float,
        max_response_bytes: Optional[int] = None,
    ) -> ProviderResponse:
        if not url.startswith("https://"):
            raise ProviderTransportError("non-https target")
        if self._closed:
            raise ProviderTransportError("transport closed")
        try:
            timeout = float(timeout_seconds)
        except (TypeError, ValueError) as exc:
            raise ProviderTransportError("invalid timeout") from exc
        if not math.isfinite(timeout) or timeout <= 0:
            raise ProviderTransportError("invalid timeout")
        response_limit = (
            self._max_response_bytes
            if max_response_bytes is None
            else max_response_bytes
        )
        if (
            isinstance(response_limit, bool)
            or not isinstance(response_limit, int)
            or response_limit < 1
        ):
            raise ProviderTransportError("invalid response limit")
        openrouter_generation_id: Optional[str] = None
        deepline_job_id: Optional[str] = None
        status = 0
        response_headers: Dict[str, str] = {}
        content = bytearray()
        oversized = False

        async def request_once() -> None:
            nonlocal status, response_headers, openrouter_generation_id
            nonlocal deepline_job_id, content, oversized
            async with self._client_factory() as client:
                async with client.stream(
                    method,
                    url,
                    headers=dict(headers),
                    content=body,
                    timeout=httpx.Timeout(timeout),
                ) as response:
                    status = int(response.status_code)
                    response_headers = {
                        key.lower(): value
                        for key, value in response.headers.items()
                    }
                    if urlsplit(url).hostname == "code.deepline.com":
                        deepline_job_id = _deepline_response_header_request_id(
                            response_headers
                        )
                    _, openrouter_generation_id = _openrouter_generation_identity(
                        None, response_headers
                    )
                    async for chunk in response.aiter_bytes(
                        chunk_size=64 * 1024
                    ):
                        if len(content) + len(chunk) > response_limit:
                            oversized = True
                            break
                        content.extend(chunk)

        log_token = _PROVIDER_HTTP_IN_FLIGHT.set(True)
        try:
            asyncio.run(asyncio.wait_for(request_once(), timeout=timeout))
        except asyncio.TimeoutError as exc:
            raise ProviderTransportError(
                "ReadTimeout",
                openrouter_generation_id=openrouter_generation_id,
                deepline_job_id=deepline_job_id,
                observed_status=status,
            ) from exc
        except httpx.HTTPError as exc:
            raise ProviderTransportError(
                type(exc).__name__,
                openrouter_generation_id=openrouter_generation_id,
                deepline_job_id=deepline_job_id,
                observed_status=status,
            ) from exc
        finally:
            _PROVIDER_HTTP_IN_FLIGHT.reset(log_token)
        if 300 <= status < 400:
            # Redirects are never followed; a redirecting provider is unavailable.
            return ProviderResponse(
                502, {"content-type": "application/json"},
                operations.GENERIC_UNAVAILABLE_BODY, "redirect_rejected",
            )
        if oversized:
            return ProviderResponse(
                502, {"content-type": "application/json"},
                operations.GENERIC_UNAVAILABLE_BODY, "response_too_large",
            )
        return ProviderResponse(status, response_headers, bytes(content))

    def close(self) -> None:
        self._closed = True


def _deepline_job_request_id(document: Any) -> Optional[str]:
    """Return one bounded Deepline request id, rejecting conflicting aliases."""

    if not isinstance(document, Mapping):
        return None
    present = [document[name] for name in ("job_id", "request_id", "requestId") if name in document]
    if not present or any(
        not isinstance(value, str) or not value.strip() or len(value) > 512
        for value in present
    ) or len(set(present)) != 1:
        return None
    return present[0]


def _deepline_async_job_ids(document: Any, flow: Mapping[str, Any]) -> Sequence[str]:
    """Read only declared provider-job paths, excluding the billing envelope ID."""

    if not isinstance(document, Mapping):
        return ()
    containers = []
    for name in ("result", "toolResponse"):
        value = document.get(name)
        if isinstance(value, Mapping):
            containers.append(value)
            if isinstance(value.get("rawV2"), Mapping):
                containers.append(value["rawV2"])
            if isinstance(value.get("data"), Mapping):
                containers.append(value["data"])
    if not containers and not any(name in document for name in ("job_id", "request_id", "requestId")):
        containers.append(document)
    result = []
    for path in flow.get("job_id_paths", ()):
        if not isinstance(path, str) or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*){0,7}", path) is None:
            continue
        for container in containers:
            value = container
            for component in path.split("."):
                value = value.get(component) if isinstance(value, Mapping) else None
            if (isinstance(value, str) and _DEEPLINE_NATIVE_REQUEST_ID_RE.fullmatch(value)
                and value not in result):
                result.append(value)
    return tuple(result[:32])


def _deepline_async_response_accepted(response: ProviderResponse, document: Any) -> bool:
    """Accept an authenticated async API reply, without claiming job completion."""
    if (response.internal_provenance is not None or not 200 <= response.status < 300
        or not isinstance(document, Mapping) or _deepline_job_request_id(document) is None):
        return False
    status = document.get("status")
    if not isinstance(status, str) or status not in {"completed", "pending", "queued", "running", "in_progress"}:
        return False
    wrapper = document.get("toolResponse")
    raw = wrapper.get("rawV2") if isinstance(wrapper, Mapping) else None
    nodes = [document, wrapper, raw, document.get("result")]
    nodes += [node.get("data") for node in tuple(nodes) if isinstance(node, Mapping)]
    for node in nodes:
        if not isinstance(node, Mapping):
            continue
        node_status = node.get("status")
        if (node.get("error") is not None or node.get("tool_error") is not None
            or ("isError" in node and node["isError"] is not False)
            or ("success" in node and node["success"] is not True)
            or (node_status is not None and not isinstance(node_status, str))
            or node_status in {"failed", "cancelled", "canceled", "error"}):
            return False
    return True


def _deepline_accepted_async_job_ids(
    response: ProviderResponse, document: Any, flow: Mapping[str, Any],
) -> Sequence[str]:
    if not _deepline_async_response_accepted(response, document):
        return ()
    return _deepline_async_job_ids(document, flow)


def _deepline_billing_readback(
    *,
    transport: ProviderTransport,
    secret: str,
    request_id: str,
    operation: str,
    reconciliation_deadline: float,
) -> Optional[provider_costs.ProviderCost]:
    """Poll bounded billing history and return only one exact terminal charge."""

    readback_deadline = min(
        reconciliation_deadline,
        time.monotonic() + operations.PROVIDER_BILLING_RECONCILIATION_SECONDS,
    )
    headers = {
        "accept": "application/json",
        "authorization": "Bearer " + secret,
        "user-agent": "leadpoet-lab-arena-broker/1",
    }
    history_url = DEEPLINE_BILLING_HISTORY_URL
    current_offset = 0
    next_deeper_offset: Optional[int] = None
    for request_number in range(_DEEPLINE_BILLING_MAX_ATTEMPTS):
        # Alternate a paced newest-page refresh with one progressively older
        # page. This catches delayed posting without losing bounded coverage
        # when account-wide traffic pushes a new charge beyond the first page.
        now = time.monotonic()
        if now >= readback_deadline:
            break
        if request_number and history_url == DEEPLINE_BILLING_HISTORY_URL:
            delay = min(
                _DEEPLINE_BILLING_POLL_SECONDS, max(0.0, readback_deadline - now)
            )
            if delay <= 0:
                break
            time.sleep(delay)
            now = time.monotonic()
            if now >= readback_deadline:
                break
        try:
            response = transport.send(
                method="GET",
                url=history_url,
                headers=headers,
                body=b"",
                timeout_seconds=max(0.001, readback_deadline - now),
            )
        except ProviderTransportError:
            continue
        if _response_contains_credential(response, secret):
            return None
        if response.status != 200:
            continue
        try:
            document = json.loads(response.body.decode("utf-8"))
        except (UnicodeDecodeError, ValueError):
            return None
        state, cost, has_more, next_offset = provider_costs.deepline_billing_history_cost(
            document, request_id=request_id, operation=operation,
            current_offset=current_offset,
        )
        if state == "invalid":
            return None
        if state == "matched":
            return cost
        if state == "pending" and has_more and current_offset == 0:
            if next_offset is None:
                return None
            current_offset = next_deeper_offset or next_offset
            history_url = DEEPLINE_BILLING_HISTORY_URL + "&recent_offset=" + str(current_offset)
            continue
        if state == "pending" and has_more and current_offset > 0:
            next_deeper_offset = next_offset
        elif state == "pending":
            next_deeper_offset = None
        history_url = DEEPLINE_BILLING_HISTORY_URL
        current_offset = 0
    return None


def _deepline_execution_response_readback(
    *, transport: ProviderTransport, secret: str, execution_key: Optional[str],
    operation: str, operation_aliases: Sequence[str], max_response_bytes: int,
    observed_status: Optional[int] = None,
    resume_request: Optional[Mapping[str, Any]] = None,
) -> Optional[ProviderResponse]:
    """Read a saved reply, or resume a proved running request with its same key."""
    if execution_key is None or _DEEPLINE_EXECUTION_KEY_RE.fullmatch(execution_key) is None:
        return None
    recovery_deadline = (time.monotonic() + float(resume_request["timeout_seconds"])
                         if resume_request is not None else None)
    try:
        lookup = transport.send(
            method="GET", url=DEEPLINE_EXECUTION_BY_KEY_URL + quote(execution_key, safe=""),
            headers={"accept": "application/json", "authorization": "Bearer " + secret,
                     "user-agent": "leadpoet-lab-arena-broker/1"}, body=b"",
            timeout_seconds=(min(DEEPLINE_DELAYED_RECONCILIATION_TIMEOUT_SECONDS,
                max(0.001, recovery_deadline - time.monotonic()))
                if recovery_deadline is not None else DEEPLINE_DELAYED_RECONCILIATION_TIMEOUT_SECONDS),
            max_response_bytes=max_response_bytes + 16_384,
        )
    except ProviderTransportError:
        return None
    if (lookup.status != 200 or lookup.internal_provenance is not None
        or len(lookup.body) > max_response_bytes + 16_384
        or [value for name, value in lookup.headers.items()
            if name.lower() == "x-deepline-idempotency-supported"] != ["true"]
        or _response_contains_credential(lookup, secret)):
        return None
    try:
        document = json.loads(lookup.body.decode("utf-8"))
        recovery = document.get("executionRecovery") if isinstance(document, Mapping) else None
        native_id = document.get("requestId") if isinstance(document, Mapping) else None
        saved = document.get("response") if isinstance(document, Mapping) else None
        status = document.get("responseStatus") if isinstance(document, Mapping) else None
        if (resume_request is not None and isinstance(recovery, Mapping)
            and recovery.get("idempotencyKey") == execution_key
            and recovery.get("state") == "running"
            and document.get("toolId") in (operation, *operation_aliases)
            and isinstance(native_id, str)
            and _DEEPLINE_NATIVE_REQUEST_ID_RE.fullmatch(native_id)
            and not _DEEPLINE_REQUEST_ID_RE.fullmatch(native_id)):
            # The observed workspace key already owns this execution. An
            # identical POST can continue its declared lifecycle, not launch
            # a new paid request. Unknown/missing keys never reach this branch.
            if resume_request.get("headers", {}).get("idempotency-key") != execution_key:
                return None
            if recovery_deadline is None or time.monotonic() >= recovery_deadline:
                return None
            try:
                resumed = transport.send(**dict(resume_request,
                    timeout_seconds=max(0.001, recovery_deadline - time.monotonic())))
            except ProviderTransportError:
                return None
            if (resumed.internal_provenance is not None
                or len(resumed.body) > max_response_bytes
                or _response_contains_credential(resumed, secret)):
                return None
            resumed_document = json.loads(resumed.body.decode("utf-8"))
            resumed_recovery = (resumed_document.get("executionRecovery")
                                if isinstance(resumed_document, Mapping) else None)
            if (_deepline_job_request_id(resumed_document) != native_id
                or not isinstance(resumed_document, Mapping)
                or resumed_document.get("status") != "completed"
                or (resumed_recovery is not None and (
                    not isinstance(resumed_recovery, Mapping)
                    or resumed_recovery.get("idempotencyKey") != execution_key
                    or resumed_recovery.get("state") != "completed"))
                or (observed_status is not None and resumed.status != observed_status)):
                return None
            # Retain the standard execute envelope, but hide the recovery key.
            if isinstance(resumed_document, Mapping):
                resumed_document = dict(resumed_document)
                resumed_document.pop("executionRecovery", None)
            body = json.dumps(resumed_document, ensure_ascii=False, allow_nan=False,
                              separators=(",", ":")).encode("utf-8")
            return ProviderResponse(resumed.status, resumed.headers, body)
        # A Vercel trace hint can differ from the keyed native billing ID.
        # The authenticated saved key/tool/body binding is stronger authority.
        if (not isinstance(recovery, Mapping) or recovery.get("idempotencyKey") != execution_key
            or recovery.get("state") != "completed" or document.get("toolId") not in (operation, *operation_aliases)
            or not isinstance(native_id, str) or _DEEPLINE_NATIVE_REQUEST_ID_RE.fullmatch(native_id) is None
            or _DEEPLINE_REQUEST_ID_RE.fullmatch(native_id) is not None
            or type(status) is not int or not 200 <= status <= 599
            or (observed_status is not None and observed_status != status)
            or not isinstance(saved, Mapping) or _deepline_job_request_id(saved) != native_id):
            return None
        saved = dict(saved)
        saved.pop("executionRecovery", None)
        body = json.dumps(saved, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")
    except (UnicodeError, ValueError, TypeError, RecursionError):
        return None
    response = ProviderResponse(status, {"content-type": "application/json", "x-deepline-request-id": native_id}, body)
    if len(body) > max_response_bytes or _response_contains_credential(response, secret):
        return None
    return response



def _deepline_exact_readback(
    *, transport: ProviderTransport, secret: str,
    request_id: Optional[str], execution_key: Optional[str], operation: str,
    reconciliation_deadline: float, provider: Optional[str] = None,
    operation_aliases: Sequence[str] = (),
    poll: bool = True,
) -> Tuple[Optional[str], Optional[provider_costs.ProviderCost]]:
    """Recover one execution identity and read only its own final charge.

    All retries are GETs. Neither an unsupported key lookup nor a missing
    execution or charge permits replay of a paid request.
    """

    native_id = request_id
    if native_id is not None and (
        _DEEPLINE_NATIVE_REQUEST_ID_RE.fullmatch(native_id) is None
        or _DEEPLINE_REQUEST_ID_RE.fullmatch(native_id) is not None
    ):
        native_id = None
    if execution_key is not None and _DEEPLINE_EXECUTION_KEY_RE.fullmatch(execution_key) is None:
        return native_id, None
    # Completed request errors get one short read, not the success/recovery
    # polling window. Their unknown bill stays eligible for delayed settlement.
    timeout = (
        operations.PROVIDER_BILLING_RECONCILIATION_SECONDS
        if poll else DEEPLINE_DELAYED_RECONCILIATION_TIMEOUT_SECONDS
    )
    deadline = min(reconciliation_deadline, time.monotonic() + timeout)
    headers = {
        "accept": "application/json", "authorization": "Bearer " + secret,
        "user-agent": "leadpoet-lab-arena-broker/1",
    }
    for index in range(_DEEPLINE_BILLING_MAX_ATTEMPTS if poll else 1):
        now = time.monotonic()
        if now >= deadline:
            break
        if index:
            time.sleep(min(_DEEPLINE_BILLING_POLL_SECONDS, deadline - now))
            now = time.monotonic()
            if now >= deadline:
                break
        if native_id is None:
            if execution_key is None:
                return None, None
            url = DEEPLINE_EXECUTION_BY_KEY_URL + quote(execution_key, safe="")
        else:
            url = DEEPLINE_EXACT_BILLING_URL + quote(native_id, safe="")
        try:
            response = transport.send(method="GET", url=url, headers=headers, body=b"",
                                      timeout_seconds=max(0.001, deadline - now))
        except ProviderTransportError:
            continue
        if _response_contains_credential(response, secret):
            return native_id, None
        if native_id is None:
            supported = [value for name, value in response.headers.items()
                         if name.lower() == "x-deepline-idempotency-supported"]
            if supported != ["true"]:
                return None, None
        if response.status == 404:
            # Absence is not proof that no dispatch or charge occurred.
            continue
        if response.status != 200:
            continue
        try:
            document = json.loads(response.body.decode("utf-8"), parse_float=Decimal)
        except (UnicodeDecodeError, ValueError):
            return native_id, None
        if native_id is None:
            recovery = document.get("executionRecovery") if isinstance(document, Mapping) else None
            recovered_id = document.get("requestId") if isinstance(document, Mapping) else None
            if (
                not isinstance(recovery, Mapping)
                or recovery.get("idempotencyKey") != execution_key
                or document.get("toolId") not in (operation, *operation_aliases)
                or not isinstance(recovered_id, str)
                or _DEEPLINE_NATIVE_REQUEST_ID_RE.fullmatch(recovered_id) is None
                or _DEEPLINE_REQUEST_ID_RE.fullmatch(recovered_id) is not None
            ):
                return None, None
            native_id = recovered_id
            # Key recovery is a read, not a polling interval. Read the charge
            # within this same deadline before the next paced poll.
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return native_id, None
            try:
                response = transport.send(
                    method="GET", url=DEEPLINE_EXACT_BILLING_URL + quote(native_id, safe=""),
                    headers=headers, body=b"", timeout_seconds=remaining,
                )
            except ProviderTransportError:
                continue
            if _response_contains_credential(response, secret):
                return native_id, None
            if response.status != 200:
                continue
            try:
                document = json.loads(response.body.decode("utf-8"), parse_float=Decimal)
            except (UnicodeDecodeError, ValueError):
                return native_id, None
        state, cost = provider_costs.deepline_exact_request_cost(
            document, request_id=native_id, operation=operation, provider=provider,
            operation_aliases=operation_aliases,
        )
        if state == "matched":
            return native_id, cost
        if state == "invalid":
            return native_id, None
    return native_id, None


def _deepline_ledger_readback(
    *,
    transport: ProviderTransport,
    secret: str,
    request_id: str,
    operation: str,
    since_at: int,
    reconciliation_deadline: float,
) -> Optional[provider_costs.ProviderCost]:
    """Read exact request billing; never repeat a provider execution."""

    if (
        _DEEPLINE_BILLING_REQUEST_ID_RE.fullmatch(request_id) is None
        or isinstance(since_at, bool)
        or not isinstance(since_at, int)
        or since_at < 0
    ):
        return None
    deadline = min(
        reconciliation_deadline,
        time.monotonic() + operations.PROVIDER_BILLING_RECONCILIATION_SECONDS,
    )
    headers = {
        "accept": "application/json",
        "authorization": "Bearer " + secret,
        "user-agent": "leadpoet-lab-arena-broker/1",
    }
    base_url = DEEPLINE_BILLING_LEDGER_URL + "?since_at=%d&limit=50" % since_at
    cursor = None
    seen_cursors = set()
    for index in range(_DEEPLINE_BILLING_MAX_ATTEMPTS):
        now = time.monotonic()
        if now >= deadline:
            break
        if index and cursor is None:
            time.sleep(min(_DEEPLINE_BILLING_POLL_SECONDS, deadline - now))
            now = time.monotonic()
            if now >= deadline:
                break
        url = base_url if cursor is None else base_url + "&cursor=" + quote(cursor, safe="")
        try:
            response = transport.send(
                method="GET", url=url, headers=headers, body=b"",
                timeout_seconds=max(0.001, deadline - now),
            )
        except ProviderTransportError:
            continue
        if _response_contains_credential(response, secret):
            return None
        if response.status != 200:
            continue
        try:
            document = json.loads(response.body.decode("utf-8"))
        except (UnicodeDecodeError, ValueError):
            return None
        state, cost, has_more, next_cursor = provider_costs.deepline_billing_ledger_cost(
            document, request_id=request_id, operation=operation, current_cursor=cursor,
        )
        if state == "invalid":
            return None
        if state == "matched":
            return cost
        if has_more:
            if not next_cursor or next_cursor in seen_cursors:
                return None
            seen_cursors.add(next_cursor)
            cursor = next_cursor
        else:
            cursor = None
            seen_cursors.clear()
    return None


def _openrouter_generation_identity(
    document: Any, headers: Mapping[str, Any]
) -> Tuple[bool, Optional[str]]:
    """Return whether a generation id was present and one valid exact value."""

    values = [
        value
        for name, value in headers.items()
        if isinstance(name, str) and name.lower() == _OPENROUTER_GENERATION_HEADER
    ]
    present = bool(values)
    if isinstance(document, Mapping) and "id" in document:
        present = True
        values.append(document["id"])
    if not present:
        return False, None
    if any(
        not isinstance(value, str)
        or _OPENROUTER_GENERATION_ID_RE.fullmatch(value.strip()) is None
        for value in values
    ):
        return True, None
    normalized = {value.strip() for value in values}
    if len(normalized) != 1:
        return True, None
    return True, normalized.pop()


def _openrouter_generation_readback(
    *,
    transport: ProviderTransport,
    secret: str,
    generation_id: str,
    reconciliation_deadline: float,
) -> Optional[provider_costs.ProviderCost]:
    """Poll one exact OpenRouter generation without retrying the paid request."""

    readback_deadline = min(
        reconciliation_deadline,
        time.monotonic() + operations.PROVIDER_BILLING_RECONCILIATION_SECONDS,
    )
    for attempt in range(_OPENROUTER_BILLING_MAX_ATTEMPTS):
        now = time.monotonic()
        if now >= readback_deadline:
            break
        if attempt:
            delay = min(
                _OPENROUTER_BILLING_POLL_SECONDS,
                max(0.0, readback_deadline - now),
            )
            if delay <= 0:
                break
            time.sleep(delay)
            now = time.monotonic()
            if now >= readback_deadline:
                break
        state, cost = _openrouter_generation_readback_once(
            transport=transport,
            secret=secret,
            generation_id=generation_id,
            timeout_seconds=min(
                operations.PROVIDER_BILLING_RECONCILIATION_SECONDS,
                max(0.001, readback_deadline - now),
            ),
        )
        if state == "retry":
            continue
        if cost is not None:
            return cost
        if state == "terminal":
            return None
    return None


def _openrouter_generation_readback_once(
    *,
    transport: ProviderTransport,
    secret: str,
    generation_id: str,
    timeout_seconds: float,
) -> Tuple[str, Optional[provider_costs.ProviderCost]]:
    """Read one exact generation once; never send another paid request."""

    if (
        _OPENROUTER_GENERATION_ID_RE.fullmatch(generation_id) is None
        or not 0 < float(timeout_seconds)
        <= operations.PROVIDER_BILLING_RECONCILIATION_SECONDS
    ):
        return "terminal", None
    headers = {
        "accept": "application/json",
        "authorization": "Bearer " + secret,
        "user-agent": "leadpoet-lab-arena-broker/1",
    }
    url = OPENROUTER_GENERATION_URL + quote(generation_id, safe="")
    try:
        response = transport.send(
            method="GET",
            url=url,
            headers=headers,
            body=b"",
            timeout_seconds=float(timeout_seconds),
        )
    except ProviderTransportError:
        return "retry", None
    if _response_contains_credential(response, secret):
        return "terminal", None
    if response.status in (401, 403):
        return "terminal", None
    if response.status != 200:
        return "retry", None
    try:
        document = json.loads(response.body.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        return "terminal", None
    cost = provider_costs.openrouter_generation_cost(
        document, generation_id=generation_id
    )
    return ("found", cost) if cost is not None else ("retry", None)


# ---------------------------------------------------------------------------
# Broker
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RunContext:
    """Identity of the leased run the worker is executing; never model-supplied."""

    run_id: str
    assignment_id: str
    icp_position: int
    lease_token_hash: str
    miner_hotkey: str
    submission_id: str
    stage: int
    kind: str = "execute"  # "execute" runs a miner model; "score" runs the Arena judge on a miner's output
    attempt: int = 1
    round_id: str = ""
    # Gateway-owned frozen scorer identity. An absent scope disables replay.
    judgment_cache_scope: Optional[Mapping[str, Any]] = None
    # Gateway-owned, hash-bound tool and pricing snapshot for this round.
    deepline_catalog: Optional[Mapping[str, Any]] = None


@dataclass(frozen=True)
class BrokerResult:
    status: int
    headers: Dict[str, str]
    body: bytes
    call: Dict[str, Any]
    # Gateway-private diagnostics for bounded champion credential retries.
    # The worker document deliberately excludes this field.
    attempt_trace: Tuple[Dict[str, Any], ...] = ()

    def to_document(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "headers": dict(self.headers),
            "body_b64": base64.b64encode(self.body).decode("ascii"),
            "call": dict(self.call),
        }


def _provider_attempt_summary(
    result: BrokerResult,
    *,
    provider_attempt: int,
    occurred_at: str,
    elapsed_ms: int,
) -> Dict[str, Any]:
    """Project one retry into safe gateway-private diagnostic metadata."""

    summary: Dict[str, Any] = {
        "occurred_at": occurred_at,
        "elapsed_ms": max(0, min(int(elapsed_ms), 3_600_000)),
        "http_status": max(0, min(int(result.status), 999)),
        "provider_attempt": max(1, min(int(provider_attempt), 4)),
    }
    call = result.call if isinstance(result.call, Mapping) else {}
    text_fields = {
        "call_identity": (r"^sha256:[0-9a-f]{64}$", 71),
        "operation_id": (r"^[a-z0-9][a-z0-9._-]{0,79}$", 80),
        "provider": (r"^[a-z][a-z0-9_-]{0,31}$", 32),
        "error_code": (r"^[a-z][a-z0-9_]{0,63}$", 64),
    }
    for name, (pattern, limit) in text_fields.items():
        value = call.get(name)
        if (
            isinstance(value, str)
            and len(value) <= limit
            and re.fullmatch(pattern, value)
        ):
            summary[name] = value
    provider_status = call.get("provider_status")
    if (
        isinstance(provider_status, int)
        and not isinstance(provider_status, bool)
        and 0 <= provider_status <= 999
    ):
        summary["provider_status"] = provider_status
    return summary


class CallStore(Protocol):
    def reserve_call(self, **kwargs: Any) -> Dict[str, Any]: ...

    def reserve_judgment_call(self, **kwargs: Any) -> Dict[str, Any]: ...

    def mark_dispatched(self, **kwargs: Any) -> Dict[str, Any]: ...

    def settle_call(self, **kwargs: Any) -> Dict[str, Any]: ...

    def mark_uncertain(self, **kwargs: Any) -> Dict[str, Any]: ...

    def recover_deepline_response(self, **kwargs: Any) -> Dict[str, Any]: ...

    def reconcile_openrouter_cost(self, **kwargs: Any) -> Dict[str, Any]: ...

    def reconcile_deepline_cost(self, **kwargs: Any) -> Dict[str, Any]: ...

    def get_run(self, run_id: str) -> Optional[Dict[str, Any]]: ...

    def list_ledger(self, **kwargs: Any) -> Sequence[Dict[str, Any]]: ...


def _reservation_readback_matches(
    store: CallStore,
    reservation_arguments: Mapping[str, Any],
    state: Mapping[str, Any],
    *,
    confirmed_cost_admission: bool = False,
    ledger_entries: Optional[Sequence[Mapping[str, Any]]] = None,
) -> bool:
    """Validate the durable reservation and current state after response loss."""

    identity = reservation_arguments.get("call_identity")
    if state.get("call_identity") != identity:
        return False
    rows = (store.list_ledger(call_identity=identity, limit=64)
            if ledger_entries is None else ledger_entries)
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes, bytearray)):
        return False
    if not rows or any(not isinstance(row, Mapping) for row in rows):
        return False
    entries = list(rows)
    reservations = [row for row in entries if row.get("entry_kind") == "reservation"]
    if len(reservations) != 1 or entries[0].get("entry_kind") != "reservation":
        return False
    reservation = reservations[0]
    expected = {
        "run_id": reservation_arguments.get("run_id"),
        "call_identity": identity,
        "operation_id": reservation_arguments.get("operation_id"),
        "provider": reservation_arguments.get("provider"),
        "funding_source": reservation_arguments.get("funding_source"),
    }
    prior_entry_id = 0
    allowed_kinds = {
        "reservation", "dispatch", "settlement", "uncertain", "recovery",
    }
    for entry in entries:
        entry_id = entry.get("entry_id")
        amount = entry.get("amount_microusd")
        if (
            any(entry.get(key) != value for key, value in expected.items())
            or isinstance(entry_id, bool)
            or not isinstance(entry_id, int)
            or entry_id <= prior_entry_id
            or entry.get("entry_kind") not in allowed_kinds
            or isinstance(amount, bool)
            or not isinstance(amount, int)
            or amount < 0
        ):
            return False
        prior_entry_id = entry_id
    if len({entry.get("entry_kind") for entry in entries}) != len(entries):
        return False
    call_doc = reservation.get("entry_doc")
    expected_call_doc = reservation_arguments.get("call_doc")
    if not isinstance(call_doc, Mapping) or not isinstance(expected_call_doc, Mapping):
        return False
    if dict(call_doc) != dict(expected_call_doc):
        source_fields = {
            "judgment_cache_source_call_identity",
            "judgment_cache_source_run_id",
        }
        terminal = state.get("terminal_response")
        if (
            set(call_doc) - set(expected_call_doc) != source_fields
            or {key: value for key, value in call_doc.items() if key not in source_fields}
            != dict(expected_call_doc)
            or not isinstance(terminal, Mapping)
            or terminal.get("judgment_cache_key")
            != expected_call_doc.get("judgment_cache_key")
            or any(call_doc[field] != terminal.get(field) for field in source_fields)
        ):
            return False
    reserved_amount = reservation.get("amount_microusd")
    if isinstance(reserved_amount, bool) or not isinstance(reserved_amount, int) or reserved_amount < 0:
        return False
    dynamic = call_doc.get("reserve_remaining_budget") is True
    if (
        not dynamic
        and reserved_amount != reservation_arguments.get("amount_microusd")
        and not (confirmed_cost_admission and reserved_amount == 0)
    ):
        return False
    kind_to_status = {
        "reservation": "reserved", "dispatch": "dispatched",
        "settlement": "settled", "uncertain": "uncertain",
        "recovery": "recovered", "refusal": "refused",
    }
    head = entries[-1]
    if state.get("status") != kind_to_status.get(head.get("entry_kind")):
        return False
    state_amount = state.get("amount_microusd")
    return (
        not isinstance(state_amount, bool)
        and isinstance(state_amount, int)
        and state_amount >= 0
        and state_amount == head.get("amount_microusd")
    )


def _validated_response_url(value: Any, *, secret: str = "") -> str:
    try:
        response_url = operations.validate_https_url(
            value,
            max_length=2000,
            field="response_url",
        )
    except operations.OperationError as exc:
        raise BrokerError("broker_unavailable") from exc
    if secret:
        encoded = response_url.encode("utf-8", errors="surrogatepass")
        secret_bytes = secret.encode("utf-8")
        if secret_bytes in encoded or secret_bytes in unquote_to_bytes(encoded):
            raise BrokerError("broker_unavailable")
    return response_url


def _terminal_response_document(
    status: int, headers: Mapping[str, str], body: bytes,
    *, call_succeeded: bool,
    provider_cost: Optional[Mapping[str, Any]] = None,
    account_failure_evidence: Optional[Mapping[str, Any]] = None,
    credit_failure_proof: Optional[Mapping[str, Any]] = None,
    judgment_cache_eligible: bool = False,
) -> Dict[str, Any]:
    document = {
        "status": int(status),
        "headers": dict(headers),
        "body_b64": base64.b64encode(bytes(body)).decode("ascii"),
        "call_succeeded": bool(call_succeeded),
    }
    if provider_cost is not None:
        document["provider_cost"] = dict(provider_cost)
    if account_failure_evidence is not None:
        document["account_failure_evidence"] = dict(account_failure_evidence)
    if credit_failure_proof is not None:
        document["credit_failure_proof"] = dict(credit_failure_proof)
    if judgment_cache_eligible:
        document["judgment_cache_eligible"] = True
    return document


def _confirmed_credit_failure_proof(
    *, provider: str, funding_source: str, response: ProviderResponse,
    raw_document: Any, raw_actual: Optional[int], actual: int,
    call_succeeded: bool,
    openrouter_generation_present: bool,
) -> Optional[Dict[str, Any]]:
    """Mark only a settled credit refusal with a proved zero charge.

    A direct OpenRouter 402 without a generation has no dispatched generation
    to bill. Embedded OpenRouter failures need a zero-cost receipt. Deepline
    needs its strict refusal or exact billing proof. Scrapingdog charges its
    fixed tariff only for an authenticated successful response.
    """

    if (
        funding_source not in ("host", "miner_key")
        or response.status != 402
        or response.internal_provenance is not None
        or call_succeeded
        or type(actual) is not int
        or actual != 0
    ):
        return None
    if provider == "openrouter":
        if raw_actual != 0 and (
            openrouter_generation_present
            or (
                isinstance(raw_document, Mapping)
                and any(
                    field in raw_document
                    for field in ("usage", "cost", "total_cost")
                )
            )
        ):
            return None
    elif provider == "deepline":
        if raw_actual != 0:
            return None
    elif provider != "scrapingdog":
        return None
    return {
        "schema_version": "leadpoet.lab_arena.credit_failure_proof.v1",
        "provider": provider,
        "reason": "out_of_credit",
        "provider_status": 402,
        "actual_microusd": 0,
    }


def _validated_credit_failure_proof(value: Any) -> Optional[Mapping[str, Any]]:
    if value is None:
        return None
    if (
        not isinstance(value, Mapping)
        or set(value) != {
            "schema_version", "provider", "reason", "provider_status",
            "actual_microusd",
        }
        or value.get("schema_version")
        != "leadpoet.lab_arena.credit_failure_proof.v1"
        or value.get("provider") not in ("openrouter", "deepline", "scrapingdog")
        or value.get("reason") != "out_of_credit"
        or type(value.get("provider_status")) is not int
        or value["provider_status"] != 402
        or type(value.get("actual_microusd")) is not int
        or value["actual_microusd"] != 0
    ):
        raise BrokerError("broker_unavailable")
    return value


def _judgment_cache_material(
    context: RunContext,
    *,
    requested_operation_id: str,
    effective_operation_id: str,
    normalized_request: Mapping[str, Any],
    outbound_body: bytes,
) -> Optional[Tuple[str, str]]:
    """Hash one exact gateway-owned scorer request, never a model cache key."""

    frozen = context.judgment_cache_scope
    expected = {
        "round_id", "evaluation_date", "scorer_image_digest",
        "scorer_image_reference", "scorer_policy",
    }
    if (
        context.kind != "score"
        or requested_operation_id not in {"openrouter.chat", "openrouter.responses"}
        or effective_operation_id != requested_operation_id
        or not isinstance(frozen, Mapping)
        or set(frozen) != expected
        or frozen.get("round_id") != context.round_id
        or not isinstance(frozen.get("evaluation_date"), str)
        or not isinstance(frozen.get("scorer_image_reference"), str)
        or not frozen["scorer_image_reference"]
        or not isinstance(frozen.get("scorer_policy"), Mapping)
    ):
        return None
    try:
        date.fromisoformat(frozen["evaluation_date"])
        contracts.require_sha256(frozen["scorer_image_digest"], "scorer image")
        policy = contracts.validate_scorer_policy(frozen["scorer_policy"])
        if (
            policy != dict(frozen["scorer_policy"])
            or normalized_request.get("model") not in policy["judge_models"].values()
        ):
            return None
        scope = {
            "schema_version": JUDGMENT_CACHE_SCHEMA_VERSION,
            **dict(frozen),
            "requested_operation_id": requested_operation_id,
            "effective_operation_id": effective_operation_id,
            "request_hash": contracts.document_hash(normalized_request),
            "outbound_body_hash": contracts.hash_bytes(outbound_body),
        }
        canonical = contracts.canonical_json(scope)
    except (ArenaContractError, TypeError, ValueError):
        return None
    if len(canonical.encode("utf-8")) > 60_000:
        return None
    return contracts.hash_bytes(canonical.encode("utf-8")), canonical


def _matches_declared_verdict_shape(value: Any, schema: Any, *, depth: int = 0) -> bool:
    """Check only the strict JSON-schema shape used for cached verdicts.

    Unsupported schema features cause a miss. This does not decide scoring;
    it only prevents an incomplete provider reply from becoming shared input.
    """

    if not isinstance(schema, Mapping) or depth > 8:
        return False
    allowed = {
        "type", "enum", "required", "properties", "additionalProperties",
        "items", "minItems", "maxItems", "minLength", "maxLength",
        "minimum", "maximum", "title", "description",
    }
    if set(schema) - allowed:
        return False
    enum = schema.get("enum")
    if enum is not None and (not isinstance(enum, list) or value not in enum):
        return False
    kind = schema.get("type")
    if isinstance(kind, list):
        return bool(kind) and all(isinstance(option, str) for option in kind) and any(
            _matches_declared_verdict_shape(
                value, {**schema, "type": option}, depth=depth + 1,
            ) for option in kind
        )
    if kind == "object":
        if not isinstance(value, Mapping):
            return False
        properties = schema.get("properties", {})
        required = schema.get("required", [])
        if (
            not isinstance(properties, Mapping)
            or not isinstance(required, list)
            or any(not isinstance(key, str) for key in required)
            or not set(required) <= set(value)
            or schema.get("additionalProperties", False) is not False
            or set(value) - set(properties)
        ):
            return False
        return all(
            _matches_declared_verdict_shape(item, properties[key], depth=depth + 1)
            for key, item in value.items()
        )
    if kind == "array":
        minimum = schema.get("minItems", 0)
        maximum = schema.get("maxItems", 1_000_000)
        return (
            isinstance(value, list)
            and type(minimum) is int and type(maximum) is int
            and 0 <= minimum <= len(value) <= maximum
            and "items" in schema
            and all(
                _matches_declared_verdict_shape(item, schema["items"], depth=depth + 1)
                for item in value
            )
        )
    valid_type = {
        "string": lambda: isinstance(value, str),
        "integer": lambda: type(value) is int,
        "number": lambda: type(value) in (int, float),
        "boolean": lambda: type(value) is bool,
        "null": lambda: value is None,
    }.get(kind)
    if valid_type is None or not valid_type():
        return False
    if kind == "string":
        minimum = schema.get("minLength", 0)
        maximum = schema.get("maxLength", 1_000_000)
        return (
            type(minimum) is int and type(maximum) is int
            and 0 <= minimum <= len(value) <= maximum
        )
    if kind in {"integer", "number"}:
        minimum = schema.get("minimum", float("-inf"))
        maximum = schema.get("maximum", float("inf"))
        return (
            type(minimum) in (int, float)
            and type(maximum) in (int, float)
            and minimum <= value <= maximum
        )
    return True


def _complete_declared_tool_calls(message: Mapping[str, Any], request: Mapping[str, Any]) -> bool:
    """Admit only finished function decisions whose arguments meet declared tools."""

    calls = message.get("tool_calls")
    tools = request.get("tools")
    if not isinstance(calls, list) or not calls or not isinstance(tools, list):
        return False
    declared: Dict[str, Mapping[str, Any]] = {}
    for tool in tools:
        if not isinstance(tool, Mapping) or tool.get("type") != "function":
            return False
        function = tool.get("function")
        if not isinstance(function, Mapping) or function.get("strict") is not True:
            return False
        name = function.get("name")
        if not isinstance(name, str) or not name or name in declared:
            return False
        declared[name] = function
    forced = request.get("tool_choice")
    if isinstance(forced, Mapping):
        forced_function = forced.get("function")
        if (
            forced.get("type") != "function"
            or not isinstance(forced_function, Mapping)
            or forced_function.get("name") not in declared
        ):
            return False
        forced_name = forced_function["name"]
    elif forced in (None, "auto", "required"):
        forced_name = None
    else:
        return False
    seen_ids = set()
    for call in calls:
        if not isinstance(call, Mapping) or call.get("type") != "function":
            return False
        call_id = call.get("id")
        function = call.get("function")
        if (
            not isinstance(call_id, str) or not call_id or call_id in seen_ids
            or not isinstance(function, Mapping)
        ):
            return False
        seen_ids.add(call_id)
        name = function.get("name")
        arguments = function.get("arguments")
        if name not in declared or (forced_name is not None and name != forced_name):
            return False
        if not isinstance(arguments, str):
            return False
        try:
            parsed = json.loads(arguments)
        except ValueError:
            return False
        if not _matches_declared_verdict_shape(parsed, declared[name].get("parameters")):
            return False
    return True


def _complete_judgment_response(
    operation_id: str,
    request: Mapping[str, Any],
    *,
    status: int,
    body: bytes,
    call_succeeded: bool,
) -> bool:
    """Cache a complete model answer only; never freeze a retryable failure."""

    if not call_succeeded or status != 200:
        return False
    try:
        response = json.loads(body)
    except (TypeError, ValueError):
        return False
    if not isinstance(response, Mapping) or response.get("error") is not None:
        return False
    if operation_id == "openrouter.chat":
        choices = response.get("choices")
        if not isinstance(choices, list) or len(choices) != 1:
            return False
        choice = choices[0]
        if not isinstance(choice, Mapping) or choice.get("finish_reason") not in {"stop", "tool_calls"}:
            return False
        message = choice.get("message")
        if not isinstance(message, Mapping) or message.get("refusal") is not None:
            return False
        if choice.get("finish_reason") == "tool_calls":
            return _complete_declared_tool_calls(message, request)
        content = message.get("content") if isinstance(message, Mapping) else None
        if (
            not isinstance(content, str)
            or not content.strip()
            or message.get("refusal") is not None
        ):
            return False
        response_format = request.get("response_format")
        if response_format is None or (
            isinstance(response_format, Mapping)
            and response_format.get("type") in {"json_object", "json_schema"}
        ):
            try:
                parsed = json.loads(content)
                if (
                    not isinstance(parsed, Mapping)
                    or not parsed
                    or parsed.get("error") is not None
                ):
                    return False
            except ValueError:
                return False
            if isinstance(response_format, Mapping) and response_format.get("type") == "json_schema":
                declared = response_format.get("json_schema")
                if (
                    not isinstance(declared, Mapping)
                    or declared.get("strict") is not True
                    or not _matches_declared_verdict_shape(
                        parsed, declared.get("schema")
                    )
                ):
                    return False
                if declared.get("name") == "verification":
                    evaluations = parsed.get("signal_evaluations")
                    if (
                        not isinstance(evaluations, list)
                        or len(evaluations) != 1
                        or not isinstance(evaluations[0], Mapping)
                        or (
                            evaluations[0].get("signal_status") == "wrong_entity"
                            and evaluations[0].get("same_entity_check") != "fail"
                        )
                    ):
                        return False
        else:
            # Free-form prose and unknown response formats are not replayable.
            return False
        return True
    if operation_id == "openrouter.responses":
        if (
            response.get("status") != "completed"
            or response.get("incomplete_details") is not None
        ):
            return False
        output = response.get("output")
        if not isinstance(output, list) or not output:
            return False
        texts = []
        for item in output:
            if not isinstance(item, Mapping) or item.get("status") not in (None, "completed"):
                return False
            if item.get("type") == "message":
                if item.get("refusal") is not None:
                    return False
                content = item.get("content")
                if not isinstance(content, list):
                    return False
                texts.extend(
                    part.get("text") for part in content
                    if isinstance(part, Mapping) and part.get("type") == "output_text"
                )
        if not texts or any(not isinstance(value, str) or not value.strip() for value in texts):
            return False
        text = request.get("text")
        format_doc = text.get("format") if isinstance(text, Mapping) else None
        if isinstance(format_doc, Mapping) and format_doc.get("type") == "json_schema":
            try:
                parsed = json.loads("".join(texts))
                if not isinstance(parsed, Mapping) or not parsed:
                    return False
            except ValueError:
                return False
            if (
                format_doc.get("strict") is not True
                or not _matches_declared_verdict_shape(
                    parsed, format_doc.get("schema")
                )
            ):
                return False
        else:
            return False
        return True
    return False


def _provider_cost_record(
    cost: provider_costs.ProviderCost, *, operation: str,
    request_id: Optional[str] = None,
) -> Dict[str, Any]:
    record = {
        "basis": cost.price_basis,
        "units": format(cost.units, "f"),
        "unit_name": cost.unit_name,
        "operation": operation,
    }
    if request_id is not None:
        record["request_id"] = request_id
    return record


def _has_successful_terminal_response(document: Any) -> bool:
    # A billing-only settlement may mark the call successful but store a 502
    # placeholder. Only a saved 2xx result can skip provider response recovery.
    return (
        isinstance(document, Mapping)
        and document.get("call_succeeded") is True
        and type(document.get("status")) is int
        and 200 <= document["status"] < 300
    )


def _decode_terminal(
    document: Any,
    *,
    secret: str = "",
) -> Tuple[int, Dict[str, str], bytes]:
    required = {"status", "headers", "body_b64"}
    allowed = required | {
        "call_succeeded",
        "provider_cost",
        "account_failure_evidence",
        "credit_failure_proof",
        "judgment_cache_eligible",
        "judgment_cache_key",
        "judgment_cache_source_call_identity",
        "judgment_cache_source_run_id",
        "deepline_async_job_ids",
        "deepline_response_missing",
    }
    if not isinstance(document, Mapping) or not required <= set(document) <= allowed:
        raise BrokerError("broker_unavailable")
    if "deepline_response_missing" in document and document["deepline_response_missing"] is not True:
        raise BrokerError("broker_unavailable")
    async_ids = document.get("deepline_async_job_ids")
    if async_ids is not None and (
        not isinstance(async_ids, list) or not 1 <= len(async_ids) <= 32
        or any(not isinstance(value, str) or _DEEPLINE_NATIVE_REQUEST_ID_RE.fullmatch(value) is None
               for value in async_ids)
    ):
        raise BrokerError("broker_unavailable")
    provider_cost = document.get("provider_cost")
    if provider_cost is not None and (
        not isinstance(provider_cost, Mapping)
        or set(provider_cost) not in (
            {"basis", "units", "unit_name", "operation"},
            {"basis", "units", "unit_name", "operation", "request_id"},
        )
    ):
        raise BrokerError("broker_unavailable")
    call_succeeded = document.get("call_succeeded")
    if call_succeeded is not None and not isinstance(call_succeeded, bool):
        raise BrokerError("broker_unavailable")
    eligible = document.get("judgment_cache_eligible")
    if eligible is not None and eligible is not True:
        raise BrokerError("broker_unavailable")
    cache_markers = {
        "judgment_cache_key", "judgment_cache_source_call_identity",
        "judgment_cache_source_run_id",
    }
    present_markers = cache_markers & set(document)
    if present_markers:
        if present_markers != cache_markers or eligible is not True:
            raise BrokerError("broker_unavailable")
        for name in ("judgment_cache_key", "judgment_cache_source_call_identity"):
            try:
                contracts.require_sha256(document[name], name)
            except ArenaContractError as exc:
                raise BrokerError("broker_unavailable") from exc
        if not isinstance(document["judgment_cache_source_run_id"], str) or not document["judgment_cache_source_run_id"]:
            raise BrokerError("broker_unavailable")
    _validated_account_failure_evidence(
        document.get("account_failure_evidence")
    )
    _validated_credit_failure_proof(document.get("credit_failure_proof"))
    try:
        status = int(document["status"])
        headers = dict(document["headers"])
        body = base64.b64decode(str(document["body_b64"]), validate=True)
    except (KeyError, TypeError, ValueError) as exc:
        raise BrokerError("broker_unavailable") from exc
    if document.get("credit_failure_proof") is not None and call_succeeded is not False:
        raise BrokerError("broker_unavailable")
    trusted_names = [
        name
        for name in headers
        if isinstance(name, str)
        and name.lower() == operations.TRUSTED_RESPONSE_URL_HEADER
    ]
    if len(trusted_names) > 1:
        raise BrokerError("broker_unavailable")
    if trusted_names:
        name = trusted_names[0]
        headers[operations.TRUSTED_RESPONSE_URL_HEADER] = _validated_response_url(
            headers.pop(name),
            secret=secret,
        )
    return status, headers, body


def _validated_account_failure_evidence(
    account_failure: Any,
) -> Optional[Mapping[str, Any]]:
    if account_failure is None:
        return None
    if (
        not isinstance(account_failure, Mapping)
        or set(account_failure) != {
            "error_class",
            "provider_status",
            "base_call_identity",
            "provider_attempt",
            "action_sequence",
        }
        or account_failure.get("error_class")
        != "account_credential_failure"
        or isinstance(account_failure.get("provider_status"), bool)
        or account_failure.get("provider_status") not in (401, 402, 403, 429)
        or not isinstance(account_failure.get("base_call_identity"), str)
        or _CREDENTIAL_FINGERPRINT_RE.fullmatch(
            account_failure["base_call_identity"]
        ) is None
        or isinstance(account_failure.get("provider_attempt"), bool)
        or account_failure.get("provider_attempt") not in (1, 2, 3, 4)
        or isinstance(account_failure.get("action_sequence"), bool)
        or not isinstance(account_failure.get("action_sequence"), int)
        or account_failure.get("action_sequence") < 0
    ):
        raise BrokerError("broker_unavailable")
    return account_failure


def _account_failure_matches_call(
    evidence: Mapping[str, Any],
    *,
    base_call_identity: str,
    provider_attempt: int,
    action_sequence: int,
) -> bool:
    return (
        evidence.get("base_call_identity") == base_call_identity
        and evidence.get("provider_attempt") == provider_attempt
        and evidence.get("action_sequence") == action_sequence
    )


def _error_result(code: str, call: Mapping[str, Any]) -> BrokerResult:
    """A generic error reply for the model; the call summary keeps the code so
    the worker can tell a refused key or quota from a judge's own failure."""

    body = json.dumps({"error": {"code": code}}, separators=(",", ":")).encode("utf-8")
    return BrokerResult(GENERIC_ERRORS[code], {"content-type": "application/json", "content-length": str(len(body))}, body, dict(call, error_code=code))


def _openrouter_responses_native_error_code(error_type: str) -> str:
    if error_type == "rate_limit_exceeded":
        return "rate_limit_exceeded"
    if error_type in ("context_length_exceeded", "invalid_request", "refusal"):
        return "invalid_prompt"
    if error_type == "content_policy_violation":
        return "image_content_policy_violation"
    return "server_error"


def _openrouter_responses_error_status(
    document: Mapping[str, Any], error: Any
) -> int:
    """Validate one documented Responses failure and return its precise status."""

    error_type = document.get("error_type")
    if (
        not isinstance(error_type, str)
        or error_type not in _OPENROUTER_RESPONSE_ERROR_TYPE_STATUSES
        or document.get("status") != "failed"
        or error is not document.get("error")
        or not isinstance(error, Mapping)
        or error.get("code") != _openrouter_responses_native_error_code(error_type)
        or not isinstance(error.get("message"), str)
        or not error["message"].strip()
    ):
        raise operations.OperationResponseError("invalid_response")
    metadata = error.get("metadata")
    if metadata is not None and not isinstance(metadata, Mapping):
        raise operations.OperationResponseError("invalid_response")
    if isinstance(metadata, Mapping) and "error_type" in metadata and (
        metadata.get("error_type") != error_type
    ):
        raise operations.OperationResponseError("invalid_response")
    return _OPENROUTER_RESPONSE_ERROR_TYPE_STATUSES[error_type]


def _openrouter_effective_response(response: ProviderResponse) -> ProviderResponse:
    """Expose errors which OpenRouter reports inside an HTTP 2xx body.

    OpenRouter can commit the HTTP 200 response before a non-streaming model
    request fails.  In that case the effective status is carried by either a
    top-level error or an error-finished choice.  Reject an unrecognizable
    error envelope rather than passing it to submitted code as a completion.
    """

    if not 200 <= response.status < 300:
        return response
    try:
        document = json.loads(response.body.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise operations.OperationResponseError("invalid_response") from exc
    if not isinstance(document, Mapping):
        return response

    errors = []
    top_level_error = document.get("error")
    if top_level_error is not None:
        errors.append(top_level_error)
    choices = document.get("choices")
    if isinstance(choices, list):
        for choice in choices:
            if isinstance(choice, Mapping) and choice.get("finish_reason") == "error":
                choice_error = choice.get("error")
                if choice_error is not None:
                    errors.append(choice_error)
                elif top_level_error is None:
                    # An error-finished choice needs an error code somewhere
                    # in the documented unified response.
                    errors.append(None)
    if not errors:
        return response

    canonical_error_type_present = "error_type" in document
    if canonical_error_type_present and (
        len(errors) != 1 or errors[0] is not top_level_error
    ):
        raise operations.OperationResponseError("invalid_response")
    statuses = []
    for error in errors:
        if canonical_error_type_present:
            statuses.append(_openrouter_responses_error_status(document, error))
            continue
        code = error.get("code") if isinstance(error, Mapping) else None
        if code == "rate_limit_exceeded":
            code = 429
        if isinstance(code, bool) or not isinstance(code, int) or not 400 <= code <= 599:
            raise operations.OperationResponseError("invalid_response")
        statuses.append(code)
    if len(set(statuses)) != 1:
        raise operations.OperationResponseError("invalid_response")
    return ProviderResponse(statuses[0], response.headers, response.body)


def _provider_call_succeeded(
    provider: str,
    response: ProviderResponse,
    document: Any,
) -> bool:
    """Classify only a valid terminal provider API success.

    Provider text is opaque. Error-looking text selected by a model is still a
    successful completion; only documented protocol fields affect this flag.
    """

    if response.internal_provenance is not None or not 200 <= response.status < 300:
        return False
    if provider == "openrouter":
        if not isinstance(document, Mapping):
            return False
        if document.get("object") == "response":
            return (
                document.get("status") == "completed"
                and document.get("error") is None
                and isinstance(document.get("output"), list)
            )
        choices = document.get("choices")
        return isinstance(choices, list) and all(
            isinstance(choice, Mapping)
            and isinstance(choice.get("finish_reason"), str)
            and choice.get("finish_reason") != "error"
            for choice in choices
        )
    if provider == "deepline":
        return (
            isinstance(document, Mapping)
            and _deepline_job_request_id(document) is not None
            and document.get("status") == "completed"
            and document.get("error") is None
            and document.get("tool_error") is None
        )
    return True


def _openrouter_request_policy_refusal(response: ProviderResponse) -> bool:
    """Identify only documented request-specific OpenRouter policy refusals."""

    if response.status != 403:
        return False
    try:
        document = json.loads(response.body.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        return False
    if not isinstance(document, Mapping):
        return False
    document_error_type_present = "error_type" in document
    document_error_type = document.get("error_type")
    if (
        document_error_type_present
        and (
            not isinstance(document_error_type, str)
            or document_error_type not in _OPENROUTER_RESPONSE_POLICY_ERROR_CODES
        )
    ):
        return False
    errors: List[Any] = []
    top_level_error = document.get("error")
    if top_level_error is not None:
        errors.append(top_level_error)
    choices = document.get("choices")
    if isinstance(choices, list):
        errors.extend(
            choice.get("error")
            for choice in choices
            if isinstance(choice, Mapping) and choice.get("error") is not None
        )
    if not errors:
        return False
    if document_error_type_present and (
        len(errors) != 1 or errors[0] is not top_level_error
    ):
        return False
    for error in errors:
        if not isinstance(error, Mapping):
            return False
        error_code = error.get("code")
        metadata = error.get("metadata")
        if document_error_type_present:
            if isinstance(metadata, Mapping) and "error_type" in metadata and (
                metadata.get("error_type") != document_error_type
            ):
                return False
            return error_code in (
                403,
                _OPENROUTER_RESPONSE_POLICY_ERROR_CODES[document_error_type],
            )
        if error_code is not None and error_code != 403:
            return False
        if not isinstance(metadata, Mapping):
            return False
        if "error_type" in metadata:
            if metadata.get("error_type") in (
                "content_policy_violation",
                "refusal",
            ):
                continue
            return False
        patterns = metadata.get("patterns")
        if (
            error_code == 403
            and isinstance(patterns, list)
            and patterns
            and all(
                isinstance(pattern, str) and pattern.strip()
                for pattern in patterns
            )
        ):
            continue
        reasons = metadata.get("reasons")
        if not (
            isinstance(reasons, list)
            and all(isinstance(reason, str) for reason in reasons)
            and isinstance(metadata.get("flagged_input"), str)
            and isinstance(metadata.get("provider_name"), str)
            and metadata["provider_name"].strip()
            and isinstance(metadata.get("model_slug"), str)
            and metadata["model_slug"].strip()
        ):
            return False
    return True


def _deepline_generic_http_request_refusal(
    parameters: Mapping[str, Any], response: ProviderResponse
) -> bool:
    """Identify the observed per-request Deepline generic HTTP denial."""

    if response.status != 403 or parameters.get("tool") != "generic_http_request":
        return False
    try:
        document = json.loads(response.body.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        return False
    if not isinstance(document, Mapping):
        return False
    tool_error = document.get("tool_error")
    return (
        document.get("code") == "PROVIDER_AUTHORIZATION_FAILED"
        and document.get("provider") == "generic_http"
        and document.get("operation") == "generic_http_request"
        and document.get("upstream_status") == 403
        and isinstance(tool_error, Mapping)
        and tool_error.get("code") == "PROVIDER_AUTHORIZATION_FAILED"
        and tool_error.get("statusCode") == 403
        and tool_error.get("provider") == "generic_http"
        and tool_error.get("operation") == "generic_http_request"
    )


def _deepline_firecrawl_enrichment_refusal(
    parameters: Mapping[str, Any], response: ProviderResponse
) -> bool:
    """Identify the observed managed-provider denial for LinkedIn enrichment."""

    if response.status != 403 or parameters.get("tool") != "firecrawl_scrape":
        return False
    try:
        document = json.loads(response.body.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        return False
    if not isinstance(document, Mapping):
        return False
    tool_error = document.get("tool_error")
    return (
        document.get("code") == "PROVIDER_AUTHORIZATION_FAILED"
        and document.get("credential_owner") == "deepline_managed"
        and document.get("credential_source") == "env"
        and document.get("error_category") == "provider_auth"
        and document.get("failure_origin") == "provider"
        and document.get("provider") == "firecrawl"
        and document.get("operation") == "firecrawl_scrape"
        and type(document.get("upstream_status")) is int
        and document["upstream_status"] == 403
        and document.get("upstream_error_code")
        == "THIRD_PARTY_DATA_ENRICHMENT_NOT_ENABLED"
        and isinstance(tool_error, Mapping)
        and tool_error.get("code") == "PROVIDER_AUTHORIZATION_FAILED"
        and tool_error.get("category") == "authentication"
        and tool_error.get("origin") == "provider"
        and type(tool_error.get("statusCode")) is int
        and tool_error["statusCode"] == 403
        and tool_error.get("provider") == "firecrawl"
        and tool_error.get("operation") == "firecrawl_scrape"
    )


def _provider_request_refused(
    provider: str,
    parameters: Mapping[str, Any],
    response: ProviderResponse,
) -> bool:
    if provider == "openrouter":
        return _openrouter_request_policy_refusal(response)
    if provider == "deepline":
        return (
            _deepline_generic_http_request_refusal(parameters, response)
            or _deepline_firecrawl_enrichment_refusal(parameters, response)
        )
    return False


def _miner_credential_failure(
    provider: str,
    response: ProviderResponse,
    *,
    champion_credential_retry: bool,
) -> bool:
    """Return true only for a failure tied to the submitted account.

    OpenRouter can relay an upstream provider 429 or reject one request under
    content policy through the submitted account. Its documented metadata
    distinguishes those responses from miner account failures. Unqualified
    403 and 429 responses remain account failures.
    """

    if response.status in (401, 402):
        return True
    if response.status == 403:
        return not (
            provider == "openrouter"
            and _openrouter_request_policy_refusal(response)
        )
    if response.status != 429 or not champion_credential_retry:
        return False
    if provider != "openrouter":
        return True
    try:
        document = json.loads(response.body.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        return True
    if not isinstance(document, Mapping):
        return True
    errors: List[Any] = []
    if document.get("error") is not None:
        errors.append(document.get("error"))
    choices = document.get("choices")
    if isinstance(choices, list):
        errors.extend(
            choice.get("error")
            for choice in choices
            if isinstance(choice, Mapping) and choice.get("error") is not None
        )
    for error in errors:
        metadata = error.get("metadata") if isinstance(error, Mapping) else None
        if not isinstance(metadata, Mapping):
            continue
        provider_name = metadata.get("provider_name")
        if isinstance(provider_name, str) and provider_name.strip():
            return False
        provider_responses = metadata.get("provider_responses")
        if isinstance(provider_responses, list) and any(
            isinstance(item, Mapping)
            and isinstance(item.get("provider_name"), str)
            and item["provider_name"].strip()
            and item.get("status") == 429
            for item in provider_responses
        ):
            return False
    return True


class Broker:
    """Section 7.5 state machine over the ledger functions."""

    def __init__(
        self,
        *,
        store: CallStore,
        key_for: Callable[[str], str],
        price_table: Mapping[str, Any],
        judge_models: Sequence[str] = (),
        transport: ProviderTransport,
        clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
        lease_ttl_seconds: int = contracts.LEASE_TTL_SECONDS,
        credential_for: Optional[Callable[[RunContext, str], str]] = None,
        funding_source_for: Optional[Callable[[RunContext], str]] = None,
        provider_funding_source_for: Optional[
            Callable[[RunContext, str], str]
        ] = None,
        retry_miner_credential_for: Optional[
            Callable[[RunContext], bool]
        ] = None,
        mark_provider_fallback: Optional[
            Callable[[RunContext, str, Mapping[str, Any]], Mapping[str, Any]]
        ] = None,
        provider_restart_required_for: Optional[
            Callable[[RunContext, str], bool]
        ] = None,
        openrouter_shared_gate: Optional[OpenRouterSharedGate] = None,
    ) -> None:
        self._store = store
        # Host-only callers retain key_for. Production supplies the scoped
        # resolver for both model execution and judging. It must never fall
        # back to a host key for a miner submission.
        self._key_for = key_for
        self._credential_for = credential_for
        self._funding_source_for = funding_source_for
        self._provider_funding_source_for = provider_funding_source_for
        self._retry_miner_credential_for = retry_miner_credential_for
        self._mark_provider_fallback = mark_provider_fallback
        self._provider_restart_required_for = provider_restart_required_for
        self._openrouter_shared_gate = openrouter_shared_gate
        self._price_table = validate_price_table(price_table)
        # Judge models are what scoring runs may call; they are pinned by the
        # scorer policy and priced from the same table.
        self._judge_models = tuple(str(model) for model in judge_models)
        for model in self._judge_models:
            if model not in self._price_table["models"]:
                raise ArenaContractError("judge model %s is missing from the price table" % model)
        self._transport = transport
        self._clock = clock
        self._lease_ttl_seconds = int(lease_ttl_seconds)

    # -- helpers ------------------------------------------------------------

    def _timestamp(self) -> str:
        return self._clock().astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    def _openrouter_parameters(self, parameters: Mapping[str, Any], *, kind: str = "execute") -> Tuple[Dict[str, Any], int]:
        model = str(parameters.get("model") or "")
        if kind == "score" and model not in self._judge_models:
            raise BrokerError("model_not_allowed")
        if model not in self._price_table["models"]:
            raise BrokerError("model_not_allowed")
        normalized = openrouter_normalized(parameters)
        return normalized, int(normalized.get("max_output_tokens", normalized.get("max_tokens")))

    def reconcile_openrouter_cost(
        self,
        candidate: Mapping[str, Any],
        *,
        timeout_seconds: float = OPENROUTER_DELAYED_RECONCILIATION_TIMEOUT_SECONDS,
    ) -> Dict[str, Any]:
        """Settle one retained exact generation without repeating its POST."""

        required = {
            "uncertain_entry_id",
            "round_id",
            "run_id",
            "submission_id",
            "miner_hotkey",
            "assignment_id",
            "stage",
            "icp_position",
            "attempt",
            "kind",
            "call_identity",
            "generation_id",
            "credential_fingerprint",
            "funding_source",
            "run_status",
            "lease_expires_at",
        }
        if not isinstance(candidate, Mapping) or set(candidate) != required:
            return {"status": "invalid"}
        generation_id = candidate.get("generation_id")
        credential_fingerprint = candidate.get("credential_fingerprint")
        if (
            not isinstance(generation_id, str)
            or _OPENROUTER_GENERATION_ID_RE.fullmatch(generation_id) is None
            or not isinstance(credential_fingerprint, str)
            or _CREDENTIAL_FINGERPRINT_RE.fullmatch(credential_fingerprint) is None
            or isinstance(candidate.get("uncertain_entry_id"), bool)
            or not isinstance(candidate.get("uncertain_entry_id"), int)
            or int(candidate["uncertain_entry_id"]) < 1
            or candidate.get("kind") not in ("execute", "score")
            or candidate.get("funding_source") not in ("host", "miner_key")
            or not 0 < float(timeout_seconds) <= OPENROUTER_DELAYED_RECONCILIATION_TIMEOUT_SECONDS
        ):
            return {"status": "invalid"}
        try:
            context = RunContext(
                run_id=str(candidate["run_id"]),
                assignment_id=str(candidate["assignment_id"]),
                icp_position=int(candidate["icp_position"]),
                lease_token_hash="",
                miner_hotkey=str(candidate["miner_hotkey"]),
                submission_id=str(candidate["submission_id"]),
                stage=int(candidate["stage"]),
                kind=str(candidate["kind"]),
                attempt=int(candidate["attempt"]),
                round_id=str(candidate["round_id"]),
            )
            funding_source = (
                self._provider_funding_source_for(context, "openrouter")
                if self._provider_funding_source_for
                else self._funding_source_for(context)
                if self._funding_source_for
                else "host"
            )
            secret = (
                self._credential_for(context, "openrouter")
                if self._credential_for
                else self._key_for("openrouter")
            )
            if (
                not isinstance(secret, str)
                or not secret
                or funding_source != candidate["funding_source"]
                or not hmac.compare_digest(
                    _credential_fingerprint(secret), credential_fingerprint
                )
            ):
                return {"status": "credential_mismatch"}
            state, cost = _openrouter_generation_readback_once(
                transport=self._transport,
                secret=secret,
                generation_id=generation_id,
                timeout_seconds=float(timeout_seconds),
            )
        except (BrokerError, KeyError, TypeError, ValueError):
            return {"status": "unavailable"}
        finally:
            if "secret" in locals():
                secret = ""
                del secret
        if cost is None:
            return {"status": "pending" if state == "retry" else "unavailable"}
        return self._store.reconcile_openrouter_cost(
            round_id=str(candidate["round_id"]),
            run_id=str(candidate["run_id"]),
            call_identity=str(candidate["call_identity"]),
            uncertain_entry_id=int(candidate["uncertain_entry_id"]),
            generation_id=generation_id,
            credential_fingerprint=credential_fingerprint,
            actual_microusd=cost.microusd,
            cost_units=format(cost.units, "f"),
        )

    def reconcile_deepline_cost(
        self,
        candidate: Mapping[str, Any],
        *,
        timeout_seconds: float = DEEPLINE_DELAYED_RECONCILIATION_TIMEOUT_SECONDS,
    ) -> Dict[str, Any]:
        """Settle one retained Deepline request without repeating its POST."""

        required = {
            "uncertain_entry_id",
            "round_id",
            "run_id",
            "submission_id",
            "miner_hotkey",
            "assignment_id",
            "stage",
            "icp_position",
            "attempt",
            "kind",
            "call_identity",
            "request_id",
            "credential_fingerprint",
            "funding_source",
            "run_status",
            "lease_expires_at",
            "uncertain_at",
            "reservation_at",
            "operation",
        }
        optional = {"execution_key", "billing_provider", "operation_aliases"}
        if (not isinstance(candidate, Mapping) or not required <= set(candidate)
            or not set(candidate) <= required | optional):
            return {"status": "invalid"}
        try:
            timeout = float(timeout_seconds)
        except (TypeError, ValueError, OverflowError):
            return {"status": "invalid"}
        operation = candidate.get("operation")
        request_id = candidate.get("request_id")
        execution_key = candidate.get("execution_key")
        billing_provider = candidate.get("billing_provider")
        operation_aliases = candidate.get("operation_aliases") or ()
        credential_fingerprint = candidate.get("credential_fingerprint")
        if (
            not isinstance(candidate.get("call_identity"), str)
            or _CREDENTIAL_FINGERPRINT_RE.fullmatch(candidate["call_identity"]) is None
            or not isinstance(operation, str)
            or not operation
            or not isinstance(request_id, str)
            or (
                _DEEPLINE_BILLING_REQUEST_ID_RE.fullmatch(request_id) is None
                and ((execution_key is None and billing_provider is None)
                     or _DEEPLINE_NATIVE_REQUEST_ID_RE.fullmatch(request_id) is None)
            )
            or (
                request_id.startswith("ctx-tool-")
                and request_id != "ctx-tool-" + candidate["call_identity"][7:39]
            )
            or not isinstance(credential_fingerprint, str)
            or _CREDENTIAL_FINGERPRINT_RE.fullmatch(credential_fingerprint) is None
            or isinstance(candidate.get("uncertain_entry_id"), bool)
            or not isinstance(candidate.get("uncertain_entry_id"), int)
            or int(candidate["uncertain_entry_id"]) < 1
            or candidate.get("kind") not in ("execute", "score")
            or candidate.get("funding_source") not in ("host", "miner_key")
            or not 0 < timeout <= DEEPLINE_DELAYED_RECONCILIATION_TIMEOUT_SECONDS
            or (
                execution_key is not None
                and execution_key != "arena:" + candidate["call_identity"][7:]
            )
            or (billing_provider is not None and (
                not isinstance(billing_provider, str)
                or re.fullmatch(r"[A-Za-z0-9_-]{1,100}", billing_provider) is None
            ))
            or not isinstance(operation_aliases, (list, tuple))
            or len(operation_aliases) > 100
            or any(not isinstance(alias, str) or re.fullmatch(r"[A-Za-z0-9_-]{1,200}", alias) is None
                   for alias in operation_aliases)
        ):
            return {"status": "invalid"}
        try:
            context = RunContext(
                run_id=str(candidate["run_id"]),
                assignment_id=str(candidate["assignment_id"]),
                icp_position=int(candidate["icp_position"]),
                lease_token_hash="",
                miner_hotkey=str(candidate["miner_hotkey"]),
                submission_id=str(candidate["submission_id"]),
                stage=int(candidate["stage"]),
                kind=str(candidate["kind"]),
                attempt=int(candidate["attempt"]),
                round_id=str(candidate["round_id"]),
            )
            funding_source = (
                self._provider_funding_source_for(context, "deepline")
                if self._provider_funding_source_for
                else self._funding_source_for(context)
                if self._funding_source_for
                else "host"
            )
            secret = (
                self._credential_for(context, "deepline")
                if self._credential_for
                else self._key_for("deepline")
            )
            if (
                not isinstance(secret, str)
                or not secret
                or funding_source != candidate["funding_source"]
                or not hmac.compare_digest(
                    _credential_fingerprint(secret), credential_fingerprint
                )
            ):
                return {"status": "credential_mismatch"}
            reservation_at = datetime.fromisoformat(
                str(candidate["reservation_at"]).replace("Z", "+00:00")
            )
            if reservation_at.tzinfo is None:
                return {"status": "invalid"}
            recovered_request_id = None
            if execution_key is not None or billing_provider is not None:
                recovered_request_id, cost = _deepline_exact_readback(
                    transport=self._transport, secret=secret,
                    request_id=request_id, execution_key=execution_key,
                    operation=operation,
                    provider=billing_provider, operation_aliases=operation_aliases,
                    reconciliation_deadline=time.monotonic() + timeout,
                )
            else:
                # Historical reservations predate execution keys and retain
                # their authenticated, request-bound ledger recovery path.
                if operation not in operations.DEEPLINE_TOOLS:
                    return {"status": "invalid"}
                cost = _deepline_ledger_readback(
                    transport=self._transport, secret=secret,
                    request_id=request_id, operation=operation,
                    since_at=max(0, int(reservation_at.timestamp() * 1000) - 1000),
                    reconciliation_deadline=time.monotonic() + timeout,
                )
        except (BrokerError, KeyError, TypeError, ValueError):
            return {"status": "unavailable"}
        finally:
            if "secret" in locals():
                secret = ""
                del secret
        if cost is None:
            return {"status": "pending"}
        return self._store.reconcile_deepline_cost(
            round_id=str(candidate["round_id"]),
            run_id=str(candidate["run_id"]),
            call_identity=str(candidate["call_identity"]),
            uncertain_entry_id=int(candidate["uncertain_entry_id"]),
            request_id=request_id,
            operation=operation,
            credential_fingerprint=credential_fingerprint,
            actual_microusd=cost.microusd,
            cost_units=format(cost.units, "f"),
            **({"execution_key": execution_key,
                "recovered_request_id": recovered_request_id}
               if execution_key is not None or billing_provider is not None else {}),
        )

    def _catalog_discovery(
        self, context: RunContext, operation: operations.Operation,
        parameters: Mapping[str, Any], action_sequence: int,
    ) -> BrokerResult:
        """Return frozen public metadata after a read-only active-lease check."""

        summary = {"operation_id": operation.operation_id, "provider": "deepline",
                   "action_sequence": action_sequence, "outcome": "local_catalog",
                   "actual_microusd": 0}
        try:
            if context.deepline_catalog is None:
                return _error_result("invalid_request", summary)
            snapshot = self._store.run_quota_snapshot(context.run_id, context.lease_token_hash)
            if not isinstance(snapshot, Mapping) or snapshot.get("status") == "stale":
                return _error_result("lease_stale", summary)
            catalog = deepline_catalog.validate_catalog(context.deepline_catalog)
            if operation.operation_id == "deepline.tools.get":
                document = deepline_catalog.public_tool_definition(
                    deepline_catalog.tool_entry(catalog, parameters["tool"])
                )
            else:
                rows = catalog["tools"]
                categories = parameters.get("categories")
                if isinstance(categories, str) and categories:
                    requested = {value.strip().casefold() for value in categories.split(",") if value.strip()}
                    rows = [row for row in rows if requested & {str(value).casefold() for value in row["categories"]}]
                if operation.operation_id == "deepline.tools.search":
                    query = str(parameters.get("query", "")).casefold()
                    rows = [row for row in rows if query in (
                        row["tool_id"] + " " + row["description"] + " " + " ".join(row["categories"])
                    ).casefold()]
                document = {"tools": [deepline_catalog.public_tool_definition(
                    row, compact=bool(parameters.get("compact", True))) for row in rows],
                            "total": len(rows)}
            body = json.dumps(document, separators=(",", ":"), allow_nan=False).encode("utf-8")
            if len(body) > operation.max_response_bytes:
                return _error_result("provider_unavailable", summary)
            summary["catalog_hash"] = catalog["catalog_hash"]
            return BrokerResult(200, {"content-type": "application/json"}, body, summary)
        except (ArenaStoreError, deepline_catalog.CatalogError, KeyError, TypeError, ValueError):
            return _error_result("broker_unavailable", summary)

    def _owns_deepline_async_job(
        self, context: RunContext, entry: Mapping[str, Any], payload: Mapping[str, Any],
    ) -> bool:
        """Authorize a declared read-only poll from this run's durable start receipt."""

        parent = entry.get("async_parent")
        if not isinstance(parent, str) or context.deepline_catalog is None:
            return False
        start = deepline_catalog.tool_entry(context.deepline_catalog, parent)
        flow = start.get("async_flow")
        if not isinstance(flow, Mapping) or entry["tool_id"] not in flow.get("poll_actions", ()):
            return False
        poll_input = flow.get("poll_input")
        # Catalog policy must name exactly one required job-ID parameter.
        # Provider paging URLs cannot be proved owned by a job ID alone.
        if not isinstance(poll_input, str) or "next" in payload:
            return False
        job_id = payload.get(poll_input)
        if not isinstance(job_id, str) or _DEEPLINE_NATIVE_REQUEST_ID_RE.fullmatch(job_id) is None:
            return False
        owned_calls = set()
        after_entry_id = 0
        for _ in range(32):
            rows = self._store.list_ledger(
                run_id=context.run_id, provider="deepline", operation_id="deepline.execute",
                limit=1000, after_entry_id=after_entry_id,
            )
            if not isinstance(rows, list):
                return False
            for row in rows:
                if (not isinstance(row, Mapping) or row.get("run_id") != context.run_id
                    or row.get("provider") != "deepline" or row.get("operation_id") != "deepline.execute"
                    or type(row.get("entry_id")) is not int or row["entry_id"] <= after_entry_id):
                    return False
                after_entry_id = row["entry_id"]
                call_doc = row.get("entry_doc")
                if not isinstance(call_doc, Mapping):
                    continue
                identity = row.get("call_identity")
                if row.get("entry_kind") == "reservation" and call_doc.get("tool") == parent:
                    owned_calls.add(identity)
                if identity not in owned_calls:
                    continue
                if isinstance(call_doc.get("call"), Mapping):
                    call_doc = call_doc["call"]
                ids = call_doc.get("deepline_async_job_ids")
                if isinstance(ids, list) and job_id in ids:
                    return True
                terminal = row.get("terminal_response")
                if not isinstance(terminal, Mapping):
                    continue
                if job_id in terminal.get("deepline_async_job_ids", []):
                    return True
                encoded = terminal.get("body_b64")
                if not isinstance(encoded, str) or len(encoded) > 3 * 1024 * 1024:
                    continue
                try:
                    document = json.loads(base64.b64decode(encoded, validate=True))
                except (UnicodeDecodeError, ValueError):
                    continue
                if job_id in _deepline_async_job_ids(document, flow):
                    return True
            if len(rows) < 1000:
                break
        return False

    # -- execution ------------------------------------------------------------

    def execute(
        self,
        context: RunContext,
        *,
        operation_id: str,
        parameters: Mapping[str, Any],
        action_sequence: int,
        timeout_ms: int,
        cancel_requested: Optional[Callable[[], bool]] = None,
    ) -> BrokerResult:
        """Execute one model action with bounded champion credential retries.

        Ordinary miner, scoring, and organizer calls still make one provider
        attempt.  A daily champion baseline execution may make the initial
        miner-funded call plus three retries.  Every retry keeps the same
        request and action sequence but receives a distinct ledger identity.
        """

        api_started_at = time.monotonic()
        try:
            retry_miner_credential = bool(
                self._retry_miner_credential_for(context)
            ) if self._retry_miner_credential_for else False
        except (BrokerError, KeyError, TypeError, ValueError):
            return _error_result(
                "broker_unavailable", {"operation_id": str(operation_id)}
            )
        last_result: Optional[BrokerResult] = None
        attempt_trace: list[Dict[str, Any]] = []

        def final_result(result: BrokerResult) -> BrokerResult:
            """Attach private retry metadata without changing the worker document."""

            if len(attempt_trace) <= 1:
                return result
            try:
                return BrokerResult(
                    result.status,
                    result.headers,
                    result.body,
                    result.call,
                    tuple(dict(item) for item in attempt_trace[:4]),
                )
            except Exception:
                return result

        account_provider_status: Optional[int] = None
        base_call_identity: Optional[str] = None
        for provider_attempt in range(1, CHAMPION_CREDENTIAL_PROVIDER_ATTEMPTS + 1):
            provider_attempt_started = time.monotonic()
            try:
                result = self._execute_once(
                    context,
                    operation_id=operation_id,
                    parameters=parameters,
                    action_sequence=action_sequence,
                    timeout_ms=timeout_ms,
                    provider_attempt=provider_attempt,
                    champion_credential_retry=retry_miner_credential,
                    api_started_at=api_started_at,
                    cancel_requested=cancel_requested,
                )
            except Exception as exc:
                try:
                    failed_attempt = {
                        "occurred_at": datetime.now(timezone.utc).isoformat(
                            timespec="milliseconds"
                        ).replace("+00:00", "Z"),
                        "elapsed_ms": int(
                            (time.monotonic() - provider_attempt_started) * 1000
                        ),
                        "provider_attempt": provider_attempt,
                        "operation_id": str(operation_id),
                        "error_class": type(exc).__name__,
                    }
                    setattr(
                        exc,
                        "_arena_trajectory_attempts",
                        tuple([*attempt_trace, failed_attempt][-4:]),
                    )
                except Exception:
                    # Observation must not replace the provider exception.
                    pass
                raise
            last_result = result
            try:
                attempt_trace.append(
                    _provider_attempt_summary(
                        result,
                        provider_attempt=provider_attempt,
                        occurred_at=datetime.now(timezone.utc).isoformat(
                            timespec="milliseconds"
                        ).replace("+00:00", "Z"),
                        elapsed_ms=int(
                            (time.monotonic() - provider_attempt_started) * 1000
                        ),
                    )
                )
            except Exception:
                # Observation must not affect provider retry or result behavior.
                pass
            if result.call.get("provider_fallback_required") is True:
                return final_result(result)
            if result.call.get("provider") == "deepline" and result.call.get("outcome") == "uncertain":
                # An unresolved first charge cannot authorize another paid
                # attempt under a different execution key or credential.
                return final_result(result)
            if not (
                retry_miner_credential
                and result.call.get("funding_source") == "miner_key"
                and result.call.get("error_code")
                == "miner_credentials_unavailable"
            ):
                return final_result(result)
            if result.call.get("provider_status") in (401, 402, 403, 429):
                account_provider_status = int(result.call["provider_status"])
            if isinstance(result.call.get("base_call_identity"), str):
                base_call_identity = result.call["base_call_identity"]
            if provider_attempt < CHAMPION_CREDENTIAL_PROVIDER_ATTEMPTS:
                continue
            provider = str(result.call.get("provider") or "")
            if not provider or self._mark_provider_fallback is None:
                return final_result(BrokerResult(
                    result.status,
                    result.headers,
                    result.body,
                    dict(
                        result.call,
                        champion_credential_attempts=provider_attempt,
                        provider_fallback_required=True,
                    ),
                ))
            evidence = {
                "error_class": "account_credential_failure",
                "provider_status": account_provider_status,
                "action_sequence": action_sequence,
                "provider_attempts": provider_attempt,
                "base_call_identity": base_call_identity,
            }
            try:
                marked = self._mark_provider_fallback(
                    context, provider, evidence
                )
            except Exception:
                return final_result(_error_result(
                    "broker_unavailable",
                    dict(
                        result.call,
                        champion_credential_attempts=provider_attempt,
                        provider_fallback_required=True,
                    ),
                ))
            if not isinstance(marked, Mapping) or marked.get("status") not in (
                "marked",
                "existing",
            ):
                code = "lease_stale" if (
                    isinstance(marked, Mapping)
                    and marked.get("status") == "stale"
                ) else "broker_unavailable"
                return final_result(_error_result(
                    code,
                    dict(
                        result.call,
                        champion_credential_attempts=provider_attempt,
                        provider_fallback_required=True,
                    ),
                ))
            return final_result(BrokerResult(
                result.status,
                result.headers,
                result.body,
                dict(
                    result.call,
                    champion_credential_attempts=provider_attempt,
                    provider_fallback_required=True,
                    provider_fallback_marked=True,
                ),
            ))
        assert last_result is not None
        return final_result(last_result)

    def _execute_once(
        self,
        context: RunContext,
        *,
        operation_id: str,
        parameters: Mapping[str, Any],
        action_sequence: int,
        timeout_ms: int,
        provider_attempt: int,
        champion_credential_retry: bool,
        api_started_at: Optional[float] = None,
        cancel_requested: Optional[Callable[[], bool]] = None,
    ) -> BrokerResult:
        if api_started_at is None:
            api_started_at = time.monotonic()
        gate_holder: list[OpenRouterGateLease] = []
        try:
            return self._execute_once_impl(
                context,
                operation_id=operation_id,
                parameters=parameters,
                action_sequence=action_sequence,
                timeout_ms=timeout_ms,
                provider_attempt=provider_attempt,
                champion_credential_retry=champion_credential_retry,
                api_started_at=api_started_at,
                cancel_requested=cancel_requested,
                gate_holder=gate_holder,
            )
        finally:
            if gate_holder:
                gate_holder[0].release()

    def _execute_once_impl(
        self,
        context: RunContext,
        *,
        operation_id: str,
        parameters: Mapping[str, Any],
        action_sequence: int,
        timeout_ms: int,
        provider_attempt: int,
        champion_credential_retry: bool,
        api_started_at: float,
        cancel_requested: Optional[Callable[[], bool]],
        gate_holder: list[OpenRouterGateLease],
    ) -> BrokerResult:
        operation = (operations.OPERATIONS.get(operation_id)
                     or getattr(operations, "CATALOG_OPERATIONS", {}).get(operation_id))
        if operation is None:
            return _error_result("invalid_request", {"operation_id": str(operation_id)})
        try:
            normalized = operations.validate_operation_request(
                operation_id, parameters, deepline_catalog=context.deepline_catalog,
            )
        except operations.OperationError:
            return _error_result("invalid_request", {"operation_id": operation_id})
        if isinstance(action_sequence, bool) or not isinstance(action_sequence, int) or action_sequence < 0:
            return _error_result("invalid_request", {"operation_id": operation_id})
        if operation_id in {"deepline.tools.list", "deepline.tools.get", "deepline.tools.search"}:
            return self._catalog_discovery(context, operation, normalized, action_sequence)
        funding_source = "host"
        try:
            funding_source = (
                self._provider_funding_source_for(context, operation.provider)
                if self._provider_funding_source_for
                else self._funding_source_for(context)
                if self._funding_source_for
                else "host"
            )
            if funding_source not in ("host", "miner_key"):
                raise BrokerError("broker_unavailable")
            if (
                champion_credential_retry
                and funding_source == "miner_key"
                and self._provider_restart_required_for is not None
                and self._provider_restart_required_for(
                    context, operation.provider
                )
            ):
                if self._mark_provider_fallback is None:
                    raise BrokerError("broker_unavailable")
                marked = self._mark_provider_fallback(
                    context,
                    operation.provider,
                    {
                        "error_class": "account_credential_failure",
                        "provider_status": None,
                        "action_sequence": action_sequence,
                        "provider_attempts": (
                            CHAMPION_CREDENTIAL_PROVIDER_ATTEMPTS
                        ),
                        "base_call_identity": None,
                    },
                )
                if not isinstance(marked, Mapping) or marked.get(
                    "status"
                ) not in ("marked", "existing"):
                    raise BrokerError("broker_unavailable")
                return _error_result(
                    "miner_credentials_unavailable",
                    {
                        "operation_id": operation_id,
                        "provider": operation.provider,
                        "funding_source": funding_source,
                        "provider_attempt": provider_attempt,
                        "provider_fallback_required": True,
                        "provider_fallback_marked": True,
                    },
                )
            route = scoring_provider_compat.route_for(
                kind=getattr(context, "kind", "execute"),
                funding_source=funding_source,
                round_id=getattr(context, "round_id", ""),
                operation_id=operation_id,
                parameters=normalized,
                timeout_ms=min(
                    max(1, int(timeout_ms)), operation.timeout_seconds * 1000
                ),
            )
            effective_operation_id = route.effective_operation_id if route else operation_id
            effective_parameters = route.effective_parameters if route else normalized
            effective_operation = operations.OPERATIONS[effective_operation_id]
            effective_normalized = operations.validate_operation_request(
                effective_operation_id, effective_parameters, deepline_catalog=context.deepline_catalog,
            )
            secret = self._credential_for(context, effective_operation.provider) if self._credential_for else self._key_for(effective_operation.provider)
            if not isinstance(secret, str) or not secret:
                raise BrokerError("miner_credentials_unavailable" if funding_source == "miner_key" else "broker_unavailable")
        except (BrokerError, KeyError, operations.OperationError) as exc:
            if not isinstance(exc, BrokerError):
                exc = BrokerError("broker_unavailable")
            return _error_result(
                exc.code,
                {
                    "operation_id": operation_id,
                    "provider": operation.provider,
                    "funding_source": funding_source,
                    "provider_attempt": provider_attempt,
                },
            )
        provider_credential_fingerprint = (
            _credential_fingerprint(secret)
            if effective_operation.provider in ("openrouter", "deepline")
            else None
        )
        openrouter_credential_fingerprint = (
            provider_credential_fingerprint
            if effective_operation.provider == "openrouter" else None
        )
        max_output_tokens = 0
        deepline_catalog_entry = None
        if effective_operation.provider == "deepline" and context.deepline_catalog is not None:
            tool = effective_normalized.get("tool") or effective_operation.deepline_tool
            deepline_catalog_entry = deepline_catalog.tool_entry(context.deepline_catalog, tool)
            if deepline_catalog_entry.get("async_parent"):
                try:
                    if not self._owns_deepline_async_job(context, deepline_catalog_entry, effective_normalized["payload"]):
                        return _error_result("invalid_request", {"operation_id": operation_id})
                except (ArenaStoreError, deepline_catalog.CatalogError, KeyError, TypeError, ValueError):
                    return _error_result("broker_unavailable", {"operation_id": operation_id})
        reservation_cost: Optional[provider_costs.ProviderCost] = None
        reserve_remaining_budget = False
        openrouter_host_route: Optional[OpenRouterHostRoute] = None
        try:
            if effective_operation.provider == "openrouter":
                # Record the request's price bound; current SQL admission uses
                # confirmed spend and stores a zero-dollar lifecycle record.
                effective_normalized, max_output_tokens = self._openrouter_parameters(effective_normalized, kind=getattr(context, "kind", "execute"))
                normalized = effective_normalized
                # Server-side search can add provider-owned context and model
                # passes that are absent from the caller body. Keep this
                # dynamic-price marker for historical reservation policies;
                # current SQL admission does not hold the remaining budget.
                reserve_remaining_budget = _openrouter_web_search_enabled(
                    normalized
                )
                pricing = self._price_table["models"][normalized["model"]]
                openrouter_host_route = _openrouter_host_route(
                    kind=getattr(context, "kind", "execute"),
                    operation_id=effective_operation_id,
                    model=normalized["model"],
                    pricing=pricing,
                    parameters=normalized,
                )
                amount = _max_openrouter_cost_for_pricing(
                    (
                        openrouter_host_route.reservation_pricing
                        if openrouter_host_route is not None
                        else pricing
                    ),
                    normalized,
                    max_output_tokens=max_output_tokens,
                )
            elif effective_operation.provider == "scrapingdog":
                reservation_cost = provider_costs.scrapingdog_cost(
                    effective_operation_id, effective_normalized
                )
                amount = reservation_cost.microusd
            elif effective_operation.provider == "deepline":
                reservation_cost = provider_costs.deepline_reservation_cost(
                    effective_normalized, catalog_entry=deepline_catalog_entry,
                )
                reserve_remaining_budget = reservation_cost is None
                amount = 0 if reservation_cost is None else reservation_cost.microusd
            else:
                raise BrokerError("invalid_request")
        except BrokerError as exc:
            return _error_result(exc.code, {"operation_id": operation_id})
        request_hash = contracts.document_hash(normalized)
        base_call_identity = contracts.provider_call_identity(
            assignment_id=context.assignment_id,
            attempt=int(getattr(context, "attempt", 1)),
            icp_position=context.icp_position,
            action_sequence=action_sequence,
            operation_id=operation_id,
            request_hash=request_hash,
        )
        call_identity = base_call_identity
        if provider_attempt > 1:
            call_identity = contracts.document_hash(
                {
                    "base_call_identity": call_identity,
                    "provider_attempt": provider_attempt,
                }
            )
        summary: Dict[str, Any] = {
            "call_identity": call_identity,
            "base_call_identity": base_call_identity,
            "operation_id": operation_id,
            "provider": effective_operation.provider,
            "funding_source": funding_source,
            "request_hash": request_hash,
            "reserved_microusd": amount,
            "action_sequence": action_sequence,
            "provider_attempt": provider_attempt,
        }
        if reservation_cost is not None:
            summary.update(
                {
                    "cost_basis": reservation_cost.price_basis,
                    "cost_units": format(reservation_cost.units, "f"),
                    "cost_unit_name": reservation_cost.unit_name,
                }
            )
        if route is not None:
            summary.update(route.summary())
        request_accounting = {}
        if effective_operation.provider == "openrouter":
            request_accounting["model"] = effective_normalized["model"]
        elif effective_operation.provider == "deepline":
            request_accounting["tool"] = effective_normalized.get("tool") or {
                "exa.search": "exa_search", "exa.contents": "exa_contents"
            }.get(effective_operation_id, "")
        if effective_operation.provider == "deepline":
            # Bind the local attempt before dispatch. The public API may use a
            # different native billing id, retained from its response below.
            # Models cannot supply or override this header.
            request_accounting["deepline_request_id"] = (
                "ctx-tool-" + call_identity.removeprefix("sha256:")[:32]
            )
            request_accounting["credential_fingerprint"] = (
                provider_credential_fingerprint
            )
            if request_accounting["tool"] not in _DEEPLINE_UNKEYED_TOOLS:
                request_accounting["deepline_execution_key"] = (
                    "arena:" + call_identity.removeprefix("sha256:")
                )
            if deepline_catalog_entry is not None:
                request_accounting["deepline_billing_provider"] = deepline_catalog_entry["provider"]
                request_accounting["deepline_operation_aliases"] = deepline_catalog_entry["operation_aliases"]
        summary.update({
            key: value for key, value in request_accounting.items()
            if key not in ("deepline_request_id", "credential_fingerprint", "deepline_execution_key",
                           "deepline_billing_provider", "deepline_operation_aliases")
        })
        contact_finder_binding = {}
        if effective_operation.provider == "deepline":
            finder_tool = effective_normalized.get("tool")
            identity_fields = operations.DEEPLINE_CONTACT_FINDER_IDENTITY_FIELDS.get(
                finder_tool, ()
            )
            if identity_fields:
                # Keep the validated identity input on the immutable host
                # receipt. A model-authored claim cannot replace this binding.
                finder_payload = effective_normalized["payload"]
                contact_finder_binding["contact_finder_input"] = {
                    "schema_version": "leadpoet.lab_arena.contact_finder_input.v1",
                    "tool": finder_tool,
                    "payload": {
                        key: finder_payload[key] for key in identity_fields
                        if isinstance(finder_payload.get(key), str)
                        and finder_payload[key].strip()
                    },
                }
        reservation_arguments = dict(
            run_id=context.run_id,
            lease_token_hash=context.lease_token_hash,
            call_identity=call_identity,
            operation_id=operation_id,
            provider=effective_operation.provider,
            funding_source=funding_source,
            amount_microusd=amount,
            call_doc={"request_hash": request_hash, "base_call_identity": base_call_identity, "provider_attempt": provider_attempt, "action_sequence": action_sequence, "max_output_tokens": max_output_tokens, **request_accounting, **contact_finder_binding, **({"reserve_remaining_budget": True} if reserve_remaining_budget else {}), **(route.summary() if route else {})},
            lease_ttl_seconds=self._lease_ttl_seconds,
        )
        judgment_cache_key = ""
        cached_outbound: Optional[operations.OutboundRequest] = None
        if (
            getattr(context, "kind", "execute") == "score"
            and effective_operation_id in {"openrouter.chat", "openrouter.responses"}
            and provider_attempt == 1
        ):
            try:
                cached_outbound = operations.build_outbound_request(
                    effective_operation_id,
                    effective_normalized,
                    deepline_catalog=context.deepline_catalog,
                    openrouter_provider_policy=(
                        openrouter_host_route.provider_policy
                        if openrouter_host_route is not None else None
                    ),
                )
            except operations.OperationError:
                return _error_result("invalid_request", summary)
            material = _judgment_cache_material(
                context,
                requested_operation_id=operation_id,
                effective_operation_id=effective_operation_id,
                normalized_request=effective_normalized,
                outbound_body=cached_outbound.body,
            )
            if material is not None:
                judgment_cache_key, canonical_scope = material
                reservation_arguments["call_doc"].update({
                    "judgment_cache_key": judgment_cache_key,
                    "judgment_cache_canonical": canonical_scope,
                    "judgment_cache_scope": json.loads(canonical_scope),
                })
        operation_timeout_seconds = min(
            max(1, int(timeout_ms)) / 1000.0,
            float(effective_operation.timeout_seconds),
        )
        gate_lease: Optional[OpenRouterGateLease] = None
        if (
            effective_operation_id == "openrouter.responses"
            and self._openrouter_shared_gate is not None
            and openrouter_credential_fingerprint is not None
        ):
            api_deadline = (
                api_started_at
                + operations.BUDGET_ADMISSION_MAX_SECONDS
                + operation_timeout_seconds
                + operations.PROVIDER_BILLING_RECONCILIATION_SECONDS
                + operations.PROVIDER_API_TIMEOUT_GRACE_SECONDS
            )
            if time.monotonic() >= api_deadline - (
                operations.PROVIDER_BILLING_RECONCILIATION_SECONDS
                + operations.PROVIDER_API_TIMEOUT_GRACE_SECONDS
            ):
                summary.update(
                    {
                        "outcome": "not_dispatched",
                        "reason": "provider_admission_deadline",
                    }
                )
                return _error_result("provider_unavailable", summary)
            required_provider_seconds = min(
                OPENROUTER_SHARED_GATE_MIN_PROVIDER_SECONDS,
                operation_timeout_seconds,
            )
            gate_deadline = api_deadline - (
                operations.BUDGET_ADMISSION_MAX_SECONDS
                + required_provider_seconds
                + operations.PROVIDER_BILLING_RECONCILIATION_SECONDS
                + operations.PROVIDER_API_TIMEOUT_GRACE_SECONDS
            )
            try:
                gate_lease = self._openrouter_shared_gate.acquire(
                    openrouter_credential_fingerprint,
                    deadline=gate_deadline,
                    cancel_requested=cancel_requested,
                )
            except OpenRouterGateCancelled:
                summary.update(
                    {
                        "outcome": "not_dispatched",
                        "reason": "provider_admission_cancelled",
                    }
                )
                return _error_result("provider_unavailable", summary)
            if gate_lease is None:
                summary.update(
                    {
                        "outcome": "not_dispatched",
                        "reason": "provider_admission_deadline",
                    }
                )
                return _error_result("provider_unavailable", summary)
            gate_holder.append(gate_lease)

        cancelled_before_reserve = (
            gate_lease is not None
            and OpenRouterSharedGate._cancelled(cancel_requested)
        )
        if cancelled_before_reserve:
            summary.update(
                {"outcome": "not_dispatched", "reason": "provider_admission_cancelled"}
            )
            return _error_result("provider_unavailable", summary)

        # Another call can hold money without having spent it. Wait briefly for
        # settlement, using the same identity; do not dispatch or charge twice.
        reserve_deadline = time.monotonic() + operations.BUDGET_ADMISSION_MAX_SECONDS
        judgment_cache_deadline = api_started_at + operation_timeout_seconds - 2.0
        if gate_lease is not None:
            reserve_deadline = min(
                reserve_deadline,
                api_deadline
                - required_provider_seconds
                - operations.PROVIDER_BILLING_RECONCILIATION_SECONDS
                - operations.PROVIDER_API_TIMEOUT_GRACE_SECONDS,
            )
        reserve_readback_used = False
        while True:
            if judgment_cache_key and OpenRouterSharedGate._cancelled(cancel_requested):
                summary.update({"outcome": "not_dispatched", "reason": "judgment_cache_cancelled"})
                return _error_result("provider_unavailable", summary)
            try:
                reserved = (
                    self._store.reserve_judgment_call(**reservation_arguments)
                    if judgment_cache_key else
                    self._store.reserve_call(**reservation_arguments)
                )
            except ArenaStoreUnavailable:
                if reserve_readback_used:
                    raise
                # The RPC can commit even when its HTTP response times out.
                # Repeating the exact deterministic call identity serializes
                # behind that transaction and returns its durable state (or
                # creates the reservation if the first transaction rolled
                # back). This never sends the paid provider request.
                reserve_readback_used = True
                continue
            if reserved.get("status") == "cache_busy":
                if not judgment_cache_key or reserved.get("judgment_cache_key") != judgment_cache_key:
                    return _error_result("broker_unavailable", summary)
                if time.monotonic() >= judgment_cache_deadline:
                    summary.update({
                        "outcome": "not_dispatched",
                        "reason": "judgment_cache_busy",
                        "idempotent": False,
                    })
                    return _error_result("provider_unavailable", summary)
                time.sleep(min(
                    JUDGMENT_CACHE_BUSY_POLL_SECONDS,
                    max(0.0, judgment_cache_deadline - time.monotonic()),
                ))
                continue
            if reserved.get("status") != "budget_busy":
                break
            busy_reason = reserved.get("reason")
            if busy_reason == "provider_cost_uncertain":
                # The driver owns exact billing reconciliation.  Return this
                # proved pre-dispatch hold immediately so the worker can keep
                # the same action identity alive without tying up one API
                # request for the ordinary short admission window.
                summary.update(
                    {
                        "outcome": "not_dispatched",
                        "reason": busy_reason,
                        "idempotent": False,
                    }
                )
                return _error_result("provider_unavailable", summary)
            if time.monotonic() >= reserve_deadline:
                summary.update(
                    {
                        "outcome": "not_dispatched",
                        "reason": (
                            busy_reason
                            if busy_reason == "provider_calls_inflight"
                            else "budget_busy"
                        ),
                        "idempotent": False,
                    }
                )
                return _error_result("provider_unavailable", summary)
            time.sleep(min(0.2, max(0.0, reserve_deadline - time.monotonic())))
        status = reserved.get("status")
        # Terminal admission replies such as stale and refused do not create a
        # reservation. Preserve their normal classification. A response-loss
        # state, and an idempotent reservation that could still dispatch, must
        # match the exact replayed call. Completed states keep replaying their
        # saved terminal response without a new provider request.
        if (
            (
                reserve_readback_used
                and status in {
                    "reserved", "dispatched", "settled", "uncertain", "recovered",
                }
                or reserved.get("idempotent") is True and status == "reserved"
            )
            and not _reservation_readback_matches(
                self._store, reservation_arguments, reserved,
                confirmed_cost_admission=getattr(context, "kind", "execute") in {"execute", "score"},
            )
        ):
            return _error_result("broker_unavailable", summary)
        if status == "stale":
            return _error_result("lease_stale", summary)
        if status == "champion_restart_required":
            if (
                not champion_credential_retry
                or funding_source != "miner_key"
                or self._mark_provider_fallback is None
            ):
                return _error_result("broker_unavailable", summary)
            try:
                marked = self._mark_provider_fallback(
                    context,
                    effective_operation.provider,
                    {
                        "error_class": "account_credential_failure",
                        "provider_status": None,
                        "action_sequence": action_sequence,
                        "provider_attempts": (
                            CHAMPION_CREDENTIAL_PROVIDER_ATTEMPTS
                        ),
                        "base_call_identity": base_call_identity,
                    },
                )
            except Exception:
                return _error_result("broker_unavailable", summary)
            if not isinstance(marked, Mapping) or marked.get("status") not in (
                "marked",
                "existing",
            ):
                return _error_result(
                    (
                        "lease_stale"
                        if isinstance(marked, Mapping)
                        and marked.get("status") == "stale"
                        else "broker_unavailable"
                    ),
                    summary,
                )
            return _error_result(
                "miner_credentials_unavailable",
                dict(
                    summary,
                    outcome="not_dispatched",
                    provider_fallback_required=True,
                    provider_fallback_marked=True,
                ),
            )
        if status == "refused":
            summary["outcome"] = "refused"
            summary["reason"] = reserved.get("reason")
            if (
                funding_source == "miner_key"
                and reserved.get("prior_miner_credential_refusal") is True
            ):
                return _error_result("miner_credentials_unavailable", summary)
            if reserved.get("reason") == "provider_cost_uncertain":
                return _error_result("provider_unavailable", summary)
            return _error_result("budget_refused", summary)
        if status == "cache_hit":
            terminal_document = reserved.get("terminal_response")
            if (
                not judgment_cache_key
                or reserved.get("call_identity") != call_identity
                or reserved.get("judgment_cache_key") != judgment_cache_key
                or reserved.get("amount_microusd") != 0
                or not isinstance(terminal_document, Mapping)
                or terminal_document.get("judgment_cache_eligible") is not True
                or terminal_document.get("judgment_cache_key") != judgment_cache_key
                or terminal_document.get("judgment_cache_source_call_identity")
                != reserved.get("source_call_identity")
                or terminal_document.get("judgment_cache_source_run_id")
                != reserved.get("source_run_id")
                or reserved.get("source_call_identity") == call_identity
            ):
                return _error_result("broker_unavailable", summary)
            try:
                cached_status, cached_headers, cached_body = _decode_terminal(
                    terminal_document, secret=secret,
                )
            except BrokerError:
                return _error_result("broker_unavailable", summary)
            if not _complete_judgment_response(
                effective_operation_id, effective_normalized,
                status=cached_status, body=cached_body,
                call_succeeded=terminal_document.get("call_succeeded") is True,
            ):
                return _error_result("broker_unavailable", summary)
            summary.update({
                "outcome": "settled", "cached": True,
                "actual_microusd": 0, "status": cached_status,
                "source_call_identity": reserved["source_call_identity"],
                "source_run_id": reserved["source_run_id"],
                "response_hash": contracts.hash_bytes(cached_body),
            })
            return BrokerResult(cached_status, cached_headers, cached_body, summary)
        deepline_recovery_response: Optional[ProviderResponse] = None
        deepline_response_recovery = False
        terminal_document = reserved.get("terminal_response")
        if (effective_operation.provider == "deepline"
            and (status in ("dispatched", "uncertain")
                 or status == "settled" and (
                     reserved.get("deepline_response_missing") is True
                     or isinstance(terminal_document, Mapping)
                        and terminal_document.get("deepline_response_missing") is True))
            and reserved.get("account_failure_evidence") is None
            and not _has_successful_terminal_response(terminal_document)):
            ledger_entries = self._store.list_ledger(call_identity=call_identity, limit=64)
            if (status in ("dispatched", "uncertain")
                and isinstance(ledger_entries, Sequence) and ledger_entries
                and isinstance(ledger_entries[-1], Mapping)
                and ledger_entries[-1].get("entry_kind") == "settlement"):
                # Billing can settle between the reservation and history reads.
                # Refresh that same reservation once, then require the full
                # binding and charge checks against the observed history.
                reserved = self._store.reserve_call(**reservation_arguments)
                status = reserved.get("status")
                terminal_document = reserved.get("terminal_response")
            if not _reservation_readback_matches(
                self._store, reservation_arguments, reserved,
                confirmed_cost_admission=getattr(context, "kind", "execute") in {"execute", "score"},
                ledger_entries=ledger_entries,
            ):
                return _error_result("broker_unavailable", summary)
            if not _has_successful_terminal_response(terminal_document):
                for entry in ledger_entries:
                    if entry.get("entry_kind") != "uncertain":
                        continue
                    entry_doc = entry.get("entry_doc")
                    saved_call = entry_doc.get("call") if isinstance(entry_doc, Mapping) else None
                    if not isinstance(saved_call, Mapping) or "deepline_terminal_response" not in saved_call:
                        continue
                    saved_terminal = saved_call["deepline_terminal_response"]
                    try:
                        saved_status, saved_headers, saved_body = _decode_terminal(saved_terminal, secret=secret)
                        if (saved_call.get("reason") != "missing_provider_cost"
                            or saved_call.get("call_succeeded") is not False
                            or saved_terminal.get("call_succeeded") is not False
                            or saved_status not in (400, 403, 404, 422)):
                            raise BrokerError("broker_unavailable")
                    except BrokerError:
                        return _error_result("broker_unavailable", summary)
                    # Delayed settlement stores billing evidence, not the original
                    # error. Replay the sanitized reply from the same bound history;
                    # the current ledger head still owns the charge and finality.
                    summary.update(
                        outcome="settled" if status == "settled" else "uncertain",
                        idempotent=True, status=saved_status, provider_status=saved_status,
                        response_hash=contracts.hash_bytes(saved_body),
                    )
                    if status == "settled":
                        summary["actual_microusd"] = reserved["amount_microusd"]
                    if saved_status == 403:
                        summary["error_code"] = "provider_request_refused"
                    return BrokerResult(saved_status, saved_headers, saved_body, summary)
                if request_accounting.get("deepline_execution_key"):
                    recovery_outbound = operations.build_outbound_request(
                        effective_operation_id, effective_normalized,
                        deepline_catalog=context.deepline_catalog,
                    )
                    recovery_url, recovery_headers = inject_credential(recovery_outbound, secret)
                    recovery_headers["idempotency-key"] = request_accounting["deepline_execution_key"]
                    recovery_headers["x-deepline-request-id"] = request_accounting["deepline_request_id"]
                    deepline_recovery_response = _deepline_execution_response_readback(
                        transport=self._transport, secret=secret,
                        execution_key=request_accounting["deepline_execution_key"],
                        operation=request_accounting["tool"],
                        operation_aliases=(deepline_catalog_entry["operation_aliases"] if deepline_catalog_entry else ()),
                        max_response_bytes=(_DEEPLINE_FIRECRAWL_ENVELOPE_MAX_BYTES
                            if route is not None and route.adapter == "firecrawl_raw_html"
                            else effective_operation.max_response_bytes),
                        resume_request=(dict(method=recovery_outbound.target.method, url=recovery_url,
                            headers=recovery_headers, body=recovery_outbound.body,
                            timeout_seconds=operation_timeout_seconds,
                            max_response_bytes=effective_operation.max_response_bytes)
                            if deepline_catalog_entry is not None and route is None else None),
                    )
                    if deepline_recovery_response is not None:
                        deepline_response_recovery = True
                        summary["transport_recovery"] = "deepline_execution_lookup"
        if status == "settled" and not deepline_response_recovery:
            # Repeated request for a settled identity: the stored response, no second dispatch.
            terminal_document = reserved.get("terminal_response")
            try:
                terminal_status, terminal_headers, terminal_body = _decode_terminal(
                    terminal_document,
                    secret=secret,
                )
            except BrokerError:
                return _error_result("broker_unavailable", summary)
            if operations.TRUSTED_RESPONSE_URL_HEADER in terminal_headers and (
                route is None or route.adapter != "firecrawl_raw_html"
            ):
                return _error_result("broker_unavailable", summary)
            if isinstance(terminal_document, Mapping) and "judgment_cache_source_call_identity" in terminal_document:
                if (
                    not judgment_cache_key
                    or terminal_document.get("judgment_cache_key") != judgment_cache_key
                    or reserved.get("amount_microusd") != 0
                    or not _complete_judgment_response(
                        effective_operation_id, effective_normalized,
                        status=terminal_status, body=terminal_body,
                        call_succeeded=terminal_document.get("call_succeeded") is True,
                    )
                ):
                    return _error_result("broker_unavailable", summary)
                summary.update({
                    "cached": True,
                    "source_call_identity": terminal_document["judgment_cache_source_call_identity"],
                    "source_run_id": terminal_document["judgment_cache_source_run_id"],
                    "response_hash": contracts.hash_bytes(terminal_body),
                })
            summary.update({"outcome": "settled", "idempotent": True, "actual_microusd": reserved.get("amount_microusd")})
            credit_failure_proof = _validated_credit_failure_proof(
                terminal_document.get("credit_failure_proof")
                if isinstance(terminal_document, Mapping) else None
            )
            if credit_failure_proof is not None:
                if (
                    reserved.get("amount_microusd") != 0
                    or funding_source not in ("host", "miner_key")
                    or credit_failure_proof["provider"] != effective_operation.provider
                ):
                    return _error_result("broker_unavailable", summary)
                summary["credit_failure_proof"] = dict(credit_failure_proof)
            if terminal_status == 403:
                try:
                    terminal_error = json.loads(terminal_body).get("error")
                except (ValueError, AttributeError):
                    terminal_error = None
                if (
                    isinstance(terminal_error, dict)
                    and terminal_error.get("code") == "provider_request_refused"
                ):
                    summary["error_code"] = "provider_request_refused"
                    summary["provider_status"] = 403
            if funding_source == "miner_key" and terminal_status == 402:
                try:
                    error = json.loads(terminal_body).get("error")
                except (ValueError, AttributeError):
                    error = None
                if isinstance(error, dict) and error.get("code") == "miner_credentials_unavailable":
                    summary["error_code"] = "miner_credentials_unavailable"
                    evidence = _validated_account_failure_evidence(
                        terminal_document.get("account_failure_evidence")
                        if isinstance(terminal_document, Mapping)
                        else None
                    )
                    if evidence is not None:
                        if not _account_failure_matches_call(
                            evidence,
                            base_call_identity=base_call_identity,
                            provider_attempt=provider_attempt,
                            action_sequence=action_sequence,
                        ):
                            return _error_result("broker_unavailable", summary)
                        summary["provider_status"] = evidence["provider_status"]
            return BrokerResult(terminal_status, terminal_headers, terminal_body, summary)
        if status in ("dispatched", "uncertain") and not deepline_response_recovery:
            summary["outcome"] = "uncertain"
            terminal_document = reserved.get("terminal_response")
            if (effective_operation.provider == "deepline"
                and isinstance(terminal_document, Mapping)
                and terminal_document.get("call_succeeded") is True):
                try:
                    terminal_status, terminal_headers, terminal_body = _decode_terminal(terminal_document, secret=secret)
                except BrokerError:
                    return _error_result("broker_unavailable", summary)
                summary.update(idempotent=True, status=terminal_status)
                return BrokerResult(terminal_status, terminal_headers, terminal_body, summary)
            if (
                status == "uncertain"
                and judgment_cache_key
                and isinstance(terminal_document, Mapping)
                and terminal_document.get("judgment_cache_eligible") is True
                and terminal_document.get("judgment_cache_key", judgment_cache_key)
                == judgment_cache_key
                and "judgment_cache_source_call_identity" not in terminal_document
            ):
                try:
                    terminal_status, terminal_headers, terminal_body = _decode_terminal(
                        terminal_document, secret=secret,
                    )
                except BrokerError:
                    return _error_result("broker_unavailable", summary)
                if not _complete_judgment_response(
                    effective_operation_id, effective_normalized,
                    status=terminal_status, body=terminal_body,
                    call_succeeded=terminal_document.get("call_succeeded") is True,
                ):
                    return _error_result("broker_unavailable", summary)
                summary.update({
                    "idempotent": True,
                    "status": terminal_status,
                    "response_hash": contracts.hash_bytes(terminal_body),
                })
                return BrokerResult(
                    terminal_status, terminal_headers, terminal_body, summary,
                )
            if (
                status == "uncertain"
                and champion_credential_retry
                and funding_source == "miner_key"
            ):
                try:
                    evidence = _validated_account_failure_evidence(
                        reserved.get("account_failure_evidence")
                    )
                except BrokerError:
                    return _error_result("broker_unavailable", summary)
                if evidence is not None:
                    if not _account_failure_matches_call(
                        evidence,
                        base_call_identity=base_call_identity,
                        provider_attempt=provider_attempt,
                        action_sequence=action_sequence,
                    ):
                        return _error_result("broker_unavailable", summary)
                    summary["provider_status"] = evidence["provider_status"]
                    return _error_result(
                        "miner_credentials_unavailable", summary
                    )
            return _error_result("call_uncertain", summary)
        if status == "recovered":
            summary["outcome"] = "recovered"
            return _error_result("call_refused", summary)
        if status != "reserved" and not deepline_response_recovery:
            return _error_result("broker_unavailable", summary)

        # The database owns admission. Current execute and score calls retain
        # zero-amount lifecycle records; historical policies may hold money.
        reserved_amount = reserved.get("amount_microusd")
        if isinstance(reserved_amount, bool) or not isinstance(reserved_amount, int) or reserved_amount < 0:
            return _error_result("broker_unavailable", summary)
        amount = reserved_amount
        summary["reserved_microusd"] = amount
        if amount == 0 and getattr(context, "kind", "execute") in {"execute", "score"}:
            summary["reservation_basis"] = "confirmed_cost_only"
        elif reserve_remaining_budget:
            summary["reservation_basis"] = (
                "remaining_budget_native_web_search"
                if effective_operation.provider == "openrouter"
                else "remaining_budget_dynamic_deepline"
            )

        # Only gated Responses calls share the worker API's absolute deadline.
        # Other providers retain their full post-admission request and billing
        # windows. Once reserve commits, preserve the existing accounting
        # sequence because no call-level pre-dispatch release transition exists.
        billing_deadline: Optional[float] = None
        request_deadline = time.monotonic() + operation_timeout_seconds
        if gate_lease is not None:
            billing_deadline = (
                api_deadline - operations.PROVIDER_API_TIMEOUT_GRACE_SECONDS
            )
            request_deadline = min(
                request_deadline,
                billing_deadline
                - operations.PROVIDER_BILLING_RECONCILIATION_SECONDS,
            )
        dispatched = ({"status": "dispatched", "idempotent": False}
            if deepline_response_recovery else self._store.mark_dispatched(
                run_id=context.run_id,
                lease_token_hash=context.lease_token_hash,
                call_identity=call_identity,
            ))
        if dispatched.get("status") == "stale":
            # The marker did not commit (stage closed or lease lost): the request is not sent.
            return _error_result("lease_stale", summary)
        if (
            dispatched.get("status") != "dispatched"
            or dispatched.get("idempotent") is not False
        ):
            summary["outcome"] = "uncertain"
            return _error_result("call_uncertain", summary)
        # Build the outbound request from the constant table and inject the credential.
        outbound = operations.build_outbound_request(
            effective_operation_id,
            effective_normalized,
            deepline_catalog=context.deepline_catalog,
            openrouter_provider_policy=(
                openrouter_host_route.provider_policy
                if openrouter_host_route is not None
                else None
            ),
        )
        raw_document: Any = None
        deepline_readback_cost: Optional[provider_costs.ProviderCost] = None
        deepline_known_free_cost: Optional[provider_costs.ProviderCost] = None
        deepline_native_cost: Optional[provider_costs.ProviderCost] = None
        deepline_async_ids: Sequence[str] = ()
        deepline_async_poll_accepted = False
        deepline_request_failed = False
        deepline_response_request_id: Optional[str] = None
        deepline_request_id: Optional[str] = request_accounting.get("deepline_request_id")
        deepline_execution_key: Optional[str] = request_accounting.get("deepline_execution_key")
        deepline_operation: Optional[str] = request_accounting.get("tool")
        openrouter_native_cost: Optional[provider_costs.ProviderCost] = None
        openrouter_readback_cost: Optional[provider_costs.ProviderCost] = None
        openrouter_insured_cost: Optional[provider_costs.ProviderCost] = None
        openrouter_effective_response: Optional[ProviderResponse] = None
        openrouter_generation_present = False
        openrouter_generation_id: Optional[str] = None
        openrouter_canonical_response_error = False
        openrouter_completed_rate_limit_proven = False
        openrouter_retry_after_seconds: object = _RETRY_AFTER_ABSENT
        scrapingdog_observed_success_status: Optional[int] = None
        try:
            url, headers = inject_credential(outbound, secret)
            if deepline_request_id is not None:
                headers["x-deepline-request-id"] = deepline_request_id
            if deepline_execution_key is not None:
                headers["idempotency-key"] = deepline_execution_key
            timeout_seconds = max(0.001, request_deadline - time.monotonic())
            try:
                try:
                    response = deepline_recovery_response or self._transport.send(
                        method=outbound.target.method,
                        url=url,
                        headers=headers,
                        body=outbound.body,
                        timeout_seconds=timeout_seconds,
                        **(
                            {
                                "max_response_bytes": (
                                    _DEEPLINE_FIRECRAWL_ENVELOPE_MAX_BYTES
                                )
                            }
                            if route is not None
                            and route.adapter == "firecrawl_raw_html"
                            else {}
                        ),
                    )
                except ProviderTransportError as transport_error:
                    recovered = (
                        _deepline_execution_response_readback(
                            transport=self._transport, secret=secret,
                            execution_key=deepline_execution_key,
                            operation=str(deepline_operation),
                            operation_aliases=(deepline_catalog_entry["operation_aliases"]
                                               if deepline_catalog_entry else ()),
                            max_response_bytes=(_DEEPLINE_FIRECRAWL_ENVELOPE_MAX_BYTES
                                if route is not None and route.adapter == "firecrawl_raw_html"
                                else effective_operation.max_response_bytes),
                            observed_status=transport_error.observed_status,
                        )
                        if effective_operation.provider == "deepline" else None
                    )
                    if recovered is None:
                        raise
                    response = recovered
                    summary["transport_recovery"] = "deepline_execution_lookup"
                    summary["transport_error_class"] = (
                        str(transport_error) if str(transport_error) in _TRANSPORT_ERROR_CLASSES
                        else "ProviderTransportError"
                    )
                # A provider must not echo its authorization secret into a
                # stored response or back to untrusted submitted code.
                if _response_contains_credential(response, secret):
                    response = ProviderResponse(
                        502,
                        {"content-type": "application/json"},
                        b'{"error":{"code":"provider_unavailable"}}',
                        "credential_echo",
                    )
                if effective_operation.provider == "openrouter":
                    # Normalize the provider status before billing classification,
                    # while retaining the raw body for strict insurance checks and
                    # the terminal response. Invalid envelopes still flow through
                    # the existing fail-closed response handling below.
                    try:
                        openrouter_effective_response = _openrouter_effective_response(
                            response
                        )
                    except operations.OperationResponseError:
                        openrouter_effective_response = None
                    try:
                        raw_document = json.loads(response.body.decode("utf-8"))
                    except (UnicodeDecodeError, ValueError):
                        raw_document = None
                    openrouter_canonical_response_error = bool(
                        openrouter_effective_response is not None
                        and 200 <= response.status < 300
                        and not 200 <= openrouter_effective_response.status < 300
                        and isinstance(raw_document, Mapping)
                        and "error_type" in raw_document
                    )
                    openrouter_retry_after_seconds = _retry_after_seconds(
                        response.headers
                    )
                    if (
                        effective_operation_id == "openrouter.responses"
                        and gate_lease is not None
                        and openrouter_effective_response is not None
                        and openrouter_effective_response.status == 429
                    ):
                        # Stop new work for this credential as soon as the
                        # trusted provider response proves a throttle. Billing
                        # readback and settlement still decide whether the
                        # worker may retry the call.
                        self._openrouter_shared_gate.observe_throttle(
                            gate_lease, openrouter_retry_after_seconds
                        )
                    openrouter_native_cost = provider_costs.openrouter_cost(
                        raw_document
                    )
                    (
                        openrouter_generation_present,
                        openrouter_generation_id,
                    ) = _openrouter_generation_identity(
                        raw_document, response.headers
                    )
                    # This proof distinguishes a complete upstream throttle
                    # from an unqualified account-level 429. It does not prove
                    # zero cost; retry authority is added only after the exact
                    # call identity is durably marked uncertain below.
                    response_model = effective_normalized.get("model")
                    openrouter_completed_rate_limit_proven = (
                        getattr(context, "kind", "execute") == "execute"
                        and effective_operation_id == "openrouter.responses"
                        and funding_source in ("host", "miner_key")
                        and response_model
                        in OPENROUTER_REGIONAL_LUNA_RESPONSES_MODELS
                        and openrouter_host_route is not None
                        and amount == 0
                        and openrouter_canonical_response_error
                        and _openrouter_completed_rate_limit_retryable(
                            raw_document,
                            model=response_model,
                            generation_id=openrouter_generation_id,
                            credential_fingerprint=provider_credential_fingerprint,
                        )
                    )
                    if (
                        openrouter_native_cost is None
                        and openrouter_generation_id is not None
                    ):
                        openrouter_readback_cost = _openrouter_generation_readback(
                            transport=self._transport,
                            secret=secret,
                            generation_id=openrouter_generation_id,
                            reconciliation_deadline=(
                                min(
                                    time.monotonic()
                                    + operations.PROVIDER_BILLING_RECONCILIATION_SECONDS,
                                    billing_deadline,
                                )
                                if billing_deadline is not None
                                else time.monotonic()
                                + operations.PROVIDER_BILLING_RECONCILIATION_SECONDS
                            ),
                        )
                    if (
                        openrouter_native_cost is None
                        and openrouter_readback_cost is None
                        # A valid tracking id does not invalidate the stricter
                        # error-envelope proof after exact readback is absent.
                        and (
                            not openrouter_generation_present
                            or openrouter_generation_id is not None
                        )
                    ):
                        openrouter_insured_cost = (
                            provider_costs.openrouter_insured_error_cost(
                                effective_normalized,
                                self._price_table["models"][
                                    effective_normalized["model"]
                                ],
                                (
                                    openrouter_effective_response.status
                                    if openrouter_effective_response is not None
                                    else response.status
                                ),
                                raw_document,
                            )
                        )
                elif effective_operation.provider == "deepline":
                    try:
                        raw_document = json.loads(response.body.decode("utf-8"))
                    except (UnicodeDecodeError, ValueError):
                        raw_document = None
                    # A complete request-specific error is useful to the model
                    # independently of billing finality. Account errors, throttles,
                    # malformed replies and transport failures keep normal recovery.
                    deepline_request_failed = (
                        response.internal_provenance is None
                        and isinstance(raw_document, Mapping)
                        and (
                            response.status in (400, 404, 422)
                            or _provider_request_refused(
                                "deepline", effective_normalized, response
                            )
                        )
                    )
                    if (deepline_catalog_entry
                        and isinstance(deepline_catalog_entry.get("async_flow"), Mapping)):
                        deepline_async_ids = _deepline_accepted_async_job_ids(
                            response, raw_document, deepline_catalog_entry["async_flow"],
                        )
                    elif deepline_catalog_entry and deepline_catalog_entry.get("async_parent"):
                        # The pre-dispatch ownership check already bound this
                        # status read to a provider job started by this run.
                        deepline_async_poll_accepted = _deepline_async_response_accepted(
                            response, raw_document,
                        )
                    deepline_known_free_cost = provider_costs.deepline_free_completed_cost(
                        effective_normalized, response.status, raw_document,
                        catalog_entry=deepline_catalog_entry,
                    )
                    if deepline_known_free_cost is None:
                        deepline_known_free_cost = (
                            provider_costs.deepline_payment_refusal_cost(
                                response.status, raw_document
                            )
                        )
                    response_request_id = _deepline_job_request_id(raw_document)
                    header_request_id = _deepline_response_header_request_id(response.headers)
                    body_identity_present = isinstance(raw_document, Mapping) and any(
                        name in raw_document for name in ("job_id", "request_id", "requestId")
                    )
                    if response_request_id is None and not body_identity_present:
                        response_request_id = header_request_id
                    elif header_request_id is not None and header_request_id != response_request_id:
                        # Conflicting provider receipts do not identify one
                        # charge. The saved key remains the recovery authority.
                        response_request_id = None
                    request_id = response_request_id
                    # Preserve the provider receipt identity for native billing.
                    # The pre-dispatch ID remains on the immutable reservation
                    # and is used if the transport loses that receipt.
                    deepline_response_request_id = response_request_id

                    if response_request_id is not None and isinstance(raw_document, Mapping) and (
                        (
                            response.status == 200
                            and (raw_document.get("status") == "completed"
                                 or deepline_async_ids or deepline_async_poll_accepted)
                        )
                        or not 200 <= response.status < 300
                    ):
                        deepline_native_cost = provider_costs.deepline_cost(raw_document)
                        inline_billing = raw_document.get("billing")
                        if isinstance(inline_billing, Mapping) and (
                            ((deepline_async_ids or deepline_async_poll_accepted)
                                and raw_document.get("status") != "completed"
                                and inline_billing.get("pricing_status") != "final")
                            or ("pricing_status" in inline_billing and inline_billing["pricing_status"] != "final")
                            or (inline_billing.get("billing_mode") == "async_hold"
                                and deepline_native_cost is not None and deepline_native_cost.units == 0)
                        ):
                            deepline_native_cost = None
                    if (
                        deepline_native_cost is None
                        and deepline_known_free_cost is None
                        and (
                            200 <= response.status < 300
                            or response.status in (400, 401, 402, 403, 404, 422, 429)
                            or 500 <= response.status < 600
                        )
                    ):
                        deepline_operation = effective_normalized.get("tool") or {
                            "exa.search": "exa_search",
                            "exa.contents": "exa_contents",
                        }.get(effective_operation_id)
                        if not isinstance(deepline_operation, str):
                            deepline_operation = ""
                        exact_deadline = (
                                min(
                                    time.monotonic()
                                    + operations.PROVIDER_BILLING_RECONCILIATION_SECONDS,
                                    billing_deadline,
                                )
                                if billing_deadline is not None
                                else time.monotonic()
                                + operations.PROVIDER_BILLING_RECONCILIATION_SECONDS
                        )
                        recovered_id, deepline_readback_cost = _deepline_exact_readback(
                                transport=self._transport, secret=secret,
                                request_id=request_id, execution_key=deepline_execution_key,
                                operation=deepline_operation, reconciliation_deadline=exact_deadline,
                                poll=not deepline_request_failed,
                                provider=(deepline_catalog_entry["provider"] if deepline_catalog_entry else None),
                                operation_aliases=(deepline_catalog_entry["operation_aliases"] if deepline_catalog_entry else ()),
                        )
                        if recovered_id is not None:
                            deepline_response_request_id = recovered_id
            except ProviderTransportError as exc:
                # A status read from the pinned transport is bounded evidence,
                # even when the response body never completed.
                transport_error_class = (
                    str(exc) if str(exc) in _TRANSPORT_ERROR_CLASSES
                    else "ProviderTransportError"
                )
                summary["transport_error_class"] = transport_error_class
                if exc.observed_status is not None:
                    summary["observed_provider_status"] = exc.observed_status
                if (
                    effective_operation.provider == "scrapingdog"
                    and exc.observed_status is not None
                    and 200 <= exc.observed_status < 300
                ):
                    # Scrapingdog has a fixed charge for an authenticated 2xx.
                    # A lost body is still a failed model call; normal settlement
                    # records the known charge without replaying the paid GET.
                    scrapingdog_observed_success_status = exc.observed_status
                    summary["provider_status"] = exc.observed_status
                    response = ProviderResponse(
                        502, {"content-type": "application/json"},
                        operations.GENERIC_UNAVAILABLE_BODY,
                    )
                # With no observed success, the charge remains unknown and the
                # full reservation stays uncertain.
                uncertain_doc: Dict[str, Any] = {
                    "reason": "transport_failure",
                    "call_succeeded": False,
                }
                if (
                    effective_operation.provider == "openrouter"
                    and exc.openrouter_generation_id is not None
                    and openrouter_credential_fingerprint is not None
                    and not _response_contains_credential(
                        ProviderResponse(
                            0,
                            {
                                _OPENROUTER_GENERATION_HEADER: (
                                    exc.openrouter_generation_id
                                )
                            },
                            b"",
                        ),
                        secret,
                    )
                ):
                    uncertain_doc = _missing_provider_cost_call_doc(
                        ProviderResponse(0, {}, b""),
                        None,
                        call_succeeded=False,
                        openrouter_generation_id=exc.openrouter_generation_id,
                        credential_fingerprint=openrouter_credential_fingerprint,
                    )
                    uncertain_doc["transport_failure"] = True
                if effective_operation.provider == "deepline":
                    if deepline_execution_key is None and exc.deepline_job_id is not None and not _response_contains_credential(
                        ProviderResponse(0, {"x-deepline-request-id": exc.deepline_job_id}, b""),
                        secret,
                    ):
                        deepline_response_request_id = exc.deepline_job_id
                        uncertain_doc["deepline_job_id"] = exc.deepline_job_id
                    uncertain_doc.update({
                        "deepline_request_id": deepline_request_id,
                        "deepline_operation": deepline_operation,
                        "credential_fingerprint": provider_credential_fingerprint,
                    })
                    if deepline_execution_key is not None:
                        uncertain_doc["deepline_execution_key"] = deepline_execution_key
                    recovered_id, deepline_readback_cost = _deepline_exact_readback(
                            transport=self._transport, secret=secret,
                            request_id=deepline_response_request_id,
                            execution_key=deepline_execution_key,
                            operation=str(deepline_operation),
                            provider=(deepline_catalog_entry["provider"] if deepline_catalog_entry else None),
                            operation_aliases=(deepline_catalog_entry["operation_aliases"] if deepline_catalog_entry else ()),
                            reconciliation_deadline=(time.monotonic()
                                + DEEPLINE_DELAYED_RECONCILIATION_TIMEOUT_SECONDS),
                    )
                    if recovered_id is not None:
                        deepline_response_request_id = recovered_id
                        uncertain_doc["deepline_job_id"] = recovered_id
                uncertain_doc["transport_error_class"] = transport_error_class
                if exc.observed_status is not None:
                    uncertain_doc["observed_provider_status"] = exc.observed_status
                if (
                    deepline_readback_cost is None
                    and scrapingdog_observed_success_status is None
                ):
                    self._store.mark_uncertain(
                        run_id=context.run_id, lease_token_hash=context.lease_token_hash, call_identity=call_identity,
                        call_doc=uncertain_doc, lease_ttl_seconds=self._lease_ttl_seconds,
                    )
                    summary.update({"outcome": "uncertain"})
                    return _error_result("provider_unavailable", summary)
                if deepline_readback_cost is not None:
                    # A billed request with a lost result is still a failed
                    # provider call. Do not repeat the paid request.
                    response = ProviderResponse(
                        502, {"content-type": "application/json"},
                        operations.GENERIC_UNAVAILABLE_BODY,
                    )
        finally:
            secret = ""
            del secret

        # Read charge metadata from the raw trusted provider reply.  Response
        # adaptation and sanitization must not erase a known provider charge.
        raw_actual: Optional[int] = None
        raw_cost: Optional[provider_costs.ProviderCost] = None
        if effective_operation.provider == "openrouter":
            raw_cost = (
                openrouter_native_cost
                or openrouter_readback_cost
                or openrouter_insured_cost
            )
            raw_actual = None if raw_cost is None else raw_cost.microusd
        elif effective_operation.provider == "deepline":
            # Native billing is scoped to this completed request. Billing
            # history can aggregate multiple requests into one charge group,
            # so it is only a fallback when the history parser proves that the
            # entry is unshared.
            raw_cost = (
                deepline_native_cost
                or deepline_known_free_cost
                or deepline_readback_cost
            )
            raw_actual = None if raw_cost is None else raw_cost.microusd
        elif effective_operation.provider == "scrapingdog" and (
            200 <= response.status < 300
            or scrapingdog_observed_success_status is not None
        ):
            raw_cost = provider_costs.scrapingdog_cost(
                effective_operation_id, effective_normalized
            )
            raw_actual = raw_cost.microusd
        if raw_cost is not None:
            summary.update(
                {
                    "cost_basis": raw_cost.price_basis,
                    "cost_units": format(raw_cost.units, "f"),
                    "cost_unit_name": raw_cost.unit_name,
                }
            )
        cost_operation = (
            str(effective_normalized.get("tool") or effective_operation_id)
            if effective_operation.provider == "deepline"
            else effective_operation_id
        )
        cost_record = None if raw_cost is None else _provider_cost_record(
            raw_cost,
            operation=cost_operation,
            request_id=(
                (deepline_response_request_id or deepline_request_id)
                if effective_operation.provider == "deepline"
                and (
                    deepline_response_recovery
                    or raw_cost is deepline_native_cost
                    or raw_cost is deepline_readback_cost
                )
                else openrouter_generation_id
                if effective_operation.provider == "openrouter"
                and raw_cost is openrouter_readback_cost
                else None
            ),
        )
        failure_stage = "response_adaptation"
        adapted_response_url = ""
        call_succeeded = False
        judgment_cache_eligible = False
        try:
            if effective_operation.provider == "openrouter":
                response = (
                    openrouter_effective_response
                    if openrouter_effective_response is not None
                    else _openrouter_effective_response(response)
                )
            provider_status_for_summary = (
                scrapingdog_observed_success_status
                if scrapingdog_observed_success_status is not None
                else int(response.status)
            )
            call_succeeded = _provider_call_succeeded(
                effective_operation.provider,
                response,
                raw_document,
            )
            if deepline_async_ids or deepline_async_poll_accepted:
                # The provider accepted this async API call. Its eventual charge
                # still needs exact settlement; accepting a job is not free proof.
                call_succeeded = True
            request_refused = _provider_request_refused(
                effective_operation.provider,
                effective_normalized,
                response,
            )
            miner_credential_failure = (
                funding_source == "miner_key"
                and not request_refused
                and not openrouter_completed_rate_limit_proven
                and _miner_credential_failure(
                    effective_operation.provider,
                    response,
                    champion_credential_retry=champion_credential_retry,
                )
            )
            # A known miner-account refusal may finish later score jobs via
            # the existing no-dispatch refusal path even while billing is
            # unknown. Keep its call binding separate from cost evidence.
            retain_account_failure_evidence = miner_credential_failure and (
                champion_credential_retry
                or (
                    effective_operation.provider == "deepline"
                    and response.status in (401, 402, 403)
                )
            )
            missing_deepline_cost = (
                effective_operation.provider == "deepline" and raw_actual is None
            )
            missing_openrouter_cost = (
                effective_operation.provider == "openrouter"
                and (
                    openrouter_canonical_response_error
                    or response.status
                    not in (400, 401, 402, 403, 404, 422, 429)
                )
                and raw_actual is None
            )
            if request_refused:
                failure_stage = "response_sanitization"
                refused = _error_result("provider_request_refused", summary)
                sanitized_status, sanitized_headers, sanitized_body = (
                    refused.status,
                    refused.headers,
                    refused.body,
                )
            elif miner_credential_failure:
                failure_stage = "response_sanitization"
                refused = _error_result("miner_credentials_unavailable", summary)
                sanitized_status, sanitized_headers, sanitized_body = refused.status, refused.headers, refused.body
            else:
                failure_stage = "response_adaptation"
                adapted_status, adapted_headers, adapted_body, adapted_response_url = (
                    scoring_provider_compat.adapt_response_with_trusted_url(
                        route,
                        status=response.status,
                        headers=response.headers,
                        body=response.body,
                    )
                    if route is not None and 200 <= response.status < 300
                    else (response.status, response.headers, response.body, "")
                )
                failure_stage = "response_sanitization"
                sanitized_status, sanitized_headers, sanitized_body = operations.sanitize_response(
                    operation_id,
                    adapted_status,
                    adapted_headers,
                    adapted_body,
                    parameters=normalized,
                )
                if adapted_response_url:
                    sanitized_headers[operations.TRUSTED_RESPONSE_URL_HEADER] = (
                        _validated_response_url(adapted_response_url)
                    )
            judgment_cache_eligible = bool(
                judgment_cache_key
                and not request_refused
                and not miner_credential_failure
                and _complete_judgment_response(
                    effective_operation_id, effective_normalized,
                    status=sanitized_status, body=sanitized_body,
                    call_succeeded=call_succeeded,
                )
            )
            if missing_deepline_cost or missing_openrouter_cost:
                account_failure_evidence = None
                if retain_account_failure_evidence:
                    account_failure_evidence = {
                        "error_class": "account_credential_failure",
                        "provider_status": int(response.status),
                        "base_call_identity": base_call_identity,
                        "provider_attempt": provider_attempt,
                        "action_sequence": action_sequence,
                    }
                uncertain_doc = _missing_provider_cost_call_doc(
                    response,
                    raw_document,
                    call_succeeded=call_succeeded,
                    deepline_request_id=deepline_request_id,
                    deepline_response_request_id=deepline_response_request_id,
                    deepline_execution_key=deepline_execution_key,
                    deepline_operation=deepline_operation,
                    openrouter_generation_id=openrouter_generation_id,
                    openrouter_model=(effective_normalized.get("model")
                                      if effective_operation.provider == "openrouter" else None),
                    credential_fingerprint=provider_credential_fingerprint,
                )
                if deepline_request_failed:
                    uncertain_doc["deepline_terminal_response"] = _terminal_response_document(
                        sanitized_status, sanitized_headers, sanitized_body, call_succeeded=False,
                    )
                if deepline_async_ids:
                    uncertain_doc["deepline_async_job_ids"] = list(deepline_async_ids)
                if account_failure_evidence is not None:
                    uncertain_doc["account_failure_evidence"] = (
                        account_failure_evidence
                    )
                if judgment_cache_eligible:
                    uncertain_doc["judgment_cache_response"] = (
                        _terminal_response_document(
                            sanitized_status, sanitized_headers, sanitized_body,
                            call_succeeded=True,
                            judgment_cache_eligible=True,
                        )
                    )
                uncertain_state = self._store.mark_uncertain(
                    run_id=context.run_id,
                    lease_token_hash=context.lease_token_hash,
                    call_identity=call_identity,
                    call_doc=uncertain_doc,
                    lease_ttl_seconds=self._lease_ttl_seconds,
                )
                if (effective_operation.provider == "deepline" and call_succeeded
                    and isinstance(raw_document, Mapping) and raw_document.get("status") == "completed"
                    and deepline_execution_key is not None
                    and deepline_response_request_id is not None
                    and hasattr(self._store, "recover_deepline_response")):
                    uncertain_state = self._store.recover_deepline_response(
                        run_id=context.run_id, lease_token_hash=context.lease_token_hash,
                        call_identity=call_identity, request_hash=request_hash,
                        execution_key=deepline_execution_key,
                        credential_fingerprint=provider_credential_fingerprint,
                        request_id=deepline_response_request_id, operation=deepline_operation,
                        actual_microusd=None,
                        terminal_response=_terminal_response_document(
                            sanitized_status, sanitized_headers, sanitized_body, call_succeeded=True),
                        lease_ttl_seconds=self._lease_ttl_seconds,
                    )
                    if uncertain_state.get("status") not in ("uncertain", "settled"):
                        return _error_result("lease_stale" if uncertain_state.get("status") == "stale"
                                             else "broker_unavailable", summary)
                    sanitized_status, sanitized_headers, sanitized_body = _decode_terminal(
                        uncertain_state.get("terminal_response"))
                    summary["response_hash"] = contracts.hash_bytes(sanitized_body)
                summary.update(
                    {
                        "outcome": "uncertain",
                        "provider_status": int(response.status),
                    }
                )
                if (effective_operation.provider == "deepline" and call_succeeded
                    and uncertain_state.get("status") == "settled"):
                    saved_actual = uncertain_state.get("amount_microusd", uncertain_state.get("actual_microusd"))
                    saved_terminal = uncertain_state.get("terminal_response")
                    if type(saved_actual) is not int or saved_actual < 0:
                        return _error_result("broker_unavailable", summary)
                    saved_status, saved_headers, saved_body = _decode_terminal(saved_terminal)
                    summary.update(outcome="settled", actual_microusd=saved_actual,
                                   status=saved_status, response_hash=contracts.hash_bytes(saved_body))
                    return BrokerResult(saved_status, saved_headers, saved_body, summary)
                completed_rate_limit_retryable = (
                    openrouter_completed_rate_limit_proven
                    and call_succeeded is False
                    and uncertain_state.get("status") == "uncertain"
                    and uncertain_state.get("idempotent") is False
                )
                if completed_rate_limit_retryable:
                    # Internal worker control only. The generic response and
                    # durable unknown-price ledger entry remain unchanged.
                    summary.update(
                        {
                            "completed_rate_limit_retryable": True,
                            "idempotent": False,
                            "status": sanitized_status,
                        }
                    )
                    if openrouter_retry_after_seconds is not _RETRY_AFTER_ABSENT:
                        summary["retry_after_seconds"] = (
                            openrouter_retry_after_seconds
                        )
                if (
                    request_refused
                    and effective_operation.provider == "openrouter"
                ):
                    return _error_result("provider_request_refused", summary)
                if miner_credential_failure:
                    return _error_result("miner_credentials_unavailable", summary)
                if deepline_request_failed and uncertain_state.get("status") == "uncertain":
                    # Persist the original failed call and its unknown cost before
                    # returning its sanitized error. Never invent a free settlement
                    # or turn a terminal 4xx into a retryable infrastructure failure.
                    if request_refused:
                        return _error_result("provider_request_refused", summary)
                    summary.update(status=sanitized_status)
                    return BrokerResult(
                        sanitized_status, sanitized_headers, sanitized_body, summary
                    )
                if (
                    uncertain_state.get("status") == "uncertain"
                    and call_succeeded is True
                    and 200 <= sanitized_status < 300
                ):
                    # A complete sanitized result remains useful while its
                    # exact bill is pending. The ledger still forbids another
                    # dispatch of this identity and later records the real cost.
                    summary.update(status=sanitized_status)
                    if gate_lease is not None:
                        self._openrouter_shared_gate.observe_success(gate_lease)
                    return BrokerResult(
                        sanitized_status, sanitized_headers, sanitized_body, summary
                    )
                return _error_result("provider_unavailable", summary)
            failure_stage = "cost_accounting"
            if effective_operation.provider == "openrouter":
                actual = 0 if raw_actual is None else raw_actual
            elif effective_operation.provider == "deepline":
                actual = 0 if raw_actual is None else raw_actual
            elif effective_operation.provider == "scrapingdog":
                actual = 0 if raw_actual is None else raw_actual
            else:
                actual = 0  # providers without a reported charge: record the bounded call, not an invented price
            credit_failure_proof = _confirmed_credit_failure_proof(
                provider=effective_operation.provider,
                funding_source=funding_source,
                response=response,
                raw_document=raw_document,
                raw_actual=raw_actual,
                actual=actual,
                call_succeeded=call_succeeded,
                openrouter_generation_present=openrouter_generation_present,
            )
            failure_stage = "terminal_response"
            terminal = _terminal_response_document(
                sanitized_status, sanitized_headers, sanitized_body,
                call_succeeded=call_succeeded,
                provider_cost=cost_record,
                credit_failure_proof=credit_failure_proof,
                judgment_cache_eligible=judgment_cache_eligible,
                account_failure_evidence=(
                    {
                        "error_class": "account_credential_failure",
                        "provider_status": int(response.status),
                        "base_call_identity": base_call_identity,
                        "provider_attempt": provider_attempt,
                        "action_sequence": action_sequence,
                    }
                    if retain_account_failure_evidence
                    else None
                ),
            )
            if (effective_operation.provider == "deepline"
                and summary.get("transport_error_class") and not call_succeeded):
                terminal["deepline_response_missing"] = True
            if deepline_async_ids:
                terminal["deepline_async_job_ids"] = list(deepline_async_ids)
            payload = dict(summary, outcome="settled", status=sanitized_status, provider_status=provider_status_for_summary, actual_microusd=actual, response_hash=contracts.hash_bytes(sanitized_body))
            failure_stage = "settlement"
            settlement = dict(
                run_id=context.run_id, lease_token_hash=context.lease_token_hash,
                call_identity=call_identity, actual_microusd=actual,
                terminal_response=terminal, lease_ttl_seconds=self._lease_ttl_seconds,
            )
            for attempt in range(_SETTLEMENT_STORE_MAX_ATTEMPTS):
                try:
                    settled = (self._store.recover_deepline_response(
                        **settlement, request_hash=request_hash,
                        execution_key=deepline_execution_key,
                        credential_fingerprint=provider_credential_fingerprint,
                        request_id=deepline_response_request_id, operation=deepline_operation,
                    ) if deepline_response_recovery else self._store.settle_call(**settlement))
                except ArenaStoreError as exc:
                    if attempt + 1 == _SETTLEMENT_STORE_MAX_ATTEMPTS:
                        if (
                            effective_operation.provider == "scrapingdog"
                            and raw_actual is not None
                            and cost_record is not None
                            and not isinstance(exc, ArenaStoreUnavailable)
                            and re.match(
                                r"^rpc lab_arena_settle_call failed: HTTP 403(?: |$)",
                                str(exc),
                            )
                        ):
                            # A storage edge can reject a large successful page
                            # before the settlement reaches SQL. Keep the exact
                            # charge, but give the worker a compact error when
                            # the full response cannot be saved. The provider
                            # request is never sent again.
                            refused = _error_result("provider_unavailable", summary)
                            compact_terminal = _terminal_response_document(
                                refused.status, refused.headers, refused.body,
                                call_succeeded=False, provider_cost=cost_record,
                            )
                            if deepline_async_ids:
                                compact_terminal["deepline_async_job_ids"] = list(deepline_async_ids)
                            settled = self._store.settle_call(
                                **dict(settlement, terminal_response=compact_terminal)
                            )
                            if settled.get("status") != "settled":
                                raise ArenaContractError("compact settlement did not settle")
                            saved_amount = settled.get(
                                "amount_microusd" if settled.get("idempotent")
                                else "actual_microusd"
                            )
                            if saved_amount != actual:
                                raise ArenaContractError("compact settlement cost changed")
                            saved_terminal = settled.get("terminal_response")
                            if saved_terminal == compact_terminal:
                                summary.update(
                                    outcome="settled", actual_microusd=actual,
                                    provider_status=provider_status_for_summary,
                                )
                                return _error_result("provider_unavailable", summary)
                            if saved_terminal == terminal:
                                break  # The original settlement committed before its reply failed.
                            raise ArenaContractError("compact settlement terminal changed")
                        raise
                    continue
                if attempt and settled.get("status") == "settled":
                    # A lost RPC reply can leave a committed settlement. The
                    # idempotent view must describe this exact paid call.
                    saved_amount = settled.get(
                        "amount_microusd" if settled.get("idempotent") else "actual_microusd"
                    )
                    if saved_amount != actual or settled.get("terminal_response") != terminal:
                        raise ArenaContractError("settlement retry returned a different terminal")
                break
        except Exception as exc:
            # A reply the sanitizer refuses (not JSON, oversized) or a settlement
            # the store rejects must not leave the call dispatched forever, which
            # would block the attempt's completion and, repeated, cancel the
            # round: consume the reservation as uncertain and tell the model the
            # provider was unavailable.
            # If the trusted raw response carried a known charge, settle that
            # exact amount with a generic terminal error even when adaptation
            # or sanitization rejected the provider payload.
            if raw_actual is not None and failure_stage != "settlement":
                refused = _error_result("provider_unavailable", summary)
                terminal = _terminal_response_document(
                    refused.status, refused.headers, refused.body,
                    call_succeeded=False,
                    provider_cost=cost_record,
                )
                try:
                    settled = self._store.settle_call(
                        run_id=context.run_id,
                        lease_token_hash=context.lease_token_hash,
                        call_identity=call_identity,
                        actual_microusd=raw_actual,
                        terminal_response=terminal,
                        lease_ttl_seconds=self._lease_ttl_seconds,
                    )
                    if settled.get("status") == "settled":
                        summary.update(
                            {
                                "outcome": "settled",
                                "actual_microusd": raw_actual,
                                "error_code": "provider_unavailable",
                            }
                        )
                        return BrokerResult(
                            refused.status, refused.headers, refused.body, summary
                        )
                except Exception:
                    pass
            uncertain_doc: Dict[str, Any] = {
                "reason": "settle_failure",
                "call_succeeded": False,
                "failure_stage": failure_stage,
                "error_class": _safe_exception_class(exc),
            }
            if failure_stage == "settlement" or (
                effective_operation.provider == "openrouter"
                and failure_stage == "response_adaptation"
            ):
                uncertain_doc = _missing_provider_cost_call_doc(
                    response, raw_document, call_succeeded=call_succeeded,
                    deepline_request_id=deepline_request_id,
                    deepline_response_request_id=deepline_response_request_id,
                    deepline_execution_key=deepline_execution_key,
                    deepline_operation=deepline_operation,
                    openrouter_generation_id=openrouter_generation_id,
                    openrouter_model=(effective_normalized.get("model")
                                      if effective_operation.provider == "openrouter" else None),
                    credential_fingerprint=provider_credential_fingerprint,
                )
                if deepline_async_ids:
                    uncertain_doc["deepline_async_job_ids"] = list(deepline_async_ids)
                uncertain_doc.update({
                    "failure_stage": failure_stage,
                    "error_class": _safe_exception_class(exc),
                })
                if failure_stage == "settlement":
                    uncertain_doc["reason"] = "settle_failure"
                if raw_actual is not None:
                    uncertain_doc["known_actual_microusd"] = raw_actual
                if cost_record is not None:
                    uncertain_doc["provider_cost"] = cost_record
                if judgment_cache_eligible:
                    uncertain_doc["judgment_cache_response"] = (
                        _terminal_response_document(
                            sanitized_status, sanitized_headers, sanitized_body,
                            call_succeeded=True,
                            judgment_cache_eligible=True,
                        )
                    )
            if scrapingdog_observed_success_status is not None:
                uncertain_doc.update({
                    "observed_provider_status": scrapingdog_observed_success_status,
                    "transport_error_class": summary["transport_error_class"],
                })
            try:
                self._store.mark_uncertain(
                    run_id=context.run_id, lease_token_hash=context.lease_token_hash, call_identity=call_identity,
                    call_doc=uncertain_doc,
                    lease_ttl_seconds=self._lease_ttl_seconds,
                )
                summary.update({"outcome": "uncertain"})
            except Exception:
                summary.update({"outcome": "uncertain"})
            return _error_result("provider_unavailable", summary)
        settle_status = settled.get("status")
        if settle_status == "settled" and credit_failure_proof is not None:
            saved_amount = settled.get(
                "amount_microusd" if settled.get("idempotent") else "actual_microusd"
            )
            if (
                type(saved_amount) is not int
                or saved_amount != 0
                or settled.get("terminal_response") != terminal
            ):
                return _error_result("broker_unavailable", summary)
            summary["credit_failure_proof"] = dict(credit_failure_proof)
        if settle_status == "settled" and request_refused:
            summary.update(
                {
                    "outcome": "settled",
                    "actual_microusd": actual,
                    "provider_status": provider_status_for_summary,
                }
            )
            return _error_result("provider_request_refused", summary)
        if settle_status == "settled" and miner_credential_failure:
            summary.update({"outcome": "settled", "actual_microusd": actual, "provider_status": provider_status_for_summary})
            return _error_result("miner_credentials_unavailable", summary)
        if settle_status == "settled" and operations.provider_status_is_infrastructure(response.status):
            # An organizer account failure or upstream outage is infrastructure.
            summary.update({"outcome": "settled", "actual_microusd": actual, "status": sanitized_status, "provider_status": provider_status_for_summary, "response_hash": payload["response_hash"]})
            if (
                effective_operation_id == "openrouter.responses"
                and provider_status_for_summary == 429
                and type(actual) is int
                and actual == 0
            ):
                # Internal worker control data only. The model still receives
                # the same generic provider-unavailable response.
                if openrouter_retry_after_seconds is not _RETRY_AFTER_ABSENT:
                    summary["retry_after_seconds"] = openrouter_retry_after_seconds
            return _error_result("provider_unavailable", summary)
        if settle_status == "settled":
            if deepline_response_recovery:
                saved_terminal = settled.get("terminal_response")
                if not isinstance(saved_terminal, Mapping):
                    return _error_result("broker_unavailable", summary)
                try:
                    sanitized_status, sanitized_headers, sanitized_body = _decode_terminal(saved_terminal)
                except BrokerError:
                    return _error_result("broker_unavailable", summary)
                actual = settled.get("actual_microusd", settled.get("amount_microusd"))
                summary["idempotent"] = settled.get("idempotent") is True
            summary.update({"outcome": "settled", "actual_microusd": actual, "status": sanitized_status, "provider_status": provider_status_for_summary, "response_hash": contracts.hash_bytes(sanitized_body)})
            if gate_lease is not None and call_succeeded:
                self._openrouter_shared_gate.observe_success(gate_lease)
            return BrokerResult(sanitized_status, sanitized_headers, sanitized_body, summary)
        if settle_status == "stale":
            # The lease or stage ended while the request was in flight: the
            # frozen ledger keeps the call uncertain and the model gets nothing.
            summary["outcome"] = "uncertain"
            return _error_result("lease_stale", summary)
        if settle_status == "uncertain":
            summary.update({"outcome": "uncertain"})
            return _error_result("call_uncertain", summary)
        return _error_result("broker_unavailable", summary)


def parse_broker_document(document: Any) -> BrokerResult:
    """Decode a serialized ``BrokerResult`` (the API response to the worker)."""

    if not isinstance(document, Mapping) or set(document) != {"status", "headers", "body_b64", "call"}:
        raise ArenaContractError("broker document is malformed")
    try:
        body = base64.b64decode(str(document["body_b64"]), validate=True)
    except (TypeError, ValueError) as exc:
        raise ArenaContractError("broker document body is not base64") from exc
    return BrokerResult(int(document["status"]), {str(k): str(v) for k, v in dict(document["headers"]).items()}, body, dict(document["call"]))
