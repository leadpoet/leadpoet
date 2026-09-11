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

import base64
import contextvars
import json
import logging
import math
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal, ROUND_CEILING
from typing import Any, Callable, Dict, Mapping, Optional, Protocol, Sequence, Tuple
from urllib import request as urlrequest
from urllib.error import HTTPError, URLError
from urllib.parse import unquote_to_bytes

import httpx

from lab_arena import contracts, operations, provider_costs, scoring_provider_compat
from lab_arena.contracts import ArenaContractError

PRICE_TABLE_SCHEMA_VERSION = "leadpoet.lab_arena.openrouter_price_table.v1"
OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"
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
_DEEPLINE_BILLING_MAX_ATTEMPTS = 4
_DEEPLINE_BILLING_MAX_SECONDS = 5.0
_DEEPLINE_BILLING_POLL_SECONDS = 2.0


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
    input_tokens = bounded_input_tokens(parameters)
    output_tokens = int(max_output_tokens)
    usd = (
        Decimal(pricing["prompt"]) * input_tokens
        + Decimal(pricing["completion"]) * output_tokens
        + Decimal(pricing["internal_reasoning"]) * output_tokens
        + Decimal(pricing["request"])
    )
    return _microusd_ceiling(usd)


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

    requested = parameters.get("max_tokens")
    cap = operations.OPENROUTER_MAX_OUTPUT_TOKENS
    max_tokens = cap if requested is None else min(int(requested), cap)
    if max_tokens < 1:
        raise BrokerError("invalid_request")
    normalized = dict(parameters)
    normalized["max_tokens"] = max_tokens
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


def _missing_provider_cost_call_doc(
    response: ProviderResponse, document: Any
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
        "provider_status": provider_status,
        "body_bytes": body_bytes,
        "body_is_mapping": is_mapping,
        "usage_present": is_mapping and "usage" in document,
        "billing_present": is_mapping and "billing" in document,
    }
    top_status = document.get("status") if is_mapping else None
    if isinstance(top_status, str) and top_status in _DEEPLINE_JOB_STATUSES:
        diagnostics["top_level_job_status"] = top_status
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


@dataclass(frozen=True)
class ProviderResponse:
    status: int
    headers: Mapping[str, str]
    body: bytes


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
    def send(self, *, method: str, url: str, headers: Mapping[str, str], body: bytes, timeout_seconds: float) -> ProviderResponse: ...


_PROVIDER_HTTP_IN_FLIGHT = contextvars.ContextVar("arena_provider_http_in_flight", default=False)


class _ProviderHTTPLogFilter(logging.Filter):
    """Keep vendor request URLs and response headers out of broker logs.

    Scrapingdog authenticates in its query string. The broker's redacted call
    records remain the operational log; unrelated HTTP traffic is unaffected.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        return not _PROVIDER_HTTP_IN_FLIGHT.get()


_PROVIDER_HTTP_LOG_FILTER = _ProviderHTTPLogFilter()


class HttpxProviderTransport:
    """HTTPS to the constant provider hosts: HTTP/1.1, no redirects, bounded."""

    def __init__(self, *, client: Optional[httpx.Client] = None, max_response_bytes: int = 4 * 1024 * 1024) -> None:
        self._client = client or httpx.Client(http1=True, http2=False, follow_redirects=False, timeout=httpx.Timeout(30.0), trust_env=False)
        self._max_response_bytes = max_response_bytes
        # These are the loggers used by the pinned synchronous HTTP/1.1 path.
        # The context-local filter does not silence concurrent unrelated work.
        for name in ("httpx", "httpcore.connection", "httpcore.http11", "httpcore.proxy"):
            logging.getLogger(name).addFilter(_PROVIDER_HTTP_LOG_FILTER)

    def send(self, *, method: str, url: str, headers: Mapping[str, str], body: bytes, timeout_seconds: float) -> ProviderResponse:
        if not url.startswith("https://"):
            raise ProviderTransportError("non-https target")
        log_token = _PROVIDER_HTTP_IN_FLIGHT.set(True)
        try:
            with self._client.stream(method, url, headers=dict(headers), content=body, timeout=httpx.Timeout(float(timeout_seconds))) as response:
                status = int(response.status_code)
                response_headers = {k.lower(): v for k, v in response.headers.items()}
                content = bytearray()
                oversized = False
                for chunk in response.iter_bytes(chunk_size=64 * 1024):
                    if len(content) + len(chunk) > self._max_response_bytes:
                        oversized = True
                        break
                    content.extend(chunk)
        except httpx.HTTPError as exc:
            raise ProviderTransportError(type(exc).__name__) from exc
        finally:
            _PROVIDER_HTTP_IN_FLIGHT.reset(log_token)
        if oversized or 300 <= status < 400:
            # Redirects are never followed; a redirecting provider is unavailable.
            return ProviderResponse(502, {"content-type": "application/json"}, operations.GENERIC_UNAVAILABLE_BODY)
        return ProviderResponse(status, response_headers, bytes(content))

    def close(self) -> None:
        self._client.close()


def _deepline_job_request_id(document: Any) -> Optional[str]:
    """Return a usable job id only from Deepline's bounded job envelope."""

    if not isinstance(document, Mapping):
        return None
    request_id = document.get("job_id")
    status = document.get("status")
    if (
        not isinstance(request_id, str)
        or not request_id.strip()
        or len(request_id) > 512
        or not isinstance(status, str)
        or status not in _DEEPLINE_JOB_STATUSES
    ):
        return None
    return request_id


def _deepline_billing_readback(
    *,
    transport: ProviderTransport,
    secret: str,
    request_id: str,
    operation: str,
    request_deadline: float,
) -> Optional[provider_costs.ProviderCost]:
    """Poll bounded billing history and return only one exact terminal charge."""

    readback_deadline = min(
        request_deadline, time.monotonic() + _DEEPLINE_BILLING_MAX_SECONDS
    )
    headers = {
        "accept": "application/json",
        "authorization": "Bearer " + secret,
        "user-agent": "leadpoet-lab-arena-broker/1",
    }
    history_url = DEEPLINE_BILLING_HISTORY_URL
    current_offset = 0
    for request_number in range(_DEEPLINE_BILLING_MAX_ATTEMPTS):
        # A newly completed job may not yet appear on the newest page. Keep
        # the existing three-page scan, then refresh the newest page instead
        # of spending every read on older history. The time bound is unchanged.
        if request_number == _DEEPLINE_BILLING_MAX_ATTEMPTS - 1:
            history_url = DEEPLINE_BILLING_HISTORY_URL
            current_offset = 0
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
        if state == "pending" and has_more:
            if next_offset is None:
                return None
            current_offset = next_offset
            history_url = DEEPLINE_BILLING_HISTORY_URL + "&recent_offset=" + str(next_offset)
            continue
        history_url = DEEPLINE_BILLING_HISTORY_URL
        current_offset = 0
    return None


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


@dataclass(frozen=True)
class BrokerResult:
    status: int
    headers: Dict[str, str]
    body: bytes
    call: Dict[str, Any]

    def to_document(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "headers": dict(self.headers),
            "body_b64": base64.b64encode(self.body).decode("ascii"),
            "call": dict(self.call),
        }


class CallStore(Protocol):
    def reserve_call(self, **kwargs: Any) -> Dict[str, Any]: ...

    def mark_dispatched(self, **kwargs: Any) -> Dict[str, Any]: ...

    def settle_call(self, **kwargs: Any) -> Dict[str, Any]: ...

    def mark_uncertain(self, **kwargs: Any) -> Dict[str, Any]: ...

    def get_run(self, run_id: str) -> Optional[Dict[str, Any]]: ...


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
    *, provider_cost: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    document = {"status": int(status), "headers": dict(headers), "body_b64": base64.b64encode(bytes(body)).decode("ascii")}
    if provider_cost is not None:
        document["provider_cost"] = dict(provider_cost)
    return document


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


def _decode_terminal(
    document: Any,
    *,
    secret: str = "",
) -> Tuple[int, Dict[str, str], bytes]:
    required = {"status", "headers", "body_b64"}
    if not isinstance(document, Mapping) or set(document) not in (required, required | {"provider_cost"}):
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
    try:
        status = int(document["status"])
        headers = dict(document["headers"])
        body = base64.b64decode(str(document["body_b64"]), validate=True)
    except (KeyError, TypeError, ValueError) as exc:
        raise BrokerError("broker_unavailable") from exc
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


def _error_result(code: str, call: Mapping[str, Any]) -> BrokerResult:
    """A generic error reply for the model; the call summary keeps the code so
    the worker can tell a refused key or quota from a judge's own failure."""

    body = json.dumps({"error": {"code": code}}, separators=(",", ":")).encode("utf-8")
    return BrokerResult(GENERIC_ERRORS[code], {"content-type": "application/json", "content-length": str(len(body))}, body, dict(call, error_code=code))


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

    statuses = []
    for error in errors:
        code = error.get("code") if isinstance(error, Mapping) else None
        if isinstance(code, bool) or not isinstance(code, int) or not 400 <= code <= 599:
            raise operations.OperationResponseError("invalid_response")
        statuses.append(code)
    if len(set(statuses)) != 1:
        raise operations.OperationResponseError("invalid_response")
    return ProviderResponse(statuses[0], response.headers, response.body)


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
    ) -> None:
        self._store = store
        # Host-only callers retain key_for. Production supplies the scoped
        # resolver for both model execution and judging. It must never fall
        # back to a host key for a miner submission.
        self._key_for = key_for
        self._credential_for = credential_for
        self._funding_source_for = funding_source_for
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
        return normalized, int(normalized["max_tokens"])

    # -- execution ------------------------------------------------------------

    def execute(
        self,
        context: RunContext,
        *,
        operation_id: str,
        parameters: Mapping[str, Any],
        action_sequence: int,
        timeout_ms: int,
    ) -> BrokerResult:
        operation = operations.OPERATIONS.get(operation_id)
        if operation is None:
            return _error_result("invalid_request", {"operation_id": str(operation_id)})
        try:
            normalized = operations.validate_operation_request(operation_id, parameters)
        except operations.OperationError:
            return _error_result("invalid_request", {"operation_id": operation_id})
        if isinstance(action_sequence, bool) or not isinstance(action_sequence, int) or action_sequence < 0:
            return _error_result("invalid_request", {"operation_id": operation_id})
        funding_source = "host"
        try:
            funding_source = self._funding_source_for(context) if self._funding_source_for else "host"
            if funding_source not in ("host", "miner_key"):
                raise BrokerError("broker_unavailable")
            route = scoring_provider_compat.route_for(
                kind=getattr(context, "kind", "execute"),
                funding_source=funding_source,
                round_id=getattr(context, "round_id", ""),
                operation_id=operation_id,
                parameters=normalized,
            )
            effective_operation_id = route.effective_operation_id if route else operation_id
            effective_parameters = route.effective_parameters if route else normalized
            effective_operation = operations.OPERATIONS[effective_operation_id]
            effective_normalized = operations.validate_operation_request(
                effective_operation_id, effective_parameters
            )
            secret = self._credential_for(context, effective_operation.provider) if self._credential_for else self._key_for(effective_operation.provider)
            if not isinstance(secret, str) or not secret:
                raise BrokerError("miner_credentials_unavailable" if funding_source == "miner_key" else "broker_unavailable")
        except (BrokerError, KeyError, operations.OperationError) as exc:
            if not isinstance(exc, BrokerError):
                exc = BrokerError("broker_unavailable")
            return _error_result(exc.code, {"operation_id": operation_id, "funding_source": funding_source})
        max_output_tokens = 0
        reservation_cost: Optional[provider_costs.ProviderCost] = None
        reserve_remaining_budget = False
        try:
            if effective_operation.provider == "openrouter":
                # Reserve the maximum cost allowed by the request and output cap.
                effective_normalized, max_output_tokens = self._openrouter_parameters(effective_normalized, kind=getattr(context, "kind", "execute"))
                normalized = effective_normalized
                amount = max_openrouter_cost_microusd(self._price_table, normalized["model"], normalized, max_output_tokens=max_output_tokens)
            elif effective_operation.provider == "scrapingdog":
                reservation_cost = provider_costs.scrapingdog_cost(
                    effective_operation_id, effective_normalized
                )
                amount = reservation_cost.microusd
            elif effective_operation.provider == "deepline":
                reservation_cost = provider_costs.deepline_reservation_cost(effective_normalized)
                reserve_remaining_budget = reservation_cost is None
                amount = 0 if reservation_cost is None else reservation_cost.microusd
            else:
                raise BrokerError("invalid_request")
        except BrokerError as exc:
            return _error_result(exc.code, {"operation_id": operation_id})
        request_hash = contracts.document_hash(normalized)
        call_identity = contracts.provider_call_identity(
            assignment_id=context.assignment_id,
            attempt=int(getattr(context, "attempt", 1)),
            icp_position=context.icp_position,
            action_sequence=action_sequence,
            operation_id=operation_id,
            request_hash=request_hash,
        )
        summary: Dict[str, Any] = {
            "call_identity": call_identity,
            "operation_id": operation_id,
            "provider": effective_operation.provider,
            "funding_source": funding_source,
            "request_hash": request_hash,
            "reserved_microusd": amount,
            "action_sequence": action_sequence,
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
        summary.update(request_accounting)
        reservation_arguments = dict(
            run_id=context.run_id,
            lease_token_hash=context.lease_token_hash,
            call_identity=call_identity,
            operation_id=operation_id,
            provider=effective_operation.provider,
            funding_source=funding_source,
            amount_microusd=amount,
            call_doc={"request_hash": request_hash, "action_sequence": action_sequence, "max_output_tokens": max_output_tokens, **request_accounting, **({"reserve_remaining_budget": True} if reserve_remaining_budget else {}), **(route.summary() if route else {})},
            lease_ttl_seconds=self._lease_ttl_seconds,
        )
        # Another call can hold money without having spent it. Wait briefly for
        # settlement, using the same identity; do not dispatch or charge twice.
        request_deadline = time.monotonic() + min(max(1, int(timeout_ms)) / 1000.0, float(effective_operation.timeout_seconds))
        reserve_deadline = min(time.monotonic() + 20.0, request_deadline - 1.0)
        while True:
            reserved = self._store.reserve_call(**reservation_arguments)
            if reserved.get("status") != "budget_busy":
                break
            if time.monotonic() >= reserve_deadline:
                summary.update({"outcome": "not_dispatched", "reason": "budget_busy"})
                return _error_result("provider_unavailable", summary)
            time.sleep(min(0.2, max(0.0, reserve_deadline - time.monotonic())))
        status = reserved.get("status")
        if status == "stale":
            return _error_result("lease_stale", summary)
        if status == "refused":
            summary["outcome"] = "refused"
            summary["reason"] = reserved.get("reason")
            return _error_result("budget_refused", summary)
        if status == "settled":
            # Repeated request for a settled identity: the stored response, no second dispatch.
            try:
                terminal_status, terminal_headers, terminal_body = _decode_terminal(
                    reserved.get("terminal_response"),
                    secret=secret,
                )
            except BrokerError:
                return _error_result("broker_unavailable", summary)
            if operations.TRUSTED_RESPONSE_URL_HEADER in terminal_headers and (
                route is None or route.adapter != "firecrawl_raw_html"
            ):
                return _error_result("broker_unavailable", summary)
            summary.update({"outcome": "settled", "idempotent": True, "actual_microusd": reserved.get("amount_microusd")})
            if funding_source == "miner_key" and terminal_status == 402:
                try:
                    error = json.loads(terminal_body).get("error")
                except (ValueError, AttributeError):
                    error = None
                if isinstance(error, dict) and error.get("code") == "miner_credentials_unavailable":
                    summary["error_code"] = "miner_credentials_unavailable"
            return BrokerResult(terminal_status, terminal_headers, terminal_body, summary)
        if status in ("dispatched", "uncertain"):
            summary["outcome"] = "uncertain"
            return _error_result("call_uncertain", summary)
        if status == "recovered":
            summary["outcome"] = "recovered"
            return _error_result("call_refused", summary)
        if status != "reserved":
            return _error_result("broker_unavailable", summary)

        # Dynamic reservations are allocated atomically by the database. Its
        # amount, not the requested zero placeholder, is the real liability.
        reserved_amount = reserved.get("amount_microusd")
        if isinstance(reserved_amount, bool) or not isinstance(reserved_amount, int) or reserved_amount < 0:
            return _error_result("broker_unavailable", summary)
        amount = reserved_amount
        summary["reserved_microusd"] = amount
        if reserve_remaining_budget:
            summary["reservation_basis"] = "remaining_budget_dynamic_deepline"

        dispatched = self._store.mark_dispatched(run_id=context.run_id, lease_token_hash=context.lease_token_hash, call_identity=call_identity)
        if dispatched.get("status") == "stale":
            # The marker did not commit (stage closed or lease lost): the request is not sent.
            return _error_result("lease_stale", summary)
        if dispatched.get("status") != "dispatched":
            summary["outcome"] = "uncertain"
            return _error_result("call_uncertain", summary)
        # Build the outbound request from the constant table and inject the credential.
        outbound = operations.build_outbound_request(effective_operation_id, effective_normalized)
        raw_document: Any = None
        deepline_readback_cost: Optional[provider_costs.ProviderCost] = None
        deepline_known_free_cost: Optional[provider_costs.ProviderCost] = None
        deepline_request_id: Optional[str] = None
        try:
            url, headers = inject_credential(outbound, secret)
            timeout_seconds = max(0.001, request_deadline - time.monotonic())
            try:
                response = self._transport.send(method=outbound.target.method, url=url, headers=headers, body=outbound.body, timeout_seconds=timeout_seconds)
                # A provider must not echo its authorization secret into a
                # stored response or back to untrusted submitted code.
                if _response_contains_credential(response, secret):
                    response = ProviderResponse(
                        502,
                        {"content-type": "application/json"},
                        b'{"error":{"code":"provider_unavailable"}}',
                    )
                if effective_operation.provider == "deepline":
                    try:
                        raw_document = json.loads(response.body.decode("utf-8"))
                    except (UnicodeDecodeError, ValueError):
                        raw_document = None
                    deepline_known_free_cost = provider_costs.deepline_free_completed_cost(
                        effective_normalized, response.status, raw_document
                    )
                    request_id = _deepline_job_request_id(raw_document)
                    deepline_request_id = request_id
                    if (
                        deepline_known_free_cost is None
                        and 200 <= response.status < 300
                        and request_id is not None
                    ):
                        deepline_operation = effective_normalized.get("tool") or {
                            "exa.search": "exa_search",
                            "exa.contents": "exa_contents",
                        }.get(effective_operation_id)
                        if not isinstance(deepline_operation, str):
                            deepline_operation = ""
                        deepline_readback_cost = _deepline_billing_readback(
                            transport=self._transport,
                            secret=secret,
                            request_id=request_id,
                            operation=deepline_operation,
                            request_deadline=request_deadline,
                        )
            except ProviderTransportError:
                # Outcome unknown after send: consume the full reservation.
                result = self._store.mark_uncertain(
                    run_id=context.run_id, lease_token_hash=context.lease_token_hash, call_identity=call_identity,
                    call_doc={"reason": "transport_failure"}, lease_ttl_seconds=self._lease_ttl_seconds,
                )
                summary.update({"outcome": "uncertain", "actual_microusd": amount})
                return _error_result("provider_unavailable", summary)
        finally:
            secret = ""
            del secret

        # Read charge metadata from the raw trusted provider reply.  Response
        # adaptation and sanitization must not erase a known provider charge.
        raw_actual: Optional[int] = None
        raw_cost: Optional[provider_costs.ProviderCost] = None
        if effective_operation.provider == "openrouter":
            try:
                raw_document = json.loads(response.body.decode("utf-8"))
            except (UnicodeDecodeError, ValueError):
                raw_document = None
            raw_actual = actual_openrouter_cost_microusd(
                self._price_table, normalized["model"], raw_document
            )
            raw_cost = provider_costs.openrouter_cost(raw_document)
        elif effective_operation.provider == "deepline":
            # A terminal per-job history entry is authoritative. Native
            # response billing has been observed to understate the posted
            # charge, so paid calls never fall back to it.
            raw_cost = deepline_known_free_cost or deepline_readback_cost
            raw_actual = None if raw_cost is None else raw_cost.microusd
        elif effective_operation.provider == "scrapingdog" and 200 <= response.status < 300:
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
                deepline_request_id
                if effective_operation.provider == "deepline"
                and raw_cost is deepline_readback_cost
                else None
            ),
        )
        failure_stage = "response_adaptation"
        adapted_response_url = ""
        try:
            if effective_operation.provider == "openrouter":
                response = _openrouter_effective_response(response)
            if (
                response.status not in (400, 401, 402, 403, 404, 422, 429)
                and effective_operation.provider in ("openrouter", "deepline")
                and raw_actual is None
            ):
                result = self._store.mark_uncertain(
                    run_id=context.run_id,
                    lease_token_hash=context.lease_token_hash,
                    call_identity=call_identity,
                    call_doc=_missing_provider_cost_call_doc(response, raw_document),
                    lease_ttl_seconds=self._lease_ttl_seconds,
                )
                summary.update({"outcome": "uncertain", "actual_microusd": amount, "provider_status": int(response.status)})
                return _error_result("provider_unavailable", summary)
            if funding_source == "miner_key" and response.status in (401, 402, 403):
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
            failure_stage = "cost_accounting"
            if effective_operation.provider == "openrouter":
                actual = 0 if raw_actual is None else raw_actual
            elif effective_operation.provider == "deepline":
                actual = 0 if raw_actual is None else raw_actual
            elif effective_operation.provider == "scrapingdog":
                actual = 0 if raw_actual is None else raw_actual
            else:
                actual = 0  # providers without a reported charge: record the bounded call, not an invented price
            failure_stage = "terminal_response"
            terminal = _terminal_response_document(
                sanitized_status, sanitized_headers, sanitized_body,
                provider_cost=cost_record,
            )
            payload = dict(summary, outcome="settled", status=sanitized_status, provider_status=int(response.status), actual_microusd=actual, response_hash=contracts.hash_bytes(sanitized_body))
            failure_stage = "settlement"
            settled = self._store.settle_call(
                run_id=context.run_id, lease_token_hash=context.lease_token_hash, call_identity=call_identity,
                actual_microusd=actual, terminal_response=terminal, lease_ttl_seconds=self._lease_ttl_seconds,
            )
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
            try:
                self._store.mark_uncertain(
                    run_id=context.run_id, lease_token_hash=context.lease_token_hash, call_identity=call_identity,
                    call_doc={
                        "reason": "settle_failure",
                        "failure_stage": failure_stage,
                        "error_class": _safe_exception_class(exc),
                    },
                    lease_ttl_seconds=self._lease_ttl_seconds,
                )
                summary.update({"outcome": "uncertain", "actual_microusd": amount})
            except Exception:
                summary.update({"outcome": "uncertain", "actual_microusd": amount})
            return _error_result("provider_unavailable", summary)
        settle_status = settled.get("status")
        if settle_status == "settled" and funding_source == "miner_key" and response.status in (401, 402, 403):
            summary.update({"outcome": "settled", "actual_microusd": actual, "provider_status": int(response.status)})
            return _error_result("miner_credentials_unavailable", summary)
        if settle_status == "settled" and operations.provider_status_is_infrastructure(response.status):
            # An organizer account failure or upstream outage is infrastructure.
            summary.update({"outcome": "settled", "actual_microusd": actual, "status": sanitized_status, "provider_status": int(response.status), "response_hash": payload["response_hash"]})
            return _error_result("provider_unavailable", summary)
        if settle_status == "settled":
            summary.update({"outcome": "settled", "actual_microusd": actual, "status": sanitized_status, "provider_status": int(response.status), "response_hash": payload["response_hash"]})
            return BrokerResult(sanitized_status, sanitized_headers, sanitized_body, summary)
        if settle_status == "stale":
            # The lease or stage ended while the request was in flight: the
            # frozen ledger keeps the call uncertain and the model gets nothing.
            summary["outcome"] = "uncertain"
            return _error_result("lease_stale", summary)
        if settle_status == "uncertain":
            summary.update({"outcome": "uncertain", "actual_microusd": amount})
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
