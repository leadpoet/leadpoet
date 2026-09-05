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
import json
import math
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal, ROUND_CEILING
from typing import Any, Callable, Dict, Mapping, Optional, Protocol, Sequence, Tuple
from urllib import request as urlrequest
from urllib.error import HTTPError, URLError

import httpx

from lab_arena import contracts, operations, scoring_provider_compat
from lab_arena.contracts import ArenaContractError

PRICE_TABLE_SCHEMA_VERSION = "leadpoet.lab_arena.openrouter_price_table.v1"
OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"
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
    # per serialized character is a conservative bound for supported models.
    return REQUEST_TOKEN_OVERHEAD + TOKENS_PER_CHAR_BOUND * len(contracts.canonical_json(dict(parameters)))


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
    """Actual cost from the response usage under the pinned table.

    Returns ``None`` when usage is missing, malformed, or names a different
    model, in which case the caller settles at the full reservation. A cost
    above the reservation is clamped by the caller (never above).
    """

    if not isinstance(response_json, Mapping):
        return None
    pricing = price_table["models"].get(model)
    if pricing is None:
        return None
    reported_model = response_json.get("model")
    if reported_model is not None and str(reported_model).split(":")[0] != model.split(":")[0]:
        return None
    usage = response_json.get("usage")
    if not isinstance(usage, Mapping):
        return None
    try:
        prompt_tokens = int(usage.get("prompt_tokens"))
        completion_tokens = int(usage.get("completion_tokens"))
    except (TypeError, ValueError):
        return None
    if isinstance(usage.get("prompt_tokens"), bool) or prompt_tokens < 0 or completion_tokens < 0:
        return None
    reasoning_tokens = 0
    details = usage.get("completion_tokens_details")
    if isinstance(details, Mapping) and details.get("reasoning_tokens") is not None:
        try:
            reasoning_tokens = max(0, int(details.get("reasoning_tokens")))
        except (TypeError, ValueError):
            return None
    usd = (
        Decimal(pricing["prompt"]) * prompt_tokens
        + Decimal(pricing["completion"]) * completion_tokens
        + Decimal(pricing["internal_reasoning"]) * reasoning_tokens
        + Decimal(pricing["request"])
    )
    return _microusd_ceiling(usd)


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


def deepline_cost_microusd(body: bytes) -> int:
    """``billing.cost_usd`` from a Deepline execute envelope as micro-USD, else 0."""

    try:
        document = json.loads(bytes(body).decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        return 0
    billing = document.get("billing") if isinstance(document, Mapping) else None
    value = billing.get("cost_usd") if isinstance(billing, Mapping) else None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0 or value != value or value in (float("inf"),):
        return 0
    return int(value * 1_000_000)


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


class ProviderTransport(Protocol):
    def send(self, *, method: str, url: str, headers: Mapping[str, str], body: bytes, timeout_seconds: float) -> ProviderResponse: ...


class HttpxProviderTransport:
    """HTTPS to the constant provider hosts: HTTP/1.1, no redirects, bounded."""

    def __init__(self, *, client: Optional[httpx.Client] = None, max_response_bytes: int = 4 * 1024 * 1024) -> None:
        self._client = client or httpx.Client(http1=True, http2=False, follow_redirects=False, timeout=httpx.Timeout(30.0), trust_env=False)
        self._max_response_bytes = max_response_bytes

    def send(self, *, method: str, url: str, headers: Mapping[str, str], body: bytes, timeout_seconds: float) -> ProviderResponse:
        if not url.startswith("https://"):
            raise ProviderTransportError("non-https target")
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
        if oversized or 300 <= status < 400:
            # Redirects are never followed; a redirecting provider is unavailable.
            return ProviderResponse(502, {"content-type": "application/json"}, operations.GENERIC_UNAVAILABLE_BODY)
        return ProviderResponse(status, response_headers, bytes(content))

    def close(self) -> None:
        self._client.close()


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


def _terminal_response_document(status: int, headers: Mapping[str, str], body: bytes) -> Dict[str, Any]:
    return {"status": int(status), "headers": dict(headers), "body_b64": base64.b64encode(bytes(body)).decode("ascii")}


def _decode_terminal(document: Any) -> Tuple[int, Dict[str, str], bytes]:
    if not isinstance(document, Mapping):
        raise BrokerError("broker_unavailable")
    try:
        return int(document["status"]), dict(document.get("headers") or {}), base64.b64decode(str(document["body_b64"]), validate=True)
    except (KeyError, TypeError, ValueError) as exc:
        raise BrokerError("broker_unavailable") from exc


def _error_result(code: str, call: Mapping[str, Any]) -> BrokerResult:
    """A generic error reply for the model; the call summary keeps the code so
    the worker can tell a refused key or quota from a judge's own failure."""

    body = json.dumps({"error": {"code": code}}, separators=(",", ":")).encode("utf-8")
    return BrokerResult(GENERIC_ERRORS[code], {"content-type": "application/json", "content-length": str(len(body))}, body, dict(call, error_code=code))


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
        try:
            if effective_operation.provider == "openrouter":
                # Reserve the maximum cost allowed by the request and output cap.
                effective_normalized, max_output_tokens = self._openrouter_parameters(effective_normalized, kind=getattr(context, "kind", "execute"))
                normalized = effective_normalized
                amount = max_openrouter_cost_microusd(self._price_table, normalized["model"], normalized, max_output_tokens=max_output_tokens)
            else:
                # Other providers are bounded by call quota, so the reservation
                # carries no estimated amount.
                amount = 0
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
        if route is not None:
            summary.update(route.summary())
        reserved = self._store.reserve_call(
            run_id=context.run_id,
            lease_token_hash=context.lease_token_hash,
            call_identity=call_identity,
            operation_id=operation_id,
            provider=effective_operation.provider,
            funding_source=funding_source,
            amount_microusd=amount,
            call_doc={"request_hash": request_hash, "action_sequence": action_sequence, "max_output_tokens": max_output_tokens, **(route.summary() if route else {})},
            lease_ttl_seconds=self._lease_ttl_seconds,
        )
        status = reserved.get("status")
        if status == "stale":
            return _error_result("lease_stale", summary)
        if status == "refused":
            summary["outcome"] = "refused"
            summary["reason"] = reserved.get("reason")
            return _error_result("budget_refused", summary)
        if status == "settled":
            # Repeated request for a settled identity: the stored response, no second dispatch.
            terminal_status, terminal_headers, terminal_body = _decode_terminal(reserved.get("terminal_response"))
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

        dispatched = self._store.mark_dispatched(run_id=context.run_id, lease_token_hash=context.lease_token_hash, call_identity=call_identity)
        if dispatched.get("status") == "stale":
            # The marker did not commit (stage closed or lease lost): the request is not sent.
            return _error_result("lease_stale", summary)
        if dispatched.get("status") != "dispatched":
            summary["outcome"] = "uncertain"
            return _error_result("call_uncertain", summary)
        # Build the outbound request from the constant table and inject the credential.
        outbound = operations.build_outbound_request(effective_operation_id, effective_normalized)
        try:
            url, headers = inject_credential(outbound, secret)
            timeout_seconds = min(max(1, int(timeout_ms)) / 1000.0, float(effective_operation.timeout_seconds))
            try:
                response = self._transport.send(method=outbound.target.method, url=url, headers=headers, body=outbound.body, timeout_seconds=timeout_seconds)
                # A provider must not echo its authorization secret into a
                # stored response or back to untrusted submitted code.
                if secret.encode("utf-8") in response.body:
                    response = ProviderResponse(
                        502,
                        {"content-type": "application/json"},
                        b'{"error":{"code":"provider_unavailable"}}',
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

        try:
            if funding_source == "miner_key" and response.status in (401, 402, 403):
                refused = _error_result("miner_credentials_unavailable", summary)
                sanitized_status, sanitized_headers, sanitized_body = refused.status, refused.headers, refused.body
            else:
                adapted_status, adapted_headers, adapted_body = (
                    scoring_provider_compat.adapt_response(
                        route,
                        status=response.status,
                        headers=response.headers,
                        body=response.body,
                    )
                    if route is not None and 200 <= response.status < 300
                    else (response.status, response.headers, response.body)
                )
                sanitized_status, sanitized_headers, sanitized_body = operations.sanitize_response(
                    operation_id,
                    adapted_status,
                    adapted_headers,
                    adapted_body,
                    parameters=normalized,
                )
            if effective_operation.provider == "openrouter":
                actual: Optional[int] = None
                if 200 <= response.status < 300:
                    try:
                        actual = actual_openrouter_cost_microusd(self._price_table, normalized["model"], json.loads(response.body.decode("utf-8")))
                    except (UnicodeDecodeError, ValueError):
                        actual = None
                # Missing, malformed, stale, or excessive usage retains the full reservation.
                actual = amount if actual is None or actual > amount else actual
            elif effective_operation.provider == "deepline" and 200 <= response.status < 300:
                # Deepline reports the charge in its envelope; record it, floor-rounded.
                actual = deepline_cost_microusd(response.body)
            else:
                actual = 0  # providers without a reported charge: record the bounded call, not an invented price
            terminal = _terminal_response_document(sanitized_status, sanitized_headers, sanitized_body)
            payload = dict(summary, outcome="settled", status=sanitized_status, provider_status=int(response.status), actual_microusd=actual, response_hash=contracts.hash_bytes(sanitized_body))
            settled = self._store.settle_call(
                run_id=context.run_id, lease_token_hash=context.lease_token_hash, call_identity=call_identity,
                actual_microusd=actual, terminal_response=terminal, lease_ttl_seconds=self._lease_ttl_seconds,
            )
        except Exception:
            # A reply the sanitizer refuses (not JSON, oversized) or a settlement
            # the store rejects must not leave the call dispatched forever, which
            # would block the attempt's completion and, repeated, cancel the
            # round: consume the reservation as uncertain and tell the model the
            # provider was unavailable.
            try:
                self._store.mark_uncertain(
                    run_id=context.run_id, lease_token_hash=context.lease_token_hash, call_identity=call_identity,
                    call_doc={"reason": "settle_failure"}, lease_ttl_seconds=self._lease_ttl_seconds,
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
