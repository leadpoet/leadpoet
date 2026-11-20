"""Provider broker for one lead order: OpenRouter and Deepline.

The private runner never holds a provider key. It calls these endpoints with
the one-order token it got at claim. For every call the broker:

1. checks the token's order is running (not finished, not past deadline);
2. reserves the call's worst-case cost against the order's spend ceiling
   *before* sending (402 ``spend_ceiling`` once the next call would pass it);
3. sends with the best-ranked key (miner keys by settled spend, Leadpoet last);
4. settles the reservation at the provider-reported cost, one ledger row per
   call. A call with no reported cost is settled ``uncertain``: its
   reservation keeps counting against the ceiling, and it is never charged
   to the buyer or rewarded to a miner;
5. on 401, 402, 403, a spent limit or a 429 without Retry-After, marks that
   key exhausted for the rest of the order and retries on the next key.

Price tables, cost parsing, credential placement and the HTTPS transport are
the Arena broker's (``lab_arena.broker``), reused rather than copied.
"""

from __future__ import annotations

import json
import re
from decimal import Decimal
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Optional
from urllib.parse import urlencode

from lab_arena import broker as arena_broker
from lab_arena import operations
from lab_arena.broker import BrokerError, ProviderResponse, ProviderTransport, ProviderTransportError
from lab_arena.lead_orders import keys as lead_keys
from lab_arena.lead_orders.auth import order_token_hash
from lab_arena.lead_orders.policy import LeadOrderPolicy
from lab_arena.lead_orders.store import LeadOrderStore

OPENROUTER_HOST = "openrouter.ai"
DEEPLINE_HOST = "code.deepline.com"
OPENROUTER_CHAT_PATH = "api/v1/chat/completions"
_DEEPLINE_PATH_RE = re.compile(r"^api/v2/integrations/([a-z0-9_]{1,64})/execute$")
# The Deepline operations the Lead Search finding core calls (leadpoet-site
# server/lead-search, Nov 19 2026). Anything else, such as a monitor or a
# play that bills later, is refused. Extend by reviewed change.
DEEPLINE_OPERATIONS = frozenset({
    "harvestapi_search_jobs", "harvestapi_get_job", "harvestapi_get_company", "harvestapi_search_posts",
    "harvestapi_get_post_reactions", "harvestapi_get_post_comments", "harvestapi_get_profile_posts",
    "harvestapi_search_leads", "harvestapi_get_profile", "bloomberry_search_job_postings",
    "parallel_run_task", "parallel_get_task_run_result", "search_contact",
    "free_company_lookup", "crustdata_v3_company_identify", "crustdata_v3_company_enrich",
    "crustdata_v3_person_search", "prospeo_enrich_person", "zerobounce_validate", "hunter_email_finder",
    "dropleads_email_finder", "leadmagic_email_finder", "findymail_find_from_name",
    "findymail_find_from_business_profile", "crustdata_v3_job_search", "quickenrich_employee_email_enrich",
    "icypeas_email_verification", "icypeas_read_results",
})
# OpenRouter request fields whose cost the reservation covers: the prompt is
# priced from the whole body, the output (reasoning included, which counts
# against the same cap) from max_tokens or max_completion_tokens, each model
# of a fallback list at its own price, the dearest one reserved. Fields that
# add cost outside that (plugins, server-side tools, web search) are refused.
OPENROUTER_FIELDS = frozenset({
    "model", "models", "route", "messages", "max_tokens", "max_completion_tokens", "reasoning", "provider",
    "user", "temperature", "top_p", "top_k", "frequency_penalty", "presence_penalty", "repetition_penalty",
    "seed", "stop", "response_format", "tools", "tool_choice", "parallel_tool_calls", "stream", "usage",
})
MAX_FALLBACK_MODELS = 4
# Provider preferences the broker accepts. Routing preferences (order, only,
# ignore, sort) are refused: they are not needed by the runner and are kept
# out until a reviewed change prices them.
OPENROUTER_PROVIDER_FIELDS = frozenset({"data_collection", "zdr", "require_parameters", "max_price"})
# max_price components, USD per million tokens except request (USD per call).
_MAX_PRICE_PER_TOKEN = {"prompt": "prompt", "completion": "completion", "internal_reasoning": "internal_reasoning"}
_MAX_PRICE_FIELDS = frozenset({"prompt", "completion", "internal_reasoning", "request"})
_REASONING_FIELDS = frozenset({"enabled", "effort", "max_tokens", "exclude"})
_REASONING_EFFORTS = frozenset({"none", "minimal", "low", "medium", "high", "xhigh"})
MAX_REQUEST_BYTES = 1_048_576
MAX_RESPONSE_BYTES = 4 * 1_048_576
TIMEOUT_SECONDS = {"openrouter": 120.0, "deepline": 240.0}
_FORWARDED_RESPONSE_HEADERS = ("content-type",)


def _nonnegative_number(value: Any) -> bool:
    return type(value) in (int, float) and value == value and 0 <= value < 1e9


def _capped_pricing(pricing: Mapping[str, Any], max_price: Any) -> Dict[str, Any]:
    """Price each component at the request's own max_price where it sets one.

    OpenRouter never routes to an endpoint above max_price, so that is the
    real worst case for the component, whether above or below the listed
    price (the listed price is not a bound across endpoints).
    """

    priced = dict(pricing)
    if not isinstance(max_price, Mapping):
        return priced
    for name, value in max_price.items():
        per_unit = Decimal(str(value))
        priced[name] = str(per_unit / Decimal(1_000_000) if name in _MAX_PRICE_PER_TOKEN else per_unit)
    return priced


class LeadOrderBrokerError(Exception):
    def __init__(self, status: int, code: str) -> None:
        self.status = status
        self.code = code
        super().__init__(code)


@dataclass(frozen=True)
class BrokerRequest:
    provider: str
    method: str
    path: str
    query: Mapping[str, str]
    body: bytes
    content_type: Optional[str]


def _json_response(status: int, code: str) -> ProviderResponse:
    return ProviderResponse(
        status=status,
        headers={"content-type": "application/json"},
        body=json.dumps({"error": code}).encode("utf-8"),
    )


class LeadOrderBroker:
    def __init__(
        self,
        *,
        store: LeadOrderStore,
        policy: LeadOrderPolicy,
        transport: ProviderTransport,
        registrar: Optional[lead_keys.MinerKeyRegistrar],
        leadpoet_secrets: Mapping[str, str],
        price_table: Callable[[], Mapping[str, Any]],
        current_epoch: Callable[[], int],
    ) -> None:
        self._store = store
        self._policy = policy
        self._transport = transport
        self._registrar = registrar
        self._leadpoet_secrets = {k: v for k, v in leadpoet_secrets.items() if v}
        self._leadpoet_keys = lead_keys.leadpoet_keys(
            self._leadpoet_secrets.get("openrouter", ""), self._leadpoet_secrets.get("deepline", "")
        )
        self._price_table = price_table
        self._current_epoch = current_epoch

    # -- request shaping ----------------------------------------------------
    def _prepare(self, request: BrokerRequest) -> tuple:
        """Return (outbound without credential, reserve micro-USD, model)."""

        if len(request.body) > MAX_REQUEST_BYTES:
            raise LeadOrderBrokerError(413, "request_too_large")
        if request.provider == "openrouter":
            if request.method != "POST" or request.path != OPENROUTER_CHAT_PATH:
                raise LeadOrderBrokerError(404, "route_not_allowed")
            try:
                params = json.loads(request.body.decode("utf-8"))
            except (UnicodeDecodeError, ValueError):
                raise LeadOrderBrokerError(400, "invalid_json") from None
            if not isinstance(params, dict):
                raise LeadOrderBrokerError(400, "model_required")
            models = self._check_openrouter(params)
            if params.get("stream"):
                # A streamed body has no final usage object to settle from.
                raise LeadOrderBrokerError(400, "stream_not_supported")
            cap = self._policy.openrouter_max_output_tokens
            token_field = "max_completion_tokens" if "max_completion_tokens" in params else "max_tokens"
            requested = params.get(token_field)
            if requested is not None and (type(requested) is not int or requested < 1):
                raise LeadOrderBrokerError(400, "%s_invalid" % token_field)
            output_tokens = cap if requested is None else min(requested, cap)
            params[token_field] = output_tokens
            reasoning = params.get("reasoning")
            if isinstance(reasoning, dict) and "max_tokens" in reasoning:
                # Reasoning is output: its budget stays inside the output cap.
                reasoning["max_tokens"] = min(reasoning["max_tokens"], output_tokens)
            # Lead data never goes to an endpoint that keeps or trains on it.
            provider_prefs = dict(params.get("provider") or {})
            provider_prefs["data_collection"] = "deny"
            provider_prefs["zdr"] = True
            params["provider"] = provider_prefs
            params["usage"] = {"include": True}
            table = self._price_table()
            reserve = 0
            for model in models:
                pricing = table["models"].get(model)
                if pricing is None:
                    raise LeadOrderBrokerError(400, "model_not_allowed")
                pricing = _capped_pricing(pricing, provider_prefs.get("max_price"))
                reserve = max(reserve, arena_broker._max_openrouter_cost_for_pricing(
                    pricing, params, max_output_tokens=output_tokens, model=model
                ))
            body = json.dumps(params, separators=(",", ":")).encode("utf-8")
            target = operations.OutboundTarget("POST", "https", OPENROUTER_HOST, 443, "/" + OPENROUTER_CHAT_PATH)
            url = "https://%s/%s" % (OPENROUTER_HOST, OPENROUTER_CHAT_PATH)
            outbound = operations.OutboundRequest(
                target=target, url=url, query={}, body=body, content_type="application/json",
                credential=operations.CredentialPlacement("header", "authorization", scheme="Bearer"),
            )
            return outbound, max(1, int(reserve)), models
        if request.provider == "deepline":
            path = request.path
            matched = _DEEPLINE_PATH_RE.fullmatch(path)
            if request.method != "POST" or not matched:
                raise LeadOrderBrokerError(404, "route_not_allowed")
            if matched.group(1) not in DEEPLINE_OPERATIONS:
                raise LeadOrderBrokerError(400, "operation_not_allowed")
            if request.query:
                raise LeadOrderBrokerError(400, "query_not_allowed")
            query: Dict[str, str] = {}
            url = "https://%s/%s" % (DEEPLINE_HOST, path)
            if query:
                url += "?" + urlencode(sorted(query.items()))
            target = operations.OutboundTarget(request.method, "https", DEEPLINE_HOST, 443, "/" + path)
            outbound = operations.OutboundRequest(
                target=target, url=url, query=query, body=request.body if request.method == "POST" else b"",
                content_type="application/json" if request.method == "POST" else None,
                credential=operations.CredentialPlacement("header", "authorization", scheme="Bearer"),
            )
            return outbound, self._policy.deepline_reserve_per_call_microusd, None
        raise LeadOrderBrokerError(404, "provider_not_allowed")

    @staticmethod
    def _check_openrouter(params: Mapping[str, Any]) -> tuple:
        """Validate the body; return the models it may run on, primary first."""

        unknown = sorted(set(params) - OPENROUTER_FIELDS)
        if unknown:
            raise LeadOrderBrokerError(400, "field_not_allowed:%s" % unknown[0])
        models = []
        if "model" in params:
            if not isinstance(params["model"], str):
                raise LeadOrderBrokerError(400, "model_required")
            models.append(params["model"])
        if "models" in params:
            fallback = params["models"]
            if (not isinstance(fallback, list) or not fallback or len(fallback) > MAX_FALLBACK_MODELS
                    or not all(isinstance(model, str) for model in fallback)):
                raise LeadOrderBrokerError(400, "models_invalid")
            models.extend(model for model in fallback if model not in models)
        if not models:
            raise LeadOrderBrokerError(400, "model_required")
        if any(":" in model for model in models):
            raise LeadOrderBrokerError(400, "model_variant_not_allowed")  # e.g. ":online" adds web search
        if "route" in params and params["route"] != "fallback":
            raise LeadOrderBrokerError(400, "route_not_allowed")
        if "max_tokens" in params and "max_completion_tokens" in params:
            raise LeadOrderBrokerError(400, "max_tokens_invalid")
        if "user" in params and (not isinstance(params["user"], str) or len(params["user"]) > 128):
            raise LeadOrderBrokerError(400, "user_invalid")
        if "reasoning" in params:
            reasoning = params["reasoning"]
            if not isinstance(reasoning, dict) or set(reasoning) - _REASONING_FIELDS:
                raise LeadOrderBrokerError(400, "reasoning_invalid")
            budget = reasoning.get("max_tokens")
            if ("effort" in reasoning and reasoning["effort"] not in _REASONING_EFFORTS
                    or budget is not None and (type(budget) is not int or budget < 1)
                    or any(name in reasoning and type(reasoning[name]) is not bool for name in ("enabled", "exclude"))):
                raise LeadOrderBrokerError(400, "reasoning_invalid")
        if "provider" in params:
            prefs = params["provider"]
            if not isinstance(prefs, dict):
                raise LeadOrderBrokerError(400, "provider_invalid")
            extra = sorted(set(prefs) - OPENROUTER_PROVIDER_FIELDS)
            if extra:
                raise LeadOrderBrokerError(400, "provider_field_not_allowed:%s" % extra[0])
            if prefs.get("data_collection", "deny") != "deny" or prefs.get("zdr", True) is not True:
                raise LeadOrderBrokerError(400, "provider_privacy_required")
            if "require_parameters" in prefs and type(prefs["require_parameters"]) is not bool:
                raise LeadOrderBrokerError(400, "provider_invalid")
            if "max_price" in prefs:
                ceiling = prefs["max_price"]
                if (not isinstance(ceiling, dict) or not ceiling or set(ceiling) - _MAX_PRICE_FIELDS
                        or not all(_nonnegative_number(value) for value in ceiling.values())):
                    raise LeadOrderBrokerError(400, "max_price_invalid")
        for tool in params.get("tools") or []:
            if not isinstance(tool, Mapping) or tool.get("type") != "function":
                raise LeadOrderBrokerError(400, "tool_not_allowed")
        if "tools" in params and not isinstance(params["tools"], list):
            raise LeadOrderBrokerError(400, "tool_not_allowed")
        messages = params.get("messages")
        if not isinstance(messages, list):
            raise LeadOrderBrokerError(400, "messages_invalid")
        for message in messages:
            if not isinstance(message, Mapping):
                raise LeadOrderBrokerError(400, "messages_invalid")
            content = message.get("content")
            if isinstance(content, list) and any(
                not isinstance(part, Mapping) or part.get("type") != "text" for part in content
            ):
                raise LeadOrderBrokerError(400, "content_not_allowed")  # images and files are priced apart
        return tuple(models)

    def _cost(self, provider: str, models: Optional[tuple], response: ProviderResponse) -> Optional[int]:
        """Provider-reported cost in micro-USD, 0 for a refused call, else None."""

        status = int(response.status)
        if provider == "openrouter":
            try:
                document = json.loads(bytes(response.body).decode("utf-8"))
            except (UnicodeDecodeError, ValueError):
                document = None
            # OpenRouter's usage.cost is the charge for the model that answered.
            used = document.get("model") if isinstance(document, Mapping) else None
            model = used if models and used in models else (models[0] if models else "")
            cost = arena_broker.actual_openrouter_cost_microusd(self._price_table(), model, document)
        else:
            cost = arena_broker.deepline_cost_microusd(bytes(response.body))
        if cost is not None:
            return int(cost)
        # Providers do not bill a call they refused (4xx). A 2xx or 5xx with
        # no billing record is unknown: never guess, settle it uncertain.
        if 400 <= status < 500:
            return 0
        return None

    # -- the call -------------------------------------------------------------
    def call(self, token: str, request: BrokerRequest) -> ProviderResponse:
        order = self._store.by_token(order_token_hash(token)) if token else None
        if not order:
            raise LeadOrderBrokerError(401, "order_token_invalid")
        if order.get("finished"):
            raise LeadOrderBrokerError(409, "order_finished")
        if order.get("expired"):
            raise LeadOrderBrokerError(410, "order_expired")
        order_id = str(order["order_id"])
        outbound, reserve, models = self._prepare(request)
        provider = request.provider
        epoch = int(self._current_epoch())
        # Decided per order, from the mode it was placed in: a staff order
        # spends only Leadpoet's keys even after the gateway moves to shadow
        # or live, since staff orders never repay miners.
        use_miner_keys = order.get("mode") in ("shadow", "live") and self._registrar is not None
        window_start = max(0, epoch - self._policy.ranking_window_epochs + 1)
        miner_keys = []
        if use_miner_keys:
            for row in self._store.keys(provider):
                miner_keys.append(lead_keys.ProviderKey(
                    key_id=row["key_id"], provider=row["provider"], miner_hotkey=row["miner_hotkey"],
                    registered_sequence=int(row["sequence"]), internal_fingerprint=row["internal_fingerprint"],
                    ciphertext_b64=row["ciphertext_b64"],
                ))
        ranked = lead_keys.rank_keys(
            provider, miner_keys, self._leadpoet_keys.get(provider),
            settled_spend_by_key=self._store.key_spend(since_epoch=window_start) if miner_keys else {},
            use_miner_keys=use_miner_keys,
            exhausted=self._store.exhausted_keys(order_id),
        )
        for key in ranked:
            try:
                secret = self._leadpoet_secrets[provider] if key.is_leadpoet else self._registrar.decrypt(key)
            except lead_keys.LeadOrderKeyError:
                self._store.exhaust_key(order_id=order_id, key_id=key.key_id, reason=lead_keys.FAILOVER_UNAUTHORIZED)
                continue
            call_id = str(uuid.uuid4())
            reserved = self._store.reserve_call(
                call_id=call_id, order_id=order_id, provider=provider, key_id=key.key_id,
                miner_hotkey=key.miner_hotkey, public_fingerprint=key.public_fingerprint, reserve_microusd=reserve,
            )
            status = reserved.get("status")
            if status == "key_exhausted":
                continue
            if status == "ceiling":
                return _json_response(402, "spend_ceiling")
            if status == "expired":
                raise LeadOrderBrokerError(410, "order_expired")
            if status != "reserved":
                raise LeadOrderBrokerError(409, "order_closed")
            response: Optional[ProviderResponse] = None
            try:
                url, headers = arena_broker.inject_credential(outbound, secret)
                response = self._transport.send(
                    method=outbound.target.method, url=url, headers=headers, body=outbound.body,
                    timeout_seconds=TIMEOUT_SECONDS[provider], max_response_bytes=MAX_RESPONSE_BYTES,
                )
            except ProviderTransportError:
                response = None
            finally:
                # Always settle the reservation, whatever happened in between.
                cost = None if response is None else self._cost(provider, models, response)
                self._store.settle_call(
                    call_id=call_id, cost_microusd=cost,
                    http_status=None if response is None else int(response.status), epoch=epoch,
                )
            if response is None:
                return _json_response(502, "provider_unreachable")
            reason = lead_keys.failover_reason(provider, response)
            if reason is not None:
                self._store.exhaust_key(order_id=order_id, key_id=key.key_id, reason=reason)
                continue
            headers = {k.lower(): v for k, v in response.headers.items()}
            return ProviderResponse(
                status=int(response.status),
                headers={name: headers[name] for name in _FORWARDED_RESPONSE_HEADERS if name in headers},
                body=bytes(response.body),
            )
        return _json_response(503, "provider_keys_exhausted")


__all__ = ["BrokerRequest", "LeadOrderBroker", "LeadOrderBrokerError"]
