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
    "findymail_find_from_business_profile",
})
# OpenRouter request fields whose cost the reservation covers: the prompt is
# priced from the messages and tools, the output from max_tokens. Fields that
# add cost outside that (models fallback lists, plugins, provider routing,
# server-side tools, reasoning budgets, web search) are refused.
OPENROUTER_FIELDS = frozenset({
    "model", "messages", "max_tokens", "temperature", "top_p", "top_k", "frequency_penalty",
    "presence_penalty", "repetition_penalty", "seed", "stop", "response_format", "tools", "tool_choice",
    "parallel_tool_calls", "stream", "usage",
})
MAX_REQUEST_BYTES = 1_048_576
MAX_RESPONSE_BYTES = 4 * 1_048_576
TIMEOUT_SECONDS = {"openrouter": 120.0, "deepline": 240.0}
_FORWARDED_RESPONSE_HEADERS = ("content-type",)


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
            if not isinstance(params, dict) or not isinstance(params.get("model"), str):
                raise LeadOrderBrokerError(400, "model_required")
            self._check_openrouter(params)
            if params.get("stream"):
                # A streamed body has no final usage object to settle from.
                raise LeadOrderBrokerError(400, "stream_not_supported")
            cap = self._policy.openrouter_max_output_tokens
            requested = params.get("max_tokens")
            if requested is not None and (type(requested) is not int or requested < 1):
                raise LeadOrderBrokerError(400, "max_tokens_invalid")
            params["max_tokens"] = cap if requested is None else min(requested, cap)
            params["usage"] = {"include": True}
            table = self._price_table()
            try:
                reserve = arena_broker.max_openrouter_cost_microusd(
                    table, params["model"], params, max_output_tokens=params["max_tokens"]
                )
            except BrokerError:
                raise LeadOrderBrokerError(400, "model_not_allowed") from None
            body = json.dumps(params, separators=(",", ":")).encode("utf-8")
            target = operations.OutboundTarget("POST", "https", OPENROUTER_HOST, 443, "/" + OPENROUTER_CHAT_PATH)
            url = "https://%s/%s" % (OPENROUTER_HOST, OPENROUTER_CHAT_PATH)
            outbound = operations.OutboundRequest(
                target=target, url=url, query={}, body=body, content_type="application/json",
                credential=operations.CredentialPlacement("header", "authorization", scheme="Bearer"),
            )
            return outbound, max(1, int(reserve)), params["model"]
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
    def _check_openrouter(params: Mapping[str, Any]) -> None:
        unknown = sorted(set(params) - OPENROUTER_FIELDS)
        if unknown:
            raise LeadOrderBrokerError(400, "field_not_allowed:%s" % unknown[0])
        if ":" in params["model"]:
            raise LeadOrderBrokerError(400, "model_variant_not_allowed")  # e.g. ":online" adds web search
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

    def _cost(self, provider: str, model: Optional[str], response: ProviderResponse) -> Optional[int]:
        """Provider-reported cost in micro-USD, 0 for a refused call, else None."""

        status = int(response.status)
        if provider == "openrouter":
            try:
                document = json.loads(bytes(response.body).decode("utf-8"))
            except (UnicodeDecodeError, ValueError):
                document = None
            cost = arena_broker.actual_openrouter_cost_microusd(self._price_table(), model or "", document)
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
        outbound, reserve, model = self._prepare(request)
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
                cost = None if response is None else self._cost(provider, model, response)
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
