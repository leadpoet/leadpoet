"""What an agent that knows only the netuid reads to use the market.

``set_service`` on chain points to ``services_document``: who publishes
(the Gamma service hotkey), the OpenAPI spec, how to pay (Gamma, btauth/1)
and where the balance and prices are. ``openapi`` is built from the catalog,
so each parameter's type and bounds in the spec are the ones ``/call``
checks; /call also applies an operation's cross-field rules and a GET
request's encoded length limit. Prices
move with the book, so they are read live from ``/tool-market/catalog``.
"""

from __future__ import annotations

from typing import Any, Dict

from lab_arena.tool_market import catalog
from lab_arena.tool_market.config import NETUID, TERMS_URL, ToolMarketConfig


def services_document(config: ToolMarketConfig) -> Dict[str, Any]:
    base = config.public_base_url
    return {
        "version": 1,
        "name": "Subnet 71 tool market",
        "description": "Pay-per-call GTM data tools (email checks, people and company lookups, web search) "
                       "on vendor keys that miners supply. Each call costs its supplier's ask plus 10%.",
        "publishers": [{"netuid": NETUID, "hotkey": config.gamma_service}],
        "services": [{"type": "openapi", "endpoint": base,
                      "spec": base + "/tool-market/openapi.json"}],
        "payments": [{"type": "gamma", "signature": "btauth/1", "balance": base + "/tool-market/gamma/balance"}],
        "pricing": base + "/tool-market/catalog",
        # A paid request accepts these terms; their text's SHA-256 pins the version.
        "terms": {"version": config.terms_version, "sha256": config.terms_sha256,
                  "url": TERMS_URL},
    }


def _operation_schema(op: catalog.Operation) -> Dict[str, Any]:
    params = {name: p.schema() for name, p in op.params.items()}
    required = sorted(name for name, p in op.params.items() if p.required)
    return {"type": "object", "description": op.does,
            "required": ["tool", "operation"] + (["params"] if required else []),
            "properties": {"tool": {"enum": [op.tool]}, "operation": {"enum": [op.name]},
                           "request_id": {"type": "string", "format": "uuid", "pattern": "^[0-9a-f-]{36}$",
                                          "description": "Lowercase; reuse for retries of this purchase."},
                           "params": {"type": "object", "properties": params, "required": required,
                                      "additionalProperties": False},
                           "max_ask_microusd": {"type": "integer", "minimum": 1,
                                                "maximum": catalog.MAX_EXACT_ASK,
                                                "description": "Skip keys asking more per unit."},
                           "private": {"type": "boolean",
                                       "description": "Only suppliers that keep the request private."}}}


def openapi(config: ToolMarketConfig) -> Dict[str, Any]:
    error = {"description": "Refused; nothing was charged.", "content": {"application/json": {"schema": {
        "type": "object", "required": ["error"], "properties": {"error": {"type": "string"}}}}}}
    operations = [op for (tool, _), op in sorted(catalog.OPERATIONS.items()) if tool in config.tools]
    return {
        "openapi": "3.0.3",
        "info": {"title": "Subnet 71 tool market", "version": "1"},
        "servers": [{"url": config.public_base_url}],
        "components": {"securitySchemes": {"btauth": {
            "type": "apiKey", "in": "header", "name": "X-Bittensor-Signature",
            "description": "btauth/1 (https://www.bittensor.com/docs/guides/signed-requests), sr25519, with "
                           "X-Bittensor-Receiver " + config.gamma_service + ". Spend Gamma at that hotkey "
                           "on netuid %d first." % NETUID},
            "tab": {"type": "http", "scheme": "bearer",
                    "description": "An API key from POST /tool-market/tabs, which holds a budget."}}},
        "paths": {
            "/tool-market/call": {"post": {
                "summary": "Call one tool operation, charged to the signing hotkey's Gamma balance.",
                "security": [{"btauth": []}, {"tab": []}],
                "requestBody": {"required": True, "content": {"application/json": {"schema": {
                    "oneOf": [_operation_schema(op) for op in operations]}}}},
                "responses": {
                    "200": {"description": "The vendor's answer: call_id, status, body and charge_microusd."},
                    "400": dict(error, description="The request names no catalog operation (operation_unknown) or does not fit it (param_invalid "
                                                   "also covers its cross-field rules and a GET request's "
                                                   "encoded length)."),
                    "402": dict(error, description="insufficient_balance: spend more Gamma; tab_budget: the "
                                                   "API key's budget is spent. Nothing was charged."),
                    "401": dict(error, description="A btauth/1 problem, or tab_unknown or tab_closed, named by "
                                                   "the error code."),
                    "409": dict(error, description="request_already_used: this request_id already bought, "
                                                   "see possible_call_ids; session_page_one_required: "
                                                   "a lead-search session starts at page one."),
                    "403": dict(error, description="buyer_not_allowed: this hotkey may not buy yet."),
                    "404": dict(error, description="tool_unavailable."),
                    "413": dict(error, description="request_too_large."),
                    "429": dict(error, description="Too many calls in flight or uncertain for this hotkey."),
                    "502": dict(error, description="vendor_unreachable: the request may have reached the "
                                                   "vendor. Not charged; carries call_id. Retry with the same "
                                                   "request_id, never a new one."),
                    "503": dict(error, description="No key can serve the call now (no_supply), or the chain "
                                                   "or store is down or busy; retry later. One with a call_id may "
                                                   "have been charged: usage shows it after 11 minutes.")}}},
            "/tool-market/catalog": {"get": {
                "summary": "Every tool's operations, parameters and current price per unit (micro-USD).",
                "responses": {"200": {"description": "The catalog."}}}},
            "/tool-market/books/{tool}": {"get": {
                "summary": "One tool's asks, cheapest first.",
                "parameters": [{"name": "tool", "in": "path", "required": True, "schema": {"type": "string"}},
                               {"name": "operation", "in": "query", "required": False, "schema": {"type": "string"},
                                "description": "Quote this SKU, including its eligible general asks."}],
                "responses": {"200": {"description": "The book."}, "404": error}}},
            "/tool-market/gamma/balance": {"get": {
                "summary": "A hotkey's balance in USD, as decimal strings; zeros for a hotkey that never spent.",
                "parameters": [{"name": "hotkey", "in": "query", "required": True, "schema": {"type": "string"}}],
                "responses": {"200": {"description": "{hotkey, credited_usd, used_usd, remaining_usd}"},
                              "400": error}}},
        },
    }


__all__ = ["openapi", "services_document"]
