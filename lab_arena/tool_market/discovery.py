"""What an agent that knows only the netuid reads to use the market.

``set_service`` on chain points to ``services_document``: who publishes
(the Gamma service hotkey), the OpenAPI spec, how to pay (Gamma, btauth/1)
and where the balance and prices are. ``openapi`` is built from the catalog,
so every operation's parameters are in the spec and cannot drift from what
``/call`` accepts. Prices move with the book, so they are read live from
``/tool-market/catalog``.
"""

from __future__ import annotations

from typing import Any, Dict

from lab_arena.tool_market import catalog
from lab_arena.tool_market.config import TERMS_URL, ToolMarketConfig

_KINDS = {"str": {"type": "string"}, "int": {"type": "integer"}, "bool": {"type": "boolean"},
          "str_list": {"type": "array", "items": {"type": "string"}}, "enum": {"type": "string"}}


def services_document(config: ToolMarketConfig) -> Dict[str, Any]:
    base = config.public_base_url
    return {
        "version": 1,
        "name": "Subnet 71 tool market",
        "description": "Pay-per-call GTM data tools (email checks, people and company lookups, web search) "
                       "on vendor keys that miners supply. Each call costs its supplier's ask plus 10%.",
        "publishers": [{"netuid": config.netuid, "hotkey": config.gamma_service}],
        "services": [{"type": "openapi", "endpoint": base + "/tool-market/openapi.json",
                      "spec": base + "/tool-market/openapi.json"}],
        "payments": [{"type": "gamma", "signature": "btauth/1", "balance": base + "/tool-market/gamma/balance"}],
        "pricing": base + "/tool-market/catalog",
        # A paid request accepts these terms; their text's SHA-256 pins the version.
        "terms": {"version": config.terms_version, "sha256": config.terms_sha256,
                  "url": TERMS_URL},
    }


def _operation_schema(op: catalog.Operation) -> Dict[str, Any]:
    params = {name: dict(_KINDS[p.kind], **({"enum": list(p.choices)} if p.choices else {}))
              for name, p in op.params.items()}
    required = sorted(name for name, p in op.params.items() if p.required)
    return {"type": "object", "description": op.does, "required": ["tool", "operation", "params"],
            "properties": {"tool": {"enum": [op.tool]}, "operation": {"enum": [op.name]},
                           "params": {"type": "object", "properties": params, "required": required,
                                      "additionalProperties": False},
                           "max_ask_microusd": {"type": "integer", "description": "Skip keys asking more per unit."},
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
                           "on netuid %d first." % config.netuid}}},
        "paths": {
            "/tool-market/call": {"post": {
                "summary": "Call one tool operation, charged to the signing hotkey's Gamma balance.",
                "security": [{"btauth": []}],
                "requestBody": {"required": True, "content": {"application/json": {"schema": {
                    "oneOf": [_operation_schema(op) for op in operations]}}}},
                "responses": {
                    "200": {"description": "The vendor's answer: call_id, status, body and charge_microusd."},
                    "402": dict(error, description="insufficient_balance: spend more Gamma; nothing was done."),
                    "401": dict(error, description="A btauth/1 problem, named by the error code."),
                    "429": dict(error, description="Too many calls in flight or uncertain for this hotkey."),
                    "503": dict(error, description="No key can serve the call now (no_supply), or the chain "
                                                   "or store is down.")}}},
            "/tool-market/catalog": {"get": {
                "summary": "Every tool's operations, parameters and current price per unit (micro-USD).",
                "responses": {"200": {"description": "The catalog."}}}},
            "/tool-market/books/{tool}": {"get": {
                "summary": "One tool's asks, cheapest first.",
                "parameters": [{"name": "tool", "in": "path", "required": True, "schema": {"type": "string"}}],
                "responses": {"200": {"description": "The book."}, "404": error}}},
            "/tool-market/gamma/balance": {"get": {
                "summary": "A hotkey's balance in USD, as decimal strings; zeros for a hotkey that never spent.",
                "parameters": [{"name": "hotkey", "in": "query", "required": True, "schema": {"type": "string"}}],
                "responses": {"200": {"description": "{hotkey, credited_usd, used_usd, remaining_usd}"},
                              "400": error}}},
        },
    }


__all__ = ["openapi", "services_document"]
