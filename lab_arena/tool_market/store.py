"""Tool-market storage: thin typed calls to the ``tool_market_*_v1`` functions.

All rules on stored state live in ``scripts/770-tool-market.sql`` and tests
run them on a disposable PostgreSQL, as ``lab_arena.lead_orders.store`` does.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from lab_arena import store as arena_store
from lab_arena.store import StoreTransport

_T, _U, _I, _B, _N, _BOOL = "text", "uuid", "integer", "bigint", "numeric", "boolean"

FUNCTION_SIGNATURES: Dict[str, Sequence[tuple]] = {
    "tool_market_record_gamma_reading_v1": (
        ("p_hotkey", _T), ("p_block_number", _B), ("p_block_hash", _T), ("p_units", _N), ("p_available", _BOOL),
    ),
    "tool_market_use_nonce_v1": (("p_hotkey", _T), ("p_signature_hash", _T), ("p_signed_at", _B)),
    "tool_market_accept_terms_v1": (
        ("p_role", _T), ("p_hotkey", _T), ("p_version", _T), ("p_terms_sha256", _T), ("p_host", _T),
        ("p_signed_at", _B), ("p_body_sha256", _T), ("p_signature", _T),
    ),
    "tool_market_open_tab_v1": (
        ("p_tab_id", _U), ("p_hotkey", _T), ("p_token_hash", _T), ("p_budget", _B), ("p_seconds", _I),
        ("p_terms_sha256", _T),
    ),
    "tool_market_tab_v1": (("p_token_hash", _T),),
    "tool_market_tab_state_v1": (("p_token_hash", _T),),
    "tool_market_close_tab_v1": (("p_token_hash", _T),),
    "tool_market_register_key_v1": (
        ("p_key_id", _U), ("p_tool", _T), ("p_hotkey", _T), ("p_envelope", _T), ("p_fingerprint", _T),
        ("p_daily_cap", _B), ("p_status", _T), ("p_host", _T), ("p_signed_at", _B),
        ("p_signed_body", _T), ("p_signature", _T),
    ),
    "tool_market_key_v1": (("p_key_id", _U),),
    "tool_market_revoke_key_v1": (("p_key_id", _U), ("p_hotkey", _T), ("p_reason", _T)),
    "tool_market_record_key_event_v1": (("p_key_id", _U), ("p_status", _T), ("p_reason", _T)),
    "tool_market_post_ask_v1": (("p_key_id", _U), ("p_hotkey", _T), ("p_ask", _B)),
    "tool_market_book_v1": (("p_tool", _T), ("p_terms_sha256", _T)),
    "tool_market_reserve_call_v1": (
        ("p_call_id", _U), ("p_token_hash", _T), ("p_tool", _T), ("p_operation", _T), ("p_key_id", _U),
        ("p_ask", _B), ("p_reserve", _B), ("p_owed_cap", _B),
    ),
    "tool_market_settle_call_v1": (
        ("p_call_id", _U), ("p_status", _T), ("p_http_status", _I), ("p_units", _B), ("p_owed", _B),
        ("p_charge", _B), ("p_epoch", _B),
    ),
    "tool_market_statement_v1": (("p_hotkey", _T),),
    "tool_market_usage_v1": (("p_hotkey", _T), ("p_after", _B), ("p_limit", _I)),
}


def register_rpc_signatures() -> None:
    """Add these functions to the Arena transports' allow-list; never replace one."""

    for name, signature in FUNCTION_SIGNATURES.items():
        existing = arena_store.FUNCTION_SIGNATURES.get(name)
        if existing is None:
            arena_store.FUNCTION_SIGNATURES[name] = signature
        elif tuple(existing) != tuple(signature):
            raise RuntimeError("tool market function signature conflicts: %s" % name)


class ToolMarketStore:
    def __init__(self, transport: StoreTransport) -> None:
        register_rpc_signatures()
        self._t = transport

    def _rpc(self, function: str, **params: Any) -> Any:
        return self._t.rpc(function, params)

    # Buyers -----------------------------------------------------------------
    def record_gamma_reading(self, *, hotkey: str, block_number: int, block_hash: str, spent_units: int,
                             available: bool) -> Dict[str, Any]:
        return self._rpc("tool_market_record_gamma_reading_v1", p_hotkey=hotkey, p_block_number=int(block_number),
                         p_block_hash=block_hash, p_units=str(int(spent_units)), p_available=bool(available))

    def use_nonce(self, *, hotkey: str, signature_hash: str, signed_at: int) -> bool:
        return bool(self._rpc("tool_market_use_nonce_v1", p_hotkey=hotkey, p_signature_hash=signature_hash,
                              p_signed_at=int(signed_at)))

    def accept_terms(self, *, role: str, hotkey: str, version: str, terms_sha256: str, host: str, signed_at: int,
                     body_sha256: str, signature: str) -> Dict[str, Any]:
        return self._rpc("tool_market_accept_terms_v1", p_role=role, p_hotkey=hotkey, p_version=version,
                         p_terms_sha256=terms_sha256, p_host=host, p_signed_at=int(signed_at),
                         p_body_sha256=body_sha256, p_signature=signature)

    def open_tab(self, *, tab_id: str, hotkey: str, token_hash: str, budget_microusd: int, seconds: int,
                 terms_sha256: str) -> Dict[str, Any]:
        return self._rpc("tool_market_open_tab_v1", p_tab_id=tab_id, p_hotkey=hotkey, p_token_hash=token_hash,
                         p_budget=int(budget_microusd), p_seconds=int(seconds), p_terms_sha256=terms_sha256)

    def tab(self, token_hash: str) -> Optional[Dict[str, Any]]:
        return self._rpc("tool_market_tab_v1", p_token_hash=token_hash)

    def tab_state(self, token_hash: str) -> Optional[Dict[str, Any]]:
        return self._rpc("tool_market_tab_state_v1", p_token_hash=token_hash)

    def close_tab(self, token_hash: str) -> Optional[Dict[str, Any]]:
        return self._rpc("tool_market_close_tab_v1", p_token_hash=token_hash)

    # Miners -----------------------------------------------------------------
    def register_key(self, *, key_id: str, tool: str, hotkey: str, envelope: str, fingerprint: str,
                     daily_cap_microusd: int, status: str, host: str, signed_at: int,
                     signed_body: str, signature: str) -> Dict[str, Any]:
        return self._rpc("tool_market_register_key_v1", p_key_id=key_id, p_tool=tool, p_hotkey=hotkey,
                         p_envelope=envelope, p_fingerprint=fingerprint, p_daily_cap=int(daily_cap_microusd),
                         p_status=status, p_host=host, p_signed_at=int(signed_at),
                         p_signed_body=signed_body, p_signature=signature)

    def key(self, key_id: str) -> Optional[Dict[str, Any]]:
        return self._rpc("tool_market_key_v1", p_key_id=key_id)

    def revoke_key(self, *, key_id: str, hotkey: str, reason: str) -> Dict[str, Any]:
        return self._rpc("tool_market_revoke_key_v1", p_key_id=key_id, p_hotkey=hotkey, p_reason=reason)

    def record_key_event(self, *, key_id: str, status: str, reason: Optional[str] = None) -> None:
        self._rpc("tool_market_record_key_event_v1", p_key_id=key_id, p_status=status, p_reason=reason)

    def post_ask(self, *, key_id: str, hotkey: str, ask_microusd: Optional[int]) -> Dict[str, Any]:
        return self._rpc("tool_market_post_ask_v1", p_key_id=key_id, p_hotkey=hotkey,
                         p_ask=None if ask_microusd is None else int(ask_microusd))

    def book(self, *, tool: str, terms_sha256: str) -> List[Dict[str, Any]]:
        return list(self._rpc("tool_market_book_v1", p_tool=tool, p_terms_sha256=terms_sha256) or [])

    def statement(self, hotkey: str) -> Dict[str, Any]:
        return self._rpc("tool_market_statement_v1", p_hotkey=hotkey)

    def usage(self, *, hotkey: str, after: int, limit: int) -> Dict[str, Any]:
        return self._rpc("tool_market_usage_v1", p_hotkey=hotkey, p_after=int(after), p_limit=int(limit))

    # Calls ------------------------------------------------------------------
    def reserve_call(self, *, call_id: str, token_hash: str, tool: str, operation: str, key_id: str,
                     ask_microusd: int, reserve_microusd: int, owed_cap_microusd: int) -> str:
        return str(self._rpc("tool_market_reserve_call_v1", p_call_id=call_id, p_token_hash=token_hash, p_tool=tool,
                             p_operation=operation, p_key_id=key_id, p_ask=int(ask_microusd),
                             p_reserve=int(reserve_microusd), p_owed_cap=int(owed_cap_microusd))["status"])

    def settle_call(self, *, call_id: str, status: str, http_status: Optional[int], units: int, owed_microusd: int,
                    charge_microusd: int, epoch: int) -> str:
        return str(self._rpc("tool_market_settle_call_v1", p_call_id=call_id, p_status=status,
                             p_http_status=http_status, p_units=int(units), p_owed=int(owed_microusd),
                             p_charge=int(charge_microusd), p_epoch=int(epoch))["status"])


__all__ = ["ToolMarketStore"]
