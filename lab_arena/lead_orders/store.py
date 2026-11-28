"""Lead-order storage: thin typed calls to the ``lead_order_*_v1`` functions.

All state lives in the append-only tables of ``scripts/707-lead-orders.sql``
and every read and write goes through its SECURITY DEFINER functions, the
same way ``lab_arena.store`` reaches the Arena tables. The Arena transports
(PostgREST in production, psycopg2 in tests) are reused as they are; this
module only adds the lead-order function signatures to their allow-list.
There is deliberately no in-memory copy of the SQL rules: tests run them on a
disposable PostgreSQL so the rules exist once.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence

from lab_arena import store as arena_store
from lab_arena.store import StoreTransport

_T = "text"
_U = "uuid"
_I = "integer"
_B = "bigint"
_N = "numeric"
_J = "jsonb"
_BOOL = "boolean"

LEAD_ORDER_FUNCTION_SIGNATURES: Dict[str, Sequence[tuple]] = {
    "lead_order_balance_v1": (("p_hotkey", _T),),
    "lead_order_record_policy_v1": (("p_version", _I), ("p_hash", _T), ("p_doc", _J)),
    "lead_order_record_gamma_reading_v1": (
        ("p_hotkey", _T), ("p_block_number", _B), ("p_block_hash", _T), ("p_units", _N), ("p_available", _BOOL),
    ),
    "lead_order_register_buyer_v1": (
        ("p_registration_id", _U), ("p_hotkey", _T), ("p_email", _T), ("p_company", _T),
        ("p_country", _T), ("p_terms", _T), ("p_token_hash", _T),
    ),
    "lead_order_confirm_buyer_v1": (("p_token_hash", _T), ("p_max_age_seconds", _I)),
    "lead_order_buyer_v1": (("p_hotkey", _T),),
    "lead_order_use_nonce_v1": (("p_hotkey", _T), ("p_signature_hash", _T), ("p_signed_at", _B)),
    "lead_order_place_v1": (
        ("p_order_id", _U), ("p_hotkey", _T), ("p_mode", _T), ("p_policy_version", _I),
        ("p_leads", _I), ("p_request_doc", _J),
    ),
    "lead_order_claim_v1": (("p_token_hash", _T),),
    "lead_order_by_token_v1": (("p_token_hash", _T),),
    "lead_order_reserve_call_v1": (
        ("p_call_id", _U), ("p_order_id", _U), ("p_provider", _T), ("p_key_id", _T),
        ("p_miner_hotkey", _T), ("p_public_fingerprint", _T), ("p_reserve", _B),
    ),
    "lead_order_settle_call_v1": (
        ("p_call_id", _U), ("p_status", _T), ("p_cost", _B), ("p_http_status", _I), ("p_epoch", _B),
        ("p_request_id", _T),
    ),
    "lead_order_exhaust_key_v1": (("p_order_id", _U), ("p_key_id", _T), ("p_reason", _T)),
    "lead_order_exhausted_keys_v1": (("p_order_id", _U),),
    "lead_order_finish_v1": (
        ("p_order_id", _U), ("p_status", _T), ("p_delivered", _I), ("p_delivery_url", _T),
        ("p_failure_reason", _T), ("p_epoch", _B), ("p_allow_open", _BOOL),
    ),
    "lead_order_overdue_v1": (("p_grace_seconds", _I),),
    "lead_order_view_v1": (("p_order_id", _U),),
    "lead_order_calls_v1": (("p_order_id", _U),),
    "lead_order_record_rewards_v1": (("p_order_id", _U), ("p_rewards", _J)),
    "lead_order_record_receipt_v1": (("p_order_id", _U), ("p_hash", _T), ("p_doc", _J), ("p_calls", _J)),
    "lead_order_register_key_v1": (
        ("p_key_id", _T), ("p_provider", _T), ("p_miner_hotkey", _T), ("p_submission_id", _T),
        ("p_ciphertext", _T), ("p_internal_fp", _T), ("p_public_fp", _T), ("p_daily_cap", _B),
    ),
    "lead_order_revoke_key_v1": (("p_key_id", _T), ("p_reason", _T)),
    "lead_order_keys_v1": (("p_provider", _T), ("p_terms_sha256", _T)),
    "lead_order_accept_miner_terms_v1": (
        ("p_hotkey", _T), ("p_version", _T), ("p_terms_sha256", _T), ("p_host", _T), ("p_signed_at", _B),
        ("p_body_sha256", _T), ("p_signature", _T),
    ),
    "lead_order_miner_terms_v1": (("p_hotkey", _T), ("p_terms_sha256", _T)),
    "lead_order_add_suppressions_v1": (("p_rows", _J),),
    "lead_order_suppressions_v1": (("p_since", _B), ("p_limit", _I)),
    "lead_order_key_spend_v1": (("p_since_epoch", _B),),
    "lead_order_owed_v1": (("p_through_epoch", _B), ("p_mode", _T)),
    "lead_order_record_payout_v1": (
        ("p_epoch", _B), ("p_mode", _T), ("p_doc_hash", _T), ("p_doc", _J), ("p_payments", _J),
    ),
    "lead_order_reward_epoch_v1": (("p_epoch", _B),),
    "lead_order_policy_v1": (("p_version", _I),),
    "lead_order_unsettled_v1": (("p_limit", _I),),
    "lead_order_uncertain_calls_v1": (("p_limit", _I),),
    "lead_order_resolve_call_v1": (("p_call_id", _U), ("p_cost", _B), ("p_epoch", _B)),
    "lead_order_record_readback_v1": (("p_call_id", _U), ("p_outcome", _T)),
    "lead_order_reward_late_v1": (("p_epoch", _B), ("p_limit", _I)),
    "lead_order_statement_v1": (("p_hotkey", _T), ("p_since", _B), ("p_limit", _I)),
    "lead_order_record_vault_receipt_v1": (("p_call_id", _U), ("p_kind", _T), ("p_receipt", _J)),
    "lead_order_vault_receipts_v1": (("p_hotkey", _T), ("p_since", _B), ("p_limit", _I)),
}


def register_rpc_signatures() -> None:
    """Add the lead-order functions to the Arena transports' allow-list.

    Idempotent; refuses to replace any existing entry so an Arena function
    can never be shadowed.
    """

    for name, signature in LEAD_ORDER_FUNCTION_SIGNATURES.items():
        existing = arena_store.FUNCTION_SIGNATURES.get(name)
        if existing is None:
            arena_store.FUNCTION_SIGNATURES[name] = signature
        elif tuple(existing) != tuple(signature):
            raise RuntimeError("lead order function signature conflicts: %s" % name)


class LeadOrderStore:
    def __init__(self, transport: StoreTransport) -> None:
        register_rpc_signatures()
        self._t = transport

    def _rpc(self, function: str, **params: Any) -> Any:
        return self._t.rpc(function, params)

    # Policy and balance -----------------------------------------------------
    def record_policy(self, document: Mapping[str, Any]) -> Dict[str, Any]:
        return self._rpc(
            "lead_order_record_policy_v1",
            p_version=int(document["version"]), p_hash=document["policy_hash"], p_doc=dict(document),
        )

    def record_gamma_reading(self, *, hotkey: str, block_number: int, block_hash: str,
                             spent_units: int, available: bool) -> Dict[str, Any]:
        return self._rpc(
            "lead_order_record_gamma_reading_v1", p_hotkey=hotkey, p_block_number=int(block_number),
            p_block_hash=block_hash, p_units=str(int(spent_units)), p_available=bool(available),
        )

    def balance(self, hotkey: str) -> Dict[str, Any]:
        return self._rpc("lead_order_balance_v1", p_hotkey=hotkey)

    # Buyers -----------------------------------------------------------------
    def register_buyer(self, *, registration_id: str, hotkey: str, email: str, company: str,
                       country: str, terms_version: str, token_hash: str) -> Dict[str, Any]:
        return self._rpc(
            "lead_order_register_buyer_v1", p_registration_id=registration_id, p_hotkey=hotkey,
            p_email=email, p_company=company, p_country=country, p_terms=terms_version, p_token_hash=token_hash,
        )

    def confirm_buyer(self, token_hash: str, *, max_age_seconds: int) -> Dict[str, Any]:
        return self._rpc("lead_order_confirm_buyer_v1", p_token_hash=token_hash, p_max_age_seconds=int(max_age_seconds))

    def buyer(self, hotkey: str) -> Dict[str, Any]:
        return self._rpc("lead_order_buyer_v1", p_hotkey=hotkey)

    def use_nonce(self, *, hotkey: str, signature_hash: str, signed_at: int) -> bool:
        return bool(self._rpc("lead_order_use_nonce_v1", p_hotkey=hotkey, p_signature_hash=signature_hash,
                              p_signed_at=int(signed_at)))

    # Orders -----------------------------------------------------------------
    def place(self, *, order_id: str, hotkey: str, mode: str, policy_version: int, leads: int,
              request_doc: Mapping[str, Any]) -> Dict[str, Any]:
        return self._rpc(
            "lead_order_place_v1", p_order_id=order_id, p_hotkey=hotkey, p_mode=mode,
            p_policy_version=int(policy_version), p_leads=int(leads), p_request_doc=dict(request_doc),
        )

    def claim(self, token_hash: str) -> Optional[Dict[str, Any]]:
        return self._rpc("lead_order_claim_v1", p_token_hash=token_hash)

    def by_token(self, token_hash: str) -> Optional[Dict[str, Any]]:
        return self._rpc("lead_order_by_token_v1", p_token_hash=token_hash)

    def finish(self, *, order_id: str, status: str, delivered: int, delivery_url: Optional[str],
               failure_reason: Optional[str], epoch: int, allow_open: bool = False) -> Dict[str, Any]:
        """``allow_open`` is for the overdue sweep only: it settles every open
        call uncertain. Without it a finish returns ``calls_open`` while a call
        opened in the last 300 s is still open; older open calls (orphaned by a
        broker restart) settle uncertain."""

        return self._rpc(
            "lead_order_finish_v1", p_order_id=order_id, p_status=status, p_delivered=int(delivered),
            p_delivery_url=delivery_url, p_failure_reason=failure_reason, p_epoch=int(epoch),
            p_allow_open=bool(allow_open),
        )

    def overdue(self, *, grace_seconds: int = 0) -> List[str]:
        return list(self._rpc("lead_order_overdue_v1", p_grace_seconds=int(grace_seconds)) or [])

    def view(self, order_id: str) -> Optional[Dict[str, Any]]:
        return self._rpc("lead_order_view_v1", p_order_id=order_id)

    # Calls ------------------------------------------------------------------
    def reserve_call(self, *, call_id: str, order_id: str, provider: str, key_id: str,
                     miner_hotkey: Optional[str], public_fingerprint: str, reserve_microusd: int) -> Dict[str, Any]:
        return self._rpc(
            "lead_order_reserve_call_v1", p_call_id=call_id, p_order_id=order_id, p_provider=provider,
            p_key_id=key_id, p_miner_hotkey=miner_hotkey, p_public_fingerprint=public_fingerprint,
            p_reserve=int(reserve_microusd),
        )

    def settle_call(self, *, call_id: str, cost_microusd: Optional[int], http_status: Optional[int],
                    epoch: int, provider_request_id: Optional[str] = None) -> Dict[str, Any]:
        return self._rpc(
            "lead_order_settle_call_v1", p_call_id=call_id,
            p_status="settled" if cost_microusd is not None else "uncertain",
            p_cost=None if cost_microusd is None else int(cost_microusd),
            p_http_status=http_status, p_epoch=int(epoch), p_request_id=provider_request_id,
        )

    def uncertain_calls(self, *, limit: int) -> List[Dict[str, Any]]:
        return list(self._rpc("lead_order_uncertain_calls_v1", p_limit=int(limit)) or [])

    def record_readback(self, *, call_id: str, outcome: str) -> Dict[str, Any]:
        """One read-back attempt (found, retry or terminal), append-only."""
        return self._rpc("lead_order_record_readback_v1", p_call_id=call_id, p_outcome=outcome)

    def resolve_call(self, *, call_id: str, cost_microusd: int, epoch: int) -> Dict[str, Any]:
        return self._rpc("lead_order_resolve_call_v1", p_call_id=call_id, p_cost=int(cost_microusd),
                         p_epoch=int(epoch))

    def reward_late(self, *, epoch: int, limit: int) -> int:
        return int((self._rpc("lead_order_reward_late_v1", p_epoch=int(epoch), p_limit=int(limit)) or {})
                   .get("recorded", 0))

    def statement(self, hotkey: str, *, since: int, limit: int) -> Dict[str, Any]:
        return self._rpc("lead_order_statement_v1", p_hotkey=hotkey, p_since=int(since), p_limit=int(limit))

    def record_vault_receipt(self, *, call_id: str, kind: str, receipt: Mapping[str, Any]) -> Dict[str, Any]:
        """Keep one signed vault receipt for a call or its cost read-back."""
        return self._rpc("lead_order_record_vault_receipt_v1", p_call_id=call_id, p_kind=kind,
                         p_receipt=dict(receipt))

    def vault_receipts(self, hotkey: str, *, since: int, limit: int) -> Dict[str, Any]:
        return self._rpc("lead_order_vault_receipts_v1", p_hotkey=hotkey, p_since=int(since), p_limit=int(limit))

    def exhaust_key(self, *, order_id: str, key_id: str, reason: str) -> None:
        self._rpc("lead_order_exhaust_key_v1", p_order_id=order_id, p_key_id=key_id, p_reason=reason)

    def exhausted_keys(self, order_id: str) -> List[str]:
        return list(self._rpc("lead_order_exhausted_keys_v1", p_order_id=order_id) or [])

    def calls(self, order_id: str) -> List[Dict[str, Any]]:
        return list(self._rpc("lead_order_calls_v1", p_order_id=order_id) or [])

    # Keys -------------------------------------------------------------------
    def register_key(self, row: Mapping[str, Any], *, public_fingerprint: str,
                     daily_cap_microusd: int) -> Dict[str, Any]:
        return self._rpc(
            "lead_order_register_key_v1", p_key_id=row["key_id"], p_provider=row["provider"],
            p_miner_hotkey=row["miner_hotkey"], p_submission_id=row["submission_id"],
            p_ciphertext=row["ciphertext_b64"], p_internal_fp=row["internal_fingerprint"],
            p_public_fp=public_fingerprint, p_daily_cap=int(daily_cap_microusd),
        )

    def revoke_key(self, key_id: str, *, reason: str) -> None:
        self._rpc("lead_order_revoke_key_v1", p_key_id=key_id, p_reason=reason)

    def keys(self, provider: str, *, terms_sha256: Optional[str] = None) -> List[Dict[str, Any]]:
        """Active keys; with ``terms_sha256``, only keys whose miner accepted that text."""

        return list(self._rpc("lead_order_keys_v1", p_provider=provider, p_terms_sha256=terms_sha256) or [])

    # Miner terms and suppressions -------------------------------------------
    def accept_miner_terms(self, *, hotkey: str, version: str, terms_sha256: str, host: str, signed_at: int,
                           body_sha256: str, signature: str) -> Dict[str, Any]:
        return self._rpc("lead_order_accept_miner_terms_v1", p_hotkey=hotkey, p_version=version,
                         p_terms_sha256=terms_sha256, p_host=host,
                         p_signed_at=int(signed_at), p_body_sha256=body_sha256, p_signature=signature)

    def has_miner_terms(self, hotkey: str, terms_sha256: str) -> bool:
        return bool(self._rpc("lead_order_miner_terms_v1", p_hotkey=hotkey, p_terms_sha256=terms_sha256))

    def add_suppressions(self, rows: List[Mapping[str, str]]) -> Dict[str, Any]:
        return self._rpc("lead_order_add_suppressions_v1", p_rows=list(rows))

    def suppressions(self, *, since: int, limit: int) -> List[Dict[str, Any]]:
        return list(self._rpc("lead_order_suppressions_v1", p_since=int(since), p_limit=int(limit)) or [])

    def key_spend(self, *, since_epoch: int) -> Dict[str, int]:
        return {k: int(v) for k, v in (self._rpc("lead_order_key_spend_v1", p_since_epoch=int(since_epoch)) or {}).items()}

    # Rewards and receipts ---------------------------------------------------
    def record_rewards(self, order_id: str, rewards: Mapping[str, int]) -> Dict[str, Any]:
        return self._rpc("lead_order_record_rewards_v1", p_order_id=order_id,
                         p_rewards={k: int(v) for k, v in rewards.items()})

    def record_receipt(self, order_id: str, receipt: Mapping[str, Any],
                       call_leaves: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
        return self._rpc("lead_order_record_receipt_v1", p_order_id=order_id,
                         p_hash=receipt["receipt_hash"], p_doc=dict(receipt),
                         p_calls=[dict(leaf) for leaf in call_leaves])

    def policy_document(self, version: int) -> Optional[Dict[str, Any]]:
        return self._rpc("lead_order_policy_v1", p_version=int(version))

    def unsettled(self, *, limit: int) -> List[str]:
        return [str(order_id) for order_id in (self._rpc("lead_order_unsettled_v1", p_limit=int(limit)) or [])]

    def owed(self, *, through_epoch: int, mode: str) -> List[Dict[str, Any]]:
        return list(self._rpc("lead_order_owed_v1", p_through_epoch=int(through_epoch), p_mode=mode) or [])

    def record_payout(self, *, epoch: int, mode: str, document: Mapping[str, Any],
                      payments: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
        return self._rpc(
            "lead_order_record_payout_v1", p_epoch=int(epoch), p_mode=mode, p_doc_hash=document["payout_hash"],
            p_doc=dict(document), p_payments=[dict(p) for p in payments],
        )

    def reward_epoch(self, epoch: int) -> Optional[Dict[str, Any]]:
        return self._rpc("lead_order_reward_epoch_v1", p_epoch=int(epoch))


__all__ = ["LEAD_ORDER_FUNCTION_SIGNATURES", "LeadOrderStore", "register_rpc_signatures"]
