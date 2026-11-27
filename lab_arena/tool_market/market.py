"""The tool market: miners list sealed vendor keys at a price, buyers call tools.

A miner seals a vendor key to the key vault on their own machine, registers
the envelope here, and posts an ask: what they earn per unit of usage. A
buyer opens a tab against their Gamma balance and calls catalog operations
with the tab's token. Each call goes to the cheapest eligible key, through
the vault, which puts the key where the vendor expects it. The buyer pays
the ask plus 10% (``book.charge_microusd``); the miner is owed the ask.

What the gateway never sees: the vendor key. What it trusts: the vendor's
own answer for usage (a 2xx for per-call tools, the reported cost for
``vendor_usd`` tools). When the usage is unknown the buyer pays nothing and
the miner is owed nothing (``uncertain``).
"""

from __future__ import annotations

import hashlib
import secrets
import uuid
from typing import Any, Callable, Dict, Mapping, Optional

from lab_arena.lead_orders.auth import BuyerIdentity
from lab_arena.lead_orders.gamma import GammaReadError
from lab_arena.lead_orders.keys import json_document
from lab_arena.lead_orders.vault_client import VaultTransportError, VaultUnavailable
from lab_arena.tool_market import book, catalog
from lab_arena.tool_market.config import (
    CHECK_MAX_AGE_SECONDS,
    MAX_KEY_DAILY_CAP_MICROUSD,
    MAX_TAB_MICROUSD,
    MAX_TAB_SECONDS,
    ToolMarketConfig,
)
from lab_arena.tool_market.store import ToolMarketStore
from lead_order_vault import envelope as vault_envelope

TOKEN_PREFIX = "tmt_"
# Keys tried for one call before the buyer is told there is no supply.
MAX_KEYS_PER_CALL = 3
CALL_TIMEOUT_SECONDS = 30.0
MAX_RESPONSE_BYTES = 4 * 1024 * 1024
MIN_TAB_SECONDS = 60


class ToolMarketError(Exception):
    """A refusal with an HTTP status and a code safe to return."""

    def __init__(self, status: int, code: str, **detail: Any) -> None:
        super().__init__(code)
        self.status = status
        self.code = code
        self.detail = detail


def token_hash(token: str) -> str:
    return "sha256:" + hashlib.sha256(token.encode("utf-8")).hexdigest()


def _int(value: Any, low: int, high: int, code: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
        raise ToolMarketError(400, code)
    return value


class ToolMarket:
    def __init__(self, *, config: ToolMarketConfig, store: ToolMarketStore, vault: Any, transport: Any,
                 gamma_reader: Any, current_epoch: Callable[[], int],
                 new_id: Callable[[], str] = lambda: str(uuid.uuid4()),
                 new_token: Callable[[], str] = lambda: TOKEN_PREFIX + secrets.token_urlsafe(32)) -> None:
        self._config = config
        self._store = store
        self._vault = vault
        self._transport = transport
        self._gamma = gamma_reader
        self._epoch = current_epoch
        self._new_id = new_id
        self._new_token = new_token

    # Terms and balance ------------------------------------------------------
    def accept_terms(self, identity: BuyerIdentity, body: Mapping[str, Any]) -> Dict[str, Any]:
        role = body.get("role")
        if role not in ("buyer", "miner") or set(body) != {"role", "terms_version", "terms_sha256"}:
            raise ToolMarketError(400, "request_invalid")
        if body["terms_version"] != self._config.terms_version or body["terms_sha256"] != self._config.terms_sha256:
            raise ToolMarketError(409, "terms_outdated")
        return self._store.accept_terms(
            role=role, hotkey=identity.hotkey, version=self._config.terms_version,
            terms_sha256=self._config.terms_sha256, host=identity.host, signed_at=identity.signed_at,
            body_sha256=identity.body_sha256, signature=identity.signature.lower().removeprefix("0x"))

    def use_nonce(self, hotkey: str, signature_hash: str, signed_at: int) -> bool:
        return self._store.use_nonce(hotkey=hotkey, signature_hash=signature_hash, signed_at=signed_at)

    def balance(self, hotkey: str) -> Dict[str, Any]:
        """Read the buyer's Gamma spend to this service and return their balance."""

        try:
            reading = self._gamma.read(hotkey)
        except GammaReadError:
            raise ToolMarketError(503, "gamma_unavailable") from None
        result = self._store.record_gamma_reading(
            hotkey=hotkey, block_number=reading.block_number, block_hash=reading.block_hash,
            spent_units=reading.spent_units, available=reading.available)
        if result.get("status") == "counter_backwards":
            raise ToolMarketError(503, "gamma_unavailable")
        return result

    # Tabs -------------------------------------------------------------------
    def open_tab(self, identity: BuyerIdentity, body: Mapping[str, Any]) -> Dict[str, Any]:
        if not self._config.may_buy(identity.hotkey):
            raise ToolMarketError(403, "buyer_not_allowed")
        if set(body) != {"budget_microusd", "seconds"}:
            raise ToolMarketError(400, "request_invalid")
        budget = _int(body["budget_microusd"], 1, MAX_TAB_MICROUSD, "budget_invalid")
        seconds = _int(body["seconds"], MIN_TAB_SECONDS, MAX_TAB_SECONDS, "seconds_invalid")
        self.balance(identity.hotkey)
        token = self._new_token()
        result = self._store.open_tab(tab_id=self._new_id(), hotkey=identity.hotkey, token_hash=token_hash(token),
                                      budget_microusd=budget, seconds=seconds,
                                      terms_sha256=self._config.terms_sha256)
        if result["status"] == "terms_required":
            raise ToolMarketError(403, "terms_required")
        if result["status"] == "balance_low":
            raise ToolMarketError(402, "balance_low", balance=result.get("balance"))
        return dict(result, token=token)

    def tab(self, token: str) -> Dict[str, Any]:
        found = self._store.tab(token_hash(token)) if token else None
        if not found:
            raise ToolMarketError(401, "tab_unknown")
        return found

    def close_tab(self, token: str) -> Dict[str, Any]:
        closed = self._store.close_tab(token_hash(token)) if token else None
        if not closed:
            raise ToolMarketError(401, "tab_unknown")
        return closed

    # Miners -----------------------------------------------------------------
    def register_key(self, identity: BuyerIdentity, body: Mapping[str, Any]) -> Dict[str, Any]:
        if set(body) != {"tool", "envelope", "daily_cap_microusd"}:
            raise ToolMarketError(400, "request_invalid")
        tool = body["tool"]
        if tool not in self._config.tools:
            raise ToolMarketError(404, "tool_unavailable")
        cap = _int(body["daily_cap_microusd"], 1, MAX_KEY_DAILY_CAP_MICROUSD, "daily_cap_invalid")
        envelope = body["envelope"]
        try:
            header, _ = vault_envelope.read_header(envelope)
        except vault_envelope.EnvelopeError as error:
            raise ToolMarketError(400, str(error)) from None
        # Checked here first so an envelope that is not this miner's never
        # makes the vault probe a vendor.
        if header.provider != tool or header.purpose != "runtime":
            raise ToolMarketError(400, "envelope_mismatch")
        if header.miner_hotkey != identity.hotkey:
            raise ToolMarketError(400, "envelope_hotkey_mismatch")
        checked = self._check(tool, envelope)
        if checked.get("status") not in ("valid", "unchecked"):
            raise ToolMarketError(422, str(checked.get("reason") or "key_invalid"))
        if checked.get("miner_hotkey") != identity.hotkey or checked.get("provider") != tool:
            raise ToolMarketError(502, "vault_answer_invalid")
        result = self._store.register_key(
            key_id=self._new_id(), tool=tool, hotkey=identity.hotkey, envelope=envelope,
            fingerprint=checked["key_fingerprint"], daily_cap_microusd=cap, status=checked["status"],
            balance=checked.get("balance"))
        if result["status"] == "duplicate":
            raise ToolMarketError(409, "key_already_listed")
        return dict(result, health=checked["status"])

    def check_key(self, identity: BuyerIdentity, key_id: Any) -> Dict[str, Any]:
        """Check a key again, e.g. after topping up a key that ran out."""

        key = self._own_key(identity, key_id)
        status = self._recheck(key["key_id"], key["tool"], key["envelope"])
        return {"key_id": key["key_id"], "tradeable": status}

    def revoke_key(self, identity: BuyerIdentity, key_id: Any) -> Dict[str, Any]:
        key = self._own_key(identity, key_id)
        return self._store.revoke_key(key_id=key["key_id"], hotkey=identity.hotkey, reason="miner_revoked")

    def post_ask(self, identity: BuyerIdentity, body: Mapping[str, Any]) -> Dict[str, Any]:
        """Set a key's price per unit, or withdraw it with ``ask_microusd: null``."""

        if set(body) != {"key_id", "ask_microusd"}:
            raise ToolMarketError(400, "request_invalid")
        ask = body["ask_microusd"]
        if ask is not None and not book.valid_ask(ask):
            raise ToolMarketError(400, "ask_invalid")
        key = self._own_key(identity, body["key_id"])
        result = self._store.post_ask(key_id=key["key_id"], hotkey=identity.hotkey, ask_microusd=ask)
        if result["status"] == "not_found":
            raise ToolMarketError(404, "key_unknown")
        return result

    def statement(self, hotkey: str) -> Dict[str, Any]:
        return self._store.statement(hotkey)

    # Public book ------------------------------------------------------------
    def order_book(self, tool: Any) -> Dict[str, Any]:
        """Asks for one tool, cheapest first, with what a buyer pays per unit."""

        if tool not in self._config.tools:
            raise ToolMarketError(404, "tool_unavailable")
        asks = book.rank(book.Ask.from_row(row) for row in self._supply(tool))
        return {"tool": tool, "unit": catalog.TOOLS[tool],
                "operations": sorted(name for (t, name) in catalog.OPERATIONS if t == tool),
                "asks": [{"ask_microusd": a.ask_microusd, "buyer_pays_microusd": book.charge_microusd(a.ask_microusd)}
                         for a in asks]}

    # Calls ------------------------------------------------------------------
    def call(self, token: str, body: Mapping[str, Any]) -> Dict[str, Any]:
        if not isinstance(body, Mapping) or set(body) - {"params"} != {"tool", "operation"}:
            raise ToolMarketError(400, "request_invalid")
        tool = body["tool"]
        if tool not in self._config.tools:
            raise ToolMarketError(404, "tool_unavailable")
        try:
            op = catalog.operation(tool, body["operation"])
            url, headers, payload = op.build(body.get("params", {}))
        except catalog.CatalogError as error:
            raise ToolMarketError(400, error.code) from None
        if not token:
            raise ToolMarketError(401, "tab_unknown")
        tab = token_hash(token)
        tried = 0
        for ask in book.rank(book.Ask.from_row(row) for row in self._supply(tool)):
            if tried >= MAX_KEYS_PER_CALL:
                break
            if ask.needs_check and not self._recheck(ask.key_id, tool, ask.envelope):
                continue
            call_id = self._new_id()
            status = self._store.reserve_call(
                call_id=call_id, token_hash=tab, tool=tool, operation=op.name, key_id=ask.key_id,
                ask_microusd=ask.ask_microusd, reserve_microusd=book.reserve_microusd(op, ask.ask_microusd),
                owed_cap_microusd=book.owed_microusd(op, op.max_units, ask.ask_microusd))
            if status == "tab_closed":
                raise ToolMarketError(401, "tab_closed")
            if status == "tab_budget":
                raise ToolMarketError(402, "tab_budget")
            if status != "reserved":
                continue
            tried += 1
            outcome = self._send(op, ask, call_id, url, headers, payload)
            if outcome is not None:
                return outcome
        raise ToolMarketError(503, "no_supply")

    def _send(self, op: catalog.Operation, ask: book.Ask, call_id: str, url: str, headers: Mapping[str, str],
              payload: bytes) -> Optional[Dict[str, Any]]:
        """One vendor request on one key. ``None`` means try the next key."""

        try:
            response = self._transport.send(
                method=op.method, url=url, headers=dict(headers, authorization="Bearer " + ask.envelope),
                body=payload, timeout_seconds=CALL_TIMEOUT_SECONDS, max_response_bytes=MAX_RESPONSE_BYTES)
        except VaultTransportError as error:
            if error.forwarded is False:
                self._settle(call_id, "skipped", None)
                return None
            # It may have reached the vendor and used the key: the miner
            # bears it, and a retry could use it twice.
            self._settle(call_id, "uncertain", error.observed_status)
            raise ToolMarketError(502, "vendor_unreachable", call_id=call_id) from None
        failure = book.key_failure(response.status)
        if failure is not None or book.rate_limited(response.status, response.headers):
            self._settle(call_id, "skipped", response.status)
            if failure is not None:
                self._store.record_key_event(key_id=ask.key_id, status="failed", reason=failure)
            return None
        document = json_document(response.body)
        units = op.units(response.status, document)
        if units is None:
            self._settle(call_id, "uncertain", response.status)
            charge = 0
        else:
            owed = book.owed_microusd(op, units, ask.ask_microusd)
            charge = book.charge_microusd(owed)
            self._settle(call_id, "settled", response.status, units=units, owed=owed, charge=charge)
        return {"call_id": call_id, "status": response.status, "body": document, "charge_microusd": charge}

    def _settle(self, call_id: str, status: str, http_status: Optional[int], *, units: int = 0, owed: int = 0,
                charge: int = 0) -> None:
        self._store.settle_call(call_id=call_id, status=status, http_status=http_status, units=units,
                                owed_microusd=owed, charge_microusd=charge, epoch=int(self._epoch()))

    # Helpers ----------------------------------------------------------------
    def _supply(self, tool: str):
        return self._store.book(tool=tool, terms_sha256=self._config.terms_sha256,
                                check_max_age_seconds=CHECK_MAX_AGE_SECONDS)

    def _check(self, tool: str, envelope: str) -> Mapping[str, Any]:
        try:
            return self._vault.check(provider=tool, envelope=envelope, management_envelope=None)
        except VaultUnavailable as error:
            raise ToolMarketError(503 if error.retryable else 400, error.code) from None

    def _recheck(self, key_id: str, tool: str, envelope: str) -> bool:
        """Check a key with its vendor again and record it; True if it may trade."""

        try:
            checked = self._vault.check(provider=tool, envelope=envelope, management_envelope=None)
        except VaultUnavailable:
            return False
        status = checked.get("status")
        if status not in ("valid", "unchecked", "invalid"):
            return False
        self._store.record_key_event(key_id=key_id, status=status, reason=checked.get("reason"),
                                     balance=checked.get("balance"))
        return status != "invalid"

    def _own_key(self, identity: BuyerIdentity, key_id: Any) -> Dict[str, Any]:
        try:
            key_id = str(uuid.UUID(str(key_id)))
        except ValueError:
            raise ToolMarketError(404, "key_unknown") from None
        key = self._store.key(key_id)
        if not key or key["miner_hotkey"] != identity.hotkey or not key["active"]:
            raise ToolMarketError(404, "key_unknown")
        return key


__all__ = ["ToolMarket", "ToolMarketError", "token_hash"]
