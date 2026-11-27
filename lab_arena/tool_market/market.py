"""The tool market: miners list sealed vendor keys at a price, buyers call tools.

A miner seals a vendor key to the key vault on their own machine, registers
the envelope here, and posts an ask: what they earn per unit of usage. A
buyer calls catalog operations with requests their hotkey signs (btauth/1),
charged to their Gamma balance, or opens a tab and calls with its token. Each call goes to the cheapest eligible key, through
the vault, which puts the key where the vendor expects it. The buyer pays
the ask plus 10% (``book.charge_microusd``); the miner is owed the ask.

What the gateway never sees: the vendor key. What it trusts: the vendor's
own answer for usage (what a 2xx shows was charged, or the reported cost
for ``vendor_usd`` tools). When the usage is unknown the buyer pays nothing and
the miner is owed nothing (``uncertain``).
"""

from __future__ import annotations

import hashlib
import json
import secrets
import threading
import time
import uuid
from contextlib import contextmanager
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

from lab_arena.lead_orders.auth import BuyerIdentity
from lab_arena.lead_orders.config import is_valid_hotkey
from lab_arena.lead_orders.keys import json_document
from lab_arena.lead_orders.vault_client import VaultTransportError, VaultUnavailable
from lab_arena.tool_market import book, catalog, discovery
from lab_arena.chain import ArenaChainError
from lab_arena.store import ArenaStoreError
from lab_arena.tool_market.config import ToolMarketConfig
from lab_arena.tool_market.store import ToolMarketStore
from lead_order_vault import envelope as vault_envelope
from lead_order_vault import tools as vault_tools

TOKEN_PREFIX = "tmt_"
# Keys tried for one call before the buyer is told there is no supply.
MAX_KEYS_PER_CALL = 3
# Reserve attempts per call, refused ones included (an ask that just moved,
# a key that just reached its cap), so one call cannot walk a whole book.
MAX_ASKS_PER_CALL = 10
# Calls one buyer has in flight in this process, so a few buyers cannot hold
# every call thread with slow free answers (a no-match search, a timeout).
MAX_CALLS_IN_FLIGHT_PER_BUYER = 8
RECHECK_SPACING_SECONDS = 60
# Gamma balance reads in flight in this process. Each is several chain
# requests under the chain client's one lock, which every paid call's epoch
# read also takes; anyone can ask for one (a fresh hotkey, the public
# balance), so past this many the answer is 503 at once.
CHAIN_READS = 2
# An envelope (4,096 characters), the attestation and the other fields.
MAX_REGISTER_BODY_BYTES = 8192
MAX_USAGE_PAGE = 500
CALL_TIMEOUT_SECONDS = 30.0
MAX_RESPONSE_BYTES = 4 * 1024 * 1024
MIN_TAB_SECONDS = 60
# A signed call's own tab: closed when the call ends, else it lapses after one
# call window (``tool_market__open_seconds_v1``).
CALL_TAB_SECONDS = 600
MAX_TAB_SECONDS = 30 * 24 * 3600  # a tab is the buyer's API key
MAX_TAB_MICROUSD = 1_000_000_000  # $1,000
# Low at launch: what one key may earn in 24 h. Raising it is a code change.
MAX_KEY_DAILY_CAP_MICROUSD = 20_000_000  # $20
# Deeper vendor answers are unknown usage: the gateway could not send them back.
MAX_ANSWER_DEPTH = 64
# The open subnet's name, as miners and buyers sign it.
NETWORK_NAME = "Subnet 71"
# What a miner signs, inside the register body, for every key they list.
ATTESTATION = ("tool-market-attestation-v3: I hold the account this key belongs to, and its terms allow me to "
               "supply capacity through Bittensor " + NETWORK_NAME + ". I agree that " + NETWORK_NAME + " may use "
               "this key to serve other buyers' requests. I alone am responsible for my account's compliance and "
               "hold the relationship with the provider, and I will revoke this key if that stops being true. "
               + NETWORK_NAME + ", its validators and its contributors make no claim about any provider's terms "
               "and are not liable for any action a provider takes on my account.")


class ToolMarketError(Exception):
    """A refusal with an HTTP status and a code safe to return."""

    def __init__(self, status: int, code: str, **detail: Any) -> None:
        super().__init__(code)
        self.status = status
        self.code = code
        self.detail = detail


def token_hash(token: str) -> str:
    return "sha256:" + hashlib.sha256(token.encode("utf-8")).hexdigest()


def _document(body: bytes) -> Any:
    """A vendor's JSON answer, or None when it is not JSON the gateway can
    send back as is (NaN, an infinite number, a lone surrogate, or nested
    deeper than MAX_ANSWER_DEPTH): then the cost readers see what the buyer
    gets."""

    document = json_document(body)
    stack = [(document, 1)]
    while stack:
        value, depth = stack.pop()
        if isinstance(value, (list, dict)):
            if depth > MAX_ANSWER_DEPTH:
                return None
            stack.extend((item, depth + 1) for item in (value.values() if isinstance(value, dict) else value)
                         if isinstance(item, (list, dict)))
    try:
        json.dumps(document, allow_nan=False, ensure_ascii=False).encode("utf-8")
    except (ValueError, UnicodeEncodeError):
        return None
    return document


def _usd(microusd: int) -> str:
    return "%d.%06d" % divmod(microusd, 1_000_000)


def _int(value: Any, low: int, high: int, code: str, **detail: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
        raise ToolMarketError(400, code, **detail)
    return value


class ToolMarket:
    def __init__(self, *, config: ToolMarketConfig, store: ToolMarketStore, vault: Any, transport: Any,
                 gamma_reader: Any, current_epoch: Callable[[], int], is_registered: Callable[[str], bool],
                 new_id: Callable[[], str] = lambda: str(uuid.uuid4()),
                 new_token: Callable[[], str] = lambda: TOKEN_PREFIX + secrets.token_urlsafe(32)) -> None:
        self._config = config
        self._store = store
        self._vault = vault
        self._transport = transport
        self._gamma = gamma_reader
        self._epoch = current_epoch
        self._is_registered = is_registered
        self._new_id = new_id
        self._new_token = new_token
        self._rechecked: Dict[str, float] = {}
        self._in_flight: Dict[str, int] = {}
        self._lock = threading.Lock()
        self._chain_reads = threading.BoundedSemaphore(CHAIN_READS)

    # Terms and balance ------------------------------------------------------
    def accept_terms(self, identity: BuyerIdentity, body: Mapping[str, Any]) -> Dict[str, Any]:
        role = body.get("role")
        if role not in ("buyer", "miner") or set(body) != {"role", "terms_version", "terms_sha256"}:
            raise ToolMarketError(400, "request_invalid")
        if body["terms_version"] != self._config.terms_version or body["terms_sha256"] != self._config.terms_sha256:
            raise ToolMarketError(409, "terms_outdated", terms_version=self._config.terms_version)
        return self._store.accept_terms(
            role=role, hotkey=identity.hotkey, version=self._config.terms_version,
            terms_sha256=self._config.terms_sha256, host=identity.host, signed_at=identity.signed_at,
            body_sha256=identity.body_sha256, signature=identity.signature.lower().removeprefix("0x"))

    def use_nonce(self, hotkey: str, signature_hash: str, signed_at: int) -> bool:
        return self._store.use_nonce(hotkey=hotkey, signature_hash=signature_hash, signed_at=signed_at)

    def balance(self, hotkey: str) -> Dict[str, Any]:
        """Read the buyer's Gamma spend to this service and return their balance."""

        if not self._chain_reads.acquire(blocking=False):
            raise ToolMarketError(503, "gamma_unavailable")
        try:
            reading = self._gamma.read(hotkey)
        except ArenaChainError:  # a Gamma read or the finalized head
            raise ToolMarketError(503, "gamma_unavailable") from None
        finally:
            self._chain_reads.release()
        result = self._store.record_gamma_reading(
            hotkey=hotkey, block_number=reading.block_number, block_hash=reading.block_hash,
            spent_units=reading.spent_units, available=reading.available)
        if result.get("status") in ("counter_backwards", "reading_conflict"):
            raise ToolMarketError(503, "gamma_unavailable")
        return result

    def gamma_balance(self, hotkey: Any) -> Dict[str, Any]:
        """The Gamma standard's balance: anyone may read any hotkey's, in USD
        as decimal strings; a hotkey that never spent here reads zeros."""

        if not isinstance(hotkey, str) or not is_valid_hotkey(hotkey):
            raise ToolMarketError(400, "invalid_hotkey")
        found = self.balance(hotkey)
        credited = max(int(found.get("credit_microusd") or 0), 0)
        remaining = min(max(int(found.get("balance_microusd") or 0), 0), credited)
        return {"hotkey": hotkey, "credited_usd": _usd(credited), "used_usd": _usd(credited - remaining),
                "remaining_usd": _usd(remaining)}

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
        try:
            result = self._store.open_tab(tab_id=self._new_id(), hotkey=identity.hotkey, token_hash=token_hash(token),
                                          budget_microusd=budget, seconds=seconds,
                                          terms_sha256=self._config.terms_sha256)
        except ArenaStoreError:
            # Its answer may be lost after it committed: an open tab only its
            # token can close would hold the budget until it expires.
            found = self._store.tab(token_hash(token))
            if not found:
                raise
            result = {"status": "opened", "tab_id": found["tab_id"], "budget_microusd": found["budget_microusd"],
                      "expires_at": found["expires_at"]}
        if result["status"] == "terms_required":
            raise ToolMarketError(403, "terms_required")
        if result["status"] == "insufficient_balance":
            raise ToolMarketError(402, "insufficient_balance", balance=result.get("balance"))
        return dict(result, token=token)

    def tab(self, token: str) -> Dict[str, Any]:
        found = self._store.tab(token_hash(token)) if token else None
        if not found:
            raise ToolMarketError(401, "tab_unknown")
        return found

    def usage(self, identity: BuyerIdentity, after: Any, limit: Any) -> Dict[str, Any]:
        """The buyer's own finished calls after ``after``; pass back ``watermark`` for the next page."""

        return self._store.usage(hotkey=identity.hotkey, after=_int(after, 0, 2**62, "after_invalid"),
                                 limit=_int(limit, 1, MAX_USAGE_PAGE, "limit_invalid"))

    def close_tab(self, token: str) -> Dict[str, Any]:
        closed = self._store.close_tab(token_hash(token)) if token else None
        if not closed:
            raise ToolMarketError(401, "tab_unknown")
        return closed

    # Miners -----------------------------------------------------------------
    def register_key(self, identity: BuyerIdentity, body: Mapping[str, Any], signed_body: bytes) -> Dict[str, Any]:
        """List a key. ``signed_body`` is the request body as signed; it is kept
        with the key, so the miner's attestation can be verified again later.

        Only a hotkey registered on the subnet lists keys: a registration
        costs TAO, so a key the vault cannot check costs more than a
        signature, and only registered hotkeys are paid anyway.
        """

        if set(body) != {"tool", "envelope", "daily_cap_microusd", "attestation"}:
            raise ToolMarketError(400, "request_invalid")
        if len(signed_body) > MAX_REGISTER_BODY_BYTES:  # it is kept for good
            raise ToolMarketError(413, "request_too_large")
        if body["attestation"] != ATTESTATION:
            raise ToolMarketError(400, "attestation_required", attestation=ATTESTATION)
        tool = body["tool"]
        if not isinstance(tool, str) or tool not in self._config.tools:
            raise ToolMarketError(404, "tool_unavailable")
        self._require_supplier(tool, identity.hotkey)
        cap = _int(body["daily_cap_microusd"], 1, MAX_KEY_DAILY_CAP_MICROUSD, "daily_cap_invalid")
        try:
            registered = self._is_registered(identity.hotkey)
        except ArenaChainError:
            raise ToolMarketError(503, "chain_unavailable") from None
        if not registered:
            raise ToolMarketError(403, "miner_not_registered")
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
        # One vendor probe per hotkey and tool a minute, unless it ends in a
        # listing, so re-posting a refused key cannot loop vendor checks.
        probe = "register:%s:%s" % (identity.hotkey, tool)
        with self._slot("check:" + identity.hotkey, 1, "check_in_flight"):
            if not self._recheck_due(probe):
                raise ToolMarketError(429, "check_too_soon", retry_after_seconds=RECHECK_SPACING_SECONDS)
            checked = self._check(tool, envelope)
        if checked.get("status") not in ("valid", "unchecked"):
            raise ToolMarketError(422, str(checked.get("reason") or "key_invalid"))
        if checked.get("miner_hotkey") != identity.hotkey or checked.get("provider") != tool:
            raise ToolMarketError(502, "vault_answer_invalid")
        result = self._store.register_key(
            key_id=self._new_id(), tool=tool, hotkey=identity.hotkey, envelope=envelope,
            fingerprint=checked["key_fingerprint"], daily_cap_microusd=cap, status=checked["status"],
            host=identity.host, signed_at=identity.signed_at,
            signed_body=signed_body.decode("utf-8"), signature=identity.signature.lower().removeprefix("0x"))
        if result["status"] == "duplicate":
            raise ToolMarketError(409, "key_already_listed")
        if result["status"] == "too_soon":
            raise ToolMarketError(429, "key_registration_too_soon", next_at=result["next_at"])
        with self._lock:
            self._rechecked.pop(probe, None)
        return dict(result, health=checked["status"])

    def check_key(self, identity: BuyerIdentity, key_id: Any) -> Dict[str, Any]:
        """Check a key again, e.g. after topping up a key that ran out.

        A tool with no free check is refused: its vault answer (``unchecked``)
        proves nothing, so it cannot put back a key a call found dead.
        """

        key = self._own_key(identity, key_id)
        self._require_supplier(key["tool"], identity.hotkey)
        if vault_tools.TOOLS[key["tool"]].balance_url is None:
            raise ToolMarketError(409, "key_cannot_be_checked")
        with self._slot("check:" + identity.hotkey, 1, "check_in_flight"):
            if not self._recheck_due(key["key_id"]):
                raise ToolMarketError(429, "check_too_soon", retry_after_seconds=RECHECK_SPACING_SECONDS)
            checked = self._recheck(key["key_id"], key["tool"], key["envelope"])
        return {"key_id": key["key_id"], "tradeable": checked is not None and checked["status"] != "invalid"}

    def revoke_key(self, identity: BuyerIdentity, key_id: Any) -> Dict[str, Any]:
        key = self._own_key(identity, key_id)
        return self._store.revoke_key(key_id=key["key_id"], hotkey=identity.hotkey, reason="miner_revoked")

    def post_ask(self, identity: BuyerIdentity, body: Mapping[str, Any]) -> Dict[str, Any]:
        """Set a key's price per unit, or withdraw it with ``ask_microusd: null``."""

        if set(body) != {"key_id", "ask_microusd"}:
            raise ToolMarketError(400, "request_invalid")
        ask = body["ask_microusd"]
        key = self._own_key(identity, body["key_id"])
        ceiling = catalog.MAX_ASK[catalog.TOOLS[key["tool"]]]
        if ask is not None:
            self._require_supplier(key["tool"], identity.hotkey)
            _int(ask, 1, ceiling, "ask_invalid", max_ask_microusd=ceiling)
            # Each operation's largest call owes something, so every call
            # reserves, and a key's cap holds 4 of them, so each buyer's
            # quarter of it (unknown usage and calls in flight) fits one.
            caps = [book.owed_microusd(op, op.max_units, ask)
                    for op in catalog.OPERATIONS.values() if op.tool == key["tool"]]
            if min(caps) < 1:
                raise ToolMarketError(400, "ask_too_low")
            if 4 * max(caps) > key["daily_cap_microusd"]:
                raise ToolMarketError(400, "ask_too_high_for_cap")
        result = self._store.post_ask(key_id=key["key_id"], hotkey=identity.hotkey, ask_microusd=ask)
        if result["status"] == "not_found":
            raise ToolMarketError(404, "key_unknown")
        return result

    def statement(self, hotkey: str) -> Dict[str, Any]:
        return self._store.statement(hotkey)

    # Public book ------------------------------------------------------------
    def order_book(self, tool: Any) -> Dict[str, Any]:
        """Asks for one tool, cheapest first, with what a buyer pays per unit.
        A key with no room left today for any call is not listed."""

        if tool not in self._config.tools:
            raise ToolMarketError(404, "tool_unavailable")
        ops = [op for op in catalog.OPERATIONS.values() if op.tool == tool]
        asks = [a for a in self._supply(tool)
                if min(book.owed_microusd(op, op.max_units, a.ask_microusd) for op in ops) <= a.remaining_microusd]
        return {"tool": tool, "unit": catalog.TOOLS[tool],
                "operations": sorted(op.name for op in ops),
                "request_privacy": catalog.REQUEST_PRIVACY[tool],
                "asks": [{"ask_microusd": a.ask_microusd, "buyer_pays_microusd": book.charge_microusd(a.ask_microusd),
                          "private": self._private(tool, a)} for a in asks]}

    def tool_catalog(self) -> Dict[str, Any]:
        """Every tool that trades: its operations and their parameters, its
        price ceiling, and the cheapest ask now. Public, for agents."""

        tools = []
        for tool in sorted(self._config.tools):
            listed = self.order_book(tool)
            best = listed["asks"][0] if listed["asks"] else None
            tools.append({
                "tool": tool, "unit": listed["unit"], "max_ask_microusd": catalog.MAX_ASK[listed["unit"]],
                "request_privacy": listed["request_privacy"],
                "best_ask_microusd": best and best["ask_microusd"],
                "buyer_pays_microusd": best and best["buyer_pays_microusd"],
                "operations": [{"operation": op.name, "does": op.does, "max_units": op.max_units,
                                "params": {name: {"kind": p.kind, "required": p.required, **(
                                    {"choices": list(p.choices)} if p.choices else {})}
                                           for name, p in op.params.items()}}
                               for (t, _), op in sorted(catalog.OPERATIONS.items()) if t == tool]})
        return {"tools": tools}

    def services(self) -> Dict[str, Any]:
        return discovery.services_document(self._config)

    def openapi(self) -> Dict[str, Any]:
        return discovery.openapi(self._config)

    # Calls ------------------------------------------------------------------
    def lead_search_validation(self, token: str, request: Mapping[str, Any]) -> dict:
        """Adapt Lead Search's raw-v2 ZeroBounce contract at the buyer boundary."""
        ceiling = self._config.lead_search_verify_max_ask
        if ceiling is None:
            raise ToolMarketError(404, "verification_route_disabled")
        if set(request) - {"payload", "forceToolRefresh"} or not isinstance(request.get("payload"), dict) or (
            type(request.get("forceToolRefresh", False)) is not bool
        ):
            raise ToolMarketError(400, "request_invalid")
        # This endpoint serves platform customer data. Every call requires private supply.
        result = self.call(token, {"tool": "zerobounce", "operation": "validate",
                                     "params": request["payload"], "private": True, "max_ask_microusd": ceiling})
        status, body = result["status"], result["body"]
        op = catalog.operation("zerobounce", "validate")
        answered = 200 <= op.status(status, body) < 300 and op.units(status, body) is not None
        return {"call_id": result["call_id"], "status": "completed" if answered else "failed",
                "toolResponse": {"raw": body} if answered else {},
                "billing": {"cost_usd": result["charge_microusd"] / 1_000_000}}

    def call(self, token: str, body: Mapping[str, Any], identity: Optional[BuyerIdentity] = None) -> Dict[str, Any]:
        """One catalog call, on a tab's ``token`` or, with ``identity`` (a
        btauth/1 request), charged to that hotkey's balance."""

        if not isinstance(body, Mapping) or set(body) - {"params", "max_ask_microusd", "private"} \
                != {"tool", "operation"}:
            raise ToolMarketError(400, "request_invalid")
        private = body.get("private", False)
        if not isinstance(private, bool):
            raise ToolMarketError(400, "private_invalid")
        tool = body["tool"]
        if not isinstance(tool, str) or tool not in self._config.tools:
            raise ToolMarketError(404, "tool_unavailable")
        ceiling = catalog.MAX_ASK[catalog.TOOLS[tool]]
        # The most the buyer will let a key ask per unit; the book can move
        # between reading it and calling.
        max_ask = body.get("max_ask_microusd", ceiling)
        _int(max_ask, 1, ceiling, "max_ask_invalid")
        try:
            op = catalog.operation(tool, body["operation"])
            url, headers, payload = op.build(body.get("params", {}))
        except catalog.CatalogError as error:
            raise ToolMarketError(400, error.code) from None
        if identity is not None:
            return self._signed_call(identity.hotkey, tool, op, max_ask, private, url, headers, payload)
        if not token:
            raise ToolMarketError(401, "tab_unknown")
        tab = token_hash(token)
        # Before the chain and the book are read, so a bad token costs one lookup.
        opened = self._store.tab_state(tab)
        if opened is None:
            raise ToolMarketError(401, "tab_unknown")
        if not opened["open"]:
            raise ToolMarketError(401, "tab_closed")
        hotkey = opened["hotkey"]
        if not self._config.may_buy(hotkey):  # also for a tab opened before the allowlist changed
            raise ToolMarketError(403, "buyer_not_allowed")
        with self._slot(hotkey, MAX_CALLS_IN_FLIGHT_PER_BUYER, "too_many_calls_in_flight"):
            return self._buy(tool, op, max_ask, private, tab, hotkey, url, headers, payload)

    def _signed_call(self, hotkey: str, tool: str, op: catalog.Operation, max_ask: int, private: bool,
                     url: str, headers: Mapping[str, str], payload: bytes) -> Dict[str, Any]:
        """A call on a tab of its own, closed when the call ends, that holds
        what the keys it may try can cost, so concurrent calls share a small
        balance and a call still fails over to a pricier key it can afford.
        Gamma spent since the last reading is credited only when the stored
        balance falls short, so most calls read no chain."""

        if not self._config.may_buy(hotkey):
            raise ToolMarketError(403, "buyer_not_allowed")
        tab = token_hash(self._new_token())

        def open_for(budget: int) -> Dict[str, Any]:
            return self._store.open_tab(tab_id=self._new_id(), hotkey=hotkey, token_hash=tab, budget_microusd=budget,
                                        seconds=CALL_TAB_SECONDS, terms_sha256=None)

        with self._slot(hotkey, MAX_CALLS_IN_FLIGHT_PER_BUYER, "too_many_calls_in_flight"):
            asks = list(self._asks(tool, op, max_ask, private, hotkey))[:MAX_ASKS_PER_CALL]
            if not asks:
                raise ToolMarketError(503, "no_supply")
            needs = [book.charge_microusd(owed_cap) for _, owed_cap in asks]
            opened, unread = open_for(max(needs)), None
            if opened["status"] == "insufficient_balance":
                have = int(opened["balance"].get("balance_microusd") or 0)
                try:  # credit Gamma spent since the last reading
                    have = int(self.balance(hotkey).get("balance_microusd") or 0)
                except ToolMarketError as error:
                    unread = error  # the chain is busy or down: the stored balance still counts
                if min(needs) <= have:
                    opened = open_for(min(max(needs), have))
            if opened["status"] != "opened":
                # Unread new Gamma may cover it: say retry, not spend more.
                raise unread or ToolMarketError(402, "insufficient_balance")
            try:
                return self._buy(tool, op, max_ask, private, tab, hotkey, url, headers, payload, asks)
            except ToolMarketError as error:
                if error.code == "tab_budget":  # the tab is this call's share of the balance
                    raise unread or ToolMarketError(402, "insufficient_balance") from None
                raise
            finally:
                try:
                    self._store.close_tab(tab)
                except ArenaStoreError:
                    pass  # it then lapses in CALL_TAB_SECONDS; the answer must still reach the buyer

    @contextmanager
    def _slot(self, name: str, limit: int, code: str):
        """Hold one of ``limit`` slots for ``name`` in this process, or 429."""

        with self._lock:
            if self._in_flight.get(name, 0) >= limit:
                raise ToolMarketError(429, code)
            self._in_flight[name] = self._in_flight.get(name, 0) + 1
        try:
            yield
        finally:
            with self._lock:
                left = self._in_flight.pop(name) - 1
                if left:
                    self._in_flight[name] = left

    def _buy(self, tool: str, op: catalog.Operation, max_ask: int, private: bool, tab: str, hotkey: str,
             url: str, headers: Mapping[str, str], payload: bytes,
             asks: Optional[List[Tuple[book.Ask, int]]] = None) -> Dict[str, Any]:
        """Reserve and send on the cheapest keys that fit, up to the per-call limits.

        ``asks``: the book as the caller already read it; each reserve re-checks
        its key under the key's lock, so a moment-old book cannot over-reserve."""

        # Read before any vendor request, so a chain outage cannot strand a paid answer.
        try:
            epoch = int(self._epoch())
        except ArenaChainError:
            raise ToolMarketError(503, "chain_unavailable") from None
        tried = attempts = 0
        short = False
        for ask, owed_cap in self._asks(tool, op, max_ask, private, hotkey) if asks is None else asks:
            if tried >= MAX_KEYS_PER_CALL or attempts >= MAX_ASKS_PER_CALL:
                break
            attempts += 1
            call_id = self._new_id()
            status = self._store.reserve_call(
                call_id=call_id, token_hash=tab, tool=tool, operation=op.name, key_id=ask.key_id,
                ask_microusd=ask.ask_microusd, owed_cap_microusd=owed_cap)
            if status == "tab_closed":
                raise ToolMarketError(401, "tab_closed")
            if status == "tab_budget":
                short = True  # a cheaper key further down may still fit
                continue
            if status == "uncertain_limit":
                raise ToolMarketError(429, "uncertain_limit")
            if status != "reserved":
                continue
            tried += 1
            outcome = self._send(op, ask, call_id, epoch, url, headers, payload)
            if outcome is not None:
                return outcome
        if short:
            raise ToolMarketError(402, "tab_budget")
        raise ToolMarketError(503, "no_supply")

    def _send(self, op: catalog.Operation, ask: book.Ask, call_id: str, epoch: int, url: str,
              headers: Mapping[str, str], payload: bytes) -> Optional[Dict[str, Any]]:
        """One vendor request on one key. ``None`` means try the next key."""

        def settle(status: str, http_status: Optional[int], *, units: int = 0, owed: int = 0, charge: int = 0) -> bool:
            args = dict(call_id=call_id, status=status, http_status=http_status, units=units,
                        owed_microusd=owed, charge_microusd=charge, epoch=epoch)
            for _ in range(2):  # a settlement already written comes back as "exists"
                try:
                    return self._store.settle_call(**args) in ("settled", "exists")
                except ArenaStoreError:  # any failure: a commit may hide behind it
                    pass
            # Charged or not is unknown: usage shows it under this call_id.
            raise ToolMarketError(503, "store_unavailable", call_id=call_id)

        try:
            response = self._transport.send(
                method=op.method, url=url, headers=dict(headers, authorization="Bearer " + ask.envelope),
                body=payload, timeout_seconds=CALL_TIMEOUT_SECONDS, max_response_bytes=MAX_RESPONSE_BYTES)
        except VaultTransportError as error:
            if error.forwarded is False:
                settle("skipped", None)
                return None
            # It may have reached the vendor and used the key: the miner
            # bears it, and a retry could use it twice.
            settle("uncertain", error.observed_status)
            raise ToolMarketError(502, "vendor_unreachable", call_id=call_id) from None
        document = _document(response.body)
        status = op.status(response.status, document)
        # 401 and 402 may mean the key is unusable. A 429, Retry-After or not,
        # is the key's limit (its miner's account), not the buyer's, so
        # another key may serve; a 403 only skips the key.
        if status in (401, 402, 403, 429):
            settle("skipped", status)  # the real status, also when a 2xx body reported it
            if status in (401, 402):
                self._key_failed(op.tool, ask, status)
            return None
        units = op.units(status, document)
        charge = 0
        if units is None:
            settle("uncertain", response.status)
        else:
            owed = book.owed_microusd(op, units, ask.ask_microusd)
            # A call that settles too late holds nothing, so the buyer pays nothing.
            charge = book.charge_microusd(owed)
            if not settle("settled", response.status, units=units, owed=owed, charge=charge):
                charge = 0
        return {"call_id": call_id, "status": response.status, "body": document, "charge_microusd": charge}

    # Helpers ----------------------------------------------------------------
    def _require_supplier(self, tool: str, hotkey: str) -> None:
        if not self._config.may_supply(tool, hotkey):
            raise ToolMarketError(403, "supplier_not_approved")

    def _asks(self, tool: str, op: catalog.Operation, max_ask: int, private: bool, buyer: str):
        """The book's asks, best first, that a reserve for this call could take
        (never the buyer's own keys), each with the most the call could owe on it."""

        for ask in self._supply(tool):
            owed_cap = book.owed_microusd(op, op.max_units, ask.ask_microusd)
            if ask.ask_microusd <= max_ask and (not private or self._private(tool, ask)) \
                    and owed_cap <= ask.remaining_microusd and ask.miner_hotkey != buyer:
                yield ask, owed_cap

    def _supply(self, tool: str) -> List[book.Ask]:
        """The book's asks from approved suppliers, best first."""

        rows = self._store.book(tool=tool, terms_sha256=self._config.terms_sha256)
        return book.rank(book.Ask.from_row(row) for row in rows if self._config.may_supply(tool, row["miner_hotkey"]))

    def _check(self, tool: str, envelope: str) -> Mapping[str, Any]:
        try:
            return self._vault.check(provider=tool, envelope=envelope, management_envelope=None)
        except VaultUnavailable as error:
            raise ToolMarketError(503 if error.retryable else 400, error.code) from None

    def _recheck(self, key_id: str, tool: str, envelope: str) -> Optional[Mapping[str, Any]]:
        """Check a listed key with its vendor again and record the answer; None when the vault gave none."""

        try:
            checked = self._check(tool, envelope)
        except ToolMarketError:
            return None
        if checked.get("status") not in ("valid", "unchecked", "invalid"):
            return None
        self._store.record_key_event(key_id=key_id, status=checked["status"], reason=checked.get("reason"))
        return checked

    def _key_failed(self, tool: str, ask: book.Ask, status: int) -> None:
        """A call answered as if the key were unusable. A buyer shapes the
        request, so the answer alone does not always take a key out:

        - 401 (a bad key or account) always fails it;
        - 402 (no credit) fails it only when the vendor's free check, due at
          most once a minute per key, sees no balance; otherwise, and for a
          tool with no check, it only skips the key.
        """

        if status == 402:
            if vault_tools.TOOLS[tool].balance_url is None or not self._recheck_due(ask.key_id):
                return
            checked = self._recheck(ask.key_id, tool, ask.envelope)
            # Unknown, already recorded as invalid, or it still has credit.
            if checked is None or checked["status"] == "invalid" or checked.get("balance") is not None:
                return
        self._store.record_key_event(key_id=ask.key_id, status="invalid",
                                     reason="unauthorized" if status == 401 else "no_credit")

    def _private(self, tool: str, ask: book.Ask) -> bool:
        """A key that may serve a private call: its miner signed a private-supply
        agreement, or Leadpoet checked that the vendor shows owners no requests."""

        return ask.miner_hotkey in self._config.private_suppliers or catalog.REQUEST_PRIVACY[tool] == "private"

    def _recheck_due(self, name: str) -> bool:
        """At most one vendor check a minute per key (or registration probe)
        in this process, so a burst cannot hammer a miner's account."""

        now = time.monotonic()
        with self._lock:
            if now - self._rechecked.get(name, -RECHECK_SPACING_SECONDS) < RECHECK_SPACING_SECONDS:
                return False
            self._rechecked[name] = now
            return True

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
