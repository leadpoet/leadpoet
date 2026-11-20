"""Lead-order orchestration: buyers, orders, runner claims, settlement.

Everything here is reached only when ``LEADPOET_LEAD_ORDERS_MODE`` is not
``off``. Money rules run in the database (``scripts/703-lead-orders.sql``)
and in ``money``; this module wires them together and never touches an
Arena table, reward or weight.

The confirmation mailer is a stub that fails closed: there is no mail sender
in this repository yet, so a registration is stored unconfirmed and the API
answers 503 ``confirmation_email_unavailable``. Live orders need a confirmed
buyer, so no live order can be placed until a real mailer is wired in.
"""

from __future__ import annotations

import re
import secrets
import time
import uuid
from typing import Any, Callable, Dict, List, Mapping, Optional, Protocol

from lab_arena import signing
from lab_arena.contracts import canonical_json
from lab_arena.lead_orders import keys as lead_keys
from lab_arena.lead_orders import receipts, rewards, suppressions
from lab_arena.lead_orders import weights as lane_weights
from lab_arena.lead_orders.auth import BuyerIdentity, order_token_hash
from lab_arena.lead_orders.config import MODE_LIVE, LeadOrdersConfig
from lab_arena.lead_orders.gamma import GammaBalanceReader, GammaReadError
from lab_arena.lead_orders.policy import LeadOrderPolicy, load_policy
from lab_arena.lead_orders.store import LeadOrderStore

CONFIRMATION_MAX_AGE_SECONDS = 48 * 3600
MAX_ORDER_BYTES = 16_384
# Longer than the slowest provider timeout (Deepline, 240 s), so calls in
# flight at the deadline settle before an overdue order is closed.
OVERDUE_GRACE_SECONDS = 300
RECONCILE_BATCH = 20
_EMAIL_RE = re.compile(r"^[A-Za-z0-9._%+-]{1,64}@([A-Za-z0-9-]{1,63}\.)+[A-Za-z]{2,24}$")
_COUNTRY_RE = re.compile(r"^[A-Z]{2}$")
_UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")

# Free-mail and disposable domains refused at registration: a buyer must use
# a company address. Extend by reviewed change; not a substitute for review.
FREE_MAIL_DOMAINS = frozenset({
    "gmail.com", "googlemail.com", "yahoo.com", "yahoo.co.uk", "ymail.com", "hotmail.com", "hotmail.co.uk",
    "outlook.com", "live.com", "msn.com", "aol.com", "icloud.com", "me.com", "mac.com", "proton.me",
    "protonmail.com", "pm.me", "gmx.com", "gmx.de", "gmx.net", "web.de", "mail.com", "yandex.com", "yandex.ru",
    "mail.ru", "zoho.com", "qq.com", "163.com", "126.com", "tutanota.com", "tuta.io", "fastmail.com",
    "hey.com", "duck.com", "mailinator.com", "guerrillamail.com", "10minutemail.com", "tempmail.com",
    "temp-mail.org", "sharklasers.com", "yopmail.com", "trashmail.com", "getnada.com", "dispostable.com",
})

_PLACE_REFUSALS = {
    "below_minimum": (422, "below_minimum"),
    "daily_cap": (429, "daily_cap"),
    "gamma_unavailable": (409, "gamma_unavailable"),
    "insufficient_balance": (402, "insufficient_balance"),
    "conflict": (409, "order_id_taken"),
}


class LeadOrderServiceError(Exception):
    def __init__(self, status: int, code: str) -> None:
        self.status = status
        self.code = code
        super().__init__(code)


class MailerUnavailable(RuntimeError):
    pass


class ConfirmationMailer(Protocol):
    def send_confirmation(self, *, email: str, confirm_url: str) -> None: ...


class UnavailableMailer:
    """Fail-closed stub: no mail sender exists in this repository yet."""

    def send_confirmation(self, *, email: str, confirm_url: str) -> None:
        raise MailerUnavailable("no confirmation mail sender is configured")


def usd(microusd: Any) -> Optional[str]:
    """Micro-USD as a decimal USD string with at least two places: 5000000 -> "5.00"."""

    if microusd is None:
        return None
    text = "%d.%06d" % divmod(int(microusd), 1_000_000)
    whole, frac = text.split(".")
    return whole + "." + (frac.rstrip("0").ljust(2, "0"))


def email_domain(email: str) -> str:
    return email.rsplit("@", 1)[-1].lower()


class LeadOrderService:
    def __init__(
        self,
        *,
        config: LeadOrdersConfig,
        policy: LeadOrderPolicy,
        policy_document: Mapping[str, Any],
        store: LeadOrderStore,
        signer: signing.ArenaSigner,
        current_epoch: Callable[[], int],
        gamma_reader: Optional[GammaBalanceReader] = None,
        mailer: Optional[ConfirmationMailer] = None,
        emission_price: Optional[rewards.EmissionPriceSource] = None,
        registrar: Optional[lead_keys.MinerKeyRegistrar] = None,
    ) -> None:
        if not config.enabled:
            raise LeadOrderServiceError(404, "lead_orders_off")
        if config.uses_gamma and gamma_reader is None:
            raise LeadOrderServiceError(503, "gamma_reader_required")
        self.config = config
        self.policy = policy
        self._store = store
        self._signer = signer
        self._current_epoch = current_epoch
        self._gamma = gamma_reader
        self._mailer = mailer or UnavailableMailer()
        self._emission_price = emission_price or rewards.UnavailableEmissionPrice()
        self._registrar = registrar
        self._policies: Dict[int, LeadOrderPolicy] = {policy.version: policy}
        store.record_policy(policy_document)

    def _order_policy(self, version: int) -> LeadOrderPolicy:
        """The policy an order was placed under: its rewards and receipt use
        it, as its charge does, so a later policy change never applies to it."""

        version = int(version)
        if version not in self._policies:
            document = self._store.policy_document(version)
            if not document:
                raise LeadOrderServiceError(500, "order_policy_unknown")
            self._policies[version] = load_policy(
                document, public_key_der=self._signer.public_key_der,
                expected_public_key_hash=self._signer.public_key_hash,
            )
        return self._policies[version]

    # -- buyers -------------------------------------------------------------
    def register_buyer(self, hotkey: str, payload: Mapping[str, Any]) -> Dict[str, Any]:
        if not self.config.may_order(hotkey):
            raise LeadOrderServiceError(403, "hotkey_not_allowed")
        if not isinstance(payload, Mapping) or set(payload) - {"email", "company", "country", "terms_version"}:
            raise LeadOrderServiceError(422, "registration_invalid")
        email = str(payload.get("email") or "").strip()
        company = str(payload.get("company") or "").strip()
        country = str(payload.get("country") or "").strip().upper()
        if not _EMAIL_RE.fullmatch(email):
            raise LeadOrderServiceError(422, "email_invalid")
        if email_domain(email) in FREE_MAIL_DOMAINS:
            raise LeadOrderServiceError(422, "company_email_required")
        if not 1 <= len(company) <= 200:
            raise LeadOrderServiceError(422, "company_invalid")
        if not _COUNTRY_RE.fullmatch(country):
            raise LeadOrderServiceError(422, "country_invalid")
        if payload.get("terms_version") != self.config.terms_version:
            raise LeadOrderServiceError(422, "terms_not_accepted")
        token = secrets.token_urlsafe(32)
        registration_id = str(uuid.uuid4())
        self._store.register_buyer(
            registration_id=registration_id, hotkey=hotkey, email=email, company=company, country=country,
            terms_version=self.config.terms_version, token_hash=order_token_hash(token),
        )
        url = "%s/lead-orders/buyers/confirm?token=%s" % (self.config.public_base_url, token)
        try:
            self._mailer.send_confirmation(email=email, confirm_url=url)
        except MailerUnavailable:
            raise LeadOrderServiceError(503, "confirmation_email_unavailable") from None
        return {"status": "pending", "registration_id": registration_id}

    def confirm_buyer(self, token: str) -> Dict[str, Any]:
        if not token or len(token) > 128:
            raise LeadOrderServiceError(404, "confirmation_unknown")
        result = self._store.confirm_buyer(order_token_hash(token), max_age_seconds=CONFIRMATION_MAX_AGE_SECONDS)
        status = result.get("status")
        if status == "unknown":
            raise LeadOrderServiceError(404, "confirmation_unknown")
        if status == "expired":
            raise LeadOrderServiceError(410, "confirmation_expired")
        return {"status": "confirmed"}

    # -- orders -------------------------------------------------------------
    def _refresh_balance(self, hotkey: str) -> Dict[str, Any]:
        assert self._gamma is not None
        try:
            reading = self._gamma.read(hotkey)
        except GammaReadError:
            raise LeadOrderServiceError(503, "gamma_read_failed") from None
        result = self._store.record_gamma_reading(
            hotkey=hotkey, block_number=reading.block_number, block_hash=reading.block_hash,
            spent_units=reading.spent_units, available=reading.available,
        )
        if result.get("status") == "counter_backwards":
            raise LeadOrderServiceError(422, "gamma_counter_backwards")
        return result

    def balance(self, hotkey: str) -> Dict[str, Any]:
        if not self.config.may_order(hotkey):
            raise LeadOrderServiceError(403, "hotkey_not_allowed")
        if not self.config.uses_gamma:
            return {"mode": self.config.mode, "gamma": False}
        return dict(self._refresh_balance(hotkey), mode=self.config.mode, gamma=True)

    def place_order(self, hotkey: str, payload: Mapping[str, Any]) -> Dict[str, Any]:
        if not self.config.may_order(hotkey):
            raise LeadOrderServiceError(403, "hotkey_not_allowed")
        if not isinstance(payload, Mapping) or set(payload) - {"order_id", "leads", "seller", "search"}:
            raise LeadOrderServiceError(422, "order_invalid")
        leads = payload.get("leads")
        if type(leads) is not int or not 1 <= leads <= 100_000:
            raise LeadOrderServiceError(422, "leads_invalid")
        request_doc = {}
        for name in ("seller", "search"):
            part = payload.get(name)
            if not isinstance(part, Mapping) or not part:
                raise LeadOrderServiceError(422, "%s_invalid" % name)
            request_doc[name] = dict(part)
        if len(canonical_json(request_doc).encode("utf-8")) > MAX_ORDER_BYTES:
            raise LeadOrderServiceError(422, "order_too_large")
        order_id = str(payload.get("order_id") or uuid.uuid4())
        if not _UUID_RE.fullmatch(order_id):
            raise LeadOrderServiceError(422, "order_id_invalid")
        if self.config.mode == MODE_LIVE:
            buyer = self._store.buyer(hotkey)
            if not buyer.get("confirmed"):
                raise LeadOrderServiceError(403, "buyer_not_confirmed")
            if buyer.get("terms_version") != self.config.terms_version:
                raise LeadOrderServiceError(403, "terms_not_accepted")
            self._refresh_balance(hotkey)
        result = self._store.place(
            order_id=order_id, hotkey=hotkey, mode=self.config.mode, policy_version=self.policy.version,
            leads=leads, request_doc=request_doc,
        )
        status = result.get("status")
        if status in _PLACE_REFUSALS:
            code, reason = _PLACE_REFUSALS[status]
            raise LeadOrderServiceError(code, reason)
        if status == "exists":
            view = self._store.view(order_id)
            if not view or view.get("hotkey") != hotkey:
                raise LeadOrderServiceError(409, "order_id_taken")
            return self._buyer_view(view)
        return self._buyer_view(self._store.view(order_id))

    @staticmethod
    def _buyer_view(view: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
        if not view:
            raise LeadOrderServiceError(404, "order_unknown")
        fields = (
            "order_id", "mode", "policy_version", "status", "leads_requested", "leads_delivered",
            "hold_microusd", "spend_ceiling_microusd", "spend_microusd", "charge_microusd",
            "delivery_url", "failure_reason", "receipt",
        )
        out = {name: view.get(name) for name in fields}
        out["hold_usd"] = usd(view.get("hold_microusd"))
        out["charged_usd"] = usd(view.get("charge_microusd"))
        return out

    def order_view(self, hotkey: str, order_id: str) -> Dict[str, Any]:
        if not _UUID_RE.fullmatch(order_id or ""):
            raise LeadOrderServiceError(404, "order_unknown")
        view = self._store.view(order_id)
        if not view or view.get("hotkey") != hotkey:
            raise LeadOrderServiceError(404, "order_unknown")
        return self._buyer_view(view)

    # -- runner ---------------------------------------------------------------
    def expire_overdue(self) -> List[str]:
        expired = []
        for order_id in self._store.overdue(grace_seconds=OVERDUE_GRACE_SECONDS):
            result = self._store.finish(
                order_id=order_id, status="failed", delivered=0, delivery_url=None,
                failure_reason="deadline_passed", epoch=int(self._current_epoch()),
            )
            if result.get("status") in ("finished", "exists"):
                expired.append(order_id)  # settled by reconcile()
        return expired

    def claim(self) -> Optional[Dict[str, Any]]:
        """Claim the next order for the runner. The one-order token is shown once."""

        self.expire_overdue()
        self.reconcile()
        token = secrets.token_urlsafe(32)
        claimed = self._store.claim(order_token_hash(token))
        if not claimed:
            return None
        request_doc = claimed.get("request_doc") or {}
        base = self.config.public_base_url
        return {
            "order_id": str(claimed["order_id"]),
            "hotkey": claimed["hotkey"],
            "mode": claimed["mode"],
            "leads_requested": int(claimed["leads_requested"]),
            "deadline": claimed["deadline"],
            "spend_ceiling_usd": usd(claimed["spend_ceiling_microusd"]),
            "seller": request_doc.get("seller"),
            "search": request_doc.get("search"),
            "broker": {
                "openrouter_base_url": base + "/lead-orders/broker/openrouter/api/v1",
                "deepline_base_url": base + "/lead-orders/broker/deepline",
                "token": token,
            },
        }

    def _running_view(self, order_id: str) -> Dict[str, Any]:
        view = self._store.view(order_id) if _UUID_RE.fullmatch(order_id or "") else None
        if not view or view.get("status") == "queued":
            raise LeadOrderServiceError(404, "order_unknown")
        return view

    def heartbeat(self, order_id: str) -> Dict[str, Any]:
        """Whether the runner should keep going: false once finished, past the
        deadline, or at the spend ceiling. ``spent_usd`` counts settled spend
        plus open and uncertain reservations, as the ceiling does."""

        view = self._running_view(order_id)
        committed = int(view.get("committed_microusd") or 0)
        keep_going = (view["status"] == "running" and not view.get("expired")
                      and committed < int(view["spend_ceiling_microusd"]))
        return {"continue": bool(keep_going), "spent_usd": usd(committed), "deadline": view.get("deadline")}

    def finish(self, order_id: str, payload: Mapping[str, Any]) -> Dict[str, Any]:
        view = self._running_view(order_id)
        if not isinstance(payload, Mapping) or set(payload) - {"status", "leads_delivered", "delivery_url", "failure_reason"}:
            raise LeadOrderServiceError(422, "finish_invalid")
        status = payload.get("status")
        delivered = payload.get("leads_delivered", 0)
        url = payload.get("delivery_url")
        reason = payload.get("failure_reason")
        if status not in ("delivered", "failed") or type(delivered) is not int or delivered < 0:
            raise LeadOrderServiceError(422, "finish_invalid")
        if delivered > int(view["leads_requested"]):
            raise LeadOrderServiceError(422, "leads_delivered_invalid")
        if status == "delivered" and (not isinstance(url, str) or not url.startswith("https://") or len(url) > 2048):
            raise LeadOrderServiceError(422, "delivery_url_invalid")
        if status == "failed" and reason is not None and (not isinstance(reason, str) or len(reason) > 200):
            raise LeadOrderServiceError(422, "failure_reason_invalid")
        result = self._store.finish(
            order_id=order_id, status=status, delivered=delivered if status == "delivered" else 0,
            delivery_url=url if status == "delivered" else None,
            failure_reason=reason if status == "failed" else None, epoch=int(self._current_epoch()),
        )
        if result.get("status") == "unknown":
            raise LeadOrderServiceError(404, "order_unknown")
        if result.get("status") == "delivered_out_of_range":
            raise LeadOrderServiceError(422, "leads_delivered_invalid")
        receipt = self._settle(order_id)
        return {"status": result.get("order_status"), "charge_microusd": result.get("charge_microusd"),
                "charged_usd": usd(result.get("charge_microusd")), "receipt_hash": receipt["receipt_hash"]}

    def _settle(self, order_id: str) -> Dict[str, Any]:
        """Record rewards and a signed receipt; idempotent.

        Uses the order's own policy version. The call rows are final at
        finish (open calls settle uncertain there), and the exact rows the
        Merkle root is built from are stored with the receipt.
        """

        view = self._store.view(order_id)
        if view and view.get("receipt"):
            return view["receipt"]
        if not view or view.get("status") not in ("delivered", "failed"):
            raise LeadOrderServiceError(409, "order_not_finished")
        calls = self._store.calls(order_id)
        if any(row.get("status") == "open" for row in calls):
            raise LeadOrderServiceError(500, "order_calls_open")
        policy = self._order_policy(view["policy_version"])
        owed: Dict[str, int] = {}
        if view["mode"] in ("shadow", "live"):
            owed = rewards.order_rewards(policy, order=view, calls=calls)
        self._store.record_rewards(order_id, owed)
        leaves = receipts.call_leaves(calls)
        receipt = receipts.sign_receipt(
            self._signer,
            receipts.build_receipt(order=view, calls=calls, rewards=owed, policy_hash=policy.policy_hash),
        )
        self._store.record_receipt(order_id, receipt, leaves)
        return receipt

    def reconcile(self, *, limit: int = RECONCILE_BATCH) -> List[str]:
        """Settle finished orders that have no receipt yet (a signer or store
        failure after the finish committed). Idempotent; a failure leaves the
        order for the next pass."""

        settled = []
        for order_id in self._store.unsettled(limit=limit):
            try:
                self._settle(order_id)
            except Exception:  # retried on the next claim or epoch pass
                continue
            settled.append(order_id)
        return settled

    # -- miner keys -----------------------------------------------------------
    def register_key(self, hotkey: str, payload: Mapping[str, Any]) -> Dict[str, Any]:
        """A miner offers one provider key; it is validated, encrypted, stored."""

        if not self.config.uses_miner_keys or self._registrar is None:
            raise LeadOrderServiceError(403, "miner_keys_not_accepted")
        if hotkey in self.config.blocked_hotkeys:
            raise LeadOrderServiceError(403, "hotkey_not_allowed")
        if not isinstance(payload, Mapping) or set(payload) != {"provider", "credentials"}:
            raise LeadOrderServiceError(422, "key_invalid")
        if not self._store.has_miner_terms(hotkey, self.policy.miner_terms_version):
            raise LeadOrderServiceError(403, "miner_terms_not_accepted")
        try:
            row = self._registrar.register(
                miner_hotkey=hotkey, provider=str(payload["provider"]), credentials=payload["credentials"],
            )
        except lead_keys.LeadOrderKeyError as exc:
            raise LeadOrderServiceError(422, exc.code) from None
        public = lead_keys.ProviderKey(
            key_id=row["key_id"], provider=row["provider"], miner_hotkey=hotkey, registered_sequence=0,
            internal_fingerprint=row["internal_fingerprint"],
        ).public_fingerprint
        result = self._store.register_key(row, public_fingerprint=public)
        if result.get("status") != "registered":
            raise LeadOrderServiceError(409, str(result.get("status") or "key_not_registered"))
        return {"status": "registered", "key_id": row["key_id"], "public_fingerprint": public}

    def revoke_key(self, hotkey: str, key_id: str) -> Dict[str, Any]:
        owned = [row for provider in lead_keys.PROVIDERS for row in self._store.keys(provider, terms_version=None)
                 if row["key_id"] == key_id and row["miner_hotkey"] == hotkey]
        if not owned:
            raise LeadOrderServiceError(404, "key_unknown")
        self._store.revoke_key(key_id, reason="revoked_by_miner")
        return {"status": "revoked", "key_id": key_id}

    # -- miner data terms and the suppression feed --------------------------------
    def accept_miner_terms(self, identity: BuyerIdentity, payload: Mapping[str, Any]) -> Dict[str, Any]:
        """Store a miner's signed acceptance of the current miner data terms."""

        if identity.hotkey in self.config.blocked_hotkeys:
            raise LeadOrderServiceError(403, "hotkey_not_allowed")
        if not isinstance(payload, Mapping) or set(payload) != {"terms_version"}:
            raise LeadOrderServiceError(422, "terms_invalid")
        if payload["terms_version"] != self.policy.miner_terms_version:
            raise LeadOrderServiceError(422, "terms_version_not_current")
        if not identity.signature or not identity.body_sha256:
            raise LeadOrderServiceError(401, "signature_invalid")
        result = self._store.accept_miner_terms(
            hotkey=identity.hotkey, version=self.policy.miner_terms_version, signed_at=identity.signed_at,
            body_sha256=identity.body_sha256, signature=identity.signature,
        )
        return {"status": str(result["status"]), "terms_version": self.policy.miner_terms_version,
                "accepted_at": result["accepted_at"]}

    def push_suppressions(self, payload: Any) -> Dict[str, Any]:
        """Runner only: append hashed suppressions. Raw values are refused."""

        try:
            rows = suppressions.rows_from_push(payload, now=time.time())
        except suppressions.SuppressionError as exc:
            raise LeadOrderServiceError(422, exc.code) from None
        result = self._store.add_suppressions(rows)
        return {"received": len(payload["suppressions"]), "added": int(result["added"])}

    def read_suppressions(self, hotkey: str, since: Any) -> Dict[str, Any]:
        """The feed for a confirmed buyer or a miner who accepted the current terms."""

        if hotkey in self.config.blocked_hotkeys:
            raise LeadOrderServiceError(403, "hotkey_not_allowed")
        allowed = (self._store.has_miner_terms(hotkey, self.policy.miner_terms_version)
                   or bool(self._store.buyer(hotkey).get("confirmed")))
        if not allowed:
            raise LeadOrderServiceError(403, "suppressions_not_allowed")
        text = "0" if since in (None, "") else str(since)
        if not text.isdigit() or len(text) > 18:
            raise LeadOrderServiceError(422, "since_invalid")
        rows = self._store.suppressions(since=int(text), limit=suppressions.PAGE_SIZE)
        return {"suppressions": rows, "next_since": rows[-1]["sequence"] if rows else int(text),
                "more": len(rows) == suppressions.PAGE_SIZE}

    # -- public -------------------------------------------------------------
    def public_epoch(self, epoch: Any) -> Dict[str, Any]:
        text = str(epoch)
        number = int(text) if text.isdigit() and len(text) <= 18 else -1
        document = self._store.reward_epoch(number) if number >= 0 else None
        if not document:
            raise LeadOrderServiceError(404, "epoch_unknown")
        return document

    def public_receipt(self, order_id: str) -> Dict[str, Any]:
        if not _UUID_RE.fullmatch(order_id or ""):
            raise LeadOrderServiceError(404, "receipt_unknown")
        view = self._store.view(order_id)
        if not view or not view.get("receipt"):
            raise LeadOrderServiceError(404, "receipt_unknown")
        return {"receipt": view["receipt"], "calls": view.get("receipt_calls") or []}

    def run_epoch(
        self,
        epoch: int,
        *,
        payable_cap_microusd: Optional[int] = None,
        registered_hotkeys: Optional[List[str]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Compute and record one epoch's payout document (shadow and live).

        In live mode the caller passes the finalized metagraph hotkeys and the
        burn residual's USD value, so nothing is paid that cannot be weighted.
        """

        if not self.config.computes_rewards:
            return None
        existing = self._store.reward_epoch(epoch)
        if existing:
            return existing
        self.reconcile()
        try:
            emission = self._emission_price.miner_emission_microusd(epoch)
        except rewards.EmissionPriceUnavailable:
            if self.config.weights_rewards:
                return None  # live: no payout, Arena weights unchanged
            emission = None
        entries = rewards.owed_entries(self._store.owed(through_epoch=epoch, mode=self.config.mode))
        if registered_hotkeys is not None:
            entries = lane_weights.registered_entries(entries, registered_hotkeys)
        elif self.config.weights_rewards:
            raise LeadOrderServiceError(500, "registered_hotkeys_required")
        document = rewards.sign_payout(self._signer, rewards.epoch_payout_document(
            self.policy, epoch=epoch, mode=self.config.mode, entries=entries,
            miner_emission_microusd=emission, payable_cap_microusd=payable_cap_microusd,
        ))
        self._store.record_payout(epoch=epoch, mode=self.config.mode, document=document,
                                  payments=document["payments"])
        return document


__all__ = [
    "FREE_MAIL_DOMAINS",
    "LeadOrderService",
    "LeadOrderServiceError",
    "MailerUnavailable",
    "UnavailableMailer",
]
