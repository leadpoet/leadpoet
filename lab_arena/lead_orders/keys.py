"""Provider key registry, ranking and failover rules for lead orders.

Miner keys are sealed to the lead-order key vault (``lead_order_vault``) on
the miner's own machine. Leadpoet stores only the envelope and sends it to
the vault as each request's credential; it never sees the key. Leadpoet's
own keys come from the environment and are always last.
"""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional

from lab_arena import broker as arena_broker
from lab_arena.lead_orders.vault_client import VaultClient, VaultUnavailable
from lead_order_vault import envelope as vault_envelope

PROVIDERS = ("openrouter", "deepline")
LEADPOET_KEY_PREFIX = "leadpoet-"
SUBMISSION_ID_PREFIX = "lead-order:"

FAILOVER_UNAUTHORIZED = "unauthorized"
FAILOVER_PAYMENT_REQUIRED = "payment_required"
FAILOVER_FORBIDDEN = "forbidden"
FAILOVER_RATE_LIMITED = "rate_limited_no_retry"
FAILOVER_SPENT_LIMIT = "spent_limit"

_KEY_ID_RE = re.compile(r"^[a-z0-9-]{1,64}$")
_FINGERPRINT_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


class LeadOrderKeyError(RuntimeError):
    """Safe key-registry error; never carries a secret."""

    def __init__(self, code: str, *, retryable: bool = False) -> None:
        self.code = code
        self.retryable = retryable
        super().__init__(code)


@dataclass(frozen=True)
class ProviderKey:
    """One usable key. ``miner_hotkey`` is ``None`` for Leadpoet's own key."""

    key_id: str
    provider: str
    miner_hotkey: Optional[str]
    registered_sequence: int
    internal_fingerprint: str
    ciphertext_b64: Optional[str] = None

    @property
    def is_leadpoet(self) -> bool:
        return self.miner_hotkey is None

    @property
    def public_fingerprint(self) -> str:
        """Published on receipts so a miner can find their own key.

        Derived from the internal fingerprint with a lead-order domain tag so
        it can never be matched against an Arena-internal binding.
        """

        material = b"leadpoet.lead_orders.key_fingerprint.v1\x00" + self.internal_fingerprint.encode("ascii")
        return "sha256:" + hashlib.sha256(material).hexdigest()[:32]


def credential_fingerprint(secret: str) -> str:
    # Same internal binding the Arena broker keeps for its call rows.
    return arena_broker._credential_fingerprint(secret)


def submission_id_for(key_id: str) -> str:
    if not _KEY_ID_RE.fullmatch(key_id):
        raise LeadOrderKeyError("key_id_invalid")
    return SUBMISSION_ID_PREFIX + key_id


def leadpoet_keys(openrouter_key: str, deepline_key: str) -> Dict[str, ProviderKey]:
    """Leadpoet's keys from configuration; a provider without a key has none."""

    out: Dict[str, ProviderKey] = {}
    for provider, secret in (("openrouter", openrouter_key), ("deepline", deepline_key)):
        if secret:
            out[provider] = ProviderKey(
                key_id=LEADPOET_KEY_PREFIX + provider,
                provider=provider,
                miner_hotkey=None,
                registered_sequence=0,
                internal_fingerprint=credential_fingerprint(secret),
            )
    return out


class MinerKeyRegistrar:
    """Register miner keys sealed to the lead-order key vault.

    The miner seals the key on their own machine (``lead_order_vault.cli
    seal``). Leadpoet stores the envelope, which only the vault can open, and
    the vault checks the key with its provider before it is accepted.
    """

    def __init__(self, vault: VaultClient, *, new_key_id: Callable[[], str] = lambda: str(uuid.uuid4())) -> None:
        self._vault = vault
        self._new_key_id = new_key_id

    def register(self, *, miner_hotkey: str, provider: str, credentials: Mapping[str, Any]) -> Dict[str, Any]:
        """Return the row to store: the envelope and the vault's key fingerprint."""

        if provider not in PROVIDERS:
            raise LeadOrderKeyError("provider_invalid")
        # OpenRouter also needs the management key's envelope: the vault uses
        # it once to prove the miner owns the runtime key, and it is not stored.
        expected = {"envelope", "management_envelope"} if provider == "openrouter" else {"envelope"}
        if not isinstance(credentials, Mapping) or set(credentials) != expected:
            raise LeadOrderKeyError("credentials_invalid")
        # The vault checks these headers again. Reading them here first means
        # an envelope that is not this miner's never makes the vault probe a
        # provider.
        for name, purpose in (("envelope", "runtime"), ("management_envelope", "management")):
            if name not in expected:
                continue
            try:
                header, _ = vault_envelope.read_header(credentials[name])
            except vault_envelope.EnvelopeError as exc:
                raise LeadOrderKeyError(str(exc)) from None
            if header.provider != provider or header.purpose != purpose:
                raise LeadOrderKeyError("envelope_mismatch")
            if header.miner_hotkey != miner_hotkey:
                raise LeadOrderKeyError("envelope_hotkey_mismatch")
        try:
            checked = self._vault.check(provider=provider, envelope=credentials["envelope"],
                                        management_envelope=credentials.get("management_envelope"))
        except VaultUnavailable as exc:
            # A 4xx is the envelope's fault (malformed, sealed elsewhere), so the
            # miner must fix it; an outage, or our caller token, is retried.
            raise LeadOrderKeyError(exc.code, retryable=exc.retryable) from None
        if checked.get("status") != "valid":
            raise LeadOrderKeyError(str(checked.get("reason") or "key_invalid"))
        fingerprint = checked.get("key_fingerprint")
        if checked.get("miner_hotkey") != miner_hotkey or checked.get("provider") != provider \
                or not isinstance(fingerprint, str) or not _FINGERPRINT_RE.fullmatch(fingerprint):
            raise LeadOrderKeyError("vault_answer_invalid", retryable=True)
        key_id = self._new_key_id()
        # The column names predate the vault: ``ciphertext_b64`` holds the
        # envelope, and ``submission_id`` only scopes the key id.
        return {
            "key_id": key_id,
            "provider": provider,
            "miner_hotkey": miner_hotkey,
            "submission_id": submission_id_for(key_id),
            "ciphertext_b64": credentials["envelope"],
            "internal_fingerprint": fingerprint,
        }

    @staticmethod
    def credential(key: ProviderKey) -> str:
        """The envelope the broker sends as the request's credential."""

        if key.is_leadpoet or key.miner_hotkey is None or key.ciphertext_b64 is None:
            raise LeadOrderKeyError("miner_credentials_unavailable")
        if sealed_to(key.ciphertext_b64) is None:
            # Stored before the vault: the miner must seal it again.
            raise LeadOrderKeyError("key_needs_reseal")
        return key.ciphertext_b64


def sealed_to(stored: Optional[str]) -> Optional[str]:
    """The vault key an envelope was sealed to, or ``None`` for anything else."""

    try:
        return vault_envelope.read_header(stored)[0].vault_key
    except vault_envelope.EnvelopeError:
        return None


def rank_keys(
    provider: str,
    miner_keys: Iterable[ProviderKey],
    leadpoet_key: Optional[ProviderKey],
    *,
    settled_spend_by_key: Mapping[str, int],
    use_miner_keys: bool,
    exhausted: Iterable[str] = (),
) -> List[ProviderKey]:
    """The order the broker tries keys in for one provider.

    Miner keys rank by USD spend settled over the ranking window, highest
    first; ties and new keys by registration order. Leadpoet's key is last.
    Keys this order already failed over from are skipped.
    """

    skip = set(exhausted)
    ranked: List[ProviderKey] = []
    if use_miner_keys:
        candidates = [k for k in miner_keys if k.provider == provider and not k.is_leadpoet and k.key_id not in skip]
        candidates.sort(key=lambda k: (-int(settled_spend_by_key.get(k.key_id, 0)), k.registered_sequence, k.key_id))
        ranked.extend(candidates)
    if leadpoet_key is not None and leadpoet_key.provider == provider and leadpoet_key.key_id not in skip:
        ranked.append(leadpoet_key)
    return ranked


def failover_reason(provider: str, response: "arena_broker.ProviderResponse") -> Optional[str]:
    """Why the broker must move to the next key for the rest of the order."""

    status = int(response.status)
    if status == 402:
        document: Any = None
        try:
            document = json.loads(bytes(response.body).decode("utf-8"))
        except (UnicodeDecodeError, ValueError):
            document = None
        spent = (
            arena_broker._openrouter_account_credit_response(response)
            if provider == "openrouter"
            else arena_broker._deepline_insufficient_credit_response(response, document)
        )
        return FAILOVER_SPENT_LIMIT if spent else FAILOVER_PAYMENT_REQUIRED
    if status == 401:
        return FAILOVER_UNAUTHORIZED
    if status == 403:
        return FAILOVER_FORBIDDEN
    if status == 429:
        headers = {str(k).lower(): v for k, v in response.headers.items()}
        if not str(headers.get("retry-after") or "").strip():
            return FAILOVER_RATE_LIMITED
    return None


__all__ = [
    "FAILOVER_FORBIDDEN",
    "FAILOVER_PAYMENT_REQUIRED",
    "FAILOVER_RATE_LIMITED",
    "FAILOVER_SPENT_LIMIT",
    "FAILOVER_UNAUTHORIZED",
    "LeadOrderKeyError",
    "MinerKeyRegistrar",
    "ProviderKey",
    "credential_fingerprint",
    "failover_reason",
    "leadpoet_keys",
    "rank_keys",
    "sealed_to",
    "submission_id_for",
]
