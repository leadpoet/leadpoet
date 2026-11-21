"""Provider key registry, ranking and failover rules for lead orders.

Miner keys are stored with the Arena credential encryption
(``lab_arena.credentials``): the same KMS key and validation probes, with a
``lead-order:`` submission id in the encryption context so a lead-order
ciphertext can never be decrypted as an Arena submission's key or the
reverse. Leadpoet's own keys come from the environment and are always last.
"""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence

from lab_arena import broker as arena_broker
from lab_arena import credentials as arena_credentials
from lab_arena.credentials import CredentialError, CredentialManager

PROVIDERS = ("openrouter", "deepline")
LEADPOET_KEY_PREFIX = "leadpoet-"
SUBMISSION_ID_PREFIX = "lead-order:"

FAILOVER_UNAUTHORIZED = "unauthorized"
FAILOVER_PAYMENT_REQUIRED = "payment_required"
FAILOVER_FORBIDDEN = "forbidden"
FAILOVER_RATE_LIMITED = "rate_limited_no_retry"
FAILOVER_SPENT_LIMIT = "spent_limit"

_KEY_ID_RE = re.compile(r"^[a-z0-9-]{1,64}$")


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
    """Validate and encrypt one miner key per provider with the Arena manager."""

    def __init__(self, manager: CredentialManager, *, new_key_id: Callable[[], str] = lambda: str(uuid.uuid4())) -> None:
        self._manager = manager
        self._new_key_id = new_key_id

    def register(self, *, miner_hotkey: str, provider: str, credentials: Mapping[str, Any]) -> Dict[str, Any]:
        """Return the row to store. Plaintext never leaves this call."""

        if provider not in PROVIDERS:
            raise LeadOrderKeyError("provider_invalid")
        key_id = self._new_key_id()
        submission_id = submission_id_for(key_id)
        # Reuse the Arena probes. OpenRouter needs the runtime key and its
        # management key (admission-only proof, never stored); Deepline needs
        # only its workspace key.
        try:
            if provider == "openrouter":
                runtime = arena_broker_secret(credentials, "openrouter_api_key")
                management = arena_broker_secret(credentials, "openrouter_management_key")
                self._manager._validate_openrouter(runtime, management)
                secret = runtime
            else:
                secret = arena_broker_secret(credentials, "deepline_api_key")
                self._manager._validate_deepline(secret)
            ciphertext = self._manager._encrypt(
                secret, submission_id=submission_id, miner_hotkey=miner_hotkey, provider=provider
            )
        except CredentialError as exc:
            raise LeadOrderKeyError(exc.code, retryable=exc.retryable) from None
        return {
            "key_id": key_id,
            "provider": provider,
            "miner_hotkey": miner_hotkey,
            "submission_id": submission_id,
            "ciphertext_b64": ciphertext,
            "internal_fingerprint": credential_fingerprint(secret),
        }

    def decrypt(self, key: ProviderKey) -> str:
        if key.is_leadpoet or key.ciphertext_b64 is None or key.miner_hotkey is None:
            raise LeadOrderKeyError("miner_credentials_unavailable")
        try:
            return self._manager.runtime_key(
                {
                    "provider": key.provider,
                    "submission_id": submission_id_for(key.key_id),
                    "miner_hotkey": key.miner_hotkey,
                    "ciphertext_b64": key.ciphertext_b64,
                },
                key.provider,
            )
        except CredentialError as exc:
            raise LeadOrderKeyError(exc.code, retryable=exc.retryable) from None


def arena_broker_secret(credentials: Mapping[str, Any], field: str) -> str:
    """Shape-check one submitted secret with the Arena validators."""

    if not isinstance(credentials, Mapping) or set(credentials) - {
        "openrouter_api_key", "openrouter_management_key", "deepline_api_key"
    }:
        raise CredentialError("submission_credentials_invalid")
    code = "%s_invalid" % field
    value = credentials.get(field)
    if field.startswith("openrouter_"):
        return arena_credentials._openrouter_key(value, code)
    return arena_credentials._bounded_secret(value, code)


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
    "submission_id_for",
]
