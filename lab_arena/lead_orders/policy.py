"""The signed, versioned lead-order policy record.

One record holds every number that decides money. It is validated when it is
loaded, and a record that lets a fake order on a miner's own key make money is
refused. The record is signed with the Arena signing helpers
(``lab_arena.signing``), so receipts can name the exact version they settled
under and anyone can verify it without AWS.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Dict, Mapping, Union

from lab_arena import signing
from lab_arena.contracts import ArenaContractError, document_hash

POLICY_SCHEMA_VERSION = "leadpoet.lead_orders.policy.v1"
POLICY_HASH_FIELD = "policy_hash"
PROVIDERS = ("openrouter", "deepline")

_DECIMAL_FIELDS = (
    "price_per_lead_usd",
    "miner_premium",
    "min_charge_ratio",
    "spend_ceiling_per_lead_usd",
    "lane_emission_cap",
    "deepline_reserve_per_call_usd",
)
_INT_FIELDS = (
    "version",
    "min_leads_per_order",
    "ranking_window_epochs",
    "daily_leads_per_hotkey",
    "order_deadline_minutes",
    "openrouter_max_output_tokens",
)
_BODY_FIELDS = (
    ("schema_version",) + _DECIMAL_FIELDS + _INT_FIELDS + ("max_provider_discounts", "miner_terms_version")
)
_TERMS_VERSION_RE = re.compile(r"^[A-Za-z0-9._-]{1,32}$")

# Start values from docs/gamma-lead-orders.md. Discounts are Gavin's Nov 19
# figures: Deepline credit discounts reach 10%, OpenRouter gives none.
DEFAULT_POLICY_BODY: Dict[str, Any] = {
    "schema_version": POLICY_SCHEMA_VERSION,
    "version": 1,
    "price_per_lead_usd": "0.20",
    "miner_premium": "0.25",
    "min_leads_per_order": 10,
    "min_charge_ratio": "0.40",
    "spend_ceiling_per_lead_usd": "0.50",
    "lane_emission_cap": "0.30",
    "ranking_window_epochs": 140,
    "daily_leads_per_hotkey": 500,
    "order_deadline_minutes": 60,
    "max_provider_discounts": {"deepline": "0.10", "openrouter": "0"},
    "deepline_reserve_per_call_usd": "0.50",
    "openrouter_max_output_tokens": 16384,
    # The miner data terms a key's owner must have signed for the key to be
    # registered or ranked (docs/gamma-lead-orders-miner-terms.md).
    "miner_terms_version": "miner-terms-placeholder-1",
}

MICROUSD = Decimal(1_000_000)


class PolicyError(ArenaContractError):
    """The policy record is malformed, unsafe, unsigned or signed by another key."""


def _decimal(value: Any, name: str) -> Decimal:
    # Decimal strings only: a JSON float would make money rounding depend on
    # binary representation.
    if not isinstance(value, str):
        raise PolicyError("%s must be a decimal string" % name)
    try:
        parsed = Decimal(value)
    except InvalidOperation as exc:
        raise PolicyError("%s is not a decimal" % name) from exc
    if not parsed.is_finite() or parsed < 0:
        raise PolicyError("%s must be a finite non-negative decimal" % name)
    # Every money field must be exact in micro-USD / parts per million.
    if (parsed * MICROUSD) != (parsed * MICROUSD).to_integral_value():
        raise PolicyError("%s has more than 6 decimal places" % name)
    return parsed


def _int(value: Any, name: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise PolicyError("%s must be an integer between %d and %d" % (name, minimum, maximum))
    return int(value)


@dataclass(frozen=True)
class LeadOrderPolicy:
    version: int
    price_per_lead_usd: Decimal
    miner_premium: Decimal
    min_leads_per_order: int
    min_charge_ratio: Decimal
    spend_ceiling_per_lead_usd: Decimal
    lane_emission_cap: Decimal
    ranking_window_epochs: int
    daily_leads_per_hotkey: int
    order_deadline_minutes: int
    max_provider_discounts: Mapping[str, Decimal]
    deepline_reserve_per_call_usd: Decimal
    openrouter_max_output_tokens: int
    miner_terms_version: str
    policy_hash: str

    # -- micro-USD views used by the money module and the SQL functions ------

    @property
    def price_per_lead_microusd(self) -> int:
        return int(self.price_per_lead_usd * MICROUSD)

    @property
    def spend_ceiling_per_lead_microusd(self) -> int:
        return int(self.spend_ceiling_per_lead_usd * MICROUSD)

    @property
    def min_charge_ratio_ppm(self) -> int:
        return int(self.min_charge_ratio * MICROUSD)

    @property
    def miner_premium_ppm(self) -> int:
        return int(self.miner_premium * MICROUSD)

    @property
    def lane_emission_cap_ppm(self) -> int:
        return int(self.lane_emission_cap * MICROUSD)

    @property
    def deepline_reserve_per_call_microusd(self) -> int:
        return int(self.deepline_reserve_per_call_usd * MICROUSD)

    @property
    def max_provider_discount(self) -> Decimal:
        return max(self.max_provider_discounts.values())


def validate_policy_body(body: Mapping[str, Any]) -> Dict[str, Any]:
    """Validate the unsigned body and return it as a plain dict."""

    if not isinstance(body, Mapping):
        raise PolicyError("policy must be an object")
    if set(body) != set(_BODY_FIELDS):
        raise PolicyError("policy fields are invalid")
    if body.get("schema_version") != POLICY_SCHEMA_VERSION:
        raise PolicyError("policy schema is unsupported")
    decimals = {name: _decimal(body[name], name) for name in _DECIMAL_FIELDS}
    ints = {
        "version": _int(body["version"], "version", 1, 1 << 31),
        "min_leads_per_order": _int(body["min_leads_per_order"], "min_leads_per_order", 1, 10_000),
        "ranking_window_epochs": _int(body["ranking_window_epochs"], "ranking_window_epochs", 1, 100_000),
        "daily_leads_per_hotkey": _int(body["daily_leads_per_hotkey"], "daily_leads_per_hotkey", 1, 1_000_000),
        "order_deadline_minutes": _int(body["order_deadline_minutes"], "order_deadline_minutes", 1, 1440),
        "openrouter_max_output_tokens": _int(body["openrouter_max_output_tokens"], "openrouter_max_output_tokens", 1, 1_000_000),
    }
    discounts = body.get("max_provider_discounts")
    if not isinstance(discounts, Mapping) or set(discounts) != set(PROVIDERS):
        raise PolicyError("max_provider_discounts must name exactly %s" % ", ".join(PROVIDERS))
    parsed_discounts = {name: _decimal(discounts[name], "max_provider_discounts.%s" % name) for name in PROVIDERS}
    terms = body.get("miner_terms_version")
    if not isinstance(terms, str) or not _TERMS_VERSION_RE.fullmatch(terms):
        raise PolicyError("miner_terms_version must be 1-32 of A-Z a-z 0-9 . _ -")

    if decimals["price_per_lead_usd"] <= 0:
        raise PolicyError("price_per_lead_usd must be positive")
    if decimals["spend_ceiling_per_lead_usd"] <= 0:
        raise PolicyError("spend_ceiling_per_lead_usd must be positive")
    if decimals["deepline_reserve_per_call_usd"] <= 0:
        raise PolicyError("deepline_reserve_per_call_usd must be positive")
    for name in ("min_charge_ratio", "lane_emission_cap"):
        if not Decimal(0) < decimals[name] <= Decimal(1):
            raise PolicyError("%s must be in (0, 1]" % name)
    if any(value >= 1 for value in parsed_discounts.values()):
        raise PolicyError("a provider discount must be below 1")
    if ints["daily_leads_per_hotkey"] < ints["min_leads_per_order"]:
        raise PolicyError("daily_leads_per_hotkey is below the minimum order")

    # Anti-farming invariant 1: on a fake order against their own key a miner
    # receives spend x (1 + premium), pays spend x (1 - discount) to the
    # provider and is charged at least min_charge_ratio x spend. The order
    # loses money only while the ratio beats premium + the largest discount.
    if not decimals["min_charge_ratio"] > decimals["miner_premium"] + max(parsed_discounts.values()):
        raise PolicyError("min_charge_ratio must exceed miner_premium + max provider discount")
    # Anti-farming invariant 2: within the spend ceiling the minimum charge
    # must fit inside the hold, or the hold cap would undercut invariant 1.
    if decimals["min_charge_ratio"] * decimals["spend_ceiling_per_lead_usd"] > decimals["price_per_lead_usd"]:
        raise PolicyError("min_charge_ratio x spend_ceiling_per_lead_usd must not exceed price_per_lead_usd")

    out = dict(body)
    out["max_provider_discounts"] = dict(discounts)
    out.update(ints)
    return out


def build_policy_document(body: Mapping[str, Any]) -> Dict[str, Any]:
    """Validated body plus its ``policy_hash``; ready for signing."""

    validated = validate_policy_body(body)
    return dict(validated, **{POLICY_HASH_FIELD: document_hash(validated)})


def sign_policy(signer: signing.ArenaSigner, body: Mapping[str, Any]) -> Dict[str, Any]:
    return signing.sign_document(signer, build_policy_document(body), hash_field=POLICY_HASH_FIELD)


def _from_validated(document: Mapping[str, Any]) -> LeadOrderPolicy:
    return LeadOrderPolicy(
        version=int(document["version"]),
        price_per_lead_usd=Decimal(document["price_per_lead_usd"]),
        miner_premium=Decimal(document["miner_premium"]),
        min_leads_per_order=int(document["min_leads_per_order"]),
        min_charge_ratio=Decimal(document["min_charge_ratio"]),
        spend_ceiling_per_lead_usd=Decimal(document["spend_ceiling_per_lead_usd"]),
        lane_emission_cap=Decimal(document["lane_emission_cap"]),
        ranking_window_epochs=int(document["ranking_window_epochs"]),
        daily_leads_per_hotkey=int(document["daily_leads_per_hotkey"]),
        order_deadline_minutes=int(document["order_deadline_minutes"]),
        max_provider_discounts={k: Decimal(v) for k, v in document["max_provider_discounts"].items()},
        deepline_reserve_per_call_usd=Decimal(document["deepline_reserve_per_call_usd"]),
        openrouter_max_output_tokens=int(document["openrouter_max_output_tokens"]),
        miner_terms_version=str(document["miner_terms_version"]),
        policy_hash=str(document[POLICY_HASH_FIELD]),
    )


def load_policy(
    document: Mapping[str, Any],
    *,
    public_key_der: bytes,
    expected_public_key_hash: str,
) -> LeadOrderPolicy:
    """Verify the signature, re-validate every rule, and return the policy."""

    if not isinstance(document, Mapping):
        raise PolicyError("policy must be an object")
    body = {k: v for k, v in document.items() if k not in (POLICY_HASH_FIELD, signing.SIGNATURE_FIELD)}
    validate_policy_body(body)
    try:
        signing.verify_document_signature(
            document,
            hash_field=POLICY_HASH_FIELD,
            public_key_der=public_key_der,
            expected_public_key_hash=expected_public_key_hash,
        )
    except ArenaContractError as exc:
        raise PolicyError("policy signature is invalid: %s" % exc) from exc
    return _from_validated(document)


def load_policy_file(
    path: Union[str, Path],
    *,
    public_key_der: bytes,
    expected_public_key_hash: str,
) -> LeadOrderPolicy:
    try:
        document = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise PolicyError("policy file is unreadable") from exc
    return load_policy(document, public_key_der=public_key_der, expected_public_key_hash=expected_public_key_hash)


def unsigned_policy(body: Mapping[str, Any] = DEFAULT_POLICY_BODY) -> LeadOrderPolicy:
    """A validated policy without a signature, for pure arithmetic and tests.

    Services never use this: they load a signed record with ``load_policy``.
    """

    return _from_validated(build_policy_document(body))


__all__ = [
    "DEFAULT_POLICY_BODY",
    "LeadOrderPolicy",
    "POLICY_HASH_FIELD",
    "POLICY_SCHEMA_VERSION",
    "PolicyError",
    "build_policy_document",
    "load_policy",
    "load_policy_file",
    "sign_policy",
    "unsigned_policy",
    "validate_policy_body",
]
