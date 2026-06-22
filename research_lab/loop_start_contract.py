"""Local/staged Research Lab loop-start payment and miner-key contract.

This module is intentionally inert by default. It models the Research Lab
loop-start contract with Research Lab names while preserving the useful safety
properties of the existing qualification model-submission payment path:
duplicate payment rejection, destination/amount/success/age checks,
coldkey-hotkey ownership, and retry credit preservation when queueing fails
before useful work starts.

No gateway routes, Supabase writes, validator paths, or production paid loops
are enabled here.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .canonical import sha256_json
from .schema_validation import validate_schema_record


SCHEMA_VERSION = "1.0"
FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "loop_start_contract_fixtures.json"

DEFAULT_LOOP_START_FEE_USD = Decimal("0.20")
DEFAULT_PAYMENT_MAX_AGE_SECONDS = 86_400
DEFAULT_AMOUNT_BUFFER_PERCENT = Decimal("0.01")
USD_QUANT = Decimal("0.000001")

VALID_TRANSFER_CALLS = (
    "transfer",
    "transfer_keep_alive",
    "transfer_allow_death",
    "transfer_all",
)

ALLOWED_OPENROUTER_KEY_HANDLING = ("encrypted_ref", "ephemeral_ref")

FORBIDDEN_SECRET_KEYS = (
    "api_key",
    "openrouter_api_key",
    "openrouter_key",
    "raw_key",
    "raw_secret",
    "secret",
    "token",
    "credential",
    "credential_value",
)

FORBIDDEN_SECRET_MARKERS = (
    "sk-or-",
    "openrouter_api_key",
    "raw_openrouter_key",
    "raw_secret",
)


def _decimal(value: Any) -> Decimal:
    return Decimal(str(value))


def _usd(value: Decimal) -> float:
    return float(value.quantize(USD_QUANT, rounding=ROUND_HALF_UP))


def _payment_ref(block_hash: str, extrinsic_index: int) -> str:
    return f"{block_hash}:{int(extrinsic_index)}"


@dataclass(frozen=True)
class LoopStartPolicy:
    policy_id: str = "research-loop-start-local-v0"
    contract_enabled: bool = False
    loop_start_fee_enabled: bool = True
    loop_start_fee_usd: Decimal = DEFAULT_LOOP_START_FEE_USD
    amount_buffer_percent: Decimal = DEFAULT_AMOUNT_BUFFER_PERCENT
    payment_max_age_seconds: int = DEFAULT_PAYMENT_MAX_AGE_SECONDS
    leadpoet_payment_wallet: str = "local-leadpoet-payment-wallet"
    miner_openrouter_key_required: bool = True
    miner_openrouter_key_preflight_required: bool = True
    gateway_payment_verifier_enabled: bool = False
    openrouter_key_handling_modes: tuple[str, ...] = ALLOWED_OPENROUTER_KEY_HANDLING
    leadpoet_exa_key_ref: str = "server-side:leadpoet:exa"
    leadpoet_scrapingdog_key_ref: str = "server-side:leadpoet:scrapingdog"

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "LoopStartPolicy":
        return cls(
            policy_id=str(data.get("policy_id", "research-loop-start-local-v0")),
            contract_enabled=bool(data.get("contract_enabled", False)),
            loop_start_fee_enabled=bool(data.get("loop_start_fee_enabled", True)),
            loop_start_fee_usd=_decimal(data.get("loop_start_fee_usd", DEFAULT_LOOP_START_FEE_USD)),
            amount_buffer_percent=_decimal(data.get("amount_buffer_percent", DEFAULT_AMOUNT_BUFFER_PERCENT)),
            payment_max_age_seconds=int(data.get("payment_max_age_seconds", DEFAULT_PAYMENT_MAX_AGE_SECONDS)),
            leadpoet_payment_wallet=str(data.get("leadpoet_payment_wallet", "local-leadpoet-payment-wallet")),
            miner_openrouter_key_required=bool(data.get("miner_openrouter_key_required", True)),
            miner_openrouter_key_preflight_required=bool(
                data.get("miner_openrouter_key_preflight_required", True)
            ),
            gateway_payment_verifier_enabled=bool(data.get("gateway_payment_verifier_enabled", False)),
            openrouter_key_handling_modes=tuple(
                str(item) for item in data.get("openrouter_key_handling_modes", ALLOWED_OPENROUTER_KEY_HANDLING)
            ),
            leadpoet_exa_key_ref=str(data.get("leadpoet_exa_key_ref", "server-side:leadpoet:exa")),
            leadpoet_scrapingdog_key_ref=str(
                data.get("leadpoet_scrapingdog_key_ref", "server-side:leadpoet:scrapingdog")
            ),
        )

    def hash_dict(self) -> dict[str, Any]:
        return {
            "policy_id": self.policy_id,
            "contract_enabled": self.contract_enabled,
            "loop_start_fee_enabled": self.loop_start_fee_enabled,
            "loop_start_fee_usd": str(self.loop_start_fee_usd),
            "amount_buffer_percent": str(self.amount_buffer_percent),
            "payment_max_age_seconds": self.payment_max_age_seconds,
            "leadpoet_payment_wallet": self.leadpoet_payment_wallet,
            "miner_openrouter_key_required": self.miner_openrouter_key_required,
            "miner_openrouter_key_preflight_required": self.miner_openrouter_key_preflight_required,
            "gateway_payment_verifier_enabled": self.gateway_payment_verifier_enabled,
            "openrouter_key_handling_modes": list(self.openrouter_key_handling_modes),
            "leadpoet_exa_key_ref": self.leadpoet_exa_key_ref,
            "leadpoet_scrapingdog_key_ref": self.leadpoet_scrapingdog_key_ref,
        }


@dataclass(frozen=True)
class LoopStartPaymentEvidence:
    block_hash: str
    extrinsic_index: int
    miner_hotkey: str
    sender_coldkey: str
    hotkey_owner_coldkey: str
    destination_wallet: str
    amount_usd: Decimal
    call_function: str
    extrinsic_success: bool
    age_seconds: int
    network: str = "local"

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "LoopStartPaymentEvidence":
        return cls(
            block_hash=str(data["block_hash"]),
            extrinsic_index=int(data["extrinsic_index"]),
            miner_hotkey=str(data["miner_hotkey"]),
            sender_coldkey=str(data["sender_coldkey"]),
            hotkey_owner_coldkey=str(data["hotkey_owner_coldkey"]),
            destination_wallet=str(data["destination_wallet"]),
            amount_usd=_decimal(data["amount_usd"]),
            call_function=str(data["call_function"]),
            extrinsic_success=bool(data["extrinsic_success"]),
            age_seconds=int(data["age_seconds"]),
            network=str(data.get("network", "local")),
        )

    @property
    def payment_ref(self) -> str:
        return _payment_ref(self.block_hash, self.extrinsic_index)

    def to_dict(self) -> dict[str, Any]:
        return {
            "block_hash": self.block_hash,
            "extrinsic_index": self.extrinsic_index,
            "miner_hotkey": self.miner_hotkey,
            "sender_coldkey": self.sender_coldkey,
            "hotkey_owner_coldkey": self.hotkey_owner_coldkey,
            "destination_wallet": self.destination_wallet,
            "amount_usd": _usd(self.amount_usd),
            "call_function": self.call_function,
            "extrinsic_success": self.extrinsic_success,
            "age_seconds": self.age_seconds,
            "network": self.network,
        }


@dataclass(frozen=True)
class PaymentCheck:
    status: str
    payment_ref: str
    valid: bool
    required_usd: Decimal
    amount_usd: Decimal
    reasons: tuple[str, ...]
    input_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "payment_ref": self.payment_ref,
            "valid": self.valid,
            "required_usd": _usd(self.required_usd),
            "amount_usd": _usd(self.amount_usd),
            "reasons": list(self.reasons),
            "input_hash": self.input_hash,
        }


@dataclass(frozen=True)
class MinerOpenRouterKeyReference:
    key_ref: str
    handling: str
    preflight_status: str
    provider: str = "openrouter"
    secret_material_stored: bool = False

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "MinerOpenRouterKeyReference":
        return cls(
            key_ref=str(data.get("key_ref", "")),
            handling=str(data.get("handling", "")),
            preflight_status=str(data.get("preflight_status", "not_run")),
            provider=str(data.get("provider", "openrouter")),
            secret_material_stored=bool(data.get("secret_material_stored", False)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "key_ref": self.key_ref,
            "handling": self.handling,
            "preflight_status": self.preflight_status,
            "secret_material_stored": self.secret_material_stored,
        }


@dataclass(frozen=True)
class ResearchLoopStartRequest:
    request_id: str
    miner_hotkey: str
    island: str
    brief_ref: str
    payment_block_hash: str
    payment_extrinsic_index: int
    miner_openrouter_key_ref: str
    requested_at: str
    loop_count: int = 1

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "ResearchLoopStartRequest":
        return cls(
            request_id=str(data["request_id"]),
            miner_hotkey=str(data["miner_hotkey"]),
            island=str(data["island"]),
            brief_ref=str(data["brief_ref"]),
            payment_block_hash=str(data.get("payment_block_hash", "")),
            payment_extrinsic_index=int(data.get("payment_extrinsic_index", -1)),
            miner_openrouter_key_ref=str(data.get("miner_openrouter_key_ref", "")),
            requested_at=str(data["requested_at"]),
            loop_count=int(data.get("loop_count", 1)),
        )

    @property
    def payment_ref(self) -> str:
        if not self.payment_block_hash or self.payment_extrinsic_index < 0:
            return ""
        return _payment_ref(self.payment_block_hash, self.payment_extrinsic_index)


@dataclass(frozen=True)
class LoopStartCredit:
    credit_id: str
    miner_hotkey: str
    payment_ref: str
    request_id: str
    status: str
    reason: str
    created_from_decision_id: str
    consumed_by_loop_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "credit_id": self.credit_id,
            "miner_hotkey": self.miner_hotkey,
            "payment_ref": self.payment_ref,
            "request_id": self.request_id,
            "status": self.status,
            "reason": self.reason,
            "created_from_decision_id": self.created_from_decision_id,
            "consumed_by_loop_id": self.consumed_by_loop_id,
        }


@dataclass(frozen=True)
class LoopStartDecision:
    decision_id: str
    request_id: str
    miner_hotkey: str
    island: str
    status: str
    can_queue: bool
    payment_required: bool
    payment_verified: bool
    payment_ref: str
    credit_id: str
    miner_openrouter_key_ref: str
    reasons: tuple[str, ...]
    provider_key_sources: tuple[dict[str, Any], ...]
    input_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision_id": self.decision_id,
            "request_id": self.request_id,
            "miner_hotkey": self.miner_hotkey,
            "island": self.island,
            "status": self.status,
            "can_queue": self.can_queue,
            "payment_required": self.payment_required,
            "payment_verified": self.payment_verified,
            "payment_ref": self.payment_ref,
            "credit_id": self.credit_id,
            "miner_openrouter_key_ref": self.miner_openrouter_key_ref,
            "reasons": list(self.reasons),
            "provider_key_sources": list(self.provider_key_sources),
            "input_hash": self.input_hash,
        }


def validate_loop_start_policy(policy: LoopStartPolicy | Mapping[str, Any]) -> list[str]:
    if not isinstance(policy, LoopStartPolicy):
        policy = LoopStartPolicy.from_mapping(policy)
    errors: list[str] = []
    if policy.loop_start_fee_usd <= 0:
        errors.append("loop_start_fee_usd must be positive")
    if policy.amount_buffer_percent < 0 or policy.amount_buffer_percent >= Decimal("0.10"):
        errors.append("amount_buffer_percent must be >= 0 and < 0.10")
    if policy.payment_max_age_seconds <= 0:
        errors.append("payment_max_age_seconds must be positive")
    if policy.loop_start_fee_enabled and not policy.leadpoet_payment_wallet:
        errors.append("leadpoet_payment_wallet is required when fee is enabled")
    for mode in policy.openrouter_key_handling_modes:
        if mode not in ALLOWED_OPENROUTER_KEY_HANDLING:
            errors.append(f"unsupported OpenRouter key handling mode: {mode}")
    if _contains_secret_material(policy.hash_dict()):
        errors.append("policy must not contain raw provider secret material")
    return errors


def verify_loop_start_payment_evidence(
    evidence: LoopStartPaymentEvidence | Mapping[str, Any],
    policy: LoopStartPolicy | Mapping[str, Any],
    *,
    used_payment_refs: Sequence[str] = (),
) -> PaymentCheck:
    if not isinstance(evidence, LoopStartPaymentEvidence):
        evidence = LoopStartPaymentEvidence.from_mapping(evidence)
    if not isinstance(policy, LoopStartPolicy):
        policy = LoopStartPolicy.from_mapping(policy)

    policy_errors = validate_loop_start_policy(policy)
    if policy_errors:
        raise ValueError("; ".join(policy_errors))

    input_hash = sha256_json(
        {
            "evidence": evidence.to_dict(),
            "policy": policy.hash_dict(),
            "used_payment_refs": sorted(str(ref) for ref in used_payment_refs),
        }
    )

    if not policy.loop_start_fee_enabled:
        return PaymentCheck(
            status="fee_disabled",
            payment_ref=evidence.payment_ref,
            valid=True,
            required_usd=Decimal("0"),
            amount_usd=evidence.amount_usd,
            reasons=(),
            input_hash=input_hash,
        )

    reasons: list[str] = []
    required_with_buffer = policy.loop_start_fee_usd * (Decimal("1") - policy.amount_buffer_percent)

    if evidence.extrinsic_index < 0:
        reasons.append("invalid_extrinsic_index")
    if evidence.payment_ref in set(str(ref) for ref in used_payment_refs):
        reasons.append("duplicate_payment")
    if evidence.call_function not in VALID_TRANSFER_CALLS:
        reasons.append("not_a_valid_transfer_call")
    if evidence.destination_wallet != policy.leadpoet_payment_wallet:
        reasons.append("wrong_destination_wallet")
    if evidence.sender_coldkey != evidence.hotkey_owner_coldkey:
        reasons.append("coldkey_hotkey_mismatch")
    if evidence.amount_usd < required_with_buffer:
        reasons.append("insufficient_payment")
    if not evidence.extrinsic_success:
        reasons.append("failed_extrinsic")
    if evidence.age_seconds > policy.payment_max_age_seconds:
        reasons.append("stale_payment")
    if evidence.age_seconds < -300:
        reasons.append("future_payment_timestamp")

    return PaymentCheck(
        status="rejected" if reasons else "verified",
        payment_ref=evidence.payment_ref,
        valid=not reasons,
        required_usd=policy.loop_start_fee_usd,
        amount_usd=evidence.amount_usd,
        reasons=tuple(reasons),
        input_hash=input_hash,
    )


def validate_miner_openrouter_key_reference(
    key_ref: MinerOpenRouterKeyReference | Mapping[str, Any] | None,
    policy: LoopStartPolicy | Mapping[str, Any],
) -> list[str]:
    if not isinstance(policy, LoopStartPolicy):
        policy = LoopStartPolicy.from_mapping(policy)

    if key_ref is None:
        return ["missing_miner_openrouter_key"] if policy.miner_openrouter_key_required else []
    if not isinstance(key_ref, MinerOpenRouterKeyReference):
        key_ref = MinerOpenRouterKeyReference.from_mapping(key_ref)

    errors: list[str] = []
    if policy.miner_openrouter_key_required and not key_ref.key_ref:
        errors.append("missing_miner_openrouter_key")
    if key_ref.provider != "openrouter":
        errors.append("provider_must_be_openrouter")
    if key_ref.handling not in policy.openrouter_key_handling_modes:
        errors.append("unsupported_key_handling")
    if policy.miner_openrouter_key_preflight_required and key_ref.preflight_status != "passed":
        errors.append("openrouter_key_preflight_failed")
    if key_ref.secret_material_stored:
        errors.append("raw_secret_material_not_allowed")
    if _contains_secret_material(key_ref.to_dict()):
        errors.append("raw_secret_material_not_allowed")
    return _dedupe(errors)


def evaluate_loop_start_request(
    request: ResearchLoopStartRequest | Mapping[str, Any],
    payment_evidence: LoopStartPaymentEvidence | Mapping[str, Any] | None,
    key_ref: MinerOpenRouterKeyReference | Mapping[str, Any] | None,
    policy: LoopStartPolicy | Mapping[str, Any],
    *,
    used_payment_refs: Sequence[str] = (),
    existing_credits: Sequence[LoopStartCredit | Mapping[str, Any]] = (),
) -> LoopStartDecision:
    if not isinstance(request, ResearchLoopStartRequest):
        request = ResearchLoopStartRequest.from_mapping(request)
    if not isinstance(policy, LoopStartPolicy):
        policy = LoopStartPolicy.from_mapping(policy)
    if key_ref is not None and not isinstance(key_ref, MinerOpenRouterKeyReference):
        key_ref = MinerOpenRouterKeyReference.from_mapping(key_ref)

    policy_errors = validate_loop_start_policy(policy)
    if policy_errors:
        raise ValueError("; ".join(policy_errors))

    provider_key_sources = build_provider_key_sources(key_ref, policy)
    base_hash = {
        "request": request.__dict__,
        "policy": policy.hash_dict(),
        "key_ref": key_ref.to_dict() if key_ref else None,
        "payment_ref": request.payment_ref,
        "used_payment_refs": sorted(str(ref) for ref in used_payment_refs),
    }

    if not policy.contract_enabled:
        return _decision(
            request=request,
            policy=policy,
            status="disabled",
            can_queue=False,
            payment_verified=False,
            payment_ref=request.payment_ref,
            credit_id="",
            key_ref=key_ref,
            reasons=("research_loop_paid_starts_disabled",),
            provider_key_sources=provider_key_sources,
            input_hash=sha256_json(base_hash),
        )

    key_errors = validate_miner_openrouter_key_reference(key_ref, policy)
    if key_errors:
        return _decision(
            request=request,
            policy=policy,
            status="key_rejected",
            can_queue=False,
            payment_verified=False,
            payment_ref=request.payment_ref,
            credit_id="",
            key_ref=key_ref,
            reasons=tuple(key_errors),
            provider_key_sources=provider_key_sources,
            input_hash=sha256_json(base_hash),
        )

    credit = _find_available_credit(request.miner_hotkey, existing_credits)
    if credit is not None:
        return _decision(
            request=request,
            policy=policy,
            status="credit_ready_to_queue",
            can_queue=True,
            payment_verified=False,
            payment_ref=credit.payment_ref,
            credit_id=credit.credit_id,
            key_ref=key_ref,
            reasons=(),
            provider_key_sources=provider_key_sources,
            input_hash=sha256_json({**base_hash, "credit": credit.to_dict()}),
        )

    if payment_evidence is None:
        return _decision(
            request=request,
            policy=policy,
            status="payment_rejected",
            can_queue=False,
            payment_verified=False,
            payment_ref=request.payment_ref,
            credit_id="",
            key_ref=key_ref,
            reasons=("missing_loop_start_payment",),
            provider_key_sources=provider_key_sources,
            input_hash=sha256_json(base_hash),
        )
    if not isinstance(payment_evidence, LoopStartPaymentEvidence):
        payment_evidence = LoopStartPaymentEvidence.from_mapping(payment_evidence)

    payment_check = verify_loop_start_payment_evidence(
        payment_evidence,
        policy,
        used_payment_refs=used_payment_refs,
    )
    reasons = list(payment_check.reasons)
    if request.payment_ref != payment_evidence.payment_ref:
        reasons.append("request_payment_ref_mismatch")

    if reasons:
        return _decision(
            request=request,
            policy=policy,
            status="payment_rejected",
            can_queue=False,
            payment_verified=False,
            payment_ref=payment_evidence.payment_ref,
            credit_id="",
            key_ref=key_ref,
            reasons=tuple(reasons),
            provider_key_sources=provider_key_sources,
            input_hash=sha256_json({**base_hash, "payment_check": payment_check.to_dict()}),
        )

    return _decision(
        request=request,
        policy=policy,
        status="ready_to_queue",
        can_queue=True,
        payment_verified=True,
        payment_ref=payment_evidence.payment_ref,
        credit_id="",
        key_ref=key_ref,
        reasons=(),
        provider_key_sources=provider_key_sources,
        input_hash=sha256_json({**base_hash, "payment_check": payment_check.to_dict()}),
    )


def preserve_loop_start_credit_after_queue_failure(
    decision: LoopStartDecision | Mapping[str, Any],
    *,
    failure_reason: str,
) -> tuple[LoopStartDecision, LoopStartCredit]:
    if not isinstance(decision, LoopStartDecision):
        decision = _decision_from_mapping(decision)
    if decision.status != "ready_to_queue" or not decision.payment_verified:
        raise ValueError("can only preserve credit for a verified payment before queueing")
    if not failure_reason:
        raise ValueError("failure_reason is required")

    credit = _build_credit(
        miner_hotkey=decision.miner_hotkey,
        payment_ref=decision.payment_ref,
        request_id=decision.request_id,
        reason=failure_reason,
        created_from_decision_id=decision.decision_id,
    )
    updated = LoopStartDecision(
        **{
            **decision.__dict__,
            "status": "credit_preserved_after_queue_failure",
            "can_queue": False,
            "credit_id": credit.credit_id,
            "reasons": (failure_reason,),
        }
    )
    updated = _with_decision_id(updated)
    return updated, credit


def consume_loop_start_credit(
    credit: LoopStartCredit | Mapping[str, Any],
    *,
    loop_id: str,
) -> LoopStartCredit:
    if not isinstance(credit, LoopStartCredit):
        credit = _credit_from_mapping(credit)
    if credit.status != "available":
        raise ValueError("only available loop-start credits can be consumed")
    if not loop_id:
        raise ValueError("loop_id is required")
    return LoopStartCredit(**{**credit.__dict__, "status": "consumed", "consumed_by_loop_id": loop_id})


def build_provider_key_sources(
    key_ref: MinerOpenRouterKeyReference | Mapping[str, Any] | None,
    policy: LoopStartPolicy | Mapping[str, Any],
) -> tuple[dict[str, Any], ...]:
    if not isinstance(policy, LoopStartPolicy):
        policy = LoopStartPolicy.from_mapping(policy)
    if key_ref is not None and not isinstance(key_ref, MinerOpenRouterKeyReference):
        key_ref = MinerOpenRouterKeyReference.from_mapping(key_ref)

    openrouter_ref = key_ref.key_ref if key_ref else ""
    return (
        {
            "provider": "openrouter",
            "key_source": "miner",
            "key_ref": openrouter_ref,
            "secret_material_stored": False,
            "usage": "miner-funded Research Lab LLM calls",
        },
        {
            "provider": "exa",
            "key_source": "leadpoet_server_side",
            "key_ref": policy.leadpoet_exa_key_ref,
            "secret_material_stored": False,
            "usage": "Leadpoet-hosted search infrastructure",
        },
        {
            "provider": "scrapingdog",
            "key_source": "leadpoet_server_side",
            "key_ref": policy.leadpoet_scrapingdog_key_ref,
            "secret_material_stored": False,
            "usage": "Leadpoet-hosted scrape infrastructure",
        },
    )


def build_loop_start_contract_record(
    decision: LoopStartDecision,
    payment_check: PaymentCheck | None = None,
    key_ref: MinerOpenRouterKeyReference | None = None,
    preserved_credit: LoopStartCredit | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "decision": decision.to_dict(),
        "payment_check": payment_check.to_dict() if payment_check else None,
        "miner_openrouter_key_ref": key_ref.to_dict() if key_ref else None,
        "provider_key_sources": list(decision.provider_key_sources),
        "preserved_credit": preserved_credit.to_dict() if preserved_credit else None,
    }


def validate_loop_start_contract_record(record: Mapping[str, Any]) -> list[str]:
    errors = validate_schema_record("research_loop_start_contract.schema.json", dict(record))
    if _contains_secret_material(record):
        errors.append("record contains raw provider secret material")
    return errors


async def verify_research_loop_start_payment_with_gateway_primitive(
    *,
    block_hash: str,
    extrinsic_index: int,
    miner_hotkey: str,
    policy: LoopStartPolicy | Mapping[str, Any],
) -> PaymentCheck:
    """Staged wrapper around the existing qualification payment verifier.

    The wrapper is disabled unless `gateway_payment_verifier_enabled` is true.
    That keeps Work Package 2A from accidentally activating chain/Supabase
    behavior. Future gateway integration should replace the duplicate check
    table with Research Lab-native loop-start payment storage.
    """

    if not isinstance(policy, LoopStartPolicy):
        policy = LoopStartPolicy.from_mapping(policy)
    if not policy.gateway_payment_verifier_enabled:
        return PaymentCheck(
            status="staged_gateway_verifier_disabled",
            payment_ref=_payment_ref(block_hash, extrinsic_index),
            valid=False,
            required_usd=policy.loop_start_fee_usd,
            amount_usd=Decimal("0"),
            reasons=("gateway_payment_verifier_disabled",),
            input_hash=sha256_json(
                {
                    "block_hash": block_hash,
                    "extrinsic_index": extrinsic_index,
                    "miner_hotkey": miner_hotkey,
                    "policy": policy.hash_dict(),
                }
            ),
        )

    import importlib

    payment_module = importlib.import_module("gateway.qualification.api.payment")
    verify_payment = payment_module.verify_payment

    valid, error = await verify_payment(
        block_hash=block_hash,
        extrinsic_index=extrinsic_index,
        miner_hotkey=miner_hotkey,
        required_usd=float(policy.loop_start_fee_usd),
    )
    return PaymentCheck(
        status="verified" if valid else "rejected",
        payment_ref=_payment_ref(block_hash, extrinsic_index),
        valid=valid,
        required_usd=policy.loop_start_fee_usd,
        amount_usd=Decimal("0"),
        reasons=() if valid else (str(error or "payment_verification_failed"),),
        input_hash=sha256_json(
            {
                "block_hash": block_hash,
                "extrinsic_index": extrinsic_index,
                "miner_hotkey": miner_hotkey,
                "policy": policy.hash_dict(),
                "gateway_wrapper_result": valid,
            }
        ),
    )


def verify_research_lab_loop_start_contract(fixture_path: Path | str = FIXTURE_PATH) -> dict[str, Any]:
    fixture = _load_fixture(Path(fixture_path))
    policy = LoopStartPolicy.from_mapping(fixture["policy"])
    disabled_policy = LoopStartPolicy.from_mapping(fixture["disabled_policy"])
    request = ResearchLoopStartRequest.from_mapping(fixture["request"])
    key_ref = MinerOpenRouterKeyReference.from_mapping(fixture["key_refs"]["valid"])
    evidence = LoopStartPaymentEvidence.from_mapping(fixture["payments"]["valid"])

    _assert(not validate_loop_start_policy(policy), "policy validates")

    disabled_decision = evaluate_loop_start_request(request, evidence, key_ref, disabled_policy)
    _assert(disabled_decision.status == "disabled", "paid loop starts disabled by default")
    _assert(not disabled_decision.can_queue, "disabled contract cannot queue")

    payment_check = verify_loop_start_payment_evidence(evidence, policy, used_payment_refs=())
    _assert(payment_check.valid and payment_check.status == "verified", "valid payment evidence verifies")

    decision = evaluate_loop_start_request(request, evidence, key_ref, policy)
    _assert(decision.status == "ready_to_queue", "valid request is ready to queue in staged mode")
    _assert(decision.can_queue and decision.payment_verified, "valid request can queue with verified payment")
    _assert(_provider_key_source(decision, "openrouter") == "miner", "OpenRouter routes to miner key")
    _assert(_provider_key_source(decision, "exa") == "leadpoet_server_side", "Exa routes to Leadpoet key")
    _assert(
        _provider_key_source(decision, "scrapingdog") == "leadpoet_server_side",
        "ScrapingDog routes to Leadpoet key",
    )

    record = build_loop_start_contract_record(decision, payment_check, key_ref)
    _assert(not validate_loop_start_contract_record(record), "valid contract record schema validates")
    _assert(
        record == build_loop_start_contract_record(
            evaluate_loop_start_request(request, evidence, key_ref, policy),
            verify_loop_start_payment_evidence(evidence, policy),
            key_ref,
        ),
        "contract record is deterministic",
    )

    failed_decision, credit = preserve_loop_start_credit_after_queue_failure(
        decision,
        failure_reason="queue_create_failed_before_work",
    )
    _assert(failed_decision.status == "credit_preserved_after_queue_failure", "queue failure preserves credit")
    _assert(credit.status == "available", "preserved credit is available")
    credit_record = build_loop_start_contract_record(failed_decision, payment_check, key_ref, credit)
    _assert(not validate_loop_start_contract_record(credit_record), "credit preservation record validates")
    consumed = consume_loop_start_credit(credit, loop_id="research-loop-local-001")
    _assert(consumed.status == "consumed", "available credit can be consumed later")

    credit_decision = evaluate_loop_start_request(
        request,
        None,
        key_ref,
        policy,
        existing_credits=(credit,),
    )
    _assert(credit_decision.status == "credit_ready_to_queue", "existing credit can start retry without payment")
    _assert(credit_decision.payment_ref == credit.payment_ref, "credit retains original payment ref")

    for case_name, expected_reason in fixture["expectations"]["payment_rejections"].items():
        rejected = verify_loop_start_payment_evidence(
            fixture["payments"][case_name],
            policy,
            used_payment_refs=fixture["used_payment_refs"] if case_name == "duplicate" else (),
        )
        _assert(not rejected.valid, f"{case_name} payment is rejected")
        _assert(expected_reason in rejected.reasons, f"{case_name} includes {expected_reason}")

    for case_name, expected_reason in fixture["expectations"]["key_rejections"].items():
        candidate_key = MinerOpenRouterKeyReference.from_mapping(fixture["key_refs"][case_name])
        key_decision = evaluate_loop_start_request(request, evidence, candidate_key, policy)
        _assert(key_decision.status == "key_rejected", f"{case_name} key is rejected")
        _assert(expected_reason in key_decision.reasons, f"{case_name} includes {expected_reason}")

    gateway_disabled_check = _run_gateway_disabled_check_fixture(policy)
    _assert(
        gateway_disabled_check.status == "staged_gateway_verifier_disabled",
        "gateway wrapper is disabled by default",
    )

    return {
        "loop_start_fee_usd": _usd(policy.loop_start_fee_usd),
        "valid_payment_ref": payment_check.payment_ref,
        "decision_status": decision.status,
        "provider_routes": {
            "openrouter": _provider_key_source(decision, "openrouter"),
            "exa": _provider_key_source(decision, "exa"),
            "scrapingdog": _provider_key_source(decision, "scrapingdog"),
        },
        "payment_rejection_cases": len(fixture["expectations"]["payment_rejections"]),
        "key_rejection_cases": len(fixture["expectations"]["key_rejections"]),
        "preserved_credit_status": credit.status,
        "gateway_wrapper_status": gateway_disabled_check.status,
    }


def _decision(
    *,
    request: ResearchLoopStartRequest,
    policy: LoopStartPolicy,
    status: str,
    can_queue: bool,
    payment_verified: bool,
    payment_ref: str,
    credit_id: str,
    key_ref: MinerOpenRouterKeyReference | None,
    reasons: Sequence[str],
    provider_key_sources: Sequence[Mapping[str, Any]],
    input_hash: str,
) -> LoopStartDecision:
    decision = LoopStartDecision(
        decision_id="",
        request_id=request.request_id,
        miner_hotkey=request.miner_hotkey,
        island=request.island,
        status=status,
        can_queue=can_queue,
        payment_required=policy.loop_start_fee_enabled,
        payment_verified=payment_verified,
        payment_ref=payment_ref,
        credit_id=credit_id,
        miner_openrouter_key_ref=key_ref.key_ref if key_ref else "",
        reasons=tuple(reasons),
        provider_key_sources=tuple(dict(item) for item in provider_key_sources),
        input_hash=input_hash,
    )
    return _with_decision_id(decision)


def _with_decision_id(decision: LoopStartDecision) -> LoopStartDecision:
    data = decision.to_dict()
    data.pop("decision_id", None)
    return LoopStartDecision(**{**decision.__dict__, "decision_id": "loop_start_decision:" + sha256_json(data)})


def _build_credit(
    *,
    miner_hotkey: str,
    payment_ref: str,
    request_id: str,
    reason: str,
    created_from_decision_id: str,
) -> LoopStartCredit:
    base = {
        "miner_hotkey": miner_hotkey,
        "payment_ref": payment_ref,
        "request_id": request_id,
        "status": "available",
        "reason": reason,
        "created_from_decision_id": created_from_decision_id,
    }
    return LoopStartCredit(credit_id="loop_start_credit:" + sha256_json(base), **base)


def _find_available_credit(
    miner_hotkey: str,
    credits: Sequence[LoopStartCredit | Mapping[str, Any]],
) -> LoopStartCredit | None:
    for credit in credits:
        if not isinstance(credit, LoopStartCredit):
            credit = _credit_from_mapping(credit)
        if credit.miner_hotkey == miner_hotkey and credit.status == "available":
            return credit
    return None


def _decision_from_mapping(data: Mapping[str, Any]) -> LoopStartDecision:
    return LoopStartDecision(
        decision_id=str(data["decision_id"]),
        request_id=str(data["request_id"]),
        miner_hotkey=str(data["miner_hotkey"]),
        island=str(data["island"]),
        status=str(data["status"]),
        can_queue=bool(data["can_queue"]),
        payment_required=bool(data["payment_required"]),
        payment_verified=bool(data["payment_verified"]),
        payment_ref=str(data.get("payment_ref", "")),
        credit_id=str(data.get("credit_id", "")),
        miner_openrouter_key_ref=str(data.get("miner_openrouter_key_ref", "")),
        reasons=tuple(str(item) for item in data.get("reasons", [])),
        provider_key_sources=tuple(dict(item) for item in data.get("provider_key_sources", [])),
        input_hash=str(data["input_hash"]),
    )


def _credit_from_mapping(data: Mapping[str, Any]) -> LoopStartCredit:
    return LoopStartCredit(
        credit_id=str(data["credit_id"]),
        miner_hotkey=str(data["miner_hotkey"]),
        payment_ref=str(data["payment_ref"]),
        request_id=str(data["request_id"]),
        status=str(data["status"]),
        reason=str(data["reason"]),
        created_from_decision_id=str(data["created_from_decision_id"]),
        consumed_by_loop_id=str(data.get("consumed_by_loop_id", "")),
    )


def _contains_secret_material(value: Any) -> bool:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            key_lower = str(key).lower()
            if key_lower in FORBIDDEN_SECRET_KEYS or key_lower.startswith("raw_") or key_lower.endswith("_api_key"):
                return True
            if _contains_secret_material(nested):
                return True
    elif isinstance(value, (list, tuple)):
        return any(_contains_secret_material(item) for item in value)
    elif isinstance(value, str):
        lowered = value.lower()
        return any(marker in lowered for marker in FORBIDDEN_SECRET_MARKERS)
    return False


def _dedupe(values: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            out.append(value)
    return out


def _provider_key_source(decision: LoopStartDecision, provider: str) -> str:
    for source in decision.provider_key_sources:
        if source.get("provider") == provider:
            return str(source.get("key_source", ""))
    return ""


def _run_gateway_disabled_check_fixture(policy: LoopStartPolicy) -> PaymentCheck:
    return PaymentCheck(
        status="staged_gateway_verifier_disabled",
        payment_ref=_payment_ref("0xlocal", 0),
        valid=False,
        required_usd=policy.loop_start_fee_usd,
        amount_usd=Decimal("0"),
        reasons=("gateway_payment_verifier_disabled",),
        input_hash=sha256_json({"gateway_payment_verifier_enabled": False}),
    )


def _load_fixture(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _assert(condition: bool, label: str) -> None:
    if not condition:
        raise AssertionError(label)
