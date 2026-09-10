"""Accepted Arena weight state and independent chain-outcome documents."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Mapping

from lab_arena import contracts, signing
from leadpoet_canonical.arena_weights import (
    ACCEPTED_WEIGHT_STATE_SCHEMA_VERSION,
    WEIGHT_STATE_SIGNATURE_PREFIX,
    validate_accepted_weight_state,
)
from leadpoet_canonical.lab_arena_rewards import sha256_json

CHAIN_OUTCOME_SCHEMA_VERSION = "leadpoet.arena.chain_outcome.v1"


def build_accepted_weight_state(
    signer: signing.ArenaSigner,
    *,
    network: str,
    genesis_hash: str,
    netuid: int,
    epoch: int,
    valid_from_block: int,
    valid_until_block: int,
    reward_basis: Mapping[str, Any],
    fixed_allocations: Any,
    fulfillment_demands: Any,
    burn_hotkey: str,
    issued_at: str,
) -> Dict[str, Any]:
    """Build and sign the small immutable state consumed by normal validators."""

    body = {
        "schema_version": ACCEPTED_WEIGHT_STATE_SCHEMA_VERSION,
        "network": str(network),
        "genesis_hash": str(genesis_hash).lower().removeprefix("0x"),
        "netuid": int(netuid),
        "epoch": int(epoch),
        "valid_from_block": int(valid_from_block),
        "valid_until_block": int(valid_until_block),
        "reward_basis": dict(reward_basis),
        "fixed_allocations": list(fixed_allocations),
        "fulfillment_demands": list(fulfillment_demands),
        "burn_hotkey": str(burn_hotkey),
        "issued_at": str(issued_at),
    }
    state_hash = sha256_json(body)
    signature = signer.sign((WEIGHT_STATE_SIGNATURE_PREFIX + state_hash).encode("utf-8"))
    import base64

    state = {
        **body,
        "state_hash": state_hash,
        "signature": {
            "algorithm": signer.algorithm,
            "public_key_hash": signer.public_key_hash,
            "signature_b64": base64.b64encode(signature).decode("ascii"),
        },
    }
    return validate_accepted_weight_state(state)


def validate_chain_outcome(value: Any, *, now: datetime, max_age_seconds: int = 300) -> Dict[str, Any]:
    """Validate a validator report. The report is evidence, not chain finality."""

    required = {
        "schema_version", "network", "netuid", "epoch", "validator_hotkey",
        "state_hash", "weights_hash", "extrinsic_hash", "finalized_block_hash",
        "finalized_block_number", "observed_at", "request_id", "signature",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise contracts.ArenaContractError("chain outcome fields are invalid")
    document = dict(value)
    if document["schema_version"] != CHAIN_OUTCOME_SCHEMA_VERSION:
        raise contracts.ArenaContractError("chain outcome schema is invalid")
    contracts.require_hotkey(document["validator_hotkey"], "validator_hotkey")
    for field in ("netuid", "epoch", "finalized_block_number"):
        number = document[field]
        if isinstance(number, bool) or not isinstance(number, int) or number < 0:
            raise contracts.ArenaContractError("%s is invalid" % field)
    for field in ("state_hash", "request_id"):
        contracts.require_sha256(document[field], field)
    weights_hash = str(document["weights_hash"]).lower()
    if len(weights_hash) != 64 or any(c not in "0123456789abcdef" for c in weights_hash):
        raise contracts.ArenaContractError("weights_hash is invalid")
    document["weights_hash"] = weights_hash
    for field in ("extrinsic_hash", "finalized_block_hash"):
        text = str(document[field]).lower()
        if len(text) != 66 or not text.startswith("0x") or any(c not in "0123456789abcdef" for c in text[2:]):
            raise contracts.ArenaContractError("%s is invalid" % field)
        document[field] = text
    try:
        observed = datetime.strptime(str(document["observed_at"]), "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except ValueError as exc:
        raise contracts.ArenaContractError("observed_at is invalid") from exc
    future_seconds = (observed - now.astimezone(timezone.utc)).total_seconds()
    if future_seconds > int(max_age_seconds):
        raise contracts.ArenaContractError("chain outcome is from the future")
    core = {key: document[key] for key in required - {"request_id", "signature"}}
    if contracts.document_hash(core) != document["request_id"]:
        raise contracts.ArenaContractError("request_id does not match chain outcome")
    if not isinstance(document["signature"], str) or not document["signature"]:
        raise contracts.ArenaContractError("chain outcome signature is invalid")
    return document


def chain_outcome_message(document: Mapping[str, Any]) -> str:
    return "arena_chain_outcome:" + str(document["request_id"])
