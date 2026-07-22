"""Canonical V2 authority for enclave-computed validator weights.

The existing weight formula remains in ``weight_computation``. This module
only commits its immutable inputs to a complete V2 receipt graph and verifies
that the validator enclave signed the result it computed. It is deliberately
Python 3.7 compatible and performs no gateway, database, AWS, or chain I/O.
"""

from __future__ import annotations

import re
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

from leadpoet_canonical.attested_v2 import (
    COORDINATOR_ROLE,
    EMPTY_HOST_OPERATION_ROOT,
    EMPTY_TRANSPORT_ROOT,
    WEIGHT_ROLE,
    AttestedV2Error,
    merkle_root,
    sha256_json,
    validate_receipt_graph,
)
from leadpoet_canonical.hotkey_authority_v2 import (
    HotkeyAuthorityV2Error,
    build_application_signature_request_v2,
    validate_weight_extrinsic_authorization_v2,
)
from leadpoet_canonical.chain_source_v2 import (
    CHAIN_ARCHIVE_ENDPOINT_HOST,
    CHAIN_ENDPOINT_HOST,
)
from leadpoet_canonical.weight_computation import compute_final_weights


WEIGHT_SNAPSHOT_V2_SCHEMA_VERSION = "leadpoet.weight_snapshot.v2"
PUBLISHED_WEIGHT_BUNDLE_V2_SCHEMA_VERSION = "leadpoet.published_weight_bundle.v2"
WEIGHT_INPUT_VALUE_SCHEMA_VERSION = "leadpoet.weight_input_value.v2"
WEIGHT_FINALIZATION_V2_SCHEMA_VERSION = "leadpoet.weight_finalization.v2"
WEIGHT_FINALIZATION_SUBMISSION_V2_SCHEMA_VERSION = (
    "leadpoet.weight_finalization_submission.v2"
)

WEIGHT_INPUT_PURPOSES = {
    "research_lab_allocation": (COORDINATOR_ROLE, "research_lab.allocation.v2"),
    "champions": (COORDINATOR_ROLE, "research_lab.champion_input.v2"),
    "reimbursements": (COORDINATOR_ROLE, "research_lab.reimbursement_input.v2"),
    "source_add_rewards": (COORDINATOR_ROLE, "research_lab.source_add_reward_input.v2"),
    "fulfillment_rewards": (COORDINATOR_ROLE, "research_lab.fulfillment_input.v2"),
    "leaderboard": (COORDINATOR_ROLE, "research_lab.leaderboard_input.v2"),
    "bans": (COORDINATOR_ROLE, "research_lab.ban_input.v2"),
    "sourcing_history": (COORDINATOR_ROLE, "research_lab.sourcing_input.v2"),
    "anomaly_adjustments": (
        COORDINATOR_ROLE,
        "research_lab.anomaly_adjustment_input.v2",
    ),
    "chain_state": (WEIGHT_ROLE, "validator.chain_state.v2"),
    "metagraph_state": (WEIGHT_ROLE, "validator.metagraph_state.v2"),
    "burn_ownership": (WEIGHT_ROLE, "validator.burn_ownership.v2"),
    "feature_flags": (WEIGHT_ROLE, "validator.feature_flags.v2"),
    "constants": (WEIGHT_ROLE, "validator.constants.v2"),
}

SUPABASE_WEIGHT_SOURCE_HOST = "qplwoislplkcegvdmbim.supabase.co"
GATEWAY_TLS_WEIGHT_INPUTS = frozenset(
    {
        "research_lab_allocation",
        "champions",
        "reimbursements",
        "source_add_rewards",
        "fulfillment_rewards",
        "leaderboard",
        "bans",
        "sourcing_history",
    }
)
VALIDATOR_CHAIN_WEIGHT_INPUTS = frozenset(
    {"chain_state", "metagraph_state"}
)
VALIDATOR_DERIVED_WEIGHT_INPUTS = frozenset(
    {"burn_ownership", "feature_flags", "constants"}
)
MEASURED_WEIGHT_INPUTS = frozenset(
    {"anomaly_adjustments"}
)
VALIDATOR_WEIGHT_INPUT_CATEGORIES = frozenset(
    category
    for category, (role, _purpose) in WEIGHT_INPUT_PURPOSES.items()
    if role == WEIGHT_ROLE
)
GATEWAY_WEIGHT_INPUT_CATEGORIES = frozenset(WEIGHT_INPUT_PURPOSES) - (
    VALIDATOR_WEIGHT_INPUT_CATEGORIES
)

_HASH_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_RAW_HASH_RE = re.compile(r"^[0-9a-f]{64}$")
_SIGNATURE_RE = re.compile(r"^[0-9a-f]{128}$")

_SNAPSHOT_BODY_FIELDS = {
    "schema_version",
    "validator_hotkey",
    "netuid",
    "epoch_id",
    "block",
    "finalized_chain_state_root",
    "gateway_authority_event_hash",
    "calculation_snapshot",
    "calculation_snapshot_hash",
    "input_receipt_hashes",
    "source_input_root",
}
_SNAPSHOT_FIELDS = _SNAPSHOT_BODY_FIELDS | {"snapshot_hash"}
_BUNDLE_FIELDS = {
    "schema_version",
    "validator_hotkey",
    "binding_message",
    "validator_hotkey_signature",
    "weight_snapshot",
    "weight_result",
    "weights_signature",
    "receipt_graph",
}


class WeightAuthorityV2Error(ValueError):
    """A V2 weight authority object is incomplete, inconsistent, or tampered."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise WeightAuthorityV2Error(message)


def _terminal_receipt_hashes(
    receipts: Mapping[str, Mapping[str, Any]],
) -> list[str]:
    parent_hashes = {
        str(parent_hash)
        for receipt in receipts.values()
        for parent_hash in receipt.get("parent_receipt_hashes") or ()
    }
    return sorted(set(receipts) - parent_hashes)


def _hash(value: Any, field: str) -> str:
    normalized = str(value or "").strip().lower()
    _require(bool(_HASH_RE.fullmatch(normalized)), "%s must be sha256:<64 lowercase hex>" % field)
    return normalized


def _validator_hotkey(value: Any) -> str:
    normalized = str(value or "").strip()
    _require(1 <= len(normalized) <= 128, "validator_hotkey is invalid")
    _require(not any(character.isspace() for character in normalized), "validator_hotkey is invalid")
    return normalized


def _int(value: Any, field: str) -> int:
    _require(isinstance(value, int) and not isinstance(value, bool), "%s must be an integer" % field)
    _require(value >= 0, "%s must be non-negative" % field)
    return value


def _normalized_input_receipts(value: Mapping[str, Any]) -> Dict[str, str]:
    _require(isinstance(value, Mapping), "input_receipt_hashes must be an object")
    _require(set(value) == set(WEIGHT_INPUT_PURPOSES), "weight input receipt categories are incomplete")
    normalized = {
        category: _hash(value[category], "%s receipt hash" % category)
        for category in sorted(WEIGHT_INPUT_PURPOSES)
    }
    _require(
        len(set(normalized.values())) == len(normalized),
        "one receipt cannot satisfy multiple weight input categories",
    )
    return normalized


def _weight_input_value_documents_v2(
    *,
    calculation_snapshot: Mapping[str, Any],
    finalized_chain_state_root: Optional[str],
    gateway_authority_event_hash: str,
) -> Dict[str, Dict[str, Any]]:
    """Return the exact value each semantic input receipt must commit.

    Receipt role and purpose labels are not sufficient: an otherwise valid
    receipt could describe unrelated data. These documents bind every value
    consumed by the unchanged final-weight formula to exactly one required
    input category. Some values intentionally appear in more than one document
    where two independent trust claims overlap (for example, burn ownership
    and metagraph membership).
    """

    _require(
        isinstance(calculation_snapshot, Mapping),
        "calculation_snapshot must be an object",
    )
    calculation = dict(calculation_snapshot)
    compute_final_weights(calculation)
    allocation = calculation.get("research_lab_allocation_doc")
    _require(
        isinstance(allocation, Mapping),
        "research_lab_allocation_doc must be an object",
    )
    metagraph_hotkeys = list(calculation.get("metagraph_hotkeys") or [])
    burn_uid = int(calculation.get("burn_target_uid"))
    actual_burn_hotkey = (
        metagraph_hotkeys[burn_uid]
        if 0 <= burn_uid < len(metagraph_hotkeys)
        else None
    )

    def document(category: str, value: Mapping[str, Any]) -> Dict[str, Any]:
        result = {
            "schema_version": WEIGHT_INPUT_VALUE_SCHEMA_VERSION,
            "category": category,
            "netuid": int(calculation["netuid"]),
            "epoch_id": int(calculation["epoch_id"]),
            "value": dict(value),
        }
        if category in VALIDATOR_WEIGHT_INPUT_CATEGORIES:
            result["block"] = int(calculation["block"])
        return result

    post_adjustment_values = {
        "research_lab_allocation_doc": dict(allocation),
        "fulfillment_rows": list(calculation["fulfillment_rows"]),
        "leaderboard_entries": list(calculation["leaderboard_entries"]),
        "banned_hotkeys": list(calculation["banned_hotkeys"]),
        "rolling_scores": list(calculation["rolling_scores"]),
    }
    documents = {
        "research_lab_allocation": document(
            "research_lab_allocation",
            {
                "allocation_doc": dict(allocation),
                "gateway_authority_event_hash": _hash(
                    gateway_authority_event_hash,
                    "gateway_authority_event_hash",
                ),
            },
        ),
        "champions": document(
            "champions",
            {
                "champion_share": calculation["champion_share"],
                "champion_uid": calculation["champion_uid"],
                "effective_champion_share": calculation[
                    "effective_champion_share"
                ],
                "champion_allocations": list(
                    allocation.get("champion_allocations") or []
                ),
                "queued_champion_allocations": list(
                    allocation.get("queued_champion_allocations") or []
                ),
            },
        ),
        "reimbursements": document(
            "reimbursements",
            {
                "reimbursement_allocations": list(
                    allocation.get("reimbursement_allocations") or []
                )
            },
        ),
        "source_add_rewards": document(
            "source_add_rewards",
            {
                "source_add_allocations": list(
                    allocation.get("source_add_allocations") or []
                )
            },
        ),
        "fulfillment_rewards": document(
            "fulfillment_rewards",
            {
                "fulfillment_share": calculation["fulfillment_share"],
                "fulfillment_rows": list(calculation["fulfillment_rows"]),
                "fulfillment_fetch_ok": calculation["fulfillment_fetch_ok"],
            },
        ),
        "leaderboard": document(
            "leaderboard",
            {
                "leaderboard_bonus_share": calculation[
                    "leaderboard_bonus_share"
                ],
                "leaderboard_rank_shares": list(
                    calculation["leaderboard_rank_shares"]
                ),
                "leaderboard_entries": list(calculation["leaderboard_entries"]),
                "leaderboard_fetch_ok": calculation["leaderboard_fetch_ok"],
            },
        ),
        "bans": document(
            "bans",
            {
                "banned_hotkeys": list(calculation["banned_hotkeys"]),
                "banned_lookup_ok": calculation["banned_lookup_ok"],
            },
        ),
        "sourcing_history": document(
            "sourcing_history",
            {
                "rolling_lead_count": calculation["rolling_lead_count"],
                "rolling_scores": list(calculation["rolling_scores"]),
            },
        ),
        "anomaly_adjustments": document(
            "anomaly_adjustments",
            {
                "post_adjustment_values_hash": sha256_json(
                    post_adjustment_values
                )
            },
        ),
    }
    if finalized_chain_state_root is None:
        return documents
    documents.update({
        "chain_state": document(
            "chain_state",
            {
                "finalized_chain_state_root": _hash(
                    finalized_chain_state_root,
                    "finalized_chain_state_root",
                ),
                "netuid": calculation["netuid"],
                "epoch_id": calculation["epoch_id"],
                "block": calculation["block"],
            },
        ),
        "metagraph_state": document(
            "metagraph_state",
            {"metagraph_hotkeys": metagraph_hotkeys},
        ),
        "burn_ownership": document(
            "burn_ownership",
            {
                "burn_target_uid": burn_uid,
                "expected_burn_target_hotkey": calculation[
                    "expected_burn_target_hotkey"
                ],
                "actual_burn_target_hotkey": actual_burn_hotkey,
            },
        ),
        "feature_flags": document(
            "feature_flags",
            {"ff_enabled": calculation["ff_enabled"]},
        ),
        "constants": document(
            "constants",
            {
                "commit_sha": calculation["commit_sha"],
                "config_hash": calculation["config_hash"],
                "base_burn_share": calculation["base_burn_share"],
                "champion_share": calculation["champion_share"],
                "research_lab_fallback_share": calculation[
                    "research_lab_fallback_share"
                ],
                "leaderboard_bonus_share": calculation[
                    "leaderboard_bonus_share"
                ],
                "leaderboard_rank_shares": list(
                    calculation["leaderboard_rank_shares"]
                ),
                "sourcing_floor_threshold": calculation[
                    "sourcing_floor_threshold"
                ],
                "min_total_rep_for_distribution": calculation[
                    "min_total_rep_for_distribution"
                ],
            },
        ),
    })
    return documents


def gateway_weight_input_value_documents_v2(
    *,
    calculation_snapshot: Mapping[str, Any],
    gateway_authority_event_hash: str,
) -> Dict[str, Dict[str, Any]]:
    """Return epoch-scoped gateway inputs without a host-selected chain block."""

    documents = _weight_input_value_documents_v2(
        calculation_snapshot=calculation_snapshot,
        finalized_chain_state_root=None,
        gateway_authority_event_hash=gateway_authority_event_hash,
    )
    _require(
        set(documents) == set(GATEWAY_WEIGHT_INPUT_CATEGORIES),
        "gateway weight input value categories are incomplete",
    )
    return documents


def weight_input_value_documents_v2(
    *,
    calculation_snapshot: Mapping[str, Any],
    finalized_chain_state_root: str,
    gateway_authority_event_hash: str,
) -> Dict[str, Dict[str, Any]]:
    """Return all gateway and validator input documents for a finalized block."""

    documents = _weight_input_value_documents_v2(
        calculation_snapshot=calculation_snapshot,
        finalized_chain_state_root=finalized_chain_state_root,
        gateway_authority_event_hash=gateway_authority_event_hash,
    )
    _require(
        set(documents) == set(WEIGHT_INPUT_PURPOSES),
        "weight input value categories are incomplete",
    )
    return documents


def weight_input_output_roots_v2(
    *,
    calculation_snapshot: Mapping[str, Any],
    finalized_chain_state_root: str,
    gateway_authority_event_hash: str,
) -> Dict[str, str]:
    documents = weight_input_value_documents_v2(
        calculation_snapshot=calculation_snapshot,
        finalized_chain_state_root=finalized_chain_state_root,
        gateway_authority_event_hash=gateway_authority_event_hash,
    )
    _require(
        set(documents) == set(WEIGHT_INPUT_PURPOSES),
        "weight input value categories are incomplete",
    )
    return {
        category: sha256_json(documents[category])
        for category in sorted(documents)
    }


def validate_weight_input_source_evidence_v2(
    *,
    category: str,
    receipt: Mapping[str, Any],
    document: Mapping[str, Any],
    transport_attempts: Sequence[Mapping[str, Any]],
) -> None:
    """Require evidence from the real source, not a host-supplied value.

    The measured producer is responsible for parsing the authenticated response
    into ``document``. This validator binds that output to every encrypted
    request/response artifact and rejects empty transport for external sources.
    """

    normalized_category = str(category or "")
    _require(
        normalized_category in WEIGHT_INPUT_PURPOSES,
        "weight input source category is invalid",
    )
    _require(
        receipt.get("host_operation_root") == EMPTY_HOST_OPERATION_ROOT,
        "%s input cannot depend on a host operation" % normalized_category,
    )
    scoped_attempts = [
        dict(attempt)
        for attempt in transport_attempts
        if attempt.get("job_id") == receipt.get("job_id")
        and attempt.get("purpose") == receipt.get("purpose")
    ]
    value_hash = sha256_json(document["value"])
    artifact_hashes = [value_hash]

    if normalized_category in GATEWAY_TLS_WEIGHT_INPUTS:
        _require(scoped_attempts, "%s input has no authenticated database read" % normalized_category)
        for attempt in scoped_attempts:
            _require(
                attempt.get("provider_id") == "supabase",
                "%s input used an unauthorized source" % normalized_category,
            )
            _require(
                attempt.get("destination_host") == SUPABASE_WEIGHT_SOURCE_HOST,
                "%s input used the wrong Supabase project" % normalized_category,
            )
            _require(
                attempt.get("method") in {"GET", "POST"},
                "%s input used an invalid database method" % normalized_category,
            )
            artifact_hashes.append(str(attempt["request_artifact_hash"]))
            if attempt.get("terminal_status") == "authenticated_response":
                artifact_hashes.append(str(attempt["response_artifact_hash"]))
        _require(
            any(
                attempt.get("terminal_status") == "authenticated_response"
                and 200 <= int(attempt.get("http_status") or 0) < 300
                for attempt in scoped_attempts
            ),
            "%s input has no successful authenticated database response"
            % normalized_category,
        )
    elif normalized_category in VALIDATOR_CHAIN_WEIGHT_INPUTS:
        _require(scoped_attempts, "%s input has no authenticated chain read" % normalized_category)
        for attempt in scoped_attempts:
            _require(
                (
                    attempt.get("provider_id") == "bittensor_chain"
                    and attempt.get("destination_host") == CHAIN_ENDPOINT_HOST
                )
                or (
                    attempt.get("provider_id") == "bittensor_archive"
                    and attempt.get("destination_host")
                    == CHAIN_ARCHIVE_ENDPOINT_HOST
                ),
                "%s input used an unauthorized chain source" % normalized_category,
            )
            _require(
                attempt.get("method") in {"WSS", "POST"},
                "%s input used an invalid chain method" % normalized_category,
            )
            artifact_hashes.append(str(attempt["request_artifact_hash"]))
            if attempt.get("terminal_status") == "authenticated_response":
                artifact_hashes.append(str(attempt["response_artifact_hash"]))
        _require(
            any(
                attempt.get("terminal_status") == "authenticated_response"
                and 200 <= int(attempt.get("http_status") or 0) < 300
                for attempt in scoped_attempts
            ),
            "%s input has no successful authenticated chain response"
            % normalized_category,
        )
    elif normalized_category in VALIDATOR_DERIVED_WEIGHT_INPUTS:
        _require(
            not scoped_attempts,
            "%s derived validator input unexpectedly used external transport"
            % normalized_category,
        )
    else:
        _require(
            normalized_category in MEASURED_WEIGHT_INPUTS,
            "%s input has no source-evidence policy" % normalized_category,
        )
        _require(
            not scoped_attempts,
            "%s measured input unexpectedly used external transport"
            % normalized_category,
        )

    expected_transport_root = (
        merkle_root(
            [str(attempt["attempt_hash"]) for attempt in scoped_attempts],
            domain="leadpoet-transport-v2",
        )
        if scoped_attempts
        else EMPTY_TRANSPORT_ROOT
    )
    _require(
        receipt.get("transport_root") == expected_transport_root,
        "%s input transport root is invalid" % normalized_category,
    )
    _require(
        receipt.get("artifact_root")
        == merkle_root(artifact_hashes, domain="leadpoet-artifact-v2"),
        "%s input artifact evidence is incomplete" % normalized_category,
    )


def weight_source_input_root(
    *,
    input_receipt_hashes: Mapping[str, Any],
    finalized_chain_state_root: str,
    gateway_authority_event_hash: str,
) -> str:
    return sha256_json(
        {
            "input_receipt_hashes": _normalized_input_receipts(input_receipt_hashes),
            "finalized_chain_state_root": _hash(
                finalized_chain_state_root, "finalized_chain_state_root"
            ),
            "gateway_authority_event_hash": _hash(
                gateway_authority_event_hash, "gateway_authority_event_hash"
            ),
        }
    )


def build_weight_snapshot_v2(
    *,
    validator_hotkey: str,
    calculation_snapshot: Mapping[str, Any],
    input_receipt_hashes: Mapping[str, Any],
    finalized_chain_state_root: str,
    gateway_authority_event_hash: str,
) -> Dict[str, Any]:
    _require(isinstance(calculation_snapshot, Mapping), "calculation_snapshot must be an object")
    normalized_calculation = dict(calculation_snapshot)
    # Canonical validation is delegated to the unchanged production formula.
    compute_final_weights(normalized_calculation)
    normalized_inputs = _normalized_input_receipts(input_receipt_hashes)
    expected_parents = sorted(normalized_inputs.values())
    _require(
        normalized_calculation.get("parent_receipt_hashes") == expected_parents,
        "calculation snapshot parents differ from complete V2 weight inputs",
    )
    _require(
        normalized_calculation.get("research_lab_allocation_receipt_hash")
        == normalized_inputs["research_lab_allocation"],
        "calculation snapshot allocation receipt differs from V2 input",
    )
    source_root = weight_source_input_root(
        input_receipt_hashes=normalized_inputs,
        finalized_chain_state_root=finalized_chain_state_root,
        gateway_authority_event_hash=gateway_authority_event_hash,
    )
    body = {
        "schema_version": WEIGHT_SNAPSHOT_V2_SCHEMA_VERSION,
        "validator_hotkey": _validator_hotkey(validator_hotkey),
        "netuid": _int(normalized_calculation.get("netuid"), "netuid"),
        "epoch_id": _int(normalized_calculation.get("epoch_id"), "epoch_id"),
        "block": _int(normalized_calculation.get("block"), "block"),
        "finalized_chain_state_root": _hash(
            finalized_chain_state_root, "finalized_chain_state_root"
        ),
        "gateway_authority_event_hash": _hash(
            gateway_authority_event_hash, "gateway_authority_event_hash"
        ),
        "calculation_snapshot": normalized_calculation,
        "calculation_snapshot_hash": sha256_json(normalized_calculation),
        "input_receipt_hashes": normalized_inputs,
        "source_input_root": source_root,
    }
    return {**body, "snapshot_hash": sha256_json(body)}


def validate_weight_snapshot_v2(snapshot: Mapping[str, Any]) -> Dict[str, Any]:
    _require(isinstance(snapshot, Mapping), "weight_snapshot must be an object")
    _require(set(snapshot) == _SNAPSHOT_FIELDS, "weight_snapshot fields do not match V2 schema")
    rebuilt = build_weight_snapshot_v2(
        validator_hotkey=snapshot["validator_hotkey"],
        calculation_snapshot=snapshot["calculation_snapshot"],
        input_receipt_hashes=snapshot["input_receipt_hashes"],
        finalized_chain_state_root=snapshot["finalized_chain_state_root"],
        gateway_authority_event_hash=snapshot["gateway_authority_event_hash"],
    )
    _require(dict(snapshot) == rebuilt, "weight_snapshot is not canonical")
    return compute_final_weights(snapshot["calculation_snapshot"])


def _verify_weight_signature(pubkey: str, signature: str, weights_hash: str) -> None:
    _require(bool(_RAW_HASH_RE.fullmatch(str(weights_hash or ""))), "weights_hash is invalid")
    normalized_signature = str(signature or "").strip().lower()
    _require(bool(_SIGNATURE_RE.fullmatch(normalized_signature)), "weights_signature is invalid")
    try:
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

        Ed25519PublicKey.from_public_bytes(bytes.fromhex(str(pubkey))).verify(
            bytes.fromhex(normalized_signature),
            bytes.fromhex(str(weights_hash)),
        )
    except Exception as exc:
        raise WeightAuthorityV2Error("invalid enclave-computed weight signature") from exc


def validate_published_weight_bundle_v2(
    bundle: Mapping[str, Any],
    *,
    boot_attestation_verifier: Optional[
        Callable[[Mapping[str, Any]], Any]
    ] = None,
    require_boot_attestation_verification: bool = False,
) -> Dict[str, Any]:
    _require(isinstance(bundle, Mapping), "published weight bundle must be an object")
    _require(set(bundle) == _BUNDLE_FIELDS, "published weight bundle fields do not match V2 schema")
    _require(
        bundle.get("schema_version") == PUBLISHED_WEIGHT_BUNDLE_V2_SCHEMA_VERSION,
        "unsupported published weight bundle schema",
    )
    snapshot = bundle.get("weight_snapshot")
    claimed_result = bundle.get("weight_result")
    graph = bundle.get("receipt_graph")
    _require(isinstance(snapshot, Mapping), "weight_snapshot is missing")
    _require(isinstance(claimed_result, Mapping), "weight_result is missing")
    _require(isinstance(graph, Mapping), "receipt_graph is missing")
    computed_result = validate_weight_snapshot_v2(snapshot)
    _require(dict(claimed_result) == computed_result, "weight_result differs from canonical computation")
    validator_hotkey = _validator_hotkey(bundle.get("validator_hotkey"))
    _require(
        validator_hotkey == snapshot.get("validator_hotkey"),
        "bundle validator_hotkey differs from immutable snapshot",
    )
    binding_message = str(bundle.get("binding_message") or "")
    _require(1 <= len(binding_message) <= 4096, "binding_message is invalid")
    hotkey_signature = str(bundle.get("validator_hotkey_signature") or "").strip().lower()
    _require(bool(_SIGNATURE_RE.fullmatch(hotkey_signature)), "validator_hotkey_signature is invalid")

    required_purposes = {
        purpose for _, purpose in WEIGHT_INPUT_PURPOSES.values()
    } | {
        "validator.weight_snapshot.v2",
        "validator.weights.computed.v2",
        "validator.hotkey_signature.v2",
    }
    try:
        validate_receipt_graph(
            graph,
            required_purposes=required_purposes,
            boot_attestation_verifier=boot_attestation_verifier,
            require_boot_attestation_verification=(
                require_boot_attestation_verification
            ),
        )
    except AttestedV2Error as exc:
        raise WeightAuthorityV2Error("invalid V2 receipt graph: %s" % exc) from exc
    receipts = {
        str(receipt["receipt_hash"]): receipt
        for receipt in graph.get("receipts", [])
    }
    root_hash = str(graph.get("root_receipt_hash") or "")
    root_receipt = receipts.get(root_hash)
    _require(root_receipt is not None, "weight receipt graph root is missing")
    _require(root_receipt.get("role") == WEIGHT_ROLE, "weight graph root role is invalid")
    _require(
        root_receipt.get("purpose") == "validator.hotkey_signature.v2",
        "weight graph root must be the enclave hotkey binding receipt",
    )

    inputs = _normalized_input_receipts(snapshot["input_receipt_hashes"])
    input_documents = weight_input_value_documents_v2(
        calculation_snapshot=snapshot["calculation_snapshot"],
        finalized_chain_state_root=snapshot["finalized_chain_state_root"],
        gateway_authority_event_hash=snapshot["gateway_authority_event_hash"],
    )
    expected_input_roots = {
        category: sha256_json(document)
        for category, document in input_documents.items()
    }
    transport_attempts = list(graph.get("transport_attempts") or [])
    for category, receipt_hash in inputs.items():
        receipt = receipts.get(receipt_hash)
        _require(receipt is not None, "weight input receipt %s is missing" % category)
        expected_role, expected_purpose = WEIGHT_INPUT_PURPOSES[category]
        _require(receipt.get("role") == expected_role, "%s receipt role is invalid" % category)
        _require(
            receipt.get("purpose") == expected_purpose,
            "%s receipt purpose is invalid" % category,
        )
        _require(
            int(receipt.get("epoch_id", -1)) == int(snapshot["epoch_id"]),
            "%s receipt epoch differs from weight epoch" % category,
        )
        _require(
            receipt.get("output_root") == expected_input_roots[category],
            "%s receipt output does not bind its weight input" % category,
        )
        validate_weight_input_source_evidence_v2(
            category=category,
            receipt=receipt,
            document=input_documents[category],
            transport_attempts=transport_attempts,
        )
    _require(
        receipts[inputs["metagraph_state"]].get("parent_receipt_hashes")
        == [inputs["chain_state"]],
        "metagraph receipt must directly bind finalized chain state",
    )
    _require(
        receipts[inputs["burn_ownership"]].get("parent_receipt_hashes")
        == [inputs["metagraph_state"]],
        "burn ownership receipt must directly bind metagraph state",
    )
    for category in ("feature_flags", "constants"):
        _require(
            receipts[inputs[category]].get("parent_receipt_hashes") == [],
            "%s receipt cannot inherit host-provided ancestry" % category,
        )

    root_parents = list(root_receipt.get("parent_receipt_hashes") or [])
    _require(
        len(root_parents) == 1,
        "hotkey binding receipt must have one computed-weight parent",
    )
    computed_receipt = receipts.get(root_parents[0])
    _require(
        isinstance(computed_receipt, Mapping)
        and computed_receipt.get("role") == WEIGHT_ROLE
        and computed_receipt.get("purpose") == "validator.weights.computed.v2",
        "hotkey binding receipt parent is not computed weights",
    )
    computed_parents = list(
        computed_receipt.get("parent_receipt_hashes") or []
    )
    _require(
        len(computed_parents) == 1,
        "computed weight receipt must have one snapshot parent",
    )
    snapshot_receipt = receipts.get(computed_parents[0])
    _require(
        isinstance(snapshot_receipt, Mapping)
        and snapshot_receipt.get("role") == WEIGHT_ROLE
        and snapshot_receipt.get("purpose") == "validator.weight_snapshot.v2",
        "computed weight receipt parent is not a weight snapshot",
    )
    _require(
        snapshot_receipt.get("boot_identity_hash")
        == computed_receipt.get("boot_identity_hash")
        == root_receipt.get("boot_identity_hash"),
        "current weight receipts use different validator boots",
    )
    current_receipt_hashes = {
        str(root_receipt["receipt_hash"]),
        str(computed_receipt["receipt_hash"]),
        str(snapshot_receipt["receipt_hash"]),
    }
    pre_snapshot_receipts = {
        receipt_hash: receipt
        for receipt_hash, receipt in receipts.items()
        if receipt_hash not in current_receipt_hashes
    }
    _require(
        snapshot_receipt.get("parent_receipt_hashes")
        == sorted(
            set(inputs.values())
            | set(_terminal_receipt_hashes(pre_snapshot_receipts))
        ),
        "snapshot receipt parents differ from the complete input proof frontier",
    )
    _require(
        snapshot_receipt.get("input_root") == snapshot["source_input_root"],
        "snapshot receipt input root mismatch",
    )
    _require(
        snapshot_receipt.get("output_root") == snapshot["snapshot_hash"],
        "snapshot receipt output root mismatch",
    )
    _require(
        int(snapshot_receipt.get("epoch_id", -1)) == int(snapshot["epoch_id"]),
        "snapshot receipt epoch mismatch",
    )
    _require(
        computed_receipt.get("parent_receipt_hashes")
        == [snapshot_receipt["receipt_hash"]],
        "computed weight receipt must directly bind the snapshot receipt",
    )
    _require(
        computed_receipt.get("input_root") == snapshot["snapshot_hash"],
        "computed weight receipt input mismatch",
    )
    _require(
        computed_receipt.get("output_root") == sha256_json(computed_result),
        "computed weight receipt output mismatch",
    )
    _require(
        int(computed_receipt.get("epoch_id", -1)) == int(snapshot["epoch_id"]),
        "computed weight receipt epoch mismatch",
    )
    _require(
        computed_receipt.get("commit_sha")
        == snapshot["calculation_snapshot"].get("commit_sha"),
        "computed weight receipt commit mismatch",
    )
    try:
        application_request = build_application_signature_request_v2(
            message=binding_message.encode("utf-8"),
            validator_hotkey=validator_hotkey,
            boot_identity_hash=str(root_receipt["boot_identity_hash"]),
        )
    except HotkeyAuthorityV2Error as exc:
        raise WeightAuthorityV2Error(
            "hotkey binding receipt input is invalid"
        ) from exc
    expected_binding_output = {
        "schema_version": "leadpoet.application_signature_result.v2",
        "request_hash": application_request["request_hash"],
        "purpose": "validator.gateway_binding.v2",
        "validator_hotkey": validator_hotkey,
        "signature": hotkey_signature,
    }
    _require(
        root_receipt.get("input_root") == application_request["request_hash"],
        "hotkey binding receipt input mismatch",
    )
    _require(
        root_receipt.get("output_root") == sha256_json(expected_binding_output),
        "hotkey binding receipt output mismatch",
    )
    _require(
        int(root_receipt.get("epoch_id", -1)) == int(snapshot["epoch_id"]),
        "hotkey binding receipt epoch mismatch",
    )
    # Receipt config_hash is the measured validator boot configuration and is
    # already bound to the boot identity by validate_receipt_graph(). The
    # calculation config_hash is independently validated by
    # compute_final_weights(); they deliberately describe different domains.
    _verify_weight_signature(
        str(computed_receipt.get("enclave_pubkey") or ""),
        str(bundle.get("weights_signature") or ""),
        str(computed_result.get("weights_hash") or ""),
    )
    return {
        "bundle_hash": sha256_json(dict(bundle)),
        "root_receipt_hash": root_hash,
        "weight_receipt_hash": str(computed_receipt["receipt_hash"]),
        "snapshot_receipt_hash": str(snapshot_receipt["receipt_hash"]),
        "snapshot_hash": str(snapshot["snapshot_hash"]),
        "validator_hotkey": validator_hotkey,
        "binding_message": binding_message,
        "validator_hotkey_signature": hotkey_signature,
        "netuid": int(computed_result["netuid"]),
        "epoch_id": int(computed_result["epoch_id"]),
        "block": int(computed_result["block"]),
        "uids": list(computed_result["sparse_uids"]),
        "weights_u16": list(computed_result["sparse_weights_u16"]),
        "weights_hash": str(computed_result["weights_hash"]),
        "validator_enclave_pubkey": str(computed_receipt["enclave_pubkey"]),
        "validator_boot_identity_hash": str(
            computed_receipt["boot_identity_hash"]
        ),
    }


def build_published_weight_bundle_v2(
    *,
    validator_hotkey: str,
    binding_message: str,
    validator_hotkey_signature: str,
    weight_snapshot: Mapping[str, Any],
    weight_result: Mapping[str, Any],
    weights_signature: str,
    receipt_graph: Mapping[str, Any],
) -> Dict[str, Any]:
    bundle = {
        "schema_version": PUBLISHED_WEIGHT_BUNDLE_V2_SCHEMA_VERSION,
        "validator_hotkey": str(validator_hotkey),
        "binding_message": str(binding_message),
        "validator_hotkey_signature": str(validator_hotkey_signature),
        "weight_snapshot": dict(weight_snapshot),
        "weight_result": dict(weight_result),
        "weights_signature": str(weights_signature),
        "receipt_graph": dict(receipt_graph),
    }
    validate_published_weight_bundle_v2(bundle)
    return bundle


def required_weight_input_purposes() -> Tuple[str, ...]:
    return tuple(sorted(purpose for _, purpose in WEIGHT_INPUT_PURPOSES.values()))


def validate_weight_finalization_submission_v2(
    value: Mapping[str, Any],
    *,
    chain_signing_profile: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Validate finalized inclusion of one exact enclave-authorized extrinsic."""

    _require(isinstance(value, Mapping), "weight finalization submission is invalid")
    _require(
        set(value)
        == {
            "schema_version",
            "validator_hotkey",
            "weight_submission_event_hash",
            "finalization",
            "receipt_graph",
        },
        "weight finalization submission fields are invalid",
    )
    _require(
        value.get("schema_version")
        == WEIGHT_FINALIZATION_SUBMISSION_V2_SCHEMA_VERSION,
        "weight finalization submission schema is invalid",
    )
    hotkey = _validator_hotkey(value.get("validator_hotkey"))
    event_hash = _hash(
        value.get("weight_submission_event_hash"),
        "weight_submission_event_hash",
    )
    finalization = value.get("finalization")
    _require(isinstance(finalization, Mapping), "weight finalization is invalid")
    expected_finalization_fields = {
        "schema_version",
        "validator_hotkey",
        "netuid",
        "epoch_id",
        "weights_hash",
        "weight_receipt_hash",
        "weight_submission_event_hash",
        "extrinsic_authorization",
        "extrinsic_authorization_hash",
        "extrinsic_signature",
        "extrinsic_receipt_hash",
        "extrinsic_hash",
        "finalized_block",
        "finalized_block_hash",
        "state_transition_hash",
    }
    _require(
        set(finalization) == expected_finalization_fields,
        "weight finalization fields are invalid",
    )
    _require(
        finalization.get("schema_version") == WEIGHT_FINALIZATION_V2_SCHEMA_VERSION,
        "weight finalization schema is invalid",
    )
    _require(
        _validator_hotkey(finalization.get("validator_hotkey")) == hotkey,
        "weight finalization hotkey differs",
    )
    _require(
        _hash(
            finalization.get("weight_submission_event_hash"),
            "weight finalization event hash",
        )
        == event_hash,
        "weight finalization event differs",
    )
    netuid = _int(finalization.get("netuid"), "netuid")
    epoch_id = _int(finalization.get("epoch_id"), "epoch_id")
    finalized_block = _int(finalization.get("finalized_block"), "finalized_block")
    weights_hash = str(finalization.get("weights_hash") or "").lower()
    _require(bool(_RAW_HASH_RE.fullmatch(weights_hash)), "weights_hash is invalid")
    weight_receipt_hash = _hash(
        finalization.get("weight_receipt_hash"), "weight_receipt_hash"
    )
    extrinsic_authorization_hash = _hash(
        finalization.get("extrinsic_authorization_hash"),
        "extrinsic_authorization_hash",
    )
    extrinsic_receipt_hash = _hash(
        finalization.get("extrinsic_receipt_hash"), "extrinsic_receipt_hash"
    )
    signature = str(finalization.get("extrinsic_signature") or "").lower()
    _require(bool(_SIGNATURE_RE.fullmatch(signature)), "extrinsic signature is invalid")
    extrinsic_hash = str(finalization.get("extrinsic_hash") or "").lower()
    finalized_block_hash = str(finalization.get("finalized_block_hash") or "").lower()
    state_transition_hash = _hash(
        finalization.get("state_transition_hash"), "state_transition_hash"
    )
    _require(
        bool(re.fullmatch(r"0x[0-9a-f]{64}", extrinsic_hash)),
        "extrinsic_hash is invalid",
    )
    _require(
        bool(_RAW_HASH_RE.fullmatch(finalized_block_hash)),
        "finalized_block_hash is invalid",
    )
    authorization = finalization.get("extrinsic_authorization")
    _require(isinstance(authorization, Mapping), "extrinsic authorization is invalid")
    _require(
        sha256_json(
            {
                key: authorization[key]
                for key in authorization
                if key != "authorization_hash"
            }
        )
        == extrinsic_authorization_hash,
        "extrinsic authorization hash differs",
    )
    if chain_signing_profile is not None:
        validate_weight_extrinsic_authorization_v2(
            authorization, profile=chain_signing_profile
        )
    for field, expected in (
        ("validator_hotkey", hotkey),
        ("netuid", netuid),
        ("epoch_id", epoch_id),
        ("weights_hash", weights_hash),
        ("weight_receipt_hash", weight_receipt_hash),
        ("weight_submission_event_hash", event_hash),
        ("authorization_hash", extrinsic_authorization_hash),
    ):
        _require(authorization.get(field) == expected, "extrinsic authorization differs at %s" % field)

    graph = value.get("receipt_graph")
    _require(isinstance(graph, Mapping), "weight finalization graph is invalid")
    validate_receipt_graph(
        graph,
        required_purposes={
            "validator.weights.computed.v2",
            "validator.set_weights_extrinsic.v2",
            "validator.weights.finalized.v2",
        },
    )
    receipt_by_hash = {
        str(receipt["receipt_hash"]): receipt for receipt in graph["receipts"]
    }
    root_hash = str(graph["root_receipt_hash"])
    root = receipt_by_hash.get(root_hash)
    extrinsic_receipt = receipt_by_hash.get(extrinsic_receipt_hash)
    computed_receipt = receipt_by_hash.get(weight_receipt_hash)
    _require(
        isinstance(root, Mapping)
        and root.get("role") == WEIGHT_ROLE
        and root.get("purpose") == "validator.weights.finalized.v2"
        and int(root.get("epoch_id", -1)) == epoch_id
        and root.get("output_root") == sha256_json(dict(finalization)),
        "weight finalization root receipt is invalid",
    )
    _require(
        isinstance(computed_receipt, Mapping)
        and computed_receipt.get("purpose") == "validator.weights.computed.v2",
        "weight computed receipt is absent from finalization graph",
    )
    expected_extrinsic_output = {
        "schema_version": "leadpoet.weight_extrinsic_signature.v2",
        "authorization_hash": extrinsic_authorization_hash,
        "validator_hotkey": hotkey,
        "signature": signature,
        "extrinsic_hash": extrinsic_hash,
    }
    _require(
        isinstance(extrinsic_receipt, Mapping)
        and extrinsic_receipt.get("role") == WEIGHT_ROLE
        and extrinsic_receipt.get("purpose") == "validator.set_weights_extrinsic.v2"
        and extrinsic_receipt.get("input_root") == extrinsic_authorization_hash
        and extrinsic_receipt.get("output_root")
        == sha256_json(expected_extrinsic_output)
        and extrinsic_receipt.get("parent_receipt_hashes")
        == [weight_receipt_hash],
        "included extrinsic receipt is invalid",
    )
    _require(
        extrinsic_receipt_hash in root.get("parent_receipt_hashes", []),
        "finalization receipt does not bind included extrinsic",
    )
    extrinsic_parent_hashes = list(root.get("parent_receipt_hashes") or [])
    _require(
        bool(extrinsic_parent_hashes)
        and extrinsic_parent_hashes == sorted(set(extrinsic_parent_hashes)),
        "finalization extrinsic receipt set is not canonical",
    )
    for parent_hash in extrinsic_parent_hashes:
        parent = receipt_by_hash.get(parent_hash)
        _require(
            isinstance(parent, Mapping)
            and parent.get("role") == WEIGHT_ROLE
            and parent.get("purpose") == "validator.set_weights_extrinsic.v2"
            and parent.get("parent_receipt_hashes") == [weight_receipt_hash],
            "finalization contains an invalid extrinsic receipt parent",
        )
    _require(
        root.get("input_root")
        == sha256_json(
            {
                "weight_submission_event_hash": event_hash,
                "extrinsic_receipt_hashes": extrinsic_parent_hashes,
            }
        ),
        "weight finalization receipt input differs",
    )
    scoped_attempts = [
        attempt
        for attempt in graph["transport_attempts"]
        if attempt.get("job_id") == root.get("job_id")
        and attempt.get("purpose") == "validator.weights.finalized.v2"
    ]
    _require(scoped_attempts, "weight finalization has no authenticated chain reads")
    _require(
        all(
            attempt.get("provider_id") == "bittensor_chain"
            and attempt.get("destination_host")
            == "entrypoint-finney.opentensor.ai"
            and attempt.get("destination_port") == 443
            and attempt.get("terminal_status") == "authenticated_response"
            for attempt in scoped_attempts
        ),
        "weight finalization chain evidence is invalid",
    )
    return {
        "validator_hotkey": hotkey,
        "netuid": netuid,
        "epoch_id": epoch_id,
        "weights_hash": weights_hash,
        "weight_receipt_hash": weight_receipt_hash,
        "weight_submission_event_hash": event_hash,
        "extrinsic_authorization_hash": extrinsic_authorization_hash,
        "extrinsic_receipt_hash": extrinsic_receipt_hash,
        "extrinsic_hash": extrinsic_hash,
        "finalized_block": finalized_block,
        "finalized_block_hash": finalized_block_hash,
        "state_transition_hash": state_transition_hash,
        "finalization_receipt_hash": root_hash,
    }


def build_weight_finalization_submission_v2(
    *,
    validator_hotkey: str,
    weight_submission_event_hash: str,
    finalization: Mapping[str, Any],
    receipt_graph: Mapping[str, Any],
    chain_signing_profile: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    value = {
        "schema_version": WEIGHT_FINALIZATION_SUBMISSION_V2_SCHEMA_VERSION,
        "validator_hotkey": str(validator_hotkey),
        "weight_submission_event_hash": str(weight_submission_event_hash),
        "finalization": dict(finalization),
        "receipt_graph": dict(receipt_graph),
    }
    validate_weight_finalization_submission_v2(
        value, chain_signing_profile=chain_signing_profile
    )
    return value
