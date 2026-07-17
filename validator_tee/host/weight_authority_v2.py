"""Host composition for an authoritative validator V2 weight bundle."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

from leadpoet_canonical.attested_v2 import build_receipt_graph, sha256_json
from leadpoet_canonical.weight_authority_v2 import (
    build_published_weight_bundle_v2,
    validate_published_weight_bundle_v2,
)


_EPOCH_SNAPSHOT_FIELDS = {
    "schema_version",
    "epoch_scheme",
    "network_genesis_hash",
    "netuid",
    "head_kind",
    "block_hash",
    "current_block",
    "last_epoch_block",
    "pending_epoch_at",
    "subnet_epoch_index",
    "tempo",
    "blocks_since_last_step",
    "observed_at",
    "epoch_id",
    "epoch_ref",
    "epoch_block",
    "next_epoch_block",
    "blocks_remaining",
    "settlement_epoch_id",
    "cutover_mapping_hash",
}


class HostWeightAuthorityV2Error(RuntimeError):
    """An enclave response cannot form the authoritative published graph."""


def build_authoritative_weight_bundle_v2(
    *,
    enclave_response: Mapping[str, Any],
    validator_hotkey: str,
    binding_message: str,
    binding_signature_result: Mapping[str, Any],
) -> Dict[str, Any]:
    required_response = {
        "weight_snapshot",
        "weight_result",
        "weights_signature",
        "receipt_graph",
        "boot_identity",
        "weight_authorization_id",
        "source_artifacts",
    }
    response_fields = (
        frozenset(enclave_response) if isinstance(enclave_response, Mapping) else frozenset()
    )
    optional_epoch_fields = {"epoch_authority", "epoch_boundary"}
    if (
        not isinstance(enclave_response, Mapping)
        or response_fields not in {
            frozenset(required_response),
            frozenset(required_response | optional_epoch_fields),
        }
    ):
        raise HostWeightAuthorityV2Error(
            "authoritative enclave response fields are invalid"
        )
    graph = enclave_response.get("receipt_graph")
    source_artifacts = enclave_response.get("source_artifacts")
    if not isinstance(source_artifacts, list):
        raise HostWeightAuthorityV2Error(
            "authoritative chain source artifacts are missing"
        )
    signature_receipt = binding_signature_result.get("receipt")
    if not isinstance(graph, Mapping) or not isinstance(signature_receipt, Mapping):
        raise HostWeightAuthorityV2Error(
            "authoritative binding receipt is missing"
        )
    if (
        binding_signature_result.get("purpose")
        != "validator.gateway_binding.v2"
        or binding_signature_result.get("validator_hotkey") != validator_hotkey
    ):
        raise HostWeightAuthorityV2Error(
            "authoritative binding result has the wrong domain"
        )
    computed_root = str(graph.get("root_receipt_hash") or "")
    if signature_receipt.get("parent_receipt_hashes") != [computed_root]:
        raise HostWeightAuthorityV2Error(
            "authoritative binding receipt does not bind computed weights"
        )
    complete_graph = build_receipt_graph(
        root_receipt_hash=str(signature_receipt["receipt_hash"]),
        boot_identities=list(graph["boot_identities"]),
        receipts=[*list(graph["receipts"]), dict(signature_receipt)],
        transport_attempts=list(graph["transport_attempts"]),
        host_operations=list(graph["host_operations"]),
    )
    return build_published_weight_bundle_v2(
        validator_hotkey=validator_hotkey,
        binding_message=binding_message,
        validator_hotkey_signature=str(binding_signature_result["signature"]),
        weight_snapshot=enclave_response["weight_snapshot"],
        weight_result=enclave_response["weight_result"],
        weights_signature=str(enclave_response["weights_signature"]),
        receipt_graph=complete_graph,
    )


def build_stateful_epoch_evidence_v1(
    *,
    enclave_response: Mapping[str, Any],
    published_bundle: Mapping[str, Any],
) -> Optional[Dict[str, Any]]:
    """Preserve measured epoch documents beside the unchanged V2 bundle.

    The public bundle schema remains byte-for-byte V2.  This separate envelope
    lets the gateway persist the exact current and boundary documents whose
    hashes are signed by the two dedicated enclave receipts in that bundle.
    """

    current = enclave_response.get("epoch_authority")
    boundary = enclave_response.get("epoch_boundary")
    if current is None and boundary is None:
        return None
    if not isinstance(current, Mapping) or not isinstance(boundary, Mapping):
        raise HostWeightAuthorityV2Error(
            "stateful epoch evidence documents are incomplete"
        )
    current_doc = dict(current)
    boundary_doc = dict(boundary)
    if set(current_doc) != _EPOCH_SNAPSHOT_FIELDS or set(
        boundary_doc
    ) != _EPOCH_SNAPSHOT_FIELDS:
        raise HostWeightAuthorityV2Error(
            "stateful epoch evidence fields are invalid"
        )
    for document in (current_doc, boundary_doc):
        if (
            document.get("schema_version")
            != "leadpoet.subnet_epoch_snapshot.v1"
            or document.get("epoch_scheme")
            != "bittensor.subnet_epoch_index.v1"
            or document.get("head_kind") != "finalized"
            or int(document.get("epoch_id", -1))
            != int(document.get("subnet_epoch_index", -2))
            or int(document.get("settlement_epoch_id", -1)) < 0
        ):
            raise HostWeightAuthorityV2Error(
                "stateful epoch evidence identity is invalid"
            )
    identity_fields = (
        "epoch_scheme",
        "network_genesis_hash",
        "netuid",
        "subnet_epoch_index",
        "epoch_ref",
        "settlement_epoch_id",
        "cutover_mapping_hash",
    )
    if any(current_doc[field] != boundary_doc[field] for field in identity_fields):
        raise HostWeightAuthorityV2Error(
            "stateful current and boundary epoch identities differ"
        )
    if (
        int(boundary_doc["current_block"])
        != int(boundary_doc["last_epoch_block"])
        or int(boundary_doc["epoch_block"]) != 0
        or int(current_doc["current_block"]) < int(boundary_doc["current_block"])
        or int(current_doc["epoch_block"])
        != int(current_doc["current_block"])
        - int(current_doc["last_epoch_block"])
    ):
        raise HostWeightAuthorityV2Error(
            "stateful epoch boundary arithmetic is invalid"
        )

    verified = validate_published_weight_bundle_v2(published_bundle)
    if (
        int(verified["netuid"]) != int(current_doc["netuid"])
        or int(verified["epoch_id"])
        != int(current_doc["settlement_epoch_id"])
        or int(verified["block"]) != int(current_doc["current_block"])
    ):
        raise HostWeightAuthorityV2Error(
            "stateful epoch evidence differs from the published bundle"
        )
    current_hash = sha256_json(current_doc)
    boundary_hash = sha256_json(boundary_doc)
    candidates = [
        receipt
        for receipt in published_bundle["receipt_graph"]["receipts"]
        if receipt.get("role") == "validator_weights"
        and receipt.get("purpose") == "validator.subnet_epoch_snapshot.v2"
        and receipt.get("output_root") in {current_hash, boundary_hash}
        and int(receipt.get("epoch_id", -1)) == int(verified["epoch_id"])
    ]
    if current_hash == boundary_hash:
        same_output_candidates = [
            receipt
            for receipt in candidates
            if receipt.get("output_root") == current_hash
        ]
        if len(same_output_candidates) != 2:
            raise HostWeightAuthorityV2Error(
                "stateful cutover epoch receipts are missing or ambiguous"
            )
        by_parent_count = sorted(
            same_output_candidates,
            key=lambda item: len(item["parent_receipt_hashes"]),
        )
        boundary_receipt, current_receipt = by_parent_count
    else:
        current_receipts = [
            receipt
            for receipt in candidates
            if receipt.get("output_root") == current_hash
        ]
        boundary_receipts = [
            receipt
            for receipt in candidates
            if receipt.get("output_root") == boundary_hash
        ]
        if len(current_receipts) != 1 or len(boundary_receipts) != 1:
            raise HostWeightAuthorityV2Error(
                "stateful epoch evidence receipts are missing or ambiguous"
            )
        current_receipt = current_receipts[0]
        boundary_receipt = boundary_receipts[0]
    if current_receipt.get("parent_receipt_hashes") != [
        boundary_receipt["receipt_hash"]
    ]:
        raise HostWeightAuthorityV2Error(
            "stateful current snapshot does not bind its epoch boundary"
        )
    return {
        "schema_version": "leadpoet.validator_subnet_epoch_evidence.v1",
        "validator_hotkey": str(verified["validator_hotkey"]),
        "bundle_hash": str(verified["bundle_hash"]),
        "cutover_mapping_hash": str(current_doc["cutover_mapping_hash"]),
        "epoch_authority": current_doc,
        "epoch_authority_hash": current_hash,
        "epoch_authority_receipt_hash": str(current_receipt["receipt_hash"]),
        "epoch_boundary": boundary_doc,
        "epoch_boundary_hash": boundary_hash,
        "epoch_boundary_receipt_hash": str(boundary_receipt["receipt_hash"]),
        "receipt_graph": dict(published_bundle["receipt_graph"]),
    }


def validate_stateful_epoch_evidence_v1(
    value: Any,
    *,
    published_bundle: Mapping[str, Any],
) -> Optional[Dict[str, Any]]:
    if value is None:
        graph = (
            published_bundle.get("receipt_graph")
            if isinstance(published_bundle, Mapping)
            else None
        )
        receipts = graph.get("receipts") if isinstance(graph, Mapping) else None
        if isinstance(receipts, list) and any(
            isinstance(receipt, Mapping)
            and receipt.get("purpose") == "validator.subnet_epoch_snapshot.v2"
            for receipt in receipts
        ):
            raise HostWeightAuthorityV2Error(
                "stateful bundle is missing its epoch evidence envelope"
            )
        return None
    if not isinstance(value, Mapping):
        raise HostWeightAuthorityV2Error(
            "stateful epoch evidence envelope is invalid"
        )
    rebuilt = build_stateful_epoch_evidence_v1(
        enclave_response={
            "epoch_authority": value.get("epoch_authority"),
            "epoch_boundary": value.get("epoch_boundary"),
        },
        published_bundle=published_bundle,
    )
    if rebuilt is None or dict(value) != rebuilt:
        raise HostWeightAuthorityV2Error(
            "stateful epoch evidence envelope is not canonical"
        )
    return rebuilt
