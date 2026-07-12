"""Host composition for an authoritative validator V2 weight bundle."""

from __future__ import annotations

from typing import Any, Dict, Mapping

from leadpoet_canonical.attested_v2 import build_receipt_graph
from leadpoet_canonical.weight_authority_v2 import (
    build_published_weight_bundle_v2,
)


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
    if not isinstance(enclave_response, Mapping) or set(enclave_response) != required_response:
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
