"""Strict temporary origin guard for the first native testnet401 allocation."""

from __future__ import annotations

from typing import Any, Mapping

from leadpoet_canonical.attested_v2 import validate_receipt_graph


TESTNET401_NETUID = 401
TESTNET401_NETWORK = "test"
TESTNET401_GENESIS_HASH = (
    "0x8f9cf856bf558a14440e75569c9e58594757048d7b3a84b5d25f6bd978263105"
)
TESTNET401_CUTOVER_BLOCK = 7_955_391
TESTNET401_CUTOVER_BLOCK_HASH = (
    "0x08d9d41b0508e1c7dc7fffdfa5d055077e01c593b016c08d42471fecd401fa15"
)
TESTNET401_CUTOVER_MAPPING_HASH = (
    "sha256:4b3941c091d3a29daf9ea863bb6cb587bad7dfd9426cf66a56bcf7de04ce4328"
)
TESTNET401_CUTOVER_AUTHORITY_HASH = (
    "sha256:2eb438e142f4c27d5ce28b2e2c5148cc030724a38ab0bf479ce4fc0fade51bc7"
)
TESTNET401_CUTOVER_RECEIPT_HASH = (
    "sha256:4db3b2649bbd2182488b3f17cbff87abf04f96c9511055cb51efe743710b6aaa"
)
TESTNET401_SNAPSHOT_RECEIPT_HASH = (
    "sha256:66c7a0176e741a1e803539e6c50a3297aa81b74dbf44c8b8d311601f3f847a59"
)
TESTNET401_FIRST_SETTLEMENT_EPOCH = 22_042
TESTNET401_VALIDATOR_UID = 9
TESTNET401_VALIDATOR_HOTKEY = "5CJyMxw6YJJvLhPf58gSpMB7mvSKSCMx9RXhXJum6cNfqMEz"
TESTNET401_BURN_UID = 0
TESTNET401_BURN_HOTKEY = "5E6zy3Dt8BrwsSKbocSF2uxjpMAbr4EksVzxgPJiXh8jq4vf"
TESTNET401_PRE_CUTOVER_LAST_UPDATE = 7_431_466


class TemporaryTestnet401FirstAllocationError(ValueError):
    """The temporary first-allocation origin is not exact."""


def validate_testnet401_cutover_scope_v1(
    *,
    network: str,
    netuid: int,
    cutover: Mapping[str, Any],
) -> dict[str, Any]:
    normalized = dict(cutover)
    if (
        str(network or "").strip().lower() != TESTNET401_NETWORK
        or int(netuid) != TESTNET401_NETUID
        or normalized.get("network_genesis_hash") != TESTNET401_GENESIS_HASH
        or int(normalized.get("netuid", -1)) != TESTNET401_NETUID
        or int(normalized.get("cutover_block", -1)) != TESTNET401_CUTOVER_BLOCK
        or normalized.get("cutover_block_hash") != TESTNET401_CUTOVER_BLOCK_HASH
        or normalized.get("mapping_hash") != TESTNET401_CUTOVER_MAPPING_HASH
        or normalized.get("schema_version") != "leadpoet.subnet_epoch_cutover.v1"
        or normalized.get("epoch_scheme") != "bittensor.subnet_epoch_index.v1"
        or int(normalized.get("first_subnet_epoch_index", -1))
        != TESTNET401_FIRST_SETTLEMENT_EPOCH
        or int(normalized.get("first_settlement_epoch_id", -1))
        != TESTNET401_FIRST_SETTLEMENT_EPOCH
        or int(normalized.get("last_legacy_epoch_id", -1))
        != TESTNET401_FIRST_SETTLEMENT_EPOCH - 1
    ):
        raise TemporaryTestnet401FirstAllocationError(
            "temporary first allocation is not the approved testnet401 origin"
        )
    return normalized


def validate_testnet401_cutover_parent_v1(
    graph: Mapping[str, Any],
    *,
    network: str,
    netuid: int,
    cutover: Mapping[str, Any],
) -> dict[str, Any]:
    validate_testnet401_cutover_scope_v1(
        network=network,
        netuid=netuid,
        cutover=cutover,
    )
    validate_receipt_graph(graph)
    if graph.get("root_receipt_hash") != TESTNET401_CUTOVER_RECEIPT_HASH:
        raise TemporaryTestnet401FirstAllocationError(
            "temporary first allocation cutover root differs"
        )
    roots = [
        receipt
        for receipt in graph.get("receipts") or ()
        if isinstance(receipt, Mapping)
        and receipt.get("receipt_hash") == TESTNET401_CUTOVER_RECEIPT_HASH
    ]
    if len(roots) != 1:
        raise TemporaryTestnet401FirstAllocationError(
            "temporary first allocation cutover receipt is unavailable"
        )
    receipt = roots[0]
    if (
        receipt.get("role") != "gateway_coordinator"
        or receipt.get("purpose") != "research_lab.subnet_epoch_cutover.v2"
        or receipt.get("status") != "succeeded"
        or int(receipt.get("epoch_id", -1)) != TESTNET401_FIRST_SETTLEMENT_EPOCH
        or receipt.get("output_root") != TESTNET401_CUTOVER_AUTHORITY_HASH
        or receipt.get("parent_receipt_hashes") != [TESTNET401_SNAPSHOT_RECEIPT_HASH]
    ):
        raise TemporaryTestnet401FirstAllocationError(
            "temporary first allocation cutover receipt differs"
        )
    return {
        "schema_version": "leadpoet.temporary_testnet401_first_allocation_origin.v1",
        "network_genesis_hash": TESTNET401_GENESIS_HASH,
        "netuid": TESTNET401_NETUID,
        "cutover_block": TESTNET401_CUTOVER_BLOCK,
        "cutover_mapping_hash": TESTNET401_CUTOVER_MAPPING_HASH,
        "cutover_authority_hash": TESTNET401_CUTOVER_AUTHORITY_HASH,
        "cutover_receipt_hash": TESTNET401_CUTOVER_RECEIPT_HASH,
        "snapshot_receipt_hash": TESTNET401_SNAPSHOT_RECEIPT_HASH,
    }
