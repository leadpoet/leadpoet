#!/usr/bin/env python3
"""Independently verify one temporary testnet401 native weight publication."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import sys
from urllib.request import urlopen


CONFIG_PATH = Path("/run/leadpoet-testnet401/config.json")
STATUS_PATH = Path("/run/leadpoet-testnet401/evidence/status.json")
PRIOR_RELEASE_CHANNEL_PATH = Path(
    "/run/leadpoet-testnet401/prior-release-channel-v2.json"
)
PRIOR_RELEASE_LINEAGE_PATH = Path(
    "/run/leadpoet-testnet401/prior-release-lineage-v1.json"
)
PRIOR_RELEASE_COMMIT = "f92748d00ced815e710e4e42ba8f7f17207507d7"
NETUID = 401
NETWORK = "test"
CHAIN_HOST = "test.finney.opentensor.ai"
CHAIN_ENDPOINT = "wss://test.finney.opentensor.ai:443"
EXPECTED_GENESIS = (
    "8f9cf856bf558a14440e75569c9e58594757048d7b3a84b5d25f6bd978263105"
)
EXPECTED_MAPPING_HASH = (
    "sha256:4b3941c091d3a29daf9ea863bb6cb587bad7dfd9426cf66a56bcf7de04ce4328"
)
EXPECTED_PROFILE_HASH = (
    "sha256:a2db2db86ffb10bbf41dd6923e1310726031bc4183841e07e6d2da50e6e58677"
)
EXPECTED_VALIDATOR = "5CJyMxw6YJJvLhPf58gSpMB7mvSKSCMx9RXhXJum6cNfqMEz"
EXPECTED_MINER = "5FEtvBzsh5Zc8nDyq4Jb2nZ7o6ZD2homYsKjbZtFj5tybqth"
EXPECTED_BURN = "5E6zy3Dt8BrwsSKbocSF2uxjpMAbr4EksVzxgPJiXh8jq4vf"
BASELINE_LAST_UPDATE = 7_431_466
MAX_AUTHORITY_BYTES = 8 * 1024 * 1024


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def raw_hash(value) -> str:
    return str(value).lower().removeprefix("0x")


def build_approved_release_lineage(
    *,
    candidate,
    gateway_release,
    validator_release,
    runtime_lineage,
    prior_release_channel_path=PRIOR_RELEASE_CHANNEL_PATH,
    prior_release_lineage_path=PRIOR_RELEASE_LINEAGE_PATH,
):
    """Validate the exact current-only or approved two-release lineage."""

    from gateway.tee.release_channel_v2 import (
        build_release_channel_v2,
        build_release_lineage_v2,
        validate_prior_release_channel_v2,
    )
    from gateway.tee.release_lineage_v2 import (
        validate_compact_release_lineage_v2,
        validate_prior_compact_release_lineage_v2,
    )

    current_channel = build_release_channel_v2(
        gateway_release_manifest=gateway_release,
        validator_release_manifest=validator_release,
    )
    require(
        current_channel["commit_sha"] == candidate,
        "current release channel differs",
    )
    normalized_runtime = validate_compact_release_lineage_v2(
        runtime_lineage,
        expected_current_commit=candidate,
        expected_current_gateway_release_hash=gateway_release["release_hash"],
    )

    prior_channel_present = prior_release_channel_path.exists()
    prior_lineage_present = prior_release_lineage_path.exists()
    require(
        prior_channel_present == prior_lineage_present,
        "prior release artifacts are incomplete",
    )
    approved_channels = []
    if prior_channel_present:
        require(
            candidate != PRIOR_RELEASE_COMMIT,
            "current and prior release commits are identical",
        )
        prior_channel = validate_prior_release_channel_v2(
            read_json(prior_release_channel_path),
            expected_commit=PRIOR_RELEASE_COMMIT,
        )
        prior_lineage = validate_prior_compact_release_lineage_v2(
            read_json(prior_release_lineage_path),
            expected_current_commit=PRIOR_RELEASE_COMMIT,
            expected_current_gateway_release_hash=prior_channel[
                "gateway_release_manifest"
            ]["release_hash"],
        )
        expected_prior_lineage = build_release_lineage_v2(
            [prior_channel],
            current_commit=PRIOR_RELEASE_COMMIT,
        )
        require(
            prior_lineage == expected_prior_lineage,
            "prior release lineage differs from its approved channel",
        )
        approved_channels.append(prior_channel)
    approved_channels.append(current_channel)
    expected_runtime = build_release_lineage_v2(
        approved_channels,
        current_commit=candidate,
    )
    require(
        normalized_runtime == expected_runtime,
        "runtime release lineage differs from the approved release set",
    )
    return normalized_runtime


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epoch-id")
    args = parser.parse_args()
    if args.epoch_id is None:
        return None
    require(
        bool(re.fullmatch(r"[0-9]{1,20}", args.epoch_id)),
        "epoch ID is invalid",
    )
    epoch_id = int(args.epoch_id)
    require(0 <= epoch_id < 1 << 64, "epoch ID is invalid")
    return epoch_id


def main() -> int:
    requested_epoch = parse_args()
    require(
        os.environ.get("BITTENSOR_NETWORK") == NETWORK,
        "network environment differs",
    )
    require(
        os.environ.get("BITTENSOR_NETUID") == str(NETUID),
        "netuid environment differs",
    )
    config = read_json(CONFIG_PATH)
    repo_root = Path(str(config.get("repo_root") or ""))
    require(repo_root.is_absolute() and repo_root.is_dir(), "repo root is invalid")
    sys.path.insert(0, str(repo_root))
    os.chdir(repo_root)

    if requested_epoch is None:
        candidate_match = read_json(STATUS_PATH)["evidence"][
            "automatic_chain_proof"
        ]
        require(
            candidate_match.get("status") == "candidate_match",
            "status has no candidate match",
        )
        epoch_id = int(candidate_match["epoch_id"])
    else:
        epoch_id = requested_epoch

    require(
        config["validator"]["expected_hotkey"] == EXPECTED_VALIDATOR,
        "validator differs",
    )
    require(
        config["validator"]["expected_selected_profile_hash"]
        == EXPECTED_PROFILE_HASH,
        "profile pin differs",
    )

    # Configure this before importing modules that copy the canonical boundary.
    from leadpoet_canonical.chain_source_v2 import (
        configure_chain_source_boundary_v2,
    )

    configure_chain_source_boundary_v2(
        chain_host=CHAIN_HOST,
        chain_archive_host=CHAIN_HOST,
    )

    from Leadpoet.utils.subnet_epoch import SubnetEpochCutover
    from gateway.tee.release_lineage_v2 import (
        build_compact_release_lineage_boot_verifier_v2,
    )
    from leadpoet_canonical.ancestry_checkpoint_v2 import (
        derive_ancestry_lineage_id_v2,
    )
    from leadpoet_canonical.attested_v2 import sha256_json
    from leadpoet_canonical.compact_auditor_authority_v2 import (
        verify_compact_published_weight_authority_v2,
    )
    from leadpoet_canonical.hotkey_authority_v2 import (
        select_chain_signing_profile,
    )
    from validator_tee.enclave.hotkey_authority_v2 import (
        load_chain_signing_profile,
    )
    from validator_tee.host.verify_chain_signing_profile_v2 import (
        read_live_chain_signing_state,
        verify_chain_signing_profile_v2,
    )

    base_profile = load_chain_signing_profile(
        Path(config["validator"]["chain_profile"])
    )
    live = read_live_chain_signing_state(NETWORK, NETUID)
    compatibility = verify_chain_signing_profile_v2(
        profile=base_profile,
        runtime_version=live["runtime_version"],
        genesis_hash=live["genesis_hash"],
        call_metadata=live["call_metadata"],
        tempo=live["tempo"],
        subnet_reveal_period_epochs=live["subnet_reveal_period_epochs"],
    )
    selected_profile = select_chain_signing_profile(
        base_profile,
        runtime_version=live["runtime_version"],
        genesis_hash=live["genesis_hash"],
    )
    selected_profile_hash = sha256_json(selected_profile)
    require(
        compatibility["selected_profile_hash"]
        == selected_profile_hash
        == EXPECTED_PROFILE_HASH,
        "live selected profile differs",
    )
    require(
        selected_profile["spec_version"] == 454
        and selected_profile["transaction_version"] == 1
        and selected_profile["chain_endpoint"] == CHAIN_ENDPOINT
        and selected_profile["tempo"] == 360
        and selected_profile["subnet_reveal_period_epochs"] == 1,
        "selected profile contract differs",
    )

    cutover = SubnetEpochCutover.from_mapping(
        read_json(Path(config["validator"]["cutover_manifest"]))
    )
    require(
        cutover.netuid == NETUID
        and raw_hash(cutover.network_genesis_hash) == EXPECTED_GENESIS
        and cutover.mapping_hash == EXPECTED_MAPPING_HASH,
        "cutover identity differs",
    )
    lineage_id = derive_ancestry_lineage_id_v2(
        cutover_mapping_hash=cutover.mapping_hash,
        network_genesis_hash=cutover.network_genesis_hash,
        netuid=NETUID,
    )

    gateway_release = read_json(Path(config["gateway"]["release_manifest"]))
    validator_release = read_json(
        Path(config["validator"]["release_manifest"])
    )
    candidate = str(config["candidate_sha"])
    require(
        bool(re.fullmatch(r"[0-9a-f]{40}", candidate)),
        "candidate SHA is invalid",
    )
    require(
        gateway_release["commit_sha"] == candidate,
        "gateway release differs",
    )
    require(
        validator_release["release"]["commit_sha"] == candidate,
        "validator release differs",
    )
    approved_lineage = build_approved_release_lineage(
        candidate=candidate,
        gateway_release=gateway_release,
        validator_release=validator_release,
        runtime_lineage=read_json(
            Path(config["gateway"]["release_lineage"])
        ),
    )
    boot_verifier = build_compact_release_lineage_boot_verifier_v2(
        approved_lineage
    )

    authority_url = (
        "http://127.0.0.1:8000/weights/v2/published-compact/"
        f"{NETUID}/{epoch_id}"
    )
    with urlopen(authority_url, timeout=20) as response:
        raw_authority = response.read(MAX_AUTHORITY_BYTES + 1)
    require(
        len(raw_authority) <= MAX_AUTHORITY_BYTES,
        "authority exceeds size bound",
    )
    authority = json.loads(raw_authority)
    require(
        authority.get("authority_stage") == "finalized",
        "authority is not finalized",
    )
    verified = verify_compact_published_weight_authority_v2(
        authority,
        identity_cache=None,
        chain_signing_profile=selected_profile,
        expected_lineage_id=lineage_id,
        expected_chain=CHAIN_ENDPOINT,
        boot_verifier=boot_verifier,
    )
    require(
        verified["authority_stage"] == "finalized"
        and verified["validator_hotkey"] == EXPECTED_VALIDATOR
        and verified["netuid"] == NETUID
        and verified["epoch_id"] == epoch_id,
        "verified authority identity differs",
    )

    expected_pairs = sorted(
        (int(uid), int(weight))
        for uid, weight in zip(
            verified["uids"],
            verified["weights_u16"],
        )
    )
    require(
        {uid for uid, _weight in expected_pairs} == {0, 11},
        "destination set differs",
    )
    require(
        4 * dict(expected_pairs)[11]
        == sum(weight for _uid, weight in expected_pairs),
        "champion share is not exactly one quarter",
    )

    authorization = authority["finalization"]["compact_submission"][
        "finalization"
    ]["extrinsic_authorization"]
    target_subnet_epoch_index = int(authorization["subnet_epoch_index"])
    require(
        int(authorization["epoch_id"]) == epoch_id
        and int(authorization["netuid"]) == NETUID,
        "authorization identity differs",
    )
    require(
        cutover.settlement_epoch_id(target_subnet_epoch_index) == epoch_id,
        "authorization epoch mapping differs",
    )

    import bittensor as bt

    subtensor = bt.Subtensor(network=NETWORK)
    try:
        finalized_head_hash = str(live["finalized_block_hash"])
        finalized_head_block = int(live["finalized_block"])
        query = subtensor.substrate.query

        def value(result):
            return getattr(result, "value", result)

        validator_uid = int(
            value(
                query(
                    "SubtensorModule",
                    "Uids",
                    [NETUID, EXPECTED_VALIDATOR],
                    block_hash=finalized_head_hash,
                )
            )
        )
        miner_uid = int(
            value(
                query(
                    "SubtensorModule",
                    "Uids",
                    [NETUID, EXPECTED_MINER],
                    block_hash=finalized_head_hash,
                )
            )
        )
        burn_uid = int(
            value(
                query(
                    "SubtensorModule",
                    "Uids",
                    [NETUID, EXPECTED_BURN],
                    block_hash=finalized_head_hash,
                )
            )
        )
        last_update = int(
            list(
                value(
                    query(
                        "SubtensorModule",
                        "LastUpdate",
                        [NETUID],
                        block_hash=finalized_head_hash,
                    )
                )
            )[validator_uid]
        )
        chain_pairs = sorted(
            (int(uid), int(weight))
            for uid, weight in value(
                query(
                    "SubtensorModule",
                    "Weights",
                    [NETUID, validator_uid],
                    block_hash=finalized_head_hash,
                )
            )
        )
        current_subnet_epoch_index = int(
            value(
                query(
                    "SubtensorModule",
                    "SubnetEpochIndex",
                    [NETUID],
                    block_hash=finalized_head_hash,
                )
            )
        )
        commit_inclusion_block = int(verified["finalized_block"])
        canonical_commit_hash = str(
            subtensor.substrate.get_block_hash(commit_inclusion_block)
        )
    finally:
        close = getattr(getattr(subtensor, "substrate", None), "close", None)
        if callable(close):
            close()

    require(
        (validator_uid, miner_uid, burn_uid) == (9, 11, 0),
        "live hotkey UIDs differ",
    )
    require(
        chain_pairs == expected_pairs,
        "finalized revealed vector differs",
    )
    require(
        commit_inclusion_block <= finalized_head_block,
        "commit inclusion is not finalized",
    )
    require(
        raw_hash(canonical_commit_hash)
        == raw_hash(verified["finalized_block_hash"]),
        "commit block hash differs",
    )
    require(
        current_subnet_epoch_index
        >= target_subnet_epoch_index
        + int(selected_profile["subnet_reveal_period_epochs"]),
        "reveal is pending",
    )
    # The authority block is commit inclusion. LastUpdate is reveal readback.
    require(
        BASELINE_LAST_UPDATE < last_update <= finalized_head_block,
        "LastUpdate did not advance on finalized chain",
    )
    require(
        last_update > int(verified["block"]),
        "LastUpdate does not postdate the bundle",
    )
    require(
        last_update >= commit_inclusion_block,
        "LastUpdate predates the committed authority",
    )

    print(
        json.dumps(
            {
                "status": "passed",
                "candidate_sha": candidate,
                "netuid": NETUID,
                "epoch_id": epoch_id,
                "selected_profile_hash": selected_profile_hash,
                "selected_spec_version": selected_profile["spec_version"],
                "authority_hash": verified["authority_hash"],
                "bundle_hash": verified["bundle_hash"],
                "weights_hash": verified["weights_hash"],
                "weight_submission_event_hash": verified[
                    "weight_submission_event_hash"
                ],
                "weight_finalization_event_hash": verified[
                    "weight_finalization_event_hash"
                ],
                "commit_inclusion_block": commit_inclusion_block,
                "commit_inclusion_block_hash": verified[
                    "finalized_block_hash"
                ],
                "revealed_last_update_block": last_update,
                "finalized_readback_block": finalized_head_block,
                "finalized_readback_block_hash": finalized_head_hash,
                "revealed_weights": expected_pairs,
                "champion_uid": miner_uid,
                "champion_share_exact": "1/4",
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
