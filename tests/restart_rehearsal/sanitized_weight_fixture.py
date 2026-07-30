#!/usr/bin/env python3.11
"""Build deterministic production-shaped V2 inputs for the local rehearsal.

This module deliberately owns its sanitized input construction.  It never
imports unit-test helpers, and every envelope, receipt, weight snapshot, and
bundle it emits is constructed by candidate production code.
"""

from __future__ import annotations

import base64
import hashlib
from typing import Any, Mapping

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from leadpoet_canonical.attested_v2 import (
    COORDINATOR_ROLE,
    EMPTY_ARTIFACT_ROOT,
    EMPTY_HOST_OPERATION_ROOT,
    EMPTY_TRANSPORT_ROOT,
    WEIGHT_ROLE,
    build_boot_attestation_user_data,
    build_boot_identity_body,
    build_execution_receipt_body,
    build_receipt_graph,
    build_transport_attempt,
    canonical_json,
    create_boot_identity,
    create_signed_execution_receipt,
    merkle_root,
    sha256_json,
)
from leadpoet_canonical.binding import create_binding_message
from leadpoet_canonical.hotkey_authority_v2 import (
    build_application_signature_request_v2,
)
from leadpoet_canonical.weight_authority_v2 import (
    PUBLISHED_WEIGHT_BUNDLE_V2_SCHEMA_VERSION,
    WEIGHT_INPUT_PURPOSES,
    build_weight_snapshot_v2,
    weight_input_output_roots_v2,
    weight_input_value_documents_v2,
)
from leadpoet_canonical.weight_computation import (
    WEIGHT_SNAPSHOT_SCHEMA_VERSION,
    compute_final_weights,
    weight_config_hash,
)


HASH = "sha256:" + "1" * 64
HASH_B = "sha256:" + "2" * 64
NOW = "2026-07-25T00:00:00Z"
VALIDATOR_HOTKEY = "5FqLp5QmNRiHGyj3xbLVnDHfCx25qxJX5CUhpndF9GFfZZiK"


class SanitizedWeightFixture:
    """One deterministic, credential-free production-shaped epoch."""

    def __init__(self, *, candidate_sha: str, epoch_id: int):
        self.candidate_sha = candidate_sha
        self.epoch_id = int(epoch_id)
        self.pcr0 = hashlib.sha384(
            b"leadpoet-local-pcr0:" + candidate_sha.encode("ascii")
        ).hexdigest()
        self.coordinator_key = Ed25519PrivateKey.from_private_bytes(
            hashlib.sha256(
                b"coordinator:" + candidate_sha.encode("ascii")
            ).digest()
        )
        self.weight_key = Ed25519PrivateKey.from_private_bytes(
            hashlib.sha256(
                b"validator:" + candidate_sha.encode("ascii")
            ).digest()
        )

    @staticmethod
    def _public_key(key: Ed25519PrivateKey) -> str:
        return key.public_key().public_bytes_raw().hex()

    def _boot(
        self,
        *,
        role: str,
        key: Ed25519PrivateKey,
        config_hash: str,
        release_identity: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        physical_role = (
            "gateway_coordinator"
            if role == COORDINATOR_ROLE
            else "validator_weights"
        )
        pcr0 = self.pcr0
        build_manifest_hash = HASH
        dependency_lock_hash = HASH_B
        if release_identity is not None:
            if physical_role != "gateway_coordinator":
                raise ValueError(
                    "release identity override is only valid for the coordinator"
                )
            if (
                str(release_identity.get("commit_sha") or "").lower()
                != self.candidate_sha
            ):
                raise ValueError("coordinator release identity commit differs")
            pcr0 = str(release_identity.get("pcr0") or "").lower()
            build_manifest_hash = str(
                release_identity.get("execution_manifest_hash") or ""
            ).lower()
            dependency_lock_hash = str(
                release_identity.get("dependency_lock_hash") or ""
            ).lower()
            if (
                len(pcr0) != 96
                or any(character not in "0123456789abcdef" for character in pcr0)
                or any(
                    len(value) != 71
                    or not value.startswith("sha256:")
                    or any(
                        character not in "0123456789abcdef"
                        for character in value[7:]
                    )
                    for value in (
                        build_manifest_hash,
                        dependency_lock_hash,
                    )
                )
            ):
                raise ValueError("coordinator release identity hashes are invalid")
        provisional = {
            "role": role,
            "physical_role": physical_role,
            "commit_sha": self.candidate_sha,
            "pcr0": pcr0,
            "build_manifest_hash": build_manifest_hash,
            "dependency_lock_hash": dependency_lock_hash,
            "config_hash": config_hash,
            "boot_nonce": hashlib.sha256(
                f"{physical_role}:{self.epoch_id}".encode()
            ).hexdigest()[:32],
            "signing_pubkey": self._public_key(key),
            "transport_pubkey": hashlib.sha256(
                f"transport:{physical_role}".encode()
            ).hexdigest(),
            "transport_certificate_hash": HASH_B,
            "issued_at": NOW,
        }
        if release_identity is None:
            attestation_user_data_hash = HASH
            attestation_document = (
                f"sanitized-attestation:{physical_role}".encode()
            )
        else:
            user_data = build_boot_attestation_user_data(provisional)
            attestation_user_data_hash = sha256_json(user_data)
            attestation_document = canonical_json(
                {
                    "schema_version": "leadpoet.local_nitro_attestation.v1",
                    "pcr0": pcr0,
                    "enclave_pubkey": provisional["signing_pubkey"],
                    "user_data": user_data,
                }
            ).encode("utf-8")
        body = build_boot_identity_body(
            **provisional,
            attestation_user_data_hash=attestation_user_data_hash,
        )
        return create_boot_identity(
            body=body,
            attestation_document_b64=base64.b64encode(
                attestation_document
            ).decode("ascii"),
        )

    def receipt(
        self,
        *,
        role: str,
        purpose: str,
        job_id: str,
        key: Ed25519PrivateKey,
        boot: dict[str, Any],
        config_hash: str,
        input_root: str = HASH,
        output_root: str = HASH_B,
        parents: tuple[str, ...] | list[str] = (),
        sequence: int = 1,
        transport_root: str = EMPTY_TRANSPORT_ROOT,
        artifact_root: str = EMPTY_ARTIFACT_ROOT,
    ) -> dict[str, Any]:
        if boot.get("commit_sha") != self.candidate_sha:
            raise ValueError("receipt boot commit differs from fixture candidate")
        body = build_execution_receipt_body(
            role=role,
            purpose=purpose,
            job_id=job_id,
            epoch_id=self.epoch_id,
            sequence=sequence,
            commit_sha=str(boot["commit_sha"]),
            pcr0=str(boot["pcr0"]),
            build_manifest_hash=str(boot["build_manifest_hash"]),
            dependency_lock_hash=str(boot["dependency_lock_hash"]),
            config_hash=config_hash,
            boot_identity_hash=boot["boot_identity_hash"],
            input_root=input_root,
            output_root=output_root,
            transport_root_hash=transport_root,
            host_operation_root_hash=EMPTY_HOST_OPERATION_ROOT,
            artifact_root=artifact_root,
            parent_receipt_hashes=parents,
            status="succeeded",
            failure_code=None,
            issued_at=NOW,
        )
        return create_signed_execution_receipt(
            body=body,
            enclave_pubkey=self._public_key(key),
            sign_digest=key.sign,
        )

    def source_attempt(
        self,
        *,
        category: str,
        job_id: str,
        purpose: str,
        sequence: int,
        provider_id: str,
        host: str,
        method: str,
    ) -> dict[str, Any]:
        return build_transport_attempt(
            request_id=f"{sequence + 1:032x}",
            logical_operation_id=f"weight-source:{category}",
            job_id=job_id,
            purpose=purpose,
            provider_id=provider_id,
            attempt_number=0,
            method=method,
            destination_host=host,
            destination_port=443,
            path_hash=sha256_json({"category": category, "path": "source"}),
            nonsecret_headers_hash=sha256_json({"accept": "application/json"}),
            body_hash=sha256_json({"body": ""}),
            credential_ref_hash=sha256_json({"credential": provider_id}),
            retry_policy_hash=sha256_json({"retry": provider_id}),
            timeout_ms=30000,
            started_at=NOW,
            terminal_status="authenticated_response",
            http_status=200,
            response_hash=sha256_json(
                {"category": category, "response": "body"}
            ),
            request_artifact_hash=sha256_json(
                {"category": category, "artifact": "request"}
            ),
            response_artifact_hash=sha256_json(
                {"category": category, "artifact": "response"}
            ),
            tls_peer_chain_hash=sha256_json({"tls": host}),
            tls_protocol="TLSv1.3",
            failure_code=None,
            completed_at=NOW,
        )

    def calculation_snapshot(
        self,
        parent_hashes: list[str],
        allocation_hash: str,
    ) -> dict[str, Any]:
        allocation_doc = {
            "epoch": self.epoch_id,
            "lab_cap_percent": 20.0,
            "unallocated_percent": 15.0,
            "champion_credit_policy": "accelerated_lifetime_cap_v1",
            "reimbursement_allocations": [],
            "champion_allocations": [
                {
                    "champion_reward_id": (
                        f"rehearsal-champion-{self.epoch_id}"
                    ),
                    "uid": 2,
                    "miner_hotkey": "lab-hotkey",
                    "paid_alpha_percent": 5.0,
                    "base_desired_alpha_percent": 7.3,
                    "total_due_alpha_percent": 146.0,
                    "paid_alpha_percent_to_date": 30.0,
                    "remaining_alpha_percent_before_epoch": 116.0,
                    "remaining_alpha_percent_after_epoch": 111.0,
                }
            ],
            "queued_champion_allocations": [],
        }
        allocation_doc["allocation_hash"] = sha256_json(allocation_doc)
        snapshot = {
            "schema_version": WEIGHT_SNAPSHOT_SCHEMA_VERSION,
            "netuid": 71,
            "epoch_id": self.epoch_id,
            "block": self.epoch_id * 360 + 99,
            "commit_sha": self.candidate_sha,
            "config_hash": "",
            "parent_receipt_hashes": sorted(parent_hashes),
            "research_lab_allocation_receipt_hash": allocation_hash,
            "burn_target_uid": 0,
            "expected_burn_target_hotkey": "burn-hotkey",
            "metagraph_hotkeys": [
                "burn-hotkey",
                "fulfillment-hotkey",
                "lab-hotkey",
                "source-hotkey",
            ],
            "banned_hotkeys": [],
            "banned_lookup_ok": True,
            "ff_enabled": True,
            "base_burn_share": 0.0,
            "champion_share": 0.0,
            "champion_uid": None,
            "effective_champion_share": 0.0,
            "research_lab_fallback_share": 0.2,
            "research_lab_allocation_doc": allocation_doc,
            "leaderboard_bonus_share": 0.095,
            "leaderboard_rank_shares": [0.05, 0.03, 0.015],
            "leaderboard_entries": [
                {"miner_hotkey": "fulfillment-hotkey", "wins": 9}
            ],
            "leaderboard_fetch_ok": True,
            "fulfillment_share": 0.705,
            "fulfillment_rows": [
                {"hotkey": "fulfillment-hotkey", "share": 0.705}
            ],
            "fulfillment_fetch_ok": True,
            "rolling_lead_count": 0,
            "rolling_scores": [],
            "sourcing_floor_threshold": 125_000,
            "min_total_rep_for_distribution": 100,
        }
        snapshot["config_hash"] = weight_config_hash(snapshot)
        return snapshot

    def bundle(self) -> dict[str, Any]:
        preliminary = self.calculation_snapshot([], "")
        weight_config = preliminary["config_hash"]
        coordinator_boot = self._boot(
            role=COORDINATOR_ROLE,
            key=self.coordinator_key,
            config_hash=HASH,
        )
        weight_boot = self._boot(
            role=WEIGHT_ROLE,
            key=self.weight_key,
            config_hash=weight_config,
        )
        finalized_chain_state_root = sha256_json(
            {"block": preliminary["block"]}
        )
        gateway_authority_event_hash = sha256_json(
            {"epoch": self.epoch_id}
        )
        expected_roots = weight_input_output_roots_v2(
            calculation_snapshot=preliminary,
            finalized_chain_state_root=finalized_chain_state_root,
            gateway_authority_event_hash=gateway_authority_event_hash,
        )
        ordered_categories = [
            "chain_state",
            "metagraph_state",
            "burn_ownership",
            *sorted(
                set(WEIGHT_INPUT_PURPOSES)
                - {"chain_state", "metagraph_state", "burn_ownership"}
            ),
        ]
        source_receipts: list[dict[str, Any]] = []
        source_attempts: list[dict[str, Any]] = []
        input_hashes: dict[str, str] = {}
        for index, category in enumerate(ordered_categories):
            role, purpose = WEIGHT_INPUT_PURPOSES[category]
            key, boot, config_hash = (
                (self.coordinator_key, coordinator_boot, HASH)
                if role == COORDINATOR_ROLE
                else (self.weight_key, weight_boot, weight_config)
            )
            job_id = f"weight-input-{category}"
            attempt = None
            if role == COORDINATOR_ROLE and category != "anomaly_adjustments":
                attempt = self.source_attempt(
                    category=category,
                    job_id=job_id,
                    purpose=purpose,
                    sequence=index,
                    provider_id="supabase",
                    host="qplwoislplkcegvdmbim.supabase.co",
                    method="GET",
                )
            elif category in {"chain_state", "metagraph_state"}:
                attempt = self.source_attempt(
                    category=category,
                    job_id=job_id,
                    purpose=purpose,
                    sequence=index,
                    provider_id="bittensor_chain",
                    host="entrypoint-finney.opentensor.ai",
                    method="WSS",
                )
            value_document = weight_input_value_documents_v2(
                calculation_snapshot=preliminary,
                finalized_chain_state_root=finalized_chain_state_root,
                gateway_authority_event_hash=gateway_authority_event_hash,
            )[category]
            artifact_hashes = [sha256_json(value_document["value"])]
            if attempt is not None:
                source_attempts.append(attempt)
                artifact_hashes.extend(
                    [
                        attempt["request_artifact_hash"],
                        attempt["response_artifact_hash"],
                    ]
                )
            receipt = self.receipt(
                role=role,
                purpose=purpose,
                job_id=job_id,
                key=key,
                boot=boot,
                config_hash=config_hash,
                input_root=sha256_json(
                    {"category": category, "kind": "input"}
                ),
                output_root=expected_roots[category],
                parents=(
                    [input_hashes["chain_state"]]
                    if category == "metagraph_state"
                    else [input_hashes["metagraph_state"]]
                    if category == "burn_ownership"
                    else []
                ),
                sequence=index,
                transport_root=(
                    merkle_root(
                        [attempt["attempt_hash"]],
                        domain="leadpoet-transport-v2",
                    )
                    if attempt is not None
                    else EMPTY_TRANSPORT_ROOT
                ),
                artifact_root=merkle_root(
                    artifact_hashes, domain="leadpoet-artifact-v2"
                ),
            )
            source_receipts.append(receipt)
            input_hashes[category] = receipt["receipt_hash"]

        calculation = self.calculation_snapshot(
            list(input_hashes.values()),
            input_hashes["research_lab_allocation"],
        )
        snapshot = build_weight_snapshot_v2(
            validator_hotkey=VALIDATOR_HOTKEY,
            calculation_snapshot=calculation,
            input_receipt_hashes=input_hashes,
            finalized_chain_state_root=finalized_chain_state_root,
            gateway_authority_event_hash=gateway_authority_event_hash,
        )
        snapshot_receipt = self.receipt(
            role=WEIGHT_ROLE,
            purpose="validator.weight_snapshot.v2",
            job_id=f"weight-snapshot-{self.epoch_id}",
            key=self.weight_key,
            boot=weight_boot,
            config_hash=weight_config,
            input_root=snapshot["source_input_root"],
            output_root=snapshot["snapshot_hash"],
            parents=sorted(input_hashes.values()),
            sequence=100,
        )
        result = compute_final_weights(calculation)
        weight_receipt = self.receipt(
            role=WEIGHT_ROLE,
            purpose="validator.weights.computed.v2",
            job_id=f"weight-computation-{self.epoch_id}",
            key=self.weight_key,
            boot=weight_boot,
            config_hash=weight_config,
            input_root=snapshot["snapshot_hash"],
            output_root=sha256_json(result),
            parents=[snapshot_receipt["receipt_hash"]],
            sequence=101,
        )
        binding_message = create_binding_message(
            netuid=71,
            chain="wss://entrypoint-finney.opentensor.ai:443",
            enclave_pubkey=self._public_key(self.weight_key),
            validator_code_hash=weight_boot["build_manifest_hash"],
            version=self.candidate_sha,
        )
        application_request = build_application_signature_request_v2(
            message=binding_message.encode(),
            validator_hotkey=VALIDATOR_HOTKEY,
            boot_identity_hash=weight_boot["boot_identity_hash"],
        )
        hotkey_signature = "9" * 128
        binding_output = {
            "schema_version": "leadpoet.application_signature_result.v2",
            "request_hash": application_request["request_hash"],
            "purpose": "validator.gateway_binding.v2",
            "validator_hotkey": VALIDATOR_HOTKEY,
            "signature": hotkey_signature,
        }
        hotkey_receipt = self.receipt(
            role=WEIGHT_ROLE,
            purpose="validator.hotkey_signature.v2",
            job_id=f"hotkey-signature-{self.epoch_id}",
            key=self.weight_key,
            boot=weight_boot,
            config_hash=weight_config,
            input_root=application_request["request_hash"],
            output_root=sha256_json(binding_output),
            parents=[weight_receipt["receipt_hash"]],
            sequence=102,
        )
        graph = build_receipt_graph(
            root_receipt_hash=hotkey_receipt["receipt_hash"],
            boot_identities=[coordinator_boot, weight_boot],
            receipts=[
                *source_receipts,
                snapshot_receipt,
                weight_receipt,
                hotkey_receipt,
            ],
            transport_attempts=source_attempts,
        )
        return {
            "schema_version": PUBLISHED_WEIGHT_BUNDLE_V2_SCHEMA_VERSION,
            "validator_hotkey": VALIDATOR_HOTKEY,
            "binding_message": binding_message,
            "validator_hotkey_signature": hotkey_signature,
            "weight_snapshot": snapshot,
            "weight_result": result,
            "weights_signature": self.weight_key.sign(
                bytes.fromhex(result["weights_hash"])
            ).hex(),
            "receipt_graph": graph,
        }

    @staticmethod
    def identity_cache(bundle: dict[str, Any]) -> dict[str, Any]:
        from leadpoet_canonical.auditor_v2 import (
            IDENTITY_CACHE_SCHEMA_VERSION,
        )

        return {
            "schema_version": IDENTITY_CACHE_SCHEMA_VERSION,
            "entries": [
                {
                    "physical_role": boot["physical_role"],
                    "role": boot["role"],
                    "commit_sha": boot["commit_sha"],
                    "pcr0": boot["pcr0"],
                    "build_manifest_hash": boot["build_manifest_hash"],
                    "dependency_lock_hash": boot["dependency_lock_hash"],
                    "verified_build_count": 3,
                }
                for boot in bundle["receipt_graph"]["boot_identities"]
            ],
        }

    @staticmethod
    def chain_profile() -> dict[str, Any]:
        return {
            "schema_version": "leadpoet.chain_signing_profile.v2",
            "network": "finney",
            "chain_endpoint": "wss://entrypoint-finney.opentensor.ai:443",
            "genesis_hash": "0" * 64,
            "spec_version": 432,
            "transaction_version": 1,
            "version_key": 10005000,
            "commit_call_index": "0776",
            "serve_axon_call_index": "0704",
            "commit_reveal_version": 4,
            "mechid": 0,
            "tempo": 360,
            "subnet_reveal_period_epochs": 1,
            "block_time_millis": 12000,
            "max_snapshot_block_drift": 64,
            "extrinsic_period": 8,
            "signed_extensions": [
                "CheckMortality",
                "CheckNonce",
                "ChargeTransactionPayment",
                "CheckMetadataHash",
                "CheckSpecVersion",
                "CheckTxVersion",
                "CheckGenesis",
                "CheckMortalityAdditionalSigned",
                "CheckMetadataHashAdditionalSigned",
            ],
        }
