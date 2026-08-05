"""Authoritative V2 weight computation inside the validator enclave."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Callable, Dict, Mapping

from leadpoet_canonical.attested_v2 import (
    EMPTY_ARTIFACT_ROOT,
    EMPTY_HOST_OPERATION_ROOT,
    EMPTY_TRANSPORT_ROOT,
    build_execution_receipt_body,
    build_receipt_graph,
    create_signed_execution_receipt,
    merkle_root,
    sha256_bytes,
    sha256_json,
    validate_boot_identity,
    validate_receipt_graph,
    validate_signed_execution_receipt,
    validate_transport_attempt,
    verify_boot_identity_nitro,
)
from leadpoet_canonical.chain_source_v2 import chain_source_policy_hash
from leadpoet_canonical.ancestry_checkpoint_v2 import (
    ANCESTRY_DELTA_SCHEMA_VERSION,
    build_compact_ancestry_proof_from_delta_v2,
    derive_ancestry_lineage_id_v2,
    issue_ancestry_certificate_v2,
    validate_compact_ancestry_proof_v2,
)
from leadpoet_canonical.compact_weight_authority_v2 import (
    build_weight_ancestry_commitment_v2,
    compact_gateway_frontier_receipt_hashes_v2,
    validate_compact_weight_ancestry_v2,
)
from leadpoet_canonical.weight_authority_v2 import (
    GATEWAY_WEIGHT_INPUT_CATEGORIES,
    VALIDATOR_WEIGHT_INPUT_CATEGORIES,
    WEIGHT_INPUT_PURPOSES,
    build_weight_snapshot_v2,
    validate_weight_input_source_evidence_v2,
    weight_input_value_documents_v2,
)
from leadpoet_canonical.weight_computation import compute_final_weights


class ValidatorWeightAuthorityV2Error(RuntimeError):
    """A weight request lacks complete verified ancestry or canonical inputs."""


def _issued_at(clock: Callable[[], datetime]) -> str:
    value = clock()
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _terminal_receipt_hashes(receipts: list[Mapping[str, Any]]) -> list[str]:
    """Return every receipt not already consumed by another receipt."""

    receipt_hashes = {str(receipt["receipt_hash"]) for receipt in receipts}
    parent_hashes = {
        str(parent_hash)
        for receipt in receipts
        for parent_hash in receipt["parent_receipt_hashes"]
    }
    return sorted(receipt_hashes - parent_hashes)


class ValidatorWeightAuthorityV2:
    def __init__(
        self,
        *,
        boot_identity_supplier: Callable[[], Mapping[str, Any]],
        gateway_release_lineage_supplier: Callable[[], Mapping[str, Any]],
        sign_digest: Callable[[bytes], Any],
        chain_source: Any,
        boot_verifier: Callable[..., Mapping[str, Any]] = verify_boot_identity_nitro,
        clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ) -> None:
        self._boot_identity_supplier = boot_identity_supplier
        self._gateway_release_lineage_supplier = gateway_release_lineage_supplier
        self._sign_digest = sign_digest
        self._chain_source = chain_source
        self._boot_verifier = boot_verifier
        self._clock = clock

    def compute(self, request: Mapping[str, Any]) -> Dict[str, Any]:
        full_fields = {
            "validator_hotkey",
            "calculation_snapshot",
            "input_receipt_hashes",
            "gateway_authority_event_hash",
            "upstream_receipt_set",
        }
        compact_fields = {
            "validator_hotkey",
            "calculation_snapshot",
            "input_receipt_hashes",
            "gateway_authority_event_hash",
            "upstream_ancestry_proofs",
            "upstream_transport_attempts",
        }
        request_fields = set(request) if isinstance(request, Mapping) else set()
        if (
            not isinstance(request, Mapping)
            or request_fields not in (full_fields, compact_fields)
        ):
            raise ValidatorWeightAuthorityV2Error("weight authority request fields are invalid")
        compact_transport = request_fields == compact_fields
        boot = dict(self._boot_identity_supplier())
        validate_boot_identity(boot)
        if boot.get("role") != "validator_weights":
            raise ValidatorWeightAuthorityV2Error("validator V2 boot role is invalid")
        input_hashes = dict(request["input_receipt_hashes"])
        if set(input_hashes) != set(GATEWAY_WEIGHT_INPUT_CATEGORIES):
            raise ValidatorWeightAuthorityV2Error(
                "gateway weight input categories are incomplete"
            )
        proposed_calculation = dict(request["calculation_snapshot"])
        try:
            chain_snapshot = self._chain_source.read_finalized_snapshot(
                netuid=int(proposed_calculation["netuid"]),
                epoch_id=int(proposed_calculation["epoch_id"]),
            )
        except Exception as exc:
            raise ValidatorWeightAuthorityV2Error(
                "finalized chain source is unavailable: %s: %s"
                % (type(exc).__name__, str(exc))
            ) from exc
        if compact_transport:
            (
                boots,
                receipts,
                attempts,
                host_operations,
                gateway_frontier_hashes,
                ancestry_commitment,
                normalized_proofs,
            ) = self._verify_compact_gateway_ancestry(
                request=request,
                input_hashes=input_hashes,
                chain_snapshot=chain_snapshot,
                validator_boot=boot,
            )
        else:
            upstream = request.get("upstream_receipt_set")
            if not isinstance(upstream, Mapping) or set(upstream) != {
                "boot_identities",
                "receipts",
                "transport_attempts",
                "host_operations",
            }:
                raise ValidatorWeightAuthorityV2Error("upstream receipt set is invalid")
            boots = [dict(item) for item in upstream["boot_identities"]]
            receipts = [dict(item) for item in upstream["receipts"]]
            attempts = [dict(item) for item in upstream["transport_attempts"]]
            host_operations = [dict(item) for item in upstream["host_operations"]]
            self._verify_boots(boots, boot)
            for receipt in receipts:
                validate_signed_execution_receipt(receipt)
            for attempt in attempts:
                validate_transport_attempt(attempt)
            gateway_frontier_hashes = _terminal_receipt_hashes(receipts)
            ancestry_commitment = None
            normalized_proofs = None
        receipt_by_hash = {
            str(receipt["receipt_hash"]): receipt for receipt in receipts
        }
        direct_input_receipts = [
            receipt_by_hash.get(str(receipt_hash))
            for receipt_hash in input_hashes.values()
        ]
        if any(
            isinstance(receipt, Mapping)
            and receipt.get("role") == "validator_weights"
            for receipt in direct_input_receipts
        ):
            raise ValidatorWeightAuthorityV2Error(
                "validator-role receipts cannot supply gateway weight inputs"
            )
        calculation = dict(proposed_calculation)
        calculation["block"] = int(chain_snapshot["header"]["block"])
        calculation["metagraph_hotkeys"] = list(
            chain_snapshot["metagraph"]["hotkeys"]
        )
        expected_burn_hotkey = str(
            calculation.get("expected_burn_target_hotkey") or ""
        )
        if (
            expected_burn_hotkey
            and chain_snapshot["metagraph"]["owner_hotkey"]
            != expected_burn_hotkey
        ):
            raise ValidatorWeightAuthorityV2Error(
                "finalized subnet owner differs from configured burn owner"
            )
        try:
            compute_final_weights(calculation)
        except Exception as exc:
            raise ValidatorWeightAuthorityV2Error(
                "finalized chain state fails the current weight formula"
            ) from exc
        finalized_chain_state_root = str(
            chain_snapshot["header"]["state_root_commitment"]
        )
        input_documents = weight_input_value_documents_v2(
            calculation_snapshot=calculation,
            finalized_chain_state_root=finalized_chain_state_root,
            gateway_authority_event_hash=str(
                request["gateway_authority_event_hash"]
            ),
        )
        expected_input_roots = {
            category: sha256_json(document)
            for category, document in input_documents.items()
        }
        for category, receipt_hash in input_hashes.items():
            receipt = receipt_by_hash.get(str(receipt_hash))
            if receipt is None:
                raise ValidatorWeightAuthorityV2Error(
                    "weight input receipt is missing: %s" % category
                )
            expected_role, expected_purpose = WEIGHT_INPUT_PURPOSES[category]
            if (
                receipt.get("role") != expected_role
                or receipt.get("purpose") != expected_purpose
            ):
                raise ValidatorWeightAuthorityV2Error(
                    "weight input receipt role/purpose differs: %s" % category
                )
            if int(receipt.get("epoch_id", -1)) != int(
                request["calculation_snapshot"]["epoch_id"]
            ):
                raise ValidatorWeightAuthorityV2Error(
                    "weight input receipt epoch differs: %s" % category
                )
            if receipt.get("output_root") != expected_input_roots[category]:
                raise ValidatorWeightAuthorityV2Error(
                    "weight input receipt output differs: %s" % category
                )
            try:
                validate_weight_input_source_evidence_v2(
                    category=category,
                    receipt=receipt,
                    document=input_documents[category],
                    transport_attempts=attempts,
                )
            except Exception as exc:
                raise ValidatorWeightAuthorityV2Error(
                    "weight input source evidence differs: %s" % category
                ) from exc
        local_receipts, local_hashes = self._validator_input_receipts(
            boot=boot,
            input_documents=input_documents,
            chain_snapshot=chain_snapshot,
            epoch_id=int(calculation["epoch_id"]),
        )
        input_hashes.update(local_hashes)
        if set(input_hashes) != set(WEIGHT_INPUT_PURPOSES):
            raise ValidatorWeightAuthorityV2Error(
                "combined weight input categories are incomplete"
            )
        if compact_transport:
            ancestry_context = dict(ancestry_commitment)
            ancestry_commitment = build_weight_ancestry_commitment_v2(
                lineage_id=str(ancestry_context["lineage_id"]),
                input_receipt_hashes=input_hashes,
                upstream_ancestry_proofs=ancestry_context[
                    "normalized_proofs"
                ],
                upstream_transport_attempts=ancestry_context["attempts"],
            )
        calculation["parent_receipt_hashes"] = sorted(input_hashes.values())
        calculation["research_lab_allocation_receipt_hash"] = input_hashes[
            "research_lab_allocation"
        ]
        all_attempts = attempts + [
            dict(item) for item in chain_snapshot["attempts"]
        ]
        for category in VALIDATOR_WEIGHT_INPUT_CATEGORIES:
            local_receipt = next(
                receipt
                for receipt in local_receipts
                if receipt["receipt_hash"] == input_hashes[category]
            )
            try:
                validate_weight_input_source_evidence_v2(
                    category=category,
                    receipt=local_receipt,
                    document=input_documents[category],
                    transport_attempts=all_attempts,
                )
            except Exception as exc:
                raise ValidatorWeightAuthorityV2Error(
                    "validator weight input source evidence differs: %s"
                    % category
                ) from exc
        snapshot = build_weight_snapshot_v2(
            validator_hotkey=str(request["validator_hotkey"]),
            calculation_snapshot=calculation,
            input_receipt_hashes=input_hashes,
            finalized_chain_state_root=finalized_chain_state_root,
            gateway_authority_event_hash=str(request["gateway_authority_event_hash"]),
        )
        issued_at = _issued_at(self._clock)
        snapshot_receipt = self._receipt(
            boot=boot,
            purpose="validator.weight_snapshot.v2",
            job_id="weight-snapshot:%s:%s"
            % (snapshot["epoch_id"], snapshot["snapshot_hash"].split(":", 1)[1][:32]),
            epoch_id=snapshot["epoch_id"],
            sequence=0,
            input_root=snapshot["source_input_root"],
            output_root=snapshot["snapshot_hash"],
            # Some measured gateway inputs have a mandatory Object-Lock
            # persistence receipt after the execution receipt named in
            # input_hashes. Link the snapshot to every terminal receipt so
            # those persistence proofs remain on the path to the final root.
            parent_receipt_hashes=sorted(
                set(input_hashes.values())
                | set(gateway_frontier_hashes)
                | set(_terminal_receipt_hashes(local_receipts))
            ),
            artifact_hashes=(
                (snapshot["calculation_snapshot_hash"], ancestry_commitment)
                if ancestry_commitment is not None
                else (snapshot["calculation_snapshot_hash"],)
            ),
            issued_at=issued_at,
        )
        result = compute_final_weights(snapshot["calculation_snapshot"])
        computed_receipt = self._receipt(
            boot=boot,
            purpose="validator.weights.computed.v2",
            job_id="weight-computation:%s:%s"
            % (snapshot["epoch_id"], result["weights_hash"][:32]),
            epoch_id=snapshot["epoch_id"],
            sequence=1,
            input_root=snapshot["snapshot_hash"],
            output_root=sha256_json(result),
            parent_receipt_hashes=(snapshot_receipt["receipt_hash"],),
            artifact_hashes=(
                "sha256:" + result["weights_hash"],
                sha256_json(result["weight_float_bits"]),
                sha256_json(
                    {
                        "uids": result["sparse_uids"],
                        "weights_u16": result["sparse_weights_u16"],
                    }
                ),
            ),
            issued_at=issued_at,
        )
        complete_boots = boots + (
            []
            if any(item["boot_identity_hash"] == boot["boot_identity_hash"] for item in boots)
            else [boot]
        )
        if compact_transport:
            local_set = {
                "schema_version": "leadpoet.validator_weight_receipt_delta.v2",
                "root_receipt_hash": computed_receipt["receipt_hash"],
                "boot_identities": [dict(boot)],
                "receipts": [
                    *[dict(item) for item in local_receipts],
                    dict(snapshot_receipt),
                    dict(computed_receipt),
                ],
                "transport_attempts": [
                    dict(item) for item in chain_snapshot["attempts"]
                ],
                "host_operations": [],
            }
            weights_signature = self._sign_digest(bytes.fromhex(result["weights_hash"]))
            if not isinstance(weights_signature, (bytes, bytearray)):
                raise ValidatorWeightAuthorityV2Error("weight signer returned invalid signature")
            return {
                "weight_snapshot": snapshot,
                "weight_result": result,
                "weights_signature": bytes(weights_signature).hex(),
                "receipt_graph_delta": local_set,
                "upstream_ancestry_proofs": normalized_proofs,
                "upstream_transport_attempts": [dict(item) for item in attempts],
                "ancestry_commitment": ancestry_commitment,
                "boot_identity": boot,
                "source_artifacts": self._validated_source_artifacts(
                    chain_snapshot["artifacts"],
                    chain_snapshot["attempts"],
                ),
                "epoch_authority": (
                    dict(chain_snapshot["epoch_authority"])
                    if chain_snapshot.get("epoch_authority") is not None
                    else None
                ),
                "epoch_boundary": (
                    dict(chain_snapshot["epoch_boundary"])
                    if chain_snapshot.get("epoch_boundary") is not None
                    else None
                ),
            }
        graph = build_receipt_graph(
            root_receipt_hash=computed_receipt["receipt_hash"],
            boot_identities=complete_boots,
            receipts=receipts + local_receipts + [snapshot_receipt, computed_receipt],
            transport_attempts=all_attempts,
            host_operations=host_operations,
        )
        required_purposes = {
            purpose for _role, purpose in WEIGHT_INPUT_PURPOSES.values()
        } | {"validator.weight_snapshot.v2", "validator.weights.computed.v2"}
        if chain_snapshot.get("epoch_boundary") is not None:
            required_purposes.add("validator.subnet_epoch_snapshot.v2")
        validate_receipt_graph(
            graph,
            required_purposes=required_purposes,
            boot_attestation_verifier=lambda identity: self._verify_one_boot(
                identity, boot
            ),
            require_boot_attestation_verification=True,
        )
        weights_signature = self._sign_digest(bytes.fromhex(result["weights_hash"]))
        if not isinstance(weights_signature, (bytes, bytearray)):
            raise ValidatorWeightAuthorityV2Error("weight signer returned invalid signature")
        return {
            "weight_snapshot": snapshot,
            "weight_result": result,
            "weights_signature": bytes(weights_signature).hex(),
            "receipt_graph": graph,
            "boot_identity": boot,
            "source_artifacts": self._validated_source_artifacts(
                chain_snapshot["artifacts"],
                chain_snapshot["attempts"],
            ),
            "epoch_authority": (
                dict(chain_snapshot["epoch_authority"])
                if chain_snapshot.get("epoch_authority") is not None
                else None
            ),
            "epoch_boundary": (
                dict(chain_snapshot["epoch_boundary"])
                if chain_snapshot.get("epoch_boundary") is not None
                else None
            ),
        }

    def issue_validator_publication_ancestry_proof(
        self,
        *,
        validator_receipt_delta: Mapping[str, Any],
        upstream_ancestry_proofs: Mapping[str, Any],
        binding_receipt: Mapping[str, Any],
        epoch_authority: Mapping[str, Any],
    ) -> Dict[str, Any]:
        """Issue the validator-root proof consumed by the coordinator enclave."""

        boot = dict(self._boot_identity_supplier())
        try:
            lineage_id = derive_ancestry_lineage_id_v2(
                cutover_mapping_hash=str(epoch_authority["cutover_mapping_hash"]),
                network_genesis_hash=str(epoch_authority["network_genesis_hash"]),
                netuid=int(epoch_authority["netuid"]),
            )
        except Exception as exc:
            raise ValidatorWeightAuthorityV2Error(
                "validator ancestry cutover lineage is invalid"
            ) from exc
        if (
            not isinstance(upstream_ancestry_proofs, Mapping)
            or set(upstream_ancestry_proofs)
            != set(GATEWAY_WEIGHT_INPUT_CATEGORIES)
        ):
            raise ValidatorWeightAuthorityV2Error(
                "validator ancestry gateway proofs are incomplete"
            )

        normalized_proofs = {}
        for category in sorted(GATEWAY_WEIGHT_INPUT_CATEGORIES):
            try:
                normalized_proofs[category] = validate_compact_ancestry_proof_v2(
                    upstream_ancestry_proofs[category],
                    expected_lineage_id=lineage_id,
                    boot_attestation_verifier=lambda identity: self._verify_one_boot(
                        identity, boot
                    ),
                    allowed_issuer_roles={"gateway_coordinator"},
                )
            except Exception as exc:
                raise ValidatorWeightAuthorityV2Error(
                    "validator ancestry gateway proof failed: %s" % category
                ) from exc
        frontier_roots = compact_gateway_frontier_receipt_hashes_v2(
            normalized_proofs
        )
        certificates_by_root: Dict[str, Dict[str, Any]] = {}
        for proof in normalized_proofs.values():
            certificate = dict(proof["certificate"])
            root = str(certificate["claim"]["output_root_receipt_hash"])
            prior = certificates_by_root.get(root)
            if prior is not None and prior != certificate:
                raise ValidatorWeightAuthorityV2Error(
                    "validator ancestry gateway certificate conflicts"
                )
            certificates_by_root[root] = certificate
        if not set(frontier_roots).issubset(certificates_by_root):
            raise ValidatorWeightAuthorityV2Error(
                "validator ancestry frontier is incomplete"
            )

        delta = validator_receipt_delta
        required_delta_fields = {
            "schema_version",
            "root_receipt_hash",
            "boot_identities",
            "receipts",
            "transport_attempts",
            "host_operations",
        }
        if (
            not isinstance(delta, Mapping)
            or set(delta) != required_delta_fields
            or delta.get("schema_version")
            != "leadpoet.validator_weight_receipt_delta.v2"
            or delta.get("boot_identities") != [boot]
            or not isinstance(delta.get("receipts"), list)
            or not isinstance(delta.get("transport_attempts"), list)
            or delta.get("host_operations") != []
        ):
            raise ValidatorWeightAuthorityV2Error(
                "validator ancestry local delta is invalid"
            )
        validate_signed_execution_receipt(binding_receipt)
        if (
            binding_receipt.get("role") != "validator_weights"
            or binding_receipt.get("purpose")
            != "validator.hotkey_signature.v2"
            or binding_receipt.get("boot_identity_hash")
            != boot["boot_identity_hash"]
            or binding_receipt.get("parent_receipt_hashes")
            != [delta["root_receipt_hash"]]
        ):
            raise ValidatorWeightAuthorityV2Error(
                "validator ancestry binding receipt differs"
            )
        local_delta = {
            "schema_version": ANCESTRY_DELTA_SCHEMA_VERSION,
            "root_receipt_hash": str(binding_receipt["receipt_hash"]),
            "boot_identities": [boot],
            "receipts": [
                *[dict(item) for item in delta["receipts"]],
                dict(binding_receipt),
            ],
            "transport_attempts": [
                dict(item) for item in delta["transport_attempts"]
            ],
            "host_operations": [],
        }
        parent_certificates = [
            certificates_by_root[root] for root in sorted(frontier_roots)
        ]
        local_receipt_hashes = {
            str(item["receipt_hash"]) for item in local_delta["receipts"]
        }
        external_parent_hashes = {
            str(parent_hash)
            for item in local_delta["receipts"]
            for parent_hash in item["parent_receipt_hashes"]
            if str(parent_hash) not in local_receipt_hashes
        }
        disclosure_parents = []
        for parent_hash in sorted(
            external_parent_hashes - set(frontier_roots)
        ):
            candidates = {
                str(proof["proof_hash"]): proof
                for proof in normalized_proofs.values()
                if parent_hash
                in set(
                    proof["certificate"]["claim"][
                        "local_delta_projection"
                    ]["receipt_hashes"]
                )
            }
            if len(candidates) != 1:
                raise ValidatorWeightAuthorityV2Error(
                    "validator ancestry direct input authority is ambiguous"
                )
            disclosure_parents.append(
                (next(iter(candidates.values())), parent_hash)
            )
        if external_parent_hashes != (
            set(frontier_roots)
            | {item[1] for item in disclosure_parents}
        ):
            raise ValidatorWeightAuthorityV2Error(
                "validator ancestry parent authority is incomplete"
            )
        parent_sequences = [
            int(item["claim"]["certificate_sequence"])
            for item in parent_certificates
        ] + [
            int(item[0]["certificate"]["claim"]["certificate_sequence"])
            for item in disclosure_parents
        ]
        verifier = lambda identity: self._verify_one_boot(identity, boot)
        try:
            certificate = issue_ancestry_certificate_v2(
                local_delta=local_delta,
                lineage_id=lineage_id,
                certificate_sequence=(
                    max(parent_sequences) + 1 if parent_sequences else 0
                ),
                issuer_boot_identity=boot,
                issued_at=_issued_at(self._clock),
                sign_digest=self._sign_digest,
                boot_attestation_verifier=verifier,
                allowed_issuer_roles={
                    "gateway_coordinator",
                    "validator_weights",
                },
                parent_certificates=parent_certificates,
                parent_proof_disclosures=disclosure_parents,
                required_purposes={
                    "validator.weight_snapshot.v2",
                    "validator.weights.computed.v2",
                    "validator.hotkey_signature.v2",
                },
            )
            return build_compact_ancestry_proof_from_delta_v2(
                local_delta,
                certificate,
                expected_lineage_id=lineage_id,
                boot_attestation_verifier=verifier,
                allowed_issuer_roles={
                    "gateway_coordinator",
                    "validator_weights",
                },
            )
        except Exception as exc:
            raise ValidatorWeightAuthorityV2Error(
                "validator ancestry proof issuance failed closed"
            ) from exc

    def issue_validator_finalization_ancestry_proof(
        self,
        *,
        validator_receipt_delta: Mapping[str, Any],
        publication_ancestry_proof: Mapping[str, Any],
        epoch_authority: Mapping[str, Any],
    ) -> Dict[str, Any]:
        """Issue a bounded proof extending the published computed receipt."""

        boot = dict(self._boot_identity_supplier())
        try:
            lineage_id = derive_ancestry_lineage_id_v2(
                cutover_mapping_hash=str(epoch_authority["cutover_mapping_hash"]),
                network_genesis_hash=str(epoch_authority["network_genesis_hash"]),
                netuid=int(epoch_authority["netuid"]),
            )
        except Exception as exc:
            raise ValidatorWeightAuthorityV2Error(
                "validator finalization lineage is invalid"
            ) from exc
        verifier = lambda identity: self._verify_one_boot(identity, boot)
        try:
            parent_proof = validate_compact_ancestry_proof_v2(
                publication_ancestry_proof,
                expected_lineage_id=lineage_id,
                boot_attestation_verifier=verifier,
                allowed_issuer_roles={"validator_weights"},
                required_purposes={"validator.weights.computed.v2"},
            )
        except Exception as exc:
            raise ValidatorWeightAuthorityV2Error(
                "validator finalization predecessor proof is invalid"
            ) from exc
        delta = validator_receipt_delta
        required_fields = {
            "schema_version",
            "root_receipt_hash",
            "boot_identities",
            "receipts",
            "transport_attempts",
            "host_operations",
        }
        if (
            not isinstance(delta, Mapping)
            or set(delta) != required_fields
            or delta.get("schema_version")
            != "leadpoet.validator_weight_finalization_delta.v2"
            or not isinstance(delta.get("boot_identities"), list)
            or not isinstance(delta.get("receipts"), list)
            or not isinstance(delta.get("transport_attempts"), list)
            or delta.get("host_operations") != []
        ):
            raise ValidatorWeightAuthorityV2Error(
                "validator finalization local delta is invalid"
            )
        receipt_hashes = {
            str(item.get("receipt_hash") or "")
            for item in delta["receipts"]
            if isinstance(item, Mapping)
        }
        external_parents = {
            str(parent_hash)
            for item in delta["receipts"]
            if isinstance(item, Mapping)
            for parent_hash in item.get("parent_receipt_hashes") or ()
            if str(parent_hash) not in receipt_hashes
        }
        parent_receipts = {
            str(item["receipt_hash"]): item
            for item in parent_proof["disclosed_receipts"]
            if item.get("purpose") == "validator.weights.computed.v2"
        }
        if (
            len(external_parents) != 1
            or not external_parents.issubset(parent_receipts)
        ):
            raise ValidatorWeightAuthorityV2Error(
                "validator finalization predecessor receipt is ambiguous"
            )
        computed_root = next(iter(external_parents))
        root_receipts = [
            item
            for item in delta["receipts"]
            if isinstance(item, Mapping)
            and item.get("receipt_hash") == delta["root_receipt_hash"]
            and item.get("purpose") == "validator.weights.finalized.v2"
        ]
        purposes = {
            str(item.get("purpose") or "")
            for item in delta["receipts"]
            if isinstance(item, Mapping)
        }
        if (
            len(root_receipts) != 1
            or "validator.set_weights_extrinsic.v2" not in purposes
            or not delta["boot_identities"]
        ):
            raise ValidatorWeightAuthorityV2Error(
                "validator finalization receipt set is incomplete"
            )
        local_delta = {
            "schema_version": ANCESTRY_DELTA_SCHEMA_VERSION,
            "root_receipt_hash": str(delta["root_receipt_hash"]),
            "boot_identities": [
                dict(item) for item in delta["boot_identities"]
            ],
            "receipts": [dict(item) for item in delta["receipts"]],
            "transport_attempts": [
                dict(item) for item in delta["transport_attempts"]
            ],
            "host_operations": [],
        }
        try:
            certificate = issue_ancestry_certificate_v2(
                local_delta=local_delta,
                lineage_id=lineage_id,
                certificate_sequence=int(
                    parent_proof["certificate"]["claim"][
                        "certificate_sequence"
                    ]
                )
                + 1,
                issuer_boot_identity=boot,
                issued_at=str(root_receipts[0]["issued_at"]),
                sign_digest=self._sign_digest,
                boot_attestation_verifier=verifier,
                allowed_issuer_roles={"validator_weights"},
                parent_proof_disclosures=((parent_proof, computed_root),),
                required_purposes={
                    "validator.set_weights_extrinsic.v2",
                    "validator.weights.finalized.v2",
                },
            )
            return build_compact_ancestry_proof_from_delta_v2(
                local_delta,
                certificate,
                expected_lineage_id=lineage_id,
                boot_attestation_verifier=verifier,
                allowed_issuer_roles={"validator_weights"},
            )
        except Exception as exc:
            raise ValidatorWeightAuthorityV2Error(
                "validator finalization ancestry proof issuance failed closed"
            ) from exc

    def capture_epoch_boundary(self, request: Mapping[str, Any]) -> Dict[str, Any]:
        """Create an independently publishable cutover candidate graph.

        This explicit operator/shadow command does not change the measured
        runtime's active epoch mode.  It signs only after the enclave has
        authenticated the proposed manifest and its predecessor transition
        against exact finalized Subtensor state.
        """

        if not isinstance(request, Mapping) or set(request) != {
            "cutover_manifest",
            "settlement_epoch_id",
        }:
            raise ValidatorWeightAuthorityV2Error(
                "subnet epoch boundary capture request is invalid"
            )
        settlement_epoch_id = request.get("settlement_epoch_id")
        if (
            not isinstance(settlement_epoch_id, int)
            or isinstance(settlement_epoch_id, bool)
            or settlement_epoch_id < 0
        ):
            raise ValidatorWeightAuthorityV2Error(
                "subnet epoch boundary settlement id is invalid"
            )
        manifest = request.get("cutover_manifest")
        if not isinstance(manifest, Mapping):
            raise ValidatorWeightAuthorityV2Error(
                "subnet epoch cutover manifest is invalid"
            )
        boot = dict(self._boot_identity_supplier())
        validate_boot_identity(boot)
        if boot.get("role") != "validator_weights":
            raise ValidatorWeightAuthorityV2Error(
                "validator V2 boot role is invalid"
            )
        try:
            capture = self._chain_source.capture_stateful_epoch_boundary(
                cutover_manifest=dict(manifest),
                settlement_epoch_id=settlement_epoch_id,
                capture_scope=str(boot["boot_identity_hash"]),
            )
        except Exception as exc:
            raise ValidatorWeightAuthorityV2Error(
                "finalized subnet epoch boundary capture failed closed"
            ) from exc
        current = capture.get("epoch_authority")
        boundary = capture.get("epoch_boundary")
        attempts = [dict(item) for item in capture.get("attempts") or []]
        jobs = capture.get("jobs")
        if (
            not isinstance(current, Mapping)
            or not isinstance(boundary, Mapping)
            or not isinstance(jobs, Mapping)
        ):
            raise ValidatorWeightAuthorityV2Error(
                "captured subnet epoch documents are invalid"
            )
        purpose = "validator.subnet_epoch_snapshot.v2"

        def scoped_receipt(
            *,
            document: Mapping[str, Any],
            job_id: str,
            parent_hashes: Any,
            input_root: str,
        ) -> Dict[str, Any]:
            scoped_attempts = [
                attempt
                for attempt in attempts
                if attempt.get("job_id") == job_id
                and attempt.get("purpose") == purpose
            ]
            if not scoped_attempts:
                raise ValidatorWeightAuthorityV2Error(
                    "captured subnet epoch source evidence is unavailable"
                )
            artifact_hashes = [sha256_json(document)]
            for attempt in scoped_attempts:
                artifact_hashes.append(str(attempt["request_artifact_hash"]))
                if attempt.get("terminal_status") == "authenticated_response":
                    artifact_hashes.append(str(attempt["response_artifact_hash"]))
            return self._receipt(
                boot=boot,
                purpose=purpose,
                job_id=job_id,
                epoch_id=settlement_epoch_id,
                sequence=0,
                input_root=input_root,
                output_root=sha256_json(document),
                parent_receipt_hashes=parent_hashes,
                artifact_hashes=artifact_hashes,
                issued_at=str(document["observed_at"]),
                transport_attempts=scoped_attempts,
                artifact_domain="leadpoet-artifact-v2",
            )

        boundary_job = str(jobs.get("subnet_epoch_boundary") or "")
        current_job = str(jobs.get("subnet_epoch_snapshot") or "")
        if (
            not boundary_job
            or not current_job
            or boundary_job != current_job
            or dict(current) != dict(boundary)
        ):
            raise ValidatorWeightAuthorityV2Error(
                "candidate capture is not one exact finalized boundary"
            )
        boundary_receipt = scoped_receipt(
            document=boundary,
            job_id=boundary_job,
            parent_hashes=(),
            input_root=sha256_json(
                {
                    "policy_hash": chain_source_policy_hash(),
                    "boundary_block_hash": boundary["block_hash"],
                    "cutover_mapping_hash": boundary["cutover_mapping_hash"],
                    "finalized_block_hash": capture["finalized_block_hash"],
                }
            ),
        )
        graph = build_receipt_graph(
            root_receipt_hash=boundary_receipt["receipt_hash"],
            boot_identities=[boot],
            receipts=[boundary_receipt],
            transport_attempts=attempts,
            host_operations=[],
        )
        validate_receipt_graph(
            graph,
            required_purposes={purpose},
            boot_attestation_verifier=lambda identity: self._verify_one_boot(
                identity, boot
            ),
            require_boot_attestation_verification=True,
        )
        return {
            "schema_version": "leadpoet.subnet_epoch_boundary_capture.v1",
            "epoch_authority": dict(current),
            "epoch_boundary": dict(boundary),
            "epoch_authority_receipt_hash": boundary_receipt["receipt_hash"],
            "epoch_boundary_receipt_hash": boundary_receipt["receipt_hash"],
            "receipt_graph": graph,
            "boot_identity": boot,
            "source_artifacts": self._validated_source_artifacts(
                capture["artifacts"],
                attempts,
            ),
        }

    def validate_compact_weight_recovery(
        self,
        compact_submission: Mapping[str, Any],
    ) -> Dict[str, Any]:
        """Re-verify checkpoint ancestry before hotkey-state recovery."""

        epoch_authority = (
            compact_submission.get("epoch_authority")
            if isinstance(compact_submission, Mapping)
            else None
        )
        if not isinstance(epoch_authority, Mapping):
            raise ValidatorWeightAuthorityV2Error(
                "compact recovery requires stateful epoch authority"
            )
        try:
            lineage_id = derive_ancestry_lineage_id_v2(
                cutover_mapping_hash=str(
                    epoch_authority["cutover_mapping_hash"]
                ),
                network_genesis_hash=str(
                    epoch_authority["network_genesis_hash"]
                ),
                netuid=int(epoch_authority["netuid"]),
            )
            boot = dict(self._boot_identity_supplier())
            return validate_compact_weight_ancestry_v2(
                compact_submission,
                expected_lineage_id=lineage_id,
                boot_attestation_verifier=lambda identity: self._verify_one_boot(
                    identity,
                    boot,
                ),
            )
        except Exception as exc:
            raise ValidatorWeightAuthorityV2Error(
                "compact recovery ancestry failed closed"
            ) from exc

    def _validator_input_receipts(
        self,
        *,
        boot: Mapping[str, Any],
        input_documents: Mapping[str, Mapping[str, Any]],
        chain_snapshot: Mapping[str, Any],
        epoch_id: int,
    ) -> Any:
        attempts = [dict(item) for item in chain_snapshot["attempts"]]
        jobs = dict(chain_snapshot["jobs"])
        observation_scope = str(
            chain_snapshot.get("observation_scope") or ""
        ).lower()
        if (
            len(observation_scope) != 32
            or any(
                character not in "0123456789abcdef"
                for character in observation_scope
            )
        ):
            raise ValidatorWeightAuthorityV2Error(
                "chain observation scope is invalid"
            )
        issued_at = _issued_at(self._clock)
        receipts = []
        hashes = {}

        def add(
            category: str,
            *,
            job_id: str,
            sequence: int,
            parent_hashes: Any,
            input_root: str,
        ) -> Dict[str, Any]:
            purpose = WEIGHT_INPUT_PURPOSES[category][1]
            scoped_attempts = [
                attempt
                for attempt in attempts
                if attempt.get("job_id") == job_id
                and attempt.get("purpose") == purpose
            ]
            artifact_hashes = [sha256_json(input_documents[category]["value"])]
            for attempt in scoped_attempts:
                artifact_hashes.append(str(attempt["request_artifact_hash"]))
                if attempt.get("terminal_status") == "authenticated_response":
                    artifact_hashes.append(str(attempt["response_artifact_hash"]))
            receipt = self._receipt(
                boot=boot,
                purpose=purpose,
                job_id=job_id,
                epoch_id=epoch_id,
                sequence=sequence,
                input_root=input_root,
                output_root=sha256_json(input_documents[category]),
                parent_receipt_hashes=parent_hashes,
                artifact_hashes=artifact_hashes,
                issued_at=issued_at,
                transport_attempts=scoped_attempts,
                artifact_domain="leadpoet-artifact-v2",
            )
            receipts.append(receipt)
            hashes[category] = receipt["receipt_hash"]
            return receipt

        chain_parent_hashes = ()
        authority_snapshot = chain_snapshot.get("epoch_authority")
        boundary_snapshot = chain_snapshot.get("epoch_boundary")
        if (authority_snapshot is None) != (boundary_snapshot is None):
            raise ValidatorWeightAuthorityV2Error(
                "subnet epoch snapshot documents are incomplete"
            )
        if authority_snapshot is not None:
            if not isinstance(authority_snapshot, Mapping) or not isinstance(
                boundary_snapshot, Mapping
            ):
                raise ValidatorWeightAuthorityV2Error(
                    "subnet epoch snapshot documents are invalid"
                )
            authority_job = jobs.get("subnet_epoch_snapshot")
            boundary_job = jobs.get("subnet_epoch_boundary")
            if (
                not isinstance(authority_job, str)
                or not authority_job
                or not isinstance(boundary_job, str)
                or not boundary_job
                or authority_job == boundary_job
            ):
                raise ValidatorWeightAuthorityV2Error(
                    "subnet epoch snapshot jobs are unavailable"
                )
            boundary_purpose = "validator.subnet_epoch_snapshot.v2"
            boundary_attempts = [
                attempt
                for attempt in attempts
                if attempt.get("job_id") == boundary_job
                and attempt.get("purpose") == boundary_purpose
            ]
            if not boundary_attempts:
                raise ValidatorWeightAuthorityV2Error(
                    "subnet epoch boundary source evidence is unavailable"
                )
            boundary_hash = sha256_json(boundary_snapshot)
            boundary_artifacts = [boundary_hash]
            for attempt in boundary_attempts:
                boundary_artifacts.append(str(attempt["request_artifact_hash"]))
                if attempt.get("terminal_status") == "authenticated_response":
                    boundary_artifacts.append(str(attempt["response_artifact_hash"]))
            boundary_receipt = self._receipt(
                boot=boot,
                purpose=boundary_purpose,
                job_id=boundary_job,
                epoch_id=epoch_id,
                sequence=0,
                input_root=sha256_json(
                    {
                        "policy_hash": chain_source_policy_hash(),
                        "finalized_head_hash": chain_snapshot[
                            "finalized_block_hash"
                        ],
                        "boundary_block_hash": boundary_snapshot["block_hash"],
                    }
                ),
                output_root=boundary_hash,
                parent_receipt_hashes=(),
                artifact_hashes=boundary_artifacts,
                issued_at=str(boundary_snapshot["observed_at"]),
                transport_attempts=boundary_attempts,
                artifact_domain="leadpoet-artifact-v2",
            )
            receipts.append(boundary_receipt)

            authority_attempts = [
                attempt
                for attempt in attempts
                if attempt.get("job_id") == authority_job
                and attempt.get("purpose") == boundary_purpose
            ]
            if not authority_attempts:
                raise ValidatorWeightAuthorityV2Error(
                    "subnet epoch current source evidence is unavailable"
                )
            authority_hash = sha256_json(authority_snapshot)
            authority_artifacts = [authority_hash]
            for attempt in authority_attempts:
                authority_artifacts.append(str(attempt["request_artifact_hash"]))
                if attempt.get("terminal_status") == "authenticated_response":
                    authority_artifacts.append(str(attempt["response_artifact_hash"]))
            authority_receipt = self._receipt(
                boot=boot,
                purpose=boundary_purpose,
                job_id=authority_job,
                epoch_id=epoch_id,
                sequence=0,
                input_root=sha256_json(
                    {
                        "policy_hash": chain_source_policy_hash(),
                        "finalized_block_hash": chain_snapshot[
                            "finalized_block_hash"
                        ],
                        "snapshot_block_hash": authority_snapshot["block_hash"],
                    }
                ),
                output_root=authority_hash,
                parent_receipt_hashes=(boundary_receipt["receipt_hash"],),
                artifact_hashes=authority_artifacts,
                issued_at=str(authority_snapshot["observed_at"]),
                transport_attempts=authority_attempts,
                artifact_domain="leadpoet-artifact-v2",
            )
            receipts.append(authority_receipt)
            chain_parent_hashes = (authority_receipt["receipt_hash"],)

        chain_receipt = add(
            "chain_state",
            job_id=str(jobs["chain_state"]),
            sequence=0,
            parent_hashes=chain_parent_hashes,
            input_root=sha256_json(
                {
                    "policy_hash": chain_source_policy_hash(),
                    "finalized_block_hash": chain_snapshot["finalized_block_hash"],
                }
            ),
        )
        metagraph_receipt = add(
            "metagraph_state",
            job_id=str(jobs["metagraph_state"]),
            sequence=1,
            parent_hashes=(chain_receipt["receipt_hash"],),
            input_root=chain_receipt["output_root"],
        )
        add(
            "burn_ownership",
            job_id="burn-ownership:%d:%s" % (epoch_id, observation_scope),
            sequence=2,
            parent_hashes=(metagraph_receipt["receipt_hash"],),
            input_root=metagraph_receipt["output_root"],
        )
        add(
            "feature_flags",
            job_id="weight-feature-flags:%d:%s"
            % (epoch_id, observation_scope),
            sequence=3,
            parent_hashes=(),
            input_root=sha256_json(
                {"boot_config_hash": boot["config_hash"], "category": "feature_flags"}
            ),
        )
        add(
            "constants",
            job_id="weight-constants:%d:%s" % (epoch_id, observation_scope),
            sequence=4,
            parent_hashes=(),
            input_root=sha256_json(
                {"boot_config_hash": boot["config_hash"], "category": "constants"}
            ),
        )
        return receipts, hashes

    @staticmethod
    def _validated_source_artifacts(value: Any, attempts: Any) -> Any:
        if not isinstance(value, list):
            raise ValidatorWeightAuthorityV2Error("chain source artifacts are invalid")
        by_hash = {}
        for item in value:
            if not isinstance(item, Mapping) or set(item) != {
                "artifact_hash",
                "kind",
                "body_b64",
            }:
                raise ValidatorWeightAuthorityV2Error(
                    "chain source artifact fields are invalid"
                )
            import base64

            try:
                body = base64.b64decode(str(item["body_b64"]), validate=True)
            except Exception as exc:
                raise ValidatorWeightAuthorityV2Error(
                    "chain source artifact body is invalid"
                ) from exc
            if sha256_bytes(body) != item["artifact_hash"]:
                raise ValidatorWeightAuthorityV2Error(
                    "chain source artifact hash differs"
                )
            by_hash[str(item["artifact_hash"])] = dict(item)
        required_hashes = set()
        for attempt in attempts:
            required_hashes.add(str(attempt["request_artifact_hash"]))
            if attempt.get("terminal_status") == "authenticated_response":
                required_hashes.add(str(attempt["response_artifact_hash"]))
        if set(by_hash) != required_hashes:
            raise ValidatorWeightAuthorityV2Error(
                "chain source artifact set is incomplete or contains extras"
            )
        return [by_hash[key] for key in sorted(by_hash)]

    def _verify_compact_gateway_ancestry(
        self,
        *,
        request: Mapping[str, Any],
        input_hashes: Mapping[str, Any],
        chain_snapshot: Mapping[str, Any],
        validator_boot: Mapping[str, Any],
    ) -> Any:
        """Verify bounded gateway checkpoints against finalized cutover state."""

        epoch_authority = chain_snapshot.get("epoch_authority")
        if not isinstance(epoch_authority, Mapping):
            raise ValidatorWeightAuthorityV2Error(
                "compact ancestry requires stateful subnet epoch authority"
            )
        try:
            lineage_id = derive_ancestry_lineage_id_v2(
                cutover_mapping_hash=str(
                    epoch_authority["cutover_mapping_hash"]
                ),
                network_genesis_hash=str(
                    epoch_authority["network_genesis_hash"]
                ),
                netuid=int(epoch_authority["netuid"]),
            )
        except Exception as exc:
            raise ValidatorWeightAuthorityV2Error(
                "compact ancestry cutover lineage is invalid"
            ) from exc
        proofs_value = request.get("upstream_ancestry_proofs")
        if not isinstance(proofs_value, Mapping) or set(proofs_value) != set(
            GATEWAY_WEIGHT_INPUT_CATEGORIES
        ):
            raise ValidatorWeightAuthorityV2Error(
                "compact gateway ancestry categories are incomplete"
            )
        boot_by_hash: dict[str, dict[str, Any]] = {}
        receipt_by_hash: dict[str, dict[str, Any]] = {}
        normalized_proofs: dict[str, dict[str, Any]] = {}
        direct_scopes = set()
        allocation_authority_observed = False

        def verify_gateway_boot(identity: Mapping[str, Any]) -> Mapping[str, Any]:
            return self._verify_one_boot(identity, validator_boot)

        for category in sorted(GATEWAY_WEIGHT_INPUT_CATEGORIES):
            direct_hash = str(input_hashes[category])
            expected_role, expected_purpose = WEIGHT_INPUT_PURPOSES[category]
            try:
                proof = validate_compact_ancestry_proof_v2(
                    proofs_value[category],
                    expected_lineage_id=lineage_id,
                    boot_attestation_verifier=verify_gateway_boot,
                    allowed_issuer_roles={"gateway_coordinator"},
                    required_receipt_hashes={direct_hash},
                    required_purposes={expected_purpose},
                )
            except Exception as exc:
                raise ValidatorWeightAuthorityV2Error(
                    "compact gateway ancestry failed for %s" % category
                ) from exc
            certificate = proof["certificate"]
            claim = certificate["claim"]
            for identity in proof["disclosed_boot_identities"]:
                key = str(identity["boot_identity_hash"])
                normalized = dict(identity)
                if key in boot_by_hash and boot_by_hash[key] != normalized:
                    raise ValidatorWeightAuthorityV2Error(
                        "compact gateway boot identity conflicts"
                    )
                boot_by_hash[key] = normalized
            direct_receipt = None
            for receipt in proof["disclosed_receipts"]:
                key = str(receipt["receipt_hash"])
                normalized = dict(receipt)
                if key in receipt_by_hash and receipt_by_hash[key] != normalized:
                    raise ValidatorWeightAuthorityV2Error(
                        "compact gateway receipt hash conflicts"
                    )
                receipt_by_hash[key] = normalized
                if key == direct_hash:
                    direct_receipt = normalized
            if (
                not isinstance(direct_receipt, Mapping)
                or direct_receipt.get("role") != expected_role
                or direct_receipt.get("purpose") != expected_purpose
                or int(direct_receipt.get("epoch_id", -1))
                != int(request["calculation_snapshot"]["epoch_id"])
            ):
                raise ValidatorWeightAuthorityV2Error(
                    "compact gateway direct receipt differs for %s" % category
                )
            direct_scopes.add(
                (str(direct_receipt["job_id"]), str(direct_receipt["purpose"]))
            )
            if category == "research_lab_allocation":
                allocation_authority_observed = any(
                    item.get("parent_receipt_hash")
                    == request["gateway_authority_event_hash"]
                    for item in claim["parent_authorities"]
                )
            normalized_proofs[category] = proof
        if not allocation_authority_observed:
            raise ValidatorWeightAuthorityV2Error(
                "compact ancestry omits the allocation authority parent"
            )

        attempts_value = request.get("upstream_transport_attempts")
        if not isinstance(attempts_value, list):
            raise ValidatorWeightAuthorityV2Error(
                "compact gateway transport evidence is invalid"
            )
        attempt_by_hash: dict[str, dict[str, Any]] = {}
        for attempt in attempts_value:
            try:
                validate_transport_attempt(attempt)
            except Exception as exc:
                raise ValidatorWeightAuthorityV2Error(
                    "compact gateway transport attempt is invalid"
                ) from exc
            scope = (str(attempt["job_id"]), str(attempt["purpose"]))
            if scope not in direct_scopes:
                raise ValidatorWeightAuthorityV2Error(
                    "compact gateway transport attempt is out of scope"
                )
            key = str(attempt["attempt_hash"])
            normalized = dict(attempt)
            if key in attempt_by_hash and attempt_by_hash[key] != normalized:
                raise ValidatorWeightAuthorityV2Error(
                    "compact gateway transport attempt hash conflicts"
                )
            attempt_by_hash[key] = normalized
        frontier_hashes = compact_gateway_frontier_receipt_hashes_v2(
            normalized_proofs
        )
        # The commitment covers the final combined input map.  Validator-owned
        # input hashes are added later, so use the canonical builder only after
        # those receipts exist.
        commitment = {
            "lineage_id": lineage_id,
            "normalized_proofs": normalized_proofs,
            "attempts": [attempt_by_hash[key] for key in sorted(attempt_by_hash)],
        }
        return (
            [boot_by_hash[key] for key in sorted(boot_by_hash)],
            [receipt_by_hash[key] for key in sorted(receipt_by_hash)],
            [attempt_by_hash[key] for key in sorted(attempt_by_hash)],
            [],
            sorted(frontier_hashes),
            commitment,
            normalized_proofs,
        )

    def _receipt(
        self,
        *,
        boot: Mapping[str, Any],
        purpose: str,
        job_id: str,
        epoch_id: int,
        sequence: int,
        input_root: str,
        output_root: str,
        parent_receipt_hashes: Any,
        artifact_hashes: Any,
        issued_at: str,
        transport_attempts: Any = (),
        artifact_domain: str = "leadpoet-validator-weight-artifact-v2",
    ) -> Dict[str, Any]:
        scoped_attempts = [dict(item) for item in transport_attempts]
        body = build_execution_receipt_body(
            role="validator_weights",
            purpose=purpose,
            job_id=job_id,
            epoch_id=epoch_id,
            sequence=sequence,
            commit_sha=boot["commit_sha"],
            pcr0=boot["pcr0"],
            build_manifest_hash=boot["build_manifest_hash"],
            dependency_lock_hash=boot["dependency_lock_hash"],
            config_hash=boot["config_hash"],
            boot_identity_hash=boot["boot_identity_hash"],
            input_root=input_root,
            output_root=output_root,
            transport_root_hash=(
                merkle_root(
                    [attempt["attempt_hash"] for attempt in scoped_attempts],
                    domain="leadpoet-transport-v2",
                )
                if scoped_attempts
                else EMPTY_TRANSPORT_ROOT
            ),
            host_operation_root_hash=EMPTY_HOST_OPERATION_ROOT,
            artifact_root=merkle_root(
                artifact_hashes,
                domain=artifact_domain,
            ),
            parent_receipt_hashes=parent_receipt_hashes,
            status="succeeded",
            failure_code=None,
            issued_at=issued_at,
        )
        return create_signed_execution_receipt(
            body=body,
            enclave_pubkey=boot["signing_pubkey"],
            sign_digest=self._sign_digest,
        )

    def _verify_boots(
        self,
        identities: Any,
        validator_boot: Mapping[str, Any],
    ) -> None:
        seen = set()
        for identity in identities:
            validate_boot_identity(identity)
            identity_hash = identity["boot_identity_hash"]
            if identity_hash in seen:
                raise ValidatorWeightAuthorityV2Error("upstream boot identity is duplicated")
            seen.add(identity_hash)
            self._verify_one_boot(identity, validator_boot)

    def _verify_one_boot(
        self,
        identity: Mapping[str, Any],
        validator_boot: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        physical_role = str(identity.get("physical_role") or "")
        if physical_role == "validator_weights" and dict(identity) == dict(
            validator_boot
        ):
            return self._boot_verifier(
                identity,
                expected_pcr0=validator_boot["pcr0"],
                certificate_validity_at_attestation_time=True,
            )
        commit = str(identity.get("commit_sha") or "").lower()
        lineage = self._gateway_release_lineage_supplier()
        release = lineage.get(commit)
        if not isinstance(release, Mapping):
            raise ValidatorWeightAuthorityV2Error(
                "boot commit is not in approved release lineage"
            )
        roles = release.get("roles")
        expectation = (
            roles.get(physical_role) if isinstance(roles, Mapping) else None
        )
        if not isinstance(expectation, Mapping):
            raise ValidatorWeightAuthorityV2Error(
                "boot role is not in approved release lineage"
            )
        if (
            identity.get("commit_sha") != expectation.get("commit_sha")
            or identity.get("build_manifest_hash")
            != expectation.get("build_manifest_hash")
            or identity.get("dependency_lock_hash")
            != expectation.get("dependency_lock_hash")
            or identity.get("pcr0") != expectation.get("pcr0")
        ):
            raise ValidatorWeightAuthorityV2Error(
                "boot differs from approved release lineage"
            )
        return self._boot_verifier(
            identity,
            expected_pcr0=str(expectation["pcr0"]),
            certificate_validity_at_attestation_time=True,
        )
