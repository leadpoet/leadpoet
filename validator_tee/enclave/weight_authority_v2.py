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
        fields = {
            "validator_hotkey",
            "calculation_snapshot",
            "input_receipt_hashes",
            "gateway_authority_event_hash",
            "upstream_receipt_set",
        }
        if not isinstance(request, Mapping) or set(request) != fields:
            raise ValidatorWeightAuthorityV2Error("weight authority request fields are invalid")
        boot = dict(self._boot_identity_supplier())
        validate_boot_identity(boot)
        if boot.get("role") != "validator_weights":
            raise ValidatorWeightAuthorityV2Error("validator V2 boot role is invalid")
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
            if receipt.get("role") == "validator_weights":
                raise ValidatorWeightAuthorityV2Error(
                    "validator-role input receipts must be created inside this enclave"
                )
        for attempt in attempts:
            validate_transport_attempt(attempt)
        receipt_by_hash = {
            str(receipt["receipt_hash"]): receipt for receipt in receipts
        }
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
                "finalized chain source is unavailable"
            ) from exc
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
            parent_receipt_hashes=sorted(input_hashes.values()),
            artifact_hashes=(snapshot["calculation_snapshot_hash"],),
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
            job_id="burn-ownership:%d" % epoch_id,
            sequence=2,
            parent_hashes=(metagraph_receipt["receipt_hash"],),
            input_root=metagraph_receipt["output_root"],
        )
        add(
            "feature_flags",
            job_id="weight-feature-flags:%d" % epoch_id,
            sequence=3,
            parent_hashes=(),
            input_root=sha256_json(
                {"boot_config_hash": boot["config_hash"], "category": "feature_flags"}
            ),
        )
        add(
            "constants",
            job_id="weight-constants:%d" % epoch_id,
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
        if physical_role == "validator_weights":
            if dict(identity) != dict(validator_boot):
                raise ValidatorWeightAuthorityV2Error("another validator boot is not trusted")
            return self._boot_verifier(identity, expected_pcr0=validator_boot["pcr0"])
        commit = str(identity.get("commit_sha") or "").lower()
        lineage = self._gateway_release_lineage_supplier()
        release = lineage.get(commit)
        if not isinstance(release, Mapping):
            raise ValidatorWeightAuthorityV2Error(
                "gateway boot commit is not in approved release lineage"
            )
        roles = release.get("roles")
        expectation = (
            roles.get(physical_role) if isinstance(roles, Mapping) else None
        )
        if not isinstance(expectation, Mapping):
            raise ValidatorWeightAuthorityV2Error(
                "gateway boot role is not in approved release lineage"
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
                "gateway boot differs from approved release lineage"
            )
        return self._boot_verifier(
            identity,
            expected_pcr0=str(expectation["pcr0"]),
            certificate_validity_at_attestation_time=True,
        )
