"""Measured coordinator entrypoints for unchanged Research Lab decisions."""

from __future__ import annotations

import re
from typing import Any, Callable, Iterable, Mapping, Optional

from gateway.research_lab.config import ResearchLabGatewayConfig

from gateway.tee.execution_job_manager_v2 import (
    ExecutionContextV2,
    ExecutionResultV2,
)
from gateway.tee.scoring_executor import (
    OP_PROMOTION_GATE_DECISION,
    OP_PROMOTION_IMPROVEMENT,
    OP_RESEARCH_LAB_ALLOCATION,
    ScoringExecutionResult,
    execute_scoring_operation,
)
from leadpoet_canonical.attested_v2 import sha256_json
from leadpoet_canonical.weight_authority_v2 import (
    WEIGHT_INPUT_PURPOSES,
)
from gateway.tee.reward_executor_v2 import (
    OP_RESEARCH_LAB_REWARD_DECISION,
    execute_reward_decision_v2,
    reward_receipt_projection_v2,
)
from gateway.tee.coordinator_source_add_v2 import (
    OP_SOURCE_ADD_FUNCTIONAL_PROBE_V2,
    OP_SOURCE_ADD_PROVENANCE_V2,
)
from gateway.tee.provider_outcome_v2 import (
    validate_provider_outcome_snapshot_v2,
)
from gateway.tee.coordinator_epoch_cutover_v2 import (
    OP_ATTEST_SUBNET_EPOCH_CUTOVER_V2,
    attest_subnet_epoch_cutover_v2,
)


OP_ATTEST_ARTIFACT_PERSISTENCE = "attest_artifact_persistence"
OP_ATTEST_QUALIFICATION_ADMISSION = "attest_qualification_admission"
OP_ATTEST_WEIGHT_INPUT = "attest_weight_input"
OP_ATTEST_WEIGHT_PUBLICATION = "attest_weight_publication"
OP_ATTEST_ACTIVE_PRIVATE_MODEL = "attest_active_private_model"
OP_SOURCE_ADD_CATALOG_SNAPSHOT_V2 = "source_add_catalog_snapshot_v2"
OP_PROVIDER_OUTCOME_SNAPSHOT_V2 = "provider_outcome_snapshot_v2"
OP_REGISTER_OPENROUTER_CREDENTIAL_V2 = "register_openrouter_credential_v2"
OP_PREFLIGHT_OPENROUTER_CREDENTIAL_V2 = "preflight_openrouter_credential_v2"
OP_ATTEST_LEGACY_FINALIZED_ALLOCATION_V2 = (
    "attest_legacy_finalized_allocation_v2"
)
OP_CLASSIFY_LEGACY_ALLOCATION_V2 = "classify_legacy_allocation_v2"
_HASH_RE = re.compile(r"^sha256:[0-9a-f]{64}$")

_COORDINATOR_WEIGHT_INPUT_PURPOSES = {
    category: purpose
    for category, (role, purpose) in WEIGHT_INPUT_PURPOSES.items()
    if role == "gateway_coordinator"
}


COORDINATOR_OPERATIONS_V2 = {
    OP_PROMOTION_IMPROVEMENT: frozenset({"research_lab.ranking.v2"}),
    OP_PROMOTION_GATE_DECISION: frozenset(
        {"research_lab.promotion_decision.v2"}
    ),
    OP_RESEARCH_LAB_ALLOCATION: frozenset({"research_lab.allocation.v2"}),
    OP_RESEARCH_LAB_REWARD_DECISION: frozenset(
        {"research_lab.reward_decision.v2"}
    ),
    OP_ATTEST_LEGACY_FINALIZED_ALLOCATION_V2: frozenset(
        {"research_lab.legacy_finalized_allocation.v2"}
    ),
    OP_CLASSIFY_LEGACY_ALLOCATION_V2: frozenset(
        {"research_lab.legacy_finalized_allocation.v2"}
    ),
    OP_ATTEST_SUBNET_EPOCH_CUTOVER_V2: frozenset(
        {"research_lab.subnet_epoch_cutover.v2"}
    ),
    OP_SOURCE_ADD_PROVENANCE_V2: frozenset(
        {"research_lab.source_add_provenance.v2"}
    ),
    OP_SOURCE_ADD_FUNCTIONAL_PROBE_V2: frozenset(
        {"research_lab.source_add_functional_probe.v2"}
    ),
    OP_SOURCE_ADD_CATALOG_SNAPSHOT_V2: frozenset(
        {"research_lab.source_add_catalog_snapshot.v2"}
    ),
    OP_PROVIDER_OUTCOME_SNAPSHOT_V2: frozenset(
        {"research_lab.provider_outcome_snapshot.v2"}
    ),
    OP_REGISTER_OPENROUTER_CREDENTIAL_V2: frozenset(
        {"research_lab.openrouter_credential.v2"}
    ),
    OP_PREFLIGHT_OPENROUTER_CREDENTIAL_V2: frozenset(
        {"research_lab.openrouter_credit_preflight.v2"}
    ),
    OP_ATTEST_ARTIFACT_PERSISTENCE: frozenset(
        {"leadpoet.artifact_persistence.v2"}
    ),
    OP_ATTEST_QUALIFICATION_ADMISSION: frozenset(
        {"research_lab.admission.v2"}
    ),
    OP_ATTEST_WEIGHT_INPUT: frozenset(
        _COORDINATOR_WEIGHT_INPUT_PURPOSES.values()
    ),
    OP_ATTEST_WEIGHT_PUBLICATION: frozenset(
        {"gateway.weights.publication.v2"}
    ),
    OP_ATTEST_ACTIVE_PRIVATE_MODEL: frozenset(
        {"research_lab.active_private_model.v2"}
    ),
}


def coordinator_failed_parent_graph_policy_v2(
    manifest: Mapping[str, Any],
    payload: Mapping[str, Any],
    graph: Mapping[str, Any],
) -> tuple[str, ...]:
    """Authorize one exact failed autoresearch root for terminal lineage only."""

    root_hash = str(graph.get("root_receipt_hash") or "")
    receipts = {
        str(item.get("receipt_hash") or ""): item
        for item in graph.get("receipts") or ()
        if isinstance(item, Mapping)
    }
    failed_hashes = {
        receipt_hash
        for receipt_hash, receipt in receipts.items()
        if receipt.get("status") == "failed"
    }
    if not failed_hashes:
        return ()
    root = receipts.get(root_hash)
    if not isinstance(root, Mapping):
        raise ValueError("failed receipt graph root is missing")

    operation = str(manifest.get("operation") or "")
    if operation == OP_ATTEST_ARTIFACT_PERSISTENCE:
        if str(payload.get("source_receipt_hash") or "") != root_hash:
            raise ValueError("artifact persistence failed source differs")
        return tuple(sorted(failed_hashes))

    if failed_hashes != {root_hash}:
        raise ValueError("failed reward ancestry must be the direct graph root")

    if operation != OP_RESEARCH_LAB_REWARD_DECISION:
        raise ValueError("failed receipt ancestry is unauthorized for operation")
    if payload.get("decision_kind") != "reimbursement":
        raise ValueError("failed receipt ancestry is reimbursement-only")
    decision_payload = payload.get("decision_payload")
    terminal_result = (
        decision_payload.get("autoresearch_result")
        if isinstance(decision_payload, Mapping)
        else None
    )
    if (
        root.get("purpose") != "research_lab.candidate_decision.v2"
        or not isinstance(terminal_result, Mapping)
        or terminal_result.get("status") != "failed"
        or root.get("output_root") != sha256_json(dict(terminal_result))
    ):
        raise ValueError("failed reimbursement ancestry does not bind terminal result")
    return (root_hash,)


class CoordinatorExecutorV2:
    """Invoke existing pure functions without owning any business formulas."""

    def __init__(
        self,
        *,
        artifact_evidence_supplier: Optional[
            Callable[
                [Iterable[str], ExecutionContextV2],
                Iterable[Mapping[str, Any]],
            ]
        ] = None,
        weight_source_resolver: Optional[
            Callable[[Mapping[str, Any], ExecutionContextV2], Mapping[str, Any]]
        ] = None,
        qualification_admission_resolver: Optional[
            Callable[[Mapping[str, Any], ExecutionContextV2], Mapping[str, Any]]
        ] = None,
        allocation_source_resolver: Optional[
            Callable[[Mapping[str, Any], ExecutionContextV2], Mapping[str, Any]]
        ] = None,
        source_add_provenance_resolver: Optional[
            Callable[[Mapping[str, Any], ExecutionContextV2], Mapping[str, Any]]
        ] = None,
        source_add_functional_probe_resolver: Optional[
            Callable[[Mapping[str, Any], ExecutionContextV2], Mapping[str, Any]]
        ] = None,
        reward_source_resolver: Optional[
            Callable[[Mapping[str, Any], ExecutionContextV2], Mapping[str, Any]]
        ] = None,
        legacy_settlement_source_resolver: Optional[
            Callable[[Mapping[str, Any], ExecutionContextV2], Mapping[str, Any]]
        ] = None,
        legacy_allocation_classification_resolver: Optional[
            Callable[[Mapping[str, Any], ExecutionContextV2], Mapping[str, Any]]
        ] = None,
        source_add_catalog_resolver: Optional[
            Callable[[Mapping[str, Any], ExecutionContextV2], Mapping[str, Any]]
        ] = None,
        provider_outcome_supplier: Optional[Callable[[], Mapping[str, Any]]] = None,
        openrouter_registration_resolver: Optional[
            Callable[
                [Mapping[str, Any], ExecutionContextV2],
                ExecutionResultV2,
            ]
        ] = None,
        openrouter_preflight_resolver: Optional[
            Callable[
                [Mapping[str, Any], ExecutionContextV2],
                ExecutionResultV2,
            ]
        ] = None,
        active_private_model_resolver: Optional[
            Callable[[Mapping[str, Any], ExecutionContextV2], Mapping[str, Any]]
        ] = None,
        config_supplier: Callable[[], ResearchLabGatewayConfig] = (
            ResearchLabGatewayConfig
        ),
    ) -> None:
        self._artifact_evidence_supplier = artifact_evidence_supplier
        self._weight_source_resolver = weight_source_resolver
        self._qualification_admission_resolver = qualification_admission_resolver
        self._allocation_source_resolver = allocation_source_resolver
        self._source_add_provenance_resolver = source_add_provenance_resolver
        self._source_add_functional_probe_resolver = (
            source_add_functional_probe_resolver
        )
        self._reward_source_resolver = reward_source_resolver
        self._legacy_settlement_source_resolver = (
            legacy_settlement_source_resolver
        )
        self._legacy_allocation_classification_resolver = (
            legacy_allocation_classification_resolver
        )
        self._source_add_catalog_resolver = source_add_catalog_resolver
        self._provider_outcome_supplier = provider_outcome_supplier
        self._openrouter_registration_resolver = openrouter_registration_resolver
        self._openrouter_preflight_resolver = openrouter_preflight_resolver
        self._active_private_model_resolver = active_private_model_resolver
        self._config = config_supplier()

    async def __call__(
        self,
        operation: str,
        payload: Mapping[str, Any],
        context: ExecutionContextV2,
    ) -> ExecutionResultV2:
        if operation not in COORDINATOR_OPERATIONS_V2:
            raise ValueError("unsupported V2 coordinator operation")
        if operation == OP_ATTEST_ARTIFACT_PERSISTENCE:
            return self._attest_artifact_persistence(payload, context)
        if operation == OP_ATTEST_QUALIFICATION_ADMISSION:
            return self._attest_qualification_admission(payload, context)
        if operation == OP_ATTEST_WEIGHT_INPUT:
            return self._attest_weight_input(payload, context)
        if operation == OP_ATTEST_WEIGHT_PUBLICATION:
            return self._attest_weight_publication(payload, context)
        if operation == OP_ATTEST_ACTIVE_PRIVATE_MODEL:
            if self._active_private_model_resolver is None:
                raise ValueError("active private model source is unavailable")
            document = dict(self._active_private_model_resolver(payload, context))
            return ExecutionResultV2(
                output=document,
                artifact_hashes=(
                    str(document["artifact"]["model_artifact_hash"]),
                    str(document["artifact"]["manifest_hash"]),
                    str(document["source_state_hash"]),
                ),
            )
        if operation == OP_ATTEST_LEGACY_FINALIZED_ALLOCATION_V2:
            if self._legacy_settlement_source_resolver is None:
                raise ValueError("measured legacy settlement source is unavailable")
            document = dict(
                self._legacy_settlement_source_resolver(payload, context)
            )
            return ExecutionResultV2(
                output=document,
                artifact_hashes=(
                    str(document["settlement_hash"]),
                    str(document["allocation_hash"]),
                    str(document["chain_compare_hash"]),
                    str(document["audit_event_hash"]),
                    str(document["checkpoint_merkle_root"]),
                ),
            )
        if operation == OP_CLASSIFY_LEGACY_ALLOCATION_V2:
            if self._legacy_allocation_classification_resolver is None:
                raise ValueError(
                    "measured legacy allocation classifier is unavailable"
                )
            document = dict(
                self._legacy_allocation_classification_resolver(
                    payload,
                    context,
                )
            )
            if "settlement_hash" in document:
                artifacts = (
                    str(document["settlement_hash"]),
                    str(document["allocation_hash"]),
                    str(document["chain_compare_hash"]),
                    str(document["audit_event_hash"]),
                    str(document["checkpoint_merkle_root"]),
                )
            elif "finding_hash" in document:
                artifacts = (
                    str(document["finding_hash"]),
                    str(document["allocation_hash"]),
                    str(document["chain_compare_hash"]),
                    str(document["audit_event_hash"]),
                    str(document["difference_hash"]),
                )
            else:
                raise ValueError(
                    "legacy allocation classification result is invalid"
                )
            return ExecutionResultV2(
                output=document,
                artifact_hashes=artifacts,
            )
        if operation == OP_ATTEST_SUBNET_EPOCH_CUTOVER_V2:
            document = attest_subnet_epoch_cutover_v2(payload, context)
            return ExecutionResultV2(
                output=document,
                artifact_hashes=(
                    str(document["mapping_hash"]),
                    str(document["first_snapshot_hash"]),
                    str(document["last_legacy_bundle_hash"]),
                    str(document["last_legacy_weight_finalization_event_hash"]),
                ),
            )
        if operation == OP_RESEARCH_LAB_ALLOCATION:
            return await self._research_lab_allocation(payload, context)
        if operation == OP_SOURCE_ADD_PROVENANCE_V2:
            if self._source_add_provenance_resolver is None:
                raise ValueError("measured SOURCE_ADD provenance is unavailable")
            output = dict(self._source_add_provenance_resolver(payload, context))
            return ExecutionResultV2(
                output=output,
                artifact_hashes=(sha256_json(output),),
            )
        if operation == OP_SOURCE_ADD_FUNCTIONAL_PROBE_V2:
            if self._source_add_functional_probe_resolver is None:
                raise ValueError("measured SOURCE_ADD functional probe is unavailable")
            output = dict(
                self._source_add_functional_probe_resolver(payload, context)
            )
            return ExecutionResultV2(
                output=output,
                artifact_hashes=(sha256_json(output),),
            )
        if operation == OP_SOURCE_ADD_CATALOG_SNAPSHOT_V2:
            if self._source_add_catalog_resolver is None:
                raise ValueError("measured SOURCE_ADD catalog is unavailable")
            output = dict(self._source_add_catalog_resolver(payload, context))
            return ExecutionResultV2(
                output=output,
                artifact_hashes=(
                    str(output["provisioned_sources_hash"]),
                    str(output["private_registry_rows_hash"]),
                    str(output["runtime_catalog_hash"]),
                ),
            )
        if operation == OP_PROVIDER_OUTCOME_SNAPSHOT_V2:
            if set(payload) != {"schema_version"} or payload.get(
                "schema_version"
            ) != "leadpoet.provider_outcome_snapshot_request.v2":
                raise ValueError("provider outcome snapshot request is invalid")
            if self._provider_outcome_supplier is None:
                raise ValueError("measured provider outcome state is unavailable")
            supplied = self._provider_outcome_supplier()
            if not isinstance(supplied, Mapping) or set(supplied) != {
                "snapshot",
                "transport_attempts",
                "evidence_artifact_hashes",
            }:
                raise ValueError("provider outcome snapshot evidence is invalid")
            output = validate_provider_outcome_snapshot_v2(
                supplied["snapshot"]
            )
            attempts = supplied["transport_attempts"]
            artifacts = supplied["evidence_artifact_hashes"]
            if not isinstance(attempts, list) or not isinstance(artifacts, list):
                raise ValueError("provider outcome snapshot evidence is invalid")
            return ExecutionResultV2(
                output=output,
                artifact_hashes=(
                    str(output["provider_outcome_digest_hash"]),
                    str(output["source_state_hash"]),
                    *[str(item) for item in artifacts],
                ),
                transport_attempts=tuple(dict(item) for item in attempts),
            )
        if operation == OP_REGISTER_OPENROUTER_CREDENTIAL_V2:
            if self._openrouter_registration_resolver is None:
                raise ValueError(
                    "measured OpenRouter credential registration is unavailable"
                )
            result = self._openrouter_registration_resolver(payload, context)
            if not isinstance(result, ExecutionResultV2):
                raise ValueError(
                    "measured OpenRouter credential registration is invalid"
                )
            return result
        if operation == OP_PREFLIGHT_OPENROUTER_CREDENTIAL_V2:
            if self._openrouter_preflight_resolver is None:
                raise ValueError(
                    "measured OpenRouter credit preflight is unavailable"
                )
            result = self._openrouter_preflight_resolver(payload, context)
            if not isinstance(result, ExecutionResultV2):
                raise ValueError(
                    "measured OpenRouter credit preflight is invalid"
                )
            return result
        if operation == OP_RESEARCH_LAB_REWARD_DECISION:
            decision_kind = str(payload.get("decision_kind") or "")
            measured_payload = payload
            if decision_kind in {
                "champion_migration",
                "source_add_leg1",
                "source_add_leg2",
                "reimbursement",
            }:
                if self._reward_source_resolver is None:
                    raise ValueError("measured reward source is unavailable")
                measured_payload = self._reward_source_resolver(payload, context)
            self._validate_reward_ancestry(measured_payload, context)
            output = execute_reward_decision_v2(measured_payload)
            return ExecutionResultV2(
                output=output,
                receipt_output=reward_receipt_projection_v2(output),
                artifact_hashes=(sha256_json(output),),
            )
        if operation == OP_PROMOTION_GATE_DECISION:
            try:
                threshold = float(payload.get("threshold_points"))
            except (TypeError, ValueError) as exc:
                raise ValueError("promotion threshold is invalid") from exc
            if (
                threshold != float(self._config.improvement_threshold_points)
                or payload.get("auto_promotion_enabled")
                is not bool(self._config.auto_promotion_enabled)
            ):
                raise ValueError(
                    "promotion policy differs from measured configuration"
                )
        result = await execute_scoring_operation(operation, payload)
        evidence_hashes = []
        if isinstance(result, ScoringExecutionResult):
            output = dict(result.result)
            evidence_hashes = list(result.evidence_roots.values())
        else:
            output = dict(result)
        return ExecutionResultV2(
            output=output,
            artifact_hashes=tuple(evidence_hashes),
        )

    @staticmethod
    def _validate_reward_ancestry(
        payload: Mapping[str, Any],
        context: ExecutionContextV2,
    ) -> None:
        kind = str(payload.get("decision_kind") or "")
        if kind == "champion_migration":
            if context.external_receipt_graphs or context.parent_receipt_hashes:
                raise ValueError(
                    "champion migration cannot inherit host-selected ancestry"
                )
            return
        expected_purpose = {
            "champion": "research_lab.promotion_decision.v2",
            "source_add_leg1": "research_lab.source_add_functional_probe.v2",
            "source_add_leg2": "research_lab.source_add_judge.v2",
            "reimbursement": "research_lab.candidate_decision.v2",
        }.get(kind)
        if expected_purpose is None:
            raise ValueError("reward ancestry kind is unsupported")
        graphs = list(context.external_receipt_graphs)
        if len(graphs) != 1:
            raise ValueError("reward decision requires exactly one parent graph")
        graph = graphs[0]
        root_hash = str(graph.get("root_receipt_hash") or "")
        receipts = {
            str(receipt.get("receipt_hash") or ""): receipt
            for receipt in graph.get("receipts") or ()
            if isinstance(receipt, Mapping)
        }
        root = receipts.get(root_hash)
        if (
            not isinstance(root, Mapping)
            or root.get("purpose") != expected_purpose
            or root_hash not in context.parent_receipt_hashes
        ):
            raise ValueError("reward decision parent purpose is invalid")
        decision_payload = payload.get("decision_payload")
        if not isinstance(decision_payload, Mapping):
            raise ValueError("reward decision input is invalid")
        bound_result = None
        if kind == "champion":
            promotion_decision = decision_payload.get("promotion_decision")
            if isinstance(promotion_decision, Mapping):
                bound_result = {"decision": dict(promotion_decision)}
        elif kind == "source_add_leg1":
            bound_result = decision_payload.get("functional_probe_result")
        elif kind == "source_add_leg2":
            bound_result = decision_payload.get("judge_result")
        elif kind == "reimbursement":
            bound_result = decision_payload.get("autoresearch_result")
        if bound_result is not None and (
            not isinstance(bound_result, Mapping)
            or root.get("output_root") != sha256_json(dict(bound_result))
        ):
            raise ValueError("reward decision parent output differs")

    async def _research_lab_allocation(
        self,
        payload: Mapping[str, Any],
        context: ExecutionContextV2,
    ) -> ExecutionResultV2:
        if self._allocation_source_resolver is None:
            raise ValueError("measured allocation source is unavailable")
        authority = dict(self._allocation_source_resolver(payload, context))
        required = {
            "allocation",
            "allocation_inputs",
            "source_state",
            "source_state_hash",
        }
        if set(authority) != required:
            raise ValueError("allocation authority result fields are invalid")
        result = await execute_scoring_operation(
            OP_RESEARCH_LAB_ALLOCATION,
            authority["allocation_inputs"],
        )
        if not isinstance(result, ScoringExecutionResult):
            raise ValueError("allocation kernel result is invalid")
        allocation = result.result.get("allocation")
        if allocation != authority["allocation"]:
            raise ValueError("allocation source and protected kernel differ")
        artifact_hashes = list(result.evidence_roots.values())
        artifact_hashes.append(str(authority["source_state_hash"]))
        return ExecutionResultV2(
            output=authority,
            receipt_output={"allocation": dict(authority["allocation"])},
            artifact_hashes=tuple(artifact_hashes),
        )

    def _attest_artifact_persistence(
        self,
        payload: Mapping[str, Any],
        context: ExecutionContextV2,
    ) -> ExecutionResultV2:
        if set(payload) != {
            "source_receipt_hash",
            "artifact_ids",
            "artifact_plaintext_hashes",
        }:
            raise ValueError("artifact persistence payload fields are invalid")
        source_receipt_hash = str(payload.get("source_receipt_hash") or "")
        artifact_ids = payload.get("artifact_ids")
        plaintext_hashes = payload.get("artifact_plaintext_hashes")
        if not _HASH_RE.fullmatch(source_receipt_hash):
            raise ValueError("artifact source receipt hash is invalid")
        if (
            not isinstance(artifact_ids, list)
            or not artifact_ids
            or len(set(artifact_ids)) != len(artifact_ids)
            or any(not _HASH_RE.fullmatch(str(item or "")) for item in artifact_ids)
        ):
            raise ValueError("artifact persistence IDs are invalid")
        if (
            not isinstance(plaintext_hashes, list)
            or len(plaintext_hashes) != len(artifact_ids)
            or any(not _HASH_RE.fullmatch(str(item or "")) for item in plaintext_hashes)
        ):
            raise ValueError("artifact plaintext commitments are invalid")
        if self._artifact_evidence_supplier is None:
            raise ValueError("artifact persistence evidence is unavailable")
        evidence = [
            dict(item)
            for item in self._artifact_evidence_supplier(artifact_ids, context)
        ]
        if sorted(item.get("artifact_id") for item in evidence) != sorted(artifact_ids):
            raise ValueError("artifact persistence evidence set differs")
        if sorted(item.get("plaintext_hash") for item in evidence) != sorted(
            plaintext_hashes
        ):
            raise ValueError("artifact plaintext commitments differ")
        if any(not item.get("persisted") for item in evidence):
            raise ValueError("artifact persistence evidence is incomplete")
        transport_attempts = []
        artifact_hashes = []
        output_artifacts = []
        for item in sorted(evidence, key=lambda value: value["artifact_id"]):
            attempts = item.get("transport_attempts")
            if not isinstance(attempts, list) or len(attempts) != 2:
                raise ValueError("artifact persistence transport evidence is incomplete")
            transport_attempts.extend(dict(attempt) for attempt in attempts)
            output_artifacts.append(
                {
                    key: item[key]
                    for key in (
                        "artifact_id",
                        "plaintext_hash",
                        "ciphertext_hash",
                        "artifact_ref",
                        "storage_document_hash",
                        "encryption_context_hash",
                        "object_lock_mode",
                        "retain_until",
                        "transport_root",
                    )
                }
            )
            artifact_hashes.extend(
                (
                    item["plaintext_hash"],
                    item["ciphertext_hash"],
                    item["storage_document_hash"],
                    item["transport_root"],
                )
            )
        output = {
            "source_receipt_hash": source_receipt_hash,
            "artifacts": output_artifacts,
            "artifact_set_root": sha256_json(output_artifacts),
        }
        return ExecutionResultV2(
            output=output,
            transport_attempts=tuple(transport_attempts),
            artifact_hashes=tuple(artifact_hashes),
        )

    def _attest_weight_input(
        self,
        payload: Mapping[str, Any],
        context: ExecutionContextV2,
    ) -> ExecutionResultV2:
        if self._weight_source_resolver is None:
            raise ValueError("measured weight input source is unavailable")
        category = str(payload.get("category") or "")
        expected_purpose = _COORDINATOR_WEIGHT_INPUT_PURPOSES.get(category)
        if expected_purpose is None:
            raise ValueError("weight input category is not coordinator-owned")
        if context.purpose != expected_purpose:
            raise ValueError("weight input category purpose is incorrect")
        document = dict(self._weight_source_resolver(payload, context))
        if int(document["epoch_id"]) != int(context.epoch_id):
            raise ValueError("weight input epoch differs from execution scope")
        return ExecutionResultV2(
            output=document,
            artifact_hashes=(sha256_json(document["value"]),),
        )

    def _attest_qualification_admission(
        self,
        payload: Mapping[str, Any],
        context: ExecutionContextV2,
    ) -> ExecutionResultV2:
        if context.purpose != "research_lab.admission.v2":
            raise ValueError("qualification admission purpose is incorrect")
        if self._qualification_admission_resolver is None:
            raise ValueError("measured qualification admission source is unavailable")
        document = dict(self._qualification_admission_resolver(payload, context))
        if int(document.get("epoch_id", -1)) != int(context.epoch_id):
            raise ValueError("qualification admission epoch differs")
        leads = document.get("leads")
        if not isinstance(leads, list):
            raise ValueError("qualification admission leads are invalid")
        return ExecutionResultV2(
            output=document,
            artifact_hashes=(sha256_json(leads),),
        )

    def _attest_weight_publication(
        self,
        payload: Mapping[str, Any],
        context: ExecutionContextV2,
    ) -> ExecutionResultV2:
        if set(payload) != {
            "bundle_hash",
            "root_receipt_hash",
            "durable_readback_hash",
            "transparency_event_hash",
        }:
            raise ValueError("weight publication payload fields are invalid")
        if context.purpose != "gateway.weights.publication.v2":
            raise ValueError("weight publication purpose is incorrect")
        normalized = {
            key: str(payload.get(key) or "").lower()
            for key in (
                "bundle_hash",
                "root_receipt_hash",
                "durable_readback_hash",
                "transparency_event_hash",
            )
        }
        if any(not _HASH_RE.fullmatch(value) for value in normalized.values()):
            raise ValueError("weight publication hash is invalid")
        document = {
            "schema_version": "leadpoet.weight_publication.v2",
            **normalized,
        }
        return ExecutionResultV2(
            output=document,
            artifact_hashes=(
                normalized["bundle_hash"],
                normalized["durable_readback_hash"],
                normalized["transparency_event_hash"],
            ),
        )
