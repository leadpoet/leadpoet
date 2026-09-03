"""V2 adapter around the unchanged Research Lab scoring implementation."""

from __future__ import annotations

import asyncio
import base64
import json
import os
from pathlib import Path
import re
import tempfile
import threading
from typing import Any, Callable, Dict, Iterable, Mapping

from gateway.tee.execution_job_manager_v2 import (
    ExecutionContextV2,
    ExecutionResultV2,
)
from gateway.research_lab.routing_experiment_attestation import (
    ROUTING_EXPERIMENT_ATTESTATION_OPERATION_V2,
    ROUTING_EXPERIMENT_ATTESTATION_PURPOSE_V2,
    routing_experiment_attestation_receipt_output_v2,
)
from gateway.research_lab.routing_execution_authorization import (
    ROUTING_PROVIDER_AUTHORIZATION_OPERATION_V2,
    ROUTING_PROVIDER_AUTHORIZATION_PURPOSE_V2,
    RoutingProviderCallAuthorizationV2,
    execute_routing_provider_call_authorization_v2,
    routing_provider_dispatch_job_id_v2,
    validate_routing_provider_authorization_request_v2,
)
from gateway.research_lab.routing_experiment_artifacts import (
    VerifiedRoutingArtifactLineage,
)
from gateway.research_lab.routing_model_binding_observation import (
    ROUTING_MODEL_BINDING_OBSERVATION_PURPOSE_V2,
    RoutingModelBindingObservationError,
    observe_routing_model_bindings_v2,
)
from gateway.research_lab.routing_model_binding_producer import (
    ROUTING_MODEL_BINDING_OBSERVATION_OPERATION_V2,
    RoutingModelBindingProducerError,
    _build_metadata_payload,
    _validate_lineage_and_artifact,
    _validate_model_result,
    _validate_source_bundle,
    resolve_verified_routing_artifact_lineage_v2,
)
from gateway.research_lab.routing_provider_terminal_protected import (
    ROUTING_BUDGET_RESERVATION_PURPOSE_V3,
    ROUTING_PROVIDER_DISPATCH_OPERATION_V2,
    ROUTING_PROVIDER_DISPATCH_PURPOSE_V2,
    ROUTING_PROVIDER_DISPATCH_REQUEST_SCHEMA_V2,
    ROUTING_PROVIDER_TERMINAL_OPERATION_V2,
    ROUTING_PROVIDER_TERMINAL_PURPOSE_V2,
    ProtectedRoutingProviderTerminalError,
    routing_provider_dispatch_receipt_output_v2,
    routing_budget_reservation_proof_v3,
    execute_protected_routing_provider_terminal_v2,
    prepared_routing_provider_call_from_mapping,
    validate_routing_budget_reservation_result_v3,
    validate_routing_budget_reservation_v3,
    validate_routing_model_completion_contract_v1,
)
from gateway.research_lab.routing_provider_bindings import (
    ReviewedDeeplineActionCompiler,
    VerifiedRoutingBindingCatalog,
    VerifiedRoutingUnitDataset,
)
from gateway.tee.provider_client_v2 import BrokeredProviderTransportV2
from gateway.tee.model_sandbox_v2 import (
    ModelSandboxV2Error,
    RunscModelSandboxV2,
    provider_evidence_tape_input_root,
)
from gateway.tee.source_bundle_v2 import extract_source_bundle_v2
from gateway.tee.qualification_executor_v2 import (
    OP_QUALIFICATION_BATCH_V2,
    OP_QUALIFICATION_EMAIL_EVIDENCE_V2,
    OP_QUALIFICATION_EPOCH_V2,
    QualificationExecutorV2,
)
from gateway.tee.qualification_network_v2 import SecureQualificationNetworkV2
from gateway.tee.qualification_epoch_guard_v2 import QualificationEpochGuardV2
from gateway.tee.scoring_executor import (
    OP_BENCHMARK_ICP_SCORE,
    OP_BUILD_BASELINE_SCORE_SUMMARY,
    OP_BUILD_SCORE_BUNDLE,
    OP_QUALIFICATION_COMPANY_SCORES,
    ScoringExecutionResult,
    configuration_hash,
    execute_scoring_operation,
)
from leadpoet_canonical.attested_v2 import (
    canonical_json,
    sha256_bytes,
    sha256_json,
    validate_transport_attempt,
)
from research_lab.eval import PrivateModelArtifactManifest
from research_lab.eval.dev_eval import (
    compute_dev_set_hash,
    evaluate_dev,
    select_snapshot_dev_icps,
)
from research_lab.eval.provider_evidence_cache import (
    EVIDENCE_CACHE_SCHEMA_VERSION,
    icp_evidence_cache_key,
)
from research_lab.eval.private_runtime import canonicalize_private_model_icp
from research_lab.eval.snapshot_store import (
    MODE_REPLAY,
    ProviderSnapshotStore,
)
from gateway.research_lab.provider_preflight import ProviderPreflight
from gateway.research_lab.config import ResearchLabGatewayConfig
from gateway.tee.research_lab_runtime_config_v2 import (
    build_research_lab_execution_config,
    measured_credential_environment_names,
    measured_dev_eval_icp_timeout_seconds,
    measured_dev_eval_total_timeout_seconds,
    measured_dev_replay_environment,
    measured_dev_snapshot_miss_policy,
    measured_git_tree_config,
    validate_model_sandbox_environment,
    validate_research_lab_execution_config,
)


SCORE_PURPOSES_V2 = frozenset(
    {
        "research_lab.company_score.v2",
        "research_lab.candidate_score.v2",
        "research_lab.baseline_score.v2",
        "research_lab.benchmark.v2",
        "research_lab.rebenchmark.v2",
        "research_lab.confirmation_score.v2",
    }
)

OP_RUN_MODEL_SANDBOX_V2 = "run_model_sandbox_v2"
MODEL_COMPATIBILITY_PURPOSE_V2 = "research_lab.model_compatibility.v2"
OP_DEV_REPLAY_V2 = "run_dev_replay_v2"
OP_DEV_HYBRID_V2 = "run_dev_hybrid_v2"
OP_PROVIDER_PREFLIGHT_V2 = "provider_preflight_v2"
OP_SOURCE_ADD_LEG2_JUDGE_V2 = "source_add_leg2_judge_v2"
OP_ATTEST_ROUTING_EXPERIMENT_V2 = ROUTING_EXPERIMENT_ATTESTATION_OPERATION_V2
OP_ATTEST_ROUTING_PROVIDER_CALL_V2 = ROUTING_PROVIDER_AUTHORIZATION_OPERATION_V2
OP_OBSERVE_ROUTING_MODEL_BINDINGS_V2 = (
    ROUTING_MODEL_BINDING_OBSERVATION_OPERATION_V2
)
OP_PROTECTED_ROUTING_PROVIDER_TERMINAL_V2 = (
    ROUTING_PROVIDER_TERMINAL_OPERATION_V2
)
OP_PROTECTED_ROUTING_PROVIDER_DISPATCH_V2 = ROUTING_PROVIDER_DISPATCH_OPERATION_V2
# Short alias for callers that refer to the operation by its protected job
# name rather than its executor family.
OP_ROUTING_PROVIDER_DISPATCH_V2 = OP_PROTECTED_ROUTING_PROVIDER_DISPATCH_V2
DEV_REPLAY_REQUEST_SCHEMA_VERSION = "leadpoet.dev_replay_request.v4"
DEV_HYBRID_REQUEST_SCHEMA_VERSION = "leadpoet.dev_hybrid_request.v4"
PROVIDER_PREFLIGHT_REQUEST_SCHEMA_VERSION = "leadpoet.provider_preflight_request.v3"
SOURCE_ADD_JUDGE_REQUEST_SCHEMA_VERSION = "leadpoet.source_add_judge_request.v2"
SOURCE_ADD_JUDGE_RESULT_SCHEMA_VERSION = "leadpoet.source_add_judge_result.v2"
PROVIDER_CREDENTIAL_REFS_FIELD = "_v2_provider_credential_ref_hashes"
PROVIDER_CREDENTIAL_PROFILE_FIELD = "_v2_provider_credential_profile"
_HASH_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


class DevEvalRunnerError(RuntimeError):
    """Measured equivalent of the legacy replay runner failure type."""

SCORING_OPERATIONS_V2 = {
    OP_QUALIFICATION_BATCH_V2: frozenset({"qualification.lead_decision.v2"}),
    OP_QUALIFICATION_EMAIL_EVIDENCE_V2: frozenset(
        {"qualification.email_evidence.v2"}
    ),
    OP_QUALIFICATION_EPOCH_V2: frozenset({"qualification.sourcing_epoch.v2"}),
    OP_RUN_MODEL_SANDBOX_V2: frozenset(
        {
            "research_lab.private_model_run.v2",
            "research_lab.candidate_model_run.v2",
            "research_lab.candidate_hybrid_discovery.v2",
            MODEL_COMPATIBILITY_PURPOSE_V2,
        }
    ),
    OP_DEV_REPLAY_V2: frozenset({"research_lab.candidate_test.v2"}),
    OP_DEV_HYBRID_V2: frozenset(
        {"research_lab.candidate_hybrid_test.v2"}
    ),
    OP_PROVIDER_PREFLIGHT_V2: frozenset(
        {"research_lab.provider_preflight.v2"}
    ),
    OP_SOURCE_ADD_LEG2_JUDGE_V2: frozenset(
        {"research_lab.source_add_judge.v2"}
    ),
    # This isolated, deterministic operation is intentionally not activated
    # by the current protected release manifest or durable purpose allowlist.
    # Keeping it separate from score/bundle operations prevents a routing Lab
    # reference from being confused with a production scoring decision.
    OP_ATTEST_ROUTING_EXPERIMENT_V2: frozenset(
        {ROUTING_EXPERIMENT_ATTESTATION_PURPOSE_V2}
    ),
    # Code-owned pre-dispatch authorization.  Durable activation is separate
    # from the final evaluation purpose and remains blocked until the new
    # append-only purpose migration and protected manifest are released.
    OP_ATTEST_ROUTING_PROVIDER_CALL_V2: frozenset(
        {ROUTING_PROVIDER_AUTHORIZATION_PURPOSE_V2}
    ),
    OP_OBSERVE_ROUTING_MODEL_BINDINGS_V2: frozenset(
        {ROUTING_MODEL_BINDING_OBSERVATION_PURPOSE_V2}
    ),
    OP_PROTECTED_ROUTING_PROVIDER_TERMINAL_V2: frozenset(
        {ROUTING_PROVIDER_TERMINAL_PURPOSE_V2}
    ),
    OP_PROTECTED_ROUTING_PROVIDER_DISPATCH_V2: frozenset(
        {ROUTING_PROVIDER_DISPATCH_PURPOSE_V2}
    ),
    OP_QUALIFICATION_COMPANY_SCORES: SCORE_PURPOSES_V2,
    OP_BENCHMARK_ICP_SCORE: SCORE_PURPOSES_V2,
    OP_BUILD_SCORE_BUNDLE: frozenset(
        {
            "research_lab.candidate_score.v2",
            "research_lab.baseline_score.v2",
            "research_lab.benchmark.v2",
            "research_lab.rebenchmark.v2",
            "research_lab.confirmation_score.v2",
        }
    ),
    OP_BUILD_BASELINE_SCORE_SUMMARY: frozenset(
        {
            "research_lab.baseline_score.v2",
            "research_lab.benchmark.v2",
            "research_lab.rebenchmark.v2",
        }
    ),
}


class ScoringExecutorV2:
    def __init__(
        self,
        *,
        provider_execute: Callable[[Mapping[str, Any]], Mapping[str, Any]],
        retry_policy_hashes: Mapping[str, str],
        model_sandbox: RunscModelSandboxV2 | None = None,
        artifact_seal: Callable[..., Mapping[str, Any]] | None = None,
        qualification_executor: QualificationExecutorV2 | None = None,
        config_supplier: Callable[[], ResearchLabGatewayConfig] = (
            ResearchLabGatewayConfig
        ),
        execution_config: Mapping[str, Any] | None = None,
        routing_binding_catalog: VerifiedRoutingBindingCatalog | None = None,
        routing_unit_dataset: VerifiedRoutingUnitDataset | None = None,
        routing_artifact_lineage: VerifiedRoutingArtifactLineage | None = None,
        routing_artifact_lineage_resolver: Callable[..., VerifiedRoutingArtifactLineage]
        | None = None,
        routing_coordinator_boot_identity_supplier: Callable[
            [], Mapping[str, Any]
        ]
        | None = None,
    ) -> None:
        self._provider_execute = provider_execute
        self._retry_policy_hashes = dict(retry_policy_hashes)
        self._transport = BrokeredProviderTransportV2(self._provider_execute)
        self._model_sandbox = model_sandbox
        self._artifact_seal = artifact_seal
        self._routing_binding_catalog = routing_binding_catalog
        self._routing_unit_dataset = routing_unit_dataset
        self._routing_artifact_lineage = routing_artifact_lineage
        self._routing_artifact_lineage_resolver = (
            routing_artifact_lineage_resolver
            or resolve_verified_routing_artifact_lineage_v2
        )
        self._routing_coordinator_boot_identity_supplier = (
            routing_coordinator_boot_identity_supplier
        )
        self._config = config_supplier()
        self._execution_config = validate_research_lab_execution_config(
            execution_config
            if execution_config is not None
            else build_research_lab_execution_config(
                config=self._config,
            )
        )
        self._transport.install()
        try:
            self._qualification_executor = (
                qualification_executor or QualificationExecutorV2(
                    epoch_checker=QualificationEpochGuardV2(
                        self._transport,
                        epoch_authority={
                            "mode": self._execution_config["epoch_authority"][
                                "mode"
                            ],
                            "cutover": self._execution_config["epoch_authority"][
                                "cutover"
                            ],
                        },
                        netuid=self._execution_config["deployment"]["netuid"],
                    )
                )
            )
            self._qualification_network = SecureQualificationNetworkV2()
            self._qualification_network.install()
        except BaseException:
            # A failed construction must never leak the process-wide transport
            # interception (httpx/requests/urllib send hooks): every later HTTP
            # call in the process would be silently swallowed or rejected.
            self._transport.restore()
            raise
        self._preflight_lock = threading.Lock()
        self._preflight_by_scope: Dict[str, ProviderPreflight] = {}
        os.environ["EXA_API_KEY"] = "leadpoet-v2-brokered-credential"
        os.environ["SCRAPINGDOG_API_KEY"] = "leadpoet-v2-brokered-credential"
        os.environ["QUALIFICATION_SCRAPINGDOG_API_KEY"] = (
            "leadpoet-v2-brokered-credential"
        )
        # These are non-secret sentinels only.  The low-level broker strips
        # credential headers and injects the KMS-held benchmark-scorer key.
        # The qualification intent verifier still requires a truthy key before
        # it reaches that intercepted HTTP boundary.
        for name in (
            "OPENROUTER_API_KEY",
            "OPENROUTER_KEY",
            "QUALIFICATION_OPENROUTER_API_KEY",
        ):
            os.environ[name] = "leadpoet-v2-brokered-credential"

    def close(self) -> None:
        self._qualification_network.restore()
        self._transport.restore()

    async def __call__(
        self,
        operation: str,
        payload: Mapping[str, Any],
        context: ExecutionContextV2,
    ) -> ExecutionResultV2:
        if operation not in SCORING_OPERATIONS_V2:
            raise ValueError("unsupported V2 scoring operation")
        payload = dict(payload)
        credential_profile = payload.pop(PROVIDER_CREDENTIAL_PROFILE_FIELD, None)
        if credential_profile is None:
            credential_profile = "default"
        if credential_profile != context.provider_credential_profile:
            raise ValueError("V2 provider credential profile differs from job manifest")
        allowed_profiles = {"default"}
        if operation == OP_RUN_MODEL_SANDBOX_V2:
            allowed_profiles.add("benchmark_model")
        elif operation == OP_PROVIDER_PREFLIGHT_V2:
            allowed_profiles.add("provider_preflight")
        elif operation == OP_QUALIFICATION_COMPANY_SCORES:
            allowed_profiles.add("benchmark_scorer")
        elif operation == OP_SOURCE_ADD_LEG2_JUDGE_V2:
            allowed_profiles.add("source_add_judge")
        if credential_profile not in allowed_profiles:
            raise ValueError("V2 provider credential profile is not allowed for operation")
        credential_refs = payload.pop(PROVIDER_CREDENTIAL_REFS_FIELD, None)
        if credential_refs is None and context.provider_credential_ref_hashes:
            raise ValueError("V2 provider credential profile is missing")
        if credential_refs is None:
            credential_refs = {}
        if not isinstance(credential_refs, Mapping):
            raise ValueError("V2 provider credential profile is invalid")
        if dict(credential_refs) != dict(context.provider_credential_ref_hashes):
            raise ValueError("V2 provider credential profile differs from job manifest")
        if operation == OP_DEV_REPLAY_V2:
            return await self._execute_dev_replay(payload, context)
        if operation == OP_DEV_HYBRID_V2:
            return await self._execute_dev_hybrid(payload, context)
        if operation == OP_ATTEST_ROUTING_EXPERIMENT_V2:
            if credential_profile != "default" or credential_refs:
                raise ValueError(
                    "routing experiment attestation must not use provider credentials"
                )
            if context.purpose != ROUTING_EXPERIMENT_ATTESTATION_PURPOSE_V2:
                raise ValueError("routing experiment attestation purpose is invalid")
            return ExecutionResultV2(
                output=routing_experiment_attestation_receipt_output_v2(payload),
                artifact_hashes=(),
            )
        if operation == OP_ATTEST_ROUTING_PROVIDER_CALL_V2:
            if credential_profile != "default" or credential_refs:
                raise ValueError(
                    "routing provider authorization must not use provider credentials"
                )
            if context.purpose != ROUTING_PROVIDER_AUTHORIZATION_PURPOSE_V2:
                raise ValueError("routing provider authorization purpose is invalid")
            if (
                self._routing_artifact_lineage is None
                or self._routing_binding_catalog is None
                or self._routing_unit_dataset is None
            ):
                raise ValueError(
                    "routing provider authorization authorities are unavailable"
                )
            try:
                (
                    authorization,
                    observation,
                    _envelope,
                    _admission,
                    protected_receipt,
                ) = validate_routing_provider_authorization_request_v2(
                    payload,
                    artifact_lineage=self._routing_artifact_lineage,
                    binding_catalog=self._routing_binding_catalog,
                    unit_dataset=self._routing_unit_dataset,
                )
                required_parent_receipts = (
                    dict(observation.signed_receipt),
                    dict(protected_receipt),
                )
                declared_parents = set(context.parent_receipt_hashes)
                observed_receipts = {
                    str(receipt.get("receipt_hash") or ""): dict(receipt)
                    for graph in context.external_receipt_graphs
                    for receipt in graph.get("receipts") or ()
                    if isinstance(receipt, Mapping)
                }
                for receipt in required_parent_receipts:
                    receipt_hash = str(receipt.get("receipt_hash") or "")
                    if (
                        receipt_hash not in declared_parents
                        or observed_receipts.get(receipt_hash) != receipt
                    ):
                        raise ValueError(
                            "routing provider authorization parent authority differs"
                        )
            except Exception as exc:  # noqa: BLE001 - protected boundary
                raise ValueError(
                    "routing provider authorization context is invalid"
                ) from exc
            authorization_result = execute_routing_provider_call_authorization_v2(
                authorization.to_dict(),
                authorization_job_id=context.job_id,
            )
            if (
                authorization_result.get("authorization_job_id") != context.job_id
                or authorization_result.get("admission_job_id")
                != authorization.admission_job_id
            ):
                raise ValueError(
                    "routing provider authorization execution identity differs"
                )
            return ExecutionResultV2(
                output=authorization_result,
                artifact_hashes=(),
            )
        if operation == OP_OBSERVE_ROUTING_MODEL_BINDINGS_V2:
            return await self._execute_routing_model_binding_observation(
                payload,
                context,
                credential_profile=credential_profile,
                credential_refs=credential_refs,
            )
        if operation == OP_PROTECTED_ROUTING_PROVIDER_TERMINAL_V2:
            if credential_profile != "default" or credential_refs:
                raise ValueError(
                    "routing provider terminal must not use provider credentials"
                )
            if context.purpose != ROUTING_PROVIDER_TERMINAL_PURPOSE_V2:
                raise ValueError("routing provider terminal purpose is invalid")
            return await self._execute_protected_routing_provider_terminal(
                payload, context
            )
        if operation == OP_PROTECTED_ROUTING_PROVIDER_DISPATCH_V2:
            if credential_profile != "default" or credential_refs:
                raise ValueError(
                    "routing provider dispatch must not use provider credentials"
                )
            if context.purpose != ROUTING_PROVIDER_DISPATCH_PURPOSE_V2:
                raise ValueError("routing provider dispatch purpose is invalid")
            return await self._execute_protected_routing_provider_dispatch(
                payload, context
            )
        if operation == OP_RUN_MODEL_SANDBOX_V2:
            if self._model_sandbox is None:
                raise ValueError("measured model sandbox is unavailable")
            if self._artifact_seal is None:
                raise ValueError("measured model artifact sealer is unavailable")
            metadata_compatibility = (
                context.purpose == MODEL_COMPATIBILITY_PURPOSE_V2
            )
            if metadata_compatibility:
                if credential_profile != "default" or credential_refs:
                    raise ValueError(
                        "model compatibility metadata credentials must be empty"
                    )
                expected_empty_fields = {
                    "input": {},
                    "environment": {},
                    "provider_evidence_cache": {},
                    "provider_evidence_cache_ref": "",
                    "provider_evidence_mode": "",
                    "provider_snapshot_bundle": {},
                    "provider_snapshot_tree_hash": "",
                    "provider_snapshot_manifest_hash": "",
                    "provider_cost_scope": "",
                    "provider_cost_cap_microusd": 0,
                    "provider_call_cap": 0,
                    "provider_runtime_catalog": {},
                    "provider_catalog_evidence": {},
                }
                if payload.get("operation") != "metadata" or any(
                    payload.get(name) != expected
                    for name, expected in expected_empty_fields.items()
                ):
                    raise ValueError(
                        "model compatibility metadata authority is not isolated"
                    )
            else:
                if payload.get("operation") == "metadata":
                    raise ValueError(
                        "model metadata requires compatibility authority"
                    )
                self._validate_model_provider_catalog_ancestry(payload, context)
                payload["environment"] = validate_model_sandbox_environment(
                    self._execution_config,
                    payload.get("environment"),
                    provider_cost_scope=str(
                        payload.get("provider_cost_scope") or ""
                    ),
                )
            cache_document = payload.get("provider_evidence_cache")
            cache_hash = sha256_json(
                dict(cache_document) if isinstance(cache_document, Mapping) else {}
            )
            if cache_document:
                expected_tape_input_root = provider_evidence_tape_input_root(
                    str(payload.get("provider_evidence_cache_ref") or ""),
                    cache_hash,
                )
                disclosed_receipt_sets = [
                    graph.get("receipts") or ()
                    for graph in context.external_receipt_graphs
                ] + [
                    proof.get("disclosed_receipts") or ()
                    for proof in context.external_ancestry_proofs
                ]
                matching_tapes = {
                    sha256_json(dict(receipt)): dict(receipt)
                    for receipts in disclosed_receipt_sets
                    for receipt in receipts
                    if isinstance(receipt, Mapping)
                    and receipt.get("role") == "gateway_scoring"
                    and receipt.get("purpose")
                    == "research_lab.provider_evidence_tape.v2"
                    and receipt.get("status") == "succeeded"
                    and receipt.get("input_root") == expected_tape_input_root
                    and receipt.get("output_root") == cache_hash
                }
                if len(matching_tapes) != 1:
                    raise ValueError(
                        "provider evidence cache has no unique measured tape ancestry"
                    )
            result = await asyncio.to_thread(
                self._model_sandbox.execute,
                payload,
                job_id=context.job_id,
                purpose=context.purpose,
                retry_policy_hashes=self._retry_policy_hashes,
                terminal_sink=context.record_transport,
                artifact_sink=context.record_artifact,
            )
            sealed_artifacts = []
            artifact_payloads = () if metadata_compatibility else (
                ("model_output", canonical_json(result.get("output")).encode("utf-8")),
                (
                    "model_trace",
                    canonical_json(result.get("trace_entries")).encode("utf-8"),
                ),
            )
            generated_cache = result.get("generated_provider_evidence_cache")
            if generated_cache:
                artifact_payloads += (
                    (
                        "provider_evidence_tape",
                        canonical_json(generated_cache).encode("utf-8"),
                    ),
                )
            for artifact_kind, plaintext in artifact_payloads:
                descriptor = await asyncio.to_thread(
                    self._artifact_seal,
                    plaintext=plaintext,
                    job_id=context.job_id,
                    purpose=context.purpose,
                    artifact_kind=artifact_kind,
                )
                if (
                    not isinstance(descriptor, Mapping)
                    or descriptor.get("status") != "sealed"
                    or descriptor.get("job_id") != context.job_id
                    or descriptor.get("purpose") != context.purpose
                    or descriptor.get("artifact_kind") != artifact_kind
                    or descriptor.get("plaintext_hash") != sha256_bytes(plaintext)
                ):
                    raise ValueError("measured model artifact seal differs")
                sealed_artifacts.append(dict(descriptor))
            result = {**dict(result), "sealed_artifacts": sealed_artifacts}
            generated_cache = result.get("generated_provider_evidence_cache")
            generated_cache_hash = result.get(
                "generated_provider_evidence_cache_hash"
            )
            if generated_cache:
                if (
                    context.purpose
                    not in {
                        "research_lab.private_model_run.v2",
                        "research_lab.candidate_hybrid_discovery.v2",
                    }
                    or sha256_json(generated_cache) != generated_cache_hash
                ):
                    raise ValueError(
                        "generated provider evidence cache commitment differs"
                    )
                context.record_stage(
                    purpose="research_lab.provider_evidence_tape.v2",
                    input_root=provider_evidence_tape_input_root(
                        str(result["provider_evidence_cache_ref"]),
                        str(generated_cache_hash),
                    ),
                    output_root=str(generated_cache_hash),
                    artifact_hashes=(
                        str(result["trace_entries_hash"]),
                        str(generated_cache_hash),
                    ),
                )
            artifact_hashes = tuple(
                str(result[field])
                for field in (
                    "model_artifact_hash",
                    "model_manifest_hash",
                    "source_bundle_hash",
                    "compatibility_policy_hash",
                    "compatibility_admission_hash",
                    "runtime_config_hash",
                    "input_hash",
                    "provider_evidence_cache_hash",
                    "provider_snapshot_archive_hash",
                    "provider_snapshot_tree_hash",
                    "provider_snapshot_manifest_hash",
                    "provider_runtime_catalog_hash",
                    "generated_provider_evidence_cache_hash",
                    "trace_entries_hash",
                    "output_hash",
                )
            ) + tuple(
                str(descriptor[field])
                for descriptor in sealed_artifacts
                for field in (
                    "artifact_id",
                    "plaintext_hash",
                    "ciphertext_hash",
                    "encryption_context_hash",
                )
            )
            return ExecutionResultV2(
                output=result,
                artifact_hashes=artifact_hashes,
            )
        if operation == OP_PROVIDER_PREFLIGHT_V2:
            return await self._execute_provider_preflight(payload, context)
        if operation == OP_SOURCE_ADD_LEG2_JUDGE_V2:
            return await self._execute_source_add_judge(payload, context)
        if operation == OP_QUALIFICATION_EPOCH_V2:
            return self._qualification_executor.aggregate_epoch(payload, context)
        if operation == OP_BUILD_BASELINE_SCORE_SUMMARY:
            self._validate_baseline_configuration(payload)
        if operation == OP_BUILD_SCORE_BUNDLE:
            self._validate_conditional_preliminary_ancestry(payload, context)
        with self._transport.scope(
            job_id=context.job_id,
            purpose=context.purpose,
            logical_operation_id=context.job_id,
            retry_policy_hashes=self._retry_policy_hashes,
            terminal_sink=context.record_transport,
            artifact_sink=context.record_artifact,
            # Qualification scoring deliberately converts an unavailable
            # evidence URL into a scored verification failure. Preserve the
            # signed transport terminal in the receipt graph without letting
            # the scope finalizer replace that authoritative scorer result
            # with an execution failure. Every other scoring operation keeps
            # the default fail-closed transport policy.
            allow_transport_failures=(
                operation == OP_QUALIFICATION_COMPANY_SCORES
            ),
        ):
            if operation == OP_QUALIFICATION_EMAIL_EVIDENCE_V2:
                return await self._qualification_executor.execute_email_evidence(
                    payload, context
                )
            if operation == OP_QUALIFICATION_BATCH_V2:
                return await self._qualification_executor.execute_batch(payload, context)
            result = await execute_scoring_operation(operation, payload)
        evidence_hashes = []
        if isinstance(result, ScoringExecutionResult):
            evidence_hashes = list(result.evidence_roots.values())
            output = dict(result.result)
        else:
            output = dict(result)
        return ExecutionResultV2(
            output=output,
            artifact_hashes=tuple(evidence_hashes),
        )

    async def _execute_routing_model_binding_observation(
        self,
        payload: Mapping[str, Any],
        context: ExecutionContextV2,
        *,
        credential_profile: str,
        credential_refs: Mapping[str, Any],
    ) -> ExecutionResultV2:
        """Run model metadata and chain a standard stage receipt.

        This operation has no provider channel. It calls the existing measured
        metadata sandbox directly with an empty provider state, then records a
        stage through the execution context. The ExecutionJobManager signs the
        synthetic stage receipt and final job receipt; no caller-supplied
        signer or receipt is accepted here.
        """

        if context.purpose != ROUTING_MODEL_BINDING_OBSERVATION_PURPOSE_V2:
            raise ValueError("routing model binding observation purpose is invalid")
        if credential_profile != "default" or credential_refs:
            raise ValueError(
                "routing model binding observation must not use provider credentials"
            )
        if self._model_sandbox is None:
            raise ValueError("measured model sandbox is unavailable")
        required = {
            "schema_version",
            "model_kind",
            "artifact_lineage",
            "artifact",
            "source_bundle",
            "provider_bindings",
        }
        if not isinstance(payload, Mapping) or set(payload) != required:
            raise ValueError("routing model binding observation payload is invalid")
        if payload.get("schema_version") != "leadpoet.routing_model_binding_request.v2":
            raise ValueError("routing model binding observation schema is invalid")
        model_kind = str(payload.get("model_kind") or "")
        if model_kind not in {"private", "candidate"}:
            raise ValueError("routing model binding observation model kind is invalid")
        try:
            lineage = self._routing_artifact_lineage_resolver(
                lineage_document=payload["artifact_lineage"],
                artifact_document=payload["artifact"],
            )
            artifact = _validate_lineage_and_artifact(lineage, payload["artifact"])
            source_bundle = _validate_source_bundle(
                payload["source_bundle"],
                artifact_hash=artifact.model_artifact_hash,
            )
        except (TypeError, KeyError, RoutingModelBindingProducerError) as exc:
            raise ValueError("routing model binding artifact lineage is invalid") from exc
        raw_bindings = payload["provider_bindings"]
        if (
            not isinstance(raw_bindings, list)
            or not raw_bindings
            or any(not isinstance(item, Mapping) for item in raw_bindings)
        ):
            raise ValueError("routing model binding identities are invalid")
        from research_lab.routing_experiments import ProviderBindingIdentity

        try:
            bindings = tuple(
                ProviderBindingIdentity.from_mapping(item) for item in raw_bindings
            )
            request = _build_metadata_payload(
                artifact=artifact,
                source_bundle=source_bundle,
                model_kind=model_kind,
            )
        except Exception as exc:  # noqa: BLE001
            raise ValueError("routing model binding metadata request is invalid") from exc
        result = await asyncio.to_thread(
            self._model_sandbox.execute,
            request,
            job_id=context.job_id,
            purpose=context.purpose,
            retry_policy_hashes={},
            terminal_sink=context.record_transport,
            artifact_sink=context.record_artifact,
        )
        try:
            measured = _validate_model_result(
                result,
                payload=request,
                artifact=artifact,
                source_bundle=source_bundle,
                model_kind=model_kind,
            )
            observation = observe_routing_model_bindings_v2(
                runtime_metadata=measured["output"]["runtime_routing"],
                provider_bindings=bindings,
                artifact_lineage_hash=lineage.identity_hash(),
            )
        except (RoutingModelBindingProducerError, RoutingModelBindingObservationError) as exc:
            raise ValueError("measured routing model binding observation is invalid") from exc
        context.record_stage(
            purpose=ROUTING_MODEL_BINDING_OBSERVATION_PURPOSE_V2,
            # This stage receipt is consumed by
            # VerifiedRoutingModelBindingRequirements, whose signed input is
            # the canonical observation request.  The complete measured
            # sandbox result is still bound by the stage artifact hashes and
            # the enclosing ExecutionJobManager receipt.
            input_root=observation["request_root"],
            output_root=sha256_json(observation),
            artifact_hashes=(
                measured["model_artifact_hash"],
                measured["model_manifest_hash"],
                measured["source_bundle_hash"],
                measured["compatibility_policy_hash"],
                measured["compatibility_admission_hash"],
                measured["runtime_config_hash"],
                measured["provider_runtime_catalog_hash"],
                measured["trace_entries_hash"],
                measured["output_hash"],
                measured["output"]["runtime_routing"]["catalog_sha256"],
                measured["output"]["runtime_routing"]["policy_sha256"],
            ),
        )
        output = {
            "schema_version": "leadpoet.routing_model_binding_result.v2",
            "operation": OP_OBSERVE_ROUTING_MODEL_BINDINGS_V2,
            "artifact_lineage_hash": lineage.identity_hash(),
            "model_result": {
                "model_artifact_hash": measured["model_artifact_hash"],
                "model_manifest_hash": measured["model_manifest_hash"],
                "source_bundle_hash": measured["source_bundle_hash"],
                "compatibility_policy_hash": measured["compatibility_policy_hash"],
                "compatibility_admission_hash": measured["compatibility_admission_hash"],
                "runtime_config_hash": measured["runtime_config_hash"],
                "provider_runtime_catalog_hash": measured["provider_runtime_catalog_hash"],
                "trace_entries_hash": measured["trace_entries_hash"],
                "output_hash": measured["output_hash"],
                "runtime_catalog_sha256": measured["output"]["runtime_routing"]["catalog_sha256"],
                "runtime_policy_sha256": measured["output"]["runtime_routing"]["policy_sha256"],
            },
            "observation": observation,
        }
        return ExecutionResultV2(
            output=output,
            artifact_hashes=tuple(
                str(value)
                for value in (
                    measured["model_artifact_hash"],
                    measured["model_manifest_hash"],
                    measured["source_bundle_hash"],
                    measured["compatibility_policy_hash"],
                    measured["compatibility_admission_hash"],
                    measured["runtime_config_hash"],
                    measured["provider_runtime_catalog_hash"],
                    measured["trace_entries_hash"],
                    measured["output_hash"],
                    measured["output"]["runtime_routing"]["catalog_sha256"],
                    measured["output"]["runtime_routing"]["policy_sha256"],
                )
            ),
        )

    async def _execute_protected_routing_provider_terminal(
        self,
        payload: Mapping[str, Any],
        context: ExecutionContextV2,
    ) -> ExecutionResultV2:
        """Normalize one provider response inside the protected scorer.

        The job payload carries only signed evidence and exact transport
        commitments.  The reviewed catalog and immutable unit dataset are
        provisioned when this executor is constructed; the caller cannot
        inject a compiler, action policy, or billing implementation.
        """

        required = {
            "schema_version",
            "authorization_proof",
            "prepared_call",
            "broker_request",
            "broker_result",
            "provider_record",
            "raw_response_body_b64",
        }
        if not isinstance(payload, Mapping) or set(payload) != required:
            raise ValueError("routing provider terminal payload is invalid")
        if payload.get("schema_version") != (
            "leadpoet.routing_provider_terminal_request.v2"
        ):
            raise ValueError("routing provider terminal schema is invalid")
        if self._routing_binding_catalog is None or self._routing_unit_dataset is None:
            raise ValueError("routing provider terminal authorities are unavailable")
        if self._routing_coordinator_boot_identity_supplier is None:
            raise ValueError(
                "routing provider trusted coordinator identity is unavailable"
            )
        try:
            prepared = prepared_routing_provider_call_from_mapping(
                payload["prepared_call"]
            )
            raw_body = base64.b64decode(
                str(payload["raw_response_body_b64"]), validate=True
            )
        except Exception as exc:  # noqa: BLE001 - protected boundary
            raise ValueError("routing provider terminal payload encoding is invalid") from exc
        if not raw_body or len(raw_body) > 8 * 1024 * 1024:
            raise ValueError("routing provider terminal response body is invalid")
        try:
            proof = payload["authorization_proof"]
            if not isinstance(proof, Mapping):
                raise ValueError("authorization proof is not an object")
            authorization_receipt = proof.get("authorization_receipt")
            if not isinstance(authorization_receipt, Mapping):
                raise ValueError("authorization receipt is not an object")
            if authorization_receipt.get("receipt_hash") not in set(
                context.parent_receipt_hashes
            ):
                raise ValueError("authorization receipt is not a declared parent")
            output = execute_protected_routing_provider_terminal_v2(
                authorization_proof=proof,
                prepared_call=prepared,
                broker_request=payload["broker_request"],
                broker_result=payload["broker_result"],
                provider_record=payload["provider_record"],
                trusted_coordinator_boot_identity=(
                    self._routing_coordinator_boot_identity_supplier()
                ),
                raw_response_body=raw_body,
                binding_catalog=self._routing_binding_catalog,
                unit_dataset=self._routing_unit_dataset,
            )
        except ProtectedRoutingProviderTerminalError as exc:
            raise ValueError("routing provider terminal validation failed") from exc
        return ExecutionResultV2(
            output=output,
            # The manager hashes this exact redacted result for the standard
            # job receipt. The protected function does not create a second
            # terminal signer or replace the manager's roots.
            receipt_output=output,
            artifact_hashes=(),
        )

    async def _execute_protected_routing_provider_dispatch(
        self,
        payload: Mapping[str, Any],
        context: ExecutionContextV2,
    ) -> ExecutionResultV2:
        """Dispatch one exact compiled routing call through the coordinator.

        This is the only host-facing provider dispatch seam for the reviewed
        routing product.  The host cannot supply a broker result, provider
        record, response body, compiler, or provider singleton.  All of those
        values are obtained and checked inside the scoring enclave after the
        signed authorization and parent ancestry have been validated.
        """

        required = {
            "schema_version",
            "authorization_proof",
            "prepared_call",
            "broker_request",
            "budget_reservation",
        }
        if (
            not isinstance(payload, Mapping)
            or frozenset(payload)
            not in {
                frozenset(required),
                frozenset(required | {"model_completion_contract"}),
            }
        ):
            raise ValueError("routing provider dispatch payload is invalid")
        if payload.get("schema_version") != ROUTING_PROVIDER_DISPATCH_REQUEST_SCHEMA_V2:
            raise ValueError("routing provider dispatch schema is invalid")
        if self._routing_binding_catalog is None or self._routing_unit_dataset is None:
            raise ValueError("routing provider dispatch authorities are unavailable")
        if self._routing_coordinator_boot_identity_supplier is None:
            raise ValueError(
                "routing provider trusted coordinator identity is unavailable"
            )

        try:
            prepared = prepared_routing_provider_call_from_mapping(
                payload["prepared_call"]
            )
            proof = payload["authorization_proof"]
            broker_request = payload["broker_request"]
            if not isinstance(proof, Mapping) or not isinstance(broker_request, Mapping):
                raise ValueError("routing provider dispatch documents are invalid")
            authorization = RoutingProviderCallAuthorizationV2.from_mapping(
                proof["authorization"]
            )
            if authorization.purpose != ROUTING_PROVIDER_DISPATCH_PURPOSE_V2:
                raise ValueError("routing provider dispatch authorization purpose differs")
            authorization_receipt = proof["authorization_receipt"]
            if not isinstance(authorization_receipt, Mapping):
                raise ValueError("routing provider dispatch authorization receipt is invalid")
            receipt_hash = str(authorization_receipt.get("receipt_hash") or "")
            if receipt_hash not in set(context.parent_receipt_hashes):
                raise ValueError("routing provider dispatch parent is not declared")
            dispatch_job_id = routing_provider_dispatch_job_id_v2(proof)
            if (
                context.job_id != dispatch_job_id
                or broker_request.get("job_id") != dispatch_job_id
            ):
                raise ValueError("routing provider dispatch job differs")
            matching_parent = any(
                receipt == dict(authorization_receipt)
                for graph in context.external_receipt_graphs
                for receipt in graph.get("receipts") or ()
                if isinstance(receipt, Mapping)
            )
            if not matching_parent:
                raise ValueError("routing provider dispatch parent receipt is unavailable")

            # Verify the complete signed proof before the provider boundary.
            # The compact request fields are also checked against a fresh
            # compiler projection, so a host cannot alter the URL, body, or
            # retry identity while retaining a valid proof.
            from gateway.tee.provider_broker_v2 import (
                validate_routing_authorization_proof_v2,
            )

            validate_routing_authorization_proof_v2(proof, broker_request)
            if (
                prepared.binding != authorization.binding
                or prepared.binding_catalog_manifest_hash
                != authorization.binding_catalog_manifest_hash
                or prepared.binding_catalog_version
                != authorization.binding_catalog_version
                or prepared.action_id != authorization.action_id
                or prepared.transport_id != authorization.transport_id
                or prepared.unit_ref != authorization.unit_ref
                or prepared.unit_input_hash != authorization.unit_input_hash
                or prepared.unit_dataset_manifest_hash
                != authorization.unit_dataset_manifest_hash
                or prepared.unit_set_hash != authorization.unit_set_hash
                or prepared.request_body_hash != authorization.request_body_hash
                or prepared.retry_policy_hash != authorization.retry_policy_hash
                or prepared.credit_ceiling_microunits
                != authorization.credit_cap_microunits
                or prepared.timeout_ms != authorization.timeout_ms
            ):
                raise ValueError("routing provider dispatch prepared call differs")
            compiler = ReviewedDeeplineActionCompiler(
                binding_catalog=self._routing_binding_catalog,
                unit_dataset=self._routing_unit_dataset,
            )
            expected_request = dict(
                compiler.broker_request(
                    prepared=prepared,
                    experiment_hash=authorization.experiment_hash,
                    dispatch_job_id=dispatch_job_id,
                    variant_id=authorization.variant_id,
                    attempt_number=authorization.attempt,
                    core_request_fingerprint=authorization.core_request_fingerprint,
                    authorization_hash=str(proof["authorization_hash"]),
                    authorization_proof_hash=str(proof["authorization_proof_hash"]),
                )
            )
            expected_request["routing_authorization"] = dict(proof)
            if dict(broker_request) != expected_request:
                raise ValueError("routing provider dispatch compiled request differs")
            budget_reservation = validate_routing_budget_reservation_v3(
                payload["budget_reservation"],
                authorization=authorization,
                prepared_call=prepared,
            )
            model_completion_contract = None
            if "model_completion_contract" in payload:
                model_completion_contract = (
                    validate_routing_model_completion_contract_v1(
                        payload["model_completion_contract"]
                    )
                )
        except Exception as exc:  # noqa: BLE001 - protected boundary
            raise ValueError("routing provider dispatch authorization is invalid") from exc

        # The budget authority is invoked inside this measured operation. A
        # caller that submits the dispatch job directly therefore cannot cross
        # the paid-provider boundary without first creating the exact durable
        # reservation under the active queue-fenced claim.
        try:
            (
                budget_reservation_proof,
                budget_transport_attempt,
                budget_artifact_hashes,
            ) = self._reserve_routing_budget_v3(
                reservation=budget_reservation,
                context=context,
            )
        except Exception as exc:  # noqa: BLE001 - protected boundary
            raise ValueError("routing provider budget reservation failed") from exc

        # This is the measured scoring -> coordinator path.  Its result never
        # enters the host payload; only the normalizer's bounded projection is
        # returned below.
        broker_result = self._provider_execute(dict(broker_request))
        if not isinstance(broker_result, Mapping):
            raise ValueError("routing provider dispatch coordinator result is invalid")
        provider_record = broker_result.get("routing_provider_record")
        if provider_record is None:
            provider_record = broker_result.get("provider_record")
        if not isinstance(provider_record, Mapping):
            raise ValueError("routing provider dispatch coordinator evidence is missing")
        try:
            raw_response_body = base64.b64decode(
                str(broker_result.get("body_b64") or ""), validate=True
            )
            output = execute_protected_routing_provider_terminal_v2(
                authorization_proof=proof,
                prepared_call=prepared,
                broker_request=broker_request,
                broker_result=broker_result,
                provider_record=provider_record,
                trusted_coordinator_boot_identity=(
                    self._routing_coordinator_boot_identity_supplier()
                ),
                raw_response_body=raw_response_body,
                binding_catalog=self._routing_binding_catalog,
                unit_dataset=self._routing_unit_dataset,
                model_completion_contract=model_completion_contract,
            )
            output = {
                **output,
                "budget_reservation": budget_reservation_proof,
            }
        except (ProtectedRoutingProviderTerminalError, ValueError, TypeError) as exc:
            raise ValueError("routing provider dispatch terminal validation failed") from exc
        transport_attempts = [budget_transport_attempt]
        provider_attempt = broker_result.get("transport_attempt")
        if not isinstance(provider_attempt, Mapping):
            raise ValueError("routing provider dispatch transport attempt is missing")
        transport_attempts.append(dict(provider_attempt))
        additional_attempts = broker_result.get("additional_transport_attempts") or ()
        if (
            not isinstance(additional_attempts, (list, tuple))
            or any(not isinstance(item, Mapping) for item in additional_attempts)
        ):
            raise ValueError("routing provider dispatch transport attempts are invalid")
        transport_attempts.extend(dict(item) for item in additional_attempts)
        deduplicated_attempts: list[dict[str, Any]] = []
        seen_attempt_hashes: set[str] = set()
        for attempt in transport_attempts:
            try:
                validate_transport_attempt(attempt)
            except Exception as exc:  # noqa: BLE001 - protected boundary
                raise ValueError(
                    "routing provider dispatch transport attempt is invalid"
                ) from exc
            attempt_hash = str(attempt["attempt_hash"])
            if attempt_hash not in seen_attempt_hashes:
                seen_attempt_hashes.add(attempt_hash)
                deduplicated_attempts.append(dict(attempt))
        artifact_hashes = set(budget_artifact_hashes)
        provider_artifact_hashes = broker_result.get("evidence_artifact_hashes") or ()
        if (
            not isinstance(provider_artifact_hashes, (list, tuple))
            or any(not re.fullmatch(r"sha256:[0-9a-f]{64}", str(item or "")) for item in provider_artifact_hashes)
        ):
            raise ValueError("routing provider dispatch artifact hashes are invalid")
        artifact_hashes.update(str(item) for item in provider_artifact_hashes)
        return ExecutionResultV2(
            output=output,
            receipt_output=routing_provider_dispatch_receipt_output_v2(output),
            transport_attempts=tuple(deduplicated_attempts),
            artifact_hashes=tuple(sorted(artifact_hashes)),
        )

    def _reserve_routing_budget_v3(
        self,
        *,
        reservation: Mapping[str, Any],
        context: ExecutionContextV2,
    ) -> tuple[dict[str, Any], dict[str, Any], tuple[str, ...]]:
        """Reserve the exact call cap through measured coordinator transport."""

        from gateway.tee.provider_broker_v2 import PROVIDER_BROKER_SCHEMA_VERSION
        from leadpoet_canonical.production_parity_boundary_v2 import (
            validate_production_parity_boundary_document_v2,
        )

        retry_policy_hash = str(self._retry_policy_hashes.get("supabase") or "")
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", retry_policy_hash):
            raise ValueError("routing budget retry policy is unavailable")
        boundary = validate_production_parity_boundary_document_v2(
            self._execution_config["behavior_environment"],
            network=str(self._execution_config["deployment"]["network"]),
            netuid=int(self._execution_config["deployment"]["netuid"]),
        )
        origin = str(boundary["supabase_origin"]).rstrip("/")
        rpc_params = {
            "p_event_key": reservation["event_key"],
            "p_reservation_id": reservation["reservation_id"],
            "p_experiment_hash": reservation["experiment_hash"],
            "p_binding_id": reservation["binding_id"],
            "p_claim_key": reservation["claim_key"],
            "p_claim_generation": reservation["claim_generation"],
            "p_credit_microunits": reservation["credit_microunits"],
            "p_lease_seconds": reservation["lease_seconds"],
            "p_event_doc": reservation["event_doc"],
        }
        request_body = canonical_json(rpc_params).encode("utf-8")
        request = {
            "schema_version": PROVIDER_BROKER_SCHEMA_VERSION,
            "logical_operation_id": (
                f"{context.job_id}:routing-budget-reservation:"
                + str(reservation["event_key"]).split(":", 1)[1][:32]
            ),
            "job_id": context.job_id,
            "purpose": ROUTING_BUDGET_RESERVATION_PURPOSE_V3,
            "provider_id": "supabase",
            "attempt_number": 0,
            "method": "POST",
            "url": (
                origin
                + "/rest/v1/rpc/research_lab_routing_reserve_budget_v3"
            ),
            "headers": {
                "accept": "application/json",
                "content-type": "application/json",
            },
            "body_b64": base64.b64encode(request_body).decode("ascii"),
            "timeout_ms": 5_000,
            "retry_policy_hash": retry_policy_hash,
        }
        result = self._provider_execute(request)
        if not isinstance(result, Mapping):
            raise ValueError("routing budget coordinator result is invalid")
        attempt = result.get("transport_attempt")
        if not isinstance(attempt, Mapping):
            raise ValueError("routing budget transport attempt is missing")
        validate_transport_attempt(attempt)
        try:
            http_status = int(result.get("http_status") or 0)
            response_body = base64.b64decode(
                str(result.get("body_b64") or ""), validate=True
            )
        except Exception as exc:  # noqa: BLE001 - protected boundary
            raise ValueError("routing budget response is invalid") from exc
        if (
            result.get("terminal_status") != "authenticated_response"
            or not 200 <= http_status < 300
            or attempt.get("terminal_status") != "authenticated_response"
            or attempt.get("provider_id") != "supabase"
            or attempt.get("purpose") != ROUTING_BUDGET_RESERVATION_PURPOSE_V3
            or attempt.get("job_id") != context.job_id
            or attempt.get("logical_operation_id")
            != request["logical_operation_id"]
            or attempt.get("body_hash") != sha256_bytes(request_body)
            or attempt.get("response_hash") != sha256_bytes(response_body)
            or attempt.get("retry_policy_hash") != retry_policy_hash
        ):
            raise ValueError("routing budget authenticated response differs")
        try:
            response_document = json.loads(response_body.decode("utf-8"))
        except Exception as exc:  # noqa: BLE001 - protected boundary
            raise ValueError("routing budget response JSON is invalid") from exc
        normalized_result = validate_routing_budget_reservation_result_v3(
            response_document,
            reservation=reservation,
        )
        proof = routing_budget_reservation_proof_v3(
            reservation_result=normalized_result,
            response_hash=str(attempt["response_hash"]),
            transport_attempt_hash=str(attempt["attempt_hash"]),
        )
        artifact_hashes = result.get("evidence_artifact_hashes") or ()
        if (
            not isinstance(artifact_hashes, (list, tuple))
            or any(
                not re.fullmatch(r"sha256:[0-9a-f]{64}", str(item or ""))
                for item in artifact_hashes
            )
        ):
            raise ValueError("routing budget artifact hashes are invalid")
        return proof, dict(attempt), tuple(str(item) for item in artifact_hashes)

    def _validate_model_provider_catalog_ancestry(
        self,
        payload: Mapping[str, Any],
        context: ExecutionContextV2,
    ) -> None:
        evidence = payload.get("provider_catalog_evidence")
        if not isinstance(evidence, Mapping) or set(evidence) != {
            "result",
            "root_receipt_hash",
        }:
            raise ValueError("model provider catalog evidence is invalid")
        result = evidence.get("result")
        root_hash = str(evidence.get("root_receipt_hash") or "")
        if not isinstance(result, Mapping):
            raise ValueError("model provider catalog result is invalid")
        matching_roots = []
        for graph in context.external_receipt_graphs:
            matching_roots.extend(
                item
                for item in graph.get("receipts") or ()
                if isinstance(item, Mapping)
                and item.get("receipt_hash") == root_hash
            )
        for proof in context.external_ancestry_proofs:
            matching_roots.extend(
                item
                for item in proof.get("disclosed_receipts") or ()
                if isinstance(item, Mapping)
                and item.get("receipt_hash") == root_hash
            )
        unique_roots = {
            sha256_json(dict(item)): item for item in matching_roots
        }
        matching_root = next(iter(unique_roots.values()), None)
        if (
            len(unique_roots) != 1
            or not isinstance(matching_root, Mapping)
            or matching_root.get("role") != "gateway_coordinator"
            or matching_root.get("purpose")
            != "research_lab.source_add_catalog_snapshot.v2"
            or matching_root.get("status") != "succeeded"
            or matching_root.get("output_root")
            != sha256_json(dict(result))
        ):
            raise ValueError("model provider catalog ancestry differs")

    def _validate_baseline_configuration(self, payload: Mapping[str, Any]) -> None:
        try:
            max_unresolved = max(
                0,
                int(
                    os.getenv(
                        "RESEARCH_LAB_BASELINE_MAX_UNRESOLVED_ICPS",
                        "2",
                    )
                ),
            )
        except ValueError:
            max_unresolved = 2
        expected = {
            "public_icps_per_day": self._config.public_benchmark_public_icps_per_day,
            "public_weak_per_day": self._config.public_benchmark_public_weak_per_day,
            "public_total_icps": self._config.public_benchmark_public_total_icps,
            "public_weak_total": self._config.public_benchmark_public_weak_total,
            "max_unresolved_icps": max_unresolved,
        }
        if any(payload.get(name) != value for name, value in expected.items()):
            raise ValueError(
                "baseline policy differs from measured configuration"
            )
        measured_conditional_policy = self._config.conditional_validation_policy()
        supplied_conditional_policy = payload.get("conditional_validation_policy")
        if measured_conditional_policy.enabled:
            if supplied_conditional_policy != measured_conditional_policy.to_dict():
                raise ValueError(
                    "conditional validation policy differs from measured configuration"
                )
        elif supplied_conditional_policy is not None:
            raise ValueError(
                "conditional validation policy is disabled in measured configuration"
            )

    def _validate_conditional_preliminary_ancestry(
        self,
        payload: Mapping[str, Any],
        context: ExecutionContextV2,
    ) -> None:
        extra = payload.get("extra_bundle_fields")
        gate = extra.get("private_holdout_gate") if isinstance(extra, Mapping) else None
        if not isinstance(gate, Mapping) or not bool(
            gate.get("conditional_validation_required")
        ):
            return
        if not bool(gate.get("conditional_holdout_evaluated")):
            return
        proof = gate.get("preliminary_promotion_gate")
        required = {
            "schema_version",
            "status",
            "preliminary_score_bundle_hash",
            "score_bundle_receipt_hash",
            "promotion_metric_receipt_hash",
            "promotion_decision_receipt_hash",
            "promotion_decision_output_root",
            "candidate_artifact_hash",
            "candidate_parent_artifact_hash",
            "active_parent_artifact_hash",
            "rolling_window_hash",
            "category_assignment_hash",
            "conditional_validation_policy_hash",
            "scoring_configuration_hash",
            "threshold_points",
            "decision",
            "proof_hash",
        }
        if not isinstance(proof, Mapping) or set(proof) != required:
            raise ValueError("conditional preliminary promotion proof is invalid")
        proof_body = {key: proof[key] for key in proof if key != "proof_hash"}
        if (
            proof.get("schema_version")
            != "research_lab_preliminary_promotion_gate.v1"
            or proof.get("proof_hash") != sha256_json(proof_body)
            or proof.get("status") != "promotion_passed"
        ):
            raise ValueError("conditional preliminary promotion proof differs")
        hash_fields = {
            "preliminary_score_bundle_hash",
            "score_bundle_receipt_hash",
            "promotion_metric_receipt_hash",
            "promotion_decision_receipt_hash",
            "promotion_decision_output_root",
            "candidate_artifact_hash",
            "candidate_parent_artifact_hash",
            "active_parent_artifact_hash",
            "rolling_window_hash",
            "category_assignment_hash",
            "conditional_validation_policy_hash",
            "scoring_configuration_hash",
        }
        if any(
            not re.fullmatch(r"sha256:[0-9a-f]{64}", str(proof.get(field) or ""))
            for field in hash_fields
        ):
            raise ValueError("conditional preliminary promotion proof hash is invalid")
        decision = proof.get("decision")
        if not isinstance(decision, Mapping) or set(decision) != {
            "status",
            "improvement_points",
            "threshold_points",
            "candidate_kind",
            "auto_promotion_enabled",
            "active_parent_matches",
            "metric_rejection_status",
        }:
            raise ValueError("conditional preliminary promotion decision is invalid")
        if (
            decision.get("status") != "promotion_passed"
            or decision.get("candidate_kind") != "image_build"
            or decision.get("auto_promotion_enabled") is not True
            or decision.get("active_parent_matches") is not True
            or decision.get("metric_rejection_status") is not None
        ):
            raise ValueError("conditional preliminary promotion decision did not pass")
        threshold = float(proof.get("threshold_points"))
        if (
            threshold != float(gate.get("threshold_points"))
            or threshold != float(decision.get("threshold_points"))
            or threshold != float(self._config.improvement_threshold_points)
            or float(decision.get("improvement_points")) < threshold
        ):
            raise ValueError("conditional preliminary promotion threshold differs")
        artifact_manifest = payload.get("artifact_manifest")
        candidate_manifest = payload.get("candidate_artifact_manifest")
        patch_manifest = payload.get("patch_manifest")
        run_context = payload.get("run_context")
        if not all(
            isinstance(value, Mapping)
            for value in (
                artifact_manifest,
                candidate_manifest,
                patch_manifest,
                run_context,
            )
        ):
            raise ValueError("conditional preliminary promotion inputs are invalid")
        candidate_parent = str(proof.get("candidate_parent_artifact_hash") or "")
        if (
            proof.get("candidate_artifact_hash")
            != candidate_manifest.get("model_artifact_hash")
            or candidate_parent != artifact_manifest.get("model_artifact_hash")
            or candidate_parent != patch_manifest.get("parent_artifact_hash")
            or proof.get("active_parent_artifact_hash") != candidate_parent
            or proof.get("rolling_window_hash")
            != str(run_context.get("rolling_window_hash") or "")
            or proof.get("category_assignment_hash")
            != gate.get("category_assignment_hash")
            or proof.get("conditional_validation_policy_hash")
            != gate.get("conditional_validation_policy_hash")
            or proof.get("scoring_configuration_hash") != configuration_hash()
        ):
            raise ValueError("conditional preliminary promotion commitment differs")
        expected_output_root = sha256_json({"decision": dict(decision)})
        if proof.get("promotion_decision_output_root") != expected_output_root:
            raise ValueError("conditional preliminary promotion output differs")
        decision_hash = str(proof.get("promotion_decision_receipt_hash") or "")
        if decision_hash not in set(getattr(context, "parent_receipt_hashes", ())):
            raise ValueError("conditional preliminary promotion ancestry is missing")
        matching_graphs = []
        for graph in context.external_receipt_graphs:
            receipts = {
                str(receipt.get("receipt_hash") or ""): receipt
                for receipt in graph.get("receipts") or ()
                if isinstance(receipt, Mapping) and receipt.get("receipt_hash")
            }
            if decision_hash in receipts:
                matching_graphs.append((graph, receipts))
        if (
            len(matching_graphs) != 1
            or matching_graphs[0][0].get("root_receipt_hash") != decision_hash
        ):
            raise ValueError("conditional preliminary promotion ancestry is missing")
        _, receipts = matching_graphs[0]
        receipt = receipts[decision_hash]
        metric_hash = str(proof.get("promotion_metric_receipt_hash") or "")
        score_bundle_hash = str(proof.get("score_bundle_receipt_hash") or "")
        metric_receipt = receipts.get(metric_hash)
        score_bundle_receipt = receipts.get(score_bundle_hash)
        if (
            receipt.get("role") != "gateway_coordinator"
            or receipt.get("purpose") != "research_lab.promotion_decision.v2"
            or receipt.get("status") != "succeeded"
            or receipt.get("output_root")
            != proof.get("promotion_decision_output_root")
            or receipt.get("parent_receipt_hashes") != [metric_hash]
            or not isinstance(metric_receipt, Mapping)
            or metric_receipt.get("role") != "gateway_coordinator"
            or metric_receipt.get("purpose") != "research_lab.ranking.v2"
            or metric_receipt.get("status") != "succeeded"
            or metric_receipt.get("parent_receipt_hashes") != [score_bundle_hash]
            or not isinstance(score_bundle_receipt, Mapping)
            or score_bundle_receipt.get("role") != "gateway_scoring"
            or score_bundle_receipt.get("purpose")
            != "research_lab.candidate_score.v2"
            or score_bundle_receipt.get("status") != "succeeded"
        ):
            raise ValueError("conditional preliminary promotion ancestry differs")

    async def _execute_source_add_judge(
        self,
        payload: Mapping[str, Any],
        context: ExecutionContextV2,
    ) -> ExecutionResultV2:
        required = {
            "schema_version",
            "candidate",
            "score_bundle",
            "provisioned_sources",
            "timeout_seconds",
        }
        if not isinstance(payload, Mapping) or set(payload) != required:
            raise ValueError("SOURCE_ADD judge request fields are invalid")
        if payload.get("schema_version") != SOURCE_ADD_JUDGE_REQUEST_SCHEMA_VERSION:
            raise ValueError("SOURCE_ADD judge request schema is invalid")
        candidate = payload.get("candidate")
        score_bundle = payload.get("score_bundle")
        provisioned_sources = payload.get("provisioned_sources")
        timeout_seconds = payload.get("timeout_seconds")
        if not isinstance(candidate, Mapping) or not isinstance(score_bundle, Mapping):
            raise ValueError("SOURCE_ADD judge business inputs are invalid")
        if not isinstance(provisioned_sources, list) or any(
            not isinstance(item, Mapping) for item in provisioned_sources
        ):
            raise ValueError("SOURCE_ADD judge source rows are invalid")
        if (
            not isinstance(timeout_seconds, int)
            or isinstance(timeout_seconds, bool)
            or timeout_seconds <= 0
            or timeout_seconds > 600
        ):
            raise ValueError("SOURCE_ADD judge timeout is invalid")
        catalog_roots = []
        for graph in context.external_receipt_graphs:
            root_hash = str(graph.get("root_receipt_hash") or "")
            receipts = {
                str(item.get("receipt_hash") or ""): item
                for item in graph.get("receipts") or ()
                if isinstance(item, Mapping)
            }
            root = receipts.get(root_hash)
            if isinstance(root, Mapping) and root.get("purpose") == (
                "research_lab.source_add_catalog_snapshot.v2"
            ):
                catalog_roots.append(root)
        expected_catalog_output = {
            "schema_version": "leadpoet.source_add_catalog_snapshot.v2",
            "provisioned_sources": [dict(item) for item in provisioned_sources],
            "provisioned_sources_hash": sha256_json(
                [dict(item) for item in provisioned_sources]
            ),
        }
        if (
            len(catalog_roots) != 1
            or catalog_roots[0].get("output_root")
            != sha256_json(expected_catalog_output)
        ):
            raise ValueError("SOURCE_ADD judge catalog ancestry differs")

        from gateway.research_lab.source_add_llm_judge import (
            judge_source_add_implementation,
        )
        from gateway.research_lab.store import canonical_hash

        with self._transport.scope(
            job_id=context.job_id,
            purpose=context.purpose,
            logical_operation_id=context.job_id,
            retry_policy_hashes=self._retry_policy_hashes,
            terminal_sink=context.record_transport,
            artifact_sink=context.record_artifact,
            allow_transport_failures=True,
        ):
            verdict = await judge_source_add_implementation(
                api_key="leadpoet-v2-brokered-credential",
                candidate=dict(candidate),
                score_bundle=dict(score_bundle),
                provisioned_sources=[dict(item) for item in provisioned_sources],
                timeout_seconds=timeout_seconds,
            )
        verdict_doc = {
            "verdict": verdict.verdict,
            "confidence": float(verdict.confidence),
            "source_used": bool(verdict.source_used),
            "adapter_id": verdict.adapter_id,
            "registry_provider_id": verdict.registry_provider_id,
            "evidence_summary": verdict.evidence_summary,
            "reason_codes": list(verdict.reason_codes),
            "model_id": verdict.model_id,
            "provider_usage": dict(verdict.provider_usage),
            "judge_doc_hash": canonical_hash(verdict.raw_doc or {}),
        }
        output = {
            "schema_version": SOURCE_ADD_JUDGE_RESULT_SCHEMA_VERSION,
            "candidate_id": str(candidate.get("candidate_id") or ""),
            "score_bundle_hash": str(score_bundle.get("score_bundle_hash") or ""),
            "provisioned_sources_hash": sha256_json(
                [dict(item) for item in provisioned_sources]
            ),
            "verdict": verdict_doc,
        }
        return ExecutionResultV2(
            output=output,
            artifact_hashes=(
                verdict_doc["judge_doc_hash"],
                output["provisioned_sources_hash"],
            ),
        )

    async def _execute_provider_preflight(
        self,
        payload: Mapping[str, Any],
        context: ExecutionContextV2,
    ) -> ExecutionResultV2:
        required = {
            "schema_version",
            "measurement_id",
            "scope_key",
            "force",
            "settings",
        }
        if not isinstance(payload, Mapping) or set(payload) != required:
            raise ValueError("provider preflight request fields are invalid")
        if payload.get("schema_version") != PROVIDER_PREFLIGHT_REQUEST_SCHEMA_VERSION:
            raise ValueError("provider preflight request schema is invalid")
        measurement_id = str(payload.get("measurement_id") or "")
        if not re.fullmatch(r"[0-9a-f]{32}", measurement_id):
            raise ValueError("provider preflight measurement identity is invalid")
        scope_key = str(payload.get("scope_key") or "")
        if not scope_key or len(scope_key) > 255 or "\x00" in scope_key:
            raise ValueError("provider preflight scope is invalid")
        force = payload.get("force")
        settings = payload.get("settings")
        if not isinstance(force, bool) or not isinstance(settings, Mapping):
            raise ValueError("provider preflight controls are invalid")
        expected_settings = {
            "enabled",
            "ttl_seconds",
            "timeout_seconds",
            "failure_streak_threshold",
        }
        if set(settings) != expected_settings:
            raise ValueError("provider preflight settings fields are invalid")
        from gateway.research_lab.provider_preflight import (
            provider_preflight_settings,
        )

        if dict(settings) != provider_preflight_settings():
            raise ValueError(
                "provider preflight settings differ from measured configuration"
            )
        with self._preflight_lock:
            preflight = self._preflight_by_scope.setdefault(
                scope_key,
                ProviderPreflight(),
            )
        def measured_preflight() -> dict[str, Any]:
            # Keep the cache transaction and transport-scope finalizer on the
            # same thread so a missing terminal rolls back the verdict/streak.
            # Provider probes are blocking, so this entire unit belongs in the
            # executor thread rather than on the enclave event loop.
            with preflight.measurement_transaction():
                with self._transport.scope(
                    job_id=context.job_id,
                    purpose=context.purpose,
                    logical_operation_id=context.job_id,
                    retry_policy_hashes=self._retry_policy_hashes,
                    terminal_sink=context.record_transport,
                    artifact_sink=context.record_artifact,
                    allow_transport_failures=True,
                ):
                    return preflight.check(
                        force=force,
                        settings=dict(settings),
                    )

        result = await asyncio.to_thread(measured_preflight)
        return ExecutionResultV2(output=dict(result))

    async def _execute_dev_replay(
        self,
        payload: Mapping[str, Any],
        context: ExecutionContextV2,
    ) -> ExecutionResultV2:
        if self._model_sandbox is None:
            raise ValueError("measured model sandbox is unavailable")
        required = {
            "schema_version",
            "artifact",
            "source_bundle",
            "snapshot_bundle",
            "snapshot_tree_hash",
            "snapshot_manifest_hash",
            "dev_selection_request",
            "module_name",
            "callable_name",
            "environment",
            "credential_env_names",
            "run_label",
            "cohort_hash",
            "miss_policy",
            "per_icp_timeout_seconds",
            "total_timeout_seconds",
        }
        if not isinstance(payload, Mapping) or set(payload) != required:
            raise ValueError("dev replay request fields are invalid")
        if payload.get("schema_version") != DEV_REPLAY_REQUEST_SCHEMA_VERSION:
            raise ValueError("dev replay request schema is invalid")
        artifact = PrivateModelArtifactManifest.from_mapping(payload["artifact"])
        source_bundle = dict(payload["source_bundle"])
        snapshot_bundle = dict(payload["snapshot_bundle"])
        snapshot_tree_hash = str(payload.get("snapshot_tree_hash") or "")
        snapshot_manifest_hash = str(payload.get("snapshot_manifest_hash") or "")
        selection_request = payload.get("dev_selection_request")
        if not isinstance(selection_request, Mapping) or set(selection_request) != {
            "selection_seed",
            "miner_direction",
            "selection_manifest_hash",
        }:
            raise ValueError("dev replay selection request is invalid")
        if snapshot_bundle.get("source_tree_hash") != snapshot_tree_hash:
            raise ValueError("dev replay snapshot bundle commitment differs")
        environment = payload.get("environment")
        credential_env_names = payload.get("credential_env_names")
        if not isinstance(environment, Mapping) or not isinstance(
            credential_env_names, list
        ):
            raise ValueError("dev replay environment fields are invalid")
        if dict(environment) != measured_dev_replay_environment(
            self._execution_config
        ):
            raise ValueError("dev replay environment differs from measured policy")
        if credential_env_names != list(
            measured_credential_environment_names(self._execution_config)
        ):
            raise ValueError(
                "dev replay credential environment differs from measured policy"
            )
        per_icp_timeout = int(payload["per_icp_timeout_seconds"])
        total_timeout = int(payload["total_timeout_seconds"])
        if per_icp_timeout < 10 or total_timeout < 30:
            raise ValueError("dev replay timeout is invalid")
        run_label = str(payload.get("run_label") or "")
        cohort_hash = str(payload.get("cohort_hash") or "")
        if not _HASH_RE.fullmatch(cohort_hash):
            raise ValueError("dev replay cohort commitment is invalid")
        if len(run_label.encode("utf-8")) > 1024:
            raise ValueError("dev replay run label is too large")

        with tempfile.TemporaryDirectory(prefix="lp-dev-snapshots-v2-") as tmp:
            snapshot_root = Path(tmp) / "snapshot-set"
            snapshot_evidence = extract_source_bundle_v2(
                snapshot_bundle,
                destination=snapshot_root,
                expected_source_tree_hash=snapshot_tree_hash,
            )
            for path in sorted(snapshot_root.rglob("*"), reverse=True):
                if path.is_file():
                    path.chmod(0o444)
                elif path.is_dir():
                    path.chmod(0o555)
            snapshot_root.chmod(0o555)
            snapshot_store = ProviderSnapshotStore(
                str(snapshot_root),
                mode=MODE_REPLAY,
                miss_policy=str(payload["miss_policy"]),
            )
            manifest = snapshot_store.load_manifest()
            verification = snapshot_store.verify_manifest(manifest)
            if (
                manifest is None
                or not verification["passed"]
                or str(manifest.get("manifest_hash") or "")
                != snapshot_manifest_hash
            ):
                raise ValueError("dev replay snapshot manifest verification failed")
            bank_items = snapshot_store.load_dev_icp_items() or []
            expected_dev_icp_count = measured_git_tree_config(
                self._execution_config
            ).live_max_icps_per_node
            if not expected_dev_icp_count <= len(bank_items) <= 100:
                raise ValueError(
                    "dev replay bank size cannot satisfy measured Git-tree policy"
                )
            if total_timeout != measured_dev_eval_total_timeout_seconds(
                self._execution_config
            ):
                raise ValueError("dev replay total timeout differs from measured policy")
            if per_icp_timeout != measured_dev_eval_icp_timeout_seconds(
                self._execution_config,
                item_count=expected_dev_icp_count,
            ):
                raise ValueError("dev replay ICP timeout differs from measured policy")
            if str(payload["miss_policy"]) != measured_dev_snapshot_miss_policy(
                self._execution_config
            ):
                raise ValueError("dev replay miss policy differs from measured policy")
            expected_dev_set_hash = str(manifest.get("icp_set_hash") or "")
            if compute_dev_set_hash(bank_items) != expected_dev_set_hash:
                raise ValueError("dev replay ICP bank commitment differs")
            selection = select_snapshot_dev_icps(
                bank_items,
                snapshot_manifest=manifest,
                size=expected_dev_icp_count,
                seed=str(selection_request.get("selection_seed") or ""),
                miner_direction=str(
                    selection_request.get("miner_direction") or ""
                ),
            )
            if str(selection_request.get("selection_manifest_hash") or "") != str(
                selection.manifest.get("selection_manifest_hash") or ""
            ):
                raise ValueError("dev replay selection commitment differs")
            dev_items = list(selection.items)

            async def candidate_runner(
                icp: Mapping[str, Any],
                run_context: Mapping[str, Any],
            ):
                try:
                    return await asyncio.to_thread(
                        self._model_sandbox.execute_dev_replay,
                        artifact_doc=artifact.to_dict(),
                        source_bundle=source_bundle,
                        snapshot_root=snapshot_root,
                        module_name=str(payload["module_name"]),
                        callable_name=str(payload["callable_name"]),
                        icp=icp,
                        context=run_context,
                        environment=dict(environment),
                        credential_env_names=list(credential_env_names),
                        miss_policy=str(payload["miss_policy"]),
                        timeout_seconds=per_icp_timeout,
                        job_id=context.job_id,
                    )
                except ModelSandboxV2Error as exc:
                    raise DevEvalRunnerError(str(exc)) from exc

            result = await asyncio.wait_for(
                evaluate_dev(
                    candidate_runner=candidate_runner,
                    dev_items=dev_items,
                    snapshot_store=snapshot_store,
                    run_label=run_label,
                    install_replay_seams=False,
                    require_manifest=True,
                    expected_icp_count=expected_dev_icp_count,
                ),
                timeout=total_timeout,
            )
        result_doc = {
            **result.to_dict(),
            "evaluation_mode": "replay",
            "overlay_hash": sha256_json({}),
            "cohort_hash": cohort_hash,
        }
        result_doc["score_commitment"] = sha256_json(
            {
                "schema_version": (
                    "research_lab.git_tree_dev_score_commitment.v1"
                ),
                "dev_score_version": str(
                    result_doc.get("dev_score_version") or ""
                ),
                "dev_set_hash": str(result_doc.get("dev_set_hash") or ""),
                "snapshot_manifest_hash": str(
                    result_doc.get("snapshot_manifest_hash") or ""
                ),
                "miss_policy": str(result_doc.get("miss_policy") or ""),
                "evaluation_mode": "replay",
                "overlay_hash": sha256_json({}),
                "cohort_hash": cohort_hash,
            }
        )
        return ExecutionResultV2(
            output=result_doc,
            artifact_hashes=(
                artifact.model_artifact_hash,
                artifact.manifest_hash,
                str(source_bundle["archive_sha256"]),
                str(snapshot_evidence["archive_sha256"]),
                snapshot_tree_hash,
                snapshot_manifest_hash,
                str(selection.manifest["selection_manifest_hash"]),
                cohort_hash,
                sha256_json(result_doc),
            ),
        )

    async def _execute_dev_hybrid(
        self,
        payload: Mapping[str, Any],
        context: ExecutionContextV2,
    ) -> ExecutionResultV2:
        """Score one candidate against a frozen, receipt-bound round overlay."""

        if self._model_sandbox is None:
            raise ValueError("measured model sandbox is unavailable")
        required = {
            "schema_version",
            "artifact",
            "source_bundle",
            "snapshot_bundle",
            "snapshot_tree_hash",
            "snapshot_manifest_hash",
            "dev_selection_request",
            "module_name",
            "callable_name",
            "environment",
            "credential_env_names",
            "run_label",
            "cohort_hash",
            "miss_policy",
            "per_icp_timeout_seconds",
            "total_timeout_seconds",
            "provider_evidence_caches",
            "overlay_hash",
        }
        if not isinstance(payload, Mapping) or set(payload) != required:
            raise ValueError("dev hybrid request fields are invalid")
        if payload.get("schema_version") != DEV_HYBRID_REQUEST_SCHEMA_VERSION:
            raise ValueError("dev hybrid request schema is invalid")
        artifact = PrivateModelArtifactManifest.from_mapping(payload["artifact"])
        source_bundle = dict(payload["source_bundle"])
        snapshot_bundle = dict(payload["snapshot_bundle"])
        snapshot_tree_hash = str(payload.get("snapshot_tree_hash") or "")
        snapshot_manifest_hash = str(payload.get("snapshot_manifest_hash") or "")
        selection_request = payload.get("dev_selection_request")
        if not isinstance(selection_request, Mapping) or set(selection_request) != {
            "selection_seed",
            "miner_direction",
            "selection_manifest_hash",
        }:
            raise ValueError("dev hybrid selection request is invalid")
        if snapshot_bundle.get("source_tree_hash") != snapshot_tree_hash:
            raise ValueError("dev hybrid snapshot bundle commitment differs")
        environment = payload.get("environment")
        credential_env_names = payload.get("credential_env_names")
        caches = payload.get("provider_evidence_caches")
        overlay_hash = str(payload.get("overlay_hash") or "")
        cohort_hash = str(payload.get("cohort_hash") or "")
        if (
            not isinstance(environment, Mapping)
            or not isinstance(credential_env_names, list)
            or not isinstance(caches, Mapping)
            or sha256_json(dict(caches)) != overlay_hash
            or not _HASH_RE.fullmatch(cohort_hash)
        ):
            raise ValueError("dev hybrid evidence fields are invalid")
        if dict(environment) != measured_dev_replay_environment(
            self._execution_config
        ):
            raise ValueError("dev hybrid environment differs from measured policy")
        if credential_env_names != list(
            measured_credential_environment_names(self._execution_config)
        ):
            raise ValueError(
                "dev hybrid credential environment differs from measured policy"
            )
        per_icp_timeout = int(payload["per_icp_timeout_seconds"])
        total_timeout = int(payload["total_timeout_seconds"])
        if per_icp_timeout < 10 or total_timeout < 30:
            raise ValueError("dev hybrid timeout is invalid")
        run_label = str(payload.get("run_label") or "")
        if len(run_label.encode("utf-8")) > 1024:
            raise ValueError("dev hybrid run label is too large")

        with tempfile.TemporaryDirectory(prefix="lp-dev-hybrid-v2-") as tmp:
            snapshot_root = Path(tmp) / "snapshot-set"
            snapshot_evidence = extract_source_bundle_v2(
                snapshot_bundle,
                destination=snapshot_root,
                expected_source_tree_hash=snapshot_tree_hash,
            )
            for path in sorted(snapshot_root.rglob("*"), reverse=True):
                if path.is_file():
                    path.chmod(0o444)
                elif path.is_dir():
                    path.chmod(0o555)
            snapshot_root.chmod(0o555)
            snapshot_store = ProviderSnapshotStore(
                str(snapshot_root),
                mode=MODE_REPLAY,
                miss_policy=str(payload["miss_policy"]),
            )
            manifest = snapshot_store.load_manifest()
            verification = snapshot_store.verify_manifest(manifest)
            if (
                manifest is None
                or not verification["passed"]
                or str(manifest.get("manifest_hash") or "")
                != snapshot_manifest_hash
            ):
                raise ValueError("dev hybrid snapshot manifest verification failed")
            bank_items = snapshot_store.load_dev_icp_items() or []
            expected_dev_icp_count = measured_git_tree_config(
                self._execution_config
            ).live_max_icps_per_node
            if not expected_dev_icp_count <= len(bank_items) <= 100:
                raise ValueError(
                    "dev hybrid bank size cannot satisfy measured Git-tree policy"
                )
            if total_timeout != measured_dev_eval_total_timeout_seconds(
                self._execution_config
            ):
                raise ValueError("dev hybrid total timeout differs from measured policy")
            if per_icp_timeout != measured_dev_eval_icp_timeout_seconds(
                self._execution_config,
                item_count=expected_dev_icp_count,
            ):
                raise ValueError("dev hybrid ICP timeout differs from measured policy")
            if str(payload["miss_policy"]) != measured_dev_snapshot_miss_policy(
                self._execution_config
            ):
                raise ValueError("dev hybrid miss policy differs from measured policy")
            expected_dev_set_hash = str(manifest.get("icp_set_hash") or "")
            if compute_dev_set_hash(bank_items) != expected_dev_set_hash:
                raise ValueError("dev hybrid ICP bank commitment differs")
            selection = select_snapshot_dev_icps(
                bank_items,
                snapshot_manifest=manifest,
                size=expected_dev_icp_count,
                seed=str(selection_request.get("selection_seed") or ""),
                miner_direction=str(
                    selection_request.get("miner_direction") or ""
                ),
            )
            if str(selection_request.get("selection_manifest_hash") or "") != str(
                selection.manifest.get("selection_manifest_hash") or ""
            ):
                raise ValueError("dev hybrid selection commitment differs")
            dev_items = list(selection.items)
            expected_refs = {
                icp_evidence_cache_key(
                    canonicalize_private_model_icp(dict(item.get("icp") or item))
                )
                for item in dev_items
            }
            if set(str(key) for key in caches) != expected_refs:
                raise ValueError("dev hybrid overlay does not cover the dev set")
            normalized_caches: Dict[str, Dict[str, Any]] = {}
            cache_hashes: list[str] = []
            for cache_ref in sorted(expected_refs):
                cache = caches.get(cache_ref)
                if (
                    not isinstance(cache, Mapping)
                    or cache.get("schema_version") != EVIDENCE_CACHE_SCHEMA_VERSION
                    or cache.get("icp_ref") != cache_ref
                    or not isinstance(cache.get("entries"), Mapping)
                ):
                    raise ValueError("dev hybrid provider evidence cache is invalid")
                normalized = dict(cache)
                cache_hash = sha256_json(normalized)
                expected_input_root = provider_evidence_tape_input_root(
                    cache_ref, cache_hash
                )
                disclosed_receipt_sets = [
                    graph.get("receipts") or ()
                    for graph in context.external_receipt_graphs
                ] + [
                    proof.get("disclosed_receipts") or ()
                    for proof in context.external_ancestry_proofs
                ]
                matches = {
                    sha256_json(dict(receipt)): dict(receipt)
                    for receipts in disclosed_receipt_sets
                    for receipt in receipts
                    if isinstance(receipt, Mapping)
                    and receipt.get("role") == "gateway_scoring"
                    and receipt.get("purpose")
                    == "research_lab.provider_evidence_tape.v2"
                    and receipt.get("status") == "succeeded"
                    and receipt.get("input_root") == expected_input_root
                    and receipt.get("output_root") == cache_hash
                }
                if len(matches) != 1:
                    raise ValueError(
                        "dev hybrid cache has no unique measured tape ancestry"
                    )
                normalized_caches[cache_ref] = normalized
                cache_hashes.append(cache_hash)

            async def candidate_runner(
                icp: Mapping[str, Any],
                run_context: Mapping[str, Any],
            ):
                canonical_icp = canonicalize_private_model_icp(icp)
                cache_ref = icp_evidence_cache_key(canonical_icp)
                try:
                    return await asyncio.to_thread(
                        self._model_sandbox.execute_dev_provider_replay,
                        artifact_doc=artifact.to_dict(),
                        source_bundle=source_bundle,
                        module_name=str(payload["module_name"]),
                        callable_name=str(payload["callable_name"]),
                        icp=canonical_icp,
                        context=run_context,
                        environment=dict(environment),
                        credential_env_names=list(credential_env_names),
                        provider_evidence_cache=normalized_caches[cache_ref],
                        snapshot_root=snapshot_root,
                        timeout_seconds=per_icp_timeout,
                        job_id=context.job_id,
                    )
                except ModelSandboxV2Error as exc:
                    raise DevEvalRunnerError(str(exc)) from exc

            result = await asyncio.wait_for(
                evaluate_dev(
                    candidate_runner=candidate_runner,
                    dev_items=dev_items,
                    snapshot_store=snapshot_store,
                    run_label=run_label,
                    install_replay_seams=False,
                    require_manifest=True,
                    expected_icp_count=expected_dev_icp_count,
                ),
                timeout=total_timeout,
            )
        result_doc = {
            **result.to_dict(),
            "evaluation_mode": "hybrid",
            "overlay_hash": overlay_hash,
            "cohort_hash": cohort_hash,
        }
        result_doc["score_commitment"] = sha256_json(
            {
                "schema_version": (
                    "research_lab.git_tree_dev_score_commitment.v1"
                ),
                "dev_score_version": str(result_doc.get("dev_score_version") or ""),
                "dev_set_hash": str(result_doc.get("dev_set_hash") or ""),
                "snapshot_manifest_hash": str(
                    result_doc.get("snapshot_manifest_hash") or ""
                ),
                "miss_policy": str(result_doc.get("miss_policy") or ""),
                "evaluation_mode": "hybrid",
                "overlay_hash": overlay_hash,
                "cohort_hash": cohort_hash,
            }
        )
        return ExecutionResultV2(
            output=result_doc,
            artifact_hashes=(
                artifact.model_artifact_hash,
                artifact.manifest_hash,
                str(source_bundle["archive_sha256"]),
                str(snapshot_evidence["archive_sha256"]),
                snapshot_tree_hash,
                snapshot_manifest_hash,
                str(selection.manifest["selection_manifest_hash"]),
                overlay_hash,
                cohort_hash,
                *tuple(cache_hashes),
                sha256_json(result_doc),
            ),
        )
