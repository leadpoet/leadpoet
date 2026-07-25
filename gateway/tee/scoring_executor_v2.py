"""V2 adapter around the unchanged Research Lab scoring implementation."""

from __future__ import annotations

import asyncio
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
from leadpoet_canonical.attested_v2 import canonical_json, sha256_bytes, sha256_json
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
OP_DEV_REPLAY_V2 = "run_dev_replay_v2"
OP_DEV_HYBRID_V2 = "run_dev_hybrid_v2"
OP_PROVIDER_PREFLIGHT_V2 = "provider_preflight_v2"
OP_SOURCE_ADD_LEG2_JUDGE_V2 = "source_add_leg2_judge_v2"
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
    ) -> None:
        self._provider_execute = provider_execute
        self._retry_policy_hashes = dict(retry_policy_hashes)
        self._transport = BrokeredProviderTransportV2(self._provider_execute)
        self._model_sandbox = model_sandbox
        self._artifact_seal = artifact_seal
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
                        epoch_authority=self._execution_config["epoch_authority"],
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
        if operation in {OP_RUN_MODEL_SANDBOX_V2, OP_PROVIDER_PREFLIGHT_V2}:
            allowed_profiles.add("benchmark_model")
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
        if operation == OP_RUN_MODEL_SANDBOX_V2:
            if self._model_sandbox is None:
                raise ValueError("measured model sandbox is unavailable")
            if self._artifact_seal is None:
                raise ValueError("measured model artifact sealer is unavailable")
            self._validate_model_provider_catalog_ancestry(payload, context)
            payload["environment"] = validate_model_sandbox_environment(
                self._execution_config,
                payload.get("environment"),
                provider_cost_scope=str(payload.get("provider_cost_scope") or ""),
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
                matching_tapes = [
                    receipt
                    for graph in context.external_receipt_graphs
                    for receipt in graph.get("receipts") or ()
                    if isinstance(receipt, Mapping)
                    and receipt.get("role") == "gateway_scoring"
                    and receipt.get("purpose")
                    == "research_lab.provider_evidence_tape.v2"
                    and receipt.get("status") == "succeeded"
                    and receipt.get("input_root") == expected_tape_input_root
                    and receipt.get("output_root") == cache_hash
                ]
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
            artifact_payloads = (
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
            if graph.get("root_receipt_hash") != root_hash:
                continue
            receipts = {
                str(item.get("receipt_hash") or ""): item
                for item in graph.get("receipts") or ()
                if isinstance(item, Mapping)
            }
            root = receipts.get(root_hash)
            if isinstance(root, Mapping):
                matching_roots.append(root)
        if (
            len(matching_roots) != 1
            or matching_roots[0].get("role") != "gateway_coordinator"
            or matching_roots[0].get("purpose")
            != "research_lab.source_add_catalog_snapshot.v2"
            or matching_roots[0].get("status") != "succeeded"
            or matching_roots[0].get("output_root")
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
        with self._transport.scope(
            job_id=context.job_id,
            purpose=context.purpose,
            logical_operation_id=context.job_id,
            retry_policy_hashes=self._retry_policy_hashes,
            terminal_sink=context.record_transport,
            artifact_sink=context.record_artifact,
            allow_transport_failures=True,
        ):
            result = await asyncio.to_thread(
                preflight.check,
                force=force,
                settings=dict(settings),
            )
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
                matches = [
                    receipt
                    for graph in context.external_receipt_graphs
                    for receipt in graph.get("receipts") or ()
                    if isinstance(receipt, Mapping)
                    and receipt.get("role") == "gateway_scoring"
                    and receipt.get("purpose")
                    == "research_lab.provider_evidence_tape.v2"
                    and receipt.get("status") == "succeeded"
                    and receipt.get("input_root") == expected_input_root
                    and receipt.get("output_root") == cache_hash
                ]
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
