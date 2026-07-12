"""V2 adapter around the unchanged Research Lab scoring implementation."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
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
    execute_scoring_operation,
)
from leadpoet_canonical.attested_v2 import canonical_json, sha256_bytes, sha256_json
from research_lab.eval import PrivateModelArtifactManifest
from research_lab.eval.dev_eval import compute_dev_set_hash, evaluate_dev
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
OP_PROVIDER_PREFLIGHT_V2 = "provider_preflight_v2"
OP_SOURCE_ADD_LEG2_JUDGE_V2 = "source_add_leg2_judge_v2"
DEV_REPLAY_REQUEST_SCHEMA_VERSION = "leadpoet.dev_replay_request.v2"
PROVIDER_PREFLIGHT_REQUEST_SCHEMA_VERSION = "leadpoet.provider_preflight_request.v2"
SOURCE_ADD_JUDGE_REQUEST_SCHEMA_VERSION = "leadpoet.source_add_judge_request.v2"
SOURCE_ADD_JUDGE_RESULT_SCHEMA_VERSION = "leadpoet.source_add_judge_result.v2"
PROVIDER_CREDENTIAL_REFS_FIELD = "_v2_provider_credential_ref_hashes"
PROVIDER_CREDENTIAL_PROFILE_FIELD = "_v2_provider_credential_profile"


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
        }
    ),
    OP_DEV_REPLAY_V2: frozenset({"research_lab.candidate_test.v2"}),
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
                environment={},
            )
        )
        self._transport.install()
        self._qualification_executor = qualification_executor or QualificationExecutorV2(
            epoch_checker=QualificationEpochGuardV2(self._transport)
        )
        self._qualification_network = SecureQualificationNetworkV2()
        self._qualification_network.install()
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
                    context.purpose != "research_lab.private_model_run.v2"
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
        required = {"schema_version", "scope_key", "force", "settings"}
        if not isinstance(payload, Mapping) or set(payload) != required:
            raise ValueError("provider preflight request fields are invalid")
        if payload.get("schema_version") != PROVIDER_PREFLIGHT_REQUEST_SCHEMA_VERSION:
            raise ValueError("provider preflight request schema is invalid")
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
            "module_name",
            "callable_name",
            "environment",
            "credential_env_names",
            "run_label",
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
            dev_items = snapshot_store.load_dev_icp_items()
            if not dev_items:
                raise ValueError("dev replay snapshot set carries no ICP items")
            if total_timeout != measured_dev_eval_total_timeout_seconds(
                self._execution_config
            ):
                raise ValueError("dev replay total timeout differs from measured policy")
            if per_icp_timeout != measured_dev_eval_icp_timeout_seconds(
                self._execution_config,
                item_count=len(dev_items),
            ):
                raise ValueError("dev replay ICP timeout differs from measured policy")
            if str(payload["miss_policy"]) != measured_dev_snapshot_miss_policy(
                self._execution_config
            ):
                raise ValueError("dev replay miss policy differs from measured policy")
            expected_dev_set_hash = str(manifest.get("icp_set_hash") or "")
            if compute_dev_set_hash(dev_items) != expected_dev_set_hash:
                raise ValueError("dev replay ICP set commitment differs")

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
                ),
                timeout=total_timeout,
            )
        result_doc = result.to_dict()
        return ExecutionResultV2(
            output=result_doc,
            artifact_hashes=(
                artifact.model_artifact_hash,
                artifact.manifest_hash,
                str(source_bundle["archive_sha256"]),
                str(snapshot_evidence["archive_sha256"]),
                snapshot_tree_hash,
                snapshot_manifest_hash,
                sha256_json(result_doc),
            ),
        )
