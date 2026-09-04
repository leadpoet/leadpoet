"""Measured scoring and SOURCE_ADD operations for gateway_scoring."""
from __future__ import annotations

import asyncio
import os
import re
import threading
from typing import Any, Callable, Dict, Mapping

from gateway.tee.execution_job_manager_v2 import (
    ExecutionContextV2,
    ExecutionResultV2,
)
from gateway.tee.provider_client_v2 import BrokeredProviderTransportV2
from gateway.tee.qualification_executor_v2 import (
    OP_QUALIFICATION_BATCH_V2,
    OP_QUALIFICATION_EMAIL_EVIDENCE_V2,
    OP_QUALIFICATION_EPOCH_V2,
    QualificationExecutorV2,
)
from gateway.tee.qualification_network_v2 import SecureQualificationNetworkV2
from gateway.tee.qualification_epoch_guard_v2 import QualificationEpochGuardV2
from leadpoet_canonical.attested_v2 import sha256_json
from gateway.research_lab.provider_preflight import ProviderPreflight
from gateway.research_lab.config import ResearchLabGatewayConfig
from gateway.tee.research_lab_runtime_config_v2 import (
    build_research_lab_execution_config,
    validate_research_lab_execution_config,
)


OP_PROVIDER_PREFLIGHT_V2 = "provider_preflight_v2"
OP_SOURCE_ADD_LEG2_JUDGE_V2 = "source_add_leg2_judge_v2"
PROVIDER_PREFLIGHT_REQUEST_SCHEMA_VERSION = "leadpoet.provider_preflight_request.v3"
SOURCE_ADD_JUDGE_REQUEST_SCHEMA_VERSION = "leadpoet.source_add_judge_request.v2"
SOURCE_ADD_JUDGE_RESULT_SCHEMA_VERSION = "leadpoet.source_add_judge_result.v2"
PROVIDER_CREDENTIAL_REFS_FIELD = "_v2_provider_credential_ref_hashes"
PROVIDER_CREDENTIAL_PROFILE_FIELD = "_v2_provider_credential_profile"

SCORING_OPERATIONS_V2 = {
    OP_QUALIFICATION_BATCH_V2: frozenset({"qualification.lead_decision.v2"}),
    OP_QUALIFICATION_EMAIL_EVIDENCE_V2: frozenset(
        {"qualification.email_evidence.v2"}
    ),
    OP_QUALIFICATION_EPOCH_V2: frozenset({"qualification.sourcing_epoch.v2"}),
    OP_PROVIDER_PREFLIGHT_V2: frozenset(
        {"research_lab.provider_preflight.v2"}
    ),
    OP_SOURCE_ADD_LEG2_JUDGE_V2: frozenset(
        {"research_lab.source_add_judge.v2"}
    ),
}


class ScoringExecutorV2:
    def __init__(
        self,
        *,
        provider_execute: Callable[[Mapping[str, Any]], Mapping[str, Any]],
        retry_policy_hashes: Mapping[str, str],
        qualification_executor: QualificationExecutorV2 | None = None,
        config_supplier: Callable[[], ResearchLabGatewayConfig] = (
            ResearchLabGatewayConfig
        ),
        execution_config: Mapping[str, Any] | None = None,
    ) -> None:
        self._provider_execute = provider_execute
        self._retry_policy_hashes = dict(retry_policy_hashes)
        self._transport = BrokeredProviderTransportV2(self._provider_execute)
        self._config = config_supplier()
        self._execution_config = validate_research_lab_execution_config(
            execution_config
            if execution_config is not None
            else build_research_lab_execution_config(config=self._config)
        )
        self._transport.install()
        try:
            self._qualification_executor = (
                qualification_executor or QualificationExecutorV2(
                    epoch_checker=QualificationEpochGuardV2(
                        self._transport,
                        epoch_authority={
                            "mode": self._execution_config["epoch_authority"]["mode"],
                            "cutover": self._execution_config["epoch_authority"]["cutover"],
                        },
                        netuid=self._execution_config["deployment"]["netuid"],
                    )
                )
            )
            self._qualification_network = SecureQualificationNetworkV2()
            self._qualification_network.install()
        except BaseException:
            self._transport.restore()
            raise
        self._preflight_lock = threading.Lock()
        self._preflight_by_scope: Dict[str, ProviderPreflight] = {}
        os.environ["EXA_API_KEY"] = "leadpoet-v2-brokered-credential"
        os.environ["SCRAPINGDOG_API_KEY"] = "leadpoet-v2-brokered-credential"
        os.environ["QUALIFICATION_SCRAPINGDOG_API_KEY"] = (
            "leadpoet-v2-brokered-credential"
        )
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
        if operation == OP_PROVIDER_PREFLIGHT_V2:
            allowed_profiles.add("provider_preflight")
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
        if operation == OP_PROVIDER_PREFLIGHT_V2:
            return await self._execute_provider_preflight(payload, context)
        if operation == OP_SOURCE_ADD_LEG2_JUDGE_V2:
            return await self._execute_source_add_judge(payload, context)
        if operation == OP_QUALIFICATION_EPOCH_V2:
            return self._qualification_executor.aggregate_epoch(payload, context)
        with self._transport.scope(
            job_id=context.job_id,
            purpose=context.purpose,
            logical_operation_id=context.job_id,
            retry_policy_hashes=self._retry_policy_hashes,
            terminal_sink=context.record_transport,
            artifact_sink=context.record_artifact,
            allow_transport_failures=False,
        ):
            if operation == OP_QUALIFICATION_EMAIL_EVIDENCE_V2:
                return await self._qualification_executor.execute_email_evidence(
                    payload, context
                )
            if operation == OP_QUALIFICATION_BATCH_V2:
                return await self._qualification_executor.execute_batch(payload, context)
        raise ValueError("unsupported V2 scoring operation")

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
