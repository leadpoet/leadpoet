"""Host adapters for the measured V2 CodeEditLoopEngine authority."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, fields
import json
from pathlib import Path
from typing import Any, Awaitable, Callable, Mapping, Sequence

from gateway.research_lab.attested_artifacts_v2 import (
    persist_execution_transport_artifacts_v2,
)
from gateway.research_lab.attested_autoresearch_v2 import (
    DEFAULT_RELEASE_MANIFEST_PATH,
    derive_autoresearch_job_id_v2,
    execute_autoresearch_v2,
)
from gateway.research_lab.attested_coordinator_v2 import (
    load_provider_outcome_snapshot_v2,
)
from gateway.research_lab.model_authority_v2 import source_bundle_for_artifact_v2
from gateway.research_lab.provider_profiles_v2 import (
    STALE_PARENT_REPAIR_PROFILE,
    load_provider_profile_v2,
    provision_provider_profile_v2,
)
from gateway.research_lab.v2_credential_envelopes import (
    load_openrouter_credential_commitments_v2,
    load_openrouter_job_credential_envelope_v2,
)
from gateway.research_lab.v2_authority import (
    load_source_add_catalog_snapshot_v2,
)
from gateway.tee.autoresearch_executor_v2 import (
    AUTORESEARCH_REQUEST_SCHEMA_VERSION,
    HOST_APPEND_EVENT,
    HOST_ARTIFACT_RESULT_SCHEMA_VERSION,
    HOST_BUILD_CANDIDATE,
    HOST_BUILD_RESULT_SCHEMA_VERSION,
    HOST_CHECK_PAUSE,
    HOST_DEV_EVALUATE,
    HOST_DEV_EVAL_RESULT_SCHEMA_VERSION,
    HOST_EVENT_RESULT_SCHEMA_VERSION,
    HOST_PAUSE_RESULT_SCHEMA_VERSION,
    HOST_READ_ARTIFACT,
    HOST_RECORD_PRIVACY,
    HOST_WRITE_ARTIFACT,
    OPENROUTER_GUARD_REQUEST_SCHEMA_VERSION,
    OPENROUTER_GUARD_RESULT_SCHEMA_VERSION,
    OP_RUN_CODE_EDIT_LOOP,
    OP_REPAIR_STALE_PARENT,
    OP_VERIFY_OPENROUTER_GUARD,
    STALE_PARENT_REPAIR_REQUEST_SCHEMA_VERSION,
    STALE_PARENT_REPAIR_RESULT_SCHEMA_VERSION,
    _result_document,
)
from gateway.research_lab.key_vault import strict_openrouter_provider_policy
from gateway.tee.release_manifest_v2 import validate_release_manifest
from gateway.tee.provider_outcome_v2 import (
    validate_provider_outcome_snapshot_v2,
)
from gateway.tee.source_add_runtime_v2 import (
    build_source_add_job_envelope_v2,
    source_add_runtime_credential_refs_v2,
    validate_source_add_runtime_catalog_v2,
)
from gateway.utils.tee_client import coordinator_tee_client
from gateway.utils.tee_kms_provision_v2 import (
    provision_job_credential_envelope_v2 as provision_job_provider_envelope_v2,
)
from leadpoet_canonical.attested_v2 import (
    canonical_json,
    sha256_bytes,
    sha256_json,
    validate_receipt_graph,
)
from research_lab.code_editing import CodeEditDraft
from research_lab.eval import (
    CandidatePatchManifest,
    PrivateModelArtifactManifest,
    validate_candidate_patch_manifest,
    validate_private_model_artifact_manifest,
)
from gateway.research_lab.code_build import CodeEditBuildResult
from gateway.research_lab.code_loop_engine import (
    BuiltCodeEditCandidate,
    CodeEditLoopResult,
)
from gateway.research_lab.loop_engine import AutoResearchLoopEvent, AutoResearchLoopSettings


class AutoresearchAuthorityV2Error(RuntimeError):
    """A host operation or enclave result failed an exact V2 binding."""


@dataclass(frozen=True)
class AuthoritativeAutoresearchV2Result:
    loop_result: CodeEditLoopResult
    authority: Mapping[str, Any]


@dataclass(frozen=True)
class OpenRouterGuardAuthorityV2:
    proof_doc: Mapping[str, Any]
    credit_depleted: bool
    credit_limit_remaining: Any
    credential_commitments: Mapping[str, str]
    authority: Mapping[str, Any]


@dataclass(frozen=True)
class AttestedStaleParentRebaseV2Result:
    draft: CodeEditDraft
    repair_used: bool
    authority: Mapping[str, Any]


def autoresearch_result_document_v2(
    result: CodeEditLoopResult,
) -> dict[str, Any]:
    """Rebuild the exact canonical output signed by the autoresearch enclave."""

    return dict(_result_document(result))


async def verify_openrouter_guard_v2(
    *,
    key_ref: str,
    miner_hotkey: str,
    epoch_id: int,
    worker_index: int = 0,
    require_egress_proxy: bool = False,
    stage: str = "autoresearch_v2_authority",
    release_manifest_path: Path = DEFAULT_RELEASE_MANIFEST_PATH,
    execute: Any = execute_autoresearch_v2,
    coordinator_client: Any = coordinator_tee_client,
) -> OpenRouterGuardAuthorityV2:
    release = _load_release(release_manifest_path)
    commitments = await load_openrouter_credential_commitments_v2(
        key_ref=str(key_ref)
    )
    miner_hotkey_hash = sha256_bytes(str(miner_hotkey).encode("utf-8"))
    if commitments["miner_hotkey_hash"] != miner_hotkey_hash:
        raise AutoresearchAuthorityV2Error(
            "OpenRouter credential envelope belongs to another miner"
        )
    payload = {
        "schema_version": OPENROUTER_GUARD_REQUEST_SCHEMA_VERSION,
        "key_ref": str(key_ref),
        "key_ref_hash": commitments["key_ref_hash"],
        "miner_hotkey_hash": miner_hotkey_hash,
        "runtime_credential_value_hash": commitments[
            "runtime_credential_value_hash"
        ],
        "management_credential_value_hash": commitments[
            "management_credential_value_hash"
        ],
        "stage": str(stage),
        "request_policy": strict_openrouter_provider_policy(),
    }
    payload_hash = sha256_bytes(canonical_json(payload).encode("utf-8"))
    provider_profile = load_provider_profile_v2(
        "default",
        execution_role="gateway_autoresearch",
        worker_index=int(worker_index),
        require_egress_proxy=bool(require_egress_proxy),
    )
    provider_refs = dict(provider_profile["credential_ref_hashes"])
    provider_refs.update(
        {
            "openrouter": commitments["runtime_credential_value_hash"],
            "openrouter_management": commitments[
                "management_credential_value_hash"
            ],
        }
    )
    input_artifact_hashes = tuple(
        sorted(set(commitments.values()) | set(provider_refs.values()))
    )
    job_id = derive_autoresearch_job_id_v2(
        operation=OP_VERIFY_OPENROUTER_GUARD,
        purpose="research_lab.openrouter_guard.v2",
        epoch_id=int(epoch_id),
        sequence=0,
        payload_sha256=payload_hash,
        parent_receipt_hashes=(),
        input_artifact_hashes=input_artifact_hashes,
        release_hash=str(release["release_hash"]),
    )
    try:
        if provider_refs:
            await provision_provider_profile_v2(
                provider_profile,
                job_id=job_id,
                client=coordinator_client,
            )
        for credential_kind in ("runtime", "management"):
            envelope = await load_openrouter_job_credential_envelope_v2(
                key_ref=str(key_ref),
                credential_kind=credential_kind,
                job_id=job_id,
            )
            await provision_job_provider_envelope_v2(
                envelope,
                client=coordinator_client,
            )
        outcome = await execute(
            operation=OP_VERIFY_OPENROUTER_GUARD,
            purpose="research_lab.openrouter_guard.v2",
            epoch_id=int(epoch_id),
            sequence=0,
            payload=payload,
            host_operation_handlers={},
            input_artifact_hashes=input_artifact_hashes,
            provider_credential_profile="default",
            provider_credential_ref_hashes=provider_refs,
            release_manifest=release,
            persist_transport_artifacts=persist_execution_transport_artifacts_v2,
        )
    finally:
        await coordinator_client.v2_release_job_credentials(job_id)
    result = outcome.get("result")
    if not isinstance(result, Mapping):
        raise AutoresearchAuthorityV2Error(
            "OpenRouter guard result is unavailable"
        )
    required = {
        "schema_version",
        "key_ref_hash",
        "miner_hotkey_hash",
        "runtime_credential_value_hash",
        "management_credential_value_hash",
        "preflight_status",
        "preflight_error_type",
        "credit_depleted",
        "credit_limit_remaining",
        "privacy_proof_doc",
    }
    if (
        set(result) != required
        or result.get("schema_version") != OPENROUTER_GUARD_RESULT_SCHEMA_VERSION
        or any(
            result.get(name) != commitments[name]
            for name in (
                "key_ref_hash",
                "runtime_credential_value_hash",
                "management_credential_value_hash",
            )
        )
        or result.get("miner_hotkey_hash") != miner_hotkey_hash
        or not isinstance(result.get("credit_depleted"), bool)
        or not isinstance(result.get("privacy_proof_doc"), Mapping)
    ):
        raise AutoresearchAuthorityV2Error(
            "OpenRouter guard result binding differs"
        )
    graph = outcome.get("receipt_graph")
    receipt = outcome.get("receipt")
    if (
        not isinstance(graph, Mapping)
        or not isinstance(receipt, Mapping)
        or graph.get("root_receipt_hash") != receipt.get("receipt_hash")
    ):
        raise AutoresearchAuthorityV2Error(
            "OpenRouter guard receipt lineage is unavailable"
        )
    return OpenRouterGuardAuthorityV2(
        proof_doc=dict(result["privacy_proof_doc"]),
        credit_depleted=bool(result["credit_depleted"]),
        credit_limit_remaining=result.get("credit_limit_remaining"),
        credential_commitments=dict(commitments),
        authority=dict(outcome),
    )


def _parse_s3_uri(uri: str) -> tuple[str, str]:
    if not str(uri).startswith("s3://"):
        raise AutoresearchAuthorityV2Error("artifact URI is not S3")
    bucket, separator, key = str(uri)[5:].partition("/")
    if not separator or not bucket or not key or ".." in key.split("/"):
        raise AutoresearchAuthorityV2Error("artifact S3 URI is invalid")
    return bucket, key


def _mapping(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise AutoresearchAuthorityV2Error("%s must be an object" % field)
    return dict(value)


async def attest_stale_parent_rebase_v2(
    *,
    candidate: Mapping[str, Any],
    original_draft: CodeEditDraft,
    active_artifact: PrivateModelArtifactManifest,
    candidate_receipt_graph: Mapping[str, Any],
    epoch_id: int,
    worker_index: int,
    require_egress_proxy: bool,
    source_bundle_timeout_seconds: int,
    release_manifest_path: Path = DEFAULT_RELEASE_MANIFEST_PATH,
    execute: Any = execute_autoresearch_v2,
    coordinator_client: Any = coordinator_tee_client,
) -> AttestedStaleParentRebaseV2Result:
    """Attest the unchanged stale-parent applicability/repair decision.

    The host still performs the compatibility Docker/ECR build.  It may use a
    draft only when this measured operation verifies the source tree, original
    candidate ancestry, patch applicability, and any required OpenRouter repair.
    """

    validate_receipt_graph(candidate_receipt_graph)
    candidate_receipt_hash = str(
        candidate_receipt_graph.get("root_receipt_hash") or ""
    )
    candidate_id = str(candidate.get("candidate_id") or "")
    run_id = str(candidate.get("run_id") or "")
    if not candidate_id or not run_id or not candidate_receipt_hash:
        raise AutoresearchAuthorityV2Error(
            "stale-parent candidate measured ancestry is incomplete"
        )
    original_source_diff_hash = sha256_json(
        {"unified_diff": original_draft.unified_diff}
    )
    declared_source_diff_hash = str(
        candidate.get("candidate_source_diff_hash") or ""
    )
    if (
        declared_source_diff_hash
        and declared_source_diff_hash != original_source_diff_hash
    ):
        raise AutoresearchAuthorityV2Error(
            "stale-parent candidate source diff differs from its draft"
        )
    source_bundle = await source_bundle_for_artifact_v2(
        active_artifact,
        timeout_seconds=max(120, int(source_bundle_timeout_seconds)),
    )
    payload = {
        "schema_version": STALE_PARENT_REPAIR_REQUEST_SCHEMA_VERSION,
        "run_id": run_id,
        "candidate_id": candidate_id,
        "active_artifact": active_artifact.to_dict(),
        "source_bundle": source_bundle,
        "original_draft": original_draft.to_dict(),
        "original_source_diff_hash": original_source_diff_hash,
    }
    release = _load_release(release_manifest_path)
    payload_hash = sha256_bytes(canonical_json(payload).encode("utf-8"))
    normalized_worker_index = int(worker_index) % 10
    provider_profile = load_provider_profile_v2(
        STALE_PARENT_REPAIR_PROFILE,
        execution_role="gateway_autoresearch",
        worker_index=normalized_worker_index,
        require_egress_proxy=bool(require_egress_proxy),
    )
    provider_refs = dict(provider_profile["credential_ref_hashes"])
    if "openrouter" not in provider_refs:
        raise AutoresearchAuthorityV2Error(
            "stale-parent repair OpenRouter envelope is unavailable"
        )
    input_artifact_hashes = sorted(
        {
            active_artifact.model_artifact_hash,
            active_artifact.manifest_hash,
            str(source_bundle["archive_sha256"]),
            original_source_diff_hash,
            candidate_receipt_hash,
            *provider_refs.values(),
        }
    )
    job_id = derive_autoresearch_job_id_v2(
        operation=OP_REPAIR_STALE_PARENT,
        purpose="research_lab.stale_parent_repair.v2",
        epoch_id=int(epoch_id),
        sequence=0,
        payload_sha256=payload_hash,
        parent_receipt_hashes=(candidate_receipt_hash,),
        input_artifact_hashes=input_artifact_hashes,
        release_hash=str(release["release_hash"]),
    )
    try:
        await provision_provider_profile_v2(
            provider_profile,
            job_id=job_id,
            client=coordinator_client,
        )
        outcome = await execute(
            operation=OP_REPAIR_STALE_PARENT,
            purpose="research_lab.stale_parent_repair.v2",
            epoch_id=int(epoch_id),
            sequence=0,
            payload=payload,
            host_operation_handlers={},
            parent_graphs=(candidate_receipt_graph,),
            input_artifact_hashes=input_artifact_hashes,
            provider_credential_profile=STALE_PARENT_REPAIR_PROFILE,
            provider_credential_ref_hashes=provider_refs,
            release_manifest=release,
            persist_transport_artifacts=persist_execution_transport_artifacts_v2,
        )
    finally:
        await coordinator_client.v2_release_job_credentials(job_id)

    result = outcome.get("result")
    receipt = outcome.get("receipt")
    graph = outcome.get("receipt_graph")
    required = {
        "schema_version",
        "run_id",
        "candidate_id",
        "draft",
        "repair_used",
        "original_source_diff_hash",
        "result_source_diff_hash",
        "active_artifact_hash",
        "source_bundle_hash",
    }
    if (
        not isinstance(result, Mapping)
        or set(result) != required
        or result.get("schema_version")
        != STALE_PARENT_REPAIR_RESULT_SCHEMA_VERSION
        or result.get("run_id") != run_id
        or result.get("candidate_id") != candidate_id
        or result.get("original_source_diff_hash")
        != original_source_diff_hash
        or result.get("active_artifact_hash")
        != active_artifact.model_artifact_hash
        or result.get("source_bundle_hash")
        != source_bundle["archive_sha256"]
        or not isinstance(result.get("repair_used"), bool)
        or not isinstance(receipt, Mapping)
        or not isinstance(graph, Mapping)
        or graph.get("root_receipt_hash") != receipt.get("receipt_hash")
    ):
        raise AutoresearchAuthorityV2Error(
            "stale-parent repair result binding differs"
        )
    draft = _draft(_mapping(result.get("draft"), "stale-parent draft"))
    result_source_diff_hash = sha256_json(
        {"unified_diff": draft.unified_diff}
    )
    if result_source_diff_hash != result.get("result_source_diff_hash"):
        raise AutoresearchAuthorityV2Error(
            "stale-parent repaired draft hash differs"
        )
    if not result["repair_used"] and result_source_diff_hash != original_source_diff_hash:
        raise AutoresearchAuthorityV2Error(
            "stale-parent direct rebase changed the original patch"
        )
    return AttestedStaleParentRebaseV2Result(
        draft=draft,
        repair_used=bool(result["repair_used"]),
        authority=dict(outcome),
    )


def _load_release(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AutoresearchAuthorityV2Error(
            "V2 release manifest is unavailable"
        ) from exc
    return validate_release_manifest(value)


def _draft(value: Mapping[str, Any]) -> CodeEditDraft:
    allowed = {item.name for item in fields(CodeEditDraft)}
    unknown = set(value) - allowed - {"unified_diff_hash"}
    if unknown:
        raise AutoresearchAuthorityV2Error("V2 code-edit draft fields are invalid")
    kwargs = {name: value[name] for name in allowed if name in value}
    kwargs["target_files"] = tuple(str(item) for item in value.get("target_files", ()))
    return CodeEditDraft(**kwargs)


def _build_result(value: Mapping[str, Any]) -> CodeEditBuildResult:
    if set(value) != {
        "candidate_model_manifest",
        "code_edit_manifest",
        "source_diff_hash",
        "build_doc",
    }:
        raise AutoresearchAuthorityV2Error("V2 build result fields are invalid")
    manifest = PrivateModelArtifactManifest.from_mapping(
        value["candidate_model_manifest"]
    )
    manifest_errors = validate_private_model_artifact_manifest(manifest)
    if manifest_errors:
        raise AutoresearchAuthorityV2Error(
            "V2 candidate manifest is invalid: " + "; ".join(manifest_errors)
        )
    patch = CandidatePatchManifest.from_mapping(value["code_edit_manifest"])
    patch_errors = validate_candidate_patch_manifest(patch)
    if patch_errors:
        raise AutoresearchAuthorityV2Error(
            "V2 candidate patch is invalid: " + "; ".join(patch_errors)
        )
    return CodeEditBuildResult(
        candidate_model_manifest=manifest,
        code_edit_manifest=dict(value["code_edit_manifest"]),
        source_diff_hash=str(value["source_diff_hash"]),
        build_doc=dict(value["build_doc"]),
    )


def _candidate(value: Mapping[str, Any]) -> BuiltCodeEditCandidate:
    required = {
        "draft",
        "build",
        "node_id",
        "iteration",
        "rehydration_artifact_uri",
        "rehydration_artifact_hash",
        "dev_score",
        "dev_score_version",
    }
    if set(value) != required:
        raise AutoresearchAuthorityV2Error("V2 selected candidate fields are invalid")
    return BuiltCodeEditCandidate(
        draft=_draft(value["draft"]),
        build=_build_result(value["build"]),
        node_id=str(value["node_id"]),
        iteration=int(value["iteration"]),
        rehydration_artifact_uri=str(value["rehydration_artifact_uri"] or ""),
        rehydration_artifact_hash=str(value["rehydration_artifact_hash"] or ""),
        dev_score=(
            float(value["dev_score"]) if value["dev_score"] is not None else None
        ),
        dev_score_version=str(value["dev_score_version"] or ""),
    )


def _loop_result(value: Mapping[str, Any]) -> CodeEditLoopResult:
    required = {
        "schema_version",
        "selected_candidates",
        "iterations_completed",
        "stop_reason",
        "elapsed_seconds",
        "estimated_cost_usd",
        "actual_openrouter_cost_usd",
        "actual_openrouter_cost_microusd",
        "openrouter_call_count",
        "provider_usage",
        "status",
        "checkpoint_doc",
    }
    if set(value) != required or value.get("schema_version") != "leadpoet.autoresearch_result.v2":
        raise AutoresearchAuthorityV2Error("V2 autoresearch result fields are invalid")
    selected = value["selected_candidates"]
    usage = value["provider_usage"]
    if not isinstance(selected, list) or not isinstance(usage, list):
        raise AutoresearchAuthorityV2Error("V2 autoresearch result arrays are invalid")
    checkpoint = value.get("checkpoint_doc")
    if checkpoint is not None and not isinstance(checkpoint, Mapping):
        raise AutoresearchAuthorityV2Error("V2 checkpoint is invalid")
    return CodeEditLoopResult(
        selected_candidates=tuple(_candidate(item) for item in selected),
        iterations_completed=int(value["iterations_completed"]),
        stop_reason=str(value["stop_reason"]),
        elapsed_seconds=float(value["elapsed_seconds"]),
        estimated_cost_usd=float(value["estimated_cost_usd"]),
        actual_openrouter_cost_usd=float(value["actual_openrouter_cost_usd"]),
        actual_openrouter_cost_microusd=int(value["actual_openrouter_cost_microusd"]),
        openrouter_call_count=int(value["openrouter_call_count"]),
        provider_usage=tuple(dict(item) for item in usage),
        status=str(value["status"]),
        checkpoint_doc=(dict(checkpoint) if checkpoint is not None else None),
    )


async def _maybe_await(value: Any) -> Any:
    if asyncio.iscoroutine(value) or isinstance(value, Awaitable):
        return await value
    return value


async def run_authoritative_autoresearch_v2(
    *,
    run_id: str,
    ticket: Mapping[str, Any],
    artifact: PrivateModelArtifactManifest,
    component_registry: Mapping[str, Any],
    benchmark_public_summary: Mapping[str, Any],
    model_id: str,
    model_doc: Mapping[str, Any],
    budget_context: Mapping[str, Any],
    requested_loop_count: int,
    resume_state: Mapping[str, Any] | None,
    loop_settings: AutoResearchLoopSettings,
    probe_private_window_term_hashes: Sequence[str],
    openrouter_key_ref: str,
    miner_hotkey: str,
    openrouter_guard: OpenRouterGuardAuthorityV2,
    component_registry_authority: Mapping[str, Any],
    expected_event_state_hash: str,
    record_loop_event: Callable[[AutoResearchLoopEvent], Any],
    code_builder: Any,
    should_pause: Callable[[], Any],
    record_privacy_proof: Callable[..., Any],
    dev_evaluator: Any = None,
    epoch_id: int,
    worker_index: int = 0,
    require_egress_proxy: bool = False,
    release_manifest_path: Path = DEFAULT_RELEASE_MANIFEST_PATH,
    execute: Any = execute_autoresearch_v2,
    coordinator_client: Any = coordinator_tee_client,
    load_catalog_snapshot: Any = load_source_add_catalog_snapshot_v2,
    load_provider_outcome_snapshot: Any = load_provider_outcome_snapshot_v2,
) -> AuthoritativeAutoresearchV2Result:
    guard_graph = openrouter_guard.authority.get("receipt_graph")
    guard_receipt = openrouter_guard.authority.get("receipt")
    if (
        not isinstance(guard_graph, Mapping)
        or not isinstance(guard_receipt, Mapping)
        or guard_graph.get("root_receipt_hash") != guard_receipt.get("receipt_hash")
    ):
        raise AutoresearchAuthorityV2Error(
            "OpenRouter guard receipt lineage is unavailable"
        )
    privacy_receipt_hash = str(guard_receipt["receipt_hash"])
    commitments = dict(openrouter_guard.credential_commitments)
    component_graph = component_registry_authority.get("receipt_graph")
    component_receipt = component_registry_authority.get("receipt")
    component_result = component_registry_authority.get("result")
    if (
        not isinstance(component_graph, Mapping)
        or not isinstance(component_receipt, Mapping)
        or not isinstance(component_result, Mapping)
        or component_graph.get("root_receipt_hash")
        != component_receipt.get("receipt_hash")
    ):
        raise AutoresearchAuthorityV2Error(
            "component registry measured lineage is unavailable"
        )
    component_receipt_hash = str(component_receipt["receipt_hash"])
    catalog_outcome = await load_catalog_snapshot(epoch_id=int(epoch_id))
    catalog_result = catalog_outcome.get("result")
    catalog_graph = catalog_outcome.get("receipt_graph")
    catalog_receipt = catalog_outcome.get("receipt") or catalog_outcome.get(
        "execution_receipt"
    )
    if (
        not isinstance(catalog_result, Mapping)
        or not isinstance(catalog_graph, Mapping)
        or not isinstance(catalog_receipt, Mapping)
        or catalog_graph.get("root_receipt_hash")
        != catalog_receipt.get("receipt_hash")
    ):
        raise AutoresearchAuthorityV2Error(
            "SOURCE_ADD catalog measured lineage is unavailable"
        )
    catalog_receipt_hash = str(catalog_receipt["receipt_hash"])
    runtime_catalog = validate_source_add_runtime_catalog_v2(
        _mapping(catalog_result.get("runtime_catalog"), "runtime catalog")
    )
    provisioned_sources = catalog_result.get("provisioned_sources")
    if not isinstance(provisioned_sources, list) or any(
        not isinstance(item, Mapping) for item in provisioned_sources
    ):
        raise AutoresearchAuthorityV2Error(
            "SOURCE_ADD provisioned source snapshot is invalid"
        )
    provider_outcome = await load_provider_outcome_snapshot(
        epoch_id=int(epoch_id)
    )
    provider_outcome_result_raw = provider_outcome.get("result")
    provider_outcome_graph = provider_outcome.get("receipt_graph")
    provider_outcome_receipt = provider_outcome.get(
        "receipt"
    ) or provider_outcome.get("execution_receipt")
    try:
        provider_outcome_result = validate_provider_outcome_snapshot_v2(
            _mapping(
                provider_outcome_result_raw,
                "provider outcome snapshot result",
            )
        )
        validate_receipt_graph(
            _mapping(
                provider_outcome_graph,
                "provider outcome snapshot receipt graph",
            ),
            required_purposes=(
                "research_lab.provider_outcome_snapshot.v2",
            ),
        )
    except Exception as exc:
        raise AutoresearchAuthorityV2Error(
            "provider outcome measured lineage is invalid"
        ) from exc
    if not isinstance(provider_outcome_receipt, Mapping):
        raise AutoresearchAuthorityV2Error(
            "provider outcome measured lineage is unavailable"
        )
    provider_outcome_receipt_hash = str(
        provider_outcome_receipt.get("receipt_hash") or ""
    )
    if (
        provider_outcome_graph.get("root_receipt_hash")
        != provider_outcome_receipt_hash
        or provider_outcome_receipt.get("role") != "gateway_coordinator"
        or provider_outcome_receipt.get("purpose")
        != "research_lab.provider_outcome_snapshot.v2"
        or provider_outcome_receipt.get("status") != "succeeded"
        or provider_outcome_receipt.get("output_root")
        != sha256_json(provider_outcome_result)
    ):
        raise AutoresearchAuthorityV2Error(
            "provider outcome snapshot differs from its measured receipt"
        )
    source_bundle = await source_bundle_for_artifact_v2(
        artifact,
        timeout_seconds=max(120, int(getattr(code_builder.config, "code_edit_build_timeout_seconds", 900))),
    )
    payload = {
        "schema_version": AUTORESEARCH_REQUEST_SCHEMA_VERSION,
        "run_id": str(run_id),
        "ticket": dict(ticket),
        "artifact": artifact.to_dict(),
        "component_registry": dict(component_registry),
        "component_registry_evidence": {
            "result": dict(component_result),
            "receipt_graph": dict(component_graph),
            "root_receipt_hash": component_receipt_hash,
        },
        "provider_catalog_evidence": {
            "result": dict(catalog_result),
            "receipt_graph": dict(catalog_graph),
            "root_receipt_hash": catalog_receipt_hash,
        },
        "provider_outcome_evidence": {
            "result": provider_outcome_result,
            "receipt_graph": dict(provider_outcome_graph),
            "root_receipt_hash": provider_outcome_receipt_hash,
        },
        "benchmark_public_summary": dict(benchmark_public_summary),
        "model_id": str(model_id),
        "model_doc": dict(model_doc),
        "budget_context": dict(budget_context),
        "requested_loop_count": int(requested_loop_count),
        "resume_state": dict(resume_state or {}),
        "loop_settings": {
            item.name: getattr(loop_settings, item.name)
            for item in fields(AutoResearchLoopSettings)
        },
        "source_bundle": source_bundle,
        "probe_private_window_term_hashes": sorted(
            str(item) for item in probe_private_window_term_hashes
        ),
        "provider_outcome_digest": dict(
            provider_outcome_result["provider_outcome_digest"]
        ),
        "dev_evaluator_enabled": dev_evaluator is not None,
        "openrouter_context": {
            "key_ref": str(openrouter_key_ref),
            "miner_hotkey": str(miner_hotkey),
            "privacy_proof_doc": dict(openrouter_guard.proof_doc),
            "privacy_receipt_hash": privacy_receipt_hash,
            "runtime_credential_value_hash": commitments[
                "runtime_credential_value_hash"
            ],
            "management_credential_value_hash": commitments[
                "management_credential_value_hash"
            ],
        },
        "expected_event_state_hash": str(expected_event_state_hash),
    }
    release = _load_release(release_manifest_path)
    payload_hash = sha256_bytes(canonical_json(payload).encode("utf-8"))
    provider_profile = load_provider_profile_v2(
        "default",
        execution_role="gateway_autoresearch",
        worker_index=int(worker_index),
        require_egress_proxy=bool(require_egress_proxy),
    )
    provider_refs = dict(provider_profile["credential_ref_hashes"])
    provider_refs.update(
        {
            "openrouter": commitments["runtime_credential_value_hash"],
            "openrouter_management": commitments[
                "management_credential_value_hash"
            ],
        }
    )
    dynamic_provider_refs = source_add_runtime_credential_refs_v2(
        runtime_catalog
    )
    if set(provider_refs) & set(dynamic_provider_refs):
        raise AutoresearchAuthorityV2Error(
            "SOURCE_ADD provider collides with the measured provider profile"
        )
    provider_refs.update(dynamic_provider_refs)
    input_artifact_hashes = sorted(
        {
            artifact.model_artifact_hash,
            artifact.manifest_hash,
            str(source_bundle["archive_sha256"]),
            privacy_receipt_hash,
            component_receipt_hash,
            catalog_receipt_hash,
            provider_outcome_receipt_hash,
            str(provider_outcome_result["provider_outcome_digest_hash"]),
            str(provider_outcome_result["source_state_hash"]),
            str(catalog_result["provisioned_sources_hash"]),
            str(catalog_result["private_registry_rows_hash"]),
            str(catalog_result["runtime_catalog_hash"]),
            *commitments.values(),
            *provider_refs.values(),
        }
    )
    job_id = derive_autoresearch_job_id_v2(
        operation=OP_RUN_CODE_EDIT_LOOP,
        purpose="research_lab.candidate_decision.v2",
        epoch_id=int(epoch_id),
        sequence=0,
        payload_sha256=payload_hash,
        parent_receipt_hashes=(
            privacy_receipt_hash,
            component_receipt_hash,
            catalog_receipt_hash,
            provider_outcome_receipt_hash,
        ),
        input_artifact_hashes=input_artifact_hashes,
        release_hash=str(release["release_hash"]),
    )
    event_state = {"hash": str(expected_event_state_hash), "sequence": 0}

    async def append_event_handler(command: Mapping[str, Any], request: Mapping[str, Any]):
        event_doc = dict(command.get("event") or {})
        event_hash = str(command.get("event_hash") or "")
        sequence = int(command.get("event_sequence") or 0)
        if (
            sha256_json(event_doc) != event_hash
            or sequence != event_state["sequence"]
            or request.get("expected_state_hash") != event_state["hash"]
        ):
            raise AutoresearchAuthorityV2Error("autoresearch event state differs")
        event = AutoResearchLoopEvent(
            event_type=str(event_doc["event_type"]),
            loop_status=str(event_doc["loop_status"]),
            elapsed_seconds=float(event_doc["elapsed_seconds"]),
            node_id=event_doc.get("node_id"),
            candidate_artifact_hash=event_doc.get("candidate_artifact_hash"),
            candidate_patch_hash=event_doc.get("candidate_patch_hash"),
            provider_usage=[dict(item) for item in event_doc.get("provider_usage") or []],
            cost_ledger=dict(event_doc.get("cost_ledger") or {}),
            event_doc=dict(event_doc.get("event_doc") or {}),
        )
        row = await _maybe_await(record_loop_event(event))
        row = dict(row or {})
        next_hash = sha256_json(
            {
                "previous_state_hash": event_state["hash"],
                "event_hash": event_hash,
                "event_sequence": sequence,
                "persisted_event_hash": str(row.get("anchored_hash") or event_hash),
            }
        )
        event_state["hash"] = next_hash
        event_state["sequence"] = sequence + 1
        return {
            "schema_version": HOST_EVENT_RESULT_SCHEMA_VERSION,
            "event_hash": event_hash,
            "event_sequence": sequence,
            "next_state_hash": next_hash,
        }

    async def build_handler(command: Mapping[str, Any], request: Mapping[str, Any]):
        expected_state = sha256_json(
            {
                "run_id": str(command["run_id"]),
                "candidate_index": int(command["candidate_index"]),
                "parent_artifact_hash": str(command["parent_artifact"]["model_artifact_hash"]),
                "source_bundle_hash": str(command["source_bundle_hash"]),
                "source_diff_hash": str(command["source_diff_hash"]),
                "expected_candidate_artifact_hash": str(
                    command["expected_candidate_artifact_hash"]
                ),
            }
        )
        if request.get("expected_state_hash") != expected_state:
            raise AutoresearchAuthorityV2Error("candidate build state differs")
        result = await asyncio.to_thread(
            code_builder.build,
            draft=_draft(command["draft"]),
            parent_artifact=PrivateModelArtifactManifest.from_mapping(
                command["parent_artifact"]
            ),
            run_id=str(command["run_id"]),
            candidate_index=int(command["candidate_index"]),
        )
        if (
            result.candidate_model_manifest.model_artifact_hash
            != command["expected_candidate_artifact_hash"]
        ):
            raise AutoresearchAuthorityV2Error(
                "host candidate image differs from measured patched source"
            )
        return {
            "schema_version": HOST_BUILD_RESULT_SCHEMA_VERSION,
            "build_result": {
                "candidate_model_manifest": result.candidate_model_manifest.to_dict(),
                "code_edit_manifest": dict(result.code_edit_manifest),
                "source_diff_hash": str(result.source_diff_hash),
                "build_doc": dict(result.build_doc),
            },
        }

    async def read_artifact_handler(command: Mapping[str, Any], request: Mapping[str, Any]):
        content_hash = str(command["content_hash"])
        if request.get("expected_state_hash") != content_hash:
            raise AutoresearchAuthorityV2Error("artifact read state differs")
        bucket, key = _parse_s3_uri(str(command["uri"]))

        def read():
            import boto3

            body = boto3.client("s3").get_object(Bucket=bucket, Key=key)["Body"].read()
            return json.loads(body.decode("utf-8"))

        document = await asyncio.to_thread(read)
        return {
            "schema_version": HOST_ARTIFACT_RESULT_SCHEMA_VERSION,
            "uri": str(command["uri"]),
            "content_hash": content_hash,
            "document": dict(document),
        }

    async def write_artifact_handler(command: Mapping[str, Any], request: Mapping[str, Any]):
        content_hash = str(command["content_hash"])
        if request.get("expected_state_hash") != content_hash:
            raise AutoresearchAuthorityV2Error("artifact write state differs")
        bucket, key = _parse_s3_uri(str(command["uri"]))
        document = dict(command["document"])

        def write():
            import boto3

            boto3.client("s3").put_object(
                Bucket=bucket,
                Key=key,
                Body=json.dumps(document, sort_keys=True).encode("utf-8"),
                ContentType="application/json",
            )

        await asyncio.to_thread(write)
        return {
            "schema_version": HOST_ARTIFACT_RESULT_SCHEMA_VERSION,
            "uri": str(command["uri"]),
            "content_hash": content_hash,
            "persisted": True,
        }

    async def pause_handler(command: Mapping[str, Any], request: Mapping[str, Any]):
        check_sequence = int(command["check_sequence"])
        expected = sha256_json({"job_id": request["job_id"], "pause_check": check_sequence})
        if request.get("expected_state_hash") != expected:
            raise AutoresearchAuthorityV2Error("pause check state differs")
        paused = bool(await _maybe_await(should_pause()))
        return {
            "schema_version": HOST_PAUSE_RESULT_SCHEMA_VERSION,
            "paused": paused,
            "state_hash": sha256_json(
                {"check_sequence": check_sequence, "paused": paused}
            ),
        }

    async def privacy_handler(command: Mapping[str, Any], request: Mapping[str, Any]):
        proof_hash = str(command["proof_hash"])
        if request.get("expected_state_hash") != proof_hash:
            raise AutoresearchAuthorityV2Error("privacy proof state differs")
        await _maybe_await(record_privacy_proof(**dict(command)))
        return {
            "schema_version": HOST_ARTIFACT_RESULT_SCHEMA_VERSION,
            "proof_hash": proof_hash,
            "persisted": True,
        }

    handlers: dict[str, Callable[..., Any]] = {
        HOST_APPEND_EVENT: append_event_handler,
        HOST_BUILD_CANDIDATE: build_handler,
        HOST_READ_ARTIFACT: read_artifact_handler,
        HOST_WRITE_ARTIFACT: write_artifact_handler,
        HOST_CHECK_PAUSE: pause_handler,
        HOST_RECORD_PRIVACY: privacy_handler,
    }
    if dev_evaluator is not None:
        async def dev_handler(command: Mapping[str, Any], request: Mapping[str, Any]):
            candidate_hash = str(command["candidate_hash"])
            if request.get("expected_state_hash") != candidate_hash:
                raise AutoresearchAuthorityV2Error("dev-eval state differs")
            response = await _maybe_await(dev_evaluator(_candidate(command["candidate"])))
            if not isinstance(response, Mapping):
                raise AutoresearchAuthorityV2Error("attested dev-eval result is invalid")
            result = response.get("result")
            graph = response.get("receipt_graph")
            if not isinstance(result, Mapping) or not isinstance(graph, Mapping):
                raise AutoresearchAuthorityV2Error(
                    "dev-eval lacks measured scoring ancestry"
                )
            return {
                "schema_version": HOST_DEV_EVAL_RESULT_SCHEMA_VERSION,
                "candidate_hash": candidate_hash,
                "result": dict(result),
                "receipt_graph": dict(graph),
            }

        handlers[HOST_DEV_EVALUATE] = dev_handler

    try:
        if provider_refs:
            await provision_provider_profile_v2(
                provider_profile,
                job_id=job_id,
                client=coordinator_client,
            )
        for credential_kind in ("runtime", "management"):
            credential_envelope = await load_openrouter_job_credential_envelope_v2(
                key_ref=str(openrouter_key_ref),
                credential_kind=credential_kind,
                job_id=job_id,
            )
            await provision_job_provider_envelope_v2(
                credential_envelope,
                client=coordinator_client,
            )
        for source_row in provisioned_sources:
            dynamic_envelope = build_source_add_job_envelope_v2(
                source_row,
                job_id=job_id,
            )
            if dynamic_envelope is not None:
                await provision_job_provider_envelope_v2(
                    dynamic_envelope,
                    client=coordinator_client,
                )
        outcome = await execute(
            operation=OP_RUN_CODE_EDIT_LOOP,
            purpose="research_lab.candidate_decision.v2",
            epoch_id=int(epoch_id),
            sequence=0,
            payload=payload,
            host_operation_handlers=handlers,
            parent_graphs=(
                guard_graph,
                component_graph,
                catalog_graph,
                provider_outcome_graph,
            ),
            input_artifact_hashes=input_artifact_hashes,
            provider_credential_profile="default",
            provider_credential_ref_hashes=provider_refs,
            release_manifest=release,
            persist_transport_artifacts=persist_execution_transport_artifacts_v2,
        )
    finally:
        await coordinator_client.v2_release_job_credentials(job_id)
    result = outcome.get("result")
    if not isinstance(result, Mapping):
        raise AutoresearchAuthorityV2Error("V2 autoresearch result is missing")
    return AuthoritativeAutoresearchV2Result(
        loop_result=_loop_result(result),
        authority=dict(outcome),
    )
