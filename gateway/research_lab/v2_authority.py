"""V2-only authority adapters around unchanged Research Lab calculations."""

from __future__ import annotations

import asyncio
import os
import re
import time
from typing import Any, Iterable, Mapping, Sequence

from gateway.research_lab.attested_coordinator_v2 import execute_coordinator_v2
from gateway.research_lab.attested_scoring_v2 import execute_scoring_v2
from gateway.research_lab.tee_protocol import legacy_v1_enabled
from gateway.tee.source_add_runtime_v2 import (
    build_source_add_probe_job_envelope_v2,
    build_source_add_probe_route_v2,
    build_source_add_runtime_catalog_v2,
    validate_source_add_runtime_catalog_v2,
)
from gateway.tee.coordinator_executor_v2 import (
    OP_ATTEST_LEGACY_FINALIZED_ALLOCATION_V2,
    OP_CLASSIFY_LEGACY_ALLOCATION_V2,
    OP_PROMOTION_GATE_DECISION,
    OP_PROMOTION_IMPROVEMENT,
    OP_RESEARCH_LAB_ALLOCATION,
)
from gateway.tee.scoring_executor import (
    OP_BUILD_BASELINE_SCORE_SUMMARY,
    OP_BUILD_SCORE_BUNDLE,
    OP_QUALIFICATION_COMPANY_SCORES,
)
from gateway.tee.scoring_executor_v2 import (
    OP_PROVIDER_PREFLIGHT_V2,
    OP_SOURCE_ADD_LEG2_JUDGE_V2,
    PROVIDER_PREFLIGHT_REQUEST_SCHEMA_VERSION,
    SOURCE_ADD_JUDGE_REQUEST_SCHEMA_VERSION,
    SOURCE_ADD_JUDGE_RESULT_SCHEMA_VERSION,
)
from gateway.tee.reward_executor_v2 import (
    OP_RESEARCH_LAB_REWARD_DECISION,
    reward_receipt_projection_v2,
)
from gateway.tee.coordinator_source_add_v2 import (
    OP_SOURCE_ADD_FUNCTIONAL_PROBE_V2,
    OP_SOURCE_ADD_PROVENANCE_V2,
    SOURCE_ADD_FUNCTIONAL_PROBE_REQUEST_SCHEMA_VERSION,
    SOURCE_ADD_FUNCTIONAL_PROBE_RESULT_SCHEMA_VERSION,
    SOURCE_ADD_PROVENANCE_REQUEST_SCHEMA_VERSION,
    SOURCE_ADD_PROVENANCE_RESULT_SCHEMA_VERSION,
)
from gateway.tee.coordinator_executor_v2 import (
    OP_SOURCE_ADD_CATALOG_SNAPSHOT_V2,
)
from leadpoet_canonical.attested_v2 import (
    canonical_json,
    sha256_json,
    validate_receipt_graph,
)


_HASH_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_PURPOSE_V2 = {
    "research_lab.candidate_score.v1": "research_lab.candidate_score.v2",
    "research_lab.baseline_score.v1": "research_lab.baseline_score.v2",
    "research_lab.benchmark.v1": "research_lab.benchmark.v2",
    "research_lab.rebenchmark.v1": "research_lab.rebenchmark.v2",
    "research_lab.confirmation_score.v1": "research_lab.confirmation_score.v2",
}


class ResearchLabV2AuthorityError(RuntimeError):
    """A protected result did not have complete V2 enclave authority."""


async def evaluate_source_add_provenance_v2(
    *,
    submission_id: str,
    source_name: str,
    source_kind: str,
    declared_base_domains: Sequence[str],
    source_metadata: Mapping[str, Any],
    epoch_id: int,
    timeout_seconds: int = 45,
    execute: Any = execute_coordinator_v2,
    persist_links: Any = None,
) -> tuple[Any, dict[str, Any]]:
    """Execute the unchanged SOURCE_ADD provenance rules in the coordinator."""

    from gateway.research_lab.source_add_provenance import (
        SourceAddProvenanceResult,
    )

    if legacy_v1_enabled():
        from gateway.research_lab.source_add_provenance import (
            evaluate_source_add_provenance,
        )

        provenance = await asyncio.to_thread(
            evaluate_source_add_provenance,
            source_name=source_name,
            source_kind=source_kind,
            declared_base_domains=declared_base_domains,
            source_metadata=source_metadata,
        )
        result = {
            "schema_version": SOURCE_ADD_PROVENANCE_RESULT_SCHEMA_VERSION,
            "submission_id": str(submission_id),
            "precheck_status": provenance.precheck_status,
            "reasons": list(provenance.reasons),
            "precheck_doc": provenance.to_record_doc(),
        }
        return provenance, {
            "status": "off",
            "protocol": "legacy_v1",
            "result": result,
            "receipt_graph": {},
        }

    outcome = await execute(
        operation=OP_SOURCE_ADD_PROVENANCE_V2,
        purpose="research_lab.source_add_provenance.v2",
        epoch_id=max(0, int(epoch_id)),
        sequence=0,
        payload={
            "schema_version": SOURCE_ADD_PROVENANCE_REQUEST_SCHEMA_VERSION,
            "submission_id": str(submission_id),
            "source_name": str(source_name),
            "source_kind": str(source_kind),
            "declared_base_domains": [str(item) for item in declared_base_domains],
            "source_metadata": dict(source_metadata),
            "timeout_seconds": int(timeout_seconds),
        },
    )
    result = outcome.get("result")
    required = {
        "schema_version",
        "submission_id",
        "precheck_status",
        "reasons",
        "precheck_doc",
    }
    if (
        not isinstance(result, Mapping)
        or set(result) != required
        or result.get("schema_version")
        != SOURCE_ADD_PROVENANCE_RESULT_SCHEMA_VERSION
        or result.get("submission_id") != str(submission_id)
        or not isinstance(result.get("reasons"), list)
        or any(not isinstance(item, str) for item in result["reasons"])
        or not isinstance(result.get("precheck_doc"), Mapping)
    ):
        raise ResearchLabV2AuthorityError(
            "SOURCE_ADD provenance result binding differs"
        )
    receipt = outcome.get("execution_receipt") or outcome.get("receipt")
    if (
        not isinstance(receipt, Mapping)
        or receipt.get("output_root") != sha256_json(dict(result))
    ):
        raise ResearchLabV2AuthorityError(
            "SOURCE_ADD provenance receipt output differs"
        )
    precheck_doc = dict(result["precheck_doc"])
    if (
        precheck_doc.get("precheck_status") != result["precheck_status"]
        or list(precheck_doc.get("reasons") or []) != list(result["reasons"])
    ):
        raise ResearchLabV2AuthorityError(
            "SOURCE_ADD provenance document projection differs"
        )
    link = await _persist_business_links(
        outcome,
        (
            {
                "artifact_kind": "source_add_provenance",
                "artifact_ref": str(submission_id),
                "artifact_hash": str(receipt["output_root"]),
            },
        ),
        persist_links=persist_links,
    )
    provenance = SourceAddProvenanceResult(
        precheck_status=str(result["precheck_status"]),
        reasons=tuple(str(item) for item in result["reasons"]),
        doc={
            key: value
            for key, value in precheck_doc.items()
            if key not in {"precheck_status", "reasons"}
        },
    )
    return provenance, {
        **dict(outcome),
        "status": "matched",
        "artifact_link_status": link,
    }


async def evaluate_source_add_functional_probe_v2(
    *,
    submission_id: str,
    config_ref: str,
    evaluation_mode: str,
    epoch_id: int,
    sequence: int,
    artifact_ref: str,
    timeout_seconds: int = 45,
    execute: Any = execute_coordinator_v2,
    load_probe_row: Any = None,
    persist_links: Any = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run one exact provisional API test through the V2 provider broker."""

    if legacy_v1_enabled():
        raise ResearchLabV2AuthorityError(
            "SOURCE_ADD functional probes require V2 coordinator authority"
        )
    if evaluation_mode not in {"functional_probe", "provisioning_smoke"}:
        raise ResearchLabV2AuthorityError(
            "SOURCE_ADD functional evaluation mode is invalid"
        )
    if load_probe_row is None:
        from gateway.research_lab.store import select_one

        async def load_probe_row(value: str) -> Mapping[str, Any] | None:
            config = await select_one(
                "research_lab_source_add_probe_config_current",
                filters=(
                    ("submission_id", value),
                    ("config_status", "active"),
                ),
            )
            submission = await select_one(
                "research_lab_source_add_submission_current",
                filters=(("submission_id", value),),
            )
            if not isinstance(config, Mapping) or not isinstance(
                submission, Mapping
            ):
                return None
            return {
                **dict(config),
                "miner_hotkey": str(submission.get("miner_hotkey") or ""),
            }

    row = await load_probe_row(str(submission_id))
    if not isinstance(row, Mapping) or str(row.get("config_ref") or "") != str(
        config_ref
    ):
        raise ResearchLabV2AuthorityError(
            "SOURCE_ADD current probe configuration is unavailable"
        )
    try:
        route = build_source_add_probe_route_v2(row)
    except Exception as exc:
        raise ResearchLabV2AuthorityError(
            "SOURCE_ADD probe route is invalid"
        ) from exc

    dynamic_refs = {}
    if route["credential_slot"]:
        dynamic_refs[str(route["credential_slot"])] = str(
            route["credential_value_hash"]
        )

    async def envelope_builder(job_id: str):
        envelope = build_source_add_probe_job_envelope_v2(row, job_id=job_id)
        return (envelope,) if envelope is not None else ()

    outcome = await execute(
        operation=OP_SOURCE_ADD_FUNCTIONAL_PROBE_V2,
        purpose="research_lab.source_add_functional_probe.v2",
        epoch_id=max(0, int(epoch_id)),
        sequence=max(0, int(sequence)),
        payload={
            "schema_version": SOURCE_ADD_FUNCTIONAL_PROBE_REQUEST_SCHEMA_VERSION,
            "submission_id": str(submission_id),
            "config_ref": str(config_ref),
            "evaluation_mode": str(evaluation_mode),
            "timeout_seconds": int(timeout_seconds),
        },
        input_artifact_hashes=(str(route["route_hash"]),),
        provider_credential_ref_hashes=dynamic_refs,
        additional_job_credential_envelope_builder=envelope_builder,
        timeout_seconds=max(60.0, float(timeout_seconds) * 3.0 + 30.0),
    )
    result = outcome.get("result")
    required = {
        "schema_version",
        "evaluator_version",
        "submission_id",
        "adapter_id",
        "config_ref",
        "evaluation_mode",
        "result_status",
        "route_hash",
        "selected_probe_index",
        "response_hash",
        "status_class",
        "content_type",
        "byte_count",
        "duration_ms",
        "retry_after_seconds",
        "reason_codes",
        "probe_summaries",
    }
    if (
        not isinstance(result, Mapping)
        or set(result) != required
        or result.get("schema_version")
        != SOURCE_ADD_FUNCTIONAL_PROBE_RESULT_SCHEMA_VERSION
        or result.get("submission_id") != str(submission_id)
        or result.get("config_ref") != str(config_ref)
        or result.get("evaluation_mode") != str(evaluation_mode)
        or result.get("route_hash") != route["route_hash"]
        or result.get("result_status")
        not in {"passed", "retryable", "awaiting_operator", "manual_review", "failed"}
        or not isinstance(result.get("reason_codes"), list)
        or not isinstance(result.get("probe_summaries"), list)
        or not isinstance(result.get("retry_after_seconds"), int)
        or not 0 <= int(result["retry_after_seconds"]) <= 21_600
        or not 1 <= len(result["probe_summaries"]) <= 3
    ):
        raise ResearchLabV2AuthorityError(
            "SOURCE_ADD functional probe result binding differs"
        )
    receipt = outcome.get("execution_receipt") or outcome.get("receipt")
    graph = outcome.get("receipt_graph")
    if (
        not isinstance(receipt, Mapping)
        or not isinstance(graph, Mapping)
        or receipt.get("output_root") != sha256_json(dict(result))
        or graph.get("root_receipt_hash") != receipt.get("receipt_hash")
    ):
        raise ResearchLabV2AuthorityError(
            "SOURCE_ADD functional probe receipt differs"
        )
    validate_receipt_graph(
        graph,
        required_purposes={"research_lab.source_add_functional_probe.v2"},
    )
    link = await _persist_business_links(
        outcome,
        (
            {
                "artifact_kind": (
                    "source_add_functional_probe"
                    if evaluation_mode == "functional_probe"
                    else "source_add_provisioning_smoke"
                ),
                "artifact_ref": str(artifact_ref),
                "artifact_hash": str(receipt["output_root"]),
            },
        ),
        persist_links=persist_links,
    )
    return dict(result), {
        **dict(outcome),
        "status": "matched",
        "artifact_link_status": link,
    }


async def authorize_reward_decision_v2(
    *,
    epoch_id: int,
    sequence: int = 0,
    decision_kind: str,
    decision_payload: Mapping[str, Any],
    expected_result: Mapping[str, Any] | None,
    artifact_kind: str,
    artifact_ref: str,
    parent_graphs: Sequence[Mapping[str, Any]] = (),
    execute: Any = execute_coordinator_v2,
    persist_links: Any = None,
) -> dict[str, Any]:
    """Require the existing reward kernel to produce one exact signed decision."""

    if legacy_v1_enabled() and decision_kind not in {
        "champion_migration",
        "source_add_migration",
    }:
        if not isinstance(expected_result, Mapping):
            raise ResearchLabV2AuthorityError(
                "legacy reward decisions without a host result must use the legacy kernel"
            )
        return {
            "status": "off",
            "protocol": "legacy_v1",
            "result": dict(expected_result),
            "artifact_link_status": {"status": "off"},
        }

    allowed_failed = set()
    for graph in parent_graphs:
        root_hash = str(graph.get("root_receipt_hash") or "")
        root = next(
            (
                item
                for item in graph.get("receipts") or ()
                if isinstance(item, Mapping)
                and item.get("receipt_hash") == root_hash
            ),
            None,
        )
        graph_allowed = set()
        if isinstance(root, Mapping) and root.get("status") == "failed":
            terminal_result = decision_payload.get("autoresearch_result")
            if (
                decision_kind != "reimbursement"
                or not isinstance(terminal_result, Mapping)
                or terminal_result.get("status") != "failed"
                or root.get("purpose") != "research_lab.candidate_decision.v2"
                or root.get("output_root") != sha256_json(dict(terminal_result))
            ):
                raise ResearchLabV2AuthorityError(
                    "failed reward ancestry is not an exact reimbursement terminal"
                )
            graph_allowed.add(root_hash)
            allowed_failed.add(root_hash)
        validate_receipt_graph(
            graph,
            allowed_failed_receipt_hashes=graph_allowed,
        )
    execute_kwargs = {
        "operation": OP_RESEARCH_LAB_REWARD_DECISION,
        "purpose": "research_lab.reward_decision.v2",
        "epoch_id": int(epoch_id),
        "sequence": max(0, int(sequence)),
        "payload": {
            "decision_kind": str(decision_kind),
            "decision_payload": dict(decision_payload),
        },
        "parent_graphs": tuple(parent_graphs),
        "input_artifact_hashes": (
            sha256_json(
                dict(expected_result)
                if isinstance(expected_result, Mapping)
                else dict(decision_payload)
            ),
        ),
    }
    if allowed_failed:
        execute_kwargs["allowed_failed_parent_receipt_hashes"] = tuple(
            sorted(allowed_failed)
        )
    outcome = await execute(
        **execute_kwargs,
    )
    actual_result = outcome.get("result")
    if not isinstance(actual_result, Mapping):
        raise ResearchLabV2AuthorityError("reward decision result is missing")
    if isinstance(expected_result, Mapping):
        _assert_equal(actual_result, dict(expected_result), "reward decision")
    receipt = outcome.get("execution_receipt") or outcome.get("receipt")
    if not isinstance(receipt, Mapping):
        raise ResearchLabV2AuthorityError("reward decision receipt is missing")
    output_root = str(receipt.get("output_root") or "")
    expected_projection = reward_receipt_projection_v2(actual_result)
    if output_root != sha256_json(expected_projection):
        raise ResearchLabV2AuthorityError("reward decision output root differs")
    resolved_artifact_ref = str(artifact_ref)
    if not resolved_artifact_ref and decision_kind == "reimbursement":
        award = actual_result.get("award")
        if isinstance(award, Mapping):
            resolved_artifact_ref = str(award.get("award_id") or "")
    if not resolved_artifact_ref:
        raise ResearchLabV2AuthorityError("reward artifact reference is missing")
    link = await _persist_business_links(
        outcome,
        (
            {
                "artifact_kind": str(artifact_kind),
                "artifact_ref": resolved_artifact_ref,
                "artifact_hash": output_root,
            },
        ),
        persist_links=persist_links,
    )
    return {**dict(outcome), "status": "matched", "artifact_link_status": link}


async def attest_historical_champion_reward_v2(
    *,
    epoch_id: int,
    champion_reward_id: str,
    execute: Any = execute_coordinator_v2,
    persist_links: Any = None,
) -> dict[str, Any]:
    """Migrate one immutable pre-V2 champion row into V2 receipt authority."""

    reward_id = str(champion_reward_id or "")
    if not re.fullmatch(r"champion_reward:sha256:[0-9a-f]{64}", reward_id):
        raise ResearchLabV2AuthorityError("champion reward id is invalid")
    return await authorize_reward_decision_v2(
        epoch_id=int(epoch_id),
        sequence=1,
        decision_kind="champion_migration",
        decision_payload={"champion_reward_id": reward_id},
        expected_result=None,
        artifact_kind="champion_reward_decision",
        artifact_ref=reward_id,
        parent_graphs=(),
        execute=execute,
        persist_links=persist_links,
    )


async def attest_historical_source_add_reward_v2(
    *,
    epoch_id: int,
    reward_ref: str,
    execute: Any = execute_coordinator_v2,
    persist_links: Any = None,
) -> dict[str, Any]:
    """Migrate one measured pre-V2 provenance reward into V2 authority."""

    normalized_ref = str(reward_ref or "")
    if not re.fullmatch(r"source_add_reward:[0-9a-f]{16}", normalized_ref):
        raise ResearchLabV2AuthorityError("SOURCE_ADD reward ref is invalid")
    return await authorize_reward_decision_v2(
        epoch_id=int(epoch_id),
        sequence=1,
        decision_kind="source_add_migration",
        decision_payload={"reward_ref": normalized_ref},
        expected_result=None,
        artifact_kind="source_add_reward_decision",
        artifact_ref=normalized_ref,
        parent_graphs=(),
        execute=execute,
        persist_links=persist_links,
    )


async def attest_historical_champion_settlement_v2(
    *,
    epoch_id: int,
    netuid: int,
    settlement_epoch_id: int,
    execute: Any = execute_coordinator_v2,
    persist_links: Any = None,
    persist_migration: Any = None,
) -> dict[str, Any]:
    """Migrate one proven pre-V2 finalized allocation into V2 authority."""

    from leadpoet_canonical.legacy_settlement_v2 import (
        LEGACY_SETTLEMENT_REQUEST_SCHEMA_VERSION,
        validate_legacy_settlement_document_v2,
    )

    normalized_netuid = int(netuid)
    normalized_settlement_epoch = int(settlement_epoch_id)
    if normalized_netuid <= 0 or normalized_settlement_epoch < 0:
        raise ResearchLabV2AuthorityError(
            "champion settlement migration scope is invalid"
        )
    outcome = await execute(
        operation=OP_ATTEST_LEGACY_FINALIZED_ALLOCATION_V2,
        purpose="research_lab.legacy_finalized_allocation.v2",
        epoch_id=int(epoch_id),
        sequence=normalized_settlement_epoch,
        payload={
            "schema_version": LEGACY_SETTLEMENT_REQUEST_SCHEMA_VERSION,
            "netuid": normalized_netuid,
            "epoch_id": normalized_settlement_epoch,
        },
        parent_graphs=(),
        input_artifact_hashes=(),
    )
    result = outcome.get("result")
    if not isinstance(result, Mapping):
        raise ResearchLabV2AuthorityError(
            "champion settlement migration result is missing"
        )
    document = validate_legacy_settlement_document_v2(result)
    if (
        int(document["netuid"]) != normalized_netuid
        or int(document["epoch_id"]) != normalized_settlement_epoch
    ):
        raise ResearchLabV2AuthorityError(
            "champion settlement migration result scope differs"
        )
    receipt = outcome.get("execution_receipt") or outcome.get("receipt")
    if not isinstance(receipt, Mapping):
        raise ResearchLabV2AuthorityError(
            "champion settlement migration receipt is missing"
        )
    receipt_hash = str(receipt.get("receipt_hash") or "")
    if receipt.get("output_root") != sha256_json(document):
        raise ResearchLabV2AuthorityError(
            "champion settlement migration output root differs"
        )
    link = await _persist_business_links(
        outcome,
        (
            {
                "artifact_kind": "legacy_finalized_allocation",
                "artifact_ref": "%d:%d"
                % (normalized_netuid, normalized_settlement_epoch),
                "artifact_hash": str(document["settlement_hash"]),
            },
        ),
        persist_links=persist_links,
    )
    if persist_migration is None:
        from gateway.research_lab.attested_v2_store import (
            persist_legacy_finalized_allocation_migration_v2,
        )

        persist_migration = persist_legacy_finalized_allocation_migration_v2
    durable = await persist_migration(
        settlement=document,
        receipt_hash=receipt_hash,
    )
    return {
        **dict(outcome),
        "status": "matched",
        "artifact_link_status": link,
        "migration_status": durable,
    }


async def classify_historical_champion_allocation_v2(
    *,
    epoch_id: int,
    netuid: int,
    settlement_epoch_id: int,
    execute: Any = execute_coordinator_v2,
    persist_links: Any = None,
    persist_migration: Any = None,
    persist_nonfinalization: Any = None,
) -> dict[str, Any]:
    """Classify one signed pre-V2 allocation without inventing payment credit."""

    from leadpoet_canonical.legacy_settlement_v2 import (
        LEGACY_NONFINALIZATION_SCHEMA_VERSION,
        LEGACY_SETTLEMENT_REQUEST_SCHEMA_VERSION,
        LEGACY_SETTLEMENT_SCHEMA_VERSION,
        validate_legacy_nonfinalization_document_v2,
        validate_legacy_settlement_document_v2,
    )

    normalized_netuid = int(netuid)
    normalized_settlement_epoch = int(settlement_epoch_id)
    if normalized_netuid <= 0 or normalized_settlement_epoch < 0:
        raise ResearchLabV2AuthorityError(
            "champion allocation classification scope is invalid"
        )
    outcome = await execute(
        operation=OP_CLASSIFY_LEGACY_ALLOCATION_V2,
        purpose="research_lab.legacy_finalized_allocation.v2",
        epoch_id=int(epoch_id),
        sequence=normalized_settlement_epoch,
        payload={
            "schema_version": LEGACY_SETTLEMENT_REQUEST_SCHEMA_VERSION,
            "netuid": normalized_netuid,
            "epoch_id": normalized_settlement_epoch,
        },
        parent_graphs=(),
        input_artifact_hashes=(),
    )
    result = outcome.get("result")
    if not isinstance(result, Mapping):
        raise ResearchLabV2AuthorityError(
            "champion allocation classification result is missing"
        )
    schema_version = str(result.get("schema_version") or "")
    if schema_version == LEGACY_SETTLEMENT_SCHEMA_VERSION:
        document = validate_legacy_settlement_document_v2(result)
        artifact_kind = "legacy_finalized_allocation"
        artifact_hash = str(document["settlement_hash"])
        status = "finalized"
    elif schema_version == LEGACY_NONFINALIZATION_SCHEMA_VERSION:
        document = validate_legacy_nonfinalization_document_v2(result)
        artifact_kind = "legacy_allocation_nonfinalization"
        artifact_hash = str(document["finding_hash"])
        status = "not_finalized"
    else:
        raise ResearchLabV2AuthorityError(
            "champion allocation classification schema is invalid"
        )
    if (
        int(document["netuid"]) != normalized_netuid
        or int(document["epoch_id"]) != normalized_settlement_epoch
    ):
        raise ResearchLabV2AuthorityError(
            "champion allocation classification result scope differs"
        )
    receipt = outcome.get("execution_receipt") or outcome.get("receipt")
    if not isinstance(receipt, Mapping):
        raise ResearchLabV2AuthorityError(
            "champion allocation classification receipt is missing"
        )
    receipt_hash = str(receipt.get("receipt_hash") or "")
    if receipt.get("output_root") != sha256_json(document):
        raise ResearchLabV2AuthorityError(
            "champion allocation classification output root differs"
        )
    link = await _persist_business_links(
        outcome,
        (
            {
                "artifact_kind": artifact_kind,
                "artifact_ref": "%d:%d"
                % (normalized_netuid, normalized_settlement_epoch),
                "artifact_hash": artifact_hash,
            },
        ),
        persist_links=persist_links,
    )
    if status == "finalized":
        if persist_migration is None:
            from gateway.research_lab.attested_v2_store import (
                persist_legacy_finalized_allocation_migration_v2,
            )

            persist_migration = (
                persist_legacy_finalized_allocation_migration_v2
            )
        durable = await persist_migration(
            settlement=document,
            receipt_hash=receipt_hash,
        )
    else:
        if persist_nonfinalization is None:
            from gateway.research_lab.attested_v2_store import (
                persist_legacy_allocation_nonfinalization_v2,
            )

            persist_nonfinalization = (
                persist_legacy_allocation_nonfinalization_v2
            )
        durable = await persist_nonfinalization(
            finding=document,
            receipt_hash=receipt_hash,
        )
    return {
        **dict(outcome),
        "status": status,
        "artifact_link_status": link,
        "classification_status": durable,
    }


async def judge_source_add_implementation_v2(
    *,
    epoch_id: int,
    candidate: Mapping[str, Any],
    score_bundle: Mapping[str, Any],
    provisioned_sources: Sequence[Mapping[str, Any]],
    timeout_seconds: int = 180,
    execute: Any = execute_scoring_v2,
    load_business_graph: Any = None,
    load_catalog_snapshot: Any = None,
) -> tuple[Any, dict[str, Any]]:
    """Run the unchanged SOURCE_ADD Leg 2 judge as measured scoring authority."""

    from gateway.research_lab.source_add_llm_judge import SourceAddJudgeVerdict

    if legacy_v1_enabled():
        from gateway.research_lab.source_add_llm_judge import (
            judge_source_add_implementation,
            openrouter_key_for_source_add_judge,
        )

        verdict = await judge_source_add_implementation(
            api_key=openrouter_key_for_source_add_judge(),
            candidate=candidate,
            score_bundle=score_bundle,
            provisioned_sources=provisioned_sources,
            timeout_seconds=timeout_seconds,
        )
        return verdict, {
            "status": "off",
            "protocol": "legacy_v1",
            "result": {"verdict": verdict.to_doc()},
            "receipt_graph": {},
        }

    bundle_hash = str(score_bundle.get("score_bundle_hash") or "").lower()
    if not _HASH_RE.fullmatch(bundle_hash):
        raise ResearchLabV2AuthorityError("SOURCE_ADD judge score bundle hash is invalid")
    if load_business_graph is None:
        from gateway.research_lab.attested_v2_store import (
            load_business_artifact_graph_v2,
        )

        load_business_graph = load_business_artifact_graph_v2
    promotion_graph = await load_business_graph(
        artifact_kind="promotion_decision",
        artifact_ref="score_bundle:" + bundle_hash.split(":", 1)[1],
        artifact_hash=bundle_hash,
    )
    validate_receipt_graph(
        promotion_graph,
        required_purposes={"research_lab.promotion_decision.v2"},
    )
    if load_catalog_snapshot is None:
        load_catalog_snapshot = load_source_add_catalog_snapshot_v2
    catalog_outcome = await load_catalog_snapshot(epoch_id=int(epoch_id))
    catalog_result = catalog_outcome.get("result")
    catalog_graph = catalog_outcome.get("receipt_graph")
    if not isinstance(catalog_result, Mapping) or not isinstance(
        catalog_graph, Mapping
    ):
        raise ResearchLabV2AuthorityError(
            "SOURCE_ADD catalog snapshot authority is unavailable"
        )
    normalized_sources = [
        dict(item) for item in catalog_result.get("provisioned_sources") or ()
    ]
    _assert_equal(
        normalized_sources,
        [dict(item) for item in provisioned_sources],
        "SOURCE_ADD provisioned source snapshot",
    )
    outcome = await execute(
        operation=OP_SOURCE_ADD_LEG2_JUDGE_V2,
        purpose="research_lab.source_add_judge.v2",
        epoch_id=int(epoch_id),
        sequence=0,
        payload={
            "schema_version": SOURCE_ADD_JUDGE_REQUEST_SCHEMA_VERSION,
            "candidate": dict(candidate),
            "score_bundle": dict(score_bundle),
            "provisioned_sources": normalized_sources,
            "timeout_seconds": int(timeout_seconds),
        },
        worker_index=_worker_index(),
        parent_graphs=(promotion_graph, catalog_graph),
        input_artifact_hashes=(
            bundle_hash,
            sha256_json(dict(candidate)),
            sha256_json(normalized_sources),
        ),
        provider_credential_profile="source_add_judge",
    )
    result = outcome.get("result")
    if not isinstance(result, Mapping) or set(result) != {
        "schema_version",
        "candidate_id",
        "score_bundle_hash",
        "provisioned_sources_hash",
        "verdict",
    }:
        raise ResearchLabV2AuthorityError("SOURCE_ADD judge result is invalid")
    if (
        result.get("schema_version") != SOURCE_ADD_JUDGE_RESULT_SCHEMA_VERSION
        or result.get("candidate_id") != str(candidate.get("candidate_id") or "")
        or result.get("score_bundle_hash") != bundle_hash
        or result.get("provisioned_sources_hash") != sha256_json(normalized_sources)
    ):
        raise ResearchLabV2AuthorityError("SOURCE_ADD judge result binding differs")
    verdict_doc = result.get("verdict")
    if not isinstance(verdict_doc, Mapping) or set(verdict_doc) != {
        "verdict",
        "confidence",
        "source_used",
        "adapter_id",
        "registry_provider_id",
        "evidence_summary",
        "reason_codes",
        "model_id",
        "provider_usage",
        "judge_doc_hash",
    }:
        raise ResearchLabV2AuthorityError("SOURCE_ADD judge verdict is invalid")
    if not _HASH_RE.fullmatch(str(verdict_doc.get("judge_doc_hash") or "")):
        raise ResearchLabV2AuthorityError("SOURCE_ADD judge document hash is invalid")
    reasons = verdict_doc.get("reason_codes")
    usage = verdict_doc.get("provider_usage")
    if not isinstance(reasons, list) or not isinstance(usage, Mapping):
        raise ResearchLabV2AuthorityError("SOURCE_ADD judge metadata is invalid")
    verdict = SourceAddJudgeVerdict(
        verdict=str(verdict_doc.get("verdict") or ""),
        confidence=float(verdict_doc.get("confidence") or 0.0),
        source_used=bool(verdict_doc.get("source_used")),
        adapter_id=str(verdict_doc.get("adapter_id") or ""),
        registry_provider_id=str(verdict_doc.get("registry_provider_id") or ""),
        evidence_summary=str(verdict_doc.get("evidence_summary") or ""),
        reason_codes=tuple(str(item) for item in reasons),
        model_id=str(verdict_doc.get("model_id") or ""),
        provider_usage=dict(usage),
        raw_doc_hash=str(verdict_doc["judge_doc_hash"]),
    )
    if verdict.verdict not in {"helped", "not_helped", "uncertain"}:
        raise ResearchLabV2AuthorityError("SOURCE_ADD judge verdict value is invalid")
    graph = outcome.get("receipt_graph")
    receipt = outcome.get("receipt")
    if not isinstance(graph, Mapping) or not isinstance(receipt, Mapping):
        raise ResearchLabV2AuthorityError("SOURCE_ADD judge receipt graph is missing")
    validate_receipt_graph(graph, required_purposes={"research_lab.source_add_judge.v2"})
    if graph.get("root_receipt_hash") != receipt.get("receipt_hash"):
        raise ResearchLabV2AuthorityError("SOURCE_ADD judge graph root differs")
    return verdict, dict(outcome)


async def load_source_add_catalog_snapshot_v2(
    *,
    epoch_id: int,
    execute: Any = execute_coordinator_v2,
) -> dict[str, Any]:
    outcome = await execute(
        operation=OP_SOURCE_ADD_CATALOG_SNAPSHOT_V2,
        purpose="research_lab.source_add_catalog_snapshot.v2",
        epoch_id=int(epoch_id),
        sequence=0,
        payload={"limit": 200},
    )
    result = outcome.get("result")
    if not isinstance(result, Mapping):
        raise ResearchLabV2AuthorityError(
            "SOURCE_ADD catalog snapshot result is missing"
        )
    rows = result.get("provisioned_sources")
    private_rows = result.get("private_registry_rows")
    runtime_catalog = result.get("runtime_catalog")
    if (
        result.get("schema_version")
        != "leadpoet.source_add_catalog_snapshot.v2"
        or not isinstance(rows, list)
        or any(not isinstance(item, Mapping) for item in rows)
        or not isinstance(private_rows, list)
        or any(not isinstance(item, Mapping) for item in private_rows)
        or not isinstance(runtime_catalog, Mapping)
        or result.get("provisioned_sources_hash")
        != sha256_json([dict(item) for item in rows])
        or result.get("private_registry_rows_hash")
        != sha256_json([dict(item) for item in private_rows])
    ):
        raise ResearchLabV2AuthorityError(
            "SOURCE_ADD catalog snapshot result is invalid"
        )
    try:
        normalized_runtime_catalog = validate_source_add_runtime_catalog_v2(
            runtime_catalog
        )
        independently_derived_catalog = build_source_add_runtime_catalog_v2(
            [dict(item) for item in rows]
        )
    except Exception as exc:
        raise ResearchLabV2AuthorityError(
            "SOURCE_ADD runtime catalog is invalid"
        ) from exc
    if (
        normalized_runtime_catalog != independently_derived_catalog
        or result.get("runtime_catalog_hash")
        != normalized_runtime_catalog["catalog_hash"]
    ):
        raise ResearchLabV2AuthorityError(
            "SOURCE_ADD runtime catalog commitment differs"
        )
    receipt = outcome.get("receipt") or outcome.get("execution_receipt")
    graph = outcome.get("receipt_graph")
    if (
        not isinstance(receipt, Mapping)
        or not isinstance(graph, Mapping)
        or receipt.get("output_root") != sha256_json(dict(result))
        or graph.get("root_receipt_hash") != receipt.get("receipt_hash")
    ):
        raise ResearchLabV2AuthorityError(
            "SOURCE_ADD catalog snapshot receipt differs"
        )
    validate_receipt_graph(
        graph,
        required_purposes={"research_lab.source_add_catalog_snapshot.v2"},
    )
    return dict(outcome)


async def persist_source_add_judge_reward_link_v2(
    *,
    outcome: Mapping[str, Any],
    reward_ref: str,
    persist_links: Any = None,
) -> dict[str, Any]:
    if legacy_v1_enabled():
        return {"status": "off", "protocol": "legacy_v1"}
    receipt = outcome.get("execution_receipt") or outcome.get("receipt")
    result = outcome.get("result")
    if not isinstance(receipt, Mapping) or not isinstance(result, Mapping):
        raise ResearchLabV2AuthorityError("SOURCE_ADD judge authority is incomplete")
    output_root = str(receipt.get("output_root") or "").lower()
    if not _HASH_RE.fullmatch(output_root) or output_root != sha256_json(dict(result)):
        raise ResearchLabV2AuthorityError("SOURCE_ADD judge output commitment differs")
    return await _persist_business_links(
        outcome,
        (
            {
                "artifact_kind": "source_add_reward_judge",
                "artifact_ref": str(reward_ref),
                "artifact_hash": output_root,
            },
        ),
        persist_links=persist_links,
    )


async def execute_provider_preflight_v2(
    *,
    scope_key: str,
    worker_index: int,
    settings: Mapping[str, Any],
    force: bool = False,
    provider_credential_profile: str = "benchmark_model",
    execute: Any = execute_scoring_v2,
) -> dict[str, Any]:
    outcome = await execute(
        operation=OP_PROVIDER_PREFLIGHT_V2,
        purpose="research_lab.provider_preflight.v2",
        epoch_id=0,
        sequence=time.time_ns(),
        payload={
            "schema_version": PROVIDER_PREFLIGHT_REQUEST_SCHEMA_VERSION,
            "scope_key": str(scope_key),
            "force": bool(force),
            "settings": dict(settings),
        },
        worker_index=int(worker_index),
        provider_credential_profile=provider_credential_profile,
    )
    result = outcome.get("result")
    if not isinstance(result, Mapping):
        raise ResearchLabV2AuthorityError("provider preflight result is missing")
    return dict(result)


def _worker_index() -> int:
    try:
        value = int(os.getenv("RESEARCH_LAB_SCORING_WORKER_INDEX", "0") or 0)
    except ValueError as exc:
        raise ResearchLabV2AuthorityError("scoring worker index is invalid") from exc
    if value < 0 or value >= 25:
        raise ResearchLabV2AuthorityError("scoring worker index is outside 0-24")
    return value


def _v2_purpose(value: str) -> str:
    purpose = _PURPOSE_V2.get(str(value), str(value))
    if not purpose.endswith(".v2"):
        raise ResearchLabV2AuthorityError("V2 scoring purpose is invalid")
    return purpose


def _assert_equal(actual: Any, expected: Any, label: str) -> None:
    if canonical_json(actual) != canonical_json(expected):
        raise ResearchLabV2AuthorityError(
            "V2 enclave %s differs from the protected calculation" % label
        )


async def _graphs_for_roots(
    roots: Iterable[str],
    *,
    load_graph: Any = None,
) -> list[dict[str, Any]]:
    if load_graph is None:
        from gateway.research_lab.attested_v2_store import load_receipt_graph_v2

        load_graph = load_receipt_graph_v2
    graphs = []
    for root in sorted({str(item or "").lower() for item in roots if str(item or "")}):
        if not _HASH_RE.fullmatch(root):
            raise ResearchLabV2AuthorityError("V2 parent receipt hash is invalid")
        graph = await load_graph(root)
        validate_receipt_graph(graph)
        if graph.get("root_receipt_hash") != root:
            raise ResearchLabV2AuthorityError("V2 parent graph root differs")
        graphs.append(dict(graph))
    return graphs


async def _persist_business_links(
    outcome: Mapping[str, Any],
    links: Sequence[Mapping[str, Any]],
    *,
    persist_links: Any = None,
) -> dict[str, Any]:
    receipt = outcome.get("execution_receipt") or outcome.get("receipt")
    graph = outcome.get("receipt_graph")
    if not isinstance(receipt, Mapping) or not isinstance(graph, Mapping):
        raise ResearchLabV2AuthorityError("V2 authority receipt graph is missing")
    root = str(receipt.get("receipt_hash") or "")
    graph_receipts = {
        str(item.get("receipt_hash") or ""): item
        for item in graph.get("receipts", [])
        if isinstance(item, Mapping)
    }
    if root not in graph_receipts or graph_receipts[root] != dict(receipt):
        raise ResearchLabV2AuthorityError(
            "V2 authority execution receipt is absent from its graph"
        )
    if persist_links is None:
        from gateway.research_lab.attested_v2_store import (
            persist_business_artifact_links_v2,
        )

        persist_links = persist_business_artifact_links_v2
    return await persist_links(receipt_hash=root, artifacts=links)


async def execute_company_scores_v2(
    *,
    epoch_id: int,
    purpose: str,
    companies: Sequence[Mapping[str, Any]],
    icp: Mapping[str, Any],
    is_reference_model: bool,
    provider_credential_profile: str = "default",
    attestation_out: dict[str, Any] | None = None,
    execute: Any = execute_scoring_v2,
) -> list[dict[str, Any]]:
    outcome = await execute(
        operation=OP_QUALIFICATION_COMPANY_SCORES,
        purpose="research_lab.company_score.v2",
        epoch_id=int(epoch_id),
        sequence=0,
        payload={
            "companies": [dict(item) for item in companies],
            "icp": dict(icp),
            "is_reference_model": bool(is_reference_model),
            "provider_execution_mode": "live_enclave",
            "scoring_context_purpose": _v2_purpose(purpose),
        },
        worker_index=_worker_index(),
        provider_credential_profile=provider_credential_profile,
    )
    result = outcome.get("result")
    if not isinstance(result, Mapping):
        raise ResearchLabV2AuthorityError("V2 company score result is missing")
    breakdowns = result.get("breakdowns")
    scores = result.get("scores")
    if not isinstance(breakdowns, list) or not isinstance(scores, list):
        raise ResearchLabV2AuthorityError("V2 company score result is invalid")
    normalized = []
    for item in breakdowns:
        if not isinstance(item, Mapping):
            raise ResearchLabV2AuthorityError("V2 company score breakdown is invalid")
        normalized.append(dict(item))
    _assert_equal(
        scores,
        [float(item.get("final_score", 0.0) or 0.0) for item in normalized],
        "company score projection",
    )
    if attestation_out is not None:
        attestation_out.clear()
        attestation_out.update(outcome)
    return normalized


async def compare_score_bundle_v2(
    *,
    epoch_id: int,
    purpose: str,
    build_payload: Mapping[str, Any],
    expected_score_bundle: Mapping[str, Any],
    parent_receipt_hashes: Sequence[str],
    execute: Any = execute_scoring_v2,
    load_graph: Any = None,
    persist_links: Any = None,
) -> dict[str, Any]:
    bundle_hash = str(expected_score_bundle.get("score_bundle_hash") or "").lower()
    if not _HASH_RE.fullmatch(bundle_hash):
        raise ResearchLabV2AuthorityError("score bundle hash is invalid")
    graphs = await _graphs_for_roots(parent_receipt_hashes, load_graph=load_graph)
    outcome = await execute(
        operation=OP_BUILD_SCORE_BUNDLE,
        purpose=_v2_purpose(purpose),
        epoch_id=int(epoch_id),
        sequence=0,
        payload=dict(build_payload),
        worker_index=_worker_index(),
        parent_graphs=graphs,
        input_artifact_hashes=(bundle_hash,),
    )
    actual = (outcome.get("result") or {}).get("score_bundle")
    _assert_equal(actual, dict(expected_score_bundle), "score bundle")
    link = await _persist_business_links(
        outcome,
        (
            {
                "artifact_kind": "score_bundle",
                "artifact_ref": "score_bundle:" + bundle_hash.split(":", 1)[1],
                "artifact_hash": bundle_hash,
            },
        ),
        persist_links=persist_links,
    )
    return {**dict(outcome), "status": "matched", "artifact_link_status": link}


async def compare_baseline_summary_v2(
    *,
    epoch_id: int,
    build_payload: Mapping[str, Any],
    expected_result: Mapping[str, Any],
    parent_receipt_hashes: Sequence[str],
    execute: Any = execute_scoring_v2,
    load_graph: Any = None,
    persist_links: Any = None,
) -> dict[str, Any]:
    from leadpoet_canonical.attested_v2 import sha256_json

    summary = expected_result.get("score_summary_doc")
    if not isinstance(summary, Mapping):
        raise ResearchLabV2AuthorityError("baseline score summary is missing")
    summary_hash = sha256_json(dict(summary))
    graphs = await _graphs_for_roots(parent_receipt_hashes, load_graph=load_graph)
    outcome = await execute(
        operation=OP_BUILD_BASELINE_SCORE_SUMMARY,
        purpose="research_lab.rebenchmark.v2",
        epoch_id=int(epoch_id),
        sequence=0,
        payload=dict(build_payload),
        worker_index=_worker_index(),
        parent_graphs=graphs,
        input_artifact_hashes=(summary_hash,),
    )
    _assert_equal(outcome.get("result"), dict(expected_result), "baseline summary")
    artifact_ref = "private_baseline:%s:%s:%s" % (
        str(build_payload.get("benchmark_date") or ""),
        int(build_payload.get("benchmark_attempt") or 0),
        str(build_payload.get("rolling_window_hash") or "").removeprefix("sha256:")[:24],
    )
    link = await _persist_business_links(
        outcome,
        (
            {
                "artifact_kind": "benchmark_score_summary",
                "artifact_ref": artifact_ref,
                "artifact_hash": summary_hash,
            },
        ),
        persist_links=persist_links,
    )
    return {
        **dict(outcome),
        "status": "matched",
        "score_summary_hash": summary_hash,
        "artifact_ref": artifact_ref,
        "artifact_link_status": link,
    }


async def compare_promotion_metric_v2(
    *,
    epoch_id: int,
    score_bundle: Mapping[str, Any],
    expected_improvement_points: float,
    expected_event_doc: Mapping[str, Any],
    parent_receipt_hashes: Sequence[str] = (),
    execute: Any = execute_coordinator_v2,
    load_graph: Any = None,
    persist_links: Any = None,
) -> dict[str, Any]:
    bundle_hash = str(score_bundle.get("score_bundle_hash") or "").lower()
    if not _HASH_RE.fullmatch(bundle_hash):
        raise ResearchLabV2AuthorityError("promotion score bundle hash is invalid")
    bundle_id = "score_bundle:" + bundle_hash.split(":", 1)[1]
    roots = list(parent_receipt_hashes)
    if not roots:
        from gateway.research_lab.attested_v2_store import (
            load_business_artifact_graph_v2,
        )

        graph = await load_business_artifact_graph_v2(
            artifact_kind="score_bundle",
            artifact_ref=bundle_id,
            artifact_hash=bundle_hash,
        )
        graphs = [graph]
    else:
        graphs = await _graphs_for_roots(roots, load_graph=load_graph)
    outcome = await execute(
        operation=OP_PROMOTION_IMPROVEMENT,
        purpose="research_lab.ranking.v2",
        epoch_id=int(epoch_id),
        sequence=0,
        payload={"score_bundle": dict(score_bundle)},
        parent_graphs=graphs,
        input_artifact_hashes=(bundle_hash,),
    )
    expected = {
        "improvement_points": float(expected_improvement_points),
        "event_doc": dict(expected_event_doc),
    }
    _assert_equal(outcome.get("result"), expected, "promotion metric")
    link = await _persist_business_links(
        outcome,
        (
            {
                "artifact_kind": "promotion_metric",
                "artifact_ref": bundle_id,
                "artifact_hash": bundle_hash,
            },
        ),
        persist_links=persist_links,
    )
    return {**dict(outcome), "status": "matched", "artifact_link_status": link}


async def compare_promotion_decision_v2(
    *,
    epoch_id: int,
    score_bundle: Mapping[str, Any],
    decision_payload: Mapping[str, Any],
    expected_decision: Mapping[str, Any],
    metric_outcome: Mapping[str, Any],
    execute: Any = execute_coordinator_v2,
    persist_links: Any = None,
) -> dict[str, Any]:
    metric_graph = metric_outcome.get("receipt_graph")
    if not isinstance(metric_graph, Mapping):
        raise ResearchLabV2AuthorityError("promotion metric V2 graph is missing")
    validate_receipt_graph(metric_graph, required_purposes=("research_lab.ranking.v2",))
    bundle_hash = str(score_bundle.get("score_bundle_hash") or "").lower()
    if not _HASH_RE.fullmatch(bundle_hash):
        raise ResearchLabV2AuthorityError("promotion score bundle hash is invalid")
    outcome = await execute(
        operation=OP_PROMOTION_GATE_DECISION,
        purpose="research_lab.promotion_decision.v2",
        epoch_id=int(epoch_id),
        sequence=0,
        payload={"score_bundle": dict(score_bundle), **dict(decision_payload)},
        parent_graphs=(metric_graph,),
        input_artifact_hashes=(bundle_hash,),
    )
    _assert_equal(
        (outcome.get("result") or {}).get("decision"),
        dict(expected_decision),
        "promotion decision",
    )
    link = await _persist_business_links(
        outcome,
        (
            {
                "artifact_kind": "promotion_decision",
                "artifact_ref": "score_bundle:" + bundle_hash.split(":", 1)[1],
                "artifact_hash": bundle_hash,
            },
        ),
        persist_links=persist_links,
    )
    return {**dict(outcome), "status": "matched", "artifact_link_status": link}


async def build_allocation_v2(
    *,
    epoch_id: int,
    netuid: int,
    policy: Mapping[str, Any],
    execute: Any = execute_coordinator_v2,
    persist_links: Any = None,
    load_allocation_parent_graphs: Any = None,
) -> dict[str, Any]:
    using_default_parent_loader = load_allocation_parent_graphs is None
    if load_allocation_parent_graphs is None:
        load_allocation_parent_graphs = _load_allocation_parent_graphs_v2
    finalized_history: list[dict[str, Any]] = []
    readiness_business_graphs: dict[
        tuple[str, str], dict[str, Any]
    ] = {}
    if using_default_parent_loader:
        from gateway.research_lab.champion_settlement_v2 import (
            champion_v2_cutover_readiness,
        )

        readiness = await champion_v2_cutover_readiness(
            epoch=int(epoch_id),
            netuid=int(netuid),
            _finalized_history_out=finalized_history,
            _business_graphs_out=readiness_business_graphs,
        )
        if (
            readiness.get("ready") is not True
            or float(readiness.get("receipt_coverage") or 0.0) != 1.0
            or float(
                readiness.get("historical_classification_coverage")
                or readiness.get("historical_settlement_coverage")
                or 0.0
            )
            != 1.0
        ):
            raise ResearchLabV2AuthorityError(
                "champion V2 cutover blocked: %d obligations and %d "
                "historical allocations lack authoritative classifications"
                % (
                    len(readiness.get("missing") or ()),
                    len(
                        readiness.get(
                            "missing_historical_classifications"
                        )
                        or readiness.get("missing_historical_settlements")
                        or ()
                    ),
                )
            )
    parent_loader_kwargs = {
        "epoch_id": int(epoch_id),
        "netuid": int(netuid),
        "policy": dict(policy),
    }
    if using_default_parent_loader:
        parent_loader_kwargs.update(
            {
                "finalized_champion_history": finalized_history,
                "preloaded_business_graphs": readiness_business_graphs,
            }
        )
    graphs = list(await load_allocation_parent_graphs(**parent_loader_kwargs))
    bindings = []
    for graph in graphs:
        validate_receipt_graph(graph)
        root_hash = str(graph.get("root_receipt_hash") or "")
        receipts = {
            str(receipt["receipt_hash"]): receipt for receipt in graph["receipts"]
        }
        root = receipts.get(root_hash)
        if not isinstance(root, Mapping):
            raise ResearchLabV2AuthorityError("allocation parent graph root is missing")
        bindings.append(
            {
                "receipt_hash": root_hash,
                "receipt_purpose": str(root.get("purpose") or ""),
                "receipt_role": str(root.get("role") or ""),
            }
        )
    outcome = await execute(
        operation=OP_RESEARCH_LAB_ALLOCATION,
        purpose="research_lab.allocation.v2",
        epoch_id=int(epoch_id),
        sequence=0,
        payload={"epoch": int(epoch_id), "netuid": int(netuid)},
        parent_graphs=graphs,
    )
    authority_result = outcome.get("result")
    if not isinstance(authority_result, Mapping):
        raise ResearchLabV2AuthorityError("allocation authority result is missing")
    if set(authority_result) != {
        "allocation",
        "allocation_inputs",
        "source_state",
        "source_state_hash",
    }:
        raise ResearchLabV2AuthorityError("allocation authority fields are invalid")
    allocation = authority_result.get("allocation")
    allocation_inputs = authority_result.get("allocation_inputs")
    source_state = authority_result.get("source_state")
    if (
        not isinstance(allocation, Mapping)
        or not isinstance(allocation_inputs, Mapping)
        or not isinstance(source_state, Mapping)
    ):
        raise ResearchLabV2AuthorityError("allocation authority documents are invalid")
    allocation_hash = str(allocation.get("allocation_hash") or "").lower()
    if not _HASH_RE.fullmatch(allocation_hash):
        raise ResearchLabV2AuthorityError("allocation hash is invalid")
    if authority_result.get("source_state_hash") != sha256_json(dict(source_state)):
        raise ResearchLabV2AuthorityError("allocation source-state hash differs")
    expected_inputs = {
        "epoch": int(source_state.get("epoch", -1)),
        "policy": dict(source_state.get("policy") or {}),
        "active_reimbursement_obligations": list(
            source_state.get("reimbursement_obligations") or []
        ),
        "active_champion_obligations": list(
            source_state.get("champion_obligations") or []
        ),
    }
    if "source_add_obligations" in source_state:
        expected_inputs["active_source_add_obligations"] = list(
            source_state.get("source_add_obligations") or []
        )
    _assert_equal(allocation_inputs, expected_inputs, "allocation source projection")
    link = await _persist_business_links(
        outcome,
        (
            {
                "artifact_kind": "allocation",
                "artifact_ref": "epoch:%s" % int(epoch_id),
                "artifact_hash": allocation_hash,
            },
        ),
        persist_links=persist_links,
    )
    return {
        **dict(outcome),
        "status": "matched",
        "lineage_bindings": sorted(
            bindings,
            key=lambda item: (item["receipt_purpose"], item["receipt_hash"]),
        ),
        "lineage_complete": True,
        "missing_lineage_score_bundle_ids": [],
        "artifact_link_status": link,
    }


async def compare_allocation_v2(
    *,
    epoch_id: int,
    netuid: int,
    payload: Mapping[str, Any],
    expected_allocation: Mapping[str, Any],
    execute: Any = execute_coordinator_v2,
    persist_links: Any = None,
    load_allocation_parent_graphs: Any = None,
) -> dict[str, Any]:
    """Differential compatibility check around the authoritative V2 builder."""

    outcome = await build_allocation_v2(
        epoch_id=epoch_id,
        netuid=netuid,
        policy=dict(payload.get("policy") or {}),
        execute=execute,
        persist_links=persist_links,
        load_allocation_parent_graphs=load_allocation_parent_graphs,
    )
    result = outcome.get("result")
    if not isinstance(result, Mapping):
        raise ResearchLabV2AuthorityError("allocation authority result is missing")
    _assert_equal(result.get("allocation"), dict(expected_allocation), "allocation")
    _assert_equal(result.get("allocation_inputs"), dict(payload), "allocation inputs")
    return outcome


async def _load_allocation_parent_graphs_v2(
    *,
    epoch_id: int,
    netuid: int,
    policy: Mapping[str, Any],
    finalized_champion_history: Sequence[Mapping[str, Any]] | None = None,
    preloaded_business_graphs: Mapping[
        tuple[str, str], Mapping[str, Any]
    ] | None = None,
) -> list[dict[str, Any]]:
    """Load candidate parent graphs; the enclave independently checks completeness."""

    from gateway.research_lab.attested_v2_store import (
        load_business_artifact_graphs_by_ref_v2,
        load_receipt_graphs_v2,
    )
    from gateway.research_lab.allocations import (
        POSTGREST_IN_FILTER_CHUNK,
        _champion_obligation_caps,
        _champion_paid_alpha_to_date_from_snapshots,
        _champion_replay_obligation,
        _source_add_paid_alpha_to_date_from_snapshots,
    )
    from gateway.research_lab.champion_settlement_v2 import (
        load_finalized_allocation_history_v2,
    )
    from gateway.research_lab.store import select_all

    graphs: dict[str, dict[str, Any]] = {}
    artifact_refs: set[tuple[str, str]] = set()
    receipt_roots: set[str] = set()
    preloaded = dict(preloaded_business_graphs or {})

    def add(kind: str, ref: str) -> None:
        key = (str(kind or ""), str(ref or ""))
        graph = preloaded.get(key)
        if isinstance(graph, Mapping):
            normalized = dict(graph)
            graphs[str(normalized.get("root_receipt_hash") or "")] = normalized
            return
        artifact_refs.add(key)

    def add_receipt_root(receipt_hash: str) -> None:
        receipt_roots.add(str(receipt_hash or ""))

    try:
        epoch_span = max(1, int(policy.get("reimbursement_epochs") or 20))
    except (TypeError, ValueError):
        epoch_span = 20
    schedules = await select_all(
        "research_reimbursement_schedules",
        filters=(
            ("schedule_status", "scheduled"),
            ("start_epoch", "lte", int(epoch_id)),
            ("start_epoch", "gte", max(0, int(epoch_id) - epoch_span)),
        ),
        order_by=(("start_epoch", True),),
    )
    award_ids = sorted(
        {
            str(schedule.get("award_id") or "")
            for schedule in schedules
            if _allocation_epoch_active(schedule, int(epoch_id))
            and str(schedule.get("award_id") or "")
        }
    )
    awards_by_id: dict[str, dict[str, Any]] = {}
    for offset in range(0, len(award_ids), POSTGREST_IN_FILTER_CHUNK):
        chunk = award_ids[offset : offset + POSTGREST_IN_FILTER_CHUNK]
        award_rows = await select_all(
            "research_reimbursement_award_current",
            filters=(
                ("award_id", "in", chunk),
                ("current_award_status", "awarded"),
            ),
            max_rows=len(chunk) + 1,
            allow_partial=False,
        )
        for award in award_rows:
            award_id = str(award.get("award_id") or "")
            status = str(
                award.get("current_award_status")
                or award.get("award_status")
                or ""
            )
            if award_id not in chunk or status != "awarded":
                raise ResearchLabV2AuthorityError(
                    "allocation reimbursement award batch differs"
                )
            if award_id in awards_by_id:
                raise ResearchLabV2AuthorityError(
                    "allocation reimbursement award is ambiguous"
                )
            awards_by_id[award_id] = dict(award)
    for award_id in sorted(awards_by_id):
        add("reimbursement_decision", award_id)

    active_statuses = ("active", "queued", "partially_paid")
    champion_rows = []
    source_rows = []
    for status in active_statuses:
        champion_rows.extend(
            await select_all(
                "research_lab_champion_reward_current",
                filters=(
                    ("current_reward_status", status),
                    ("start_epoch", "lte", int(epoch_id)),
                ),
            )
        )
        source_rows.extend(
            await select_all(
                "research_lab_source_add_reward_current",
                filters=(
                    ("current_reward_status", status),
                    ("start_epoch", "lte", int(epoch_id)),
                ),
            )
        )
    history_starts = [
        int(row.get("start_epoch") or 0)
        for row in champion_rows + source_rows
        if int(row.get("start_epoch") or 0) <= int(epoch_id)
    ]
    normalized_finalized_history: list[dict[str, Any]] = []
    if history_starts and int(epoch_id) > 0:
        history_start = min(history_starts)
        if finalized_champion_history is None:
            normalized_finalized_history = (
                await load_finalized_allocation_history_v2(
                    netuid=int(netuid),
                    start_epoch=history_start,
                    end_epoch=int(epoch_id) - 1,
                )
            )
        else:
            normalized_finalized_history = [
                dict(row)
                for row in finalized_champion_history
                if history_start
                <= int(row.get("epoch") or -1)
                < int(epoch_id)
            ]
            if any(
                int(row.get("netuid") or -1) != int(netuid)
                for row in normalized_finalized_history
            ):
                raise ResearchLabV2AuthorityError(
                    "allocation finalized history netuid differs"
                )
    paid_by_reward = _champion_paid_alpha_to_date_from_snapshots(
        normalized_finalized_history,
        obligation_caps=_champion_obligation_caps(champion_rows),
    )
    for row in champion_rows:
        if _champion_replay_obligation(
            row,
            paid_by_reward=paid_by_reward,
            epoch=int(epoch_id),
        ) is None:
            continue
        add(
            "champion_reward_decision",
            str(row.get("champion_reward_id") or ""),
        )
    for row in normalized_finalized_history:
        authority_types = set(row.get("authority_types") or ())
        if "native_v2_finalization" in authority_types:
            add("allocation", "epoch:%d" % int(row.get("epoch") or 0))
            for receipt_hash in row.get("finalization_receipt_hashes") or ():
                add_receipt_root(str(receipt_hash))
        if "legacy_finalized_chain_migration_v2" in authority_types:
            add_receipt_root(
                str(row.get("legacy_settlement_receipt_hash") or "")
            )
    source_paid_by_reward = _source_add_paid_alpha_to_date_from_snapshots(
        normalized_finalized_history
    )
    for row in source_rows:
        reward_ref = str(row.get("reward_ref") or "")
        if _champion_replay_obligation(
            {
                "champion_reward_id": reward_ref,
                "start_epoch": int(row.get("start_epoch") or 0),
                "epoch_count": int(
                    row.get("epoch_count") or row.get("reward_epochs") or 0
                ),
                "desired_alpha_percent": float(
                    row.get("desired_alpha_percent")
                    or row.get("alpha_percent")
                    or 0.0
                ),
            },
            paid_by_reward=source_paid_by_reward,
            epoch=int(epoch_id),
        ) is None:
            continue
        add("source_add_reward_decision", reward_ref)

    loaded_business_graphs = await load_business_artifact_graphs_by_ref_v2(
        artifact_refs
    )
    for graph in loaded_business_graphs.values():
        graphs[str(graph["root_receipt_hash"])] = graph
    loaded_receipt_graphs = await load_receipt_graphs_v2(receipt_roots)
    for receipt_hash, graph in loaded_receipt_graphs.items():
        if str(graph.get("root_receipt_hash") or "") != receipt_hash:
            raise ResearchLabV2AuthorityError(
                "allocation finalized-chain graph root differs"
            )
        graphs[receipt_hash] = graph
    return [graphs[root] for root in sorted(graphs)]


def _allocation_epoch_active(row: Mapping[str, Any], epoch_id: int) -> bool:
    try:
        start_epoch = int(row.get("start_epoch") or 0)
        epoch_count = int(row.get("epoch_count") or 0)
    except (TypeError, ValueError) as exc:
        raise ResearchLabV2AuthorityError(
            "allocation schedule epoch fields are invalid"
        ) from exc
    return epoch_count > 0 and start_epoch <= epoch_id < start_epoch + epoch_count
