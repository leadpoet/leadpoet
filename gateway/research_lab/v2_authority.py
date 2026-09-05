"""V2-only authority adapters around unchanged Research Lab calculations."""

from __future__ import annotations

import asyncio
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import logging
import os
import re
import time
import uuid
from typing import Any, Iterable, Iterator, Mapping, Sequence

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
    OP_RESEARCH_LAB_ALLOCATION,
)
from gateway.tee.coordinator_chain_realized_settlement_v1 import (
    CHAIN_REALIZED_SETTLEMENT_PURPOSE_V1,
    CHAIN_WEIGHT_OBSERVATION_PURPOSE_V1,
    OP_ATTEST_CHAIN_REALIZED_SETTLEMENT_V1,
    OP_OBSERVE_CHAIN_REALIZED_WEIGHTS_V1,
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
    source_add_reward_row_projection_v2,
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
    validate_receipt_graphs,
)
_HASH_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_MEASURED_OPERATION_LOGGER = logging.getLogger(
    "leadpoet.measured.operations"
)
_MEASURED_OPERATION_FIELDS = frozenset(
    {
        "authority_epoch_id",
        "authority_mode",
        "backlog_count",
        "bundle_hash",
        "component",
        "correlation_id",
        "cutover_epoch_id",
        "duration_seconds",
        "epoch_id",
        "frontier_epoch",
        "netuid",
        "observed_block",
        "observed_vector_count",
        "operation",
        "parent_count",
        "reason_code",
        "root_receipt_hash",
        "row_count",
        "runtime_sha",
        "sequence",
        "settlement_attempt",
        "settlement_hash",
        "source_epoch_id",
        "stage",
        "status",
        "validator_id_hash",
        "vector_hash",
    }
)


def observability_hash_identifier(value: Any) -> str:
    """Return an enclave-local, non-reversible telemetry join key."""

    encoded = str(value or "").encode("utf-8", errors="replace")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def record_operation_stage(**fields: Any) -> None:
    """Emit bounded measured-runtime telemetry without host SDK imports."""

    try:
        payload = {
            str(key): value
            for key, value in fields.items()
            if key in _MEASURED_OPERATION_FIELDS
            and value is not None
            and isinstance(value, (bool, int, float, str))
        }
        _MEASURED_OPERATION_LOGGER.log(
            logging.WARNING
            if payload.get("status") in {"failed", "rejected", "blocked"}
            else logging.INFO,
            "leadpoet_measured_operation_event %s",
            json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
            ),
        )
    except BaseException:
        return


@contextmanager
def operation_stage(**fields: Any) -> Iterator[None]:
    """Record one measured stage while preserving fail-closed exceptions."""

    started = time.monotonic()
    record_operation_stage(status="started", **fields)
    try:
        yield
    except BaseException:
        record_operation_stage(
            status="failed",
            duration_seconds=round(time.monotonic() - started, 3),
            **fields,
        )
        raise
    else:
        record_operation_stage(
            status="passed",
            duration_seconds=round(time.monotonic() - started, 3),
            **fields,
        )


_PURPOSE_V2 = {
    "research_lab.candidate_score.v1": "research_lab.candidate_score.v2",
    "research_lab.baseline_score.v1": "research_lab.baseline_score.v2",
    "research_lab.benchmark.v1": "research_lab.benchmark.v2",
    "research_lab.rebenchmark.v1": "research_lab.rebenchmark.v2",
    "research_lab.confirmation_score.v1": "research_lab.confirmation_score.v2",
}

_CHAIN_SETTLEMENT_RETRY_COOLDOWN_SECONDS = 300.0


class ResearchLabV2AuthorityError(RuntimeError):
    """A protected result did not have complete V2 enclave authority."""


def _validate_allocation_parent_graphs(
    graphs: Sequence[Mapping[str, Any]],
) -> list[dict[str, str]]:
    validate_receipt_graphs(graphs)
    bindings: list[dict[str, str]] = []
    for graph in graphs:
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
    return bindings


async def evaluate_source_add_provenance_v2(
    *,
    submission_id: str,
    source_name: str,
    source_kind: str,
    declared_base_domains: Sequence[str],
    source_metadata: Mapping[str, Any],
    epoch_id: int,
    sequence: int,
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
        sequence=max(0, int(sequence)),
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
    artifact = {
        "artifact_kind": "source_add_provenance",
        "artifact_ref": str(submission_id),
        "artifact_hash": str(receipt["output_root"]),
    }
    authority_outcome = dict(outcome)
    try:
        link = await _persist_business_links(
            outcome,
            (artifact,),
            persist_links=persist_links,
        )
    except Exception as exc:
        from gateway.research_lab.attested_v2_store import (
            AttestedV2StoreError,
            load_business_artifact_graph_v2,
        )

        if not isinstance(exc, AttestedV2StoreError) or str(exc) != (
            "research_lab_attested_business_artifact_links_v2 "
            "stored row conflicts at receipt_hash"
        ):
            raise
        existing_graph = await load_business_artifact_graph_v2(
            artifact_kind=artifact["artifact_kind"],
            artifact_ref=artifact["artifact_ref"],
            artifact_hash=artifact["artifact_hash"],
        )
        try:
            validate_receipt_graph(
                existing_graph,
                required_purposes={"research_lab.source_add_provenance.v2"},
            )
        except Exception as validation_exc:
            raise ResearchLabV2AuthorityError(
                "existing SOURCE_ADD provenance authority is invalid"
            ) from validation_exc
        existing_root_hash = str(existing_graph.get("root_receipt_hash") or "")
        existing_roots = [
            item
            for item in existing_graph.get("receipts") or ()
            if isinstance(item, Mapping)
            and item.get("receipt_hash") == existing_root_hash
        ]
        if (
            len(existing_roots) != 1
            or existing_roots[0].get("role") != "gateway_coordinator"
            or existing_roots[0].get("purpose")
            != "research_lab.source_add_provenance.v2"
            or existing_roots[0].get("status") != "succeeded"
            or existing_roots[0].get("output_root") != artifact["artifact_hash"]
        ):
            raise ResearchLabV2AuthorityError(
                "existing SOURCE_ADD provenance authority differs"
            ) from exc
        existing_receipt = dict(existing_roots[0])
        authority_outcome.update(
            {
                "execution_receipt": existing_receipt,
                "receipt": existing_receipt,
                "execution_receipt_graph": dict(existing_graph),
                "receipt_graph": dict(existing_graph),
            }
        )
        link = {
            "status": "reused_existing_authority",
            "receipt_hash": existing_root_hash,
        }
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
        **authority_outcome,
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
    graph = outcome.get("execution_receipt_graph") or outcome.get("receipt_graph")
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
    graph = outcome.get("execution_receipt_graph") or outcome.get("receipt_graph")
    receipt = outcome.get("execution_receipt") or outcome.get("receipt")
    if not isinstance(graph, Mapping) or not isinstance(receipt, Mapping):
        raise ResearchLabV2AuthorityError("SOURCE_ADD judge receipt graph is missing")
    validate_receipt_graph(graph, required_purposes={"research_lab.source_add_judge.v2"})
    if (
        graph.get("root_receipt_hash") != receipt.get("receipt_hash")
        or receipt.get("output_root") != sha256_json(dict(result))
    ):
        raise ResearchLabV2AuthorityError("SOURCE_ADD judge receipt differs")
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
    receipt = outcome.get("execution_receipt") or outcome.get("receipt")
    graph = outcome.get("execution_receipt_graph") or outcome.get("receipt_graph")
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
    provider_credential_profile: str = "provider_preflight",
    execute: Any = execute_scoring_v2,
) -> dict[str, Any]:
    measurement_id = uuid.uuid4().hex
    outcome = await execute(
        operation=OP_PROVIDER_PREFLIGHT_V2,
        purpose="research_lab.provider_preflight.v2",
        epoch_id=0,
        # Keep the receipt sequence inside the V2 INTEGER schema. Freshness is
        # committed by measurement_id in the payload so every requested probe
        # derives a new enclave job instead of replaying a terminal job for up
        # to the execution manager's one-hour retention window.
        sequence=0,
        payload={
            "schema_version": PROVIDER_PREFLIGHT_REQUEST_SCHEMA_VERSION,
            "measurement_id": measurement_id,
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
    graph = outcome.get("execution_receipt_graph")
    if graph is None:
        graph = outcome.get("receipt_graph")
    if not isinstance(receipt, Mapping) or not isinstance(graph, Mapping):
        raise ResearchLabV2AuthorityError("V2 authority receipt graph is missing")
    root = str(receipt.get("receipt_hash") or "")
    graph_receipts = {
        str(item.get("receipt_hash") or ""): item
        for item in graph.get("receipts", [])
        if isinstance(item, Mapping)
    }
    if (
        graph.get("root_receipt_hash") != root
        or root not in graph_receipts
        or graph_receipts[root] != dict(receipt)
    ):
        raise ResearchLabV2AuthorityError(
            "V2 authority execution receipt is absent from its graph"
        )
    if persist_links is None:
        from gateway.research_lab.attested_v2_store import (
            persist_business_artifact_links_v2,
        )

        persist_links = persist_business_artifact_links_v2
    return await persist_links(receipt_hash=root, artifacts=links)


def _current_allocation_frontier_outcome_v2(
    context: Mapping[str, Any],
    *,
    epoch_id: int,
    netuid: int,
) -> dict[str, Any]:
    """Recover the exact durable allocation authority for an active epoch.

    Execution receipts are release-bound. Re-executing an already persisted
    current-epoch frontier after a release change therefore creates a different
    receipt even when the protected result is byte-identical. The frontier is
    the immutable epoch authority, so recovery must reuse its authenticated
    source instead of attempting to mint a competing receipt.
    """

    source = context.get("source")
    frontier_row = context.get("row")
    frontier = context.get("frontier")
    if not isinstance(source, Mapping):
        raise ResearchLabV2AuthorityError(
            "current allocation frontier source is incomplete"
        )
    row = source.get("row")
    result = source.get("result")
    receipt = source.get("receipt")
    graph = source.get("receipt_graph")
    artifact_hashes = source.get("artifact_hashes")
    if (
        not isinstance(row, Mapping)
        or not isinstance(frontier_row, Mapping)
        or not isinstance(frontier, Mapping)
        or not isinstance(result, Mapping)
        or not isinstance(receipt, Mapping)
        or not isinstance(graph, Mapping)
        or not isinstance(artifact_hashes, list)
    ):
        raise ResearchLabV2AuthorityError(
            "current allocation frontier source is incomplete"
        )
    receipt_hash = str(receipt.get("receipt_hash") or "")
    release_hash = str(row.get("release_hash") or "").lower()
    source_state = result.get("source_state")
    if (
        row.get("operation") != OP_RESEARCH_LAB_ALLOCATION
        or row.get("purpose") != "research_lab.allocation.v2"
        or row.get("role") != "gateway_coordinator"
        or int(row.get("epoch_id", -1)) != int(epoch_id)
        or row.get("receipt_hash") != receipt_hash
        or not _HASH_RE.fullmatch(release_hash)
        or receipt.get("role") != "gateway_coordinator"
        or receipt.get("purpose") != "research_lab.allocation.v2"
        or receipt.get("status") != "succeeded"
        or int(receipt.get("epoch_id", -1)) != int(epoch_id)
        or graph.get("root_receipt_hash") != receipt_hash
        or frontier_row.get("source_receipt_hash") != receipt_hash
        or not isinstance(source_state, Mapping)
        or int(source_state.get("epoch", -1)) != int(epoch_id)
        or int(source_state.get("netuid", -1)) != int(netuid)
        or source_state.get("settlement_frontier") != frontier
    ):
        raise ResearchLabV2AuthorityError(
            "current allocation frontier source authority differs"
        )
    validate_receipt_graph(
        graph,
        required_purposes={"research_lab.allocation.v2"},
    )
    receipts = {
        str(item.get("receipt_hash") or ""): item
        for item in graph.get("receipts") or ()
        if isinstance(item, Mapping)
    }
    if receipts.get(receipt_hash) != dict(receipt):
        raise ResearchLabV2AuthorityError(
            "current allocation frontier receipt is absent from its graph"
        )
    persistence = {
        "graph_hash": sha256_json(dict(graph)),
        "root_receipt_hash": receipt_hash,
        "boot_count": len(graph.get("boot_identities") or ()),
        "receipt_count": len(graph.get("receipts") or ()),
        "transport_attempt_count": len(graph.get("transport_attempts") or ()),
        "host_operation_count": len(graph.get("host_operations") or ()),
    }
    return {
        "status": "succeeded",
        "result": dict(result),
        "receipt": dict(receipt),
        "execution_receipt": dict(receipt),
        "receipt_graph": dict(graph),
        "execution_receipt_graph": dict(graph),
        "transitions": [],
        "transport_attempts": [],
        "artifact_persistence": [],
        "artifact_hashes": list(artifact_hashes),
        "persistence": persistence,
        "sidecar_persistence": {},
        "release_hash": release_hash,
        "physical_role": "gateway_coordinator",
        "replay_status": "durable_current_frontier",
    }


async def build_allocation_v2(
    *,
    epoch_id: int,
    netuid: int,
    policy: Mapping[str, Any],
    allocation_sequence: int = 0,
    execute: Any = execute_coordinator_v2,
    persist_links: Any = None,
    load_allocation_parent_graphs: Any = None,
) -> dict[str, Any]:
    if (
        isinstance(allocation_sequence, bool)
        or not isinstance(allocation_sequence, int)
        or allocation_sequence < 0
    ):
        raise ResearchLabV2AuthorityError(
            "allocation sequence must be a non-negative integer"
        )
    using_default_parent_loader = load_allocation_parent_graphs is None
    if load_allocation_parent_graphs is None:
        load_allocation_parent_graphs = _load_allocation_parent_graphs_v2
    finalized_history: list[dict[str, Any]] = []
    settlement_frontier_context: Mapping[str, Any] | None = None
    current_frontier_context: Mapping[str, Any] | None = None
    readiness_authority_graph_records: dict[str, dict[str, Any]] = {}
    readiness_business_graphs: dict[
        tuple[str, str], dict[str, Any]
    ] = {}
    if using_default_parent_loader:
        from gateway.research_lab.attested_v2_store import (
            load_allocation_settlement_frontier_context_v2,
        )

        settlement_frontier_context = (
            await load_allocation_settlement_frontier_context_v2(
                netuid=int(netuid),
                before_epoch=int(epoch_id) + 1,
            )
        )
        if (
            settlement_frontier_context is not None
            and int(
                (settlement_frontier_context.get("frontier") or {}).get(
                    "allocation_epoch", -1
                )
            )
            == int(epoch_id)
        ):
            current_frontier_context = settlement_frontier_context
            settlement_frontier_context = None
        if current_frontier_context is None and execute is execute_coordinator_v2:
            await ensure_chain_realized_settlements_v1(
                epoch_id=int(epoch_id),
                netuid=int(netuid),
                execute=execute,
                settlement_attempt=int(allocation_sequence),
            )
        if settlement_frontier_context is None:
            if current_frontier_context is not None:
                source = current_frontier_context.get("source")
                source_receipt = (
                    source.get("receipt") if isinstance(source, Mapping) else None
                )
                if not isinstance(source_receipt, Mapping):
                    raise ResearchLabV2AuthorityError(
                        "current allocation frontier source is incomplete"
                    )
                parent_roots = source_receipt.get("parent_receipt_hashes")
                if not isinstance(parent_roots, list):
                    raise ResearchLabV2AuthorityError(
                        "current allocation frontier parents are invalid"
                    )
                graphs = await _graphs_for_roots(parent_roots)
            else:
                graphs = []
        if settlement_frontier_context is None and current_frontier_context is None:
            from gateway.research_lab.champion_settlement_v2 import (
                champion_v2_cutover_readiness,
            )

            readiness = await champion_v2_cutover_readiness(
                epoch=int(epoch_id),
                netuid=int(netuid),
                _finalized_history_out=finalized_history,
                _authority_graph_records_out=readiness_authority_graph_records,
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
    if current_frontier_context is None:
        parent_loader_kwargs = {
            "epoch_id": int(epoch_id),
            "netuid": int(netuid),
            "policy": dict(policy),
        }
        if using_default_parent_loader:
            parent_loader_kwargs.update(
                {
                    "finalized_champion_history": (
                        finalized_history
                        if settlement_frontier_context is None
                        else None
                    ),
                    "preloaded_receipt_graph_records": (
                        readiness_authority_graph_records
                    ),
                    "preloaded_business_graphs": readiness_business_graphs,
                    "settlement_frontier_context": settlement_frontier_context,
                }
            )
        graphs = list(await load_allocation_parent_graphs(**parent_loader_kwargs))
    bindings = await asyncio.to_thread(_validate_allocation_parent_graphs, graphs)
    if current_frontier_context is not None:
        outcome = _current_allocation_frontier_outcome_v2(
            current_frontier_context,
            epoch_id=int(epoch_id),
            netuid=int(netuid),
        )
    else:
        outcome = await execute(
            operation=OP_RESEARCH_LAB_ALLOCATION,
            purpose="research_lab.allocation.v2",
            epoch_id=int(epoch_id),
            sequence=int(allocation_sequence),
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
    from leadpoet_canonical.allocation_settlement_frontier_v2 import (
        validate_allocation_settlement_frontier_v2,
    )

    settlement_frontier = validate_allocation_settlement_frontier_v2(
        source_state.get("settlement_frontier")
    )
    if (
        int(settlement_frontier["netuid"]) != int(netuid)
        or int(settlement_frontier["allocation_epoch"]) != int(epoch_id)
    ):
        raise ResearchLabV2AuthorityError(
            "allocation settlement frontier scope differs"
        )
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
    fallback_obligations = list(
        source_state.get("fallback_reimbursement_obligations") or []
    )
    if fallback_obligations:
        expected_inputs["fallback_reimbursement_obligations"] = (
            fallback_obligations
        )
    _assert_equal(allocation_inputs, expected_inputs, "allocation source projection")
    if (
        using_default_parent_loader
        and execute is execute_coordinator_v2
        and current_frontier_context is None
    ):
        from gateway.research_lab.attested_v2_store import (
            persist_allocation_settlement_frontier_v2,
        )

        execution_receipt = outcome.get("execution_receipt") or outcome.get(
            "receipt"
        )
        if not isinstance(execution_receipt, Mapping):
            raise ResearchLabV2AuthorityError(
                "allocation settlement frontier receipt is missing"
            )
        await persist_allocation_settlement_frontier_v2(
            frontier=settlement_frontier,
            source_receipt_hash=str(
                execution_receipt.get("receipt_hash") or ""
            ),
            source_state_hash=str(authority_result["source_state_hash"]),
        )
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


async def _resolve_chain_settlement_attempt_v1(
    *,
    epoch_id: int,
    requested_attempt: int,
    load_attempt_history: Any,
) -> int:
    rows = await load_attempt_history(
        "research_lab_attested_execution_receipts_v2",
        columns="purpose,sequence,receipt_status,issued_at",
        filters=(
            ("role", "gateway_coordinator"),
            ("epoch_id", int(epoch_id)),
            (
                "purpose",
                "in",
                (
                    CHAIN_WEIGHT_OBSERVATION_PURPOSE_V1,
                    CHAIN_REALIZED_SETTLEMENT_PURPOSE_V1,
                ),
            ),
        ),
        order_by=(("sequence", True),),
        limit=1,
    )
    if not rows:
        return int(requested_attempt)
    latest = rows[0]
    durable_sequence = latest.get("sequence")
    if (
        isinstance(durable_sequence, bool)
        or not isinstance(durable_sequence, int)
        or durable_sequence < 0
    ):
        raise ResearchLabV2AuthorityError(
            "chain-realized settlement attempt history is invalid"
        )
    receipt_status = latest.get("receipt_status")
    if receipt_status not in {"failed", "succeeded"}:
        raise ResearchLabV2AuthorityError(
            "chain-realized settlement attempt history is invalid"
        )
    if receipt_status == "failed":
        raw_issued_at = latest.get("issued_at")
        try:
            issued_at = datetime.fromisoformat(
                str(raw_issued_at).replace("Z", "+00:00")
            )
            if issued_at.tzinfo is None:
                issued_at = issued_at.replace(tzinfo=timezone.utc)
            age_seconds = (
                datetime.now(timezone.utc) - issued_at.astimezone(timezone.utc)
            ).total_seconds()
        except (TypeError, ValueError):
            raise ResearchLabV2AuthorityError(
                "chain-realized settlement attempt history is invalid"
            ) from None
        if age_seconds < _CHAIN_SETTLEMENT_RETRY_COOLDOWN_SECONDS:
            raise ResearchLabV2AuthorityError(
                "chain-realized settlement retry is cooling down"
            )
    return max(int(requested_attempt), (durable_sequence // 2) + 1)


async def settle_chain_realized_epoch_v1(
    *,
    epoch_id: int,
    netuid: int,
    settlement_attempt: int = 0,
    execute: Any = execute_coordinator_v2,
    load_attempt_history: Any = None,
    persist_settlement: Any = None,
    select_candidates: Any = None,
    load_graph: Any = None,
) -> dict[str, Any]:
    """Prove and atomically persist one epoch's realized primary vector."""

    from gateway.research_lab.champion_settlement_v2 import (
        CHAIN_REALIZED_EPOCH_SETTLEMENT_SCHEMA_VERSION_V1,
        CHAIN_REALIZED_EPOCH_SETTLEMENT_SCHEMA_VERSION_V2,
        CHAIN_REALIZED_EPOCH_SETTLEMENT_SCHEMA_VERSION_V3,
        CHAIN_REALIZED_CHAMPION_CREDIT_POLICY_LEGACY_V1,
        CHAIN_WEIGHT_OBSERVATION_SCHEMA_VERSION_V2,
        COMPACT_WEIGHT_AUTHORITY_TABLE_V2,
        ChampionSettlementV2Error,
        select_compact_chain_realized_bundle_candidate_v2,
        select_chain_realized_bundle_candidate_v1,
        validate_chain_realized_epoch_settlements_v1,
        validate_chain_realized_obligation_credits_v1,
        validate_chain_weight_observation_v1,
    )
    from gateway.research_lab.store import select_many

    normalized_epoch = int(epoch_id)
    normalized_netuid = int(netuid)
    if (
        isinstance(settlement_attempt, bool)
        or not isinstance(settlement_attempt, int)
        or settlement_attempt < 0
    ):
        raise ResearchLabV2AuthorityError(
            "chain-realized settlement attempt must be a non-negative integer"
        )
    resolved_settlement_attempt = int(settlement_attempt)
    if load_attempt_history is not None or execute is execute_coordinator_v2:
        if load_attempt_history is None:
            from gateway.research_lab.store import select_many

            load_attempt_history = select_many
        resolved_settlement_attempt = await _resolve_chain_settlement_attempt_v1(
            epoch_id=normalized_epoch,
            requested_attempt=resolved_settlement_attempt,
            load_attempt_history=load_attempt_history,
        )
    observation_sequence = resolved_settlement_attempt * 2
    settlement_sequence = observation_sequence + 1
    if normalized_epoch < 0 or normalized_netuid <= 0:
        raise ResearchLabV2AuthorityError(
            "chain-realized settlement scope is invalid"
        )
    settlement_telemetry = {
        "runtime_sha": str(
            os.environ.get("GITHUB_SHA")
            or os.environ.get("GITHUB_COMMIT")
            or os.environ.get("GIT_COMMIT")
            or ""
        ).lower(),
        "netuid": normalized_netuid,
        "epoch_id": normalized_epoch,
        "settlement_attempt": resolved_settlement_attempt,
        "correlation_id": observability_hash_identifier(
            "chain_realized_settlement:%d:%d"
            % (normalized_netuid, normalized_epoch)
        ),
    }
    record_operation_stage(
        component="research_lab",
        operation="chain_realized_settlement",
        stage="settlement_run",
        status="started",
        **settlement_telemetry,
    )
    with operation_stage(
        component="research_lab",
        operation="chain_realized_settlement",
        stage="measured_chain_observation",
        sequence=observation_sequence,
        **settlement_telemetry,
    ):
        observation_outcome = await execute(
            operation=OP_OBSERVE_CHAIN_REALIZED_WEIGHTS_V1,
            purpose=CHAIN_WEIGHT_OBSERVATION_PURPOSE_V1,
            epoch_id=normalized_epoch,
            sequence=observation_sequence,
            payload={
                "schema_version": (
                    "leadpoet.chain_realized_weight_observation_request.v1"
                ),
                "netuid": normalized_netuid,
                "epoch_id": normalized_epoch,
            },
            parent_graphs=(),
        )
    observation_result = observation_outcome.get("result")
    observation_receipt = (
        observation_outcome.get("execution_receipt")
        or observation_outcome.get("receipt")
    )
    observation_graph = (
        observation_outcome.get("execution_receipt_graph")
        or observation_outcome.get("receipt_graph")
    )
    with operation_stage(
        component="research_lab",
        operation="chain_realized_settlement",
        stage="chain_observation_verification",
        **settlement_telemetry,
    ):
        if (
            not isinstance(observation_result, Mapping)
            or not isinstance(observation_receipt, Mapping)
            or not isinstance(observation_graph, Mapping)
        ):
            raise ResearchLabV2AuthorityError(
                "chain weight observation authority is incomplete"
            )
        observation = validate_chain_weight_observation_v1(
            observation_result
        )
        observation_hash = sha256_json(observation)
        observation_receipt_hash = str(
            observation_receipt.get("receipt_hash") or ""
        )
        if (
            observation_receipt.get("role") != "gateway_coordinator"
            or observation_receipt.get("purpose")
            != CHAIN_WEIGHT_OBSERVATION_PURPOSE_V1
            or observation_receipt.get("status") != "succeeded"
            or int(observation_receipt.get("epoch_id", -1))
            != normalized_epoch
            or observation_receipt.get("output_root") != observation_hash
            or observation_graph.get("root_receipt_hash")
            != observation_receipt_hash
        ):
            raise ResearchLabV2AuthorityError(
                "chain weight observation receipt differs"
            )
        validate_receipt_graph(
            observation_graph,
            required_purposes={CHAIN_WEIGHT_OBSERVATION_PURPOSE_V1},
        )
    settlement_telemetry.update(
        {
            "source_epoch_id": int(
                observation.get("latest_commit_source_epoch_id")
                if observation["schema_version"]
                == CHAIN_WEIGHT_OBSERVATION_SCHEMA_VERSION_V2
                else observation["active_source_epoch_id"]
            ),
            "validator_id_hash": observability_hash_identifier(
                observation["validator_hotkey"]
            ),
            "observed_block": int(observation["last_update_block"]),
            "observed_vector_count": len(observation["weights"]),
            "vector_hash": str(observation["weights_vector_hash"]),
            "root_receipt_hash": observation_receipt_hash,
        }
    )

    if select_candidates is None:
        select_candidates = select_many
    observation_v2 = (
        observation["schema_version"]
        == CHAIN_WEIGHT_OBSERVATION_SCHEMA_VERSION_V2
    )
    with operation_stage(
        component="research_lab",
        operation="chain_realized_settlement",
        stage="compact_authority_cutover_lookup",
        **settlement_telemetry,
    ):
        compact_cutover_rows = (
            []
            if observation_v2
            else await select_candidates(
                COMPACT_WEIGHT_AUTHORITY_TABLE_V2,
                columns="epoch_id",
                filters=(
                    ("netuid", normalized_netuid),
                    ("authority_stage", "finalized"),
                ),
                order_by=(("epoch_id", False),),
                limit=1,
            )
        )
    compact_cutover_epoch: int | None = None
    if compact_cutover_rows:
        if len(compact_cutover_rows) != 1 or set(
            compact_cutover_rows[0]
        ) != {"epoch_id"}:
            raise ResearchLabV2AuthorityError(
                "compact weight authority cutover is invalid"
            )
        raw_cutover_epoch = compact_cutover_rows[0]["epoch_id"]
        if isinstance(raw_cutover_epoch, bool):
            raise ResearchLabV2AuthorityError(
                "compact weight authority cutover is invalid"
            )
        try:
            compact_cutover_epoch = int(raw_cutover_epoch)
        except (TypeError, ValueError) as exc:
            raise ResearchLabV2AuthorityError(
                "compact weight authority cutover is invalid"
            ) from exc
        if compact_cutover_epoch < 0:
            raise ResearchLabV2AuthorityError(
                "compact weight authority cutover is invalid"
            )
    source_epoch_id = int(
        observation[
            "latest_commit_source_epoch_id"
            if observation_v2
            else "active_source_epoch_id"
        ]
    )
    use_compact_authority = (
        observation_v2
        and observation["revealed_bundle_hash"] is not None
    ) or (
        not observation_v2
        and compact_cutover_epoch is not None
        and source_epoch_id >= compact_cutover_epoch
    )
    authority_mode_label = (
        (
            "event_proved_compact"
            if use_compact_authority
            else "event_unattributed"
        )
        if observation_v2
        else "compact_finalized"
        if use_compact_authority
        else "legacy_finalized"
    )
    authority_telemetry = {
        **settlement_telemetry,
        "authority_mode": authority_mode_label,
        "cutover_epoch_id": compact_cutover_epoch,
    }
    with operation_stage(
        component="research_lab",
        operation="chain_realized_settlement",
        stage="finalized_authority_lookup",
        **authority_telemetry,
    ):
        candidate = None
        candidate_rows = []
        candidate_row_count = 0
        if observation_v2:
            revealed_bundle_hash = observation["revealed_bundle_hash"]
            if revealed_bundle_hash is not None:
                candidate_rows = await select_candidates(
                    COMPACT_WEIGHT_AUTHORITY_TABLE_V2,
                    columns=(
                        "bundle_hash,compact_submission_hash,netuid,epoch_id,"
                        "validator_hotkey,authority_stage,schema_version,lineage_id,"
                        "authority_hash,publication_receipt_hash,"
                        "compact_finalization_hash,finalization_receipt_hash,authority_doc"
                    ),
                    filters=(
                        ("netuid", normalized_netuid),
                        ("bundle_hash", str(revealed_bundle_hash)),
                        ("authority_stage", "finalized"),
                    ),
                    limit=2,
                )
                try:
                    candidate = select_compact_chain_realized_bundle_candidate_v2(
                        candidate_rows,
                        observation=observation,
                    )
                except ChampionSettlementV2Error as exc:
                    raise ResearchLabV2AuthorityError(str(exc)) from exc
            candidate_row_count = len(candidate_rows)
        elif use_compact_authority:
            candidate_rows = await select_candidates(
                COMPACT_WEIGHT_AUTHORITY_TABLE_V2,
                columns=(
                    "bundle_hash,compact_submission_hash,netuid,epoch_id,"
                    "validator_hotkey,authority_stage,schema_version,lineage_id,"
                    "authority_hash,publication_receipt_hash,"
                    "compact_finalization_hash,finalization_receipt_hash,authority_doc"
                ),
                filters=(
                    ("netuid", normalized_netuid),
                    ("epoch_id", source_epoch_id),
                    (
                        "validator_hotkey",
                        str(observation["validator_hotkey"]),
                    ),
                    ("authority_stage", "finalized"),
                ),
                order_by=(("bundle_hash", False),),
                limit=2,
            )
            candidate_row_count = len(candidate_rows)
        else:
            candidate_rows = await select_candidates(
                "research_lab_finalized_weight_vector_candidates_v1",
                filters=(
                    ("netuid", normalized_netuid),
                    ("epoch_id", source_epoch_id),
                    ("validator_hotkey", str(observation["validator_hotkey"])),
                    ("finalized_block", int(observation["last_update_block"])),
                    (
                        "finalized_block_hash",
                        str(observation["last_update_block_hash"]),
                    ),
                    ("uids", [int(item[0]) for item in observation["weights"]]),
                    (
                        "weights_u16",
                        [int(item[1]) for item in observation["weights"]],
                    ),
                ),
                order_by=(("finalized_block", True), ("bundle_hash", False)),
                limit=100,
            )
            candidate_row_count = len(candidate_rows)
    record_operation_stage(
        component="research_lab",
        operation="chain_realized_settlement",
        stage="finalized_authority_lookup_result",
        status="passed",
        row_count=candidate_row_count,
        **authority_telemetry,
    )
    finalization_graphs: list[dict[str, Any]] = []
    if candidate_rows and candidate is None:
        try:
            candidate = (
                select_compact_chain_realized_bundle_candidate_v2(
                    candidate_rows,
                    observation=observation,
                )
                if use_compact_authority
                else select_chain_realized_bundle_candidate_v1(
                    candidate_rows,
                    observation=observation,
                )
            )
        except ChampionSettlementV2Error as exc:
            raise ResearchLabV2AuthorityError(str(exc)) from exc
    if observation_v2 and (
        (candidate is None)
        != (observation["revealed_bundle_hash"] is None)
    ):
        raise ResearchLabV2AuthorityError(
            "event-proved compact authority is unavailable"
        )
    if candidate is not None:
        finalization_receipt_hash = str(
            candidate["finalization_receipt_hash"]
        )
        with operation_stage(
            component="research_lab",
            operation="chain_realized_settlement",
            stage="finalization_receipt_graph_load",
            **{
                **authority_telemetry,
                "bundle_hash": candidate["bundle_hash"],
                "root_receipt_hash": finalization_receipt_hash,
            },
        ):
            finalization_graphs = await _graphs_for_roots(
                {finalization_receipt_hash},
                load_graph=load_graph,
            )
        if len(finalization_graphs) != 1:
            raise ResearchLabV2AuthorityError(
                "chain settlement finalization graph is ambiguous"
            )
    authority_mode = (
        "finalized_bundle" if candidate is not None else "unattributed"
    )

    measured_telemetry = {
        **authority_telemetry,
        "authority_mode": authority_mode,
        "bundle_hash": (
            str(candidate["bundle_hash"])
            if candidate is not None
            else None
        ),
    }
    with operation_stage(
        component="research_lab",
        operation="chain_realized_settlement",
        stage="measured_settlement_execution",
        sequence=settlement_sequence,
        parent_count=1 + len(finalization_graphs),
        **measured_telemetry,
    ):
        settlement_outcome = await execute(
            operation=OP_ATTEST_CHAIN_REALIZED_SETTLEMENT_V1,
            purpose=CHAIN_REALIZED_SETTLEMENT_PURPOSE_V1,
            epoch_id=normalized_epoch,
            sequence=settlement_sequence,
            payload={
                "schema_version": "leadpoet.chain_realized_settlement_request.v1",
                "netuid": normalized_netuid,
                "epoch_id": normalized_epoch,
                "observation": observation,
                "observation_receipt_hash": observation_receipt_hash,
                "authority_mode": authority_mode,
                "bundle_hash": (
                    str(candidate["bundle_hash"])
                    if candidate is not None
                    else None
                ),
            },
            parent_graphs=(
                dict(observation_graph),
                *(dict(graph) for graph in finalization_graphs),
            ),
        )
    package = settlement_outcome.get("result")
    settlement_receipt = (
        settlement_outcome.get("execution_receipt")
        or settlement_outcome.get("receipt")
    )
    settlement_graph = (
        settlement_outcome.get("execution_receipt_graph")
        or settlement_outcome.get("receipt_graph")
    )
    record_operation_stage(
        component="research_lab",
        operation="chain_realized_settlement",
        stage="settlement_package_verification",
        status="started",
        **measured_telemetry,
    )
    if (
        not isinstance(package, Mapping)
        or not isinstance(settlement_receipt, Mapping)
        or not isinstance(settlement_graph, Mapping)
    ):
        record_operation_stage(
            component="research_lab",
            operation="chain_realized_settlement",
            stage="settlement_package_verification",
            status="failed",
            reason_code="settlement_authority_incomplete",
            **measured_telemetry,
        )
        raise ResearchLabV2AuthorityError(
            "chain-realized settlement authority is incomplete"
        )
    settlement_doc = package.get("settlement_doc")
    credits = package.get("credits")
    settlement_hash = str(package.get("settlement_hash") or "")
    settlement_receipt_hash = str(
        settlement_receipt.get("receipt_hash") or ""
    )
    if (
        set(package) != {"settlement_doc", "settlement_hash", "credits"}
        or not isinstance(settlement_doc, Mapping)
        or not isinstance(credits, list)
        or settlement_hash != sha256_json(dict(settlement_doc))
        or settlement_doc.get("schema_version")
        not in {
            CHAIN_REALIZED_EPOCH_SETTLEMENT_SCHEMA_VERSION_V1,
            CHAIN_REALIZED_EPOCH_SETTLEMENT_SCHEMA_VERSION_V2,
            CHAIN_REALIZED_EPOCH_SETTLEMENT_SCHEMA_VERSION_V3,
        }
        or settlement_receipt.get("role") != "gateway_coordinator"
        or settlement_receipt.get("purpose")
        != CHAIN_REALIZED_SETTLEMENT_PURPOSE_V1
        or settlement_receipt.get("status") != "succeeded"
        or int(settlement_receipt.get("epoch_id", -1))
        != normalized_epoch
        or settlement_receipt.get("output_root") != settlement_hash
        or settlement_graph.get("root_receipt_hash")
        != settlement_receipt_hash
    ):
        record_operation_stage(
            component="research_lab",
            operation="chain_realized_settlement",
            stage="settlement_package_verification",
            status="failed",
            reason_code="settlement_receipt_differs",
            settlement_hash=settlement_hash,
            **measured_telemetry,
        )
        raise ResearchLabV2AuthorityError(
            "chain-realized settlement receipt differs"
        )
    with operation_stage(
        component="research_lab",
        operation="chain_realized_settlement",
        stage="settlement_receipt_graph_verification",
        **{
            **measured_telemetry,
            "settlement_hash": settlement_hash,
            "root_receipt_hash": settlement_receipt_hash,
        },
    ):
        validate_receipt_graph(
            settlement_graph,
            required_purposes={
                CHAIN_WEIGHT_OBSERVATION_PURPOSE_V1,
                CHAIN_REALIZED_SETTLEMENT_PURPOSE_V1,
            },
        )
    record_operation_stage(
        component="research_lab",
        operation="chain_realized_settlement",
        stage="settlement_package_verification",
        status="passed",
        **{
            **measured_telemetry,
            "settlement_hash": settlement_hash,
            "root_receipt_hash": settlement_receipt_hash,
            "row_count": len(credits),
        },
    )
    settlement_row = {
        "netuid": normalized_netuid,
        "epoch_id": normalized_epoch,
        "schema_version": str(settlement_doc["schema_version"]),
        "settlement_hash": settlement_hash,
        "settlement_receipt_hash": settlement_receipt_hash,
        "settlement_doc": dict(settlement_doc),
    }
    graph_by_root = {settlement_receipt_hash: dict(settlement_graph)}
    normalized_settlements = validate_chain_realized_epoch_settlements_v1(
        [settlement_row],
        receipt_graphs=graph_by_root,
    )
    credit_rows = [
        {
            "netuid": normalized_netuid,
            "epoch_id": normalized_epoch,
            "settlement_hash": settlement_hash,
            "schema_version": str(item["credit_doc"]["schema_version"]),
            "obligation_kind": str(
                item["credit_doc"]["obligation_kind"]
            ),
            "obligation_source_id": str(
                item["credit_doc"]["obligation_source_id"]
            ),
            "miner_hotkey": str(item["credit_doc"]["miner_hotkey"]),
            "miner_uid": int(item["credit_doc"]["miner_uid"]),
            "observed_chain_alpha_percent": str(
                item["credit_doc"]["observed_chain_alpha_percent"]
            ),
            "lab_attributed_alpha_percent": str(
                item["credit_doc"]["lab_attributed_alpha_percent"]
            ),
            "scheduled_alpha_percent": str(
                item["credit_doc"]["scheduled_alpha_percent"]
            ),
            "credited_alpha_percent": str(
                item["credit_doc"]["credited_alpha_percent"]
            ),
            "champion_credit_policy": str(
                item["credit_doc"].get("champion_credit_policy")
                or CHAIN_REALIZED_CHAMPION_CREDIT_POLICY_LEGACY_V1
            ),
            "credit_hash": str(item["credit_hash"]),
            "credit_receipt_hash": settlement_receipt_hash,
            "credit_doc": dict(item["credit_doc"]),
        }
        for item in credits
        if isinstance(item, Mapping)
        and isinstance(item.get("credit_doc"), Mapping)
    ]
    if len(credit_rows) != len(credits):
        raise ResearchLabV2AuthorityError(
            "chain-realized settlement credit package is invalid"
        )
    validate_chain_realized_obligation_credits_v1(
        credit_rows,
        settlement_rows=normalized_settlements,
        receipt_graphs=graph_by_root,
    )
    if persist_settlement is None:
        from gateway.research_lab.attested_v2_store import (
            persist_chain_realized_settlement_v1,
        )

        persist_settlement = persist_chain_realized_settlement_v1
    durable_telemetry = {
        **measured_telemetry,
        "settlement_hash": settlement_hash,
        "root_receipt_hash": settlement_receipt_hash,
        "row_count": len(credit_rows),
    }
    with operation_stage(
        component="research_lab",
        operation="chain_realized_settlement",
        stage="settlement_durable_persistence",
        **durable_telemetry,
    ):
        durable = await persist_settlement(
            package=package,
            receipt_hash=settlement_receipt_hash,
        )
    record_operation_stage(
        component="research_lab",
        operation="chain_realized_settlement",
        stage="settlement_run",
        status="completed",
        **durable_telemetry,
    )
    return {
        **dict(settlement_outcome),
        "status": "settled",
        "observation": observation,
        "durable_settlement": durable,
    }


async def ensure_chain_realized_settlements_v1(
    *,
    epoch_id: int,
    netuid: int,
    settlement_attempt: int = 0,
    execute: Any = execute_coordinator_v2,
    settle: Any = settle_chain_realized_epoch_v1,
    load_latest: Any = None,
    maximum_backlog: int = 100,
) -> list[dict[str, Any]]:
    """Fill every post-activation settlement through the prior epoch."""

    from gateway.research_lab.champion_settlement_v2 import (
        CHAIN_REALIZED_EPOCH_SETTLEMENT_TABLE_V1,
        CHAIN_REALIZED_SETTLEMENT_ACTIVATION_TABLE_V1,
    )
    from gateway.research_lab.store import select_many

    current_epoch = int(epoch_id)
    normalized_netuid = int(netuid)
    if (
        isinstance(settlement_attempt, bool)
        or not isinstance(settlement_attempt, int)
        or settlement_attempt < 0
    ):
        raise ResearchLabV2AuthorityError(
            "chain-realized settlement attempt must be a non-negative integer"
        )
    target_epoch = current_epoch - 1
    if target_epoch < 0:
        return []
    if load_latest is None:
        load_latest = select_many
    activation_rows = await load_latest(
        CHAIN_REALIZED_SETTLEMENT_ACTIVATION_TABLE_V1,
        filters=(("netuid", normalized_netuid),),
        order_by=(("first_epoch_id", False),),
        limit=2,
    )
    if len(activation_rows) != 1:
        raise ResearchLabV2AuthorityError(
            "chain-realized settlement activation is unavailable or ambiguous"
        )
    activation = activation_rows[0]
    try:
        activation_netuid = int(activation["netuid"])
        activation_epoch = int(activation["first_epoch_id"])
        source_epoch = int(activation["source_bundle_epoch_id"])
        source_finalized_block = int(activation["source_finalized_block"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ResearchLabV2AuthorityError(
            "chain-realized settlement activation is invalid"
        ) from exc
    if (
        activation.get("schema_version")
        != "leadpoet.research_lab_chain_realized_settlement_activation.v1"
        or activation_netuid != normalized_netuid
        or activation_epoch < 0
        or source_epoch != activation_epoch
        or source_finalized_block < 0
        or not _HASH_RE.fullmatch(
            str(activation.get("source_bundle_hash") or "")
        )
    ):
        raise ResearchLabV2AuthorityError(
            "chain-realized settlement activation is invalid"
        )
    rows = await load_latest(
        CHAIN_REALIZED_EPOCH_SETTLEMENT_TABLE_V1,
        filters=(("netuid", normalized_netuid),),
        order_by=(("epoch_id", True),),
        limit=2,
    )
    if len(rows) > 1 and int(rows[0]["epoch_id"]) == int(rows[1]["epoch_id"]):
        raise ResearchLabV2AuthorityError(
            "chain-realized settlement latest epoch is ambiguous"
        )
    if rows:
        latest_epoch = int(rows[0]["epoch_id"])
        if latest_epoch > target_epoch:
            raise ResearchLabV2AuthorityError(
                "chain-realized settlement is ahead of allocation epoch"
            )
        if latest_epoch < activation_epoch:
            raise ResearchLabV2AuthorityError(
                "chain-realized settlement predates its activation"
            )
        first_epoch = latest_epoch + 1
    else:
        first_epoch = activation_epoch
    if first_epoch > target_epoch:
        return []
    backlog = target_epoch - first_epoch + 1
    if backlog > int(maximum_backlog):
        raise ResearchLabV2AuthorityError(
            "chain-realized settlement backlog exceeds policy"
        )
    results = []
    backlog_telemetry = {
        "runtime_sha": str(
            os.environ.get("GITHUB_SHA")
            or os.environ.get("GITHUB_COMMIT")
            or os.environ.get("GIT_COMMIT")
            or ""
        ).lower(),
        "netuid": normalized_netuid,
        "epoch_id": current_epoch,
        "frontier_epoch": first_epoch,
        "authority_epoch_id": target_epoch,
        "backlog_count": backlog,
        "settlement_attempt": settlement_attempt,
        "correlation_id": observability_hash_identifier(
            "chain_realized_settlement_backlog:%d:%d"
            % (normalized_netuid, current_epoch)
        ),
    }
    record_operation_stage(
        component="research_lab",
        operation="chain_realized_settlement",
        stage="settlement_backlog_plan",
        status="passed",
        **backlog_telemetry,
    )
    for settlement_epoch in range(first_epoch, target_epoch + 1):
        with operation_stage(
            component="research_lab",
            operation="chain_realized_settlement",
            stage="settlement_backlog_epoch",
            source_epoch_id=settlement_epoch,
            **backlog_telemetry,
        ):
            results.append(
                await settle(
                    epoch_id=settlement_epoch,
                    netuid=normalized_netuid,
                    settlement_attempt=int(settlement_attempt),
                    execute=execute,
                )
            )
    record_operation_stage(
        component="research_lab",
        operation="chain_realized_settlement",
        stage="settlement_backlog_complete",
        status="completed",
        row_count=len(results),
        **backlog_telemetry,
    )
    return results


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
    preloaded_receipt_graph_records: (
        Mapping[str, Mapping[str, Any]] | None
    ) = None,
    preloaded_business_graphs: Mapping[
        tuple[str, str], Mapping[str, Any]
    ] | None = None,
    settlement_frontier_context: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Load candidate parent graphs; the enclave independently checks completeness."""

    from gateway.research_lab.attested_v2_store import (
        load_business_artifact_graphs_v2,
        load_business_artifact_graphs_by_ref_v2,
        load_receipt_graphs_v2,
    )
    from gateway.research_lab.allocations import (
        POSTGREST_IN_FILTER_CHUNK,
        _load_latest_finalized_compute_snapshot_v2,
        champion_reward_requires_allocation_history_v2,
    )
    from gateway.research_lab.champion_settlement_v2 import (
        CHAIN_REALIZED_AUTHORITY_TYPE_V1,
        load_settled_allocation_history_v2,
    )
    from gateway.research_lab.store import select_all

    graphs: dict[str, dict[str, Any]] = {}
    artifact_refs: set[tuple[str, str]] = set()
    exact_artifact_refs: dict[tuple[str, str], str] = {}
    receipt_roots: set[str] = set()
    preloaded = dict(preloaded_business_graphs or {})
    preloaded_receipts: dict[str, tuple[int, dict[str, Any]]] = {}
    prior_frontier: Mapping[str, Any] | None = None

    if settlement_frontier_context is not None:
        from leadpoet_canonical.allocation_settlement_frontier_v2 import (
            validate_allocation_settlement_frontier_v2,
        )

        raw_frontier = settlement_frontier_context.get("frontier")
        source = settlement_frontier_context.get("source")
        activation = settlement_frontier_context.get("activation")
        activation_source = settlement_frontier_context.get(
            "activation_source"
        )
        source_graph = (
            source.get("receipt_graph") if isinstance(source, Mapping) else None
        )
        source_receipt_hash = str(
            (settlement_frontier_context.get("row") or {}).get(
                "source_receipt_hash"
            )
            if isinstance(settlement_frontier_context.get("row"), Mapping)
            else ""
        )
        try:
            prior_frontier = validate_allocation_settlement_frontier_v2(
                raw_frontier
            )
        except ValueError as exc:
            raise ResearchLabV2AuthorityError(
                "allocation settlement frontier is invalid"
            ) from exc
        if (
            int(prior_frontier["netuid"]) != int(netuid)
            or int(prior_frontier["allocation_epoch"]) >= int(epoch_id)
            or not isinstance(source_graph, Mapping)
            or source_graph.get("root_receipt_hash") != source_receipt_hash
        ):
            raise ResearchLabV2AuthorityError(
                "allocation settlement frontier context differs"
            )
        validate_receipt_graph(source_graph)
        graphs[source_receipt_hash] = dict(source_graph)
        activation_receipt_hash = str(
            activation.get("source_receipt_hash")
            if isinstance(activation, Mapping)
            else ""
        )
        activation_graph = (
            activation_source.get("receipt_graph")
            if isinstance(activation_source, Mapping)
            else None
        )
        if (
            not _HASH_RE.fullmatch(activation_receipt_hash)
            or not isinstance(activation_graph, Mapping)
            or activation_graph.get("root_receipt_hash")
            != activation_receipt_hash
        ):
            raise ResearchLabV2AuthorityError(
                "allocation settlement frontier activation context differs"
            )
        validate_receipt_graph(activation_graph)
        graphs[activation_receipt_hash] = dict(activation_graph)

    def add_preloaded_receipt_record(
        declared_root: str,
        raw_record: Mapping[str, Any],
    ) -> None:
        root = str(declared_root or "")
        if not isinstance(raw_record, Mapping):
            raise ResearchLabV2AuthorityError(
                "allocation preloaded receipt graph record is invalid"
            )
        raw_graph = raw_record.get("graph")
        try:
            source_epoch = int(raw_record["epoch_id"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ResearchLabV2AuthorityError(
                "allocation preloaded receipt graph epoch is invalid"
            ) from exc
        if not isinstance(raw_graph, Mapping):
            raise ResearchLabV2AuthorityError(
                "allocation preloaded receipt graph is invalid"
            )
        normalized = dict(raw_graph)
        if (
            source_epoch < 0
            or not _HASH_RE.fullmatch(root)
            or str(normalized.get("root_receipt_hash") or "") != root
        ):
            raise ResearchLabV2AuthorityError(
                "allocation preloaded receipt graph root differs"
            )
        preloaded_receipts[root] = (source_epoch, normalized)

    for declared_root, raw_record in dict(
        preloaded_receipt_graph_records or {}
    ).items():
        add_preloaded_receipt_record(declared_root, raw_record)

    def add(kind: str, ref: str) -> None:
        key = (str(kind or ""), str(ref or ""))
        if key in exact_artifact_refs:
            return
        graph = preloaded.get(key)
        if isinstance(graph, Mapping):
            normalized = dict(graph)
            graphs[str(normalized.get("root_receipt_hash") or "")] = normalized
            return
        artifact_refs.add(key)

    def add_exact(kind: str, ref: str, artifact_hash: str) -> None:
        key = (str(kind or ""), str(ref or ""))
        digest = str(artifact_hash or "").lower()
        if not key[0] or not key[1] or not _HASH_RE.fullmatch(digest):
            raise ResearchLabV2AuthorityError(
                "allocation finalized artifact identity is invalid"
            )
        existing = exact_artifact_refs.get(key)
        if existing is not None and existing != digest:
            raise ResearchLabV2AuthorityError(
                "allocation finalized artifact identities conflict"
            )
        exact_artifact_refs[key] = digest
        artifact_refs.discard(key)

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

    if policy.get("enable_conservative", True) is False:
        try:
            fallback = await _load_latest_finalized_compute_snapshot_v2(
                epoch=int(epoch_id),
                netuid=int(netuid),
            )
        except ValueError as exc:
            raise ResearchLabV2AuthorityError(str(exc)) from exc
        if fallback is not None:
            _fallback_row, fallback_authority = fallback
            fallback_epoch = int(fallback_authority["epoch"])
            fallback_hash = str(fallback_authority["allocation_hash"])
            fallback_types = set(
                fallback_authority.get("authority_types") or ()
            )
            if "native_v2_finalization" in fallback_types:
                add_exact(
                    "allocation",
                    "epoch:%d" % fallback_epoch,
                    fallback_hash,
                )
                for receipt_hash in (
                    fallback_authority.get("finalization_receipt_hashes")
                    or ()
                ):
                    add_receipt_root(str(receipt_hash))
            if "legacy_finalized_chain_migration_v2" in fallback_types:
                add_receipt_root(
                    str(
                        fallback_authority.get(
                            "legacy_settlement_receipt_hash"
                        )
                        or ""
                    )
                )
            if not fallback_types.intersection(
                {
                    "native_v2_finalization",
                    "legacy_finalized_chain_migration_v2",
                }
            ):
                raise ResearchLabV2AuthorityError(
                    "historical compute fallback authority type is unsupported"
                )

    champion_statuses = (
        ("active", "queued", "partially_paid")
        if bool(policy.get("enable_champ_cap", True))
        else ("active", "queued", "partially_paid", "paid")
    )
    source_statuses = ("active", "queued", "partially_paid")
    champion_rows = []
    source_rows = []
    for status in champion_statuses:
        selected_rows = await select_all(
            "research_lab_champion_reward_current",
            filters=(
                ("current_reward_status", status),
                ("start_epoch", "lte", int(epoch_id)),
            ),
        )
        try:
            champion_rows.extend(
                row
                for row in selected_rows
                if champion_reward_requires_allocation_history_v2(
                    row,
                    epoch=int(epoch_id),
                    enable_champ_cap=bool(
                        policy.get("enable_champ_cap", True)
                    ),
                )
            )
        except ValueError as exc:
            raise ResearchLabV2AuthorityError(str(exc)) from exc
    for status in source_statuses:
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
    if (history_starts or prior_frontier is not None) and int(epoch_id) > 0:
        history_start = (
            int(prior_frontier["settled_through_epoch"]) + 1
            if prior_frontier is not None
            else min(history_starts)
        )
        for root, (source_epoch, graph) in preloaded_receipts.items():
            if history_start <= source_epoch < int(epoch_id):
                graphs[root] = graph
        if finalized_champion_history is None:
            loaded_receipt_records: dict[str, dict[str, Any]] = {}
            normalized_finalized_history = (
                await load_settled_allocation_history_v2(
                    netuid=int(netuid),
                    start_epoch=history_start,
                    end_epoch=int(epoch_id) - 1,
                    _receipt_graph_records_out=loaded_receipt_records,
                )
            )
            for root, record in loaded_receipt_records.items():
                add_preloaded_receipt_record(root, record)
                source_epoch, graph = preloaded_receipts[root]
                if history_start <= source_epoch < int(epoch_id):
                    graphs[root] = graph
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
    for row in champion_rows:
        add(
            "champion_reward_decision",
            str(row.get("champion_reward_id") or ""),
        )
    for row in normalized_finalized_history:
        authority_types = set(row.get("authority_types") or ())
        if "native_v2_finalization" in authority_types:
            add_exact(
                "allocation",
                "epoch:%d" % int(row.get("epoch") or 0),
                str(row.get("allocation_hash") or ""),
            )
            for receipt_hash in row.get("finalization_receipt_hashes") or ():
                add_receipt_root(str(receipt_hash))
        if "legacy_finalized_chain_migration_v2" in authority_types:
            add_receipt_root(
                str(row.get("legacy_settlement_receipt_hash") or "")
            )
        if CHAIN_REALIZED_AUTHORITY_TYPE_V1 in authority_types:
            add_receipt_root(
                str(row.get("chain_realized_settlement_receipt_hash") or "")
            )
            for receipt_hash in row.get("chain_realized_credit_receipt_hashes") or ():
                add_receipt_root(str(receipt_hash))
    for row in source_rows:
        reward_ref = str(row.get("reward_ref") or "")
        try:
            decision_hash = sha256_json(
                source_add_reward_row_projection_v2(
                    "source_add_leg%d" % int(row.get("leg") or 0),
                    {
                        **dict(row),
                        "initial_reward_status": "active",
                    },
                )
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ResearchLabV2AuthorityError(
                "allocation SOURCE_ADD reward identity is invalid"
            ) from exc
        add_exact("source_add_reward_decision", reward_ref, decision_hash)

    exact_items = sorted(
        (kind, ref, digest)
        for (kind, ref), digest in exact_artifact_refs.items()
    )
    loaded_exact_graphs = await load_business_artifact_graphs_v2(exact_items)
    for graph in loaded_exact_graphs.values():
        graphs[str(graph["root_receipt_hash"])] = graph

    loaded_business_graphs = await load_business_artifact_graphs_by_ref_v2(
        artifact_refs
    )
    for graph in loaded_business_graphs.values():
        graphs[str(graph["root_receipt_hash"])] = graph
    for receipt_hash in receipt_roots:
        preloaded_receipt = preloaded_receipts.get(receipt_hash)
        if preloaded_receipt is not None:
            graphs[receipt_hash] = preloaded_receipt[1]
    loaded_receipt_graphs = await load_receipt_graphs_v2(
        receipt_roots.difference(graphs)
    )
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
