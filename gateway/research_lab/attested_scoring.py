"""V2-only attested scoring facade for existing Research Lab call sites.

The public function names are retained so scoring, rebenchmark, promotion, and
allocation workflow code does not change.  There is no V1, off, shadow, replay,
or host-authoritative branch in this module.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping


class AttestedScoringError(RuntimeError):
    """An authoritative V2 result or its complete ancestry is unavailable."""


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def attested_scoring_mode() -> str:
    """Compatibility accessor; V2 authority is unconditionally required."""

    return "required"


def attested_receipt_persistence_enabled() -> bool:
    return True


def attested_live_provider_enabled() -> bool:
    return True


def scoring_enclave_shard_for_worker(worker_index: int) -> int:
    from gateway.research_lab.attested_scoring_v2 import (
        AttestedScoringV2Error,
        scoring_enclave_shard_for_worker as shard,
    )

    try:
        return shard(worker_index)
    except AttestedScoringV2Error as exc:
        raise AttestedScoringError(str(exc)) from exc


async def execute_attested_scoring_operation(**_kwargs: Any) -> dict[str, Any]:
    raise AttestedScoringError(
        "legacy attested scoring RPC is removed; use an exact V2 authority adapter"
    )


async def compare_qualification_company_scores(**_kwargs: Any) -> dict[str, Any]:
    raise AttestedScoringError(
        "provider-tape comparison is not an authority path in V2"
    )


async def execute_required_qualification_company_scores(
    *,
    epoch_id: int,
    purpose: str,
    companies: list[Mapping[str, Any]],
    icp: Mapping[str, Any],
    is_reference_model: bool,
    provider_credential_profile: str = "default",
    attestation_out: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    from gateway.research_lab.v2_authority import execute_company_scores_v2

    try:
        return await execute_company_scores_v2(
            epoch_id=int(epoch_id),
            purpose=purpose,
            companies=companies,
            icp=icp,
            is_reference_model=bool(is_reference_model),
            provider_credential_profile=provider_credential_profile,
            attestation_out=attestation_out,
        )
    except Exception as exc:
        if isinstance(exc, AttestedScoringError):
            raise
        raise AttestedScoringError("required V2 company scoring failed") from exc


async def compare_score_bundle(
    *,
    epoch_id: int,
    purpose: str,
    build_payload: Mapping[str, Any],
    expected_score_bundle: Mapping[str, Any],
    evidence_roots: Mapping[str, Any] | None = None,
    parent_receipts: list[Mapping[str, Any]] | None = None,
    direct_parent_receipt_hashes: list[str] | None = None,
) -> dict[str, Any]:
    del evidence_roots
    from gateway.research_lab.v2_authority import compare_score_bundle_v2

    roots = (
        list(direct_parent_receipt_hashes)
        if direct_parent_receipt_hashes is not None
        else [
            str(item.get("receipt_hash") or "")
            for item in (parent_receipts or [])
            if isinstance(item, Mapping)
        ]
    )
    try:
        return await compare_score_bundle_v2(
            epoch_id=int(epoch_id),
            purpose=purpose,
            build_payload=build_payload,
            expected_score_bundle=expected_score_bundle,
            parent_receipt_hashes=roots,
        )
    except Exception as exc:
        raise AttestedScoringError("required V2 score-bundle authority failed") from exc


async def compare_baseline_score_summary(
    *,
    epoch_id: int,
    build_payload: Mapping[str, Any],
    expected_result: Mapping[str, Any],
    parent_receipts: list[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    from gateway.research_lab.v2_authority import compare_baseline_summary_v2

    try:
        return await compare_baseline_summary_v2(
            epoch_id=int(epoch_id),
            build_payload=build_payload,
            expected_result=expected_result,
            parent_receipt_hashes=[
                str(item.get("receipt_hash") or "")
                for item in (parent_receipts or [])
                if isinstance(item, Mapping)
            ],
        )
    except Exception as exc:
        raise AttestedScoringError("required V2 baseline authority failed") from exc


async def compare_promotion_metric(
    *,
    epoch_id: int,
    score_bundle: Mapping[str, Any],
    expected_improvement_points: float,
    expected_event_doc: Mapping[str, Any],
    parent_receipt_hashes: list[str] | None = None,
) -> dict[str, Any]:
    from gateway.research_lab.v2_authority import compare_promotion_metric_v2

    try:
        return await compare_promotion_metric_v2(
            epoch_id=int(epoch_id),
            score_bundle=score_bundle,
            expected_improvement_points=float(expected_improvement_points),
            expected_event_doc=expected_event_doc,
            parent_receipt_hashes=parent_receipt_hashes or (),
        )
    except Exception as exc:
        raise AttestedScoringError("required V2 promotion metric authority failed") from exc


async def compare_promotion_gate_decision(
    *,
    epoch_id: int,
    score_bundle: Mapping[str, Any],
    decision_payload: Mapping[str, Any],
    expected_decision: Mapping[str, Any],
    metric_outcome: Mapping[str, Any],
) -> dict[str, Any]:
    from gateway.research_lab.v2_authority import compare_promotion_decision_v2

    try:
        return await compare_promotion_decision_v2(
            epoch_id=int(epoch_id),
            score_bundle=score_bundle,
            decision_payload=decision_payload,
            expected_decision=expected_decision,
            metric_outcome=metric_outcome,
        )
    except Exception as exc:
        raise AttestedScoringError("required V2 promotion decision authority failed") from exc


async def compare_allocation(
    *,
    epoch_id: int,
    netuid: int,
    payload: Mapping[str, Any],
    expected_allocation: Mapping[str, Any],
) -> dict[str, Any]:
    from gateway.research_lab.v2_authority import compare_allocation_v2

    try:
        return await compare_allocation_v2(
            epoch_id=int(epoch_id),
            netuid=int(netuid),
            payload=payload,
            expected_allocation=expected_allocation,
        )
    except Exception as exc:
        raise AttestedScoringError("required V2 allocation authority failed") from exc


async def resolve_attested_artifact_lineage(
    *,
    artifact_kind: str,
    artifact_ref: str,
    artifact_hash: str | None = None,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    if artifact_hash is None:
        raise AttestedScoringError("V2 artifact lineage requires an exact hash")
    try:
        from gateway.research_lab.attested_v2_store import (
            load_business_artifact_graph_v2,
        )

        graph = await load_business_artifact_graph_v2(
            artifact_kind=artifact_kind,
            artifact_ref=artifact_ref,
            artifact_hash=artifact_hash,
        )
        receipts = [dict(item) for item in graph["receipts"]]
        root = next(
            item
            for item in receipts
            if item.get("receipt_hash") == graph["root_receipt_hash"]
        )
        return root, receipts
    except Exception as exc:
        raise AttestedScoringError("required V2 artifact lineage failed") from exc


async def persist_attested_outcome_artifact_links(
    outcome: Mapping[str, Any],
    *,
    artifact_links: list[Mapping[str, Any]],
) -> str:
    if outcome.get("status") not in {"succeeded", "matched"}:
        raise AttestedScoringError("required V2 outcome is unavailable for linking")
    receipt = outcome.get("execution_receipt") or outcome.get("receipt")
    if not isinstance(receipt, Mapping):
        raise AttestedScoringError("required V2 outcome receipt is missing")
    try:
        from gateway.research_lab.attested_v2_store import (
            persist_business_artifact_links_v2,
        )

        await persist_business_artifact_links_v2(
            receipt_hash=str(receipt["receipt_hash"]),
            artifacts=artifact_links,
        )
    except Exception as exc:
        raise AttestedScoringError("required V2 artifact linking failed") from exc
    return "persisted"
