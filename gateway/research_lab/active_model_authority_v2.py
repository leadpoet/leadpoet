"""V2 bridge binding private model execution to the active lineage row."""

from __future__ import annotations

from typing import Any, Mapping

from gateway.research_lab.attested_coordinator_v2 import execute_coordinator_v2
from gateway.research_lab.attested_v2_store import (
    load_business_artifact_graph_v2,
    persist_business_artifact_links_v2,
)
from gateway.research_lab.store import select_many
from gateway.tee.coordinator_executor_v2 import OP_ATTEST_ACTIVE_PRIVATE_MODEL
from leadpoet_canonical.attested_v2 import sha256_json, validate_receipt_graph
from research_lab.eval.artifacts import PrivateModelArtifactManifest


class ActivePrivateModelAuthorityV2Error(RuntimeError):
    """The host-selected model lacks one exact measured active-lineage receipt."""


async def attest_active_private_model_v2(
    *,
    artifact: PrivateModelArtifactManifest,
    epoch_id: int,
    execute: Any = execute_coordinator_v2,
    persist_links: Any = persist_business_artifact_links_v2,
) -> dict[str, Any]:
    rows = await select_many(
        "research_lab_private_model_version_current",
        columns=(
            "private_model_version_id,model_artifact_hash,private_model_manifest_hash,"
            "source_candidate_id,source_score_bundle_id,current_version_status"
        ),
        filters=(
            ("current_version_status", "active"),
            ("model_artifact_hash", artifact.model_artifact_hash),
        ),
        order_by=(("current_status_at", True),),
        limit=2,
    )
    if len(rows) != 1:
        raise ActivePrivateModelAuthorityV2Error(
            "active private model row is missing or ambiguous"
        )
    row = rows[0]
    if str(row.get("private_model_manifest_hash") or "") != artifact.manifest_hash:
        raise ActivePrivateModelAuthorityV2Error(
            "active private model manifest hash differs"
        )

    parent_graphs = ()
    score_bundle_id = str(row.get("source_score_bundle_id") or "")
    if score_bundle_id:
        if not score_bundle_id.startswith("score_bundle:"):
            raise ActivePrivateModelAuthorityV2Error(
                "active private model score bundle ID is invalid"
            )
        score_bundle_hash = "sha256:" + score_bundle_id.split(":", 1)[1]
        promotion_graph = await load_business_artifact_graph_v2(
            artifact_kind="promotion_decision",
            artifact_ref=score_bundle_id,
            artifact_hash=score_bundle_hash,
        )
        parent_graphs = (promotion_graph,)

    outcome = await execute(
        operation=OP_ATTEST_ACTIVE_PRIVATE_MODEL,
        purpose="research_lab.active_private_model.v2",
        epoch_id=max(0, int(epoch_id)),
        sequence=0,
        payload={"artifact": artifact.to_dict()},
        parent_graphs=parent_graphs,
        input_artifact_hashes=(
            artifact.model_artifact_hash,
            artifact.manifest_hash,
        ),
    )
    result = outcome.get("result")
    receipt = outcome.get("receipt") or outcome.get("execution_receipt")
    graph = outcome.get("receipt_graph")
    if (
        not isinstance(result, Mapping)
        or not isinstance(receipt, Mapping)
        or not isinstance(graph, Mapping)
        or result.get("schema_version") != "leadpoet.active_private_model.v2"
        or result.get("artifact") != artifact.to_dict()
        or str((result.get("active_model") or {}).get("private_model_version_id") or "")
        != str(row.get("private_model_version_id") or "")
        or receipt.get("output_root") != sha256_json(dict(result))
        or graph.get("root_receipt_hash") != receipt.get("receipt_hash")
    ):
        raise ActivePrivateModelAuthorityV2Error(
            "active private model measured result differs"
        )
    validate_receipt_graph(
        graph,
        required_purposes=("research_lab.active_private_model.v2",),
    )
    link = await persist_links(
        receipt_hash=str(receipt["receipt_hash"]),
        artifacts=(
            {
                "artifact_kind": "active_private_model",
                "artifact_ref": str(row["private_model_version_id"]),
                "artifact_hash": artifact.manifest_hash,
            },
        ),
    )
    return {
        **dict(outcome),
        "status": "matched",
        "artifact_link_status": dict(link),
    }
