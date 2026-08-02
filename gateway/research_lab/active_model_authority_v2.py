"""V2 bridge binding private model execution to the active lineage row."""

from __future__ import annotations

import re
from typing import Any, Mapping

from gateway.research_lab.attested_coordinator_v2 import execute_coordinator_v2
from gateway.research_lab.attested_scoring_v2 import (
    DEFAULT_RELEASE_MANIFEST_PATH,
    _load_release,
)
from gateway.research_lab.attested_v2_store import (
    BUSINESS_ARTIFACT_TABLE,
    AttestedV2StoreError,
    load_business_artifact_graph_v2,
    load_execution_result_by_receipt_v2,
    persist_business_artifact_links_v2,
)
from gateway.research_lab.store import select_many
from gateway.tee.coordinator_executor_v2 import OP_ATTEST_ACTIVE_PRIVATE_MODEL
from gateway.tee.coordinator_active_model_source_v2 import (
    CoordinatorActiveModelSourceV2,
    CoordinatorActiveModelSourceV2Error,
)
from gateway.tee.release_manifest_v2 import role_expectation
from leadpoet_canonical.attested_v2 import (
    sha256_json,
    validate_receipt_graph,
    verify_boot_identity_nitro,
)
from research_lab.eval.artifacts import (
    PrivateModelArtifactManifest,
    private_model_artifact_replay_identity_v2,
)


class ActivePrivateModelAuthorityV2Error(RuntimeError):
    """The host-selected model lacks one exact measured active-lineage receipt."""


_HASH_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_ASSERTION_KIND = "active_private_model_assertion_v2"
_PURPOSE = "research_lab.active_private_model.v2"
_ACTIVE_MODEL_COLUMNS = (
    "private_model_version_id,model_artifact_hash,private_model_manifest_hash,"
    "private_model_manifest_uri,git_commit_sha,config_hash,"
    "component_registry_version,scoring_adapter_version,signature_ref,build_id,"
    "source_candidate_id,source_score_bundle_id,source_benchmark_bundle_id,"
    "redacted_version_doc,current_version_status,current_status_at"
)


async def _select_active_model_rows_v2(
    artifact: PrivateModelArtifactManifest,
) -> list[dict[str, Any]]:
    return await select_many(
        "research_lab_private_model_version_current",
        columns=_ACTIVE_MODEL_COLUMNS,
        filters=(
            ("current_version_status", "active"),
            ("model_artifact_hash", artifact.model_artifact_hash),
        ),
        order_by=(("current_status_at", True),),
        limit=2,
    )


def _assertion_ref_v2(
    *,
    artifact: PrivateModelArtifactManifest,
    row: Mapping[str, Any],
    epoch_id: int,
    release_hash: str,
) -> str:
    current_status_at = str(row.get("current_status_at") or "").strip()
    redacted_version_doc = row.get("redacted_version_doc")
    if not current_status_at or not isinstance(redacted_version_doc, Mapping):
        raise ActivePrivateModelAuthorityV2Error(
            "active private model state identity is incomplete"
        )
    state_hash = sha256_json(
        {
            "schema_version": "leadpoet.active_private_model_assertion_state.v2",
            "private_model_version_id": str(
                row.get("private_model_version_id") or ""
            ),
            "artifact": private_model_artifact_replay_identity_v2(artifact),
            "source_candidate_id": str(row.get("source_candidate_id") or ""),
            "source_score_bundle_id": str(
                row.get("source_score_bundle_id") or ""
            ),
            "source_benchmark_bundle_id": str(
                row.get("source_benchmark_bundle_id") or ""
            ),
            "redacted_version_doc": dict(redacted_version_doc),
            "current_status_at": current_status_at,
        }
    )
    return (
        f"{row['private_model_version_id']}:epoch:{epoch_id}:"
        f"release:{release_hash}:state:{state_hash}"
    )


def _expected_active_model_authority_v2(
    *,
    artifact: PrivateModelArtifactManifest,
    row: Mapping[str, Any],
    promotion_graph: Mapping[str, Any] | None,
) -> tuple[dict[str, str], tuple[str, ...]]:
    expected_artifact_fields = {
        "model_artifact_hash": artifact.model_artifact_hash,
        "private_model_manifest_hash": artifact.manifest_hash,
        "private_model_manifest_uri": artifact.manifest_uri,
        "git_commit_sha": artifact.git_commit_sha,
        "config_hash": artifact.config_hash,
        "component_registry_version": artifact.component_registry_version,
        "scoring_adapter_version": artifact.scoring_adapter_version,
        "signature_ref": artifact.signature_ref,
        "build_id": artifact.build_id,
    }
    if str(row.get("current_version_status") or "") != "active" or any(
        str(row.get(field) or "") != str(value or "")
        for field, value in expected_artifact_fields.items()
    ):
        raise ActivePrivateModelAuthorityV2Error(
            "active private model row differs from its artifact"
        )
    source_candidate_id = str(row.get("source_candidate_id") or "")
    source_score_bundle_id = str(row.get("source_score_bundle_id") or "")
    if bool(source_candidate_id) != bool(source_score_bundle_id):
        raise ActivePrivateModelAuthorityV2Error(
            "active private model promotion lineage is incomplete"
        )

    if source_score_bundle_id:
        if not isinstance(promotion_graph, Mapping):
            raise ActivePrivateModelAuthorityV2Error(
                "active private model promotion graph is missing"
            )
        promotion_root = str(
            promotion_graph.get("root_receipt_hash") or ""
        ).lower()
        if not _HASH_RE.fullmatch(promotion_root):
            raise ActivePrivateModelAuthorityV2Error(
                "active private model promotion graph root is invalid"
            )
        lineage_kind = "attested_promotion"
        lineage_root = promotion_root
        lineage_receipt_hash = promotion_root
        expected_parent_receipt_hashes = (promotion_root,)
    else:
        if promotion_graph is not None:
            raise ActivePrivateModelAuthorityV2Error(
                "direct active private model has unexpected promotion ancestry"
            )
        try:
            lineage_kind, lineage_root = (
                CoordinatorActiveModelSourceV2._direct_release_lineage(
                    row=row,
                    artifact=artifact,
                )
            )
        except CoordinatorActiveModelSourceV2Error as exc:
            raise ActivePrivateModelAuthorityV2Error(str(exc)) from exc
        lineage_receipt_hash = ""
        expected_parent_receipt_hashes = ()

    expected_active = {
        "private_model_version_id": str(
            row.get("private_model_version_id") or ""
        ),
        **expected_artifact_fields,
        "source_candidate_id": source_candidate_id,
        "source_score_bundle_id": source_score_bundle_id,
        "source_benchmark_bundle_id": str(
            row.get("source_benchmark_bundle_id") or ""
        ),
        "lineage_kind": lineage_kind,
        "lineage_root": lineage_root,
        "lineage_receipt_hash": lineage_receipt_hash,
    }
    return expected_active, expected_parent_receipt_hashes


def _validate_active_model_result_v2(
    *,
    artifact: PrivateModelArtifactManifest,
    row: Mapping[str, Any],
    result: Mapping[str, Any],
    expected_active_model: Mapping[str, str],
) -> None:
    active_model = result.get("active_model")
    redacted_version_doc = row.get("redacted_version_doc")
    if (
        set(result)
        != {
            "schema_version",
            "artifact",
            "active_model",
            "source_state_hash",
        }
        or result.get("schema_version") != "leadpoet.active_private_model.v2"
        or result.get("artifact")
        != private_model_artifact_replay_identity_v2(artifact)
        or not isinstance(active_model, Mapping)
        or set(active_model) != set(expected_active_model)
        or any(
            active_model.get(field) != value
            for field, value in expected_active_model.items()
        )
        or not isinstance(redacted_version_doc, Mapping)
    ):
        raise ActivePrivateModelAuthorityV2Error(
            "active private model measured result differs"
        )
    source_state = {
        "active_model": dict(active_model),
        "redacted_version_doc": dict(redacted_version_doc),
        "current_status_at": str(row.get("current_status_at") or ""),
    }
    if result.get("source_state_hash") != sha256_json(source_state):
        raise ActivePrivateModelAuthorityV2Error(
            "active private model source state differs"
        )


def _validate_assertion_authority_v2(
    *,
    artifact: PrivateModelArtifactManifest,
    row: Mapping[str, Any],
    epoch_id: int,
    release: Mapping[str, Any],
    result: Mapping[str, Any],
    receipt: Mapping[str, Any],
    graph: Mapping[str, Any],
    replay_row: Mapping[str, Any] | None = None,
    replay_graph: Mapping[str, Any] | None = None,
    expected_artifact_hash: str | None = None,
    expected_active_model: Mapping[str, str],
    expected_parent_receipt_hashes: tuple[str, ...],
) -> str:
    result_hash = sha256_json(dict(result))
    release_hash = str(release.get("release_hash") or "").lower()
    if (
        not _HASH_RE.fullmatch(release_hash)
        or (expected_artifact_hash is not None and result_hash != expected_artifact_hash)
    ):
        raise ActivePrivateModelAuthorityV2Error(
            "active private model assertion identity differs"
        )
    _validate_active_model_result_v2(
        artifact=artifact,
        row=row,
        result=result,
        expected_active_model=expected_active_model,
    )
    receipt_parent_hashes = receipt.get("parent_receipt_hashes")
    if (
        receipt.get("role") != "gateway_coordinator"
        or receipt.get("purpose") != _PURPOSE
        or receipt.get("epoch_id") != epoch_id
        or receipt.get("sequence") != 0
        or receipt.get("status") != "succeeded"
        or receipt.get("output_root") != result_hash
        or not isinstance(receipt_parent_hashes, (list, tuple))
        or tuple(sorted(str(value) for value in receipt_parent_hashes))
        != expected_parent_receipt_hashes
        or graph.get("root_receipt_hash") != receipt.get("receipt_hash")
    ):
        raise ActivePrivateModelAuthorityV2Error(
            "active private model receipt authority differs"
        )
    validate_receipt_graph(graph, required_purposes=(_PURPOSE,))

    if replay_row is not None:
        expected_artifacts = sorted(
            {
                artifact.model_artifact_hash,
                artifact.manifest_hash,
                str(result.get("source_state_hash") or ""),
            }
        )
        if (
            replay_row.get("role") != "gateway_coordinator"
            or replay_row.get("operation") != OP_ATTEST_ACTIVE_PRIVATE_MODEL
            or replay_row.get("purpose") != _PURPOSE
            or replay_row.get("epoch_id") != epoch_id
            or replay_row.get("sequence") != 0
            or replay_row.get("release_hash") != release_hash
            or replay_row.get("result_hash") != result_hash
            or replay_row.get("output_root") != result_hash
            or replay_row.get("artifact_hashes") != expected_artifacts
            or not isinstance(replay_graph, Mapping)
            or sha256_json(dict(replay_graph)) != sha256_json(dict(graph))
        ):
            raise ActivePrivateModelAuthorityV2Error(
                "stored active private model assertion differs"
            )
        root_boot_hash = str(receipt.get("boot_identity_hash") or "")
        root_boots = {
            str(item.get("boot_identity_hash") or ""): item
            for item in graph.get("boot_identities") or ()
            if isinstance(item, Mapping)
        }
        root_boot = root_boots.get(root_boot_hash)
        expectation = role_expectation(release, "gateway_coordinator")
        if (
            not isinstance(root_boot, Mapping)
            or any(
                root_boot.get(field) != expectation[field]
                for field in (
                    "physical_role",
                    "commit_sha",
                    "pcr0",
                    "build_manifest_hash",
                    "dependency_lock_hash",
                )
            )
        ):
            raise ActivePrivateModelAuthorityV2Error(
                "stored active private model release identity differs"
            )
        verify_boot_identity_nitro(
            root_boot,
            expected_pcr0=expectation["pcr0"],
            certificate_validity_at_attestation_time=True,
        )
    return result_hash


async def _load_existing_assertion_v2(
    *,
    artifact: PrivateModelArtifactManifest,
    row: Mapping[str, Any],
    epoch_id: int,
    release: Mapping[str, Any],
    assertion_ref: str,
    expected_active_model: Mapping[str, str],
    expected_parent_receipt_hashes: tuple[str, ...],
    promotion_graph: Mapping[str, Any] | None,
    expected_artifact_hash: str | None = None,
) -> dict[str, Any] | None:
    links = await select_many(
        BUSINESS_ARTIFACT_TABLE,
        columns="receipt_hash,artifact_kind,artifact_ref,artifact_hash",
        filters=(
            ("artifact_kind", _ASSERTION_KIND),
            ("artifact_ref", assertion_ref),
        ),
        order_by=(("artifact_hash", False),),
        limit=2,
    )
    if not links:
        return None
    if len(links) != 1:
        raise ActivePrivateModelAuthorityV2Error(
            "active private model assertion is ambiguous"
        )
    link = links[0]
    artifact_hash = str(link.get("artifact_hash") or "").lower()
    receipt_hash = str(link.get("receipt_hash") or "").lower()
    if (
        link.get("artifact_kind") != _ASSERTION_KIND
        or link.get("artifact_ref") != assertion_ref
        or not _HASH_RE.fullmatch(artifact_hash)
        or not _HASH_RE.fullmatch(receipt_hash)
        or (
            expected_artifact_hash is not None
            and artifact_hash != expected_artifact_hash
        )
    ):
        raise ActivePrivateModelAuthorityV2Error(
            "active private model assertion link differs"
        )
    graph = await load_business_artifact_graph_v2(
        artifact_kind=_ASSERTION_KIND,
        artifact_ref=assertion_ref,
        artifact_hash=artifact_hash,
    )
    replay = await load_execution_result_by_receipt_v2(
        receipt_hash,
        expected_operation=OP_ATTEST_ACTIVE_PRIVATE_MODEL,
        expected_purpose=_PURPOSE,
    )
    replay_result = replay.get("result")
    replay_receipt = replay.get("receipt")
    replay_graph = replay.get("receipt_graph")
    replay_row = replay.get("row")
    if (
        not isinstance(replay_result, Mapping)
        or not isinstance(replay_receipt, Mapping)
        or not isinstance(replay_graph, Mapping)
        or not isinstance(replay_row, Mapping)
        or replay_receipt.get("receipt_hash") != receipt_hash
    ):
        raise ActivePrivateModelAuthorityV2Error(
            "active private model assertion replay is incomplete"
        )
    _validate_assertion_authority_v2(
        artifact=artifact,
        row=row,
        epoch_id=epoch_id,
        release=release,
        result=replay_result,
        receipt=replay_receipt,
        graph=graph,
        replay_row=replay_row,
        replay_graph=replay_graph,
        expected_artifact_hash=artifact_hash,
        expected_active_model=expected_active_model,
        expected_parent_receipt_hashes=expected_parent_receipt_hashes,
    )
    current_rows = await _select_active_model_rows_v2(artifact)
    current_expected_active: Mapping[str, str] | None = None
    current_expected_parents: tuple[str, ...] | None = None
    if len(current_rows) == 1:
        current_expected_active, current_expected_parents = (
            _expected_active_model_authority_v2(
                artifact=artifact,
                row=current_rows[0],
                promotion_graph=promotion_graph,
            )
        )
    if (
        len(current_rows) != 1
        or current_rows[0].get("current_version_status") != "active"
        or current_expected_active != expected_active_model
        or current_expected_parents != expected_parent_receipt_hashes
        or _assertion_ref_v2(
            artifact=artifact,
            row=current_rows[0],
            epoch_id=epoch_id,
            release_hash=str(release["release_hash"]),
        )
        != assertion_ref
    ):
        raise ActivePrivateModelAuthorityV2Error(
            "active private model changed during assertion replay"
        )
    normalized_link = {
        "receipt_hash": receipt_hash,
        "artifact_kind": _ASSERTION_KIND,
        "artifact_ref": assertion_ref,
        "artifact_hash": artifact_hash,
    }
    return {
        "status": "matched",
        "result": dict(replay_result),
        "receipt": dict(replay_receipt),
        "execution_receipt": dict(replay_receipt),
        "receipt_graph": dict(graph),
        "execution_receipt_graph": dict(graph),
        "artifact_hashes": list(replay_row["artifact_hashes"]),
        "release_hash": str(release["release_hash"]),
        "replay_status": "business_artifact_exact",
        "artifact_link_status": {
            "business_artifact_link_count": 1,
            "business_artifact_link_set_hash": sha256_json([normalized_link]),
        },
    }


async def attest_active_private_model_v2(
    *,
    artifact: PrivateModelArtifactManifest,
    epoch_id: int,
    execute: Any = execute_coordinator_v2,
    persist_links: Any = persist_business_artifact_links_v2,
) -> dict[str, Any]:
    assertion_epoch = max(0, int(epoch_id))
    release = _load_release(DEFAULT_RELEASE_MANIFEST_PATH)
    release_hash = str(release.get("release_hash") or "").lower()
    if not _HASH_RE.fullmatch(release_hash):
        raise ActivePrivateModelAuthorityV2Error(
            "active private model release identity is invalid"
        )
    rows = await _select_active_model_rows_v2(artifact)
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
    promotion_graph: Mapping[str, Any] | None = None
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

    expected_active_model, expected_parent_receipt_hashes = (
        _expected_active_model_authority_v2(
            artifact=artifact,
            row=row,
            promotion_graph=promotion_graph,
        )
    )

    assertion_ref = _assertion_ref_v2(
        artifact=artifact,
        row=row,
        epoch_id=assertion_epoch,
        release_hash=release_hash,
    )
    existing = await _load_existing_assertion_v2(
        artifact=artifact,
        row=row,
        epoch_id=assertion_epoch,
        release=release,
        assertion_ref=assertion_ref,
        expected_active_model=expected_active_model,
        expected_parent_receipt_hashes=expected_parent_receipt_hashes,
        promotion_graph=promotion_graph,
    )
    if existing is not None:
        return existing

    outcome = await execute(
        operation=OP_ATTEST_ACTIVE_PRIVATE_MODEL,
        purpose=_PURPOSE,
        epoch_id=assertion_epoch,
        sequence=0,
        payload={"artifact": artifact.to_dict()},
        parent_graphs=parent_graphs,
        input_artifact_hashes=(
            artifact.model_artifact_hash,
            artifact.manifest_hash,
        ),
        release_manifest=release,
    )
    result = outcome.get("result")
    receipt = outcome.get("execution_receipt") or outcome.get("receipt")
    graph = outcome.get("execution_receipt_graph") or outcome.get("receipt_graph")
    if (
        not isinstance(result, Mapping)
        or not isinstance(receipt, Mapping)
        or not isinstance(graph, Mapping)
        or outcome.get("release_hash") != release_hash
    ):
        raise ActivePrivateModelAuthorityV2Error(
            "active private model measured result differs"
        )
    result_hash = _validate_assertion_authority_v2(
        artifact=artifact,
        row=row,
        epoch_id=assertion_epoch,
        release=release,
        result=result,
        receipt=receipt,
        graph=graph,
        expected_active_model=expected_active_model,
        expected_parent_receipt_hashes=expected_parent_receipt_hashes,
    )
    try:
        link = await persist_links(
            receipt_hash=str(receipt["receipt_hash"]),
            artifacts=(
                {
                    "artifact_kind": _ASSERTION_KIND,
                    "artifact_ref": assertion_ref,
                    "artifact_hash": result_hash,
                },
            ),
        )
    except AttestedV2StoreError as exc:
        if "stored row conflicts at receipt_hash" not in str(exc):
            raise
        winner = await _load_existing_assertion_v2(
            artifact=artifact,
            row=row,
            epoch_id=assertion_epoch,
            release=release,
            assertion_ref=assertion_ref,
            expected_active_model=expected_active_model,
            expected_parent_receipt_hashes=expected_parent_receipt_hashes,
            promotion_graph=promotion_graph,
            expected_artifact_hash=result_hash,
        )
        if winner is None:
            raise ActivePrivateModelAuthorityV2Error(
                "concurrent active private model assertion is unavailable"
            ) from exc
        return winner
    return {
        **dict(outcome),
        "status": "matched",
        "artifact_link_status": dict(link),
    }
