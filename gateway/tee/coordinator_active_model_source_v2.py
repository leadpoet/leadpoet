"""Authenticated active-private-model lineage for measured model execution."""

from __future__ import annotations

from typing import Any, Callable, Dict, Mapping

from gateway.tee.execution_job_manager_v2 import ExecutionContextV2
from gateway.tee.supabase_source_v2 import SupabaseSourceReaderV2
from leadpoet_canonical.attested_v2 import (
    sha256_json,
    validate_signed_execution_receipt,
)
from research_lab.eval.artifacts import (
    PrivateModelArtifactManifest,
    validate_private_model_artifact_manifest,
)
from research_lab.eval.promotion_metric import promotion_gate_decision


class CoordinatorActiveModelSourceV2Error(RuntimeError):
    """The requested private model differs from authenticated active lineage."""


_DIRECT_RELEASE_SOURCES = frozenset(
    {
        "bootstrap_private_model_manifest_uri",
        "repo_head_sync",
        "operator_manifest_reregister",
    }
)


class CoordinatorActiveModelSourceV2:
    def __init__(
        self,
        *,
        reader: SupabaseSourceReaderV2,
        config_supplier: Callable[[], Any],
    ) -> None:
        self._reader = reader
        self._config_supplier = config_supplier

    def resolve(
        self,
        *,
        payload: Mapping[str, Any],
        context: ExecutionContextV2,
    ) -> Dict[str, Any]:
        if not isinstance(payload, Mapping) or set(payload) != {"artifact"}:
            raise CoordinatorActiveModelSourceV2Error(
                "active private model payload fields are invalid"
            )
        if context.purpose != "research_lab.active_private_model.v2":
            raise CoordinatorActiveModelSourceV2Error(
                "active private model purpose is invalid"
            )
        artifact_value = payload.get("artifact")
        if not isinstance(artifact_value, Mapping):
            raise CoordinatorActiveModelSourceV2Error(
                "active private model manifest is missing"
            )
        artifact = PrivateModelArtifactManifest.from_mapping(artifact_value)
        errors = validate_private_model_artifact_manifest(artifact)
        if errors:
            raise CoordinatorActiveModelSourceV2Error(
                "active private model manifest is invalid: %s" % "; ".join(errors)
            )

        rows = self._read("active_private_model_current", {}, context)
        if len(rows) != 1:
            raise CoordinatorActiveModelSourceV2Error(
                "active private model row is missing or ambiguous"
            )
        row = dict(rows[0])
        expected = {
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
        for field, value in expected.items():
            if str(row.get(field) or "") != str(value or ""):
                raise CoordinatorActiveModelSourceV2Error(
                    "active private model %s differs from its manifest" % field
                )
        if str(row.get("current_version_status") or "") != "active":
            raise CoordinatorActiveModelSourceV2Error(
                "private model lineage row is not active"
            )

        lineage_receipt_hash = ""
        lineage_kind = ""
        lineage_root = ""
        source_score_bundle_id = str(row.get("source_score_bundle_id") or "")
        source_candidate_id = str(row.get("source_candidate_id") or "")
        if source_score_bundle_id or source_candidate_id:
            if not source_score_bundle_id or not source_candidate_id:
                raise CoordinatorActiveModelSourceV2Error(
                    "promoted private model source lineage is incomplete"
                )
            lineage_receipt_hash = self._validate_promotion_lineage(
                score_bundle_id=source_score_bundle_id,
                context=context,
            )
            lineage_kind = "attested_promotion"
            lineage_root = lineage_receipt_hash
        elif context.external_receipt_graphs:
            raise CoordinatorActiveModelSourceV2Error(
                "bootstrap private model cannot declare promotion ancestry"
            )
        else:
            lineage_kind, lineage_root = self._direct_release_lineage(
                row=row,
                artifact=artifact,
            )

        active_model = {
            "private_model_version_id": str(row.get("private_model_version_id") or ""),
            **expected,
            "source_candidate_id": source_candidate_id,
            "source_score_bundle_id": source_score_bundle_id,
            "source_benchmark_bundle_id": str(
                row.get("source_benchmark_bundle_id") or ""
            ),
            "lineage_kind": lineage_kind,
            "lineage_root": lineage_root,
            "lineage_receipt_hash": lineage_receipt_hash,
        }
        source_state = {
            "active_model": active_model,
            "redacted_version_doc": dict(row.get("redacted_version_doc") or {}),
            "current_status_at": str(row.get("current_status_at") or ""),
        }
        return {
            "schema_version": "leadpoet.active_private_model.v2",
            "artifact": artifact.to_dict(),
            "active_model": active_model,
            "source_state_hash": sha256_json(source_state),
        }

    @staticmethod
    def _direct_release_lineage(
        *,
        row: Mapping[str, Any],
        artifact: PrivateModelArtifactManifest,
    ) -> tuple[str, str]:
        redacted = row.get("redacted_version_doc")
        if not isinstance(redacted, Mapping):
            raise CoordinatorActiveModelSourceV2Error(
                "direct private model release evidence is missing"
            )
        source = str(redacted.get("source") or "")
        if source not in _DIRECT_RELEASE_SOURCES:
            raise CoordinatorActiveModelSourceV2Error(
                "direct private model release source is not authorized"
            )
        expected = {
            "model_artifact_hash": artifact.model_artifact_hash,
            "private_model_manifest_hash": artifact.manifest_hash,
            "git_commit_sha": artifact.git_commit_sha,
            "component_registry_version": artifact.component_registry_version,
            "scoring_adapter_version": artifact.scoring_adapter_version,
        }
        for field, value in expected.items():
            if str(redacted.get(field) or "") != str(value or ""):
                raise CoordinatorActiveModelSourceV2Error(
                    "direct private model release %s differs" % field
                )
        if source == "repo_head_sync":
            repo_main_sha = str(redacted.get("repo_main_sha") or "")
            if repo_main_sha != artifact.git_commit_sha:
                raise CoordinatorActiveModelSourceV2Error(
                    "repo-head release commit differs from its artifact"
                )
            if str(redacted.get("current_json_manifest_uri") or "") != artifact.manifest_uri:
                raise CoordinatorActiveModelSourceV2Error(
                    "repo-head release manifest URI differs"
                )
        release_evidence = {
            "schema_version": "leadpoet.private_model_release_evidence.v2",
            "release_source": source,
            "private_model_version_id": str(
                row.get("private_model_version_id") or ""
            ),
            "artifact": artifact.to_dict(),
            "redacted_release_evidence": dict(redacted),
            "current_status_at": str(row.get("current_status_at") or ""),
        }
        return "attested_%s" % source, sha256_json(release_evidence)

    def _validate_promotion_lineage(
        self,
        *,
        score_bundle_id: str,
        context: ExecutionContextV2,
    ) -> str:
        if not score_bundle_id.startswith("score_bundle:"):
            raise CoordinatorActiveModelSourceV2Error(
                "private model score bundle ID is invalid"
            )
        score_bundle_hash = "sha256:" + score_bundle_id.split(":", 1)[1]
        bundle_rows = self._read(
            "score_bundle_by_id",
            {"score_bundle_id": score_bundle_id},
            context,
        )
        if len(bundle_rows) != 1:
            raise CoordinatorActiveModelSourceV2Error(
                "private model score bundle is missing or ambiguous"
            )
        bundle_row = bundle_rows[0]
        score_bundle = bundle_row.get("score_bundle_doc")
        if (
            str(bundle_row.get("score_bundle_hash") or "") != score_bundle_hash
            or not isinstance(score_bundle, Mapping)
            or str(score_bundle.get("score_bundle_hash") or "") != score_bundle_hash
        ):
            raise CoordinatorActiveModelSourceV2Error(
                "private model score bundle hash differs"
            )

        links = self._read(
            "attested_business_artifact_by_ref",
            {
                "artifact_kind": "promotion_decision",
                "artifact_ref": score_bundle_id,
            },
            context,
        )
        if len(links) != 1 or str(links[0].get("artifact_hash") or "") != score_bundle_hash:
            raise CoordinatorActiveModelSourceV2Error(
                "promotion receipt link is missing or ambiguous"
            )
        receipt_hash = str(links[0].get("receipt_hash") or "")
        receipt_rows = self._read(
            "attested_receipt_by_hash",
            {"receipt_hash": receipt_hash},
            context,
        )
        if len(receipt_rows) != 1 or not isinstance(
            receipt_rows[0].get("receipt_doc"), Mapping
        ):
            raise CoordinatorActiveModelSourceV2Error(
                "promotion receipt is not persisted"
            )
        receipt = dict(receipt_rows[0]["receipt_doc"])
        validate_signed_execution_receipt(receipt)
        config = self._config_supplier()
        decision = promotion_gate_decision(
            score_bundle,
            candidate_kind="image_build",
            candidate_parent=str(score_bundle.get("parent_artifact_hash") or ""),
            active_parent=str(score_bundle.get("parent_artifact_hash") or ""),
            threshold_points=float(config.improvement_threshold_points),
            auto_promotion_enabled=True,
        ).to_dict()
        graph_roots = {
            str(graph.get("root_receipt_hash") or ""): graph
            for graph in context.external_receipt_graphs
        }
        if (
            decision.get("status") != "promotion_passed"
            or receipt.get("receipt_hash") != receipt_hash
            or receipt.get("role") != "gateway_coordinator"
            or receipt.get("purpose") != "research_lab.promotion_decision.v2"
            or receipt.get("output_root") != sha256_json({"decision": decision})
            or receipt_hash not in context.parent_receipt_hashes
            or receipt_hash not in graph_roots
        ):
            raise CoordinatorActiveModelSourceV2Error(
                "active private model promotion ancestry is invalid"
            )
        root_receipts = {
            str(item.get("receipt_hash") or ""): item
            for item in graph_roots[receipt_hash].get("receipts") or ()
            if isinstance(item, Mapping)
        }
        if root_receipts.get(receipt_hash) != receipt:
            raise CoordinatorActiveModelSourceV2Error(
                "promotion receipt graph differs from durable receipt"
            )
        return receipt_hash

    def _read(
        self,
        policy_id: str,
        parameters: Mapping[str, Any],
        context: ExecutionContextV2,
    ) -> list[Dict[str, Any]]:
        return self._reader.read(
            policy_id=policy_id,
            parameters=parameters,
            job_id=context.job_id,
            purpose=context.purpose,
            record_transport=context.record_transport,
            record_artifact=context.record_artifact,
        )
