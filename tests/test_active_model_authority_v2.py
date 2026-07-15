from __future__ import annotations

import base64
from types import SimpleNamespace

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from gateway.tee.coordinator_active_model_source_v2 import (
    CoordinatorActiveModelSourceV2,
    CoordinatorActiveModelSourceV2Error,
)
from gateway.tee.execution_job_manager_v2 import ExecutionContextV2
from leadpoet_canonical.attested_v2 import (
    EMPTY_ARTIFACT_ROOT,
    EMPTY_HOST_OPERATION_ROOT,
    EMPTY_TRANSPORT_ROOT,
    build_boot_identity_body,
    build_execution_receipt_body,
    build_receipt_graph,
    create_boot_identity,
    create_signed_execution_receipt,
    sha256_json,
)
from research_lab.eval import build_local_private_artifact_manifest
from research_lab.eval.artifacts import PrivateModelArtifactManifest
from research_lab.eval.promotion_metric import promotion_gate_decision


class _Reader:
    def __init__(self, rows):
        self.rows = rows
        self.calls = []

    def read(self, *, policy_id, parameters, **kwargs):
        self.calls.append((policy_id, dict(parameters)))
        return [dict(item) for item in self.rows.get(policy_id, [])]


def _artifact(tmp_path) -> PrivateModelArtifactManifest:
    source = tmp_path / "source"
    source.mkdir()
    (source / "research_lab_adapter.py").write_text(
        "def run():\n    return 1\n",
        encoding="utf-8",
    )
    return PrivateModelArtifactManifest.from_mapping(
        build_local_private_artifact_manifest(
            source_path=source,
            git_commit_sha="a" * 40,
            image_digest=(
                "123456789012.dkr.ecr.us-east-1.amazonaws.com/private@sha256:"
                + "b" * 64
            ),
            manifest_uri="s3://private/manifests/model.json",
            signature_ref="kms:signature",
            component_registry_version="1",
            scoring_adapter_version="1",
        )
    )


def _active_row(artifact, **overrides):
    row = {
        "private_model_version_id": "private_model_version:" + "1" * 64,
        "model_artifact_hash": artifact.model_artifact_hash,
        "private_model_manifest_hash": artifact.manifest_hash,
        "private_model_manifest_uri": artifact.manifest_uri,
        "git_commit_sha": artifact.git_commit_sha,
        "config_hash": artifact.config_hash,
        "component_registry_version": artifact.component_registry_version,
        "scoring_adapter_version": artifact.scoring_adapter_version,
        "source_candidate_id": None,
        "source_score_bundle_id": None,
        "source_benchmark_bundle_id": None,
        "signature_ref": artifact.signature_ref,
        "build_id": artifact.build_id,
        "redacted_version_doc": {
            "source": "bootstrap_private_model_manifest_uri",
            "model_artifact_hash": artifact.model_artifact_hash,
            "private_model_manifest_hash": artifact.manifest_hash,
            "git_commit_sha": artifact.git_commit_sha,
            "component_registry_version": artifact.component_registry_version,
            "scoring_adapter_version": artifact.scoring_adapter_version,
        },
        "current_version_status": "active",
        "current_status_at": "2026-07-12T00:00:00Z",
    }
    row.update(overrides)
    return row


def _context(*, graph=None):
    root = str((graph or {}).get("root_receipt_hash") or "")
    return ExecutionContextV2(
        job_id="active-model-job",
        purpose="research_lab.active_private_model.v2",
        epoch_id=42,
        parent_receipt_hashes=((root,) if root else ()),
        external_receipt_graphs=([graph] if graph else []),
    )


def _promotion_graph(decision):
    key = Ed25519PrivateKey.generate()
    public_key = key.public_key().public_bytes_raw().hex()
    boot = create_boot_identity(
        body=build_boot_identity_body(
            role="gateway_coordinator",
            physical_role="gateway_coordinator",
            commit_sha="a" * 40,
            pcr0="b" * 96,
            build_manifest_hash="sha256:" + "c" * 64,
            dependency_lock_hash="sha256:" + "d" * 64,
            config_hash="sha256:" + "e" * 64,
            boot_nonce="1" * 32,
            signing_pubkey=public_key,
            transport_pubkey="2" * 64,
            transport_certificate_hash="sha256:" + "3" * 64,
            attestation_user_data_hash="sha256:" + "4" * 64,
            issued_at="2026-07-12T00:00:00Z",
        ),
        attestation_document_b64=base64.b64encode(b"attestation").decode(),
    )
    receipt = create_signed_execution_receipt(
        body=build_execution_receipt_body(
            role="gateway_coordinator",
            purpose="research_lab.promotion_decision.v2",
            job_id="promotion-job",
            epoch_id=41,
            sequence=0,
            commit_sha="a" * 40,
            pcr0="b" * 96,
            build_manifest_hash="sha256:" + "c" * 64,
            dependency_lock_hash="sha256:" + "d" * 64,
            config_hash="sha256:" + "e" * 64,
            boot_identity_hash=boot["boot_identity_hash"],
            input_root="sha256:" + "5" * 64,
            output_root=sha256_json({"decision": decision}),
            transport_root_hash=EMPTY_TRANSPORT_ROOT,
            host_operation_root_hash=EMPTY_HOST_OPERATION_ROOT,
            artifact_root=EMPTY_ARTIFACT_ROOT,
            parent_receipt_hashes=(),
            status="succeeded",
            failure_code=None,
            issued_at="2026-07-12T00:00:00Z",
        ),
        enclave_pubkey=public_key,
        sign_digest=key.sign,
    )
    graph = build_receipt_graph(
        root_receipt_hash=receipt["receipt_hash"],
        boot_identities=(boot,),
        receipts=(receipt,),
        transport_attempts=(),
    )
    return graph, receipt


def test_bootstrap_model_must_match_authenticated_active_row(tmp_path):
    artifact = _artifact(tmp_path)
    reader = _Reader({"active_private_model_current": [_active_row(artifact)]})
    source = CoordinatorActiveModelSourceV2(
        reader=reader,
        config_supplier=lambda: SimpleNamespace(improvement_threshold_points=0.25),
    )

    result = source.resolve(
        payload={"artifact": artifact.to_dict()},
        context=_context(),
    )

    assert result["artifact"] == artifact.to_dict()
    assert result["active_model"]["lineage_kind"] == (
        "attested_bootstrap_private_model_manifest_uri"
    )
    assert result["active_model"]["lineage_root"].startswith("sha256:")
    assert result["active_model"]["lineage_receipt_hash"] == ""
    assert reader.calls == [("active_private_model_current", {})]


def test_promoted_model_requires_exact_promotion_passed_receipt(tmp_path):
    artifact = _artifact(tmp_path)
    bundle_hash = "sha256:" + "6" * 64
    bundle_id = "score_bundle:" + "6" * 64
    score_bundle = {
        "score_bundle_hash": bundle_hash,
        "parent_artifact_hash": "sha256:" + "7" * 64,
        "private_holdout_gate": {
            "decision": "private_holdout_approved",
            "private_holdout_evaluated": True,
            "baseline_aggregate_score": 1.0,
            "candidate_total_score": 2.0,
            "candidate_delta_vs_daily_baseline": 1.0,
        },
        "aggregates": {},
    }
    decision = promotion_gate_decision(
        score_bundle,
        candidate_kind="image_build",
        candidate_parent=score_bundle["parent_artifact_hash"],
        active_parent=score_bundle["parent_artifact_hash"],
        threshold_points=0.25,
        auto_promotion_enabled=True,
    ).to_dict()
    graph, receipt = _promotion_graph(decision)
    reader = _Reader(
        {
            "active_private_model_current": [
                _active_row(
                    artifact,
                    source_candidate_id="candidate-1",
                    source_score_bundle_id=bundle_id,
                )
            ],
            "score_bundle_by_id": [
                {
                    "score_bundle_id": bundle_id,
                    "score_bundle_hash": bundle_hash,
                    "score_bundle_doc": score_bundle,
                    "current_event_status": "scored",
                }
            ],
            "attested_business_artifact_by_ref": [
                {
                    "receipt_hash": receipt["receipt_hash"],
                    "artifact_kind": "promotion_decision",
                    "artifact_ref": bundle_id,
                    "artifact_hash": bundle_hash,
                }
            ],
            "attested_receipt_by_hash": [
                {"receipt_doc": receipt}
            ],
        }
    )
    source = CoordinatorActiveModelSourceV2(
        reader=reader,
        config_supplier=lambda: SimpleNamespace(improvement_threshold_points=0.25),
    )

    result = source.resolve(
        payload={"artifact": artifact.to_dict()},
        context=_context(graph=graph),
    )

    assert result["active_model"]["lineage_receipt_hash"] == receipt["receipt_hash"]
    assert result["active_model"]["lineage_kind"] == "attested_promotion"
    assert result["active_model"]["lineage_root"] == receipt["receipt_hash"]


def test_promoted_model_rejects_missing_external_promotion_graph(tmp_path):
    artifact = _artifact(tmp_path)
    reader = _Reader(
        {
            "active_private_model_current": [
                _active_row(
                    artifact,
                    source_candidate_id="candidate-1",
                    source_score_bundle_id="score_bundle:" + "6" * 64,
                )
            ],
            "score_bundle_by_id": [],
        }
    )
    source = CoordinatorActiveModelSourceV2(
        reader=reader,
        config_supplier=lambda: SimpleNamespace(improvement_threshold_points=0.25),
    )

    with pytest.raises(CoordinatorActiveModelSourceV2Error):
        source.resolve(
            payload={"artifact": artifact.to_dict()},
            context=_context(),
        )


def test_active_model_rejects_manifest_substitution(tmp_path):
    artifact = _artifact(tmp_path)
    row = _active_row(artifact, private_model_manifest_hash="sha256:" + "f" * 64)
    source = CoordinatorActiveModelSourceV2(
        reader=_Reader({"active_private_model_current": [row]}),
        config_supplier=lambda: SimpleNamespace(improvement_threshold_points=0.25),
    )

    with pytest.raises(
        CoordinatorActiveModelSourceV2Error,
        match="private_model_manifest_hash",
    ):
        source.resolve(
            payload={"artifact": artifact.to_dict()},
            context=_context(),
        )


def test_direct_release_requires_complete_redacted_evidence(tmp_path):
    artifact = _artifact(tmp_path)
    row = _active_row(
        artifact,
        redacted_version_doc={
            "source": "bootstrap_private_model_manifest_uri",
            "model_artifact_hash": artifact.model_artifact_hash,
        },
    )
    source = CoordinatorActiveModelSourceV2(
        reader=_Reader({"active_private_model_current": [row]}),
        config_supplier=lambda: SimpleNamespace(improvement_threshold_points=0.25),
    )

    with pytest.raises(
        CoordinatorActiveModelSourceV2Error,
        match="private_model_manifest_hash differs",
    ):
        source.resolve(payload={"artifact": artifact.to_dict()}, context=_context())


def test_repo_head_release_binds_repo_sha_and_manifest_uri(tmp_path):
    artifact = _artifact(tmp_path)
    release_doc = {
        "source": "repo_head_sync",
        "model_artifact_hash": artifact.model_artifact_hash,
        "private_model_manifest_hash": artifact.manifest_hash,
        "git_commit_sha": artifact.git_commit_sha,
        "component_registry_version": artifact.component_registry_version,
        "scoring_adapter_version": artifact.scoring_adapter_version,
        "repo_main_sha": artifact.git_commit_sha,
        "current_json_manifest_uri": artifact.manifest_uri,
    }
    source = CoordinatorActiveModelSourceV2(
        reader=_Reader(
            {
                "active_private_model_current": [
                    _active_row(artifact, redacted_version_doc=release_doc)
                ]
            }
        ),
        config_supplier=lambda: SimpleNamespace(improvement_threshold_points=0.25),
    )

    result = source.resolve(
        payload={"artifact": artifact.to_dict()}, context=_context()
    )
    assert result["active_model"]["lineage_kind"] == "attested_repo_head_sync"
    assert result["active_model"]["lineage_root"].startswith("sha256:")

    release_doc["repo_main_sha"] = "f" * 40
    with pytest.raises(
        CoordinatorActiveModelSourceV2Error,
        match="repo-head release commit differs",
    ):
        source.resolve(payload={"artifact": artifact.to_dict()}, context=_context())
