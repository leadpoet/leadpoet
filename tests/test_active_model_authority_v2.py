from __future__ import annotations

import base64
from types import SimpleNamespace

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from gateway.research_lab import active_model_authority_v2
from gateway.tee.coordinator_active_model_source_v2 import (
    CoordinatorActiveModelSourceV2,
    CoordinatorActiveModelSourceV2Error,
)
from gateway.research_lab.bundles import contains_secret_material
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
from research_lab.eval.artifacts import (
    PrivateModelArtifactManifest,
    private_model_artifact_replay_identity_v2,
)
from research_lab.eval.promotion_metric import promotion_gate_decision
from tests.private_model_artifact_fixtures import install_reviewed_consumer_snapshot


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
    install_reviewed_consumer_snapshot(source)
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


def _active_result(artifact, row):
    source = CoordinatorActiveModelSourceV2(
        reader=_Reader({"active_private_model_current": [row]}),
        config_supplier=lambda: SimpleNamespace(improvement_threshold_points=0.25),
    )
    return source.resolve(payload={"artifact": artifact.to_dict()}, context=_context())


def _active_result_with_overrides(result, row, **overrides):
    active_model = {**result["active_model"], **overrides}
    return {
        **result,
        "active_model": active_model,
        "source_state_hash": sha256_json(
            {
                "active_model": active_model,
                "redacted_version_doc": dict(row["redacted_version_doc"]),
                "current_status_at": row["current_status_at"],
            }
        ),
    }


def _active_result_for_expected(artifact, row, expected_active_model):
    active_model = dict(expected_active_model)
    return {
        "schema_version": "leadpoet.active_private_model.v2",
        "artifact": private_model_artifact_replay_identity_v2(artifact),
        "active_model": active_model,
        "source_state_hash": sha256_json(
            {
                "active_model": active_model,
                "redacted_version_doc": dict(row["redacted_version_doc"]),
                "current_status_at": row["current_status_at"],
            }
        ),
    }


def _install_release_identity(monkeypatch, release_hash):
    release = {"release_hash": release_hash}
    expectation = {
        "physical_role": "gateway_coordinator",
        "service_role": "gateway_coordinator",
        "commit_sha": "a" * 40,
        "pcr0": "b" * 96,
        "build_manifest_hash": "sha256:" + "c" * 64,
        "dependency_lock_hash": "sha256:" + "d" * 64,
        "release_hash": release_hash,
    }
    monkeypatch.setattr(active_model_authority_v2, "_load_release", lambda _path: release)
    monkeypatch.setattr(
        active_model_authority_v2,
        "role_expectation",
        lambda value, role: (
            expectation
            if value == release and role == "gateway_coordinator"
            else pytest.fail("unexpected release expectation lookup")
        ),
    )
    monkeypatch.setattr(
        active_model_authority_v2,
        "verify_boot_identity_nitro",
        lambda identity, **_kwargs: identity,
    )
    return release, expectation


def _assertion_authority(
    result,
    *,
    epoch_id,
    receipt_char,
    expectation,
    parent_receipt_hashes=(),
):
    receipt_hash = "sha256:" + receipt_char * 64
    boot_identity_hash = "sha256:" + "e" * 64
    receipt = {
        "receipt_hash": receipt_hash,
        "role": "gateway_coordinator",
        "purpose": "research_lab.active_private_model.v2",
        "epoch_id": epoch_id,
        "sequence": 0,
        "status": "succeeded",
        "output_root": sha256_json(result),
        "boot_identity_hash": boot_identity_hash,
        "parent_receipt_hashes": list(parent_receipt_hashes),
    }
    graph = {
        "root_receipt_hash": receipt_hash,
        "receipts": [receipt],
        "boot_identities": [
            {
                "boot_identity_hash": boot_identity_hash,
                **{
                    field: expectation[field]
                    for field in (
                        "physical_role",
                        "commit_sha",
                        "pcr0",
                        "build_manifest_hash",
                        "dependency_lock_hash",
                    )
                },
            }
        ],
    }
    return receipt, graph


def _execution_replay(result, receipt, graph, *, artifact, release_hash):
    return {
        "row": {
            "role": "gateway_coordinator",
            "operation": "attest_active_private_model",
            "purpose": "research_lab.active_private_model.v2",
            "epoch_id": receipt["epoch_id"],
            "sequence": 0,
            "release_hash": release_hash,
            "result_hash": sha256_json(result),
            "output_root": sha256_json(result),
            "artifact_hashes": sorted(
                {
                    artifact.model_artifact_hash,
                    artifact.manifest_hash,
                    result["source_state_hash"],
                }
            ),
        },
        "result": result,
        "receipt": receipt,
        "receipt_graph": graph,
    }


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

    assert result["artifact"] == private_model_artifact_replay_identity_v2(
        artifact
    )
    assert contains_secret_material(result) is False
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


def test_repo_head_release_accepts_historical_missing_redundant_manifest_uri(tmp_path):
    artifact = _artifact(tmp_path)
    release_doc = {
        "source": "repo_head_sync",
        "model_artifact_hash": artifact.model_artifact_hash,
        "private_model_manifest_hash": artifact.manifest_hash,
        "git_commit_sha": artifact.git_commit_sha,
        "component_registry_version": artifact.component_registry_version,
        "scoring_adapter_version": artifact.scoring_adapter_version,
        "repo_main_sha": artifact.git_commit_sha,
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

    release_doc["current_json_manifest_uri"] = "s3://wrong/manifest.json"
    with pytest.raises(
        CoordinatorActiveModelSourceV2Error,
        match="repo-head release manifest URI differs",
    ):
        source.resolve(payload={"artifact": artifact.to_dict()}, context=_context())


@pytest.mark.asyncio
async def test_active_model_authority_uses_measured_execution_receipt(
    tmp_path,
    monkeypatch,
):
    artifact = _artifact(tmp_path)
    row = _active_row(artifact)
    result = _active_result(artifact, row)
    release_hash = "sha256:" + "c" * 64
    release, expectation = _install_release_identity(monkeypatch, release_hash)
    execution_receipt, execution_graph = _assertion_authority(
        result,
        epoch_id=42,
        receipt_char="a",
        expectation=expectation,
    )
    artifact_receipt = {
        "receipt_hash": "sha256:" + "b" * 64,
        "output_root": "sha256:" + "b" * 64,
    }
    artifact_graph = {
        "root_receipt_hash": artifact_receipt["receipt_hash"],
        "receipts": [artifact_receipt],
    }
    validated = []
    linked = []

    async def select_many(table, *_args, **_kwargs):
        if table == active_model_authority_v2.BUSINESS_ARTIFACT_TABLE:
            return []
        return [row]

    async def execute(**kwargs):
        assert kwargs["release_manifest"] == release
        return {
            "result": result,
            "receipt": artifact_receipt,
            "receipt_graph": artifact_graph,
            "execution_receipt": execution_receipt,
            "execution_receipt_graph": execution_graph,
            "release_hash": release_hash,
        }

    async def persist_links(**kwargs):
        linked.append(kwargs)
        return {"business_artifact_link_count": 1}

    def validate(graph, **kwargs):
        validated.append((graph, kwargs))

    monkeypatch.setattr(active_model_authority_v2, "select_many", select_many)
    monkeypatch.setattr(
        active_model_authority_v2,
        "validate_receipt_graph",
        validate,
    )

    outcome = await active_model_authority_v2.attest_active_private_model_v2(
        artifact=artifact,
        epoch_id=42,
        execute=execute,
        persist_links=persist_links,
    )

    assert outcome["status"] == "matched"
    assert validated == [
        (
            execution_graph,
            {"required_purposes": ("research_lab.active_private_model.v2",)},
        )
    ]
    assert linked[0]["receipt_hash"] == execution_receipt["receipt_hash"]
    assert linked[0]["artifacts"] == (
        {
            "artifact_kind": "active_private_model_assertion_v2",
            "artifact_ref": active_model_authority_v2._assertion_ref_v2(
                artifact=artifact,
                row=row,
                epoch_id=42,
                release_hash=release_hash,
            ),
            "artifact_hash": sha256_json(result),
        },
    )


@pytest.mark.asyncio
async def test_active_model_assertion_link_is_epoch_scoped_and_replay_safe(
    tmp_path,
    monkeypatch,
):
    artifact = _artifact(tmp_path)
    row = _active_row(artifact)
    result = _active_result(artifact, row)
    release_hash = "sha256:" + "d" * 64
    _release, expectation = _install_release_identity(monkeypatch, release_hash)
    links = {}

    async def select_many(table, *_args, **_kwargs):
        if table == active_model_authority_v2.BUSINESS_ARTIFACT_TABLE:
            return []
        return [row]

    async def execute(**kwargs):
        receipt, graph = _assertion_authority(
            result,
            epoch_id=kwargs["epoch_id"],
            receipt_char="a" if kwargs["epoch_id"] == 42 else "b",
            expectation=expectation,
        )
        return {
            "result": result,
            "receipt": receipt,
            "receipt_graph": graph,
            "release_hash": release_hash,
        }

    async def persist_links(**kwargs):
        artifact_link = kwargs["artifacts"][0]
        key = (
            artifact_link["artifact_kind"],
            artifact_link["artifact_ref"],
            artifact_link["artifact_hash"],
        )
        previous = links.get(key)
        assert previous in {None, kwargs["receipt_hash"]}
        links[key] = kwargs["receipt_hash"]
        return {"business_artifact_link_count": 1}

    monkeypatch.setattr(active_model_authority_v2, "select_many", select_many)
    monkeypatch.setattr(
        active_model_authority_v2,
        "validate_receipt_graph",
        lambda *_args, **_kwargs: None,
    )

    for epoch_id in (42, 43, 43):
        await active_model_authority_v2.attest_active_private_model_v2(
            artifact=artifact,
            epoch_id=epoch_id,
            execute=execute,
            persist_links=persist_links,
        )

    assert len(links) == 2
    assert set(links.values()) == {
        "sha256:" + "a" * 64,
        "sha256:" + "b" * 64,
    }


@pytest.mark.asyncio
async def test_active_model_reuses_verified_exact_assertion_before_execution(
    tmp_path,
    monkeypatch,
):
    artifact = _artifact(tmp_path)
    row = _active_row(artifact)
    result = _active_result(artifact, row)
    release_hash = "sha256:" + "f" * 64
    _release, expectation = _install_release_identity(monkeypatch, release_hash)
    receipt, graph = _assertion_authority(
        result,
        epoch_id=42,
        receipt_char="6",
        expectation=expectation,
    )
    assertion_ref = active_model_authority_v2._assertion_ref_v2(
        artifact=artifact,
        row=row,
        epoch_id=42,
        release_hash=release_hash,
    )
    artifact_hash = sha256_json(result)
    replay = _execution_replay(
        result,
        receipt,
        graph,
        artifact=artifact,
        release_hash=release_hash,
    )
    executions = 0

    async def select_many(table, *_args, **_kwargs):
        if table == active_model_authority_v2.BUSINESS_ARTIFACT_TABLE:
            return [
                {
                    "receipt_hash": receipt["receipt_hash"],
                    "artifact_kind": "active_private_model_assertion_v2",
                    "artifact_ref": assertion_ref,
                    "artifact_hash": artifact_hash,
                }
            ]
        return [row]

    async def execute(**_kwargs):
        nonlocal executions
        executions += 1
        pytest.fail("an exact durable assertion must be reused before execution")

    async def load_graph(**kwargs):
        assert kwargs == {
            "artifact_kind": "active_private_model_assertion_v2",
            "artifact_ref": assertion_ref,
            "artifact_hash": artifact_hash,
        }
        return graph

    async def load_replay(receipt_hash, **kwargs):
        assert receipt_hash == receipt["receipt_hash"]
        assert kwargs == {
            "expected_operation": "attest_active_private_model",
            "expected_purpose": "research_lab.active_private_model.v2",
        }
        return replay

    monkeypatch.setattr(active_model_authority_v2, "select_many", select_many)
    monkeypatch.setattr(
        active_model_authority_v2,
        "load_business_artifact_graph_v2",
        load_graph,
    )
    monkeypatch.setattr(
        active_model_authority_v2,
        "load_execution_result_by_receipt_v2",
        load_replay,
    )
    monkeypatch.setattr(
        active_model_authority_v2,
        "validate_receipt_graph",
        lambda *_args, **_kwargs: None,
    )

    outcome = await active_model_authority_v2.attest_active_private_model_v2(
        artifact=artifact,
        epoch_id=42,
        execute=execute,
    )

    assert executions == 0
    assert outcome["replay_status"] == "business_artifact_exact"
    assert outcome["execution_receipt"] == receipt
    assert outcome["execution_receipt_graph"] == graph


@pytest.mark.asyncio
async def test_active_model_recovers_verified_concurrent_link_winner(
    tmp_path,
    monkeypatch,
):
    artifact = _artifact(tmp_path)
    row = _active_row(artifact)
    result = _active_result(artifact, row)
    release_hash = "sha256:" + "7" * 64
    release, expectation = _install_release_identity(monkeypatch, release_hash)
    loser_receipt, loser_graph = _assertion_authority(
        result,
        epoch_id=42,
        receipt_char="8",
        expectation=expectation,
    )
    winner_receipt, winner_graph = _assertion_authority(
        result,
        epoch_id=42,
        receipt_char="9",
        expectation=expectation,
    )
    assertion_ref = active_model_authority_v2._assertion_ref_v2(
        artifact=artifact,
        row=row,
        epoch_id=42,
        release_hash=release_hash,
    )
    artifact_hash = sha256_json(result)
    winner_replay = _execution_replay(
        result,
        winner_receipt,
        winner_graph,
        artifact=artifact,
        release_hash=release_hash,
    )
    business_reads = 0

    async def select_many(table, *_args, **_kwargs):
        nonlocal business_reads
        if table == active_model_authority_v2.BUSINESS_ARTIFACT_TABLE:
            business_reads += 1
            if business_reads == 1:
                return []
            return [
                {
                    "receipt_hash": winner_receipt["receipt_hash"],
                    "artifact_kind": "active_private_model_assertion_v2",
                    "artifact_ref": assertion_ref,
                    "artifact_hash": artifact_hash,
                }
            ]
        return [row]

    async def execute(**kwargs):
        assert kwargs["release_manifest"] == release
        return {
            "result": result,
            "receipt": loser_receipt,
            "receipt_graph": loser_graph,
            "release_hash": release_hash,
        }

    async def persist_links(**_kwargs):
        raise active_model_authority_v2.AttestedV2StoreError(
            "research_lab_attested_business_artifact_links_v2 "
            "stored row conflicts at receipt_hash"
        )

    async def load_winner_graph(**_kwargs):
        return winner_graph

    async def load_winner_replay(*_args, **_kwargs):
        return winner_replay

    monkeypatch.setattr(active_model_authority_v2, "select_many", select_many)
    monkeypatch.setattr(
        active_model_authority_v2,
        "load_business_artifact_graph_v2",
        load_winner_graph,
    )
    monkeypatch.setattr(
        active_model_authority_v2,
        "load_execution_result_by_receipt_v2",
        load_winner_replay,
    )
    monkeypatch.setattr(
        active_model_authority_v2,
        "validate_receipt_graph",
        lambda *_args, **_kwargs: None,
    )

    outcome = await active_model_authority_v2.attest_active_private_model_v2(
        artifact=artifact,
        epoch_id=42,
        execute=execute,
        persist_links=persist_links,
    )

    assert business_reads == 2
    assert outcome["replay_status"] == "business_artifact_exact"
    assert outcome["execution_receipt"] == winner_receipt


@pytest.mark.asyncio
async def test_active_model_replay_fails_closed_on_release_substitution(
    tmp_path,
    monkeypatch,
):
    artifact = _artifact(tmp_path)
    row = _active_row(artifact)
    result = _active_result(artifact, row)
    release_hash = "sha256:" + "1" * 64
    _release, expectation = _install_release_identity(monkeypatch, release_hash)
    receipt, graph = _assertion_authority(
        result,
        epoch_id=42,
        receipt_char="2",
        expectation=expectation,
    )
    assertion_ref = active_model_authority_v2._assertion_ref_v2(
        artifact=artifact,
        row=row,
        epoch_id=42,
        release_hash=release_hash,
    )
    artifact_hash = sha256_json(result)
    replay = _execution_replay(
        result,
        receipt,
        graph,
        artifact=artifact,
        release_hash="sha256:" + "3" * 64,
    )

    async def select_many(table, *_args, **_kwargs):
        if table == active_model_authority_v2.BUSINESS_ARTIFACT_TABLE:
            return [
                {
                    "receipt_hash": receipt["receipt_hash"],
                    "artifact_kind": "active_private_model_assertion_v2",
                    "artifact_ref": assertion_ref,
                    "artifact_hash": artifact_hash,
                }
            ]
        return [row]

    async def load_graph(**_kwargs):
        return graph

    async def load_replay(*_args, **_kwargs):
        return replay

    monkeypatch.setattr(active_model_authority_v2, "select_many", select_many)
    monkeypatch.setattr(
        active_model_authority_v2,
        "load_business_artifact_graph_v2",
        load_graph,
    )
    monkeypatch.setattr(
        active_model_authority_v2,
        "load_execution_result_by_receipt_v2",
        load_replay,
    )
    monkeypatch.setattr(
        active_model_authority_v2,
        "validate_receipt_graph",
        lambda *_args, **_kwargs: None,
    )

    with pytest.raises(
        active_model_authority_v2.ActivePrivateModelAuthorityV2Error,
        match="stored active private model assertion differs",
    ):
        await active_model_authority_v2.attest_active_private_model_v2(
            artifact=artifact,
            epoch_id=42,
            execute=lambda **_kwargs: pytest.fail("substitution must fail closed"),
        )


def test_active_model_assertion_ref_changes_after_same_epoch_reactivation(tmp_path):
    artifact = _artifact(tmp_path)
    before = _active_row(artifact, current_status_at="2026-07-12T00:00:00Z")
    after = _active_row(artifact, current_status_at="2026-07-12T00:05:00Z")
    release_hash = "sha256:" + "4" * 64

    assert active_model_authority_v2._assertion_ref_v2(
        artifact=artifact,
        row=before,
        epoch_id=42,
        release_hash=release_hash,
    ) != active_model_authority_v2._assertion_ref_v2(
        artifact=artifact,
        row=after,
        epoch_id=42,
        release_hash=release_hash,
    )


def test_active_model_result_rejects_self_consistent_artifact_substitution(tmp_path):
    artifact = _artifact(tmp_path)
    row = _active_row(artifact)
    result = _active_result(artifact, row)
    expected_active, _parents = (
        active_model_authority_v2._expected_active_model_authority_v2(
            artifact=artifact,
            row=row,
            promotion_graph=None,
        )
    )
    substituted = _active_result_with_overrides(
        result,
        row,
        private_model_manifest_uri="s3://private/manifests/substituted.json",
    )

    with pytest.raises(
        active_model_authority_v2.ActivePrivateModelAuthorityV2Error,
        match="measured result differs",
    ):
        active_model_authority_v2._validate_active_model_result_v2(
            artifact=artifact,
            row=row,
            result=substituted,
            expected_active_model=expected_active,
        )


@pytest.mark.parametrize("field", ("lineage_root", "lineage_receipt_hash"))
def test_active_model_result_rejects_self_consistent_lineage_substitution(
    tmp_path,
    field,
):
    artifact = _artifact(tmp_path)
    row = _active_row(artifact)
    result = _active_result(artifact, row)
    expected_active, _parents = (
        active_model_authority_v2._expected_active_model_authority_v2(
            artifact=artifact,
            row=row,
            promotion_graph=None,
        )
    )
    substituted = _active_result_with_overrides(
        result,
        row,
        **{field: "sha256:" + "9" * 64},
    )

    with pytest.raises(
        active_model_authority_v2.ActivePrivateModelAuthorityV2Error,
        match="measured result differs",
    ):
        active_model_authority_v2._validate_active_model_result_v2(
            artifact=artifact,
            row=row,
            result=substituted,
            expected_active_model=expected_active,
        )


def test_active_model_result_rejects_unexpected_active_field(tmp_path):
    artifact = _artifact(tmp_path)
    row = _active_row(artifact)
    result = _active_result(artifact, row)
    expected_active, _parents = (
        active_model_authority_v2._expected_active_model_authority_v2(
            artifact=artifact,
            row=row,
            promotion_graph=None,
        )
    )
    substituted = _active_result_with_overrides(
        result,
        row,
        unmeasured_authority="accepted",
    )

    with pytest.raises(
        active_model_authority_v2.ActivePrivateModelAuthorityV2Error,
        match="measured result differs",
    ):
        active_model_authority_v2._validate_active_model_result_v2(
            artifact=artifact,
            row=row,
            result=substituted,
            expected_active_model=expected_active,
        )


def test_active_model_receipt_rejects_unexpected_direct_parent(tmp_path, monkeypatch):
    artifact = _artifact(tmp_path)
    row = _active_row(artifact)
    result = _active_result(artifact, row)
    expected_active, expected_parents = (
        active_model_authority_v2._expected_active_model_authority_v2(
            artifact=artifact,
            row=row,
            promotion_graph=None,
        )
    )
    release = {"release_hash": "sha256:" + "8" * 64}
    _unused_release, expectation = _install_release_identity(
        monkeypatch,
        release["release_hash"],
    )
    receipt, graph = _assertion_authority(
        result,
        epoch_id=42,
        receipt_char="7",
        expectation=expectation,
        parent_receipt_hashes=("sha256:" + "6" * 64,),
    )

    with pytest.raises(
        active_model_authority_v2.ActivePrivateModelAuthorityV2Error,
        match="receipt authority differs",
    ):
        active_model_authority_v2._validate_assertion_authority_v2(
            artifact=artifact,
            row=row,
            epoch_id=42,
            release=release,
            result=result,
            receipt=receipt,
            graph=graph,
            expected_active_model=expected_active,
            expected_parent_receipt_hashes=expected_parents,
        )


@pytest.mark.asyncio
async def test_active_model_replay_rejects_changed_current_promotion_graph(
    tmp_path,
    monkeypatch,
):
    artifact = _artifact(tmp_path)
    score_bundle_id = "score_bundle:" + "5" * 64
    row = _active_row(
        artifact,
        source_candidate_id="candidate:" + "4" * 64,
        source_score_bundle_id=score_bundle_id,
    )
    previous_promotion_graph = {
        "root_receipt_hash": "sha256:" + "3" * 64
    }
    current_promotion_graph = {
        "root_receipt_hash": "sha256:" + "2" * 64
    }
    previous_active, previous_parents = (
        active_model_authority_v2._expected_active_model_authority_v2(
            artifact=artifact,
            row=row,
            promotion_graph=previous_promotion_graph,
        )
    )
    result = _active_result_for_expected(artifact, row, previous_active)
    release_hash = "sha256:" + "1" * 64
    _release, expectation = _install_release_identity(monkeypatch, release_hash)
    receipt, assertion_graph = _assertion_authority(
        result,
        epoch_id=42,
        receipt_char="a",
        expectation=expectation,
        parent_receipt_hashes=previous_parents,
    )
    assertion_ref = active_model_authority_v2._assertion_ref_v2(
        artifact=artifact,
        row=row,
        epoch_id=42,
        release_hash=release_hash,
    )
    artifact_hash = sha256_json(result)
    replay = _execution_replay(
        result,
        receipt,
        assertion_graph,
        artifact=artifact,
        release_hash=release_hash,
    )

    async def select_many(table, *_args, **_kwargs):
        if table == active_model_authority_v2.BUSINESS_ARTIFACT_TABLE:
            return [
                {
                    "receipt_hash": receipt["receipt_hash"],
                    "artifact_kind": "active_private_model_assertion_v2",
                    "artifact_ref": assertion_ref,
                    "artifact_hash": artifact_hash,
                }
            ]
        return [row]

    async def load_graph(**kwargs):
        if kwargs["artifact_kind"] == "promotion_decision":
            assert kwargs["artifact_ref"] == score_bundle_id
            return current_promotion_graph
        assert kwargs["artifact_kind"] == "active_private_model_assertion_v2"
        return assertion_graph

    async def load_replay(*_args, **_kwargs):
        return replay

    monkeypatch.setattr(active_model_authority_v2, "select_many", select_many)
    monkeypatch.setattr(
        active_model_authority_v2,
        "load_business_artifact_graph_v2",
        load_graph,
    )
    monkeypatch.setattr(
        active_model_authority_v2,
        "load_execution_result_by_receipt_v2",
        load_replay,
    )
    monkeypatch.setattr(
        active_model_authority_v2,
        "validate_receipt_graph",
        lambda *_args, **_kwargs: None,
    )

    with pytest.raises(
        active_model_authority_v2.ActivePrivateModelAuthorityV2Error,
        match="measured result differs",
    ):
        await active_model_authority_v2.attest_active_private_model_v2(
            artifact=artifact,
            epoch_id=42,
            execute=lambda **_kwargs: pytest.fail(
                "changed current promotion authority must fail before execution"
            ),
        )
