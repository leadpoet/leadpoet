from __future__ import annotations

import asyncio
import base64
from copy import deepcopy
import io
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

import gateway.research_lab.autoresearch_authority_v2 as authority
from gateway.tee.provider_outcome_v2 import ProviderOutcomeLedgerV2
from gateway.tee.source_add_runtime_v2 import build_source_add_runtime_catalog_v2
from leadpoet_canonical.attested_v2 import (
    EMPTY_ARTIFACT_ROOT,
    EMPTY_HOST_OPERATION_ROOT,
    EMPTY_TRANSPORT_ROOT,
    build_boot_identity_body,
    build_execution_receipt_body,
    build_receipt_graph,
    create_boot_identity,
    create_signed_execution_receipt,
    sha256_bytes,
    sha256_json,
)
from research_lab.eval import private_model_artifact_replay_identity_v2
from gateway.research_lab.autoresearch_runtime import AutoResearchRuntimeSettings
from gateway.research_lab.git_tree_models import (
    TreeCheckpoint,
    TreePolicy,
    TreeResult,
    derive_tree_id,
)
from gateway.research_lab.git_tree_repository import GitTreeRepository
from gateway.tee.autoresearch_executor_v2 import _candidate_document
from tests.private_model_artifact_fixtures import (
    build_private_artifact_with_adapted_source_admission,
    install_reviewed_consumer_snapshot,
)
from research_lab.code_editing import code_edit_candidate_manifest


MINER_HOTKEY = "miner-hotkey"
HASHES = {
    "key_ref_hash": "sha256:" + "1" * 64,
    "miner_hotkey_hash": sha256_bytes(MINER_HOTKEY.encode("utf-8")),
    "runtime_credential_value_hash": "sha256:" + "3" * 64,
    "management_credential_value_hash": "sha256:" + "4" * 64,
}
KEY_REF = "encrypted_ref:openrouter:" + "a" * 32
RECEIPT_HASH = "sha256:" + "5" * 64
RUN_ID = "run-credential-transition"
QUEUE_EVENT_HASH = "sha256:" + "6" * 64


class _CoordinatorClient:
    def __init__(self) -> None:
        self.released = []

    async def v2_release_job_credentials(self, job_id):
        self.released.append(str(job_id))
        return {"status": "released", "job_id": str(job_id)}


def _coordinator_graph(
    result,
    purpose="research_lab.provider_outcome_snapshot.v2",
):
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
            issued_at="2026-07-10T20:00:00Z",
        ),
        attestation_document_b64=base64.b64encode(b"attestation").decode(
            "ascii"
        ),
    )
    receipt = create_signed_execution_receipt(
        body=build_execution_receipt_body(
            role="gateway_coordinator",
            purpose=purpose,
            job_id="provider-outcome-snapshot",
            epoch_id=10,
            sequence=0,
            commit_sha="a" * 40,
            pcr0="b" * 96,
            build_manifest_hash="sha256:" + "c" * 64,
            dependency_lock_hash="sha256:" + "d" * 64,
            config_hash="sha256:" + "e" * 64,
            boot_identity_hash=boot["boot_identity_hash"],
            input_root="sha256:" + "5" * 64,
            output_root=sha256_json(result),
            transport_root_hash=EMPTY_TRANSPORT_ROOT,
            host_operation_root_hash=EMPTY_HOST_OPERATION_ROOT,
            artifact_root=EMPTY_ARTIFACT_ROOT,
            parent_receipt_hashes=(),
            status="succeeded",
            failure_code=None,
            issued_at="2026-07-10T20:00:00Z",
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


def _autoresearch_guard_authority(result, *, artifact_wrapped=False):
    key = Ed25519PrivateKey.generate()
    public_key = key.public_key().public_bytes_raw().hex()
    boot = create_boot_identity(
        body=build_boot_identity_body(
            role="gateway_autoresearch",
            physical_role="gateway_autoresearch",
            commit_sha="a" * 40,
            pcr0="b" * 96,
            build_manifest_hash="sha256:" + "c" * 64,
            dependency_lock_hash="sha256:" + "d" * 64,
            config_hash="sha256:" + "e" * 64,
            boot_nonce="5" * 32,
            signing_pubkey=public_key,
            transport_pubkey="6" * 64,
            transport_certificate_hash="sha256:" + "7" * 64,
            attestation_user_data_hash="sha256:" + "8" * 64,
            issued_at="2026-07-10T20:00:00Z",
        ),
        attestation_document_b64=base64.b64encode(b"attestation").decode(
            "ascii"
        ),
    )
    receipt = create_signed_execution_receipt(
        body=build_execution_receipt_body(
            role="gateway_autoresearch",
            purpose="research_lab.openrouter_guard.v2",
            job_id="openrouter-guard",
            epoch_id=10,
            sequence=0,
            commit_sha="a" * 40,
            pcr0="b" * 96,
            build_manifest_hash="sha256:" + "c" * 64,
            dependency_lock_hash="sha256:" + "d" * 64,
            config_hash="sha256:" + "e" * 64,
            boot_identity_hash=boot["boot_identity_hash"],
            input_root="sha256:" + "9" * 64,
            output_root=sha256_json(result),
            transport_root_hash=EMPTY_TRANSPORT_ROOT,
            host_operation_root_hash=EMPTY_HOST_OPERATION_ROOT,
            artifact_root=EMPTY_ARTIFACT_ROOT,
            parent_receipt_hashes=(),
            status="succeeded",
            failure_code=None,
            issued_at="2026-07-10T20:00:00Z",
        ),
        enclave_pubkey=public_key,
        sign_digest=key.sign,
    )
    graph = build_receipt_graph(
        root_receipt_hash=receipt["receipt_hash"],
        boot_identities=(boot,),
        receipts=(receipt,),
        transport_attempts=(),
        host_operations=(),
    )
    authority_document = {
        "result": dict(result),
        "receipt": receipt,
        "receipt_graph": graph,
    }
    if not artifact_wrapped:
        return authority_document

    artifact_key = Ed25519PrivateKey.generate()
    artifact_public_key = artifact_key.public_key().public_bytes_raw().hex()
    artifact_boot = create_boot_identity(
        body=build_boot_identity_body(
            role="gateway_coordinator",
            physical_role="gateway_coordinator",
            commit_sha="a" * 40,
            pcr0="b" * 96,
            build_manifest_hash="sha256:" + "c" * 64,
            dependency_lock_hash="sha256:" + "d" * 64,
            config_hash="sha256:" + "e" * 64,
            boot_nonce="9" * 32,
            signing_pubkey=artifact_public_key,
            transport_pubkey="a" * 64,
            transport_certificate_hash="sha256:" + "b" * 64,
            attestation_user_data_hash="sha256:" + "c" * 64,
            issued_at="2026-07-10T20:00:01Z",
        ),
        attestation_document_b64=base64.b64encode(b"attestation").decode(
            "ascii"
        ),
    )
    artifact_receipt = create_signed_execution_receipt(
        body=build_execution_receipt_body(
            role="gateway_coordinator",
            purpose="leadpoet.artifact_persistence.v2",
            job_id="openrouter-guard-artifact-persistence",
            epoch_id=10,
            sequence=0,
            commit_sha="a" * 40,
            pcr0="b" * 96,
            build_manifest_hash="sha256:" + "c" * 64,
            dependency_lock_hash="sha256:" + "d" * 64,
            config_hash="sha256:" + "e" * 64,
            boot_identity_hash=artifact_boot["boot_identity_hash"],
            input_root="sha256:" + "d" * 64,
            output_root="sha256:" + "e" * 64,
            transport_root_hash=EMPTY_TRANSPORT_ROOT,
            host_operation_root_hash=EMPTY_HOST_OPERATION_ROOT,
            artifact_root=EMPTY_ARTIFACT_ROOT,
            parent_receipt_hashes=(receipt["receipt_hash"],),
            status="succeeded",
            failure_code=None,
            issued_at="2026-07-10T20:00:01Z",
        ),
        enclave_pubkey=artifact_public_key,
        sign_digest=artifact_key.sign,
    )
    lineage_graph = build_receipt_graph(
        root_receipt_hash=artifact_receipt["receipt_hash"],
        boot_identities=(boot, artifact_boot),
        receipts=(receipt, artifact_receipt),
        transport_attempts=(),
        host_operations=(),
    )
    return {
        "result": dict(result),
        "receipt": artifact_receipt,
        "receipt_graph": lineage_graph,
        "execution_receipt": receipt,
        "execution_receipt_graph": graph,
    }


def _artifact_wrapped_coordinator_authority(result, *, purpose):
    execution_graph, execution_receipt = _coordinator_graph(
        result,
        purpose=purpose,
    )
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
            boot_nonce="6" * 32,
            signing_pubkey=public_key,
            transport_pubkey="7" * 64,
            transport_certificate_hash="sha256:" + "8" * 64,
            attestation_user_data_hash="sha256:" + "9" * 64,
            issued_at="2026-07-10T20:00:01Z",
        ),
        attestation_document_b64=base64.b64encode(b"attestation").decode(
            "ascii"
        ),
    )
    artifact_receipt = create_signed_execution_receipt(
        body=build_execution_receipt_body(
            role="gateway_coordinator",
            purpose="leadpoet.artifact_persistence.v2",
            job_id="artifact-persistence",
            epoch_id=10,
            sequence=0,
            commit_sha="a" * 40,
            pcr0="b" * 96,
            build_manifest_hash="sha256:" + "c" * 64,
            dependency_lock_hash="sha256:" + "d" * 64,
            config_hash="sha256:" + "e" * 64,
            boot_identity_hash=boot["boot_identity_hash"],
            input_root="sha256:" + "6" * 64,
            output_root="sha256:" + "7" * 64,
            transport_root_hash=EMPTY_TRANSPORT_ROOT,
            host_operation_root_hash=EMPTY_HOST_OPERATION_ROOT,
            artifact_root=EMPTY_ARTIFACT_ROOT,
            parent_receipt_hashes=(execution_receipt["receipt_hash"],),
            status="succeeded",
            failure_code=None,
            issued_at="2026-07-10T20:00:01Z",
        ),
        enclave_pubkey=public_key,
        sign_digest=key.sign,
    )
    lineage_graph = build_receipt_graph(
        root_receipt_hash=artifact_receipt["receipt_hash"],
        boot_identities=(
            *execution_graph["boot_identities"],
            boot,
        ),
        receipts=(execution_receipt, artifact_receipt),
        transport_attempts=(),
    )
    return {
        "result": result,
        "receipt": artifact_receipt,
        "receipt_graph": lineage_graph,
        "execution_receipt": execution_receipt,
        "execution_receipt_graph": execution_graph,
    }


def _artifact(tmp_path: Path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "research_lab_adapter.py").write_text(
        "def run():\n    return 1\n",
        encoding="utf-8",
    )
    install_reviewed_consumer_snapshot(source)
    return authority.PrivateModelArtifactManifest.from_mapping(
        build_private_artifact_with_adapted_source_admission(
            source_path=source,
            git_commit_sha="a" * 40,
            image_digest=(
                "123456789012.dkr.ecr.us-east-1.amazonaws.com/private@sha256:"
                + "b" * 64
            ),
            manifest_uri="s3://private/manifests/current.json",
            signature_ref="kms:signature",
            component_registry_version="1",
            scoring_adapter_version="1",
        )
    )


def _canonical_image_build_candidate(tmp_path: Path):
    parent = _artifact(tmp_path)
    candidate_source = tmp_path / "candidate-source"
    shutil.copytree(tmp_path / "source", candidate_source)
    (candidate_source / "research_lab_adapter.py").write_text(
        "def run():\n    return 2\n",
        encoding="utf-8",
    )
    candidate_manifest = authority.PrivateModelArtifactManifest.from_mapping(
        build_private_artifact_with_adapted_source_admission(
            source_path=candidate_source,
            git_commit_sha="c" * 40,
            image_digest=(
                "123456789012.dkr.ecr.us-east-1.amazonaws.com/private@sha256:"
                + "d" * 64
            ),
            manifest_uri="s3://private/manifests/candidate.json",
            signature_ref="kms:candidate-signature",
            component_registry_version="1",
            scoring_adapter_version="1",
        )
    )
    draft = authority.CodeEditDraft(
        failure_mode="bounded recall",
        mechanism="increase the source runtime value",
        expected_improvement="recover more valid companies",
        risk="bounded runtime increase",
        lane="query_construction",
        target_files=("research_lab_adapter.py",),
        unified_diff=(
            "diff --git a/research_lab_adapter.py b/research_lab_adapter.py\n"
            "--- a/research_lab_adapter.py\n"
            "+++ b/research_lab_adapter.py\n"
            "@@ -1,2 +1,2 @@\n"
            " def run():\n"
            "-    return 1\n"
            "+    return 2\n"
        ),
        redacted_summary="increase a bounded sourcing runtime value",
        test_plan="run private tests",
        rollback_plan="revert the patch",
    )
    source_diff_hash = sha256_json({"unified_diff": draft.unified_diff})
    build_payload = {
        "schema_version": "1.1",
        "candidate_kind": "image_build",
        "parent_artifact_hash": parent.model_artifact_hash,
        "candidate_model_artifact_hash": candidate_manifest.model_artifact_hash,
        "candidate_model_manifest_hash": candidate_manifest.manifest_hash,
        "source_diff_hash": source_diff_hash,
    }
    build_doc = {
        **build_payload,
        "build_doc_hash": sha256_json(build_payload),
    }
    code_edit_manifest = code_edit_candidate_manifest(
        draft=draft,
        parent_artifact_hash=parent.model_artifact_hash,
        candidate_artifact_hash=candidate_manifest.model_artifact_hash,
        candidate_model_manifest_hash=candidate_manifest.manifest_hash,
        source_diff_hash=source_diff_hash,
        build_doc_hash=build_doc["build_doc_hash"],
    )
    build = authority.CodeEditBuildResult(
        candidate_model_manifest=candidate_manifest,
        code_edit_manifest=code_edit_manifest,
        source_diff_hash=source_diff_hash,
        build_doc=build_doc,
    )
    candidate = authority.BuiltCodeEditCandidate(
        draft=draft,
        build=build,
        node_id="tree-node:" + "1" * 64,
        iteration=1,
        tree_parent_artifact_hash=parent.model_artifact_hash,
    )
    return _candidate_document(candidate), candidate


def test_candidate_accepts_measured_image_build_round_trip(tmp_path):
    document, candidate = _canonical_image_build_candidate(tmp_path)

    parsed = authority._candidate(document)

    assert parsed.draft == candidate.draft
    assert parsed.build == candidate.build
    assert parsed.tree_parent_artifact_hash == candidate.tree_parent_artifact_hash


def test_candidate_rejects_self_consistent_forged_image_build_manifest(tmp_path):
    document, _candidate_value = _canonical_image_build_candidate(tmp_path)
    forged = deepcopy(document)
    forged["build"]["code_edit_manifest"]["patch_doc"]["target_files"] = [
        "sourcing_model/other.py"
    ]
    payload = {
        key: value
        for key, value in forged["build"]["code_edit_manifest"].items()
        if key != "manifest_hash"
    }
    forged["build"]["code_edit_manifest"]["manifest_hash"] = sha256_json(payload)

    with pytest.raises(
        authority.AutoresearchAuthorityV2Error,
        match="code-edit manifest differs from measured build inputs",
    ):
        authority._candidate(forged)


def test_guard_bridge_provisions_both_envelopes_and_releases_job(monkeypatch):
    loaded_kinds = []
    provisioned = []
    client = _CoordinatorClient()
    execution_payloads = []
    monkeypatch.setattr(
        authority,
        "_load_release",
        lambda _path: {"release_hash": "sha256:" + "9" * 64},
    )

    async def commitments(**_kwargs):
        return dict(HASHES)

    async def load_envelope(*, credential_kind, job_id, **_kwargs):
        loaded_kinds.append(credential_kind)
        return {"credential_kind": credential_kind, "job_id": job_id}

    async def provision(envelope, **_kwargs):
        provisioned.append(dict(envelope))
        return {"status": "ready"}

    async def execute(**kwargs):
        assert kwargs["operation"] == authority.OP_VERIFY_OPENROUTER_GUARD
        assert kwargs["purpose"] == "research_lab.openrouter_guard.v2"
        assert set(kwargs["input_artifact_hashes"]) == set(HASHES.values())
        execution_payloads.append(dict(kwargs["payload"]))
        result = {
            "schema_version": authority.OPENROUTER_GUARD_RESULT_SCHEMA_VERSION,
            **HASHES,
            "run_state_hash": kwargs["payload"]["run_state_hash"],
            "preflight_status": "passed",
            "preflight_error_type": "",
            "credit_depleted": False,
            "credit_limit_remaining": 10,
            "privacy_proof_doc": {"proof_hash": "sha256:" + "6" * 64},
        }
        return {
            "result": result,
            "receipt": {"receipt_hash": RECEIPT_HASH},
            "receipt_graph": {"root_receipt_hash": RECEIPT_HASH},
        }

    monkeypatch.setattr(
        authority,
        "load_openrouter_credential_commitments_v2",
        commitments,
    )
    monkeypatch.setattr(
        authority,
        "load_openrouter_job_credential_envelope_v2",
        load_envelope,
    )
    monkeypatch.setattr(authority, "provision_job_provider_envelope_v2", provision)

    first = asyncio.run(
        authority.verify_openrouter_guard_v2(
            key_ref=KEY_REF,
            miner_hotkey=MINER_HOTKEY,
            run_id=RUN_ID,
            queue_event_hash=QUEUE_EVENT_HASH,
            epoch_id=10,
            execute=execute,
            coordinator_client=client,
        )
    )
    second = asyncio.run(
        authority.verify_openrouter_guard_v2(
            key_ref=KEY_REF,
            miner_hotkey=MINER_HOTKEY,
            run_id=RUN_ID,
            queue_event_hash="sha256:" + "7" * 64,
            epoch_id=10,
            execute=execute,
            coordinator_client=client,
        )
    )
    replay = asyncio.run(
        authority.verify_openrouter_guard_v2(
            key_ref=KEY_REF,
            miner_hotkey=MINER_HOTKEY,
            run_id=RUN_ID,
            queue_event_hash=QUEUE_EVENT_HASH,
            epoch_id=10,
            execute=execute,
            coordinator_client=client,
        )
    )

    assert loaded_kinds == ["runtime", "management"] * 3
    assert [item["credential_kind"] for item in provisioned] == [
        "runtime",
        "management",
    ] * 3
    assert len(client.released) == 3
    assert client.released[0] != client.released[1]
    assert client.released[0] == client.released[2]
    assert first.credential_commitments == HASHES
    assert first.credit_depleted is False
    assert first.run_state_hash != second.run_state_hash
    assert first.run_state_hash == replay.run_state_hash
    assert execution_payloads[0]["run_id"] == RUN_ID
    assert execution_payloads[0]["queue_event_hash"] == QUEUE_EVENT_HASH


def test_guard_bridge_releases_partial_lease_when_provisioning_fails(monkeypatch):
    client = _CoordinatorClient()
    monkeypatch.setattr(
        authority,
        "_load_release",
        lambda _path: {"release_hash": "sha256:" + "9" * 64},
    )
    monkeypatch.setattr(
        authority,
        "load_openrouter_credential_commitments_v2",
        lambda **_kwargs: _async_value(dict(HASHES)),
    )
    monkeypatch.setattr(
        authority,
        "load_openrouter_job_credential_envelope_v2",
        lambda **kwargs: _async_value(dict(kwargs)),
    )

    async def fail_provision(*_args, **_kwargs):
        raise RuntimeError("KMS unavailable")

    monkeypatch.setattr(
        authority,
        "provision_job_provider_envelope_v2",
        fail_provision,
    )

    with pytest.raises(RuntimeError, match="KMS unavailable"):
        asyncio.run(
            authority.verify_openrouter_guard_v2(
                key_ref=KEY_REF,
                miner_hotkey=MINER_HOTKEY,
                run_id=RUN_ID,
                queue_event_hash=QUEUE_EVENT_HASH,
                epoch_id=10,
                execute=lambda **_kwargs: pytest.fail("execution must not start"),
                coordinator_client=client,
            )
        )
    assert len(client.released) == 1


def test_stale_parent_repair_preserves_dynamic_worker_proxy_index(
    tmp_path, monkeypatch
):
    worker_indexes = []
    client = _CoordinatorClient()
    artifact = _artifact(tmp_path)
    draft = authority.CodeEditDraft(
        failure_mode="fixture",
        mechanism="fixture",
        expected_improvement="fixture",
        risk="fixture",
        lane="code_edit",
        target_files=("research_lab_adapter.py",),
        unified_diff=(
            "--- a/research_lab_adapter.py\n"
            "+++ b/research_lab_adapter.py\n"
            "@@ -1,2 +1,2 @@\n"
            " def run():\n"
            "-    return 1\n"
            "+    return 2\n"
        ),
        redacted_summary="fixture",
        test_plan="fixture",
        rollback_plan="fixture",
    )
    candidate_graph, _receipt = _coordinator_graph(
        {"candidate_id": "candidate-1"}
    )
    source_diff_hash = sha256_json({"unified_diff": draft.unified_diff})

    monkeypatch.setattr(
        authority,
        "_load_release",
        lambda _path: {"release_hash": "sha256:" + "9" * 64},
    )

    async def source_bundle(*_args, **_kwargs):
        return {"archive_sha256": "sha256:" + "8" * 64}

    def load_profile(_profile, *, worker_index, **_kwargs):
        worker_indexes.append(worker_index)
        return {
            "credential_ref_hashes": {"openrouter": "sha256:" + "7" * 64}
        }

    async def provision(*_args, **_kwargs):
        return {"status": "ready"}

    async def execute(**kwargs):
        assert kwargs["provider_credential_profile"] == (
            authority.STALE_PARENT_REPAIR_PROFILE
        )
        return {
            "result": {
                "schema_version": authority.STALE_PARENT_REPAIR_RESULT_SCHEMA_VERSION,
                "run_id": "run-1",
                "candidate_id": "candidate-1",
                "draft": draft.to_dict(),
                "repair_used": False,
                "original_source_diff_hash": source_diff_hash,
                "result_source_diff_hash": source_diff_hash,
                "active_artifact_hash": artifact.model_artifact_hash,
                "source_bundle_hash": "sha256:" + "8" * 64,
            },
            "receipt": {"receipt_hash": "sha256:" + "6" * 64},
            "receipt_graph": {"root_receipt_hash": "sha256:" + "6" * 64},
        }

    monkeypatch.setattr(authority, "source_bundle_for_artifact_v2", source_bundle)
    monkeypatch.setattr(authority, "load_provider_profile_v2", load_profile)
    monkeypatch.setattr(authority, "provision_provider_profile_v2", provision)

    result = asyncio.run(
        authority.attest_stale_parent_rebase_v2(
            candidate={
                "candidate_id": "candidate-1",
                "run_id": "run-1",
                "candidate_source_diff_hash": source_diff_hash,
            },
            original_draft=draft,
            active_artifact=artifact,
            candidate_receipt_graph=candidate_graph,
            epoch_id=10,
            worker_index=12,
            require_egress_proxy=True,
            source_bundle_timeout_seconds=120,
            execute=execute,
            coordinator_client=client,
        )
    )

    assert worker_indexes == [12]
    assert result.draft == draft
    assert client.released


def test_authoritative_loop_binds_measured_provider_outcome_parent(
    tmp_path,
    monkeypatch,
):
    client = _CoordinatorClient()
    artifact = _artifact(tmp_path)
    outcome_result = ProviderOutcomeLedgerV2(
        clock=lambda: "2026-07-10T20:00:00Z"
    ).snapshot()
    outcome_authority = _artifact_wrapped_coordinator_authority(
        outcome_result,
        purpose="research_lab.provider_outcome_snapshot.v2",
    )
    outcome_graph = outcome_authority["receipt_graph"]
    outcome_receipt = outcome_authority["receipt"]
    outcome_execution_graph = outcome_authority["execution_receipt_graph"]
    outcome_execution_receipt = outcome_authority["execution_receipt"]
    active_result = {
        "schema_version": "leadpoet.active_private_model.v2",
        "artifact": private_model_artifact_replay_identity_v2(artifact),
        "active_model": {
            "private_model_version_id": "private_model_version:" + "1" * 64
        },
        "source_state_hash": "sha256:" + "2" * 64,
    }
    active_authority = _artifact_wrapped_coordinator_authority(
        active_result,
        purpose="research_lab.active_private_model.v2",
    )
    active_graph = active_authority["receipt_graph"]
    active_receipt = active_authority["receipt"]
    active_execution_graph = active_authority["execution_receipt_graph"]
    active_execution_receipt = active_authority["execution_receipt"]
    catalog = build_source_add_runtime_catalog_v2([])
    catalog_result = {
        "schema_version": "leadpoet.source_add_catalog_snapshot.v2",
        "provisioned_sources": [],
        "provisioned_sources_hash": sha256_json([]),
        "private_registry_rows": [],
        "private_registry_rows_hash": sha256_json([]),
        "runtime_catalog": catalog,
        "runtime_catalog_hash": catalog["catalog_hash"],
    }
    catalog_authority = _artifact_wrapped_coordinator_authority(
        catalog_result,
        purpose="research_lab.source_add_catalog_snapshot.v2",
    )
    catalog_hash = catalog_authority["receipt"]["receipt_hash"]
    catalog_execution_graph = catalog_authority["execution_receipt_graph"]
    catalog_execution_hash = catalog_authority["execution_receipt"][
        "receipt_hash"
    ]
    guard_queue_event_hash = "sha256:" + "8" * 64
    guard_run_state_hash = sha256_json(
        {"run_id": "run-1", "queue_event_hash": guard_queue_event_hash}
    )
    guard_result = {
        "schema_version": authority.OPENROUTER_GUARD_RESULT_SCHEMA_VERSION,
        "key_ref_hash": HASHES["key_ref_hash"],
        "miner_hotkey_hash": HASHES["miner_hotkey_hash"],
        "runtime_credential_value_hash": HASHES[
            "runtime_credential_value_hash"
        ],
        "management_credential_value_hash": HASHES[
            "management_credential_value_hash"
        ],
        "run_state_hash": guard_run_state_hash,
        "preflight_status": "passed",
        "preflight_error_type": "",
        "credit_depleted": False,
        "credit_limit_remaining": 1,
        "privacy_proof_doc": {"status": "verified"},
    }
    guard_authority = _autoresearch_guard_authority(
        guard_result,
        artifact_wrapped=True,
    )
    guard_hash = guard_authority["receipt"]["receipt_hash"]
    guard_execution_hash = guard_authority["execution_receipt"][
        "receipt_hash"
    ]
    component_registry = {"schema_version": "1.0", "components": []}
    component_result = {
        "schema_version": "leadpoet.model_sandbox_result.v2",
        "operation": "metadata",
        "output": component_registry,
        "output_hash": sha256_json(component_registry),
    }
    component_lineage_hash = "sha256:" + "9" * 64
    component_execution_hash = "sha256:" + "0" * 64
    component_lineage_graph = {
        "root_receipt_hash": component_lineage_hash
    }
    component_execution_graph = {
        "root_receipt_hash": component_execution_hash
    }
    component_authority = {
        "result": component_result,
        "receipt": {"receipt_hash": component_lineage_hash},
        "receipt_graph": component_lineage_graph,
        "execution_receipt": {
            "receipt_hash": component_execution_hash,
            "role": "gateway_scoring",
            "purpose": authority.COMPONENT_REGISTRY_EVIDENCE_PURPOSE_V2,
            "status": "succeeded",
            "output_root": sha256_json(component_result),
        },
        "execution_receipt_graph": component_execution_graph,
    }
    observed = {}

    monkeypatch.setattr(
        authority,
        "_load_release",
        lambda _path: {"release_hash": "sha256:" + "a" * 64},
    )
    monkeypatch.setattr(
        authority,
        "source_bundle_for_artifact_v2",
        lambda *_args, **_kwargs: _async_value(
            {"archive_sha256": "sha256:" + "b" * 64}
        ),
    )
    monkeypatch.setattr(
        authority,
        "load_provider_profile_v2",
        lambda *_args, **_kwargs: {"credential_ref_hashes": {}},
    )
    monkeypatch.setattr(
        authority,
        "provision_provider_profile_v2",
        lambda *_args, **_kwargs: _async_value({"status": "ready"}),
    )
    monkeypatch.setattr(
        authority,
        "load_openrouter_job_credential_envelope_v2",
        lambda **kwargs: _async_value(dict(kwargs)),
    )
    monkeypatch.setattr(
        authority,
        "provision_job_provider_envelope_v2",
        lambda *_args, **_kwargs: _async_value({"status": "ready"}),
    )

    async def load_catalog_snapshot(**_kwargs):
        return catalog_authority

    async def load_outcome_snapshot(**_kwargs):
        return outcome_authority

    async def execute(**kwargs):
        observed.update(kwargs)
        policy = TreePolicy(mode="active")
        tree_id = derive_tree_id(
            run_id="run-1",
            root_artifact_hash=artifact.model_artifact_hash,
            policy=policy,
        )
        checkpoint = TreeCheckpoint(
            tree_id=tree_id,
            root_artifact_hash=artifact.model_artifact_hash,
            policy=policy,
            nodes=(),
            frontier_hash="sha256:" + "7" * 64,
            operation_settlement_hash="sha256:" + "8" * 64,
            stop_reason="tree_final_selection_committed",
        )
        tree_result = TreeResult(
            tree_id=tree_id,
            status="failed",
            stop_reason="no_eligible_tree_finalist",
            selected_node_id="",
            nodes=(),
            checkpoint=checkpoint,
        )
        return {
            "result": {
                "schema_version": "leadpoet.autoresearch_result.v2",
                "selected_candidates": [],
                "iterations_completed": 1,
                "stop_reason": "no_eligible_tree_finalist",
                "elapsed_seconds": 1.0,
                "estimated_cost_usd": 0.5,
                "actual_openrouter_cost_usd": 0.0,
                "actual_openrouter_cost_microusd": 0,
                "openrouter_call_count": 0,
                "tree_result": tree_result.to_dict(),
                "provider_usage": [],
                "status": "failed",
                "checkpoint_doc": {
                    "git_tree_checkpoint": checkpoint.to_dict()
                },
            }
        }

    result = asyncio.run(
        authority.run_authoritative_autoresearch_v2(
            run_id="run-1",
            ticket={"ticket_id": "ticket-1"},
            artifact=artifact,
            component_registry=component_registry,
            benchmark_public_summary={},
            model_id="openai/test",
            model_doc={},
            budget_context={},
            requested_loop_count=1,
            resume_state=None,
            loop_settings=AutoResearchRuntimeSettings(
                min_seconds=0,
                max_seconds=60,
                min_iterations=1,
                max_iterations=1,
                draft_timeout_seconds=30,
                reflection_timeout_seconds=30,
                estimated_iteration_cost_usd=0.5,
                max_candidates=1,
            ),
            probe_private_window_term_hashes=(),
            openrouter_key_ref=KEY_REF,
            miner_hotkey=MINER_HOTKEY,
            openrouter_guard=authority.OpenRouterGuardAuthorityV2(
                proof_doc={"status": "verified"},
                credit_depleted=False,
                credit_limit_remaining=1,
                credential_commitments={
                    "runtime_credential_value_hash": HASHES[
                        "runtime_credential_value_hash"
                    ],
                    "management_credential_value_hash": HASHES[
                        "management_credential_value_hash"
                    ],
                },
                run_id="run-1",
                queue_event_hash=guard_queue_event_hash,
                run_state_hash=guard_run_state_hash,
                authority=guard_authority,
            ),
            component_registry_authority=component_authority,
            active_model_authority=active_authority,
            expected_event_state_hash="sha256:" + "c" * 64,
            record_loop_event=lambda _event: {},
            code_builder=SimpleNamespace(
                config=SimpleNamespace(code_edit_build_timeout_seconds=900)
            ),
            should_pause=lambda: False,
            record_privacy_proof=lambda **_kwargs: None,
            epoch_id=10,
            execute=execute,
            coordinator_client=client,
            load_catalog_snapshot=load_catalog_snapshot,
            load_provider_outcome_snapshot=load_outcome_snapshot,
        )
    )

    assert result.loop_result.status == "failed"
    assert observed["payload"]["provider_outcome_digest"] == outcome_result[
        "provider_outcome_digest"
    ]
    assert outcome_graph in observed["parent_graphs"]
    assert outcome_execution_graph in observed["parent_graphs"]
    assert component_lineage_graph in observed["parent_graphs"]
    assert component_execution_graph in observed["parent_graphs"]
    assert active_graph in observed["parent_graphs"]
    assert active_execution_graph in observed["parent_graphs"]
    assert catalog_execution_graph in observed["parent_graphs"]
    assert observed["payload"]["active_model_evidence"] == {
        "result": active_result,
        "receipt_graph": active_execution_graph,
        "root_receipt_hash": active_execution_receipt["receipt_hash"],
    }
    assert observed["payload"]["component_registry_evidence"] == {
        "result": component_result,
        "receipt_graph": component_execution_graph,
        "root_receipt_hash": component_execution_hash,
    }
    assert observed["payload"]["provider_catalog_evidence"][
        "root_receipt_hash"
    ] == catalog_execution_hash
    assert observed["payload"]["provider_outcome_evidence"][
        "root_receipt_hash"
    ] == outcome_execution_receipt["receipt_hash"]
    assert observed["payload"]["openrouter_guard_evidence"] == {
        "result": guard_result,
        "receipt_graph": guard_authority["execution_receipt_graph"],
        "root_receipt_hash": guard_execution_hash,
        "queue_event_hash": guard_queue_event_hash,
    }
    assert guard_authority["receipt_graph"] in observed["parent_graphs"]
    assert guard_authority["execution_receipt_graph"] in observed[
        "parent_graphs"
    ]
    assert guard_hash in observed["input_artifact_hashes"]
    assert guard_execution_hash in observed["input_artifact_hashes"]
    assert component_lineage_hash in observed["input_artifact_hashes"]
    assert component_execution_hash in observed["input_artifact_hashes"]
    assert active_receipt["receipt_hash"] in observed["input_artifact_hashes"]
    assert active_execution_receipt["receipt_hash"] in observed[
        "input_artifact_hashes"
    ]
    assert catalog_hash in observed["input_artifact_hashes"]
    assert catalog_execution_hash in observed["input_artifact_hashes"]
    assert outcome_receipt["receipt_hash"] in observed["input_artifact_hashes"]
    assert outcome_execution_receipt["receipt_hash"] in observed[
        "input_artifact_hashes"
    ]
    assert len(client.released) == 1


def test_tree_recovery_objects_are_kms_encrypted_read_back_and_restorable(
    tmp_path, monkeypatch
):
    import boto3

    class FakeS3:
        def __init__(self):
            self.objects = {}
            self.puts = []

        def put_object(self, **kwargs):
            assert kwargs["ServerSideEncryption"] == "aws:kms"
            assert kwargs["SSEKMSKeyId"] == "alias/test-tree-key"
            self.puts.append(dict(kwargs))
            self.objects[(kwargs["Bucket"], kwargs["Key"])] = bytes(
                kwargs["Body"]
            )
            return {"ETag": "fixture"}

        def get_object(self, **kwargs):
            return {
                "ServerSideEncryption": "aws:kms",
                "Body": io.BytesIO(
                    self.objects[(kwargs["Bucket"], kwargs["Key"])]
                ),
            }

    fake_s3 = FakeS3()
    monkeypatch.setattr(boto3, "client", lambda service: fake_s3)
    monkeypatch.setenv(
        authority.TREE_ARTIFACT_KMS_KEY_ENV, "alias/test-tree-key"
    )
    source = tmp_path / "source"
    source.mkdir()
    (source / "research_lab_adapter.py").write_text(
        "def run():\n    return []\n", encoding="utf-8"
    )
    policy = TreePolicy(mode="active")
    tree_id = derive_tree_id(
        run_id="run-authority-recovery",
        root_artifact_hash="sha256:" + "a" * 64,
        policy=policy,
    )
    workspace = tmp_path / "tree"
    repository = GitTreeRepository(workspace=workspace, tree_id=tree_id)
    root_commit = repository.initialize(
        source_root=source,
        root_artifact_hash="sha256:" + "a" * 64,
        policy_hash=policy.policy_hash,
    )
    checkpoint_doc = {"tree_id": tree_id, "frontier": []}
    checkpoint_hash = sha256_json(checkpoint_doc)
    repository.commit_checkpoint(
        checkpoint_hash=checkpoint_hash,
        checkpoint_doc=checkpoint_doc,
    )

    descriptor = asyncio.run(
        authority._publish_tree_recovery(
            repository=repository,
            tree_id=tree_id,
            checkpoint_hash=checkpoint_hash,
            manifest_uri="s3://private-bucket/manifests/current.json",
        )
    )
    recovery_state, bundle_bytes = asyncio.run(
        authority._load_tree_recovery(
            descriptor=descriptor,
            expected_tree_id=tree_id,
        )
    )
    assert descriptor["kms_encrypted"] is True
    assert len(fake_s3.puts) == 2

    shutil.rmtree(workspace)
    bundle_path = tmp_path / "restored.bundle"
    bundle_path.write_bytes(bundle_bytes)
    restored = GitTreeRepository(workspace=workspace, tree_id=tree_id)
    assert restored.restore_recovery_state(
        recovery_state=recovery_state,
        bundle_path=bundle_path,
    ) == root_commit
    assert restored.state_status() == "complete"


def test_tree_recovery_rejects_unencrypted_s3_readback(monkeypatch):
    import boto3

    monkeypatch.setattr(
        boto3,
        "client",
        lambda _service: SimpleNamespace(
            get_object=lambda **_kwargs: {
                "ServerSideEncryption": "AES256",
                "Body": io.BytesIO(b"private"),
            }
        ),
    )
    with pytest.raises(authority.AutoresearchAuthorityV2Error, match="SSE-KMS"):
        authority._read_tree_object(
            uri="s3://private-bucket/object",
            content_hash=sha256_bytes(b"private"),
            size_bytes=len(b"private"),
        )


def test_tree_final_selection_requires_artifact_and_lineage_authority():
    tree_id = "sha256:" + "1" * 64
    selection = {
        "schema_version": "research_lab.git_tree_selection.v1",
        "tree_id": tree_id,
        "selected_node_id": "tree-node:" + "2" * 64,
        "selected_candidate_artifact_hash": "sha256:" + "3" * 64,
        "selected_node_git_commit": "4" * 64,
        "selected_lineage_hash": "sha256:" + "5" * 64,
        "paid_finalist_count": 1,
    }
    assert authority._validated_tree_final_selection(
        selection,
        expected_tree_id=tree_id,
        expected_selection_hash=sha256_json(selection),
    ) == selection["selected_node_id"]

    for field in (
        "selected_candidate_artifact_hash",
        "selected_node_git_commit",
        "selected_lineage_hash",
    ):
        with pytest.raises(
            authority.AutoresearchAuthorityV2Error,
            match="selection authority is incomplete",
        ):
            authority._validated_tree_final_selection(
                {**selection, field: ""},
                expected_tree_id=tree_id,
                expected_selection_hash=sha256_json({**selection, field: ""}),
            )

    with pytest.raises(
        authority.AutoresearchAuthorityV2Error,
        match="selection authority is incomplete",
    ):
        authority._validated_tree_final_selection(
            selection,
            expected_tree_id=tree_id,
            expected_selection_hash="sha256:" + "0" * 64,
        )


@pytest.mark.asyncio
async def test_tree_created_event_recovers_row_insert_crash_idempotently():
    tree_id = "sha256:" + "1" * 64

    class FakeTreeStore:
        def __init__(self):
            self.current = {"tree_id": tree_id, "current_event_hash": None}
            self.events = []

        async def get_tree_current(self, *, tree_id):
            return dict(self.current)

        async def append_event_next(self, **kwargs):
            self.events.append(dict(kwargs))
            self.current["current_event_hash"] = "sha256:" + "2" * 64

    store = FakeTreeStore()
    event_doc = {
        "schema_version": "research_lab.git_tree_created.v1",
        "tree_id": tree_id,
    }

    await authority._ensure_tree_created_event(
        tree_store=store,
        tree_id=tree_id,
        event_doc=event_doc,
    )
    await authority._ensure_tree_created_event(
        tree_store=store,
        tree_id=tree_id,
        event_doc=event_doc,
    )

    assert store.events == [
        {
            "tree_id": tree_id,
            "event_type": "tree_created",
            "event_doc": event_doc,
        }
    ]


@pytest.mark.asyncio
async def test_tree_created_event_accepts_concurrent_winner_after_cas_conflict():
    tree_id = "sha256:" + "1" * 64

    class FakeTreeStore:
        def __init__(self):
            self.current = {"tree_id": tree_id, "current_event_hash": None}

        async def get_tree_current(self, *, tree_id):
            return dict(self.current)

        async def append_event_next(self, **kwargs):
            self.current["current_event_hash"] = "sha256:" + "2" * 64
            raise RuntimeError("research_lab_git_tree_event_identity_conflict")

    await authority._ensure_tree_created_event(
        tree_store=FakeTreeStore(),
        tree_id=tree_id,
        event_doc={
            "schema_version": "research_lab.git_tree_created.v1",
            "tree_id": tree_id,
        },
    )


async def _async_value(value):
    return value
