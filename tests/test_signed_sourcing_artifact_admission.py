from __future__ import annotations

from copy import deepcopy
from io import BytesIO
import json
from dataclasses import replace

import pytest

from gateway.qualification.company_fit_proof_receipt import (
    company_fit_proof_receipt_contract_identity,
)
from gateway.research_lab.official_baseline_model_runner import (
    EXACT_MODEL_RUNNER_FAMILY,
    OFFICIAL_BASELINE_EXECUTION_SCHEMA_VERSION,
    OfficialBaselineReleaseSelectionError,
)
from research_lab.canonical import sha256_json
from research_lab.eval import private_runtime
from research_lab.eval.private_runtime import (
    QUALIFICATION_OUTCOME_CONTRACT_SHA256_V2,
)
from scripts import verify_signed_sourcing_artifact_admission as admission
from tests.test_qualification_outcome_protocol_v2 import (
    _ready_v2_adapter_metadata,
)


def _manifest_document(*, image_digest: str | None = None) -> dict:
    commit = "1" * 40
    release_identity = {"schema_version": "fixture-release:v1"}
    payload = {
        "model_artifact_hash": "sha256:" + "2" * 64,
        "git_commit_sha": commit,
        "image_digest": image_digest
        or (
            "493765492819.dkr.ecr.us-east-1.amazonaws.com/"
            "leadpoet/sourcing-model@sha256:" + "3" * 64
        ),
        "config_hash": "sha256:" + "4" * 64,
        "component_registry_version": "components:v1",
        "scoring_adapter_version": "scoring:v1",
        "manifest_uri": (
            "s3://leadpoet-private-model-artifacts-493765492819/"
            f"research-lab/sourcing-model/{commit}.json"
        ),
        "signature_ref": (
            "s3://leadpoet-private-model-artifacts-493765492819/"
            f"research-lab/sourcing-model/{commit}.sig.b64"
        ),
        "build_id": "artifact-" + commit,
        "model_release_identity": release_identity,
        "official_baseline_execution": {
            "schema_version": OFFICIAL_BASELINE_EXECUTION_SCHEMA_VERSION,
            "runner_family": EXACT_MODEL_RUNNER_FAMILY,
            "execution_mode": "measured_lab",
            "release_identity_sha256": sha256_json(release_identity),
            "protocol_generation_sha256": "sha256:" + "5" * 64,
            "benchmark_projection_sha256": "sha256:" + "6" * 64,
            "protected_action_authority_sha256": "sha256:" + "7" * 64,
        },
    }
    return {**payload, "manifest_hash": sha256_json(payload)}


def _legacy_manifest_document() -> tuple[dict, dict]:
    document = _manifest_document()
    payload = {
        key: value
        for key, value in document.items()
        if key
        not in {
            "manifest_hash",
            "model_release_identity",
            "official_baseline_execution",
        }
    }
    metadata = {
        "adapter_version": "sourcing-model-research-lab-adapter:v9",
        "component_registry_version": "sourcing-model-components:v2",
        "scoring_adapter_version": "qualification-company-scorer:v2",
    }
    payload.update(
        {
            "config_hash": sha256_json(metadata),
            "component_registry_version": metadata[
                "component_registry_version"
            ],
            "scoring_adapter_version": metadata["scoring_adapter_version"],
            "compatibility_contract": {
                "contract_id": "leadpoet-sourcing-wrapper-contract-v68",
                "path": "sourcing_model/consumer_contract.json",
                "sha256": "sha256:" + "8" * 64,
            },
            "consumer_parity_fixtures": {
                "path": "sourcing_model/consumer_parity_fixtures.json",
                "sha256": "sha256:" + "9" * 64,
            },
            "dependency_lock": {
                "file": ".research-lab-requirements.lock",
                "sha256": "a" * 64,
            },
            "qualification_outcome_contract": {
                "protocol_id": "sourcing-model.qualification-outcome",
                "path": "sourcing_model/qualification_outcome_contract_v2.json",
                "sha256": "sha256:" + "b" * 64,
                "contract_sha256": QUALIFICATION_OUTCOME_CONTRACT_SHA256_V2,
            },
            "intent_release_benchmark": {
                "contract": {
                    "contract_id": "intent-release-contracts:v1",
                    "path": "sourcing_model/intent_release_contract_v1.json",
                    "sha256": "sha256:" + "c" * 64,
                },
                "policy": {
                    "policy_id": "intent-release-policy:v1",
                    "path": "sourcing_model/intent_release_policy_v1.json",
                    "sha256": "sha256:" + "d" * 64,
                    "payload_sha256": "sha256:" + "e" * 64,
                },
            },
            "facility_evidence_contract": {
                "contract_id": "facility-evidence:v1",
                "path": "sourcing_model/facility_evidence_contract_v1.json",
                "sha256": "sha256:" + "f" * 64,
                "identity_policy": {
                    "policy_version": "facility-identity-proof:v2",
                    "sha256": "0" * 64,
                },
            },
        }
    )
    return {**payload, "manifest_hash": sha256_json(payload)}, metadata


def _compatible_v9_metadata() -> dict:
    metadata = deepcopy(_ready_v2_adapter_metadata())
    metadata["adapter_version"] = "sourcing-model-research-lab-adapter:v9"
    metadata["scoring_adapter_version"] = (
        "qualification-company-scorer:v2"
    )
    metadata["company_fit_proof_receipt"] = (
        company_fit_proof_receipt_contract_identity()
    )
    return metadata


class _S3:
    def __init__(self, pointer: bytes, archive: bytes | None = None):
        self.pointer = pointer
        self.archive = pointer if archive is None else archive

    def get_object(self, *, Bucket, Key):  # noqa: N803
        assert Bucket == admission.ARTIFACT_BUCKET
        value = self.pointer if Key.endswith("current.json") else self.archive
        return {"Body": BytesIO(value)}


class _PromotingS3(_S3):
    def __init__(self, pointer: bytes, promoted: bytes):
        super().__init__(pointer)
        self.promoted = promoted
        self.pointer_reads = 0

    def get_object(self, *, Bucket, Key):  # noqa: N803
        if Key.endswith("current.json"):
            self.pointer_reads += 1
            value = self.pointer if self.pointer_reads == 1 else self.promoted
            return {"Body": BytesIO(value)}
        return super().get_object(Bucket=Bucket, Key=Key)


def test_current_signed_artifact_uses_exact_no_spend_production_preflight(
):
    document = _manifest_document()
    raw = (json.dumps(document, sort_keys=True) + "\n").encode()
    observed = {}

    def preflight(*, artifact, selection, spec):
        observed.update(
            artifact=artifact,
            selection=selection,
            spec=spec,
        )

    receipt = admission.admit_current_signed_artifact(
        s3_client=_S3(raw),
        signature_verifier=lambda *_args, **_kwargs: {"verified": True},
        protocol_preflight=preflight,
        legacy_protocol_preflight=lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("exact artifact used legacy preflight")
        ),
    )

    assert receipt["schema_version"] == (
        "leadpoet.signed_sourcing_artifact_admission.v2"
    )
    assert receipt["status"] == "passed"
    assert receipt["runner_family"] == EXACT_MODEL_RUNNER_FAMILY
    assert receipt["protocol_generation_sha256"] == "sha256:" + "5" * 64
    assert receipt["preflight_mode"] == "exact_model_runner_protocol"
    assert "legacy_release_sha256" not in receipt
    assert receipt["network"] == "none"
    assert receipt["provider_credentials_forwarded"] is False
    assert receipt["pointer_stable_through_admission"] is True
    assert observed["selection"].is_exact
    assert observed["spec"].network_disabled is True
    assert observed["spec"].env_passthrough == ()
    assert observed["spec"].extra_env == {}
    assert observed["spec"].image_digest == document["image_digest"]


def test_current_producer_shaped_legacy_artifact_uses_network_none_metadata(
    monkeypatch,
):
    document, _config_metadata = _legacy_manifest_document()
    metadata = _compatible_v9_metadata()
    raw = (json.dumps(document, sort_keys=True) + "\n").encode()
    observed = {}

    class _Runner:
        def __init__(self, spec):
            observed["spec"] = spec

        def metadata(self):
            return dict(metadata)

    monkeypatch.setattr(admission, "DockerPrivateModelRunner", _Runner)

    receipt = admission.admit_current_signed_artifact(
        s3_client=_S3(raw),
        signature_verifier=lambda *_args, **_kwargs: {"verified": True},
        protocol_preflight=lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("legacy artifact used exact preflight")
        ),
    )

    assert receipt["status"] == "passed"
    assert receipt["runner_family"] == "attested_private_model:v2"
    assert receipt["preflight_mode"] == "network_none_container_metadata"
    assert receipt["legacy_release_sha256"].startswith("sha256:")
    assert receipt["container_metadata_sha256"] == sha256_json(metadata)
    assert "protocol_generation_sha256" not in receipt
    assert observed["spec"].network_disabled is True
    assert observed["spec"].env_passthrough == ()
    assert observed["spec"].extra_env == {}


@pytest.mark.parametrize(
    "spec_changes",
    (
        {"network_disabled": False},
        {"env_passthrough": ("AWS_ACCESS_KEY_ID",)},
        {"extra_env": {"OPENROUTER_API_KEY": "forbidden"}},
    ),
)
def test_default_legacy_preflight_rejects_unsafe_runtime_context(
    monkeypatch,
    spec_changes,
):
    document, _metadata = _legacy_manifest_document()
    artifact = admission.PrivateModelArtifactManifest.from_mapping(document)
    selection = admission.select_official_baseline_release(artifact)
    spec = admission.DockerPrivateModelSpec(
        image_digest=artifact.image_digest,
        env_passthrough=(),
        extra_env={},
        network_disabled=True,
    )
    spec = replace(spec, **spec_changes)
    monkeypatch.setattr(
        admission,
        "DockerPrivateModelRunner",
        lambda _spec: (_ for _ in ()).throw(
            AssertionError("unsafe legacy context reached the container")
        ),
    )

    with pytest.raises(
        admission.SignedArtifactAdmissionError,
        match="preflight context differs",
    ):
        admission._preflight_legacy_signed_artifact(
            artifact=artifact,
            selection=selection,
            spec=spec,
        )


def test_default_legacy_preflight_rejects_family_mismatch(monkeypatch):
    document, _metadata = _legacy_manifest_document()
    artifact = admission.PrivateModelArtifactManifest.from_mapping(document)
    selection = admission.select_official_baseline_release(artifact)
    wrong_selection = replace(
        selection,
        runner_family=EXACT_MODEL_RUNNER_FAMILY,
    )
    spec = admission.DockerPrivateModelSpec(
        image_digest=artifact.image_digest,
        env_passthrough=(),
        extra_env={},
        network_disabled=True,
    )
    monkeypatch.setattr(
        admission,
        "DockerPrivateModelRunner",
        lambda _spec: (_ for _ in ()).throw(
            AssertionError("wrong family reached the container")
        ),
    )

    with pytest.raises(
        admission.SignedArtifactAdmissionError,
        match="preflight context differs",
    ):
        admission._preflight_legacy_signed_artifact(
            artifact=artifact,
            selection=wrong_selection,
            spec=spec,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("adapter_version", "sourcing-model-research-lab-adapter:v10"),
        ("component_registry_version", "sourcing-model-components:v3"),
        ("scoring_adapter_version", "qualification-company-scorer:v1"),
    ),
)
def test_legacy_container_configuration_drift_fails_closed(
    monkeypatch,
    field,
    value,
):
    document, metadata = _legacy_manifest_document()
    raw = (json.dumps(document, sort_keys=True) + "\n").encode()
    mutated = {**metadata, field: value}
    monkeypatch.setattr(
        admission,
        "validate_sourcing_adapter_metadata",
        lambda candidate, **_kwargs: dict(candidate),
    )

    with pytest.raises(
        admission.SignedArtifactAdmissionError,
        match="container configuration differs",
    ):
        admission.admit_current_signed_artifact(
            s3_client=_S3(raw),
            signature_verifier=lambda *_args, **_kwargs: {"verified": True},
            protocol_preflight=lambda **_kwargs: None,
            legacy_protocol_preflight=lambda **_kwargs: mutated,
        )


def test_legacy_preflight_requires_mapping_output():
    document, _metadata = _legacy_manifest_document()
    raw = (json.dumps(document, sort_keys=True) + "\n").encode()

    with pytest.raises(
        admission.SignedArtifactAdmissionError,
        match="container metadata is unavailable",
    ):
        admission.admit_current_signed_artifact(
            s3_client=_S3(raw),
            signature_verifier=lambda *_args, **_kwargs: {"verified": True},
            protocol_preflight=lambda **_kwargs: None,
            legacy_protocol_preflight=lambda **_kwargs: None,
        )


def test_mixed_or_malformed_legacy_release_fails_closed():
    mixed, _metadata = _legacy_manifest_document()
    mixed["model_release_identity"] = {"schema_version": "forged:v1"}
    mixed["manifest_hash"] = sha256_json(
        {key: value for key, value in mixed.items() if key != "manifest_hash"}
    )
    raw = (json.dumps(mixed, sort_keys=True) + "\n").encode()
    with pytest.raises(
        OfficialBaselineReleaseSelectionError,
        match="missing official baseline selection",
    ):
        admission.admit_current_signed_artifact(
            s3_client=_S3(raw),
            signature_verifier=lambda *_args, **_kwargs: {"verified": True},
        )

    malformed, _metadata = _legacy_manifest_document()
    malformed["consumer_parity_fixtures"]["path"] = ""
    malformed["manifest_hash"] = sha256_json(
        {
            key: value
            for key, value in malformed.items()
            if key != "manifest_hash"
        }
    )
    raw = (json.dumps(malformed, sort_keys=True) + "\n").encode()
    with pytest.raises(
        OfficialBaselineReleaseSelectionError,
        match="legacy release identity is invalid",
    ):
        admission.admit_current_signed_artifact(
            s3_client=_S3(raw),
            signature_verifier=lambda *_args, **_kwargs: {"verified": True},
        )


def test_pointer_promotion_during_preflight_invalidates_admission():
    document = _manifest_document()
    pointer = (json.dumps(document, sort_keys=True) + "\n").encode()
    promoted_document = _manifest_document(
        image_digest=(
            "493765492819.dkr.ecr.us-east-1.amazonaws.com/"
            "leadpoet/sourcing-model@sha256:" + "6" * 64
        )
    )
    promoted = (json.dumps(promoted_document, sort_keys=True) + "\n").encode()
    with pytest.raises(
        admission.SignedArtifactAdmissionError,
        match="changed during artifact admission",
    ):
        admission.admit_current_signed_artifact(
            s3_client=_PromotingS3(pointer, promoted),
            signature_verifier=lambda *_args, **_kwargs: {"verified": True},
            protocol_preflight=lambda **_kwargs: None,
        )


def test_current_pointer_must_equal_immutable_archive_bytes():
    document = _manifest_document()
    pointer = (json.dumps(document, sort_keys=True) + "\n").encode()
    archive = json.dumps(document, sort_keys=True).encode()
    with pytest.raises(
        admission.SignedArtifactAdmissionError,
        match="differs byte-for-byte",
    ):
        admission.admit_current_signed_artifact(
            s3_client=_S3(pointer, archive),
            signature_verifier=lambda *_args, **_kwargs: {"verified": True},
            protocol_preflight=lambda **_kwargs: None,
        )


@pytest.mark.parametrize(
    "field,value,error",
    (
        (
            "manifest_uri",
            "s3://leadpoet-private-model-artifacts-493765492819/"
            "research-lab/sourcing-model/branches/leadpoet-lab/current.json",
            "immutable release path",
        ),
        (
            "signature_ref",
            "s3://different/research-lab/sourcing-model/" + "1" * 40 + ".sig.b64",
            "signature URI",
        ),
        (
            "image_digest",
            "493765492819.dkr.ecr.us-east-1.amazonaws.com/other@sha256:"
            + "3" * 64,
            "production ECR repository",
        ),
    ),
)
def test_artifact_location_authority_is_closed(field, value, error):
    document = _manifest_document()
    document[field] = value
    document["manifest_hash"] = sha256_json({
        key: item for key, item in document.items() if key != "manifest_hash"
    })
    artifact = admission.PrivateModelArtifactManifest.from_mapping(document)

    with pytest.raises(admission.SignedArtifactAdmissionError, match=error):
        admission._validate_artifact_locations(artifact)


def test_network_disabled_runner_passes_no_ambient_credentials(
    tmp_path,
    monkeypatch,
):
    commands = []

    def run(command, **_kwargs):
        if list(command)[-1:] == ["info"]:
            return private_runtime.subprocess.CompletedProcess(
                command, 0, stdout="", stderr=""
            )
        commands.append(list(command))
        return private_runtime.subprocess.CompletedProcess(
            command, 0, stdout="{}", stderr=""
        )

    monkeypatch.setenv(
        "LEADPOET_DOCKER_OPERATION_LOCK_FILE",
        str(tmp_path / "docker.lock"),
    )
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "must-not-enter-container")
    monkeypatch.setenv("OPENROUTER_API_KEY", "must-not-enter-container")
    monkeypatch.setenv("DOCKER_CONFIG", str(tmp_path / "docker-config"))
    monkeypatch.setattr(private_runtime.subprocess, "run", run)
    runner = private_runtime.DockerPrivateModelRunner(
        private_runtime.DockerPrivateModelSpec(
            image_digest=(
                "493765492819.dkr.ecr.us-east-1.amazonaws.com/"
                "leadpoet/sourcing-model@sha256:" + "3" * 64
            ),
            env_passthrough=(),
            extra_env={},
            pull_before_run=False,
            network_disabled=True,
        )
    )
    runner._run_json(
        bootstrap="pass",
        argv=("research_lab_adapter", "runner_protocol_generation"),
        stdin_payload={},
    )

    command = commands[-1]
    assert command[command.index("--network") + 1] == "none"
    container_env = [
        command[index + 1]
        for index, item in enumerate(command[:-1])
        if item == "-e"
    ]
    assert not any("AWS_" in item for item in container_env)
    assert not any("OPENROUTER" in item for item in container_env)
