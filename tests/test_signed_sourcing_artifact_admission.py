from __future__ import annotations

from io import BytesIO
import json
from types import SimpleNamespace

import pytest

from research_lab.canonical import sha256_json
from research_lab.eval import private_runtime
from scripts import verify_signed_sourcing_artifact_admission as admission


def _manifest_document(*, image_digest: str | None = None) -> dict:
    commit = "1" * 40
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
        "model_release_identity": {"schema_version": "fixture:v1"},
        "official_baseline_execution": {"schema_version": "fixture:v1"},
    }
    return {**payload, "manifest_hash": sha256_json(payload)}


class _S3:
    def __init__(self, pointer: bytes, archive: bytes | None = None):
        self.pointer = pointer
        self.archive = pointer if archive is None else archive

    def get_object(self, *, Bucket, Key):  # noqa: N803
        assert Bucket == admission.ARTIFACT_BUCKET
        value = self.pointer if Key.endswith("current.json") else self.archive
        return {"Body": BytesIO(value)}


def test_current_signed_artifact_uses_exact_no_spend_production_preflight(
    monkeypatch,
):
    document = _manifest_document()
    raw = (json.dumps(document, sort_keys=True) + "\n").encode()
    selection = SimpleNamespace(
        is_exact=True,
        selection_document={
            "protocol_generation_sha256": "sha256:" + "5" * 64,
        },
    )
    observed = {}

    monkeypatch.setattr(
        admission,
        "select_official_baseline_release",
        lambda artifact: selection,
    )

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
    )

    assert receipt["status"] == "passed"
    assert receipt["network"] == "none"
    assert receipt["provider_credentials_forwarded"] is False
    assert observed["selection"] is selection
    assert observed["spec"].network_disabled is True
    assert observed["spec"].env_passthrough == ()
    assert observed["spec"].extra_env == {}
    assert observed["spec"].image_digest == document["image_digest"]


def test_current_pointer_must_equal_immutable_archive_bytes(monkeypatch):
    document = _manifest_document()
    pointer = (json.dumps(document, sort_keys=True) + "\n").encode()
    archive = json.dumps(document, sort_keys=True).encode()
    monkeypatch.setattr(
        admission,
        "select_official_baseline_release",
        lambda artifact: SimpleNamespace(selection_document={}),
    )

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
