from __future__ import annotations

import base64
import json
from io import BytesIO
from pathlib import Path
from typing import Any

import pytest

import gateway.research_lab.promotion as promotion
from research_lab.canonical import sha256_bytes, sha256_json
from research_lab.eval import (
    DEFAULT_PRIVATE_MODEL_ARTIFACT_SIGNING_KMS_KEY_ID,
    PrivateModelArtifactManifest,
    PrivateModelRuntimeError,
    build_local_private_artifact_manifest,
    validate_private_model_artifact_manifest,
    verify_private_artifact_manifest_signature,
)
from research_lab.sourcing_model_contract_check import reviewed_consumer_snapshots
from tests.private_model_artifact_fixtures import install_reviewed_consumer_snapshot


class FakeS3:
    def __init__(self, signature: bytes = b"der-signature") -> None:
        self.signature = signature
        self.calls: list[dict[str, Any]] = []

    def get_object(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        return {
            "Body": BytesIO(base64.b64encode(self.signature)),
        }


class FakeKms:
    def __init__(self, *, valid: bool = True) -> None:
        self.valid = valid
        self.calls: list[dict[str, Any]] = []

    def verify(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        return {
            "SignatureValid": self.valid,
            "KeyId": "arn:aws:kms:us-east-1:493765492819:key/test",
            "SigningAlgorithm": "ECDSA_SHA_256",
        }


RESEARCH_LAB_ROOT = Path(__file__).resolve().parents[1] / "research_lab"
CONSUMER_CONTRACT_ID = json.loads(
    (RESEARCH_LAB_ROOT / "sourcing_model_contract.json").read_text(
        encoding="utf-8"
    )
)["contract_id"]


def _consumer_manifest_pair(contract_id: str) -> tuple[dict[str, str], dict[str, str]]:
    snapshot = reviewed_consumer_snapshots()[contract_id]
    contract = snapshot["contract"]
    return (
        {
            "contract_id": contract_id,
            "path": str(contract["canonical_path"]),
            "sha256": sha256_bytes(Path(snapshot["contract_path"]).read_bytes()),
        },
        {
            "path": str(contract["parity_fixture_path"]),
            "sha256": sha256_bytes(Path(snapshot["parity_path"]).read_bytes()),
        },
    )


def artifact_mapping(
    *,
    consumer_contract_id: str = CONSUMER_CONTRACT_ID,
    **overrides: Any,
) -> dict[str, Any]:
    compatibility_contract, consumer_parity_fixtures = _consumer_manifest_pair(
        consumer_contract_id
    )
    payload = {
        "model_artifact_hash": "sha256:" + "1" * 64,
        "git_commit_sha": "2" * 40,
        "image_digest": (
            "493765492819.dkr.ecr.us-east-1.amazonaws.com/"
            "leadpoet/sourcing-model@sha256:" + "3" * 64
        ),
        "config_hash": "sha256:" + "4" * 64,
        "component_registry_version": "components:v1",
        "scoring_adapter_version": "scorer:v1",
        "compatibility_contract": compatibility_contract,
        "consumer_parity_fixtures": consumer_parity_fixtures,
        "manifest_uri": "s3://artifacts/model.json",
        "signature_ref": "s3://artifacts/model.sig.b64",
        "build_id": "build-1",
    }
    payload.update(overrides)
    return {
        **payload,
        "manifest_hash": sha256_json(payload),
    }


def _build_manifest_for_source(source: Path, **kwargs: Any) -> dict[str, Any]:
    (source / "research_lab_adapter.py").write_text(
        "def run_icp(icp, context=None):\n    return []\n",
        encoding="utf-8",
    )
    return build_local_private_artifact_manifest(
        source_path=source,
        git_commit_sha="a" * 40,
        image_digest=(
            "123456789012.dkr.ecr.us-east-1.amazonaws.com/private@sha256:"
            + "b" * 64
        ),
        manifest_uri="s3://private/manifests/model.json",
        signature_ref="s3://private/manifests/model.sig.b64",
        component_registry_version="1",
        scoring_adapter_version="1",
        **kwargs,
    )


def test_kms_verifies_the_final_manifest_hash() -> None:
    manifest = PrivateModelArtifactManifest.from_mapping(artifact_mapping())
    s3 = FakeS3()
    kms = FakeKms()

    result = verify_private_artifact_manifest_signature(
        manifest,
        s3_client=s3,
        kms_client=kms,
    )

    assert result["verified"] is True
    assert s3.calls == [{"Bucket": "artifacts", "Key": "model.sig.b64"}]
    assert kms.calls == [
        {
            "KeyId": DEFAULT_PRIVATE_MODEL_ARTIFACT_SIGNING_KMS_KEY_ID,
            "Message": manifest.manifest_hash.encode("utf-8"),
            "MessageType": "RAW",
            "Signature": b"der-signature",
            "SigningAlgorithm": "ECDSA_SHA_256",
        }
    ]


def test_signed_manifest_extensions_round_trip_without_hash_drift() -> None:
    intent_release_benchmark = {
        "contract": {
            "contract_id": "intent-release-contracts:v1",
            "path": "sourcing_model/intent_release_contract_v1.json",
            "sha256": "sha256:" + "5" * 64,
        },
        "policy": {
            "policy_id": "intent-release-policy:v1",
            "path": "sourcing_model/intent_release_policy_v1.json",
            "sha256": "sha256:" + "6" * 64,
            "payload_sha256": "sha256:" + "7" * 64,
        },
    }
    facility_evidence_contract = {
        "contract_id": "facility-evidence:v1",
        "path": "sourcing_model/facility_evidence_contract_v1.json",
        "sha256": "sha256:" + "8" * 64,
        "identity_policy": {
            "policy_version": "facility-identity-proof:v2",
            "sha256": "9" * 64,
        },
    }
    value = artifact_mapping(
        intent_release_benchmark=intent_release_benchmark,
        facility_evidence_contract=facility_evidence_contract,
    )

    manifest = PrivateModelArtifactManifest.from_mapping(value)

    assert manifest.to_dict() == value
    assert validate_private_model_artifact_manifest(manifest) == []


def test_signed_model_contract_section_tampering_fails_hash_validation() -> None:
    value = artifact_mapping(
        intent_release_benchmark={
            "contract": {
                "contract_id": "intent-release-contracts:v1",
                "sha256": "sha256:" + "5" * 64,
            }
        }
    )
    value["intent_release_benchmark"]["contract"]["sha256"] = (
        "sha256:" + "6" * 64
    )

    errors = validate_private_model_artifact_manifest(
        PrivateModelArtifactManifest.from_mapping(value)
    )

    assert errors == ["manifest_hash_mismatch"]


def test_signed_manifest_extensions_remain_subject_to_secret_scanning() -> None:
    value = artifact_mapping(
        future_contract={"raw_secret": "must-not-enter-a-signed-manifest"}
    )

    errors = validate_private_model_artifact_manifest(
        PrivateModelArtifactManifest.from_mapping(value)
    )

    assert errors == ["artifact_manifest_contains_raw_secret_material"]


@pytest.mark.parametrize(
    "contract_id",
    tuple(sorted(reviewed_consumer_snapshots())),
)
def test_each_reviewed_contract_pair_verifies(contract_id: str) -> None:
    result = verify_private_artifact_manifest_signature(
        artifact_mapping(consumer_contract_id=contract_id),
        s3_client=FakeS3(),
        kms_client=FakeKms(),
    )

    assert result["verified"] is True


def test_oldest_newest_oldest_pointer_transition_remains_exact_and_rollback_safe() -> None:
    snapshots = reviewed_consumer_snapshots()
    v7 = next(item for item in snapshots if item.endswith("v7"))
    v11 = next(item for item in snapshots if item.endswith("v11"))

    for contract_id in (v7, v11, v7):
        result = verify_private_artifact_manifest_signature(
            artifact_mapping(consumer_contract_id=contract_id),
            s3_client=FakeS3(),
            kms_client=FakeKms(),
        )
        assert result["verified"] is True


def test_reviewed_contract_cannot_be_paired_with_other_version_fixtures() -> None:
    snapshots = reviewed_consumer_snapshots()
    v7 = next(item for item in snapshots if item.endswith("v7"))
    v8 = next(item for item in snapshots if item.endswith("v8"))
    manifest = artifact_mapping(consumer_contract_id=v7)
    _contract, v8_fixtures = _consumer_manifest_pair(v8)
    manifest["consumer_parity_fixtures"] = v8_fixtures
    payload = dict(manifest)
    payload.pop("manifest_hash")
    manifest["manifest_hash"] = sha256_json(payload)
    s3 = FakeS3()
    kms = FakeKms()

    with pytest.raises(
        PrivateModelRuntimeError,
        match="parity fixtures differ",
    ):
        verify_private_artifact_manifest_signature(
            manifest,
            s3_client=s3,
            kms_client=kms,
        )

    assert s3.calls == []
    assert kms.calls == []


@pytest.mark.parametrize(
    "contract_id",
    tuple(sorted(reviewed_consumer_snapshots())),
)
def test_local_manifest_builder_derives_exact_source_pair(
    tmp_path: Path,
    contract_id: str,
) -> None:
    source = tmp_path / contract_id
    source.mkdir()
    install_reviewed_consumer_snapshot(source, contract_id=contract_id)

    manifest = _build_manifest_for_source(source)
    expected_contract, expected_fixtures = _consumer_manifest_pair(contract_id)

    assert manifest["compatibility_contract"] == expected_contract
    assert manifest["consumer_parity_fixtures"] == expected_fixtures


def test_local_manifest_builder_rejects_hybrid_source_pair(tmp_path: Path) -> None:
    snapshots = reviewed_consumer_snapshots()
    v7 = next(item for item in snapshots if item.endswith("v7"))
    v8 = next(item for item in snapshots if item.endswith("v8"))
    source = tmp_path / "hybrid"
    source.mkdir()
    install_reviewed_consumer_snapshot(source, contract_id=v7)
    v7_contract = snapshots[v7]["contract"]
    source_parity = source / str(v7_contract["parity_fixture_path"])
    source_parity.write_bytes(Path(snapshots[v8]["parity_path"]).read_bytes())

    with pytest.raises(
        PrivateModelRuntimeError,
        match="no reviewed contract/parity pair",
    ):
        _build_manifest_for_source(source)


def test_local_manifest_builder_rejects_unknown_source_pair(tmp_path: Path) -> None:
    source = tmp_path / "unknown"
    source.mkdir()
    (source / "sourcing_model").mkdir()
    (source / "sourcing_model" / "consumer_contract.json").write_text(
        json.dumps(
            {
                "contract_id": "leadpoet-sourcing-wrapper-contract-v999",
                "consumer_parity_fixture_path": (
                    "sourcing_model/consumer_parity_fixtures.json"
                ),
            }
        ),
        encoding="utf-8",
    )
    (source / "sourcing_model" / "consumer_parity_fixtures.json").write_text(
        "{}\n",
        encoding="utf-8",
    )

    with pytest.raises(
        PrivateModelRuntimeError,
        match="no reviewed contract/parity pair",
    ):
        _build_manifest_for_source(source)


def test_local_manifest_builder_rejects_requested_version_mismatch(
    tmp_path: Path,
) -> None:
    snapshots = reviewed_consumer_snapshots()
    v7 = next(item for item in snapshots if item.endswith("v7"))
    v8 = next(item for item in snapshots if item.endswith("v8"))
    source = tmp_path / "v7"
    source.mkdir()
    install_reviewed_consumer_snapshot(source, contract_id=v7)

    with pytest.raises(
        PrivateModelRuntimeError,
        match="differs from requested contract id",
    ):
        _build_manifest_for_source(source, consumer_contract_id=v8)


def test_invalid_kms_signature_fails_closed() -> None:
    with pytest.raises(
        PrivateModelRuntimeError,
        match="KMS signature was rejected",
    ):
        verify_private_artifact_manifest_signature(
            artifact_mapping(),
            s3_client=FakeS3(),
            kms_client=FakeKms(valid=False),
        )


def test_manifest_payload_must_match_the_signed_hash() -> None:
    manifest = artifact_mapping()
    manifest["build_id"] = "tampered-after-signing"
    s3 = FakeS3()
    kms = FakeKms()

    with pytest.raises(
        PrivateModelRuntimeError,
        match="manifest hash does not match its payload",
    ):
        verify_private_artifact_manifest_signature(
            manifest,
            s3_client=s3,
            kms_client=kms,
        )

    assert s3.calls == []
    assert kms.calls == []


def test_manifest_rejects_nonidentical_contract_before_kms() -> None:
    manifest = artifact_mapping()
    manifest["compatibility_contract"]["sha256"] = "sha256:" + "0" * 64
    payload = dict(manifest)
    payload.pop("manifest_hash")
    manifest["manifest_hash"] = sha256_json(payload)
    s3 = FakeS3()
    kms = FakeKms()

    with pytest.raises(
        PrivateModelRuntimeError,
        match="compatibility contract differs",
    ):
        verify_private_artifact_manifest_signature(
            manifest,
            s3_client=s3,
            kms_client=kms,
        )

    assert s3.calls == []
    assert kms.calls == []


def test_historical_manifest_hash_remains_readable_but_not_current_eligible(
) -> None:
    manifest = artifact_mapping()
    manifest.pop("compatibility_contract")
    manifest.pop("consumer_parity_fixtures")
    payload = dict(manifest)
    payload.pop("manifest_hash")
    manifest["manifest_hash"] = sha256_json(payload)

    parsed = PrivateModelArtifactManifest.from_mapping(manifest)

    assert "compatibility_contract" not in parsed.to_dict()
    assert "consumer_parity_fixtures" not in parsed.to_dict()
    assert validate_private_model_artifact_manifest(parsed) == []
    with pytest.raises(
        PrivateModelRuntimeError,
        match="compatibility contract differs",
    ):
        verify_private_artifact_manifest_signature(
            parsed,
            s3_client=FakeS3(),
            kms_client=FakeKms(),
        )


def test_signature_download_or_kms_error_fails_closed() -> None:
    class BrokenS3:
        def get_object(self, **_kwargs: Any) -> dict[str, Any]:
            raise RuntimeError("unavailable")

    with pytest.raises(
        PrivateModelRuntimeError,
        match="KMS signature verification failed",
    ):
        verify_private_artifact_manifest_signature(
            artifact_mapping(),
            s3_client=BrokenS3(),
            kms_client=FakeKms(),
        )


def test_non_s3_signature_reference_is_rejected() -> None:
    with pytest.raises(
        PrivateModelRuntimeError,
        match="signature_ref must be an s3:// URI",
    ):
        verify_private_artifact_manifest_signature(
            artifact_mapping(signature_ref="kms:signature"),
            s3_client=FakeS3(),
            kms_client=FakeKms(),
        )


def test_promotion_loader_requires_signature_verification(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = artifact_mapping()
    verified: dict[str, Any] = {}

    monkeypatch.setattr(
        promotion,
        "load_private_artifact_manifest",
        lambda _uri: manifest,
    )

    def fake_verify(artifact: PrivateModelArtifactManifest, *, key_id: str):
        verified["manifest_hash"] = artifact.manifest_hash
        verified["key_id"] = key_id
        return {"verified": True}

    monkeypatch.setattr(
        promotion,
        "verify_private_artifact_manifest_signature",
        fake_verify,
    )

    loaded = promotion._load_valid_artifact("s3://artifacts/current.json")

    assert loaded.manifest_hash == manifest["manifest_hash"]
    assert verified == {
        "manifest_hash": manifest["manifest_hash"],
        "key_id": DEFAULT_PRIVATE_MODEL_ARTIFACT_SIGNING_KMS_KEY_ID,
    }


def test_promotion_loader_propagates_signature_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        promotion,
        "load_private_artifact_manifest",
        lambda _uri: artifact_mapping(),
    )
    monkeypatch.setattr(
        promotion,
        "verify_private_artifact_manifest_signature",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            PrivateModelRuntimeError(
                "private artifact manifest KMS signature was rejected"
            )
        ),
    )

    with pytest.raises(
        PrivateModelRuntimeError,
        match="KMS signature was rejected",
    ):
        promotion._load_valid_artifact("s3://artifacts/current.json")
