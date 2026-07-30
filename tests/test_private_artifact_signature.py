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
    validate_private_model_artifact_manifest,
    verify_private_artifact_manifest_signature,
)


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


def artifact_mapping(**overrides: Any) -> dict[str, Any]:
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
        "compatibility_contract": {
            "contract_id": CONSUMER_CONTRACT_ID,
            "path": "sourcing_model/consumer_contract.json",
            "sha256": sha256_bytes(
                (
                    RESEARCH_LAB_ROOT / "sourcing_model_contract.json"
                ).read_bytes()
            ),
        },
        "consumer_parity_fixtures": {
            "path": "sourcing_model/consumer_parity_fixtures.json",
            "sha256": sha256_bytes(
                (
                    RESEARCH_LAB_ROOT
                    / "sourcing_model_parity_fixtures.json"
                ).read_bytes()
            ),
        },
        "manifest_uri": "s3://artifacts/model.json",
        "signature_ref": "s3://artifacts/model.sig.b64",
        "build_id": "build-1",
    }
    payload.update(overrides)
    return {
        **payload,
        "manifest_hash": sha256_json(payload),
    }


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
