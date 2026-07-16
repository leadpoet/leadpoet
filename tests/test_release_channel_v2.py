import copy
import json

import pytest

from gateway.tee.release_channel_v2 import (
    ReleaseChannelV2Error,
    build_release_channel_v2,
    fetch_release_channel_v2,
    install_release_channel_v2,
    publish_release_channel_v2,
    release_channel_key,
    validate_release_channel_v2,
)
from gateway.tee.release_manifest_v2 import (
    BUILD_EVIDENCE_SCHEMA_VERSION,
    build_release_manifest,
)
from gateway.tee.topology import ROLE_SPECS, topology_hash
from validator_tee.host.release_v2 import (
    build_validator_build_evidence,
    build_validator_release,
    build_validator_release_manifest,
)


COMMIT = "1" * 40


def _hash(character):
    return "sha256:" + character * 64


def _gateway_manifest():
    rows = []
    for index, (role, spec) in enumerate(sorted(ROLE_SPECS.items())):
        character = "abcdef0123456789"[index]
        deterministic = {
            "commit_sha": COMMIT,
            "pcr0": character * 96,
            "normalized_image_hash": _hash(character),
            "eif_hash": _hash(character),
            "source_manifest_hash": _hash("2"),
            "build_identity_hash": _hash(character),
            "execution_manifest_hash": _hash(character),
            "dependency_lock_hash": _hash("3"),
            "dockerfile_hash": _hash("4"),
            "topology_hash": topology_hash(),
        }
        for domain in ("gateway", "validator"):
            for ordinal in (1, 2, 3):
                rows.append(
                    {
                        "schema_version": BUILD_EVIDENCE_SCHEMA_VERSION,
                        "builder_domain": domain,
                        "builder_id": domain + "-parent",
                        "build_ordinal": ordinal,
                        "physical_role": role,
                        "service_role": spec["service_role"],
                        **deterministic,
                    }
                )
    return build_release_manifest(rows, acceptance_signer_pubkey_hash=_hash("f"))


def _validator_manifest():
    release = build_validator_release(
        commit_sha=COMMIT,
        pcr0="2" * 96,
        app_manifest_hash=_hash("3"),
        dependency_lock_hash=_hash("4"),
        normalized_image_hash=_hash("5"),
        eif_hash=_hash("6"),
        dockerfile_hash=_hash("7"),
        base_dockerfile_hash=_hash("8"),
    )
    evidence = [
        build_validator_build_evidence(
            release,
            builder_domain=domain,
            builder_id=domain + "-parent",
            build_ordinal=ordinal,
        )
        for domain in ("gateway", "validator")
        for ordinal in (1, 2, 3)
    ]
    return build_validator_release_manifest(evidence)


class _Body:
    def __init__(self, value):
        self._value = value

    def read(self):
        return self._value


class _S3:
    def __init__(self):
        self.objects = {}
        self.puts = []

    def get_object(self, *, Bucket, Key):
        if (Bucket, Key) not in self.objects:
            raise KeyError(Key)
        return {"Body": _Body(self.objects[(Bucket, Key)])}

    def put_object(self, **kwargs):
        key = (kwargs["Bucket"], kwargs["Key"])
        if key in self.objects:
            raise RuntimeError("precondition failed")
        self.objects[key] = kwargs["Body"]
        self.puts.append(kwargs)


def test_channel_binds_both_independent_release_manifests():
    value = build_release_channel_v2(
        gateway_release_manifest=_gateway_manifest(),
        validator_release_manifest=_validator_manifest(),
    )
    assert validate_release_channel_v2(value, expected_commit=COMMIT) == value
    tampered = copy.deepcopy(value)
    tampered["commit_sha"] = "2" * 40
    with pytest.raises(ReleaseChannelV2Error, match="commit"):
        validate_release_channel_v2(tampered)


def test_channel_rejects_cross_commit_manifests():
    validator = _validator_manifest()
    validator["release"]["commit_sha"] = "2" * 40
    with pytest.raises(Exception):
        build_release_channel_v2(
            gateway_release_manifest=_gateway_manifest(),
            validator_release_manifest=validator,
        )


def test_channel_publish_is_immutable_and_fetch_installs_atomically(tmp_path):
    channel = build_release_channel_v2(
        gateway_release_manifest=_gateway_manifest(),
        validator_release_manifest=_validator_manifest(),
    )
    s3 = _S3()
    published = publish_release_channel_v2(
        channel, bucket="release-bucket", s3_client=s3
    )
    assert published["commit_sha"] == COMMIT
    assert s3.puts[0]["IfNoneMatch"] == "*"
    assert s3.puts[0]["ObjectLockMode"] == "COMPLIANCE"

    fetched = fetch_release_channel_v2(
        bucket="release-bucket", commit_sha=COMMIT, s3_client=s3
    )
    gateway_output = tmp_path / "gateway.json"
    validator_output = tmp_path / "validator.json"
    install_release_channel_v2(
        fetched,
        expected_commit=COMMIT,
        gateway_output=gateway_output,
        validator_output=validator_output,
    )
    assert json.loads(gateway_output.read_text())["commit_sha"] == COMMIT
    assert (
        json.loads(validator_output.read_text())["release"]["commit_sha"]
        == COMMIT
    )
    assert gateway_output.stat().st_mode & 0o777 == 0o600


def test_release_channel_key_is_content_addressed_by_commit():
    assert release_channel_key(COMMIT).endswith(
        f"/{COMMIT}/release-channel-v2.json"
    )
