"""Gateway-only immutable current-release channel contract."""

import copy
import json

import pytest

from gateway.tee.release_channel_v2 import (
    SCHEMA_VERSION,
    ReleaseChannelV2Error,
    build_release_channel_v2,
    fetch_release_channel_v2,
    install_release_channel_v2,
    publish_release_channel_v2,
    release_channel_role_identities_v2,
    validate_release_channel_v2,
)
from gateway.tee.release_manifest_v2 import (
    BUILD_EVIDENCE_SCHEMA_VERSION,
    build_release_manifest,
)
from gateway.tee.topology import ROLE_SPECS, topology_hash


COMMIT = "1" * 40


def _hash(character):
    return "sha256:" + character * 64


def _manifest(commit=COMMIT):
    rows = []
    for index, (role, spec) in enumerate(sorted(ROLE_SPECS.items())):
        character = "abcdef0123456789"[index]
        fixed = {
            "commit_sha": commit,
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
                        **fixed,
                    }
                )
    return build_release_manifest(rows, acceptance_signer_pubkey_hash=_hash("f"))


class Body:
    def __init__(self, value):
        self.value = value

    def read(self):
        return self.value


class S3:
    def __init__(self):
        self.objects = {}
        self.puts = []

    def get_object(self, **kwargs):
        return {"Body": Body(self.objects[(kwargs["Bucket"], kwargs["Key"])])}

    def put_object(self, **kwargs):
        self.objects[(kwargs["Bucket"], kwargs["Key"])] = kwargs["Body"]
        self.puts.append(kwargs)


def test_current_channel_contains_only_current_gateway_authority():
    channel = build_release_channel_v2(gateway_release_manifest=_manifest())
    assert channel["schema_version"] == SCHEMA_VERSION
    assert set(channel) == {
        "schema_version",
        "commit_sha",
        "gateway_release_manifest",
        "channel_hash",
    }
    assert validate_release_channel_v2(channel, expected_commit=COMMIT) == channel
    assert set(release_channel_role_identities_v2(channel)) == set(ROLE_SPECS)
    bad = copy.deepcopy(channel)
    bad["validator_release_manifest"] = {}
    with pytest.raises(ReleaseChannelV2Error, match="fields"):
        validate_release_channel_v2(bad)


def test_local_current_release_fetch_uses_exact_validated_manifest(monkeypatch, tmp_path):
    manifest_path = tmp_path / "gateway.json"
    manifest_path.write_text(json.dumps(_manifest()), encoding="utf-8")
    monkeypatch.setenv("LEADPOET_LOCAL_RELEASE_COMMIT_SHA", COMMIT)
    monkeypatch.setenv("LEADPOET_LOCAL_GATEWAY_RELEASE", str(manifest_path))

    channel = fetch_release_channel_v2(bucket="unused", commit_sha=COMMIT)

    assert channel == build_release_channel_v2(gateway_release_manifest=_manifest())


def test_publish_is_immutable_object_locked_and_install_is_gateway_only(tmp_path):
    s3 = S3()
    channel = build_release_channel_v2(gateway_release_manifest=_manifest())
    published = publish_release_channel_v2(channel, bucket="bucket", s3_client=s3)
    assert published["commit_sha"] == COMMIT
    assert s3.puts[0]["ObjectLockMode"] == "COMPLIANCE"
    assert s3.puts[0]["IfNoneMatch"] == "*"
    output = tmp_path / "gateway.json"
    install_release_channel_v2(
        channel, expected_commit=COMMIT, gateway_output=output
    )
    assert json.loads(output.read_text()) == channel["gateway_release_manifest"]
