import copy
import json

import pytest

from gateway.tee.release_channel_v2 import (
    ReleaseChannelV2Error,
    build_release_channel_v2,
    build_release_lineage_v2,
    cli,
    fetch_release_channel_v2,
    fetch_release_lineage_v2,
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


def _gateway_manifest(commit=COMMIT):
    rows = []
    for index, (role, spec) in enumerate(sorted(ROLE_SPECS.items())):
        character = "abcdef0123456789"[index]
        deterministic = {
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
                        **deterministic,
                    }
                )
    return build_release_manifest(rows, acceptance_signer_pubkey_hash=_hash("f"))


def _validator_manifest(commit=COMMIT):
    release = build_validator_release(
        commit_sha=commit,
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

    def list_objects_v2(self, *, Bucket, Prefix, MaxKeys, **kwargs):
        del MaxKeys, kwargs
        keys = sorted(
            key
            for bucket, key in self.objects
            if bucket == Bucket and key.startswith(Prefix)
        )
        return {
            "Contents": [{"Key": key} for key in keys],
            "IsTruncated": False,
        }


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


def test_release_lineage_binds_historical_exact_role_measurements():
    historical_commit = "2" * 40
    current = build_release_channel_v2(
        gateway_release_manifest=_gateway_manifest(),
        validator_release_manifest=_validator_manifest(),
    )
    historical = build_release_channel_v2(
        gateway_release_manifest=_gateway_manifest(historical_commit),
        validator_release_manifest=_validator_manifest(historical_commit),
    )
    unrelated_commit = "3" * 40
    unrelated = build_release_channel_v2(
        gateway_release_manifest=_gateway_manifest(unrelated_commit),
        validator_release_manifest=_validator_manifest(unrelated_commit),
    )
    lineage = build_release_lineage_v2(
        [historical, current],
        current_commit=COMMIT,
    )
    assert lineage["current_gateway_release_hash"] == (
        current["gateway_release_manifest"]["release_hash"]
    )
    expected = lineage["releases"][historical_commit]["roles"][
        "gateway_coordinator"
    ]
    summary = historical["gateway_release_manifest"]["roles"][
        "gateway_coordinator"
    ]
    assert expected == {
        "commit_sha": historical_commit,
        "pcr0": summary["pcr0"],
        "build_manifest_hash": summary["execution_manifest_hash"],
        "dependency_lock_hash": summary["dependency_lock_hash"],
    }
    validator_expected = lineage["releases"][historical_commit]["roles"][
        "validator_weights"
    ]
    validator_summary = historical["validator_release_manifest"]["release"]
    assert validator_expected == {
        "commit_sha": historical_commit,
        "pcr0": validator_summary["pcr0"],
        "build_manifest_hash": validator_summary["app_manifest_hash"],
        "dependency_lock_hash": validator_summary["dependency_lock_hash"],
    }

    s3 = _S3()
    for channel in (historical, current, unrelated):
        key = release_channel_key(channel["commit_sha"])
        s3.objects[("release-bucket", key)] = (
            json.dumps(channel, sort_keys=True, separators=(",", ":")) + "\n"
        ).encode()
    assert fetch_release_lineage_v2(
        bucket="release-bucket",
        current_commit=COMMIT,
        s3_client=s3,
        allowed_commits=(historical_commit, COMMIT),
    ) == lineage


def test_release_lineage_rejects_missing_current_channel():
    historical_commit = "2" * 40
    historical = build_release_channel_v2(
        gateway_release_manifest=_gateway_manifest(historical_commit),
        validator_release_manifest=_validator_manifest(historical_commit),
    )
    with pytest.raises(ReleaseChannelV2Error, match="current release"):
        build_release_lineage_v2([historical], current_commit=COMMIT)


def test_cli_reports_unpublished_channel_without_traceback(monkeypatch, capsys):
    def _unavailable(**_kwargs):
        raise ReleaseChannelV2Error("approved release channel is unavailable")

    monkeypatch.setattr(
        "gateway.tee.release_channel_v2.fetch_release_channel_v2",
        _unavailable,
    )

    result = cli(["--ensure", "--expected-commit", COMMIT])

    captured = capsys.readouterr()
    assert result == 75
    assert captured.out == ""
    assert captured.err == (
        "Release channel unavailable: "
        "approved release channel is unavailable\n"
    )
    assert "Traceback" not in captured.err
