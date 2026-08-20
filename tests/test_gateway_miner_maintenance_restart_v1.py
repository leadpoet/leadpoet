from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
import hashlib
import io
import os
from pathlib import Path
import stat
import subprocess
import sys
from types import SimpleNamespace

import pytest

from gateway.tee import disable_gateway_miner_submissions_secret as disable_operation
from gateway.tee import gateway_miner_maintenance_restart_v1 as maintenance


INITIAL_VERSION = "11111111-1111-4111-8111-111111111111"
CONCURRENT_VERSION = "22222222-2222-4222-8222-222222222222"
CANDIDATE_COMMIT = "a" * 40
TREE_HASH = "b" * 40
RELEASE_HASH = "sha256:" + "c" * 64
CHANNEL_HASH = "sha256:" + "d" * 64
BLOB_HASH = "e" * 64
CONTROLLER_COMMIT = next(iter(maintenance.SUPPORTED_N_MINUS_ONE_CONTROLLER_COMMITS))
RECOVERY_VERSION = "33333333-3333-4333-8333-333333333333"
PREVIOUS_VERSION = "44444444-4444-4444-8444-444444444444"
PENDING_VERSION = "55555555-5555-4555-8555-555555555555"
OBJECT_VERSION = "version-identity-0000000000000001"
OBJECT_SHA = "sha256:" + "f" * 64
RETAIN_UNTIL = "2099-01-01T00:00:00Z"
REAL_REQUIRE_HYDRATED_ENVIRONMENT_COMMITMENT = (
    maintenance._require_hydrated_environment_commitment
)
REAL_LIVE_GATEWAY_RESTART_AUTHORITY_COMMITMENT = (
    maintenance._live_gateway_restart_authority_commitment
)


@pytest.fixture(autouse=True)
def _stable_live_gateway_identity(monkeypatch: pytest.MonkeyPatch):
    for name in disable_operation._FORBIDDEN_AWS_ENV_NAMES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(
        maintenance,
        "_live_gateway_restart_authority_commitment",
        lambda **_kwargs: "sha256:" + "1" * 64,
    )
    monkeypatch.setattr(
        maintenance,
        "_require_hydrated_environment_commitment",
        lambda **kwargs: str(kwargs["expected_commitment"]),
    )


def test_ci_environment_fixture_scrubs_static_aws_authority() -> None:
    assert not (
        disable_operation._FORBIDDEN_AWS_ENV_NAMES & set(os.environ)
    )


def test_prepare_still_rejects_explicit_static_aws_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = FakeSecretsClient(
        "RESEARCH_LAB_MINER_SUBMISSIONS_ENABLED=true\n"
    )
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "explicit-rejection-sentinel")
    with pytest.raises(
        maintenance.GatewayMinerMaintenanceRestartError,
        match="authority differs from production",
    ):
        _prepare(tmp_path, monkeypatch, client)


def _locked_release_evidence() -> dict[str, object]:
    return {
        "channel": {
            "commit_sha": CANDIDATE_COMMIT,
            "channel_hash": CHANNEL_HASH,
            "gateway_release_manifest": {
                "commit_sha": CANDIDATE_COMMIT,
                "release_hash": RELEASE_HASH,
            },
        },
        "object_version_id": OBJECT_VERSION,
        "object_sha256": OBJECT_SHA,
        "object_lock_mode": "COMPLIANCE",
        "object_retain_until": RETAIN_UNTIL,
    }


class FakeLockedReleaseS3:
    def __init__(self):
        self.payload = json.dumps(
            {"commit_sha": CANDIDATE_COMMIT, "marker": "locked"},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
        self.version_id = OBJECT_VERSION
        self.etag = '"exact-etag"'
        self.retain_until = datetime.now(timezone.utc) + timedelta(days=30)
        self.versions = 1
        self.delete_markers = 0
        self.latest = True
        self.get_etag = self.etag
        self.get_retain_until = self.retain_until
        self.unversioned_heads = 0
        self.replace_after_get = False
        self.calls: list[tuple[str, dict[str, object]]] = []
        self.last_body: io.BytesIO | None = None

    def _metadata(
        self,
        *,
        version_id: str | None = None,
        etag: str | None = None,
        retain_until: datetime | None = None,
    ):
        return {
            "VersionId": version_id or self.version_id,
            "ObjectLockMode": "COMPLIANCE",
            "ObjectLockRetainUntilDate": retain_until or self.retain_until,
            "ETag": etag or self.etag,
            "ContentLength": len(self.payload),
        }

    def list_object_versions(self, **kwargs):
        self.calls.append(("list", dict(kwargs)))
        key = maintenance.release_channel_key(
            CANDIDATE_COMMIT,
            prefix=maintenance.DEFAULT_RELEASE_PREFIX,
        )
        versions = [
            {
                "Key": key,
                "VersionId": self.version_id,
                "ETag": self.etag,
                "Size": len(self.payload),
                "IsLatest": self.latest,
            }
            for _index in range(self.versions)
        ]
        return {
            "IsTruncated": False,
            "Versions": versions
            + [
                {
                    "Key": key + ".unrelated",
                    "VersionId": "ignored",
                    "ETag": '"ignored"',
                    "Size": 2,
                    "IsLatest": True,
                }
            ],
            "DeleteMarkers": [
                {
                    "Key": key,
                    "VersionId": f"delete-{index}",
                    "IsLatest": False,
                }
                for index in range(self.delete_markers)
            ],
        }

    def head_object(self, **kwargs):
        self.calls.append(("head", dict(kwargs)))
        if "VersionId" not in kwargs:
            self.unversioned_heads += 1
            if self.replace_after_get and self.unversioned_heads > 1:
                return self._metadata(version_id="replacement-version")
        return self._metadata()

    def get_object(self, **kwargs):
        self.calls.append(("get", dict(kwargs)))
        self.last_body = io.BytesIO(self.payload)
        return {
            **self._metadata(
                etag=self.get_etag,
                retain_until=self.get_retain_until,
            ),
            "Body": self.last_body,
        }


def _installed_controller_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    controller_commit: str = CONTROLLER_COMMIT,
):
    tmp_path.mkdir(parents=True, exist_ok=True)
    controller_parent = tmp_path / "restart-controller"
    controller_root = controller_parent / "gateway"
    releases_root = controller_root / "releases"
    release = releases_root / controller_commit
    for directory in (controller_parent, controller_root, releases_root):
        directory.mkdir(exist_ok=True)
        directory.chmod(0o775)
    (release / "scripts").mkdir(parents=True)
    (release / "Leadpoet/utils").mkdir(parents=True)
    (release / "gateway/tee").mkdir(parents=True)
    release.chmod(0o700)
    files = {
        "gw_restart.sh": b"#!/bin/bash\nexit 0\n",
        "scripts/gateway_git_deploy.py": b"HELPER = True\n",
        "Leadpoet/utils/exact_commit_restart_v2.py": b"EXACT = True\n",
        "gateway/tee/host_memory_guard_v2.py": b"GUARD = True\n",
    }
    for relative, payload in files.items():
        destination = release / relative
        destination.write_bytes(payload)
        destination.chmod(0o700 if relative == "gw_restart.sh" else 0o600)
    current = controller_root / "current"
    current.symlink_to(f"releases/{controller_commit}")
    host_restart = tmp_path / "gw_restart.sh"
    host_restart.write_bytes(files["gw_restart.sh"])
    host_restart.chmod(0o700)
    monkeypatch.setattr(
        maintenance,
        "_run_git_bytes",
        lambda _repo, _show, object_name: files[object_name.split(":", 1)[1]],
    )
    monkeypatch.setattr(
        maintenance.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=0),
    )
    return controller_parent, controller_root, releases_root, release, current, host_restart


class FakeSecretsClient:
    def __init__(self, secret: str):
        self.versions = {INITIAL_VERSION: secret}
        self.stages = {INITIAL_VERSION: {"AWSCURRENT"}}
        self.read_count = 0

    @property
    def current(self) -> str:
        current = [
            version for version, labels in self.stages.items() if "AWSCURRENT" in labels
        ]
        if len(current) != 1:
            raise RuntimeError("current stage is ambiguous")
        return current[0]

    def get_secret_value(
        self,
        *,
        SecretId,
        VersionId=None,
        VersionStage=None,
    ):
        assert SecretId == disable_operation.GATEWAY_SECRET_ID
        self.read_count += 1
        if VersionId is not None:
            version = VersionId
        elif VersionStage == "AWSCURRENT":
            version = self.current
        else:
            raise AssertionError("test reads must be version-bound")
        return {
            "Name": SecretId,
            "VersionId": version,
            "SecretString": self.versions[version],
        }

    def describe_secret(self, *, SecretId):
        assert SecretId == disable_operation.GATEWAY_SECRET_ID
        return {
            "Name": SecretId,
            "VersionIdsToStages": {
                version: sorted(labels)
                for version, labels in self.stages.items()
                if labels
            },
        }

    def put_secret_value(
        self,
        *,
        SecretId,
        SecretString,
        ClientRequestToken,
        VersionStages,
    ):
        assert SecretId == disable_operation.GATEWAY_SECRET_ID
        if ClientRequestToken in self.versions:
            raise RuntimeError("version token already exists")
        self.versions[ClientRequestToken] = SecretString
        self.stages[ClientRequestToken] = set(VersionStages)
        return {"VersionId": ClientRequestToken}

    def update_secret_version_stage(
        self,
        *,
        SecretId,
        VersionStage,
        MoveToVersionId=None,
        RemoveFromVersionId=None,
    ):
        assert SecretId == disable_operation.GATEWAY_SECRET_ID
        if RemoveFromVersionId is not None:
            if VersionStage not in self.stages.get(RemoveFromVersionId, set()):
                raise RuntimeError("version-stage fence failed")
        if MoveToVersionId is not None:
            if VersionStage == "AWSCURRENT" and RemoveFromVersionId is not None:
                for labels in self.stages.values():
                    labels.discard("AWSPREVIOUS")
                self.stages[RemoveFromVersionId].add("AWSPREVIOUS")
            if RemoveFromVersionId is not None:
                self.stages[RemoveFromVersionId].discard(VersionStage)
            self.stages[MoveToVersionId].add(VersionStage)
        elif RemoveFromVersionId is not None:
            self.stages[RemoveFromVersionId].remove(VersionStage)
        return {}

    def install_concurrent_current(self) -> None:
        prior = self.current
        self.versions[CONCURRENT_VERSION] = self.versions[prior] + "DRIFT=value\n"
        for labels in self.stages.values():
            labels.discard("AWSPREVIOUS")
        self.stages[prior].discard("AWSCURRENT")
        self.stages[prior].add("AWSPREVIOUS")
        self.stages[CONCURRENT_VERSION] = {"AWSCURRENT"}


def _controller_bundle(
    controller_commit: str = CONTROLLER_COMMIT,
) -> dict[str, object]:
    payloads = {
        "wrapper": b"#!/bin/bash\nexit 0\n",
        "git_helper": b"HELPER = True\n",
        "exact_commit_helper": b"EXACT = True\n",
        "memory_guard": b"GUARD = True\n",
    }
    return {
        "controller_commit": controller_commit,
        "payloads": payloads,
        "commitments": {
            name: "sha256:" + hashlib.sha256(payload).hexdigest()
            for name, payload in payloads.items()
        },
    }


def _release_evidence(
    commit: str = CANDIDATE_COMMIT,
    release_hash: str = RELEASE_HASH,
) -> dict[str, object]:
    evidence = _locked_release_evidence()
    channel = dict(evidence["channel"])
    gateway_release = dict(channel["gateway_release_manifest"])
    channel["commit_sha"] = commit
    gateway_release["commit_sha"] = commit
    gateway_release["release_hash"] = release_hash
    channel["gateway_release_manifest"] = gateway_release
    evidence["channel"] = channel
    return evidence


def _prepare(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    client: FakeSecretsClient,
    *,
    candidate_commit: str = CANDIDATE_COMMIT,
    controller_commit: str = CONTROLLER_COMMIT,
    invocation_id: str = "gateway-test-invocation",
) -> dict[str, object]:
    monkeypatch.setattr(
        maintenance,
        "_require_canonical_restart_lock_fd",
        lambda: None,
    )
    monkeypatch.setattr(
        maintenance,
        "_validate_candidate_identity",
        lambda **_kwargs: {
            "tree_hash": TREE_HASH,
            "blob_manifest_sha256": BLOB_HASH,
            "previous_sha": controller_commit,
            "n_minus_one_controller_commit": controller_commit,
            "controller_bundle": _controller_bundle(controller_commit),
        },
    )
    monkeypatch.setattr(maintenance, "_verify_protected_source", lambda: None)
    monkeypatch.setattr(
        maintenance,
        "_live_gateway_restart_authority_commitment",
        lambda **_kwargs: "sha256:" + "1" * 64,
    )
    monkeypatch.setattr(
        maintenance,
        "_fetch_locked_release_channel",
        lambda **_kwargs: _release_evidence(candidate_commit),
    )
    monkeypatch.setattr(
        maintenance,
        "validate_release_manifest",
        lambda value: dict(value),
    )
    return maintenance.prepare_gateway_miner_maintenance_restart(
        repo_root=tmp_path / "repo",
        candidate_root=tmp_path / "candidate",
        plan_file=tmp_path / "plan.json",
        expected_commit=candidate_commit,
        controller_current=tmp_path / "controller/current",
        host_restart_path=tmp_path / "gw_restart.sh",
        restart_invocation_id=invocation_id,
        recovery_journal_path=tmp_path / "private" / "transaction.json",
        secrets_client=client,
        release_s3_client=object(),
    )


def test_prepare_changes_only_target_and_returns_redacted_invocation_proof(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    secret_marker = "unrelated-secret-must-not-escape"
    client = FakeSecretsClient(
        "RESEARCH_LAB_MINER_SUBMISSIONS_ENABLED=true\n"
        f"UNRELATED_SECRET={secret_marker}\n"
        "RESEARCH_LAB_AUTORESEARCH_WORKER_COUNT=0\n"
    )

    result = _prepare(tmp_path, monkeypatch, client)
    proof = result["proof"]

    assert result["status"] == "prepared"
    assert proof["candidate_commit"] == CANDIDATE_COMMIT
    assert proof["candidate_tree_hash"] == TREE_HASH
    assert proof["gateway_release_hash"] == RELEASE_HASH
    assert proof["current_secret_version_id"] == client.current
    assert proof["current_document_commitment"].startswith("sha256:")
    assert proof["current_stage_topology_commitment"].startswith("sha256:")
    assert secret_marker not in json.dumps(proof)
    assert secret_marker not in json.dumps(
        {name: value for name, value in result.items() if name != "tree_evidence"}
    )
    current_secret = client.versions[client.current]
    assert "RESEARCH_LAB_MINER_SUBMISSIONS_ENABLED=false\n" in current_secret
    assert f"UNRELATED_SECRET={secret_marker}\n" in current_secret
    assert "RESEARCH_LAB_AUTORESEARCH_WORKER_COUNT=0\n" in current_secret
    assert not (tmp_path / "private" / "transaction.json").exists()
    assert not any(path.name.endswith("receipt.json") for path in tmp_path.rglob("*"))


@pytest.mark.parametrize(
    "crash_point",
    ["after_stage", "after_promotion", "during_rollback"],
)
def test_prepare_recovers_crashed_secret_transaction_and_remains_idempotent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    crash_point: str,
):
    client = FakeSecretsClient("RESEARCH_LAB_MINER_SUBMISSIONS_ENABLED=true\n")
    client.versions[PREVIOUS_VERSION] = "OLDER=value\n"
    client.stages[PREVIOUS_VERSION] = {"AWSPREVIOUS"}
    client.versions[PENDING_VERSION] = "PENDING=value\n"
    client.stages[PENDING_VERSION] = {"AWSPENDING"}
    initial_topology = disable_operation._version_stages(client)
    prior_secret = client.versions[INITIAL_VERSION]
    candidate_secret, _document_format, status = (
        disable_operation._validated_candidate(prior_secret)
    )
    assert status == "verified"
    custom_label = disable_operation._custom_stage_label(RECOVERY_VERSION)
    journal_path = tmp_path / "private" / "transaction.json"
    disable_operation._write_recovery_journal(
        journal_path,
        disable_operation._recovery_journal_body(
            prior_version_id=INITIAL_VERSION,
            candidate_version_id=RECOVERY_VERSION,
            custom_stage_label=custom_label,
            initial_topology=initial_topology,
            prior_document_commitment=disable_operation._document_commitment(
                prior_secret
            ),
            candidate_document_commitment=disable_operation._document_commitment(
                candidate_secret
            ),
        ),
    )
    client.put_secret_value(
        SecretId=disable_operation.GATEWAY_SECRET_ID,
        SecretString=candidate_secret,
        ClientRequestToken=RECOVERY_VERSION,
        VersionStages=[custom_label],
    )
    if crash_point in {"after_promotion", "during_rollback"}:
        client.update_secret_version_stage(
            SecretId=disable_operation.GATEWAY_SECRET_ID,
            VersionStage="AWSCURRENT",
            MoveToVersionId=RECOVERY_VERSION,
            RemoveFromVersionId=INITIAL_VERSION,
        )
    if crash_point == "during_rollback":
        client.update_secret_version_stage(
            SecretId=disable_operation.GATEWAY_SECRET_ID,
            VersionStage="AWSCURRENT",
            MoveToVersionId=INITIAL_VERSION,
            RemoveFromVersionId=RECOVERY_VERSION,
        )

    first = _prepare(tmp_path, monkeypatch, client)
    second = _prepare(tmp_path, monkeypatch, client)

    assert first["status"] == "prepared"
    assert second["status"] == "prepared"
    assert not journal_path.exists()
    assert disable_operation._validated_candidate(client.versions[client.current])[2] == (
        "already_disabled"
    )
    topology = disable_operation._version_stages(client)
    assert topology[PENDING_VERSION] == frozenset({"AWSPENDING"})
    assert all(
        not any(
            stage.startswith(disable_operation._CUSTOM_STAGE_PREFIX)
            for stage in stages
        )
        for stages in topology.values()
    )


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("LEADPOET_GATEWAY_ENV_SECRET_ID", "another/secret"),
        ("GATEWAY_V2_RELEASE_BUCKET", "another-bucket"),
        ("AWS_REGION", "us-west-2"),
        ("AWS_DEFAULT_REGION", "eu-west-1"),
    ],
)
def test_prepare_rejects_nonproduction_authority_before_aws_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    value: str,
):
    client = FakeSecretsClient("RESEARCH_LAB_MINER_SUBMISSIONS_ENABLED=true\n")
    before_versions = dict(client.versions)
    before_stages = {key: set(labels) for key, labels in client.stages.items()}
    monkeypatch.setenv(name, value)

    with pytest.raises(
        maintenance.GatewayMinerMaintenanceRestartError,
        match="authority differs from production",
    ):
        _prepare(tmp_path, monkeypatch, client)

    assert client.versions == before_versions
    assert client.stages == before_stages


def test_fresh_same_and_different_candidate_retries_accept_already_false_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    client = FakeSecretsClient("RESEARCH_LAB_MINER_SUBMISSIONS_ENABLED=true\n")
    first = _prepare(
        tmp_path,
        monkeypatch,
        client,
        invocation_id="gateway-first",
    )
    after_first_versions = dict(client.versions)
    after_first_stages = {key: set(labels) for key, labels in client.stages.items()}

    second = _prepare(
        tmp_path,
        monkeypatch,
        client,
        invocation_id="gateway-second",
    )
    different_commit = "f" * 40
    third = _prepare(
        tmp_path,
        monkeypatch,
        client,
        candidate_commit=different_commit,
        invocation_id="gateway-third",
    )

    assert first["proof"]["restart_invocation_id"] == "gateway-first"
    assert second["proof"]["restart_invocation_id"] == "gateway-second"
    assert third["proof"]["candidate_commit"] == different_commit
    assert len({
        first["proof"]["proof_hash"],
        second["proof"]["proof_hash"],
        third["proof"]["proof_hash"],
    }) == 3
    assert client.versions == after_first_versions
    assert client.stages == after_first_stages


def test_locked_release_fetch_pins_singleton_compliance_version_and_content(
    monkeypatch: pytest.MonkeyPatch,
):
    s3 = FakeLockedReleaseS3()
    monkeypatch.setattr(
        maintenance,
        "validate_release_channel_v2",
        lambda value, *, expected_commit: {
            **value,
            "commit_sha": expected_commit,
        },
    )
    monkeypatch.setattr(
        maintenance,
        "_require_six_release_identities",
        lambda _channel, *, expected_commit: None,
    )

    evidence = maintenance._fetch_locked_release_channel(
        commit_sha=CANDIDATE_COMMIT,
        s3_client=s3,
    )

    assert evidence["object_version_id"] == OBJECT_VERSION
    assert evidence["object_sha256"] == (
        "sha256:" + hashlib.sha256(s3.payload).hexdigest()
    )
    get_call = next(arguments for name, arguments in s3.calls if name == "get")
    assert get_call["VersionId"] == OBJECT_VERSION
    assert [name for name, _arguments in s3.calls].count("list") == 2
    assert s3.last_body is not None and s3.last_body.closed


def test_locked_release_fetch_closes_body_when_read_fails(
    monkeypatch: pytest.MonkeyPatch,
):
    class ExplodingBody(io.BytesIO):
        def read(self, *_args, **_kwargs):
            raise RuntimeError("read failed")

    class ReadFailureS3(FakeLockedReleaseS3):
        def get_object(self, **kwargs):
            self.calls.append(("get", dict(kwargs)))
            self.last_body = ExplodingBody(self.payload)
            return {**self._metadata(), "Body": self.last_body}

    s3 = ReadFailureS3()
    monkeypatch.setattr(
        maintenance,
        "validate_release_channel_v2",
        lambda value, *, expected_commit: value,
    )
    monkeypatch.setattr(
        maintenance,
        "_require_six_release_identities",
        lambda _channel, *, expected_commit: None,
    )

    with pytest.raises(
        maintenance.GatewayMinerMaintenanceRestartError,
        match="lock evidence is unavailable",
    ):
        maintenance._fetch_locked_release_channel(
            commit_sha=CANDIDATE_COMMIT,
            s3_client=s3,
        )

    assert s3.last_body is not None and s3.last_body.closed


def test_locked_release_fetch_closes_body_when_get_metadata_is_invalid(
    monkeypatch: pytest.MonkeyPatch,
):
    class InvalidMetadataS3(FakeLockedReleaseS3):
        def get_object(self, **kwargs):
            self.calls.append(("get", dict(kwargs)))
            self.last_body = io.BytesIO(self.payload)
            return {
                **self._metadata(),
                "ObjectLockMode": "GOVERNANCE",
                "Body": self.last_body,
            }

    s3 = InvalidMetadataS3()
    monkeypatch.setattr(
        maintenance,
        "validate_release_channel_v2",
        lambda value, *, expected_commit: value,
    )
    monkeypatch.setattr(
        maintenance,
        "_require_six_release_identities",
        lambda _channel, *, expected_commit: None,
    )

    with pytest.raises(
        maintenance.GatewayMinerMaintenanceRestartError,
        match="lacks COMPLIANCE retention",
    ):
        maintenance._fetch_locked_release_channel(
            commit_sha=CANDIDATE_COMMIT,
            s3_client=s3,
        )

    assert s3.last_body is not None and s3.last_body.closed


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("replacement_history", "history is not a singleton"),
        ("delete_marker", "history is not a singleton"),
        ("expired_lock", "retention is not active"),
        ("head_get_drift", "identities differ"),
        ("retention_drift", "identities differ"),
        ("latest_drift", "object version changed"),
    ],
)
def test_locked_release_fetch_rejects_replacement_expiry_and_head_get_drift(
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
    message: str,
):
    s3 = FakeLockedReleaseS3()
    if mutation == "replacement_history":
        s3.versions = 2
    elif mutation == "delete_marker":
        s3.delete_markers = 1
    elif mutation == "expired_lock":
        s3.retain_until = datetime.now(timezone.utc) - timedelta(seconds=1)
    elif mutation == "head_get_drift":
        s3.get_etag = '"drifted-etag"'
    elif mutation == "retention_drift":
        s3.get_retain_until = s3.retain_until + timedelta(days=1)
    elif mutation == "latest_drift":
        s3.replace_after_get = True
    monkeypatch.setattr(
        maintenance,
        "validate_release_channel_v2",
        lambda value, *, expected_commit: value,
    )
    monkeypatch.setattr(
        maintenance,
        "_require_six_release_identities",
        lambda _channel, *, expected_commit: None,
    )

    with pytest.raises(maintenance.GatewayMinerMaintenanceRestartError, match=message):
        maintenance._fetch_locked_release_channel(
            commit_sha=CANDIDATE_COMMIT,
            s3_client=s3,
        )


def test_release_channel_requires_all_six_exact_commit_identities():
    channel = {
        "commit_sha": CANDIDATE_COMMIT,
        "gateway_release_manifest": {
            "commit_sha": CANDIDATE_COMMIT,
            "roles": {
                role: {"commit_sha": CANDIDATE_COMMIT}
                for role in maintenance.ROLE_SPECS
            },
        },
        "validator_release_manifest": {
            "release": {"commit_sha": CANDIDATE_COMMIT},
        },
    }
    maintenance._require_six_release_identities(
        channel,
        expected_commit=CANDIDATE_COMMIT,
    )
    first_role = next(iter(maintenance.ROLE_SPECS))
    channel["gateway_release_manifest"]["roles"][first_role]["commit_sha"] = "9" * 40
    with pytest.raises(
        maintenance.GatewayMinerMaintenanceRestartError,
        match="identities differ",
    ):
        maintenance._require_six_release_identities(
            channel,
            expected_commit=CANDIDATE_COMMIT,
        )


def test_direct_restart_requires_durable_false_and_locked_release(
    monkeypatch: pytest.MonkeyPatch,
):
    client = FakeSecretsClient("RESEARCH_LAB_MINER_SUBMISSIONS_ENABLED=false\n")
    calls: list[str] = []
    monkeypatch.setattr(
        maintenance,
        "_fetch_locked_release_channel",
        lambda **_kwargs: calls.append("locked") or _locked_release_evidence(),
    )

    result = maintenance.verify_gateway_miner_maintenance_state(
        deploy_commit=CANDIDATE_COMMIT,
        candidate_tree_hash=TREE_HASH,
        gateway_release_hash=RELEASE_HASH,
        parent_environment={
            "RESEARCH_LAB_MINER_SUBMISSIONS_ENABLED": "false",
        },
        secrets_client=client,
        release_s3_client=object(),
    )

    assert result["status"] == "durable_false_verified"
    assert result["current_secret_version_id"] == INITIAL_VERSION
    assert result["release_channel_object_version_id"] == OBJECT_VERSION
    assert calls == ["locked"]


def test_receiptless_second_restart_binds_actual_hydrated_cache_bytes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    raw_secret = (
        "UNRELATED='preserved value'\n"
        "RESEARCH_LAB_MINER_SUBMISSIONS_ENABLED=false\n"
    )
    client = FakeSecretsClient(raw_secret)
    hydrated = tmp_path / "private" / "gateway.env"
    hydrated.parent.mkdir(mode=0o700)
    hydrated.write_text(
        disable_operation._n_minus_one_hydrated_environment(raw_secret),
        encoding="utf-8",
    )
    hydrated.chmod(0o600)
    monkeypatch.setattr(
        maintenance,
        "_require_hydrated_environment_commitment",
        REAL_REQUIRE_HYDRATED_ENVIRONMENT_COMMITMENT,
    )
    monkeypatch.setattr(
        maintenance,
        "_fetch_locked_release_channel",
        lambda **_kwargs: _locked_release_evidence(),
    )

    result = maintenance.verify_gateway_miner_maintenance_state(
        deploy_commit=CANDIDATE_COMMIT,
        candidate_tree_hash=TREE_HASH,
        gateway_release_hash=RELEASE_HASH,
        parent_environment={
            "RESEARCH_LAB_MINER_SUBMISSIONS_ENABLED": "false",
            "LEADPOET_AWS_INSTANCE_ROLE_ONLY": "true",
            "AWS_REGION": "us-east-1",
            "AWS_DEFAULT_REGION": "us-east-1",
        },
        secrets_client=client,
        release_s3_client=object(),
        hydrated_environment_path=hydrated,
    )

    assert result["status"] == "durable_false_verified"
    assert client.read_count >= 2

    hydrated.write_text(
        "UNRELATED=tampered\n"
        "RESEARCH_LAB_MINER_SUBMISSIONS_ENABLED=false\n",
        encoding="utf-8",
    )
    with pytest.raises(
        maintenance.GatewayMinerMaintenanceRestartError,
        match="hydrated gateway environment differs",
    ):
        maintenance.verify_gateway_miner_maintenance_state(
            deploy_commit=CANDIDATE_COMMIT,
            candidate_tree_hash=TREE_HASH,
            gateway_release_hash=RELEASE_HASH,
            parent_environment={
                "RESEARCH_LAB_MINER_SUBMISSIONS_ENABLED": "false",
                "LEADPOET_AWS_INSTANCE_ROLE_ONLY": "true",
                "AWS_REGION": "us-east-1",
                "AWS_DEFAULT_REGION": "us-east-1",
            },
            secrets_client=client,
            release_s3_client=object(),
            hydrated_environment_path=hydrated,
        )


@pytest.mark.parametrize(
    ("parent_value", "secret_value", "message"),
    [
        ("true", "false", "did not hydrate"),
        ("false", "true", "durable gateway secret"),
    ],
)
def test_direct_restart_never_bypasses_false_state(
    monkeypatch: pytest.MonkeyPatch,
    parent_value: str,
    secret_value: str,
    message: str,
):
    client = FakeSecretsClient(
        f"RESEARCH_LAB_MINER_SUBMISSIONS_ENABLED={secret_value}\n"
    )
    monkeypatch.setattr(
        maintenance,
        "_fetch_locked_release_channel",
        lambda **_kwargs: _locked_release_evidence(),
    )
    with pytest.raises(maintenance.GatewayMinerMaintenanceRestartError, match=message):
        maintenance.verify_gateway_miner_maintenance_state(
            deploy_commit=CANDIDATE_COMMIT,
            candidate_tree_hash=TREE_HASH,
            gateway_release_hash=RELEASE_HASH,
            parent_environment={
                "RESEARCH_LAB_MINER_SUBMISSIONS_ENABLED": parent_value,
            },
            secrets_client=client,
            release_s3_client=object(),
        )


def _verification_environment() -> dict[str, str]:
    return {
        "RESEARCH_LAB_MINER_SUBMISSIONS_ENABLED": "false",
        maintenance.PROOF_FD_ENV_NAME: str(maintenance.PROOF_FD_NUMBER),
        "GATEWAY_RESTART_INVOCATION_ID": "gateway-proof-test",
    }


def _fake_running_gateway_proc(
    tmp_path: Path,
    *,
    runtime_commit: str = CONTROLLER_COMMIT,
    controller_helper: str = maintenance.LEGACY_N_MINUS_ONE_GATEWAY_GIT_HELPER,
    identity_overrides: dict[str, str | None] | None = None,
) -> Path:
    proc_root = tmp_path / "proc"
    process = proc_root / "4242"
    process.mkdir(parents=True)
    (process / "cmdline").write_bytes(b"python3\0-m\0gateway.main\0")
    stat_fields = ["S", *("1" for _index in range(18)), "987654"]
    (process / "stat").write_text(
        f"4242 (python3) {' '.join(stat_fields)}\n",
        encoding="ascii",
    )
    overrides = identity_overrides or {}
    environment = []
    for name in maintenance.RUNTIME_BUILD_IDENTITY_NAMES:
        value = overrides.get(name, runtime_commit)
        if value is not None:
            environment.append(f"{name}={value}".encode("ascii"))
    environment.append(f"GATEWAY_GIT_HELPER={controller_helper}".encode("ascii"))
    (process / "environ").write_bytes(b"\0".join(environment) + b"\0")
    return proc_root


def _proof_with_current_hash(
    proof: dict[str, object],
    **updates: str,
) -> dict[str, object]:
    updated = {**proof, **updates}
    body = {
        name: str(updated[name])
        for name in maintenance._PROOF_FIELDS
        if name != "proof_hash"
    }
    updated["proof_hash"] = maintenance.sha256_json(body)
    return updated


@pytest.mark.parametrize("pointer", ["", "191", "not-a-fd"])
def test_open_fixed_proof_fd_cannot_be_downgraded_by_pointer(
    pointer: str,
):
    source = os.open("/dev/null", os.O_RDONLY)
    try:
        os.dup2(source, maintenance.PROOF_FD_NUMBER)
        with pytest.raises(
            maintenance.GatewayMinerMaintenanceRestartError,
            match="pointer was downgraded",
        ):
            maintenance._proof_fd_from_environment(
                {maintenance.PROOF_FD_ENV_NAME: pointer}
            )
    finally:
        os.close(source)
        os.close(maintenance.PROOF_FD_NUMBER)


def test_pointer_without_fixed_proof_fd_fails_closed():
    try:
        os.close(maintenance.PROOF_FD_NUMBER)
    except OSError:
        pass
    with pytest.raises(
        maintenance.GatewayMinerMaintenanceRestartError,
        match="descriptor was lost",
    ):
        maintenance._proof_fd_from_environment(
            {
                maintenance.PROOF_FD_ENV_NAME: str(
                    maintenance.PROOF_FD_NUMBER
                )
            }
        )


@pytest.mark.parametrize(
    "name",
    sorted(maintenance._RESTART_AUTHORITY_NAMES),
)
@pytest.mark.parametrize("value", ["", "/tmp/arbitrary-helper"])
def test_live_gateway_restart_authority_collision_fails_closed(
    name: str,
    value: str,
):
    with pytest.raises(
        maintenance.GatewayMinerMaintenanceRestartError,
        match="restart-only authority",
    ):
        maintenance._require_restart_authority_absent_from_environment_payload(
            f"SAFE=value\0{name}={value}\0".encode("ascii")
        )


def test_exact_frozen_n_minus_one_legacy_git_helper_is_bound_and_accepted():
    commit = maintenance.LEGACY_N_MINUS_ONE_CONTROLLER_COMMIT
    payload = (
        "".join(
            f"{name}={commit}\0"
            for name in maintenance.RUNTIME_BUILD_IDENTITY_NAMES
        )
        + "GATEWAY_GIT_HELPER="
        f"{maintenance.LEGACY_N_MINUS_ONE_GATEWAY_GIT_HELPER}\0"
    ).encode("ascii")

    authority = maintenance._require_restart_authority_absent_from_environment_payload(
        payload,
        expected_runtime_commit=commit,
        verified_controller_commit=commit,
        allow_legacy_n_minus_one_git_helper=True,
    )

    assert authority["restart_authority_names"] == ("GATEWAY_GIT_HELPER",)
    assert authority["runtime_build_identities"] == {
        name: commit for name in maintenance.RUNTIME_BUILD_IDENTITY_NAMES
    }


@pytest.mark.parametrize(
    ("runtime_commit", "controller_commit", "helper_path"),
    [
        (
            maintenance.LEGACY_N_MINUS_ONE_CONTROLLER_COMMIT,
            maintenance.LEGACY_N_MINUS_ONE_CONTROLLER_COMMIT,
            "/tmp/gateway_git_deploy.py",
        ),
        (
            CANDIDATE_COMMIT,
            maintenance.LEGACY_N_MINUS_ONE_CONTROLLER_COMMIT,
            maintenance.LEGACY_N_MINUS_ONE_GATEWAY_GIT_HELPER,
        ),
        (
            maintenance.LEGACY_N_MINUS_ONE_CONTROLLER_COMMIT,
            CANDIDATE_COMMIT,
            maintenance.LEGACY_N_MINUS_ONE_GATEWAY_GIT_HELPER,
        ),
    ],
)
def test_legacy_git_helper_near_misses_fail_closed(
    runtime_commit: str,
    controller_commit: str,
    helper_path: str,
):
    payload = (
        "".join(
            f"{name}={runtime_commit}\0"
            for name in maintenance.RUNTIME_BUILD_IDENTITY_NAMES
        )
        + f"GATEWAY_GIT_HELPER={helper_path}\0"
    ).encode("ascii")

    with pytest.raises(
        maintenance.GatewayMinerMaintenanceRestartError,
        match="restart-only authority|build identity differs",
    ):
        maintenance._require_restart_authority_absent_from_environment_payload(
            payload,
            expected_runtime_commit=runtime_commit,
            verified_controller_commit=controller_commit,
            allow_legacy_n_minus_one_git_helper=True,
        )


@pytest.mark.parametrize("identity_name", maintenance.RUNTIME_BUILD_IDENTITY_NAMES)
@pytest.mark.parametrize("failure_kind", ["missing", "mismatch"])
def test_legacy_git_helper_requires_every_exact_build_identity(
    identity_name: str,
    failure_kind: str,
):
    commit = maintenance.LEGACY_N_MINUS_ONE_CONTROLLER_COMMIT
    records = []
    for name in maintenance.RUNTIME_BUILD_IDENTITY_NAMES:
        if name == identity_name and failure_kind == "missing":
            continue
        value = "b" * 40 if name == identity_name else commit
        records.append(f"{name}={value}\0")
    records.append(
        "GATEWAY_GIT_HELPER="
        f"{maintenance.LEGACY_N_MINUS_ONE_GATEWAY_GIT_HELPER}\0"
    )

    with pytest.raises(
        maintenance.GatewayMinerMaintenanceRestartError,
        match="build identity differs",
    ):
        maintenance._require_restart_authority_absent_from_environment_payload(
            "".join(records).encode("ascii"),
            expected_runtime_commit=commit,
            verified_controller_commit=commit,
            allow_legacy_n_minus_one_git_helper=True,
        )


def test_candidate_runtime_requires_restart_authority_absent():
    payload = "".join(
        f"{name}={CANDIDATE_COMMIT}\0"
        for name in maintenance.RUNTIME_BUILD_IDENTITY_NAMES
    ).encode("ascii")

    authority = maintenance._require_restart_authority_absent_from_environment_payload(
        payload,
        expected_runtime_commit=CANDIDATE_COMMIT,
        verified_controller_commit=CANDIDATE_COMMIT,
    )
    assert authority["restart_authority_names"] == ()
    assert authority["runtime_build_identities"] == {
        name: CANDIDATE_COMMIT
        for name in maintenance.RUNTIME_BUILD_IDENTITY_NAMES
    }


def test_candidate_preflight_accepts_only_proof_bound_exact_n_minus_one_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = FakeSecretsClient("RESEARCH_LAB_MINER_SUBMISSIONS_ENABLED=true\n")
    proof = _prepare(
        tmp_path,
        monkeypatch,
        client,
        invocation_id="gateway-proof-test",
    )["proof"]
    proc_root = _fake_running_gateway_proc(tmp_path)
    live_commitment = REAL_LIVE_GATEWAY_RESTART_AUTHORITY_COMMITMENT(
        expected_runtime_commit=CONTROLLER_COMMIT,
        verified_controller_commit=CONTROLLER_COMMIT,
        allow_legacy_n_minus_one_git_helper=True,
        proc_root=proc_root,
    )
    proof = _proof_with_current_hash(
        proof,
        pre_hydration_live_process_commitment=live_commitment,
    )
    monkeypatch.setattr(maintenance, "_proof_from_fd", lambda _fd: proof)
    monkeypatch.setattr(
        maintenance,
        "_proof_fd_from_environment",
        lambda _environment: maintenance.PROOF_FD_NUMBER,
    )
    monkeypatch.setattr(
        maintenance,
        "_live_gateway_restart_authority_commitment",
        lambda **kwargs: REAL_LIVE_GATEWAY_RESTART_AUTHORITY_COMMITMENT(
            **kwargs,
            proc_root=proc_root,
        ),
    )

    result = maintenance.verify_gateway_miner_maintenance_state(
        deploy_commit=CANDIDATE_COMMIT,
        candidate_tree_hash=TREE_HASH,
        gateway_release_hash=RELEASE_HASH,
        parent_environment=_verification_environment(),
        secrets_client=client,
        release_s3_client=object(),
    )

    assert result["status"] == "invocation_verified"
    assert result["proof_hash"] == proof["proof_hash"]


@pytest.mark.parametrize(
    "failure_case",
    [
        *(f"identity:{name}" for name in maintenance.RUNTIME_BUILD_IDENTITY_NAMES),
        "proof-runtime-candidate",
        "proof-controller-candidate",
        "helper-path-near-miss",
    ],
)
def test_candidate_preflight_rejects_n_minus_one_runtime_near_misses(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_case: str,
) -> None:
    client = FakeSecretsClient("RESEARCH_LAB_MINER_SUBMISSIONS_ENABLED=true\n")
    proof = _prepare(
        tmp_path,
        monkeypatch,
        client,
        invocation_id="gateway-proof-test",
    )["proof"]
    identity_overrides: dict[str, str | None] = {}
    helper_path = maintenance.LEGACY_N_MINUS_ONE_GATEWAY_GIT_HELPER
    if failure_case.startswith("identity:"):
        identity_overrides[failure_case.split(":", 1)[1]] = CANDIDATE_COMMIT
    elif failure_case == "proof-runtime-candidate":
        proof = _proof_with_current_hash(
            proof,
            pre_hydration_runtime_commit=CANDIDATE_COMMIT,
        )
    elif failure_case == "proof-controller-candidate":
        proof = _proof_with_current_hash(
            proof,
            n_minus_one_controller_commit=CANDIDATE_COMMIT,
        )
    elif failure_case == "helper-path-near-miss":
        helper_path = "/tmp/gateway_git_deploy.py"
    proc_root = _fake_running_gateway_proc(
        tmp_path,
        identity_overrides=identity_overrides,
        controller_helper=helper_path,
    )
    monkeypatch.setattr(maintenance, "_proof_from_fd", lambda _fd: proof)
    monkeypatch.setattr(
        maintenance,
        "_proof_fd_from_environment",
        lambda _environment: maintenance.PROOF_FD_NUMBER,
    )
    monkeypatch.setattr(
        maintenance,
        "_live_gateway_restart_authority_commitment",
        lambda **kwargs: REAL_LIVE_GATEWAY_RESTART_AUTHORITY_COMMITMENT(
            **kwargs,
            proc_root=proc_root,
        ),
    )

    with pytest.raises(
        maintenance.GatewayMinerMaintenanceRestartError,
        match="build identity differs|restart-only authority",
    ):
        maintenance.verify_gateway_miner_maintenance_state(
            deploy_commit=CANDIDATE_COMMIT,
            candidate_tree_hash=TREE_HASH,
            gateway_release_hash=RELEASE_HASH,
            parent_environment=_verification_environment(),
            secrets_client=client,
            release_s3_client=object(),
        )


def test_candidate_preflight_binds_sealed_proof_to_exact_secret_and_channel(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    client = FakeSecretsClient("RESEARCH_LAB_MINER_SUBMISSIONS_ENABLED=true\n")
    prepared = _prepare(
        tmp_path,
        monkeypatch,
        client,
        invocation_id="gateway-proof-test",
    )
    proof = prepared["proof"]
    monkeypatch.setattr(maintenance, "_proof_from_fd", lambda _fd: proof)
    monkeypatch.setattr(
        maintenance,
        "_proof_fd_from_environment",
        lambda _environment: maintenance.PROOF_FD_NUMBER,
    )

    verified = maintenance.verify_gateway_miner_maintenance_state(
        deploy_commit=CANDIDATE_COMMIT,
        candidate_tree_hash=TREE_HASH,
        gateway_release_hash=RELEASE_HASH,
        parent_environment=_verification_environment(),
        secrets_client=client,
        release_s3_client=object(),
    )
    assert verified["status"] == "invocation_verified"
    assert verified["proof_hash"] == proof["proof_hash"]

    client.install_concurrent_current()
    with pytest.raises(
        disable_operation.GatewayMinerSubmissionsDisableError,
        match="differs from the expected current version",
    ):
        maintenance.verify_gateway_miner_maintenance_state(
            deploy_commit=CANDIDATE_COMMIT,
            candidate_tree_hash=TREE_HASH,
            gateway_release_hash=RELEASE_HASH,
            parent_environment=_verification_environment(),
            secrets_client=client,
            release_s3_client=object(),
        )

@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("deploy_commit", "f" * 40),
        ("candidate_tree_hash", "f" * 40),
        ("gateway_release_hash", "sha256:" + "f" * 64),
    ],
)
def test_candidate_preflight_rejects_proof_candidate_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: str,
):
    client = FakeSecretsClient("RESEARCH_LAB_MINER_SUBMISSIONS_ENABLED=true\n")
    proof = _prepare(
        tmp_path,
        monkeypatch,
        client,
        invocation_id="gateway-proof-test",
    )["proof"]
    monkeypatch.setattr(maintenance, "_proof_from_fd", lambda _fd: proof)
    monkeypatch.setattr(
        maintenance,
        "_proof_fd_from_environment",
        lambda _environment: maintenance.PROOF_FD_NUMBER,
    )
    arguments = {
        "deploy_commit": CANDIDATE_COMMIT,
        "candidate_tree_hash": TREE_HASH,
        "gateway_release_hash": RELEASE_HASH,
        "parent_environment": _verification_environment(),
        "secrets_client": client,
        "release_s3_client": object(),
    }
    arguments[field] = value

    with pytest.raises(
        maintenance.GatewayMinerMaintenanceRestartError,
        match="differs from the candidate",
    ):
        maintenance.verify_gateway_miner_maintenance_state(**arguments)


def test_invocation_proof_rejects_topology_and_locked_channel_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    client = FakeSecretsClient("RESEARCH_LAB_MINER_SUBMISSIONS_ENABLED=true\n")
    proof = _prepare(
        tmp_path,
        monkeypatch,
        client,
        invocation_id="gateway-proof-test",
    )["proof"]
    monkeypatch.setattr(maintenance, "_proof_from_fd", lambda _fd: proof)
    monkeypatch.setattr(
        maintenance,
        "_proof_fd_from_environment",
        lambda _environment: maintenance.PROOF_FD_NUMBER,
    )
    client.stages[client.current].add("UNEXPECTED_LABEL")

    with pytest.raises(
        maintenance.GatewayMinerMaintenanceRestartError,
        match="durable miner-maintenance state differs",
    ):
        maintenance.verify_gateway_miner_maintenance_state(
            deploy_commit=CANDIDATE_COMMIT,
            candidate_tree_hash=TREE_HASH,
            gateway_release_hash=RELEASE_HASH,
            parent_environment=_verification_environment(),
            secrets_client=client,
            release_s3_client=object(),
        )

    client.stages[client.current].remove("UNEXPECTED_LABEL")
    drifted = _locked_release_evidence()
    drifted["object_version_id"] = "replacement-version-identity-000001"
    monkeypatch.setattr(
        maintenance,
        "_fetch_locked_release_channel",
        lambda **_kwargs: drifted,
    )
    with pytest.raises(
        maintenance.GatewayMinerMaintenanceRestartError,
        match="immutable release differs",
    ):
        maintenance.verify_gateway_miner_maintenance_state(
            deploy_commit=CANDIDATE_COMMIT,
            candidate_tree_hash=TREE_HASH,
            gateway_release_hash=RELEASE_HASH,
            parent_environment=_verification_environment(),
            secrets_client=client,
            release_s3_client=object(),
        )


def test_invocation_proof_rejects_false_document_hydration_aba(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    client = FakeSecretsClient(
        "CONFIG_GENERATION=v1\n"
        "RESEARCH_LAB_MINER_SUBMISSIONS_ENABLED=true\n"
    )
    proof = _prepare(
        tmp_path,
        monkeypatch,
        client,
        invocation_id="gateway-proof-hydration-aba",
    )["proof"]
    monkeypatch.setattr(maintenance, "_proof_from_fd", lambda _fd: proof)
    monkeypatch.setattr(
        maintenance,
        "_proof_fd_from_environment",
        lambda _environment: maintenance.PROOF_FD_NUMBER,
    )
    monkeypatch.setattr(
        maintenance,
        "_require_hydrated_environment_commitment",
        REAL_REQUIRE_HYDRATED_ENVIRONMENT_COMMITMENT,
    )
    hydrated_path = tmp_path / "hydrated-cache" / "gateway.env"
    hydrated_path.parent.mkdir(mode=0o700)
    hydrated_path.write_text(
        disable_operation._n_minus_one_hydrated_environment(
            "CONFIG_GENERATION=v2\n"
            "RESEARCH_LAB_MINER_SUBMISSIONS_ENABLED=false\n"
        ),
        encoding="utf-8",
    )
    hydrated_path.chmod(0o600)

    with pytest.raises(
        maintenance.GatewayMinerMaintenanceRestartError,
        match="hydrated gateway environment differs",
    ):
        maintenance.verify_gateway_miner_maintenance_state(
            deploy_commit=CANDIDATE_COMMIT,
            candidate_tree_hash=TREE_HASH,
            gateway_release_hash=RELEASE_HASH,
            parent_environment={
                **_verification_environment(),
                "GATEWAY_RESTART_INVOCATION_ID": "gateway-proof-hydration-aba",
            },
            secrets_client=client,
            release_s3_client=object(),
            hydrated_environment_path=hydrated_path,
        )


def test_identical_document_alternate_version_hydration_equivalence_passes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    client = FakeSecretsClient(
        "CONFIG_GENERATION=v1\n"
        "RESEARCH_LAB_MINER_SUBMISSIONS_ENABLED=true\n"
    )
    invocation_id = "gateway-proof-identical-version-aba"
    proof = _prepare(
        tmp_path,
        monkeypatch,
        client,
        invocation_id=invocation_id,
    )["proof"]
    proof_current = client.current
    proof_secret = client.versions[proof_current]
    proof_stages = {
        version: set(labels) for version, labels in client.stages.items()
    }
    client.versions[CONCURRENT_VERSION] = proof_secret
    client.stages[CONCURRENT_VERSION] = {"AWSCURRENT"}
    client.stages[proof_current].discard("AWSCURRENT")
    hydrated_path = tmp_path / "identical-cache" / "gateway.env"
    hydrated_path.parent.mkdir(mode=0o700)
    hydrated_path.write_text(
        disable_operation._n_minus_one_hydrated_environment(
            client.versions[CONCURRENT_VERSION]
        ),
        encoding="utf-8",
    )
    hydrated_path.chmod(0o600)
    client.stages = proof_stages
    client.stages[CONCURRENT_VERSION] = set()
    monkeypatch.setattr(maintenance, "_proof_from_fd", lambda _fd: proof)
    monkeypatch.setattr(
        maintenance,
        "_proof_fd_from_environment",
        lambda _environment: maintenance.PROOF_FD_NUMBER,
    )
    monkeypatch.setattr(
        maintenance,
        "_require_hydrated_environment_commitment",
        REAL_REQUIRE_HYDRATED_ENVIRONMENT_COMMITMENT,
    )

    result = maintenance.verify_gateway_miner_maintenance_state(
        deploy_commit=CANDIDATE_COMMIT,
        candidate_tree_hash=TREE_HASH,
        gateway_release_hash=RELEASE_HASH,
        parent_environment={
            **_verification_environment(),
            "GATEWAY_RESTART_INVOCATION_ID": invocation_id,
        },
        secrets_client=client,
        release_s3_client=object(),
        hydrated_environment_path=hydrated_path,
    )

    assert result["status"] == "invocation_verified"
    assert result["current_secret_version_id"] == proof_current


@pytest.mark.parametrize("runtime_value", [True, None, "false", 0])
def test_runtime_state_requires_live_boolean_exact_false(
    monkeypatch: pytest.MonkeyPatch,
    runtime_value,
):
    client = FakeSecretsClient("RESEARCH_LAB_MINER_SUBMISSIONS_ENABLED=false\n")
    monkeypatch.setattr(
        maintenance,
        "_fetch_locked_release_channel",
        lambda **_kwargs: _locked_release_evidence(),
    )
    with pytest.raises(
        maintenance.GatewayMinerMaintenanceRestartError,
        match="running gateway has miner submissions enabled",
    ):
        maintenance.verify_gateway_miner_maintenance_runtime_state(
            deploy_commit=CANDIDATE_COMMIT,
            candidate_tree_hash=TREE_HASH,
            gateway_release_hash=RELEASE_HASH,
            runtime_environment={
                "RESEARCH_LAB_MINER_SUBMISSIONS_ENABLED": "false",
            },
            runtime_status={"miner_submissions_enabled": runtime_value},
            secrets_client=client,
            release_s3_client=object(),
        )


def test_runtime_state_rechecks_live_false_durable_state_and_channel(
    monkeypatch: pytest.MonkeyPatch,
):
    client = FakeSecretsClient("RESEARCH_LAB_MINER_SUBMISSIONS_ENABLED=false\n")
    monkeypatch.setattr(
        maintenance,
        "_fetch_locked_release_channel",
        lambda **_kwargs: _locked_release_evidence(),
    )
    result = maintenance.verify_gateway_miner_maintenance_runtime_state(
        deploy_commit=CANDIDATE_COMMIT,
        candidate_tree_hash=TREE_HASH,
        gateway_release_hash=RELEASE_HASH,
        runtime_environment={
            "RESEARCH_LAB_MINER_SUBMISSIONS_ENABLED": "false",
        },
        runtime_status={"miner_submissions_enabled": False},
        secrets_client=client,
        release_s3_client=object(),
    )
    assert result["runtime_status"] == "disabled"
    assert result["status"] == "durable_false_verified"


def test_runtime_status_fetch_uses_exact_loopback_path_and_never_follows_redirects(
    monkeypatch: pytest.MonkeyPatch,
):
    requests: list[tuple[object, ...]] = []

    class RedirectResponse:
        status = 302

        def getheader(self, _name):
            return None

    class FakeConnection:
        def __init__(self, host, port, timeout):
            requests.append(("connect", host, port, timeout))

        def request(self, method, path, body, headers):
            requests.append(("request", method, path, body, dict(headers)))

        def getresponse(self):
            return RedirectResponse()

        def close(self):
            requests.append(("close",))

    monkeypatch.setattr(maintenance.http.client, "HTTPConnection", FakeConnection)

    with pytest.raises(
        maintenance.GatewayMinerMaintenanceRestartError,
        match="response is not successful",
    ):
        maintenance._fetch_runtime_status()

    assert requests == [
        ("connect", "127.0.0.1", 8000, 15.0),
        (
            "request",
            "GET",
            "/research-lab/status",
            None,
            {"Host": "127.0.0.1:8000", "Connection": "close"},
        ),
        ("close",),
    ]
    with pytest.raises(
        maintenance.GatewayMinerMaintenanceRestartError,
        match="URL is not canonical",
    ):
        maintenance._fetch_runtime_status(
            url="http://169.254.169.254/latest/meta-data/",
        )


@pytest.mark.skipif(
    not hasattr(os, "memfd_create")
    or not hasattr(maintenance.fcntl, "F_ADD_SEALS"),
    reason="Linux sealed memfd behavior",
)
def test_sealed_invocation_proof_survives_exec_and_rejects_writes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    client = FakeSecretsClient("RESEARCH_LAB_MINER_SUBMISSIONS_ENABLED=true\n")
    proof = _prepare(
        tmp_path,
        monkeypatch,
        client,
        invocation_id="gateway-proof-test",
    )["proof"]
    try:
        maintenance._seal_payload_at_fd_number(
            payload=maintenance._serialized_proof(proof),
            fd_number=maintenance.PROOF_FD_NUMBER,
            name="test-proof",
            max_bytes=maintenance.MAX_PROOF_BYTES,
        )
        assert maintenance._proof_from_fd(maintenance.PROOF_FD_NUMBER) == proof
        with pytest.raises(OSError):
            os.write(maintenance.PROOF_FD_NUMBER, b"tamper")
        environment = dict(os.environ)
        environment[maintenance.PROOF_FD_ENV_NAME] = str(
            maintenance.PROOF_FD_NUMBER
        )
        child = subprocess.run(
            [
                "/bin/bash",
                "-c",
                'test "$GATEWAY_MINER_MAINTENANCE_PROOF_FD" = 190 '
                '&& test -r /proc/$$/fd/190 '
                '&& exec python3 -c "import os; os.fstat(190)"',
            ],
            check=False,
            env=environment,
            pass_fds=(maintenance.PROOF_FD_NUMBER,),
        )
        assert child.returncode == 0
    finally:
        try:
            os.close(maintenance.PROOF_FD_NUMBER)
        except OSError:
            pass


@pytest.mark.skipif(
    not hasattr(os, "memfd_create")
    or not hasattr(maintenance.fcntl, "F_ADD_SEALS"),
    reason="Linux sealed memfd behavior",
)
def test_closed_or_tampered_proof_fd_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
):
    try:
        os.close(maintenance.PROOF_FD_NUMBER)
    except OSError:
        pass
    with pytest.raises(
        maintenance.GatewayMinerMaintenanceRestartError,
        match="unavailable",
    ):
        maintenance._proof_from_fd(maintenance.PROOF_FD_NUMBER)

    proof = {
        name: "invalid"
        for name in maintenance._PROOF_FIELDS
    }
    proof["schema_version"] = maintenance.SCHEMA_VERSION
    with pytest.raises(
        maintenance.GatewayMinerMaintenanceRestartError,
        match="commitments are invalid",
    ):
        maintenance._validate_proof_document(proof)


def test_bootstrap_cleanup_leaves_a_valid_exec_working_directory() -> None:
    bootstrap_root = Path(
        f"/tmp/gateway-miner-maintenance-bootstrap.{os.getpid()}cwd"
    )
    candidate_root = bootstrap_root / "candidate"
    original_cwd = os.open(".", os.O_RDONLY)
    try:
        bootstrap_root.mkdir(mode=0o700)
        candidate_root.mkdir()
        os.chdir(candidate_root)

        maintenance._leave_and_close_bootstrap_tree(bootstrap_root)

        assert Path.cwd() == Path("/")
        assert not bootstrap_root.exists()
    finally:
        os.fchdir(original_cwd)
        os.close(original_cwd)
        if bootstrap_root.exists():
            maintenance.shutil.rmtree(bootstrap_root)


def test_unexpected_cli_failure_never_renders_exception_detail(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    secret_marker = "raw-secret-must-not-render"
    monkeypatch.setattr(
        maintenance,
        "bootstrap_gateway_miner_maintenance_restart",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError(secret_marker)),
    )

    status = maintenance.main(
        [
            "--bootstrap-exec",
            "--expected-commit",
            CANDIDATE_COMMIT,
            "--plan-file",
            "/tmp/nonexistent-plan",
            "--bootstrap-root",
            "/tmp/gateway-miner-maintenance-bootstrap.test",
            "--handoff-file",
            "/tmp/leadpoet-gateway-miner-maintenance-handoff.test",
            "--handoff-nonce",
            "0" * 64,
        ]
    )

    captured = capsys.readouterr()
    assert status == 2
    assert secret_marker not in captured.err
    assert "unexpected miner-maintenance restart failure" in captured.err


def test_candidate_identity_binds_isolated_n_minus_one_plan_and_archive(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    repo = tmp_path / "repo"
    candidate = tmp_path / "candidate"
    repo.mkdir()
    candidate.mkdir()
    plan = tmp_path / "plan.json"
    plan.write_text(
        json.dumps(
            {
                "schema_version": maintenance.GIT_DEPLOYMENT_SCHEMA_VERSION,
                "source": "github",
                "status": "prepared",
                "stage": "git_prepare",
                "mode": "pinned",
                "branch": maintenance.DEFAULT_BRANCH,
                "target_sha": CANDIDATE_COMMIT,
                "branch_head_sha": CANDIDATE_COMMIT,
                "repo_root": str(repo.resolve()),
                "remote_url": maintenance.DEFAULT_REPO_URL,
                "previous_sha": "9" * 40,
                "tree_hash": TREE_HASH,
            }
        ),
        encoding="utf-8",
    )
    responses = {
        ("rev-parse", "HEAD"): "9" * 40,
        ("rev-parse", "origin/main^{commit}"): CANDIDATE_COMMIT,
        ("remote", "get-url", "origin"): maintenance.DEFAULT_REPO_URL,
        ("status", "--porcelain=v1", "--untracked-files=all"): "",
    }
    monkeypatch.setattr(
        maintenance,
        "_run_git",
        lambda _repo, *arguments: responses[arguments],
    )
    monkeypatch.setattr(
        maintenance,
        "_require_unmodified_git_object_authority",
        lambda _repo: None,
    )
    monkeypatch.setattr(
        maintenance,
        "verify_materialized_tree",
        lambda **_kwargs: {
            "tree_hash": TREE_HASH,
            "blob_manifest_sha256": BLOB_HASH,
        },
    )
    monkeypatch.setattr(
        maintenance,
        "_verified_installed_controller_bundle",
        lambda **_kwargs: _controller_bundle(),
    )

    evidence = maintenance._validate_candidate_identity(
        repo_root=repo,
        candidate_root=candidate,
        plan_file=plan,
        expected_commit=CANDIDATE_COMMIT,
        controller_current=tmp_path / "controller/current",
        host_restart_path=tmp_path / "gw_restart.sh",
    )

    assert evidence["tree_hash"] == TREE_HASH
    tampered = json.loads(plan.read_text(encoding="utf-8"))
    tampered["branch_head_sha"] = "f" * 40
    plan.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(
        maintenance.GatewayMinerMaintenanceRestartError,
        match="differs from the exact candidate",
    ):
        maintenance._validate_candidate_identity(
            repo_root=repo,
            candidate_root=candidate,
            plan_file=plan,
            expected_commit=CANDIDATE_COMMIT,
            controller_current=tmp_path / "controller/current",
            host_restart_path=tmp_path / "gw_restart.sh",
        )


def test_git_replacement_and_graft_authority_fail_before_candidate_resolution(
    tmp_path: Path,
):
    repository = tmp_path / "git-authority"
    repository.mkdir()
    subprocess.run(["git", "init", "-q", str(repository)], check=True)
    tracked = repository / "tracked.txt"
    tracked.write_text("official\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(repository), "add", "tracked.txt"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(repository),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.com",
            "commit",
            "-qm",
            "official",
        ],
        check=True,
    )
    official = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    tracked.write_text("replacement\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(repository), "add", "tracked.txt"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(repository),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.com",
            "commit",
            "-qm",
            "replacement",
        ],
        check=True,
    )
    replacement = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    subprocess.run(
        ["git", "-C", str(repository), "replace", official, replacement],
        check=True,
    )

    with pytest.raises(
        maintenance.GatewayMinerMaintenanceRestartError,
        match="replacement refs",
    ):
        maintenance._require_unmodified_git_object_authority(repository)

    subprocess.run(
        ["git", "-C", str(repository), "replace", "-d", official],
        check=True,
    )
    graft = Path(
        subprocess.run(
            ["git", "-C", str(repository), "rev-parse", "--git-path", "info/grafts"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    if not graft.is_absolute():
        graft = repository / graft
    graft.parent.mkdir(parents=True, exist_ok=True)
    graft.write_text(f"{replacement} {official}\n", encoding="ascii")
    with pytest.raises(
        maintenance.GatewayMinerMaintenanceRestartError,
        match="graft or alternate",
    ):
        maintenance._require_unmodified_git_object_authority(repository)


def test_git_object_override_environment_is_rejected(monkeypatch):
    monkeypatch.setenv("GIT_OBJECT_DIRECTORY", "/tmp/unsafe-objects")
    with pytest.raises(
        maintenance.GatewayMinerMaintenanceRestartError,
        match="object-resolution overrides",
    ):
        maintenance._safe_git_environment()


def test_live_0775_controller_ancestry_is_hardened_and_all_four_files_bound(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    (
        controller_parent,
        controller_root,
        releases_root,
        _release,
        current,
        host_restart,
    ) = _installed_controller_fixture(tmp_path, monkeypatch)

    observed = maintenance._verify_installed_controller(
        repo_root=tmp_path,
        controller_current=current,
        host_restart_path=host_restart,
        expected_commit=CANDIDATE_COMMIT,
    )

    assert observed == CONTROLLER_COMMIT
    assert [
        path.stat().st_mode & 0o777
        for path in (controller_parent, controller_root, releases_root)
    ] == [0o700, 0o700, 0o700]


def test_controller_hardening_rejects_wrong_owner_and_symlink_ancestry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    (
        _controller_parent,
        controller_root,
        _releases_root,
        _release,
        current,
        host_restart,
    ) = _installed_controller_fixture(tmp_path, monkeypatch)
    actual_euid = os.geteuid()
    monkeypatch.setattr(maintenance.os, "geteuid", lambda: actual_euid + 1)
    with pytest.raises(
        maintenance.GatewayMinerMaintenanceRestartError,
        match="ancestry is unsafe",
    ):
        maintenance._verify_installed_controller(
            repo_root=tmp_path,
            controller_current=current,
            host_restart_path=host_restart,
            expected_commit=CANDIDATE_COMMIT,
        )
    monkeypatch.undo()

    (
        _controller_parent,
        second_root,
        _releases_root,
        _release,
        second_current,
        second_host,
    ) = _installed_controller_fixture(
        tmp_path / "second",
        monkeypatch,
    )
    real_root = second_root.with_name("gateway-real")
    second_root.rename(real_root)
    second_root.symlink_to(real_root, target_is_directory=True)
    with pytest.raises(
        maintenance.GatewayMinerMaintenanceRestartError,
        match="ancestry",
    ):
        maintenance._verify_installed_controller(
            repo_root=tmp_path,
            controller_current=second_current,
            host_restart_path=second_host,
            expected_commit=CANDIDATE_COMMIT,
        )


def test_controller_verifier_rejects_tampered_memory_guard(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    (
        _controller_parent,
        _controller_root,
        _releases_root,
        release,
        current,
        host_restart,
    ) = _installed_controller_fixture(tmp_path, monkeypatch)
    memory_guard = release / "gateway/tee/host_memory_guard_v2.py"
    memory_guard.write_bytes(b"TAMPERED = True\n")
    memory_guard.chmod(0o600)

    with pytest.raises(
        maintenance.GatewayMinerMaintenanceRestartError,
        match="bytes differ",
    ):
        maintenance._verify_installed_controller(
            repo_root=tmp_path,
            controller_current=current,
            host_restart_path=host_restart,
            expected_commit=CANDIDATE_COMMIT,
        )


def test_exact_candidate_controller_is_allowed_for_post_install_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    (
        _controller_parent,
        _controller_root,
        _releases_root,
        _release,
        current,
        host_restart,
    ) = _installed_controller_fixture(
        tmp_path,
        monkeypatch,
        controller_commit=CANDIDATE_COMMIT,
    )

    assert maintenance._verify_installed_controller(
        repo_root=tmp_path,
        controller_current=current,
        host_restart_path=host_restart,
        expected_commit=CANDIDATE_COMMIT,
    ) == CANDIDATE_COMMIT


def test_partial_controller_cutover_reconciles_exact_old_host_under_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    (
        _controller_parent,
        _controller_root,
        _releases_root,
        release,
        current,
        host_restart,
    ) = _installed_controller_fixture(
        tmp_path,
        monkeypatch,
        controller_commit=CANDIDATE_COMMIT,
    )
    candidate_payloads = {
        "gw_restart.sh": (release / "gw_restart.sh").read_bytes(),
        "scripts/gateway_git_deploy.py": (
            release / "scripts/gateway_git_deploy.py"
        ).read_bytes(),
        "Leadpoet/utils/exact_commit_restart_v2.py": (
            release / "Leadpoet/utils/exact_commit_restart_v2.py"
        ).read_bytes(),
        "gateway/tee/host_memory_guard_v2.py": (
            release / "gateway/tee/host_memory_guard_v2.py"
        ).read_bytes(),
    }
    old_wrapper = b"#!/bin/bash\n# exact supported N-1 wrapper\nexit 0\n"
    host_restart.write_bytes(old_wrapper)
    host_restart.chmod(0o700)

    def git_bytes(_repo, _show, object_name):
        commit, relative_path = object_name.split(":", 1)
        if commit == CANDIDATE_COMMIT:
            return candidate_payloads[relative_path]
        assert commit == CONTROLLER_COMMIT
        assert relative_path == "gw_restart.sh"
        return old_wrapper

    monkeypatch.setattr(maintenance, "_run_git_bytes", git_bytes)

    with pytest.raises(
        maintenance.GatewayMinerMaintenanceRestartError,
        match="differs from current controller",
    ):
        maintenance._verified_installed_controller_bundle(
            repo_root=tmp_path,
            controller_current=current,
            host_restart_path=host_restart,
            expected_commit=CANDIDATE_COMMIT,
        )

    bundle = maintenance._verified_installed_controller_bundle(
        repo_root=tmp_path,
        controller_current=current,
        host_restart_path=host_restart,
        expected_commit=CANDIDATE_COMMIT,
        reconcile_host_wrapper=True,
    )

    assert bundle["controller_commit"] == CANDIDATE_COMMIT
    assert host_restart.read_bytes() == candidate_payloads["gw_restart.sh"]
    assert host_restart.stat().st_mode & 0o777 == 0o700


def test_exact_deployed_n_minus_one_preserves_proof_until_candidate_gates():
    root = Path(__file__).resolve().parents[1]
    deployed_n_minus_one = subprocess.run(
        [
            "git",
            "show",
            "0dd3a385a23a3af0fa17210bfe02a39cc4023952:gw_restart.sh",
        ],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    ).stdout
    hydration = deployed_n_minus_one.index(
        "Hydrating gateway env from Secrets Manager before stopping processes"
    )
    git_prepare = deployed_n_minus_one.index(
        'GATEWAY_DEPLOY_STAGE="git_prepare"', hydration
    )
    archive = deployed_n_minus_one.index(
        'git -C "$LEADPOET_REPO_ROOT" archive "$PREPARED_GATEWAY_SHA"',
        git_prepare,
    )
    candidate_preflight = deployed_n_minus_one.index(
        "gateway.tee.restart_preflight_v2", archive
    )
    shutdown = deployed_n_minus_one.index(
        "Stopping existing gateway and Research Lab worker processes",
        candidate_preflight,
    )
    assert hydration < git_prepare < archive < candidate_preflight < shutdown
    post_activate = deployed_n_minus_one.index("exec env", shutdown)
    assert "env -i" not in deployed_n_minus_one[post_activate:post_activate + 200]
    assert "190>&-" not in deployed_n_minus_one

    preflight_source = (root / "gateway/tee/restart_preflight_v2.py").read_text(
        encoding="utf-8"
    )
    tree_verification = preflight_source.index("write_tree_verification_evidence")
    state_gate = preflight_source.index(
        "verify_gateway_miner_maintenance_state(",
        tree_verification,
    )
    shared_aws_authority = preflight_source.index(
        "aws_clients = _instance_role_aws_clients(",
        tree_verification,
    )
    output = preflight_source.index(
        "print(json.dumps(result, sort_keys=True, indent=2))",
        state_gate,
    )
    assert tree_verification < shared_aws_authority < state_gate < output
    assert 'boto3.client("s3")' not in preflight_source
    assert 'artifact_s3_client=aws_clients["s3"]' in preflight_source
    assert 'secrets_client=aws_clients["secretsmanager"]' in preflight_source

    candidate_restart = (root / "gw_restart.sh").read_text(encoding="utf-8")
    health = candidate_restart.index("Verifying Research Lab maintenance state")
    install = candidate_restart.index(
        'GATEWAY_DEPLOY_STAGE="host_restart_script_install"', health
    )
    runtime_verify = candidate_restart.index("--verify-runtime", install)
    finalize = candidate_restart.index(
        "finalize_deployment_record succeeded", runtime_verify
    )
    close_parent = candidate_restart.index(
        "exec 190>&- 191>&- 192>&- 193>&- 194>&-",
        finalize,
    )
    completed = candidate_restart.index("GATEWAY_DEPLOY_COMPLETED=1", close_parent)
    assert health < install < runtime_verify < finalize < close_parent < completed


def test_long_lived_runtime_children_receive_no_proof_or_controller_fds():
    restart = (
        Path(__file__).resolve().parents[1] / "gw_restart.sh"
    ).read_text(encoding="utf-8")
    close_set = "190>&- 191>&- 192>&- 193>&- 194>&-"
    for module_name in (
        "gateway.utils.tee_egress_forwarder",
        "gateway.utils.tee_inter_enclave_relay",
        "gateway.main",
    ):
        position = restart.rindex(f"-m {module_name}")
        position = restart.rfind("env -u", 0, position)
        command = restart[position:position + 700]
        assert close_set in command
        assert "-u GATEWAY_MINER_MAINTENANCE_PROOF_FD" in command
        assert "-u GATEWAY_GIT_HELPER" in command
        assert "-u GATEWAY_EXACT_COMMIT_HELPER" in command
        assert "-u GATEWAY_HOST_MEMORY_GUARD_PATH" in command
    for function_name, marker in (
        ("start_gateway_offline_artifact_prepare", '"${prepare_command[@]}"'),
        ("start_gateway_ancestry_checkpoint_bootstrap", '"${checkpoint_command[@]}"'),
    ):
        function_start = restart.index(f"{function_name}() {{")
        command_start = restart.index("env -u", function_start)
        position = restart.index(marker, command_start)
        command = restart[command_start:position + 300]
        assert close_set in command
        assert "-u GATEWAY_MINER_MAINTENANCE_PROOF_FD" in command
        assert "-u GATEWAY_GIT_HELPER" in command
        assert "-u GATEWAY_EXACT_COMMIT_HELPER" in command
        assert "-u GATEWAY_HOST_MEMORY_GUARD_PATH" in command


def test_hydrated_and_live_env_clones_reserve_invocation_only_keys():
    restart = (
        Path(__file__).resolve().parents[1] / "gw_restart.sh"
    ).read_text(encoding="utf-8")
    assert restart.count('"GATEWAY_MINER_MAINTENANCE_PROOF_FD",') >= 3
    assert restart.count('"GATEWAY_EXACT_COMMIT_HELPER",') >= 3
    assert restart.count('"GATEWAY_HOST_MEMORY_GUARD_PATH",') >= 3


@pytest.mark.skipif(
    sys.platform != "linux",
    reason="production installer uses GNU stat and Linux mv -T",
)
@pytest.mark.parametrize("crash_point", ["release", "current", "host"])
def test_controller_install_recovers_every_publication_crash_point(
    tmp_path: Path,
    crash_point: str,
):
    repository = tmp_path / "candidate"
    controller_root = tmp_path / "restart-controller" / "gateway"
    releases_root = controller_root / "releases"
    candidate_release = releases_root / CANDIDATE_COMMIT
    previous_release = releases_root / CONTROLLER_COMMIT
    host_restart = tmp_path / "gw_restart.sh"
    candidate_payloads = {
        "gw_restart.sh": b"#!/bin/bash\necho candidate\n",
        "scripts/gateway_git_deploy.py": b"CANDIDATE_HELPER = True\n",
        "Leadpoet/utils/exact_commit_restart_v2.py": b"CANDIDATE_EXACT = True\n",
        "gateway/tee/host_memory_guard_v2.py": b"CANDIDATE_GUARD = True\n",
    }
    for relative_path, payload in candidate_payloads.items():
        source = repository / relative_path
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_bytes(payload)
        source.chmod(0o700 if relative_path == "gw_restart.sh" else 0o600)
        installed = candidate_release / relative_path
        installed.parent.mkdir(parents=True, exist_ok=True)
        installed.write_bytes(payload)
        installed.chmod(0o700 if relative_path == "gw_restart.sh" else 0o600)
    candidate_release.chmod(0o700)
    previous_release.mkdir(parents=True)
    previous_wrapper = previous_release / "gw_restart.sh"
    previous_wrapper.write_bytes(b"#!/bin/bash\necho previous\n")
    previous_wrapper.chmod(0o700)
    previous_release.chmod(0o700)
    current = controller_root / "current"
    current.symlink_to(
        f"releases/{CANDIDATE_COMMIT if crash_point != 'release' else CONTROLLER_COMMIT}"
    )
    host_restart.write_bytes(
        candidate_payloads["gw_restart.sh"]
        if crash_point == "host"
        else previous_wrapper.read_bytes()
    )
    host_restart.chmod(0o700)

    restart_source = (
        Path(__file__).resolve().parents[1] / "gw_restart.sh"
    ).read_text(encoding="utf-8")
    body = restart_source.split(
        "install_successful_restart_script() {\n",
        1,
    )[1].split("\n}\n\ninstall_research_lab_admin_wrapper()", 1)[0]
    script = (
        "set -euo pipefail\n"
        "install_successful_restart_script() {\n"
        + body
        + "\n}\ninstall_successful_restart_script\n"
    )
    result = subprocess.run(
        ["bash", "-c", script],
        check=False,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "GATEWAY_DEPLOY_SHA": CANDIDATE_COMMIT,
            "GATEWAY_RESTART_CONTROLLER_ROOT": str(controller_root),
            "GATEWAY_RESTART_CONTROLLER_CURRENT": str(current),
            "GATEWAY_HOST_RESTART_SCRIPT": str(host_restart),
            "LEADPOET_REPO_ROOT": str(repository),
        },
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    assert os.readlink(current) == f"releases/{CANDIDATE_COMMIT}"
    assert host_restart.read_bytes() == candidate_payloads["gw_restart.sh"]
    assert stat.S_IMODE(host_restart.stat().st_mode) == 0o700
    for relative_path, payload in candidate_payloads.items():
        installed = candidate_release / relative_path
        assert installed.read_bytes() == payload
        assert stat.S_IMODE(installed.stat().st_mode) == (
            0o700 if relative_path == "gw_restart.sh" else 0o600
        )
