"""Reject forged, misplaced and stale build artifacts before AWS admission."""

import importlib.util
import json
import os
from pathlib import Path
import subprocess

import pytest

from gateway.tee.release_manifest_v2 import (
    BUILD_EVIDENCE_SCHEMA_VERSION,
    ReleaseManifestV2Error,
)
from gateway.tee.topology import ROLE_SPECS, topology_hash


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / ".github/scripts/verify_attested_release_evidence.py"
SPEC = importlib.util.spec_from_file_location("attested_release_evidence", SCRIPT)
EVIDENCE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EVIDENCE)
COMMIT = "1" * 40


def _rows(domain):
    return [
        {
            "schema_version": BUILD_EVIDENCE_SCHEMA_VERSION,
            "builder_domain": domain,
            "builder_id": domain + "-parent-123-1",
            "build_ordinal": ordinal,
            "physical_role": role,
            "service_role": spec["service_role"],
            "commit_sha": COMMIT,
            "pcr0": "a" * 96,
            "normalized_image_hash": "sha256:" + "2" * 64,
            "eif_hash": "sha256:" + "3" * 64,
            "source_manifest_hash": "sha256:" + "4" * 64,
            "build_identity_hash": "sha256:" + "5" * 64,
            "execution_manifest_hash": "sha256:" + "6" * 64,
            "dependency_lock_hash": "sha256:" + "7" * 64,
            "dockerfile_hash": "sha256:" + "8" * 64,
            "topology_hash": topology_hash(),
        }
        for role, spec in ROLE_SPECS.items()
        for ordinal in (1, 2, 3)
    ]


@pytest.fixture
def artifacts(tmp_path):
    for domain in ("gateway", "validator"):
        directory = tmp_path / domain
        directory.mkdir()
        (directory / f"gateway-{domain}-evidence.json").write_text(
            json.dumps(_rows(domain)), encoding="utf-8"
        )
    return tmp_path


def _verify(root):
    return EVIDENCE.main(
        [
            "--gateway-artifact", str(root / "gateway"),
            "--validator-artifact", str(root / "validator"),
            "--gateway-builder-id", "gateway-parent-123-1",
            "--validator-builder-id", "validator-parent-123-1",
            "--commit", COMMIT,
            "--acceptance-signer-pubkey-hash", "sha256:" + "f" * 64,
        ]
    )


def test_exact_artifacts_pass_full_manifest_consensus(artifacts):
    assert _verify(artifacts) == 0


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("commit_sha", "9" * 40),
        ("builder_domain", "gateway"),
        ("builder_id", "validator-parent-122-1"),
        ("pcr0", "b" * 96),
        ("build_ordinal", 1),
    ],
)
def test_forged_or_replayed_artifacts_fail_before_publication(artifacts, field, value):
    path = artifacts / "validator/gateway-validator-evidence.json"
    rows = _rows("validator")
    rows[-1][field] = value
    path.write_text(json.dumps(rows), encoding="utf-8")
    with pytest.raises(ReleaseManifestV2Error):
        _verify(artifacts)


def test_one_builder_cannot_supply_both_domain_artifacts(artifacts):
    path = artifacts / "validator/gateway-validator-evidence.json"
    path.write_text(json.dumps(_rows("gateway")), encoding="utf-8")
    with pytest.raises(ReleaseManifestV2Error, match="identity differs"):
        _verify(artifacts)


@pytest.mark.parametrize("entry", ["executable", "symlink", "directory", "duplicate-json"])
def test_artifacts_are_bounded_data_only(artifacts, entry):
    path = artifacts / "validator/gateway-validator-evidence.json"
    if entry == "executable":
        (path.parent / "payload.sh").write_text("exit 0\n", encoding="utf-8")
    elif entry == "symlink":
        path.unlink()
        path.symlink_to(artifacts / "gateway/gateway-gateway-evidence.json")
    elif entry == "directory":
        path.unlink()
        path.mkdir()
    else:
        path.write_text('[{"commit_sha":"a","commit_sha":"b"}]', encoding="utf-8")
    with pytest.raises(ReleaseManifestV2Error):
        _verify(artifacts)


def test_artifact_file_size_is_limited(artifacts, monkeypatch):
    monkeypatch.setattr(EVIDENCE, "MAX_EVIDENCE_BYTES", 1)
    with pytest.raises(ReleaseManifestV2Error, match="size limit"):
        _verify(artifacts)


@pytest.mark.parametrize(
    "override",
    [
        {"GITHUB_REPOSITORY": "fork/leadpoet"},
        {"GITHUB_EVENT_NAME": "pull_request_target"},
        {"GITHUB_REF": "refs/heads/untrusted"},
        {"GITHUB_SHA": "2" * 40},
        {"TEST_REMOTE_SHA": "2" * 40},
    ],
)
def test_release_source_gate_rejects_wrong_authority(tmp_path, override):
    assert _source_gate(tmp_path, override).returncode != 0


def _source_gate(tmp_path, override):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    git = fake_bin / "git"
    git.write_text(
        '#!/bin/bash\nset -eu\n'
        'if [ "$1" = fetch ]; then exit 0; fi\n'
        'if [ "$2" = HEAD ]; then printf "%s\\n" "$TEST_LOCAL_SHA"; '
        'else printf "%s\\n" "$TEST_REMOTE_SHA"; fi\n',
        encoding="utf-8",
    )
    git.chmod(0o700)
    env = {
        **os.environ,
        "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
        "GITHUB_REPOSITORY": "leadpoet/leadpoet",
        "GITHUB_EVENT_NAME": "workflow_dispatch",
        "GITHUB_REF": "refs/heads/main",
        "GITHUB_SHA": COMMIT,
        "TEST_LOCAL_SHA": COMMIT,
        "TEST_REMOTE_SHA": COMMIT,
        **override,
    }
    return subprocess.run(
        ["bash", str(ROOT / ".github/scripts/require_attested_release_source.sh")],
        env=env, capture_output=True, text=True, check=False,
    )


def test_release_source_gate_accepts_exact_main(tmp_path):
    assert _source_gate(tmp_path, {}).returncode == 0
