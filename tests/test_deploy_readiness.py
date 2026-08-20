from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from gateway import deploy_readiness
from gateway.tee.release_channel_v2 import (
    build_release_channel_v2,
    build_release_lineage_v2,
)
from gateway.tee.topology import ROLE_SPECS
from leadpoet_canonical.attested_v2 import sha256_json
from tests.test_release_channel_v2 import _gateway_manifest, _validator_manifest


def _status(role: str, pcr0: str, *, allowed: bool, commits: list[str]) -> dict:
    return {
        "role": role,
        "pcr0": pcr0,
        "allowed": allowed,
        "allowed_count": 1 if allowed else 0,
        "allowed_source": "test",
        "allowed_error": None,
        "local_allowlist_path": "test",
        "local_match_count": 1 if commits else 0,
        "matched_entry_commits": commits,
        "matched_entries": [],
    }


def test_deploy_readiness_blocks_pcr0_commit_drift(monkeypatch) -> None:
    gateway_commit = "a" * 40
    stale_commit = "b" * 40
    gateway_pcr0 = "1" * 96
    validator_pcr0 = "2" * 96

    monkeypatch.setattr(
        deploy_readiness,
        "get_build_info",
        lambda: {
            "git_commit": gateway_commit,
            "build_time_utc": "2026-07-06T12:00:00Z",
        },
    )
    monkeypatch.setattr(deploy_readiness, "read_source_commit", lambda: (gateway_commit, "test-source"))
    monkeypatch.setattr(
        deploy_readiness,
        "_static_allowlist_status",
        lambda pcr0, *, role: _status(
            role,
            pcr0,
            allowed=True,
            commits=[stale_commit if role == "gateway" else gateway_commit],
        ),
    )
    monkeypatch.setattr(
        deploy_readiness,
        "_dynamic_validator_status",
        lambda pcr0, expected_commit=None: {
            "available": True,
            "valid": False,
            "verification": {},
            "cache_status": {},
        },
    )

    result = deploy_readiness.build_deploy_readiness(
        validator_commit=gateway_commit,
        gateway_pcr0=gateway_pcr0,
        validator_pcr0=validator_pcr0,
        require_same_commit=True,
        require_pcr0=True,
        require_pcr0_commit_match=True,
    )

    assert result["ok"] is False
    failed = {check["name"] for check in result["checks"] if not check["ok"]}
    assert failed == {"gateway_pcr0_commit_matches_gateway_commit"}
    assert result["validator"]["pcr0_accepted"] is True


def test_deploy_readiness_accepts_matching_commits(monkeypatch) -> None:
    commit = "a" * 40
    gateway_pcr0 = "1" * 96
    validator_pcr0 = "2" * 96

    monkeypatch.setattr(
        deploy_readiness,
        "get_build_info",
        lambda: {
            "git_commit": commit,
            "build_time_utc": "2026-07-06T12:00:00Z",
        },
    )
    monkeypatch.setattr(deploy_readiness, "read_source_commit", lambda: (commit, "test-source"))
    monkeypatch.setattr(
        deploy_readiness,
        "_static_allowlist_status",
        lambda pcr0, *, role: _status(role, pcr0, allowed=True, commits=[commit]),
    )
    monkeypatch.setattr(
        deploy_readiness,
        "_dynamic_validator_status",
        lambda pcr0, expected_commit=None: {
            "available": True,
            "valid": False,
            "verification": {},
            "cache_status": {},
        },
    )

    result = deploy_readiness.build_deploy_readiness(
        validator_commit=commit,
        gateway_pcr0=gateway_pcr0,
        validator_pcr0=validator_pcr0,
        expected_gateway_commit=commit[:12],
        expected_validator_commit=commit,
        require_same_commit=True,
        require_pcr0=True,
        require_pcr0_commit_match=True,
    )

    assert result["ok"] is True
    assert all(check["ok"] for check in result["checks"])


def test_deploy_readiness_accepts_exact_dynamic_validator_commit(monkeypatch) -> None:
    commit = "a" * 40
    stale_commit = "b" * 40
    validator_pcr0 = "2" * 96
    observed = []

    monkeypatch.setattr(
        deploy_readiness,
        "get_build_info",
        lambda: {"git_commit": commit, "build_time_utc": "2026-07-06T12:00:00Z"},
    )
    monkeypatch.setattr(
        deploy_readiness,
        "read_source_commit",
        lambda: (commit, "test-source"),
    )
    monkeypatch.setattr(
        deploy_readiness,
        "_static_allowlist_status",
        lambda pcr0, *, role: _status(
            role,
            pcr0,
            allowed=True,
            commits=[stale_commit],
        ),
    )

    def dynamic_status(pcr0, expected_commit=None):
        observed.append((pcr0, expected_commit))
        return {
            "available": True,
            "valid": True,
            "verification": {"commit_hash": expected_commit},
            "cache_status": {},
        }

    monkeypatch.setattr(
        deploy_readiness,
        "_dynamic_validator_status",
        dynamic_status,
    )

    result = deploy_readiness.build_deploy_readiness(
        validator_commit=commit,
        validator_pcr0=validator_pcr0,
        require_pcr0_commit_match=True,
    )

    assert result["ok"] is True
    assert observed == [(validator_pcr0, commit)]
    check = next(
        row
        for row in result["checks"]
        if row["name"] == "validator_pcr0_commit_matches_validator_commit"
    )
    assert check["actual"] == {
        "dynamic": commit,
        "static": [stale_commit],
    }


def test_resume_guard_blocks_failed_manifest(tmp_path: Path) -> None:
    manifest = tmp_path / "deploy_readiness.json"
    manifest.write_text(
        json.dumps(
            {
                "ok": False,
                "enforce_resume_block": True,
                "checks": [
                    {
                        "name": "validator_pcr0_accepted",
                        "ok": False,
                        "severity": "error",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="validator_pcr0_accepted"):
        deploy_readiness.assert_resume_allowed(manifest)


def test_resume_guard_rejects_historical_ok_manifest(tmp_path: Path) -> None:
    manifest = tmp_path / "deploy_readiness.json"
    manifest.write_text(
        json.dumps({"ok": True, "enforce_resume_block": True, "checks": []}),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="schema v2 is required"):
        deploy_readiness.assert_resume_allowed(manifest)


def _v2_release_authority(commit: str) -> tuple[dict, dict, dict]:
    gateway_release = _gateway_manifest(commit)
    validator_release = _validator_manifest(commit)
    channel = build_release_channel_v2(
        gateway_release_manifest=gateway_release,
        validator_release_manifest=validator_release,
    )
    return (
        gateway_release,
        validator_release,
        build_release_lineage_v2([channel], current_commit=commit),
    )


def _boot(role: str, expectation: dict, character: str) -> dict:
    return {
        "role": role,
        "physical_role": role,
        "commit_sha": expectation["commit_sha"],
        "pcr0": expectation["pcr0"],
        "build_manifest_hash": expectation["build_manifest_hash"],
        "dependency_lock_hash": expectation["dependency_lock_hash"],
        "boot_identity_hash": "sha256:" + character * 64,
        "config_hash": "sha256:" + character * 64,
    }


def _runtime_readiness(boots: dict[str, dict]) -> dict:
    return {
        "schema_version": "leadpoet.gateway_v2_runtime_readiness.v2",
        "status": "ready",
        "provider_registry_hash": "sha256:" + "9" * 64,
        "roles": [
            {
                "physical_role": role,
                "role": ROLE_SPECS[role]["service_role"],
                "worker_count": 1,
                "configured_worker_count": 1,
                "boot_identity_hash": boots[role]["boot_identity_hash"],
            }
            for role in sorted(ROLE_SPECS)
        ],
    }


def _v2_host_observations(commit: str) -> tuple[dict, dict]:
    gateway_release, validator_release, lineage = _v2_release_authority(commit)
    current_roles = lineage["releases"][commit]["roles"]
    gateway_boots = {
        role: _boot(role, current_roles[role], character)
        for role, character in zip(sorted(ROLE_SPECS), "567")
    }
    validator_boot = _boot(
        "validator_weights",
        current_roles["validator_weights"],
        "8",
    )
    gateway = {
        "schema_version": deploy_readiness.GATEWAY_READINESS_OBSERVATION_V2_SCHEMA_VERSION,
        "source_commit": commit,
        "build_commit": commit,
        "gateway_release_manifest": gateway_release,
        "validator_release_manifest": validator_release,
        "compact_lineage": lineage,
        "boot_identities": gateway_boots,
        "expected_role_config_hashes": {
            role: boot["config_hash"] for role, boot in gateway_boots.items()
        },
        "runtime_readiness": _runtime_readiness(gateway_boots),
        "coordinator_attestation_pcr0": gateway_boots[
            "gateway_coordinator"
        ]["pcr0"],
    }
    validator = {
        "schema_version": deploy_readiness.VALIDATOR_READINESS_OBSERVATION_V2_SCHEMA_VERSION,
        "host_commit": commit,
        "gateway_release_manifest": gateway_release,
        "validator_release_manifest": validator_release,
        "compact_lineage": lineage,
        "boot_identity": validator_boot,
        "expected_config_hash": validator_boot["config_hash"],
    }
    return gateway, validator


def _v2_host_evidence(commit: str) -> tuple[dict, dict]:
    gateway_observation, validator_observation = _v2_host_observations(commit)
    verified = []

    def verify(boot, **kwargs):
        verified.append((boot["physical_role"], kwargs))

    gateway = deploy_readiness.build_gateway_v2_readiness_evidence(
        expected_commit=commit,
        source_commit=gateway_observation["source_commit"],
        build_commit=gateway_observation["build_commit"],
        gateway_release_manifest=gateway_observation["gateway_release_manifest"],
        validator_release_manifest=gateway_observation["validator_release_manifest"],
        compact_lineage=gateway_observation["compact_lineage"],
        boot_identities=gateway_observation["boot_identities"],
        expected_role_config_hashes=gateway_observation[
            "expected_role_config_hashes"
        ],
        runtime_readiness=gateway_observation["runtime_readiness"],
        coordinator_attestation_pcr0=gateway_observation[
            "coordinator_attestation_pcr0"
        ],
        boot_verifier=verify,
    )
    validator = deploy_readiness.build_validator_v2_readiness_evidence(
        expected_commit=commit,
        host_commit=validator_observation["host_commit"],
        gateway_release_manifest=validator_observation[
            "gateway_release_manifest"
        ],
        validator_release_manifest=validator_observation[
            "validator_release_manifest"
        ],
        compact_lineage=validator_observation["compact_lineage"],
        boot_identity=validator_observation["boot_identity"],
        expected_config_hash=validator_observation["expected_config_hash"],
        boot_verifier=verify,
    )
    current_roles = gateway_observation["compact_lineage"]["releases"][commit][
        "roles"
    ]
    assert {role for role, _ in verified} == set(ROLE_SPECS) | {
        "validator_weights"
    }
    assert all(
        values == {"expected_pcr0": current_roles[role]["pcr0"],
                   "certificate_validity_at_attestation_time": True}
        for role, values in verified
    )
    return gateway, validator


def test_canonical_readiness_finalizer_runs_without_validator_wallet_dependency(
    tmp_path: Path,
) -> None:
    """The local restart controller needs release validation, not a live wallet."""

    commit = "1" * 40
    gateway_release, validator_release, lineage = _v2_release_authority(commit)
    gateway_evidence, validator_evidence = _v2_host_evidence(commit)
    input_path = tmp_path / "readiness-input.json"
    output_path = tmp_path / "deploy-readiness.json"
    input_path.write_text(
        json.dumps(
            {
                "commit": commit,
                "gateway_release": gateway_release,
                "validator_release": validator_release,
                "lineage": lineage,
                "gateway_evidence": gateway_evidence,
                "validator_evidence": validator_evidence,
            }
        ),
        encoding="utf-8",
    )
    repo_root = Path(__file__).resolve().parents[1]
    controller = r"""
import copy
import importlib.util
import json
from pathlib import Path
import sys

if importlib.util.find_spec("bittensor_wallet") is not None:
    raise RuntimeError("dependency-minimal controller unexpectedly has bittensor_wallet")

from gateway.deploy_readiness import (
    _validated_v2_release_authority,
    build_v2_deploy_readiness_manifest,
    validate_v2_deploy_readiness_manifest,
)

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
commit = payload["commit"]
channel, _lineage, _current = _validated_v2_release_authority(
    expected_commit=commit,
    gateway_release_manifest=payload["gateway_release"],
    validator_release_manifest=payload["validator_release"],
    compact_lineage=payload["lineage"],
)
if channel["commit_sha"] != commit:
    raise RuntimeError("validated release channel commit differs")

tampered_validator = copy.deepcopy(payload["validator_evidence"])
tampered_validator["channel_hash"] = "sha256:" + "f" * 64
try:
    build_v2_deploy_readiness_manifest(
        expected_commit=commit,
        gateway_evidence=payload["gateway_evidence"],
        validator_evidence=tampered_validator,
    )
except RuntimeError as exc:
    if "gateway and validator evidence differ at channel_hash" not in str(exc):
        raise
else:
    raise RuntimeError("cross-host channel drift was accepted")

manifest = build_v2_deploy_readiness_manifest(
    expected_commit=commit,
    gateway_evidence=payload["gateway_evidence"],
    validator_evidence=payload["validator_evidence"],
)
validated = validate_v2_deploy_readiness_manifest(
    manifest,
    runtime_source_commit=commit,
    runtime_build_commit=commit,
)
Path(sys.argv[2]).write_text(
    json.dumps(validated, sort_keys=True, separators=(",", ":")) + "\n",
    encoding="ascii",
)
print("ready")
"""

    completed = subprocess.run(
        [
            sys.executable,
            "-S",
            "-c",
            controller,
            str(input_path),
            str(output_path),
        ],
        cwd=repo_root,
        env={"PATH": os.defpath, "PYTHONPATH": str(repo_root)},
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout == "ready\n"
    assert completed.stderr == ""
    manifest = json.loads(output_path.read_text(encoding="ascii"))
    assert manifest["schema_version"] == deploy_readiness.DEPLOY_READINESS_V2_SCHEMA_VERSION
    assert manifest["expected_commit_sha"] == commit
    assert manifest["ok"] is True
    assert len(manifest["checks"]) == 14
    assert all(check["ok"] is True for check in manifest["checks"])


def test_isolated_dependency_complete_controller_builds_all_readiness_evidence(
    tmp_path: Path,
) -> None:
    commit = "1" * 40
    gateway_observation, validator_observation = _v2_host_observations(commit)
    input_path = tmp_path / "readiness-observations.json"
    output_path = tmp_path / "deploy-readiness.json"
    input_path.write_text(
        json.dumps(
            {
                "commit": commit,
                "gateway": gateway_observation,
                "validator": validator_observation,
            }
        ),
        encoding="utf-8",
    )
    repo_root = Path(__file__).resolve().parents[1]
    venv_root = Path(sys.executable).parent.parent
    assert (venv_root / "pyvenv.cfg").is_file()
    site_packages = (
        venv_root
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
    )
    assert site_packages.is_dir()
    bootstrap = r"""
from pathlib import Path
import sys

root = Path(sys.argv.pop(1)).resolve(strict=True)
site_packages = Path(sys.argv.pop(1)).resolve(strict=True)
sys.path.insert(0, str(root))
sys.path.append(str(site_packages))
source = sys.stdin.read()
exec(
    compile(source, "<test-local-readiness>", "exec"),
    {"__name__": "__main__", "__builtins__": __builtins__},
)
"""
    controller = r"""
import json
from pathlib import Path
import sys

if "site" in sys.modules or "sitecustomize" in sys.modules:
    raise RuntimeError("isolated controller activated site configuration")
if Path(sys.path[-1]).name != "site-packages":
    raise RuntimeError("isolated controller site-packages does not follow stdlib")
import argparse
try:
    Path(argparse.__file__).resolve(strict=True).relative_to(
        Path(sys.path[-1]).resolve(strict=True)
    )
except ValueError:
    pass
else:
    raise RuntimeError("site-packages shadowed the stdlib argparse module")

from leadpoet_canonical import attested_v2, nitro

available, diagnostic = nitro.verify_nitro_attestation_full(
    attestation_b64="!",
    expected_pcr0="1" * 96,
)
if available or "Missing required library" in str(diagnostic.get("error") or ""):
    raise RuntimeError("Nitro verifier dependencies are unavailable")

# Synthetic boot summaries exercise the exact observation join after the real
# dependency import/execution probe above. Production never replaces this verifier.
attested_v2.verify_boot_identity_nitro = lambda *args, **kwargs: {}

from gateway.deploy_readiness import (
    build_gateway_v2_readiness_evidence_from_observation,
    build_validator_v2_readiness_evidence_from_observation,
    build_v2_deploy_readiness_manifest,
    validate_v2_deploy_readiness_manifest,
)

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
commit = payload["commit"]
gateway = build_gateway_v2_readiness_evidence_from_observation(
    expected_commit=commit,
    observation=payload["gateway"],
)
validator = build_validator_v2_readiness_evidence_from_observation(
    expected_commit=commit,
    observation=payload["validator"],
)
manifest = build_v2_deploy_readiness_manifest(
    expected_commit=commit,
    gateway_evidence=gateway,
    validator_evidence=validator,
)
validate_v2_deploy_readiness_manifest(
    manifest,
    runtime_source_commit=commit,
    runtime_build_commit=commit,
)
if any(
    name == "bittensor_wallet" or name.startswith("bittensor_wallet.")
    for name in sys.modules
):
    raise RuntimeError("readiness controller loaded the validator wallet")
if "site" in sys.modules or "sitecustomize" in sys.modules:
    raise RuntimeError("readiness controller activated site configuration")
Path(sys.argv[2]).write_text(
    json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n",
    encoding="ascii",
)
print("full-readiness-ready")
"""

    completed = subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            "-B",
            "-c",
            bootstrap,
            str(repo_root),
            str(site_packages),
            str(input_path),
            str(output_path),
        ],
        cwd=repo_root,
        env={"PATH": os.defpath},
        input=controller,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout == "full-readiness-ready\n"
    assert completed.stderr == ""
    manifest = json.loads(output_path.read_text(encoding="ascii"))
    assert manifest["schema_version"] == deploy_readiness.DEPLOY_READINESS_V2_SCHEMA_VERSION
    assert manifest["expected_commit_sha"] == commit
    assert manifest["ok"] is True
    assert len(manifest["checks"]) == 14
    assert all(check["ok"] is True for check in manifest["checks"])


def test_validator_tee_package_preserves_authoritative_runtime_exports() -> None:
    import validator_tee
    from validator_tee.host import enclave_hotkey_v2, weight_authority_v2

    assert set(validator_tee.__all__) == {
        "AuthoritativeSetWeightsContextV2",
        "AuthoritativeServeAxonContextV2",
        "EnclaveBackedKeypairV2",
        "EnclaveBackedWalletV2",
        "build_enclave_backed_wallet_v2",
        "build_authoritative_weight_bundle_v2",
    }
    for name in validator_tee.__all__:
        source = (
            weight_authority_v2
            if name == "build_authoritative_weight_bundle_v2"
            else enclave_hotkey_v2
        )
        assert getattr(validator_tee, name) is getattr(source, name)
    assert set(validator_tee.__all__).issubset(dir(validator_tee))
    with pytest.raises(AttributeError, match="has no attribute"):
        getattr(validator_tee, "not_an_authoritative_export")


def _write_v2_manifest(path: Path, commit: str) -> dict:
    gateway, validator = _v2_host_evidence(commit)
    manifest = deploy_readiness.build_v2_deploy_readiness_manifest(
        expected_commit=commit,
        gateway_evidence=gateway,
        validator_evidence=validator,
    )
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest


def test_resume_guard_allows_exact_fresh_v2_manifest(
    monkeypatch, tmp_path: Path
) -> None:
    commit = "1" * 40
    manifest_path = tmp_path / "deploy_readiness.json"
    _write_v2_manifest(manifest_path, commit)
    monkeypatch.setattr(
        deploy_readiness, "read_source_commit", lambda: (commit, "test-source")
    )
    monkeypatch.setattr(
        deploy_readiness,
        "get_build_info",
        lambda: {"git_commit": commit},
    )

    result = deploy_readiness.assert_resume_allowed(manifest_path)

    assert result["schema_version"] == deploy_readiness.DEPLOY_READINESS_V2_SCHEMA_VERSION
    assert result["expected_commit_sha"] == commit
    assert result["ok"] is True


def test_resume_guard_rejects_missing_manifest(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="manifest is missing"):
        deploy_readiness.assert_resume_allowed(tmp_path / "missing.json")


def test_resume_guard_rejects_coherent_stale_v2_manifest(
    monkeypatch, tmp_path: Path
) -> None:
    stale_commit = "1" * 40
    runtime_commit = "2" * 40
    manifest_path = tmp_path / "deploy_readiness.json"
    _write_v2_manifest(manifest_path, stale_commit)
    monkeypatch.setattr(
        deploy_readiness,
        "read_source_commit",
        lambda: (runtime_commit, "test-source"),
    )
    monkeypatch.setattr(
        deploy_readiness,
        "get_build_info",
        lambda: {"git_commit": runtime_commit},
    )

    with pytest.raises(RuntimeError, match="stale for current gateway source"):
        deploy_readiness.assert_resume_allowed(manifest_path)


@pytest.mark.parametrize(
    "mutation", ["enforcement", "failed_check", "missing_check"]
)
def test_resume_guard_rejects_semantically_invalid_v2_manifest(
    monkeypatch, tmp_path: Path, mutation: str
) -> None:
    commit = "1" * 40
    manifest_path = tmp_path / "deploy_readiness.json"
    manifest = _write_v2_manifest(manifest_path, commit)
    if mutation == "enforcement":
        manifest["enforce_resume_block"] = False
    elif mutation == "failed_check":
        manifest["checks"][0]["ok"] = False
    else:
        manifest["checks"].pop()
    body = {key: value for key, value in manifest.items() if key != "manifest_hash"}
    manifest["manifest_hash"] = deploy_readiness._canonical_hash(body)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(
        deploy_readiness, "read_source_commit", lambda: (commit, "test-source")
    )
    monkeypatch.setattr(
        deploy_readiness, "get_build_info", lambda: {"git_commit": commit}
    )

    with pytest.raises(RuntimeError):
        deploy_readiness.assert_resume_allowed(manifest_path)


def test_transition_marker_blocks_previous_and_current_resume_guards(
    tmp_path: Path,
) -> None:
    commit = "1" * 40
    marker = deploy_readiness.build_deploy_readiness_transition_marker(
        expected_commit=commit
    )
    manifest_path = tmp_path / "deploy_readiness.json"
    manifest_path.write_text(json.dumps(marker), encoding="utf-8")

    def previous_guard(value: dict) -> None:
        if value.get("enforce_resume_block", True) and value.get("ok") is not True:
            raise RuntimeError("blocked")

    with pytest.raises(RuntimeError, match="blocked"):
        previous_guard(marker)
    with pytest.raises(RuntimeError, match="canonical_restart_in_progress"):
        deploy_readiness.assert_resume_allowed(manifest_path)


def test_gateway_evidence_rejects_self_consistent_lineage_not_bound_to_channel() -> None:
    commit = "1" * 40
    gateway_release, validator_release, lineage = _v2_release_authority(commit)
    tampered = copy.deepcopy(lineage)
    tampered["releases"][commit]["channel_hash"] = "sha256:" + "9" * 64
    body = {key: value for key, value in tampered.items() if key != "lineage_hash"}
    tampered["lineage_hash"] = sha256_json(body)
    roles = lineage["releases"][commit]["roles"]
    gateway_boots = {
        role: _boot(role, roles[role], character)
        for role, character in zip(sorted(ROLE_SPECS), "567")
    }

    with pytest.raises(RuntimeError, match="differs from release channel"):
        deploy_readiness.build_gateway_v2_readiness_evidence(
            expected_commit=commit,
            source_commit=commit,
            build_commit=commit,
            gateway_release_manifest=gateway_release,
            validator_release_manifest=validator_release,
            compact_lineage=tampered,
            boot_identities=gateway_boots,
            expected_role_config_hashes={
                role: boot["config_hash"]
                for role, boot in gateway_boots.items()
            },
            runtime_readiness=_runtime_readiness(gateway_boots),
            coordinator_attestation_pcr0=gateway_boots[
                "gateway_coordinator"
            ]["pcr0"],
            boot_verifier=lambda *args, **kwargs: None,
        )


def test_gateway_evidence_rejects_runtime_health_boot_identity_drift() -> None:
    commit = "1" * 40
    gateway_release, validator_release, lineage = _v2_release_authority(commit)
    roles = lineage["releases"][commit]["roles"]
    gateway_boots = {
        role: _boot(role, roles[role], character)
        for role, character in zip(sorted(ROLE_SPECS), "567")
    }
    health = _runtime_readiness(gateway_boots)
    next(
        row
        for row in health["roles"]
        if row["physical_role"] == "gateway_scoring"
    )["boot_identity_hash"] = "sha256:" + "9" * 64

    with pytest.raises(RuntimeError, match="runtime health differs"):
        deploy_readiness.build_gateway_v2_readiness_evidence(
            expected_commit=commit,
            source_commit=commit,
            build_commit=commit,
            gateway_release_manifest=gateway_release,
            validator_release_manifest=validator_release,
            compact_lineage=lineage,
            boot_identities=gateway_boots,
            expected_role_config_hashes={
                role: boot["config_hash"]
                for role, boot in gateway_boots.items()
            },
            runtime_readiness=health,
            coordinator_attestation_pcr0=gateway_boots[
                "gateway_coordinator"
            ]["pcr0"],
            boot_verifier=lambda *args, **kwargs: None,
        )


def test_gateway_evidence_rejects_runtime_document_config_drift() -> None:
    commit = "1" * 40
    gateway_release, validator_release, lineage = _v2_release_authority(commit)
    roles = lineage["releases"][commit]["roles"]
    gateway_boots = {
        role: _boot(role, roles[role], character)
        for role, character in zip(sorted(ROLE_SPECS), "567")
    }
    expected_configs = {
        role: boot["config_hash"] for role, boot in gateway_boots.items()
    }
    expected_configs["gateway_scoring"] = "sha256:" + "9" * 64

    with pytest.raises(RuntimeError, match="config differs from runtime document"):
        deploy_readiness.build_gateway_v2_readiness_evidence(
            expected_commit=commit,
            source_commit=commit,
            build_commit=commit,
            gateway_release_manifest=gateway_release,
            validator_release_manifest=validator_release,
            compact_lineage=lineage,
            boot_identities=gateway_boots,
            expected_role_config_hashes=expected_configs,
            runtime_readiness=_runtime_readiness(gateway_boots),
            coordinator_attestation_pcr0=gateway_boots[
                "gateway_coordinator"
            ]["pcr0"],
            boot_verifier=lambda *args, **kwargs: None,
        )


def test_gateway_evidence_rejects_unsuccessful_runtime_readiness() -> None:
    commit = "1" * 40
    gateway_release, validator_release, lineage = _v2_release_authority(commit)
    roles = lineage["releases"][commit]["roles"]
    gateway_boots = {
        role: _boot(role, roles[role], character)
        for role, character in zip(sorted(ROLE_SPECS), "567")
    }
    runtime_readiness = _runtime_readiness(gateway_boots)
    runtime_readiness["status"] = "not_ready"

    with pytest.raises(RuntimeError, match="runtime readiness is not successful"):
        deploy_readiness.build_gateway_v2_readiness_evidence(
            expected_commit=commit,
            source_commit=commit,
            build_commit=commit,
            gateway_release_manifest=gateway_release,
            validator_release_manifest=validator_release,
            compact_lineage=lineage,
            boot_identities=gateway_boots,
            expected_role_config_hashes={
                role: boot["config_hash"]
                for role, boot in gateway_boots.items()
            },
            runtime_readiness=runtime_readiness,
            coordinator_attestation_pcr0=gateway_boots[
                "gateway_coordinator"
            ]["pcr0"],
            boot_verifier=lambda *args, **kwargs: None,
        )


def test_optional_docker_health_is_warning_only(monkeypatch) -> None:
    commit = "a" * 40

    monkeypatch.setattr(
        deploy_readiness,
        "get_build_info",
        lambda: {"git_commit": commit, "build_time_utc": "2026-07-06T12:00:00Z"},
    )
    monkeypatch.setattr(deploy_readiness, "read_source_commit", lambda: (commit, "test-source"))
    monkeypatch.setattr(
        deploy_readiness,
        "_static_allowlist_status",
        lambda pcr0, *, role: _status(role, pcr0 or "", allowed=False, commits=[]),
    )
    monkeypatch.setattr(
        deploy_readiness,
        "_dynamic_validator_status",
        lambda pcr0, expected_commit=None: {
            "available": True,
            "valid": False,
            "verification": {},
            "cache_status": {},
        },
    )
    monkeypatch.setattr(
        deploy_readiness,
        "docker_build_health",
        lambda *, smoke_build=False: {
            "ok": False,
            "docker_info": {"docker_root": "/var/lib/docker"},
            "disk": {"ok": False, "free_gb": 0.1},
            "smoke_build_requested": smoke_build,
            "smoke_build": None,
        },
    )

    result = deploy_readiness.build_deploy_readiness(include_docker_health=True)

    assert result["ok"] is True
    docker_checks = [check for check in result["checks"] if check["name"] == "docker_build_health"]
    assert docker_checks == [
        {
            "name": "docker_build_health",
            "ok": False,
            "severity": "warning",
            "detail": (
                "Docker host/build health; require flag runs a tiny scratch-image smoke build "
                "and blocks resume on failure"
            ),
            "expected": None,
            "actual": {
                "docker_root": "/var/lib/docker",
                "disk": {"ok": False, "free_gb": 0.1},
                "smoke_build_requested": False,
                "smoke_build_ok": None,
            },
        }
    ]


def test_required_docker_build_health_blocks_readiness(monkeypatch) -> None:
    commit = "a" * 40

    monkeypatch.setattr(
        deploy_readiness,
        "get_build_info",
        lambda: {"git_commit": commit, "build_time_utc": "2026-07-06T12:00:00Z"},
    )
    monkeypatch.setattr(deploy_readiness, "read_source_commit", lambda: (commit, "test-source"))
    monkeypatch.setattr(
        deploy_readiness,
        "_static_allowlist_status",
        lambda pcr0, *, role: _status(role, pcr0 or "", allowed=False, commits=[]),
    )
    monkeypatch.setattr(
        deploy_readiness,
        "_dynamic_validator_status",
        lambda pcr0, expected_commit=None: {
            "available": True,
            "valid": False,
            "verification": {},
            "cache_status": {},
        },
    )
    monkeypatch.setattr(
        deploy_readiness,
        "docker_build_health",
        lambda *, smoke_build=False: {
            "ok": False,
            "docker_info": {"docker_root": "/var/lib/docker"},
            "disk": {"ok": True, "free_gb": 92.0},
            "smoke_build_requested": smoke_build,
            "smoke_build": {"ok": False},
        },
    )

    result = deploy_readiness.build_deploy_readiness(require_docker_build_health=True)

    assert result["ok"] is False
    failed = [check for check in result["checks"] if check["name"] == "docker_build_health"]
    assert len(failed) == 1
    assert failed[0]["severity"] == "error"
    assert failed[0]["actual"]["smoke_build_requested"] is True
