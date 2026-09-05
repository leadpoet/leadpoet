import json
import os
import subprocess
from pathlib import Path

import pytest

from gateway.tee.release_channel_v2 import (
    build_release_channel_v2,
    build_release_lineage_v2,
    fetch_release_channel_v2,
    fetch_release_lineage_v2,
)
from gateway.tee.release_manifest_v2 import (
    build_local_release_identity,
    validate_release_manifest,
)
from gateway.tee.topology import ROLE_SPECS, topology_hash
from leadpoet_canonical.auditor_v2 import fetch_locked_release_identity_cache
from scripts import gateway_git_deploy
from validator_tee.host.release_v2 import (
    build_local_validator_release_identity,
    build_validator_release,
    validate_validator_release_manifest,
)


COMMIT = "a" * 40
HISTORICAL_CHANNEL_RELEASE = "f90b5eb3739eb3871a0d7bde0a3a1c41c62016ea"


def _hash(character: str) -> str:
    return "sha256:" + character * 64


def _gateway_results(commit: str = COMMIT) -> list[dict]:
    results = []
    for index, role in enumerate(sorted(ROLE_SPECS), start=1):
        character = format(index, "x")
        results.append(
            {
                "role": role,
                "commit_sha": commit,
                "pcr0": character * 96,
                "image_id": _hash(character),
                "source_manifest_hash": _hash(character),
                "build_identity_hash": _hash(character),
                "execution_manifest_hash": _hash(character),
                "dependency_lock_hash": _hash(character),
                "dockerfile_hash": _hash(character),
                "topology_hash": topology_hash(),
            }
        )
    return results


def _channel(commit: str = COMMIT) -> dict:
    gateway = build_local_release_identity(_gateway_results(commit))
    release = build_validator_release(
        commit_sha=commit,
        pcr0="e" * 96,
        app_manifest_hash=_hash("1"),
        dependency_lock_hash=_hash("2"),
        normalized_image_hash=_hash("3"),
        eif_hash=_hash("4"),
        dockerfile_hash=_hash("5"),
        base_dockerfile_hash=_hash("6"),
    )
    validator = build_local_validator_release_identity(release)
    return build_release_channel_v2(
        gateway_release_manifest=gateway,
        validator_release_manifest=validator,
    )


def _channel_with_eif_hash(eif_hash: str) -> dict:
    gateway = build_local_release_identity(_gateway_results())
    release = build_validator_release(
        commit_sha=COMMIT,
        pcr0="e" * 96,
        app_manifest_hash=_hash("1"),
        dependency_lock_hash=_hash("2"),
        normalized_image_hash=_hash("3"),
        eif_hash=eif_hash,
        dockerfile_hash=_hash("5"),
        base_dockerfile_hash=_hash("6"),
    )
    return build_release_channel_v2(
        gateway_release_manifest=gateway,
        validator_release_manifest=build_local_validator_release_identity(release),
    )


def test_local_release_identities_validate_without_external_evidence() -> None:
    channel = _channel()
    gateway = validate_release_manifest(channel["gateway_release_manifest"])
    validator = validate_validator_release_manifest(
        channel["validator_release_manifest"]
    )

    assert gateway["commit_sha"] == COMMIT
    assert gateway["verified_build_count"] == len(ROLE_SPECS)
    assert {row["verified_build_count"] for row in gateway["roles"].values()} == {
        1
    }
    assert validator["verified_build_count"] == 1


def test_local_release_channel_precedes_s3(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    channel = _channel()
    gateway_path = tmp_path / "gateway.json"
    validator_path = tmp_path / "validator.json"
    gateway_path.write_text(
        json.dumps(channel["gateway_release_manifest"]), encoding="utf-8"
    )
    validator_path.write_text(
        json.dumps(channel["validator_release_manifest"]), encoding="utf-8"
    )
    monkeypatch.setenv("LEADPOET_LOCAL_RELEASE_COMMIT_SHA", COMMIT)
    monkeypatch.setenv("LEADPOET_LOCAL_GATEWAY_RELEASE", str(gateway_path))
    monkeypatch.setenv("LEADPOET_LOCAL_VALIDATOR_RELEASE", str(validator_path))

    class NoS3:
        def get_object(self, **_kwargs):
            raise AssertionError("S3 must not be called for the local commit")

    assert fetch_release_channel_v2(
        bucket="unused", commit_sha=COMMIT, s3_client=NoS3()
    ) == channel


def test_installed_lineage_removes_s3_from_next_restart(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    next_commit = "b" * 40
    prior = build_release_lineage_v2(
        [_channel()],
        current_commit=COMMIT,
    )
    current = _channel(next_commit)
    prior_path = tmp_path / "prior-lineage.json"
    gateway_path = tmp_path / "gateway.json"
    validator_path = tmp_path / "validator.json"
    prior_path.write_text(json.dumps(prior), encoding="utf-8")
    gateway_path.write_text(
        json.dumps(current["gateway_release_manifest"]), encoding="utf-8"
    )
    validator_path.write_text(
        json.dumps(current["validator_release_manifest"]), encoding="utf-8"
    )
    monkeypatch.setenv("LEADPOET_LOCAL_RELEASE_COMMIT_SHA", next_commit)
    monkeypatch.setenv("LEADPOET_LOCAL_GATEWAY_RELEASE", str(gateway_path))
    monkeypatch.setenv("LEADPOET_LOCAL_VALIDATOR_RELEASE", str(validator_path))
    monkeypatch.setenv(
        "LEADPOET_LOCAL_PRIOR_RELEASE_LINEAGE", str(prior_path)
    )

    class NoS3:
        def get_object(self, **_kwargs):
            raise AssertionError("S3 must not be called for installed lineage")

    lineage = fetch_release_lineage_v2(
        bucket="unused",
        current_commit=next_commit,
        allowed_commits=[COMMIT, next_commit],
        required_commits=[COMMIT, next_commit],
        s3_client=NoS3(),
    )

    assert lineage["current_commit_sha"] == next_commit
    assert set(lineage["releases"]) == {COMMIT, next_commit}


def test_local_channel_identity_ignores_raw_validator_eif_bytes() -> None:
    first = _channel_with_eif_hash(_hash("4"))
    second = _channel_with_eif_hash(_hash("5"))

    assert first["validator_release_manifest"] != second[
        "validator_release_manifest"
    ]
    assert first["channel_hash"] == second["channel_hash"]


def test_auditor_accepts_inline_local_release_identity() -> None:
    cache = fetch_locked_release_identity_cache(
        {
            "schema_version": "leadpoet.auditor_local_release_evidence.v1",
            "commit_sha": COMMIT,
            "release_channel": _channel(),
        }
    )
    assert len(cache["entries"]) == len(ROLE_SPECS) + 1
    assert {entry["verified_build_count"] for entry in cache["entries"]} == {1}


def test_restart_scripts_use_attested_channel_only_for_explicit_historical_release() -> None:
    root = Path(__file__).resolve().parents[1]
    gateway = (root / "gw_restart.sh").read_text(encoding="utf-8")
    validator = (root / "validator_restart.sh").read_text(encoding="utf-8")
    operator = (root / "scripts/restart_attested_release_local.sh").read_text(
        encoding="utf-8"
    )

    assert "Approved V2 release is not published yet" not in gateway
    assert "Approved V2 release is not published yet" not in validator
    assert gateway.count("--ensure") == 1
    assert validator.count("--ensure") == 1
    assert "build_local_release_v2.sh" in gateway
    assert "build_local_release_v2.sh" in validator
    assert '[ -n "$REQUESTED_GATEWAY_DEPLOY_COMMIT" ]' in gateway
    assert '[ -n "$REQUESTED_VALIDATOR_DEPLOY_COMMIT" ]' in validator
    assert (
        "selected release has an incomplete or unsupported V2 release "
        "acquisition contract" in gateway
    )
    assert (
        "selected release has an incomplete or unsupported V2 release "
        "acquisition contract" in validator
    )
    assert "fetch_release_channel_v2" in operator
    assert "historical_topology_hash is not None" in operator
    assert "active gateway release differs from immutable channel" in operator


def test_real_historical_release_has_only_the_channel_capability() -> None:
    root = Path(__file__).resolve().parents[1]

    def exists(path: str) -> bool:
        result = subprocess.run(
            ["git", "cat-file", "-e", f"{HISTORICAL_CHANNEL_RELEASE}:{path}"],
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
        )
        return result.returncode == 0

    assert not exists("gateway/tee/build_local_release_v2.sh")
    assert not exists("gateway/tee/local_release_v2.py")
    assert exists("gateway/tee/release_channel_v2.py")
    topology_entry = subprocess.run(
        [
            "git",
            "ls-tree",
            HISTORICAL_CHANNEL_RELEASE,
            "--",
            "gateway/tee/topology.json",
        ],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert topology_entry == (
        "100644 blob f79cf108e4a98ca950a0087d786958f92c5f691f"
        "\tgateway/tee/topology.json"
    )
    historical_profiles = subprocess.run(
        [
            "git",
            "show",
            f"{HISTORICAL_CHANNEL_RELEASE}:gateway/research_lab/provider_profiles_v2.py",
        ],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    historical_bootstrap = subprocess.run(
        [
            "git",
            "show",
            f"{HISTORICAL_CHANNEL_RELEASE}:gateway/utils/tee_v2_bootstrap.py",
        ],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    assert "def verify_required_worker_proxy_profiles_v2(" in historical_profiles
    assert "def configured_scoring_worker_count(" not in historical_bootstrap
    assert (root / "gateway/tee/build_local_release_v2.sh").is_file()
    assert (root / "gateway/tee/local_release_v2.py").is_file()


def test_local_release_builder_runs_modules_from_candidate_root() -> None:
    root = Path(__file__).resolve().parents[1]
    script = (root / "gateway/tee/build_local_release_v2.sh").read_text(
        encoding="utf-8"
    )

    frozen_archive = script.index('git -C "$REPOSITORY" archive "$REVISION"')
    frozen_reexec = script.index(
        '/bin/bash "$FROZEN_SOURCE_ROOT/gateway/tee/build_local_release_v2.sh"'
    )
    candidate_root = script.index('cd "$CANDIDATE_ROOT"')
    gateway_builder = script.index(
        "python3 -m validator_tee.host.gateway_pcr0_builder"
    )
    release_builder = script.index("python3 -m gateway.tee.local_release_v2")
    assert frozen_archive < frozen_reexec < candidate_root
    assert candidate_root < gateway_builder < release_builder


@pytest.mark.parametrize("assembler_status", [0, 65])
def test_local_release_builder_uses_and_cleans_frozen_candidate_tree(
    tmp_path: Path, assembler_status: int,
) -> None:
    root = Path(__file__).resolve().parents[1]
    candidate = tmp_path / "candidate"
    builder = candidate / "gateway/tee/build_local_release_v2.sh"
    builder.parent.mkdir(parents=True)
    builder.write_bytes(
        (root / "gateway/tee/build_local_release_v2.sh").read_bytes()
    )
    builder.chmod(0o755)
    lock = candidate / "validator_tee/scripts/docker_operation_lock_v2.sh"
    lock.parent.mkdir(parents=True)
    lock.write_text(
        "leadpoet_acquire_docker_operation_lock_v2() { :; }\n"
        "leadpoet_release_docker_operation_lock_v2() { :; }\n",
        encoding="utf-8",
    )
    assembler = candidate / "gateway/tee/local_release_v2.py"
    assembler.write_text("# exact revision assembler\n", encoding="utf-8")

    subprocess.run(["git", "init", "-q", str(candidate)], check=True)
    subprocess.run(
        ["git", "-C", str(candidate), "config", "user.email", "test@example.com"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(candidate), "config", "user.name", "Test"],
        check=True,
    )
    subprocess.run(["git", "-C", str(candidate), "add", "."], check=True)
    subprocess.run(
        ["git", "-C", str(candidate), "commit", "-qm", "candidate"],
        check=True,
    )
    revision = subprocess.run(
        ["git", "-C", str(candidate), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_bash = fake_bin / "bash"
    fake_bash.write_text(
        "#!/bin/sh\n"
        "set -eu\n"
        "candidate_root=${1%/validator_tee/scripts/build_enclave.sh}\n"
        "mkdir -p \"$candidate_root/.validator-tee-artifacts\"\n"
        "printf 'stamp\\n' > \"$candidate_root/.validator-base.dockerfile.sha256\"\n"
        "printf 'artifact\\n' > \"$candidate_root/.validator-tee-artifacts/runtime.bin\"\n"
        "printf 'eif\\n' > \"$candidate_root/validator_tee/validator-enclave.eif\"\n"
        "printf 'measurements\\n' > \"$candidate_root/validator_tee/enclave_build_output.txt\"\n"
        "printf '{}\\n' > \"$candidate_root/validator_tee/validator-v2-release.json\"\n",
        encoding="utf-8",
    )
    fake_bash.chmod(0o755)
    fake_python = fake_bin / "python3"
    fake_python.write_text(
        "#!/bin/sh\n"
        "set -eu\n"
        "printf '%s\\n' \"$PYTHONPATH\" >> \"$LOCAL_RELEASE_TEST_TRACE\"\n"
        "test \"${1:-}\" = -m\n"
        "module=${2:-}\n"
        "shift 2\n"
        "output_file= gateway_output= validator_output= validator_release=\n"
        "while [ \"$#\" -gt 0 ]; do\n"
        "  case \"$1\" in\n"
        "    --output-file) output_file=$2; shift 2 ;;\n"
        "    --gateway-output) gateway_output=$2; shift 2 ;;\n"
        "    --validator-output) validator_output=$2; shift 2 ;;\n"
        "    --validator-release) validator_release=$2; shift 2 ;;\n"
        "    *) shift ;;\n"
        "  esac\n"
        "done\n"
        "case \"$module\" in\n"
        "  validator_tee.host.gateway_pcr0_builder)\n"
        "    rm \"$LOCAL_RELEASE_MUTABLE_ROOT/gateway/tee/local_release_v2.py\"\n"
        "    printf '[]\\n' > \"$output_file\"\n"
        "    ;;\n"
        "  gateway.tee.local_release_v2)\n"
        "    test -r \"$PYTHONPATH/gateway/tee/local_release_v2.py\"\n"
        "    test \"$LOCAL_RELEASE_ASSEMBLER_STATUS\" = 0 || exit \"$LOCAL_RELEASE_ASSEMBLER_STATUS\"\n"
        "    test -s \"$validator_release\"\n"
        "    printf '{}\\n' > \"$gateway_output\"\n"
        "    printf '{}\\n' > \"$validator_output\"\n"
        "    ;;\n"
        "  *) exit 64 ;;\n"
        "esac\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)

    gateway_output = tmp_path / "gateway-release.json"
    validator_output = tmp_path / "validator-release.json"
    environment = os.environ.copy()
    environment["GATEWAY_V2_BUILD_WORK_ROOT"] = str(tmp_path / "build-work")
    trace = tmp_path / "python-paths.txt"
    environment["LOCAL_RELEASE_TEST_TRACE"] = str(trace)
    environment["LOCAL_RELEASE_MUTABLE_ROOT"] = str(candidate)
    environment["LOCAL_RELEASE_ASSEMBLER_STATUS"] = str(assembler_status)
    environment["PATH"] = f"{fake_bin}{os.pathsep}{environment['PATH']}"
    result = subprocess.run(
        [
            "/bin/bash",
            str(builder),
            "--repository",
            str(candidate),
            "--revision",
            revision,
            "--gateway-output",
            str(gateway_output),
            "--validator-output",
            str(validator_output),
        ],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert result.returncode == assembler_status, result.stderr
    assert gateway_output.is_file() is (assembler_status == 0)
    assert validator_output.is_file() is (assembler_status == 0)
    assert not assembler.exists()
    python_paths = trace.read_text(encoding="utf-8").splitlines()
    assert len(python_paths) == 2
    assert python_paths[0] == python_paths[1]
    assert python_paths[0] != str(candidate)
    assert python_paths[0].startswith("/tmp/leadpoet-local-release-source.")
    assert not Path(python_paths[0]).exists()
    for relative_path in (
        ".validator-base.dockerfile.sha256",
        ".validator-tee-artifacts",
        "validator_tee/validator-enclave.eif",
        "validator_tee/enclave_build_output.txt",
        "validator_tee/validator-v2-release.json",
    ):
        assert not (candidate / relative_path).exists()

    assembler.write_text("# exact revision assembler\n", encoding="utf-8")

    evidence = gateway_git_deploy.verify_materialized_tree(
        repo_root=candidate,
        materialized_root=candidate,
        target_sha=revision,
        strict_extras=True,
    )
    assert evidence["strict_extras"] is True
