"""Bounded staging contract checks; these do not count as live chain proof."""

from datetime import datetime, timedelta, timezone
from io import BytesIO
import json
import os
from pathlib import Path
import stat
import subprocess

import pytest

from scripts import stage_temporary_testnet_weights_host as stage


def test_prior_public_pair_validates_and_public_builder_accepts_next_release(tmp_path):
    from gateway.tee.release_channel_v2 import (
        build_release_channel_v2,
        build_release_lineage_v2,
    )
    from tests.test_release_channel_v2 import _gateway_manifest, _validator_manifest

    prior = "a" * 40
    current = "b" * 40
    prior_channel = build_release_channel_v2(
        gateway_release_manifest=_gateway_manifest(prior),
        validator_release_manifest=_validator_manifest(prior),
    )
    prior_lineage = build_release_lineage_v2([prior_channel], current_commit=prior)
    channel_path = tmp_path / "prior-release-channel-v2.json"
    lineage_path = tmp_path / "prior-release-lineage-v1.json"
    channel_path.write_text(json.dumps(prior_channel))
    lineage_path.write_text(json.dumps(prior_lineage))

    loaded, loaded_lineage = stage.load_prior_release_documents(
        channel_path=channel_path,
        lineage_path=lineage_path,
        expected_commit=prior,
    )
    current_channel = build_release_channel_v2(
        gateway_release_manifest=_gateway_manifest(current),
        validator_release_manifest=_validator_manifest(current),
    )
    merged = build_release_lineage_v2(
        [loaded, current_channel], current_commit=current
    )

    assert loaded_lineage == prior_lineage
    assert set(merged["releases"]) == {prior, current}
    assert merged["current_commit_sha"] == current


def test_gateway_env_dump_roundtrips_through_native_load_and_scrub(tmp_path):
    from gateway.tee import prepare_gateway_envelopes_v2 as envelopes
    from scripts.materialize_production_parity_secrets import _parse_environment_document

    raw = ('GIT_SSH_COMMAND=ssh -i /task/key -o IdentitiesOnly=yes\n'
           'SSH_CLIENT=192.0.2.1 12000 22\n'
           'QUOTED="two words"\nOPENROUTER_API_KEY=secret-canary\n')
    path = tmp_path / "gateway.env"
    stage.private_write(path, stage.normalized_gateway_environment(raw))
    expected = _parse_environment_document(raw, field="test")
    assert envelopes.load_environment_file(path) == expected
    report = tmp_path / "transition.json"
    report.write_text(json.dumps({
        "plaintext_environment_names_to_remove": ["OPENROUTER_API_KEY"],
        "plaintext_credential_ref_hashes_to_remove": [
            envelopes.credential_reference_hash("secret-canary")],
        "required_count_environment": {"RESEARCH_LAB_SCORING_WORKER_PROCESS_COUNT": "1"},
        "scoring_worker_count": 1,
    }))
    envelopes.scrub_parent_environment_file_v2(
        environment_path=path, transition_report_path=report)
    expected.pop("OPENROUTER_API_KEY")
    expected["RESEARCH_LAB_SCORING_WORKER_PROCESS_COUNT"] = "1"
    assert envelopes.load_environment_file(path) == expected
    assert "secret-canary" not in path.read_text()
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_gateway_env_normalization_rejects_multiline_before_native_consumers():
    with pytest.raises(ValueError, match="multiline"):
        stage.normalized_gateway_environment(json.dumps({"KEY": "one\ntwo"}))


class _Body(BytesIO):
    def close(self):
        self.was_closed = True
        super().close()


@pytest.mark.parametrize("response_version", ["exact-locked-version", "replaced-version"])
def test_runner_pins_and_reads_only_unexpired_compliance_version(response_version):
    now = datetime(2026, 9, 8, tzinfo=timezone.utc)
    body = _Body(b"encrypted-input")

    class S3:
        def head_object(self, **kwargs):
            assert kwargs == {"Bucket": "task-bucket", "Key": "fixed/input"}
            return {
                "ContentLength": 15,
                "ServerSideEncryption": "AES256",
                "ObjectLockMode": "COMPLIANCE",
                "ObjectLockRetainUntilDate": now + timedelta(days=1),
                "VersionId": "exact-locked-version",
            }

        def get_object(self, **kwargs):
            assert kwargs == {
                "Bucket": "task-bucket",
                "Key": "fixed/input",
            }
            return {
                "Body": body,
                "ContentLength": 15,
                "ServerSideEncryption": "AES256",
                "VersionId": response_version,
            }

    if response_version == "exact-locked-version":
        assert stage.read_locked_private_input(
            S3(), bucket="task-bucket", key="fixed/input", now=now
        ) == b"encrypted-input"
    else:
        with pytest.raises(ValueError, match="version differs"):
            stage.read_locked_private_input(
                S3(), bucket="task-bucket", key="fixed/input", now=now
            )
    assert body.was_closed is True


@pytest.mark.parametrize(
    "override",
    (
        {"ObjectLockMode": None},
        {"ObjectLockRetainUntilDate": datetime(2026, 9, 7, tzinfo=timezone.utc)},
        {"VersionId": ""},
        {"ServerSideEncryption": "aws:kms"},
    ),
)
def test_runner_rejects_unlocked_or_unpinned_private_input_before_read(override):
    now = datetime(2026, 9, 8, tzinfo=timezone.utc)

    class S3:
        get_called = False

        def head_object(self, **_kwargs):
            return {
                "ContentLength": 15,
                "ServerSideEncryption": "AES256",
                "ObjectLockMode": "COMPLIANCE",
                "ObjectLockRetainUntilDate": now + timedelta(days=1),
                "VersionId": "exact-locked-version",
                **override,
            }

        def get_object(self, **_kwargs):
            self.get_called = True
            raise AssertionError("untrusted object must not be read")

    s3 = S3()
    with pytest.raises(ValueError, match="retention metadata"):
        stage.read_locked_private_input(
            s3, bucket="task-bucket", key="fixed/input", now=now
        )
    assert s3.get_called is False


def test_config_matches_native_shape_without_private_keyfile():
    value = stage.build_config(
        repository=Path("/opt/leadpoet-testnet/repo"), candidate="a" * 40,
        run_id="pp-12345-1", instance_id="i-0123456789abcdef0", expiry=1788840000,
    )
    assert value["runtime_root"] == "/run/leadpoet-testnet401"
    assert value["validator"]["enclave_cid"] == 18
    assert value["validator"]["chain_profile"].endswith("chain_signing_profile_test_v2.json")
    assert value["validator"]["expected_hotkey"] == stage.native.EXPECTED_VALIDATOR_HOTKEY
    assert value["gateway"]["eif_root"].startswith(value["runtime_root"] + "/")
    assert "seed" not in value["validator"]
    assert "private_key" not in value["validator"]


def test_private_output_is_exclusive_and_owner_only(tmp_path):
    destination = tmp_path / "ciphertext.json"
    previous = os.umask(0)
    try:
        stage.private_write(destination, b"encrypted")
    finally:
        os.umask(previous)
    assert stat.S_IMODE(destination.stat().st_mode) == 0o600
    with pytest.raises(FileExistsError):
        stage.private_write(destination, b"replacement")
    assert destination.read_bytes() == b"encrypted"


def test_private_output_does_not_follow_symlinks(tmp_path):
    target = tmp_path / "target"
    target.write_text("preserve")
    link = tmp_path / "link"
    link.symlink_to(target)
    with pytest.raises(FileExistsError):
        stage.private_write(link, b"replacement")
    assert target.read_text() == "preserve"


def test_failure_receipt_reports_only_bounded_identity():
    def verify_host_is_empty():
        raise FileNotFoundError(2, "secret-bearing message", "/private/nitro-cli")

    try:
        verify_host_is_empty()
    except FileNotFoundError as exc:
        result = stage.failure_receipt(exc)

    assert result["status"] == "failed"
    assert result["error_type"] == "FileNotFoundError"
    assert result["operation"] == "verify_host_is_empty"
    assert result["location"].startswith(
        "tests/test_temporary_testnet_weights_staging.py:"
    )
    assert "code" not in result
    encoded = json.dumps(result, sort_keys=True)
    assert "secret-bearing" not in encoded
    assert "/private/nitro-cli" not in encoded


def test_nitro_cli_environment_uses_bound_artifacts_and_rpm_blobs(tmp_path):
    assert stage.NITRO_CLI_ARTIFACTS == Path(
        "/run/leadpoet-testnet401/nitro-cli-artifacts"
    )
    assert stage.NITRO_CLI_BLOBS == Path("/usr/share/nitro_enclaves/blobs")
    blobs = tmp_path / "rpm-blobs"
    blobs.mkdir()
    for name in stage.NITRO_CLI_BLOB_NAMES:
        (blobs / name).write_bytes(b"rpm-owned")
    artifacts = tmp_path / "task" / "nitro-cli-artifacts"
    artifacts.parent.mkdir()

    result = stage.prepare_nitro_cli_environment(
        artifacts=artifacts,
        blobs=blobs,
    )

    assert result == {
        "NITRO_CLI_ARTIFACTS": str(artifacts),
        "NITRO_CLI_BLOBS": str(blobs),
    }
    assert stat.S_IMODE(artifacts.stat().st_mode) == 0o700
    assert "HOME" not in result
    with pytest.raises(FileExistsError):
        stage.prepare_nitro_cli_environment(artifacts=artifacts, blobs=blobs)


def test_nitro_cli_environment_rejects_non_rpm_blob_shape(tmp_path):
    blobs = tmp_path / "rpm-blobs"
    blobs.mkdir()
    (blobs / "cmdline").write_bytes(b"incomplete")

    with pytest.raises(stage.native.TemporaryTestnetBootstrapError):
        stage.prepare_nitro_cli_environment(
            artifacts=tmp_path / "nitro-cli-artifacts",
            blobs=blobs,
        )


def test_drand_cache_assignment_runs_in_nonlogin_environment():
    builder = Path(stage.__file__).resolve().parents[1] / "validator_tee/scripts/build_drand_cabi_v2.sh"
    assignment = next(line for line in builder.read_text().splitlines() if line.startswith("CACHE_DIR="))
    assert stage.NATIVE_BUILD_CACHE_ENV == {
        "VALIDATOR_DRAND_CARGO_CACHE_DIR": "/run/leadpoet-testnet401/drand-cargo-cache",
    }
    command = ["bash", "-c", 'set -eu\n' + assignment + '\nprintf "%s" "$CACHE_DIR"']
    absent = subprocess.run(command, env={"PATH": os.environ["PATH"]}, capture_output=True, text=True)
    assert absent.returncode != 0
    assert "unbound variable" in absent.stderr
    result = subprocess.run(command, env={"PATH": os.environ["PATH"], **stage.NATIVE_BUILD_CACHE_ENV},
                            capture_output=True, text=True, check=True)
    assert result.stdout == "/run/leadpoet-testnet401/drand-cargo-cache"
    assert "HOME" not in stage.NATIVE_BUILD_CACHE_ENV


def test_gateway_source_assignment_runs_in_nonlogin_environment():
    repository = Path(stage.__file__).resolve().parents[1]
    builder = repository / "gateway/tee/stage_attested_runtime.sh"
    assignment = next(line for line in builder.read_text().splitlines() if line.startswith("DEPLOY_SOURCE_ROOT="))
    environment = stage.source_build_environment(repository)
    result = subprocess.run(["bash", "-c", 'set -eu\n' + assignment + '\nprintf "%s" "$DEPLOY_SOURCE_ROOT"'],
                            env={"PATH": os.environ["PATH"], **environment}, capture_output=True, text=True, check=True)
    assert result.stdout == str(repository)
    assert environment["ATTESTED_RUNTIME_GIT_SOURCE_ROOT"] == "/run/leadpoet-testnet401/gateway-stage-source"
    assert "HOME" not in environment


@pytest.mark.parametrize("field,value", [
    ("candidate", "main"),
    ("run_id", "../../another-run"),
    ("instance_id", "all"),
    ("assets_bucket", "leadpoet-attested-v2-artifacts-493765492819"),
    ("assets_prefix", "production-parity/runs/pp-OTHER/testnet401"),
])
def test_wrong_scope_rejected_before_host_or_aws_access(field, value):
    arguments = dict(candidate="a" * 40, run_id="pp-12345-1",
                     instance_id="i-0123456789abcdef0",
                     assets_bucket="leadpoet-parity-493765492819-pp-12345-1",
                     assets_prefix="production-parity/runs/pp-12345-1/testnet401")
    arguments[field] = value
    with pytest.raises(ValueError):
        stage.stage(**arguments)
