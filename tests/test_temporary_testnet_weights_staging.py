"""Bounded staging contract checks; these do not count as live chain proof."""

from datetime import datetime, timedelta, timezone
from io import BytesIO
import os
from pathlib import Path
import stat

import pytest

from scripts import stage_temporary_testnet_weights_host as stage


class _Body(BytesIO):
    def close(self):
        self.was_closed = True
        super().close()


def test_runner_pins_and_reads_only_unexpired_compliance_version():
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
                "VersionId": "exact-locked-version",
            }
            return {
                "Body": body,
                "ContentLength": 15,
                "ServerSideEncryption": "AES256",
                "VersionId": "exact-locked-version",
            }

    assert stage.read_locked_private_input(
        S3(), bucket="task-bucket", key="fixed/input", now=now
    ) == b"encrypted-input"
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
