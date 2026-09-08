"""Bounded staging contract checks; these do not count as live chain proof."""

import os
from pathlib import Path
import stat

import pytest

from scripts import stage_temporary_testnet_weights_host as stage


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
