import os

import pytest

from scripts import run_arena_validator
from scripts.run_arena_validator import load_environment


def _env(tmp_path, text):
    path = tmp_path / "arena.env"
    path.write_text(text)
    path.chmod(0o600)
    return path


def test_environment_is_data_and_overrides_ambient_settings(tmp_path, monkeypatch):
    monkeypatch.setenv("LAB_ARENA_API_BASE_URL", "old")
    path = _env(tmp_path, 'LAB_ARENA_API_BASE_URL="https://arena.example/$(false)"\n')
    load_environment(path)
    assert os.environ["LAB_ARENA_API_BASE_URL"] == "https://arena.example/$(false)"


@pytest.mark.parametrize("body", [
    "LEADPOET_WEIGHT_MODE=legacy\n",
    "ENCLAVE_CID=8\n",
    "PATH=/tmp\n",
    "LAB_ARENA_API_BASE_URL=one\nLAB_ARENA_API_BASE_URL=two\n",
    "LAB_ARENA_SECRET=private value\n",
])
def test_invalid_configuration_is_rejected_without_values_in_errors(tmp_path, body):
    with pytest.raises(ValueError) as error:
        load_environment(_env(tmp_path, body))
    assert "private value" not in str(error.value)


def test_public_or_symlinked_environment_is_rejected(tmp_path):
    path = _env(tmp_path, "# Arena validator environment\n")
    path.chmod(0o644)
    with pytest.raises(ValueError, match="private regular"):
        load_environment(path)
    path.chmod(0o600)
    link = tmp_path / "link.env"
    link.symlink_to(path)
    with pytest.raises(OSError):
        load_environment(link)


def test_standard_wallet_flags_are_forwarded_unchanged(tmp_path, monkeypatch):
    path = _env(tmp_path, "LAB_ARENA_API_BASE_URL=https://arena.example\n")
    observed = {}

    def validator_main(argv):
        observed["argv"] = argv
        return 0

    import lab_arena.validator

    monkeypatch.setattr(lab_arena.validator, "main", validator_main)
    assert run_arena_validator.main(
        [
            "--environment-file", str(path),
            "--wallet-name", "arena_runner",
            "--hotkey", "default",
            "--wallet-path", "/var/lib/arena-wallets",
            "--check-only",
        ]
    ) == 0
    assert observed == {
        "argv": [
            "--wallet-name", "arena_runner",
            "--hotkey", "default",
            "--wallet-path", "/var/lib/arena-wallets",
            "--check-only",
        ]
    }


def test_check_only_is_forwarded_without_enclave_configuration(tmp_path, monkeypatch):
    path = _env(tmp_path, "# empty\n")
    observed = {}

    def validator_main(argv):
        observed["argv"] = argv
        observed["enclave"] = os.environ.get("ENCLAVE_CID")
        return 0

    import lab_arena.validator

    monkeypatch.delenv("ENCLAVE_CID", raising=False)
    monkeypatch.setattr(lab_arena.validator, "main", validator_main)
    assert run_arena_validator.main(
        ["--environment-file", str(path), "--check-only"]
    ) == 0
    assert observed == {"argv": ["--check-only"], "enclave": None}
