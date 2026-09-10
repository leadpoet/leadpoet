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


def test_explicit_candidate_enclave_cid_overrides_environment(tmp_path, monkeypatch):
    path = _env(tmp_path, "ENCLAVE_CID=8\n")
    observed = {}

    def validator_main(argv):
        observed["cid"] = os.environ["ENCLAVE_CID"]
        observed["argv"] = argv
        return 0

    import lab_arena.validator

    monkeypatch.setattr(lab_arena.validator, "main", validator_main)
    assert run_arena_validator.main(
        ["--environment-file", str(path), "--enclave-cid", "19", "--check-only"]
    ) == 0
    assert observed == {"cid": "19", "argv": ["--check-only"]}


def test_parent_cid_is_rejected_before_validator_import(tmp_path):
    with pytest.raises(SystemExit):
        run_arena_validator.main(
            ["--environment-file", str(_env(tmp_path, "# empty\n")), "--enclave-cid", "3"]
        )
