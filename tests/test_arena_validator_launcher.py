import os

import pytest

from scripts.run_arena_validator import load_environment


def _env(tmp_path, text):
    path = tmp_path / "arena.env"
    path.write_text(text)
    path.chmod(0o600)
    return path


def test_environment_is_data_and_overrides_ambient_settings(tmp_path, monkeypatch):
    monkeypatch.setenv("LEADPOET_WEIGHT_MODE", "legacy")
    monkeypatch.setenv("LAB_ARENA_API_BASE_URL", "old")
    path = _env(tmp_path, 'LEADPOET_WEIGHT_MODE=arena\nLAB_ARENA_API_BASE_URL="https://arena.example/$(false)"\n')
    load_environment(path)
    assert os.environ["LEADPOET_WEIGHT_MODE"] == "arena"
    assert os.environ["LAB_ARENA_API_BASE_URL"] == "https://arena.example/$(false)"


@pytest.mark.parametrize("body", [
    "LEADPOET_WEIGHT_MODE=legacy\n",
    "LEADPOET_WEIGHT_MODE=arena\nPATH=/tmp\n",
    "LEADPOET_WEIGHT_MODE=arena\nLEADPOET_WEIGHT_MODE=arena\n",
    "LEADPOET_WEIGHT_MODE=arena\nLAB_ARENA_SECRET=private value\n",
])
def test_invalid_configuration_is_rejected_without_values_in_errors(tmp_path, body):
    with pytest.raises(ValueError) as error:
        load_environment(_env(tmp_path, body))
    assert "private value" not in str(error.value)


def test_public_or_symlinked_environment_is_rejected(tmp_path):
    path = _env(tmp_path, "LEADPOET_WEIGHT_MODE=arena\n")
    path.chmod(0o644)
    with pytest.raises(ValueError, match="private regular"):
        load_environment(path)
    path.chmod(0o600)
    link = tmp_path / "link.env"
    link.symlink_to(path)
    with pytest.raises(OSError):
        load_environment(link)
