import os
import shlex

import pytest

from scripts import run_arena_validator
from scripts.run_arena_validator import load_environment, write_environment_snapshot


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


def test_snapshot_imports_only_canonical_proxy_entries(tmp_path, capsys):
    proxy_env = tmp_path / "validator.env"
    first_proxy = "https://user:first-secret@8.8.8.8:443"
    second_proxy = "http://user:second-secret@1.1.1.1:8080"
    proxy_env.write_text(
        "\n".join(
            (
                f"QUALIFICATION_WEBSHARE_PROXY_2={shlex.quote(first_proxy)}",
                f"QUALIFICATION_WEBSHARE_PROXY_9={shlex.quote(second_proxy)}",
                "OPENROUTER_API_KEY=unrelated-provider-secret",
                "BT_WALLET_COLDKEY=unrelated-wallet-secret",
                "",
            )
        )
    )
    proxy_env.chmod(0o600)
    source = _env(
        tmp_path,
        "LAB_ARENA_API_BASE_URL=https://arena.example\n"
        f"LAB_ARENA_PROXY_ENV_FILE={shlex.quote(str(proxy_env))}\n",
    )
    original_source = source.read_bytes()
    original_mode = source.stat().st_mode & 0o777
    process_environment = dict(os.environ)
    destination = tmp_path / "service.env"

    write_environment_snapshot(source, destination)

    generated = destination.read_text()
    assert "LAB_ARENA_PROXY_ENV_FILE" not in generated
    assert "QUALIFICATION_WEBSHARE_PROXY" not in generated
    assert "OPENROUTER_API_KEY" not in generated
    assert "unrelated-provider-secret" not in generated
    assert "BT_WALLET_COLDKEY" not in generated
    assert "unrelated-wallet-secret" not in generated
    assert f"LAB_ARENA_WEBSHARE_PROXY_1={shlex.quote(first_proxy)}\n" in generated
    assert f"LAB_ARENA_WEBSHARE_PROXY_2={shlex.quote(second_proxy)}\n" in generated
    assert destination.stat().st_mode & 0o777 == 0o600
    assert source.read_bytes() == original_source
    assert source.stat().st_mode & 0o777 == original_mode == 0o600
    assert dict(os.environ) == process_environment
    assert capsys.readouterr() == ("", "")


def test_snapshot_without_secondary_proxy_file_is_exact_copy(tmp_path):
    content = "# preserved operator environment\nLAB_ARENA_API_BASE_URL='https://arena.example'\n"
    source = _env(tmp_path, content)
    destination = tmp_path / "service.env"

    write_environment_snapshot(source, destination)

    assert destination.read_text() == content
    assert destination.stat().st_mode & 0o777 == 0o600


def test_snapshot_rejects_nonprivate_secondary_without_creating_output(
    tmp_path, capsys
):
    proxy_env = tmp_path / "validator.env"
    proxy_env.write_text(
        "QUALIFICATION_WEBSHARE_PROXY_1=https://user:private-value@proxy.example:443\n"
    )
    proxy_env.chmod(0o644)
    source = _env(
        tmp_path,
        f"LAB_ARENA_PROXY_ENV_FILE={shlex.quote(str(proxy_env))}\n",
    )
    destination = tmp_path / "service.env"

    with pytest.raises(ValueError, match="private owned regular file") as error:
        write_environment_snapshot(source, destination)

    assert "private-value" not in str(error.value)
    assert not destination.exists()
    assert capsys.readouterr() == ("", "")


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
