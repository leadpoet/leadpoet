import argparse
import importlib.util
import json
import shlex
from pathlib import Path

import pytest


PATH = Path(__file__).resolve().parents[1] / "scripts" / "configure_lab_arena_production.py"
SPEC = importlib.util.spec_from_file_location("configure_lab_arena_production", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def args():
    return argparse.Namespace(
        bucket="arena-bucket",
        scorer_image="registry.example/team/scorer:production",
        runner_hotkey="runner-key",
        baseline_hotkey="baseline-key",
        chain_endpoint="wss://chain.example",
        api_base_url="https://gateway.example",
    )


def existing():
    return {
        "OPENROUTER_API_KEY": "openrouter-secret",
        "SCRAPINGDOG_API_KEY": "scrapingdog-secret",
        "DEEPLINE_API_KEY": "deepline-secret",
        "SUPABASE_URL": "https://db.example",
        "SUPABASE_ANON_KEY": "anon-secret",
    }


def service_key():
    return "sb_secret_scoped-test-value"


def test_gateway_configuration_uses_existing_aliases_and_disables_rewards():
    updates = MODULE.gateway_updates(existing(), args(), service_key())
    assert updates["LAB_ARENA_MODE"] == "live"
    assert updates["LAB_ARENA_REWARDS_ENABLED"] == "false"
    assert updates["LAB_ARENA_OPENROUTER_API_KEY"] == "openrouter-secret"
    assert updates["LAB_ARENA_SUPABASE_URL"] == "https://db.example"


def test_gateway_nonsecret_configuration_derives_registry_repository():
    updates = MODULE.gateway_nonsecret_updates(args())
    assert updates["LAB_ARENA_REGISTRY_REPOSITORY"] == "registry.example/team/scorer"


def test_validator_uses_dedicated_host_only_runner_wallet():
    updates = MODULE.validator_updates(args())
    assert updates["LAB_ARENA_WALLET_NAME"] == "arena_runner"
    assert updates["LAB_ARENA_HOTKEY_NAME"] == "default"
    assert updates["LAB_ARENA_WALLET_PATH"] == "/var/lib/lab-arena/runner-wallets"


@pytest.mark.parametrize("missing", [
    "OPENROUTER_API_KEY", "SCRAPINGDOG_API_KEY", "DEEPLINE_API_KEY",
    "SUPABASE_URL", "SUPABASE_ANON_KEY",
])
def test_gateway_configuration_rejects_missing_existing_values(missing):
    values = existing()
    values[missing] = ""
    with pytest.raises(MODULE.ConfigurationError, match="required existing value"):
        MODULE.gateway_updates(values, args(), service_key())


def test_gateway_configuration_rejects_malformed_service_key():
    with pytest.raises(MODULE.ConfigurationError, match="service key is malformed"):
        MODULE.gateway_updates(existing(), args(), "not-a-scoped-key")


def test_document_merge_preserves_unrelated_json_and_dotenv_content():
    encoded, kind = MODULE.update_document('{"KEEP":"same","LAB_ARENA_MODE":"off"}', {"LAB_ARENA_MODE": "live"})
    assert kind == "json"
    assert json.loads(encoded) == {"KEEP": "same", "LAB_ARENA_MODE": "live"}

    encoded, kind = MODULE.update_document("# keep me\nexport KEEP='same value'\nLAB_ARENA_MODE=off\n", {"LAB_ARENA_MODE": "live"})
    assert kind == "dotenv"
    assert "# keep me" in encoded
    assert "export KEEP='same value'" in encoded
    assert "export LAB_ARENA_MODE=live" in encoded


def test_ssh_keeps_secret_payload_out_of_argv_and_output(monkeypatch, tmp_path, capsys):
    key = service_key()
    request = {"service_key": key, "updated": "SECRET=value"}
    seen = {}

    def run(command, **kwargs):
        seen.update(command=command, input=kwargs["input"])
        return argparse.Namespace(returncode=0, stdout='{"ok":true,"account":"187445349696"}', stderr="")

    monkeypatch.setattr(MODULE.subprocess, "run", run)
    result = MODULE._ssh("host", tmp_path / "key", request)
    assert result["ok"] is True
    assert key not in " ".join(seen["command"])
    assert "SECRET=value" not in " ".join(seen["command"])
    assert key in seen["input"]
    assert key not in capsys.readouterr().out
    remote_command = seen["command"][-1]
    assert shlex.split(remote_command) == ["python3", "-c", MODULE._REMOTE]


def test_remote_protocol_checks_version_again_before_stage_move():
    source = MODULE._REMOTE
    assert "current_id != before_id or current_raw != raw" in source
    assert 'fail("version_race")' in source
    assert '"--remove-from-version-id", before_id' in source
    assert 'fail("secret_document_duplicate_key")' in source


def test_runner_wallet_dry_run_does_not_generate_and_apply_is_shell_quoted(monkeypatch, tmp_path):
    seen = []

    def run(command, **kwargs):
        seen.append(command)
        return argparse.Namespace(returncode=0, stdout='{"ok":true,"exists":false,"created":false}', stderr="")

    monkeypatch.setattr(MODULE.subprocess, "run", run)
    result = MODULE._prepare_runner_wallet("host", tmp_path / "key", apply=False)
    assert result == {"ok": True, "exists": False, "created": False}
    parsed = shlex.split(seen[0][-1])
    assert parsed == ["sudo", "/home/ec2-user/venv311/bin/python3", "-c", MODULE._RUNNER_WALLET_REMOTE, "0"]
    assert 'if not exists and not bool(int(sys.argv[1]))' in MODULE._RUNNER_WALLET_REMOTE
    assert "generate_mnemonic" in MODULE._RUNNER_WALLET_REMOTE
    assert '"ss58_address": address' in MODULE._RUNNER_WALLET_REMOTE
    assert "redirect_stdout(sink)" in MODULE._RUNNER_WALLET_REMOTE


def test_prepare_runner_is_standalone_and_does_not_read_or_write_config(monkeypatch, tmp_path, capsys):
    key = tmp_path / "ssh.pem"
    key.write_text("test")
    monkeypatch.setenv(MODULE.AUTH_ENV, "1")
    monkeypatch.setattr(MODULE, "_check_remote_account", lambda *a, **k: {"ok": True, "account": "493765492819"})
    monkeypatch.setattr(MODULE, "_prepare_runner_wallet", lambda *a, **k: {"ok": True, "created": True, "ss58_address": "public"})
    monkeypatch.setattr(MODULE, "_read_fd", lambda fd: pytest.fail("must not read service key"))
    monkeypatch.setattr(MODULE, "_ssh", lambda *a, **k: pytest.fail("must not access env secrets"))
    result = MODULE.main([
        "--prepare-runner", "--apply", "--allowed-account", "493765492819",
        "--ssh-key", str(key),
    ])
    assert result == 0
    output = json.loads(capsys.readouterr().out)
    assert output["runner_wallet"]["ss58_address"] == "public"


def test_service_uses_configured_registry_client_for_scorer():
    source = (Path(__file__).resolve().parents[1] / "lab_arena" / "wiring.py").read_text()
    scorer_block = source[source.index("# The trusted scorer"):source.index("defaults = RoundDefaults")]
    assert "registry = registry_client_from_environment()" in scorer_block
    assert "registry = images.RegistryClient()" not in scorer_block
