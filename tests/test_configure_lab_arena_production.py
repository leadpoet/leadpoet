import argparse
import importlib.util
import json
import os
import shlex
import subprocess
import sys
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
        daily_cutoff_utc=0,
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
    assert updates["LAB_ARENA_DAILY_CUTOFF_UTC"] == "0"


def test_daily_cutoff_accepts_hour_six_and_rejects_out_of_range(tmp_path):
    parsed = MODULE.build_parser().parse_args(["--allowed-account", "493765492819", "--daily-cutoff-utc", "6"])
    assert parsed.daily_cutoff_utc == 6
    parsed.prepare_runner = True
    parsed.apply = False
    parsed.ssh_key = tmp_path / "key"
    parsed.ssh_key.write_text("x")
    MODULE._validate_args(parsed)
    configured = args()
    configured.daily_cutoff_utc = 6
    assert MODULE.gateway_nonsecret_updates(configured)["LAB_ARENA_DAILY_CUTOFF_UTC"] == "6"
    parsed.daily_cutoff_utc = 24
    with pytest.raises(MODULE.ConfigurationError, match="between 0 and 23"):
        MODULE._validate_args(parsed)


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
    assert '"--version-stages", "AWSCURRENT"' in source
    assert 'if updated == raw:' in source
    assert 'fail("secret_document_conflicting_duplicate")' in source
    assert 'comments=True, posix=True' in source


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


def _run_remote_with_fake_aws(tmp_path, raw, *, expect_success=True, return_state=False, request_override=None):
    state = tmp_path / "state.json"
    state.write_text(json.dumps({"current": "initial", "versions": {"initial": raw}}))
    aws = tmp_path / "aws"
    aws.write_text("""#!/usr/bin/env python3
import json, os, sys
path = os.environ["TEST_AWS_STATE"]
state = json.load(open(path))
args = sys.argv[1:]
if args[:2] == ["sts", "get-caller-identity"]:
    print(json.dumps({"Account": "493765492819", "Arn": "redacted"}))
elif args[:2] == ["secretsmanager", "get-secret-value"]:
    version = args[args.index("--version-id") + 1] if "--version-id" in args else state["current"]
    print(json.dumps({"VersionId": version, "SecretString": state["versions"][version]}))
elif args[:2] == ["secretsmanager", "put-secret-value"]:
    version = args[args.index("--client-request-token") + 1]
    state["versions"][version] = sys.stdin.read()
    if args[args.index("--version-stages") + 1] == "AWSCURRENT":
        state["current"] = version
    json.dump(state, open(path, "w"))
    print(json.dumps({"VersionId": version}))
else:
    raise SystemExit(2)
""")
    aws.chmod(0o755)
    request = {
        "secret_id": MODULE.GATEWAY_SECRET,
        "allowed_accounts": ["493765492819"],
        "apply": True,
        "role": "gateway",
        "updates": {"LAB_ARENA_MODE": "live"},
        "service_key": "sb_secret_scoped-test",
        "aliases": {
            "OPENROUTER_API_KEY": "LAB_ARENA_OPENROUTER_API_KEY",
            "SCRAPINGDOG_API_KEY": "LAB_ARENA_SCRAPINGDOG_API_KEY",
            "DEEPLINE_API_KEY": "LAB_ARENA_DEEPLINE_API_KEY",
            "SUPABASE_URL": "LAB_ARENA_SUPABASE_URL",
            "SUPABASE_ANON_KEY": "LAB_ARENA_SUPABASE_ANON_KEY",
        },
    }
    if request_override:
        request.update(request_override)
    env = dict(os.environ, TEST_AWS_STATE=str(state), PATH=str(tmp_path) + os.pathsep + os.environ["PATH"])
    result = subprocess.run([sys.executable, "-c", MODULE._REMOTE], input=json.dumps(request), text=True, capture_output=True, env=env)
    if not expect_success:
        assert result.returncode != 0
        return json.loads(result.stdout)
    assert result.returncode == 0, result.stdout
    final = json.loads(state.read_text())
    if return_state:
        return json.loads(result.stdout), final
    return final["versions"][final["current"]]


def test_remote_fake_aws_preserves_space_values_and_identical_unrelated_duplicates(tmp_path):
    raw = "SSH_CLIENT=10.0.0.1 123 22\nREPEAT=same\nREPEAT=same\nOPENROUTER_API_KEY=or\nSCRAPINGDOG_API_KEY=sd\nDEEPLINE_API_KEY=deep\nSUPABASE_URL=https://db.example\nSUPABASE_ANON_KEY=anon\n"
    updated = _run_remote_with_fake_aws(tmp_path, raw)
    assert "SSH_CLIENT=10.0.0.1 123 22\n" in updated
    assert updated.count("REPEAT=same\n") == 2
    assert "export LAB_ARENA_OPENROUTER_API_KEY=or\n" in updated


def test_remote_fake_aws_preserves_json_alias_source_keys(tmp_path):
    source = {"OPENROUTER_API_KEY": "or", "SCRAPINGDOG_API_KEY": "sd", "DEEPLINE_API_KEY": "deep", "SUPABASE_URL": "https://db.example", "SUPABASE_ANON_KEY": "anon", "KEEP": "same"}
    updated = json.loads(_run_remote_with_fake_aws(tmp_path, json.dumps(source, separators=(",", ":"))))
    assert updated["OPENROUTER_API_KEY"] == "or"
    assert updated["LAB_ARENA_OPENROUTER_API_KEY"] == "or"
    assert updated["KEEP"] == "same"


def test_remote_fake_aws_rejects_duplicate_target_key(tmp_path):
    raw = "LAB_ARENA_MODE=off\nLAB_ARENA_MODE=off\nOPENROUTER_API_KEY=or\nSCRAPINGDOG_API_KEY=sd\nDEEPLINE_API_KEY=deep\nSUPABASE_URL=https://db.example\nSUPABASE_ANON_KEY=anon\n"
    result = _run_remote_with_fake_aws(tmp_path, raw, expect_success=False)
    assert result == {"ok": False, "code": "target_key_duplicate"}


def test_remote_fake_aws_noops_when_current_document_already_matches(tmp_path):
    source = {
        "OPENROUTER_API_KEY": "or", "SCRAPINGDOG_API_KEY": "sd",
        "DEEPLINE_API_KEY": "deep", "SUPABASE_URL": "https://db.example",
        "SUPABASE_ANON_KEY": "anon", "LAB_ARENA_MODE": "live",
        "LAB_ARENA_OPENROUTER_API_KEY": "or", "LAB_ARENA_SCRAPINGDOG_API_KEY": "sd",
        "LAB_ARENA_DEEPLINE_API_KEY": "deep", "LAB_ARENA_SUPABASE_URL": "https://db.example",
        "LAB_ARENA_SUPABASE_ANON_KEY": "anon", "LAB_ARENA_SERVICE_KEY": "sb_secret_scoped-test",
    }
    raw = json.dumps(source, separators=(",", ":"))
    result, state = _run_remote_with_fake_aws(tmp_path, raw, return_state=True)
    assert result["unchanged"] is True
    assert state == {"current": "initial", "versions": {"initial": raw}}


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


def test_miner_credentials_scope_changes_only_kms_alias(tmp_path):
    source = {"RESEARCH_LAB_OPENROUTER_KEY_KMS_KEY_ID": "alias/existing-key",
              "LAB_ARENA_MODE": "live", "LAB_ARENA_DAILY_CUTOFF_UTC": "6",
              "LAB_ARENA_SERVICE_KEY": "sb_secret_unchanged"}
    updated = json.loads(_run_remote_with_fake_aws(
        tmp_path, json.dumps(source), request_override={
            "role": "miner_credentials", "updates": {}, "service_key": "",
            "aliases": {"RESEARCH_LAB_OPENROUTER_KEY_KMS_KEY_ID": "LAB_ARENA_CREDENTIAL_KMS_KEY_ID"},
        }))
    assert updated == dict(source, LAB_ARENA_CREDENTIAL_KMS_KEY_ID="alias/existing-key")


def test_miner_credentials_scope_does_not_require_service_key_or_touch_validator(monkeypatch, tmp_path, capsys):
    key = tmp_path / "ssh.pem"
    key.write_text("test")
    calls = []
    monkeypatch.setattr(MODULE, "_read_fd", lambda fd: pytest.fail("must not read service key"))
    monkeypatch.setattr(MODULE, "_ssh", lambda host, key, request: calls.append((host, request)) or {"ok": True})
    assert MODULE.main(["--miner-credentials-only", "--check", "--allowed-account", "493765492819", "--ssh-key", str(key)]) == 0
    assert len(calls) == 1 and calls[0][0] == MODULE.GATEWAY_HOST
    assert calls[0][1]["updates"] == {} and calls[0][1]["apply"] is False
    assert list(calls[0][1]["aliases"].values()) == ["LAB_ARENA_CREDENTIAL_KMS_KEY_ID"]
    assert json.loads(capsys.readouterr().out)["ok"] is True


def test_miner_credentials_can_be_disabled_without_changing_other_configuration(monkeypatch, tmp_path):
    key = tmp_path / "ssh.pem"
    key.write_text("test")
    calls = []
    monkeypatch.setattr(MODULE, "_ssh", lambda host, key, request: calls.append(request) or {"ok": True})
    assert MODULE.main(["--miner-credentials-only", "--miner-credential-kms-key-id", "", "--check", "--allowed-account", "493765492819", "--ssh-key", str(key)]) == 0
    assert calls[0]["updates"] == {"LAB_ARENA_CREDENTIAL_KMS_KEY_ID": ""}
    assert calls[0]["aliases"] == {}
