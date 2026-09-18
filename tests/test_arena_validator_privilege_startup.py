"""Standard validator startup must retain identity and independent weights."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from lab_arena import validator_startup as startup
from lab_arena.validator import _parser


@pytest.fixture
def host(monkeypatch, tmp_path):
    monkeypatch.setattr(startup.platform, "system", lambda: "Linux")
    monkeypatch.setattr(startup.os, "geteuid", lambda: 1000)
    monkeypatch.setattr(startup.os.path, "isfile", lambda path: path == "/usr/bin/sudo")
    monkeypatch.setattr(startup.os, "access", lambda *_: True)
    monkeypatch.setattr(startup.os, "environ", {"HOME": str(tmp_path / "operator")})
    monkeypatch.chdir(tmp_path)
    checks, launches = [], []

    def run(command, **kwargs):
        checks.append((command, kwargs))
        return SimpleNamespace(returncode=0)

    def execve(path, command, environment):
        launches.append((path, command, environment))

    monkeypatch.setattr(startup.subprocess, "run", run)
    monkeypatch.setattr(startup.os, "execve", execve)
    return checks, launches


def test_standard_script_delegates_to_one_validator_from_unrelated_cwd(tmp_path):
    script = Path(__file__).resolve().parents[1] / "neurons/validator.py"
    result = subprocess.run(
        [sys.executable, str(script), "--help"], cwd=tmp_path,
        capture_output=True, text=True, timeout=10,
    )
    assert result.returncode == 0
    assert "--wallet.path" in result.stdout
    assert "--subtensor.chain_endpoint" in result.stdout
    assert "sudo permission" not in result.stdout


def test_reexec_keeps_venv_wallet_state_and_proxy_out_of_arguments(host, monkeypatch, tmp_path):
    checks, launches = host
    python = str(tmp_path / "venv/bin/python")
    monkeypatch.setattr(startup.sys, "executable", python)
    monkeypatch.setenv("LAB_ARENA_VALIDATOR_STATE_DIR", "existing-state")
    monkeypatch.setenv("LAB_ARENA_WEBSHARE_PROXY_1", "https://user:secret@proxy.example:443")
    monkeypatch.setenv("LAB_ARENA_SIGNING_KEY_HASH", "trusted-pin")
    monkeypatch.setenv("LEADPOET_SUBNET_EPOCH_CUTOVER_PATH", "epoch.json")
    argv = ["--wallet.name", "existing", "--wallet.hotkey", "hk", "--arena-work-dir", "runner"]
    args = _parser().parse_args(argv)
    startup.maybe_reexec_rootful(args, argv)

    assert len(checks) == 2 and len(launches) == 1
    path, command, environment = launches[0]
    assert path == "/usr/bin/sudo"
    assert command[1] == "-n"
    assert "--preserve-env=LAB_ARENA_SIGNING_KEY_HASH" in command[2]
    assert command[4] == python
    assert command[5] == "-B"
    assert command[6] == str(Path(startup.__file__).resolve())
    child = _parser().parse_args(command[7:])
    assert child.wallet_name == "existing" and child.hotkey_name == "hk"
    assert child.wallet_path == str(tmp_path / "operator/.bittensor/wallets")
    assert child.work_dir == str(tmp_path / "runner")
    assert environment["LAB_ARENA_VALIDATOR_STATE_DIR"] == str(tmp_path / "existing-state")
    assert environment["LEADPOET_SUBNET_EPOCH_CUTOVER_PATH"] == str(tmp_path / "epoch.json")
    assert environment["LAB_ARENA_SIGNING_KEY_HASH"] == "trusted-pin"
    assert environment["LAB_ARENA_WEBSHARE_PROXY_1"].endswith("proxy.example:443")
    assert "secret" not in repr(command) + repr([c for c, _ in checks])
    preserved = command[2].partition("=")[2].split(",")
    assert "PATH" not in preserved and "HOME" not in preserved
    for _, options in checks:
        assert options["stdin"] == subprocess.DEVNULL
        assert options["stdout"] == subprocess.DEVNULL
        assert options["stderr"] == subprocess.DEVNULL
        assert options["timeout"] == startup.SUDO_CHECK_TIMEOUT_SECONDS
    assert checks[0][0] == ["/usr/bin/sudo", "-n", "-l", "--", *command[4:]]
    assert checks[1][0][-4:] == ["-I", "-S", "-c", startup._ROOT_PROBE]


def test_explicit_wallet_path_overrides_environment_and_keeps_symlink(host, monkeypatch, tmp_path):
    monkeypatch.setenv("LAB_ARENA_WALLET_PATH", "/different/wallets")
    argv = ["--wallet.path", "wallet-link"]
    startup.maybe_reexec_rootful(_parser().parse_args(argv), argv)
    child = _parser().parse_args(host[1][0][1][7:])
    assert child.wallet_path == str(tmp_path / "wallet-link")


def test_secondary_proxy_file_is_read_as_caller_and_not_again_as_root(monkeypatch, tmp_path):
    proxy = tmp_path / "proxies.env"
    proxy.write_text("QUALIFICATION_WEBSHARE_PROXY_3='http://user:secret@proxy.example:80'\nOPENROUTER_API_KEY=not-imported\n")
    proxy.chmod(0o600)
    original = {"LAB_ARENA_PROXY_ENV_FILE": str(proxy), "UNRELATED": "untouched"}
    command, environment, preserve = startup.rootful_startup_command(
        _parser().parse_args([]), [], original,
    )
    assert "LAB_ARENA_PROXY_ENV_FILE" not in environment
    assert "LAB_ARENA_PROXY_ENV_FILE" not in preserve
    assert environment["QUALIFICATION_WEBSHARE_PROXY_3"] == "http://user:secret@proxy.example:80"
    assert "OPENROUTER_API_KEY" not in environment
    assert "UNRELATED" not in preserve
    assert "secret" not in repr(command) + preserve
    assert original == {"LAB_ARENA_PROXY_ENV_FILE": str(proxy), "UNRELATED": "untouched"}


@pytest.mark.parametrize("mode", ["root", "not_linux", "--check-only", "--check-scoring-only", "no_sudo"])
def test_root_and_diagnostic_paths_do_not_elevate(host, monkeypatch, mode):
    argv = [mode] if mode.startswith("--") else []
    if mode == "root":
        monkeypatch.setattr(startup.os, "geteuid", lambda: 0)
    if mode == "not_linux":
        monkeypatch.setattr(startup.platform, "system", lambda: "Darwin")
    if mode == "no_sudo":
        monkeypatch.setattr(startup.os.path, "isfile", lambda _: False)
    startup.maybe_reexec_rootful(_parser().parse_args(argv), argv)
    assert host == ([], [])


@pytest.mark.parametrize("stage", [0, 1])
def test_denied_policy_or_authentication_returns_to_weight_startup(host, monkeypatch, stage, capsys):
    calls = []
    def run(*args, **kwargs):
        calls.append(args[0])
        return SimpleNamespace(returncode=1 if len(calls) == stage + 1 else 0)
    monkeypatch.setattr(startup.subprocess, "run", run)
    assert startup.maybe_reexec_rootful(_parser().parse_args([]), []) is None
    assert not host[1]
    assert len(calls) == stage + 1
    assert capsys.readouterr() == ("", "")


@pytest.mark.parametrize("failure", [OSError("private diagnostic"), subprocess.TimeoutExpired("sudo", 5)])
def test_probe_errors_do_not_stop_weights_or_leak_diagnostics(host, monkeypatch, failure, capsys):
    def run(*args, **kwargs):
        raise failure
    monkeypatch.setattr(startup.subprocess, "run", run)
    startup.maybe_reexec_rootful(_parser().parse_args([]), [])
    assert not host[1]
    assert capsys.readouterr() == ("", "")


def test_failed_exec_returns_to_weight_startup(host, monkeypatch, capsys):
    def execve(*args):
        raise OSError("private diagnostic")
    monkeypatch.setattr(startup.os, "execve", execve)
    startup.maybe_reexec_rootful(_parser().parse_args([]), [])
    output = capsys.readouterr()
    assert "weight loop continues" in output.err
    assert "private diagnostic" not in output.err


def test_main_attempts_startup_before_wallet_or_chain(monkeypatch):
    from lab_arena import validator
    class Handoff(Exception):
        pass
    def startup_check(args, argv):
        assert argv == ["--wallet.name", "existing"]
        assert args.wallet_name == "existing"
        raise Handoff
    monkeypatch.setattr(startup, "maybe_reexec_rootful", startup_check)
    monkeypatch.setattr(validator, "load_local_hotkey", lambda _: pytest.fail("wallet loaded before handoff"))
    with pytest.raises(Handoff):
        validator.main(["--wallet.name", "existing"])
