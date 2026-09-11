from pathlib import Path
import shlex
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "validator_restart.sh"


def test_restart_script_is_valid_bash():
    subprocess.run(["bash", "-n", str(SCRIPT)], check=True)


def test_restart_sources_exact_release_and_preflights_before_stop():
    text = SCRIPT.read_text()
    stop = text.index('timeout "$STOP_TIMEOUT" sudo systemctl stop "$SERVICE"')
    preflight = text.index('( cd "$RELEASE"')
    assert preflight < stop
    assert 'git archive --format=tar "$TARGET_SHA"' in text
    assert 'git merge-base --is-ancestor "$TARGET_SHA" origin/main' in text
    assert 'diff -qr "$STAGE" "$RELEASE"' in text
    assert 'PYTHONPATH="$RELEASE"' in text
    assert "git stash" not in text
    assert "git checkout" not in text
    assert "git reset" not in text


def test_preflight_failure_cannot_reach_service_stop(tmp_path):
    text = SCRIPT.read_text()
    start = text.index('( cd "$RELEASE"')
    end = text.index('\n\nif [ -e "$CURRENT_LINK"', start)
    preflight = text[start:end]
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    sudo = fake_bin / "sudo"
    sudo.write_text("#!/bin/sh\nexit 17\n")
    sudo.chmod(0o755)
    program = f"""set -euo pipefail
RELEASE={shlex.quote(str(ROOT))}
CANDIDATE_SERVICE_ENV=/tmp/candidate.env
PYTHON=/usr/bin/python3
{preflight}
echo STOP_REACHED
"""
    result = subprocess.run(
        ["bash", "-c", program],
        env={"PATH": f"{fake_bin}:/usr/bin:/bin"},
        text=True,
        capture_output=True,
    )
    assert result.returncode != 0
    assert "STOP_REACHED" not in result.stdout


def test_restart_uses_no_hardware_signer_or_manual_weight_call():
    text = SCRIPT.read_text()
    for forbidden in (
        "nitro-cli",
        "docker",
        "vsock",
        "aws ",
        "ENCLAVE_CID",
        "pkill",
        "set_weights",
        "author_submitExtrinsic",
    ):
        assert forbidden not in text
    assert "CANDIDATE_CID" not in text
    assert 'systemctl show -p MainPID --value "$SERVICE"' in text


def test_restart_requires_local_wallet_preflight_and_stable_supervision():
    text = SCRIPT.read_text()
    assert 'scripts/run_arena_validator.py --environment-file "$CANDIDATE_SERVICE_ENV" --check-only' in text
    assert 'systemctl start "$SERVICE"' in text
    assert 'systemctl show -p MainPID --value "$SERVICE"' in text
    assert 'stable_since=$SECONDS' in text
    assert '"$((SECONDS - stable_since))" -ge 10' in text
    assert 'install -m 0600 -o root -g root "$ENV_FILE" "$CANDIDATE_SERVICE_ENV"' in text
    assert 'sudo mv -f "$CANDIDATE_SERVICE_ENV" "$SERVICE_ENV"' in text
    assert 'WorkingDirectory=$CURRENT_LINK' in text
    unit = (ROOT / "deploy/leadpoet-arena-validator.service").read_text()
    assert "User=root" in unit
    assert "arena-validator-service.env" in unit
    assert "PYTHONDONTWRITEBYTECODE=1" in unit
    assert "EnvironmentFile=" not in unit
    assert "--enclave-cid" not in unit


def _run_durable_directory_setup(tmp_path, state_path, runner_path):
    text = SCRIPT.read_text()
    start = text.index('for durable in "$STATE_PATH" "$RUNNER_PATH"; do')
    end = text.index("\ndone", start) + len("\ndone")
    setup = text[start:end]
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    sudo = fake_bin / "sudo"
    sudo.write_text(
        """#!/bin/bash
set -euo pipefail
chmod 0700 "$GUARDED_PARENT"
status=0
if [ "$1" = install ]; then
  shift
  filtered=()
  while [ "$#" -gt 0 ]; do
    case "$1" in
      -o|-g) shift 2 ;;
      *) filtered+=("$1"); shift ;;
    esac
  done
  install "${filtered[@]}" || status=$?
else
  "$@" || status=$?
fi
chmod 0000 "$GUARDED_PARENT"
exit "$status"
"""
    )
    sudo.chmod(0o755)
    program = f"""set -euo pipefail
fail() {{ echo "ERROR: $*" >&2; exit 1; }}
STATE_PATH={shlex.quote(str(state_path))}
RUNNER_PATH={shlex.quote(str(runner_path))}
{setup}
"""
    guarded_parent = state_path.parent
    guarded_parent.chmod(0)
    try:
        return subprocess.run(
            ["bash", "-c", program],
            env={
                "PATH": f"{fake_bin}:/usr/bin:/bin",
                "GUARDED_PARENT": str(guarded_parent),
            },
            text=True,
            capture_output=True,
        )
    finally:
        guarded_parent.chmod(0o700)


def test_durable_directory_setup_checks_root_private_paths_as_root(tmp_path):
    guarded = tmp_path / "durable"
    guarded.mkdir()
    state = guarded / "state"
    runner = guarded / "runner"

    result = _run_durable_directory_setup(tmp_path, state, runner)

    assert result.returncode == 0, result.stderr
    assert state.is_dir() and runner.is_dir()
    assert state.stat().st_mode & 0o777 == 0o700
    assert runner.stat().st_mode & 0o777 == 0o700


def test_durable_directory_setup_rejects_symlink_before_install(tmp_path):
    guarded = tmp_path / "durable"
    guarded.mkdir()
    target = tmp_path / "target"
    target.mkdir(mode=0o755)
    target.chmod(0o755)
    state = guarded / "state"
    state.symlink_to(target, target_is_directory=True)
    runner = guarded / "runner"

    result = _run_durable_directory_setup(tmp_path, state, runner)

    assert result.returncode != 0
    assert "Arena durable directory is unsafe" in result.stderr
    assert target.stat().st_mode & 0o777 == 0o755
    assert not runner.exists()


@pytest.mark.parametrize("had_previous_config", [True, False])
def test_startup_failure_restores_private_env_and_previous_release(tmp_path, had_previous_config):
    text = SCRIPT.read_text()
    start = text.index("cleanup() {")
    end = text.index("\ntrap cleanup EXIT", start)
    cleanup = text[start:end]
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    sudo = fake_bin / "sudo"
    sudo.write_text(
        """#!/bin/bash
set -euo pipefail
printf '%s\n' "$*" >> "$CALL_LOG"
case "$1" in
  systemctl) exit 0 ;;
  test) shift; test "$@" ;;
  install)
    shift
    filtered=()
    while [ "$#" -gt 0 ]; do
      case "$1" in
        -o|-g) shift 2 ;;
        *) filtered+=("$1"); shift ;;
      esac
    done
    /usr/bin/install "${filtered[@]}" ;;
  rm) shift; /bin/rm "$@" ;;
  *) "$@" ;;
esac
"""
    )
    sudo.chmod(0o755)
    mv = fake_bin / "mv"
    mv.write_text(
        """#!/bin/sh
if [ "$1" = -Tf ]; then
  shift
  exec /bin/mv -f "$@"
fi
exec /bin/mv "$@"
"""
    )
    mv.chmod(0o755)

    release_root = tmp_path / "releases"
    release_root.mkdir()
    old_release = release_root / "old"
    old_release.mkdir()
    current = tmp_path / "current"
    current.symlink_to(old_release, target_is_directory=True)
    rollback = tmp_path / "rollback"
    rollback.mkdir()
    if had_previous_config:
        (rollback / "service.env").write_text("old-private-env\n")
        (rollback / "service.unit").write_text("old-unit\n")
    service_env = tmp_path / "service.env"
    service_env.write_text("new-private-env\n")
    unit_path = tmp_path / "service.unit"
    unit_path.write_text("new-unit\n")
    stage = release_root / ".candidate.test"
    stage.mkdir()
    candidate_env = tmp_path / "candidate.env"
    candidate_env.write_text("candidate\n")
    call_log = tmp_path / "calls.log"
    values = {
        "SERVICE": "validator.service",
        "CURRENT_LINK": str(current),
        "RELEASE_ROOT": str(release_root),
        "STAGE": str(stage),
        "CANDIDATE_SERVICE_ENV": str(candidate_env),
        "DRAINED": "1",
        "ACTIVATED": "0",
        "PROMOTED": "1",
        "OLD_ACTIVE": "1",
        "PREVIOUS_CURRENT_TARGET": str(old_release),
        "ROLLBACK": str(rollback),
        "SERVICE_ENV": str(service_env),
        "UNIT_PATH": str(unit_path),
    }
    assignments = "\n".join(f"{key}={shlex.quote(value)}" for key, value in values.items())
    program = f"""set -u
{assignments}
{cleanup}
false
cleanup
"""
    result = subprocess.run(
        ["bash", "-c", program],
        env={"PATH": f"{fake_bin}:/usr/bin:/bin", "CALL_LOG": str(call_log)},
        text=True,
        capture_output=True,
    )
    assert result.returncode == 1
    assert current.resolve() == old_release
    if had_previous_config:
        assert service_env.read_text() == "old-private-env\n"
        assert unit_path.read_text() == "old-unit\n"
    else:
        assert not service_env.exists()
        assert not unit_path.exists()
    assert not stage.exists()
    assert not candidate_env.exists()
    calls = call_log.read_text().splitlines()
    assert "systemctl stop validator.service" in calls
    assert "systemctl daemon-reload" in calls
    assert "systemctl start validator.service" in calls
    assert calls.index("systemctl stop validator.service") < calls.index("systemctl daemon-reload")
    assert calls.index("systemctl daemon-reload") < calls.index("systemctl start validator.service")


def test_readiness_requires_one_pid_for_ten_seconds(tmp_path):
    text = SCRIPT.read_text()
    start = text.index("deadline=$((SECONDS + READY_TIMEOUT))")
    end = text.index('\necho "SUCCESS:', start)
    readiness = text[start:end]
    program = f"""set -euo pipefail
READY_TIMEOUT=12
SERVICE=validator.service
SECONDS=0
ACTIVATED=0
fail() {{ echo "$*" >&2; exit 1; }}
sudo() {{
  if [ "$1" = systemctl ] && [ "$2" = show ]; then echo 314; fi
  return 0
}}
sleep() {{ SECONDS=$((SECONDS + 2)); }}
{readiness}
printf '%s' "$ACTIVATED"
"""
    result = subprocess.run(["bash", "-c", program], text=True, capture_output=True, check=True)
    assert result.stdout == "1"


def test_generated_unit_uses_selected_python_for_both_commands(tmp_path):
    text = SCRIPT.read_text()
    start = text.index('sed -e "s|^WorkingDirectory=')
    end = text.index('\nsudo install -m 0644 "$ROLLBACK/candidate.unit"', start)
    generation = text[start:end]
    release = tmp_path / "release"
    deploy = release / "deploy"
    deploy.mkdir(parents=True)
    (deploy / "leadpoet-arena-validator.service").write_text(
        """[Service]
WorkingDirectory=/home/ec2-user/leadpoet/validator-current
ExecStartPre=/usr/bin/python3 scripts/run_arena_validator.py --check-only
ExecStart=/usr/bin/python3 scripts/run_arena_validator.py
Environment=/home/ec2-user/.config/leadpoet/arena-validator-service.env
"""
    )
    rollback = tmp_path / "rollback"
    rollback.mkdir()
    program = f"""set -euo pipefail
RELEASE={shlex.quote(str(release))}
CURRENT_LINK=/srv/validator-current
SERVICE_ENV=/etc/leadpoet/arena-validator.env
PYTHON=/opt/venv/bin/python3
ROLLBACK={shlex.quote(str(rollback))}
{generation}
cat "$ROLLBACK/candidate.unit"
"""
    result = subprocess.run(["bash", "-c", program], text=True, capture_output=True, check=True)
    unit = result.stdout
    assert "WorkingDirectory=/srv/validator-current" in unit
    assert "ExecStartPre=/opt/venv/bin/python3 scripts/run_arena_validator.py --check-only" in unit
    assert "ExecStart=/opt/venv/bin/python3 scripts/run_arena_validator.py" in unit
    assert "Environment=/etc/leadpoet/arena-validator.env" in unit
