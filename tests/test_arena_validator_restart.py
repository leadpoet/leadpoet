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
    snapshot = text.index('( cd "$RELEASE" && PYTHONPATH="$RELEASE" "$PYTHON" - "$ENV_FILE"')
    preflight = text.index('( cd "$RELEASE" && sudo env PYTHONDONTWRITEBYTECODE=1')
    assert snapshot < preflight
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
    start = text.index('( cd "$RELEASE" && sudo env PYTHONDONTWRITEBYTECODE=1')
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
    assert 'if ! sudo systemctl start "$SERVICE"; then' in text
    assert 'systemctl show -p MainPID --value "$SERVICE"' in text
    assert 'stable_since=$SECONDS' in text
    assert '"$((SECONDS - stable_since))" -ge 10' in text
    assert "write_environment_snapshot(Path(sys.argv[1]), Path(sys.argv[2]))" in text
    assert 'install -m 0600 -o root -g root "$SNAPSHOT_SERVICE_ENV" "$CANDIDATE_SERVICE_ENV"' in text
    assert 'install -m 0600 -o root -g root "$ENV_FILE" "$CANDIDATE_SERVICE_ENV"' not in text
    assert 'sudo mv -f "$CANDIDATE_SERVICE_ENV" "$SERVICE_ENV"' in text
    assert 'WorkingDirectory=$CURRENT_LINK' in text
    unit = (ROOT / "deploy/leadpoet-arena-validator.service").read_text()
    assert "User=root" in unit
    assert "arena-validator-service.env" in unit
    assert "PYTHONDONTWRITEBYTECODE=1" in unit
    assert "Restart=on-failure" in unit
    assert "RestartSec=15" in unit
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
    snapshot_env = tmp_path / "snapshot.env"
    snapshot_env.write_text("snapshot\n")
    call_log = tmp_path / "calls.log"
    values = {
        "SERVICE": "validator.service",
        "CURRENT_LINK": str(current),
        "RELEASE_ROOT": str(release_root),
        "STAGE": str(stage),
        "CANDIDATE_SERVICE_ENV": str(candidate_env),
        "SNAPSHOT_SERVICE_ENV": str(snapshot_env),
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
    assert not snapshot_env.exists()
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


def test_initial_start_failure_waits_for_supervised_retry():
    text = SCRIPT.read_text()
    start = text.index('if ! sudo systemctl start "$SERVICE"; then')
    end = text.index('\necho "SUCCESS:', start)
    startup = text[start:end]
    program = f"""set -euo pipefail
READY_TIMEOUT=20
SERVICE=validator.service
SECONDS=0
ACTIVATED=0
fail() {{ echo "$*" >&2; exit 1; }}
sudo() {{
  if [ "$1" = systemctl ] && [ "$2" = start ]; then return 1; fi
  if [ "$1" = systemctl ] && [ "$2" = show ]; then
    if [ "$SECONDS" -ge 4 ]; then echo 314; else echo 0; fi
    return 0
  fi
  if [ "$1" = systemctl ] && [ "$2" = is-active ]; then
    [ "$SECONDS" -ge 4 ]
    return
  fi
  return 1
}}
sleep() {{ SECONDS=$((SECONDS + 2)); }}
{startup}
printf '%s' "$ACTIVATED"
"""
    result = subprocess.run(
        ["bash", "-c", program], text=True, capture_output=True, check=True
    )
    assert result.stdout == "1"
    assert "waiting for supervised systemd retries" in result.stderr


@pytest.mark.parametrize(
    ("ready_after", "expected_success"),
    ((100, True), (280, True), (295, False), (1000, False)),
)
def test_default_startup_window_covers_rate_limited_retries(
    ready_after, expected_success
):
    text = SCRIPT.read_text()
    timeout_setting = next(
        line for line in text.splitlines() if line.startswith("READY_TIMEOUT=")
    )
    start = text.index('if ! sudo systemctl start "$SERVICE"; then')
    end = text.index('\necho "SUCCESS:', start)
    program = f"""set -euo pipefail
unset VALIDATOR_READY_TIMEOUT_SECONDS
{timeout_setting}
SERVICE=validator.service
SECONDS=0
ACTIVATED=0
fail() {{ echo "$*" >&2; exit 1; }}
sudo() {{
  if [ "$2" = start ]; then return 1; fi
  if [ "$2" = show ]; then
    if [ "$SECONDS" -ge {ready_after} ]; then echo 314; else echo 0; fi
    return 0
  fi
  if [ "$2" = is-active ]; then [ "$SECONDS" -ge {ready_after} ]; return; fi
  return 1
}}
sleep() {{ SECONDS=$((SECONDS + 2)); }}
{text[start:end]}
printf '%s' "$ACTIVATED"
"""
    result = subprocess.run(
        ["bash", "-c", program], text=True, capture_output=True, timeout=5
    )
    assert (result.returncode == 0) is expected_success, result.stderr
    if expected_success:
        assert result.stdout == "1"
    else:
        assert "normal Arena validator did not remain ready" in result.stderr


def test_failed_start_and_exhausted_readiness_window_remains_fail_closed():
    text = SCRIPT.read_text()
    start = text.index('if ! sudo systemctl start "$SERVICE"; then')
    end = text.index('\necho "SUCCESS:', start)
    startup = text[start:end]
    program = f"""set -euo pipefail
READY_TIMEOUT=6
SERVICE=validator.service
SECONDS=0
ACTIVATED=0
fail() {{ echo "ERROR: $*" >&2; exit 1; }}
sudo() {{
  if [ "$1" = systemctl ] && [ "$2" = show ]; then echo 0; return 0; fi
  return 1
}}
sleep() {{ SECONDS=$((SECONDS + 2)); }}
{startup}
"""
    result = subprocess.run(
        ["bash", "-c", program], text=True, capture_output=True
    )
    assert result.returncode != 0
    assert "normal Arena validator did not remain ready" in result.stderr


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


# Interpreter selection reads the real canonical unit format.
TEXT = (ROOT / "validator_restart.sh").read_text()
SELECTOR = TEXT[TEXT.index("select_validator_python() {"):TEXT.index("\ncleanup() {")]
TEMPLATE = (ROOT / "deploy/leadpoet-arena-validator.service").read_text()
PATCHED = "/home/ec2-user/verified-validator/bin/python3"
UNSET = object()


def _select_interpreter(tmp_path, unit=TEMPLATE, override=UNSET, metadata=None, show_exit=0, link=None):
    unit_path = tmp_path / "validator.service"
    if unit is not None:
        unit_path.write_text(unit)
    if link is not None:
        unit_path.symlink_to(link)
    if metadata is None:
        metadata = ("LoadState=loaded\nFragmentPath=" + str(unit_path)
                    + "\nDropInPaths=\nNeedDaemonReload=no\n")
    show = tmp_path / "systemctl"
    show.write_text("#!/bin/sh\n"
                    '[ "$#" = 4 ] && [ "$1" = show ] && [ "$2" = leadpoet-arena-validator.service ] '
                    '&& [ "$3" = --all ] && [ "$4" = --property=LoadState,FragmentPath,DropInPaths,NeedDaemonReload ] || exit 29\n'
                    + "printf '%s' " + shlex.quote(metadata) + "\nexit " + str(show_exit) + "\n")
    show.chmod(0o755)
    setting = "unset VALIDATOR_PYTHON_BIN" if override is UNSET else (
        "VALIDATOR_PYTHON_BIN=" + shlex.quote(override))
    program = f"""set -euo pipefail
fail() {{ echo "$*" >&2; exit 1; }}
UNIT_PATH={shlex.quote(str(unit_path))}
SERVICE=leadpoet-arena-validator.service
{setting}
{SELECTOR}
select_validator_python
[[ "$PYTHON" =~ ^/[A-Za-z0-9_./-]+$ ]] && [ "$PYTHON" != / ] || fail invalid-path
echo "$PYTHON"
echo SELECTION_PASSED
"""
    return subprocess.run(["bash", "-c", program], text=True, capture_output=True,
                          env={"PATH": str(tmp_path) + ":/usr/bin:/bin"})


def _installed_unit():
    return TEMPLATE.replace("=/usr/bin/python3 ", "=" + PATCHED + " ")


def test_installed_canonical_interpreter_is_reused(tmp_path):
    result = _select_interpreter(tmp_path, _installed_unit())
    assert result.returncode == 0
    assert result.stdout.splitlines() == [PATCHED, "SELECTION_PASSED"]


def test_canonical_command_whitespace_and_comments_are_accepted(tmp_path):
    text = _installed_unit().replace("ExecStart=", "  ExecStart=").replace("--environment-file ", "--environment-file\t")
    result = _select_interpreter(tmp_path, "# comment\n" + text)
    assert result.returncode == 0
    assert result.stdout.splitlines()[0] == PATCHED


def test_systemd_inspection_failure_cannot_fall_back(tmp_path):
    result = _select_interpreter(tmp_path, None, show_exit=4)
    assert result.returncode != 0
    assert "SELECTION_PASSED" not in result.stdout


@pytest.mark.parametrize("exists", [True, False])
def test_unit_symlinks_are_rejected(tmp_path, exists):
    target = tmp_path / "target.service"
    if exists:
        target.write_text(_installed_unit())
    result = _select_interpreter(tmp_path, None, link=target)
    assert result.returncode != 0
    assert "SELECTION_PASSED" not in result.stdout


def test_explicit_override_precedes_even_ambiguous_installed_unit(tmp_path):
    result = _select_interpreter(tmp_path, "not a unit", override="/operator/venv/bin/python3",
                    metadata="must not be inspected")
    assert result.returncode == 0
    assert result.stdout.splitlines()[0] == "/operator/venv/bin/python3"


@pytest.mark.parametrize("override", ["", "python3", "/path with spaces/python3", "/"])
def test_invalid_explicit_override_cannot_fall_back(tmp_path, override):
    result = _select_interpreter(tmp_path, _installed_unit(), override=override)
    assert result.returncode != 0
    assert "SELECTION_PASSED" not in result.stdout


def test_initial_install_only_uses_old_default_when_unit_not_found(tmp_path):
    metadata = "LoadState=not-found\nFragmentPath=\nDropInPaths=\nNeedDaemonReload=no\n"
    result = _select_interpreter(tmp_path, None, metadata=metadata)
    assert result.returncode == 0
    assert result.stdout.splitlines()[0] == "/home/ec2-user/arena-validator-venv311/bin/python3"


@pytest.mark.parametrize("change", [
    lambda text: text.replace("ExecStart=", "ExecStart=/different/python3 ignored\nExecStart="),
    lambda text: text.replace("ExecStartPre=", "ExecStartPre=\nExecStartPre="),
    lambda text: text.replace("ExecStartPre=" + PATCHED, "ExecStartPre=/different/python3"),
    lambda text: text.replace("ExecStart=" + PATCHED, "ExecStart=/usr/bin/env python3"),
    lambda text: text.replace("ExecStart=" + PATCHED, 'ExecStart="' + PATCHED + '"'),
    lambda text: text.replace("ExecStart=" + PATCHED, "ExecStart=$PYTHON"),
    lambda text: text.replace("ExecStart=" + PATCHED, "ExecStart=%h/venv/bin/python3"),
    lambda text: text.replace("ExecStart=", "ExecStart ="),
    lambda text: text.replace("ExecStart=", "ExecStart=\x00"),
    lambda text: text.replace("Environment=PYTHONDONTWRITEBYTECODE=1", "Environment=PYTHONDONTWRITEBYTECODE=1 \\"),
    lambda text: ".include /other/unit\n" + text,
    lambda text: text.replace(" --check-only", " --check-only --extra"),
    lambda text: text.replace("[Service]", "[Unexpected]"),
    lambda text: text.replace("ExecStartPre=", "#ExecStartPre="),
    lambda text: text.replace("--environment-file /home", "--environment-file /different", 1),
])
def test_ambiguous_or_unsupported_unit_fails_before_followons(tmp_path, change):
    result = _select_interpreter(tmp_path, change(_installed_unit()))
    assert result.returncode != 0
    assert "SELECTION_PASSED" not in result.stdout


@pytest.mark.parametrize("metadata", [
    "LoadState=loaded\nFragmentPath=/other/unit\nDropInPaths=\nNeedDaemonReload=no\n",
    "LoadState=loaded\nFragmentPath=UNIT\nDropInPaths=/override.conf\nNeedDaemonReload=no\n",
    "LoadState=loaded\nFragmentPath=UNIT\nDropInPaths=\nNeedDaemonReload=yes\n",
    "LoadState=loaded\nLoadState=loaded\nFragmentPath=UNIT\nDropInPaths=\nNeedDaemonReload=no\n",
    "LoadState=loaded\nFragmentPath=UNIT\nNeedDaemonReload=no\n",
])
def test_effective_unit_ambiguity_is_rejected(tmp_path, metadata):
    result = _select_interpreter(tmp_path, _installed_unit(), metadata=metadata.replace("UNIT", str(tmp_path / "validator.service")))
    assert result.returncode != 0
    assert "SELECTION_PASSED" not in result.stdout


def test_missing_file_but_loaded_unit_is_not_initial_install(tmp_path):
    result = _select_interpreter(tmp_path, None)
    assert result.returncode != 0
    assert "SELECTION_PASSED" not in result.stdout


def test_selection_runs_under_lock_before_existing_trust_gate_and_stop():
    selection = TEXT.index("\nselect_validator_python\n")
    assert TEXT.index('flock -n 9') < selection
    assert selection < TEXT.index('"$PYTHON" -I -B "$RELEASE/scripts/check_validator_python_security.py"')
    assert selection < TEXT.index('timeout "$STOP_TIMEOUT" sudo systemctl stop "$SERVICE"')
