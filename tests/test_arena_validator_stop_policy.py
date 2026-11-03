"""The loaded old unit must allow a full drain before any destructive stop."""
from pathlib import Path
import shlex
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "validator_restart.sh"
PYTHON = "/usr/bin/python3"
SAFE = "LoadState=loaded\nKillMode=mixed\nKillSignal=15\nTimeoutStopUSec=2h 35min\n"


def _run_to_stop(tmp_path, policy=SAFE, *, override=False, inspection_exit=0,
                 installed=True):
    text = SCRIPT.read_text()
    functions = text[text.index("select_validator_python() {"):text.index("\ncleanup() {")]
    cleanup = text[text.index("cleanup() {"):text.index("\ntrap cleanup EXIT")]
    stop = text[text.index("# Only now drain existing scoring work"):
                text.index("\nPROMOTED=1", text.index("# Only now drain existing scoring work"))]
    # Use the actual main-path call if present. On the predecessor, this is
    # empty and the unsafe old unit reaches the real destructive-stop block.
    policy_call = "validate_validator_stop_policy" if "\nvalidate_validator_stop_policy\n" in text else ""
    unit = tmp_path / "validator.service"
    if installed:
        template = (ROOT / "deploy/leadpoet-arena-validator.service").read_text()
        unit.write_text(template)
    calls = tmp_path / "calls"
    selected = ("VALIDATOR_PYTHON_BIN=" + shlex.quote(PYTHON)
                if override else "unset VALIDATOR_PYTHON_BIN")
    load_state = "loaded" if installed else "not-found"
    selection = (f"LoadState={load_state}\nFragmentPath={unit if installed else ''}"
                 "\nDropInPaths=\nNeedDaemonReload=no\n")
    program = f"""set -euo pipefail
SERVICE=leadpoet-arena-validator.service
UNIT_PATH={shlex.quote(str(unit))}
CALL_LOG={shlex.quote(str(calls))}
STOP_TIMEOUT=9300
DRAINED=0
ACTIVATED=0
PROMOTED=0
OLD_ACTIVE=0
STAGE=""
CANDIDATE_SERVICE_ENV=""
SNAPSHOT_SERVICE_ENV=""
{selected}
fail() {{ echo "ERROR: $*" >&2; exit 1; }}
systemctl() {{
  case "$*" in
    *--property=LoadState,FragmentPath,DropInPaths,NeedDaemonReload)
      printf '%s' {shlex.quote(selection)} ;;
    *--property=LoadState,KillMode,KillSignal,TimeoutStopUSec)
      printf 'inspect\\n' >> "$CALL_LOG"
      printf '%s' {shlex.quote(policy)}
      return {inspection_exit} ;;
    *) return 29 ;;
  esac
}}
sudo() {{
  case "$*" in
    'systemctl show -p LoadState --value '* ) printf '%s\\n' {load_state} ;;
    'systemctl show -p MainPID --value '* ) echo 0 ;;
    'systemctl stop '* ) printf 'stop\\n' >> "$CALL_LOG" ;;
    *) return 30 ;;
  esac
}}
timeout() {{ shift; "$@"; }}
{functions}
{cleanup}
trap cleanup EXIT
select_validator_python
{policy_call}
{stop}
echo FOLLOWON_REACHED
"""
    result = subprocess.run(["bash", "-c", program], text=True, capture_output=True,
                            timeout=5)
    return result, calls.read_text().splitlines() if calls.exists() else []


@pytest.mark.parametrize("override", [False, True])
def test_unsafe_loaded_old_unit_is_rejected_without_stop_or_cleanup_stop(tmp_path, override):
    policy = SAFE.replace("mixed", "control-group").replace("2h 35min", "1min 30s")
    result, calls = _run_to_stop(tmp_path, policy, override=override)
    assert result.returncode != 0
    assert "stop" not in calls
    assert "FOLLOWON_REACHED" not in result.stdout


@pytest.mark.parametrize("override", [False, True])
def test_safe_loaded_unit_still_drains_normally(tmp_path, override):
    result, calls = _run_to_stop(tmp_path, override=override)
    assert result.returncode == 0, result.stderr
    assert calls == ["inspect", "stop"]
    assert "FOLLOWON_REACHED" in result.stdout


@pytest.mark.parametrize("override", [False, True])
def test_missing_unit_first_install_does_not_stop(tmp_path, override):
    result, calls = _run_to_stop(tmp_path, "LoadState=not-found\n", override=override,
                                 installed=False)
    assert result.returncode == 0, result.stderr
    assert calls == ["inspect"]


@pytest.mark.parametrize("duration", ["9300s", "155min", "2h35min", "2h 35min 0s",
                                     "2h 34min 60s", "9300000ms", "9300000000us",
                                     "0.107638888889d", "3h", "infinity"])
def test_sufficient_loaded_duration_formats_are_accepted(tmp_path, duration):
    result, calls = _run_to_stop(tmp_path, SAFE.replace("2h 35min", duration), override=True)
    assert result.returncode == 0, result.stderr
    assert calls == ["inspect", "stop"]


@pytest.mark.parametrize("policy", [
    SAFE.replace("mixed", "control-group"),
    SAFE.replace("mixed", "process"),
    SAFE.replace("KillSignal=15", "KillSignal=9"),
    SAFE.replace("2h 35min", "9299.999999s"),
    SAFE.replace("2h 35min", "9299.9999999999999999999999999999999999s"),
    SAFE.replace("2h 35min", ""),
    SAFE.replace("2h 35min", "9300"),
    SAFE.replace("2h 35min", "9300seconds garbage"),
    SAFE.replace("2h 35min", "-9300s"),
    SAFE.replace("2h 35min", "NaNs"),
    SAFE.replace("2h 35min", "Infinity"),
    SAFE.replace("2h 35min", "1e10s"),
    SAFE.replace("LoadState=loaded", "LoadState=masked"),
    SAFE.replace("KillMode=mixed\n", ""),
    SAFE + "KillSignal=15\n",
    SAFE + "UnknownSetting=safe\n",
    "",
])
def test_unknown_unsafe_or_ambiguous_loaded_policy_fails_closed(tmp_path, policy):
    result, calls = _run_to_stop(tmp_path, policy, override=True)
    assert result.returncode != 0
    assert "stop" not in calls
    assert "FOLLOWON_REACHED" not in result.stdout


def test_failed_loaded_policy_inspection_cannot_stop_even_with_override(tmp_path):
    result, calls = _run_to_stop(tmp_path, override=True, inspection_exit=4)
    assert result.returncode != 0
    assert calls == ["inspect"]


def test_stop_validation_precedes_pair_permit_and_destructive_cleanup_flag():
    text = SCRIPT.read_text()
    check = text.index("\nvalidate_validator_stop_policy\n")
    assert check < text.index('if [ -n "${VALIDATOR_PAIR_PERMIT:-}" ]; then')
    assert check < text.index("\nDRAINED=1\n")
    assert check < text.index('timeout "$STOP_TIMEOUT" sudo systemctl stop "$SERVICE"')
    assert "TimeoutStopSec=9300" in (ROOT / "deploy/leadpoet-arena-validator.service").read_text()
