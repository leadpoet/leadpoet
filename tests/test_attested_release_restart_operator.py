from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "restart_attested_release_local.sh"


def test_attested_release_restart_operator_is_fail_closed() -> None:
    source = SCRIPT.read_text(encoding="utf-8")

    assert 'component="all"' in source
    assert "exact_commit_restart_v2.py" in source
    assert "--compatibility-floor" not in source
    assert "VALIDATOR_PINNED_GATEWAY_COORDINATION_MAX_ATTEMPTS" in source
    assert "VALIDATOR_PINNED_GATEWAY_COORDINATION_FILE" in source
    assert "Acquiring the independently built V2 release channel" in source
    assert "gw_restart.sh" in source
    assert "validator_restart.sh" in source
    assert "gateway_exact_release_ready" in source
    assert "validator_exact_release_ready" in source
    assert "/health/v2-authority" in source
    assert "/weights/v2/release-evidence/" in source
    assert "VALIDATOR_V2_DEPLOY_COMMIT" in source
    assert "VALIDATOR_WEIGHT_PROTOCOL" in source
    assert "use --component all" in source
    assert "trap cleanup EXIT" in source
    assert "trap 'exit 130' INT" in source
    assert "trap 'exit 143' TERM" in source
    assert source.index("run_gateway_restart") < source.index(
        "Gateway exact release is ready; waiting for the paired validator restart"
    )
    assert source.index("run_gateway_restart") < source.index(
        "printf '%s\\\\n' '$commit'"
    )


def test_attested_release_restart_operator_rejects_invalid_input() -> None:
    result = subprocess.run(
        ["bash", str(SCRIPT), "--commit", "abc123"],
        check=False,
        capture_output=True,
        text=True,
        timeout=5,
    )

    assert result.returncode == 2
    assert "lowercase full 40-character SHA" in result.stderr
    assert "Fetching current public V2 compatibility authority" not in result.stdout


def test_attested_release_restart_operator_documents_one_command_modes() -> None:
    result = subprocess.run(
        ["bash", str(SCRIPT), "--help"],
        check=False,
        capture_output=True,
        text=True,
        timeout=5,
    )

    assert result.returncode == 0
    assert "--component all|gateway|validator" in result.stdout
    assert "single-component restart is accepted only when the other component" in (
        result.stdout
    )


def _fake_operator_commands(tmp_path: Path, commit: str) -> tuple[Path, Path]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    events = tmp_path / "events"
    barrier = tmp_path / "barrier"
    gateway_complete = tmp_path / "gateway-complete"

    real_git = shutil.which("git")
    assert real_git
    git = bin_dir / "git"
    git.write_text(
        f"""#!/bin/bash
set -euo pipefail
for arg in "$@"; do
  if [ "$arg" = "fetch" ]; then
    exit 0
  fi
  if [ "$arg" = "origin/main:Leadpoet/utils/exact_commit_restart_v2.py" ]; then
    cat "$FAKE_OPERATOR_EXACT_HELPER"
    exit 0
  fi
done
exec {real_git} "$@"
""",
        encoding="utf-8",
    )

    ssh = bin_dir / "ssh"
    ssh.write_text(
        """#!/bin/bash
set -euo pipefail
command="${!#}"
record() {
  printf '%s\\n' "$1" >> "$FAKE_OPERATOR_EVENTS"
}
case "$command" in
  *validator_restart.sh*)
    record validator_start
    printf '%s\\n' "Capturing the official subnet restart start before release acquisition"
    printf '%s\\n' "Acquiring the independently built V2 release channel"
    record validator_captured
    if [[ "$command" == *"VALIDATOR_PINNED_GATEWAY_COORDINATION_FILE=''"* ]]; then
      record validator_complete
      exit 0
    fi
    for _ in $(seq 1 500); do
      if [ -e "$FAKE_OPERATOR_BARRIER" ]; then
        record validator_complete
        exit 0
      fi
      sleep 0.01
    done
    record validator_barrier_timeout
    exit 70
    ;;
  *gw_restart.sh*)
    record gateway_start
    sleep 0.05
    if [ "${FAKE_GATEWAY_RESTART_FAIL:-0}" = "1" ]; then
      record gateway_failed
      exit 73
    fi
    touch "$FAKE_OPERATOR_GATEWAY_COMPLETE"
    record gateway_complete
    ;;
  *"mv -f --"*)
    if [ ! -e "$FAKE_OPERATOR_GATEWAY_COMPLETE" ]; then
      record barrier_before_gateway
      exit 71
    fi
    touch "$FAKE_OPERATOR_BARRIER"
    record barrier_released
    ;;
  *gateway_exact_release_ready*)
    record gateway_verified
    if [ "${FAKE_GATEWAY_VERIFY_FAIL:-0}" = "1" ]; then
      exit 74
    fi
    ;;
  *validator_exact_release_ready*)
    record validator_verified
    if [ "${FAKE_VALIDATOR_VERIFY_FAIL:-0}" = "1" ]; then
      exit 75
    fi
    ;;
  *"docker inspect"*VALIDATOR_V2_DEPLOY_COMMIT*)
    printf '%s\\n' "$FAKE_VALIDATOR_COMMIT"
    record validator_active_probe
    ;;
  *"/build-info"*git_commit*)
    printf '%s\\n' "$FAKE_GATEWAY_COMMIT"
    record gateway_active_probe
    ;;
  *"rm -f --"*)
    rm -f "$FAKE_OPERATOR_BARRIER"
    record barrier_cleanup
    ;;
  *)
    record unknown_ssh_command
    printf '%s\\n' "$command" >&2
    exit 72
    ;;
esac
""",
        encoding="utf-8",
    )
    git.chmod(0o755)
    ssh.chmod(0o755)

    for name in ("gateway.pem", "validator.pem"):
        path = tmp_path / name
        path.write_text("test-only\n", encoding="utf-8")
        path.chmod(0o600)

    os.environ.pop("FAKE_OPERATOR_EVENTS", None)
    env = tmp_path / "operator-env"
    env.write_text(
        "\n".join(
            (
                f"FAKE_OPERATOR_EVENTS={events}",
                f"FAKE_OPERATOR_BARRIER={barrier}",
                f"FAKE_OPERATOR_GATEWAY_COMPLETE={gateway_complete}",
                f"FAKE_GATEWAY_COMMIT={commit}",
                f"FAKE_VALIDATOR_COMMIT={commit}",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    return bin_dir, events


def _operator_env(tmp_path: Path, bin_dir: Path, commit: str) -> dict[str, str]:
    values = {
        **os.environ,
        "PATH": str(bin_dir) + os.pathsep + os.environ["PATH"],
        "LEADPOET_GATEWAY_SSH_KEY": str(tmp_path / "gateway.pem"),
        "LEADPOET_VALIDATOR_SSH_KEY": str(tmp_path / "validator.pem"),
        "FAKE_GATEWAY_COMMIT": commit,
        "FAKE_VALIDATOR_COMMIT": commit,
        "FAKE_OPERATOR_EXACT_HELPER": str(
            ROOT / "Leadpoet" / "utils" / "exact_commit_restart_v2.py"
        ),
    }
    for line in (tmp_path / "operator-env").read_text(encoding="utf-8").splitlines():
        key, value = line.split("=", 1)
        values[key] = value
    return values


def test_paired_operator_waits_for_full_gateway_success_before_validator(
    tmp_path: Path,
) -> None:
    commit = subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "origin/main"],
        text=True,
    ).strip()
    bin_dir, events = _fake_operator_commands(tmp_path, commit)

    result = subprocess.run(
        ["bash", str(SCRIPT), "--commit", commit],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
        env=_operator_env(tmp_path, bin_dir, commit),
    )

    assert result.returncode == 0, result.stderr
    observed = events.read_text(encoding="utf-8").splitlines()
    required = [
        "validator_start",
        "validator_captured",
        "gateway_start",
        "gateway_complete",
        "barrier_released",
        "validator_complete",
        "gateway_verified",
        "validator_verified",
    ]
    positions = {event: observed.index(event) for event in required}
    assert (
        positions["validator_start"]
        < positions["validator_captured"]
        < positions["gateway_start"]
        < positions["gateway_complete"]
    )
    # The validator may observe the atomically published coordination marker
    # before the parent SSH process records its post-move diagnostic event.
    # Both events must remain gateway-gated and precede release verification,
    # but their relative log order is intentionally unconstrained.
    assert (
        positions["gateway_complete"]
        < positions["barrier_released"]
        < positions["gateway_verified"]
        < positions["validator_verified"]
    )
    assert (
        positions["gateway_complete"]
        < positions["validator_complete"]
        < positions["gateway_verified"]
        < positions["validator_verified"]
    )
    assert "barrier_before_gateway" not in observed
    assert "SUCCESS: gateway and validator are aligned" in result.stdout


def test_gateway_only_operator_rejects_mismatched_validator(
    tmp_path: Path,
) -> None:
    commit = subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "origin/main"],
        text=True,
    ).strip()
    bin_dir, events = _fake_operator_commands(tmp_path, commit)
    environment = _operator_env(tmp_path, bin_dir, commit)
    environment["FAKE_VALIDATOR_COMMIT"] = "b" * 40

    result = subprocess.run(
        [
            "bash",
            str(SCRIPT),
            "--commit",
            commit,
            "--component",
            "gateway",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
        env=environment,
    )

    assert result.returncode == 1
    assert "use --component all" in result.stderr
    observed = events.read_text(encoding="utf-8").splitlines()
    assert "validator_active_probe" in observed
    assert "gateway_start" not in observed


def test_gateway_only_operator_requires_healthy_matching_validator(
    tmp_path: Path,
) -> None:
    commit = subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "origin/main"],
        text=True,
    ).strip()
    bin_dir, events = _fake_operator_commands(tmp_path, commit)

    result = subprocess.run(
        [
            "bash",
            str(SCRIPT),
            "--commit",
            commit,
            "--component",
            "gateway",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
        env=_operator_env(tmp_path, bin_dir, commit),
    )

    assert result.returncode == 0, result.stderr
    observed = events.read_text(encoding="utf-8").splitlines()
    required = [
        "validator_active_probe",
        "validator_verified",
        "gateway_start",
        "gateway_complete",
        "gateway_verified",
    ]
    positions = [observed.index(event) for event in required]
    assert positions == sorted(positions)
    assert observed.count("validator_verified") == 2


def test_gateway_only_operator_rejects_unhealthy_matching_validator(
    tmp_path: Path,
) -> None:
    commit = subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "origin/main"],
        text=True,
    ).strip()
    bin_dir, events = _fake_operator_commands(tmp_path, commit)
    environment = _operator_env(tmp_path, bin_dir, commit)
    environment["FAKE_VALIDATOR_VERIFY_FAIL"] = "1"

    result = subprocess.run(
        [
            "bash",
            str(SCRIPT),
            "--commit",
            commit,
            "--component",
            "gateway",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
        env=environment,
    )

    assert result.returncode == 75
    observed = events.read_text(encoding="utf-8").splitlines()
    assert "validator_verified" in observed
    assert "gateway_start" not in observed


def test_validator_only_operator_rejects_mismatched_gateway(
    tmp_path: Path,
) -> None:
    commit = subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "origin/main"],
        text=True,
    ).strip()
    bin_dir, events = _fake_operator_commands(tmp_path, commit)
    environment = _operator_env(tmp_path, bin_dir, commit)
    environment["FAKE_GATEWAY_COMMIT"] = "b" * 40

    result = subprocess.run(
        [
            "bash",
            str(SCRIPT),
            "--commit",
            commit,
            "--component",
            "validator",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
        env=environment,
    )

    assert result.returncode == 1
    assert "use --component all" in result.stderr
    observed = events.read_text(encoding="utf-8").splitlines()
    assert "gateway_active_probe" in observed
    assert "validator_start" not in observed


def test_validator_only_operator_requires_healthy_matching_gateway(
    tmp_path: Path,
) -> None:
    commit = subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "origin/main"],
        text=True,
    ).strip()
    bin_dir, events = _fake_operator_commands(tmp_path, commit)

    result = subprocess.run(
        [
            "bash",
            str(SCRIPT),
            "--commit",
            commit,
            "--component",
            "validator",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
        env=_operator_env(tmp_path, bin_dir, commit),
    )

    assert result.returncode == 0, result.stderr
    observed = events.read_text(encoding="utf-8").splitlines()
    required = [
        "gateway_active_probe",
        "gateway_verified",
        "validator_start",
        "validator_complete",
        "validator_verified",
    ]
    positions = [observed.index(event) for event in required]
    assert positions == sorted(positions)
    assert observed.count("gateway_verified") == 2


def test_validator_only_operator_rejects_unhealthy_matching_gateway(
    tmp_path: Path,
) -> None:
    commit = subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "origin/main"],
        text=True,
    ).strip()
    bin_dir, events = _fake_operator_commands(tmp_path, commit)
    environment = _operator_env(tmp_path, bin_dir, commit)
    environment["FAKE_GATEWAY_VERIFY_FAIL"] = "1"

    result = subprocess.run(
        [
            "bash",
            str(SCRIPT),
            "--commit",
            commit,
            "--component",
            "validator",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
        env=environment,
    )

    assert result.returncode == 74
    observed = events.read_text(encoding="utf-8").splitlines()
    assert "gateway_verified" in observed
    assert "validator_start" not in observed


def test_paired_operator_does_not_release_validator_after_gateway_failure(
    tmp_path: Path,
) -> None:
    commit = subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "origin/main"],
        text=True,
    ).strip()
    bin_dir, events = _fake_operator_commands(tmp_path, commit)
    environment = _operator_env(tmp_path, bin_dir, commit)
    environment["FAKE_GATEWAY_RESTART_FAIL"] = "1"

    result = subprocess.run(
        ["bash", str(SCRIPT), "--commit", commit],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
        env=environment,
    )

    assert result.returncode == 73
    observed = events.read_text(encoding="utf-8").splitlines()
    assert "validator_captured" in observed
    assert "gateway_failed" in observed
    assert "barrier_released" not in observed
    assert "validator_complete" not in observed
    assert "gateway_verified" not in observed
    assert "validator_verified" not in observed
    assert observed[-1] == "barrier_cleanup"
