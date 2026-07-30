from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "restart_attested_release_local.sh"


def test_attested_release_restart_operator_is_fail_closed() -> None:
    source = SCRIPT.read_text(encoding="utf-8")

    expected_key = "$HOME/Downloads/leadpoet-2026-07-28.pem"
    assert (
        f'GATEWAY_KEY="${{LEADPOET_GATEWAY_SSH_KEY:-{expected_key}}}"'
        in source
    )
    assert (
        f'VALIDATOR_KEY="${{LEADPOET_VALIDATOR_SSH_KEY:-{expected_key}}}"'
        in source
    )
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
    assert 'if [ "$commit" != "$branch_commit" ]; then' in source
    assert 'restart_arguments="--commit \'$commit\'"' in source
    assert "VALIDATOR_COORDINATED_EXPECTED_COMMIT" in source
    assert "selected validator launcher is not the exact candidate Git blob" in source
    assert "git -C '$VALIDATOR_REPO_ROOT' diff --quiet" in source
    assert source.index("    run_gateway_restart\n") < source.index(
        '    publish_coordination_value "$commit"\n'
    )
    assert source.index('kill -TERM "$validator_job"') < source.index(
        'coordination_remote_command "failed:$commit"'
    )
    assert source.index('kill -TERM "$validator_job"') < source.index(
        'for _ in $(seq 1 "$VALIDATOR_FAILURE_CLEANUP_ATTEMPTS")'
    )
    assert "VALIDATOR_FAILURE_MARKER_ATTEMPTS" in source


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
    gateway_started = tmp_path / "gateway-started"
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
    bash -n -c "$command"
    record validator_command_syntax
    record validator_start
    trap 'record validator_cancelled; record validator_cleanup; exit 143' HUP INT TERM
    if [[ "$command" == *" --commit "* ]]; then
      record validator_exact_commit_handoff
    else
      record validator_forward_handoff
    fi
    record validator_prepare_started
    printf '%s\\n' "Capturing the official subnet restart start before release acquisition"
    printf '%s\\n' "Acquiring the independently built V2 release channel"
    record validator_captured
    if [[ "$command" == *"VALIDATOR_PINNED_GATEWAY_COORDINATION_FILE=''"* ]]; then
      record validator_image_prepared
      record validator_activation
      record validator_complete
      exit 0
    fi
    for _ in $(seq 1 500); do
      if [ -e "$FAKE_OPERATOR_GATEWAY_STARTED" ]; then
        record validator_image_prepared
        break
      fi
      sleep 0.01
    done
    for _ in $(seq 1 500); do
      if [ -e "$FAKE_OPERATOR_BARRIER" ]; then
        marker="$(cat "$FAKE_OPERATOR_BARRIER")"
        if [ "$marker" = "$FAKE_VALIDATOR_COMMIT" ]; then
          record validator_activation
          record validator_complete
          exit 0
        fi
        if [ "$marker" = "failed:$FAKE_VALIDATOR_COMMIT" ]; then
          record validator_alignment_failed
          record validator_cleanup
          exit 76
        fi
        record validator_invalid_barrier
        exit 77
      fi
      sleep 0.01
    done
    record validator_barrier_timeout
    exit 70
    ;;
  *gw_restart.sh*)
    record gateway_start
    touch "$FAKE_OPERATOR_GATEWAY_STARTED"
    sleep 0.10
    if [ "${FAKE_GATEWAY_RESTART_FAIL:-0}" = "1" ]; then
      record gateway_failed
      exit 73
    fi
    touch "$FAKE_OPERATOR_GATEWAY_COMPLETE"
    record gateway_complete
    ;;
  *"mv -f --"*)
    if [[ "$command" == *"failed:"* ]]; then
      record failure_barrier_publish_started
      if [ -n "${FAKE_FAILURE_MARKER_DELAY_SECONDS:-}" ]; then
        sleep "$FAKE_FAILURE_MARKER_DELAY_SECONDS"
      fi
      printf '%s\\n' "failed:$FAKE_VALIDATOR_COMMIT" > "$FAKE_OPERATOR_BARRIER"
      record failure_barrier_released
    elif [ ! -e "$FAKE_OPERATOR_GATEWAY_COMPLETE" ]; then
      record barrier_before_gateway
      exit 71
    else
      printf '%s\\n' "$FAKE_VALIDATOR_COMMIT" > "$FAKE_OPERATOR_BARRIER"
      record barrier_released
    fi
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
                f"FAKE_OPERATOR_GATEWAY_STARTED={gateway_started}",
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


def test_paired_operator_overlaps_preparation_and_gates_validator_activation(
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
        "validator_forward_handoff",
        "validator_command_syntax",
        "validator_prepare_started",
        "validator_captured",
        "gateway_start",
        "validator_image_prepared",
        "gateway_complete",
        "barrier_released",
        "validator_activation",
        "validator_complete",
        "gateway_verified",
        "validator_verified",
    ]
    positions = {event: observed.index(event) for event in required}
    assert (
        positions["validator_prepare_started"]
        < positions["validator_captured"]
        < positions["gateway_start"]
        < positions["validator_image_prepared"]
        < positions["gateway_complete"]
    )
    assert (
        positions["gateway_complete"]
        < positions["barrier_released"]
        < positions["validator_activation"]
        < positions["validator_complete"]
        < positions["gateway_verified"]
        < positions["validator_verified"]
    )
    assert "barrier_before_gateway" not in observed
    assert "validator_exact_commit_handoff" not in observed
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


def test_paired_operator_failure_marker_cleans_prepared_validator(
    tmp_path: Path,
) -> None:
    commit = subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "origin/main"],
        text=True,
    ).strip()
    bin_dir, events = _fake_operator_commands(tmp_path, commit)
    environment = _operator_env(tmp_path, bin_dir, commit)
    environment["FAKE_GATEWAY_RESTART_FAIL"] = "1"
    environment["FAKE_FAILURE_MARKER_DELAY_SECONDS"] = "0.5"

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
    required = [
        "validator_forward_handoff",
        "validator_prepare_started",
        "validator_captured",
        "gateway_start",
        "validator_image_prepared",
        "gateway_failed",
        "validator_cancelled",
        "validator_cleanup",
        "failure_barrier_released",
    ]
    positions = [observed.index(event) for event in required]
    assert positions == sorted(positions)
    assert observed.index("gateway_failed") < observed.index(
        "failure_barrier_publish_started"
    )
    assert observed.index("validator_cleanup") < observed.index(
        "failure_barrier_released"
    )
    assert "validator_alignment_failed" not in observed
    assert "barrier_released" not in observed
    assert "validator_activation" not in observed
    assert "validator_complete" not in observed
    assert "gateway_verified" not in observed
    assert "validator_verified" not in observed
    assert "barrier_cleanup" in observed
