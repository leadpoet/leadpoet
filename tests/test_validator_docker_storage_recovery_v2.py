import os
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]


def _write_executable(path: Path, source: str) -> None:
    path.write_text(source, encoding="utf-8")
    path.chmod(0o755)


def _run_recovery(
    tmp_path: Path,
    *,
    available: int,
    containers: int = 0,
    images: int = 0,
    volumes: int = 0,
    layerdb_images: int = 0,
    layerdb_mounts: int = 0,
    overlay_directories: int = 0,
    containerd_containers: int = 0,
    containerd_tasks: int = 0,
    containerd_running_tasks: int = 0,
    running_tasks_clear_after: int = -1,
    task_probe_failures: int = 0,
    system_prune_failures: int = 0,
    non_moby_namespaces: int = 0,
    moby_shims: int = 0,
    live_runtime_min_free_bytes: int = 18_000_000_000,
) -> tuple[subprocess.CompletedProcess[str], str]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    proc_root = tmp_path / "proc"
    proc_root.mkdir()
    sudo_log = tmp_path / "sudo.log"

    _write_executable(
        bin_dir / "df",
        """#!/bin/bash
printf 'Avail\\n%s\\n' "$FAKE_AVAILABLE"
""",
    )
    _write_executable(
        bin_dir / "docker",
        """#!/bin/bash
emit_rows() {
  local count="$1"
  local index=0
  while [ "$index" -lt "$count" ]; do
    printf 'row-%s\\n' "$index"
    index=$((index + 1))
  done
}
case "${1:-}:${2:-}" in
  info:)
    exit 0
    ;;
  image:prune|builder:prune)
    exit 0
    ;;
  system:prune)
    attempt=0
    if [ -f "$FAKE_SYSTEM_PRUNE_STATE" ]; then
      attempt="$(cat "$FAKE_SYSTEM_PRUNE_STATE")"
    fi
    attempt=$((attempt + 1))
    printf '%s\\n' "$attempt" > "$FAKE_SYSTEM_PRUNE_STATE"
    if [ "$attempt" -le "$FAKE_SYSTEM_PRUNE_FAILURES" ]; then
      exit 1
    fi
    exit 0
    ;;
  ps:-aq)
    emit_rows "$FAKE_CONTAINERS"
    exit 0
    ;;
  image:ls)
    emit_rows "$FAKE_IMAGES"
    exit 0
    ;;
  volume:ls)
    emit_rows "$FAKE_VOLUMES"
    exit 0
    ;;
  info:--format)
    printf '/var/lib/docker\\n'
    exit 0
    ;;
  info:)
    # Bare `docker info` availability/readiness probes (pre-inventory preamble
    # and the post-reset readiness loop) — the fake daemon is always healthy.
    exit 0
    ;;
esac
exit 2
""",
    )
    _write_executable(
        bin_dir / "findmnt",
        """#!/bin/bash
# No stale mounts under the fake docker/containerd roots (hermetic on CI and
# macOS, which lacks findmnt entirely).
exit 0
""",
    )
    _write_executable(
        bin_dir / "sleep",
        """#!/bin/bash
exit 0
""",
    )
    _write_executable(
        bin_dir / "flock",
        """#!/bin/bash
# Lock semantics are covered separately; this recovery fixture has one process.
exit 0
""",
    )
    _write_executable(
        bin_dir / "sudo",
        """#!/bin/bash
command="$1"
shift
emit_rows() {
  local count="$1"
  local index=0
  while [ "$index" -lt "$count" ]; do
    printf '/fake/%s\\n' "$index"
    index=$((index + 1))
  done
}
case "$command" in
  test)
    exit 0
    ;;
  du)
    printf '%s\\t/var/lib/docker\\n' "$FAKE_DOCKER_ROOT_BYTES"
    ;;
  find)
    case "$1" in
      */layerdb/sha256) emit_rows "$FAKE_LAYERDB_IMAGES" ;;
      */layerdb/mounts) emit_rows "$FAKE_LAYERDB_MOUNTS" ;;
      */overlay2) emit_rows "$FAKE_OVERLAY_DIRECTORIES" ;;
      *) exit 2 ;;
    esac
    ;;
  ctr)
    if [ -f "$FAKE_CONTAINERD_RESET_MARKER" ]; then
      exit 0
    fi
    if [ "${1:-}" = "namespaces" ]; then
      emit_rows "$FAKE_NON_MOBY_NAMESPACES"
      exit 0
    fi
    case "${3:-}:${4:-}" in
      containers:list)
        emit_rows "$FAKE_CONTAINERD_CONTAINERS"
        ;;
      tasks:list)
        if [ "${5:-}" = "-q" ]; then
          emit_rows "$FAKE_CONTAINERD_TASKS"
        else
          probe=0
          if [ -f "$FAKE_RUNNING_TASK_PROBE_STATE" ]; then
            probe="$(cat "$FAKE_RUNNING_TASK_PROBE_STATE")"
          fi
          probe=$((probe + 1))
          printf '%s\\n' "$probe" > "$FAKE_RUNNING_TASK_PROBE_STATE"
          if [ "$probe" -le "$FAKE_TASK_PROBE_FAILURES" ]; then
            exit 1
          fi
          printf 'TASK PID STATUS\\n'
          index=0
          running="$FAKE_CONTAINERD_RUNNING_TASKS"
          if [ "$FAKE_RUNNING_TASKS_CLEAR_AFTER" -ge 0 ] \
              && [ "$probe" -gt "$FAKE_RUNNING_TASKS_CLEAR_AFTER" ]; then
            running=0
          fi
          while [ "$index" -lt "$running" ]; do
            printf 'task-%s 100 RUNNING\\n' "$index"
            index=$((index + 1))
          done
        fi
        ;;
      *) exit 2 ;;
    esac
    ;;
  rm)
    printf '%s %s\\n' "$command" "$*" >> "$FAKE_SUDO_LOG"
    if [ "$*" = "-rf --one-file-system /var/lib/containerd" ]; then
      touch "$FAKE_CONTAINERD_RESET_MARKER"
    fi
    ;;
  systemctl|install|pkill)
    printf '%s %s\\n' "$command" "$*" >> "$FAKE_SUDO_LOG"
    ;;
  *)
    exit 2
    ;;
esac
""",
    )
    _write_executable(
        bin_dir / "pgrep",
        """#!/bin/bash
if [ -f "$FAKE_CONTAINERD_RESET_MARKER" ]; then
  printf '0\\n'
else
  printf '%s\\n' "$FAKE_MOBY_SHIMS"
fi
""",
    )

    env = {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "VALIDATOR_DOCKER_ALLOW_DATA_ROOT_RESET": "1",
        "VALIDATOR_DOCKER_MIN_FREE_BYTES": "30000000000",
        "VALIDATOR_DOCKER_LIVE_RUNTIME_MIN_FREE_BYTES": str(
            live_runtime_min_free_bytes
        ),
        "LEADPOET_DOCKER_OPERATION_LOCK_FILE": str(
            tmp_path / "docker-operation.lock"
        ),
        "LEADPOET_PROC_ROOT": str(proc_root),
        "FAKE_AVAILABLE": str(available),
        "FAKE_CONTAINERS": str(containers),
        "FAKE_IMAGES": str(images),
        "FAKE_VOLUMES": str(volumes),
        "FAKE_LAYERDB_IMAGES": str(layerdb_images),
        "FAKE_LAYERDB_MOUNTS": str(layerdb_mounts),
        "FAKE_OVERLAY_DIRECTORIES": str(overlay_directories),
        "FAKE_CONTAINERD_CONTAINERS": str(containerd_containers),
        "FAKE_CONTAINERD_TASKS": str(containerd_tasks),
        "FAKE_CONTAINERD_RUNNING_TASKS": str(containerd_running_tasks),
        "FAKE_RUNNING_TASKS_CLEAR_AFTER": str(running_tasks_clear_after),
        "FAKE_RUNNING_TASK_PROBE_STATE": str(tmp_path / "running-task-probes"),
        "FAKE_TASK_PROBE_FAILURES": str(task_probe_failures),
        "FAKE_SYSTEM_PRUNE_FAILURES": str(system_prune_failures),
        "FAKE_SYSTEM_PRUNE_STATE": str(tmp_path / "system-prune-attempts"),
        "FAKE_NON_MOBY_NAMESPACES": str(non_moby_namespaces),
        "FAKE_MOBY_SHIMS": str(moby_shims),
        "FAKE_CONTAINERD_RESET_MARKER": str(tmp_path / "containerd-reset"),
        "FAKE_DOCKER_ROOT_BYTES": "229720371200",
        "FAKE_SUDO_LOG": str(sudo_log),
        "VALIDATOR_DOCKER_PRUNE_ATTEMPTS": str(system_prune_failures + 1),
        "VALIDATOR_DOCKER_SETTLE_ATTEMPTS": str(
            max(
                1,
                task_probe_failures + 1,
                3 if running_tasks_clear_after >= 0 else 1,
            )
        ),
    }
    result = subprocess.run(
        ["bash", str(ROOT / "validator_tee/scripts/reclaim_docker_storage_v2.sh")],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    return result, sudo_log.read_text(encoding="utf-8") if sudo_log.exists() else ""


def test_validator_docker_recovery_is_guarded_and_runs_after_shutdown():
    recovery = (
        ROOT / "validator_tee" / "scripts" / "reclaim_docker_storage_v2.sh"
    ).read_text(encoding="utf-8")
    restart = (ROOT / "validator_restart.sh").read_text(encoding="utf-8")
    workflow = (
        ROOT / ".github" / "workflows" / "attested-v2-release.yml"
    ).read_text(encoding="utf-8")

    assert 'CONTAINER_COUNT="$(docker ps -aq' in recovery
    assert 'if [ "$CONTAINER_COUNT" -ne 0 ]' in recovery
    assert 'if [ "$IMAGE_COUNT" -ne 0 ]' in recovery
    assert 'if [ "$VOLUME_COUNT" -ne 0 ]' in recovery
    assert 'if [ "$DOCKER_ROOT" != "/var/lib/docker" ]' in recovery
    assert "ORPHANED_DOCKER_STATE=1" in recovery
    assert 'OVERLAY_DIRECTORY_COUNT="$(\n' in recovery
    assert "systemctl stop docker.service docker.socket containerd.service" in recovery
    assert 'rm -rf --one-file-system "$DOCKER_ROOT"' in recovery
    assert 'rm -rf --one-file-system "$CONTAINERD_ROOT"' in recovery
    assert "docker system prune --all --force --volumes" not in recovery

    remove_containers = restart.index("| xargs -r docker rm")
    journal_cleanup = restart.index(
        'sudo journalctl \\\n          --vacuum-size="$VALIDATOR_JOURNAL_VACUUM_SIZE"'
    )
    pip_cleanup = restart.index('rm -rf -- "$HOME/.cache/pip"')
    reclaim = restart.index("reclaim_docker_storage_v2.sh")
    build = restart.index("bash validator_tee/scripts/build_enclave.sh")
    assert remove_containers < journal_cleanup < pip_cleanup < reclaim < build
    cleanup = restart[journal_cleanup:reclaim]
    for protected_path in (
        ".bittensor",
        ".config/leadpoet",
        "leadpoet-legacy",
        "validator_weights",
        "validator-releases-v2",
        "actions-runner",
        "leadpoet-v2-artifacts",
        "drand-cabi-v2",
    ):
        assert protected_path not in cleanup
    assert "VALIDATOR_DOCKER_ALLOW_DATA_ROOT_RESET=1" in restart
    assert "VALIDATOR_DOCKER_ALLOW_DATA_ROOT_RESET=1" in workflow

    subprocess.run(
        [
            "bash",
            "-n",
            str(
                ROOT
                / "validator_tee"
                / "scripts"
                / "reclaim_docker_storage_v2.sh"
            ),
        ],
        check=True,
    )


def test_validator_docker_recovery_resets_orphaned_empty_root_above_floor(
    tmp_path: Path,
) -> None:
    result, sudo_log = _run_recovery(
        tmp_path,
        available=40_000_000_000,
        layerdb_images=3_904,
        layerdb_mounts=322,
        overlay_directories=4_559,
    )

    assert result.returncode == 0, result.stderr
    assert "orphaned=1" in result.stdout
    assert "systemctl stop docker.service docker.socket containerd.service" in sudo_log
    assert "rm -rf --one-file-system /var/lib/docker" in sudo_log
    assert "rm -rf --one-file-system /var/lib/containerd" in sudo_log


def test_validator_docker_recovery_resets_stale_containerd_state(
    tmp_path: Path,
) -> None:
    result, sudo_log = _run_recovery(
        tmp_path,
        available=40_000_000_000,
        containerd_containers=22,
        containerd_tasks=7,
        moby_shims=20,
    )

    assert result.returncode == 0, result.stderr
    assert "containerd_containers=22" in result.stdout
    assert "containerd_tasks=7" in result.stdout
    assert "moby_shims=20" in result.stdout
    assert "orphaned=1" in result.stdout
    assert "systemctl stop docker.service docker.socket containerd.service" in sudo_log


def test_validator_docker_recovery_refuses_running_containerd_task(
    tmp_path: Path,
) -> None:
    result, sudo_log = _run_recovery(
        tmp_path,
        available=40_000_000_000,
        containerd_containers=1,
        containerd_tasks=1,
        containerd_running_tasks=1,
    )

    assert result.returncode == 1
    assert "refusing containerd reset while 1 moby task(s) are running" in result.stderr
    assert "systemctl stop" not in sudo_log


def test_validator_docker_recovery_waits_for_transient_containerd_task(
    tmp_path: Path,
) -> None:
    result, sudo_log = _run_recovery(
        tmp_path,
        available=40_000_000_000,
        containerd_running_tasks=1,
        running_tasks_clear_after=1,
    )

    assert result.returncode == 0, result.stderr
    assert "Waiting for Docker teardown to settle" in result.stderr
    assert "systemctl stop" not in sudo_log


def test_validator_docker_recovery_retries_transient_system_prune(
    tmp_path: Path,
) -> None:
    result, sudo_log = _run_recovery(
        tmp_path,
        available=40_000_000_000,
        system_prune_failures=1,
    )

    assert result.returncode == 0, result.stderr
    assert "Docker system prune did not settle; retrying (1/2)" in result.stderr
    assert "systemctl stop" not in sudo_log


def test_validator_docker_recovery_retries_unreadable_teardown_state(
    tmp_path: Path,
) -> None:
    result, sudo_log = _run_recovery(
        tmp_path,
        available=40_000_000_000,
        task_probe_failures=1,
    )

    assert result.returncode == 0, result.stderr
    assert "Waiting for Docker teardown state to become readable" in result.stderr
    assert "systemctl stop" not in sudo_log


def test_validator_docker_recovery_preserves_healthy_live_runtime(
    tmp_path: Path,
) -> None:
    result, sudo_log = _run_recovery(
        tmp_path,
        available=40_000_000_000,
        containers=11,
        containerd_containers=11,
        containerd_tasks=11,
        containerd_running_tasks=11,
        moby_shims=11,
    )

    assert result.returncode == 0, result.stderr
    assert "containers=11" in result.stdout
    assert "containerd_running_tasks=11" in result.stdout
    assert "orphaned=0" in result.stdout
    assert "runtime_mode=live required_free_bytes=18000000000" in result.stdout
    assert "Waiting for Docker teardown" not in result.stderr
    assert "systemctl stop" not in sudo_log
    assert "rm -rf" not in sudo_log


def test_validator_docker_recovery_allows_one_recovery_build_after_failed_deploy(
    tmp_path: Path,
) -> None:
    result, sudo_log = _run_recovery(
        tmp_path,
        available=25_000_000_000,
        containers=11,
        containerd_containers=11,
        containerd_tasks=11,
        containerd_running_tasks=11,
        moby_shims=11,
        layerdb_images=3_889,
        layerdb_mounts=312,
        overlay_directories=4_514,
    )

    assert result.returncode == 0, result.stderr
    assert "runtime_mode=live required_free_bytes=18000000000" in result.stdout
    assert "Docker storage ready: free_bytes=25000000000" in result.stdout
    assert "systemctl stop" not in sudo_log
    assert "rm -rf" not in sudo_log


def test_validator_docker_recovery_refuses_live_runtime_below_recovery_reserve(
    tmp_path: Path,
) -> None:
    result, sudo_log = _run_recovery(
        tmp_path,
        available=17_999_999_999,
        containers=11,
        containerd_containers=11,
        containerd_tasks=11,
        containerd_running_tasks=11,
        moby_shims=11,
    )

    assert result.returncode == 1
    assert "live validator runtime has only 17999999999 free bytes" in result.stderr
    assert "refusing to stop containers or reset Docker storage" in result.stderr
    assert "systemctl stop" not in sudo_log
    assert "rm -rf" not in sudo_log


def test_validator_docker_recovery_keeps_strict_empty_runtime_floor(
    tmp_path: Path,
) -> None:
    result, sudo_log = _run_recovery(
        tmp_path,
        available=25_000_000_000,
    )

    assert result.returncode == 1
    assert "runtime_mode=empty required_free_bytes=30000000000" in result.stdout
    assert "Docker data-root reset left only 25000000000 free bytes" in result.stderr
    assert "systemctl stop docker.service docker.socket containerd.service" in sudo_log


def test_validator_docker_recovery_leaves_clean_root_above_floor_untouched(
    tmp_path: Path,
) -> None:
    result, sudo_log = _run_recovery(
        tmp_path,
        available=40_000_000_000,
    )

    assert result.returncode == 0, result.stderr
    assert "orphaned=0" in result.stdout
    assert "systemctl stop" not in sudo_log
    assert "rm -rf" not in sudo_log


def test_validator_docker_recovery_preserves_any_named_volume(
    tmp_path: Path,
) -> None:
    result, sudo_log = _run_recovery(
        tmp_path,
        available=20_000_000_000,
        volumes=1,
    )

    assert result.returncode == 1
    assert "refusing Docker data-root reset while 1 volume(s) remain" in result.stderr
    assert "systemctl stop" not in sudo_log
    assert "rm -rf" not in sudo_log
