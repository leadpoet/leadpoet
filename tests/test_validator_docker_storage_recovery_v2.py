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
) -> tuple[subprocess.CompletedProcess[str], str]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
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
  image:prune|builder:prune|system:prune)
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
esac
exit 2
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
  systemctl|rm|install)
    printf '%s %s\\n' "$command" "$*" >> "$FAKE_SUDO_LOG"
    ;;
  *)
    exit 2
    ;;
esac
""",
    )

    env = {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "VALIDATOR_DOCKER_ALLOW_DATA_ROOT_RESET": "1",
        "VALIDATOR_DOCKER_MIN_FREE_BYTES": "30000000000",
        "FAKE_AVAILABLE": str(available),
        "FAKE_CONTAINERS": str(containers),
        "FAKE_IMAGES": str(images),
        "FAKE_VOLUMES": str(volumes),
        "FAKE_LAYERDB_IMAGES": str(layerdb_images),
        "FAKE_LAYERDB_MOUNTS": str(layerdb_mounts),
        "FAKE_OVERLAY_DIRECTORIES": str(overlay_directories),
        "FAKE_DOCKER_ROOT_BYTES": "229720371200",
        "FAKE_SUDO_LOG": str(sudo_log),
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
    assert "systemctl stop docker.service docker.socket" in recovery
    assert 'rm -rf --one-file-system "$DOCKER_ROOT"' in recovery
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
    assert "systemctl stop docker.service docker.socket" in sudo_log
    assert "rm -rf --one-file-system /var/lib/docker" in sudo_log


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
