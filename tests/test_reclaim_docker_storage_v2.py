from pathlib import Path
import subprocess
import sys


SCRIPT = Path("validator_tee/scripts/reclaim_docker_storage_v2.sh").read_text()


def test_builder_daemons_recover_before_any_docker_inventory():
    recovery = SCRIPT.index("if ! docker info >/dev/null 2>&1; then")
    start = SCRIPT.index("sudo systemctl start containerd.service docker.service")
    inventory = SCRIPT.index("docker image prune --all --force")

    assert recovery < start < inventory
    assert "Docker/containerd did not recover before storage inventory" in SCRIPT


def test_data_root_reset_unmounts_only_after_empty_runtime_guards():
    guard = SCRIPT.index('if [ "$NON_MOBY_NAMESPACE_COUNT" -ne 0 ]')
    stop = SCRIPT.index("sudo systemctl stop docker.service")
    mounts = SCRIPT.index("while IFS= read -r mount_target")
    remove = SCRIPT.index('sudo rm -rf --one-file-system "$DOCKER_ROOT"')

    assert guard < stop < mounts < remove
    assert 'docker_root="$DOCKER_ROOT/"' in SCRIPT
    assert 'containerd_root="$CONTAINERD_ROOT/"' in SCRIPT
    assert 'index($0, docker_root) == 1' in SCRIPT
    assert 'index($0, containerd_root) == 1' in SCRIPT


def test_data_root_reset_never_uses_forced_or_lazy_unmount():
    assert 'sudo umount "$mount_target"' in SCRIPT
    assert "umount -f" not in SCRIPT
    assert "umount -l" not in SCRIPT
    assert "umount --force" not in SCRIPT
    assert "umount --lazy" not in SCRIPT


def test_data_root_reset_fails_if_mounts_remain():
    check = SCRIPT.index(
        "ERROR: stale Docker/containerd mounts remain after guarded unmount"
    )
    remove = SCRIPT.index('sudo rm -rf --one-file-system "$DOCKER_ROOT"')

    assert check < remove


def test_live_runtime_reclaims_stale_state_before_rechecking_capacity():
    initial_inventory = SCRIPT.index('INITIAL_CONTAINER_COUNT="$(')
    reclaim = SCRIPT.index("docker_stale_mount_reclaimer_v2")
    repeated_prune = SCRIPT.index(
        "run_prune_with_retry image docker image prune --all --force",
        reclaim,
    )
    capacity = SCRIPT.index('AVAILABLE="$(available_bytes)"')

    assert initial_inventory < reclaim < repeated_prune < capacity
    assert 'if [ "$INITIAL_CONTAINER_COUNT" -ne 0 ]' in SCRIPT
    assert "Reclaiming unreachable Docker overlay state" in SCRIPT
    assert "sudo env PYTHONSAFEPATH=1 python3" in SCRIPT
    assert (
        '"$REPO_ROOT/validator_tee/host/docker_stale_mount_reclaimer_v2.py"'
        in SCRIPT
    )
    assert (
        '"$REPO_ROOT/validator_tee/host/docker_live_restore_reconciler_v2.py"'
        in SCRIPT
    )
    assert 'docker info --format \'{{json .LiveRestoreEnabled}}\'' in SCRIPT
    assert 'if [ "$RAW_RECLAIM_PERFORMED" -eq 1 ] \\' in SCRIPT
    assert '|| [ "$LIVE_RESTORE_ENABLED" != "true" ]; then' in SCRIPT
    assert "-m validator_tee.host.docker_stale_mount_reclaimer_v2" not in SCRIPT
    assert "-m validator_tee.host.docker_live_restore_reconciler_v2" not in SCRIPT


def test_live_reclaimer_bootstraps_without_validator_package_dependencies():
    helper = Path("validator_tee/host/docker_stale_mount_reclaimer_v2.py").resolve()
    result = subprocess.run(
        [sys.executable, "-I", str(helper), "--help"],
        cwd="/",
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "Reclaim unreachable Docker overlay state" in result.stdout


def test_live_restore_reconciler_bootstraps_without_validator_dependencies():
    helper = Path(
        "validator_tee/host/docker_live_restore_reconciler_v2.py"
    ).resolve()
    result = subprocess.run(
        [sys.executable, "-I", str(helper), "--help"],
        cwd="/",
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "Refresh Docker metadata" in result.stdout
