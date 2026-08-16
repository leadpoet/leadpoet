from pathlib import Path
import subprocess
import sys


SCRIPT = Path("validator_tee/scripts/reclaim_docker_storage_v2.sh").read_text()


def test_builder_daemons_recover_before_any_docker_inventory():
    recovery = SCRIPT.index("if ! docker_daemons_ready; then")
    start = SCRIPT.index("sudo systemctl start containerd.service docker.service")
    inventory = SCRIPT.index("docker image prune --all --force")

    assert recovery < start < inventory
    assert "Docker/containerd did not recover before storage inventory" in SCRIPT
    assert "docker_daemons_ready" in SCRIPT
    assert "timeout=timeout_seconds" in SCRIPT


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


def test_exact_host_gateway_guard_precedes_any_data_root_reset():
    acquire = SCRIPT.index("leadpoet_acquire_docker_operation_lock_v2")
    detect = SCRIPT.index("--detect-exact-host-gateway")
    defer = SCRIPT.index(
        "Docker storage maintenance deferred while the exact host gateway is live"
    )
    prune = SCRIPT.index("docker image prune --all --force")
    stop = SCRIPT.index("sudo systemctl stop docker.service")

    assert acquire < detect < defer < prune < stop
    assert "runtime_mode=host-gateway-live" in SCRIPT[detect:prune]
    assert "pgrep -f.*gateway.main" not in SCRIPT


def test_exact_host_gateway_guard_is_rechecked_at_the_stop_boundary():
    prune = SCRIPT.index("docker image prune --all --force")
    final_detect = SCRIPT.rindex('protect_exact_host_gateway_runtime "pre-reset"')
    arm = SCRIPT.index("trap recover_docker_daemons_on_exit EXIT")
    stop = SCRIPT.index("sudo systemctl stop docker.service")

    assert prune < final_detect < arm < stop
    assert SCRIPT.count("protect_exact_host_gateway_runtime") == 3


def test_data_root_reset_failure_recovers_daemons_before_exit():
    recovery = SCRIPT.index("recover_docker_daemons_on_exit()")
    arm = SCRIPT.index("trap recover_docker_daemons_on_exit EXIT")
    stop = SCRIPT.index("sudo systemctl stop docker.service")
    ready = SCRIPT.index("if ! start_docker_daemons_and_wait; then", stop)
    disarm = SCRIPT.index("trap - EXIT", ready)

    assert recovery < arm < stop < ready < disarm
    assert "DOCKER_RESET_STARTED=1" in SCRIPT[arm - 80 : arm]
    assert "Recovering Docker/containerd readiness after failed data-root reset" in SCRIPT


def test_live_runtime_reclaims_stale_state_before_rechecking_capacity():
    initial_inventory = SCRIPT.index('INITIAL_CONTAINER_COUNT="$(')
    reclaim = SCRIPT.index("docker_stale_mount_reclaimer_v2")
    repeated_prune = SCRIPT.index(
        "run_prune_with_retry image docker image prune --all --force",
        reclaim,
    )
    capacity = SCRIPT.index('AVAILABLE="$(available_bytes)"', repeated_prune)

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
