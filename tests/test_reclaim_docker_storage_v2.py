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


def test_live_host_gateway_online_lane_is_terminal_and_preserves_images():
    inventory = SCRIPT.index("inventory_empty_online_runtime()")
    lane = SCRIPT.index('if [ "$HOST_GATEWAY_LIVE" -eq 1 ]; then', inventory)
    offline_image_prune = SCRIPT.index(
        "run_prune_with_retry image docker image prune --all --force",
        lane,
    )
    source = SCRIPT[lane:offline_image_prune]
    reconcile_function = SCRIPT[
        SCRIPT.index("reconcile_empty_docker_runtime()") : SCRIPT.index(
            "docker_daemons_ready()"
        )
    ]

    assert "run_prune_with_retry builder docker builder prune --all --force" in source
    assert "docker_stale_mount_reclaimer_v2.py" in source
    assert "reconcile_empty_docker_runtime" in source
    assert "docker_zero_runtime_reconciler_v2.py" in reconcile_function
    assert "require_same_online_gateway" in source
    assert "require_same_online_images" in source
    assert "online_overlay_metadata_layout" in source
    assert "online_fully_empty_stale_audit_manifest" in source
    assert 'inventory_empty_online_runtime "pre-stale-reclaim"' in source
    assert 'inventory_empty_online_runtime "post-reclaim"' in source
    assert "docker image prune" not in source
    assert "docker system prune" not in source
    assert "docker_live_restore_reconciler_v2.py" not in source
    assert source.count("--audit-only") == 4
    assert "--docker-lock-file" in reconcile_function
    assert "--docker-admission-lock-file" in reconcile_function
    assert "--docker-lock-owner-pid" in reconcile_function
    assert "systemctl" not in source
    assert "pkill" not in source
    assert "rm -rf" not in source
    assert "exit 0" in source


def test_required_zero_runtime_reconcile_is_strict_and_builder_first():
    validation = SCRIPT.index(
        "ERROR: REQUIRE_ZERO_RUNTIME_RECONCILE must be 0 or 1"
    )
    acquire = SCRIPT.index("leadpoet_acquire_docker_operation_lock_v2")
    lane = SCRIPT.index(
        'if [ "$HOST_GATEWAY_LIVE" -eq 1 ]; then',
        SCRIPT.index("inventory_empty_online_runtime()"),
    )
    builder_prune = SCRIPT.index(
        "run_prune_with_retry builder docker builder prune --all --force", lane
    )
    reconcile_trigger = SCRIPT.index(
        '|| [ "$REQUIRE_ZERO_RUNTIME_RECONCILE" = "1" ]', builder_prune
    )
    pre_audit = SCRIPT.index("--audit-only", reconcile_trigger)
    raw_apply = SCRIPT.index("if ! ONLINE_RECLAIM_RESULT", pre_audit)
    post_apply_audit = SCRIPT.index("--audit-only", raw_apply)
    helper = SCRIPT.index("reconcile_empty_docker_runtime", post_apply_audit)
    post_helper_builder_prune = SCRIPT.index(
        "run_prune_with_retry builder docker builder prune --all --force", helper
    )
    final_audit = SCRIPT.index("--audit-only", post_helper_builder_prune)

    assert validation < acquire < lane < builder_prune < reconcile_trigger
    assert reconcile_trigger < pre_audit < raw_apply < post_apply_audit < helper
    assert helper < post_helper_builder_prune
    assert post_helper_builder_prune < final_audit
    assert (
        '[ "$REQUIRE_ZERO_RUNTIME_RECONCILE" != "1" ]'
        in SCRIPT[SCRIPT.index("protect_exact_host_gateway_runtime()"):lane]
    )


def test_required_absent_gateway_reconciliation_uses_the_offline_empty_root_lane():
    offline_image_prune = SCRIPT.index(
        "run_prune_with_retry image docker image prune --all --force"
    )
    ready = SCRIPT.index(
        'if [ "$AVAILABLE" -ge "$REQUIRED_FREE_BYTES" ]', offline_image_prune
    )
    reset = SCRIPT.index('protect_exact_host_gateway_runtime "pre-reset"', ready)
    source = SCRIPT[ready:reset]

    assert 'if [ "$REQUIRE_ZERO_RUNTIME_RECONCILE" = "1" ]' in source
    assert 'inventory_empty_online_runtime "pre-absent-daemon-reconcile"' in source
    assert 'require_exact_host_gateway_absent "pre-absent-daemon-reconcile-apply"' in source
    assert "reconcile_empty_docker_runtime" in source
    assert 'inventory_empty_online_runtime "post-absent-daemon-reconcile"' in source
    assert 'require_exact_host_gateway_absent "post-absent-daemon-reconcile"' in source
    assert "docker_stale_mount_reclaimer_v2.py" not in source


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
    reclaim = SCRIPT.index("docker_stale_mount_reclaimer_v2", initial_inventory)
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


def test_zero_runtime_reconciler_bootstraps_without_validator_dependencies():
    helper = Path(
        "validator_tee/host/docker_zero_runtime_reconciler_v2.py"
    ).resolve()
    result = subprocess.run(
        [sys.executable, "-I", str(helper), "--help"],
        cwd="/",
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "empty-runtime guard" in result.stdout
