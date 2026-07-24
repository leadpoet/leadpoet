from pathlib import Path


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
