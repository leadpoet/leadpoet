from __future__ import annotations

import subprocess
from typing import Sequence

import pytest

from validator_tee.host.docker_stale_mount_reclaimer_v2 import (
    DockerStaleMountReclaimerV2Error,
    reclaim_stale_docker_overlay_mounts_v2,
)


def _target(index: int) -> str:
    return f"/var/lib/docker/overlay2/{index:064x}/merged"


class _FakeRunner:
    def __init__(
        self,
        *,
        active: set[str],
        mounted: set[str],
        docker_root: str = "/var/lib/docker",
        unmount_failure: str | None = None,
        retain_after_unmount: bool = False,
    ) -> None:
        self.active = set(active)
        self.mounted = set(mounted)
        self.docker_root = docker_root
        self.unmount_failure = unmount_failure
        self.retain_after_unmount = retain_after_unmount
        self.unmounted: list[str] = []

    def __call__(self, command: Sequence[str]) -> subprocess.CompletedProcess[str]:
        argv = list(command)
        if argv == ["docker", "info", "--format", "{{.DockerRootDir}}"]:
            return subprocess.CompletedProcess(argv, 0, self.docker_root + "\n", "")
        if argv == ["docker", "ps", "-aq"]:
            ids = [f"container-{index}" for index, _ in enumerate(sorted(self.active))]
            return subprocess.CompletedProcess(argv, 0, "\n".join(ids) + "\n", "")
        if argv[:3] == ["docker", "inspect", "--format"]:
            return subprocess.CompletedProcess(
                argv,
                0,
                "\n".join(sorted(self.active)) + "\n",
                "",
            )
        if argv == ["findmnt", "-rn", "-t", "overlay", "-o", "TARGET"]:
            return subprocess.CompletedProcess(
                argv,
                0,
                "\n".join(sorted(self.mounted)) + "\n",
                "",
            )
        if argv[:3] == ["sudo", "umount", "--"]:
            target = argv[3]
            if target == self.unmount_failure:
                return subprocess.CompletedProcess(argv, 1, "", "target is busy")
            self.unmounted.append(target)
            if not self.retain_after_unmount:
                self.mounted.remove(target)
            return subprocess.CompletedProcess(argv, 0, "", "")
        raise AssertionError(f"unexpected command: {argv}")


def test_reclaims_only_mounts_absent_from_docker_inventory() -> None:
    active = {_target(1), _target(2)}
    stale = {_target(101), _target(102)}
    runner = _FakeRunner(active=active, mounted=active | stale)

    result = reclaim_stale_docker_overlay_mounts_v2(runner=runner)

    assert runner.unmounted == sorted(stale)
    assert runner.mounted == active
    assert result.active_mount_count == 2
    assert result.mounted_overlay_count == 4
    assert result.reclaimed_mount_count == 2


def test_refuses_unexpected_docker_root() -> None:
    runner = _FakeRunner(active=set(), mounted=set(), docker_root="/srv/docker")

    with pytest.raises(
        DockerStaleMountReclaimerV2Error,
        match="refusing unexpected Docker root",
    ):
        reclaim_stale_docker_overlay_mounts_v2(runner=runner)


def test_refuses_malformed_mount_inside_docker_overlay_root() -> None:
    malformed = "/var/lib/docker/overlay2/not-a-layer/merged"
    runner = _FakeRunner(active=set(), mounted={malformed})

    with pytest.raises(
        DockerStaleMountReclaimerV2Error,
        match="refusing malformed Docker overlay mount target",
    ):
        reclaim_stale_docker_overlay_mounts_v2(runner=runner)


def test_fails_closed_when_stale_mount_is_busy() -> None:
    stale = _target(103)
    runner = _FakeRunner(
        active={_target(1)},
        mounted={_target(1), stale},
        unmount_failure=stale,
    )

    with pytest.raises(
        DockerStaleMountReclaimerV2Error,
        match="stale Docker overlay unmount failed",
    ):
        reclaim_stale_docker_overlay_mounts_v2(runner=runner)


def test_fails_closed_when_mount_remains_after_successful_command() -> None:
    stale = _target(104)
    runner = _FakeRunner(
        active={_target(1)},
        mounted={_target(1), stale},
        retain_after_unmount=True,
    )

    with pytest.raises(
        DockerStaleMountReclaimerV2Error,
        match="stale Docker overlay mounts remain",
    ):
        reclaim_stale_docker_overlay_mounts_v2(runner=runner)
