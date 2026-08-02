"""Reclaim stale Docker overlay mounts without disturbing live containers."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import re
import subprocess
from typing import Callable, Optional, Sequence


class DockerStaleMountReclaimerV2Error(RuntimeError):
    """Raised when stale Docker mounts cannot be reclaimed safely."""


CommandRunner = Callable[[Sequence[str]], subprocess.CompletedProcess[str]]


@dataclass(frozen=True)
class DockerMountReclaimResult:
    docker_root: str
    active_mount_count: int
    mounted_overlay_count: int
    reclaimed_mount_count: int

    def as_dict(self) -> dict[str, object]:
        return {
            "active_mount_count": self.active_mount_count,
            "docker_root": self.docker_root,
            "mounted_overlay_count": self.mounted_overlay_count,
            "reclaimed_mount_count": self.reclaimed_mount_count,
            "schema_version": "leadpoet.docker_stale_mount_reclaim.v2",
            "status": "ready",
        }


def _run(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(command),
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )


def _checked_output(
    runner: CommandRunner,
    command: Sequence[str],
    *,
    label: str,
) -> str:
    result = runner(command)
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        raise DockerStaleMountReclaimerV2Error(
            f"{label} failed" + (f": {detail}" if detail else "")
        )
    return result.stdout


def _docker_root(runner: CommandRunner) -> Path:
    raw_root = _checked_output(
        runner,
        ["docker", "info", "--format", "{{.DockerRootDir}}"],
        label="Docker root discovery",
    ).strip()
    root = Path(raw_root)
    if not root.is_absolute() or root != Path("/var/lib/docker"):
        raise DockerStaleMountReclaimerV2Error(
            f"refusing unexpected Docker root: {raw_root or '<empty>'}"
        )
    return root


def _overlay_target_pattern(docker_root: Path) -> re.Pattern[str]:
    return re.compile(
        rf"^{re.escape(str(docker_root))}/overlay2/[0-9a-f]{{64}}/merged$"
    )


def _validate_overlay_target(
    target: str,
    *,
    docker_root: Path,
) -> str:
    if not _overlay_target_pattern(docker_root).fullmatch(target):
        raise DockerStaleMountReclaimerV2Error(
            f"refusing malformed Docker overlay mount target: {target}"
        )
    return target


def _container_merged_dirs(
    runner: CommandRunner,
    *,
    docker_root: Path,
) -> set[str]:
    container_ids = [
        value.strip()
        for value in _checked_output(
            runner,
            ["docker", "ps", "-aq"],
            label="Docker container inventory",
        ).splitlines()
        if value.strip()
    ]
    if not container_ids:
        return set()
    output = _checked_output(
        runner,
        [
            "docker",
            "inspect",
            "--format",
            "{{.GraphDriver.Data.MergedDir}}",
            *container_ids,
        ],
        label="Docker container mount inventory",
    )
    merged_dirs = {value.strip() for value in output.splitlines() if value.strip()}
    if len(merged_dirs) != len(container_ids):
        raise DockerStaleMountReclaimerV2Error(
            "Docker container mount inventory is incomplete"
        )
    return {
        _validate_overlay_target(target, docker_root=docker_root)
        for target in merged_dirs
    }


def _mounted_overlay_dirs(
    runner: CommandRunner,
    *,
    docker_root: Path,
) -> set[str]:
    output = _checked_output(
        runner,
        ["findmnt", "-rn", "-t", "overlay", "-o", "TARGET"],
        label="mounted overlay inventory",
    )
    prefix = f"{docker_root}/overlay2/"
    candidates = {
        value.strip()
        for value in output.splitlines()
        if value.strip().startswith(prefix) and value.strip().endswith("/merged")
    }
    return {
        _validate_overlay_target(target, docker_root=docker_root)
        for target in candidates
    }


def reclaim_stale_docker_overlay_mounts_v2(
    *,
    runner: CommandRunner = _run,
) -> DockerMountReclaimResult:
    """Unmount overlay targets no longer owned by any Docker container."""

    docker_root = _docker_root(runner)
    active_mounts = _container_merged_dirs(runner, docker_root=docker_root)
    mounted_overlays = _mounted_overlay_dirs(runner, docker_root=docker_root)
    stale_mounts = sorted(mounted_overlays - active_mounts)

    for target in stale_mounts:
        result = runner(["sudo", "umount", "--", target])
        if result.returncode != 0:
            detail = (result.stderr or result.stdout or "").strip()
            raise DockerStaleMountReclaimerV2Error(
                f"stale Docker overlay unmount failed for {target}"
                + (f": {detail}" if detail else "")
            )

    remaining = _mounted_overlay_dirs(runner, docker_root=docker_root) - active_mounts
    if remaining:
        raise DockerStaleMountReclaimerV2Error(
            "stale Docker overlay mounts remain after guarded reclaim: "
            + ",".join(sorted(remaining))
        )

    return DockerMountReclaimResult(
        docker_root=str(docker_root),
        active_mount_count=len(active_mounts),
        mounted_overlay_count=len(mounted_overlays),
        reclaimed_mount_count=len(stale_mounts),
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)
    try:
        result = reclaim_stale_docker_overlay_mounts_v2()
    except DockerStaleMountReclaimerV2Error as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(result.as_dict(), sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
