from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
from types import SimpleNamespace
from typing import Sequence

import pytest

from validator_tee.host.docker_stale_mount_reclaimer_v2 import (
    DockerStaleMountReclaimerV2Error,
    audit_stale_docker_overlay_state_v2,
    reclaim_stale_docker_overlay_mounts_v2,
)


def _digest(seed: str) -> str:
    return "sha256:" + hashlib.sha256(seed.encode("utf-8")).hexdigest()


def _chain_id(parent: str, diff_id: str) -> str:
    return "sha256:" + hashlib.sha256(
        f"{parent} {diff_id}".encode("ascii")
    ).hexdigest()


class _DockerFixture:
    def __init__(
        self,
        root: Path,
        *,
        build_cache_count: int = 0,
        duplicate_image_id: bool = False,
        unmount_failure: bool = False,
        retain_after_unmount: bool = False,
    ) -> None:
        self.root = root
        self.build_cache_count = build_cache_count
        self.duplicate_image_id = duplicate_image_id
        self.unmount_failure = unmount_failure
        self.retain_after_unmount = retain_after_unmount
        self.unmounted: list[str] = []

        self.image_id = _digest("active-image")
        self.container_id = hashlib.sha256(b"active-container").hexdigest()
        self.stale_container_id = hashlib.sha256(b"stale-container").hexdigest()
        self.diff_one = _digest("active-diff-one")
        self.diff_two = _digest("active-diff-two")
        self.chain_one = self.diff_one
        self.chain_two = _chain_id(self.chain_one, self.diff_two)
        self.stale_chain = _digest("stale-chain")
        self.active_cache_one = "activecacheone00000000001"
        self.active_cache_two = "activecachetwo00000000002"
        self.stale_cache = "stalecachelayer0000000003"
        self.active_mount = hashlib.sha256(b"active-mount").hexdigest()
        self.active_init = self.active_mount + "-init"
        self.stale_mount = hashlib.sha256(b"stale-mount").hexdigest()
        self.stale_init = self.stale_mount + "-init"
        self.orphan_overlay = "orphanoverlay000000000004"
        self.links = {
            self.active_cache_one: "AAAAAAAAAAAAAAAAAAAAAAAAAA",
            self.active_cache_two: "BBBBBBBBBBBBBBBBBBBBBBBBBB",
            self.active_init: "CCCCCCCCCCCCCCCCCCCCCCCCCC",
            self.active_mount: "DDDDDDDDDDDDDDDDDDDDDDDDDD",
            self.stale_cache: "EEEEEEEEEEEEEEEEEEEEEEEEEE",
            self.stale_init: "FFFFFFFFFFFFFFFFFFFFFFFFFF",
            self.stale_mount: "GGGGGGGGGGGGGGGGGGGGGGGGGG",
            self.orphan_overlay: "HHHHHHHHHHHHHHHHHHHHHHHHHH",
        }
        self._build_filesystem()
        self.mounted = {
            str(self.root / "overlay2" / self.active_mount / "merged"),
            str(self.root / "overlay2" / self.stale_mount / "merged"),
        }

    @property
    def image_doc(self) -> dict[str, object]:
        return {
            "Id": self.image_id,
            "RootFS": {"Layers": [self.diff_one, self.diff_two]},
        }

    @property
    def container_doc(self) -> dict[str, object]:
        return {
            "Id": self.container_id,
            "Image": self.image_id,
            "GraphDriver": {
                "Data": {
                    "MergedDir": str(
                        self.root / "overlay2" / self.active_mount / "merged"
                    )
                }
            },
        }

    def _write_layer(
        self,
        chain_id: str,
        cache_id: str,
        diff_id: str,
        *,
        parent: str | None = None,
    ) -> None:
        path = self.root / "image/overlay2/layerdb/sha256" / chain_id.split(":", 1)[1]
        path.mkdir(parents=True)
        (path / "cache-id").write_text(cache_id, encoding="utf-8")
        (path / "diff").write_text(diff_id, encoding="utf-8")
        (path / "size").write_text("1", encoding="utf-8")
        if parent is not None:
            (path / "parent").write_text(parent, encoding="utf-8")

    def _write_mount(
        self,
        container_id: str,
        mount_id: str,
        init_id: str,
        parent: str,
    ) -> None:
        path = self.root / "image/overlay2/layerdb/mounts" / container_id
        path.mkdir(parents=True)
        (path / "mount-id").write_text(mount_id, encoding="utf-8")
        (path / "init-id").write_text(init_id, encoding="utf-8")
        (path / "parent").write_text(parent, encoding="utf-8")

    def _write_overlay(self, overlay_id: str, *, lower: list[str]) -> None:
        path = self.root / "overlay2" / overlay_id
        (path / "diff").mkdir(parents=True)
        (path / "work").mkdir()
        (path / "merged").mkdir()
        link_id = self.links[overlay_id]
        (path / "link").write_text(link_id, encoding="utf-8")
        if lower:
            (path / "lower").write_text(
                ":".join(f"l/{self.links[item]}" for item in lower),
                encoding="utf-8",
            )
        (self.root / "overlay2/l" / link_id).symlink_to(
            f"../{overlay_id}/diff"
        )

    def _build_filesystem(self) -> None:
        (self.root / "image/overlay2/layerdb/sha256").mkdir(parents=True)
        (self.root / "image/overlay2/layerdb/mounts").mkdir(parents=True)
        (self.root / "overlay2/l").mkdir(parents=True)
        self._write_layer(self.chain_one, self.active_cache_one, self.diff_one)
        self._write_layer(
            self.chain_two,
            self.active_cache_two,
            self.diff_two,
            parent=self.chain_one,
        )
        self._write_layer(
            self.stale_chain,
            self.stale_cache,
            _digest("stale-diff"),
            parent=self.chain_two,
        )
        self._write_mount(
            self.container_id,
            self.active_mount,
            self.active_init,
            self.chain_two,
        )
        self._write_mount(
            self.stale_container_id,
            self.stale_mount,
            self.stale_init,
            self.stale_chain,
        )
        self._write_overlay(self.active_cache_one, lower=[])
        self._write_overlay(self.active_cache_two, lower=[self.active_cache_one])
        self._write_overlay(self.active_init, lower=[self.active_cache_two, self.active_cache_one])
        self._write_overlay(
            self.active_mount,
            lower=[self.active_init, self.active_cache_two, self.active_cache_one],
        )
        self._write_overlay(self.stale_cache, lower=[self.active_cache_one])
        self._write_overlay(self.stale_init, lower=[self.stale_cache])
        self._write_overlay(self.stale_mount, lower=[self.stale_init, self.stale_cache])
        self._write_overlay(self.orphan_overlay, lower=[self.active_cache_one])

    def install_backing_device_fixture(
        self,
        monkeypatch: pytest.MonkeyPatch,
        *,
        mode: int = 0o600,
    ) -> None:
        path = self.root / "overlay2/backingFsBlockDev"
        path.touch()
        original_lstat = Path.lstat

        def fake_lstat(candidate: Path):
            if candidate == path:
                return SimpleNamespace(
                    st_mode=stat.S_IFBLK | mode,
                    st_uid=0,
                    st_gid=0,
                    st_rdev=os.makedev(259, 1),
                    st_size=0,
                )
            return original_lstat(candidate)

        monkeypatch.setattr(Path, "lstat", fake_lstat)

    def __call__(self, command: Sequence[str]) -> subprocess.CompletedProcess[str]:
        argv = list(command)
        if argv == ["docker", "info", "--format", "{{.DockerRootDir}}"]:
            return subprocess.CompletedProcess(argv, 0, str(self.root) + "\n", "")
        if argv == ["docker", "images", "-aq", "--no-trunc"]:
            count = 2 if self.duplicate_image_id else 1
            return subprocess.CompletedProcess(
                argv,
                0,
                (self.image_id + "\n") * count,
                "",
            )
        if argv == ["docker", "ps", "-aq", "--no-trunc"]:
            return subprocess.CompletedProcess(argv, 0, self.container_id + "\n", "")
        if argv[:3] == ["docker", "image", "inspect"]:
            return subprocess.CompletedProcess(argv, 0, json.dumps([self.image_doc]), "")
        if argv[:3] == ["docker", "container", "inspect"]:
            return subprocess.CompletedProcess(
                argv,
                0,
                json.dumps([self.container_doc]),
                "",
            )
        if argv == ["docker", "system", "df", "--format", "{{json .}}"]:
            rows = [
                {"Type": "Images", "TotalCount": "1", "Active": "1"},
                {
                    "Type": "Build Cache",
                    "TotalCount": str(self.build_cache_count),
                    "Active": str(self.build_cache_count),
                },
            ]
            return subprocess.CompletedProcess(
                argv,
                0,
                "\n".join(json.dumps(row) for row in rows) + "\n",
                "",
            )
        if argv == ["findmnt", "-rn", "-t", "overlay", "-o", "TARGET"]:
            return subprocess.CompletedProcess(
                argv,
                0,
                "\n".join(sorted(self.mounted)) + "\n",
                "",
            )
        if argv == ["findmnt", "-rn", "-o", "TARGET"]:
            return subprocess.CompletedProcess(
                argv,
                0,
                "\n".join(sorted(self.mounted)) + "\n",
                "",
            )
        if argv[:2] == ["umount", "--"]:
            target = argv[2]
            if self.unmount_failure:
                return subprocess.CompletedProcess(argv, 1, "", "target is busy")
            self.unmounted.append(target)
            if not self.retain_after_unmount:
                self.mounted.remove(target)
            return subprocess.CompletedProcess(argv, 0, "", "")
        raise AssertionError(f"unexpected command: {argv}")


def test_reclaims_only_unreachable_docker_state(tmp_path: Path) -> None:
    fixture = _DockerFixture(tmp_path / "docker")

    result = reclaim_stale_docker_overlay_mounts_v2(
        runner=fixture,
        expected_root=fixture.root,
    )

    assert fixture.unmounted == [
        str(fixture.root / "overlay2" / fixture.stale_mount / "merged")
    ]
    assert result.active_container_count == 1
    assert result.active_image_count == 1
    assert result.active_layer_count == 2
    assert result.reclaimed_layer_record_count == 1
    assert result.reclaimed_mount_record_count == 1
    assert result.reclaimed_overlay_dir_count == 4
    assert result.reclaimed_overlay_link_count == 4
    for overlay_id in (
        fixture.active_cache_one,
        fixture.active_cache_two,
        fixture.active_init,
        fixture.active_mount,
    ):
        assert (fixture.root / "overlay2" / overlay_id).is_dir()
    for overlay_id in (
        fixture.stale_cache,
        fixture.stale_init,
        fixture.stale_mount,
        fixture.orphan_overlay,
    ):
        assert not (fixture.root / "overlay2" / overlay_id).exists()


def test_audit_reports_exact_partition_without_mutation(tmp_path: Path) -> None:
    fixture = _DockerFixture(tmp_path / "docker")

    report = audit_stale_docker_overlay_state_v2(
        runner=fixture,
        expected_root=fixture.root,
    )

    assert report["active_layer_count"] == 2
    assert report["active_overlay_dir_count"] == 4
    assert report["stale_layer_record_count"] == 1
    assert report["stale_mount_record_count"] == 1
    assert report["stale_overlay_dir_count"] == 4
    assert fixture.unmounted == []
    assert (fixture.root / "overlay2" / fixture.stale_cache).is_dir()


def test_accepts_exact_docker_backing_device_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _DockerFixture(tmp_path / "docker")
    fixture.install_backing_device_fixture(monkeypatch)

    report = audit_stale_docker_overlay_state_v2(
        runner=fixture,
        expected_root=fixture.root,
    )

    assert report["status"] == "ready"


def test_deduplicates_image_ids_emitted_for_multiple_tags(tmp_path: Path) -> None:
    fixture = _DockerFixture(tmp_path / "docker", duplicate_image_id=True)

    report = audit_stale_docker_overlay_state_v2(
        runner=fixture,
        expected_root=fixture.root,
    )

    assert report["active_image_count"] == 1


def test_refuses_backing_device_with_unsafe_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _DockerFixture(tmp_path / "docker")
    fixture.install_backing_device_fixture(monkeypatch, mode=0o666)

    with pytest.raises(
        DockerStaleMountReclaimerV2Error,
        match="unsafe ownership or mode",
    ):
        audit_stale_docker_overlay_state_v2(
            runner=fixture,
            expected_root=fixture.root,
        )


def test_refuses_backing_device_regular_file(tmp_path: Path) -> None:
    fixture = _DockerFixture(tmp_path / "docker")
    (fixture.root / "overlay2/backingFsBlockDev").touch()

    with pytest.raises(
        DockerStaleMountReclaimerV2Error,
        match="backing-device metadata is not a block device",
    ):
        audit_stale_docker_overlay_state_v2(
            runner=fixture,
            expected_root=fixture.root,
        )


def test_refuses_unrecognized_overlay_root_special_entry(tmp_path: Path) -> None:
    fixture = _DockerFixture(tmp_path / "docker")
    (fixture.root / "overlay2/unexpected-metadata").touch()

    with pytest.raises(
        DockerStaleMountReclaimerV2Error,
        match="contains a non-directory entry",
    ):
        audit_stale_docker_overlay_state_v2(
            runner=fixture,
            expected_root=fixture.root,
        )


def test_interrupted_reclaim_shape_is_resumable(tmp_path: Path) -> None:
    fixture = _DockerFixture(tmp_path / "docker")
    stale_layer_record = (
        fixture.root
        / "image/overlay2/layerdb/sha256"
        / fixture.stale_chain.split(":", 1)[1]
    )
    # Simulate an earlier run that removed one stale record but not its data.
    shutil.rmtree(stale_layer_record)
    fixture.mounted.remove(
        str(fixture.root / "overlay2" / fixture.stale_mount / "merged")
    )

    result = reclaim_stale_docker_overlay_mounts_v2(
        runner=fixture,
        expected_root=fixture.root,
    )

    assert result.reclaimed_layer_record_count == 0
    assert result.reclaimed_overlay_dir_count == 4


@pytest.mark.parametrize(
    "missing_part",
    ["link", "overlay", "overlay-link-metadata"],
)
def test_interrupted_overlay_cleanup_is_resumable(
    tmp_path: Path,
    missing_part: str,
) -> None:
    fixture = _DockerFixture(tmp_path / "docker")
    stale_overlay = fixture.root / "overlay2" / fixture.orphan_overlay
    stale_link = (
        fixture.root / "overlay2/l" / fixture.links[fixture.orphan_overlay]
    )
    if missing_part == "link":
        stale_link.unlink()
    elif missing_part == "overlay":
        shutil.rmtree(stale_overlay)
    else:
        (stale_overlay / "link").unlink()
    fixture.mounted.remove(
        str(fixture.root / "overlay2" / fixture.stale_mount / "merged")
    )

    result = reclaim_stale_docker_overlay_mounts_v2(
        runner=fixture,
        expected_root=fixture.root,
    )

    assert result.reclaimed_layer_record_count == 1
    assert not stale_overlay.exists()
    assert not stale_link.exists()


def test_partially_removed_stale_metadata_is_resumable(tmp_path: Path) -> None:
    fixture = _DockerFixture(tmp_path / "docker")
    stale_layer = (
        fixture.root
        / "image/overlay2/layerdb/sha256"
        / fixture.stale_chain.split(":", 1)[1]
    )
    stale_mount = (
        fixture.root
        / "image/overlay2/layerdb/mounts"
        / fixture.stale_container_id
    )
    (stale_layer / "cache-id").unlink()
    (stale_mount / "mount-id").unlink()
    (stale_mount / "parent").unlink()
    fixture.mounted.remove(
        str(fixture.root / "overlay2" / fixture.stale_mount / "merged")
    )

    result = reclaim_stale_docker_overlay_mounts_v2(
        runner=fixture,
        expected_root=fixture.root,
    )

    assert result.reclaimed_layer_record_count == 1
    assert result.reclaimed_mount_record_count == 1


def test_refuses_unexpected_docker_root(tmp_path: Path) -> None:
    fixture = _DockerFixture(tmp_path / "docker")

    with pytest.raises(
        DockerStaleMountReclaimerV2Error,
        match="refusing unexpected Docker root",
    ):
        reclaim_stale_docker_overlay_mounts_v2(
            runner=fixture,
            expected_root=tmp_path / "different-root",
        )


def test_refuses_nonempty_build_cache(tmp_path: Path) -> None:
    fixture = _DockerFixture(tmp_path / "docker", build_cache_count=1)

    with pytest.raises(
        DockerStaleMountReclaimerV2Error,
        match="build-cache references remain",
    ):
        reclaim_stale_docker_overlay_mounts_v2(
            runner=fixture,
            expected_root=fixture.root,
        )


def test_refuses_active_lower_reference_to_stale_layer(tmp_path: Path) -> None:
    fixture = _DockerFixture(tmp_path / "docker")
    lower = fixture.root / "overlay2" / fixture.active_mount / "lower"
    lower.write_text(f"l/{fixture.links[fixture.stale_cache]}", encoding="utf-8")

    with pytest.raises(
        DockerStaleMountReclaimerV2Error,
        match="references a non-active lower layer",
    ):
        reclaim_stale_docker_overlay_mounts_v2(
            runner=fixture,
            expected_root=fixture.root,
        )


def test_refuses_malformed_overlay_directory(tmp_path: Path) -> None:
    fixture = _DockerFixture(tmp_path / "docker")
    (fixture.root / "overlay2/not-a-layer").mkdir()

    with pytest.raises(
        DockerStaleMountReclaimerV2Error,
        match="not a canonical Docker overlay identifier",
    ):
        reclaim_stale_docker_overlay_mounts_v2(
            runner=fixture,
            expected_root=fixture.root,
        )


def test_fails_closed_when_stale_mount_is_busy(tmp_path: Path) -> None:
    fixture = _DockerFixture(tmp_path / "docker", unmount_failure=True)

    with pytest.raises(
        DockerStaleMountReclaimerV2Error,
        match="stale Docker overlay unmount failed",
    ):
        reclaim_stale_docker_overlay_mounts_v2(
            runner=fixture,
            expected_root=fixture.root,
        )


def test_fails_closed_when_mount_remains_after_unmount(tmp_path: Path) -> None:
    fixture = _DockerFixture(tmp_path / "docker", retain_after_unmount=True)

    with pytest.raises(
        DockerStaleMountReclaimerV2Error,
        match="stale Docker overlay mounts remain",
    ):
        reclaim_stale_docker_overlay_mounts_v2(
            runner=fixture,
            expected_root=fixture.root,
        )


def test_refuses_non_overlay_mount_below_stale_directory(tmp_path: Path) -> None:
    fixture = _DockerFixture(tmp_path / "docker")
    fixture.mounted.add(
        str(fixture.root / "overlay2" / fixture.orphan_overlay / "diff")
    )

    with pytest.raises(
        DockerStaleMountReclaimerV2Error,
        match="mounted descendants remain",
    ):
        reclaim_stale_docker_overlay_mounts_v2(
            runner=fixture,
            expected_root=fixture.root,
        )
