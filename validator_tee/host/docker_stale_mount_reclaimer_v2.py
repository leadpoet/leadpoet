"""Reclaim unreachable Docker overlay state without disturbing live containers."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
from typing import Callable, Iterable, Optional, Sequence


class DockerStaleMountReclaimerV2Error(RuntimeError):
    """Raised when stale Docker state cannot be reclaimed safely."""


CommandRunner = Callable[[Sequence[str]], subprocess.CompletedProcess[str]]

_SHA256_RE = re.compile(r"^sha256:([0-9a-f]{64})$")
_HEX_ID_RE = re.compile(r"^[0-9a-f]{64}$")
_OVERLAY_ID_RE = re.compile(r"^(?:[a-z0-9]{20,64}|[0-9a-f]{64}-init)$")
_LINK_ID_RE = re.compile(r"^[A-Z2-7]{26}$")
_BACKING_DEVICE_NAME = "backingFsBlockDev"


@dataclass(frozen=True)
class DockerMountReclaimResult:
    docker_root: str
    active_container_count: int
    active_image_count: int
    active_layer_count: int
    active_mount_count: int
    mounted_overlay_count: int
    reclaimed_layer_record_count: int
    reclaimed_mount_count: int
    reclaimed_mount_record_count: int
    reclaimed_overlay_dir_count: int
    reclaimed_overlay_link_count: int

    def as_dict(self) -> dict[str, object]:
        return {
            "active_container_count": self.active_container_count,
            "active_image_count": self.active_image_count,
            "active_layer_count": self.active_layer_count,
            "active_mount_count": self.active_mount_count,
            "docker_root": self.docker_root,
            "mounted_overlay_count": self.mounted_overlay_count,
            "reclaimed_layer_record_count": self.reclaimed_layer_record_count,
            "reclaimed_mount_count": self.reclaimed_mount_count,
            "reclaimed_mount_record_count": self.reclaimed_mount_record_count,
            "reclaimed_overlay_dir_count": self.reclaimed_overlay_dir_count,
            "reclaimed_overlay_link_count": self.reclaimed_overlay_link_count,
            "schema_version": "leadpoet.docker_stale_mount_reclaim.v3",
            "status": "ready",
        }


@dataclass(frozen=True)
class _DockerStorageInventory:
    docker_root: Path
    image_ids: frozenset[str]
    container_ids: frozenset[str]
    active_chain_ids: frozenset[str]
    active_overlay_ids: frozenset[str]
    active_mount_targets: frozenset[str]
    mounted_overlay_targets: frozenset[str]
    stale_layer_records: tuple[Path, ...]
    stale_mount_records: tuple[Path, ...]
    stale_overlay_dirs: tuple[Path, ...]
    stale_overlay_links: tuple[Path, ...]
    active_manifest_hash: str


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


def _checked_json(
    runner: CommandRunner,
    command: Sequence[str],
    *,
    label: str,
) -> object:
    raw = _checked_output(runner, command, label=label)
    try:
        return json.loads(raw)
    except (TypeError, ValueError) as exc:
        raise DockerStaleMountReclaimerV2Error(f"{label} returned invalid JSON") from exc


def _docker_root(runner: CommandRunner, *, expected_root: Path) -> Path:
    raw_root = _checked_output(
        runner,
        ["docker", "info", "--format", "{{.DockerRootDir}}"],
        label="Docker root discovery",
    ).strip()
    root = Path(raw_root)
    if not root.is_absolute() or root != expected_root:
        raise DockerStaleMountReclaimerV2Error(
            f"refusing unexpected Docker root: {raw_root or '<empty>'}"
        )
    return root


def _validate_sha256(value: object, *, field: str) -> str:
    if not isinstance(value, str):
        raise DockerStaleMountReclaimerV2Error(f"{field} must be a sha256 digest")
    match = _SHA256_RE.fullmatch(value)
    if match is None:
        raise DockerStaleMountReclaimerV2Error(f"{field} must be a sha256 digest")
    return match.group(1)


def _validate_hex_id(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _HEX_ID_RE.fullmatch(value) is None:
        raise DockerStaleMountReclaimerV2Error(
            f"{field} must be a 64-character lowercase hex identifier"
        )
    return value


def _validate_overlay_id(value: str, *, field: str) -> str:
    if _OVERLAY_ID_RE.fullmatch(value) is None or value in {".", "..", "l"}:
        raise DockerStaleMountReclaimerV2Error(
            f"{field} is not a canonical Docker overlay identifier: {value}"
        )
    return value


def _validate_link_id(value: str, *, field: str) -> str:
    if _LINK_ID_RE.fullmatch(value) is None:
        raise DockerStaleMountReclaimerV2Error(
            f"{field} is not a canonical Docker overlay link identifier: {value}"
        )
    return value


def _read_small_file(path: Path, *, field: str, maximum_bytes: int = 4096) -> str:
    if path.is_symlink() or not path.is_file():
        raise DockerStaleMountReclaimerV2Error(f"{field} is not a regular file")
    if path.stat().st_size > maximum_bytes:
        raise DockerStaleMountReclaimerV2Error(f"{field} exceeds its size bound")
    try:
        return path.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeError) as exc:
        raise DockerStaleMountReclaimerV2Error(f"{field} is unreadable") from exc


def _chain_ids(diff_ids: object, *, image_id: str) -> set[str]:
    if not isinstance(diff_ids, list) or not diff_ids:
        raise DockerStaleMountReclaimerV2Error(
            f"Docker image {image_id} has no RootFS layer list"
        )
    result: set[str] = set()
    parent: Optional[str] = None
    for index, raw_diff_id in enumerate(diff_ids):
        diff_hex = _validate_sha256(
            raw_diff_id,
            field=f"Docker image {image_id} RootFS layer {index}",
        )
        diff_id = f"sha256:{diff_hex}"
        if parent is None:
            chain_id = diff_id
        else:
            chain_id = "sha256:" + hashlib.sha256(
                f"{parent} {diff_id}".encode("ascii")
            ).hexdigest()
        result.add(chain_id.split(":", 1)[1])
        parent = chain_id
    return result


def _docker_ids(
    runner: CommandRunner,
    command: Sequence[str],
    *,
    label: str,
    image_ids: bool,
) -> list[str]:
    values = [
        value.strip()
        for value in _checked_output(runner, command, label=label).splitlines()
        if value.strip()
    ]
    if image_ids:
        # One image may have several tags, and Docker emits one row per tag.
        values = sorted(set(values))
    elif len(values) != len(set(values)):
        raise DockerStaleMountReclaimerV2Error(f"{label} contains duplicates")
    if image_ids:
        for value in values:
            _validate_sha256(value, field=label)
    else:
        for value in values:
            _validate_hex_id(value, field=label)
    return values


def _inspect_documents(
    runner: CommandRunner,
    kind: str,
    identifiers: Sequence[str],
) -> list[dict[str, object]]:
    if not identifiers:
        return []
    document = _checked_json(
        runner,
        ["docker", kind, "inspect", *identifiers],
        label=f"Docker {kind} inspection",
    )
    if not isinstance(document, list) or len(document) != len(identifiers):
        raise DockerStaleMountReclaimerV2Error(
            f"Docker {kind} inspection is incomplete"
        )
    if not all(isinstance(row, dict) for row in document):
        raise DockerStaleMountReclaimerV2Error(
            f"Docker {kind} inspection returned malformed rows"
        )
    return document  # type: ignore[return-value]


def _require_empty_build_cache(runner: CommandRunner) -> None:
    output = _checked_output(
        runner,
        ["docker", "system", "df", "--format", "{{json .}}"],
        label="Docker storage reference inventory",
    )
    rows: list[dict[str, object]] = []
    for line in output.splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError as exc:
            raise DockerStaleMountReclaimerV2Error(
                "Docker storage reference inventory returned invalid JSON"
            ) from exc
        if not isinstance(row, dict):
            raise DockerStaleMountReclaimerV2Error(
                "Docker storage reference inventory returned a malformed row"
            )
        rows.append(row)
    build_rows = [row for row in rows if row.get("Type") == "Build Cache"]
    if len(build_rows) != 1:
        raise DockerStaleMountReclaimerV2Error(
            "Docker build-cache inventory is missing or ambiguous"
        )
    try:
        total = int(str(build_rows[0].get("TotalCount")))
        active = int(str(build_rows[0].get("Active")))
    except ValueError as exc:
        raise DockerStaleMountReclaimerV2Error(
            "Docker build-cache inventory contains invalid counts"
        ) from exc
    if total != 0 or active != 0:
        raise DockerStaleMountReclaimerV2Error(
            "refusing orphan reclaim while Docker build-cache references remain"
        )


def _overlay_target_pattern(docker_root: Path) -> re.Pattern[str]:
    return re.compile(
        rf"^{re.escape(str(docker_root))}/overlay2/([0-9a-f]{{64}})/merged$"
    )


def _validate_overlay_target(target: str, *, docker_root: Path) -> str:
    if _overlay_target_pattern(docker_root).fullmatch(target) is None:
        raise DockerStaleMountReclaimerV2Error(
            f"refusing malformed Docker overlay mount target: {target}"
        )
    return target


def _mounted_overlay_dirs(
    runner: CommandRunner,
    *,
    docker_root: Path,
) -> set[str]:
    command = ["findmnt", "-rn", "-t", "overlay", "-o", "TARGET"]
    result = runner(command)
    stdout = result.stdout or ""
    stderr = result.stderr or ""
    if result.returncode == 1 and stdout == "" and stderr == "":
        # util-linux findmnt reports a clean no-match inventory with status 1.
        output = ""
    elif result.returncode != 0:
        detail = (stderr or stdout).strip()
        raise DockerStaleMountReclaimerV2Error(
            "mounted overlay inventory failed"
            + (f": {detail}" if detail else "")
        )
    else:
        output = stdout
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


def _all_mount_targets(runner: CommandRunner) -> set[str]:
    return {
        value.strip()
        for value in _checked_output(
            runner,
            ["findmnt", "-rn", "-o", "TARGET"],
            label="complete mount inventory",
        ).splitlines()
        if value.strip()
    }


def _iter_directories(path: Path, *, label: str) -> list[Path]:
    if path.is_symlink() or not path.is_dir():
        raise DockerStaleMountReclaimerV2Error(f"{label} is not a directory")
    rows = []
    for child in path.iterdir():
        if child.is_symlink() or not child.is_dir():
            raise DockerStaleMountReclaimerV2Error(
                f"{label} contains a non-directory entry: {child.name}"
            )
        rows.append(child)
    return sorted(rows)


def _overlay_links(
    overlay_root: Path,
    overlay_ids: Iterable[str],
) -> dict[str, str]:
    result: dict[str, str] = {}
    for overlay_id in sorted(set(overlay_ids)):
        _validate_overlay_id(overlay_id, field="Docker overlay directory")
        overlay_dir = overlay_root / overlay_id
        if overlay_dir.is_symlink() or not overlay_dir.is_dir():
            raise DockerStaleMountReclaimerV2Error(
                f"Docker overlay directory is missing: {overlay_dir}"
            )
        link_id = _validate_link_id(
            _read_small_file(
                overlay_dir / "link",
                field=f"Docker overlay link for {overlay_id}",
            ),
            field=f"Docker overlay link for {overlay_id}",
        )
        if link_id in result:
            raise DockerStaleMountReclaimerV2Error(
                f"Docker overlay link identifier is duplicated: {link_id}"
            )
        result[link_id] = overlay_id
    return result


def _backing_device_identity(path: Path) -> Optional[dict[str, int]]:
    """Bind Docker's optional overlay backing-device node without reading it."""

    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise DockerStaleMountReclaimerV2Error(
            "Docker overlay backing-device metadata is unreadable"
        ) from exc
    if not stat.S_ISBLK(metadata.st_mode):
        raise DockerStaleMountReclaimerV2Error(
            "Docker overlay backing-device metadata is not a block device"
        )
    mode = stat.S_IMODE(metadata.st_mode)
    if mode != 0o600 or metadata.st_uid != 0 or metadata.st_gid != 0:
        raise DockerStaleMountReclaimerV2Error(
            "Docker overlay backing-device metadata has unsafe ownership or mode"
        )
    major = os.major(metadata.st_rdev)
    minor = os.minor(metadata.st_rdev)
    if major <= 0 or minor < 0 or metadata.st_size != 0:
        raise DockerStaleMountReclaimerV2Error(
            "Docker overlay backing-device metadata has an invalid device identity"
        )
    return {
        "gid": metadata.st_gid,
        "major": major,
        "minor": minor,
        "mode": mode,
        "uid": metadata.st_uid,
    }


def _metadata_manifest_hash(
    *,
    image_rows: Sequence[dict[str, object]],
    container_rows: Sequence[dict[str, object]],
    active_chain_ids: Iterable[str],
    active_overlay_ids: Iterable[str],
    layerdb_root: Path,
    mountdb_root: Path,
    overlay_root: Path,
    active_links: dict[str, str],
    backing_device_identity: Optional[dict[str, int]],
) -> str:
    metadata: dict[str, object] = {
        "containers": [],
        "images": [],
        "layers": {},
        "mounts": {},
        "overlays": {},
        "backing_fs_block_device": backing_device_identity,
    }
    for row in sorted(image_rows, key=lambda value: str(value.get("Id"))):
        image_id = str(row.get("Id"))
        rootfs = row.get("RootFS")
        metadata["images"].append(  # type: ignore[union-attr]
            {"id": image_id, "layers": rootfs.get("Layers") if isinstance(rootfs, dict) else None}
        )
    for row in sorted(container_rows, key=lambda value: str(value.get("Id"))):
        graph = row.get("GraphDriver")
        graph_data = graph.get("Data") if isinstance(graph, dict) else None
        metadata["containers"].append(  # type: ignore[union-attr]
            {
                "id": row.get("Id"),
                "image": row.get("Image"),
                "merged_dir": graph_data.get("MergedDir") if isinstance(graph_data, dict) else None,
            }
        )
    for chain_id in sorted(set(active_chain_ids)):
        directory = layerdb_root / chain_id
        metadata["layers"][chain_id] = {  # type: ignore[index]
            name: _read_small_file(directory / name, field=f"active layer {chain_id} {name}")
            for name in ("cache-id", "diff", "size")
        }
        parent = directory / "parent"
        if parent.exists():
            metadata["layers"][chain_id]["parent"] = _read_small_file(  # type: ignore[index]
                parent,
                field=f"active layer {chain_id} parent",
            )
    for row in container_rows:
        container_id = str(row.get("Id"))
        directory = mountdb_root / container_id
        metadata["mounts"][container_id] = {  # type: ignore[index]
            name: _read_small_file(
                directory / name,
                field=f"active container {container_id} {name}",
            )
            for name in ("init-id", "mount-id", "parent")
        }
    for overlay_id in sorted(set(active_overlay_ids)):
        directory = overlay_root / overlay_id
        lower = directory / "lower"
        metadata["overlays"][overlay_id] = {  # type: ignore[index]
            "link": _read_small_file(directory / "link", field=f"active overlay {overlay_id} link"),
            "lower": _read_small_file(lower, field=f"active overlay {overlay_id} lower") if lower.exists() else None,
        }
    metadata["links"] = dict(sorted(active_links.items()))
    payload = json.dumps(metadata, sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _inventory(
    *,
    runner: CommandRunner,
    expected_root: Path,
) -> _DockerStorageInventory:
    docker_root = _docker_root(runner, expected_root=expected_root)
    _require_empty_build_cache(runner)
    overlay_root = docker_root / "overlay2"
    layerdb_root = docker_root / "image" / "overlay2" / "layerdb" / "sha256"
    mountdb_root = docker_root / "image" / "overlay2" / "layerdb" / "mounts"

    image_ids = _docker_ids(
        runner,
        ["docker", "images", "-aq", "--no-trunc"],
        label="Docker image inventory",
        image_ids=True,
    )
    container_ids = _docker_ids(
        runner,
        ["docker", "ps", "-aq", "--no-trunc"],
        label="Docker container inventory",
        image_ids=False,
    )
    image_rows = _inspect_documents(runner, "image", image_ids)
    container_rows = _inspect_documents(runner, "container", container_ids)

    inspected_image_ids = {str(row.get("Id")) for row in image_rows}
    if inspected_image_ids != set(image_ids):
        raise DockerStaleMountReclaimerV2Error("Docker image inspection changed identity")
    active_chain_ids: set[str] = set()
    for row in image_rows:
        image_id = str(row["Id"])
        rootfs = row.get("RootFS")
        if not isinstance(rootfs, dict):
            raise DockerStaleMountReclaimerV2Error(
                f"Docker image {image_id} has malformed RootFS metadata"
            )
        active_chain_ids.update(_chain_ids(rootfs.get("Layers"), image_id=image_id))

    inspected_container_ids = {str(row.get("Id")) for row in container_rows}
    if inspected_container_ids != set(container_ids):
        raise DockerStaleMountReclaimerV2Error(
            "Docker container inspection changed identity"
        )

    if docker_root.is_symlink() or not docker_root.is_dir():
        raise DockerStaleMountReclaimerV2Error(
            "Docker data-root is not a canonical directory"
        )
    metadata_ancestors = (
        docker_root / "image",
        docker_root / "image" / "overlay2",
        docker_root / "image" / "overlay2" / "layerdb",
    )
    for path in metadata_ancestors:
        if path.is_symlink() or (path.exists() and not path.is_dir()):
            raise DockerStaleMountReclaimerV2Error(
                f"Docker overlay metadata ancestor is unsafe: {path}"
            )

    metadata_roots = (layerdb_root, mountdb_root, overlay_root)
    metadata_root_states: list[bool] = []
    for path in metadata_roots:
        if path.is_symlink():
            raise DockerStaleMountReclaimerV2Error(
                f"Docker overlay metadata root is a symlink: {path}"
            )
        exists = path.exists()
        if exists and not path.is_dir():
            raise DockerStaleMountReclaimerV2Error(
                f"Docker overlay metadata root is not a directory: {path}"
            )
        metadata_root_states.append(exists)
    if any(metadata_root_states) and not all(metadata_root_states):
        raise DockerStaleMountReclaimerV2Error(
            "Docker overlay metadata roots are only partially initialized"
        )
    if not any(metadata_root_states):
        if image_ids or container_ids:
            raise DockerStaleMountReclaimerV2Error(
                "Docker overlay metadata is absent while active objects remain"
            )
        mounted_overlay_targets = _mounted_overlay_dirs(
            runner,
            docker_root=docker_root,
        )
        if mounted_overlay_targets:
            raise DockerStaleMountReclaimerV2Error(
                "Docker overlay metadata is absent while overlay mounts remain"
            )
        protected_roots = (
            str(docker_root / "image" / "overlay2"),
            str(overlay_root),
        )
        conflicting_mounts = sorted(
            target
            for target in _all_mount_targets(runner)
            if any(
                target == root or target.startswith(root + "/")
                for root in protected_roots
            )
        )
        if conflicting_mounts:
            raise DockerStaleMountReclaimerV2Error(
                "Docker overlay metadata is absent while protected mounts remain: "
                + ",".join(conflicting_mounts)
            )
        active_manifest_hash = _metadata_manifest_hash(
            image_rows=(),
            container_rows=(),
            active_chain_ids=(),
            active_overlay_ids=(),
            layerdb_root=layerdb_root,
            mountdb_root=mountdb_root,
            overlay_root=overlay_root,
            active_links={},
            backing_device_identity=None,
        )
        return _DockerStorageInventory(
            docker_root=docker_root,
            image_ids=frozenset(),
            container_ids=frozenset(),
            active_chain_ids=frozenset(),
            active_overlay_ids=frozenset(),
            active_mount_targets=frozenset(),
            mounted_overlay_targets=frozenset(),
            stale_layer_records=(),
            stale_mount_records=(),
            stale_overlay_dirs=(),
            stale_overlay_links=(),
            active_manifest_hash=active_manifest_hash,
        )

    layer_records = _iter_directories(layerdb_root, label="Docker layer database")
    mount_records = _iter_directories(mountdb_root, label="Docker mount database")
    layer_record_ids = {
        _validate_hex_id(path.name, field="Docker layer database entry"): path
        for path in layer_records
    }
    mount_record_ids = {
        _validate_hex_id(path.name, field="Docker mount database entry"): path
        for path in mount_records
    }
    missing_active_layers = active_chain_ids - set(layer_record_ids)
    if missing_active_layers:
        raise DockerStaleMountReclaimerV2Error(
            "Docker active layer records are missing: "
            + ",".join(sorted(missing_active_layers))
        )
    missing_active_mounts = set(container_ids) - set(mount_record_ids)
    if missing_active_mounts:
        raise DockerStaleMountReclaimerV2Error(
            "Docker active mount records are missing: "
            + ",".join(sorted(missing_active_mounts))
        )

    active_layer_cache_ids: dict[str, str] = {}
    for chain_id in active_chain_ids:
        directory = layer_record_ids[chain_id]
        active_layer_cache_ids[chain_id] = _validate_overlay_id(
            _read_small_file(
                directory / "cache-id",
                field=f"Docker layer {chain_id} cache-id",
            ),
            field=f"Docker layer {chain_id} cache-id",
        )

    mount_overlay_ids: dict[str, tuple[str, str]] = {}
    active_mount_targets: set[str] = set()
    for row in container_rows:
        container_id = str(row["Id"])
        if row.get("Image") not in image_ids:
            raise DockerStaleMountReclaimerV2Error(
                f"Docker container {container_id} references an unlisted image"
            )
        directory = mount_record_ids[container_id]
        mount_id = _validate_overlay_id(
            _read_small_file(directory / "mount-id", field=f"container {container_id} mount-id"),
            field=f"container {container_id} mount-id",
        )
        init_id = _validate_overlay_id(
            _read_small_file(directory / "init-id", field=f"container {container_id} init-id"),
            field=f"container {container_id} init-id",
        )
        parent = _read_small_file(directory / "parent", field=f"container {container_id} parent")
        parent_hex = _validate_sha256(parent, field=f"container {container_id} parent")
        if parent_hex not in active_chain_ids:
            raise DockerStaleMountReclaimerV2Error(
                f"Docker container {container_id} parent is not an active image layer"
            )
        graph = row.get("GraphDriver")
        data = graph.get("Data") if isinstance(graph, dict) else None
        merged_dir = data.get("MergedDir") if isinstance(data, dict) else None
        expected_merged = str(overlay_root / mount_id / "merged")
        if merged_dir != expected_merged:
            raise DockerStaleMountReclaimerV2Error(
                f"Docker container {container_id} merged directory differs from mount metadata"
            )
        active_mount_targets.add(
            _validate_overlay_target(expected_merged, docker_root=docker_root)
        )
        mount_overlay_ids[container_id] = (mount_id, init_id)

    stale_mount_records = tuple(
        path for identifier, path in mount_record_ids.items() if identifier not in container_ids
    )
    stale_mount_overlay_ids: set[str] = set()
    for directory in stale_mount_records:
        for name in ("mount-id", "init-id"):
            metadata_path = directory / name
            if not metadata_path.exists() and not metadata_path.is_symlink():
                continue
            stale_mount_overlay_ids.add(
                _validate_overlay_id(
                    _read_small_file(
                        metadata_path,
                        field=f"stale mount {directory.name} {name}",
                    ),
                    field=f"stale mount {directory.name} {name}",
                )
            )
        parent_path = directory / "parent"
        if parent_path.exists() or parent_path.is_symlink():
            _validate_sha256(
                _read_small_file(
                    parent_path,
                    field=f"stale mount {directory.name} parent",
                ),
                field=f"stale mount {directory.name} parent",
            )

    active_layer_overlay_ids = set(active_layer_cache_ids.values())
    active_container_overlay_ids = {
        overlay_id for values in mount_overlay_ids.values() for overlay_id in values
    }
    active_overlay_ids = active_layer_overlay_ids | active_container_overlay_ids
    stale_layer_records = tuple(
        path for identifier, path in layer_record_ids.items() if identifier not in active_chain_ids
    )
    stale_layer_overlay_ids: set[str] = set()
    for directory in stale_layer_records:
        cache_path = directory / "cache-id"
        if not cache_path.exists() and not cache_path.is_symlink():
            continue
        stale_layer_overlay_ids.add(
            _validate_overlay_id(
                _read_small_file(
                    cache_path,
                    field=f"stale layer {directory.name} cache-id",
                ),
                field=f"stale layer {directory.name} cache-id",
            )
        )
    all_declared_overlay_ids = (
        list(active_layer_cache_ids.values())
        + [value for values in mount_overlay_ids.values() for value in values]
        + list(stale_layer_overlay_ids)
        + list(stale_mount_overlay_ids)
    )
    if len(all_declared_overlay_ids) != len(set(all_declared_overlay_ids)):
        raise DockerStaleMountReclaimerV2Error(
            "Docker layer or mount metadata contains duplicate overlay identifiers"
        )
    if active_overlay_ids & (stale_layer_overlay_ids | stale_mount_overlay_ids):
        raise DockerStaleMountReclaimerV2Error(
            "Docker stale metadata aliases an active overlay directory"
        )

    all_overlay_dirs: dict[str, Path] = {}
    backing_device_identity: Optional[dict[str, int]] = None
    if overlay_root.is_symlink() or not overlay_root.is_dir():
        raise DockerStaleMountReclaimerV2Error("Docker overlay root is not a directory")
    for path in overlay_root.iterdir():
        if path.name == "l":
            continue
        if path.name == _BACKING_DEVICE_NAME:
            backing_device_identity = _backing_device_identity(path)
            continue
        if path.is_symlink() or not path.is_dir():
            raise DockerStaleMountReclaimerV2Error(
                f"Docker overlay root contains a non-directory entry: {path.name}"
            )
        overlay_id = _validate_overlay_id(path.name, field="Docker overlay directory")
        all_overlay_dirs[overlay_id] = path
    missing_active_overlays = active_overlay_ids - set(all_overlay_dirs)
    if missing_active_overlays:
        raise DockerStaleMountReclaimerV2Error(
            "Docker active overlay directories are missing: "
            + ",".join(sorted(missing_active_overlays))
        )

    mounted_overlay_targets = _mounted_overlay_dirs(runner, docker_root=docker_root)
    stale_mounted_targets = sorted(mounted_overlay_targets - active_mount_targets)
    known_stale_overlay_ids = (
        stale_layer_overlay_ids
        | stale_mount_overlay_ids
        | (set(all_overlay_dirs) - active_overlay_ids)
    )
    for target in stale_mounted_targets:
        match = _overlay_target_pattern(docker_root).fullmatch(target)
        assert match is not None
        if match.group(1) not in known_stale_overlay_ids:
            raise DockerStaleMountReclaimerV2Error(
                f"mounted Docker overlay is neither active nor proven stale: {target}"
            )

    stale_overlay_dirs = tuple(
        path
        for overlay_id, path in all_overlay_dirs.items()
        if overlay_id not in active_overlay_ids
    )
    declared_link_map = _overlay_links(overlay_root, active_overlay_ids)
    for stale_dir in sorted(stale_overlay_dirs):
        link_metadata = stale_dir / "link"
        if not link_metadata.exists() and not link_metadata.is_symlink():
            continue
        link_id = _validate_link_id(
            _read_small_file(
                link_metadata,
                field=f"Docker stale overlay link for {stale_dir.name}",
            ),
            field=f"Docker stale overlay link for {stale_dir.name}",
        )
        if link_id in declared_link_map:
            raise DockerStaleMountReclaimerV2Error(
                f"Docker overlay link identifier is duplicated: {link_id}"
            )
        declared_link_map[link_id] = stale_dir.name
    link_root = overlay_root / "l"
    if link_root.is_symlink() or not link_root.is_dir():
        raise DockerStaleMountReclaimerV2Error("Docker overlay link root is not a directory")
    link_paths: dict[str, Path] = {}
    link_targets: dict[str, str] = {}
    for path in link_root.iterdir():
        link_id = _validate_link_id(path.name, field="Docker overlay link entry")
        if not path.is_symlink():
            raise DockerStaleMountReclaimerV2Error(
                f"Docker overlay link entry is not a symlink: {path.name}"
            )
        target = os.readlink(path)
        match = re.fullmatch(r"\.\./([a-z0-9-]{20,69})/diff", target)
        if match is None:
            raise DockerStaleMountReclaimerV2Error(
                f"Docker overlay link has an invalid target: {path.name}"
            )
        overlay_id = _validate_overlay_id(match.group(1), field="Docker overlay link target")
        link_paths[link_id] = path
        link_targets[link_id] = overlay_id

    active_link_map = {
        link_id: overlay_id
        for link_id, overlay_id in declared_link_map.items()
        if overlay_id in active_overlay_ids
    }
    for link_id, overlay_id in active_link_map.items():
        if link_targets.get(link_id) != overlay_id:
            raise DockerStaleMountReclaimerV2Error(
                f"active Docker overlay link is missing or mismatched: {link_id}"
            )
    for link_id, overlay_id in link_targets.items():
        if overlay_id in active_overlay_ids and active_link_map.get(link_id) != overlay_id:
            raise DockerStaleMountReclaimerV2Error(
                f"Docker overlay link aliases an active layer: {link_id}"
            )
        expected_overlay = declared_link_map.get(link_id)
        if expected_overlay is not None and expected_overlay != overlay_id:
            raise DockerStaleMountReclaimerV2Error(
                f"Docker overlay link does not match its overlay metadata: {link_id}"
            )
    for overlay_id in active_overlay_ids:
        lower_path = overlay_root / overlay_id / "lower"
        if not lower_path.exists():
            continue
        lower = _read_small_file(lower_path, field=f"active overlay {overlay_id} lower")
        for token in lower.split(":"):
            match = re.fullmatch(r"l/([A-Z2-7]{26})", token)
            if match is None or match.group(1) not in active_link_map:
                raise DockerStaleMountReclaimerV2Error(
                    f"active overlay {overlay_id} references a non-active lower layer"
                )

    stale_overlay_links = tuple(
        path for link_id, path in link_paths.items() if link_id not in active_link_map
    )
    active_manifest_hash = _metadata_manifest_hash(
        image_rows=image_rows,
        container_rows=container_rows,
        active_chain_ids=active_chain_ids,
        active_overlay_ids=active_overlay_ids,
        layerdb_root=layerdb_root,
        mountdb_root=mountdb_root,
        overlay_root=overlay_root,
        active_links=active_link_map,
        backing_device_identity=backing_device_identity,
    )
    return _DockerStorageInventory(
        docker_root=docker_root,
        image_ids=frozenset(image_ids),
        container_ids=frozenset(container_ids),
        active_chain_ids=frozenset(active_chain_ids),
        active_overlay_ids=frozenset(active_overlay_ids),
        active_mount_targets=frozenset(active_mount_targets),
        mounted_overlay_targets=frozenset(mounted_overlay_targets),
        stale_layer_records=tuple(sorted(stale_layer_records)),
        stale_mount_records=tuple(sorted(stale_mount_records)),
        stale_overlay_dirs=tuple(sorted(stale_overlay_dirs)),
        stale_overlay_links=tuple(sorted(stale_overlay_links)),
        active_manifest_hash=active_manifest_hash,
    )


def _remove_directory(path: Path, *, root: Path) -> None:
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise DockerStaleMountReclaimerV2Error(
            f"refusing cleanup path outside guarded root: {path}"
        ) from exc
    if path.is_symlink():
        raise DockerStaleMountReclaimerV2Error(
            f"refusing symlink where a stale directory was expected: {path}"
        )
    if path.exists():
        shutil.rmtree(path)


def reclaim_stale_docker_overlay_mounts_v2(
    *,
    runner: CommandRunner = _run,
    expected_root: Path = Path("/var/lib/docker"),
) -> DockerMountReclaimResult:
    """Remove only Docker overlay state unreachable from live inventory."""

    before = _inventory(runner=runner, expected_root=expected_root)
    stale_mounts = sorted(
        before.mounted_overlay_targets - before.active_mount_targets
    )
    for target in stale_mounts:
        result = runner(["umount", "--", target])
        if result.returncode != 0:
            detail = (result.stderr or result.stdout or "").strip()
            raise DockerStaleMountReclaimerV2Error(
                f"stale Docker overlay unmount failed for {target}"
                + (f": {detail}" if detail else "")
            )
    remaining_mounts = _mounted_overlay_dirs(
        runner,
        docker_root=before.docker_root,
    ) - before.active_mount_targets
    if remaining_mounts:
        raise DockerStaleMountReclaimerV2Error(
            "stale Docker overlay mounts remain after guarded reclaim: "
            + ",".join(sorted(remaining_mounts))
        )

    complete_mounts = _all_mount_targets(runner)
    for stale_dir in before.stale_overlay_dirs:
        prefix = str(stale_dir) + "/"
        conflicts = sorted(
            target
            for target in complete_mounts
            if target == str(stale_dir) or target.startswith(prefix)
        )
        if conflicts:
            raise DockerStaleMountReclaimerV2Error(
                f"refusing stale overlay cleanup while mounted descendants remain under {stale_dir}: "
                + ",".join(conflicts)
            )

    layerdb_root = before.docker_root / "image" / "overlay2" / "layerdb"
    overlay_root = before.docker_root / "overlay2"
    for path in before.stale_layer_records:
        _remove_directory(path, root=layerdb_root / "sha256")
    for path in before.stale_mount_records:
        _remove_directory(path, root=layerdb_root / "mounts")
    for path in before.stale_overlay_links:
        if path.parent != overlay_root / "l" or not path.is_symlink():
            raise DockerStaleMountReclaimerV2Error(
                f"refusing malformed stale overlay link: {path}"
            )
        path.unlink()
    for path in before.stale_overlay_dirs:
        _remove_directory(path, root=overlay_root)

    after = _inventory(runner=runner, expected_root=expected_root)
    if after.active_manifest_hash != before.active_manifest_hash:
        raise DockerStaleMountReclaimerV2Error(
            "active Docker image/container identity changed during orphan reclaim"
        )
    if after.stale_layer_records or after.stale_mount_records:
        raise DockerStaleMountReclaimerV2Error(
            "stale Docker layer metadata remains after guarded reclaim"
        )
    if after.stale_overlay_dirs or after.stale_overlay_links:
        raise DockerStaleMountReclaimerV2Error(
            "stale Docker overlay data remains after guarded reclaim"
        )
    if after.mounted_overlay_targets - after.active_mount_targets:
        raise DockerStaleMountReclaimerV2Error(
            "stale Docker overlay mounts remain after guarded reclaim"
        )
    return DockerMountReclaimResult(
        docker_root=str(before.docker_root),
        active_container_count=len(before.container_ids),
        active_image_count=len(before.image_ids),
        active_layer_count=len(before.active_chain_ids),
        active_mount_count=len(before.active_mount_targets),
        mounted_overlay_count=len(before.mounted_overlay_targets),
        reclaimed_layer_record_count=len(before.stale_layer_records),
        reclaimed_mount_count=len(stale_mounts),
        reclaimed_mount_record_count=len(before.stale_mount_records),
        reclaimed_overlay_dir_count=len(before.stale_overlay_dirs),
        reclaimed_overlay_link_count=len(before.stale_overlay_links),
    )


def audit_stale_docker_overlay_state_v2(
    *,
    runner: CommandRunner = _run,
    expected_root: Path = Path("/var/lib/docker"),
) -> dict[str, object]:
    """Return the exact active/stale partition without modifying Docker state."""

    inventory = _inventory(runner=runner, expected_root=expected_root)
    return {
        "active_container_count": len(inventory.container_ids),
        "active_image_count": len(inventory.image_ids),
        "active_layer_count": len(inventory.active_chain_ids),
        "active_mount_count": len(inventory.active_mount_targets),
        "active_overlay_dir_count": len(inventory.active_overlay_ids),
        "active_manifest_hash": inventory.active_manifest_hash,
        "docker_root": str(inventory.docker_root),
        "mounted_overlay_count": len(inventory.mounted_overlay_targets),
        "stale_layer_record_count": len(inventory.stale_layer_records),
        "stale_mount_record_count": len(inventory.stale_mount_records),
        "stale_overlay_dir_count": len(inventory.stale_overlay_dirs),
        "stale_overlay_link_count": len(inventory.stale_overlay_links),
        "schema_version": "leadpoet.docker_stale_mount_audit.v3",
        "status": "ready",
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args(argv)
    if os.geteuid() != 0:
        raise SystemExit("Docker orphan-state reclaim must run as root")
    try:
        if args.audit_only:
            print(
                json.dumps(
                    audit_stale_docker_overlay_state_v2(),
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
            return 0
        result = reclaim_stale_docker_overlay_mounts_v2()
    except DockerStaleMountReclaimerV2Error as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(result.as_dict(), sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
