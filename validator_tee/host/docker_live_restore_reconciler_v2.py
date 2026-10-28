"""Refresh Docker metadata after guarded raw cleanup without stopping containers."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import time
from typing import Callable, Optional, Sequence

# Keep the existing direct, isolated pre-dependency invocation working.
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from docker_operation_guard_v2 import find_foreign_docker_operations, inspect_exact_host_gateway_runtime
    from docker_stale_mount_reclaimer_v2 import (
        DockerStaleMountReclaimerV2Error, _build_cache_counts,
        reclaim_stale_docker_overlay_mounts_v2, audit_stale_docker_overlay_state_v2,
    )
    from docker_zero_runtime_reconciler_v2 import (
        DockerZeroRuntimeReconcilerV2Error, _service_identity,
        verify_docker_operation_locks_v2,
    )
else:
    from .docker_operation_guard_v2 import find_foreign_docker_operations, inspect_exact_host_gateway_runtime
    from .docker_stale_mount_reclaimer_v2 import (
        DockerStaleMountReclaimerV2Error, _build_cache_counts,
        reclaim_stale_docker_overlay_mounts_v2, audit_stale_docker_overlay_state_v2,
    )
    from .docker_zero_runtime_reconciler_v2 import (
        DockerZeroRuntimeReconcilerV2Error, _service_identity,
        verify_docker_operation_locks_v2,
    )


class DockerLiveRestoreReconcilerV2Error(RuntimeError):
    """Raised when dockerd cannot be refreshed without changing live state."""


CommandRunner = Callable[[Sequence[str]], subprocess.CompletedProcess[str]]
Sleeper = Callable[[float], None]

_CONTAINER_ID_RE = re.compile(r"^[0-9a-f]{64}$")
_IMAGE_ID_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_MAX_CONFIG_BYTES = 1024 * 1024
_PENDING_RECONCILE = ".leadpoet-pending-overlay-reconcile"
_RUNTIME_DIRECTORY = Path("/etc/docker-runtimes.d")
_RUNTIME_SETUP_SCRIPT = Path("/usr/libexec/docker/docker-setup-runtimes.sh")
_RUNTIME_ENV_FILE = Path("/run/docker/runtimes.env")
_DOCKER_ENV_FILES = (Path("/etc/sysconfig/docker"), Path("/etc/sysconfig/docker-storage"))


@dataclass(frozen=True)
class DockerRuntimeSnapshot:
    container_ids: tuple[str, ...]
    image_ids: tuple[str, ...]
    volume_names: tuple[str, ...]
    running_container_ids: tuple[str, ...]
    task_ids: tuple[str, ...]
    server_version: str
    manifest_hash: str


@dataclass(frozen=True)
class DockerDaemonReconcileResult:
    config_changed: bool
    container_count: int
    image_count: int
    manifest_hash: str

    def as_dict(self) -> dict[str, object]:
        return {
            "config_changed": self.config_changed,
            "container_count": self.container_count,
            "image_count": self.image_count,
            "manifest_hash": self.manifest_hash,
            "schema_version": "leadpoet.docker_live_restore_reconcile.v1",
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
        raise DockerLiveRestoreReconcilerV2Error(
            f"{label} failed" + (f": {detail}" if detail else "")
        )
    return result.stdout


def _validated_ids(raw: str, *, image: bool, field: str) -> tuple[str, ...]:
    values = tuple(sorted(value for value in raw.splitlines() if value))
    if image:
        # Docker emits one row per tag; one immutable image may have many tags.
        values = tuple(sorted(set(values)))
    if len(values) != len(set(values)):
        raise DockerLiveRestoreReconcilerV2Error(f"{field} contains duplicates")
    pattern = _IMAGE_ID_RE if image else _CONTAINER_ID_RE
    if any(pattern.fullmatch(value) is None for value in values):
        raise DockerLiveRestoreReconcilerV2Error(
            f"{field} contains a malformed identifier"
        )
    return values


def _runtime_snapshot(runner: CommandRunner) -> DockerRuntimeSnapshot:
    container_ids = _validated_ids(
        _checked_output(
            runner,
            ["docker", "ps", "-aq", "--no-trunc"],
            label="Docker container inventory",
        ),
        image=False,
        field="Docker container inventory",
    )
    running_ids = _validated_ids(
        _checked_output(
            runner,
            ["docker", "ps", "-q", "--no-trunc"],
            label="Docker running-container inventory",
        ),
        image=False,
        field="Docker running-container inventory",
    )
    if container_ids != running_ids:
        raise DockerLiveRestoreReconcilerV2Error(
            "refusing dockerd refresh unless every Docker container is running"
        )
    if not container_ids:
        raise DockerLiveRestoreReconcilerV2Error(
            "live dockerd reconciliation requires at least one running container"
        )
    raw_inspection = _checked_output(
        runner,
        ["docker", "container", "inspect", *container_ids],
        label="Docker container inspection",
    )
    try:
        rows = json.loads(raw_inspection)
    except ValueError as exc:
        raise DockerLiveRestoreReconcilerV2Error(
            "Docker container inspection returned invalid JSON"
        ) from exc
    if not isinstance(rows, list) or len(rows) != len(container_ids):
        raise DockerLiveRestoreReconcilerV2Error(
            "Docker container inspection is incomplete"
        )
    normalized_rows: list[dict[str, object]] = []
    image_ids = _validated_ids(_checked_output(
        runner, ["docker", "images", "-aq", "--no-trunc"],
        label="Docker image inventory",
    ), image=True, field="Docker image inventory")
    try:
        image_rows = json.loads(_checked_output(
            runner, ["docker", "image", "inspect", *image_ids],
            label="Docker image inspection",
        )) if image_ids else []
    except ValueError as exc:
        raise DockerLiveRestoreReconcilerV2Error("Docker image inspection is malformed") from exc
    if (
        not isinstance(image_rows, list) or len(image_rows) != len(image_ids)
        or any(not isinstance(row, dict) for row in image_rows)
        or {row.get("Id") for row in image_rows} != set(image_ids)
        or any(not isinstance(row.get("RootFS"), dict)
               or not isinstance(row["RootFS"].get("Layers"), list)
               or any(not isinstance(layer, str) or _IMAGE_ID_RE.fullmatch(layer) is None
                      for layer in row["RootFS"]["Layers"]) for row in image_rows)
    ):
        raise DockerLiveRestoreReconcilerV2Error("Docker image inspection is incomplete or malformed")
    volume_names = tuple(sorted(_checked_output(
        runner, ["docker", "volume", "ls", "-q"], label="Docker volume inventory",
    ).splitlines()))
    if len(volume_names) != len(set(volume_names)) or any(
        re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name) is None for name in volume_names
    ):
        raise DockerLiveRestoreReconcilerV2Error("Docker volume inventory is malformed")
    try:
        volume_rows = json.loads(_checked_output(
            runner, ["docker", "volume", "inspect", *volume_names],
            label="Docker volume inspection",
        )) if volume_names else []
    except ValueError as exc:
        raise DockerLiveRestoreReconcilerV2Error("Docker volume inspection is malformed") from exc
    if (
        not isinstance(volume_rows, list) or len(volume_rows) != len(volume_names)
        or any(not isinstance(row, dict) for row in volume_rows)
        or {row.get("Name") for row in volume_rows} != set(volume_names)
    ):
        raise DockerLiveRestoreReconcilerV2Error("Docker volume inspection is incomplete")
    for row in rows:
        if not isinstance(row, dict):
            raise DockerLiveRestoreReconcilerV2Error(
                "Docker container inspection contains a malformed row"
            )
        state = row.get("State")
        graph = row.get("GraphDriver")
        graph_data = graph.get("Data") if isinstance(graph, dict) else None
        container_id = row.get("Id")
        if container_id not in container_ids or not isinstance(state, dict):
            raise DockerLiveRestoreReconcilerV2Error(
                "Docker container inspection changed identity"
            )
        if state.get("Running") is not True:
            raise DockerLiveRestoreReconcilerV2Error(
                f"Docker container is not running: {container_id}"
            )
        pid = state.get("Pid")
        if not isinstance(pid, int) or pid <= 0:
            raise DockerLiveRestoreReconcilerV2Error(
                f"Docker container has no live pid: {container_id}"
            )
        image_id = row.get("Image")
        if not isinstance(image_id, str) or _IMAGE_ID_RE.fullmatch(image_id) is None:
            raise DockerLiveRestoreReconcilerV2Error(
                f"Docker container has malformed image identity: {container_id}"
            )
        if image_id not in image_ids:
            raise DockerLiveRestoreReconcilerV2Error("Container references an unlisted image")
        mounts = row.get("Mounts")
        if isinstance(mounts, list) and all(isinstance(mount, dict) for mount in mounts):
            # Docker may reorder Mounts. Compare complete records as a multiset,
            # retaining duplicate records, value types and nested array order.
            mounts = sorted(mounts, key=lambda mount: json.dumps(
                mount, sort_keys=True, separators=(",", ":"),
            ))
        normalized_rows.append(
            {
                "graph_data": graph_data,
                "graph_name": graph.get("Name") if isinstance(graph, dict) else None,
                "id": container_id,
                "image": image_id,
                "pid": pid,
                "started_at": state.get("StartedAt"),
                "mounts": mounts,
            }
        )
    task_ids = _validated_ids(
        _checked_output(
            runner,
            ["ctr", "-n", "moby", "tasks", "list", "-q"],
            label="containerd task inventory",
        ),
        image=False,
        field="containerd task inventory",
    )
    if task_ids != running_ids:
        raise DockerLiveRestoreReconcilerV2Error(
            "containerd task inventory differs from running Docker containers"
        )
    server_version = _checked_output(
        runner,
        ["docker", "version", "--format", "{{.Server.Version}}"],
        label="Docker server version",
    ).strip()
    if not server_version:
        raise DockerLiveRestoreReconcilerV2Error("Docker server version is empty")
    payload = json.dumps(
        {
            "containers": sorted(normalized_rows, key=lambda row: str(row["id"])),
            "images": sorted(({
                "id": row["Id"], "rootfs": row["RootFS"],
                "repo_tags": sorted(row.get("RepoTags") or []),
                "repo_digests": sorted(row.get("RepoDigests") or []),
            } for row in image_rows), key=lambda row: row["id"]),
            "volumes": sorted(({
                key: row.get(key) for key in (
                    "Name", "Driver", "Mountpoint", "Options", "Labels", "Scope", "CreatedAt",
                )
            } for row in volume_rows), key=lambda row: row["Name"]),
            "tasks": task_ids,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return DockerRuntimeSnapshot(
        container_ids=container_ids,
        image_ids=image_ids,
        volume_names=volume_names,
        running_container_ids=running_ids,
        task_ids=task_ids,
        server_version=server_version,
        manifest_hash="sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest(),
    )


def _live_restore_enabled(runner: CommandRunner) -> bool:
    raw = _checked_output(
        runner,
        ["docker", "info", "--format", "{{json .LiveRestoreEnabled}}"],
        label="Docker live-restore status",
    ).strip()
    if raw not in {"true", "false"}:
        raise DockerLiveRestoreReconcilerV2Error(
            "Docker live-restore status is malformed"
        )
    return raw == "true"


def _read_config(path: Path) -> tuple[dict[str, object], Optional[bytes], int]:
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return {}, None, 0o644
    if not stat.S_ISREG(metadata.st_mode) or path.is_symlink():
        raise DockerLiveRestoreReconcilerV2Error(
            "Docker daemon configuration is not a regular file"
        )
    expected_uid = os.geteuid()
    expected_gid = os.getegid()
    if (
        metadata.st_uid != expected_uid
        or metadata.st_gid != expected_gid
        or metadata.st_mode & 0o022
    ):
        raise DockerLiveRestoreReconcilerV2Error(
            "Docker daemon configuration has unsafe ownership or mode"
        )
    if metadata.st_size > _MAX_CONFIG_BYTES:
        raise DockerLiveRestoreReconcilerV2Error(
            "Docker daemon configuration exceeds its size bound"
        )
    raw = path.read_bytes()
    try:
        document = json.loads(raw)
    except (UnicodeError, ValueError) as exc:
        raise DockerLiveRestoreReconcilerV2Error(
            "Docker daemon configuration is invalid JSON"
        ) from exc
    if not isinstance(document, dict):
        raise DockerLiveRestoreReconcilerV2Error(
            "Docker daemon configuration must be a JSON object"
        )
    return document, raw, stat.S_IMODE(metadata.st_mode)


def _atomic_write(path: Path, payload: bytes, *, mode: int) -> None:
    parent = path.parent
    if parent.is_symlink() or not parent.is_dir():
        raise DockerLiveRestoreReconcilerV2Error(
            "Docker daemon configuration directory is unsafe"
        )
    temporary = parent / f".{path.name}.leadpoet-v2.{os.getpid()}.tmp"
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        mode,
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, mode)
        os.chown(temporary, os.geteuid(), os.getegid())
        os.replace(temporary, path)
        directory_fd = os.open(parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _configure_live_restore(
    runner: CommandRunner,
    *,
    config_path: Path,
) -> bool:
    document, original, mode = _read_config(config_path)
    configured = document.get("live-restore")
    if configured is not None and not isinstance(configured, bool):
        raise DockerLiveRestoreReconcilerV2Error(
            "Docker live-restore configuration must be boolean"
        )
    if configured is True and _live_restore_enabled(runner):
        return False
    updated = dict(document)
    updated["live-restore"] = True
    payload = (json.dumps(updated, sort_keys=True, indent=2) + "\n").encode("utf-8")
    validation_path = config_path.parent / (
        f".{config_path.name}.leadpoet-v2-validate.{os.getpid()}"
    )
    _atomic_write(validation_path, payload, mode=mode)
    try:
        _checked_output(
            runner,
            ["dockerd", "--validate", "--config-file", str(validation_path)],
            label="Docker live-restore configuration validation",
        )
    finally:
        validation_path.unlink(missing_ok=True)
    if original is not None:
        backup_hash = hashlib.sha256(original).hexdigest()
        backup_path = config_path.parent / (
            f"{config_path.name}.leadpoet-v2-backup-{backup_hash}"
        )
        if not backup_path.exists():
            _atomic_write(backup_path, original, mode=0o600)
    _atomic_write(config_path, payload, mode=mode)
    try:
        _checked_output(
            runner,
            ["systemctl", "reload", "docker.service"],
            label="Docker live-restore configuration reload",
        )
    except DockerLiveRestoreReconcilerV2Error:
        if original is None:
            config_path.unlink(missing_ok=True)
        else:
            _atomic_write(config_path, original, mode=mode)
        raise
    return True


def reconcile_live_docker_daemon_v2(
    *,
    runner: CommandRunner = _run,
    config_path: Path = Path("/etc/docker/daemon.json"),
    ready_attempts: int = 30,
    sleeper: Sleeper = time.sleep,
    require_enabled: bool = False,
    pre_restart_check: Optional[Callable[[], None]] = None,
) -> DockerDaemonReconcileResult:
    """Reload dockerd state while proving all live containers are unchanged."""

    if ready_attempts < 1 or ready_attempts > 300:
        raise DockerLiveRestoreReconcilerV2Error(
            "Docker readiness attempts must be between 1 and 300"
        )
    before = _runtime_snapshot(runner)
    if require_enabled:
        document, _raw, _mode = _read_config(config_path)
        if document.get("live-restore") is not True or not _live_restore_enabled(runner):
            raise DockerLiveRestoreReconcilerV2Error("Live restore must already be enabled")
        config_changed = False
    else:
        config_changed = _configure_live_restore(runner, config_path=config_path)
    for attempt in range(1, ready_attempts + 1):
        if _live_restore_enabled(runner):
            break
        if attempt == ready_attempts:
            raise DockerLiveRestoreReconcilerV2Error(
                "Docker did not activate live-restore after configuration reload"
            )
        sleeper(1)
    if _checked_output(
        runner,
        ["systemctl", "is-active", "containerd.service"],
        label="containerd service status",
    ).strip() != "active":
        raise DockerLiveRestoreReconcilerV2Error(
            "containerd is not active before dockerd reconciliation"
        )
    if pre_restart_check is not None:
        pre_restart_check()
    _checked_output(
        runner,
        ["systemctl", "restart", "docker.service"],
        label="Docker daemon reconciliation",
    )
    after: Optional[DockerRuntimeSnapshot] = None
    last_error: Optional[Exception] = None
    for attempt in range(1, ready_attempts + 1):
        try:
            if not _live_restore_enabled(runner):
                raise DockerLiveRestoreReconcilerV2Error(
                    "Docker live-restore disabled after daemon reconciliation"
                )
            after = _runtime_snapshot(runner)
            break
        except DockerLiveRestoreReconcilerV2Error as exc:
            last_error = exc
        if attempt < ready_attempts:
            sleeper(1)
    if after is None:
        raise DockerLiveRestoreReconcilerV2Error(
            "Docker runtime did not recover after daemon reconciliation"
        ) from last_error
    if after != before:
        raise DockerLiveRestoreReconcilerV2Error(
            "Docker image/container identity changed during daemon reconciliation"
        )
    if _checked_output(
        runner,
        ["systemctl", "is-active", "containerd.service"],
        label="containerd service status",
    ).strip() != "active":
        raise DockerLiveRestoreReconcilerV2Error(
            "containerd changed state during dockerd reconciliation"
        )
    return DockerDaemonReconcileResult(
        config_changed=config_changed,
        container_count=len(after.container_ids),
        image_count=len(after.image_ids),
        manifest_hash=after.manifest_hash,
    )


def _restart_inputs(
    runner: CommandRunner, config_path: Path, proc_root: Path,
) -> dict[str, object]:
    """Refuse a restart that could activate pending config, unit or binary edits."""
    service = _service_identity(runner, "docker.service")
    raw = _checked_output(runner, [
        "systemctl", "show", "docker.service", "--all", "--property=FragmentPath",
        "--property=DropInPaths", "--property=NeedDaemonReload",
        "--property=ExecMainStartTimestampMonotonic", "--no-pager",
        "--property=EnvironmentFiles", "--property=ExecStart", "--property=ExecStartPre",
        "--property=ExecStartPost", "--property=ExecStop", "--property=ExecStopPost",
        "--property=ExecCondition", "--property=PassEnvironment", "--property=KillMode",
    ], label="Docker restart inputs")
    fields: dict[str, str] = {}
    environment_files: list[str] = []
    startup_commands: list[str] = []
    for line in raw.splitlines():
        key, separator, value = line.partition("=")
        if separator and key == "EnvironmentFiles":
            environment_files.append(value)
            continue
        if separator and key == "ExecStartPre":
            startup_commands.append(value)
            continue
        if not separator or key in fields:
            raise DockerLiveRestoreReconcilerV2Error("Docker restart inputs are malformed")
        fields[key] = value
    # systemd 252 omits empty execution-command arrays even with --all.
    # Nonempty hooks are always printed and remain unsupported below.
    for key in ("ExecStartPost", "ExecStop", "ExecStopPost", "ExecCondition"):
        fields.setdefault(key, "")
    if set(fields) != {"FragmentPath", "DropInPaths", "NeedDaemonReload",
                       "ExecMainStartTimestampMonotonic", "ExecStart", "ExecStartPost",
                       "ExecStop", "ExecStopPost", "ExecCondition", "PassEnvironment", "KillMode"} or fields["NeedDaemonReload"] != "no":
        raise DockerLiveRestoreReconcilerV2Error("Docker unit has pending or unknown changes")
    if any(fields[key] for key in ("ExecStartPost", "ExecStop", "ExecStopPost",
                                  "ExecCondition", "PassEnvironment")) or fields["KillMode"] != "process":
        raise DockerLiveRestoreReconcilerV2Error("Docker unit has an unsupported launch or stop shape")
    monotonic = fields["ExecMainStartTimestampMonotonic"]
    boot = [line.split()[1] for line in (proc_root / "stat").read_text().splitlines()
            if line.startswith("btime ")]
    if not monotonic.isdecimal() or int(monotonic) <= 0 or len(boot) != 1 or not boot[0].isdecimal():
        raise DockerLiveRestoreReconcilerV2Error("Docker start time is unavailable")
    started_ns = int(boot[0]) * 1_000_000_000 + int(monotonic) * 1_000
    document, raw_config, _mode = _read_config(config_path)
    if raw_config is None or document.get("live-restore") is not True or not _live_restore_enabled(runner):
        raise DockerLiveRestoreReconcilerV2Error("Live restore must already be enabled")

    def identity(path: Path, maximum_bytes: int = _MAX_CONFIG_BYTES, *, require_old: bool = True) -> tuple[object, ...]:
        metadata = path.lstat()
        if (not path.is_absolute() or not stat.S_ISREG(metadata.st_mode)
                or metadata.st_uid != os.geteuid() or metadata.st_mode & 0o022
                or (require_old and metadata.st_mtime_ns >= started_ns) or metadata.st_size > maximum_bytes):
            raise DockerLiveRestoreReconcilerV2Error("Docker restart file has pending or unsafe changes")
        return (str(path), metadata.st_dev, metadata.st_ino, metadata.st_mtime_ns,
                metadata.st_size, stat.S_IMODE(metadata.st_mode), metadata.st_uid,
                metadata.st_gid, hashlib.sha256(path.read_bytes()).hexdigest())

    unit_paths = [Path(fields["FragmentPath"]), *map(Path, fields["DropInPaths"].split())]
    if not fields["FragmentPath"]:
        raise DockerLiveRestoreReconcilerV2Error("Docker unit path is unavailable")
    files = (identity(config_path), *(identity(path) for path in unit_paths))
    executable = proc_root / str(service.main_pid) / "exe"
    target = os.readlink(executable)
    if not target.startswith("/") or target.endswith(" (deleted)"):
        raise DockerLiveRestoreReconcilerV2Error("Docker running binary has changed")
    binary = Path(target)
    binary_identity = identity(binary, 256 * 1024 * 1024)
    running = executable.stat()
    if ((running.st_dev, running.st_ino) != (binary_identity[1], binary_identity[2])
            or hashlib.sha256(executable.read_bytes()).hexdigest() != binary_identity[-1]):
        raise DockerLiveRestoreReconcilerV2Error("Docker disk binary differs from the running binary")

    def commands(values: list[str]) -> tuple[tuple[str, str, str], ...]:
        result = []
        for value in values:
            groups = re.findall(r"\{ ([^{}]*) \}", value)
            if not groups or " ".join("{ " + group + " }" for group in groups) != value:
                raise DockerLiveRestoreReconcilerV2Error("Docker systemd command is malformed")
            for group in groups:
                parts = [part.partition("=") for part in group.split(" ; ")]
                command = {key: item for key, separator, item in parts if separator}
                if (len(command) != len(parts) or set(command) != {
                        "path", "argv[]", "ignore_errors", "start_time", "stop_time", "pid", "code", "status"}):
                    raise DockerLiveRestoreReconcilerV2Error("Docker systemd command has unknown controls")
                result.append((command["path"], command["argv[]"], command["ignore_errors"]))
        return tuple(result)

    launch = commands([fields["ExecStart"]])
    prepare = commands(startup_commands)
    if launch != ((str(binary), f"{binary} -H fd:// --containerd=/run/containerd/containerd.sock "
                  "$OPTIONS $DOCKER_STORAGE_OPTIONS $DOCKER_ADD_RUNTIMES", "no"),) or prepare != (
            ("/bin/mkdir", "/bin/mkdir -p /run/docker", "no"),
            (str(_RUNTIME_SETUP_SCRIPT), str(_RUNTIME_SETUP_SCRIPT), "no")):
        raise DockerLiveRestoreReconcilerV2Error("Docker startup commands are not the supported direct launch")
    expected_env = [f"{path} (ignore_errors=yes)" for path in (*_DOCKER_ENV_FILES, _RUNTIME_ENV_FILE)]
    if environment_files != expected_env:
        raise DockerLiveRestoreReconcilerV2Error("Docker startup environment paths are unbound")
    files = (*files, *(identity(path) for path in _DOCKER_ENV_FILES), identity(_RUNTIME_SETUP_SCRIPT))
    directory = _RUNTIME_DIRECTORY.lstat()
    if (not stat.S_ISDIR(directory.st_mode) or _RUNTIME_DIRECTORY.resolve() != _RUNTIME_DIRECTORY
            or directory.st_uid != os.geteuid() or directory.st_mode & 0o022
            or any(_RUNTIME_DIRECTORY.iterdir())):
        raise DockerLiveRestoreReconcilerV2Error("Custom Docker runtime setup is unsupported for live reclaim")
    generated_identity = identity(_RUNTIME_ENV_FILE, require_old=False)
    if _RUNTIME_ENV_FILE.read_bytes() != b'DOCKER_ADD_RUNTIMES=""\n':
        raise DockerLiveRestoreReconcilerV2Error("Generated Docker runtime environment is not empty")
    # ExecStartPre recreates this file; bind content, ownership and mode, not inode/time.
    generated = (generated_identity[0], *generated_identity[4:])
    gateway = inspect_exact_host_gateway_runtime(proc_root=proc_root)
    host = gateway["gateway_process"]
    host_identity = None
    if host is not None:
        raw_stat = (proc_root / str(host["pid"]) / "stat").read_text()
        closing = raw_stat.rfind(")")
        stat_fields = raw_stat[closing + 2:].split()
        if closing < 0 or len(raw_stat) > 4096 or len(stat_fields) < 20:
            raise DockerLiveRestoreReconcilerV2Error("Gateway start identity is invalid")
        ticks = stat_fields[19]
        if not ticks.isdecimal() or int(ticks) <= 0:
            raise DockerLiveRestoreReconcilerV2Error("Gateway start identity is invalid")
        host_identity = (host["pid"], ticks)
    return {"docker_service": service, "files": files, "binary": binary_identity,
            "launch": launch, "prepare": prepare, "generated_environment": generated,
            "runtime_directory": (directory.st_dev, directory.st_ino, directory.st_mode, directory.st_uid, directory.st_gid),
            "containerd_service": _service_identity(runner, "containerd.service"),
            "host_gateway": host_identity}


def _pending_reconcile(root: Path, *, create: bool = False, clear: bool = False) -> bool:
    """Keep one durable interruption flag inside the already validated Docker root."""
    metadata = root.lstat()
    if (not root.is_absolute() or root.resolve() != root or not stat.S_ISDIR(metadata.st_mode)
            or metadata.st_uid != os.geteuid() or metadata.st_mode & 0o022):
        raise DockerLiveRestoreReconcilerV2Error("Pending-reconcile directory is unsafe")
    marker = root / _PENDING_RECONCILE
    try:
        metadata = marker.lstat()
    except FileNotFoundError:
        if not create:
            return False
        payload = b'{"pending":true}\n'
        # Atomic publication means interruption cannot expose a half-written flag.
        temporary = root / f"{_PENDING_RECONCILE}.{os.getpid()}.tmp"
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL
                             | getattr(os, "O_NOFOLLOW", 0), 0o600)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.link(temporary, marker, follow_symlinks=False)
        finally:
            temporary.unlink()
        directory = os.open(root, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        return True
    if (not stat.S_ISREG(metadata.st_mode) or metadata.st_uid != os.geteuid()
            or stat.S_IMODE(metadata.st_mode) != 0o600 or metadata.st_size > 4096
            or marker.read_bytes() != b'{"pending":true}\n'):
        raise DockerLiveRestoreReconcilerV2Error("Pending reconcile has unsafe or changed identity")
    if clear:
        marker.unlink()
        directory = os.open(root, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    return True


def reclaim_live_docker_overlay_state_v2(
    *, lock_file: Path, admission_lock_file: Path, lock_owner_pid: int,
    runner: CommandRunner = _run, config_path: Path = Path("/etc/docker/daemon.json"),
    proc_root: Path = Path("/proc"), expected_root: Path = Path("/var/lib/docker"),
    ready_attempts: int = 30, sleeper: Sleeper = time.sleep,
) -> dict[str, object]:
    """Reclaim export leaks under existing locks without dropping retained objects."""
    if not 1 <= ready_attempts <= 300:
        raise DockerLiveRestoreReconcilerV2Error("Docker readiness attempts must be between 1 and 300")
    def locked_and_idle() -> None:
        verify_docker_operation_locks_v2(lock_file=lock_file,
            admission_lock_file=admission_lock_file, owner_pid=lock_owner_pid,
            proc_root=proc_root)
        if find_foreign_docker_operations(proc_root=proc_root):
            raise DockerLiveRestoreReconcilerV2Error("Foreign Docker operation prevents reclaim")

    locked_and_idle()
    before = _runtime_snapshot(runner)
    inputs = _restart_inputs(runner, config_path, proc_root)

    def unchanged() -> None:
        locked_and_idle()
        if _restart_inputs(runner, config_path, proc_root) != inputs or _runtime_snapshot(runner) != before:
            raise DockerLiveRestoreReconcilerV2Error("Docker runtime or restart inputs changed before mutation")

    _checked_output(runner, [str(inputs["binary"][0]), "--validate", "--config-file", str(config_path)],
                    label="Docker current configuration validation")
    total, active = _build_cache_counts(runner)
    if active:
        raise DockerLiveRestoreReconcilerV2Error("Docker build cache is active")
    if total:
        unchanged()
        _checked_output(runner, ["docker", "builder", "prune", "--all", "--force"],
                        label="Idle Docker build-cache prune")
    unchanged()
    before_audit = audit_stale_docker_overlay_state_v2(runner=runner, expected_root=expected_root)
    pending = _pending_reconcile(expected_root)
    raw_mutation_started = False
    def note_mutation() -> None:
        nonlocal raw_mutation_started
        if not raw_mutation_started:
            _pending_reconcile(expected_root, create=True)
        raw_mutation_started = True

    def verify_refreshed(*, require_clean: bool = True) -> bool:
        locked_and_idle()
        after_inputs = _restart_inputs(runner, config_path, proc_root)
        if (_runtime_snapshot(runner) != before or any(
            after_inputs[key] != value for key, value in inputs.items() if key != "docker_service"
        )):
            raise DockerLiveRestoreReconcilerV2Error("Retained Docker runtime or restart inputs changed")
        after_audit = audit_stale_docker_overlay_state_v2(runner=runner, expected_root=expected_root)
        if any(after_audit[key] != before_audit[key] for key in (
                "active_manifest_hash", "active_container_count", "active_image_count",
                "active_layer_count", "active_mount_count", "active_overlay_dir_count",
                "docker_root", "metadata_layout")):
            raise DockerLiveRestoreReconcilerV2Error("Docker active graph changed after refresh")
        clean = not any(after_audit[key] for key in (
            "stale_layer_record_count", "stale_mount_record_count",
            "stale_overlay_dir_count", "stale_overlay_link_count")) and (
            after_audit["mounted_overlay_count"] == after_audit["active_mount_count"])
        if not clean and require_clean:
            raise DockerLiveRestoreReconcilerV2Error("Docker stale state returned after refresh")
        if clean:
            _pending_reconcile(expected_root, clear=True)
        return clean

    try:
        reclaimed = reclaim_stale_docker_overlay_mounts_v2(
            runner=runner, expected_root=expected_root, pre_mutation_check=unchanged,
            mutation_started=note_mutation,
        )
    except BaseException as original:
        if raw_mutation_started or pending:
            try:
                unchanged()
                reconcile_live_docker_daemon_v2(runner=runner, config_path=config_path,
                    ready_attempts=ready_attempts, sleeper=sleeper, require_enabled=True,
                    pre_restart_check=unchanged)
                verify_refreshed(require_clean=False)
            except BaseException as recovery_error:
                raise DockerLiveRestoreReconcilerV2Error(
                    "Raw reclaim failed; guarded daemon recovery is incomplete "
                    f"({type(recovery_error).__name__})"
                ) from original
        raise
    restart = pending or any((reclaimed.reclaimed_layer_record_count, reclaimed.reclaimed_mount_count,
                   reclaimed.reclaimed_mount_record_count, reclaimed.reclaimed_overlay_dir_count,
                   reclaimed.reclaimed_overlay_link_count))
    if restart:
        reconcile_live_docker_daemon_v2(runner=runner, config_path=config_path,
            ready_attempts=ready_attempts, sleeper=sleeper, require_enabled=True,
            pre_restart_check=unchanged)
    verify_refreshed()
    return {**reclaimed.as_dict(), "daemon_reconciled": restart,
            "preserved_runtime_manifest_hash": before.manifest_hash}


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ready-attempts", type=int, default=30)
    parser.add_argument("--reclaim-stale", action="store_true")
    parser.add_argument("--docker-lock-file", type=Path)
    parser.add_argument("--docker-admission-lock-file", type=Path)
    parser.add_argument("--docker-lock-owner-pid", type=int)
    args = parser.parse_args(argv)
    if os.geteuid() != 0:
        raise SystemExit("Docker live-restore reconciliation must run as root")
    try:
        if args.reclaim_stale:
            if not args.docker_lock_file or not args.docker_admission_lock_file or not args.docker_lock_owner_pid:
                parser.error("--reclaim-stale requires both explicit lock paths and their owner PID")
            result = reclaim_live_docker_overlay_state_v2(
                lock_file=args.docker_lock_file, admission_lock_file=args.docker_admission_lock_file,
                lock_owner_pid=args.docker_lock_owner_pid, ready_attempts=args.ready_attempts,
            )
        else:
            result = reconcile_live_docker_daemon_v2(ready_attempts=args.ready_attempts).as_dict()
    except (DockerLiveRestoreReconcilerV2Error, DockerStaleMountReclaimerV2Error,
            DockerZeroRuntimeReconcilerV2Error, OSError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
