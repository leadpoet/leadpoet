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
import time
from typing import Callable, Optional, Sequence


class DockerLiveRestoreReconcilerV2Error(RuntimeError):
    """Raised when dockerd cannot be refreshed without changing live state."""


CommandRunner = Callable[[Sequence[str]], subprocess.CompletedProcess[str]]
Sleeper = Callable[[float], None]

_CONTAINER_ID_RE = re.compile(r"^[0-9a-f]{64}$")
_IMAGE_ID_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_MAX_CONFIG_BYTES = 1024 * 1024


@dataclass(frozen=True)
class DockerRuntimeSnapshot:
    container_ids: tuple[str, ...]
    image_ids: tuple[str, ...]
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
    active_image_ids: set[str] = set()
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
        active_image_ids.add(image_id)
        normalized_rows.append(
            {
                "graph_data": graph_data,
                "graph_name": graph.get("Name") if isinstance(graph, dict) else None,
                "id": container_id,
                "image": image_id,
                "pid": pid,
                "started_at": state.get("StartedAt"),
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
            "images": sorted(active_image_ids),
            "tasks": task_ids,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return DockerRuntimeSnapshot(
        container_ids=container_ids,
        image_ids=tuple(sorted(active_image_ids)),
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
) -> DockerDaemonReconcileResult:
    """Reload dockerd state while proving all live containers are unchanged."""

    if ready_attempts < 1 or ready_attempts > 300:
        raise DockerLiveRestoreReconcilerV2Error(
            "Docker readiness attempts must be between 1 and 300"
        )
    before = _runtime_snapshot(runner)
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


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ready-attempts", type=int, default=30)
    args = parser.parse_args(argv)
    if os.geteuid() != 0:
        raise SystemExit("Docker live-restore reconciliation must run as root")
    try:
        result = reconcile_live_docker_daemon_v2(ready_attempts=args.ready_attempts)
    except DockerLiveRestoreReconcilerV2Error as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(result.as_dict(), sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
