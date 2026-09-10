"""Fail-closed ownership check for the canonical Arena validator restart."""
from __future__ import annotations

import fnmatch
import os
import re
import stat
from pathlib import Path
from typing import Any, List, Mapping, Optional, Sequence, Tuple


def validate_legacy_container_inventory(
    items: Sequence[Mapping[str, Any]], source_root: Path
) -> Mapping[str, Any]:
    """Bind the retired coordinator and any discovered FF workers to one image."""

    if not items:
        return {"main_id": "", "main_pid": 0, "workers": []}
    by_name = {item.get("Name"): item for item in items}
    if len(by_name) != len(items) or "/leadpoet-validator-main" not in by_name:
        raise RuntimeError("legacy validator container identity is ambiguous")
    main = by_name.pop("/leadpoet-validator-main")
    config = main.get("Config") or {}
    revision = (config.get("Labels") or {}).get(
        "org.opencontainers.image.revision"
    )
    image = main.get("Image")
    weights_source = str(source_root / "validator_weights")

    def common(item: Mapping[str, Any]) -> bool:
        item_config = item.get("Config") or {}
        mounts = item.get("Mounts") or []
        weight_mounts = [
            mount
            for mount in mounts
            if mount.get("Type") == "bind"
            and mount.get("Source") == weights_source
            and mount.get("Destination") == "/app/validator_weights"
            and mount.get("RW") is True
        ]
        return (
            (item_config.get("Entrypoint") or [])
            == ["python3", "neurons/validator.py"]
            and item_config.get("WorkingDir") == "/app"
            and item.get("Image") == image
            and (item_config.get("Labels") or {}).get(
                "org.opencontainers.image.revision"
            )
            == revision
            and len(weight_mounts) == 1
        )

    def runtime_identity(item: Mapping[str, Any]) -> Tuple[str, int]:
        container_id = item.get("Id")
        state = item.get("State") or {}
        running = state.get("Running")
        pid = state.get("Pid")
        if (
            not isinstance(container_id, str)
            or re.fullmatch(r"[0-9a-f]{64}", container_id) is None
            or not isinstance(running, bool)
            or isinstance(pid, bool)
            or not isinstance(pid, int)
            or pid < 0
            or (running and pid == 0)
        ):
            raise RuntimeError("legacy validator container runtime identity is invalid")
        return container_id, pid if running else 0

    def coordinator_command_is_valid(command: Any) -> bool:
        if not isinstance(command, list) or len(command) % 2:
            return False
        allowed = {
            "--netuid",
            "--subtensor_network",
            "--wallet_name",
            "--wallet_hotkey",
            "--container-id",
            "--total-containers",
            "--mode",
        }
        parsed = {}
        for offset in range(0, len(command), 2):
            flag, value = command[offset : offset + 2]
            if (
                not isinstance(flag, str)
                or flag not in allowed
                or flag in parsed
                or not isinstance(value, str)
                or not value
            ):
                return False
            parsed[flag] = value
        return (
            set(parsed) == allowed
            and parsed["--netuid"] == "71"
            and parsed["--subtensor_network"] == "finney"
            and parsed["--container-id"] == "0"
            and parsed["--total-containers"] == "1"
            and parsed["--mode"] == "coordinator"
        )

    if (
        not isinstance(image, str)
        or re.fullmatch(r"sha256:[0-9a-f]{64}", image) is None
        or not isinstance(revision, str)
        or re.fullmatch(r"[0-9a-f]{40}", revision) is None
        or not common(main)
        or not coordinator_command_is_valid(config.get("Cmd"))
    ):
        raise RuntimeError("legacy validator container identity is invalid")
    workers = []
    for name, item in by_name.items():
        match = re.fullmatch(r"/leadpoet-ff-worker-(10|[1-9])", str(name))
        if not match:
            raise RuntimeError("legacy validator worker identity is ambiguous")
        worker_id = int(match.group(1))
        command = (item.get("Config") or {}).get("Cmd") or []
        expected_command = [
            "--mode",
            "fulfillment_worker",
            "--container-id",
            str(worker_id),
        ]
        host_config = item.get("HostConfig") or {}
        if (
            not common(item)
            or command != expected_command
            or item.get("Args") != ["neurons/validator.py", *expected_command]
            or len(item.get("Mounts") or []) != 1
            or (host_config.get("RestartPolicy") or {}).get("Name")
            != "unless-stopped"
            or host_config.get("NetworkMode") != "host"
            or host_config.get("Privileged") is not False
            or host_config.get("Devices") not in (None, [])
            or host_config.get("CapAdd") not in (None, [])
        ):
            raise RuntimeError("legacy validator worker identity is invalid")
        container_id, pid = runtime_identity(item)
        workers.append(
            {
                "worker_id": worker_id,
                "container_id": container_id,
                "pid": pid,
            }
        )
    workers.sort(key=lambda item: item["worker_id"])
    main_id, main_pid = runtime_identity(main)
    return {
        "main_id": main_id,
        "main_pid": main_pid,
        "workers": workers,
    }


def legacy_fulfillment_queue_is_quiescent(weights_dir: Path) -> bool:
    """Return true only when every old worker work file has a regular result."""

    root = Path(weights_dir)
    try:
        metadata = root.lstat()
        if not stat.S_ISDIR(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
            return False
        with os.scandir(str(root)) as entries:
            work_files = [
                entry
                for entry in entries
                if fnmatch.fnmatchcase(
                    entry.name, "fulfillment_worker_*_work_*.json"
                )
            ]
        for work_file in work_files:
            if not work_file.is_file(follow_symlinks=False):
                return False
            result = root / work_file.name.replace("_work_", "_results_", 1)
            result_metadata = os.stat(str(result), follow_symlinks=False)
            if not stat.S_ISREG(result_metadata.st_mode):
                return False
    except OSError:
        return False
    return True


def _stat(entry: Path) -> List[str]:
    return (entry / "stat").read_text().split()


def _is_descendant(proc_root: Path, pid: int, ancestor: int) -> bool:
    seen = set()
    while pid > 1 and pid not in seen:
        if pid == ancestor:
            return True
        seen.add(pid)
        try:
            pid = int(_stat(proc_root / str(pid))[3])
        except (OSError, ValueError, IndexError):
            return False
    return False


def find_validator_process(
    source_root: Path, current_root: Path, service_pid: int, proc_root: Path = Path("/proc"),
    ignored_tree_roots: Sequence[int] = (),
) -> Optional[Tuple[int, str]]:
    source_root = source_root.resolve()
    current_root = current_root.resolve(strict=False)
    exact: List[Tuple[int, str]] = []
    suspicious: List[int] = []
    for entry in proc_root.iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        if any(
            root > 0 and _is_descendant(proc_root, pid, root)
            for root in ignored_tree_roots
        ):
            continue
        try:
            args = [
                value.decode("utf-8", "surrogateescape")
                for value in (entry / "cmdline").read_bytes().split(b"\0")
                if value
            ]
            cwd = Path(os.readlink(entry / "cwd")).resolve()
            hits = [
                value
                for value in args
                if value.endswith("neurons/validator.py")
                or value.endswith("scripts/run_arena_validator.py")
            ]
            if not hits:
                continue
            resolved = [
                (cwd / Path(value)).resolve()
                if not Path(value).is_absolute()
                else Path(value).resolve()
                for value in hits
            ]
            source_owned = all(source_root in (path, *path.parents) for path in resolved)
            supervised = pid == service_pid and all(
                current_root in (path, *path.parents) for path in resolved
            )
            if source_owned or supervised:
                exact.append((pid, (entry / "stat").read_text().split()[21]))
            else:
                suspicious.append(pid)
        except (OSError, ValueError, IndexError):
            continue
    if suspicious or len(exact) > 1:
        raise RuntimeError("validator process ownership is ambiguous")
    return exact[0] if exact else None


def find_owned_auxiliary(
    source_root: Path, kind: str, proc_root: Path = Path("/proc")
) -> Optional[Tuple[int, str]]:
    suffix = {
        "runner": "scripts/run_lab_arena_runner.py",
        "relay": "validator_tee.host.chain_relay_v2",
    }[kind]
    source_root = source_root.resolve()
    matches = []
    for entry in proc_root.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            args = [
                value.decode("utf-8", "surrogateescape")
                for value in (entry / "cmdline").read_bytes().split(b"\0") if value
            ]
            if kind == "runner":
                selected = any(value.endswith(suffix) for value in args)
            else:
                selected = any(value == suffix for value in args)
            if not selected:
                continue
            cwd = Path(os.readlink(entry / "cwd")).resolve()
            if cwd != source_root:
                raise RuntimeError("%s process ownership is ambiguous" % kind)
            fields = _stat(entry)
            matches.append((int(entry.name), int(fields[4])))
        except (OSError, ValueError, IndexError):
            continue
    if not matches:
        return None
    groups = {group for _, group in matches}
    if len(groups) != 1:
        raise RuntimeError("%s process ownership is ambiguous" % kind)
    owner = groups.pop() if kind == "runner" else matches[0][0]
    if kind == "relay" and len(matches) != 1:
        raise RuntimeError("relay process ownership is ambiguous")
    try:
        start = _stat(proc_root / str(owner))[21]
    except (OSError, IndexError) as exc:
        raise RuntimeError("%s process owner is unavailable" % kind) from exc
    return owner, start


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("source_root", type=Path)
    parser.add_argument("current_root", type=Path)
    parser.add_argument("service_pid", type=int)
    parser.add_argument("--proc-root", type=Path, default=Path("/proc"))
    parser.add_argument("--ignore-tree-root", type=int, action="append", default=[])
    parser.add_argument("--kind", choices=("validator", "runner", "relay"), default="validator")
    args = parser.parse_args()
    found = (find_validator_process(
        args.source_root, args.current_root, args.service_pid, args.proc_root,
        args.ignore_tree_root,
    ) if args.kind == "validator" else find_owned_auxiliary(
        args.source_root, args.kind, args.proc_root
    ))
    print(*(found or ("", "")))
