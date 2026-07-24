from __future__ import annotations

from pathlib import Path

from validator_tee.host.docker_operation_guard_v2 import (
    find_foreign_docker_operations,
)


def _process(
    proc_root: Path,
    *,
    pid: int,
    parent_pid: int,
    argv: tuple[str, ...],
) -> None:
    root = proc_root / str(pid)
    root.mkdir(parents=True)
    (root / "status").write_text(
        f"Name:\t{Path(argv[0]).name}\nPPid:\t{parent_pid}\n",
        encoding="utf-8",
    )
    (root / "cmdline").write_bytes(
        b"\0".join(value.encode("utf-8") for value in argv) + b"\0"
    )


def test_guard_does_not_treat_runner_processes_as_docker_operations(
    tmp_path: Path,
) -> None:
    proc_root = tmp_path / "proc"
    proc_root.mkdir()
    _process(
        proc_root,
        pid=10,
        parent_pid=1,
        argv=("/runner/Runner.Worker",),
    )
    _process(
        proc_root,
        pid=11,
        parent_pid=10,
        argv=("python3", "-m", "validator_tee.host.docker_operation_guard_v2"),
    )
    _process(
        proc_root,
        pid=20,
        parent_pid=1,
        argv=("/other/Runner.Worker",),
    )

    blockers = find_foreign_docker_operations(
        proc_root=proc_root,
        current_pid=11,
    )

    assert blockers == []


def test_guard_detects_mutating_builds_without_blocking_read_only_docker(
    tmp_path: Path,
) -> None:
    proc_root = tmp_path / "proc"
    proc_root.mkdir()
    _process(proc_root, pid=11, parent_pid=1, argv=("python3", "guard.py"))
    _process(
        proc_root,
        pid=21,
        parent_pid=1,
        argv=("docker", "build", "-t", "image", "."),
    )
    _process(
        proc_root,
        pid=22,
        parent_pid=1,
        argv=("sudo", "docker", "builder", "prune", "--all", "--force"),
    )
    _process(
        proc_root,
        pid=23,
        parent_pid=1,
        argv=("docker", "ps", "-a"),
    )

    blockers = find_foreign_docker_operations(
        proc_root=proc_root,
        current_pid=11,
    )

    assert blockers == [
        {"pid": 21, "command": "docker", "kind": "docker_build"},
        {"pid": 22, "command": "sudo", "kind": "docker_maintenance"},
    ]


def test_guard_detects_compose_and_buildx_build_variants(tmp_path: Path) -> None:
    proc_root = tmp_path / "proc"
    proc_root.mkdir()
    _process(proc_root, pid=11, parent_pid=1, argv=("python3", "guard.py"))
    _process(
        proc_root,
        pid=21,
        parent_pid=1,
        argv=("docker", "compose", "up", "--build"),
    )
    _process(
        proc_root,
        pid=22,
        parent_pid=1,
        argv=("docker", "buildx", "bake"),
    )

    blockers = find_foreign_docker_operations(
        proc_root=proc_root,
        current_pid=11,
    )

    assert blockers == [
        {"pid": 21, "command": "docker", "kind": "docker_build"},
        {"pid": 22, "command": "docker", "kind": "docker_build"},
    ]
