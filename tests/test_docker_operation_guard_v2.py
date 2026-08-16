from __future__ import annotations

from pathlib import Path
import sys

import pytest

from validator_tee.host.docker_operation_guard_v2 import (
    DockerOperationGuardV2Error,
    find_foreign_docker_operations,
    inspect_exact_host_gateway_runtime,
    main,
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


def test_exact_host_gateway_scan_matches_only_production_launcher_argv(
    tmp_path: Path,
) -> None:
    proc_root = tmp_path / "proc"
    proc_root.mkdir()
    _process(
        proc_root,
        pid=20,
        parent_pid=1,
        argv=(
            "/home/ec2-user/venv311/bin/python3",
            "-u",
            "-m",
            "gateway.main",
        ),
    )

    report = inspect_exact_host_gateway_runtime(proc_root=proc_root)

    assert report == {
        "schema_version": "leadpoet.host_gateway_process_guard.v1",
        "status": "live",
        "gateway_process": {"pid": 20, "command": "python3"},
        "scanned_entry_count": 1,
        "scanned_process_count": 1,
    }


def test_exact_host_gateway_scan_rejects_near_matches(tmp_path: Path) -> None:
    proc_root = tmp_path / "proc"
    proc_root.mkdir()
    near_matches = (
        ("python3", "-m", "gateway.main"),
        ("python3", "-u", "-m", "gateway.main", "--port", "8080"),
        ("python3", "-u", "-m", "gateway.main.worker"),
        ("bash", "-u", "-m", "gateway.main"),
        ("pgrep", "-f", "python3 -u -m gateway.main"),
    )
    for pid, argv in enumerate(near_matches, start=20):
        _process(proc_root, pid=pid, parent_pid=1, argv=argv)

    report = inspect_exact_host_gateway_runtime(proc_root=proc_root)

    assert report["status"] == "absent"
    assert report["gateway_process"] is None
    assert report["scanned_process_count"] == len(near_matches)


def test_exact_host_gateway_scan_fails_closed_when_process_bound_is_exceeded(
    tmp_path: Path,
) -> None:
    proc_root = tmp_path / "proc"
    proc_root.mkdir()
    _process(proc_root, pid=20, parent_pid=1, argv=("sleep", "1"))
    _process(proc_root, pid=21, parent_pid=1, argv=("sleep", "2"))

    with pytest.raises(
        DockerOperationGuardV2Error,
        match="process table exceeded the bounded host gateway scan",
    ):
        inspect_exact_host_gateway_runtime(
            proc_root=proc_root,
            max_process_entries=1,
        )


@pytest.mark.parametrize("timeout_seconds", [float("nan"), float("inf")])
def test_exact_host_gateway_scan_rejects_non_finite_deadlines(
    tmp_path: Path,
    timeout_seconds: float,
) -> None:
    proc_root = tmp_path / "proc"
    proc_root.mkdir()

    with pytest.raises(
        DockerOperationGuardV2Error,
        match="timeout_seconds must be positive",
    ):
        inspect_exact_host_gateway_runtime(
            proc_root=proc_root,
            timeout_seconds=timeout_seconds,
        )


@pytest.mark.parametrize("timeout_seconds", ["nan", "inf"])
def test_exact_host_gateway_cli_rejects_non_finite_scan_deadlines(
    monkeypatch: pytest.MonkeyPatch,
    timeout_seconds: str,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "docker_operation_guard_v2.py",
            "--detect-exact-host-gateway",
            "--scan-timeout-seconds",
            timeout_seconds,
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        main()

    assert exc_info.value.code == 2
