"""Linux tests for the sealed process-owner helper handoff.

These tests intentionally execute the shell validator and the helper from a
sealed memfd.  They do not start any repository service or use production
paths.
"""

from __future__ import annotations

from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import shlex
import subprocess
import sys
from typing import Iterator

import pytest


REPOSITORY = Path(__file__).resolve().parents[1]
GW_RESTART = REPOSITORY / "gw_restart.sh"
PROCESS_HELPER = REPOSITORY / "scripts/manage_owned_process_group.py"
PROCESS_HELPER_FD = 195
PROOF_FD = 190
REQUIRED_SEALS = sum(
    int(getattr(fcntl, name, fallback))
    for name, fallback in (
        ("F_SEAL_WRITE", 0x0008),
        ("F_SEAL_GROW", 0x0004),
        ("F_SEAL_SHRINK", 0x0002),
        ("F_SEAL_SEAL", 0x0001),
    )
)


def _function_source() -> str:
    source = GW_RESTART.read_text(encoding="utf-8")
    marker = "verify_controller_process_helper() {\n"
    if marker not in source:
        raise AssertionError("controller process-helper validator is missing")
    start = source.index(marker)
    end_marker = "\n}\n\nstart_lab_arena_service"
    end = source.index(end_marker, start) + len("\n}")
    return source[start:end]


def _sealed_memfd(name: str, payload: bytes, *, seal: bool = True) -> int:
    fd = os.memfd_create(name, os.MFD_ALLOW_SEALING)
    try:
        os.fchmod(fd, 0o400)
        written = 0
        while written < len(payload):
            count = os.write(fd, payload[written:])
            if count <= 0:
                raise AssertionError("memfd write made no progress")
            written += count
        os.fsync(fd)
        if seal:
            fcntl.fcntl(fd, fcntl.F_ADD_SEALS, REQUIRED_SEALS)
            assert fcntl.fcntl(fd, fcntl.F_GET_SEALS) & REQUIRED_SEALS == REQUIRED_SEALS
        os.lseek(fd, 0, os.SEEK_SET)
        os.set_inheritable(fd, True)
        return fd
    except BaseException:
        os.close(fd)
        raise


@contextmanager
def _descriptor_slots(proof_fd: int | None, helper_fd: int | None) -> Iterator[None]:
    saved: dict[int, int | None] = {}
    for target in (PROOF_FD, PROCESS_HELPER_FD):
        try:
            saved[target] = os.dup(target)
        except OSError:
            saved[target] = None
    try:
        if proof_fd is not None:
            os.dup2(proof_fd, PROOF_FD, inheritable=True)
        elif saved[PROOF_FD] is not None:
            os.close(PROOF_FD)
        if helper_fd is not None:
            os.dup2(helper_fd, PROCESS_HELPER_FD, inheritable=True)
        elif saved[PROCESS_HELPER_FD] is not None:
            os.close(PROCESS_HELPER_FD)
        yield
    finally:
        for target in (PROOF_FD, PROCESS_HELPER_FD):
            previous = saved[target]
            try:
                os.close(target)
            except OSError:
                pass
            if previous is not None:
                os.dup2(previous, target, inheritable=True)
                os.close(previous)


def _run_validator(
    *,
    helper: bytes,
    helper_name: str = "leadpoet-process-helper",
    seal_helper: bool = True,
    expected_helper_hash: str | None = None,
    close_proof: bool = False,
) -> subprocess.CompletedProcess[str]:
    expected = expected_helper_hash or "sha256:" + hashlib.sha256(helper).hexdigest()
    proof_document = json.dumps(
        {"controller_process_helper_sha256": expected}, separators=(",", ":")
    ).encode("ascii")
    proof_fd = _sealed_memfd("leadpoet-miner-maintenance-proof", proof_document)
    helper_fd = _sealed_memfd(helper_name, helper, seal=seal_helper)
    try:
        with _descriptor_slots(proof_fd, helper_fd):
            if close_proof:
                os.close(PROOF_FD)
            function = _function_source()
            command = "set -eu\n" + f"GATEWAY_PYTHON_BIN={shlex.quote(sys.executable)}\n"
            command += function + "\nverify_controller_process_helper /proc/self/fd/195\n"
            environment = os.environ.copy()
            environment["GATEWAY_MINER_MAINTENANCE_PROOF_FD"] = "190"
            pass_fds = (PROCESS_HELPER_FD,) if close_proof else (PROOF_FD, PROCESS_HELPER_FD)
            return subprocess.run(
                ["bash", "-c", command],
                check=False,
                capture_output=True,
                text=True,
                env=environment,
                pass_fds=pass_fds,
            )
    finally:
        os.close(proof_fd)
        os.close(helper_fd)


@pytest.mark.skipif(sys.platform != "linux", reason="memfd and /proc tests require Linux")
def test_sealed_process_helper_accepts_exact_proof_binding() -> None:
    helper = b"#!/usr/bin/env python3\nprint('owned')\n"
    result = _run_validator(helper=helper)
    assert result.returncode == 0, result.stderr


@pytest.mark.skipif(sys.platform != "linux", reason="memfd and /proc tests require Linux")
@pytest.mark.parametrize(
    ("case", "kwargs"),
    [
        ("wrong_name", {"helper_name": "leadpoet-process-helper-extra"}),
        ("unsealed", {"seal_helper": False}),
        (
            "hash_mismatch",
            {"expected_helper_hash": "sha256:" + "0" * 64},
        ),
        ("missing_proof_fd", {"close_proof": True}),
    ],
)
def test_process_helper_rejects_unsafe_memfd_handoff(case: str, kwargs: dict[str, object]) -> None:
    helper = b"#!/usr/bin/env python3\nprint('owned')\n"
    result = _run_validator(helper=helper, **kwargs)
    assert result.returncode != 0, (case, result.stdout, result.stderr)


@pytest.mark.skipif(sys.platform != "linux", reason="memfd and /proc tests require Linux")
def test_process_helper_rejects_ordinary_symlink(tmp_path: Path) -> None:
    helper = b"#!/usr/bin/env python3\nprint('owned')\n"
    proof_fd = _sealed_memfd("leadpoet-miner-maintenance-proof", b"{}")
    helper_fd = _sealed_memfd("leadpoet-process-helper", helper)
    try:
        with _descriptor_slots(proof_fd, helper_fd):
            symlink_target = tmp_path / "process-helper-target"
            symlink_target.write_bytes(helper)
            symlink = symlink_target.with_name(symlink_target.name + ".link")
            symlink.symlink_to(symlink_target)
            try:
                function = _function_source()
                command = (
                    "set -eu\n"
                    + f"GATEWAY_PYTHON_BIN={shlex.quote(sys.executable)}\n"
                    + function
                    + f"\nverify_controller_process_helper {shlex.quote(str(symlink))}\n"
                )
                result = subprocess.run(
                    ["bash", "-c", command],
                    check=False,
                    capture_output=True,
                    text=True,
                    env=os.environ.copy(),
                    pass_fds=(PROOF_FD, PROCESS_HELPER_FD),
                )
            finally:
                symlink.unlink(missing_ok=True)
                symlink_target.unlink(missing_ok=True)
        assert result.returncode != 0
    finally:
        os.close(proof_fd)
        os.close(helper_fd)


@pytest.mark.skipif(sys.platform != "linux", reason="process-group ownership requires Linux")
def test_memfd_helper_records_and_stops_group_after_bootstrap_tree_removal(
    tmp_path: Path,
) -> None:
    process_cwd = tmp_path / "owned-cwd"
    process_cwd.mkdir()
    sidecar = tmp_path / "unrelated-sidecar.json"
    sidecar.write_text('{"keep":true}\n', encoding="utf-8")
    bootstrap_root = tmp_path / "bootstrap"
    n_minus_one_root = bootstrap_root / "controller"
    n_minus_one_root.mkdir(parents=True)
    (n_minus_one_root / "legacy-marker").write_text("four-file controller\n")
    shutil.rmtree(bootstrap_root)

    process_code = "import time; time.sleep(30)"
    process_argv = [sys.executable, "-c", process_code]
    child = subprocess.Popen(
        process_argv,
        cwd=process_cwd,
        start_new_session=True,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    helper_fd = _sealed_memfd("leadpoet-process-helper", PROCESS_HELPER.read_bytes())
    try:
        state_file = tmp_path / "owned-process.json"
        def invoke(action: str) -> subprocess.CompletedProcess[str]:
            return subprocess.run(
                [
                    sys.executable,
                    "/proc/self/fd/195",
                    action,
                    "--state-file",
                    str(state_file),
                    "--cwd",
                    str(process_cwd),
                    "--uid",
                    str(os.getuid()),
                    *(["--launch-pgid", str(child.pid)] if action == "record" else []),
                    "--",
                    *process_argv,
                ],
                check=False,
                capture_output=True,
                text=True,
                pass_fds=(PROCESS_HELPER_FD,),
            )

        with _descriptor_slots(None, helper_fd):
            recorded = invoke("record")
            assert recorded.returncode == 0, recorded.stderr
            assert state_file.is_file()
            stopped = invoke("stop")
            assert stopped.returncode == 0, stopped.stderr
        child.wait(timeout=5)
        assert not state_file.exists()
        assert not bootstrap_root.exists()
        assert sidecar.read_text(encoding="utf-8") == '{"keep":true}\n'
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=5)
        os.close(helper_fd)
