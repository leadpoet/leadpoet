import subprocess
import sys

import pytest

from scripts import run_local_restart_rehearsal


def test_exact_rehearsal_lock_rejects_concurrent_controller(
    tmp_path,
    monkeypatch,
):
    lock_path = tmp_path / "controller.lock"
    monkeypatch.setattr(
        run_local_restart_rehearsal,
        "REHEARSAL_LOCK_PATH",
        lock_path,
    )
    holder = subprocess.Popen(
        [
            sys.executable,
            "-c",
            (
                "import fcntl, pathlib, sys; "
                "path = pathlib.Path(sys.argv[1]); "
                "path.parent.mkdir(parents=True, exist_ok=True); "
                "handle = path.open('a+', encoding='utf-8'); "
                "fcntl.flock(handle.fileno(), fcntl.LOCK_EX); "
                "handle.write('external-test-owner'); handle.flush(); "
                "print('ready', flush=True); "
                "sys.stdin.readline()"
            ),
            str(lock_path),
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert holder.stdout is not None
        assert holder.stdout.readline().strip() == "ready"
        with pytest.raises(SystemExit, match="another exact local restart rehearsal"):
            with run_local_restart_rehearsal._exclusive_rehearsal_lock():
                raise AssertionError("concurrent controller acquired the lock")
    finally:
        if holder.stdin is not None:
            holder.stdin.write("\n")
            holder.stdin.flush()
        holder.wait(timeout=5)


def test_exact_rehearsal_lock_releases_after_context(tmp_path, monkeypatch):
    lock_path = tmp_path / "controller.lock"
    monkeypatch.setattr(
        run_local_restart_rehearsal,
        "REHEARSAL_LOCK_PATH",
        lock_path,
    )

    with run_local_restart_rehearsal._exclusive_rehearsal_lock():
        assert '"pid":' in lock_path.read_text(encoding="utf-8")

    with run_local_restart_rehearsal._exclusive_rehearsal_lock():
        assert '"cwd":' in lock_path.read_text(encoding="utf-8")
