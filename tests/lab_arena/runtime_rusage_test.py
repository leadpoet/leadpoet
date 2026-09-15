from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

from lab_arena import runtime


def _finish(process: runtime._RusagePopen) -> None:
    if process.poll() is None:
        process.kill()
        process.wait(timeout=5)


def test_rusage_popen_keeps_concurrent_child_usage_separate() -> None:
    cpu_child = runtime._RusagePopen(
        [
            sys.executable,
            "-c",
            "import time\nend=time.process_time()+0.25\nwhile time.process_time()<end: pass",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    sleeping_child = runtime._RusagePopen(
        [sys.executable, "-c", "import time; time.sleep(0.1)"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        deadline = time.monotonic() + 5
        while sleeping_child.poll() is None and time.monotonic() < deadline:
            time.sleep(0.01)
        assert sleeping_child.poll() is not None
        while cpu_child.poll() is None and time.monotonic() < deadline:
            time.sleep(0.01)
        assert cpu_child.poll() is not None
    finally:
        _finish(sleeping_child)
        _finish(cpu_child)

    assert sleeping_child.child_rusage is not None
    assert cpu_child.child_rusage is not None
    sleeping_cpu, sleeping_rss = sleeping_child.child_rusage
    cpu_time, cpu_rss = cpu_child.child_rusage
    assert cpu_time >= 0.2
    assert cpu_time > sleeping_cpu + 0.1
    assert sleeping_rss > 0
    assert cpu_rss > 0


def test_stop_process_reaps_terminated_child_with_its_usage() -> None:
    process = runtime._RusagePopen(
        [
            sys.executable,
            "-c",
            (
                "import os,signal,time; "
                "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
                "os.write(1,b'ready\\n'); time.sleep(60)"
            ),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    try:
        assert process.stdout is not None
        assert process.stdout.readline() == b"ready\n"
        runtime._stop_process(process, grace=0.05)
    finally:
        _finish(process)

    assert process.returncode == -signal.SIGKILL
    assert process.child_rusage is not None
    with pytest.raises(ChildProcessError):
        os.waitpid(process.pid, os.WNOHANG)


def test_rusage_popen_communicate_keeps_exact_child_usage() -> None:
    process = runtime._RusagePopen(
        [sys.executable, "-c", "print('complete')"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    stdout, stderr = process.communicate(timeout=5)

    assert stdout == b"complete\n"
    assert stderr == b""
    assert process.returncode == 0
    assert process.child_rusage is not None


def test_missing_child_rusage_fails_without_process_wide_fallback() -> None:
    process = SimpleNamespace(child_rusage=None)

    with pytest.raises(runtime.ArenaRuntimeError, match="resource usage unavailable"):
        runtime._completed_process_rusage(process)
