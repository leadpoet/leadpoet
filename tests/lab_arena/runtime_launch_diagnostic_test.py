"""Real launcher pipes and exit status; mounting/runsc cleanup stay injected."""

import os
import signal
import sys
import time

import pytest

from lab_arena import runtime as rt
from lab_arena import runtime_host as host
from tests.lab_arena.test_lab_arena_runtime import (
    FakeClock, FakeProcess, FakeRunner, _checkpoint_spec, make_config, make_spec,
)


class RealLauncher:
    def __init__(self, script, *, fail_cleanup=False):
        self.script = script
        self.fail_cleanup = fail_cleanup
        self.calls = []
        self.process = None

    def __call__(self, argv, **kwargs):
        kind = FakeRunner._kind(argv)
        self.calls.append(kind)
        if kind == "run":
            self.process = rt._RusagePopen(
                [sys.executable, "-I", "-u", "-c", self.script, *argv], **kwargs,
            )
            return self.process
        clock = FakeClock()
        return FakeProcess(argv, clock=clock, finish_at=clock(),
                           returncode=int(self.fail_cleanup and kind == "delete"))


def test_real_launcher_failure_retains_redacted_diagnostic_and_cleanup(tmp_path):
    secret = "arbitrary-exact-private-value"
    config = make_config(tmp_path)
    spec = make_spec(tmp_path, extra_environment={"PRIVATE_SETTING": secret})
    runner = RealLauncher(
        "import sys\n"
        "print('mount cgroup failed: permission denied', file=sys.stderr)\n"
        "print('{\"msg\":\"boot child fatal: mount namespace denied\",\"api_key\":\"private-child-value\"}', file=sys.stderr)\n"
        "print('OPENROUTER_API_KEY=sk-or-v1-0123456789abcdef', file=sys.stderr)\n"
        "print('Bearer bearer-value-0123456789', file=sys.stderr)\n"
        "print('https://user:pass@host.invalid/path?secret=value', file=sys.stderr)\n"
        "print('opaque=%s', file=sys.stderr)\n" % secret
        + "sys.exit(128)\n"
    )
    with pytest.raises(rt.RuntimeHostError) as raised:
        rt.run_sandbox(config, spec, process_runner=runner)
    error = raised.value
    assert error.reason == "sandbox_launch_failed"
    assert error.launch_exit_code == 128
    assert not error.launch_timed_out
    assert "permission denied" in error.launch_stderr
    assert "boot child fatal: mount namespace denied" in error.launch_stderr
    private = host.runtime_host_private_diagnostic(error)
    for sensitive in (secret, "private-child-value", "sk-or-v1-0123456789abcdef", "bearer-value-0123456789", "user:pass", "secret=value"):
        assert sensitive not in private
    assert "permission denied" not in host.runtime_host_diagnostic(error)
    assert runner.process.poll() == 128
    assert runner.calls == ["mount", "run", "delete", "umount"]
    assert not spec.output_dir.exists()
    assert list(config.work_dir.iterdir()) == []


def test_real_startup_timeout_stops_then_drains_launcher(tmp_path, monkeypatch):
    monkeypatch.setattr(rt, "SANDBOX_STARTUP_TIMEOUT_SECONDS", 0.5)
    config = make_config(tmp_path)
    spec = _checkpoint_spec(tmp_path)
    runner = RealLauncher(
        "import signal, sys, time\n"
        "def stopped(signum, frame):\n"
        "    print('launcher stopped before pidfile', file=sys.stderr, flush=True)\n"
        "    sys.exit(7)\n"
        "signal.signal(signal.SIGTERM, stopped)\n"
        "print('waiting for container creation', file=sys.stderr, flush=True)\n"
        "time.sleep(30)\n"
    )
    started = time.monotonic()
    with pytest.raises(rt.RuntimeHostError) as raised:
        rt.run_sandbox(config, spec, process_runner=runner)
    assert time.monotonic() - started < 5
    error = raised.value
    assert error.reason == "sandbox_startup_timeout"
    assert error.launch_timed_out
    assert error.launch_exit_code == 7
    assert "launcher stopped before pidfile" in error.launch_stderr
    assert runner.process.poll() == 7
    assert runner.calls == ["mount", "run", "kill", "delete", "umount"]
    assert not spec.output_dir.exists()
    assert list(config.work_dir.iterdir()) == []


def test_real_launcher_after_pidfile_keeps_normal_result(tmp_path):
    config = make_config(tmp_path)
    spec = make_spec(tmp_path)
    runner = RealLauncher(
        "import pathlib, sys\n"
        "pidfile = next(arg.split('=', 1)[1] for arg in sys.argv if arg.startswith('--pid-file='))\n"
        "pathlib.Path(pidfile).write_text('4242')\n"
        "print('model diagnostic', file=sys.stderr)\n"
        "sys.exit(3)\n"
    )
    result = rt.run_sandbox(config, spec, process_runner=runner)
    assert result.exit_code == 3
    assert not result.timed_out
    assert result.stderr == b"model diagnostic\n"
    assert runner.calls == ["mount", "run", "delete", "umount"]
    assert list(config.work_dir.iterdir()) == []


@pytest.mark.skipif(not hasattr(os, "fork"), reason="POSIX launcher pipes")
@pytest.mark.parametrize("signaled,fail_cleanup", [(True, False), (True, True), (False, False)])
def test_descendant_held_pipes_do_not_block_failure_or_cleanup(
    tmp_path, monkeypatch, signaled, fail_cleanup,
):
    monkeypatch.setattr(rt, "STOP_GRACE_SECONDS", 0.1)
    marker = tmp_path / "descendant-finished"
    config = make_config(tmp_path)
    spec = make_spec(tmp_path)
    script = (
        "import os, pathlib, signal, sys, time\n"
        "pidfile = next(arg.split('=', 1)[1] for arg in sys.argv if arg.startswith('--pid-file='))\n"
        "pathlib.Path(pidfile).write_text(str(os.getpid()))\n"
        "print('retained launcher diagnostic', file=sys.stderr, flush=True)\n"
        "child = os.fork()\n"
        "if child == 0:\n"
        "    time.sleep(1.2)\n"
        "    pathlib.Path(%r).touch()\n"
        "    os._exit(0)\n" % str(marker)
    )
    if signaled:
        script += "signal.signal(signal.SIGINT, signal.SIG_DFL)\nos.kill(os.getpid(), signal.SIGINT)\n"
    launcher = RealLauncher(script, fail_cleanup=fail_cleanup)
    expected = (rt.SandboxCleanupError if fail_cleanup else
                rt.RuntimeHostError if signaled else rt.ArenaRuntimeError)
    started = time.monotonic()
    try:
        with pytest.raises(expected) as failure:
            rt.run_sandbox(config, spec, process_runner=launcher)
        assert time.monotonic() - started < 0.8
        if signaled and not fail_cleanup:
            assert failure.value.reason == "sandbox_launcher_signaled"
            assert failure.value.launch_exit_code == -signal.SIGINT
            assert "retained launcher diagnostic" in failure.value.launch_stderr
            assert failure.value.launch_stderr_truncated
        elif not signaled:
            assert str(failure.value) == "sandbox pipes did not close"
        assert launcher.calls == ["mount", "run", "delete", "umount"]
        assert not spec.output_dir.exists()
        assert list(config.work_dir.iterdir()) == []
    finally:
        # This test owns the short-lived descendant. Let it exit without any
        # process lookup or kill, even when the regression fails on old code.
        deadline = time.monotonic() + 3
        while not marker.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert marker.exists()


@pytest.mark.skipif(not hasattr(os, "fork"), reason="POSIX launcher pipes")
@pytest.mark.parametrize("fail_cleanup", [False, True])
def test_timed_out_checkpoint_survives_descendant_held_pipes(
    tmp_path, monkeypatch, fail_cleanup,
):
    monkeypatch.setattr(rt, "STOP_GRACE_SECONDS", 0.1)
    marker = tmp_path / "descendant-finished"
    config = make_config(tmp_path)
    spec = _checkpoint_spec(tmp_path, seconds=1)
    checkpoint = b'{"companies":[]}'
    launcher = RealLauncher(
        "import os, pathlib, sys, time\n"
        "pidfile = next(arg.split('=', 1)[1] for arg in sys.argv if arg.startswith('--pid-file='))\n"
        "pathlib.Path(pidfile).write_text(str(os.getpid()))\n"
        "pathlib.Path(%r).write_bytes(%r)\n"
        "print('retained stdout', flush=True)\n"
        "print('retained stderr', file=sys.stderr, flush=True)\n"
        "child = os.fork()\n"
        "if child == 0:\n"
        "    time.sleep(3)\n"
        "    pathlib.Path(%r).touch()\n"
        "    os._exit(0)\n"
        "time.sleep(30)\n" % (str(spec.output_path), checkpoint, str(marker)),
        fail_cleanup=fail_cleanup,
    )
    started = time.monotonic()
    try:
        if fail_cleanup:
            with pytest.raises(rt.SandboxCleanupError) as failure:
                rt.run_sandbox(config, spec, process_runner=launcher)
            result = failure.value.result
        else:
            result = rt.run_sandbox(config, spec, process_runner=launcher)
        assert time.monotonic() - started < 2
        assert result is not None and result.timed_out and result.exit_code is None
        assert result.output_bytes == checkpoint
        assert result.stdout == b"retained stdout\n"
        assert result.stderr == b"retained stderr\n"
        assert result.stdout_truncated and result.stderr_truncated
        assert launcher.calls == ["mount", "run", "kill", "delete", "umount"]
        assert not spec.output_dir.exists()
        assert list(config.work_dir.iterdir()) == []
    finally:
        deadline = time.monotonic() + 4
        while not marker.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert marker.exists()


def test_launch_stderr_is_bounded_and_drops_a_truncated_partial_line():
    error = host.RuntimeHostError(
        reason="sandbox_launch_failed", launch_stderr=b"safe line\nTOKEN=partial",
        launch_stderr_truncated=True,
    )
    assert error.launch_stderr == "safe line"
    assert error.launch_stderr_truncated
    error = host.RuntimeHostError(
        reason="sandbox_launch_failed", launch_stderr=b"x" * 10000,
    )
    assert len(error.launch_stderr) == host.MAX_LAUNCH_DIAGNOSTIC_CHARS
    assert error.launch_stderr_truncated


def test_launch_diagnostic_redacts_quoted_credentials_and_controls():
    error = host.RuntimeHostError(
        reason="sandbox_launch_failed",
        launch_stderr=b'password="two secret words"\n{"api_key":"private-value"}\nCookie: session=private-session; other=private-cookie\x1b[2J\nAuthorization=Bearer private-bearer-token',
    )
    rendered = host.runtime_host_private_diagnostic(error)
    assert "secret words" not in rendered
    assert "private-value" not in rendered
    assert "private-session" not in rendered
    assert "private-cookie" not in rendered
    assert "private-bearer-token" not in rendered
    assert "\x1b" not in rendered
