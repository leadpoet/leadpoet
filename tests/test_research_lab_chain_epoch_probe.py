from __future__ import annotations

import asyncio
import signal
import subprocess
import time

import pytest

from gateway.research_lab import chain


class _FakeProbeProcess:
    _next_pid = 40_000

    def __init__(
        self,
        *,
        stdout="",
        stderr="",
        returncode=0,
        block=False,
        resist_term=False,
        descendant_holds_pipe=False,
    ):
        type(self)._next_pid += 1
        self.pid = type(self)._next_pid
        self.returncode = None
        self._final_returncode = returncode
        self._stdout = stdout if isinstance(stdout, bytes) else stdout.encode("utf-8")
        self._stderr = stderr if isinstance(stderr, bytes) else stderr.encode("utf-8")
        self._block = block
        self._resist_term = resist_term
        self._descendant_holds_pipe = descendant_holds_pipe
        self._leader_release = asyncio.Event()
        self._pipe_release = asyncio.Event()
        self.group_alive = True
        self.waited = False

    async def communicate(self):
        if self._block:
            await self._pipe_release.wait()
        if self.returncode is None:
            self.returncode = self._final_returncode
            self._leader_release.set()
            self.group_alive = False
        return self._stdout, self._stderr

    async def wait(self):
        self.waited = True
        if self.returncode is None:
            await self._leader_release.wait()
        return self.returncode

    def signal(self, signum):
        if signum == 0:
            return
        if signum == signal.SIGTERM and self._resist_term:
            return
        if self.returncode is None:
            self.returncode = -signum
        self._leader_release.set()
        if signum == signal.SIGKILL or not self._descendant_holds_pipe:
            self.group_alive = False
            self._pipe_release.set()


@pytest.mark.asyncio
async def test_direct_epoch_probe_runs_in_killable_proxy_free_subprocess(monkeypatch):
    captured = {}
    process = _FakeProbeProcess(
        stdout=(
            "bittensor log\n"
            "LEADPOET_EPOCH_RESULT={\"epoch\":23978,\"block\":8632213,"
            "\"network\":\"finney\",\"official_subnet_epoch_id\":23913,"
            "\"epoch_ref\":\"sha256:fixture\"}\n"
        )
    )

    async def fake_create(*command, **kwargs):
        captured["command"] = list(command)
        captured.update(kwargs)
        return process

    monkeypatch.setenv("BITTENSOR_NETWORK", "finney")
    monkeypatch.setenv("HTTPS_PROXY", "http://secret-proxy.example")
    monkeypatch.setattr(chain.asyncio, "create_subprocess_exec", fake_create)

    assert await chain._fetch_current_chain_epoch_direct() == (
        23978,
        8632213,
        "finney",
    )
    assert captured["command"][:2] == [chain.sys.executable, "-c"]
    probe = captured["command"][2]
    assert "snapshot = read_subnet_epoch_snapshot(" in probe
    assert "subtensor," in probe
    assert "finalized=True" in probe
    assert "validate_cutover_anchor_from_archive(cutover)" in probe
    assert "_bind_probe_lifetime_to_parent()" in probe
    assert probe.index("_bind_probe_lifetime_to_parent()") < probe.index(
        "subtensor = bt.Subtensor"
    )
    assert "sys.stdout.flush()" in probe
    assert "os._exit(0)" in probe
    assert "subtensor.close()" not in probe
    assert "assert_legacy_epoch_namespace_open" not in probe
    assert "validate_subnet_epoch_cutover_anchor" not in probe
    assert "HTTPS_PROXY" not in captured["env"]
    assert captured["start_new_session"] is (chain.os.name == "posix")
    assert captured["stdout"] is subprocess.PIPE
    assert captured["stderr"] is subprocess.PIPE


def test_probe_lifetime_is_bound_to_unchanged_linux_parent(monkeypatch):
    calls = []

    class FakeLibc:
        def prctl(self, *args):
            calls.append(args)
            return 0

    parent_ids = iter((4321, 4321))
    monkeypatch.setattr(chain.sys, "platform", "linux")
    monkeypatch.setattr(chain.os, "getppid", lambda: next(parent_ids))
    monkeypatch.setattr("ctypes.CDLL", lambda *_args, **_kwargs: FakeLibc())

    chain._bind_probe_lifetime_to_parent()

    assert calls == [(1, signal.SIGKILL, 0, 0, 0)]


def test_probe_lifetime_exits_if_parent_is_already_gone(monkeypatch):
    class ProbeExited(RuntimeError):
        pass

    monkeypatch.setattr(chain.sys, "platform", "linux")
    monkeypatch.setattr(chain.os, "getppid", lambda: 1)
    monkeypatch.setattr(
        chain.os,
        "_exit",
        lambda code: (_ for _ in ()).throw(ProbeExited(code)),
    )

    with pytest.raises(ProbeExited, match="70"):
        chain._bind_probe_lifetime_to_parent()


@pytest.mark.asyncio
async def test_direct_epoch_probe_timeout_is_visible(monkeypatch):
    processes = []

    async def fake_create(*_command, **_kwargs):
        process = _FakeProbeProcess(block=True)
        processes.append(process)
        return process

    def killpg(pid, signum):
        process = next(process for process in processes if process.pid == pid)
        if signum == 0 and not process.group_alive:
            raise ProcessLookupError
        process.signal(signum)

    monkeypatch.setattr(chain.asyncio, "create_subprocess_exec", fake_create)
    monkeypatch.setattr(chain.os, "killpg", killpg)
    monkeypatch.setenv("RESEARCH_LAB_DIRECT_EPOCH_TIMEOUT_SECONDS", "2")
    monkeypatch.setenv("RESEARCH_LAB_DIRECT_EPOCH_ATTEMPTS", "1")

    started_at = time.monotonic()
    with pytest.raises(RuntimeError, match="exhausted exact-hash attempts") as exc:
        await chain._fetch_current_chain_epoch_direct()
    elapsed = time.monotonic() - started_at

    assert 0.9 <= elapsed < 2.0
    assert len(processes) == 1
    assert all(process.waited for process in processes)
    assert str(exc.value).count("timed out after 1.0s") == 1


@pytest.mark.asyncio
async def test_direct_epoch_probe_retries_transient_failure(monkeypatch):
    processes = [
        _FakeProbeProcess(returncode=1, stderr="temporary failure"),
        _FakeProbeProcess(
            stdout=(
                "LEADPOET_EPOCH_RESULT={\"epoch\":23978,\"block\":8632213,"
                "\"network\":\"finney\",\"official_subnet_epoch_id\":23913,"
                "\"epoch_ref\":\"sha256:fixture\"}\n"
            )
        ),
    ]
    calls = []

    async def fake_create(*command, **_kwargs):
        calls.append(command)
        return processes[len(calls) - 1]

    monkeypatch.setenv("BITTENSOR_NETWORK", "finney")
    monkeypatch.setattr(chain.asyncio, "create_subprocess_exec", fake_create)

    assert await chain._fetch_current_chain_epoch_direct() == (
        23978,
        8632213,
        "finney",
    )
    assert len(calls) == 2


@pytest.mark.asyncio
async def test_direct_epoch_probe_cancellation_reaps_without_retry(monkeypatch):
    started = asyncio.Event()
    signals = []
    processes = []

    class BlockingProcess(_FakeProbeProcess):
        async def communicate(self):
            started.set()
            return await super().communicate()

    async def fake_create(*_command, **_kwargs):
        process = BlockingProcess(block=True)
        processes.append(process)
        return process

    def killpg(pid, signum):
        process = next(process for process in processes if process.pid == pid)
        if signum == 0:
            if not process.group_alive:
                raise ProcessLookupError
            return
        signals.append((pid, signum))
        process.signal(signum)

    monkeypatch.setattr(chain.os, "name", "posix")
    monkeypatch.setattr(chain.os, "killpg", killpg)
    monkeypatch.setattr(chain.asyncio, "create_subprocess_exec", fake_create)

    task = asyncio.create_task(chain._fetch_current_chain_epoch_direct())
    await asyncio.wait_for(started.wait(), timeout=1.0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=1.0)
    await asyncio.sleep(0)

    assert len(processes) == 1
    assert processes[0].waited is True
    assert signals == [(processes[0].pid, signal.SIGTERM)]


@pytest.mark.asyncio
async def test_direct_epoch_probe_stop_escalates_and_drains_pipes(monkeypatch):
    process = _FakeProbeProcess(block=True, resist_term=True)
    signals = []

    def killpg(pid, signum):
        assert pid == process.pid
        if signum == 0:
            if not process.group_alive:
                raise ProcessLookupError
            return
        signals.append(signum)
        process.signal(signum)

    monkeypatch.setattr(chain.os, "name", "posix")
    monkeypatch.setattr(chain.os, "killpg", killpg)
    monkeypatch.setattr(chain, "_DIRECT_EPOCH_STOP_SECONDS", 0.01)
    communication_task = asyncio.create_task(process.communicate())

    await chain._stop_direct_epoch_probe(process, communication_task)

    assert signals == [signal.SIGTERM, signal.SIGKILL]
    assert process.waited is True
    assert communication_task.done() is True


@pytest.mark.asyncio
async def test_direct_epoch_probe_kills_descendant_after_leader_exits(monkeypatch):
    process = _FakeProbeProcess(block=True, descendant_holds_pipe=True)
    signals = []

    def killpg(pid, signum):
        assert pid == process.pid
        if signum == 0:
            if not process.group_alive:
                raise ProcessLookupError
            return
        signals.append(signum)
        process.signal(signum)

    monkeypatch.setattr(chain.os, "name", "posix")
    monkeypatch.setattr(chain.os, "killpg", killpg)
    monkeypatch.setattr(chain, "_DIRECT_EPOCH_STOP_SECONDS", 0.01)
    communication_task = asyncio.create_task(process.communicate())

    await chain._stop_direct_epoch_probe(process, communication_task)

    assert process.returncode == -signal.SIGTERM
    assert signals == [signal.SIGTERM, signal.SIGKILL]
    assert process.group_alive is False
    assert communication_task.done() is True


@pytest.mark.asyncio
async def test_direct_epoch_probe_cleanup_survives_repeated_cancellation(monkeypatch):
    started = asyncio.Event()
    term_seen = asyncio.Event()
    process = _FakeProbeProcess(block=True, resist_term=True)
    signals = []

    original_communicate = process.communicate

    async def communicate():
        started.set()
        return await original_communicate()

    process.communicate = communicate

    async def fake_create(*_command, **_kwargs):
        return process

    def killpg(pid, signum):
        assert pid == process.pid
        if signum == 0:
            if not process.group_alive:
                raise ProcessLookupError
            return
        signals.append(signum)
        process.signal(signum)
        if signum == signal.SIGTERM:
            term_seen.set()

    monkeypatch.setattr(chain.os, "name", "posix")
    monkeypatch.setattr(chain.os, "killpg", killpg)
    monkeypatch.setattr(chain, "_DIRECT_EPOCH_STOP_SECONDS", 0.02)
    monkeypatch.setattr(chain.asyncio, "create_subprocess_exec", fake_create)

    task = asyncio.create_task(chain._fetch_current_chain_epoch_direct())
    await asyncio.wait_for(started.wait(), timeout=1.0)
    task.cancel()
    await asyncio.wait_for(term_seen.wait(), timeout=1.0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=1.0)

    assert signals == [signal.SIGTERM, signal.SIGKILL]
    assert process.group_alive is False


@pytest.mark.asyncio
async def test_direct_epoch_probe_setup_interruption_reaps_owned_process(monkeypatch):
    process = _FakeProbeProcess(block=True)
    signals = []
    real_create_task = asyncio.create_task
    create_calls = 0

    async def fake_create(*_command, **_kwargs):
        return process

    def create_task(coroutine):
        nonlocal create_calls
        create_calls += 1
        if create_calls == 1:
            coroutine.close()
            raise KeyboardInterrupt
        return real_create_task(coroutine)

    def killpg(pid, signum):
        assert pid == process.pid
        if signum == 0:
            if not process.group_alive:
                raise ProcessLookupError
            return
        signals.append(signum)
        process.signal(signum)

    monkeypatch.setattr(chain.os, "name", "posix")
    monkeypatch.setattr(chain.os, "killpg", killpg)
    monkeypatch.setattr(chain.asyncio, "create_subprocess_exec", fake_create)
    monkeypatch.setattr(chain.asyncio, "create_task", create_task)

    with pytest.raises(KeyboardInterrupt):
        await chain._fetch_current_chain_epoch_direct()

    assert create_calls >= 3
    assert signals == [signal.SIGTERM]
    assert process.waited is True
    assert process.group_alive is False


@pytest.mark.parametrize(
    "stdout",
    (
        "",
        "LEADPOET_EPOCH_RESULT=not-json\n",
        "LEADPOET_EPOCH_RESULT={\"block\":0,\"network\":\"finney\"}\n",
        "LEADPOET_EPOCH_RESULT={\"block\":8632213,\"network\":\"test\"}\n",
        b"LEADPOET_EPOCH_RESULT=\xff\n",
    ),
)
@pytest.mark.asyncio
async def test_direct_epoch_probe_rejects_invalid_or_inconsistent_output(
    monkeypatch, stdout
):
    monkeypatch.setenv("BITTENSOR_NETWORK", "finney")

    async def fake_create(*_command, **_kwargs):
        return _FakeProbeProcess(stdout=stdout)

    monkeypatch.setattr(
        chain.asyncio,
        "create_subprocess_exec",
        fake_create,
    )

    with pytest.raises(RuntimeError, match="invalid output|inconsistent output"):
        await chain._fetch_current_chain_epoch_direct()
