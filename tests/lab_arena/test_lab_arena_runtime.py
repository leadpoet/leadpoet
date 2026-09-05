"""Arena sandbox runtime tests (labarena.md sections 9.2 and 18.4).

Real runsc execution needs a Linux x86_64 root host with an executable binary;
these tests drive ``run_sandbox`` with a fake process runner, clock, and
sleep so the orchestration (bundle, mount, deadline, kill, delete, cleanup,
bounded capture, output cap) is exercised anywhere. What only a Linux host
can prove is listed in the module-level report, not faked here.
"""

from __future__ import annotations

import io
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from lab_arena import contracts
from lab_arena import runtime as rt

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeClock:
    def __init__(self, start: float = 1000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


class FakeProcess:
    """Popen-shaped: exits at ``finish_at`` on the fake clock, or when killed."""

    def __init__(self, argv, *, clock, finish_at=None, returncode=0, stdout=b"", stderr=b""):
        self.args = list(argv)
        self._clock = clock
        self._finish_at = finish_at
        self._returncode = returncode
        self.stdout = io.BytesIO(stdout)
        self.stderr = io.BytesIO(stderr)
        self.returncode = None
        self.terminated = False
        self.killed = False
        self.pid = 4242

    def poll(self):
        if self.returncode is None:
            if self.killed or self.terminated:
                self.returncode = -9
            elif self._finish_at is not None and self._clock() >= self._finish_at:
                self.returncode = self._returncode
        return self.returncode

    def wait(self, timeout=None):
        if self.poll() is None:
            if timeout is not None and self._finish_at is not None and self._finish_at <= self._clock() + timeout:
                self._clock.now = self._finish_at
                return self.poll()
            raise subprocess.TimeoutExpired(self.args, timeout)
        return self.returncode

    def communicate(self, timeout=None):
        self.wait(timeout=timeout)
        return self.stdout.read(), self.stderr.read()

    def terminate(self):
        self.terminated = True

    def kill(self):
        self.killed = True


class FakeRunner:
    """Records every argv and builds processes per subcommand."""

    def __init__(self, clock, *, run_process=None, fail=()):
        self.clock = clock
        self.calls: list = []
        self.run_process = run_process
        self.fail = set(fail)

    def __call__(self, argv, **kwargs):
        argv = list(argv)
        self.calls.append((argv, kwargs))
        kind = self._kind(argv)
        if kind in self.fail:
            return FakeProcess(argv, clock=self.clock, finish_at=self.clock(), returncode=1)
        if kind == "run":
            assert kwargs["stdout"] is subprocess.PIPE and kwargs["stderr"] is subprocess.PIPE
            assert kwargs["stdin"] is subprocess.DEVNULL
            assert kwargs["start_new_session"] is True
            assert kwargs["env"] == {"PATH": "/usr/local/bin:/usr/bin:/bin"}
            if self.run_process is None:
                return FakeProcess(argv, clock=self.clock, finish_at=self.clock() + 1.0)
            return self.run_process(argv)
        return FakeProcess(argv, clock=self.clock, finish_at=self.clock())

    @staticmethod
    def _kind(argv):
        if argv[0] in ("mount", "umount"):
            return argv[0]
        for token in argv[1:]:
            if token in ("run", "kill", "delete"):
                return token
        return "other"

    def kinds(self):
        return [self._kind(argv) for argv, _ in self.calls]


def make_config(tmp_path: Path, **overrides) -> rt.RuntimeConfig:
    work = tmp_path / "work"
    work.mkdir(exist_ok=True)
    values = dict(runsc_path=Path("/usr/local/bin/runsc"), work_dir=work)
    values.update(overrides)
    return rt.RuntimeConfig(**values)


def make_spec(tmp_path: Path, **overrides) -> rt.SandboxSpec:
    input_dir = tmp_path / "input"
    input_dir.mkdir(exist_ok=True)
    (input_dir / rt.INPUT_FILE_NAME).write_text("{}", encoding="utf-8")
    socket_dir = tmp_path / "sock"
    socket_dir.mkdir(exist_ok=True)
    source_dir = tmp_path / "source"
    source_dir.mkdir(exist_ok=True)
    (source_dir / "harness.py").write_text("def run_icp(icp): return []\n")
    dependency_dir = tmp_path / "deps"
    dependency_dir.mkdir(exist_ok=True)
    entrypoint = tmp_path / "entrypoint.py"
    entrypoint.write_text("pass\n", encoding="utf-8")
    values = dict(
        sandbox_id="arena-icp-7",
        rootfs_path=tmp_path / "rootfs",
        input_dir=input_dir,
        output_dir=tmp_path / "output",
        socket_path=socket_dir / rt.SANDBOX_SOCKET_NAME,
        source_dir=source_dir,
        dependency_dir=dependency_dir,
        agent_entrypoint_path=entrypoint,
        entry_command=rt.AGENT_ENTRY_COMMAND,
        working_dir=rt.AGENT_WORKING_DIR,
        evaluation_date="2026-09-02",
        random_seed=12345,
    )
    values.update(overrides)
    return rt.SandboxSpec(**values)


# ---------------------------------------------------------------------------
# Runtime host contract
# ---------------------------------------------------------------------------


def test_runsc_path_must_be_a_regular_executable(tmp_path):
    binary = tmp_path / "runsc"
    binary.write_bytes(b"#!/bin/sh\n")
    binary.chmod(0o755)
    assert rt.require_runsc_executable(binary) == binary
    with pytest.raises(rt.RuntimeHostError):
        rt.require_runsc_executable(tmp_path / "missing")
    binary.chmod(0o644)
    with pytest.raises(rt.RuntimeHostError):
        rt.require_runsc_executable(binary)
    binary.chmod(0o755)
    link = tmp_path / "runsc-link"
    link.symlink_to(binary)
    assert rt.require_runsc_executable(link) == link


def test_runsc_runtime_cannot_be_constructed_on_an_unverified_host(tmp_path):
    config = make_config(tmp_path, runsc_path=tmp_path / "nope")
    with pytest.raises(rt.RuntimeHostError):
        rt.RunscRuntime(config)


def test_privileged_runtime_makes_input_and_socket_private_to_sandbox_uid(tmp_path):
    short_socket_dir = Path(tempfile.mkdtemp(prefix="la", dir="/tmp"))
    spec = make_spec(tmp_path, socket_path=short_socket_dir / rt.SANDBOX_SOCKET_NAME)
    worker = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    worker.bind(str(spec.socket_path))
    for path in (spec.input_dir, spec.input_dir / rt.INPUT_FILE_NAME, spec.socket_dir, spec.socket_path):
        os.chmod(path, 0o700)
    ownership = []
    try:
        rt.prepare_sandbox_access(
            spec,
            chown=lambda path, uid, gid: ownership.append((Path(path), uid, gid)),
        )
        assert {path for path, _uid, _gid in ownership} == {
            spec.input_dir,
            spec.input_dir / rt.INPUT_FILE_NAME,
            spec.socket_dir,
            spec.socket_path,
        }
        assert all((uid, gid) == (rt.SANDBOX_UID, rt.SANDBOX_GID) for _path, uid, gid in ownership)
        assert (spec.input_dir.stat().st_mode & 0o777) == 0o500
        assert ((spec.input_dir / rt.INPUT_FILE_NAME).stat().st_mode & 0o777) == 0o400
        assert (spec.socket_dir.stat().st_mode & 0o777) == 0o700
        assert (spec.socket_path.stat().st_mode & 0o777) == 0o600
    finally:
        worker.close()
        shutil.rmtree(short_socket_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# OCI spec, environment, commands
# ---------------------------------------------------------------------------


def test_oci_spec_invariants(tmp_path):
    spec = make_spec(tmp_path)
    document = rt.oci_spec(spec)
    json.dumps(document)
    assert document["root"] == {"path": str(spec.rootfs_path), "readonly": True}
    process = document["process"]
    assert process["user"] == {"uid": 65534, "gid": 65534}
    assert process["args"] == ["python3", "-I", "/agent/entrypoint.py"]
    assert process["cwd"] == "/agent/source"
    assert process["noNewPrivileges"] is True
    assert process["terminal"] is False
    assert all(process["capabilities"][name] == [] for name in ("bounding", "effective", "inheritable", "permitted", "ambient"))
    assert {"type": "RLIMIT_NOFILE", "hard": 1024, "soft": 1024} in process["rlimits"]
    assert {"type": "RLIMIT_NPROC", "hard": spec.pids_limit, "soft": spec.pids_limit} in process["rlimits"]
    env = dict(item.split("=", 1) for item in process["env"])
    assert env["TZ"] == "UTC" and env["LANG"] == "C.UTF-8" and env["LC_ALL"] == "C.UTF-8"
    assert env["PYTHONPATH"] == "/model"
    assert env["PYTHONHASHSEED"] == "12345" and env["LAB_ARENA_RANDOM_SEED"] == "12345"
    assert env["LAB_ARENA_EVALUATION_DATE"] == "2026-09-02"
    assert env["LAB_ARENA_WORKER_SOCKET"] == "/run/lab_arena/worker.sock"
    assert env["DEEPLINE_BASE_URL"] == "https://code.deepline.com" and "EXA_BASE_URL" not in env
    assert env["SCRAPINGDOG_BASE_URL"] == "https://api.scrapingdog.com"
    assert env["OPENROUTER_BASE_URL"] == "https://openrouter.ai/api/v1"
    assert env["LAB_ARENA_INPUT_PATH"] == "/input/icp.json" and env["LAB_ARENA_OUTPUT_PATH"] == "/output/companies.json"
    linux = document["linux"]
    assert {entry["type"] for entry in linux["namespaces"]} == {"pid", "ipc", "uts", "mount", "user"}
    assert linux["resources"] == {
        "memory": {"limit": spec.memory_limit_bytes},
        "cpu": {"quota": spec.cpu_quota, "period": spec.cpu_period},
        "pids": {"limit": spec.pids_limit},
    }
    assert linux["uidMappings"] == [{"containerID": 0, "hostID": 0, "size": 1}, {"containerID": 65534, "hostID": 65534, "size": 1}]
    assert linux["gidMappings"] == [{"containerID": 0, "hostID": 0, "size": 1}, {"containerID": 65534, "hostID": 65534, "size": 1}]
    seccomp = linux["seccomp"]
    assert seccomp["defaultAction"] == "SCMP_ACT_ALLOW"
    socket_rule = next(rule for rule in seccomp["syscalls"] if rule["names"] == ["socket"])
    assert socket_rule["action"] == "SCMP_ACT_ERRNO"
    assert socket_rule["args"] == [{"index": 0, "value": 1, "op": "SCMP_CMP_NE"}]
    deny_rule = next(rule for rule in seccomp["syscalls"] if "ptrace" in rule["names"])
    assert set(deny_rule["names"]) == {"mount", "pivot_root", "ptrace", "bpf", "keyctl", "perf_event_open"}
    assert "/proc/kcore" in linux["maskedPaths"] and "/proc/sys" in linux["readonlyPaths"]
    mounts = {mount["destination"]: mount for mount in document["mounts"]}
    assert set(mounts) == {
        "/proc", "/dev", "/tmp", "/input", "/output", "/run/lab_arena",
        "/agent/source", "/agent/deps", "/agent/entrypoint.py",
    }
    assert mounts["/input"]["source"] == str(spec.input_dir) and "ro" in mounts["/input"]["options"]
    assert mounts["/output"]["source"] == str(spec.output_dir) and "rw" in mounts["/output"]["options"]
    assert "noexec" in mounts["/output"]["options"]
    assert mounts["/run/lab_arena"]["source"] == str(spec.socket_dir)
    assert mounts["/agent/source"]["source"] == str(spec.source_dir)
    assert "ro" in mounts["/agent/source"]["options"]
    assert mounts["/agent/deps"]["source"] == str(spec.dependency_dir)
    assert "ro" in mounts["/agent/deps"]["options"]
    assert mounts["/agent/entrypoint.py"]["source"] == str(spec.agent_entrypoint_path)
    assert "ro" in mounts["/agent/entrypoint.py"]["options"]
    assert mounts["/tmp"]["type"] == "tmpfs" and "size=%d" % spec.tmp_tmpfs_bytes in mounts["/tmp"]["options"]
    assert document["hostname"] == "leadpoet-lab-arena"
    assert "network" not in json.dumps(linux["namespaces"])


def test_output_tmpfs_is_size_bound_and_owned_by_the_sandbox_user(tmp_path):
    spec = make_spec(tmp_path)
    command = rt.output_mount_command(spec)
    assert command[:4] == ["mount", "-t", "tmpfs", "-o"]
    assert "size=%d" % (64 * 1024 * 1024) in command[4]
    assert "uid=65534" in command[4] and "gid=65534" in command[4] and "noexec" in command[4]
    assert command[-1] == str(spec.output_dir)
    assert rt.output_unmount_command(spec) == ["umount", str(spec.output_dir)]


def test_command_construction(tmp_path):
    config = make_config(tmp_path)
    root = tmp_path / "root"
    bundle = tmp_path / "bundle"
    assert rt.runsc_run_command(config, root, bundle, "arena-1") == [
        "/usr/local/bin/runsc",
        "--root=%s" % root,
        "--rootless=false",
        "--network=none",
        "--host-uds=open",
        "--platform=systrap",
        "run",
        "--bundle=%s" % bundle,
        "arena-1",
    ]
    assert rt.runsc_kill_command(config, root, "arena-1") == ["/usr/local/bin/runsc", "--root=%s" % root, "kill", "arena-1", "KILL"]
    assert rt.runsc_delete_command(config, root, "arena-1") == ["/usr/local/bin/runsc", "--root=%s" % root, "delete", "--force", "arena-1"]
    with pytest.raises(rt.SandboxSpecError):
        make_config(tmp_path, platform="vmware")


@pytest.mark.parametrize(
    "overrides",
    [
        {"sandbox_id": "Bad Id"},
        {"sandbox_id": ""},
        {"entry_command": ()},
        {"entry_command": ("python3", "bad\nline")},
        {"entry_command": "python3 /model/main.py"},
        {"working_dir": "relative/dir"},
        {"evaluation_date": "2026-9-2"},
        {"random_seed": -1},
        {"random_seed": 2 ** 32},
        {"random_seed": True},
        {"uid": 0},
        {"wall_clock_seconds": 0},
        {"memory_limit_bytes": 0},
        {"socket_path": "/run/other.sock"},
        {"rootfs_path": "relative/rootfs"},
    ],
)
def test_sandbox_spec_is_validated(tmp_path, overrides):
    with pytest.raises(rt.SandboxSpecError):
        make_spec(tmp_path, **overrides)


def test_sandbox_spec_defaults_follow_the_public_constants(tmp_path):
    spec = make_spec(tmp_path)
    assert spec.wall_clock_seconds == contracts.ICP_WALL_CLOCK_SECONDS == 300
    assert spec.uid == 65534 and spec.gid == 65534
    assert spec.output_tmpfs_bytes == 64 * 1024 * 1024
    assert spec.argv == rt.AGENT_ENTRY_COMMAND == ("python3", "-I", "/agent/entrypoint.py")
    assert spec.cwd == rt.AGENT_WORKING_DIR == "/agent/source"


def test_host_process_and_environment_are_independent_of_image_metadata(tmp_path):
    spec = make_spec(tmp_path, extra_environment={"TRUSTED_SCORER": "1", "TZ": "America/New_York"})
    document = rt.oci_spec(spec)
    process = document["process"]
    assert process["args"] == ["python3", "-I", "/agent/entrypoint.py"]
    assert process["cwd"] == "/agent/source"
    environment = dict(item.split("=", 1) for item in process["env"])
    assert environment["PATH"] == rt.PROCESS_ENV["PATH"] and environment["TRUSTED_SCORER"] == "1"
    assert environment["TZ"] == "UTC" and environment["LAB_ARENA_OUTPUT_PATH"] == rt.SANDBOX_OUTPUT_PATH and environment["HOME"] == "/tmp"


def test_agent_isolated_python_skips_model_sitecustomize_and_loads_bound_mounts(tmp_path):
    model = tmp_path / "model"
    source = tmp_path / "source"
    dependencies = tmp_path / "deps"
    for path in (model, source, dependencies):
        path.mkdir()
    marker = tmp_path / "sitecustomize-imported"
    output = tmp_path / "output"
    (model / "sitecustomize.py").write_text(
        "from pathlib import Path\n"
        "Path(%r).write_text('imported', encoding='utf-8')\n" % str(marker),
        encoding="utf-8",
    )
    (dependencies / "bound_dependency.py").write_text(
        "VALUE = 'dependency-loaded'\n", encoding="utf-8"
    )
    (source / "bound_source.py").write_text(
        "from bound_dependency import VALUE\nRESULT = 'source-' + VALUE\n",
        encoding="utf-8",
    )
    entrypoint = tmp_path / "entrypoint.py"
    entrypoint.write_text(
        "import sys\n"
        "from pathlib import Path\n"
        "sys.path.insert(0, sys.argv[2])\n"
        "sys.path.insert(0, sys.argv[1])\n"
        "from bound_source import RESULT\n"
        "Path(sys.argv[3]).write_text(RESULT, encoding='utf-8')\n",
        encoding="utf-8",
    )
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(model)
    completed = subprocess.run(
        [
            sys.executable,
            *rt.AGENT_ENTRY_COMMAND[1:-1],
            str(entrypoint),
            str(source),
            str(dependencies),
            str(output),
        ],
        env=environment,
        capture_output=True,
        timeout=10,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr.decode("utf-8", "replace")
    assert output.read_text(encoding="utf-8") == "source-dependency-loaded"
    assert not marker.exists()


def test_scorer_python_keeps_trusted_model_startup(tmp_path):
    spec = make_spec(
        tmp_path,
        source_dir=None,
        dependency_dir=None,
        agent_entrypoint_path=None,
        entry_command=rt.SCORER_ENTRY_COMMAND,
        working_dir=rt.SCORER_WORKING_DIR,
    )
    document = rt.oci_spec(spec)
    process = document["process"]
    environment = dict(item.split("=", 1) for item in process["env"])
    assert process["args"] == ["python3", "/model/scorer_entrypoint.py"]
    assert environment["PYTHONPATH"] == "/model"


# ---------------------------------------------------------------------------
# run_sandbox orchestration
# ---------------------------------------------------------------------------


def test_normal_exit_collects_output_logs_and_cleans_everything(tmp_path):
    config = make_config(tmp_path)
    spec = make_spec(tmp_path)
    clock = FakeClock()

    def run_process(argv):
        # The "model" writes its output while running.
        spec.output_path.write_bytes(b'{"companies": []}')
        return FakeProcess(argv, clock=clock, finish_at=clock() + 2.5, returncode=0, stdout=b"hello\n", stderr=b"warn\n")

    runner = FakeRunner(clock, run_process=run_process)
    result = rt.run_sandbox(config, spec, process_runner=runner, clock=clock, sleep=clock.sleep, rusage=lambda: (3.0, 2048))
    assert result.exit_code == 0 and result.timed_out is False
    assert result.output_bytes == b'{"companies": []}' and result.output_error is None
    assert result.stdout == b"hello\n" and result.stderr == b"warn\n"
    assert result.stdout_truncated is False and result.stderr_truncated is False
    assert result.wall_seconds == pytest.approx(2.5, abs=0.2)
    assert result.max_rss_bytes == 2048
    assert result.output_path == str(spec.output_path)
    assert runner.kinds() == ["mount", "run", "delete", "umount"]
    run_argv = runner.calls[1][0]
    assert run_argv[0] == "/usr/local/bin/runsc" and "--network=none" in run_argv and run_argv[-1] == "arena-icp-7"
    bundle = Path(run_argv[-2].split("=", 1)[1])
    assert runner.calls[1][1]["cwd"] == str(bundle)
    # The bundle, its runsc root, and the output directory are gone.
    assert not bundle.exists() and not spec.output_dir.exists()
    assert list((tmp_path / "work").iterdir()) == []


def test_config_json_is_written_into_the_bundle_before_run(tmp_path):
    config = make_config(tmp_path)
    spec = make_spec(tmp_path)
    clock = FakeClock()
    seen = {}

    def run_process(argv):
        bundle = Path(argv[-2].split("=", 1)[1])
        seen["config"] = json.loads((bundle / "config.json").read_text(encoding="utf-8"))
        seen["mode"] = oct((bundle / "config.json").stat().st_mode & 0o777)
        seen["root"] = argv[1] == "--root=%s" % (bundle / "runsc")
        return FakeProcess(argv, clock=clock, finish_at=clock())

    rt.run_sandbox(config, spec, process_runner=FakeRunner(clock, run_process=run_process), clock=clock, sleep=clock.sleep, rusage=lambda: (0.0, 0))
    assert seen["config"] == rt.oci_spec(spec)
    assert seen["mode"] == "0o600" and seen["root"] is True


def test_timeout_kills_deletes_and_never_keeps_output(tmp_path):
    config = make_config(tmp_path)
    spec = make_spec(tmp_path, wall_clock_seconds=30)
    clock = FakeClock()

    def run_process(argv):
        spec.output_path.write_bytes(b"{}")
        return FakeProcess(argv, clock=clock, finish_at=None)  # never exits on its own

    runner = FakeRunner(clock, run_process=run_process)
    result = rt.run_sandbox(config, spec, process_runner=runner, clock=clock, sleep=clock.sleep, rusage=lambda: (0.0, 0))
    assert result.timed_out is True and result.exit_code is None
    assert result.wall_seconds >= 30 + rt.TIMEOUT_GRACE_SECONDS
    assert result.wall_seconds < 30 + rt.TIMEOUT_GRACE_SECONDS + 1
    assert runner.kinds() == ["mount", "run", "kill", "delete", "umount"]
    kill_argv = runner.calls[2][0]
    assert kill_argv[2:] == ["kill", "arena-icp-7", "KILL"]
    delete_argv = runner.calls[3][0]
    assert delete_argv[2:] == ["delete", "--force", "arena-icp-7"]
    launcher = runner.calls[1]
    assert not spec.output_dir.exists()
    # A timed-out model's partial output is still surfaced to the caller for
    # the invalid_output decision, but the directory itself never survives.
    assert result.output_bytes == b"{}"


def test_cleanup_runs_even_when_the_launcher_raises(tmp_path):
    config = make_config(tmp_path)
    spec = make_spec(tmp_path)
    clock = FakeClock()

    def run_process(argv):
        raise OSError("runsc exploded")

    runner = FakeRunner(clock, run_process=run_process)
    with pytest.raises(OSError):
        rt.run_sandbox(config, spec, process_runner=runner, clock=clock, sleep=clock.sleep, rusage=lambda: (0.0, 0))
    assert runner.kinds() == ["mount", "run", "delete", "umount"]
    assert not spec.output_dir.exists()
    assert list((tmp_path / "work").iterdir()) == []


def test_cleanup_failures_are_reported_after_every_step_runs(tmp_path):
    config = make_config(tmp_path)
    spec = make_spec(tmp_path)
    clock = FakeClock()
    runner = FakeRunner(clock, fail={"delete", "umount"})
    with pytest.raises(rt.SandboxCleanupError) as excinfo:
        rt.run_sandbox(config, spec, process_runner=runner, clock=clock, sleep=clock.sleep, rusage=lambda: (0.0, 0))
    assert "delete" in str(excinfo.value) and "umount" in str(excinfo.value)
    assert runner.kinds() == ["mount", "run", "delete", "umount"]
    assert not spec.output_dir.exists()
    assert list((tmp_path / "work").iterdir()) == []


def test_mount_failure_fails_closed_and_still_deletes(tmp_path):
    config = make_config(tmp_path)
    spec = make_spec(tmp_path)
    clock = FakeClock()
    runner = FakeRunner(clock, fail={"mount"})
    with pytest.raises(rt.ArenaRuntimeError):
        rt.run_sandbox(config, spec, process_runner=runner, clock=clock, sleep=clock.sleep, rusage=lambda: (0.0, 0))
    assert runner.kinds() == ["mount", "delete"]
    assert not spec.output_dir.exists()


def test_stdout_and_stderr_are_bounded_with_truncation_flags(tmp_path):
    config = make_config(tmp_path)
    spec = make_spec(tmp_path)
    clock = FakeClock()
    big = b"x" * (rt.MAX_LOG_BYTES * 3 + 17)

    def run_process(argv):
        return FakeProcess(argv, clock=clock, finish_at=clock(), stdout=big, stderr=b"e" * rt.MAX_LOG_BYTES)

    result = rt.run_sandbox(config, spec, process_runner=FakeRunner(clock, run_process=run_process), clock=clock, sleep=clock.sleep, rusage=lambda: (0.0, 0))
    assert len(result.stdout) == rt.MAX_LOG_BYTES and result.stdout_truncated is True
    assert len(result.stderr) == rt.MAX_LOG_BYTES and result.stderr_truncated is False
    assert result.output_bytes is None and result.output_error is None


def test_oversized_output_is_reported_not_returned(tmp_path):
    config = make_config(tmp_path)
    spec = make_spec(tmp_path)
    clock = FakeClock()

    def run_process(argv):
        spec.output_path.write_bytes(b"x" * (rt.MAX_OUTPUT_BYTES + 1))
        return FakeProcess(argv, clock=clock, finish_at=clock())

    result = rt.run_sandbox(config, spec, process_runner=FakeRunner(clock, run_process=run_process), clock=clock, sleep=clock.sleep, rusage=lambda: (0.0, 0))
    assert result.output_bytes is None
    assert result.output_error is not None and "bounded" in result.output_error


def test_read_output_enforces_the_cap_and_regular_files(tmp_path):
    spec = make_spec(tmp_path)
    assert rt.read_output(spec) is None
    spec.output_dir.mkdir()
    spec.output_path.write_bytes(b"x" * rt.MAX_OUTPUT_BYTES)
    assert len(rt.read_output(spec)) == rt.MAX_OUTPUT_BYTES
    spec.output_path.write_bytes(b"x" * (rt.MAX_OUTPUT_BYTES + 1))
    with pytest.raises(rt.SandboxOutputError):
        rt.read_output(spec)
    spec.output_path.unlink()
    target = tmp_path / "elsewhere.json"
    target.write_bytes(b"{}")
    spec.output_path.symlink_to(target)
    with pytest.raises(rt.SandboxOutputError):
        rt.read_output(spec)
    spec.output_path.unlink()
    spec.output_path.mkdir()
    with pytest.raises(rt.SandboxOutputError):
        rt.read_output(spec)


def test_run_sandbox_requires_input_file_and_empty_output_dir(tmp_path):
    config = make_config(tmp_path)
    spec = make_spec(tmp_path)
    clock = FakeClock()
    (spec.input_dir / rt.INPUT_FILE_NAME).unlink()
    with pytest.raises(rt.SandboxSpecError):
        rt.run_sandbox(config, spec, process_runner=FakeRunner(clock), clock=clock, sleep=clock.sleep)
    (spec.input_dir / rt.INPUT_FILE_NAME).write_text("{}", encoding="utf-8")
    spec.output_dir.mkdir()
    (spec.output_dir / "stale.json").write_text("{}", encoding="utf-8")
    with pytest.raises(rt.SandboxSpecError):
        rt.run_sandbox(config, spec, process_runner=FakeRunner(clock), clock=clock, sleep=clock.sleep)


def test_fake_runtime_records_specs_and_returns_preset_results(tmp_path):
    spec = make_spec(tmp_path)
    fake = rt.FakeRuntime([rt.fake_result(output_bytes=b"{}"), rt.fake_result(timed_out=True, exit_code=None)])
    first = fake.run_icp(spec)
    second = fake.run_icp(spec, process_runner=None, clock=None)
    assert first.output_bytes == b"{}" and second.timed_out is True
    assert fake.specs == [spec, spec]
    with pytest.raises(rt.ArenaRuntimeError):
        fake.run_icp(spec)
    assert fake.read_output(spec) is None
