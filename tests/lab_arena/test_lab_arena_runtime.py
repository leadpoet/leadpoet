"""Arena sandbox runtime tests (labarena.md sections 9.2 and 18.4).

Real runsc execution needs a Linux x86_64 root host with the locked binary;
these tests drive ``run_sandbox`` with a fake process runner, clock, and
sleep so the orchestration (bundle, mount, deadline, kill, delete, cleanup,
bounded capture, output cap) is exercised anywhere. What only a Linux host
can prove is listed in the module-level report, not faked here.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import platform
import subprocess
from pathlib import Path

import pytest

from lab_arena import contracts
from lab_arena import runtime as rt

REPO_ROOT = Path(__file__).resolve().parents[2]
GATEWAY_LOCK_PATH = REPO_ROOT / "gateway" / "tee" / "runsc-runtime.lock.json"


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


def make_lock():
    return rt.load_runtime_lock(rt.DEFAULT_RUNTIME_LOCK_PATH)


def make_config(tmp_path: Path, **overrides) -> rt.RuntimeConfig:
    work = tmp_path / "work"
    work.mkdir(exist_ok=True)
    values = dict(runsc_path=Path("/usr/local/bin/runsc"), lock=make_lock(), work_dir=work)
    values.update(overrides)
    return rt.RuntimeConfig(**values)


def make_spec(tmp_path: Path, **overrides) -> rt.SandboxSpec:
    input_dir = tmp_path / "input"
    input_dir.mkdir(exist_ok=True)
    (input_dir / rt.INPUT_FILE_NAME).write_text("{}", encoding="utf-8")
    socket_dir = tmp_path / "sock"
    socket_dir.mkdir(exist_ok=True)
    values = dict(
        sandbox_id="arena-icp-7",
        rootfs_path=tmp_path / "rootfs",
        input_dir=input_dir,
        output_dir=tmp_path / "output",
        socket_path=socket_dir / rt.SANDBOX_SOCKET_NAME,
        entry_command=("python3", "/model/main.py"),
        evaluation_date="2026-09-02",
        random_seed=12345,
    )
    values.update(overrides)
    return rt.SandboxSpec(**values)


# ---------------------------------------------------------------------------
# Lock parity and identity
# ---------------------------------------------------------------------------


def test_runtime_lock_is_a_verbatim_copy_of_the_gateway_lock_plus_schema_key():
    gateway = json.loads(GATEWAY_LOCK_PATH.read_text(encoding="utf-8"))
    arena = json.loads(rt.DEFAULT_RUNTIME_LOCK_PATH.read_text(encoding="utf-8"))
    assert set(gateway) == rt.GATEWAY_LOCK_FIELDS
    for key, value in gateway.items():
        assert arena[key] == value, key
    assert set(arena) - set(gateway) == {"lab_arena_lock_schema"}
    assert arena["lab_arena_lock_schema"] == rt.RUNTIME_LOCK_SCHEMA_VERSION
    lock = rt.load_runtime_lock()
    assert lock.runtime_lock_hash == contracts.document_hash(arena)
    assert lock.sha256 == gateway["sha256"] and lock.sha512 == gateway["sha512"]
    assert lock.size_bytes == gateway["size_bytes"]
    assert lock.install_path == Path("/usr/local/bin/runsc")
    assert lock.version.startswith("release-")


@pytest.mark.parametrize(
    "mutate",
    [
        lambda doc: doc.pop("sha512"),
        lambda doc: doc.update({"extra": 1}),
        lambda doc: doc.update({"lab_arena_lock_schema": "leadpoet.lab_arena.runtime_lock.v2"}),
        lambda doc: doc.update({"schema_version": "leadpoet.runsc_runtime_lock.v1"}),
        lambda doc: doc.update({"architecture": "aarch64"}),
        lambda doc: doc.update({"install_path": "/tmp/runsc"}),
        lambda doc: doc.update({"source_url": "https://example.com/runsc"}),
        lambda doc: doc.update({"sha256": "abc"}),
        lambda doc: doc.update({"size_bytes": 0}),
        lambda doc: doc.update({"size_bytes": True}),
    ],
)
def test_runtime_lock_shape_is_fail_closed(tmp_path, mutate):
    document = json.loads(rt.DEFAULT_RUNTIME_LOCK_PATH.read_text(encoding="utf-8"))
    mutate(document)
    path = tmp_path / "lock.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(rt.RuntimeLockError):
        rt.load_runtime_lock(path)
    with pytest.raises(rt.RuntimeLockError):
        rt.load_runtime_lock(tmp_path / "missing.json")
    (tmp_path / "bad.json").write_text("{", encoding="utf-8")
    with pytest.raises(rt.RuntimeLockError):
        rt.load_runtime_lock(tmp_path / "bad.json")


def test_verify_runsc_binary_checks_size_and_both_digests(tmp_path):
    data = os.urandom(4096)
    binary = tmp_path / "runsc"
    binary.write_bytes(data)
    binary.chmod(0o755)
    base = json.loads(rt.DEFAULT_RUNTIME_LOCK_PATH.read_text(encoding="utf-8"))

    def lock_for(**overrides):
        document = dict(base)
        document.update({"size_bytes": len(data), "sha256": contracts.hash_bytes(data), "sha512": hashlib.sha512(data).hexdigest()})
        document.update(overrides)
        return rt.RuntimeLock(document, contracts.document_hash(document))

    assert rt.verify_runsc_binary(binary, lock_for()) == contracts.hash_bytes(data)
    with pytest.raises(rt.RuntimeIdentityError):
        rt.verify_runsc_binary(binary, lock_for(size_bytes=len(data) + 1))
    with pytest.raises(rt.RuntimeIdentityError):
        rt.verify_runsc_binary(binary, lock_for(sha256=contracts.hash_bytes(b"other")))
    with pytest.raises(rt.RuntimeIdentityError):
        rt.verify_runsc_binary(binary, lock_for(sha512=hashlib.sha512(b"other").hexdigest()))
    with pytest.raises(rt.RuntimeIdentityError):
        rt.verify_runsc_binary(tmp_path / "missing", lock_for())
    binary.chmod(0o644)
    with pytest.raises(rt.RuntimeIdentityError):
        rt.verify_runsc_binary(binary, lock_for())
    binary.chmod(0o755)
    link = tmp_path / "runsc-link"
    link.symlink_to(binary)
    with pytest.raises(rt.RuntimeIdentityError):
        rt.verify_runsc_binary(link, lock_for())


def test_runsc_runtime_cannot_be_constructed_on_an_unverified_host(tmp_path):
    config = make_config(tmp_path, runsc_path=tmp_path / "nope")
    with pytest.raises(rt.ArenaRuntimeError) as excinfo:
        rt.RunscRuntime(config)
    if platform.system() != "Linux" or platform.machine().lower() not in ("x86_64", "amd64"):
        assert isinstance(excinfo.value, rt.RuntimeHostError)
    else:
        assert isinstance(excinfo.value, rt.RuntimeIdentityError)


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
    assert process["args"] == ["python3", "/model/main.py"]
    assert process["noNewPrivileges"] is True
    assert process["terminal"] is False
    assert all(process["capabilities"][name] == [] for name in ("bounding", "effective", "inheritable", "permitted", "ambient"))
    assert {"type": "RLIMIT_NOFILE", "hard": 1024, "soft": 1024} in process["rlimits"]
    assert {"type": "RLIMIT_NPROC", "hard": spec.pids_limit, "soft": spec.pids_limit} in process["rlimits"]
    env = dict(item.split("=", 1) for item in process["env"])
    assert env["TZ"] == "UTC" and env["LANG"] == "C.UTF-8" and env["LC_ALL"] == "C.UTF-8"
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
    assert set(mounts) == {"/proc", "/dev", "/tmp", "/input", "/output", "/run/lab_arena"}
    assert mounts["/input"]["source"] == str(spec.input_dir) and "ro" in mounts["/input"]["options"]
    assert mounts["/output"]["source"] == str(spec.output_dir) and "rw" in mounts["/output"]["options"]
    assert "noexec" in mounts["/output"]["options"]
    assert mounts["/run/lab_arena"]["source"] == str(spec.socket_dir)
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
        {"image_environment": {"LAB_ARENA_OUTPUT_PATH": "/etc/passwd"}},
        {"image_environment": {"bad name": "x"}},
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
    assert spec.argv == ("python3", "/model/main.py") and spec.cwd == "/tmp"


def test_image_process_env_and_workdir_apply_and_the_arena_names_win(tmp_path):
    """The image's ENTRYPOINT, ENV, and WORKDIR run as pinned; the Arena's own names cannot be overridden."""

    spec = make_spec(tmp_path, entry_command=("node", "agent.js", "--fast"), image_environment={"PATH": "/opt/app/bin:/usr/bin", "APP_MODE": "fast", "TZ": "America/New_York"}, working_dir="/opt/app")
    document = rt.oci_spec(spec)
    process = document["process"]
    assert process["args"] == ["node", "agent.js", "--fast"] and process["cwd"] == "/opt/app"
    environment = dict(item.split("=", 1) for item in process["env"])
    assert environment["PATH"] == "/opt/app/bin:/usr/bin" and environment["APP_MODE"] == "fast"
    assert environment["TZ"] == "UTC" and environment["LAB_ARENA_OUTPUT_PATH"] == rt.SANDBOX_OUTPUT_PATH and environment["HOME"] == "/tmp"
    plain = rt.oci_spec(make_spec(tmp_path))
    assert dict(item.split("=", 1) for item in plain["process"]["env"])["PATH"] == rt.PROCESS_ENV["PATH"]


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
