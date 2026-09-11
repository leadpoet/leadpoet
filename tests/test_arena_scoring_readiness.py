"""Scoring host checks must be useful without touching wallets or production."""

import builtins
import errno
import stat
import signal
import subprocess
import sys
from pathlib import Path

import pytest

from lab_arena import runtime_host as host, validator


@pytest.fixture
def scoring_host(tmp_path, monkeypatch):
    monkeypatch.setattr(host.platform, "system", lambda: "Linux")
    monkeypatch.setattr(host.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(host.os, "geteuid", lambda: 0)
    binary = tmp_path / "runsc"
    binary.write_text("#!/bin/sh\nexit 0\n")
    binary.chmod(0o755)
    return binary, tmp_path / "runner"


def test_scoring_check_needs_no_wallet_chain_gateway_or_signing_key(
    scoring_host, monkeypatch, capsys
):
    binary, work = scoring_host
    monkeypatch.setenv("LAB_ARENA_API_BASE_URL", "https://unreachable.invalid")
    monkeypatch.setenv("LAB_ARENA_SIGNING_KEY_HASH", "not-a-key")
    monkeypatch.setenv("LAB_ARENA_CHAIN_ENDPOINT", "not-a-node")
    monkeypatch.setenv("LAB_ARENA_NETUID", "not-an-integer")
    monkeypatch.setenv("LAB_ARENA_VALIDATOR_POLL_SECONDS", "not-an-integer")
    monkeypatch.setenv("LAB_ARENA_RUNSC_PATH", str(binary))
    monkeypatch.setenv("LAB_ARENA_RUNNER_WORK_DIR", str(work))
    original_import = builtins.__import__

    def local_only(name, globals=None, locals=None, fromlist=(), level=0):
        blocked = ("chain", "local_weight_signer", "wiring", "runtime")
        if name.startswith("bittensor") or name in tuple(
            "lab_arena." + item for item in blocked
        ):
            pytest.fail("scoring host check imported " + name)
        if name == "lab_arena" and any(item in blocked for item in fromlist):
            pytest.fail("scoring host check imported production wiring")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", local_only)
    monkeypatch.setattr(
        validator, "load_local_hotkey", lambda *_: pytest.fail("wallet accessed")
    )
    monkeypatch.setattr(
        validator,
        "run_validator_loops",
        lambda **_: pytest.fail("production loops started"),
    )

    assert validator.main(["--check-scoring-only"]) == 0
    output = capsys.readouterr()
    assert "host checks passed" in output.out
    assert "sandbox_execution=not_checked" in output.out
    assert str(binary) in output.out
    assert not output.err
    assert sorted(path.name for path in work.iterdir()) == [
        "images",
        "runs",
        "sandboxes",
        "sources",
    ]


def test_explicit_runsc_path_wins_over_environment_and_path_search(
    scoring_host, monkeypatch
):
    binary, work = scoring_host
    monkeypatch.setenv("LAB_ARENA_RUNSC_PATH", "/missing/runsc")
    monkeypatch.setenv("PATH", "/missing")
    assert (
        validator.main(
            [
                "--check-scoring-only",
                "--arena-runsc-path",
                str(binary),
                "--arena-work-dir",
                str(work),
            ]
        )
        == 0
    )


@pytest.mark.parametrize(
    "arguments",
    [
        ["--check-scoring-only", "--check-only"],
        ["--check-scoring-only", "--once"],
    ],
)
def test_readiness_modes_cannot_claim_or_broadcast(arguments):
    with pytest.raises(SystemExit) as failure:
        validator._parser().parse_args(arguments)
    assert failure.value.code == 2


@pytest.mark.parametrize(
    "kind,reason",
    [
        ("missing", "runsc_missing"),
        ("non_executable", "runsc_not_executable"),
        ("directory", "runsc_not_executable"),
        ("relative", "runsc_path_invalid"),
        ("bad_executable", "runsc_unusable"),
        ("nonzero", "runsc_unusable"),
        ("non_root", "root_required"),
        ("wrong_cpu", "unsupported_host"),
        ("wrong_os", "unsupported_host"),
    ],
)
def test_host_failures_have_actionable_reasons_without_mutating_work(
    scoring_host, monkeypatch, capsys, kind, reason
):
    binary, work = scoring_host
    if kind == "missing":
        binary.unlink()
    elif kind == "non_executable":
        binary.chmod(0o644)
    elif kind == "directory":
        binary.unlink()
        binary.mkdir()
    elif kind == "relative":
        binary = Path("runsc")
    elif kind == "bad_executable":
        binary.write_text("not an executable file\n")
    elif kind == "nonzero":
        binary.write_text("#!/bin/sh\necho secret-stderr >&2\nexit 1\n")
    elif kind == "non_root":
        monkeypatch.setattr(host.os, "geteuid", lambda: 1000)
    elif kind == "wrong_cpu":
        monkeypatch.setattr(host.platform, "machine", lambda: "aarch64")
    elif kind == "wrong_os":
        monkeypatch.setattr(host.platform, "system", lambda: "Darwin")
    assert (
        validator.main(
            [
                "--check-scoring-only",
                "--arena-runsc-path",
                str(binary),
                "--arena-work-dir",
                str(work),
            ]
        )
        == 1
    )
    output = capsys.readouterr()
    assert "reason=" + reason in output.err
    assert "Traceback" not in output.err and "secret-stderr" not in output.err
    assert "passed" not in output.out
    assert not work.exists()


def test_runtime_version_check_has_no_provider_environment(scoring_host, monkeypatch):
    binary, work = scoring_host
    binary.write_text('#!/bin/sh\ntest -z "${OPENROUTER_API_KEY+x}"\n')
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-secret-never-forward")
    host.prepare_scoring_host(binary, work)


def test_runtime_version_check_times_out(scoring_host, monkeypatch):
    binary, work = scoring_host
    binary.write_text("#!/bin/sh\nexec /bin/sleep 60\n")
    monkeypatch.setattr(host, "RUNSC_CHECK_TIMEOUT_SECONDS", 0.02)
    with pytest.raises(host.RuntimeHostError) as failure:
        host.prepare_scoring_host(binary, work)
    assert failure.value.reason == "runsc_probe_timeout"
    assert not work.exists()


def test_version_check_cleanup_timeout_has_safe_diagnostics(
    scoring_host, monkeypatch, capsys
):
    binary, work = scoring_host
    waits = []
    kills = []

    class HungProcess:
        pid = 123

        def wait(self, *, timeout):
            waits.append(timeout)
            raise subprocess.TimeoutExpired("secret-command", timeout)

    monkeypatch.setattr(host.subprocess, "Popen", lambda *_a, **_kw: HungProcess())
    monkeypatch.setattr(host.os, "killpg", lambda *args: kills.append(args))
    assert (
        validator.main(
            [
                "--check-scoring-only",
                "--arena-runsc-path",
                str(binary),
                "--arena-work-dir",
                str(work),
            ]
        )
        == 1
    )
    output = capsys.readouterr().err
    assert "reason=runsc_probe_cleanup_failed" in output
    assert "secret-command" not in output and "Traceback" not in output
    assert waits == [host.RUNSC_CHECK_TIMEOUT_SECONDS, host.RUNSC_CHECK_CLEANUP_SECONDS]
    assert kills == [(123, signal.SIGKILL)]
    assert not work.exists()


def test_hanging_version_check_terminates_helpers_and_reaps_parent(
    scoring_host, monkeypatch
):
    binary, work = scoring_host
    child_file = binary.parent / "helper.pid"
    binary.write_text(
        "#!%s\nimport subprocess, time\n" % sys.executable
        + "child = subprocess.Popen(['/bin/sleep', '60'])\n"
        + "with open(%r, 'w') as output: output.write(str(child.pid))\n"
        % str(child_file)
        + "time.sleep(60)\n"
    )
    real_popen = subprocess.Popen
    parents = []

    def record_process(*args, **kwargs):
        process = real_popen(*args, **kwargs)
        parents.append(process)
        return process

    monkeypatch.setattr(host.subprocess, "Popen", record_process)
    monkeypatch.setattr(host, "RUNSC_CHECK_TIMEOUT_SECONDS", 1)
    with pytest.raises(host.RuntimeHostError) as failure:
        host.prepare_scoring_host(binary, work)
    assert failure.value.reason == "runsc_probe_timeout"
    assert parents[0].returncode == -signal.SIGKILL
    child_pid = int(child_file.read_text())
    # An orphan may briefly remain as an init-owned zombie, but must not run.
    with real_popen(
        ["ps", "-o", "stat=", "-p", str(child_pid)], stdout=subprocess.PIPE
    ) as inspection:
        status = inspection.communicate(timeout=2)[0].decode().strip()
    assert not status or status.startswith("Z")
    assert not work.exists()


def test_runtime_constructor_rejects_non_root_even_with_a_valid_binary(
    scoring_host, monkeypatch
):
    from lab_arena import runtime

    binary, work = scoring_host
    monkeypatch.setattr(host.os, "geteuid", lambda: 1000)
    with pytest.raises(host.RuntimeHostError) as failure:
        runtime.RunscRuntime(runtime.RuntimeConfig(runsc_path=binary, work_dir=work))
    assert failure.value.reason == "root_required"


def test_service_environment_runs_scoring_check_without_touching_weight_journal(
    scoring_host, tmp_path, monkeypatch
):
    from scripts import run_arena_validator

    binary, work = scoring_host
    state = tmp_path / "state"
    state.mkdir()
    journal = state / "epoch-1-signed.json"
    journal.write_bytes(b"existing signed transaction bytes\n")
    settings = {
        "LAB_ARENA_RUNSC_PATH": str(binary),
        "LAB_ARENA_RUNNER_WORK_DIR": str(work),
        "LAB_ARENA_VALIDATOR_STATE_DIR": str(state),
    }
    for key in settings:
        monkeypatch.setenv(key, "/missing/ambient-setting")
    environment = tmp_path / "service.env"
    environment.write_text("".join('%s="%s"\n' % item for item in settings.items()))
    environment.chmod(0o600)
    # Exercise the real env-file ownership check under the test account;
    # privileged-host behavior is covered separately by the host tests.
    monkeypatch.setattr(host.os, "geteuid", lambda: environment.stat().st_uid)
    monkeypatch.setattr(host, "require_rootful_runtime", lambda: None)
    assert (
        run_arena_validator.main(
            ["--environment-file", str(environment), "--check-scoring-only"]
        )
        == 0
    )
    assert journal.read_bytes() == b"existing signed transaction bytes\n"
    assert list(state.iterdir()) == [journal]


def test_executable_on_noexec_mount_is_classified(scoring_host, monkeypatch):
    binary, work = scoring_host

    def denied(*_args, **_kwargs):
        raise PermissionError(errno.EACCES, "sensitive system error")

    monkeypatch.setattr(host.subprocess, "Popen", denied)
    with pytest.raises(host.RuntimeHostError) as failure:
        host.prepare_scoring_host(binary, work)
    assert failure.value.reason == "runsc_not_executable"
    assert "sensitive" not in host.runtime_host_diagnostic(failure.value)


@pytest.mark.parametrize("location", ["root", "sandboxes", "runs", "images", "sources"])
def test_directory_symlinks_fail_without_touching_target(tmp_path, location):
    target = tmp_path / "target"
    target.mkdir(mode=0o755)
    sentinel = target / "keep"
    sentinel.write_bytes(b"existing user data")
    before = target.stat().st_mode
    work = tmp_path / "runner"
    if location == "root":
        work.symlink_to(target, target_is_directory=True)
    else:
        work.mkdir()
        (work / location).symlink_to(target, target_is_directory=True)
    with pytest.raises(host.RuntimeHostError) as failure:
        host.prepare_runner_directories(work)
    assert failure.value.reason == "unsafe_work_directory"
    assert sentinel.read_bytes() == b"existing user data"
    assert target.stat().st_mode == before
    assert sorted(path.name for path in target.iterdir()) == ["keep"]


def test_directory_write_failure_preserves_existing_files(tmp_path, monkeypatch):
    work = tmp_path / "runner"
    work.mkdir()
    sentinel = work / "keep"
    sentinel.write_bytes(b"existing user data")

    def disk_full(*_args, **_kwargs):
        raise OSError(errno.ENOSPC, "secret system detail")

    monkeypatch.setattr(host.os, "write", disk_full)
    with pytest.raises(host.RuntimeHostError) as failure:
        host.prepare_runner_directories(work)
    assert failure.value.reason == "work_directory_unwritable"
    assert sentinel.read_bytes() == b"existing user data"
    assert sorted(path.name for path in work.iterdir()) == ["keep"]


def test_existing_runner_root_permissions_and_files_survive_checks(tmp_path):
    work = tmp_path / "runner"
    work.mkdir(mode=0o750)
    (work / "keep").write_text("existing cache")
    before = stat.S_IMODE(work.stat().st_mode)
    host.prepare_runner_directories(work)
    assert stat.S_IMODE(work.stat().st_mode) == before
    assert (work / "keep").read_text() == "existing cache"
    for name in ("sandboxes", "runs", "images", "sources"):
        assert stat.S_IMODE((work / name).stat().st_mode) == 0o700
        assert not list((work / name).iterdir())


@pytest.mark.parametrize("path", ["/", "/tmp", "/var/tmp", "/var/lib", "/etc"])
def test_broad_work_paths_are_rejected_before_mutation(path, monkeypatch):
    monkeypatch.setattr(
        host.os, "open", lambda *_args, **_kwargs: pytest.fail("broad path opened")
    )
    with pytest.raises(host.RuntimeHostError) as failure:
        host.prepare_runner_directories(Path(path))
    assert failure.value.reason == "unsafe_work_directory"


def test_symlinked_work_parent_is_rejected_without_creating_target(tmp_path):
    target = tmp_path / "target"
    target.mkdir(mode=0o755)
    link = tmp_path / "link"
    link.symlink_to(target, target_is_directory=True)
    before = target.stat().st_mode
    with pytest.raises(host.RuntimeHostError) as failure:
        host.prepare_runner_directories(link / "runner")
    assert failure.value.reason == "unsafe_work_directory"
    assert target.stat().st_mode == before
    assert not list(target.iterdir())


@pytest.mark.parametrize(
    "field,reason", [("runsc", "runsc_path_invalid"), ("work", "unsafe_work_directory")]
)
def test_malformed_paths_have_safe_diagnostics(scoring_host, capsys, field, reason):
    binary, work = scoring_host
    malformed = "/tmp/\x00secret-token"
    assert (
        validator.main(
            [
                "--check-scoring-only",
                "--arena-runsc-path",
                malformed if field == "runsc" else str(binary),
                "--arena-work-dir",
                malformed if field == "work" else str(work),
            ]
        )
        == 1
    )
    output = capsys.readouterr().err
    assert "reason=" + reason in output
    assert "Traceback" not in output and "secret-token" not in output


def test_socket_directory_check_preserves_shared_permissions_and_files(
    tmp_path, monkeypatch
):
    sockets = tmp_path / "sockets"
    sockets.mkdir(mode=0o1777)
    (sockets / "keep").write_text("existing")
    before = sockets.stat().st_mode
    monkeypatch.setattr(host, "DEFAULT_RUNNER_SOCKET_ROOT", sockets)
    host.check_runner_socket_directory()
    assert sockets.stat().st_mode == before
    assert [path.name for path in sockets.iterdir()] == ["keep"]


def test_readonly_socket_directory_has_actionable_failure(tmp_path, monkeypatch):
    sockets = tmp_path / "sockets"
    sockets.mkdir()
    monkeypatch.setattr(host, "DEFAULT_RUNNER_SOCKET_ROOT", sockets)

    def readonly(_descriptor):
        raise OSError(errno.EROFS, "sensitive mount detail")

    monkeypatch.setattr(host, "_probe_directory_write", readonly)
    with pytest.raises(host.RuntimeHostError) as failure:
        host.check_runner_socket_directory()
    assert failure.value.reason == "work_directory_unwritable"
    assert "sensitive" not in host.runtime_host_diagnostic(failure.value)


def test_diagnostic_never_logs_freeform_exception_text_or_unsafe_paths():
    error = host.RuntimeHostError(
        "https://provider.invalid?key=secret-token",
        reason="runsc_missing",
        runsc_path="/tmp/runsc\nsecret-token",
        work_dir="https://provider.invalid?key=secret-token",
    )
    rendered = host.runtime_host_diagnostic(error)
    assert "reason=runsc_missing" in rendered
    assert "secret-token" not in rendered
    assert "provider.invalid" not in rendered
    assert "\n" not in rendered
    unknown = host.RuntimeHostError("secret-token", reason="secret-token")
    assert "secret-token" not in host.runtime_host_diagnostic(unknown)


def test_normal_runner_uses_the_same_checks_before_wallet_access(monkeypatch):
    from types import SimpleNamespace
    from lab_arena import wiring

    args = SimpleNamespace(runsc_path="/missing/runsc", work_dir="/unused/runner")
    calls = []

    def fail_before_wallet(binary, work):
        calls.append((binary, work))
        raise host.RuntimeHostError(reason="runsc_missing", runsc_path=binary)

    monkeypatch.setattr(wiring, "prepare_scoring_host", fail_before_wallet)
    with pytest.raises(host.RuntimeHostError) as failure:
        wiring.build_runner_from_environment(args)
    assert failure.value.reason == "runsc_missing"
    assert calls == [(Path(args.runsc_path), Path(args.work_dir))]
