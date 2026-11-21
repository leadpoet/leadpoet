"""The dependency interpreter and reusable cache match the trusted sandbox."""

import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from lab_arena import runner


def host_python(monkeypatch, version=(3, 12, 9), implementation="cpython"):
    monkeypatch.setattr(runner, "sys", SimpleNamespace(
        executable="/host/python", version_info=version,
        implementation=SimpleNamespace(name=implementation),
    ))


def requirements_and_target(tmp_path, text="jsonschema==4.25.1\n"):
    requirements = tmp_path / "requirements.txt"
    requirements.write_text(text)
    target = tmp_path / "deps"
    target.mkdir()
    return requirements, target


@pytest.mark.parametrize("host_version,expected", [
    ((3, 12, 9), "/usr/local/bin/python3.11"),
    ((3, 11, 13), "/host/python"),
])
def test_wheels_and_transitive_markers_use_sandbox_python(
    tmp_path, monkeypatch, host_version, expected,
):
    host_python(monkeypatch, host_version)
    requirements, target = requirements_and_target(tmp_path)
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        if "pip" in command:
            # Model the native wheel pip selects for its own interpreter.
            abi = "311" if command[0] == expected else "312"
            install_target = Path(command[command.index("--target") + 1])
            (install_target / f"rpds.cpython-{abi}-x86_64-linux-gnu.so").write_bytes(b"wheel")
        elif command[0] == "umount":
            for directory, _names, _files in os.walk(command[-1]):
                os.chmod(directory, 0o700)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(runner.subprocess, "run", run)
    runner.install_binary_requirements(requirements, target)
    pip, kwargs = next(item for item in calls if "pip" in item[0])
    assert pip[:4] == [expected, "-I", "-m", "pip"]
    assert "--python-version" not in pip  # Markers must also use Python 3.11.
    assert "--only-binary=:all:" in pip and "--no-compile" in pip
    assert (target / "rpds.cpython-311-x86_64-linux-gnu.so").is_file()
    assert "PYTHONPATH" not in kwargs["env"]
    if host_version[:2] != (3, 11):
        probe, options = calls[0]
        assert probe[:3] == [expected, "-I", "-c"]
        assert "sys.version_info[:2]" in probe[3]
        assert "sys.implementation.name" in probe[3]
        assert options["timeout"] <= 30
        assert options["stdout"] == options["stderr"] == subprocess.DEVNULL


@pytest.mark.parametrize("failure", ["missing", "wrong_version", "timeout"])
def test_unavailable_compatible_python_fails_before_mount_or_install(
    tmp_path, monkeypatch, failure,
):
    host_python(monkeypatch)
    requirements, target = requirements_and_target(tmp_path)
    commands = []

    def run(command, **kwargs):
        commands.append(command)
        assert command[0] in ("/usr/local/bin/python3.11", "/usr/bin/python3.11")
        if failure == "missing":
            raise FileNotFoundError("private path detail")
        if failure == "timeout":
            raise subprocess.TimeoutExpired(command, kwargs["timeout"])
        return SimpleNamespace(returncode=1)

    monkeypatch.setattr(runner.subprocess, "run", run)
    with pytest.raises(runner.DependencyInstallInfrastructureError) as raised:
        runner.install_binary_requirements(requirements, target)
    assert raised.value.failure_kind == "installer_error"
    assert "CPython 3.11" in raised.value.diagnostic_detail
    assert not any(target.iterdir()) and len(commands) == 2


def test_empty_requirements_need_no_installer(tmp_path, monkeypatch):
    host_python(monkeypatch)
    requirements, target = requirements_and_target(tmp_path, "# no dependencies\n\n")
    monkeypatch.setattr(runner.subprocess, "run", lambda *_a, **_k: pytest.fail("unexpected process"))
    runner.install_binary_requirements(requirements, target)
    assert not any(target.iterdir())


@pytest.mark.parametrize("implementation", ["cpython", "pypy"])
def test_verified_system_python_fallback(monkeypatch, implementation):
    host_python(monkeypatch, (3, 11, 13) if implementation == "pypy" else (3, 12, 9), implementation)
    commands = []

    def run(command, **_kwargs):
        commands.append(command)
        if command[0] == "/usr/local/bin/python3.11":
            raise FileNotFoundError
        assert command[0] == "/usr/bin/python3.11"
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(runner.subprocess, "run", run)
    assert runner._dependency_python() == "/usr/bin/python3.11"
    assert len(commands) == 2


def test_dependency_target_matches_canonical_image():
    dockerfile = Path(runner.__file__).parent / "judge" / "Dockerfile"
    version = ".".join(map(str, runner.DEPENDENCY_PYTHON_VERSION))
    assert f"ARG BASE_IMAGE=python:{version}-slim\n" in dockerfile.read_text()
    assert runner._DEPENDENCY_CACHE_TARGET == f"binary-wheels-v2:cpython-{version}\n".encode()

