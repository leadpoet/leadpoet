from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
import shlex
import ssl
import subprocess
import sys
import types

import pytest


ROOT = Path(__file__).resolve().parents[1]
PIP_URL = "https://files.pythonhosted.org/packages/3f/06/3e4420302363fbb71be3c7f35ce042f54bfca330e3e66b3a2e3c4cca579f/pip-26.2.2-py3-none-any.whl"
PIP_SHA256 = "fd7f4dfa8190e014013b9f08f3e973f4e4c26ffd179a204895533a7254813a70"


@pytest.fixture(autouse=True)
def old_pip(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("PIP_CERT", "CURL_CA_BUNDLE", "REQUESTS_CA_BUNDLE"):
        monkeypatch.delenv(name, raising=False)
    pip = types.ModuleType("pip")
    pip.__version__ = "22.3.1"
    vendor = types.ModuleType("pip._vendor")
    vendor.certifi = types.SimpleNamespace(__version__="2022.12.7")
    monkeypatch.setitem(sys.modules, "pip", pip)
    monkeypatch.setitem(sys.modules, "pip._vendor", vendor)


def shell_function(name: str) -> str:
    script = (ROOT / "gw_restart.sh").read_text()
    return script.split(f"{name}() {{\n", 1)[1].split("\n}\n", 1)[0]


def bootstrap_source() -> str:
    return shell_function("bootstrap_gateway_pip").split("<<'PY'\n", 1)[1].rsplit("\nPY", 1)[0]


@pytest.mark.parametrize(
    "pip_version,certifi_version,system_bundle,skip",
    [
        ("26.2.2", "2026.10.19", False, True),
        ("27.0", "2027.1.1", False, True),
        ("26.2.2", "2026.10.19", True, False),
        ("26.2.2rc1", "2026.10.19", False, False),
        ("26.2.1", "2026.10.19", False, False),
        ("27.0", "2026.10.18", False, False),
        ("invalid", "2026.10.19", False, False),
        ("27.0", "2026.10.19rc1", False, False),
    ],
)
def test_reuse_requires_patched_stable_pip_and_bundled_certifi(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, pip_version: str,
    certifi_version: str, system_bundle: bool, skip: bool,
) -> None:
    import urllib.request

    pip = sys.modules["pip"]
    pip.__version__ = pip_version
    pip.__file__ = str(tmp_path / "pip" / "__init__.py")
    bundle = tmp_path / ("system-ca.pem" if system_bundle else "pip/_vendor/certifi/cacert.pem")
    vendor = sys.modules["pip._vendor"]
    vendor.certifi = types.SimpleNamespace(__version__=certifi_version, where=lambda: str(bundle))
    downloads = []

    def download(*args, **kwargs):
        downloads.append(True)
        return io.BytesIO(b"invalid fixture")

    monkeypatch.setattr(urllib.request, "urlopen", download)
    with pytest.raises(SystemExit) as error:
        exec(compile(bootstrap_source(), "gateway-pip-bootstrap", "exec"), {})
    assert error.value.code == (0 if skip else "ERROR: gateway pip bootstrap SHA256 mismatch")
    assert bool(downloads) is not skip


@pytest.mark.parametrize("scope", [[], ["--user"]])
@pytest.mark.parametrize("custom_ca", [None, "PIP_CERT", "CURL_CA_BUNDLE", "REQUESTS_CA_BUNDLE"])
def test_bootstrap_hashes_before_executing_only_verified_wheel(
    monkeypatch: pytest.MonkeyPatch, scope: list[str], custom_ca: str | None,
) -> None:
    import urllib.request

    payload = b"stand-in for the authenticated pip wheel"
    calls = []
    for name in ("PIP_CERT", "CURL_CA_BUNDLE", "REQUESTS_CA_BUNDLE"):
        monkeypatch.delenv(name, raising=False)
    if custom_ca:
        monkeypatch.setenv(custom_ca, "/explicit/custom-ca.pem")
    context = ssl.create_default_context()
    assert context.verify_mode == ssl.CERT_REQUIRED and context.check_hostname

    def create_context(*, cafile):
        assert cafile == ("/explicit/custom-ca.pem" if custom_ca else None)
        return context

    def download(url, *, context: ssl.SSLContext, timeout):
        assert url == PIP_URL and timeout == 60
        assert context.verify_mode == ssl.CERT_REQUIRED and context.check_hostname
        calls.append("download")
        return io.BytesIO(payload)

    def install(argv, *, check):
        assert calls == ["download"]
        assert check is True
        assert argv[0] == sys.executable
        assert argv[1] == "-c"
        assert 'sys.path.insert(0, sys.argv.pop(1))' in argv[2]
        wheel = Path(argv[3])
        assert wheel.read_bytes() == payload
        assert wheel.name == "pip-26.2.2-py3-none-any.whl"
        assert argv[4:] == [
            "install", "--no-index", "--no-deps", "--disable-pip-version-check",
            "--force-reinstall", *scope, str(wheel),
        ]
        calls.append(wheel)

    monkeypatch.setattr(ssl, "create_default_context", create_context)
    monkeypatch.setattr(urllib.request, "urlopen", download)
    monkeypatch.setattr(subprocess, "run", install)
    monkeypatch.setattr(sys, "argv", ["-", *scope])
    source = bootstrap_source()
    assert PIP_SHA256 in source
    # Use a known fixture hash to exercise successful validation without network.
    source = source.replace(PIP_SHA256, hashlib.sha256(payload).hexdigest())
    exec(compile(source, "gateway-pip-bootstrap", "exec"), {})
    assert len(calls) == 2
    assert not calls[1].exists()


@pytest.mark.parametrize("failure", ["hash", "tls", "install"])
def test_bootstrap_rejects_failure_without_executing_unverified_bytes(
    monkeypatch: pytest.MonkeyPatch, failure: str,
) -> None:
    import urllib.request

    payload = b"tampered wheel"
    executed = []
    monkeypatch.setattr(sys, "argv", ["-"])
    for name in ("PIP_CERT", "CURL_CA_BUNDLE", "REQUESTS_CA_BUNDLE"):
        monkeypatch.delenv(name, raising=False)

    def download(*args, **kwargs):
        if failure == "tls":
            raise ssl.SSLCertVerificationError("fixture rejected certificate")
        return io.BytesIO(payload)

    def install(argv, **kwargs):
        executed.append(argv)
        raise subprocess.CalledProcessError(1, "verified offline pip")

    monkeypatch.setattr(urllib.request, "urlopen", download)
    monkeypatch.setattr(subprocess, "run", install)
    source = bootstrap_source()
    if failure == "install":
        source = source.replace(PIP_SHA256, hashlib.sha256(payload).hexdigest())
    with pytest.raises((SystemExit, ssl.SSLCertVerificationError, subprocess.CalledProcessError)):
        exec(compile(source, "gateway-pip-bootstrap", "exec"), {})
    assert len(executed) == (1 if failure == "install" else 0)
    if executed:
        assert not Path(executed[0][3]).exists()


@pytest.mark.parametrize("venv", [False, True])
@pytest.mark.parametrize("reuse", [False, True])
@pytest.mark.parametrize("bootstrap_fails", [False, True])
def test_dependency_install_bootstraps_before_download_and_fingerprint_reuse(
    tmp_path: Path, venv: bool, reuse: bool, bootstrap_fails: bool,
) -> None:
    (tmp_path / "requirements.txt").write_text("certifi>=2026.10.19\n")
    capture = tmp_path / "calls.jsonl"
    python = tmp_path / "gateway-python"
    python.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        f"with open({str(capture)!r}, 'a') as f: f.write(json.dumps(sys.argv[1:])+'\\n')\n"
        "if sys.argv[1] == '-c': sys.exit(0 if os.environ['TEST_VENV']=='1' else 1)\n"
        "if sys.argv[1] == '-':\n"
        "    sys.stdin.read()\n"
        "    if len(sys.argv)==2 or sys.argv[2]=='--user':\n"
        "        sys.exit(int(os.environ['TEST_BOOTSTRAP_FAILS']))\n"
        "    print('fixture-fingerprint')\n"
    )
    python.chmod(0o755)
    source = "\n".join(
        f"{name}() {{\n{shell_function(name)}\n}}"
        for name in ("bootstrap_gateway_pip", "install_gateway_python_dependencies")
    )
    harness = (
        "set -euo pipefail\n"
        f"GATEWAY_PYTHON_BIN={shlex.quote(str(python))}\n"
        f"LEADPOET_REPO_ROOT={shlex.quote(str(tmp_path))}\n"
        "GATEWAY_PREFLIGHT_TREE=\nGATEWAY_HOST_EXTRA_PYTHON_PACKAGES=(sentry-sdk)\n"
        f"GATEWAY_DEPENDENCY_INSTALL_FINGERPRINT={'fixture-fingerprint' if reuse else ''}\n"
        + source + "\n"
        # Match the real pre-shutdown conditional: errexit is disabled here.
        "if ! install_gateway_python_dependencies; then exit 42; fi\n"
    )
    env = dict(os.environ, TEST_VENV=str(int(venv)), TEST_BOOTSTRAP_FAILS=str(int(bootstrap_fails)))
    result = subprocess.run(["bash", "-c", harness], env=env, capture_output=True, text=True)
    assert result.returncode == (42 if bootstrap_fails else 0), result.stderr
    calls = [json.loads(line) for line in capture.read_text().splitlines()]
    assert calls[0][0] == "-c"
    assert calls[1] == ["-", *([] if venv else ["--user"])]
    if bootstrap_fails:
        assert len(calls) == 2
        return
    assert calls[2][0] == "-" and calls[2][1] == str(tmp_path / "requirements.txt")
    pip_calls = [call for call in calls if call[:2] == ["-m", "pip"]]
    assert pip_calls[-1] == ["-m", "pip", "check"]
    installs = [call for call in pip_calls if call[2] == "install"]
    assert len(installs) == (0 if reuse else 1)
    if installs:
        assert installs[0][3:] == [
            "--use-deprecated=legacy-certs", *([] if venv else ["--user"]),
            "--requirement", str(tmp_path / "requirements.txt"), "sentry-sdk",
        ]


def test_no_unhashed_get_pip_fallback_or_tls_bypass() -> None:
    source = shell_function("bootstrap_gateway_pip") + shell_function("install_gateway_python_dependencies")
    for forbidden in ("get-pip.py", "--trusted-host", "CERT_NONE", "check_hostname = False"):
        # Comment names the removed get-pip path, but it must never be invoked.
        assert forbidden not in "\n".join(line for line in source.splitlines() if not line.lstrip().startswith("#"))
