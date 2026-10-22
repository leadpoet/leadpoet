"""Exercise trust checks without downloading or changing an interpreter."""

import hashlib
import importlib.util
from pathlib import Path
import shlex
import ssl
import subprocess
import sys
from types import ModuleType

import certifi
import pytest


ROOT = Path(__file__).resolve().parents[1]
CHECKER = ROOT / "scripts/check_validator_python_security.py"
spec = importlib.util.spec_from_file_location("validator_python_security", CHECKER)
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)


@pytest.fixture
def packages(tmp_path, monkeypatch):
    pip_root = tmp_path / "pip"
    pip_root.mkdir()
    pip_file = pip_root / "__init__.py"
    pip_file.touch()
    bundle = pip_root / "_vendor/certifi/cacert.pem"
    bundle.parent.mkdir(parents=True)
    bundle.touch()
    modules = {}
    for name, version in (("certifi", "2026.10.19"), ("pip", "26.2.2"),
                          ("pip._vendor.certifi", "2026.10.19")):
        module = ModuleType(name)
        module.__version__ = version
        module.__file__ = str(pip_file)
        module.where = lambda: str(bundle)
        modules[name] = module
        monkeypatch.setitem(sys.modules, name, module)
    vendor = ModuleType("pip._vendor")
    vendor.certifi = modules["pip._vendor.certifi"]
    monkeypatch.setitem(sys.modules, "pip._vendor", vendor)
    return modules


def test_patched_and_newer_stable_packages_pass(packages):
    checker.check_security()
    packages["pip"].__version__ = "27.0"
    packages["certifi"].__version__ = "2027.1.1"
    packages["pip._vendor.certifi"].__version__ = "2027.1.1"
    checker.check_security()


@pytest.mark.parametrize("name,version", [
    ("certifi", "2026.07.22"), ("pip", "22.3.1"),
    ("pip._vendor.certifi", "2022.09.24"),
    ("certifi", "2026.10.19rc1"), ("pip", "26.2.2rc1"),
    ("pip._vendor.certifi", "2026.10.19.dev1"),
])
def test_unpatched_or_unproven_version_fails(packages, name, version):
    packages[name].__version__ = version
    with pytest.raises(ValueError):
        checker.check_security()


def test_distro_bundle_is_rejected_even_with_high_versions(packages, tmp_path):
    outside = tmp_path / "os-ca-bundle.pem"
    outside.touch()
    packages["pip._vendor.certifi"].where = lambda: str(outside)
    with pytest.raises(ValueError, match="private CA bundle"):
        checker.check_security()


def test_bundle_symlink_cannot_escape_pip(packages, tmp_path):
    bundle = Path(packages["pip._vendor.certifi"].where())
    bundle.unlink()
    outside = tmp_path / "os-ca-bundle.pem"
    outside.touch()
    bundle.symlink_to(outside)
    with pytest.raises(ValueError, match="private CA bundle"):
        checker.check_security()


def test_missing_bundle_and_import_fail_closed(packages, monkeypatch, capsys):
    Path(packages["pip._vendor.certifi"].where()).unlink()
    assert checker.main() == 1
    monkeypatch.setitem(sys.modules, "pip._vendor.certifi", None)
    monkeypatch.delattr(sys.modules["pip._vendor"], "certifi")
    assert checker.main() == 1
    assert "running service has not been stopped" in capsys.readouterr().err


def test_installed_runtime_bundle_excludes_exact_affected_roots():
    # An independent TLS parser verifies the security result of the test runtime,
    # rather than trusting a package label or matching source constants alone.
    roots = {hashlib.sha256(cert).hexdigest() for cert in
             ssl.create_default_context(cafile=certifi.where()).get_ca_certs(binary_form=True)}
    assert "86a1ecba089c4a8d3bbe2734c612ba341d813e043cf9e8a862cd5c57a36bbe6b" not in roots
    assert "d7a7a0fb5d7e2731d771e9484ebcdef71d5f0c3e0a2948782bc83ee0ea699ef4" not in roots


def test_canonical_gate_precedes_wallet_and_stop_and_propagates_failure(tmp_path):
    text = (ROOT / "validator_restart.sh").read_text()
    start = text.index('"$PYTHON" -I -B "$RELEASE/scripts/check_validator_python_security.py"')
    end = text.index("\n\n", start)
    assert start < text.index("write_environment_snapshot(Path")
    assert start < text.index('sudo env PYTHONDONTWRITEBYTECODE=1')
    assert start < text.index('timeout "$STOP_TIMEOUT" sudo systemctl stop "$SERVICE"')
    fake_python = tmp_path / "python"
    fake_python.write_text("#!/bin/sh\nexit 23\n")
    fake_python.chmod(0o755)
    result = subprocess.run(["bash", "-c", f"""set -euo pipefail
fail() {{ exit 1; }}
PYTHON={shlex.quote(str(fake_python))}
RELEASE={shlex.quote(str(ROOT))}
{text[start:end]}
echo STOP_REACHED
"""], text=True, capture_output=True)
    assert result.returncode == 1
    assert "STOP_REACHED" not in result.stdout
