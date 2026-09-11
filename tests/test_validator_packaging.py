"""Exercise real distribution artifacts without a checkout or third-party SDKs."""

from __future__ import annotations

from email.parser import Parser
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import zipfile

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
import pytest


ROOT = Path(__file__).resolve().parents[1]


def _run(command, *, cwd):
    result = subprocess.run(
        command, cwd=cwd, text=True, capture_output=True, timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


@pytest.fixture(scope="module")
def distributions(tmp_path_factory):
    """Build from source and again from an sdist, without dirtying the checkout."""
    root = tmp_path_factory.mktemp("validator-packaging")
    source = root / "source"
    source.mkdir()
    for name in ("setup.py", "requirements.txt", "README.md", "MANIFEST.in"):
        shutil.copy2(ROOT / name, source / name)
    for directory in ROOT.iterdir():
        if directory.is_dir() and (
            (directory / "__init__.py").is_file() or directory.name == "config"
        ):
            shutil.copytree(
                directory, source / directory.name,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
            )
    dist = root / "dist"
    _run(
        [sys.executable, "setup.py", "sdist", "--dist-dir", str(dist)],
        cwd=source,
    )
    sdist = next(dist.glob("*.tar.gz"))
    with tarfile.open(sdist) as archive:
        archive.extractall(root / "unpacked", filter="data")
    unpacked = next((root / "unpacked").iterdir())
    wheels = {}
    for name, checkout in (("source", source), ("sdist", unpacked)):
        wheel_dir = root / (name + "-wheel")
        _run(
            [sys.executable, "-m", "pip", "wheel", "--no-deps",
             "--no-build-isolation", "--no-index", "--disable-pip-version-check",
             "--wheel-dir", str(wheel_dir), str(checkout)],
            cwd=root,
        )
        wheels[name] = next(wheel_dir.glob("*.whl"))
    return wheels


@pytest.mark.parametrize("origin", ["source", "sdist"])
def test_wheel_dependencies_match_requirements(distributions, origin):
    with zipfile.ZipFile(distributions[origin]) as archive:
        metadata_path = next(n for n in archive.namelist() if n.endswith("/METADATA"))
        metadata = Parser().parsestr(archive.read(metadata_path).decode())
    declared = [
        Requirement(line.split("#", 1)[0].strip())
        for line in (ROOT / "requirements.txt").read_text().splitlines()
        if line.split("#", 1)[0].strip()
    ]
    packaged = [Requirement(value) for value in metadata.get_all("Requires-Dist")]
    assert {str(req) for req in packaged} == {str(req) for req in declared}
    names = [canonicalize_name(req.name) for req in declared]
    assert "bt" not in names
    assert "firecrawl-py" not in names
    assert names.count("firecrawl") == 1
    assert names.count("pyyaml") == 1
    assert metadata["Requires-Python"] == ">=3.11"


@pytest.mark.parametrize("origin", ["source", "sdist"])
def test_installed_wheel_imports_and_loads_signer_resources(distributions, origin, tmp_path):
    installed = tmp_path / "installed"
    _run(
        [sys.executable, "-m", "pip", "install", "--no-deps", "--no-index",
         "--disable-pip-version-check", "--target", str(installed),
         str(distributions[origin])],
        cwd=tmp_path,
    )
    # An unrelated top-level config directory must not override wheel data.
    (installed / "config").mkdir()
    (installed / "config/stateful-epoch-cutover-sn71.json").write_text("{}")
    # -I -S removes the checkout, PYTHONPATH, site-packages, and editable hooks.
    # These imports/profile loads must need neither Nitro nor third-party SDKs.
    probe = r"""
import json
from pathlib import Path
import sys
sys.path.insert(0, sys.argv[1])
import leadpoet_canonical.config
from neurons.validator import main
from lab_arena.local_weight_signer import load_public_chain_signing_profile
from Leadpoet.utils.subnet_epoch import (
    CUTOVER_JSON_ENV, CUTOVER_PATH_ENV, DEFAULT_SN71_CUTOVER_MANIFEST_PATH,
    ensure_cutover_manifest_configured, load_subnet_epoch_cutover,
)
assert callable(main)
for network in ('finney', 'test'):
    assert load_public_chain_signing_profile(network)['network'] == network
assert DEFAULT_SN71_CUTOVER_MANIFEST_PATH.is_relative_to(Path(sys.argv[1]))
environ = {}
ensure_cutover_manifest_configured(environ)
assert load_subnet_epoch_cutover(environ).netuid == 71
for override in ({CUTOVER_PATH_ENV: '/operator/mapping.json'},
                 {CUTOVER_JSON_ENV: '{"operator": "supplied"}'}):
    environ = dict(override)
    ensure_cutover_manifest_configured(environ)
    assert environ == override
print(DEFAULT_SN71_CUTOVER_MANIFEST_PATH.read_text())
"""
    output = _run(
        [sys.executable, "-I", "-S", "-c", probe, str(installed)], cwd=tmp_path,
    )
    assert json.loads(output) == json.loads(
        (ROOT / "config/stateful-epoch-cutover-sn71.json").read_text()
    )


def test_source_checkout_keeps_existing_cutover_default():
    from Leadpoet.utils.subnet_epoch import DEFAULT_SN71_CUTOVER_MANIFEST_PATH

    assert DEFAULT_SN71_CUTOVER_MANIFEST_PATH == ROOT / "config/stateful-epoch-cutover-sn71.json"
