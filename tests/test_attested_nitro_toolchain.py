import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "attested_nitro_toolchain",
    ROOT / ".github/scripts/verify_attested_nitro_toolchain.py",
)
TOOLCHAIN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TOOLCHAIN)


@pytest.fixture
def toolchain(tmp_path):
    marker = tmp_path / "executed"
    binary = tmp_path / "nitro-cli"
    binary.write_text(
        f"#!/bin/sh\ntouch '{marker}'\nprintf 'Nitro CLI 1.4.2\\n'\n", encoding="utf-8"
    )
    binary.chmod(0o700)
    blobs = tmp_path / "blobs"
    blobs.mkdir()
    blob = blobs / "init"
    blob.write_bytes(b"approved init")
    lock = {
        "nitro_cli_version": "Nitro CLI 1.4.2",
        "nitro_cli_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
        "blobs": {"init": hashlib.sha256(blob.read_bytes()).hexdigest()},
    }
    return binary, blobs, lock, marker


def test_exact_approved_toolchain_runs_version_check(toolchain):
    binary, blobs, lock, marker = toolchain
    TOOLCHAIN.verify_toolchain(nitro_cli=binary, blobs=blobs, lock=lock)
    assert marker.exists()


@pytest.mark.parametrize("changed", ["binary", "blob", "symlink", "missing"])
def test_changed_toolchain_is_rejected_before_any_execution(toolchain, changed):
    binary, blobs, lock, marker = toolchain
    if changed == "binary":
        binary.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    elif changed == "blob":
        (blobs / "init").write_bytes(b"different init")
    elif changed == "symlink":
        (blobs / "init").unlink()
        (blobs / "init").symlink_to(binary)
    else:
        (blobs / "init").unlink()
    with pytest.raises(ValueError):
        TOOLCHAIN.verify_toolchain(nitro_cli=binary, blobs=blobs, lock=lock)
    assert not marker.exists()


def test_matching_hash_with_wrong_version_is_rejected(toolchain):
    binary, blobs, lock, marker = toolchain
    lock["nitro_cli_version"] = "Nitro CLI 1.4.1"
    with pytest.raises(ValueError, match="version differs"):
        TOOLCHAIN.verify_toolchain(nitro_cli=binary, blobs=blobs, lock=lock)
    assert marker.exists()


@pytest.fixture
def installer(monkeypatch):
    monkeypatch.setitem(sys.modules, "verify_attested_nitro_toolchain", TOOLCHAIN)
    spec = importlib.util.spec_from_file_location(
        "install_attested_toolchain",
        ROOT / ".github/scripts/install_attested_nitro_toolchain.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_package_hash_is_verified_before_extracting_any_file(installer, tmp_path, monkeypatch):
    lock = tmp_path / "lock.json"
    lock.write_text(json.dumps({"packages": [{
        "name": "aws-nitro-enclaves-cli", "url": "https://example.invalid/package.rpm",
        "sha256": hashlib.sha256(b"approved package").hexdigest(),
    }]}), encoding="utf-8")
    monkeypatch.setattr(installer, "LOCK", lock)
    monkeypatch.setattr(installer.urllib.request, "urlopen", lambda *a, **k: io.BytesIO(b"corrupt package"))
    extracted = []
    monkeypatch.setattr(installer, "extract_file", lambda *a, **k: extracted.append(a))
    with pytest.raises(ValueError, match="hash differs"):
        installer.install_toolchain(tmp_path / "toolchain")
    assert extracted == []


def test_package_download_is_bounded(installer, tmp_path, monkeypatch):
    monkeypatch.setattr(installer, "MAX_PACKAGE_BYTES", 1)
    monkeypatch.setattr(installer.urllib.request, "urlopen", lambda *a, **k: io.BytesIO(b"too large"))
    with pytest.raises(ValueError, match="size limit"):
        installer.download_package(
            {"url": "https://example.invalid/package.rpm", "sha256": "0" * 64},
            tmp_path / "package.rpm",
        )


def test_installer_cannot_overwrite_existing_toolchain(installer, tmp_path):
    marker = tmp_path / "keep"
    marker.write_bytes(b"keep")
    with pytest.raises(FileExistsError):
        installer.install_toolchain(tmp_path)
    assert marker.read_bytes() == b"keep"
