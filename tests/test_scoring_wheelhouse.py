from __future__ import annotations

import hashlib
from pathlib import Path
import zipfile

import pytest

from gateway.tee.scoring_wheelhouse import (
    ScoringWheelhouseError,
    _input_records,
    _lock_records,
    verify_wheelhouse,
)
from gateway.tee.scoring_executor import SCORING_RUNTIME_ENV_NAMES


ROOT = Path(__file__).resolve().parents[1]


def _write_fake_wheel(path: Path, *, name: str = "demo", version: str = "1.0") -> None:
    dist = name.replace("-", "_") + "-" + version + ".dist-info"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(dist + "/METADATA", "Name: %s\nVersion: %s\n" % (name, version))
        archive.writestr(dist + "/WHEEL", "Wheel-Version: 1.0\nTag: py3-none-any\n")
        archive.writestr(name.replace("-", "_") + "/__init__.py", "")


def test_production_scoring_lock_is_exact_complete_and_matches_input():
    input_path = ROOT / "gateway" / "tee" / "requirements-scoring-py39.in"
    lock_path = ROOT / "gateway" / "tee" / "requirements-scoring-py39.lock"
    inputs = _input_records(input_path)
    locked = _lock_records(lock_path)

    assert len(inputs) == 54
    assert inputs == {name: value[0] for name, value in locked.items()}
    assert all(len(digest) == 64 for _version, digest in locked.values())
    assert inputs["python-dateutil"] == "2.9.0"
    assert inputs["pydantic"] == "2.12.4"
    assert inputs["aiohttp"] == "3.13.2"
    assert inputs["click"] == "8.1.8"
    assert inputs["chardet"] == "4.0.0"
    assert inputs["pysocks"] == "1.7.1"
    assert inputs["numpy"] == "2.0.2"
    assert inputs["requests"] == "2.32.5"
    assert inputs["dnspython"] == "2.7.0"
    assert inputs["python-whois"] == "0.9.6"
    assert inputs["python-dotenv"] == "1.2.1"
    assert inputs["disposable-email-domains"] == "0.0.218"
    assert "rapidfuzz" not in inputs


def test_scoring_runtime_provisions_all_openrouter_fallback_keys():
    assert {
        "OPENROUTER_API_KEY",
        "QUALIFICATION_OPENROUTER_API_KEY",
        "FULFILLMENT_OPENROUTER_API_KEY",
        "OPENROUTER_KEY",
    } <= set(SCORING_RUNTIME_ENV_NAMES)


def test_wheelhouse_verifier_accepts_exact_wheel_and_rejects_tamper(tmp_path: Path):
    wheelhouse = tmp_path / "wheels"
    wheelhouse.mkdir()
    wheel = wheelhouse / "demo-1.0-py3-none-any.whl"
    _write_fake_wheel(wheel)
    digest = hashlib.sha256(wheel.read_bytes()).hexdigest()
    input_path = tmp_path / "requirements.in"
    lock_path = tmp_path / "requirements.lock"
    input_path.write_text("demo==1.0\n", encoding="utf-8")
    lock_path.write_text(
        "demo==1.0 \\\n    --hash=sha256:%s\n" % digest,
        encoding="utf-8",
    )

    assert verify_wheelhouse(
        input_path=input_path,
        lock_path=lock_path,
        wheelhouse=wheelhouse,
    ) == {"demo": "1.0"}

    wheel.write_bytes(wheel.read_bytes() + b"tampered")
    with pytest.raises(ScoringWheelhouseError, match="hash"):
        verify_wheelhouse(
            input_path=input_path,
            lock_path=lock_path,
            wheelhouse=wheelhouse,
        )


def test_gateway_eif_uses_pinned_python39_offline_lock_and_unchanged_outer_context():
    dockerfile = (ROOT / "gateway" / "tee" / "Dockerfile.enclave").read_text()
    stage = (ROOT / "gateway" / "tee" / "stage_attested_runtime.sh").read_text()
    prepare = (
        ROOT / "gateway" / "tee" / "prepare_offline_artifacts_v2.sh"
    ).read_text()

    assert (
        "FROM python:3.9.24-slim-bookworm@sha256:"
        "9b89d7e0d7e84de70c5e441198fdccb25b8d38310b0ccde36dbcd79a175fc1d9"
    ) in dockerfile
    assert "requirements-scoring-py39.lock" in dockerfile
    assert "--no-index" in dockerfile
    assert "--no-deps" in dockerfile
    assert "verify-installed" in dockerfile
    assert "yum install" not in dockerfile
    assert "pip download" not in stage
    assert "prepared offline scoring wheelhouse is unavailable" in stage
    assert "pip download" in prepare
    assert "--python-version 39" in prepare
    assert "--abi cp39" in prepare
    assert "--no-deps" in prepare
    assert "--require-hashes" in prepare
    assert "verify-wheelhouse" in stage
    assert prepare.count(
        'PYTHONPATH="$REPO_ROOT" python3 '
        '"$SCRIPT_DIR/sandbox_runtime_artifact.py" verify'
    ) == 3
