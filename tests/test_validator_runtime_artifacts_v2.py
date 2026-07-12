import hashlib
import io
import json
from pathlib import Path

import pytest

from validator_tee.scripts import stage_runtime_artifacts_v2 as artifacts


class _Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()


def _lock(tmp_path, content=b"wheel-bytes"):
    value = {
        "schema_version": artifacts.SCHEMA_VERSION,
        "artifacts": {
            "sr25519_cp37": {
                "filename": "sr25519.whl",
                "sha256": hashlib.sha256(content).hexdigest(),
                "url": "https://example.test/sr25519.whl",
            }
        },
    }
    path = tmp_path / "lock.json"
    path.write_text(json.dumps(value))
    return path, content


def test_artifact_stage_downloads_hash_checks_and_normalizes(tmp_path, monkeypatch):
    lock, content = _lock(tmp_path)
    requests = []

    def open_url(request, timeout):
        requests.append((request.full_url, timeout))
        return _Response(content)

    monkeypatch.setattr(artifacts, "urlopen", open_url)
    output = tmp_path / "out"
    manifest = artifacts.stage_artifacts(
        lock_path=lock,
        output_dir=output,
        allow_download=True,
    )
    staged = output / "sr25519.whl"
    assert staged.read_bytes() == content
    assert staged.stat().st_mode & 0o777 == 0o644
    assert int(staged.stat().st_mtime) == 0
    assert requests == [("https://example.test/sr25519.whl", 120)]
    assert manifest["artifacts"][0]["sha256"] == hashlib.sha256(content).hexdigest()
    assert json.loads((output / "manifest.json").read_text()) == manifest


def test_artifact_stage_reuses_only_exact_cached_bytes(tmp_path, monkeypatch):
    lock, content = _lock(tmp_path)
    output = tmp_path / "out"
    output.mkdir()
    (output / "sr25519.whl").write_bytes(content)

    monkeypatch.setattr(
        artifacts,
        "urlopen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("network used")),
    )
    artifacts.stage_artifacts(lock_path=lock, output_dir=output)


def test_artifact_stage_rejects_download_hash_mismatch(tmp_path, monkeypatch):
    lock, _content = _lock(tmp_path)
    monkeypatch.setattr(
        artifacts,
        "urlopen",
        lambda *_args, **_kwargs: _Response(b"attacker-artifact"),
    )
    with pytest.raises(artifacts.RuntimeArtifactV2Error, match="hash mismatch"):
        artifacts.stage_artifacts(
            lock_path=lock,
            output_dir=tmp_path / "out",
            allow_download=True,
        )


def test_artifact_stage_defaults_to_no_network(tmp_path, monkeypatch):
    lock, _content = _lock(tmp_path)
    monkeypatch.setattr(
        artifacts,
        "urlopen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("network used")),
    )

    with pytest.raises(
        artifacts.RuntimeArtifactV2Error,
        match="unavailable without network access",
    ):
        artifacts.stage_artifacts(lock_path=lock, output_dir=tmp_path / "out")


def test_artifact_stage_copies_exact_offline_bytes_without_network(tmp_path, monkeypatch):
    lock, content = _lock(tmp_path)
    offline = tmp_path / "offline"
    offline.mkdir()
    (offline / "sr25519.whl").write_bytes(content)
    monkeypatch.setattr(
        artifacts,
        "urlopen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("network used")),
    )

    output = tmp_path / "out"
    artifacts.stage_artifacts(
        lock_path=lock,
        output_dir=output,
        offline_artifact_root=offline,
    )

    assert (output / "sr25519.whl").read_bytes() == content


def test_artifact_stage_rejects_corrupt_offline_bytes_without_fallback(
    tmp_path,
    monkeypatch,
):
    lock, _content = _lock(tmp_path)
    offline = tmp_path / "offline"
    offline.mkdir()
    (offline / "sr25519.whl").write_bytes(b"tampered")
    monkeypatch.setattr(
        artifacts,
        "urlopen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("network used")),
    )

    with pytest.raises(artifacts.RuntimeArtifactV2Error, match="offline.*hash mismatch"):
        artifacts.stage_artifacts(
            lock_path=lock,
            output_dir=tmp_path / "out",
            offline_artifact_root=offline,
            allow_download=True,
        )


def test_checked_in_sr25519_pin_matches_verified_cp37_wheel():
    root = Path(__file__).resolve().parents[1]
    lock = artifacts.load_lock(root / "validator_tee/runtime-artifacts-v2.lock.json")
    pin = lock["artifacts"]["sr25519_cp37"]
    assert pin["sha256"] == "b74c31e2960c4af5b709b562aaf610989af532aee771fcdf175533de60441607"
    assert "cp37-cp37m-manylinux_2_17_x86_64" in pin["filename"]
