from pathlib import Path

import pytest

from gateway.tee.source_bundle_v2 import (
    SourceBundleV2Error,
    build_source_bundle_v2,
    extract_source_bundle_v2,
)


def _source(root: Path) -> None:
    for path, content in {
        "gateway/scorer.py": "VALUE = 1\n",
        "qualification/scoring/core.py": "def score(): return 1\n",
        "sourcing_model/__init__.py": "\n",
        "validator_models/model.py": "MODEL = 'x'\n",
        "research_lab_adapter.py": "def adapter_metadata(): return {}\n",
        "requirements.txt": "httpx==0.27.0\n",
    }.items():
        target = root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")


def test_source_bundle_is_deterministic_and_reconstructs_exact_tree(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    _source(source)
    first = build_source_bundle_v2(source)
    second = build_source_bundle_v2(source)
    assert first == second

    result = extract_source_bundle_v2(
        first,
        destination=tmp_path / "extracted",
        expected_source_tree_hash=first["source_tree_hash"],
    )
    assert result["source_tree_hash"] == first["source_tree_hash"]
    assert (tmp_path / "extracted/gateway/scorer.py").read_text() == "VALUE = 1\n"


def test_source_bundle_rejects_tampering_and_wrong_artifact(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    _source(source)
    bundle = build_source_bundle_v2(source)
    tampered = dict(bundle)
    tampered["archive_b64"] = tampered["archive_b64"][:-4] + "AAAA"
    with pytest.raises(SourceBundleV2Error, match="commitment"):
        extract_source_bundle_v2(
            tampered,
            destination=tmp_path / "tampered",
            expected_source_tree_hash=bundle["source_tree_hash"],
        )
    with pytest.raises(SourceBundleV2Error, match="declared tree"):
        extract_source_bundle_v2(
            bundle,
            destination=tmp_path / "wrong",
            expected_source_tree_hash="sha256:" + "f" * 64,
        )


def test_source_bundle_excludes_private_key_and_env_files(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    _source(source)
    (source / ".env").write_text("SECRET=value\n")
    (source / "private.pem").write_text("secret")
    bundle = build_source_bundle_v2(source)
    extract_source_bundle_v2(
        bundle,
        destination=tmp_path / "extracted",
        expected_source_tree_hash=bundle["source_tree_hash"],
    )
    assert not (tmp_path / "extracted/.env").exists()
    assert not (tmp_path / "extracted/private.pem").exists()
