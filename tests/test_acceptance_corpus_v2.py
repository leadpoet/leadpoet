from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from gateway.tee.acceptance_corpus_v2 import (
    AcceptanceCorpusV2Error,
    REQUIRED_PROMOTION_BRANCHES,
    build_acceptance_corpus_from_index_v2,
    build_acceptance_corpus_v2,
    validate_acceptance_corpus_v2,
)
from leadpoet_canonical.attested_v2 import sha256_bytes


def _hash(label: str) -> str:
    return "sha256:" + hashlib.sha256(label.encode("utf-8")).hexdigest()


def _fixture(root: Path, kind: str, index: int, metadata=None):
    relative = Path(kind) / ("%04d.json" % index)
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    body = ('{"index":%d,"kind":"%s"}\n' % (index, kind)).encode("ascii")
    path.write_bytes(body)
    return {
        "kind": kind,
        "fixture_id": "%s:%04d" % (kind, index),
        "captured_at": "2026-06-%02dT00:00:00Z" % (1 + (index % 30)),
        "artifact_path": relative.as_posix(),
        "artifact_hash": sha256_bytes(body),
        "expected_output_hash": _hash("output:%s:%s" % (kind, index)),
        "receipt_root": _hash("receipt:%s:%s" % (kind, index)),
        "metadata": dict(metadata or {}),
    }


def _fixtures(root: Path, *, score_bundle_count: int = 100):
    values = [
        _fixture(root, "autoresearch_run", 0),
        _fixture(root, "provider_tape", 0),
        _fixture(root, "reward_allocation", 0),
    ]
    values.extend(
        _fixture(root, "score_bundle", index)
        for index in range(score_bundle_count)
    )
    values.extend(
        _fixture(
            root,
            "daily_benchmark",
            index,
            {"benchmark_date": "2026-06-%02d" % (index + 1)},
        )
        for index in range(14)
    )
    values.extend(
        _fixture(root, "promotion_branch", index, {"status": status})
        for index, status in enumerate(sorted(REQUIRED_PROMOTION_BRANCHES))
    )
    values.extend(
        _fixture(root, "weight_epoch", index, {"epoch_id": 23000 + index})
        for index in range(50)
    )
    return values


def _manifest(root: Path):
    key = Ed25519PrivateKey.generate()
    public_key = key.public_key().public_bytes_raw()
    manifest = build_acceptance_corpus_v2(
        fixtures=_fixtures(root),
        captured_from="2026-06-01T00:00:00Z",
        captured_through="2026-07-01T00:00:00Z",
        signing_pubkey_hex=public_key.hex(),
        sign_digest=key.sign,
    )
    return manifest, sha256_bytes(public_key)


def test_signed_acceptance_corpus_requires_complete_historical_coverage(tmp_path):
    manifest, signer_hash = _manifest(tmp_path)
    verified = validate_acceptance_corpus_v2(
        manifest,
        corpus_root=tmp_path,
        expected_signing_pubkey_hash=signer_hash,
    )
    assert verified == manifest
    assert verified["coverage"]["fixture_counts"]["score_bundle"] == 100
    assert len(verified["coverage"]["benchmark_dates"]) == 14
    assert len(verified["coverage"]["weight_epochs"]) == 50


def test_acceptance_corpus_rejects_changed_fixture_bytes(tmp_path):
    manifest, signer_hash = _manifest(tmp_path)
    (tmp_path / manifest["fixtures"][0]["artifact_path"]).write_text(
        "changed", encoding="utf-8"
    )
    with pytest.raises(AcceptanceCorpusV2Error, match="fixture hash differs"):
        validate_acceptance_corpus_v2(
            manifest,
            corpus_root=tmp_path,
            expected_signing_pubkey_hash=signer_hash,
        )


def test_acceptance_corpus_rejects_incomplete_score_history(tmp_path):
    key = Ed25519PrivateKey.generate()
    with pytest.raises(AcceptanceCorpusV2Error, match="score_bundle coverage"):
        build_acceptance_corpus_v2(
            fixtures=_fixtures(tmp_path, score_bundle_count=99),
            captured_from="2026-06-01T00:00:00Z",
            captured_through="2026-07-01T00:00:00Z",
            signing_pubkey_hex=key.public_key().public_bytes_raw().hex(),
            sign_digest=key.sign,
        )


def test_acceptance_corpus_rejects_unapproved_signer(tmp_path):
    manifest, _signer_hash = _manifest(tmp_path)
    with pytest.raises(AcceptanceCorpusV2Error, match="signer is not approved"):
        validate_acceptance_corpus_v2(
            manifest,
            corpus_root=tmp_path,
            expected_signing_pubkey_hash=_hash("another signer"),
        )


def test_acceptance_index_hashes_fixture_bytes_before_signing(tmp_path):
    fixtures = _fixtures(tmp_path)
    fixture_index = [
        {name: value for name, value in fixture.items() if name != "artifact_hash"}
        for fixture in fixtures
    ]
    key = Ed25519PrivateKey.generate()
    manifest = build_acceptance_corpus_from_index_v2(
        fixture_index=fixture_index,
        corpus_root=tmp_path,
        captured_from="2026-06-01T00:00:00Z",
        captured_through="2026-07-01T00:00:00Z",
        signing_key=key,
    )
    signer_hash = sha256_bytes(key.public_key().public_bytes_raw())

    assert validate_acceptance_corpus_v2(
        manifest,
        corpus_root=tmp_path,
        expected_signing_pubkey_hash=signer_hash,
    ) == manifest
    assert manifest["fixtures"][0]["artifact_hash"].startswith("sha256:")
