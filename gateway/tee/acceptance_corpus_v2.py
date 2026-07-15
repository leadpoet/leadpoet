"""Validate the immutable V2 historical acceptance corpus.

The corpus remains outside Git because it contains sanitized production
fixtures.  Its signed manifest is content-addressed and bound into the approved
gateway release.  Full-topology deployment is refused unless every referenced
fixture is present and the required historical coverage is complete.
"""

from __future__ import annotations

import argparse
import base64
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Dict, Mapping, Optional, Sequence

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)


SCHEMA_VERSION = "leadpoet.v2_acceptance_corpus.v2"
BASELINE_COMMIT = "7c9766b71d4c08b0059f6e3230dbe742b1d58e79"
MIN_CAPTURE_DAYS = 30
MIN_COUNTS = {
    "autoresearch_run": 1,
    "provider_tape": 1,
    "score_bundle": 100,
    "daily_benchmark": 14,
    "promotion_branch": 6,
    "reward_allocation": 1,
    "weight_epoch": 50,
}
REQUIRED_PROMOTION_BRANCHES = frozenset(
    {
        "disabled",
        "rejected_legacy_patch_candidate",
        "rejected_basis_unavailable",
        "rejected_below_threshold",
        "stale_parent_needs_rescore",
        "promotion_passed",
    }
)

_HASH_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}$")
_FORBIDDEN_MANIFEST_MARKERS = (
    "sk-or-",
    "service_role",
    "raw_secret",
    "openrouter_api_key",
    "scrapingdog_api_key",
    "exa_api_key",
    "proxy_url",
    "hidden_prompt",
    "provider_output",
)


class AcceptanceCorpusV2Error(ValueError):
    """The historical corpus is incomplete, mutable, or unauthenticated."""


def _canonical(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise AcceptanceCorpusV2Error(
            "acceptance corpus is not canonical JSON"
        ) from exc


def _sha256_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _timestamp(value: Any, field: str) -> datetime:
    text = str(value or "")
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise AcceptanceCorpusV2Error("%s is invalid" % field) from exc
    if parsed.tzinfo is None:
        raise AcceptanceCorpusV2Error("%s must include a timezone" % field)
    return parsed.astimezone(timezone.utc)


def _normalized_fixture(value: Mapping[str, Any]) -> Dict[str, Any]:
    fields = {
        "kind",
        "fixture_id",
        "captured_at",
        "artifact_path",
        "artifact_hash",
        "expected_output_hash",
        "receipt_root",
        "metadata",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise AcceptanceCorpusV2Error("acceptance fixture fields are invalid")
    kind = str(value.get("kind") or "")
    if kind not in MIN_COUNTS:
        raise AcceptanceCorpusV2Error("acceptance fixture kind is invalid")
    fixture_id = str(value.get("fixture_id") or "")
    if not _ID_RE.fullmatch(fixture_id):
        raise AcceptanceCorpusV2Error("acceptance fixture ID is invalid")
    captured_at = str(value.get("captured_at") or "")
    _timestamp(captured_at, "captured_at")
    relative = Path(str(value.get("artifact_path") or ""))
    if relative.is_absolute() or not relative.parts or any(
        part in {"", ".", ".."} for part in relative.parts
    ):
        raise AcceptanceCorpusV2Error("acceptance fixture path is invalid")
    metadata = value.get("metadata")
    if not isinstance(metadata, Mapping):
        raise AcceptanceCorpusV2Error("acceptance fixture metadata is invalid")
    for field in ("artifact_hash", "expected_output_hash", "receipt_root"):
        if not _HASH_RE.fullmatch(str(value.get(field) or "")):
            raise AcceptanceCorpusV2Error("acceptance fixture %s is invalid" % field)
    return {
        "kind": kind,
        "fixture_id": fixture_id,
        "captured_at": captured_at,
        "artifact_path": relative.as_posix(),
        "artifact_hash": str(value["artifact_hash"]),
        "expected_output_hash": str(value["expected_output_hash"]),
        "receipt_root": str(value["receipt_root"]),
        "metadata": dict(metadata),
    }


def _coverage_summary(fixtures: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    counts = Counter(str(item["kind"]) for item in fixtures)
    for kind, minimum in MIN_COUNTS.items():
        if counts[kind] < minimum:
            raise AcceptanceCorpusV2Error(
                "acceptance corpus has insufficient %s coverage" % kind
            )
    benchmark_dates = {
        str(item["metadata"].get("benchmark_date") or "")
        for item in fixtures
        if item["kind"] == "daily_benchmark"
    }
    if len(benchmark_dates - {""}) < 14:
        raise AcceptanceCorpusV2Error(
            "acceptance corpus has fewer than 14 benchmark dates"
        )
    weight_epochs = {
        item["metadata"].get("epoch_id")
        for item in fixtures
        if item["kind"] == "weight_epoch"
        and isinstance(item["metadata"].get("epoch_id"), int)
        and not isinstance(item["metadata"].get("epoch_id"), bool)
    }
    if len(weight_epochs) < 50:
        raise AcceptanceCorpusV2Error(
            "acceptance corpus has fewer than 50 unique weight epochs"
        )
    promotion_branches = {
        str(item["metadata"].get("status") or "")
        for item in fixtures
        if item["kind"] == "promotion_branch"
    }
    missing_branches = sorted(REQUIRED_PROMOTION_BRANCHES - promotion_branches)
    if missing_branches:
        raise AcceptanceCorpusV2Error(
            "acceptance corpus is missing promotion branches: %s"
            % ", ".join(missing_branches)
        )
    return {
        "fixture_counts": dict(sorted(counts.items())),
        "benchmark_dates": sorted(benchmark_dates - {""}),
        "promotion_branches": sorted(promotion_branches),
        "weight_epochs": sorted(weight_epochs),
    }


def build_acceptance_corpus_v2(
    *,
    fixtures: Sequence[Mapping[str, Any]],
    captured_from: str,
    captured_through: str,
    signing_pubkey_hex: str,
    sign_digest: Any,
) -> Dict[str, Any]:
    start = _timestamp(captured_from, "captured_from")
    end = _timestamp(captured_through, "captured_through")
    if (end - start).total_seconds() < MIN_CAPTURE_DAYS * 86400:
        raise AcceptanceCorpusV2Error("acceptance corpus covers fewer than 30 days")
    normalized = sorted(
        (_normalized_fixture(item) for item in fixtures),
        key=lambda item: (item["kind"], item["fixture_id"]),
    )
    body = {
        "schema_version": SCHEMA_VERSION,
        "baseline_commit": BASELINE_COMMIT,
        "captured_from": str(captured_from),
        "captured_through": str(captured_through),
        "fixtures": normalized,
        "coverage": _coverage_summary(normalized),
    }
    manifest_hash = _sha256_bytes(_canonical(body))
    signature = bytes(sign_digest(bytes.fromhex(manifest_hash.split(":", 1)[1])))
    if len(signature) != 64:
        raise AcceptanceCorpusV2Error("acceptance corpus signer returned invalid data")
    return {
        **body,
        "manifest_hash": manifest_hash,
        "signing_pubkey_hex": str(signing_pubkey_hex or "").lower(),
        "signature_b64": base64.b64encode(signature).decode("ascii"),
    }


def validate_acceptance_corpus_v2(
    value: Mapping[str, Any],
    *,
    corpus_root: Path,
    expected_signing_pubkey_hash: str,
) -> Dict[str, Any]:
    fields = {
        "schema_version",
        "baseline_commit",
        "captured_from",
        "captured_through",
        "fixtures",
        "coverage",
        "manifest_hash",
        "signing_pubkey_hex",
        "signature_b64",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise AcceptanceCorpusV2Error("acceptance corpus fields are invalid")
    if value.get("schema_version") != SCHEMA_VERSION:
        raise AcceptanceCorpusV2Error("acceptance corpus schema is invalid")
    baseline_commit = str(value.get("baseline_commit") or "").lower()
    if not _COMMIT_RE.fullmatch(baseline_commit) or baseline_commit != BASELINE_COMMIT:
        raise AcceptanceCorpusV2Error("acceptance corpus baseline commit differs")

    captured_from = _timestamp(value.get("captured_from"), "captured_from")
    captured_through = _timestamp(value.get("captured_through"), "captured_through")
    if (captured_through - captured_from).total_seconds() < MIN_CAPTURE_DAYS * 86400:
        raise AcceptanceCorpusV2Error("acceptance corpus covers fewer than 30 days")

    raw_fixtures = value.get("fixtures")
    if not isinstance(raw_fixtures, list):
        raise AcceptanceCorpusV2Error("acceptance corpus fixtures are invalid")
    fixtures = [_normalized_fixture(item) for item in raw_fixtures]
    if fixtures != sorted(fixtures, key=lambda item: (item["kind"], item["fixture_id"])):
        raise AcceptanceCorpusV2Error("acceptance fixtures are not canonical")
    keys = {(item["kind"], item["fixture_id"]) for item in fixtures}
    paths = {item["artifact_path"] for item in fixtures}
    if len(keys) != len(fixtures) or len(paths) != len(fixtures):
        raise AcceptanceCorpusV2Error("acceptance fixtures are duplicated")

    root = Path(corpus_root).resolve(strict=True)
    for item in fixtures:
        candidate = root.joinpath(*Path(item["artifact_path"]).parts)
        if candidate.is_symlink() or not candidate.is_file():
            raise AcceptanceCorpusV2Error("acceptance fixture file is unavailable")
        resolved = candidate.resolve(strict=True)
        try:
            resolved.relative_to(root)
        except ValueError as exc:
            raise AcceptanceCorpusV2Error(
                "acceptance fixture escapes the corpus root"
            ) from exc
        if _sha256_file(resolved) != item["artifact_hash"]:
            raise AcceptanceCorpusV2Error("acceptance fixture hash differs")

    expected_coverage = _coverage_summary(fixtures)
    if value.get("coverage") != expected_coverage:
        raise AcceptanceCorpusV2Error("acceptance corpus coverage summary differs")

    unsigned_body = {
        "schema_version": SCHEMA_VERSION,
        "baseline_commit": baseline_commit,
        "captured_from": str(value["captured_from"]),
        "captured_through": str(value["captured_through"]),
        "fixtures": fixtures,
        "coverage": expected_coverage,
    }
    if any(marker in _canonical(unsigned_body).decode("ascii").lower() for marker in _FORBIDDEN_MANIFEST_MARKERS):
        raise AcceptanceCorpusV2Error(
            "acceptance corpus manifest contains forbidden secret metadata"
        )
    manifest_hash = _sha256_bytes(_canonical(unsigned_body))
    if value.get("manifest_hash") != manifest_hash:
        raise AcceptanceCorpusV2Error("acceptance corpus manifest hash differs")

    public_key_hex = str(value.get("signing_pubkey_hex") or "").lower()
    try:
        public_key = bytes.fromhex(public_key_hex)
        signature = base64.b64decode(str(value.get("signature_b64") or ""), validate=True)
    except (ValueError, TypeError) as exc:
        raise AcceptanceCorpusV2Error(
            "acceptance corpus signature encoding is invalid"
        ) from exc
    if len(public_key) != 32 or len(signature) != 64:
        raise AcceptanceCorpusV2Error("acceptance corpus signature shape is invalid")
    pubkey_hash = _sha256_bytes(public_key)
    if pubkey_hash != str(expected_signing_pubkey_hash or "").lower():
        raise AcceptanceCorpusV2Error("acceptance corpus signer is not approved")
    try:
        Ed25519PublicKey.from_public_bytes(public_key).verify(
            signature,
            bytes.fromhex(manifest_hash.split(":", 1)[1]),
        )
    except Exception as exc:
        raise AcceptanceCorpusV2Error(
            "acceptance corpus signature is invalid"
        ) from exc
    return {
        **unsigned_body,
        "manifest_hash": manifest_hash,
        "signing_pubkey_hex": public_key_hex,
        "signature_b64": str(value["signature_b64"]),
    }


def load_and_validate_acceptance_corpus_v2(
    manifest_path: Path,
    *,
    corpus_root: Path,
    expected_signing_pubkey_hash: str,
) -> Dict[str, Any]:
    try:
        value = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AcceptanceCorpusV2Error(
            "acceptance corpus manifest is unavailable"
        ) from exc
    return validate_acceptance_corpus_v2(
        value,
        corpus_root=corpus_root,
        expected_signing_pubkey_hash=expected_signing_pubkey_hash,
    )


def build_acceptance_corpus_from_index_v2(
    *,
    fixture_index: Sequence[Mapping[str, Any]],
    corpus_root: Path,
    captured_from: str,
    captured_through: str,
    signing_key: Ed25519PrivateKey,
) -> Dict[str, Any]:
    """Hash external fixture bytes and sign their immutable manifest."""

    root = Path(corpus_root).resolve(strict=True)
    fixtures = []
    for raw in fixture_index:
        if not isinstance(raw, Mapping):
            raise AcceptanceCorpusV2Error("acceptance fixture index is invalid")
        required = {
            "kind",
            "fixture_id",
            "captured_at",
            "artifact_path",
            "expected_output_hash",
            "receipt_root",
            "metadata",
        }
        if set(raw) != required:
            raise AcceptanceCorpusV2Error(
                "acceptance fixture index fields are invalid"
            )
        relative = Path(str(raw.get("artifact_path") or ""))
        candidate = root.joinpath(*relative.parts)
        if candidate.is_symlink() or not candidate.is_file():
            raise AcceptanceCorpusV2Error(
                "acceptance fixture file is unavailable"
            )
        resolved = candidate.resolve(strict=True)
        try:
            resolved.relative_to(root)
        except ValueError as exc:
            raise AcceptanceCorpusV2Error(
                "acceptance fixture escapes the corpus root"
            ) from exc
        fixtures.append({**dict(raw), "artifact_hash": _sha256_file(resolved)})
    public_key = signing_key.public_key().public_bytes_raw()
    return build_acceptance_corpus_v2(
        fixtures=fixtures,
        captured_from=captured_from,
        captured_through=captured_through,
        signing_pubkey_hex=public_key.hex(),
        sign_digest=signing_key.sign,
    )


def _load_private_key(path: Path) -> Ed25519PrivateKey:
    try:
        payload = Path(path).read_bytes()
        key = serialization.load_pem_private_key(payload, password=None)
    except (OSError, TypeError, ValueError) as exc:
        raise AcceptanceCorpusV2Error(
            "acceptance signing key is unavailable or invalid"
        ) from exc
    if not isinstance(key, Ed25519PrivateKey):
        raise AcceptanceCorpusV2Error("acceptance signing key is not Ed25519")
    return key


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture-index", required=True, type=Path)
    parser.add_argument("--corpus-root", required=True, type=Path)
    parser.add_argument("--captured-from", required=True)
    parser.add_argument("--captured-through", required=True)
    parser.add_argument("--signing-key", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise AcceptanceCorpusV2Error(
            "acceptance corpus manifest already exists; overwrite is forbidden"
        )
    try:
        fixture_index = json.loads(args.fixture_index.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AcceptanceCorpusV2Error(
            "acceptance fixture index is unavailable or invalid"
        ) from exc
    if not isinstance(fixture_index, list):
        raise AcceptanceCorpusV2Error("acceptance fixture index must be an array")
    manifest = build_acceptance_corpus_from_index_v2(
        fixture_index=fixture_index,
        corpus_root=args.corpus_root,
        captured_from=args.captured_from,
        captured_through=args.captured_through,
        signing_key=_load_private_key(args.signing_key),
    )
    signer_hash = _sha256_bytes(bytes.fromhex(manifest["signing_pubkey_hex"]))
    validate_acceptance_corpus_v2(
        manifest,
        corpus_root=args.corpus_root,
        expected_signing_pubkey_hash=signer_hash,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    descriptor = args.output.open("x", encoding="utf-8")
    try:
        json.dump(manifest, descriptor, sort_keys=True, indent=2)
        descriptor.write("\n")
    finally:
        descriptor.close()
    args.output.chmod(0o600)
    print("acceptance_corpus_manifest=%s" % args.output)
    print("acceptance_signer_pubkey_hash=%s" % signer_hash)
    print("acceptance_manifest_hash=%s" % manifest["manifest_hash"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
