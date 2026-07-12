"""Build and validate deterministic metadata for the validator V2 EIF."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
from typing import Any, Dict, Mapping, Optional, Sequence


VALIDATOR_RELEASE_SCHEMA_VERSION = "leadpoet.validator_release.v2"
VALIDATOR_BUILD_EVIDENCE_SCHEMA_VERSION = "leadpoet.validator_build_evidence.v2"
VALIDATOR_RELEASE_MANIFEST_SCHEMA_VERSION = "leadpoet.validator_release_manifest.v2"
VALIDATOR_BUILDER_DOMAINS = frozenset({"gateway", "validator"})
VALIDATOR_BUILDS_PER_DOMAIN = 3
_HASH_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$")
_PCR0_RE = re.compile(r"^[0-9a-f]{96}$")
_FIELDS = {
    "schema_version",
    "commit_sha",
    "pcr0",
    "app_manifest_hash",
    "dependency_lock_hash",
    "normalized_image_hash",
    "eif_hash",
    "dockerfile_hash",
    "base_dockerfile_hash",
    "release_hash",
}
_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")


class ValidatorReleaseV2Error(ValueError):
    """Validator build output is incomplete, divergent, or malformed."""


def _sha256_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _canonical_hash(value: Mapping[str, Any]) -> str:
    return _sha256_bytes(
        json.dumps(
            dict(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("ascii")
    )


def _hash(value: Any, field: str) -> str:
    normalized = str(value or "").strip().lower()
    if not _HASH_RE.fullmatch(normalized):
        raise ValidatorReleaseV2Error("%s is invalid" % field)
    return normalized


def parse_pcr0(output: str) -> str:
    candidates = []
    try:
        candidates.append(json.loads(output))
    except json.JSONDecodeError:
        pass
    for start in (
        index for index, character in enumerate(str(output or "")) if character == "{"
    ):
        try:
            candidates.append(json.loads(output[start:]))
        except json.JSONDecodeError:
            continue
    for value in candidates:
        if not isinstance(value, Mapping):
            continue
        measurements = value.get("Measurements")
        if not isinstance(measurements, Mapping):
            continue
        pcr0 = str(measurements.get("PCR0") or "").strip().lower()
        if _PCR0_RE.fullmatch(pcr0) and pcr0 != "0" * 96:
            return pcr0
    raise ValidatorReleaseV2Error("validator build output lacks a valid PCR0")


def build_validator_release(
    *,
    commit_sha: str,
    pcr0: str,
    app_manifest_hash: str,
    dependency_lock_hash: str,
    normalized_image_hash: str,
    eif_hash: str,
    dockerfile_hash: str,
    base_dockerfile_hash: str,
) -> Dict[str, Any]:
    commit = str(commit_sha or "").strip().lower()
    normalized_pcr0 = str(pcr0 or "").strip().lower()
    if not _COMMIT_RE.fullmatch(commit):
        raise ValidatorReleaseV2Error("validator release commit is invalid")
    if not _PCR0_RE.fullmatch(normalized_pcr0) or normalized_pcr0 == "0" * 96:
        raise ValidatorReleaseV2Error("validator release PCR0 is invalid")
    body = {
        "schema_version": VALIDATOR_RELEASE_SCHEMA_VERSION,
        "commit_sha": commit,
        "pcr0": normalized_pcr0,
        "app_manifest_hash": _hash(app_manifest_hash, "app_manifest_hash"),
        "dependency_lock_hash": _hash(
            dependency_lock_hash, "dependency_lock_hash"
        ),
        "normalized_image_hash": _hash(
            normalized_image_hash, "normalized_image_hash"
        ),
        "eif_hash": _hash(eif_hash, "eif_hash"),
        "dockerfile_hash": _hash(dockerfile_hash, "dockerfile_hash"),
        "base_dockerfile_hash": _hash(
            base_dockerfile_hash, "base_dockerfile_hash"
        ),
    }
    return {**body, "release_hash": _canonical_hash(body)}


def validate_validator_release(value: Mapping[str, Any]) -> Dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _FIELDS:
        raise ValidatorReleaseV2Error("validator release fields are invalid")
    rebuilt = build_validator_release(
        commit_sha=value["commit_sha"],
        pcr0=value["pcr0"],
        app_manifest_hash=value["app_manifest_hash"],
        dependency_lock_hash=value["dependency_lock_hash"],
        normalized_image_hash=value["normalized_image_hash"],
        eif_hash=value["eif_hash"],
        dockerfile_hash=value["dockerfile_hash"],
        base_dockerfile_hash=value["base_dockerfile_hash"],
    )
    if dict(value) != rebuilt:
        raise ValidatorReleaseV2Error("validator release hash mismatch")
    return rebuilt


def build_validator_build_evidence(
    release: Mapping[str, Any],
    *,
    builder_domain: str,
    builder_id: str,
    build_ordinal: int,
) -> Dict[str, Any]:
    normalized_release = validate_validator_release(release)
    domain = str(builder_domain or "")
    identifier = str(builder_id or "")
    if domain not in VALIDATOR_BUILDER_DOMAINS:
        raise ValidatorReleaseV2Error("validator build evidence domain is invalid")
    if not _IDENTIFIER_RE.fullmatch(identifier):
        raise ValidatorReleaseV2Error("validator build evidence builder is invalid")
    if (
        not isinstance(build_ordinal, int)
        or isinstance(build_ordinal, bool)
        or not 1 <= build_ordinal <= VALIDATOR_BUILDS_PER_DOMAIN
    ):
        raise ValidatorReleaseV2Error("validator build evidence ordinal is invalid")
    return {
        "schema_version": VALIDATOR_BUILD_EVIDENCE_SCHEMA_VERSION,
        "builder_domain": domain,
        "builder_id": identifier,
        "build_ordinal": build_ordinal,
        "release": normalized_release,
    }


def validate_validator_build_evidence(value: Mapping[str, Any]) -> Dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {
        "schema_version",
        "builder_domain",
        "builder_id",
        "build_ordinal",
        "release",
    }:
        raise ValidatorReleaseV2Error("validator build evidence fields are invalid")
    if value.get("schema_version") != VALIDATOR_BUILD_EVIDENCE_SCHEMA_VERSION:
        raise ValidatorReleaseV2Error("validator build evidence schema is invalid")
    return build_validator_build_evidence(
        value.get("release"),
        builder_domain=value.get("builder_domain"),
        builder_id=value.get("builder_id"),
        build_ordinal=value.get("build_ordinal"),
    )


def build_validator_release_manifest(
    evidence: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    normalized = [validate_validator_build_evidence(item) for item in evidence]
    expected_count = len(VALIDATOR_BUILDER_DOMAINS) * VALIDATOR_BUILDS_PER_DOMAIN
    if len(normalized) != expected_count:
        raise ValidatorReleaseV2Error(
            "validator release requires exactly six independent builds"
        )
    keys = {
        (item["builder_domain"], item["builder_id"], item["build_ordinal"])
        for item in normalized
    }
    if len(keys) != len(normalized):
        raise ValidatorReleaseV2Error("validator build evidence is duplicated")
    for domain in VALIDATOR_BUILDER_DOMAINS:
        domain_evidence = [
            item for item in normalized if item["builder_domain"] == domain
        ]
        if len(domain_evidence) != VALIDATOR_BUILDS_PER_DOMAIN:
            raise ValidatorReleaseV2Error(
                "validator release requires three %s builds" % domain
            )
        if len({item["builder_id"] for item in domain_evidence}) != 1:
            raise ValidatorReleaseV2Error(
                "validator %s builds use multiple parent identities" % domain
            )
        if {item["build_ordinal"] for item in domain_evidence} != {1, 2, 3}:
            raise ValidatorReleaseV2Error(
                "validator %s build ordinals are incomplete" % domain
            )
    releases = [item["release"] for item in normalized]
    for field in sorted(_FIELDS):
        if len({item[field] for item in releases}) != 1:
            raise ValidatorReleaseV2Error(
                "independent validator builds diverged at %s" % field
            )
    sorted_evidence = sorted(
        normalized,
        key=lambda item: (
            item["builder_domain"],
            item["builder_id"],
            item["build_ordinal"],
        ),
    )
    body = {
        "schema_version": VALIDATOR_RELEASE_MANIFEST_SCHEMA_VERSION,
        "release": releases[0],
        "build_evidence_root": _sha256_bytes(
            json.dumps(
                sorted_evidence,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
            ).encode("ascii")
        ),
        "verified_build_count": expected_count,
        "builder_domains": sorted(VALIDATOR_BUILDER_DOMAINS),
    }
    return {**body, "release_manifest_hash": _canonical_hash(body)}


def validate_validator_release_manifest(value: Mapping[str, Any]) -> Dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {
        "schema_version",
        "release",
        "build_evidence_root",
        "verified_build_count",
        "builder_domains",
        "release_manifest_hash",
    }:
        raise ValidatorReleaseV2Error("validator release manifest fields are invalid")
    if value.get("schema_version") != VALIDATOR_RELEASE_MANIFEST_SCHEMA_VERSION:
        raise ValidatorReleaseV2Error("validator release manifest schema is invalid")
    release = validate_validator_release(value.get("release"))
    build_evidence_root = _hash(
        value.get("build_evidence_root"),
        "build_evidence_root",
    )
    if value.get("verified_build_count") != 6:
        raise ValidatorReleaseV2Error("validator release build count is invalid")
    if value.get("builder_domains") != sorted(VALIDATOR_BUILDER_DOMAINS):
        raise ValidatorReleaseV2Error("validator release builder domains are invalid")
    body = {
        "schema_version": VALIDATOR_RELEASE_MANIFEST_SCHEMA_VERSION,
        "release": release,
        "build_evidence_root": build_evidence_root,
        "verified_build_count": 6,
        "builder_domains": sorted(VALIDATOR_BUILDER_DOMAINS),
    }
    if value.get("release_manifest_hash") != _canonical_hash(body):
        raise ValidatorReleaseV2Error("validator release manifest hash mismatch")
    return {**body, "release_manifest_hash": value["release_manifest_hash"]}


def validator_release_authority(value: Mapping[str, Any]) -> Dict[str, Any]:
    """Return the six-build-authorized deterministic validator release."""

    return validate_validator_release_manifest(value)["release"]


def _git_commit(repo_root: Path) -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout.strip().lower()
    except Exception as exc:
        raise ValidatorReleaseV2Error("validator release commit is unavailable") from exc


def release_from_build_outputs(
    *,
    repo_root: Path,
    measurements_path: Path,
    eif_path: Path,
    app_manifest_hash: str,
    dependency_lock_hash: str,
    normalized_image_hash: str,
) -> Dict[str, Any]:
    try:
        measurements = measurements_path.read_text(encoding="utf-8")
        eif_bytes = eif_path.read_bytes()
        dockerfile = (repo_root / "validator_tee" / "Dockerfile.enclave").read_bytes()
        base_dockerfile = (repo_root / "validator_tee" / "Dockerfile.base").read_bytes()
    except OSError as exc:
        raise ValidatorReleaseV2Error("validator build output is unavailable") from exc
    return build_validator_release(
        commit_sha=_git_commit(repo_root),
        pcr0=parse_pcr0(measurements),
        app_manifest_hash=app_manifest_hash,
        dependency_lock_hash=dependency_lock_hash,
        normalized_image_hash=normalized_image_hash,
        eif_hash=_sha256_bytes(eif_bytes),
        dockerfile_hash=_sha256_bytes(dockerfile),
        base_dockerfile_hash=_sha256_bytes(base_dockerfile),
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--measurements", type=Path, required=True)
    parser.add_argument("--eif", type=Path, required=True)
    parser.add_argument("--app-manifest-hash", required=True)
    parser.add_argument("--dependency-lock-hash", required=True)
    parser.add_argument("--normalized-image-hash", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    release = release_from_build_outputs(
        repo_root=args.repo_root.resolve(),
        measurements_path=args.measurements.resolve(),
        eif_path=args.eif.resolve(),
        app_manifest_hash=args.app_manifest_hash,
        dependency_lock_hash=args.dependency_lock_hash,
        normalized_image_hash=args.normalized_image_hash,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(release, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print("validator_v2_release_hash=%s" % release["release_hash"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
