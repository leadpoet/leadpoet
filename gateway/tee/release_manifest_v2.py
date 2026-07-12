"""Canonical release gate for independently reproduced gateway role EIFs.

The gateway and validator each build every physical role three times from the
same clean Git commit.  A release manifest is accepted only when all six builds
agree on every deterministic identity field.  The manifest is deployment
metadata; it contains no credentials and is not itself measured into the EIF.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

from gateway.tee.topology import ROLE_SPECS, topology_hash


BUILD_EVIDENCE_SCHEMA_VERSION = "leadpoet.gateway_role_build_evidence.v2"
RELEASE_MANIFEST_SCHEMA_VERSION = "leadpoet.gateway_release_manifest.v2"
BUILDER_DOMAINS = frozenset({"gateway", "validator"})
BUILDS_PER_DOMAIN = 3

_HASH_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_PCR0_RE = re.compile(r"^[0-9a-f]{96}$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$")
_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")

DETERMINISTIC_FIELDS = (
    "commit_sha",
    "pcr0",
    "normalized_image_hash",
    "eif_hash",
    "source_manifest_hash",
    "build_identity_hash",
    "execution_manifest_hash",
    "dependency_lock_hash",
    "dockerfile_hash",
    "topology_hash",
)


class ReleaseManifestV2Error(ValueError):
    """Independent build evidence is incomplete, divergent, or malformed."""


def _canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise ReleaseManifestV2Error("release value is not canonical JSON") from exc


def _sha256_json(value: Any) -> str:
    return "sha256:" + hashlib.sha256(_canonical_json(value)).hexdigest()


def _hash(value: Any, field: str) -> str:
    normalized = str(value or "").strip().lower()
    if not _HASH_RE.fullmatch(normalized):
        raise ReleaseManifestV2Error("%s must be sha256:<64 lowercase hex>" % field)
    return normalized


def _identifier(value: Any, field: str) -> str:
    normalized = str(value or "").strip()
    if not _IDENTIFIER_RE.fullmatch(normalized):
        raise ReleaseManifestV2Error("%s is invalid" % field)
    return normalized


def normalize_build_evidence(value: Mapping[str, Any]) -> Dict[str, Any]:
    fields = {
        "schema_version",
        "builder_domain",
        "builder_id",
        "build_ordinal",
        "physical_role",
        "service_role",
        *DETERMINISTIC_FIELDS,
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ReleaseManifestV2Error("build evidence fields do not match schema")
    if value.get("schema_version") != BUILD_EVIDENCE_SCHEMA_VERSION:
        raise ReleaseManifestV2Error("unsupported build evidence schema")
    builder_domain = str(value.get("builder_domain") or "")
    if builder_domain not in BUILDER_DOMAINS:
        raise ReleaseManifestV2Error("build evidence domain is invalid")
    role = str(value.get("physical_role") or "")
    if role not in ROLE_SPECS:
        raise ReleaseManifestV2Error("build evidence role is invalid")
    service_role = str(value.get("service_role") or "")
    if service_role != ROLE_SPECS[role]["service_role"]:
        raise ReleaseManifestV2Error("build evidence service role mismatch")
    ordinal = value.get("build_ordinal")
    if not isinstance(ordinal, int) or isinstance(ordinal, bool) or not 1 <= ordinal <= BUILDS_PER_DOMAIN:
        raise ReleaseManifestV2Error("build evidence ordinal is invalid")
    commit = str(value.get("commit_sha") or "").strip().lower()
    if not _COMMIT_RE.fullmatch(commit):
        raise ReleaseManifestV2Error("build evidence commit is invalid")
    pcr0 = str(value.get("pcr0") or "").strip().lower()
    if not _PCR0_RE.fullmatch(pcr0) or pcr0 == "0" * 96:
        raise ReleaseManifestV2Error("build evidence PCR0 is invalid")
    normalized = {
        "schema_version": BUILD_EVIDENCE_SCHEMA_VERSION,
        "builder_domain": builder_domain,
        "builder_id": _identifier(value.get("builder_id"), "builder_id"),
        "build_ordinal": ordinal,
        "physical_role": role,
        "service_role": service_role,
        "commit_sha": commit,
        "pcr0": pcr0,
    }
    for field in DETERMINISTIC_FIELDS:
        if field in {"commit_sha", "pcr0"}:
            continue
        normalized[field] = _hash(value.get(field), field)
    return normalized


def build_release_manifest(evidence: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
    normalized = [normalize_build_evidence(item) for item in evidence]
    expected_count = len(ROLE_SPECS) * len(BUILDER_DOMAINS) * BUILDS_PER_DOMAIN
    if len(normalized) != expected_count:
        raise ReleaseManifestV2Error(
            "release needs exactly %s independent build records" % expected_count
        )
    unique_keys = {
        (
            item["physical_role"],
            item["builder_domain"],
            item["builder_id"],
            item["build_ordinal"],
        )
        for item in normalized
    }
    if len(unique_keys) != len(normalized):
        raise ReleaseManifestV2Error("release build evidence is duplicated")

    role_documents = {}
    release_commit = None
    for role in sorted(ROLE_SPECS):
        role_evidence = [item for item in normalized if item["physical_role"] == role]
        if len(role_evidence) != len(BUILDER_DOMAINS) * BUILDS_PER_DOMAIN:
            raise ReleaseManifestV2Error("role build evidence is incomplete: %s" % role)
        for domain in BUILDER_DOMAINS:
            domain_evidence = [
                item for item in role_evidence if item["builder_domain"] == domain
            ]
            if len(domain_evidence) != BUILDS_PER_DOMAIN:
                raise ReleaseManifestV2Error(
                    "%s role needs three %s builds" % (role, domain)
                )
            builder_ids = {item["builder_id"] for item in domain_evidence}
            if len(builder_ids) != 1:
                raise ReleaseManifestV2Error(
                    "%s %s builds came from multiple parent identities" % (role, domain)
                )
            if {item["build_ordinal"] for item in domain_evidence} != {1, 2, 3}:
                raise ReleaseManifestV2Error(
                    "%s %s build ordinals are incomplete" % (role, domain)
                )
        for field in DETERMINISTIC_FIELDS:
            if len({item[field] for item in role_evidence}) != 1:
                raise ReleaseManifestV2Error(
                    "%s independent builds diverged at %s" % (role, field)
                )
        summary = {
            "physical_role": role,
            "service_role": ROLE_SPECS[role]["service_role"],
            **{field: role_evidence[0][field] for field in DETERMINISTIC_FIELDS},
            "verified_build_count": len(role_evidence),
            "builder_domains": sorted(BUILDER_DOMAINS),
        }
        role_documents[role] = summary
        if release_commit is None:
            release_commit = summary["commit_sha"]
        elif release_commit != summary["commit_sha"]:
            raise ReleaseManifestV2Error("gateway roles were built from different commits")
        if summary["topology_hash"] != topology_hash():
            raise ReleaseManifestV2Error("role topology hash differs from canonical topology")

    sorted_evidence = sorted(
        normalized,
        key=lambda item: (
            item["physical_role"],
            item["builder_domain"],
            item["builder_id"],
            item["build_ordinal"],
        ),
    )
    body = {
        "schema_version": RELEASE_MANIFEST_SCHEMA_VERSION,
        "commit_sha": release_commit,
        "topology_hash": topology_hash(),
        "roles": role_documents,
        "build_evidence_root": _sha256_json(sorted_evidence),
        "verified_build_count": len(sorted_evidence),
    }
    return {**body, "release_hash": _sha256_json(body)}


def validate_release_manifest(value: Mapping[str, Any]) -> Dict[str, Any]:
    fields = {
        "schema_version",
        "commit_sha",
        "topology_hash",
        "roles",
        "build_evidence_root",
        "verified_build_count",
        "release_hash",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ReleaseManifestV2Error("release manifest fields do not match schema")
    if value.get("schema_version") != RELEASE_MANIFEST_SCHEMA_VERSION:
        raise ReleaseManifestV2Error("unsupported release manifest schema")
    if set(value.get("roles") or {}) != set(ROLE_SPECS):
        raise ReleaseManifestV2Error("release manifest roles are incomplete")
    if value.get("topology_hash") != topology_hash():
        raise ReleaseManifestV2Error("release manifest topology hash mismatch")
    if value.get("verified_build_count") != len(ROLE_SPECS) * 6:
        raise ReleaseManifestV2Error("release manifest build count is invalid")
    commit = str(value.get("commit_sha") or "").lower()
    if not _COMMIT_RE.fullmatch(commit):
        raise ReleaseManifestV2Error("release manifest commit is invalid")
    _hash(value.get("build_evidence_root"), "build_evidence_root")
    for role, summary in value["roles"].items():
        expected_fields = {
            "physical_role",
            "service_role",
            *DETERMINISTIC_FIELDS,
            "verified_build_count",
            "builder_domains",
        }
        if not isinstance(summary, Mapping) or set(summary) != expected_fields:
            raise ReleaseManifestV2Error("release role summary fields are invalid")
        if summary["physical_role"] != role:
            raise ReleaseManifestV2Error("release role summary name mismatch")
        if summary["service_role"] != ROLE_SPECS[role]["service_role"]:
            raise ReleaseManifestV2Error("release role service mismatch")
        if summary["commit_sha"] != commit:
            raise ReleaseManifestV2Error("release role commit mismatch")
        if summary["topology_hash"] != topology_hash():
            raise ReleaseManifestV2Error("release role topology mismatch")
        if summary["verified_build_count"] != 6:
            raise ReleaseManifestV2Error("release role build count is invalid")
        if summary["builder_domains"] != sorted(BUILDER_DOMAINS):
            raise ReleaseManifestV2Error("release role builder domains are invalid")
        pcr0 = str(summary["pcr0"] or "")
        if not _PCR0_RE.fullmatch(pcr0) or pcr0 == "0" * 96:
            raise ReleaseManifestV2Error("release role PCR0 is invalid")
        for field in DETERMINISTIC_FIELDS:
            if field in {"commit_sha", "pcr0"}:
                continue
            _hash(summary[field], field)
    body = {field: value[field] for field in fields if field != "release_hash"}
    if value.get("release_hash") != _sha256_json(body):
        raise ReleaseManifestV2Error("release manifest hash mismatch")
    return dict(value)


def role_expectation(manifest: Mapping[str, Any], role: str) -> Dict[str, str]:
    release = validate_release_manifest(manifest)
    if role not in ROLE_SPECS:
        raise ReleaseManifestV2Error("unknown release role")
    summary = release["roles"][role]
    return {
        "physical_role": role,
        "service_role": summary["service_role"],
        "commit_sha": summary["commit_sha"],
        "pcr0": summary["pcr0"],
        "build_manifest_hash": summary["execution_manifest_hash"],
        "dependency_lock_hash": summary["dependency_lock_hash"],
        "build_identity_hash": summary["build_identity_hash"],
        "release_hash": release["release_hash"],
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", action="append", type=Path, default=[])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args(argv)
    if bool(args.verify) == bool(args.evidence):
        raise ReleaseManifestV2Error("select --verify or one or more --evidence files")
    if args.verify:
        release = validate_release_manifest(
            json.loads(args.verify.read_text(encoding="utf-8"))
        )
    else:
        evidence = []
        for path in args.evidence:
            value = json.loads(path.read_text(encoding="utf-8"))
            evidence.extend(value if isinstance(value, list) else [value])
        release = build_release_manifest(evidence)
        if not args.output:
            raise ReleaseManifestV2Error("--output is required when building a release")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(release, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
    print("gateway_v2_release_hash=%s" % release["release_hash"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
