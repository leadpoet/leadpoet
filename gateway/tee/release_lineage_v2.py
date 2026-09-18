"""Verify measured gateway boots against immutable release lineage."""

from __future__ import annotations

import re
from typing import Any, Callable, Dict, Mapping

from gateway.tee.release_manifest_v2 import validate_release_manifest
from gateway.tee.topology import ROLE_SPECS
from leadpoet_canonical.attested_v2 import (
    sha256_json,
    verify_boot_identity_nitro,
)


_RELEASE_CHANNEL_BUCKET = "leadpoet-attested-v2-artifacts-493765492819"
_RELEASE_CHANNEL_PREFIX = "attested-v2/releases"
_RELEASE_CHANNEL_SCHEMA = "leadpoet.attested_release_channel.v2"
_COMPACT_LINEAGE_SCHEMA = "leadpoet.attested_release_lineage.v1"
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_HASH_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_PCR0_RE = re.compile(r"^[0-9a-f]{96}$")
_MAX_COMPACT_RELEASES = 512
_APPROVED_RELEASE_ROLES = frozenset(ROLE_SPECS)
_PRIOR_RELEASE_ROLES = _APPROVED_RELEASE_ROLES | {"gateway_scoring"}


class ReleaseLineageV2Error(RuntimeError):
    """A receipt ancestor is not bound to an approved V2 release."""


def _validate_compact_release_lineage_v2(
    value: Mapping[str, Any],
    *,
    expected_current_commit: str | None = None,
    expected_current_gateway_release_hash: str | None = None,
    allow_historical_current: bool = False,
) -> Dict[str, Any]:
    """Validate the immutable compact release authority used inside enclaves."""

    fields = {
        "schema_version",
        "current_commit_sha",
        "current_gateway_release_hash",
        "releases",
        "lineage_hash",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ReleaseLineageV2Error("compact release lineage fields are invalid")
    if value.get("schema_version") != _COMPACT_LINEAGE_SCHEMA:
        raise ReleaseLineageV2Error("compact release lineage schema is invalid")
    current_commit = str(value.get("current_commit_sha") or "").lower()
    current_release_hash = str(
        value.get("current_gateway_release_hash") or ""
    ).lower()
    releases = value.get("releases")
    if (
        not _COMMIT_RE.fullmatch(current_commit)
        or not _HASH_RE.fullmatch(current_release_hash)
        or not isinstance(releases, Mapping)
        or not 1 <= len(releases) <= _MAX_COMPACT_RELEASES
    ):
        raise ReleaseLineageV2Error("compact release lineage is invalid")

    normalized_releases: Dict[str, Any] = {}
    for release_commit, release in releases.items():
        commit = str(release_commit or "").lower()
        if (
            not _COMMIT_RE.fullmatch(commit)
            or not isinstance(release, Mapping)
            or set(release)
            != {"channel_hash", "gateway_release_hash", "roles"}
            or not _HASH_RE.fullmatch(str(release.get("channel_hash") or ""))
            or not _HASH_RE.fullmatch(
                str(release.get("gateway_release_hash") or "")
            )
        ):
            raise ReleaseLineageV2Error(
                "compact release lineage entry is invalid"
            )
        roles = release.get("roles")
        observed_roles = set(roles) if isinstance(roles, Mapping) else set()
        allowed_roles = (
            {_APPROVED_RELEASE_ROLES}
            if commit == current_commit and not allow_historical_current
            else {
                _APPROVED_RELEASE_ROLES,
                _PRIOR_RELEASE_ROLES,
            }
        )
        if not isinstance(roles, Mapping) or frozenset(observed_roles) not in allowed_roles:
            raise ReleaseLineageV2Error(
                "compact release lineage roles are incomplete"
            )
        normalized_roles: Dict[str, Dict[str, str]] = {}
        for role, expectation in roles.items():
            if (
                not isinstance(expectation, Mapping)
                or set(expectation)
                != {
                    "commit_sha",
                    "pcr0",
                    "build_manifest_hash",
                    "dependency_lock_hash",
                }
            ):
                raise ReleaseLineageV2Error(
                    "compact release expectation is invalid"
                )
            expected_commit = str(expectation.get("commit_sha") or "").lower()
            expected_pcr0 = str(expectation.get("pcr0") or "").lower()
            expected_manifest = str(
                expectation.get("build_manifest_hash") or ""
            ).lower()
            expected_lock = str(
                expectation.get("dependency_lock_hash") or ""
            ).lower()
            if (
                expected_commit != commit
                or not _PCR0_RE.fullmatch(expected_pcr0)
                or expected_pcr0 == "0" * 96
                or not _HASH_RE.fullmatch(expected_manifest)
                or not _HASH_RE.fullmatch(expected_lock)
            ):
                raise ReleaseLineageV2Error(
                    "compact release expectation is invalid"
                )
            normalized_roles[str(role)] = {
                "commit_sha": expected_commit,
                "pcr0": expected_pcr0,
                "build_manifest_hash": expected_manifest,
                "dependency_lock_hash": expected_lock,
            }
        normalized_releases[commit] = {
            "channel_hash": str(release["channel_hash"]).lower(),
            "gateway_release_hash": str(
                release["gateway_release_hash"]
            ).lower(),
            "roles": {
                role: normalized_roles[role] for role in sorted(normalized_roles)
            },
        }

    current = normalized_releases.get(current_commit)
    if current is None or current["gateway_release_hash"] != current_release_hash:
        raise ReleaseLineageV2Error(
            "current gateway release is absent from compact lineage"
        )
    body = {
        "schema_version": _COMPACT_LINEAGE_SCHEMA,
        "current_commit_sha": current_commit,
        "current_gateway_release_hash": current_release_hash,
        "releases": {
            commit: normalized_releases[commit]
            for commit in sorted(normalized_releases)
        },
    }
    if value.get("lineage_hash") != sha256_json(body):
        raise ReleaseLineageV2Error("compact release lineage hash differs")
    if (
        expected_current_commit is not None
        and current_commit != str(expected_current_commit).lower()
    ):
        raise ReleaseLineageV2Error("compact release lineage commit differs")
    if (
        expected_current_gateway_release_hash is not None
        and current_release_hash
        != str(expected_current_gateway_release_hash).lower()
    ):
        raise ReleaseLineageV2Error("compact release lineage release differs")
    return {**body, "lineage_hash": str(value["lineage_hash"])}


def validate_compact_release_lineage_v2(
    value: Mapping[str, Any],
    *,
    expected_current_commit: str | None = None,
    expected_current_gateway_release_hash: str | None = None,
) -> Dict[str, Any]:
    """Validate the canonical current-topology compact lineage."""

    return _validate_compact_release_lineage_v2(
        value,
        expected_current_commit=expected_current_commit,
        expected_current_gateway_release_hash=(
            expected_current_gateway_release_hash
        ),
        allow_historical_current=False,
    )


def validate_prior_compact_release_lineage_v2(
    value: Mapping[str, Any],
    *,
    expected_current_commit: str | None = None,
    expected_current_gateway_release_hash: str | None = None,
) -> Dict[str, Any]:
    """Validate an installed prior lineage across the one retired role set."""

    return _validate_compact_release_lineage_v2(
        value,
        expected_current_commit=expected_current_commit,
        expected_current_gateway_release_hash=(
            expected_current_gateway_release_hash
        ),
        allow_historical_current=True,
    )


def _build_compact_release_lineage_boot_verifier_v2(
    lineage: Mapping[str, Any],
    *,
    lineage_validator: Callable[[Mapping[str, Any]], Mapping[str, Any]],
    boot_verifier: Callable[..., Mapping[str, Any]] | None = None,
) -> Callable[[Mapping[str, Any]], Mapping[str, Any]]:
    """Build an exact-PCR verifier from one hash-bound compact lineage."""

    normalized = lineage_validator(lineage)
    verify_nitro = boot_verifier or verify_boot_identity_nitro

    def verify(identity: Mapping[str, Any]) -> Mapping[str, Any]:
        commit = str(identity.get("commit_sha") or "").lower()
        physical_role = str(identity.get("physical_role") or "")
        release = normalized["releases"].get(commit)
        if not isinstance(release, Mapping):
            raise ReleaseLineageV2Error(
                "boot commit is absent from compact release lineage"
            )
        roles = release.get("roles")
        expectation = roles.get(physical_role) if isinstance(roles, Mapping) else None
        if not isinstance(expectation, Mapping):
            raise ReleaseLineageV2Error(
                "boot role is absent from compact release lineage"
            )
        for field in (
            "commit_sha",
            "pcr0",
            "build_manifest_hash",
            "dependency_lock_hash",
        ):
            if identity.get(field) != expectation[field]:
                raise ReleaseLineageV2Error(
                    "boot %s differs from compact release lineage" % field
                )
        return verify_nitro(
            identity,
            expected_pcr0=str(expectation["pcr0"]),
            certificate_validity_at_attestation_time=True,
        )

    return verify


def build_compact_release_lineage_boot_verifier_v2(
    lineage: Mapping[str, Any],
    *,
    boot_verifier: Callable[..., Mapping[str, Any]] | None = None,
) -> Callable[[Mapping[str, Any]], Mapping[str, Any]]:
    return _build_compact_release_lineage_boot_verifier_v2(
        lineage,
        lineage_validator=validate_compact_release_lineage_v2,
        boot_verifier=boot_verifier,
    )
