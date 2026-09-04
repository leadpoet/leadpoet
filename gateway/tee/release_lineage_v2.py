"""Verify receipt ancestry across independently attested V2 releases."""

from __future__ import annotations

from functools import lru_cache
import json
import re
from typing import Any, Callable, Dict, Mapping, Sequence

from gateway.tee.release_manifest_v2 import (
    HISTORICAL_THREE_ROLE_TOPOLOGY_HASH,
    prior_role_expectation,
    validate_prior_release_manifest,
    validate_release_manifest,
)
from gateway.tee.topology import ROLE_SPECS
from leadpoet_canonical.attested_v2 import (
    CHECKPOINTED_RECEIPT_GRAPH_SCHEMA_VERSION,
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
_VALIDATOR_PHYSICAL_ROLE = "validator_weights"
_APPROVED_RELEASE_ROLES = frozenset(ROLE_SPECS) | {_VALIDATOR_PHYSICAL_ROLE}
_HISTORICAL_RELEASE_ROLES = _APPROVED_RELEASE_ROLES | {
    "gateway_autoresearch"
}


class ReleaseLineageV2Error(RuntimeError):
    """A receipt ancestor is not bound to an approved V2 release."""


def _validate_compact_release_lineage_v2(
    value: Mapping[str, Any],
    *,
    expected_current_commit: str | None = None,
    expected_current_gateway_release_hash: str | None = None,
    allow_historical_current: bool = False,
    expected_historical_topology_hash: str | None = None,
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
        if expected_historical_topology_hash is not None:
            if (
                expected_historical_topology_hash
                != HISTORICAL_THREE_ROLE_TOPOLOGY_HASH
            ):
                raise ReleaseLineageV2Error(
                    "historical release topology hash is unsupported"
                )
            allowed_roles = {_HISTORICAL_RELEASE_ROLES}
        else:
            allowed_roles = (
                {_APPROVED_RELEASE_ROLES}
                if commit == current_commit and not allow_historical_current
                else {_APPROVED_RELEASE_ROLES, _HISTORICAL_RELEASE_ROLES}
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


def validate_historical_compact_release_lineage_v2(
    value: Mapping[str, Any],
    *,
    expected_topology_hash: str,
    expected_current_commit: str | None = None,
    expected_current_gateway_release_hash: str | None = None,
) -> Dict[str, Any]:
    """Validate an all-three-role lineage for one exact historical target."""

    return _validate_compact_release_lineage_v2(
        value,
        expected_current_commit=expected_current_commit,
        expected_current_gateway_release_hash=(
            expected_current_gateway_release_hash
        ),
        expected_historical_topology_hash=expected_topology_hash,
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


def build_historical_compact_release_lineage_boot_verifier_v2(
    lineage: Mapping[str, Any],
    *,
    expected_topology_hash: str,
    boot_verifier: Callable[..., Mapping[str, Any]] | None = None,
) -> Callable[[Mapping[str, Any]], Mapping[str, Any]]:
    return _build_compact_release_lineage_boot_verifier_v2(
        lineage,
        lineage_validator=lambda value: (
            validate_historical_compact_release_lineage_v2(
                value,
                expected_topology_hash=expected_topology_hash,
            )
        ),
        boot_verifier=boot_verifier,
    )


def _checkpoint_issuer_boot_identity(
    proof: Mapping[str, Any],
) -> Mapping[str, Any]:
    certificate = proof.get("certificate")
    issuer = (
        certificate.get("issuer_boot_identity")
        if isinstance(certificate, Mapping)
        else None
    )
    if not isinstance(issuer, Mapping):
        raise ReleaseLineageV2Error(
            "checkpoint ancestry issuer boot identity is unavailable"
        )
    return issuer


def _required_boot_identities(
    parent_graphs: Sequence[Mapping[str, Any]],
    parent_ancestry_proofs: Sequence[Mapping[str, Any]] = (),
) -> tuple[Mapping[str, Any], ...]:
    identities: list[Mapping[str, Any]] = []
    for graph in parent_graphs:
        if not isinstance(graph, Mapping):
            raise ReleaseLineageV2Error("receipt ancestry graph is invalid")
        for identity in graph.get("boot_identities") or ():
            if not isinstance(identity, Mapping):
                raise ReleaseLineageV2Error(
                    "receipt ancestry boot identity is invalid"
                )
            identities.append(identity)
        if graph.get("schema_version") == (
            CHECKPOINTED_RECEIPT_GRAPH_SCHEMA_VERSION
        ):
            proof = graph.get("ancestry_proof")
            if not isinstance(proof, Mapping):
                raise ReleaseLineageV2Error(
                    "checkpoint ancestry proof is unavailable"
                )
            identities.append(_checkpoint_issuer_boot_identity(proof))
    for proof in parent_ancestry_proofs:
        if not isinstance(proof, Mapping):
            raise ReleaseLineageV2Error("checkpoint ancestry proof is invalid")
        identities.append(_checkpoint_issuer_boot_identity(proof))
    return tuple(identities)


def _required_commits(
    parent_graphs: Sequence[Mapping[str, Any]],
    parent_ancestry_proofs: Sequence[Mapping[str, Any]] = (),
) -> set[str]:
    commits = {
        str(identity.get("commit_sha") or "").lower()
        for identity in _required_boot_identities(
            parent_graphs,
            parent_ancestry_proofs,
        )
    }
    if "" in commits:
        raise ReleaseLineageV2Error(
            "receipt ancestry contains a boot identity without a commit"
        )
    return commits


@lru_cache(maxsize=512)
def _fetch_historical_release(commit: str) -> Dict[str, Any]:
    """Fetch a gateway release without importing validator-only packages."""

    normalized_commit = str(commit or "").lower()
    if not _COMMIT_RE.fullmatch(normalized_commit):
        raise ReleaseLineageV2Error("historical release commit is invalid")

    import boto3

    key = (
        f"{_RELEASE_CHANNEL_PREFIX}/{normalized_commit}/"
        "release-channel-v2.json"
    )
    try:
        response = boto3.client("s3").get_object(
            Bucket=_RELEASE_CHANNEL_BUCKET,
            Key=key,
        )
        channel = json.loads(response["Body"].read())
    except Exception as exc:
        raise ReleaseLineageV2Error(
            "historical release channel is unavailable or invalid"
        ) from exc

    fields = {
        "schema_version",
        "commit_sha",
        "gateway_release_manifest",
        "validator_release_manifest",
        "channel_hash",
    }
    if not isinstance(channel, Mapping) or set(channel) != fields:
        raise ReleaseLineageV2Error("historical release channel fields are invalid")
    if channel.get("schema_version") != _RELEASE_CHANNEL_SCHEMA:
        raise ReleaseLineageV2Error("historical release channel schema is invalid")
    if channel.get("commit_sha") != normalized_commit:
        raise ReleaseLineageV2Error("historical release channel commit differs")
    body = {key: channel[key] for key in fields - {"channel_hash"}}
    if channel.get("channel_hash") != sha256_json(body):
        raise ReleaseLineageV2Error("historical release channel hash differs")

    from gateway.tee.release_channel_v2 import validate_prior_release_channel_v2

    normalized_channel = validate_prior_release_channel_v2(
        channel,
        expected_commit=normalized_commit,
    )
    release = normalized_channel["gateway_release_manifest"]
    if release.get("commit_sha") != normalized_commit:
        raise ReleaseLineageV2Error("historical gateway release commit differs")
    return normalized_channel


def load_approved_release_lineage_v2(
    *,
    current_release: Mapping[str, Any],
    parent_graphs: Sequence[Mapping[str, Any]],
    parent_ancestry_proofs: Sequence[Mapping[str, Any]] = (),
    release_channel_loader: Callable[[str], Mapping[str, Any]] | None = None,
) -> Dict[str, Dict[str, Any]]:
    """Load exact manifests for every boot that can authorize ancestry."""

    current = validate_release_manifest(current_release)
    identities = _required_boot_identities(
        parent_graphs,
        parent_ancestry_proofs,
    )
    required = {
        str(identity.get("commit_sha") or "").lower()
        for identity in identities
    }
    if "" in required:
        raise ReleaseLineageV2Error(
            "receipt ancestry contains a boot identity without a commit"
        )
    required_validator_commits = {
        str(identity.get("commit_sha") or "").lower()
        for identity in identities
        if str(identity.get("physical_role") or "")
        == _VALIDATOR_PHYSICAL_ROLE
    }
    releases: Dict[str, Dict[str, Any]] = {
        str(current["commit_sha"]): current
    }
    loader = release_channel_loader or _fetch_historical_release
    commits_to_load = (required - set(releases)) | required_validator_commits
    for commit in sorted(commits_to_load):
        loaded = loader(commit)
        if not isinstance(loaded, Mapping):
            raise ReleaseLineageV2Error("historical release channel is invalid")
        manifest = loaded.get("gateway_release_manifest", loaded)
        release = validate_prior_release_manifest(manifest)
        if release.get("commit_sha") != commit:
            raise ReleaseLineageV2Error("historical release channel commit differs")
        if commit in required_validator_commits:
            validator_manifest = loaded.get("validator_release_manifest")
            if not isinstance(validator_manifest, Mapping):
                raise ReleaseLineageV2Error(
                    "historical validator release manifest is unavailable"
                )
            try:
                from validator_tee.host.release_v2 import (
                    validate_validator_release_manifest,
                )

                validator_manifest = validate_validator_release_manifest(
                    validator_manifest
                )
            except Exception as exc:
                raise ReleaseLineageV2Error(
                    "historical validator release manifest is invalid"
                ) from exc
            if validator_manifest["release"].get("commit_sha") != commit:
                raise ReleaseLineageV2Error(
                    "historical validator release commit differs"
                )
            releases[commit] = {
                "gateway_release_manifest": release,
                "validator_release_manifest": validator_manifest,
            }
        else:
            releases[commit] = release
    if required - set(releases):
        raise ReleaseLineageV2Error("receipt release lineage is incomplete")
    return releases


def build_release_lineage_boot_verifier_v2(
    releases: Mapping[str, Mapping[str, Any]],
):
    """Build a fail-closed Nitro verifier for approved release manifests."""

    approved_gateway = {}
    approved_validator = {}
    for commit, entry in releases.items():
        normalized_commit = str(commit).lower()
        gateway_manifest = entry.get("gateway_release_manifest", entry)
        approved_gateway[normalized_commit] = validate_prior_release_manifest(
            gateway_manifest
        )
        validator_manifest = entry.get("validator_release_manifest")
        if validator_manifest is not None:
            try:
                from validator_tee.host.release_v2 import (
                    validator_release_authority,
                )

                approved_validator[normalized_commit] = (
                    validator_release_authority(validator_manifest)
                )
            except Exception as exc:
                raise ReleaseLineageV2Error(
                    "approved validator release manifest is invalid"
                ) from exc

    def verify(identity: Mapping[str, Any]) -> Mapping[str, Any]:
        commit = str(identity.get("commit_sha") or "").lower()
        physical_role = str(identity.get("physical_role") or "")
        if physical_role == _VALIDATOR_PHYSICAL_ROLE:
            release = approved_validator.get(commit)
            if release is None:
                raise ReleaseLineageV2Error(
                    "validator boot commit is absent from approved V2 release lineage"
                )
            expectation = {
                "role": _VALIDATOR_PHYSICAL_ROLE,
                "physical_role": _VALIDATOR_PHYSICAL_ROLE,
                "commit_sha": release["commit_sha"],
                "pcr0": release["pcr0"],
                "build_manifest_hash": release["app_manifest_hash"],
                "dependency_lock_hash": release["dependency_lock_hash"],
            }
            for field, expected in expectation.items():
                if identity.get(field) != expected:
                    raise ReleaseLineageV2Error(
                        f"validator boot {field} differs from approved V2 release lineage"
                    )
            return verify_boot_identity_nitro(
                identity,
                expected_pcr0=release["pcr0"],
                certificate_validity_at_attestation_time=True,
            )

        release = approved_gateway.get(commit)
        if release is None:
            raise ReleaseLineageV2Error(
                "boot commit is absent from approved V2 release lineage"
            )
        expectation = prior_role_expectation(release, physical_role)
        for field in (
            "commit_sha",
            "pcr0",
            "build_manifest_hash",
            "dependency_lock_hash",
        ):
            if identity.get(field) != expectation[field]:
                raise ReleaseLineageV2Error(
                    f"boot {field} differs from approved V2 release lineage"
                )
        return verify_boot_identity_nitro(
            identity,
            expected_pcr0=expectation["pcr0"],
            certificate_validity_at_attestation_time=True,
        )

    return verify
