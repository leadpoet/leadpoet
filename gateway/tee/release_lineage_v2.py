"""Verify receipt ancestry across independently attested V2 releases."""

from __future__ import annotations

from functools import lru_cache
import json
import re
from typing import Any, Callable, Dict, Mapping, Sequence

from gateway.tee.release_manifest_v2 import (
    role_expectation,
    validate_release_manifest,
)
from leadpoet_canonical.attested_v2 import sha256_json, verify_boot_identity_nitro


_RELEASE_CHANNEL_BUCKET = "leadpoet-attested-v2-artifacts-493765492819"
_RELEASE_CHANNEL_PREFIX = "attested-v2/releases"
_RELEASE_CHANNEL_SCHEMA = "leadpoet.attested_release_channel.v2"
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")


class ReleaseLineageV2Error(RuntimeError):
    """A receipt ancestor is not bound to an approved V2 release."""


_VALIDATOR_PHYSICAL_ROLE = "validator_weights"


def _required_commits(
    parent_graphs: Sequence[Mapping[str, Any]],
) -> set[str]:
    commits = {
        str(identity.get("commit_sha") or "").lower()
        for graph in parent_graphs
        for identity in graph.get("boot_identities") or ()
        if isinstance(identity, Mapping)
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

    from gateway.tee.release_channel_v2 import validate_release_channel_v2

    normalized_channel = validate_release_channel_v2(
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
    release_channel_loader: Callable[[str], Mapping[str, Any]] | None = None,
) -> Dict[str, Dict[str, Any]]:
    """Load exact manifests for every boot commit present in parent graphs."""

    current = validate_release_manifest(current_release)
    required = _required_commits(parent_graphs)
    required_validator_commits = {
        str(identity.get("commit_sha") or "").lower()
        for graph in parent_graphs
        for identity in graph.get("boot_identities") or ()
        if isinstance(identity, Mapping)
        and str(identity.get("physical_role") or "")
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
        release = validate_release_manifest(manifest)
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
        approved_gateway[normalized_commit] = validate_release_manifest(
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
        expectation = role_expectation(release, physical_role)
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
