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


# Validator enclaves are built dynamically from Git and verified against the
# gateway's reproducible PCR0 build cache, not against a six-build gateway
# release channel. Receipt ancestry legitimately mixes gateway and validator
# boots (an allocation graph now embeds the finalized weight receipt ancestry,
# which carries the validator boot), so validator boots must be routed to
# dynamic PCR0 verification instead of a gateway release-role expectation.
_VALIDATOR_PHYSICAL_ROLE = "validator_weights"


def _is_gateway_release_boot(identity: Mapping[str, Any]) -> bool:
    return str(identity.get("physical_role") or "") != _VALIDATOR_PHYSICAL_ROLE


def _verify_validator_dynamic_boot(identity: Mapping[str, Any]) -> Mapping[str, Any]:
    """Verify a validator boot via the gateway's dynamic PCR0 build cache.

    Mirrors the authoritative weight-submission boot check so scoring and
    submission apply identical trust to the same validator boot identity.
    """

    from gateway.utils.pcr0_builder import verify_pcr0

    rebuilt = verify_pcr0(str(identity.get("pcr0") or ""))
    if not rebuilt.get("valid"):
        raise ReleaseLineageV2Error(
            "validator PCR0 is absent from the dynamic Git build cache"
        )
    if str(rebuilt.get("commit_hash") or "").lower() != str(
        identity.get("commit_sha") or ""
    ).lower():
        raise ReleaseLineageV2Error(
            "validator PCR0 commit differs from boot identity"
        )
    return verify_boot_identity_nitro(
        identity,
        expected_pcr0=str(identity.get("pcr0") or ""),
        certificate_validity_at_attestation_time=True,
    )


def _required_commits(
    parent_graphs: Sequence[Mapping[str, Any]],
) -> set[str]:
    # Only gateway boots require an approved gateway release channel; validator
    # boots are verified out of band via dynamic PCR0.
    commits = {
        str(identity.get("commit_sha") or "").lower()
        for graph in parent_graphs
        for identity in graph.get("boot_identities") or ()
        if isinstance(identity, Mapping) and _is_gateway_release_boot(identity)
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

    release = validate_release_manifest(channel["gateway_release_manifest"])
    if release.get("commit_sha") != normalized_commit:
        raise ReleaseLineageV2Error("historical gateway release commit differs")
    return release


def load_approved_release_lineage_v2(
    *,
    current_release: Mapping[str, Any],
    parent_graphs: Sequence[Mapping[str, Any]],
    release_channel_loader: Callable[[str], Mapping[str, Any]] | None = None,
) -> Dict[str, Dict[str, Any]]:
    """Load exact manifests for every boot commit present in parent graphs."""

    current = validate_release_manifest(current_release)
    required = _required_commits(parent_graphs)
    releases = {str(current["commit_sha"]): current}
    loader = release_channel_loader or _fetch_historical_release
    for commit in sorted(required - set(releases)):
        loaded = loader(commit)
        if not isinstance(loaded, Mapping):
            raise ReleaseLineageV2Error("historical release channel is invalid")
        manifest = loaded.get("gateway_release_manifest", loaded)
        release = validate_release_manifest(manifest)
        if release.get("commit_sha") != commit:
            raise ReleaseLineageV2Error("historical release channel commit differs")
        releases[commit] = release
    if required - set(releases):
        raise ReleaseLineageV2Error("receipt release lineage is incomplete")
    return releases


def build_release_lineage_boot_verifier_v2(
    releases: Mapping[str, Mapping[str, Any]],
):
    """Build a fail-closed Nitro verifier for approved release manifests."""

    approved = {
        str(commit).lower(): validate_release_manifest(release)
        for commit, release in releases.items()
    }

    def verify(identity: Mapping[str, Any]) -> Mapping[str, Any]:
        if not _is_gateway_release_boot(identity):
            # Validator boots in mixed receipt ancestry verify via the
            # gateway's dynamic PCR0 build cache, not a gateway release role.
            return _verify_validator_dynamic_boot(identity)
        commit = str(identity.get("commit_sha") or "").lower()
        release = approved.get(commit)
        if release is None:
            raise ReleaseLineageV2Error(
                "boot commit is absent from approved V2 release lineage"
            )
        physical_role = str(identity.get("physical_role") or "")
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
