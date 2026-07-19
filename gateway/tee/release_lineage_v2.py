"""Verify receipt ancestry across independently attested V2 releases."""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Callable, Dict, Mapping, Sequence

from gateway.tee.release_channel_v2 import DEFAULT_BUCKET, fetch_release_channel_v2
from gateway.tee.release_manifest_v2 import (
    role_expectation,
    validate_release_manifest,
)
from leadpoet_canonical.attested_v2 import verify_boot_identity_nitro


class ReleaseLineageV2Error(RuntimeError):
    """A receipt ancestor is not bound to an approved V2 release."""


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
    channel = fetch_release_channel_v2(
        bucket=DEFAULT_BUCKET,
        commit_sha=commit,
    )
    return dict(channel["gateway_release_manifest"])


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
        )

    return verify
