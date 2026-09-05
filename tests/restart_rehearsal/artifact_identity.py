"""Deterministic identities supplied by the local external-build boundary.

The restart rehearsal executes the production image normalizer and release
verifiers unchanged. Docker and Nitro are privileged external boundaries, so
the local replica supplies deterministic, commit- and role-bound artifacts
instead of consuming host Docker/Nitro resources.
"""

from __future__ import annotations

import hashlib
import io
import json
import tarfile


GATEWAY_ROLES = (
    "gateway_coordinator",
    "gateway_scoring",
)
VALIDATOR_ROLE = "validator_weights"
ALL_ROLES = frozenset((*GATEWAY_ROLES, VALIDATOR_ROLE))
_EPOCH = "1970-01-01T00:00:00Z"


def _require_commit(commit: str) -> str:
    value = str(commit).strip().lower()
    if len(value) != 40 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError("rehearsal artifact commit is invalid")
    return value


def _require_role(role: str) -> str:
    value = str(role).strip()
    if value not in ALL_ROLES:
        raise ValueError("rehearsal artifact role is invalid")
    return value


def pcr0(commit: str) -> str:
    """Return the commit-bound PCR0 emitted by the Nitro boundary."""

    normalized_commit = _require_commit(commit)
    return hashlib.sha384(
        b"leadpoet-local-pcr0:" + normalized_commit.encode("ascii")
    ).hexdigest()


def _layer_archive(
    commit: str,
    role: str,
    *,
    member_mtime: int = 0,
) -> bytes:
    normalized_commit = _require_commit(commit)
    normalized_role = _require_role(role)
    content = (
        "leadpoet external build boundary v2\n"
        f"commit={normalized_commit}\n"
        f"role={normalized_role}\n"
    ).encode("ascii")
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w:") as archive:
        member = tarfile.TarInfo("leadpoet-build-identity.txt")
        member.size = len(content)
        member.mtime = member_mtime
        member.uname = ""
        member.gname = ""
        archive.addfile(member, io.BytesIO(content))
    return output.getvalue()


def normalized_config(commit: str, role: str) -> tuple[bytes, bytes]:
    """Return the canonical layer and Docker config used by normalization."""

    normalized_commit = _require_commit(commit)
    normalized_role = _require_role(role)
    layer = _layer_archive(normalized_commit, normalized_role)
    layer_hash = hashlib.sha256(layer).hexdigest()
    config = {
        "config": {
            "Labels": {
                "org.leadpoet.rehearsal.commit": normalized_commit,
                "org.leadpoet.rehearsal.role": normalized_role,
            }
        },
        "created": _EPOCH,
        "history": [{"created": _EPOCH}],
        "rootfs": {
            "diff_ids": [f"sha256:{layer_hash}"],
            "type": "layers",
        },
    }
    config_bytes = json.dumps(
        config,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return layer, config_bytes


def normalized_image_id(commit: str, role: str) -> str:
    """Return the image ID produced by the real production normalizer."""

    _, config = normalized_config(commit, role)
    return "sha256:" + hashlib.sha256(config).hexdigest()


def docker_save_archive(commit: str, role: str, source_tag: str) -> bytes:
    """Return a deterministic one-image archive accepted by ``docker load``."""

    if not isinstance(source_tag, str) or not source_tag:
        raise ValueError("rehearsal Docker source tag is invalid")
    normalized_commit = _require_commit(commit)
    normalized_role = _require_role(role)
    # This is the *raw* external build result. Its timestamps intentionally
    # differ from the normalized identity so the unchanged production
    # normalizer must really rewrite both the layer and config paths. Supplying
    # an already-normalized archive would make that script unlink its own
    # replacement when the old and new content hashes are identical.
    layer = _layer_archive(
        normalized_commit,
        normalized_role,
        member_mtime=1_750_000_000,
    )
    layer_hash = hashlib.sha256(layer).hexdigest()
    config = json.dumps(
        {
            "config": {
                "Labels": {
                    "org.leadpoet.rehearsal.commit": normalized_commit,
                    "org.leadpoet.rehearsal.role": normalized_role,
                }
            },
            "created": "2025-06-15T00:00:00Z",
            "history": [{"created": "2025-06-15T00:00:00Z"}],
            "rootfs": {
                "diff_ids": [f"sha256:{layer_hash}"],
                "type": "layers",
            },
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    config_hash = hashlib.sha256(config).hexdigest()
    manifest = [
        {
            "Config": f"blobs/sha256/{config_hash}",
            "Layers": [f"blobs/sha256/{layer_hash}"],
            "RepoTags": [source_tag],
        }
    ]
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w:") as archive:
        for name, data in (
            (
                "manifest.json",
                json.dumps(manifest, separators=(",", ":")).encode("utf-8"),
            ),
            (f"blobs/sha256/{config_hash}", config),
            (f"blobs/sha256/{layer_hash}", layer),
        ):
            member = tarfile.TarInfo(name)
            member.size = len(data)
            member.mtime = 0
            member.uname = ""
            member.gname = ""
            archive.addfile(member, io.BytesIO(data))
    return output.getvalue()


def eif_bytes(commit: str, role: str) -> bytes:
    """Return the deterministic EIF payload supplied by the Nitro boundary."""

    normalized_commit = _require_commit(commit)
    normalized_role = _require_role(role)
    return (
        b"leadpoet-rehearsal-eif-v2\0"
        + normalized_commit.encode("ascii")
        + b"\0"
        + normalized_role.encode("ascii")
        + b"\n"
    )


def eif_hash(commit: str, role: str) -> str:
    return "sha256:" + hashlib.sha256(eif_bytes(commit, role)).hexdigest()
