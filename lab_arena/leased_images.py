"""Materialize a lease-bound private ECR image from transient HTTPS URLs."""

from __future__ import annotations

import base64
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Protocol, Tuple

from lab_arena import images, scorer_image_access


IMAGE_ACCESS_SCHEMA_VERSION = scorer_image_access.SCHEMA_VERSION
MAX_IMAGE_ACCESS_DOCUMENT_BYTES = 6 * 1024 * 1024
MAX_SIGNED_BLOB_URL_LENGTH = scorer_image_access.MAX_SIGNED_URL_BYTES
_ECR_HOST_RE = re.compile(
    r"^[0-9]{12}\.dkr\.ecr\.[a-z0-9-]+\.amazonaws\.com$"
)


class LeasedImageError(RuntimeError):
    """A lease image capability was unavailable or did not match its lease."""


class ImageAccessApi(Protocol):
    def image_access(self, run_id: str, lease_token: str) -> Dict[str, Any]: ...


@dataclass(frozen=True)
class LeasedBlob:
    digest: str
    size: int
    media_type: str
    url: str


@dataclass(frozen=True)
class LeasedImageAccess:
    image_reference: str
    image_digest: str
    manifest: bytes
    manifest_media_type: str
    blobs: Tuple[LeasedBlob, ...]


def is_ecr_reference(image_reference: str) -> bool:
    """Return true only for a valid private Amazon ECR image reference."""

    try:
        reference = images.parse_reference(image_reference)
    except images.ImageError:
        return False
    return bool(_ECR_HOST_RE.fullmatch(reference.registry))


def _manifest_document(raw: bytes) -> Dict[str, Any]:
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise LeasedImageError("leased image manifest is invalid") from exc
    if not isinstance(value, dict):
        raise LeasedImageError("leased image manifest is invalid")
    return value


def validate_image_access(
    document: Any,
    *,
    image_reference: str,
    image_digest: str,
    rules: images.ImageRules,
) -> LeasedImageAccess:
    """Bind one transient access document to the exact leased OCI manifest."""

    fields = {
        "schema_version",
        "image_reference",
        "image_digest",
        "manifest_b64",
        "manifest_media_type",
        "blobs",
    }
    if not isinstance(document, Mapping) or set(document) != fields:
        raise LeasedImageError("leased image access fields are invalid")
    if document.get("schema_version") != IMAGE_ACCESS_SCHEMA_VERSION:
        raise LeasedImageError("leased image access schema is invalid")
    if (
        document.get("image_reference") != image_reference
        or document.get("image_digest") != image_digest
    ):
        raise LeasedImageError("leased image access differs from the lease")
    try:
        reference = images.parse_reference(image_reference)
    except images.ImageError as exc:
        raise LeasedImageError("lease image reference is invalid") from exc
    if reference.digest != image_digest or not is_ecr_reference(image_reference):
        raise LeasedImageError("lease image reference is not a pinned private ECR image")

    encoded_manifest = document.get("manifest_b64")
    if not isinstance(encoded_manifest, str) or len(encoded_manifest) > (
        (images.MAX_MANIFEST_BYTES + 2) // 3 * 4 + 4
    ):
        raise LeasedImageError("leased image manifest is invalid")
    try:
        manifest = base64.b64decode(encoded_manifest, validate=True)
    except (TypeError, ValueError) as exc:
        raise LeasedImageError("leased image manifest is invalid") from exc
    media_type = document.get("manifest_media_type")
    if (
        not manifest
        or len(manifest) > images.MAX_MANIFEST_BYTES
        or images.sha256_digest(manifest) != image_digest
        or media_type not in images.MANIFEST_MEDIA_TYPES
    ):
        raise LeasedImageError("leased image manifest differs from the lease")
    manifest_document = _manifest_document(manifest)
    if manifest_document.get("mediaType") != media_type:
        raise LeasedImageError("leased image manifest media type differs")
    try:
        descriptors = images.manifest_layers(manifest_document, rules)
    except images.ImageError as exc:
        raise LeasedImageError("leased image manifest violates image rules") from exc

    raw_blobs = document.get("blobs")
    if not isinstance(raw_blobs, list) or len(raw_blobs) != len(descriptors):
        raise LeasedImageError("leased image blob descriptors differ from the manifest")
    blobs = []
    for raw, expected in zip(raw_blobs, descriptors):
        if not isinstance(raw, Mapping) or set(raw) != {
            "digest", "size", "media_type", "url"
        }:
            raise LeasedImageError("leased image blob descriptor fields are invalid")
        url = raw.get("url")
        if (
            not isinstance(raw.get("digest"), str)
            or raw.get("digest") != expected.digest
            or isinstance(raw.get("size"), bool)
            or not isinstance(raw.get("size"), int)
            or raw.get("size") != expected.size
            or not isinstance(raw.get("media_type"), str)
            or raw.get("media_type") != expected.media_type
            or not isinstance(url, str)
            or not url
            or len(url) > MAX_SIGNED_BLOB_URL_LENGTH
        ):
            raise LeasedImageError("leased image blob descriptor differs from the manifest")
        try:
            scorer_image_access.validate_ecr_download_url(
                url,
                now=datetime.now(timezone.utc),
            )
        except scorer_image_access.ScorerImageAccessError:
            raise LeasedImageError("leased image blob URL is invalid") from None
        blobs.append(
            LeasedBlob(
                digest=expected.digest,
                size=expected.size,
                media_type=expected.media_type,
                url=url,
            )
        )
    return LeasedImageAccess(
        image_reference=image_reference,
        image_digest=image_digest,
        manifest=manifest,
        manifest_media_type=str(media_type),
        blobs=tuple(blobs),
    )


class _LeasedRegistryView:
    """Registry-shaped view over one validated, transient access document."""

    def __init__(
        self,
        access: LeasedImageAccess,
        client: images.RegistryClient,
    ) -> None:
        self._access = access
        self._client = client
        self._reference = images.parse_reference(access.image_reference)
        self._by_digest = {blob.digest: blob for blob in access.blobs}

    def get_manifest(
        self,
        reference: images.ImageReference,
        *,
        deadline: Optional[float] = None,
    ) -> Tuple[bytes, str]:
        del deadline
        if reference != self._reference:
            raise images.ImageError(
                images.RULE_DIGEST_MISMATCH,
                "leased manifest reference differs",
            )
        return self._access.manifest, self._access.manifest_media_type

    def stream_blob(
        self,
        registry: str,
        repository: str,
        digest: str,
        *,
        expected_size: int,
        sink: Callable[[bytes], Any],
        deadline: Optional[float] = None,
    ) -> int:
        blob = self._by_digest.get(digest)
        if (
            registry != self._reference.registry
            or repository != self._reference.repository
            or blob is None
            or expected_size != blob.size
        ):
            raise images.ImageError(
                images.RULE_DIGEST_MISMATCH,
                "leased blob request differs from the manifest",
            )
        return self._client.stream_url(
            blob.url,
            expected_size=blob.size,
            expected_digest=blob.digest,
            sink=sink,
            deadline=deadline,
        )


def materialize_leased_image(
    client: images.RegistryClient,
    document: Any,
    image_reference: str,
    image_digest: str,
    target_dir: Path,
    *,
    rules: images.ImageRules,
) -> Dict[str, Any]:
    """Validate, download, and harden one lease-scoped image root filesystem."""

    access = validate_image_access(
        document,
        image_reference=image_reference,
        image_digest=image_digest,
        rules=rules,
    )
    config = access.blobs[0]
    config_chunks = []
    client.stream_url(
        config.url,
        expected_size=config.size,
        expected_digest=config.digest,
        sink=config_chunks.append,
    )
    try:
        config_document = json.loads(b"".join(config_chunks).decode("utf-8"))
        if not isinstance(config_document, dict):
            raise ValueError("not an object")
        images.validate_config_platform(config_document, rules)
    except (UnicodeDecodeError, ValueError, images.ImageError) as exc:
        raise LeasedImageError("leased image config is invalid") from exc
    view = _LeasedRegistryView(access, client)
    try:
        return images.materialize_rootfs(
            view,
            images.parse_reference(image_reference),
            Path(target_dir),
            rules=rules,
        )
    except images.ImageError as exc:
        raise LeasedImageError(
            "leased image could not be materialized: %s" % exc.rule_id
        ) from None


def leased_image_exporter(
    api: ImageAccessApi,
    run_id: str,
    lease_token: str,
    *,
    rules: Optional[images.ImageRules] = None,
    client_factory: Callable[[], images.RegistryClient] = images.RegistryClient,
) -> Callable[[str, str, Path], None]:
    """Return a cache-miss exporter bound to one active run lease."""

    image_rules = rules or images.ImageRules()

    def export(image_reference: str, image_digest: str, target_dir: Path) -> None:
        client = None
        document = None
        failure = None
        try:
            document = api.image_access(run_id, lease_token)
            client = client_factory()
            materialize_leased_image(
                client,
                document,
                image_reference,
                image_digest,
                target_dir,
                rules=image_rules,
            )
        except LeasedImageError as exc:
            failure = str(exc)
        except Exception:
            # API and HTTP errors can carry the signed capability URL in a
            # chained exception. Re-raise outside the handler so no original
            # exception or traceback is retained by the runner.
            failure = "leased image access failed"
        finally:
            document = None
            if client is not None:
                try:
                    client.close()
                except Exception:
                    failure = failure or "leased image access failed"
        if failure is not None:
            raise LeasedImageError(failure)

    return export


__all__ = [
    "IMAGE_ACCESS_SCHEMA_VERSION",
    "MAX_IMAGE_ACCESS_DOCUMENT_BYTES",
    "LeasedImageAccess",
    "LeasedImageError",
    "is_ecr_reference",
    "leased_image_exporter",
    "materialize_leased_image",
    "validate_image_access",
]
