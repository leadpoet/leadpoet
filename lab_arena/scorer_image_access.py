"""Lease-scoped access documents for an organizer-owned ECR scorer image.

The document contains the exact pinned manifest and temporary ECR download
URLs for only the blobs referenced by that manifest. It never contains an AWS
credential and must not be persisted or logged. ECR fixes each URL lifetime;
the URL can remain usable after its issuing lease ends, so issuance is allowed
only while that lease is active and the caller is still a subnet validator.
"""

from __future__ import annotations

import base64
import json
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, Mapping, Optional
from urllib.parse import parse_qs, urlsplit

from lab_arena import images


SCHEMA_VERSION = "leadpoet.arena.scorer_image_access.v1"
MAX_SIGNED_URL_BYTES = 8_192
MAX_SIGNED_URL_TTL_SECONDS = 3_600
MAX_CLOCK_SKEW_SECONDS = 300

_ECR_HOST = re.compile(
    r"^(?P<account>[0-9]{12})\.dkr\.ecr\.(?P<region>[a-z0-9-]+)\.amazonaws\.com$"
)


class ScorerImageAccessError(RuntimeError):
    """A safe, non-secret failure while preparing an access document."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


def _utc(moment: datetime) -> datetime:
    if moment.tzinfo is None:
        return moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(timezone.utc)


def _single_query_value(query: Mapping[str, list[str]], name: str) -> str:
    values = query.get(name)
    if not isinstance(values, list) or len(values) != 1 or not values[0]:
        raise ScorerImageAccessError("signed_url_invalid")
    return values[0]


def validate_ecr_download_url(url: Any, *, now: datetime) -> str:
    """Accept one current, bounded SigV4 URL for a public AWS S3 endpoint."""

    if (
        not isinstance(url, str)
        or not url
        or any(ord(character) <= 32 or ord(character) == 127 for character in url)
    ):
        raise ScorerImageAccessError("signed_url_invalid")
    try:
        if len(url.encode("utf-8")) > MAX_SIGNED_URL_BYTES:
            raise ScorerImageAccessError("signed_url_invalid")
        parsed = urlsplit(url)
        port = parsed.port
        query = parse_qs(parsed.query, keep_blank_values=True, strict_parsing=True)
    except ScorerImageAccessError:
        raise
    except (UnicodeEncodeError, ValueError):
        raise ScorerImageAccessError("signed_url_invalid") from None
    hostname = str(parsed.hostname or "").lower().rstrip(".")
    if (
        parsed.scheme != "https"
        or not hostname.endswith(".amazonaws.com")
        or ".s3." not in hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.fragment
        or port not in (None, 443)
        or not parsed.path.startswith("/")
    ):
        raise ScorerImageAccessError("signed_url_invalid")
    if len(query) > 16 or any(
        len(key) > 64
        or not isinstance(values, list)
        or len(values) > 1
        or any(len(value) > 4_096 for value in values)
        for key, values in query.items()
    ):
        raise ScorerImageAccessError("signed_url_invalid")
    try:
        if _single_query_value(query, "X-Amz-Algorithm") != "AWS4-HMAC-SHA256":
            raise ScorerImageAccessError("signed_url_invalid")
        issued = datetime.strptime(
            _single_query_value(query, "X-Amz-Date"), "%Y%m%dT%H%M%SZ"
        ).replace(tzinfo=timezone.utc)
        lifetime = int(_single_query_value(query, "X-Amz-Expires"))
        _single_query_value(query, "X-Amz-Credential")
        _single_query_value(query, "X-Amz-Signature")
        _single_query_value(query, "X-Amz-SignedHeaders")
    except ScorerImageAccessError:
        raise
    except (TypeError, ValueError):
        raise ScorerImageAccessError("signed_url_invalid") from None
    if not 1 <= lifetime <= MAX_SIGNED_URL_TTL_SECONDS:
        raise ScorerImageAccessError("signed_url_lifetime_invalid")
    current = _utc(now)
    if issued > current + timedelta(seconds=MAX_CLOCK_SKEW_SECONDS):
        raise ScorerImageAccessError("signed_url_invalid")
    if issued + timedelta(seconds=lifetime) <= current:
        raise ScorerImageAccessError("signed_url_expired")
    return url


@dataclass(frozen=True)
class EcrScorerImageAccess:
    """Build access documents for one configured private ECR repository."""

    client: Any
    registry: str
    repository: str
    registry_id: str
    rules: images.ImageRules
    clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc)

    def __call__(self, image_reference: str, image_digest: str) -> Dict[str, Any]:
        try:
            reference = images.parse_reference(image_reference)
        except images.ImageError:
            raise ScorerImageAccessError("image_reference_invalid") from None
        if (
            reference.registry != self.registry
            or reference.repository != self.repository
            or reference.digest != image_digest
        ):
            raise ScorerImageAccessError("image_repository_mismatch")

        try:
            response = self.client.batch_get_image(
                registryId=self.registry_id,
                repositoryName=self.repository,
                imageIds=[{"imageDigest": image_digest}],
                acceptedMediaTypes=list(images.MANIFEST_MEDIA_TYPES),
            )
        except Exception:
            raise ScorerImageAccessError("ecr_manifest_unavailable") from None
        if not isinstance(response, Mapping):
            raise ScorerImageAccessError("ecr_manifest_invalid")
        returned = response.get("images")
        failures = response.get("failures") or []
        if not isinstance(returned, list) or len(returned) != 1 or failures:
            raise ScorerImageAccessError("ecr_manifest_unavailable")
        image = returned[0]
        if not isinstance(image, Mapping):
            raise ScorerImageAccessError("ecr_manifest_invalid")
        image_id = image.get("imageId")
        manifest_text = image.get("imageManifest")
        media_type = image.get("imageManifestMediaType")
        if (
            image.get("registryId") != self.registry_id
            or image.get("repositoryName") != self.repository
            or not isinstance(image_id, Mapping)
            or image_id.get("imageDigest") != image_digest
            or not isinstance(manifest_text, str)
            or not isinstance(media_type, str)
            or media_type not in images.MANIFEST_MEDIA_TYPES
        ):
            raise ScorerImageAccessError("ecr_manifest_invalid")
        try:
            manifest = manifest_text.encode("utf-8")
        except UnicodeEncodeError:
            raise ScorerImageAccessError("ecr_manifest_invalid") from None
        if (
            len(manifest) > images.MAX_MANIFEST_BYTES
            or images.sha256_digest(manifest) != image_digest
        ):
            raise ScorerImageAccessError("ecr_manifest_digest_mismatch")
        try:
            document = json.loads(manifest_text)
            if not isinstance(document, Mapping):
                raise ValueError("manifest is not an object")
            if document.get("mediaType") != media_type:
                raise ValueError("manifest media type differs")
            descriptors = images.manifest_layers(document, self.rules)
        except (images.ImageError, RecursionError, TypeError, ValueError):
            raise ScorerImageAccessError("ecr_manifest_invalid") from None

        blobs = []
        for descriptor in descriptors:
            try:
                layer = self.client.get_download_url_for_layer(
                    registryId=self.registry_id,
                    repositoryName=self.repository,
                    layerDigest=descriptor.digest,
                )
            except Exception:
                raise ScorerImageAccessError("ecr_blob_unavailable") from None
            if (
                not isinstance(layer, Mapping)
                or layer.get("layerDigest") != descriptor.digest
            ):
                raise ScorerImageAccessError("ecr_blob_invalid")
            url = validate_ecr_download_url(layer.get("downloadUrl"), now=self.clock())
            blobs.append(
                {
                    "digest": descriptor.digest,
                    "size": descriptor.size,
                    "media_type": descriptor.media_type,
                    "url": url,
                }
            )
        return {
            "schema_version": SCHEMA_VERSION,
            "image_reference": image_reference,
            "image_digest": image_digest,
            "manifest_b64": base64.b64encode(manifest).decode("ascii"),
            "manifest_media_type": media_type,
            "blobs": blobs,
        }


def ecr_provider_from_repository(
    repository: str,
    *,
    rules: images.ImageRules,
    client: Any = None,
    clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
) -> Optional[EcrScorerImageAccess]:
    """Return an ECR provider, or ``None`` for a generic OCI repository."""

    try:
        registry, name = images.parse_repository(repository)
    except images.ImageError:
        raise ScorerImageAccessError("trusted_repository_invalid") from None
    match = _ECR_HOST.fullmatch(registry)
    if match is None:
        return None
    if client is None:
        try:
            import boto3

            client = boto3.client("ecr", region_name=match.group("region"))
        except Exception:
            raise ScorerImageAccessError("ecr_client_unavailable") from None
    return EcrScorerImageAccess(
        client=client,
        registry=registry,
        repository=name,
        registry_id=match.group("account"),
        rules=rules,
        clock=clock,
    )


__all__ = [
    "EcrScorerImageAccess",
    "MAX_SIGNED_URL_TTL_SECONDS",
    "SCHEMA_VERSION",
    "ScorerImageAccessError",
    "ecr_provider_from_repository",
    "validate_ecr_download_url",
]
