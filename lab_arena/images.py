"""Resolve and materialize the organizer-owned Arena runtime image.

The Arena resolves one configured OCI tag or digest, validates its platform
and size, and pins the single-platform manifest digest. A runner materializes
that digest with a hardened extractor. No Docker daemon unpacks image layers.
The read-only registry client supports anonymous and credentialed pulls so the
common runtime image can be held in a public or private registry.
"""

from __future__ import annotations

import base64
import copy
import hashlib
import ipaddress
import json
import os
import posixpath
import re
import shutil
import socket
import tarfile
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple
from urllib.parse import urljoin, urlsplit

import httpx

# ---------------------------------------------------------------------------
# Public rule ids and limits
# ---------------------------------------------------------------------------

RULE_REFERENCE_INVALID = "image.reference_invalid"
RULE_UNAVAILABLE = "image.unavailable"
RULE_MANIFEST_INVALID = "image.manifest_invalid"
RULE_PLATFORM_UNSUPPORTED = "image.platform_unsupported"
RULE_TOO_LARGE = "image.too_large"
RULE_TOO_MANY_LAYERS = "image.too_many_layers"
RULE_LAYER_UNSUPPORTED = "image.layer_unsupported"
RULE_LAYER_INVALID = "image.layer_invalid"
RULE_CONFIG_INVALID = "image.config_invalid"
RULE_DIGEST_MISMATCH = "image.digest_mismatch"

IMAGE_RULE_IDS = (
    RULE_REFERENCE_INVALID,
    RULE_UNAVAILABLE,
    RULE_MANIFEST_INVALID,
    RULE_PLATFORM_UNSUPPORTED,
    RULE_TOO_LARGE,
    RULE_TOO_MANY_LAYERS,
    RULE_LAYER_UNSUPPORTED,
    RULE_LAYER_INVALID,
    RULE_CONFIG_INVALID,
    RULE_DIGEST_MISMATCH,
)

IMAGE_RULES_SCHEMA_VERSION = "leadpoet.lab_arena.image_rules.v1"
DEFAULT_MAX_IMAGE_BYTES = 2 * 1024 * 1024 * 1024  # compressed layers plus config
DEFAULT_MAX_LAYERS = 64
DEFAULT_MAX_ROOTFS_BYTES = 8 * 1024 * 1024 * 1024  # uncompressed regular-file bytes
MAX_MANIFEST_BYTES = 4 * 1024 * 1024
MAX_CONFIG_BYTES = 4 * 1024 * 1024
MAX_TOKEN_RESPONSE_BYTES = 64 * 1024
# This is one total across all layers. It bounds TarInfo memory and the number
# of host filesystem objects an image can ask the runner to create.
MAX_IMAGE_TAR_MEMBERS = 200_000
MAX_REFERENCE_LENGTH = 512
MAX_BLOB_REDIRECTS = 3

MANIFEST_OCI = "application/vnd.oci.image.manifest.v1+json"
MANIFEST_DOCKER = "application/vnd.docker.distribution.manifest.v2+json"
INDEX_OCI = "application/vnd.oci.image.index.v1+json"
INDEX_DOCKER = "application/vnd.docker.distribution.manifest.list.v2+json"
MANIFEST_MEDIA_TYPES = (MANIFEST_OCI, MANIFEST_DOCKER)
INDEX_MEDIA_TYPES = (INDEX_OCI, INDEX_DOCKER)
CONFIG_MEDIA_TYPES = ("application/vnd.oci.image.config.v1+json", "application/vnd.docker.container.image.v1+json")
LAYER_GZIP_MEDIA_TYPES = ("application/vnd.oci.image.layer.v1.tar+gzip", "application/vnd.docker.image.rootfs.diff.tar.gzip")
LAYER_TAR_MEDIA_TYPES = ("application/vnd.oci.image.layer.v1.tar",)
LAYER_MEDIA_TYPES = LAYER_GZIP_MEDIA_TYPES + LAYER_TAR_MEDIA_TYPES
ACCEPT_MANIFESTS = ", ".join(MANIFEST_MEDIA_TYPES + INDEX_MEDIA_TYPES)
ATTESTATION_REFERENCE_TYPE = "attestation-manifest"

DOCKER_HUB_HOSTS = ("docker.io", "index.docker.io")
DOCKER_HUB_REGISTRY = "registry-1.docker.io"

_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_COMPONENT = r"[a-z0-9]+(?:(?:[._]|__|-+)[a-z0-9]+)*"
_PATH_RE = re.compile(r"^" + _COMPONENT + r"(?:/" + _COMPONENT + r")*$")
_HOST_RE = re.compile(r"^(?:localhost|127\.0\.0\.1|[a-z0-9](?:[a-z0-9-]*[a-z0-9])?(?:\.[a-z0-9](?:[a-z0-9-]*[a-z0-9])?)+)(?::[0-9]{1,5})?$")
_TAG_RE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}$")
_CHALLENGE_RE = re.compile(r'(\w+)="([^"]*)"')


class ImageError(RuntimeError):
    """An image was refused or could not be handled; ``rule_id`` is public."""

    def __init__(self, rule_id: str, detail: str = "") -> None:
        if rule_id not in IMAGE_RULE_IDS:
            raise ValueError("unknown image rule id")
        self.rule_id = rule_id
        super().__init__(detail or rule_id)


# ---------------------------------------------------------------------------
# References
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ImageReference:
    """An OCI ``registry/repository:tag`` or digest reference."""

    registry: str
    repository: str
    digest: Optional[str] = None
    tag: Optional[str] = None

    @property
    def name(self) -> str:
        return self.registry + "/" + self.repository

    @property
    def api_registry(self) -> str:
        return DOCKER_HUB_REGISTRY if self.registry in DOCKER_HUB_HOSTS else self.registry

    def __str__(self) -> str:
        tag = ":" + self.tag if self.tag else ""
        digest = "@" + self.digest if self.digest else ""
        return "%s/%s%s%s" % (self.registry, self.repository, tag, digest)

    @property
    def selector(self) -> str:
        selector = self.digest or self.tag
        if not selector:
            raise ImageError(RULE_REFERENCE_INVALID, "reference must name a tag or digest")
        return selector

    def with_digest(self, digest: str) -> "ImageReference":
        if not _DIGEST_RE.match(str(digest)):
            raise ImageError(RULE_REFERENCE_INVALID, "digest is invalid")
        return ImageReference(self.registry, self.repository, str(digest))


def parse_reference(text: Any) -> ImageReference:
    """Parse a registry reference. A host and tag or digest are required."""

    if not isinstance(text, str) or not text or len(text) > MAX_REFERENCE_LENGTH or any(ch.isspace() or ord(ch) < 32 for ch in text):
        raise ImageError(RULE_REFERENCE_INVALID, "reference must be a bounded single-line string")
    if "@" in text:
        name_and_tag, digest = text.rsplit("@", 1)
        if not _DIGEST_RE.match(digest):
            raise ImageError(RULE_REFERENCE_INVALID, "digest must be sha256:<64 hex>")
    else:
        name_and_tag, digest = text, None
    host, separator, path = name_and_tag.partition("/")
    if not separator or not path:
        raise ImageError(RULE_REFERENCE_INVALID, "reference must name its registry host")
    host = host.lower()
    if not ("." in host or ":" in host or host == "localhost"):
        raise ImageError(RULE_REFERENCE_INVALID, "reference must name its registry host")
    if not _HOST_RE.match(host):
        raise ImageError(RULE_REFERENCE_INVALID, "registry host is invalid")
    tag: Optional[str] = None
    last = path.rsplit("/", 1)[-1]
    if ":" in last:
        path, tag = path.rsplit(":", 1)
        if not _TAG_RE.match(tag):
            raise ImageError(RULE_REFERENCE_INVALID, "tag is invalid")
    if digest is None and tag is None:
        raise ImageError(RULE_REFERENCE_INVALID, "reference must name a tag or digest")
    if not _PATH_RE.match(path):
        raise ImageError(RULE_REFERENCE_INVALID, "repository path is invalid")
    if host in DOCKER_HUB_HOSTS and "/" not in path:
        path = "library/" + path
    return ImageReference(host, path, digest, tag)


def parse_repository(text: Any) -> Tuple[str, str]:
    """Parse ``registry/repository`` (no tag, no digest) into its two parts."""

    if not isinstance(text, str) or not text or "@" in text or ":" in text.rsplit("/", 1)[-1]:
        raise ImageError(RULE_REFERENCE_INVALID, "repository must be registry/path without tag or digest")
    reference = parse_reference(text + "@sha256:" + "0" * 64)
    return reference.registry, reference.repository


def sha256_digest(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


# ---------------------------------------------------------------------------
# Rules and descriptors
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ImageRules:
    max_image_bytes: int = DEFAULT_MAX_IMAGE_BYTES
    max_layers: int = DEFAULT_MAX_LAYERS
    max_rootfs_bytes: int = DEFAULT_MAX_ROOTFS_BYTES
    platform_os: str = "linux"
    platform_architecture: str = "amd64"

    def __post_init__(self) -> None:
        for name in ("max_image_bytes", "max_layers", "max_rootfs_bytes"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ImageError(RULE_CONFIG_INVALID, "%s must be a positive integer" % name)

    def to_document(self) -> Dict[str, Any]:
        return {
            "schema_version": IMAGE_RULES_SCHEMA_VERSION,
            "max_image_bytes": int(self.max_image_bytes),
            "max_layers": int(self.max_layers),
            "max_rootfs_bytes": int(self.max_rootfs_bytes),
            "platform": {"os": self.platform_os, "architecture": self.platform_architecture},
            "layer_media_types": list(LAYER_MEDIA_TYPES),
        }

    @classmethod
    def from_document(cls, document: Mapping[str, Any]) -> "ImageRules":
        platform = document.get("platform") or {}
        return cls(
            max_image_bytes=int(document["max_image_bytes"]),
            max_layers=int(document["max_layers"]),
            max_rootfs_bytes=int(document.get("max_rootfs_bytes") or DEFAULT_MAX_ROOTFS_BYTES),
            platform_os=str(platform.get("os") or "linux"),
            platform_architecture=str(platform.get("architecture") or "amd64"),
        )


@dataclass(frozen=True)
class BlobDescriptor:
    digest: str
    size: int
    media_type: str


@dataclass(frozen=True)
class ImageDescriptor:
    """A resolved single-platform runtime image."""

    reference: ImageReference

    @property
    def image_digest(self) -> str:
        digest = self.reference.digest
        if digest is None:
            raise ImageError(RULE_MANIFEST_INVALID, "resolved image has no digest")
        return digest


def _json_object(data: bytes, rule_id: str) -> Dict[str, Any]:
    try:
        document = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise ImageError(rule_id, "document is not UTF-8 JSON") from exc
    if not isinstance(document, dict):
        raise ImageError(rule_id, "document is not a JSON object")
    return document


def _blob_descriptor(entry: Any, rule_id: str) -> BlobDescriptor:
    if not isinstance(entry, Mapping):
        raise ImageError(rule_id, "descriptor is not an object")
    digest = entry.get("digest")
    size = entry.get("size")
    media_type = entry.get("mediaType")
    if not isinstance(digest, str) or not _DIGEST_RE.match(digest):
        raise ImageError(rule_id, "descriptor digest is invalid")
    if isinstance(size, bool) or not isinstance(size, int) or size < 0:
        raise ImageError(rule_id, "descriptor size is invalid")
    if not isinstance(media_type, str) or not media_type:
        raise ImageError(rule_id, "descriptor media type is missing")
    return BlobDescriptor(digest, int(size), media_type)


def manifest_layers(document: Mapping[str, Any], rules: ImageRules) -> Tuple[BlobDescriptor, BlobDescriptor, ...]:
    """Validate a single-platform manifest; returns ``(config, *layers)``."""

    layers_raw = document.get("layers")
    if not isinstance(layers_raw, list) or not layers_raw:
        raise ImageError(RULE_MANIFEST_INVALID, "manifest has no layers")
    if len(layers_raw) > rules.max_layers:
        raise ImageError(RULE_TOO_MANY_LAYERS, "manifest has %d layers; at most %d" % (len(layers_raw), rules.max_layers))
    config = _blob_descriptor(document.get("config"), RULE_MANIFEST_INVALID)
    if config.media_type not in CONFIG_MEDIA_TYPES:
        raise ImageError(RULE_MANIFEST_INVALID, "config media type is unsupported")
    if config.size > MAX_CONFIG_BYTES:
        raise ImageError(RULE_CONFIG_INVALID, "config blob is too large")
    layers = []
    total = config.size
    for entry in layers_raw:
        layer = _blob_descriptor(entry, RULE_MANIFEST_INVALID)
        if layer.media_type not in LAYER_MEDIA_TYPES:
            raise ImageError(RULE_LAYER_UNSUPPORTED, "layer media type %s is unsupported" % layer.media_type[:80])
        total += layer.size
        layers.append(layer)
    if total > rules.max_image_bytes:
        raise ImageError(RULE_TOO_LARGE, "image is %d bytes; at most %d" % (total, rules.max_image_bytes))
    return (config, *layers)


def _select_platform_manifest(index: Mapping[str, Any], rules: ImageRules) -> BlobDescriptor:
    manifests = index.get("manifests")
    if not isinstance(manifests, list):
        raise ImageError(RULE_MANIFEST_INVALID, "index has no manifests")
    for entry in manifests:
        if not isinstance(entry, Mapping):
            raise ImageError(RULE_MANIFEST_INVALID, "index entry is not an object")
        annotations = entry.get("annotations") or {}
        if isinstance(annotations, Mapping) and annotations.get("vnd.docker.reference.type") == ATTESTATION_REFERENCE_TYPE:
            continue
        platform = entry.get("platform") or {}
        if not isinstance(platform, Mapping):
            continue
        if platform.get("os") == rules.platform_os and platform.get("architecture") == rules.platform_architecture and not platform.get("variant"):
            descriptor = _blob_descriptor(entry, RULE_MANIFEST_INVALID)
            if descriptor.media_type not in MANIFEST_MEDIA_TYPES:
                raise ImageError(RULE_MANIFEST_INVALID, "index child media type is unsupported")
            return descriptor
    raise ImageError(RULE_PLATFORM_UNSUPPORTED, "no %s/%s manifest in the index" % (rules.platform_os, rules.platform_architecture))


def validate_config_platform(config_document: Mapping[str, Any], rules: ImageRules) -> None:
    """Check only the image platform. The competition owns the process contract."""

    if config_document.get("os") != rules.platform_os or config_document.get("architecture") != rules.platform_architecture:
        raise ImageError(RULE_PLATFORM_UNSUPPORTED, "image platform is not %s/%s" % (rules.platform_os, rules.platform_architecture))


# ---------------------------------------------------------------------------
# Registry client
# ---------------------------------------------------------------------------

Credentials = Callable[[str], Optional[Tuple[str, str]]]  # registry host -> (username, password)
AddressResolver = Callable[[str, int], Sequence[str]]


def _resolve_addresses(host: str, port: int) -> Tuple[str, ...]:
    try:
        answers = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except OSError as exc:
        raise ImageError(RULE_UNAVAILABLE, "registry host cannot be resolved") from exc
    return tuple(sorted({str(answer[4][0]) for answer in answers}))


def _url_origin(parts: Any) -> Tuple[str, str, int]:
    try:
        port = parts.port
    except ValueError as exc:
        raise ImageError(RULE_UNAVAILABLE, "network URL port is invalid") from exc
    scheme = str(parts.scheme).lower()
    host = str(parts.hostname or "").lower().rstrip(".")
    return scheme, host, int(port or (443 if scheme == "https" else 80))


def _is_public_address(address: Any) -> bool:
    return bool(
        address.is_global
        and not address.is_private
        and not address.is_loopback
        and not address.is_link_local
        and not address.is_reserved
        and not address.is_multicast
        and not address.is_unspecified
    )


class RegistryClient:
    """A minimal read-only OCI client for manifests, blobs, and tokens.

    ``credentials(registry)`` can return a read credential for the configured
    runtime registry and ``None`` elsewhere. Blob downloads follow redirects
    without forwarding the registry authorization header. ``address_resolver``
    is an injection seam for hermetic tests; production uses the system
    resolver.
    """

    def __init__(
        self,
        *,
        http: Optional[httpx.Client] = None,
        credentials: Optional[Credentials] = None,
        timeout_seconds: float = 60.0,
        address_resolver: AddressResolver = _resolve_addresses,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if timeout_seconds <= 0:
            raise ImageError(RULE_CONFIG_INVALID, "registry timeout must be positive")
        self._http = http or httpx.Client(
            http1=True,
            http2=False,
            follow_redirects=False,
            timeout=httpx.Timeout(timeout_seconds),
            trust_env=False,
        )
        self._credentials = credentials or (lambda registry: None)
        self._address_resolver = address_resolver
        self._timeout_seconds = float(timeout_seconds)
        self._clock = clock
        self._tokens: Dict[Tuple[str, str], str] = {}
        self._last_token: Dict[str, str] = {}  # the newest bearer per registry, sent proactively
        self._lock = threading.Lock()

    def _remaining(self, deadline: Optional[float]) -> float:
        if deadline is None:
            return self._timeout_seconds
        remaining = float(deadline) - float(self._clock())
        if remaining <= 0:
            raise ImageError(RULE_UNAVAILABLE, "image transfer deadline exceeded")
        return min(self._timeout_seconds, remaining)

    def _build_request(self, method: str, url: str, *, deadline: Optional[float], **kwargs: Any) -> httpx.Request:
        timeout = httpx.Timeout(self._remaining(deadline)).as_dict()
        return self._http.build_request(method, url, extensions={"timeout": timeout}, **kwargs)

    # -- transport ------------------------------------------------------------

    def base_url(self, registry: str) -> str:
        api_host = DOCKER_HUB_REGISTRY if registry in DOCKER_HUB_HOSTS else registry
        return "https://%s" % api_host

    def _validate_network_url(self, url: str, purpose: str) -> Tuple[str, str, int]:
        parts = urlsplit(url)
        origin = _url_origin(parts)
        scheme, host, port = origin
        if scheme != "https" or not parts.netloc or not host:
            raise ImageError(RULE_UNAVAILABLE, "%s URL is invalid" % purpose)
        if parts.username is not None or parts.password is not None or parts.fragment:
            raise ImageError(RULE_UNAVAILABLE, "%s URL is invalid" % purpose)
        if host == "localhost" or host.endswith(".localhost"):
            raise ImageError(RULE_UNAVAILABLE, "%s host is not public" % purpose)
        try:
            literal = ipaddress.ip_address(host)
        except ValueError:
            literal = None
        if literal is not None and not _is_public_address(literal):
            raise ImageError(RULE_UNAVAILABLE, "%s host is not public" % purpose)
        try:
            addresses = tuple(self._address_resolver(host, port))
        except ImageError:
            raise
        except Exception as exc:
            raise ImageError(RULE_UNAVAILABLE, "%s host cannot be resolved" % purpose) from exc
        if not addresses:
            raise ImageError(RULE_UNAVAILABLE, "%s host cannot be resolved" % purpose)
        for address in addresses:
            try:
                resolved = ipaddress.ip_address(str(address).split("%", 1)[0])
            except ValueError as exc:
                raise ImageError(RULE_UNAVAILABLE, "%s host resolution is invalid" % purpose) from exc
            if not _is_public_address(resolved):
                raise ImageError(RULE_UNAVAILABLE, "%s host is not public" % purpose)
        return origin

    def _basic(self, registry: str) -> Optional[str]:
        credential = self._credentials(registry)
        if not credential:
            return None
        username, password = credential
        return "Basic " + base64.b64encode(("%s:%s" % (username, password)).encode("utf-8")).decode("ascii")

    @staticmethod
    def _validate_response_peer(response: httpx.Response, purpose: str) -> None:
        """Reject a real connection whose peer is not a public IP.

        ``MockTransport`` responses have no network stream, so hermetic tests
        remain injectable. A production httpcore response supplies the stream
        and its connected server address; this second check closes the DNS
        rebinding gap between URL validation and the actual request.
        """

        stream = response.extensions.get("network_stream")
        if stream is None:
            return
        getter = getattr(stream, "get_extra_info", None)
        try:
            peer = getter("server_addr") if callable(getter) else None
        except Exception as exc:
            response.close()
            raise ImageError(RULE_UNAVAILABLE, "%s peer address is unavailable" % purpose) from exc
        address = peer[0] if isinstance(peer, (tuple, list)) and peer else peer
        try:
            parsed = ipaddress.ip_address(str(address).split("%", 1)[0])
        except ValueError as exc:
            response.close()
            raise ImageError(RULE_UNAVAILABLE, "%s peer address is invalid" % purpose) from exc
        if not _is_public_address(parsed):
            response.close()
            raise ImageError(RULE_UNAVAILABLE, "%s peer is not public" % purpose)

    def _token(self, registry: str, challenge: Mapping[str, str], *, deadline: Optional[float] = None) -> str:
        realm = challenge.get("realm") or ""
        realm_origin = self._validate_network_url(realm, "token realm")
        registry_origin = self._validate_network_url(self.base_url(registry), "registry")
        scope = challenge.get("scope") or ""
        key = (registry, scope)
        with self._lock:
            cached = self._tokens.get(key)
        if cached:
            return cached
        params = {}
        if challenge.get("service"):
            params["service"] = challenge["service"]
        if scope:
            params["scope"] = scope
        headers = {}
        basic = self._basic(registry)
        if basic and realm_origin == registry_origin:
            headers["Authorization"] = basic
        try:
            request = self._build_request("GET", realm, deadline=deadline, params=params, headers=headers)
            response = self._http.send(request, stream=True, follow_redirects=False)
            try:
                self._validate_response_peer(response, "token endpoint")
                if response.status_code != 200:
                    raise ImageError(RULE_UNAVAILABLE, "token endpoint answered %d" % response.status_code)
                raw = _read_bounded(
                    response,
                    MAX_TOKEN_RESPONSE_BYTES,
                    RULE_UNAVAILABLE,
                    deadline_check=lambda: self._remaining(deadline),
                )
            finally:
                response.close()
        except httpx.HTTPError as exc:
            raise ImageError(RULE_UNAVAILABLE, "token endpoint unreachable: %s" % type(exc).__name__) from exc
        try:
            document = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, ValueError) as exc:
            raise ImageError(RULE_UNAVAILABLE, "token endpoint returned non-JSON") from exc
        token = document.get("token") or document.get("access_token") if isinstance(document, dict) else None
        if not isinstance(token, str) or not token:
            raise ImageError(RULE_UNAVAILABLE, "token endpoint returned no token")
        with self._lock:
            self._tokens[key] = token
            self._last_token[registry] = token
        return token

    def _send(
        self,
        registry: str,
        path: str,
        *,
        headers: Optional[Mapping[str, str]] = None,
        stream: bool = False,
        deadline: Optional[float] = None,
    ) -> httpx.Response:
        """Read once; on a 401 challenge, obtain a token and read once more."""

        registry_url = self.base_url(registry)
        self._validate_network_url(registry_url, "registry")
        url = registry_url + path
        self._validate_network_url(url, "registry")
        request_headers = dict(headers or {})
        with self._lock:
            bearer = self._last_token.get(registry)
        basic = self._basic(registry)
        if "Authorization" not in request_headers:
            # Reuse the newest bearer obtained for this registry.
            if bearer:
                request_headers["Authorization"] = "Bearer " + bearer
            elif basic:
                request_headers["Authorization"] = basic
        try:
            request = self._build_request("GET", url, deadline=deadline, headers=request_headers)
            response = self._http.send(request, stream=stream, follow_redirects=False)
        except httpx.HTTPError as exc:
            raise ImageError(RULE_UNAVAILABLE, "registry unreachable: %s" % type(exc).__name__) from exc
        try:
            self._remaining(deadline)
        except ImageError:
            response.close()
            raise
        self._validate_response_peer(response, "registry")
        if response.status_code != 401:
            return response
        challenge_header = response.headers.get("www-authenticate", "")
        response.close()
        scheme = challenge_header.split(" ", 1)[0].strip().lower()
        if scheme != "bearer":
            raise ImageError(RULE_UNAVAILABLE, "registry refused the request (401)")
        challenge = {key.lower(): value for key, value in _CHALLENGE_RE.findall(challenge_header)}
        token = self._token(registry, challenge, deadline=deadline)
        request_headers["Authorization"] = "Bearer " + token
        try:
            request = self._build_request("GET", url, deadline=deadline, headers=request_headers)
            response = self._http.send(request, stream=stream, follow_redirects=False)
        except httpx.HTTPError as exc:
            raise ImageError(RULE_UNAVAILABLE, "registry unreachable: %s" % type(exc).__name__) from exc
        try:
            self._remaining(deadline)
        except ImageError:
            response.close()
            raise
        self._validate_response_peer(response, "registry")
        if response.status_code == 401:
            response.close()
            with self._lock:
                self._tokens.pop((registry, challenge.get("scope") or ""), None)
                self._last_token.pop(registry, None)
            raise ImageError(RULE_UNAVAILABLE, "registry refused the credential (401)")
        return response

    # -- manifests --------------------------------------------------------------

    def get_manifest(self, reference: ImageReference, *, deadline: Optional[float] = None) -> Tuple[bytes, str]:
        """The manifest bytes named by the reference and their media type."""

        selector = reference.selector
        response = self._send(
            reference.registry,
            "/v2/%s/manifests/%s" % (reference.repository, selector),
            headers={"Accept": ACCEPT_MANIFESTS},
            stream=True,
            deadline=deadline,
        )
        try:
            if response.status_code == 404:
                raise ImageError(RULE_UNAVAILABLE, "manifest is not in %s" % reference.name)
            if response.status_code != 200:
                raise ImageError(RULE_UNAVAILABLE, "manifest request answered %d" % response.status_code)
            body = _read_bounded(
                response,
                MAX_MANIFEST_BYTES,
                RULE_MANIFEST_INVALID,
                deadline_check=lambda: self._remaining(deadline),
            )
        finally:
            response.close()
        actual_digest = sha256_digest(body)
        reported_digest = response.headers.get("docker-content-digest")
        if reported_digest and reported_digest != actual_digest:
            raise ImageError(RULE_DIGEST_MISMATCH, "registry reports different manifest bytes")
        if reference.digest is not None and actual_digest != reference.digest:
            raise ImageError(RULE_DIGEST_MISMATCH, "manifest bytes do not hash to the named digest")
        media_type = (response.headers.get("content-type") or "").split(";", 1)[0].strip()
        if not media_type or media_type == "application/json" or media_type == "application/octet-stream":
            declared = _json_object(body, RULE_MANIFEST_INVALID).get("mediaType")
            media_type = str(declared or media_type)
        return body, media_type

    # -- blobs --------------------------------------------------------------------

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
        """Stream one blob into ``sink``; the byte count and digest must match."""

        response = self._send(registry, "/v2/%s/blobs/%s" % (repository, digest), stream=True, deadline=deadline)
        try:
            hops = 0
            while response.status_code in (301, 302, 303, 307, 308):
                location = response.headers.get("location")
                response.close()
                hops += 1
                if not location or hops > MAX_BLOB_REDIRECTS:
                    raise ImageError(RULE_UNAVAILABLE, "blob redirect chain is invalid")
                location = urljoin(str(response.request.url), location)
                self._validate_network_url(location, "blob redirect")
                try:
                    response = self._http.send(
                        self._build_request("GET", location, deadline=deadline),
                        stream=True,
                        follow_redirects=False,
                    )
                except httpx.HTTPError as exc:
                    raise ImageError(RULE_UNAVAILABLE, "blob store unreachable: %s" % type(exc).__name__) from exc
                self._remaining(deadline)
                self._validate_response_peer(response, "blob store")
            if response.status_code == 404:
                raise ImageError(RULE_UNAVAILABLE, "blob %s is missing" % digest[:19])
            if response.status_code != 200:
                raise ImageError(RULE_UNAVAILABLE, "blob request answered %d" % response.status_code)
            hasher = hashlib.sha256()
            received = 0
            for chunk in response.iter_bytes():
                self._remaining(deadline)
                received += len(chunk)
                if received > expected_size:
                    raise ImageError(RULE_DIGEST_MISMATCH, "blob %s is larger than its descriptor" % digest[:19])
                hasher.update(chunk)
                sink(chunk)
        finally:
            response.close()
        if received != expected_size or "sha256:" + hasher.hexdigest() != digest:
            raise ImageError(RULE_DIGEST_MISMATCH, "blob %s does not match its descriptor" % digest[:19])
        return received

    def get_blob(
        self,
        registry: str,
        repository: str,
        digest: str,
        *,
        expected_size: int,
        max_bytes: int,
        deadline: Optional[float] = None,
    ) -> bytes:
        if expected_size > max_bytes:
            raise ImageError(RULE_TOO_LARGE, "blob exceeds %d bytes" % max_bytes)
        chunks: List[bytes] = []
        self.stream_blob(registry, repository, digest, expected_size=expected_size, sink=chunks.append, deadline=deadline)
        return b"".join(chunks)

    def close(self) -> None:
        self._http.close()


def _read_bounded(
    response: httpx.Response,
    limit: int,
    rule_id: str,
    *,
    deadline_check: Optional[Callable[[], Any]] = None,
) -> bytes:
    chunks: List[bytes] = []
    total = 0
    for chunk in response.iter_bytes():
        if deadline_check is not None:
            deadline_check()
        total += len(chunk)
        if total > limit:
            raise ImageError(rule_id, "document exceeds %d bytes" % limit)
        chunks.append(chunk)
    return b"".join(chunks)


# ---------------------------------------------------------------------------
# Common runtime image resolution
# ---------------------------------------------------------------------------


def resolve_image(
    client: RegistryClient,
    reference: ImageReference,
    rules: ImageRules,
    *,
    deadline: Optional[float] = None,
) -> ImageDescriptor:
    """Resolve a tag once and return a pinned single-platform descriptor."""

    body, media_type = client.get_manifest(reference, deadline=deadline)
    reference = reference.with_digest(sha256_digest(body))
    document = _json_object(body, RULE_MANIFEST_INVALID)
    if media_type in INDEX_MEDIA_TYPES or (media_type not in MANIFEST_MEDIA_TYPES and "manifests" in document):
        child = _select_platform_manifest(document, rules)
        reference = reference.with_digest(child.digest)
        body, media_type = client.get_manifest(reference, deadline=deadline)
        document = _json_object(body, RULE_MANIFEST_INVALID)
    if media_type not in MANIFEST_MEDIA_TYPES:
        raise ImageError(RULE_MANIFEST_INVALID, "manifest media type %s is unsupported" % media_type[:80])
    config, *layers = manifest_layers(document, rules)
    config_bytes = client.get_blob(
        reference.registry,
        reference.repository,
        config.digest,
        expected_size=config.size,
        max_bytes=MAX_CONFIG_BYTES,
        deadline=deadline,
    )
    config_document = _json_object(config_bytes, RULE_CONFIG_INVALID)
    validate_config_platform(config_document, rules)
    return ImageDescriptor(reference=reference)


# ---------------------------------------------------------------------------
# Root filesystem materialization (runner side)
# ---------------------------------------------------------------------------

WHITEOUT_PREFIX = ".wh."
OPAQUE_WHITEOUT = ".wh..wh..opq"


def _inside(root_real: str, candidate: str) -> bool:
    try:
        return os.path.commonpath([root_real, candidate]) == root_real
    except ValueError:
        return False


def _rootfs_filter(member: tarfile.TarInfo, dest_path: str) -> Optional[tarfile.TarInfo]:
    """A small Python-3.8-safe tar filter for an untrusted image layer.

    It runs immediately before each member is extracted. Thus, a symlink made
    by an earlier member is present when the next destination is checked.
    Absolute symlink targets are allowed because they resolve from the sandbox
    root at run time; a later host-side write through one is still refused.
    """

    if member.isdev() or member.isfifo():
        return None  # /dev is a tmpfs at run time; nothing in a rootfs needs device nodes
    raw_name = str(member.name).replace("\\", "/")
    name = posixpath.normpath(raw_name)
    if not raw_name or "\x00" in raw_name or raw_name.startswith("/") or name in ("", ".", "..") or name.startswith("../"):
        raise ImageError(RULE_LAYER_INVALID, "layer member path escapes the root filesystem")
    root_real = os.path.realpath(dest_path)
    destination = os.path.realpath(os.path.join(root_real, name))
    if not _inside(root_real, destination):
        raise ImageError(RULE_LAYER_INVALID, "layer member path escapes the root filesystem")
    if member.islnk():
        linkname = str(member.linkname).replace("\\", "/")
        if not linkname or "\x00" in linkname:
            raise ImageError(RULE_LAYER_INVALID, "hardlink target is invalid")
        target = os.path.realpath(os.path.join(root_real, linkname.lstrip("/")))
        if not _inside(root_real, target):
            raise ImageError(RULE_LAYER_INVALID, "hardlink escapes the root filesystem")
    filtered = copy.copy(member)
    filtered.name = name
    filtered.mode = int(member.mode) & 0o0777  # clear setuid, setgid, sticky, and non-permission bits
    return filtered


def _remove_path(rootfs: Path, relative: str) -> None:
    root_real = os.path.realpath(rootfs)
    target = os.path.join(root_real, relative)
    if not _inside(root_real, os.path.realpath(os.path.dirname(target))) or not _inside(root_real, os.path.normpath(target)):
        raise ImageError(RULE_LAYER_INVALID, "whiteout escapes the root filesystem")
    try:
        os.lstat(target)
    except FileNotFoundError:
        return
    if os.path.isdir(target) and not os.path.islink(target):
        shutil.rmtree(target)
    else:
        os.unlink(target)


def _clear_directory(rootfs: Path, relative: str) -> None:
    root_real = os.path.realpath(rootfs)
    target = os.path.join(root_real, relative) if relative else root_real
    if not _inside(root_real, os.path.realpath(target)):
        raise ImageError(RULE_LAYER_INVALID, "opaque whiteout escapes the root filesystem")
    if not os.path.isdir(target) or os.path.islink(target):
        return
    for entry in os.listdir(target):
        _remove_path(rootfs, posixpath.join(relative, entry) if relative else entry)


def _normalize_member_name(name: str) -> str:
    normalized = posixpath.normpath("/" + name.replace("\\", "/")).lstrip("/")
    return "" if normalized == "." else normalized


def _apply_layer(
    archive_path: Path,
    rootfs: Path,
    *,
    compressed: bool,
    budget_bytes: int,
    member_budget: int,
) -> Tuple[int, int]:
    """Apply one layer and return its regular bytes and tar-member count."""

    if member_budget < 1:
        raise ImageError(RULE_TOO_LARGE, "image exceeds its tar-member budget")
    try:
        archive = tarfile.open(str(archive_path), "r:gz" if compressed else "r:")
    except (tarfile.TarError, OSError, EOFError) as exc:
        raise ImageError(RULE_LAYER_INVALID, "layer is not a readable tar: %s" % type(exc).__name__) from exc
    with archive:
        members: List[tarfile.TarInfo] = []
        regular_bytes = 0
        try:
            for member in archive:
                if len(members) >= member_budget:
                    raise ImageError(RULE_TOO_LARGE, "image exceeds its tar-member budget")
                members.append(member)
                if member.isfile():
                    regular_bytes += int(member.size)
                    if regular_bytes > budget_bytes:
                        raise ImageError(RULE_TOO_LARGE, "root filesystem exceeds its byte budget")
        except ImageError:
            raise
        except (tarfile.TarError, OSError, EOFError) as exc:
            raise ImageError(RULE_LAYER_INVALID, "layer index is unreadable: %s" % type(exc).__name__) from exc
        extract: List[tarfile.TarInfo] = []
        for member in members:
            name = _normalize_member_name(member.name)
            if not name:
                continue
            parent, base = posixpath.split(name)
            if base == OPAQUE_WHITEOUT:
                _clear_directory(rootfs, parent)
                continue
            if base.startswith(WHITEOUT_PREFIX):
                _remove_path(rootfs, posixpath.join(parent, base[len(WHITEOUT_PREFIX):]) if parent else base[len(WHITEOUT_PREFIX):])
                continue
            extract.append(member)
        for member in extract:
            try:
                filtered = _rootfs_filter(member, str(rootfs))
                if filtered is not None:
                    # Calling the local filter immediately before each extract
                    # keeps this safe on Python 3.8, where tarfile has no
                    # extraction_filter argument.
                    archive.extract(filtered, path=str(rootfs), set_attrs=True, numeric_owner=True)
            except ImageError:
                raise
            except (tarfile.TarError, OSError, EOFError, ValueError) as exc:
                raise ImageError(RULE_LAYER_INVALID, "layer extraction failed: %s" % type(exc).__name__) from exc
    return regular_bytes, len(members)


def apply_layer(archive_path: Path, rootfs: Path, *, compressed: bool, budget_bytes: int) -> int:
    """Apply one bounded layer; retained as a small test helper."""

    regular_bytes, _members = _apply_layer(
        archive_path,
        rootfs,
        compressed=compressed,
        budget_bytes=budget_bytes,
        member_budget=MAX_IMAGE_TAR_MEMBERS,
    )
    return regular_bytes


def materialize_rootfs(client: RegistryClient, reference: ImageReference, target_dir: Path, *, rules: ImageRules) -> Dict[str, Any]:
    """Build ``target_dir/rootfs`` from the pinned single-platform manifest.

    Every blob is verified against the manifest before it is applied, and the
    manifest itself is verified against ``reference.digest``.
    """

    body, media_type = client.get_manifest(reference)
    if media_type not in MANIFEST_MEDIA_TYPES:
        raise ImageError(RULE_MANIFEST_INVALID, "a pinned image must be a single-platform manifest")
    _config, *layers = manifest_layers(_json_object(body, RULE_MANIFEST_INVALID), rules)
    rootfs = Path(target_dir) / "rootfs"
    rootfs.mkdir(parents=True, exist_ok=False)
    # The root is mounted read-only, but the unprivileged sandbox uid must be
    # able to traverse it even when the runner uses a restrictive umask.
    rootfs.chmod(0o755)
    total = 0
    members = 0
    spool = Path(tempfile.mkdtemp(prefix="lab-arena-layers-", dir=str(target_dir)))
    try:
        for index, layer in enumerate(layers):
            path = spool / ("%03d.layer" % index)
            with open(path, "wb") as handle:
                client.stream_blob(reference.registry, reference.repository, layer.digest, expected_size=layer.size, sink=handle.write)
            layer_bytes, layer_members = _apply_layer(
                path,
                rootfs,
                compressed=layer.media_type in LAYER_GZIP_MEDIA_TYPES,
                budget_bytes=rules.max_rootfs_bytes - total,
                member_budget=MAX_IMAGE_TAR_MEMBERS - members,
            )
            total += layer_bytes
            members += layer_members
            path.unlink()
    finally:
        shutil.rmtree(spool, ignore_errors=True)
    return {"image_digest": reference.digest, "layers": len(layers), "rootfs_bytes": total}


__all__ = [
    "BlobDescriptor",
    "ImageDescriptor",
    "ImageError",
    "ImageReference",
    "ImageRules",
    "IMAGE_RULE_IDS",
    "RegistryClient",
    "apply_layer",
    "manifest_layers",
    "materialize_rootfs",
    "parse_reference",
    "parse_repository",
    "resolve_image",
    "sha256_digest",
    "validate_config_platform",
]
