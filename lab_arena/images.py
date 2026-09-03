"""Container images named by digest (image-by-digest plan, sections 1, 2, 4).

A miner names one image by digest in any public registry. The Arena resolves
the manifest, checks it against the public image rules, copies its blobs into
the Arena repository, and pins the single-platform manifest digest. A runner
materializes the root filesystem of a pinned digest from that repository with
a hardened extractor. No Docker daemon takes part on either side, so no host
unpacks attacker-controlled layers through a root daemon.

Every rejection maps to a published rule id (``IMAGE_RULE_IDS``). The client
speaks the OCI distribution API over ``httpx`` with the bearer challenge flow
registries use for anonymous pulls and credentialed pushes.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import posixpath
import re
import shutil
import tarfile
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Dict, IO, List, Mapping, Optional, Sequence, Tuple
from urllib.parse import urlsplit

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
RULE_NO_ENTRY_COMMAND = "image.no_entry_command"
RULE_DIGEST_MISMATCH = "image.digest_mismatch"
RULE_MIRROR_FAILED = "image.mirror_failed"
RULE_DUPLICATE_ARTIFACT = "image.duplicate_artifact"

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
    RULE_NO_ENTRY_COMMAND,
    RULE_DIGEST_MISMATCH,
    RULE_MIRROR_FAILED,
    RULE_DUPLICATE_ARTIFACT,
)

IMAGE_RULES_SCHEMA_VERSION = "leadpoet.lab_arena.image_rules.v1"
DEFAULT_MAX_IMAGE_BYTES = 2 * 1024 * 1024 * 1024  # compressed layers plus config
DEFAULT_MAX_LAYERS = 64
DEFAULT_MAX_ROOTFS_BYTES = 8 * 1024 * 1024 * 1024  # uncompressed regular-file bytes
MAX_MANIFEST_BYTES = 4 * 1024 * 1024
MAX_CONFIG_BYTES = 4 * 1024 * 1024
MAX_REFERENCE_LENGTH = 512
MAX_ENTRY_COMMAND_ITEMS = 64
MAX_ENTRY_COMMAND_ITEM_CHARS = 4096
MAX_ENVIRONMENT_ENTRIES = 256
MAX_ENVIRONMENT_VALUE_CHARS = 8192
MAX_WORKING_DIR_CHARS = 4096
RESERVED_ENVIRONMENT_PREFIX = "LAB_ARENA_"
STREAM_CHUNK_BYTES = 1024 * 1024
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
PLAIN_HTTP_HOSTS = ("localhost", "127.0.0.1")

_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_COMPONENT = r"[a-z0-9]+(?:(?:[._]|__|-+)[a-z0-9]+)*"
_PATH_RE = re.compile(r"^" + _COMPONENT + r"(?:/" + _COMPONENT + r")*$")
_HOST_RE = re.compile(r"^(?:localhost|127\.0\.0\.1|[a-z0-9](?:[a-z0-9-]*[a-z0-9])?(?:\.[a-z0-9](?:[a-z0-9-]*[a-z0-9])?)+)(?::[0-9]{1,5})?$")
_TAG_RE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}$")
_ENV_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,127}$")
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
    """``registry/repository[:tag]@sha256:<digest>``; the digest is required."""

    registry: str
    repository: str
    digest: str
    tag: Optional[str] = None

    @property
    def name(self) -> str:
        return self.registry + "/" + self.repository

    @property
    def api_registry(self) -> str:
        return DOCKER_HUB_REGISTRY if self.registry in DOCKER_HUB_HOSTS else self.registry

    def __str__(self) -> str:
        tag = ":" + self.tag if self.tag else ""
        return "%s/%s%s@%s" % (self.registry, self.repository, tag, self.digest)

    def with_digest(self, digest: str) -> "ImageReference":
        if not _DIGEST_RE.match(str(digest)):
            raise ImageError(RULE_REFERENCE_INVALID, "digest is invalid")
        return ImageReference(self.registry, self.repository, str(digest), self.tag)


def parse_reference(text: Any) -> ImageReference:
    """Parse a miner's reference. A registry host and a digest are both required."""

    if not isinstance(text, str) or not text or len(text) > MAX_REFERENCE_LENGTH or any(ch.isspace() or ord(ch) < 32 for ch in text):
        raise ImageError(RULE_REFERENCE_INVALID, "reference must be a bounded single-line string")
    if "@" not in text:
        raise ImageError(RULE_REFERENCE_INVALID, "reference must name a digest")
    name_and_tag, digest = text.rsplit("@", 1)
    if not _DIGEST_RE.match(digest):
        raise ImageError(RULE_REFERENCE_INVALID, "digest must be sha256:<64 hex>")
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
    """A resolved single-platform image: what the Arena pins and publishes."""

    reference: ImageReference  # the single-platform manifest, at its source
    submitted_digest: str  # what the miner named (an index digest, or the same)
    manifest_bytes: bytes
    manifest_media_type: str
    config: BlobDescriptor
    layers: Tuple[BlobDescriptor, ...]
    entry_command: Tuple[str, ...]
    environment: Mapping[str, str]
    working_dir: str

    @property
    def image_digest(self) -> str:
        return self.reference.digest

    @property
    def image_size_bytes(self) -> int:
        return int(self.config.size) + sum(int(layer.size) for layer in self.layers)

    def to_document(self) -> Dict[str, Any]:
        """The public, byte-free description recorded on the submission."""

        return {
            "image_digest": self.image_digest,
            "submitted_digest": self.submitted_digest,
            "manifest_media_type": self.manifest_media_type,
            "entry_command": list(self.entry_command),
            "image_environment": dict(self.environment),
            "working_dir": self.working_dir,
            "image_size_bytes": self.image_size_bytes,
            "layer_count": len(self.layers),
        }


def validate_entry_command(value: Any) -> Tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or not value or len(value) > MAX_ENTRY_COMMAND_ITEMS:
        raise ImageError(RULE_NO_ENTRY_COMMAND, "the image must declare ENTRYPOINT or CMD")
    items: List[str] = []
    for item in value:
        if not isinstance(item, str) or not item or len(item) > MAX_ENTRY_COMMAND_ITEM_CHARS or "\x00" in item or "\n" in item:
            raise ImageError(RULE_CONFIG_INVALID, "entry command item is invalid")
        items.append(item)
    return tuple(items)


def validate_image_environment(value: Any) -> Dict[str, str]:
    """``ENV`` as a mapping; the Arena's own ``LAB_ARENA_*`` names are reserved."""

    if value is None:
        return {}
    entries: Dict[str, str] = {}
    if isinstance(value, Mapping):
        pairs = list(value.items())
    elif isinstance(value, (list, tuple)):
        pairs = []
        for item in value:
            if not isinstance(item, str) or "=" not in item:
                raise ImageError(RULE_CONFIG_INVALID, "environment entry is not NAME=value")
            name, _, entry_value = item.partition("=")
            pairs.append((name, entry_value))
    else:
        raise ImageError(RULE_CONFIG_INVALID, "environment must be a list or object")
    if len(pairs) > MAX_ENVIRONMENT_ENTRIES:
        raise ImageError(RULE_CONFIG_INVALID, "too many environment entries")
    for name, entry_value in pairs:
        if not isinstance(name, str) or not _ENV_NAME_RE.match(name):
            raise ImageError(RULE_CONFIG_INVALID, "environment name is invalid")
        if name.startswith(RESERVED_ENVIRONMENT_PREFIX):
            raise ImageError(RULE_CONFIG_INVALID, "%s* environment names are reserved for the Arena" % RESERVED_ENVIRONMENT_PREFIX)
        if not isinstance(entry_value, str) or len(entry_value) > MAX_ENVIRONMENT_VALUE_CHARS or "\x00" in entry_value:
            raise ImageError(RULE_CONFIG_INVALID, "environment value is invalid")
        entries[name] = entry_value
    return entries


def validate_working_dir(value: Any) -> str:
    if value in (None, ""):
        return ""
    if not isinstance(value, str) or not value.startswith("/") or len(value) > MAX_WORKING_DIR_CHARS or "\x00" in value or "\n" in value:
        raise ImageError(RULE_CONFIG_INVALID, "working directory must be an absolute path")
    return value


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


def process_from_config(config_document: Mapping[str, Any], rules: ImageRules) -> Tuple[Tuple[str, ...], Dict[str, str], str]:
    """The pinned process: ENTRYPOINT + CMD, ENV, WORKDIR from the image config."""

    if config_document.get("os") != rules.platform_os or config_document.get("architecture") != rules.platform_architecture:
        raise ImageError(RULE_PLATFORM_UNSUPPORTED, "image platform is not %s/%s" % (rules.platform_os, rules.platform_architecture))
    process = config_document.get("config") or {}
    if not isinstance(process, Mapping):
        raise ImageError(RULE_CONFIG_INVALID, "config.config is not an object")
    entrypoint = process.get("Entrypoint") or []
    cmd = process.get("Cmd") or []
    if not isinstance(entrypoint, list) or not isinstance(cmd, list):
        raise ImageError(RULE_CONFIG_INVALID, "Entrypoint and Cmd must be lists")
    entry_command = validate_entry_command(list(entrypoint) + list(cmd))
    environment = validate_image_environment(process.get("Env"))
    working_dir = validate_working_dir(process.get("WorkingDir"))
    return entry_command, environment, working_dir


# ---------------------------------------------------------------------------
# Registry client
# ---------------------------------------------------------------------------

Credentials = Callable[[str], Optional[Tuple[str, str]]]  # registry host -> (username, password)


class RegistryClient:
    """A minimal OCI distribution client: manifests, blobs, uploads, and tokens.

    ``credentials(registry)`` returns the push credential for the Arena's own
    registry and ``None`` elsewhere; pulls from miners' registries use the
    anonymous bearer challenge flow. Blob downloads follow the registry's
    redirect to its content store without forwarding the authorization header.
    """

    def __init__(
        self,
        *,
        http: Optional[httpx.Client] = None,
        credentials: Optional[Credentials] = None,
        timeout_seconds: float = 60.0,
        plain_http_hosts: Sequence[str] = PLAIN_HTTP_HOSTS,
    ) -> None:
        self._http = http or httpx.Client(http1=True, http2=False, follow_redirects=False, timeout=httpx.Timeout(timeout_seconds))
        self._credentials = credentials or (lambda registry: None)
        self._plain_hosts = tuple(plain_http_hosts)
        self._tokens: Dict[Tuple[str, str], str] = {}
        self._last_token: Dict[str, str] = {}  # the newest bearer per registry, sent proactively
        self._lock = threading.Lock()

    # -- transport ------------------------------------------------------------

    def base_url(self, registry: str) -> str:
        api_host = DOCKER_HUB_REGISTRY if registry in DOCKER_HUB_HOSTS else registry
        host = api_host.rsplit(":", 1)[0] if ":" in api_host and not api_host.endswith("]") else api_host
        scheme = "http" if host in self._plain_hosts else "https"
        return "%s://%s" % (scheme, api_host)

    def _basic(self, registry: str) -> Optional[str]:
        credential = self._credentials(registry)
        if not credential:
            return None
        username, password = credential
        return "Basic " + base64.b64encode(("%s:%s" % (username, password)).encode("utf-8")).decode("ascii")

    def _token(self, registry: str, challenge: Mapping[str, str]) -> str:
        realm = challenge.get("realm") or ""
        parts = urlsplit(realm)
        if parts.scheme not in ("https", "http") or not parts.netloc or (parts.scheme == "http" and parts.hostname not in self._plain_hosts):
            raise ImageError(RULE_UNAVAILABLE, "token realm is invalid")
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
        if basic:
            headers["Authorization"] = basic
        try:
            response = self._http.get(realm, params=params, headers=headers)
        except httpx.HTTPError as exc:
            raise ImageError(RULE_UNAVAILABLE, "token endpoint unreachable: %s" % type(exc).__name__) from exc
        if response.status_code != 200:
            raise ImageError(RULE_UNAVAILABLE, "token endpoint answered %d" % response.status_code)
        try:
            document = response.json()
        except ValueError as exc:
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
        method: str,
        registry: str,
        path: str,
        *,
        headers: Optional[Mapping[str, str]] = None,
        params: Optional[Mapping[str, str]] = None,
        content: Any = None,
        stream: bool = False,
        absolute_url: Optional[str] = None,
    ) -> httpx.Response:
        """Send once; on a 401 challenge, obtain a token and send once more."""

        url = absolute_url or (self.base_url(registry) + path)
        request_headers = dict(headers or {})
        with self._lock:
            bearer = self._last_token.get(registry)
        basic = self._basic(registry)
        if "Authorization" not in request_headers:
            # A bearer already obtained for this registry is sent first: a
            # streamed upload body cannot be replayed after a challenge.
            if bearer:
                request_headers["Authorization"] = "Bearer " + bearer
            elif basic:
                request_headers["Authorization"] = basic
        try:
            request = self._http.build_request(method, url, headers=request_headers, params=params, content=content)
            response = self._http.send(request, stream=stream)
        except httpx.HTTPError as exc:
            raise ImageError(RULE_UNAVAILABLE, "registry unreachable: %s" % type(exc).__name__) from exc
        if response.status_code != 401:
            return response
        challenge_header = response.headers.get("www-authenticate", "")
        response.close()
        scheme = challenge_header.split(" ", 1)[0].strip().lower()
        if scheme != "bearer":
            raise ImageError(RULE_UNAVAILABLE, "registry refused the request (401)")
        challenge = {key.lower(): value for key, value in _CHALLENGE_RE.findall(challenge_header)}
        token = self._token(registry, challenge)
        request_headers["Authorization"] = "Bearer " + token
        if hasattr(content, "seek"):
            content.seek(0)
        try:
            request = self._http.build_request(method, url, headers=request_headers, params=params, content=content)
            response = self._http.send(request, stream=stream)
        except httpx.HTTPError as exc:
            raise ImageError(RULE_UNAVAILABLE, "registry unreachable: %s" % type(exc).__name__) from exc
        if response.status_code == 401:
            response.close()
            with self._lock:
                self._tokens.pop((registry, challenge.get("scope") or ""), None)
                self._last_token.pop(registry, None)
            raise ImageError(RULE_UNAVAILABLE, "registry refused the credential (401)")
        return response

    # -- manifests --------------------------------------------------------------

    def get_manifest(self, reference: ImageReference) -> Tuple[bytes, str]:
        """The manifest bytes named by ``reference.digest`` and their media type."""

        response = self._send("GET", reference.registry, "/v2/%s/manifests/%s" % (reference.repository, reference.digest), headers={"Accept": ACCEPT_MANIFESTS}, stream=True)
        try:
            if response.status_code == 404:
                raise ImageError(RULE_UNAVAILABLE, "manifest %s is not in %s" % (reference.digest[:19], reference.name))
            if response.status_code != 200:
                raise ImageError(RULE_UNAVAILABLE, "manifest request answered %d" % response.status_code)
            body = _read_bounded(response, MAX_MANIFEST_BYTES, RULE_MANIFEST_INVALID)
        finally:
            response.close()
        if sha256_digest(body) != reference.digest:
            raise ImageError(RULE_DIGEST_MISMATCH, "manifest bytes do not hash to the named digest")
        media_type = (response.headers.get("content-type") or "").split(";", 1)[0].strip()
        if not media_type or media_type == "application/json" or media_type == "application/octet-stream":
            declared = _json_object(body, RULE_MANIFEST_INVALID).get("mediaType")
            media_type = str(declared or media_type)
        return body, media_type

    def put_manifest(self, registry: str, repository: str, digest: str, media_type: str, body: bytes) -> None:
        if sha256_digest(body) != digest:
            raise ImageError(RULE_MIRROR_FAILED, "manifest bytes do not hash to the digest being written")
        response = self._send("PUT", registry, "/v2/%s/manifests/%s" % (repository, digest), headers={"Content-Type": media_type}, content=body)
        response.close()
        if response.status_code not in (200, 201, 202):
            raise ImageError(RULE_MIRROR_FAILED, "manifest write answered %d" % response.status_code)
        written = response.headers.get("docker-content-digest")
        if written and written != digest:
            raise ImageError(RULE_MIRROR_FAILED, "registry reports a different manifest digest")

    # -- blobs --------------------------------------------------------------------

    def blob_exists(self, registry: str, repository: str, digest: str) -> bool:
        response = self._send("HEAD", registry, "/v2/%s/blobs/%s" % (repository, digest))
        response.close()
        if response.status_code == 200:
            return True
        if response.status_code == 404:
            return False
        raise ImageError(RULE_UNAVAILABLE, "blob check answered %d" % response.status_code)

    def stream_blob(self, registry: str, repository: str, digest: str, *, expected_size: int, sink: Callable[[bytes], Any]) -> int:
        """Stream one blob into ``sink``; the byte count and digest must match."""

        response = self._send("GET", registry, "/v2/%s/blobs/%s" % (repository, digest), stream=True)
        try:
            hops = 0
            while response.status_code in (301, 302, 303, 307, 308):
                location = response.headers.get("location")
                response.close()
                hops += 1
                if not location or hops > MAX_BLOB_REDIRECTS:
                    raise ImageError(RULE_UNAVAILABLE, "blob redirect chain is invalid")
                if location.startswith("/"):
                    location = self.base_url(registry) + location
                parts = urlsplit(location)
                if parts.scheme != "https" and not (parts.scheme == "http" and parts.hostname in self._plain_hosts):
                    raise ImageError(RULE_UNAVAILABLE, "blob redirect is not https")
                try:
                    response = self._http.send(self._http.build_request("GET", location), stream=True)
                except httpx.HTTPError as exc:
                    raise ImageError(RULE_UNAVAILABLE, "blob store unreachable: %s" % type(exc).__name__) from exc
            if response.status_code == 404:
                raise ImageError(RULE_UNAVAILABLE, "blob %s is missing" % digest[:19])
            if response.status_code != 200:
                raise ImageError(RULE_UNAVAILABLE, "blob request answered %d" % response.status_code)
            hasher = hashlib.sha256()
            received = 0
            for chunk in response.iter_bytes(STREAM_CHUNK_BYTES):
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

    def get_blob(self, registry: str, repository: str, digest: str, *, expected_size: int, max_bytes: int) -> bytes:
        if expected_size > max_bytes:
            raise ImageError(RULE_TOO_LARGE, "blob exceeds %d bytes" % max_bytes)
        chunks: List[bytes] = []
        self.stream_blob(registry, repository, digest, expected_size=expected_size, sink=chunks.append)
        return b"".join(chunks)

    def upload_blob(self, registry: str, repository: str, digest: str, *, size: int, source: Callable[[], IO[bytes]], mount_from: Optional[str] = None) -> str:
        """Ensure ``digest`` exists in ``repository``: existing, mounted, or uploaded.

        ``source()`` opens a readable, seekable stream of the blob bytes and is
        called only when an upload is needed. Returns ``existing``, ``mounted``,
        or ``uploaded``.
        """

        if self.blob_exists(registry, repository, digest):
            return "existing"
        params: Dict[str, str] = {}
        if mount_from:
            params = {"mount": digest, "from": mount_from}
        response = self._send("POST", registry, "/v2/%s/blobs/uploads/" % repository, params=params or None, headers={"Content-Length": "0"})
        response.close()
        if response.status_code == 201:
            return "mounted"
        if response.status_code != 202:
            raise ImageError(RULE_MIRROR_FAILED, "upload start answered %d" % response.status_code)
        location = response.headers.get("location")
        if not location:
            raise ImageError(RULE_MIRROR_FAILED, "upload start returned no location")
        if location.startswith("/"):
            location = self.base_url(registry) + location
        with source() as handle:
            put = self._send(
                "PUT", registry, "", absolute_url=location, params={"digest": digest},
                headers={"Content-Type": "application/octet-stream", "Content-Length": str(int(size))}, content=handle,
            )
        put.close()
        if put.status_code not in (201, 204):
            raise ImageError(RULE_MIRROR_FAILED, "blob upload answered %d" % put.status_code)
        if not self.blob_exists(registry, repository, digest):
            raise ImageError(RULE_MIRROR_FAILED, "uploaded blob did not read back")
        return "uploaded"

    def close(self) -> None:
        self._http.close()


def _read_bounded(response: httpx.Response, limit: int, rule_id: str) -> bytes:
    chunks: List[bytes] = []
    total = 0
    for chunk in response.iter_bytes(STREAM_CHUNK_BYTES):
        total += len(chunk)
        if total > limit:
            raise ImageError(rule_id, "document exceeds %d bytes" % limit)
        chunks.append(chunk)
    return b"".join(chunks)


# ---------------------------------------------------------------------------
# Resolve and mirror (Arena side)
# ---------------------------------------------------------------------------


def resolve_image(client: RegistryClient, reference: ImageReference, rules: ImageRules) -> ImageDescriptor:
    """Fetch and check the image a miner named; returns the pinned single-platform descriptor."""

    submitted_digest = reference.digest
    body, media_type = client.get_manifest(reference)
    document = _json_object(body, RULE_MANIFEST_INVALID)
    if media_type in INDEX_MEDIA_TYPES or (media_type not in MANIFEST_MEDIA_TYPES and "manifests" in document):
        child = _select_platform_manifest(document, rules)
        reference = reference.with_digest(child.digest)
        body, media_type = client.get_manifest(reference)
        document = _json_object(body, RULE_MANIFEST_INVALID)
    if media_type not in MANIFEST_MEDIA_TYPES:
        raise ImageError(RULE_MANIFEST_INVALID, "manifest media type %s is unsupported" % media_type[:80])
    config, *layers = manifest_layers(document, rules)
    config_bytes = client.get_blob(reference.registry, reference.repository, config.digest, expected_size=config.size, max_bytes=MAX_CONFIG_BYTES)
    config_document = _json_object(config_bytes, RULE_CONFIG_INVALID)
    entry_command, environment, working_dir = process_from_config(config_document, rules)
    return ImageDescriptor(
        reference=reference,
        submitted_digest=submitted_digest,
        manifest_bytes=body,
        manifest_media_type=media_type,
        config=config,
        layers=tuple(layers),
        entry_command=entry_command,
        environment=MappingProxyType(dict(environment)),
        working_dir=working_dir,
    )


def mirror_image(client: RegistryClient, descriptor: ImageDescriptor, destination_repository: str, *, spool_dir: Optional[Path] = None) -> ImageReference:
    """Copy every blob of ``descriptor`` into the Arena repository and write the same manifest bytes.

    The manifest digest is preserved because the exact bytes are written.
    Nothing is unpacked: blobs are spooled to disk only long enough to upload.
    """

    registry, repository = parse_repository(destination_repository)
    target = ImageReference(registry, repository, descriptor.image_digest)
    source = descriptor.reference
    mount_from = source.repository if source.api_registry == target.api_registry and source.repository != repository else None
    for blob in (descriptor.config, *descriptor.layers):
        if client.blob_exists(target.registry, target.repository, blob.digest):
            continue
        spool = tempfile.NamedTemporaryFile(prefix="lab-arena-blob-", dir=str(spool_dir) if spool_dir else None, delete=False)
        try:
            def open_source(_blob: BlobDescriptor = blob, _path: str = spool.name) -> IO[bytes]:
                with open(_path, "wb") as handle:
                    client.stream_blob(source.registry, source.repository, _blob.digest, expected_size=_blob.size, sink=handle.write)
                return open(_path, "rb")

            spool.close()
            try:
                client.upload_blob(target.registry, target.repository, blob.digest, size=blob.size, source=open_source, mount_from=mount_from)
            except ImageError as exc:
                if exc.rule_id == RULE_MIRROR_FAILED and mount_from:
                    # A registry that refuses cross-repository mounts still accepts a plain upload.
                    client.upload_blob(target.registry, target.repository, blob.digest, size=blob.size, source=open_source, mount_from=None)
                else:
                    raise
        finally:
            try:
                os.unlink(spool.name)
            except OSError:
                pass
    client.put_manifest(target.registry, target.repository, descriptor.image_digest, descriptor.manifest_media_type, descriptor.manifest_bytes)
    readback, _media = client.get_manifest(target)
    if readback != descriptor.manifest_bytes:
        raise ImageError(RULE_MIRROR_FAILED, "mirrored manifest did not read back byte-identical")
    return target


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
    """The extraction filter: no devices or FIFOs, no hardlink escape, then ``tar_filter``.

    ``tarfile.tar_filter`` strips absolute names, refuses paths that resolve
    outside the destination (through earlier symlinks too), and clears setuid,
    setgid, and sticky bits. Symlink targets may be absolute: inside the
    sandbox they resolve against the sandbox root, which is the intent.
    """

    if member.isdev() or member.isfifo():
        return None  # /dev is a tmpfs at run time; nothing in a rootfs needs device nodes
    if member.islnk():
        root_real = os.path.realpath(dest_path)
        target = os.path.realpath(os.path.join(root_real, member.linkname.lstrip("/")))
        if not _inside(root_real, target):
            raise ImageError(RULE_LAYER_INVALID, "hardlink escapes the root filesystem")
    try:
        return tarfile.tar_filter(member, dest_path)
    except tarfile.FilterError as exc:
        raise ImageError(RULE_LAYER_INVALID, "layer member refused: %s" % type(exc).__name__) from exc


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


def apply_layer(archive_path: Path, rootfs: Path, *, compressed: bool, budget_bytes: int) -> int:
    """Apply one layer tar (whiteouts, then a filtered extraction); returns regular-file bytes."""

    if not hasattr(tarfile, "tar_filter"):
        raise ImageError(RULE_LAYER_INVALID, "this Python lacks tarfile extraction filters (PEP 706)")
    try:
        archive = tarfile.open(str(archive_path), "r:gz" if compressed else "r:")
    except (tarfile.TarError, OSError, EOFError) as exc:
        raise ImageError(RULE_LAYER_INVALID, "layer is not a readable tar: %s" % type(exc).__name__) from exc
    with archive:
        try:
            members = archive.getmembers()
        except (tarfile.TarError, OSError, EOFError) as exc:
            raise ImageError(RULE_LAYER_INVALID, "layer index is unreadable: %s" % type(exc).__name__) from exc
        regular_bytes = sum(int(member.size) for member in members if member.isfile())
        if regular_bytes > budget_bytes:
            raise ImageError(RULE_TOO_LARGE, "root filesystem exceeds its byte budget")
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
        try:
            archive.extractall(path=str(rootfs), members=extract, filter=_rootfs_filter)
        except ImageError:
            raise
        except (tarfile.TarError, OSError, EOFError, ValueError) as exc:
            raise ImageError(RULE_LAYER_INVALID, "layer extraction failed: %s" % type(exc).__name__) from exc
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
    total = 0
    spool = Path(tempfile.mkdtemp(prefix="lab-arena-layers-", dir=str(target_dir)))
    try:
        for index, layer in enumerate(layers):
            path = spool / ("%03d.layer" % index)
            with open(path, "wb") as handle:
                client.stream_blob(reference.registry, reference.repository, layer.digest, expected_size=layer.size, sink=handle.write)
            total += apply_layer(path, rootfs, compressed=layer.media_type in LAYER_GZIP_MEDIA_TYPES, budget_bytes=rules.max_rootfs_bytes - total)
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
    "mirror_image",
    "parse_reference",
    "parse_repository",
    "process_from_config",
    "resolve_image",
    "sha256_digest",
    "validate_entry_command",
    "validate_image_environment",
    "validate_working_dir",
]
