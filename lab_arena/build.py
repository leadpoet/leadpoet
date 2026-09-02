"""Arena submission build pipeline (labarena.md sections 6.2, 6.3, 12.4, 18.5).

Four stages, each a pure function with injectable I/O:

1. ``inspect_package``: bounded gzip decompression, streaming ``tarfile``
   header checks before any member is read, strict manifest validation,
   approved-dependency enforcement, and the canonical ``source_tree_hash``.
   Nothing is ever extracted to disk.
2. Secret scan: the Arena's own marker set, value patterns, and environment
   file rules, run in raise mode over source archives and public documents.
   A finding names a rule id and a location, never the matched value.
3. Offline image build: a rendered Dockerfile that only copies files and
   installs pre-built wheels from an offline wheelhouse. No instruction
   executes miner code, the build runs with ``--network=none``, and the
   builder refuses to start with any credential-shaped environment variable.
4. Screening: one fixture ICP with providers live must yield valid
   ``CompanyOutput`` rows; three synthetic ICPs with providers refused must
   yield none, and identical company sets across ICPs are rejected.

Every rejection maps to a published rule id (``PACKAGE_RULE_IDS``,
``SCREENING_RULE_IDS``, ``secret.*``).
"""

from __future__ import annotations

import fnmatch
import gzip
import hashlib
import io
import json
import re
import shutil
import tarfile
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Any,
    Callable,
    Collection,
    Dict,
    FrozenSet,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
)

from pydantic import ValidationError

from gateway.qualification.models import CompanyOutput
from lab_arena.contracts import (
    ArenaContractError,
    F,
    REQUEST_LIMITS,
    SUBMISSION_PACKAGE_SCHEMA_VERSION,
    check_strict_document,
    document_hash,
    hash_bytes,
    validate_document,
)

# ---------------------------------------------------------------------------
# Package rules and rejection vocabulary (section 6.3)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PackageRules:
    max_package_bytes: int = 25 * 1024 * 1024
    max_files: int = 2000
    max_file_bytes: int = 5 * 1024 * 1024
    max_total_uncompressed_bytes: int = 100 * 1024 * 1024
    max_path_length: int = 200
    max_compression_ratio: int = 200


DEFAULT_PACKAGE_RULES = PackageRules()

RULE_PACKAGE_TOO_LARGE = "package.too_large"
RULE_ARCHIVE_INVALID = "package.archive_invalid"
RULE_PARENT_PATH = "package.parent_path"
RULE_ABSOLUTE_PATH = "package.absolute_path"
RULE_INVALID_PATH = "package.invalid_path"
RULE_PATH_TOO_LONG = "package.path_too_long"
RULE_LINK = "package.link"
RULE_SPECIAL_FILE = "package.special_file"
RULE_DUPLICATE_PATH = "package.duplicate_path"
RULE_CASE_COLLISION = "package.case_collision"
RULE_SPARSE_FILE = "package.sparse_file"
RULE_COMPRESSION_BOMB = "package.compression_bomb"
RULE_TOO_MANY_FILES = "package.too_many_files"
RULE_FILE_TOO_LARGE = "package.file_too_large"
RULE_RESERVED_PATH = "package.reserved_path"
RULE_MANIFEST_MISSING = "package.manifest_missing"
RULE_MANIFEST_INVALID = "package.manifest_invalid"
RULE_ENTRY_POINT_MISSING = "package.entry_point_missing"
RULE_DEPENDENCY_NOT_APPROVED = "package.dependency_not_approved"
RULE_CONSENT_MISSING = "package.consent_missing"
RULE_BUILD_FAILED = "build.docker_failed"
RULE_BUILD_IMAGE_ID_INVALID = "build.image_id_invalid"
RULE_BUILD_INSPECT_MISMATCH = "build.inspect_mismatch"
RULE_BUILD_CONTEXT_INVALID = "build.context_invalid"

PACKAGE_RULE_IDS = (
    RULE_PACKAGE_TOO_LARGE,
    RULE_ARCHIVE_INVALID,
    RULE_PARENT_PATH,
    RULE_ABSOLUTE_PATH,
    RULE_INVALID_PATH,
    RULE_PATH_TOO_LONG,
    RULE_LINK,
    RULE_SPECIAL_FILE,
    RULE_DUPLICATE_PATH,
    RULE_CASE_COLLISION,
    RULE_SPARSE_FILE,
    RULE_COMPRESSION_BOMB,
    RULE_TOO_MANY_FILES,
    RULE_FILE_TOO_LARGE,
    RULE_RESERVED_PATH,
    RULE_MANIFEST_MISSING,
    RULE_MANIFEST_INVALID,
    RULE_ENTRY_POINT_MISSING,
    RULE_DEPENDENCY_NOT_APPROVED,
    RULE_CONSENT_MISSING,
    RULE_BUILD_FAILED,
    RULE_BUILD_IMAGE_ID_INVALID,
    RULE_BUILD_INSPECT_MISMATCH,
    RULE_BUILD_CONTEXT_INVALID,
)

MANIFEST_PATH = "manifest.json"
REQUIREMENTS_LOCK_PATH = "requirements.lock"
# Paths the builder writes itself; a package may not carry them.
RESERVED_PATHS = frozenset({REQUIREMENTS_LOCK_PATH})
_MAX_MANIFEST_BYTES = 262_144
_DECOMPRESS_CHUNK = 1024 * 1024
_PATH_COMPONENT_RE = re.compile(r"^[A-Za-z0-9._-]+$")
_DRIVE_RE = re.compile(r"^[A-Za-z]:")
_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f]")


class PackageRejected(ArenaContractError):
    """A package violates a published rule. ``rule_id`` names the rule."""

    def __init__(self, rule_id: str, detail: str = "") -> None:
        self.rule_id = rule_id
        self.detail = detail
        super().__init__("%s: %s" % (rule_id, detail) if detail else rule_id)


def normalize_member_path(name: Any) -> str:
    """Return the canonical relative path for an archive member or raise.

    Accepts ``a/b.py`` and ``./a/b.py``; rejects absolute paths, drive
    letters, ``..`` components, empty or ``.`` components, backslashes,
    control characters, and components outside ``[A-Za-z0-9._-]``.
    """

    if not isinstance(name, str) or not name:
        raise PackageRejected(RULE_INVALID_PATH, "member path is empty")
    if "\\" in name or _CONTROL_RE.search(name):
        raise PackageRejected(RULE_INVALID_PATH, "member path contains forbidden characters")
    if name.startswith("/") or _DRIVE_RE.match(name):
        raise PackageRejected(RULE_ABSOLUTE_PATH, "member path is absolute")
    parts = name.split("/")
    if parts and parts[0] == ".":
        parts = parts[1:]
    if not parts:
        raise PackageRejected(RULE_INVALID_PATH, "member path is empty")
    for part in parts:
        if part == "..":
            raise PackageRejected(RULE_PARENT_PATH, "member path contains a parent reference")
        if part in ("", "."):
            raise PackageRejected(RULE_INVALID_PATH, "member path has an empty or dot component")
        if not _PATH_COMPONENT_RE.match(part):
            raise PackageRejected(RULE_INVALID_PATH, "member path component has forbidden characters")
    return "/".join(parts)


def _safe_detail_name(name: Any) -> str:
    text = str(name)
    text = _CONTROL_RE.sub("?", text)
    return text if len(text) <= 120 else text[:117] + "..."


def _decompress_bounded(archive: bytes, rules: PackageRules) -> bytes:
    """Gunzip with a hard cap on output bytes and on the compression ratio.

    The tar stream is bounded before any header is parsed so a crafted
    extended header can never make ``tarfile`` allocate an unbounded buffer.
    """

    if archive[:2] != b"\x1f\x8b":
        raise PackageRejected(RULE_ARCHIVE_INVALID, "package must be a gzip tarball")
    limit = rules.max_total_uncompressed_bytes
    ratio_limit = rules.max_compression_ratio * len(archive)
    chunks: List[bytes] = []
    total = 0
    try:
        with gzip.GzipFile(fileobj=io.BytesIO(archive), mode="rb") as stream:
            while True:
                chunk = stream.read(_DECOMPRESS_CHUNK)
                if not chunk:
                    break
                total += len(chunk)
                if total > limit:
                    raise PackageRejected(RULE_COMPRESSION_BOMB, "uncompressed stream exceeds %d bytes" % limit)
                if total > ratio_limit:
                    raise PackageRejected(
                        RULE_COMPRESSION_BOMB,
                        "compression ratio exceeds %d" % rules.max_compression_ratio,
                    )
                chunks.append(chunk)
    except (OSError, EOFError, zlib.error) as exc:
        raise PackageRejected(RULE_ARCHIVE_INVALID, "gzip stream is invalid") from exc
    return b"".join(chunks)


PathEntries = Dict[str, Tuple[str, str]]


def _check_member(member: tarfile.TarInfo, rules: PackageRules, entries: PathEntries) -> Optional[str]:
    """Validate one header before its data is read. ``None`` skips a directory."""

    if member.isdir():
        stripped = member.name.rstrip("/")
        if stripped in ("", "."):
            return None
        path = normalize_member_path(stripped)
        _register_path(path, "dir", entries)
        return None
    path = normalize_member_path(member.name)
    if member.issym() or member.islnk():
        raise PackageRejected(RULE_LINK, "member %s is a link" % _safe_detail_name(path))
    if member.issparse():
        raise PackageRejected(RULE_SPARSE_FILE, "member %s is sparse" % _safe_detail_name(path))
    if not member.isfile():
        raise PackageRejected(RULE_SPECIAL_FILE, "member %s is not a regular file" % _safe_detail_name(path))
    if len(path) > rules.max_path_length:
        raise PackageRejected(RULE_PATH_TOO_LONG, "member path exceeds %d characters" % rules.max_path_length)
    if path in RESERVED_PATHS:
        raise PackageRejected(RULE_RESERVED_PATH, "member %s is written by the builder" % path)
    _register_path(path, "file", entries)
    if member.size > rules.max_file_bytes:
        raise PackageRejected(RULE_FILE_TOO_LARGE, "member %s exceeds %d bytes" % (_safe_detail_name(path), rules.max_file_bytes))
    return path


def _register_path(path: str, kind: str, entries: PathEntries) -> None:
    """Track every path, explicit directory, and implied parent directory.

    Two entries whose case-folded paths match but whose exact paths differ
    are a case collision; the same exact path twice is a duplicate, except
    that an explicit directory may follow the implied parent of a file.
    """

    parts = path.split("/")
    for depth in range(1, len(parts)):
        _register_one("/".join(parts[:depth]), "implied", entries)
    _register_one(path, kind, entries)


def _register_one(path: str, kind: str, entries: PathEntries) -> None:
    folded = path.casefold()
    existing = entries.get(folded)
    if existing is None:
        entries[folded] = (path, kind)
        return
    existing_path, existing_kind = existing
    if existing_path != path:
        raise PackageRejected(RULE_CASE_COLLISION, "member %s collides with %s" % (_safe_detail_name(path), _safe_detail_name(existing_path)))
    if kind == "implied" and existing_kind != "file":
        return
    if kind == "dir" and existing_kind == "implied":
        entries[folded] = (path, "dir")
        return
    raise PackageRejected(RULE_DUPLICATE_PATH, "member %s appears twice" % _safe_detail_name(path))


def source_tree_hash(files: Mapping[str, bytes]) -> str:
    """Canonical hash of the source tree: sorted ``{path, sha256, size, mode}``."""

    entries = [
        {"path": path, "sha256": hash_bytes(files[path]), "size": len(files[path]), "mode": "file"}
        for path in sorted(files)
    ]
    return document_hash(entries)


@dataclass(frozen=True)
class PackageInspection:
    package_hash: str
    source_tree_hash: str
    manifest: Dict[str, Any]
    entry_point: str
    dependency_lock: Tuple[str, ...]
    files: Dict[str, bytes]
    executable_paths: FrozenSet[str]
    package_bytes: int
    uncompressed_bytes: int
    file_count: int


def inspect_package(archive_bytes: bytes, rules: PackageRules = DEFAULT_PACKAGE_RULES) -> PackageInspection:
    """Inspect a submission package without extracting anything to disk.

    Per-member rules are evaluated in this order before the member's data is
    read: path shape, links, sparse members, special files, path length,
    reserved names, duplicates and case collisions, per-file size, then file
    count. The gzip layer is bounded first, so a compression bomb is rejected
    before ``tarfile`` sees a single header.
    """

    if not isinstance(archive_bytes, (bytes, bytearray, memoryview)):
        raise PackageRejected(RULE_ARCHIVE_INVALID, "archive must be bytes")
    archive = bytes(archive_bytes)
    if not archive:
        raise PackageRejected(RULE_ARCHIVE_INVALID, "archive is empty")
    if len(archive) > rules.max_package_bytes:
        raise PackageRejected(RULE_PACKAGE_TOO_LARGE, "package exceeds %d bytes" % rules.max_package_bytes)
    stream = _decompress_bounded(archive, rules)
    files: Dict[str, bytes] = {}
    executable: Set[str] = set()
    entries: PathEntries = {}
    try:
        with tarfile.open(fileobj=io.BytesIO(stream), mode="r:") as tar:
            for member in tar:
                path = _check_member(member, rules, entries)
                if path is None:
                    continue
                if len(files) + 1 > rules.max_files:
                    raise PackageRejected(RULE_TOO_MANY_FILES, "package exceeds %d files" % rules.max_files)
                extracted = tar.extractfile(member)
                if extracted is None:
                    raise PackageRejected(RULE_ARCHIVE_INVALID, "member %s has no data" % _safe_detail_name(path))
                content = extracted.read(member.size + 1)
                if len(content) != member.size:
                    raise PackageRejected(RULE_ARCHIVE_INVALID, "member %s is truncated" % _safe_detail_name(path))
                files[path] = content
                if member.mode & 0o111:
                    executable.add(path)
    except PackageRejected:
        raise
    except (tarfile.TarError, EOFError, OSError, ValueError) as exc:
        raise PackageRejected(RULE_ARCHIVE_INVALID, "tar stream is invalid: %s" % type(exc).__name__) from exc
    manifest, entry_point, lock = _parse_manifest(files)
    return PackageInspection(
        package_hash=hash_bytes(archive),
        source_tree_hash=source_tree_hash(files),
        manifest=manifest,
        entry_point=entry_point,
        dependency_lock=lock,
        files=files,
        executable_paths=frozenset(executable),
        package_bytes=len(archive),
        uncompressed_bytes=len(stream),
        file_count=len(files),
    )


def accept_package(archive_bytes: bytes, rules: PackageRules = DEFAULT_PACKAGE_RULES) -> PackageInspection:
    """Inspect a package and run the raise-mode secret scan (section 6.3 steps 2-3)."""

    inspection = inspect_package(archive_bytes, rules)
    scan_source_archive_raise(inspection.files, executable_paths=inspection.executable_paths)
    return inspection


# ---------------------------------------------------------------------------
# Manifest (section 6.2)
# ---------------------------------------------------------------------------

MANIFEST_FIELDS = (
    F("schema_version", "str", choices=(SUBMISSION_PACKAGE_SCHEMA_VERSION,)),
    F("entry_point", "str", minimum=1, maximum=200),
    F("dependency_lock", "list[str]", minimum=0, maximum=128),
    F(
        "consent",
        "object",
        fields=(
            F("source_publication", "bool"),
            F("public_rerun", "bool"),
        ),
    ),
    F("files", "list[str]", required=False, minimum=0, maximum=4096),
)


def _parse_manifest(files: Mapping[str, bytes]) -> Tuple[Dict[str, Any], str, Tuple[str, ...]]:
    raw = files.get(MANIFEST_PATH)
    if raw is None:
        raise PackageRejected(RULE_MANIFEST_MISSING, "%s is required at the package root" % MANIFEST_PATH)
    if len(raw) > _MAX_MANIFEST_BYTES:
        raise PackageRejected(RULE_MANIFEST_INVALID, "manifest exceeds %d bytes" % _MAX_MANIFEST_BYTES)
    try:
        parsed = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise PackageRejected(RULE_MANIFEST_INVALID, "manifest is not UTF-8 JSON") from None
    if not isinstance(parsed, Mapping):
        raise PackageRejected(RULE_MANIFEST_INVALID, "manifest must be a JSON object")
    consent = parsed.get("consent")
    if (
        not isinstance(consent, Mapping)
        or consent.get("source_publication") is not True
        or consent.get("public_rerun") is not True
    ):
        raise PackageRejected(RULE_CONSENT_MISSING, "source_publication and public_rerun consent must both be true")
    try:
        check_strict_document(parsed, REQUEST_LIMITS)
        manifest = validate_document(parsed, MANIFEST_FIELDS)
    except ArenaContractError as exc:
        raise PackageRejected(RULE_MANIFEST_INVALID, str(exc)) from None
    try:
        entry_point = normalize_member_path(manifest["entry_point"])
    except PackageRejected as exc:
        raise PackageRejected(RULE_MANIFEST_INVALID, "entry_point path is invalid: %s" % exc.rule_id) from None
    if not entry_point.endswith(".py"):
        raise PackageRejected(RULE_MANIFEST_INVALID, "entry_point must be a Python file")
    if entry_point == MANIFEST_PATH or entry_point not in files:
        raise PackageRejected(RULE_ENTRY_POINT_MISSING, "entry_point %s is not in the package" % entry_point)
    lock = validate_dependency_lock(manifest["dependency_lock"])
    listed = manifest.get("files")
    if listed is not None:
        try:
            normalized = sorted({normalize_member_path(item) for item in listed})
        except PackageRejected as exc:
            raise PackageRejected(RULE_MANIFEST_INVALID, "files entry is invalid: %s" % exc.rule_id) from None
        actual = sorted(path for path in files if path != MANIFEST_PATH)
        if normalized != actual:
            raise PackageRejected(RULE_MANIFEST_INVALID, "files list does not match the archive contents")
    manifest["entry_point"] = entry_point
    manifest["dependency_lock"] = list(lock)
    return manifest, entry_point, lock


# ---------------------------------------------------------------------------
# Approved dependency set (section 6.2)
# ---------------------------------------------------------------------------

# V1 seed pins. Changing any pin or the requires table changes
# ``approved_dependency_set_hash`` which is bound into every round
# configuration, so this is a versioned decision, not a convenience edit.
# The wheelhouse built by the operator must contain exactly these wheels.
APPROVED_DEPENDENCIES: Tuple[str, ...] = (
    "aiohappyeyeballs==2.6.1",
    "aiohttp==3.13.2",
    "aiosignal==1.4.0",
    "annotated-types==0.7.0",
    "anyio==4.13.0",
    "attrs==25.4.0",
    "beautifulsoup4==4.13.4",
    "certifi==2025.10.5",
    "charset-normalizer==3.4.4",
    "frozenlist==1.8.0",
    "h11==0.16.0",
    "httpcore==1.0.9",
    "httpx==0.28.1",
    "idna==3.11",
    "lxml==6.1.1",
    "multidict==6.7.0",
    "numpy==2.3.4",
    "propcache==0.4.1",
    "pydantic==2.12.4",
    "pydantic-core==2.41.5",
    "python-dateutil==2.9.0.post0",
    "requests==2.32.5",
    "six==1.17.0",
    "soupsieve==2.7",
    "typing-extensions==4.15.0",
    "typing-inspection==0.4.2",
    "urllib3==2.5.0",
    "yarl==1.22.0",
)

# Runtime requirements of each approved distribution, by normalized name.
# The image installs with ``--no-deps``, so the builder closes a miner's lock
# over this table; every name here must itself be an approved pin.
APPROVED_DEPENDENCY_REQUIRES: Mapping[str, Tuple[str, ...]] = {
    "aiohttp": ("aiohappyeyeballs", "aiosignal", "attrs", "frozenlist", "multidict", "propcache", "yarl"),
    "aiosignal": ("frozenlist", "typing-extensions"),
    "anyio": ("idna", "typing-extensions"),
    "beautifulsoup4": ("soupsieve", "typing-extensions"),
    "httpcore": ("certifi", "h11"),
    "httpx": ("anyio", "certifi", "httpcore", "idna"),
    "pydantic": ("annotated-types", "pydantic-core", "typing-extensions", "typing-inspection"),
    "pydantic-core": ("typing-extensions",),
    "python-dateutil": ("six",),
    "requests": ("certifi", "charset-normalizer", "idna", "urllib3"),
    "typing-inspection": ("typing-extensions",),
    "yarl": ("idna", "multidict", "propcache"),
}

_LOCK_ENTRY_RE = re.compile(r"^([A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?)==([A-Za-z0-9][A-Za-z0-9.!+-]*)$")


def normalize_distribution_name(name: str) -> str:
    """PEP 503 normalization: runs of ``-_.`` collapse to ``-``, lowercase."""

    return re.sub(r"[-_.]+", "-", name).lower()


def _approved_index() -> Dict[str, str]:
    index: Dict[str, str] = {}
    for pin in APPROVED_DEPENDENCIES:
        match = _LOCK_ENTRY_RE.match(pin)
        if match is None:
            raise ArenaContractError("approved dependency pin is malformed")
        key = normalize_distribution_name(match.group(1))
        if key in index:
            raise ArenaContractError("approved dependency set lists %s twice" % key)
        index[key] = pin
    for name, requires in APPROVED_DEPENDENCY_REQUIRES.items():
        if name not in index or any(item not in index for item in requires):
            raise ArenaContractError("approved dependency requires table names an unapproved distribution")
    return index


APPROVED_DEPENDENCY_INDEX: Mapping[str, str] = _approved_index()


def approved_dependency_set_hash() -> str:
    return document_hash(
        {
            "schema_version": "leadpoet.lab_arena.approved_dependencies.v1",
            "pins": sorted(APPROVED_DEPENDENCIES),
            "requires": {name: list(APPROVED_DEPENDENCY_REQUIRES[name]) for name in sorted(APPROVED_DEPENDENCY_REQUIRES)},
        }
    )


def validate_dependency_lock(entries: Sequence[Any]) -> Tuple[str, ...]:
    """Return the canonical approved pins for a lock or raise ``PackageRejected``."""

    seen: Set[str] = set()
    out: List[str] = []
    for entry in entries:
        match = _LOCK_ENTRY_RE.match(entry) if isinstance(entry, str) else None
        if match is None:
            raise PackageRejected(RULE_MANIFEST_INVALID, "dependency lock entries must be name==version")
        name, version = match.groups()
        key = normalize_distribution_name(name)
        if key in seen:
            raise PackageRejected(RULE_MANIFEST_INVALID, "dependency %s is listed twice" % key)
        seen.add(key)
        pin = APPROVED_DEPENDENCY_INDEX.get(key)
        if pin is None or pin.split("==", 1)[1] != version:
            raise PackageRejected(RULE_DEPENDENCY_NOT_APPROVED, "%s==%s is not in the approved dependency set" % (key, version))
        out.append(pin)
    return tuple(out)


def resolve_dependency_closure(lock: Sequence[str]) -> Tuple[str, ...]:
    """Close an approved lock over ``APPROVED_DEPENDENCY_REQUIRES``, sorted by name."""

    pins = validate_dependency_lock(lock)
    wanted: Set[str] = set()
    queue = [normalize_distribution_name(pin.split("==", 1)[0]) for pin in pins]
    while queue:
        name = queue.pop()
        if name in wanted:
            continue
        wanted.add(name)
        queue.extend(APPROVED_DEPENDENCY_REQUIRES.get(name, ()))
    return tuple(APPROVED_DEPENDENCY_INDEX[name] for name in sorted(wanted))


# ---------------------------------------------------------------------------
# Arena secret scan (sections 6.3, 12.4, 18.5)
# ---------------------------------------------------------------------------

SECRET_MARKERS = (
    "sk-or-",
    "openrouter_api_key",
    "raw_openrouter_key",
    "raw_secret",
    "service_role",
    "private_repo",
    "exa_api_key",
    "scrapingdog_api_key",
    "aws_secret_access_key",
    "-----begin",
    "authorization: bearer",
    "api_key=",
    "sb_secret_",
)
SECRET_KEY_MARKERS = (
    "api_key",
    "raw_secret",
    "raw_openrouter",
    "credential",
    "proxy_url",
    "private_key",
    "secret_key",
    "password",
    "passwd",
)
SECRET_TOKEN_KEY_MARKERS = (
    "access_token",
    "api_token",
    "auth_token",
    "bearer_token",
    "refresh_token",
    "session_token",
    "token_key",
    "token_secret",
    "token_value",
)


@dataclass(frozen=True)
class SecretPattern:
    """A value shape. ``key_hint`` restricts it to keys or lines naming a provider."""

    name: str
    regex: "re.Pattern[str]"
    key_hint: Optional[str] = None


SECRET_VALUE_PATTERNS: Tuple[SecretPattern, ...] = (
    SecretPattern("openrouter_key", re.compile(r"sk-or-[A-Za-z0-9_-]{20,}")),
    SecretPattern("generic_sk_key", re.compile(r"(?<![A-Za-z0-9])sk-[A-Za-z0-9]{20,}")),
    SecretPattern("bearer_token", re.compile(r"Bearer\s+[A-Za-z0-9\-._~+/]{16,}=*", re.IGNORECASE)),
    SecretPattern(
        "exa_key",
        re.compile(r"(?<![0-9a-fA-F])[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}(?![0-9a-fA-F])"),
        key_hint="exa",
    ),
    SecretPattern(
        "scrapingdog_key",
        re.compile(r"(?<![0-9a-zA-Z])[0-9a-f]{32}(?![0-9a-zA-Z])"),
        key_hint="scrapingdog",
    ),
    SecretPattern("aws_access_key_id", re.compile(r"(?<![A-Z0-9])AKIA[0-9A-Z]{16}(?![A-Z0-9])")),
    # Broader than an https-only shape: any URL scheme carrying user:password@
    # (postgres://, redis://, amqp://, ...) is a secret-bearing URL.
    SecretPattern("url_userinfo", re.compile(r"(?<![A-Za-z0-9+.-])[A-Za-z][A-Za-z0-9+.-]{0,15}://[^/\s:@]+:[^/\s@]+@")),
)

# Basename globs (case-insensitive) rejected from any source archive.
ENVIRONMENT_FILE_PATTERNS = (
    ".env",
    ".env.*",
    "*.env",
    ".envrc",
    ".netrc",
    "credentials.json",
    "*.pem",
    "id_rsa*",
)

RULE_SECRET_ENVIRONMENT_FILE = "secret.environment_file"
RULE_SECRET_BINARY_MEMBER = "secret.binary_member"
RULE_BUILDER_ENVIRONMENT_SECRET = "builder.environment_secret"
_MARKER_LABEL_TRANSLATION = str.maketrans({"-": "_", "_": "-", ".": "-"})


class SecretMaterialFound(ArenaContractError):
    """Secret material was found. Carries a rule id and a location, never a value."""

    def __init__(self, rule_id: str, location: str) -> None:
        self.rule_id = rule_id
        self.location = location
        super().__init__("%s at %s" % (rule_id, location))


def _secret_marker_label(marker: str) -> str:
    """Defanged marker label: swapping ``-``/``_`` keeps the label readable while
    guaranteeing a rule id never re-trips this scanner or a downstream one."""

    label = marker.translate(_MARKER_LABEL_TRANSLATION)
    label = re.sub(r"[^A-Za-z0-9_-]+", "-", label).strip("-_")
    if not label or any(existing in label.lower() for existing in SECRET_MARKERS):
        return "marker-" + hashlib.sha256(marker.encode("utf-8")).hexdigest()[:8]
    return label


def _first_secret_key_marker(lowered_key: str) -> Optional[str]:
    for marker in SECRET_KEY_MARKERS:
        if marker in lowered_key:
            return marker
    for marker in SECRET_TOKEN_KEY_MARKERS:
        if marker in lowered_key:
            return marker
    return None


_HINT_PATTERNS: Dict[str, "re.Pattern[str]"] = {}


def _hint_matches(hint_word: str, text: str) -> bool:
    """``exa`` matches ``EXA_KEY``, ``exa.search`` or ``$.exa`` but not ``example``."""

    regex = _HINT_PATTERNS.get(hint_word)
    if regex is None:
        regex = re.compile(r"(?<![a-z])%s(?![a-z])" % re.escape(hint_word), re.IGNORECASE)
        _HINT_PATTERNS[hint_word] = regex
    return regex.search(text) is not None


def _scan_text(text: str, location: str, *, hint: str = "") -> None:
    """Markers and unconditional patterns over ``text``; hinted patterns when
    ``hint`` (a key path or file path) names the provider as a word."""

    lowered = text.lower()
    for marker in SECRET_MARKERS:
        if marker in lowered:
            raise SecretMaterialFound("secret.marker." + _secret_marker_label(marker), location)
    for pattern in SECRET_VALUE_PATTERNS:
        if pattern.key_hint is not None and not _hint_matches(pattern.key_hint, hint):
            continue
        if pattern.regex.search(text):
            raise SecretMaterialFound("secret.pattern." + pattern.name, location)


def _scan_source_text(text: str, path: str) -> None:
    _scan_text(path, path)
    _scan_text(text, path, hint=path)
    hinted = [pattern for pattern in SECRET_VALUE_PATTERNS if pattern.key_hint is not None]
    for number, line in enumerate(text.splitlines(), start=1):
        for pattern in hinted:
            if _hint_matches(pattern.key_hint or "", line) and pattern.regex.search(line):
                raise SecretMaterialFound("secret.pattern." + pattern.name, "%s:%d" % (path, number))


def is_environment_file(path: str) -> bool:
    basename = path.rsplit("/", 1)[-1].lower()
    return any(fnmatch.fnmatchcase(basename, pattern) for pattern in ENVIRONMENT_FILE_PATTERNS)


def scan_source_archive_raise(files: Mapping[str, bytes], *, executable_paths: Collection[str] = ()) -> None:
    """Raise-mode scan over a source archive (section 6.3 step 3).

    Rules, in order per file: environment files are rejected by basename; a
    member containing a NUL byte is binary and is rejected unless it lives
    under ``data/`` and has no executable bit (its bytes are still scanned
    as latin-1 text); every other member is decoded as UTF-8 with
    replacement and scanned. Marker and unconditional pattern hits anywhere
    in a file, including docstrings and comments, are positives: a ``Bearer``
    token example in documentation is rejected by design. The Exa and
    ScrapingDog value shapes apply only when the path or the same line
    mentions the provider name. Image digests (``sha256:<64 hex>``) never
    match.
    """

    executable = set(executable_paths)
    for path in sorted(files):
        if is_environment_file(path):
            raise SecretMaterialFound(RULE_SECRET_ENVIRONMENT_FILE, path)
        content = bytes(files[path])
        if b"\x00" in content:
            if not path.startswith("data/") or path in executable:
                raise SecretMaterialFound(RULE_SECRET_BINARY_MEMBER, path)
            text = content.decode("latin-1")
        else:
            text = content.decode("utf-8", errors="replace")
        _scan_source_text(text, path)


def scan_document_raise(document: Any, *, path: str = "$") -> None:
    """Raise-mode scan over a public document (section 12.4).

    Mapping keys that look like credentials are rejected by name; string
    values are scanned for markers and patterns; the hinted provider shapes
    apply when any enclosing key on the value's path names the provider.
    """

    _walk_document(document, path, "")


def _walk_document(value: Any, location: str, key_hint: str) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            name = str(key)
            marker = _first_secret_key_marker(name.lower())
            child = location + "." + name
            if marker is not None:
                raise SecretMaterialFound("secret.key_name." + _secret_marker_label(marker), child)
            _walk_document(item, child, child)
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _walk_document(item, "%s[%d]" % (location, index), key_hint)
        return
    if isinstance(value, str):
        _scan_text(value, location, hint=key_hint)


def contains_secret_material(value: Any) -> bool:
    try:
        scan_document_raise(value)
    except SecretMaterialFound:
        return True
    return False


# ---------------------------------------------------------------------------
# Offline image build (section 6.3 step 4)
# ---------------------------------------------------------------------------

IMAGE_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_IMAGE_REPOSITORY_RE = re.compile(r"^[a-z0-9][a-z0-9._/:-]{0,254}$")
BUILDER_FORBIDDEN_ENVIRONMENT_MARKERS = (
    "secret",
    "token",
    "password",
    "passwd",
    "api_key",
    "apikey",
    "credential",
    "private_key",
    "authorization",
    "openrouter",
    "aws_access_key",
    "aws_session",
    "exa_",
    "scrapingdog",
    "supabase",
    "kms",
)
DockerRunner = Callable[[Sequence[str], int], Any]


@dataclass(frozen=True)
class BuildSpec:
    """Everything the builder needs; validated on construction."""

    base_image: str
    base_image_digest: str
    wheelhouse_dir: Path
    entry_point: str
    source_files: Mapping[str, bytes]
    dependency_lock: Tuple[str, ...]
    image_tag: Optional[str] = None

    def __post_init__(self) -> None:
        if not isinstance(self.base_image, str) or not _IMAGE_REPOSITORY_RE.match(self.base_image) or "@" in self.base_image:
            raise ArenaContractError("base_image must be a plain image repository")
        if not isinstance(self.base_image_digest, str) or not IMAGE_DIGEST_RE.match(self.base_image_digest):
            raise ArenaContractError("base_image_digest must be sha256:<64 hex>")
        if not isinstance(self.source_files, Mapping) or not self.source_files:
            raise ArenaContractError("source_files must be a non-empty mapping")
        for path, content in self.source_files.items():
            if normalize_member_path(path) != path:
                raise ArenaContractError("source path %s is not canonical" % _safe_detail_name(path))
            if path in RESERVED_PATHS:
                raise ArenaContractError("source path %s is reserved for the builder" % path)
            if not isinstance(content, (bytes, bytearray)):
                raise ArenaContractError("source file %s must be bytes" % path)
        entry = normalize_member_path(self.entry_point)
        if entry != self.entry_point or not entry.endswith(".py") or entry not in self.source_files:
            raise ArenaContractError("entry_point must be a canonical .py path inside source_files")
        validate_dependency_lock(self.dependency_lock)
        if self.image_tag is not None and (
            not isinstance(self.image_tag, str) or not re.match(r"^[a-z0-9][a-z0-9._/-]*(?::[A-Za-z0-9._-]{1,128})?$", self.image_tag)
        ):
            raise ArenaContractError("image_tag has an invalid shape")


def render_dockerfile(spec: BuildSpec) -> str:
    """Render the only Dockerfile shape the builder accepts.

    The single ``RUN`` installs pre-built wheels offline with ``--no-deps``
    and ``--only-binary=:all:`` so no setup script, install hook, or miner
    file is ever executed during the build.
    """

    lines = [
        "FROM %s@%s" % (spec.base_image, spec.base_image_digest),
        "COPY source/ /model/",
        "COPY wheelhouse/ /wheelhouse/",
        "RUN pip install --no-index --no-deps --only-binary=:all: --find-links /wheelhouse -r /model/%s" % REQUIREMENTS_LOCK_PATH,
        "USER 65534:65534",
        "ENV PYTHONDONTWRITEBYTECODE=1 TZ=UTC",
        "ENTRYPOINT " + json.dumps(["python3", "/model/" + spec.entry_point]),
    ]
    return "\n".join(lines) + "\n"


def write_build_context(spec: BuildSpec, context_dir: Path) -> Path:
    """Materialize ``Dockerfile``, ``source/``, and ``wheelhouse/`` in an empty directory.

    Source files are written with mode ``0644`` (no executable bit is ever
    propagated), the lock file is the closed dependency set, and only
    ``*.whl`` files are copied from the wheelhouse.
    """

    context = Path(context_dir)
    context.mkdir(parents=True, exist_ok=True)
    if any(context.iterdir()):
        raise PackageRejected(RULE_BUILD_CONTEXT_INVALID, "build context directory must be empty")
    source_root = context / "source"
    source_root.mkdir()
    resolved_root = source_root.resolve()
    for path, content in spec.source_files.items():
        target = source_root / path
        if resolved_root not in target.resolve().parents:
            raise PackageRejected(RULE_BUILD_CONTEXT_INVALID, "source path escapes the context")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(bytes(content))
        target.chmod(0o644)
    lock_lines = resolve_dependency_closure(spec.dependency_lock)
    lock_path = source_root / REQUIREMENTS_LOCK_PATH
    lock_path.write_text("".join(line + "\n" for line in lock_lines), encoding="utf-8")
    lock_path.chmod(0o644)
    wheelhouse = context / "wheelhouse"
    wheelhouse.mkdir()
    source_wheelhouse = Path(spec.wheelhouse_dir)
    if not source_wheelhouse.is_dir():
        raise PackageRejected(RULE_BUILD_CONTEXT_INVALID, "wheelhouse directory does not exist")
    for item in sorted(source_wheelhouse.iterdir()):
        if item.is_symlink() or not item.is_file() or item.suffix != ".whl":
            raise PackageRejected(RULE_BUILD_CONTEXT_INVALID, "wheelhouse may contain only wheel files")
        shutil.copyfile(item, wheelhouse / item.name)
        (wheelhouse / item.name).chmod(0o644)
    dockerfile = context / "Dockerfile"
    dockerfile.write_text(render_dockerfile(spec), encoding="utf-8")
    dockerfile.chmod(0o644)
    return dockerfile


def docker_build_argv(spec: BuildSpec, context_dir: Path, iidfile: Path, *, docker_binary: str = "docker") -> Tuple[str, ...]:
    argv = [
        docker_binary,
        "build",
        "--network=none",
        "--pull=false",
        "--iidfile",
        str(iidfile),
        "--file",
        str(Path(context_dir) / "Dockerfile"),
    ]
    if spec.image_tag:
        argv.extend(["--tag", spec.image_tag])
    argv.append(str(context_dir))
    return tuple(argv)


def _environment_name_looks_secret(name: str) -> bool:
    lowered = name.lower()
    return any(marker in lowered for marker in BUILDER_FORBIDDEN_ENVIRONMENT_MARKERS) or _first_secret_key_marker(lowered) is not None


def check_builder_environment(environment: Mapping[str, str]) -> None:
    """Refuse any credential-shaped variable: the builder never has credentials."""

    for name, value in environment.items():
        if _environment_name_looks_secret(str(name)):
            raise SecretMaterialFound(RULE_BUILDER_ENVIRONMENT_SECRET, str(name))
        if isinstance(value, str) and contains_secret_material(value):
            raise SecretMaterialFound(RULE_BUILDER_ENVIRONMENT_SECRET, str(name))


@dataclass(frozen=True)
class BuildResult:
    image_id: str
    image_digest: str
    source_tree_hash: str
    dependency_closure: Tuple[str, ...]
    dockerfile: str
    build_argv: Tuple[str, ...]


def _sanitized_output(completed: Any) -> str:
    text = ""
    for attribute in ("stdout", "stderr"):
        chunk = getattr(completed, attribute, None)
        if isinstance(chunk, bytes):
            chunk = chunk.decode("utf-8", errors="replace")
        if isinstance(chunk, str) and chunk:
            text += chunk[-2000:]
    if contains_secret_material(text):
        return "[redacted]"
    return _CONTROL_RE.sub(" ", text)[-2000:]


def build_image(
    spec: BuildSpec,
    *,
    docker_runner: DockerRunner,
    context_dir: Path,
    environment: Optional[Mapping[str, str]] = None,
    timeout_seconds: int = 900,
    docker_binary: str = "docker",
) -> BuildResult:
    """Build the miner image offline and return its immutable digest.

    ``docker_runner(argv, timeout_seconds) -> CompletedProcess`` is injected
    so tests never invoke Docker. The build argv always carries
    ``--network=none`` and ``--pull=false``; the only commands run are
    ``docker build`` and ``docker image inspect``. The digest is the
    registry ``RepoDigests`` entry when present and the content-addressed
    image ``Id`` otherwise.
    """

    check_builder_environment(dict(environment or {}))
    context = Path(context_dir)
    write_build_context(spec, context)
    iidfile = context / "image.iid"
    argv = docker_build_argv(spec, context, iidfile, docker_binary=docker_binary)
    completed = docker_runner(argv, timeout_seconds)
    if getattr(completed, "returncode", None) != 0:
        raise PackageRejected(RULE_BUILD_FAILED, "docker build failed: %s" % _sanitized_output(completed))
    try:
        image_id = iidfile.read_text(encoding="utf-8").strip()
    except OSError:
        raise PackageRejected(RULE_BUILD_IMAGE_ID_INVALID, "docker build produced no image id") from None
    if not IMAGE_DIGEST_RE.match(image_id):
        raise PackageRejected(RULE_BUILD_IMAGE_ID_INVALID, "docker build produced an invalid image id")
    inspect_argv = (docker_binary, "image", "inspect", image_id, "--format", "{{json .}}")
    inspected = docker_runner(inspect_argv, 60)
    if getattr(inspected, "returncode", None) != 0:
        raise PackageRejected(RULE_BUILD_INSPECT_MISMATCH, "docker image inspect failed: %s" % _sanitized_output(inspected))
    stdout = getattr(inspected, "stdout", "")
    if isinstance(stdout, bytes):
        stdout = stdout.decode("utf-8", errors="replace")
    try:
        parsed = json.loads(stdout)
    except (TypeError, ValueError):
        raise PackageRejected(RULE_BUILD_INSPECT_MISMATCH, "docker image inspect returned invalid JSON") from None
    if isinstance(parsed, list):
        parsed = parsed[0] if len(parsed) == 1 else None
    if not isinstance(parsed, Mapping) or parsed.get("Id") != image_id:
        raise PackageRejected(RULE_BUILD_INSPECT_MISMATCH, "docker image inspect does not describe the built image")
    repo_digests = parsed.get("RepoDigests") or []
    image_digest = image_id
    if repo_digests:
        candidate = str(repo_digests[0])
        digest_part = candidate.rsplit("@", 1)[-1]
        if not IMAGE_DIGEST_RE.match(digest_part):
            raise PackageRejected(RULE_BUILD_INSPECT_MISMATCH, "docker image inspect returned an invalid repo digest")
        image_digest = candidate
    return BuildResult(
        image_id=image_id,
        image_digest=image_digest,
        source_tree_hash=source_tree_hash(spec.source_files),
        dependency_closure=resolve_dependency_closure(spec.dependency_lock),
        dockerfile=render_dockerfile(spec),
        build_argv=tuple(argv),
    )


# ---------------------------------------------------------------------------
# Screening pass (section 6.3 step 6)
# ---------------------------------------------------------------------------

SCREENING_SYNTHETIC_ICP_COUNT = 3
RULE_SCREENING_NO_COMPANIES = "screening.no_companies_with_providers"
RULE_SCREENING_INVALID_OUTPUT = "screening.invalid_company_output"
RULE_SCREENING_WITHOUT_PROVIDERS = "screening.companies_without_providers"
RULE_SCREENING_IDENTICAL = "screening.identical_companies_across_icps"
RULE_SCREENING_MODEL_ERROR = "screening.model_error"
SCREENING_RULE_IDS = (
    RULE_SCREENING_NO_COMPANIES,
    RULE_SCREENING_INVALID_OUTPUT,
    RULE_SCREENING_WITHOUT_PROVIDERS,
    RULE_SCREENING_IDENTICAL,
    RULE_SCREENING_MODEL_ERROR,
)
RunModel = Callable[[Mapping[str, Any], bool], Any]


@dataclass(frozen=True)
class ScreeningResult:
    accepted: bool
    rule_id: Optional[str]
    detail: str
    fixture_company_count: int
    synthetic_company_counts: Tuple[int, ...]


def _reject(rule_id: str, detail: str, fixture_count: int, synthetic_counts: Sequence[int]) -> ScreeningResult:
    return ScreeningResult(False, rule_id, detail, fixture_count, tuple(synthetic_counts))


def _website_set(companies: Sequence[Any]) -> FrozenSet[str]:
    websites: Set[str] = set()
    for item in companies:
        if isinstance(item, CompanyOutput):
            websites.add(item.company_website.strip().lower())
        elif isinstance(item, Mapping) and isinstance(item.get("company_website"), str):
            websites.add(item["company_website"].strip().lower())
    return frozenset(websites)


def screen_model(
    run_model: RunModel,
    *,
    fixture_icp: Mapping[str, Any],
    synthetic_icps: Sequence[Mapping[str, Any]],
) -> ScreeningResult:
    """Run the screening pass; the caller executes the model through ``run_model``.

    ``run_model(icp, providers_enabled) -> list`` returns the companies the
    model wrote. The fixture ICP runs with providers enabled and must yield
    at least one row that validates as ``CompanyOutput``. Each synthetic ICP
    runs with every provider call refused and must yield no rows. When a
    synthetic run does yield rows, an identical website set to any other ICP
    is reported as ``screening.identical_companies_across_icps`` (hardcoded
    answers) and any other non-empty result as
    ``screening.companies_without_providers``. A model exception is a
    rejection, never a silent pass.
    """

    if len(synthetic_icps) != SCREENING_SYNTHETIC_ICP_COUNT:
        raise ArenaContractError("screening requires exactly %d synthetic ICPs" % SCREENING_SYNTHETIC_ICP_COUNT)
    icp_hashes = [document_hash(fixture_icp)] + [document_hash(icp) for icp in synthetic_icps]
    if len(set(icp_hashes)) != len(icp_hashes):
        raise ArenaContractError("screening ICPs must be distinct")

    try:
        raw_fixture = run_model(fixture_icp, True)
    except Exception as exc:  # noqa: BLE001 - converted into an explicit rejection
        return _reject(RULE_SCREENING_MODEL_ERROR, "%s during the fixture run" % type(exc).__name__, 0, ())
    if not isinstance(raw_fixture, (list, tuple)):
        return _reject(RULE_SCREENING_INVALID_OUTPUT, "fixture output is not a list", 0, ())
    fixture_companies: List[CompanyOutput] = []
    for index, item in enumerate(raw_fixture):
        try:
            fixture_companies.append(item if isinstance(item, CompanyOutput) else CompanyOutput.model_validate(item))
        except (ValidationError, TypeError, ValueError):
            return _reject(RULE_SCREENING_INVALID_OUTPUT, "fixture company %d failed CompanyOutput validation" % index, len(raw_fixture), ())
    if not fixture_companies:
        return _reject(RULE_SCREENING_NO_COMPANIES, "fixture ICP produced no companies with providers enabled", 0, ())

    synthetic_outputs: List[Sequence[Any]] = []
    synthetic_counts: List[int] = []
    for position, icp in enumerate(synthetic_icps):
        try:
            raw = run_model(icp, False)
        except Exception as exc:  # noqa: BLE001 - converted into an explicit rejection
            return _reject(RULE_SCREENING_MODEL_ERROR, "%s during synthetic run %d" % (type(exc).__name__, position), len(fixture_companies), synthetic_counts)
        if not isinstance(raw, (list, tuple)):
            return _reject(RULE_SCREENING_INVALID_OUTPUT, "synthetic output %d is not a list" % position, len(fixture_companies), synthetic_counts)
        synthetic_outputs.append(raw)
        synthetic_counts.append(len(raw))

    website_sets = [_website_set(fixture_companies)] + [_website_set(raw) for raw in synthetic_outputs]
    for left in range(len(website_sets)):
        for right in range(left + 1, len(website_sets)):
            if website_sets[left] and website_sets[left] == website_sets[right]:
                return _reject(
                    RULE_SCREENING_IDENTICAL,
                    "ICPs %d and %d produced identical company website sets" % (left, right),
                    len(fixture_companies),
                    synthetic_counts,
                )
    for position, count in enumerate(synthetic_counts):
        if count:
            return _reject(
                RULE_SCREENING_WITHOUT_PROVIDERS,
                "synthetic ICP %d produced %d companies with providers refused" % (position, count),
                len(fixture_companies),
                synthetic_counts,
            )
    return ScreeningResult(True, None, "accepted", len(fixture_companies), tuple(synthetic_counts))


__all__ = [
    "APPROVED_DEPENDENCIES",
    "APPROVED_DEPENDENCY_REQUIRES",
    "BUILDER_FORBIDDEN_ENVIRONMENT_MARKERS",
    "BuildResult",
    "BuildSpec",
    "DEFAULT_PACKAGE_RULES",
    "ENVIRONMENT_FILE_PATTERNS",
    "MANIFEST_FIELDS",
    "MANIFEST_PATH",
    "PACKAGE_RULE_IDS",
    "PackageInspection",
    "PackageRejected",
    "PackageRules",
    "SCREENING_RULE_IDS",
    "SECRET_KEY_MARKERS",
    "SECRET_MARKERS",
    "SECRET_TOKEN_KEY_MARKERS",
    "SECRET_VALUE_PATTERNS",
    "ScreeningResult",
    "SecretMaterialFound",
    "accept_package",
    "approved_dependency_set_hash",
    "build_image",
    "check_builder_environment",
    "contains_secret_material",
    "docker_build_argv",
    "inspect_package",
    "is_environment_file",
    "normalize_member_path",
    "render_dockerfile",
    "resolve_dependency_closure",
    "scan_document_raise",
    "scan_source_archive_raise",
    "screen_model",
    "source_tree_hash",
    "validate_dependency_lock",
    "write_build_context",
]
