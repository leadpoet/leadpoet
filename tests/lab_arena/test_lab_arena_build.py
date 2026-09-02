"""Tests for lab_arena.build (labarena.md sections 6.2, 6.3, 12.4, 18.5).

Packages are built in memory with ``tarfile``; Docker is replaced by a fake
runner that records argv; the model under screening is an injected callable.
"""

from __future__ import annotations

import io
import json
import subprocess
import tarfile
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import pytest

from lab_arena import build
from lab_arena.contracts import SHA256_RE, SUBMISSION_PACKAGE_SCHEMA_VERSION, ArenaContractError, document_hash, hash_bytes

ENTRY_SOURCE = b"import json\n\n\ndef main():\n    print(json.dumps([]))\n\n\nif __name__ == '__main__':\n    main()\n"
HELPER_SOURCE = b"def helper():\n    return 1\n"
IMAGE_DIGEST = "sha256:" + "ab" * 32


def base_manifest(**overrides: Any) -> Dict[str, Any]:
    manifest: Dict[str, Any] = {
        "schema_version": SUBMISSION_PACKAGE_SCHEMA_VERSION,
        "entry_point": "model/main.py",
        "dependency_lock": ["requests==2.32.5"],
        "consent": {"source_publication": True, "public_rerun": True},
    }
    manifest.update(overrides)
    return manifest


def manifest_bytes(manifest: Optional[Dict[str, Any]] = None) -> bytes:
    return json.dumps(base_manifest() if manifest is None else manifest).encode("utf-8")


def member(
    name: str,
    data: bytes = b"",
    *,
    kind: Optional[bytes] = None,
    linkname: str = "",
    mode: int = 0o644,
    size: Optional[int] = None,
    fileobj: Any = None,
) -> Tuple[tarfile.TarInfo, Any]:
    info = tarfile.TarInfo(name)
    info.mode = mode
    if kind is not None:
        info.type = kind
    if linkname:
        info.linkname = linkname
    info.size = len(data) if size is None else size
    if fileobj is not None:
        return info, fileobj
    return info, (io.BytesIO(data) if info.size and kind in (None, tarfile.REGTYPE) else None)


def make_archive(members: Sequence[Tuple[tarfile.TarInfo, Any]], fmt: int = tarfile.GNU_FORMAT) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz", format=fmt) as tar:
        for info, fileobj in members:
            tar.addfile(info, fileobj)
    return buffer.getvalue()


def valid_members(manifest: Optional[Dict[str, Any]] = None) -> List[Tuple[tarfile.TarInfo, Any]]:
    return [
        member("manifest.json", manifest_bytes(manifest)),
        member("model/main.py", ENTRY_SOURCE),
        member("model/helper.py", HELPER_SOURCE),
    ]


class _Zeros(io.RawIOBase):
    """Lazily yields ``size`` zero bytes without materializing them."""

    def __init__(self, size: int) -> None:
        self.remaining = size

    def readable(self) -> bool:
        return True

    def readinto(self, buffer: Any) -> int:
        count = min(len(buffer), self.remaining)
        buffer[:count] = bytes(count)
        self.remaining -= count
        return count


def expect_rejection(archive: bytes, rule_id: str, rules: build.PackageRules = build.DEFAULT_PACKAGE_RULES) -> build.PackageRejected:
    with pytest.raises(build.PackageRejected) as info:
        build.inspect_package(archive, rules)
    assert info.value.rule_id == rule_id
    assert rule_id in build.PACKAGE_RULE_IDS
    assert isinstance(info.value, ArenaContractError)
    return info.value


# ---------------------------------------------------------------------------
# Package inspection: the accept path
# ---------------------------------------------------------------------------


def test_valid_package_inspection() -> None:
    manifest = base_manifest(files=["model/main.py", "model/helper.py"], dependency_lock=["Requests==2.32.5", "python_dateutil==2.9.0.post0"])
    members = valid_members(manifest)
    members[1][0].mode = 0o755
    archive = make_archive(members)
    inspection = build.inspect_package(archive)
    assert inspection.package_hash == hash_bytes(archive)
    assert SHA256_RE.match(inspection.source_tree_hash)
    assert inspection.entry_point == "model/main.py"
    assert inspection.dependency_lock == ("requests==2.32.5", "python-dateutil==2.9.0.post0")
    assert inspection.manifest["dependency_lock"] == ["requests==2.32.5", "python-dateutil==2.9.0.post0"]
    assert set(inspection.files) == {"manifest.json", "model/main.py", "model/helper.py"}
    assert inspection.files["model/main.py"] == ENTRY_SOURCE
    assert inspection.executable_paths == frozenset({"model/main.py"})
    assert inspection.file_count == 3
    assert inspection.package_bytes == len(archive)
    assert inspection.uncompressed_bytes > sum(len(v) for v in inspection.files.values())
    expected_tree = document_hash(
        [
            {"path": path, "sha256": hash_bytes(inspection.files[path]), "size": len(inspection.files[path]), "mode": "file"}
            for path in sorted(inspection.files)
        ]
    )
    assert inspection.source_tree_hash == expected_tree
    assert build.accept_package(archive).source_tree_hash == expected_tree


def test_source_tree_hash_is_independent_of_member_order() -> None:
    forward = make_archive(valid_members())
    reversed_members = list(reversed(valid_members()))
    backward = make_archive(reversed_members)
    with_dirs = make_archive([member("model", kind=tarfile.DIRTYPE), member("./", kind=tarfile.DIRTYPE)] + valid_members())
    assert forward != backward
    hashes = {build.inspect_package(archive).source_tree_hash for archive in (forward, backward, with_dirs)}
    assert len(hashes) == 1
    assert build.inspect_package(forward).package_hash != build.inspect_package(backward).package_hash


def test_dot_prefixed_paths_normalize() -> None:
    archive = make_archive([member("./manifest.json", manifest_bytes()), member("./model/main.py", ENTRY_SOURCE)])
    assert set(build.inspect_package(archive).files) == {"manifest.json", "model/main.py"}


# ---------------------------------------------------------------------------
# Package inspection: every published rejection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "extra,rule_id",
    [
        (member("../evil.py", b"x"), build.RULE_PARENT_PATH),
        (member("model/../../evil.py", b"x"), build.RULE_PARENT_PATH),
        (member("/etc/passwd", b"x"), build.RULE_ABSOLUTE_PATH),
        (member("C:/windows/x.py", b"x"), build.RULE_ABSOLUTE_PATH),
        (member("model\\evil.py", b"x"), build.RULE_INVALID_PATH),
        (member("model//evil.py", b"x"), build.RULE_INVALID_PATH),
        (member("model/evil name.py", b"x"), build.RULE_INVALID_PATH),
        (member("model/link.py", kind=tarfile.SYMTYPE, linkname="/etc/passwd"), build.RULE_LINK),
        (member("model/hard.py", kind=tarfile.LNKTYPE, linkname="model/main.py"), build.RULE_LINK),
        (member("model/fifo", kind=tarfile.FIFOTYPE), build.RULE_SPECIAL_FILE),
        (member("model/chr", kind=tarfile.CHRTYPE), build.RULE_SPECIAL_FILE),
        (member("model/blk", kind=tarfile.BLKTYPE), build.RULE_SPECIAL_FILE),
        (member("model/main.py", ENTRY_SOURCE), build.RULE_DUPLICATE_PATH),
        (member("model/MAIN.py", b"x"), build.RULE_CASE_COLLISION),
        (member("Model", kind=tarfile.DIRTYPE), build.RULE_CASE_COLLISION),
        (member("model/sparse.bin", kind=tarfile.GNUTYPE_SPARSE), build.RULE_SPARSE_FILE),
        (member("requirements.lock", b"requests==2.32.5\n"), build.RULE_RESERVED_PATH),
    ],
)
def test_member_rejections(extra: Tuple[tarfile.TarInfo, Any], rule_id: str) -> None:
    archive = make_archive(valid_members() + [extra])
    rejection = expect_rejection(archive, rule_id)
    assert str(rejection).startswith(rule_id)


def test_case_collision_on_directory_and_file() -> None:
    archive = make_archive(valid_members() + [member("MODEL/other.py", b"x")])
    expect_rejection(archive, build.RULE_CASE_COLLISION)


def test_too_many_files() -> None:
    rules = build.PackageRules(max_files=3)
    assert build.inspect_package(make_archive(valid_members()), rules).file_count == 3
    archive = make_archive(valid_members() + [member("model/extra.py", b"x")])
    expect_rejection(archive, build.RULE_TOO_MANY_FILES, rules)


def test_file_too_large_by_declared_size() -> None:
    limit = max(len(manifest_bytes()), len(ENTRY_SOURCE), len(HELPER_SOURCE))
    rules = build.PackageRules(max_file_bytes=limit)
    build.inspect_package(make_archive(valid_members()), rules)
    archive = make_archive(valid_members() + [member("model/big.py", b"#" * (limit + 1))])
    expect_rejection(archive, build.RULE_FILE_TOO_LARGE, rules)


def test_path_too_long() -> None:
    rules = build.PackageRules(max_path_length=20)
    archive = make_archive(valid_members() + [member("model/" + "a" * 30 + ".py", b"x")])
    expect_rejection(archive, build.RULE_PATH_TOO_LONG, rules)


def test_compression_bomb_200_mib_zero_member_with_default_rules() -> None:
    size = 200 * 1024 * 1024
    bomb = member("model/zeros.bin", size=size, fileobj=io.BufferedReader(_Zeros(size)))
    archive = make_archive(valid_members() + [bomb])
    assert len(archive) < build.DEFAULT_PACKAGE_RULES.max_package_bytes
    # 200 MiB of zeros gzips to roughly 200 KiB, so the ratio bound (200x)
    # trips before the 100 MiB total cap; either detail is the same rule.
    rejection = expect_rejection(archive, build.RULE_COMPRESSION_BOMB)
    assert "exceeds" in rejection.detail


def test_compression_ratio_bomb_with_small_rules() -> None:
    rules = build.PackageRules(max_compression_ratio=10, max_file_bytes=1024 * 1024, max_total_uncompressed_bytes=8 * 1024 * 1024)
    archive = make_archive(valid_members() + [member("model/zeros.bin", bytes(256 * 1024))])
    rejection = expect_rejection(archive, build.RULE_COMPRESSION_BOMB, rules)
    assert "compression ratio exceeds 10" in rejection.detail


def test_total_uncompressed_cap_with_small_rules() -> None:
    rules = build.PackageRules(max_total_uncompressed_bytes=64 * 1024, max_file_bytes=64 * 1024, max_compression_ratio=1_000_000)
    archive = make_archive(valid_members() + [member("model/zeros.bin", bytes(60 * 1024)), member("model/zeros2.bin", bytes(60 * 1024))])
    expect_rejection(archive, build.RULE_COMPRESSION_BOMB, rules)


@pytest.mark.parametrize(
    "archive,rule_id,rules",
    [
        (b"", build.RULE_ARCHIVE_INVALID, build.DEFAULT_PACKAGE_RULES),
        (b"not a gzip stream at all", build.RULE_ARCHIVE_INVALID, build.DEFAULT_PACKAGE_RULES),
        (make_archive(valid_members())[:-40], build.RULE_ARCHIVE_INVALID, build.DEFAULT_PACKAGE_RULES),
        (make_archive(valid_members()), build.RULE_PACKAGE_TOO_LARGE, build.PackageRules(max_package_bytes=10)),
    ],
)
def test_archive_level_rejections(archive: bytes, rule_id: str, rules: build.PackageRules) -> None:
    expect_rejection(archive, rule_id, rules)


def test_plain_tar_without_gzip_is_rejected() -> None:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:") as tar:
        for info, fileobj in valid_members():
            tar.addfile(info, fileobj)
    expect_rejection(buffer.getvalue(), build.RULE_ARCHIVE_INVALID)


def test_gzip_of_garbage_is_rejected() -> None:
    import gzip

    expect_rejection(gzip.compress(b"\x01" * 4096), build.RULE_ARCHIVE_INVALID)


def test_non_bytes_archive_is_rejected() -> None:
    with pytest.raises(build.PackageRejected) as info:
        build.inspect_package("text")  # type: ignore[arg-type]
    assert info.value.rule_id == build.RULE_ARCHIVE_INVALID


# ---------------------------------------------------------------------------
# Manifest rules
# ---------------------------------------------------------------------------


def test_manifest_missing() -> None:
    expect_rejection(make_archive(valid_members()[1:]), build.RULE_MANIFEST_MISSING)


@pytest.mark.parametrize(
    "raw,rule_id",
    [
        (b"{not json", build.RULE_MANIFEST_INVALID),
        (b"[1, 2]", build.RULE_MANIFEST_INVALID),
        (b"\xff\xfe", build.RULE_MANIFEST_INVALID),
        (json.dumps(base_manifest(schema_version="leadpoet.lab_arena.submission_package.v0")).encode(), build.RULE_MANIFEST_INVALID),
        (json.dumps({**base_manifest(), "extra": 1}).encode(), build.RULE_MANIFEST_INVALID),
        (json.dumps(base_manifest(entry_point="model/main.txt")).encode(), build.RULE_MANIFEST_INVALID),
        (json.dumps(base_manifest(entry_point="../main.py")).encode(), build.RULE_MANIFEST_INVALID),
        (json.dumps(base_manifest(entry_point="")).encode(), build.RULE_MANIFEST_INVALID),
        (json.dumps(base_manifest(files=["model/main.py"])).encode(), build.RULE_MANIFEST_INVALID),
        (json.dumps(base_manifest(dependency_lock=["requests"])).encode(), build.RULE_MANIFEST_INVALID),
        (json.dumps(base_manifest(dependency_lock=["requests==2.32.5", "REQUESTS==2.32.5"])).encode(), build.RULE_MANIFEST_INVALID),
        (json.dumps(base_manifest(dependency_lock="requests==2.32.5")).encode(), build.RULE_MANIFEST_INVALID),
        (json.dumps(base_manifest(entry_point="model/missing.py")).encode(), build.RULE_ENTRY_POINT_MISSING),
        (json.dumps(base_manifest(entry_point="manifest.json")).encode(), build.RULE_MANIFEST_INVALID),
        (json.dumps(base_manifest(dependency_lock=["requests==1.0.0"])).encode(), build.RULE_DEPENDENCY_NOT_APPROVED),
        (json.dumps(base_manifest(dependency_lock=["leftpad==1.0.0"])).encode(), build.RULE_DEPENDENCY_NOT_APPROVED),
        (json.dumps(base_manifest(consent={"source_publication": True, "public_rerun": False})).encode(), build.RULE_CONSENT_MISSING),
        (json.dumps(base_manifest(consent={"source_publication": True})).encode(), build.RULE_CONSENT_MISSING),
        (json.dumps(base_manifest(consent="yes")).encode(), build.RULE_CONSENT_MISSING),
        (json.dumps({k: v for k, v in base_manifest().items() if k != "consent"}).encode(), build.RULE_CONSENT_MISSING),
        (json.dumps(base_manifest(consent={"source_publication": True, "public_rerun": True, "marketing": True})).encode(), build.RULE_MANIFEST_INVALID),
    ],
)
def test_manifest_rejections(raw: bytes, rule_id: str) -> None:
    members = valid_members()
    members[0] = member("manifest.json", raw)
    expect_rejection(make_archive(members), rule_id)


def test_manifest_too_large_is_invalid() -> None:
    import os

    members = valid_members()
    # Incompressible padding keeps the archive under the ratio bound so the
    # manifest size rule, not the bomb rule, is what fires.
    members[0] = member("manifest.json", os.urandom(300_000))
    rejection = expect_rejection(make_archive(members), build.RULE_MANIFEST_INVALID)
    assert "exceeds" in rejection.detail


# ---------------------------------------------------------------------------
# Approved dependency set
# ---------------------------------------------------------------------------


def test_approved_dependency_set_is_pinned_and_hashed() -> None:
    assert all("==" in pin for pin in build.APPROVED_DEPENDENCIES)
    assert len({build.normalize_distribution_name(pin.split("==")[0]) for pin in build.APPROVED_DEPENDENCIES}) == len(build.APPROVED_DEPENDENCIES)
    for name in ("requests", "httpx", "aiohttp", "pydantic", "beautifulsoup4", "lxml", "python-dateutil", "numpy"):
        assert name in build.APPROVED_DEPENDENCY_INDEX
    digest = build.approved_dependency_set_hash()
    assert SHA256_RE.match(digest)
    assert digest == build.approved_dependency_set_hash()
    for requires in build.APPROVED_DEPENDENCY_REQUIRES.values():
        assert all(item in build.APPROVED_DEPENDENCY_INDEX for item in requires)


def test_dependency_closure_expands_transitive_pins() -> None:
    closure = build.resolve_dependency_closure(["httpx==0.28.1"])
    names = [pin.split("==")[0] for pin in closure]
    assert names == sorted(names)
    assert set(names) == {"anyio", "certifi", "h11", "httpcore", "httpx", "idna", "typing-extensions"}
    assert build.resolve_dependency_closure([]) == ()
    with pytest.raises(build.PackageRejected) as info:
        build.resolve_dependency_closure(["httpx==0.1.0"])
    assert info.value.rule_id == build.RULE_DEPENDENCY_NOT_APPROVED


# ---------------------------------------------------------------------------
# Secret scan
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text,rule_id",
    [
        ("token = 'sk-or-v1-" + "a" * 40 + "'", "secret.marker.sk_or"),
        ("OPENROUTER_API_KEY = os.environ['X']", "secret.marker.openrouter-api-key"),
        ("raw_openrouter_key = value", "secret.marker.raw-openrouter-key"),
        ("raw_secret = 1", "secret.marker.raw-secret"),
        ("role = 'service_role'", "secret.marker.service-role"),
        ("PRIVATE_REPO = 'git@github.com:x/y'", "secret.marker.private-repo"),
        ("EXA_API_KEY = ''", "secret.marker.exa-api-key"),
        ("ScrapingDog_API_Key = ''", "secret.marker.scrapingdog-api-key"),
        ("AWS_SECRET_ACCESS_KEY=abc", "secret.marker.aws-secret-access-key"),
        ("-----BEGIN RSA PRIVATE KEY-----", "secret.marker.begin"),
        ("headers = {'Authorization: Bearer x'}", "secret.marker.authorization-bearer"),
        ("url = 'https://api.example.com/?api_key=abc'", "secret.marker.api-key"),
        ("SB_SECRET_abc", "secret.marker.sb-secret"),
        ("x = 'sk-" + "A1b2C3d4E5f6G7h8I9j0k1" + "'", "secret.pattern.generic_sk_key"),
        ('"""Send the header ``Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9`` on each call."""', "secret.pattern.bearer_token"),
        ("bearer AbCdEfGhIjKlMnOpQrStUv==", "secret.pattern.bearer_token"),
        ("AWS_KEY_ID = 'AKIAIOSFODNN7EXAMPLE'", "secret.pattern.aws_access_key_id"),
        ("db = 'postgres://user:hunter2@db.example.com/x'", "secret.pattern.url_userinfo"),
        ("db = 'https://user:hunter2@example.com'", "secret.pattern.url_userinfo"),
        ("exa_key = '3f2504e0-4f89-11d3-9a0c-0305e82c3301'", "secret.pattern.exa_key"),
        ("SCRAPINGDOG_KEY = '0123456789abcdef0123456789abcdef'", "secret.pattern.scrapingdog_key"),
    ],
)
def test_source_scan_positives(text: str, rule_id: str) -> None:
    with pytest.raises(build.SecretMaterialFound) as info:
        build.scan_source_archive_raise({"model/main.py": text.encode("utf-8")})
    assert info.value.rule_id == rule_id
    assert info.value.location.startswith("model/main.py")
    for candidate in ("hunter2", "AKIAIOSFODNN7EXAMPLE", "3f2504e0", "0123456789abcdef", "eyJhbGci", "A1b2C3d4"):
        assert candidate not in str(info.value)


@pytest.mark.parametrize(
    "text",
    [
        "image_digest = 'sha256:" + "0123456789abcdef" * 4 + "'",
        "image_digest = 'sha256:" + "0" * 64 + "'",
        "scrapingdog_image = 'sha256:" + "a" * 64 + "'",
        "exa_image = 'sha256:" + "a" * 64 + "'",
        "session_id = '3f2504e0-4f89-11d3-9a0c-0305e82c3301'",
        "cache_key = '0123456789abcdef0123456789abcdef'",
        "url = 'https://api.example.com:443/path'",
        "url = 'https://example.com/a:b@c'",
        "desk-abcdefghijklmnopqrstuvwxyz",
        "def fetch(api_key_missing=None):\n    return api_key_missing\n",
        "task-runner-0123456789abcdefghij",
        "authorization = 'basic'",
        "ENTRY = 'model/main.py'",
        ENTRY_SOURCE.decode("utf-8"),
    ],
)
def test_source_scan_negatives(text: str) -> None:
    build.scan_source_archive_raise({"model/main.py": text.encode("utf-8")})
    assert build.contains_secret_material(text) is False


def test_hinted_patterns_apply_through_path_and_line() -> None:
    uuid = "3f2504e0-4f89-11d3-9a0c-0305e82c3301"
    build.scan_source_archive_raise({"model/ids.py": ("VALUE = '%s'\n" % uuid).encode()})
    with pytest.raises(build.SecretMaterialFound) as info:
        build.scan_source_archive_raise({"model/exa_config.py": ("VALUE = '%s'\n" % uuid).encode()})
    assert info.value.rule_id == "secret.pattern.exa_key"
    with pytest.raises(build.SecretMaterialFound) as info:
        build.scan_source_archive_raise({"model/ids.py": ("other = 1\nexa = '%s'\n" % uuid).encode()})
    assert info.value.rule_id == "secret.pattern.exa_key"
    assert info.value.location == "model/ids.py:2"
    hexkey = "0123456789abcdef0123456789abcdef"
    build.scan_source_archive_raise({"model/ids.py": ("digest = '%s'\n" % hexkey).encode()})
    with pytest.raises(build.SecretMaterialFound) as info:
        build.scan_source_archive_raise({"model/ids.py": ("scrapingdog_token_value = '%s'\n" % hexkey).encode()})
    assert info.value.rule_id == "secret.pattern.scrapingdog_key"


@pytest.mark.parametrize(
    "path",
    [".env", "model/.env", "config/prod.env", ".env.local", "model/.envrc", "credentials.json", "keys/server.pem", "keys/id_rsa", "keys/id_rsa.pub", ".netrc", "KEYS/SERVER.PEM"],
)
def test_environment_files_rejected_by_path(path: str) -> None:
    assert build.is_environment_file(path)
    with pytest.raises(build.SecretMaterialFound) as info:
        build.scan_source_archive_raise({path: b"harmless", "model/main.py": ENTRY_SOURCE})
    assert info.value.rule_id == build.RULE_SECRET_ENVIRONMENT_FILE
    assert info.value.location == path


def test_binary_member_rules() -> None:
    binary = b"\x00\x01\x02PNG\x00"
    with pytest.raises(build.SecretMaterialFound) as info:
        build.scan_source_archive_raise({"model/blob.bin": binary})
    assert info.value.rule_id == build.RULE_SECRET_BINARY_MEMBER
    build.scan_source_archive_raise({"data/blob.bin": binary})
    with pytest.raises(build.SecretMaterialFound) as info:
        build.scan_source_archive_raise({"data/blob.bin": binary}, executable_paths={"data/blob.bin"})
    assert info.value.rule_id == build.RULE_SECRET_BINARY_MEMBER
    with pytest.raises(build.SecretMaterialFound) as info:
        build.scan_source_archive_raise({"data/blob.bin": b"\x00\x00sk-or-v1-" + b"a" * 40})
    assert info.value.rule_id == "secret.marker.sk_or"


def test_secret_in_file_name_is_rejected() -> None:
    with pytest.raises(build.SecretMaterialFound) as info:
        build.scan_source_archive_raise({"model/sk-or-v1-notes.py": b"x = 1\n"})
    assert info.value.rule_id == "secret.marker.sk_or"


def test_document_scan_key_names_and_values() -> None:
    clean = {
        "image_digest": "sha256:" + "f" * 64,
        "release": {"repository_commit": "0" * 40, "base_image_digest": "sha256:" + "e" * 64},
        "runner_public_key_hash": "sha256:" + "1" * 64,
        "items": [{"provider": "exa", "request_hash": "sha256:" + "2" * 64}],
        "cost_total_microusd": 1234,
        "session_id": "3f2504e0-4f89-11d3-9a0c-0305e82c3301",
    }
    build.scan_document_raise(clean)
    assert build.contains_secret_material(clean) is False
    with pytest.raises(build.SecretMaterialFound) as info:
        build.scan_document_raise({"config": {"exa_api_key": "x"}})
    assert info.value.rule_id == "secret.key_name.api-key"
    assert info.value.location == "$.config.exa_api_key"
    for key in ("credentials", "proxy_url", "refresh_token", "private_key_pem", "db_password"):
        assert build.contains_secret_material({key: "value"}) is True
    with pytest.raises(build.SecretMaterialFound) as info:
        build.scan_document_raise({"items": [{"exa": "3f2504e0-4f89-11d3-9a0c-0305e82c3301"}]})
    assert info.value.rule_id == "secret.pattern.exa_key"
    assert info.value.location == "$.items[0].exa"
    assert build.contains_secret_material({"notes": ["fine", "sk-or-v1-" + "b" * 40]}) is True
    assert build.contains_secret_material("Authorization: Bearer abc") is True
    assert build.contains_secret_material(["sha256:" + "0" * 64, 5, None, True]) is False
    with pytest.raises(build.SecretMaterialFound) as info:
        build.scan_document_raise({"scrapingdog": {"value": "0123456789abcdef0123456789abcdef"}})
    assert info.value.rule_id == "secret.pattern.scrapingdog_key"


def test_rule_ids_never_retrip_the_scanner() -> None:
    labels = [build._secret_marker_label(marker) for marker in build.SECRET_MARKERS + build.SECRET_KEY_MARKERS + build.SECRET_TOKEN_KEY_MARKERS]
    document = {"rejection_rule": ["secret.marker." + label for label in labels] + ["secret.pattern." + p.name for p in build.SECRET_VALUE_PATTERNS]}
    build.scan_document_raise(document)


# ---------------------------------------------------------------------------
# Offline image build
# ---------------------------------------------------------------------------


def make_spec(tmp_path: Path, **overrides: Any) -> build.BuildSpec:
    wheelhouse = tmp_path / "wheelhouse"
    wheelhouse.mkdir(exist_ok=True)
    (wheelhouse / "requests-2.32.5-py3-none-any.whl").write_bytes(b"PK\x03\x04fake")
    kwargs: Dict[str, Any] = dict(
        base_image="leadpoet/lab-arena-base",
        base_image_digest=IMAGE_DIGEST,
        wheelhouse_dir=wheelhouse,
        entry_point="model/main.py",
        source_files={"manifest.json": manifest_bytes(), "model/main.py": ENTRY_SOURCE, "model/helper.py": HELPER_SOURCE},
        dependency_lock=("requests==2.32.5",),
    )
    kwargs.update(overrides)
    return build.BuildSpec(**kwargs)


def test_render_dockerfile_only_copies_and_installs_wheels(tmp_path: Path) -> None:
    text = build.render_dockerfile(make_spec(tmp_path))
    lines = text.strip().split("\n")
    assert lines == [
        "FROM leadpoet/lab-arena-base@" + IMAGE_DIGEST,
        "COPY source/ /model/",
        "COPY wheelhouse/ /wheelhouse/",
        "RUN pip install --no-index --no-deps --only-binary=:all: --find-links /wheelhouse -r /model/requirements.lock",
        "USER 65534:65534",
        "ENV PYTHONDONTWRITEBYTECODE=1 TZ=UTC",
        'ENTRYPOINT ["python3", "/model/model/main.py"]',
    ]
    run_lines = [line for line in lines if line.startswith("RUN")]
    assert len(run_lines) == 1
    assert "python" not in run_lines[0] and "sh -c" not in run_lines[0] and "/model/main" not in run_lines[0]
    assert "--no-deps" in run_lines[0] and "--only-binary=:all:" in run_lines[0] and "--no-index" in run_lines[0]
    assert "ADD " not in text and "ARG " not in text


@pytest.mark.parametrize(
    "overrides,message",
    [
        ({"base_image_digest": "sha256:abc"}, "sha256:<64 hex>"),
        ({"base_image": "leadpoet/base@sha256:" + "a" * 64}, "plain image repository"),
        ({"base_image": "Leadpoet/Base"}, "plain image repository"),
        ({"entry_point": "model/missing.py"}, "entry_point"),
        ({"entry_point": "./model/main.py"}, "entry_point"),
        ({"entry_point": "model/helper.txt"}, "entry_point"),
        ({"dependency_lock": ("leftpad==1.0.0",)}, "not in the approved"),
        ({"source_files": {}}, "non-empty"),
        ({"source_files": {"../x.py": b"", "model/main.py": ENTRY_SOURCE}}, "parent"),
        ({"source_files": {"requirements.lock": b"", "model/main.py": ENTRY_SOURCE}}, "reserved"),
        ({"source_files": {"model/main.py": "text"}}, "must be bytes"),
        ({"image_tag": "Bad Tag!"}, "image_tag"),
    ],
)
def test_build_spec_validation(tmp_path: Path, overrides: Dict[str, Any], message: str) -> None:
    with pytest.raises(ArenaContractError, match=message):
        make_spec(tmp_path, **overrides)


class FakeDocker:
    def __init__(self, image_id: str, *, repo_digests: Sequence[str] = (), build_returncode: int = 0, inspect_stdout: Optional[str] = None, write_iid: bool = True) -> None:
        self.image_id = image_id
        self.repo_digests = list(repo_digests)
        self.build_returncode = build_returncode
        self.inspect_stdout = inspect_stdout
        self.write_iid = write_iid
        self.calls: List[Tuple[Tuple[str, ...], int]] = []

    def __call__(self, argv: Sequence[str], timeout: int) -> subprocess.CompletedProcess:
        argv = tuple(argv)
        self.calls.append((argv, timeout))
        if argv[1] == "build":
            if self.write_iid:
                Path(argv[argv.index("--iidfile") + 1]).write_text(self.image_id + "\n")
            return subprocess.CompletedProcess(list(argv), self.build_returncode, stdout="", stderr="error: sk-or-v1-" + "q" * 40 if self.build_returncode else "")
        if argv[1:3] == ("image", "inspect"):
            stdout = self.inspect_stdout
            if stdout is None:
                stdout = json.dumps([{"Id": self.image_id, "RepoDigests": self.repo_digests}])
            return subprocess.CompletedProcess(list(argv), 0, stdout=stdout, stderr="")
        raise AssertionError("unexpected docker command %r" % (argv,))


def test_build_image_with_fake_docker(tmp_path: Path) -> None:
    spec = make_spec(tmp_path, image_tag="lab-arena/submission:abc123")
    image_id = "sha256:" + "c" * 64
    docker = FakeDocker(image_id)
    context = tmp_path / "context"
    result = build.build_image(spec, docker_runner=docker, context_dir=context, environment={"PATH": "/usr/bin", "HOME": "/nonexistent"}, timeout_seconds=123)
    assert result.image_id == image_id
    assert result.image_digest == image_id
    assert result.source_tree_hash == build.source_tree_hash(spec.source_files)
    assert result.dependency_closure == build.resolve_dependency_closure(spec.dependency_lock)
    assert result.dockerfile == build.render_dockerfile(spec)
    assert len(docker.calls) == 2
    build_argv, timeout = docker.calls[0]
    assert timeout == 123
    assert build_argv[:2] == ("docker", "build")
    assert "--network=none" in build_argv
    assert "--pull=false" in build_argv
    assert "--iidfile" in build_argv
    assert build_argv[build_argv.index("--tag") + 1] == "lab-arena/submission:abc123"
    assert build_argv[-1] == str(context)
    assert build_argv == result.build_argv
    inspect_argv, inspect_timeout = docker.calls[1]
    assert inspect_argv[:3] == ("docker", "image", "inspect") and image_id in inspect_argv
    assert inspect_timeout == 60
    for argv, _ in docker.calls:
        assert argv[0] == "docker"
        assert not any(element.endswith(".py") or element in ("python3", "python", "sh", "bash", "-c") for element in argv)
        assert not any("/model/" in element for element in argv)
    # The context holds exactly what the Dockerfile copies.
    assert (context / "Dockerfile").read_text() == result.dockerfile
    assert (context / "source" / "model" / "main.py").read_bytes() == ENTRY_SOURCE
    assert (context / "source" / "model" / "main.py").stat().st_mode & 0o777 == 0o644
    lock = (context / "source" / "requirements.lock").read_text().splitlines()
    assert lock == list(result.dependency_closure)
    assert set(lock) == {"certifi==2025.10.5", "charset-normalizer==3.4.4", "idna==3.11", "requests==2.32.5", "urllib3==2.5.0"}
    assert sorted(p.name for p in (context / "wheelhouse").iterdir()) == ["requests-2.32.5-py3-none-any.whl"]
    assert sorted(p.name for p in context.iterdir()) == ["Dockerfile", "image.iid", "source", "wheelhouse"]


def test_build_image_prefers_repo_digest(tmp_path: Path) -> None:
    image_id = "sha256:" + "c" * 64
    repo_digest = "registry.example.com/lab-arena/submission@sha256:" + "d" * 64
    result = build.build_image(make_spec(tmp_path), docker_runner=FakeDocker(image_id, repo_digests=[repo_digest]), context_dir=tmp_path / "ctx")
    assert result.image_digest == repo_digest
    assert result.image_id == image_id


@pytest.mark.parametrize(
    "environment,name",
    [
        ({"OPENROUTER_API_KEY": "x"}, "OPENROUTER_API_KEY"),
        ({"AWS_SECRET_ACCESS_KEY": "x"}, "AWS_SECRET_ACCESS_KEY"),
        ({"AWS_ACCESS_KEY_ID": "x"}, "AWS_ACCESS_KEY_ID"),
        ({"EXA_KEY": "x"}, "EXA_KEY"),
        ({"SCRAPINGDOG_KEY": "x"}, "SCRAPINGDOG_KEY"),
        ({"SUPABASE_URL": "x"}, "SUPABASE_URL"),
        ({"GITHUB_TOKEN": "x"}, "GITHUB_TOKEN"),
        ({"DB_PASSWORD": "x"}, "DB_PASSWORD"),
        ({"PATH": "/usr/bin", "LEADPOET_KMS_KEY_ID": "alias/x"}, "LEADPOET_KMS_KEY_ID"),
        ({"HARMLESS": "sk-or-v1-" + "a" * 40}, "HARMLESS"),
        ({"HARMLESS": "Authorization: Bearer abc"}, "HARMLESS"),
    ],
)
def test_build_image_refuses_credentials_in_environment(tmp_path: Path, environment: Dict[str, str], name: str) -> None:
    docker = FakeDocker("sha256:" + "c" * 64)
    with pytest.raises(build.SecretMaterialFound) as info:
        build.build_image(make_spec(tmp_path), docker_runner=docker, context_dir=tmp_path / "ctx", environment=environment)
    assert info.value.rule_id == build.RULE_BUILDER_ENVIRONMENT_SECRET
    assert info.value.location == name
    assert "sk-or-v1-" not in str(info.value) and "Bearer" not in str(info.value)
    assert docker.calls == []
    assert not (tmp_path / "ctx").exists()


def test_build_image_failure_paths(tmp_path: Path) -> None:
    image_id = "sha256:" + "c" * 64
    with pytest.raises(build.PackageRejected) as info:
        build.build_image(make_spec(tmp_path), docker_runner=FakeDocker(image_id, build_returncode=1), context_dir=tmp_path / "a")
    assert info.value.rule_id == build.RULE_BUILD_FAILED
    assert "sk-or-v1-" not in str(info.value) and "[redacted]" in str(info.value)
    with pytest.raises(build.PackageRejected) as info:
        build.build_image(make_spec(tmp_path), docker_runner=FakeDocker(image_id, write_iid=False), context_dir=tmp_path / "b")
    assert info.value.rule_id == build.RULE_BUILD_IMAGE_ID_INVALID
    with pytest.raises(build.PackageRejected) as info:
        build.build_image(make_spec(tmp_path), docker_runner=FakeDocker("not-a-digest"), context_dir=tmp_path / "c")
    assert info.value.rule_id == build.RULE_BUILD_IMAGE_ID_INVALID
    other = json.dumps([{"Id": "sha256:" + "9" * 64, "RepoDigests": []}])
    with pytest.raises(build.PackageRejected) as info:
        build.build_image(make_spec(tmp_path), docker_runner=FakeDocker(image_id, inspect_stdout=other), context_dir=tmp_path / "d")
    assert info.value.rule_id == build.RULE_BUILD_INSPECT_MISMATCH
    with pytest.raises(build.PackageRejected) as info:
        build.build_image(make_spec(tmp_path), docker_runner=FakeDocker(image_id, inspect_stdout="{bad"), context_dir=tmp_path / "e")
    assert info.value.rule_id == build.RULE_BUILD_INSPECT_MISMATCH
    bad_repo = json.dumps([{"Id": image_id, "RepoDigests": ["registry/x@sha256:short"]}])
    with pytest.raises(build.PackageRejected) as info:
        build.build_image(make_spec(tmp_path), docker_runner=FakeDocker(image_id, inspect_stdout=bad_repo), context_dir=tmp_path / "f")
    assert info.value.rule_id == build.RULE_BUILD_INSPECT_MISMATCH


def test_write_build_context_requires_empty_directory_and_clean_wheelhouse(tmp_path: Path) -> None:
    spec = make_spec(tmp_path)
    context = tmp_path / "ctx"
    context.mkdir()
    (context / "stale").write_text("x")
    with pytest.raises(build.PackageRejected) as info:
        build.write_build_context(spec, context)
    assert info.value.rule_id == build.RULE_BUILD_CONTEXT_INVALID
    (tmp_path / "wheelhouse" / "setup.py").write_text("print('never')")
    with pytest.raises(build.PackageRejected) as info:
        build.write_build_context(spec, tmp_path / "ctx2")
    assert info.value.rule_id == build.RULE_BUILD_CONTEXT_INVALID
    assert "wheel files" in info.value.detail


# ---------------------------------------------------------------------------
# Screening
# ---------------------------------------------------------------------------

FIXTURE_ICP = {"icp_id": "fixture", "prompt": "Series B robotics manufacturers in the United States", "industry": "Manufacturing"}
SYNTHETIC_ICPS = [
    {"icp_id": "synthetic-%d" % index, "prompt": "synthetic prompt %d" % index, "industry": "Software"}
    for index in range(3)
]


def company(website: str, name: str = "Acme Robotics") -> Dict[str, Any]:
    return {
        "company_name": name,
        "company_website": website,
        "industry": "Manufacturing",
        "employee_count": "51-200",
        "country": "United States",
        "intent_signals": [
            {
                "source": "news",
                "description": "Raised a Series B to expand production capacity",
                "url": "https://news.example.com/acme-series-b",
                "date": "2026-08-01",
                "snippet": "Acme Robotics raised a Series B round to expand production.",
            }
        ],
    }


def model(fixture_output: Any, synthetic_output: Any = None, *, raise_on: Optional[str] = None) -> Callable[[Dict[str, Any], bool], Any]:
    calls: List[Tuple[str, bool]] = []

    def run(icp: Dict[str, Any], providers_enabled: bool) -> Any:
        calls.append((icp["icp_id"], providers_enabled))
        if raise_on is not None and icp["icp_id"].startswith(raise_on):
            raise RuntimeError("model exploded")
        if providers_enabled:
            return fixture_output
        if synthetic_output is None:
            return []
        return synthetic_output(icp) if callable(synthetic_output) else synthetic_output

    run.calls = calls  # type: ignore[attr-defined]
    return run


def test_screening_accepts_a_model_that_needs_providers() -> None:
    run = model([company("https://acme-robotics.example.com")])
    result = build.screen_model(run, fixture_icp=FIXTURE_ICP, synthetic_icps=SYNTHETIC_ICPS)
    assert result.accepted is True
    assert result.rule_id is None
    assert result.fixture_company_count == 1
    assert result.synthetic_company_counts == (0, 0, 0)
    assert run.calls == [("fixture", True), ("synthetic-0", False), ("synthetic-1", False), ("synthetic-2", False)]  # type: ignore[attr-defined]


def test_screening_rejects_no_companies_with_providers() -> None:
    result = build.screen_model(model([]), fixture_icp=FIXTURE_ICP, synthetic_icps=SYNTHETIC_ICPS)
    assert (result.accepted, result.rule_id) == (False, build.RULE_SCREENING_NO_COMPANIES)


def test_screening_rejects_invalid_company_output() -> None:
    invalid = {**company("https://acme.example.com"), "contact_email": "ceo@acme.example.com"}
    result = build.screen_model(model([invalid]), fixture_icp=FIXTURE_ICP, synthetic_icps=SYNTHETIC_ICPS)
    assert (result.accepted, result.rule_id) == (False, build.RULE_SCREENING_INVALID_OUTPUT)
    assert "ceo@" not in result.detail
    result = build.screen_model(model({"companies": []}), fixture_icp=FIXTURE_ICP, synthetic_icps=SYNTHETIC_ICPS)
    assert result.rule_id == build.RULE_SCREENING_INVALID_OUTPUT


def test_screening_rejects_companies_without_providers() -> None:
    run = model(
        [company("https://acme-robotics.example.com")],
        lambda icp: [company("https://%s.example.com" % icp["icp_id"], "Other")],
    )
    result = build.screen_model(run, fixture_icp=FIXTURE_ICP, synthetic_icps=SYNTHETIC_ICPS)
    assert (result.accepted, result.rule_id) == (False, build.RULE_SCREENING_WITHOUT_PROVIDERS)
    assert result.synthetic_company_counts == (1, 1, 1)


def test_screening_rejects_identical_companies_across_icps() -> None:
    hardcoded = [company("https://acme-robotics.example.com")]
    run = model(hardcoded, [dict(item) for item in hardcoded])
    result = build.screen_model(run, fixture_icp=FIXTURE_ICP, synthetic_icps=SYNTHETIC_ICPS)
    assert (result.accepted, result.rule_id) == (False, build.RULE_SCREENING_IDENTICAL)
    assert result.rule_id in build.SCREENING_RULE_IDS
    assert "ICPs 0 and 1" in result.detail
    # Three synthetic ICPs answering with the same set (different from the
    # fixture) is also "the same companies for different ICPs".
    run = model([company("https://acme-robotics.example.com")], [company("https://other.example.com", "Other")])
    result = build.screen_model(run, fixture_icp=FIXTURE_ICP, synthetic_icps=SYNTHETIC_ICPS)
    assert (result.accepted, result.rule_id) == (False, build.RULE_SCREENING_IDENTICAL)
    assert "ICPs 1 and 2" in result.detail


def test_screening_rejects_model_errors_instead_of_passing() -> None:
    run = model([company("https://acme-robotics.example.com")], raise_on="synthetic-1")
    result = build.screen_model(run, fixture_icp=FIXTURE_ICP, synthetic_icps=SYNTHETIC_ICPS)
    assert (result.accepted, result.rule_id) == (False, build.RULE_SCREENING_MODEL_ERROR)
    assert "RuntimeError" in result.detail and "exploded" not in result.detail
    result = build.screen_model(model([], raise_on="fixture"), fixture_icp=FIXTURE_ICP, synthetic_icps=SYNTHETIC_ICPS)
    assert result.rule_id == build.RULE_SCREENING_MODEL_ERROR


def test_screening_configuration_errors_are_not_model_rejections() -> None:
    run = model([company("https://acme-robotics.example.com")])
    with pytest.raises(ArenaContractError, match="exactly 3"):
        build.screen_model(run, fixture_icp=FIXTURE_ICP, synthetic_icps=SYNTHETIC_ICPS[:2])
    with pytest.raises(ArenaContractError, match="distinct"):
        build.screen_model(run, fixture_icp=FIXTURE_ICP, synthetic_icps=[FIXTURE_ICP] + SYNTHETIC_ICPS[:2])
