"""Deterministic private-model source bundles for measured autoresearch jobs."""

from __future__ import annotations

import base64
import io
from pathlib import Path, PurePosixPath
import shutil
import tarfile
from typing import Any, Dict, Mapping

from leadpoet_canonical.attested_v2 import sha256_bytes
from research_lab.eval.private_runtime import compute_private_source_tree_hash


SOURCE_BUNDLE_SCHEMA_VERSION = "leadpoet.private_source_bundle.v2"
MAX_SOURCE_BUNDLE_BYTES = 64 * 1024 * 1024
MAX_SOURCE_FILES = 5000


class SourceBundleV2Error(ValueError):
    """A source bundle is non-canonical, unsafe, or differs from its artifact."""


def _excluded(rel: str) -> bool:
    parts = rel.split("/")
    if any(
        part
        in {
            ".git",
            "__pycache__",
            ".pytest_cache",
            ".mypy_cache",
            ".venv",
            "venv",
        }
        for part in parts
    ):
        return True
    if rel.endswith((".pyc", ".pyo", ".env", ".pem", ".key")):
        return True
    return rel == ".env" or rel.startswith(".env.")


def _safe_relative_path(value: str) -> str:
    raw = str(value or "")
    path = PurePosixPath(raw)
    if (
        not raw
        or raw.startswith("/")
        or "\\" in raw
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise SourceBundleV2Error("source bundle contains an unsafe path")
    normalized = path.as_posix()
    if _excluded(normalized):
        raise SourceBundleV2Error("source bundle contains an excluded path")
    return normalized


def _archive_bytes(source_root: Path) -> bytes:
    root = Path(source_root).expanduser().resolve()
    if not root.is_dir():
        raise SourceBundleV2Error("source root is unavailable")
    files = []
    total = 0
    for path in sorted(root.rglob("*")):
        if path.is_symlink() or not path.is_file():
            continue
        rel = path.relative_to(root).as_posix()
        if _excluded(rel):
            continue
        _safe_relative_path(rel)
        size = path.stat().st_size
        total += size
        if total > MAX_SOURCE_BUNDLE_BYTES:
            raise SourceBundleV2Error("source bundle content exceeds size limit")
        files.append((rel, path, size))
    if not files or len(files) > MAX_SOURCE_FILES:
        raise SourceBundleV2Error("source bundle file count is outside limit")

    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w", format=tarfile.PAX_FORMAT) as archive:
        for rel, path, size in files:
            info = tarfile.TarInfo(rel)
            info.size = size
            info.mode = 0o644
            info.uid = 0
            info.gid = 0
            info.uname = "root"
            info.gname = "root"
            info.mtime = 0
            info.pax_headers = {}
            with path.open("rb") as handle:
                archive.addfile(info, handle)
    encoded = output.getvalue()
    if len(encoded) > MAX_SOURCE_BUNDLE_BYTES:
        raise SourceBundleV2Error("source bundle archive exceeds size limit")
    return encoded


def build_source_bundle_v2(source_root: Path) -> Dict[str, Any]:
    archive = _archive_bytes(source_root)
    return {
        "schema_version": SOURCE_BUNDLE_SCHEMA_VERSION,
        "archive_sha256": sha256_bytes(archive),
        "source_tree_hash": compute_private_source_tree_hash(source_root),
        "archive_size_bytes": len(archive),
        "archive_b64": base64.b64encode(archive).decode("ascii"),
    }


def extract_source_bundle_v2(
    bundle: Mapping[str, Any],
    *,
    destination: Path,
    expected_source_tree_hash: str,
) -> Dict[str, Any]:
    required = {
        "schema_version",
        "archive_sha256",
        "source_tree_hash",
        "archive_size_bytes",
        "archive_b64",
    }
    if not isinstance(bundle, Mapping) or set(bundle) != required:
        raise SourceBundleV2Error("source bundle fields are invalid")
    if bundle["schema_version"] != SOURCE_BUNDLE_SCHEMA_VERSION:
        raise SourceBundleV2Error("source bundle schema is invalid")
    try:
        archive_bytes = base64.b64decode(str(bundle["archive_b64"]), validate=True)
    except Exception as exc:
        raise SourceBundleV2Error("source bundle archive is invalid base64") from exc
    if (
        not archive_bytes
        or len(archive_bytes) > MAX_SOURCE_BUNDLE_BYTES
        or bundle["archive_size_bytes"] != len(archive_bytes)
        or bundle["archive_sha256"] != sha256_bytes(archive_bytes)
    ):
        raise SourceBundleV2Error("source bundle archive commitment mismatch")
    if bundle["source_tree_hash"] != expected_source_tree_hash:
        raise SourceBundleV2Error("source bundle declared tree differs from artifact")

    root = Path(destination)
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=False)
    file_count = 0
    total_size = 0
    try:
        with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:") as archive:
            members = archive.getmembers()
            if not members or len(members) > MAX_SOURCE_FILES:
                raise SourceBundleV2Error("source bundle member count is outside limit")
            for member in members:
                rel = _safe_relative_path(member.name)
                if not member.isfile() or member.issym() or member.islnk():
                    raise SourceBundleV2Error(
                        "source bundle contains a non-regular member"
                    )
                file_count += 1
                total_size += int(member.size)
                if total_size > MAX_SOURCE_BUNDLE_BYTES:
                    raise SourceBundleV2Error("source bundle expands beyond size limit")
                target = root / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                source = archive.extractfile(member)
                if source is None:
                    raise SourceBundleV2Error("source bundle member body is missing")
                payload = source.read(MAX_SOURCE_BUNDLE_BYTES + 1)
                if len(payload) != member.size:
                    raise SourceBundleV2Error("source bundle member size mismatch")
                target.write_bytes(payload)
                target.chmod(0o644)
    except Exception:
        shutil.rmtree(root, ignore_errors=True)
        raise
    observed_tree_hash = compute_private_source_tree_hash(root)
    if observed_tree_hash != expected_source_tree_hash:
        shutil.rmtree(root, ignore_errors=True)
        raise SourceBundleV2Error("reconstructed source tree differs from artifact")
    return {
        "schema_version": SOURCE_BUNDLE_SCHEMA_VERSION,
        "archive_sha256": bundle["archive_sha256"],
        "source_tree_hash": observed_tree_hash,
        "file_count": file_count,
        "total_size_bytes": total_size,
    }
