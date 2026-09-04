"""Small source archive contract for Arena agent submissions."""

from __future__ import annotations

import ast
import gzip
import hashlib
import io
import os
import stat
import tarfile
import zlib
from pathlib import Path, PurePosixPath
from typing import Any, Dict, List, Tuple

SOURCE_CONTENT_TYPE = "application/gzip"
MAX_SOURCE_ARCHIVE_BYTES = 10 * 1024 * 1024
MAX_SOURCE_UNPACKED_BYTES = 50 * 1024 * 1024
MAX_SOURCE_FILES = 1_000
MAX_SOURCE_PATH_BYTES = 255
MAX_HARNESS_BYTES = 1 * 1024 * 1024
IGNORED_DIRECTORY_NAMES = frozenset({".git", ".pytest_cache", ".venv", "__pycache__", "node_modules"})


class SourceBundleError(ValueError):
    """The submitted source does not meet the small public boundary."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


def validate_harness_source(source: str) -> None:
    """Check only Python syntax without importing or running miner code.

    A public harness can define ``run_icp`` here or re-export it from another
    module.  The sandbox entrypoint imports the completed bundle and enforces
    the callable contract before it runs untrusted code.
    """

    try:
        tree = ast.parse(source, filename="harness.py")
    except (SyntaxError, ValueError) as exc:
        raise SourceBundleError("harness_invalid") from exc
    del tree


def validate_source_directory(source_dir: str | Path) -> Path:
    """Validate the local source boundary without following links."""

    source = Path(source_dir).expanduser().resolve()
    if not source.is_dir():
        raise SourceBundleError("source_directory_missing")
    harness = source / "harness.py"
    if not harness.is_file() or harness.is_symlink():
        raise SourceBundleError("harness_file_missing")
    try:
        raw = harness.read_bytes()
    except OSError as exc:
        raise SourceBundleError("harness_invalid") from exc
    if len(raw) > MAX_HARNESS_BYTES:
        raise SourceBundleError("harness_too_large")
    try:
        validate_harness_source(raw.decode("utf-8"))
    except UnicodeDecodeError as exc:
        raise SourceBundleError("harness_invalid") from exc
    return source


def _source_files(source: Path) -> List[Tuple[Path, str, os.stat_result]]:
    files: List[Tuple[Path, str, os.stat_result]] = []
    total = 0
    for candidate in sorted(source.rglob("*"), key=lambda item: item.relative_to(source).as_posix()):
        relative = candidate.relative_to(source)
        if any(part in IGNORED_DIRECTORY_NAMES for part in relative.parts):
            continue
        try:
            details = candidate.lstat()
        except OSError as exc:
            raise SourceBundleError("source_unreadable") from exc
        if stat.S_ISDIR(details.st_mode):
            continue
        if not stat.S_ISREG(details.st_mode):
            raise SourceBundleError("source_entry_type_invalid")
        name = relative.as_posix()
        try:
            encoded_name = name.encode("utf-8")
        except UnicodeError as exc:
            raise SourceBundleError("source_path_invalid") from exc
        if not name or len(encoded_name) > MAX_SOURCE_PATH_BYTES:
            raise SourceBundleError("source_path_invalid")
        total += int(details.st_size)
        if total > MAX_SOURCE_UNPACKED_BYTES:
            raise SourceBundleError("source_unpacked_too_large")
        files.append((candidate, name, details))
        if len(files) > MAX_SOURCE_FILES:
            raise SourceBundleError("source_file_count_exceeded")
    if not files:
        raise SourceBundleError("source_empty")
    return files


def write_source_archive(source_dir: str | Path, target: str | Path) -> Dict[str, Any]:
    """Write one sorted archive with normalized metadata and return its transport facts."""

    source = validate_source_directory(source_dir)
    files = _source_files(source)
    output = Path(target)
    try:
        with output.open("wb") as raw:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
                with tarfile.open(fileobj=compressed, mode="w", format=tarfile.USTAR_FORMAT) as archive:
                    for path, name, details in files:
                        info = tarfile.TarInfo(name=name)
                        info.size = int(details.st_size)
                        info.mode = 0o755 if details.st_mode & 0o111 else 0o644
                        info.uid = 0
                        info.gid = 0
                        info.uname = ""
                        info.gname = ""
                        info.mtime = 0
                        with path.open("rb") as handle:
                            archive.addfile(info, handle)
    except (OSError, tarfile.TarError, ValueError) as exc:
        try:
            output.unlink()
        except FileNotFoundError:
            pass
        raise SourceBundleError("source_archive_failed") from exc
    size = output.stat().st_size
    if size < 1 or size > MAX_SOURCE_ARCHIVE_BYTES:
        output.unlink()
        raise SourceBundleError("source_archive_too_large")
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    return {
        "source_sha256": "sha256:" + digest,
        "source_size_bytes": size,
    }


def _read_harness(archive: tarfile.TarFile, member: tarfile.TarInfo) -> bytes:
    if member.size > MAX_HARNESS_BYTES:
        raise SourceBundleError("harness_too_large")
    handle = archive.extractfile(member)
    if handle is None:
        raise SourceBundleError("harness_invalid")
    value = handle.read(MAX_HARNESS_BYTES + 1)
    if len(value) > MAX_HARNESS_BYTES:
        raise SourceBundleError("harness_too_large")
    return value


def _safe_members(archive: tarfile.TarFile) -> Tuple[bytes, str]:
    names = set()
    file_names = set()
    harnesses: Dict[str, bytes] = {}
    total = 0
    count = 0
    for member in archive:
        count += 1
        if count > MAX_SOURCE_FILES:
            raise SourceBundleError("source_file_count_exceeded")
        path = PurePosixPath(member.name)
        if path.is_absolute() or not path.parts or any(part in ("", ".", "..") for part in path.parts):
            raise SourceBundleError("source_path_invalid")
        try:
            encoded_name = member.name.encode("utf-8")
        except UnicodeError as exc:
            raise SourceBundleError("source_path_invalid") from exc
        if len(encoded_name) > MAX_SOURCE_PATH_BYTES or member.name in names:
            raise SourceBundleError("source_path_invalid")
        names.add(member.name)
        if member.isdir():
            continue
        if not member.isfile():
            raise SourceBundleError("source_entry_type_invalid")
        total += int(member.size)
        if total > MAX_SOURCE_UNPACKED_BYTES:
            raise SourceBundleError("source_unpacked_too_large")
        file_names.add(member.name)
        if path.name == "harness.py" and len(path.parts) <= 2:
            harnesses[member.name] = _read_harness(archive, member)
    if "harness.py" in file_names:
        return harnesses["harness.py"], "harness.py"
    roots = {PurePosixPath(name).parts[0] for name in file_names}
    if len(roots) == 1:
        wrapped = next(iter(roots)) + "/harness.py"
        if wrapped in file_names:
            return harnesses[wrapped], wrapped
    raise SourceBundleError("harness_file_missing")


def validate_source_archive(data: bytes) -> Dict[str, Any]:
    """Validate bounded archive structure and the final callable without extraction."""

    payload = bytes(data)
    if not 1 <= len(payload) <= MAX_SOURCE_ARCHIVE_BYTES:
        raise SourceBundleError("source_archive_too_large")
    try:
        # Stream members so a small compressed archive cannot force an
        # unbounded expanded member list into memory.
        with tarfile.open(fileobj=io.BytesIO(payload), mode="r|gz") as archive:
            harness, harness_name = _safe_members(archive)
    except SourceBundleError:
        raise
    except (OSError, EOFError, tarfile.TarError, zlib.error) as exc:
        raise SourceBundleError("source_archive_invalid") from exc
    try:
        validate_harness_source(harness.decode("utf-8"))
    except UnicodeDecodeError as exc:
        raise SourceBundleError("harness_invalid") from exc
    return {
        "source_sha256": "sha256:" + hashlib.sha256(payload).hexdigest(),
        "source_size_bytes": len(payload),
        "source_root": harness_name.rsplit("/", 1)[0] if "/" in harness_name else "",
    }


def extract_source_archive(data: bytes, target_dir: str | Path) -> Dict[str, Any]:
    """Validate and safely extract an archive into one empty host directory.

    The implementation never calls ``TarFile.extract``.  It creates only the
    validated regular files and directories below ``target_dir`` and removes
    the optional single GitHub wrapper directory.
    """

    payload = bytes(data)
    facts = validate_source_archive(payload)
    target = Path(target_dir)
    if target.is_symlink() or not target.is_dir() or any(target.iterdir()):
        raise SourceBundleError("source_extract_target_invalid")
    source_root = str(facts["source_root"])
    prefix = source_root + "/" if source_root else ""
    extracted_files = 0
    extracted_bytes = 0
    try:
        with tarfile.open(fileobj=io.BytesIO(payload), mode="r|gz") as archive:
            for member in archive:
                path = PurePosixPath(member.name)
                if path.is_absolute() or not path.parts or any(
                    part in ("", ".", "..") for part in path.parts
                ):
                    raise SourceBundleError("source_path_invalid")
                if source_root:
                    if member.name == source_root or member.name == source_root + "/":
                        continue
                    if not member.name.startswith(prefix):
                        raise SourceBundleError("source_path_invalid")
                    relative_name = member.name[len(prefix) :]
                else:
                    relative_name = member.name
                relative = PurePosixPath(relative_name)
                if not relative.parts or any(part in ("", ".", "..") for part in relative.parts):
                    raise SourceBundleError("source_path_invalid")
                destination = target.joinpath(*relative.parts)
                if member.isdir():
                    destination.mkdir(parents=True, exist_ok=True)
                    continue
                if not member.isfile():
                    raise SourceBundleError("source_entry_type_invalid")
                extracted_files += 1
                extracted_bytes += int(member.size)
                if extracted_files > MAX_SOURCE_FILES:
                    raise SourceBundleError("source_file_count_exceeded")
                if extracted_bytes > MAX_SOURCE_UNPACKED_BYTES:
                    raise SourceBundleError("source_unpacked_too_large")
                destination.parent.mkdir(parents=True, exist_ok=True)
                source = archive.extractfile(member)
                if source is None:
                    raise SourceBundleError("source_archive_invalid")
                flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
                if hasattr(os, "O_NOFOLLOW"):
                    flags |= os.O_NOFOLLOW
                descriptor = os.open(destination, flags, 0o444)
                remaining = int(member.size)
                try:
                    with os.fdopen(descriptor, "wb") as output:
                        while remaining:
                            chunk = source.read(min(64 * 1024, remaining))
                            if not chunk:
                                raise SourceBundleError("source_archive_invalid")
                            output.write(chunk)
                            remaining -= len(chunk)
                        if source.read(1):
                            raise SourceBundleError("source_archive_invalid")
                except Exception:
                    try:
                        destination.unlink()
                    except OSError:
                        pass
                    raise
        harness = target / "harness.py"
        if harness.is_symlink() or not harness.is_file():
            raise SourceBundleError("harness_file_missing")
        for directory, names, files in os.walk(target, topdown=False, followlinks=False):
            for name in files:
                os.chmod(Path(directory) / name, 0o444)
            for name in names:
                os.chmod(Path(directory) / name, 0o555)
        os.chmod(target, 0o555)
    except SourceBundleError:
        raise
    except (OSError, EOFError, tarfile.TarError, zlib.error) as exc:
        raise SourceBundleError("source_archive_invalid") from exc
    return facts
