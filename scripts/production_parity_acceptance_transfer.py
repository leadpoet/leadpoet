#!/usr/bin/env python3
"""Transfer one signed acceptance corpus into a disposable parity host."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import sys
import tarfile
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gateway.tee.acceptance_corpus_v2 import (
    load_and_validate_acceptance_corpus_v2,
)
from gateway.tee.release_manifest_v2 import validate_release_manifest
from leadpoet_canonical.attested_v2 import sha256_json


SCHEMA_VERSION = "leadpoet.production_parity_acceptance_transfer.v1"
ARCHIVE_NAME = "acceptance-corpus-v2.tar"
BINDING_NAME = "acceptance-corpus-v2-binding.json"
HASH_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
SHA_RE = re.compile(r"^[0-9a-f]{40}$")
RUN_RE = re.compile(r"^[a-z0-9-]{6,40}$")
BUCKET_RE = re.compile(r"^[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]$")
MAX_ARCHIVE_BYTES = 16 * 1024 * 1024
MAX_BINDING_BYTES = 16 * 1024
MAX_MEMBER_BYTES = 4 * 1024 * 1024
MAX_MEMBER_COUNT = 1024
MAX_TOTAL_FILE_BYTES = 64 * 1024 * 1024


class AcceptanceTransferError(RuntimeError):
    """The signed acceptance corpus transfer is incomplete or conflicting."""


def _sha256_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _read_bounded_body(response: Mapping[str, Any], maximum: int) -> bytes:
    stream = response.get("Body")
    close = getattr(stream, "close", None)
    try:
        length = response.get("ContentLength")
        if not isinstance(length, int) or not 0 < length <= maximum:
            raise AcceptanceTransferError(
                "acceptance transfer object size is invalid"
            )
        if stream is None or not callable(getattr(stream, "read", None)):
            raise AcceptanceTransferError(
                "acceptance transfer object body is invalid"
            )
        payload = stream.read(maximum + 1)
    finally:
        if callable(close):
            close()
    if not isinstance(payload, bytes) or len(payload) != length:
        raise AcceptanceTransferError("acceptance transfer object body differs")
    return payload


def _write_exclusive(path: Path, payload: bytes, *, mode: int = 0o600) -> None:
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | os.O_CLOEXEC
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = os.open(path, flags, mode)
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise AcceptanceTransferError("acceptance transfer write failed")
            view = view[written:]
        os.fchmod(descriptor, mode)
        os.fsync(descriptor)
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    finally:
        os.close(descriptor)


def _read_regular(
    path: Path,
    maximum: int,
    *,
    owner: tuple[int, int] | None = None,
    mode: int = 0o600,
) -> bytes:
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0),
        )
    except OSError as exc:
        raise AcceptanceTransferError(
            "acceptance transfer file is unavailable"
        ) from exc
    try:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or stat.S_IMODE(metadata.st_mode) != mode
            or not 0 < metadata.st_size <= maximum
            or (
                owner is not None
                and (metadata.st_uid, metadata.st_gid) != owner
            )
        ):
            raise AcceptanceTransferError("acceptance transfer file differs")
        chunks: list[bytes] = []
        observed = 0
        while True:
            chunk = os.read(descriptor, min(1024 * 1024, maximum + 1 - observed))
            if not chunk:
                break
            observed += len(chunk)
            if observed > maximum:
                raise AcceptanceTransferError("acceptance transfer file is oversized")
            chunks.append(chunk)
        payload = b"".join(chunks)
        if len(payload) != metadata.st_size:
            raise AcceptanceTransferError("acceptance transfer file body differs")
        return payload
    finally:
        os.close(descriptor)


def _release_identity(
    *,
    candidate_sha: str,
    candidate_release_manifest: Mapping[str, Any],
) -> tuple[dict[str, Any], str]:
    if SHA_RE.fullmatch(candidate_sha) is None:
        raise AcceptanceTransferError("acceptance transfer candidate is invalid")
    try:
        release = validate_release_manifest(candidate_release_manifest)
    except Exception as exc:
        raise AcceptanceTransferError(
            "acceptance release identity differs"
        ) from exc
    signer_hash = str(release.get("acceptance_signer_pubkey_hash") or "")
    if (
        release.get("commit_sha") != candidate_sha
        or HASH_RE.fullmatch(str(release.get("release_hash") or "")) is None
        or HASH_RE.fullmatch(signer_hash) is None
    ):
        raise AcceptanceTransferError("acceptance release identity differs")
    return release, signer_hash


def _declared_signer_hash(manifest: Path) -> str:
    try:
        payload = _read_regular(manifest, MAX_MEMBER_BYTES)
        value = json.loads(payload.decode("ascii"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise AcceptanceTransferError(
            "acceptance corpus signer identity differs"
        ) from exc
    public_key = (
        str(value.get("signing_pubkey_hex") or "")
        if isinstance(value, Mapping)
        else ""
    )
    if re.fullmatch(r"[0-9a-f]{64}", public_key) is None:
        raise AcceptanceTransferError(
            "acceptance corpus signer identity differs"
        )
    return _sha256_bytes(bytes.fromhex(public_key))


def _validated_tree(
    config_dir: Path,
    *,
    signer_hash: str,
) -> tuple[dict[str, Any], list[Path], list[Path], tuple[int, int]]:
    root = Path(config_dir)
    manifest = root / "acceptance-corpus-v2.json"
    corpus = root / "acceptance-corpus-v2"
    try:
        root_metadata = root.lstat()
        corpus_metadata = corpus.lstat()
    except OSError as exc:
        raise AcceptanceTransferError("acceptance corpus is unavailable") from exc
    owner = (root_metadata.st_uid, root_metadata.st_gid)
    if (
        root.is_symlink()
        or corpus.is_symlink()
        or not stat.S_ISDIR(root_metadata.st_mode)
        or not stat.S_ISDIR(corpus_metadata.st_mode)
        or stat.S_IMODE(root_metadata.st_mode) != 0o700
        or stat.S_IMODE(corpus_metadata.st_mode) != 0o700
        or (corpus_metadata.st_uid, corpus_metadata.st_gid) != owner
    ):
        raise AcceptanceTransferError("acceptance corpus ownership differs")
    manifest_payload = _read_regular(
        manifest,
        MAX_MEMBER_BYTES,
        owner=owner,
    )

    directories: list[Path] = []
    files: list[Path] = []
    total_bytes = len(manifest_payload)
    for current, names, filenames in os.walk(corpus, followlinks=False):
        current_path = Path(current)
        for name in sorted(names):
            path = current_path / name
            metadata = path.lstat()
            if (
                path.is_symlink()
                or not stat.S_ISDIR(metadata.st_mode)
                or stat.S_IMODE(metadata.st_mode) != 0o700
                or (metadata.st_uid, metadata.st_gid) != owner
            ):
                raise AcceptanceTransferError(
                    "acceptance corpus ownership differs"
                )
            directories.append(path)
        for name in sorted(filenames):
            path = current_path / name
            metadata = path.lstat()
            if (
                path.is_symlink()
                or not stat.S_ISREG(metadata.st_mode)
                or stat.S_IMODE(metadata.st_mode) != 0o600
                or (metadata.st_uid, metadata.st_gid) != owner
                or not 0 < metadata.st_size <= MAX_MEMBER_BYTES
            ):
                raise AcceptanceTransferError("acceptance corpus file differs")
            total_bytes += metadata.st_size
            files.append(path)
    if (
        not files
        or len(directories) + len(files) + 2 > MAX_MEMBER_COUNT
        or total_bytes > MAX_TOTAL_FILE_BYTES
    ):
        raise AcceptanceTransferError("acceptance corpus size differs")
    try:
        document = load_and_validate_acceptance_corpus_v2(
            manifest,
            corpus_root=corpus,
            expected_signing_pubkey_hash=signer_hash,
        )
    except Exception as exc:
        raise AcceptanceTransferError(
            "acceptance corpus signature differs"
        ) from exc
    listed = {
        PurePosixPath(str(item.get("artifact_path") or "")).as_posix()
        for item in document.get("fixtures") or ()
        if isinstance(item, Mapping)
    }
    discovered = {path.relative_to(corpus).as_posix() for path in files}
    expected_directories = {
        parent.as_posix()
        for listed_file in listed
        for parent in PurePosixPath(listed_file).parents
        if parent != PurePosixPath(".")
    }
    discovered_directories = {
        path.relative_to(corpus).as_posix() for path in directories
    }
    if (
        not listed
        or listed != discovered
        or expected_directories != discovered_directories
        or len(listed) != len(document.get("fixtures") or ())
    ):
        raise AcceptanceTransferError("acceptance corpus file set differs")
    return document, directories, files, owner


def _tar_info(name: str, *, directory: bool, size: int = 0) -> tarfile.TarInfo:
    info = tarfile.TarInfo(name)
    info.type = tarfile.DIRTYPE if directory else tarfile.REGTYPE
    info.mode = 0o700 if directory else 0o600
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    info.mtime = 0
    info.size = 0 if directory else size
    return info


def _secure_directory(path: Path, *, field: str) -> None:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise AcceptanceTransferError(f"{field} is unavailable") from exc
    if (
        path.is_symlink()
        or not stat.S_ISDIR(metadata.st_mode)
        or stat.S_IMODE(metadata.st_mode) != 0o700
        or (metadata.st_uid, metadata.st_gid) != (os.getuid(), os.getgid())
    ):
        raise AcceptanceTransferError(f"{field} differs")


def package_transfer(
    *,
    source_config_dir: Path,
    candidate_sha: str,
    archive_path: Path,
    binding_path: Path,
) -> dict[str, Any]:
    if SHA_RE.fullmatch(candidate_sha) is None:
        raise AcceptanceTransferError("acceptance transfer candidate is invalid")
    source_root = Path(source_config_dir)
    signer_hash = _declared_signer_hash(
        source_root / "acceptance-corpus-v2.json"
    )
    document, directories, files, owner = _validated_tree(
        source_root,
        signer_hash=signer_hash,
    )
    corpus_root = source_root / "acceptance-corpus-v2"
    manifest = source_root / "acceptance-corpus-v2.json"
    if (
        archive_path.parent != binding_path.parent
        or archive_path.exists()
        or binding_path.exists()
    ):
        raise AcceptanceTransferError("acceptance transfer output differs")
    _secure_directory(
        archive_path.parent,
        field="acceptance transfer output root",
    )
    descriptor = os.open(
        archive_path,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | os.O_CLOEXEC
        | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            with tarfile.open(fileobj=handle, mode="w") as archive:
                manifest_payload = _read_regular(
                    manifest,
                    MAX_MEMBER_BYTES,
                    owner=owner,
                )
                archive.addfile(
                    _tar_info(
                        "acceptance-corpus-v2.json",
                        directory=False,
                        size=len(manifest_payload),
                    ),
                    io.BytesIO(manifest_payload),
                )
                archive.addfile(
                    _tar_info("acceptance-corpus-v2", directory=True)
                )
                for directory in sorted(directories):
                    archive.addfile(
                        _tar_info(
                            "acceptance-corpus-v2/"
                            + directory.relative_to(corpus_root).as_posix(),
                            directory=True,
                        )
                    )
                for source in sorted(files):
                    payload = _read_regular(
                        source,
                        MAX_MEMBER_BYTES,
                        owner=owner,
                    )
                    archive.addfile(
                        _tar_info(
                            "acceptance-corpus-v2/"
                            + source.relative_to(corpus_root).as_posix(),
                            directory=False,
                            size=len(payload),
                        ),
                        io.BytesIO(payload),
                    )
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        archive_path.unlink(missing_ok=True)
        raise
    archive_payload = _read_regular(archive_path, MAX_ARCHIVE_BYTES)
    body = {
        "schema_version": SCHEMA_VERSION,
        "candidate_sha": candidate_sha,
        "manifest_hash": document["manifest_hash"],
        "fixture_count": len(files),
        "archive_sha256": _sha256_bytes(archive_payload),
        "archive_size_bytes": len(archive_payload),
    }
    binding = {**body, "binding_hash": sha256_json(body)}
    binding_payload = (
        json.dumps(binding, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("ascii")
    try:
        _write_exclusive(binding_path, binding_payload)
    except BaseException:
        archive_path.unlink(missing_ok=True)
        raise
    return binding


def _validated_binding(payload: bytes, *, candidate_sha: str) -> dict[str, Any]:
    if not 0 < len(payload) <= MAX_BINDING_BYTES:
        raise AcceptanceTransferError("acceptance transfer binding size differs")
    try:
        value = json.loads(payload.decode("ascii"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise AcceptanceTransferError(
            "acceptance transfer binding is invalid"
        ) from exc
    fields = {
        "schema_version",
        "candidate_sha",
        "manifest_hash",
        "fixture_count",
        "archive_sha256",
        "archive_size_bytes",
        "binding_hash",
    }
    if not isinstance(value, dict) or set(value) != fields:
        raise AcceptanceTransferError("acceptance transfer binding fields differ")
    body = {key: value[key] for key in fields if key != "binding_hash"}
    if (
        value.get("schema_version") != SCHEMA_VERSION
        or value.get("candidate_sha") != candidate_sha
        or HASH_RE.fullmatch(str(value.get("manifest_hash") or "")) is None
        or HASH_RE.fullmatch(str(value.get("archive_sha256") or "")) is None
        or not isinstance(value.get("fixture_count"), int)
        or not 0 < value["fixture_count"] < MAX_MEMBER_COUNT
        or not isinstance(value.get("archive_size_bytes"), int)
        or not 0 < value["archive_size_bytes"] <= MAX_ARCHIVE_BYTES
        or value.get("binding_hash") != sha256_json(body)
    ):
        raise AcceptanceTransferError("acceptance transfer binding differs")
    return value


def unpack_transfer(
    *,
    archive_payload: bytes,
    binding_payload: bytes,
    candidate_sha: str,
    destination_config_dir: Path,
    candidate_release_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    binding = _validated_binding(binding_payload, candidate_sha=candidate_sha)
    if (
        len(archive_payload) != binding["archive_size_bytes"]
        or _sha256_bytes(archive_payload) != binding["archive_sha256"]
    ):
        raise AcceptanceTransferError("acceptance transfer archive differs")
    release, signer_hash = _release_identity(
        candidate_sha=candidate_sha,
        candidate_release_manifest=candidate_release_manifest,
    )
    destination = Path(destination_config_dir)
    if destination.exists() or destination.is_symlink():
        raise AcceptanceTransferError("acceptance transfer destination exists")
    members: list[tuple[tarfile.TarInfo, bytes | None]] = []
    total_bytes = 0
    try:
        with tarfile.open(fileobj=io.BytesIO(archive_payload), mode="r:") as archive:
            for member in archive.getmembers():
                path = PurePosixPath(member.name)
                is_manifest = path.parts == ("acceptance-corpus-v2.json",)
                is_corpus_root = path.parts == ("acceptance-corpus-v2",)
                is_corpus_member = (
                    len(path.parts) > 1
                    and path.parts[0] == "acceptance-corpus-v2"
                )
                if (
                    path.is_absolute()
                    or ".." in path.parts
                    or not path.parts
                    or path.as_posix() != member.name
                    or not (is_manifest or is_corpus_root or is_corpus_member)
                    or (is_manifest and not member.isfile())
                    or (is_corpus_root and not member.isdir())
                    or member.uid != 0
                    or member.gid != 0
                    or member.mtime != 0
                ):
                    raise AcceptanceTransferError(
                        "acceptance transfer member differs"
                    )
                if member.isdir():
                    if member.mode != 0o700 or member.size != 0:
                        raise AcceptanceTransferError(
                            "acceptance directory member differs"
                        )
                    payload = None
                elif member.isfile():
                    if (
                        member.mode != 0o600
                        or not 0 < member.size <= MAX_MEMBER_BYTES
                    ):
                        raise AcceptanceTransferError(
                            "acceptance file member differs"
                        )
                    extracted = archive.extractfile(member)
                    payload = (
                        extracted.read(MAX_MEMBER_BYTES + 1)
                        if extracted is not None
                        else b""
                    )
                    if len(payload) != member.size:
                        raise AcceptanceTransferError(
                            "acceptance file member body differs"
                        )
                    total_bytes += len(payload)
                else:
                    raise AcceptanceTransferError(
                        "acceptance transfer member type differs"
                    )
                members.append((member, payload))
    except (tarfile.TarError, OSError) as exc:
        raise AcceptanceTransferError(
            "acceptance transfer archive is invalid"
        ) from exc
    names = [member.name for member, _ in members]
    directory_names = {
        member.name for member, _ in members if member.isdir()
    }
    expected_order = [
        "acceptance-corpus-v2.json",
        "acceptance-corpus-v2",
        *sorted(directory_names - {"acceptance-corpus-v2"}),
        *sorted(
            member.name
            for member, _ in members
            if member.isfile()
            and member.name != "acceptance-corpus-v2.json"
        ),
    ]
    parents_are_explicit = all(
        PurePosixPath(name).parent.as_posix() in directory_names
        for name in names
        if name
        not in {"acceptance-corpus-v2.json", "acceptance-corpus-v2"}
    )
    if (
        len(names) != len(set(names))
        or not 2 < len(names) <= MAX_MEMBER_COUNT
        or names.count("acceptance-corpus-v2.json") != 1
        or names.count("acceptance-corpus-v2") != 1
        or names != expected_order
        or not parents_are_explicit
        or total_bytes > MAX_TOTAL_FILE_BYTES
    ):
        raise AcceptanceTransferError("acceptance transfer member set differs")
    _secure_directory(
        destination.parent,
        field="acceptance transfer destination root",
    )
    try:
        destination.mkdir(mode=0o700)
        for member, payload in sorted(
            members,
            key=lambda item: (
                0 if item[0].isdir() else 1,
                len(PurePosixPath(item[0].name).parts),
                item[0].name,
            ),
        ):
            target = destination.joinpath(*PurePosixPath(member.name).parts)
            if member.isdir():
                target.mkdir(mode=0o700)
            else:
                if not target.parent.is_dir() or target.parent.is_symlink():
                    raise AcceptanceTransferError(
                        "acceptance transfer member parent differs"
                    )
                _write_exclusive(target, payload or b"")
        document, _directories, files, _owner = _validated_tree(
            destination,
            signer_hash=signer_hash,
        )
        if (
            document["manifest_hash"] != binding["manifest_hash"]
            or len(files) != binding["fixture_count"]
        ):
            raise AcceptanceTransferError(
                "acceptance transfer content differs"
            )
    except BaseException:
        shutil.rmtree(destination, ignore_errors=True)
        raise
    return {
        **binding,
        "release_hash": release["release_hash"],
        "copied_exact": True,
    }


def fetch_and_unpack_transfer(
    *,
    s3_client: Any,
    artifact_bucket: str,
    run_id: str,
    candidate_sha: str,
    destination_config_dir: Path,
    candidate_release_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    if (
        BUCKET_RE.fullmatch(artifact_bucket) is None
        or RUN_RE.fullmatch(run_id) is None
        or SHA_RE.fullmatch(candidate_sha) is None
    ):
        raise AcceptanceTransferError(
            "acceptance transfer source identity differs"
        )
    prefix = f"production-parity/runs/{run_id}"
    archive_payload = _read_bounded_body(
        s3_client.get_object(
            Bucket=artifact_bucket,
            Key=f"{prefix}/{ARCHIVE_NAME}",
        ),
        MAX_ARCHIVE_BYTES,
    )
    binding_payload = _read_bounded_body(
        s3_client.get_object(
            Bucket=artifact_bucket,
            Key=f"{prefix}/{BINDING_NAME}",
        ),
        MAX_BINDING_BYTES,
    )
    return unpack_transfer(
        archive_payload=archive_payload,
        binding_payload=binding_payload,
        candidate_sha=candidate_sha,
        destination_config_dir=destination_config_dir,
        candidate_release_manifest=candidate_release_manifest,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-sha", required=True)
    parser.add_argument("--source-config-dir", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--binding", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = package_transfer(
            source_config_dir=args.source_config_dir,
            candidate_sha=args.candidate_sha.lower(),
            archive_path=args.archive,
            binding_path=args.binding,
        )
    except (OSError, ValueError, AcceptanceTransferError):
        print(
            "ERROR: acceptance corpus transfer packaging failed",
            file=sys.stderr,
        )
        return 1
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "candidate_sha",
                    "manifest_hash",
                    "fixture_count",
                    "archive_sha256",
                    "archive_size_bytes",
                    "binding_hash",
                )
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
