"""Immutable testnet epoch authority for disposable production-parity runs."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path, PurePosixPath
import re
import tarfile
from typing import Any, Mapping

from Leadpoet.utils.subnet_epoch import SubnetEpochCutover


SCHEMA_VERSION = "leadpoet.production_parity_epoch_authority_artifact.v1"
SPEC_SCHEMA_VERSION = "leadpoet.production_parity_epoch_authority_spec.v1"
CEREMONY_SCHEMA_VERSION = "leadpoet.production_parity_epoch_authority_ceremony.v1"
RUN_RE = re.compile(r"^[a-z0-9-]{6,40}$")
HASH_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
HEX_HASH_RE = re.compile(r"^0x[0-9a-f]{64}$")
TABLE_RE = re.compile(r"^research_lab_[a-z0-9_]{1,96}$")
S3_RE = re.compile(r"^s3://([A-Za-z0-9][A-Za-z0-9.-]{1,62})/([^\s]+)$")
VERSION_RE = re.compile(r"^[A-Za-z0-9_.=+-]{1,1024}$")
MAX_ARCHIVE_BYTES = 128 * 1024 * 1024
MAX_EXPANDED_BYTES = 256 * 1024 * 1024
ARCHIVE_PATHS = {
    "manifest.json",
    "stateful-epoch-cutover.json",
    "authority.dump",
}
REQUIRED_TABLES = {
    "research_lab_attested_execution_receipts_v2",
    "research_lab_attested_weight_bundles_v2",
    "research_lab_attested_publication_events_v2",
    "research_lab_attested_weight_finalizations_v2",
    "research_lab_legacy_finalized_allocation_migrations_v2",
    "research_lab_stateful_subnet_epoch_candidates_v1",
    "research_lab_stateful_subnet_epoch_cutovers_v1",
    "research_lab_stateful_subnet_epoch_boundaries_v1",
    "research_lab_stateful_subnet_epoch_snapshots_v1",
    "research_lab_stateful_subnet_epoch_cutover_state_v1",
}
REQUIRED_NONEMPTY_TABLES = REQUIRED_TABLES - {
    "research_lab_stateful_subnet_epoch_boundaries_v1",
    "research_lab_stateful_subnet_epoch_snapshots_v1",
}


class ProductionParityEpochAuthorityError(RuntimeError):
    pass


def sha256_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def install_base(run_id: str) -> Path:
    if not RUN_RE.fullmatch(run_id):
        raise ProductionParityEpochAuthorityError(
            "epoch authority install identity is invalid"
        )
    return Path("/home/ec2-user/.config/leadpoet/parity") / run_id / "epoch-authority"


def cutover_manifest_path(run_id: str) -> Path:
    return install_base(run_id) / "stateful-epoch-cutover.json"


def normalize_spec(
    value: Mapping[str, Any], *, network: str, netuid: int
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ProductionParityEpochAuthorityError(
            "epoch authority artifact spec is invalid"
        )
    document = dict(value)
    if document.get("schema_version") != SPEC_SCHEMA_VERSION:
        raise ProductionParityEpochAuthorityError(
            "epoch authority artifact spec schema differs"
        )
    uri = str(document.get("s3_uri") or "")
    match = S3_RE.fullmatch(uri)
    version_id = str(document.get("version_id") or "")
    digest = str(document.get("sha256") or "").lower()
    kms_key_arn = str(document.get("kms_key_arn") or "")
    mapping_hash = str(document.get("mapping_hash") or "").lower()
    genesis_hash = str(document.get("network_genesis_hash") or "").lower()
    if (
        match is None
        or not VERSION_RE.fullmatch(version_id)
        or not HASH_RE.fullmatch(digest)
        or not kms_key_arn.startswith("arn:aws:kms:")
        or not HASH_RE.fullmatch(mapping_hash)
        or not HEX_HASH_RE.fullmatch(genesis_hash)
        or network != "test"
        or not isinstance(netuid, int)
        or isinstance(netuid, bool)
        or netuid <= 0
    ):
        raise ProductionParityEpochAuthorityError(
            "epoch authority artifact spec fields are invalid"
        )
    return {
        "schema_version": SPEC_SCHEMA_VERSION,
        "network": "test",
        "netuid": netuid,
        "s3_uri": uri,
        "bucket": match.group(1),
        "key": match.group(2),
        "version_id": version_id,
        "sha256": digest,
        "kms_key_arn": kms_key_arn,
        "mapping_hash": mapping_hash,
        "network_genesis_hash": genesis_hash,
    }


def validate_head(
    spec: Mapping[str, Any], head: Mapping[str, Any], *, now: datetime | None = None
) -> dict[str, Any]:
    current = now or datetime.now(timezone.utc)
    retained_until = head.get("ObjectLockRetainUntilDate")
    if isinstance(retained_until, str):
        try:
            retained_until = datetime.fromisoformat(
                retained_until.replace("Z", "+00:00")
            )
        except ValueError as exc:
            raise ProductionParityEpochAuthorityError(
                "epoch authority retention timestamp is invalid"
            ) from exc
    size = head.get("ContentLength")
    if (
        head.get("ServerSideEncryption") != "aws:kms"
        or str(head.get("SSEKMSKeyId") or "") != spec["kms_key_arn"]
        or str(head.get("VersionId") or "") != spec["version_id"]
        or head.get("ObjectLockMode") not in {"GOVERNANCE", "COMPLIANCE"}
        or not isinstance(retained_until, datetime)
        or retained_until.astimezone(timezone.utc) <= current.astimezone(timezone.utc)
        or not isinstance(size, int)
        or isinstance(size, bool)
        or size <= 0
        or size > MAX_ARCHIVE_BYTES
    ):
        raise ProductionParityEpochAuthorityError(
            "epoch authority artifact is not immutable, KMS-bound, and size-bounded"
        )
    return {
        "version_id": spec["version_id"],
        "kms_key_arn": spec["kms_key_arn"],
        "object_lock_mode": str(head["ObjectLockMode"]),
        "retain_until": retained_until.astimezone(timezone.utc).isoformat(),
        "size_bytes": size,
    }


def _json(value: bytes, *, field: str) -> dict[str, Any]:
    try:
        document = json.loads(value.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise ProductionParityEpochAuthorityError(f"{field} is invalid") from exc
    if not isinstance(document, dict):
        raise ProductionParityEpochAuthorityError(f"{field} must be an object")
    return document


def _archive_files(payload: bytes) -> dict[str, bytes]:
    if not payload or len(payload) > MAX_ARCHIVE_BYTES:
        raise ProductionParityEpochAuthorityError(
            "epoch authority archive size is invalid"
        )
    files: dict[str, bytes] = {}
    expanded = 0
    try:
        with tarfile.open(fileobj=BytesIO(payload), mode="r:*") as archive:
            for member in archive.getmembers():
                path = PurePosixPath(member.name)
                name = str(path)
                if member.isdir():
                    continue
                if (
                    not member.isfile()
                    or not name
                    or path.is_absolute()
                    or ".." in path.parts
                    or name in files
                ):
                    raise ProductionParityEpochAuthorityError(
                        "epoch authority artifact contains an unsafe member"
                    )
                extracted = archive.extractfile(member)
                if extracted is None:
                    raise ProductionParityEpochAuthorityError(
                        "epoch authority artifact member is unreadable"
                    )
                value = extracted.read(MAX_EXPANDED_BYTES + 1)
                expanded += len(value)
                if len(value) > MAX_EXPANDED_BYTES or expanded > MAX_EXPANDED_BYTES:
                    raise ProductionParityEpochAuthorityError(
                        "epoch authority expanded size exceeds limit"
                    )
                files[name] = value
    except (tarfile.TarError, OSError) as exc:
        raise ProductionParityEpochAuthorityError(
            "epoch authority archive is invalid"
        ) from exc
    if set(files) != ARCHIVE_PATHS:
        raise ProductionParityEpochAuthorityError(
            "epoch authority artifact file inventory differs"
        )
    return files


def _validate_files(
    files: Mapping[str, bytes], spec: Mapping[str, Any]
) -> dict[str, Any]:
    cutover_document = _json(
        files["stateful-epoch-cutover.json"], field="testnet cutover manifest"
    )
    try:
        cutover = SubnetEpochCutover.from_mapping(cutover_document)
    except Exception as exc:
        raise ProductionParityEpochAuthorityError(
            "testnet cutover manifest is invalid"
        ) from exc
    if cutover_document != cutover.to_dict():
        raise ProductionParityEpochAuthorityError(
            "testnet cutover manifest is not canonical"
        )
    manifest = _json(files["manifest.json"], field="epoch authority manifest")
    tables = manifest.get("database_tables")
    row_counts = manifest.get("database_row_counts")
    file_hashes = manifest.get("files")
    expected_fields = {
        "schema_version",
        "network",
        "netuid",
        "network_genesis_hash",
        "mapping_hash",
        "database_tables",
        "database_row_counts",
        "files",
        "ceremony_evidence",
        "ceremony_evidence_hash",
    }
    ceremony = manifest.get("ceremony_evidence")
    ceremony_fields = {
        "schema_version",
        "network",
        "netuid",
        "network_genesis_hash",
        "mapping_hash",
        "cutover_manifest_hash",
        "database_fingerprint_hash",
        "authority_dump_hash",
        "table_count",
        "row_count",
    }
    if (
        set(manifest) != expected_fields
        or manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("network") != "test"
        or manifest.get("netuid") != spec["netuid"]
        or manifest.get("network_genesis_hash") != spec["network_genesis_hash"]
        or manifest.get("mapping_hash") != spec["mapping_hash"]
        or cutover.network_genesis_hash != spec["network_genesis_hash"]
        or cutover.netuid != spec["netuid"]
        or cutover.mapping_hash != spec["mapping_hash"]
        or not isinstance(tables, list)
        or len(tables) != len(set(tables))
        or any(not TABLE_RE.fullmatch(str(table or "")) for table in tables)
        or not REQUIRED_TABLES.issubset(set(tables))
        or not isinstance(row_counts, Mapping)
        or set(row_counts) != set(tables)
        or any(
            not isinstance(count, int) or isinstance(count, bool) or count < 0
            for count in row_counts.values()
        )
        or any(
            int(row_counts.get(table) or 0) <= 0
            for table in REQUIRED_NONEMPTY_TABLES
        )
        or not isinstance(file_hashes, Mapping)
        or set(file_hashes) != {
            "stateful-epoch-cutover.json",
            "authority.dump",
        }
        or any(
            not HASH_RE.fullmatch(str(digest or ""))
            or str(digest) != sha256_bytes(files[path])
            for path, digest in file_hashes.items()
        )
        or not isinstance(ceremony, Mapping)
        or set(ceremony) != ceremony_fields
        or ceremony.get("schema_version") != CEREMONY_SCHEMA_VERSION
        or ceremony.get("network") != "test"
        or ceremony.get("netuid") != spec["netuid"]
        or ceremony.get("network_genesis_hash") != spec["network_genesis_hash"]
        or ceremony.get("mapping_hash") != spec["mapping_hash"]
        or ceremony.get("cutover_manifest_hash")
        != sha256_bytes(files["stateful-epoch-cutover.json"])
        or ceremony.get("authority_dump_hash")
        != sha256_bytes(files["authority.dump"])
        or ceremony.get("table_count") != len(tables)
        or ceremony.get("row_count") != sum(int(value) for value in row_counts.values())
        or not HASH_RE.fullmatch(
            str(ceremony.get("database_fingerprint_hash") or "")
        )
        or manifest.get("ceremony_evidence_hash")
        != sha256_bytes(
            json.dumps(
                dict(ceremony),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
            ).encode("ascii")
        )
    ):
        raise ProductionParityEpochAuthorityError(
            "epoch authority artifact manifest differs"
        )
    return manifest


def validate_archive(
    payload: bytes, spec: Mapping[str, Any]
) -> tuple[dict[str, bytes], dict[str, Any]]:
    files = _archive_files(payload)
    return files, _validate_files(files, spec)


def validate_installed(base: Path) -> dict[str, Any]:
    try:
        files = {name: (base / name).read_bytes() for name in ARCHIVE_PATHS}
    except OSError as exc:
        raise ProductionParityEpochAuthorityError(
            "installed epoch authority is incomplete"
        ) from exc
    manifest = _json(files["manifest.json"], field="installed epoch authority manifest")
    spec = {
        "netuid": manifest.get("netuid"),
        "mapping_hash": manifest.get("mapping_hash"),
        "network_genesis_hash": manifest.get("network_genesis_hash"),
    }
    normalized = _validate_files(files, spec)
    return {
        "network": "test",
        "netuid": normalized["netuid"],
        "mapping_hash": normalized["mapping_hash"],
        "network_genesis_hash": normalized["network_genesis_hash"],
        "database_tables": list(normalized["database_tables"]),
        "database_row_counts": dict(normalized["database_row_counts"]),
        "ceremony_evidence_hash": normalized["ceremony_evidence_hash"],
        "cutover_manifest_path": str(base / "stateful-epoch-cutover.json"),
        "database_archive_path": str(base / "authority.dump"),
    }


def _write_file(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
    path.parent.chmod(0o700)
    with path.open("xb") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    path.chmod(0o600)


def _remove_tree(path: Path) -> None:
    if not path.exists():
        return
    for item in sorted(path.rglob("*"), key=lambda value: len(value.parts), reverse=True):
        if item.is_file() or item.is_symlink():
            item.unlink()
        elif item.is_dir():
            item.rmdir()
    path.rmdir()


def install(
    spec: Mapping[str, Any], *, run_id: str, region: str, s3_client: Any = None
) -> dict[str, Any]:
    normalized = normalize_spec(
        spec,
        network=str(spec.get("network") or ""),
        netuid=int(spec.get("netuid") or 0),
    )
    if s3_client is None:
        import boto3

        s3_client = boto3.client("s3", region_name=region)
    request = {
        "Bucket": normalized["bucket"],
        "Key": normalized["key"],
        "VersionId": normalized["version_id"],
    }
    head_evidence = validate_head(normalized, s3_client.head_object(**request))
    response = s3_client.get_object(**request)
    body = response.get("Body")
    if body is None:
        raise ProductionParityEpochAuthorityError(
            "epoch authority artifact body is unavailable"
        )
    payload = body.read(MAX_ARCHIVE_BYTES + 1)
    try:
        body.close()
    except Exception:
        pass
    if sha256_bytes(payload) != normalized["sha256"]:
        raise ProductionParityEpochAuthorityError(
            "epoch authority artifact archive hash differs"
        )
    files, manifest = validate_archive(payload, normalized)
    base = install_base(run_id)
    staging = base.with_name(base.name + f".install-{os.getpid()}")
    if staging.exists():
        raise ProductionParityEpochAuthorityError(
            "epoch authority staging target already exists"
        )
    if base.exists():
        try:
            installed = {name: (base / name).read_bytes() for name in ARCHIVE_PATHS}
        except OSError as exc:
            raise ProductionParityEpochAuthorityError(
                "installed epoch authority is incomplete"
            ) from exc
        if installed != files:
            raise ProductionParityEpochAuthorityError(
                "installed epoch authority differs"
            )
    else:
        try:
            staging.mkdir(parents=True, mode=0o700)
            staging.chmod(0o700)
            for name, value in files.items():
                _write_file(staging / name, value)
            staging.rename(base)
        except Exception:
            _remove_tree(staging)
            raise
    cutover_path = cutover_manifest_path(run_id)
    return {
        "network": "test",
        "netuid": normalized["netuid"],
        "mapping_hash": normalized["mapping_hash"],
        "network_genesis_hash": normalized["network_genesis_hash"],
        "artifact_sha256": normalized["sha256"],
        "artifact_version_id": normalized["version_id"],
        "install_root": str(base),
        "cutover_manifest_path": str(cutover_path),
        "database_tables": list(manifest["database_tables"]),
        "database_row_counts": dict(manifest["database_row_counts"]),
        "ceremony_evidence_hash": manifest["ceremony_evidence_hash"],
        "head": head_evidence,
    }


__all__ = [
    "ProductionParityEpochAuthorityError",
    "SCHEMA_VERSION",
    "SPEC_SCHEMA_VERSION",
    "cutover_manifest_path",
    "install",
    "install_base",
    "normalize_spec",
    "sha256_bytes",
    "validate_archive",
    "validate_head",
    "validate_installed",
]
