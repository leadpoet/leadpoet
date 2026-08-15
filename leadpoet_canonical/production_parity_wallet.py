"""Immutable testnet wallet artifacts for disposable production-parity hosts."""

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


SCHEMA_VERSION = "leadpoet.production_parity_wallet_artifact.v1"
SPEC_SCHEMA_VERSION = "leadpoet.production_parity_wallet_spec.v1"
RUN_RE = re.compile(r"^[a-z0-9-]{6,40}$")
HASH_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
SS58_RE = re.compile(r"^[1-9A-HJ-NP-Za-km-z]{40,64}$")
SAFE_NAME_RE = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")
S3_RE = re.compile(r"^s3://([A-Za-z0-9][A-Za-z0-9.-]{1,62})/([^\s]+)$")
VERSION_RE = re.compile(r"^[A-Za-z0-9_.=+-]{1,1024}$")
ROLE_NAMES = {"primary-validator", "auditor-a", "auditor-b"}
MAX_ARCHIVE_BYTES = 16 * 1024 * 1024
MAX_EXPANDED_BYTES = 4 * 1024 * 1024


class ProductionParityWalletError(RuntimeError):
    pass


def sha256_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def install_base(run_id: str, role: str) -> Path:
    if not RUN_RE.fullmatch(run_id) or role not in ROLE_NAMES:
        raise ProductionParityWalletError("wallet install identity is invalid")
    return Path("/home/ec2-user/.config/leadpoet/parity") / run_id / role


def normalize_spec(
    value: Mapping[str, Any], *, role: str, network: str, netuid: int
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or role not in ROLE_NAMES:
        raise ProductionParityWalletError("wallet artifact spec is invalid")
    document = dict(value)
    if document.get("schema_version") != SPEC_SCHEMA_VERSION:
        raise ProductionParityWalletError("wallet artifact spec schema differs")
    uri = str(document.get("s3_uri") or "")
    match = S3_RE.fullmatch(uri)
    version_id = str(document.get("version_id") or "")
    digest = str(document.get("sha256") or "").lower()
    kms_key_arn = str(document.get("kms_key_arn") or "")
    wallet_name = str(document.get("wallet_name") or "")
    wallet_hotkey = str(document.get("wallet_hotkey") or "")
    expected_hotkey = str(document.get("expected_hotkey") or "")
    if (
        match is None
        or not VERSION_RE.fullmatch(version_id)
        or not HASH_RE.fullmatch(digest)
        or not kms_key_arn.startswith("arn:aws:kms:")
        or not SAFE_NAME_RE.fullmatch(wallet_name)
        or not SAFE_NAME_RE.fullmatch(wallet_hotkey)
        or not SS58_RE.fullmatch(expected_hotkey)
        or network != "test"
        or not isinstance(netuid, int)
        or isinstance(netuid, bool)
        or netuid <= 0
    ):
        raise ProductionParityWalletError("wallet artifact spec fields are invalid")
    return {
        "schema_version": SPEC_SCHEMA_VERSION,
        "role": role,
        "network": "test",
        "netuid": netuid,
        "s3_uri": uri,
        "bucket": match.group(1),
        "key": match.group(2),
        "version_id": version_id,
        "sha256": digest,
        "kms_key_arn": kms_key_arn,
        "wallet_name": wallet_name,
        "wallet_hotkey": wallet_hotkey,
        "expected_hotkey": expected_hotkey,
    }


def validate_head(
    spec: Mapping[str, Any], head: Mapping[str, Any], *, now: datetime | None = None
) -> dict[str, Any]:
    current = now or datetime.now(timezone.utc)
    retained_until = head.get("ObjectLockRetainUntilDate")
    if isinstance(retained_until, str):
        try:
            retained_until = datetime.fromisoformat(retained_until.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ProductionParityWalletError(
                "wallet artifact retention timestamp is invalid"
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
        raise ProductionParityWalletError(
            "wallet artifact is not immutable, KMS-bound, and size-bounded"
        )
    return {
        "version_id": spec["version_id"],
        "kms_key_arn": spec["kms_key_arn"],
        "object_lock_mode": str(head["ObjectLockMode"]),
        "retain_until": retained_until.astimezone(timezone.utc).isoformat(),
        "size_bytes": size,
    }


def _expected_paths(spec: Mapping[str, Any]) -> set[str]:
    paths = {"manifest.json", "wallet/coldkeypub.txt"}
    if spec["role"] == "primary-validator":
        paths.update(
            {
                "validator-hotkey-config-v2.json",
                "validator-hotkey-envelope-v2.json",
            }
        )
    else:
        paths.add(f"wallet/hotkeys/{spec['wallet_hotkey']}")
    return paths


def _archive_files(payload: bytes, spec: Mapping[str, Any]) -> dict[str, bytes]:
    if not payload or len(payload) > MAX_ARCHIVE_BYTES:
        raise ProductionParityWalletError("wallet artifact archive size is invalid")
    files: dict[str, bytes] = {}
    expanded = 0
    try:
        with tarfile.open(fileobj=BytesIO(payload), mode="r:*") as archive:
            for member in archive.getmembers():
                path = PurePosixPath(member.name)
                name = str(path)
                if (
                    member.isdir()
                    or name in {".", ""}
                    or path.is_absolute()
                    or ".." in path.parts
                ):
                    if member.isdir():
                        continue
                    raise ProductionParityWalletError(
                        "wallet artifact contains an unsafe path"
                    )
                if not member.isfile() or name in files:
                    raise ProductionParityWalletError(
                        "wallet artifact contains a link, special file, or duplicate"
                    )
                extracted = archive.extractfile(member)
                if extracted is None:
                    raise ProductionParityWalletError(
                        "wallet artifact member is unreadable"
                    )
                value = extracted.read(MAX_EXPANDED_BYTES + 1)
                expanded += len(value)
                if len(value) > MAX_EXPANDED_BYTES or expanded > MAX_EXPANDED_BYTES:
                    raise ProductionParityWalletError(
                        "wallet artifact expanded size exceeds limit"
                    )
                files[name] = value
    except (tarfile.TarError, OSError) as exc:
        raise ProductionParityWalletError("wallet artifact archive is invalid") from exc
    if set(files) != _expected_paths(spec):
        raise ProductionParityWalletError("wallet artifact file inventory differs")
    return files


def _json(value: bytes, *, field: str) -> dict[str, Any]:
    try:
        document = json.loads(value.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise ProductionParityWalletError(f"{field} is invalid") from exc
    if not isinstance(document, dict):
        raise ProductionParityWalletError(f"{field} must be an object")
    return document


def validate_archive(payload: bytes, spec: Mapping[str, Any]) -> dict[str, bytes]:
    files = _archive_files(payload, spec)
    manifest = _json(files["manifest.json"], field="wallet artifact manifest")
    expected_manifest_fields = {
        "schema_version",
        "role",
        "network",
        "netuid",
        "wallet_name",
        "wallet_hotkey",
        "expected_hotkey",
        "files",
    }
    file_hashes = manifest.get("files")
    if (
        set(manifest) != expected_manifest_fields
        or manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("role") != spec["role"]
        or manifest.get("network") != "test"
        or manifest.get("netuid") != spec["netuid"]
        or manifest.get("wallet_name") != spec["wallet_name"]
        or manifest.get("wallet_hotkey") != spec["wallet_hotkey"]
        or manifest.get("expected_hotkey") != spec["expected_hotkey"]
        or not isinstance(file_hashes, Mapping)
        or set(file_hashes) != (set(files) - {"manifest.json"})
        or any(
            not HASH_RE.fullmatch(str(digest or ""))
            or str(digest) != sha256_bytes(files[path])
            for path, digest in file_hashes.items()
        )
    ):
        raise ProductionParityWalletError("wallet artifact manifest differs")
    coldkey = _json(files["wallet/coldkeypub.txt"], field="wallet coldkey")
    if not SS58_RE.fullmatch(str(coldkey.get("ss58Address") or "")):
        raise ProductionParityWalletError("wallet coldkey identity is invalid")
    if spec["role"] == "primary-validator":
        from validator_tee.enclave.hotkey_authority_v2 import (
            validate_hotkey_authority_configuration,
        )
        from validator_tee.host.hotkey_bootstrap_v2 import validate_hotkey_envelope

        config = validate_hotkey_authority_configuration(
            _json(
                files["validator-hotkey-config-v2.json"],
                field="validator hotkey config",
            )
        )
        envelope = validate_hotkey_envelope(
            _json(
                files["validator-hotkey-envelope-v2.json"],
                field="validator hotkey envelope",
            )
        )
        if (
            config["validator_hotkey"] != spec["expected_hotkey"]
            or envelope["validator_hotkey"] != spec["expected_hotkey"]
            or config["hotkey_public_key"] != envelope["hotkey_public_key"]
        ):
            raise ProductionParityWalletError(
                "primary validator wallet identity differs"
            )
    else:
        hotkey = _json(
            files[f"wallet/hotkeys/{spec['wallet_hotkey']}"],
            field="auditor hotkey",
        )
        if str(hotkey.get("ss58Address") or "") != spec["expected_hotkey"]:
            raise ProductionParityWalletError("auditor hotkey identity differs")
    return files


def _write_file(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
    path.parent.chmod(0o700)
    with path.open("xb") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    path.chmod(0o600)


def install(
    spec: Mapping[str, Any], *, run_id: str, region: str, s3_client: Any = None
) -> dict[str, Any]:
    normalized = normalize_spec(
        spec,
        role=str(spec.get("role") or ""),
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
    head = s3_client.head_object(**request)
    head_evidence = validate_head(normalized, head)
    response = s3_client.get_object(**request)
    body = response.get("Body")
    if body is None:
        raise ProductionParityWalletError("wallet artifact body is unavailable")
    payload = body.read(MAX_ARCHIVE_BYTES + 1)
    try:
        body.close()
    except Exception:
        pass
    if sha256_bytes(payload) != normalized["sha256"]:
        raise ProductionParityWalletError("wallet artifact archive hash differs")
    files = validate_archive(payload, normalized)
    base = install_base(run_id, normalized["role"])
    staging = base.with_name(base.name + f".install-{os.getpid()}")
    if base.exists() or staging.exists():
        raise ProductionParityWalletError("wallet artifact target already exists")
    try:
        staging.mkdir(parents=True, mode=0o700)
        staging.chmod(0o700)
        wallet = staging / "wallets" / normalized["wallet_name"]
        _write_file(wallet / "coldkeypub.txt", files["wallet/coldkeypub.txt"])
        hotkeys = wallet / "hotkeys"
        hotkeys.mkdir(parents=True, mode=0o700)
        hotkeys.chmod(0o700)
        if normalized["role"] == "primary-validator":
            _write_file(
                staging / "validator-hotkey-config-v2.json",
                files["validator-hotkey-config-v2.json"],
            )
            _write_file(
                staging / "validator-hotkey-envelope-v2.json",
                files["validator-hotkey-envelope-v2.json"],
            )
        else:
            _write_file(
                hotkeys / normalized["wallet_hotkey"],
                files[f"wallet/hotkeys/{normalized['wallet_hotkey']}"],
            )
        staging.rename(base)
    except Exception:
        if staging.exists():
            for path in sorted(
                staging.rglob("*"),
                key=lambda value: len(value.parts),
                reverse=True,
            ):
                if path.is_file():
                    path.unlink()
                elif path.is_dir():
                    path.rmdir()
            staging.rmdir()
        raise
    return {
        "role": normalized["role"],
        "network": "test",
        "netuid": normalized["netuid"],
        "expected_hotkey": normalized["expected_hotkey"],
        "wallet_root": str(base / "wallets"),
        "wallet_name": normalized["wallet_name"],
        "wallet_hotkey": normalized["wallet_hotkey"],
        "artifact_sha256": normalized["sha256"],
        "artifact_version_id": normalized["version_id"],
        "head": head_evidence,
    }


__all__ = [
    "ProductionParityWalletError",
    "SCHEMA_VERSION",
    "SPEC_SCHEMA_VERSION",
    "install",
    "install_base",
    "normalize_spec",
    "sha256_bytes",
    "validate_archive",
    "validate_head",
]
