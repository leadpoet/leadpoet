"""Local notary snapshot capture for Research Lab Phase 0."""

from __future__ import annotations

from dataclasses import dataclass
import hmac
import json
from pathlib import Path
from typing import Any, Optional

from .canonical import (
    canonical_json,
    coerce_iso_z,
    normalize_snapshot_text,
    sha256_json,
    sha256_text,
    utc_now_iso,
)


@dataclass(frozen=True)
class SnapshotRecord:
    url: str
    fetch_ts: str
    content_hash: str
    normalized_text_hash: str
    snapshot_ref: str
    signature: str
    l0_verification_state: str = "active"
    deleted_at: Optional[str] = None
    deletion_reason_ref: Optional[str] = None

    def to_schema_snapshot(self) -> dict[str, Any]:
        return {
            "url": self.url,
            "fetch_ts": self.fetch_ts,
            "content_hash": self.content_hash,
            "normalized_text_hash": self.normalized_text_hash,
            "snapshot_ref": self.snapshot_ref,
            "signature": self.signature,
            "l0_verification_state": self.l0_verification_state,
            "deleted_at": self.deleted_at,
            "deletion_reason_ref": self.deletion_reason_ref,
        }


class NotarySigner:
    """HMAC signer for local Phase 0 notary fixtures.

    Production should replace this with fabric/KMS signing. The signature
    prefix makes local signatures distinguishable from production fabric
    signatures while still giving deterministic tamper checks in tests.
    """

    def __init__(self, key: str | bytes, *, key_id: str = "local-dev") -> None:
        self.key = key.encode("utf-8") if isinstance(key, str) else key
        self.key_id = key_id

    def sign(self, payload: dict[str, Any]) -> str:
        digest = hmac.new(self.key, canonical_json(payload).encode("utf-8"), "sha256").hexdigest()
        return f"local_hmac:v1:{self.key_id}:{digest}"

    def verify(self, payload: dict[str, Any], signature: str) -> bool:
        return hmac.compare_digest(self.sign(payload), signature)


class LocalSnapshotStore:
    """Content-addressed local raw snapshot store.

    Structured corpus records store only hashes and refs. Raw content lives in
    this store so L0 replay can verify snippets without leaking page text into
    `evidence_bundles`.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def write(self, record: SnapshotRecord, *, content: str, metadata: dict[str, Any] | None = None) -> None:
        path = self.path_for_ref(record.snapshot_ref)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": "snapshot_store.v1",
            "record": record.to_schema_snapshot(),
            "content": content,
            "normalized_text": normalize_snapshot_text(content),
            "metadata": metadata or {},
        }
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True, indent=2)

    def read(self, snapshot_ref: str) -> dict[str, Any]:
        path = self.path_for_ref(snapshot_ref)
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def read_content(self, snapshot_ref: str) -> str:
        payload = self.read(snapshot_ref)
        content = payload.get("content")
        if content is None:
            raise ValueError(f"snapshot content deleted for {snapshot_ref}")
        return str(content)

    def delete_content(
        self,
        snapshot_ref: str,
        *,
        deletion_reason_ref: str,
        deleted_at: str | None = None,
    ) -> SnapshotRecord:
        payload = self.read(snapshot_ref)
        record_data = dict(payload["record"])
        deleted_ts = coerce_iso_z(deleted_at) if deleted_at else utc_now_iso()
        record_data["l0_verification_state"] = "content_deleted"
        record_data["deleted_at"] = deleted_ts
        record_data["deletion_reason_ref"] = deletion_reason_ref
        payload["record"] = record_data
        payload["content"] = None
        payload["normalized_text"] = None
        with self.path_for_ref(snapshot_ref).open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True, indent=2)
        return SnapshotRecord(**record_data)

    def path_for_ref(self, snapshot_ref: str) -> Path:
        safe_ref = snapshot_ref.strip().lstrip("/")
        path = (self.root / safe_ref).resolve()
        root = self.root.resolve()
        if root not in path.parents and path != root:
            raise ValueError(f"snapshot_ref escapes store root: {snapshot_ref}")
        return path


def capture_snapshot(
    *,
    url: str,
    content: str,
    store: LocalSnapshotStore,
    signer: NotarySigner,
    fetch_ts: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> SnapshotRecord:
    """Capture already-fetched content as a signed, content-addressed snapshot."""
    captured_at = coerce_iso_z(fetch_ts) if fetch_ts else utc_now_iso()
    normalized = normalize_snapshot_text(content)
    content_hash = sha256_text(content)
    normalized_hash = sha256_text(normalized)
    hash_hex = content_hash.split(":", 1)[1]
    day = captured_at[:10].replace("-", "/")
    snapshot_ref = f"snapshots/{day}/{hash_hex}.json"
    signing_payload = {
        "url": url,
        "fetch_ts": captured_at,
        "content_hash": content_hash,
        "normalized_text_hash": normalized_hash,
        "snapshot_ref": snapshot_ref,
        "metadata_hash": sha256_json(metadata or {}),
    }
    signature = signer.sign(signing_payload)
    record = SnapshotRecord(
        url=url,
        fetch_ts=captured_at,
        content_hash=content_hash,
        normalized_text_hash=normalized_hash,
        snapshot_ref=snapshot_ref,
        signature=signature,
    )
    store.write(record, content=content, metadata=metadata)
    return record
