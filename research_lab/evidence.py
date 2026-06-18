"""Evidence bundle emitter for notarized Research Lab snapshots."""

from __future__ import annotations

from typing import Any, Iterable, Optional
import uuid

from .canonical import coerce_iso_z, sha256_json, utc_now_iso
from .notary import SnapshotRecord
from .schema_validation import assert_schema_record


def build_evidence_bundle(
    *,
    artifact_hash: str,
    snapshots: Iterable[SnapshotRecord | dict[str, Any]],
    run_id: str | None = None,
    created_at: str | None = None,
    retention_class: str = "live_verification",
    verification_state: str = "active",
    merkle_anchor_ref: str | None = "arweave:pending",
    deletion_request_ref: str | None = None,
    validate: bool = True,
) -> dict[str, Any]:
    snapshot_records = [
        snap.to_schema_snapshot() if isinstance(snap, SnapshotRecord) else dict(snap)
        for snap in snapshots
    ]
    if not snapshot_records:
        raise ValueError("evidence bundle requires at least one snapshot")

    created = coerce_iso_z(created_at) if created_at else utc_now_iso()
    identity = {
        "run_id": run_id,
        "artifact_hash": artifact_hash,
        "snapshots": [
            {
                "url": snap["url"],
                "fetch_ts": snap["fetch_ts"],
                "content_hash": snap["content_hash"],
                "snapshot_ref": snap["snapshot_ref"],
            }
            for snap in snapshot_records
        ],
    }
    bundle_id = str(uuid.uuid5(uuid.NAMESPACE_URL, sha256_json(identity)))
    payload_without_hash = {
        "bundle_id": bundle_id,
        "schema_version": "1.0",
        "run_id": run_id,
        "artifact_hash": artifact_hash,
        "created_at": created,
        "retention_class": retention_class,
        "verification_state": verification_state,
        "merkle_anchor_ref": merkle_anchor_ref,
        "deletion_request_ref": deletion_request_ref,
        "snapshots": snapshot_records,
    }
    bundle = {
        **payload_without_hash,
        "bundle_hash": sha256_json(payload_without_hash),
    }
    assert_no_raw_content(bundle)
    if validate:
        assert_schema_record("evidence_bundle.schema.json", bundle)
    return bundle


def evidence_refs_from_bundle(
    bundle: dict[str, Any],
    *,
    signal_indices: Optional[dict[str, int | None]] = None,
) -> list[dict[str, Any]]:
    refs = []
    by_ref = signal_indices or {}
    for idx, snap in enumerate(bundle["snapshots"]):
        refs.append(
            {
                "url": snap["url"],
                "fetch_ts": snap["fetch_ts"],
                "content_hash": snap["content_hash"],
                "snapshot_ref": snap["snapshot_ref"],
                "signature": snap["signature"],
                "used_in_signal_idx": by_ref.get(snap["snapshot_ref"], idx),
            }
        )
    return refs


def assert_no_raw_content(bundle: dict[str, Any]) -> None:
    forbidden = {"content", "page_content", "raw_content", "normalized_text"}
    _assert_forbidden_keys_absent(bundle, forbidden, "$")


def _assert_forbidden_keys_absent(value: Any, forbidden: set[str], path: str) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if key in forbidden:
                raise ValueError(f"evidence bundle must not contain raw field {key!r} at {path}")
            _assert_forbidden_keys_absent(item, forbidden, f"{path}.{key}")
    elif isinstance(value, list):
        for idx, item in enumerate(value):
            _assert_forbidden_keys_absent(item, forbidden, f"{path}[{idx}]")
