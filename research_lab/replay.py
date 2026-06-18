"""Local L0 verifier replay over emitted evidence bundles."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from leadpoet_verifier import L0Result, run_l0_checks

from .canonical import normalize_snapshot_text, sha256_text
from .notary import LocalSnapshotStore


@dataclass(frozen=True)
class L0ReplayResult:
    snapshot_ref: str
    used_in_signal_idx: int | None
    verification_state: str
    result: L0Result | None

    @property
    def passed(self) -> bool:
        if self.verification_state != "active":
            return self.verification_state in {"content_deleted", "hash_attested"}
        return bool(self.result and self.result.passed)


def run_l0_replay(
    *,
    evidence_bundle: dict[str, Any],
    execution_trace: dict[str, Any],
    signals_by_index: Mapping[int, dict[str, Any]],
    snapshot_store: LocalSnapshotStore,
) -> list[L0ReplayResult]:
    """Run open-verifier L0 checks against stored snapshots and trace refs."""
    bundle_by_ref = {
        snap["snapshot_ref"]: snap
        for snap in evidence_bundle.get("snapshots", [])
    }
    results: list[L0ReplayResult] = []
    for evidence_ref in execution_trace.get("evidence_bundles", []):
        snapshot_ref = evidence_ref["snapshot_ref"]
        bundle_snap = bundle_by_ref[snapshot_ref]
        signal_idx = evidence_ref.get("used_in_signal_idx")
        state = bundle_snap.get("l0_verification_state", "active")
        if state != "active":
            results.append(
                L0ReplayResult(
                    snapshot_ref=snapshot_ref,
                    used_in_signal_idx=signal_idx,
                    verification_state=state,
                    result=None,
                )
            )
            continue
        if signal_idx is None or signal_idx not in signals_by_index:
            raise ValueError(f"missing signal for snapshot_ref {snapshot_ref}")
        content = snapshot_store.read_content(snapshot_ref)
        _verify_snapshot_integrity(snapshot_ref, bundle_snap, content)
        snapshot = {"url": bundle_snap["url"], "content": content}
        result = run_l0_checks(signals_by_index[signal_idx], snapshot)
        results.append(
            L0ReplayResult(
                snapshot_ref=snapshot_ref,
                used_in_signal_idx=signal_idx,
                verification_state=state,
                result=result,
            )
        )
    return results


def _verify_snapshot_integrity(
    snapshot_ref: str,
    bundle_snap: dict[str, Any],
    content: str,
) -> None:
    content_hash = sha256_text(content)
    if content_hash != bundle_snap.get("content_hash"):
        raise ValueError(
            f"snapshot content hash mismatch for {snapshot_ref}: "
            f"{content_hash} != {bundle_snap.get('content_hash')}"
        )
    normalized_hash = sha256_text(normalize_snapshot_text(content))
    if normalized_hash != bundle_snap.get("normalized_text_hash"):
        raise ValueError(
            f"snapshot normalized text hash mismatch for {snapshot_ref}: "
            f"{normalized_hash} != {bundle_snap.get('normalized_text_hash')}"
        )
