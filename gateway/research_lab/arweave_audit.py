"""Research Lab Arweave audit anchoring helpers."""

from __future__ import annotations

import logging
import os
from typing import Any, Mapping, Sequence

from .bundles import contains_secret_material, sha256_json
from .config import ResearchLabGatewayConfig
from .store import (
    canonical_hash,
    create_arweave_epoch_audit_anchor,
    create_arweave_epoch_audit_anchor_event,
    select_all,
    select_many,
    select_one,
)


RESEARCH_LAB_EPOCH_AUDIT_EVENT_TYPE = "RESEARCH_LAB_EPOCH_AUDIT"
logger = logging.getLogger(__name__)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return int(default)


def _audit_select_limits() -> tuple[int, int]:
    max_rows = max(1000, _env_int("RESEARCH_LAB_ARWEAVE_AUDIT_SELECT_MAX_ROWS", 10000))
    batch_size = max(
        100,
        min(max_rows, _env_int("RESEARCH_LAB_ARWEAVE_AUDIT_SELECT_BATCH_SIZE", 500)),
    )
    return max_rows, batch_size


def _normalize_sha256_ref(value: object) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    return raw if raw.startswith("sha256:") else f"sha256:{raw}"


async def publish_research_lab_epoch_audit(
    *,
    epoch: int,
    netuid: int,
    audit_kind: str,
    actor_hotkey: str | None = None,
    weight_bundle: Mapping[str, Any] | None = None,
    config: ResearchLabGatewayConfig | None = None,
) -> dict[str, Any] | None:
    """Publish one compact Research Lab epoch audit event to the Arweave buffer."""
    config = config or ResearchLabGatewayConfig.from_env()
    if audit_kind == "active" and not config.arweave_audit_enabled:
        return None
    if audit_kind == "shadow" and not config.arweave_audit_shadow_enabled:
        return None
    if audit_kind not in {"active", "shadow"}:
        raise ValueError("audit_kind must be active or shadow")

    payload = await build_research_lab_epoch_audit_payload(
        epoch=epoch,
        netuid=netuid,
        audit_kind=audit_kind,
        actor_hotkey=actor_hotkey,
        weight_bundle=weight_bundle,
    )
    payload_hash = sha256_json(payload)
    existing = await _existing_anchor_for_payload(
        epoch=epoch,
        netuid=netuid,
        audit_kind=audit_kind,
        payload_hash=payload_hash,
    )
    if existing:
        return existing

    from gateway.utils.logger import log_event

    log_entry = await log_event(RESEARCH_LAB_EPOCH_AUDIT_EVENT_TYPE, payload)
    signed_event = log_entry.get("signed_event") if isinstance(log_entry.get("signed_event"), Mapping) else {}
    anchor, event = await create_arweave_epoch_audit_anchor(
        epoch=epoch,
        netuid=netuid,
        audit_kind=audit_kind,
        audit_bundle_id=payload["audit_bundle"].get("audit_bundle_id"),
        audit_bundle_hash=payload["audit_bundle"].get("audit_bundle_hash"),
        allocation_hash=payload["lab_allocation"].get("allocation_hash"),
        weights_hash=payload["weights"].get("weights_hash"),
        payload_hash=payload_hash,
        transparency_event_hash=log_entry.get("event_hash"),
        tee_sequence=log_entry.get("tee_sequence"),
        event_doc={
            "event_hash": log_entry.get("event_hash"),
            "tee_sequence": log_entry.get("tee_sequence"),
            "tee_buffer_size": log_entry.get("tee_buffer_size"),
            "monotonic_seq": signed_event.get("monotonic_seq"),
            "boot_id": signed_event.get("boot_id"),
        },
    )
    return {"anchor": anchor, "event": event, "payload": payload}


async def build_research_lab_epoch_audit_payload(
    *,
    epoch: int,
    netuid: int,
    audit_kind: str,
    actor_hotkey: str | None = None,
    weight_bundle: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    audit_bundle = await _latest_signed_audit_bundle(epoch)
    allocation = await _latest_lab_allocation(epoch=epoch, netuid=netuid)
    weight_bundle = dict(weight_bundle or await _latest_weight_bundle(epoch=epoch, netuid=netuid) or {})
    actor_hotkey = actor_hotkey or weight_bundle.get("validator_hotkey") or "system"
    score_bundles = await _audit_select_all(
        "research_evaluation_score_bundle_current",
        filters=(("evaluation_epoch", epoch),),
        order_by=(("created_at", True),),
    )
    benchmarks = await _audit_select_all(
        "research_lab_private_model_benchmark_current",
        filters=(("evaluation_epoch", epoch),),
        order_by=(("created_at", True),),
    )
    windows = await _audit_select_all("research_lab_rolling_icp_windows", order_by=(("created_at", True),))
    model_versions = await _audit_select_all(
        "research_lab_private_model_version_current",
        order_by=(("created_at", True),),
    )
    promotions = await _audit_select_all(
        "research_lab_candidate_promotion_events",
        order_by=(("created_at", True),),
    )
    public_reports = await _audit_select_all(
        "research_lab_public_benchmark_report_current",
        order_by=(("created_at", True),),
    )
    champion_rewards = await _audit_select_all(
        "research_lab_champion_reward_current",
        order_by=(("created_at", True),),
    )
    reimbursement_awards = await _audit_select_all(
        "research_reimbursement_award_current",
        order_by=(("created_at", True),),
    )

    payload = {
        "schema_version": "1.0",
        "event_type": RESEARCH_LAB_EPOCH_AUDIT_EVENT_TYPE,
        "actor_hotkey": actor_hotkey,
        "epoch": int(epoch),
        "netuid": int(netuid),
        "audit_kind": audit_kind,
        "audit_bundle": _audit_bundle_ref(audit_bundle),
        "weights": _weight_ref(weight_bundle),
        "lab_allocation": _allocation_ref(allocation),
        "score_bundles": [_score_bundle_ref(row) for row in score_bundles],
        "private_baseline_benchmarks": [_benchmark_ref(row) for row in benchmarks],
        "rolling_icp_windows": [_rolling_window_ref(row) for row in windows],
        "private_model_versions": [_model_version_ref(row) for row in model_versions],
        "promotion_events": [_promotion_ref(row) for row in promotions],
        "public_benchmark_reports": [_public_report_ref(row) for row in public_reports],
        "champion_rewards": [_champion_reward_ref(row) for row in champion_rewards],
        "reimbursement_awards": [_reimbursement_ref(row) for row in reimbursement_awards],
        "observability": {
            "score_bundle_count": len(score_bundles),
            "private_baseline_benchmark_count": len(benchmarks),
            "rolling_icp_window_count": len(windows),
            "private_model_version_count": len(model_versions),
            "promotion_event_count": len(promotions),
            "public_benchmark_report_count": len(public_reports),
            "champion_reward_count": len(champion_rewards),
            "reimbursement_award_count": len(reimbursement_awards),
        },
        "verifier_contract": {
            "arweave_payload": "compact_hash_refs_only",
            "private_model_recompute": "forbidden_for_main_validators_v1",
            "required_checks": [
                "no_secret_material",
                "audit_bundle_hash_matches",
                "allocation_hash_matches",
                "score_bundle_hashes_match",
                "weights_hash_matches",
                "transparency_event_in_arweave_checkpoint",
            ],
        },
    }
    payload["payload_hash"] = sha256_json(payload)
    if contains_secret_material(payload):
        raise ValueError("Research Lab Arweave audit payload contains private or secret material")
    return payload


async def _audit_select_all(
    table: str,
    *,
    filters: tuple[tuple[Any, ...], ...] = (),
    order_by: tuple[tuple[str, bool], ...] = (("created_at", True),),
) -> list[dict[str, Any]]:
    """Fetch audit rows with deterministic pagination instead of PostgREST defaults."""

    max_rows, batch_size = _audit_select_limits()
    try:
        return await select_all(
            table,
            filters=filters,
            order_by=order_by,
            max_rows=max_rows,
            batch_size=batch_size,
            allow_partial=True,
        )
    except Exception as exc:
        logger.warning(
            "research_lab_arweave_audit_select_order_fallback table=%s order=%s error=%s",
            table,
            order_by,
            str(exc)[:200],
        )
        return await select_all(
            table,
            filters=filters,
            max_rows=max_rows,
            batch_size=batch_size,
            allow_partial=True,
        )


async def record_research_lab_checkpointed_events(
    *,
    events: Sequence[Mapping[str, Any]],
    header: Mapping[str, Any],
    arweave_tx_id: str,
) -> int:
    recorded = 0
    for event in events:
        if event.get("event_type") != RESEARCH_LAB_EPOCH_AUDIT_EVENT_TYPE:
            continue
        log_entry = event.get("signed_log_entry") if isinstance(event.get("signed_log_entry"), Mapping) else {}
        signed_event = log_entry.get("signed_event") if isinstance(log_entry.get("signed_event"), Mapping) else {}
        payload = signed_event.get("payload") if isinstance(signed_event.get("payload"), Mapping) else {}
        transparency_event_hash = str(event.get("event_hash") or log_entry.get("event_hash") or "")
        anchor = None
        if transparency_event_hash:
            anchor = await select_one(
                "research_lab_arweave_epoch_audit_anchor_current",
                filters=(("current_transparency_event_hash", transparency_event_hash),),
            )
        if not anchor:
            payload_hash_candidates: list[str] = []
            for candidate in (
                event.get("payload_hash"),
                log_entry.get("payload_hash"),
                payload.get("payload_hash"),
                sha256_json(payload) if payload else "",
            ):
                payload_hash = _normalize_sha256_ref(candidate)
                if payload_hash and payload_hash not in payload_hash_candidates:
                    payload_hash_candidates.append(payload_hash)
            for payload_hash in payload_hash_candidates:
                anchor = await _existing_anchor_for_payload(
                    epoch=int(payload.get("epoch", 0)),
                    netuid=int(payload.get("netuid", 0)),
                    audit_kind=str(payload.get("audit_kind", "")),
                    payload_hash=payload_hash,
                )
                if anchor:
                    break
        if not anchor:
            continue
        await create_arweave_epoch_audit_anchor_event(
            anchor_id=str(anchor["anchor_id"]),
            event_type="checkpointed",
            anchor_status="checkpointed",
            transparency_event_hash=transparency_event_hash or None,
            tee_sequence=event.get("sequence"),
            checkpoint_number=header.get("checkpoint_number"),
            checkpoint_merkle_root=header.get("merkle_root"),
            arweave_tx_id=arweave_tx_id,
            event_doc={
                "arweave_tx_id": arweave_tx_id,
                "checkpoint_number": header.get("checkpoint_number"),
                "checkpoint_merkle_root": header.get("merkle_root"),
                "checkpoint_sequence_range": header.get("sequence_range"),
                "checkpoint_event_count": header.get("event_count"),
                "tee_sequence": event.get("sequence"),
                "transparency_event_hash": transparency_event_hash,
            },
        )
        recorded += 1
    return recorded


async def rebuffer_research_lab_buffered_audit_events(*, limit: int = 200) -> int:
    """Rehydrate DB-buffered Research Lab audit events into the TEE buffer.

    The enclave buffer is process/runtime state. If the gateway restarts after
    ``publish_research_lab_epoch_audit`` records a buffered anchor but before
    the hourly checkpoint includes it, the DB still says "buffered" while the
    TEE buffer can lose that event. Re-append missing signed transparency-log
    events so the next checkpoint can mark the anchors ``checkpointed``.
    """
    rows = await select_many(
        "research_lab_arweave_epoch_audit_anchor_current",
        columns="anchor_id,payload_hash,transparency_event_hash,current_transparency_event_hash,current_anchor_status,current_status_at",
        filters=(("current_anchor_status", "eq", "buffered"),),
        order_by=(("current_status_at", False),),
        limit=max(1, int(limit)),
    )
    if not rows:
        return 0

    try:
        from gateway.utils.tee_client import tee_client
    except ImportError:
        from utils.tee_client import tee_client

    existing_event_hashes: set[str] = set()
    try:
        current_buffer = await tee_client.get_buffer()
        if isinstance(current_buffer, list):
            for event in current_buffer:
                if isinstance(event, Mapping):
                    event_hash = str(event.get("event_hash") or "")
                    if event_hash:
                        existing_event_hashes.add(event_hash)
    except Exception as exc:  # noqa: BLE001 - rebuffering is best-effort
        logger.warning(
            "research_lab_arweave_rebuffer_existing_buffer_scan_failed error=%s",
            str(exc)[:240],
        )

    rebuffered = 0
    for row in rows:
        event_hash = str(
            row.get("current_transparency_event_hash")
            or row.get("transparency_event_hash")
            or ""
        )
        if not event_hash:
            continue
        if event_hash in existing_event_hashes:
            continue
        log_row = await select_one(
            "transparency_log",
            columns="event_type,event_hash,payload_hash,signed_log_entry",
            filters=(("event_hash", event_hash),),
        )
        if not log_row:
            logger.warning(
                "research_lab_arweave_rebuffer_missing_transparency_event anchor_id=%s event_hash=%s",
                row.get("anchor_id"),
                event_hash[:16],
            )
            continue
        signed_log_entry = log_row.get("signed_log_entry")
        if not isinstance(signed_log_entry, Mapping):
            logger.warning(
                "research_lab_arweave_rebuffer_missing_signed_log_entry anchor_id=%s event_hash=%s",
                row.get("anchor_id"),
                event_hash[:16],
            )
            continue
        event_type = str(log_row.get("event_type") or RESEARCH_LAB_EPOCH_AUDIT_EVENT_TYPE)
        await tee_client.append_event(
            {
                "event_type": event_type,
                "event_hash": event_hash,
                "payload_hash": log_row.get("payload_hash") or row.get("payload_hash"),
                "signed_log_entry": signed_log_entry,
            }
        )
        existing_event_hashes.add(event_hash)
        rebuffered += 1
    if rebuffered:
        logger.info(
            "research_lab_arweave_audit_events_rebuffered count=%s limit=%s",
            rebuffered,
            limit,
        )
    return rebuffered


async def latest_arweave_anchor(epoch: int, netuid: int | None = None) -> dict[str, Any] | None:
    filters: tuple[tuple[str, Any], ...]
    if netuid is None:
        filters = (("epoch", epoch),)
    else:
        filters = (("epoch", epoch), ("netuid", netuid))
    rows = await select_many(
        "research_lab_arweave_epoch_audit_anchor_current",
        filters=filters,
        order_by=(("created_at", True),),
        limit=1,
    )
    return rows[0] if rows else None


async def _existing_anchor_for_payload(
    *,
    epoch: int,
    netuid: int,
    audit_kind: str,
    payload_hash: str,
) -> dict[str, Any] | None:
    anchor_hash = canonical_hash(
        {
            "epoch": int(epoch),
            "netuid": int(netuid),
            "audit_kind": audit_kind,
            "payload_hash": payload_hash,
        }
    )
    anchor_id = "research_lab_arweave_anchor:" + anchor_hash.split(":", 1)[1]
    return await select_one("research_lab_arweave_epoch_audit_anchors", filters=(("anchor_id", anchor_id),))


async def _latest_signed_audit_bundle(epoch: int) -> dict[str, Any]:
    rows = await select_many(
        "research_lab_signed_audit_bundle_current",
        filters=(("epoch", epoch),),
        order_by=(("created_at", True),),
        limit=1,
    )
    return rows[0] if rows else {}


async def _latest_lab_allocation(epoch: int, netuid: int) -> dict[str, Any]:
    rows = await select_many(
        "research_lab_emission_allocation_current",
        filters=(("epoch", epoch), ("netuid", netuid)),
        order_by=(("created_at", True),),
        limit=1,
    )
    return rows[0] if rows else {}


async def _latest_weight_bundle(epoch: int, netuid: int) -> dict[str, Any] | None:
    try:
        from gateway.db.client import get_write_client
    except ImportError:
        from db.client import get_write_client

    def _call() -> Any:
        return (
            get_write_client()
            .table("published_weight_bundles")
            .select("*")
            .eq("epoch_id", int(epoch))
            .eq("netuid", int(netuid))
            .order("created_at", desc=True)
            .limit(1)
            .execute()
        )

    import asyncio

    response = await asyncio.to_thread(_call)
    data = getattr(response, "data", None) or []
    return dict(data[0]) if data else None


def _audit_bundle_ref(row: Mapping[str, Any]) -> dict[str, Any]:
    return _compact_ref(
        row,
        (
            "audit_bundle_id",
            "audit_bundle_hash",
            "epoch",
            "signature_ref",
            "current_audit_status",
            "current_event_hash",
            "anchored_hash",
        ),
    )


def _weight_ref(row: Mapping[str, Any]) -> dict[str, Any]:
    return _compact_ref(
        row,
        (
            "netuid",
            "epoch_id",
            "block",
            "weights_hash",
            "validator_hotkey",
            "validator_pcr0",
            "pcr0_commit_hash",
            "chain_snapshot_block",
            "chain_snapshot_compare_hash",
            "weight_submission_event_hash",
        ),
    )


def _allocation_ref(row: Mapping[str, Any]) -> dict[str, Any]:
    allocation_doc = row.get("allocation_doc") if isinstance(row.get("allocation_doc"), Mapping) else {}
    return {
        **_compact_ref(
            row,
            (
                "allocation_id",
                "epoch",
                "netuid",
                "policy_id",
                "snapshot_status",
                "lab_cap_alpha_percent",
                "reimbursement_alpha_percent",
                "champion_alpha_percent",
                "queued_champion_alpha_percent",
                "unallocated_alpha_percent",
                "input_hash",
                "allocation_hash",
            ),
        ),
        "allocations": {
            "reimbursements": allocation_doc.get("reimbursement_allocations", []),
            "champions": allocation_doc.get("champion_allocations", []),
            "queued_champions": allocation_doc.get("queued_champion_allocations", []),
        },
    }


def _score_bundle_ref(row: Mapping[str, Any]) -> dict[str, Any]:
    doc = row.get("score_bundle_doc") if isinstance(row.get("score_bundle_doc"), Mapping) else {}
    aggregates = doc.get("aggregates") if isinstance(doc.get("aggregates"), Mapping) else {}
    return {
        **_compact_ref(
            row,
            (
                "score_bundle_id",
                "run_id",
                "ticket_id",
                "receipt_id",
                "miner_hotkey",
                "island",
                "evaluation_epoch",
                "bundle_status",
                "parent_artifact_hash",
                "candidate_artifact_hash",
                "private_model_manifest_hash",
                "candidate_patch_hash",
                "icp_set_hash",
                "score_bundle_hash",
                "anchored_hash",
                "signature_ref",
                "current_event_status",
            ),
        ),
        "aggregates": _compact_ref(
            aggregates,
            ("mean_base_score", "mean_candidate_score", "mean_delta", "delta_lcb", "icp_count"),
        ),
    }


def _benchmark_ref(row: Mapping[str, Any]) -> dict[str, Any]:
    return _compact_ref(
        row,
        (
            "benchmark_bundle_id",
            "benchmark_date",
            "private_model_artifact_hash",
            "private_model_manifest_hash",
            "rolling_window_hash",
            "evaluation_epoch",
            "aggregate_score",
            "scoring_worker_ref",
            "proxy_ref_hash",
            "signature_ref",
            "benchmark_bundle_hash",
            "anchored_hash",
            "current_benchmark_status",
        ),
    )


def _rolling_window_ref(row: Mapping[str, Any]) -> dict[str, Any]:
    return _compact_ref(
        row,
        (
            "rolling_window_hash",
            "required_days",
            "icps_per_day",
            "selected_set_count",
            "selected_icp_count",
            "anchored_hash",
        ),
    )


def _model_version_ref(row: Mapping[str, Any]) -> dict[str, Any]:
    return _compact_ref(
        row,
        (
            "private_model_version_id",
            "model_artifact_hash",
            "private_model_manifest_hash",
            "git_commit_sha",
            "config_hash",
            "component_registry_version",
            "scoring_adapter_version",
            "source_candidate_id",
            "source_score_bundle_id",
            "source_benchmark_bundle_id",
            "version_hash",
            "anchored_hash",
            "current_version_status",
        ),
    )


def _promotion_ref(row: Mapping[str, Any]) -> dict[str, Any]:
    return _compact_ref(
        row,
        (
            "promotion_event_id",
            "candidate_id",
            "derived_candidate_id",
            "source_score_bundle_id",
            "derived_score_bundle_id",
            "private_model_version_id",
            "event_type",
            "promotion_status",
            "active_parent_artifact_hash",
            "candidate_parent_artifact_hash",
            "rolling_window_hash",
            "improvement_points",
            "threshold_points",
            "anchored_hash",
        ),
    )


def _public_report_ref(row: Mapping[str, Any]) -> dict[str, Any]:
    report_doc = row.get("report_doc") if isinstance(row.get("report_doc"), Mapping) else {}
    return {
        **_compact_ref(
            row,
            (
                "report_id",
                "benchmark_date",
                "benchmark_bundle_id",
                "private_model_artifact_hash",
                "private_model_manifest_hash",
                "rolling_window_hash",
                "aggregate_score",
                "report_hash",
                "anchored_hash",
                "current_report_status",
            ),
        ),
        "public_summary_hash": sha256_json(report_doc) if report_doc else None,
    }


def _champion_reward_ref(row: Mapping[str, Any]) -> dict[str, Any]:
    return _compact_ref(
        row,
        (
            "champion_reward_id",
            "score_bundle_id",
            "candidate_id",
            "run_id",
            "ticket_id",
            "miner_hotkey",
            "miner_uid",
            "island",
            "policy_id",
            "evaluation_epoch",
            "start_epoch",
            "epoch_count",
            "improvement_points",
            "threshold_points",
            "desired_alpha_percent",
            "source_score_bundle_hash",
            "anchored_hash",
            "current_reward_status",
            "current_event_hash",
        ),
    )


def _reimbursement_ref(row: Mapping[str, Any]) -> dict[str, Any]:
    return _compact_ref(
        row,
        (
            "award_id",
            "run_id",
            "ticket_id",
            "receipt_id",
            "miner_hotkey",
            "miner_uid",
            "island",
            "policy_id",
            "actual_openrouter_cost_usd",
            "target_reimbursement_usd",
            "target_alpha_percent_per_epoch",
            "start_epoch",
            "epoch_count",
            "award_hash",
            "anchored_hash",
            "current_award_status",
            "current_event_hash",
        ),
    )


def _compact_ref(row: Mapping[str, Any], fields: Sequence[str]) -> dict[str, Any]:
    return {field: row.get(field) for field in fields if row.get(field) is not None}
