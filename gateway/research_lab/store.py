"""Supabase persistence helpers for Research Lab gateway endpoints."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
import logging
import os
from typing import Any, Iterable, Mapping
from uuid import UUID, uuid4, uuid5, NAMESPACE_URL

from gateway.db.client import get_write_client

logger = logging.getLogger(__name__)

# A read on the weight-critical path (allocation graph, publication,
# finalization) that hits a transient edge/proxy failure — a Cloudflare or
# gateway 5xx/timeout in front of Supabase, or a dropped connection — must
# not burn a whole 72-minute epoch. These are idempotent reads, so a bounded
# retry is safe. The classifier is an allowlist: anything that is not a
# recognized transient propagates immediately, exactly as before, so genuine
# query-logic errors still fail closed.
_TRANSIENT_READ_ATTEMPTS = 4
_TRANSIENT_READ_BACKOFF_SECONDS = (0.25, 0.75, 1.5)
_TRANSIENT_ERROR_SIGNATURES = (
    "cloudflare",
    "<html",
    "json could not be generated",
    "bad gateway",
    "gateway time-out",
    "gateway timeout",
    "service temporarily unavailable",
    "temporarily unavailable",
    "connection reset",
    "connection aborted",
    "connection refused",
    "server disconnected",
    "unexpected eof",
    "unexpected_eof",
    "timed out",
    "timeout",
)
_TRANSIENT_ERROR_TYPE_SIGNATURES = (
    "timeout",
    "connection",
    "connecterror",
    "readerror",
    "remoteprotocol",
    "serverdisconnected",
)


def _is_transient_store_error(exc: BaseException) -> bool:
    """Return whether a store failure is a retryable edge/network transient.

    Fail-safe: only a recognized transient returns True. An unknown error —
    including a genuine PostgREST/Postgres query error — returns False and
    propagates unchanged.
    """

    type_name = type(exc).__name__.lower()
    if any(token in type_name for token in _TRANSIENT_ERROR_TYPE_SIGNATURES):
        return True
    message = str(getattr(exc, "message", "") or "").lower()
    detail = str(exc).lower()
    haystack = message + "\n" + detail
    # A genuine PostgREST logic error carries a SQLSTATE or PGRST code; never
    # retry those even if some transient token also appears in the payload.
    code = str(getattr(exc, "code", "") or "").strip().lower()
    edge_codes = {"408", "429", "500", "502", "503", "504", "520", "521", "522", "523", "524"}
    if code in edge_codes:
        return True
    if code and code not in edge_codes and (code.startswith("pgrst") or len(code) == 5):
        return False
    return any(token in haystack for token in _TRANSIENT_ERROR_SIGNATURES)


def _is_transient_read_error(exc: BaseException) -> bool:
    """Backward-compatible name for the shared fail-closed classifier."""

    return _is_transient_store_error(exc)


async def _execute_read_with_retry(call, *, label: str):
    """Run an idempotent PostgREST read, retrying only transient failures."""

    last_exc: BaseException | None = None
    for attempt in range(_TRANSIENT_READ_ATTEMPTS):
        try:
            return await asyncio.to_thread(call)
        except Exception as exc:  # noqa: BLE001 - reclassified below
            if not _is_transient_read_error(exc) or attempt == (
                _TRANSIENT_READ_ATTEMPTS - 1
            ):
                raise
            last_exc = exc
            backoff = _TRANSIENT_READ_BACKOFF_SECONDS[
                min(attempt, len(_TRANSIENT_READ_BACKOFF_SECONDS) - 1)
            ]
            logger.warning(
                "transient_read_retry label=%s attempt=%s/%s type=%s error=%s",
                label,
                attempt + 1,
                _TRANSIENT_READ_ATTEMPTS,
                type(exc).__name__,
                str(exc)[:160],
            )
            await asyncio.sleep(backoff)
    # Unreachable: the loop either returns or raises, but keep mypy honest.
    assert last_exc is not None
    raise last_exc



RESEARCH_LAB_UUID_NAMESPACE = uuid5(NAMESPACE_URL, "leadpoet:research_lab:gateway")


def canonical_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def deterministic_uuid(*parts: Any) -> str:
    return str(uuid5(RESEARCH_LAB_UUID_NAMESPACE, canonical_hash(parts)))


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _apply_filters(query: Any, filters: Iterable[tuple[Any, ...]]) -> Any:
    for raw_filter in filters:
        if len(raw_filter) == 2:
            field, value = raw_filter
            query = query.eq(field, str(value) if isinstance(value, UUID) else value)
            continue
        if len(raw_filter) != 3:
            raise ValueError(f"invalid PostgREST filter spec: {raw_filter!r}")
        field, operator, value = raw_filter
        value = str(value) if isinstance(value, UUID) else value
        if operator == "eq":
            query = query.eq(field, value)
        elif operator == "neq":
            query = query.neq(field, value)
        elif operator == "lt":
            query = query.lt(field, value)
        elif operator == "lte":
            query = query.lte(field, value)
        elif operator == "gt":
            query = query.gt(field, value)
        elif operator == "gte":
            query = query.gte(field, value)
        elif operator == "in":
            query = query.in_(field, value)
        elif operator == "is":
            query = query.is_(field, value)
        else:
            raise ValueError(f"unsupported PostgREST filter operator: {operator}")
    return query


async def insert_row(table: str, row: dict[str, Any]) -> dict[str, Any]:
    def _call() -> Any:
        return get_write_client().table(table).insert(row).execute()

    response = await asyncio.to_thread(_call)
    data = getattr(response, "data", None) or []
    if not data:
        raise RuntimeError(f"{table}: insert returned no rows")
    return dict(data[0])


async def insert_rows(
    table: str, rows: Iterable[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """Insert one nonempty PostgREST batch and return its row representations."""

    payload = [dict(row) for row in rows]
    if not payload:
        raise ValueError(f"{table}: batch insert requires at least one row")

    def _call() -> Any:
        return get_write_client().table(table).insert(payload).execute()

    response = await asyncio.to_thread(_call)
    return [dict(row) for row in (getattr(response, "data", None) or [])]


async def call_rpc(function_name: str, params: Mapping[str, Any]) -> Any:
    """Call one service-role PostgREST function without blocking the event loop."""
    def _call() -> Any:
        return get_write_client().rpc(function_name, dict(params)).execute()

    response = await asyncio.to_thread(_call)
    return getattr(response, "data", None)


async def persist_source_add_submission(record_doc: dict[str, Any]) -> None:
    """Persist a SOURCE_ADD submission's funnel stages (W5, append-only).

    One row per ``stage_history`` entry, keyed UNIQUE(submission_id, seq).
    Idempotent: already-persisted stage rows are skipped, so calling this
    after every funnel transition writes only the new stages. The full
    submission doc rides on the newest row (earlier rows keep a stub) so the
    current view always returns the freshest state.
    """

    submission_id = str(record_doc.get("submission_id") or "")
    stage_history = [str(stage) for stage in (record_doc.get("stage_history") or [])]
    if not submission_id or not stage_history:
        raise ValueError("source_add submission doc requires submission_id and stage_history")
    yield_value = record_doc.get("measured_trial_yield")
    measured_yield = float(yield_value) if isinstance(yield_value, (int, float)) and float(yield_value) >= 0 else None
    last_seq = len(stage_history) - 1
    precheck_doc = record_doc.get("precheck_doc") if isinstance(record_doc.get("precheck_doc"), Mapping) else {}
    for seq, stage in enumerate(stage_history):
        row = {
            "submission_id": submission_id,
            "adapter_id": str(record_doc.get("adapter_id") or ""),
            "miner_hotkey": str(record_doc.get("miner_hotkey") or ""),
            "stage": stage,
            "seq": seq,
            "measured_trial_yield": measured_yield if seq == last_seq else None,
            "submission_doc": record_doc if seq == last_seq else {},
            "precheck_status": str(record_doc.get("precheck_status") or "") if seq == last_seq else "",
            "precheck_doc": dict(precheck_doc) if seq == last_seq else {},
            "source_identity_hash": str(record_doc.get("source_identity_hash") or "") if seq == last_seq else "",
        }
        try:
            await insert_row("research_lab_source_add_submissions", row)
        except Exception as exc:
            if "duplicate" in str(exc).lower() or "unique" in str(exc).lower() or "23505" in str(exc):
                continue
            raise


async def select_one(
    table: str,
    *,
    columns: str = "*",
    filters: Iterable[tuple[str, Any]],
) -> dict[str, Any] | None:
    normalized_filters = tuple(filters)

    def _call() -> Any:
        query = get_write_client().table(table).select(columns)
        query = _apply_filters(query, normalized_filters)
        return query.limit(1).execute()

    response = await _execute_read_with_retry(
        _call, label="select_one:%s" % table
    )
    data = getattr(response, "data", None) or []
    return dict(data[0]) if data else None


async def select_many(
    table: str,
    *,
    columns: str = "*",
    filters: Iterable[tuple[str, Any]],
    order_by: Iterable[tuple[str, bool]] = (),
    limit: int = 100,
) -> list[dict[str, Any]]:
    normalized_filters = tuple(filters)
    normalized_order = tuple(order_by)

    def _call() -> Any:
        query = get_write_client().table(table).select(columns)
        query = _apply_filters(query, normalized_filters)
        for field, desc in normalized_order:
            query = query.order(field, desc=desc)
        return query.limit(limit).execute()

    response = await _execute_read_with_retry(
        _call, label="select_many:%s" % table
    )
    return [dict(row) for row in (getattr(response, "data", None) or [])]


async def select_all(
    table: str,
    *,
    columns: str = "*",
    filters: Iterable[tuple[str, Any]],
    order_by: Iterable[tuple[str, bool]] = (),
    batch_size: int = 1000,
    max_rows: int = 10000,
    allow_partial: bool = False,
) -> list[dict[str, Any]]:
    """Fetch rows with explicit PostgREST pagination for weight-critical paths."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if max_rows <= 0:
        raise ValueError("max_rows must be positive")
    normalized_filters = tuple(filters)
    normalized_order = tuple(order_by)
    rows: list[dict[str, Any]] = []
    offset = 0
    while offset < max_rows:
        end = min(offset + batch_size - 1, max_rows - 1)

        def _call() -> Any:
            query = get_write_client().table(table).select(columns)
            query = _apply_filters(query, normalized_filters)
            for field, desc in normalized_order:
                query = query.order(field, desc=desc)
            return query.range(offset, end).execute()

        response = await _execute_read_with_retry(
            _call, label="select_all:%s" % table
        )
        batch = [dict(row) for row in (getattr(response, "data", None) or [])]
        rows.extend(batch)
        if len(batch) < batch_size:
            return rows
        offset += batch_size
    if allow_partial:
        return rows
    raise RuntimeError(f"{table}: paginated select exceeded max_rows={max_rows}")


async def next_event_seq(table: str, key_field: str, key_value: Any) -> int:
    def _call() -> Any:
        return (
            get_write_client()
            .table(table)
            .select("seq")
            .eq(key_field, str(key_value))
            .order("seq", desc=True)
            .limit(1)
            .execute()
        )

    response = await asyncio.to_thread(_call)
    data = getattr(response, "data", None) or []
    return int(data[0]["seq"]) + 1 if data else 0


def _is_seq_conflict(exc: BaseException) -> bool:
    """True for a UNIQUE(key, seq) violation — the signature of a concurrent event-seq
    race. Other unique violations (e.g. content-addressed hashes) are NOT retried, since
    they indicate a genuine duplicate rather than a seq race."""
    message = str(exc).lower()
    is_unique = "duplicate key" in message or "unique constraint" in message or "23505" in message
    return is_unique and "seq" in message


async def append_event_with_seq(
    table: str,
    key_field: str,
    key_value: Any,
    build_payload: Any,
    *,
    attempts: int = 5,
    event_id: str | None = None,
) -> dict[str, Any]:
    """Allocate the next event seq and insert atomically against concurrent appends.

    ``next_event_seq`` is read-max-then-insert, so two concurrent appends for the same
    key can pick the same seq; the DB ``UNIQUE(key, seq)`` constraint rejects the loser.
    This retries the loser (re-read seq, rebuild payload, re-insert) so both appends land
    instead of one crashing. The row is built identically to the legacy inline form
    (``event_id`` + ``schema_version`` + payload + ``anchored_hash`` over the payload),
    so audit hashes are unchanged. ``build_payload(seq)`` returns the payload dict.
    Callers may supply a deterministic ``event_id`` when the logical event itself
    must remain idempotent across a committed insert whose response was lost.
    """
    last_exc: BaseException | None = None
    for attempt in range(1, max(1, int(attempts)) + 1):
        seq = await next_event_seq(table, key_field, key_value)
        payload = build_payload(seq)
        row = {
            "event_id": event_id or str(uuid4()),
            "schema_version": "1.0",
            **payload,
            "anchored_hash": canonical_hash(payload),
        }
        try:
            return await insert_row(table, row)
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if _is_seq_conflict(exc) and attempt < int(attempts):
                continue
            raise
    assert last_exc is not None  # pragma: no cover - loop always returns or raises
    raise last_exc


async def _existing_or_recovered_event(
    event_table: str,
    key_field: str,
    key_value: Any,
    create_opening_event: Any,
) -> dict[str, Any]:
    """Return an idempotency event, recreating seq=0 if a prior insert crashed."""
    event = await select_one(
        event_table,
        filters=((key_field, key_value), ("seq", 0)),
    )
    if event:
        return event
    existing_events = await select_many(
        event_table,
        filters=((key_field, key_value),),
        order_by=(("seq", False),),
        limit=1,
    )
    if existing_events:
        return existing_events[0]
    try:
        return await create_opening_event()
    except Exception:
        event = await select_one(
            event_table,
            filters=((key_field, key_value), ("seq", 0)),
        )
        if event:
            return event
        existing_events = await select_many(
            event_table,
            filters=((key_field, key_value),),
            order_by=(("seq", False),),
            limit=1,
        )
        if existing_events:
            return existing_events[0]
        raise












































































async def create_champion_reward_obligation(
    *,
    obligation: dict[str, Any],
    ticket_id: str | None = None,
    obligation_doc: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    existing = await select_one(
        "research_lab_champion_reward_obligations",
        filters=(("champion_reward_id", obligation["champion_reward_id"]),),
    )
    if existing:
        event = await _existing_or_recovered_event(
            "research_lab_champion_reward_events",
            "champion_reward_id",
            obligation["champion_reward_id"],
            lambda: create_champion_reward_event(
                champion_reward_id=obligation["champion_reward_id"],
                event_type="active",
                reward_status="active",
                reason="created_from_gateway_promotion_event",
                event_doc={
                    "candidate_id": obligation.get("candidate_id"),
                    "score_bundle_id": obligation.get("score_bundle_id"),
                },
            ),
        )
        return existing, event
    row = {
        "champion_reward_id": obligation["champion_reward_id"],
        "schema_version": "1.0",
        "score_bundle_id": obligation.get("score_bundle_id") or None,
        "candidate_id": obligation.get("candidate_id") or None,
        "run_id": obligation["run_id"],
        "ticket_id": ticket_id,
        "miner_hotkey": obligation["miner_hotkey"],
        "miner_uid": int(obligation["uid"]),
        "island": obligation["island"],
        "policy_id": str((obligation_doc or {}).get("policy_id") or "research-lab-promotion-v1"),
        "evaluation_epoch": int(obligation["evaluation_epoch"]),
        "start_epoch": int(obligation["start_epoch"]),
        "epoch_count": int(obligation["epoch_count"]),
        "improvement_points": float(obligation["improvement_points"]),
        "threshold_points": float(obligation["threshold_points"]),
        "desired_alpha_percent": float(obligation["desired_alpha_percent"]),
        "source_score_bundle_hash": (obligation_doc or {}).get("source_score_bundle_hash"),
        "input_hash": obligation["input_hash"],
        "anchored_hash": obligation["anchored_hash"],
        "obligation_doc": dict(obligation_doc or {}),
    }
    inserted = await insert_row("research_lab_champion_reward_obligations", row)
    event = await create_champion_reward_event(
        champion_reward_id=obligation["champion_reward_id"],
        event_type="active",
        reward_status="active",
        reason="created_from_gateway_promotion_event",
        event_doc={"candidate_id": obligation.get("candidate_id"), "score_bundle_id": obligation.get("score_bundle_id")},
    )
    return inserted, event


async def create_champion_reward_event(
    *,
    champion_reward_id: str,
    event_type: str,
    reward_status: str,
    reason: str | None = None,
    event_doc: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return await append_event_with_seq(
        "research_lab_champion_reward_events",
        "champion_reward_id",
        champion_reward_id,
        lambda seq: {
            "champion_reward_id": champion_reward_id,
            "seq": seq,
            "event_type": event_type,
            "reward_status": reward_status,
            "reason": reason,
            "event_doc": event_doc or {},
        },
    )


async def create_research_lab_emission_allocation_snapshot(
    *,
    epoch: int,
    netuid: int,
    policy_id: str,
    snapshot_status: str,
    allocation_doc: dict[str, Any],
) -> dict[str, Any]:
    allocation_hash = str(allocation_doc["allocation_hash"])
    allocation_id = "lab_allocation:" + allocation_hash
    row = {
        "allocation_id": allocation_id,
        "schema_version": "1.0",
        "epoch": int(epoch),
        "netuid": int(netuid),
        "policy_id": str(policy_id),
        "snapshot_status": str(snapshot_status),
        "lab_cap_alpha_percent": float(allocation_doc.get("lab_cap_percent") or 0.0),
        "source_add_alpha_percent": float(allocation_doc.get("source_add_alpha_percent") or 0.0),
        "reimbursement_alpha_percent": float(allocation_doc.get("reimbursement_alpha_percent") or 0.0),
        "champion_alpha_percent": float(allocation_doc.get("champion_alpha_percent") or 0.0),
        "queued_champion_alpha_percent": float(allocation_doc.get("queued_champion_alpha_percent") or 0.0),
        "unallocated_alpha_percent": float(allocation_doc.get("unallocated_percent") or 0.0),
        "input_hash": str(allocation_doc.get("input_hash") or ""),
        "allocation_hash": allocation_hash,
        "allocation_doc": allocation_doc,
    }

    def validate_exact(existing: Mapping[str, Any]) -> dict[str, Any]:
        numeric_fields = {
            "lab_cap_alpha_percent",
            "source_add_alpha_percent",
            "reimbursement_alpha_percent",
            "champion_alpha_percent",
            "queued_champion_alpha_percent",
            "unallocated_alpha_percent",
        }
        for key, expected in row.items():
            observed = existing.get(key)
            if key in numeric_fields:
                try:
                    matches = Decimal(str(observed)) == Decimal(str(expected))
                except Exception:
                    matches = False
            else:
                matches = observed == expected
            if not matches:
                raise RuntimeError(
                    "research_lab_emission_allocation_snapshots: "
                    f"existing {key} differs for {allocation_id}"
                )
        return dict(existing)

    filters = (("allocation_id", allocation_id),)
    existing = await select_one(
        "research_lab_emission_allocation_snapshots",
        filters=filters,
    )
    if existing:
        return validate_exact(existing)
    try:
        return validate_exact(
            await insert_row("research_lab_emission_allocation_snapshots", row)
        )
    except Exception as exc:
        message = str(exc).lower()
        if (
            "duplicate" not in message
            and "unique" not in message
            and "23505" not in message
        ):
            raise
        existing = await select_one(
            "research_lab_emission_allocation_snapshots",
            filters=filters,
        )
        if not existing:
            raise RuntimeError(
                "research_lab_emission_allocation_snapshots: "
                "duplicate snapshot could not be reloaded"
            ) from exc
        logger.warning(
            "research_lab_allocation_snapshot_duplicate_replayed "
            "allocation_id=%s epoch=%s netuid=%s",
            allocation_id,
            int(epoch),
            int(netuid),
        )
        return validate_exact(existing)






async def create_arweave_epoch_audit_anchor(
    *,
    epoch: int,
    netuid: int,
    audit_kind: str,
    audit_bundle_id: str | None,
    audit_bundle_hash: str | None,
    allocation_hash: str | None,
    weights_hash: str | None,
    payload_hash: str,
    transparency_event_hash: str | None,
    tee_sequence: int | None,
    event_doc: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    identity_payload = {
        "epoch": int(epoch),
        "netuid": int(netuid),
        "audit_kind": audit_kind,
        "payload_hash": payload_hash,
    }
    anchor_hash = canonical_hash(identity_payload)
    anchor_id = "research_lab_arweave_anchor:" + anchor_hash.split(":", 1)[1]
    existing = await select_one(
        "research_lab_arweave_epoch_audit_anchors",
        filters=(("anchor_id", anchor_id),),
    )
    payload = {
        "epoch": int(epoch),
        "netuid": int(netuid),
        "audit_kind": audit_kind,
        "audit_bundle_id": audit_bundle_id,
        "audit_bundle_hash": audit_bundle_hash,
        "allocation_hash": allocation_hash,
        "weights_hash": weights_hash,
        "payload_hash": payload_hash,
        "transparency_event_hash": transparency_event_hash,
        "tee_sequence": tee_sequence,
    }
    if existing:
        event = await select_one(
            "research_lab_arweave_epoch_audit_anchor_events",
            filters=(("anchor_id", anchor_id), ("event_type", "buffered")),
        )
        if event:
            return existing, event
        event = await create_arweave_epoch_audit_anchor_event(
            anchor_id=anchor_id,
            event_type="buffered",
            anchor_status="buffered",
            event_doc=event_doc or {},
        )
        return existing, event

    row = {
        "anchor_id": anchor_id,
        "schema_version": "1.0",
        **payload,
        "anchor_hash": anchor_hash,
        "anchored_hash": anchor_hash,
    }
    inserted = await insert_row("research_lab_arweave_epoch_audit_anchors", row)
    await create_arweave_epoch_audit_anchor_event(
        anchor_id=anchor_id,
        event_type="created",
        anchor_status="created",
        event_doc={
            "payload_hash": payload_hash,
            "audit_kind": audit_kind,
        },
    )
    event = await create_arweave_epoch_audit_anchor_event(
        anchor_id=anchor_id,
        event_type="buffered",
        anchor_status="buffered",
        event_doc=event_doc or {},
    )
    return inserted, event


async def create_arweave_epoch_audit_anchor_event(
    *,
    anchor_id: str,
    event_type: str,
    anchor_status: str,
    transparency_event_hash: str | None = None,
    tee_sequence: int | None = None,
    checkpoint_number: int | None = None,
    checkpoint_merkle_root: str | None = None,
    arweave_tx_id: str | None = None,
    event_doc: dict[str, Any] | None = None,
) -> dict[str, Any]:
    doc = event_doc or {}
    return await append_event_with_seq(
        "research_lab_arweave_epoch_audit_anchor_events",
        "anchor_id",
        anchor_id,
        lambda seq: {
            "anchor_id": anchor_id,
            "seq": seq,
            "event_type": event_type,
            "anchor_status": anchor_status,
            "transparency_event_hash": transparency_event_hash or doc.get("transparency_event_hash") or doc.get("event_hash"),
            "tee_sequence": tee_sequence if tee_sequence is not None else doc.get("tee_sequence"),
            "checkpoint_number": (
                checkpoint_number
                if checkpoint_number is not None
                else doc.get("checkpoint_number")
            ),
            "checkpoint_merkle_root": checkpoint_merkle_root or doc.get("checkpoint_merkle_root"),
            "arweave_tx_id": arweave_tx_id or doc.get("arweave_tx_id"),
            "event_doc": doc,
        },
    )


async def create_participation_snapshot(
    *,
    island: str,
    lookback_start: str,
    lookback_end: str,
    distinct_funded_hotkeys: int,
    paid_loop_count: int,
    unique_brief_count: int,
    participation_score: float,
    policy_id: str,
    snapshot_doc: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "island": island,
        "lookback_start": lookback_start,
        "lookback_end": lookback_end,
        "distinct_funded_hotkeys": max(0, int(distinct_funded_hotkeys)),
        "paid_loop_count": max(0, int(paid_loop_count)),
        "unique_brief_count": max(0, int(unique_brief_count)),
        "source_add_count": 0,
        "red_team_count": 0,
        "participation_score": float(participation_score),
        "policy_id": policy_id,
        "snapshot_doc": snapshot_doc or {},
    }
    input_hash = canonical_hash(payload)
    existing = await select_one(
        "research_island_participation_snapshots",
        filters=(("input_hash", input_hash),),
    )
    if existing:
        return existing
    snapshot_ref = "research_island_participation_snapshot:" + input_hash.split(":", 1)[1]
    row = {
        "participation_snapshot_id": str(uuid4()),
        "schema_version": "1.0",
        "snapshot_ref": snapshot_ref,
        **payload,
        "input_hash": input_hash,
    }
    return await insert_row("research_island_participation_snapshots", row)


def _award_supplement_events_enabled() -> bool:
    return os.getenv(
        "RESEARCH_LAB_REIMBURSEMENT_SUPPLEMENT_EVENTS", "true"
    ).strip().lower() in {"1", "true", "yes", "on"}


async def _maybe_record_award_supplement(
    existing: Mapping[str, Any], award: Mapping[str, Any]
) -> dict[str, Any] | None:
    """Record post-resume spend that exceeds the recorded award target (bug #25).

    Awards are append-only (first terminal event wins the base row), so a
    resumed run's extra spend used to vanish entirely. The base row cannot be
    rewritten; instead a supplemental ``awarded`` event carries the reconcilable
    delta for operators/allocators to consume.
    """
    if not _award_supplement_events_enabled():
        return None
    try:
        existing_status = str(existing.get("award_status") or "")
        if existing_status != "awarded":
            return None
        existing_target = int(existing.get("target_reimbursement_microusd") or 0)
        new_target = int(award.get("target_reimbursement_microusd") or 0)
        if new_target <= existing_target:
            return None
        event = await create_reimbursement_award_event(
            award_id=str(existing["award_id"]),
            event_type="awarded",
            award_status="awarded",
            event_doc={
                "award_id": str(existing["award_id"]),
                "reason": "supplemental_spend_after_first_terminal_event",
                "previous_target_reimbursement_microusd": existing_target,
                "recomputed_target_reimbursement_microusd": new_target,
                "supplemental_target_reimbursement_microusd": new_target - existing_target,
                "target_reimbursement_microusd": new_target,
            },
        )
        logger.warning(
            "research_lab_reimbursement_supplement_recorded award_id=%s previous_microusd=%s new_microusd=%s",
            str(existing.get("award_id") or "")[:24],
            existing_target,
            new_target,
        )
        return event
    except Exception as exc:
        logger.warning(
            "research_lab_reimbursement_supplement_failed award_id=%s error=%s",
            str(existing.get("award_id") or "")[:24],
            str(exc)[:200],
        )
        return None


async def create_reimbursement_award(
    *,
    award: dict[str, Any],
    receipt_id: str | None,
    participation_snapshot_id: str | None,
    policy_id: str,
    award_doc: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    existing = await select_one("research_reimbursement_awards", filters=(("award_id", award["award_id"]),))
    if not existing:
        existing = await select_one("research_reimbursement_awards", filters=(("run_id", str(award["run_id"])),))
    if existing:
        existing_award_id = str(existing["award_id"])
        event = await _existing_or_recovered_event(
            "research_reimbursement_award_events",
            "award_id",
            existing_award_id,
            lambda: create_reimbursement_award_event(
                award_id=existing_award_id,
                event_type=str(existing.get("award_status") or award["status"]),
                award_status=str(existing.get("award_status") or award["status"]),
                event_doc={
                    "award_id": existing_award_id,
                    "target_reimbursement_microusd": int(
                        existing.get("target_reimbursement_microusd")
                        or award["target_reimbursement_microusd"]
                    ),
                },
            ),
        )
        supplement_event = await _maybe_record_award_supplement(existing, award)
        return existing, supplement_event or event
    row = {
        "award_id": str(award["award_id"]),
        "schema_version": "1.0",
        "receipt_id": receipt_id,
        "participation_snapshot_id": participation_snapshot_id,
        "run_id": str(award["run_id"]),
        "miner_hotkey": str(award["miner_hotkey"]),
        "island": str(award["island"]),
        "run_day": str(award["run_day"]),
        "policy_id": policy_id,
        "award_status": str(award["status"]),
        "participation_score": float(award["participation_score"]),
        "participation_fraction": float(award["participation_fraction"]),
        "rebate_rate": float(award["rebate_rate"]),
        "eligible_cost_microusd": int(award["eligible_cost_microusd"]),
        "target_reimbursement_microusd": int(award["target_reimbursement_microusd"]),
        "reimbursement_epochs": int(award["reimbursement_epochs"]),
        "loop_start_fee_included": bool(award["loop_start_fee_included"]),
        "input_hash": str(award["input_hash"]),
        "award_doc": award_doc or award,
    }
    inserted = await insert_row("research_reimbursement_awards", row)
    event = await create_reimbursement_award_event(
        award_id=str(award["award_id"]),
        event_type=str(award["status"]),
        award_status=str(award["status"]),
        event_doc={"award_id": str(award["award_id"]), "target_reimbursement_microusd": int(award["target_reimbursement_microusd"])},
    )
    return inserted, event


async def create_reimbursement_award_event(
    *,
    award_id: str,
    event_type: str,
    award_status: str,
    event_doc: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return await append_event_with_seq(
        "research_reimbursement_award_events",
        "award_id",
        award_id,
        lambda seq: {
            "award_id": award_id,
            "seq": seq,
            "event_type": event_type,
            "award_status": award_status,
            "event_doc": event_doc or {},
        },
    )


async def create_reimbursement_schedule(
    *,
    schedule: dict[str, Any],
    schedule_doc: dict[str, Any] | None = None,
) -> dict[str, Any]:
    existing = await select_one("research_reimbursement_schedules", filters=(("schedule_id", schedule["schedule_id"]),))
    if existing:
        return existing
    doc = schedule_doc or schedule
    row = {
        "schedule_id": str(schedule["schedule_id"]),
        "schema_version": "1.0",
        "award_id": str(schedule["award_id"]),
        "schedule_status": str(schedule["status"]),
        "start_epoch": int(schedule["start_epoch"]),
        "epoch_count": int(schedule["epoch_count"]),
        "total_microusd": int(schedule["total_microusd"]),
        "entries": list(schedule.get("entries", [])),
        "schedule_hash": canonical_hash(doc),
        "schedule_doc": doc,
    }
    return await insert_row("research_reimbursement_schedules", row)
