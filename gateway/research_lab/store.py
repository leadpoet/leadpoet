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

# Retry idempotent reads only for recognized network or gateway failures.
# Database query errors propagate immediately.
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
