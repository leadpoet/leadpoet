"""Load and freeze the active UTC-day ICP set for public competition."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping


DAILY_ICP_COUNT = 20


class DailyIcpSetUnavailable(RuntimeError):
    """The required daily ICP set is missing or invalid."""


@dataclass(frozen=True)
class DailyIcpSet:
    set_id: int
    public_doc: dict[str, Any]
    input_doc: dict[str, Any]
    benchmark_items: tuple[dict[str, Any], ...]
    item_refs: tuple[str, ...]


def utc_day_start(value: datetime | None = None) -> datetime:
    current = (value or datetime.now(timezone.utc)).astimezone(timezone.utc)
    return datetime(current.year, current.month, current.day, tzinfo=timezone.utc)


def utc_set_id_for_datetime(value: datetime | None = None) -> int:
    return int(utc_day_start(value).strftime("%Y%m%d"))


def _timestamp(value: Any) -> datetime | None:
    if value in (None, ""):
        return None
    if isinstance(value, datetime):
        parsed = value
    else:
        try:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except ValueError as exc:
            raise DailyIcpSetUnavailable("daily_set_timestamp_is_invalid") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _validate_items(raw_icps: Any, *, set_id: int) -> tuple[dict[str, Any], ...]:
    if not isinstance(raw_icps, list) or len(raw_icps) != DAILY_ICP_COUNT:
        raise DailyIcpSetUnavailable(
            f"daily_set_{set_id}_requires_{DAILY_ICP_COUNT}_icps"
        )
    items: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for position, raw_icp in enumerate(raw_icps):
        if not isinstance(raw_icp, Mapping):
            raise DailyIcpSetUnavailable(
                f"daily_set_{set_id}_icp_{position + 1}_must_be_object"
            )
        icp = dict(raw_icp)
        icp_id = str(icp.get("icp_id") or "").strip()
        if not icp_id:
            raise DailyIcpSetUnavailable(
                f"daily_set_{set_id}_icp_{position + 1}_missing_id"
            )
        if icp_id in seen_ids:
            raise DailyIcpSetUnavailable(
                f"daily_set_{set_id}_duplicate_icp_id:{icp_id}"
            )
        seen_ids.add(icp_id)
        items.append(
            {
                "icp": icp,
                "icp_ref": f"qualification_private_icp_sets:{set_id}:{icp_id}",
                "set_id": set_id,
                "position": position,
            }
        )
    return tuple(items)


def _documents(
    *, set_id: int, items: tuple[dict[str, Any], ...]
) -> tuple[dict[str, Any], dict[str, Any], tuple[str, ...]]:
    refs = tuple(str(item["icp_ref"]) for item in items)
    public_doc = {
        "schema_version": "research_lab_daily_icp_set.v1",
        "set_id": set_id,
        "icp_count": len(items),
        "icp_refs": list(refs),
    }
    input_doc = {
        "schema_version": "research_lab_daily_icp_inputs.v1",
        "set_id": set_id,
        "icps": [dict(item["icp"]) for item in items],
        "icp_refs": list(refs),
    }
    return public_doc, input_doc, refs


def select_daily_icp_set(
    row: Mapping[str, Any], *, required_set_id: int, active_at: datetime
) -> DailyIcpSet:
    """Validate all 20 inputs from the active set and freeze their order."""

    try:
        set_id = int(row["set_id"])
    except (KeyError, TypeError, ValueError) as exc:
        raise DailyIcpSetUnavailable("daily_set_id_is_invalid") from exc
    if set_id != int(required_set_id):
        raise DailyIcpSetUnavailable(f"required_daily_set_{required_set_id}_missing")
    current = active_at.astimezone(timezone.utc)
    active_from = _timestamp(row.get("active_from"))
    active_until = _timestamp(row.get("active_until"))
    if (
        row.get("is_active") is not True
        or (active_from is not None and current < active_from)
        or (active_until is not None and current >= active_until)
    ):
        raise DailyIcpSetUnavailable(f"required_daily_set_{required_set_id}_not_active")
    items = _validate_items(row.get("icps"), set_id=set_id)
    public_doc, input_doc, refs = _documents(set_id=set_id, items=items)
    return DailyIcpSet(set_id, public_doc, input_doc, items, refs)


def daily_icp_set_from_input_doc(
    document: Mapping[str, Any], *, required_set_id: int
) -> DailyIcpSet:
    """Restore the ordinary private input document saved when the run started."""

    if document.get("schema_version") != "research_lab_daily_icp_inputs.v1":
        raise DailyIcpSetUnavailable("stored_daily_icp_input_schema_is_invalid")
    try:
        set_id = int(document.get("set_id"))
    except (TypeError, ValueError) as exc:
        raise DailyIcpSetUnavailable("stored_daily_icp_set_id_is_invalid") from exc
    if set_id != int(required_set_id):
        raise DailyIcpSetUnavailable("stored_daily_icp_set_id_differs")
    items = _validate_items(document.get("icps"), set_id=set_id)
    public_doc, input_doc, refs = _documents(set_id=set_id, items=items)
    if list(document.get("icp_refs") or []) != list(refs):
        raise DailyIcpSetUnavailable("stored_daily_icp_refs_differ")
    return DailyIcpSet(set_id, public_doc, input_doc, items, refs)


async def fetch_daily_icp_set(*, set_id: int, active_at: datetime) -> DailyIcpSet:
    """Read today's active set from the host-only qualification table."""

    from gateway.db.client import get_write_client

    def _call() -> Any:
        return (
            get_write_client()
            .table("qualification_private_icp_sets")
            .select("set_id,icps,active_from,active_until,is_active")
            .eq("set_id", int(set_id))
            .limit(1)
            .execute()
        )

    response = await asyncio.to_thread(_call)
    rows = getattr(response, "data", None) or []
    if not rows:
        raise DailyIcpSetUnavailable(f"required_daily_set_{set_id}_missing")
    return select_daily_icp_set(
        rows[0], required_set_id=int(set_id), active_at=active_at
    )
