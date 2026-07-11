"""Shared Research Lab ticket-lifecycle policy helpers."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Mapping


UNPAID_TICKET_TTL_SECONDS = 24 * 60 * 60
UNPAID_TICKET_ELIGIBLE_STATUSES = frozenset({"opened", "probe_created", "funding_pending"})
TERMINAL_TICKET_STATUSES = frozenset({"completed", "failed", "cancelled", "tombstoned", "expired"})

_EXPIRY_CONFLICT_MARKERS = (
    "research_lab_ticket_expired",
    "research_lab_ticket_expiry_conflict",
)


def normalized_ticket_status(ticket: Mapping[str, Any] | None) -> str:
    if not ticket:
        return ""
    return str(ticket.get("current_ticket_status") or ticket.get("ticket_status") or "").strip().lower()


def ticket_is_house_arm(ticket: Mapping[str, Any] | None) -> bool:
    if not ticket:
        return False
    ticket_doc = ticket.get("ticket_doc")
    return isinstance(ticket_doc, Mapping) and str(ticket_doc.get("arm") or "").strip().lower() == "house"


def parse_utc_timestamp(value: Any) -> datetime | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    normalized = raw[:-1] + "+00:00" if raw.endswith("Z") else raw
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        # PostgREST may return nanosecond fractions while datetime accepts six digits.
        if "." not in normalized:
            return None
        prefix, fraction_and_suffix = normalized.split(".", 1)
        suffix_index = len(fraction_and_suffix)
        for marker in ("+", "-"):
            marker_index = fraction_and_suffix.find(marker)
            if marker_index >= 0:
                suffix_index = min(suffix_index, marker_index)
        fraction = fraction_and_suffix[:suffix_index]
        suffix = fraction_and_suffix[suffix_index:]
        if not fraction.isdigit():
            return None
        try:
            parsed = datetime.fromisoformat(f"{prefix}.{fraction[:6].ljust(6, '0')}{suffix}")
        except ValueError:
            return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def unpaid_ticket_expires_at(ticket: Mapping[str, Any] | None) -> datetime | None:
    if not ticket:
        return None
    database_deadline = parse_utc_timestamp(ticket.get("unpaid_expires_at"))
    if database_deadline is not None:
        return database_deadline
    created_at = parse_utc_timestamp(ticket.get("created_at"))
    if created_at is None:
        return None
    return created_at + timedelta(seconds=UNPAID_TICKET_TTL_SECONDS)


def unpaid_ticket_deadline_passed(
    ticket: Mapping[str, Any] | None,
    *,
    now: datetime | None = None,
) -> bool:
    deadline = unpaid_ticket_expires_at(ticket)
    if deadline is None:
        return False
    effective_now = now or datetime.now(timezone.utc)
    if effective_now.tzinfo is None:
        effective_now = effective_now.replace(tzinfo=timezone.utc)
    return effective_now.astimezone(timezone.utc) >= deadline


def ticket_is_expired(ticket: Mapping[str, Any] | None) -> bool:
    return normalized_ticket_status(ticket) == "expired"


def is_ticket_expiry_conflict(exc: BaseException) -> bool:
    message = str(exc).lower()
    return any(marker in message for marker in _EXPIRY_CONFLICT_MARKERS)
