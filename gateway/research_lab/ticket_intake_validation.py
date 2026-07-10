"""Free ticket-intake validation for Research Lab auto-research directions.

Runs inside the free ticket-create endpoint, before any charge and before a
single planner token is spent:

1. **Path validation** — a direction that names concrete repository paths is
   checked against the code-edit path allowlist (prefix/suffix/escape rules).
   A direction bound to a path the builder could never edit fails every loop
   after the miner has paid; rejecting it at intake is free.
2. **Refusal dedup** — a direction whose exact focus signature was already
   refused (``no_viable_patch``) N times against the currently active model
   source is rejected with the refusal count. The signature is the same
   normalized hash the loop engine stamps on refusal events, so only
   *identical* directions are deduplicated — similar or exploratory
   directions (different wording, prompt experiments) hash differently and
   are always allowed through.

Both checks fail open on infrastructure errors: intake validation must never
block ticket creation because a lookup was down.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any, Mapping

from research_lab.canonical import sha256_json

logger = logging.getLogger(__name__)

_PATH_LIKE_RE = re.compile(r"[A-Za-z0-9_\-./]+\.(?:py|json|yaml|yml|toml|txt|md)\b")


def intake_direction_validation_enabled() -> bool:
    return str(
        os.getenv("RESEARCH_LAB_INTAKE_DIRECTION_VALIDATION", "true") or ""
    ).strip().lower() in {"1", "true", "yes", "on"}


def _intake_refusal_threshold() -> int:
    try:
        return max(1, int(os.getenv("RESEARCH_LAB_INTAKE_REFUSAL_THRESHOLD", "3")))
    except ValueError:
        return 3


def focus_signature_hash_for_brief(brief_public_summary: str) -> str:
    """Direction signature — must stay byte-identical to the loop engine's
    ``_focus_signature_hash`` so intake dedup matches refusal events."""
    normalized = re.sub(r"\s+", " ", str(brief_public_summary or "").strip().lower())[:2000]
    return sha256_json({"focus": normalized})


def path_references_from_brief(brief_public_summary: str) -> tuple[str, ...]:
    """Concrete repository-file references named in the direction text."""
    seen: list[str] = []
    for match in _PATH_LIKE_RE.finditer(str(brief_public_summary or "")):
        value = match.group(0).strip().strip(".")
        if not value or value in seen:
            continue
        if len(value) > 200:
            continue
        seen.append(value)
        if len(seen) >= 16:
            break
    return tuple(seen)


def invalid_path_reference_reason(
    path_value: str,
    *,
    allowed_prefixes: tuple[str, ...],
    allowed_exact: tuple[str, ...],
    allowed_suffixes: tuple[str, ...],
) -> str:
    """Why a referenced path could never be edited, or '' when plausible.

    Bare filenames (no directory) are never rejected — the planner can
    resolve them against the real source tree at run time.
    """
    value = str(path_value or "").strip()
    if not value or "/" not in value:
        return ""
    if ".." in value or value.startswith("/") or "://" in value:
        return "path_escapes_repository"
    if value in allowed_exact:
        return ""
    if allowed_suffixes and not value.endswith(tuple(allowed_suffixes)):
        return "path_suffix_not_editable"
    if allowed_prefixes and not value.startswith(tuple(allowed_prefixes)):
        return "path_outside_editable_roots"
    return ""


async def direction_refusal_count(
    *,
    focus_signature_hash: str,
    active_since: str,
    scan_limit: int = 200,
) -> int:
    """How many no_viable_patch refusals this exact direction accumulated
    against the currently active model source (bounded scan, fails open)."""
    from gateway.research_lab.store import select_many

    try:
        rows = await select_many(
            "research_lab_auto_research_loop_events",
            columns="created_at,event_doc",
            filters=(("event_type", "no_viable_patch"),),
            order_by=(("created_at", True),),
            limit=max(1, scan_limit),
        )
    except Exception as exc:  # noqa: BLE001 - fail open
        logger.warning(
            "research_lab_intake_refusal_scan_failed error=%s", str(exc)[:200]
        )
        return 0
    count = 0
    for row in rows:
        if active_since and str(row.get("created_at") or "") < active_since:
            continue
        doc = row.get("event_doc")
        if not isinstance(doc, Mapping):
            continue
        if str(doc.get("focus_signature_hash") or "") == focus_signature_hash:
            count += 1
    return count


async def validate_ticket_direction(
    *,
    brief_public_summary: str,
    allowed_prefixes: tuple[str, ...],
    allowed_exact: tuple[str, ...],
    allowed_suffixes: tuple[str, ...],
) -> dict[str, Any] | None:
    """Free intake validation. Returns a rejection doc or None (accepted).

    Rejection doc: {"reason": str, "detail": str, ...} — the caller turns it
    into a 422 so the miner learns why before paying anything.
    """
    if not intake_direction_validation_enabled():
        return None
    brief = str(brief_public_summary or "")

    for path_value in path_references_from_brief(brief):
        reason = invalid_path_reference_reason(
            path_value,
            allowed_prefixes=allowed_prefixes,
            allowed_exact=allowed_exact,
            allowed_suffixes=allowed_suffixes,
        )
        if reason:
            return {
                "reason": "direction_references_uneditable_path",
                "detail": (
                    f"direction references '{path_value}' ({reason}); "
                    "editable paths must sit under the allowed repository roots"
                ),
                "path": path_value,
                "path_reason": reason,
            }

    signature = focus_signature_hash_for_brief(brief)
    active_since = ""
    try:
        from gateway.research_lab.store import select_many

        rows = await select_many(
            "research_lab_private_model_version_current",
            columns="current_status_at",
            filters=(("current_version_status", "active"),),
            limit=1,
        )
        if rows:
            active_since = str(rows[0].get("current_status_at") or "")
    except Exception as exc:  # noqa: BLE001 - fail open
        logger.warning(
            "research_lab_intake_active_version_lookup_failed error=%s", str(exc)[:200]
        )
    threshold = _intake_refusal_threshold()
    refusals = await direction_refusal_count(
        focus_signature_hash=signature,
        active_since=active_since,
    )
    if refusals >= threshold:
        return {
            "reason": "direction_repeatedly_refused_against_current_source",
            "detail": (
                f"this exact direction was refused {refusals} times against the "
                "currently active model source; rephrase it or target a "
                "different improvement (similar/exploratory directions are fine)"
            ),
            "refusal_count": refusals,
            "refusal_threshold": threshold,
            "focus_signature_hash": signature,
        }
    return None
