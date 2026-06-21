#!/usr/bin/env python3
"""Static smoke checks for Research Lab event/projection migration.

This verifier is local only. It never connects to Supabase or Postgres.
"""

from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MIGRATION = ROOT / "scripts" / "29-research-lab-event-projections.sql"

REQUIRED_EVENT_TABLES = (
    "research_loop_ticket_events",
    "research_loop_start_credit_events",
    "research_loop_run_queue_events",
    "research_loop_receipt_events",
    "research_reimbursement_award_events",
)

REQUIRED_PROJECTION_VIEWS = (
    "research_loop_ticket_current",
    "research_loop_start_credit_current",
    "research_loop_available_credits",
    "research_loop_run_queue_current",
    "research_loop_receipt_current",
    "research_reimbursement_award_current",
    "research_loop_balance_current",
    "research_loop_shadow_weight_inputs",
)

FORBIDDEN_GRANT_RE = re.compile(
    r"\bGRANT\b(?!\s+EXECUTE\b)[^;]*\b(?:anon|authenticated)\b",
    re.IGNORECASE | re.DOTALL,
)


def main() -> int:
    sql = MIGRATION.read_text(encoding="utf-8")
    errors = verify_sql(sql)
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    print(
        "Research Lab event/projection SQL verified: "
        f"{len(REQUIRED_EVENT_TABLES)} event tables, "
        f"{len(REQUIRED_PROJECTION_VIEWS)} projection views, "
        "service-role only, append-only triggers, no fulfillment writes."
    )
    return 0


def verify_sql(sql: str) -> list[str]:
    errors: list[str] = []
    lowered = sql.lower()
    if "begin;" not in lowered or "commit;" not in lowered:
        errors.append("migration must be wrapped in BEGIN/COMMIT")
    if "does not activate paid loops" not in lowered:
        errors.append("migration must state live workflow activation is out of scope")
    if re.search(r"\b(?:CREATE|ALTER|DROP)\s+TABLE\b[^;]*\bfulfillment_", sql, re.IGNORECASE):
        errors.append("migration must not modify fulfillment tables")
    if FORBIDDEN_GRANT_RE.search(sql):
        errors.append("migration must not grant privileges to anon/authenticated")

    for table in REQUIRED_EVENT_TABLES:
        require_event_table(sql, table, errors)

    for view in REQUIRED_PROJECTION_VIEWS:
        require_projection_view(sql, view, errors)

    for marker in (
        "research_loop_ticket_current",
        "research_loop_available_credits",
        "research_loop_balance_current",
        "WITH (security_invoker = true)",
        "BEFORE UPDATE OR DELETE",
        "prevent_research_lab_append_only_mutation",
        "sk-or-",
    ):
        if marker not in sql:
            errors.append(f"missing marker: {marker}")

    if re.search(r"GRANT\s+[^;]*(UPDATE|DELETE)[^;]*TO\s+service_role", sql, re.IGNORECASE):
        errors.append("append-only event tables must not grant UPDATE/DELETE to service_role")

    return errors


def require_event_table(sql: str, table: str, errors: list[str]) -> None:
    if f"CREATE TABLE IF NOT EXISTS public.{table}" not in sql:
        errors.append(f"{table}: missing CREATE TABLE")
    if f"GRANT SELECT, INSERT ON TABLE public.{table} TO service_role;" not in sql:
        errors.append(f"{table}: missing service_role SELECT/INSERT grant")
    if f"ALTER TABLE public.{table} ENABLE ROW LEVEL SECURITY;" not in sql:
        errors.append(f"{table}: missing RLS enable")
    if f"BEFORE UPDATE OR DELETE ON public.{table}" not in sql:
        errors.append(f"{table}: missing append-only trigger")
    if f"REVOKE ALL ON TABLE public.{table} FROM anon, authenticated;" not in sql:
        errors.append(f"{table}: missing anon/authenticated revoke")


def require_projection_view(sql: str, view: str, errors: list[str]) -> None:
    if f"CREATE OR REPLACE VIEW public.{view}" not in sql:
        errors.append(f"{view}: missing projection view")
    if f"REVOKE ALL ON TABLE public.{view} FROM anon, authenticated;" not in sql:
        errors.append(f"{view}: missing anon/authenticated revoke")
    if f"GRANT SELECT ON TABLE public.{view} TO service_role;" not in sql:
        errors.append(f"{view}: missing service_role SELECT grant")


if __name__ == "__main__":
    raise SystemExit(main())
