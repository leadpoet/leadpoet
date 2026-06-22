#!/usr/bin/env python3
"""Static smoke checks for Research Lab persistence migration drafts.

This verifier intentionally does not connect to Supabase/Postgres. It checks
the local SQL contract for the Work Package 3 invariants that can be verified
without applying the migration.
"""

from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MIGRATION = ROOT / "scripts" / "28-research-lab-persistence-state.sql"

REQUIRED_TABLES = (
    "research_loop_tickets",
    "research_loop_balance_ledger",
    "research_loop_start_payments",
    "research_loop_start_credits",
    "research_loop_receipts",
    "research_island_participation_snapshots",
    "research_reimbursement_awards",
    "research_reimbursement_schedules",
    "research_weight_input_snapshots",
    "research_evidence_trajectory_links",
)

FORBIDDEN_GRANT_RE = re.compile(
    r"\bGRANT\b(?!\s+EXECUTE\b)[^;]*\b(?:anon|authenticated)\b",
    re.IGNORECASE | re.DOTALL,
)

SECRET_COLUMN_RE = re.compile(
    r"\b(?:openrouter_api_key|raw_openrouter_key|raw_secret|api_key)\b\s+(?:TEXT|JSONB|VARCHAR)",
    re.IGNORECASE,
)


def main() -> int:
    sql = MIGRATION.read_text(encoding="utf-8")
    errors = verify_sql(sql)
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    print(
        "Research Lab persistence SQL verified: "
        f"{len(REQUIRED_TABLES)} tables, service-role RLS, append-only triggers, "
        "payment duplicate guard, key-ref storage, no fulfillment writes."
    )
    return 0


def verify_sql(sql: str) -> list[str]:
    errors: list[str] = []
    lowered = sql.lower()

    if "begin;" not in lowered or "commit;" not in lowered:
        errors.append("migration must be wrapped in BEGIN/COMMIT")
    if "does not touch fulfillment tables" not in lowered and "do not touch fulfillment tables" not in lowered:
        errors.append("migration must explicitly state fulfillment tables are out of scope")
    fulfillment_table_write = re.search(
        r"\b(?:CREATE|ALTER|DROP)\s+TABLE\s+(?:IF\s+(?:NOT\s+)?EXISTS\s+)?(?:public\.)?fulfillment_",
        sql,
        re.IGNORECASE,
    ) or re.search(
        r"\b(?:GRANT|REVOKE)\b[^;]*\bON\s+TABLE\s+(?:public\.)?fulfillment_",
        sql,
        re.IGNORECASE,
    ) or re.search(
        r"\b(?:INSERT\s+INTO|UPDATE|DELETE\s+FROM)\s+(?:public\.)?fulfillment_",
        sql,
        re.IGNORECASE,
    )
    if fulfillment_table_write:
        errors.append("migration must not modify fulfillment tables")

    for table in REQUIRED_TABLES:
        _require_table_contract(sql, table, errors)

    if "prevent_research_lab_append_only_mutation" not in sql:
        errors.append("append-only trigger function is missing")
    if "research_loop_start_payments_block_extrinsic_key" not in sql:
        errors.append("loop-start payment duplicate constraint is missing")
    if "UNIQUE (block_hash, extrinsic_index)" not in sql:
        errors.append("loop-start payment must be unique by block_hash/extrinsic_index")
    if "CHECK (payment_ref = block_hash || ':' || extrinsic_index::TEXT)" not in sql:
        errors.append("payment_ref canonical check is missing")

    for required in (
        "miner_openrouter_key_ref",
        "miner_openrouter_key_handling",
        "encrypted_ref",
        "ephemeral_ref",
        "provider_usage",
        "leadpoet_server_side",
    ):
        if required not in sql:
            errors.append(f"missing key-handling marker: {required}")

    for required_ref in (
        "REFERENCES public.research_trajectories",
        "REFERENCES public.research_trajectory_events",
        "REFERENCES public.evidence_bundles",
        "REFERENCES public.execution_traces",
    ):
        if required_ref not in sql:
            errors.append(f"missing evidence/trajectory linkage reference: {required_ref}")

    receipt_table = _extract_create_table(sql, "research_loop_receipts")
    if "REFERENCES public.execution_traces" in receipt_table:
        errors.append("research_loop_receipts.run_id must not reference execution_traces")
    if "Hosted Research Lab run UUID from the event-sourced run queue" not in receipt_table:
        errors.append("research_loop_receipts.run_id must document hosted run queue semantics")

    if FORBIDDEN_GRANT_RE.search(sql):
        errors.append("migration must not grant privileges to anon/authenticated")
    if re.search(r"GRANT\s+[^;]*(UPDATE|DELETE)[^;]*TO\s+service_role", sql, re.IGNORECASE):
        errors.append("append-only tables must not grant UPDATE/DELETE to service_role")
    if SECRET_COLUMN_RE.search(sql):
        errors.append("migration must not define raw provider secret columns")
    if "sk-or-" not in sql:
        errors.append("migration should include raw OpenRouter key tripwire checks")

    return errors


def _extract_create_table(sql: str, table: str) -> str:
    marker = f"CREATE TABLE IF NOT EXISTS public.{table}"
    start = sql.find(marker)
    if start == -1:
        return ""
    next_create = sql.find("CREATE TABLE IF NOT EXISTS public.", start + len(marker))
    if next_create == -1:
        return sql[start:]
    return sql[start:next_create]


def _require_table_contract(sql: str, table: str, errors: list[str]) -> None:
    create = f"CREATE TABLE IF NOT EXISTS public.{table}"
    if create not in sql:
        errors.append(f"{table}: missing CREATE TABLE")
    if f"REVOKE ALL ON TABLE public.{table} FROM anon, authenticated;" not in sql:
        errors.append(f"{table}: missing anon/authenticated revoke")
    if f"GRANT SELECT, INSERT ON TABLE public.{table} TO service_role;" not in sql:
        errors.append(f"{table}: missing service_role SELECT/INSERT grant")
    if f"ALTER TABLE public.{table} ENABLE ROW LEVEL SECURITY;" not in sql:
        errors.append(f"{table}: missing RLS enable")
    if f"ON public.{table}" not in sql or f"BEFORE UPDATE OR DELETE ON public.{table}" not in sql:
        errors.append(f"{table}: missing append-only mutation trigger")


if __name__ == "__main__":
    raise SystemExit(main())
