#!/usr/bin/env python3
"""Static checks for Research Lab promotion and public benchmark SQL."""

from __future__ import annotations

from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
MIGRATION = ROOT / "scripts" / "37-research-lab-promotion-and-public-benchmarks.sql"

REQUIRED_TABLES = (
    "research_lab_private_model_versions",
    "research_lab_private_model_version_events",
    "research_lab_candidate_promotion_events",
    "research_lab_private_repo_commit_events",
    "research_lab_public_benchmark_reports",
    "research_lab_public_benchmark_report_events",
)
REQUIRED_VIEWS = (
    "research_lab_private_model_version_current",
    "research_lab_public_benchmark_report_current",
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
        "Research Lab promotion SQL verified: private model lineage, promotion events, "
        "private repo commit events, public benchmark reports, current views, service-role RLS, append-only triggers."
    )
    return 0


def verify_sql(sql: str) -> list[str]:
    errors: list[str] = []
    lowered = sql.lower()
    if "begin;" not in lowered or "commit;" not in lowered:
        errors.append("migration must be wrapped in BEGIN/COMMIT")
    if "gateway promotion" not in lowered or "public benchmark reports" not in lowered:
        errors.append("migration must document gateway promotion and public benchmark reports")
    if re.search(r"\b(?:CREATE|ALTER|DROP)\s+TABLE\b[^;]*\bfulfillment_", sql, re.IGNORECASE):
        errors.append("migration must not modify fulfillment tables")
    if FORBIDDEN_GRANT_RE.search(sql):
        errors.append("migration must not grant privileges to anon/authenticated")
    if re.search(r"GRANT\s+[^;]*(UPDATE|DELETE)[^;]*TO\s+service_role", sql, re.IGNORECASE):
        errors.append("append-only tables must not grant UPDATE/DELETE to service_role")

    for table in REQUIRED_TABLES:
        _require_table(sql, table, errors)
    for view in REQUIRED_VIEWS:
        _require_view(sql, view, errors)

    for marker in (
        "stale_parent_detected",
        "rebase_queued",
        "active_version_created",
        "champion_reward_pending_uid",
        "champion_reward_created",
        "exact intent signals",
        "private_model_manifest_uri",
        "private_model_manifest_doc",
        "candidate_patch_manifest",
        "image_digest",
        "hidden_icp",
        "icp_plaintext",
        "WITH (security_invoker = true)",
    ):
        if marker not in sql:
            errors.append(f"missing marker: {marker}")
    return errors


def _require_table(sql: str, table: str, errors: list[str]) -> None:
    if f"CREATE TABLE IF NOT EXISTS public.{table}" not in sql:
        errors.append(f"{table}: missing CREATE TABLE")
    if f"REVOKE ALL ON TABLE public.{table} FROM anon, authenticated;" not in sql:
        errors.append(f"{table}: missing anon/authenticated revoke")
    if f"GRANT SELECT, INSERT ON TABLE public.{table} TO service_role;" not in sql:
        errors.append(f"{table}: missing service_role SELECT/INSERT grant")
    if f"ALTER TABLE public.{table} ENABLE ROW LEVEL SECURITY;" not in sql:
        errors.append(f"{table}: missing RLS enable")
    if f"BEFORE UPDATE OR DELETE ON public.{table}" not in sql:
        errors.append(f"{table}: missing append-only trigger")


def _require_view(sql: str, view: str, errors: list[str]) -> None:
    if f"CREATE OR REPLACE VIEW public.{view}" not in sql:
        errors.append(f"{view}: missing CREATE VIEW")
    if f"REVOKE ALL ON TABLE public.{view} FROM anon, authenticated;" not in sql:
        errors.append(f"{view}: missing anon/authenticated revoke")
    if f"GRANT SELECT ON TABLE public.{view} TO service_role;" not in sql:
        errors.append(f"{view}: missing service_role SELECT grant")


if __name__ == "__main__":
    raise SystemExit(main())
