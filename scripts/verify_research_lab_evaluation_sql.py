#!/usr/bin/env python3
"""Static smoke checks for Research Lab evaluation score-bundle SQL."""

from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MIGRATION = ROOT / "scripts" / "30-research-lab-evaluation-bundles.sql"
CANDIDATE_MIGRATION = ROOT / "scripts" / "33-research-lab-candidate-evaluation-queue.sql"

REQUIRED_TABLES = (
    "research_evaluation_score_bundles",
    "research_evaluation_score_bundle_events",
)
REQUIRED_VIEWS = ("research_evaluation_score_bundle_current",)
CANDIDATE_REQUIRED_TABLES = (
    "research_lab_candidate_artifacts",
    "research_lab_candidate_evaluation_events",
)
CANDIDATE_REQUIRED_VIEWS = ("research_lab_candidate_evaluation_current",)

FORBIDDEN_GRANT_RE = re.compile(
    r"\bGRANT\b(?!\s+EXECUTE\b)[^;]*\b(?:anon|authenticated)\b",
    re.IGNORECASE | re.DOTALL,
)


def main() -> int:
    sql = MIGRATION.read_text(encoding="utf-8")
    candidate_sql = CANDIDATE_MIGRATION.read_text(encoding="utf-8")
    errors = verify_sql(sql)
    errors.extend(verify_candidate_sql(candidate_sql))
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    print(
        "Research Lab evaluation SQL verified: score-bundle table, event table, "
        "candidate queue tables, current views, service-role RLS, append-only triggers, no fulfillment writes."
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
    if re.search(r"GRANT\s+[^;]*(UPDATE|DELETE)[^;]*TO\s+service_role", sql, re.IGNORECASE):
        errors.append("append-only evaluation tables must not grant UPDATE/DELETE to service_role")

    for table in REQUIRED_TABLES:
        _require_table(sql, table, errors)
    for view in REQUIRED_VIEWS:
        _require_view(sql, view, errors)

    for marker in (
        "candidate_artifact_hash <> parent_artifact_hash",
        "score_bundle_doc::TEXT !~*",
        "private_repo",
        "judge_prompt",
        "score_bundle_hash",
        "anchored_hash = score_bundle_hash",
        "WITH (security_invoker = true)",
    ):
        if marker not in sql:
            errors.append(f"missing marker: {marker}")
    return errors


def verify_candidate_sql(sql: str) -> list[str]:
    errors: list[str] = []
    lowered = sql.lower()
    if "begin;" not in lowered or "commit;" not in lowered:
        errors.append("candidate migration must be wrapped in BEGIN/COMMIT")
    if "validator" not in lowered or "candidate" not in lowered:
        errors.append("candidate migration must document validator candidate evaluation ownership")
    if re.search(r"\b(?:CREATE|ALTER|DROP)\s+TABLE\b[^;]*\bfulfillment_", sql, re.IGNORECASE):
        errors.append("candidate migration must not modify fulfillment tables")
    if FORBIDDEN_GRANT_RE.search(sql):
        errors.append("candidate migration must not grant privileges to anon/authenticated")
    if re.search(r"GRANT\s+[^;]*(UPDATE|DELETE)[^;]*TO\s+service_role", sql, re.IGNORECASE):
        errors.append("append-only candidate tables must not grant UPDATE/DELETE to service_role")

    for table in CANDIDATE_REQUIRED_TABLES:
        _require_table(sql, table, errors)
    for view in CANDIDATE_REQUIRED_VIEWS:
        _require_view(sql, view, errors)

    for marker in (
        "candidate_status IN",
        "'queued'",
        "'assigned'",
        "'evaluating'",
        "'scored'",
        "private_model_manifest_doc::TEXT !~*",
        "candidate_patch_manifest::TEXT !~*",
        "judge_prompt",
        "WITH (security_invoker = true)",
    ):
        if marker not in sql:
            errors.append(f"candidate migration missing marker: {marker}")
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
