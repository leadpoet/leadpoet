#!/usr/bin/env python3
"""Repair the latest Research Lab public benchmark model-issue payload.

This is an operator-only one-off repair for the first public Research Lab
rebenchmark row written before report_doc schema 1.2. It deletes the stale
public benchmark report row/events and reinserts a corrected public report row
using the same source private benchmark bundle. The private benchmark bundle is
not modified; it remains the internal full-20-ICP diagnostic source.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
import os
import subprocess
from typing import Any, Mapping, Sequence
from urllib import error, parse, request
from uuid import uuid4


REPORTS_TABLE = "research_lab_public_benchmark_reports"
EVENTS_TABLE = "research_lab_public_benchmark_report_events"
REPORT_TRIGGER = "prevent_research_lab_public_benchmark_reports_mutation"
EVENT_TRIGGER = "prevent_research_lab_public_benchmark_report_events_mutation"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-id", help="Specific public report_id to repair. Defaults to latest published report.")
    parser.add_argument("--benchmark-date", help="Latest published report on this date, YYYY-MM-DD.")
    parser.add_argument("--emit-sql", metavar="PATH", help="Write the generated SQL to PATH, or '-' for stdout.")
    parser.add_argument("--apply", action="store_true", help="Execute generated SQL using SUPABASE_DB_URL or DATABASE_URL.")
    parser.add_argument(
        "--confirm-delete-replace",
        action="store_true",
        help="Required with --apply. Confirms old report rows/events will be deleted and replaced.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Only print a summary. This is the default.")
    args = parser.parse_args()

    if args.apply and not args.confirm_delete_replace:
        raise SystemExit("--apply requires --confirm-delete-replace")

    repair = build_repair(args)
    print(json.dumps(repair.summary, indent=2, sort_keys=True))

    if args.emit_sql:
        if args.emit_sql == "-":
            print(repair.sql)
        else:
            with open(args.emit_sql, "w", encoding="utf-8") as handle:
                handle.write(repair.sql)
            print(f"wrote SQL: {args.emit_sql}")

    if args.apply:
        db_url = os.environ.get("SUPABASE_DB_URL") or os.environ.get("DATABASE_URL")
        if not db_url:
            raise SystemExit("--apply requires SUPABASE_DB_URL or DATABASE_URL for direct Postgres access")
        subprocess.run(["psql", db_url, "-v", "ON_ERROR_STOP=1", "-f", "-"], input=repair.sql, text=True, check=True)
        print("applied delete-and-replace repair")
    return 0


@dataclass(frozen=True)
class RepairPlan:
    summary: dict[str, Any]
    sql: str


def build_repair(args: argparse.Namespace) -> RepairPlan:
    public_row = fetch_public_report(args)
    base_report = fetch_one(
        REPORTS_TABLE,
        {
            "select": "*",
            "report_id": f"eq.{public_row['report_id']}",
            "limit": "1",
        },
    )
    if not base_report:
        raise SystemExit(f"base public report row not found: {public_row['report_id']}")

    private_row = fetch_one(
        "research_lab_private_model_benchmark_current",
        {
            "select": "*",
            "benchmark_bundle_id": f"eq.{public_row['benchmark_bundle_id']}",
            "limit": "1",
        },
    )
    if not private_row:
        raise SystemExit(f"source private benchmark bundle not found: {public_row['benchmark_bundle_id']}")

    old_report_doc = public_row.get("report_doc")
    private_doc = private_row.get("score_summary_doc")
    if not isinstance(old_report_doc, Mapping):
        raise SystemExit("public report_doc is not an object")
    if not isinstance(private_doc, Mapping):
        raise SystemExit("private score_summary_doc is not an object")

    new_report_doc = build_corrected_report_doc(old_report_doc, private_doc)
    payload = {
        "benchmark_date": base_report["benchmark_date"],
        "benchmark_bundle_id": base_report["benchmark_bundle_id"],
        "private_model_artifact_hash": base_report["private_model_artifact_hash"],
        "private_model_manifest_hash": base_report["private_model_manifest_hash"],
        "rolling_window_hash": base_report["rolling_window_hash"],
        "aggregate_score": float(base_report["aggregate_score"]),
        "benchmark_attempt": int(base_report.get("benchmark_attempt") or 0),
        "benchmark_quality": str(base_report.get("benchmark_quality") or "passed"),
        "report_doc": new_report_doc,
    }
    report_hash = canonical_hash(payload)
    new_report_id = "public_benchmark:" + report_hash
    event_payload = {
        "report_id": new_report_id,
        "seq": 0,
        "event_type": "published",
        "report_status": "published",
        "event_doc": {
            "report_hash": report_hash,
            "benchmark_bundle_id": base_report["benchmark_bundle_id"],
        },
    }
    report_row = {
        "report_id": new_report_id,
        "schema_version": str(base_report.get("schema_version") or "1.0"),
        **payload,
        "report_hash": report_hash,
        "anchored_hash": report_hash,
    }
    event_row = {
        "event_id": str(uuid4()),
        "schema_version": "1.0",
        **event_payload,
        "anchored_hash": canonical_hash(event_payload),
    }
    sql = build_sql(old_report_id=str(base_report["report_id"]), report_row=report_row, event_row=event_row)
    return RepairPlan(
        summary={
            "apply": bool(args.apply),
            "old_report_id": str(base_report["report_id"]),
            "new_report_id": new_report_id,
            "benchmark_date": str(base_report["benchmark_date"]),
            "benchmark_attempt": int(base_report.get("benchmark_attempt") or 0),
            "old_report_doc_schema": old_report_doc.get("schema_version"),
            "new_report_doc_schema": new_report_doc.get("schema_version"),
            "new_model_issue_counts": new_report_doc.get("model_issue_counts"),
            "new_failure_category_counts": new_report_doc.get("failure_category_counts"),
            "new_zero_lead_icp_count": new_report_doc.get("zero_lead_icp_count"),
            "new_low_intent_fit_icp_count": new_report_doc.get("low_intent_fit_icp_count"),
            "new_low_icp_fit_count": new_report_doc.get("low_icp_fit_count"),
        },
        sql=sql,
    )


def build_corrected_report_doc(old_report_doc: Mapping[str, Any], private_doc: Mapping[str, Any]) -> dict[str, Any]:
    public_icps = old_report_doc.get("public_icps")
    summaries = private_doc.get("per_icp_summaries")
    if not isinstance(public_icps, Sequence) or isinstance(public_icps, (str, bytes)):
        raise SystemExit("public report_doc.public_icps is not an array")
    if not isinstance(summaries, Sequence) or isinstance(summaries, (str, bytes)):
        raise SystemExit("private score_summary_doc.per_icp_summaries is not an array")

    summary_by_ref = {
        str(summary.get("icp_ref")): summary
        for summary in summaries
        if isinstance(summary, Mapping) and summary.get("icp_ref")
    }
    public_by_ref = {
        str(icp.get("icp_ref")): icp
        for icp in public_icps
        if isinstance(icp, Mapping) and icp.get("icp_ref")
    }
    missing = sorted(ref for ref in public_by_ref if ref not in summary_by_ref)
    if missing:
        raise SystemExit(f"public ICP refs missing from private summaries: {missing[:5]}")

    issue_counts: Counter[str] = Counter()
    failure_counts: Counter[str] = Counter()
    issue_public_icps: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for ref, public_icp in public_by_ref.items():
        summary = summary_by_ref[ref]
        failures = normalized_failure_categories(diagnostics(summary).get("failure_categories") or [])
        failure_counts.update(failures)
        for issue_key in issue_keys(summary):
            issue_counts[issue_key] += 1
            issue_public_icps[issue_key].append(issue_public_icp_entry(public_icp, summary))

    corrected = dict(old_report_doc)
    corrected.pop("report_public_hash", None)
    corrected["schema_version"] = "1.2"
    corrected["failure_category_counts"] = dict(sorted(failure_counts.items()))
    corrected["model_issue_counts"] = dict(sorted(issue_counts.items()))
    corrected["model_issue_public_icps"] = {
        key: sorted(rows, key=lambda row: int(row.get("item_rank") or 0))
        for key, rows in sorted(issue_public_icps.items())
    }
    corrected["zero_lead_icp_count"] = int(issue_counts.get("zero_company_results", 0))
    corrected["low_intent_fit_icp_count"] = int(issue_counts.get("low_intent_fit", 0))
    corrected["low_icp_fit_count"] = int(issue_counts.get("icp_or_geo_mismatch", 0))

    if isinstance(corrected.get("icp_buckets"), list):
        for bucket in corrected["icp_buckets"]:
            if not isinstance(bucket, dict):
                continue
            ref = str(bucket.get("icp_ref") or "")
            if ref in public_by_ref and ref in summary_by_ref:
                bucket["failure_categories"] = normalized_failure_categories(
                    diagnostics(summary_by_ref[ref]).get("failure_categories") or []
                )
            else:
                bucket["failure_categories"] = []

    corrected["report_public_hash"] = canonical_hash(corrected)
    return corrected


def issue_keys(summary: Mapping[str, Any]) -> list[str]:
    diag = diagnostics(summary)
    keys = set(normalized_failure_categories(diag.get("failure_categories") or []))
    company_count = int(summary.get("company_count") or 0)
    if company_count <= 0:
        keys.add("zero_company_results")
    elif not keys and float(diag.get("avg_intent_signal_final") or 0.0) < 15.0:
        keys.add("low_intent_fit")
    return sorted(keys)


def diagnostics(summary: Mapping[str, Any]) -> Mapping[str, Any]:
    value = summary.get("diagnostics")
    return value if isinstance(value, Mapping) else {}


def normalized_failure_categories(value: Any) -> list[str]:
    if isinstance(value, str):
        raw = [value]
    elif isinstance(value, Sequence):
        raw = [str(item) for item in value if str(item).strip()]
    else:
        raw = []
    categories = {item.strip() for item in raw if item.strip()}
    if "provider_http_4xx" in categories or "provider_http_5xx" in categories:
        categories.discard("runtime_provider_error")
    return sorted(categories)


def issue_public_icp_entry(public_icp: Mapping[str, Any], summary: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "item_rank": int(public_icp.get("item_rank") or 0),
        "icp_ref": str(public_icp.get("icp_ref") or ""),
        "icp_hash": str(public_icp.get("icp_hash") or ""),
        "set_id": int(public_icp.get("set_id") or 0),
        "day_index": int(public_icp.get("day_index") or 0),
        "day_rank": int(public_icp.get("day_rank") or 0),
        "industry_bucket": str(summary.get("industry") or "unspecified"),
        "score": round(float(summary.get("score") or 0.0), 6),
        "company_count": int(summary.get("company_count") or 0),
    }


def build_sql(*, old_report_id: str, report_row: Mapping[str, Any], event_row: Mapping[str, Any]) -> str:
    report_columns = [
        "report_id",
        "schema_version",
        "benchmark_date",
        "benchmark_bundle_id",
        "private_model_artifact_hash",
        "private_model_manifest_hash",
        "rolling_window_hash",
        "aggregate_score",
        "benchmark_attempt",
        "benchmark_quality",
        "report_doc",
        "report_hash",
        "anchored_hash",
    ]
    event_columns = [
        "event_id",
        "schema_version",
        "report_id",
        "seq",
        "event_type",
        "report_status",
        "event_doc",
        "anchored_hash",
    ]
    return "\n".join(
        [
            "BEGIN;",
            "",
            f"DROP TRIGGER IF EXISTS {EVENT_TRIGGER} ON public.{EVENTS_TABLE};",
            f"DROP TRIGGER IF EXISTS {REPORT_TRIGGER} ON public.{REPORTS_TABLE};",
            "",
            f"DELETE FROM public.{EVENTS_TABLE}",
            f"WHERE report_id = {sql_literal(old_report_id)};",
            "",
            f"DELETE FROM public.{REPORTS_TABLE}",
            f"WHERE report_id = {sql_literal(old_report_id)};",
            "",
            insert_sql(REPORTS_TABLE, report_columns, report_row),
            "",
            insert_sql(EVENTS_TABLE, event_columns, event_row),
            "",
            f"CREATE TRIGGER {REPORT_TRIGGER}",
            f"BEFORE UPDATE OR DELETE ON public.{REPORTS_TABLE}",
            "FOR EACH ROW EXECUTE FUNCTION public.prevent_research_lab_append_only_mutation();",
            "",
            f"CREATE TRIGGER {EVENT_TRIGGER}",
            f"BEFORE UPDATE OR DELETE ON public.{EVENTS_TABLE}",
            "FOR EACH ROW EXECUTE FUNCTION public.prevent_research_lab_append_only_mutation();",
            "",
            "COMMIT;",
            "",
        ]
    )


def insert_sql(table: str, columns: Sequence[str], row: Mapping[str, Any]) -> str:
    values = ",\n    ".join(sql_value(row[column]) for column in columns)
    cols = ", ".join(columns)
    return f"INSERT INTO public.{table} ({cols})\nVALUES (\n    {values}\n);"


def sql_value(value: Any) -> str:
    if isinstance(value, (dict, list)):
        return f"{sql_literal(json.dumps(value, sort_keys=True, separators=(',', ':')))}::jsonb"
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return str(value)
    if value is None:
        return "NULL"
    return sql_literal(str(value))


def sql_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def fetch_public_report(args: argparse.Namespace) -> dict[str, Any]:
    params = {
        "select": "*",
        "current_report_status": "eq.published",
        "order": "benchmark_date.desc,created_at.desc",
        "limit": "1",
    }
    if args.report_id:
        params["report_id"] = f"eq.{args.report_id}"
    if args.benchmark_date:
        params["benchmark_date"] = f"eq.{args.benchmark_date}"
    rows = fetch("research_lab_public_benchmark_report_current", params)
    if not rows:
        raise SystemExit("no matching published public benchmark report found")
    return dict(rows[0])


def fetch_one(table: str, params: Mapping[str, str]) -> dict[str, Any] | None:
    rows = fetch(table, params)
    return dict(rows[0]) if rows else None


def fetch(table: str, params: Mapping[str, str]) -> list[dict[str, Any]]:
    url = os.environ.get("SUPABASE_URL") or os.environ.get("NEXT_PUBLIC_SUPABASE_URL")
    key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY") or os.environ.get("SUPABASE_SECRET_KEY")
    if not url or not key:
        raise SystemExit("set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY")
    encoded = parse.urlencode(params)
    req = request.Request(f"{url.rstrip('/')}/rest/v1/{table}?{encoded}")
    req.add_header("apikey", key)
    req.add_header("Authorization", f"Bearer {key}")
    req.add_header("Accept", "application/json")
    try:
        with request.urlopen(req, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))
    except error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise SystemExit(f"Supabase read failed for {table}: HTTP {exc.code}: {body[:1000]}") from exc


def canonical_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
