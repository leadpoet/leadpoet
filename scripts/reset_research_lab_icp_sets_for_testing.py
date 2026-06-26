#!/usr/bin/env python3
"""Reset Research Lab private ICP sets for a controlled test window.

Default mode is read-only. Pass --apply and --confirm-delete-all to delete all
existing rows from qualification_private_icp_sets and insert a fresh rolling
window of Tasnimul's benchmark ICPs.

Required env:
  SUPABASE_URL
  SUPABASE_SERVICE_ROLE_KEY
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from datetime import date, datetime, time as dt_time, timedelta, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gateway.tasks.icp_generator import compute_icp_set_hash  # noqa: E402
from research_lab.employee_buckets import (  # noqa: E402
    DEFAULT_EMPLOYEE_BUCKET_RADIUS,
    normalize_employee_count_bucket,
    normalize_employee_count_buckets,
)


TABLE = "qualification_private_icp_sets"
REQUEST_ATTEMPTS = 4
REQUEST_BACKOFF_SECONDS = 1.5
DEFAULT_END_DATE = date(2026, 6, 26)
DEFAULT_DAYS = 10
TASNIMUL_BENCHMARK_ICPS: list[dict[str, Any]] = [
    {
        "icp_id": "icp_20260617_005",
        "industry": "Data and Analytics",
        "geography": "United States",
        "employee_count": "501-1,000",
        "required_attribute": "The company offers or provides Cloud-native data warehouse, lakehouse, and analytics platforms",
        "intent_signal": "Announced a strategic partnership",
        "intent_category": "PARTNERSHIP",
        "intent_max_age_days": 365,
        "bonus_intents": [],
    },
    {
        "icp_id": "icp_20260617_004",
        "industry": "Hardware",
        "geography": "United States, West Coast",
        "employee_count": "51-200",
        "required_attribute": "The company offers or provides Connected hardware devices with companion software for consumers and professionals",
        "intent_signal": "Launched or announced a new product",
        "intent_category": "PRODUCT_LAUNCH",
        "intent_max_age_days": 365,
        "bonus_intents": [],
    },
    {
        "icp_id": "icp_20260618_017",
        "industry": "Real Estate",
        "geography": "United States",
        "employee_count": "5,001-10,000",
        "required_attribute": "The company offers or provides Commercial real estate development and management",
        "intent_signal": "Expanded to new markets",
        "intent_category": "MARKET_EXPANSION",
        "intent_max_age_days": 365,
        "bonus_intents": [
            {
                "intent_signal": "Recent factory / facility / store opening",
                "intent_category": "FACILITY_OPENING",
                "intent_max_age_days": 365,
            }
        ],
    },
    {
        "icp_id": "icp_20260618_008",
        "industry": "Biotechnology",
        "geography": "United States, Northeast",
        "employee_count": "51-200",
        "required_attribute": "The company offers or provides Biotech therapeutics and drug discovery platforms",
        "intent_signal": "Recently raised funding",
        "intent_category": "FUNDING",
        "intent_max_age_days": 365,
        "bonus_intents": [
            {
                "intent_signal": "Achieved regulatory clearance or certification",
                "intent_category": "REGULATORY_CLEARANCE",
                "intent_max_age_days": 365,
            }
        ],
    },
    {
        "icp_id": "icp_20260619_019",
        "industry": "Education",
        "geography": "United States",
        "employee_count": "2-10",
        "required_attribute": "The company offers or provides Online learning and education technology platforms",
        "intent_signal": "Recently raised funding",
        "intent_category": "FUNDING",
        "intent_max_age_days": 365,
        "bonus_intents": [],
    },
    {
        "icp_id": "icp_20260619_012",
        "industry": "Manufacturing",
        "geography": "United States, Midwest",
        "employee_count": "10,001+",
        "required_attribute": "The company offers or provides Industrial and engineered products manufacturing",
        "intent_signal": "Recent factory / facility / store opening",
        "intent_category": "FACILITY_OPENING",
        "intent_max_age_days": 365,
        "bonus_intents": [],
    },
    {
        "icp_id": "icp_20260620_019",
        "industry": "Education",
        "geography": "United States",
        "employee_count": "11-50",
        "required_attribute": "The company offers or provides Online learning and education technology platforms",
        "intent_signal": "Launched or announced a new product",
        "intent_category": "PRODUCT_LAUNCH",
        "intent_max_age_days": 365,
        "bonus_intents": [],
    },
    {
        "icp_id": "icp_20260620_002",
        "industry": "Information Technology",
        "geography": "United States, West Coast",
        "employee_count": "201-500",
        "required_attribute": "The company offers or provides Cloud infrastructure and managed IT services",
        "intent_signal": "Expanded to new markets",
        "intent_category": "MARKET_EXPANSION",
        "intent_max_age_days": 365,
        "bonus_intents": [],
    },
    {
        "icp_id": "icp_20260621_003",
        "industry": "Artificial Intelligence",
        "geography": "United States",
        "employee_count": "201-500",
        "required_attribute": "The company offers or provides AI automation platform",
        "intent_signal": "Just closed a round",
        "intent_category": "FUNDING",
        "intent_max_age_days": 365,
        "bonus_intents": [
            {
                "intent_signal": "Launched or announced a new product",
                "intent_category": "PRODUCT_LAUNCH",
                "intent_max_age_days": 365,
            }
        ],
    },
    {
        "icp_id": "icp_20260621_020",
        "industry": "Transportation",
        "geography": "United States, West Coast",
        "employee_count": "10,001+",
        "required_attribute": "The company offers or provides transportation logistics platform",
        "intent_signal": "Recent factory / facility / store opening",
        "intent_category": "FACILITY_OPENING",
        "intent_max_age_days": 365,
        "bonus_intents": [
            {
                "intent_signal": "Announced a strategic partnership",
                "intent_category": "PARTNERSHIP",
                "intent_max_age_days": 365,
            }
        ],
    },
    {
        "icp_id": "icp_20260622_017",
        "industry": "Real Estate",
        "geography": "United States",
        "employee_count": "1,001-5,000",
        "required_attribute": "The company offers or provides Commercial real estate ownership, management, and proptech platforms",
        "intent_signal": "Acquired another company",
        "intent_category": "ACQUISITION",
        "intent_max_age_days": 365,
        "bonus_intents": [
            {
                "intent_signal": "Expanded to new markets",
                "intent_category": "MARKET_EXPANSION",
                "intent_max_age_days": 365,
            }
        ],
    },
    {
        "icp_id": "icp_20260622_012",
        "industry": "Manufacturing",
        "geography": "United States, Midwest",
        "employee_count": "10,001+",
        "required_attribute": "The company offers or provides Industrial manufacturing of components and equipment",
        "intent_signal": "Recent factory / facility / store opening",
        "intent_category": "FACILITY_OPENING",
        "intent_max_age_days": 365,
        "bonus_intents": [
            {
                "intent_signal": "Announced a strategic partnership",
                "intent_category": "PARTNERSHIP",
                "intent_max_age_days": 365,
            }
        ],
    },
    {
        "icp_id": "icp_20260623_004",
        "industry": "Hardware",
        "geography": "United States, West Coast",
        "employee_count": "51-200",
        "required_attribute": "The company offers or provides Hardware devices and electronics manufacturing",
        "intent_signal": "Recent factory / facility / store opening",
        "intent_category": "FACILITY_OPENING",
        "intent_max_age_days": 365,
        "bonus_intents": [],
    },
    {
        "icp_id": "icp_20260623_009",
        "industry": "Financial Services",
        "geography": "United States",
        "employee_count": "1,001-5,000",
        "required_attribute": "The company offers or provides Digital banking and financial technology services",
        "intent_signal": "Announced a strategic partnership",
        "intent_category": "PARTNERSHIP",
        "intent_max_age_days": 365,
        "bonus_intents": [],
    },
    {
        "icp_id": "icp_20260624_007",
        "industry": "Health Care",
        "geography": "United States, South",
        "employee_count": "1,001-5,000",
        "required_attribute": "The company offers or provides Multi-site outpatient and specialty care clinical services",
        "intent_signal": "Expanded to new markets",
        "intent_category": "MARKET_EXPANSION",
        "intent_max_age_days": 365,
        "bonus_intents": [
            {
                "intent_signal": "Recent factory / facility / store opening",
                "intent_category": "FACILITY_OPENING",
                "intent_max_age_days": 365,
            }
        ],
    },
    {
        "icp_id": "icp_20260624_015",
        "industry": "Advertising",
        "geography": "United States",
        "employee_count": "1,001-5,000",
        "required_attribute": "The company offers or provides Digital advertising technology and media monetization platforms",
        "intent_signal": "Launched or announced a new product",
        "intent_category": "PRODUCT_LAUNCH",
        "intent_max_age_days": 365,
        "bonus_intents": [
            {
                "intent_signal": "Announced a strategic partnership",
                "intent_category": "PARTNERSHIP",
                "intent_max_age_days": 365,
            }
        ],
    },
    {
        "icp_id": "icp_20260625_010",
        "industry": "Lending and Investments",
        "geography": "United States, South",
        "employee_count": "1,001-5,000",
        "required_attribute": "The company offers or provides Online consumer lending and financing platforms",
        "intent_signal": "Expanded to new markets",
        "intent_category": "MARKET_EXPANSION",
        "intent_max_age_days": 365,
        "bonus_intents": [],
    },
    {
        "icp_id": "icp_20260625_005",
        "industry": "Data and Analytics",
        "geography": "United States",
        "employee_count": "51-200",
        "required_attribute": "The company offers or provides Cloud-based data analytics and BI platforms",
        "intent_signal": "Just closed a round",
        "intent_category": "FUNDING",
        "intent_max_age_days": 365,
        "bonus_intents": [
            {
                "intent_signal": "Launched or announced a new product",
                "intent_category": "PRODUCT_LAUNCH",
                "intent_max_age_days": 365,
            }
        ],
    },
    {
        "icp_id": "icp_20260626_001",
        "industry": "Software",
        "geography": "United States",
        "employee_count": "1,001-5,000",
        "required_attribute": "The company offers or provides Business software platform",
        "intent_signal": "Recently raised funding",
        "intent_category": "FUNDING",
        "intent_max_age_days": 365,
        "bonus_intents": [
            {
                "intent_signal": "Hiring for senior engineering or sales roles",
                "intent_category": "HIRING",
                "intent_max_age_days": 90,
            }
        ],
    },
    {
        "icp_id": "icp_20260626_002",
        "industry": "Information Technology",
        "geography": "United States, West Coast",
        "employee_count": "51-200",
        "required_attribute": "The company offers or provides IT services and support platform",
        "intent_signal": "Recently raised funding",
        "intent_category": "FUNDING",
        "intent_max_age_days": 365,
        "bonus_intents": [
            {
                "intent_signal": "Expanded to new markets",
                "intent_category": "MARKET_EXPANSION",
                "intent_max_age_days": 365,
            }
        ],
    },
]


def _headers() -> dict[str, str]:
    key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "").strip()
    if not key:
        raise SystemExit("SUPABASE_SERVICE_ROLE_KEY is required")
    return {
        "apikey": key,
        "Authorization": f"Bearer {key}",
        "Accept": "application/json",
        "Content-Type": "application/json",
    }


def _base_url() -> str:
    url = os.environ.get("SUPABASE_URL", "").strip().rstrip("/")
    if not url:
        raise SystemExit("SUPABASE_URL is required")
    return url


def _request(method: str, path: str, *, body: Any | None = None, prefer: str = "") -> Any:
    data = None
    if body is not None:
        data = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    headers = _headers()
    if prefer:
        headers["Prefer"] = prefer
    req = urllib.request.Request(
        _base_url() + path,
        data=data,
        method=method,
        headers=headers,
    )
    last_error: BaseException | None = None
    for attempt in range(1, REQUEST_ATTEMPTS + 1):
        try:
            with urllib.request.urlopen(req, timeout=60) as response:
                raw = response.read().decode()
            return json.loads(raw) if raw else None
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")
            if exc.code not in {429, 500, 502, 503, 504}:
                raise SystemExit(f"Supabase {method} {path} failed: HTTP {exc.code}: {detail}") from exc
            last_error = RuntimeError(f"HTTP {exc.code}: {detail}")
        except (TimeoutError, urllib.error.URLError) as exc:
            last_error = exc
        if attempt < REQUEST_ATTEMPTS:
            sleep_seconds = REQUEST_BACKOFF_SECONDS * attempt
            print(
                f"retrying Supabase {method} {path} after transient error "
                f"({attempt}/{REQUEST_ATTEMPTS}): {last_error}",
                file=sys.stderr,
            )
            time.sleep(sleep_seconds)
    raise SystemExit(f"Supabase {method} {path} failed after {REQUEST_ATTEMPTS} attempts: {last_error}") from last_error


def _fetch_existing_summary() -> dict[str, Any]:
    params = urllib.parse.urlencode(
        {
            "select": "set_id,is_active,active_from,active_until",
            "order": "set_id.asc",
        }
    )
    rows = _request("GET", f"/rest/v1/{TABLE}?{params}") or []
    return {
        "existing_count": len(rows),
        "existing_set_ids": [row.get("set_id") for row in rows],
        "existing_active_set_ids": [row.get("set_id") for row in rows if row.get("is_active")],
    }


def _set_id_from_icp(icp: dict[str, Any]) -> int:
    icp_id = str(icp.get("icp_id") or "")
    parts = icp_id.split("_")
    if len(parts) < 3 or parts[0] != "icp" or not parts[1].isdigit():
        raise SystemExit(f"ICP {icp_id!r} must use icp_YYYYMMDD_NNN id format")
    return int(parts[1])


def _normalize_exact_icp(icp: dict[str, Any], *, radius: int, all_buckets: bool) -> dict[str, Any]:
    normalized = dict(icp)
    primary = normalize_employee_count_bucket(normalized.get("employee_count"), default=None)
    if not primary:
        raise SystemExit(f"ICP {normalized.get('icp_id')} has invalid employee_count={normalized.get('employee_count')!r}")
    normalized["employee_count"] = normalize_employee_count_buckets(
        normalized.get("employee_count"),
        primary_bucket=primary,
        radius=radius,
        all_buckets=all_buckets,
    )
    normalized.pop("employee_count_buckets", None)
    normalized.pop("employee_counts", None)
    return normalized


def _industry_distribution(icps: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(str(icp.get("industry") or "") for icp in icps)
    return dict(sorted((industry, count) for industry, count in counts.items() if industry))


def _source_icps_by_set_id(*, radius: int, all_buckets: bool) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    seen_ids: set[str] = set()
    for source_icp in TASNIMUL_BENCHMARK_ICPS:
        normalized = _normalize_exact_icp(source_icp, radius=radius, all_buckets=all_buckets)
        icp_id = str(normalized["icp_id"])
        if icp_id in seen_ids:
            raise SystemExit(f"duplicate source ICP id: {icp_id}")
        seen_ids.add(icp_id)
        grouped[_set_id_from_icp(normalized)].append(normalized)
    return {
        set_id: sorted(icps, key=lambda icp: str(icp.get("icp_id") or ""))
        for set_id, icps in sorted(grouped.items())
    }


def _build_rows(*, end_date: date, days: int, radius: int, all_buckets: bool) -> list[dict[str, Any]]:
    start_date = end_date - timedelta(days=days - 1)
    source_by_set_id = _source_icps_by_set_id(radius=radius, all_buckets=all_buckets)
    rows: list[dict[str, Any]] = []
    for offset in range(days):
        current_day = start_date + timedelta(days=offset)
        set_id = int(current_day.strftime("%Y%m%d"))
        icps = source_by_set_id.get(set_id, [])
        if not icps:
            raise SystemExit(f"no source ICPs found for set_id={set_id}")
        icp_set_hash = compute_icp_set_hash(icps)
        active_from = datetime.combine(current_day, dt_time.min, tzinfo=timezone.utc)
        active_until = active_from + timedelta(days=1)
        rows.append(
            {
                "set_id": set_id,
                "icps": icps,
                "icp_set_hash": icp_set_hash,
                "industry_distribution": _industry_distribution(icps),
                "active_from": active_from.isoformat(),
                "active_until": active_until.isoformat(),
                "generation_seed": f"tasnimul_exact_benchmark_icps_{end_date.isoformat()}_radius_{radius}",
                "is_active": current_day == end_date,
            }
        )
    return rows


def _validate_rows(rows: list[dict[str, Any]]) -> None:
    active_rows = [row for row in rows if row.get("is_active")]
    if len(active_rows) != 1:
        raise SystemExit(f"expected exactly one active row, got {len(active_rows)}")
    for row in rows:
        icps = row.get("icps") or []
        if not icps:
            raise SystemExit(f"set_id={row.get('set_id')} expected at least one ICP")
        for icp in icps:
            employee_count = icp.get("employee_count")
            if not isinstance(employee_count, list) or not employee_count:
                raise SystemExit(f"set_id={row.get('set_id')} ICP {icp.get('icp_id')} employee_count must be a non-empty list")
            if "employee_count_buckets" in icp or "employee_counts" in icp:
                raise SystemExit(f"set_id={row.get('set_id')} ICP {icp.get('icp_id')} contains legacy employee count fields")
    total_icps = sum(len(row.get("icps") or []) for row in rows)
    if total_icps != len(TASNIMUL_BENCHMARK_ICPS):
        raise SystemExit(f"expected {len(TASNIMUL_BENCHMARK_ICPS)} total ICPs, got {total_icps}")


def _delete_all_rows() -> None:
    _request(
        "DELETE",
        f"/rest/v1/{TABLE}?set_id=not.is.null",
        prefer="return=minimal",
    )


def _insert_rows(rows: list[dict[str, Any]]) -> None:
    _request(
        "POST",
        f"/rest/v1/{TABLE}",
        body=rows,
        prefer="return=minimal",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--end-date", default=DEFAULT_END_DATE.isoformat(), help="Last active day to create, YYYY-MM-DD")
    parser.add_argument("--days", type=int, default=DEFAULT_DAYS, help="Number of daily sets to create")
    parser.add_argument("--radius", type=int, default=DEFAULT_EMPLOYEE_BUCKET_RADIUS, help="Adjacent LinkedIn bucket radius for employee_count lists")
    parser.add_argument("--all-buckets", action="store_true", help="Use all generated LinkedIn employee buckets for every ICP")
    parser.add_argument("--apply", action="store_true", help="Actually delete and insert Supabase rows")
    parser.add_argument("--confirm-delete-all", action="store_true", help="Required with --apply because this deletes all existing private ICP sets")
    args = parser.parse_args()

    end_date = date.fromisoformat(args.end_date)
    days = max(1, int(args.days))
    radius = max(0, int(args.radius))
    existing = _fetch_existing_summary()
    rows = _build_rows(
        end_date=end_date,
        days=days,
        radius=radius,
        all_buckets=bool(args.all_buckets),
    )
    _validate_rows(rows)
    summary = {
        "apply": bool(args.apply),
        "all_buckets": bool(args.all_buckets),
        "end_date": end_date.isoformat(),
        "days": days,
        "employee_count_radius": radius,
        "new_set_ids": [row["set_id"] for row in rows],
        "new_active_set_ids": [row["set_id"] for row in rows if row["is_active"]],
        "new_icps_per_set": {str(row["set_id"]): len(row["icps"]) for row in rows},
        "new_hash_prefixes": {str(row["set_id"]): str(row["icp_set_hash"])[:16] for row in rows},
        **existing,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))

    if not args.apply:
        print("DRY RUN: pass --apply --confirm-delete-all to reset Supabase.")
        return 0
    if not args.confirm_delete_all:
        raise SystemExit("--confirm-delete-all is required with --apply")

    _delete_all_rows()
    _insert_rows(rows)
    print(f"deleted {existing['existing_count']} existing set(s); inserted {len(rows)} new set(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
