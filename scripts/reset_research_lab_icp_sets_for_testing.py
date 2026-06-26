#!/usr/bin/env python3
"""Reset Research Lab private ICP sets for a controlled test window.

Default mode is read-only. Pass --apply and --confirm-delete-all to delete all
existing rows from qualification_private_icp_sets and insert a fresh rolling
window of OpenRouter/Sonar-generated sets.

Required env:
  SUPABASE_URL
  SUPABASE_SERVICE_ROLE_KEY
  OPENROUTER_API_KEY
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import date, datetime, time as dt_time, timedelta, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gateway.tasks.icp_generator import generate_icps_with_openrouter  # noqa: E402


TABLE = "qualification_private_icp_sets"
REQUEST_ATTEMPTS = 4
REQUEST_BACKOFF_SECONDS = 1.5
DEFAULT_END_DATE = date(2026, 6, 26)
DEFAULT_DAYS = 10


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


async def _build_rows(*, end_date: date, days: int) -> list[dict[str, Any]]:
    if not os.environ.get("OPENROUTER_API_KEY", "").strip():
        raise SystemExit("OPENROUTER_API_KEY is required; reset script uses production OpenRouter/Sonar ICP generation")
    start_date = end_date - timedelta(days=days - 1)
    rows: list[dict[str, Any]] = []
    for offset in range(days):
        current_day = start_date + timedelta(days=offset)
        set_id = int(current_day.strftime("%Y%m%d"))
        result = await generate_icps_with_openrouter(set_id, total_icps=20)
        if not result:
            raise SystemExit(f"OpenRouter/Sonar ICP generation failed for set_id={set_id}; no rows were written")
        icps, industry_distribution, icp_set_hash = result
        active_from = datetime.combine(current_day, dt_time.min, tzinfo=timezone.utc)
        active_until = active_from + timedelta(days=1)
        rows.append(
            {
                "set_id": set_id,
                "icps": icps,
                "icp_set_hash": icp_set_hash,
                "industry_distribution": industry_distribution,
                "active_from": active_from.isoformat(),
                "active_until": active_until.isoformat(),
                "generation_seed": f"openrouter_sonar_reset_{end_date.isoformat()}_{set_id}",
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
        if len(icps) != 20:
            raise SystemExit(f"set_id={row.get('set_id')} expected 20 ICPs, got {len(icps)}")
        for icp in icps:
            employee_count = icp.get("employee_count")
            if not isinstance(employee_count, list) or not employee_count:
                raise SystemExit(f"set_id={row.get('set_id')} ICP {icp.get('icp_id')} employee_count must be a non-empty list")
            if "employee_count_buckets" in icp or "employee_counts" in icp:
                raise SystemExit(f"set_id={row.get('set_id')} ICP {icp.get('icp_id')} contains legacy employee count fields")


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
    parser.add_argument("--apply", action="store_true", help="Actually delete and insert Supabase rows")
    parser.add_argument("--confirm-delete-all", action="store_true", help="Required with --apply because this deletes all existing private ICP sets")
    args = parser.parse_args()

    end_date = date.fromisoformat(args.end_date)
    days = max(1, int(args.days))
    existing = _fetch_existing_summary()
    rows = asyncio.run(_build_rows(end_date=end_date, days=days))
    _validate_rows(rows)
    summary = {
        "apply": bool(args.apply),
        "end_date": end_date.isoformat(),
        "days": days,
        "new_set_ids": [row["set_id"] for row in rows],
        "new_active_set_ids": [row["set_id"] for row in rows if row["is_active"]],
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
