#!/usr/bin/env python3
"""Normalize current private ICP sets to the private sourcing-model contract.

Default mode is read-only. Pass --apply to update Supabase.
Required env:
  SUPABASE_URL
  SUPABASE_SERVICE_ROLE_KEY
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gateway.tasks.icp_generator import (  # noqa: E402
    canonicalize_generated_icp,
    compute_icp_set_hash,
)


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


def _request(method: str, path: str, *, body: Any | None = None) -> Any:
    data = None
    if body is not None:
        data = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    req = urllib.request.Request(
        _base_url() + path,
        data=data,
        method=method,
        headers=_headers(),
    )
    try:
        with urllib.request.urlopen(req, timeout=45) as response:
            raw = response.read().decode()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")
        raise SystemExit(f"Supabase {method} {path} failed: HTTP {exc.code}: {detail}") from exc
    return json.loads(raw) if raw else None


def _fetch_sets(limit: int, set_ids: list[int]) -> list[dict[str, Any]]:
    select = "set_id,icps,icp_set_hash,is_active,active_from,active_until"
    if set_ids:
        encoded_ids = ",".join(str(set_id) for set_id in set_ids)
        path = (
            "/rest/v1/qualification_private_icp_sets?"
            + urllib.parse.urlencode({"select": select, "order": "set_id.desc"})
            + f"&set_id=in.({encoded_ids})"
        )
    else:
        path = "/rest/v1/qualification_private_icp_sets?" + urllib.parse.urlencode(
            {"select": select, "order": "set_id.desc", "limit": str(limit)}
        )
    rows = _request("GET", path)
    return rows or []


def _normalize_set(row: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    before_icps = row.get("icps") or []
    after_icps: list[dict[str, Any]] = []
    changed_fields = Counter()
    for icp in before_icps:
        if not isinstance(icp, dict):
            after_icps.append(icp)
            continue
        before = deepcopy(icp)
        industry = str(icp.get("industry") or "")
        sub_industry = str(icp.get("sub_industry") or "")
        after = canonicalize_generated_icp(icp, industry=industry, sub_industry=sub_industry)
        for field in (
            "employee_count",
            "required_attribute",
            "intent_signal",
            "intent_category",
            "intent_max_age_days",
            "bonus_intents",
            "intent_signals",
            "prompt",
        ):
            if before.get(field) != after.get(field):
                changed_fields[field] += 1
        after_icps.append(after)
    new_hash = compute_icp_set_hash(after_icps)
    return after_icps, {
        "set_id": row.get("set_id"),
        "icp_count": len(after_icps),
        "old_hash": row.get("icp_set_hash"),
        "new_hash": new_hash,
        "hash_changed": row.get("icp_set_hash") != new_hash,
        "changed_fields": dict(sorted(changed_fields.items())),
        "employee_counts": dict(sorted(Counter(str(icp.get("employee_count") or "") for icp in after_icps if isinstance(icp, dict)).items())),
    }


def _update_set(set_id: int, icps: list[dict[str, Any]], icp_set_hash: str) -> None:
    path = (
        "/rest/v1/qualification_private_icp_sets?"
        + urllib.parse.urlencode({"set_id": f"eq.{set_id}"})
    )
    req = urllib.request.Request(
        _base_url() + path,
        data=json.dumps({"icps": icps, "icp_set_hash": icp_set_hash}, sort_keys=True, separators=(",", ":")).encode(),
        method="PATCH",
        headers={**_headers(), "Prefer": "return=minimal"},
    )
    try:
        with urllib.request.urlopen(req, timeout=45) as response:
            response.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")
        raise SystemExit(f"Supabase PATCH set_id={set_id} failed: HTTP {exc.code}: {detail}") from exc


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=10, help="Newest set count when --set-id is omitted")
    parser.add_argument("--set-id", type=int, action="append", default=[], help="Specific set_id to normalize; repeatable")
    parser.add_argument("--apply", action="store_true", help="Actually update Supabase")
    args = parser.parse_args()

    rows = _fetch_sets(max(1, args.limit), args.set_id)
    summaries = []
    updates: list[tuple[int, list[dict[str, Any]], str]] = []
    for row in rows:
        normalized_icps, summary = _normalize_set(row)
        summaries.append(summary)
        if summary["hash_changed"]:
            updates.append((int(row["set_id"]), normalized_icps, str(summary["new_hash"])))

    print(json.dumps({"apply": bool(args.apply), "set_count": len(rows), "summaries": summaries}, indent=2, sort_keys=True))
    if not args.apply:
        print("DRY RUN: pass --apply to update Supabase.")
        return 0
    for set_id, icps, new_hash in updates:
        _update_set(set_id, icps, new_hash)
        print(f"updated set_id={set_id} icp_set_hash={new_hash}")
    print(f"updated {len(updates)} set(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
