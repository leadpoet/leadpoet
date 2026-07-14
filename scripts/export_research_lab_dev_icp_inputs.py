#!/usr/bin/env python3
"""Export eight deterministic retired ICPs and the full current holdout guard."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research_lab.canonical import sha256_json  # noqa: E402
from research_lab.eval.dev_eval import build_dev_icp_set, intent_signal_signature  # noqa: E402


def _rows(client: Any, table: str, columns: str, *, order: str, desc: bool) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    offset = 0
    while True:
        response = (
            client.table(table)
            .select(columns)
            .order(order, desc=desc)
            .range(offset, offset + 999)
            .execute()
        )
        batch = [dict(row) for row in (getattr(response, "data", None) or [])]
        output.extend(batch)
        if len(batch) < 1000:
            return output
        offset += 1000


def _item(set_id: int, icp: Mapping[str, Any]) -> dict[str, Any]:
    icp_doc = dict(icp)
    icp_id = str(icp_doc.get("icp_id") or "").strip()
    if not icp_id:
        raise ValueError(f"set {set_id} carries an ICP without icp_id")
    return {
        "set_id": int(set_id),
        "icp": icp_doc,
        "icp_ref": f"qualification_private_icp_sets:{int(set_id)}:{icp_id}",
        "icp_hash": sha256_json({"icp": icp_doc}),
        "intent_signal_signature": intent_signal_signature(icp_doc),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="dev_icp_inputs")
    parser.add_argument("--seed", default="research-lab-dev-v1")
    args = parser.parse_args()

    url = os.getenv("SUPABASE_URL", "").strip()
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "").strip()
    if not url or not key:
        print("ERROR: SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY are required")
        return 1
    try:
        from supabase import create_client
    except Exception as exc:
        print(f"ERROR: supabase client unavailable: {exc}")
        return 1

    client = create_client(url, key)
    windows = _rows(
        client,
        "research_lab_rolling_icp_windows",
        "rolling_window_hash,required_days,window_doc,created_at",
        order="created_at",
        desc=True,
    )
    if not windows:
        print("ERROR: no finalized Research Lab rolling window exists")
        return 1
    window = windows[0]
    window_doc = window.get("window_doc") if isinstance(window.get("window_doc"), Mapping) else {}
    set_ids = [
        int(row.get("set_id") or 0)
        for row in (window_doc.get("sets") or ())
        if isinstance(row, Mapping) and int(row.get("set_id") or 0) > 0
    ]
    fresh_set_id = int(window_doc.get("fresh_set_id") or (max(set_ids) if set_ids else 0))
    required_days = max(1, int(window.get("required_days") or window_doc.get("required_days") or 10))
    if fresh_set_id <= 0:
        print("ERROR: finalized window does not identify its fresh set")
        return 1

    private_sets = _rows(
        client,
        "qualification_private_icp_sets",
        "set_id,icps,icp_set_hash,active_from,active_until,is_active",
        order="set_id",
        desc=True,
    )
    eligible_sets = [row for row in private_sets if int(row.get("set_id") or 0) <= fresh_set_id]
    horizon_rows = eligible_sets[:required_days]
    horizon_ids = {int(row["set_id"]) for row in horizon_rows}
    retired_rows = [row for row in eligible_sets if int(row["set_id"]) not in horizon_ids]
    if len(horizon_rows) < required_days:
        print(f"ERROR: current horizon requires {required_days} sets, found {len(horizon_rows)}")
        return 1

    exclusions: set[str] = set()
    for row in horizon_rows:
        for icp in row.get("icps") or ():
            if not isinstance(icp, Mapping):
                continue
            item = _item(int(row["set_id"]), icp)
            exclusions.update(
                {item["icp_ref"], item["icp_hash"], item["intent_signal_signature"]}
            )
    retired_items = [
        _item(int(row["set_id"]), icp)
        for row in retired_rows
        for icp in (row.get("icps") or ())
        if isinstance(icp, Mapping)
    ]
    try:
        selected = build_dev_icp_set(
            retired_items,
            exclude_window_hashes=sorted(exclusions),
            size=8,
            seed=str(args.seed),
        )
    except Exception as exc:
        print(f"ERROR: could not select eight retired ICPs: {exc}")
        return 1

    out = Path(args.out_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    source_doc = {
        "schema_version": "research_lab.dev_icp_export.v1",
        "selection_seed": str(args.seed),
        "rolling_window_hash": str(window.get("rolling_window_hash") or ""),
        "fresh_set_id": fresh_set_id,
        "horizon_set_ids": sorted(horizon_ids),
        "retired_set_count": len(retired_rows),
        "items": list(selected.items),
        "dev_set_hash": selected.dev_set_hash,
        "selection_manifest": dict(selected.manifest),
    }
    exclusion_doc = {
        "schema_version": "research_lab.dev_holdout_exclusions.v1",
        "rolling_window_hash": str(window.get("rolling_window_hash") or ""),
        "horizon_set_ids": sorted(horizon_ids),
        "hashes": sorted(exclusions),
        "exclusion_hash": sha256_json(sorted(exclusions)),
    }
    (out / "source_icps.json").write_text(
        json.dumps(source_doc, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (out / "holdout_window_hashes.json").write_text(
        json.dumps(exclusion_doc, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"selected_icps={len(selected.items)}")
    print(f"dev_set_hash={selected.dev_set_hash}")
    print(f"holdout_exclusions={len(exclusions)}")
    print(f"output_dir={out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
