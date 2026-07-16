#!/usr/bin/env python3
"""Export the completed current-day rebenchmark bank for inner-loop replay."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research_lab.eval.dev_eval import build_current_day_dev_bank  # noqa: E402
from gateway.research_lab.icp_window import (  # noqa: E402
    reconstruct_icp_window_from_doc,
)
from gateway.research_lab.config import (  # noqa: E402
    MAX_RESEARCH_LAB_GIT_TREE_ICP_COUNT,
    ResearchLabGitTreeConfig,
    ResearchLabGitTreeConfigError,
)


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


def _category_by_ref(score_summary_doc: Mapping[str, Any]) -> dict[str, str]:
    assignment = score_summary_doc.get("category_assignment")
    if isinstance(assignment, Mapping):
        rows = assignment.get("items")
        if isinstance(rows, list):
            return {
                str(row.get("icp_ref") or ""): str(row.get("category") or "")
                for row in rows
                if isinstance(row, Mapping) and str(row.get("icp_ref") or "")
            }
    split = score_summary_doc.get("visibility_split")
    rows = split.get("items") if isinstance(split, Mapping) else None
    if isinstance(rows, list):
        return {
            str(row.get("icp_ref") or ""): str(
                row.get("category") or row.get("visibility") or ""
            )
            for row in rows
            if isinstance(row, Mapping) and str(row.get("icp_ref") or "")
        }
    return {}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="dev_icp_inputs")
    parser.add_argument(
        "--benchmark-date",
        default=datetime.now(timezone.utc).date().isoformat(),
        help="UTC benchmark date to export (default: current UTC date)",
    )
    parser.add_argument(
        "--expected-private-model-manifest-hash",
        default="",
        help="Require the completed baseline to use this active model manifest",
    )
    # Retained so older operator wrappers do not fail. Selection is per tree,
    # not during daily bank export, so this value is intentionally unused.
    parser.add_argument("--seed", default="research-lab-dev-v1")
    args = parser.parse_args()

    try:
        configured_icp_count = (
            ResearchLabGitTreeConfig.from_env().live_max_icps_per_node
        )
    except ResearchLabGitTreeConfigError as exc:
        print(f"ERROR: invalid Git-tree configuration: {exc}")
        return 1
    if not 1 <= configured_icp_count <= MAX_RESEARCH_LAB_GIT_TREE_ICP_COUNT:
        print("ERROR: configured Git-tree development ICP count is invalid")
        return 1

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
    benchmark_rows = _rows(
        client,
        "research_lab_private_model_benchmark_current",
        (
            "benchmark_bundle_id,benchmark_bundle_hash,benchmark_date,"
            "private_model_manifest_hash,rolling_window_hash,evaluation_epoch,"
            "benchmark_quality,score_summary_doc,current_benchmark_status,created_at"
        ),
        order="created_at",
        desc=True,
    )
    expected_manifest_hash = str(
        args.expected_private_model_manifest_hash or ""
    ).strip()
    eligible_benchmarks = [
        row
        for row in benchmark_rows
        if str(row.get("benchmark_date") or "") == str(args.benchmark_date)
        and str(row.get("current_benchmark_status") or "") == "completed"
        and str(row.get("benchmark_quality") or "") == "passed"
        and (
            not expected_manifest_hash
            or str(row.get("private_model_manifest_hash") or "")
            == expected_manifest_hash
        )
    ]
    if not eligible_benchmarks:
        print(
            "ERROR: no completed current-day Research Lab baseline matches "
            "the requested active model"
        )
        return 1
    benchmark = eligible_benchmarks[0]
    summary_doc = (
        benchmark.get("score_summary_doc")
        if isinstance(benchmark.get("score_summary_doc"), Mapping)
        else {}
    )
    summaries = summary_doc.get("per_icp_summaries")
    if not isinstance(summaries, list) or not summaries:
        print("ERROR: completed current-day baseline has no per-ICP scores")
        return 1
    window_hash = str(benchmark.get("rolling_window_hash") or "")
    windows = _rows(
        client,
        "research_lab_rolling_icp_windows",
        "rolling_window_hash,window_doc,created_at",
        order="created_at",
        desc=True,
    )
    matching_windows = [
        row for row in windows
        if str(row.get("rolling_window_hash") or "") == window_hash
    ]
    if len(matching_windows) != 1:
        print("ERROR: completed baseline does not have one exact rolling window")
        return 1
    window = matching_windows[0]
    window_doc = window.get("window_doc") if isinstance(window.get("window_doc"), Mapping) else {}
    set_ids = [
        int(row.get("set_id") or 0)
        for row in (window_doc.get("sets") or ())
        if isinstance(row, Mapping) and int(row.get("set_id") or 0) > 0
    ]
    if not set_ids:
        print("ERROR: finalized window does not identify its source sets")
        return 1

    private_sets = _rows(
        client,
        "qualification_private_icp_sets",
        "set_id,icps,icp_set_hash,active_from,active_until,is_active",
        order="set_id",
        desc=True,
    )
    source_sets = [
        row for row in private_sets if int(row.get("set_id") or 0) in set(set_ids)
    ]
    if len({int(row.get("set_id") or 0) for row in source_sets}) != len(set(set_ids)):
        print("ERROR: one or more current-day rolling-window source sets are missing")
        return 1
    try:
        reconstructed = reconstruct_icp_window_from_doc(source_sets, window_doc)
        if reconstructed.window_hash != window_hash:
            raise ValueError("reconstructed rolling-window hash differs")
        summary_by_ref = {
            str(row.get("icp_ref") or ""): row
            for row in summaries
            if isinstance(row, Mapping) and str(row.get("icp_ref") or "")
        }
        if set(summary_by_ref) != set(reconstructed.item_refs):
            raise ValueError(
                "baseline per-ICP coverage differs from its rolling window"
            )
        category_by_ref = _category_by_ref(summary_doc)
        items: list[dict[str, Any]] = []
        for item in reconstructed.benchmark_items:
            ref = str(item["icp_ref"])
            summary = summary_by_ref[ref]
            if str(summary.get("icp_hash") or "") != str(item["icp_hash"]):
                raise ValueError(f"baseline ICP hash differs for {ref}")
            items.append(
                {
                    **dict(item),
                    "baseline_score": float(summary["score"]),
                    "benchmark_category": category_by_ref.get(
                        ref, "unclassified"
                    ),
                }
            )
        bank = build_current_day_dev_bank(
            items,
            benchmark_date=str(benchmark["benchmark_date"]),
            benchmark_bundle_id=str(benchmark["benchmark_bundle_id"]),
            benchmark_bundle_hash=str(benchmark["benchmark_bundle_hash"]),
            rolling_window_hash=window_hash,
            private_model_manifest_hash=str(
                benchmark["private_model_manifest_hash"]
            ),
            evaluation_epoch=int(benchmark.get("evaluation_epoch") or 0),
        )
    except Exception as exc:
        print(
            "ERROR: could not construct the completed current-day benchmark "
            f"bank: {exc}"
        )
        return 1
    if len(bank.items) < configured_icp_count:
        print(
            "ERROR: current-day benchmark bank is smaller than the configured "
            "per-candidate ICP count"
        )
        return 1

    out = Path(args.out_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    source_doc = {
        "schema_version": "research_lab.dev_icp_export.v2",
        "configured_dev_icp_count": int(configured_icp_count),
        "items": list(bank.items),
        "daily_bank_manifest": dict(bank.manifest),
    }
    (out / "source_icps.json").write_text(
        json.dumps(source_doc, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (out / "daily_benchmark_identity.json").write_text(
        json.dumps(bank.manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"benchmark_date={bank.manifest['benchmark_date']}")
    print(f"benchmark_bundle_id={bank.manifest['benchmark_bundle_id']}")
    print(f"daily_bank_icps={len(bank.items)}")
    print(f"daily_bank_hash={bank.manifest['daily_bank_hash']}")
    print(f"output_dir={out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
