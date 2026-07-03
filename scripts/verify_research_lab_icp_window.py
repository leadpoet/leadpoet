#!/usr/bin/env python3
"""Verify deterministic Research Lab hybrid fresh/retained ICP window selection."""

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gateway.research_lab.bundles import contains_secret_material
from gateway.research_lab.icp_window import (
    SELECTION_POLICY_HYBRID,
    WINDOW_MODE_HYBRID_FRESH_RETAINED,
    intent_signal_signature,
    reconstruct_icp_window_from_doc,
    select_rolling_icp_window_from_sets,
)
from research_lab.canonical import sha256_json


def main() -> int:
    rows = [_fake_set(20260601 + day) for day in range(10)]
    first = select_rolling_icp_window_from_sets(rows)
    second = select_rolling_icp_window_from_sets(list(reversed(rows)))
    next_day = select_rolling_icp_window_from_sets([_fake_set(20260602 + day) for day in range(10)])
    reconstructed = reconstruct_icp_window_from_doc(rows, first.public_doc)

    errors: list[str] = []
    if first.window_hash != second.window_hash:
        errors.append("selection must be deterministic independent of input row order")
    if reconstructed.window_hash != first.window_hash:
        errors.append("stored hybrid window doc must reconstruct to the same hash")
    if len(first.benchmark_items) != 20:
        errors.append(f"expected 20 selected ICPs, got {len(first.benchmark_items)}")
    fresh = [item for item in first.benchmark_items if item.get("cohort") == "fresh"]
    retained = [item for item in first.benchmark_items if item.get("cohort") == "retained"]
    if len(fresh) != 10:
        errors.append(f"expected 10 fresh ICPs, got {len(fresh)}")
    if len(retained) != 10:
        errors.append(f"expected 10 retained ICPs, got {len(retained)}")
    retained_set_count = len({int(item["set_id"]) for item in retained})
    if retained_set_count < 5:
        errors.append(f"expected retained ICPs to spread across recent prior sets, got {retained_set_count}")
    overlap = len(set(first.item_refs) & set(next_day.item_refs))
    if len(next_day.item_refs) - overlap < 10:
        errors.append(f"expected at least 10 new day-over-day ICPs, got {len(next_day.item_refs) - overlap}")
    if any(not item["icp_hash"].startswith("sha256:") for item in first.benchmark_items):
        errors.append("all selected ICPs must have sha256 hashes")
    if any("icp" in selected for day in first.public_doc["sets"] for selected in day["selected_icps"]):
        errors.append("public window doc must not include hidden ICP plaintext")
    if contains_secret_material(first.public_doc):
        errors.append("public window doc contains forbidden secret/private markers")
    if first.public_doc["schema_version"] != "1.1":
        errors.append("hybrid public window doc must be schema_version 1.1")
    if first.public_doc["window_mode"] != WINDOW_MODE_HYBRID_FRESH_RETAINED:
        errors.append("hybrid public window doc must record window_mode")
    if first.public_doc["selection_policy"] != SELECTION_POLICY_HYBRID:
        errors.append("selection policy did not use hybrid fresh/retained selector")
    if first.public_doc.get("fresh_icp_count") != 10 or first.public_doc.get("retained_icp_count") != 10:
        errors.append("hybrid public window doc must record 10 fresh and 10 retained ICPs")
    for day in first.public_doc["sets"]:
        signatures = [item["intent_signal_signature"] for item in day["selected_icps"] if item.get("cohort") == "fresh"]
        if signatures and len(signatures) != len(set(signatures)):
            errors.append(f"set {day['set_id']} did not prefer unique fresh intent signatures")
    if len({item["icp_ref"] for item in first.benchmark_items}) != 20:
        errors.append("selected ICP refs must be unique")

    fresh_score = _selection_diversity(fresh)
    retained_score = _selection_diversity(retained)
    if fresh_score <= 0 or retained_score <= 0:
        errors.append("hybrid selector did not produce diversity-checkable fresh and retained cohorts")

    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    print(
        "Research Lab ICP window verified: "
        f"mode={first.public_doc['window_mode']} fresh={len(fresh)} retained={len(retained)} "
        f"sets={len(first.set_ids)}, icps={len(first.benchmark_items)}, hash={first.window_hash}"
    )
    return 0


def _fake_set(set_id: int) -> dict[str, object]:
    icps = []
    industries = [
        "Software",
        "Healthcare",
        "Manufacturing",
        "Financial Services",
        "Logistics",
        "Cybersecurity",
    ]
    for idx in range(12):
        diverse = idx >= 6
        industry = industries[idx - 6] if diverse else "Software"
        signal_prefix = "zz diverse signal" if diverse else "aa narrow signal"
        icps.append(
            {
                "icp_id": f"icp_{set_id}_{idx + 1:03d}",
                "prompt": f"Find companies for {set_id} #{idx}",
                "industry": industry,
                "sub_industry": f"{industry} Sub {idx}",
                "geography": "United States",
                "country": "United States",
                "employee_count": "50-200" if idx % 2 == 0 else "201-1000",
                "company_stage": "Series A",
                "product_service": f"{industry} platform",
                "intent_signals": [f"{signal_prefix} {idx}"],
            }
        )
    return {
        "set_id": set_id,
        "icps": icps,
        "icp_set_hash": sha256_json(icps).split(":", 1)[1],
    }


def _selection_diversity(items: list[dict[str, object]] | tuple[dict[str, object], ...]) -> int:
    return _icp_diversity([dict(item["icp"]) for item in items])


def _icp_diversity(icps: list[dict[str, object]]) -> int:
    industries = {str(icp.get("industry") or "") for icp in icps}
    sub_industries = {str(icp.get("sub_industry") or "") for icp in icps}
    signatures = {intent_signal_signature(icp) for icp in icps}
    return len(industries) + len(sub_industries) + len(signatures)


if __name__ == "__main__":
    raise SystemExit(main())
