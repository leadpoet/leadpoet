#!/usr/bin/env python3
"""Verify deterministic Research Lab 10-day / 60-ICP window selection."""

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gateway.research_lab.bundles import contains_secret_material
from gateway.research_lab.icp_window import intent_signal_signature, select_rolling_icp_window_from_sets
from research_lab.canonical import sha256_json


def main() -> int:
    rows = [_fake_set(20260601 + day) for day in range(10)]
    first = select_rolling_icp_window_from_sets(rows, days=10, icps_per_day=6)
    second = select_rolling_icp_window_from_sets(list(reversed(rows)), days=10, icps_per_day=6)

    errors: list[str] = []
    if first.window_hash != second.window_hash:
        errors.append("selection must be deterministic independent of input row order")
    if len(first.set_ids) != 10:
        errors.append(f"expected 10 sets, got {len(first.set_ids)}")
    if len(first.benchmark_items) != 60:
        errors.append(f"expected 60 selected ICPs, got {len(first.benchmark_items)}")
    if any(not item["icp_hash"].startswith("sha256:") for item in first.benchmark_items):
        errors.append("all selected ICPs must have sha256 hashes")
    if any("icp" in selected for day in first.public_doc["sets"] for selected in day["selected_icps"]):
        errors.append("public window doc must not include hidden ICP plaintext")
    if contains_secret_material(first.public_doc):
        errors.append("public window doc contains forbidden secret/private markers")
    if first.public_doc["selection_policy"] != "diverse_intent_industry_stable_hash:v2":
        errors.append("selection policy did not use v2 diverse selector")
    for day in first.public_doc["sets"]:
        if len(day["selected_icps"]) != 6:
            errors.append(f"set {day['set_id']} expected 6 ICPs, got {len(day['selected_icps'])}")
        signatures = [item["intent_signal_signature"] for item in day["selected_icps"]]
        if len(signatures) != len(set(signatures)):
            errors.append(f"set {day['set_id']} did not prefer unique intent signatures")
    if len({item["icp_ref"] for item in first.benchmark_items}) != 60:
        errors.append("selected ICP refs must be unique")

    old_score = _old_selection_diversity(rows[0]["icps"], 6)
    new_score = _selection_diversity(first.benchmark_items[:6])
    if new_score <= old_score:
        errors.append(f"v2 selector did not improve first-day diversity: old={old_score}, new={new_score}")

    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    print(
        "Research Lab ICP window verified: "
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


def _old_selection_diversity(icps: list[dict[str, object]], icps_per_day: int) -> int:
    ranked = sorted(
        icps,
        key=lambda icp: (
            intent_signal_signature(icp),
            sha256_json(
                {
                    "icp_id": icp.get("icp_id"),
                    "industry": icp.get("industry"),
                    "sub_industry": icp.get("sub_industry"),
                    "intent_signals": icp.get("intent_signals"),
                    "prompt": icp.get("prompt"),
                }
            ),
        ),
    )
    return _icp_diversity(ranked[:icps_per_day])


def _selection_diversity(items: list[dict[str, object]] | tuple[dict[str, object], ...]) -> int:
    return _icp_diversity([dict(item["icp"]) for item in items])


def _icp_diversity(icps: list[dict[str, object]]) -> int:
    industries = {str(icp.get("industry") or "") for icp in icps}
    sub_industries = {str(icp.get("sub_industry") or "") for icp in icps}
    signatures = {intent_signal_signature(icp) for icp in icps}
    return len(industries) + len(sub_industries) + len(signatures)


if __name__ == "__main__":
    raise SystemExit(main())
