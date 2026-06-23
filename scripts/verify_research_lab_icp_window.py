#!/usr/bin/env python3
"""Verify deterministic Research Lab 10-day / 50-ICP window selection."""

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gateway.research_lab.bundles import contains_secret_material
from gateway.research_lab.icp_window import select_rolling_icp_window_from_sets
from research_lab.canonical import sha256_json


def main() -> int:
    rows = [_fake_set(20260601 + day) for day in range(10)]
    first = select_rolling_icp_window_from_sets(rows, days=10, icps_per_day=5)
    second = select_rolling_icp_window_from_sets(list(reversed(rows)), days=10, icps_per_day=5)

    errors: list[str] = []
    if first.window_hash != second.window_hash:
        errors.append("selection must be deterministic independent of input row order")
    if len(first.set_ids) != 10:
        errors.append(f"expected 10 sets, got {len(first.set_ids)}")
    if len(first.benchmark_items) != 50:
        errors.append(f"expected 50 selected ICPs, got {len(first.benchmark_items)}")
    if any(not item["icp_hash"].startswith("sha256:") for item in first.benchmark_items):
        errors.append("all selected ICPs must have sha256 hashes")
    if any("icp" in selected for day in first.public_doc["sets"] for selected in day["selected_icps"]):
        errors.append("public window doc must not include hidden ICP plaintext")
    if contains_secret_material(first.public_doc):
        errors.append("public window doc contains forbidden secret/private markers")
    for day in first.public_doc["sets"]:
        signatures = [item["intent_signal_signature"] for item in day["selected_icps"]]
        if len(signatures) != len(set(signatures)):
            errors.append(f"set {day['set_id']} did not prefer unique intent signatures")

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
    for idx in range(6):
        signal = f"Signal {idx}" if idx < 5 else "Signal 0"
        icps.append(
            {
                "icp_id": f"icp_{set_id}_{idx + 1:03d}",
                "prompt": f"Find companies for {set_id} #{idx}",
                "industry": "Software",
                "sub_industry": f"Sub {idx}",
                "geography": "United States",
                "country": "United States",
                "employee_count": "50-200",
                "company_stage": "Series A",
                "product_service": "SaaS platform",
                "intent_signals": [signal],
            }
        )
    return {
        "set_id": set_id,
        "icps": icps,
        "icp_set_hash": sha256_json(icps).split(":", 1)[1],
    }


if __name__ == "__main__":
    raise SystemExit(main())
