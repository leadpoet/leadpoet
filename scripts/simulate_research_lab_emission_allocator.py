#!/usr/bin/env python3
"""Simulate Research Lab per-epoch emission allocation scenarios."""

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from leadpoet_verifier.economics import allocate_research_lab_epoch


POLICY = {
    "research_lab_emission_percent": 10.0,
    "fulfillment_emission_percent": 80.5,
    "fulfillment_leaderboard_emission_percent": 9.5,
    "reward_epochs": 20,
    "usd_per_0_1_percent_epoch": 0.162,
    "reimbursement_allow_overpay_without_champions": True,
    "reimbursement_max_cost_multiplier_with_champions": 1.0,
    "champion_min_alpha_percent": 2.0,
    "champion_extra_alpha_percent_per_point": 0.1,
    "champion_max_alpha_percent": 5.0,
    "champion_placeholder_alpha_percent": 0.0001,
    "champion_threshold_points": 2.0,
}


def main() -> int:
    scenarios = {
        "low": {
            "reimbursements": [_reimbursement(1, 10, island_weight=2.0)],
            "champions": [],
        },
        "medium": {
            "reimbursements": [_reimbursement(uid, 20 if uid <= 5 else 10) for uid in range(1, 11)],
            "champions": [],
        },
        "high": {
            "reimbursements": [_reimbursement(uid, 20 if uid <= 5 else 10) for uid in range(1, 13)],
            "champions": [_champion(101, 105, 5.0), _champion(102, 108, 8.0)],
        },
    }
    for name, scenario in scenarios.items():
        print(f"\n{name.upper()} PARTICIPATION")
        for epoch in range(100, 125):
            allocation = allocate_research_lab_epoch(
                epoch,
                POLICY,
                scenario["reimbursements"],
                scenario["champions"],
            )
            print(
                f"epoch={epoch} "
                f"reimburse={allocation['reimbursement_alpha_percent']:.6f}% "
                f"champions={allocation['champion_alpha_percent']:.6f}% "
                f"queued={allocation['queued_champion_alpha_percent']:.6f}% "
                f"unused={allocation['unallocated_percent']:.6f}%"
            )
    return 0


def _reimbursement(uid: int, spend_usd: float, *, island_weight: float = 1.0) -> dict[str, object]:
    return {
        "uid": uid,
        "miner_hotkey": f"5Freimburse{uid}",
        "source_id": f"reimbursement:{uid}",
        "island": "generalist",
        "start_epoch": 100,
        "epoch_count": 20,
        "actual_openrouter_cost_usd": spend_usd,
        "island_weight": island_weight,
    }


def _champion(uid: int, start_epoch: int, points: float) -> dict[str, object]:
    return {
        "uid": uid,
        "miner_hotkey": f"5Fchampion{uid}",
        "source_id": f"champion:{uid}",
        "island": "generalist",
        "start_epoch": start_epoch,
        "epoch_count": 20,
        "improvement_points": points,
    }


if __name__ == "__main__":
    raise SystemExit(main())
