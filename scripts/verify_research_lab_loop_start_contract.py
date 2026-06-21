#!/usr/bin/env python3
"""Verify Research Lab local/staged loop-start payment and key contract."""

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research_lab.loop_start_contract import verify_research_lab_loop_start_contract


def main() -> int:
    summary = verify_research_lab_loop_start_contract()
    routes = summary["provider_routes"]
    print(
        "Research Lab loop-start contract verified: "
        f"${summary['loop_start_fee_usd']:.2f} fee, "
        f"decision {summary['decision_status']}, "
        f"payment {summary['valid_payment_ref']}, "
        f"routes openrouter={routes['openrouter']} "
        f"exa={routes['exa']} scrapingdog={routes['scrapingdog']}, "
        f"{summary['payment_rejection_cases']} payment rejection cases, "
        f"{summary['key_rejection_cases']} key rejection cases, "
        f"credit {summary['preserved_credit_status']}, "
        f"gateway wrapper {summary['gateway_wrapper_status']}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
