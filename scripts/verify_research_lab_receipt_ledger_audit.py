#!/usr/bin/env python3
"""Verify Research Lab Phase 2.8 receipts-v2 and ledger-audit contracts."""

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research_lab.receipt_ledger_audit import verify_research_lab_receipt_ledger_audit


def main() -> int:
    summary = verify_research_lab_receipt_ledger_audit()
    print(
        "Research Lab Phase 2.8 receipt/ledger audit verified: "
        f"receipt={summary['receipt_ref']}, "
        f"balance_audit={summary['balance_audit_id']} ({summary['balance_state']}), "
        f"cost_audit={summary['cost_audit_id']} ({summary['cost_state']}), "
        f"anchor={summary['anchor_proposal']}, "
        f"market gates={summary['market_dependency_gates']}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
