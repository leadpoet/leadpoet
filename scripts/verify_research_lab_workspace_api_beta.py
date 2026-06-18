#!/usr/bin/env python3
"""Verify Research Lab Phase 3.6 Workspace API beta contracts."""

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research_lab.workspace_api_beta import verify_research_lab_workspace_api_beta


def main() -> int:
    summary = verify_research_lab_workspace_api_beta()
    print(
        "Research Lab Phase 3.6 Workspace API beta verified: "
        f"methods={summary['method_contracts']}, "
        f"allowlist={summary['allowlist_id']}, "
        f"miner={summary['miner_ref']}, "
        f"local_beta_enablement_claimed={summary['local_beta_enablement_claimed']}, "
        f"local_workspace_api_calls_enabled={summary['local_workspace_api_calls_enabled']}, "
        f"measured_beta_claim_validates={summary['measured_beta_claim_validates']}, "
        f"entropy_budget_bits={summary['entropy_budget_bits']}, "
        f"max_cohort_size={summary['max_cohort_size']}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
