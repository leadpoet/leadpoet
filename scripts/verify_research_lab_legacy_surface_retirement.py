#!/usr/bin/env python3
"""Verify Phase 4.2 legacy surface retirement contracts."""

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research_lab.legacy_surface_retirement import verify_legacy_surface_retirement


def main() -> int:
    summary = verify_legacy_surface_retirement()
    print(
        "Research Lab Phase 4.2 legacy surface retirement verified: "
        f"audit={summary['current_audit_id']}, "
        f"routes={summary['legacy_route_refs']}, "
        f"tables={summary['legacy_table_refs']}, "
        f"payments={summary['legacy_payment_refs']}, "
        f"submission_credits={summary['submission_credit_refs']}, "
        f"artifact_competition={summary['artifact_competition_refs']}, "
        f"legacy_surface_retired={summary['legacy_surface_retired']}, "
        f"blockers={summary['blockers']}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
