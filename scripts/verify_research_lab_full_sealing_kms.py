#!/usr/bin/env python3
"""Verify Research Lab Phase 2.6 full-sealing/KMS contracts."""

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research_lab.full_sealing_kms import verify_research_lab_full_sealing_kms


def main() -> int:
    summary = verify_research_lab_full_sealing_kms()
    print(
        "Research Lab Phase 2.6 full sealing/KMS verified: "
        f"gate={summary['gate_id']} ({summary['gate_status']}), "
        f"kms_request={summary['kms_request_id']}, "
        f"kms_decision={summary['kms_decision_id']}:{summary['kms_decision']}, "
        f"enclave_run={summary['enclave_run_id']}, "
        f"denial_reasons={summary['denial_reasons']}, "
        f"market gates={summary['market_dependency_gates']}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
