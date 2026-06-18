#!/usr/bin/env python3
"""Verify Research Lab Phase 1.6 validator-flow contracts."""

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research_lab.validator_flow import verify_research_lab_validator_flow


def main() -> int:
    summary = verify_research_lab_validator_flow()
    print(
        "Research Lab Phase 1.6 validator flow verified: "
        f"attestation {summary['attestation_status']}, "
        f"PCR0 allowed {summary['pcr0_allowed']}, "
        f"open-verifier byte-identical {summary['open_verifier_byte_identical']}, "
        f"{summary['judge_samples']} judge sample, "
        f"{summary['liveness_samples']} liveness sample, "
        f"consensus {summary['consensus_id']}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
