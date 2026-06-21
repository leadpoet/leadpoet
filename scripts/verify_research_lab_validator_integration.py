#!/usr/bin/env python3
"""Verify production-oriented Research Lab validator shadow integration."""

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research_lab.validator_integration import verify_research_lab_validator_integration


def main() -> int:
    summary = verify_research_lab_validator_integration()
    print(
        "Research Lab validator integration verified: "
        f"epoch {summary['epoch']}, "
        f"weight_sum {summary['weight_sum']}, "
        f"hash {summary['weight_vector_hash']}, "
        f"unsafe flags rejected {summary['unsafe_errors']}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
