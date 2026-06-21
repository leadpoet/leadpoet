#!/usr/bin/env python3
"""Run Research Lab economics golden vectors in the open verifier."""

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from leadpoet_verifier.golden_vectors import load_golden_vectors, run_golden_vectors


def main() -> int:
    fixture = load_golden_vectors()
    case_count = len(fixture.get("research_lab_cases", []))
    if case_count == 0:
        print("open verifier Research Lab golden-vector verification failed: no research_lab_cases found")
        return 1

    errors = run_golden_vectors(pcr0_allowlist_path=str(ROOT / "pcr0_allowlist.json"))
    if errors:
        print("open verifier Research Lab golden-vector verification failed:")
        for error in errors:
            if error.startswith("research_lab "):
                print(f"- {error}")
        if not any(error.startswith("research_lab ") for error in errors):
            print("- shared golden-vector dependency failed; run scripts/verify_open_verifier.py for full details")
        return 1

    print(f"open verifier Research Lab golden-vector verification passed ({case_count} cases)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
