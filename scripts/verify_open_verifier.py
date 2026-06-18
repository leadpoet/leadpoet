#!/usr/bin/env python3
"""Run Leadpoet Research Lab open-verifier golden vectors."""

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from leadpoet_verifier.golden_vectors import run_golden_vectors


def main() -> int:
    errors = run_golden_vectors(pcr0_allowlist_path=str(ROOT / "pcr0_allowlist.json"))
    if errors:
        print("open verifier golden-vector verification failed:")
        for error in errors:
            print(f"- {error}")
        return 1
    print("open verifier golden-vector verification passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
