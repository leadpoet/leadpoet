#!/usr/bin/env python3
"""Verify Research Lab Phase 3.3 fine-tune dataset contracts."""

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research_lab.fine_tune_dataset import verify_research_lab_fine_tune_dataset


def main() -> int:
    summary = verify_research_lab_fine_tune_dataset()
    print(
        "Research Lab Phase 3.3 fine-tune dataset verified: "
        f"corpus={summary['corpus_manifest_id']}, "
        f"dataset={summary['dataset_id']}, "
        f"dataset_ready={summary['dataset_ready_claimed']}, "
        f"training_run={summary['training_run_id']}, "
        f"training_started={summary['training_started']}, "
        f"success_claimed={summary['success_claimed']}, "
        f"builder={summary['builder_dataset_id']}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
