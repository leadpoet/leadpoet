#!/usr/bin/env python3
"""Verify Phase 4.4 confidential-GPU training contracts."""

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research_lab.confidential_gpu_training import verify_research_lab_confidential_gpu_training


def main() -> int:
    summary = verify_research_lab_confidential_gpu_training()
    print(
        "Research Lab Phase 4.4 confidential GPU training verified: "
        f"data_policy={summary['data_policy_id']}, "
        f"policy_approved={summary['data_policy_approved']}, "
        f"training={summary['training_run_id']}, "
        f"production_training_valid={summary['production_training_valid']}, "
        f"weight_artifact={summary['weight_artifact_id']}, "
        f"private_artifact_ready={summary['private_artifact_ready']}, "
        f"public_release_enabled={summary['public_release_enabled']}, "
        f"ready_control_validates={summary['ready_control_validates']}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
