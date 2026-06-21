#!/usr/bin/env python3
"""Verify Research Lab controlled production shadow mode."""

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research_lab.production_shadow import verify_controlled_production_shadow


def main() -> int:
    summary = verify_controlled_production_shadow()
    print(
        "Research Lab production shadow verified: "
        f"bundle {summary['bundle_id']}, "
        f"report {summary['report_id']}, "
        f"epoch {summary['epoch']}, "
        f"{summary['awarded_reimbursement_count']} shadow reimbursements, "
        f"scheduled {summary['scheduled_reimbursement_microusd_by_epoch']}, "
        f"{summary['changed_uid_count']} live/shadow changed UIDs, "
        f"max diff {summary['max_abs_delta_u16']} u16, "
        f"verifier divergence status {summary['verifier_divergence_status']}, "
        f"read_only={summary['read_only']}, "
        f"window_epochs={summary['window_epoch_count']}, "
        f"window_ready={summary['window_activation_ready']}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
