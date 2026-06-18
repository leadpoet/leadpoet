#!/usr/bin/env python3
"""Verify Phase 4.8 exit-gate contracts."""

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research_lab.scale_exit_gate import verify_scale_exit_gate


def main() -> int:
    summary = verify_scale_exit_gate()
    print(
        "Research Lab Phase 4.8 exit gate verified: "
        f"exit_gate={summary['exit_gate_id']}, "
        f"local_code_complete={summary['local_code_complete']}, "
        f"production_scale_ready={summary['production_scale_ready']}, "
        f"missing_checks={','.join(summary['missing_checks'])}, "
        f"measured_control_validates={summary['measured_control_validates']}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
