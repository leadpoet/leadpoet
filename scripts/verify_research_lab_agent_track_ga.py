#!/usr/bin/env python3
"""Verify Phase 4.6 Agent-track GA gate contracts."""

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research_lab.agent_track_ga import verify_research_lab_agent_track_ga


def main() -> int:
    summary = verify_research_lab_agent_track_ga()
    print(
        "Research Lab Phase 4.6 Agent-track GA verified: "
        f"criteria={summary['criteria_id']}, "
        f"criteria_approved={summary['criteria_approved']}, "
        f"candidate={summary['candidate_id']}, "
        f"candidate_eligible={summary['candidate_eligible']}, "
        f"ga={summary['ga_readiness_id']}, "
        f"ga_claimed={summary['ga_claimed']}, "
        f"public_agent_track_enabled={summary['public_agent_track_enabled']}, "
        f"ready_control_validates={summary['ready_control_validates']}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
