#!/usr/bin/env python3
"""Verify Research Lab Phase 1.8 recruiting/admin contracts."""

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research_lab.recruiting_admin import verify_research_lab_recruiting_admin


def main() -> int:
    summary = verify_research_lab_recruiting_admin()
    print(
        "Research Lab Phase 1.8 recruiting/admin verified: "
        f"{summary['template_briefs']} template brief, "
        f"{summary['candidates']} candidate, "
        f"cohort target {summary['cohort_target_min']}-{summary['cohort_target_max']}, "
        f"{summary['onboarding_checklists']} onboarding checklist, "
        f"{summary['map_cells']} map cell inputs."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
