#!/usr/bin/env python3
"""Prove the legacy combined Arena worker has no paid review to abandon."""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def require_idle(service, *, now: datetime) -> None:
    # The old combined process has a five-second daemon-thread join. It can
    # only be retired when review cannot start during the handoff window.
    rounds = service.active_rounds()
    for active in rounds:
        round_row = service._round(active["round_id"])
        open_before_cutoff = round_row["status"] == "open"
        # The legacy API can have a long in-flight request. Leave more than
        # the controller's 6,600-second graceful drain plus ten minutes.
        if open_before_cutoff and service._submission_replacement_cutoff(round_row) <= now + timedelta(seconds=7200):
            raise ValueError("legacy Arena code review cutoff is too near")
        for status in ("accepted", "frozen"):
            for row in service._store.list_submissions(round_row["round_id"], status=status):
                # Finish retains the claim for idempotency and audit. Only a
                # final decision with an explicitly cleared lease proves that
                # retained claim no longer represents paid work.
                completed = (row.get("code_review_status") in ("passed", "rejected")
                             and "code_review_expires_at" in row
                             and row["code_review_expires_at"] is None)
                if row.get("code_review_claim") is not None and not completed:
                    raise ValueError("legacy Arena paid review is claimed")
                if row.get("code_review_expires_at") is not None:
                    raise ValueError("legacy Arena paid review has a lease")
                if (not open_before_cutoff
                        and not service._is_daily_baseline(row, round_row)
                        and row.get("code_review_status") not in ("passed", "rejected")):
                    raise ValueError("legacy Arena paid review can still start")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--environment-file", type=Path, required=True)
    args = parser.parse_args()
    from scripts.run_lab_arena_service import load_scoped_environment
    from lab_arena.wiring import build_service_from_environment
    import os

    load_scoped_environment(args.environment_file)
    mode = os.environ.get("LAB_ARENA_MODE", "off").strip().lower()
    if mode not in ("shadow", "live"):
        raise ValueError("Arena mode is not active for first cutover")
    service, _app = build_service_from_environment(mode)
    service.startup_checks()
    require_idle(service, now=datetime.now(timezone.utc))
    print("legacy Arena paid-review handoff idle")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
