#!/usr/bin/env python3
"""Lab Arena service entrypoint (labarena.md sections 14, 16).

Runs the ``/arena/v1`` API and the once-a-minute driver (one ``advance_round`` per active round) for
one Arena host. ``LAB_ARENA_MODE=off`` starts nothing and serves nothing.
Production wiring (PostgREST JWT, KMS keys, provider credentials, S3 bucket,
chain endpoint) is read from the environment; secrets are never printed.
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Leadpoet Lab Arena service")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8791)
    parser.add_argument("--tick-seconds", type=int, default=60)
    parser.add_argument("--check-only", action="store_true", help="run startup checks and exit")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--no-driver", action="store_true", help="serve the API only; another process runs the driver")
    mode.add_argument("--driver-only", action="store_true", help="run the driver ticks only; serve nothing")
    return parser


def drive_once(service) -> str:
    """One driver tick: advance every active round, then open the next daily round.

    Rounds overlap: the day's round runs its benchmark while the next round is
    open for submissions, so a tick advances each round that is not published
    or cancelled, oldest first. Every failure, including one while listing the
    rounds, is contained and reported by exception type only (never a secret
    or a payload): the next tick retries, and the driver thread never dies
    while the process serves.
    """

    outcome = _advance_active(service)
    # A published round is no longer active, so its king model release runs
    # as its own step; a release failure never stops the next round.
    try:
        release = service.release_pending()
    except Exception as exc:
        return "%s; failed release_pending: %s" % (outcome, type(exc).__name__)
    if release.get("status") == "ok" and release.get("round_id"):
        return "%s; released %s" % (outcome, release["round_id"])
    return outcome


def _advance_active(service) -> str:
    try:
        active = list(service.active_rounds())
    except Exception as exc:
        return "failed active_rounds: %s" % type(exc).__name__
    outcomes = []
    for row in active:
        try:
            service.advance_round(row["round_id"])
        except Exception as exc:
            outcomes.append("failed advance_round %s: %s" % (row["round_id"], type(exc).__name__))
        else:
            outcomes.append("advanced %s" % row["round_id"])
    if not any(row.get("status") == "open" for row in active):
        # No round is open for submissions: create the next daily round if the
        # service is configured to, otherwise wait for the operator.
        try:
            ensured = service.ensure_daily_round()
        except Exception as exc:
            outcomes.append("failed ensure_daily_round: %s" % type(exc).__name__)
        else:
            if ensured.get("status") == "created":
                outcomes.append("created %s" % ensured.get("round_id"))
    return "; ".join(outcomes) if outcomes else "idle"


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    mode = os.environ.get("LAB_ARENA_MODE", "off").strip().lower()
    if mode == "off":
        print("LAB_ARENA_MODE=off: nothing starts and nothing is served")
        return 0
    from lab_arena.wiring import build_service_from_environment  # lazy: production dependencies

    service, app = build_service_from_environment(mode)
    checks = service.startup_checks()
    print("lab arena service identity", {k: v for k, v in checks.items() if k != "database_identity"}, "role", checks["database_identity"].get("current_user"))
    if args.check_only:
        return 0
    stop = threading.Event()

    def driver() -> None:
        while not stop.is_set():
            outcome = drive_once(service)
            if "failed" in outcome:
                print("driver tick", outcome, file=sys.stderr)
            stop.wait(max(5, int(args.tick_seconds)))

    # The driver is one process's job: several API replicas run with
    # --no-driver and exactly one scheduler process runs with --driver-only.
    if args.driver_only:
        try:
            driver()
        except KeyboardInterrupt:
            stop.set()
        return 0
    if not args.no_driver:
        threading.Thread(target=driver, name="lab-arena-driver", daemon=True).start()
    import uvicorn

    uvicorn.run(app, host=args.host, port=int(args.port), log_level="info")
    stop.set()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
