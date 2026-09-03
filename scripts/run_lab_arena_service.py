#!/usr/bin/env python3
"""Lab Arena service entrypoint (labarena.md sections 14, 16).

Runs the ``/arena/v1`` API and the once-a-minute ``advance_round`` driver for
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
    return parser


def drive_once(service) -> str:
    """One driver tick: advance the current round if there is one.

    Every failure, including one while looking the round up, is contained and
    reported by exception type only (never a secret or a payload): the next
    tick retries, and the driver thread never dies while the process serves.
    """

    outcome = _advance_current(service)
    # The published round is no longer current, so its king model release runs
    # as its own step; a release failure never stops the next round.
    try:
        release = service.release_pending()
    except Exception as exc:
        return "%s; failed release_pending: %s" % (outcome, type(exc).__name__)
    if release.get("status") in ("ok", "released") and release.get("round_id") and release.get("status") == "ok":
        return "%s; released %s" % (outcome, release["round_id"])
    return outcome


def _advance_current(service) -> str:
    try:
        current = service.current_round()
    except Exception as exc:
        return "failed current_round: %s" % type(exc).__name__
    if current is None:
        # No round is open or running: create the next daily round if the
        # service is configured to, otherwise wait for the operator.
        try:
            ensured = service.ensure_daily_round()
        except Exception as exc:
            return "failed ensure_daily_round: %s" % type(exc).__name__
        if ensured.get("status") == "created":
            return "created %s" % ensured.get("round_id")
        return "idle"
    try:
        service.advance_round(current["round_id"])
    except Exception as exc:
        return "failed advance_round: %s" % type(exc).__name__
    return "advanced"


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

    threading.Thread(target=driver, name="lab-arena-driver", daemon=True).start()
    import uvicorn

    uvicorn.run(app, host=args.host, port=int(args.port), log_level="info")
    stop.set()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
