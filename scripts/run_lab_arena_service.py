#!/usr/bin/env python3
"""Start the Lab Arena service.

Runs the ``/arena/v1`` API and the once-a-minute driver (one ``advance_round`` per active round) for
one Arena host. ``LAB_ARENA_MODE=off`` starts nothing and serves nothing.
Production wiring reads competition dependencies at startup. Reward signing
and epoch cutover load only when a published live round needs activation.
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lab_arena.driver import drive_once  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Leadpoet Lab Arena service")
    parser.add_argument(
        "--environment-file",
        type=Path,
        help="load only LAB_ARENA_* values from the protected gateway env cache",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8792)
    parser.add_argument("--tick-seconds", type=int, default=60)
    parser.add_argument("--check-only", action="store_true", help="run startup checks and exit")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--no-driver", action="store_true", help="serve the API only; another process runs the driver")
    mode.add_argument("--driver-only", action="store_true", help="run the driver ticks only; serve nothing")
    return parser


def load_scoped_environment(path: Path) -> None:
    """Load only Arena-owned values without restoring gateway provider aliases."""

    from gateway.tee.prepare_gateway_envelopes_v2 import load_environment_file

    for name, value in load_environment_file(path).items():
        if name.startswith("LAB_ARENA_"):
            os.environ.setdefault(name, value)


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.environment_file is not None:
        load_scoped_environment(args.environment_file)
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
    if not args.no_driver:
        initial = drive_once(service)
        if "failed" in initial:
            print("initial driver tick", initial, file=sys.stderr)
        elif initial != "idle":
            print("initial driver tick", initial)

    stop = threading.Event()

    def driver() -> None:
        while not stop.wait(max(5, int(args.tick_seconds))):
            outcome = drive_once(service)
            if "failed" in outcome:
                print("driver tick", outcome, file=sys.stderr)

    # The driver is one process's job: several API replicas run with
    # --no-driver and exactly one scheduler process runs with --driver-only.
    if args.driver_only:
        try:
            driver()
        except KeyboardInterrupt:
            stop.set()
        return 0
    driver_thread = None
    if not args.no_driver:
        driver_thread = threading.Thread(
            target=driver, name="lab-arena-driver", daemon=True
        )
        driver_thread.start()
    import uvicorn

    try:
        uvicorn.run(app, host=args.host, port=int(args.port), log_level="info")
    finally:
        stop.set()
        if driver_thread is not None:
            driver_thread.join(timeout=5)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
