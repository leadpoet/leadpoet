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
            current = service.current_round()
            if current is not None:
                try:
                    service.advance_round(current["round_id"])
                except Exception as exc:  # the next tick retries; the failure is logged without secrets
                    print("advance_round failed:", type(exc).__name__, file=sys.stderr)
            stop.wait(max(5, int(args.tick_seconds)))

    threading.Thread(target=driver, name="lab-arena-driver", daemon=True).start()
    import uvicorn

    uvicorn.run(app, host=args.host, port=int(args.port), log_level="info")
    stop.set()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
