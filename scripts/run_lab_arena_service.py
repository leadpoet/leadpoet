#!/usr/bin/env python3
"""Start the Lab Arena service.

Runs the ``/arena/v1`` API and the once-a-minute driver (one ``advance_round`` per active round) for
one Arena host. ``LAB_ARENA_MODE=off`` starts nothing and serves nothing.
Production wiring reads competition dependencies at startup. Reward signing
and epoch cutover load only when a published live round needs activation.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import signal
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lab_arena import telemetry  # noqa: E402
from lab_arena.driver import drive_once, next_tick_delay  # noqa: E402
from lab_arena.runtime_version import SOURCE_METADATA  # noqa: E402
from leadpoet_observability.sentry_bootstrap import INGEST_ENVIRONMENT_KEYS  # noqa: E402


def _install_arena_telemetry(app) -> None:
    """Attach request, stage, provider, run, and gate spans.

    A complete no-op when telemetry is disabled.
    """

    try:
        from gateway.observability.otel_bootstrap import configure_arena_otel

        telemetry.install_recorder(configure_arena_otel(app))
    except Exception as exc:
        print("arena telemetry unavailable", type(exc).__name__, file=sys.stderr)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Leadpoet Lab Arena service")
    parser.add_argument(
        "--environment-file",
        type=Path,
        help="load Arena and Sentry ingest values from the protected gateway env cache",
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
    """Load Arena and explicit Sentry ingest values, never provider aliases."""

    try:
        raw = Path(path).read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(
            "gateway source environment is unavailable"
        ) from exc
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = None
    if parsed is not None:
        if not isinstance(parsed, dict):
            raise ValueError(
                "gateway source environment JSON must be an object"
            )
        scoped = {
            str(name): str(value)
            for name, value in parsed.items()
            if str(name).startswith("LAB_ARENA_") or str(name) in INGEST_ENVIRONMENT_KEYS
        }
    else:
        scoped = {}
        for raw_line in raw.replace("\x00", "\n").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export ") :].strip()
            name, separator, raw_value = line.partition("=")
            name = name.strip()
            if not separator or not (
                name.startswith("LAB_ARENA_") or name in INGEST_ENVIRONMENT_KEYS
            ):
                continue
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
                raise ValueError(
                    "gateway Arena environment key is malformed"
                )
            try:
                parts = shlex.split("VALUE=" + raw_value, comments=True, posix=True)
            except ValueError as exc:
                raise ValueError(
                    "gateway Arena environment value is malformed"
                ) from exc
            if len(parts) != 1 or not parts[0].startswith("VALUE="):
                raise ValueError(
                    "gateway Arena environment value is malformed"
                )
            value = parts[0].split("=", 1)[1]
            if name in scoped and scoped[name] != value:
                raise ValueError(
                    "gateway Arena environment key is duplicated"
                )
            scoped[name] = value
    for name, value in scoped.items():
        os.environ.setdefault(name, value)


OTEL_ENVIRONMENT_KEYS = (
    "GATEWAY_OTEL_ENABLED",
    "GATEWAY_OTEL_ENDPOINT",
    "GATEWAY_OTEL_TOKEN",
)


def load_otel_environment(path: Path) -> None:
    """Load ONLY the three telemetry values from the protected gateway env.

    ``load_scoped_environment`` accepts only Arena and explicit Sentry ingest
    settings so gateway provider credentials cannot be resurrected in this
    process. OTel is the other explicit telemetry exception:
    endpoint plus ingest token are a write-only pair that can do nothing but
    append spans, and reusing them is what lets Arena telemetry turn on with
    no new secret and no new host configuration. The keys are enumerated by
    the dedicated reader, which never executes the file.
    """

    try:
        from gateway.observability.read_gateway_otel_env import parse_env_file

        values = parse_env_file(Path(path))
    except Exception as exc:
        # Telemetry configuration can never stop the Arena service starting.
        print("arena otel environment unavailable", type(exc).__name__, file=sys.stderr)
        return
    for name in OTEL_ENVIRONMENT_KEYS:
        value = values.get(name, "")
        if value:
            os.environ.setdefault(name, value)


def _record_service_lifecycle(event: str, outcome: str, *, physical_role: str) -> None:
    """Report the actual Arena process transition through existing telemetry."""
    source_sha = SOURCE_METADATA.get("validator_source_commit", "")
    if not re.fullmatch(r"[0-9a-f]{40}", source_sha):
        source_sha = "unknown"
    invocation_id = os.getenv("GATEWAY_RESTART_INVOCATION_ID", "")
    if not re.fullmatch(r"gateway-[0-9]{1,20}-[0-9]{1,20}", invocation_id):
        invocation_id = "unknown"
    try:
        print(
            "ARENA_SERVICE_LIFECYCLE "
            f"event={event} outcome={outcome} source_sha={source_sha} "
            f"physical_role={physical_role} invocation_id={invocation_id}",
            flush=True,
        )
    except BaseException:
        pass
    try:
        telemetry.record(f"service_{event}", outcome, source_sha=source_sha)
    except BaseException:
        pass
    try:
        from leadpoet_observability import record_stage

        record_stage(
            component="arena-service",
            stage=f"service_{event}",
            status="ready" if outcome == "ok" and event == "startup" else (
                "completed" if outcome == "ok" else "failed"
            ),
            candidate_sha=source_sha,
            restart_invocation_id=invocation_id,
            physical_role=physical_role,
        )
    except BaseException:
        pass
    if event == "shutdown":
        telemetry.flush()


def restart_guard_withholds_review(guard: object) -> bool:
    """Report whether a restart guard currently withholds paid review work.

    The guard carries its own lease. An owner that never reaches its release
    step leaves the commitment behind, so holding on presence alone stops
    review work for good; honour the expiry once the state reports it.

    ``guard_active`` arrives with the hand-applied migration that teaches the
    read about the lease, so it is accepted as optional and the stricter
    presence test stands in until that migration lands. Either deploy order is
    therefore safe, and an absent or malformed field never releases work.
    """

    if (
        not isinstance(guard, Mapping)
        or guard.get("schema_version") != "leadpoet.lab_arena.review_guard_state.v1"
        or type(guard.get("guard_present")) is not bool
        or ("guard_active" in guard and type(guard.get("guard_active")) is not bool)
    ):
        raise ValueError("Arena restart guard state is invalid")
    if "guard_active" in guard:
        return guard["guard_active"]
    return guard["guard_present"]


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    physical_role = (
        "arena-driver" if args.driver_only else
        f"arena-api-{args.port}" if args.no_driver and args.port in (8792, 8794) else
        "arena-api" if args.no_driver else "arena-combined"
    )
    if args.environment_file is not None:
        load_scoped_environment(args.environment_file)
        load_otel_environment(args.environment_file)
    mode = os.environ.get("LAB_ARENA_MODE", "off").strip().lower()
    if mode == "off":
        print("LAB_ARENA_MODE=off: nothing starts and nothing is served")
        return 0
    try:
        from leadpoet_observability import init_sentry, configure_sentry_context

        init_sentry(component="arena-service")
        configure_sentry_context(
            component="arena-service", physical_role=physical_role,
            runtime_sha=SOURCE_METADATA.get("validator_source_commit", ""),
            restart_invocation_id=os.getenv("GATEWAY_RESTART_INVOCATION_ID", ""),
        )
    except Exception as exc:
        print("arena sentry unavailable", type(exc).__name__, file=sys.stderr)
    try:
        from lab_arena.wiring import build_service_from_environment  # lazy: production dependencies

        service, app = build_service_from_environment(mode)
        _install_arena_telemetry(app)
        checks = service.startup_checks()
    except BaseException:
        if "app" not in locals():
            _install_arena_telemetry(None)
        _record_service_lifecycle(
            "preflight" if args.check_only else "startup", "failed",
            physical_role=physical_role,
        )
        raise
    print("lab arena service identity", {k: v for k, v in checks.items() if k != "database_identity"}, "role", checks["database_identity"].get("current_user"))
    if args.check_only:
        return 0
    _record_service_lifecycle("startup", "ok", physical_role=physical_role)
    stop = threading.Event()
    if args.driver_only:
        # The scheduler owns paid review work. A controller TERM stops new
        # cycles but lets the current review and driver call finish.
        signal.signal(signal.SIGTERM, lambda _signum, _frame: stop.set())

    def review_allowed() -> bool:
        if stop.is_set():
            return False
        with telemetry.stage("review_guard"):
            guard = service._store._transport.rpc("lab_arena_review_guard_state_v1", {})
            return not stop.is_set() and not restart_guard_withholds_review(guard)

    def safe_review_allowed() -> bool:
        try:
            return review_allowed()
        except Exception:
            # A guard read failure must never reserve new paid review work.
            return False

    def review_submissions() -> None:
        while not stop.is_set():
            try:
                if not review_allowed():
                    stop.wait(5)
                    continue
                with telemetry.stage("review_submissions"):
                    service.review_pending_submissions(
                        should_continue=safe_review_allowed,
                    )
            except Exception as exc:
                # Never emit source, model responses, or credential errors.
                print("code review worker unavailable", type(exc).__name__, file=sys.stderr)
            stop.wait(5)

    review_thread = None
    if not args.no_driver:
        review_thread = threading.Thread(
            target=review_submissions, name="lab-arena-code-review",
            daemon=not args.driver_only,
        )
        review_thread.start()

    def driver() -> None:
        # A full scoring/publication cycle can take longer than API startup.
        # Run the initial cycle in the existing worker, just like later ticks.
        started = datetime.now(timezone.utc)
        if stop.is_set():
            return
        initial = drive_once(service)
        if "failed" in initial:
            print("initial driver tick", initial, file=sys.stderr)
        elif initial != "idle":
            print("initial driver tick", initial)
        while True:
            delay = next_tick_delay(
                args.tick_seconds, service._config.defaults.daily_cutoff_hour_utc,
                started=started, now=datetime.now(timezone.utc),
            )
            if stop.wait(delay):
                break
            started = datetime.now(timezone.utc)
            outcome = drive_once(service)
            if "failed" in outcome:
                print("driver tick", outcome, file=sys.stderr)

    # The driver is one process's job: several API replicas run with
    # --no-driver and exactly one scheduler process runs with --driver-only.
    if args.driver_only:
        shutdown_outcome = "ok"
        try:
            driver()
        except KeyboardInterrupt:
            stop.set()
        except BaseException:
            shutdown_outcome = "failed"
            raise
        finally:
            stop.set()
            if review_thread is not None:
                review_thread.join()
            _record_service_lifecycle("shutdown", shutdown_outcome, physical_role=physical_role)
        return 0
    driver_thread = None
    if not args.no_driver:
        driver_thread = threading.Thread(
            target=driver, name="lab-arena-driver", daemon=True
        )
        driver_thread.start()
    import uvicorn

    shutdown_recorded = False

    def record_shutdown(outcome: str) -> None:
        nonlocal shutdown_recorded
        if not shutdown_recorded:
            shutdown_recorded = True
            _record_service_lifecycle("shutdown", outcome, physical_role=physical_role)

    # Uvicorn restores and re-raises SIGTERM after the ASGI shutdown event.
    # The outer wrapper cannot observe that signal before process exit.
    app.router.add_event_handler("shutdown", lambda: record_shutdown("ok"))

    try:
        uvicorn.run(app, host=args.host, port=int(args.port), log_level="info")
    except BaseException:
        record_shutdown("failed")
        raise
    else:
        record_shutdown("ok")
    finally:
        stop.set()
        if driver_thread is not None:
            driver_thread.join(timeout=5)
        if review_thread is not None:
            review_thread.join(timeout=5)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
