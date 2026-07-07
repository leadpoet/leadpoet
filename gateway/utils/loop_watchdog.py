"""
Event-loop stall watchdog.

The gateway is a single-event-loop process: one blocking call on the loop
freezes every endpoint at once (health checks included), and from the
outside that is indistinguishable from the process being down.  Past
incidents surfaced only when miners reported read-timeouts.

This watchdog runs in a plain thread, so it keeps working precisely when
the loop does not.  Every ``ping_interval`` seconds it schedules a no-op
callback onto the loop and waits up to ``stall_threshold`` seconds for it
to run.  If the callback never runs, the loop is wedged: the watchdog logs
CRITICAL and dumps every thread's Python stack to stderr (which the
gateway redirects into gateway.log), so the blocking frame is captured in
the log at the moment it happens instead of being reconstructed after a
restart.
"""

import faulthandler
import logging
import os
import sys
import threading
import time

logger = logging.getLogger(__name__)

PING_INTERVAL_SECONDS = float(os.getenv("LOOP_WATCHDOG_PING_INTERVAL_SECONDS", "10"))
STALL_THRESHOLD_SECONDS = float(os.getenv("LOOP_WATCHDOG_STALL_THRESHOLD_SECONDS", "30"))
# Re-dump at most this often while a stall persists, so a long wedge doesn't
# flood the log with identical tracebacks.
DUMP_COOLDOWN_SECONDS = float(os.getenv("LOOP_WATCHDOG_DUMP_COOLDOWN_SECONDS", "300"))

_started = threading.Event()


def start_loop_watchdog(loop) -> None:
    """Start the watchdog thread for ``loop``.  Idempotent."""
    if _started.is_set():
        return
    _started.set()

    def _watch() -> None:
        last_dump_mono = 0.0
        stall_started_mono = None
        while True:
            pong = threading.Event()
            try:
                loop.call_soon_threadsafe(pong.set)
            except RuntimeError:
                # Loop is closed — process is shutting down.
                return
            responded = pong.wait(STALL_THRESHOLD_SECONDS)
            now_mono = time.monotonic()
            if responded:
                if stall_started_mono is not None:
                    logger.critical(
                        "EVENT LOOP RECOVERED after %.1fs stall",
                        now_mono - stall_started_mono,
                    )
                    stall_started_mono = None
                time.sleep(PING_INTERVAL_SECONDS)
                continue

            if stall_started_mono is None:
                stall_started_mono = now_mono - STALL_THRESHOLD_SECONDS
            stalled_for = now_mono - stall_started_mono
            logger.critical(
                "EVENT LOOP STALLED: no-op callback not run within %.0fs "
                "(stalled ~%.0fs total) — every endpoint is frozen; "
                "dumping all thread stacks",
                STALL_THRESHOLD_SECONDS,
                stalled_for,
            )
            if now_mono - last_dump_mono >= DUMP_COOLDOWN_SECONDS:
                last_dump_mono = now_mono
                try:
                    faulthandler.dump_traceback(file=sys.stderr, all_threads=True)
                except Exception:  # pragma: no cover — never let the watchdog die
                    logger.exception("loop watchdog failed to dump tracebacks")

    thread = threading.Thread(target=_watch, name="loop-watchdog", daemon=True)
    thread.start()
    logger.info(
        "Loop watchdog started (ping every %.0fs, stall threshold %.0fs)",
        PING_INTERVAL_SECONDS,
        STALL_THRESHOLD_SECONDS,
    )
