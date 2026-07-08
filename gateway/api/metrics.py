"""Minimal gateway metrics endpoint."""

from __future__ import annotations

from fastapi import APIRouter
from starlette.responses import PlainTextResponse

from gateway.utils.circuit_breaker import db_breaker
from gateway.utils.db_executor import DB_QUEUE_HIGH_WATER, DB_THREADS, db_queue_depth
from gateway.utils.hotkey_bucket import ALL_BUCKETS, RECENT_NONCES
from gateway.utils.loop_watchdog import last_ping_age_seconds
from gateway.utils import ops_registry


router = APIRouter()


@router.get("/metrics", response_class=PlainTextResponse)
async def metrics() -> str:
    lines = [
        f"gateway_db_queue_depth {db_queue_depth()}",
        f"gateway_db_queue_high_water {DB_QUEUE_HIGH_WATER}",
        f"gateway_db_threads_max {DB_THREADS}",
        f"gateway_loop_last_ping_age_seconds {last_ping_age_seconds():.3f}",
    ]
    breaker = db_breaker.snapshot()
    lines.extend([
        f'gateway_breaker_state{{state="{breaker["state"]}"}} 1',
        f'gateway_breaker_times_opened {breaker["times_opened"]}',
        f'gateway_breaker_consecutive_failures {breaker["consecutive_failures"]}',
    ])
    pm = ops_registry.priority_middleware
    if pm is not None:
        for key, value in pm.snapshot().items():
            if isinstance(value, dict):
                for label, count in value.items():
                    lines.append(f'gateway_priority_{key}{{class="{label}"}} {count}')
            else:
                lines.append(f"gateway_priority_{key} {value}")
    for bucket in ALL_BUCKETS:
        snap = bucket.snapshot()
        name = snap["name"]
        for key in ("active_keys", "allowed", "denied", "observed_denied"):
            lines.append(f'gateway_hotkey_bucket_{key}{{bucket="{name}"}} {snap[key]}')
    lines.append(f'gateway_recent_nonces_active {RECENT_NONCES.snapshot()["active_nonces"]}')
    return "\n".join(lines) + "\n"
