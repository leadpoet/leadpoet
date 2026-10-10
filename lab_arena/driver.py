"""Small, restart-safe driver for daily Arena rounds."""

from __future__ import annotations

import time

from lab_arena import telemetry


def drive_once(service) -> str:
    """Advance all active rounds and ensure one submission round is open."""

    tick_start_ns = time.time_ns()
    tick_start_mono = time.monotonic()
    parts = []
    try:
        with telemetry.stage("promote_baselines") as observed:
            promotion = service.promote_pending_baselines()
            observed.count = int(promotion.get("promoted") or 0)
            observed.idle = observed.count == 0
    except Exception as exc:
        parts.append("failed promote_baselines: %s" % type(exc).__name__)
    else:
        promoted = int(promotion.get("promoted") or 0)
        if promoted:
            parts.append("promoted baselines %d" % promoted)
    outcome = _advance_active(service)
    if outcome != "idle":
        parts.append(outcome)
    try:
        with telemetry.stage("activate_rewards") as observed:
            rewards = service.activate_pending_rewards()
            observed.count = int(rewards.get("activated") or 0)
            observed.idle = observed.count == 0
    except Exception as exc:
        parts.append("failed activate_rewards: %s" % type(exc).__name__)
    else:
        activated = int(rewards.get("activated") or 0)
        if activated:
            parts.append("activated rewards %d" % activated)
    try:
        with telemetry.stage("reconcile_provider_costs") as observed:
            billing = service.reconcile_closed_provider_costs()
            observed.idle = billing.get("status") != "settled"
            observed.count = 0 if observed.idle else 1
    except Exception as exc:
        parts.append("failed closed_provider_costs: %s" % type(exc).__name__)
    else:
        if billing.get("status") == "settled":
            parts.append("reconciled closed provider cost")
    summary = "; ".join(parts) if parts else "idle"
    # One span per tick is the liveness signal: without it a wedged driver and
    # an idle one are indistinguishable from outside the process.
    telemetry.record(
        "driver_tick",
        "failed" if "failed" in summary else ("idle" if summary == "idle" else "ok"),
        count=len(parts),
        duration_ms=(time.monotonic() - tick_start_mono) * 1000.0,
        start_ns=tick_start_ns,
    )
    return summary


def _advance_active(service) -> str:
    try:
        with telemetry.stage("active_rounds") as observed:
            active = list(service.active_rounds())
            observed.count = len(active)
            observed.idle = not active
    except Exception as exc:
        return "failed active_rounds: %s" % type(exc).__name__
    outcomes = []
    for row in active:
        try:
            with telemetry.stage("advance_round") as observed:
                service.advance_round(row["round_id"])
                observed.count = 1
        except Exception as exc:
            outcomes.append(
                "failed advance_round %s: %s"
                % (row["round_id"], type(exc).__name__)
            )
        else:
            outcomes.append("advanced %s" % row["round_id"])
    # Advancing the open round can commit it during this tick. The snapshot in
    # ``active`` is then stale, so always use the service's idempotent open-round
    # check to create its successor without waiting for another driver tick.
    try:
        with telemetry.stage("ensure_daily_round") as observed:
            ensured = service.ensure_daily_round()
            observed.idle = ensured.get("status") != "created"
            observed.count = 0 if observed.idle else 1
    except Exception as exc:
        outcomes.append("failed ensure_daily_round: %s" % type(exc).__name__)
    else:
        if ensured.get("status") == "created":
            outcomes.append("created %s" % ensured.get("round_id"))
    return "; ".join(outcomes) if outcomes else "idle"


__all__ = ["drive_once"]
