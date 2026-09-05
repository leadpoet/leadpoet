"""Small, restart-safe driver for daily Arena rounds."""

from __future__ import annotations


def drive_once(service) -> str:
    """Advance all active rounds and ensure one submission round is open."""

    outcome = _advance_active(service)
    parts = [] if outcome == "idle" else [outcome]
    try:
        rewards = service.activate_pending_rewards()
    except Exception as exc:
        parts.append("failed activate_rewards: %s" % type(exc).__name__)
    else:
        activated = int(rewards.get("activated") or 0)
        if activated:
            parts.append("activated rewards %d" % activated)
    return "; ".join(parts) if parts else "idle"


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
            outcomes.append(
                "failed advance_round %s: %s"
                % (row["round_id"], type(exc).__name__)
            )
        else:
            outcomes.append("advanced %s" % row["round_id"])
    if not any(row.get("status") == "open" for row in active):
        try:
            ensured = service.ensure_daily_round()
        except Exception as exc:
            outcomes.append("failed ensure_daily_round: %s" % type(exc).__name__)
        else:
            if ensured.get("status") == "created":
                outcomes.append("created %s" % ensured.get("round_id"))
    return "; ".join(outcomes) if outcomes else "idle"


__all__ = ["drive_once"]
