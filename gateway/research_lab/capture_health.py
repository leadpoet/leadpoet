"""Capture-config preflight for Research Lab workers.

trajectoryimprovements.md P5 ("Make Capture Config Fail-Loud In Production"):
the capture-enabled flags default true while the S3 destinations have no
default, so an unset prefix used to be a "green" config that captured nothing
after a single INFO log. This module asserts the invariant chain

    enabled  ⇒  S3 prefix set  ⇒  KMS key set

per capture channel, emits one structured health block at worker startup, and
in the production profile refuses to start unless an explicit override is set
(``RESEARCH_LAB_CAPTURE_HEALTH_ALLOW_DEGRADED=true``).

Channels covered:

* ``raw_trace``        — OpenRouter request/response traces (worker.py)
* ``scorer_trace``     — scorer judgment breakdowns (scoring_worker.py)
* ``incontainer_trace``— private-model in-container provider traces
* ``trajectory_projector`` — Supabase corpus projection flag (P16)
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - import cycle guard (worker imports us)
    from gateway.research_lab.config import ResearchLabGatewayConfig

logger = logging.getLogger(__name__)

# Env names are duplicated here (rather than imported from worker/scoring
# modules) to avoid import cycles; a pinning test asserts they stay in sync.
RAW_TRACE_CAPTURE_ENABLED_ENV = "RESEARCH_LAB_RAW_TRACE_CAPTURE_ENABLED"
RAW_TRACE_S3_PREFIX_ENV = "RESEARCH_LAB_RAW_TRACE_S3_PREFIX"
SCORER_TRACE_CAPTURE_ENV = "RESEARCH_LAB_SCORER_TRACE_CAPTURE"
SCORER_TRACE_S3_PREFIX_ENV = "RESEARCH_LAB_SCORER_TRACE_S3_PREFIX"
INCONTAINER_TRACE_CAPTURE_ENV = "RESEARCH_LAB_INCONTAINER_TRACE_CAPTURE"
INCONTAINER_TRACE_S3_PREFIX_ENV = "RESEARCH_LAB_INCONTAINER_TRACE_S3_PREFIX"
INCONTAINER_TRACE_KMS_KEY_ENV = "RESEARCH_LAB_INCONTAINER_TRACE_KMS_KEY_ID"
PROJECTOR_ENABLED_ENV = "RESEARCH_LAB_TRAJECTORY_PROJECTOR_ENABLED"

ALLOW_DEGRADED_ENV = "RESEARCH_LAB_CAPTURE_HEALTH_ALLOW_DEGRADED"
ENFORCE_ENV = "RESEARCH_LAB_CAPTURE_HEALTH_ENFORCE"

_TRUTHY = {"1", "true", "yes", "on"}


class CaptureHealthError(RuntimeError):
    """Raised when production capture config is degraded without an override."""


def _flag(name: str, default: str) -> bool:
    return str(os.getenv(name, default)).strip().lower() in _TRUTHY


def _manifest_s3_parent(config: "ResearchLabGatewayConfig") -> str:
    manifest_uri = str(getattr(config, "private_model_manifest_uri", "") or "").strip()
    if not manifest_uri.startswith("s3://"):
        return ""
    bucket, _sep, key = manifest_uri[5:].partition("/")
    if not bucket:
        return ""
    base_prefix = key.rsplit("/", 1)[0] if "/" in key else ""
    return f"s3://{bucket}/{base_prefix}".rstrip("/")


def _resolved_prefix(env_name: str, config: "ResearchLabGatewayConfig", *, manifest_fallback: bool) -> str:
    prefix = str(os.getenv(env_name, "")).strip().rstrip("/")
    if prefix:
        return prefix if prefix.startswith("s3://") else ""
    if manifest_fallback:
        return _manifest_s3_parent(config)
    return ""


def _channel(*, enabled: bool, prefix: str, kms_key: str) -> dict[str, Any]:
    if not enabled:
        status = "disabled"
    elif not prefix:
        status = "misconfigured_missing_s3_prefix"
    elif not kms_key:
        status = "misconfigured_missing_kms_key"
    else:
        status = "ok"
    return {
        "enabled": enabled,
        "s3_configured": bool(prefix),
        "s3_prefix": prefix,
        "kms_configured": bool(kms_key),
        "status": status,
    }


def collect_capture_health(config: "ResearchLabGatewayConfig") -> dict[str, Any]:
    """Build the structured capture health block (pure env/config, no I/O)."""
    score_bundle_kms = str(getattr(config, "score_bundle_kms_key_id", "") or "").strip()
    projector_on = str(os.getenv(PROJECTOR_ENABLED_ENV, "")).strip().lower() in _TRUTHY
    channels = {
        "raw_trace": _channel(
            enabled=_flag(RAW_TRACE_CAPTURE_ENABLED_ENV, "true"),
            prefix=_resolved_prefix(RAW_TRACE_S3_PREFIX_ENV, config, manifest_fallback=True),
            kms_key=score_bundle_kms,
        ),
        "scorer_trace": _channel(
            enabled=_flag(SCORER_TRACE_CAPTURE_ENV, "true"),
            prefix=_resolved_prefix(SCORER_TRACE_S3_PREFIX_ENV, config, manifest_fallback=True),
            kms_key=score_bundle_kms,
        ),
        "incontainer_trace": _channel(
            enabled=_flag(INCONTAINER_TRACE_CAPTURE_ENV, "true"),
            prefix=_resolved_prefix(INCONTAINER_TRACE_S3_PREFIX_ENV, config, manifest_fallback=False),
            kms_key=str(os.getenv(INCONTAINER_TRACE_KMS_KEY_ENV, "") or "").strip(),
        ),
    }
    return {
        "channels": channels,
        "trajectory_projector_enabled": projector_on,
        "production_writes_enabled": bool(getattr(config, "production_writes_enabled", False)),
    }


def capture_health_violations(health: dict[str, Any]) -> list[str]:
    """List every capture invariant broken by this config.

    In the production profile a disabled channel is a violation too: corpus
    capture is mission-critical, so opting out must go through the explicit
    ALLOW_DEGRADED override, never a quietly-unset env var.
    """
    violations: list[str] = []
    for name, channel in (health.get("channels") or {}).items():
        status = str(channel.get("status") or "")
        if status.startswith("misconfigured"):
            violations.append(f"{name}:{status}")
        elif status == "disabled":
            violations.append(f"{name}:disabled")
    if not health.get("trajectory_projector_enabled"):
        violations.append(f"trajectory_projector:disabled (set {PROJECTOR_ENABLED_ENV}=true)")
    return violations


def enforce_capture_health(
    config: "ResearchLabGatewayConfig",
    *,
    worker_kind: str,
) -> dict[str, Any]:
    """Log one structured capture health block; fail loud in production.

    Enforcement applies when ``production_writes_enabled`` is true (or the
    ``RESEARCH_LAB_CAPTURE_HEALTH_ENFORCE`` env forces it); the escape hatch is
    ``RESEARCH_LAB_CAPTURE_HEALTH_ALLOW_DEGRADED=true``, which downgrades the
    refusal to a loud error log so a deliberate degraded start stays visible.
    """
    health = collect_capture_health(config)
    violations = capture_health_violations(health)
    health["violations"] = violations
    summary = ", ".join(
        f"{name}={channel.get('status')}" for name, channel in health["channels"].items()
    )
    logger.info(
        "research_lab_capture_health worker=%s %s projector_enabled=%s production=%s violations=%s",
        worker_kind,
        summary,
        health["trajectory_projector_enabled"],
        health["production_writes_enabled"],
        len(violations),
    )
    if not violations:
        return health
    enforce = _flag(ENFORCE_ENV, "true" if health["production_writes_enabled"] else "false")
    allow_degraded = _flag(ALLOW_DEGRADED_ENV, "false")
    detail = "; ".join(violations)
    if enforce and not allow_degraded:
        logger.error(
            "research_lab_capture_health_refusing_start worker=%s violations=[%s] "
            "(set %s=true to start degraded — capture gaps are unrecoverable data loss)",
            worker_kind,
            detail,
            ALLOW_DEGRADED_ENV,
        )
        raise CaptureHealthError(
            f"capture config degraded for {worker_kind}: {detail}; "
            f"set {ALLOW_DEGRADED_ENV}=true to override"
        )
    log = logger.error if enforce else logger.warning
    log(
        "research_lab_capture_health_degraded worker=%s violations=[%s] override=%s",
        worker_kind,
        detail,
        allow_degraded,
    )
    return health


async def check_projector_tables() -> dict[str, str]:
    """Best-effort presence probe for the schema-27 corpus tables.

    Returns ``{table: "present" | "error:<short>"}``; never raises. Callers
    treat this as reporting, not enforcement (the projector itself degrades
    safely when tables are missing).
    """
    from gateway.research_lab.store import select_many

    tables = (
        "research_trajectories",
        "research_trajectory_events",
        "execution_traces",
        "evidence_bundles",
        "research_lab_results_ledger",
    )
    out: dict[str, str] = {}
    for table in tables:
        try:
            await select_many(table, columns="id", limit=1)
            out[table] = "present"
        except Exception as exc:  # noqa: BLE001 - reporting only
            out[table] = f"error:{str(exc)[:80]}"
    return out
