"""Export canonical Research Lab score-bundle aggregates as Langfuse scores."""

from __future__ import annotations

import logging
from typing import Any, Mapping

from .langfuse_client import get_langfuse_client, redaction_mode
from .redaction import RedactionBlocked, redact_for_langfuse


logger = logging.getLogger(__name__)


def langfuse_trace_id_from_observation(obs: Any | None) -> str:
    if obs is None:
        return ""
    for name in ("trace_id", "traceId", "id"):
        value = getattr(obs, name, None)
        if value:
            return str(value)
    return ""


def score_payload_from_bundle(score_bundle: Mapping[str, Any]) -> dict[str, Any]:
    aggregates = score_bundle.get("aggregates") if isinstance(score_bundle.get("aggregates"), Mapping) else {}
    reward_path = score_bundle.get("reward_path") if isinstance(score_bundle.get("reward_path"), Mapping) else {}
    scoring_health = score_bundle.get("scoring_health") if isinstance(score_bundle.get("scoring_health"), Mapping) else {}
    baseline_health = score_bundle.get("baseline_health") if isinstance(score_bundle.get("baseline_health"), Mapping) else {}
    private_gate = score_bundle.get("private_holdout_gate") if isinstance(score_bundle.get("private_holdout_gate"), Mapping) else {}
    return {
        "base_score": aggregates.get("base_score"),
        "candidate_score": aggregates.get("candidate_score"),
        "mean_delta": aggregates.get("mean_delta"),
        "delta_lcb": aggregates.get("delta_lcb"),
        "sd_delta": aggregates.get("sd_delta"),
        "se_delta": aggregates.get("se_delta"),
        "icp_count": aggregates.get("icp_count"),
        "successful_icp_count": aggregates.get("successful_icp_count"),
        "hard_failure_count": aggregates.get("hard_failure_count"),
        "total_cost_usd": aggregates.get("total_cost_usd") or score_bundle.get("total_cost_usd"),
        "eligible_for_probation": reward_path.get("eligible_for_probation"),
        "eligible_for_crown": reward_path.get("eligible_for_crown"),
        "eligible_for_improvement_grant": reward_path.get("eligible_for_improvement_grant"),
        "baseline_health_gate_passed": baseline_health.get("gate_passed"),
        "scoring_health_status": scoring_health.get("health_status"),
        "sourced_zero_no_error_count": scoring_health.get("sourced_zero_no_error_count"),
        "public_holdout_decision": scoring_health.get("public_holdout_decision") or private_gate.get("decision"),
    }


def export_score_bundle_scores(
    trace_id: str,
    score_bundle: Mapping[str, Any],
    *,
    metadata: Mapping[str, Any] | None = None,
) -> None:
    if not trace_id:
        return
    client = get_langfuse_client()
    if client is None:
        return
    base_metadata = {
        "score_bundle_hash": score_bundle.get("score_bundle_hash"),
        "run_id": score_bundle.get("run_id"),
        "ticket_id": score_bundle.get("ticket_id"),
        "execution_trace_ref": score_bundle.get("execution_trace_ref"),
        **dict(metadata or {}),
    }
    for name, value in score_payload_from_bundle(score_bundle).items():
        if value is None:
            continue
        try:
            client.score(
                trace_id=trace_id,
                name=name,
                value=value,
                metadata=redact_for_langfuse(base_metadata, mode=redaction_mode()),
            )
        except RedactionBlocked as exc:
            logger.warning("langfuse_score_redaction_blocked name=%s error=%s", name, str(exc)[:200])
        except Exception as exc:
            logger.warning("langfuse_score_export_failed name=%s error=%s", name, str(exc)[:200])
