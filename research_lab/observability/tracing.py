"""Higher-level Research Lab trace helpers."""

from __future__ import annotations

from typing import Any, Mapping

from .langfuse_client import observation, update_observation
from .score_export import export_score_bundle_scores, langfuse_trace_id_from_observation

__all__ = [
    "observation",
    "safe_score_bundle_output",
    "finish_score_bundle_observation",
]


def safe_score_bundle_output(score_bundle: Mapping[str, Any]) -> dict[str, Any]:
    aggregates = score_bundle.get("aggregates") if isinstance(score_bundle.get("aggregates"), Mapping) else {}
    return {
        "score_bundle_hash": score_bundle.get("score_bundle_hash"),
        "run_id": score_bundle.get("run_id"),
        "ticket_id": score_bundle.get("ticket_id"),
        "candidate_artifact_hash": score_bundle.get("candidate_artifact_hash"),
        "mean_delta": aggregates.get("mean_delta"),
        "candidate_score": aggregates.get("candidate_score"),
        "base_score": aggregates.get("base_score"),
    }


def finish_score_bundle_observation(obs: Any | None, score_bundle: Mapping[str, Any]) -> str:
    update_observation(obs, output=safe_score_bundle_output(score_bundle))
    trace_id = langfuse_trace_id_from_observation(obs)
    export_score_bundle_scores(trace_id, score_bundle)
    return trace_id
