"""Gated candidate submission placeholder for Engine-generated fixes."""

from __future__ import annotations

from .config import ImprovementEngineConfig


def candidate_submission_enabled(config: ImprovementEngineConfig) -> bool:
    return config.mode == "active" and config.auto_submit_candidates
