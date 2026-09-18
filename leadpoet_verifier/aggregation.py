"""Score normalization used by Arena competition scoring."""

from __future__ import annotations

from typing import Iterable


_MAX_COMPANY_TOTAL_SCORE = 100.0


def per_icp_normalized_score(
    lead_scores: Iterable[float],
    *,
    max_leads: int = 5,
) -> float:
    """Normalize one ICP's company scores by the fixed maximum lead count."""
    if max_leads <= 0:
        raise ValueError("max_leads must be positive")
    total = sum(
        max(0.0, min(float(score), _MAX_COMPANY_TOTAL_SCORE))
        for score in lead_scores
    )
    return total / float(max_leads)
