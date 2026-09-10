"""Worker count and proxy names for the retained qualification service."""

from __future__ import annotations



# The V2 scoring enclave still supports the normal qualification pipeline.
# These proxy names are used only to size and seal that current worker fleet.
V2_SCORING_PROXY_PREFIXES = ("RESEARCH_LAB_V2_SCORING_HTTPS_PROXY",)
LEGACY_SCORING_PROXY_PREFIXES = (
    "RESEARCH_LAB_QUALIFICATION_WEBSHARE_PROXY",
    "QUALIFICATION_WEBSHARE_PROXY",
    "RESEARCH_LAB_SCORING_WORKER_PROXY",
)
SCORING_PROXY_PREFIXES = (
    *V2_SCORING_PROXY_PREFIXES,
    *LEGACY_SCORING_PROXY_PREFIXES,
)

MAX_WORKER_PROCESSES = 500












def resolve_worker_process_count(
    explicit_count: int,
    fallback_count: int,
    *,
    minimum: int = 0,
) -> int:
    """Return one bounded scoring-worker count for sealing and startup."""

    chosen = explicit_count if explicit_count > 0 else fallback_count
    return max(minimum, min(int(chosen), MAX_WORKER_PROCESSES))
