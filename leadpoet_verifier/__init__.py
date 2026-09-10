"""Open deterministic verifier surface for Leadpoet Research Lab.

This package intentionally depends only on the Python standard library. It is
the Phase 0 extraction target for checks and arithmetic that validators should
be able to rerun without sealed judge code, gateway services, or network I/O.
"""

from .aggregation import (
    DEFAULT_INTENT_SIGNAL_DECAY_25_PCT_MONTHS,
    DEFAULT_INTENT_SIGNAL_DECAY_50_PCT_MONTHS,
    MAX_COMPANY_ICP_FIT_SCORE,
    MAX_COMPANY_INTENT_SIGNAL_SCORE,
    MAX_COMPANY_TOTAL_SCORE,
    NO_DATE_DECAY_MULTIPLIER,
    SOURCE_TYPE_MULTIPLIERS,
    aggregate_set_score,
    apply_signal_time_decay,
    calculate_age_months,
    calculate_time_decay_multiplier,
    company_final_score,
    per_icp_normalized_score,
    u16_weights_from_scores,
)
from .attestation import (
    is_pcr0_allowed,
    load_pcr0_allowlist,
    validate_attestation_response_shape,
)
from .l0 import (
    Finding,
    L0Result,
    check_date_precision,
    compute_snippet_overlap,
    run_l0_checks,
)

__all__ = [
    "Finding",
    "L0Result",
    "DEFAULT_INTENT_SIGNAL_DECAY_25_PCT_MONTHS",
    "DEFAULT_INTENT_SIGNAL_DECAY_50_PCT_MONTHS",
    "MAX_COMPANY_ICP_FIT_SCORE",
    "MAX_COMPANY_INTENT_SIGNAL_SCORE",
    "MAX_COMPANY_TOTAL_SCORE",
    "NO_DATE_DECAY_MULTIPLIER",
    "SOURCE_TYPE_MULTIPLIERS",
    "aggregate_set_score",
    "apply_signal_time_decay",
    "calculate_age_months",
    "calculate_time_decay_multiplier",
    "check_date_precision",
    "company_final_score",
    "compute_snippet_overlap",
    "is_pcr0_allowed",
    "load_pcr0_allowlist",
    "per_icp_normalized_score",
    "run_l0_checks",
    "u16_weights_from_scores",
    "validate_attestation_response_shape",
]
