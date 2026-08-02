"""Cross-runtime observability helpers for Leadpoet HOST processes.

Currently: opt-in, non-authoritative Sentry host observability. See
``docs/sentry_error_monitoring.md`` and ``sentry_bootstrap.py`` for the
contract. This package must stay stdlib-only at import time so every entry
point (gateway, validator, miner, auditor, Research Lab workers) can wire
it unconditionally with zero side effects when disabled.
"""

from leadpoet_observability.sentry_bootstrap import (
    init_sentry,
    sentry_enabled,
    set_sentry_tag,
)
from leadpoet_observability.sentry_operations import (
    INCIDENT_FAILURE_CODES,
    capture_failure,
    configure_sentry_context,
    emit_release_summary,
    failure_code_for_exception,
    hash_identifier,
    record_retry,
    record_stage,
    release_correlation_id,
    sentry_context,
    sentry_stage,
    weight_correlation_id,
)

__all__ = [
    "INCIDENT_FAILURE_CODES",
    "capture_failure",
    "configure_sentry_context",
    "emit_release_summary",
    "failure_code_for_exception",
    "hash_identifier",
    "init_sentry",
    "record_retry",
    "record_stage",
    "release_correlation_id",
    "sentry_context",
    "sentry_enabled",
    "sentry_stage",
    "set_sentry_tag",
    "weight_correlation_id",
]
