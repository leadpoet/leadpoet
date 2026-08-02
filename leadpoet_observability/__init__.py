"""Cross-runtime observability helpers for Leadpoet HOST processes.

Currently: opt-in, fail-closed Sentry error monitoring. See
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

__all__ = ["init_sentry", "sentry_enabled", "set_sentry_tag"]
