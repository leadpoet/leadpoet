"""Fixed loader for signed official-baseline release components.

The active artifact is supplied by the scoring worker and is never reloaded
here.  The reviewed fixed module below must construct a context-bound
registration, artifact projector, and protected action bridge.  Import or
identity failure keeps exact-v3 activation closed.
"""

from __future__ import annotations

import importlib
from typing import Any

from gateway.research_lab.official_baseline_model_runner import (
    OfficialBaselineAuthorityUnavailable,
    OfficialBaselineDependencyContext,
)
from gateway.research_lab.official_baseline_custody import (
    S3OfficialBaselineDocumentCustody,
)


OFFICIAL_BASELINE_RELEASE_AUTHORITIES_MODULE = (
    "gateway.research_lab.official_baseline_release_authorities"
)
OFFICIAL_BASELINE_RELEASE_COMPONENTS_LOADER = (
    "load_official_baseline_release_components"
)


def load_official_baseline_release_components(
    *,
    context: OfficialBaselineDependencyContext,
    custody: S3OfficialBaselineDocumentCustody,
) -> Any:
    """Load components from one protected fixed path, bound to frozen context."""

    if (
        not isinstance(context, OfficialBaselineDependencyContext)
        or not isinstance(custody, S3OfficialBaselineDocumentCustody)
    ):
        raise OfficialBaselineAuthorityUnavailable(
            "official baseline frozen release context is unavailable"
        )
    context.validate()
    try:
        module = importlib.import_module(OFFICIAL_BASELINE_RELEASE_AUTHORITIES_MODULE)
        loader = getattr(module, OFFICIAL_BASELINE_RELEASE_COMPONENTS_LOADER)
    except Exception as exc:
        raise OfficialBaselineAuthorityUnavailable(
            "official baseline signed release authority package is unavailable"
        ) from exc
    if not callable(loader):
        raise OfficialBaselineAuthorityUnavailable(
            "official baseline signed release authority loader is invalid"
        )
    return loader(context=context, custody=custody)


__all__ = [
    "OFFICIAL_BASELINE_RELEASE_AUTHORITIES_MODULE",
    "OFFICIAL_BASELINE_RELEASE_COMPONENTS_LOADER",
    "load_official_baseline_release_components",
]
