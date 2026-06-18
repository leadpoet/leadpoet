"""Lab-only qualification fork for Research Lab Engine v0 calibration.

This package preserves the validator-facing ``qualify(icp)`` shape while
running only against frozen Research Lab fixtures. It is not a miner submission,
not used by champion gating, and never calls live discovery or scoring APIs.
"""

from .qualify import (
    MAX_LEADS_PER_ICP,
    qualify,
    qualify_overbroad_v0,
    qualify_reference,
    qualify_source_routing_v0,
)

__all__ = [
    "MAX_LEADS_PER_ICP",
    "qualify",
    "qualify_overbroad_v0",
    "qualify_reference",
    "qualify_source_routing_v0",
]
