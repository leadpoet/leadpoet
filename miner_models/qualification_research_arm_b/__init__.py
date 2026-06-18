"""Leadpoet Research Lab qualification Arm B.

This package is a fixed lab artifact for paired daily baseline measurements.
It is not used by champion gating and is not a miner submission.
"""
from .qualify import MAX_LEADS_PER_ICP, qualify

__all__ = ["MAX_LEADS_PER_ICP", "qualify"]
