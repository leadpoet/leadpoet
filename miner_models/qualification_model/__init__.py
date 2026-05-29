"""qualification_model — universal evidence-first qualification miner.

See ``qualify.py`` for the validator entry point.
"""
from .qualify import qualify, MAX_LEADS_PER_ICP

__all__ = ["qualify", "MAX_LEADS_PER_ICP"]
