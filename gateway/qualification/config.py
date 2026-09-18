"""Configuration used by the active qualification scoring library."""

import os
from dataclasses import dataclass


@dataclass
class QualificationConfig:
    """Scoring limits that remain part of the Arena scorer contract."""

    MAX_COST_PER_LEAD_USD: float = 0.10
    COST_VARIABILITY_THRESHOLD_MULTIPLIER: float = 2.0
    VARIABILITY_PENALTY_POINTS: int = 5
    RUNNING_MODEL_TIMEOUT_SECONDS: int = 320
    INTENT_SIGNAL_DECAY_50_PCT_MONTHS: int = 2
    INTENT_SIGNAL_DECAY_25_PCT_MONTHS: int = 12

    @classmethod
    def from_env(cls) -> "QualificationConfig":
        return cls(
            MAX_COST_PER_LEAD_USD=float(
                os.getenv("QUAL_MAX_COST_PER_LEAD_USD", "0.10")
            ),
            RUNNING_MODEL_TIMEOUT_SECONDS=int(
                os.getenv("QUAL_RUNNING_MODEL_TIMEOUT_SECONDS", "320")
            ),
            INTENT_SIGNAL_DECAY_50_PCT_MONTHS=int(
                os.getenv("QUAL_INTENT_SIGNAL_DECAY_50_PCT_MONTHS", "2")
            ),
            INTENT_SIGNAL_DECAY_25_PCT_MONTHS=int(
                os.getenv("QUAL_INTENT_SIGNAL_DECAY_25_PCT_MONTHS", "12")
            ),
        )

    def get_cost_penalty_threshold(self) -> float:
        """Return the per-company cost threshold for a variability penalty."""

        return (
            self.MAX_COST_PER_LEAD_USD
            * self.COST_VARIABILITY_THRESHOLD_MULTIPLIER
        )


CONFIG = QualificationConfig.from_env()
