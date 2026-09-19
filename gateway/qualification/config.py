"""Configuration used by the active qualification scoring library."""

import os
from dataclasses import dataclass


@dataclass
class QualificationConfig:
    """Scoring limits that remain part of the Arena scorer contract."""

    RUNNING_MODEL_TIMEOUT_SECONDS: int = 320
    INTENT_SIGNAL_DECAY_50_PCT_MONTHS: int = 2
    INTENT_SIGNAL_DECAY_25_PCT_MONTHS: int = 12

    @classmethod
    def from_env(cls) -> "QualificationConfig":
        return cls(
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


CONFIG = QualificationConfig.from_env()
