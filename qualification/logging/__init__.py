"""
Qualification System: Logging Module

Phase 7.1 from tasks10.md

This module integrates the Lead Qualification Agent competition with
the EXISTING transparency log infrastructure. It uses the same signed
envelope format as sourcing events.

Event Types (added to gateway/models/events.py):
- MODEL_SUBMITTED - Miner submits a model for evaluation
- EVALUATION_COMPLETE - Model evaluation finished
- CHAMPION_SELECTED - New champion selected or retained
- EMISSIONS_DISTRIBUTED - 5% emissions distributed or burned

CRITICAL: This module uses the EXISTING transparency_log table and
enclave signer. Do NOT create duplicate logging infrastructure.
"""

from qualification.logging.events import (
    # Main logging function
    log_qualification_event,
    # Specific event loggers
    log_model_submitted,
    log_evaluation_complete,
    log_champion_selected,
    log_emissions_distributed,
    # Event type constants
    EVENT_MODEL_SUBMITTED,
    EVENT_EVALUATION_COMPLETE,
    EVENT_CHAMPION_SELECTED,
    EVENT_EMISSIONS_DISTRIBUTED,
    # Payload models
    ModelSubmittedPayload,
    EvaluationCompletePayload,
    ChampionSelectedPayload,
    EmissionsDistributedPayload,
    # Utilities
    get_qualification_events,
    get_event_by_hash,
    verify_event_signature,
    is_logging_configured,
)

__all__ = [
    # Main logging function
    "log_qualification_event",
    # Specific event loggers
    "log_model_submitted",
    "log_evaluation_complete",
    "log_champion_selected",
    "log_emissions_distributed",
    # Event type constants
    "EVENT_MODEL_SUBMITTED",
    "EVENT_EVALUATION_COMPLETE",
    "EVENT_CHAMPION_SELECTED",
    "EVENT_EMISSIONS_DISTRIBUTED",
    # Payload models
    "ModelSubmittedPayload",
    "EvaluationCompletePayload",
    "ChampionSelectedPayload",
    "EmissionsDistributedPayload",
    # Utilities
    "get_qualification_events",
    "get_event_by_hash",
    "verify_event_signature",
    "is_logging_configured",
]
