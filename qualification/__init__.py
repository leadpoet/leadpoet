"""
Lead Qualification Agent Competition System

This module implements the King-of-the-Hill competition for lead qualification
models. Miners submit open-source models that compete to earn 5% of subnet
emissions.

Key Components:
- config: Configuration and environment variables
- models: Pydantic schemas for all data types
- api: FastAPI endpoints for model submission and status
- validator: TEE sandbox execution and evaluation
- proxy: API proxy gateway with cost tracking
- scoring: Intent verification and lead scoring
- logging: Transparency log integration

See business_files/tasks10.md for full implementation plan.
See business_files/FINAL Lead Qualification IM BRD.md for requirements.
"""

__version__ = "0.1.0"
__author__ = "Leadpoet"

# Configuration
from gateway.qualification.config import CONFIG, QualificationConfig

# Models - Enums
from gateway.qualification.models import (
    ModelStatus,
    EvaluationStatus,
    EvaluationRunStatus,
    EvaluationStage,
    IntentSignalSource,
    Seniority,
    CostModel,
)

# Models - Core
from gateway.qualification.models import (
    IntentSignal,
    LeadOutput,
    LeadOutputRedacted,
    ICPPrompt,
    ICPSet,
    ModelSubmission,
    ModelSubmissionResponse,
    EvaluationResult,
    LeadScoreBreakdown,
    ModelStatusResponse,
    ModelScoreResponse,
    ChampionInfo,
)

# Models - Validator
from gateway.qualification.models import (
    ValidatorRegistration,
    ValidatorHeartbeat,
    EvaluationWorkItem,
    EvaluationWorkResponse,
)

# Models - API
from gateway.qualification.models import AllowedAPI

# Models - Transparency Log Payloads
from gateway.qualification.models import (
    ModelSubmittedPayload,
    EvaluationCompletePayload,
    ChampionSelectedPayload,
    EmissionsDistributedPayload,
)

__all__ = [
    # Configuration
    "CONFIG",
    "QualificationConfig",
    
    # Enums
    "ModelStatus",
    "EvaluationStatus",
    "EvaluationRunStatus",
    "EvaluationStage",
    "IntentSignalSource",
    "Seniority",
    "CostModel",
    
    # Core Models
    "IntentSignal",
    "LeadOutput",
    "LeadOutputRedacted",
    "ICPPrompt",
    "ICPSet",
    "ModelSubmission",
    "ModelSubmissionResponse",
    "EvaluationResult",
    "LeadScoreBreakdown",
    "ModelStatusResponse",
    "ModelScoreResponse",
    "ChampionInfo",
    
    # Validator Models
    "ValidatorRegistration",
    "ValidatorHeartbeat",
    "EvaluationWorkItem",
    "EvaluationWorkResponse",
    
    # API Models
    "AllowedAPI",
    
    # Transparency Log Payloads
    "ModelSubmittedPayload",
    "EvaluationCompletePayload",
    "ChampionSelectedPayload",
    "EmissionsDistributedPayload",
]
