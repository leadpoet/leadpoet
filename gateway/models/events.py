"""
Gateway Event Models
===================

Pydantic models for event types and payloads.
"""

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from datetime import datetime
from enum import Enum


class EventType(str, Enum):
    """Event types for transparency log"""
    
    # Miner lifecycle
    SUBMISSION_REQUEST = "SUBMISSION_REQUEST"
    STORAGE_PROOF = "STORAGE_PROOF"
    UPLOAD_FAILED = "UPLOAD_FAILED"
    SUBMISSION = "SUBMISSION"
    
    # Validator lifecycle
    GET_LEAD = "GET_LEAD"
    RECEIPT = "RECEIPT"
    VALIDATION_RESULT = "VALIDATION_RESULT"
    EPOCH_MANIFEST = "EPOCH_MANIFEST"
    
    # Epoch management
    EPOCH_INITIALIZATION = "EPOCH_INITIALIZATION"  # Combined: boundaries + queue + assignment
    EPOCH_INPUTS = "EPOCH_INPUTS"
    EPOCH_END = "EPOCH_END"
    WEIGHT_COMMIT = "WEIGHT_COMMIT"
    
    # Integrity
    CHECKPOINT_ROOT = "CHECKPOINT_ROOT"
    ANCHOR_ROOT = "ANCHOR_ROOT"
    UNAVAILABLE = "UNAVAILABLE"
    COMMIT_MISMATCH = "COMMIT_MISMATCH"
    
    # Consensus
    CONSENSUS_RESULT = "CONSENSUS_RESULT"


class BaseEvent(BaseModel):
    """Base event structure for all transparency log events"""
    
    event_type: EventType
    actor_hotkey: str = Field(..., description="SS58 address")
    nonce: str = Field(..., description="UUID v4")
    ts: datetime
    payload_hash: str = Field(..., description="SHA256 hex")
    build_id: str
    signature: str = Field(..., description="Ed25519 signature")


class SubmissionRequestPayload(BaseModel):
    """Payload for SUBMISSION_REQUEST event"""
    
    lead_id: str = Field(..., description="UUID")
    lead_blob_hash: str = Field(..., description="SHA256 of lead_blob")
    email_hash: str = Field(..., description="SHA256 of email field - for duplicate detection")


class SubmissionRequestEvent(BaseEvent):
    """Miner requests presigned URLs for lead submission"""
    
    event_type: EventType = EventType.SUBMISSION_REQUEST
    payload: SubmissionRequestPayload


class StorageProofPayload(BaseModel):
    """Payload for STORAGE_PROOF event"""
    
    lead_id: str
    lead_blob_hash: str
    email_hash: str = Field(..., description="SHA256 of email field - for duplicate detection")
    mirror: str = Field(..., description="Storage backend (currently only s3 is supported)")
    verified: bool


class StorageProofEvent(BaseEvent):
    """Gateway verified storage on a mirror"""
    
    event_type: EventType = EventType.STORAGE_PROOF
    payload: StorageProofPayload


class ValidationResultPayload(BaseModel):
    """Payload for VALIDATION_RESULT event"""
    
    lead_id: str
    decision_hash: str = Field(..., description="H(decision + salt)")
    rep_score_hash: str = Field(..., description="H(rep_score + salt)")
    rejection_reason_hash: str = Field(..., description="H(rejection_reason + salt)")
    evidence_hash: str = Field(..., description="H(evidence_blob) - no salt")
    # Note: evidence_blob is sent separately for private storage


class ValidationResultEvent(BaseEvent):
    """Validator submits validation result (commit phase)"""
    
    event_type: EventType = EventType.VALIDATION_RESULT
    payload: ValidationResultPayload


class EpochManifestPayload(BaseModel):
    """Payload for EPOCH_MANIFEST event"""
    
    epoch_id: int
    manifest_root: str = Field(..., description="Merkle root of validator's work")
    validation_count: int


class EpochManifestEvent(BaseEvent):
    """Validator submits epoch manifest"""
    
    event_type: EventType = EventType.EPOCH_MANIFEST
    payload: EpochManifestPayload


class WeightCommitPayload(BaseModel):
    """Payload for WEIGHT_COMMIT event"""
    
    epoch_id: int
    inputs_hash: str = Field(..., description="Hash of EPOCH_INPUTS")
    manifest_root: str
    csv_hash: str = Field(..., description="Hash of emissions CSV")


class WeightCommitEvent(BaseEvent):
    """Validator commits to weight calculation"""
    
    event_type: EventType = EventType.WEIGHT_COMMIT
    payload: WeightCommitPayload

