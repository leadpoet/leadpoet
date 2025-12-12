"""
Gateway Response Models
======================

Pydantic models for API responses.
"""

from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime


class PresignedURLResponse(BaseModel):
    """
    Response from /presign endpoint.
    
    NOTE (Phase 4): TEE-based trust model
    - Event is buffered in TEE (hardware-protected memory)
    - Will be included in next hourly Arweave checkpoint (signed by TEE)
    - Verify gateway code integrity: GET /attest
    """
    lead_id: str
    presigned_url: str  # S3 URL for upload (miner uploads here)
    s3_url: str  # Alias for backward compatibility
    expires_in: int
    timestamp: Optional[str] = None  # ISO 8601 timestamp when request was accepted


class SubmissionResponse(BaseModel):
    """
    Response from final submission confirmation.
    
    NOTE (Phase 4): TEE-based trust model
    - Events buffered in TEE (hardware-protected memory)
    - Will be included in next hourly Arweave checkpoint (signed by TEE)
    - Verify gateway code integrity: GET /attest
    """
    status: str
    lead_id: str
    storage_backends: Optional[List[str]] = []
    submission_timestamp: Optional[str] = None  # ISO 8601 timestamp of submission
    queue_position: Optional[int] = None  # Position in validation queue
    message: Optional[str] = None  # Informational message


class LeadResponse(BaseModel):
    """Response from /lead/{lead_id} endpoint"""
    
    lead_id: str
    lead_blob_hash: str
    miner_hotkey: str
    lead_blob: Dict[str, Any]
    inclusion_proof: List[str]


class ErrorResponse(BaseModel):
    """Error response"""
    
    error: str
    detail: Optional[str] = None


class HealthResponse(BaseModel):
    """Health check response"""
    
    service: str
    status: str
    build_id: str
    github_commit: str
    timestamp: str


class ValidationResultResponse(BaseModel):
    """
    Response from /validate endpoint.
    
    NOTE (Phase 4): TEE-based trust model
    - Event is buffered in TEE (hardware-protected memory)
    - Will be included in next hourly Arweave checkpoint (signed by TEE)
    - Verify gateway code integrity: GET /attest
    """
    status: str
    evidence_id: str
    lead_id: str
    epoch_id: int
    v_score: Optional[float] = None
    timestamp: Optional[str] = None  # ISO 8601 timestamp when validation was recorded
    message: Optional[str] = None


class RevealResponse(BaseModel):
    """
    Response from /reveal endpoint.
    
    NOTE (Phase 4): TEE-based trust model
    - Event is buffered in TEE (hardware-protected memory)
    - Will be included in next hourly Arweave checkpoint (signed by TEE)
    - Verify gateway code integrity: GET /attest
    """
    status: str
    evidence_id: str
    epoch_id: int
    decision: str
    rep_score: float
    rejection_reason: Optional[str] = None
    timestamp: Optional[str] = None  # ISO 8601 timestamp when reveal was accepted
    message: Optional[str] = None


class ManifestResponse(BaseModel):
    """Response from /manifest endpoint"""
    
    status: str
    epoch_id: int
    manifest_root: str

