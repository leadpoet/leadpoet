"""
Gateway Response Models
======================

Pydantic models for API responses.
"""

from pydantic import BaseModel
from typing import List, Optional, Dict, Any


class PresignedURLResponse(BaseModel):
    """Response from /presign endpoint"""
    
    lead_id: str
    presigned_url: str  # S3 URL for upload (miner uploads here)
    s3_url: str  # Alias for backward compatibility
    expires_in: int
    # Note: MinIO mirroring happens gateway-side after S3 upload verification


class SubmissionResponse(BaseModel):
    """Response from final submission confirmation"""
    
    status: str
    lead_id: str
    merkle_proof: List[str]
    checkpoint_root: str


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
    """Response from /validate endpoint"""
    
    status: str
    merkle_proof: List[str]


class ManifestResponse(BaseModel):
    """Response from /manifest endpoint"""
    
    status: str
    epoch_id: int
    manifest_root: str

