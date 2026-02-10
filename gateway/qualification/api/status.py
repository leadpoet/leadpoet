"""
Qualification System: Status & Score Endpoints

Phase 2.3 from tasks10.md

This module provides endpoints for miners to check their model status
and retrieve evaluation scores (miner receipts with PII redacted).

CRITICAL: These are NEW endpoints. Do NOT modify any existing status
endpoints in gateway/api/. Keep the qualification API completely separate.
"""

import logging
from uuid import UUID
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from gateway.qualification.models import (
    ModelStatus,
    ModelStatusResponse,
    ModelScoreResponse,
    EvaluationResult,
    ICPPrompt,
)

logger = logging.getLogger(__name__)

# Router for status endpoints (no prefix - parent router adds /qualification)
router = APIRouter()


# =============================================================================
# Response Models
# =============================================================================

class ModelDetailResponse(BaseModel):
    """Detailed model information response."""
    model_id: UUID
    miner_hotkey: str
    repo_url: str
    commit_hash: str
    status: ModelStatus
    created_epoch: int
    created_at: datetime
    version_num: int


class LeaderboardEntry(BaseModel):
    """Single entry on the leaderboard."""
    rank: int
    model_id: UUID
    miner_hotkey: str
    repo_url: str
    status: ModelStatus
    total_score: float
    screening_1_score: Optional[float] = None
    screening_2_score: Optional[float] = None
    final_benchmark_score: Optional[float] = None
    is_current_champion: bool = False
    created_at: datetime


class LeaderboardResponse(BaseModel):
    """Leaderboard response."""
    set_id: int
    entries: List[LeaderboardEntry]
    total_models: int


# =============================================================================
# Internal Data Models (for database records)
# =============================================================================

class ModelRecord:
    """Internal model record from database."""
    def __init__(self, data: Dict[str, Any]):
        self.model_id = UUID(data["model_id"]) if isinstance(data["model_id"], str) else data["model_id"]
        self.miner_hotkey = data["miner_hotkey"]
        self.repo_url = data.get("repo_url", "")
        self.commit_hash = data.get("commit_hash", "")
        self.version_num = data.get("version_num", 1)
        self.status = data["status"]
        self.code_s3_path = data.get("code_s3_path", "")
        self.code_hash = data.get("code_hash", "")
        self.created_epoch = data.get("created_epoch", 0)
        self.created_at = data.get("created_at")
        self.ip_address = data.get("ip_address")


class EvaluationRunRecord:
    """Internal evaluation run record from database."""
    def __init__(self, data: Dict[str, Any]):
        self.evaluation_run_id = data.get("evaluation_run_id")
        self.evaluation_id = data.get("evaluation_id")
        self.probe_id = data["probe_id"]
        self.stage = data.get("stage")
        self.status = data.get("status")
        self.lead_returned = data.get("lead_returned")
        self.lead_score = data.get("lead_score")
        self.icp_fit_score = data.get("icp_fit_score")
        self.decision_maker_score = data.get("decision_maker_score")
        self.intent_signal_score = data.get("intent_signal_score")
        self.cost_penalty = data.get("cost_penalty")
        self.time_penalty = data.get("time_penalty")
        self.final_lead_score = data.get("final_lead_score")
        self.run_cost_usd = data.get("run_cost_usd")
        self.run_time_seconds = data.get("run_time_seconds")
        self.error_code = data.get("error_code")
        self.error_message = data.get("error_message")
        self.created_at = data.get("created_at")


class ModelScoresRecord:
    """Internal model scores record from database."""
    def __init__(self, data: Dict[str, Any]):
        self.model_id = data.get("model_id")
        self.miner_hotkey = data.get("miner_hotkey")
        self.set_id = data.get("set_id")
        self.screening_1_score = data.get("screening_1_score")
        self.screening_2_score = data.get("screening_2_score")
        self.final_benchmark_score = data.get("final_benchmark_score")
        self.total_score = data.get("total_score", 0)
        self.total_cost_usd = data.get("total_cost_usd")
        self.leads_evaluated = data.get("leads_evaluated")
        self.created_epoch = data.get("created_epoch")
        self.created_at = data.get("created_at")


# =============================================================================
# Status Endpoints
# =============================================================================

@router.get("/model/status/{model_id}", response_model=ModelStatusResponse)
async def get_model_status(model_id: UUID):
    """
    Get current status of a submitted model.
    
    Args:
        model_id: UUID of the model to check
    
    Returns:
        ModelStatusResponse with model_id, status, created_at, created_epoch
    
    Raises:
        404: Model not found
    """
    model = await get_model(model_id)
    
    if not model:
        logger.warning(f"Model not found: {model_id}")
        raise HTTPException(
            status_code=404,
            detail="Model not found"
        )
    
    return ModelStatusResponse(
        model_id=model.model_id,
        status=ModelStatus(model.status),
        created_at=model.created_at,
        created_epoch=model.created_epoch,
        repo_url=model.repo_url,
        commit_hash=model.commit_hash
    )


@router.get("/model/detail/{model_id}", response_model=ModelDetailResponse)
async def get_model_detail(model_id: UUID):
    """
    Get detailed information about a submitted model.
    
    Args:
        model_id: UUID of the model
    
    Returns:
        ModelDetailResponse with full model information
    
    Raises:
        404: Model not found
    """
    model = await get_model(model_id)
    
    if not model:
        logger.warning(f"Model not found: {model_id}")
        raise HTTPException(
            status_code=404,
            detail="Model not found"
        )
    
    return ModelDetailResponse(
        model_id=model.model_id,
        miner_hotkey=model.miner_hotkey,
        repo_url=model.repo_url,
        commit_hash=model.commit_hash,
        status=ModelStatus(model.status),
        created_epoch=model.created_epoch,
        created_at=model.created_at,
        version_num=model.version_num
    )


# =============================================================================
# Score Endpoints
# =============================================================================

@router.get("/model/score/{model_id}", response_model=ModelScoreResponse)
async def get_model_score(model_id: UUID):
    """
    Get scores for a completed model (miner receipt).
    
    This endpoint returns the full evaluation results with PII redacted.
    Miners use this to see how their model performed on each ICP.
    
    Args:
        model_id: UUID of the model to get scores for
    
    Returns:
        ModelScoreResponse with total scores and per-ICP results (PII redacted)
    
    Raises:
        404: Model not found
        400: Evaluation not complete (model still in progress)
    """
    model = await get_model(model_id)
    
    if not model:
        logger.warning(f"Model not found: {model_id}")
        raise HTTPException(
            status_code=404,
            detail="Model not found"
        )
    
    # Check if evaluation is complete
    completed_statuses = [
        ModelStatus.FINISHED.value,
        ModelStatus.FAILED_SCREENING_1.value,
        ModelStatus.FAILED_SCREENING_2.value,
    ]
    
    if model.status not in completed_statuses:
        raise HTTPException(
            status_code=400,
            detail=f"Evaluation not complete (current status: {model.status})"
        )
    
    # Get evaluation runs
    runs = await get_evaluation_runs(model_id)
    
    # Build results with PII redacted
    results = []
    for run in runs:
        lead = run.lead_returned
        
        if lead:
            # Redact PII fields
            lead = redact_pii(lead)
        
        # Get ICP prompt for this run
        icp_prompt = await get_icp(run.probe_id)
        
        results.append(EvaluationResult(
            icp_id=run.probe_id,
            icp_prompt=icp_prompt,
            lead_returned=lead,
            icp_fit_score=run.icp_fit_score or 0.0,
            decision_maker_score=run.decision_maker_score or 0.0,
            intent_signal_score=run.intent_signal_score or 0.0,
            final_lead_score=run.final_lead_score or 0.0,
            run_cost_usd=run.run_cost_usd or 0.0,
            run_time_seconds=run.run_time_seconds or 0.0,
            error_code=run.error_code,
            error_message=run.error_message,
            failure_reason=run.lead_score.get("failure_reason") if run.lead_score else None
        ))
    
    # Get aggregated scores
    scores = await get_model_scores(model_id)
    
    return ModelScoreResponse(
        model_id=model_id,
        status=ModelStatus(model.status),
        total_score=scores.total_score if scores else 0.0,
        screening_1_score=scores.screening_1_score if scores else None,
        screening_2_score=scores.screening_2_score if scores else None,
        final_benchmark_score=scores.final_benchmark_score if scores else None,
        total_cost_usd=scores.total_cost_usd if scores else None,
        results=results
    )


# =============================================================================
# Leaderboard Endpoints
# =============================================================================

@router.get("/leaderboard", response_model=LeaderboardResponse)
async def get_leaderboard(set_id: Optional[int] = None, limit: int = 100):
    """
    Get the qualification leaderboard.
    
    Args:
        set_id: Optional evaluation set ID (defaults to current set)
        limit: Maximum number of entries to return (default 100)
    
    Returns:
        LeaderboardResponse with ranked models
    """
    # Get current set ID if not specified
    if set_id is None:
        set_id = await get_current_evaluation_set_id()
    
    # Get leaderboard entries
    entries = await get_leaderboard_entries(set_id, limit)
    
    return LeaderboardResponse(
        set_id=set_id,
        entries=entries,
        total_models=len(entries)
    )


@router.get("/champion")
async def get_current_champion():
    """
    Get information about the current champion model.
    
    Returns:
        Champion model information or null if no champion
    """
    champion = await get_champion()
    
    if not champion:
        return {
            "champion": None,
            "message": "No current champion"
        }
    
    return {
        "champion": champion,
        "message": "Current champion"
    }


# =============================================================================
# PII Redaction
# =============================================================================

# Fields to redact from lead data
PII_FIELDS = [
    "full_name",
    "first_name", 
    "last_name",
    "email",
    "linkedin_url",
    "phone",
]

REDACTED_VALUE = "[REDACTED]"


def redact_pii(lead: Dict[str, Any]) -> Dict[str, Any]:
    """
    Redact PII fields from a lead dictionary.
    
    Args:
        lead: Lead data dictionary
    
    Returns:
        Lead data with PII fields redacted
    """
    if not lead:
        return lead
    
    # Create a copy to avoid modifying the original
    redacted = lead.copy()
    
    for field in PII_FIELDS:
        if field in redacted and redacted[field]:
            redacted[field] = REDACTED_VALUE
    
    return redacted


# =============================================================================
# Placeholder Functions (to be implemented)
# =============================================================================

async def get_model(model_id: UUID) -> Optional[ModelRecord]:
    """
    Get a model record from the database.
    
    Args:
        model_id: UUID of the model
    
    Returns:
        ModelRecord or None if not found
    
    TODO: Implement database query
    """
    # Placeholder - in production, query qualification_models table
    # 
    # from gateway.db.client import get_supabase_client
    # supabase = await get_supabase_client()
    # 
    # response = await supabase.table("qualification_models") \
    #     .select("*") \
    #     .eq("model_id", str(model_id)) \
    #     .single() \
    #     .execute()
    # 
    # if response.data:
    #     return ModelRecord(response.data)
    # return None
    
    logger.warning("PLACEHOLDER: get_model - implement database query")
    
    # Return None to simulate "not found" for now
    # In production, this would query the database
    return None


async def get_evaluation_runs(model_id: UUID) -> List[EvaluationRunRecord]:
    """
    Get all evaluation runs for a model.
    
    Args:
        model_id: UUID of the model
    
    Returns:
        List of EvaluationRunRecord objects
    
    TODO: Implement database query
    """
    # Placeholder - in production, query qualification_evaluation_runs table
    # 
    # from gateway.db.client import get_supabase_client
    # supabase = await get_supabase_client()
    # 
    # # First get evaluation IDs for this model
    # evals_response = await supabase.table("qualification_evaluations") \
    #     .select("evaluation_id") \
    #     .eq("model_id", str(model_id)) \
    #     .execute()
    # 
    # eval_ids = [e["evaluation_id"] for e in evals_response.data]
    # 
    # # Then get all runs for those evaluations
    # runs_response = await supabase.table("qualification_evaluation_runs") \
    #     .select("*") \
    #     .in_("evaluation_id", eval_ids) \
    #     .order("created_at") \
    #     .execute()
    # 
    # return [EvaluationRunRecord(r) for r in runs_response.data]
    
    logger.warning("PLACEHOLDER: get_evaluation_runs - implement database query")
    return []


async def get_icp(icp_id: str) -> Optional[ICPPrompt]:
    """
    Get an ICP prompt by ID.
    
    Args:
        icp_id: The ICP identifier
    
    Returns:
        ICPPrompt or None if not found
    
    TODO: Implement ICP lookup (from JSON files or database)
    """
    # Placeholder - in production, load from ICP dataset
    # 
    # ICPs could be stored in:
    # 1. JSON files in qualification/data/icp_sets/
    # 2. Database table qualification_evaluation_sets with icp_ids JSONB
    # 
    # import json
    # with open(f"qualification/data/icp_sets/current.json") as f:
    #     data = json.load(f)
    #     for icp in data["icps"]:
    #         if icp["icp_id"] == icp_id:
    #             return ICPPrompt(**icp)
    # return None
    
    logger.warning("PLACEHOLDER: get_icp - implement ICP lookup")
    
    # Return a placeholder ICP
    return ICPPrompt(
        icp_id=icp_id,
        industry="Technology",
        sub_industry="SaaS",
        target_role="VP of Sales",
        target_seniority="VP",
        employee_count="50-200 employees",
        company_stage="Series B",
        geography="United States",
        product_service="CRM software",
        additional_context=None,
        created_at=None
    )


async def get_model_scores(model_id: UUID) -> Optional[ModelScoresRecord]:
    """
    Get aggregated scores for a model.
    
    Args:
        model_id: UUID of the model
    
    Returns:
        ModelScoresRecord or None if not found
    
    TODO: Implement database query
    """
    # Placeholder - in production, query qualification_model_scores table
    # 
    # from gateway.db.client import get_supabase_client
    # supabase = await get_supabase_client()
    # 
    # response = await supabase.table("qualification_model_scores") \
    #     .select("*") \
    #     .eq("model_id", str(model_id)) \
    #     .single() \
    #     .execute()
    # 
    # if response.data:
    #     return ModelScoresRecord(response.data)
    # return None
    
    logger.warning("PLACEHOLDER: get_model_scores - implement database query")
    return None


async def get_current_evaluation_set_id() -> int:
    """
    Get the current evaluation set ID.
    
    Returns:
        Current set ID based on current epoch
    """
    from gateway.qualification.config import CONFIG
    
    # Calculate set ID from current epoch
    current_epoch = await get_current_bittensor_epoch()
    return current_epoch // CONFIG.EVALUATION_SET_ROTATION_EPOCHS


async def get_current_bittensor_epoch() -> int:
    """
    Get the current Bittensor epoch.
    
    Returns:
        Current epoch number
    """
    # Import from submit.py to avoid duplication
    from gateway.qualification.api.submit import get_current_bittensor_epoch as _get_epoch
    return await _get_epoch()


async def get_leaderboard_entries(set_id: int, limit: int) -> List[LeaderboardEntry]:
    """
    Get leaderboard entries for a given evaluation set.
    
    Args:
        set_id: Evaluation set ID
        limit: Maximum entries to return
    
    Returns:
        List of LeaderboardEntry objects, sorted by rank
    
    TODO: Implement database query using qualification_leaderboard view
    """
    # Placeholder - in production, query qualification_leaderboard view
    # 
    # from gateway.db.client import get_supabase_client
    # supabase = await get_supabase_client()
    # 
    # response = await supabase.table("qualification_leaderboard") \
    #     .select("*") \
    #     .eq("set_id", set_id) \
    #     .order("rank") \
    #     .limit(limit) \
    #     .execute()
    # 
    # return [
    #     LeaderboardEntry(
    #         rank=entry["rank"],
    #         model_id=UUID(entry["model_id"]),
    #         miner_hotkey=entry["miner_hotkey"],
    #         repo_url=entry["repo_url"],
    #         status=ModelStatus(entry["status"]),
    #         total_score=entry["total_score"],
    #         screening_1_score=entry.get("screening_1_score"),
    #         screening_2_score=entry.get("screening_2_score"),
    #         final_benchmark_score=entry.get("final_benchmark_score"),
    #         is_current_champion=entry.get("is_current_champion", False),
    #         created_at=entry["created_at"]
    #     )
    #     for entry in response.data
    # ]
    
    logger.warning("PLACEHOLDER: get_leaderboard_entries - implement database query")
    return []


async def get_champion() -> Optional[Dict[str, Any]]:
    """
    Get the current champion model.
    
    Returns:
        Champion information dict or None if no champion
    
    TODO: Implement database query using qualification_current_champion view
    """
    # Placeholder - in production, query qualification_current_champion view
    # 
    # from gateway.db.client import get_supabase_client
    # supabase = await get_supabase_client()
    # 
    # response = await supabase.table("qualification_current_champion") \
    #     .select("*") \
    #     .single() \
    #     .execute()
    # 
    # return response.data if response.data else None
    
    logger.warning("PLACEHOLDER: get_champion - implement database query")
    return None
