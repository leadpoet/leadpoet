"""
Qualification System: Work Distribution Endpoints

These endpoints are called BY validators to:
1. Register for work
2. Send heartbeats
3. Request evaluation work
4. Report results

The gateway distributes work, validators execute.

CRITICAL: This is a NEW API. Do NOT modify any existing gateway APIs.
"""

import os
import json
import base64
import logging
from uuid import UUID, uuid4
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Any, List

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from gateway.qualification.config import CONFIG
from gateway.qualification.models import (
    ModelStatus,
    EvaluationStatus,
    EvaluationRunStatus,
    ICPPrompt,
)

logger = logging.getLogger(__name__)

# =============================================================================
# Champion Selection Constants
# =============================================================================
# These can be adjusted without code changes - just update the values here

# Import champion threshold from central config (single source of truth)
from gateway.qualification.config import CONFIG
CHAMPION_BEAT_THRESHOLD = CONFIG.CHAMPION_DETHRONING_THRESHOLD_PCT  # Currently 2%

router = APIRouter(prefix="/validator", tags=["validator-work"])


# =============================================================================
# Request/Response Models
# =============================================================================

class ValidatorRegisterRequest(BaseModel):
    """Request to register a validator for work."""
    timestamp: int = Field(..., description="Current unix timestamp")
    signed_timestamp: str = Field(..., description="Timestamp signed by hotkey")
    hotkey: str = Field(..., description="Validator's Bittensor hotkey")
    commit_hash: str = Field(..., description="Validator code version")


class ValidatorRegisterResponse(BaseModel):
    """Response from validator registration."""
    session_id: str = Field(..., description="Session UUID for this validator")
    message: str = Field(default="Registered successfully")


class HeartbeatRequest(BaseModel):
    """Validator heartbeat request."""
    session_id: str = Field(..., description="Session UUID")
    timestamp: int = Field(..., description="Current unix timestamp")
    system_metrics: Optional[Dict[str, Any]] = Field(default=None)
    current_evaluation: Optional[str] = Field(default=None)


class RequestEvaluationRequest(BaseModel):
    """Request for evaluation work."""
    session_id: str = Field(..., description="Session UUID")


class EvaluationRunWork(BaseModel):
    """Single evaluation run (one ICP)."""
    evaluation_run_id: str
    probe_id: str
    probe_name: Optional[str] = None
    icp_data: Dict[str, Any]
    stage: str = "final"


class EvaluationWork(BaseModel):
    """Work assignment for a validator."""
    evaluation_id: str
    model_id: str
    agent_code: str  # Base64-encoded model code
    evaluation_runs: List[EvaluationRunWork]


class EvaluationWorkResponse(BaseModel):
    """Response with evaluation work (or empty if no work)."""
    has_work: bool = False
    evaluation_id: Optional[str] = None
    model_id: Optional[str] = None
    model_name: Optional[str] = None  # For display in validator logs
    miner_hotkey: Optional[str] = None  # For display in validator logs
    agent_code: Optional[str] = None  # Base64-encoded
    evaluation_runs: List[EvaluationRunWork] = Field(default_factory=list)
    icp_set_hash: Optional[str] = None  # Hash of ICP set for verifiability logging


class ReportResultsRequest(BaseModel):
    """Report results for an evaluation run."""
    evaluation_run_id: str
    lead_returned: Optional[Dict[str, Any]] = None
    lead_score: Optional[Dict[str, Any]] = None
    icp_fit_score: float = 0
    decision_maker_score: float = 0
    intent_signal_score: float = 0
    cost_penalty: float = 0
    time_penalty: float = 0
    final_lead_score: float = 0
    run_cost_usd: float = 0
    run_time_seconds: float = 0
    status: str = "finished"


class ReportErrorRequest(BaseModel):
    """Report an error for an evaluation run."""
    evaluation_run_id: str
    error_code: int
    error_message: str
    status: str = "error"


class UpdateRunStatusRequest(BaseModel):
    """Update status of an evaluation run."""
    evaluation_run_id: str
    status: str
    timestamp: int


# =============================================================================
# In-Memory State (for development - use DB in production)
# =============================================================================

# Active validator sessions
_validator_sessions: Dict[str, Dict[str, Any]] = {}

# Work queue: models waiting for evaluation
_work_queue: List[Dict[str, Any]] = []

# Assigned work: evaluation_id -> session_id
_assigned_work: Dict[str, str] = {}

# Assigned work items: evaluation_id -> full work item (STORES THE FULL WORK ITEM!)
# This is needed so we can find the work item when results come in
_assigned_work_items: Dict[str, Dict[str, Any]] = {}

# Models store: model_id -> model info
_models_store: Dict[str, Dict[str, Any]] = {}

# Evaluation results: evaluation_run_id -> results
_evaluation_results: Dict[str, Dict[str, Any]] = {}


# =============================================================================
# Validator Registration
# =============================================================================

@router.post("/register", response_model=ValidatorRegisterResponse)
async def register_validator(request: ValidatorRegisterRequest):
    """
    Register a validator to receive evaluation work.
    
    Validators must:
    1. Provide their hotkey
    2. Sign a timestamp with their hotkey
    3. Be registered on the subnet
    
    Returns a session_id for subsequent requests.
    """
    # Verify timestamp is recent (within 5 minutes)
    now = int(datetime.now(timezone.utc).timestamp())
    if abs(now - request.timestamp) > 300:
        raise HTTPException(
            status_code=400,
            detail="Timestamp too old or in future"
        )
    
    # Verify hotkey is registered on subnet
    is_registered = await is_validator_registered(request.hotkey)
    if not is_registered:
        raise HTTPException(
            status_code=403,
            detail="Hotkey not registered as validator on subnet"
        )
    
    # Verify signature
    signature_valid = verify_validator_signature(
        request.hotkey,
        request.signed_timestamp,
        request.timestamp
    )
    if not signature_valid:
        logger.warning(f"Invalid signature from validator: {request.hotkey[:16]}...")
        raise HTTPException(
            status_code=401,
            detail="Invalid signature"
        )
    
    # Create session
    session_id = str(uuid4())
    _validator_sessions[session_id] = {
        "hotkey": request.hotkey,
        "commit_hash": request.commit_hash,
        "registered_at": datetime.now(timezone.utc).isoformat(),
        "last_heartbeat": datetime.now(timezone.utc).isoformat(),
        "current_evaluation": None
    }
    
    logger.info(f"Validator registered: hotkey={request.hotkey[:16]}..., session={session_id[:8]}...")
    
    return ValidatorRegisterResponse(
        session_id=session_id,
        message="Registered successfully"
    )


# =============================================================================
# Heartbeat
# =============================================================================

@router.post("/heartbeat")
async def validator_heartbeat(request: HeartbeatRequest):
    """
    Receive heartbeat from a validator.
    
    Updates last_seen timestamp and system metrics.
    Returns current status and any pending commands.
    """
    session = _validator_sessions.get(request.session_id)
    if not session:
        raise HTTPException(
            status_code=404,
            detail="Session not found - please re-register"
        )
    
    # Update session
    session["last_heartbeat"] = datetime.now(timezone.utc).isoformat()
    session["system_metrics"] = request.system_metrics
    session["current_evaluation"] = request.current_evaluation
    
    logger.debug(
        f"Heartbeat from session={request.session_id[:8]}..., "
        f"evaluation={request.current_evaluation}"
    )
    
    return {
        "status": "ok",
        "server_time": int(datetime.now(timezone.utc).timestamp())
    }


# =============================================================================
# Request Evaluation Work
# =============================================================================

@router.post("/request-evaluation", response_model=EvaluationWorkResponse)
async def request_evaluation(request: RequestEvaluationRequest):
    """
    Request evaluation work from the platform.
    
    Returns the next model to evaluate with:
    - Model code (base64 encoded)
    - List of ICP evaluation runs
    
    If no work available, returns has_work=False.
    """
    session = _validator_sessions.get(request.session_id)
    if not session:
        raise HTTPException(
            status_code=404,
            detail="Session not found - please re-register"
        )
    
    # NOTE: We no longer block if validator has existing work assigned.
    # In distributed mode, the coordinator may have batch work (for workers)
    # AND need to request rebenchmark work for itself.
    # The old check blocked rebenchmarks because batch work was assigned first.
    
    # Try to get work from queue
    work = await get_next_work()
    
    if not work:
        return EvaluationWorkResponse(has_work=False)
    
    # Assign work to this validator
    _assigned_work[work["evaluation_id"]] = request.session_id
    _assigned_work_items[work["evaluation_id"]] = work  # Store full work item for result tracking!
    session["current_evaluation"] = work["evaluation_id"]
    
    # Get ICP set hash for verifiability logging
    icp_set_hash = work.get("icp_set_hash", "")
    
    logger.info(
        f"Assigned work to session={request.session_id[:8]}...: "
        f"evaluation={work['evaluation_id'][:8]}..., "
        f"runs={len(work['evaluation_runs'])}, "
        f"icp_hash={icp_set_hash[:16] if icp_set_hash else 'N/A'}..."
    )
    
    return EvaluationWorkResponse(
        has_work=True,
        evaluation_id=work["evaluation_id"],
        model_id=work["model_id"],
        model_name=work.get("model_name", "Unnamed Model"),
        miner_hotkey=work.get("miner_hotkey", "Unknown"),
        agent_code=work["agent_code"],
        evaluation_runs=[
            EvaluationRunWork(**run) for run in work["evaluation_runs"]
        ],
        icp_set_hash=icp_set_hash
    )


# =============================================================================
# Request Batch Evaluation Work (for Coordinator)
# =============================================================================

class BatchEvaluationRequest(BaseModel):
    """Request to get a batch of models for evaluation."""
    session_id: str = Field(..., description="Validator session ID")
    max_models: int = Field(default=None, description="Max models to return (default: CONFIG.MAX_MODELS_PER_EPOCH)")
    epoch: int = Field(default=None, description="Current epoch (for logging)")


class BatchEvaluationResponse(BaseModel):
    """Response containing multiple models for evaluation."""
    has_work: bool = Field(default=False, description="Whether there is work available")
    models: List[Dict[str, Any]] = Field(default_factory=list, description="List of model work items")
    queue_depth: int = Field(default=0, description="Total models remaining in queue after this batch")


@router.post("/request-batch-evaluation", response_model=BatchEvaluationResponse)
async def request_batch_evaluation(request: BatchEvaluationRequest):
    """
    Request a batch of models for evaluation (used by coordinator).
    
    This endpoint returns up to MAX_MODELS_PER_EPOCH models for evaluation
    in a single request. Models are returned FIFO (oldest created_at first).
    
    The coordinator uses this to:
    1. Pull all pending models for the epoch at once
    2. Distribute to workers based on container_id
    3. Handle rebenchmark separately on coordinator
    
    Returns:
        BatchEvaluationResponse with list of work items
    """
    session = _validator_sessions.get(request.session_id)
    if not session:
        raise HTTPException(
            status_code=404,
            detail="Session not found - please re-register"
        )
    
    # Determine max models to return
    max_models = request.max_models if request.max_models else CONFIG.MAX_MODELS_PER_EPOCH
    
    logger.info(
        f"📋 Batch evaluation request: session={request.session_id[:8]}..., "
        f"max_models={max_models}, epoch={request.epoch or 'N/A'}"
    )
    
    # Get pending models from database (FIFO order)
    work_items = await get_pending_models_from_db(limit=max_models)
    
    if not work_items:
        logger.debug("No pending models in queue")
        return BatchEvaluationResponse(has_work=False, models=[], queue_depth=0)
    
    # Track assigned work
    for work in work_items:
        eval_id = work["evaluation_id"]
        _assigned_work[eval_id] = request.session_id
        _assigned_work_items[eval_id] = work
    
    # Get remaining queue depth
    try:
        from gateway.db.client import get_read_client
        supabase = get_read_client()
        count_result = supabase.table("qualification_models").select(
            "id", count="exact"
        ).is_(
            "evaluated_at", "null"
        ).eq(
            "status", "submitted"
        ).execute()
        total_pending = count_result.count if count_result.count else 0
        queue_depth = max(0, total_pending - len(work_items))
    except Exception as e:
        logger.warning(f"Could not get queue depth: {e}")
        queue_depth = 0
    
    logger.info(
        f"✅ Returning {len(work_items)} models for evaluation "
        f"(queue_depth={queue_depth} remaining)"
    )
    
    return BatchEvaluationResponse(
        has_work=True,
        models=work_items,
        queue_depth=queue_depth
    )


# =============================================================================
# Report Results
# =============================================================================

@router.post("/report-results")
async def report_results(request: ReportResultsRequest):
    """
    Report results for a completed evaluation run.
    
    Stores in gateway memory (and eventually Supabase):
    - Lead data (if returned)
    - Score breakdown
    - Cost and timing metrics
    
    NOTE: Champion determination is done LOCALLY by the validator, not here.
    Validator calls /champion-status separately to notify gateway of champion changes.
    """
    # Find the session that owns this run
    evaluation_run_id = request.evaluation_run_id
    
    # Store results (in production, update database)
    logger.info(
        f"Results received: run={evaluation_run_id[:8]}..., "
        f"score={request.final_lead_score:.2f}, "
        f"cost=${request.run_cost_usd:.4f}, "
        f"time={request.run_time_seconds:.2f}s"
    )
    
    # Update evaluation run in database
    await store_evaluation_run_result(
        evaluation_run_id=evaluation_run_id,
        lead_returned=request.lead_returned,
        lead_score=request.lead_score,
        icp_fit_score=request.icp_fit_score,
        decision_maker_score=request.decision_maker_score,
        intent_signal_score=request.intent_signal_score,
        cost_penalty=request.cost_penalty,
        time_penalty=request.time_penalty,
        final_lead_score=request.final_lead_score,
        run_cost_usd=request.run_cost_usd,
        run_time_seconds=request.run_time_seconds,
        status=request.status
    )
    
    # Gateway just stores results - champion determination is done by validator locally
    return {"status": "ok", "message": "Results recorded"}


# =============================================================================
# Champion Status (Validator → Gateway, one-way notification)
# =============================================================================

class ChampionStatusRequest(BaseModel):
    """Request from validator notifying gateway of champion status."""
    model_id: str
    became_champion: bool
    score: float
    determined_by: str = "validator"  # Always "validator" - validator makes the decision
    is_rebenchmark: bool = False  # True if this is a rebenchmark of the existing champion
    # Optional fields for full DB update (cost, time, code)
    evaluation_cost_usd: Optional[float] = None
    evaluation_time_seconds: Optional[int] = None
    code_content: Optional[str] = None  # JSON string of code files


class RebenchmarkRequest(BaseModel):
    """Request from validator to rebenchmark the current champion."""
    model_id: str
    session_id: str


@router.post("/request-rebenchmark")
async def request_rebenchmark(request: RebenchmarkRequest):
    """
    Request rebenchmarking of the current champion model.
    
    Champions are re-evaluated every 30 epochs (~36 hours) to ensure
    they still perform well on new ICP sets.
    
    The gateway:
    1. Looks up the model in Supabase
    2. Re-queues it for evaluation
    3. Returns success/failure
    
    The validator will receive the work via the normal /request-evaluation flow.
    """
    logger.info(f"🔄 Rebenchmark request: model={request.model_id[:8]}...")
    
    try:
        # Validate session
        session = _validator_sessions.get(request.session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Look up model in Supabase
        from gateway.db.client import get_write_client
        supabase = get_write_client()
        
        result = supabase.table("qualification_models").select(
            "id, s3_path, code_hash, miner_hotkey, model_name"
        ).eq("id", request.model_id).execute()
        
        if not result.data:
            raise HTTPException(status_code=404, detail="Model not found")
        
        model = result.data[0]
        
        # Extract s3_key from s3_path (s3://bucket/key -> key)
        s3_path = model.get("s3_path", "")
        if s3_path.startswith("s3://"):
            # Format: s3://bucket/key -> extract key
            s3_key = "/".join(s3_path.split("/")[3:])
        else:
            s3_key = s3_path
        
        logger.info(f"Rebenchmark: s3_path={s3_path}, extracted s3_key={s3_key}")
        
        # Check if already in queue
        for work_item in _work_queue:
            if work_item.get("model_id") == request.model_id:
                logger.info(f"Model {request.model_id[:8]}... already in queue, skipping rebenchmark")
                return {
                    "status": "already_queued",
                    "message": "Model is already queued for evaluation"
                }
        
        # Queue for re-evaluation
        from gateway.qualification.api.work import queue_model_for_evaluation
        
        eval_result = await queue_model_for_evaluation(
            model_id=model["id"],
            s3_key=s3_key,
            code_hash=model["code_hash"],
            miner_hotkey=model["miner_hotkey"],
            model_name=model.get("model_name", "Champion Rebenchmark"),
            stage="rebenchmark"
        )
        
        logger.info(f"✅ Champion rebenchmark queued: {eval_result}")
        
        return {
            "status": "ok",
            "message": "Champion queued for rebenchmark",
            "evaluation_id": eval_result.get("evaluation_id")
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to queue rebenchmark: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/champion-status")
async def receive_champion_status(request: ChampionStatusRequest):
    """
    Receive champion status notification from validator.
    
    The VALIDATOR determines who is champion locally (based on its JSON file).
    It then notifies the gateway so we can update Supabase for:
    - Leaderboard display
    - Auditing/transparency
    - Historical records
    
    This is ONE-WAY: validator tells gateway, gateway does NOT send anything back.
    The validator's local JSON is the source of truth for weight submission.
    """
    logger.info(
        f"📥 Champion status from validator: model={request.model_id[:8]}..., "
        f"became_champion={request.became_champion}, score={request.score:.2f}"
    )
    
    try:
        from gateway.db.client import get_write_client
        supabase = get_write_client()
        
        # Build base update data (always include these if provided)
        base_update = {
            "score": request.score,
            "status": "evaluated",
            "evaluated_at": datetime.now(timezone.utc).isoformat()
        }
        
        # Add optional fields if provided (for full DB update)
        if request.evaluation_cost_usd is not None:
            base_update["evaluation_cost_usd"] = request.evaluation_cost_usd
        if request.evaluation_time_seconds is not None:
            base_update["evaluation_time_seconds"] = request.evaluation_time_seconds
        if request.code_content is not None:
            base_update["code_content"] = request.code_content
        
        if request.became_champion:
            # Dethrone any existing champion
            supabase.table("qualification_models").update({
                "is_champion": False,
                "dethroned_at": datetime.now(timezone.utc).isoformat()
            }).eq("is_champion", True).execute()
            
            # Crown new champion AND mark as evaluated (merge with base update)
            champion_update = {
                **base_update,
                "is_champion": True,
                "champion_at": datetime.now(timezone.utc).isoformat()
            }
            supabase.table("qualification_models").update(champion_update).eq("id", request.model_id).execute()
            
            logger.info(f"👑 Updated Supabase: {request.model_id[:8]}... is now champion (score: {request.score:.2f}, status=evaluated)")
        
        elif request.is_rebenchmark:
            # Champion was re-evaluated - update their score (even if lower)
            # Rebenchmark uses base_update which includes all fields
            supabase.table("qualification_models").update(base_update).eq("id", request.model_id).execute()
            
            logger.info(f"🔄 Updated Supabase: Champion {request.model_id[:8]}... rebenchmarked (new score: {request.score:.2f})")
        
        else:
            # Model was evaluated but didn't become champion - still mark as evaluated
            supabase.table("qualification_models").update(base_update).eq("id", request.model_id).execute()
            
            logger.info(f"📊 Model {request.model_id[:8]}... evaluated (score: {request.score:.2f}) but did not become champion")
        
        return {"status": "ok", "message": "Champion status recorded"}
        
    except Exception as e:
        logger.error(f"Failed to update champion status in Supabase: {e}")
        # Still return OK - validator's local JSON is source of truth
        return {"status": "ok", "message": "Champion status acknowledged (Supabase update failed)"}


# =============================================================================
# Report Error
# =============================================================================

@router.post("/report-error")
async def report_error(request: ReportErrorRequest):
    """
    Report an error for a failed evaluation run.
    
    Records:
    - Error code
    - Error message
    - Failed status
    """
    logger.error(
        f"Error reported: run={request.evaluation_run_id[:8]}..., "
        f"code={request.error_code}, message={request.error_message}"
    )
    
    # Update evaluation run in database
    await store_evaluation_run_error(
        evaluation_run_id=request.evaluation_run_id,
        error_code=request.error_code,
        error_message=request.error_message
    )
    
    return {"status": "ok", "message": "Error recorded"}


# =============================================================================
# Update Run Status
# =============================================================================

@router.post("/update-run-status")
async def update_run_status(request: UpdateRunStatusRequest):
    """
    Update the status of an evaluation run.
    
    Used for progress tracking (initializing_model, running_model, etc.)
    """
    logger.debug(
        f"Status update: run={request.evaluation_run_id[:8]}..., "
        f"status={request.status}"
    )
    
    # Update in database (placeholder)
    # await update_evaluation_run_status(request.evaluation_run_id, request.status)
    
    return {"status": "ok"}


# =============================================================================
# Debug/Status Endpoints (helpful for testing)
# =============================================================================

@router.get("/debug/queue-status")
async def get_queue_status():
    """
    Get the current status of the work queue and evaluations.
    
    Useful for debugging during testnet development.
    """
    return {
        "work_queue_length": len(_work_queue),
        "work_queue": [
            {
                "evaluation_id": w.get("evaluation_id", "")[:16] + "...",
                "model_id": w.get("model_id", "")[:16] + "...",
                "miner_hotkey": w.get("miner_hotkey", "")[:16] + "...",
                "model_name": w.get("model_name"),
                "runs_count": len(w.get("evaluation_runs", [])),
                "queued_at": w.get("queued_at"),
                "has_code": len(w.get("agent_code", "")) > 0
            }
            for w in _work_queue
        ],
        "validator_sessions": len(_validator_sessions),
        "assigned_work": len(_assigned_work),
        "models_store": len(_models_store),
        "evaluation_results": len(_evaluation_results),
        "models": [
            {
                "model_id": m.get("model_id", "")[:16] + "...",
                "miner_hotkey": m.get("miner_hotkey", "")[:16] + "...",
                "model_name": m.get("model_name"),
                "status": m.get("status"),
                "avg_score": m.get("avg_score"),
                "created_at": m.get("created_at")
            }
            for m in _models_store.values()
        ]
    }


@router.get("/debug/model/{model_id}")
async def get_model_status(model_id: str):
    """
    Get detailed status for a specific model.
    """
    model = _models_store.get(model_id)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    
    # Find evaluation results for this model
    evaluation_id = model.get("evaluation_id")
    results = []
    
    for work_item in _work_queue:
        if work_item.get("model_id") == model_id:
            for run in work_item.get("evaluation_runs", []):
                run_id = run.get("evaluation_run_id")
                result = _evaluation_results.get(run_id, {"status": "pending"})
                results.append({
                    "run_id": run_id[:16] + "..." if run_id else None,
                    "icp": run.get("probe_name"),
                    "status": result.get("status"),
                    "score": result.get("final_lead_score")
                })
    
    return {
        "model": model,
        "evaluation_id": evaluation_id,
        "results": results,
        "total_runs": len(results),
        "completed_runs": len([r for r in results if r.get("status") in ["finished", "error"]])
    }


# =============================================================================
# Auto-Queue Function (Called from submit.py after model submission)
# =============================================================================

async def queue_model_for_evaluation(
    model_id: str,
    s3_key: str,
    code_hash: str,
    miner_hotkey: str,
    model_name: Optional[str] = None,
    icp_set_id: int = 1,
    stage: str = "final"
) -> Dict[str, Any]:
    """
    Queue a newly submitted model for evaluation.
    
    This is called automatically from submit.py after a model is submitted.
    It downloads the model from S3 and creates work for validators.
    
    Args:
        model_id: The model UUID
        s3_key: S3 key where model tarball is stored
        code_hash: SHA256 hash of the model code
        miner_hotkey: The miner's Bittensor hotkey
        model_name: Optional display name for the model
        icp_set_id: Which ICP set to use for evaluation (default 1)
        stage: Evaluation stage ("screening" or "final")
    
    Returns:
        Dict with evaluation_id and runs_queued
    """
    logger.info(f"Queueing model for evaluation: model_id={model_id[:8]}..., s3_key={s3_key}")
    
    # Fetch model code from S3
    model_code = await fetch_model_code(s3_key)
    
    if not model_code:
        logger.error(f"Failed to download model from S3: {s3_key}")
        # Still queue it - validator will retry download
        model_code = b""
    
    # Fetch ICP set (returns tuple of (icps, icp_set_hash))
    icp_result = await get_icp_set(icp_set_id)
    if not icp_result or not icp_result[0]:
        logger.warning(f"ICP set {icp_set_id} not found, using fallback")
        icp_result = await get_icp_set(1)  # Fallback to default
    
    icps, icp_set_hash = icp_result
    logger.info(f"Using ICP set with {len(icps)} ICPs, hash={icp_set_hash[:16] if icp_set_hash else 'N/A'}...")
    
    # Create evaluation record
    evaluation_id = str(uuid4())
    
    # Create evaluation runs (one per ICP)
    evaluation_runs = []
    for icp in icps:
        run_id = str(uuid4())
        evaluation_runs.append({
            "evaluation_run_id": run_id,
            "probe_id": icp.get("icp_id", str(uuid4())),
            "probe_name": icp.get("industry", "Unknown"),
            "icp_data": icp,
            "stage": stage
        })
    
    # Add to work queue (including ICP hash for verifiability)
    work_item = {
        "evaluation_id": evaluation_id,
        "model_id": model_id,
        "miner_hotkey": miner_hotkey,
        "model_name": model_name,
        "code_hash": code_hash,
        "s3_key": s3_key,
        "agent_code": base64.b64encode(model_code).decode() if model_code else "",
        "evaluation_runs": evaluation_runs,
        "icp_set_hash": icp_set_hash,  # For verifiability logging
        "queued_at": datetime.now(timezone.utc).isoformat()
    }
    
    # Rebenchmarks get PRIORITY - insert at front of queue
    # This ensures coordinator gets rebenchmark work immediately
    if stage == "rebenchmark":
        _work_queue.insert(0, work_item)
        logger.info(f"🔄 Rebenchmark added to FRONT of queue (priority)")
    else:
        _work_queue.append(work_item)
    
    # Also store in the models dictionary for status lookups
    _models_store[model_id] = {
        "model_id": model_id,
        "miner_hotkey": miner_hotkey,
        "model_name": model_name,
        "code_hash": code_hash,
        "s3_key": s3_key,
        "evaluation_id": evaluation_id,
        "status": "queued",
        "created_at": datetime.now(timezone.utc).isoformat()
    }
    
    logger.info(
        f"✅ Model queued for evaluation: evaluation={evaluation_id[:8]}..., "
        f"model={model_id[:8]}..., runs={len(evaluation_runs)}, "
        f"hotkey={miner_hotkey[:16]}..."
    )
    
    return {
        "evaluation_id": evaluation_id,
        "runs_queued": len(evaluation_runs),
        "message": "Evaluation queued"
    }


# =============================================================================
# Admin: Queue Model for Evaluation (Manual)
# =============================================================================

@router.post("/queue-evaluation")
async def queue_evaluation(
    model_id: str,
    icp_set_id: int = 1,
    stage: str = "final"
):
    """
    Queue a model for evaluation (admin endpoint).
    
    Called when:
    1. Model passes screening
    2. Champion re-benchmark is needed
    3. Manual evaluation is requested
    """
    # Check if model exists in our store
    model = _models_store.get(model_id)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found in store")
    
    # Use the auto-queue function
    result = await queue_model_for_evaluation(
        model_id=model_id,
        s3_key=model.get("s3_key", ""),
        code_hash=model.get("code_hash", ""),
        miner_hotkey=model.get("miner_hotkey", ""),
        model_name=model.get("model_name"),
        icp_set_id=icp_set_id,
        stage=stage
    )
    
    return result


# =============================================================================
# Helper Functions (Using shared gateway utilities)
# =============================================================================

async def is_validator_registered(hotkey: str) -> bool:
    """
    Check if hotkey is registered as a validator on the subnet.
    
    Uses the shared gateway metagraph cache for efficiency.
    """
    try:
        # Import the shared gateway registry (same metagraph cache as sourcing)
        from gateway.utils.registry import is_registered_hotkey_async
        
        is_registered, role = await is_registered_hotkey_async(hotkey)
        
        if not is_registered:
            logger.warning(f"Hotkey {hotkey[:20]}... not registered on subnet")
            return False
        
        if role != "validator":
            logger.warning(f"Hotkey {hotkey[:20]}... is registered as {role}, not validator")
            return False
        
        logger.info(f"✅ Hotkey {hotkey[:20]}... verified as registered validator")
        return True
        
    except Exception as e:
        logger.error(f"Error checking validator registration: {e}")
        return False


def verify_validator_signature(hotkey: str, signature: str, timestamp: int) -> bool:
    """
    Verify the timestamp signature from validator.
    
    Uses the shared gateway signature verification (sr25519/Ed25519).
    """
    try:
        # Import the shared gateway signature utility
        from gateway.utils.signature import verify_wallet_signature
        
        # The message signed by the validator is just the timestamp as a string
        message = str(timestamp)
        
        is_valid = verify_wallet_signature(message, signature, hotkey)
        
        if is_valid:
            logger.info(f"✅ Signature verified for {hotkey[:20]}... (ts={timestamp})")
        else:
            logger.warning(f"❌ Invalid signature from {hotkey[:20]}... (ts={timestamp})")
        
        return is_valid
        
    except Exception as e:
        logger.error(f"Error verifying validator signature: {e}")
        return False


async def get_next_work() -> Optional[Dict[str, Any]]:
    """Get the next model evaluation from the queue."""
    # First check in-memory queue (for models queued during this gateway session)
    if _work_queue:
        return _work_queue.pop(0)
    
    # Then check database for pending models
    pending = await get_pending_models_from_db(limit=1)
    if pending:
        return pending[0]
    
    return None


async def get_pending_models_from_db(limit: int = None) -> List[Dict[str, Any]]:
    """
    Get pending models from qualification_models table in Supabase.
    
    Queries for models where:
    - evaluated_at IS NULL (not yet evaluated)
    - status = 'submitted' (submitted by miner, payment verified)
    
    Orders by created_at ASC (FIFO - oldest first).
    
    Args:
        limit: Max models to return. If None, uses CONFIG.MAX_MODELS_PER_EPOCH.
    
    Returns:
        List of work items ready for evaluation.
    """
    if limit is None:
        limit = CONFIG.MAX_MODELS_PER_EPOCH
    
    try:
        from gateway.db.client import get_read_client
        supabase = get_read_client()
        
        # Query for pending models (FIFO order)
        result = supabase.table("qualification_models").select(
            "id, miner_hotkey, model_name, code_hash, s3_path, created_at, icp_set_id"
        ).is_(
            "evaluated_at", "null"
        ).eq(
            "status", "submitted"
        ).order(
            "created_at", desc=False  # ASC - oldest first (FIFO)
        ).limit(limit).execute()
        
        if not result.data:
            logger.debug("No pending models in database")
            return []
        
        logger.info(f"📋 Found {len(result.data)} pending models in queue (limit={limit})")
        
        # Convert to work items
        work_items = []
        for model in result.data:
            work_item = await prepare_work_item_from_model(model)
            if work_item:
                work_items.append(work_item)
        
        return work_items
        
    except Exception as e:
        logger.error(f"Error querying pending models from DB: {e}")
        return []


async def prepare_work_item_from_model(model: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Prepare a work item from a model record.
    
    Downloads model code from S3 and fetches ICP set.
    """
    model_id = model.get("id")
    s3_path = model.get("s3_path")
    
    if not model_id or not s3_path:
        logger.warning(f"Model missing required fields: id={model_id}, s3_path={s3_path}")
        return None
    
    try:
        # Convert S3 URI to key if needed
        # s3_path can be:
        #   - Full URI: "s3://bucket-name/key/path.tar.gz"
        #   - Just key: "key/path.tar.gz"
        s3_key = s3_path
        if s3_path.startswith("s3://"):
            # Strip "s3://bucket-name/" prefix to get just the key
            # Example: "s3://leadpoet-leads-primary/qualification/uploads/xxx.tar.gz"
            #       -> "qualification/uploads/xxx.tar.gz"
            parts = s3_path[5:].split("/", 1)  # Remove "s3://" and split by first "/"
            if len(parts) > 1:
                s3_key = parts[1]  # Everything after the bucket name
            else:
                logger.warning(f"Invalid S3 URI format: {s3_path}")
                s3_key = s3_path
        
        logger.info(f"📥 Downloading model {model_id[:8]}... from S3 key: {s3_key}")
        
        # Download model code from S3
        model_code = await fetch_model_code(s3_key)
        if not model_code:
            logger.warning(f"Failed to download model {model_id[:8]}... from S3, will retry later")
            model_code = b""
        
        # Get ICP set
        icp_set_id = model.get("icp_set_id", 1)
        icp_result = await get_icp_set(icp_set_id)
        if not icp_result or not icp_result[0]:
            logger.warning(f"ICP set {icp_set_id} not found, using default")
            icp_result = await get_icp_set(1)
        
        icps, icp_set_hash = icp_result
        
        # Create evaluation runs (one per ICP)
        evaluation_id = str(uuid4())
        evaluation_runs = []
        for icp in icps:
            run_id = str(uuid4())
            evaluation_runs.append({
                "evaluation_run_id": run_id,
                "probe_id": icp.get("icp_id", str(uuid4())),
                "probe_name": icp.get("industry", "Unknown"),
                "icp_data": icp,
                "stage": "final"
            })
        
        work_item = {
            "evaluation_id": evaluation_id,
            "model_id": model_id,
            "miner_hotkey": model.get("miner_hotkey", "Unknown"),
            "model_name": model.get("model_name", f"model_{model_id[:8]}"),
            "code_hash": model.get("code_hash", ""),
            "s3_key": s3_key,  # Use s3_key (not s3_path) to match check_evaluation_complete()
            "agent_code": base64.b64encode(model_code).decode() if model_code else "",
            "evaluation_runs": evaluation_runs,
            "icp_set_hash": icp_set_hash,
            "queued_at": model.get("created_at", datetime.now(timezone.utc).isoformat())
        }
        
        logger.info(f"📦 Prepared work item for model {model_id[:8]}... ({len(icps)} ICPs)")
        return work_item
        
    except Exception as e:
        logger.error(f"Error preparing work item for model {model_id[:8]}...: {e}")
        return None


async def get_pending_evaluations() -> List[Dict[str, Any]]:
    """
    Get pending evaluations from database.
    
    DEPRECATED: Use get_pending_models_from_db() instead.
    Kept for backwards compatibility.
    """
    return await get_pending_models_from_db(limit=1)


async def get_model(model_id: str) -> Optional[Dict[str, Any]]:
    """Get model details from database."""
    # Placeholder - in production, query qualification_models table
    logger.warning("PLACEHOLDER: get_model - implement DB query")
    return {
        "model_id": model_id,
        "code_s3_path": f"qualification/models/{model_id}/model.tar.gz"
    }


async def get_icp_set(set_id: int = None) -> tuple:
    """
    Get the active ICP set from private database.
    
    CRITICAL: ICPs are stored PRIVATELY in qualification_private_icp_sets
    and generated dynamically. Miners cannot see them until evaluation.
    
    NOTE: This reads from DB on BOTH testnet and production.
    The testnet guard only prevents WRITES to transparency_log,
    not reads from the ICP database.
    
    Args:
        set_id: Optional specific set_id. If None, gets the active set.
    
    Returns:
        Tuple of (icps_list, icp_set_hash) for logging
    """
    try:
        # Import gateway's ICP generator
        from gateway.tasks.icp_generator import get_active_icp_set
        
        active_set = await get_active_icp_set()
        
        if active_set:
            icps = active_set.get("icps", [])
            icp_hash = active_set.get("icp_set_hash", "")
            fetched_set_id = active_set.get("set_id")
            
            logger.info(f"Loaded {len(icps)} ICPs from active set {fetched_set_id}, hash={icp_hash[:16]}...")
            return icps, icp_hash
        
        # No active set - generate one
        logger.warning("No active ICP set found, generating emergency set...")
        from gateway.tasks.icp_generator import generate_and_activate_icp_set
        
        new_set_id = await generate_and_activate_icp_set()
        if new_set_id:
            active_set = await get_active_icp_set()
            if active_set:
                return active_set.get("icps", []), active_set.get("icp_set_hash", "")
        
        # Fallback if all else fails
        raise Exception("Failed to generate ICP set")
        
    except ImportError as e:
        logger.warning(f"Cannot import gateway ICP generator (running in validator?): {e}")
        # Validators shouldn't be calling this directly
        raise ValueError("Validators must receive ICPs from gateway, not generate them")
    
    except Exception as e:
        logger.error(f"Failed to get ICP set: {e}")
        raise


async def fetch_model_code(s3_key: str) -> bytes:
    """
    Fetch model code from S3.
    
    Downloads the model tarball from S3 and returns the raw bytes.
    
    Args:
        s3_key: The S3 key (path within bucket) e.g. "qualification/uploads/uuid.tar.gz"
    
    Returns:
        bytes: The raw tarball bytes
    """
    import boto3
    from botocore.exceptions import ClientError
    import io
    
    # Get S3 config from environment or defaults
    bucket = os.getenv("QUAL_S3_BUCKET", "leadpoet-leads-primary")
    region = os.getenv("QUAL_S3_REGION", "us-east-2")
    
    try:
        s3_client = boto3.client('s3', region_name=region)
        
        # Download to memory
        response = s3_client.get_object(Bucket=bucket, Key=s3_key)
        model_bytes = response['Body'].read()
        
        logger.info(f"Downloaded model from S3: {s3_key} ({len(model_bytes)} bytes)")
        return model_bytes
        
    except ClientError as e:
        error_code = e.response.get('Error', {}).get('Code', 'Unknown')
        logger.error(f"S3 download failed: {error_code} - {e}")
        return b""
    except Exception as e:
        logger.error(f"Failed to fetch model from S3: {e}")
        return b""


async def extract_code_content_from_s3(s3_path: str) -> Optional[Dict[str, str]]:
    """
    Extract code files from S3 tarball for public viewing.
    
    Downloads the tarball from S3, extracts Python/text files,
    and returns them as a dict {filename: content}.
    
    Args:
        s3_path: S3 path - can be:
            - Full URI: "s3://bucket/qualification/uploads/uuid.tar.gz"
            - Just key: "qualification/uploads/uuid.tar.gz"
    
    Returns:
        Dict mapping filename to content, or None on error
    """
    import tarfile
    import io
    
    if not s3_path:
        logger.warning("Empty S3 path provided")
        return None
    
    # Handle both full URI and key-only formats
    if s3_path.startswith("s3://"):
        # Full URI format: extract key from s3://bucket/path
        parts = s3_path.replace("s3://", "").split("/", 1)
        if len(parts) < 2:
            logger.warning(f"Cannot parse S3 path: {s3_path}")
            return None
        s3_key = parts[1]  # Everything after bucket name
    else:
        # Already just the key
        s3_key = s3_path
    
    try:
        # Fetch tarball bytes
        tarball_bytes = await fetch_model_code(s3_key)
        if not tarball_bytes:
            logger.warning(f"Empty tarball from {s3_key}")
            return None
        
        # Extract files
        code_files = {}
        allowed_extensions = {'.py', '.txt', '.md', '.json', '.yaml', '.yml', '.toml'}
        max_file_size = 100 * 1024  # 100KB per file
        max_total_size = 500 * 1024  # 500KB total
        total_size = 0
        
        with tarfile.open(fileobj=io.BytesIO(tarball_bytes), mode='r:gz') as tar:
            for member in tar.getmembers():
                # Skip directories and non-files
                if not member.isfile():
                    continue
                
                # Get filename (strip leading directory if present)
                filename = member.name
                if '/' in filename:
                    filename = filename.split('/', 1)[1] if filename.count('/') == 1 else filename
                
                # Check extension
                ext = '.' + filename.split('.')[-1].lower() if '.' in filename else ''
                if ext not in allowed_extensions:
                    continue
                
                # Skip hidden files
                if filename.startswith('.') or '/__' in filename:
                    continue
                
                # Check size limits
                if member.size > max_file_size:
                    logger.debug(f"Skipping large file {filename}: {member.size} bytes")
                    continue
                
                if total_size + member.size > max_total_size:
                    logger.debug(f"Reached total size limit, skipping {filename}")
                    break
                
                # Extract content
                try:
                    f = tar.extractfile(member)
                    if f:
                        content = f.read().decode('utf-8', errors='replace')
                        code_files[filename] = content
                        total_size += len(content)
                except Exception as e:
                    logger.debug(f"Could not extract {filename}: {e}")
                    continue
        
        logger.info(f"Extracted {len(code_files)} code files from {s3_key} ({total_size} bytes)")
        return code_files if code_files else None
        
    except Exception as e:
        logger.error(f"Failed to extract code from {s3_path}: {e}")
        return None


async def store_evaluation_run_result(
    evaluation_run_id: str,
    lead_returned: Optional[Dict],
    lead_score: Optional[Dict],
    icp_fit_score: float,
    decision_maker_score: float,
    intent_signal_score: float,
    cost_penalty: float,
    time_penalty: float,
    final_lead_score: float,
    run_cost_usd: float,
    run_time_seconds: float,
    status: str
) -> None:
    """
    Store evaluation run results.
    
    For testnet: stores in memory (_evaluation_results)
    For production: would UPDATE qualification_evaluation_runs table
    
    NOTE: Champion determination is done by validator locally, not here.
    """
    _evaluation_results[evaluation_run_id] = {
        "evaluation_run_id": evaluation_run_id,
        "lead_returned": lead_returned,
        "lead_score": lead_score,
        "icp_fit_score": icp_fit_score,
        "decision_maker_score": decision_maker_score,
        "intent_signal_score": intent_signal_score,
        "cost_penalty": cost_penalty,
        "time_penalty": time_penalty,
        "final_lead_score": final_lead_score,
        "run_cost_usd": run_cost_usd,
        "run_time_seconds": run_time_seconds,
        "status": status,
        "completed_at": datetime.now(timezone.utc).isoformat()
    }
    
    logger.info(
        f"📊 Stored result for run {evaluation_run_id[:8]}...: "
        f"score={final_lead_score:.2f}, cost=${run_cost_usd:.4f}"
    )
    
    # Check if all runs for this evaluation are complete
    # This updates Supabase but does NOT determine champion (validator does that locally)
    await check_evaluation_complete(evaluation_run_id)


async def check_evaluation_complete(evaluation_run_id: str) -> None:
    """
    Check if all runs for an evaluation are complete.
    
    If so, calculate aggregate score and update model status in Supabase.
    
    NOTE: Champion determination is done LOCALLY by the validator, not here.
    The validator will call /champion-status to notify us of any champion changes.
    
    Searches both:
    - _work_queue (pending work)
    - _assigned_work_items (work assigned to validators)
    """
    # Find which evaluation this run belongs to
    # First check _work_queue (pending)
    work_item = None
    for item in _work_queue:
        for run in item.get("evaluation_runs", []):
            if run.get("evaluation_run_id") == evaluation_run_id:
                work_item = item
                break
        if work_item:
            break
    
    # If not found in queue, check _assigned_work_items (active evaluations)
    if not work_item:
        for eval_id, item in _assigned_work_items.items():
            for run in item.get("evaluation_runs", []):
                if run.get("evaluation_run_id") == evaluation_run_id:
                    work_item = item
                    break
            if work_item:
                break
    
    if not work_item:
        logger.warning(f"Could not find work item for run {evaluation_run_id[:8]}...")
        return None
    
    # Found the work item!
    evaluation_id = work_item.get("evaluation_id")
    model_id = work_item.get("model_id")
    miner_hotkey = work_item.get("miner_hotkey", "unknown")
    
    # Check if all runs are complete
    run_ids = [r["evaluation_run_id"] for r in work_item.get("evaluation_runs", [])]
    completed_runs = [rid for rid in run_ids if rid in _evaluation_results]
    
    logger.debug(
        f"Checking evaluation {evaluation_id[:8]}...: "
        f"{len(completed_runs)}/{len(run_ids)} runs complete"
    )
    
    if len(completed_runs) < len(run_ids):
        # Not all runs complete yet
        return None
    
    # All runs complete! Calculate aggregate score
    total_score = sum(
        _evaluation_results[rid].get("final_lead_score", 0)
        for rid in run_ids
    )
    avg_score = total_score / len(run_ids) if run_ids else 0
    
    total_cost = sum(
        _evaluation_results[rid].get("run_cost_usd", 0)
        for rid in run_ids
    )
    
    total_time = sum(
        _evaluation_results[rid].get("run_time_seconds", 0)
        for rid in run_ids
    )
    
    logger.info(
        f"🏆 EVALUATION COMPLETE: model={model_id[:8]}..., "
        f"avg_score={avg_score:.2f}, total_cost=${total_cost:.4f}, "
        f"runs={len(run_ids)}"
    )
    
    # Update model status in memory
    if model_id in _models_store:
        _models_store[model_id]["status"] = "evaluated"
        _models_store[model_id]["avg_score"] = avg_score
        _models_store[model_id]["total_cost"] = total_cost
        _models_store[model_id]["evaluated_at"] = datetime.now(timezone.utc).isoformat()
    
    # =========================================================
    # WRITE TO DATABASE: Update qualification tables
    # =========================================================
    try:
        from gateway.db.client import get_write_client
        supabase = get_write_client()
        
        # 1. Update model status in qualification_models
        update_data = {
            "status": "evaluated",
            "score": avg_score,
            "evaluation_cost_usd": total_cost,
            "evaluation_time_seconds": int(total_time),
            "evaluated_at": datetime.now(timezone.utc).isoformat()
        }
        
        # 2. Extract and store code content for public viewing
        s3_key = work_item.get("s3_key")
        if s3_key:
            try:
                # Construct full S3 path from key
                bucket = os.getenv("QUAL_S3_BUCKET", "leadpoet-leads-primary")
                s3_path = f"s3://{bucket}/{s3_key}"
                code_content = await extract_code_content_from_s3(s3_path)
                if code_content:
                    update_data["code_content"] = code_content
                    logger.info(f"📄 Extracted {len(code_content)} code files for model {model_id[:8]}...")
            except Exception as code_err:
                logger.warning(f"Could not extract code content: {code_err}")
        
        supabase.table("qualification_models").update(update_data).eq("id", model_id).execute()
        logger.info(f"✅ Updated qualification_models: id={model_id[:8]}..., score={avg_score:.2f}")
        
        # NOTE: qualification_leaderboard is a VIEW that automatically reflects
        # the best scores from qualification_models. No need to INSERT into it.
        # The VIEW definition (in 003_leaderboard.sql) aggregates scores automatically.
        logger.info(f"📊 Leaderboard updated via VIEW: miner={miner_hotkey[:16]}..., score={avg_score:.2f}")
        
        # NOTE: Champion determination is done LOCALLY by the validator
        # Validator will call /champion-status endpoint to notify us
        
        # Clean up: remove from assigned work after successful DB update
        if evaluation_id in _assigned_work:
            del _assigned_work[evaluation_id]
        if evaluation_id in _assigned_work_items:
            del _assigned_work_items[evaluation_id]
        logger.debug(f"Cleaned up assigned work for evaluation {evaluation_id[:8]}...")
        
    except Exception as e:
        logger.error(f"❌ Failed to update qualification tables: {e}")
        import traceback
        logger.error(traceback.format_exc())


async def store_evaluation_run_error(
    evaluation_run_id: str,
    error_code: int,
    error_message: str
):
    """
    Store evaluation run error.
    
    For testnet: stores in memory (_evaluation_results)
    For production: would UPDATE qualification_evaluation_runs table
    """
    _evaluation_results[evaluation_run_id] = {
        "evaluation_run_id": evaluation_run_id,
        "status": "error",
        "error_code": error_code,
        "error_message": error_message,
        "final_lead_score": 0,  # Errors get 0 score
        "completed_at": datetime.now(timezone.utc).isoformat()
    }
    
    logger.error(
        f"❌ Stored error for run {evaluation_run_id[:8]}...: "
        f"code={error_code}, message={error_message}"
    )


# =============================================================================
# Champion Selection Logic
# =============================================================================

async def get_current_champion() -> Optional[Dict[str, Any]]:
    """
    Get the current champion model.
    
    Uses the qualification_models table with is_champion=True.
    
    Returns:
        Dict with champion info (model_id, miner_hotkey, score, champion_at, cost/time stats)
        or None if no champion exists
    """
    try:
        from gateway.db.client import get_read_client
        supabase = get_read_client()
        
        # Query qualification_models for the current champion
        # Include evaluation_cost_usd and evaluation_time_seconds for avg calculations
        result = supabase.table("qualification_models").select(
            "id, miner_hotkey, score, model_name, champion_at, evaluation_cost_usd, evaluation_time_seconds"
        ).eq("is_champion", True).limit(1).execute()
        
        if result.data:
            champion = result.data[0]
            # Calculate avg cost/time per lead (100 ICPs evaluated)
            total_cost = champion.get("evaluation_cost_usd") or 0
            total_time = champion.get("evaluation_time_seconds") or 0
            num_leads = 100  # Standard evaluation is 100 ICPs
            
            logger.info(
                f"🏆 Current champion: model={champion['id'][:8]}..., "
                f"miner={champion['miner_hotkey'][:16]}..., score={champion.get('score', 0):.2f}"
            )
            return {
                "model_id": champion["id"],
                "miner_hotkey": champion["miner_hotkey"],
                "score": champion.get("score", 0),
                "model_name": champion.get("model_name"),
                "became_champion_at": champion.get("champion_at"),
                "total_cost_usd": total_cost,
                "total_time_seconds": total_time,
                "avg_cost_per_lead_usd": total_cost / num_leads if num_leads > 0 else 0,
                "avg_time_per_lead_seconds": total_time / num_leads if num_leads > 0 else 0
            }
        
        logger.info("📭 No current champion - first valid model will become champion")
        return None
        
    except Exception as e:
        logger.error(f"Failed to get current champion: {e}")
        return None


async def check_and_update_champion(
    model_id: str,
    miner_hotkey: str,
    new_score: float,
    set_id: int = 1
) -> bool:
    """
    Check if a newly evaluated model should become the champion.
    
    Champion selection rules:
    1. If no current champion exists, new model becomes champion (if score > 0)
    2. If current champion exists, new model must beat it by CHAMPION_BEAT_THRESHOLD (2%)
    
    Uses the qualification_models table with is_champion flag.
    
    Args:
        model_id: The evaluated model's UUID
        miner_hotkey: The miner's hotkey
        new_score: The model's evaluation score
        set_id: The evaluation set ID (not used with simplified schema)
        
    Returns:
        True if model became the new champion, False otherwise
    """
    try:
        from gateway.db.client import get_write_client
        supabase = get_write_client()
        
        # Get current champion
        current_champion = await get_current_champion()
        
        # Case 1: No current champion - become champion if score > 0
        if current_champion is None:
            if new_score <= 0:
                logger.info(f"❌ Model {model_id[:8]}... scored 0 - cannot become champion")
                return False
            
            # Mark this model as champion
            supabase.table("qualification_models").update({
                "is_champion": True,
                "champion_at": datetime.now(timezone.utc).isoformat()
            }).eq("id", model_id).execute()
            
            logger.info(
                f"👑 NEW CHAMPION (first)! Model {model_id[:8]}... "
                f"by miner {miner_hotkey[:16]}... with score {new_score:.2f}"
            )
            print(f"\n{'='*60}")
            print(f"👑 NEW CHAMPION CROWNED!")
            print(f"   Model: {model_id[:8]}...")
            print(f"   Miner: {miner_hotkey[:16]}...")
            print(f"   Score: {new_score:.2f}")
            print(f"   (First champion - no prior champion existed)")
            print(f"{'='*60}\n")
            return True
        
        # Case 2: Current champion exists - need to beat by threshold
        current_score = current_champion.get("score", 0) or 0
        required_score = current_score * (1 + CHAMPION_BEAT_THRESHOLD)
        
        logger.info(
            f"🏆 Champion check: new_score={new_score:.2f} vs "
            f"required={required_score:.2f} (current={current_score:.2f} + {CHAMPION_BEAT_THRESHOLD*100:.0f}%)"
        )
        
        if new_score <= current_score:
            # Not even close - didn't beat current score
            logger.info(
                f"❌ Model {model_id[:8]}... did not beat current champion "
                f"({new_score:.2f} <= {current_score:.2f})"
            )
            return False
        
        if new_score < required_score:
            # Beat current but not by enough
            improvement = ((new_score - current_score) / current_score) * 100 if current_score > 0 else 100
            logger.info(
                f"📊 Model {model_id[:8]}... beat champion by {improvement:.1f}% "
                f"but needs {CHAMPION_BEAT_THRESHOLD*100:.0f}% improvement"
            )
            print(f"\n{'='*60}")
            print(f"📊 CHALLENGER FELL SHORT")
            print(f"   Model: {model_id[:8]}...")
            print(f"   Score: {new_score:.2f} (improvement: +{improvement:.1f}%)")
            print(f"   Required: {required_score:.2f} (need +{CHAMPION_BEAT_THRESHOLD*100:.0f}%)")
            print(f"   Current Champion Score: {current_score:.2f}")
            print(f"{'='*60}\n")
            return False
        
        # Case 3: New model beat champion by required threshold - CROWN NEW CHAMPION!
        improvement = ((new_score - current_score) / current_score) * 100 if current_score > 0 else 100
        
        # Dethrone old champion
        old_champion_id = current_champion.get("model_id")
        if old_champion_id:
            supabase.table("qualification_models").update({
                "is_champion": False,
                "dethroned_at": datetime.now(timezone.utc).isoformat()
            }).eq("id", old_champion_id).execute()
        
        # Crown new champion
        supabase.table("qualification_models").update({
            "is_champion": True,
            "champion_at": datetime.now(timezone.utc).isoformat()
        }).eq("id", model_id).execute()
        
        logger.info(
            f"👑 NEW CHAMPION! Model {model_id[:8]}... dethroned "
            f"{current_champion['model_id'][:8]}... with {improvement:.1f}% improvement"
        )
        print(f"\n{'='*60}")
        print(f"👑👑👑 NEW CHAMPION CROWNED! 👑👑👑")
        print(f"   New Champion Model: {model_id[:8]}...")
        print(f"   New Champion Miner: {miner_hotkey[:16]}...")
        print(f"   New Score: {new_score:.2f}")
        print(f"   Old Champion Score: {current_score:.2f}")
        print(f"   Improvement: +{improvement:.1f}%")
        print(f"   (Exceeded {CHAMPION_BEAT_THRESHOLD*100:.0f}% threshold!)")
        print(f"{'='*60}\n")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to update champion: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False
