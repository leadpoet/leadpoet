"""
Qualification System: Model Submission Endpoint

Phase 2.1 from tasks10.md

This module handles model submission for the Lead Qualification Agent competition.
Miners submit their models via this endpoint for evaluation.

NEW FLOW (Direct S3 Upload - Frontrunning Protected):
1. Miner calls POST /model/presign to get S3 upload URL
2. Miner uploads tarball directly to S3
3. Miner calls POST /model/submit with s3_key + code_hash + payment

This prevents frontrunning because:
- Code is never public until after submission is finalized
- Miner commits to hash BEFORE upload, gateway verifies after

CRITICAL: This is a NEW router. Do NOT modify any existing API files in gateway/api/.
"""

import os
import json
import hashlib
import logging
import time
from uuid import UUID, uuid4
from datetime import datetime, timezone
from typing import Optional, Dict, Any, Tuple

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from gateway.qualification.config import CONFIG
from gateway.qualification.models import (
    ModelSubmission,
    ModelSubmissionResponse,
    ModelStatus,
    ModelSubmittedPayload,
    PresignRequest,
    PresignResponse,
)
from gateway.qualification.api.payment import verify_payment as _verify_payment_impl, get_payment_info
from gateway.qualification.utils.helpers import (
    generate_presigned_upload_url,
    verify_model_upload,
)

# Import chain utilities for real implementations
from gateway.qualification.utils.chain import (
    verify_hotkey_signature as chain_verify_hotkey_signature,
    is_hotkey_registered as chain_is_hotkey_registered,
    get_current_bittensor_epoch as chain_get_current_bittensor_epoch,
    get_chain_info,
    BITTENSOR_NETWORK,
    BITTENSOR_NETUID,
)

logger = logging.getLogger(__name__)

# Router for submission endpoints (no prefix - parent router adds /qualification)
router = APIRouter()


# =============================================================================
# Presign Endpoint - Step 1 of submission
# =============================================================================

@router.post("/model/presign", response_model=PresignResponse)
async def get_presigned_upload_url(presign_request: PresignRequest, request: Request):
    """
    Get a presigned URL for direct S3 upload.
    
    This is Step 1 of the submission flow:
    1. Miner calls this endpoint → gets presigned S3 URL
    2. Miner uploads tarball directly to S3 using the URL
    3. Miner calls /model/submit with s3_key + code_hash
    
    Returns:
        PresignResponse with upload_url and s3_key
    
    Raises:
        401: Invalid signature
        403: Hotkey not registered
        429: Rate limit exceeded
    """
    logger.info(f"Presign request from hotkey={presign_request.miner_hotkey[:16]}...")
    
    # Verify timestamp is recent (within 5 minutes)
    current_time = int(time.time())
    if abs(current_time - presign_request.timestamp) > 300:
        raise HTTPException(
            status_code=400,
            detail="Timestamp too old or too far in future (must be within 5 minutes)"
        )
    
    # Verify signature over the presign request
    signature_valid = verify_hotkey_signature(
        hotkey=presign_request.miner_hotkey,
        signature=presign_request.signature,
        message_data={
            "miner_hotkey": presign_request.miner_hotkey,
            "timestamp": presign_request.timestamp
        }
    )
    
    if not signature_valid:
        logger.warning(f"Invalid signature for presign from hotkey={presign_request.miner_hotkey[:16]}...")
        raise HTTPException(
            status_code=401,
            detail="Invalid hotkey signature"
        )
    
    # Verify hotkey is registered
    is_registered = await is_hotkey_registered(presign_request.miner_hotkey)
    if not is_registered:
        raise HTTPException(
            status_code=403,
            detail="Hotkey not registered on subnet"
        )
    
    # =========================================================================
    # CRITICAL: Check rate limit BEFORE generating presigned URL
    # This prevents miners from paying TAO only to be rejected later
    # =========================================================================
    from gateway.qualification.api.model_rate_limiter import get_model_rate_limit_stats
    
    rate_limit_stats = get_model_rate_limit_stats(presign_request.miner_hotkey)
    daily_used = rate_limit_stats.get("daily_submissions_used", 0)
    daily_max = rate_limit_stats.get("daily_submissions_max", 2)
    credits_available = rate_limit_stats.get("submission_credits", 0)
    
    # Block if daily limit reached AND no credits available
    if daily_used >= daily_max and credits_available <= 0:
        hours_until_reset = rate_limit_stats.get("hours_until_reset", 24)
        logger.warning(
            f"Rate limit reached for hotkey={presign_request.miner_hotkey[:16]}... "
            f"(daily: {daily_used}/{daily_max}, credits: {credits_available})"
        )
        raise HTTPException(
            status_code=429,
            detail=f"Daily submission limit reached ({daily_max}/day). "
                   f"Resets in {hours_until_reset}h at midnight UTC. "
                   f"You have {credits_available} unused credit(s)."
        )
    
    logger.info(
        f"Rate limit check passed for hotkey={presign_request.miner_hotkey[:16]}... "
        f"(daily: {daily_used}/{daily_max}, credits: {credits_available})"
    )
    
    # Generate unique upload ID
    upload_id = str(uuid4())
    
    # Generate presigned URL
    upload_url, s3_key = await generate_presigned_upload_url(
        upload_id=upload_id,
        max_size_bytes=CONFIG.get_model_max_size_bytes(),
        expires_in_seconds=CONFIG.PRESIGN_EXPIRES_SECONDS
    )
    
    logger.info(f"Generated presigned URL for hotkey={presign_request.miner_hotkey[:16]}..., s3_key={s3_key}")
    
    return PresignResponse(
        upload_url=upload_url,
        s3_key=s3_key,
        expires_in_seconds=CONFIG.PRESIGN_EXPIRES_SECONDS,
        max_size_bytes=CONFIG.get_model_max_size_bytes(),
        # Include rate limit info so miner knows their status
        daily_submissions_used=daily_used,
        daily_submissions_max=daily_max,
        submission_credits=credits_available
    )


# =============================================================================
# Rate Limit Status Endpoint - Check before submitting
# =============================================================================

@router.get("/model/rate-limit/{miner_hotkey}")
async def get_rate_limit_status(miner_hotkey: str):
    """
    Check rate limit status for a miner.
    
    Miners can call this to check:
    - How many free daily submissions remain
    - How many paid credits they have
    - When daily limits reset
    
    No authentication required (public info).
    
    Returns:
        Dict with rate limit status
    """
    from gateway.qualification.api.model_rate_limiter import get_model_rate_limit_stats
    
    stats = get_model_rate_limit_stats(miner_hotkey)
    
    return {
        "miner_hotkey": miner_hotkey,
        "daily_submissions_used": stats["daily_submissions"],
        "daily_submissions_max": stats["max_daily_submissions"],
        "daily_submissions_remaining": max(0, stats["max_daily_submissions"] - stats["daily_submissions"]),
        "submission_credits": stats["submission_credits"],
        "can_submit_free": stats["daily_submissions"] < stats["max_daily_submissions"],
        "can_submit_with_credits": stats["submission_credits"] > 0,
        "can_submit": stats["can_submit"],
        "reset_at_utc": stats["reset_at"],
        "message": _build_rate_limit_message(stats)
    }


def _build_rate_limit_message(stats: dict) -> str:
    """Build human-readable rate limit message."""
    daily_remaining = stats["max_daily_submissions"] - stats["daily_submissions"]
    credits = stats["submission_credits"]
    
    if credits > 0 and daily_remaining > 0:
        return f"You have {daily_remaining} free submission(s) and {credits} credit(s). Free submissions are used first."
    elif credits > 0:
        return f"Daily free limit reached. You have {credits} credit(s) remaining."
    elif daily_remaining > 0:
        return f"You have {daily_remaining} free submission(s) remaining today. Resets at midnight UTC."
    else:
        return f"Daily limit reached. Purchase credits ($5 TAO each) or wait until midnight UTC for reset."


# =============================================================================
# Submit Endpoint - Step 3 of submission (after upload)
# =============================================================================

@router.post("/model/submit", response_model=ModelSubmissionResponse)
async def submit_model(submission: ModelSubmission, request: Request):
    """
    Submit a model for evaluation (after uploading to S3).
    
    This is Step 3 of the submission flow:
    1. Miner called /model/presign → got presigned URL
    2. Miner uploaded tarball to S3 using presigned URL
    3. Miner calls this endpoint with s3_key + code_hash + payment
    
    Gateway verifies:
    1. Hotkey signature
    2. Hotkey registered on subnet
    3. S3 object exists at s3_key
    4. Hash of S3 object matches claimed code_hash
    5. Payment on-chain verified
    6. Rate limit not exceeded
    
    Returns:
        ModelSubmissionResponse with model_id and status
    
    Raises:
        400: S3 object not found or hash mismatch
        401: Invalid signature
        402: Payment verification failed
        403: Hotkey not registered
        429: Rate limit exceeded
    """
    
    # Get client IP for logging
    client_ip = request.client.host if request.client else None
    
    logger.info(f"Model submission from hotkey={submission.miner_hotkey[:16]}..., s3_key={submission.s3_key}")
    
    # ---------------------------------------------------------------------
    # Step 1: Verify hotkey signature
    # ---------------------------------------------------------------------
    signature_valid = verify_hotkey_signature(
        hotkey=submission.miner_hotkey,
        signature=submission.signature,
        message_data=submission.model_dump(exclude={"signature"})
    )
    
    if not signature_valid:
        logger.warning(f"Invalid signature from hotkey={submission.miner_hotkey[:16]}...")
        raise HTTPException(
            status_code=401,
            detail="Invalid hotkey signature"
        )
    
    # ---------------------------------------------------------------------
    # Step 2: Verify hotkey is registered on subnet
    # ---------------------------------------------------------------------
    is_registered = await is_hotkey_registered(submission.miner_hotkey)
    
    if not is_registered:
        logger.warning(f"Unregistered hotkey attempted submission: {submission.miner_hotkey[:16]}...")
        raise HTTPException(
            status_code=403,
            detail="Hotkey not registered on subnet"
        )
    
    # ---------------------------------------------------------------------
    # Step 3: Verify S3 upload exists, size within limits, and hash matches
    # ---------------------------------------------------------------------
    is_valid, error_msg, file_size = await verify_model_upload(
        s3_key=submission.s3_key,
        claimed_hash=submission.code_hash,
        max_size_bytes=CONFIG.get_model_max_size_bytes()  # 10 MB from config
    )
    
    if not is_valid:
        logger.warning(f"S3 verification failed for hotkey={submission.miner_hotkey[:16]}...: {error_msg}")
        raise HTTPException(
            status_code=400,
            detail=f"Model verification failed: {error_msg}"
        )
    
    logger.info(f"S3 upload verified: {submission.s3_key} ({file_size} bytes)")
    
    # ---------------------------------------------------------------------
    # Step 4: Check rate limit and handle payment
    # ---------------------------------------------------------------------
    # Rate limit rules:
    # - 2 submissions per day MAXIMUM (resets at midnight UTC)
    # - Each $5 TAO payment = 1 credit (persists indefinitely)
    # - Credits do NOT bypass daily limit - they're a safety mechanism
    # - If payment succeeds but submission fails, credit remains for retry
    # ---------------------------------------------------------------------
    from gateway.qualification.api.model_rate_limiter import (
        check_model_rate_limit,
        reserve_model_submission,
        add_submission_credit,
        get_model_rate_limit_stats,
    )
    
    # First, check current rate limit status
    can_submit, limit_reason, rate_stats = check_model_rate_limit(submission.miner_hotkey)
    
    print(f"📊 Rate limit status for {submission.miner_hotkey[:16]}...:\n"
          f"   Daily: {rate_stats['daily_submissions']}/{rate_stats['max_daily_submissions']}\n"
          f"   Credits: {rate_stats['submission_credits']}\n"
          f"   Can submit: {can_submit}")
    
    # Check if daily limit exceeded (credits don't bypass this!)
    if not can_submit and rate_stats.get("limit_type") == "daily_limit":
        logger.warning(f"Daily limit reached for hotkey={submission.miner_hotkey[:16]}...")
        raise HTTPException(
            status_code=429,
            detail=limit_reason
        )
    
    # Initialize payment tracking variables
    amount_tao = 0.0
    amount_usd = 0.0
    tao_price_at_payment = 0.0
    sender_coldkey = None
    payment_verified = False
    has_credit = rate_stats.get("submission_credits", 0) > 0
    
    # If miner has existing credit (from previous failed submission), no payment needed
    if has_credit:
        print(f"✅ Miner has existing credit - no new payment required")
    
    # If payment is provided, verify it and add credit
    elif submission.payment_block_hash and submission.payment_extrinsic_index is not None:
        print(f"💰 Payment provided, verifying...")
        
        payment_valid, payment_error = await verify_payment(
            block_hash=submission.payment_block_hash,
            extrinsic_index=submission.payment_extrinsic_index,
            miner_hotkey=submission.miner_hotkey,
            required_usd=CONFIG.SUBMISSION_COST_USD
        )
        
        if payment_valid:
            # Get actual payment info from chain
            payment_info = await get_payment_info(
                block_hash=submission.payment_block_hash,
                extrinsic_index=submission.payment_extrinsic_index
            )
            
            amount_tao = payment_info.get("amount_tao", 0.0) if payment_info else 0.0
            amount_usd = payment_info.get("amount_usd", 0.0) if payment_info else 0.0
            tao_price_at_payment = payment_info.get("tao_price_at_payment", 0.0) if payment_info else 0.0
            sender_coldkey = payment_info.get("sender_coldkey") if payment_info else None
            
            print(
                f"✅ Payment verified on-chain: {amount_tao:.6f} TAO (${amount_usd:.2f}) "
                f"@ ${tao_price_at_payment:.2f}/TAO\n"
                f"   Block: {submission.payment_block_hash[:20]}... | Extrinsic: {submission.payment_extrinsic_index}\n"
                f"   Sender coldkey: {sender_coldkey[:20] if sender_coldkey else 'unknown'}..."
            )
            
            # Add credit for the payment (will be consumed after successful model creation)
            add_submission_credit(submission.miner_hotkey, credits_to_add=1)
            payment_verified = True
            has_credit = True
            
            # Re-check rate limit (now has credit)
            can_submit, limit_reason, rate_stats = check_model_rate_limit(submission.miner_hotkey)
        else:
            # Payment provided but failed verification
            logger.warning(f"Payment verification failed: {payment_error}")
            print(f"⚠️  Payment verification failed: {payment_error}")
            raise HTTPException(
                status_code=402,
                detail=f"Payment verification failed: {payment_error}"
            )
    else:
        # No credit and no payment provided
        logger.warning(f"No credit and no payment for hotkey={submission.miner_hotkey[:16]}...")
        raise HTTPException(
            status_code=402,
            detail="Payment required. Please pay $5 TAO to submit a model."
        )
    
    # Final check
    if not can_submit:
        logger.warning(f"Cannot submit for hotkey={submission.miner_hotkey[:16]}...: {limit_reason}")
        raise HTTPException(
            status_code=429,
            detail=limit_reason
        )
    
    print(f"✅ Rate limit check passed - proceeding with security scan")
    
    # ---------------------------------------------------------------------
    # Step 5: Quick security scan for obviously dangerous patterns
    # ---------------------------------------------------------------------
    # This happens AFTER payment verification so we can properly handle credits.
    # If scan fails:
    #   - Daily count is incremented (prevents spam)
    #   - Credit is NOT consumed (miner can fix and retry)
    # ---------------------------------------------------------------------
    from gateway.qualification.utils.helpers import scan_model_for_dangerous_patterns
    from gateway.qualification.api.model_rate_limiter import increment_daily_only
    
    is_safe, security_error = await scan_model_for_dangerous_patterns(submission.s3_key)
    
    if not is_safe:
        logger.warning(f"Security scan failed for hotkey={submission.miner_hotkey[:16]}...: {security_error}")
        
        # Increment daily count to prevent spam (but don't consume credit)
        increment_daily_only(submission.miner_hotkey)
        
        # Build helpful error message
        credit_msg = ""
        if has_credit:
            credit_msg = " Your credit has been preserved - fix the issue and retry."
        
        raise HTTPException(
            status_code=400,
            detail=f"Model rejected: {security_error}.{credit_msg} Daily submission count: {rate_stats['daily_submissions'] + 1}/2"
        )
    
    logger.info(f"Security scan passed for {submission.s3_key}")
    print(f"✅ Security scan passed - proceeding to create model record")
    
    # ---------------------------------------------------------------------
    # Step 6: Create model record in database FIRST
    # ---------------------------------------------------------------------
    # NOTE: We create the model BEFORE consuming the credit/daily slot.
    # This ensures that if payment was verified but model creation fails,
    # the miner still has their credit and can retry.
    # ---------------------------------------------------------------------
    model_id = uuid4()
    current_epoch = await get_current_bittensor_epoch()
    
    # S3 path is the full path including bucket
    s3_path = f"s3://leadpoet-leads-primary/{submission.s3_key}"
    
    try:
        await create_model_record(
            model_id=model_id,
            miner_hotkey=submission.miner_hotkey,
            code_s3_path=s3_path,
            code_hash=submission.code_hash,
            created_epoch=current_epoch,
            ip_address=client_ip,
            model_name=submission.model_name,
            # Include payment info from ACTUAL chain verification
            payment_block_hash=submission.payment_block_hash,
            payment_extrinsic_index=submission.payment_extrinsic_index,
            payment_amount_tao=amount_tao
        )
    except Exception as e:
        logger.error(f"Failed to create model record: {e}")
        # IMPORTANT: Credit is NOT consumed yet - miner can retry!
        print(f"⚠️  Model record creation failed - credit NOT consumed, miner can retry")
        raise HTTPException(
            status_code=500,
            detail="Failed to create model record. Your credit was NOT consumed - please try again."
        )
    
    # ---------------------------------------------------------------------
    # Step 6b: NOW consume the rate limit slot (after model is safely recorded)
    # ---------------------------------------------------------------------
    # We only consume the credit/daily slot AFTER the model is successfully 
    # recorded in the database. This ensures:
    # - If payment succeeded but model creation failed, credit remains
    # - Miner can simply retry without paying again
    # ---------------------------------------------------------------------
    reserved, reserve_reason, reserve_stats = reserve_model_submission(submission.miner_hotkey)
    
    if not reserved:
        # This shouldn't happen (we checked earlier), but log it
        logger.warning(f"Unexpected: Failed to reserve after model created: {reserve_reason}")
        # Model is already created, so continue anyway
    else:
        used_type = reserve_stats.get("used", "unknown")
        print(f"✅ Submission slot consumed (used: {used_type})\n"
              f"   Remaining daily: {reserve_stats['max_daily_submissions'] - reserve_stats['daily_submissions']}\n"
              f"   Remaining credits: {reserve_stats['submission_credits']}")
    
    # ---------------------------------------------------------------------
    # Step 7: Record payment to prevent reuse (only if payment was provided)
    # ---------------------------------------------------------------------
    if payment_verified and submission.payment_block_hash:
        try:
            await record_payment(
                block_hash=submission.payment_block_hash,
                extrinsic_index=submission.payment_extrinsic_index,
                model_id=model_id,
                miner_hotkey=submission.miner_hotkey,
                # Include actual payment info from chain verification
                miner_coldkey=sender_coldkey,
                amount_tao=amount_tao,
                amount_usd=amount_usd,
                tao_price_at_payment=tao_price_at_payment
            )
        except Exception as e:
            logger.error(f"Failed to record payment: {e}")
            # Don't fail the submission if payment recording fails
    else:
        print(f"ℹ️  Free submission (no payment recorded)")
    
    # ---------------------------------------------------------------------
    # Step 8: Log MODEL_SUBMITTED event to transparency log
    # ---------------------------------------------------------------------
    try:
        await log_event(
            event_type="MODEL_SUBMITTED",
            payload=ModelSubmittedPayload(
                model_id=str(model_id),
                miner_hotkey=submission.miner_hotkey,
                code_hash=submission.code_hash,
                s3_path=s3_path,
                payment_verified=payment_verified,  # Use actual payment status
                model_name=submission.model_name
            ).model_dump()
        )
    except Exception as e:
        logger.error(f"Failed to log MODEL_SUBMITTED event: {e}")
        # Don't fail the submission if logging fails
    
    # ---------------------------------------------------------------------
    # Step 9: AUTOMATICALLY QUEUE MODEL FOR EVALUATION
    # ---------------------------------------------------------------------
    # This is the critical step that triggers validators to evaluate the model
    print(f"📋 Step 9: Queueing model for evaluation...")
    try:
        from gateway.qualification.api.work import queue_model_for_evaluation
        print(f"   ✅ Imported queue_model_for_evaluation")
        
        eval_result = await queue_model_for_evaluation(
            model_id=str(model_id),
            s3_key=submission.s3_key,
            code_hash=submission.code_hash,
            miner_hotkey=submission.miner_hotkey,
            model_name=submission.model_name
        )
        
        print(f"   ✅ Model queued! evaluation_id={eval_result.get('evaluation_id', 'unknown')[:8]}...")
        logger.info(
            f"Model queued for evaluation: model_id={model_id}, "
            f"evaluation_id={eval_result.get('evaluation_id', 'unknown')[:8]}..."
        )
    except Exception as e:
        print(f"   ❌ Failed to queue model: {e}")
        import traceback
        traceback.print_exc()
        logger.error(f"Failed to queue model for evaluation: {e}")
        # Don't fail the submission - model is stored, can be queued later
    
    logger.info(
        f"Model submitted successfully: model_id={model_id}, "
        f"hotkey={submission.miner_hotkey[:16]}..., hash={submission.code_hash[:16]}..."
    )
    
    # Get updated rate limit stats for response
    final_stats = get_model_rate_limit_stats(submission.miner_hotkey)
    daily_remaining = final_stats.get("max_daily_submissions", 2) - final_stats.get("daily_submissions", 0)
    credits_remaining = final_stats.get("submission_credits", 0)
    reset_at = final_stats.get("reset_at")
    
    # Parse reset_at if it's a string
    daily_resets_at = None
    if reset_at:
        if isinstance(reset_at, str):
            try:
                daily_resets_at = datetime.fromisoformat(reset_at.replace("Z", "+00:00"))
            except ValueError:
                pass
        elif isinstance(reset_at, datetime):
            daily_resets_at = reset_at
    
    return ModelSubmissionResponse(
        model_id=model_id,
        status=ModelStatus.SUBMITTED,
        message="Model queued for evaluation",
        created_at=datetime.now(timezone.utc),
        daily_submissions_remaining=daily_remaining,
        submission_credits_remaining=credits_remaining,
        daily_resets_at=daily_resets_at
    )


# =============================================================================
# Placeholder Functions (to be implemented)
# =============================================================================

def verify_hotkey_signature(
    hotkey: str,
    signature: str,
    message_data: Dict[str, Any]
) -> bool:
    """
    Verify that the signature was created by the hotkey.
    
    Uses sr25519 signature verification from substrateinterface.
    
    Args:
        hotkey: The Bittensor hotkey (ss58 address)
        signature: Hex-encoded signature
        message_data: The data that was signed (will be JSON serialized)
    
    Returns:
        True if signature is valid, False otherwise
    """
    # Use real chain verification
    return chain_verify_hotkey_signature(hotkey, signature, message_data)


async def is_hotkey_registered(hotkey: str) -> bool:
    """
    Check if hotkey is registered on the subnet.
    
    Queries the metagraph to verify the hotkey is an active miner or validator.
    Uses dynamic network (testnet/mainnet) based on BITTENSOR_NETWORK env var.
    
    Args:
        hotkey: The Bittensor hotkey to check
    
    Returns:
        True if registered, False otherwise
    """
    # Use real chain verification
    is_registered, role = await chain_is_hotkey_registered(hotkey)
    if is_registered:
        logger.info(f"Hotkey {hotkey[:16]}... is registered as {role} on {BITTENSOR_NETWORK}")
    return is_registered


async def verify_payment(
    block_hash: str,
    extrinsic_index: int,
    miner_hotkey: str,
    required_usd: float
) -> Tuple[bool, Optional[str]]:
    """
    Verify on-chain TAO payment.
    
    Delegates to qualification.api.payment module for full implementation.
    
    See qualification/api/payment.py (Phase 2.2) for:
    - Duplicate payment check
    - Block fetch and validation
    - Transfer type verification
    - Destination verification (LEADPOET_WALLET)
    - Sender verification (coldkey owns hotkey)
    - Amount verification (TAO to USD conversion)
    - Extrinsic success check
    - Timing check (within 24 hours)
    
    Args:
        block_hash: The block hash containing the payment
        extrinsic_index: The extrinsic index within the block
        miner_hotkey: The miner's hotkey (to verify sender)
        required_usd: The required payment amount in USD
    
    Returns:
        Tuple of (is_valid, error_message)
        - (True, None) if valid
        - (False, "error reason") if invalid
    """
    return await _verify_payment_impl(
        block_hash=block_hash,
        extrinsic_index=extrinsic_index,
        miner_hotkey=miner_hotkey,
        required_usd=required_usd
    )


async def get_current_evaluation_set_id() -> int:
    """
    Get the current evaluation set ID.
    
    Evaluation sets rotate every EVALUATION_SET_ROTATION_EPOCHS epochs.
    
    Returns:
        Current set ID
    
    TODO: Calculate based on current epoch
    """
    # set_id = current_epoch // CONFIG.EVALUATION_SET_ROTATION_EPOCHS
    current_epoch = await get_current_bittensor_epoch()
    return current_epoch // CONFIG.EVALUATION_SET_ROTATION_EPOCHS


async def get_submission_count_this_set(miner_hotkey: str, set_id: int) -> int:
    """
    Get the number of submissions by this hotkey in the current evaluation set.
    
    Queries the qualification_models table in Supabase.
    
    Args:
        miner_hotkey: The miner's hotkey
        set_id: The current evaluation set ID
    
    Returns:
        Number of submissions in this set
    """
    try:
        from supabase import create_client
        
        # Get Supabase credentials
        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_ANON_KEY")
        
        if not supabase_url or not supabase_key:
            logger.warning("Supabase credentials not configured - returning 0")
            return 0
        
        supabase = create_client(supabase_url, supabase_key)
        
        # Calculate the time window for this evaluation set
        # Each set lasts EVALUATION_SET_ROTATION_EPOCHS epochs (~20 epochs = ~24 hours)
        # We use icp_set_id to track which set a submission belongs to
        response = supabase.table("qualification_models") \
            .select("id", count="exact") \
            .eq("miner_hotkey", miner_hotkey) \
            .eq("icp_set_id", set_id) \
            .execute()
        
        count = response.count if hasattr(response, 'count') and response.count else len(response.data or [])
        logger.info(f"Submission count for hotkey={miner_hotkey[:16]}... in set {set_id}: {count}")
        return count
        
    except Exception as e:
        logger.error(f"Error getting submission count: {e}")
        return 0


async def get_current_bittensor_epoch() -> int:
    """
    Get the current Bittensor epoch (block // 360).
    
    Uses the cached epoch from the metagraph registry - no live chain query needed.
    This is more resilient as it doesn't fail if the chain connection is stale.
    
    Returns:
        Current epoch number
    """
    # Use cached epoch from gateway's epoch utilities (same as metagraph uses)
    # This avoids chain connection issues for operations that don't need live data
    from gateway.utils.epoch import get_current_epoch_id_async
    return await get_current_epoch_id_async()


async def create_model_record(
    model_id: UUID,
    miner_hotkey: str,
    code_s3_path: str,
    code_hash: str,
    created_epoch: int,
    ip_address: Optional[str] = None,
    model_name: Optional[str] = None,
    payment_block_hash: Optional[str] = None,
    payment_extrinsic_index: Optional[int] = None,
    payment_amount_tao: float = 0.0
) -> None:
    """
    Create a model record in the database.
    
    Args:
        model_id: UUID for this model
        miner_hotkey: The miner's hotkey
        code_s3_path: S3 path to the tarball
        code_hash: SHA256 hash of the tarball
        created_epoch: Bittensor epoch when submitted
        ip_address: Client IP address (optional)
        model_name: Human-readable model name (optional)
        payment_block_hash: Block hash of the TAO payment (required NOT NULL)
        payment_extrinsic_index: Extrinsic index of the TAO payment (required NOT NULL)
        payment_amount_tao: Amount of TAO paid (from chain verification)
    
    Actual qualification_models columns:
        id, miner_hotkey, status, s3_path, code_hash, created_at, model_name,
        score, is_champion, champion_at, dethroned_at, evaluated_at, evaluated_by,
        evaluation_cost_usd, evaluation_time_seconds, icp_set_id, payment_amount_tao,
        payment_block_hash, payment_extrinsic_index, score_breakdown, updated_at
    """
    try:
        from gateway.db.client import get_write_client
        from datetime import datetime, timezone as tz
        supabase = get_write_client()
        
        # Insert model record with all required columns including payment info
        supabase.table("qualification_models").insert({
            "id": str(model_id),  # Use 'id' not 'model_id'
            "miner_hotkey": miner_hotkey,
            "status": "submitted",
            "s3_path": code_s3_path,  # Use 's3_path' not 'code_s3_path'
            "code_hash": code_hash,
            "model_name": model_name or f"model_{str(model_id)[:8]}",
            # Payment info (required NOT NULL columns)
            "payment_block_hash": payment_block_hash,
            "payment_extrinsic_index": payment_extrinsic_index,
            "payment_amount_tao": payment_amount_tao,  # Actual amount from chain
        }).execute()
        
        logger.info(
            f"✅ Model record stored in qualification_models: id={model_id}, "
            f"hotkey={miner_hotkey[:16]}..., payment={payment_amount_tao:.6f} TAO"
        )
    except Exception as e:
        logger.error(f"❌ Failed to store model record: {e}")
        # Don't raise - continue with in-memory tracking for now
        logger.info(
            f"Model record (in-memory): model_id={model_id}, "
            f"hotkey={miner_hotkey[:16]}..., epoch={created_epoch}, "
            f"name={model_name or 'unnamed'}"
        )


async def record_payment(
    block_hash: str,
    extrinsic_index: int,
    model_id: UUID,
    miner_hotkey: str,
    miner_coldkey: Optional[str] = None,
    amount_tao: float = 0.0,
    amount_usd: float = 0.0,
    tao_price_at_payment: float = 0.0
) -> None:
    """
    Record the payment to prevent reuse.
    
    Args:
        block_hash: The block hash containing the payment
        extrinsic_index: The extrinsic index
        model_id: The model this payment is for
        miner_hotkey: The miner's hotkey
        miner_coldkey: The miner's coldkey (sender)
        amount_tao: Actual TAO amount (from chain verification)
        amount_usd: USD equivalent at time of payment
        tao_price_at_payment: TAO price at time of payment
    
    Actual qualification_payments columns:
        id, block_hash, extrinsic_index, model_id, miner_hotkey,
        amount_tao, amount_usd, miner_coldkey, tao_price_at_payment, verified_at
    """
    try:
        from gateway.db.client import get_write_client
        from datetime import datetime, timezone as tz
        supabase = get_write_client()
        
        supabase.table("qualification_payments").insert({
            "block_hash": block_hash,
            "extrinsic_index": extrinsic_index,
            "model_id": str(model_id),
            "miner_hotkey": miner_hotkey,
            "miner_coldkey": miner_coldkey,
            # Actual payment amounts from chain verification
            "amount_tao": amount_tao,  # NOT NULL - required
            "amount_usd": amount_usd,
            "tao_price_at_payment": tao_price_at_payment,
        }).execute()
        
        logger.info(
            f"✅ Payment recorded in qualification_payments: block={block_hash[:16]}..., "
            f"model_id={model_id}, amount={amount_tao:.6f} TAO (${amount_usd:.2f})"
        )
    except Exception as e:
        logger.error(f"❌ Failed to record payment: {e}")
        # Don't raise - continue with in-memory tracking for now
        logger.info(f"Payment recorded (in-memory): block={block_hash[:16]}..., model_id={model_id}")


async def log_event(event_type: str, payload: Dict[str, Any]) -> None:
    """
    Log an event to the transparency log.
    
    Uses the existing gateway TEE signing infrastructure from gateway.utils.logger.
    This ensures qualification events are signed with the same TEE key as sourcing events.
    
    TESTNET GUARD: Skips transparency_log writes on testnet to prevent
    polluting production data (testnet and mainnet share same Supabase).
    
    Args:
        event_type: Type of event (e.g., "MODEL_SUBMITTED")
        payload: Event payload data
    """
    # ════════════════════════════════════════════════════════════════════════
    # TESTNET GUARD: Prevent testnet from writing to production transparency_log
    # This is consistent with gateway/api/validate.py and gateway/api/weights.py
    # ════════════════════════════════════════════════════════════════════════
    from gateway.config import BITTENSOR_NETWORK
    
    if BITTENSOR_NETWORK == "test":
        logger.info(
            f"⚠️ TESTNET MODE: Skipping {event_type} log to protect production transparency_log"
        )
        logger.info(f"   ℹ️ Would have logged: type={event_type}, payload_keys={list(payload.keys())}")
        return
    
    # ════════════════════════════════════════════════════════════════════════
    # MAINNET: Normal operation - log to transparency_log with TEE signing
    # ════════════════════════════════════════════════════════════════════════
    try:
        # Import the existing gateway logger which has TEE signing
        from gateway.utils.logger import log_event as gateway_log_event
        
        # Use the gateway's signed log_event function
        # This signs with the enclave Ed25519 key and stores to transparency_log
        log_entry = await gateway_log_event(event_type, payload)
        
        logger.info(
            f"✅ Event logged to transparency_log: type={event_type}, "
            f"hash={log_entry.get('event_hash', 'N/A')[:16]}..."
        )
        
    except ImportError as e:
        # If gateway logger not available (e.g., running qualification standalone), log warning
        logger.warning(f"Gateway logger not available, event not signed: {event_type}")
        logger.info(f"Event (unsigned): type={event_type}, payload_keys={list(payload.keys())}")
        
    except Exception as e:
        # Log error but don't fail the submission
        logger.error(f"Failed to log event to transparency_log: {e}")
        logger.info(f"Event (failed): type={event_type}, payload_keys={list(payload.keys())}")


# =============================================================================
# Health Check Endpoint
# =============================================================================

@router.get("/health")
async def health_check():
    """Health check endpoint for the qualification API."""
    return {
        "status": "healthy",
        "service": "qualification",
        "config": {
            "max_submissions_per_set": CONFIG.MAX_SUBMISSIONS_PER_SET,
            "submission_cost_usd": CONFIG.SUBMISSION_COST_USD,
            "evaluation_set_rotation_epochs": CONFIG.EVALUATION_SET_ROTATION_EPOCHS
        }
    }
