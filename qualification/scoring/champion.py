"""
Qualification System: Champion Selection Logic

Phase 6.1 from tasks10.md

This module implements the "King of the Hill" champion selection mechanism
for the Lead Qualification Agent competition. The champion model receives
5% of subnet emissions.

Champion Selection Rules:
1. Challenger must beat current champion by >5% to dethrone
2. If multiple challengers beat by >5%, highest scorer wins
3. Champion is locked for MIN_CHAMPION_DURATION_EPOCHS after winning
4. Champion is automatically re-benchmarked when evaluation sets rotate

Storage Rules:
- Non-champions: DELETED from S3 immediately after evaluation
- Champions: COPIED to champions/ folder and kept FOREVER
- All historical champions are preserved for audit purposes

Key Functions:
- run_champion_selection() - Called at end of each epoch
- champion_rebenchmark() - Called when evaluation set rotates (every 20 epochs)
- cleanup_non_champion() - Delete non-champion after evaluation
- promote_to_champion() - Copy winning model to permanent storage

CRITICAL: This is NEW champion selection for qualification only.
Do NOT modify any existing weight calculation or consensus code
in the sourcing workflow.
"""

import logging
from datetime import datetime, timezone
from typing import Optional, List, NamedTuple
from uuid import UUID

from gateway.qualification.config import CONFIG
from gateway.qualification.utils.helpers import (
    copy_to_champions,
    delete_model_from_s3,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Types
# =============================================================================

class ChampionInfo(NamedTuple):
    """Information about the current champion."""
    model_id: UUID
    miner_hotkey: str
    set_id: int
    score: float
    became_champion_epoch: int
    became_champion_at: datetime
    code_hash: str  # SHA256 hash of the model code
    s3_path: str    # Permanent S3 path in champions/ folder
    model_name: Optional[str] = None


class ModelScore(NamedTuple):
    """Model with its score for champion selection."""
    model_id: UUID
    miner_hotkey: str
    total_score: float
    set_id: int
    status: str
    code_hash: str  # SHA256 hash
    upload_s3_key: str  # S3 key in uploads/ folder (will be deleted or promoted)
    model_name: Optional[str] = None


class ChampionSelectionResult(NamedTuple):
    """Result of champion selection."""
    new_champion: Optional[ChampionInfo]
    previous_champion: Optional[ChampionInfo]
    action: str  # "no_change", "new_champion", "dethroned", "first_champion"
    margin: Optional[float]  # Percentage margin over previous champion
    reason: str


# =============================================================================
# In-Memory State (for testing/development)
# In production, this is stored in qualification_champion_history table
# =============================================================================

_current_champion: Optional[ChampionInfo] = None
_champion_history: List[dict] = []


# =============================================================================
# Main Champion Selection Function
# =============================================================================

async def run_champion_selection(
    set_id: int,
    current_epoch: int
) -> ChampionSelectionResult:
    """
    Run champion selection at end of each Bittensor epoch.
    
    This is called at the end of each epoch to determine if a new champion
    should be crowned based on evaluation results.
    
    Rules:
    1. Challenger must beat champion by >5% to dethrone
    2. If multiple challengers beat by >5%, highest scorer wins
    3. Champion locked for MIN_CHAMPION_DURATION_EPOCHS epochs after winning
    
    Args:
        set_id: Current evaluation set ID
        current_epoch: Current Bittensor epoch/block number
    
    Returns:
        ChampionSelectionResult with details of the selection
    """
    logger.info(f"Running champion selection for set {set_id} at epoch {current_epoch}")
    
    # Get current champion
    current_champion = await get_current_champion()
    
    # Get all finished models in this set
    models = await get_finished_models(set_id)
    
    if not models:
        logger.warning("No finished models for champion selection")
        return ChampionSelectionResult(
            new_champion=current_champion,
            previous_champion=None,
            action="no_change",
            margin=None,
            reason="No finished models in this evaluation set"
        )
    
    # Sort by score descending
    models.sort(key=lambda m: m.total_score, reverse=True)
    top_model = models[0]
    
    logger.info(f"Top model: {top_model.model_id} with score {top_model.total_score}")
    
    # =========================================================================
    # Case 1: No current champion - first champion is crowned
    # =========================================================================
    if not current_champion:
        new_champion = await set_champion(top_model, current_epoch, set_id)
        await log_champion_selected(new_champion, None, current_epoch)
        
        logger.info(f"First champion crowned: {top_model.model_id}")
        return ChampionSelectionResult(
            new_champion=new_champion,
            previous_champion=None,
            action="first_champion",
            margin=None,
            reason="First champion crowned (no previous champion)"
        )
    
    # =========================================================================
    # Case 2: Champion is locked (minimum duration not met)
    # =========================================================================
    epochs_as_champion = current_epoch - current_champion.became_champion_epoch
    if epochs_as_champion < CONFIG.MIN_CHAMPION_DURATION_EPOCHS:
        remaining = CONFIG.MIN_CHAMPION_DURATION_EPOCHS - epochs_as_champion
        logger.info(
            f"Champion {current_champion.model_id} is locked for "
            f"{remaining} more epoch(s)"
        )
        return ChampionSelectionResult(
            new_champion=current_champion,
            previous_champion=None,
            action="no_change",
            margin=None,
            reason=f"Champion locked for {remaining} more epoch(s)"
        )
    
    # =========================================================================
    # Case 3: Check if any challenger can dethrone
    # =========================================================================
    
    # Get champion's current score on this set (may have been re-benchmarked)
    champion_score = await get_model_score(current_champion.model_id, set_id)
    
    if champion_score is None:
        # Champion hasn't been evaluated on this set yet
        logger.warning(
            f"Champion {current_champion.model_id} has no score for set {set_id}. "
            "Using previous score."
        )
        champion_score = current_champion.score
    
    # Calculate threshold: must beat champion by >5%
    threshold = champion_score * (1 + CONFIG.CHAMPION_DETHRONING_THRESHOLD_PCT)
    
    logger.info(
        f"Champion score: {champion_score:.2f}, "
        f"Threshold to dethrone: {threshold:.2f} "
        f"(+{CONFIG.CHAMPION_DETHRONING_THRESHOLD_PCT * 100:.0f}%)"
    )
    
    # Find challengers that beat the threshold
    challengers_beating_threshold = [
        m for m in models
        if m.model_id != current_champion.model_id and m.total_score > threshold
    ]
    
    if challengers_beating_threshold:
        # Highest scoring challenger wins
        new_champion_model = challengers_beating_threshold[0]
        margin = (new_champion_model.total_score - champion_score) / champion_score * 100
        
        logger.info(
            f"Challenger {new_champion_model.model_id} beats threshold with "
            f"score {new_champion_model.total_score:.2f} (+{margin:.1f}%)"
        )
        
        # Dethrone current champion (but DON'T delete from S3 - champions are kept forever)
        await dethrone_champion(current_champion, current_epoch)
        
        # Set new champion (this promotes to champions/ folder)
        new_champion = await set_champion(new_champion_model, current_epoch, set_id)
        
        # Delete all non-champions from uploads/ folder (cleanup)
        await cleanup_all_non_champions(models, winner_model_id=new_champion_model.model_id)
        
        # Log the change
        await log_champion_selected(new_champion, current_champion, current_epoch)
        
        return ChampionSelectionResult(
            new_champion=new_champion,
            previous_champion=current_champion,
            action="dethroned",
            margin=margin,
            reason=f"Challenger beat champion by {margin:.1f}% (threshold: {CONFIG.CHAMPION_DETHRONING_THRESHOLD_PCT * 100:.0f}%)"
        )
    else:
        # No challenger beats threshold - champion retained
        logger.info(f"Champion {current_champion.model_id} retained (no challenger beat threshold)")
        
        # Delete all non-champions from uploads/ folder (cleanup)
        # Note: Current champion's code is already in champions/ folder, not uploads/
        await cleanup_all_non_champions(models, winner_model_id=current_champion.model_id)
        
        # Log retention
        await log_champion_selected(current_champion, None, current_epoch)
        
        # Find best challenger for logging
        best_challenger = next(
            (m for m in models if m.model_id != current_champion.model_id),
            None
        )
        if best_challenger:
            best_margin = (best_challenger.total_score - champion_score) / champion_score * 100
            reason = f"Best challenger at {best_margin:.1f}% (need >{CONFIG.CHAMPION_DETHRONING_THRESHOLD_PCT * 100:.0f}%)"
        else:
            reason = "No challengers"
        
        return ChampionSelectionResult(
            new_champion=current_champion,
            previous_champion=None,
            action="no_change",
            margin=None,
            reason=reason
        )


# =============================================================================
# Champion Rebenchmarking
# =============================================================================

async def champion_rebenchmark(
    new_set_id: int,
    current_epoch: int
) -> bool:
    """
    Re-benchmark champion on a new evaluation set.
    
    Called at the start of each new evaluation set (every 20 epochs).
    This ensures the champion is evaluated on the same benchmark as challengers.
    
    Args:
        new_set_id: The new evaluation set ID
        current_epoch: Current Bittensor epoch
    
    Returns:
        True if rebenchmark was scheduled, False otherwise
    """
    current_champion = await get_current_champion()
    
    if not current_champion:
        logger.info("No current champion to rebenchmark")
        return False
    
    logger.info(
        f"Re-benchmarking champion {current_champion.model_id} "
        f"on new set {new_set_id}"
    )
    
    # Create new evaluation for champion on the new set
    evaluation_id = await create_evaluation(
        model_id=current_champion.model_id,
        set_id=new_set_id,
        created_epoch=current_epoch
    )
    
    logger.info(f"Created rebenchmark evaluation: {evaluation_id}")
    return True


async def check_evaluation_set_rotation(
    current_epoch: int
) -> Optional[int]:
    """
    Check if we need to rotate to a new evaluation set.
    
    Rotation happens every EVALUATION_SET_ROTATION_EPOCHS epochs.
    
    Args:
        current_epoch: Current Bittensor epoch
    
    Returns:
        New set_id if rotation needed, None otherwise
    """
    # Calculate if this epoch starts a new evaluation set
    if current_epoch % CONFIG.EVALUATION_SET_ROTATION_EPOCHS == 0:
        new_set_id = current_epoch // CONFIG.EVALUATION_SET_ROTATION_EPOCHS
        logger.info(f"Evaluation set rotation: new set_id = {new_set_id}")
        return new_set_id
    
    return None


# =============================================================================
# Database Operations (Placeholders)
# In production, these interact with qualification_champion_history table
# =============================================================================

async def get_current_champion() -> Optional[ChampionInfo]:
    """
    Get the current champion from database.
    
    Returns:
        ChampionInfo if there's an active champion, None otherwise
    """
    global _current_champion
    
    # TODO: In production, query qualification_champion_history table:
    # SELECT * FROM qualification_champion_history
    # WHERE dethroned_epoch IS NULL
    # ORDER BY became_champion_epoch DESC
    # LIMIT 1
    
    logger.debug(f"Current champion: {_current_champion}")
    return _current_champion


async def get_finished_models(set_id: int) -> List[ModelScore]:
    """
    Get all finished models for a given evaluation set.
    
    Args:
        set_id: Evaluation set ID
    
    Returns:
        List of ModelScore objects sorted by score descending
    """
    # TODO: In production, query qualification_model_scores table:
    # SELECT m.model_id, m.miner_hotkey, s.total_score, s.set_id, m.status
    # FROM qualification_models m
    # JOIN qualification_model_scores s ON m.model_id = s.model_id
    # WHERE m.status = 'finished'
    #   AND s.set_id = {set_id}
    #   AND m.model_id NOT IN (SELECT model_id FROM qualification_benchmark_models)
    # ORDER BY s.total_score DESC
    
    logger.warning(f"PLACEHOLDER: get_finished_models(set_id={set_id})")
    return []


async def get_model_score(model_id: UUID, set_id: int) -> Optional[float]:
    """
    Get a specific model's score for a given evaluation set.
    
    Args:
        model_id: Model UUID
        set_id: Evaluation set ID
    
    Returns:
        Total score or None if not found
    """
    # TODO: In production, query qualification_model_scores table:
    # SELECT total_score FROM qualification_model_scores
    # WHERE model_id = {model_id} AND set_id = {set_id}
    
    logger.warning(f"PLACEHOLDER: get_model_score(model_id={model_id}, set_id={set_id})")
    return None


async def set_champion(
    model: ModelScore,
    epoch: int,
    set_id: int
) -> ChampionInfo:
    """
    Set a new champion in the database and promote to permanent storage.
    
    This function:
    1. Copies model from uploads/ to champions/ folder (PERMANENT storage)
    2. Deletes model from uploads/ folder
    3. Creates ChampionInfo with permanent S3 path
    4. Updates database
    
    Champions are NEVER deleted from S3 - kept forever for historical record.
    
    Args:
        model: The model to make champion
        epoch: Epoch when champion was crowned
        set_id: Current evaluation set
    
    Returns:
        ChampionInfo for the new champion
    """
    global _current_champion
    
    now = datetime.now(timezone.utc)
    
    # Step 1: Promote to permanent champion storage
    # Copies to champions/ and deletes from uploads/
    champion_s3_path = await promote_to_champion(model)
    
    new_champion = ChampionInfo(
        model_id=model.model_id,
        miner_hotkey=model.miner_hotkey,
        set_id=set_id,
        score=model.total_score,
        became_champion_epoch=epoch,
        became_champion_at=now,
        code_hash=model.code_hash,
        s3_path=champion_s3_path,
        model_name=model.model_name
    )
    
    _current_champion = new_champion
    
    # TODO: In production, insert into qualification_champion_history:
    # INSERT INTO qualification_champion_history (
    #     model_id, miner_hotkey, set_id, score,
    #     became_champion_epoch, became_champion_at,
    #     code_hash, s3_path, model_name
    # ) VALUES (...)
    
    logger.info(f"Set new champion: {model.model_id} at epoch {epoch}, s3_path={champion_s3_path}")
    return new_champion


async def dethrone_champion(
    champion: ChampionInfo,
    epoch: int
) -> None:
    """
    Mark the current champion as dethroned.
    
    Args:
        champion: The champion being dethroned
        epoch: Epoch when dethroned
    """
    global _champion_history
    
    now = datetime.now(timezone.utc)
    
    # Record in history
    _champion_history.append({
        "model_id": champion.model_id,
        "miner_hotkey": champion.miner_hotkey,
        "set_id": champion.set_id,
        "score": champion.score,
        "became_champion_epoch": champion.became_champion_epoch,
        "became_champion_at": champion.became_champion_at,
        "dethroned_epoch": epoch,
        "dethroned_at": now,
    })
    
    # TODO: In production, update qualification_champion_history:
    # UPDATE qualification_champion_history
    # SET dethroned_epoch = {epoch}, dethroned_at = NOW()
    # WHERE model_id = {champion.model_id}
    #   AND dethroned_epoch IS NULL
    
    logger.info(f"Dethroned champion: {champion.model_id} at epoch {epoch}")


async def create_evaluation(
    model_id: UUID,
    set_id: int,
    created_epoch: int
) -> UUID:
    """
    Create a new evaluation record for a model.
    
    Used for champion rebenchmarking.
    
    Args:
        model_id: Model to evaluate
        set_id: Evaluation set
        created_epoch: Current epoch
    
    Returns:
        New evaluation UUID
    """
    from uuid import uuid4
    
    evaluation_id = uuid4()
    
    # TODO: In production, insert into qualification_evaluations:
    # INSERT INTO qualification_evaluations (
    #     evaluation_id, model_id, set_id, status, created_epoch
    # ) VALUES ({evaluation_id}, {model_id}, {set_id}, 'pending', {created_epoch})
    
    logger.info(f"Created evaluation {evaluation_id} for model {model_id} on set {set_id}")
    return evaluation_id


async def log_champion_selected(
    champion: ChampionInfo,
    previous_champion: Optional[ChampionInfo],
    epoch: int
) -> None:
    """
    Log champion selection event to transparency log.
    
    Args:
        champion: The current/new champion
        previous_champion: The dethroned champion (if any)
        epoch: Current epoch
    """
    # Calculate margin if there was a previous champion
    margin = None
    if previous_champion:
        margin = (champion.score - previous_champion.score) / previous_champion.score * 100
    
    log_payload = {
        "event_type": "CHAMPION_SELECTED",
        "epoch": epoch,
        "champion_model_id": str(champion.model_id),
        "champion_hotkey": champion.miner_hotkey,
        "champion_score": champion.score,
        "champion_set_id": champion.set_id,
        "previous_champion_id": str(previous_champion.model_id) if previous_champion else None,
        "previous_champion_score": previous_champion.score if previous_champion else None,
        "margin_pct": margin,
    }
    
    # TODO: In production, call transparency log:
    # await log_qualification_event("CHAMPION_SELECTED", log_payload)
    
    logger.info(f"Logged champion selection: {log_payload}")


# =============================================================================
# Utility Functions
# =============================================================================

def calculate_margin(challenger_score: float, champion_score: float) -> float:
    """
    Calculate percentage margin of challenger over champion.
    
    Args:
        challenger_score: Challenger's score
        champion_score: Champion's score
    
    Returns:
        Percentage margin (positive means challenger ahead)
    """
    if champion_score == 0:
        return float('inf') if challenger_score > 0 else 0.0
    return (challenger_score - champion_score) / champion_score * 100


def is_valid_dethrone_margin(margin: float) -> bool:
    """
    Check if margin is sufficient to dethrone champion.
    
    Args:
        margin: Percentage margin over champion
    
    Returns:
        True if margin exceeds threshold
    """
    return margin > CONFIG.CHAMPION_DETHRONING_THRESHOLD_PCT * 100


async def get_champion_history(limit: int = 10) -> List[dict]:
    """
    Get recent champion history.
    
    Args:
        limit: Maximum entries to return
    
    Returns:
        List of champion history entries
    """
    # TODO: In production, query qualification_champion_history:
    # SELECT * FROM qualification_champion_history
    # ORDER BY became_champion_epoch DESC
    # LIMIT {limit}
    
    return _champion_history[-limit:][::-1]


async def get_current_set_id(current_epoch: int) -> int:
    """
    Calculate current evaluation set ID from epoch.
    
    Args:
        current_epoch: Current Bittensor epoch
    
    Returns:
        Current set_id
    """
    return current_epoch // CONFIG.EVALUATION_SET_ROTATION_EPOCHS


def get_champion_selection_summary() -> dict:
    """
    Get summary of champion selection configuration and state.
    
    Returns:
        Dict with configuration and current state
    """
    return {
        "config": {
            "dethroning_threshold_pct": CONFIG.CHAMPION_DETHRONING_THRESHOLD_PCT * 100,
            "min_champion_duration_epochs": CONFIG.MIN_CHAMPION_DURATION_EPOCHS,
            "evaluation_set_rotation_epochs": CONFIG.EVALUATION_SET_ROTATION_EPOCHS,
        },
        "current_champion": {
            "model_id": str(_current_champion.model_id) if _current_champion else None,
            "miner_hotkey": _current_champion.miner_hotkey if _current_champion else None,
            "score": _current_champion.score if _current_champion else None,
            "became_champion_epoch": _current_champion.became_champion_epoch if _current_champion else None,
        },
        "history_count": len(_champion_history),
    }


# =============================================================================
# Storage Management (Non-champions deleted, champions kept forever)
# =============================================================================

async def promote_to_champion(model: ModelScore) -> str:
    """
    Promote a winning model to permanent champion storage.
    
    Copies the model from uploads/ to champions/ folder.
    Champion models are NEVER deleted.
    
    Args:
        model: The model being promoted to champion
    
    Returns:
        New S3 path in champions/ folder
    """
    logger.info(f"Promoting model {model.model_id} to champion storage...")
    
    # Copy from uploads/ to champions/
    champion_s3_key = await copy_to_champions(
        source_s3_key=model.upload_s3_key,
        model_id=model.model_id
    )
    
    # Delete from uploads/ (now in champions/)
    await delete_model_from_s3(model.upload_s3_key)
    
    logger.info(f"Model promoted: {model.upload_s3_key} → {champion_s3_key}")
    return f"s3://leadpoet-leads-primary/{champion_s3_key}"


async def cleanup_non_champion(model: ModelScore) -> None:
    """
    Delete a non-champion model from S3 after evaluation.
    
    Called immediately after evaluation completes for models that:
    - Did not beat the current champion by >5%
    - Failed screening stages
    
    Args:
        model: The model to delete
    """
    logger.info(f"Cleaning up non-champion model {model.model_id}...")
    
    success = await delete_model_from_s3(model.upload_s3_key)
    
    if success:
        logger.info(f"Deleted non-champion: {model.upload_s3_key}")
    else:
        logger.warning(f"Failed to delete non-champion: {model.upload_s3_key}")


async def cleanup_all_non_champions(
    models: List[ModelScore],
    winner_model_id: Optional[UUID] = None
) -> int:
    """
    Delete all non-champion models after champion selection.
    
    Args:
        models: List of all evaluated models
        winner_model_id: The model that won (skip deletion for this one)
    
    Returns:
        Number of models deleted
    """
    deleted_count = 0
    
    for model in models:
        if winner_model_id and model.model_id == winner_model_id:
            continue  # Skip the winner
        
        await cleanup_non_champion(model)
        deleted_count += 1
    
    logger.info(f"Cleaned up {deleted_count} non-champion models")
    return deleted_count


# =============================================================================
# Testing Helpers
# =============================================================================

def reset_champion_state():
    """Reset champion state for testing."""
    global _current_champion, _champion_history
    _current_champion = None
    _champion_history = []
    logger.info("Champion state reset")


def set_mock_champion(
    model_id: UUID,
    miner_hotkey: str,
    score: float,
    epoch: int,
    set_id: int = 1,
    code_hash: str = "mock_hash_1234567890abcdef",
    s3_path: str = "s3://leadpoet-leads-primary/qualification/champions/mock.tar.gz"
):
    """Set a mock champion for testing."""
    global _current_champion
    _current_champion = ChampionInfo(
        model_id=model_id,
        miner_hotkey=miner_hotkey,
        set_id=set_id,
        score=score,
        became_champion_epoch=epoch,
        became_champion_at=datetime.now(timezone.utc),
        code_hash=code_hash,
        s3_path=s3_path
    )
    logger.info(f"Set mock champion: {model_id}")
