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
from qualification.scoring.baseline import (
    REFERENCE_MODEL_ID,
    is_reference_model_id,
    load_baseline,
    load_baseline_from_db,
)


def _today_yyyymmdd_set_id() -> int:
    """Return today's UTC date as an int in YYYYMMDD format.

    This MUST match the keying scheme ``gateway/tasks/icp_generator.py``
    uses for ``qualification_private_icp_sets.set_id`` (and therefore
    ``qualification_baselines`` set_id/model_id rows). The function ``get_set_id_for_date``
    in icp_generator.py is the source of truth:

        int(dt.strftime("%Y%m%d"))     # e.g. 20260530

    Champion-selection's own ``set_id`` parameter is epoch-based
    (``current_epoch // EVALUATION_SET_ROTATION_EPOCHS``) — a DIFFERENT
    scheme used for evaluation bookkeeping. The baseline is keyed by the
    DAY THE ICP SET WAS ACTIVE, which is the YYYYMMDD form. Conflating
    the two was the bug that made every baseline lookup return None.
    """
    return int(datetime.now(timezone.utc).strftime("%Y%m%d"))


def _get_baseline_supabase_client():
    """Resolve a Supabase client that has read access to the RLS-protected
    ``qualification_baselines`` table.

    Tries, in order:
      1. ``gateway.db.client.get_write_client()`` — service_role, present on
         the gateway box (where the baseline is also WRITTEN, so this path
         is the source of truth).
      2. Direct ``supabase.create_client`` with ``SUPABASE_SERVICE_ROLE_KEY``
         from env — works on the validator IF the env var is present.
      3. None — caller falls back to file / legacy thresholding.

    The RLS policy ``service_role_all`` on ``qualification_baselines``
    requires service_role; the anon key WILL be rejected. If neither path
    yields a service_role client (e.g., validator container without the
    env var), this returns None and champion-selection silently falls
    back to legacy thresholding.
    """
    # Path 1: gateway's centralized client (works on gateway)
    try:
        from gateway.db.client import get_write_client
        client = get_write_client()
        if client is not None:
            return client
    except Exception as e:
        logger.debug(f"gateway.db.client.get_write_client unavailable: {e}")

    # Path 2: direct supabase create_client with service_role from env
    try:
        import os
        from supabase import create_client
        url = os.environ.get("SUPABASE_URL")
        key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
        if url and key:
            return create_client(url, key)
    except Exception as e:
        logger.debug(f"direct supabase create_client unavailable: {e}")

    return None


def _load_baseline_with_fallback(set_id: int):  # set_id arg kept for API compat; ignored
    """Load today's reference-model baseline.

    NOTE: the ``set_id`` parameter is intentionally IGNORED. Callers pass
    champion-selection's epoch-based set_id (e.g. 1175); the baseline is
    keyed by the icp_generator's YYYYMMDD set_id (e.g. 20260530). We
    compute the right key internally to avoid forcing every caller to
    know the difference.

    Returns ``BaselineRecord`` if today's daily run has completed and
    persisted, else ``None``. None → legacy champion-only thresholding,
    which is safe for first-day deployment and for the ~60 min window
    after 00:00 UTC while the baseline runner is in flight.
    """
    target_set_id = _today_yyyymmdd_set_id()

    # Try DB first
    client = _get_baseline_supabase_client()
    if client is not None:
        rec = load_baseline_from_db(
            target_set_id,
            client,
            model_id=REFERENCE_MODEL_ID,
        )
        if rec is not None:
            return rec
    else:
        logger.info(
            "No Supabase service-role client available for baseline lookup "
            "(neither gateway.db.client nor SUPABASE_SERVICE_ROLE_KEY env). "
            "Falling back to file."
        )

    # File fallback (dev/test, or DB returned nothing for this set)
    return load_baseline(target_set_id, model_id=REFERENCE_MODEL_ID)

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

    # Drop the daily reference run from the challenger pool — its role is
    # to set the baseline floor, not to compete for the crown. See
    # qualification/scoring/baseline.py:REFERENCE_MODEL_ID. Without this
    # filter the reference model could win champion against weak miner
    # submissions, which would defeat the "miners must beat baseline"
    # design.
    models = [m for m in models if not is_reference_model_id(m.model_id)]

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
    # Load today's reference-model baseline (None if daily run hasn't fired
    # yet for this set — falls through to legacy champion-only thresholding).
    # =========================================================================
    baseline_record = _load_baseline_with_fallback(set_id)
    baseline_score = baseline_record.baseline_score if baseline_record else None
    if baseline_score is not None:
        logger.info(
            f"Baseline floor (reference model run {baseline_record.scored_at}): "
            f"{baseline_score:.2f}"
        )
    else:
        logger.info(
            f"No baseline record for set {set_id} yet — falling back to "
            f"champion-only threshold (legacy behavior)"
        )

    # =========================================================================
    # Case 1: No current champion — first crowning is gated by the baseline.
    # Pre-baseline behavior auto-crowned the top model; that lets a weak
    # miner take the throne unopposed when the system is fresh (or after a
    # reset). With a baseline floor in place, the first champion must beat
    # baseline_score + CHAMPION_DETHRONING_THRESHOLD_POINTS, same threshold
    # any later challenger faces. If no baseline exists, preserve legacy
    # behavior (auto-crown) so deployment is safe before the daily runner
    # is wired up.
    # =========================================================================
    if not current_champion:
        # First-crown gate: top model must clear BOTH
        #   (a) baseline + CHAMPION_DETHRONING_THRESHOLD_POINTS (if baseline available)
        #   (b) MINIMUM_CHAMPION_SCORE
        # Whichever is HIGHER. (b) is the absolute floor a champion must
        # clear so a catastrophically-bad baseline day doesn't let a weak
        # miner take the throne (e.g. baseline=5 → baseline+10=15 < 20).
        if baseline_score is not None:
            first_champion_threshold = max(
                baseline_score + CONFIG.CHAMPION_DETHRONING_THRESHOLD_POINTS,
                CONFIG.MINIMUM_CHAMPION_SCORE,
            )
        else:
            # Pre-baseline / file-only case: enforce MINIMUM_CHAMPION_SCORE
            # so legacy auto-crown doesn't crown a sub-20 model.
            first_champion_threshold = CONFIG.MINIMUM_CHAMPION_SCORE

        if top_model.total_score <= first_champion_threshold:
            logger.info(
                f"No challenger exceeds first-crown threshold "
                f"({top_model.total_score:.2f} <= {first_champion_threshold:.2f}); "
                f"no champion crowned this epoch"
            )
            return ChampionSelectionResult(
                new_champion=None,
                previous_champion=None,
                action="no_change",
                margin=None,
                reason=(
                    f"No champion: top challenger {top_model.total_score:.2f} did not "
                    f"exceed first-crown threshold {first_champion_threshold:.2f} "
                    f"(baseline={baseline_score if baseline_score is not None else 'n/a'}, "
                    f"min={CONFIG.MINIMUM_CHAMPION_SCORE:.0f})"
                ),
            )

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
    
    # Threshold floor = max(champion_score, baseline_score). The dethroning
    # rule is "beat whichever is higher by CHAMPION_DETHRONING_THRESHOLD_POINTS":
    # this prevents a weak champion from blocking a stronger reference baseline
    # from acting as the floor — and prevents miners from taking the crown by
    # beating only a stale baseline that's now below the current champion.
    #
    # Also clamped at MINIMUM_CHAMPION_SCORE so a catastrophically-bad
    # baseline + a slightly-bad champion can't lower the bar below the
    # absolute champion floor (e.g. baseline=3 + champion=5 → floor=5,
    # threshold=15, but min=20 — must still clear 20).
    floor = champion_score
    floor_source = "champion"
    if baseline_score is not None and baseline_score > champion_score:
        floor = baseline_score
        floor_source = "baseline"

    threshold = max(
        floor + CONFIG.CHAMPION_DETHRONING_THRESHOLD_POINTS,
        CONFIG.MINIMUM_CHAMPION_SCORE,
    )

    logger.info(
        f"Champion score: {champion_score:.2f}, "
        f"Baseline score: {baseline_score if baseline_score is not None else 'n/a'}, "
        f"Floor: {floor:.2f} ({floor_source}), "
        f"Min champion: {CONFIG.MINIMUM_CHAMPION_SCORE:.0f}, "
        f"Threshold to dethrone: {threshold:.2f}"
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
            reason=f"Challenger beat champion by {margin:.1f}% (threshold: +{CONFIG.CHAMPION_DETHRONING_THRESHOLD_POINTS:.0f} points)"
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
            reason = f"Best challenger at {best_margin:.1f}% (need +{CONFIG.CHAMPION_DETHRONING_THRESHOLD_POINTS:.0f} points)"
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


def is_valid_dethrone_margin(challenger_score: float, champion_score: float) -> bool:
    """
    Check if score difference is sufficient to dethrone champion.
    
    Args:
        challenger_score: Challenger's absolute score
        champion_score: Champion's absolute score
    
    Returns:
        True if challenger leads by more than CHAMPION_DETHRONING_THRESHOLD_POINTS
    """
    return (challenger_score - champion_score) > CONFIG.CHAMPION_DETHRONING_THRESHOLD_POINTS


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
            "dethroning_threshold_points": CONFIG.CHAMPION_DETHRONING_THRESHOLD_POINTS,
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
