"""
Qualification System: Emissions Distribution

Phase 6.2 from tasks10.md

This module handles the distribution of 5% of subnet emissions to the
current champion of the Lead Qualification Agent competition.

Emissions Flow:
1. Called at end of each Bittensor epoch
2. Get current champion from champion selection
3. Verify champion's hotkey is still registered on subnet
4. If valid champion → distribute 5% emissions
5. If no champion or unregistered → burn emissions
6. Log EMISSIONS_DISTRIBUTED event to transparency log

CRITICAL: This is NEW emissions logic for qualification (5% allocation).
Do NOT modify any existing emissions or weight submission code in
`neurons/validator.py`. The qualification emissions are handled
separately from the main subnet emissions mechanism.

Note: Actual emission distribution happens via weight updates in the
validator. This module handles the logic and logging - the actual
weight vector update that allocates 5% to the champion is done
during the validator's weight submission.
"""

import logging
from datetime import datetime, timezone
from typing import Optional, NamedTuple
from uuid import UUID

from gateway.qualification.config import CONFIG
from qualification.scoring.champion import get_current_champion, ChampionInfo

logger = logging.getLogger(__name__)


# =============================================================================
# Types
# =============================================================================

class EmissionsResult(NamedTuple):
    """Result of emissions distribution."""
    epoch: int
    champion_hotkey: Optional[str]
    champion_model_id: Optional[UUID]
    champion_score: Optional[float]
    emissions_pct: float
    burned: bool
    reason: str
    distributed_at: datetime


class EmissionsSummary(NamedTuple):
    """Summary of emissions over multiple epochs."""
    total_epochs: int
    distributed_epochs: int
    burned_epochs: int
    unique_champions: int
    burn_reasons: dict  # reason -> count


# =============================================================================
# In-Memory State (for tracking/testing)
# =============================================================================

_emissions_history: list[EmissionsResult] = []


# =============================================================================
# Main Emissions Distribution Function
# =============================================================================

async def distribute_emissions(current_epoch: int) -> EmissionsResult:
    """
    Distribute 5% of subnet emissions to current champion.
    
    Called at the end of each Bittensor epoch. This function:
    1. Gets the current champion
    2. Verifies the champion's hotkey is registered
    3. Logs the emissions distribution event
    4. Returns the result for further processing
    
    The actual weight allocation happens in the validator's weight
    submission code, which reads the champion's hotkey and allocates
    5% of weight to it.
    
    Args:
        current_epoch: Current Bittensor epoch/block number
    
    Returns:
        EmissionsResult with distribution details
    """
    logger.info(f"Processing emissions distribution for epoch {current_epoch}")
    
    now = datetime.now(timezone.utc)
    emissions_pct = CONFIG.QUALIFICATION_EMISSIONS_PCT
    
    # Get current champion
    champion = await get_current_champion()
    
    # =========================================================================
    # Case 1: No champion - burn emissions
    # =========================================================================
    if not champion:
        logger.warning("No champion - burning qualification emissions")
        
        result = EmissionsResult(
            epoch=current_epoch,
            champion_hotkey=None,
            champion_model_id=None,
            champion_score=None,
            emissions_pct=emissions_pct,
            burned=True,
            reason="No valid champion",
            distributed_at=now
        )
        
        await log_emissions_event(result)
        _emissions_history.append(result)
        return result
    
    # =========================================================================
    # Case 2: Champion hotkey unregistered - burn emissions
    # =========================================================================
    if not await is_hotkey_registered(champion.miner_hotkey):
        logger.warning(
            f"Champion hotkey {champion.miner_hotkey} unregistered - "
            "burning emissions"
        )
        
        result = EmissionsResult(
            epoch=current_epoch,
            champion_hotkey=champion.miner_hotkey,
            champion_model_id=champion.model_id,
            champion_score=champion.score,
            emissions_pct=emissions_pct,
            burned=True,
            reason="Champion hotkey unregistered",
            distributed_at=now
        )
        
        await log_emissions_event(result)
        _emissions_history.append(result)
        return result
    
    # =========================================================================
    # Case 3: Valid champion - distribute emissions
    # =========================================================================
    logger.info(
        f"Distributing {emissions_pct * 100:.1f}% emissions to champion "
        f"{champion.miner_hotkey} (model: {champion.model_id})"
    )
    
    result = EmissionsResult(
        epoch=current_epoch,
        champion_hotkey=champion.miner_hotkey,
        champion_model_id=champion.model_id,
        champion_score=champion.score,
        emissions_pct=emissions_pct,
        burned=False,
        reason="Distributed to champion",
        distributed_at=now
    )
    
    await log_emissions_event(result)
    _emissions_history.append(result)
    
    logger.info(f"Emissions distributed to champion {champion.miner_hotkey}")
    return result


# =============================================================================
# Hotkey Registration Check
# =============================================================================

async def is_hotkey_registered(
    hotkey: str,
    netuid: int = None
) -> bool:
    """
    Check if a hotkey is registered on the subnet.
    
    This is a placeholder that should be implemented to query the
    Bittensor metagraph for registration status.
    
    Args:
        hotkey: The hotkey address (SS58 format)
        netuid: Subnet network UID (defaults to CONFIG if not specified)
    
    Returns:
        True if registered, False otherwise
    """
    if netuid is None:
        import os
        netuid = int(os.getenv("SUBNET_NETUID", "41"))
    
    try:
        import bittensor as bt
        import asyncio
        
        loop = asyncio.get_event_loop()
        
        def _check_registration():
            subtensor = bt.subtensor(network="finney")
            metagraph = subtensor.metagraph(netuid)
            return hotkey in metagraph.hotkeys
        
        return await loop.run_in_executor(None, _check_registration)
        
    except ImportError:
        logger.warning("bittensor not installed - assuming hotkey is registered")
        return True
    except Exception as e:
        logger.error(f"Failed to check hotkey registration: {e}")
        # Fail closed - don't distribute if we can't verify
        return False


# =============================================================================
# Transparency Log Integration
# =============================================================================

async def log_emissions_event(result: EmissionsResult) -> None:
    """
    Log emissions distribution event to transparency log.
    
    Args:
        result: The emissions distribution result
    """
    payload = {
        "event_type": "EMISSIONS_DISTRIBUTED",
        "epoch": result.epoch,
        "champion_hotkey": result.champion_hotkey,
        "champion_model_id": str(result.champion_model_id) if result.champion_model_id else None,
        "champion_score": result.champion_score,
        "emissions_pct": result.emissions_pct,
        "burned": result.burned,
        "reason": result.reason,
        "distributed_at": result.distributed_at.isoformat(),
    }
    
    # TODO: In production, call transparency log:
    # await log_qualification_event("EMISSIONS_DISTRIBUTED", payload)
    
    logger.info(f"Logged emissions event: {payload}")


# =============================================================================
# Champion Weight Calculation
# =============================================================================

def get_champion_weight_allocation() -> float:
    """
    Get the weight allocation percentage for the champion.
    
    Returns:
        Weight allocation as decimal (e.g., 0.05 for 5%)
    """
    return CONFIG.QUALIFICATION_EMISSIONS_PCT


async def get_champion_for_weights() -> Optional[str]:
    """
    Get the champion's hotkey for weight allocation.
    
    Used by the validator during weight submission to allocate
    5% of weights to the champion.
    
    Returns:
        Champion's hotkey if valid, None otherwise
    """
    champion = await get_current_champion()
    
    if not champion:
        logger.debug("No champion for weight allocation")
        return None
    
    if not await is_hotkey_registered(champion.miner_hotkey):
        logger.debug(f"Champion {champion.miner_hotkey} unregistered - no weight allocation")
        return None
    
    return champion.miner_hotkey


def calculate_weight_with_champion(
    base_weights: dict,
    champion_hotkey: Optional[str],
    champion_allocation: float = None
) -> dict:
    """
    Adjust weight vector to include champion allocation.
    
    This is a helper function for the validator to use when
    constructing the weight vector. It takes the base weights
    (from standard scoring) and adjusts them to allocate
    a portion to the champion.
    
    Args:
        base_weights: Dict of {hotkey: weight} from standard scoring
        champion_hotkey: The champion's hotkey (or None)
        champion_allocation: Portion to allocate (default: CONFIG value)
    
    Returns:
        Adjusted weights dict
    
    Example:
        Base weights: {A: 0.6, B: 0.4}
        Champion: A with 5% allocation
        Result: {A: 0.62, B: 0.38} (A gets base + champion share)
    """
    if champion_allocation is None:
        champion_allocation = CONFIG.QUALIFICATION_EMISSIONS_PCT
    
    if not champion_hotkey or champion_hotkey not in base_weights:
        # No valid champion or champion not in weight set
        # Burn the champion allocation (distribute proportionally)
        logger.debug("No champion weight adjustment (burning champion allocation)")
        return base_weights.copy()
    
    # Scale base weights to make room for champion allocation
    remaining_allocation = 1.0 - champion_allocation
    
    adjusted_weights = {}
    total_base = sum(base_weights.values())
    
    for hotkey, weight in base_weights.items():
        if total_base > 0:
            # Scale weight to remaining allocation
            scaled_weight = (weight / total_base) * remaining_allocation
        else:
            scaled_weight = 0.0
        
        if hotkey == champion_hotkey:
            # Add champion allocation to champion's weight
            scaled_weight += champion_allocation
        
        adjusted_weights[hotkey] = scaled_weight
    
    return adjusted_weights


# =============================================================================
# Emissions History & Summary
# =============================================================================

def get_emissions_history(limit: int = 100) -> list[EmissionsResult]:
    """
    Get recent emissions distribution history.
    
    Args:
        limit: Maximum entries to return
    
    Returns:
        List of EmissionsResult (most recent first)
    """
    return _emissions_history[-limit:][::-1]


def get_emissions_summary(epochs: int = None) -> EmissionsSummary:
    """
    Get summary of emissions distributions.
    
    Args:
        epochs: Number of recent epochs to summarize (None = all)
    
    Returns:
        EmissionsSummary with statistics
    """
    history = _emissions_history
    if epochs:
        history = history[-epochs:]
    
    distributed = [r for r in history if not r.burned]
    burned = [r for r in history if r.burned]
    
    # Count unique champions
    unique_champions = len(set(
        r.champion_hotkey for r in distributed if r.champion_hotkey
    ))
    
    # Count burn reasons
    burn_reasons = {}
    for r in burned:
        burn_reasons[r.reason] = burn_reasons.get(r.reason, 0) + 1
    
    return EmissionsSummary(
        total_epochs=len(history),
        distributed_epochs=len(distributed),
        burned_epochs=len(burned),
        unique_champions=unique_champions,
        burn_reasons=burn_reasons
    )


def get_emissions_config() -> dict:
    """
    Get emissions configuration.
    
    Returns:
        Dict with emissions config
    """
    return {
        "qualification_emissions_pct": CONFIG.QUALIFICATION_EMISSIONS_PCT,
        "qualification_emissions_pct_display": f"{CONFIG.QUALIFICATION_EMISSIONS_PCT * 100:.1f}%",
    }


# =============================================================================
# Testing Helpers
# =============================================================================

def reset_emissions_history():
    """Reset emissions history for testing."""
    global _emissions_history
    _emissions_history = []
    logger.info("Emissions history reset")


def add_mock_emissions_result(
    epoch: int,
    champion_hotkey: Optional[str],
    burned: bool,
    reason: str
):
    """Add a mock emissions result for testing."""
    result = EmissionsResult(
        epoch=epoch,
        champion_hotkey=champion_hotkey,
        champion_model_id=None,
        champion_score=75.0 if champion_hotkey else None,
        emissions_pct=CONFIG.QUALIFICATION_EMISSIONS_PCT,
        burned=burned,
        reason=reason,
        distributed_at=datetime.now(timezone.utc)
    )
    _emissions_history.append(result)
    logger.info(f"Added mock emissions result for epoch {epoch}")
