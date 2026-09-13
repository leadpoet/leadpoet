"""Public chain epoch information for the gateway and Arena operators."""

from fastapi import APIRouter, HTTPException, Response

router = APIRouter(prefix="/epoch", tags=["Epoch"])


@router.get("/state")
async def get_epoch_state(response: Response):
    """Expose the exact-hash SN71 scheduler state used for cutover verification."""

    from gateway.utils.epoch import get_epoch_authority_status_async

    try:
        result = await get_epoch_authority_status_async()
        response.headers["Cache-Control"] = "private, no-store"
        response.headers["Pragma"] = "no-cache"
        return result
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Authoritative subnet epoch state unavailable: {exc}",
        ) from exc


@router.get("/{epoch_id}/info")
async def get_epoch_information(epoch_id: int):
    """
    Get comprehensive information about an epoch.
    
    Public endpoint (no authentication required) for checking epoch status.
    
    Args:
        epoch_id: Epoch number
    
    Returns:
        Epoch information dictionary
    
    Example:
        GET /epoch/100/info
    """
    try:
        from gateway.utils.epoch import get_epoch_info_async

        info = await get_epoch_info_async(epoch_id)
        return info
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get epoch info: {str(e)}"
        )


@router.get("/current")
async def get_current_epoch():
    """
    Get current epoch ID and information.
    
    Public endpoint for checking current epoch.
    
    Returns:
        {
            "current_epoch_id": int,
            "epoch_info": dict
        }
    
    Example:
        GET /epoch/current
    """
    try:
        from gateway.utils.epoch import (
            get_current_epoch_context_async,
            get_current_epoch_info_from_snapshot,
        )

        epoch_snapshot, current_epoch = await get_current_epoch_context_async()
        info = get_current_epoch_info_from_snapshot(epoch_snapshot)
        
        return {
            "current_epoch_id": current_epoch,
            "epoch_info": info
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get current epoch: {str(e)}"
        )
