"""
Qualification API Router Assembly

Combines all qualification sub-routers into a single router for mounting
on the gateway. This is the main entry point for the qualification API.

Usage:
    from gateway.qualification.api.router import qualification_router
    app.include_router(qualification_router)

CRITICAL: This router is completely self-contained. Do NOT modify any
existing gateway routers or middleware.

NOTE: The API proxy for model execution is now handled LOCALLY by the
validator (qualification/validator/local_proxy.py), not by the gateway.
This improves reliability and removes gateway dependency during evaluation.
"""

from fastapi import APIRouter

from gateway.qualification.api.submit import router as submit_router
from gateway.qualification.api.status import router as status_router
from gateway.qualification.api.work import router as work_router
# NOTE: proxy_router removed - API proxy now runs locally on validator
# See qualification/validator/local_proxy.py

# =============================================================================
# Main Qualification Router
# =============================================================================

qualification_router = APIRouter(
    prefix="/qualification",
    tags=["qualification"],
    responses={
        401: {"description": "Invalid signature"},
        402: {"description": "Payment required"},
        403: {"description": "Forbidden - hotkey not registered"},
        404: {"description": "Not found"},
        429: {"description": "Rate limit exceeded"},
        500: {"description": "Internal server error"},
    }
)

# Include sub-routers (no additional prefix - they define their own paths)
qualification_router.include_router(submit_router)
qualification_router.include_router(status_router)
qualification_router.include_router(work_router)  # Validator work distribution
# NOTE: No proxy router - models call local proxy on validator, not gateway


# =============================================================================
# Export
# =============================================================================

__all__ = ["qualification_router"]
