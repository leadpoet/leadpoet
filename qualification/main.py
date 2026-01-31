"""
Qualification System - Standalone Entry Point

This module provides a standalone FastAPI application for testing
the qualification system independently of the main gateway.

Usage:
    # Development/testing
    python -m qualification.main
    
    # Or with uvicorn
    uvicorn qualification.main:app --reload --port 8001

The qualification API is also available on the main gateway at /qualification/*
when running the full gateway (gateway/main.py).

CRITICAL: This is for TESTING ONLY. In production, the qualification
endpoints should be accessed through the main gateway which provides:
- TEE enclave keypair initialization
- AsyncSubtensor connection
- Background tasks (epoch monitor, etc.)
- Rate limiting middleware
"""

import os
import sys
import asyncio
import logging
from datetime import datetime, timezone
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from gateway.qualification.api.router import qualification_router
from gateway.qualification.config import CONFIG

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# =============================================================================
# Lifespan Context Manager
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for the qualification test app.
    
    This is a simplified version for testing. The production gateway
    has more extensive initialization (TEE, AsyncSubtensor, etc.).
    """
    logger.info("=" * 60)
    logger.info("🚀 Starting Qualification System (Standalone Mode)")
    logger.info("=" * 60)
    logger.info(f"   Mode: TESTING ONLY")
    logger.info(f"   Config:")
    logger.info(f"     - Max submissions per set: {CONFIG.MAX_SUBMISSIONS_PER_SET}")
    logger.info(f"     - Submission cost USD: ${CONFIG.SUBMISSION_COST_USD:.2f}")
    logger.info(f"     - Evaluation set rotation: {CONFIG.EVALUATION_SET_ROTATION_EPOCHS} epochs")
    logger.info(f"     - Max cost per lead: ${CONFIG.MAX_COST_PER_LEAD_USD:.4f}")
    logger.info(f"     - Running model timeout: {CONFIG.RUNNING_MODEL_TIMEOUT_SECONDS}s")
    logger.info("=" * 60)
    logger.info("")
    logger.info("⚠️  WARNING: Running in standalone mode")
    logger.info("   Many features require the full gateway:")
    logger.info("   - TEE event signing")
    logger.info("   - On-chain payment verification")
    logger.info("   - Metagraph registration checks")
    logger.info("   - Database persistence")
    logger.info("")
    logger.info("   For production, use the full gateway (gateway/main.py)")
    logger.info("=" * 60 + "\n")
    
    yield
    
    logger.info("\n" + "=" * 60)
    logger.info("🛑 Qualification System (Standalone) Shutting Down")
    logger.info("=" * 60 + "\n")


# =============================================================================
# Create FastAPI App
# =============================================================================

app = FastAPI(
    title="Lead Qualification Agent Competition API",
    description="""
Lead Qualification Agent Competition System

This API powers the Lead Qualification Agent competition mechanism,
where miners submit open-source lead qualification models that compete
in a "King of the Hill" format for 5% of subnet emissions.

## Key Endpoints

### Model Submission
- `POST /qualification/model/submit` - Submit a model for evaluation

### Status & Scores
- `GET /qualification/model/status/{model_id}` - Check model status
- `GET /qualification/model/score/{model_id}` - Get evaluation scores (PII redacted)
- `GET /qualification/model/detail/{model_id}` - Get full model details

### Leaderboard
- `GET /qualification/leaderboard` - View current leaderboard
- `GET /qualification/champion` - Get current champion information

### Health
- `GET /qualification/health` - Health check

## Authentication

All submission endpoints require:
1. Valid Bittensor hotkey signature
2. On-chain TAO payment ($6 USD equivalent)
3. Registered hotkey on subnet

## Rate Limits

- 1 submission per evaluation set (configurable)
- Evaluation sets rotate every 20 epochs
    """,
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# =============================================================================
# CORS Middleware
# =============================================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# Include Qualification Router
# =============================================================================

app.include_router(qualification_router)

# =============================================================================
# Root Endpoints
# =============================================================================

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "service": "leadpoet-qualification",
        "description": "Lead Qualification Agent Competition API",
        "status": "ok",
        "mode": "standalone-testing",
        "version": "1.0.0",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": {
            "max_submissions_per_set": CONFIG.MAX_SUBMISSIONS_PER_SET,
            "submission_cost_usd": CONFIG.SUBMISSION_COST_USD,
            "evaluation_set_rotation_epochs": CONFIG.EVALUATION_SET_ROTATION_EPOCHS,
        },
        "docs": "/docs",
        "redoc": "/redoc",
    }


@app.get("/health")
async def health():
    """Kubernetes health check."""
    return {"status": "healthy", "service": "qualification"}


# =============================================================================
# Run Server
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("QUALIFICATION_PORT", "8001"))
    
    print("=" * 60)
    print("🚀 Starting Lead Qualification Agent Competition API")
    print("=" * 60)
    print(f"   Port: {port}")
    print(f"   Docs: http://localhost:{port}/docs")
    print("=" * 60 + "\n")
    
    uvicorn.run(
        "qualification.main:app",
        host="0.0.0.0",
        port=port,
        reload=True,
        log_level="info"
    )
