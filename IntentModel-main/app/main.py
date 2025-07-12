"""
Leadpoet Intent Model v1.1 - FastAPI Application
Main API server with < 400ms P95 latency target.
"""

import asyncio
import logging
import time
import uuid
import uvicorn
from contextlib import asynccontextmanager
from typing import Dict, Any, List, Optional

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

from app.core.config import settings
from app.services.prompt_parser import PromptParser, ParsedQuery
from app.services.retrieval import RetrievalService, RetrievalCandidate
from app.services.scoring import ScoringService, ScoredLead
from app.services.lgbm_service import LightGBMService
from app.core.metrics import MetricsCollector
from app.core.cost_telemetry import CostTelemetry
from app.models.query import Query
from app.services.query_performance_service import QueryPerformanceService
from sqlalchemy.orm import Session
from app.core.database import SessionLocal, get_db
from sqlalchemy import text
from app.middleware.telemetry import TelemetryMiddleware
from app.core.datadog_apm import DatadogAPMMiddleware
from app.services.query_quality_reputation_service import QueryQualityReputationService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ServiceRegistry:
    """Registry for all application services."""
    
    def __init__(self):
        self.prompt_parser = PromptParser()
        self.scoring_service = ScoringService()
        self.retrieval_service: Optional[RetrievalService] = None
        self.lgbm_service: Optional[LightGBMService] = None
        self.metrics_collector = MetricsCollector()
        self.cost_telemetry = CostTelemetry()
        self._db_session: Optional[Session] = None
    
    async def initialize_services(self):
        logger.info("Initializing services...")
        
        # Create a database session that will be shared by services
        self._db_session = SessionLocal()
        
        try:
            # Initialize services with the shared session
            self.retrieval_service = RetrievalService(self._db_session)
            self.lgbm_service = LightGBMService()
            
            # Test database connection
            await self._test_database_connection()
            
            await self.scoring_service.cleanup()
            logger.info("Services initialized successfully.")
            
        except Exception as e:
            logger.error(f"Failed to initialize services: {e}")
            # Clean up session on initialization failure
            if self._db_session:
                self._db_session.close()
                self._db_session = None
            raise
    
    async def _test_database_connection(self):
        """Test database connection to ensure it's working."""
        try:
            # Simple test query
            result = self._db_session.execute(text("SELECT 1"))
            result.fetchone()
            logger.info("Database connection test successful")
        except Exception as e:
            logger.error(f"Database connection test failed: {e}")
            raise
    
    async def cleanup(self):
        """Cleanup all services and close database session."""
        logger.info("Cleaning up services...")
        
        try:
            # Cleanup services first
            if self.retrieval_service:
                await self.retrieval_service.cleanup()
            if self.scoring_service:
                await self.scoring_service.cleanup()
            
            # Close database session last
            if self._db_session:
                self._db_session.close()
                self._db_session = None
                logger.info("Database session closed")
            
            logger.info("✅ Services cleaned up successfully")
            
        except Exception as e:
            logger.error(f"❌ Error during service cleanup: {e}")
            # Ensure session is closed even if cleanup fails
            if self._db_session:
                self._db_session.close()
                self._db_session = None
    
    def get_db_session(self) -> Optional[Session]:
        """Get the current database session for services that need it."""
        return self._db_session
    
    def is_ready(self) -> bool:
        """Check if all services are initialized."""
        return all([
            self.prompt_parser is not None,
            self.retrieval_service is not None,
            self.scoring_service is not None,
            self.metrics_collector is not None,
            self.cost_telemetry is not None,
            self._db_session is not None
        ])


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup and shutdown."""
    # Startup
    logger.info("Starting Leadpoet API service...")
    
    # Initialize service registry
    services = ServiceRegistry()
    await services.initialize_services()
    
    # Store in app state
    app.state.services = services
    
    logger.info("✅ Application startup completed")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Leadpoet API service...")
    
    # Cleanup services
    await services.cleanup()
    
    logger.info("✅ Application shutdown completed")


# Create FastAPI app
app = FastAPI(
    title="Leadpoet Intent Model v1.1",
    description="API for B2B lead scoring and ranking with < 400ms P95 latency",
    version="1.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ALLOWED_ORIGINS,
    allow_credentials=settings.CORS_ALLOW_CREDENTIALS,
    allow_methods=settings.CORS_ALLOWED_METHODS,
    allow_headers=settings.CORS_ALLOWED_HEADERS,
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=settings.TRUSTED_HOSTS
)

# Add Datadog APM middleware (Epic 7 Task 2)
app.add_middleware(DatadogAPMMiddleware)

app.add_middleware(TelemetryMiddleware)


# Pydantic models
class QueryRequest(BaseModel):
    """Request model for lead queries."""
    query: str
    desired_count: int = 10
    user_id: Optional[str] = "default-user" # Placeholder for user ID
    filters: Optional[Dict[str, Any]] = Field(default=None, description="Optional ICP filters")
    
    class Config:
        schema_extra = {
            "example": {
                "query": "Find companies evaluating CRM software with 100-500 employees in the US",
                "desired_count": 25,
                "filters": {
                    "industry": ["technology", "saas"],
                    "size": ["100-500"],
                    "region": ["us-west", "us-east"]
                }
            }
        }


class QueryPerformanceRequest(BaseModel):
    query_id: str
    ctr: float = Field(..., ge=0.0, le=1.0, description="Click-Through Rate for the query results")
    conversion_rate: float = Field(..., ge=0.0, le=1.0, description="Conversion Rate for the query results")


class LeadResponse(BaseModel):
    """Response model for individual leads."""
    lead_id: str
    company_id: str
    company_name: str
    email: str
    fit_score: float
    intent_score: float
    final_score: float
    explanation: Dict[str, Any]
    firmographics: Dict[str, Any]
    technographics: Dict[str, Any]


class QueryResponse(BaseModel):
    """Response model for query results."""
    query_id: str
    leads: List[LeadResponse]
    total_count: int
    processing_time_ms: float
    cost_usd: float
    lead_metadata: Dict[str, Any]


# Dependency injection functions
async def get_services(request: Request) -> ServiceRegistry:
    """Get service registry from app state."""
    services = getattr(request.app.state, 'services', None)
    if not services or not services.is_ready():
        raise HTTPException(status_code=503, detail="Services not ready")
    return services


async def get_db_session(request: Request) -> Session:
    """Get database session from service registry."""
    services = await get_services(request)
    db_session = services.get_db_session()
    if not db_session:
        raise HTTPException(status_code=503, detail="Database session not available")
    return db_session


async def get_prompt_parser(request: Request) -> PromptParser:
    """Get prompt parser service."""
    services = await get_services(request)
    return services.prompt_parser


async def get_retrieval_service(request: Request) -> RetrievalService:
    """Get retrieval service."""
    services = await get_services(request)
    return services.retrieval_service


async def get_scoring_service(request: Request) -> ScoringService:
    """Get scoring service."""
    services = await get_services(request)
    return services.scoring_service


async def get_metrics_collector(request: Request) -> MetricsCollector:
    """Get metrics collector."""
    services = await get_services(request)
    return services.metrics_collector


async def get_cost_telemetry(request: Request) -> CostTelemetry:
    """Get cost telemetry service."""
    services = await get_services(request)
    return services.cost_telemetry


# Health check endpoint
@app.get("/healthz")
async def health_check(request: Request):
    """Health check endpoint for load balancers and monitoring."""
    services = getattr(request.app.state, 'services', None)
    
    return {
        "status": "healthy" if services and services.is_ready() else "unhealthy",
        "timestamp": time.time(),
        "version": "1.1.0",
        "services": {
            "prompt_parser": services.prompt_parser is not None if services else False,
            "retrieval_service": services.retrieval_service is not None if services else False,
            "scoring_service": services.scoring_service is not None if services else False,
            "metrics_collector": services.metrics_collector is not None if services else False,
            "cost_telemetry": services.cost_telemetry is not None if services else False
        }
    }


# Main query endpoint
@app.post("/query", response_model=List[ScoredLead], tags=["Query"])
async def query(
    request: QueryRequest,
    db: Session = Depends(get_db),
    services: ServiceRegistry = Depends(get_services)):
    """
    Main endpoint to process a query and return scored leads.
    """
    logger.info(f"Received query from user '{request.user_id}': '{request.query}'")
    
    # Create and save the query record
    new_query = Query(id=str(uuid.uuid4()), user_id=request.user_id, query_text=request.query)
    db.add(new_query)
    db.commit()
    db.refresh(new_query)
    query_id = str(new_query.id)
    
    # Parse the user's prompt
    parsed_query = await services.prompt_parser.parse(prompt=request.query)
    
    # Retrieve candidates
    candidates = await services.retrieval_service.retrieve_candidates(
        parsed_query=parsed_query, 
        desired_count=request.desired_count
    )
    if not candidates:
        logger.warning("No candidates were retrieved.")
        return []
    
    # Score and rank the candidates
    scored_leads = await services.scoring_service.score_and_rank(
        query_id=query_id,
        candidates=candidates,
        parsed_query=parsed_query,
        desired_count=request.desired_count,
        lgbm_service=services.lgbm_service
    )
    
    return scored_leads


# Metrics endpoint
@app.get("/metrics")
async def metrics(request: Request) -> PlainTextResponse:
    """Prometheus metrics endpoint."""
    services = await get_services(request)
    metrics_data = services.metrics_collector.get_metrics()
    return PlainTextResponse(metrics_data, media_type="text/plain")


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Leadpoet Intent Model v1.1",
        "version": "1.1.0",
        "description": "B2B lead scoring and ranking API",
        "target_latency": "400ms P95",
        "docs": "/docs",
        "health": "/healthz",
        "metrics": "/metrics"
    }


@app.post("/track/performance", status_code=204, tags=["Performance Tracking"])
async def track_query_performance(
    request: QueryPerformanceRequest,
    db: Session = Depends(get_db)
):
    """
    Endpoint to receive query performance data (CTR, conversion rate)
    and update the performance score for a given query.
    """
    performance_service = QueryPerformanceService(db)
    performance_service.update_query_performance(
        query_id=request.query_id,
        ctr=request.ctr,
        conversion_rate=request.conversion_rate
    )
    return


@app.get("/query-quality/reputation", tags=["Query Quality Management"])
async def get_query_quality_reputation(
    user_id: Optional[str] = None,
    db: Session = Depends(get_db),
    services: ServiceRegistry = Depends(get_services)):
    """
    Get query quality reputation information.
    If user_id is provided, returns reputation for that specific user.
    Otherwise, returns summary statistics for all users.
    """
    try:
        reputation_service = QueryQualityReputationService(db, services.metrics_collector)
        
        if user_id:
            # Get specific user reputation
            reputation = reputation_service.get_reputation(user_id)
            if not reputation:
                raise HTTPException(status_code=404, detail=f"User {user_id} not found")
            
            throttle_info = reputation_service.get_throttle_info(user_id)
            
            return {
                "user_id": reputation.user_id,
                "query_quality_score": reputation.query_quality_score,
                "pass_rate": reputation.pass_rate,
                "flag_rate": reputation.flag_rate,
                "total_queries": reputation.total_queries,
                "passed_queries": reputation.passed_queries,
                "flagged_queries": reputation.flagged_queries,
                "throttle_info": throttle_info,
                "last_calculation_time": reputation.last_calculation_time.isoformat() if reputation.last_calculation_time else None,
                "thresholds": {
                    "quality_score_threshold": reputation_service.quality_score_threshold,
                    "flag_rate_threshold": reputation_service.flag_rate_threshold
                }
            }
        else:
            # Get summary for all users
            summary = reputation_service.get_reputation_summary()
            all_reputations = reputation_service.get_all_reputations(limit=100)
            all_throttles = reputation_service.get_all_throttles(limit=100)
            
            return {
                "summary": summary,
                "reputations": all_reputations,
                "throttles": all_throttles,
                "thresholds": {
                    "quality_score_threshold": reputation_service.quality_score_threshold,
                    "flag_rate_threshold": reputation_service.flag_rate_threshold
                }
            }
            
    except Exception as e:
        logger.error(f"Error getting query quality reputation: {e}")
        raise HTTPException(status_code=500, detail=f"Error retrieving query quality reputation: {str(e)}")


@app.get("/query-quality/alerts", tags=["Query Quality Management"])
async def get_query_quality_alerts(
    db: Session = Depends(get_db),
    services: ServiceRegistry = Depends(get_services)):
    """
    Get active query quality alerts for reputation drops and throttling.
    """
    try:
        reputation_service = QueryQualityReputationService(db, services.metrics_collector)
        
        all_reputations = reputation_service.get_all_reputations()
        all_throttles = reputation_service.get_all_throttles()
        
        alerts = []
        
        # Add low quality alerts
        for reputation_data in all_reputations:
            user_id = reputation_data['user_id']
            quality_score = reputation_data['query_quality_score']
            flag_rate = reputation_data['flag_rate']
            
            if (quality_score < reputation_service.alert_quality_score_threshold or 
                flag_rate > reputation_service.alert_flag_rate_threshold):
                alerts.append({
                    "type": "low_quality",
                    "severity": "warning" if quality_score < reputation_service.alert_quality_score_threshold else "critical",
                    "user_id": user_id,
                    "message": f"User {user_id} quality score dropped below threshold: {quality_score:.3f}",
                    "details": {
                        "query_quality_score": quality_score,
                        "pass_rate": reputation_data['pass_rate'],
                        "flag_rate": flag_rate,
                        "total_queries": reputation_data['total_queries']
                    },
                    "timestamp": reputation_data['updated_at']
                })
        
        # Add throttling alerts
        for throttle_data in all_throttles:
            if throttle_data['is_throttled']:
                alerts.append({
                    "type": "user_throttled",
                    "severity": "critical",
                    "user_id": throttle_data['user_id'],
                    "message": f"User {throttle_data['user_id']} has been throttled due to poor query quality",
                    "details": {
                        "throttle_reason": throttle_data['throttle_reason'],
                        "throttle_start_time": throttle_data['throttle_start_time'],
                        "throttle_end_time": throttle_data['throttle_end_time'],
                        "throttle_count": throttle_data['throttle_count']
                    },
                    "timestamp": throttle_data['updated_at']
                })
        
        return {
            "total_alerts": len(alerts),
            "alerts": alerts,
            "thresholds": {
                "quality_score_threshold": reputation_service.quality_score_threshold,
                "flag_rate_threshold": reputation_service.flag_rate_threshold,
                "alert_quality_score_threshold": reputation_service.alert_quality_score_threshold,
                "alert_flag_rate_threshold": reputation_service.alert_flag_rate_threshold
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting query quality alerts: {e}")
        raise HTTPException(status_code=500, detail=f"Error retrieving query quality alerts: {str(e)}")


@app.post("/query-quality/clear-expired-throttles", tags=["Query Quality Management"])
async def clear_expired_throttles(
    db: Session = Depends(get_db),
    services: ServiceRegistry = Depends(get_services)):
    """
    Clear expired throttles for all users.
    """
    try:
        reputation_service = QueryQualityReputationService(db, services.metrics_collector)
        cleared_count = reputation_service.clear_expired_throttles()
        
        return {
            "message": f"Cleared {cleared_count} expired throttles",
            "cleared_count": cleared_count
        }
        
    except Exception as e:
        logger.error(f"Error clearing expired throttles: {e}")
        raise HTTPException(status_code=500, detail=f"Error clearing expired throttles: {str(e)}")


@app.on_event("shutdown")
async def shutdown_event():
    # This is a placeholder for any cleanup logic
    pass


if __name__ == "__main__":
    import os

    uvicorn.run(
        "app.main:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        reload=os.getenv("ENV", "production") == "development",
        log_level=os.getenv("LOG_LEVEL", "info").lower()
    )