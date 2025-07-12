import time
import json
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response
from starlette.types import ASGIApp
from loguru import logger

from app.main import app  # Import the app to access the services

class TelemetryMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: ASGIApp):
        super().__init__(app)

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        start_time = time.time()

        # Process the request
        response = await call_next(request)

        # After response is created, calculate metrics
        process_time = time.time() - start_time
        
        # Get services from the application state with error handling
        try:
            metrics_collector = app.state.services.metrics_collector
            cost_telemetry = app.state.services.cost_telemetry
            
            # 1. Record basic request metrics
            metrics_collector.record_request(request.url.path, request.method, response.status_code)
            metrics_collector.record_latency(request.url.path, request.method, process_time)

            # 2. Record cost and LLM-specific metrics for the /query endpoint
            if request.url.path == "/query" and response.status_code == 200:
                # We need to access the response body to get the scored leads.
                # This is a bit tricky with streaming responses.
                response_body = b""
                async for chunk in response.body_iterator:
                    response_body += chunk
                
                try:
                    scored_leads = json.loads(response_body)
                    
                    # Check for LLM usage
                    if any(lead.get("llm_score") is not None for lead in scored_leads):
                        metrics_collector.record_llm_hit()

                    # Calculate and record cost
                    cost = cost_telemetry.calculate_cost(
                        scored_leads=scored_leads,
                        llm_call_count=1 if any(lead.get("llm_score") is not None for lead in scored_leads) else 0
                    )
                    if cost > 0:
                        # The cost is for the whole batch, let's record the per-lead average
                        avg_cost_per_lead = cost / len(scored_leads) if scored_leads else 0
                        metrics_collector.record_lead_cost(avg_cost_per_lead)
                        logger.info(f"Request to /query cost: ${cost:.6f} (${avg_cost_per_lead:.6f}/lead)")

                except json.JSONDecodeError:
                    logger.warning("Could not decode response body to calculate telemetry.")
                
                # Return a new response with the consumed body, so the client still gets it
                return Response(
                    content=response_body,
                    status_code=response.status_code,
                    headers=dict(response.headers),
                    media_type=response.media_type,
                )

        except AttributeError as e:
            logger.error(f"Service registry not properly initialized: {e}. Skipping telemetry collection.")
            return response

        return response 