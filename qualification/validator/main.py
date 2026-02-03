"""
Qualification System: Validator Main Loop

Phase 3.1 from tasks10.md

This module implements the QualificationValidator class that:
- Registers with the platform API
- Sends periodic heartbeats
- Requests evaluation work
- Executes models in TEE sandboxes
- Reports results back to the platform

CRITICAL: This is a completely NEW validator class for qualification only.
Do NOT modify neurons/validator.py or any existing validator code.
This runs as a separate, independent process.
"""

import os
import sys
import time
import json
import base64
import asyncio
import logging
import traceback
from uuid import UUID, uuid4
from datetime import datetime, timezone
from typing import Optional, Set, Dict, Any, List

import httpx

from gateway.qualification.config import CONFIG, QualificationConfig
from gateway.qualification.models import (
    ICPPrompt,
    LeadOutput,
    LeadScoreBreakdown,
    EvaluationRunStatus,
    ValidatorHeartbeat,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Error Codes
# =============================================================================

class ErrorCodes:
    """Standard error codes for evaluation failures."""
    UNKNOWN_ERROR = 1000
    MODEL_TIMEOUT = 1010
    MODEL_EXCEPTION = 1020
    MODEL_INVALID_OUTPUT = 1030
    SANDBOX_INIT_FAILED = 1040
    SANDBOX_CRASH = 1050
    LEAD_VALIDATION_FAILED = 1060
    SCORING_FAILED = 1070
    ICP_NOT_FOUND = 1080
    COST_LIMIT_EXCEEDED = 1090
    NETWORK_ERROR = 1100


# =============================================================================
# QualificationValidator Class
# =============================================================================

class QualificationValidator:
    """
    Validator for the Lead Qualification Agent competition.
    
    Runs as a separate process to:
    1. Register with the gateway platform
    2. Send periodic heartbeats
    3. Request evaluation work
    4. Execute models in TEE sandboxes
    5. Score leads and report results
    
    Attributes:
        config: QualificationConfig instance
        hotkey: Validator's Bittensor hotkey
        code_version: Git commit hash of validator code
        platform_url: Gateway platform API URL
        session_id: Registered session UUID
        current_evaluation: Currently running evaluation UUID
        running: Whether the validator is running
    """
    
    def __init__(
        self,
        config: Optional[QualificationConfig] = None,
        hotkey: Optional[str] = None,
        code_version: Optional[str] = None,
        platform_url: Optional[str] = None
    ):
        """
        Initialize the QualificationValidator.
        
        Args:
            config: Configuration object (defaults to global CONFIG)
            hotkey: Validator's Bittensor hotkey (from env if not provided)
            code_version: Git commit hash (from env if not provided)
            platform_url: Gateway platform URL (from env if not provided)
        """
        self.config = config or CONFIG
        
        # Validator identity
        self.hotkey = hotkey or os.getenv("VALIDATOR_HOTKEY", "")
        self.code_version = code_version or os.getenv("VALIDATOR_CODE_VERSION", "unknown")
        
        # Platform connection
        self.platform_url = platform_url or os.getenv(
            "QUALIFICATION_PLATFORM_URL",
            "http://localhost:8000"
        )
        
        # Session state
        self.session_id: Optional[UUID] = None
        self.current_evaluation: Optional[UUID] = None
        self.running = False
        
        # HTTP client (will be initialized in start())
        self._http_client: Optional[httpx.AsyncClient] = None
        
        # Sandbox reference (will hold current sandbox)
        self._current_sandbox = None
        
        logger.info(
            f"QualificationValidator initialized: "
            f"hotkey={self.hotkey[:16] if self.hotkey else 'NOT SET'}..., "
            f"platform={self.platform_url}"
        )
    
    # =========================================================================
    # Main Entry Point
    # =========================================================================
    
    async def start(self):
        """
        Main validator entry point.
        
        Registers with the platform and starts background loops for:
        - Heartbeats (every HEARTBEAT_INTERVAL_SECONDS)
        - Evaluation requests (every REQUEST_EVALUATION_INTERVAL_SECONDS)
        """
        logger.info("Starting QualificationValidator...")
        
        # Validate configuration
        if not self.hotkey:
            raise ValueError("VALIDATOR_HOTKEY environment variable not set")
        
        # Initialize HTTP client
        self._http_client = httpx.AsyncClient(
            base_url=self.platform_url,
            timeout=30.0
        )
        
        try:
            # Register with platform
            self.session_id = await self.register()
            logger.info(f"Registered with platform: session_id={self.session_id}")
            
            self.running = True
            
            # Start background tasks
            heartbeat_task = asyncio.create_task(
                self.heartbeat_loop(),
                name="heartbeat_loop"
            )
            evaluation_task = asyncio.create_task(
                self.evaluation_loop(),
                name="evaluation_loop"
            )
            
            # Wait for tasks (they run until stopped)
            await asyncio.gather(heartbeat_task, evaluation_task)
            
        except asyncio.CancelledError:
            logger.info("Validator tasks cancelled")
        except Exception as e:
            logger.error(f"Validator error: {e}")
            raise
        finally:
            self.running = False
            if self._http_client:
                await self._http_client.aclose()
    
    async def stop(self):
        """Stop the validator gracefully."""
        logger.info("Stopping QualificationValidator...")
        self.running = False
        
        # Cleanup current sandbox if any
        if self._current_sandbox:
            try:
                await self._current_sandbox.cleanup()
            except Exception as e:
                logger.warning(f"Error cleaning up sandbox: {e}")
    
    # =========================================================================
    # Platform Registration
    # =========================================================================
    
    async def register(self) -> UUID:
        """
        Register with the platform API.
        
        Returns:
            UUID: Session ID for this validator instance
        
        Raises:
            Exception: If registration fails
        """
        timestamp = int(time.time())
        signature = sign_timestamp(timestamp, self.hotkey)
        
        payload = {
            "timestamp": timestamp,
            "signed_timestamp": signature,
            "hotkey": self.hotkey,
            "commit_hash": self.code_version
        }
        
        logger.info(f"Registering with platform: {self.platform_url}/qualification/validator/register")
        
        response = await self._platform_request(
            "POST",
            "/qualification/validator/register",
            json=payload
        )
        
        if "session_id" not in response:
            raise Exception(f"Registration failed: {response}")
        
        return UUID(response["session_id"])
    
    # =========================================================================
    # Heartbeat Loop
    # =========================================================================
    
    async def heartbeat_loop(self):
        """
        Send periodic heartbeats to the platform.
        
        Heartbeats include:
        - Session ID
        - Current timestamp
        - System metrics (CPU, memory, etc.)
        - Currently running evaluation (if any)
        """
        logger.info(
            f"Starting heartbeat loop "
            f"(interval: {self.config.HEARTBEAT_INTERVAL_SECONDS}s)"
        )
        
        while self.running:
            try:
                heartbeat = ValidatorHeartbeat(
                    session_id=self.session_id,
                    timestamp=int(time.time()),
                    system_metrics=get_system_metrics(),
                    current_evaluation=self.current_evaluation
                )
                
                await self._platform_request(
                    "POST",
                    "/qualification/validator/heartbeat",
                    json=heartbeat.model_dump(mode="json")
                )
                
                logger.debug(f"Heartbeat sent: evaluation={self.current_evaluation}")
                
            except Exception as e:
                logger.warning(f"Heartbeat failed: {e}")
            
            await asyncio.sleep(self.config.HEARTBEAT_INTERVAL_SECONDS)
    
    # =========================================================================
    # Evaluation Loop
    # =========================================================================
    
    async def evaluation_loop(self):
        """
        Request and execute evaluations.
        
        Continuously polls the platform for work, executes evaluations
        in TEE sandboxes, and reports results.
        """
        logger.info(
            f"Starting evaluation loop "
            f"(interval: {self.config.REQUEST_EVALUATION_INTERVAL_SECONDS}s)"
        )
        
        while self.running:
            try:
                # Request work from platform
                work = await self._platform_request(
                    "POST",
                    "/qualification/validator/request-evaluation",
                    json={"session_id": str(self.session_id)}
                )
                
                if work.get("evaluation_runs"):
                    # Got work to do
                    logger.info(
                        f"Received evaluation work: "
                        f"{len(work['evaluation_runs'])} runs"
                    )
                    await self.execute_evaluation(work)
                else:
                    # No work available, wait before polling again
                    logger.debug("No evaluation work available")
                    await asyncio.sleep(self.config.REQUEST_EVALUATION_INTERVAL_SECONDS)
                    
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error(f"Evaluation loop error: {e}\n{traceback.format_exc()}")
                await asyncio.sleep(5)  # Brief pause before retry
    
    # =========================================================================
    # Evaluation Execution
    # =========================================================================
    
    async def execute_evaluation(self, work: Dict[str, Any]):
        """
        Execute a model evaluation in TEE sandbox.
        
        Processes each evaluation run (ICP):
        1. Initialize sandbox with model code
        2. Run model with ICP prompt
        3. Validate returned lead
        4. Score lead against ICP
        5. Report results
        
        Args:
            work: Work assignment from platform containing:
                - agent_code: Base64-encoded model code
                - evaluation_runs: List of runs to execute
                - evaluation_id: UUID of this evaluation
        """
        # Decode model code
        agent_code_b64 = work.get("agent_code", "")
        if agent_code_b64:
            model_code = base64.b64decode(agent_code_b64)
        else:
            model_code = b""
        
        runs = work.get("evaluation_runs", [])
        evaluation_id = UUID(work["evaluation_id"])
        
        logger.info(
            f"Starting evaluation: id={evaluation_id}, "
            f"runs={len(runs)}, code_size={len(model_code)} bytes"
        )
        
        # Track current evaluation for heartbeat reporting
        self.current_evaluation = evaluation_id
        
        # Track seen companies for duplicate handling (reset per evaluation)
        seen_companies: Set[str] = set()
        
        # Initialize sandbox once for all runs
        sandbox = None
        
        try:
            for run in runs:
                evaluation_run_id = UUID(run["evaluation_run_id"])
                probe_id = run.get("probe_id") or run.get("probe_name")
                stage = run.get("stage", "final")
                
                logger.info(f"Processing run: id={evaluation_run_id}, probe={probe_id}, stage={stage}")
                
                try:
                    # -----------------------------------------------------
                    # Step 1: Initialize sandbox
                    # -----------------------------------------------------
                    await self.update_run_status(evaluation_run_id, "initializing_model")
                    
                    if sandbox is None:
                        sandbox = await self.create_tee_sandbox(
                            model_code=model_code,
                            evaluation_run_id=evaluation_run_id
                        )
                        self._current_sandbox = sandbox
                    
                    # -----------------------------------------------------
                    # Step 2: Get ICP prompt
                    # -----------------------------------------------------
                    icp = await self.get_icp(probe_id)
                    
                    if not icp:
                        await self.report_error(
                            evaluation_run_id,
                            ErrorCodes.ICP_NOT_FOUND,
                            f"ICP not found: {probe_id}"
                        )
                        continue
                    
                    # -----------------------------------------------------
                    # Step 3: Run model with timeout
                    # -----------------------------------------------------
                    await self.update_run_status(evaluation_run_id, "running_model")
                    start_time = time.time()
                    
                    try:
                        result = await asyncio.wait_for(
                            sandbox.run_model(icp),
                            timeout=self.config.RUNNING_MODEL_TIMEOUT_SECONDS
                        )
                    except asyncio.TimeoutError:
                        await self.report_error(
                            evaluation_run_id,
                            ErrorCodes.MODEL_TIMEOUT,
                            f"Model timeout exceeded ({self.config.RUNNING_MODEL_TIMEOUT_SECONDS}s)"
                        )
                        continue
                    
                    run_time = time.time() - start_time
                    run_cost = await self.get_run_cost(evaluation_run_id)
                    
                    logger.info(
                        f"Model completed: time={run_time:.2f}s, cost=${run_cost:.4f}"
                    )
                    
                    # -----------------------------------------------------
                    # Step 4: Validate lead
                    # -----------------------------------------------------
                    await self.update_run_status(evaluation_run_id, "validating_lead")
                    
                    lead_data = result.get("lead") if isinstance(result, dict) else None
                    lead = None
                    lead_valid = False
                    
                    if lead_data:
                        try:
                            # Parse lead output
                            lead = LeadOutput(**lead_data)
                            lead_valid = await self.validate_lead(lead, icp)
                        except Exception as e:
                            logger.warning(f"Lead parsing/validation failed: {e}")
                            lead_valid = False
                    else:
                        logger.info("Model returned no lead")
                    
                    # -----------------------------------------------------
                    # Step 5: Score lead
                    # -----------------------------------------------------
                    await self.update_run_status(evaluation_run_id, "scoring")
                    
                    scores = None
                    if lead and lead_valid:
                        scores = await self.score_lead(
                            lead=lead,
                            icp=icp,
                            run_cost_usd=run_cost,
                            run_time_seconds=run_time,
                            seen_companies=seen_companies
                        )
                    else:
                        # Create zero score for invalid/missing lead
                        scores = LeadScoreBreakdown(
                            icp_fit=0,
                            decision_maker=0,
                            intent_signal_raw=0,
                            time_decay_multiplier=1.0,
                            intent_signal_final=0,
                            cost_penalty=0,
                            time_penalty=0,
                            final_score=0,
                            failure_reason="No valid lead returned" if not lead else "Lead validation failed"
                        )
                    
                    # -----------------------------------------------------
                    # Step 6: Report results
                    # -----------------------------------------------------
                    await self.report_results(
                        evaluation_run_id=evaluation_run_id,
                        lead=lead,
                        scores=scores,
                        run_cost_usd=run_cost,
                        run_time_seconds=run_time
                    )
                    
                    logger.info(
                        f"Run completed: id={evaluation_run_id}, "
                        f"score={scores.final_score if scores else 0}"
                    )
                    
                except asyncio.TimeoutError:
                    await self.report_error(
                        evaluation_run_id,
                        ErrorCodes.MODEL_TIMEOUT,
                        "Model execution timeout"
                    )
                except Exception as e:
                    logger.error(f"Run error: {e}\n{traceback.format_exc()}")
                    await self.report_error(
                        evaluation_run_id,
                        ErrorCodes.UNKNOWN_ERROR,
                        str(e)
                    )
                    
        finally:
            # Cleanup
            self.current_evaluation = None
            self._current_sandbox = None
            
            if sandbox:
                try:
                    await sandbox.cleanup()
                except Exception as e:
                    logger.warning(f"Sandbox cleanup error: {e}")
            
            logger.info(f"Evaluation completed: id={evaluation_id}")
    
    # =========================================================================
    # Helper Methods
    # =========================================================================
    
    async def _platform_request(
        self,
        method: str,
        endpoint: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Make a request to the platform API.
        
        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint path
            **kwargs: Additional arguments for httpx
        
        Returns:
            Response JSON as dict
        
        Raises:
            Exception: If request fails
        """
        try:
            response = await self._http_client.request(method, endpoint, **kwargs)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            logger.error(f"Platform request failed: {e.response.status_code} {e.response.text}")
            raise
        except Exception as e:
            logger.error(f"Platform request error: {e}")
            raise
    
    async def create_tee_sandbox(self, model_code: bytes, evaluation_run_id: UUID):
        """
        Create a TEE sandbox for model execution.
        
        Args:
            model_code: Model code as bytes
            evaluation_run_id: UUID of the evaluation run
        
        Returns:
            TEESandbox instance (started)
        """
        from qualification.validator.sandbox import TEESandbox
        
        # Create sandbox with API proxy URL for cost tracking
        # NOTE: api_proxy_url is the API Proxy Gateway, NOT Webshare proxy
        api_proxy_url = f"{self.platform_url}/qualification/proxy"
        
        sandbox = TEESandbox(
            model_code=model_code,
            evaluation_run_id=evaluation_run_id,
            api_proxy_url=api_proxy_url,
            evaluation_id=self.current_evaluation
        )
        
        await sandbox.start()
        return sandbox
    
    async def update_run_status(self, evaluation_run_id: UUID, status: str):
        """
        Update the status of an evaluation run.
        
        Args:
            evaluation_run_id: UUID of the run
            status: New status (from EvaluationRunStatus)
        
        TODO: Implement platform API call
        """
        logger.debug(f"Run status update: {evaluation_run_id} -> {status}")
        
        # Placeholder - in production, call platform API
        # await self._platform_request(
        #     "POST",
        #     "/qualification/validator/update-run-status",
        #     json={
        #         "evaluation_run_id": str(evaluation_run_id),
        #         "status": status,
        #         "timestamp": int(time.time())
        #     }
        # )
    
    async def get_icp(self, probe_id: str) -> Optional[ICPPrompt]:
        """
        Get an ICP prompt by ID.
        
        Args:
            probe_id: The ICP identifier
        
        Returns:
            ICPPrompt or None if not found
        
        TODO: Implement ICP lookup
        """
        # Placeholder - reuse from status module
        from gateway.qualification.api.status import get_icp as _get_icp
        return await _get_icp(probe_id)
    
    async def get_run_cost(self, evaluation_run_id: UUID) -> float:
        """
        Get the accumulated API cost for a run.
        
        The cost is tracked by the LOCAL PROXY running in the sandbox.
        The proxy intercepts all API calls and tracks actual costs from:
        - OpenRouter: usage.cost from response
        - ScrapingDog: credits calculated based on endpoint/params
        
        Args:
            evaluation_run_id: UUID of the run
        
        Returns:
            Total cost in USD
        """
        # Get cost from the sandbox's proxy (authoritative source)
        if self._current_sandbox:
            return self._current_sandbox.get_last_run_cost()
        return 0.0
    
    async def validate_lead(self, lead: LeadOutput, icp: ICPPrompt) -> bool:
        """
        Validate a lead against basic requirements.
        
        Args:
            lead: The lead to validate
            icp: The ICP it should match
        
        Returns:
            True if lead passes basic validation
        
        TODO: Implement lead validation
        """
        # Basic validation - check required fields exist
        # NOTE: email and full_name are NOT allowed (models cannot fabricate PII)
        if not lead.business:
            return False
        
        if not lead.role or not lead.industry or not lead.sub_industry:
            return False
        
        if not lead.intent_signal:
            return False
        
        return True
    
    async def score_lead(
        self,
        lead: LeadOutput,
        icp: ICPPrompt,
        run_cost_usd: float,
        run_time_seconds: float,
        seen_companies: Set[str]
    ) -> LeadScoreBreakdown:
        """
        Score a lead against an ICP using the full scoring pipeline.
        
        This calls the LLM-based scoring system which:
        1. Runs automatic-zero pre-checks
        2. Scores ICP fit (0-20 pts) via LLM
        3. Scores decision maker (0-30 pts) via LLM
        4. Verifies and scores intent signal (0-50 pts) via LLM
        5. Applies time decay to intent signal
        6. Calculates penalties (cost × 1000 + time × 1)
        
        Args:
            lead: The lead to score
            icp: The ICP to score against
            run_cost_usd: API cost for this run
            run_time_seconds: Execution time
            seen_companies: Set of companies already scored (for duplicate detection)
        
        Returns:
            LeadScoreBreakdown with detailed scores
        
        Requires:
            OPENROUTER_API_KEY environment variable for LLM scoring
            SCRAPINGDOG_API_KEY environment variable for intent verification
        """
        # Use the full scoring pipeline from qualification.scoring
        try:
            from qualification.scoring.lead_scorer import score_lead as _score_lead
            
            logger.info(f"Scoring lead: {lead.business} / {lead.role} against ICP {icp.icp_id}")
            
            scores = await _score_lead(
                lead=lead,
                icp=icp,
                run_cost_usd=run_cost_usd,
                run_time_seconds=run_time_seconds,
                seen_companies=seen_companies
            )
            
            logger.info(
                f"Lead scored: {scores.final_score:.2f} "
                f"(ICP:{scores.icp_fit:.1f}, DM:{scores.decision_maker:.1f}, "
                f"Intent:{scores.intent_signal_final:.1f})"
            )
            
            return scores
            
        except ImportError as e:
            logger.error(f"Failed to import scoring module: {e}")
            # Fallback to basic scoring if module not available
            return self._basic_score_lead(lead, icp, run_cost_usd, run_time_seconds)
            
        except Exception as e:
            logger.error(f"Scoring failed: {e}")
            # Return zero score with failure reason
            return LeadScoreBreakdown(
                icp_fit=0,
                decision_maker=0,
                intent_signal_raw=0,
                time_decay_multiplier=1.0,
                intent_signal_final=0,
                cost_penalty=0,
                time_penalty=0,
                final_score=0,
                failure_reason=f"Scoring error: {str(e)[:100]}"
            )
    
    def _basic_score_lead(
        self,
        lead: LeadOutput,
        icp: ICPPrompt,
        run_cost_usd: float,
        run_time_seconds: float
    ) -> LeadScoreBreakdown:
        """
        Basic scoring fallback when full scoring is unavailable.
        
        Uses simple heuristics instead of LLM:
        - ICP Fit: Check industry/role match
        - Decision Maker: Check seniority
        - Intent Signal: Check if present and recent
        """
        logger.warning("Using basic scoring fallback (no LLM)")
        
        # Basic ICP fit (check industry match)
        icp_fit = 10.0 if lead.industry.lower() in icp.industry.lower() else 5.0
        
        # Basic decision maker (check seniority)
        dm_score = 15.0
        if lead.seniority:
            seniority_lower = lead.seniority.lower()
            if 'c-suite' in seniority_lower or 'vp' in seniority_lower:
                dm_score = 25.0
            elif 'director' in seniority_lower:
                dm_score = 20.0
        
        # Basic intent signal (check if present)
        intent_score = 0.0
        if lead.intent_signal and lead.intent_signal.url:
            intent_score = 25.0  # Give partial credit for having a signal
        
        # Calculate penalties
        cost_penalty = run_cost_usd * self.config.COST_PENALTY_MULTIPLIER
        time_penalty = run_time_seconds * self.config.TIME_PENALTY_MULTIPLIER
        
        total = icp_fit + dm_score + intent_score
        final_score = max(0, total - cost_penalty - time_penalty)
        
        return LeadScoreBreakdown(
            icp_fit=icp_fit,
            decision_maker=dm_score,
            intent_signal_raw=intent_score,
            time_decay_multiplier=1.0,
            intent_signal_final=intent_score,
            cost_penalty=cost_penalty,
            time_penalty=time_penalty,
            final_score=final_score,
            failure_reason="Basic scoring fallback (no LLM available)"
        )
    
    async def report_results(
        self,
        evaluation_run_id: UUID,
        lead: Optional[LeadOutput],
        scores: Optional[LeadScoreBreakdown],
        run_cost_usd: float,
        run_time_seconds: float
    ):
        """
        Report evaluation results to the platform.
        
        Args:
            evaluation_run_id: UUID of the run
            lead: The lead returned (may be None)
            scores: Score breakdown (may be None)
            run_cost_usd: Total API cost
            run_time_seconds: Total execution time
        
        TODO: Implement platform API call
        """
        logger.info(
            f"Reporting results: run={evaluation_run_id}, "
            f"score={scores.final_score if scores else 0:.2f}"
        )
        
        # Placeholder - in production, call platform API
        # await self._platform_request(
        #     "POST",
        #     "/qualification/validator/report-results",
        #     json={
        #         "evaluation_run_id": str(evaluation_run_id),
        #         "lead_returned": lead.model_dump() if lead else None,
        #         "lead_score": scores.model_dump() if scores else None,
        #         "icp_fit_score": scores.icp_fit if scores else 0,
        #         "decision_maker_score": scores.decision_maker if scores else 0,
        #         "intent_signal_score": scores.intent_signal_final if scores else 0,
        #         "cost_penalty": scores.cost_penalty if scores else 0,
        #         "time_penalty": scores.time_penalty if scores else 0,
        #         "final_lead_score": scores.final_score if scores else 0,
        #         "run_cost_usd": run_cost_usd,
        #         "run_time_seconds": run_time_seconds,
        #         "status": "finished"
        #     }
        # )
    
    async def report_error(
        self,
        evaluation_run_id: UUID,
        error_code: int,
        error_message: str
    ):
        """
        Report an error for an evaluation run.
        
        Args:
            evaluation_run_id: UUID of the run
            error_code: Error code from ErrorCodes
            error_message: Human-readable error message
        
        TODO: Implement platform API call
        """
        logger.error(
            f"Reporting error: run={evaluation_run_id}, "
            f"code={error_code}, message={error_message}"
        )
        
        # Placeholder - in production, call platform API
        # await self._platform_request(
        #     "POST",
        #     "/qualification/validator/report-error",
        #     json={
        #         "evaluation_run_id": str(evaluation_run_id),
        #         "error_code": error_code,
        #         "error_message": error_message,
        #         "status": "error"
        #     }
        # )


# =============================================================================
# Placeholder Sandbox (to be replaced in Phase 3.2)
# =============================================================================

class PlaceholderSandbox:
    """Placeholder sandbox for testing before TEE implementation."""
    
    def __init__(self, model_code: bytes, evaluation_run_id: UUID):
        self.model_code = model_code
        self.evaluation_run_id = evaluation_run_id
    
    async def run_model(self, icp: ICPPrompt) -> Dict[str, Any]:
        """Run model and return result."""
        logger.warning("PLACEHOLDER: PlaceholderSandbox.run_model - implement TEE sandbox")
        
        # Return empty result for now
        return {"lead": None}
    
    async def cleanup(self):
        """Cleanup sandbox resources."""
        pass


# =============================================================================
# Placeholder Functions
# =============================================================================

def sign_timestamp(timestamp: int, hotkey: str) -> str:
    """
    Sign a timestamp with the validator's hotkey.
    
    Args:
        timestamp: Unix timestamp to sign
        hotkey: Validator's hotkey
    
    Returns:
        Hex-encoded signature
    
    TODO: Implement actual sr25519 signing
    """
    # Placeholder - in production, use sr25519 signing
    # from substrateinterface import Keypair
    # 
    # keypair = Keypair.create_from_uri(hotkey)
    # message = str(timestamp).encode()
    # signature = keypair.sign(message)
    # return signature.hex()
    
    logger.warning("PLACEHOLDER: sign_timestamp - implement sr25519 signing")
    
    # Return placeholder signature
    import hashlib
    return hashlib.sha256(f"{timestamp}:{hotkey}".encode()).hexdigest()


def get_system_metrics() -> Dict[str, Any]:
    """
    Get system metrics for heartbeat reporting.
    
    Returns:
        Dict with CPU, memory, disk metrics
    """
    try:
        import psutil
        
        return {
            "cpu_percent": psutil.cpu_percent(interval=0.1),
            "memory_percent": psutil.virtual_memory().percent,
            "memory_available_gb": psutil.virtual_memory().available / (1024**3),
            "disk_percent": psutil.disk_usage('/').percent,
            "timestamp": int(time.time())
        }
    except ImportError:
        logger.warning("psutil not installed - returning empty metrics")
        return {"timestamp": int(time.time())}
    except Exception as e:
        logger.warning(f"Failed to get system metrics: {e}")
        return {"timestamp": int(time.time()), "error": str(e)}


# =============================================================================
# CLI Entry Point
# =============================================================================

async def main():
    """Main entry point for running the validator."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Qualification Validator")
    parser.add_argument("--hotkey", help="Validator hotkey")
    parser.add_argument("--platform-url", help="Platform API URL")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()
    
    # Configure logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Create and start validator
    validator = QualificationValidator(
        hotkey=args.hotkey,
        platform_url=args.platform_url
    )
    
    try:
        await validator.start()
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        await validator.stop()


if __name__ == "__main__":
    asyncio.run(main())
