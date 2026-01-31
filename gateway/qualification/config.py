"""
Qualification System Configuration Module

All configurable variables for the Lead Qualification Agent competition.
Values can be overridden via environment variables.

See business_files/FINAL Lead Qualification IM BRD.md for requirements.
"""

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class QualificationConfig:
    """
    Configuration for the Lead Qualification Agent competition.
    
    All values have sensible defaults but can be overridden via:
    1. Environment variables (for operational tuning)
    2. Direct instantiation (for testing)
    """
    
    # =========================================================================
    # ICP Configuration
    # =========================================================================
    TOTAL_ICPS: int = 100  # Total ICPs per evaluation set
    LEADS_PER_ICP: int = 1  # Leads model must return per ICP
    SCREENING_1_ICPS: int = 5  # ICPs in screening 1 (quick filter)
    SCREENING_2_ICPS: int = 20  # ICPs in screening 2 (deeper filter)
    FINAL_BENCHMARK_ICPS: int = 75  # ICPs in final benchmark
    
    # =========================================================================
    # Evaluation Set Rotation
    # =========================================================================
    EVALUATION_SET_ROTATION_EPOCHS: int = 20  # Epochs between ICP set changes
    
    # =========================================================================
    # Cost/Time Limits (per-lead base values - totals are computed dynamically)
    # =========================================================================
    MAX_COST_PER_LEAD_USD: float = 0.05  # $0.05 average per lead (total = leads × $0.05)
    MAX_TIME_PER_LEAD_SECONDS: float = 8.0  # 8s average per lead (total = leads × 8s)
    RUNNING_MODEL_TIMEOUT_SECONDS: int = 30  # 30s HARD max per single lead - if exceeded, INSTANT FAIL
    TOTAL_EVALUATION_TIMEOUT_MINUTES: int = 60  # 1 hour absolute max (safety net)
    
    # =========================================================================
    # Screening Thresholds
    # =========================================================================
    SCREENING_1_THRESHOLD: float = 0.20  # 20% of max score to pass screening 1
    SCREENING_2_THRESHOLD: float = 0.40  # 40% of max score to pass screening 2
    PRUNE_THRESHOLD: float = 0.10  # 10% of champion score (below = pruned from queue)
    
    # =========================================================================
    # Champion Rules
    # =========================================================================
    CHAMPION_DETHRONING_THRESHOLD_PCT: float = 0.02  # >2% better to dethrone champion
    MIN_CHAMPION_DURATION_EPOCHS: int = 1  # Minimum epochs as champion before can be dethroned
    
    # =========================================================================
    # Champion Rebenchmark Timing (UTC)
    # =========================================================================
    # Champion is rebenchmarked on the FIRST FULL epoch that STARTS after this time
    # This aligns with when the ICP set is refreshed (default: 12:00 AM UTC)
    # We use 12:05 AM to ensure the new ICP set is fully propagated
    #
    # Example: If set to hour=0, minute=5 (12:05 AM):
    #   - Epoch spanning 11:30 PM → 12:42 AM does NOT trigger (started before 12:05 AM)
    #   - Epoch starting 12:42 AM → 1:54 AM DOES trigger (started after 12:05 AM)
    # =========================================================================
    CHAMPION_REBENCHMARK_HOUR_UTC: int = 0  # Hour in UTC (0-23), 0 = 12 AM
    CHAMPION_REBENCHMARK_MINUTE_UTC: int = 5  # Minute (0-59), 5 = 12:05 AM
    
    # =========================================================================
    # Qualification Model Queue
    # =========================================================================
    # Maximum models to pull from queue per epoch for evaluation
    # Models are pulled FIFO (oldest created_at first) where evaluated_at is NULL
    # Remaining models stay in queue for next epoch
    #
    # Example: If 34 models pending and MAX_MODELS_PER_EPOCH=10:
    #   - Epoch N: Pull and evaluate models 1-10 (oldest 10)
    #   - Epoch N+1: Pull and evaluate models 11-20
    #   - Epoch N+2: Pull and evaluate models 21-30
    #   - Epoch N+3: Pull and evaluate models 31-34 (remaining 4)
    # =========================================================================
    MAX_MODELS_PER_EPOCH: int = 10  # Max models to evaluate per epoch
    
    # =========================================================================
    # Qualification Block Cutoff
    # =========================================================================
    # No new qualification evaluations should START after this block number
    # This ensures evaluations complete before the epoch ends (360 blocks total)
    #
    # Example: If cutoff is 280, evaluations have 80 blocks (~16 minutes) to complete
    # Recommended: 280 (gives ~80 blocks buffer before epoch end)
    # =========================================================================
    QUALIFICATION_BLOCK_CUTOFF: int = 320  # Don't start evaluations after this block (TEMP: was 280)
    
    # =========================================================================
    # Submission Rules
    # =========================================================================
    SUBMISSION_COST_USD: float = 5.00  # $5 TAO to submit a model
    MIN_TIME_BETWEEN_SUBMISSIONS_EPOCHS: int = 20  # Minimum epochs between submissions
    MAX_SUBMISSIONS_PER_SET: int = 1  # Submissions allowed per hotkey per evaluation set
    
    # =========================================================================
    # Intent Signal Decay
    # =========================================================================
    INTENT_SIGNAL_DECAY_50_PCT_MONTHS: int = 2  # 50% decay after 2 months
    INTENT_SIGNAL_DECAY_25_PCT_MONTHS: int = 12  # 25% decay after 12 months
    
    # =========================================================================
    # Scoring Weights
    # =========================================================================
    MAX_POINTS_PER_LEAD: int = 100  # Maximum score per lead
    ICP_FIT_POINTS: int = 20  # Points for ICP fit (0-20)
    DECISION_MAKER_POINTS: int = 30  # Points for decision-maker quality (0-30)
    INTENT_SIGNAL_POINTS: int = 50  # Points for intent signal (0-50)
    
    # =========================================================================
    # Cost/Time Variability Penalties
    # =========================================================================
    # NEW SYSTEM: No penalty if within budget, small penalty for high variability
    #
    # - NO penalty if cost ≤ MAX_COST_PER_LEAD_USD ($0.05)
    # - NO penalty if time ≤ MAX_TIME_PER_LEAD_SECONDS (8s)
    # - 5-point penalty if cost > 2× MAX_COST_PER_LEAD_USD ($0.10)
    # - 5-point penalty if time > 2× MAX_TIME_PER_LEAD_SECONDS (16s)
    #
    # These thresholds are DYNAMIC: if you change MAX_COST_PER_LEAD_USD from
    # $0.05 to $0.10, the penalty threshold automatically becomes $0.20.
    # =========================================================================
    VARIABILITY_PENALTY_POINTS: int = 5  # Points deducted for high-variability leads
    COST_VARIABILITY_THRESHOLD_MULTIPLIER: float = 2.0  # Penalty if cost > 2× average
    TIME_VARIABILITY_THRESHOLD_MULTIPLIER: float = 2.0  # Penalty if time > 2× average
    
    # DEPRECATED - kept for backwards compatibility but no longer used in scoring
    COST_PENALTY_MULTIPLIER: int = 1000  # OLD: Penalty = 1000 × run_cost_usd (DEPRECATED)
    TIME_PENALTY_MULTIPLIER: int = 1  # OLD: Penalty = 1 × run_time_seconds (DEPRECATED)
    
    # =========================================================================
    # Validator Configuration
    # =========================================================================
    HEARTBEAT_TIMEOUT_SECONDS: int = 120  # Mark validator offline after 2 min no heartbeat
    HEARTBEAT_INTERVAL_SECONDS: int = 30  # Send heartbeat every 30 seconds
    REQUEST_EVALUATION_INTERVAL_SECONDS: int = 10  # Poll for work every 10 seconds
    
    # =========================================================================
    # Emissions
    # =========================================================================
    QUALIFICATION_EMISSIONS_PCT: float = 0.05  # 5% of total subnet emissions
    
    # =========================================================================
    # Fuzzy Matching Thresholds (for automatic-zero checks)
    # =========================================================================
    INDUSTRY_MATCH_THRESHOLD: int = 80  # 80% fuzzy match for industry
    SUB_INDUSTRY_MATCH_THRESHOLD: int = 70  # 70% fuzzy match for sub-industry
    ROLE_MATCH_THRESHOLD: int = 60  # 60% fuzzy match for role (more lenient)
    
    # =========================================================================
    # Intent Verification
    # =========================================================================
    INTENT_VERIFICATION_CONFIDENCE_THRESHOLD: int = 70  # 70% LLM confidence required
    INTENT_CACHE_TTL_DAYS: int = 7  # Cache verified intents for 7 days
    
    # =========================================================================
    # API Proxy
    # =========================================================================
    API_PROXY_TIMEOUT_SECONDS: int = 30  # Timeout for proxied API calls
    
    # =========================================================================
    # ScrapingDog Pricing Configuration
    # =========================================================================
    # Plan: $2,000/month for 42,000,000 credits
    # Cost per credit = $2000 / 42,000,000 = $0.0000476 USD
    #
    # Update these values when upgrading/downgrading the ScrapingDog plan.
    # The cost per credit is computed dynamically: plan_cost / plan_credits
    # =========================================================================
    SCRAPINGDOG_PLAN_COST_USD: float = 2000.00  # Monthly plan cost
    SCRAPINGDOG_PLAN_CREDITS: int = 42_000_000  # Credits included in plan
    
    def get_scrapingdog_cost_per_credit(self) -> float:
        """
        Calculate USD cost per ScrapingDog credit.
        
        Formula: SCRAPINGDOG_PLAN_COST_USD / SCRAPINGDOG_PLAN_CREDITS
        
        Example: $2000 / 42,000,000 = $0.0000476 per credit
        """
        if self.SCRAPINGDOG_PLAN_CREDITS <= 0:
            return 0.0
        return self.SCRAPINGDOG_PLAN_COST_USD / self.SCRAPINGDOG_PLAN_CREDITS
    
    # =========================================================================
    # Desearch API Pricing Configuration
    # =========================================================================
    # From https://desearch.ai/pricing (as of 2026):
    #   - X (Twitter) API: $0.30 / 1000 posts = $0.0003 per post/result
    #   - Web API: $0.25 / 100 searches = $0.0025 per search
    #   - AI Search API: $0.40 / 100 searches = $0.004 per search
    #
    # These values can be updated if Desearch changes pricing.
    # =========================================================================
    DESEARCH_X_SEARCH_COST_PER_1000: float = 0.30  # $0.30 per 1000 posts
    DESEARCH_WEB_SEARCH_COST_PER_100: float = 0.25  # $0.25 per 100 searches
    DESEARCH_AI_SEARCH_COST_PER_100: float = 0.40  # $0.40 per 100 searches
    
    def get_desearch_x_cost_per_call(self, num_results: int = 10) -> float:
        """
        Calculate cost for a Desearch X (Twitter) search.
        
        Args:
            num_results: Number of results returned (default 10)
        
        Returns:
            Cost in USD
        """
        cost_per_result = self.DESEARCH_X_SEARCH_COST_PER_1000 / 1000
        return cost_per_result * num_results
    
    def get_desearch_web_cost_per_call(self) -> float:
        """
        Calculate cost for a Desearch Web search.
        
        Returns:
            Cost in USD (fixed per search)
        """
        return self.DESEARCH_WEB_SEARCH_COST_PER_100 / 100
    
    def get_desearch_ai_cost_per_call(self) -> float:
        """
        Calculate cost for a Desearch AI Search.
        
        Returns:
            Cost in USD (fixed per search)
        """
        return self.DESEARCH_AI_SEARCH_COST_PER_100 / 100
    
    # =========================================================================
    # Hardcoding Detection Configuration
    # =========================================================================
    # Pre-execution analysis using a reasoning LLM to detect:
    # - Hardcoded answers (lookup tables, pre-computed results)
    # - Hard steering (gaming evaluation with specific patterns)
    # - Malicious code patterns
    #
    # Available models (in order of cost):
    # Uses o3-mini (200K context, good reasoning, consistent results)
    # =========================================================================
    
    # Confidence threshold for rejection (0-100)
    # Models with confidence >= this value are REJECTED as likely hardcoded
    # 70 = fairly confident it's hardcoded
    # 85 = very confident (more lenient - only reject obvious cases)
    HARDCODING_REJECTION_THRESHOLD: int = 70
    
    # Enable/disable hardcoding detection (useful for testing)
    ENABLE_HARDCODING_DETECTION: bool = True
    
    # Maximum time to wait for hardcoding analysis (seconds)
    HARDCODING_DETECTION_TIMEOUT: int = 120
    
    # Maximum total submission size for analysis (bytes)
    # ALL files count: .py, .md, .txt, .json, requirements.txt, etc.
    # 200KB ≈ 50K tokens, well within o3-mini's 200K context
    # Miner decides allocation: big README = less space for code
    # Models exceeding this are REJECTED (prevents hiding code beyond LLM context)
    HARDCODING_MAX_SUBMISSION_SIZE_BYTES: int = 200_000
    
    # =========================================================================
    # S3/Storage
    # =========================================================================
    MODEL_S3_BUCKET: str = "leadpoet-leads-primary"  # Uses qualification/ prefix
    MODEL_CODE_MAX_SIZE_MB: int = 10  # Max model code size (10 MB)
    PRESIGN_EXPIRES_SECONDS: int = 900  # Presigned URL expiration (15 minutes)
    
    def get_model_max_size_bytes(self) -> int:
        """Convert MODEL_CODE_MAX_SIZE_MB to bytes."""
        return self.MODEL_CODE_MAX_SIZE_MB * 1024 * 1024
    
    # =========================================================================
    # Platform URLs (for validators)
    # =========================================================================
    PLATFORM_API_URL: str = "http://localhost:8000"  # Gateway URL
    
    @classmethod
    def from_env(cls) -> "QualificationConfig":
        """
        Load configuration from environment variables.
        
        Only operational parameters that might need tuning are loaded from env.
        Structural parameters (like scoring weights) use defaults.
        """
        return cls(
            # ICP Configuration
            TOTAL_ICPS=int(os.getenv("QUAL_TOTAL_ICPS", 100)),
            LEADS_PER_ICP=int(os.getenv("QUAL_LEADS_PER_ICP", 1)),
            SCREENING_1_ICPS=int(os.getenv("QUAL_SCREENING_1_ICPS", 5)),
            SCREENING_2_ICPS=int(os.getenv("QUAL_SCREENING_2_ICPS", 20)),
            FINAL_BENCHMARK_ICPS=int(os.getenv("QUAL_FINAL_BENCHMARK_ICPS", 75)),
            
            # Evaluation Set Rotation
            EVALUATION_SET_ROTATION_EPOCHS=int(os.getenv("QUAL_EVALUATION_SET_ROTATION_EPOCHS", 20)),
            
            # Cost/Time Limits (per-lead base values)
            MAX_COST_PER_LEAD_USD=float(os.getenv("QUAL_MAX_COST_PER_LEAD_USD", 0.05)),
            MAX_TIME_PER_LEAD_SECONDS=float(os.getenv("QUAL_MAX_TIME_PER_LEAD_SECONDS", 8.0)),
            RUNNING_MODEL_TIMEOUT_SECONDS=int(os.getenv("QUAL_RUNNING_MODEL_TIMEOUT_SECONDS", 30)),
            TOTAL_EVALUATION_TIMEOUT_MINUTES=int(os.getenv("QUAL_TOTAL_EVALUATION_TIMEOUT_MINUTES", 60)),
            
            # Screening Thresholds
            SCREENING_1_THRESHOLD=float(os.getenv("QUAL_SCREENING_1_THRESHOLD", 0.20)),
            SCREENING_2_THRESHOLD=float(os.getenv("QUAL_SCREENING_2_THRESHOLD", 0.40)),
            PRUNE_THRESHOLD=float(os.getenv("QUAL_PRUNE_THRESHOLD", 0.10)),
            
            # Champion Rules
            CHAMPION_DETHRONING_THRESHOLD_PCT=float(os.getenv("QUAL_CHAMPION_DETHRONING_THRESHOLD_PCT", 0.02)),
            MIN_CHAMPION_DURATION_EPOCHS=int(os.getenv("QUAL_MIN_CHAMPION_DURATION_EPOCHS", 1)),
            
            # Champion Rebenchmark Timing (UTC)
            CHAMPION_REBENCHMARK_HOUR_UTC=int(os.getenv("QUAL_CHAMPION_REBENCHMARK_HOUR_UTC", 0)),
            CHAMPION_REBENCHMARK_MINUTE_UTC=int(os.getenv("QUAL_CHAMPION_REBENCHMARK_MINUTE_UTC", 5)),
            
            # Qualification Model Queue
            MAX_MODELS_PER_EPOCH=int(os.getenv("QUAL_MAX_MODELS_PER_EPOCH", 10)),
            QUALIFICATION_BLOCK_CUTOFF=int(os.getenv("QUAL_BLOCK_CUTOFF", 320)),
            
            # Submission Rules
            SUBMISSION_COST_USD=float(os.getenv("QUAL_SUBMISSION_COST_USD", 5.00)),
            MIN_TIME_BETWEEN_SUBMISSIONS_EPOCHS=int(os.getenv("QUAL_MIN_TIME_BETWEEN_SUBMISSIONS_EPOCHS", 20)),
            MAX_SUBMISSIONS_PER_SET=int(os.getenv("QUAL_MAX_SUBMISSIONS_PER_SET", 1)),
            
            # Intent Signal Decay
            INTENT_SIGNAL_DECAY_50_PCT_MONTHS=int(os.getenv("QUAL_INTENT_SIGNAL_DECAY_50_PCT_MONTHS", 2)),
            INTENT_SIGNAL_DECAY_25_PCT_MONTHS=int(os.getenv("QUAL_INTENT_SIGNAL_DECAY_25_PCT_MONTHS", 12)),
            
            # Validator Configuration
            HEARTBEAT_TIMEOUT_SECONDS=int(os.getenv("QUAL_HEARTBEAT_TIMEOUT_SECONDS", 120)),
            HEARTBEAT_INTERVAL_SECONDS=int(os.getenv("QUAL_HEARTBEAT_INTERVAL_SECONDS", 30)),
            REQUEST_EVALUATION_INTERVAL_SECONDS=int(os.getenv("QUAL_REQUEST_EVALUATION_INTERVAL_SECONDS", 10)),
            
            # Emissions
            QUALIFICATION_EMISSIONS_PCT=float(os.getenv("QUAL_EMISSIONS_PCT", 0.05)),
            
            # Intent Verification
            INTENT_VERIFICATION_CONFIDENCE_THRESHOLD=int(os.getenv("QUAL_INTENT_CONFIDENCE_THRESHOLD", 70)),
            INTENT_CACHE_TTL_DAYS=int(os.getenv("QUAL_INTENT_CACHE_TTL_DAYS", 7)),
            
            # API Proxy
            API_PROXY_TIMEOUT_SECONDS=int(os.getenv("QUAL_API_PROXY_TIMEOUT_SECONDS", 30)),
            
            # ScrapingDog Pricing
            SCRAPINGDOG_PLAN_COST_USD=float(os.getenv("SCRAPINGDOG_PLAN_COST_USD", 2000.00)),
            SCRAPINGDOG_PLAN_CREDITS=int(os.getenv("SCRAPINGDOG_PLAN_CREDITS", 42_000_000)),
            
            # S3/Storage
            MODEL_S3_BUCKET=os.getenv("QUAL_MODEL_S3_BUCKET", "leadpoet-leads-primary"),
            MODEL_CODE_MAX_SIZE_MB=int(os.getenv("QUAL_MODEL_CODE_MAX_SIZE_MB", 10)),
            PRESIGN_EXPIRES_SECONDS=int(os.getenv("QUAL_PRESIGN_EXPIRES_SECONDS", 900)),
            
            # Platform URLs
            PLATFORM_API_URL=os.getenv("QUAL_PLATFORM_API_URL", "http://localhost:8000"),
        )
    
    def get_max_screening_1_score(self) -> float:
        """Maximum possible score for screening 1."""
        return self.SCREENING_1_ICPS * self.MAX_POINTS_PER_LEAD
    
    def get_max_screening_2_score(self) -> float:
        """Maximum possible score for screening 2."""
        return self.SCREENING_2_ICPS * self.MAX_POINTS_PER_LEAD
    
    def get_max_final_score(self) -> float:
        """Maximum possible score for final benchmark."""
        return self.FINAL_BENCHMARK_ICPS * self.MAX_POINTS_PER_LEAD
    
    def get_screening_1_pass_threshold(self) -> float:
        """Score needed to pass screening 1."""
        return self.get_max_screening_1_score() * self.SCREENING_1_THRESHOLD
    
    def get_screening_2_pass_threshold(self) -> float:
        """Score needed to pass screening 2."""
        return self.get_max_screening_2_score() * self.SCREENING_2_THRESHOLD
    
    # =========================================================================
    # DYNAMIC LIMITS - Computed based on TOTAL_ICPS × LEADS_PER_ICP
    # =========================================================================
    
    def get_total_leads(self) -> int:
        """
        Total leads the model must produce.
        
        Examples:
            100 ICPs × 1 lead each = 100 leads
            50 ICPs × 1 lead each = 50 leads
            25 ICPs × 2 leads each = 50 leads
        """
        return self.TOTAL_ICPS * self.LEADS_PER_ICP
    
    def get_max_total_cost_usd(self) -> float:
        """
        Maximum total cost for entire evaluation.
        
        Formula: total_leads × MAX_COST_PER_LEAD_USD
        
        Examples:
            100 leads × $0.05 = $5.00
            50 leads × $0.05 = $2.50
            200 leads × $0.05 = $10.00
        """
        return self.get_total_leads() * self.MAX_COST_PER_LEAD_USD
    
    def get_max_total_time_seconds(self) -> float:
        """
        Maximum total time for entire evaluation.
        
        Formula: total_leads × MAX_TIME_PER_LEAD_SECONDS
        
        Examples:
            100 leads × 8s = 800s
            50 leads × 8s = 400s
            200 leads × 8s = 1600s
        """
        return self.get_total_leads() * self.MAX_TIME_PER_LEAD_SECONDS
    
    # Backwards compatibility property
    @property
    def MAX_COST_PER_EVALUATION_USD(self) -> float:
        """Backwards compatible - now computed dynamically."""
        return self.get_max_total_cost_usd()
    
    # =========================================================================
    # DYNAMIC VARIABILITY THRESHOLDS
    # =========================================================================
    
    def get_cost_penalty_threshold(self) -> float:
        """
        Get the cost threshold above which a variability penalty is applied.
        
        Formula: MAX_COST_PER_LEAD_USD × COST_VARIABILITY_THRESHOLD_MULTIPLIER
        
        Examples:
            $0.05 × 2.0 = $0.10 (penalty if cost > $0.10 per lead)
            $0.10 × 2.0 = $0.20 (penalty if cost > $0.20 per lead)
        """
        return self.MAX_COST_PER_LEAD_USD * self.COST_VARIABILITY_THRESHOLD_MULTIPLIER
    
    def get_time_penalty_threshold(self) -> float:
        """
        Get the time threshold above which a variability penalty is applied.
        
        Formula: MAX_TIME_PER_LEAD_SECONDS × TIME_VARIABILITY_THRESHOLD_MULTIPLIER
        
        Examples:
            8s × 2.0 = 16s (penalty if time > 16s per lead)
            10s × 2.0 = 20s (penalty if time > 20s per lead)
        """
        return self.MAX_TIME_PER_LEAD_SECONDS * self.TIME_VARIABILITY_THRESHOLD_MULTIPLIER


# Global config instance - loaded from environment
CONFIG = QualificationConfig.from_env()
