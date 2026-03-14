"""
Qualification System Pydantic Models

All data models for the Lead Qualification Agent competition.
These models are specific to the qualification system and do NOT
modify any existing models in gateway/models/ or Leadpoet/protocol.py.

See business_files/tasks10.md Phase 1.2 for specification.
"""

from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Optional, List, Literal, Dict, Any
from datetime import date, datetime
from urllib.parse import urlparse, urlunparse
from uuid import UUID
from enum import Enum


# =============================================================================
# Enums
# =============================================================================

class ModelStatus(str, Enum):
    """Lifecycle status of a submitted qualification model."""
    SUBMITTED = "submitted"
    SCREENING_1 = "screening_1"
    FAILED_SCREENING_1 = "failed_screening_1"
    SCREENING_2 = "screening_2"
    FAILED_SCREENING_2 = "failed_screening_2"
    EVALUATING = "evaluating"
    FINISHED = "finished"


class EvaluationStatus(str, Enum):
    """Status of an evaluation session."""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILURE = "failure"


class EvaluationRunStatus(str, Enum):
    """Status of an individual evaluation run (per-ICP)."""
    PENDING = "pending"
    INITIALIZING_MODEL = "initializing_model"
    RUNNING_MODEL = "running_model"
    VALIDATING_LEAD = "validating_lead"
    SCORING = "scoring"
    FINISHED = "finished"
    ERROR = "error"


class EvaluationStage(str, Enum):
    """Which stage of progressive evaluation."""
    SCREENING_1 = "screening_1"
    SCREENING_2 = "screening_2"
    FINAL = "final"


class IntentSignalSource(str, Enum):
    """Source of an intent signal."""
    LINKEDIN = "linkedin"
    JOB_BOARD = "job_board"
    SOCIAL_MEDIA = "social_media"
    NEWS = "news"
    GITHUB = "github"
    REVIEW_SITE = "review_site"
    COMPANY_WEBSITE = "company_website"
    WIKIPEDIA = "wikipedia"
    OTHER = "other"

    @classmethod
    def _missing_(cls, value: str):
        """Case-insensitive + whitespace-tolerant enum lookup.
        
        Miners may submit 'LinkedIn', 'Job Board', 'COMPANY_WEBSITE', etc.
        """
        if not isinstance(value, str):
            return None
        normalized = value.strip().lower().replace(" ", "_").replace("-", "_")
        for member in cls:
            if member.value == normalized:
                return member
        return None


class Seniority(str, Enum):
    """Seniority levels for leads."""
    C_SUITE = "C-Suite"
    VP = "VP"
    DIRECTOR = "Director"
    MANAGER = "Manager"
    INDIVIDUAL_CONTRIBUTOR = "Individual Contributor"

    @classmethod
    def _missing_(cls, value: str):
        """Case-insensitive enum lookup for seniority.
        
        Miners may submit 'c-suite', 'vp', 'individual contributor', etc.
        """
        if not isinstance(value, str):
            return None
        normalized = value.strip().lower()
        lookup = {m.value.lower(): m for m in cls}
        if normalized in lookup:
            return lookup[normalized]
        aliases = {
            "c_suite": cls.C_SUITE, "csuite": cls.C_SUITE, "c suite": cls.C_SUITE,
            "exec": cls.C_SUITE, "executive": cls.C_SUITE,
            "vice president": cls.VP, "vice_president": cls.VP,
            "dir": cls.DIRECTOR,
            "mgr": cls.MANAGER,
            "ic": cls.INDIVIDUAL_CONTRIBUTOR, "individual_contributor": cls.INDIVIDUAL_CONTRIBUTOR,
        }
        return aliases.get(normalized)


# =============================================================================
# Intent Signal Models
# =============================================================================

class IntentSignal(BaseModel):
    """
    Intent signal attached to a lead.
    Models must provide evidence of buying intent.
    """
    source: IntentSignalSource
    description: str = Field(..., max_length=500, description="Description of the intent signal")
    url: str = Field(..., description="URL to the source of the intent signal")
    date: Optional[str] = Field(None, description="Date of the signal in ISO 8601 format (YYYY-MM-DD), or null if no verifiable date")
    snippet: str = Field(..., max_length=1000, description="Relevant text snippet extracted from source URL")
    
    @field_validator('date')
    @classmethod
    def validate_date_format(cls, v: Optional[str]) -> Optional[str]:
        """Normalize date to YYYY-MM-DD — handles miner variability.
        
        Accepts common formats miners might use:
        - 2026-02-01 (correct ISO)
        - 2026/02/01 (slashes)
        - 02-01-2026, 02/01/2026 (US format)
        - 2026-2-1 (no leading zeros)
        - null, empty string (treated as no date)
        """
        if v is None or v.strip() == "":
            return None
        v = v.strip()
        for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%m-%d-%Y", "%m/%d/%Y", "%d-%m-%Y", "%d/%m/%Y"):
            try:
                parsed = datetime.strptime(v, fmt)
                return parsed.strftime("%Y-%m-%d")
            except ValueError:
                continue
        raise ValueError("Date must be in YYYY-MM-DD format or null")
    
    @field_validator('url')
    @classmethod
    def validate_url(cls, v: str) -> str:
        """Normalize and validate URL — handles miner variability.
        
        Fixes common issues from miner models:
        - Missing scheme (techcrunch.com/article → https://techcrunch.com/article)
        - Wrong case (HTTP://WWW.TECHCRUNCH.COM → https://www.techcrunch.com)
        - Whitespace
        """
        v = v.strip()
        if not v:
            raise ValueError("URL cannot be empty")
        
        if not v.lower().startswith(('http://', 'https://')):
            v = 'https://' + v
        
        parsed = urlparse(v)
        if not parsed.hostname:
            raise ValueError("URL must contain a valid hostname")
        
        normalized = urlunparse((
            parsed.scheme.lower(),
            parsed.netloc.lower(),
            parsed.path,
            parsed.params,
            parsed.query,
            parsed.fragment,
        ))
        return normalized


# =============================================================================
# Lead Models
# =============================================================================

class LeadOutput(BaseModel):
    """
    Schema for leads returned by qualification models.
    This is what the model's qualify() function must return.
    
    IMPORTANT: Models must ONLY return the required fields below.
    Any extra fields (email, full_name, first_name, last_name, phone, etc.)
    will cause validation to FAIL with score 0.
    
    This prevents models from fabricating person-level data.
    """
    # Pydantic config: FORBID extra fields - any extra field = validation error
    model_config = {"extra": "forbid"}
    
    # =========================================================================
    # LEAD ID - REQUIRED for DB field verification
    # =========================================================================
    # The `id` column from the leads table. Used to verify that
    # the model hasn't tampered with lead fields (employee_count, role, etc.).
    # Models must include this for every lead they return.
    lead_id: int = Field(..., description="ID from the leads table (the 'id' column)")
    
    # =========================================================================
    # REQUIRED FIELDS - All fields below must be provided
    # =========================================================================
    
    # Company info (from the leads table)
    business: str = Field(..., description="Company name")
    company_linkedin: str = Field(..., description="Company LinkedIn URL")
    company_website: str = Field(..., description="Company website URL")
    employee_count: str = Field(..., description="Employee count range (e.g., '51-200', '1001-5000')")
    
    # Industry info
    industry: str = Field(..., description="Company industry")
    sub_industry: str = Field(..., description="Company sub-industry")
    
    # Location (separate fields, NOT a combined 'geography' field)
    country: str = Field(..., description="Country (e.g., 'United States')")
    city: str = Field(..., description="City (e.g., 'San Francisco')")
    state: str = Field(..., description="State/region (e.g., 'California')")
    
    # Role info
    role: str = Field(..., description="Job role/title to target (e.g., 'Software Engineer', 'VP of Sales')")
    role_type: str = Field(..., description="Role category (e.g., 'Engineer/Technical', 'Sales', 'C-Level Executive')")
    seniority: Seniority = Field(..., description="Seniority level")
    
    # Intent signals (evidence of buying intent — at least one required)
    intent_signals: List[IntentSignal] = Field(..., min_length=1, description="Evidence of buying intent (one or more signals)")
    
    # =========================================================================
    # NOT ALLOWED - Any of these fields will cause instant validation failure
    # =========================================================================
    # - email (PII - models cannot fabricate)
    # - full_name (PII - models cannot fabricate)
    # - first_name (PII - models cannot fabricate)
    # - last_name (PII - models cannot fabricate)
    # - phone (PII - models cannot fabricate)
    # - linkedin_url (person-level PII)
    # - geography (use country/city/state instead)
    # - company_size (use employee_count instead)


class LeadOutputRedacted(BaseModel):
    """
    Lead output for miner receipts.
    Same as LeadOutput since we no longer allow PII fields.
    """
    lead_id: int
    business: str
    company_linkedin: str
    company_website: str
    employee_count: str
    industry: str
    sub_industry: str
    country: str
    city: str
    state: str
    role: str
    role_type: str
    seniority: str
    intent_signals: List[IntentSignal]


# =============================================================================
# ICP Models
# =============================================================================

class ICPPrompt(BaseModel):
    """
    Schema for ICP (Ideal Customer Profile) prompts used in evaluation.
    
    CRITICAL: The PRIMARY field is 'prompt' - a natural language description
    that models must INTERPRET to find matching leads.
    
    Example prompt:
        "VP Sales and Heads of Revenue at Series A-C SaaS companies in the US.
         Showing signals: researching outbound tools, hiring SDRs, or 
         evaluating competitors."
    
    Models receive this and must:
    1. Parse/interpret the natural language prompt
    2. Query the database intelligently  
    3. Return the best matching leads
    """
    icp_id: str = Field(..., description="Unique identifier for this ICP")
    
    # PRIMARY FIELD - Models should interpret this natural language prompt
    prompt: str = Field("", description="Natural language prompt describing the ideal customer (PRIMARY)")
    
    # Structured fields for reference/validation
    industry: str = Field(..., description="Target industry category")
    sub_industry: str = Field(..., description="Target sub-industry")
    target_roles: List[str] = Field(default_factory=list, description="List of target job roles/titles")
    target_seniority: str = Field("", description="Target seniority level")
    employee_count: str = Field(..., description="Target employee count range (e.g., '50-200')")
    company_stage: str = Field(..., description="Target company stage (Seed, Series A, etc.)")
    
    @model_validator(mode='before')
    @classmethod
    def handle_legacy_company_size(cls, data: Any) -> Any:
        """Map legacy 'company_size' field to 'employee_count' for backward compatibility."""
        if isinstance(data, dict):
            # If company_size exists but employee_count doesn't, use company_size
            if 'company_size' in data and 'employee_count' not in data:
                data['employee_count'] = data.pop('company_size')
            elif 'company_size' in data and 'employee_count' in data:
                # Both exist - prefer employee_count, remove company_size
                data.pop('company_size')
        return data
    geography: str = Field(..., description="Target geography (full)")
    country: str = Field("", description="Target country (extracted)")
    product_service: str = Field(..., description="Product/service being sold")
    intent_signals: List[str] = Field(default_factory=list, description="Intent signals to look for")
    
    # Legacy fields for backward compatibility
    target_role: Optional[str] = Field(None, description="DEPRECATED: Use target_roles list")
    additional_context: Optional[str] = Field(None, description="DEPRECATED: Use intent_signals list")
    buyer_description: Optional[str] = Field(None, description="DEPRECATED: Use prompt field")
    
    created_at: Optional[datetime] = Field(None, description="When this ICP was created")


class ICPSet(BaseModel):
    """A set of ICPs for an evaluation period."""
    set_id: int
    set_epoch: int
    icps: List[ICPPrompt]
    public_icp_ids: List[str]  # IDs of public ICPs (visible to miners)
    private_icp_ids: List[str]  # IDs of private ICPs
    created_at: datetime


# =============================================================================
# Submission Models
# =============================================================================

class ModelSubmission(BaseModel):
    """
    Request body for model submission.
    Miners submit this to register a new model for evaluation.
    
    Flow:
    1. Miner calls POST /qualification/model/presign to get S3 upload URL
    2. Miner uploads tarball to S3 directly
    3. Miner calls POST /qualification/model/submit with s3_key
    
    This prevents frontrunning (GitHub repos can be copied before submission).
    
    RATE LIMITS:
    - 2 FREE submissions per day (resets at midnight UTC)
    - Additional submissions require payment ($5 TAO = 1 credit)
    - Payment is OPTIONAL if free daily slots are available
    """
    s3_key: str = Field(..., description="S3 object key from presign response")
    code_hash: str = Field(..., min_length=64, max_length=64, description="SHA256 hash of uploaded tarball")
    miner_hotkey: str = Field(..., description="Miner's Bittensor hotkey")
    # Payment fields are OPTIONAL - only required after daily free limit
    payment_block_hash: Optional[str] = Field(None, description="Block hash containing TAO payment (optional if free daily slots available)")
    payment_extrinsic_index: Optional[int] = Field(None, ge=0, description="Extrinsic index of payment (optional if free daily slots available)")
    timestamp: int = Field(..., description="Unix timestamp when submission was signed")
    signature: str = Field(..., description="Signature over submission data")
    model_name: str = Field(..., max_length=100, description="Model name for display (required)")
    
    @field_validator('s3_key')
    @classmethod
    def validate_s3_key(cls, v: str) -> str:
        """Validate S3 key format."""
        if not v.startswith('qualification/uploads/'):
            raise ValueError("S3 key must start with 'qualification/uploads/'")
        if not v.endswith('.tar.gz'):
            raise ValueError("S3 key must end with '.tar.gz'")
        return v
    
    @field_validator('code_hash')
    @classmethod
    def validate_code_hash(cls, v: str) -> str:
        """Validate code hash is valid hex (SHA256 = 64 chars)."""
        if not all(c in '0123456789abcdef' for c in v.lower()):
            raise ValueError("Code hash must be valid hexadecimal")
        return v.lower()


class PresignRequest(BaseModel):
    """Request for presigned S3 upload URL."""
    miner_hotkey: str = Field(..., description="Miner's Bittensor hotkey")
    signature: str = Field(..., description="Signature over request")
    timestamp: int = Field(..., description="Unix timestamp for replay protection")


class PresignResponse(BaseModel):
    """Response with presigned S3 upload URL."""
    upload_url: str = Field(..., description="Presigned URL for direct S3 upload")
    s3_key: str = Field(..., description="S3 object key (use this in ModelSubmission)")
    expires_in_seconds: int = Field(..., description="URL expiration time")
    max_size_bytes: int = Field(..., description="Maximum allowed file size")
    # Rate limit info - so miner knows their status
    daily_submissions_used: int = Field(0, description="Submissions used today")
    daily_submissions_max: int = Field(2, description="Max submissions per day")
    submission_credits: int = Field(0, description="Credits from previous failed submissions")


class ModelSubmissionResponse(BaseModel):
    """Response after successful model submission."""
    model_id: UUID
    status: ModelStatus
    message: str
    created_at: datetime
    
    # Rate limit info (helps miners track their submissions)
    daily_submissions_remaining: Optional[int] = None
    submission_credits_remaining: Optional[int] = None
    daily_resets_at: Optional[datetime] = None


# =============================================================================
# Evaluation Models
# =============================================================================

class EvaluationResult(BaseModel):
    """
    Per-ICP evaluation result (included in miner receipt).
    PII is redacted from lead_returned.
    """
    icp_id: str
    icp_prompt: ICPPrompt
    lead_returned: Optional[Dict[str, Any]] = Field(None, description="Lead data with PII redacted")
    icp_fit_score: float = Field(..., ge=0, le=20)
    decision_maker_score: float = Field(..., ge=0, le=30)
    intent_signal_score: float = Field(..., ge=0, le=50)
    final_lead_score: float = Field(..., ge=0)
    run_cost_usd: float = Field(..., ge=0)
    run_time_seconds: float = Field(..., ge=0)
    error_code: Optional[int] = None
    error_message: Optional[str] = None
    failure_reason: Optional[str] = Field(None, description="Reason for automatic zero score")


class LeadScoreBreakdown(BaseModel):
    """
    Detailed score breakdown for a single lead.
    Used internally during scoring and included in transparency logs.
    """
    # Component scores
    icp_fit: float = Field(..., ge=0, le=20, description="ICP fit score (0-20)")
    decision_maker: float = Field(..., ge=0, le=30, description="Decision-maker score (0-30)")
    intent_signal_raw: float = Field(..., ge=0, le=50, description="Intent signal score before decay (0-50)")
    
    # Time decay
    time_decay_multiplier: float = Field(..., ge=0, le=1, description="1.0, 0.5, or 0.25 based on signal age")
    intent_signal_final: float = Field(..., ge=0, le=50, description="Intent signal score after decay")
    
    # Penalties
    cost_penalty: float = Field(..., ge=0, description="Penalty for API costs")
    time_penalty: float = Field(..., ge=0, description="Penalty for execution time")
    
    # Final
    final_score: float = Field(..., ge=0, description="Final score (floor at 0)")
    
    # Failure tracking
    failure_reason: Optional[str] = Field(None, description="Set when pre-checks fail (score = 0)")
    
    @model_validator(mode='after')
    def validate_score_consistency(self) -> 'LeadScoreBreakdown':
        """Validate that scores are consistent."""
        # If there's a failure reason, final score should be 0
        if self.failure_reason and self.final_score != 0:
            raise ValueError("final_score must be 0 when failure_reason is set")
        return self


# =============================================================================
# Model Status/Score Models
# =============================================================================

class ModelStatusResponse(BaseModel):
    """Response for model status query."""
    model_id: UUID
    status: ModelStatus
    created_at: datetime
    created_epoch: int
    code_hash: Optional[str] = None
    model_name: Optional[str] = None
    s3_path: Optional[str] = None


class ModelScoreResponse(BaseModel):
    """Response for model score query (miner receipt)."""
    model_id: UUID
    status: ModelStatus
    total_score: float
    screening_1_score: Optional[float] = None
    screening_2_score: Optional[float] = None
    final_benchmark_score: Optional[float] = None
    total_cost_usd: Optional[float] = None
    results: List[EvaluationResult]


# =============================================================================
# Champion Models
# =============================================================================

class ChampionInfo(BaseModel):
    """Information about the current champion model."""
    model_id: UUID
    miner_hotkey: str
    code_hash: str
    s3_path: str  # S3 path where champion code is stored (NEVER deleted)
    score: float
    became_champion_epoch: int
    became_champion_at: datetime
    set_id: int
    model_name: Optional[str] = None


# =============================================================================
# Validator Models
# =============================================================================

class ValidatorRegistration(BaseModel):
    """Request to register a qualification validator."""
    timestamp: int
    signed_timestamp: str
    hotkey: str
    commit_hash: str


class ValidatorHeartbeat(BaseModel):
    """Heartbeat from validator."""
    session_id: UUID
    timestamp: int
    system_metrics: Optional[Dict[str, Any]] = None
    current_evaluation: Optional[UUID] = None


class EvaluationWorkItem(BaseModel):
    """Work item for validator to execute."""
    evaluation_id: UUID
    evaluation_run_id: UUID
    model_id: UUID
    probe_id: str
    stage: EvaluationStage
    agent_code_s3_path: str


class EvaluationWorkResponse(BaseModel):
    """Response with work for validator."""
    evaluation_runs: List[EvaluationWorkItem]
    agent_code: Optional[str] = Field(None, description="Base64-encoded model code")
    evaluation_id: Optional[UUID] = None


# =============================================================================
# API Allowlist Models
# =============================================================================

class CostModel(str, Enum):
    """Cost model for allowed APIs."""
    PER_CALL = "per_call"
    PER_CREDIT = "per_credit"
    PER_TOKEN = "per_token"
    FREE = "free"


class AllowedAPI(BaseModel):
    """Configuration for an allowed API provider."""
    provider_id: str
    display_name: str
    base_url: str
    cost_model: CostModel
    cost_per_unit: float
    rate_limit_per_minute: Optional[int] = None
    requires_auth: bool = True
    enabled: bool = True
    notes: str = ""


# =============================================================================
# Transparency Log Event Payloads
# =============================================================================

class ModelSubmittedPayload(BaseModel):
    """Payload for MODEL_SUBMITTED transparency log event."""
    model_id: str
    miner_hotkey: str
    code_hash: str
    s3_path: str
    payment_verified: bool
    model_name: Optional[str] = None


class EvaluationCompletePayload(BaseModel):
    """Payload for EVALUATION_COMPLETE transparency log event."""
    set_id: int
    model_id: str
    miner_hotkey: str
    final_score: float
    total_cost_usd: float
    icps_evaluated: int
    status: str
    db_hash: str  # SHA256 of lead_ids at benchmark start


class ChampionSelectedPayload(BaseModel):
    """Payload for CHAMPION_SELECTED transparency log event."""
    log_epoch: int
    champion_model_id: str
    champion_hotkey: str
    champion_score: float
    previous_champion_id: Optional[str] = None
    margin_over_previous: Optional[float] = None


class EmissionsDistributedPayload(BaseModel):
    """Payload for EMISSIONS_DISTRIBUTED transparency log event."""
    epoch: int
    champion_model_id: Optional[str] = None
    champion_hotkey: Optional[str] = None
    champion_score: Optional[float] = None
    emissions_pct: float
    burned: bool
    reason: Optional[str] = None
