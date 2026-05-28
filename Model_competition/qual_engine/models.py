"""Pydantic schemas for every layer boundary."""

from __future__ import annotations

from datetime import date as _date, datetime
from typing import Literal, Optional, Any
from pydantic import BaseModel, Field, HttpUrl


IntentClass = Literal[
    "hiring",
    "funding",
    "product_launch",
    "tech_adoption",
    "expansion",
    "partnership",
    "leadership_change",
    "compliance_event",
    "other",
]

SourceType = Literal[
    "company_website",
    "linkedin",
    "job_board",
    "news",
    "social_media",
    "github",
    "wikipedia",
    "other",
]


# -----------------------------------------------------------------------------
# Input
# -----------------------------------------------------------------------------

class ICPPrompt(BaseModel):
    icp_id: str
    prompt: str = ""
    industry: str
    sub_industry: str = ""
    geography: str
    country: str = ""
    employee_count: str = ""
    company_stage: str = ""
    product_service: str = ""
    intent_signals: list[str] = Field(default_factory=list)


# -----------------------------------------------------------------------------
# L1 — ICP Parser
# -----------------------------------------------------------------------------

class ParsedICP(BaseModel):
    intent_class: IntentClass                    # primary/dominant class (kept for compat)
    intent_classes: list[IntentClass] = Field(default_factory=list)  # aligned with ICPPrompt.intent_signals
    is_time_bound: bool = False
    time_window_days: Optional[int] = None
    hard_filters: dict = Field(default_factory=dict)
    semantic_queries: list[str] = Field(default_factory=list)
    sonar_query_angles: list[str] = Field(default_factory=list)
    keyword_queries: list[str] = Field(default_factory=list)
    linkedin_jobs_query: Optional[str] = None
    news_keywords: list[str] = Field(default_factory=list)


# -----------------------------------------------------------------------------
# L2 — Discovery
# -----------------------------------------------------------------------------

CandidateSource = Literal[
    "sonar",
    "exa_neural",
    "exa_keyword",
    "sd_google",
    "sd_linkedin_jobs",
    "extracted_from_article",
]


class CandidateCompany(BaseModel):
    name: str
    domain: str
    source: CandidateSource
    discovery_url: str = ""
    discovery_snippet: str = ""
    # Optional attribute hints from discovery (Sonar can return these; Exa cannot).
    # NOT verified — used only for cheap pre-filtering at L2 before L3 resolves.
    hint_hq: Optional[str] = None        # "City, State, Country" string as returned by Sonar
    hint_stage: Optional[str] = None     # "Series B", "Seed", etc.
    hint_size: Optional[str] = None      # "50-200", "20-50", etc.


# -----------------------------------------------------------------------------
# L3 — Resolver
# -----------------------------------------------------------------------------

class ResolvedCompany(BaseModel):
    canonical_name: str
    primary_domain: str
    aliases: list[str] = Field(default_factory=list)
    country: Optional[str] = None
    industry_tags: list[str] = Field(default_factory=list)
    employee_count_band: Optional[str] = None
    funding_stage: Optional[str] = None
    linkedin_url: Optional[str] = None
    linkedin_slug: Optional[str] = None
    headquarters: Optional[str] = None  # raw "City, State, Country"
    description: Optional[str] = None
    discovery_sources: list[str] = Field(default_factory=list)
    anchor_source: Literal["linkedin", "website"] = "linkedin"


# -----------------------------------------------------------------------------
# L5 — Evidence
# -----------------------------------------------------------------------------

class EvidenceURL(BaseModel):
    url: str
    source_type: SourceType
    raw_content: Optional[str] = None
    claimed_date: Optional[_date] = None
    discovered_via: str = ""
    title: Optional[str] = None


# -----------------------------------------------------------------------------
# L6 — Grounding
# -----------------------------------------------------------------------------

class VerifiedSignal(BaseModel):
    url: str
    source_type: SourceType
    date: Optional[_date] = None
    snippet: str
    description: str
    matched_icp_signal_idx: int
    grounding_confidence: int  # 0-100
    sonar_corroborated: bool = False
    proof_quote: str = ""


# -----------------------------------------------------------------------------
# L7 / L8 — Output
# -----------------------------------------------------------------------------

class CompanyOutput(BaseModel):
    company_name: str
    company_website: str
    company_linkedin: Optional[str] = None
    industry: str = ""
    sub_industry: str = ""
    employee_count: str = ""
    company_stage: str = ""
    country: str = ""
    state: str = ""
    description: str = ""
    intent_signals: list[VerifiedSignal] = Field(default_factory=list)


class CompanyMatch(BaseModel):
    """One qualifying company. Every signal carries a clickable URL + verbatim proof."""
    company: CompanyOutput
    score: int                       # ranker 0-100
    overall_confidence: int          # blended (ranker + grounding + corroboration)
    industry_match: int = 0
    structural_match: int = 0
    intent_strength: int = 0
    reasoning: str = ""


class QualificationResult(BaseModel):
    # Multi-answer output: every company here passed all 10 grounding gates AND
    # the ranker. Sort order: descending by `overall_confidence`.
    matches: list[CompanyMatch] = Field(default_factory=list)
    total_matches: int = 0
    abstention_reason: Optional[str] = None   # set only when matches is empty
    reasoning_trace: list[dict] = Field(default_factory=list)
    cost_breakdown: dict = Field(default_factory=dict)
    latency_ms: int = 0
