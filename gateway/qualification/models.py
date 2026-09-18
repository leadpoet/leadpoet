"""
Qualification System Pydantic Models

All data models for the Lead Qualification Agent competition.
These models are specific to the qualification system and do NOT
modify any existing models in gateway/models/.

See business_files/tasks10.md Phase 1.2 for specification.
"""

import re
import unicodedata
from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Optional, List, Dict, Any, Literal
from datetime import datetime
from urllib.parse import unquote, urlparse, urlunparse
from enum import Enum

# =============================================================================
# Enums
# =============================================================================

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
            # C-Suite ----------------------------------------------------------
            "c_suite": cls.C_SUITE, "csuite": cls.C_SUITE, "c suite": cls.C_SUITE,
            "c-level": cls.C_SUITE, "c level": cls.C_SUITE, "clevel": cls.C_SUITE,
            "c-level executive": cls.C_SUITE, "c_level": cls.C_SUITE,
            "cxo": cls.C_SUITE, "chief": cls.C_SUITE,
            "ceo": cls.C_SUITE, "cfo": cls.C_SUITE, "cto": cls.C_SUITE,
            "coo": cls.C_SUITE, "cmo": cls.C_SUITE, "cro": cls.C_SUITE,
            "chro": cls.C_SUITE, "cpo": cls.C_SUITE, "cso": cls.C_SUITE,
            "ciso": cls.C_SUITE, "cdo": cls.C_SUITE,
            "president": cls.C_SUITE, "vice chairman": cls.C_SUITE,
            "exec": cls.C_SUITE, "executive": cls.C_SUITE,
            "owner": cls.C_SUITE, "founder": cls.C_SUITE, "co-founder": cls.C_SUITE,
            "co_founder": cls.C_SUITE, "cofounder": cls.C_SUITE,
            "proprietor": cls.C_SUITE, "partner": cls.C_SUITE,
            # VP ---------------------------------------------------------------
            "vice president": cls.VP, "vice_president": cls.VP,
            "svp": cls.VP, "evp": cls.VP, "avp": cls.VP,
            # Director ---------------------------------------------------------
            "dir": cls.DIRECTOR,
            "senior director": cls.DIRECTOR, "senior_director": cls.DIRECTOR,
            "sr director": cls.DIRECTOR, "sr_director": cls.DIRECTOR,
            "sr. director": cls.DIRECTOR,
            "head": cls.DIRECTOR, "head of": cls.DIRECTOR,
            "principal": cls.DIRECTOR,  # Principal Engineer/Consultant ~ director-equivalent
            # Manager ----------------------------------------------------------
            "mgr": cls.MANAGER,
            "senior": cls.MANAGER,  # legacy mapping preserved from to_lead_output
            "senior manager": cls.MANAGER, "senior_manager": cls.MANAGER,
            "sr manager": cls.MANAGER, "sr_manager": cls.MANAGER,
            "sr. manager": cls.MANAGER,
            "lead": cls.MANAGER,  # Team Lead / Engineering Lead — usually people-mgr
            "team lead": cls.MANAGER, "team_lead": cls.MANAGER,
            "supervisor": cls.MANAGER,
            # Individual Contributor ------------------------------------------
            "ic": cls.INDIVIDUAL_CONTRIBUTOR,
            "individual_contributor": cls.INDIVIDUAL_CONTRIBUTOR,
            "individual contributor": cls.INDIVIDUAL_CONTRIBUTOR,
            "staff": cls.INDIVIDUAL_CONTRIBUTOR,  # Staff Engineer = high IC
            "junior": cls.INDIVIDUAL_CONTRIBUTOR, "jr": cls.INDIVIDUAL_CONTRIBUTOR,
            "associate": cls.INDIVIDUAL_CONTRIBUTOR,
            "engineer": cls.INDIVIDUAL_CONTRIBUTOR,
            "analyst": cls.INDIVIDUAL_CONTRIBUTOR,
            "specialist": cls.INDIVIDUAL_CONTRIBUTOR,
            "coordinator": cls.INDIVIDUAL_CONTRIBUTOR,
            "contributor": cls.INDIVIDUAL_CONTRIBUTOR,
            "entry": cls.INDIVIDUAL_CONTRIBUTOR,
            "entry level": cls.INDIVIDUAL_CONTRIBUTOR,
            "entry_level": cls.INDIVIDUAL_CONTRIBUTOR,
        }
        return aliases.get(normalized)


# =============================================================================
# Intent Signal Models
# =============================================================================

_INTENT_INJECTION_PATTERNS = [
    re.compile(
        r"\b(?:ignore|disregard|forget|skip|bypass|override|nullify|cancel)\s+"
        r"(?:all\s+|any\s+|the\s+|every\s+|whatever\s+|what\s+(?:was\s+)?)?"
        r"(?:previous|prior|above|earlier|preceding|former|original|initial)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\b(?:ignore|disregard)\s+(?:everything|all)\b", re.IGNORECASE),
    re.compile(r"\bforget\s+(?:everything|all|what|that)\b", re.IGNORECASE),
    re.compile(
        r"\b(?:new|updated?|revised?|fresh|different)\s+"
        r"(?:instructions?|task|prompt|rules?|directives?|orders?|guidelines?)\s*"
        r"(?:[:.]|are|is|to|that)",
        re.IGNORECASE,
    ),
    re.compile(
        r"<\|(?:im_(?:start|end)|endoftext|fim_[a-z]+|begin_of_text|end_of_text)\|>",
        re.IGNORECASE,
    ),
    re.compile(r"(?:^|[\r\n])\s*(?:system|assistant|user)\s*[:>]", re.IGNORECASE),
    re.compile(
        r"\b(?:return|respond|reply|output|give|set|make|use|score|assign)\s+"
        r"(?:with\s+|this\s+|a\s+|the\s+)?(?:score|value|rating)?\s*"
        r"(?:of\s+|=\s*|:\s*|to\s+)?\s*(?:5\d|60)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\bscore\s*[:=]\s*(?:5\d|60)\b", re.IGNORECASE),
    re.compile(r"\bmatched_icp_signal_idx\s*[:=]", re.IGNORECASE),
    re.compile(r"\bact\s+as\s+(?:a\s+)?(?:different|new)", re.IGNORECASE),
    re.compile(r"\byou\s+are\s+now\s+(?:a\s+)?(?:different|new)", re.IGNORECASE),
    re.compile(r"\bfollow\s+(?:these|the)\s+new\b", re.IGNORECASE),
]


def _scan_for_prompt_injection(text: str, field_name: str) -> None:
    """Raise ``ValueError`` if ``text`` contains evident prompt-injection.
    Used by IntentSignal field validators so that signals embedding gaming
    instructions are rejected at parse time, BEFORE they ever reach a
    scoring or verification LLM.

    The regex set is intentionally specific to known attack phrases; legit
    intent descriptions ("hiring for new SDRs", "raised Series B in 2026")
    do not trigger any of these patterns.  See lead_scorer + intent_verification
    LLM hardening for the second-line defenses (system/user split, JSON
    schema lock) that catch any subtler attempts that slip past this gate.
    """
    if not text:
        return
    for rx in _INTENT_INJECTION_PATTERNS:
        m = rx.search(text)
        if m:
            raise ValueError(
                f"prompt_injection_detected in {field_name!r}: matched {m.group(0)[:60]!r}"
            )


_PROMPT_URL_ROLE_MARKER_RE = re.compile(
    r"(?:^|[/?&#=])(?:system|assistant|user)\s*(?::|>)",
    re.IGNORECASE,
)
_PROMPT_REGISTRABLE_DOMAIN_RE = re.compile(
    r"(?=.{1,253}\Z)"
    r"(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+"
    r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?",
    re.ASCII,
)
_PROMPT_LINKEDIN_COMPANY_SLUG_RE = re.compile(
    r"[a-z0-9][a-z0-9_.-]{0,99}",
    re.ASCII,
)


def _contains_prompt_control(value: str) -> bool:
    return any(
        unicodedata.category(character) in {"Cc", "Cf", "Cs", "Zl", "Zp"}
        for character in value
    )


def validate_candidate_prompt_text(
    text: str, field_name: str, *, allow_layout_whitespace: bool = False
) -> str:
    """Reject candidate controls, role markers, and known prompt steering."""

    if not isinstance(text, str):
        raise ValueError(f"{field_name} must be a string")
    allowed_controls = "\t\n\r" if allow_layout_whitespace else ""
    if any(
        character not in allowed_controls
        and unicodedata.category(character) in {"Cc", "Cf", "Cs", "Zl", "Zp"}
        for character in text
    ):
        raise ValueError(f"{field_name} contains control or format characters")
    _scan_for_prompt_injection(text, field_name)
    return text


def _encoded_ascii_space_confined_to_url_path_or_query(value: str) -> bool:
    """Allow direct ``%20`` only in the raw URL path or query."""

    try:
        parsed = urlparse(value)
    except (TypeError, ValueError):
        return False
    if "%" in parsed.scheme or "%" in parsed.netloc:
        return False
    return not any(
        re.search(r"%20", component, re.IGNORECASE)
        for component in (
            parsed.scheme,
            parsed.netloc,
            parsed.params,
            parsed.fragment,
        )
    )


def canonical_candidate_prompt_url(
    value: str,
    field_name: str,
    *,
    allow_empty: bool = False,
) -> str:
    """Validate and canonicalize a candidate URL before any judge sees it."""

    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")
    if _contains_prompt_control(value):
        raise ValueError(f"{field_name} contains control or format characters")
    raw = value.strip()
    if not raw:
        if allow_empty:
            return ""
        raise ValueError(f"{field_name} cannot be empty")
    if not raw.lower().startswith(("http://", "https://")):
        raw = "https://" + raw
    if any(character.isspace() for character in raw) or _contains_prompt_control(raw):
        raise ValueError(f"{field_name} contains whitespace or controls")
    if "\\" in raw:
        raise ValueError(f"{field_name} contains an ambiguous URL path separator")
    decoded = raw
    for _decode_round in range(4):
        next_decoded = unquote(decoded)
        disallowed_whitespace = any(
            character.isspace() and character != " "
            for character in next_decoded
        )
        newly_decoded_ascii_space = (
            next_decoded.count(" ") > decoded.count(" ")
        )
        if (
            disallowed_whitespace
            or _contains_prompt_control(next_decoded)
            or (
                newly_decoded_ascii_space
                and (
                    _decode_round != 0
                    or not _encoded_ascii_space_confined_to_url_path_or_query(
                        raw
                    )
                )
            )
        ):
            raise ValueError(f"{field_name} contains encoded controls")
        _scan_for_prompt_injection(next_decoded, field_name)
        if _PROMPT_URL_ROLE_MARKER_RE.search(next_decoded):
            raise ValueError(
                f"prompt_injection_detected in {field_name!r}: role marker"
            )
        if next_decoded == decoded:
            break
        decoded = next_decoded
    else:
        raise ValueError(f"{field_name} has excessive percent encoding")
    try:
        parsed = urlparse(raw)
        port = parsed.port
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} is not a valid URL") from exc
    if (
        parsed.scheme.casefold() not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or (port is not None and not 1 <= port <= 65535)
    ):
        raise ValueError(
            f"{field_name} must be an absolute credential-free HTTP(S) URL"
        )
    host = parsed.hostname.casefold()
    if any(ord(character) > 0x7F for character in host):
        raise ValueError(f"{field_name} hostname must use ASCII or punycode")
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    netloc = host + (f":{port}" if port is not None else "")
    return urlunparse(
        (
            parsed.scheme.casefold(),
            netloc,
            parsed.path,
            parsed.params,
            parsed.query,
            "",
        )
    )


def candidate_prompt_url_origin(
    value: str,
    field_name: str,
    *,
    allow_empty: bool = False,
) -> str:
    """Return only a strict DNS origin for an untrusted candidate URL."""

    canonical = canonical_candidate_prompt_url(
        value,
        field_name,
        allow_empty=allow_empty,
    )
    if not canonical:
        return ""
    parsed = urlparse(canonical)
    from leadpoet_verifier.identity.normalization import normalize_url

    registrable_domain = str(
        normalize_url(canonical).domain.registrable_domain or ""
    ).casefold()
    if _PROMPT_REGISTRABLE_DOMAIN_RE.fullmatch(registrable_domain) is None:
        raise ValueError(f"{field_name} has no strict registrable DNS domain")
    port = parsed.port
    return urlunparse(
        (
            parsed.scheme.casefold(),
            registrable_domain + (f":{port}" if port is not None else ""),
            "",
            "",
            "",
            "",
        )
    )


def candidate_linkedin_prompt_slug(
    value: str,
    field_name: str,
    *,
    allow_empty: bool = False,
) -> str:
    """Return a strict LinkedIn person/company slug, never a raw URL."""

    canonical = canonical_candidate_prompt_url(
        value,
        field_name,
        allow_empty=allow_empty,
    )
    if not canonical:
        return ""
    parsed = urlparse(canonical)
    host = str(parsed.hostname or "").casefold()
    if host != "linkedin.com" and not host.endswith(".linkedin.com"):
        raise ValueError(f"{field_name} must use linkedin.com")
    parts = [unquote(part).casefold() for part in parsed.path.split("/") if part]
    if len(parts) != 2 or parts[0] not in {"company", "in"}:
        raise ValueError(f"{field_name} must contain one strict LinkedIn slug")
    slug = parts[1]
    if _PROMPT_LINKEDIN_COMPANY_SLUG_RE.fullmatch(slug) is None:
        raise ValueError(f"{field_name} slug is invalid")
    return slug


def candidate_company_prompt_identity(
    *,
    company_name: str,
    company_website: str,
    company_linkedin: str,
) -> Dict[str, str]:
    """Project raw candidate identity to inert, bounded judge locators."""

    validate_candidate_prompt_text(company_name, "company_name")
    website = candidate_prompt_url_origin(
        company_website,
        "company_website",
        allow_empty=True,
    )
    registrable_domain = str(urlparse(website).hostname or "").casefold()
    linkedin_slug = candidate_linkedin_prompt_slug(
        company_linkedin,
        "company_linkedin",
        allow_empty=True,
    )

    return {
        "company": registrable_domain,
        "website": (
            f"https://{registrable_domain}" if registrable_domain else ""
        ),
        "company_linkedin": linkedin_slug,
    }


class IntentSignal(BaseModel):
    """
    Intent signal attached to a lead.
    Models must provide evidence of buying intent.

    PROMPT-INJECTION DEFENSE
    ------------------------
    The ``description`` and ``snippet`` fields are interpolated into LLM
    prompts at scoring + verification time.  Two classes of defenses
    protect that surface:

      1. **Parse-time rejection** (field validators below): the regex set
         in ``_INTENT_INJECTION_PATTERNS`` scans for evident gaming
         phrases ("ignore previous instructions", ChatML control tokens,
         direct score steering, role-hijacking lines, etc.).  Any match
         raises a ``ValueError``, which causes the parent model to fail
         validation. This shared model accepts up to 4,096 characters; Arena
         enforces its tighter 4,096-byte string boundary before constructing
         the shared model.

      2. **LLM-call defenses** (in ``qualification/scoring/verification_helpers.py``
         and ``lead_scorer.py``): system/user message separation, neutral
         ``<<<MINER_*>>>`` delimited blocks, and strict JSON Schema
         response_format on every LLM call.  Even if a subtler injection
         slipped past the regex, those defenses constrain the worst-case
         outcome to "the LLM ignores the injection and returns a real
         score" — there is no output channel that escapes the schema.
    """
    source: IntentSignalSource
    description: str = Field(..., max_length=4096, description="Description of the intent signal")
    url: str = Field(..., description="URL to the source of the intent signal")
    date: Optional[str] = Field(None, description="Date of the signal in ISO 8601 format (YYYY-MM-DD), or null if no verifiable date")
    snippet: str = Field(..., max_length=4096, description="Relevant text snippet extracted from source URL")
    # REQUIRED on miner submission: index into the ICP's intent_signals list
    # for the signal that this evidence is meant to satisfy. The scorer
    # rejects any signal with matched_icp_signal == -1 or out of range
    # (see qualification/scoring/lead_scorer.py::_score_single_intent_signal).
    #
    # Model-level default is -1 (NOT strictly required at the Pydantic
    # layer) so older stored JSON without the field can still be parsed.
    # Enforcement that the value is set (≥ 0 AND < len(icp.intent_signals))
    # happens at scoring time, not at parse time.
    matched_icp_signal: int = Field(
        default=-1,
        ge=-1,
        description=(
            "REQUIRED on miner submission. Zero-based index into the "
            "ICP intent_signals list of the "
            "client-listed intent signal that this evidence is meant to "
            "prove.  The gateway rejects intent signals with -1 or out-of-"
            "range values at Tier 3 scoring time."
        ),
    )

    @field_validator('description')
    @classmethod
    def validate_description_no_injection(cls, v: str) -> str:
        validate_candidate_prompt_text(
            v, "description", allow_layout_whitespace=True
        )
        return v

    @field_validator('snippet')
    @classmethod
    def validate_snippet_no_injection(cls, v: str) -> str:
        validate_candidate_prompt_text(v, "snippet", allow_layout_whitespace=True)
        return v

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
        return canonical_candidate_prompt_url(v, "intent_signal.url")


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




# =============================================================================
# Company Models (company-mode model competition)
# =============================================================================
#
# As of May 2026 the model competition is transitioning from "surface a
# specific high-intent lead (company + contact) from the published leads
# table" to "surface companies from the open web with verified intent
# signals." Why the shift:
#
#   * Finding contacts requires Apify / LinkedIn scraping; baking that
#     into the base miner model would force every miner who builds on it
#     to take on those licensing risks.
# This is THE output schema for the model competition (as of May 2026).
# ``LeadOutput`` remains for stored-data compatibility, but the model
# competition no longer produces or scores it.

class RequiredAttributeClaim(BaseModel):
    """The model's required-attribute validation result for one company.

    Carried through scoring so the fit gate can enforce that an ICP's
    ``required_attribute`` was actually validated with evidence — previously
    the claim was stripped before scoring and never checked. These fields are
    bounded for storage and audit, but model-authored evidence is never
    interpolated into the independent scorer prompt.
    """
    model_config = {"extra": "ignore"}

    text: str = Field("", max_length=2000, description="The attribute requirement text")
    passed: bool = Field(False, description="Model-reported validation outcome")
    evidence_url: str = Field("", max_length=2000, description="URL backing the claim")
    # Public Arena output strings are bounded to 4,096 UTF-8 bytes before
    # this internal bridge runs. Keep the quote field compatible with that
    # accepted boundary so a valid source excerpt can reach independent
    # verification. The other claim fields retain their narrower limits.
    evidence_quote: str = Field("", max_length=4096, description="Quote backing the claim")
    explanation: str = Field("", max_length=2000, description="Model's reasoning")

    @field_validator("evidence_quote")
    @classmethod
    def _bound_evidence_quote_bytes(cls, value: str) -> str:
        if len(value.encode("utf-8", errors="surrogatepass")) > 4096:
            raise ValueError("evidence_quote exceeds the public UTF-8 byte limit")
        return value


class CompanyOutput(BaseModel):
    """Schema returned by qualification models in the model competition.

    Models receive an ``ICPPrompt`` and must return ONE company that
    matches the ICP criteria (industry / sub-industry / size / geography /
    stage) AND has at least one verifiable intent signal.

    No contact-level fields (no person name, no role, no email, no
    person LinkedIn, no phone, no seniority).  Any extra field causes
    immediate validation failure with score 0, same gaming-prevention
    rule as LeadOutput.
    """
    model_config = {"extra": "forbid"}

    # Identity
    company_name: str = Field(..., min_length=1, max_length=200, description="Legal or common business name")
    company_website: str = Field(..., description="Company website URL (root domain preferred)")
    company_linkedin: str = Field("", description="Company LinkedIn URL (optional but strongly preferred)")

    # Classification
    industry: str = Field(..., description="Company industry")
    sub_industry: str = Field("", description="Company sub-industry (optional)")
    employee_count: str = Field(..., description="Employee count range (e.g. '51-200', '1001-5000')")
    company_stage: str = Field("", description="Funding / lifecycle stage (e.g. 'Series B', 'Public', 'Bootstrapped')")

    # Location — HQ-level only.  No city, since company-mode targets
    # account-level intent and city granularity is meaningless without
    # contacts.  State is optional.
    country: str = Field(..., description="HQ country (e.g. 'United States')")
    state: str = Field("", description="HQ state / region (optional)")

    # Optional descriptive blurb to help the ICP-fit LLM disambiguate
    # similarly-named or generically-named companies.  Bounded so it
    # cannot be used as a prompt-injection lever (same pattern as
    # IntentSignal.description).
    description: str = Field("", max_length=500, description="Short company description / one-liner (optional)")
    intent_details: Optional[str] = Field(
        default=None,
        max_length=2000,
        description="Company-level explanation of verified activity and ICP relevance for v5 Arena outputs",
    )
    fit_evidence_urls: List[str] = Field(
        default_factory=list,
        description="Untrusted public URLs that may help independent company-fit discovery",
    )

    # Intent signals — at least one, same schema as LeadOutput.  This is
    # the load-bearing field for company-mode scoring; the whole point
    # of the competition is verifiable intent.
    intent_signals: List[IntentSignal] = Field(..., min_length=1, description="Verifiable intent signals tied to this company")

    # The model's required-attribute validation, enforced by the scorer when
    # the ICP pins a required_attribute. Optional so legacy outputs still
    # parse — they fail the scorer's attribute gate instead of the schema.
    required_attribute: Optional[RequiredAttributeClaim] = Field(
        None, description="Model-reported required_attribute validation (scorer-enforced)")

    @field_validator('company_name')
    @classmethod
    def _reject_unsafe_company_name(cls, v: str) -> str:
        normalized = v.strip()
        if not normalized:
            raise ValueError("company_name cannot be blank")
        validate_candidate_prompt_text(normalized, "company_name")
        return normalized

    @field_validator('company_website')
    @classmethod
    def _normalize_company_website(cls, v: str) -> str:
        """Normalize same way IntentSignal.url does — accept miner variability."""
        return canonical_candidate_prompt_url(v, "company_website")

    @field_validator('company_linkedin')
    @classmethod
    def _reject_unsafe_company_linkedin(cls, v: str) -> str:
        raw = v.strip()
        if raw:
            candidate_linkedin_prompt_slug(raw, "company_linkedin")
        return raw

    @field_validator('description')
    @classmethod
    def _no_injection_in_description(cls, v: str) -> str:
        if v:
            _scan_for_prompt_injection(v, "description")
        return v

    @field_validator('intent_details')
    @classmethod
    def _validate_intent_details(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        from qualification.intent_details import validate_intent_details_text

        return validate_intent_details_text(value)


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

    # NOTE: As of May 2026 the model competition is single-path company-mode
    # (miners return a ``CompanyOutput``).  There is no ``mode`` field on
    # this schema — it was briefly present during the transition but has
    # been removed.

    # PRIMARY FIELD - Models should interpret this natural language prompt
    prompt: str = Field("", description="Natural language prompt describing the ideal customer (PRIMARY)")
    
    # Structured fields for reference/validation
    industry: str = Field(..., description="Target industry category")
    sub_industry: str = Field(..., description="Target sub-industry")
    target_roles: List[str] = Field(default_factory=list, description="List of target job roles/titles")
    target_seniority: str = Field("", description="Target seniority level")
    contact_policy: Optional[Literal["contacts_v1"]] = Field(
        default=None,
        description="Opt-in marker requiring one verified contact per company",
    )
    contact_geography: Dict[str, List[str]] = Field(
        default_factory=lambda: {"countries": [], "regions": [], "cities": []},
        description="Optional person-level geography filters, independent of company HQ",
    )
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

    @field_validator('contact_geography', mode='before')
    @classmethod
    def validate_contact_geography(cls, value: Any) -> Dict[str, List[str]]:
        if value is None:
            return {"countries": [], "regions": [], "cities": []}
        if not isinstance(value, dict):
            raise ValueError("contact_geography must be an object")
        allowed = {"countries", "regions", "cities"}
        if set(value) - allowed:
            raise ValueError("contact_geography contains unsupported fields")
        normalized: Dict[str, List[str]] = {}
        for key in ("countries", "regions", "cities"):
            items = value.get(key, [])
            if not isinstance(items, list) or len(items) > 25:
                raise ValueError(f"contact_geography.{key} must be a bounded list")
            cleaned: List[str] = []
            for item in items:
                if not isinstance(item, str):
                    raise ValueError(f"contact_geography.{key} must contain strings")
                text = " ".join(item.strip().split())
                if not text or len(text) > 120 or any(ord(char) < 32 for char in item):
                    raise ValueError(f"contact_geography.{key} contains an invalid value")
                if text not in cleaned:
                    cleaned.append(text)
            normalized[key] = cleaned
        return normalized
    geography: str = Field(..., description="Target geography (full)")
    country: str = Field("", description="Target country (extracted)")
    product_service: str = Field(..., description="Product/service being sold")
    required_attribute: str = Field(
        "", description="Attribute every returned company must satisfy (scorer-enforced)")
    excluded_companies: List[str] = Field(
        default_factory=list,
        description="Companies the model must never return: domain, LinkedIn company URL, or name (scorer-enforced)")
    intent_signals: List[str] = Field(default_factory=list, description="Intent signals to look for")
    intent_max_age_days: int = Field(
        365,
        ge=1,
        description="Maximum age in days for intent evidence when the ICP carries an explicit cap",
    )
    # Sibling list mapping 1:1 to ``intent_signals`` carrying the
    # buyer-side evidence_type for each signal (HIRING / FUNDING /
    # SOCIAL_POSTING / PODCAST_APPEARANCE / TECHSTACK / CASE_STUDY /
    # OTHER / None). Left empty for legacy qualification ICPs that only
    # have a plain text list. Consumers
    # (lead_scorer.py) prefer this when present; absence means
    # ``evidence_type=None`` and the downstream prompt dispatcher falls
    # through to the default builder (legacy-compat behavior).
    intent_signal_evidence_types: List[Optional[str]] = Field(
        default_factory=list,
        description="Per-signal evidence_type aligned to intent_signals index",
    )
    intent_signal_max_age_days: List[Optional[int]] = Field(
        default_factory=list,
        description="Per-signal freshness cap aligned to intent_signals index",
    )

    # Legacy fields for backward compatibility
    target_role: Optional[str] = Field(None, description="DEPRECATED: Use target_roles list")
    additional_context: Optional[str] = Field(None, description="DEPRECATED: Use intent_signals list")
    buyer_description: Optional[str] = Field(None, description="DEPRECATED: Use prompt field")
    
    created_at: Optional[datetime] = Field(None, description="When this ICP was created")





# =============================================================================
# Score Models
# =============================================================================

class LeadScoreBreakdown(BaseModel):
    """
    Detailed score breakdown for a single company.
    Used internally during scoring and included in transparency logs.

    Score caps used by ``score_company`` and the Arena scorer:
      * ``icp_fit``        ≤ 40
      * ``decision_maker`` = 0   (no contact dimension in the model
                                  competition; field kept on the
                                  breakdown for backward compatibility
                                  with downstream readers)
      * ``intent_signal``  ≤ 100  (after time decay; the public scorer caps at
                                   60 and the Arena scorer caps at 100)
      * Total              ≤ 100

    NOTE: The historical class name ``LeadScoreBreakdown`` is retained
    rather than renamed, because the breakdown shape is also written to
    Supabase (``qualification_leaderboard`` etc.) and consumed by the
    admin dashboard; renaming would require a coordinated rollout that
    isn't worth the churn for a cosmetic rename.
    """
    # Component scores
    icp_fit: float = Field(..., ge=0, le=40, description="ICP fit score (0-40)")
    decision_maker: float = Field(..., ge=0, le=30, description="Always 0 in company-mode (no contact)")
    intent_signal_raw: float = Field(..., ge=0, le=100, description="Intent signal score before decay (0-100)")
    
    # Time decay
    time_decay_multiplier: float = Field(..., ge=0, le=1, description="1.0, 0.5, or 0.25 based on signal age")
    intent_signal_final: float = Field(..., ge=0, le=100, description="Intent signal score after decay (0-100)")
    
    # Penalties
    cost_penalty: float = Field(..., ge=0, description="Penalty for API costs")
    time_penalty: float = Field(..., ge=0, description="Penalty for execution time")
    
    # Final
    final_score: float = Field(..., ge=0, description="Final score (floor at 0)")
    
    # Failure tracking
    failure_reason: Optional[str] = Field(None, description="Set when pre-checks fail (score = 0)")

    # Per-signal detail for the Arena benchmark. One row
    # per company intent signal: {raw, after_decay, decay, confidence,
    # date_status, matched_icp_signal, evidence_type}.  Optional and defaults to
    # None so legacy lead-mode callers and persisted breakdown readers are
    # unaffected; consumed by the benchmark layer to build per-signal funnel /
    # coverage stats without re-scoring.
    intent_signals_detail: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Per-signal Arena scoring detail; None for lead-mode",
    )
    # Durable verifier-gate receipts (PR-28 audit): audit documents from the
    # deterministic/semantic industry gate — modes, deterministic detail,
    # semantic model/input-hash/source-hashes/judgment, and the final scoring
    # effect. Optional and None by default so legacy callers and persisted
    # breakdown readers are unaffected.
    verifier_gate_receipts: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Durable company-fit gate receipts; None when no gate receipt is needed",
    )

    @model_validator(mode='after')
    def validate_score_consistency(self) -> 'LeadScoreBreakdown':
        """Validate that scores are consistent."""
        # If there's a failure reason, final score should be 0
        if self.failure_reason and self.final_score != 0:
            raise ValueError("final_score must be 0 when failure_reason is set")
        return self
