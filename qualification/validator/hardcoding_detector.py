"""
Qualification System: Hardcoding & Gaming Detection Module

This module analyzes submitted model code BEFORE execution to detect:
1. Hardcoded answers (lookup tables, pre-computed results)
2. Hard steering (gaming the evaluation with specific patterns)
3. Obviously malicious patterns (downloading payloads, accessing sensitive files)
4. Gaming attempts (prompt injection, data manipulation, hidden payloads)

HOW IT WORKS:
- Extracts ALL files from the submitted tarball
- ALL files count toward 200KB limit (py + md + txt + json + etc.)
- Only .py files are analyzed
- Two-layer detection:
  Layer 1: Fast static regex checks (free, instant)
  Layer 2: LLM analysis for sophisticated patterns

SIZE LIMIT: 200KB total for all files. Miner decides how to allocate:
- Big README? Less space for code.
- Multiple model files? Less space for docs.

GAMING PATTERNS DETECTED:
- Prompt injection: Hidden encoded strings prepended to intent_signal.description
- Data manipulation: Copying ICP industry directly to output instead of lead's industry
- Hidden crypto: Custom decode/encrypt functions with hardcoded keys
- Obfuscation: Base64 encoded payloads, XOR operations

Author: LeadPoet
"""

import os
import re
import json
import tarfile
import tempfile
import logging
import httpx
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

def get_max_submission_size() -> int:
    """Get the max total submission size from config or use default (200KB)."""
    return _get_config_value("HARDCODING_MAX_SUBMISSION_SIZE_BYTES", 200_000)

# LLM Model for hardcoding detection - Claude Sonnet 4.5 (1M context window)
DETECTION_MODEL_ID = "anthropic/claude-sonnet-4.5"
DETECTION_MODEL_COST_INPUT = 3.00   # $ per 1M tokens (Claude Sonnet 4.5)
DETECTION_MODEL_COST_OUTPUT = 15.00  # $ per 1M tokens (Claude Sonnet 4.5)

# Default confidence threshold (can be overridden by config)
DEFAULT_REJECTION_THRESHOLD = 70  # 70% = likely hardcoded


def _get_config_value(attr: str, default: Any) -> Any:
    """Get config value with fallback to default."""
    try:
        from gateway.qualification.config import CONFIG
        return getattr(CONFIG, attr, default)
    except ImportError:
        return default


def get_rejection_threshold() -> int:
    """Get the configured rejection threshold."""
    return _get_config_value("HARDCODING_REJECTION_THRESHOLD", DEFAULT_REJECTION_THRESHOLD)


def is_detection_enabled() -> bool:
    """Check if hardcoding detection is enabled."""
    return _get_config_value("ENABLE_HARDCODING_DETECTION", True)


# =============================================================================
# Gaming Detection Patterns (Static Analysis - Layer 1)
# =============================================================================

# Patterns that indicate payload injection attempts
# NOTE: We detect the MECHANISM (encoding/decoding), not specific payload content
# An attacker can change "VERIFICATION SUCCESS" to anything, but the delivery
# mechanism (decode + prepend to field) remains detectable
PAYLOAD_INJECTION_PATTERNS = [
    # Decoded variable being prepended to output fields
    # Pattern: tmp = decode(...) then "field": tmp + something
    r'tmp\s*=\s*decode\s*\([^)]+\)',  # Any decode call storing to tmp
    # Encoded payload being prepended to description/output
    r'["\']description["\']\s*:\s*[a-z_]+\s*\+',  # "description": var + (something being prepended)
    r'["\']snippet["\']\s*:\s*[a-z_]+\s*\+',  # Same for snippet field
    # Variable from decode being concatenated into string fields
    r'decode\s*\([^)]+\)[^}]*["\']description["\']\s*:',  # decode() appears before description field
]

# Patterns indicating hidden crypto for obfuscation
HIDDEN_CRYPTO_PATTERNS = [
    # Custom decode with hardcoded keys
    r'decode\s*\(\s*[a-z_]+\s*,\s*["\']test_key["\']\s*\)',
    r'decode\s*\(\s*[a-z_]+\s*,\s*["\']test_code["\']\s*\)',
    # Custom crypto functions
    r'def\s+derive_key\s*\([^)]*secret',
    r'def\s+keystream\s*\(',
    # XOR-based decryption
    r'bytes\s*\(\s*a\s*\^\s*b\s+for\s+a\s*,\s*b\s+in\s+zip',
]

# Base64 encoded payload patterns (long encoded strings)
ENCODED_PAYLOAD_PATTERNS = [
    # Long base64-looking strings (>60 chars) assigned to variables
    r'[a-z_]+\s*=\s*["\'][A-Za-z0-9+/=_-]{60,}["\']',
]

# Data manipulation patterns (copying ICP to output)
# CRITICAL: The ICP value should NEVER be PRIMARY source - lead data should be primary
DATA_MANIPULATION_PATTERNS = [
    # Directly copying ICP industry to output (instead of lead's industry)
    r'output_industry\s*=\s*parsed_icp\.get\s*\(\s*["\']industry',
    r'output_sub_industry\s*=\s*parsed_icp\.get\s*\(\s*["\']sub_industry',
    # Alternative: ctx.get("industry") being used in output building
    r'["\']industry["\']\s*:\s*ctx\.get\s*\(\s*["\']industry',
    # Comment that indicates intentional gaming
    r'Use\s+ICP.*industry.*for\s+scoring\s+alignment',
    r'CRITICAL.*Use\s+ICP.*industry',
    
    # === ICP ECHO-BACK PATTERNS (with wrapper functions like _norm) ===
    # Pattern: "industry": _norm(ctx.get("industry")) or _norm(lead.get("industry"))
    # The ICP value (ctx) is PRIMARY, lead is fallback - guarantees 100% match
    r'["\']industry["\']\s*:\s*(?:_norm\s*\()?\s*ctx\.get\s*\(\s*["\']industry["\']\s*\)',
    r'["\']sub_industry["\']\s*:\s*(?:_norm\s*\()?\s*ctx\.get\s*\(\s*["\']sub_industry["\']\s*\)',
    
    # Pattern: _norm(ctx.get("X")) or _norm(lead.get("X")) - ICP first, lead fallback
    # The wrapping with _norm doesn't change the priority issue
    r'_norm\s*\(\s*ctx\.get\s*\([^)]+\)\s*\)\s*or\s*_norm\s*\(\s*lead\.get',
    
    # Comments that explicitly admit gaming intent
    r'#.*[Aa]lign\s+with\s+ICP',
    r'#.*maximize.*precheck.*compat',
    r'#.*[Aa]lign.*ICP.*strings',
    r'#.*deterministic\s+precheck\s+compat',
]

# Data fabrication patterns (using random to generate data that should be extracted)
# These are generalizable - ANY model using random for dates/data is fabricating
#
# MECHANISM DETECTION: We detect the combination of random + date generation,
# not specific variable names. Gamers can rename variables but the mechanism
# (random number → timedelta → fake date) remains constant.
#
# Legitimate random uses in ML models:
#   - K-means initialization (random centroids)
#   - Random seeds for reproducibility  
#   - Shuffling/sampling training data
#
# Illegitimate random uses (fabrication):
#   - Generating dates for intent signals
#   - Creating fake timestamps
#   - Randomizing output dates to game recency scoring

DATA_FABRICATION_PATTERNS = [
    # === SINGLE-LINE PATTERNS (original) ===
    # Direct random in timedelta call
    r'timedelta\s*\(\s*days\s*=\s*random\.(randint|uniform|choice)',
    
    # Random for selecting names/companies (fabrication)
    r'random\.choice\s*\([^)]*(?:name|company|business|first_name|last_name)',
    
    # Cache date manipulation (refreshing old dates to look recent)
    r'cached\s*\[\s*["\']date["\']\s*\]\s*=\s*date\.today',
    r'["\']date["\']\s*\]\s*=\s*date\.today\s*\(\s*\)',
    
    # === FIXED TIMEDELTA FALLBACK PATTERNS (HIGH SEVERITY) ===
    # These catch the time decay gaming pattern where models use hardcoded dates
    # like `date.today() - timedelta(days=14)` as fallbacks instead of extracting
    # real dates from scraped content.
    #
    # Pattern: ... or (date.today() - timedelta(days=N))
    # Used as fallback in or-expression to game time decay scoring
    r'or\s*\(?\s*date\.today\s*\(\s*\)\s*-\s*timedelta\s*\(\s*days\s*=\s*\d{1,3}\s*\)',
    
    # Pattern: published = ... or (date.today() - timedelta(days=N))
    # Fallback assigned to published/date variable
    r'published\s*=\s*[^=]*\bor\b[^=]*date\.today\s*\(\s*\)\s*-\s*timedelta',
    
    # Pattern: "date": ... or (date.today() - timedelta(days=N))
    # Fallback in dict literal for date field
    r'["\']date["\']\s*:[^,}]*\bor\b[^,}]*date\.today\s*\(\s*\)\s*-\s*timedelta',
    
    # Pattern: if not published: published = (date.today() - timedelta(days=N))
    # Explicit fallback when extraction fails
    r'if\s+not\s+\w*(?:date|published)\w*\s*:[^:]*date\.today\s*\(\s*\)\s*-\s*timedelta',
]

# === MULTILINE DATE FABRICATION PATTERNS ===
# These catch the split-line evasion where random is on one line
# and date calculation is on another. Uses [\s\S] for cross-line matching.
# Window of 300 chars is enough for 5-10 lines of code.
MULTILINE_DATE_FABRICATION_PATTERNS = [
    # Pattern: random.randint/uniform followed by timedelta(days=...) within 300 chars
    # Catches: x = random.randint(1, 30) \n ... timedelta(days=x)
    r'random\.(randint|uniform)\s*\(\s*\d+\s*,\s*\d+\s*\)[\s\S]{0,300}timedelta\s*\(\s*days\s*=',
    
    # Pattern: random.randint followed by .isoformat() within 300 chars
    # Catches: the full pattern of generating fake ISO date strings
    r'random\.(randint|uniform)\s*\(\s*\d+\s*,\s*\d+\s*\)[\s\S]{0,300}\.isoformat\s*\(',
    
    # Pattern: random.randint followed by date.today() within 300 chars
    # Catches: x = random.randint(...) \n y = date.today() - timedelta(days=x)
    r'random\.(randint|uniform)\s*\(\s*\d+\s*,\s*\d+\s*\)[\s\S]{0,300}date\.today\s*\(\s*\)\s*-\s*timedelta',
    
    # Pattern: Variable with day/offset/ago in name assigned from random
    # Catches: days_offset = random.randint, time_ago = random.uniform, etc.
    r'\w*(day|offset|ago|delta)\w*\s*=\s*random\.(randint|uniform)\s*\(',
    
    # Pattern: random followed by "date" key assignment in dict within 400 chars
    # Catches: x = random... \n signal = {"date": ...}
    r'random\.(randint|uniform)\s*\(\s*\d+\s*,\s*\d+\s*\)[\s\S]{0,400}["\']date["\']\s*:',
    
    # Pattern: random followed by signal_date or intent_date assignment
    # Catches various naming conventions for date variables in intent/signal context
    r'random\.(randint|uniform)\s*\(\s*\d+\s*,\s*\d+\s*\)[\s\S]{0,300}(signal|intent|event).*date\s*=',
]

# === GENERIC INTENT FALLBACK PATTERNS ===
# These detect templated/generic intent descriptions that are designed to
# always pass verification by being true for ANY company.
# 
# The problem: These descriptions are technically "verifiable" from any website
# (if a company has a website, they're "actively operating"), but they provide
# NO actual buying intent signal value.
#
# Examples of gaming fallback intents:
#   f"{company} is actively operating in {industry}"
#   f"{company} market activity and company updates."
#   f"{company} is expanding operations..."
#
# These are NOT real intent signals - they're fallbacks that always "pass"
GENERIC_INTENT_FALLBACK_PATTERNS = [
    # Pattern: f-string with company + "is actively operating" (the exact cipher pattern)
    r'f["\'][^"\']*\{[^}]*company[^}]*\}[^"\']*is\s+actively\s+operating',
    
    # Pattern: f-string with "visible market activity"
    r'f["\'][^"\']*visible\s+market\s+activity',
    
    # Pattern: f-string with company + "market activity and company updates"
    r'f["\'][^"\']*\{[^}]*company[^}]*\}[^"\']*market\s+activity',
    
    # Pattern: Generic hardcoded description strings that are templated
    r'["\']description["\']\s*:\s*f["\'][^"\']*\{[^}]*(company|business)[^}]*\}[^"\']*(?:operating|active|activity|expanding)',
    
    # Pattern: Fallback intent with generic "operating in {industry}"
    r'f["\'][^"\']*operating\s+in\s+\{[^}]*industry',
    
    # Pattern: Deterministic fallback comment followed by generic return
    r'#.*(?:fallback|conservative|default|generic)[\s\S]{0,200}return\s*\{[^}]*description',
    
    # Pattern: date.today() - timedelta(days=N) with hardcoded N (not extracted from content)
    # This is a fallback date pattern - real dates should be extracted from content
    r'date\.today\s*\(\s*\)\s*-\s*timedelta\s*\(\s*days\s*=\s*\d{1,3}\s*\)',
    
    # Pattern: Generic snippet about "company updates" or "market activity"
    r'["\']snippet["\']\s*:\s*f["\'][^"\']*(?:company\s+updates|market\s+activity|business\s+operations)',
    
    # === SOURCE TYPE INFLATION PATTERNS ===
    # Company's own /careers or /jobs pages classified as "job_board" (1.0x multiplier)
    # These are company websites (0.85x), not real job boards like Greenhouse/Lever
    # This inflates intent score by ~18%
    r'\(\s*f?["\'][^"\']*(?:/careers|/jobs)[^"\']*["\']\s*,\s*["\']job_board["\']\s*\)',
    r'["\'](?:/careers|/jobs)["\']\s*[,\]]\s*["\']job_board["\']',
    
    # === ROLE/SENIORITY ECHO-BACK PATTERNS ===
    # Role fallback to ICP's target_roles[0] - stamps ICP's desired role onto lead
    r'ctx\.get\s*\(\s*["\']target_roles["\']\s*\)\s*\[\s*0\s*\]',
    r'ctx\.get\s*\(\s*["\']target_roles["\']\s*,\s*\[\s*\]\s*\)\s*\[\s*0\s*\]',
    
    # Seniority fallback to ICP's target_seniority - stamps ICP's desired seniority
    r'_norm\s*\(\s*ctx\.get\s*\(\s*["\']target_seniority["\']\s*\)\s*\)',
    r'target_seniority\s*=\s*ctx\.get\s*\(\s*["\']target_seniority["\']\s*\)',
    
    # Pattern: role = lead.get("role") or ctx.get("target_roles")[0]
    # Role defaulting to ICP when lead has no role
    r'role\s*=\s*[^=]*\bor\b[^=]*ctx\.get\s*\(\s*["\']target_roles["\']\s*\)',
    
    # Hardcoded "Manager" or "Individual Contributor" fallback
    r'return\s*["\'](?:Manager|Individual Contributor)["\']',
    r'else\s*["\'](?:Manager|Individual Contributor)["\']',
]

# Low-value intent source types that should be penalized
# "other" source means they couldn't categorize the source properly
LOW_VALUE_INTENT_SOURCES = [
    r'["\']source["\']\s*:\s*["\']other["\']',
    r'source\s*=\s*["\']other["\']',
]


def _run_static_gaming_checks(code_content: str) -> Tuple[bool, List[str], int]:
    """
    Run fast static checks for gaming patterns.
    
    This runs BEFORE the LLM call to catch obvious gaming attempts
    without spending money on API calls.
    
    IMPORTANT: We detect MECHANISMS (how gaming is done), not specific content.
    This makes detection robust against variations in payload text.
    
    Args:
        code_content: Combined Python code from submission
        
    Returns:
        Tuple of (passed, red_flags, confidence_score)
        - passed: True if no gaming detected, False if gaming found
        - red_flags: List of specific patterns found
        - confidence_score: 0-100 (higher = more likely gaming)
    """
    red_flags = []
    confidence = 0
    
    # Check payload injection patterns (decoded content being prepended to fields)
    # This detects the MECHANISM regardless of what the payload says
    injection_matches = 0
    for pattern in PAYLOAD_INJECTION_PATTERNS:
        if re.search(pattern, code_content, re.IGNORECASE):
            injection_matches += 1
    
    if injection_matches >= 2:
        # Multiple injection patterns = high confidence gaming
        red_flags.append(f"Payload injection mechanism detected ({injection_matches} patterns: decode + prepend to output)")
        confidence = max(confidence, 90)
    
    # Check hidden crypto patterns (HIGH severity)
    crypto_matches = 0
    for pattern in HIDDEN_CRYPTO_PATTERNS:
        if re.search(pattern, code_content, re.IGNORECASE | re.DOTALL):
            crypto_matches += 1
    
    if crypto_matches >= 2:
        # Multiple crypto patterns = likely obfuscation
        red_flags.append(f"Hidden crypto/obfuscation detected ({crypto_matches} patterns)")
        confidence = max(confidence, 90)
    elif crypto_matches == 1:
        # Single pattern might be legitimate, flag for LLM review
        red_flags.append("Suspicious crypto pattern (needs LLM review)")
        confidence = max(confidence, 50)
    
    # Check for encoded payloads
    payload_matches = re.findall(ENCODED_PAYLOAD_PATTERNS[0], code_content)
    if payload_matches:
        # Check if the encoded string is used in output (not just a constant)
        for match in payload_matches:
            var_name = match.split('=')[0].strip()
            # Check if this variable is used in description or output
            usage_pattern = rf'{var_name}\s*\+'
            if re.search(usage_pattern, code_content):
                red_flags.append(f"Encoded payload prepended to output via variable")
                confidence = max(confidence, 85)
                break
    
    # Check data manipulation patterns (MEDIUM to HIGH severity)
    # Separate ICP echo-back patterns (HIGH) from general data manipulation (MEDIUM)
    data_manip_matches = 0
    icp_echo_matches = 0
    gaming_comment_matches = 0
    
    # ICP echo-back patterns (HIGH severity) - these guarantee 100% match
    ICP_ECHO_PATTERNS = [
        r'["\']industry["\']\s*:\s*(?:_norm\s*\()?\s*ctx\.get\s*\(\s*["\']industry["\']\s*\)',
        r'["\']sub_industry["\']\s*:\s*(?:_norm\s*\()?\s*ctx\.get\s*\(\s*["\']sub_industry["\']\s*\)',
        r'_norm\s*\(\s*ctx\.get\s*\([^)]+\)\s*\)\s*or\s*_norm\s*\(\s*lead\.get',
    ]
    
    # Gaming admission comment patterns (VERY HIGH severity)
    GAMING_COMMENT_PATTERNS = [
        r'#.*[Aa]lign\s+with\s+ICP',
        r'#.*maximize.*precheck.*compat',
        r'#.*[Aa]lign.*ICP.*strings',
        r'#.*deterministic\s+precheck\s+compat',
    ]
    
    for pattern in DATA_MANIPULATION_PATTERNS:
        if re.search(pattern, code_content, re.IGNORECASE):
            data_manip_matches += 1
    
    # Check specifically for ICP echo-back patterns
    for pattern in ICP_ECHO_PATTERNS:
        if re.search(pattern, code_content, re.IGNORECASE):
            icp_echo_matches += 1
    
    # Check specifically for gaming admission comments
    for pattern in GAMING_COMMENT_PATTERNS:
        if re.search(pattern, code_content, re.IGNORECASE):
            gaming_comment_matches += 1
    
    if gaming_comment_matches >= 1:
        red_flags.append(f"GAMING ADMISSION: Code contains comments admitting gaming intent (e.g., 'align with ICP', 'maximize precheck compatibility')")
        confidence = max(confidence, 90)  # Explicit admission = auto-fail
    if icp_echo_matches >= 1:
        red_flags.append(f"ICP ECHO-BACK: Output uses ICP values as PRIMARY source with lead data as fallback - guarantees 100% fuzzy match ({icp_echo_matches} patterns)")
        confidence = max(confidence, 85)  # This is pure gaming
    elif data_manip_matches >= 1 and icp_echo_matches == 0:
        red_flags.append(f"Data manipulation: ICP data may be copied to output ({data_manip_matches} patterns)")
        confidence = max(confidence, 60)  # Medium confidence, LLM should confirm
    
    # Check data fabrication patterns (HIGH severity for dates, MEDIUM for others)
    fabrication_matches = 0
    fabrication_details = []
    for pattern in DATA_FABRICATION_PATTERNS:
        if re.search(pattern, code_content, re.IGNORECASE | re.DOTALL):
            fabrication_matches += 1
            fabrication_details.append(f"single-line: {pattern[:50]}...")
    
    # Check MULTILINE date fabrication patterns (HIGH severity)
    # These catch split-line evasion attempts where random and date ops are on different lines
    multiline_fabrication_matches = 0
    for pattern in MULTILINE_DATE_FABRICATION_PATTERNS:
        if re.search(pattern, code_content, re.IGNORECASE | re.DOTALL):
            multiline_fabrication_matches += 1
            fabrication_details.append(f"multiline: {pattern[:50]}...")
    
    total_fabrication = fabrication_matches + multiline_fabrication_matches
    
    if multiline_fabrication_matches >= 1:
        # ANY multiline date fabrication pattern = HIGH confidence gaming
        # These patterns specifically detect the mechanism of using random to generate dates
        # which is ALWAYS fabrication (dates should come from real content)
        red_flags.append(f"Date fabrication detected: random used to generate fake dates ({multiline_fabrication_matches} multiline pattern(s))")
        confidence = max(confidence, 88)  # Very high - this is always gaming
    elif total_fabrication >= 2:
        # Multiple fabrication patterns = high confidence gaming
        red_flags.append(f"Data fabrication: random used to generate dates/data ({total_fabrication} patterns)")
        confidence = max(confidence, 85)  # High severity
    elif total_fabrication == 1:
        red_flags.append("Potential data fabrication: random used for date/data generation")
        confidence = max(confidence, 60)  # Medium - let LLM confirm
    
    # ==========================================================================
    # HARDCODED FALLBACK DATE DETECTION (HIGH SEVERITY)
    # ==========================================================================
    # This specifically detects the time decay gaming pattern where models use
    # hardcoded dates like `date.today() - timedelta(days=14)` as fallbacks.
    # Real dates should be EXTRACTED from scraped content, not fabricated.
    # 
    # Pattern examples that are ALWAYS gaming:
    #   - `or (date.today() - timedelta(days=14)).isoformat()`
    #   - `fallback_date = (date.today() - timedelta(days=N)).isoformat()`
    #   - `"date": ... or (date.today() - timedelta(days=21)).isoformat()`
    #
    # This is a SEPARATE check from generic fallbacks because it's HIGH severity
    hardcoded_date_patterns = [
        # Fallback date with `or` (the cipher pattern)
        r'\bor\s*\(\s*date\.today\s*\(\s*\)\s*-\s*timedelta\s*\(\s*days\s*=\s*\d+',
        # Fallback date assigned to variable
        r'fallback.*date.*=.*date\.today\s*\(\s*\)\s*-\s*timedelta\s*\(\s*days\s*=\s*\d+',
        # Date field with hardcoded fallback
        r'["\']date["\']\s*:.*\bor\b.*date\.today\s*\(\s*\)\s*-\s*timedelta',
        # Any date.today() - timedelta(days=N) where N is 7-30 (typical gaming range)
        r'date\.today\s*\(\s*\)\s*-\s*timedelta\s*\(\s*days\s*=\s*(7|14|21|28|30)\s*\)',
    ]
    
    hardcoded_date_matches = 0
    for pattern in hardcoded_date_patterns:
        if re.search(pattern, code_content, re.IGNORECASE | re.DOTALL):
            hardcoded_date_matches += 1
    
    if hardcoded_date_matches >= 2:
        # Multiple hardcoded date patterns = CERTAIN gaming of time decay
        red_flags.append(f"TIME DECAY GAMING: Hardcoded fallback dates detected ({hardcoded_date_matches} patterns) - dates should be extracted from content, not fabricated")
        confidence = max(confidence, 90)  # Instant fail - this is always cheating
    elif hardcoded_date_matches == 1:
        red_flags.append("Suspicious hardcoded fallback date: model may be gaming time decay scoring")
        confidence = max(confidence, 70)  # High - needs LLM review
    
    # Check for generic intent fallback patterns (MEDIUM-HIGH severity)
    # These detect models that use templated "always pass" fallback intents
    generic_intent_matches = 0
    generic_intent_details = []
    for pattern in GENERIC_INTENT_FALLBACK_PATTERNS:
        matches = re.findall(pattern, code_content, re.IGNORECASE | re.DOTALL)
        if matches:
            generic_intent_matches += 1
            generic_intent_details.append(pattern[:40])
    
    if generic_intent_matches >= 3:
        # Multiple generic intent patterns = gaming the fallback system
        red_flags.append(f"Generic intent fallback: model uses templated 'always pass' intent descriptions ({generic_intent_matches} patterns)")
        confidence = max(confidence, 75)  # High enough to flag, but let LLM confirm
    elif generic_intent_matches >= 2:
        red_flags.append(f"Suspicious intent fallback: templated descriptions detected ({generic_intent_matches} patterns)")
        confidence = max(confidence, 55)  # Medium - needs LLM review
    elif generic_intent_matches == 1:
        # Single pattern might be legitimate fallback
        red_flags.append("Potential generic intent pattern (needs LLM review)")
        confidence = max(confidence, 35)  # Low - informational
    
    # Check for "other" source type usage (MEDIUM severity)
    # Models that frequently use "other" as source are likely fabricating
    other_source_matches = 0
    for pattern in LOW_VALUE_INTENT_SOURCES:
        if re.search(pattern, code_content, re.IGNORECASE):
            other_source_matches += 1
    
    if other_source_matches >= 2:
        red_flags.append(f"Low-value intent sources: model uses 'other' source type ({other_source_matches} occurrences)")
        confidence = max(confidence, 45)  # Informational, let LLM and scoring handle
    
    # ==========================================================================
    # ROLE/SENIORITY ECHO-BACK DETECTION (HIGH SEVERITY)
    # ==========================================================================
    # Models that stamp ICP's desired role/seniority onto leads with missing data
    # instead of inferring from the lead's actual data
    role_seniority_patterns = [
        # Role fallback to ICP's target_roles[0]
        r'ctx\.get\s*\(\s*["\']target_roles["\']\s*\)\s*\[\s*0\s*\]',
        # Seniority fallback to ICP's target_seniority
        r'_norm\s*\(\s*ctx\.get\s*\(\s*["\']target_seniority["\']\s*\)\s*\)',
        r'target_seniority\s*=\s*ctx\.get\s*\(\s*["\']target_seniority["\']\s*\)',
        # Role defaulting to ICP
        r'role\s*=\s*[^=]*\bor\b[^=]*ctx\.get\s*\(\s*["\']target_roles["\']\s*\)',
    ]
    
    role_seniority_matches = 0
    for pattern in role_seniority_patterns:
        if re.search(pattern, code_content, re.IGNORECASE | re.DOTALL):
            role_seniority_matches += 1
    
    if role_seniority_matches >= 2:
        red_flags.append(f"ROLE/SENIORITY ECHO-BACK: Model stamps ICP's desired role/seniority onto leads ({role_seniority_matches} patterns) - should infer from lead data, not ICP")
        confidence = max(confidence, 80)  # High severity - gaming persona matching
    elif role_seniority_matches == 1:
        red_flags.append("Suspicious role/seniority fallback to ICP values (needs LLM review)")
        confidence = max(confidence, 55)
    
    # ==========================================================================
    # SOURCE TYPE INFLATION DETECTION (MEDIUM-HIGH SEVERITY)
    # ==========================================================================
    # /careers or /jobs pages classified as "job_board" instead of "company_website"
    source_inflation_patterns = [
        r'\(\s*f?["\'][^"\']*(?:/careers|/jobs)[^"\']*["\']\s*,\s*["\']job_board["\']\s*\)',
        r'["\'](?:/careers|/jobs)["\']\s*[,\]]\s*["\']job_board["\']',
        r'(?:/careers|/jobs)[^}]*["\']job_board["\']',
    ]
    
    source_inflation_matches = 0
    for pattern in source_inflation_patterns:
        if re.search(pattern, code_content, re.IGNORECASE):
            source_inflation_matches += 1
    
    if source_inflation_matches >= 1:
        red_flags.append(f"SOURCE TYPE INFLATION: /careers or /jobs pages classified as 'job_board' instead of 'company_website' - inflates intent score multiplier")
        confidence = max(confidence, 65)  # Medium-high - clear gaming
    
    # Determine if we should fail immediately or defer to LLM
    passed = confidence < 85  # 85+ = instant fail, below = let LLM decide
    
    return passed, red_flags, confidence


# =============================================================================
# Main Detection Function
# =============================================================================

async def analyze_model_for_hardcoding(
    model_code: bytes,
    icp_samples: List[Dict[str, Any]],
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Analyze a model submission for hardcoding patterns using an LLM.
    
    This runs BEFORE the model is executed in the sandbox. If the model
    appears to be hardcoded (confidence >= threshold), it should be rejected
    without running.
    
    SECURITY: The FULL code is sent to the LLM - no truncation. Models
    exceeding 100KB are rejected to ensure complete analysis.
    
    Args:
        model_code: Tarball bytes of the submitted model
        icp_samples: Sample ICPs that will be used for testing (3-5 examples)
        api_key: OpenRouter API key (defaults to env var)
    
    Returns:
        {
            "passed": bool,  # True if model appears legitimate, False if hardcoded
            "confidence_hardcoded": int,  # 0-100 (0=valid, 100=obviously hardcoded)
            "red_flags": List[str],  # Specific patterns found
            "evidence": str,  # LLM's explanation
            "model_used": str,  # Which LLM was used
            "analysis_cost_usd": float,  # Cost of this analysis
        }
    """
    # Check if detection is enabled
    if not is_detection_enabled():
        logger.info("Hardcoding detection is disabled via config")
        return {
            "passed": True,
            "confidence_hardcoded": 0,
            "red_flags": [],
            "evidence": "Hardcoding detection disabled",
            "model_used": None,
            "analysis_cost_usd": 0.0
        }
    
    try:
        # Get API key
        # SECURITY: Qualification uses SEPARATE API keys with limited funds.
        # TODO: After beta release, change back to "OPENROUTER_API_KEY" (shared with sourcing)
        openrouter_key = api_key or os.environ.get("QUALIFICATION_OPENROUTER_API_KEY")
        if not openrouter_key:
            logger.error("❌ CRITICAL: No OpenRouter API key for hardcoding detection - FAILING CLOSED")
            return {
                "passed": False,  # SECURITY: Fail closed - don't allow unchecked models
                "confidence_hardcoded": 100,
                "red_flags": ["No API key available for hardcoding detection"],
                "evidence": "BLOCKED - Cannot verify model integrity without API key",
                "model_used": None,
                "analysis_cost_usd": 0.0
            }
        
        # Extract code from tarball (ALL files count toward size, only .py analyzed)
        py_content, py_file_count, total_size = await _extract_code_from_tarball(model_code)
        
        # SECURITY: Check total size of ALL files FIRST (prevents memory attacks)
        max_size = get_max_submission_size()
        if total_size > max_size:
            logger.warning(
                f"❌ Model REJECTED: Total size {total_size:,} bytes exceeds "
                f"limit of {max_size:,} bytes (200KB)"
            )
            return {
                "passed": False,
                "confidence_hardcoded": 100,
                "red_flags": [f"Total submission size {total_size:,} bytes exceeds {max_size:,} byte limit"],
                "evidence": (
                    f"Model submission contains {total_size:,} bytes total across all files "
                    f"(py, md, txt, json, etc.). This exceeds the 200KB limit. "
                    f"Reduce file sizes or remove unnecessary files."
                ),
                "model_used": None,
                "analysis_cost_usd": 0.0
            }
        
        if not py_content:
            logger.warning("Could not extract Python code from tarball")
            return {
                "passed": True,  # Allow if no .py files (will fail elsewhere)
                "confidence_hardcoded": 0,
                "red_flags": [],
                "evidence": "No Python files found in tarball",
                "model_used": None,
                "analysis_cost_usd": 0.0
            }
        
        py_size = len(py_content.encode('utf-8'))
        logger.info(f"   Submission: {total_size:,} bytes total, {py_size:,} bytes Python across {py_file_count} file(s)")
        
        # =================================================================
        # LAYER 1: Fast Static Gaming Checks (free, instant)
        # =================================================================
        static_passed, static_flags, static_confidence = _run_static_gaming_checks(py_content)
        
        if static_flags:
            logger.info(f"   🔍 Static analysis found {len(static_flags)} potential issue(s)")
            for flag in static_flags:
                logger.info(f"      - {flag}")
        
        # If static check confidence is very high (85+), fail immediately
        if not static_passed:
            logger.warning(f"   ❌ GAMING DETECTED (static): {static_confidence}% confidence")
            for flag in static_flags:
                logger.warning(f"      🚨 {flag}")
            
            return {
                "passed": False,
                "confidence_hardcoded": static_confidence,
                "red_flags": static_flags,
                "evidence": (
                    "Static analysis detected gaming patterns in the code. "
                    "This appears to be an attempt to manipulate the evaluation system "
                    "through prompt injection, data manipulation, or hidden payloads."
                ),
                "model_used": "static_analysis",
                "analysis_cost_usd": 0.0  # No LLM cost - caught by static check
            }
        
        # =================================================================
        # LAYER 2: LLM Analysis (for sophisticated patterns)
        # =================================================================
        # Pass static flags to LLM for additional context
        
        # Build the analysis prompt with FULL Python code and real ICP samples
        prompt = _build_analysis_prompt(py_content, icp_samples, static_flags)
        
        # Get timeout from config
        timeout = _get_config_value("HARDCODING_DETECTION_TIMEOUT", 120)
        
        # Call the LLM (o3-mini)
        analysis_result, cost = await _call_reasoning_llm(
            prompt=prompt,
            api_key=openrouter_key,
            timeout=float(timeout)
        )
        
        # Parse the LLM response
        parsed = _parse_llm_response(analysis_result)
        
        # Determine if model passes
        rejection_threshold = get_rejection_threshold()
        passed = parsed["confidence_hardcoded"] < rejection_threshold
        
        result = {
            "passed": passed,
            "confidence_hardcoded": parsed["confidence_hardcoded"],
            "red_flags": parsed["red_flags"],
            "evidence": parsed["evidence"],
            "model_used": DETECTION_MODEL_ID,
            "analysis_cost_usd": cost
        }
        
        # Log the result
        if passed:
            logger.info(
                f"   ✅ Hardcoding check PASSED: {parsed['confidence_hardcoded']}% confidence "
                f"(threshold: {rejection_threshold}%)"
            )
        else:
            logger.warning(
                f"   ❌ Hardcoding check FAILED: {parsed['confidence_hardcoded']}% confidence "
                f"(threshold: {rejection_threshold}%)"
            )
            logger.warning(f"   Red flags: {parsed['red_flags']}")
        
        return result
        
    except Exception as e:
        logger.error(f"Hardcoding detection error: {e}")
        # Allow the model to run on error (don't block legitimate submissions)
        return {
            "passed": True,
            "confidence_hardcoded": 0,
            "red_flags": [],
            "evidence": f"Detection error: {str(e)}",
            "model_used": None,
            "analysis_cost_usd": 0.0
        }


# =============================================================================
# Helper Functions
# =============================================================================

async def _extract_code_from_tarball(model_code: bytes) -> Tuple[Optional[str], int, int]:
    """
    Extract files from tarball for analysis.
    
    ALL files count toward size limit (py, md, txt, json, etc.)
    Only .py files are sent to LLM for hardcoding analysis.
    
    Returns:
        Tuple of (combined_py_content, py_file_count, total_size_all_files)
        Returns (None, 0, 0) if extraction fails
    """
    try:
        # Write to temp file
        with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as f:
            f.write(model_code)
            temp_path = f.name
        
        # Extract ALL files and track total size
        py_files = {}
        total_size = 0
        
        with tarfile.open(temp_path, "r:gz") as tar:
            for member in tar.getmembers():
                if member.isfile():
                    f = tar.extractfile(member)
                    if f:
                        content = f.read().decode("utf-8", errors="ignore")
                        total_size += len(content.encode('utf-8'))
                        
                        # Only keep .py files for LLM analysis
                        if member.name.endswith(".py"):
                            py_files[member.name] = content
        
        # Clean up
        Path(temp_path).unlink()
        
        if not py_files:
            return None, 0, total_size
        
        # Combine .py files for LLM analysis
        combined = []
        for filepath, content in sorted(py_files.items()):
            combined.append(f"# ===== FILE: {filepath} =====")
            combined.append(content)
            combined.append("")
        
        return "\n".join(combined), len(py_files), total_size
        
    except Exception as e:
        logger.error(f"Error extracting code from tarball: {e}")
        return None, 0, 0


def _build_analysis_prompt(
    code_content: str, 
    icp_samples: List[Dict[str, Any]],
    static_flags: List[str] = None
) -> str:
    """
    Build the prompt for hardcoding and gaming detection.
    
    The core question: Does this model's output genuinely depend on the ICP input?
    Would it work on a brand new ICP it has never seen before?
    Is it trying to game/manipulate the evaluation system?
    """
    
    # Format the actual ICP samples that will be tested
    icp_json_samples = ""
    for i, icp in enumerate(icp_samples[:5], 1):  # Show up to 5 ICPs
        # Extract key fields
        icp_display = {
            "industry": icp.get("industry", ""),
            "sub_industry": icp.get("sub_industry", ""),
            "target_roles": icp.get("target_roles", icp.get("target_role_titles", [])),
            "geography": icp.get("geography", icp.get("target_geography", "")),
            "employee_count": icp.get("employee_count", ""),
        }
        icp_json_samples += f"\n**ICP {i}:**\n```json\n{json.dumps(icp_display, indent=2)}\n```\n"
    
    # Add static analysis context if available
    static_context = ""
    if static_flags:
        flags_str = "\n".join(f"- {flag}" for flag in static_flags)
        static_context = f"""
## Static Analysis Findings

Our automated scan found these potential issues that need your review:
{flags_str}

Please verify if these are genuine gaming attempts or false positives.
"""
    
    prompt = f"""## Competition Overview

This is a B2B lead qualification agent competition. Models receive an ICP (Ideal Customer Profile) and must find a matching sales lead by querying the leads database (table name provided via config).

**What a legitimate model does:**
1. Parse the ICP to understand what kind of lead is needed
2. Query the database with filters derived from the ICP (industry, geography, etc.)
3. Use APIs to enrich or validate the lead
4. Return a lead that matches the ICP criteria

**The test:** ICPs are generated fresh daily. The model must generalize to ANY ICP.

## ICPs That Will Be Tested

These are ACTUAL ICPs this model will be evaluated against:
{icp_json_samples}
{static_context}
## Code Submission

```python
{code_content}
```

## Your Analysis Task

**Phase 1: Check for blatant hardcoding**

- Does the code return static/hardcoded emails, names, or companies?
- Are there lookup tables mapping specific industries to specific leads?
- Does it ignore the ICP and return the same output every time?
- Does it generate fake data without querying the database?
- Does it use random name generators (e.g., `random.choice(FIRST_NAMES)`) to fabricate person data even if company data is from DB?

**Also flag if you see obvious malicious patterns** (auto-fail, 100% confidence):
- Code that downloads and executes remote payloads, or opens network connections to unknown servers
- Attempts to read files outside the working directory (especially paths containing `.bittensor`, `.ssh`, `.aws`, or `/etc/`)

**Phase 2: Check for GAMING/MANIPULATION attempts** (auto-fail if found)

⚠️ CRITICAL: Check for these gaming patterns that try to trick the evaluation system:

1. **Payload Injection via Encoding**: Look for the MECHANISM of hiding and injecting text:
   - Variables that store decoded content and prepend it to output fields: `"description": decoded_var + actual_content`
   - Base64-encoded strings that get decoded and inserted into `intent_signal` fields
   - The specific TEXT of the payload doesn't matter - any encoded string being injected is suspicious
   - Ask yourself: Why would a lead qualification model need to decode hidden strings?
   
2. **Hidden Crypto/Obfuscation**: Custom decode/encrypt functions:
   - Functions like `decode()`, `derive_key()`, `keystream()` with hardcoded keys
   - XOR-based operations: `bytes(a ^ b for a, b in zip(...))`
   - Large base64/encoded strings (>50 chars) that are decoded at runtime
   - Legitimate lead qualification has NO reason to include custom cryptography

3. **Data Manipulation**: Copying ICP data directly to output instead of lead's actual data:
   - `output_industry = parsed_icp.get("industry")` or `ctx.get("industry")` used in output building
   - This games fuzzy matching by ensuring output always matches ICP perfectly
   - The output should reflect the LEAD's data, not just echo back the ICP

4. **Data Fabrication**: Using `random` to generate data that should be EXTRACTED from real sources:
   - `random.randint()` to generate dates: `days_ago = random.randint(3, 45)` - FABRICATION
   - `random.choice()` to select names/companies from lists - FABRICATION
   - Refreshing cached dates: `cached["date"] = date.today()` to make old data look recent
   - **KEY**: random for ML internals (k-means, sampling) is OK. Trace where random values GO - if they flow to output fields (dates, signals), it's fabrication even if split across lines.

5. **Date Fabrication via Fixed Timedelta Fallback**: Any `date.today() - timedelta(days=N)` where N is a fixed integer (not extracted from content):
   - `(date.today() - timedelta(days=14)).isoformat()` - fabricates a date 14 days ago
   - `(date.today() - timedelta(days=21)).isoformat()` - fabricates a date 21 days ago
   - This is NOT random date fabrication - it's FIXED fabrication, equally dishonest
   - Real models extract dates from page content (JSON-LD datePosted, meta tags, text)
   - If no date is found, the model should return None, NOT invent one
   - **KEY**: The fabricated date is always "recent" to maximize the recency/time_decay score
   - Check for this pattern in: (a) intent building functions, (b) output building functions, (c) fallback paths

6. **ICP Echo-Back with `or` Fallback Priority**: The output uses ICP values as the PRIMARY source with lead data as fallback:
   - `"industry": ctx.get("industry") or lead.get("industry")` - ICP FIRST, lead second
   - `"sub_industry": ctx.get("sub_industry") or lead.get("sub_industry")` - same pattern
   - This guarantees 100% fuzzy match because the validator compares output vs ICP - if you echo the ICP back, it's comparing the string against itself
   - The CORRECT pattern is: `lead.get("industry")` as primary (the lead's actual data), return None if missing
   - Also check for wrapper functions: `_norm(ctx.get("industry")) or _norm(lead.get("industry"))` - the wrapping doesn't change the priority
   - Comments like "align with ICP" or "maximize precheck compatibility" are explicit admissions of gaming

7. **Seniority/Role Echo-Back from ICP**: Output fields use ICP target values as fallback:
   - Seniority inferred from role, but falls back to `target_seniority` from ICP if keywords don't match - this stamps the ICP's desired seniority on any lead with a vague role
   - Role falls back to `ctx.get("target_roles")[0]` or hardcoded "Manager" when lead has no role
   - The CORRECT behavior is to infer from the lead's own data, defaulting to "Individual Contributor" (most common/conservative), not the ICP's desired value

8. **Source Type Inflation**: Company's own /careers or /jobs pages classified as "job_board":
   - `("/careers", "job_board")` or `("/jobs", "job_board")` in URL candidate lists
   - Real job boards are external platforms (greenhouse.io, lever.co, indeed.com) with structured JSON-LD datePosted
   - A company's /careers page is "company_website" (0.85x multiplier), not "job_board" (1.0x)
   - This inflates the intent score multiplier by ~18%

9. **"Defensive Fallback" Pattern** - the meta-gaming trick: The model has a genuine pipeline that tries real scraping first, but EVERY path has a fabrication fallback:
   - Intent discovery: tries real pages → falls back to generic template
   - Date extraction: tries real dates → falls back to `date.today() - timedelta(...)`
   - Industry: tries lead data → falls back to ICP echo
   - The model appears legitimate on a quick read because the real code exists - but in production, the fallbacks fire frequently (pages fail to load, dates aren't found)
   - **KEY QUESTION**: What percentage of outputs would use real vs fabricated data? If every failure path fabricates, the model is gaming with plausible deniability

**Phase 3: Trace ICP flow through the code**

- Is `icp["industry"]` used in database queries?
- Is `icp["target_roles"]` used to filter or score candidates?
- Is `icp["geography"]` used meaningfully?
- Would different ICPs produce different queries?

**Phase 4: Check against the actual test ICPs above**

- Does the code have hardcoded answers for any of the industries shown above?
- Would it produce meaningfully different results for each ICP?

## Confidence Scale

- **0-30%**: Legitimate - ICP flows through to database queries and API calls, dates/data extracted from real sources
- **31-69%**: Suspicious - Partial ICP usage mixed with hardcoded elements OR some fallback patterns detected
- **70-100%**: Hardcoded/Gaming - Output doesn't depend on ICP input OR gaming detected

**IMPORTANT**: Any gaming/manipulation attempt should result in confidence 85-100%. This includes:
- Prompt injection, data manipulation, hidden crypto (items 1-4)
- Fixed timedelta date fabrication (item 5)
- ICP echo-back patterns where ICP is PRIMARY source (item 6)
- Seniority/role stamping from ICP values (item 7)
- Source type inflation (/careers as job_board) (item 8)
- Defensive fallback pattern with fabrication on every failure path (item 9)

A model with MULTIPLE of these patterns (e.g., date fabrication + ICP echo-back + seniority stamping) is 95-100% confidence gaming.

## Response

```json
{{
    "confidence_hardcoded": <0-100>,
    "red_flags": ["specific issue found", ...],
    "evidence": "How ICP flows (or doesn't flow) through the code, plus any gaming patterns found",
    "verdict": "LEGITIMATE|SUSPICIOUS|HARDCODED|GAMING"
}}
```"""
    
    return prompt


async def _call_reasoning_llm(
    prompt: str,
    api_key: str,
    timeout: float = 120.0
) -> Tuple[str, float]:
    """
    Call OpenRouter API with o3-mini for hardcoding detection.
    
    Returns:
        Tuple of (response_text, cost_usd)
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://leadpoet.io",
        "X-Title": "LeadPoet Hardcoding Detection"
    }
    
    payload = {
        "model": DETECTION_MODEL_ID,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.0,  # Zero temp for deterministic results
        "max_tokens": 4000
    }
    
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=payload
        )
        response.raise_for_status()
        
        data = response.json()
        
        # Extract response text
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        
        # Extract cost from response
        cost = 0.0
        usage = data.get("usage", {})
        if "cost" in usage:
            cost = usage["cost"]
        elif "total_cost" in usage:
            cost = usage["total_cost"]
        else:
            # Estimate cost from tokens using o3-mini pricing
            input_tokens = usage.get("prompt_tokens", 0)
            output_tokens = usage.get("completion_tokens", 0)
            cost = (
                (input_tokens / 1_000_000 * DETECTION_MODEL_COST_INPUT) +
                (output_tokens / 1_000_000 * DETECTION_MODEL_COST_OUTPUT)
            )
        
        return content, cost


def _parse_llm_response(response: str) -> Dict[str, Any]:
    """
    Parse the LLM's JSON response with multiple fallback strategies.
    
    CRITICAL: Must reliably extract confidence_hardcoded from any LLM
    output format. A parsing failure must NOT allow a hardcoded model
    to slip through.
    """
    # Strategy 1: Extract JSON from markdown code block
    json_str = response
    if "```json" in response:
        start = response.find("```json") + 7
        end = response.find("```", start)
        if end > start:
            json_str = response[start:end].strip()
    elif "```" in response:
        start = response.find("```") + 3
        end = response.find("```", start)
        if end > start:
            json_str = response[start:end].strip()
    
    # Strategy 2: Try direct JSON parse
    try:
        parsed = json.loads(json_str)
        confidence = int(parsed.get("confidence_hardcoded", 50))
        return {
            "confidence_hardcoded": max(0, min(100, confidence)),
            "red_flags": parsed.get("red_flags", []),
            "evidence": parsed.get("evidence", "No explanation provided"),
            "verdict": parsed.get("verdict", "UNKNOWN")
        }
    except json.JSONDecodeError:
        pass
    
    # Strategy 3: Fix common JSON issues (unescaped newlines in strings)
    try:
        fixed = re.sub(r'(?<!\\)\n', '\\n', json_str)
        parsed = json.loads(fixed)
        confidence = int(parsed.get("confidence_hardcoded", 50))
        logger.info("Parsed LLM response after fixing unescaped newlines")
        return {
            "confidence_hardcoded": max(0, min(100, confidence)),
            "red_flags": parsed.get("red_flags", []),
            "evidence": parsed.get("evidence", "No explanation provided"),
            "verdict": parsed.get("verdict", "UNKNOWN")
        }
    except (json.JSONDecodeError, Exception):
        pass
    
    # Strategy 4: Regex extraction of confidence_hardcoded from raw text
    # This handles cases where the JSON is malformed but the number is visible
    confidence_match = re.search(
        r'"confidence_hardcoded"\s*:\s*(\d+)', response
    )
    if confidence_match:
        confidence = int(confidence_match.group(1))
        confidence = max(0, min(100, confidence))
        
        # Also try to extract red_flags
        red_flags = []
        flags_match = re.search(r'"red_flags"\s*:\s*\[(.*?)\]', response, re.DOTALL)
        if flags_match:
            red_flags = re.findall(r'"([^"]+)"', flags_match.group(1))
        
        logger.warning(
            f"JSON parse failed — extracted confidence={confidence} via regex. "
            f"Raw response: {response[:300]}..."
        )
        return {
            "confidence_hardcoded": confidence,
            "red_flags": red_flags or ["Parsed via regex fallback"],
            "evidence": response[:1000],
            "verdict": "UNKNOWN"
        }
    
    # Strategy 5: Complete parse failure — default to REJECT (fail-safe)
    # If we can't parse the response AT ALL, we can't trust the model.
    # Defaulting to pass would let hardcoded models through.
    logger.error(
        f"COMPLETE PARSE FAILURE — could not extract confidence from LLM response. "
        f"Defaulting to REJECT (fail-safe). Raw: {response[:300]}..."
    )
    return {
        "confidence_hardcoded": 100,
        "red_flags": ["Complete LLM response parse failure — fail-safe rejection"],
        "evidence": response[:1000],
        "verdict": "REJECT"
    }


# =============================================================================
# Output Validation (Post-Execution Gaming Detection - Layer 3)
# =============================================================================

def validate_model_output_for_gaming(
    output: Dict[str, Any],
    icp: Dict[str, Any],
    lead_from_db: Dict[str, Any] = None
) -> Tuple[bool, List[str]]:
    """
    Validate model output for gaming patterns AFTER execution.
    
    NOTE ON APPROACH:
    We focus on STRUCTURAL anomalies, not specific phrases.
    An attacker can change "VERIFICATION SUCCESS" to anything, but structural
    patterns (like injected prefix before company name) are harder to avoid.
    
    Specific phrase detection is intentionally NOT done here because:
    1. It's trivially bypassed by changing the text
    2. The actual intent verification (LLM checking URL content) will catch
       mismatches between description and actual page content
    
    What we CAN detect:
    - Data manipulation (ICP industry copied to output instead of lead's actual industry)
    - Structural anomalies in intent_signal.description
    
    Args:
        output: The model's LeadOutput
        icp: The ICP that was provided to the model
        lead_from_db: Optional - the actual lead data from database (for comparison)
        
    Returns:
        Tuple of (is_valid, issues)
        - is_valid: True if output appears legitimate, False if gaming detected
        - issues: List of specific issues found
    """
    issues = []
    
    if not output:
        return True, []
    
    company = output.get("business", "")
    intent_signal = output.get("intent_signal", {})
    description = intent_signal.get("description", "")
    
    # Structural check: Does the description mention the company at all?
    # A legitimate intent signal should reference the company
    if description and company:
        company_lower = company.lower()
        description_lower = description.lower()
        
        # Check if company name appears in description
        # (Simple check - company should be mentioned somewhere)
        company_words = [w for w in company_lower.split() if len(w) > 3]
        company_mentioned = any(word in description_lower for word in company_words)
        
        if not company_mentioned and len(description) > 100:
            # Long description that doesn't mention the company is suspicious
            # But don't flag short descriptions (might legitimately omit company name)
            issues.append(
                f"Structural anomaly: intent description ({len(description)} chars) "
                f"doesn't mention company '{company}'"
            )
    
    # Check for data manipulation (industry copied from ICP)
    # This is the most reliable gaming detection we can do
    if lead_from_db:
        output_industry = output.get("industry", "")
        icp_industry = icp.get("industry", "")
        lead_industry = lead_from_db.get("industry", "")
        
        # If output industry matches ICP exactly but differs from lead's actual industry
        if (output_industry and icp_industry and lead_industry and
            output_industry.lower() == icp_industry.lower() and
            output_industry.lower() != lead_industry.lower()):
            issues.append(
                f"Data manipulation: output industry '{output_industry}' matches ICP "
                f"but lead's actual industry is '{lead_industry}'"
            )
    
    is_valid = len(issues) == 0
    return is_valid, issues


# =============================================================================
# Utility Functions for Coordinator/Worker Integration
# =============================================================================

def should_skip_hardcoding_check(model_data: Dict[str, Any]) -> bool:
    """
    Determine if hardcoding check should be skipped for this model.
    
    Skip for:
    - Champion rebenchmarks (already passed check when first submitted)
    - Models that have already been evaluated before
    """
    # Check if this is a rebenchmark
    is_rebenchmark = model_data.get("is_rebenchmark", False)
    if is_rebenchmark:
        return True
    
    # Check if model has been evaluated before (re-evaluation)
    evaluated_at = model_data.get("evaluated_at")
    if evaluated_at:
        return True
    
    return False
