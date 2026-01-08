"""
POST /submit - Verify lead upload and finalize submission
=========================================================

After miner uploads lead blob to S3 via presigned URL,
they call this endpoint to trigger verification.

Flow per BRD Section 4.1:
1. Gateway fetches uploaded blob from each mirror
2. Recomputes SHA256 hash
3. Verifies hash matches committed lead_blob_hash from SUBMISSION_REQUEST
4. If verification succeeds:
   - Logs STORAGE_PROOF event per mirror
   - Stores lead in leads_private table
   - Logs SUBMISSION event
5. If verification fails:
   - Logs UPLOAD_FAILED event
   - Returns error

This prevents blob substitution attacks (BRD Section 5.2).
"""

import sys
import os
import hashlib
import json
import re
from datetime import datetime
from typing import Dict, List

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from fastapi import APIRouter, HTTPException, Body
from pydantic import BaseModel, Field

# Import configuration
from gateway.config import SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY

# Import utilities
from gateway.utils.signature import verify_wallet_signature, construct_signed_message, compute_payload_hash
from gateway.utils.registry import is_registered_hotkey_async  # Use async version
from gateway.utils.nonce import check_and_store_nonce, validate_nonce_format
from gateway.utils.storage import verify_storage_proof
from gateway.utils.rate_limiter import MAX_SUBMISSIONS_PER_DAY, MAX_REJECTIONS_PER_DAY

# Import Supabase
from supabase import create_client, Client

# Create Supabase client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)

# ============================================================
# Role Sanity Check Configuration (loaded from JSON)
# ============================================================
# Load role validation patterns from JSON config file
# This allows updating patterns without code changes
_role_patterns_path = os.path.join(os.path.dirname(__file__), 'role_patterns.json')
with open(_role_patterns_path, 'r') as f:
    ROLE_PATTERNS = json.load(f)

# Build typo dictionary for fast lookup
ROLE_TYPO_DICT = {}
for correct, typos in ROLE_PATTERNS['typos'].items():
    for typo in typos:
        ROLE_TYPO_DICT[typo.lower()] = correct.lower()

# Build URL patterns from TLDs
ROLE_URL_PATTERNS = [r'https?://', r'\bwww\.']
for tld in ROLE_PATTERNS['url_tlds']:
    ROLE_URL_PATTERNS.append(rf'\b\w+\.{tld}\b')

# Compile regex patterns for performance
ROLE_NON_LATIN_RE = re.compile(ROLE_PATTERNS['non_latin_regex'])
ROLE_EMOJI_RE = re.compile(ROLE_PATTERNS['emoji_regex'])

print(f"[submit.py] Loaded {len(ROLE_TYPO_DICT)} typo patterns, {len(ROLE_URL_PATTERNS)} URL patterns")

# Create router
router = APIRouter(prefix="/submit", tags=["Submission"])


# ============================================================
# LinkedIn URL Normalization (for duplicate detection)
# ============================================================
from gateway.utils.linkedin import normalize_linkedin_url, compute_linkedin_combo_hash

# ============================================================
# Geographic Normalization (standardizes city/state/country)
# ============================================================
from gateway.utils.geo_normalize import normalize_location, validate_location, normalize_country


# ============================================================
# Role Sanity Check Function
# ============================================================

def check_role_sanity(role_raw: str) -> tuple:
    """
    Validate role format - returns (error_code, error_message) or (None, None) if valid.

    Checks loaded from role_patterns.json for easy maintenance.
    Catches garbage roles at gateway BEFORE entering validation queue.
    """
    role_raw = role_raw.strip()
    role_lower = role_raw.lower()
    thresholds = ROLE_PATTERNS['thresholds']
    letters_only = re.sub(r'[^a-zA-Z]', '', role_raw)

    # ==========================================
    # CURRENT 11 CHECKS
    # ==========================================

    # Check 1: Too short
    if len(role_raw) < thresholds['min_length']:
        return ("role_too_short", f"Role too short ({len(role_raw)} chars). Minimum {thresholds['min_length']} characters required.")

    # Check 2: Too long
    if len(role_raw) > thresholds['max_length']:
        return ("role_too_long", f"Role too long ({len(role_raw)} chars). Maximum {thresholds['max_length']} characters allowed.")

    # Check 3: No letters
    if not any(c.isalpha() for c in role_raw):
        return ("role_no_letters", "Role must contain at least one letter.")

    # Check 4: Mostly numbers
    if sum(c.isdigit() for c in role_raw) > len(role_raw) * thresholds['max_digit_ratio']:
        return ("role_mostly_numbers", "Role cannot be mostly numbers.")

    # Check 5: Placeholder patterns
    if role_lower in ROLE_PATTERNS['placeholders']:
        return ("role_placeholder", "Role appears to be a placeholder or keyboard spam.")

    # Check 6: Repeated character 4+ times
    if re.search(r'(.)\1{3,}', role_raw):
        return ("role_repeated_chars", "Role contains repeated characters (spam pattern).")

    # Check 7: Repeated words 3+ times
    role_words = role_lower.split()
    word_counts = {}
    for w in role_words:
        if len(w) > 1:
            word_counts[w] = word_counts.get(w, 0) + 1
    if any(count >= 3 for count in word_counts.values()):
        return ("role_repeated_words", "Role contains the same word repeated 3+ times.")

    # Check 8: Scam/spam phrases
    for pattern in ROLE_PATTERNS['scam_patterns']:
        if pattern in role_lower:
            return ("role_scam_pattern", f"Role contains spam/scam pattern: '{pattern}'")

    # Check 9: URL in role (basic check)
    if re.search(r'https?://|www\.|\.com/|\.org/|\.net/|\.io/', role_lower):
        return ("role_contains_url", "Role cannot contain URLs.")

    # Check 10: Email in role
    if re.search(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', role_raw):
        return ("role_contains_email", "Role cannot contain email addresses.")

    # Check 11: Phone number in role
    if re.search(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b|\b\+\d{10,}', role_raw):
        return ("role_contains_phone", "Role cannot contain phone numbers.")

    # ==========================================
    # NEW CHECKS (loaded from JSON)
    # ==========================================

    # Check 12: Non-English characters (non-Latin scripts)
    if ROLE_NON_LATIN_RE.findall(role_raw):
        return ("role_non_english", "Role contains non-English characters.")

    # Check 13: URLs and websites (comprehensive TLD check)
    role_for_url = role_lower.replace('.net', '_NET_')  # Preserve .NET framework
    for pattern in ROLE_URL_PATTERNS:
        if re.search(pattern, role_for_url):
            return ("role_contains_website", "Role cannot contain website domains.")

    # Check 14: Typos in common job words
    role_words_alpha = re.findall(r'[a-zA-Z]+', role_lower)
    for word in role_words_alpha:
        if word in ROLE_TYPO_DICT:
            return ("role_typo", f"Role contains typo: '{word}' should be '{ROLE_TYPO_DICT[word]}'")

    # Check 15: Too few letters
    if len(letters_only) < thresholds['min_letters']:
        return ("role_too_few_letters", "Role must contain at least 3 letters.")

    # Check 16: Starts with special character
    if role_raw and role_raw[0] in ROLE_PATTERNS['special_chars']:
        return ("role_starts_special_char", "Role cannot start with a special character.")

    # Check 17: Achievement/stat statements
    for pattern in ROLE_PATTERNS['achievement_patterns']:
        if re.search(pattern, role_raw, re.IGNORECASE):
            return ("role_achievement_statement", "Role appears to be an achievement statement, not a job title.")

    # Check 18: Incomplete titles (ending with "of")
    for pattern in ROLE_PATTERNS['incomplete_patterns']:
        if re.search(pattern, role_lower.strip()):
            return ("role_incomplete_title", "Role appears incomplete (ends with 'of').")

    # Check 19: Contains company name
    for pattern in ROLE_PATTERNS['company_patterns']:
        if re.search(pattern, role_raw, re.IGNORECASE):
            return ("role_contains_company", "Role should not contain company name (use separate field).")

    # Check 20: Contains emojis
    if ROLE_EMOJI_RE.search(role_raw):
        return ("role_contains_emoji", "Role cannot contain emojis.")

    # Check 21: Hiring markers
    for pattern in ROLE_PATTERNS['hiring_patterns']:
        if re.search(pattern, role_lower):
            return ("role_hiring_marker", "Role contains hiring/recruiting markers.")

    # Check 22: Bio/description phrases
    for pattern in ROLE_PATTERNS['bio_patterns']:
        if re.search(pattern, role_lower):
            return ("role_bio_description", "Role appears to be a bio description, not a job title.")

    # Check 23: Long role without job keywords
    if len(role_raw) > thresholds['long_role_threshold']:
        if not any(kw in role_lower for kw in ROLE_PATTERNS['job_keywords']):
            return ("role_no_job_keywords", "Long role doesn't contain recognizable job title keywords.")

    # Check 24: Gibberish (no vowels)
    if len(letters_only) > 5:
        vowels = sum(1 for c in letters_only.lower() if c in 'aeiou')
        if vowels / len(letters_only) < thresholds['min_vowel_ratio']:
            return ("role_gibberish", "Role appears to be gibberish (no vowels).")

    return (None, None)  # Passed all checks


# ============================================================
# Description Sanity Check Function
# ============================================================

def check_description_sanity(desc_raw: str) -> tuple:
    """
    Validate company description format - returns (error_code, error_message) or (None, None) if valid.
    
    Catches garbage descriptions at gateway BEFORE entering validation queue.
    Common issues from miner submissions:
    - Truncated descriptions ending with "..."
    - Garbled Unicode (e.g., "ä½LinkedIn é—œæ³¨è€…")
    - LinkedIn follower count patterns (e.g., "Company | 2457 followers on LinkedIn")
    - Too short to be meaningful
    """
    desc_raw = desc_raw.strip()
    desc_lower = desc_raw.lower()
    letters_only = re.sub(r'[^a-zA-Z]', '', desc_raw)
    
    # ==========================================
    # Thresholds
    # ==========================================
    MIN_LENGTH = 70          # Minimum 70 characters
    MAX_LENGTH = 2000        # Maximum 2000 characters
    MIN_LETTERS = 50         # Must have at least 50 letters
    MIN_VOWEL_RATIO = 0.15   # At least 15% vowels (to catch gibberish)
    
    # ==========================================
    # Check 1: Too short
    # ==========================================
    if len(desc_raw) < MIN_LENGTH:
        return ("desc_too_short", f"Description too short ({len(desc_raw)} chars). Minimum {MIN_LENGTH} characters required.")
    
    # ==========================================
    # Check 2: Too long
    # ==========================================
    if len(desc_raw) > MAX_LENGTH:
        return ("desc_too_long", f"Description too long ({len(desc_raw)} chars). Maximum {MAX_LENGTH} characters allowed.")
    
    # ==========================================
    # Check 3: No letters
    # ==========================================
    if not any(c.isalpha() for c in desc_raw):
        return ("desc_no_letters", "Description must contain letters.")
    
    # ==========================================
    # Check 4: Too few letters
    # ==========================================
    if len(letters_only) < MIN_LETTERS:
        return ("desc_too_few_letters", f"Description must contain at least {MIN_LETTERS} letters.")
    
    # ==========================================
    # Check 5: Truncated description (ends with "...")
    # ==========================================
    # Miners are submitting truncated LinkedIn descriptions
    if desc_raw.rstrip().endswith('...'):
        return ("desc_truncated", "Description appears truncated (ends with '...'). Please provide complete description.")
    
    # ==========================================
    # Check 6: LinkedIn follower count pattern (English)
    # ==========================================
    # Pattern: "Company | 2457 followers on LinkedIn" - this is scraped junk, not a description
    # Also catches without pipe: "34,857 followers on LinkedIn"
    if re.search(r'\d[\d,\.]*\s*followers?\s*(on\s*)?linkedin', desc_lower):
        return ("desc_linkedin_followers", "Description contains LinkedIn follower count instead of actual company description.")
    
    # ==========================================
    # Check 6b: LinkedIn follower patterns (non-English)
    # ==========================================
    # Spanish: "seguidores en LinkedIn"
    # French: "abonnés" 
    # German: "Follower:innen auf LinkedIn"
    # Czech: "sledujících uživatelů na LinkedIn"
    # Arabic: "متابع" or "من المتابعين"
    # Thai: "ผู้ติดตาม X คนบน LinkedIn"
    linkedin_foreign_patterns = [
        r'\d[\d,\.]*\s*seguidores?\s*(en\s*)?linkedin',  # Spanish
        r'\d[\d,\.]*\s*abonnés?',  # French
        r'\d[\d,\.]*\s*follower:?innen\s*(auf\s*)?linkedin',  # German
        r'\d[\d,\.]*\s*sledujících',  # Czech
        r'متابع.*linkedin',  # Arabic
        r'ผู้ติดตาม.*linkedin',  # Thai
    ]
    for pattern in linkedin_foreign_patterns:
        if re.search(pattern, desc_lower, re.IGNORECASE):
            return ("desc_linkedin_foreign", "Description contains non-English LinkedIn metadata instead of actual company description.")
    
    # ==========================================
    # Check 6c: Thai text mixed with English
    # ==========================================
    # Thai characters indicate scraped LinkedIn with wrong locale
    thai_pattern = re.compile(r'[\u0e00-\u0e7f]')
    if thai_pattern.search(desc_raw):
        latin_count = len(re.findall(r'[a-zA-Z]', desc_raw))
        thai_count = len(thai_pattern.findall(desc_raw))
        # If Thai is mixed with significant Latin text, it's scraped junk
        if latin_count > 20 and thai_count > 3:
            return ("desc_thai_mixed", "Description contains Thai text mixed with English (scraped LinkedIn metadata).")
    
    # ==========================================
    # Check 6d: Website navigation/UI text
    # ==========================================
    # Catches: "Follow · Report this company; Close menu"
    # These are scraped from LinkedIn UI, not actual descriptions
    nav_patterns = [
        r'report\s+this\s+company',
        r'close\s+menu',
        r'view\s+all\s*[\.;]?\s*about\s+us',
        r'follow\s*[·•]\s*report',
        r'external\s+(na\s+)?link\s+(for|para)',  # Filipino/Spanish
        r'enlace\s+externo\s+para',  # Spanish
        r'laki\s+ng\s+kompanya',  # Filipino
        r'tamaño\s+de\s+la\s+empresa',  # Spanish  
        r'webbplats:\s*http',  # Swedish
        r'nettsted:\s*http',  # Norwegian
        r'sitio\s+web:\s*http',  # Spanish
        r'om\s+oss\.',  # Norwegian "About us."
    ]
    for pattern in nav_patterns:
        if re.search(pattern, desc_lower):
            return ("desc_navigation_text", "Description contains website navigation/UI text instead of actual company description.")
    
    # ==========================================
    # Check 7: Non-Latin/garbled Unicode characters
    # ==========================================
    # Catches: "ä½LinkedIn é—œæ³¨è€…ã€‚" type garbage
    # Allow: Basic Latin, Extended Latin (accents), common punctuation
    # Block: CJK characters mixed with English (indicates encoding issues)
    
    # Check for CJK characters (Chinese/Japanese/Korean) - these indicate garbled encoding
    cjk_pattern = re.compile(r'[\u4e00-\u9fff\u3400-\u4dbf\u3040-\u309f\u30a0-\u30ff]')
    if cjk_pattern.search(desc_raw):
        # If there's CJK mixed with Latin letters, it's likely garbled
        latin_count = len(re.findall(r'[a-zA-Z]', desc_raw))
        cjk_count = len(cjk_pattern.findall(desc_raw))
        
        # If CJK is mixed with significant Latin text, it's garbled
        if latin_count > 20 and cjk_count > 0:
            return ("desc_garbled_unicode", "Description contains garbled Unicode characters. Please provide clean text.")
    
    # ==========================================
    # Check 7b: Arabic text mixed with English
    # ==========================================
    arabic_pattern = re.compile(r'[\u0600-\u06ff]')
    if arabic_pattern.search(desc_raw):
        latin_count = len(re.findall(r'[a-zA-Z]', desc_raw))
        arabic_count = len(arabic_pattern.findall(desc_raw))
        # If Arabic is mixed with significant Latin text, it's scraped junk
        if latin_count > 20 and arabic_count > 3:
            return ("desc_arabic_mixed", "Description contains Arabic text mixed with English (scraped LinkedIn metadata).")
    
    # ==========================================
    # Check 8: Gibberish (no vowels in long text)
    # ==========================================
    if len(letters_only) > 30:
        vowels = sum(1 for c in letters_only.lower() if c in 'aeiou')
        if vowels / len(letters_only) < MIN_VOWEL_RATIO:
            return ("desc_gibberish", "Description appears to be gibberish (insufficient vowels).")
    
    # ==========================================
    # Check 9: Just company name repeated or placeholder
    # ==========================================
    placeholders = [
        "company description",
        "no description",
        "n/a",
        "none",
        "not available",
        "lorem ipsum",
        "test description",
        "placeholder",
        "description here",
        "enter description",
    ]
    for placeholder in placeholders:
        if desc_lower.strip() == placeholder or desc_lower.startswith(placeholder + " "):
            return ("desc_placeholder", "Description appears to be a placeholder, not actual company information.")
    
    # ==========================================
    # Check 10: Repeated character 5+ times (spam)
    # ==========================================
    if re.search(r'(.)\1{4,}', desc_raw):
        return ("desc_repeated_chars", "Description contains repeated characters (spam pattern).")
    
    # ==========================================
    # Check 11: Just a URL
    # ==========================================
    # Description shouldn't be ONLY a URL
    url_pattern = re.compile(r'^https?://\S+$')
    if url_pattern.match(desc_raw.strip()):
        return ("desc_just_url", "Description cannot be just a URL. Please provide actual company description.")
    
    # ==========================================
    # Check 12: Contains email as main content
    # ==========================================
    email_pattern = re.compile(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}')
    emails_found = email_pattern.findall(desc_raw)
    if emails_found:
        # If email takes up significant portion, reject
        email_chars = sum(len(e) for e in emails_found)
        if email_chars > len(desc_raw) * 0.3:
            return ("desc_mostly_email", "Description appears to contain contact info instead of company description.")
    
    # ==========================================
    # Check 13: Starts with pipe or special formatting junk
    # ==========================================
    if desc_raw.startswith('|') or desc_raw.startswith(' |'):
        return ("desc_formatting_junk", "Description contains formatting artifacts. Please provide clean text.")
    
    return (None, None)  # Passed all checks


# ============================================================
# Field Normalization Helper
# ============================================================

def normalize_lead_fields(lead_blob: dict) -> dict:
    """
    Normalize lead fields for standardized storage in the database.
    
    This function:
    1. Normalizes geographic fields (city/state/country) using geo_normalize
       - Standardizes variations: "SF" -> "San Francisco", "CA" -> "California"
       - Infers country from state if missing: ("NYC", "NY", "") -> "United States"
       - Handles alternate names: "Bombay" -> "Mumbai"
    2. Title-cases other text fields
    3. Preserves URLs and technical fields
    
    Called BEFORE storing lead in leads_private to ensure consistent formatting.
    
    NOTE: This does NOT affect validation - automated_checks.py uses .lower()
    for all comparisons, so capitalization doesn't impact verification.
    """
    # ================================================================
    # Step 1: Geographic normalization (city, state, country)
    # ================================================================
    # This handles:
    # - "SF", "CA", "USA" -> "San Francisco", "California", "United States"
    # - "nyc", "ny", "" -> "New York City", "New York", "United States" (country inferred!)
    # - "Bombay" -> "Mumbai", "Peking" -> "Beijing" (alternate names)
    city = lead_blob.get("city", "")
    state = lead_blob.get("state", "")
    country = lead_blob.get("country", "")
    
    norm_city, norm_state, norm_country = normalize_location(city, state, country)
    
    # Update with normalized values
    if norm_city:
        lead_blob["city"] = norm_city
    if norm_state:
        lead_blob["state"] = norm_state
    if norm_country:
        lead_blob["country"] = norm_country
    
    # ================================================================
    # Step 2: Title-case other text fields
    # ================================================================
    # Note: city/state already handled above, so removed from this list
    TITLE_CASE_FIELDS = [
        "industry",         # e.g., "financial services" → "Financial Services"
        "sub_industry",     # e.g., "investment banking" → "Investment Banking"
        "role",             # e.g., "vice president of sales" → "Vice President Of Sales"
        "full_name",        # e.g., "john smith" → "John Smith"
        "first",            # e.g., "john" → "John"
        "last",             # e.g., "smith" → "Smith"
        "business",         # e.g., "acme corporation" → "Acme Corporation"
    ]
    
    # Fields to lowercase (email should always be lowercase)
    LOWERCASE_FIELDS = [
        "email",
    ]
    
    # Fields to preserve as-is (URLs, hashes, technical data)
    # These are NOT modified: linkedin, website, source_url, company_linkedin, etc.
    
    for field in TITLE_CASE_FIELDS:
        if field in lead_blob and isinstance(lead_blob[field], str) and lead_blob[field].strip():
            lead_blob[field] = lead_blob[field].strip().title()
    
    for field in LOWERCASE_FIELDS:
        if field in lead_blob and isinstance(lead_blob[field], str) and lead_blob[field].strip():
            lead_blob[field] = lead_blob[field].strip().lower()
    
    return lead_blob


# ============================================================
# Request Models
# ============================================================

class SubmitLeadPayload(BaseModel):
    """Payload for submit request"""
    lead_id: str = Field(..., description="UUID of lead")


class SubmitLeadEvent(BaseModel):
    """
    Event for finalizing lead submission after upload.
    
    Miner signs this event after uploading to S3.
    Gateway verifies uploaded blob matches committed hash.
    """
    event_type: str = "SUBMIT_LEAD"
    actor_hotkey: str = Field(..., description="Miner's SS58 address")
    nonce: str = Field(..., description="UUID v4 nonce")
    ts: datetime = Field(..., description="ISO timestamp")
    payload_hash: str = Field(..., description="SHA256 of payload")
    build_id: str = Field(default="miner-client", description="Client build ID")
    signature: str = Field(..., description="Ed25519 signature")
    payload: SubmitLeadPayload


# ============================================================
# POST /submit - Verify and finalize lead submission
# ============================================================

@router.post("/")
async def submit_lead(event: SubmitLeadEvent):
    """
    Verify uploaded lead blob and finalize submission.
    
    Called by miner after uploading lead blob to S3 via presigned URL.
    
    Flow (BRD Section 4.1, Steps 5-6):
    1. Verify payload hash
    2. Verify wallet signature
    3. Check actor is registered miner
    4. Verify nonce is fresh
    5. Verify timestamp within tolerance
    6. Fetch SUBMISSION_REQUEST event to get committed lead_blob_hash
    7. Verify uploaded blob from S3 matches lead_blob_hash
    8. SUCCESS PATH (if S3 verifies):
        - Log STORAGE_PROOF event for S3
        - Store lead in leads_private table
        - Log SUBMISSION event
        - Return {status: "accepted", lead_id, merkle_proof}
    9. FAILURE PATH (if verification fails):
        - Log UPLOAD_FAILED event
        - Return HTTPException 400
    
    Args:
        event: SubmitLeadEvent with lead_id and miner signature
    
    Returns:
        {
            "status": "accepted",
            "lead_id": "uuid",
            "storage_backends": ["s3"],
            "merkle_proof": ["hash1", "hash2", ...],
            "submission_ts": "ISO timestamp"
        }
    
    Raises:
        400: Bad request (payload hash, nonce, timestamp, verification failed)
        403: Forbidden (invalid signature, not registered, not miner)
        404: SUBMISSION_REQUEST not found
        500: Server error
    
    Security:
        - Ed25519 signature verification
        - Nonce replay protection
        - Hash verification (prevents blob substitution)
        - Only registered miners can submit
    """
    
    import uuid  # For generating nonces for transparency log events
    
    print(f"\n🔍 POST /submit called - lead_id={event.payload.lead_id}")
    
    # ========================================
    # Step 0: Quick rate limit check (BEFORE expensive operations)
    # ========================================
    # This is a DoS protection mechanism - we do a quick READ-ONLY check
    # using only the actor_hotkey field BEFORE any expensive crypto operations.
    # 
    # NOTE: This is a preliminary check only. The actual atomic reservation
    # happens in Step 2.5 AFTER signature verification (to prevent attackers
    # from exhausting a victim's rate limit with fake requests).
    print("🔍 Step 0: Quick rate limit check...")
    from gateway.utils.rate_limiter import check_rate_limit
    
    allowed, reason, stats = check_rate_limit(event.actor_hotkey)
    if not allowed:
        print(f"❌ Rate limit exceeded for {event.actor_hotkey[:20]}...")
        print(f"   Reason: {reason}")
        print(f"   Stats: {stats}")
        
        # Log RATE_LIMIT_HIT event to TEE buffer (for transparency)
        try:
            from gateway.utils.logger import log_event
            
            rate_limit_event = {
                "event_type": "RATE_LIMIT_HIT",
                "actor_hotkey": event.actor_hotkey,
                "nonce": str(uuid.uuid4()),
                "ts": datetime.utcnow().isoformat(),
                "payload_hash": hashlib.sha256(json.dumps({
                    "lead_id": event.payload.lead_id,
                    "reason": reason,
                    "stats": stats
                }, sort_keys=True).encode()).hexdigest(),
                "build_id": "gateway",
                "signature": "rate_limit_check",  # No signature needed (gateway-generated)
                "payload": {
                    "lead_id": event.payload.lead_id,
                    "reason": reason,
                    "stats": stats
                }
            }
            
            await log_event(rate_limit_event)
            print(f"   ✅ Logged RATE_LIMIT_HIT to TEE buffer")
        except Exception as e:
            print(f"   ⚠️  Failed to log RATE_LIMIT_HIT: {e}")
        
        # Return 429 Too Many Requests
        raise HTTPException(
            status_code=429,
            detail={
                "error": "rate_limit_exceeded",
                "message": reason,
                "stats": stats
            }
        )
    
    print(f"🔍 Step 0 complete: Preliminary check OK (submissions={stats['submissions']}, rejections={stats['rejections']})")
    
    # ========================================
    # Step 1: Verify payload hash
    # ========================================
    print("🔍 Step 1: Verifying payload hash...")
    computed_hash = compute_payload_hash(event.payload.model_dump())
    if computed_hash != event.payload_hash:
        raise HTTPException(
            status_code=400,
            detail=f"Payload hash mismatch: expected {event.payload_hash[:16]}..., got {computed_hash[:16]}..."
        )
    print("🔍 Step 1 complete: Payload hash valid")
    
    # ========================================
    # Step 2: Verify wallet signature
    # ========================================
    print("🔍 Step 2: Verifying signature...")
    message = construct_signed_message(event)
    is_valid = verify_wallet_signature(message, event.signature, event.actor_hotkey)
    
    if not is_valid:
        raise HTTPException(
            status_code=403,
            detail="Invalid signature"
        )
    print("🔍 Step 2 complete: Signature valid")
    
    # ========================================
    # ========================================
    # Step 2.5: Check actor is registered miner BEFORE reserving slot
    # ========================================
    # CRITICAL: Registration check MUST happen BEFORE reserve_submission_slot()
    # Otherwise, unregistered hotkeys get their submissions counter incremented
    # even though they fail registration (causing 216 hotkeys with submissions > 0
    # when only 128 UIDs exist in the subnet)
    print("🔍 Step 2.5: Checking registration...")
    import asyncio
    try:
        is_registered, role = await asyncio.wait_for(
            is_registered_hotkey_async(event.actor_hotkey),  # Direct async call (no thread wrapper)
            timeout=45.0  # 45 second timeout for metagraph query (cache refresh can be slow under load)
        )
    except asyncio.TimeoutError:
        print(f"❌ Metagraph query timed out after 45s for {event.actor_hotkey[:20]}...")
        raise HTTPException(
            status_code=504,
            detail="Metagraph query timeout - please retry in a moment (cache warming)"
        )
    
    if not is_registered:
        raise HTTPException(
            status_code=403,
            detail="Hotkey not registered on subnet"
        )
    
    if role != "miner":
        raise HTTPException(
            status_code=403,
            detail="Only miners can submit leads"
        )
    print(f"🔍 Step 2.5 complete: Miner registered (hotkey={event.actor_hotkey[:10]}...)")
    
    # ========================================
    # Step 3: Reserve submission slot (atomic)
    # ========================================
    # Now that we've verified the signature AND registration, we KNOW this is a real registered miner.
    # We atomically reserve a submission slot to prevent race conditions.
    # 
    # RACE CONDITION FIX:
    # Previously, check_rate_limit() and increment_submission() were separate,
    # allowing multiple simultaneous requests to all pass the check before any
    # incremented. Now we atomically check AND increment in one operation.
    print("🔍 Step 3: Reserving submission slot (atomic)...")
    from gateway.utils.rate_limiter import reserve_submission_slot, mark_submission_failed
    
    slot_reserved, reservation_reason, reservation_stats = reserve_submission_slot(event.actor_hotkey)
    if not slot_reserved:
        print(f"❌ Could not reserve submission slot for {event.actor_hotkey[:20]}...")
        print(f"   Reason: {reservation_reason}")
        print(f"   Stats: {reservation_stats}")
        
        # Return 429 Too Many Requests
        raise HTTPException(
            status_code=429,
            detail={
                "error": "rate_limit_exceeded",
                "message": reservation_reason,
                "stats": reservation_stats
            }
        )
    
    print(f"🔍 Step 3 complete: Slot reserved (submissions={reservation_stats['submissions']}/{reservation_stats['max_submissions']})")
    
    # From this point on, a slot is RESERVED. If processing fails, we must call
    # mark_submission_failed() to increment the rejections counter.
    # If processing succeeds, the slot is already consumed (no further action needed).
    
    # ========================================
    # Step 4: Verify nonce format and freshness
    # ========================================
    print("🔍 Step 4: Verifying nonce...")
    if not validate_nonce_format(event.nonce):
        raise HTTPException(
            status_code=400,
            detail="Invalid nonce format (must be UUID v4)"
        )
    
    if not check_and_store_nonce(event.nonce, event.actor_hotkey):
        raise HTTPException(
            status_code=400,
            detail="Nonce already used (replay attack detected)"
        )
    print("🔍 Step 4 complete: Nonce valid")
    
    # ========================================
    # Step 5: Verify timestamp
    # ========================================
    print("🔍 Step 5: Verifying timestamp...")
    from datetime import timezone as tz
    from gateway.config import TIMESTAMP_TOLERANCE_SECONDS
    
    now = datetime.now(tz.utc)
    event_ts = event.ts if event.ts.tzinfo else event.ts.replace(tzinfo=tz.utc)
    time_diff = abs((now - event_ts).total_seconds())
    
    if time_diff > TIMESTAMP_TOLERANCE_SECONDS:
        raise HTTPException(
            status_code=400,
            detail=f"Timestamp out of range: {time_diff:.0f}s (max: {TIMESTAMP_TOLERANCE_SECONDS}s)"
        )
    print(f"🔍 Step 5 complete: Timestamp valid (diff={time_diff:.2f}s)")
    
    # ========================================
    # Step 6: Fetch SUBMISSION_REQUEST event
    # ========================================
    print(f"🔍 Step 6: Fetching SUBMISSION_REQUEST for lead_id={event.payload.lead_id}...")
    try:
        # Query directly for the specific lead_id using JSONB operator
        # This avoids the Supabase 1000 row default limit issue when miners have many submissions
        result = supabase.table("transparency_log") \
            .select("*") \
            .eq("event_type", "SUBMISSION_REQUEST") \
            .eq("actor_hotkey", event.actor_hotkey) \
            .eq("payload->>lead_id", event.payload.lead_id) \
            .limit(1) \
            .execute()
        
        print(f"🔍 Found {len(result.data) if result.data else 0} SUBMISSION_REQUEST events for lead_id={event.payload.lead_id[:8]}...")
        
        if not result.data:
            raise HTTPException(
                status_code=404,
                detail=f"SUBMISSION_REQUEST not found for lead_id={event.payload.lead_id}"
            )
        
        submission_request = result.data[0]
        
        # Extract committed lead_blob_hash and email_hash
        payload = submission_request.get("payload", {})
        if isinstance(payload, str):
            payload = json.loads(payload)
        
        committed_lead_blob_hash = payload.get("lead_blob_hash")
        committed_email_hash = payload.get("email_hash")
        
        if not committed_lead_blob_hash:
            raise HTTPException(
                status_code=500,
                detail="SUBMISSION_REQUEST missing lead_blob_hash"
            )
        
        if not committed_email_hash:
            raise HTTPException(
                status_code=500,
                detail="SUBMISSION_REQUEST missing email_hash"
            )
        
        print(f"🔍 Step 6 complete: Found SUBMISSION_REQUEST")
        print(f"   Committed lead_blob_hash: {committed_lead_blob_hash[:32]}...{committed_lead_blob_hash[-8:]}")
        print(f"   Committed email_hash: {committed_email_hash[:32]}...{committed_email_hash[-8:]}")
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error fetching SUBMISSION_REQUEST: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch SUBMISSION_REQUEST: {str(e)}"
        )
    
    # ========================================
    # Step 6.5: Check for duplicate email (PUBLIC - transparency_log)
    # ========================================
    # Uses transparency_log for VERIFIABLE fairness - miners can query same data
    # 
    # Logic:
    # 1. Check for CONSENSUS_RESULT events with this email_hash
    # 2. If most recent consensus is 'deny' → ALLOW resubmission (rejected leads can retry)
    # 3. If most recent consensus is 'approve' → BLOCK (already approved)
    # 4. If NO consensus yet but SUBMISSION exists → BLOCK (still processing)
    # 5. If no records at all → ALLOW (new email)
    #
    # This is 100% verifiable: miners can run the EXACT same query to check fairness
    print(f"🔍 Step 6.5: Checking for duplicate email (using transparency_log)...")
    try:
        # Step 1: Check for CONSENSUS_RESULT with this email_hash
        # This tells us the final outcome of any previous submission with this email
        consensus_check = supabase.table("transparency_log") \
            .select("payload, created_at") \
            .eq("email_hash", committed_email_hash) \
            .eq("event_type", "CONSENSUS_RESULT") \
            .order("created_at", desc=True) \
            .limit(1) \
            .execute()
        
        if consensus_check.data:
            # There's a consensus result for this email
            consensus = consensus_check.data[0]
            consensus_payload = consensus.get("payload", {})
            if isinstance(consensus_payload, str):
                consensus_payload = json.loads(consensus_payload)
            
            final_decision = consensus_payload.get("final_decision")
            consensus_lead_id = consensus_payload.get("lead_id", "unknown")
            consensus_time = consensus.get("created_at")
            
            print(f"   Found CONSENSUS_RESULT: lead={consensus_lead_id[:10]}..., decision={final_decision}, time={consensus_time}")
            
            if final_decision == "approve":
                # Already approved - BLOCK duplicate
                print(f"❌ Duplicate email detected - already APPROVED!")
                print(f"   Email hash: {committed_email_hash[:32]}...")
                print(f"   Original lead: {consensus_lead_id[:10]}...")
                
                # Mark submission as failed
                updated_stats = mark_submission_failed(event.actor_hotkey)
                print(f"   📊 Rate limit updated: submissions={updated_stats['submissions']}/{MAX_SUBMISSIONS_PER_DAY}, rejections={updated_stats['rejections']}/{MAX_REJECTIONS_PER_DAY}")
                
                # Log VALIDATION_FAILED event
                try:
                    from gateway.utils.logger import log_event
                    
                    validation_failed_event = {
                        "event_type": "VALIDATION_FAILED",
                        "actor_hotkey": event.actor_hotkey,
                        "nonce": str(uuid.uuid4()),
                        "ts": datetime.now(tz.utc).isoformat(),
                        "payload_hash": hashlib.sha256(json.dumps({
                            "lead_id": event.payload.lead_id,
                            "reason": "duplicate_email_approved",
                            "email_hash": committed_email_hash
                        }, sort_keys=True).encode()).hexdigest(),
                        "build_id": "gateway",
                        "signature": "duplicate_check",
                        "payload": {
                            "lead_id": event.payload.lead_id,
                            "reason": "duplicate_email_approved",
                            "email_hash": committed_email_hash,
                            "original_lead_id": consensus_lead_id,
                            "original_decision": "approve",
                            "miner_hotkey": event.actor_hotkey
                        }
                    }
                    
                    await log_event(validation_failed_event)
                    print(f"   ✅ Logged VALIDATION_FAILED (duplicate_approved) to TEE buffer")
                except Exception as e:
                    print(f"   ⚠️  Failed to log VALIDATION_FAILED: {e}")
                
                raise HTTPException(
                    status_code=409,
                    detail={
                        "error": "duplicate_email",
                        "message": "This email has already been approved by the network",
                        "email_hash": committed_email_hash,
                        "original_submission": {
                            "lead_id": consensus_lead_id,
                            "final_decision": "approve",
                            "consensus_at": consensus_time
                        },
                        "rate_limit_stats": {
                            "submissions": updated_stats["submissions"],
                            "max_submissions": MAX_SUBMISSIONS_PER_DAY,
                            "rejections": updated_stats["rejections"],
                            "max_rejections": MAX_REJECTIONS_PER_DAY,
                            "reset_at": updated_stats["reset_at"]
                        }
                    }
                )
            
            elif final_decision == "deny":
                # Was rejected - ALLOW resubmission!
                print(f"✅ Email was previously REJECTED - allowing resubmission")
                print(f"   Previous lead: {consensus_lead_id[:10]}... was denied")
                print(f"   Miner can now submit corrected lead data")
                # Continue to next step (no raise, no block)
            
            else:
                # Unknown decision - treat as block to be safe
                print(f"⚠️  Unknown consensus decision '{final_decision}' - blocking for safety")
                raise HTTPException(
                    status_code=409,
                    detail={
                        "error": "duplicate_email",
                        "message": f"This email has an unknown consensus state: {final_decision}",
                        "email_hash": committed_email_hash
                    }
                )
        
        else:
            # No CONSENSUS_RESULT found - check if there's a pending submission
            print(f"   No CONSENSUS_RESULT found for this email")
            
            # Check for any SUBMISSION with this email (still processing)
            # NOTE: SUBMISSION (not SUBMISSION_REQUEST) means lead was actually accepted into queue
            # SUBMISSION_REQUEST is just the presign intent - doesn't mean lead was accepted
            submission_check = supabase.table("transparency_log") \
                .select("payload, created_at, actor_hotkey") \
                .eq("email_hash", committed_email_hash) \
                .eq("event_type", "SUBMISSION") \
                .order("created_at", desc=True) \
                .limit(1) \
                .execute()
            
            if submission_check.data:
                # There's a submission but no consensus yet - BLOCK (still processing)
                existing_submission = submission_check.data[0]
                existing_payload = existing_submission.get("payload", {})
                if isinstance(existing_payload, str):
                    existing_payload = json.loads(existing_payload)
                
                existing_lead_id = existing_payload.get("lead_id", "unknown")
                existing_time = existing_submission.get("created_at")
                existing_miner = existing_submission.get("actor_hotkey", "unknown")
                
                print(f"❌ Duplicate email detected - still PROCESSING!")
                print(f"   Email hash: {committed_email_hash[:32]}...")
                print(f"   Pending lead: {existing_lead_id[:10]}..., miner={existing_miner[:10]}..., ts={existing_time}")
                
                # Mark submission as failed
                updated_stats = mark_submission_failed(event.actor_hotkey)
                print(f"   📊 Rate limit updated: submissions={updated_stats['submissions']}/{MAX_SUBMISSIONS_PER_DAY}, rejections={updated_stats['rejections']}/{MAX_REJECTIONS_PER_DAY}")
                
                # Log VALIDATION_FAILED event
                try:
                    from gateway.utils.logger import log_event
                    
                    validation_failed_event = {
                        "event_type": "VALIDATION_FAILED",
                        "actor_hotkey": event.actor_hotkey,
                        "nonce": str(uuid.uuid4()),
                        "ts": datetime.now(tz.utc).isoformat(),
                        "payload_hash": hashlib.sha256(json.dumps({
                            "lead_id": event.payload.lead_id,
                            "reason": "duplicate_email_processing",
                            "email_hash": committed_email_hash
                        }, sort_keys=True).encode()).hexdigest(),
                        "build_id": "gateway",
                        "signature": "duplicate_check",
                        "payload": {
                            "lead_id": event.payload.lead_id,
                            "reason": "duplicate_email_processing",
                            "email_hash": committed_email_hash,
                            "original_lead_id": existing_lead_id,
                            "original_miner": existing_miner,
                            "miner_hotkey": event.actor_hotkey
                        }
                    }
                    
                    await log_event(validation_failed_event)
                    print(f"   ✅ Logged VALIDATION_FAILED (duplicate_processing) to TEE buffer")
                except Exception as e:
                    print(f"   ⚠️  Failed to log VALIDATION_FAILED: {e}")
                
                raise HTTPException(
                    status_code=409,
                    detail={
                        "error": "duplicate_email_processing",
                        "message": "This email is currently being processed by the network. Please wait for consensus.",
                        "email_hash": committed_email_hash,
                        "original_submission": {
                            "lead_id": existing_lead_id,
                            "submitted_at": existing_time,
                            "status": "pending_consensus"
                        },
                        "rate_limit_stats": {
                            "submissions": updated_stats["submissions"],
                            "max_submissions": MAX_SUBMISSIONS_PER_DAY,
                            "rejections": updated_stats["rejections"],
                            "max_rejections": MAX_REJECTIONS_PER_DAY,
                            "reset_at": updated_stats["reset_at"]
                        }
                    }
                )
        
            # No SUBMISSION either - new email!
            print(f"✅ No prior submission found - email is unique")
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"⚠️  Duplicate check error: {e}")
        import traceback
        traceback.print_exc()
        # Continue anyway - don't block submission on duplicate check failure
        # This prevents gateway outages if transparency_log is temporarily unavailable
        print(f"⚠️  Continuing with submission despite duplicate check error")
    
    # ========================================
    # Step 7: Verify S3 upload
    # ========================================
    print(f"🔍 Step 7: Verifying S3 upload...")
    s3_verified = verify_storage_proof(committed_lead_blob_hash, "s3")
    
    if s3_verified:
        print(f"✅ S3 verification successful")
    else:
        print(f"❌ S3 verification failed")
    
    # ========================================
    # Step 8: SUCCESS PATH - S3 verified
    # ========================================
    if s3_verified:
        print(f"🔍 Step 8: SUCCESS PATH - S3 verified")
        
        try:
            # Log STORAGE_PROOF event to TEE buffer (hardware-protected)
            from gateway.utils.logger import log_event
            import asyncio
            
            storage_proof_tee_seqs = {}
            
            # Log S3 storage proof
            mirror = "s3"
            storage_proof_payload = {
                "lead_id": event.payload.lead_id,
                "lead_blob_hash": committed_lead_blob_hash,
                "email_hash": committed_email_hash,
                "mirror": mirror,
                "verified": True
            }
            
            storage_proof_log_entry = {
                "event_type": "STORAGE_PROOF",
                "actor_hotkey": "gateway",
                "nonce": str(uuid.uuid4()),  # Generate fresh UUID for this event
                "ts": datetime.now(tz.utc).isoformat(),
                "payload_hash": hashlib.sha256(
                    json.dumps(storage_proof_payload, sort_keys=True).encode()
                ).hexdigest(),
                "build_id": "gateway",
                "signature": "gateway_internal",
                "payload": storage_proof_payload
            }
            
            print(f"   🔍 Logging STORAGE_PROOF for {mirror} to TEE buffer...")
            result = await log_event(storage_proof_log_entry)
            
            tee_sequence = result.get("sequence")
            storage_proof_tee_seqs[mirror] = tee_sequence
            print(f"   ✅ STORAGE_PROOF buffered in TEE for {mirror}: seq={tee_sequence}")
                
        except Exception as e:
            print(f"❌ Error logging STORAGE_PROOF: {e}")
            import traceback
            traceback.print_exc()
            # CRITICAL: If TEE write fails, request MUST fail
            print(f"🚨 CRITICAL: TEE buffer unavailable - failing request")
            raise HTTPException(
                status_code=503,
                detail=f"TEE buffer unavailable: {str(e)}"
            )
        
        # Fetch the lead blob from S3 to store in leads_private
        from gateway.utils.storage import s3_client
        from gateway.config import AWS_S3_BUCKET
        
        print(f"   🔍 Fetching lead blob from S3 for database storage...")
        object_key = f"leads/{committed_lead_blob_hash}.json"
        try:
            response = s3_client.get_object(Bucket=AWS_S3_BUCKET, Key=object_key)
            lead_blob = json.loads(response['Body'].read().decode('utf-8'))
            print(f"   ✅ Lead blob fetched from S3")
        except Exception as e:
            print(f"❌ Failed to fetch lead blob from S3: {e}")
            import traceback
            traceback.print_exc()
            raise HTTPException(
                status_code=500,
                detail=f"Failed to fetch lead blob: {str(e)}"
            )
        
        # ========================================
        # CRITICAL: Verify email hash matches committed value
        # ========================================
        # This prevents email substitution attacks where miner commits email_hash_A
        # but uploads lead_blob with different email_B to bypass duplicate detection.
        # 
        # Flow:
        # 1. Miner commits email_hash in SUBMISSION_REQUEST (for duplicate check)
        # 2. Miner uploads lead_blob with actual email
        # 3. Gateway verifies: SHA256(actual_email) == committed_email_hash
        # 4. MISMATCH → REJECT (prevents gaming duplicate detection)
        #
        # Performance: ~1 microsecond (SHA256 of ~50 byte email string)
        print(f"   🔍 Verifying email hash integrity...")
        actual_email = lead_blob.get("email", "").strip().lower()
        actual_email_hash = hashlib.sha256(actual_email.encode()).hexdigest()
        
        if actual_email_hash != committed_email_hash:
            print(f"❌ EMAIL HASH MISMATCH DETECTED!")
            print(f"   Committed email_hash: {committed_email_hash[:32]}...")
            print(f"   Actual email_hash:    {actual_email_hash[:32]}...")
            print(f"   This indicates miner tried to substitute email to bypass duplicate detection!")
            
            # Mark submission as failed
            updated_stats = mark_submission_failed(event.actor_hotkey)
            print(f"   📊 Rate limit updated: rejections={updated_stats['rejections']}/{MAX_REJECTIONS_PER_DAY}")
            
            # Log VALIDATION_FAILED event
            try:
                validation_failed_event = {
                    "event_type": "VALIDATION_FAILED",
                    "actor_hotkey": event.actor_hotkey,
                    "nonce": str(uuid.uuid4()),
                    "ts": datetime.now(tz.utc).isoformat(),
                    "payload_hash": hashlib.sha256(json.dumps({
                        "lead_id": event.payload.lead_id,
                        "reason": "email_hash_mismatch",
                        "committed_email_hash": committed_email_hash,
                        "actual_email_hash": actual_email_hash
                    }, sort_keys=True).encode()).hexdigest(),
                    "build_id": "gateway",
                    "signature": "email_hash_verification",
                    "payload": {
                        "lead_id": event.payload.lead_id,
                        "reason": "email_hash_mismatch",
                        "committed_email_hash": committed_email_hash,
                        "actual_email_hash": actual_email_hash,
                        "miner_hotkey": event.actor_hotkey
                    }
                }
                
                await log_event(validation_failed_event)
                print(f"   ✅ Logged VALIDATION_FAILED (email_hash_mismatch) to TEE buffer")
            except Exception as e:
                print(f"   ⚠️  Failed to log VALIDATION_FAILED: {e}")
            
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "email_hash_mismatch",
                    "message": "Email in uploaded lead does not match committed email hash. This is not allowed.",
                    "committed_email_hash": committed_email_hash[:16] + "...",
                    "rate_limit_stats": {
                        "submissions": updated_stats["submissions"],
                        "max_submissions": MAX_SUBMISSIONS_PER_DAY,
                        "rejections": updated_stats["rejections"],
                        "max_rejections": MAX_REJECTIONS_PER_DAY,
                        "reset_at": updated_stats["reset_at"]
                    }
                }
            )
        
        print(f"   ✅ Email hash verified: {actual_email_hash[:16]}... matches committed value")
        
        # ========================================
        # Compute LinkedIn combo hash for duplicate detection
        # ========================================
        # This creates a unique identifier for "person X at company Y"
        # to prevent duplicate submissions of the same person at the same company
        print(f"   🔍 Computing LinkedIn combo hash...")
        linkedin_url = lead_blob.get("linkedin", "")
        company_linkedin_url = lead_blob.get("company_linkedin", "")
        
        actual_linkedin_combo_hash = compute_linkedin_combo_hash(linkedin_url, company_linkedin_url)
        
        if actual_linkedin_combo_hash:
            print(f"   ✅ LinkedIn combo hash computed: {actual_linkedin_combo_hash[:16]}...")
            print(f"      Profile: {normalize_linkedin_url(linkedin_url, 'profile')}")
            print(f"      Company: {normalize_linkedin_url(company_linkedin_url, 'company')}")
        else:
            print(f"   ⚠️  Could not compute LinkedIn combo hash (invalid URLs)")
            print(f"      Profile URL: {linkedin_url[:50] if linkedin_url else 'MISSING'}...")
            print(f"      Company URL: {company_linkedin_url[:50] if company_linkedin_url else 'MISSING'}...")
            # Don't fail here - the required fields check below will catch missing fields
        
        # ========================================
        # Check for duplicate LinkedIn combo (person + company)
        # ========================================
        # Similar to email duplicate check, but for person+company combination
        # This prevents miners from resubmitting the same person at the same company
        # with a different email address.
        if actual_linkedin_combo_hash:
            print(f"   🔍 Checking for duplicate LinkedIn combo...")
            try:
                # Check for CONSENSUS_RESULT with this linkedin_combo_hash
                linkedin_consensus_check = supabase.table("transparency_log") \
                    .select("payload, created_at") \
                    .eq("linkedin_combo_hash", actual_linkedin_combo_hash) \
                    .eq("event_type", "CONSENSUS_RESULT") \
                    .order("created_at", desc=True) \
                    .limit(1) \
                    .execute()
                
                if linkedin_consensus_check.data:
                    # There's a consensus result for this person+company combo
                    linkedin_consensus = linkedin_consensus_check.data[0]
                    linkedin_consensus_payload = linkedin_consensus.get("payload", {})
                    if isinstance(linkedin_consensus_payload, str):
                        linkedin_consensus_payload = json.loads(linkedin_consensus_payload)
                    
                    linkedin_final_decision = linkedin_consensus_payload.get("final_decision")
                    linkedin_consensus_lead_id = linkedin_consensus_payload.get("lead_id", "unknown")
                    linkedin_consensus_time = linkedin_consensus.get("created_at")
                    
                    print(f"      Found CONSENSUS_RESULT: lead={linkedin_consensus_lead_id[:10]}..., decision={linkedin_final_decision}")
                    
                    if linkedin_final_decision == "approve":
                        # Already approved - BLOCK duplicate person+company
                        print(f"   ❌ Duplicate person+company detected - already APPROVED!")
                        
                        updated_stats = mark_submission_failed(event.actor_hotkey)
                        
                        try:
                            validation_failed_event = {
                                "event_type": "VALIDATION_FAILED",
                                "actor_hotkey": event.actor_hotkey,
                                "nonce": str(uuid.uuid4()),
                                "ts": datetime.now(tz.utc).isoformat(),
                                "payload_hash": hashlib.sha256(json.dumps({
                                    "lead_id": event.payload.lead_id,
                                    "reason": "duplicate_linkedin_combo_approved",
                                    "linkedin_combo_hash": actual_linkedin_combo_hash
                                }, sort_keys=True).encode()).hexdigest(),
                                "build_id": "gateway",
                                "signature": "linkedin_combo_duplicate_check",
                                "payload": {
                                    "lead_id": event.payload.lead_id,
                                    "reason": "duplicate_linkedin_combo_approved",
                                    "linkedin_combo_hash": actual_linkedin_combo_hash,
                                    "original_lead_id": linkedin_consensus_lead_id,
                                    "miner_hotkey": event.actor_hotkey
                                }
                            }
                            await log_event(validation_failed_event)
                        except Exception as e:
                            print(f"      ⚠️  Failed to log VALIDATION_FAILED: {e}")
                        
                        raise HTTPException(
                            status_code=409,
                            detail={
                                "error": "duplicate_linkedin_combo",
                                "message": "This person+company combination has already been approved. Same person at same company cannot be submitted with different email.",
                                "linkedin_combo_hash": actual_linkedin_combo_hash[:16] + "...",
                                "original_lead_id": linkedin_consensus_lead_id,
                                "rate_limit_stats": {
                                    "submissions": updated_stats["submissions"],
                                    "max_submissions": MAX_SUBMISSIONS_PER_DAY,
                                    "rejections": updated_stats["rejections"],
                                    "max_rejections": MAX_REJECTIONS_PER_DAY,
                                    "reset_at": updated_stats["reset_at"]
                                }
                            }
                        )
                    
                    elif linkedin_final_decision == "deny":
                        # Was rejected - allow resubmission
                        print(f"   ✅ LinkedIn combo was previously REJECTED - allowing resubmission")
                
                else:
                    # No CONSENSUS_RESULT - check for pending SUBMISSION
                    # NOTE: SUBMISSION (not SUBMISSION_REQUEST) means lead was actually accepted into queue
                    linkedin_submission_check = supabase.table("transparency_log") \
                        .select("payload, created_at, actor_hotkey") \
                        .eq("linkedin_combo_hash", actual_linkedin_combo_hash) \
                        .eq("event_type", "SUBMISSION") \
                        .order("created_at", desc=True) \
                        .limit(1) \
                        .execute()
                    
                    if linkedin_submission_check.data:
                        # There's a submission but no consensus yet - BLOCK (still processing)
                        existing_linkedin = linkedin_submission_check.data[0]
                        existing_linkedin_payload = existing_linkedin.get("payload", {})
                        if isinstance(existing_linkedin_payload, str):
                            existing_linkedin_payload = json.loads(existing_linkedin_payload)
                        
                        existing_linkedin_lead_id = existing_linkedin_payload.get("lead_id", "unknown")
                        existing_linkedin_time = existing_linkedin.get("created_at")
                        
                        print(f"   ❌ Duplicate person+company detected - still PROCESSING!")
                        print(f"      Pending lead: {existing_linkedin_lead_id[:10]}..., ts={existing_linkedin_time}")
                        
                        updated_stats = mark_submission_failed(event.actor_hotkey)
                        
                        raise HTTPException(
                            status_code=409,
                            detail={
                                "error": "duplicate_linkedin_combo_processing",
                                "message": "This person+company combination is currently being processed. Please wait for consensus.",
                                "linkedin_combo_hash": actual_linkedin_combo_hash[:16] + "...",
                                "original_lead_id": existing_linkedin_lead_id,
                                "rate_limit_stats": {
                                    "submissions": updated_stats["submissions"],
                                    "max_submissions": MAX_SUBMISSIONS_PER_DAY,
                                    "rejections": updated_stats["rejections"],
                                    "max_rejections": MAX_REJECTIONS_PER_DAY,
                                    "reset_at": updated_stats["reset_at"]
                                }
                            }
                        )
                    
                    # No prior submission - new person+company combo!
                    print(f"   ✅ No prior LinkedIn combo found - unique person+company")
            
            except HTTPException:
                raise
            except Exception as e:
                print(f"   ⚠️  LinkedIn combo duplicate check error: {e}")
                # Continue anyway - don't block on check failure
        
        # ========================================
        # CRITICAL: Validate Required Fields (README.md lines 239-258)
        # ========================================
        print(f"   🔍 Validating required fields...")
        
        REQUIRED_FIELDS = [
            "business",         # Company name
            "full_name",        # Contact full name
            "first",            # First name
            "last",             # Last name
            "email",            # Email address
            "role",             # Job title
            "website",          # Company website
            "industry",         # Primary industry (must match Crunchbase industry_group)
            "sub_industry",     # Sub-industry/niche (must match Crunchbase industry key)
            "country",          # Country (REQUIRED) - e.g., "United States", "Canada"
            "city",             # City (REQUIRED for all leads) - e.g., "San Francisco", "London"
            # "state" - REQUIRED for US only (validated in region validation section below)
            "linkedin",         # LinkedIn URL (person)
            "company_linkedin", # Company LinkedIn URL (for industry/sub_industry/description verification)
            "source_url",       # Source URL where lead was found
            "description",      # Company description 
            "employee_count"    # Company size/headcount 
        ]
        
        missing_fields = []
        for field in REQUIRED_FIELDS:
            value = lead_blob.get(field)
            if not value or (isinstance(value, str) and not value.strip()):
                missing_fields.append(field)
        
        if missing_fields:
            print(f"❌ Required fields validation failed: Missing {len(missing_fields)} field(s)")
            print(f"   Missing: {', '.join(missing_fields)}")
            
            # Mark submission as failed (FAILURE - missing required fields)
            # NOTE: Submission slot was already reserved in Step 2.5, just increment rejections
            updated_stats = mark_submission_failed(event.actor_hotkey)
            print(f"   📊 Rate limit updated: submissions={updated_stats['submissions']}/{MAX_SUBMISSIONS_PER_DAY}, rejections={updated_stats['rejections']}/{MAX_REJECTIONS_PER_DAY}")
            
            # Log VALIDATION_FAILED event to TEE buffer (for transparency)
            try:
                from gateway.utils.logger import log_event
                
                validation_failed_event = {
                    "event_type": "VALIDATION_FAILED",
                    "actor_hotkey": event.actor_hotkey,
                    "nonce": str(uuid.uuid4()),
                    "ts": datetime.utcnow().isoformat(),
                    "payload_hash": hashlib.sha256(json.dumps({
                        "lead_id": event.payload.lead_id,
                        "reason": "missing_required_fields",
                        "missing_fields": missing_fields
                    }, sort_keys=True).encode()).hexdigest(),
                    "build_id": "gateway",
                    "signature": "required_fields_check",  # Gateway-generated
                    "payload": {
                        "lead_id": event.payload.lead_id,
                        "reason": "missing_required_fields",
                        "missing_fields": missing_fields,
                        "miner_hotkey": event.actor_hotkey
                    }
                }
                
                await log_event(validation_failed_event)
                print(f"   ✅ Logged VALIDATION_FAILED to TEE buffer")
            except Exception as e:
                print(f"   ⚠️  Failed to log VALIDATION_FAILED: {e}")
            
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "missing_required_fields",
                    "message": f"Lead is missing {len(missing_fields)} required field(s)",
                    "missing_fields": missing_fields,
                    "required_fields": REQUIRED_FIELDS,
                    "rate_limit_stats": {
                        "submissions": updated_stats["submissions"],
                        "max_submissions": MAX_SUBMISSIONS_PER_DAY,
                        "rejections": updated_stats["rejections"],
                        "max_rejections": MAX_REJECTIONS_PER_DAY,
                        "reset_at": updated_stats["reset_at"]
                    }
                }
            )
        
        print(f"   ✅ All required fields present")

        # ========================================
        # EARLY EXIT: Role Format Sanity Check
        # ========================================
        # Catch obviously garbage roles at gateway BEFORE entering validation queue
        # Saves validator time and API costs by rejecting spam/garbage early
        # Checks loaded from role_patterns.json (24 checks total)
        print(f"   🔍 Validating role format (early sanity check)...")
        role_raw = lead_blob.get("role", "").strip()

        # Call comprehensive role sanity check function
        error_code, error_message = check_role_sanity(role_raw)
        role_sanity_error = (error_code, error_message) if error_code else None

        # Reject if any sanity check failed
        if role_sanity_error:
            error_code, error_message = role_sanity_error
            print(f"❌ Role sanity check failed: {error_code} - '{role_raw[:50]}{'...' if len(role_raw) > 50 else ''}'")

            updated_stats = mark_submission_failed(event.actor_hotkey)
            print(f"   📊 Rate limit updated: rejections={updated_stats['rejections']}/{MAX_REJECTIONS_PER_DAY}")

            # Log VALIDATION_FAILED event
            try:
                from datetime import timezone as tz_module
                validation_failed_event = {
                    "event_type": "VALIDATION_FAILED",
                    "actor_hotkey": event.actor_hotkey,
                    "nonce": str(uuid.uuid4()),
                    "ts": datetime.now(tz_module.utc).isoformat(),
                    "payload_hash": hashlib.sha256(json.dumps({
                        "lead_id": event.payload.lead_id,
                        "reason": error_code,
                        "role": role_raw[:100]
                    }, sort_keys=True).encode()).hexdigest(),
                    "build_id": "gateway",
                    "signature": "role_sanity_check",
                    "payload": {
                        "lead_id": event.payload.lead_id,
                        "reason": error_code,
                        "role": role_raw[:100],
                        "miner_hotkey": event.actor_hotkey
                    }
                }
                await log_event(validation_failed_event)
                print(f"   ✅ Logged VALIDATION_FAILED ({error_code}) to TEE buffer")
            except Exception as e:
                print(f"   ⚠️  Failed to log VALIDATION_FAILED: {e}")

            raise HTTPException(
                status_code=400,
                detail={
                    "error": error_code,
                    "message": error_message,
                    "role": role_raw[:100] + ("..." if len(role_raw) > 100 else ""),
                    "rate_limit_stats": {
                        "submissions": updated_stats["submissions"],
                        "max_submissions": MAX_SUBMISSIONS_PER_DAY,
                        "rejections": updated_stats["rejections"],
                        "max_rejections": MAX_REJECTIONS_PER_DAY,
                        "reset_at": updated_stats["reset_at"]
                    }
                }
            )

        print(f"   ✅ Role sanity check passed: '{role_raw[:40]}{'...' if len(role_raw) > 40 else ''}'")

        # ========================================
        # EARLY EXIT: Description Format Sanity Check
        # ========================================
        # Catch garbage descriptions at gateway BEFORE entering validation queue
        # Common issues: truncated "...", garbled Unicode, LinkedIn follower counts
        print(f"   🔍 Validating description format (early sanity check)...")
        desc_raw = lead_blob.get("description", "").strip()

        # Call comprehensive description sanity check function
        desc_error_code, desc_error_message = check_description_sanity(desc_raw)
        desc_sanity_error = (desc_error_code, desc_error_message) if desc_error_code else None

        # Reject if any sanity check failed
        if desc_sanity_error:
            desc_error_code, desc_error_message = desc_sanity_error
            print(f"❌ Description sanity check failed: {desc_error_code} - '{desc_raw[:80]}{'...' if len(desc_raw) > 80 else ''}'")

            updated_stats = mark_submission_failed(event.actor_hotkey)
            print(f"   📊 Rate limit updated: rejections={updated_stats['rejections']}/{MAX_REJECTIONS_PER_DAY}")

            # Log VALIDATION_FAILED event
            try:
                from datetime import timezone as tz_module
                validation_failed_event = {
                    "event_type": "VALIDATION_FAILED",
                    "actor_hotkey": event.actor_hotkey,
                    "nonce": str(uuid.uuid4()),
                    "ts": datetime.now(tz_module.utc).isoformat(),
                    "payload_hash": hashlib.sha256(json.dumps({
                        "lead_id": event.payload.lead_id,
                        "reason": desc_error_code,
                        "description": desc_raw[:200]
                    }, sort_keys=True).encode()).hexdigest(),
                    "build_id": "gateway",
                    "signature": "description_sanity_check",
                    "payload": {
                        "lead_id": event.payload.lead_id,
                        "reason": desc_error_code,
                        "description": desc_raw[:200],
                        "miner_hotkey": event.actor_hotkey
                    }
                }
                await log_event(validation_failed_event)
                print(f"   ✅ Logged VALIDATION_FAILED ({desc_error_code}) to TEE buffer")
            except Exception as e:
                print(f"   ⚠️  Failed to log VALIDATION_FAILED: {e}")

            raise HTTPException(
                status_code=400,
                detail={
                    "error": desc_error_code,
                    "message": desc_error_message,
                    "description": desc_raw[:200] + ("..." if len(desc_raw) > 200 else ""),
                    "rate_limit_stats": {
                        "submissions": updated_stats["submissions"],
                        "max_submissions": MAX_SUBMISSIONS_PER_DAY,
                        "rejections": updated_stats["rejections"],
                        "max_rejections": MAX_REJECTIONS_PER_DAY,
                        "reset_at": updated_stats["reset_at"]
                    }
                }
            )

        print(f"   ✅ Description sanity check passed: '{desc_raw[:60]}{'...' if len(desc_raw) > 60 else ''}'")

        # ========================================
        # Validate country/state/city logic
        # ========================================
        country_raw = lead_blob.get("country", "").strip()
        state = lead_blob.get("state", "").strip()
        city = lead_blob.get("city", "").strip()

        # Normalize country using geo_normalize (handles aliases + title case)
        country = normalize_country(country_raw)
        if country != country_raw:
            print(f"   📝 Country normalized: '{country_raw}' → '{country}'")

        # Validate location: country (199 valid), state (51 US states), city (exists in state/country)
        is_valid, rejection_reason = validate_location(city, state, country)

        if not is_valid:
            # Map rejection reasons to user-friendly error messages
            ERROR_MESSAGES = {
                "country_empty": ("invalid_country", "Country field is required."),
                "country_invalid": ("invalid_country", f"Country '{country_raw}' is not recognized. Use standard names like 'United States', 'Germany', etc."),
                "state_empty_for_usa": ("invalid_region_format", "United States leads require state field."),
                "state_invalid": ("invalid_region_format", f"State '{state}' is not a valid US state."),
                "city_empty": ("invalid_region_format", "City field is required."),
                "city_invalid_for_state": ("invalid_region_format", f"City '{city}' not found in {state}, {country}."),
                "city_invalid_for_country": ("invalid_region_format", f"City '{city}' not found in {country}."),
            }

            error_code, error_message = ERROR_MESSAGES.get(
                rejection_reason,
                ("invalid_region_format", f"Invalid location: {rejection_reason}")
            )

            print(f"❌ Location validation failed: {rejection_reason} - {city}/{state}/{country}")

            updated_stats = mark_submission_failed(event.actor_hotkey)

            raise HTTPException(
                status_code=400,
                detail={
                    "error": error_code,
                    "message": error_message,
                    "rejection_reason": rejection_reason,
                    "country": country,
                    "state": state,
                    "city": city,
                    "rate_limit_stats": {
                        "submissions": updated_stats["submissions"],
                        "max_submissions": MAX_SUBMISSIONS_PER_DAY,
                        "rejections": updated_stats["rejections"],
                        "max_rejections": MAX_REJECTIONS_PER_DAY
                    }
                }
            )
        
        # Validation 4: City and State fields cannot contain commas (anti-gaming)
        # This prevents miners from stuffing multiple cities/states into a single field
        if city and ',' in city:
            print(f"❌ City field contains comma (gaming attempt): '{city}'")
            
            updated_stats = mark_submission_failed(event.actor_hotkey)
            
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "invalid_city_format",
                    "message": "City field should contain only one city (no commas allowed)",
                    "city": city,
                    "rate_limit_stats": {
                        "submissions": updated_stats["submissions"],
                        "max_submissions": MAX_SUBMISSIONS_PER_DAY,
                        "rejections": updated_stats["rejections"],
                        "max_rejections": MAX_REJECTIONS_PER_DAY
                    }
                }
            )
        
        if state and ',' in state:
            print(f"❌ State field contains comma (gaming attempt): '{state}'")
            
            updated_stats = mark_submission_failed(event.actor_hotkey)
            
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "invalid_state_format",
                    "message": "State field should contain only one state (no commas allowed)",
                    "state": state,
                    "rate_limit_stats": {
                        "submissions": updated_stats["submissions"],
                        "max_submissions": MAX_SUBMISSIONS_PER_DAY,
                        "rejections": updated_stats["rejections"],
                        "max_rejections": MAX_REJECTIONS_PER_DAY
                    }
                }
            )
        
        # Update lead_blob with normalized country (in case alias was used)
        lead_blob["country"] = country
        
        state_display = state if state else "(empty)"
        city_display = city if city else "(empty)"
        print(f"   ✅ Region fields validated: country='{country}', state='{state_display}', city='{city_display}'")
        
        # ========================================
        # Validate employee_count format
        # ========================================
        VALID_EMPLOYEE_COUNTS = [
            "0-1", "2-10", "11-50", "51-200", "201-500", 
            "501-1,000", "1,001-5,000", "5,001-10,000", "10,001+"
        ]
        
        employee_count = lead_blob.get("employee_count", "").strip()
        if employee_count not in VALID_EMPLOYEE_COUNTS:
            print(f"❌ Invalid employee_count: '{employee_count}'")
            
            updated_stats = mark_submission_failed(event.actor_hotkey)
            
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "invalid_employee_count",
                    "message": f"employee_count must be one of the valid ranges",
                    "provided": employee_count,
                    "valid_values": VALID_EMPLOYEE_COUNTS,
                    "rate_limit_stats": {
                        "submissions": updated_stats["submissions"],
                        "max_submissions": MAX_SUBMISSIONS_PER_DAY,
                        "rejections": updated_stats["rejections"],
                        "max_rejections": MAX_REJECTIONS_PER_DAY
                    }
                }
            )
        
        print(f"   ✅ employee_count '{employee_count}' is valid")
        
        # ========================================
        # Verify source_type and source_url consistency
        # ========================================
        source_type = lead_blob.get("source_type", "").strip()
        source_url = lead_blob.get("source_url", "").strip()
        
        if source_type == "proprietary_database" and source_url != "proprietary_database":
            print(f"❌ Source provenance mismatch: source_type='proprietary_database' but source_url='{source_url[:50]}...'")
            
            # Mark submission as failed (FAILURE - source provenance mismatch)
            # NOTE: Submission slot was already reserved in Step 2.5, just increment rejections
            updated_stats = mark_submission_failed(event.actor_hotkey)
            
            raise HTTPException(
                status_code=400,
                detail={
                    "message": "Source provenance mismatch: If source_type is 'proprietary_database', source_url must also be 'proprietary_database'",
                    "source_type": source_type,
                    "source_url": source_url,
                    "rate_limit_stats": {
                        "submissions": updated_stats["submissions"],
                        "max_submissions": MAX_SUBMISSIONS_PER_DAY,
                        "rejections": updated_stats["rejections"],
                        "max_rejections": MAX_REJECTIONS_PER_DAY
                    }
                }
            )
        
        # Block LinkedIn URLs in source_url (miners should use source_type="linkedin" instead)
        if "linkedin" in source_url.lower():
            print(f"❌ LinkedIn URL detected in source_url: {source_url[:50]}...")
            
            # Mark submission as failed (FAILURE - LinkedIn URL in source_url)
            # NOTE: Submission slot was already reserved in Step 2.5, just increment rejections
            updated_stats = mark_submission_failed(event.actor_hotkey)
            
            raise HTTPException(
                status_code=400,
                detail={
                    "message": "LinkedIn URLs are not allowed in source_url. Use source_type='linkedin' and source_url='linkedin' instead.",
                    "source_url": source_url,
                    "rate_limit_stats": {
                        "submissions": updated_stats["submissions"],
                        "max_submissions": MAX_SUBMISSIONS_PER_DAY,
                        "rejections": updated_stats["rejections"],
                        "max_rejections": MAX_REJECTIONS_PER_DAY
                    }
                }
            )
        
        print(f"   ✅ Source provenance verified: source_type={source_type}")
        
        # ========================================
        # CRITICAL: Verify Miner Attestation (Trustless Model)
        # ========================================
        # In the trustless model, attestations are stored locally by miners
        # and verified via the lead metadata itself (not database lookup)
        print(f"   🔍 Verifying miner attestation...")
        try:
            wallet_ss58 = lead_blob.get("wallet_ss58")
            terms_version_hash = lead_blob.get("terms_version_hash")
            lawful_collection = lead_blob.get("lawful_collection")
            no_restricted_sources = lead_blob.get("no_restricted_sources")
            license_granted = lead_blob.get("license_granted")
            
            # Check required attestation fields are present
            if not wallet_ss58 or not terms_version_hash:
                print(f"❌ Attestation check failed: Missing wallet_ss58 or terms_version_hash in lead")
                raise HTTPException(
                    status_code=400,
                    detail="Lead missing required attestation fields (wallet_ss58, terms_version_hash)"
                )
            
            # ========================================
            # CRITICAL: Verify terms_version_hash matches current canonical terms
            # ========================================
            # This prevents miners from using outdated or fake terms versions
            from gateway.utils.contributor_terms import get_terms_version_hash
            
            try:
                current_terms_hash = get_terms_version_hash()
            except Exception as e:
                print(f"⚠️  Failed to fetch current terms hash from GitHub: {e}")
                # Don't fail submission if GitHub is temporarily unavailable
                # Gateway should not be a single point of failure
                print(f"   ⚠️  Continuing without hash verification (GitHub unavailable)")
                current_terms_hash = None
            
            if current_terms_hash and terms_version_hash != current_terms_hash:
                print(f"❌ Attestation check failed: Outdated or invalid terms version")
                print(f"   Submitted: {terms_version_hash[:16]}...")
                print(f"   Current:   {current_terms_hash[:16]}...")
                raise HTTPException(
                    status_code=400,
                    detail=f"Outdated or invalid terms version. Your miner is using an old terms version. Please restart your miner to accept the current terms."
                )
            
            # Verify wallet matches actor (prevent impersonation)
            if wallet_ss58 != event.actor_hotkey:
                print(f"❌ Attestation check failed: wallet_ss58 ({wallet_ss58[:20]}...) doesn't match actor_hotkey ({event.actor_hotkey[:20]}...)")
                raise HTTPException(
                    status_code=403,
                    detail="Wallet mismatch: lead wallet_ss58 doesn't match submission actor_hotkey"
                )
            
            # Verify attestation fields have expected values
            if lawful_collection != True:
                print(f"❌ Attestation check failed: lawful_collection must be True")
                raise HTTPException(
                    status_code=400,
                    detail="Attestation failed: lawful_collection must be True"
                )
            
            if no_restricted_sources != True:
                print(f"❌ Attestation check failed: no_restricted_sources must be True")
                raise HTTPException(
                    status_code=400,
                    detail="Attestation failed: no_restricted_sources must be True"
                )
            
            if license_granted != True:
                print(f"❌ Attestation check failed: license_granted must be True")
                raise HTTPException(
                    status_code=400,
                    detail="Attestation failed: license_granted must be True"
                )
            
            print(f"   ✅ Attestation verified for wallet {wallet_ss58[:20]}...")
            print(f"      Terms version: {terms_version_hash[:16]}...")
            print(f"      Lawful: {lawful_collection}, No restricted: {no_restricted_sources}, Licensed: {license_granted}")
            
            # ========================================
            # Store attestation in Supabase (for record-keeping, not verification)
            # ========================================
            # This creates an audit trail but does NOT affect verification (trustless)
            print(f"   📊 Recording attestation to Supabase...")
            try:
                from datetime import timezone as tz
                
                # Check if attestation already exists for this wallet
                existing = supabase.table("contributor_attestations") \
                    .select("id, wallet_ss58") \
                    .eq("wallet_ss58", wallet_ss58) \
                    .execute()
                
                attestation_data = {
                    "wallet_ss58": wallet_ss58,
                    "terms_version_hash": terms_version_hash,
                    "accepted": True,
                    "timestamp_utc": datetime.now(tz.utc).isoformat(),
                    "ip_address": None  # Privacy: Don't store IP in trustless model
                }
                
                # Note: Boolean attestation fields (lawful_collection, no_restricted_sources, license_granted)
                # are stored in the lead metadata, not the attestation table
                
                if existing.data and len(existing.data) > 0:
                    # Update existing record
                    result = supabase.table("contributor_attestations") \
                        .update(attestation_data) \
                        .eq("wallet_ss58", wallet_ss58) \
                        .execute()
                    print(f"   ✅ Attestation updated in database (audit trail)")
                else:
                    # Insert new record
                    result = supabase.table("contributor_attestations") \
                        .insert(attestation_data) \
                        .execute()
                    print(f"   ✅ Attestation inserted in database (audit trail)")
                
            except Exception as e:
                # Don't fail the submission if database write fails
                # Verification already passed (trustless)
                print(f"   ⚠️  Failed to record attestation to database: {e}")
                print(f"      (Submission continues - attestation verification already passed)")
            
        except HTTPException:
            # Mark submission as failed (FAILURE - attestation check)
            # NOTE: Submission slot was already reserved in Step 2.5, just increment rejections
            updated_stats = mark_submission_failed(event.actor_hotkey)
            print(f"   📊 Rate limit updated: submissions={updated_stats['submissions']}/{MAX_SUBMISSIONS_PER_DAY}, rejections={updated_stats['rejections']}/{MAX_REJECTIONS_PER_DAY}")
            
            # Re-raise HTTP exceptions
            raise
        except Exception as e:
            print(f"❌ Attestation verification error: {e}")
            import traceback
            traceback.print_exc()
            raise HTTPException(
                status_code=500,
                detail=f"Attestation verification failed: {str(e)}"
            )
        
        # ========================================
        # Normalize lead fields for standardized storage
        # ========================================
        # Title-case fields like industry, role, full_name, city, etc.
        # This ensures consistent formatting in the database
        # NOTE: Does NOT affect validation (automated_checks uses .lower() for comparisons)
        #
        # HASH INTEGRITY: We preserve the original geo fields in "_original_geo" key
        # so hash(lead_blob without _original_geo) == lead_blob_hash always works.
        # Validators use the normalized top-level fields (city, state, country).
        print(f"   🔍 Normalizing lead fields for standardized storage...")
        
        # Preserve original geo fields BEFORE normalization (for hash verification)
        original_geo = {
            "city": lead_blob.get("city", ""),
            "state": lead_blob.get("state", ""),
            "country": lead_blob.get("country", ""),
        }
        
        # Normalize the lead_blob (modifies city, state, country, etc.)
        lead_blob = normalize_lead_fields(lead_blob)
        
        # Embed original geo fields for hash verification
        # NOTE: This key is ignored by validators - they use top-level normalized fields
        lead_blob["_original_geo"] = original_geo
        
        print(f"   ✅ Lead fields normalized (industry='{lead_blob.get('industry', '')}', role='{lead_blob.get('role', '')[:30]}...')")
        
        # Store lead in leads_private table
        print(f"   🔍 Storing lead in leads_private database...")
        try:
            # Note: salt is NOT stored here - validators generate their own salt
            # for the commit-reveal scheme and store it in validation_evidence_private
            
            # ========================================
            # Handle resubmission of denied leads
            # ========================================
            # If Step 6.5 allowed resubmission (because prior lead was denied),
            # we need to delete the old denied record before inserting new one.
            # The CONSENSUS_RESULT for the denied lead is already in transparency_log (immutable).
            submitted_email = lead_blob.get("email", "").strip().lower()
            if submitted_email:
                try:
                    # Check if there's a denied lead with same email
                    denied_check = supabase.table("leads_private") \
                        .select("lead_id") \
                        .eq("lead_blob->>email", submitted_email) \
                        .eq("status", "denied") \
                        .limit(1) \
                        .execute()
                    
                    if denied_check.data:
                        old_lead_id = denied_check.data[0].get("lead_id")
                        print(f"   🔄 Found denied lead with same email: {old_lead_id[:10]}...")
                        print(f"      Deleting old record to allow resubmission (CONSENSUS_RESULT preserved in transparency_log)")
                        
                        # IMPORTANT: Must delete in correct order due to foreign key constraints
                        # 1. First delete from validation_evidence_private (references leads_private)
                        # 2. Then delete from leads_private
                        
                        # Step 1: Delete validation evidence for the denied lead
                        evidence_delete = supabase.table("validation_evidence_private") \
                            .delete() \
                            .eq("lead_id", old_lead_id) \
                            .execute()
                        
                        evidence_count = len(evidence_delete.data) if evidence_delete.data else 0
                        if evidence_count > 0:
                            print(f"      ✅ Deleted {evidence_count} validation_evidence_private record(s)")
                        
                        # Step 2: Delete the old denied lead from leads_private
                        # Extra safety: re-verify status is 'denied' before deleting
                        supabase.table("leads_private") \
                            .delete() \
                            .eq("lead_id", old_lead_id) \
                            .eq("status", "denied") \
                            .execute()
                        
                        print(f"   ✅ Old denied lead deleted - resubmission can proceed")
                except Exception as cleanup_error:
                    print(f"   ⚠️  Error during denied lead cleanup: {cleanup_error}")
                    # Continue anyway - insert might still succeed if no constraint conflict
            
            lead_private_entry = {
                "lead_id": event.payload.lead_id,
                "lead_blob_hash": committed_lead_blob_hash,
                "miner_hotkey": event.actor_hotkey,  # Extract from signature
                "lead_blob": lead_blob,
                "status": "pending_validation",  # Initial state when entering queue
                "created_ts": datetime.now(tz.utc).isoformat()
            }
            
            supabase.table("leads_private").insert(lead_private_entry).execute()
            print(f"   ✅ Lead stored in leads_private (miner: {event.actor_hotkey[:10]}..., status: pending_validation)")
            
        except Exception as e:
            error_str = str(e).lower()
            
            # Check if this is a duplicate email constraint violation
            if "duplicate" in error_str or "unique" in error_str or "23505" in error_str:
                print(f"❌ Duplicate email detected at database level (race condition caught)!")
                print(f"   Email from lead_blob: {lead_blob.get('email', 'unknown')}")
                print(f"   This could be a lead that's still processing (not yet denied)")
                
                # Mark submission as failed (FAILURE - duplicate at DB level)
                # NOTE: Submission slot was already reserved in Step 2.5, just increment rejections
                updated_stats = mark_submission_failed(event.actor_hotkey)
                print(f"   📊 Rate limit updated: rejections={updated_stats['rejections']}/{MAX_REJECTIONS_PER_DAY}")
                
                raise HTTPException(
                    status_code=409,  # 409 Conflict
                    detail={
                        "error": "duplicate_email",
                        "message": "This email is still being processed or has been approved (race condition)",
                        "email_hash": committed_email_hash,
                        "rate_limit_stats": {
                            "submissions": updated_stats["submissions"],
                            "max_submissions": MAX_SUBMISSIONS_PER_DAY,
                            "rejections": updated_stats["rejections"],
                            "max_rejections": MAX_REJECTIONS_PER_DAY,
                            "reset_at": updated_stats["reset_at"]
                        }
                    }
                )
            
            print(f"❌ Failed to store lead in leads_private: {e}")
            import traceback
            traceback.print_exc()
            raise HTTPException(
                status_code=500,
                detail=f"Failed to store lead: {str(e)}"
            )
        
        # Log SUBMISSION event to Arweave FIRST
        print(f"   🔍 Logging SUBMISSION event to TEE buffer...")
        try:
            submission_payload = {
                "lead_id": event.payload.lead_id,
                "lead_blob_hash": committed_lead_blob_hash,
                "email_hash": committed_email_hash,
                "linkedin_combo_hash": actual_linkedin_combo_hash if actual_linkedin_combo_hash else None,
                "miner_hotkey": event.actor_hotkey,
                "submission_timestamp": datetime.now(tz.utc).isoformat(),
                "s3_proof_tee_seq": storage_proof_tee_seqs.get("s3")
            }
            
            submission_log_entry = {
                "event_type": "SUBMISSION",
                "actor_hotkey": event.actor_hotkey,
                "nonce": str(uuid.uuid4()),  # Generate fresh UUID for this event
                "ts": datetime.now(tz.utc).isoformat(),
                "payload_hash": hashlib.sha256(
                    json.dumps(submission_payload, sort_keys=True).encode()
                ).hexdigest(),
                "build_id": event.build_id,
                "signature": event.signature,
                "payload": submission_payload
            }
            
            result = await log_event(submission_log_entry)
            
            submission_tee_seq = result.get("sequence")
            print(f"   ✅ SUBMISSION event buffered in TEE: seq={submission_tee_seq}")
                
        except Exception as e:
            print(f"❌ Error logging SUBMISSION event: {e}")
            import traceback
            traceback.print_exc()
            # CRITICAL: If TEE write fails, request MUST fail
            print(f"🚨 CRITICAL: TEE buffer unavailable - failing request")
            raise HTTPException(
                status_code=503,
                detail=f"TEE buffer unavailable: {str(e)}"
            )
        
        # Compute queue_position (simplified - just count total submissions)
        submission_timestamp = datetime.now(tz.utc).isoformat()
        try:
            queue_count_result = supabase.table("leads_private").select("lead_id", count="exact").execute()
            queue_position = queue_count_result.count if hasattr(queue_count_result, 'count') else None
        except Exception as e:
            print(f"⚠️  Could not compute queue_position: {e}")
            queue_position = None
        
        # Return success with simple acknowledgment
        # NOTE (Phase 4): TEE-based trust model
        # - Events buffered in TEE (hardware-protected memory)
        # - Will be included in next hourly Arweave checkpoint (signed by TEE)
        # - Verify gateway code integrity: GET /attest
        
        # NOTE: Submission slot was already reserved in Step 2.5 (atomic rate limiting)
        # No need to increment again - just log the current stats from reservation
        print(f"   📊 Rate limit (from reservation): submissions={reservation_stats['submissions']}/{reservation_stats['max_submissions']}, rejections={reservation_stats['rejections']}/{reservation_stats['max_rejections']}")
        
        print(f"✅ /submit complete - lead accepted")
        return {
            "status": "accepted",
            "lead_id": event.payload.lead_id,
            "storage_backends": ["s3"],  # Only S3 storage is used
            "submission_timestamp": submission_timestamp,
            "queue_position": queue_position,
            "message": "Lead accepted. Proof available in next hourly Arweave checkpoint.",
            "rate_limit_stats": {
                "submissions": reservation_stats["submissions"],
                "max_submissions": reservation_stats["max_submissions"],
                "rejections": reservation_stats["rejections"],
                "max_rejections": reservation_stats["max_rejections"],
                "reset_at": reservation_stats["reset_at"]
            }
        }
    
    # ========================================
    # Step 9b: FAILURE PATH - Verification failed
    # ========================================
    else:
        print(f"🔍 Step 9: FAILURE PATH - S3 verification failed")
        
        # Log UPLOAD_FAILED event to Arweave FIRST
        upload_failed_payload = {
            "lead_id": event.payload.lead_id,
            "lead_blob_hash": committed_lead_blob_hash,
            "email_hash": committed_email_hash,
            "miner_hotkey": event.actor_hotkey,
            "failed_mirrors": ["s3"],
            "reason": "Hash mismatch or blob not found in S3",
            "timestamp": datetime.now(tz.utc).isoformat()
        }
        
        upload_failed_log_entry = {
            "event_type": "UPLOAD_FAILED",
            "actor_hotkey": event.actor_hotkey,
            "nonce": str(uuid.uuid4()),  # Generate fresh UUID for this event
            "ts": datetime.now(tz.utc).isoformat(),
            "payload_hash": hashlib.sha256(
                json.dumps(upload_failed_payload, sort_keys=True).encode()
            ).hexdigest(),
            "build_id": event.build_id,
            "signature": event.signature,
            "payload": upload_failed_payload
        }
        
        try:
            from gateway.utils.logger import log_event
            result = await log_event(upload_failed_log_entry)
            
            upload_failed_tee_seq = result.get("sequence")
            print(f"   ❌ UPLOAD_FAILED event buffered in TEE: seq={upload_failed_tee_seq}")
        except Exception as e:
            print(f"   ⚠️  Error logging UPLOAD_FAILED: {e} (continuing with error response)")
        
        # Mark submission as failed (FAILURE - verification failed)
        # NOTE: Submission slot was already reserved in Step 2.5, just increment rejections
        updated_stats = mark_submission_failed(event.actor_hotkey)
        print(f"   📊 Rate limit updated: submissions={updated_stats['submissions']}/{MAX_SUBMISSIONS_PER_DAY}, rejections={updated_stats['rejections']}/{MAX_REJECTIONS_PER_DAY}")
        
        raise HTTPException(
            status_code=400,
            detail={
                "error": "upload_verification_failed",
                "message": f"Upload verification failed for mirrors: {', '.join(failed_mirrors)}",
                "failed_mirrors": failed_mirrors,
                "rate_limit_stats": {
                    "submissions": updated_stats["submissions"],
                    "max_submissions": MAX_SUBMISSIONS_PER_DAY,
                    "rejections": updated_stats["rejections"],
                    "max_rejections": MAX_REJECTIONS_PER_DAY,
                    "reset_at": updated_stats["reset_at"]
                }
            }
        )

