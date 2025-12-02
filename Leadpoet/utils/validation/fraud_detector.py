"""
Advanced Fraud Detection Module

Comprehensive fraud and anomaly detection for lead validation.

Features:
- Duplicate lead fingerprinting
- Suspicious pattern detection
- Cross-lead relationship analysis
- Temporal consistency checks
- Data quality anomaly detection
- Gaming pattern identification
"""

import re
import hashlib
import asyncio
from typing import Dict, Tuple, Optional, List, Set
from datetime import datetime, timedelta
from collections import defaultdict
import json

# Global fraud detection state
_lead_fingerprints: Dict[str, Dict] = {}  # lead_hash -> lead_info
_submission_history: List[Dict] = []  # Recent submissions for pattern analysis
_suspicious_patterns: Dict[str, int] = defaultdict(int)  # pattern -> count

# Configuration
MAX_HISTORY_SIZE = 1000
FINGERPRINT_TTL_HOURS = 168  # 7 days
SIMILARITY_THRESHOLD = 0.85  # 85% similarity triggers duplicate warning


def _compute_lead_fingerprint(lead: dict) -> str:
    """
    Compute unique fingerprint for a lead based on key identifying fields.
    
    Args:
        lead: Lead dictionary
    
    Returns:
        SHA256 hash of normalized lead data
    """
    # Extract key fields for fingerprinting
    email = (lead.get("email") or "").lower().strip()
    business = (lead.get("business") or "").lower().strip()
    website = (lead.get("website") or "").lower().strip()
    
    # Normalize website (remove protocol, www, trailing slash)
    if website:
        website = re.sub(r'^https?://', '', website)
        website = re.sub(r'^www\.', '', website)
        website = website.rstrip('/')
    
    # Create fingerprint string
    fingerprint_data = f"{email}|{business}|{website}"
    
    # Hash it
    return hashlib.sha256(fingerprint_data.encode()).hexdigest()


def _compute_similarity(lead1: dict, lead2: dict) -> float:
    """
    Compute similarity score between two leads (0.0 to 1.0).
    
    Args:
        lead1: First lead
        lead2: Second lead
    
    Returns:
        Similarity score (0.0 = completely different, 1.0 = identical)
    """
    # Fields to compare
    fields = [
        "email", "business", "website", "full_name",
        "first", "last", "role", "industry", "region"
    ]
    
    matches = 0
    total_fields = 0
    
    for field in fields:
        val1 = (lead1.get(field) or "").lower().strip()
        val2 = (lead2.get(field) or "").lower().strip()
        
        if val1 or val2:  # At least one has value
            total_fields += 1
            if val1 == val2:
                matches += 1
    
    if total_fields == 0:
        return 0.0
    
    return matches / total_fields


def _is_suspicious_email_pattern(email: str) -> Tuple[bool, Optional[str]]:
    """
    Detect suspicious email patterns that may indicate fraud.
    
    Args:
        email: Email address
    
    Returns:
        (is_suspicious, reason)
    """
    if not email:
        return False, None
    
    email_lower = email.lower()
    
    # Pattern 1: Sequential numbers (test123@, test456@)
    if re.search(r'(test|demo|sample)\d{2,}@', email_lower):
        return True, "Sequential test email pattern"
    
    # Pattern 2: Random character sequences
    if re.search(r'[a-z]{10,}@', email_lower):
        # Check if it's just random letters (low entropy)
        local_part = email_lower.split('@')[0]
        unique_chars = len(set(local_part))
        if unique_chars < len(local_part) * 0.5:  # Less than 50% unique chars
            return True, "Low entropy email (possible random generation)"
    
    # Pattern 3: Excessive numbers
    local_part = email_lower.split('@')[0] if '@' in email_lower else email_lower
    digit_ratio = sum(c.isdigit() for c in local_part) / max(len(local_part), 1)
    if digit_ratio > 0.5:  # More than 50% digits
        return True, "Excessive numbers in email"
    
    # Pattern 4: Repeated characters
    if re.search(r'(.)\1{3,}', email_lower):  # Same char 4+ times
        return True, "Repeated characters in email"
    
    return False, None


def _is_suspicious_name_pattern(first_name: str, last_name: str) -> Tuple[bool, Optional[str]]:
    """
    Detect suspicious name patterns.
    
    Args:
        first_name: First name
        last_name: Last name
    
    Returns:
        (is_suspicious, reason)
    """
    if not first_name or not last_name:
        return False, None
    
    first_lower = first_name.lower()
    last_lower = last_name.lower()
    
    # Pattern 1: Test names
    test_names = ['test', 'demo', 'sample', 'example', 'fake', 'dummy']
    if any(name in first_lower or name in last_lower for name in test_names):
        return True, "Test/demo name detected"
    
    # Pattern 2: Single character names
    if len(first_name) == 1 or len(last_name) == 1:
        return True, "Single character name"
    
    # Pattern 3: Identical first and last name
    if first_lower == last_lower:
        return True, "Identical first and last name"
    
    # Pattern 4: Numbers in names
    if re.search(r'\d', first_name) or re.search(r'\d', last_name):
        return True, "Numbers in name"
    
    return False, None


def _is_suspicious_company_pattern(business: str, website: str) -> Tuple[bool, Optional[str]]:
    """
    Detect suspicious company patterns.
    
    Args:
        business: Company name
        website: Company website
    
    Returns:
        (is_suspicious, reason)
    """
    if not business:
        return False, None
    
    business_lower = business.lower()
    
    # Pattern 1: Test companies
    test_companies = ['test', 'demo', 'sample', 'example', 'acme', 'fake', 'dummy']
    if any(name in business_lower for name in test_companies):
        return True, "Test/demo company name"
    
    # Pattern 2: Generic names
    generic_names = ['company', 'corporation', 'business', 'enterprise', 'inc', 'llc']
    words = business_lower.split()
    if len(words) == 1 and words[0] in generic_names:
        return True, "Generic company name"
    
    # Pattern 3: Website mismatch
    if website:
        website_normalized = re.sub(r'^https?://', '', website.lower())
        website_normalized = re.sub(r'^www\.', '', website_normalized)
        website_domain = website_normalized.split('/')[0].split('.')[0]
        
        business_normalized = re.sub(r'[^a-z0-9]', '', business_lower)
        
        # Check if business name appears in domain
        if len(business_normalized) >= 4:
            if business_normalized not in website_domain and website_domain not in business_normalized:
                # Soft warning, not hard rejection
                return False, f"Company name '{business}' may not match website domain '{website_domain}'"
    
    return False, None


async def detect_duplicate_patterns(
    lead: dict,
    check_history: bool = True,
    similarity_threshold: float = SIMILARITY_THRESHOLD
) -> Tuple[bool, Dict]:
    """
    Detect if lead is a duplicate or near-duplicate of existing leads.
    
    Args:
        lead: Lead dictionary
        check_history: Whether to check against submission history
        similarity_threshold: Minimum similarity to flag as duplicate
    
    Returns:
        (is_unique, duplicate_info)
        
    Example:
        >>> await detect_duplicate_patterns(lead)
        (True, {
            "is_unique": True,
            "fingerprint": "abc123...",
            "similar_leads": []
        })
    """
    try:
        # Compute fingerprint
        fingerprint = _compute_lead_fingerprint(lead)
        
        # Check for exact duplicate
        if fingerprint in _lead_fingerprints:
            existing = _lead_fingerprints[fingerprint]
            
            # Check if fingerprint is still valid (TTL)
            age = datetime.now() - existing["timestamp"]
            if age.total_seconds() < (FINGERPRINT_TTL_HOURS * 3600):
                return False, {
                    "stage": "Fraud Detection",
                    "check_name": "detect_duplicate_patterns",
                    "message": f"Duplicate lead detected (exact match)",
                    "failed_fields": ["email", "business", "website"],
                    "is_unique": False,
                    "fingerprint": fingerprint,
                    "duplicate_type": "exact",
                    "first_seen": existing["timestamp"].isoformat()
                }
        
        # Check for near-duplicates in history
        similar_leads = []
        if check_history:
            for historical_lead in _submission_history[-100:]:  # Check last 100
                similarity = _compute_similarity(lead, historical_lead)
                if similarity >= similarity_threshold:
                    similar_leads.append({
                        "similarity": round(similarity, 2),
                        "email": historical_lead.get("email"),
                        "business": historical_lead.get("business"),
                        "timestamp": historical_lead.get("_timestamp", "unknown")
                    })
        
        # Store fingerprint
        _lead_fingerprints[fingerprint] = {
            "lead": lead,
            "timestamp": datetime.now()
        }
        
        # Add to history
        lead_copy = lead.copy()
        lead_copy["_timestamp"] = datetime.now().isoformat()
        _submission_history.append(lead_copy)
        
        # Trim history if too large
        if len(_submission_history) > MAX_HISTORY_SIZE:
            _submission_history.pop(0)
        
        # Clean old fingerprints
        cutoff = datetime.now() - timedelta(hours=FINGERPRINT_TTL_HOURS)
        expired = [fp for fp, data in _lead_fingerprints.items() 
                   if data["timestamp"] < cutoff]
        for fp in expired:
            del _lead_fingerprints[fp]
        
        result = {
            "is_unique": len(similar_leads) == 0,
            "fingerprint": fingerprint,
            "similar_leads": similar_leads
        }
        
        if similar_leads:
            result["warning"] = f"Found {len(similar_leads)} similar leads (similarity >= {similarity_threshold})"
        
        return True, result
        
    except Exception as e:
        rejection = {
            "stage": "Fraud Detection",
            "check_name": "detect_duplicate_patterns",
            "message": f"Duplicate detection error: {str(e)}",
            "error": str(e)
        }
        return False, rejection


async def detect_suspicious_patterns(lead: dict) -> Tuple[bool, Dict]:
    """
    Detect suspicious patterns in lead data that may indicate fraud or gaming.
    
    Checks:
    - Suspicious email patterns
    - Suspicious name patterns
    - Suspicious company patterns
    - Data quality anomalies
    
    Args:
        lead: Lead dictionary
    
    Returns:
        (is_clean, pattern_info)
    """
    try:
        warnings = []
        red_flags = []
        
        # Extract fields
        email = lead.get("email") or ""
        first_name = lead.get("first") or lead.get("first_name") or ""
        last_name = lead.get("last") or lead.get("last_name") or ""
        business = lead.get("business") or ""
        website = lead.get("website") or ""
        
        # Check 1: Email patterns
        is_suspicious, reason = _is_suspicious_email_pattern(email)
        if is_suspicious:
            red_flags.append(f"Email: {reason}")
            _suspicious_patterns[f"email:{reason}"] += 1
        
        # Check 2: Name patterns
        is_suspicious, reason = _is_suspicious_name_pattern(first_name, last_name)
        if is_suspicious:
            red_flags.append(f"Name: {reason}")
            _suspicious_patterns[f"name:{reason}"] += 1
        
        # Check 3: Company patterns
        is_suspicious, reason = _is_suspicious_company_pattern(business, website)
        if is_suspicious:
            warnings.append(f"Company: {reason}")
            _suspicious_patterns[f"company:{reason}"] += 1
        
        # Check 4: Data completeness anomalies
        required_fields = ["email", "business", "website", "first", "last", "role", "industry", "region"]
        missing_fields = [f for f in required_fields if not lead.get(f)]
        
        if len(missing_fields) > 3:  # More than 3 missing fields
            warnings.append(f"Low data completeness: {len(missing_fields)} missing fields")
        
        # Check 5: Unusual field values
        if email and len(email) > 100:
            warnings.append("Unusually long email address")
        
        if business and len(business) > 200:
            warnings.append("Unusually long business name")
        
        pattern_info = {
            "red_flags": red_flags,
            "warnings": warnings,
            "is_clean": len(red_flags) == 0,
            "suspicious_score": len(red_flags) + len(warnings) * 0.5
        }
        
        # Hard rejection if multiple red flags
        if len(red_flags) >= 2:
            pattern_info["stage"] = "Fraud Detection"
            pattern_info["check_name"] = "detect_suspicious_patterns"
            pattern_info["message"] = f"Multiple suspicious patterns detected: {', '.join(red_flags)}"
            pattern_info["failed_fields"] = ["email", "first", "last", "business"]
            return False, pattern_info
        
        # Soft warning for single red flag or multiple warnings
        if len(red_flags) == 1:
            pattern_info["warning"] = f"Suspicious pattern detected: {red_flags[0]}"
        
        return True, pattern_info
        
    except Exception as e:
        rejection = {
            "stage": "Fraud Detection",
            "check_name": "detect_suspicious_patterns",
            "message": f"Pattern detection error: {str(e)}",
            "error": str(e)
        }
        return False, rejection


async def analyze_lead_fingerprint(
    lead: dict,
    miner_hotkey: str = None
) -> Tuple[bool, Dict]:
    """
    Analyze lead fingerprint for quality and uniqueness.
    
    Args:
        lead: Lead dictionary
        miner_hotkey: Miner's hotkey (for tracking)
    
    Returns:
        (is_valid, fingerprint_analysis)
    """
    try:
        # Compute fingerprint
        fingerprint = _compute_lead_fingerprint(lead)
        
        # Analyze field entropy (data quality)
        email = lead.get("email") or ""
        business = lead.get("business") or ""
        
        def calculate_entropy(text: str) -> float:
            """Calculate Shannon entropy of text"""
            if not text:
                return 0.0
            
            from collections import Counter
            import math
            
            counts = Counter(text.lower())
            total = len(text)
            
            entropy = 0.0
            for count in counts.values():
                p = count / total
                entropy -= p * math.log2(p)
            
            return entropy
        
        email_entropy = calculate_entropy(email)
        business_entropy = calculate_entropy(business)
        
        # High entropy = more random/unique (good)
        # Low entropy = repetitive/suspicious (bad)
        
        analysis = {
            "fingerprint": fingerprint,
            "email_entropy": round(email_entropy, 2),
            "business_entropy": round(business_entropy, 2),
            "miner_hotkey": miner_hotkey,
            "timestamp": datetime.now().isoformat()
        }
        
        # Flag low entropy
        if email_entropy < 2.0:  # Very low entropy
            analysis["warning"] = f"Low email entropy ({email_entropy:.2f}) - may indicate generated data"
        
        if business_entropy < 2.0:
            analysis["warning"] = f"Low business name entropy ({business_entropy:.2f})"
        
        return True, analysis
        
    except Exception as e:
        rejection = {
            "stage": "Fraud Detection",
            "check_name": "analyze_lead_fingerprint",
            "message": f"Fingerprint analysis error: {str(e)}",
            "error": str(e)
        }
        return False, rejection


async def check_temporal_consistency(
    lead: dict,
    submission_timestamp: datetime = None
) -> Tuple[bool, Dict]:
    """
    Check temporal consistency of lead data.
    
    Validates:
    - Submission timing patterns
    - Data freshness
    - Temporal anomalies
    
    Args:
        lead: Lead dictionary
        submission_timestamp: When lead was submitted
    
    Returns:
        (is_consistent, temporal_info)
    """
    try:
        if not submission_timestamp:
            submission_timestamp = datetime.now()
        
        # Check submission timing
        hour = submission_timestamp.hour
        day_of_week = submission_timestamp.weekday()
        
        # Flag submissions at unusual times (potential bot activity)
        is_night = hour < 5 or hour > 23  # 11 PM - 5 AM
        is_weekend = day_of_week >= 5  # Saturday, Sunday
        
        temporal_info = {
            "submission_time": submission_timestamp.isoformat(),
            "hour": hour,
            "day_of_week": day_of_week,
            "is_night": is_night,
            "is_weekend": is_weekend
        }
        
        # Check for rapid submissions from same source
        # (This would require miner tracking - placeholder for now)
        
        # Soft warnings for unusual timing
        if is_night and is_weekend:
            temporal_info["warning"] = "Submission at unusual time (night + weekend) - potential bot activity"
        
        return True, temporal_info
        
    except Exception as e:
        rejection = {
            "stage": "Fraud Detection",
            "check_name": "check_temporal_consistency",
            "message": f"Temporal consistency check error: {str(e)}",
            "error": str(e)
        }
        return False, rejection


# Utility functions
def get_fraud_statistics() -> Dict:
    """Get fraud detection statistics"""
    return {
        "total_fingerprints": len(_lead_fingerprints),
        "history_size": len(_submission_history),
        "suspicious_patterns": dict(_suspicious_patterns),
        "most_common_patterns": sorted(
            _suspicious_patterns.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]
    }


def clear_fraud_cache():
    """Clear fraud detection cache (for testing)"""
    global _lead_fingerprints, _submission_history, _suspicious_patterns
    _lead_fingerprints.clear()
    _submission_history.clear()
    _suspicious_patterns.clear()
