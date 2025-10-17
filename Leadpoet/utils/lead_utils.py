"""
Lead field extraction utilities.

This module provides helper functions to extract fields from lead dictionaries
that may have inconsistent key naming (e.g., "Email 1" vs "email" vs "Owner(s) Email").

Instead of nested .get() calls throughout the codebase, use these standardized extractors.
"""

from typing import Dict, Any, Optional


def get_field(data: Dict[str, Any], *keys: str, default: str = "") -> str:
    """
    Try multiple keys in priority order, return first non-empty value.
    
    This is the base function for all field extraction. It tries each key
    in order and returns the first non-empty value found.
    
    Args:
        data: Dictionary to search (typically a lead dict)
        *keys: Keys to try in priority order
        default: Default value if all keys fail or return empty values
    
    Returns:
        First non-empty value found, or default
    
    Example:
        >>> lead = {"Email 1": "test@example.com", "website": "example.com"}
        >>> get_field(lead, "Email 1", "email", "Owner(s) Email")
        "test@example.com"
        >>> get_field(lead, "missing_key", "also_missing", default="N/A")
        "N/A"
    """
    for key in keys:
        value = data.get(key)
        if value:  # Non-empty check (excludes None, "", [], {}, etc.)
            return str(value).strip()
    return default


def get_email(lead: Dict[str, Any], default: str = "") -> str:
    """
    Extract email from lead with standard key priority.
    
    Tries: "Email 1" → "Owner(s) Email" → "email"
    
    Args:
        lead: Lead dictionary
        default: Default value if no email found
    
    Returns:
        Email address or default
    """
    return get_field(lead, "Email 1", "Owner(s) Email", "email", default=default)


def get_website(lead: Dict[str, Any], default: str = "") -> str:
    """
    Extract website from lead.
    
    Tries: "Website" → "website"
    
    Args:
        lead: Lead dictionary
        default: Default value if no website found
    
    Returns:
        Website URL or default
    """
    return get_field(lead, "Website", "website", default=default)


def get_company(lead: Dict[str, Any], default: str = "") -> str:
    """
    Extract company/business name from lead.
    
    Tries: "Business" → "Company" → "business"
    
    Args:
        lead: Lead dictionary
        default: Default value if no company found
    
    Returns:
        Company name or default
    """
    return get_field(lead, "Business", "Company", "business", default=default)


def get_first_name(lead: Dict[str, Any], default: str = "") -> str:
    """
    Extract first name from lead.
    
    Tries: "First" → "First Name" → "first"
    
    Args:
        lead: Lead dictionary
        default: Default value if no first name found
    
    Returns:
        First name or default
    """
    return get_field(lead, "First", "First Name", "first", default=default)


def get_last_name(lead: Dict[str, Any], default: str = "") -> str:
    """
    Extract last name from lead.
    
    Tries: "Last" → "Last Name" → "last"
    
    Args:
        lead: Lead dictionary
        default: Default value if no last name found
    
    Returns:
        Last name or default
    """
    return get_field(lead, "Last", "Last Name", "last", default=default)


def get_location(lead: Dict[str, Any], default: str = "") -> str:
    """
    Extract location/region from lead.
    
    Tries: "region" → "Region" → "location"
    
    Args:
        lead: Lead dictionary
        default: Default value if no location found
    
    Returns:
        Location or default
    """
    return get_field(lead, "region", "Region", "location", default=default)


def get_industry(lead: Dict[str, Any], default: str = "") -> str:
    """
    Extract industry from lead.
    
    Tries: "Industry" → "industry"
    
    Args:
        lead: Lead dictionary
        default: Default value if no industry found
    
    Returns:
        Industry or default
    """
    return get_field(lead, "Industry", "industry", default=default)


def get_role(lead: Dict[str, Any], default: str = "") -> str:
    """
    Extract role/title from lead.
    
    Tries: "role" → "Role"
    
    Args:
        lead: Lead dictionary
        default: Default value if no role found
    
    Returns:
        Role or default
    """
    return get_field(lead, "role", "Role", default=default)


def get_linkedin(lead: Dict[str, Any], default: str = "") -> str:
    """
    Extract LinkedIn URL from lead.
    
    Tries: "linkedin" → "LinkedIn"
    
    Args:
        lead: Lead dictionary
        default: Default value if no LinkedIn found
    
    Returns:
        LinkedIn URL or default
    """
    return get_field(lead, "linkedin", "LinkedIn", default=default)


def get_sub_industry(lead: Dict[str, Any], default: str = "") -> str:
    """
    Extract sub-industry from lead.
    
    Tries: "sub_industry" → "Sub-industry"
    
    Args:
        lead: Lead dictionary
        default: Default value if no sub-industry found
    
    Returns:
        Sub-industry or default
    """
    return get_field(lead, "sub_industry", "Sub-industry", default=default)


def get_prospect_id(lead: Dict[str, Any], default: str = "") -> str:
    """
    Extract prospect ID from lead.
    
    Tries: "prospect_id" → "id"
    
    Args:
        lead: Lead dictionary
        default: Default value if no ID found
    
    Returns:
        Prospect ID or default
    """
    return get_field(lead, "prospect_id", "id", default=default)


def get_score(lead: Dict[str, Any], default: float = 0.0) -> float:
    """
    Extract score from lead (handles multiple score field names).
    
    Tries: "score" → "intent_score" → "conversion_score"
    
    Args:
        lead: Lead dictionary
        default: Default value if no score found
    
    Returns:
        Score as float or default
    """
    value = get_field(lead, "score", "intent_score", "conversion_score", default=str(default))
    try:
        return float(value)
    except (ValueError, TypeError):
        return default
