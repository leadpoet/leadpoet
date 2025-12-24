"""
Lead field extraction utilities.

This module provides helper functions to extract fields from lead dictionaries
that may have inconsistent key naming (e.g., "Email 1" vs "email").

Instead of nested .get() calls throughout the codebase, use these standardized extractors.
"""

from typing import Dict, Any


def get_field(data: Dict[str, Any], *keys: str, default: Any = ""):
    """
    Try multiple keys in priority order, return first found value.
    
    This EXACTLY replicates nested dict.get() behavior:
    - Returns the EXACT value from the first key that exists (no transformations!)
    - Only tries the next key if the current key doesn't exist in the dict
    - Returns default only if none of the keys exist
    
    IMPORTANT: This returns values UNCHANGED (preserves type, whitespace, etc.)
    Just like: data.get(key1, data.get(key2, data.get(key3, default)))
    
    Args:
        data: Dictionary to search (typically a lead dict)
        *keys: Keys to try in priority order
        default: Default value if all keys are missing (can be any type)
    
    Returns:
        Exact value from first existing key, or default if none exist
    
    Examples:
        >>> lead = {"Email 1": "test@example.com"}
        >>> get_field(lead, "Email 1", "email")
        "test@example.com"
        
        >>> lead = {"Email 1": ""}  # Empty string
        >>> get_field(lead, "Email 1", "email")
        ""  # Returns empty string, does NOT try "email"
        
        >>> lead = {"Email 1": "  spaces  "}  # With whitespace
        >>> get_field(lead, "Email 1", "email")
        "  spaces  "  # Preserves whitespace!
        
        >>> lead = {"score": 0.95}  # Float value
        >>> get_field(lead, "score", "backup_score")
        0.95  # Returns as float, NOT string!
    """
    for key in keys:
        if key in data:
            return data[key]
    return default


def get_email(lead: Dict[str, Any], default: str = "") -> str:
    """
    Extract email from lead with standard key priority.
    
    Tries: "email" → "Email 1"
    
    Args:
        lead: Lead dictionary
        default: Default value if no email found
    
    Returns:
        Email address or default
    """
    return get_field(lead, "email", "Email 1", default=default)


def get_full_name(lead: Dict[str, Any], default: str = "") -> str:
    """
    Extract full name from lead with standard key priority.
    
    Tries: "full_name"
    
    Args:
        lead: Lead dictionary
        default: Default value if no name found
    
    Returns:
        Full name or default
    """
    return get_field(lead, "full_name", default=default)


def get_website(lead: Dict[str, Any], default: str = "") -> str:
    """
    Extract website from lead.
    
    Tries: "website" → "Website"
    
    Args:
        lead: Lead dictionary
        default: Default value if no website found
    
    Returns:
        Website URL or default
    """
    return get_field(lead, "website", "Website", default=default)


def get_company(lead: Dict[str, Any], default: str = "") -> str:
    """
    Extract company/business name from lead.
    
    Tries: "business" → "Business" → "Company"
    
    Args:
        lead: Lead dictionary
        default: Default value if no company found
    
    Returns:
        Company name or default
    """
    return get_field(lead, "business", "Business", "Company", default=default)


def get_first_name(lead: Dict[str, Any], default: str = "") -> str:
    """
    Extract first name from lead.
    
    Tries: "first" → "First" → "First Name"
    
    Args:
        lead: Lead dictionary
        default: Default value if no first name found
    
    Returns:
        First name or default
    """
    return get_field(lead, "first", "First", "First Name", default=default)


def get_last_name(lead: Dict[str, Any], default: str = "") -> str:
    """
    Extract last name from lead.
    
    Tries: "last" → "Last" → "Last Name"
    
    Args:
        lead: Lead dictionary
        default: Default value if no last name found
    
    Returns:
        Last name or default
    """
    return get_field(lead, "last", "Last", "Last Name", default=default)


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
    
    Tries: "industry" → "Industry"
    
    Args:
        lead: Lead dictionary
        default: Default value if no industry found
    
    Returns:
        Industry or default
    """
    return get_field(lead, "industry", "Industry", default=default)


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


def get_employee_count(lead: Dict[str, Any], default: str = "") -> str:
    """
    Extract employee count from lead.
    
    Tries: "employee_count" → "Employee Count" → "company_size" → "headcount"
    
    Args:
        lead: Lead dictionary
        default: Default value if no employee count found
    
    Returns:
        Employee count/company size string or default
    """
    return get_field(lead, "employee_count", "Employee Count", "company_size", "headcount", default=default)


def get_description(lead: Dict[str, Any], default: str = "") -> str:
    """
    Extract company description from lead.
    
    Tries: "description" → "Description" → "company_description"
    
    Args:
        lead: Lead dictionary
        default: Default value if no description found
    
    Returns:
        Company description or default
    """
    return get_field(lead, "description", "Description", "company_description", default=default)


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
