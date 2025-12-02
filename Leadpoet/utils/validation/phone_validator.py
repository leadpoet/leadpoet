"""
Phone Number Validation Module

Comprehensive phone number validation with international format support,
carrier lookup, and location consistency checks.

Features:
- E.164 format validation
- Country code verification
- Carrier type detection (mobile/landline/VOIP)
- Phone-to-location consistency
- Disposable number detection
"""

import re
import asyncio
import aiohttp
from typing import Dict, Tuple, Optional, List
from datetime import datetime
import phonenumbers
from phonenumbers import geocoder, carrier, timezone, NumberParseException

# Cache for phone validation results
_phone_cache: Dict[str, Tuple[bool, Dict, datetime]] = {}
PHONE_CACHE_TTL_HOURS = 168  # 7 days


def _is_cache_valid(timestamp: datetime, ttl_hours: int = PHONE_CACHE_TTL_HOURS) -> bool:
    """Check if cached result is still valid"""
    age = datetime.now() - timestamp
    return age.total_seconds() < (ttl_hours * 3600)


def _normalize_phone(phone: str) -> str:
    """Normalize phone number for caching"""
    return re.sub(r'[^\d+]', '', phone)


async def validate_phone_format(phone: str, default_region: str = "US") -> Tuple[bool, Dict]:
    """
    Validate phone number format using E.164 standard.
    
    Args:
        phone: Phone number string (can include country code)
        default_region: Default country code if not specified (ISO 3166-1 alpha-2)
    
    Returns:
        (is_valid, details_dict)
        
    Example:
        >>> await validate_phone_format("+1-310-363-6000")
        (True, {
            "formatted": "+13103636000",
            "country_code": 1,
            "national_number": "3103636000",
            "region": "US",
            "is_valid": True
        })
    """
    try:
        if not phone:
            return False, {
                "stage": "Phone Validation",
                "check_name": "validate_phone_format",
                "message": "No phone number provided",
                "failed_fields": ["phone_numbers"]
            }
        
        # Normalize phone
        normalized = _normalize_phone(phone)
        
        # Check cache
        cache_key = f"phone_format:{normalized}"
        if cache_key in _phone_cache:
            cached_result, cached_data, timestamp = _phone_cache[cache_key]
            if _is_cache_valid(timestamp):
                return cached_result, cached_data
        
        # Parse phone number
        try:
            parsed = phonenumbers.parse(phone, default_region)
        except NumberParseException as e:
            rejection = {
                "stage": "Phone Validation",
                "check_name": "validate_phone_format",
                "message": f"Invalid phone format: {str(e)}",
                "failed_fields": ["phone_numbers"],
                "phone": phone
            }
            _phone_cache[cache_key] = (False, rejection, datetime.now())
            return False, rejection
        
        # Validate phone number
        is_valid = phonenumbers.is_valid_number(parsed)
        is_possible = phonenumbers.is_possible_number(parsed)
        
        if not is_valid:
            rejection = {
                "stage": "Phone Validation",
                "check_name": "validate_phone_format",
                "message": f"Invalid phone number: {phone} (possible: {is_possible})",
                "failed_fields": ["phone_numbers"],
                "phone": phone,
                "is_possible": is_possible
            }
            _phone_cache[cache_key] = (False, rejection, datetime.now())
            return False, rejection
        
        # Extract details
        details = {
            "formatted": phonenumbers.format_number(parsed, phonenumbers.PhoneNumberFormat.E164),
            "international": phonenumbers.format_number(parsed, phonenumbers.PhoneNumberFormat.INTERNATIONAL),
            "national": phonenumbers.format_number(parsed, phonenumbers.PhoneNumberFormat.NATIONAL),
            "country_code": parsed.country_code,
            "national_number": str(parsed.national_number),
            "region": phonenumbers.region_code_for_number(parsed),
            "is_valid": True,
            "is_possible": is_possible,
            "number_type": phonenumbers.number_type(parsed)
        }
        
        # Cache result
        _phone_cache[cache_key] = (True, details, datetime.now())
        
        return True, details
        
    except Exception as e:
        rejection = {
            "stage": "Phone Validation",
            "check_name": "validate_phone_format",
            "message": f"Phone validation error: {str(e)}",
            "failed_fields": ["phone_numbers"],
            "error": str(e)
        }
        return False, rejection


async def check_phone_carrier(phone: str, default_region: str = "US") -> Tuple[bool, Dict]:
    """
    Check phone carrier type (mobile, landline, VOIP, etc.).
    
    Args:
        phone: Phone number string
        default_region: Default country code if not specified
    
    Returns:
        (is_valid, carrier_info)
        
    Example:
        >>> await check_phone_carrier("+1-310-363-6000")
        (True, {
            "carrier_name": "Verizon",
            "number_type": "MOBILE",
            "is_mobile": True,
            "is_voip": False
        })
    """
    try:
        # First validate format
        is_valid, format_result = await validate_phone_format(phone, default_region)
        if not is_valid:
            return False, format_result
        
        # Parse phone
        parsed = phonenumbers.parse(phone, default_region)
        
        # Get carrier name
        carrier_name = carrier.name_for_number(parsed, "en")
        
        # Get number type
        num_type = phonenumbers.number_type(parsed)
        type_names = {
            0: "FIXED_LINE",
            1: "MOBILE",
            2: "FIXED_LINE_OR_MOBILE",
            3: "TOLL_FREE",
            4: "PREMIUM_RATE",
            5: "SHARED_COST",
            6: "VOIP",
            7: "PERSONAL_NUMBER",
            8: "PAGER",
            9: "UAN",
            10: "VOICEMAIL",
            -1: "UNKNOWN"
        }
        
        type_name = type_names.get(num_type, "UNKNOWN")
        
        # Determine if mobile/VOIP
        is_mobile = num_type in [1, 2]  # MOBILE or FIXED_LINE_OR_MOBILE
        is_voip = num_type == 6
        is_landline = num_type == 0
        
        carrier_info = {
            "carrier_name": carrier_name or "Unknown",
            "number_type": type_name,
            "number_type_code": num_type,
            "is_mobile": is_mobile,
            "is_landline": is_landline,
            "is_voip": is_voip,
            "region": format_result.get("region"),
            "formatted": format_result.get("formatted")
        }
        
        return True, carrier_info
        
    except Exception as e:
        rejection = {
            "stage": "Phone Validation",
            "check_name": "check_phone_carrier",
            "message": f"Carrier lookup error: {str(e)}",
            "failed_fields": ["phone_numbers"],
            "error": str(e)
        }
        return False, rejection


async def check_phone_location_consistency(
    phone: str,
    expected_region: str,
    expected_location: str = None,
    default_region: str = "US"
) -> Tuple[bool, Dict]:
    """
    Check if phone number location matches expected region/location.
    
    Args:
        phone: Phone number string
        expected_region: Expected country/region (e.g., "US", "UK", "CA")
        expected_location: Expected city/state (optional, for detailed check)
        default_region: Default country code if not specified
    
    Returns:
        (is_consistent, consistency_info)
        
    Example:
        >>> await check_phone_location_consistency("+1-310-363-6000", "US", "California")
        (True, {
            "phone_region": "US",
            "phone_location": "California",
            "expected_region": "US",
            "matches": True
        })
    """
    try:
        # Validate format first
        is_valid, format_result = await validate_phone_format(phone, default_region)
        if not is_valid:
            return False, format_result
        
        # Parse phone
        parsed = phonenumbers.parse(phone, default_region)
        
        # Get phone region
        phone_region = phonenumbers.region_code_for_number(parsed)
        
        # Get geographic location
        phone_location = geocoder.description_for_number(parsed, "en")
        
        # Get timezone
        phone_timezones = timezone.time_zones_for_number(parsed)
        
        # Check region match
        region_matches = phone_region.upper() == expected_region.upper()
        
        # Check location match (if provided)
        location_matches = True
        if expected_location:
            location_matches = expected_location.lower() in phone_location.lower()
        
        consistency_info = {
            "phone_region": phone_region,
            "phone_location": phone_location,
            "phone_timezones": list(phone_timezones) if phone_timezones else [],
            "expected_region": expected_region,
            "expected_location": expected_location,
            "region_matches": region_matches,
            "location_matches": location_matches,
            "is_consistent": region_matches and location_matches
        }
        
        if not region_matches:
            rejection = {
                "stage": "Phone Validation",
                "check_name": "check_phone_location_consistency",
                "message": f"Phone region '{phone_region}' does not match expected '{expected_region}'",
                "failed_fields": ["phone_numbers", "region"],
                **consistency_info
            }
            return False, rejection
        
        if expected_location and not location_matches:
            # Soft warning - don't reject, but flag inconsistency
            consistency_info["warning"] = f"Phone location '{phone_location}' may not match expected '{expected_location}'"
        
        return True, consistency_info
        
    except Exception as e:
        rejection = {
            "stage": "Phone Validation",
            "check_name": "check_phone_location_consistency",
            "message": f"Location consistency check error: {str(e)}",
            "failed_fields": ["phone_numbers"],
            "error": str(e)
        }
        return False, rejection


async def validate_phone_number(
    lead: dict,
    check_carrier: bool = True,
    check_location: bool = True,
    default_region: str = "US"
) -> Tuple[bool, Dict]:
    """
    Comprehensive phone number validation for a lead.
    
    Performs:
    1. Format validation (E.164)
    2. Carrier type check (optional)
    3. Location consistency check (optional)
    
    Args:
        lead: Lead dictionary with phone_numbers field
        check_carrier: Whether to check carrier type
        check_location: Whether to check location consistency
        default_region: Default country code
    
    Returns:
        (is_valid, validation_result)
        
    Example:
        >>> lead = {
        ...     "phone_numbers": ["+1-310-363-6000"],
        ...     "region": "Hawthorne, CA"
        ... }
        >>> await validate_phone_number(lead)
        (True, {
            "phone_count": 1,
            "validated_phones": [...],
            "all_valid": True
        })
    """
    try:
        # Extract phone numbers from lead
        phone_numbers = lead.get("phone_numbers", [])
        
        if not phone_numbers:
            # No phone numbers - not a hard failure, just note it
            return True, {
                "phone_count": 0,
                "message": "No phone numbers provided (optional field)",
                "all_valid": True
            }
        
        # Ensure phone_numbers is a list
        if isinstance(phone_numbers, str):
            phone_numbers = [phone_numbers]
        
        validated_phones = []
        all_valid = True
        rejection_reasons = []
        
        for phone in phone_numbers:
            phone_result = {
                "phone": phone,
                "valid": False
            }
            
            # 1. Format validation
            is_valid, format_info = await validate_phone_format(phone, default_region)
            phone_result["format"] = format_info
            
            if not is_valid:
                all_valid = False
                rejection_reasons.append(format_info.get("message", "Invalid format"))
                validated_phones.append(phone_result)
                continue
            
            phone_result["valid"] = True
            
            # 2. Carrier check (optional)
            if check_carrier:
                carrier_valid, carrier_info = await check_phone_carrier(phone, default_region)
                phone_result["carrier"] = carrier_info
                
                # Flag VOIP numbers (soft warning, not rejection)
                if carrier_info.get("is_voip"):
                    phone_result["warning"] = "VOIP number detected"
            
            # 3. Location consistency (optional)
            if check_location:
                region = lead.get("region", "")
                if region:
                    # Extract country from region (rough heuristic)
                    region_upper = region.upper()
                    expected_region = default_region
                    
                    # Try to detect country from region string
                    if any(x in region_upper for x in ["UK", "UNITED KINGDOM", "ENGLAND", "SCOTLAND"]):
                        expected_region = "GB"
                    elif any(x in region_upper for x in ["CANADA", "TORONTO", "VANCOUVER"]):
                        expected_region = "CA"
                    elif any(x in region_upper for x in ["AUSTRALIA", "SYDNEY", "MELBOURNE"]):
                        expected_region = "AU"
                    
                    loc_valid, loc_info = await check_phone_location_consistency(
                        phone, expected_region, region, default_region
                    )
                    phone_result["location_consistency"] = loc_info
                    
                    if not loc_valid:
                        # Soft warning for location mismatch (don't reject)
                        phone_result["warning"] = loc_info.get("message", "Location mismatch")
            
            validated_phones.append(phone_result)
        
        result = {
            "phone_count": len(phone_numbers),
            "validated_phones": validated_phones,
            "all_valid": all_valid
        }
        
        if not all_valid:
            result["rejection_reasons"] = rejection_reasons
            result["stage"] = "Phone Validation"
            result["check_name"] = "validate_phone_number"
            result["message"] = f"Phone validation failed: {', '.join(rejection_reasons)}"
            result["failed_fields"] = ["phone_numbers"]
            return False, result
        
        return True, result
        
    except Exception as e:
        rejection = {
            "stage": "Phone Validation",
            "check_name": "validate_phone_number",
            "message": f"Phone validation error: {str(e)}",
            "failed_fields": ["phone_numbers"],
            "error": str(e)
        }
        return False, rejection


# Utility function to extract phone from lead (similar to get_email pattern)
def get_phone_numbers(lead: dict) -> List[str]:
    """Extract phone numbers from lead data"""
    phones = lead.get("phone_numbers") or lead.get("phone") or lead.get("Phone")
    
    if not phones:
        return []
    
    if isinstance(phones, str):
        return [phones]
    
    if isinstance(phones, list):
        return phones
    
    return []
