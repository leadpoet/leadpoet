"""
Leadpoet Validation Utilities

Comprehensive validation modules for lead quality assurance.
"""

from .phone_validator import (
    validate_phone_number,
    validate_phone_format,
    check_phone_carrier,
    check_phone_location_consistency
)

from .social_media_validator import (
    validate_twitter_profile,
    validate_linkedin_profile,
    validate_facebook_page,
    validate_social_consistency
)

from .company_size_validator import (
    validate_employee_count,
    validate_company_size_tier,
    check_size_consistency,
    validate_growth_rate
)

from .fraud_detector import (
    detect_duplicate_patterns,
    detect_suspicious_patterns,
    analyze_lead_fingerprint,
    check_temporal_consistency,
    get_fraud_statistics,
    clear_fraud_cache
)

__all__ = [
    # Phone validation
    'validate_phone_number',
    'validate_phone_format',
    'check_phone_carrier',
    'check_phone_location_consistency',
    
    # Social media validation
    'validate_twitter_profile',
    'validate_linkedin_profile',
    'validate_facebook_page',
    'validate_social_consistency',
    
    # Company size validation
    'validate_employee_count',
    'validate_company_size_tier',
    'check_size_consistency',
    'validate_growth_rate',
    
    # Fraud detection
    'detect_duplicate_patterns',
    'detect_suspicious_patterns',
    'analyze_lead_fingerprint',
    'check_temporal_consistency',
    'get_fraud_statistics',
    'clear_fraud_cache',
]
