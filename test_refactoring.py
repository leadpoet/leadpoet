#!/usr/bin/env python3
"""
Test script to verify the lead field extraction refactoring works correctly.
"""

def test_imports():
    """Test that all refactored modules can be imported."""
    print("Testing imports...")
    
    try:
        from Leadpoet.utils.lead_utils import (
            get_field,
            get_email,
            get_website,
            get_company,
            get_first_name,
            get_last_name,
            get_location,
            get_industry,
            get_role,
            get_linkedin,
            get_sub_industry
        )
        print("✅ lead_utils imports successful")
    except Exception as e:
        print(f"❌ lead_utils import failed: {e}")
        return False
    
    try:
        from validator_models import automated_checks
        print("✅ automated_checks imports successful")
    except Exception as e:
        print(f"❌ automated_checks import failed: {e}")
        return False
    
    try:
        from Leadpoet.validator import consensus
        print("✅ consensus imports successful")
    except Exception as e:
        print(f"❌ consensus import failed: {e}")
        return False
    
    try:
        from Leadpoet.utils import cloud_db
        print("✅ cloud_db imports successful")
    except Exception as e:
        print(f"❌ cloud_db import failed: {e}")
        return False
    
    return True

def test_field_extractors():
    """Test that field extractors work correctly."""
    print("\nTesting field extractors...")
    
    from Leadpoet.utils.lead_utils import (
        get_email,
        get_website,
        get_company,
        get_first_name,
        get_last_name,
        get_field
    )
    
    # Test lead with old-style keys
    lead1 = {
        "Email 1": "test@example.com",
        "Website": "https://example.com",
        "Business": "Example Corp",
        "First": "John",
        "Last": "Doe"
    }
    
    # Test lead with new-style keys
    lead2 = {
        "email": "test2@example.com",
        "website": "https://example2.com",
        "business": "Example Corp 2",
        "first": "Jane",
        "last": "Smith"
    }
    
    # Test lead with mixed keys
    lead3 = {
        "Owner(s) Email": "test3@example.com",
        "Website": "https://example3.com",
        "Company": "Example Corp 3"
    }
    
    # CRITICAL: Test empty string behavior - must match original .get() behavior
    lead4 = {
        "Email 1": "",  # Empty string - should return "", NOT try next key
        "email": "should_not_use@example.com"
    }
    
    lead5 = {
        "Business": "",  # Empty business - should return "", NOT try "Company"
        "Company": "Should Not Use This"
    }
    
    tests = [
        (get_email(lead1), "test@example.com", "Old-style email"),
        (get_email(lead2), "test2@example.com", "New-style email"),
        (get_email(lead3), "test3@example.com", "Owner email"),
        (get_website(lead1), "https://example.com", "Old-style website"),
        (get_company(lead1), "Example Corp", "Business key"),
        (get_company(lead3), "Example Corp 3", "Company key"),
        (get_first_name(lead1), "John", "Old-style first name"),
        (get_last_name(lead2), "Smith", "New-style last name"),
        # CRITICAL TESTS: Empty string preservation
        (get_email(lead4), "", "Empty Email 1 returns empty (NOT next key)"),
        (get_company(lead5), "", "Empty Business returns empty (NOT Company)"),
        (get_field(lead4, "Email 1", "email"), "", "get_field with empty first key"),
    ]
    
    all_passed = True
    for actual, expected, description in tests:
        if actual == expected:
            print(f"  ✅ {description}: '{actual}'")
        else:
            print(f"  ❌ {description}: expected '{expected}', got '{actual}'")
            all_passed = False
    
    return all_passed

if __name__ == "__main__":
    print("=" * 60)
    print("Lead Field Extraction Refactoring Test")
    print("=" * 60)
    
    imports_ok = test_imports()
    extractors_ok = test_field_extractors()
    
    print("\n" + "=" * 60)
    if imports_ok and extractors_ok:
        print("✅ ALL TESTS PASSED - Refactoring successful!")
        print("=" * 60)
        exit(0)
    else:
        print("❌ SOME TESTS FAILED")
        print("=" * 60)
        exit(1)
