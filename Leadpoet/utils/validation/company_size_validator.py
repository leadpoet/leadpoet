"""
Company Size Validation Module

Comprehensive company size and employee count validation for lead quality.

Features:
- Employee count range validation
- Company size tier classification
- Growth rate analysis
- Headcount-to-revenue consistency checks
- Location-to-employee distribution validation
- Industry-specific size benchmarks
"""

import re
import asyncio
from typing import Dict, Tuple, Optional, List
from datetime import datetime
from enum import Enum

# Cache for company size validation results
_size_cache: Dict[str, Tuple[bool, Dict, datetime]] = {}
SIZE_CACHE_TTL_HOURS = 168  # 7 days


class CompanySizeTier(Enum):
    """Company size tiers based on employee count"""
    MICRO = "micro"  # 1-10 employees
    SMALL = "small"  # 11-50 employees
    MEDIUM = "medium"  # 51-250 employees
    LARGE = "large"  # 251-1000 employees
    ENTERPRISE = "enterprise"  # 1001+ employees


# Industry-specific employee count benchmarks
INDUSTRY_BENCHMARKS = {
    "technology": {
        "typical_range": (10, 500),
        "growth_rate_threshold": 0.3,  # 30% YoY growth is common
        "revenue_per_employee": (150000, 500000)  # $150K-$500K per employee
    },
    "software": {
        "typical_range": (5, 200),
        "growth_rate_threshold": 0.4,
        "revenue_per_employee": (200000, 800000)
    },
    "manufacturing": {
        "typical_range": (50, 5000),
        "growth_rate_threshold": 0.1,
        "revenue_per_employee": (80000, 200000)
    },
    "retail": {
        "typical_range": (20, 10000),
        "growth_rate_threshold": 0.15,
        "revenue_per_employee": (50000, 150000)
    },
    "finance": {
        "typical_range": (10, 1000),
        "growth_rate_threshold": 0.2,
        "revenue_per_employee": (100000, 400000)
    },
    "healthcare": {
        "typical_range": (20, 5000),
        "growth_rate_threshold": 0.15,
        "revenue_per_employee": (70000, 180000)
    },
    "default": {
        "typical_range": (10, 1000),
        "growth_rate_threshold": 0.2,
        "revenue_per_employee": (80000, 300000)
    }
}


def _is_cache_valid(timestamp: datetime, ttl_hours: int = SIZE_CACHE_TTL_HOURS) -> bool:
    """Check if cached result is still valid"""
    age = datetime.now() - timestamp
    return age.total_seconds() < (ttl_hours * 3600)


def _classify_company_size(employee_count: int) -> CompanySizeTier:
    """
    Classify company into size tier based on employee count.
    
    Args:
        employee_count: Number of employees
    
    Returns:
        CompanySizeTier enum value
    """
    if employee_count <= 10:
        return CompanySizeTier.MICRO
    elif employee_count <= 50:
        return CompanySizeTier.SMALL
    elif employee_count <= 250:
        return CompanySizeTier.MEDIUM
    elif employee_count <= 1000:
        return CompanySizeTier.LARGE
    else:
        return CompanySizeTier.ENTERPRISE


def _get_industry_benchmark(industry: str) -> Dict:
    """
    Get industry-specific benchmarks.
    
    Args:
        industry: Industry name
    
    Returns:
        Dictionary with benchmark values
    """
    industry_lower = industry.lower() if industry else ""
    
    # Try to match industry keywords
    for key, benchmark in INDUSTRY_BENCHMARKS.items():
        if key in industry_lower:
            return benchmark
    
    return INDUSTRY_BENCHMARKS["default"]


async def validate_employee_count(
    employee_count: Optional[int],
    industry: str = None,
    company_name: str = None
) -> Tuple[bool, Dict]:
    """
    Validate employee count is within reasonable range for industry.
    
    Args:
        employee_count: Number of employees
        industry: Company industry
        company_name: Company name (for caching)
    
    Returns:
        (is_valid, validation_info)
        
    Example:
        >>> await validate_employee_count(150, "Technology", "Acme Corp")
        (True, {
            "employee_count": 150,
            "size_tier": "medium",
            "within_industry_range": True
        })
    """
    try:
        if employee_count is None:
            # Employee count is optional - not a hard failure
            return True, {
                "employee_count": None,
                "message": "No employee count provided (optional field)",
                "validation_skipped": True
            }
        
        # Check cache
        cache_key = f"employee_count:{company_name}:{employee_count}"
        if cache_key in _size_cache:
            cached_result, cached_data, timestamp = _size_cache[cache_key]
            if _is_cache_valid(timestamp):
                return cached_result, cached_data
        
        # Validate employee count is positive
        if employee_count < 0:
            rejection = {
                "stage": "Company Size Validation",
                "check_name": "validate_employee_count",
                "message": f"Invalid employee count: {employee_count} (must be positive)",
                "failed_fields": ["employee_count"],
                "employee_count": employee_count
            }
            _size_cache[cache_key] = (False, rejection, datetime.now())
            return False, rejection
        
        # Classify size tier
        size_tier = _classify_company_size(employee_count)
        
        # Get industry benchmark
        benchmark = _get_industry_benchmark(industry)
        min_expected, max_expected = benchmark["typical_range"]
        
        # Check if within industry range (soft warning, not rejection)
        within_range = min_expected <= employee_count <= max_expected
        
        validation_info = {
            "employee_count": employee_count,
            "size_tier": size_tier.value,
            "industry": industry,
            "within_industry_range": within_range,
            "industry_typical_range": benchmark["typical_range"]
        }
        
        # Add warning if outside typical range
        if not within_range:
            if employee_count < min_expected:
                validation_info["warning"] = f"Employee count ({employee_count}) is below typical range for {industry} ({min_expected}-{max_expected})"
            else:
                validation_info["warning"] = f"Employee count ({employee_count}) is above typical range for {industry} ({min_expected}-{max_expected})"
        
        # Cache result
        _size_cache[cache_key] = (True, validation_info, datetime.now())
        
        return True, validation_info
        
    except Exception as e:
        rejection = {
            "stage": "Company Size Validation",
            "check_name": "validate_employee_count",
            "message": f"Employee count validation error: {str(e)}",
            "failed_fields": ["employee_count"],
            "error": str(e)
        }
        return False, rejection


async def validate_company_size_tier(
    lead: dict,
    expected_tier: CompanySizeTier = None
) -> Tuple[bool, Dict]:
    """
    Validate company size tier classification.
    
    Args:
        lead: Lead dictionary with employee count
        expected_tier: Expected size tier (optional)
    
    Returns:
        (is_valid, tier_info)
    """
    try:
        # Extract employee count from various possible fields
        employee_count = (
            lead.get("employee_count") or
            lead.get("number_of_employees") or
            lead.get("employees") or
            lead.get("company_size")
        )
        
        if employee_count is None:
            return True, {
                "message": "No employee count provided",
                "validation_skipped": True
            }
        
        # Convert to int if string
        if isinstance(employee_count, str):
            # Handle ranges like "50-100"
            if "-" in employee_count:
                parts = employee_count.split("-")
                try:
                    employee_count = int(parts[0])  # Use lower bound
                except ValueError:
                    return True, {
                        "message": f"Cannot parse employee count range: {employee_count}",
                        "validation_skipped": True
                    }
            else:
                try:
                    employee_count = int(employee_count)
                except ValueError:
                    return True, {
                        "message": f"Cannot parse employee count: {employee_count}",
                        "validation_skipped": True
                    }
        
        # Classify tier
        actual_tier = _classify_company_size(employee_count)
        
        tier_info = {
            "employee_count": employee_count,
            "size_tier": actual_tier.value,
            "tier_description": {
                "micro": "1-10 employees",
                "small": "11-50 employees",
                "medium": "51-250 employees",
                "large": "251-1000 employees",
                "enterprise": "1001+ employees"
            }[actual_tier.value]
        }
        
        # Check against expected tier (if provided)
        if expected_tier:
            tier_matches = actual_tier == expected_tier
            tier_info["expected_tier"] = expected_tier.value
            tier_info["tier_matches"] = tier_matches
            
            if not tier_matches:
                tier_info["warning"] = f"Company size tier '{actual_tier.value}' does not match expected '{expected_tier.value}'"
        
        return True, tier_info
        
    except Exception as e:
        rejection = {
            "stage": "Company Size Validation",
            "check_name": "validate_company_size_tier",
            "message": f"Size tier validation error: {str(e)}",
            "failed_fields": ["employee_count"],
            "error": str(e)
        }
        return False, rejection


async def check_size_consistency(
    lead: dict,
    check_revenue: bool = True,
    check_locations: bool = True
) -> Tuple[bool, Dict]:
    """
    Check consistency between company size and other metrics.
    
    Validates:
    - Revenue-to-employee ratio
    - Number of locations vs employee count
    - Industry-specific benchmarks
    
    Args:
        lead: Lead dictionary
        check_revenue: Whether to check revenue consistency
        check_locations: Whether to check location consistency
    
    Returns:
        (is_consistent, consistency_info)
    """
    try:
        # Extract data
        employee_count = (
            lead.get("employee_count") or
            lead.get("number_of_employees") or
            lead.get("employees")
        )
        
        if not employee_count:
            return True, {
                "message": "No employee count provided",
                "validation_skipped": True
            }
        
        # Convert to int
        if isinstance(employee_count, str):
            try:
                employee_count = int(employee_count.split("-")[0])
            except ValueError:
                return True, {"validation_skipped": True}
        
        industry = lead.get("industry") or lead.get("Industry")
        benchmark = _get_industry_benchmark(industry)
        
        consistency_checks = []
        warnings = []
        
        # Check 1: Revenue consistency
        if check_revenue:
            revenue = lead.get("revenue") or lead.get("annual_revenue")
            if revenue:
                # Convert revenue to number
                if isinstance(revenue, str):
                    # Remove currency symbols and parse
                    revenue_str = re.sub(r'[^\d.]', '', revenue)
                    try:
                        revenue = float(revenue_str)
                    except ValueError:
                        revenue = None
                
                if revenue:
                    revenue_per_employee = revenue / employee_count
                    min_expected, max_expected = benchmark["revenue_per_employee"]
                    
                    within_range = min_expected <= revenue_per_employee <= max_expected
                    
                    consistency_checks.append({
                        "check": "revenue_per_employee",
                        "value": round(revenue_per_employee, 2),
                        "expected_range": benchmark["revenue_per_employee"],
                        "within_range": within_range
                    })
                    
                    if not within_range:
                        if revenue_per_employee < min_expected:
                            warnings.append(f"Low revenue per employee: ${revenue_per_employee:,.0f} (expected ${min_expected:,.0f}-${max_expected:,.0f})")
                        else:
                            warnings.append(f"High revenue per employee: ${revenue_per_employee:,.0f} (expected ${min_expected:,.0f}-${max_expected:,.0f})")
        
        # Check 2: Location consistency
        if check_locations:
            num_locations = lead.get("number_of_locations")
            if num_locations:
                if isinstance(num_locations, str):
                    try:
                        num_locations = int(num_locations)
                    except ValueError:
                        num_locations = None
                
                if num_locations:
                    # Rough heuristic: expect at least 10 employees per location
                    min_employees_per_location = 5
                    max_employees_per_location = 500
                    
                    employees_per_location = employee_count / num_locations
                    
                    reasonable = min_employees_per_location <= employees_per_location <= max_employees_per_location
                    
                    consistency_checks.append({
                        "check": "employees_per_location",
                        "value": round(employees_per_location, 1),
                        "num_locations": num_locations,
                        "reasonable": reasonable
                    })
                    
                    if not reasonable:
                        if employees_per_location < min_employees_per_location:
                            warnings.append(f"Too many locations ({num_locations}) for employee count ({employee_count})")
                        else:
                            warnings.append(f"Very large locations: {employees_per_location:.0f} employees per location")
        
        consistency_info = {
            "employee_count": employee_count,
            "industry": industry,
            "consistency_checks": consistency_checks,
            "warnings": warnings,
            "all_consistent": len(warnings) == 0
        }
        
        return True, consistency_info
        
    except Exception as e:
        rejection = {
            "stage": "Company Size Validation",
            "check_name": "check_size_consistency",
            "message": f"Size consistency check error: {str(e)}",
            "failed_fields": ["employee_count"],
            "error": str(e)
        }
        return False, rejection


async def validate_growth_rate(
    current_count: int,
    previous_count: int,
    time_period_years: float,
    industry: str = None
) -> Tuple[bool, Dict]:
    """
    Validate company growth rate is reasonable for industry.
    
    Args:
        current_count: Current employee count
        previous_count: Previous employee count
        time_period_years: Time period in years
        industry: Company industry
    
    Returns:
        (is_reasonable, growth_info)
        
    Example:
        >>> await validate_growth_rate(150, 100, 1.0, "Technology")
        (True, {
            "growth_rate": 0.5,
            "reasonable": True,
            "message": "50% growth over 1.0 years"
        })
    """
    try:
        if previous_count <= 0 or time_period_years <= 0:
            return True, {
                "message": "Invalid input for growth rate calculation",
                "validation_skipped": True
            }
        
        # Calculate growth rate
        growth_rate = (current_count - previous_count) / previous_count
        annualized_growth = growth_rate / time_period_years
        
        # Get industry benchmark
        benchmark = _get_industry_benchmark(industry)
        max_reasonable_growth = benchmark["growth_rate_threshold"]
        
        # Check if growth is reasonable
        # Allow negative growth (shrinkage) but flag excessive growth
        is_reasonable = annualized_growth <= max_reasonable_growth * 2  # 2x threshold
        
        growth_info = {
            "current_count": current_count,
            "previous_count": previous_count,
            "time_period_years": time_period_years,
            "growth_rate": round(growth_rate, 3),
            "annualized_growth_rate": round(annualized_growth, 3),
            "industry_threshold": max_reasonable_growth,
            "reasonable": is_reasonable
        }
        
        if not is_reasonable:
            growth_info["warning"] = f"Unusually high growth rate: {annualized_growth*100:.1f}% per year (industry typical: {max_reasonable_growth*100:.1f}%)"
        
        return True, growth_info
        
    except Exception as e:
        rejection = {
            "stage": "Company Size Validation",
            "check_name": "validate_growth_rate",
            "message": f"Growth rate validation error: {str(e)}",
            "error": str(e)
        }
        return False, rejection


# Utility functions
def extract_employee_count(lead: dict) -> Optional[int]:
    """Extract employee count from lead data"""
    employee_count = (
        lead.get("employee_count") or
        lead.get("number_of_employees") or
        lead.get("employees") or
        lead.get("company_size")
    )
    
    if not employee_count:
        return None
    
    # Convert to int
    if isinstance(employee_count, str):
        # Handle ranges
        if "-" in employee_count:
            try:
                return int(employee_count.split("-")[0])
            except ValueError:
                return None
        
        # Remove non-numeric characters
        cleaned = re.sub(r'[^\d]', '', employee_count)
        try:
            return int(cleaned)
        except ValueError:
            return None
    
    return int(employee_count)
