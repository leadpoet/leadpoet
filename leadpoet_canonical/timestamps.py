"""
LeadPoet Canonical Timestamp Functions

This module provides canonical timestamp generation for the TEE logging system.

CRITICAL RULES:
1. Timestamps are RFC3339 format, UTC timezone, with Z suffix
2. NO microseconds (truncated to seconds for determinism)
3. Timestamps ONLY appear in signed_event.timestamp, NEVER in payload
4. All components MUST use this function for timestamp generation

Security Note:
The single canonical timestamp source prevents canonicalization bugs where
the same logical time could be stored in multiple places with slight differences.
"""

from datetime import datetime, timezone


def canonical_timestamp() -> str:
    """
    Generate a canonical timestamp for TEE event logging.
    
    Format: RFC3339 UTC with Z suffix, truncated to seconds (no microseconds).
    Example: "2024-01-15T12:30:45Z"
    
    Returns:
        Timestamp string in format "YYYY-MM-DDTHH:MM:SSZ"
        
    Security Notes:
        - Always UTC timezone (Z suffix)
        - No microseconds (ensures reproducibility in tests, prevents drift)
        - Used ONLY in signed_event.timestamp, NOT in payloads
        
    Example:
        >>> ts = canonical_timestamp()
        >>> print(ts)
        "2024-01-15T12:30:45Z"
        >>> # Verify format
        >>> assert ts.endswith("Z")
        >>> assert "." not in ts  # No microseconds
    """
    # Get current UTC time
    now = datetime.now(timezone.utc)
    
    # Format as RFC3339 with Z suffix, truncated to seconds
    # strftime %Y-%m-%dT%H:%M:%S gives us ISO format without timezone
    # We explicitly add Z to indicate UTC
    return now.strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_canonical_timestamp(ts: str) -> datetime:
    """
    Parse a canonical timestamp back to a datetime object.
    
    Args:
        ts: Timestamp string in format "YYYY-MM-DDTHH:MM:SSZ"
        
    Returns:
        datetime object in UTC timezone
        
    Raises:
        ValueError: If timestamp format is invalid
    """
    if not ts.endswith("Z"):
        raise ValueError(f"Timestamp must end with Z (UTC): {ts}")
    
    # Remove Z and parse
    ts_without_z = ts[:-1]
    dt = datetime.strptime(ts_without_z, "%Y-%m-%dT%H:%M:%S")
    
    # Attach UTC timezone
    return dt.replace(tzinfo=timezone.utc)


def validate_timestamp_format(ts: str) -> bool:
    """
    Validate that a timestamp matches the canonical format.
    
    Args:
        ts: Timestamp string to validate
        
    Returns:
        True if format is valid, False otherwise
    """
    try:
        # Must end with Z
        if not ts.endswith("Z"):
            return False
        
        # Must not have microseconds (no . character)
        if "." in ts:
            return False
        
        # Must parse successfully
        parse_canonical_timestamp(ts)
        return True
        
    except (ValueError, AttributeError):
        return False


# =============================================================================
# UNIT TESTS
# =============================================================================

def test_canonical_timestamp_format():
    """Test that canonical_timestamp produces valid format."""
    ts = canonical_timestamp()
    
    # Must end with Z (UTC)
    assert ts.endswith("Z"), f"Timestamp must end with Z: {ts}"
    
    # Must not have microseconds
    assert "." not in ts, f"Timestamp must not have microseconds: {ts}"
    
    # Must have correct length (YYYY-MM-DDTHH:MM:SSZ = 20 chars)
    assert len(ts) == 20, f"Timestamp wrong length: {len(ts)} (expected 20)"
    
    # Must have T separator
    assert "T" in ts, f"Timestamp must have T separator: {ts}"
    
    print(f"✅ Timestamp format test passed: {ts}")


def test_parse_canonical_timestamp():
    """Test parsing a canonical timestamp."""
    ts = "2024-01-15T12:30:45Z"
    dt = parse_canonical_timestamp(ts)
    
    assert dt.year == 2024, f"Wrong year: {dt.year}"
    assert dt.month == 1, f"Wrong month: {dt.month}"
    assert dt.day == 15, f"Wrong day: {dt.day}"
    assert dt.hour == 12, f"Wrong hour: {dt.hour}"
    assert dt.minute == 30, f"Wrong minute: {dt.minute}"
    assert dt.second == 45, f"Wrong second: {dt.second}"
    assert dt.tzinfo == timezone.utc, f"Wrong timezone: {dt.tzinfo}"
    
    print("✅ Parse timestamp test passed")


def test_validate_timestamp_format():
    """Test timestamp validation."""
    # Valid
    assert validate_timestamp_format("2024-01-15T12:30:45Z") == True
    
    # Invalid: no Z suffix
    assert validate_timestamp_format("2024-01-15T12:30:45") == False
    
    # Invalid: has microseconds
    assert validate_timestamp_format("2024-01-15T12:30:45.123456Z") == False
    
    # Invalid: wrong format
    assert validate_timestamp_format("2024/01/15 12:30:45Z") == False
    
    print("✅ Validate timestamp format test passed")


def test_round_trip():
    """Test that generate -> parse -> format produces same result."""
    ts = canonical_timestamp()
    dt = parse_canonical_timestamp(ts)
    
    # Reformat should match
    reformatted = dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    assert ts == reformatted, f"Round-trip mismatch: {ts} != {reformatted}"
    
    print("✅ Round-trip test passed")


if __name__ == "__main__":
    print("Running leadpoet_canonical/timestamps.py unit tests...\n")
    
    test_canonical_timestamp_format()
    test_parse_canonical_timestamp()
    test_validate_timestamp_format()
    test_round_trip()
    
    print("\n" + "=" * 50)
    print("All tests completed!")

