"""
LeadPoet Canonical Chain Weight Functions

This module provides canonical chain weight normalization used across
gateway, validator, and auditor components.

CRITICAL: All components MUST use this function to normalize chain weights.
Using int() directly on float weights TRUNCATES to 0, causing false equivocation!

Security Model:
- subtensor.weights() may return floats [0,1] or ints [0,65535] depending on version
- This module ensures consistent handling regardless of return type
- Float detection uses isinstance() (type-based), not magnitude comparison
"""

from typing import List, Tuple


def normalize_chain_weights(raw_pairs: List[Tuple]) -> List[Tuple[int, int]]:
    """
    Normalize chain weights to u16 pairs, handling both float and int returns.
    
    CRITICAL: subtensor.weights() may return floats [0,1] or ints [0,65535].
    Using int(w) on floats TRUNCATES to 0, causing false equivocation detection!
    
    Args:
        raw_pairs: List of (uid, weight) tuples from subtensor.weights()
                   Weights may be floats [0,1] or ints [0,65535]
                   
    Returns:
        List of (uid: int, weight: int) pairs, sorted ascending by uid
        Weights are normalized to u16 range [0, 65535]
        
    Example:
        >>> # Float weights (normalized [0,1])
        >>> raw = [(0, 0.5), (1, 0.3), (2, 0.2)]
        >>> normalize_chain_weights(raw)
        [(0, 32767), (1, 19660), (2, 13107)]
        
        >>> # Int weights (already u16)
        >>> raw = [(0, 32767), (1, 19660), (2, 13107)]
        >>> normalize_chain_weights(raw)
        [(0, 32767), (1, 19660), (2, 13107)]
        
    Security Note:
        This function uses type-based float detection (isinstance) which is more
        robust than magnitude-based detection (max_w <= 1.0). Magnitude detection
        fails on edge cases like very small u16 values.
    """
    if not raw_pairs:
        return []
    
    # Import here to avoid circular imports
    from leadpoet_canonical.weights import normalize_to_u16
    
    # FIX #7: Detect floats by TYPE, not magnitude (more robust)
    # Magnitude-based detection (max_w <= 1.0) fails on edge cases
    has_floats = any(isinstance(w, float) for _, w in raw_pairs)
    
    if has_floats:
        # Float weights - convert to u16 using BT's normalization
        uids = [int(u) for u, _ in raw_pairs]
        floats = [float(w) for _, w in raw_pairs]
        weights_u16 = normalize_to_u16(uids, floats)
        pairs = list(zip(uids, weights_u16))
    else:
        # Already u16 - safe to cast
        pairs = [(int(u), int(w)) for u, w in raw_pairs]
    
    # Sort by UID (canonical ordering)
    return sorted(pairs, key=lambda x: x[0])


def filter_nonzero_chain_weights(pairs: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """
    Filter out zero weights from normalized chain weight pairs.
    
    This converts dense representation to sparse representation as required
    by the canonical hash functions.
    
    Args:
        pairs: List of (uid, weight_u16) tuples from normalize_chain_weights()
        
    Returns:
        List of (uid, weight_u16) tuples with zeros removed, sorted by UID
    """
    return [(uid, w) for uid, w in pairs if w > 0]


# =============================================================================
# UNIT TESTS
# =============================================================================

def test_normalize_chain_weights_floats():
    """Test normalization of float weights."""
    # Simulate float weights from subtensor.weights()
    raw = [(0, 0.5), (1, 0.3), (2, 0.2)]
    result = normalize_chain_weights(raw)
    
    # Should convert to u16
    assert len(result) == 3, f"Expected 3 pairs, got {len(result)}"
    assert all(isinstance(w, int) for _, w in result), "All weights should be ints"
    assert all(0 <= w <= 65535 for _, w in result), "All weights should be in u16 range"
    
    # Should be sorted by UID
    uids = [u for u, _ in result]
    assert uids == sorted(uids), "UIDs should be sorted"
    
    print("✅ Float normalization test passed")


def test_normalize_chain_weights_ints():
    """Test pass-through of int weights."""
    # Simulate int weights (already u16)
    raw = [(2, 30000), (0, 20000), (1, 15535)]  # Unsorted
    result = normalize_chain_weights(raw)
    
    # Should preserve values and sort
    assert len(result) == 3, f"Expected 3 pairs, got {len(result)}"
    assert result[0] == (0, 20000), f"First pair should be (0, 20000), got {result[0]}"
    assert result[1] == (1, 15535), f"Second pair should be (1, 15535), got {result[1]}"
    assert result[2] == (2, 30000), f"Third pair should be (2, 30000), got {result[2]}"
    
    print("✅ Int pass-through test passed")


def test_normalize_chain_weights_empty():
    """Test handling of empty input."""
    result = normalize_chain_weights([])
    assert result == [], "Empty input should return empty list"
    print("✅ Empty input test passed")


def test_normalize_chain_weights_type_detection():
    """Test type-based float detection (not magnitude-based)."""
    # Edge case: small int values that could be confused for floats if using magnitude
    raw = [(0, 1), (1, 0), (2, 1)]  # Small ints, not floats
    result = normalize_chain_weights(raw)
    
    # Should treat as ints (pass-through), not floats
    # If treated as floats, normalization would change the values
    assert (0, 1) in result, "Small int should be preserved"
    assert (2, 1) in result, "Small int should be preserved"
    
    print("✅ Type detection test passed")


def test_filter_nonzero():
    """Test filtering zero weights for sparse representation."""
    pairs = [(0, 100), (1, 0), (2, 200), (3, 0), (4, 300)]
    result = filter_nonzero_chain_weights(pairs)
    
    assert len(result) == 3, f"Expected 3 non-zero pairs, got {len(result)}"
    assert result == [(0, 100), (2, 200), (4, 300)], f"Unexpected result: {result}"
    
    print("✅ Filter nonzero test passed")


if __name__ == "__main__":
    print("Running leadpoet_canonical/chain.py unit tests...\n")
    
    test_normalize_chain_weights_floats()
    test_normalize_chain_weights_ints()
    test_normalize_chain_weights_empty()
    test_normalize_chain_weights_type_detection()
    test_filter_nonzero()
    
    print("\n" + "=" * 50)
    print("All tests completed!")

