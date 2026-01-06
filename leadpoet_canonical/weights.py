"""
LeadPoet Canonical Weight Functions

This module provides the canonical implementations for all weight-related operations
used across gateway, validator, and auditor components.

CRITICAL: All components MUST use these exact implementations. Any deviation will
cause hash mismatches and false equivocation detection.

Security Model:
- bundle_weights_hash() → validator signs, gateway verifies bundle integrity
- compare_weights_hash() → auditors compare TEE vs on-chain (equivocation check)
- weights_within_tolerance() → auditor ±1 drift tolerance comparison

Sparse Representation:
- All weight bundles use SPARSE representation (only non-zero weights)
- UIDs must be sorted ascending
- No padding to metagraph.n
- This matches how Bittensor stores weights on-chain
"""

import hashlib
import json
from typing import List, Tuple, Optional

from leadpoet_canonical.constants import VERSION_KEY, AUDITOR_WEIGHT_TOLERANCE


def bundle_weights_hash(netuid: int, epoch_id: int, block: int, weights: List[Tuple[int, int]]) -> str:
    """
    H_bundle: Hash for INTERNAL bundle integrity (includes block).
    
    Use this for:
    - Signing weight bundles (validator TEE signs this)
    - Verifying bundle hasn't been tampered
    
    Args:
        netuid: Subnet ID
        epoch_id: Epoch identifier
        block: Block number when weights were computed
        weights: List of (uid, weight_u16) tuples
                 MUST be sparse (no zeros) and sorted ascending by uid.
                 Filter out zero weights before calling this function.
                 
    Returns:
        SHA256 hash as hex string (64 characters)
        
    Security Note:
        The validator TEE signs this hash. Any modification to the bundle
        will produce a different hash, making tampering detectable.
    """
    # Sort by UID (canonical ordering)
    sorted_weights = sorted(weights, key=lambda x: x[0])
    
    payload = {
        "netuid": netuid,
        "epoch_id": epoch_id,
        "block": block,
        "weights": [[uid, w_u16] for uid, w_u16 in sorted_weights]
    }
    
    # Canonical JSON: sorted keys, no whitespace
    canonical_json = json.dumps(payload, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(canonical_json.encode('utf-8')).hexdigest()


def compare_weights_hash(netuid: int, epoch_id: int, weights: List[Tuple[int, int]]) -> str:
    """
    H_compare: Hash for CHAIN COMPARISON (NO block - on-chain weights don't have it).
    
    Use this for:
    - Comparing TEE bundle vs on-chain weights (equivocation detection)
    - Epoch audit logs
    
    CRITICAL: On-chain weights don't have "block" metadata, so we can't
    include it when comparing. Both sides must use the same hash.
    
    Args:
        netuid: Subnet ID
        epoch_id: Epoch identifier
        weights: List of (uid, weight_u16) tuples
                 MUST be sparse (no zeros) and sorted ascending by uid.
                 Filter out zero weights before calling this function.
                 
    Returns:
        SHA256 hash as hex string (64 characters)
        
    Security Note:
        This hash is used for equivocation detection. If a validator submits
        different weights to the gateway vs the chain, the hashes won't match.
    """
    # Sort by UID (canonical ordering)
    sorted_weights = sorted(weights, key=lambda x: x[0])
    
    payload = {
        "netuid": netuid,
        "epoch_id": epoch_id,
        # NO BLOCK - on-chain weights don't have this
        "weights": [[uid, w_u16] for uid, w_u16 in sorted_weights]
    }
    
    # Canonical JSON: sorted keys, no whitespace
    canonical_json = json.dumps(payload, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(canonical_json.encode('utf-8')).hexdigest()


def weights_within_tolerance(
    expected: List[Tuple[int, int]], 
    actual: List[Tuple[int, int]], 
    tolerance: int = AUDITOR_WEIGHT_TOLERANCE
) -> bool:
    """
    Tolerance-based weight comparison for auditor validators (handles sparse vectors).
    
    Issue 17: u16 → float → u16 round-trip may cause ±1 differences.
    For AUDITOR comparisons, we accept this drift as "matching".
    
    Args:
        expected: List of (uid, weight_u16) tuples (TEE bundle)
        actual: List of (uid, weight_u16) tuples (auditor's on-chain weights)
        tolerance: Maximum per-weight drift (default: 1 for ±1)
    
    Returns:
        True if all weights match within tolerance, False otherwise.
    
    SPARSE HANDLING: A weight of 1 can drift to 0 and be filtered out, causing
    UID to disappear. We handle this by comparing on UNION of UIDs with missing
    treated as 0. An absent UID with weight <= tolerance on the other side is OK.
    
    NOTE: Use EXACT hash equality for PRIMARY validator equivocation checks.
    Tolerance is ONLY for auditor comparisons where round-trip drift is expected.
    """
    expected_dict = {uid: w for uid, w in expected}
    actual_dict = {uid: w for uid, w in actual}
    
    # Compare on UNION of UIDs (handles sparse drift where weight=1 becomes 0)
    all_uids = set(expected_dict.keys()) | set(actual_dict.keys())
    
    for uid in all_uids:
        expected_w = expected_dict.get(uid, 0)  # Missing = 0 (filtered out)
        actual_w = actual_dict.get(uid, 0)
        
        if abs(expected_w - actual_w) > tolerance:
            return False
    
    return True


def normalize_to_u16(uids: List[int], weights: List[float]) -> List[int]:
    """
    Convert float weights to u16 using Bittensor's EXACT method.
    
    CRITICAL: Do NOT reimplement this. Rounding differences will cause
    false "equivocation detected" events because hashes won't match.
    
    Args:
        uids: List of UIDs (MUST pass real UIDs, not range(len(weights)))
        weights: List of float weights corresponding to uids
        
    Returns:
        List of u16 weights [0-65535]
    
    NOTE: This function lives in leadpoet_canonical/ shared module
    to ensure gateway, validator, and auditor all use identical logic.
    
    Fail-Closed: If bittensor import fails, raises ImportError.
    We do NOT fall back to a custom implementation that might differ.
    """
    if not weights:
        return []
    
    try:
        from bittensor.utils.weight_utils import convert_weights_and_uids_for_emit
        import numpy as np
    except ImportError as e:
        raise ImportError(
            "Failed to import bittensor.utils.weight_utils. "
            "This module requires the exact Bittensor library to ensure hash consistency. "
            f"Original error: {e}"
        )
    
    # CRITICAL: Pass the REAL uids, not range(len(weights))
    # Convert to numpy arrays as required by Bittensor API
    uids_array = np.array(uids, dtype=np.int64)
    weights_array = np.array(weights, dtype=np.float32)
    
    _, normalized = convert_weights_and_uids_for_emit(
        uids_array,
        weights_array,
    )
    return list(normalized)


def u16_to_emit_floats(uids: List[int], weights_u16: List[int]) -> List[float]:
    """
    Convert u16 weights to floats that will round-trip through Bittensor.
    
    DESIGN GOAL: Minimize drift when round-tripping through Bittensor.
    TOLERANCE: ±1 u16 drift is acceptable due to floating point precision.
    
    CRITICAL: This must be tested against your PINNED Bittensor version.
    Run test_u16_round_trip_with_tolerance() to verify drift stays within ±1.
    
    Args:
        uids: List of UIDs (not used in calculation but included for API consistency)
        weights_u16: List of u16 weights [0-65535]
        
    Returns:
        List of float weights normalized to sum to 1.0
    """
    if not weights_u16:
        return []
    
    # Simple division - works for most cases
    total = sum(weights_u16)
    if total == 0:
        return [0.0] * len(weights_u16)
    
    # Convert to floats normalized to sum to 1.0
    floats = [w / total for w in weights_u16]
    
    return floats


def filter_sparse_weights(weights: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """
    Filter out zero weights and sort by UID for sparse representation.
    
    This helper ensures weights meet the canonical sparse format required
    by bundle_weights_hash() and compare_weights_hash().
    
    Args:
        weights: List of (uid, weight_u16) tuples (may include zeros)
        
    Returns:
        List of (uid, weight_u16) tuples with zeros removed, sorted by UID
    """
    # Filter out zeros and sort by UID
    return sorted([(uid, w) for uid, w in weights if w > 0], key=lambda x: x[0])


def validate_weights_invariants(uids: List[int], weights_u16: List[int]) -> Tuple[bool, Optional[str]]:
    """
    Validate that weights meet all required invariants for submission.
    
    Invariants:
    1. Arrays have same length
    2. Arrays are not empty
    3. UIDs are strictly increasing (sorted ascending, no duplicates)
    4. All weights are in valid u16 range [1, 65535] (sparse: no zeros)
    
    Args:
        uids: List of UIDs
        weights_u16: List of u16 weights
        
    Returns:
        Tuple of (is_valid, error_message)
        If valid: (True, None)
        If invalid: (False, "error description")
    """
    # Check 1: Same length
    if len(uids) != len(weights_u16):
        return False, f"Array length mismatch: {len(uids)} UIDs vs {len(weights_u16)} weights"
    
    # Check 2: Non-empty
    if len(uids) == 0:
        return False, "Empty weight submission not allowed"
    
    # Check 3: UIDs strictly increasing
    for i in range(1, len(uids)):
        if uids[i] <= uids[i-1]:
            return False, f"UIDs not strictly increasing: uids[{i-1}]={uids[i-1]}, uids[{i}]={uids[i]}"
    
    # Check 4: All weights in valid range [1, 65535] (sparse representation requires > 0)
    for i, w in enumerate(weights_u16):
        if w < 1 or w > 65535:
            return False, f"Weight out of range [1, 65535]: weights[{i}]={w} (uid={uids[i]})"
    
    return True, None


# =============================================================================
# UNIT TESTS
# Run these before deployment to verify round-trip behavior
# =============================================================================

def test_u16_round_trip_exact():
    """
    Verify u16 → float → u16 round-trips exactly (DIAGNOSTIC ONLY).
    
    NOTE: This test is informational. Some ±1 drift is acceptable per Issue 17.
    Use test_u16_round_trip_with_tolerance() for the authoritative test.
    
    ⚠️ CRITICAL TEST METHODOLOGY:
    We ONLY test u16 vectors that Bittensor can actually emit.
    Not all arbitrary u16 vectors are emittable by BT's normalization.
    
    Test approach: START from floats, normalize to u16, then verify round-trip.
    This ensures we only test u16 values that BT can produce.
    """
    import random
    
    passed = 0
    failed = 0
    
    for _ in range(1000):
        # Step 1: Generate RANDOM FLOATS (this is what validators compute)
        n = random.randint(10, 200)
        uids = list(range(n))
        raw_floats = [random.random() for _ in range(n)]
        
        # Normalize floats to sum to 1.0
        total = sum(raw_floats)
        if total == 0:
            continue
        normalized_floats = [f / total for f in raw_floats]
        
        # Step 2: Convert to u16 using Bittensor's EXACT method
        # This produces u16 values that BT CAN emit
        weights_u16 = normalize_to_u16(uids, normalized_floats)
        
        # Step 3: Now test round-trip from these EMITTABLE u16 values
        floats_for_emit = u16_to_emit_floats(uids, weights_u16)
        result_u16 = normalize_to_u16(uids, floats_for_emit)
        
        # Step 4: Verify EXACT match
        all_match = True
        for orig, result in zip(weights_u16, result_u16):
            diff = abs(orig - result)
            if diff > 0:
                all_match = False
                break
        
        if all_match:
            passed += 1
        else:
            failed += 1
            # Don't fail immediately - collect stats
    
    print(f"Round-trip test: {passed} passed, {failed} failed")
    
    if failed > 0:
        # Some vectors may not round-trip perfectly due to BT's internal normalization
        # This is acceptable IF the difference is small (±1 is common)
        print(f"⚠️  {failed} vectors had drift - this may be acceptable")
        print(f"    Consider accepting ±1 tolerance if rate is low (<5%)")
        if failed / (passed + failed) > 0.05:
            raise AssertionError(f"Too many failures: {failed}/{passed+failed} ({100*failed/(passed+failed):.1f}%)")
    else:
        print("✅ u16 round-trip test passed (all 1000 vectors)")


def test_u16_round_trip_with_tolerance():
    """
    Alternative test: Accept ±1 tolerance (more realistic).
    
    BT's normalization can cause small drifts that are practically harmless.
    """
    import random
    
    max_drift = 0
    
    for _ in range(1000):
        n = random.randint(10, 200)
        uids = list(range(n))
        raw_floats = [random.random() for _ in range(n)]
        total = sum(raw_floats)
        if total == 0:
            continue
        normalized_floats = [f / total for f in raw_floats]
        
        weights_u16 = normalize_to_u16(uids, normalized_floats)
        
        # IMPORTANT: u16_to_emit_floats must receive the SAME number of elements
        # as weights_u16, not the original uids list (which may have different length
        # if some weights became 0 during normalization)
        uids_for_emit = list(range(len(weights_u16)))
        floats_for_emit = u16_to_emit_floats(uids_for_emit, weights_u16)
        result_u16 = normalize_to_u16(uids_for_emit, floats_for_emit)
        
        for orig, result in zip(weights_u16, result_u16):
            drift = abs(orig - result)
            max_drift = max(max_drift, drift)
    
    print(f"Max observed drift: ±{max_drift}")
    if max_drift <= AUDITOR_WEIGHT_TOLERANCE:
        print("✅ All drifts within acceptable ±1 tolerance")
    else:
        print(f"❌ Drift exceeds ±{AUDITOR_WEIGHT_TOLERANCE} - investigate u16_to_emit_floats implementation")
        raise AssertionError(f"Drift {max_drift} exceeds tolerance {AUDITOR_WEIGHT_TOLERANCE}")


def test_hash_determinism():
    """
    Verify hash functions are deterministic.
    Same inputs must always produce same hash.
    """
    weights = [(1, 1000), (5, 2000), (10, 3000)]
    
    hash1 = bundle_weights_hash(71, 100, 12345, weights)
    hash2 = bundle_weights_hash(71, 100, 12345, weights)
    assert hash1 == hash2, "bundle_weights_hash not deterministic"
    
    compare1 = compare_weights_hash(71, 100, weights)
    compare2 = compare_weights_hash(71, 100, weights)
    assert compare1 == compare2, "compare_weights_hash not deterministic"
    
    # Verify bundle hash != compare hash (block is included in bundle)
    assert hash1 != compare1, "bundle_weights_hash should differ from compare_weights_hash"
    
    print("✅ Hash determinism test passed")


def test_weights_within_tolerance_sparse():
    """
    Test sparse vector handling in weights_within_tolerance.
    A weight of 1 drifting to 0 should be acceptable within tolerance.
    """
    # Case 1: Exact match
    expected = [(1, 1000), (5, 2000)]
    actual = [(1, 1000), (5, 2000)]
    assert weights_within_tolerance(expected, actual) == True, "Exact match should pass"
    
    # Case 2: Within tolerance (±1)
    expected = [(1, 1000), (5, 2000)]
    actual = [(1, 1001), (5, 1999)]
    assert weights_within_tolerance(expected, actual) == True, "±1 drift should pass"
    
    # Case 3: Outside tolerance
    expected = [(1, 1000), (5, 2000)]
    actual = [(1, 1002), (5, 2000)]
    assert weights_within_tolerance(expected, actual) == False, "±2 drift should fail"
    
    # Case 4: Sparse drift - weight 1 becomes 0 (UID disappears)
    expected = [(1, 1), (5, 2000)]  # UID 1 has weight 1
    actual = [(5, 2000)]  # UID 1 dropped (weight drifted to 0)
    assert weights_within_tolerance(expected, actual) == True, "Weight 1→0 drift should pass"
    
    # Case 5: Sparse drift - new UID appears with weight 1
    expected = [(5, 2000)]
    actual = [(1, 1), (5, 2000)]  # UID 1 appeared with weight 1
    assert weights_within_tolerance(expected, actual) == True, "New UID with weight 1 should pass"
    
    # Case 6: Missing UID with weight > tolerance should fail
    expected = [(1, 5), (5, 2000)]  # UID 1 has weight 5
    actual = [(5, 2000)]  # UID 1 dropped
    assert weights_within_tolerance(expected, actual) == False, "Missing UID with weight 5 should fail"
    
    print("✅ Sparse tolerance test passed")


if __name__ == "__main__":
    # Run all tests when module is executed directly
    print("Running leadpoet_canonical/weights.py unit tests...\n")
    
    print("1. Testing hash determinism...")
    test_hash_determinism()
    print()
    
    print("2. Testing sparse tolerance handling...")
    test_weights_within_tolerance_sparse()
    print()
    
    print("3. Testing u16 round-trip (exact)...")
    try:
        test_u16_round_trip_exact()
    except ImportError as e:
        print(f"⚠️  Skipped: {e}")
    print()
    
    print("4. Testing u16 round-trip (with tolerance)...")
    try:
        test_u16_round_trip_with_tolerance()
    except ImportError as e:
        print(f"⚠️  Skipped: {e}")
    print()
    
    print("=" * 50)
    print("All tests completed!")

