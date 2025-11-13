"""
Proof-of-Work (PoW) Anti-DDoS Protection

Prevents DDoS attacks by requiring computational work for each request.
Designed to be:
- Fast to verify (O(1) - single hash)
- Expensive to compute (~65k attempts for 4 leading zeros)
- Trustless (all verification in TEE, logged to Arweave)

Economics:
- Without PoW: 1M req/sec = FREE for attacker
- With PoW: 1M req/sec = $50k/hour (100k CPUs required)

Design:
- Difficulty: 4 leading zeros (~65k hash attempts = ~0.1s on modern CPU)
- Timestamp: Must be within last 60 seconds (prevents replay attacks)
- Nonce: Random value computed by miner to satisfy difficulty
"""

import hashlib
from datetime import datetime, timezone, timedelta
from typing import Tuple


# PoW difficulty (number of leading zeros required)
DIFFICULTY = 4  # 4 zeros = ~65,536 attempts = ~0.1s on modern CPU

# Timestamp tolerance (seconds)
TIMESTAMP_TOLERANCE = 60  # Proof must be computed within last 60 seconds


def verify_pow(
    challenge: str,
    timestamp: str,
    nonce: str,
    difficulty: int = DIFFICULTY
) -> Tuple[bool, str]:
    """
    Verify a Proof-of-Work submission.
    
    Args:
        challenge: Unique challenge string (e.g., lead_id, endpoint name)
        timestamp: ISO 8601 timestamp when PoW was computed
        nonce: Nonce value that satisfies the difficulty requirement
        difficulty: Number of leading zeros required (default: 4)
        
    Returns:
        Tuple[bool, str]: (is_valid, error_message)
            - is_valid: True if PoW is valid, False otherwise
            - error_message: Empty if valid, error description if invalid
    
    Example:
        >>> verify_pow("lead_123", "2025-11-13T12:00:00Z", "45678", 4)
        (True, "")
    """
    # Step 1: Verify timestamp is recent
    try:
        pow_time = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        now = datetime.now(timezone.utc)
        
        time_diff = (now - pow_time).total_seconds()
        
        if time_diff < 0:
            return False, "Timestamp is in the future"
        
        if time_diff > TIMESTAMP_TOLERANCE:
            return False, f"Timestamp too old (max {TIMESTAMP_TOLERANCE}s)"
            
    except Exception as e:
        return False, f"Invalid timestamp format: {str(e)}"
    
    # Step 2: Compute hash and verify difficulty
    try:
        # Hash format: SHA256(challenge:timestamp:nonce)
        message = f"{challenge}:{timestamp}:{nonce}"
        hash_result = hashlib.sha256(message.encode()).hexdigest()
        
        # Check if hash has required leading zeros
        required_prefix = "0" * difficulty
        if not hash_result.startswith(required_prefix):
            return False, f"Insufficient difficulty (expected {difficulty} leading zeros, got {hash_result[:difficulty]})"
        
        # Valid PoW!
        return True, ""
        
    except Exception as e:
        return False, f"Hash computation error: {str(e)}"


def compute_pow(
    challenge: str,
    difficulty: int = DIFFICULTY,
    max_attempts: int = 10_000_000
) -> Tuple[str, str, int]:
    """
    Compute a valid Proof-of-Work (for testing/miner reference implementation).
    
    NOTE: Miners should implement this locally for best performance.
    This is just a reference implementation.
    
    Args:
        challenge: Unique challenge string (e.g., lead_id)
        difficulty: Number of leading zeros required
        max_attempts: Maximum nonce attempts before giving up
        
    Returns:
        Tuple[str, str, int]: (timestamp, nonce, attempts)
            - timestamp: ISO 8601 timestamp when PoW was computed
            - nonce: Nonce value that satisfies difficulty
            - attempts: Number of attempts required
            
    Raises:
        Exception: If no valid nonce found within max_attempts
    """
    timestamp = datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
    required_prefix = "0" * difficulty
    
    for nonce in range(max_attempts):
        message = f"{challenge}:{timestamp}:{nonce}"
        hash_result = hashlib.sha256(message.encode()).hexdigest()
        
        if hash_result.startswith(required_prefix):
            return timestamp, str(nonce), nonce + 1
    
    raise Exception(f"Could not find valid nonce after {max_attempts} attempts")


def get_pow_stats() -> dict:
    """
    Get current PoW configuration stats.
    
    Returns:
        dict: {difficulty, avg_attempts, avg_time_ms, timestamp_tolerance}
    """
    avg_attempts = 16 ** DIFFICULTY  # 16^n for n leading hex zeros
    avg_time_ms = avg_attempts / 1_000_000 * 100  # Assume 1M hashes/sec
    
    return {
        "difficulty": DIFFICULTY,
        "avg_attempts": avg_attempts,
        "avg_time_ms": avg_time_ms,
        "timestamp_tolerance_seconds": TIMESTAMP_TOLERANCE
    }

