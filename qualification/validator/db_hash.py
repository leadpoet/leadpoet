"""
Qualification System: Database Integrity Hash

Phase 3.3 from tasks10.md

This module computes a SHA256 hash of all lead_ids accessible to models.
The hash is computed at the start of each benchmark run inside the TEE
and included in the transparency log for audit purposes.

Purpose:
- Proves which leads were in the database when the benchmark ran
- Enables auditors to verify the model had access to the same data
- Detects if leads were added/removed during evaluation
- Provides cryptographic proof for anti-equivocation

CRITICAL: This is a NEW utility for qualification auditing.
Do NOT modify any existing database query code or hash computation
in other parts of the codebase.
"""

import hashlib
import logging
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any, Tuple

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

# Table containing leads accessible to models
LEADS_TABLE = "leads_private"

# Column containing the lead ID
LEAD_ID_COLUMN = "lead_id"

# Maximum leads to query (safety limit)
MAX_LEADS_QUERY = 100000

# Batch size for pagination
QUERY_BATCH_SIZE = 10000


# =============================================================================
# Main Hash Computation Function
# =============================================================================

async def compute_db_hash(supabase_client) -> str:
    """
    Compute SHA256 hash of all lead_ids accessible to models.
    
    Called at the start of each benchmark run inside TEE.
    The hash proves which leads were available when the evaluation started.
    
    Algorithm:
    1. Query all lead_ids from the leads database
    2. Sort them lexicographically
    3. Join with ":" separator
    4. Compute SHA256 hash of the joined string
    
    Args:
        supabase_client: Supabase client instance (async)
    
    Returns:
        SHA256 hash as hexadecimal string (64 chars)
    
    Example:
        >>> hash = await compute_db_hash(supabase)
        >>> len(hash)
        64
        >>> hash
        'a1b2c3d4e5f6...'
    """
    logger.info("Computing database integrity hash...")
    start_time = datetime.now(timezone.utc)
    
    try:
        # Query all lead_ids from the leads database
        # This queries the same DB that models have access to
        lead_ids = await _fetch_all_lead_ids(supabase_client)
        
        if not lead_ids:
            logger.warning("No leads found in database - returning empty hash")
            return hashlib.sha256(b"").hexdigest()
        
        # Sort lead_ids lexicographically for deterministic ordering
        lead_ids_sorted = sorted(lead_ids)
        
        # Compute hash
        hash_input = ":".join(lead_ids_sorted)
        db_hash = hashlib.sha256(hash_input.encode("utf-8")).hexdigest()
        
        elapsed = (datetime.now(timezone.utc) - start_time).total_seconds()
        logger.info(
            f"Database hash computed: hash={db_hash[:16]}..., "
            f"leads={len(lead_ids_sorted)}, time={elapsed:.2f}s"
        )
        
        return db_hash
        
    except Exception as e:
        logger.error(f"Failed to compute database hash: {e}")
        raise DatabaseHashError(f"Failed to compute database hash: {e}") from e


async def _fetch_all_lead_ids(supabase_client) -> List[str]:
    """
    Fetch all lead_ids from the leads database.
    
    Uses pagination to handle large datasets efficiently.
    
    Args:
        supabase_client: Supabase client instance
    
    Returns:
        List of lead_id strings
    """
    all_lead_ids = []
    offset = 0
    
    while True:
        # Query batch of lead_ids
        response = await supabase_client.table(LEADS_TABLE) \
            .select(LEAD_ID_COLUMN) \
            .order(LEAD_ID_COLUMN) \
            .range(offset, offset + QUERY_BATCH_SIZE - 1) \
            .execute()
        
        if not response.data:
            break
        
        # Extract lead_ids from response
        batch_ids = [row[LEAD_ID_COLUMN] for row in response.data]
        all_lead_ids.extend(batch_ids)
        
        # Check if we got fewer than batch size (end of data)
        if len(response.data) < QUERY_BATCH_SIZE:
            break
        
        # Safety check
        if len(all_lead_ids) >= MAX_LEADS_QUERY:
            logger.warning(f"Hit max leads limit: {MAX_LEADS_QUERY}")
            break
        
        offset += QUERY_BATCH_SIZE
    
    return all_lead_ids


# =============================================================================
# Extended Hash Functions
# =============================================================================

async def compute_db_hash_with_metadata(supabase_client) -> Dict[str, Any]:
    """
    Compute database hash with additional metadata for auditing.
    
    Returns hash along with:
    - Total lead count
    - Computation timestamp
    - First and last lead_ids (for quick verification)
    
    Args:
        supabase_client: Supabase client instance
    
    Returns:
        Dict with hash and metadata
    """
    logger.info("Computing database hash with metadata...")
    start_time = datetime.now(timezone.utc)
    
    try:
        # Fetch all lead_ids
        lead_ids = await _fetch_all_lead_ids(supabase_client)
        
        if not lead_ids:
            return {
                "db_hash": hashlib.sha256(b"").hexdigest(),
                "lead_count": 0,
                "computed_at": start_time.isoformat(),
                "first_lead_id": None,
                "last_lead_id": None
            }
        
        # Sort lead_ids
        lead_ids_sorted = sorted(lead_ids)
        
        # Compute hash
        hash_input = ":".join(lead_ids_sorted)
        db_hash = hashlib.sha256(hash_input.encode("utf-8")).hexdigest()
        
        elapsed = (datetime.now(timezone.utc) - start_time).total_seconds()
        
        return {
            "db_hash": db_hash,
            "lead_count": len(lead_ids_sorted),
            "computed_at": start_time.isoformat(),
            "computation_time_seconds": elapsed,
            "first_lead_id": lead_ids_sorted[0],
            "last_lead_id": lead_ids_sorted[-1]
        }
        
    except Exception as e:
        logger.error(f"Failed to compute database hash with metadata: {e}")
        raise DatabaseHashError(f"Failed to compute database hash: {e}") from e


async def verify_db_hash(
    supabase_client,
    expected_hash: str
) -> Tuple[bool, Optional[str]]:
    """
    Verify that the current database hash matches an expected hash.
    
    Used by auditors to verify the database state hasn't changed.
    
    Args:
        supabase_client: Supabase client instance
        expected_hash: The expected SHA256 hash (64 hex chars)
    
    Returns:
        Tuple of (matches: bool, current_hash: str)
    """
    current_hash = await compute_db_hash(supabase_client)
    matches = current_hash == expected_hash
    
    if not matches:
        logger.warning(
            f"Database hash mismatch: expected={expected_hash[:16]}..., "
            f"got={current_hash[:16]}..."
        )
    
    return matches, current_hash


# =============================================================================
# Subset Hash Functions
# =============================================================================

async def compute_icp_leads_hash(
    supabase_client,
    icp_ids: List[str]
) -> str:
    """
    Compute hash of leads matching specific ICPs.
    
    Useful for verifying the subset of leads used in an evaluation.
    
    Args:
        supabase_client: Supabase client instance
        icp_ids: List of ICP identifiers
    
    Returns:
        SHA256 hash of matching lead_ids
    """
    # For now, this is a placeholder that computes the full hash
    # In production, you might filter leads by ICP criteria
    logger.warning(
        "compute_icp_leads_hash: Using full database hash "
        "(ICP filtering not implemented)"
    )
    return await compute_db_hash(supabase_client)


def compute_hash_from_ids(lead_ids: List[str]) -> str:
    """
    Compute hash from a list of lead_ids.
    
    Utility function for computing hash without database query.
    Useful for testing or when you already have the lead_ids.
    
    Args:
        lead_ids: List of lead_id strings
    
    Returns:
        SHA256 hash as hexadecimal string
    """
    # Sort for deterministic ordering
    lead_ids_sorted = sorted(lead_ids)
    
    # Compute hash
    hash_input = ":".join(lead_ids_sorted)
    return hashlib.sha256(hash_input.encode("utf-8")).hexdigest()


# =============================================================================
# Placeholder Supabase Client (for testing)
# =============================================================================

class PlaceholderSupabaseClient:
    """
    Placeholder Supabase client for testing db_hash functions.
    
    In production, use the real Supabase client from gateway.db.client.
    """
    
    def __init__(self, lead_ids: Optional[List[str]] = None):
        self._lead_ids = lead_ids or []
        self._current_table = None
        self._current_select = None
        self._current_order = None
        self._range_start = 0
        self._range_end = 999
    
    def table(self, name: str):
        self._current_table = name
        return self
    
    def select(self, columns: str):
        self._current_select = columns
        return self
    
    def order(self, column: str):
        self._current_order = column
        return self
    
    def range(self, start: int, end: int):
        self._range_start = start
        self._range_end = end
        return self
    
    async def execute(self):
        """Return mock response."""
        # Slice lead_ids based on range
        sliced = self._lead_ids[self._range_start:self._range_end + 1]
        
        class MockResponse:
            def __init__(self, data):
                self.data = data
        
        return MockResponse([{LEAD_ID_COLUMN: lid} for lid in sliced])


# =============================================================================
# Exceptions
# =============================================================================

class DatabaseHashError(Exception):
    """Raised when database hash computation fails."""
    pass


# =============================================================================
# CLI Entry Point (for testing)
# =============================================================================

async def main():
    """Test the database hash computation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Compute database integrity hash")
    parser.add_argument("--test", action="store_true", help="Run with test data")
    parser.add_argument("--verify", type=str, help="Hash to verify against")
    args = parser.parse_args()
    
    if args.test:
        # Use placeholder client with test data
        test_lead_ids = [f"lead_{i:06d}" for i in range(1000)]
        client = PlaceholderSupabaseClient(test_lead_ids)
        
        print("Testing with placeholder client...")
        result = await compute_db_hash_with_metadata(client)
        
        print(f"\nDatabase Hash: {result['db_hash']}")
        print(f"Lead Count: {result['lead_count']}")
        print(f"Computed At: {result['computed_at']}")
        print(f"First Lead: {result['first_lead_id']}")
        print(f"Last Lead: {result['last_lead_id']}")
        
        # Verify reproducibility
        hash2 = await compute_db_hash(client)
        print(f"\nReproducibility check: {result['db_hash'] == hash2}")
        
    else:
        print("Use --test flag to run with test data")
        print("In production, pass a real Supabase client to compute_db_hash()")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
