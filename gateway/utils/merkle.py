"""
Merkle Tree Utilities for LeadPoet Gateway

Provides functions for:
- Computing Merkle roots from lists of values
- Generating inclusion proofs
- Verifying inclusion proofs

Used for:
- QUEUE_ROOT snapshots (pending leads at epoch start)
- EPOCH_MANIFEST verification (validator work proofs)
- CHECKPOINT_ROOT (transparency log integrity)
"""

import hashlib
from typing import List


def compute_merkle_root(leaves: List[str]) -> str:
    """
    Compute Merkle root from list of leaf values.
    
    Builds a binary Merkle tree bottom-up by repeatedly hashing pairs
    of nodes until a single root hash remains.
    
    Args:
        leaves: List of strings (UUIDs, hashes, etc.)
    
    Returns:
        Merkle root hash (SHA256 hex string)
    
    Example:
        >>> leaves = ["lead_id_1", "lead_id_2", "lead_id_3"]
        >>> root = compute_merkle_root(leaves)
        >>> root
        'a3b2c1d4e5f6789...'
    """
    if not leaves:
        # Empty tree returns all zeros
        return "0" * 64
    
    # Hash each leaf to create initial level
    hashed_leaves = [hashlib.sha256(leaf.encode('utf-8')).digest() for leaf in leaves]
    
    # Build tree bottom-up
    current_level = hashed_leaves
    
    while len(current_level) > 1:
        next_level = []
        
        # Process pairs of nodes
        for i in range(0, len(current_level), 2):
            left = current_level[i]
            
            # If odd number of nodes, duplicate the last one
            if i + 1 < len(current_level):
                right = current_level[i + 1]
            else:
                right = left  # Duplicate last node
            
            # Parent hash = H(left || right)
            parent = hashlib.sha256(left + right).digest()
            next_level.append(parent)
        
        current_level = next_level
    
    # Return root as hex string
    return current_level[0].hex()


def compute_merkle_proof(leaves: List[str], target_index: int) -> List[str]:
    """
    Compute Merkle inclusion proof for leaf at target_index.
    
    Returns the list of (sibling_hash, is_left) tuples needed to reconstruct
    the path from the target leaf to the root.
    
    Args:
        leaves: List of leaf values
        target_index: Index of target leaf (0-based)
    
    Returns:
        List of proof elements in format "L:hash" or "R:hash"
        where L means sibling is on the left, R means sibling is on the right
    
    Example:
        >>> leaves = ["lead_1", "lead_2", "lead_3", "lead_4"]
        >>> proof = compute_merkle_proof(leaves, 1)
        >>> proof
        ['L:abc123...', 'R:def456...']  # Sibling positions and hashes
    
    Raises:
        IndexError: If target_index is out of bounds
    """
    if target_index >= len(leaves) or target_index < 0:
        return []
    
    # Hash all leaves
    hashed_leaves = [hashlib.sha256(leaf.encode('utf-8')).digest() for leaf in leaves]
    
    proof = []
    current_level = hashed_leaves
    current_index = target_index
    
    while len(current_level) > 1:
        next_level = []
        
        for i in range(0, len(current_level), 2):
            left = current_level[i]
            
            if i + 1 < len(current_level):
                right = current_level[i + 1]
            else:
                right = left
            
            # If current_index is in this pair, add sibling with position
            if i == current_index:
                # Sibling is on the right
                proof.append(f"R:{right.hex()}")
            elif i + 1 == current_index:
                # Sibling is on the left
                proof.append(f"L:{left.hex()}")
            
            parent = hashlib.sha256(left + right).digest()
            next_level.append(parent)
        
        # Update index for next level (parent's index)
        current_index = current_index // 2
        current_level = next_level
    
    return proof


def verify_merkle_proof(leaf: str, proof: List[str], root: str) -> bool:
    """
    Verify Merkle inclusion proof.
    
    Reconstructs the path from leaf to root using the proof and checks
    if it matches the expected root hash.
    
    Args:
        leaf: Leaf value to verify
        proof: List of proof elements in format "L:hash" or "R:hash"
        root: Expected Merkle root hash
    
    Returns:
        True if proof is valid and leaf is in the tree
    
    Example:
        >>> leaf = "lead_2"
        >>> proof = ['L:abc123...', 'R:def456...']
        >>> root = 'a3b2c1d4...'
        >>> verify_merkle_proof(leaf, proof, root)
        True
    """
    # Start with hash of leaf
    current = hashlib.sha256(leaf.encode('utf-8')).digest()
    
    # Climb the tree using sibling hashes
    for proof_element in proof:
        # Parse position and hash
        if ':' in proof_element:
            position, sibling_hex = proof_element.split(':', 1)
            sibling = bytes.fromhex(sibling_hex)
            
            # Compute parent hash based on sibling position
            if position == 'L':
                # Sibling is on the left, so: parent = H(sibling || current)
                current = hashlib.sha256(sibling + current).digest()
            else:  # position == 'R'
                # Sibling is on the right, so: parent = H(current || sibling)
                current = hashlib.sha256(current + sibling).digest()
        else:
            # Legacy format without position (shouldn't happen in production)
            # Fall back to lexicographic ordering
            sibling = bytes.fromhex(proof_element)
            if current <= sibling:
                current = hashlib.sha256(current + sibling).digest()
            else:
                current = hashlib.sha256(sibling + current).digest()
    
    # Check if we arrived at the expected root
    return current.hex() == root


def compute_merkle_root_from_hashes(hashes: List[str]) -> str:
    """
    Compute Merkle root from list of pre-computed hashes.
    
    Similar to compute_merkle_root but takes hex hashes instead of raw values.
    Useful for computing checkpoint roots from event hashes.
    
    Args:
        hashes: List of hex hash strings
    
    Returns:
        Merkle root hash (SHA256 hex string)
    
    Example:
        >>> event_hashes = ['abc123...', 'def456...', '789ghi...']
        >>> root = compute_merkle_root_from_hashes(event_hashes)
        >>> root
        'a3b2c1d4e5f6789...'
    """
    if not hashes:
        return "0" * 64
    
    # Convert hex hashes to bytes
    hashed_leaves = [bytes.fromhex(h) for h in hashes]
    
    # Build tree bottom-up (same as compute_merkle_root)
    current_level = hashed_leaves
    
    while len(current_level) > 1:
        next_level = []
        
        for i in range(0, len(current_level), 2):
            left = current_level[i]
            
            if i + 1 < len(current_level):
                right = current_level[i + 1]
            else:
                right = left
            
            parent = hashlib.sha256(left + right).digest()
            next_level.append(parent)
        
        current_level = next_level
    
    return current_level[0].hex()

