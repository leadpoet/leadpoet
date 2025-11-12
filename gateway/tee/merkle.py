"""
Merkle Tree Implementation for TEE Event Batching

This module provides Merkle tree computation for hourly event checkpoints.
The Merkle root creates a tamper-evident commitment to all buffered events.

Security Properties:
- Changing any event invalidates the root hash
- Inclusion proofs allow verifying specific events without downloading all events
- TEE signs the root hash, proving events came from verified code

Standard Merkle Tree Rules:
- Leaves are SHA256(canonical_event_json)
- Internal nodes are SHA256(left_child + right_child)
- If odd number of leaves, duplicate the last one
- Tree is built bottom-up from leaves to root
"""

import hashlib
import json
from typing import List, Tuple, Optional


def hash_leaf(data: bytes) -> bytes:
    """
    Compute SHA256 hash of leaf data.
    
    Args:
        data: Canonical event JSON bytes
    
    Returns:
        32-byte SHA256 hash
    """
    return hashlib.sha256(data).digest()


def hash_pair(left: bytes, right: bytes) -> bytes:
    """
    Compute SHA256 hash of two child hashes (internal node).
    
    Args:
        left: 32-byte hash of left child
        right: 32-byte hash of right child
    
    Returns:
        32-byte SHA256 hash of concatenated children
    """
    return hashlib.sha256(left + right).digest()


def compute_merkle_tree(events: List[dict]) -> Tuple[bytes, List[List[bytes]]]:
    """
    Computes Merkle tree from list of events.
    
    Algorithm:
    1. Convert each event to canonical JSON (sorted keys)
    2. Hash each event to create leaf nodes
    3. Build tree bottom-up, pairing adjacent nodes
    4. If odd number of nodes at any level, duplicate the last one
    5. Continue until single root node remains
    
    Args:
        events: List of event dicts (must have at least 1 event)
    
    Returns:
        Tuple of (merkle_root, tree_levels):
        - merkle_root: 32-byte root hash
        - tree_levels: List of levels, each level is list of node hashes
                      [0] = leaves, [-1] = root
                      Needed for generating inclusion proofs
    
    Raises:
        ValueError: If events list is empty
    
    Example:
        events = [{"event_type": "TEST", "data": "foo"}]
        root, levels = compute_merkle_tree(events)
        print(root.hex())  # "abc123..."
    """
    if not events:
        raise ValueError("Cannot compute Merkle tree from empty event list")
    
    # Level 0: Convert events to leaf hashes
    leaves = []
    for event in events:
        # Canonicalize event JSON (CRITICAL: sort_keys=True for determinism)
        canonical_json = json.dumps(event, sort_keys=True, separators=(',', ':'))
        canonical_bytes = canonical_json.encode('utf-8')
        leaf_hash = hash_leaf(canonical_bytes)
        leaves.append(leaf_hash)
    
    # Store all tree levels (needed for inclusion proofs)
    tree_levels = [leaves]
    
    # Build tree bottom-up
    current_level = leaves
    
    while len(current_level) > 1:
        next_level = []
        
        # Process pairs
        for i in range(0, len(current_level), 2):
            left = current_level[i]
            
            # If odd number of nodes, duplicate the last one
            if i + 1 < len(current_level):
                right = current_level[i + 1]
            else:
                right = current_level[i]  # Duplicate last node
            
            # Hash the pair to create parent node
            parent = hash_pair(left, right)
            next_level.append(parent)
        
        tree_levels.append(next_level)
        current_level = next_level
    
    # Root is the single node at the top level
    merkle_root = current_level[0]
    
    return merkle_root, tree_levels


def generate_inclusion_proof(
    tree_levels: List[List[bytes]],
    leaf_index: int
) -> List[Tuple[bytes, str]]:
    """
    Generate Merkle inclusion proof for leaf at given index.
    
    The proof consists of sibling hashes needed to reconstruct the path
    from leaf to root. Each sibling is tagged with position ('left' or 'right').
    
    Args:
        tree_levels: Tree levels from compute_merkle_tree()
        leaf_index: Index of leaf in the leaves array (0-based)
    
    Returns:
        List of (sibling_hash, position) tuples
        - sibling_hash: 32-byte hash of sibling node
        - position: 'left' or 'right' (where sibling is relative to path)
    
    Raises:
        ValueError: If leaf_index is out of range
    
    Example:
        proof = generate_inclusion_proof(tree_levels, 0)
        # [(<hash_of_sibling_1>, 'right'), (<hash_of_sibling_2>, 'left'), ...]
    """
    if not tree_levels:
        raise ValueError("Tree levels cannot be empty")
    
    leaves = tree_levels[0]
    if leaf_index < 0 or leaf_index >= len(leaves):
        raise ValueError(f"Leaf index {leaf_index} out of range (0-{len(leaves)-1})")
    
    proof = []
    current_index = leaf_index
    
    # Traverse from leaf to root, collecting sibling hashes
    for level_idx in range(len(tree_levels) - 1):
        level = tree_levels[level_idx]
        
        # Determine if current node is left or right child
        is_left_child = (current_index % 2 == 0)
        
        if is_left_child:
            # Current node is left child, sibling is to the right
            sibling_index = current_index + 1
            if sibling_index < len(level):
                sibling_hash = level[sibling_index]
                proof.append((sibling_hash, 'right'))
            else:
                # Odd number of nodes - sibling is duplicate of current
                sibling_hash = level[current_index]
                proof.append((sibling_hash, 'right'))
        else:
            # Current node is right child, sibling is to the left
            sibling_index = current_index - 1
            sibling_hash = level[sibling_index]
            proof.append((sibling_hash, 'left'))
        
        # Move up to parent node
        current_index = current_index // 2
    
    return proof


def verify_inclusion_proof(
    leaf_hash: bytes,
    proof: List[Tuple[bytes, str]],
    expected_root: bytes
) -> bool:
    """
    Verify that a leaf is included in a Merkle tree with given root.
    
    Reconstructs the root hash by hashing the leaf with proof siblings,
    then compares to expected root.
    
    Args:
        leaf_hash: 32-byte hash of the leaf to verify
        proof: Inclusion proof from generate_inclusion_proof()
        expected_root: 32-byte root hash to verify against
    
    Returns:
        True if leaf is in tree with expected_root, False otherwise
    
    Example:
        is_valid = verify_inclusion_proof(leaf_hash, proof, root)
        if is_valid:
            print("✅ Event verified in checkpoint")
        else:
            print("❌ Event NOT in checkpoint")
    """
    # Start with leaf hash
    current_hash = leaf_hash
    
    # Apply each proof step
    for sibling_hash, position in proof:
        if position == 'left':
            # Sibling is on left, current is on right
            current_hash = hash_pair(sibling_hash, current_hash)
        elif position == 'right':
            # Current is on left, sibling is on right
            current_hash = hash_pair(current_hash, sibling_hash)
        else:
            raise ValueError(f"Invalid proof position: {position}")
    
    # Final hash should match expected root
    return current_hash == expected_root


def compute_event_leaf_hash(event: dict) -> bytes:
    """
    Helper function to compute leaf hash for a single event.
    
    This is useful when you need to verify an event against a proof
    without recomputing the entire tree.
    
    Args:
        event: Event dict
    
    Returns:
        32-byte SHA256 hash of canonical event JSON
    """
    canonical_json = json.dumps(event, sort_keys=True, separators=(',', ':'))
    canonical_bytes = canonical_json.encode('utf-8')
    return hash_leaf(canonical_bytes)

