"""
Epoch Audit Log Generator - USES CANONICAL FORMAT

This module generates audit logs at the end of each epoch, comparing all validators'
on-chain weights against the TEE-published bundle.

IMPORTANT: On-chain weights don't have "block" metadata. The block in the TEE bundle
represents "when computed" but chain weights are simply stored without that timestamp.
We use a WEIGHTS-ONLY hash (netuid, epoch_id, weights) for comparison since block
is not available for chain weights.

SECURITY MODEL:
- TEE bundle is the source of truth (signed by validator enclave)
- PRIMARY validator: Compare against captured chain snapshot (not live chain)
- AUDITOR validators: Use tolerance-based comparison (±1 u16 drift allowed)
- Audit results are logged to transparency_log (gateway-signed)

STATUS CODES:
- VERIFIED: All validators' weights match TEE computation
- EQUIVOCATION_DETECTED: Primary validator submitted different weights to chain
- AUDITOR_MISMATCH: Auditor validators have mismatched weights
- NO_TEE_BUNDLE: No TEE bundle published for this epoch
- NO_PRIMARY_WEIGHTS: Primary validator hasn't set weights on chain
"""

import logging
from typing import Dict, List, Optional, Any

from gateway.db.client import get_read_client, get_write_client

# Import canonical functions from shared module
from leadpoet_canonical.timestamps import canonical_timestamp
from leadpoet_canonical.weights import compare_weights_hash, weights_within_tolerance
from leadpoet_canonical.chain import normalize_chain_weights

logger = logging.getLogger(__name__)


async def generate_epoch_audit_log(
    epoch_id: int,
    subtensor,
    netuid: int = 71,
) -> Dict[str, Any]:
    """
    Generate audit log comparing TEE weights vs on-chain weights.
    Uses CANONICAL format: UIDs + u16 weights.
    
    This audits ALL validators on the network to see if their on-chain
    weights match the TEE-published bundle.
    
    CRITICAL SECURITY RULES:
    1. PRIMARY validator: Compare against chain_snapshot_compare_hash (captured at submission)
       - subtensor.weights() returns CURRENT weights, which may have changed
       - Using live chain would allow equivocation to go undetected
    2. AUDITOR validators: Use tolerance-based comparison (±1 u16)
       - u16 → float → u16 round-trip may cause ±1 differences
       - This is expected behavior, not equivocation
    
    Args:
        epoch_id: Epoch to audit
        subtensor: Bittensor subtensor client
        netuid: Subnet ID (default 71)
        
    Returns:
        Audit log dict with status and per-validator results
    """
    read_client = get_read_client()  # Anon key for reads (RLS-safe)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # Step 1: Fetch TEE bundle from Supabase
    # ═══════════════════════════════════════════════════════════════════════════
    result = read_client.table("published_weight_bundles") \
        .select("*") \
        .eq("netuid", netuid) \
        .eq("epoch_id", epoch_id) \
        .order("created_at", desc=True) \
        .limit(1) \
        .execute()
    
    if not result.data:
        logger.info(f"No TEE bundle found for epoch {epoch_id}")
        return {
            "epoch_id": epoch_id,
            "netuid": netuid,
            "status": "NO_TEE_BUNDLE",
            "message": f"No TEE bundle published for epoch {epoch_id}",
            # NOTE: No timestamp in payload - sign_event() adds it
        }
    
    bundle = result.data[0]
    tee_validator_hotkey = bundle["validator_hotkey"]
    tee_validator_pubkey = bundle["validator_enclave_pubkey"]  # CANONICAL name
    
    # Compute compare hash from TEE bundle (NO block for fair chain comparison)
    tee_pairs = list(zip(bundle["uids"], bundle["weights_u16"]))
    tee_hash = compare_weights_hash(netuid, epoch_id, tee_pairs)
    
    # Get captured snapshot hash for PRIMARY validator comparison
    # CRITICAL: Use snapshot, NOT live chain (prevents undetected equivocation)
    chain_snapshot_hash = bundle.get("chain_snapshot_compare_hash")
    chain_snapshot_block = bundle.get("chain_snapshot_block")
    
    logger.info(
        f"📊 Auditing epoch {epoch_id}: TEE hash={tee_hash[:16]}..., "
        f"snapshot_hash={chain_snapshot_hash[:16] if chain_snapshot_hash else 'N/A'}..."
    )
    
    # ═══════════════════════════════════════════════════════════════════════════
    # Step 2: Audit ALL validators on the network
    # ═══════════════════════════════════════════════════════════════════════════
    try:
        metagraph = subtensor.metagraph(netuid=netuid)
    except Exception as e:
        logger.error(f"Failed to fetch metagraph for netuid {netuid}: {e}")
        return {
            "epoch_id": epoch_id,
            "netuid": netuid,
            "status": "METAGRAPH_ERROR",
            "message": f"Failed to fetch metagraph: {str(e)}",
        }
    
    validators_audit: List[Dict[str, Any]] = []
    
    for uid, hotkey in enumerate(metagraph.hotkeys):
        try:
            is_primary = (hotkey == tee_validator_hotkey)
            
            # ═══════════════════════════════════════════════════════════════════
            # PRIMARY VALIDATOR: Use captured snapshot (not live chain)
            # ═══════════════════════════════════════════════════════════════════
            if is_primary and chain_snapshot_hash:
                # Compare TEE bundle against captured snapshot
                matches_tee = (tee_hash == chain_snapshot_hash)
                chain_hash = chain_snapshot_hash
                comparison_method = "snapshot"
                
                logger.debug(
                    f"UID {uid} (PRIMARY): snapshot match={matches_tee}, "
                    f"tee_hash={tee_hash[:16]}..., chain_hash={chain_hash[:16]}..."
                )
                
            else:
                # ═══════════════════════════════════════════════════════════════════
                # AUDITOR VALIDATORS: Use tolerance-based comparison
                # ═══════════════════════════════════════════════════════════════════
                try:
                    on_chain_weights = subtensor.weights(netuid=netuid, uid=uid)
                except Exception as e:
                    logger.debug(f"UID {uid}: Could not fetch weights: {e}")
                    continue  # Not a validator or error fetching
                
                if not on_chain_weights:
                    continue  # Not a validator or hasn't set weights
                
                # Normalize chain weights to canonical format
                chain_pairs = normalize_chain_weights(on_chain_weights)
                
                if not chain_pairs:
                    continue  # Empty weights
                
                # Compute compare hash (NO block - chain doesn't have block metadata)
                chain_hash = compare_weights_hash(netuid, epoch_id, chain_pairs)
                
                # Check for exact match first
                if chain_hash == tee_hash:
                    matches_tee = True
                    comparison_method = "exact"
                else:
                    # Fallback: compare actual weights with ±1 tolerance per weight
                    # This handles u16 → float → u16 round-trip drift
                    matches_tee = weights_within_tolerance(tee_pairs, chain_pairs, tolerance=1)
                    comparison_method = "tolerance" if matches_tee else "mismatch"
                
                logger.debug(
                    f"UID {uid} ({'PRIMARY' if is_primary else 'AUDITOR'}): "
                    f"match={matches_tee} ({comparison_method}), "
                    f"tee_hash={tee_hash[:16]}..., chain_hash={chain_hash[:16]}..."
                )
            
            validators_audit.append({
                "uid": uid,
                "hotkey": hotkey,
                "on_chain_hash": chain_hash,
                "matches_tee": matches_tee,
                "is_primary": is_primary,
                "comparison_method": comparison_method,
            })
            
        except Exception as e:
            logger.warning(f"⚠️ Could not audit UID {uid}: {e}")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # Step 3: Determine overall status
    # ═══════════════════════════════════════════════════════════════════════════
    primary_validators = [v for v in validators_audit if v["is_primary"]]
    primary_matches = all(v["matches_tee"] for v in primary_validators) if primary_validators else False
    
    auditor_validators = [v for v in validators_audit if not v["is_primary"]]
    auditors_match = all(v["matches_tee"] for v in auditor_validators) if auditor_validators else True
    
    if not primary_validators:
        status = "NO_PRIMARY_WEIGHTS"
        message = "Primary validator has not set weights on chain"
    elif primary_matches and auditors_match:
        status = "VERIFIED"
        message = "All validators submitted weights matching TEE computation"
    elif not primary_matches:
        status = "EQUIVOCATION_DETECTED"
        message = "Primary validator submitted different weights to chain than to gateway"
    else:
        status = "AUDITOR_MISMATCH"
        message = "Some auditor validators submitted different weights"
    
    logger.info(f"📊 Epoch {epoch_id} audit complete: {status}")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # Step 4: Build audit log
    # NOTE: No "timestamp" here - sign_event() envelope adds it when passed to log_event()
    # ═══════════════════════════════════════════════════════════════════════════
    audit_log = {
        "epoch_id": epoch_id,
        "netuid": netuid,
        "status": status,
        "message": message,
        
        # TEE source of truth
        "tee_weights_hash": tee_hash,
        "tee_validator_hotkey": tee_validator_hotkey,
        "tee_validator_pubkey": tee_validator_pubkey,
        
        # Chain snapshot info (for primary comparison)
        "chain_snapshot_block": chain_snapshot_block,
        "chain_snapshot_hash": chain_snapshot_hash,
        
        # Per-validator audit results
        "validators": validators_audit,
        
        # Summary stats
        "summary": {
            "total_validators": len(validators_audit),
            "primary_count": len(primary_validators),
            "auditor_count": len(auditor_validators),
            "matching_tee": sum(1 for v in validators_audit if v["matches_tee"]),
            "mismatched": sum(1 for v in validators_audit if not v["matches_tee"]),
        },
    }
    
    return audit_log


async def store_audit_log(
    netuid: int,
    epoch_id: int,
    audit_log: Dict[str, Any],
) -> None:
    """
    Store audit log to Supabase and log to transparency log.
    
    Args:
        netuid: Subnet ID
        epoch_id: Epoch ID
        audit_log: Audit log dict from generate_epoch_audit_log()
    """
    write_client = get_write_client()  # Service role for writes
    
    # ═══════════════════════════════════════════════════════════════════════════
    # Store to Supabase (fast queries)
    # NOTE: Use canonical_timestamp() directly (not from audit_log - payloads don't have timestamps)
    # ═══════════════════════════════════════════════════════════════════════════
    try:
        write_client.table("epoch_audit_logs").upsert({
            "netuid": netuid,
            "epoch_id": epoch_id,
            "status": audit_log["status"],
            "audit_data": audit_log,
            "created_at": canonical_timestamp(),
        }, on_conflict="netuid,epoch_id").execute()
        
        logger.info(f"✅ Audit log stored in Supabase for epoch {epoch_id}")
    except Exception as e:
        logger.error(f"Failed to store audit log to Supabase: {e}")
        raise
    
    # ═══════════════════════════════════════════════════════════════════════════
    # Log to TEE buffer for Arweave (permanent record)
    # ═══════════════════════════════════════════════════════════════════════════
    try:
        from gateway.utils.logger import log_event
        await log_event("EPOCH_AUDIT", audit_log)
        
        logger.info(f"✅ Audit log logged to TEE buffer for epoch {epoch_id}: {audit_log['status']}")
    except Exception as e:
        logger.error(f"Failed to log audit to TEE buffer: {e}")
        raise
    
    print(f"✅ Audit log stored for epoch {epoch_id}: {audit_log['status']}")


async def run_epoch_audit(
    epoch_id: int,
    subtensor,
    netuid: int = 71,
) -> Dict[str, Any]:
    """
    Convenience function to generate and store audit log in one call.
    
    Args:
        epoch_id: Epoch to audit
        subtensor: Bittensor subtensor client
        netuid: Subnet ID
        
    Returns:
        Audit log dict
    """
    # Generate audit log
    audit_log = await generate_epoch_audit_log(epoch_id, subtensor, netuid)
    
    # Store it
    await store_audit_log(netuid, epoch_id, audit_log)
    
    return audit_log

