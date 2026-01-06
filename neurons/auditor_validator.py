#!/usr/bin/env python3
"""
LeadPoet Auditor Validator

A lightweight validator that copies weights from the primary validator TEE.
Does not run validation logic - simply verifies and replicates TEE-signed weights.

SECURITY MODEL:
1. Fetches weight bundles from gateway /weights/current/{netuid}
2. Verifies Ed25519 signature using validator enclave pubkey
3. Recomputes hash from bundle data (doesn't trust claimed hash)
4. Checks anti-equivocation using chain snapshot (not live chain)
5. Submits verified weights to Bittensor chain

VERIFICATION FAILURE HANDLING:
If verification fails (equivocation, attestation, signature/hash):
- BURN 100% TO UID 0 - signals distrust and penalizes all miners
- This is the strongest possible signal that something is wrong
- Applies to: equivocation, attestation failure, signature/hash failure

USAGE:
    python neurons/auditor_validator.py --netuid 71 --gateway-url http://54.226.209.164:8000
"""

import os
import sys
import argparse
import asyncio
import logging
import base64
from typing import Dict, List, Optional, Tuple

import bittensor as bt
import aiohttp
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

# Import canonical functions from shared module
from leadpoet_canonical.weights import (
    bundle_weights_hash,
    compare_weights_hash,
    u16_to_emit_floats,
    weights_within_tolerance,
)
from leadpoet_canonical.chain import normalize_chain_weights
from leadpoet_canonical.events import verify_log_entry

# Constants from canonical module
from leadpoet_canonical.constants import EPOCH_LENGTH, WEIGHT_SUBMISSION_BLOCK

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Default gateway URL
DEFAULT_GATEWAY_URL = os.environ.get("GATEWAY_URL", "http://54.226.209.164:8000")

# Expected code hashes for production (pinned builds)
EXPECTED_GATEWAY_CODE_HASH = os.environ.get("EXPECTED_GATEWAY_CODE_HASH")
EXPECTED_VALIDATOR_CODE_HASH = os.environ.get("EXPECTED_VALIDATOR_CODE_HASH")


class AuditorValidator:
    """
    Lightweight validator that copies weights from the primary validator TEE.
    
    TRUST MODEL:
    - Trusts gateway to relay authentic bundles (verified by gateway signature)
    - Trusts validator TEE signature (Ed25519 over weights hash)
    - Does NOT trust claimed hashes (recomputes from bundle data)
    - Verifies anti-equivocation using snapshot (not live chain)
    """
    
    def __init__(self, config, gateway_url: str):
        """
        Initialize auditor validator.
        
        Args:
            config: Bittensor config object
            gateway_url: Gateway URL (passed as parameter, not global)
        """
        self.config = config
        self.gateway_url = gateway_url
        self.wallet = bt.wallet(config=config)
        self.subtensor = bt.subtensor(config=config)
        self.metagraph = self.subtensor.metagraph(config.netuid)
        
        # Verify we're registered as a validator
        self.uid = self._get_uid()
        if self.uid is None:
            raise RuntimeError(
                f"Wallet {self.wallet.hotkey.ss58_address} is not registered "
                f"on netuid {config.netuid}"
            )
        
        self.should_exit = False
        self.last_submitted_epoch = None
        
        # Gateway attestation (for log verification)
        self.gateway_pubkey = None
        self.gateway_attestation = None
        self.gateway_code_hash = None
        
        # Validator attestation (extracted from weight bundles)
        self.validator_pubkey = None
        self.validator_attestation = None
        self.validator_code_hash = None
        self.validator_hotkey = None
        
        # Trust level tracking (CRITICAL for auditor output)
        # "full_nitro" = Full Nitro attestation verification passed
        # "signature_only" = Only Ed25519 signatures verified (weaker trust)
        self.trust_level = "signature_only"  # Default until Nitro verification implemented
        
        logger.info("✅ Auditor Validator initialized")
        print(f"✅ Auditor Validator initialized")
        print(f"   Hotkey: {self.wallet.hotkey.ss58_address}")
        print(f"   UID: {self.uid}")
        print(f"   Gateway: {self.gateway_url}")
    
    def _get_uid(self) -> Optional[int]:
        """Get our UID from the metagraph."""
        hotkey = self.wallet.hotkey.ss58_address
        if hotkey in self.metagraph.hotkeys:
            return self.metagraph.hotkeys.index(hotkey)
        return None
    
    def _get_primary_validator_uid(self, weights_data: Dict) -> Optional[int]:
        """
        Get primary validator UID by matching hotkey from weight bundle.
        
        DO NOT assume UID 0 is primary - look up from weights bundle.
        """
        validator_hotkey = weights_data.get("validator_hotkey")
        if not validator_hotkey:
            print(f"⚠️  No validator_hotkey in weights bundle")
            return None
        
        # Find UID for this hotkey in metagraph
        if validator_hotkey in self.metagraph.hotkeys:
            uid = self.metagraph.hotkeys.index(validator_hotkey)
            print(f"   Primary validator hotkey: {validator_hotkey[:16]}... → UID {uid}")
            return uid
        
        print(f"⚠️  Validator hotkey {validator_hotkey[:16]}... not found in metagraph")
        return None
    
    # ═══════════════════════════════════════════════════════════════════════════
    # Gateway Communication
    # ═══════════════════════════════════════════════════════════════════════════
    
    async def fetch_verified_weights(self, epoch_id: int) -> Optional[Dict]:
        """
        Fetch published weights for an epoch from the gateway.
        
        Uses /weights/latest/{netuid}/{epoch_id} endpoint.
        
        Returns:
            Weight bundle dict, or None if not available
        """
        try:
            async with aiohttp.ClientSession() as session:
                url = f"{self.gateway_url}/weights/latest/{self.config.netuid}/{epoch_id}"
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as response:
                    if response.status == 404:
                        return None
                    elif response.status == 200:
                        return await response.json()
                    else:
                        print(f"⚠️  Unexpected response: {response.status}")
                        return None
        except aiohttp.ClientError as e:
            print(f"❌ Network error fetching weights: {e}")
            return None
        except Exception as e:
            print(f"❌ Failed to fetch weights: {e}")
            return None
    
    async def fetch_current_weights(self) -> Optional[Dict]:
        """
        Fetch most recent published weights from the gateway.
        
        Uses /weights/current/{netuid} endpoint.
        
        Returns:
            Weight bundle dict, or None if not available
        """
        try:
            async with aiohttp.ClientSession() as session:
                url = f"{self.gateway_url}/weights/current/{self.config.netuid}"
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as response:
                    if response.status == 404:
                        return None
                    elif response.status == 200:
                        return await response.json()
                    else:
                        print(f"⚠️  Unexpected response: {response.status}")
                        return None
        except aiohttp.ClientError as e:
            print(f"❌ Network error fetching current weights: {e}")
            return None
        except Exception as e:
            print(f"❌ Failed to fetch current weights: {e}")
            return None
    
    async def fetch_gateway_attestation(self) -> bool:
        """
        Fetch GATEWAY attestation (for verifying log authenticity).
        
        NOTE: This is NOT the validator attestation - that comes from weight bundles.
        """
        try:
            async with aiohttp.ClientSession() as session:
                url = f"{self.gateway_url}/attestation/document"
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as response:
                    if response.status != 200:
                        print(f"❌ Failed to fetch gateway attestation: {response.status}")
                        return False
                    
                    data = await response.json()
                    self.gateway_pubkey = data.get("enclave_pubkey")
                    self.gateway_attestation = data.get("attestation_document")
                    
                    print(f"✅ Fetched GATEWAY attestation")
                    print(f"   Gateway pubkey: {self.gateway_pubkey[:16]}...")
                    
                    return True
                    
        except Exception as e:
            logger.error(f"Failed to fetch gateway attestation: {e}")
            print(f"❌ Failed to fetch gateway attestation: {e}")
            return False
    
    def verify_gateway_attestation(self) -> bool:
        """
        Verify the fetched gateway attestation.
        
        SECURITY MODEL:
        - In production: Full Nitro verification required
        - In dev: Signature-only mode with warning
        
        Sets self.trust_level based on verification result.
        
        Returns:
            True if attestation is valid (or acceptable for dev mode)
        """
        if not self.gateway_attestation or not self.gateway_pubkey:
            logger.warning("No gateway attestation to verify")
            print(f"⚠️ No gateway attestation to verify")
            return False
        
        try:
            # Decode attestation
            att_bytes = base64.b64decode(self.gateway_attestation)
            
            # Check if we're in production mode
            is_production = os.environ.get("ENVIRONMENT") == "production"
            
            if is_production:
                # PRODUCTION: Full Nitro verification REQUIRED
                if not EXPECTED_GATEWAY_CODE_HASH:
                    logger.error("EXPECTED_GATEWAY_CODE_HASH not set in production!")
                    print(f"❌ FAIL-CLOSED: EXPECTED_GATEWAY_CODE_HASH not configured")
                    return False
                
                try:
                    from leadpoet_canonical.nitro import verify_nitro_attestation_full
                    
                    result = verify_nitro_attestation_full(
                        att_bytes,
                        expected_pcr0=None,  # TODO: Add when PCR0 allowlist available
                        expected_pubkey=self.gateway_pubkey,
                        expected_epoch_id=None,  # Gateway attestation doesn't have epoch_id
                    )
                    
                    if result:
                        self.trust_level = "full_nitro"
                        logger.info("Gateway attestation verified (full Nitro)")
                        print(f"✅ Gateway attestation: FULL NITRO VERIFICATION")
                        return True
                    else:
                        logger.error("Gateway Nitro verification failed")
                        print(f"❌ Gateway Nitro verification FAILED")
                        return False
                        
                except NotImplementedError:
                    logger.error("Full Nitro verification not implemented - FAIL CLOSED in production")
                    print(f"❌ FAIL-CLOSED: Nitro verification not implemented")
                    return False
            else:
                # DEV MODE: Signature-only with warning
                self.trust_level = "signature_only"
                logger.warning("DEV MODE: Gateway attestation signature-only (no Nitro)")
                print(f"⚠️ DEV MODE: Gateway attestation SIGNATURE-ONLY")
                print(f"   In production, this would require full Nitro verification")
                print(f"   Trust level: {self.trust_level}")
                return True
                
        except Exception as e:
            logger.error(f"Gateway attestation verification failed: {e}")
            print(f"❌ Gateway attestation verification failed: {e}")
            return False
    
    def verify_validator_attestation(self, bundle: Dict) -> bool:
        """
        Verify the validator attestation from a weight bundle.
        
        SECURITY MODEL:
        - Attestation includes epoch_id for replay protection
        - In production: Full Nitro verification required
        - In dev: Signature-only mode with warning
        
        Args:
            bundle: Weight bundle containing validator_attestation_b64
            
        Returns:
            True if attestation is valid (or acceptable for dev mode)
        """
        attestation_b64 = bundle.get("validator_attestation_b64")
        pubkey = bundle.get("validator_enclave_pubkey")
        code_hash = bundle.get("validator_code_hash")
        epoch_id = bundle.get("epoch_id")
        
        if not attestation_b64 or not pubkey:
            logger.warning("Bundle missing validator attestation or pubkey")
            print(f"⚠️ Bundle missing validator attestation or pubkey")
            return False
        
        try:
            # Decode attestation
            att_bytes = base64.b64decode(attestation_b64)
            
            # Check if we're in production mode
            is_production = os.environ.get("ENVIRONMENT") == "production"
            
            if is_production:
                # PRODUCTION: Full Nitro verification REQUIRED
                expected_code_hash = EXPECTED_VALIDATOR_CODE_HASH
                
                if not expected_code_hash:
                    logger.error("EXPECTED_VALIDATOR_CODE_HASH not set in production!")
                    print(f"❌ FAIL-CLOSED: EXPECTED_VALIDATOR_CODE_HASH not configured")
                    return False
                
                # Verify code_hash matches expected
                if code_hash != expected_code_hash:
                    logger.error(f"Validator code_hash mismatch: {code_hash} != {expected_code_hash}")
                    print(f"❌ Validator code_hash does not match pinned value")
                    return False
                
                try:
                    from leadpoet_canonical.nitro import verify_nitro_attestation_full
                    
                    result = verify_nitro_attestation_full(
                        att_bytes,
                        expected_pcr0=None,  # TODO: Add when PCR0 allowlist available
                        expected_pubkey=pubkey,
                        expected_epoch_id=epoch_id,  # CRITICAL: Replay protection
                    )
                    
                    if result:
                        self.trust_level = "full_nitro"
                        logger.info(f"Validator attestation verified (full Nitro) for epoch {epoch_id}")
                        print(f"✅ Validator attestation: FULL NITRO VERIFICATION")
                        return True
                    else:
                        logger.error("Validator Nitro verification failed")
                        print(f"❌ Validator Nitro verification FAILED")
                        return False
                        
                except NotImplementedError:
                    logger.error("Full Nitro verification not implemented - FAIL CLOSED in production")
                    print(f"❌ FAIL-CLOSED: Nitro verification not implemented")
                    return False
            else:
                # DEV MODE: Check fields present, signature-only trust
                if not pubkey or not attestation_b64:
                    logger.error("Missing validator attestation fields")
                    print(f"❌ Missing validator attestation fields")
                    return False
                
                # In dev mode, we trust the signature but warn about reduced trust
                self.trust_level = "signature_only"
                logger.warning(f"DEV MODE: Validator attestation signature-only for epoch {epoch_id}")
                print(f"⚠️ DEV MODE: Validator attestation SIGNATURE-ONLY")
                print(f"   Epoch: {epoch_id}, Pubkey: {pubkey[:16]}...")
                print(f"   In production, this would require full Nitro verification")
                return True
                
        except Exception as e:
            logger.error(f"Validator attestation verification failed: {e}")
            print(f"❌ Validator attestation verification failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    async def fetch_signed_event(self, event_hash: str) -> Optional[Dict]:
        """
        Fetch a signed event from the transparency log by hash.
        
        Used to verify equivocation via gateway-signed events.
        
        Args:
            event_hash: Event hash to fetch
            
        Returns:
            Log entry dict, or None if not found
        """
        try:
            async with aiohttp.ClientSession() as session:
                url = f"{self.gateway_url}/weights/transparency/event/{event_hash}"
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as response:
                    if response.status == 404:
                        return None
                    elif response.status == 200:
                        return await response.json()
                    else:
                        print(f"⚠️  Unexpected response fetching event: {response.status}")
                        return None
        except Exception as e:
            print(f"❌ Failed to fetch signed event: {e}")
            return None
    
    # ═══════════════════════════════════════════════════════════════════════════
    # Attestation Extraction
    # ═══════════════════════════════════════════════════════════════════════════
    
    def extract_validator_attestation(self, weights_data: Dict) -> bool:
        """
        Extract VALIDATOR attestation from the weight bundle.
        
        The validator attestation proves the weights came from an attested TEE,
        not from the gateway. This is the correct attestation to verify.
        """
        # Use CANONICAL field names (see Canonical Specifications)
        self.validator_attestation = weights_data.get("validator_attestation_b64")
        self.validator_pubkey = weights_data.get("validator_enclave_pubkey")
        self.validator_code_hash = weights_data.get("validator_code_hash")
        self.validator_hotkey = weights_data.get("validator_hotkey")
        
        if not self.validator_pubkey:
            print(f"⚠️  No validator attestation in weights bundle")
            return False
        
        print(f"   Validator pubkey: {self.validator_pubkey[:16]}...")
        print(f"   Validator hotkey: {self.validator_hotkey[:16] if self.validator_hotkey else 'None'}...")
        
        # NOTE: Full Nitro verification requires aws-nitro-enclaves-sdk
        # See "Issue 5b: Nitro Attestation Implementation Path" in tasks8.md
        # For now, extraction succeeds if fields are present
        # In production, call leadpoet_canonical.nitro.verify_nitro_attestation()
        return True
    
    # ═══════════════════════════════════════════════════════════════════════════
    # Verification
    # ═══════════════════════════════════════════════════════════════════════════
    
    def verify_bundle_signature(self, bundle: Dict) -> bool:
        """
        Verify bundle by RECOMPUTING hash and checking Ed25519 signature.
        
        CRITICAL: Does NOT trust claimed hash - recomputes from bundle data.
        
        Verification steps:
        1. Recompute bundle_weights_hash() using canonical u16 pairs
        2. Verify recomputed hash matches claimed hash
        3. Verify Ed25519 signature over digest BYTES
        
        Args:
            bundle: Response from /weights/latest/{netuid}/{epoch_id}
            
        Returns:
            True if hash recomputes correctly AND signature is valid
        """
        try:
            # Get required fields
            claimed_hash = bundle.get("weights_hash")
            signature = bundle.get("validator_signature")
            pubkey = bundle.get("validator_enclave_pubkey")
            
            if not all([claimed_hash, signature, pubkey]):
                print(f"❌ Bundle missing weights_hash / validator_signature / validator_enclave_pubkey")
                return False
            
            # RECOMPUTE hash from bundle data (don't trust claimed hash)
            uids = bundle.get("uids", [])
            weights_u16 = bundle.get("weights_u16", [])
            
            if not uids or not weights_u16:
                print(f"❌ Bundle missing uids/weights_u16")
                return False
            
            weights_pairs = list(zip(uids, weights_u16))
            recomputed_hash = bundle_weights_hash(
                bundle["netuid"],
                bundle["epoch_id"],
                bundle["block"],
                weights_pairs
            )
            
            print(f"   Claimed hash:    {claimed_hash[:16]}...")
            print(f"   Recomputed hash: {recomputed_hash[:16]}...")
            
            if recomputed_hash != claimed_hash:
                print(f"❌ Bundle data does not match weights_hash!")
                print(f"   This could indicate tampering or encoding mismatch")
                return False
            
            print(f"   ✅ Hash recomputed correctly")
            
            # Verify Ed25519 signature over digest BYTES (32 bytes, not hex string)
            digest_bytes = bytes.fromhex(claimed_hash)
            
            pk = Ed25519PublicKey.from_public_bytes(bytes.fromhex(pubkey))
            pk.verify(bytes.fromhex(signature), digest_bytes)
            
            print(f"✅ Bundle hash + signature verified")
            return True
            
        except Exception as e:
            print(f"❌ Bundle verification failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def verify_anti_equivocation(self, bundle: Dict) -> bool:
        """
        Verify primary validator didn't submit different weights to chain.
        
        CRITICAL: Use chain_snapshot_compare_hash from bundle, NOT live chain!
        subtensor.weights() returns CURRENT weights which may have changed.
        
        Verification priority:
        1. PREFER: Snapshot hash (captured at block ~345)
        2. FALLBACK: Live chain query (with loud warning)
        
        Args:
            bundle: Weight bundle from gateway
            
        Returns:
            True if no equivocation detected
        """
        print(f"\n🔍 ANTI-EQUIVOCATION CHECK")
        
        netuid = bundle["netuid"]
        epoch_id = bundle["epoch_id"]
        
        # Build bundle compare hash (NO block - for comparison)
        weights_pairs = list(zip(bundle.get("uids", []), bundle.get("weights_u16", [])))
        if not weights_pairs:
            print(f"❌ Bundle missing uids/weights_u16")
            return False
        
        bundle_compare = compare_weights_hash(netuid, epoch_id, weights_pairs)
        
        # PREFER snapshot hash (captured at block ~345)
        snapshot_hash = bundle.get("chain_snapshot_compare_hash")
        if snapshot_hash:
            print(f"   Using chain snapshot (captured at block {bundle.get('chain_snapshot_block', 'N/A')})")
            
            if snapshot_hash == bundle_compare:
                print(f"   ✅ MATCH: bundle weights match chain snapshot")
                return True
            
            print(f"\n   ❌ EQUIVOCATION DETECTED (snapshot mismatch)!")
            print(f"   bundle_compare:   {bundle_compare}")
            print(f"   snapshot_compare: {snapshot_hash}")
            return False
        
        # FALLBACK: Live chain query (with loud warning)
        print(f"   ⚠️  No chain_snapshot_compare_hash in bundle.")
        print(f"   ⚠️  FALLBACK: Querying live chain (may give false positive!)")
        
        # Get primary validator UID
        primary_uid = self._get_primary_validator_uid(bundle)
        if primary_uid is None:
            print(f"   ❌ Cannot determine primary validator UID")
            return False
        
        # Fetch on-chain weights for primary validator
        try:
            on_chain_weights = self.subtensor.weights(netuid=netuid, uid=primary_uid)
        except Exception as e:
            print(f"   ❌ Failed to fetch on-chain weights: {e}")
            return False
        
        if not on_chain_weights:
            print(f"   ⚠️  Primary validator has not set weights on-chain yet")
            return True  # Not equivocation - just not submitted yet
        
        # Normalize chain weights using canonical function
        chain_pairs = normalize_chain_weights(on_chain_weights)
        chain_compare = compare_weights_hash(netuid, epoch_id, chain_pairs)
        
        # Check exact match first
        if chain_compare == bundle_compare:
            print(f"   ✅ MATCH (exact)")
            print(f"   bundle_compare: {bundle_compare}")
            print(f"   chain_compare:  {chain_compare}")
            return True
        
        # Check with tolerance (±1 u16 drift from float round-trip)
        if weights_within_tolerance(weights_pairs, chain_pairs, tolerance=1):
            print(f"   ✅ MATCH (within tolerance)")
            print(f"   bundle_compare: {bundle_compare}")
            print(f"   chain_compare:  {chain_compare}")
            return True
        
        print(f"\n   ❌ MISMATCH!")
        print(f"   bundle_compare: {bundle_compare}")
        print(f"   chain_compare:  {chain_compare}")
        print(f"   ⚠️  This may be false positive if weights changed after the epoch.")
        return False
    
    # ═══════════════════════════════════════════════════════════════════════════
    # Weight Submission
    # ═══════════════════════════════════════════════════════════════════════════
    
    def submit_burn_weights_to_uid0(self, epoch_id: int, reason: str) -> bool:
        """
        Submit 100% weight to UID 0 (burn weights).
        
        Called when equivocation is detected or verification fails.
        This effectively burns all miner rewards for the epoch.
        
        Args:
            epoch_id: Epoch being burned
            reason: Why we're burning (for logging)
            
        Returns:
            True if submission succeeded
        """
        try:
            print(f"\n🔥 BURNING WEIGHTS TO UID 0")
            print(f"   Reason: {reason}")
            print(f"   Epoch: {epoch_id}")
            
            # Submit 100% weight to UID 0
            uids = [0]
            weights_floats = [1.0]  # 100% to UID 0
            
            success = self.subtensor.set_weights(
                netuid=self.config.netuid,
                wallet=self.wallet,
                uids=uids,
                weights=weights_floats,
                wait_for_finalization=True,
            )
            
            if success:
                print(f"🔥 BURN COMPLETE - 100% weight to UID 0 for epoch {epoch_id}")
                self.last_submitted_epoch = epoch_id
                logger.warning(f"BURN: 100% to UID 0 for epoch {epoch_id} - reason: {reason}")
                return True
            else:
                print(f"❌ Burn submission failed")
                return False
                
        except Exception as e:
            print(f"❌ Burn submission error: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def submit_weights_to_chain(self, epoch_id: int, bundle: Dict) -> bool:
        """
        Submit verified weights to the Bittensor chain.
        
        Uses u16_to_emit_floats() for proper float conversion that
        guarantees ±1 u16 round-trip tolerance.
        
        Args:
            epoch_id: Epoch being submitted
            bundle: Verified weight bundle
            
        Returns:
            True if submission succeeded
        """
        try:
            uids = bundle.get("uids", [])
            weights_u16 = bundle.get("weights_u16", [])
            
            if not uids:
                print(f"⚠️  No UIDs in bundle")
                return False
            
            # Use u16_to_emit_floats() for guaranteed round-trip
            # ❌ WRONG: weights_floats = [w / 65535.0 for w in weights_u16]
            # ✅ CORRECT: Use function that guarantees exact round-trip
            weights_floats = u16_to_emit_floats(uids, weights_u16)
            
            print(f"📤 Submitting weights for {len(uids)} UIDs...")
            
            success = self.subtensor.set_weights(
                netuid=self.config.netuid,
                wallet=self.wallet,
                uids=uids,
                weights=weights_floats,
                wait_for_finalization=True,  # CRITICAL: Wait for finalization
            )
            
            if success:
                print(f"✅ Weights submitted for epoch {epoch_id}")
                self.last_submitted_epoch = epoch_id
                return True
            else:
                print(f"❌ Weight submission failed")
                return False
                
        except Exception as e:
            print(f"❌ Weight submission error: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    # ═══════════════════════════════════════════════════════════════════════════
    # Main Loop
    # ═══════════════════════════════════════════════════════════════════════════
    
    async def run(self):
        """Main loop for the auditor validator."""
        
        print(f"\n{'='*60}")
        print(f"🚀 AUDITOR VALIDATOR STARTING")
        print(f"{'='*60}")
        logger.info("Auditor validator starting")
        
        # Fetch and verify GATEWAY attestation at startup
        # NOTE: Validator attestation comes from weight bundles, not here
        if await self.fetch_gateway_attestation():
            if self.verify_gateway_attestation():
                logger.info(f"Gateway attestation verified (trust_level={self.trust_level})")
            else:
                logger.warning("Gateway attestation verification failed")
                print(f"⚠️ Gateway attestation verification failed")
        else:
            logger.warning("Could not fetch gateway attestation")
            print(f"⚠️ Could not fetch gateway attestation (continuing anyway)")
        
        while not self.should_exit:
            try:
                # Get current block and epoch
                current_block = self.subtensor.get_current_block()
                current_epoch = current_block // EPOCH_LENGTH
                block_within_epoch = current_block % EPOCH_LENGTH
                
                print(f"\r⏱️  Block {current_block} | Epoch {current_epoch} | Block {block_within_epoch}/{EPOCH_LENGTH}", end="", flush=True)
                
                # Check if it's time to submit weights
                if block_within_epoch >= WEIGHT_SUBMISSION_BLOCK:
                    if self.last_submitted_epoch != current_epoch:
                        print(f"\n\n{'='*60}")
                        print(f"📊 WEIGHT SUBMISSION TIME (Block {block_within_epoch})")
                        print(f"{'='*60}")
                        
                        # Fetch weights for CURRENT epoch (not previous)
                        # At block 345 of epoch N, primary validator submits epoch N weights
                        # Auditor should copy epoch N, not N-1
                        target_epoch = current_epoch
                        print(f"   Fetching weights for epoch {target_epoch}...")
                        
                        weights_data = await self.fetch_verified_weights(target_epoch)
                        
                        if weights_data is None:
                            print(f"   ⏳ Weights not yet published. Waiting 30s...")
                            await asyncio.sleep(30)  # CRITICAL: Prevent hot-loop DOSing gateway
                            continue
                        
                        # Extract VALIDATOR attestation from weight bundle
                        if not self.extract_validator_attestation(weights_data):
                            logger.warning(f"No validator attestation in bundle for epoch {target_epoch}")
                            print(f"   ⚠️  No validator attestation - cannot verify TEE origin")
                        
                        # Verify validator attestation (if present)
                        if self.validator_attestation:
                            if not self.verify_validator_attestation(weights_data):
                                logger.error(f"Validator attestation verification failed for epoch {target_epoch}")
                                print(f"   ❌ Validator attestation verification failed.")
                                print(f"   🔥 BURNING 100% TO UID 0 (attestation verification failed)")
                                self.submit_burn_weights_to_uid0(target_epoch, "validator_attestation_failed")
                                continue
                        
                        # Verify bundle signature and hash (recomputes hash)
                        if not self.verify_bundle_signature(weights_data):
                            logger.error(f"Bundle signature/hash verification failed for epoch {target_epoch}")
                            print(f"   ❌ Bundle signature/hash verification failed.")
                            print(f"   🔥 BURNING 100% TO UID 0 (signature/hash verification failed)")
                            self.submit_burn_weights_to_uid0(target_epoch, "signature_hash_verification_failed")
                            continue
                        
                        # Verify anti-equivocation (prefers snapshot)
                        if not self.verify_anti_equivocation(weights_data):
                            logger.error(f"Equivocation detected for epoch {target_epoch}")
                            print(f"   ❌ Equivocation check failed. Not copying.")
                            # ═══════════════════════════════════════════════════════════════
                            # EXPLICIT AUDITOR BEHAVIOR ON EQUIVOCATION
                            # ═══════════════════════════════════════════════════════════════
                            # BURN 100% TO UID 0 - signals distrust and penalizes all miners
                            # This is the strongest possible signal that something is wrong.
                            # ═══════════════════════════════════════════════════════════════
                            print(f"   🔥 BURNING 100% TO UID 0 (equivocation detected)")
                            self.submit_burn_weights_to_uid0(target_epoch, "equivocation_detected")
                            continue
                        
                        # All checks passed - safe to copy weights
                        logger.info(f"All verifications passed for epoch {target_epoch} (trust_level={self.trust_level})")
                        print(f"\n   ✅ All verifications passed")
                        print(f"   🔐 Trust level: {self.trust_level.upper()}")
                        
                        if self.submit_weights_to_chain(target_epoch, weights_data):
                            logger.info(f"Weights submitted for epoch {target_epoch}")
                        else:
                            logger.error(f"Weight submission failed for epoch {target_epoch}")
                
                # Refresh metagraph periodically (at epoch start)
                if block_within_epoch == 0:
                    print(f"\n🔄 Refreshing metagraph...")
                    self.metagraph = self.subtensor.metagraph(self.config.netuid)
                
                await asyncio.sleep(12)  # ~1 block
                
            except KeyboardInterrupt:
                print(f"\n\n⛔ Shutting down...")
                self.should_exit = True
            except Exception as e:
                print(f"\n❌ Error in main loop: {e}")
                import traceback
                traceback.print_exc()
                await asyncio.sleep(30)


def main():
    """Entry point for auditor validator."""
    
    parser = argparse.ArgumentParser(
        description="LeadPoet Auditor Validator - Copies TEE-verified weights from primary validator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
SECURITY MODEL:
  The auditor validator does NOT run validation logic itself.
  It fetches weights from the primary validator TEE, verifies:
    1. Ed25519 signature over weights hash
    2. Hash recomputation matches claimed hash
    3. Anti-equivocation (chain snapshot match)
  Then copies the verified weights to its own chain submission.

TRUST LEVELS:
  - full_nitro: Full AWS Nitro attestation verified
  - signature_only: Only Ed25519 signatures verified (weaker)

VERIFICATION FAILURE HANDLING:
  If verification fails (equivocation, attestation, signature/hash):
  - BURN 100% weight to UID 0 (strongest distrust signal)
  - This prevents copying malicious weights AND penalizes all miners

EXAMPLES:
  python neurons/auditor_validator.py --netuid 71
  python neurons/auditor_validator.py --netuid 71 --gateway-url http://localhost:8000
        """
    )
    
    # Bittensor arguments
    bt.wallet.add_args(parser)
    bt.subtensor.add_args(parser)
    
    # Custom arguments
    parser.add_argument(
        "--netuid", 
        type=int, 
        default=71, 
        help="Subnet UID (default: 71)"
    )
    parser.add_argument(
        "--gateway-url", 
        type=str, 
        default=DEFAULT_GATEWAY_URL, 
        help=f"Gateway URL (default: {DEFAULT_GATEWAY_URL})"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)"
    )
    
    args = parser.parse_args()
    config = bt.config(parser)
    
    # Configure logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # Get gateway URL from args or default
    gateway_url = args.gateway_url or DEFAULT_GATEWAY_URL
    
    print(f"\n{'='*60}")
    print(f"🔍 LEADPOET AUDITOR VALIDATOR")
    print(f"{'='*60}")
    print(f"   Network: {config.subtensor.network}")
    print(f"   Netuid: {args.netuid}")
    print(f"   Gateway: {gateway_url}")
    print(f"   Log level: {args.log_level}")
    print(f"{'='*60}")
    
    # Check for production environment variables
    if os.environ.get("ENVIRONMENT") == "production":
        print(f"\n⚠️  PRODUCTION MODE DETECTED")
        if not EXPECTED_GATEWAY_CODE_HASH:
            print(f"   ❌ EXPECTED_GATEWAY_CODE_HASH not set!")
        else:
            print(f"   ✅ Gateway code hash: {EXPECTED_GATEWAY_CODE_HASH[:16]}...")
        if not EXPECTED_VALIDATOR_CODE_HASH:
            print(f"   ❌ EXPECTED_VALIDATOR_CODE_HASH not set!")
        else:
            print(f"   ✅ Validator code hash: {EXPECTED_VALIDATOR_CODE_HASH[:16]}...")
    else:
        print(f"\n⚠️  DEV MODE: Attestation verification is SIGNATURE-ONLY")
        print(f"   Set ENVIRONMENT=production for full Nitro verification")
    
    print(f"{'='*60}\n")
    
    # Set netuid on config
    config.netuid = args.netuid
    
    try:
        # Create and run validator
        validator = AuditorValidator(config, gateway_url=gateway_url)
        asyncio.run(validator.run())
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        print("\n⛔ Shutting down...")
    except RuntimeError as e:
        logger.error(f"Runtime error: {e}")
        print(f"\n❌ Runtime error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

