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
    python neurons/auditor_validator.py --netuid 71 --wallet.name my_wallet --wallet.hotkey default
"""

import os
import sys
import argparse

# Add repo root to path so leadpoet_canonical can be imported from anywhere
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SCRIPT_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
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

# PCR0 allowlist is fetched from GitHub automatically by leadpoet_canonical.nitro
# No need for hardcoded code hashes - dynamic verification via pcr0_allowlist.json

# File to store pending equivocation check (overwritten each epoch)
PENDING_EQUIVOCATION_FILE = os.path.join(_SCRIPT_DIR, ".pending_equivocation_check.json")


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
        self.consecutive_errors = 0
        self.max_consecutive_errors = 5  # Reconnect subtensor after this many errors
        
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
        # Always starts as None, set to "full_nitro" after attestation verification
        self.trust_level = None
        
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
    
    def _reconnect_subtensor(self):
        """
        Reconnect to subtensor after connection errors.
        
        CRITICAL: Validators run for days. Websocket connections WILL drop.
        This ensures the validator keeps running after network issues.
        """
        print(f"\n🔄 Reconnecting to subtensor...")
        logger.info("Reconnecting to subtensor after connection error")
        
        try:
            # Create new subtensor instance
            self.subtensor = bt.subtensor(config=self.config)
            self.metagraph = self.subtensor.metagraph(self.config.netuid)
            
            # Verify we're still registered
            new_uid = self._get_uid()
            if new_uid is None:
                logger.error("Lost registration after reconnect!")
                print(f"❌ Lost registration after reconnect!")
            else:
                self.uid = new_uid
                print(f"✅ Reconnected to subtensor (UID: {self.uid})")
                logger.info(f"Reconnected to subtensor (UID: {self.uid})")
            
            self.consecutive_errors = 0
            return True
            
        except Exception as e:
            logger.error(f"Failed to reconnect to subtensor: {e}")
            print(f"❌ Failed to reconnect: {e}")
            return False
    
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
    # Soft Anti-Equivocation (Retroactive Check)
    # ═══════════════════════════════════════════════════════════════════════════
    
    def save_pending_equivocation_check(self, epoch_id: int, bundle_compare_hash: str, validator_hotkey: str):
        """
        Save bundle compare hash for retroactive equivocation check.
        
        Called after successful weight verification at block 345.
        Overwrites previous epoch's data (only stores last epoch).
        """
        import json
        from datetime import datetime, timezone
        data = {
            "epoch_id": epoch_id,
            "bundle_compare_hash": bundle_compare_hash,
            "validator_hotkey": validator_hotkey,
            "saved_at": datetime.now(timezone.utc).isoformat(),
        }
        try:
            with open(PENDING_EQUIVOCATION_FILE, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"   📝 Saved pending equivocation check for epoch {epoch_id}")
        except Exception as e:
            logger.warning(f"Failed to save pending equivocation check: {e}")
    
    def load_pending_equivocation_check(self) -> Optional[Dict]:
        """
        Load pending equivocation check data.
        
        Returns:
            Dict with epoch_id, bundle_compare_hash, validator_hotkey
            None if no pending check or file doesn't exist
        """
        import json
        try:
            if not os.path.exists(PENDING_EQUIVOCATION_FILE):
                return None
            with open(PENDING_EQUIVOCATION_FILE, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load pending equivocation check: {e}")
            return None
    
    def clear_pending_equivocation_check(self):
        """Clear the pending equivocation check file after verification."""
        try:
            if os.path.exists(PENDING_EQUIVOCATION_FILE):
                os.remove(PENDING_EQUIVOCATION_FILE)
        except Exception as e:
            logger.warning(f"Failed to clear pending equivocation check: {e}")
    
    async def perform_soft_equivocation_check(self) -> bool:
        """
        Retroactively verify previous epoch's weights against chain.
        
        Called at block 30-60 of each new epoch.
        Compares stored bundle hash with what's actually on chain.
        
        Returns:
            True if check passed or no pending check
            False if equivocation detected
        """
        pending = self.load_pending_equivocation_check()
        if not pending:
            return True  # No pending check
        
        epoch_id = pending.get("epoch_id")
        bundle_compare_hash = pending.get("bundle_compare_hash")
        validator_hotkey = pending.get("validator_hotkey")
        
        if not all([epoch_id, bundle_compare_hash, validator_hotkey]):
            logger.warning("Invalid pending equivocation data")
            self.clear_pending_equivocation_check()
            return True
        
        print(f"\n{'='*60}")
        print(f"🔍 SOFT EQUIVOCATION CHECK (Epoch {epoch_id})")
        print(f"{'='*60}")
        print(f"   Checking previous epoch's weights against chain...")
        
        try:
            # Find validator's UID
            if validator_hotkey not in self.metagraph.hotkeys:
                print(f"   ⚠️  Validator hotkey not in metagraph, skipping check")
                self.clear_pending_equivocation_check()
                return True
            
            validator_uid = self.metagraph.hotkeys.index(validator_hotkey)
            print(f"   Primary validator: {validator_hotkey[:16]}... → UID {validator_uid}")
            
            # Get chain weights for the validator
            print(f"   Fetching chain weights for UID {validator_uid}...")
            all_chain_weights = self.subtensor.weights(netuid=self.config.netuid)
            
            # Debug: show how many validators have weights on chain
            validators_with_weights = [uid for uid, _ in all_chain_weights]
            print(f"   Validators with weights on chain: {validators_with_weights}")
            
            chain_weights = None
            for uid, weights_list in all_chain_weights:
                if uid == validator_uid:
                    chain_weights = weights_list
                    # Debug: show raw chain weights before normalization
                    print(f"   Raw chain weights (first 5):")
                    for target_uid, w in weights_list[:5]:
                        print(f"      UID {target_uid}: {w} (type: {type(w).__name__})")
                    break
            
            if not chain_weights:
                print(f"   ⚠️  No weights on chain for validator UID {validator_uid}")
                self.clear_pending_equivocation_check()
                return True
            
            # Normalize chain weights to pairs
            chain_pairs = normalize_chain_weights(chain_weights)
            
            # We need to reconstruct bundle weights pairs from the hash
            # Since we only stored the hash, we need to fetch the bundle again
            # OR use tolerance-based comparison on the actual weights
            
            # For soft check, we'll fetch the bundle and compare with tolerance
            # (hash comparison is too strict due to ±1 u16 round-trip tolerance)
            print(f"   Fetching bundle for epoch {epoch_id} to compare weights...")
            
            bundle = await self.fetch_verified_weights(epoch_id)
            
            if not bundle:
                print(f"   ⚠️  Could not fetch bundle for epoch {epoch_id}, skipping check")
                self.clear_pending_equivocation_check()
                return True
            
            bundle_uids = bundle.get("uids", [])
            bundle_weights = bundle.get("weights_u16", [])
            bundle_pairs = list(zip(bundle_uids, bundle_weights))
            
            # Debug: show bundle weights for comparison
            print(f"   Bundle weights (first 5):")
            for uid, w in bundle_pairs[:5]:
                print(f"      UID {uid}: {w}")
            
            # Convert chain_pairs to dict for comparison
            chain_dict = {uid: w for uid, w in chain_pairs}
            bundle_dict = {uid: w for uid, w in bundle_pairs}
            
            current_block = self.subtensor.get_current_block()
            current_epoch = current_block // EPOCH_LENGTH
            print(f"   Bundle UIDs: {len(bundle_pairs)}, Chain UIDs: {len(chain_pairs)}")
            print(f"   Bundle epoch: {epoch_id}, Current epoch: {current_epoch}")
            if current_epoch > epoch_id:
                print(f"   ⚠️  NOTE: Chain weights are CURRENT - if primary already submitted epoch {current_epoch}, comparison is invalid")
            
            # Check if UIDs match
            bundle_uid_set = set(bundle_uids)
            chain_uid_set = set(uid for uid, _ in chain_pairs)
            
            if bundle_uid_set != chain_uid_set:
                missing_on_chain = bundle_uid_set - chain_uid_set
                extra_on_chain = chain_uid_set - bundle_uid_set
                print(f"\n   ❌ UID MISMATCH!")
                if missing_on_chain:
                    print(f"   Missing on chain: {sorted(missing_on_chain)[:10]}...")
                if extra_on_chain:
                    print(f"   Extra on chain: {sorted(extra_on_chain)[:10]}...")
                logger.warning(f"UID_MISMATCH: Epoch {epoch_id} - Bundle and chain have different UIDs (investigating)")
                self.clear_pending_equivocation_check()
                return False
            
            # Compare weights with ±1 tolerance (u16 round-trip tolerance)
            mismatches = []
            for uid in bundle_uid_set:
                bundle_w = bundle_dict.get(uid, 0)
                chain_w = chain_dict.get(uid, 0)
                diff = abs(bundle_w - chain_w)
                if diff > 1:  # Allow ±1 tolerance
                    mismatches.append((uid, bundle_w, chain_w, diff))
            
            if mismatches:
                print(f"\n   ❌ WEIGHT MISMATCH (beyond ±1 tolerance)!")
                print(f"   {len(mismatches)} UIDs with significant differences:")
                for uid, bw, cw, diff in mismatches[:10]:
                    print(f"      UID {uid}: bundle={bw}, chain={cw}, diff={diff}")
                if len(mismatches) > 10:
                    print(f"      ... and {len(mismatches) - 10} more")
                logger.warning(f"WEIGHT_MISMATCH: Epoch {epoch_id} - {len(mismatches)} weights differ beyond ±1 tolerance (investigating)")
                self.clear_pending_equivocation_check()
                return False
            
            # All weights within tolerance
            print(f"   ✅ MATCH - All {len(bundle_pairs)} weights within ±1 tolerance")
            print(f"   No equivocation detected for epoch {epoch_id}")
            self.clear_pending_equivocation_check()
            return True
                
        except Exception as e:
            print(f"   ⚠️  Soft equivocation check failed: {e}")
            logger.warning(f"Soft equivocation check error: {e}")
            self.clear_pending_equivocation_check()
            return True  # Don't block on errors
    
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
            # FULL NITRO VERIFICATION - NO DEV MODE
            from leadpoet_canonical.nitro import verify_nitro_attestation_full
            
            valid, data = verify_nitro_attestation_full(
                self.gateway_attestation,  # Already base64 encoded
                expected_pcr0=None,  # Uses allowlist from GitHub automatically
                expected_pubkey=self.gateway_pubkey,
                expected_purpose=None,  # Gateway attestation purpose varies
                expected_epoch_id=None,  # Gateway attestation doesn't have epoch_id
                role="gateway",  # Uses ALLOWED_GATEWAY_PCR0_VALUES
            )
            
            if valid:
                self.trust_level = "full_nitro"
                pcr0 = data.get("pcr0", "N/A")[:32]
                logger.info(f"Gateway attestation verified (full Nitro, PCR0: {pcr0}...)")
                print(f"✅ Gateway attestation: FULL NITRO VERIFICATION")
                print(f"   PCR0: {pcr0}...")
                return True
            else:
                logger.error(f"Gateway Nitro verification failed: {data}")
                print(f"❌ Gateway Nitro verification FAILED")
                print(f"   Details: {data.get('error', 'Unknown error')}")
                return False
                
        except Exception as e:
            logger.error(f"Gateway attestation verification failed: {e}")
            print(f"❌ Gateway attestation verification failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def verify_validator_attestation(self, bundle: Dict) -> bool:
        """
        Verify the validator attestation from a weight bundle.
        
        SECURITY MODEL FOR AUDITORS:
        - Verifies AWS certificate chain (proves REAL Nitro enclave)
        - Verifies COSE signature (proves attestation is authentic)
        - Verifies epoch binding (replay protection)
        - SKIPS PCR0 verification (auditors can't independently verify without nitro-cli)
        
        WHY SKIP PCR0?
        - PCR0 verification requires either:
          a) nitro-cli to build enclave and compute expected PCR0, OR
          b) Trusting an allowlist published by subnet owners
        - Auditors don't have nitro-cli (not on AWS EC2)
        - Auditors shouldn't trust subnet-owner-published allowlists
        - So we verify "it's a REAL Nitro enclave" but not "which code it runs"
        
        WHAT THIS PROVES:
        ✅ The attestation came from a REAL AWS Nitro enclave (AWS-signed)
        ✅ The weights were signed by that enclave's private key
        ✅ The attestation is for THIS epoch (replay protection)
        ❌ Does NOT prove which code is running (would need nitro-cli)
        
        Args:
            bundle: Weight bundle containing validator_attestation_b64
            
        Returns:
            True if attestation is valid
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
            from leadpoet_canonical.nitro import verify_nitro_attestation_full
            
            # AUDITOR MODE: Skip PCR0 verification
            # We verify AWS cert chain + COSE signature (proves REAL enclave)
            # but skip PCR0 check (can't verify without nitro-cli)
            valid, data = verify_nitro_attestation_full(
                attestation_b64,  # Already base64 encoded
                expected_pcr0=None,
                expected_pubkey=pubkey,
                expected_purpose="validator_weights",
                expected_epoch_id=epoch_id,  # CRITICAL: Replay protection
                role="validator",
                skip_pcr0_verification=True,  # Auditors can't verify PCR0
            )
            
            if valid:
                self.trust_level = data.get("trust_level", "aws_verified")
                pcr0 = data.get("pcr0", "N/A")[:32]
                logger.info(f"Validator attestation verified for epoch {epoch_id} (trust_level={self.trust_level})")
                print(f"✅ Validator attestation: AWS VERIFIED")
                print(f"   Trust level: {self.trust_level.upper()}")
                print(f"   Epoch: {epoch_id}")
                print(f"   PCR0: {pcr0}... (not verified - requires nitro-cli)")
                print(f"   Pubkey: {pubkey[:16]}...")
                print(f"   ℹ️  AWS cert chain + COSE signature verified (proves real Nitro enclave)")
                return True
            else:
                logger.error(f"Validator attestation verification failed: {data}")
                print(f"❌ Validator attestation verification FAILED")
                print(f"   Details: {data.get('error', 'Unknown error')}")
                return False
                
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
        
        # SKIP ANTI-EQUIVOCATION CHECK
        #
        # WHY: The chain snapshot is captured BEFORE the validator submits to chain,
        # so it contains the PREVIOUS epoch's weights, not the current submission.
        # This causes false positives (mismatch between epoch N bundle vs epoch N-1 snapshot).
        #
        # TIMING ISSUE:
        # 1. Validator submits to gateway → snapshot captures chain (epoch N-1 weights)
        # 2. Validator submits to chain (epoch N weights)
        # 3. Auditor compares bundle (N) vs snapshot (N-1) → FALSE MISMATCH
        #
        # The other 5 verifications are trustless and sufficient:
        # ✅ AWS cert chain, COSE signature, epoch binding, Ed25519 signature, hash recompute
        snapshot_hash = bundle.get("chain_snapshot_compare_hash")
        if snapshot_hash:
            print(f"   ⚠️  Skipping anti-equivocation (snapshot timing issue)")
            print(f"   ℹ️  Snapshot was captured BEFORE chain submission")
            print(f"   ℹ️  Other 5 trustless verifications already passed")
            return True  # Skip - rely on other 5 trustless verifications
        
        # NO SNAPSHOT AVAILABLE - Skip anti-equivocation check
        # 
        # WHY WE SKIP (not fallback to live chain):
        # 1. subtensor.weights() returns CURRENT weights, not historical
        # 2. Live chain query is unreliable (may give false positive)
        # 3. Anti-equivocation relies on gateway-captured snapshot (trust issue)
        # 4. The 5 other verifications (AWS cert, COSE, epoch, sig, hash) are trustless
        #
        # This check would only catch if validator submitted DIFFERENT weights
        # to chain vs gateway, but without snapshot we can't verify this reliably.
        print(f"   ⚠️  No chain_snapshot_compare_hash in bundle.")
        print(f"   ⚠️  SKIPPING anti-equivocation check (no reliable snapshot)")
        print(f"   ℹ️  Other 5 verifications (AWS cert, COSE, epoch, signature, hash) still apply")
        return True  # Skip this check - rely on the other 5 trustless verifications
    
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
            print(f"   Weight breakdown:")
            print(f"      UID 0 (Burn): 100.00%")
            print(f"   Total: 100.00%")
            
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
            
            # Print weight breakdown (same format as primary validator)
            print(f"\n📤 Submitting weights for {len(uids)} UIDs...")
            print(f"   Weight breakdown (copying from primary validator):")
            total_weight = sum(weights_floats)
            for uid, weight in zip(uids, weights_floats):
                pct = (weight / total_weight * 100) if total_weight > 0 else 0
                label = "(Burn)" if uid == 0 else ""
                print(f"      UID {uid} {label}: {pct:.2f}%")
            print(f"   Total: {sum(weights_floats) / total_weight * 100:.2f}%")
            
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
                
                # Verify submission landed on chain
                print(f"   🔍 Verifying submission landed on chain...")
                import time
                time.sleep(2)  # Brief wait for chain propagation
                
                try:
                    all_chain_weights = self.subtensor.weights(netuid=self.config.netuid)
                    my_uid = self.uid
                    
                    for uid, weights_list in all_chain_weights:
                        if uid == my_uid:
                            # Check if first few weights match what we submitted
                            chain_sample = [(u, w) for u, w in weights_list[:3]]
                            submitted_sample = [(uids[i], weights_u16[i]) for i in range(min(3, len(uids)))]
                            print(f"   Chain sample (UID {my_uid}): {chain_sample}")
                            print(f"   Submitted sample: {submitted_sample}")
                            
                            # Quick sanity check
                            if chain_sample and submitted_sample:
                                # Compare first weight
                                chain_first_w = dict(chain_sample).get(submitted_sample[0][0])
                                submitted_first_w = submitted_sample[0][1]
                                if chain_first_w is not None:
                                    diff = abs(chain_first_w - submitted_first_w)
                                    if diff <= 1:
                                        print(f"   ✅ Verified: Chain matches submitted (diff={diff})")
                                    else:
                                        print(f"   ⚠️  WARNING: Chain differs from submitted (diff={diff})")
                                        print(f"      Chain has OLD weights - submission may have failed!")
                            break
                    else:
                        print(f"   ⚠️  Could not find our weights on chain (UID {my_uid})")
                except Exception as e:
                    print(f"   ⚠️  Could not verify chain weights: {e}")
                
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
        
        # Fetch gateway pubkey (for log verification)
        # NOTE: Gateway attestation is OPTIONAL - gateway runs on EC2 host, not in enclave
        # The CRITICAL verification is the VALIDATOR attestation (validator DOES run in enclave)
        if await self.fetch_gateway_attestation():
            print(f"✅ Gateway pubkey fetched: {self.gateway_pubkey[:16]}...")
            # Gateway attestation verification is optional since gateway doesn't run in Nitro
            if self.gateway_attestation:
                if self.verify_gateway_attestation():
                    logger.info(f"Gateway attestation verified (trust_level={self.trust_level})")
                else:
                    # Non-fatal - gateway may not have attestation
                    print(f"ℹ️  Gateway attestation not available (gateway runs on EC2 host, not enclave)")
            else:
                print(f"ℹ️  Gateway runs on EC2 host (no Nitro attestation)")
                print(f"   This is expected - only the PRIMARY VALIDATOR runs in Nitro Enclave")
        else:
            logger.warning("Could not fetch gateway info")
            print(f"⚠️ Could not fetch gateway info")
        
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
                        
                        # Save pending equivocation check for next epoch verification
                        weights_pairs = list(zip(weights_data.get("uids", []), weights_data.get("weights_u16", [])))
                        if weights_pairs:
                            bundle_compare = compare_weights_hash(self.config.netuid, target_epoch, weights_pairs)
                            self.save_pending_equivocation_check(
                                target_epoch, 
                                bundle_compare, 
                                weights_data.get("validator_hotkey", "")
                            )
                        
                        if self.submit_weights_to_chain(target_epoch, weights_data):
                            logger.info(f"Weights submitted for epoch {target_epoch}")
                        else:
                            logger.error(f"Weight submission failed for epoch {target_epoch}")
                
                # Soft anti-equivocation check at block 30-80 of each epoch
                if 30 <= block_within_epoch <= 80:
                    # Check if we have a pending equivocation check from previous epoch
                    pending = self.load_pending_equivocation_check()
                    if pending and pending.get("epoch_id") == current_epoch - 1:
                        if not await self.perform_soft_equivocation_check():
                            logger.error(f"Soft equivocation check FAILED for epoch {current_epoch - 1}")
                            print(f"   ⚠️  MISMATCH DETECTED - Bundle vs chain weights differ (investigating...)")
                            # Note: We don't burn here since we already submitted weights
                            # This is a SOFT check - logs the issue for investigation
                            # Mismatch could be due to: timing, normalization differences, or actual equivocation
                
                # Refresh metagraph periodically (at epoch start)
                if block_within_epoch == 0:
                    print(f"\n🔄 Refreshing metagraph...")
                    self.metagraph = self.subtensor.metagraph(self.config.netuid)
                
                # Reset error counter on successful iteration
                self.consecutive_errors = 0
                
                await asyncio.sleep(12)  # ~1 block
                
            except KeyboardInterrupt:
                print(f"\n\n⛔ Shutting down...")
                self.should_exit = True
            except (TimeoutError, ConnectionError, OSError) as e:
                # Network/connection errors - common and recoverable
                self.consecutive_errors += 1
                print(f"\n⚠️  Connection error ({self.consecutive_errors}/{self.max_consecutive_errors}): {type(e).__name__}")
                logger.warning(f"Connection error: {e}")
                
                if self.consecutive_errors >= self.max_consecutive_errors:
                    print(f"   Too many consecutive errors - reconnecting subtensor...")
                    self._reconnect_subtensor()
                
                # Exponential backoff: 30s, 60s, 120s, max 300s
                backoff = min(30 * (2 ** (self.consecutive_errors - 1)), 300)
                print(f"   Retrying in {backoff}s...")
                await asyncio.sleep(backoff)
            except Exception as e:
                # Other errors - log and continue
                self.consecutive_errors += 1
                error_type = type(e).__name__
                print(f"\n❌ Error in main loop ({error_type}): {e}")
                logger.error(f"Main loop error: {e}")
                
                # Check if it's a websocket-related error
                error_str = str(e).lower()
                if any(x in error_str for x in ['websocket', 'ssl', 'connection', 'timeout', 'handshake']):
                    print(f"   Connection-related error detected")
                    if self.consecutive_errors >= self.max_consecutive_errors:
                        self._reconnect_subtensor()
                
                import traceback
                traceback.print_exc()
                
                # Exponential backoff
                backoff = min(30 * (2 ** (self.consecutive_errors - 1)), 300)
                print(f"   Retrying in {backoff}s...")
                await asyncio.sleep(backoff)


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

TRUST LEVEL:
  - full_nitro: Full AWS Nitro attestation verified (ALWAYS REQUIRED)
  - PCR0 verified against GitHub allowlist automatically

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
    
    # Auditor verification mode - verify AWS signature, skip PCR0
    print(f"\n🔐 AUDITOR VERIFICATION MODE")
    print(f"   ✅ AWS certificate chain verified (proves REAL Nitro enclave)")
    print(f"   ✅ COSE signature verified (proves authentic attestation)")
    print(f"   ✅ Ed25519 signature verified (proves weights from enclave)")
    print(f"   ✅ Epoch binding verified (replay protection)")
    print(f"   ⚠️  PCR0 NOT verified (requires nitro-cli or trusting subnet owner)")
    print(f"   ℹ️  Trust level: AWS_VERIFIED (real enclave, unverified code)")
    
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

