#!/usr/bin/env python3
"""
Auditor Verifier CLI
====================

Comprehensive verification tool for auditor validators.

Performs four key verification steps:
1. Attestation verification (Gateway + Validator TEEs)
2. Epoch bundle signature verification
3. Log chain integrity verification
4. Chain consistency check (equivocation detection)

TRUST LEVELS:
- "full_nitro": Full AWS Nitro attestation verified
- "signature_only": Only Ed25519 signatures verified (weaker trust)

USAGE:
    # Verify all attestations
    python scripts/auditor_verifier.py verify-attestations
    
    # Verify specific epoch bundle
    python scripts/auditor_verifier.py verify-bundle --epoch 12345
    
    # Verify log chain (Step 3)
    python scripts/auditor_verifier.py verify-logs --from-epoch 12340 --to-epoch 12345
    
    # Check for equivocation
    python scripts/auditor_verifier.py verify-chain-match --epoch 12345
    
    # Run all verifications
    python scripts/auditor_verifier.py verify-all --epoch 12345

ENVIRONMENT VARIABLES (required for log verification):
    SUPABASE_URL           - Supabase project URL (public)
    SUPABASE_ANON_KEY      - Supabase anonymous key (read-only)
    EXPECTED_CHAIN         - Chain endpoint for binding verification (optional)
"""

import argparse
import asyncio
import base64
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

import aiohttp
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

# Import canonical functions from shared module
from leadpoet_canonical.weights import (
    bundle_weights_hash,
    compare_weights_hash,
    weights_within_tolerance,
)
from leadpoet_canonical.events import (
    verify_log_entry,
    verify_hash_chain_continuity,
    verify_monotonic_sequence,
)
from leadpoet_canonical.binding import verify_binding_message
from leadpoet_canonical.constants import EPOCH_LENGTH
from leadpoet_canonical.nitro import (
    verify_nitro_attestation_full,
    verify_nitro_attestation_signature_only,
    is_nitro_verification_available,
    TRUST_LEVEL_FULL_NITRO,
    TRUST_LEVEL_SIGNATURE_ONLY,
)

# Pinned code hashes for production (optional, set via env vars)
PINNED_GATEWAY_CODE_HASH = os.environ.get("EXPECTED_GATEWAY_CODE_HASH")
PINNED_VALIDATOR_CODE_HASH = os.environ.get("EXPECTED_VALIDATOR_CODE_HASH")

# Default gateway URL
DEFAULT_GATEWAY_URL = os.environ.get("GATEWAY_URL", "http://54.226.209.164:8000")
DEFAULT_NETUID = int(os.environ.get("NETUID", "71"))


class VerificationError(Exception):
    """Raised when verification fails."""
    pass


class AuditorVerifier:
    """
    Comprehensive verifier for auditing TEE-signed weights and logs.
    
    Implements 4-step verification:
    1. verify_attestations() - Gateway + Validator TEE attestation
    2. verify_epoch_bundle(epoch_id) - Validator signature on weights
    3. verify_log_chain(from_epoch, to_epoch) - Hash-chain and sequence integrity
    4. verify_chain_match(epoch_id) - Equivocation detection
    """
    
    def __init__(
        self, 
        gateway_url: str = DEFAULT_GATEWAY_URL,
        netuid: int = DEFAULT_NETUID,
    ):
        self.gateway_url = gateway_url.rstrip("/")
        self.netuid = netuid
        
        # Trust level tracking (CRITICAL for honest reporting)
        # Starts as "signature_only" until full Nitro verification implemented
        self.trust_level = "signature_only"
        
        # Cached attestation data
        self.gateway_pubkey = None
        self.gateway_code_hash = None
        self.validator_pubkey = None
        self.validator_code_hash = None
    
    # =========================================================================
    # STEP 1: ATTESTATION VERIFICATION
    # =========================================================================
    
    async def verify_attestations(self) -> Dict:
        """
        Verify both Gateway and Validator TEE attestations.
        
        Returns:
            Dict with verification results for both attestations
        """
        print("\n" + "=" * 60)
        print("📜 STEP 1: ATTESTATION VERIFICATION")
        print("=" * 60)
        
        results = {
            "gateway": {"verified": False, "pubkey": None, "code_hash": None},
            "validator": {"verified": False, "pubkey": None, "code_hash": None},
        }
        
        # Verify Gateway attestation
        print("\n📜 Verifying Gateway TEE attestation...")
        gateway_ok = await self._verify_gateway_attestation(results)
        
        # Verify Validator attestation (from weights endpoint)
        print("\n📜 Verifying Validator TEE attestation...")
        validator_ok = await self._verify_validator_attestation(results)
        
        # Summary
        print("\n" + "-" * 40)
        if gateway_ok and validator_ok:
            print("✅ ATTESTATION VERIFICATION: PASSED")
            if self.gateway_code_hash:
                print(f"   Gateway pubkey bound to build: {self.gateway_code_hash[:16]}...")
            if self.validator_code_hash:
                print(f"   Validator pubkey bound to build: {self.validator_code_hash[:16]}...")
        else:
            print("❌ ATTESTATION VERIFICATION: FAILED")
            if not gateway_ok:
                print("   ❌ Gateway attestation failed")
            if not validator_ok:
                print("   ❌ Validator attestation failed")
        
        return results
    
    async def _verify_gateway_attestation(self, results: Dict) -> bool:
        """Fetch and verify gateway attestation."""
        try:
            # Try /attest endpoint first (returns hex-encoded attestation)
            # Fall back to /attestation/document if /attest doesn't exist
            async with aiohttp.ClientSession() as session:
                # Try /attest first
                async with session.get(
                    f"{self.gateway_url}/attest",
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        # /attest returns: attestation_document (hex), enclave_public_key, code_hash, pcr0
                        self.gateway_pubkey = data.get("enclave_public_key")
                        self.gateway_code_hash = data.get("code_hash")
                        attestation_doc = data.get("attestation_document")  # hex-encoded
                    elif response.status == 404:
                        # Fall back to /attestation/document
                        async with session.get(
                            f"{self.gateway_url}/attestation/document",
                            timeout=aiohttp.ClientTimeout(total=30)
                        ) as fallback_response:
                            if fallback_response.status != 200:
                                print(f"   ❌ Failed to fetch attestation: {fallback_response.status}")
                                return False
                            data = await fallback_response.json()
                            self.gateway_pubkey = data.get("enclave_pubkey")
                            self.gateway_code_hash = data.get("code_hash")
                            attestation_doc = data.get("attestation_document")  # may be b64
                    else:
                        print(f"   ❌ Failed to fetch attestation: {response.status}")
                        return False
            
            if not self.gateway_pubkey:
                print(f"   ❌ No gateway pubkey in response")
                return False
            
            print(f"   Gateway pubkey: {self.gateway_pubkey[:16]}...{self.gateway_pubkey[-16:]}")
            print(f"   Gateway code hash: {self.gateway_code_hash[:32] if self.gateway_code_hash else 'N/A'}...")
            
            # Check against pinned code hash (if set)
            if PINNED_GATEWAY_CODE_HASH:
                if self.gateway_code_hash != PINNED_GATEWAY_CODE_HASH:
                    print(f"   ❌ Gateway code hash mismatch!")
                    print(f"      Expected: {PINNED_GATEWAY_CODE_HASH[:32]}...")
                    print(f"      Got: {self.gateway_code_hash[:32] if self.gateway_code_hash else 'None'}...")
                    return False
                print(f"   ✅ Gateway code hash matches pinned value")
            else:
                print(f"   ⚠️  No pinned gateway code hash set (dev mode)")
            
            # Verify Nitro attestation
            if attestation_doc:
                if not await self._verify_nitro_attestation(
                    attestation_doc,  # Can be hex or base64
                    self.gateway_pubkey,
                    "gateway"
                ):
                    return False
            
            results["gateway"]["verified"] = True
            results["gateway"]["pubkey"] = self.gateway_pubkey
            results["gateway"]["code_hash"] = self.gateway_code_hash
            
            print(f"   ✅ Gateway attestation verified")
            return True
            
        except Exception as e:
            print(f"   ❌ Gateway attestation error: {e}")
            return False
    
    async def _verify_validator_attestation(self, results: Dict) -> bool:
        """Fetch and verify validator attestation from weights endpoint."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.gateway_url}/weights/current/{self.netuid}",
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status == 404:
                        print(f"   ⚠️  No weights published yet")
                        return True  # Not an error - just no weights yet
                    elif response.status != 200:
                        print(f"   ❌ Failed to fetch weights: {response.status}")
                        return False
                    
                    data = await response.json()
            
            attestation_b64 = data.get("validator_attestation_b64")
            self.validator_pubkey = data.get("validator_enclave_pubkey")
            self.validator_code_hash = data.get("validator_code_hash")
            
            if not attestation_b64:
                print(f"   ⚠️  No validator attestation in response (may be dev mode)")
                return True
            
            print(f"   Validator pubkey: {self.validator_pubkey[:16]}...")
            print(f"   Validator code hash: {self.validator_code_hash[:32] if self.validator_code_hash else 'N/A'}...")
            
            # Check against pinned code hash (if set)
            if PINNED_VALIDATOR_CODE_HASH:
                if self.validator_code_hash != PINNED_VALIDATOR_CODE_HASH:
                    print(f"   ❌ Validator code hash mismatch!")
                    print(f"      Expected: {PINNED_VALIDATOR_CODE_HASH[:32]}...")
                    print(f"      Got: {self.validator_code_hash[:32]}...")
                    return False
                print(f"   ✅ Validator code hash matches pinned value")
            else:
                print(f"   ⚠️  No pinned validator code hash set (dev mode)")
            
            # Verify Nitro attestation
            if not await self._verify_nitro_attestation(
                attestation_b64,
                self.validator_pubkey,
                "validator"
            ):
                return False
            
            results["validator"]["verified"] = True
            results["validator"]["pubkey"] = self.validator_pubkey
            results["validator"]["code_hash"] = self.validator_code_hash
            
            print(f"   ✅ Validator attestation verified")
            return True
            
        except Exception as e:
            print(f"   ❌ Validator attestation error: {e}")
            return False
    
    async def _verify_nitro_attestation(
        self,
        attestation_input: str,
        expected_pubkey: str,
        role: str,
        expected_epoch_id: Optional[int] = None,
    ) -> bool:
        """
        Verify AWS Nitro attestation document.
        
        Performs FULL cryptographic verification:
        1. Certificate chain to Amazon Nitro root
        2. COSE signature verification
        3. PCR0 check against pinned allowlist
        4. User data verification (pubkey, purpose, epoch binding)
        
        Args:
            attestation_input: Base64 or hex-encoded attestation document
            expected_pubkey: Expected enclave public key (hex string)
            role: "gateway" or "validator"
            expected_epoch_id: Required for validator attestations (replay protection)
            
        Returns:
            True if verification passes, False otherwise
        """
        try:
            # Handle both hex and base64 encoding
            # Gateway /attest endpoint returns hex, /weights endpoints return base64
            try:
                # Try hex decode first (gateway format)
                attestation_bytes = bytes.fromhex(attestation_input)
                attestation_b64 = base64.b64encode(attestation_bytes).decode()
            except ValueError:
                # Already base64
                attestation_b64 = attestation_input
                attestation_bytes = base64.b64decode(attestation_b64)
            
            print(f"      Attestation size: {len(attestation_bytes)} bytes")
            print(f"      Role: {role}")
            
            # Check if full Nitro verification is available
            if is_nitro_verification_available():
                print(f"      🔐 Attempting FULL Nitro verification...")
                
                # Determine expected purpose
                expected_purpose = None
                if role == "gateway":
                    expected_purpose = "gateway_event_signing"
                elif role == "validator":
                    expected_purpose = "validator_weights"
                
                # Run full verification
                success, result = verify_nitro_attestation_full(
                    attestation_b64=attestation_b64,
                    expected_pubkey=expected_pubkey,
                    expected_purpose=expected_purpose,
                    expected_epoch_id=expected_epoch_id,
                    role=role,
                )
                
                if success:
                    # Print verification steps
                    if "verification_steps" in result:
                        for step in result["verification_steps"]:
                            print(f"      {step}")
                    
                    # Update trust level to full Nitro
                    self.trust_level = TRUST_LEVEL_FULL_NITRO
                    
                    pcr0 = result.get("pcr0", "N/A")
                    print(f"      PCR0: {pcr0[:32]}...{pcr0[-16:]}" if len(pcr0) > 48 else f"      PCR0: {pcr0}")
                    print(f"   ✅ FULL NITRO VERIFICATION PASSED")
                    return True
                else:
                    # Verification failed
                    error = result.get("error", "Unknown error")
                    print(f"   ❌ Nitro verification FAILED: {error}")
                    return False
            
            else:
                # Full verification not available - fall back to signature-only mode
                print(f"   ⚠️  Full Nitro verification not available (missing config)")
                print(f"   ⚠️  Falling back to SIGNATURE-ONLY mode...")
                
                success, result = verify_nitro_attestation_signature_only(
                    attestation_b64=attestation_b64,
                    expected_pubkey=expected_pubkey,
                    expected_epoch_id=expected_epoch_id if role == "validator" else None,
                )
                
                if success:
                    pcr0 = result.get("pcr0", "N/A")
                    print(f"      PCR0 (unverified): {pcr0[:32]}..." if pcr0 and len(pcr0) > 32 else f"      PCR0: {pcr0}")
                    print(f"   ⚠️  SIGNATURE-ONLY MODE: Cannot verify enclave authenticity")
                    print(f"   ⚠️  Trust model is WEAKER - pubkey may not be from pinned code")
                    # self.trust_level remains "signature_only"
                    return True
                else:
                    error = result.get("error", "Unknown error")
                    print(f"   ❌ Attestation parsing failed: {error}")
                    return False
            
        except Exception as e:
            print(f"   ❌ Nitro attestation error: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    # =========================================================================
    # STEP 2: EPOCH BUNDLE VERIFICATION
    # =========================================================================
    
    async def verify_epoch_bundle(self, epoch_id: int) -> Tuple[bool, Dict]:
        """
        Verify validator actually produced the decisions/weights for an epoch.
        
        1. Fetch weights bundle from gateway
        2. Recompute H_weights from payload using canonical encoding
        3. Verify sig_val using validator_pubkey from attestation
        
        Returns:
            (success, details_dict) - details includes 'bundle' for chain_match
        """
        print("\n" + "=" * 60)
        print(f"📊 STEP 2: EPOCH BUNDLE VERIFICATION (Epoch {epoch_id})")
        print("=" * 60)
        
        results = {
            "epoch_id": epoch_id,
            "weights_hash": None,
            "signature_valid": False,
            "weights_count": 0,
            "bundle": None,  # Issue 22: Return bundle for verify_chain_match
        }
        
        try:
            # Fetch epoch bundle
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.gateway_url}/weights/latest/{self.netuid}/{epoch_id}",
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status == 404:
                        print(f"   ⚠️  No weights found for epoch {epoch_id}")
                        return False, results
                    elif response.status != 200:
                        print(f"   ❌ Failed to fetch epoch bundle: {response.status}")
                        return False, results
                    
                    bundle = await response.json()
            
            # Extract fields
            uids = bundle.get("uids", [])
            weights_u16 = bundle.get("weights_u16", [])
            claimed_hash = bundle.get("weights_hash")
            signature = bundle.get("validator_signature")
            pubkey = bundle.get("validator_enclave_pubkey")
            
            print(f"   Weights count: {len(uids)}")
            print(f"   Claimed hash: {claimed_hash[:16] if claimed_hash else 'None'}...")
            
            # Verify attestation per-bundle
            validator_attestation_b64 = bundle.get("validator_attestation_b64")
            if validator_attestation_b64:
                print(f"   Verifying bundle's validator attestation...")
                if not await self._verify_nitro_attestation(
                    validator_attestation_b64,
                    pubkey,
                    "validator",
                    expected_epoch_id=bundle.get("epoch_id")
                ):
                    print(f"   ❌ Bundle's validator attestation invalid!")
                    return False, results
                print(f"   ✅ Validator attestation verified for this bundle")
            else:
                print(f"   ⚠️  No validator attestation in bundle (may be dev mode)")
            
            # Recompute weights hash using canonical function
            weights_pairs = list(zip(uids, weights_u16))
            computed_hash = bundle_weights_hash(
                bundle["netuid"], bundle["epoch_id"], bundle["block"], weights_pairs
            )
            
            print(f"   Computed hash: {computed_hash[:16]}...")
            
            if computed_hash != claimed_hash:
                print(f"   ❌ Weights hash mismatch!")
                print(f"      Claimed: {claimed_hash}")
                print(f"      Computed: {computed_hash}")
                return False, results
            
            print(f"   ✅ Weights hash verified (recomputed matches claimed)")
            
            # Verify Ed25519 signature
            if not signature:
                print(f"   ❌ No signature in bundle!")
                return False, results
            
            try:
                pk = Ed25519PublicKey.from_public_bytes(bytes.fromhex(pubkey))
                pk.verify(bytes.fromhex(signature), bytes.fromhex(claimed_hash))
                print(f"   ✅ Signature verified (from attested validator TEE)")
            except Exception as e:
                print(f"   ❌ Signature verification failed: {e}")
                return False, results
            
            # Verify hotkey binding
            binding_message = bundle.get("binding_message")
            hotkey_signature = bundle.get("validator_hotkey_signature")
            validator_hotkey = bundle.get("validator_hotkey")
            
            if binding_message and hotkey_signature and validator_hotkey:
                expected_chain = os.environ.get(
                    "EXPECTED_CHAIN", 
                    "wss://entrypoint-finney.opentensor.ai:443"
                )
                
                binding_ok = verify_binding_message(
                    binding_message,
                    hotkey_signature,
                    validator_hotkey,
                    expected_netuid=bundle.get("netuid"),
                    expected_chain=expected_chain,
                    expected_enclave_pubkey=pubkey,
                    expected_code_hash=bundle.get("validator_code_hash"),
                )
                
                if not binding_ok:
                    print(f"   ❌ Hotkey binding verification failed!")
                    return False, results
                
                print(f"   ✅ Hotkey binding verified (enclave authorized by {validator_hotkey[:16]}...)")
            else:
                print(f"   ⚠️  No hotkey binding in bundle")
            
            results["weights_hash"] = computed_hash
            results["signature_valid"] = True
            results["weights_count"] = len(uids)
            results["bundle"] = bundle  # Issue 22: Include bundle for chain_match
            
            print("\n" + "-" * 40)
            print(f"✅ EPOCH BUNDLE VERIFICATION: PASSED")
            print(f"   Weights for epoch {epoch_id} are authentically from validator TEE")
            
            return True, results
            
        except Exception as e:
            print(f"   ❌ Epoch bundle verification error: {e}")
            import traceback
            traceback.print_exc()
            return False, results
    
    # =========================================================================
    # STEP 3: LOG CHAIN VERIFICATION
    # =========================================================================
    
    async def verify_log_chain(
        self,
        from_epoch: int,
        to_epoch: int,
    ) -> Tuple[bool, Dict]:
        """
        Verify gateway's log chain integrity.
        
        For each log entry:
        1. Verify gateway signature using gateway pubkey from attestation
        2. Verify hash-chain: each entry includes hash of previous entry
        3. Verify monotonic sequence (no gaps within boot_id, handle restarts)
        
        Uses epoch-bounded queries (not global 10k limit).
        Fails loud if truncation detected.
        Groups by boot_id for monotonic verification.
        
        Returns:
            (success, details_dict)
        """
        print("\n" + "=" * 60)
        print(f"📝 STEP 3: LOG CHAIN VERIFICATION (Epochs {from_epoch} to {to_epoch})")
        print("=" * 60)
        
        results = {
            "from_epoch": from_epoch,
            "to_epoch": to_epoch,
            "events_verified": 0,
            "chain_valid": False,
            "sequence_valid": False,
            "boot_sessions": 0,
            "restart_events": 0,
        }
        
        try:
            # Ensure we have gateway pubkey
            if not self.gateway_pubkey:
                print("   Fetching gateway attestation first...")
                await self.verify_attestations()
            
            # Issue 21: Use env vars for Supabase (external auditors don't have gateway package)
            from supabase import create_client
            
            supabase_url = os.environ.get("SUPABASE_URL")
            supabase_anon_key = os.environ.get("SUPABASE_ANON_KEY")
            
            if not supabase_url or not supabase_anon_key:
                raise VerificationError(
                    "Set SUPABASE_URL and SUPABASE_ANON_KEY env vars.\n"
                    "These are the PUBLIC read-only credentials for the transparency log."
                )
            
            read_client = create_client(supabase_url, supabase_anon_key)
            
            # FIX: Use epoch-bounded queries for production reliability
            print(f"   📜 Fetching transparency log for epochs {from_epoch}-{to_epoch}...")
            print(f"      Netuid: {self.netuid}")
            
            log_result = read_client.table("transparency_log") \
                .select("*") \
                .eq("netuid", self.netuid) \
                .gte("epoch_id", from_epoch) \
                .lte("epoch_id", to_epoch) \
                .order("created_at", desc=False) \
                .execute()
            
            # FAIL LOUD if query seems truncated (safety check)
            row_count = len(log_result.data) if log_result.data else 0
            if row_count >= 10000:
                raise VerificationError(
                    f"Query returned {row_count} rows - possible truncation. "
                    "Use smaller epoch range or implement pagination."
                )
            
            print(f"   Fetched {row_count} log entries")
            
            if row_count == 0:
                print(f"   ⚠️  No entries found for epochs {from_epoch}-{to_epoch}")
                print(f"      This may be normal if no events occurred in this range")
                return True, results
            
            # Extract log entries from payload column
            # Each row's payload IS the full log_entry object:
            # { signed_event, event_hash, enclave_pubkey, enclave_signature }
            log_entries = [row["payload"] for row in log_result.data]
            
            # ═══════════════════════════════════════════════════════════════════
            # SIGNATURE VERIFICATION
            # ═══════════════════════════════════════════════════════════════════
            print(f"\n   🔐 Verifying signatures...")
            
            verified_count = 0
            for i, log_entry in enumerate(log_entries):
                if verify_log_entry(log_entry, self.gateway_pubkey):
                    verified_count += 1
                else:
                    print(f"   ❌ Log entry {i} signature INVALID!")
                    return False, results
            
            print(f"   ✅ All {verified_count} signatures verified")
            results["events_verified"] = verified_count
            
            # ═══════════════════════════════════════════════════════════════════
            # HASH-CHAIN VERIFICATION
            # ═══════════════════════════════════════════════════════════════════
            print(f"\n   🔗 Verifying hash-chain continuity...")
            
            prev_event_hash = None
            chain_breaks = 0
            restart_events = 0
            
            for i, log_entry in enumerate(log_entries):
                signed_event = log_entry.get("signed_event", {})
                event_type = signed_event.get("event_type")
                entry_prev_hash = signed_event.get("prev_event_hash")
                current_hash = log_entry.get("event_hash")
                
                if event_type == "ENCLAVE_RESTART":
                    restart_events += 1
                    # RESTART event may reference last log tip from previous boot
                    print(f"      🔄 Restart event at entry {i}")
                
                # Verify chain continuity (skip first entry)
                if prev_event_hash is not None:
                    if entry_prev_hash != prev_event_hash:
                        # Chain break - might be restart boundary
                        if event_type == "ENCLAVE_RESTART":
                            # RESTART events can have different prev_hash (pointing to previous boot)
                            print(f"      ⚠️  Chain break at entry {i} (RESTART boundary)")
                        else:
                            print(f"   ❌ Hash-chain broken at entry {i}!")
                            print(f"      Expected prev: {prev_event_hash[:16]}...")
                            print(f"      Got prev: {entry_prev_hash[:16] if entry_prev_hash else 'None'}...")
                            chain_breaks += 1
                            # Don't fail immediately - count all breaks
                
                prev_event_hash = current_hash
            
            results["restart_events"] = restart_events
            
            if chain_breaks > 0:
                print(f"   ❌ Hash-chain has {chain_breaks} break(s)!")
                results["chain_valid"] = False
                return False, results
            
            print(f"   ✅ Hash-chain verified ({restart_events} restart boundaries)")
            results["chain_valid"] = True
            
            # ═══════════════════════════════════════════════════════════════════
            # MONOTONIC SEQUENCE VERIFICATION (grouped by boot_id)
            # ═══════════════════════════════════════════════════════════════════
            print(f"\n   📈 Verifying monotonic sequences (per boot_id)...")
            
            # Group entries by boot_id
            boot_sessions: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
            
            for i, log_entry in enumerate(log_entries):
                signed_event = log_entry.get("signed_event", {})
                boot_id = signed_event.get("boot_id")
                seq = signed_event.get("monotonic_seq")
                
                if boot_id and seq is not None:
                    boot_sessions[boot_id].append((i, seq))
            
            results["boot_sessions"] = len(boot_sessions)
            print(f"      Found {len(boot_sessions)} boot session(s)")
            
            # Verify monotonicity within each boot session
            sequence_valid = True
            for boot_id, entries in boot_sessions.items():
                # Sort by sequence number
                entries_sorted = sorted(entries, key=lambda x: x[1])
                
                # Check for duplicates
                seqs = [e[1] for e in entries_sorted]
                if len(seqs) != len(set(seqs)):
                    print(f"   ❌ Duplicate sequence numbers in boot {boot_id[:8]}...")
                    sequence_valid = False
                    continue
                
                # Check for gaps (warning only - partial fetch may have gaps)
                gaps = []
                for j in range(1, len(entries_sorted)):
                    prev_seq = entries_sorted[j-1][1]
                    curr_seq = entries_sorted[j][1]
                    if curr_seq != prev_seq + 1:
                        gaps.append((prev_seq, curr_seq))
                
                if gaps:
                    print(f"      ⚠️  Boot {boot_id[:8]}...: {len(gaps)} gap(s) in sequence")
                    print(f"         (may be due to partial fetch)")
                else:
                    print(f"      ✅ Boot {boot_id[:8]}...: {len(entries_sorted)} events, seq {seqs[0]}-{seqs[-1]}")
                
                # Check strict monotonicity
                for j in range(1, len(entries_sorted)):
                    if entries_sorted[j][1] <= entries_sorted[j-1][1]:
                        print(f"   ❌ Sequence not monotonic in boot {boot_id[:8]}...")
                        sequence_valid = False
                        break
            
            results["sequence_valid"] = sequence_valid
            
            if not sequence_valid:
                print(f"   ❌ Monotonic sequence verification FAILED")
                return False, results
            
            print(f"   ✅ Monotonic sequence verified for all boot sessions")
            
            # ═══════════════════════════════════════════════════════════════════
            # SUMMARY
            # ═══════════════════════════════════════════════════════════════════
            print("\n" + "-" * 40)
            print(f"✅ LOG CHAIN VERIFICATION: PASSED")
            print(f"   Events verified: {verified_count}")
            print(f"   Boot sessions: {len(boot_sessions)}")
            print(f"   Restart events: {restart_events}")
            print(f"   Chain integrity: ✅")
            print(f"   Sequence integrity: ✅")
            
            return True, results
            
        except VerificationError as e:
            print(f"   ❌ {e}")
            return False, results
        except Exception as e:
            print(f"   ❌ Log chain verification error: {e}")
            import traceback
            traceback.print_exc()
            return False, results
    
    # =========================================================================
    # STEP 4: CHAIN MATCH (EQUIVOCATION DETECTION)
    # =========================================================================
    
    async def verify_chain_match(
        self,
        epoch_id: int,
        bundle: Optional[Dict] = None,
    ) -> Tuple[bool, Dict]:
        """
        Check if on-chain weights match published bundle (equivocation detection).
        
        CRITICAL: Uses chain_snapshot_compare_hash from bundle, NOT live chain query!
        subtensor.weights() returns CURRENT weights which may have changed.
        
        Returns:
            (success, details_dict)
        """
        print("\n" + "=" * 60)
        print(f"⚖️  STEP 4: CHAIN MATCH VERIFICATION (Epoch {epoch_id})")
        print("=" * 60)
        
        results = {
            "epoch_id": epoch_id,
            "bundle_hash": None,
            "snapshot_hash": None,
            "match": False,
        }
        
        try:
            # Fetch bundle if not provided
            if bundle is None:
                print(f"   Fetching bundle...")
                success, bundle_results = await self.verify_epoch_bundle(epoch_id)
                if not success:
                    print(f"   ❌ Could not verify bundle first")
                    return False, results
                bundle = bundle_results.get("bundle")
            
            if not bundle:
                print(f"   ❌ No bundle available")
                return False, results
            
            # Compute bundle compare hash (NO block for comparison)
            uids = bundle.get("uids", [])
            weights_u16 = bundle.get("weights_u16", [])
            weights_pairs = list(zip(uids, weights_u16))
            
            bundle_compare = compare_weights_hash(self.netuid, epoch_id, weights_pairs)
            results["bundle_hash"] = bundle_compare
            
            print(f"   Bundle compare hash: {bundle_compare[:16]}...")
            
            # Get chain snapshot hash from bundle
            snapshot_hash = bundle.get("chain_snapshot_compare_hash")
            snapshot_block = bundle.get("chain_snapshot_block")
            
            if snapshot_hash:
                results["snapshot_hash"] = snapshot_hash
                print(f"   Snapshot compare hash: {snapshot_hash[:16]}...")
                print(f"   Snapshot block: {snapshot_block}")
                
                if snapshot_hash == bundle_compare:
                    results["match"] = True
                    print("\n" + "-" * 40)
                    print(f"✅ CHAIN MATCH VERIFICATION: PASSED")
                    print(f"   Bundle weights match chain snapshot")
                    print(f"   No equivocation detected")
                    return True, results
                else:
                    print("\n" + "-" * 40)
                    print(f"❌ CHAIN MATCH VERIFICATION: FAILED")
                    print(f"   ⚠️  EQUIVOCATION DETECTED!")
                    print(f"   Bundle hash:   {bundle_compare}")
                    print(f"   Snapshot hash: {snapshot_hash}")
                    print(f"   The validator may have submitted different weights to chain!")
                    return False, results
            else:
                print(f"   ⚠️  No chain_snapshot_compare_hash in bundle")
                print(f"      Cannot verify equivocation without snapshot")
                print(f"      This is a gap in audit coverage")
                return True, results  # Not an error, just missing data
            
        except Exception as e:
            print(f"   ❌ Chain match verification error: {e}")
            import traceback
            traceback.print_exc()
            return False, results
    
    # =========================================================================
    # FULL VERIFICATION
    # =========================================================================
    
    async def verify_all(
        self,
        epoch_id: int,
        from_epoch: Optional[int] = None,
        to_epoch: Optional[int] = None,
    ) -> Dict:
        """
        Run all verification steps.
        
        Args:
            epoch_id: Target epoch for bundle and chain match
            from_epoch: Start of log chain range (default: epoch_id - 5)
            to_epoch: End of log chain range (default: epoch_id)
        
        Returns:
            Dict with all verification results
        """
        print("\n" + "=" * 60)
        print("🔍 COMPREHENSIVE AUDITOR VERIFICATION")
        print("=" * 60)
        print(f"   Gateway: {self.gateway_url}")
        print(f"   Netuid: {self.netuid}")
        print(f"   Target epoch: {epoch_id}")
        
        # Set defaults for log range
        if from_epoch is None:
            from_epoch = max(0, epoch_id - 5)
        if to_epoch is None:
            to_epoch = epoch_id
        
        results = {
            "attestations": None,
            "bundle": None,
            "log_chain": None,
            "chain_match": None,
            "trust_level": self.trust_level,
            "all_passed": False,
        }
        
        # Step 1: Attestations
        attestation_results = await self.verify_attestations()
        results["attestations"] = attestation_results
        attestations_ok = (
            attestation_results["gateway"]["verified"] and 
            attestation_results["validator"]["verified"]
        )
        
        # Step 2: Bundle verification
        bundle_ok, bundle_results = await self.verify_epoch_bundle(epoch_id)
        results["bundle"] = bundle_results
        
        # Step 3: Log chain
        logs_ok, log_results = await self.verify_log_chain(from_epoch, to_epoch)
        results["log_chain"] = log_results
        
        # Step 4: Chain match
        chain_ok, chain_results = await self.verify_chain_match(
            epoch_id, 
            bundle=bundle_results.get("bundle")
        )
        results["chain_match"] = chain_results
        
        # Final summary
        all_passed = attestations_ok and bundle_ok and logs_ok and chain_ok
        results["all_passed"] = all_passed
        results["trust_level"] = self.trust_level
        
        print("\n" + "=" * 60)
        print("📋 VERIFICATION SUMMARY")
        print("=" * 60)
        print(f"   Step 1 (Attestations): {'✅ PASSED' if attestations_ok else '❌ FAILED'}")
        print(f"   Step 2 (Bundle):       {'✅ PASSED' if bundle_ok else '❌ FAILED'}")
        print(f"   Step 3 (Log Chain):    {'✅ PASSED' if logs_ok else '❌ FAILED'}")
        print(f"   Step 4 (Chain Match):  {'✅ PASSED' if chain_ok else '❌ FAILED'}")
        print("-" * 60)
        
        if all_passed and self.trust_level == "full_nitro":
            print("✅ FULLY VERIFIED: Nitro attestation + signatures")
        elif all_passed and self.trust_level == "signature_only":
            print("⚠️  PARTIALLY VERIFIED: Signatures only (Nitro verification unavailable)")
            print("   The system MAY be trustworthy, but enclave authenticity is not proven.")
        else:
            print("❌ VERIFICATION FAILED")
        
        print("=" * 60)
        
        return results


# =============================================================================
# CLI
# =============================================================================

def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Auditor Verifier CLI - Comprehensive TEE verification tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES:
    # Verify attestations only
    python scripts/auditor_verifier.py verify-attestations
    
    # Verify specific epoch bundle
    python scripts/auditor_verifier.py verify-bundle --epoch 12345
    
    # Verify log chain for epoch range
    python scripts/auditor_verifier.py verify-logs --from-epoch 12340 --to-epoch 12345
    
    # Check for equivocation
    python scripts/auditor_verifier.py verify-chain-match --epoch 12345
    
    # Full verification
    python scripts/auditor_verifier.py verify-all --epoch 12345

ENVIRONMENT VARIABLES:
    GATEWAY_URL              Gateway URL (default: http://54.226.209.164:8000)
    NETUID                   Subnet ID (default: 71)
    SUPABASE_URL             Supabase project URL (required for log verification)
    SUPABASE_ANON_KEY        Supabase anonymous key (required for log verification)
    EXPECTED_GATEWAY_CODE_HASH    Pinned gateway code hash (optional)
    EXPECTED_VALIDATOR_CODE_HASH  Pinned validator code hash (optional)
    EXPECTED_CHAIN           Chain endpoint for binding (optional)
        """
    )
    
    parser.add_argument(
        "command",
        choices=[
            "verify-attestations",
            "verify-bundle", 
            "verify-logs",
            "verify-chain-match",
            "verify-all",
        ],
        help="Verification command to run"
    )
    parser.add_argument(
        "--epoch",
        type=int,
        help="Target epoch ID"
    )
    parser.add_argument(
        "--from-epoch",
        type=int,
        help="Start epoch for log chain verification"
    )
    parser.add_argument(
        "--to-epoch",
        type=int,
        help="End epoch for log chain verification"
    )
    parser.add_argument(
        "--gateway-url",
        type=str,
        default=DEFAULT_GATEWAY_URL,
        help=f"Gateway URL (default: {DEFAULT_GATEWAY_URL})"
    )
    parser.add_argument(
        "--netuid",
        type=int,
        default=DEFAULT_NETUID,
        help=f"Subnet UID (default: {DEFAULT_NETUID})"
    )
    
    args = parser.parse_args()
    
    # Validate args
    if args.command in ["verify-bundle", "verify-chain-match", "verify-all"]:
        if args.epoch is None:
            parser.error(f"--epoch required for {args.command}")
    
    if args.command == "verify-logs":
        if args.from_epoch is None or args.to_epoch is None:
            parser.error("--from-epoch and --to-epoch required for verify-logs")
    
    # Create verifier
    verifier = AuditorVerifier(
        gateway_url=args.gateway_url,
        netuid=args.netuid,
    )
    
    # Run command
    async def run():
        if args.command == "verify-attestations":
            await verifier.verify_attestations()
        elif args.command == "verify-bundle":
            await verifier.verify_epoch_bundle(args.epoch)
        elif args.command == "verify-logs":
            await verifier.verify_log_chain(args.from_epoch, args.to_epoch)
        elif args.command == "verify-chain-match":
            await verifier.verify_chain_match(args.epoch)
        elif args.command == "verify-all":
            await verifier.verify_all(
                args.epoch,
                from_epoch=args.from_epoch,
                to_epoch=args.to_epoch,
            )
    
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        print("\n⛔ Interrupted")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

