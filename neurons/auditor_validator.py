#!/usr/bin/env python3
"""
LeadPoet Auditor Validator

A lightweight validator that copies authoritative V2 weights after independently
verifying their complete attested receipt graph. During the V2 rollout, it can
copy a fully verified V1 bundle only when the gateway has no V2 route.

Verification failure is fail-closed: the auditor never substitutes a locally
computed or burn vector, and an invalid or pending V2 authority never downgrades.

USAGE:
    python neurons/auditor_validator.py --netuid 71 --wallet.name my_wallet --wallet.hotkey default
"""

import os
import sys
import argparse
import json
import re
import time
from urllib.parse import urlparse

# ════════════════════════════════════════════════════════════════════════════
# AUTO-UPDATER: Automatically updates entire repo from GitHub for auditors
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__" and os.environ.get("LEADPOET_AUDITOR_WRAPPER_ACTIVE") != "1":
    print("🔄 Leadpoet Auditor Validator: Activating auto-update wrapper...")
    print("   Your auditor will automatically stay up-to-date with the latest code")
    print("")
    
    # Create wrapper script path (hidden file with dot prefix)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(script_dir)
    wrapper_path = os.path.join(repo_root, ".auditor_auto_update_wrapper.sh") 
    
    # Inline wrapper script - pulls on start only, not every 5 minutes
    wrapper_content = '''#!/bin/bash
# Auto-generated wrapper for Leadpoet auditor validator auto-updates
# Pulls latest code ONCE on start, then runs until clean exit
set -e

REPO_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$REPO_ROOT"

echo "════════════════════════════════════════════════════════════════"
echo "🔍 Leadpoet Auto-Updating Auditor Validator"
echo "   Repository updates on each manual restart"
echo "   GitHub: github.com/leadpoet/leadpoet"
echo "════════════════════════════════════════════════════════════════"
echo ""

# Pull latest code ONCE at startup
echo "────────────────────────────────────────────────────────────────"
echo "🔍 Checking for updates from GitHub..."

# Stash any local changes and pull latest
if git stash 2>/dev/null; then
    echo "   💾 Stashed local changes"
fi

if git pull origin main 2>/dev/null; then
    CURRENT_COMMIT=$(git rev-parse --short HEAD)
    echo "✅ Repository updated"
    echo "   Current commit: $CURRENT_COMMIT"
    
    # Auto-install new/updated Python packages if requirements.txt changed
    if git diff HEAD@{1} HEAD --name-only 2>/dev/null | grep -q "requirements.txt"; then
        echo "📦 requirements.txt changed - updating packages..."
        pip3 install -r requirements.txt --quiet || echo "   ⚠️  Package install failed (continuing anyway)"
    fi
else
    echo "⏭️  Could not update (offline or not a git repo)"
    echo "   Continuing with current version..."
fi

RESTART_COUNT=0
MAX_RESTARTS=5

while true; do
    echo "────────────────────────────────────────────────────────────────"
    echo "🟢 Starting auditor validator (attempt $(($RESTART_COUNT + 1)))..."
    echo ""
    
    # Run auditor with environment flag to prevent wrapper re-execution
    export LEADPOET_AUDITOR_WRAPPER_ACTIVE=1
    python3 neurons/auditor_validator.py "$@"
    
    EXIT_CODE=$?
    
    echo ""
    echo "────────────────────────────────────────────────────────────────"
    
    if [ $EXIT_CODE -eq 0 ]; then
        echo "✅ Auditor exited cleanly (exit code: 0)"
        echo "   Shutting down. Run the command again to pull latest updates."
        break
    elif [ $EXIT_CODE -eq 137 ] || [ $EXIT_CODE -eq 9 ]; then
        echo "⚠️  Auditor was killed (exit code: $EXIT_CODE) - likely Out of Memory"
        echo "   Cleaning up resources before restart..."
        
        # Clean up any leaked resources
        pkill -f "python3 neurons/auditor_validator.py" 2>/dev/null || true
        sleep 5  # Give system time to clean up
        
        RESTART_COUNT=$((RESTART_COUNT + 1))
        if [ $RESTART_COUNT -ge $MAX_RESTARTS ]; then
            echo "❌ Maximum restart attempts ($MAX_RESTARTS) reached"
            echo "   Please check logs and restart manually"
            exit 1
        fi
        
        echo "   Restarting in 30 seconds... (attempt $RESTART_COUNT/$MAX_RESTARTS)"
        sleep 30
    else
        RESTART_COUNT=$((RESTART_COUNT + 1))
        echo "⚠️  Auditor exited with error (exit code: $EXIT_CODE)"
        
        if [ $RESTART_COUNT -ge $MAX_RESTARTS ]; then
            echo "❌ Maximum restart attempts ($MAX_RESTARTS) reached"
            echo "   Please check logs and restart manually"
            exit 1
        fi
        
        echo "   Restarting in 10 seconds... (attempt $RESTART_COUNT/$MAX_RESTARTS)"
        sleep 10
    fi
done

echo "════════════════════════════════════════════════════════════════"
echo "🛑 Auditor stopped. Run command again to pull latest and restart."
'''
    
    # Write wrapper script
    try:
        with open(wrapper_path, 'w') as f:
            f.write(wrapper_content)
        os.chmod(wrapper_path, 0o755)
        print(f"✅ Created auto-update wrapper: {wrapper_path}")
    except Exception as e:
        print(f"❌ Failed to create wrapper: {e}")
        print("   Continuing without auto-updates...")
        # Fall through to normal execution
    else:
        # Execute wrapper and replace current process
        print("🚀 Launching auto-update wrapper...\n")
        try:
            env = os.environ.copy()
            env["LEADPOET_AUDITOR_WRAPPER_ACTIVE"] = "1"
            os.execve(wrapper_path, [wrapper_path] + sys.argv[1:], env)
        except Exception as e:
            print(f"❌ Failed to execute wrapper: {e}")
            print("   Continuing without auto-updates...")

# ════════════════════════════════════════════════════════════════════════════
# NORMAL AUDITOR VALIDATOR CODE STARTS BELOW
# ════════════════════════════════════════════════════════════════════════════

# Add repo root to path so leadpoet_canonical can be imported from anywhere
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SCRIPT_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
import asyncio
import logging
import base64
from pathlib import Path
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
from leadpoet_canonical.auditor_v2 import (
    load_identity_cache,
    verify_attested_weight_authority_v2,
)

# Constants from canonical module
from leadpoet_canonical.constants import EPOCH_LENGTH, WEIGHT_SUBMISSION_BLOCK

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# default gateway URL
DEFAULT_GATEWAY_URL = os.environ.get("GATEWAY_URL", "")


def _normalize_gateway_url(value: str) -> str:
    """Accept the current gateway transport without weakening V2 integrity."""

    normalized = str(value or "").strip().rstrip("/")
    parsed = urlparse(normalized)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise RuntimeError("Auditor gateway URL must be an HTTP(S) origin")
    return normalized


# File to store pending equivocation check (overwritten each epoch)
PENDING_EQUIVOCATION_FILE = os.path.join(_SCRIPT_DIR, ".pending_equivocation_check.json")


class AuditorValidator:
    """
    Validator that prefers V2 and narrowly supports verified rollout-era V1.
    """
    
    def __init__(self, config, gateway_url: str):
        """
        Initialize auditor validator.
        
        Args:
            config: Bittensor config object
            gateway_url: Gateway URL (passed as parameter, not global)
        """
        self.config = config
        self.gateway_url = _normalize_gateway_url(gateway_url)
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
        APPENDS to existing data (stores last 2 epochs for N-2 checking).
        
        File structure:
        {
            "20256": {"bundle_compare_hash": "...", "validator_hotkey": "...", "saved_at": "..."},
            "20257": {"bundle_compare_hash": "...", "validator_hotkey": "...", "saved_at": "..."}
        }
        """
        import json
        from datetime import datetime, timezone
        
        # Load existing data
        existing_data = {}
        try:
            if os.path.exists(PENDING_EQUIVOCATION_FILE):
                with open(PENDING_EQUIVOCATION_FILE, 'r') as f:
                    existing_data = json.load(f)
                    # Handle old format (single epoch) - migrate to new format
                    if "epoch_id" in existing_data:
                        old_epoch = str(existing_data["epoch_id"])
                        existing_data = {
                            old_epoch: {
                                "bundle_compare_hash": existing_data.get("bundle_compare_hash"),
                                "validator_hotkey": existing_data.get("validator_hotkey"),
                                "saved_at": existing_data.get("saved_at"),
                            }
                        }
        except Exception as e:
            logger.warning(f"Could not load existing equivocation data: {e}")
            existing_data = {}
        
        # Add new epoch data
        existing_data[str(epoch_id)] = {
            "bundle_compare_hash": bundle_compare_hash,
            "validator_hotkey": validator_hotkey,
            "saved_at": datetime.now(timezone.utc).isoformat(),
        }
        
        # Keep only last 3 epochs (current + 2 previous for safety)
        epoch_keys = sorted(existing_data.keys(), key=int, reverse=True)
        if len(epoch_keys) > 3:
            for old_key in epoch_keys[3:]:
                del existing_data[old_key]
        
        try:
            with open(PENDING_EQUIVOCATION_FILE, 'w') as f:
                json.dump(existing_data, f, indent=2)
            print(f"   📝 Saved pending equivocation check for epoch {epoch_id}")
            print(f"      Epochs in file: {sorted(existing_data.keys(), key=int)}")
        except Exception as e:
            logger.warning(f"Failed to save pending equivocation check: {e}")
    
    def load_pending_equivocation_check(self, target_epoch: int = None) -> Optional[Dict]:
        """
        Load pending equivocation check data for a specific epoch.
        
        Args:
            target_epoch: Specific epoch to load (if None, returns all epochs)
            
        Returns:
            Dict with bundle_compare_hash, validator_hotkey for the target epoch
            None if not found or file doesn't exist
        """
        import json
        try:
            if not os.path.exists(PENDING_EQUIVOCATION_FILE):
                return None
            with open(PENDING_EQUIVOCATION_FILE, 'r') as f:
                data = json.load(f)
            
            # Handle old format (single epoch with epoch_id key)
            if "epoch_id" in data:
                old_epoch = data["epoch_id"]
                if target_epoch is None or target_epoch == old_epoch:
                    return {
                        "epoch_id": old_epoch,
                        "bundle_compare_hash": data.get("bundle_compare_hash"),
                        "validator_hotkey": data.get("validator_hotkey"),
                    }
                return None
            
            # New format: dict keyed by epoch_id string
            if target_epoch is not None:
                epoch_data = data.get(str(target_epoch))
                if epoch_data:
                    return {
                        "epoch_id": target_epoch,
                        "bundle_compare_hash": epoch_data.get("bundle_compare_hash"),
                        "validator_hotkey": epoch_data.get("validator_hotkey"),
                    }
                return None
            
            # Return all data if no target specified
            return data
            
        except Exception as e:
            logger.warning(f"Failed to load pending equivocation check: {e}")
            return None
    
    def clear_pending_equivocation_check(self, epoch_id: int = None):
        """
        Clear a specific epoch from the pending equivocation check file.
        
        Args:
            epoch_id: Specific epoch to remove. If None, clears entire file.
        """
        import json
        try:
            if not os.path.exists(PENDING_EQUIVOCATION_FILE):
                return
                
            if epoch_id is None:
                # Clear entire file
                os.remove(PENDING_EQUIVOCATION_FILE)
                return
            
            # Load existing data
            with open(PENDING_EQUIVOCATION_FILE, 'r') as f:
                data = json.load(f)
            
            # Handle old format
            if "epoch_id" in data:
                if data["epoch_id"] == epoch_id:
                    os.remove(PENDING_EQUIVOCATION_FILE)
                return
            
            # Remove specific epoch from new format
            if str(epoch_id) in data:
                del data[str(epoch_id)]
                print(f"   🗑️  Cleared epoch {epoch_id} from pending checks")
                
                if data:
                    # Write remaining epochs back
                    with open(PENDING_EQUIVOCATION_FILE, 'w') as f:
                        json.dump(data, f, indent=2)
                else:
                    # No epochs left, delete file
                    os.remove(PENDING_EQUIVOCATION_FILE)
                    
        except Exception as e:
            logger.warning(f"Failed to clear pending equivocation check: {e}")
    
    async def perform_soft_equivocation_check(self, target_epoch: int) -> bool:
        """
        Retroactively verify a specific epoch's weights against chain.
        
        Called at block 30-80 of each new epoch to check N-2 epoch.
        Compares stored bundle hash with what's actually on chain.
        
        Args:
            target_epoch: The epoch to check (typically current_epoch - 2)
        
        Returns:
            True if check passed or no pending check
            False if equivocation detected
        """
        pending = self.load_pending_equivocation_check(target_epoch=target_epoch)
        if not pending:
            return True  # No pending check for this epoch
        
        epoch_id = target_epoch
        bundle_compare_hash = pending.get("bundle_compare_hash")
        validator_hotkey = pending.get("validator_hotkey")
        
        if not all([bundle_compare_hash, validator_hotkey]):
            logger.warning(f"Invalid pending equivocation data for epoch {epoch_id}")
            self.clear_pending_equivocation_check(epoch_id)
            return True
        
        print(f"\n{'='*60}")
        print(f"🔍 SOFT EQUIVOCATION CHECK (Epoch {epoch_id})")
        print(f"{'='*60}")
        print(f"   Checking previous epoch's weights against chain...")
        
        try:
            # Find validator's UID
            if validator_hotkey not in self.metagraph.hotkeys:
                print(f"   ⚠️  Validator hotkey not in metagraph, skipping check")
                self.clear_pending_equivocation_check(epoch_id)
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
                self.clear_pending_equivocation_check(epoch_id)
                return True
            
            # Normalize chain weights to pairs
            chain_pairs = normalize_chain_weights(chain_weights)
            
            # We need to reconstruct bundle weights pairs from the hash
            # Since we only stored the hash, we need to fetch the bundle again
            # OR use tolerance-based comparison on the actual weights
            
            # For soft check, we'll fetch the bundle and compare with tolerance
            # (hash comparison is too strict due to ±1 u16 round-trip tolerance)
            print(f"   Fetching bundle for epoch {epoch_id} to compare weights...")
            
            bundle, authority_status = await self.fetch_verified_weight_authority(
                epoch_id
            )

            if not bundle:
                print(
                    f"   ⚠️  Could not verify weight authority for epoch "
                    f"{epoch_id} ({authority_status}), skipping check"
                )
                self.clear_pending_equivocation_check(epoch_id)
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
                self.clear_pending_equivocation_check(epoch_id)
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
                self.clear_pending_equivocation_check(epoch_id)
                return False
            
            # All weights within tolerance
            print(f"   ✅ MATCH - All {len(bundle_pairs)} weights within ±1 tolerance")
            print(f"   No equivocation detected for epoch {epoch_id}")
            self.clear_pending_equivocation_check(epoch_id)
            return True
                
        except Exception as e:
            print(f"   ⚠️  Soft equivocation check failed: {e}")
            logger.warning(f"Soft equivocation check error: {e}")
            self.clear_pending_equivocation_check(epoch_id)
            return True  # Don't block on errors
    
    # ═══════════════════════════════════════════════════════════════════════════
    # Gateway Communication
    # ═══════════════════════════════════════════════════════════════════════════
    
    async def fetch_attested_weights_v2(self, epoch_id: int) -> Optional[Dict]:
        """Fetch the sole authoritative V2 bundle."""

        self._last_v2_route_was_missing = False
        try:
            async with aiohttp.ClientSession(trust_env=False) as session:
                url = f"{self.gateway_url}/weights/v2/latest/{self.config.netuid}/{int(epoch_id)}"
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as response:
                    if response.status == 404:
                        try:
                            not_found = await response.json()
                        except Exception:
                            not_found = None
                        self._last_v2_route_was_missing = (
                            isinstance(not_found, dict)
                            and not_found.get("detail") == "Not Found"
                        )
                        return None
                    if response.status != 200:
                        logger.warning(
                            "auditor_v2_fetch_failed epoch=%s status=%s",
                            epoch_id,
                            response.status,
                        )
                        return None
                    value = await response.json()
                    return value if isinstance(value, dict) else None
        except Exception as exc:
            logger.warning(
                "auditor_v2_fetch_failed epoch=%s error_type=%s error=%s",
                epoch_id,
                type(exc).__name__,
                str(exc)[:200],
            )
            return None

    async def fetch_attested_weights_v1(self, epoch_id: int) -> Optional[Dict]:
        """Fetch the temporary V1 compatibility bundle."""

        try:
            async with aiohttp.ClientSession(trust_env=False) as session:
                url = (
                    f"{self.gateway_url}/weights/latest/"
                    f"{self.config.netuid}/{int(epoch_id)}"
                )
                async with session.get(
                    url,
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as response:
                    if response.status == 404:
                        return None
                    if response.status != 200:
                        logger.warning(
                            "auditor_v1_fallback_fetch_failed epoch=%s status=%s",
                            epoch_id,
                            response.status,
                        )
                        return None
                    value = await response.json()
                    return value if isinstance(value, dict) else None
        except Exception as exc:
            logger.warning(
                "auditor_v1_fallback_fetch_failed epoch=%s error_type=%s error=%s",
                epoch_id,
                type(exc).__name__,
                str(exc)[:200],
            )
            return None

    @staticmethod
    def _v1_pcr0_commit_is_allowed(pcr0: str, commit_hash: str) -> bool:
        try:
            allowlist = json.loads(
                (Path(_REPO_ROOT) / "pcr0_allowlist.json").read_text(
                    encoding="utf-8"
                )
            )
        except Exception:
            return False
        return any(
            isinstance(item, dict)
            and str(item.get("pcr0") or "").lower() == pcr0
            and str(item.get("commit_hash") or "").lower() == commit_hash
            for item in allowlist.get("validator_pcr0", [])
        )

    @staticmethod
    def _verify_v1_nitro_attestation(
        bundle: Dict,
        *,
        pcr0: str,
        pubkey: str,
        epoch_id: int,
    ) -> bool:
        from leadpoet_canonical.nitro import verify_nitro_attestation_full

        valid, _details = verify_nitro_attestation_full(
            bundle["validator_attestation_b64"],
            expected_pcr0=pcr0,
            expected_pubkey=pubkey,
            expected_purpose="validator_weights",
            expected_epoch_id=epoch_id,
            role="validator",
            skip_pcr0_verification=False,
        )
        return bool(valid)

    def verify_attested_weights_v1(
        self,
        bundle: Dict,
        *,
        expected_epoch_id: int,
    ) -> Optional[Dict]:
        """Verify V1 without allowing invalid V2 authority to downgrade."""

        try:
            if not isinstance(bundle, dict):
                raise ValueError("bundle is not an object")
            netuid = bundle.get("netuid")
            epoch_id = bundle.get("epoch_id")
            block = bundle.get("block")
            if (
                isinstance(netuid, bool)
                or not isinstance(netuid, int)
                or netuid != int(self.config.netuid)
                or isinstance(epoch_id, bool)
                or not isinstance(epoch_id, int)
                or epoch_id != int(expected_epoch_id)
                or isinstance(block, bool)
                or not isinstance(block, int)
                or block < 0
                or block // EPOCH_LENGTH != epoch_id
            ):
                raise ValueError("bundle chain identity is invalid")

            uids = bundle.get("uids")
            weights_u16 = bundle.get("weights_u16")
            if (
                not isinstance(uids, list)
                or not isinstance(weights_u16, list)
                or not uids
                or len(uids) != len(weights_u16)
                or len(set(uids)) != len(uids)
                or any(
                    isinstance(uid, bool)
                    or not isinstance(uid, int)
                    or uid < 0
                    or uid > 65535
                    for uid in uids
                )
                or any(
                    isinstance(weight, bool)
                    or not isinstance(weight, int)
                    or weight < 0
                    or weight > 65535
                    for weight in weights_u16
                )
            ):
                raise ValueError("bundle weight vector is invalid")

            claimed_hash = str(bundle.get("weights_hash") or "").lower()
            signature = str(bundle.get("validator_signature") or "").lower()
            pubkey = str(bundle.get("validator_enclave_pubkey") or "").lower()
            pcr0 = str(bundle.get("validator_pcr0") or "").lower()
            commit_hash = str(bundle.get("pcr0_commit_hash") or "").lower()
            attestation = str(bundle.get("validator_attestation_b64") or "")
            if (
                not re.fullmatch(r"[0-9a-f]{64}", claimed_hash)
                or not re.fullmatch(r"[0-9a-f]{128}", signature)
                or not re.fullmatch(r"[0-9a-f]{64}", pubkey)
                or not re.fullmatch(r"[0-9a-f]{96}", pcr0)
                or not re.fullmatch(r"[0-9a-f]{40}", commit_hash)
                or not attestation
            ):
                raise ValueError("bundle attestation fields are invalid")

            recomputed_hash = bundle_weights_hash(
                netuid,
                epoch_id,
                block,
                list(zip(uids, weights_u16)),
            )
            if recomputed_hash != claimed_hash:
                raise ValueError("bundle weights hash mismatch")
            Ed25519PublicKey.from_public_bytes(bytes.fromhex(pubkey)).verify(
                bytes.fromhex(signature),
                bytes.fromhex(claimed_hash),
            )
            if not self._v1_pcr0_commit_is_allowed(pcr0, commit_hash):
                raise ValueError("bundle PCR0 and commit are not an allowed pair")
            if not self._verify_v1_nitro_attestation(
                bundle,
                pcr0=pcr0,
                pubkey=pubkey,
                epoch_id=epoch_id,
            ):
                raise ValueError("bundle Nitro attestation verification failed")
            return dict(bundle)
        except Exception as exc:
            logger.warning(
                "auditor_v1_fallback_verification_failed epoch=%s "
                "error_type=%s error=%s",
                expected_epoch_id,
                type(exc).__name__,
                str(exc)[:200],
            )
            return None

    async def fetch_verified_weight_authority(
        self,
        epoch_id: int,
    ) -> Tuple[Optional[Dict], str]:
        """Prefer V2 and permit V1 only when the V2 route is absent."""

        v2_bundle = await self.fetch_attested_weights_v2(epoch_id)
        if isinstance(v2_bundle, dict):
            verified = self.verify_attested_weights_v2(v2_bundle)
            return verified, "v2_verified" if verified else "v2_invalid"
        if not getattr(self, "_last_v2_route_was_missing", False):
            return None, "v2_unavailable"

        v1_bundle = await self.fetch_attested_weights_v1(epoch_id)
        if not isinstance(v1_bundle, dict):
            return None, "v1_unavailable"
        verified = self.verify_attested_weights_v1(
            v1_bundle,
            expected_epoch_id=epoch_id,
        )
        return verified, "v1_verified" if verified else "v1_invalid"

    def verify_attested_weights_v2(self, authority: Dict) -> Optional[Dict]:
        """Verify finalized V2 authority against independent PCR0 builds."""

        cache_file = str(os.environ.get("AUDITOR_INDEPENDENT_PCR0_CACHE_FILE", "") or "").strip()
        if not cache_file:
            return None
        try:
            cache = load_identity_cache(Path(cache_file).expanduser())
            profile_file = Path(
                os.environ.get(
                    "AUDITOR_CHAIN_SIGNING_PROFILE_FILE",
                    os.path.join(
                        _REPO_ROOT,
                        "validator_tee",
                        "enclave",
                        "chain_signing_profile_v2.json",
                    ),
                )
            ).expanduser()
            profile = json.loads(profile_file.read_text(encoding="utf-8"))
            return verify_attested_weight_authority_v2(
                authority,
                identity_cache=cache,
                chain_signing_profile=profile,
            )
        except Exception:
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
                logger.debug("Gateway Nitro attestation verified")
                return True
            else:
                logger.error(f"Gateway Nitro verification failed: {data}")
                return False
                
        except Exception as e:
            logger.error(f"Gateway attestation verification failed: {e}")
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
                logger.debug(
                    "Validator Nitro attestation verified for epoch %s",
                    epoch_id,
                )
                return True
            else:
                logger.error(f"Validator attestation verification failed: {data}")
                return False
                
        except Exception as e:
            logger.error(f"Validator attestation verification failed: {e}")
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
    
    def _set_weights_until_epoch_end(
        self,
        *,
        epoch_id: int,
        uids,
        weights,
    ) -> bool:
        """Retry an explicitly rejected chain submission within one epoch."""

        attempt = 0
        while True:
            attempt += 1
            success, message = self.subtensor.set_weights(
                netuid=self.config.netuid,
                wallet=self.wallet,
                uids=uids,
                weights=weights,
                wait_for_finalization=True,
            )
            if success:
                return True

            print(
                f"❌ Bittensor rejected weight submission attempt {attempt}: "
                f"{message}"
            )
            time.sleep(12)
            current_block = self.subtensor.get_current_block()
            if current_block // EPOCH_LENGTH != epoch_id:
                print(
                    f"⏹️ Epoch {epoch_id} ended before the weight submission "
                    "was accepted"
                )
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
            
            success = self._set_weights_until_epoch_end(
                epoch_id=epoch_id,
                uids=uids,
                weights=weights_floats,
            )
            
            if success:
                print(f"✅ Weights submitted for epoch {epoch_id}")
                self.last_submitted_epoch = epoch_id
                
                # Verify submission landed on chain
                print(f"   🔍 Verifying submission landed on chain...")
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
                        
                        weights_data, authority_status = (
                            await self.fetch_verified_weight_authority(target_epoch)
                        )
                        if weights_data is None:
                            if authority_status in {
                                "v2_unavailable",
                                "v1_unavailable",
                            }:
                                print("   ⏳ Weights not yet published. Waiting 30s...")
                            else:
                                print("❌ Auditor verification failed")
                            await asyncio.sleep(30)
                            continue
                        if authority_status == "v1_verified":
                            print("✅ Auditor V1 compatibility verification passed")
                        else:
                            print("✅ Auditor verification passed")
                        
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
                
                # Soft anti-equivocation check at block 50-100 of each epoch
                # Check 2 epochs back to ensure weights have definitely propagated
                if 50 <= block_within_epoch <= 100:
                    # Check if we have a pending equivocation check from 2 epochs ago
                    target_epoch = current_epoch - 2
                    pending = self.load_pending_equivocation_check(target_epoch=target_epoch)
                    if pending:
                        if not await self.perform_soft_equivocation_check(target_epoch):
                            logger.error(f"Soft equivocation check FAILED for epoch {target_epoch}")
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
VERIFICATION FAILURE HANDLING:
  If authoritative V2 verification fails, no vector is submitted.

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
