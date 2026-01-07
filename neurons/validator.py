#!/usr/bin/env python3
# Suppress multiprocessing warnings BEFORE any imports
# Auto-update trigger: 2025-12-12
import os
import sys
os.environ["PYTHONWARNINGS"] = "ignore::UserWarning"

import re
import time
import random
import requests
import textwrap
import numpy as np
import bittensor as bt
import argparse
import json
from datetime import datetime, timedelta, timezone
from Leadpoet.base.validator import BaseValidatorNeuron
from Leadpoet.protocol import LeadRequest
from validator_models.automated_checks import validate_lead_list as auto_check_leads, run_automated_checks, MAX_REP_SCORE
from Leadpoet.base.utils.config import add_validator_args
import threading
from Leadpoet.base.utils import queue as lead_queue
from Leadpoet.base.utils import pool as lead_pool
import asyncio
from typing import List, Dict, Optional
from aiohttp import web
from Leadpoet.utils.cloud_db import (
    fetch_prospects_from_cloud,
    fetch_curation_requests,
    push_curation_result,
    push_miner_curation_request,
    fetch_miner_curation_result,
    push_validator_ranking,
)
# TokenManager removed - JWT system deprecated in favor of TEE gateway
# from Leadpoet.utils.token_manager import TokenManager
from Leadpoet.utils.utils_lead_extraction import (
    get_email,
    get_website,
    get_company,
    get_industry,
    get_role,
    get_sub_industry,
    get_first_name,
    get_last_name,
    get_linkedin,
    get_location,
    get_field
)
from supabase import Client
import socket
from math import isclose
from pathlib import Path
import warnings
import subprocess
import aiohttp

# ════════════════════════════════════════════════════════════════════════════
# TEE SIGNING IMPORTS (Phase 2.3 - Validator TEE Weight Submission)
# ════════════════════════════════════════════════════════════════════════════
# These imports are optional at startup - only used if TEE is enabled
try:
    from validator_tee import (
        initialize_enclave_keypair,
        sign_weights,
        get_enclave_pubkey,
        get_attestation_document_b64,
        get_attestation,
        get_code_hash,
        is_keypair_initialized,
        is_enclave_running,
    )
    from leadpoet_canonical.weights import normalize_to_u16, bundle_weights_hash
    from leadpoet_canonical.binding import create_binding_message
    TEE_AVAILABLE = True
except ImportError as e:
    TEE_AVAILABLE = False
    # Will log warning at runtime if TEE submission is attempted

# Additional warning suppression
warnings.filterwarnings("ignore", message=".*leaked semaphore objects.*")

# ════════════════════════════════════════════════════════════════════════════
# AUTO-UPDATER: Automatically updates entire repo from GitHub for validators
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__" and os.environ.get("LEADPOET_WRAPPER_ACTIVE") != "1":
    print("🔄 Leadpoet Validator: Activating auto-update wrapper...")
    print("   Your validator will automatically stay up-to-date with the latest code")
    print("")
    
    # Create wrapper script path (hidden file with dot prefix)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(script_dir)
    wrapper_path = os.path.join(repo_root, ".auto_update_wrapper.sh") 
    
    # Inline wrapper script - simple and clean
    wrapper_content = '''#!/bin/bash
# Auto-generated wrapper for Leadpoet validator auto-updates
set -e

REPO_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$REPO_ROOT"

echo "════════════════════════════════════════════════════════════════"
echo "🚀 Leadpoet Auto-Updating Validator"
echo "   Repository updates every 5 minutes"
echo "   GitHub: github.com/leadpoet/leadpoet"
echo "════════════════════════════════════════════════════════════════"
echo ""

RESTART_COUNT=0
MAX_RESTARTS=5

while true; do
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
        if git diff HEAD@{1} HEAD --name-only | grep -q "requirements.txt"; then
            echo "📦 requirements.txt changed - updating packages..."
            pip3 install -r requirements.txt --quiet || echo "   ⚠️  Package install failed (continuing anyway)"
        fi
    else
        echo "⏭️  Could not update (offline or not a git repo)"
        echo "   Continuing with current version..."
    fi
    
    echo "────────────────────────────────────────────────────────────────"
    echo "🟢 Starting validator (attempt $(($RESTART_COUNT + 1)))..."
    echo ""
    
    # Run validator with environment flag to prevent wrapper re-execution
    # Suppress multiprocessing semaphore warnings by setting PYTHONWARNINGS
    export LEADPOET_WRAPPER_ACTIVE=1
    export PYTHONWARNINGS="ignore::UserWarning"
    python3 neurons/validator.py "$@"
    
    EXIT_CODE=$?
    
    echo ""
    echo "────────────────────────────────────────────────────────────────"
    
    if [ $EXIT_CODE -eq 0 ]; then
        echo "✅ Validator exited cleanly (exit code: 0)"
        echo "   Shutting down auto-updater..."
        break
    elif [ $EXIT_CODE -eq 137 ] || [ $EXIT_CODE -eq 9 ]; then
        echo "⚠️  Validator was killed (exit code: $EXIT_CODE) - likely Out of Memory"
        echo "   Cleaning up resources before restart..."
        
        # Clean up any leaked resources
        pkill -f "python3 neurons/validator.py" 2>/dev/null || true
        sleep 5  # Give system time to clean up
        
        RESTART_COUNT=$((RESTART_COUNT + 1))
        if [ $RESTART_COUNT -ge $MAX_RESTARTS ]; then
            echo "❌ Maximum restart attempts ($MAX_RESTARTS) reached"
            echo "   Your system may not have enough RAM. Consider:"
            echo "   1. Increasing server RAM"
            echo "   2. Reducing batch sizes in validator config"
            echo "   3. Monitoring memory usage with 'htop'"
            exit 1
        fi
        
        echo "   Restarting in 30 seconds... (attempt $RESTART_COUNT/$MAX_RESTARTS)"
        sleep 30
    else
        RESTART_COUNT=$((RESTART_COUNT + 1))
        echo "⚠️  Validator exited with error (exit code: $EXIT_CODE)"
        
        if [ $RESTART_COUNT -ge $MAX_RESTARTS ]; then
            echo "❌ Maximum restart attempts ($MAX_RESTARTS) reached"
            echo "   Please check logs and restart manually"
            exit 1
        fi
        
        echo "   Restarting in 10 seconds... (attempt $RESTART_COUNT/$MAX_RESTARTS)"
        sleep 10
    fi
    
    echo ""
    echo "⏰ Next update check in 5 minutes..."
    sleep 300
    
    # Reset restart counter after successful check
    RESTART_COUNT=0
done

echo "════════════════════════════════════════════════════════════════"
echo "🛑 Auto-updater stopped"
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
            env["LEADPOET_WRAPPER_ACTIVE"] = "1"
            os.execve(wrapper_path, [wrapper_path] + sys.argv[1:], env)
        except Exception as e:
            print(f"❌ Failed to execute wrapper: {e}")
            print("   Continuing without auto-updates...")

# normal validator code starts below

# ════════════════════════════════════════════════════════════════════════════
# AUTO-CONTAINERIZATION: Automatically containerize if proxies detected
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__" and os.environ.get("LEADPOET_CONTAINER_MODE") != "1":
    # Check if proxies are configured for containerization
    proxies_found = []
    for i in range(1, 50):  # Check for up to 49 proxies (supports scaling)
        proxy_var = f"WEBSHARE_PROXY_{i}"
        proxy_value = os.getenv(proxy_var)
        if proxy_value and proxy_value != "http://YOUR_USERNAME:YOUR_PASSWORD@p.webshare.io:80":
            proxies_found.append((proxy_var, proxy_value))
    
    if proxies_found:
        print("════════════════════════════════════════════════════════════════")
        print("🐳 AUTO-CONTAINERIZATION ACTIVATED")
        print("════════════════════════════════════════════════════════════════")
        print(f"📊 Detected {len(proxies_found)} proxy URLs in environment")
        print(f"   Total containers: {len(proxies_found) + 1} (1 coordinator + {len(proxies_found)} workers)")
        print("")
        print("🔧 Building Docker image and spawning containers...")
        print("   (This may take a few minutes on first run)")
        print("")
        
        # Determine paths
        script_dir = os.path.dirname(os.path.abspath(__file__))
        repo_root = os.path.dirname(script_dir)
        containerizing_dir = os.path.join(repo_root, "validator_models", "containerizing")
        deploy_script = os.path.join(containerizing_dir, "deploy_dynamic.sh")
        
        # Check if deploy script exists
        if not os.path.exists(deploy_script):
            print(f"❌ ERROR: Deploy script not found: {deploy_script}")
            print("   Falling back to non-containerized mode...")
            print("")
        else:
            # Execute deployment script
            try:
                import subprocess
                result = subprocess.run(
                    ["/bin/bash", deploy_script],
                    cwd=containerizing_dir,
                    check=True,
                    capture_output=False
                )
                
                print("")
                print("✅ Containerized deployment complete!")
                print(f"   {len(proxies_found) + 1} validator containers are now running in parallel")
                print(f"   (1 coordinator + {len(proxies_found)} workers)")
                print("")
                print("📺 Following main validator logs...")
                print("   (Press Ctrl+C to detach - containers will keep running)")
                print("════════════════════════════════════════════════════════════════")
                print("")
                
                # Follow main container logs (blocking call)
                try:
                    subprocess.run(
                        ["docker", "logs", "-f", "leadpoet-validator-main"],
                        check=False  # Don't raise exception on Ctrl+C
                    )
                except KeyboardInterrupt:
                    print("")
                    print("════════════════════════════════════════════════════════════════")
                    print("🔌 Detached from logs (containers still running)")
                    print("")
                    print("📋 To reattach: docker logs -f leadpoet-validator-main")
                    print("📊 Check status: docker ps")
                    print("🛑 Stop all: docker stop leadpoet-validator-main leadpoet-validator-worker-1 leadpoet-validator-worker-2")
                    print("════════════════════════════════════════════════════════════════")
                
                sys.exit(0)
                
            except subprocess.CalledProcessError as e:
                print(f"❌ ERROR: Deployment failed with exit code {e.returncode}")
                print("   Falling back to non-containerized mode...")
                print("")
            except Exception as e:
                print(f"❌ ERROR: {e}")
                print("   Falling back to non-containerized mode...")
                print("")

# ════════════════════════════════════════════════════════════════════════════

AVAILABLE_MODELS = [
    "openai/o3-mini:online",                    
    "openai/gpt-4o-mini:online",                 
    "google/gemini-2.5-flash:online",
    "openai/gpt-4o:online",            
]

FALLBACK_MODEL = "openai/gpt-4o:online"   

OPENROUTER_KEY = os.getenv("OPENROUTER_KEY")

def _llm_score_lead(lead: dict, description: str, model: str) -> float:
    """Return a 0-0.5 score for how well this lead fits the buyer description."""
    def _heuristic() -> float:
        d  = description.lower()
        txt = (get_company(lead) + " " + get_industry(lead)).lower()
        overlap = len(set(d.split()) & set(txt.split()))
        return min(overlap * 0.05, 0.5)

    if not OPENROUTER_KEY:
        return _heuristic()

    prompt_system = (
            "You are an expert B2B match-maker.\n"
            "FIRST LINE → JSON ONLY  {\"score\": <float between 0.0 and 0.5>}  (0.0 = bad match ⇢ 0.5 = perfect match)\n"
            "SECOND LINE → ≤40-word reason referencing the single lead.\n"
            "⚠️ Do not go outside the 0.0–0.5 range."
        )

    prompt_user = (
        f"BUYER:\n{description}\n\n"
        f"LEAD:\n"
        f"Company:  {get_company(lead)}\n"
        f"Industry: {get_industry(lead)}\n"
        f"Role:     {get_role(lead)}\n"
        f"Website:  {get_website(lead)}"
    )



    print("\n🛈  VALIDATOR-LLM INPUT ↓")
    print(textwrap.shorten(prompt_user, width=250, placeholder=" …"))

    def _extract(json_plus_reason: str) -> float:
        """Return score from first {...} block; raise if not parsable."""
        txt = json_plus_reason.strip()
        if not txt:
            raise ValueError("Empty response from model")
        
        if txt.startswith("```"):
            txt = txt.strip("`").lstrip("json").strip()
        start, end = txt.find("{"), txt.find("}")
        if start == -1 or end == -1:
            raise ValueError("No JSON object found")
        payload = txt[start:end + 1]
        score = float(json.loads(payload).get("score", 0))
        score = max(0.0, min(score, 0.5))     # <= clamp every time
        print("🛈  VALIDATOR-LLM OUTPUT ↓")
        print(textwrap.shorten(txt, width=250, placeholder="…"))
        return max(0.0, min(score, 0.5))

    def _try(model_name: str) -> float:
        r = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={ "Authorization": f"Bearer {OPENROUTER_KEY}",
                      "Content-Type": "application/json"},
            json={ "model": model_name, "temperature": 0.2,
                   "messages":[{"role":"system","content":prompt_system},
                               {"role":"user","content":prompt_user}]},
            timeout=15)
        r.raise_for_status()
        return _extract(r.json()["choices"][0]["message"]["content"])

    try:
        return _try(model)
    except Exception as e:
        print(f"⚠️  Primary model failed ({model}): {e}")
        print(f"🔄 Trying fallback model: {FALLBACK_MODEL}")

    try:
        time.sleep(1)
        return _try(FALLBACK_MODEL)
    except Exception as e:
        print(f"⚠️  Fallback model failed: {e}")
        print("🛈  VALIDATOR-LLM OUTPUT ↓")
        print("<< no JSON response – all models failed >>")
        return None

def _extract_first_json_array(text: str) -> str:
    """Extract the first complete JSON array from text."""
    import json
    from json.decoder import JSONDecodeError

    start = text.find("[")
    if start == -1:
        raise ValueError("No JSON array found")

    decoder = json.JSONDecoder()
    try:
        obj, end_idx = decoder.raw_decode(text, start)
        return json.dumps(obj)
    except JSONDecodeError:
        end = text.rfind("]")
        if end == -1:
            raise ValueError("No JSON array found")
        return text[start:end+1]

def _llm_score_batch(leads: list[dict], description: str, model: str) -> dict:
    """Score all leads in a single LLM call. Returns dict mapping lead id() -> score (0.0-0.5)."""
    if not leads:
        return {}

    if not OPENROUTER_KEY:
        result = {}
        for lead in leads:
            d = description.lower()
            txt = (get_company(lead) + " " + get_industry(lead)).lower()
            overlap = len(set(d.split()) & set(txt.split()))
            result[id(lead)] = min(overlap * 0.05, 0.5)
        return result

    prompt_system = (
        "You are an expert B2B lead validation specialist performing quality assurance.\n"
        "\n"
        "TASK: Validate and score each lead based on fit with the buyer's ideal customer profile (ICP).\n"
        "\n"
        "SCORING CRITERIA (0.0 - 0.5 scale for consensus aggregation):\n"
        "• 0.45-0.50: Excellent match - company type, industry, and role perfectly align with buyer's ICP\n"
        "• 0.35-0.44: Good match - strong alignment with minor gaps\n"
        "• 0.25-0.34: Fair match - moderate relevance but notable misalignment\n"
        "• 0.15-0.24: Weak match - limited relevance, significant gaps\n"
        "• 0.00-0.14: Poor match - minimal to no relevance to buyer's ICP\n"
        "\n"
        "VALIDATION FACTORS:\n"
        "1. Industry specificity - Does the sub-industry/niche match the buyer's target?\n"
        "2. Business model fit - B2B vs B2C, enterprise vs SMB, SaaS vs services, etc.\n"
        "3. Company signals - Website quality, role seniority, geographic fit\n"
        "4. Buyer intent likelihood - Would this company realistically need the buyer's solution?\n"
        "5. Competitive landscape - Is this company in a position to buy similar offerings?\n"
        "\n"
        "OUTPUT FORMAT: Return ONLY a JSON array with one score per lead:\n"
        '[{"lead_index": 0, "score": <0.0-0.5 float>}, {"lead_index": 1, "score": <0.0-0.5 float>}, ...]\n'
        "\n"
        "⚠️ CRITICAL: Scores must be between 0.0 and 0.5. Be precise and differentiate - avoid giving identical scores.\n"
        "Consider: A generic 'Tech' buyer might target SaaS/AI companies (0.4-0.5) over general IT services (0.2-0.3)."
    )

    lines = [f"BUYER'S IDEAL CUSTOMER PROFILE (ICP):\n{description}\n\n"]
    lines.append(f"LEADS TO VALIDATE ({len(leads)} total):\n")

    for idx, lead in enumerate(leads):
        lines.append(
            f"\nLead #{idx}:\n"
            f"  Company: {get_company(lead, default='Unknown')}\n"
            f"  Industry: {get_industry(lead, default='Unknown')}\n"
            f"  Sub-industry: {get_sub_industry(lead, default='Unknown')}\n"
            f"  Contact Role: {get_role(lead, default='Unknown')}\n"
            f"  Website: {get_website(lead, default='Unknown')}"
        )

    prompt_user = "\n".join(lines)

    print("\n🛈  VALIDATOR-LLM BATCH INPUT ↓")
    print(f"   Scoring {len(leads)} leads in single prompt")
    print(textwrap.shorten(prompt_user, width=300, placeholder=" …"))

    def _try_batch(model_name: str):
        r = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": model_name,
                "temperature": 0.2,
                "messages": [
                    {"role": "system", "content": prompt_system},
                    {"role": "user", "content": prompt_user}
                ]
            },
            timeout=30
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]

    try:
        response_text = _try_batch(model)
    except Exception as e:
        print(f"⚠️  Primary batch model failed ({model}): {e}")
        print(f"🔄 Trying fallback model: {FALLBACK_MODEL}")
        try:
            time.sleep(1)
            response_text = _try_batch(FALLBACK_MODEL)
        except Exception as e2:
            print(f"⚠️  Fallback batch model failed: {e2}")
            print("🛈  VALIDATOR-LLM BATCH OUTPUT ↓")
            print("<< no JSON response – all models failed >>")
            return {id(lead): None for lead in leads}

        # Parse response
    print("🛈  VALIDATOR-LLM BATCH OUTPUT ↓")
    print(textwrap.shorten(response_text, width=300, placeholder=" …"))

    try:
        # Extract JSON array (handles reasoning models like o3-mini)
        txt = response_text.strip()
        if txt.startswith("```"):
            txt = txt.strip("`").lstrip("json").strip()

        # Use robust extraction that handles extra reasoning content
        json_str = _extract_first_json_array(txt)
        scores_array = json.loads(json_str)

        # Map scores back to leads
        result = {}

        for item in scores_array:
            idx = item.get("lead_index")
            score = item.get("score", 0.0)
            if idx is not None and 0 <= idx < len(leads):
                # Clamp to 0.0-0.5 range
                clamped_score = max(0.0, min(score, 0.5))
                result[id(leads[idx])] = clamped_score

        # Fill in any missing leads with None
        for lead in leads:
            if id(lead) not in result:
                result[id(lead)] = None

        print(f"✅ Batch scoring succeeded (model: {model if 'mistralai' not in response_text else 'mistralai/mistral-7b-instruct'})")
        return result

    except Exception as e:
        print(f"⚠️  Failed to parse batch response: {e}")
        # Fallback to heuristic
        result = {}
        for lead in leads:
            d = description.lower()
            txt = (get_company(lead) + " " + get_industry(lead)).lower()
            overlap = len(set(d.split()) & set(txt.split()))
            result[id(lead)] = min(overlap * 0.05, 0.5)
        return result

class Validator(BaseValidatorNeuron):
    def __init__(self, config=None):
        super().__init__(config=config)
        
        # Add async subtensor (initialized later in run())
        # This eliminates memory leaks and HTTP 429 errors from repeated instance creation
        self.async_subtensor = None

        bt.logging.info("Registering validator wallet on network...")
        max_retries = 3
        retry_delay = 5
        for attempt in range(max_retries):
            try:
                self.uid = self.subtensor.get_uid_for_hotkey_on_subnet(
                    hotkey_ss58=self.wallet.hotkey.ss58_address,
                    netuid=self.config.netuid,
                )
                if self.uid is not None:
                    bt.logging.success(f"Validator registered with UID: {self.uid}")
                    break
                else:
                    bt.logging.warning(f"Attempt {attempt + 1}/{max_retries}: Validator not registered on netuid {self.config.netuid}")
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay)
            except Exception as e:
                bt.logging.error(f"Attempt {attempt + 1}/{max_retries}: Failed to set UID: {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
        if self.uid is None:
            bt.logging.warning(f"Validator {self.config.wallet_name}/{self.config.wallet_hotkey} not registered on netuid {self.config.netuid} after {max_retries} attempts")

        self.validator_trust = 0.0
        if self.uid is not None:
            try:
                self.validator_trust = self.metagraph.validator_trust[self.uid].item()
                bt.logging.info(f"📊 Validator trust initialized: {self.validator_trust:.4f}")
            except Exception as e:
                bt.logging.warning(f"Failed to get validator trust: {e}")
                self.validator_trust = 0.0

        bt.logging.info("load_state()")
        self.load_state()

        self.app = web.Application()
        self.app.add_routes([
            web.post('/api/leads', self.handle_api_request),
            web.get('/api/leads/status/{request_id}', self.handle_status_request),
        ])
        
        self.email_regex = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
        self.sample_ratio = 0.2
        self.use_open_source_model = config.get("neuron", {}).get("use_open_source_validator_model", True)

        self.processing_broadcast = False
        self._processed_requests = set()
        
        self.precision = 15.0 
        self.consistency = 1.0  
        self.collusion_flag = 1
        self.reputation = self.precision * self.consistency * self.collusion_flag  
        self.validation_history = []  
        self.trusted_validator = False  
        self.registration_time = datetime.now()  
        self.appeal_status = None  
        
        from Leadpoet.base.utils.pool import initialize_pool
        initialize_pool()

        self.broadcast_mode = False
        self.broadcast_lock = threading.Lock()
        
        # TokenManager removed - JWT system deprecated in favor of TEE gateway (tasks6.md)
        # Validators now authenticate with gateway using wallet signatures + metagraph verification
        # No JWT tokens needed!
        bt.logging.info("🔐 Using TEE gateway authentication (no JWT tokens)")
        
        # Supabase client not needed for main validation flow
        # Validators get leads from TEE gateway via /epoch/{epoch_id}/leads
        self.supabase_url = "https://qplwoislplkcegvdmbim.supabase.co"
        self.supabase_client: Optional[Client] = None
        # Skip Supabase init - not needed for TEE gateway workflow
    
    async def initialize_async_subtensor(self):
        """
        Create single AsyncSubtensor instance at validator startup.
        
        This eliminates memory leaks and HTTP 429 errors from repeated instance creation.
        Call this from run() before entering main validation loop.
        """
        import bittensor as bt
        import os
        
        bt.logging.info(f"🔗 Initializing AsyncSubtensor for network: {self.config.subtensor.network}")
        
        # ════════════════════════════════════════════════════════════
        # PROXY BYPASS FOR ASYNC BITTENSOR WEBSOCKET
        # ════════════════════════════════════════════════════════════
        # Temporarily unset proxy env vars for async Bittensor init
        proxy_env_vars = ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy']
        saved_proxies = {}
        for var in proxy_env_vars:
            if var in os.environ:
                saved_proxies[var] = os.environ[var]
                del os.environ[var]
        
        try:
            # Create async subtensor (single instance for entire lifecycle)
            self.async_subtensor = bt.AsyncSubtensor(network=self.config.subtensor.network)
            
            bt.logging.info(f"✅ AsyncSubtensor initialized")
            bt.logging.info(f"   Endpoint: {self.async_subtensor.chain_endpoint}")
            bt.logging.info(f"   Network: {self.async_subtensor.network}")
        finally:
            # Restore proxy environment variables for API calls
            for var, value in saved_proxies.items():
                os.environ[var] = value
    
    async def get_current_block_async(self) -> int:
        """
        Get current block using async subtensor (NO new instances).
        
        Use this instead of self.subtensor.get_current_block() to avoid memory leaks.
        
        Returns:
            Current block number
        
        Raises:
            Exception: If async_subtensor not initialized
        """
        # ALWAYS use sync subtensor for block queries
        # This avoids WebSocket subscription conflicts from AsyncSubtensor
        # Block queries are frequent (every few seconds) and fast, so sync is preferred
        return self.subtensor.block
    
    def _write_shared_block_file(self, block: int, epoch: int, blocks_into_epoch: int):
        """
        Write current block/epoch info to shared file for worker containers.
        
        This allows workers to check block/epoch without connecting to Bittensor.
        Only coordinator calls this (every 12 seconds).
        """
        import json
        import time
        from pathlib import Path
        
        block_file = Path("validator_weights") / "current_block.json"
        data = {
            "block": block,
            "epoch": epoch,
            "blocks_into_epoch": blocks_into_epoch,
            "timestamp": int(time.time())
        }
        
        try:
            with open(block_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            bt.logging.warning(f"Failed to write shared block file: {e}")
    
    def _read_shared_block_file(self) -> tuple:
        """
        Read current block/epoch info from shared file (for worker containers).
        
        Returns:
            (block, epoch, blocks_into_epoch) tuple
        
        Raises:
            Exception: If file doesn't exist, is too old (>30s), or is corrupted
        """
        import json
        import time
        from pathlib import Path
        
        block_file = Path("validator_weights") / "current_block.json"
        
        if not block_file.exists():
            raise Exception("Shared block file not found (coordinator hasn't written it yet)")
        
        try:
            with open(block_file, 'r') as f:
                data = json.load(f)
            
            # Check if data is stale (>30 seconds old)
            current_time = int(time.time())
            file_age = current_time - data.get("timestamp", 0)
            
            if file_age > 30:
                raise Exception(f"Shared block file is stale ({file_age}s old)")
            
            return (data["block"], data["epoch"], data["blocks_into_epoch"])
        
        except Exception as e:
            raise Exception(f"Failed to read shared block file: {e}")
    
    # _start_block_file_updater() removed - no longer needed
    # Block file is now updated inline in process_gateway_validation_workflow()
    # This eliminates the separate background thread and prevents websocket concurrency issues
    
    async def cleanup_async_subtensor(self):
        """Clean up async subtensor on shutdown."""
        if self.async_subtensor:
            bt.logging.info("🔌 Closing AsyncSubtensor...")
            await self.async_subtensor.close()
            bt.logging.info("✅ AsyncSubtensor closed")
    
    def _init_supabase_client(self):
        """Initialize or refresh Supabase client with current JWT token."""
        try:
            from Leadpoet.utils.cloud_db import get_supabase_client
            
            # Use the centralized client creation function
            # This ensures consistency with miner and other validator operations
            self.supabase_client = get_supabase_client()
            
            if self.supabase_client:
                bt.logging.info("✅ Supabase client initialized for validator")
            else:
                bt.logging.warning("⚠️ No JWT token available for Supabase client")
        except Exception as e:
            bt.logging.error(f"Failed to initialize Supabase client: {e}")
            self.supabase_client = None

    def validate_email(self, email: str) -> bool:
        return bool(self.email_regex.match(email))

    def check_duplicates(self, leads: list) -> set:
        emails = [lead.get('email', '') for lead in leads]
        seen = set()
        duplicates = set(email for email in emails if email in seen or seen.add(email))
        return duplicates

    async def validate_leads(self, leads: list, industry: str = None) -> dict:
        if not leads:
            return {"score": 0.0, "O_v": 0.0}

        # Check if leads already have validation scores
        existing_scores = [lead.get("conversion_score") for lead in leads if lead.get("conversion_score") is not None]
        if existing_scores:
            # If leads already have scores, use the average of existing scores
            avg_score = sum(existing_scores) / len(existing_scores)
            return {"score": avg_score * 100, "O_v": avg_score}

        # Use automated_checks for all validation
        report = await auto_check_leads(leads)
        valid_count = sum(1 for entry in report if entry["status"] == "Valid")
        score = (valid_count / len(leads)) * 100 if leads else 0
        O_v = score / 100.0
        return {"score": score, "O_v": O_v}

    async def run_automated_checks(self, leads: list) -> bool:
        report = await auto_check_leads(leads)
        valid_count = sum(1 for entry in report if entry["status"] == "Valid")
        return valid_count / len(leads) >= 0.9 if leads else False

    async def reputation_challenge(self):
        dummy_leads = [
            {"business": f"Test Business {i}", "email": f"owner{i}@testleadpoet.com", "website": f"https://business{i}.com", "industry": "Tech & AI"}
            for i in range(10)
        ]
        known_score = random.uniform(0.8, 1.0)
        validation = await self.validate_leads(dummy_leads)
        O_v = validation["O_v"]
        if abs(O_v - known_score) <= 0.1:
            bt.logging.info("Passed reputation challenge")
        else:
            self.precision = max(0, self.precision - 10)
            bt.logging.warning(f"Failed reputation challenge, P_v reduced to {self.precision}")
        self.update_reputation()

    def update_consistency(self):
        now = datetime.now()
        periods = {
            "14_days": timedelta(days=14),
            "30_days": timedelta(days=30),
            "90_days": timedelta(days=90)
        }
        J_v = {}
        for period, delta in periods.items():
            start_time = now - delta
            relevant_validations = [v for v in self.validation_history if v["timestamp"] >= start_time]
            if not relevant_validations:
                J_v[period] = 0
                continue
            correct = sum(1 for v in relevant_validations if abs(v["O_v"] - v["F"]) <= 0.1)
            J_v[period] = correct / len(relevant_validations)
        
        self.consistency = 1 + (0.55 * J_v["14_days"] + 0.25 * J_v["30_days"] + 0.2 * J_v["90_days"])
        self.consistency = min(max(self.consistency, 1.0), 2.0)
        bt.logging.debug(f"Updated C_v: {self.consistency}, J_v: {J_v}")

    def update_reputation(self):
        self.reputation = self.precision * self.consistency * self.collusion_flag
        registration_duration = (datetime.now() - self.registration_time).days
        self.trusted_validator = self.reputation > 85 and registration_duration >= 30
        bt.logging.debug(f"Updated R_v: {self.reputation}, Trusted: {self.trusted_validator}")

    async def handle_buyer_feedback(self, leads: list, feedback_score: float):
        feedback_map = {
            (0, 1): (-20, 0.0),
            (1, 5): (-10, 0.2),
            (5, 7): (1, 0.5),
            (7, 8): (5, 0.7),
            (8, 9): (8, 0.9),
            (9, float('inf')): (15, 1.0)
        }
        for (low, high), (p_adj, f_new) in feedback_map.items():
            if low < feedback_score <= high:
                self.precision = max(0, min(100, self.precision + p_adj))
                for validation in self.validation_history:
                    if validation["leads"] == leads:
                        validation["F"] = f_new
                bt.logging.info(f"Applied buyer feedback B={feedback_score}: P_v={self.precision}, F={f_new}")
                break
        self.update_reputation()

    async def submit_appeal(self):
        if self.collusion_flag == 1:
            bt.logging.info("No collusion flag to appeal")
            return
        self.appeal_status = {"votes": [], "start_time": datetime.now()}
        bt.logging.info("Collusion flag appeal submitted")

    async def vote_on_appeal(self, validator_hotkey: str, vote: int):
        if self.appeal_status is None or self.appeal_status != "pending":
            bt.logging.warning("No active appeal to vote on")
            return
        weight = {90: 5, 80: 3, 70: 2, 0: 1}.get(next(k for k in [90, 80, 70, 0] if self.precision > k), 1)
        self.appeal_status["votes"].append({"hotkey": validator_hotkey, "E_v": vote, "H_v": weight})
        bt.logging.debug(f"Vote submitted: E_v={vote}, H_v={weight}")

    async def resolve_appeal(self):
        if self.appeal_status is None or (datetime.now() - self.appeal_status["start_time"]).days < 7:
            return
        votes = self.appeal_status["votes"]
        if not votes:
            self.collusion_flag = 0
            bt.logging.warning("Appeal failed: No votes received")
        else:
            K_v_sum = sum(v["E_v"] * v["H_v"] for v in votes)
            H_v_sum = sum(v["H_v"] for v in votes)
            if K_v_sum / H_v_sum > 0.66:
                self.collusion_flag = 1
                bt.logging.info("Appeal approved: Collusion flag removed")
            else:
                self.collusion_flag = 0
                bt.logging.warning("Appeal denied")
        self.appeal_status = None
        self.update_reputation()

# ------------------------------------------------------------------+
#  Buyer → validator  (runs once per API call, not in a loop)       +
# ------------------------------------------------------------------+
    async def forward(self, synapse: LeadRequest) -> LeadRequest:
        """
        Respond to a buyer's LeadRequest arriving over Bittensor.
        Delegates to miners for curation, then ranks the results.
        """
        print(f"\n🟡 RECEIVED QUERY from buyer: {synapse.num_leads} leads | "
              f"desc='{synapse.business_desc[:40]}…'")

        # Always refresh metagraph just before selecting miners so we don't use stale flags.
        try:
            self.metagraph.sync(subtensor=self.subtensor)
            print("🔄 Metagraph refreshed for miner selection.")
        except Exception as e:
            print(f"⚠️  Metagraph refresh failed (continuing with cached state): {e}")

        # build the FULL list of miner axons (exclude validators)
        # IMPORTANT: Follow user's semantics:
        # - ACTIVE == True → validator (exclude)
        # - ACTIVE == False → miner (include)
        # Also require is_serving == True.
        active_flags = getattr(self.metagraph, "active", [False] * self.metagraph.n)
        vperm_flags  = getattr(self.metagraph, "validator_permit", [False] * self.metagraph.n)
        print("DBG flags:", {
            "n": self.metagraph.n,
            "serving": [bool(self.metagraph.axons[u].is_serving) for u in range(self.metagraph.n)],
            "active":  [bool(active_flags[u]) for u in range(self.metagraph.n)],
            "vperm":   [bool(vperm_flags[u]) for u in range(self.metagraph.n)],
        })
        my_uid = getattr(self, "uid", None)
        miner_uids = [
            uid for uid in range(self.metagraph.n)
            if getattr(self.metagraph.axons[uid], "is_serving", False)
            and uid != my_uid   # exclude the validator itself
        ]
        axons = [self.metagraph.axons[uid] for uid in miner_uids]

        print(f"🔍 Found {len(miner_uids)} active miners: {miner_uids}")
        print(f"🔍 Axon status: {[self.metagraph.axons[uid].is_serving for uid in miner_uids]}")
        if miner_uids:
            endpoints = [f"{self.metagraph.axons[uid].ip}:{self.metagraph.axons[uid].port}" for uid in miner_uids]
            print(f"🔍 Miner endpoints: {endpoints}")
            my_pub_ip = None
            try:
                if my_uid is not None:
                    my_pub_ip = getattr(self.metagraph.axons[my_uid], "ip", None)
            except Exception:
                pass

            for uid in miner_uids:
                ax = self.metagraph.axons[uid]
                if ax.ip == my_pub_ip:
                    print(f"🔧 Hairpin bypass for UID {uid}: {ax.ip} → 127.0.0.1")
                    ax.ip = "127.0.0.1"

        all_miner_leads: list = []

        print("\n─────────  VALIDATOR ➜ DENDRITE  ─────────")
        print(f"📡  Dialing {len(axons)} miners: {[f'UID{u}' for u in miner_uids]}")
        print(f"⏱️   at {datetime.utcnow().isoformat()} UTC")

        _t0 = time.time()
        miner_req = LeadRequest(num_leads=synapse.num_leads,
                                business_desc=synapse.business_desc)

        responses_task = asyncio.create_task(self.dendrite(
            axons       = axons,
            synapse     = miner_req,
            timeout     = 85,
            deserialize = False,
        ))
        responses = await responses_task
        print(f"⏲️  Dendrite completed in {(time.time() - _t0):.2f}s, analysing responses…")
        for uid, resp in zip(miner_uids, responses):
            if isinstance(resp, LeadRequest):
                sc = getattr(resp.dendrite, "status_code", None)
                sm = getattr(resp.dendrite, "status_message", None)
                pl = len(getattr(resp, "leads", []) or [])
                print(f"📥 UID {uid} dendrite status={sc} msg={sm} leads={pl}")
                if resp.leads:
                    all_miner_leads.extend(resp.leads)
            else:
                print(f"❌ UID {uid}: unexpected response type {type(resp).__name__} → {repr(resp)[:80]}")
        print("─────────  END DENDRITE BLOCK  ─────────\n")

        if not all_miner_leads:
            print("⚠️  Axon unreachable – falling back to cloud broker")
            for target_uid in miner_uids:
                req_id = push_miner_curation_request(
                    self.wallet,
                    {
                        "num_leads":      synapse.num_leads,
                        "business_desc":  synapse.business_desc,
                        "target_uid":     int(target_uid),
                    },
                )
                print(f"📤 Sent curation request to Cloud-Run for UID {target_uid}: {req_id}")

            # Wait for miner response via Cloud-Run
            MAX_ATTEMPTS = 40      # 40 × 5 s  = 200 s
            SLEEP_SEC    = 5
            total_wait   = MAX_ATTEMPTS * SLEEP_SEC
            print(f"⏳ Waiting for miner response (up to {total_wait} s)…")

            expected_miners = len(miner_uids)  # Number of miners we sent requests to
            received_responses = 0
            first_response_time = None
            
            for attempt in range(MAX_ATTEMPTS):
                res = fetch_miner_curation_result(self.wallet)
                if res and res.get("leads"):
                    # Collect from multiple miners
                    all_miner_leads.extend(res["leads"])
                    received_responses += 1
                    
                    # Track when we got the first response
                    if received_responses == 1:
                        first_response_time = attempt
                        print(f"✅ Received first response ({len(res['leads'])} leads) from Cloud-Run")
                        
                        # If expecting multiple miners, wait additional 30s for others
                        if expected_miners > 1:
                            print(f"⏳ Waiting additional 30s for {expected_miners - 1} more miners...")
                    else:
                        print(f"✅ Received response {received_responses}/{expected_miners} with {len(res['leads'])} leads")
                    
                    # Exit conditions:
                    # 1. Got all expected responses
                    if received_responses >= expected_miners:
                        print(f"✅ Received all {expected_miners} responses from miners")
                        break
                    
                    # 2. Got first response and waited 30s (6 attempts) for others
                    elif first_response_time is not None and (attempt - first_response_time) >= 6:
                        print(f"⏰ 30s timeout reached, proceeding with {received_responses}/{expected_miners} responses")
                        break
                
                time.sleep(SLEEP_SEC)
            
            if received_responses > 0:
                print(f"📊 Final collection: {len(all_miner_leads)} leads from {received_responses}/{expected_miners} miners")
            else:
                print("❌ No responses received from any miner via Cloud-Run")

        # Rank leads using LLM scoring (TWO rounds with BATCHING)
        if all_miner_leads:
            print(f"🔍 Ranking {len(all_miner_leads)} leads with LLM...")
            scored_leads = []
            
            aggregated = {id(lead): 0.0 for lead in all_miner_leads}
            failed_leads = set()
            first_model = random.choice(AVAILABLE_MODELS)
            print(f"🔄 LLM round 1/2 (model: {first_model})")
            batch_scores_r1 = _llm_score_batch(all_miner_leads, synapse.business_desc, first_model)
            for lead in all_miner_leads:
                score = batch_scores_r1.get(id(lead))
                if score is None:
                    failed_leads.add(id(lead))
                    print("⚠️  LLM failed for lead, will skip this lead")
                else:
                    aggregated[id(lead)] += score
            
            # ROUND 2: Second LLM scoring (BATCHED, random model selection)
            # Only score leads that didn't fail in round 1
            leads_for_r2 = [lead for lead in all_miner_leads if id(lead) not in failed_leads]
            if leads_for_r2:
                second_model = random.choice(AVAILABLE_MODELS)
                print(f"🔄 LLM round 2/2 (model: {second_model})")
                batch_scores_r2 = _llm_score_batch(leads_for_r2, synapse.business_desc, second_model)
                for lead in leads_for_r2:
                    score = batch_scores_r2.get(id(lead))
                    if score is None:
                        failed_leads.add(id(lead))
                        print("⚠️  LLM failed for lead, will skip this lead")
                    else:
                        aggregated[id(lead)] += score
            
            # Apply aggregated scores to leads (skip failed ones)
            for lead in all_miner_leads:
                if id(lead) not in failed_leads:
                    lead["intent_score"] = round(aggregated[id(lead)], 3)
                    scored_leads.append(lead)

            if not scored_leads:
                print("❌ All leads failed LLM scoring - check your OPENROUTER_KEY environment variable!")
                print("   Set it with: export OPENROUTER_KEY='your-key-here'")
                synapse.leads = []
                synapse.dendrite.status_code = 500
                return synapse

            # Sort by aggregated intent_score and take top N
            scored_leads.sort(key=lambda x: x["intent_score"], reverse=True)
            top_leads = scored_leads[:synapse.num_leads]

            print(f"✅ Ranked top {len(top_leads)} leads:")
            for i, lead in enumerate(top_leads, 1):
                business = get_company(lead, default='Unknown')
                score = lead.get('intent_score', 0)
                print(f"  {i}. {business} (score={score:.3f})")

            # Add c_validator_hotkey to leads being sent to client via Bittensor
            for lead in top_leads:
                lead["c_validator_hotkey"] = self.wallet.hotkey.ss58_address

            synapse.leads = top_leads
        else:
            print("❌ No leads received from any source")
            synapse.leads = []

        synapse.dendrite.status_code = 200
        return synapse

    async def _post_process_with_checks(self, rewards: np.ndarray, miner_uids: list, responses: list):
        validators = [self]
        validator_scores = []
        trusted_validators = [v for v in validators if v.trusted_validator]
        
        for i, response in enumerate(responses):
            if not isinstance(response, LeadRequest) or not response.leads:
                bt.logging.warning(f"Skipping invalid response from UID {miner_uids[i]}")
                continue
            validation = await self.validate_leads(response.leads, industry=response.industry)
            O_v = validation["O_v"]
            validator_scores.append({"O_v": O_v, "R_v": self.reputation, "leads": response.leads})
        
        trusted_low_scores = sum(1 for v in trusted_validators for s in validator_scores if v == self and s["O_v"] < 0.8)
        trusted_rejections = sum(1 for v in trusted_validators for s in validator_scores if v == self and s["O_v"] == 0)
        use_trusted = trusted_low_scores / len(trusted_validators) > 0.67 if trusted_validators else False
        reject = trusted_rejections / len(trusted_validators) > 0.5 if trusted_validators else False
        
        if reject:
            bt.logging.info("Submission rejected by >50% trusted validators")
            return
        
        Rs_total = sum(s["R_v"] for s in validator_scores if s["R_v"] > 15)
        F = sum(s["O_v"] * (s["R_v"] / Rs_total) for s in validator_scores if s["R_v"] > 15) if Rs_total > 0 else 0
        if use_trusted:
            trusted_scores = [s for s in validator_scores if any(v == self and v.trusted_validator for v in validators)]
            Rs_total_trusted = sum(s["R_v"] for s in trusted_scores if s["R_v"] > 15)
            F = sum(s["O_v"] * (s["R_v"] / Rs_total_trusted) for s in trusted_scores if s["R_v"] > 15) if Rs_total_trusted > 0 else 0
        
        for s in validator_scores:
            if abs(s["O_v"] - F) <= 0.1:
                self.precision = min(100, self.precision + 10)
            elif s["O_v"] > 0 and not await self.run_automated_checks(s["leads"]):
                self.precision = max(0, self.precision - 15)
            self.validation_history.append({"O_v": s["O_v"], "F": F, "timestamp": datetime.now(), "leads": s["leads"]})
        
        self.update_consistency()
        self.update_reputation()
        
        for i, (reward, response) in enumerate(zip(rewards, responses)):
            if reward >= 0.9 and isinstance(response, LeadRequest) and response.leads:
                if await self.run_automated_checks(response.leads):
                    from Leadpoet.base.utils.pool import add_to_pool
                    add_to_pool(response.leads)
                    bt.logging.info(f"Added {len(response.leads)} leads from UID {miner_uids[i]} to pool")
                else:
                    self.precision = max(0, self.precision - 15)
                    bt.logging.warning(f"Post-approval check failed for UID {miner_uids[i]}, P_v reduced: {self.precision}")
        
        if random.random() < 0.1:
            await self.reputation_challenge()

        # Reward bookkeeping for delivered leads is handled in the main
        # `run_validator` validation loop, so nothing to do here.

    def save_state(self):
        bt.logging.info("Saving validator state.")
        
        try:
            # Save everything to validator_weights/ directory for consistency
            weights_dir = Path("validator_weights")
            weights_dir.mkdir(exist_ok=True)
            
            # Save validator state (numpy)
            state_path = weights_dir / "validator_state.npz"
            
            np.savez(
                state_path,
                step=self.step,
                scores=self.scores,
                hotkeys=self.hotkeys,
                precision=self.precision,
                consistency=self.consistency,
                collusion_flag=self.collusion_flag,
                reputation=self.reputation,
                validation_history=np.array(self.validation_history, dtype=object),
                registration_time=np.datetime64(self.registration_time),
                appeal_status=self.appeal_status
            )
            bt.logging.info(f"✅ State saved to {state_path}")
            
            # Save pending reveals separately (JSON-serializable) to validator_weights/
            reveals_path = weights_dir / "pending_reveals.json"
                
            if hasattr(self, '_pending_reveals') and self._pending_reveals:
                import json
                with open(reveals_path, 'w') as f:
                    json.dump(self._pending_reveals, f, indent=2)
                bt.logging.info(f"Saved {len(self._pending_reveals)} pending reveal epoch(s) to {reveals_path}")
            elif hasattr(self, '_pending_reveals'):
                # Save empty dict if no pending reveals (clean slate)
                import json
                with open(reveals_path, 'w') as f:
                    json.dump({}, f, indent=2)
                bt.logging.debug(f"Saved empty pending reveals to {reveals_path}")
        except Exception as e:
            bt.logging.error(f"Failed to save state: {e}")
            bt.logging.error(f"   Attempted path: {state_path if 'state_path' in locals() else 'unknown'}")

    def load_state(self):
        # Load from validator_weights/ directory (new location)
        weights_dir = Path("validator_weights")
        state_path = weights_dir / "validator_state.npz"
        
        if state_path.exists():
            bt.logging.info("Loading validator state.")
            try:
                state = np.load(state_path, allow_pickle=True)
                self.step = state["step"]
                self.scores = state["scores"]
                self.hotkeys = state["hotkeys"]
                self.precision = state["precision"]
                self.consistency = state["consistency"]
                self.collusion_flag = state["collusion_flag"]
                self.reputation = state["reputation"]
                self.validation_history = state["validation_history"].tolist()
                self.registration_time = datetime.fromtimestamp(state["registration_time"].astype('datetime64[ns]').item() / 1e9)
                self.appeal_status = state["appeal_status"].item()
                bt.logging.info(f"✅ Loaded state from {state_path}")
            except Exception as e:
                bt.logging.warning(f"Failed to load state: {e}. Using defaults.")
                self._initialize_default_state()
        else:
            bt.logging.info("No state file found. Initializing with defaults.")
            self._initialize_default_state()
        
        # Load pending reveals separately from validator_weights/
        reveals_path = weights_dir / "pending_reveals.json"
        if reveals_path.exists():
            try:
                import json
                with open(reveals_path, 'r') as f:
                    self._pending_reveals = json.load(f)
                # Convert string keys to integers (JSON converts int keys to strings)
                self._pending_reveals = {int(k): v for k, v in self._pending_reveals.items()}
                bt.logging.info(f"✅ Loaded {len(self._pending_reveals)} pending reveal epoch(s) from {reveals_path}")
            except Exception as e:
                bt.logging.warning(f"Failed to load pending reveals: {e}")
                self._pending_reveals = {}
        else:
            bt.logging.info("No pending reveals file found. Starting fresh.")
            self._pending_reveals = {}

    def _initialize_default_state(self):
        self.step = 0
        self.scores = np.zeros(self.metagraph.n, dtype=np.float32)
        self.hotkeys = self.metagraph.hotkeys.copy()
        self.precision = 15.0
        self.consistency = 1.0
        self.collusion_flag = 1
        self.reputation = self.precision * self.consistency * self.collusion_flag
        self.validation_history = []
        self.registration_time = datetime.now()
        self.appeal_status = None
        self.trusted_validator = False
        self._pending_reveals = {}
    
    def _save_pending_reveals(self):
        """
        Save pending reveals to disk immediately.
        
        This is called after cleanup operations to persist state changes
        without waiting for the next full save_state() call.
        """
        try:
            weights_dir = Path("validator_weights")
            weights_dir.mkdir(exist_ok=True)
            reveals_path = weights_dir / "pending_reveals.json"
            
            with open(reveals_path, 'w') as f:
                json.dump(self._pending_reveals, f, indent=2)
            
            bt.logging.debug(f"💾 Saved pending reveals to {reveals_path}")
        except Exception as e:
            bt.logging.error(f"Failed to save pending reveals: {e}")

    async def handle_api_request(self, request):
        """
        Handle API requests from clients using broadcast mechanism.

        Flow:
        1. Broadcast request to all validators/miners via Firestore
        2. Return request_id immediately to client
        3. Client polls /api/leads/status/{request_id} for results
        """
        try:
            data = await request.json()
            num_leads     = data.get("num_leads", 1)
            business_desc = data.get("business_desc", "")
            client_id     = data.get("client_id", "unknown")

            print(f"\n🔔 RECEIVED API QUERY from client: {num_leads} leads | desc='{business_desc[:10]}…'")
            bt.logging.info("📡 Broadcasting to ALL validators and miners via Firestore...")

            # Broadcast the request to all validators and miners
            try:
                from Leadpoet.utils.cloud_db import broadcast_api_request

                # FIX: Wrap synchronous broadcast call to prevent blocking
                request_id = await asyncio.to_thread(
                    broadcast_api_request,
                    wallet=self.wallet,
                    num_leads=num_leads,
                    business_desc=business_desc,
                    client_id=client_id
                )

                print(f"📡 Broadcast API request {request_id[:8]}... to subnet")
                bt.logging.info(f"📡 Broadcast API request {request_id[:8]}... to subnet")

                # Return request_id immediately - client will poll for results
                return web.json_response({
                    "request_id": request_id,
                    "status": "processing",
                    "message": "Request broadcast to subnet. Poll /api/leads/status/{request_id} for results.",
                    "poll_url": f"/api/leads/status/{request_id}",
                    "status_code": 202,
                }, status=202)

            except Exception as e:
                print(f"❌ Failed to broadcast request: {e}")
                bt.logging.error(f"Failed to broadcast request: {e}")

                # Fallback to old direct method if broadcast fails
                return web.json_response({
                    "leads": [],
                    "status_code": 500,
                    "status_message": f"Failed to broadcast request: {str(e)}",
                    "process_time": "0"
                }, status=500)

        except Exception as e:
            print(f"❌ Error handling API request: {e}")
            bt.logging.error(f"Error handling API request: {e}")
            return web.json_response({
                "leads": [],
                "status_code": 500,
                "status_message": f"Error: {str(e)}",
                "process_time": "0"
            }, status=500)

    async def handle_status_request(self, request):
        """Handle status polling requests - returns quickly for test requests."""
        try:
            request_id = request.match_info.get('request_id')

            # Quick return for port discovery tests
            if request_id == "test":
                return web.json_response({
                    "status": "ok",
                    "request_id": "test"
                })

            # Fetch validator rankings from Firestore
            from Leadpoet.utils.cloud_db import fetch_validator_rankings, get_broadcast_status

            # Get broadcast request status
            status_data = get_broadcast_status(request_id)

            # Fetch all validator rankings for this request
            validator_rankings = fetch_validator_rankings(request_id, timeout_sec=2)

            # Determine if timeout reached (check if request is older than 90 seconds)
            from datetime import datetime, timezone
            request_time = status_data.get("created_at", "")
            timeout_reached = False
            if request_time:
                try:
                    # Parse ISO timestamp
                    req_dt = datetime.fromisoformat(request_time.replace('Z', '+00:00'))
                    elapsed = (datetime.now(timezone.utc) - req_dt).total_seconds()
                    timeout_reached = elapsed > 90
                except Exception:
                    pass

            # Return data matching API client's expected format
            return web.json_response({
                "request_id": request_id,
                "status": status_data.get("status", "processing"),
                "validator_rankings": validator_rankings,
                "validators_submitted": len(validator_rankings),
                "timeout_reached": timeout_reached,
                "num_validators_responded": len(validator_rankings),  # Keep for backward compat
                "leads": status_data.get("leads", []),
                "metadata": status_data.get("metadata", {}),
            })

        except Exception as e:
            bt.logging.error(f"Error in handle_status_request: {e}")
            import traceback
            bt.logging.error(traceback.format_exc())
            return web.json_response({
                "request_id": request_id,
                "status": "error",
                "error": str(e),
                "validator_rankings": [],
                "validators_submitted": 0,
                "timeout_reached": False,
                "leads": [],
            }, status=500)

    def check_port_availability(self, port: int) -> bool:
        """Check if a port is available for binding."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(('0.0.0.0', port))
                return True
            except socket.error:
                return False

    def find_available_port(self, start_port: int, max_attempts: int = 10) -> int:
        """Find an available port starting from start_port."""
        port = start_port
        for _ in range(max_attempts):
            if self.check_port_availability(port):
                return port
            port += 1
        raise RuntimeError(f"No available ports found between {start_port} and {start_port + max_attempts - 1}")

    async def start_http_server(self):
        """Start HTTP server for API requests."""
        runner = web.AppRunner(self.app)
        await runner.setup()

        # Find available port
        port = self.find_available_port(8093)
        site = web.TCPSite(runner, '0.0.0.0', port)
        await site.start()
        bt.logging.info(f"🔴 Validator HTTP server started on port {port}")
        return port

    def run(self):
        """Override the base run method to not run continuous validation"""
        self.sync()

        # Check if validator is properly registered
        if not hasattr(self, 'uid') or self.uid is None:
            bt.logging.error("Cannot run validator: UID not set. Please register the wallet on the network.")
            return

        print(f"Running validator for subnet: {self.config.netuid} on network: {self.subtensor.chain_endpoint}")
        print(f"🔍 Validator UID: {self.uid}")
        print(f"🔍 Validator hotkey: {self.wallet.hotkey.ss58_address}")

        # Build the axon with the correct port
        self.axon = bt.axon(
            wallet=self.wallet,
            ip      = "0.0.0.0",
            port    = self.config.axon.port,
            external_ip   = self.config.axon.external_ip,
            external_port = self.config.axon.external_port,
        )
        # expose buyer-query endpoint (LeadRequest → LeadRequest)
        self.axon.attach(self.forward)
        # Defer on-chain publish/start to run() to avoid double-serve hangs.
        print("───────────────────────────────────────────")
        # publish endpoint as PLAINTEXT so validators use insecure gRPC
        self.subtensor.serve_axon(
            netuid = self.config.netuid,
            axon   = self.axon,
        )
        print("✅ Axon published on-chain (plaintext)")
        self.axon.start()
        print("   Axon started successfully!")
        # Post-start visibility
        print(f"🖧  Local gRPC listener  : 0.0.0.0:{self.config.axon.port}")
        print(f"🌐  External endpoint   : {self.config.axon.external_ip}:{self.config.axon.external_port}")
        print("───────────────────────────────────────────")

        # Start HTTP server in background thread with dedicated event loop
        print("🔴 Starting HTTP server for REST API...")

        http_port_container = [None]  # Use list to share value between threads

        def run_http_server():
            """Run HTTP server in a dedicated event loop."""
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            async def start_and_serve():
                """Start server and keep it alive."""
                runner = web.AppRunner(self.app)
                await runner.setup()

                # Find available port
                port = self.find_available_port(8093)
                site = web.TCPSite(runner, '0.0.0.0', port)
                await site.start()

                http_port_container[0] = port  # Share port with main thread

                print(f"✅ HTTP server started on port {port}")
                print(f"📡 API endpoint: http://localhost:{port}/api/leads")
                print("───────────────────────────────────────────")

                # Keep the server running by awaiting an event that never completes
                # This is the proper way to keep an aiohttp server alive
                stop_event = asyncio.Event()
                await stop_event.wait()  # Wait forever

            try:
                # Run the server - this will block forever until KeyboardInterrupt
                loop.run_until_complete(start_and_serve())
            except KeyboardInterrupt:
                print("🛑 HTTP server shutting down...")
            except Exception as e:
                print(f"❌ HTTP server error: {e}")
                import traceback
                traceback.print_exc()
            finally:
                loop.close()

        # Start HTTP server in background thread
        http_thread = threading.Thread(target=run_http_server, daemon=True)
        http_thread.start()

        # Wait for server to start and get port
        for _ in range(50):  # Wait up to 5 seconds
            if http_port_container[0] is not None:
                break
            time.sleep(0.1)

        if http_port_container[0] is None:
            print("❌ HTTP server failed to start!")
        else:
            print(f"✅ HTTP server confirmed running on port {http_port_container[0]}")

        # Start broadcast polling loop in background thread
        def run_broadcast_polling():
            """Run broadcast polling in its own async event loop"""
            print("🟢 Broadcast polling thread started!")
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            async def polling_loop():
                print("🟢 Broadcast polling loop initialized!")
                while not self.should_exit:
                    try:
                        await self.process_broadcast_requests_continuous()
                    except Exception as e:
                        bt.logging.error(f"Error in broadcast polling: {e}")
                        import traceback
                        bt.logging.error(traceback.format_exc())
                        await asyncio.sleep(5)  # Wait before retrying

            try:
                loop.run_until_complete(polling_loop())
            except KeyboardInterrupt:
                bt.logging.info("🛑 Broadcast polling shutting down...")
            except Exception as e:
                print(f"❌ Broadcast polling error: {e}")
                import traceback
                traceback.print_exc()
            finally:
                loop.close()

        # Start broadcast polling in background thread
        broadcast_thread = threading.Thread(target=run_broadcast_polling, daemon=True, name="BroadcastPolling")
        broadcast_thread.start()
        # ══════════════════════════════════════════════════════════════════

        print(f"Validator starting at block: {self.block}")
        print("✅ Validator is now serving on the Bittensor network")
        print("   Processing sourced leads and waiting for client requests...")

        # Show available miners
        self.discover_miners()

        # ═══════════════════════════════════════════════════════════════
        # ASYNC MAIN LOOP: Initialize async subtensor and run async workflow
        # ═══════════════════════════════════════════════════════════════
        async def run_async_main_loop():
            """
            Async main validator loop.
            
            Uses async subtensor with block subscription for WebSocket health.
            """
            # Initialize async subtensor (single instance for entire lifecycle)
            await self.initialize_async_subtensor()
            
            # Inject into reward module
            try:
                from Leadpoet.validator import reward
                from Leadpoet.utils import cloud_db
                
                reward.inject_async_subtensor(self.async_subtensor)
                cloud_db._VERIFY.inject_async_subtensor(self.async_subtensor)
                
                bt.logging.info("✅ AsyncSubtensor injected into reward and cloud_db modules")
            except Exception as e:
                bt.logging.warning(f"Failed to inject async subtensor: {e}")
            
            # ════════════════════════════════════════════════════════════
            # BLOCK SUBSCRIPTION: Keep WebSocket alive (prevents HTTP 429)
            # ════════════════════════════════════════════════════════════
            stop_event = asyncio.Event()
            
            async def block_callback(obj: dict):
                """Callback for new blocks (keeps WebSocket alive)."""
                if stop_event.is_set():
                    return True  # Stop subscription
                
                # Just log block number (no processing needed)
                # The subscription itself is what keeps WebSocket alive
                try:
                    block_number = obj["header"]["number"]
                    bt.logging.debug(f"📦 Block #{block_number} received (WebSocket alive)")
                except Exception as e:
                    bt.logging.debug(f"Block callback error: {e}")
                
                return None  # Continue subscription
            
            # Start block subscription in background (keeps WebSocket alive)
            bt.logging.info("🔔 Starting block subscription to keep WebSocket alive...")
            subscription_task = asyncio.create_task(
                self.async_subtensor.substrate.subscribe_block_headers(
                    subscription_handler=block_callback,
                    finalized_only=True
                )
            )
            bt.logging.info("✅ Block subscription started (WebSocket will stay alive)")
            
            # ════════════════════════════════════════════════════════════
            # SHARED BLOCK FILE UPDATER: For worker containers
            # ════════════════════════════════════════════════════════════
            # Block file is now updated inline in process_gateway_validation_workflow()
            # (No separate background thread needed - eliminates websocket concurrency)
            
            try:
                # Keep the validator running and continuously process leads
                while not self.should_exit:
                    # Process gateway validation workflow (TEE-based, now async)
                    try:
                        await self.process_gateway_validation_workflow()
                    except Exception as e:
                        bt.logging.warning(f"Error in gateway validation workflow: {e}")
                        await asyncio.sleep(5)  # Wait before retrying
                    
                    # Check if we should submit accumulated weights (block 345+)
                    try:
                        await self.submit_weights_at_epoch_end()
                    except Exception as e:
                        bt.logging.warning(f"Error in submit_weights_at_epoch_end: {e}")
                    
                    try:
                        self.process_curation_requests_continuous()
                    except Exception as e:
                        bt.logging.warning(f"Error in process_curation_requests_continuous: {e}")
                        await asyncio.sleep(5)  # Wait before retrying

                    # process_broadcast_requests_continuous() runs in background thread

                    # Sync less frequently to avoid websocket concurrency issues
                    # Only sync every 10 iterations (approx every 10 seconds)
                    if not hasattr(self, '_sync_counter'):
                        self._sync_counter = 0

                    self._sync_counter += 1
                    if self._sync_counter >= 10:
                        try:
                            self.sync()
                            self._sync_counter = 0
                        except Exception as e:
                            bt.logging.warning(f"Sync error (will retry): {e}")
                            # Don't crash on sync errors, just skip this sync
                            self._sync_counter = 0

                    await asyncio.sleep(1)  # Small delay to prevent tight loop
                    
            except KeyboardInterrupt:
                self.axon.stop()
                bt.logging.success("Validator killed by keyboard interrupt.")
                exit()
            except Exception as e:
                bt.logging.error(f"Critical error in validator main loop: {e}")
                import traceback
                bt.logging.error(traceback.format_exc())
                # Continue running instead of crashing
                await asyncio.sleep(10)  # Wait longer before retrying main loop
            finally:
                # Stop block subscription
                bt.logging.info("🛑 Stopping block subscription...")
                stop_event.set()
                subscription_task.cancel()
                try:
                    await subscription_task
                except asyncio.CancelledError:
                    pass
                bt.logging.info("✅ Block subscription stopped")
                
                # Cleanup async subtensor on exit
                await self.cleanup_async_subtensor()
        
        # Run async main loop
        try:
            asyncio.run(run_async_main_loop())
        except KeyboardInterrupt:
            bt.logging.success("Validator killed by keyboard interrupt.")
            exit()
        except Exception as e:
            bt.logging.error(f"Fatal error in async main loop: {e}")
            import traceback
            bt.logging.error(traceback.format_exc())

    # Add this method after the run() method (around line 1195)

    def sync(self):
        """
        Override sync to refresh validator trust after metagraph sync.

        This ensures we always have up-to-date trust values for consensus weighting.
        """
        # Call parent sync to refresh metagraph
        super().sync()

        # Refresh validator trust after metagraph sync
        # Handle case where uid might not be set yet (during initialization)
        if not hasattr(self, 'uid') or self.uid is None:
            return

        try:
            old_trust = getattr(self, 'validator_trust', 0.0)
            self.validator_trust = self.metagraph.validator_trust[self.uid].item()

            # Log significant changes in trust
            if abs(self.validator_trust - old_trust) > 0.01:
                bt.logging.info(
                    f"📊 Validator trust updated: {old_trust:.4f} → {self.validator_trust:.4f} "
                    f"(Δ{self.validator_trust - old_trust:+.4f})"
                )
        except Exception as e:
            bt.logging.warning(f"Failed to refresh validator trust: {e}")

    def discover_miners(self):
        """Show all available miners on the network"""
        try:
            print(f"\n🔍 Discovering available miners on subnet {self.config.netuid}...")
            self.sync()  # Sync metagraph to get latest data

            available_miners = []
            running_miners = []
            for uid in range(self.metagraph.n):
                if uid != self.uid:  # Don't include self
                    hotkey = self.metagraph.hotkeys[uid]
                    stake = self.metagraph.S[uid].item()
                    axon_info = self.metagraph.axons[uid]

                    miner_info = {
                        'uid': uid,
                        'hotkey': hotkey,
                        'stake': stake,
                        'ip': axon_info.ip,
                        'port': axon_info.port
                    }
                    available_miners.append(miner_info)

                    # Check if this miner is currently running (has axon info)
                    if axon_info.ip != '0.0.0.0' and axon_info.port != 0:
                        running_miners.append(miner_info)

            # Miner discovery completed - details logged in debug mode if needed
            bt.logging.debug(f"Found {len(available_miners)} registered miners, {len(running_miners)} currently running")

            if not available_miners:
                print("   ⚠️  No miners found on the network")
            elif not running_miners:
                print("   ⚠️  No miners currently running")

        except Exception as e:
            print(f"❌ Error discovering miners: {e}")

    async def process_gateway_validation_workflow(self):
        """
        GATEWAY WORKFLOW (Passages 1 & 2): Fetch leads from gateway, validate, submit hashed results.
        This replaces process_sourced_leads_continuous for the new gateway-based architecture.
        
        ASYNC VERSION: Uses async subtensor for block queries (no memory leaks).
        """
        # Skip if processing broadcast request
        if self.processing_broadcast:
            return
        
        try:
            # Get current epoch_id from Bittensor block
            # Workers read from shared file (no Bittensor connection), coordinator uses Bittensor
            container_mode_check = getattr(self.config.neuron, 'mode', None)
            
            if container_mode_check == "worker":
                # WORKER: Read from shared block file (no Bittensor connection)
                try:
                    current_block, current_epoch, blocks_into_epoch = self._read_shared_block_file()
                except Exception as e:
                    print(f"⏳ Worker: Waiting for coordinator to write block file... ({e})")
                    await asyncio.sleep(5)
                    return
            else:
                # COORDINATOR or SINGLE: Use Bittensor connection
                current_block = await self.get_current_block_async()
                epoch_length = 360  # blocks per epoch
                current_epoch = current_block // epoch_length
                blocks_into_epoch = current_block % epoch_length
                
                # Write block info to shared file for workers (if coordinator/single mode)
                # This happens inline (no separate thread) to avoid websocket concurrency issues
                # Only write every 12 seconds to reduce disk I/O
                if container_mode_check != "worker":
                    if not hasattr(self, '_block_file_write_counter'):
                        self._block_file_write_counter = 0
                        # CRITICAL: Write immediately on first run to prevent worker deadlock
                        self._write_shared_block_file(current_block, current_epoch, blocks_into_epoch)
                    
                    self._block_file_write_counter += 1
                    if self._block_file_write_counter >= 12:
                        self._write_shared_block_file(current_block, current_epoch, blocks_into_epoch)
                        self._block_file_write_counter = 0
            
            # DEBUG: Always log epoch status
            print(f"[DEBUG] Current epoch: {current_epoch}, Block: {current_block}, Last processed: {getattr(self, '_last_processed_epoch', 'None')}")
            
            # Check if we've already processed this epoch
            if not hasattr(self, '_last_processed_epoch'):
                self._last_processed_epoch = current_epoch - 1
                print(f"[DEBUG] Initialized _last_processed_epoch to {self._last_processed_epoch}")
            
            if current_epoch <= self._last_processed_epoch:
                # Already processed this epoch - no need to spam logs
                print(f"[DEBUG] Skipping epoch {current_epoch} (already processed)")
                await asyncio.sleep(5)
                return
            
            print(f"[DEBUG] Processing epoch {current_epoch} for the FIRST TIME")
            
            # ═══════════════════════════════════════════════════════════════════
            # EPOCH TRANSITION: Clear old epochs from validator_weights file
            # This prevents file bloat and ensures clean state for new epoch
            # ═══════════════════════════════════════════════════════════════════
            self._clear_old_epochs_from_weights(current_epoch)
            
            print(f"\n{'='*80}")
            print(f"🔍 EPOCH {current_epoch}: Starting validation workflow")
            print(f"{'='*80}")
            
            # Fetch assigned leads from gateway
            from Leadpoet.utils.cloud_db import gateway_get_epoch_leads, gateway_submit_validation, gateway_submit_reveal
            
            # ═══════════════════════════════════════════════════════════════════
            # OPTIMIZED LEAD FETCHING: Only coordinator calls gateway
            # Workers read from shared file to avoid N duplicate API calls
            # ═══════════════════════════════════════════════════════════════════
            container_mode = getattr(self.config.neuron, 'mode', None)
            container_id = getattr(self.config.neuron, 'container_id', None)
            
            # Import os and hashlib early (needed for salt generation)
            import os
            import hashlib
            
            # CRITICAL: Check if leads file already exists with salt for this epoch
            # This prevents salt mismatch if coordinator restarts mid-epoch
            leads_file = Path("validator_weights") / f"epoch_{current_epoch}_leads.json"
            salt_hex = None
            
            if leads_file.exists():
                try:
                    with open(leads_file, 'r') as f:
                        existing_data = json.load(f)
                    if existing_data.get("epoch_id") == current_epoch and existing_data.get("salt"):
                        salt_hex = existing_data["salt"]
                        print(f"🔐 Reusing existing epoch salt: {salt_hex[:16]}... (from leads file)")
                except Exception as e:
                    print(f"⚠️  Could not read existing leads file: {e}")
            
            # Generate new salt only if we don't have one
            if not salt_hex:
                salt = os.urandom(32)
                salt_hex = salt.hex()
                print(f"🔐 Generated new epoch salt: {salt_hex[:16]}... (shared across all containers)")
            
            # Initialize truelist_results (will be populated by coordinator, read by workers)
            truelist_results = {}
            centralized_truelist_results = {}  # For workers reading from shared file
            
            if container_mode == "coordinator":
                # COORDINATOR: Fetch from gateway and share via file
                print(f"📡 Coordinator fetching leads from gateway for epoch {current_epoch}...")
                leads, max_leads_per_epoch = gateway_get_epoch_leads(self.wallet, current_epoch)
                
                # ================================================================
                # STEP 1: Write INITIAL file so workers can start Stage 0-2 immediately
                # truelist_results = None indicates "in progress" - workers will poll later
                # ================================================================
                leads_file = Path("validator_weights") / f"epoch_{current_epoch}_leads.json"
                with open(leads_file, 'w') as f:
                    json.dump({
                        "epoch_id": current_epoch,
                        "leads": leads, 
                        "max_leads_per_epoch": max_leads_per_epoch,
                        "created_at_block": current_block,
                        "salt": salt_hex,  # CRITICAL: Workers need this to hash results
                        "truelist_results": None  # None = "in progress", workers will poll after Stage 0-2
                    }, f)
                print(f"   💾 Initial file written: {len(leads) if leads else 0} leads + salt (TrueList in progress...)")
                
                # ================================================================
                # STEP 2: Start centralized TrueList as BACKGROUND TASK
                # Workers can now start Stage 0-2 while TrueList runs
                # ================================================================
                truelist_task = None
                truelist_results = {}
                all_leads_for_file = leads  # Save original list before any slicing
                if leads:
                    from validator_models.automated_checks import run_centralized_truelist_batch
                    
                    print(f"\n📧 COORDINATOR: Starting centralized TrueList batch for ALL {len(leads)} leads (BACKGROUND)...")
                    truelist_task = asyncio.create_task(run_centralized_truelist_batch(leads))
                
            elif container_mode == "worker":
                # WORKER: Wait for coordinator to fetch and share
                print(f"⏳ Worker waiting for coordinator to fetch leads for epoch {current_epoch}...")
                leads_file = Path("validator_weights") / f"epoch_{current_epoch}_leads.json"
                
                # Keep checking but with epoch boundary protection
                waited = 0
                log_interval = 300  # Log every 5 minutes
                check_interval = 5  # Check every 5 seconds
                
                while not leads_file.exists():
                    await asyncio.sleep(check_interval)
                    waited += check_interval
                    
                    # CRITICAL: Check current block and epoch from shared file
                    try:
                        check_block, check_epoch, blocks_into_epoch = self._read_shared_block_file()
                    except Exception as e:
                        # Coordinator hasn't updated file yet, keep waiting
                        continue
                    
                    # Epoch changed while waiting - abort this epoch
                    if check_epoch > current_epoch:
                        print(f"❌ Worker: Epoch changed ({current_epoch} → {check_epoch}) while waiting")
                        print(f"   Aborting - will process epoch {check_epoch} in next iteration")
                        await asyncio.sleep(10)
                        return
                    
                    # Too late to start validation (block 275+ cutoff)
                    if blocks_into_epoch >= 275:
                        print(f"❌ Worker: Too late to start validation (block {blocks_into_epoch}/360)")
                        print(f"   Cutoff is block 275 - not enough time to complete before epoch end")
                        print(f"   Skipping epoch {current_epoch}, will process next epoch")
                        await asyncio.sleep(10)
                        return
                    
                    # Log progress every 5 minutes
                    if waited % log_interval == 0:
                        print(f"   ⏳ Still waiting for coordinator... ({waited}s elapsed, block {blocks_into_epoch}/360)")
                        print(f"      Checking for: {leads_file}")
                
                # Read leads from shared file
                with open(leads_file, 'r') as f:
                    data = json.load(f)
                    file_epoch = data.get("epoch_id")
                    leads = data.get("leads")
                    max_leads_per_epoch = data.get("max_leads_per_epoch")
                    centralized_truelist_results = data.get("truelist_results", {})  # Precomputed by coordinator
                
                # Verify epoch matches (safety check)
                if file_epoch != current_epoch:
                    print(f"❌ Worker: Epoch mismatch in leads file!")
                    print(f"   Expected epoch: {current_epoch}")
                    print(f"   File has epoch: {file_epoch}")
                    print(f"   Skipping - stale file detected")
                    await asyncio.sleep(10)
                    return
                
                print(f"✅ Worker loaded {len(leads) if leads else 0} leads from coordinator (waited {waited}s)")
                # Note: truelist_results might be None (in progress) or {} (complete/failed)
                # Workers will run Stage 0-2 first, then poll for truelist_results
                if centralized_truelist_results:
                    print(f"   ✅ TrueList already complete: {len(centralized_truelist_results)} results from coordinator")
                elif centralized_truelist_results is None:
                    print(f"   ⏳ TrueList still in progress - will poll after Stage 0-2 completes")
                else:
                    print(f"   ⚠️ TrueList returned empty results - leads will fail email verification")
                
            else:
                # DEFAULT: Single validator mode (no containers)
                print(f"📡 Fetching leads from gateway for epoch {current_epoch}...")
                leads, max_leads_per_epoch = gateway_get_epoch_leads(self.wallet, current_epoch)
            
            # Store max_leads_per_epoch for use in submit_weights_at_epoch_end
            # This value comes dynamically from the gateway config
            self._max_leads_per_epoch = max_leads_per_epoch
            
            # Handle different response types:
            # - None = Already submitted (gateway returned explicit message)
            # - [] = Timeout/error (should retry)
            # - [lead1, lead2, ...] = Got leads
            
            if leads is None:
                # Gateway explicitly said "already submitted" or "queue empty"
                print(f"ℹ️  No leads to process for epoch {current_epoch}")
                print(f"   Gateway confirmed: You've already submitted or queue is empty")
                
                # Mark as processed (don't retry - would be duplicate submission)
                self._last_processed_epoch = current_epoch
                print(f"✅ Marked epoch {current_epoch} as processed (already submitted)\n")
                await asyncio.sleep(10)
                return
            
            print(f"[DEBUG] Received {len(leads)} leads from gateway (max_leads_per_epoch={max_leads_per_epoch})")
            
            if not leads:
                # Empty list = timeout or error (NOT already submitted)
                print(f"⚠️  Gateway returned 0 leads (timeout or error)")
                print(f"   This is likely a temporary issue - validator will retry automatically")
                print(f"   NOT marking epoch as processed - will retry next iteration\n")
                await asyncio.sleep(30)  # Wait longer before retry
                return
            
            print(f"✅ Received {len(leads)} leads from gateway")
            
            # ═══════════════════════════════════════════════════════════════════
            # DYNAMIC LEAD DISTRIBUTION: Auto-calculate ranges for containers
            # ═══════════════════════════════════════════════════════════════════
            container_id = getattr(self.config.neuron, 'container_id', None)
            total_containers = getattr(self.config.neuron, 'total_containers', None)
            
            if container_id is not None and total_containers is not None:
                # DYNAMIC CALCULATION: Auto-distribute leads across containers
                original_count = len(leads)
                
                # Calculate this container's slice
                leads_per_container = original_count // total_containers
                remainder = original_count % total_containers
                
                # First 'remainder' containers get 1 extra lead to distribute remainder evenly
                if container_id < remainder:
                    start = container_id * (leads_per_container + 1)
                    end = start + leads_per_container + 1
                else:
                    start = (remainder * (leads_per_container + 1)) + ((container_id - remainder) * leads_per_container)
                    end = start + leads_per_container
                
                leads = leads[start:end]
                lead_range_str = f"{start}-{end}"
                
                print(f"📦 Container {container_id}/{total_containers}: Processing leads {start}-{end}")
                print(f"   ({len(leads)}/{original_count} leads assigned to this container)")
                print(f"   Gateway MAX_LEADS_PER_EPOCH: {max_leads_per_epoch}")
                print(f"   (Dynamic distribution - adapts to any gateway setting)")
                print("")
            else:
                # No containerization - process all leads
                lead_range_str = None
            
            # ================================================================
            # BATCH VALIDATION: Stage 0-2 runs in PARALLEL with TrueList
            # After Stage 0-2, poll file for truelist_results before Stage 4-5
            # ================================================================
            print(f"🔍 Running BATCH automated checks on {len(leads)} leads...")
            print("")
            
            from validator_models.automated_checks import run_batch_automated_checks, get_email
            
            # (os and hashlib already imported at line 1845)
            validation_results = []
            local_validation_data = []  # Store for weight calculation
            
            # Salt already generated earlier (line 1850) and shared with workers via leads file
            # Convert back from hex for coordinator's own validation
            salt = bytes.fromhex(salt_hex)
            
            # Extract lead_blobs for batch processing
            lead_blobs = [lead.get('lead_blob', {}) for lead in leads]
            
            # ================================================================
            # COORDINATOR: Background task to wait for TrueList and update file
            # This allows Stage 0-2 to run in parallel with TrueList
            # ================================================================
            async def truelist_file_updater():
                """Wait for centralized TrueList to complete, then update file."""
                nonlocal truelist_results
                if truelist_task is None:
                    return  # No TrueList task (no leads)
                try:
                    print(f"   🔄 Background: Waiting for centralized TrueList to complete...")
                    truelist_results = await truelist_task
                    print(f"   ✅ Background: Centralized TrueList complete ({len(truelist_results)} results)")
                    
                    # Update the file with truelist_results
                    leads_file = Path("validator_weights") / f"epoch_{current_epoch}_leads.json"
                    with open(leads_file, 'w') as f:
                        json.dump({
                            "epoch_id": current_epoch,
                            "leads": all_leads_for_file,  # All leads (not just coordinator's slice)
                            "max_leads_per_epoch": max_leads_per_epoch,
                            "created_at_block": current_block,
                            "salt": salt_hex,
                            "truelist_results": truelist_results  # NOW POPULATED
                        }, f)
                    print(f"   💾 Background: Updated file with {len(truelist_results)} TrueList results")
                except Exception as e:
                    print(f"   ❌ Background: TrueList failed: {e}")
                    truelist_results = {}  # Empty = leads fail email verification
                    # Still update file to unblock workers (with empty results)
                    leads_file = Path("validator_weights") / f"epoch_{current_epoch}_leads.json"
                    with open(leads_file, 'w') as f:
                        json.dump({
                            "epoch_id": current_epoch,
                            "leads": all_leads_for_file,
                            "max_leads_per_epoch": max_leads_per_epoch,
                            "created_at_block": current_block,
                            "salt": salt_hex,
                            "truelist_results": {}  # Empty due to failure
                        }, f)
                    print(f"   💾 Background: Updated file with EMPTY TrueList results (failure)")
            
            # Start TrueList file updater in background (coordinator only)
            truelist_updater_task = None
            if container_mode == "coordinator" and truelist_task is not None:
                truelist_updater_task = asyncio.create_task(truelist_file_updater())
            
            # CRITICAL: Batch validation takes 10+ minutes. During this time, we MUST keep
            # updating the block file so workers don't see stale data and get stuck.
            # Solution: Run a background task that updates block file every 10 seconds.
            
            async def block_file_updater():
                """Background task to keep block file fresh AND check for weight submission during batch validation."""
                while True:
                    try:
                        await asyncio.sleep(10)  # Update every 10 seconds
                        current_block_bg = await self.get_current_block_async()
                        current_epoch_bg = current_block_bg // 360
                        blocks_into_epoch_bg = current_block_bg % 360
                        self._write_shared_block_file(current_block_bg, current_epoch_bg, blocks_into_epoch_bg)
                        
                        # CRITICAL: Check for weight submission at block 345+
                        # This ensures weights are submitted even if Stage 4-5 is still running
                        if blocks_into_epoch_bg >= 345:
                            try:
                                await self.submit_weights_at_epoch_end()
                            except Exception as weight_err:
                                print(f"   ⚠️ Weight submission check error: {weight_err}")
                    except asyncio.CancelledError:
                        break  # Stop when batch validation completes
                    except Exception as e:
                        print(f"   ⚠️ Block file update error: {e}")
            
            # Start block file updater in background
            block_updater_task = asyncio.create_task(block_file_updater())
            
            # Path to leads file for polling TrueList results
            leads_file_str = str(Path("validator_weights") / f"epoch_{current_epoch}_leads.json")
            
            try:
                batch_results = await run_batch_automated_checks(
                    lead_blobs, 
                    container_id=0 if container_mode == "coordinator" else int(os.environ.get('CONTAINER_ID', 0)),
                    leads_file_path=leads_file_str  # Poll file for TrueList results after Stage 0-2
                )
            except Exception as e:
                print(f"   ❌ Batch validation failed: {e}")
                import traceback
                traceback.print_exc()
                # Fallback: Mark all leads as validation errors
                batch_results = [
                    (False, {
                        "passed": False,
                        "rejection_reason": {
                            "stage": "Batch Validation",
                            "check_name": "run_batch_automated_checks",
                            "message": f"Batch validation error: {str(e)}"
                        }
                    })
                    for _ in leads
                ]
            finally:
                # Stop the block file updater
                block_updater_task.cancel()
                try:
                    await block_updater_task
                except asyncio.CancelledError:
                    pass
            
            print(f"\n📦 Batch validation complete. Processing {len(batch_results)} results...")
            
            # Process batch results - this loop PRESERVES block file updates and epoch detection
            for idx, (lead, (passed, automated_checks_data)) in enumerate(zip(leads, batch_results), 1):
                try:
                    lead_blob = lead.get("lead_blob", {})
                    email = lead_blob.get("email", "unknown@example.com")
                    company = lead_blob.get("Company") or lead_blob.get("business", "Unknown")
                    
                    print(f"{'─'*80}")
                    print(f"📋 Processing result {idx}/{len(leads)}: {email} @ {company}")
                    
                    # Handle skipped leads (passed=None means TrueList errors after retries)
                    if passed is None:
                        is_valid = False
                        decision = "deny"
                        rep_score = 0
                        rejection_reason = {
                            "stage": "Batch Validation",
                            "check_name": "truelist_batch_skipped",
                            "message": "Lead skipped due to persistent TrueList errors"
                        }
                        result = {"is_legitimate": False, "reason": rejection_reason, "skipped": True}
                    else:
                        is_valid = passed
                        decision = "approve" if is_valid else "deny"
                        # CRITICAL: Use validator-calculated rep_score, NOT miner's submitted value
                        # Denied leads get 0, approved leads get score from automated checks
                        # rep_score is a dict with 'total_score' key, not a simple integer
                        rep_score_data = automated_checks_data.get('rep_score', {})
                        if isinstance(rep_score_data, dict):
                            rep_score = int(rep_score_data.get('total_score', 0)) if is_valid else 0
                        else:
                            # Fallback for legacy format where rep_score was an integer
                            rep_score = int(rep_score_data) if is_valid else 0
                        rejection_reason = automated_checks_data.get("rejection_reason") or {} if not is_valid else {"message": "pass"}
                        
                        # Build result structure matching old validate_lead() output
                        result = {
                            "is_legitimate": is_valid,
                            "enhanced_lead": automated_checks_data if is_valid else {},
                            "reason": rejection_reason if not is_valid else None
                        }
                        if is_valid:
                            result["enhanced_lead"]["rep_score"] = rep_score
                    
                    # Strip internal cache fields from evidence (they contain datetime objects and aren't needed)
                    # These are Stage 4 optimization artifacts, not part of the validation evidence
                    clean_result = result.copy()
                    if "enhanced_lead" in clean_result and isinstance(clean_result["enhanced_lead"], dict):
                        clean_enhanced = clean_result["enhanced_lead"].copy()
                        # Remove internal cache fields that shouldn't be in evidence
                        for internal_field in ["company_linkedin_data", "company_linkedin_slug", "company_linkedin_from_cache"]:
                            clean_enhanced.pop(internal_field, None)
                        clean_result["enhanced_lead"] = clean_enhanced
                    
                    evidence_blob = json.dumps(clean_result, default=str)  # Handle any remaining datetime objects
                    
                    # Compute hashes (SHA256 with salt)
                    decision_hash = hashlib.sha256((decision + salt.hex()).encode()).hexdigest()
                    rep_score_hash = hashlib.sha256((str(rep_score) + salt.hex()).encode()).hexdigest()
                    rejection_reason_hash = hashlib.sha256((json.dumps(rejection_reason, default=str) + salt.hex()).encode()).hexdigest()  # Handle datetime
                    evidence_hash = hashlib.sha256(evidence_blob.encode()).hexdigest()
                    
                    # Store hashed result for gateway submission
                    # lead_id and miner_hotkey are at top level (not in lead_blob)
                    validation_results.append({
                        "lead_id": lead.get("lead_id"),  # Top level
                        "decision_hash": decision_hash,
                        "rep_score_hash": rep_score_hash,
                        "rejection_reason_hash": rejection_reason_hash,
                        "evidence_hash": evidence_hash,
                        "evidence_blob": result  # Include full evidence for gateway storage
                    })
                    
                    # Store local data for weight calculation
                    local_validation_data.append({
                        "lead_id": lead.get("lead_id"),  # Top level
                        "miner_hotkey": lead.get("miner_hotkey"),  # Top level
                        "decision": decision,
                        "rep_score": rep_score,
                        "rejection_reason": rejection_reason,
                        "salt": salt.hex()
                    })
                    
                    # Store weight data for later accumulation
                    # Workers: Save in JSON for coordinator to aggregate
                    # Coordinator/Default: Accumulate immediately (single validator)
                    # Coordinator in containerized mode: Will re-accumulate all after aggregation
                    container_mode = getattr(self.config.neuron, 'mode', None)
                    
                    # Store weight info in local_validation_data for aggregation
                    if len(local_validation_data) > 0:
                        local_validation_data[-1]["is_icp_multiplier"] = lead.get("is_icp_multiplier", 1.0)
                    
                    # Only accumulate now if NOT in container mode (backward compatibility)
                    # In container mode, coordinator will accumulate ALL leads after aggregation
                    if container_mode is None:
                        # Traditional single-validator mode
                        is_icp_multiplier = lead.get("is_icp_multiplier", 1.0)
                        await self.accumulate_miner_weights(
                            miner_hotkey=lead.get("miner_hotkey"),
                            rep_score=rep_score,
                            is_icp_multiplier=is_icp_multiplier,
                            decision=decision
                        )
                    
                    # Pretty output
                    status_icon = "✅" if is_valid else "❌"
                    decision_text = "APPROVED" if is_valid else "DENIED"
                    print(f"   {status_icon} Decision: {decision_text}")
                    print(f"   📊 Rep Score: {rep_score}/{MAX_REP_SCORE}")
                    if not is_valid:
                        # Print full rejection details
                        print(f"   ❌ REJECTION DETAILS:")
                        print(f"      Stage: {rejection_reason.get('stage', 'Unknown')}")
                        print(f"      Check: {rejection_reason.get('check_name', 'Unknown')}")
                        print(f"      Message: {rejection_reason.get('message', 'Unknown reason')}")
                        failed_fields = rejection_reason.get('failed_fields', [])
                        if failed_fields:
                            print(f"      Failed Fields: {', '.join(failed_fields)}")
                    print("")
                    
                    # Check block/epoch status every 20 leads (no delay - this is just hash preparation)
                    if idx < len(leads) and idx % 20 == 0:
                        # Check if we should submit weights mid-processing (block 345+)
                        await self.submit_weights_at_epoch_end()
                        
                        # Check if epoch changed - if so, stop processing old epoch's leads
                        new_block = await self.get_current_block_async()
                        new_epoch = new_block // 360
                        blocks_into_epoch = new_block % 360
                        
                        # Update block file for workers
                        container_mode_check = getattr(self.config.neuron, 'mode', None)
                        if container_mode_check != "worker":
                            self._write_shared_block_file(new_block, new_epoch, blocks_into_epoch)
                        
                        if new_epoch > current_epoch:
                            print(f"\n{'='*80}")
                            print(f"⚠️  EPOCH CHANGED: {current_epoch} → {new_epoch}")
                            print(f"   Stopping validation of epoch {current_epoch} leads ({idx}/{len(leads)} complete)")
                            print(f"   Remaining {len(leads) - idx} leads cannot be submitted (epoch closed)")
                            print(f"{'='*80}\n")
                            break  # Exit the lead processing loop
                        
                        # FORCE STOP at block 345 for WORKERS (weight submission time)
                        # Coordinator needs to submit weights, workers must finish before that
                        container_mode = getattr(self.config.neuron, 'mode', None)
                        if container_mode == "worker" and blocks_into_epoch >= 345:
                            print(f"\n{'='*80}")
                            print(f"⏰ WORKER FORCE STOP: Block 345+ reached (block {blocks_into_epoch}/360)")
                            print(f"   Workers must complete before coordinator submits weights")
                            print(f"   Completed: {idx}/{len(leads)} leads")
                            print(f"   📦 Saving partial results for coordinator to aggregate")
                            print(f"{'='*80}\n")
                            break  # Exit the lead processing loop and proceed to worker JSON write
                    
                except Exception as e:
                    # Error processing batch result (rare - validation already complete)
                    lead_id = lead.get('lead_id', 'unknown')
                    email = lead.get('lead_blob', {}).get('email', 'unknown')
                    
                    print(f"❌ Error processing result for lead {lead_id[:8]}: {e}")
                    import traceback
                    traceback.print_exc()
                    print("")
                    # Continue to next lead after error (no delay needed for hash preparation)
                    continue
            
            # ═══════════════════════════════════════════════════════════════════
            # CONTAINER MODE HANDLING: Worker vs Coordinator
            # ═══════════════════════════════════════════════════════════════════
            container_mode = getattr(self.config.neuron, 'mode', None)
            
            if container_mode == "worker" and lead_range_str:
                # WORKER MODE: Write results to JSON and exit (don't submit to gateway)
                print(f"{'='*80}")
                print(f"👷 WORKER MODE: Writing validation results to shared file")
                print(f"{'='*80}")
                
                worker_results = {
                    "validation_results": validation_results,  # For gateway submission
                    "local_validation_data": local_validation_data,  # For reveals
                    "epoch_id": current_epoch,
                    "lead_range": lead_range_str,
                    "container_id": container_id,
                    "timestamp": time.time()
                }
                
                # Write to shared volume (validator_weights/worker_results_<container_id>.json)
                worker_file = os.path.join("validator_weights", f"worker_results_container_{container_id}.json")
                with open(worker_file, 'w') as f:
                    json.dump(worker_results, f, indent=2)
                
                print(f"✅ Worker wrote {len(validation_results)} validation results to {worker_file}")
                print(f"   Epoch: {current_epoch}")
                print(f"   Container ID: {container_id}")
                print(f"   Lead range: {lead_range_str}")
                print(f"   Worker exiting (coordinator will submit to gateway)")
                print(f"{'='*80}\n")
                
                # Mark epoch as processed so we don't repeat this work
                self._last_processed_epoch = current_epoch
                
                # Exit worker process
                import sys
                sys.exit(0)
            
            elif container_mode == "coordinator" and container_id is not None and total_containers is not None:
                # COORDINATOR MODE: Wait for workers, aggregate results, then submit
                print(f"{'='*80}")
                print(f"📡 COORDINATOR MODE: Waiting for worker results")
                print(f"{'='*80}")
                
                # Determine worker IDs (all containers except coordinator)
                worker_ids = [i for i in range(total_containers) if i != container_id]
                num_workers = len(worker_ids)
                
                print(f"   Coordinator (Container {container_id}): Processed {lead_range_str} ({len(validation_results)} results)")
                print(f"   Waiting for {num_workers} workers: Container IDs {worker_ids}")
                
                # Wait for worker result files (with timeout)
                import time as time_module
                max_wait = 3600  # 60 minutes max wait
                check_interval = 5  # Check every 5 seconds
                waited = 0
                
                worker_files = []
                for worker_id in worker_ids:
                    # Lightweight workers write: worker_{worker_id}_epoch_{epoch}_results.json
                    worker_file = os.path.join("validator_weights", f"worker_{worker_id}_epoch_{current_epoch}_results.json")
                    worker_files.append((worker_id, worker_file))
                
                all_workers_ready = False
                while waited < max_wait and not all_workers_ready:
                    all_workers_ready = all(os.path.exists(wf[1]) for wf in worker_files)
                    if not all_workers_ready:
                        # Check if we're approaching block 335 (hash submission deadline)
                        current_block_check = await self.get_current_block_async()
                        current_epoch_check = current_block_check // 360
                        blocks_into_epoch_check = current_block_check % 360
                        
                        # CRITICAL: Update block file so workers get fresh epoch/block info
                        # Without this, workers see stale data and get stuck in "too late" loop
                        self._write_shared_block_file(current_block_check, current_epoch_check, blocks_into_epoch_check)
                        
                        # FORCE PROCEED at block 320 (must submit before reveal deadline at block 328)
                        if blocks_into_epoch_check >= 320:
                            print(f"   ⏰ BLOCK 320+ REACHED: Force proceeding with available results")
                            print(f"      Block: {blocks_into_epoch_check}/360")
                            print(f"      Must submit hashes before reveal deadline (block 328)")
                            missing = [f"Container-{wf[0]}" for wf in worker_files if not os.path.exists(wf[1])]
                            print(f"      Missing workers: {missing}")
                            print(f"      Proceeding with partial results")
                            break
                        
                        missing = [f"Container-{wf[0]}" for wf in worker_files if not os.path.exists(wf[1])]
                        print(f"   ⏳ Waiting for workers: {missing} ({waited}s / {max_wait}s, block {blocks_into_epoch_check}/360)")
                        await asyncio.sleep(check_interval)
                        waited += check_interval
                    else:
                        print(f"   ✅ All {len(worker_files)} workers finished in {waited}s")
                        break
                
                if not all_workers_ready:
                    print(f"   ⚠️  TIMEOUT: Not all workers finished after {max_wait}s")
                    print(f"   Proceeding with coordinator results only")
                
                # Aggregate results from all workers
                aggregated_validation_results = list(validation_results)  # Copy coordinator's results
                aggregated_local_validation_data = list(local_validation_data)  # Copy coordinator's reveals
                
                for worker_id, worker_file in worker_files:
                    if os.path.exists(worker_file):
                        try:
                            with open(worker_file, 'r') as f:
                                worker_data = json.load(f)
                            
                            worker_validations = worker_data.get("validation_results", [])
                            worker_reveals = worker_data.get("local_validation_data", [])
                            worker_range = worker_data.get("lead_range", "unknown")
                            
                            aggregated_validation_results.extend(worker_validations)
                            aggregated_local_validation_data.extend(worker_reveals)
                            
                            print(f"   ✅ Aggregated {len(worker_validations)} results from Container-{worker_id} (range: {worker_range})")
                            
                            # Delete worker file after successful aggregation
                            os.remove(worker_file)
                        except Exception as e:
                            print(f"   ⚠️  Failed to load worker Container-{worker_id}: {e}")
                
                # Replace local lists with aggregated results
                validation_results = aggregated_validation_results
                local_validation_data = aggregated_local_validation_data
                
                print(f"   📊 Total aggregated: {len(validation_results)} validations")
                
                # Clean up shared leads file (no longer needed)
                leads_file = Path("validator_weights") / f"epoch_{current_epoch}_leads.json"
                if leads_file.exists():
                    os.remove(leads_file)
                    print(f"   🧹 Cleaned up {leads_file.name}")
                
                # Clean up any stale leads files from previous epochs
                try:
                    weights_dir = Path("validator_weights")
                    for old_file in weights_dir.glob("epoch_*_leads.json"):
                        # Extract epoch from filename
                        try:
                            file_epoch = int(old_file.stem.split('_')[1])
                            if file_epoch < current_epoch:
                                os.remove(old_file)
                                print(f"   🧹 Cleaned up stale file: {old_file.name}")
                        except (IndexError, ValueError):
                            pass
                except Exception as e:
                    print(f"   ⚠️  Could not clean up stale files: {e}")
                
                # ═══════════════════════════════════════════════════════════════════
                # COORDINATOR: Accumulate weights for ALL leads (coordinator + workers)
                # This ensures all leads are counted in validator_weights_history
                # ═══════════════════════════════════════════════════════════════════
                print(f"   ⚖️  Accumulating weights for all {len(local_validation_data)} leads...")
                for val_data in local_validation_data:
                    miner_hotkey = val_data.get("miner_hotkey")
                    decision = val_data.get("decision")
                    rep_score = val_data.get("rep_score", 0)
                    is_icp_multiplier = val_data.get("is_icp_multiplier", 1.0)
                    
                    await self.accumulate_miner_weights(
                        miner_hotkey=miner_hotkey,
                        rep_score=rep_score,
                        is_icp_multiplier=is_icp_multiplier,
                        decision=decision
                    )
                print(f"   ✅ Weight accumulation complete")
                
                print(f"   Proceeding with gateway submission...")
                print(f"{'='*80}\n")
            
            # Submit hashed validation results to gateway
            print(f"{'='*80}")
            
            # Check if epoch changed before attempting submission
            submit_block = await self.get_current_block_async()
            submit_epoch = submit_block // 360
            
            if submit_epoch > current_epoch:
                print(f"⚠️  Epoch changed ({current_epoch} → {submit_epoch}) - skipping hash submission")
                print(f"   {len(validation_results)} validations for epoch {current_epoch} cannot be submitted")
                print(f"   (Weights already submitted, epoch will be marked as processed)")
                success = False  # Mark as failed to skip storing reveals
            elif validation_results:
                print(f"📤 Submitting {len(validation_results)} hashed validations to gateway...")
                success = gateway_submit_validation(self.wallet, current_epoch, validation_results)
                if success:
                    print(f"✅ Successfully submitted {len(validation_results)} validations for epoch {current_epoch}")
                    print(f"   (Hashed: decision, rep_score, rejection_reason, evidence)")
                    print(f"   Gateway logged to TEE buffer → will be in next Arweave checkpoint")
                    
                    # Store local data for reveal later
                    if not hasattr(self, '_pending_reveals'):
                        self._pending_reveals = {}
                    self._pending_reveals[current_epoch] = local_validation_data
                    
                    # 🚨 CRITICAL: Save reveals to disk immediately (crash protection)
                    self._save_pending_reveals()
                    print(f"   💾 Saved {len(local_validation_data)} pending reveals to disk for epoch {current_epoch}")
                else:
                    print(f"❌ Failed to submit validations for epoch {current_epoch}")
                    print(f"   Epoch may have changed - skipping to avoid re-processing")
                    # Still mark as processed to avoid re-validating 80 leads
                    # Weights will still be submitted at epoch end
            else:
                print(f"⚠️  No validation results to submit (all leads failed validation)")
            
            # Weights already accumulated (coordinator mode) or accumulation skipped (container mode)
            # Weight submission to blockchain happens at block 345+ via submit_weights_at_epoch_end()
            if container_mode is None:
                print(f"\n{'='*80}")
                print(f"⚖️  Weights accumulated for this epoch")
                print(f"   (Will submit at block 345+ via submit_weights_at_epoch_end())")
                print(f"{'='*80}")
            
            # Mark epoch as processed
            self._last_processed_epoch = current_epoch
            print(f"\n{'='*80}")
            print(f"✅ EPOCH {current_epoch}: Validation workflow complete")
            print(f"{'='*80}\n")
            
            # Check for reveals from previous epochs
            await self.process_pending_reveals()
            
            # Check if we should submit weights (block 345+)
            await self.submit_weights_at_epoch_end()
            
        except Exception as e:
            print(f"[DEBUG] Exception caught in gateway validation workflow: {e}")
            import traceback
            print(f"[DEBUG] Full traceback:\n{traceback.format_exc()}")
            bt.logging.error(f"Error in gateway validation workflow: {e}")
            import traceback
            bt.logging.error(traceback.format_exc())
    
    async def accumulate_miner_weights(self, miner_hotkey: str, rep_score: int, is_icp_multiplier: float, decision: str):
        """
        Accumulate weights for approved leads in real-time as validation happens.
        
        ASYNC VERSION: Uses async subtensor for block queries.
        
        This updates BOTH files after each lead validation:
        - validator_weights/validator_weights (current epoch only)
        - validator_weights/validator_weights_history (all epochs, never cleared)
        
        This provides crash resilience - if validator disconnects before epoch end,
        the latest weights are already saved in history.
        
        Tracks both:
        - miner_scores: Sum of effective_rep_score per miner (for weight distribution)
        - approved_lead_count: Number of approved leads (for linear emissions scaling)
        
        ICP ADJUSTMENT SYSTEM (NEW):
        - is_icp_multiplier now stores ADJUSTMENT value (-15 to +20)
        - effective_rep_score = base_rep_score + icp_adjustment (floor at 0)
        
        BACKWARDS COMPATIBILITY:
        - OLD format: is_icp_multiplier in {1.0, 1.5, 5.0} → use multiplication
        - NEW format: all other values → use addition
        
        Args:
            miner_hotkey: Miner's hotkey who submitted the lead
            rep_score: Base reputation score (0-48) from automated checks (NOT inflated)
            is_icp_multiplier: OLD: multiplier (1.0, 1.5, 5.0) / NEW: adjustment (-15 to +20)
            decision: "approve" or "deny"
        """
        try:
            weights_dir = Path("validator_weights")
            weights_dir.mkdir(exist_ok=True)
            weights_file = weights_dir / "validator_weights"
            history_file = weights_dir / "validator_weights_history"
            
            # Get current epoch using async subtensor
            current_block = await self.get_current_block_async()
            current_epoch = current_block // 360
            
            # ═══════════════════════════════════════════════════════════
            # 1. UPDATE validator_weights (current epoch only)
            # ═══════════════════════════════════════════════════════════
            if weights_file.exists():
                with open(weights_file, 'r') as f:
                    weights_data = json.load(f)
            else:
                weights_data = {"curators": [], "sourcers_of_curated": []}
            
            # Initialize epoch if not exists (ensures burn weights can be submitted even if all leads denied)
            if str(current_epoch) not in weights_data:
                weights_data[str(current_epoch)] = {
                    "epoch": current_epoch,
                    "start_block": current_epoch * 360,
                    "end_block": (current_epoch + 1) * 360,
                    "miner_scores": {},
                    "approved_lead_count": 0,  # Track number of approved leads for linear emissions
                    "max_leads_per_epoch": getattr(self, '_max_leads_per_epoch', 50),  # Persist for restart recovery
                    "last_updated": datetime.utcnow().isoformat()
                }
                # Save immediately so epoch exists even if all leads are denied
                with open(weights_file, 'w') as f:
                    json.dump(weights_data, f, indent=2)
            
            # Early return for denied leads (epoch entry already created and saved above)
            if decision != "approve":
                return
            
            # ═══════════════════════════════════════════════════════════
            # ICP VALUE INTERPRETATION (BACKWARDS COMPATIBLE)
            # ═══════════════════════════════════════════════════════════
            # OLD FORMAT: is_icp_multiplier in {1.0, 1.5, 5.0} → multiply
            # NEW FORMAT: any other value (integers -15 to +20) → add
            OLD_MULTIPLIER_VALUES = {1.0, 1.5, 5.0}
            
            if is_icp_multiplier in OLD_MULTIPLIER_VALUES:
                # OLD FORMAT: Use multiplication (legacy leads)
                effective_rep_score = rep_score * is_icp_multiplier
                print(f"      📊 Legacy ICP multiplier: {rep_score} × {is_icp_multiplier} = {effective_rep_score}")
            else:
                # NEW FORMAT: Use addition with floor at 0
                icp_adjustment = int(is_icp_multiplier)
                effective_rep_score = max(0, rep_score + icp_adjustment)
                print(f"      📊 ICP adjustment: {rep_score} + ({icp_adjustment:+d}) = {effective_rep_score}")
            
            # Add effective score to miner's total (only for approved leads)
            epoch_data = weights_data[str(current_epoch)]
            if miner_hotkey not in epoch_data["miner_scores"]:
                epoch_data["miner_scores"][miner_hotkey] = 0
            
            epoch_data["miner_scores"][miner_hotkey] += effective_rep_score
            
            # Increment approved lead count for linear emissions
            if "approved_lead_count" not in epoch_data:
                epoch_data["approved_lead_count"] = 0
            epoch_data["approved_lead_count"] += 1
            
            epoch_data["last_updated"] = datetime.utcnow().isoformat()
            
            # Save updated weights
            with open(weights_file, 'w') as f:
                json.dump(weights_data, f, indent=2)
            
            # ═══════════════════════════════════════════════════════════
            # 2. UPDATE validator_weights_history (all epochs, real-time)
            # ═══════════════════════════════════════════════════════════
            if history_file.exists():
                with open(history_file, 'r') as f:
                    history_data = json.load(f)
            else:
                history_data = {"curators": [], "sourcers_of_curated": []}
            
            # Update history with same epoch data (or create new entry)
            history_data[str(current_epoch)] = {
                "epoch": current_epoch,
                "start_block": current_epoch * 360,
                "end_block": (current_epoch + 1) * 360,
                "miner_scores": epoch_data["miner_scores"].copy(),  # Deep copy of scores
                "approved_lead_count": epoch_data.get("approved_lead_count", 0),  # Track for linear emissions
                "max_leads_per_epoch": getattr(self, '_max_leads_per_epoch', epoch_data.get("max_leads_per_epoch", 50)),  # Persist for restart recovery
                "last_updated": datetime.utcnow().isoformat()
            }
            
            # Save updated history (accumulates all epochs)
            with open(history_file, 'w') as f:
                json.dump(history_data, f, indent=2)
            
            # Prune old epochs to prevent file bloat (keep max 50 epochs)
            self.prune_history_file(current_epoch, max_epochs=50)
            
            approved_count = epoch_data.get("approved_lead_count", 0)
            print(f"      💾 Accumulated {rep_score} points for miner {miner_hotkey[:10]}... (total: {epoch_data['miner_scores'][miner_hotkey]})")
            print(f"      📊 Epoch approved leads: {approved_count}")
            print(f"      📚 Updated history file (crash-resilient)")
            
        except Exception as e:
            bt.logging.error(f"Failed to accumulate miner weights: {e}")
    
    async def submit_weights_at_epoch_end(self):
        """
        Submit accumulated weights to Bittensor chain at end of epoch (block 345+).
        
        ASYNC VERSION: Uses async subtensor for block queries.
        
        This reads from validator_weights/validator_weights and submits to chain.
        After submission, archives weights to history and clears active file.
        """
        try:
            if self.config.neuron.disable_set_weights:
                bt.logging.info("⏸️  Weight submission disabled (--neuron.disable_set_weights flag is set)")
                return False
            
            current_block = await self.get_current_block_async()
            epoch_length = 360
            current_epoch = current_block // 360
            blocks_into_epoch = current_block % epoch_length
            
            # Only submit after block 345 (near end of epoch)
            if blocks_into_epoch < 345:
                return False
            
            # ═══════════════════════════════════════════════════════════════════
            # CRITICAL: Check if we've already submitted weights for this epoch
            # Prevents duplicate submissions (which would show 0 leads after clear)
            # ═══════════════════════════════════════════════════════════════════
            if not hasattr(self, '_last_weight_submission_epoch'):
                self._last_weight_submission_epoch = None
            
            if self._last_weight_submission_epoch == current_epoch:
                # Already submitted for this epoch - don't resubmit!
                # This is the PRIMARY guard against duplicate submissions
                return True
            
            # ═══════════════════════════════════════════════════════════════════
            # Load current epoch data (may be empty if gateway was down)
            # ═══════════════════════════════════════════════════════════════════
            weights_file = Path("validator_weights") / "validator_weights"
            miner_scores = {}
            current_epoch_lead_count = 0
            epoch_data = None
            
            if weights_file.exists():
                with open(weights_file, 'r') as f:
                    weights_data = json.load(f)
                
                if str(current_epoch) in weights_data:
                    epoch_data = weights_data[str(current_epoch)]
                    miner_scores = epoch_data.get("miner_scores", {})
                    current_epoch_lead_count = epoch_data.get("approved_lead_count", 0)
            
            # ═══════════════════════════════════════════════════════════════════
            # Constants for weight distribution
            # ═══════════════════════════════════════════════════════════════════
            UID_ZERO = 0  # LeadPoet revenue UID
            EXPECTED_UID_ZERO_HOTKEY = "5FNVgRnrxMibhcBGEAaajGrYjsaCn441a5HuGUBUNnxEBLo9"
            BASE_BURN_SHARE = 0.05         # 5% base burn to UID 0
            MAX_CURRENT_EPOCH_SHARE = 0.0  # 0% max to miners (current epoch)
            MAX_ROLLING_EPOCH_SHARE = 0.95 # 95% max to miners (rolling 30 epochs)
            # Dynamic MAX_LEADS_PER_EPOCH from gateway (fetched during process_gateway_validation_workflow)
            # If not in memory (e.g., after restart), try to recover from history file
            MAX_LEADS_PER_EPOCH = getattr(self, '_max_leads_per_epoch', None)
            if MAX_LEADS_PER_EPOCH is None:
                # Try to recover from history file (survives restarts)
                try:
                    history_file = Path("validator_weights") / "validator_weights_history"
                    if history_file.exists():
                        with open(history_file, 'r') as f:
                            history_data = json.load(f)
                        epoch_data = history_data.get(str(current_epoch), {})
                        MAX_LEADS_PER_EPOCH = epoch_data.get("max_leads_per_epoch", 50)
                        print(f"   ℹ️  Recovered max_leads_per_epoch={MAX_LEADS_PER_EPOCH} from history file")
                    else:
                        MAX_LEADS_PER_EPOCH = 50
                except Exception as e:
                    print(f"   ⚠️  Could not recover max_leads_per_epoch from history: {e}")
                    MAX_LEADS_PER_EPOCH = 50
            ROLLING_WINDOW = 30
            
            # ═══════════════════════════════════════════════════════════════════
            # Get rolling 30 epoch scores BEFORE checking if we should proceed
            # This ensures we still distribute rolling 15% even if gateway was down
            # ═══════════════════════════════════════════════════════════════════
            rolling_scores, rolling_lead_count = self.get_rolling_epoch_scores(current_epoch, window=ROLLING_WINDOW)
            
            # ═══════════════════════════════════════════════════════════════════
            # Check if we have ANYTHING to submit (current OR rolling)
            # If both are empty, submit 100% burn weights
            # ═══════════════════════════════════════════════════════════════════
            if not miner_scores and not rolling_scores:
                print(f"   ⚠️  No current epoch OR rolling epoch data for epoch {current_epoch}")
                print(f"   🔥 Submitting 100% burn weights (first epoch or history cleared)...")
                
                try:
                    # Verify UID 0 is correct before burning
                    actual_uid0_hotkey = self.metagraph.hotkeys[UID_ZERO]
                    if actual_uid0_hotkey != EXPECTED_UID_ZERO_HOTKEY:
                        print(f"   ❌ CRITICAL ERROR: UID 0 ownership changed!")
                        return False
                    
                    result = self.subtensor.set_weights(
                        netuid=self.config.netuid,
                        wallet=self.wallet,
                        uids=[UID_ZERO],
                        weights=[1.0],
                        wait_for_finalization=True
                    )
                    
                    if result:
                        print(f"   ✅ 100% burn weights submitted successfully")
                        # Note: Don't clear weights immediately - keep until epoch transition
                        # This prevents wrong resubmission if validator restarts
                        self._last_weight_submission_epoch = current_epoch
                        return True
                    else:
                        print(f"   ❌ Failed to submit burn weights")
                        return False
                        
                except Exception as e:
                    print(f"   ❌ Error submitting burn weights: {e}")
                    return False
            
            # Log what we have
            has_current_epoch = bool(miner_scores)
            has_rolling_history = bool(rolling_scores)
            
            if not has_current_epoch and has_rolling_history:
                print(f"\n{'='*80}")
                print(f"⚠️  GATEWAY DOWN FALLBACK: Epoch {current_epoch}")
                print(f"{'='*80}")
                print(f"   No leads received for current epoch (gateway was likely down)")
                print(f"   But rolling history exists: {len(rolling_scores)} miners, {rolling_lead_count} leads")
                print(f"   → Will distribute rolling 15% to historical miners")
                print(f"   → Will burn current epoch 10% (no leads to distribute)")
                print()
            
            # Normal case: we have current epoch data
            if has_current_epoch:
                print(f"\n{'='*80}")
                print(f"⚖️  SUBMITTING WEIGHTS FOR EPOCH {current_epoch}")
                print(f"{'='*80}")
                print(f"   Block: {current_block} (block {blocks_into_epoch}/360 into epoch)")
                print(f"   MAX_LEADS_PER_EPOCH: {MAX_LEADS_PER_EPOCH} (from gateway config)")
                print(f"   Current epoch miners: {len(miner_scores)}")
                print(f"   Current epoch points: {sum(miner_scores.values())}")
                print()
            
            # CRITICAL: Verify UID 0 is the expected LeadPoet hotkey (safety check)
            try:
                actual_uid0_hotkey = self.metagraph.hotkeys[UID_ZERO]
                if actual_uid0_hotkey != EXPECTED_UID_ZERO_HOTKEY:
                    print(f"   ❌ CRITICAL ERROR: UID 0 ownership changed!")
                    print(f"      Expected: {EXPECTED_UID_ZERO_HOTKEY[:20]}...")
                    print(f"      Actual:   {actual_uid0_hotkey[:20]}...")
                    print(f"      Revenue would go to WRONG address - aborting weight submission")
                    return False
            except Exception as e:
                print(f"   ❌ Error verifying UID 0 ownership: {e}")
                return False
            # Log rolling epoch data
            print(f"   Rolling {ROLLING_WINDOW} epoch miners: {len(rolling_scores)}")
            print(f"   Rolling {ROLLING_WINDOW} epoch points: {sum(rolling_scores.values()) if rolling_scores else 0}")
            print()
            
            # ═══════════════════════════════════════════════════════════════════
            # LINEAR EMISSIONS: Calculate approval rates and effective shares
            # Handle case where current epoch has no leads (gateway was down)
            # ═══════════════════════════════════════════════════════════════════
            # Current epoch approval rate (0 if no leads received)
            current_epoch_approval_rate = min(current_epoch_lead_count / MAX_LEADS_PER_EPOCH, 1.0) if current_epoch_lead_count > 0 else 0.0
            
            # Rolling epochs approval rate (max possible = MAX_LEADS_PER_EPOCH * ROLLING_WINDOW)
            max_rolling_leads = MAX_LEADS_PER_EPOCH * ROLLING_WINDOW
            rolling_approval_rate = min(rolling_lead_count / max_rolling_leads, 1.0) if max_rolling_leads > 0 else 0
            
            # Calculate effective shares (scaled by approval rate)
            effective_current_share = MAX_CURRENT_EPOCH_SHARE * current_epoch_approval_rate
            effective_rolling_share = MAX_ROLLING_EPOCH_SHARE * rolling_approval_rate
            
            # Calculate additional burn from unapproved slots
            unused_current_share = MAX_CURRENT_EPOCH_SHARE - effective_current_share
            unused_rolling_share = MAX_ROLLING_EPOCH_SHARE - effective_rolling_share
            total_burn_share = BASE_BURN_SHARE + unused_current_share + unused_rolling_share
            
            print(f"   📈 LINEAR EMISSIONS SCALING:")
            print(f"      Current epoch: {current_epoch_lead_count}/{MAX_LEADS_PER_EPOCH} leads = {current_epoch_approval_rate*100:.1f}% approval")
            print(f"         → {MAX_CURRENT_EPOCH_SHARE*100:.0f}% × {current_epoch_approval_rate*100:.1f}% = {effective_current_share*100:.2f}% to miners")
            print(f"         → {unused_current_share*100:.2f}% burned (unused slots)")
            print(f"      Rolling {ROLLING_WINDOW} epochs: {rolling_lead_count}/{max_rolling_leads} leads = {rolling_approval_rate*100:.1f}% approval")
            print(f"         → {MAX_ROLLING_EPOCH_SHARE*100:.0f}% × {rolling_approval_rate*100:.1f}% = {effective_rolling_share*100:.2f}% to miners")
            print(f"         → {unused_rolling_share*100:.2f}% burned (unused slots)")
            print(f"      Total burn: {BASE_BURN_SHARE*100:.0f}% base + {unused_current_share*100:.2f}% + {unused_rolling_share*100:.2f}% = {total_burn_share*100:.2f}%")
            print()
            
            # ═══════════════════════════════════════════════════════════════════
            # Combine all miner scores (current epoch + rolling) for UID mapping
            # ═══════════════════════════════════════════════════════════════════
            all_miner_hotkeys = set(miner_scores.keys()) | set(rolling_scores.keys())
            
            # Convert miner hotkeys to UIDs
            hotkey_to_uid = {}
            for hotkey in all_miner_hotkeys:
                try:
                    if hotkey in self.metagraph.hotkeys:
                        uid = self.metagraph.hotkeys.index(hotkey)
                        hotkey_to_uid[hotkey] = uid
                except Exception as e:
                    print(f"   ⚠️  Skipping miner {hotkey[:10]}...: {e}")
            
            if not hotkey_to_uid:
                # FALLBACK: No valid miner UIDs found - submit burn weights
                print(f"   ⚠️  No valid miner UIDs found")
                print(f"      Miners have left the subnet or are not registered")
                print(f"   🔥 Submitting burn weights instead...")
                
                try:
                    # Get subnet owner's hotkey
                    owner_hotkey = self.subtensor.query_subtensor(
                        "SubnetOwnerHotkey",
                        params=[self.config.netuid]
                    )
                    
                    # Get owner's UID
                    burn_uid = self.subtensor.get_uid_for_hotkey_on_subnet(
                        hotkey_ss58=str(owner_hotkey),
                        netuid=self.config.netuid
                    )
                    
                    print(f"   🔥 Burn UID: {burn_uid} (subnet owner)")
                    
                    # Submit burn weights (100% to owner)
                    result = self.subtensor.set_weights(
                        netuid=self.config.netuid,
                        wallet=self.wallet,
                        uids=[burn_uid],
                        weights=[1.0],
                        wait_for_finalization=True
                    )
                    
                    if result:
                        print(f"   ✅ Burn weights submitted successfully")
                        
                        # Note: Don't clear weights immediately - keep until epoch transition
                        # This prevents wrong resubmission if validator restarts
                        self._last_weight_submission_epoch = current_epoch
                        
                        return True
                    else:
                        print(f"   ❌ Failed to submit burn weights")
                        return False
                
                except Exception as e:
                    print(f"   ❌ Error submitting burn weights: {e}")
                    return False
            
            # ═══════════════════════════════════════════════════════════════════
            # Filter scores to only include REGISTERED miners
            # Deregistered miners' shares go to BURN (UID 0), not redistributed
            # ═══════════════════════════════════════════════════════════════════
            registered_current_scores = {h: p for h, p in miner_scores.items() if h in hotkey_to_uid}
            registered_rolling_scores = {h: p for h, p in rolling_scores.items() if h in hotkey_to_uid}
            
            # Calculate totals - use ALL miners for denominator (deregistered share goes to burn)
            all_current_total = sum(miner_scores.values()) if miner_scores else 0
            all_rolling_total = sum(rolling_scores.values()) if rolling_scores else 0
            registered_current_total = sum(registered_current_scores.values()) if registered_current_scores else 0
            registered_rolling_total = sum(registered_rolling_scores.values()) if registered_rolling_scores else 0
            
            # Calculate how much goes to deregistered miners (this will be burned)
            deregistered_current_points = all_current_total - registered_current_total
            deregistered_rolling_points = all_rolling_total - registered_rolling_total
            
            # Calculate burn amounts for deregistered miners
            current_dereg_burn = 0
            rolling_dereg_burn = 0
            if all_current_total > 0 and deregistered_current_points > 0:
                current_dereg_burn = effective_current_share * (deregistered_current_points / all_current_total)
            if all_rolling_total > 0 and deregistered_rolling_points > 0:
                rolling_dereg_burn = effective_rolling_share * (deregistered_rolling_points / all_rolling_total)
            
            # Log deregistered miners
            if deregistered_current_points > 0 or deregistered_rolling_points > 0:
                print(f"   ⚠️  Deregistered miners' shares → BURN:")
                if deregistered_current_points > 0:
                    print(f"      Current epoch: {deregistered_current_points} pts deregistered → {current_dereg_burn*100:.2f}% to burn")
                if deregistered_rolling_points > 0:
                    print(f"      Rolling epochs: {deregistered_rolling_points} pts deregistered → {rolling_dereg_burn*100:.2f}% to burn")
                print()
            
            # Effective shares for registered miners only
            effective_current_to_miners = effective_current_share - current_dereg_burn
            effective_rolling_to_miners = effective_rolling_share - rolling_dereg_burn
            
            print(f"\n    Alpha Split (with linear emissions):")
            print(f"      {(total_burn_share + current_dereg_burn + rolling_dereg_burn)*100:.2f}% → UID {UID_ZERO} (Burn)")
            print(f"      {effective_current_to_miners*100:.2f}% → Current epoch miners ({len(registered_current_scores)} registered)")
            print(f"      {effective_rolling_to_miners*100:.2f}% → Rolling {ROLLING_WINDOW} epoch miners ({len(registered_rolling_scores)} registered)")
            print()
            
            # Build final UIDs and weights
            uid_weights = {}
            
            # UID 0 gets: base burn + unused slots + deregistered miners' shares
            uid_weights[UID_ZERO] = total_burn_share + current_dereg_burn + rolling_dereg_burn
            
            # Distribute to REGISTERED current epoch miners
            print(f"    Current Epoch ({effective_current_to_miners*100:.2f}% to registered miners):")
            if registered_current_total > 0 and effective_current_to_miners > 0:
                for hotkey, points in registered_current_scores.items():
                    uid = hotkey_to_uid[hotkey]
                    miner_proportion = points / registered_current_total
                    miner_weight = effective_current_to_miners * miner_proportion
                    
                    if uid not in uid_weights:
                        uid_weights[uid] = 0
                    uid_weights[uid] += miner_weight
                    
                    print(f"      UID {uid}: {points}/{registered_current_total} pts = {miner_weight*100:.2f}%")
            else:
                print(f"      (No registered miners)")
            
            # Distribute to REGISTERED rolling epoch miners
            print(f"\n    Rolling {ROLLING_WINDOW} Epochs ({effective_rolling_to_miners*100:.2f}% to registered miners):")
            if registered_rolling_total > 0 and effective_rolling_to_miners > 0:
                for hotkey, points in registered_rolling_scores.items():
                    uid = hotkey_to_uid[hotkey]
                    miner_proportion = points / registered_rolling_total
                    miner_weight = effective_rolling_to_miners * miner_proportion
                    
                    if uid not in uid_weights:
                        uid_weights[uid] = 0
                    uid_weights[uid] += miner_weight
                    
                    print(f"      UID {uid}: {points}/{registered_rolling_total} pts = {miner_weight*100:.2f}%")
            else:
                print(f"      (No registered miners)")
            
            # Convert to final lists
            final_uids = list(uid_weights.keys())
            final_weights = list(uid_weights.values())
            
            print()
            print(f"   Final weights (should sum to 1.0):")
            for uid in sorted(final_uids):
                weight = uid_weights[uid]
                if uid == UID_ZERO:
                    print(f"      UID {uid} (Burn): {weight*100:.2f}%")
                else:
                    print(f"      UID {uid}: {weight*100:.2f}%")
            print(f"   Total: {sum(final_weights)*100:.2f}%")
            
            # Verify weights sum to 1.0 (with small floating point tolerance)
            weight_sum = sum(final_weights)
            if not (0.999 <= weight_sum <= 1.001):
                print(f"   ❌ ERROR: Weights sum to {weight_sum}, not 1.0!")
                return False
            
            # Use final_uids and final_weights
            uids = final_uids
            normalized_weights = final_weights
            
            # ═══════════════════════════════════════════════════════════════════
            # TEE GATEWAY SUBMISSION (Phase 2.3)
            # Submit to gateway BEFORE chain for auditor validators
            # ═══════════════════════════════════════════════════════════════════
            tee_event_hash = None
            if TEE_AVAILABLE and os.environ.get("ENABLE_TEE_SUBMISSION", "").lower() == "true":
                print(f"\n🔐 TEE weight submission enabled - submitting to gateway first...")
                tee_event_hash = await self._submit_weights_to_gateway(
                    epoch_id=current_epoch,
                    block=current_block,
                    uids=uids,
                    weights=normalized_weights,
                )
                if tee_event_hash:
                    print(f"   ✅ Gateway accepted weights (hash: {tee_event_hash[:16]}...)")
                else:
                    # Gateway submission failed - but we still proceed to chain
                    # This ensures chain submission is not blocked by gateway issues
                    print(f"   ⚠️ Gateway submission failed - proceeding to chain anyway")
            elif TEE_AVAILABLE:
                print(f"\nℹ️ TEE available but submission disabled (set ENABLE_TEE_SUBMISSION=true to enable)")
            
            # Submit to Bittensor chain
            print(f"\n📡 Submitting weights to Bittensor chain...")
            result = self.subtensor.set_weights(
                netuid=self.config.netuid,
                wallet=self.wallet,
                uids=uids,
                weights=normalized_weights,
                wait_for_finalization=True
            )
            
            if result:
                print(f"✅ Successfully submitted weights to Bittensor chain")
                print(f"{'='*80}\n")
                
                # CRITICAL: Mark this epoch as submitted BEFORE any cleanup
                # This prevents duplicate submissions if the function is called again
                self._last_weight_submission_epoch = current_epoch
                
                # Archive weights to history (only if we had current epoch data)
                if epoch_data is not None:
                    self.archive_weights_to_history(current_epoch, epoch_data)
                else:
                    # Gateway was down - just mark in history that we submitted rolling-only weights
                    print(f"   📚 Submitted rolling-only weights (no current epoch leads received)")
                
                # Note: Don't clear weights immediately - keep until epoch transition
                # This prevents wrong resubmission if validator restarts within the same epoch
                # The _last_weight_submission_epoch guard prevents duplicates during normal operation
                # Old epoch data in the file doesn't interfere since we only look up current_epoch
                
                return True
            else:
                print(f"❌ Failed to submit weights to Bittensor chain")
                print(f"{'='*80}\n")
                return False
                
        except Exception as e:
            bt.logging.error(f"Error submitting weights at epoch end: {e}")
            import traceback
            bt.logging.error(traceback.format_exc())
            return False
    
    async def _submit_weights_to_gateway(
        self,
        epoch_id: int,
        block: int,
        uids: List[int],
        weights: List[float],
    ) -> Optional[str]:
        """
        Submit weights to TEE gateway for auditor validators (Phase 2.3).
        
        Uses CANONICAL format: UIDs + u16 weights, not floats/hotkeys.
        See business_files/tasks8.md for exact format specification.
        
        SECURITY:
        - Signs weights inside enclave (private key never leaves)
        - Attestation includes epoch_id for replay protection
        - Binding message proves hotkey authorized enclave
        
        Args:
            epoch_id: Current epoch
            block: Block number when weights were computed
            uids: List of UIDs (sorted ascending)
            weights: Corresponding float weights (will be converted to u16)
            
        Returns:
            weight_submission_event_hash if accepted, None if failed/rejected
        """
        # Check if TEE is available
        if not TEE_AVAILABLE:
            bt.logging.warning("⚠️ TEE modules not available - skipping gateway submission")
            bt.logging.warning("   Install validator_tee package to enable gateway submission")
            return None
        
        # Check if enclave is initialized
        if not is_keypair_initialized():
            bt.logging.warning("⚠️ Validator enclave not initialized - skipping gateway submission")
            return None
        
        # Check if gateway submission is enabled
        gateway_url = os.environ.get("GATEWAY_URL", "http://54.226.209.164:8000")
        if os.environ.get("DISABLE_GATEWAY_WEIGHT_SUBMISSION", "").lower() == "true":
            bt.logging.info("ℹ️ Gateway weight submission disabled via env var")
            return None
        
        try:
            netuid = self.config.netuid
            
            # Get expected chain endpoint for binding message
            expected_chain = os.environ.get(
                "EXPECTED_CHAIN", 
                "wss://entrypoint-finney.opentensor.ai:443"
            )
            
            # Get git commit for version info
            try:
                git_commit_short = subprocess.check_output(
                    ["git", "rev-parse", "--short", "HEAD"],
                    text=True,
                    stderr=subprocess.DEVNULL,
                ).strip()
            except Exception:
                git_commit_short = "unknown"
            
            # ═══════════════════════════════════════════════════════════════════
            # Step 1: Convert floats to u16 using canonical function
            # ═══════════════════════════════════════════════════════════════════
            weights_u16 = normalize_to_u16(uids, weights)
            
            # Filter to sparse (remove zeros) and ensure sorted
            sparse_pairs = [(uid, w) for uid, w in zip(uids, weights_u16) if w > 0]
            sparse_pairs.sort(key=lambda x: x[0])  # Sort by UID
            
            if not sparse_pairs:
                bt.logging.warning("⚠️ No non-zero weights after u16 conversion")
                return None
            
            sparse_uids = [p[0] for p in sparse_pairs]
            sparse_weights_u16 = [p[1] for p in sparse_pairs]
            
            # ═══════════════════════════════════════════════════════════════════
            # Step 2: Sign weights with enclave key
            # ═══════════════════════════════════════════════════════════════════
            weights_hash, signature_hex = sign_weights(
                netuid=netuid,
                epoch_id=epoch_id,
                block=block,
                uids=sparse_uids,
                weights_u16=sparse_weights_u16,
            )
            
            enclave_pubkey = get_enclave_pubkey()
            
            # ═══════════════════════════════════════════════════════════════════
            # Step 3: Get attestation (includes epoch_id for replay protection)
            # ═══════════════════════════════════════════════════════════════════
            attestation_b64 = get_attestation(epoch_id=epoch_id)
            code_hash = get_code_hash()
            
            # ═══════════════════════════════════════════════════════════════════
            # Step 4: Build binding message (proves hotkey authorized enclave)
            # ═══════════════════════════════════════════════════════════════════
            binding_message = create_binding_message(
                netuid=netuid,
                chain=expected_chain,
                enclave_pubkey=enclave_pubkey,
                validator_code_hash=code_hash,
                version=git_commit_short,
            )
            
            # Sign binding message with hotkey (sr25519)
            hotkey_signature = self.wallet.hotkey.sign(binding_message.encode())
            
            # ═══════════════════════════════════════════════════════════════════
            # Step 5: Build submission payload (matches WeightSubmission model)
            # ═══════════════════════════════════════════════════════════════════
            submission = {
                "netuid": netuid,
                "epoch_id": epoch_id,
                "block": block,
                "uids": sparse_uids,
                "weights_u16": sparse_weights_u16,
                "weights_hash": weights_hash,
                "validator_hotkey": self.wallet.hotkey.ss58_address,
                "validator_enclave_pubkey": enclave_pubkey,
                "validator_signature": signature_hex,
                "validator_attestation_b64": attestation_b64,
                "validator_code_hash": code_hash,
                "binding_message": binding_message,
                "validator_hotkey_signature": hotkey_signature.hex(),
            }
            
            # ═══════════════════════════════════════════════════════════════════
            # Step 6: Submit to gateway
            # ═══════════════════════════════════════════════════════════════════
            print(f"📡 Submitting TEE-signed weights to gateway...")
            print(f"   Endpoint: {gateway_url}/weights/submit")
            print(f"   Epoch: {epoch_id}, Block: {block}, UIDs: {len(sparse_uids)}")
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{gateway_url}/weights/submit",
                    json=submission,
                    timeout=aiohttp.ClientTimeout(total=60),
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        event_hash = result.get("weight_submission_event_hash")
                        print(f"✅ Weights accepted by gateway")
                        print(f"   Event hash: {event_hash[:16] if event_hash else 'N/A'}...")
                        return event_hash
                        
                    elif response.status == 409:
                        # Duplicate submission - already submitted for this epoch
                        print(f"⚠️ Duplicate submission rejected (already submitted for epoch {epoch_id})")
                        return None
                        
                    else:
                        error = await response.text()
                        print(f"❌ Gateway rejected submission: {response.status}")
                        print(f"   Error: {error[:200]}...")
                        return None
                        
        except aiohttp.ClientError as e:
            bt.logging.error(f"Network error submitting to gateway: {e}")
            return None
        except Exception as e:
            bt.logging.error(f"Error submitting weights to gateway: {e}")
            import traceback
            bt.logging.error(traceback.format_exc())
            return None
    
    def archive_weights_to_history(self, epoch_id: int, epoch_data: Dict):
        """
        [DEPRECATED] Archive submitted weights to validator_weights_history for record keeping.
        
        This function is now a no-op because validator_weights_history is updated
        in real-time by accumulate_miner_weights() after each lead validation.
        
        The history file is already up-to-date when weights are submitted.
        
        Args:
            epoch_id: Epoch number
            epoch_data: Dict containing epoch weights data
        """
        try:
            weights_dir = Path("validator_weights")
            weights_dir.mkdir(exist_ok=True)
            history_file = weights_dir / "validator_weights_history"
            
            # Load existing history (should already have this epoch from real-time updates)
            if history_file.exists():
                with open(history_file, 'r') as f:
                    history = json.load(f)
            else:
                # Should never happen - history is created in accumulate_miner_weights()
                bt.logging.warning("History file doesn't exist at submission time - creating it now")
                history = {"curators": [], "sourcers_of_curated": []}
            
            # Add submission timestamp to the existing epoch entry
            if str(epoch_id) in history:
                history[str(epoch_id)]["submitted_at"] = datetime.utcnow().isoformat()
                history[str(epoch_id)]["submitted_to_chain"] = True
                
                # Save updated history
                with open(history_file, 'w') as f:
                    json.dump(history, f, indent=2)
                
                print(f"   📚 Marked epoch {epoch_id} as submitted in history")
            else:
                # Shouldn't happen - history should already have this epoch
                bt.logging.warning(f"Epoch {epoch_id} not found in history at submission time")
            
        except Exception as e:
            bt.logging.error(f"Failed to update history submission status: {e}")
    
    def _clear_old_epochs_from_weights(self, current_epoch: int):
        """
        Clear OLD epochs from validator_weights file at epoch transition.
        
        Called at the START of each new epoch to remove data from previous epochs.
        This prevents file bloat while keeping current epoch data intact.
        
        Args:
            current_epoch: The NEW epoch we're transitioning to
        """
        try:
            weights_file = Path("validator_weights") / "validator_weights"
            
            if not weights_file.exists():
                return
            
            with open(weights_file, 'r') as f:
                weights_data = json.load(f)
            
            # Find all epoch entries (numeric keys)
            epoch_keys = [k for k in weights_data.keys() if k.isdigit()]
            
            if not epoch_keys:
                return  # No epoch data to clear
            
            # Remove all epochs BEFORE the current epoch
            epochs_removed = 0
            for epoch_key in epoch_keys:
                epoch_id = int(epoch_key)
                if epoch_id < current_epoch:
                    del weights_data[epoch_key]
                    epochs_removed += 1
            
            if epochs_removed > 0:
                # Save the cleaned file
                with open(weights_file, 'w') as f:
                    json.dump(weights_data, f, indent=2)
                
                print(f"   🧹 Epoch transition: Cleared {epochs_removed} old epoch(s) from validator_weights")
            
        except Exception as e:
            bt.logging.error(f"Failed to clear old epochs from weights: {e}")
    
    def get_rolling_epoch_scores(self, current_epoch: int, window: int = 30) -> tuple:
        """
        Get aggregated miner scores and lead counts from the last N epochs (rolling window).
        
        This reads from validator_weights_history and sums up scores for each miner
        across the specified window of epochs.
        
        Args:
            current_epoch: Current epoch number
            window: Number of past epochs to include (default: 30)
            
        Returns:
            Tuple of:
            - Dict mapping miner_hotkey -> total_rep_score across rolling window
            - int: Total approved lead count across rolling window
        """
        try:
            history_file = Path("validator_weights") / "validator_weights_history"
            
            if not history_file.exists():
                print(f"   ℹ️  No history file found - no rolling scores available")
                return {}, 0
            
            with open(history_file, 'r') as f:
                history_data = json.load(f)
            
            # Calculate epoch range for rolling window
            # Include epochs from (current_epoch - window) to (current_epoch - 1)
            # We exclude current_epoch since that's handled separately by the 10% allocation
            start_epoch = current_epoch - window
            end_epoch = current_epoch - 1
            
            rolling_scores = {}
            rolling_lead_count = 0
            epochs_included = 0
            
            for epoch_str, epoch_data in history_data.items():
                # Skip non-epoch entries (curators, sourcers_of_curated)
                if not epoch_str.isdigit():
                    continue
                
                epoch_id = int(epoch_str)
                
                # Check if epoch is within rolling window
                if start_epoch <= epoch_id <= end_epoch:
                    epochs_included += 1
                    miner_scores = epoch_data.get("miner_scores", {})
                    
                    for hotkey, score in miner_scores.items():
                        if hotkey not in rolling_scores:
                            rolling_scores[hotkey] = 0
                        rolling_scores[hotkey] += score
                    
                    # Sum up approved lead counts for linear emissions
                    rolling_lead_count += epoch_data.get("approved_lead_count", 0)
            
            print(f"   📊 Rolling window: epochs {start_epoch}-{end_epoch} ({epochs_included} epochs with data)")
            print(f"   📊 Rolling scores: {len(rolling_scores)} miners, {rolling_lead_count} total approved leads")
            
            return rolling_scores, rolling_lead_count
            
        except Exception as e:
            bt.logging.error(f"Failed to get rolling epoch scores: {e}")
            return {}, 0
    
    def prune_history_file(self, current_epoch: int, max_epochs: int = 50):
        """
        Prune old epochs from validator_weights_history to prevent file bloat.
        
        Keeps only the most recent max_epochs entries.
        
        Args:
            current_epoch: Current epoch number
            max_epochs: Maximum epochs to retain (default: 50)
        """
        try:
            history_file = Path("validator_weights") / "validator_weights_history"
            
            if not history_file.exists():
                return
            
            with open(history_file, 'r') as f:
                history_data = json.load(f)
            
            # Find all epoch entries (numeric keys)
            epoch_entries = [k for k in history_data.keys() if k.isdigit()]
            
            if len(epoch_entries) <= max_epochs:
                return  # No pruning needed
            
            # Calculate cutoff epoch
            cutoff_epoch = current_epoch - max_epochs
            
            # Remove epochs older than cutoff
            epochs_removed = 0
            for epoch_str in epoch_entries:
                epoch_id = int(epoch_str)
                if epoch_id < cutoff_epoch:
                    del history_data[epoch_str]
                    epochs_removed += 1
            
            if epochs_removed > 0:
                # Save pruned history
                with open(history_file, 'w') as f:
                    json.dump(history_data, f, indent=2)
                
                print(f"   🗑️  Pruned {epochs_removed} old epochs from history (keeping last {max_epochs})")
            
        except Exception as e:
            bt.logging.error(f"Failed to prune history file: {e}")
    
    def calculate_and_submit_weights_local(self, validation_data: List[Dict]):
        """
        [DEPRECATED] Calculate miner weights based on LOCAL validation results (Passage 2).
        
        This function is now replaced by:
        - accumulate_miner_weights() - called after each lead validation
        - submit_weights_at_epoch_end() - called at block 345+ to submit accumulated weights
        
        Keeping for backwards compatibility, but new code should use the accumulation system.
        """
        # Accumulate weights instead of calculating at once
        for validation in validation_data:
            self.accumulate_miner_weights(
                miner_hotkey=validation['miner_hotkey'],
                rep_score=validation['rep_score'],
                decision=validation['decision']
            )
    
    async def process_pending_reveals(self):
        """
        Check if previous epochs need reveal submission (after epoch closes).
        
        REVEAL WINDOW LOGIC:
        - Epoch N: Submit hashes → Save reveals to pending_reveals.json
        - Epoch N+1: Submit hashes for N+1 → Reveal epoch N → Remove N from pending_reveals.json
        - Epoch N+2+: If epoch N wasn't revealed, DELETE as expired (too late)
        
        CRITICAL RULES:
        1. Reveals can ONLY be submitted in epoch N+1 (not N+2, N+3, etc.)
        2. If current_epoch > epoch_id + 1, the reveal window is EXPIRED → DELETE
        3. pending_reveals.json should only contain current epoch and previous epoch at most
        
        ASYNC VERSION: Uses async subtensor for block queries.
        """
        if not hasattr(self, '_pending_reveals'):
            print(f"[DEBUG] No _pending_reveals attribute - initializing empty dict")
            self._pending_reveals = {}
            return
        
        try:
            current_block = await self.get_current_block_async()
            epoch_length = 360
            current_epoch = current_block // epoch_length
            blocks_into_epoch = current_block % epoch_length
            
            if not self._pending_reveals:
                print(f"[DEBUG] No pending reveals to process (current epoch: {current_epoch})")
                return
            
            # CUTOFF: Don't submit reveals after block 328 (gateway deadline, consensus at 330)
            if blocks_into_epoch > 328:
                print(f"[DEBUG] Past reveal deadline (block {blocks_into_epoch}/360 > 328) - skipping reveal submission")
                return
            
            print(f"\n{'='*80}")
            print(f"🔍 CHECKING REVEALS: Current epoch {current_epoch}, Block {blocks_into_epoch}/328, Pending: {list(self._pending_reveals.keys())}")
            print(f"{'='*80}")
            
            from Leadpoet.utils.cloud_db import gateway_submit_reveal
            
            # CRITICAL: Clean up expired reveals BEFORE attempting submission
            # Reveal window is N+1 only - anything older should be purged
            epochs_to_remove = []
            for epoch_id in list(self._pending_reveals.keys()):
                # Check if reveal window has expired (current_epoch > epoch_id + 1)
                if current_epoch > epoch_id + 1:
                    print(f"   🗑️  Epoch {epoch_id} reveal window EXPIRED")
                    print(f"      Should have revealed in epoch {epoch_id + 1}, current is {current_epoch}")
                    print(f"      Removing from pending reveals (no longer valid)")
                    epochs_to_remove.append(epoch_id)
            
            # Remove expired epochs
            for epoch_id in epochs_to_remove:
                del self._pending_reveals[epoch_id]
            
            # Save state after cleanup
            if epochs_to_remove:
                self._save_pending_reveals()
                print(f"   ✅ Removed {len(epochs_to_remove)} expired epoch(s) from pending reveals")
                print(f"      Remaining: {list(self._pending_reveals.keys())}")
            
            # Check each pending epoch (only those still valid)
            epochs_to_reveal = list(self._pending_reveals.keys())
            for epoch_id in epochs_to_reveal:
                print(f"   📋 Epoch {epoch_id}: Current={current_epoch}, Ready={current_epoch > epoch_id}")
                
                # Reveal after epoch closes (current_epoch > epoch_id)
                if current_epoch > epoch_id:
                    reveal_data = self._pending_reveals[epoch_id]
                    print(f"   🔓 Revealing {len(reveal_data)} validations for epoch {epoch_id}...")
                    
                    # Format reveals for gateway
                    reveals = []
                    for validation in reveal_data:
                        # DEFENSE IN DEPTH: Provide fallback for any remaining null rejection_reasons
                        # (Should not happen with current code, but handles old corrupted data in pending_reveals)
                        rejection_reason = validation.get("rejection_reason")
                        if rejection_reason is None:
                            print(f"   ⚠️  Fixing null rejection_reason for lead {validation.get('lead_id', 'unknown')[:8]}...")
                            rejection_reason = {
                                "stage": "Unknown",
                                "check_name": "data_corruption",
                                "message": "Rejection reason was null (corrupted data from previous epoch)",
                                "failed_fields": []
                            }
                        
                        reveals.append({
                            "lead_id": validation["lead_id"],
                            "decision": validation["decision"],
                            "rep_score": validation["rep_score"],
                            "rejection_reason": rejection_reason,
                            "salt": validation["salt"]
                        })
                    
                    # Submit reveal
                    print(f"   📤 Submitting {len(reveals)} reveals to gateway...")
                    success = gateway_submit_reveal(self.wallet, epoch_id, reveals)
                    if success:
                        print(f"   ✅ Successfully revealed {len(reveals)} validations for epoch {epoch_id}")
                        print(f"   🗑️  Removing epoch {epoch_id} from pending_reveals.json")
                        bt.logging.info(f"✅ Revealed validation for epoch {epoch_id}")
                        del self._pending_reveals[epoch_id]
                        self._save_pending_reveals()  # Save immediately after successful reveal
                        print(f"   💾 Updated pending_reveals.json (remaining epochs: {list(self._pending_reveals.keys())})")
                    else:
                        print(f"   ❌ Failed to reveal validation for epoch {epoch_id}")
                        bt.logging.error(f"Failed to reveal validation for epoch {epoch_id}")
                else:
                    print(f"   ⏳ Epoch {epoch_id} not yet closed, waiting...")
                        
        except Exception as e:
            print(f"[DEBUG] Exception in process_pending_reveals: {e}")
            import traceback
            print(f"[DEBUG] Traceback:\n{traceback.format_exc()}")
            bt.logging.error(f"Error processing reveals: {e}")

    def process_sourced_leads_continuous(self):
        """
        CONSENSUS VERSION: Process leads with consensus-based validation.
        Pulls prospects using first-come-first-served, validates them,
        and submits assessments to the consensus tracking system.
        """
        # Skip if processing broadcast request
        if self.processing_broadcast:
            return  # Pause sourcing during broadcast processing

        try:
            # Import consensus functions
            from Leadpoet.utils.cloud_db import submit_validation_assessment
            import uuid
            
            # Fetch prospects using the new consensus-aware function
            # Returns list of {'prospect_id': UUID, 'data': lead_dict}
            prospects_batch = fetch_prospects_from_cloud(
                wallet=self.wallet,
                limit=50,
                network=self.config.subtensor.network,
                netuid=self.config.netuid
            )

            if not prospects_batch:
                time.sleep(5)  # Wait longer if no prospects available
                return

            print(f"🛎️  Pulled {len(prospects_batch)} prospects from queue (consensus mode)")
            
            # Process each prospect
            for prospect_item in prospects_batch:
                try:
                    # Extract prospect_id and lead data based on the new format
                    if isinstance(prospect_item, dict) and 'prospect_id' in prospect_item:
                        # New consensus format: {'prospect_id': UUID, 'data': lead_dict}
                        prospect_id = prospect_item['prospect_id']
                        lead = prospect_item['data']
                    else:
                        # Fallback for old format (direct lead data)
                        prospect_id = str(uuid.uuid4())  # Generate one if not provided
                        lead = prospect_item
                    
                    # Generate unique lead_id for this validation
                    lead_id = str(uuid.uuid4())
                    
                    # Extract miner info for logging
                    if not lead or not isinstance(lead, dict):
                        bt.logging.error(f"Invalid lead data for prospect {prospect_id[:8]}: {type(lead)}")
                        continue
                        
                    miner_hotkey = lead.get("miner_hotkey", "unknown")
                    business_name = get_field(lead, 'business', 'website', default='Unknown')
                    email = get_email(lead, default='?')
                    
                    print(f"\n🟣 Validating prospect {prospect_id[:8]}...")
                    print(f"   Lead ID: {lead_id[:8]}...")
                    print(f"   Business: {business_name}")
                    print(f"   Email: {email}")
                    print(f"   Miner: {miner_hotkey[:10] if miner_hotkey and miner_hotkey != 'unknown' else 'unknown'}...")
                    
                    # Run async validate_lead in sync context
                    try:
                        result = asyncio.run(self.validate_lead(lead))
                    except Exception as validation_error:
                        # Check if this is an EmailVerificationUnavailableError
                        from validator_models.automated_checks import EmailVerificationUnavailableError
                        if isinstance(validation_error, EmailVerificationUnavailableError):
                            print(f"❌ Lead not processed due to API error\n")
                            continue  # Skip this lead entirely - don't submit anything
                        else:
                            # Some other error - re-raise it
                            raise
                    
                    # Extract validation results and enhanced lead data
                    is_valid = result.get("is_legitimate", False)
                    rejection_reason = result.get("reason", None)  # Now a structured dict from Task 3.1
                    enhanced_lead = result.get("enhanced_lead", lead)  # Get enhanced lead with DNSBL/WHOIS data
                    
                    # Log validation result
                    if is_valid:
                        print(f"   ✅ Valid")
                    else:
                        # Extract message from rejection_reason dict for logging
                        if isinstance(rejection_reason, dict):
                            reason_msg = rejection_reason.get("message", "Unknown error")
                        else:
                            reason_msg = str(rejection_reason) if rejection_reason else "Unknown error"
                        print(f"   ❌ Invalid: {reason_msg}")
                    
                    # Submit validation assessment to consensus system with enhanced lead data
                    submission_success = submit_validation_assessment(
                        wallet=self.wallet,
                        prospect_id=prospect_id,
                        lead_id=lead_id,
                        lead_data=enhanced_lead,  # Use enhanced lead with DNSBL/WHOIS data
                        is_valid=is_valid,
                        rejection_reason=rejection_reason if not is_valid else None,  # Pass structured rejection
                        network=self.config.subtensor.network,
                        netuid=self.config.netuid
                    )
                    
                    if submission_success:
                        print("   📤 Assessment submitted to consensus system")
                        print(f"✅ Processed 1 prospect in consensus mode\n")
                    else:
                        print("   ⚠️ Failed to submit assessment to consensus system")
                    
                    # Note: We do NOT directly save to leads table anymore
                    # The consensus system will handle that when 3 validators agree
                    
                except Exception as e:
                    print(f"   ❌ Error processing prospect: {e}")
                    bt.logging.error(f"Error processing prospect: {e}")
                    import traceback
                    bt.logging.debug(traceback.format_exc())
                    continue
            
        except Exception as e:
            bt.logging.error(f"process_sourced_leads_continuous failure: {e}")
            import traceback
            bt.logging.debug(traceback.format_exc())
            time.sleep(5)

# ─────────────────────────────────────────────────────────
#  NEW: handle buyer curation requests coming via Cloud Run
# ─────────────────────────────────────────────────────────
    def process_curation_requests_continuous(self):
        req = fetch_curation_requests()
        if not req:
            return

        print(f"\n💼 Buyer curation request: {req}")
        syn = LeadRequest(num_leads=req["num_leads"],
                          business_desc=req["business_desc"])

        # run the existing async pipeline inside the event-loop
        leads = asyncio.run(self.forward(syn)).leads

        # ── annotate each lead with the curation timestamp (seconds since epoch)
        curated_at = time.time()
        for lead in leads:
         
            lead["created_at"]    = datetime.utcfromtimestamp(curated_at).isoformat() + "Z"

        push_curation_result({"request_id": req["request_id"], "leads": leads})
        print(f"✅ Curated {len(leads)} leads for request {req['request_id']}")


    async def process_broadcast_requests_continuous(self):
        """
        Continuously poll for broadcast API requests from Firestore and process them.
        """
        await asyncio.sleep(2)
        print("📡 Polling for broadcast API requests... (will notify when requests are found)")

        poll_count = 0
        while True:
            try:
                poll_count += 1

                # Fetch pending broadcast requests from Firestore
                from Leadpoet.utils.cloud_db import fetch_broadcast_requests
                requests_list = fetch_broadcast_requests(self.wallet, role="validator")

                # fetch_broadcast_requests() will print when requests are found
                # No need to log anything here when empty

                if requests_list:
                    print(f"🔔 Found {len(requests_list)} NEW broadcast request(s) to process!")

                for req in requests_list:
                    request_id = req.get("request_id")

                    # Skip if already processed locally
                    if request_id in self._processed_requests:
                        print(f"⏭️  Skipping already processed request {request_id[:8]}...")
                        continue

                    # Mark as processed locally
                    self._processed_requests.add(request_id)

                    num_leads = req.get("num_leads", 1)
                    business_desc = req.get("business_desc", "")

                    # Set flag IMMEDIATELY to pause sourcing
                    self.processing_broadcast = True

                    print(f"\n📨 🔔 BROADCAST API REQUEST RECEIVED {request_id[:8]}...")
                    print(f"   Requested: {num_leads} leads")
                    print(f"   Description: {business_desc[:50]}...")
                    print(f"   🕐 Request received at {time.strftime('%H:%M:%S')}")
                    print("   ⏳ Waiting up to 180 seconds for miners to send curated leads...")

                    try:
                        # Wait for miners to send curated leads to Firestore
                        from Leadpoet.utils.cloud_db import fetch_miner_leads_for_request

                        MAX_WAIT = 180  
                        POLL_INTERVAL = 2  # Poll every 2 seconds

                        miner_leads_collected = []
                        start_time = time.time()
                        polls_done = 0

                        while time.time() - start_time < MAX_WAIT:
                            submissions = fetch_miner_leads_for_request(request_id)

                            if submissions:
                                # Flatten all leads from all miners
                                for submission in submissions:
                                    leads = submission.get("leads", [])
                                    miner_leads_collected.extend(leads)

                                if miner_leads_collected:
                                    elapsed = time.time() - start_time
                                    bt.logging.info(f"📥 Received leads from {len(submissions)} miner(s) after {elapsed:.1f}s")
                                    break

                            # Progress update every 10 seconds
                            polls_done += 1
                            if polls_done % 5 == 0:  # Every 10 seconds (5 polls * 2 sec)
                                elapsed = time.time() - start_time
                                bt.logging.info(f"⏳ Still waiting for miners... ({elapsed:.0f}s / {MAX_WAIT}s elapsed)")

                            await asyncio.sleep(POLL_INTERVAL)

                        if not miner_leads_collected:
                            bt.logging.warning(f"⚠️  No miner leads received after {MAX_WAIT}s, skipping ranking")
                            continue

                        bt.logging.info(f"📊 Received {len(miner_leads_collected)} total leads from miners")

                        # Rank leads using LLM scoring (TWO rounds with BATCHING)
                        if miner_leads_collected:
                            print(f"🔍 Ranking {len(miner_leads_collected)} leads with LLM...")
                            scored_leads = []

                            # Initialize aggregation dictionary for each lead
                            aggregated = {id(lead): 0.0 for lead in miner_leads_collected}
                            failed_leads = set()  # Track leads that failed LLM scoring

                            # ROUND 1: First LLM scoring (BATCHED)
                            first_model = random.choice(AVAILABLE_MODELS)
                            print(f"🔄 LLM round 1/2 (model: {first_model})")
                            batch_scores_r1 = _llm_score_batch(miner_leads_collected, business_desc, first_model)
                            for lead in miner_leads_collected:
                                score = batch_scores_r1.get(id(lead))
                                if score is None:
                                    failed_leads.add(id(lead))
                                    print("⚠️  LLM failed for lead, will skip this lead")
                                else:
                                    aggregated[id(lead)] += score

                            # ROUND 2: Second LLM scoring (BATCHED, random model selection)
                            # Only score leads that didn't fail in round 1
                            leads_for_r2 = [lead for lead in miner_leads_collected if id(lead) not in failed_leads]
                            if leads_for_r2:
                                second_model = random.choice(AVAILABLE_MODELS)
                                print(f"🔄 LLM round 2/2 (model: {second_model})")
                                batch_scores_r2 = _llm_score_batch(leads_for_r2, business_desc, second_model)
                                for lead in leads_for_r2:
                                    score = batch_scores_r2.get(id(lead))
                                    if score is None:
                                        failed_leads.add(id(lead))
                                        print("⚠️  LLM failed for lead, will skip this lead")
                                    else:
                                        aggregated[id(lead)] += score

                            # Apply aggregated scores to leads (skip failed ones)
                            for lead in miner_leads_collected:
                                if id(lead) not in failed_leads:
                                    lead["intent_score"] = round(aggregated[id(lead)], 3)
                                    scored_leads.append(lead)

                            if not scored_leads:
                                print("❌ All leads failed LLM scoring")
                                continue

                            # Sort by aggregated intent_score and take top N
                            scored_leads.sort(key=lambda x: x["intent_score"], reverse=True)
                            top_leads = scored_leads[:num_leads]

                            print(f"✅ Ranked top {len(top_leads)} leads:")
                            for i, lead in enumerate(top_leads, 1):
                                business = get_company(lead, default='Unknown')[:30]
                                score = lead.get('intent_score', 0)
                                print(f"  {i}. {business} (score={score:.3f})")

                        # SUBMIT VALIDATOR RANKING for consensus
                        try:
                            validator_trust = self.metagraph.validator_trust[self.uid].item()

                            ranking_submission = []
                            for rank, lead in enumerate(top_leads, 1):
                                ranking_submission.append({
                                    "lead": lead,
                                    "score": lead.get("intent_score", 0.0),
                                    "rank": rank,
                                })

                            success = push_validator_ranking(
                                wallet=self.wallet,
                                request_id=request_id,
                                ranked_leads=ranking_submission,
                                validator_trust=validator_trust
                            )

                            if success:
                                print(f"📊 Submitted ranking for consensus (trust={validator_trust:.4f})")
                            else:
                                print("⚠️  Failed to submit ranking for consensus")

                        except Exception as e:
                            print(f"⚠️  Error submitting validator ranking: {e}")
                            bt.logging.error(f"Error submitting validator ranking: {e}")

                        print(f"✅ Validator {self.wallet.hotkey.ss58_address[:10]}... completed processing broadcast {request_id[:8]}...")

                    except Exception as e:
                        print(f"❌ Error processing broadcast request {request_id[:8]}...: {e}")
                        bt.logging.error(f"Error processing broadcast request: {e}")
                        import traceback
                        bt.logging.error(traceback.format_exc())

                    finally:
                        # Always resume sourcing after processing
                        self.processing_broadcast = False

            except Exception as e:
                # Catch any errors in the outer loop (fetching requests, etc.)
                bt.logging.error(f"Error in broadcast polling loop: {e}")
                import traceback
                bt.logging.error(traceback.format_exc())

            # Clear old processed requests every 100 iterations to prevent memory buildup
            if poll_count % 100 == 0:
                bt.logging.info(f"🧹 Clearing old processed requests cache ({len(self._processed_requests)} entries)")
                self._processed_requests.clear()

            # Sleep before next poll
            await asyncio.sleep(1)  

    def move_to_validated_leads(self, lead, score):
        """
        [DEPRECATED IN CONSENSUS MODE]
        This function is no longer used when consensus validation is enabled.
        Leads are now saved through the consensus system after 3 validators agree.
        See submit_validation_assessment() in cloud_db.py instead.
        """
        # Prepare lead data
        lead["validator_hotkey"] = self.wallet.hotkey.ss58_address
        lead["validated_at"] = datetime.now(timezone.utc).isoformat()

        try:
            # Save to Supabase (write-only, no duplicate checking)
            if not self.supabase_client:
                bt.logging.error("❌ Supabase client not available - cannot save validated lead")
                return
                
            success = self.save_validated_lead_to_supabase(lead)
            email = get_email(lead, default='?')
            biz = get_field(lead, "business", "website")
            
            if success:
                print(f"✅ Added verified lead to Supabase → {biz} ({email})")
            else:
                # Duplicate or error - already logged in save function
                pass
                
        except Exception as e:
            bt.logging.error(f"Failed to save lead to Supabase: {e}")

    # Local prospect queue no longer exists
    def remove_from_prospect_queue(self, lead):
        return

    def is_disposable_email(self, email):
        """Check if email is from a disposable email provider"""
        disposable_domains = {
            '10minutemail.com', 'guerrillamail.com', 'mailinator.com', 'tempmail.org',
            'throwaway.email', 'temp-mail.org', 'yopmail.com', 'getnada.com'
        }
        domain = email.split('@')[-1].lower()
        return domain in disposable_domains

    def check_domain_legitimacy(self, domain):
        """Return True iff the domain looks syntactically valid (dot & no spaces)."""
        try:
            return "." in domain and " " not in domain
        except Exception:
            return False

    def should_run_deep_verification(self, lead: Dict) -> bool:
        """
        Determine if lead should undergo deep verification.
        
        Returns True for:
        - 100% of licensed_resale submissions
        - 5% random sample of other submissions
        
        Deep verification includes:
        - License OCR validation (for licensed_resale)
        - Cross-domain authenticity checks
        - Behavioral anomaly scoring
        """
        source_type = lead.get("source_type", "")
        
        # Always verify licensed resale
        if source_type == "licensed_resale":
            bt.logging.info(f"🔬 Deep verification triggered: licensed_resale source")
            return True
        
        # 5% random sample for others
        if random.random() < 0.05:
            bt.logging.info(f"🔬 Deep verification triggered: random 5% sample")
            return True
        
        return False

    async def run_deep_verification(self, lead: Dict) -> Dict:
        """
        Execute deep verification checks.
        
        Returns dict with:
        - passed: bool (overall pass/fail)
        - checks: list of individual check results
        - manual_review_required: bool (if flagged for admin review)
        """
        results = {
            "passed": True,
            "checks": [],
            "manual_review_required": False
        }
        
        # Check 1: License OCR validation (if applicable)
        if lead.get("source_type") == "licensed_resale":
            bt.logging.info("   🔍 Deep Check 1: License OCR validation")
            ocr_result = await self.verify_license_ocr(lead)
            results["checks"].append(ocr_result)
            
            if not ocr_result["passed"]:
                results["passed"] = False
                bt.logging.warning(f"   ❌ License OCR failed: {ocr_result['reason']}")
            else:
                bt.logging.info(f"   ✅ License OCR: {ocr_result['reason']}")
            
            if ocr_result.get("manual_review_required"):
                results["manual_review_required"] = True
        
        # Check 2: Cross-domain authenticity
        bt.logging.info("   🔍 Deep Check 2: Cross-domain authenticity")
        domain_result = await self.verify_cross_domain_authenticity(lead)
        results["checks"].append(domain_result)
        
        if not domain_result["passed"]:
            results["passed"] = False
            bt.logging.warning(f"   ❌ Cross-domain check failed: {domain_result['reason']}")
        else:
            bt.logging.info(f"   ✅ Cross-domain: {domain_result['reason']}")
        
        # Check 3: Behavioral anomaly scoring
        bt.logging.info("   🔍 Deep Check 3: Behavioral anomaly scoring")
        anomaly_result = await self.score_behavioral_anomalies(lead)
        results["checks"].append(anomaly_result)
        
        if not anomaly_result["passed"]:
            results["passed"] = False
            bt.logging.warning(f"   ❌ Anomaly check failed: {anomaly_result['reason']}")
        else:
            bt.logging.info(f"   ✅ Anomaly scoring: {anomaly_result['reason']}")
        
        return results

    async def verify_license_ocr(self, lead: Dict) -> Dict:
        """
        Validate license document via hash verification.
        
        Steps:
        1. Download document from license_doc_url
        2. Verify hash matches license_doc_hash (SHA-256)
        3. Flag for manual OCR review
        
        Future enhancement: Implement OCR text extraction to search for
        key terms (resale, redistribute, transfer, sub-license).
        
        Returns dict with:
        - passed: bool
        - check: str (check name)
        - reason: str (result description)
        - manual_review_required: bool (optional)
        """
        import hashlib
        import aiohttp
        
        license_url = lead.get("license_doc_url")
        license_hash = lead.get("license_doc_hash")
        
        if not license_url:
            return {
                "passed": False,
                "check": "license_ocr",
                "reason": "No license_doc_url provided for OCR verification"
            }
        
        if not license_hash:
            return {
                "passed": False,
                "check": "license_ocr",
                "reason": "No license_doc_hash provided"
            }
        
        try:
            # Download document
            bt.logging.info(f"   📥 Downloading license doc from: {license_url[:50]}...")
            
            async with aiohttp.ClientSession() as session:
                async with session.get(license_url, timeout=30) as response:
                    if response.status != 200:
                        return {
                            "passed": False,
                            "check": "license_ocr",
                            "reason": f"License doc unreachable: HTTP {response.status}"
                        }
                    
                    doc_content = await response.read()
            
            # Verify hash matches
            computed_hash = hashlib.sha256(doc_content).hexdigest()
            
            if computed_hash != license_hash:
                return {
                    "passed": False,
                    "check": "license_ocr",
                    "reason": f"License doc hash mismatch (expected: {license_hash[:8]}..., got: {computed_hash[:8]}...)"
                }
            
            bt.logging.info(f"   ✅ License hash verified: {computed_hash[:16]}...")
            
            # TODO: Implement OCR text extraction (requires pytesseract or cloud OCR API)
            # For now, flag for manual review
            return {
                "passed": True,
                "check": "license_ocr",
                "reason": "Hash verified - flagged for manual OCR review",
                "manual_review_required": True,
                "license_hash": computed_hash,
                "license_url": license_url
            }
            
        except asyncio.TimeoutError:
            return {
                "passed": False,
                "check": "license_ocr",
                "reason": "License doc download timeout (>30s)"
            }
        except Exception as e:
            return {
                "passed": False,
                "check": "license_ocr",
                "reason": f"License verification error: {str(e)}"
            }

    async def verify_cross_domain_authenticity(self, lead: Dict) -> Dict:
        """
        Verify entity-domain relationship authenticity.
        
        Checks:
        - Email domain should match company domain
        - Detects throwaway/temporary domains
        - Validates domain relationships
        
        This helps detect:
        - Spoofed email addresses
        - Temporary/disposable domains
        - Mismatched company-email relationships
        
        Returns dict with:
        - passed: bool
        - check: str (check name)
        - reason: str (result description)
        - severity: str (optional - "high" for critical mismatches)
        """
        from urllib.parse import urlparse
        
        email = get_email(lead)
        website = get_website(lead)
        company = get_company(lead)
        
        # If insufficient data, pass through (can't verify)
        if not email or not website:
            return {
                "passed": True,
                "check": "cross_domain",
                "reason": "Insufficient data for cross-domain verification"
            }
        
        # Extract domains
        email_domain = email.split("@")[1].lower() if "@" in email else ""
        
        # Parse website domain
        try:
            parsed_website = urlparse(website if website.startswith(('http://', 'https://')) else f'https://{website}')
            website_domain = parsed_website.netloc.lower()
            
            # Remove www. prefix for comparison
            if website_domain.startswith("www."):
                website_domain = website_domain[4:]
            if email_domain.startswith("www."):
                email_domain = email_domain[4:]
                
        except Exception as e:
            bt.logging.warning(f"   Failed to parse website domain: {website} - {e}")
            return {
                "passed": True,
                "check": "cross_domain",
                "reason": "Could not parse website domain"
            }
        
        # Check for throwaway/temporary domain indicators
        throwaway_indicators = [
            "-sales", "-marketing", "-temp", "tempmail", "guerrilla",
            "throwaway", "disposable", "fake", "test", "temporary"
        ]
        
        for indicator in throwaway_indicators:
            if indicator in email_domain:
                return {
                    "passed": False,
                    "check": "cross_domain",
                    "reason": f"Email domain appears to be temporary: {email_domain}",
                    "severity": "high"
                }
        
        # Check if domains match
        if email_domain == website_domain:
            return {
                "passed": True,
                "check": "cross_domain",
                "reason": "Email domain matches website domain"
            }
        
        # Check if they're related (subdomain or parent domain)
        if website_domain in email_domain or email_domain in website_domain:
            return {
                "passed": True,
                "check": "cross_domain",
                "reason": f"Related domains (email: {email_domain}, website: {website_domain})"
            }
        
        # Domains don't match - this could be legitimate (e.g., gmail.com for small business)
        # or could be suspicious. We'll flag but not fail for now.
        # In a stricter implementation, this could be a failure.
        return {
            "passed": True,  # Pass but log warning
            "check": "cross_domain",
            "reason": f"Email domain ({email_domain}) differs from website ({website_domain})",
            "severity": "low",
            "warning": True
        }

    async def score_behavioral_anomalies(self, lead: Dict) -> Dict:
        """
        Score lead for behavioral anomalies.
        
        Checks for:
        - Excessive use of same source_url (possible scraping/automation)
        - Unlikely role-industry combinations
        - Statistical outliers
        
        Returns dict with:
        - passed: bool (True if anomaly_score < 0.7)
        - check: str (check name)
        - score: float (0-1, where 0=normal, 1=highly anomalous)
        - flags: list (descriptions of detected anomalies)
        - reason: str (summary)
        """
        anomaly_score = 0.0
        flags = []
        
        # Check 1: Duplicate source_url usage
        source_url = lead.get("source_url", "")
        if source_url:
            try:
                from Leadpoet.utils.cloud_db import get_supabase_client
                supabase = get_supabase_client()
                
                if supabase:
                    # Query recent submissions with same source_url
                    recent_cutoff = (datetime.now(timezone.utc) - timedelta(hours=24)).isoformat()
                    result = supabase.table("prospect_queue")\
                        .select("miner_hotkey, source_url")\
                        .eq("source_url", source_url)\
                        .gte("created_at", recent_cutoff)\
                        .execute()
                    
                    if result.data and len(result.data) > 10:
                        anomaly_score += 0.3
                        flags.append(f"Source URL used {len(result.data)} times in 24h")
                        bt.logging.warning(f"   ⚠️  High source_url reuse: {len(result.data)} times")
            except Exception as e:
                bt.logging.debug(f"   Could not check source_url duplicates: {e}")
        
        # Check 2: Role-industry mismatch
        # This is a simplified check - in production, use ML model or extensive mapping
        role = get_role(lead)
        industry = get_industry(lead)
        
        if role and industry:
            # Define obviously unlikely combinations
            unlikely_combinations = [
                ("Doctor", "Technology"),
                ("Doctor", "Software"),
                ("CTO", "Healthcare"),
                ("CTO", "Medical"),
                ("Nurse", "Finance"),
                ("Engineer", "Healthcare"),
                ("Surgeon", "Retail"),
            ]
            
            # Normalize for comparison
            role_normalized = role.upper()
            industry_normalized = industry.upper()
            
            for unlikely_role, unlikely_industry in unlikely_combinations:
                if unlikely_role.upper() in role_normalized and unlikely_industry.upper() in industry_normalized:
                    anomaly_score += 0.2
                    flags.append(f"Unlikely role-industry: {role} in {industry}")
                    bt.logging.warning(f"   ⚠️  Unlikely combination: {role} in {industry}")
                    break
        
        # Check 3: Missing critical fields (possible data quality issue)
        critical_fields = ["email", "company", "website"]
        missing_fields = [field for field in critical_fields if not lead.get(field)]
        
        if len(missing_fields) >= 2:
            anomaly_score += 0.1
            flags.append(f"Missing {len(missing_fields)} critical fields: {', '.join(missing_fields)}")
        
        # Determine pass/fail based on threshold
        threshold = 0.7
        passed = anomaly_score < threshold
        
        return {
            "passed": passed,
            "check": "anomaly_scoring",
            "score": anomaly_score,
            "flags": flags,
            "reason": f"Anomaly score: {anomaly_score:.2f} (threshold: {threshold})",
            "threshold": threshold
        }

    async def validate_lead(self, lead):
        """Validate a single lead using automated_checks. Returns pass/fail."""
        try:
            # Check for required email field first
            email = get_email(lead)
            if not email:
                return {
                    'is_legitimate': False,
                    'reason': {
                        "stage": "Pre-validation",
                        "check_name": "email_check",
                        "message": "Missing email",
                        "failed_fields": ["email"]
                    },
                    'enhanced_lead': lead  # Return original lead if no email
                }
            
            # Map your field names to what automated_checks expects
            mapped_lead = {
                "email": email,  # Map to "email" field
                "Email 1": email,  # Also map to "Email 1" as backup
                "Company": get_field(lead, 'business', 'website'),  # Map business -> Company
                "Website": get_field(lead, 'website', 'business'),  # Map to Website
                "website": get_field(lead, 'website', 'business'),  # Also lowercase
                "First Name": lead.get('first', ''),
                "Last Name": lead.get('last', ''),
                # Include any other fields that might be useful
                **lead  # Include all original fields too
            }
            
            # Use automated_checks for comprehensive validation
            # NEW: run_automated_checks returns (passed, automated_checks_data) with structured data
            passed, automated_checks_data = await run_automated_checks(mapped_lead)
            
            # Extract rejection_reason from structured data for backwards compatibility
            reason = automated_checks_data.get("rejection_reason") if not passed else None
            
            # Append automated_checks data to mapped_lead so it gets stored in validation_tracking
            mapped_lead["automated_checks"] = automated_checks_data

            # If standard validation passed, check if deep verification is needed
            if passed and self.should_run_deep_verification(mapped_lead):
                bt.logging.info(f"🔬 Running deep verification on {email}")
                
                deep_results = await self.run_deep_verification(mapped_lead)
                
                if not deep_results["passed"]:
                    bt.logging.warning(f"❌ Deep verification failed: {deep_results}")
                    # Mark lead for manual review or reject
                    lead["deep_verification_failed"] = True
                    lead["deep_verification_results"] = deep_results
                
                    # Return structured rejection reason 
                    deep_reason = deep_results["checks"][0]["reason"] if deep_results.get("checks") else "unknown"
                    return {
                        'is_legitimate': False,
                        'reason': {
                            "stage": "Deep Verification",
                            "check_name": "deep_verification",
                            "message": f"Deep verification failed: {deep_reason}",
                            "failed_fields": []
                        },
                        'deep_verification_results': deep_results,
                        'enhanced_lead': mapped_lead  # Include enhanced lead even on deep verification failure
                    }
                else:
                    bt.logging.info(f"✅ Deep verification passed")
                    lead["deep_verification_passed"] = True
                    lead["deep_verification_results"] = deep_results
                    
                    # If manual review required, flag it but don't fail
                    if deep_results.get("manual_review_required"):
                        lead["manual_review_required"] = True
                        bt.logging.info(f"📋 Lead flagged for manual review")

            # Copy validator-calculated rep_score from mapped_lead back to original lead
            # This ensures the rep_score in enhanced_lead is from automated checks, not miner data
            if "rep_score" in mapped_lead:
                lead["rep_score"] = mapped_lead["rep_score"]
            
            # Prepare validation result with enhanced lead data
            validation_result = {
                'is_legitimate': passed,
                'reason': reason,
                'enhanced_lead': mapped_lead  # Include enhanced lead with DNSBL/WHOIS data
            }
            
            # NOTE: Audit logging removed - validators should NOT write directly to Supabase.
            # All logging is handled by the gateway via POST /validate (TEE architecture).
            # The gateway stores evidence_blob in validation_evidence_private and logs to TEE buffer.
            
            return validation_result
            
        except Exception as e:
            # Check if this is an EmailVerificationUnavailableError - if so, re-raise it
            from validator_models.automated_checks import EmailVerificationUnavailableError
            if isinstance(e, EmailVerificationUnavailableError):
                # Re-raise to propagate to process_sourced_leads_continuous
                raise
            
            bt.logging.error(f"Error in validate_lead: {e}")
            
            # Create structured rejection reason for error case
            error_rejection = {
                "stage": "Validation Error",
                "check_name": "exception",
                "message": f"Validation error: {str(e)}",
                "failed_fields": []
            }
            
            # NOTE: Audit logging removed - validators should NOT write directly to Supabase.
            # All logging is handled by the gateway via POST /validate (TEE architecture).
            
            return {
                'is_legitimate': False,
                'reason': error_rejection,
                'enhanced_lead': lead  # Return original lead on error
            }

    def calculate_validation_score_breakdown(self, lead):
        """Calculate validation score with detailed breakdown"""
        try:
            website_score = 0.2 if lead.get('website') else 0.0
            industry_score = 0.1 if lead.get('industry') else 0.0
            region_score = 0.1 if lead.get('region') else 0.0

            return {
                'website_score': website_score,
                'industry_score': industry_score,
                'region_score': region_score
            }
        except Exception:
            return {'website_score': 0.0, 'industry_score': 0.0, 'region_score': 0.0}

    def save_validated_lead_to_supabase(self, lead: Dict) -> bool:
        """
        Write validated lead directly to Supabase.
        Validators have INSERT-only access (enforced by RLS).
        Duplicates are handled by database unique constraint + trigger notification.
        
        Args:
            lead: Lead dictionary with all required fields
            
        Returns:
            bool: True if successfully inserted, False if duplicate or error
        """
        if not self.supabase_client:
            bt.logging.error("❌ Supabase client not initialized, cannot save lead")
            return False
        
        try:
            # Prepare lead data for insertion
            lead_data = {
                "email": get_email(lead),
                "company": get_field(lead, "business", "company"),
                "validated_at": datetime.now(timezone.utc).isoformat(),
                "validator_hotkey": self.wallet.hotkey.ss58_address,
                "miner_hotkey": get_field(lead, "source", "miner_hotkey"),
                "score": get_field(lead, "conversion_score", "score"),
                "metadata": {
                    "full_name": lead.get("full_name", ""),
                    "first": lead.get("first", ""),
                    "last": lead.get("last", ""),
                    "linkedin": lead.get("linkedin", ""),
                    "website": lead.get("website", ""),
                    "industry": lead.get("industry", ""),
                    "sub_industry": lead.get("sub_industry", ""),
                    "region": lead.get("region", ""),
                    "region_country": lead.get("region_country", ""),
                    "region_state": lead.get("region_state", ""),
                    "region_city": lead.get("region_city", ""),
                    "role": lead.get("role", ""),
                    "description": lead.get("description", ""),
                    "phone_numbers": lead.get("phone_numbers", []),
                    "founded_year": lead.get("founded_year", ""),
                    "ownership_type": lead.get("ownership_type", ""),
                    "company_type": lead.get("company_type", ""),
                    "number_of_locations": lead.get("number_of_locations", ""),
                    "socials": lead.get("socials", {}),
                }
            }
            
            # DEBUG: Log what we're trying to insert
            bt.logging.debug(f"🔍 INSERT attempt - validator_hotkey: {lead_data['validator_hotkey'][:10]}...")
            
            # Insert into Supabase - database will enforce unique constraint
            # Trigger will automatically notify miner if duplicate
            # NOTE: Wrap in array to match how miner inserts to prospect_queue
            self.supabase_client.table("leads").insert([lead_data])
            
            bt.logging.info(f"✅ Saved lead to Supabase: {lead_data['email']} ({lead_data['company']})")
            return True
            
        except Exception as e:
            error_str = str(e).lower()
            
            # Handle duplicate email (caught by unique constraint)
            if "duplicate" in error_str or "unique" in error_str or "23505" in error_str:
                bt.logging.debug(f"⏭️  Duplicate lead (trigger will notify miner): {get_email(lead)}")
                return False
            
            # Handle RLS policy violations
            elif "row-level security" in error_str or "42501" in error_str:
                bt.logging.error("❌ RLS policy violation - check JWT and validator_hotkey match")
                bt.logging.error(f"   Validator hotkey in data: {lead_data.get('validator_hotkey', 'missing')[:10]}...")
                bt.logging.error("   JWT should contain same hotkey in 'hotkey' claim")
                return False
            
            # Other errors
            else:
                bt.logging.error(f"❌ Failed to save lead to Supabase: {e}")
                return False

DATA_DIR = "data"
VALIDATION_LOG = os.path.join(DATA_DIR, "validation_logs.json")
VALIDATORS_LOG = os.path.join(DATA_DIR, "validators.json")

def ensure_data_files():
    os.makedirs(DATA_DIR, exist_ok=True)
    for file in [VALIDATION_LOG, VALIDATORS_LOG]:
        if not os.path.exists(file):
            with open(file, "w") as f:
                json.dump([], f)

def log_validation(hotkey, num_valid, num_rejected, issues):
    entry = {
        "timestamp": datetime.now().isoformat(),
        "hotkey": hotkey,
        "num_valid": num_valid,
        "num_rejected": num_rejected,
        "issues": issues
    }
    with open(VALIDATION_LOG, "r+") as f:
        try:
            logs = json.load(f)
        except Exception:
            logs = []
        logs.append(entry)
        f.seek(0)
        json.dump(logs, f, indent=2)

def update_validator_stats(hotkey, precision):
    with open(VALIDATORS_LOG, "r+") as f:
        try:
            validators = json.load(f)
        except Exception:
            validators = []
        found = False
        for v in validators:
            if v["hotkey"] == hotkey:
                v["precision"] = precision
                v["last_updated"] = datetime.now().isoformat()
                found = True
                break
        if not found:
            validators.append({
                "hotkey": hotkey,
                "precision": precision,
                "last_updated": datetime.now().isoformat()
            })
        f.seek(0)
        json.dump(validators, f, indent=2)

class LeadQueue:
    def __init__(self, maxsize: int = 1000):
        self.maxsize = maxsize
        self.queue_file = "lead_queue.json"
        self._ensure_queue_file()

    def _ensure_queue_file(self):
        """Ensure queue file exists and is valid JSON"""
        try:
            # Try to read existing file
            with open(self.queue_file, 'r') as f:
                try:
                    json.load(f)
                except json.JSONDecodeError:
                    # If file is corrupted, create new empty queue
                    bt.logging.warning("Queue file corrupted, creating new empty queue")
                    self._create_empty_queue()
        except FileNotFoundError:
            # If file doesn't exist, create new empty queue
            self._create_empty_queue()

    def _create_empty_queue(self):
        """Create a new empty queue file"""
        with open(self.queue_file, 'w') as f:
            json.dump([], f)

    def enqueue_prospects(self, prospects: List[Dict], miner_hotkey: str,
                          request_type: str = "sourced", **meta):
        """Add prospects to queue with validation"""
        try:
            with open(self.queue_file, 'r') as f:
                try:
                    queue = json.load(f)
                except json.JSONDecodeError:
                    bt.logging.warning("Queue file corrupted during read, creating new queue")
                    queue = []

            # append once
            queue.append({
                "prospects": prospects,
                "miner_hotkey": miner_hotkey,
                "request_type": request_type,
                **meta
            })

            # trim & write back
            if len(queue) > self.maxsize:
                queue = queue[-self.maxsize:]

            with open(self.queue_file, 'w') as f:
                json.dump(queue, f, indent=2)

        except Exception as e:
            bt.logging.error(f"Error enqueueing prospects: {e}")
            self._create_empty_queue()

    def dequeue_prospects(self) -> List[Dict]:
        """Get and remove prospects from queue with validation"""
        try:
            # Read current queue
            with open(self.queue_file, 'r') as f:
                try:
                    queue = json.load(f)
                except json.JSONDecodeError:
                    bt.logging.warning("Queue file corrupted during read, creating new queue")
                    queue = []

            if not queue:
                return []

            # Get all prospects and clear queue
            prospects = queue
            with open(self.queue_file, 'w') as f:
                json.dump([], f)

            return prospects

        except Exception as e:
            bt.logging.error(f"Error dequeuing prospects: {e}")
            # If any error occurs, try to create new queue
            self._create_empty_queue()
            return []

async def run_validator(validator_hotkey, queue_maxsize):
    print("Validator event loop started.")

    # Create validator instance
    config = bt.config()
    validator = Validator(config=config)

    # Start HTTP server
    await validator.start_http_server()

    # Track all delivered leads for this API query
    all_delivered_leads = []

    async def validation_loop():
        nonlocal all_delivered_leads
        print("🔄 Validation loop running - waiting for leads to process...")
        while True:
            lead_request = lead_queue.dequeue_prospects()
            if not lead_request:
                await asyncio.sleep(1)
                continue

            request_type = lead_request.get("request_type", "sourced")
            prospects     = lead_request["prospects"]
            miner_hotkey  = lead_request["miner_hotkey"]

            print(f"\n📥 Processing {request_type} batch of {len(prospects)} prospects from miner {miner_hotkey[:8]}...")

            # curated list
            if request_type == "curated":
                print(f"🔍 Processing curated leads from {miner_hotkey[:20]}...")
                # Set the curator hotkey for all prospects in this batch
                for prospect in prospects:
                    prospect["curated_by"] = miner_hotkey

                # score with your open-source conversion model
                report  = await auto_check_leads(prospects)
                scores  = report.get("detailed_scores", [1.0]*len(prospects))
                for p, s in zip(prospects, scores):
                    p["conversion_score"] = s

                # print human-readable ranking
                ranked = sorted(prospects, key=lambda x: x["conversion_score"], reverse=True)
                print(f"\n Curated leads from {miner_hotkey[:20]} (ranked by score):")
                for idx, lead in enumerate(ranked, 1):
                    business = get_company(lead, default='Unknown')[:30]
                    # accept either lowercase or capitalised field
                    business = get_company(lead, default='Unknown')
                    business = business[:30]
                    score = lead['conversion_score']
                    print(f"  {idx:2d}. {business:30s}  score={score:.3f}")

                asked_for = lead_request.get("requested", len(ranked))
                top_n = min(asked_for, len(ranked))
                print(f"✅ Sending top-{top_n} leads to buyer")

                # store in pool and record reward-event for delivered leads
                delivered_leads = ranked[:top_n]
                add_validated_leads_to_pool(delivered_leads)

                # Add to all delivered leads for this query
                all_delivered_leads.extend(delivered_leads)

                # Record rewards for ALL delivered leads in this query
                from Leadpoet.base.utils.pool import record_delivery_rewards
                record_delivery_rewards(all_delivered_leads)

                # Send leads to buyer
                print(f"✅ Sent {len(delivered_leads)} leads to buyer")

                # Add source hotkey display
                for lead in delivered_leads:
                    source_hotkey = lead.get('source', 'unknown')
                    print(f"   Lead sourced by: {source_hotkey}")   # show full hotkey

                # Save curated leads to separate file
                from Leadpoet.base.utils.pool import save_curated_leads
                save_curated_leads(delivered_leads)

                # Reset all_delivered_leads after recording rewards
                all_delivered_leads = []

                continue          # skip legitimacy audit branch altogether

            # sourced list
            print(f"🔍 Validating {len(prospects)} sourced leads...")
            valid, rejected, issues = [], [], []

            for prospect in prospects:
                business = prospect.get('business', 'Unknown Business')
                print(f"\n  Validating: {business}")

                # Get email
                email = prospect.get("email", "")
                print(f"    Email: {email}")

                if not re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', email):
                    issue = f"Invalid email: {email}"
                    print(f"    ❌ Rejected: {issue}")
                    issues.append(issue)
                    rejected.append(prospect)
                    continue

                if any(domain in email for domain in ["mailinator.com", "tempmail.com"]):
                    issue = f"Disposable email: {email}"
                    print(f"    ❌ Rejected: {issue}")
                    issues.append(issue)
                    rejected.append(prospect)
                    continue

                if prospect["source"] != miner_hotkey:
                    issue = f"Source mismatch: {prospect['source']} != {miner_hotkey}"
                    print(f"    ❌ Rejected: {issue}")
                    issues.append(issue)
                    rejected.append(prospect)
                    continue

                if lead_pool.check_duplicates(email):
                    issue = f"Duplicate email: {email}"
                    print(f"    ❌ Rejected: {issue}")
                    issues.append(issue)
                    rejected.append(prospect)
                    continue

                # All checks passed ⇒ accept
                valid.append(prospect)

            if valid:
                add_validated_leads_to_pool(valid)
                print(f"\n✅ Added {len(valid)} valid prospects to pool")

            log_validation(validator_hotkey, len(valid), len(rejected), issues)
            total = len(valid) + len(rejected)
            precision = (len(valid) / total) if total else 0.0
            update_validator_stats(validator_hotkey, precision)
            print(f"\n Validation summary: {len(valid)} accepted, {len(rejected)} rejected.")
            await asyncio.sleep(0.1)

    # Run both the HTTP server and validation loop
    await asyncio.gather(
        validation_loop(),
        asyncio.sleep(float('inf'))  # Keep HTTP server running
    )

def add_validated_leads_to_pool(leads):
    """Add validated leads to the pool with consistent field names."""
    mapped_leads = []
    for lead in leads:
        # Get the actual validation score from the lead
        validation_score = lead.get("conversion_score", 1.0)  # Use existing score or default to 1.0

        mapped_lead = {
            "business": get_company(lead),
            "full_name": get_field(lead, "full_name"),
            "first": get_first_name(lead),
            "last": get_last_name(lead),
            "email": get_email(lead),
            "linkedin": get_linkedin(lead),
            "website": get_website(lead),
            "industry": get_industry(lead),
            "sub_industry": get_sub_industry(lead),
            "region": get_location(lead),
            "role": lead.get("role", ""),
            "description": lead.get("description", ""),
            "phone_numbers": lead.get("phone_numbers", []),
            "founded_year": lead.get("founded_year", ""),
            "ownership_type": lead.get("ownership_type", ""),
            "company_type": lead.get("company_type", ""),
            "number_of_locations": lead.get("number_of_locations", ""),
            "socials": lead.get("socials", {}),
            "source":     lead.get("source", ""),
            "curated_by": lead.get("curated_by", ""),
        }

        # score is kept only if the lead already has it (i.e. curated phase)
        if "conversion_score" in lead:
            mapped_lead["conversion_score"] = validation_score
        mapped_leads.append(mapped_lead)

    lead_pool.add_to_pool(mapped_leads)


def run_lightweight_worker(config):
    """
    Lightweight worker loop for containerized validators.
    
    Workers skip ALL heavy initialization and only:
    1. Read current_block.json for epoch timing
    2. Read epoch_{N}_leads.json for lead data
    3. Validate leads (CPU/IO work)
    4. Write results to JSON file
    
    No Bittensor connection, no axon, no epoch monitor, no weight setting.
    """
    import asyncio
    import json
    from pathlib import Path
    
    print("🚀 Starting lightweight worker...")
    print(f"   Container ID: {config.neuron.container_id}")
    print(f"   Total containers: {config.neuron.total_containers}")
    print("")
    
    # Create minimal validator-like object for process_gateway_validation_workflow
    class LightweightWorker:
        def __init__(self, config):
            self.config = config
            self.should_exit = False
            
        def _read_shared_block_file(self):
            """Read current block from shared file (written by coordinator)"""
            block_file = Path("validator_weights") / "current_block.json"
            
            if not block_file.exists():
                raise FileNotFoundError("Coordinator hasn't written block file yet")
            
            # Check if file is stale (> 60 seconds old)
            import time
            file_age = time.time() - block_file.stat().st_mtime
            if file_age > 60:
                raise Exception(f"Shared block file is stale ({int(file_age)}s old)")
            
            with open(block_file, 'r') as f:
                data = json.load(f)
                return data['block'], data['epoch'], data['blocks_into_epoch']
        
        async def process_gateway_validation_workflow(self):
            """
            Simplified worker validation loop.
            
            This is a COPY of the worker-specific logic from Validator.process_gateway_validation_workflow(),
            but without any Bittensor dependencies.
            """
            import time
            from validator_models.automated_checks import run_automated_checks, run_batch_automated_checks
            
            print("🔄 Worker validation loop started")
            
            while not self.should_exit:
                try:
                    # Read current epoch from coordinator's shared file
                    try:
                        current_block, current_epoch, blocks_into_epoch = self._read_shared_block_file()
                    except FileNotFoundError:
                        print("⏳ Worker: Waiting for coordinator to write block file...")
                        await asyncio.sleep(5)
                        continue
                    except Exception as e:
                        # Extract just the error message, don't try to parse it
                        print(f"⏳ Worker: Waiting for coordinator to write block file... ({str(e)})")
                        await asyncio.sleep(5)
                        continue
                    
                    print(f"\n🔍 WORKER EPOCH {current_epoch}: Starting validation (block {blocks_into_epoch}/360)")
                    
                    # CRITICAL FIX: Check if we already completed this epoch
                    # Prevents re-processing the same epoch while waiting for coordinator to aggregate
                    container_id = self.config.neuron.container_id
                    results_file = Path("validator_weights") / f"worker_{container_id}_epoch_{current_epoch}_results.json"
                    if results_file.exists():
                        print(f"⏭️  Worker {container_id}: Already completed epoch {current_epoch}, waiting for next epoch...")
                        await asyncio.sleep(30)
                        continue
                    
                    # Wait for coordinator to fetch and share leads
                    leads_file = Path("validator_weights") / f"epoch_{current_epoch}_leads.json"
                    
                    waited = 0
                    log_interval = 300  # Log every 5 minutes
                    check_interval = 5  # Check every 5 seconds
                    
                    while not leads_file.exists():
                        await asyncio.sleep(check_interval)
                        waited += check_interval
                        
                        # Check current block and epoch from shared file
                        try:
                            check_block, check_epoch, blocks_into_epoch = self._read_shared_block_file()
                        except Exception:
                            continue
                        
                        # Epoch changed while waiting - abort
                        if check_epoch > current_epoch:
                            print(f"❌ Worker: Epoch changed ({current_epoch} → {check_epoch}) while waiting")
                            await asyncio.sleep(10)
                            break
                        
                        # Too late to start validation
                        if blocks_into_epoch >= 275:
                            print(f"❌ Worker: Too late to start validation (block {blocks_into_epoch}/360)")
                            await asyncio.sleep(10)
                            break
                        
                        # Log progress
                        if waited % log_interval == 0 and waited > 0:
                            print(f"⏳ Worker: Still waiting for coordinator ({waited}s elapsed)...")
                    
                    if not leads_file.exists():
                        continue  # Epoch changed or too late
                    
                    # Read leads from file (including centralized TrueList results)
                    with open(leads_file, 'r') as f:
                        data = json.load(f)
                        all_leads = data.get('leads', [])
                        epoch_id = data.get('epoch_id')
                        salt_hex = data.get('salt')  # CRITICAL: Read shared salt
                        centralized_truelist = data.get('truelist_results')  # None = in progress, {} = failed, {...} = success
                    
                    if epoch_id != current_epoch:
                        print(f"⚠️  Worker: Leads file epoch mismatch ({epoch_id} != {current_epoch})")
                        await asyncio.sleep(10)
                        continue
                    
                    if not salt_hex:
                        print(f"❌ Worker: No salt in leads file! Cannot hash results.")
                        await asyncio.sleep(10)
                        continue
                    
                    # Log TrueList status from file
                    # None = in progress (coordinator still running), {} = failed, {...} = success
                    if centralized_truelist is None:
                        print(f"   ⏳ Worker: TrueList in progress - will poll after Stage 0-2 completes")
                    elif centralized_truelist:
                        print(f"   ✅ Worker: TrueList already complete ({len(centralized_truelist)} results)")
                    else:
                        print(f"   ⚠️ Worker: TrueList failed (empty results) - leads will fail email verification")
                    
                    # Check if leads were actually fetched by coordinator
                    if all_leads is None or len(all_leads) == 0:
                        print(f"ℹ️  Worker: No leads in file for epoch {current_epoch} (coordinator returned null/empty)")
                        print(f"   This happens when: already submitted, gateway queue empty, or epoch just started")
                        print(f"   Waiting for next epoch...")
                        await asyncio.sleep(30)
                        continue
                    
                    # Calculate worker's lead subset (moved before salt print to avoid UnboundLocalError)
                    container_id = self.config.neuron.container_id
                    total_containers = self.config.neuron.total_containers
                    
                    # Convert salt from hex
                    salt = bytes.fromhex(salt_hex)
                    print(f"   Worker {container_id}: Using shared salt {salt_hex[:16]}...")
                    
                    # CRITICAL: Use SAME range slicing as coordinator (lines 1975-1991)
                    # NOT modulo - modulo causes overlap with coordinator's range!
                    original_count = len(all_leads)
                    leads_per_container = original_count // total_containers
                    remainder = original_count % total_containers
                    
                    # First 'remainder' containers get 1 extra lead to distribute remainder evenly
                    if container_id < remainder:
                        start = container_id * (leads_per_container + 1)
                        end = start + leads_per_container + 1
                    else:
                        start = (remainder * (leads_per_container + 1)) + ((container_id - remainder) * leads_per_container)
                        end = start + leads_per_container
                    
                    worker_leads = all_leads[start:end]
                    
                    print(f"   Worker {container_id}: Processing leads {start}-{end} ({len(worker_leads)}/{original_count} leads)")
                    
                    # ================================================================
                    # BATCH VALIDATION: Stage 0-2 runs in parallel with coordinator's TrueList
                    # After Stage 0-2, poll file for TrueList results before Stage 4-5
                    # ================================================================
                    
                    # Extract lead_blobs for batch processing
                    lead_blobs = [lead_data.get('lead_blob', {}) for lead_data in worker_leads]
                    
                    # Log TrueList status (might be ready or in progress)
                    if centralized_truelist:
                        print(f"   ✅ Worker {container_id}: TrueList already complete ({len(centralized_truelist)} results)")
                    elif centralized_truelist is None:
                        print(f"   ⏳ Worker {container_id}: TrueList in progress - will poll after Stage 0-2")
                    else:
                        print(f"   ⚠️ Worker {container_id}: TrueList returned empty (coordinator may have failed)")
                    
                    # Run batch validation - polls file for TrueList results after Stage 0-2
                    leads_file_str = str(leads_file)
                    try:
                        batch_results = await run_batch_automated_checks(
                            lead_blobs, 
                            container_id=container_id,
                            leads_file_path=leads_file_str  # Poll file for TrueList results after Stage 0-2
                        )
                    except Exception as e:
                        print(f"   ❌ Batch validation failed: {e}")
                        import traceback
                        traceback.print_exc()
                        # Fallback: Mark all leads as validation errors
                        batch_results = [
                            (False, {
                                "passed": False,
                                "rejection_reason": {
                                    "stage": "Batch Validation",
                                    "check_name": "run_batch_automated_checks",
                                    "message": f"Batch validation error: {str(e)}"
                                }
                            })
                            for _ in lead_blobs
                        ]
                    
                    # Map results back to validated_leads format (SAME ORDER guaranteed)
                    validated_leads = []
                    for i, (passed, automated_checks_data) in enumerate(batch_results):
                        lead_data = worker_leads[i]
                        lead_id = lead_data.get('lead_id', 'unknown')
                        lead_blob = lead_data.get('lead_blob', {})
                        miner_hotkey = lead_data.get('miner_hotkey', lead_blob.get('wallet_ss58', 'unknown'))
                        
                        # Handle skipped leads (passed=None means email verification unavailable)
                        if passed is None:
                            validated_leads.append({
                                'lead_id': lead_id,
                                'is_valid': False,  # Treat skipped as invalid for this epoch
                                'rejection_reason': {'message': 'EmailVerificationUnavailable'},
                                'automated_checks_data': automated_checks_data,
                                'lead_blob': lead_blob,
                                'miner_hotkey': miner_hotkey,
                                'skipped': True
                            })
                        else:
                            # Normal pass/fail
                            rejection_reason = automated_checks_data.get("rejection_reason") if not passed else None
                            validated_leads.append({
                                'lead_id': lead_id,
                                'is_valid': passed,
                                'rejection_reason': rejection_reason,
                                'automated_checks_data': automated_checks_data,
                                'lead_blob': lead_blob,
                                'miner_hotkey': miner_hotkey
                            })
                    
                    # Write results to file for coordinator
                    # CRITICAL: Hash results using shared salt (EXACT same format as coordinator)
                    results_file = Path("validator_weights") / f"worker_{container_id}_epoch_{current_epoch}_results.json"
                    
                    import hashlib
                    validation_results = []
                    local_validation_data = []
                    
                    for lead in validated_leads:
                        # Extract data
                        is_valid = lead['is_valid']
                        decision = "approve" if is_valid else "deny"
                        # CRITICAL: Use validator-calculated rep_score, NOT miner's submitted value
                        # Denied leads get 0, approved leads get score from automated checks
                        automated_checks_data = lead.get('automated_checks_data', {})
                        rep_score = int(automated_checks_data.get('rep_score', {}).get('total_score', 0)) if is_valid else 0
                        rejection_reason = lead.get('rejection_reason') or {} if not is_valid else {"message": "pass"}
                        evidence_blob = json.dumps(lead.get('automated_checks_data', {}), default=str)  # Handle datetime objects
                        
                        # Compute hashes (SHA256 with salt) - EXACT same as coordinator lines 2036-2040
                        decision_hash = hashlib.sha256((decision + salt.hex()).encode()).hexdigest()
                        rep_score_hash = hashlib.sha256((str(rep_score) + salt.hex()).encode()).hexdigest()
                        rejection_reason_hash = hashlib.sha256((json.dumps(rejection_reason, default=str) + salt.hex()).encode()).hexdigest()  # Handle datetime
                        evidence_hash = hashlib.sha256(evidence_blob.encode()).hexdigest()
                        
                        # Format for validation_results (for gateway hash submission) - EXACT format
                        validation_results.append({
                            'lead_id': lead['lead_id'],
                            'decision_hash': decision_hash,
                            'rep_score_hash': rep_score_hash,
                            'rejection_reason_hash': rejection_reason_hash,
                            'evidence_hash': evidence_hash,
                            'evidence_blob': lead.get('automated_checks_data', {})
                        })
                        
                        # Format for local_validation_data (for gateway reveal submission) - EXACT format
                        local_validation_data.append({
                            'lead_id': lead['lead_id'],
                            'miner_hotkey': lead.get('miner_hotkey'),
                            'decision': decision,
                            'rep_score': rep_score,
                            'rejection_reason': rejection_reason,
                            'salt': salt.hex()
                        })
                    
                    with open(results_file, 'w') as f:
                        json.dump({
                            'epoch_id': current_epoch,
                            'container_id': container_id,
                            'validation_results': validation_results,
                            'local_validation_data': local_validation_data,
                            'lead_range': f"{len(validated_leads)} leads",
                            'timestamp': time.time()
                        }, f)
                    
                    print(f"✅ Worker {container_id}: Completed {len(validated_leads)} validations")
                    print(f"   Results saved to {results_file}")
                    
                    # Wait before checking for next epoch
                    await asyncio.sleep(30)
                    
                except Exception as e:
                    print(f"❌ Worker error: {e}")
                    import traceback
                    traceback.print_exc()
                    await asyncio.sleep(30)
    
    # Create worker and run
    worker = LightweightWorker(config)
    
    # Run async loop
    try:
        asyncio.run(worker.process_gateway_validation_workflow())
    except KeyboardInterrupt:
        print("\n🛑 Worker shutting down...")
        worker.should_exit = True


def main():
    parser = argparse.ArgumentParser(description="LeadPoet Validator")
    add_validator_args(None, parser)
    parser.add_argument("--wallet_name", type=str, help="Wallet name")
    parser.add_argument("--wallet_hotkey", type=str, help="Wallet hotkey")
    parser.add_argument("--wallet_path", type=str, default="~/.bittensor/wallets", help="Path to wallets directory (default: ~/.bittensor/wallets)")
    parser.add_argument("--netuid", type=int, default=71, help="Network UID")
    parser.add_argument("--subtensor_network", type=str, default=os.getenv("SUBTENSOR_NETWORK", "finney"), help="Subtensor network (default: finney, or from SUBTENSOR_NETWORK env var)")
    parser.add_argument("--logging_trace", action="store_true", help="Enable trace logging")
    parser.add_argument("--container-id", type=int, help="Container ID (0, 1, 2, etc.) for dynamic lead distribution. Container 0 is coordinator.")
    parser.add_argument("--total-containers", type=int, help="Total number of containers running (for dynamic lead distribution)")
    parser.add_argument("--mode", type=str, choices=["coordinator", "worker"], help="Container mode: 'coordinator' waits for workers and submits to gateway, 'worker' validates and writes results to JSON")
    args = parser.parse_args()

    if args.logging_trace:
        bt.logging.set_trace(True)

    ensure_data_files()

    # ════════════════════════════════════════════════════════════════════════════
    # WORKER MODE: Skip ALL heavy initialization
    # ════════════════════════════════════════════════════════════════════════════
    # Workers don't need:
    # - Bittensor wallet/subtensor/metagraph (no chain connection)
    # - Axon serving (no API endpoints)
    # - Epoch monitor thread (coordinator writes current_block.json)
    # - Dendrite (no outgoing Bittensor requests)
    # - Weight setting (only coordinator submits weights)
    # 
    # Workers ONLY need:
    # - Read current_block.json (for epoch timing)
    # - Read epoch_{N}_leads.json (for lead data)
    # - Validate leads (CPU/IO work)
    # - Write results to JSON file
    # ════════════════════════════════════════════════════════════════════════════
    if getattr(args, 'mode', None) == "worker":
        print("════════════════════════════════════════════════════════════════")
        print("🔧 LIGHTWEIGHT WORKER MODE")
        print("════════════════════════════════════════════════════════════════")
        print("   Skipping heavy initialization:")
        print("   ✗ Bittensor wallet/subtensor/metagraph")
        print("   ✗ Axon serving")
        print("   ✗ Epoch monitor thread")
        print("   ✗ Weight setting")
        print("")
        print("   Worker responsibilities:")
        print("   ✓ Read current_block.json for epoch timing")
        print("   ✓ Read epoch_{N}_leads.json for lead data")
        print("   ✓ Validate leads (CPU/IO work)")
        print("   ✓ Write results to JSON file")
        print("════════════════════════════════════════════════════════════════")
        print("")
        
        # Create minimal config for worker
        config = bt.Config()
        config.neuron = bt.Config()
        config.neuron.container_id = getattr(args, 'container_id', None)
        config.neuron.total_containers = getattr(args, 'total_containers', None)
        config.neuron.mode = "worker"
        
        # Run lightweight worker loop
        run_lightweight_worker(config)
        return  # Exit early - don't initialize full validator

    # ════════════════════════════════════════════════════════════════════════════
    # COORDINATOR MODE: Full initialization
    # ════════════════════════════════════════════════════════════════════════════
    # Add this near the beginning of your validator startup, after imports
    from Leadpoet.validator.reward import start_epoch_monitor

    # Run the proper Bittensor validator
    config = bt.Config()
    config.wallet = bt.Config()
    config.wallet.name = args.wallet_name
    config.wallet.hotkey = args.wallet_hotkey
    # Only set custom wallet path if default doesn't exist
    # Use wallet_path from args, or default to ~/.bittensor/wallets
    if args.wallet_path:
        config.wallet.path = str(Path(args.wallet_path).expanduser())
    else:
        config.wallet.path = str(Path.home() / ".bittensor" / "wallets")
    config.netuid = args.netuid
    config.subtensor = bt.Config()
    config.subtensor.network = args.subtensor_network
    config.neuron = bt.Config()
    config.neuron.disable_set_weights = getattr(args, 'neuron_disable_set_weights', False)
    config.neuron.container_id = getattr(args, 'container_id', None)  # Container ID (0, 1, 2, ...)
    config.neuron.total_containers = getattr(args, 'total_containers', None)  # Total containers
    config.neuron.mode = getattr(args, 'mode', None)  # Container mode: coordinator/worker

    # Start the background epoch monitor AFTER config is set (so network is correct)
    start_epoch_monitor(network=args.subtensor_network)

    validator = Validator(config=config)

    print("🚀 Starting LeadPoet Validator on Bittensor Network...")
    print(f"   Wallet: {validator.wallet.hotkey.ss58_address}")
    print(f"   NetUID: {config.netuid}")
    print("   Validator will process sourced leads and respond to API requests via Bittensor network")

    # Run the validator on the Bittensor network
    validator.run()

    # Add cleanup on shutdown (if you have a shutdown handler)
    # stop_epoch_monitor()

if __name__ == "__main__":
    import signal
    import atexit
    
    def cleanup_handler(signum=None, frame=None):
        """Clean up resources on shutdown"""
        try:
            print("\n🛑 Shutting down validator...")
            from Leadpoet.validator.reward import stop_epoch_monitor
            stop_epoch_monitor()
            
            # Give threads time to clean up
            import time
            time.sleep(1)
            
            print("✅ Cleanup complete")
        except Exception as e:
            print(f"⚠️  Cleanup error: {e}")
        finally:
            if signum is not None:
                sys.exit(0)
    
    # Register cleanup handlers
    signal.signal(signal.SIGTERM, cleanup_handler)
    signal.signal(signal.SIGINT, cleanup_handler)
    atexit.register(cleanup_handler)
    
    try:
        main()
    except KeyboardInterrupt:
        cleanup_handler()
    except Exception as e:
        print(f"❌ Validator crashed: {e}")
        cleanup_handler()
        raise
