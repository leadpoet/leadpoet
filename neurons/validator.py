#!/usr/bin/env python3
# Suppress multiprocessing warnings BEFORE any imports
# Auto-update trigger: 2025-12-12
import os
import sys
from pathlib import Path

# CRITICAL: Add project root to sys.path BEFORE any local imports
# When running 'python3 neurons/validator.py', sys.path[0] = neurons/
# But qualification/, gateway/, etc. are in the project root
# This ensures all local modules can be imported regardless of how the script is run
sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))

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
import gc  # For explicit memory cleanup
from datetime import datetime, timedelta, timezone
from Leadpoet.base.validator import BaseValidatorNeuron
from Leadpoet.protocol import LeadRequest
from validator_models.automated_checks import validate_lead_list as auto_check_leads, run_automated_checks, MAX_REP_SCORE
from Leadpoet.base.utils.config import add_validator_args
import threading
from Leadpoet.base.utils import queue as lead_queue
from Leadpoet.base.utils import pool as lead_pool
from Leadpoet.base.utils.pool import (
    initialize_pool,
    add_to_pool,
    record_delivery_rewards,
    save_curated_leads,
)
# Import modules that have inject_async_subtensor methods
from Leadpoet.validator import reward as reward_module
from Leadpoet.utils import cloud_db as cloud_db_module
from Leadpoet.validator.reward import start_epoch_monitor, stop_epoch_monitor
import asyncio
from typing import List, Dict, Optional, Any
from aiohttp import web
from Leadpoet.utils.cloud_db import (
    fetch_prospects_from_cloud,
    fetch_curation_requests,
    push_curation_result,
    push_miner_curation_request,
    fetch_miner_curation_result,
    push_validator_ranking,
    fetch_broadcast_requests,  # Must be at module level to avoid sandbox blocking
    # Additional imports moved from lazy to module-level to avoid sandbox blocking:
    get_supabase_client,
    broadcast_api_request,
    fetch_validator_rankings,
    get_broadcast_status,
    gateway_get_epoch_leads,
    gateway_submit_validation,
    # NOTE: gateway_submit_reveal REMOVED (Jan 2026) - IMMEDIATE REVEAL MODE
    submit_validation_assessment,
    fetch_miner_leads_for_request,
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

# ════════════════════════════════════════════════════════════════════════════
# QUALIFICATION MODEL EVALUATION IMPORTS
# ════════════════════════════════════════════════════════════════════════════
# Imports for evaluating miner-submitted qualification models
QUALIFICATION_AVAILABLE = False
QUALIFICATION_IMPORT_ERROR = None
try:
    from qualification.validator.main import QualificationValidator
    from gateway.qualification.config import CONFIG as QUALIFICATION_CONFIG
    QUALIFICATION_AVAILABLE = True
    print("✅ Qualification system modules loaded successfully")
except ImportError as e:
    QUALIFICATION_IMPORT_ERROR = str(e)
    print(f"⚠️ Qualification system NOT available: {e}")
    # Qualification module not installed - will log at runtime if needed

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

# Skip auto-containerization for worker modes (they should NOT trigger deployment)
_is_worker_mode = "--mode" in sys.argv and any(
    m in sys.argv for m in ["qualification_worker", "worker"]
)

if __name__ == "__main__" and os.environ.get("LEADPOET_CONTAINER_MODE") != "1" and not _is_worker_mode:
    # Check if proxies are configured for containerization
    proxies_found = []
    for i in range(1, 151):  # Check for up to 150 proxies (supports scaling)
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

# ════════════════════════════════════════════════════════════════════════════
# DEDICATED QUALIFICATION CONTAINERS CONFIGURATION
# ════════════════════════════════════════════════════════════════════════════
# 5 containers dedicated ONLY to qualification model evaluation.
# These run PARALLEL to sourcing (not after).
# Set via QUALIFICATION_WEBSHARE_PROXY_1 through QUALIFICATION_WEBSHARE_PROXY_5
# ════════════════════════════════════════════════════════════════════════════

QUALIFICATION_CONTAINERS_COUNT = 5  # 5 dedicated qualification containers
QUALIFICATION_MODELS_PER_CONTAINER = 2  # Each container handles 2 models per epoch
QUALIFICATION_MAX_MODELS_PER_EPOCH = QUALIFICATION_CONTAINERS_COUNT * QUALIFICATION_MODELS_PER_CONTAINER  # 10 models
QUALIFICATION_MAX_MODELS_WITH_REBENCHMARK = (QUALIFICATION_CONTAINERS_COUNT - 1) * QUALIFICATION_MODELS_PER_CONTAINER  # 8 models (1 container does rebenchmark)

def detect_qualification_proxies():
    """Detect QUALIFICATION_WEBSHARE_PROXY_* environment variables."""
    proxies_found = []
    for i in range(1, QUALIFICATION_CONTAINERS_COUNT + 1):
        proxy_var = f"QUALIFICATION_WEBSHARE_PROXY_{i}"
        proxy_value = os.getenv(proxy_var)
        if proxy_value and proxy_value != "http://YOUR_USERNAME:YOUR_PASSWORD@p.webshare.io:80":
            proxies_found.append((proxy_var, proxy_value))
    return proxies_found

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
        
        # Qualification system state (initialized here to avoid AttributeError)
        self._qualification_session_id = None
        self._qualification_validator = None
        
        self.precision = 15.0 
        self.consistency = 1.0  
        self.collusion_flag = 1
        self.reputation = self.precision * self.consistency * self.collusion_flag  
        self.validation_history = []  
        self.trusted_validator = False  
        self.registration_time = datetime.now()  
        self.appeal_status = None  
        
        # initialize_pool imported at module level
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
            # get_supabase_client imported at module level
            
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
        """Legacy method - buyer feedback not currently used in gateway architecture."""
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
            self.validation_history.append({"O_v": s["O_v"], "F": F, "timestamp": datetime.now()})
        
        self.update_consistency()
        self.update_reputation()
        
        for i, (reward, response) in enumerate(zip(rewards, responses)):
            if reward >= 0.9 and isinstance(response, LeadRequest) and response.leads:
                if await self.run_automated_checks(response.leads):
                    # add_to_pool imported at module level
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
            
            # NOTE: pending_reveals saving REMOVED (Jan 2026) - IMMEDIATE REVEAL MODE
            # Validators now submit hash+values in one request, no separate reveal phase
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
        
        # NOTE: pending_reveals loading REMOVED (Jan 2026) - IMMEDIATE REVEAL MODE
        # Validators now submit hash+values in one request, no separate reveal phase

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
        # NOTE: _pending_reveals REMOVED (Jan 2026) - IMMEDIATE REVEAL MODE

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
                # broadcast_api_request imported at module level

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
            # fetch_validator_rankings and get_broadcast_status imported at module level

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
                # reward_module and cloud_db_module imported at module level
                
                reward_module.inject_async_subtensor(self.async_subtensor)
                cloud_db_module._VERIFY.inject_async_subtensor(self.async_subtensor)
                
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
                    
                    # ════════════════════════════════════════════════════════════
                    # QUALIFICATION MODEL EVALUATION (polls gateway for miner models)
                    # Enable with: export ENABLE_QUALIFICATION_EVALUATION=true
                    # ════════════════════════════════════════════════════════════
                    try:
                        await self.process_qualification_workflow()
                    except Exception as e:
                        bt.logging.warning(f"Error in process_qualification_workflow: {e}")

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
            
            # ═══════════════════════════════════════════════════════════════════
            # QUALIFICATION MODEL ASSIGNMENT (FIRST - before sourcing!)
            # ═══════════════════════════════════════════════════════════════════
            # Coordinator FIRST assigns qualification models to dedicated qual workers
            # This happens IMMEDIATELY at epoch start, PARALLEL to sourcing
            # Only coordinator does this - workers just read from assigned files
            # ═══════════════════════════════════════════════════════════════════
            container_mode_check = getattr(self.config.neuron, 'mode', None)
            if container_mode_check == "coordinator" or container_mode_check is None:
                # Check if qualification is enabled
                qual_enabled = os.environ.get("ENABLE_QUALIFICATION_EVALUATION", "").lower() in ("true", "1", "yes")
                qual_proxies = detect_qualification_proxies()
                
                if qual_enabled and qual_proxies:
                    print(f"\n🎯 QUALIFICATION: Assigning models to {len(qual_proxies)} dedicated workers...")
                    try:
                        await self._assign_qualification_to_dedicated_workers(current_epoch)
                        print(f"   ✅ Qualification assignment complete - sourcing begins NOW")
                    except Exception as qual_assign_err:
                        print(f"   ⚠️ Qualification assignment failed: {qual_assign_err}")
                        print(f"   Continuing with sourcing...")
            
            # Fetch assigned leads from gateway
            # gateway_get_epoch_leads, gateway_submit_validation imported at module level
            # NOTE: gateway_submit_reveal REMOVED (Jan 2026) - IMMEDIATE REVEAL MODE
            
            # ═══════════════════════════════════════════════════════════════════
            # OPTIMIZED LEAD FETCHING: Only coordinator calls gateway
            # Workers read from shared file to avoid N duplicate API calls
            # ═══════════════════════════════════════════════════════════════════
            container_mode = getattr(self.config.neuron, 'mode', None)
            container_id = getattr(self.config.neuron, 'container_id', None)
            
            # hashlib needed for salt generation (os is already imported at module level)
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
                    
                    # Too late to start validation (coordinator aggregates at block 300)
                    # Workers need ~8-10 min to process leads, so cutoff at block 260
                    # gives them 40 blocks (8 min) before coordinator forces aggregation
                    if blocks_into_epoch >= 260:
                        print(f"❌ Worker: Too late to start validation (block {blocks_into_epoch}/360)")
                        print(f"   Coordinator aggregates at block 300 - not enough time to finish")
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
                    leads_file_path=leads_file_str,  # Poll file for TrueList results after Stage 0-2
                    current_epoch=current_epoch  # For epoch boundary detection mid-processing
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
                    
                    # Store result for gateway submission (IMMEDIATE REVEAL MODE)
                    # IMMEDIATE REVEAL MODE (Jan 2026): Include BOTH hashes AND actual values
                    # No separate reveal phase - gateway verifies hashes and stores values immediately
                    # lead_id and miner_hotkey are at top level (not in lead_blob)
                    validation_results.append({
                        "lead_id": lead.get("lead_id"),  # Top level
                        # Hash fields (for transparency log integrity)
                        "decision_hash": decision_hash,
                        "rep_score_hash": rep_score_hash,
                        "rejection_reason_hash": rejection_reason_hash,
                        "evidence_hash": evidence_hash,
                        "evidence_blob": result,  # Include full evidence for gateway storage
                        # IMMEDIATE REVEAL FIELDS - no separate reveal phase
                        "decision": decision,
                        "rep_score": rep_score,
                        "rejection_reason": rejection_reason,
                        "salt": salt.hex()
                    })
                    
                    # Store local data for weight calculation (still needed for local weight accumulation)
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
                    # CRITICAL FIX: Get is_icp_multiplier from automated_checks_data (where it's calculated)
                    # NOT from lead (which is the gateway lead object, not the lead_blob that was validated)
                    if len(local_validation_data) > 0:
                        local_validation_data[-1]["is_icp_multiplier"] = automated_checks_data.get("is_icp_multiplier", 0.0)
                    
                    # Only accumulate now if NOT in container mode (backward compatibility)
                    # In container mode, coordinator will accumulate ALL leads after aggregation
                    if container_mode is None:
                        # Traditional single-validator mode
                        # CRITICAL FIX: Get from automated_checks_data, not lead
                        is_icp_multiplier = automated_checks_data.get("is_icp_multiplier", 0.0)
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
                        
                        # EPOCH CHANGE CHECK: If epoch changed, abort immediately
                        # Without this, coordinator sits in wait loop for 60min doing nothing
                        if current_epoch_check > current_epoch:
                            print(f"\n{'='*60}")
                            print(f"❌ COORDINATOR: EPOCH CHANGED while waiting for workers!")
                            print(f"   Started: epoch {current_epoch}")
                            print(f"   Current: epoch {current_epoch_check}")
                            print(f"   Aborting - stale results cannot be submitted")
                            print(f"{'='*60}\n")
                            break
                        
                        # FORCE PROCEED at block 300 (provides ~12 min buffer for weight accum + gateway submit)
                        # Block 300 = 60 min into epoch, leaves 12 min before epoch ends
                        # Weight accumulation (~5 min) + gateway submit (~5 sec) = ~5 min total
                        # Buffer: 12 - 5 = ~7 minutes spare
                        if blocks_into_epoch_check >= 300:
                            print(f"   ⏰ BLOCK 300+ REACHED: Force proceeding with available results")
                            print(f"      Block: {blocks_into_epoch_check}/360")
                            print(f"      ~12 minutes remaining for weight accumulation + gateway submission")
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
                
                # Clean up stale worker result files from previous epochs
                try:
                    for old_worker_file in weights_dir.glob("worker_*_epoch_*_results.json"):
                        try:
                            # Extract epoch from filename: worker_X_epoch_YYYY_results.json
                            parts = old_worker_file.stem.split('_')
                            epoch_idx = parts.index('epoch') + 1
                            file_epoch = int(parts[epoch_idx])
                            if file_epoch < current_epoch:
                                os.remove(old_worker_file)
                                print(f"   🧹 Cleaned up stale worker file: {old_worker_file.name}")
                        except (IndexError, ValueError):
                            pass
                except Exception as e:
                    print(f"   ⚠️  Could not clean up stale worker files: {e}")
                
                # ═══════════════════════════════════════════════════════════════════
                # COORDINATOR: Accumulate weights for ALL leads (coordinator + workers)
                # This ensures all leads are counted in validator_weights_history
                # ═══════════════════════════════════════════════════════════════════
                print(f"   ⚖️  Accumulating weights for all {len(local_validation_data)} leads...")
                for val_data in local_validation_data:
                    miner_hotkey = val_data.get("miner_hotkey")
                    decision = val_data.get("decision")
                    rep_score = val_data.get("rep_score", 0)
                    # Default to 0.0 (new format: no adjustment) instead of 1.0 (old format: multiplier)
                    is_icp_multiplier = val_data.get("is_icp_multiplier", 0.0)
                    
                    await self.accumulate_miner_weights(
                        miner_hotkey=miner_hotkey,
                        rep_score=rep_score,
                        is_icp_multiplier=is_icp_multiplier,
                        decision=decision
                    )
                print(f"   ✅ Weight accumulation complete")
                
                print(f"   Proceeding with gateway submission...")
                print(f"{'='*80}\n")
            
            # Submit validation results to gateway (IMMEDIATE REVEAL MODE)
            # IMMEDIATE REVEAL MODE (Jan 2026): Submit both hashes AND actual values
            # No separate reveal phase - gateway verifies hashes and stores values immediately
            # Consensus runs at end of CURRENT epoch (not N+1)
            print(f"{'='*80}")
            
            # Check if epoch changed before attempting submission
            submit_block = await self.get_current_block_async()
            submit_epoch = submit_block // 360
            
            if submit_epoch > current_epoch:
                print(f"⚠️  Epoch changed ({current_epoch} → {submit_epoch}) - skipping validation submission")
                print(f"   {len(validation_results)} validations for epoch {current_epoch} cannot be submitted")
                print(f"   (Weights already submitted, epoch will be marked as processed)")
                success = False
            elif validation_results:
                print(f"📤 Submitting {len(validation_results)} validations to gateway (IMMEDIATE REVEAL MODE)...")
                success = gateway_submit_validation(self.wallet, current_epoch, validation_results)
                if success:
                    print(f"✅ Successfully submitted {len(validation_results)} validations for epoch {current_epoch}")
                    print(f"   Mode: IMMEDIATE REVEAL (hashes + actual values submitted together)")
                    print(f"   Gateway logged to TEE buffer → will be in next Arweave checkpoint")
                    print(f"   ✅ No separate reveal phase needed - consensus will run at block 330")
                    # NOTE: No _pending_reveals storage needed - values already submitted
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
            
            # NOTE: process_pending_reveals() REMOVED - IMMEDIATE REVEAL MODE
            # With immediate reveal, validators submit both hashes AND values in one request
            # No separate reveal phase is needed - consensus runs at block 330 of CURRENT epoch
            
            # Check if we should submit weights (block 345+)
            await self.submit_weights_at_epoch_end()
            
            # MEMORY CLEANUP: Force garbage collection after each epoch
            # This prevents memory accumulation over long-running sessions
            collected = gc.collect()
            if collected > 100:  # Only log if significant cleanup
                print(f"🧹 Memory cleanup: freed {collected} objects")
            
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
                    "max_leads_per_epoch": getattr(self, '_max_leads_per_epoch', 3000),  # Persist for restart recovery
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
                # NEW FORMAT: Use addition with floor at 0 (for normal leads)
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
                "max_leads_per_epoch": getattr(self, '_max_leads_per_epoch', epoch_data.get("max_leads_per_epoch", 3000)),  # Persist for restart recovery
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
            
            # ═══════════════════════════════════════════════════════════════════
            # SOURCING EMISSIONS SYSTEM (Threshold-Based)
            # ═══════════════════════════════════════════════════════════════════
            # Allocation shares (must sum to 1.0)
            BASE_BURN_SHARE = 0.0          # 0% base burn to UID 0
            CHAMPION_SHARE = 0.05          # 5% to qualification model champion
            MAX_SOURCING_SHARE = 0.95      # 95% max to sourcing miners
            
            # CONFIGURABLE THRESHOLD: Approved leads needed in 30 epochs for full 95%
            # If network produces >= this many leads, full 95% is distributed
            # If below, proportional share distributed and rest burned
            # This puts a floor at roughly current profitability level
            SOURCING_FLOOR_THRESHOLD = 125_000  # EASILY ADJUSTABLE
            
            # Minimum total rep score to distribute (prevents tiny denominator instability)
            # If total rep < this, sourcing share goes to burn
            MIN_TOTAL_REP_FOR_DISTRIBUTION = 100
            
            # Rolling window for historical lead count and rep scores
            ROLLING_WINDOW = 30
            
            # Champion beat threshold is defined in qualification/config.py (CHAMPION_DETHRONING_THRESHOLD_PCT)
            # Currently set to 2% - challenger must beat champion by 2% to dethrone
            # Champion rebenchmark time is defined in qualification/config.py:
            #   CHAMPION_REBENCHMARK_HOUR_UTC, CHAMPION_REBENCHMARK_MINUTE_UTC
            # Default: 05:00 UTC (5:00 AM) - first full epoch after this time triggers rebenchmark
            
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
            has_rolling_history = bool(rolling_scores)
            
            print(f"\n{'='*80}")
            print(f"⚖️  SUBMITTING WEIGHTS FOR EPOCH {current_epoch}")
            print(f"{'='*80}")
            print(f"   Block: {current_block} (block {blocks_into_epoch}/360 into epoch)")
            print(f"   Rolling {ROLLING_WINDOW} epoch miners: {len(rolling_scores)}")
            print(f"   Rolling {ROLLING_WINDOW} epoch leads: {rolling_lead_count:,}")
            print(f"   Sourcing floor threshold: {SOURCING_FLOOR_THRESHOLD:,}")
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
            
            # ═══════════════════════════════════════════════════════════════════
            # QUALIFICATION CHAMPION: Read from local JSON for 1% emission share
            # ═══════════════════════════════════════════════════════════════════
            # Champion info is stored locally in validator_weights/qualification_champion.json
            # Updated when validator evaluates models via _store_qualification_champion()
            champion_hotkey = None
            champion_uid = None
            effective_champion_share = 0.0
            
            try:
                champion_data = self._read_qualification_champion()
                
                if champion_data:
                    champion_hotkey = champion_data.get("miner_hotkey")
                    print(f"   👑 QUALIFICATION CHAMPION (from local JSON):")
                    print(f"      Model: {champion_data.get('model_name', 'Unknown')}")
                    print(f"      Miner: {champion_hotkey[:20] if champion_hotkey else 'Unknown'}...")
                    print(f"      Score: {champion_data.get('score', 0):.2f}")
                    print(f"      Since: {champion_data.get('became_champion_at', 'Unknown')}")
                    
                    # Verify champion is registered on subnet
                    if champion_hotkey and champion_hotkey in self.metagraph.hotkeys:
                        champion_uid = self.metagraph.hotkeys.index(champion_hotkey)
                        effective_champion_share = CHAMPION_SHARE
                        print(f"      UID: {champion_uid}")
                        print(f"      Emission Share: {CHAMPION_SHARE*100:.0f}%")
                    else:
                        print(f"      ⚠️  Champion not registered on subnet - share goes to burn")
                else:
                    print(f"   📭 No qualification champion yet - {CHAMPION_SHARE*100:.0f}% goes to burn")
                    print(f"      (File: validator_weights/qualification_champion.json does not exist or is empty)")
            except Exception as e:
                print(f"   ⚠️  Error reading champion: {e} - {CHAMPION_SHARE*100:.0f}% goes to burn")
            
            print()
            
            # ═══════════════════════════════════════════════════════════════════
            # THRESHOLD-BASED SOURCING EMISSIONS
            # - If ≥SOURCING_FLOOR_THRESHOLD leads in 30 epochs: Full 95% distributed
            # - If <SOURCING_FLOOR_THRESHOLD: Proportional share, rest burned
            # - Within that share: split by rep score proportion
            # ═══════════════════════════════════════════════════════════════════
            
            # Convert miner hotkeys to UIDs (needed for all paths)
            all_miner_hotkeys = set(rolling_scores.keys())
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
                print(f"   🔥 Submitting 100% burn weights...")
                
                result = self.subtensor.set_weights(
                    netuid=self.config.netuid,
                    wallet=self.wallet,
                    uids=[UID_ZERO],
                    weights=[1.0],
                    wait_for_finalization=True
                )
                
                if result:
                    print(f"   ✅ Burn weights submitted successfully")
                    self._last_weight_submission_epoch = current_epoch
                    return True
                else:
                    print(f"   ❌ Failed to submit burn weights")
                    return False
            
            # ═══════════════════════════════════════════════════════════════════
            # Filter to REGISTERED miners only - deregistered miners' share → BURN
            # ═══════════════════════════════════════════════════════════════════
            registered_rolling_scores = {h: p for h, p in rolling_scores.items() if h in hotkey_to_uid}
            
            # Calculate totals
            all_rolling_total = sum(rolling_scores.values()) if rolling_scores else 0
            registered_rolling_total = sum(registered_rolling_scores.values()) if registered_rolling_scores else 0
            deregistered_rolling_points = all_rolling_total - registered_rolling_total
            
            # Log deregistered miners
            if deregistered_rolling_points > 0:
                print(f"   ⚠️  Deregistered miners: {deregistered_rolling_points:,} pts → share goes to BURN")
            
            # ═══════════════════════════════════════════════════════════════════
            # THRESHOLD CALCULATION
            # ═══════════════════════════════════════════════════════════════════
            if rolling_lead_count >= SOURCING_FLOOR_THRESHOLD:
                # ✅ Network healthy: ≥125k approved leads in 30 epochs
                # Full 95% split among sourcing miners
                effective_sourcing_share = MAX_SOURCING_SHARE
                print(f"   ✅ NETWORK HEALTHY - Full {MAX_SOURCING_SHARE*100:.0f}% to miners")
                print(f"      Approved leads ({ROLLING_WINDOW} epochs): {rolling_lead_count:,} ≥ {SOURCING_FLOOR_THRESHOLD:,}")
            else:
                # ⚠️ Below threshold: <125k approved leads
                # Proportional share to miners, rest burned
                effective_sourcing_share = (rolling_lead_count / SOURCING_FLOOR_THRESHOLD) * MAX_SOURCING_SHARE
                print(f"   ⚠️  BELOW THRESHOLD - Proportional distribution")
                print(f"      Approved leads ({ROLLING_WINDOW} epochs): {rolling_lead_count:,} < {SOURCING_FLOOR_THRESHOLD:,}")
                print(f"      Rate: {rolling_lead_count:,} / {SOURCING_FLOOR_THRESHOLD:,} = {(rolling_lead_count/SOURCING_FLOOR_THRESHOLD)*100:.1f}%")
                print(f"      → {effective_sourcing_share*100:.2f}% to miners")
                print(f"      → {(MAX_SOURCING_SHARE - effective_sourcing_share)*100:.2f}% burned (underperformance)")
            
            # Calculate burn for deregistered miners (proportional to their share of total)
            dereg_burn = 0.0
            if all_rolling_total > 0 and deregistered_rolling_points > 0:
                dereg_burn = effective_sourcing_share * (deregistered_rolling_points / all_rolling_total)
                print(f"      + {dereg_burn*100:.2f}% burned (deregistered miners)")
            
            # Effective sourcing share for registered miners only
            effective_sourcing_to_miners = effective_sourcing_share - dereg_burn
            
            # Calculate total burn share
            unused_sourcing_share = MAX_SOURCING_SHARE - effective_sourcing_share
            unused_champion_share = CHAMPION_SHARE - effective_champion_share
            total_burn_share = BASE_BURN_SHARE + unused_sourcing_share + unused_champion_share + dereg_burn
            
            print()
            print(f"   📊 WEIGHT DISTRIBUTION:")
            print(f"      Base burn:            {BASE_BURN_SHARE*100:.0f}%")
            print(f"      Unused sourcing:      {unused_sourcing_share*100:.2f}%")
            print(f"      Deregistered miners:  {dereg_burn*100:.2f}%")
            print(f"      Unused champion:      {unused_champion_share*100:.0f}%")
            print(f"      ─────────────────────────────")
            print(f"      Total burn → UID 0:   {total_burn_share*100:.2f}%")
            print(f"      Champion → UID {champion_uid if champion_uid else '?'}:     {effective_champion_share*100:.0f}%")
            print(f"      Sourcing miners:      {effective_sourcing_to_miners*100:.2f}%")
            print()
            
            # ═══════════════════════════════════════════════════════════════════
            # BUILD FINAL WEIGHTS
            # ═══════════════════════════════════════════════════════════════════
            uid_weights = {}
            
            # UID 0 gets total burn share
            uid_weights[UID_ZERO] = total_burn_share
            
            # Champion gets their share (if registered)
            if effective_champion_share > 0 and champion_uid is not None:
                if champion_uid not in uid_weights:
                    uid_weights[champion_uid] = 0
                uid_weights[champion_uid] += effective_champion_share
                print(f"   👑 Champion (UID {champion_uid}): {effective_champion_share*100:.0f}%")
            
            # ═══════════════════════════════════════════════════════════════════
            # DISTRIBUTE SOURCING SHARE BY REP SCORE
            # Formula: miner_weight = (miner_rep / total_rep) × effective_sourcing_to_miners
            # ═══════════════════════════════════════════════════════════════════
            print(f"   📈 Sourcing Miners ({effective_sourcing_to_miners*100:.2f}% split by rep score):")
            print(f"      Total registered rep score: {registered_rolling_total:,}")
            
            # Edge case: If total rep is below minimum OR zero, burn the sourcing share
            if registered_rolling_total < MIN_TOTAL_REP_FOR_DISTRIBUTION:
                print(f"      ⚠️  Total rep ({registered_rolling_total:,}) below minimum ({MIN_TOTAL_REP_FOR_DISTRIBUTION})")
                print(f"      → Burning sourcing share to prevent division instability")
                uid_weights[UID_ZERO] += effective_sourcing_to_miners
            else:
                # Distribute to registered miners by rep score proportion
                for hotkey, rep_score in registered_rolling_scores.items():
                    if rep_score <= 0:
                        continue  # Skip miners with 0 rep
            
                    uid = hotkey_to_uid[hotkey]
                    
                    # Core formula: proportion × effective share
                    miner_proportion = rep_score / registered_rolling_total
                    miner_weight = effective_sourcing_to_miners * miner_proportion
                    
                    if uid not in uid_weights:
                        uid_weights[uid] = 0
                    uid_weights[uid] += miner_weight
                    
                    print(f"      UID {uid}: {rep_score:,} / {registered_rolling_total:,} = {miner_proportion*100:.2f}% → {miner_weight*100:.4f}%")
            
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
    
    # ═══════════════════════════════════════════════════════════════════
    # NOTE (Jan 2026): process_pending_reveals() REMOVED - IMMEDIATE REVEAL MODE
    # ═══════════════════════════════════════════════════════════════════
    # Validators now submit both hashes AND actual values in one request to
    # gateway_submit_validation(). No separate reveal phase needed.
    # 
    # Benefits:
    # - Eliminates ~4500 UPDATE queries per epoch (reveals were updates)
    # - Reduces latency - consensus runs same epoch instead of N+1
    # - Simplifies workflow - one submission instead of two
    # ═══════════════════════════════════════════════════════════════════

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
            # submit_validation_assessment imported at module level
            import uuid
            
            # Fetch prospects using the new consensus-aware function
            # Returns list of {'prospect_id': UUID, 'data': lead_dict}
            prospects_batch = fetch_prospects_from_cloud(
                wallet=self.wallet,
                limit=3000,
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

    # ═══════════════════════════════════════════════════════════════════════════
    # QUALIFICATION MODEL EVALUATION WORKFLOW
    # ═══════════════════════════════════════════════════════════════════════════
    
    async def process_qualification_workflow(self):
        """
        Process qualification model evaluations.
        
        This polls the gateway for miner-submitted models and:
        1. Fetches models that need evaluation (new submissions or rebenchmarks)
        2. Runs them in TEE sandbox
        3. Scores the leads they return
        4. Reports results back to gateway
        5. Updates champion if a model beats current champion by >5%
        
        Called every iteration of main loop if ENABLE_QUALIFICATION_EVALUATION=true.
        """
        # Check if qualification is enabled
        env_value = os.environ.get("ENABLE_QUALIFICATION_EVALUATION", "")
        if not env_value.lower() in ("true", "1", "yes"):
            # Only log once to avoid spam
            if not hasattr(self, '_qual_disabled_logged'):
                bt.logging.debug(f"🎯 Qualification disabled (env={env_value!r})")
                self._qual_disabled_logged = True
            return
        
        if not QUALIFICATION_AVAILABLE:
            # Only log once to avoid spam
            if not hasattr(self, '_qual_unavailable_logged'):
                bt.logging.warning(f"🎯 ENABLE_QUALIFICATION_EVALUATION=true but qualification module not available")
                if QUALIFICATION_IMPORT_ERROR:
                    bt.logging.warning(f"   Import error: {QUALIFICATION_IMPORT_ERROR}")
                self._qual_unavailable_logged = True
            return
        
        # ═══════════════════════════════════════════════════════════════════
        # DEDICATED QUALIFICATION WORKERS: Skip this old flow if active
        # ═══════════════════════════════════════════════════════════════════
        # When QUALIFICATION_WEBSHARE_PROXY_* env vars are set, dedicated
        # qualification workers handle model evaluation PARALLEL to sourcing.
        # This old "after sourcing" flow should be disabled.
        # ═══════════════════════════════════════════════════════════════════
        qual_proxies = detect_qualification_proxies()
        if qual_proxies:
            # Dedicated workers are active - collect results instead of running old flow
            if not hasattr(self, '_qual_dedicated_results_logged'):
                self._qual_dedicated_results_logged = set()
            
            try:
                current_block = self.subtensor.block
                current_epoch = current_block // 360
                blocks_into_epoch = current_block % 360
            except:
                return
            
            # ═══════════════════════════════════════════════════════════════════
            # WAIT FOR SOURCING COMPLETE: Hash+values submission done
            # IMMEDIATE REVEAL MODE (Jan 2026): Only collect qualification results AFTER:
            # 1. Lead validation complete (_last_processed_epoch >= current_epoch)
            # 2. Hash+values submitted to gateway (no separate reveal phase)
            # ═══════════════════════════════════════════════════════════════════
            last_processed = getattr(self, '_last_processed_epoch', -1)
            
            # Check if sourcing is complete for this epoch
            if last_processed < current_epoch:
                # Sourcing still in progress - don't collect yet
                return
            
            # Track which epochs have had their qualification results collected
            # IMMEDIATE REVEAL MODE: No reveal checking needed - data submitted with hashes
            if not hasattr(self, '_qual_results_collected_epochs'):
                self._qual_results_collected_epochs = set()
            
            # Already collected results for this epoch?
            if current_epoch in self._qual_results_collected_epochs:
                return
            
            # ═══════════════════════════════════════════════════════════════════
            # QUALIFICATION COLLECTION LOGIC:
            # IMMEDIATE REVEAL MODE (Jan 2026): No reveal waiting needed
            # Validators submit hash+values in one request, so there's no separate reveal phase.
            # 1. Check if workers are done IMMEDIATELY after sourcing completes
            # 2. Block 335 is the CUTOFF - if workers aren't done by then, submit what's ready
            # ═══════════════════════════════════════════════════════════════════
            
            # NOTE: Previous reveal waiting logic REMOVED - IMMEDIATE REVEAL MODE
            # With immediate reveal, validators submit both hashes AND values in one request.
            # No separate reveal phase - just proceed directly to qualification collection.
            
            # Proceed directly to qualification workers.
            # Block 335 is the CUTOFF - if workers aren't done by then, submit what's ready
            past_cutoff = blocks_into_epoch >= 335
            
            # Collect results (with cutoff logic handled in the collection function)
            try:
                results, all_workers_done = await self._collect_dedicated_qualification_results(
                    current_epoch, 
                    force_submit=past_cutoff
                )
                
                # If not past cutoff and workers aren't done, wait and show progress
                if not past_cutoff and not all_workers_done:
                    # Track last log time for periodic updates
                    if not hasattr(self, '_qual_waiting_last_log_time'):
                        self._qual_waiting_last_log_time = 0
                    if not hasattr(self, '_qual_waiting_last_log_block'):
                        self._qual_waiting_last_log_block = -1
                    
                    import time
                    current_time = time.time()
                    # Log every 30 seconds OR on first call OR if block changed significantly
                    should_log = (
                        current_time - self._qual_waiting_last_log_time >= 30 or
                        self._qual_waiting_last_log_block == -1 or
                        blocks_into_epoch - self._qual_waiting_last_log_block >= 10
                    )
                    
                    if should_log:
                        # Get worker status for detailed logging
                        from pathlib import Path
                        weights_dir = Path("validator_weights")
                        pending_workers = []
                        completed_workers = []
                        for i in range(1, QUALIFICATION_CONTAINERS_COUNT + 1):
                            work_file = weights_dir / f"qual_worker_{i}_work_{current_epoch}.json"
                            results_file = weights_dir / f"qual_worker_{i}_results_{current_epoch}.json"
                            if work_file.exists():
                                if results_file.exists():
                                    completed_workers.append(i)
                                else:
                                    pending_workers.append(i)
                        
                        if pending_workers:
                            print(f"🎯 QUALIFICATION: Waiting for workers to finish (block {blocks_into_epoch}/335)")
                            print(f"   ⏳ Pending: Qual Workers {pending_workers}")
                            if completed_workers:
                                print(f"   ✅ Complete: Qual Workers {completed_workers}")
                        
                        self._qual_waiting_last_log_time = current_time
                        self._qual_waiting_last_log_block = blocks_into_epoch
                    return
                
                # Log collection
                print(f"\n{'='*70}")
                print(f"🎯 QUALIFICATION: Collecting worker results")
                print(f"{'='*70}")
                print(f"   Epoch: {current_epoch}")
                print(f"   Block: {blocks_into_epoch}/360")
                if past_cutoff and not all_workers_done:
                    print(f"   ⚠️ Block 335 cutoff reached - submitting available results")
                else:
                    print(f"   ✅ All workers complete - submitting results")
                
                if results:
                    print(f"   📊 Collected {len(results)} model result(s) from workers")
                    await self._process_dedicated_qualification_results(results, current_epoch)
                else:
                    print(f"   ℹ️ No qualification results to process")
                
                # Mark as collected
                self._qual_results_collected_epochs.add(current_epoch)
                self._qual_dedicated_results_logged.add(current_epoch)
                
            except Exception as e:
                print(f"   ⚠️ Error collecting qualification results: {e}")
                import traceback
                traceback.print_exc()
                # Still mark as attempted to avoid infinite retries
                self._qual_results_collected_epochs.add(current_epoch)
                self._qual_dedicated_results_logged.add(current_epoch)
            return
        
        # ═══════════════════════════════════════════════════════════════════
        # DEDICATED QUALIFICATION WORKERS REQUIRED
        # ═══════════════════════════════════════════════════════════════════
        # Qualification now runs PARALLEL to sourcing via dedicated workers.
        # Set QUALIFICATION_WEBSHARE_PROXY_1 through QUALIFICATION_WEBSHARE_PROXY_5
        # to enable dedicated qualification containers.
        # ═══════════════════════════════════════════════════════════════════
        if not hasattr(self, '_qual_no_dedicated_workers_logged'):
            self._qual_no_dedicated_workers_logged = True
            print(f"⚠️ QUALIFICATION: No dedicated workers detected")
            print(f"   Set QUALIFICATION_WEBSHARE_PROXY_1 through QUALIFICATION_WEBSHARE_PROXY_5")
            print(f"   to enable parallel qualification model evaluation")
        return
    
    async def _qualification_register(self):
        """Register with the gateway for qualification work."""
        try:
            import httpx
            import hashlib
            
            gateway_url = os.environ.get("GATEWAY_URL", "http://54.226.209.164:8000")
            hotkey = self.wallet.hotkey.ss58_address
            timestamp = int(time.time())
            
            # Sign timestamp with hotkey
            message = str(timestamp).encode()
            signature = self.wallet.hotkey.sign(message).hex()
            
            payload = {
                "timestamp": timestamp,
                "signed_timestamp": signature,
                "hotkey": hotkey,
                "commit_hash": os.environ.get("VALIDATOR_CODE_VERSION", "unknown")
            }
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{gateway_url}/qualification/validator/register",
                    json=payload
                )
                response.raise_for_status()
                data = response.json()
                
                self._qualification_session_id = data.get("session_id")
                bt.logging.info(f"🎯 Qualification registered: session={self._qualification_session_id[:8]}...")
                
        except Exception as e:
            bt.logging.warning(f"Qualification registration failed: {e}")
            raise
    
    async def _qualification_request_work(self) -> Optional[Dict]:
        """Request single evaluation work from gateway (for backwards compatibility)."""
        if not self._qualification_session_id:
            return None
            
        try:
            import httpx
            
            gateway_url = os.environ.get("GATEWAY_URL", "http://54.226.209.164:8000")
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{gateway_url}/qualification/validator/request-evaluation",
                    json={"session_id": self._qualification_session_id}
                )
                
                if response.status_code == 404:
                    # Session expired, need to re-register
                    self._qualification_session_id = None
                    return None
                    
                response.raise_for_status()
                return response.json()
                
        except Exception as e:
            bt.logging.warning(f"Qualification work request failed: {e}")
            return None
    
    async def _qualification_request_batch_work(self, max_models: int = None, epoch: int = None) -> Optional[Dict]:
        """
        Request a batch of models for evaluation from gateway.
        
        This is the preferred method for coordinator to fetch all pending models
        at once (FIFO order by created_at, where evaluated_at is NULL).
        
        Args:
            max_models: Max models to request. Defaults to QUALIFICATION_CONFIG.MAX_MODELS_PER_EPOCH
            epoch: Current epoch (for logging)
        
        Returns:
            Dict with 'has_work', 'models', and 'queue_depth' fields
        """
        if not self._qualification_session_id:
            return None
        
        if max_models is None:
            max_models = QUALIFICATION_CONFIG.MAX_MODELS_PER_EPOCH
        
        try:
            import httpx
            
            gateway_url = os.environ.get("GATEWAY_URL", "http://54.226.209.164:8000")
            
            async with httpx.AsyncClient(timeout=60.0) as client:  # Longer timeout for batch
                response = await client.post(
                    f"{gateway_url}/qualification/validator/request-batch-evaluation",
                    json={
                        "session_id": self._qualification_session_id,
                        "max_models": max_models,
                        "epoch": epoch
                    }
                )
                
                if response.status_code == 404:
                    # Session expired, need to re-register
                    self._qualification_session_id = None
                    return None
                
                response.raise_for_status()
                return response.json()
                
        except Exception as e:
            bt.logging.warning(f"Qualification batch work request failed: {e}")
            return None
    
    async def _qualification_execute_work(self, work: Dict):
        """Execute qualification evaluation work."""
        try:
            import base64
            import httpx
            from qualification.validator.sandbox import TEESandbox
            from gateway.qualification.models import LeadOutput, ICPPrompt, LeadScoreBreakdown
            from qualification.scoring.lead_scorer import score_lead
            from qualification.scoring.pre_checks import run_automatic_zero_checks
            
            gateway_url = os.environ.get("GATEWAY_URL", "http://54.226.209.164:8000")
            evaluation_id = work.get("evaluation_id")
            model_code_b64 = work.get("agent_code", "")
            runs = work.get("evaluation_runs", [])
            is_rebenchmark = work.get("is_rebenchmark", False)  # Whether this is a rebenchmark of existing champion
            model_name = work.get("model_name", "Unknown")
            
            # Decode model code
            model_code = base64.b64decode(model_code_b64) if model_code_b64 else b""
            
            print(f"\n{'='*60}")
            print(f"🎯 QUALIFICATION EVALUATION STARTING")
            print(f"   Evaluation ID: {evaluation_id[:16] if evaluation_id else 'N/A'}...")
            print(f"   Model Name: {model_name}")
            print(f"   ICPs to test: {len(runs)}")
            print(f"   Model code size: {len(model_code)} bytes")
            print(f"   Is Rebenchmark: {'Yes' if is_rebenchmark else 'No'}")
            print(f"{'='*60}")
            
            # ═══════════════════════════════════════════════════════════════════════
            # HARDCODING DETECTION: Analyze code BEFORE running (ALL evaluations)
            # Note: Run on rebenchmarks too - catches models that became champion
            # before gaming detection was added
            # ═══════════════════════════════════════════════════════════════════════
            if model_code:
                try:
                    from qualification.validator.hardcoding_detector import (
                        analyze_model_for_hardcoding,
                        is_detection_enabled
                    )
                    
                    if is_detection_enabled():
                        print(f"\n🔍 HARDCODING DETECTION: Analyzing model code...")
                        
                        # Get ICP samples for context (first 5 ICPs)
                        icp_samples = [run.get("icp_data", {}) for run in runs[:5]]
                        
                        # Run detection with real ICP samples
                        detection_result = await analyze_model_for_hardcoding(
                            model_code=model_code,
                            icp_samples=icp_samples
                        )
                        
                        print(f"   Detection model: {detection_result.get('model_used', 'N/A')}")
                        print(f"   Confidence hardcoded: {detection_result.get('confidence_hardcoded', 0)}%")
                        print(f"   Analysis cost: ${detection_result.get('analysis_cost_usd', 0):.4f}")
                        
                        if not detection_result.get("passed", True):
                            # Model appears to be hardcoded - REJECT without running
                            print(f"\n❌ HARDCODING DETECTED: Model appears to be hardcoded!")
                            print(f"   Red flags: {detection_result.get('red_flags', [])}")
                            print(f"   Evidence: {detection_result.get('evidence', 'N/A')[:200]}...")
                            print(f"   Skipping model execution — score = 0")
                            
                            return  # Don't run the model
                        
                        print(f"   ✅ Hardcoding check PASSED")
                    else:
                        print(f"   ℹ️ Hardcoding detection disabled in config")
                        
                except ImportError as ie:
                    print(f"   ⚠️ Hardcoding detector not available: {ie}")
                except Exception as det_err:
                    print(f"   ⚠️ Hardcoding detection error (continuing): {det_err}")
                    # Only allow model to run if the detection itself failed to execute.
                    # If detection SUCCEEDED and returned passed=False, we already returned above.
            
            bt.logging.info(
                f"🎯 Starting qualification evaluation: "
                f"id={evaluation_id[:8] if evaluation_id else 'N/A'}..., "
                f"runs={len(runs)}, code_size={len(model_code)} bytes"
            )
            
            # Track seen companies for duplicate handling
            seen_companies = set()
            
            # Track total evaluation cost for $5 hard stop
            total_evaluation_cost = 0.0
            MAX_TOTAL_COST = QUALIFICATION_CONFIG.MAX_COST_PER_EVALUATION_USD  # $5.00
            evaluation_stopped_early = False
            
            # Track total scores and time for summary
            total_score = 0.0
            total_time = 0.0
            leads_scored = 0
            
            # Initialize sandbox
            sandbox = None
            try:
                api_proxy_url = f"{gateway_url}/qualification/proxy"
                
                sandbox = TEESandbox(
                    model_code=model_code,
                    evaluation_run_id=runs[0]["evaluation_run_id"] if runs else None,
                    api_proxy_url=api_proxy_url,
                    evaluation_id=evaluation_id
                )
                await sandbox.start()
                
                # Process each ICP
                for run_idx, run in enumerate(runs, 1):
                    # ═══════════════════════════════════════════════════════════
                    # $5 HARD STOP: Check BEFORE processing each lead
                    # ═══════════════════════════════════════════════════════════
                    if total_evaluation_cost >= MAX_TOTAL_COST:
                        print(f"\n   🛑 $5 HARD STOP: Total cost ${total_evaluation_cost:.2f} >= ${MAX_TOTAL_COST:.2f}")
                        print(f"   🛑 Stopping evaluation at ICP {run_idx}/{len(runs)} to protect costs")
                        evaluation_stopped_early = True
                        # Report remaining runs as cost-stopped
                        for remaining_run in runs[run_idx-1:]:
                            await self._qualification_report_error(
                                evaluation_run_id=remaining_run.get("evaluation_run_id"),
                                error_code=1005,
                                error_message=f"Evaluation stopped: $5 cost limit reached (${total_evaluation_cost:.2f})"
                            )
                        break
                    evaluation_run_id = run.get("evaluation_run_id")
                    icp_data = run.get("icp_data", {})
                    icp_industry = icp_data.get("industry", "Unknown")
                    
                    print(f"\n   📋 ICP {run_idx}/{len(runs)}: {icp_industry}")
                    
                    try:
                        # Create ICP prompt
                        icp = ICPPrompt(**icp_data)
                        
                        # Run model with timeout
                        start_time = time.time()
                        result = await asyncio.wait_for(
                            sandbox.run_model(icp),
                            timeout=QUALIFICATION_CONFIG.RUNNING_MODEL_TIMEOUT_SECONDS
                        )
                        run_time = time.time() - start_time
                        
                        # Get cost from sandbox (tracks API calls made by model)
                        run_cost = sandbox.get_run_cost() if hasattr(sandbox, 'get_run_cost') else 0.01
                        
                        # Accumulate total cost for $5 hard stop
                        total_evaluation_cost += run_cost
                        
                        # Parse lead output and check for errors
                        lead_data = result.get("lead") if isinstance(result, dict) else None
                        error_msg = result.get("error") if isinstance(result, dict) else None
                        lead = LeadOutput(**lead_data) if lead_data else None
                        
                        # Score lead
                        if lead:
                            scores = await score_lead(
                                lead=lead,
                                icp=icp,
                                run_cost_usd=run_cost,
                                run_time_seconds=run_time,
                                seen_companies=seen_companies
                            )
                        else:
                            # Use actual error message if available
                            failure_reason = error_msg if error_msg else "No lead returned"
                            scores = LeadScoreBreakdown(
                                icp_fit=0, decision_maker=0, intent_signal_raw=0,
                                time_decay_multiplier=1.0, intent_signal_final=0,
                                cost_penalty=0, time_penalty=0, final_score=0,
                                failure_reason=failure_reason
                            )
                        
                        # Report results
                        await self._qualification_report_results(
                            evaluation_run_id=evaluation_run_id,
                            lead=lead,
                            scores=scores,
                            run_cost_usd=run_cost,
                            run_time_seconds=run_time
                        )
                        
                        # Accumulate scores and time for summary
                        total_score += scores.final_score
                        total_time += run_time
                        leads_scored += 1
                        
                        if lead:
                            print(f"      ✅ Lead returned: {lead.role} @ {lead.business}")
                            print(f"      📊 Score: {scores.final_score:.2f} (ICP:{scores.icp_fit}, DM:{scores.decision_maker}, Intent:{scores.intent_signal_final:.2f})")
                        else:
                            print(f"      ❌ No lead returned: {scores.failure_reason}")
                        print(f"      ⏱️  Time: {run_time:.2f}s, 💰 Cost: ${run_cost:.6f} (Total: ${total_evaluation_cost:.4f}/${MAX_TOTAL_COST:.2f})")
                        
                        bt.logging.info(
                            f"🎯 Run completed: score={scores.final_score:.2f}, "
                            f"time={run_time:.2f}s"
                        )
                        
                    except asyncio.TimeoutError:
                        print(f"      ⚠️  TIMEOUT: Model took too long")
                        # Still count time for timeouts (use timeout value)
                        total_time += QUALIFICATION_CONFIG.RUNNING_MODEL_TIMEOUT_SECONDS
                        await self._qualification_report_error(
                            evaluation_run_id=evaluation_run_id,
                            error_code=1010,
                            error_message="Model timeout"
                        )
                    except Exception as e:
                        print(f"      ❌ ERROR: {e}")
                        bt.logging.error(f"Qualification run error: {e}")
                        # Still count time even on errors (run_time is set if sandbox ran)
                        if 'run_time' in dir() and run_time > 0:
                            total_time += run_time
                        await self._qualification_report_error(
                            evaluation_run_id=evaluation_run_id,
                            error_code=1000,
                            error_message=str(e)
                        )
                        
            finally:
                if sandbox:
                    try:
                        await sandbox.cleanup()
                    except Exception as e:
                        bt.logging.warning(f"Sandbox cleanup error: {e}")
            
            # Calculate final average score
            avg_score = total_score / leads_scored if leads_scored > 0 else 0.0
            
            print(f"\n{'='*60}")
            if evaluation_stopped_early:
                print(f"🛑 QUALIFICATION EVALUATION STOPPED - $5 COST LIMIT")
            else:
                print(f"🎯 QUALIFICATION EVALUATION COMPLETE")
            print(f"   Model: {work.get('model_name', 'Unknown')}")
            print(f"   Miner: {work.get('miner_hotkey', 'Unknown')[:16]}...")
            print(f"   ICPs evaluated: {run_idx if 'run_idx' in dir() else len(runs)}/{len(runs)}")
            print(f"   📊 Final Score: {avg_score:.2f} / 100 (avg per ICP)")
            print(f"   ⏱️  Total Time: {total_time:.2f}s ({total_time/60:.1f} min)")
            print(f"   💰 Total cost: ${total_evaluation_cost:.4f}")
            if evaluation_stopped_early:
                print(f"   ⚠️  Remaining {len(runs) - run_idx + 1} ICPs skipped due to cost limit")
            print(f"{'='*60}\n")
            
            # ═══════════════════════════════════════════════════════════════════
            # CHAMPION DETERMINATION: Done locally by validator (not gateway)
            # Updates validator_weights/qualification_champion.json
            # ═══════════════════════════════════════════════════════════════════
            became_champion, is_rebenchmark = self._update_champion_if_needed(
                model_id=work.get("model_id", "unknown"),
                model_name=work.get("model_name", "Unknown"),
                miner_hotkey=work.get("miner_hotkey", "unknown"),
                score=avg_score,
                total_cost_usd=total_evaluation_cost,
                total_time_seconds=total_time,
                num_leads=leads_scored
            )
            
            # Extract code content from tarball for leaderboard display
            code_content = None
            if model_code:
                try:
                    import tarfile, io
                    code_files = {}
                    with tarfile.open(fileobj=io.BytesIO(model_code), mode='r:gz') as tar:
                        for member in tar.getmembers():
                            if not member.isfile():
                                continue
                            filename = member.name
                            if '/' in filename:
                                filename = filename.split('/', 1)[1] if filename.count('/') == 1 else filename
                            ext = '.' + filename.split('.')[-1].lower() if '.' in filename else ''
                            if ext not in {'.py', '.txt', '.md', '.json', '.yaml', '.yml', '.toml'}:
                                continue
                            if filename.startswith('.') or '/__' in filename:
                                continue
                            try:
                                f = tar.extractfile(member)
                                if f:
                                    code_files[filename] = f.read().decode('utf-8', errors='replace')
                            except Exception:
                                continue
                    if code_files:
                        code_content = json.dumps(code_files)
                        print(f"      📄 Extracted {len(code_files)} code files for display")
                except Exception as e:
                    print(f"      ⚠️ Could not extract code content: {e}")
            
            # Send champion status to gateway for Supabase storage (one-way, for auditing)
            # For rebenchmarks, this updates the champion's score (even if lower)
            # Include cost/time + code content for full DB update
            await self._notify_gateway_champion_status(
                model_id=work.get("model_id", "unknown"),
                became_champion=became_champion,
                score=avg_score,
                is_rebenchmark=is_rebenchmark,
                evaluation_cost_usd=total_evaluation_cost,
                evaluation_time_seconds=int(total_time),
                code_content=code_content
            )
                        
        except Exception as e:
            print(f"❌ QUALIFICATION ERROR: {e}")
            bt.logging.error(f"Qualification execution error: {e}")
            import traceback
            bt.logging.error(traceback.format_exc())
    
    async def _qualification_report_results(
        self,
        evaluation_run_id: str,
        lead: Optional[Any],
        scores: Any,
        run_cost_usd: float,
        run_time_seconds: float
    ):
        """
        Report qualification results to gateway.
        
        Gateway stores results in Supabase for leaderboard/auditing.
        Champion determination happens locally via _update_champion_if_needed().
        
        Handles both:
        - LeadOutput/LeadScoreBreakdown objects (from coordinator direct evaluation)
        - Dict data (from worker results forwarded by coordinator)
        """
        try:
            import httpx
            
            gateway_url = os.environ.get("GATEWAY_URL", "http://54.226.209.164:8000")
            
            # Handle both object and dict formats (coordinator vs worker-forwarded data)
            if hasattr(lead, 'model_dump'):
                lead_data = lead.model_dump()
            elif isinstance(lead, dict):
                lead_data = lead
            else:
                lead_data = None
            
            if hasattr(scores, 'model_dump'):
                scores_data = scores.model_dump()
                icp_fit = scores.icp_fit
                dm_score = scores.decision_maker
                intent_score = scores.intent_signal_final
                cost_penalty = scores.cost_penalty
                time_penalty = scores.time_penalty
                final_score = scores.final_score
            elif isinstance(scores, dict):
                scores_data = scores
                icp_fit = scores.get("icp_fit", 0)
                dm_score = scores.get("decision_maker", 0)
                intent_score = scores.get("intent_signal_final", 0)
                cost_penalty = scores.get("cost_penalty", 0)
                time_penalty = scores.get("time_penalty", 0)
                final_score = scores.get("final_score", 0)
            else:
                scores_data = None
                icp_fit = dm_score = intent_score = cost_penalty = time_penalty = final_score = 0
            
            payload = {
                "evaluation_run_id": evaluation_run_id,
                "lead_returned": lead_data,
                "lead_score": scores_data,
                "icp_fit_score": icp_fit,
                "decision_maker_score": dm_score,
                "intent_signal_score": intent_score,
                "cost_penalty": cost_penalty,
                "time_penalty": time_penalty,
                "final_lead_score": final_score,
                "run_cost_usd": run_cost_usd,
                "run_time_seconds": run_time_seconds,
                "status": "finished"
            }
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{gateway_url}/qualification/validator/report-results",
                    json=payload
                )
                response.raise_for_status()
                # Gateway stores for auditing - champion determination is local
                
        except Exception as e:
            bt.logging.warning(f"Failed to report qualification results: {e}")
    
    def _update_champion_if_needed(
        self,
        model_id: str,
        model_name: str,
        miner_hotkey: str,
        score: float,
        total_cost_usd: float,
        total_time_seconds: float,
        num_leads: int
    ) -> tuple:
        """
        Check if evaluated model should become champion, update local JSON.
        
        Champion determination is done LOCALLY by the validator (not gateway).
        
        Rules:
        1. If no current champion exists, new model becomes champion (if score > 0)
        2. If current champion exists, new model must beat by 2% (CHAMPION_DETHRONING_THRESHOLD_PCT)
        3. If this is a REBENCHMARK of the existing champion, ALWAYS update the score
        
        Format: validator_weights/qualification_champion.json
        {
            "current_champion": {
                "model_id": "...",
                "miner_hotkey": "...",
                "score": 0.93,
                "model_name": "Main7",
                "became_champion_at": "2026-01-24T...",
                "avg_cost_per_lead_usd": 0.00003,
                "avg_time_per_lead_seconds": 0.25
            },
            "ex_champion": {...} or null,
            "last_evaluated_model": {...},
            "last_updated": "2026-01-24T...",
            "champion_beat_threshold": 0.02
        }
        
        NOTE: Only stores current + ex champion. When new champion is crowned,
        old champion becomes ex, previous ex is deleted.
        
        Returns:
            Tuple of (became_champion: bool, is_rebenchmark: bool)
            - became_champion: True if model is now champion (new or rebenchmarked)
            - is_rebenchmark: True if this was a rebenchmark of existing champion
        """
        from gateway.qualification.config import CONFIG
        THRESHOLD = CONFIG.CHAMPION_DETHRONING_THRESHOLD_PCT
        
        try:
            weights_dir = Path("validator_weights")
            weights_dir.mkdir(exist_ok=True)
            champion_file = weights_dir / "qualification_champion.json"
            
            # Load existing data
            existing_data = {}
            if champion_file.exists():
                with open(champion_file, 'r') as f:
                    existing_data = json.load(f)
            
            current_champion = existing_data.get("current_champion")
            
            # Calculate averages
            avg_cost = total_cost_usd / num_leads if num_leads > 0 else 0
            avg_time = total_time_seconds / num_leads if num_leads > 0 else 0
            timestamp = datetime.utcnow().isoformat()
            
            # Build model data
            model_data = {
                "model_id": model_id,
                "model_name": model_name,
                "miner_hotkey": miner_hotkey,
                "score": score,
                "total_cost_usd": total_cost_usd,
                "total_time_seconds": total_time_seconds,
                "avg_cost_per_lead_usd": avg_cost,
                "avg_time_per_lead_seconds": avg_time,
                "num_leads_evaluated": num_leads
            }
            
            became_champion = False
            ex_champion = existing_data.get("ex_champion")  # Keep previous ex by default
            is_rebenchmark = False
            
            # Check if this is a REBENCHMARK of the current champion
            # Rebenchmarks should ALWAYS update the champion's score (regardless of higher/lower)
            if current_champion and current_champion.get("model_id") == model_id:
                is_rebenchmark = True
                old_score = current_champion.get("score", 0)
                score_change = score - old_score
                score_change_pct = (score_change / old_score * 100) if old_score > 0 else 0
                
                # Update champion's score (ALWAYS for rebenchmark)
                current_champion["score"] = score
                current_champion["total_cost_usd"] = total_cost_usd
                current_champion["total_time_seconds"] = total_time_seconds
                current_champion["avg_cost_per_lead_usd"] = avg_cost
                current_champion["avg_time_per_lead_seconds"] = avg_time
                current_champion["num_leads_evaluated"] = num_leads
                current_champion["last_rebenchmark_at"] = timestamp
                
                # Keep became_champion as True if they were already champion
                became_champion = True
                
                # Log the rebenchmark result
                change_indicator = "📈" if score_change > 0 else "📉" if score_change < 0 else "➡️"
                print(f"\n{'='*60}")
                print(f"🔄 CHAMPION REBENCHMARK COMPLETE")
                print(f"   Model: {model_name}")
                print(f"   Miner: {miner_hotkey[:20]}...")
                print(f"   {change_indicator} Score: {old_score:.2f} → {score:.2f} ({score_change_pct:+.1f}%)")
                print(f"   Avg Cost/Lead: ${avg_cost:.6f}")
                print(f"   Avg Time/Lead: {avg_time:.2f}s")
                print(f"   ✅ Champion score UPDATED to latest rebenchmark")
                print(f"{'='*60}\n")
            
            # Case 1: No current champion - become champion if score >= MINIMUM_CHAMPION_SCORE
            elif current_champion is None:
                MINIMUM_CHAMPION_SCORE = CONFIG.MINIMUM_CHAMPION_SCORE
                if score >= MINIMUM_CHAMPION_SCORE:
                    model_data["became_champion_at"] = timestamp
                    current_champion = model_data
                    became_champion = True
                    ex_champion = None  # No ex if this is first champion
                    print(f"\n{'='*60}")
                    print(f"👑 FIRST CHAMPION CROWNED!")
                    print(f"   Model: {model_name}")
                    print(f"   Miner: {miner_hotkey[:20]}...")
                    print(f"   Score: {score:.2f}")
                    print(f"   Avg Cost/Lead: ${avg_cost:.6f}")
                    print(f"   Avg Time/Lead: {avg_time:.2f}s")
                    print(f"{'='*60}\n")
                else:
                    print(f"\n📊 Model {model_name} scored {score:.2f} - below minimum champion threshold ({MINIMUM_CHAMPION_SCORE})")
            else:
                # Case 2: Current champion exists - new challenger needs to beat by threshold
                current_score = current_champion.get("score", 0)
                required_score = current_score * (1 + THRESHOLD)
                
                if score <= current_score:
                    # Didn't beat current score
                    print(f"\n📊 Model {model_name} (score: {score:.2f}) did not beat champion (score: {current_score:.2f})")
                elif score < required_score:
                    # Beat current but not by enough
                    improvement = ((score - current_score) / current_score) * 100 if current_score > 0 else 100
                    print(f"\n{'='*60}")
                    print(f"📊 CHALLENGER FELL SHORT")
                    print(f"   Model: {model_name} (score: {score:.2f})")
                    print(f"   Improvement: +{improvement:.1f}%")
                    print(f"   Required: +{THRESHOLD*100:.0f}% ({required_score:.2f})")
                    print(f"   Champion: {current_champion.get('model_name')} (score: {current_score:.2f})")
                    print(f"{'='*60}\n")
                else:
                    # Check minimum score requirement
                    MINIMUM_CHAMPION_SCORE = CONFIG.MINIMUM_CHAMPION_SCORE
                    if score < MINIMUM_CHAMPION_SCORE:
                        print(f"\n{'='*60}")
                        print(f"❌ MODEL BEAT CHAMPION BUT BELOW MINIMUM THRESHOLD")
                        print(f"   Model: {model_name} (score: {score:.2f})")
                        print(f"   Minimum Required: {MINIMUM_CHAMPION_SCORE}")
                        print(f"   Cannot become champion - score too low")
                        print(f"{'='*60}\n")
                    else:
                        # NEW CHAMPION! Beat by required threshold AND meets minimum
                        improvement = ((score - current_score) / current_score) * 100 if current_score > 0 else 100
                        
                        # Old champion becomes ex-champion
                        ex_champion = current_champion.copy()
                        ex_champion["dethroned_at"] = timestamp
                        
                        # New champion
                        model_data["became_champion_at"] = timestamp
                        current_champion = model_data
                        became_champion = True
                        
                        print(f"\n{'='*60}")
                        print(f"👑👑👑 NEW CHAMPION CROWNED! 👑👑👑")
                        print(f"   New Champion: {model_name}")
                        print(f"   Miner: {miner_hotkey[:20]}...")
                        print(f"   Score: {score:.2f} (+{improvement:.1f}%)")
                        print(f"   Avg Cost/Lead: ${avg_cost:.6f}")
                        print(f"   Avg Time/Lead: {avg_time:.2f}s")
                        print(f"   Dethroned: {ex_champion.get('model_name')} (score: {ex_champion.get('score'):.2f})")
                        print(f"{'='*60}\n")
            
            # Get current epoch for tracking rebenchmark timing
            try:
                current_block = self.subtensor.block
                current_epoch = current_block // 360
            except Exception:
                current_epoch = 0
            
            # Update champion's last evaluated epoch and UTC date if this is the champion or a rebenchmark
            if current_champion and (became_champion or is_rebenchmark):
                current_champion["last_evaluated_epoch"] = current_epoch
                # Track UTC date for ICP set refresh-based rebenchmark
                from datetime import datetime as dt_datetime, timezone as dt_timezone
                current_utc_date = dt_datetime.now(dt_timezone.utc).date().isoformat()
                current_champion["last_evaluated_utc_date"] = current_utc_date
            
            # Save to JSON
            new_data = {
                "current_champion": current_champion,
                "ex_champion": ex_champion,
                "last_evaluated_model": model_data,
                "last_updated": timestamp,
                "champion_beat_threshold": THRESHOLD,
                "current_epoch": current_epoch  # Track epoch for rebenchmark
            }
            
            with open(champion_file, 'w') as f:
                json.dump(new_data, f, indent=2)
            
            bt.logging.info(f"✅ Updated qualification_champion.json (became_champion={became_champion}, is_rebenchmark={is_rebenchmark})")
            return (became_champion, is_rebenchmark)
            
        except Exception as e:
            bt.logging.error(f"Failed to update champion: {e}")
            import traceback
            bt.logging.error(traceback.format_exc())
            return (False, False)
    
    def _read_qualification_champion(self) -> Optional[Dict[str, Any]]:
        """
        Read qualification champion info from local JSON file.
        
        Also checks Supabase banned_hotkeys table - if champion is banned,
        clears local file and returns None (5% goes to burn instead).
        
        Returns:
            Dict with current_champion info, or None if no champion
        """
        try:
            champion_file = Path("validator_weights") / "qualification_champion.json"
            
            if not champion_file.exists():
                bt.logging.debug("No qualification champion file found")
                return None
            
            with open(champion_file, 'r') as f:
                data = json.load(f)
            
            champion = data.get("current_champion")
            if not champion:
                return None
            
            # Check if champion's hotkey is banned in Supabase
            champion_hotkey = champion.get("miner_hotkey")
            if champion_hotkey and self._is_champion_hotkey_banned(champion_hotkey):
                bt.logging.warning(f"🚨 Champion hotkey {champion_hotkey[:20]}... is BANNED - clearing local champion")
                self._clear_qualification_champion_for_ban(champion_hotkey)
                return None
            
            return champion
            
        except Exception as e:
            bt.logging.warning(f"Failed to read qualification champion: {e}")
            return None
    
    def _is_champion_hotkey_banned(self, hotkey: str) -> bool:
        """Check if hotkey is in Supabase banned_hotkeys table.
        
        Uses public ANON key for read-only access to banned_hotkeys table
        (RLS policy allows public SELECT on this table).
        """
        try:
            from supabase import create_client
            
            # Public Supabase credentials (same as in cloud_db.py)
            SUPABASE_URL = "https://qplwoislplkcegvdmbim.supabase.co"
            SUPABASE_ANON_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InFwbHdvaXNscGxrY2VndmRtYmltIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDQ4NDcwMDUsImV4cCI6MjA2MDQyMzAwNX0.5E0WjAthYDXaCWY6qjzXm2k20EhadWfigak9hleKZk8"
            
            # Create client with ANON key for public read access
            supabase = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
            
            result = supabase.table("banned_hotkeys")\
                .select("hotkey")\
                .eq("hotkey", hotkey)\
                .limit(1)\
                .execute()
            
            is_banned = bool(result.data and len(result.data) > 0)
            if is_banned:
                bt.logging.info(f"🚨 Hotkey {hotkey[:20]}... found in banned_hotkeys table")
            return is_banned
        except Exception as e:
            bt.logging.warning(f"Failed to check banned hotkeys: {e}")
            return False  # On error, don't clear champion
    
    def _clear_qualification_champion_for_ban(self, banned_hotkey: str):
        """Clear champion from local JSON due to hotkey ban."""
        try:
            champion_file = Path("validator_weights") / "qualification_champion.json"
            if not champion_file.exists():
                return
            
            with open(champion_file, 'r') as f:
                data = json.load(f)
            
            old_champion = data.get("current_champion")
            if old_champion:
                # Track in dethronement history
                if "dethronement_history" not in data:
                    data["dethronement_history"] = []
                old_champion["dethroned_at"] = datetime.utcnow().isoformat()
                old_champion["dethrone_reason"] = "hotkey_banned"
                data["dethronement_history"].append(old_champion)
            
            data["current_champion"] = None
            
            with open(champion_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            print(f"\n{'='*60}")
            print(f"🚨 CHAMPION DETHRONED (HOTKEY BANNED)")
            print(f"   Hotkey: {banned_hotkey[:20]}...")
            print(f"   5% champion share → burn (UID 0)")
            print(f"{'='*60}\n")
            
        except Exception as e:
            bt.logging.error(f"Failed to clear banned champion: {e}")
    
    def _clear_qualification_champion(self):
        """
        Clear the qualification champion (dethrone without replacement).
        
        Called when champion's rebenchmark score falls below minimum threshold.
        Removes the current_champion from the local JSON file.
        """
        try:
            champion_file = Path("validator_weights") / "qualification_champion.json"
            
            if not champion_file.exists():
                bt.logging.debug("No qualification champion file to clear")
                return
            
            # Read existing data
            with open(champion_file, 'r') as f:
                data = json.load(f)
            
            # Record who was dethroned
            old_champion = data.get("current_champion")
            if old_champion:
                bt.logging.info(
                    f"👎 DETHRONING CHAMPION: {old_champion.get('model_name', 'Unknown')} "
                    f"(score: {old_champion.get('score', 0):.2f})"
                )
                
                # Track dethronement history
                if "dethronement_history" not in data:
                    data["dethronement_history"] = []
                
                from datetime import datetime, timezone
                old_champion["dethroned_at"] = datetime.now(timezone.utc).isoformat()
                old_champion["dethrone_reason"] = "score_below_minimum"
                data["dethronement_history"].append(old_champion)
            
            # Clear current champion
            data["current_champion"] = None
            data["last_cleared_at"] = datetime.now(timezone.utc).isoformat()
            
            # Save updated data
            with open(champion_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            bt.logging.info("✅ Champion cleared from local JSON - no champion exists")
            print(f"\n{'='*60}")
            print(f"👎 CHAMPION DETHRONED - NO REPLACEMENT")
            print(f"   There is currently NO CHAMPION")
            print(f"   Next qualifying model (score >= minimum) will become champion")
            print(f"{'='*60}\n")
            
        except Exception as e:
            bt.logging.error(f"Failed to clear qualification champion: {e}")
            import traceback
            bt.logging.error(traceback.format_exc())
    
    def _check_champion_rebenchmark_needed(self) -> Optional[Dict[str, Any]]:
        """
        Check if the current champion needs rebenchmarking.
        
        Rebenchmark Logic (ICP Set Refresh):
        - Rebenchmark on the FIRST FULL epoch that STARTS after 12:05 AM UTC daily
        - This aligns with when the ICP set is refreshed (12:00 AM UTC)
        - Track which UTC date the champion was last evaluated on
        - Only rebenchmark ONCE per day (on the new ICP set)
        
        Example:
        - Epoch 100: 11:30 PM -> 12:42 AM UTC (spans midnight) → Does NOT count
        - Epoch 101: 12:42 AM -> 1:54 AM UTC (starts after 12:05 AM) → REBENCHMARK
        - Epoch 102: 1:54 AM -> 3:06 AM UTC → No rebenchmark (already done today)
        
        Returns:
            Dict with champion info if rebenchmark needed, None otherwise
        """
        from datetime import datetime, timezone, timedelta
        
        try:
            champion_file = Path("validator_weights") / "qualification_champion.json"
            
            if not champion_file.exists():
                return None
            
            with open(champion_file, 'r') as f:
                data = json.load(f)
            
            champion = data.get("current_champion")
            if not champion:
                return None
            
            # Get current block and calculate epoch timing
            try:
                current_block = self.subtensor.block
                current_epoch = current_block // 360
            except Exception:
                bt.logging.warning("Could not get current block for rebenchmark check")
                return None
            
            # ═══════════════════════════════════════════════════════════════════
            # Calculate when the CURRENT EPOCH STARTED in UTC
            # Each epoch is 360 blocks, each block is ~12 seconds
            # ═══════════════════════════════════════════════════════════════════
            epoch_start_block = current_epoch * 360
            blocks_since_epoch_start = current_block - epoch_start_block
            seconds_since_epoch_start = blocks_since_epoch_start * 12  # ~12 sec per block
            
            now_utc = datetime.now(timezone.utc)
            epoch_start_utc = now_utc - timedelta(seconds=seconds_since_epoch_start)
            
            # Get the UTC date of the epoch START
            epoch_start_date = epoch_start_utc.date()
            epoch_start_hour = epoch_start_utc.hour
            epoch_start_minute = epoch_start_utc.minute
            
            # Check if this epoch STARTED after the configured rebenchmark time (UTC)
            # Default: 5:00 AM UTC (hour=5, minute=0) - for testing, production uses 12:05 AM
            rebenchmark_hour = QUALIFICATION_CONFIG.CHAMPION_REBENCHMARK_HOUR_UTC if QUALIFICATION_AVAILABLE else 0
            rebenchmark_minute = QUALIFICATION_CONFIG.CHAMPION_REBENCHMARK_MINUTE_UTC if QUALIFICATION_AVAILABLE else 5
            
            # Convert rebenchmark time and epoch start time to minutes since midnight for comparison
            rebenchmark_minutes_since_midnight = rebenchmark_hour * 60 + rebenchmark_minute
            epoch_start_minutes_since_midnight = epoch_start_hour * 60 + epoch_start_minute
            
            epoch_started_after_refresh = epoch_start_minutes_since_midnight >= rebenchmark_minutes_since_midnight
            
            # Get the UTC date we last evaluated on (if tracked)
            last_evaluated_utc_date = champion.get("last_evaluated_utc_date")
            today_utc_date_str = epoch_start_date.isoformat()  # YYYY-MM-DD format
            
            print(f"   📊 Rebenchmark check:")
            print(f"      Current epoch: {current_epoch}")
            print(f"      Epoch start (UTC): {epoch_start_utc.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"      Rebenchmark trigger time: {rebenchmark_hour:02d}:{rebenchmark_minute:02d} UTC")
            print(f"      Epoch started after trigger time: {epoch_started_after_refresh}")
            print(f"      Last evaluated UTC date: {last_evaluated_utc_date or 'Never'}")
            print(f"      Today's UTC date: {today_utc_date_str}")
            
            # ═══════════════════════════════════════════════════════════════════
            # REBENCHMARK CONDITIONS:
            # 1. This epoch STARTED after 12:05 AM UTC (not spanning midnight)
            # 2. Champion hasn't been evaluated today (new ICP set)
            # ═══════════════════════════════════════════════════════════════════
            needs_rebenchmark = (
                epoch_started_after_refresh and
                last_evaluated_utc_date != today_utc_date_str
            )
            
            if needs_rebenchmark:
                bt.logging.info(
                    f"🔄 Champion rebenchmark needed: New ICP set for {today_utc_date_str} "
                    f"(last evaluated: {last_evaluated_utc_date or 'Never'})"
                )
                print(f"   ✅ REBENCHMARK TRIGGERED: New ICP set for {today_utc_date_str}")
                return champion
            else:
                if not epoch_started_after_refresh:
                    print(f"   ⏭️ Skipping: Epoch started before {rebenchmark_hour:02d}:{rebenchmark_minute:02d} UTC")
                else:
                    print(f"   ⏭️ Skipping: Already evaluated on today's ICP set ({today_utc_date_str})")
            
            return None
            
        except Exception as e:
            bt.logging.warning(f"Failed to check champion rebenchmark: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    async def _request_champion_rebenchmark(self, champion: Dict[str, Any]) -> bool:
        """
        Request the gateway to queue the champion model for rebenchmarking.
        
        Args:
            champion: Champion info dict with model_id
        
        Returns:
            True if rebenchmark was queued successfully
        """
        try:
            import httpx
            
            gateway_url = os.environ.get("GATEWAY_URL", "http://54.226.209.164:8000")
            model_id = champion.get("model_id")
            
            if not model_id:
                bt.logging.warning("Champion has no model_id, cannot request rebenchmark")
                return False
            
            print(f"\n{'='*60}")
            print(f"🔄 REQUESTING CHAMPION REBENCHMARK")
            print(f"   Model: {champion.get('model_name', 'Unknown')}")
            print(f"   Model ID: {model_id[:16]}...")
            print(f"   Current Score: {champion.get('score', 0):.2f}")
            print(f"{'='*60}\n")
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{gateway_url}/qualification/validator/request-rebenchmark",
                    json={
                        "model_id": model_id,
                        "session_id": self._qualification_session_id
                    }
                )
                
                if response.status_code == 200:
                    result = response.json()
                    bt.logging.info(f"✅ Champion rebenchmark queued: {result}")
                    return True
                else:
                    bt.logging.warning(f"Failed to queue rebenchmark: {response.status_code} - {response.text}")
                    return False
                    
        except Exception as e:
            bt.logging.warning(f"Failed to request champion rebenchmark: {e}")
            return False
    
    async def _qualification_report_error(
        self,
        evaluation_run_id: str,
        error_code: int,
        error_message: str
    ):
        """Report qualification error to gateway."""
        try:
            import httpx
            
            gateway_url = os.environ.get("GATEWAY_URL", "http://54.226.209.164:8000")
            
            payload = {
                "evaluation_run_id": evaluation_run_id,
                "error_code": error_code,
                "error_message": error_message,
                "status": "error"
            }
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{gateway_url}/qualification/validator/report-error",
                    json=payload
                )
                response.raise_for_status()
                
        except Exception as e:
            bt.logging.warning(f"Failed to report qualification error: {e}")
    
    async def _notify_gateway_champion_status(
        self,
        model_id: str,
        became_champion: bool,
        score: float,
        is_rebenchmark: bool = False,
        was_dethroned: bool = False,
        evaluation_cost_usd: float = None,
        evaluation_time_seconds: int = None,
        code_content: str = None
    ):
        """
        Notify gateway about champion status (one-way, for Supabase storage).
        
        The validator determines champion locally and tells gateway the result.
        Gateway stores this in Supabase for leaderboard/auditing purposes.
        
        This is a one-way notification - gateway does not send anything back.
        
        Args:
            model_id: UUID of the model
            became_champion: True if this model is now champion
            score: The evaluation score
            is_rebenchmark: True if this is a rebenchmark of the existing champion
                           (signals gateway to update score even if became_champion=True)
            was_dethroned: True if champion was dethroned (score below minimum) with NO replacement
            evaluation_cost_usd: Total cost of the evaluation (optional)
            evaluation_time_seconds: Total time of the evaluation in seconds (optional)
            code_content: JSON string of code files for display (optional)
        """
        try:
            import httpx
            
            gateway_url = os.environ.get("GATEWAY_URL", "http://54.226.209.164:8000")
            
            payload = {
                "model_id": model_id,
                "became_champion": became_champion,
                "score": score,
                "is_rebenchmark": is_rebenchmark,  # Tell gateway this is a rebenchmark
                "was_dethroned": was_dethroned,  # Tell gateway champion was dethroned (no replacement)
                "determined_by": "validator"  # Indicates validator made the decision
            }
            
            # Add optional fields if provided (for full DB update)
            if evaluation_cost_usd is not None:
                payload["evaluation_cost_usd"] = evaluation_cost_usd
            if evaluation_time_seconds is not None:
                payload["evaluation_time_seconds"] = evaluation_time_seconds
            if code_content is not None:
                payload["code_content"] = code_content
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{gateway_url}/qualification/validator/champion-status",
                    json=payload
                )
                response.raise_for_status()
                bt.logging.info(f"✅ Notified gateway of champion status: model={model_id[:8]}..., became_champion={became_champion}, was_dethroned={was_dethroned}")
                
        except Exception as e:
            # Non-critical - local JSON is source of truth
            bt.logging.warning(f"Failed to notify gateway of champion status: {e}")

    # ═══════════════════════════════════════════════════════════════════════════
    # DEDICATED QUALIFICATION WORKERS: Assignment at Epoch Start
    # ═══════════════════════════════════════════════════════════════════════════
    # These methods handle assigning models to the 5 dedicated qualification containers
    # that run PARALLEL to sourcing (not after it)
    # ═══════════════════════════════════════════════════════════════════════════
    
    async def _assign_qualification_to_dedicated_workers(self, current_epoch: int):
        """
        Assign qualification models to dedicated qualification workers at EPOCH START.
        
        This runs BEFORE sourcing begins (parallel, not sequential).
        
        Distribution logic:
        - 5 dedicated qualification containers
        - Each handles up to 2 models per epoch
        - If rebenchmark needed: Worker 1 does rebenchmark, others get 2 each (8 max from queue)
        - If no rebenchmark: All 5 workers get 2 each (10 max from queue)
        
        Args:
            current_epoch: Current epoch number
        """
        import httpx
        from pathlib import Path
        
        # Check if we've already assigned for this epoch
        if not hasattr(self, '_qual_dedicated_last_assigned_epoch'):
            self._qual_dedicated_last_assigned_epoch = -1
        
        if self._qual_dedicated_last_assigned_epoch == current_epoch:
            print(f"   ℹ️ Already assigned qualification work for epoch {current_epoch}")
            return
        
        weights_dir = Path("validator_weights")
        weights_dir.mkdir(exist_ok=True)
        
        # Clean up old qualification worker files
        for old_file in weights_dir.glob("qual_worker_*_work_*.json"):
            try:
                file_epoch = int(old_file.stem.split('_')[-1])
                if file_epoch < current_epoch:
                    old_file.unlink()
                    print(f"   🧹 Cleaned up stale qual work: {old_file.name}")
            except:
                pass
        
        for old_file in weights_dir.glob("qual_worker_*_results_*.json"):
            try:
                file_epoch = int(old_file.stem.split('_')[-1])
                if file_epoch < current_epoch:
                    old_file.unlink()
                    print(f"   🧹 Cleaned up stale qual results: {old_file.name}")
            except:
                pass
        
        # Initialize qualification validator if needed
        if not QUALIFICATION_AVAILABLE:
            print(f"   ⚠️ Qualification module not available")
            return
        
        if not hasattr(self, '_qualification_validator') or not self._qualification_session_id:
            try:
                gateway_url = os.environ.get("GATEWAY_URL", "http://54.226.209.164:8000")
                
                self._qualification_validator = QualificationValidator(
                    hotkey=self.wallet.hotkey.ss58_address,
                    code_version=os.environ.get("VALIDATOR_CODE_VERSION", "unknown"),
                    platform_url=gateway_url
                )
                
                # Register with gateway
                await self._qualification_register()
                
            except Exception as e:
                print(f"   ❌ Failed to initialize qualification: {e}")
                return
        
        # Check if rebenchmark is needed
        rebenchmark_needed = self._check_champion_rebenchmark_needed()
        rebenchmark_model = None
        
        # Determine max models to pull
        if rebenchmark_needed:
            # Worker 1 does rebenchmark, others get 2 each = 4*2 = 8 from queue
            max_models = QUALIFICATION_MAX_MODELS_WITH_REBENCHMARK
            print(f"   🔄 Rebenchmark needed - pulling max {max_models} new models from queue")
            
            # ═══════════════════════════════════════════════════════════════════
            # FETCH REBENCHMARK MODEL: 
            # 1. Call /request-rebenchmark to queue champion (in-memory)
            # 2. Call /request-evaluation to get the work item
            # ═══════════════════════════════════════════════════════════════════
            try:
                champion_data = self._read_qualification_champion()
                if champion_data and champion_data.get("model_id"):
                    champion_model_id = champion_data.get("model_id")
                    print(f"   📥 Requesting rebenchmark for champion: {champion_data.get('model_name', 'Unknown')}")
                    
                    gateway_url = os.environ.get("GATEWAY_URL", "http://54.226.209.164:8000")
                    
                    # Step 1: Queue the rebenchmark (adds to in-memory queue on gateway)
                    async with httpx.AsyncClient(timeout=30.0) as client:
                        rebench_resp = await client.post(
                            f"{gateway_url}/qualification/validator/request-rebenchmark",
                            json={
                                "model_id": champion_model_id,
                                "session_id": self._qualification_session_id
                            }
                        )
                        if rebench_resp.status_code == 200:
                            rebench_result = rebench_resp.json()
                            print(f"   ✅ Champion queued for rebenchmark: {rebench_result.get('status', 'unknown')}")
                            
                            # Step 2: Fetch the work item from gateway (now in in-memory queue)
                            eval_resp = await client.post(
                                f"{gateway_url}/qualification/validator/request-evaluation",
                                json={"session_id": self._qualification_session_id}
                            )
                            if eval_resp.status_code == 200:
                                eval_data = eval_resp.json()
                                if eval_data.get("has_work"):
                                    rebenchmark_model = {
                                        "evaluation_id": eval_data.get("evaluation_id"),
                                        "model_id": eval_data.get("model_id"),
                                        "model_name": eval_data.get("model_name"),
                                        "miner_hotkey": eval_data.get("miner_hotkey"),
                                        "agent_code": eval_data.get("agent_code"),
                                        "evaluation_runs": [run.__dict__ if hasattr(run, '__dict__') else run for run in eval_data.get("evaluation_runs", [])],
                                        "icp_set_hash": eval_data.get("icp_set_hash", ""),
                                        "is_rebenchmark": True
                                    }
                                    print(f"   ✅ Got rebenchmark work item: {rebenchmark_model['model_name']}")
                        else:
                            print(f"   ⚠️ Failed to queue rebenchmark: {rebench_resp.status_code}")
                else:
                    print(f"   ⚠️ No champion found for rebenchmark")
            except Exception as rebench_err:
                print(f"   ⚠️ Rebenchmark request failed: {rebench_err}")
        else:
            # All 5 workers get 2 each = 10 from queue
            max_models = QUALIFICATION_MAX_MODELS_PER_EPOCH
            print(f"   📦 No rebenchmark - pulling max {max_models} models from queue")
        
        # Fetch batch of NEW models from gateway (DB query - excludes rebenchmark)
        all_models = []
        try:
            gateway_url = os.environ.get("GATEWAY_URL", "http://54.226.209.164:8000")
            
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    f"{gateway_url}/qualification/validator/request-batch-evaluation",
                    json={
                        "session_id": self._qualification_session_id,
                        "max_models": max_models,
                        "epoch": current_epoch
                    }
                )
                
                # Handle 404 specially - session expired, need to re-register
                if response.status_code == 404:
                    print(f"   🔄 Session expired (404), re-registering with gateway...")
                    self._qualification_session_id = None
                    # Try to re-register immediately
                    try:
                        await self._qualification_register()
                        # Retry the request with new session
                        response = await client.post(
                            f"{gateway_url}/qualification/validator/request-batch-evaluation",
                            json={
                                "session_id": self._qualification_session_id,
                                "max_models": max_models,
                                "epoch": current_epoch
                            }
                        )
                    except Exception as re_register_err:
                        print(f"   ❌ Re-registration failed: {re_register_err}")
                        # Will continue with empty all_models
                
                response.raise_for_status()
                batch_response = response.json()
                all_models = batch_response.get("models", [])
                
        except Exception as e:
            print(f"   ❌ Failed to fetch models from gateway: {e}")
            # Clear session on any error to force re-registration next epoch
            if "404" in str(e) or "Session not found" in str(e):
                self._qualification_session_id = None
            # Continue with rebenchmark if we have it
        
        if not all_models and not rebenchmark_model:
            print(f"   ℹ️ No models to evaluate this epoch")
            self._qual_dedicated_last_assigned_epoch = current_epoch
            return
        
        print(f"   📥 Received {len(all_models)} new models" + (f" + 1 rebenchmark" if rebenchmark_model else ""))
        
        # ═══════════════════════════════════════════════════════════════════
        # DISTRIBUTE MODELS TO DEDICATED QUALIFICATION WORKERS (ROUND-ROBIN)
        # ═══════════════════════════════════════════════════════════════════
        # Round-robin distribution: 1 model per worker, then wrap around
        # Example with 6 models and 5 workers:
        #   Worker 1 → Model 1, Model 6
        #   Worker 2 → Model 2
        #   Worker 3 → Model 3
        #   Worker 4 → Model 4
        #   Worker 5 → Model 5
        # If rebenchmark needed: Worker 1 gets rebenchmark only, others get new models
        # ═══════════════════════════════════════════════════════════════════
        
        worker_assignments = {}  # {qual_container_id: {"models": [...], "is_rebenchmark_container": bool}}
        
        if rebenchmark_needed and rebenchmark_model:
            # Worker 1 does ONLY the rebenchmark (full focus)
            worker_assignments[1] = {
                "models": [rebenchmark_model],
                "is_rebenchmark_container": True
            }
            # Mark rebenchmark model
            rebenchmark_model["is_rebenchmark"] = True
            
            # Distribute remaining models to workers 2-5 (round-robin)
            models_to_distribute = all_models
            available_workers = [2, 3, 4, 5]
        else:
            # No rebenchmark - all 5 workers available (round-robin)
            models_to_distribute = all_models
            available_workers = [1, 2, 3, 4, 5]
        
        # ROUND-ROBIN DISTRIBUTION: Each model goes to next worker, wrap around
        for model_idx, model in enumerate(models_to_distribute):
            # Round-robin: model 0 → worker[0], model 1 → worker[1], ...
            # model 5 → worker[0] again (wrap around)
            worker_id = available_workers[model_idx % len(available_workers)]
            
            if worker_id not in worker_assignments:
                worker_assignments[worker_id] = {"models": [], "is_rebenchmark_container": False}
            worker_assignments[worker_id]["models"].append(model)
        
        # Write work files for each worker
        for worker_id, assignment in worker_assignments.items():
            work_file = weights_dir / f"qual_worker_{worker_id}_work_{current_epoch}.json"
            with open(work_file, 'w') as f:
                json.dump({
                    "epoch": current_epoch,
                    "qual_worker_id": worker_id,
                    "models": assignment["models"],
                    "is_rebenchmark_container": assignment["is_rebenchmark_container"],
                    "assigned_at": time.time()
                }, f, indent=2)
            
            num_models = len(assignment["models"])
            rebench_str = " (REBENCHMARK)" if assignment["is_rebenchmark_container"] else ""
            print(f"   📤 Qual Worker {worker_id}: {num_models} model(s){rebench_str}")
        
        self._qual_dedicated_last_assigned_epoch = current_epoch
        print(f"   ✅ Assigned {sum(len(a['models']) for a in worker_assignments.values())} models to {len(worker_assignments)} workers")

    async def _collect_dedicated_qualification_results(
        self, 
        current_epoch: int, 
        force_submit: bool = False
    ) -> tuple:
        """
        Collect results from dedicated qualification workers.
        
        Called by coordinator to aggregate results and update champion status.
        NO timeout - just checks current state of workers.
        
        Args:
            current_epoch: Current epoch number
            force_submit: If True (block 335 cutoff), collect whatever is ready and
                         clear work files for incomplete workers
            
        Returns:
            Tuple of (results_list, all_workers_done)
            - results_list: List of model results from completed workers
            - all_workers_done: True if all workers have finished
        """
        from pathlib import Path
        
        weights_dir = Path("validator_weights")
        all_results = []
        
        # Check which workers have work assigned
        expected_workers = set()
        for i in range(1, QUALIFICATION_CONTAINERS_COUNT + 1):
            work_file = weights_dir / f"qual_worker_{i}_work_{current_epoch}.json"
            if work_file.exists():
                expected_workers.add(i)
        
        if not expected_workers:
            return [], True  # No workers = done
        
        # Check which workers have completed
        completed_workers = set()
        for worker_id in expected_workers:
            results_file = weights_dir / f"qual_worker_{worker_id}_results_{current_epoch}.json"
            if results_file.exists():
                try:
                    with open(results_file, 'r') as f:
                        worker_data = json.load(f)
                    all_results.extend(worker_data.get("model_results", []))
                    completed_workers.add(worker_id)
                except Exception as e:
                    print(f"   ⚠️ Error reading results from worker {worker_id}: {e}")
        
        all_done = (completed_workers == expected_workers)
        
        # If force_submit (block 335 cutoff), clear work files for incomplete workers
        # so they can process models for the next epoch
        if force_submit and not all_done:
            incomplete_workers = expected_workers - completed_workers
            print(f"   ⚠️ Block 335 cutoff - clearing {len(incomplete_workers)} incomplete worker(s): {sorted(incomplete_workers)}")
            for worker_id in incomplete_workers:
                work_file = weights_dir / f"qual_worker_{worker_id}_work_{current_epoch}.json"
                try:
                    work_file.unlink()
                    print(f"      🗑️ Cleared work file for Qual Worker {worker_id}")
                except Exception as e:
                    print(f"      ⚠️ Failed to clear work file for Qual Worker {worker_id}: {e}")
            print(f"   ℹ️ Models from incomplete workers will be re-evaluated next epoch (status stays 'submitted')")
        
        # Log collection status
        if all_done and completed_workers:
            print(f"   ✅ All {len(expected_workers)} workers complete")
            for worker_id in sorted(completed_workers):
                results_file = weights_dir / f"qual_worker_{worker_id}_results_{current_epoch}.json"
                try:
                    with open(results_file, 'r') as f:
                        worker_data = json.load(f)
                    model_count = len(worker_data.get('model_results', []))
                    print(f"      Qual Worker {worker_id}: {model_count} model(s)")
                except:
                    pass
        elif completed_workers:
            print(f"   📊 {len(completed_workers)}/{len(expected_workers)} workers complete")
            for worker_id in sorted(completed_workers):
                results_file = weights_dir / f"qual_worker_{worker_id}_results_{current_epoch}.json"
                try:
                    with open(results_file, 'r') as f:
                        worker_data = json.load(f)
                    model_count = len(worker_data.get('model_results', []))
                    print(f"      ✅ Qual Worker {worker_id}: {model_count} model(s)")
                except:
                    pass
            pending = expected_workers - completed_workers
            print(f"      ⏳ Pending: {sorted(pending)}")
        
        return all_results, all_done

    async def _process_dedicated_qualification_results(self, results: List[Dict], current_epoch: int):
        """
        Process results from dedicated qualification workers.
        
        Updates champion status and notifies gateway.
        
        Args:
            results: List of model results from all qualification workers
            current_epoch: Current epoch number
        """
        if not results:
            return
        
        print(f"\n{'='*70}")
        print(f"🏆 PROCESSING {len(results)} QUALIFICATION RESULTS")
        print(f"{'='*70}")
        
        # Read current champion
        champion_data = self._read_qualification_champion()
        current_champion_score = champion_data.get("score", 0.0) if champion_data else 0.0
        current_champion_id = champion_data.get("model_id") if champion_data else None
        
        # Process each result
        for result in results:
            model_id = result.get("model_id", "unknown")
            model_name = result.get("model_name", "Unknown")
            avg_score = result.get("avg_score", 0.0)
            is_rebenchmark = result.get("is_rebenchmark", False)
            total_cost = result.get("total_cost_usd", 0.0)
            total_time = result.get("total_time_seconds", 0.0)
            error = result.get("error")
            
            print(f"\n   📊 Model: {model_name}")
            print(f"      Score: {avg_score:.2f}")
            print(f"      Cost: ${total_cost:.4f}")
            print(f"      Time: {total_time:.1f}s")
            
            if error:
                print(f"      ❌ Error: {error}")
                continue
            
            # Minimum score required to become/remain champion
            MINIMUM_CHAMPION_SCORE = QUALIFICATION_CONFIG.MINIMUM_CHAMPION_SCORE
            
            # Determine if this model beats the champion
            became_champion = False
            was_dethroned = False
            
            if is_rebenchmark:
                # Rebenchmark - check if champion still meets minimum threshold
                print(f"      🔄 Rebenchmark - checking champion status")
                
                if avg_score < MINIMUM_CHAMPION_SCORE:
                    # Champion score dropped below minimum - DETHRONE with no replacement
                    print(f"      👎 CHAMPION DETHRONED! Score {avg_score:.2f} < minimum {MINIMUM_CHAMPION_SCORE}")
                    print(f"         There is now NO CHAMPION until a model scores >= {MINIMUM_CHAMPION_SCORE}")
                    
                    # Clear local champion
                    self._clear_qualification_champion()
                    was_dethroned = True
                    became_champion = False
                else:
                    # Champion still meets threshold - update score
                    print(f"      ✅ Champion score updated: {avg_score:.2f} (above minimum {MINIMUM_CHAMPION_SCORE})")
                    num_leads = result.get("num_leads_evaluated", 100)
                    self._update_champion_if_needed(
                        model_id=model_id,
                        model_name=model_name,
                        miner_hotkey=result.get("miner_hotkey", "unknown"),
                        score=avg_score,
                        total_cost_usd=total_cost,
                        total_time_seconds=total_time,
                        num_leads=num_leads
                    )
                    became_champion = True  # Still champion after rebenchmark
            else:
                # New challenger - check if beats champion AND meets minimum threshold
                beat_threshold = QUALIFICATION_CONFIG.CHAMPION_DETHRONING_THRESHOLD_PCT / 100.0
                threshold_score = current_champion_score * (1 + beat_threshold)
                
                if avg_score > threshold_score:
                    # Check minimum score requirement
                    if avg_score < MINIMUM_CHAMPION_SCORE:
                        print(f"      ❌ Beat champion but below minimum ({avg_score:.2f} < {MINIMUM_CHAMPION_SCORE})")
                        print(f"         Cannot become champion - need score >= {MINIMUM_CHAMPION_SCORE}")
                    else:
                        print(f"      🏆 NEW CHAMPION! Score {avg_score:.2f} > {threshold_score:.2f}")
                        num_leads = result.get("num_leads_evaluated", 100)
                        became_champion, _ = self._update_champion_if_needed(
                            model_id=model_id,
                            model_name=model_name,
                            miner_hotkey=result.get("miner_hotkey", "unknown"),
                            score=avg_score,
                            total_cost_usd=total_cost,
                            total_time_seconds=total_time,
                            num_leads=num_leads
                        )
                        current_champion_score = avg_score  # Update for next comparison
                        current_champion_id = model_id
                else:
                    print(f"      ❌ Did not beat champion ({avg_score:.2f} <= {threshold_score:.2f})")
            
            # Notify gateway
            try:
                # Get code_content if available (for displaying model code in leaderboard)
                code_content = result.get("code_content")
                
                await self._notify_gateway_champion_status(
                    model_id=model_id,
                    became_champion=became_champion,
                    score=avg_score,
                    is_rebenchmark=is_rebenchmark,
                    was_dethroned=was_dethroned,
                    evaluation_cost_usd=total_cost,
                    evaluation_time_seconds=int(total_time),
                    code_content=code_content
                )
            except Exception as e:
                print(f"      ⚠️ Failed to notify gateway: {e}")
        
        print(f"\n{'='*70}")
        print(f"✅ QUALIFICATION PROCESSING COMPLETE")
        print(f"{'='*70}\n")

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
                # Note: fetch_broadcast_requests imported at module level to avoid sandbox blocking
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
                        # fetch_miner_leads_for_request imported at module level

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
                # get_supabase_client imported at module level
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
                # record_delivery_rewards imported at module level
                record_delivery_rewards(all_delivered_leads)

                # Send leads to buyer
                print(f"✅ Sent {len(delivered_leads)} leads to buyer")

                # Add source hotkey display
                for lead in delivered_leads:
                    source_hotkey = lead.get('source', 'unknown')
                    print(f"   Lead sourced by: {source_hotkey}")   # show full hotkey

                # Save curated leads to separate file
                # save_curated_leads imported at module level
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
            # Track completed epochs IN MEMORY (not files, since coordinator deletes result files)
            # This prevents workers from trying to redo lead validation after coordinator aggregates
            self._completed_lead_validation_epochs = set()
            
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
                    
                    # CRITICAL FIX: Check if we already completed this epoch using IN-MEMORY tracking
                    # (Not file-based, because coordinator deletes result files after aggregation)
                    container_id = self.config.neuron.container_id
                    if current_epoch in self._completed_lead_validation_epochs:
                        # Lead validation already done - wait for next epoch
                        # (Qualification is now handled by dedicated qualification workers)
                        # Clear old epochs from memory to prevent unbounded growth
                        if len(self._completed_lead_validation_epochs) > 10:
                            oldest = min(self._completed_lead_validation_epochs)
                            if oldest < current_epoch - 5:
                                self._completed_lead_validation_epochs.discard(oldest)
                        print(f"⏭️  Worker {container_id}: Epoch {current_epoch} lead validation complete, waiting for next epoch...")
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
                        
                        # Too late to start validation (coordinator aggregates at block 300)
                        # Workers need ~8-10 min to process 50 leads, so cutoff at block 260
                        # gives them 40 blocks (8 min) before coordinator forces aggregation
                        if blocks_into_epoch >= 260:
                            print(f"❌ Worker: Too late to start validation (block {blocks_into_epoch}/360)")
                            print(f"   Coordinator aggregates at block 300 - not enough time to finish")
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
                            leads_file_path=leads_file_str,  # Poll file for TrueList results after Stage 0-2
                            current_epoch=current_epoch  # For epoch boundary detection mid-processing
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
                    
                    # ════════════════════════════════════════════════════════════════════
                    # EPOCH BOUNDARY CHECK: Abort if epoch changed during validation
                    # This prevents workers from writing stale results for old epochs
                    # ════════════════════════════════════════════════════════════════════
                    try:
                        post_validation_block, post_validation_epoch, _ = self._read_shared_block_file()
                        if post_validation_epoch > current_epoch:
                            print(f"\n❌ Worker {container_id}: EPOCH CHANGED during validation!")
                            print(f"   Started processing: epoch {current_epoch}")
                            print(f"   Current epoch now: {post_validation_epoch}")
                            print(f"   Aborting stale results - will start fresh on new epoch")
                            print(f"   (This prevents cascading lag from old epoch processing)\n")
                            # Don't write results, don't mark as completed
                            # Worker will re-read leads file for new epoch on next iteration
                            await asyncio.sleep(5)
                            continue  # Skip to next iteration of main loop
                    except Exception as e:
                        print(f"   ⚠️ Worker {container_id}: Could not check epoch boundary: {e}")
                        # Continue anyway - better to write potentially stale results than lose them
                    
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
                        
                        # Format for validation_results (IMMEDIATE REVEAL MODE)
                        # Include BOTH hashes AND actual values - no separate reveal phase
                        validation_results.append({
                            'lead_id': lead['lead_id'],
                            # Hash fields (for transparency log integrity)
                            'decision_hash': decision_hash,
                            'rep_score_hash': rep_score_hash,
                            'rejection_reason_hash': rejection_reason_hash,
                            'evidence_hash': evidence_hash,
                            'evidence_blob': lead.get('automated_checks_data', {}),
                            # IMMEDIATE REVEAL FIELDS - no separate reveal phase
                            'decision': decision,
                            'rep_score': rep_score,
                            'rejection_reason': rejection_reason,
                            'salt': salt.hex()
                        })
                        
                        # Format for local_validation_data (for local weight calculation)
                        # CRITICAL FIX: Include is_icp_multiplier from automated_checks_data for proper weight calc
                        local_validation_data.append({
                            'lead_id': lead['lead_id'],
                            'miner_hotkey': lead.get('miner_hotkey'),
                            'decision': decision,
                            'rep_score': rep_score,
                            'is_icp_multiplier': automated_checks_data.get("is_icp_multiplier", 0.0),
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
                    
                    # CRITICAL: Mark epoch as completed IN MEMORY before file gets deleted
                    # (Coordinator deletes result files after aggregation, so we can't rely on files)
                    self._completed_lead_validation_epochs.add(current_epoch)
                    
                    # MEMORY CLEANUP: Force garbage collection after each epoch
                    collected = gc.collect()
                    if collected > 100:
                        print(f"🧹 Worker {container_id}: Memory cleanup freed {collected} objects")
                    
                    # Wait before checking for next epoch
                    # (Qualification is handled by dedicated qualification workers, not sourcing workers)
                    await asyncio.sleep(5)
                    
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


# ════════════════════════════════════════════════════════════════════════════════
# DEDICATED QUALIFICATION WORKER
# ════════════════════════════════════════════════════════════════════════════════
# These 5 containers ONLY evaluate qualification models (not sourcing).
# They run PARALLEL to sourcing from epoch start.
# Each container handles 2 models per epoch.
# ════════════════════════════════════════════════════════════════════════════════

def run_dedicated_qualification_worker(config):
    """
    Run a dedicated qualification worker that ONLY evaluates miner models.
    
    Unlike regular workers that validate leads, qualification workers:
    1. Start at EPOCH START (parallel to sourcing)
    2. ONLY evaluate qualification models (no lead validation)
    3. Handle 2 models per epoch each
    4. Container 1 handles rebenchmark when needed
    
    No Bittensor connection, no axon, no lead validation.
    """
    import asyncio
    import json
    import time
    import base64
    from pathlib import Path
    
    qual_container_id = config.neuron.qualification_container_id
    total_qual_containers = config.neuron.total_qualification_containers
    
    # ═══════════════════════════════════════════════════════════════════════════
    # SET HTTP PROXY FOR THIS WORKER (so ddgs/free APIs go through different IPs)
    # ═══════════════════════════════════════════════════════════════════════════
    proxy_var = f"QUALIFICATION_WEBSHARE_PROXY_{qual_container_id}"
    proxy_url = os.environ.get(proxy_var)
    if proxy_url:
        os.environ["HTTP_PROXY"] = proxy_url
        os.environ["HTTPS_PROXY"] = proxy_url
        print(f"🌐 Using proxy for qualification worker {qual_container_id}")
        print(f"   Proxy: {proxy_url[:50]}...")
    else:
        print(f"⚠️ No proxy configured for qualification worker {qual_container_id}")
        print(f"   Expected env var: {proxy_var}")
        print(f"   Free APIs (ddgs) will use validator's IP directly")
    
    print("")
    print("🚀 Starting dedicated qualification worker...")
    print(f"   Qualification Container ID: {qual_container_id}")
    print(f"   Total qualification containers: {total_qual_containers}")
    print(f"   Models per container per epoch: {QUALIFICATION_MODELS_PER_CONTAINER}")
    print("")
    
    class DedicatedQualificationWorker:
        def __init__(self, config):
            self.config = config
            self.should_exit = False
            self._completed_epochs = set()  # Track completed epochs in memory
            
        def _read_shared_block_file(self):
            """Read current block from shared file (written by coordinator)"""
            block_file = Path("validator_weights") / "current_block.json"
            
            if not block_file.exists():
                raise FileNotFoundError("Coordinator hasn't written block file yet")
            
            # Check if file is stale (> 1800 seconds = 30 minutes for qualification workers)
            # NOTE: Coordinator only updates block file during batch validation phase.
            # Between epochs, the file may be 15-20 minutes old. This is NORMAL.
            # We use 30 minutes as threshold to detect truly crashed coordinators.
            file_age = time.time() - block_file.stat().st_mtime
            if file_age > 1800:
                raise Exception(f"Shared block file is stale ({int(file_age)}s old)")
            
            with open(block_file, 'r') as f:
                data = json.load(f)
                return data['block'], data['epoch'], data['blocks_into_epoch']
        
        async def process_qualification_models(self, current_epoch: int):
            """
            Process qualification models assigned to this dedicated worker.
            
            Qualification workers start at epoch start and process up to 2 models.
            """
            qual_container_id = self.config.neuron.qualification_container_id
            weights_dir = Path("validator_weights")
            
            # Check if already completed this epoch
            if current_epoch in self._completed_epochs:
                return
            
            # Look for work file specific to this qualification worker
            # Format: qual_worker_{id}_work_{epoch}.json
            work_file = weights_dir / f"qual_worker_{qual_container_id}_work_{current_epoch}.json"
            results_file = weights_dir / f"qual_worker_{qual_container_id}_results_{current_epoch}.json"
            
            if not work_file.exists():
                return  # No work assigned yet
            
            if results_file.exists():
                # Already completed
                self._completed_epochs.add(current_epoch)
                return
            
            print(f"\n{'='*70}")
            print(f"🎯 QUALIFICATION WORKER {qual_container_id}: Work found for epoch {current_epoch}")
            print(f"{'='*70}")
            
            try:
                # Read work assignment
                with open(work_file, 'r') as f:
                    work_data = json.load(f)
                
                models_to_evaluate = work_data.get("models", [])
                is_rebenchmark_container = work_data.get("is_rebenchmark_container", False)
                
                print(f"   📦 Models assigned: {len(models_to_evaluate)}")
                print(f"   🔄 Rebenchmark container: {is_rebenchmark_container}")
                
                # Import qualification modules
                try:
                    from qualification.validator.sandbox import TEESandbox
                    from gateway.qualification.models import LeadOutput, ICPPrompt, LeadScoreBreakdown
                    from qualification.scoring.lead_scorer import score_lead
                    from qualification.scoring.pre_checks import run_automatic_zero_checks
                    from gateway.qualification.config import CONFIG as QUAL_CONFIG
                except ImportError as e:
                    print(f"   ❌ Qualification module not available: {e}")
                    with open(results_file, 'w') as f:
                        json.dump({
                            "epoch": current_epoch,
                            "qual_worker_id": qual_container_id,
                            "error": f"Qualification module not available: {e}",
                            "model_results": [],
                            "timestamp": time.time()
                        }, f)
                    self._completed_epochs.add(current_epoch)
                    return
                
                # Process each model
                model_results = []
                
                for model_idx, model in enumerate(models_to_evaluate):
                    model_name = model.get("model_name", "Unknown")
                    model_id = model.get("model_id", "unknown")
                    miner_hotkey = model.get("miner_hotkey", "unknown")
                    model_code_b64 = model.get("agent_code", "")
                    runs = model.get("evaluation_runs", [])
                    is_rebenchmark = model.get("is_rebenchmark", False)
                    
                    print(f"\n   📋 [{model_idx + 1}/{len(models_to_evaluate)}] Model: {model_name}")
                    print(f"      Miner: {miner_hotkey[:16]}...")
                    print(f"      ICPs: {len(runs)}")
                    print(f"      Rebenchmark: {is_rebenchmark}")
                    
                    # Decode model code
                    model_code = base64.b64decode(model_code_b64) if model_code_b64 else b""
                    
                    # ═══════════════════════════════════════════════════════════════
                    # EXTRACT CODE CONTENT: For displaying model code in leaderboard
                    # ═══════════════════════════════════════════════════════════════
                    code_content = None
                    if model_code:
                        try:
                            import tarfile
                            import io
                            
                            code_files = {}
                            allowed_extensions = {'.py', '.txt', '.md', '.json', '.yaml', '.yml', '.toml'}
                            with tarfile.open(fileobj=io.BytesIO(model_code), mode='r:gz') as tar:
                                for member in tar.getmembers():
                                    if not member.isfile():
                                        continue
                                    
                                    filename = member.name
                                    if '/' in filename:
                                        filename = filename.split('/', 1)[1] if filename.count('/') == 1 else filename
                                    
                                    ext = '.' + filename.split('.')[-1].lower() if '.' in filename else ''
                                    if ext not in allowed_extensions:
                                        continue
                                    
                                    if filename.startswith('.') or '/__' in filename:
                                        continue
                                    
                                    try:
                                        f = tar.extractfile(member)
                                        if f:
                                            content = f.read().decode('utf-8', errors='replace')
                                            code_files[filename] = content
                                            total_size += member.size
                                    except Exception:
                                        continue
                            
                            if code_files:
                                code_content = json.dumps(code_files)
                                print(f"      📄 Extracted {len(code_files)} code files for display")
                        except Exception as extract_err:
                            print(f"      ⚠️ Could not extract code content: {extract_err}")
                    
                    # ═══════════════════════════════════════════════════════════════
                    # HARDCODING DETECTION: Run on ALL evaluations (including rebenchmarks)
                    # Note: Catches models that became champion before detection was added
                    # ═══════════════════════════════════════════════════════════════
                    if model_code:
                        try:
                            from qualification.validator.hardcoding_detector import (
                                analyze_model_for_hardcoding,
                                is_detection_enabled
                            )
                            
                            if is_detection_enabled():
                                print(f"\n      🔍 HARDCODING DETECTION: Analyzing model code...")
                                
                                # Get ICP samples
                                icp_samples = [run.get("icp_data", {}) for run in runs[:5]]
                                
                                detection_result = await analyze_model_for_hardcoding(
                                    model_code=model_code,
                                    icp_samples=icp_samples
                                )
                                
                                confidence = detection_result.get("confidence_hardcoded", 0)
                                verdict = detection_result.get("verdict", "UNKNOWN")
                                
                                print(f"      🔍 Hardcoding detection result:")
                                print(f"         Verdict: {verdict}")
                                print(f"         Confidence: {confidence}%")
                                
                                if verdict == "HARDCODED" and confidence >= 70:
                                    print(f"      ❌ MODEL REJECTED: Hardcoding detected ({confidence}% confidence)")
                                    model_results.append({
                                        "model_id": model_id,
                                        "model_name": model_name,
                                        "miner_hotkey": miner_hotkey,
                                        "error": f"Hardcoding detected ({confidence}% confidence)",
                                        "rejection_reason": "hardcoding_detected",
                                        "avg_score": 0.0,
                                        "total_cost_usd": 0.0,
                                        "total_time_seconds": 0.0,
                                        "is_rebenchmark": is_rebenchmark
                                    })
                                    continue  # Skip to next model
                        except Exception as hc_err:
                            print(f"      ⚠️ Hardcoding detection error: {hc_err}")
                    
                    # ═══════════════════════════════════════════════════════════════
                    # RUN MODEL EVALUATION (matches OLD LightweightWorker pattern)
                    # ═══════════════════════════════════════════════════════════════
                    total_score = 0.0
                    leads_scored = 0
                    total_cost = 0.0
                    total_time = 0.0
                    per_icp_results = []
                    seen_companies = set()
                    MAX_TOTAL_COST = QUAL_CONFIG.MAX_COST_PER_EVALUATION_USD
                    evaluation_stopped_early = False
                    
                    # Get gateway URL and create proxy URL (CRITICAL for cost tracking!)
                    gateway_url = os.environ.get("GATEWAY_URL", "http://54.226.209.164:8000")
                    api_proxy_url = f"{gateway_url}/qualification/proxy"
                    
                    # Initialize ONE sandbox per model (NOT per ICP!)
                    sandbox = None
                    try:
                        sandbox = TEESandbox(
                            model_code=model_code,
                            evaluation_run_id=runs[0]["evaluation_run_id"] if runs else None,
                            api_proxy_url=api_proxy_url,
                            evaluation_id=model.get("evaluation_id")
                        )
                        await sandbox.start()
                        
                        # Process each ICP with the SAME sandbox
                        for run_idx, run in enumerate(runs, 1):
                            # $5 cost limit check
                            if total_cost >= MAX_TOTAL_COST:
                                print(f"      🛑 $5 HARD STOP at ICP {run_idx}/{len(runs)}")
                                evaluation_stopped_early = True
                                break
                            
                            evaluation_run_id = run.get("evaluation_run_id")
                            icp_data = run.get("icp_data", {})
                            icp_industry = icp_data.get("industry", "Unknown")
                            
                            print(f"\n      📋 ICP {run_idx}/{len(runs)}: {icp_industry}")
                            
                            try:
                                # Create ICP prompt
                                icp = ICPPrompt(**icp_data)
                                
                                # Run model with timeout (asyncio imported at top of file)
                                start_time = time.time()
                                result = await asyncio.wait_for(
                                    sandbox.run_model(icp),
                                    timeout=QUAL_CONFIG.RUNNING_MODEL_TIMEOUT_SECONDS
                                )
                                run_time = time.time() - start_time
                                
                                # Get cost from sandbox
                                run_cost = sandbox.get_run_cost() if hasattr(sandbox, 'get_run_cost') else 0.01
                                total_cost += run_cost
                                
                                # Parse result
                                lead_data = result.get("lead") if isinstance(result, dict) else None
                                error_msg = result.get("error") if isinstance(result, dict) else None
                                lead = LeadOutput(**lead_data) if lead_data else None
                                
                                # Score lead
                                if lead:
                                    scores = await score_lead(
                                        lead=lead,
                                        icp=icp,
                                        run_cost_usd=run_cost,
                                        run_time_seconds=run_time,
                                        seen_companies=seen_companies
                                    )
                                    score = scores.final_score
                                else:
                                    failure_reason = error_msg if error_msg else "No lead returned"
                                    scores = LeadScoreBreakdown(
                                        icp_fit=0, decision_maker=0, intent_signal_raw=0,
                                        time_decay_multiplier=1.0, intent_signal_final=0,
                                        cost_penalty=0, time_penalty=0, final_score=0,
                                        failure_reason=failure_reason
                                    )
                                    score = 0.0
                                
                                # Accumulate
                                total_score += score
                                total_time += run_time
                                leads_scored += 1
                                
                                # Store per-ICP result
                                per_icp_results.append({
                                    "evaluation_run_id": evaluation_run_id,
                                    "lead_returned": lead.model_dump() if lead else None,
                                    "scores": scores.model_dump() if scores else None,
                                    "run_cost_usd": run_cost,
                                    "run_time_seconds": run_time
                                })
                                
                                if lead:
                                    print(f"         ✅ {lead.role} @ {lead.business} (Score: {score:.2f})")
                                else:
                                    print(f"         ❌ {scores.failure_reason}")
                                
                            except asyncio.TimeoutError:
                                print(f"         ⚠️ TIMEOUT")
                                total_time += QUAL_CONFIG.RUNNING_MODEL_TIMEOUT_SECONDS
                                per_icp_results.append({
                                    "evaluation_run_id": evaluation_run_id,
                                    "lead_returned": None,
                                    "scores": {"final_score": 0, "failure_reason": "timeout"},
                                    "run_cost_usd": 0,
                                    "run_time_seconds": QUAL_CONFIG.RUNNING_MODEL_TIMEOUT_SECONDS
                                })
                            except Exception as e:
                                print(f"         ❌ ERROR: {e}")
                                run_time_val = time.time() - start_time if 'start_time' in dir() else 0
                                total_time += run_time_val
                                per_icp_results.append({
                                    "evaluation_run_id": evaluation_run_id,
                                    "lead_returned": None,
                                    "scores": {"final_score": 0, "failure_reason": str(e)[:200]},
                                    "run_cost_usd": run_cost if 'run_cost' in dir() else 0,
                                    "run_time_seconds": run_time_val
                                })
                    
                    finally:
                        # Clean up sandbox
                        if sandbox:
                            try:
                                await sandbox.cleanup()
                            except Exception:
                                pass
                    
                    avg_score = total_score / leads_scored if leads_scored > 0 else 0.0
                    
                    print(f"\n      ✅ Model complete!")
                    print(f"         Avg Score: {avg_score:.2f}")
                    print(f"         Total Time: {total_time:.1f}s")
                    print(f"         Total Cost: ${total_cost:.4f}")
                    
                    model_results.append({
                        "model_id": model_id,
                        "model_name": model_name,
                        "miner_hotkey": miner_hotkey,
                        "avg_score": avg_score,
                        "total_score": total_score,
                        "leads_scored": leads_scored,
                        "total_cost_usd": total_cost,
                        "total_time_seconds": total_time,
                        "is_rebenchmark": is_rebenchmark,
                        "per_icp_results": per_icp_results,
                        "code_content": code_content  # For displaying model code in leaderboard
                    })
                
                # Write all results
                with open(results_file, 'w') as f:
                    json.dump({
                        "epoch": current_epoch,
                        "qual_worker_id": qual_container_id,
                        "is_rebenchmark_container": is_rebenchmark_container,
                        "model_results": model_results,
                        "timestamp": time.time()
                    }, f, indent=2)
                
                self._completed_epochs.add(current_epoch)
                print(f"\n{'='*70}")
                print(f"✅ QUALIFICATION WORKER {qual_container_id}: Completed {len(model_results)} models")
                print(f"{'='*70}\n")
                
            except Exception as e:
                print(f"❌ Qualification worker error: {e}")
                import traceback
                traceback.print_exc()
                # Write error result
                with open(results_file, 'w') as f:
                    json.dump({
                        "epoch": current_epoch,
                        "qual_worker_id": qual_container_id,
                        "error": str(e),
                        "model_results": [],
                        "timestamp": time.time()
                    }, f)
                self._completed_epochs.add(current_epoch)
        
        async def run_loop(self):
            """Main loop for dedicated qualification worker."""
            print("🔄 Qualification worker starting main loop...")
            print("   (Waiting for coordinator to assign work)")
            
            last_epoch = -1
            
            while not self.should_exit:
                try:
                    # Read current epoch from shared block file
                    try:
                        current_block, current_epoch, blocks_into_epoch = self._read_shared_block_file()
                    except FileNotFoundError:
                        print("   ⏳ Waiting for coordinator to write block file...")
                        await asyncio.sleep(10)
                        continue
                    except Exception as e:
                        print(f"   ⚠️ Block file error: {e}")
                        await asyncio.sleep(10)
                        continue
                    
                    # New epoch - check for work
                    if current_epoch != last_epoch:
                        print(f"\n📅 Epoch {current_epoch} (block {blocks_into_epoch}/360)")
                        last_epoch = current_epoch
                    
                    # Process qualification models
                    await self.process_qualification_models(current_epoch)
                    
                    # Sleep briefly
                    await asyncio.sleep(5)
                    
                except Exception as e:
                    print(f"❌ Qualification worker loop error: {e}")
                    import traceback
                    traceback.print_exc()
                    await asyncio.sleep(30)
    
    # Create and run qualification worker
    worker = DedicatedQualificationWorker(config)
    
    try:
        asyncio.run(worker.run_loop())
    except KeyboardInterrupt:
        print("\n🛑 Qualification worker shutting down...")
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
    parser.add_argument("--mode", type=str, choices=["coordinator", "worker", "qualification_worker"], help="Container mode: 'coordinator' waits for workers and submits to gateway, 'worker' validates leads, 'qualification_worker' evaluates miner models")
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
    # QUALIFICATION WORKER MODE: Dedicated model evaluation containers
    # ════════════════════════════════════════════════════════════════════════════
    # Qualification workers don't need:
    # - Bittensor wallet/subtensor/metagraph (no chain connection)
    # - Axon serving (no API endpoints)
    # - Lead validation (handled by regular workers)
    # - Weight setting (only coordinator submits weights)
    # 
    # Qualification workers ONLY need:
    # - Read current_block.json (for epoch timing)
    # - Read qual_worker_{id}_work_{epoch}.json (model assignments)
    # - Evaluate miner models via TEE sandbox
    # - Write results to qual_worker_{id}_results_{epoch}.json
    # ════════════════════════════════════════════════════════════════════════════
    if getattr(args, 'mode', None) == "qualification_worker":
        print("════════════════════════════════════════════════════════════════")
        print("🎯 DEDICATED QUALIFICATION WORKER MODE")
        print("════════════════════════════════════════════════════════════════")
        print("   Skipping heavy initialization:")
        print("   ✗ Bittensor wallet/subtensor/metagraph")
        print("   ✗ Axon serving")
        print("   ✗ Lead validation")
        print("   ✗ Weight setting")
        print("")
        print("   Qualification worker responsibilities:")
        print("   ✓ Read current_block.json for epoch timing")
        print("   ✓ Read qual_worker_N_work_EPOCH.json (model assignments)")
        print("   ✓ Evaluate miner models via TEE sandbox")
        print(f"   ✓ Process up to {QUALIFICATION_MODELS_PER_CONTAINER} models per epoch")
        print("   ✓ Write results to qual_worker_N_results_EPOCH.json")
        print("════════════════════════════════════════════════════════════════")
        print("")
        
        # Create minimal config for qualification worker
        config = bt.Config()
        config.neuron = bt.Config()
        config.neuron.qualification_container_id = getattr(args, 'container_id', 1)
        config.neuron.total_qualification_containers = QUALIFICATION_CONTAINERS_COUNT
        config.neuron.mode = "qualification_worker"
        
        # Run dedicated qualification worker loop
        run_dedicated_qualification_worker(config)
        return  # Exit early - don't initialize full validator

    # ════════════════════════════════════════════════════════════════════════════
    # COORDINATOR MODE: Full initialization
    # ════════════════════════════════════════════════════════════════════════════
    # start_epoch_monitor imported at module level

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
            # stop_epoch_monitor imported at module level
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
