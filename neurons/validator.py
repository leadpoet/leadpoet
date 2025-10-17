#!/usr/bin/env python3
# Suppress multiprocessing warnings BEFORE any imports
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
from validator_models.automated_checks import validate_lead_list as auto_check_leads, run_automated_checks
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
    push_validator_weights,
    push_validator_ranking,
)
from Leadpoet.utils.token_manager import TokenManager
from Leadpoet.utils.lead_utils import (
    get_email,
    get_website,
    get_company,
    get_industry,
    get_role,
    get_sub_industry
)
from supabase import Client
import socket
from math import isclose
from pathlib import Path
import warnings

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
echo "   GitHub: github.com/leadpoet/Leadpoet"
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
        
        # Metagraph sync handled server-side
        
        try:
            self.token_manager = TokenManager(
                hotkey=self.wallet.hotkey.ss58_address,
                wallet=self.wallet
            )
            bt.logging.info("🔑 TokenManager initialized")
        except Exception as e:
            bt.logging.error(f"Failed to initialize TokenManager: {e}")
            raise
        
        status = self.token_manager.get_status()
        
        if status.get('valid'):
            bt.logging.info(f"✅ Token valid - Role: {status['role']}, Hours remaining: {status.get('hours_remaining', 0):.1f}")
        else:
            bt.logging.warning("⚠️ Token invalid or missing - will attempt refresh")
        
        status = self.token_manager.get_status()
        if status.get('needs_refresh') or not status.get('valid'):
            success = self.token_manager.refresh_token()
            if success:
                bt.logging.info("✅ Token refreshed successfully")
            else:
                bt.logging.error("❌ Failed to refresh token")
        else:
            bt.logging.info("✅ Using existing valid token")
        
        self.supabase_url = "https://qplwoislplkcegvdmbim.supabase.co"
        self.supabase_client: Optional[Client] = None
        self._init_supabase_client()
    
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
        emails = [lead.get('Owner(s) Email', '') for lead in leads]
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
            {"Business": f"Test Business {i}", "Owner(s) Email": f"owner{i}@testleadpoet.com", "Website": f"https://business{i}.com", "Industry": "Tech & AI"}
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
                business = lead.get('Business', lead.get('business', 'Unknown'))
                score = lead.get('intent_score', 0)
                print(f"  {i}. {business} (score={score:.3f})")

            # Add c_validator_hotkey to leads being sent to client via Bittensor
            for lead in top_leads:
                lead["c_validator_hotkey"] = self.wallet.hotkey.ss58_address

            synapse.leads = top_leads

            # V2: After Final Curated List is frozen, call reward calculation
            if top_leads:
                try:
                    from Leadpoet.validator.reward import calculate_weights, record_event

                    # Record events for each lead in the Final Curated List
                    for lead in top_leads:
                        if lead.get("source") and lead.get("curated_by"):
                          
                            record_event(lead)

                    # STEP 4: CALCULATE WEIGHTS WITH CRYPTOGRAPHIC PROOF
                    # Pass validator's wallet to prove hotkey ownership
                    rewards = calculate_weights(
                        total_emission=100.0,
                        validator_wallet=self.wallet  # Proves we control this hotkey
                    )
                    
                    # Check if we were eligible
                    if "error" in rewards:
                        print("\n❌ VALIDATOR NOT ELIGIBLE FOR WEIGHTS:")
                        print(f"   Reason: {rewards['error']}")
                        print(f"   Validated: {rewards.get('validated_count', 0)}/{rewards.get('total_count', 0)} leads")
                        print(f"   Percentage: {rewards.get('percentage', 0):.1f}% (need >= 10.0%)")
                        print("   No weights will be set this epoch - increase validation activity!")
                    else:
                        # Validator is eligible - set weights on-chain
                        # Log final weights and emissions
                        print("\n🎯 V2 REWARD CALCULATION COMPLETE:")
                        print(f"   ✅ Validator eligible - validated {rewards.get('percentage', 0):.1f}% of epoch leads")
                        print(f"   Final Curated List: {len(top_leads)} prospects")
                        print(f"   Final weights (W): {rewards['W']}")
                        print(f"   Emissions: {rewards['E']}")

                        weights_dict = rewards["W"]                   # miner-hotkey ➜ share
                        # publish weights on-chain
                        try:
                            # map hotkeys → uids present in current metagraph
                            uids, weights = [], []
                            for hk, w in weights_dict.items():
                                if hk in self.metagraph.hotkeys and w > 0:
                                    uids.append(self.metagraph.hotkeys.index(hk))
                                    weights.append(float(w))

                            # normalise to 1.0 as required by bittensor
                            s = sum(weights)
                            if not isclose(s, 1.0) and s > 0:
                                weights = [w / s for w in weights]

                            self.subtensor.set_weights(
                                wallet             = self.wallet,
                                netuid             = self.config.netuid,
                                uids               = uids,
                                weights            = weights,
                                wait_for_inclusion = True,     # non-blocking
                            )
                            print(self.wallet, self.config.netuid, uids, weights)
                            print("✅ Published new weights on-chain")
                        except Exception as e:
                            print(f"⚠️  Failed to publish weights on-chain: {e}")

                        # Firestore write
                        push_validator_weights(self.wallet, self.uid, weights_dict)

                except Exception as e:
                    print(f"⚠️  V2 reward calculation failed: {e}")
            else:
                print("⚠️  No prospects in Final Curated List - skipping reward calculation")
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

                    # V2: Record events for reward calculation before adding to pool
                    try:
                        from Leadpoet.validator.reward import record_event
                        for lead in response.leads:
                            if lead.get("source") and lead.get("curated_by") and lead.get("conversion_score"):
                                record_event(lead)
                                print(f"🎯 V2: Recorded event for lead {lead.get('owner_email', 'unknown')} "
                                      f"(source: {lead['source']}, curator: {lead['curated_by']})")
                    except Exception as e:
                        print(f"⚠️  V2: Failed to record events: {e}")

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
        state_path = os.path.join(self.config.neuron.full_path or os.getcwd(), "validator_state.npz")
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

    def load_state(self):
        state_path = os.path.join(self.config.neuron.full_path or os.getcwd(), "validator_state.npz")
        if os.path.exists(state_path):
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
            except Exception as e:
                bt.logging.warning(f"Failed to load state: {e}. Using defaults.")
                self._initialize_default_state()
        else:
            bt.logging.info("No state file found. Initializing with defaults.")
            self._initialize_default_state()

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

        try:
            # Keep the validator running and continuously process leads
            while not self.should_exit:
        # AUTOMATIC WEIGHT CALCULATION NEAR EPOCH END
                # Check if we're near the end of the epoch (blocks 355-360)
                try:
                    from Leadpoet.validator.reward import _get_epoch_status
                    
                    # Check every 5 iterations (approx every 5 seconds) for faster response
                    if not hasattr(self, '_epoch_check_counter'):
                        self._epoch_check_counter = 0  
                        self._last_weight_calc_block = 0
                    
                    self._epoch_check_counter += 1
                    if self._epoch_check_counter >= 2:  # Check every 2 iterations
                        self._epoch_check_counter = 0
                        
                        status = _get_epoch_status()
                        blocks_remaining = status.get("blocks_remaining", 360)
                        current_epoch = status.get("current_epoch", 0)
                        current_block = 360 - blocks_remaining
                        
                        # Reset block counter when entering a new epoch
                        if hasattr(self, '_last_tracked_epoch') and self._last_tracked_epoch != current_epoch:
                            self._last_weight_calc_block = 0
                        self._last_tracked_epoch = current_epoch
                        
                        # Debug output to see current status
                        print(f"⏰ Epoch check: Block {current_block}/360, Blocks remaining: {blocks_remaining}, Epoch: {current_epoch}")
                        
                        # TESTING: Trigger between blocks 90-120 (blocks_remaining 270-240)
                        # Normal: blocks_remaining <= 5 (blocks 355-360)
                        if blocks_remaining <= 350 and blocks_remaining >= 1:  # Blocks 90-120 for testing
                            # Check if 2 blocks have passed since last attempt (or first attempt)
                            current_block = 360 - blocks_remaining
                            should_attempt = (current_block - self._last_weight_calc_block >= 2 or self._last_weight_calc_block == 0)
                            
                            if should_attempt:
                                print("\n🎯 AUTOMATIC WEIGHT CALCULATION TRIGGERED")
                                print(f"   Epoch: {current_epoch}")
                                print(f"   Block: {360 - blocks_remaining}/360")
                                print("   Requesting weight calculation...")
                                
                                # Mark that we attempted at this block
                                self._last_weight_calc_block = current_block
                                
                                # Get JWT token and call Edge Function for weights
                                import requests
                                jwt_token = self.token_manager.get_token()
                                
                                if not jwt_token:
                                    print("❌ No JWT token available - cannot get weights")
                                    continue
                                
                                # Call Edge Function to get weights (server-side eligibility check)
                                try:
                                    print("   Calling Edge Function with JWT token...")
                                    response = requests.get(
                                        "https://qplwoislplkcegvdmbim.supabase.co/functions/v1/get-validator-weights",
                                        headers={"Authorization": f"Bearer {jwt_token}"},
                                        timeout=30
                                    )
                                    
                                    print(f"   Response status: {response.status_code}")
                                    
                                    # Try to parse JSON response
                                    try:
                                        weights_result = response.json()
                                    except json.JSONDecodeError:
                                        print("❌ Invalid JSON response from Edge Function")
                                        print(f"   Response text: {response.text[:500]}")
                                        continue
                                    
                                    # Edge Function returns 403 if not eligible
                                    if response.status_code == 403:
                                        print("❌ VALIDATOR NOT ELIGIBLE")
                                        print(f"   Reason: {weights_result.get('error', 'Unknown')}")
                                        print(f"   Validated: {weights_result.get('validated_count', 0)}/{weights_result.get('total_count', 0)} leads")
                                        print(f"   Percentage: {weights_result.get('percentage', 0):.1f}% (need >= 10.0%)")
                                        # Don't mark as attempted - allow retry every 2 blocks
                                        # self._last_epoch_weights_set = current_epoch  # Mark as attempted
                                        continue
                                    
                                    if response.status_code == 404:
                                        print("❌ Edge Function not found - it may not be deployed yet")
                                        print("   Deploy with: supabase functions deploy get-validator-weights")
                                        continue
                                    
                                    if response.status_code != 200:
                                        print(f"❌ Error from Edge Function (status {response.status_code})")
                                        print(f"   Error: {weights_result.get('error', 'Unknown error')}")
                                        if 'details' in weights_result:
                                            print(f"   Details: {weights_result['details']}")
                                        # For 500 errors, also show the full response for debugging
                                        if response.status_code == 500:
                                            print(f"   Full response: {weights_result}")
                                        continue
                                        
                                except requests.exceptions.RequestException as e:
                                    print(f"❌ Network error calling Edge Function: {e}")
                                    continue
                                except Exception as e:
                                    print(f"❌ Unexpected error calling Edge Function: {e}")
                                    import traceback
                                    traceback.print_exc()
                                    continue
                                
                                # Check if validator is eligible (Edge Function already enforced this)
                                # Note: weights can be empty dict if no leads accepted yet
                                if weights_result.get("eligible") and weights_result.get("weights") is not None:
                                    print("✅ VALIDATOR ELIGIBLE - Setting weights on chain")
                                    print(f"   Consensus participation: {weights_result.get('validated_count', 0)}/{weights_result.get('total_count', 0)} leads")
                                    print(f"   Percentage: {weights_result.get('percentage', 0):.1f}%")
                                    print(f"   Total leads in epoch: {weights_result.get('total_leads', 0)}")
                                    
                                    # Convert weights dict to format expected by set_weights
                                    # Edge Function returns weights directly
                                    weight_dict = weights_result["weights"]
                                    
                                    # Check if there are any weights to set
                                    if not weight_dict:
                                        print("   ℹ️ No leads accepted yet in epoch - no weights to set")
                                        print("   (Validator is eligible but waiting for leads to be accepted)")
                                        continue
                                    
                                    # Get UIDs for the hotkeys
                                    uids_and_weights = []
                                    for hotkey, weight in weight_dict.items():
                                        # Find UID for this hotkey
                                        try:
                                            uid = self.metagraph.hotkeys.index(hotkey)
                                            uids_and_weights.append((uid, weight))
                                            print(f"     Miner {hotkey[:10]}... (UID {uid}): {weight:.4f}")
                                        except ValueError:
                                            print(f"     ⚠️ Hotkey {hotkey[:10]}... not found in metagraph")
                                    
                                    if uids_and_weights:
                                        # Set weights on chain
                                        try:
                                            success = self.subtensor.set_weights(
                                                netuid=self.config.netuid,
                                                wallet=self.wallet,
                                                uids=[uid for uid, _ in uids_and_weights],
                                                weights=[weight for _, weight in uids_and_weights],
                                                wait_for_inclusion=False,
                                                wait_for_finalization=False
                                            )
                                            
                                            if success:
                                                print(f"✅ Weights successfully set on chain for epoch {current_epoch}")
                                                # Don't mark epoch as done - allow continuous updates every 2 blocks
                                                # self._last_epoch_weights_set = current_epoch
                                            else:
                                                print("❌ Failed to set weights on chain")
                                        except Exception as e:
                                            print(f"❌ Error setting weights on chain: {e}")
                                    else:
                                        print("❌ No valid UIDs found for weight setting")
                                        
                                else:
                                    # Debug: Show what we actually got from Edge Function
                                    print("⚠️ No weights returned from Edge Function")
                                    print(f"   Debug - Eligible: {weights_result.get('eligible', 'missing')}")
                                    print(f"   Debug - Weights: {weights_result.get('weights', 'missing')}")
                                    print(f"   Debug - Validated: {weights_result.get('validated_count', 0)}/{weights_result.get('total_count', 0)}")
                                    print(f"   Debug - Percentage: {weights_result.get('percentage', 0):.1f}%")
                                    # Don't mark as attempted - allow retry every 2 blocks
                                    # self._last_epoch_weights_set = current_epoch  # Don't mark as attempted
                                    
                except Exception as e:
                    bt.logging.warning(f"Error in automatic weight calculation: {e}")

                # Check and refresh token every iteration
                token_refreshed = self.token_manager.refresh_if_needed(threshold_hours=1)
                if not token_refreshed and not self.token_manager.get_token():
                    bt.logging.warning("⚠️ Token refresh failed, continuing with existing token...")
                
                # Refresh Supabase client if token was refreshed
                if token_refreshed:
                    bt.logging.info("🔄 Token was refreshed, reinitializing Supabase client...")
                    self._init_supabase_client()
                
                # Process any new leads that need validation (continuous)
                try:
                    self.process_sourced_leads_continuous()
                except Exception as e:
                    bt.logging.warning(f"Error in process_sourced_leads_continuous: {e}")
                    time.sleep(5)  # Wait before retrying
                
                try:
                    self.process_curation_requests_continuous()
                except Exception as e:
                    bt.logging.warning(f"Error in process_curation_requests_continuous: {e}")
                    time.sleep(5)  # Wait before retrying

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

                time.sleep(1)  # Small delay to prevent tight loop
        except KeyboardInterrupt:
            self.axon.stop()
            bt.logging.success("Validator killed by keyboard interrupt.")
            exit()
        except Exception as e:
            bt.logging.error(f"Critical error in validator main loop: {e}")
            import traceback
            bt.logging.error(traceback.format_exc())
            # Continue running instead of crashing
            time.sleep(10)  # Wait longer before retrying main loop

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

            print(f"📊 Found {len(available_miners)} registered miners:")
            for miner in available_miners:
                print(f"   UID {miner['uid']}: {miner['hotkey'][:10]}... (stake: {miner['stake']:.2f})")

            print(f"\n🔍 Found {len(running_miners)} currently running miners:")
            for miner in running_miners:
                print(f"   UID {miner['uid']}: {miner['hotkey'][:10]}... (IP: {miner['ip']}:{miner['port']})")

            if not available_miners:
                print("   ⚠️  No miners found on the network")
            elif not running_miners:
                print("   ⚠️  No miners currently running")

        except Exception as e:
            print(f"❌ Error discovering miners: {e}")

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
                    business_name = lead.get('business', lead.get('website', 'Unknown'))
                    email = lead.get('owner_email', lead.get('email', '?'))
                    
                    print(f"\n🟣 Validating prospect {prospect_id[:8]}...")
                    print(f"   Lead ID: {lead_id[:8]}...")
                    print(f"   Business: {business_name}")
                    print(f"   Email: {email}")
                    print(f"   Miner: {miner_hotkey[:10] if miner_hotkey and miner_hotkey != 'unknown' else 'unknown'}...")
                    
                    # Run async validate_lead in sync context
                    result = asyncio.run(self.validate_lead(lead))
                    
                    # Extract validation results
                    is_valid = result.get("is_legitimate", False)
                    score = result.get("score", 0.0)
                    reason = result.get("reason", "Unknown")
                    
                    # Log validation result
                    if is_valid:
                        print(f"   ✅ Valid (score: {score:.2f})")
                    else:
                        print(f"   ❌ Invalid: {reason} (score: {score:.2f})")
                    
                    # Submit validation assessment to consensus system
                    submission_success = submit_validation_assessment(
                        wallet=self.wallet,
                        prospect_id=prospect_id,
                        lead_id=lead_id,
                        lead_data=lead,
                        score=score,
                        is_valid=is_valid,
                        network=self.config.subtensor.network,
                        netuid=self.config.netuid
                    )
                    
                    if submission_success:
                        print("   📤 Assessment submitted to consensus system")
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
            
            print(f"\n✅ Processed {len(prospects_batch)} prospects in consensus mode")
            
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
                                business = lead.get('Business', lead.get('business', 'Unknown'))[:30]
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

                        # PUBLISH WEIGHTS for miners who provided leads
                        try:
                            from Leadpoet.validator.reward import calculate_weights, record_event

                            # Record events for each lead in the ranked list
                            for lead in top_leads:
                                if lead.get("source") and lead.get("curated_by"):
                                    record_event(lead)

                            # STEP 4: CALCULATE WEIGHTS WITH CRYPTOGRAPHIC PROOF
                            # Pass validator's wallet to prove hotkey ownership
                            rewards = calculate_weights(
                                total_emission=100.0,
                                validator_wallet=self.wallet  # Proves we control this hotkey
                            )
                            
                            # Check if we were eligible
                            if "error" in rewards:
                                print("\n❌ VALIDATOR NOT ELIGIBLE FOR WEIGHTS:")
                                print(f"   Reason: {rewards['error']}")
                                print(f"   Validated: {rewards.get('validated_count', 0)}/{rewards.get('total_count', 0)} leads")
                                print(f"   Percentage: {rewards.get('percentage', 0):.1f}% (need >= 10.0%)")
                                print("   No weights will be set this epoch - increase validation activity!")
                            else:
                                # Validator is eligible - set weights on-chain
                                # Log final weights
                                print("\n🎯 V2 REWARD CALCULATION COMPLETE:")
                                print(f"   ✅ Validator eligible - validated {rewards.get('percentage', 0):.1f}% of epoch leads")
                                print(f"   Ranked leads: {len(top_leads)} prospects")
                                print(f"   Final weights (W): {rewards['W']}")
                                print(f"   Emissions: {rewards['E']}")

                                weights_dict = rewards["W"]

                                # Publish weights on-chain
                                try:
                                    from math import isclose
                                    uids, weights = [], []
                                    for hk, w in weights_dict.items():
                                        if hk in self.metagraph.hotkeys and w > 0:
                                            uids.append(self.metagraph.hotkeys.index(hk))
                                            weights.append(float(w))

                                    # Normalize to 1.0
                                    s = sum(weights)
                                    if not isclose(s, 1.0) and s > 0:
                                        weights = [w / s for w in weights]

                                    self.subtensor.set_weights(
                                        wallet=self.wallet,
                                        netuid=self.config.netuid,
                                        uids=uids,
                                        weights=weights,
                                        wait_for_inclusion=True,
                                    )
                                    print("✅ Published weights on-chain")
                                except Exception as e:
                                    print(f"⚠️  Failed to publish weights on-chain: {e}")

                                # Store in Firestore
                                from Leadpoet.utils.cloud_db import push_validator_weights
                                push_validator_weights(self.wallet, self.uid, weights_dict)

                        except Exception as e:
                            print(f"⚠️  V2 reward calculation failed: {e}")

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

        # Include ZeroBounce AI score if present
        if "email_score" in lead:
            lead["email_score"] = lead["email_score"]

        try:
            # Save to Supabase (write-only, no duplicate checking)
            if not self.supabase_client:
                bt.logging.error("❌ Supabase client not available - cannot save validated lead")
                return
                
            success = self.save_validated_lead_to_supabase(lead)
            email = lead.get("owner_email", lead.get("email", "?"))
            biz = lead.get("business", lead.get("website", ""))
            
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

    async def validate_lead(self, lead):
        """Validate a single lead using automated_checks. Returns pass/fail."""
        try:
            # Check for required email field first
            email = lead.get('owner_email', lead.get('email', ''))
            if not email:
                return {'is_legitimate': False,
                        'reason': 'Missing email',
                        'score': 0.0}
            
            # Map your field names to what automated_checks expects
            mapped_lead = {
                "email": email,  # Map to "email" field
                "Email 1": email,  # Also map to "Email 1" as backup
                "Company": lead.get('business', lead.get('website', '')),  # Map business -> Company
                "Website": lead.get('website', lead.get('business', '')),  # Map to Website
                "website": lead.get('website', lead.get('business', '')),  # Also lowercase
                "First Name": lead.get('first', ''),
                "Last Name": lead.get('last', ''),
                # Include any other fields that might be useful
                **lead  # Include all original fields too
            }
            
            # Use automated_checks for comprehensive validation
            passed, reason = await run_automated_checks(mapped_lead)

            # grab ZeroBounce AI score (set in check_zerobounce_email)
            email_score = mapped_lead.get("email_score", None)
            if email_score is not None:
                lead["email_score"] = email_score      # propagate to original lead

            return {
                'is_legitimate': passed,
                'reason': reason,
                'score': 1.0 if passed else 0.0
            }
            
        except Exception as e:
            bt.logging.error(f"Error in validate_lead: {e}")
            return {'is_legitimate': False,
                    'reason': f'Validation error: {e}',
                    'score': 0.0}

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
                "email": lead.get("owner_email", lead.get("email", "")),
                "company": lead.get("business", lead.get("company", "")),
                "validated_at": datetime.now(timezone.utc).isoformat(),
                "validator_hotkey": self.wallet.hotkey.ss58_address,
                "miner_hotkey": lead.get("source", lead.get("miner_hotkey")),
                "score": lead.get("conversion_score", lead.get("score")),
                "metadata": {
                    "owner_full_name": lead.get("owner_full_name", ""),
                    "first": lead.get("first", ""),
                    "last": lead.get("last", ""),
                    "linkedin": lead.get("linkedin", ""),
                    "website": lead.get("website", ""),
                    "industry": lead.get("industry", ""),
                    "sub_industry": lead.get("sub_industry", ""),
                    "region": lead.get("region", ""),
                    "role": lead.get("role", ""),
                    "email_score": lead.get("email_score"),
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
                bt.logging.debug(f"⏭️  Duplicate lead (trigger will notify miner): {lead.get('owner_email', lead.get('email', ''))}")
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
                    business = lead.get('business', 'Unknown')[:30]
                    # accept either lowercase or capitalised field
                    business = lead.get('business') or lead.get('Business', 'Unknown')
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

                # Get email from either field name
                email = prospect.get("owner_email", prospect.get("Owner(s) Email", ""))
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
            "business": lead.get("business", lead.get("Business", "")),
            "owner_full_name": lead.get("owner_full_name", lead.get("Owner Full name", "")),
            "first": lead.get("first", lead.get("First", "")),
            "last": lead.get("last", lead.get("Last", "")),
            "owner_email": lead.get("owner_email", lead.get("Owner(s) Email", "")),
            "linkedin": lead.get("linkedin", lead.get("LinkedIn", "")),
            "website": lead.get("website", lead.get("Website", "")),
            "industry": lead.get("industry", lead.get("Industry", "")),
            "sub_industry": lead.get("sub_industry", lead.get("Sub Industry", "")),
            "region": lead.get("region", lead.get("Region", "")),
            "source":     lead.get("source", ""),
            "curated_by": lead.get("curated_by", ""),
        }

        # score is kept only if the lead already has it (i.e. curated phase)
        if "conversion_score" in lead:
            mapped_lead["conversion_score"] = validation_score
        mapped_leads.append(mapped_lead)

    # V2: Record events for reward calculation when leads are added to pool
    try:
        from Leadpoet.validator.reward import record_event
        for lead in leads:
            if lead.get("source") and lead.get("curated_by") and lead.get("conversion_score"):
                record_event(lead)
                print(f"🎯 V2: Recorded event for lead {lead.get('owner_email', 'unknown')} "
                      f"(source: {lead['source']}, curator: {lead['curated_by']})")
    except Exception as e:
        print(f"⚠️  V2: Failed to record events: {e}")

    lead_pool.add_to_pool(mapped_leads)


def main():
    parser = argparse.ArgumentParser(description="LeadPoet Validator")
    add_validator_args(None, parser)
    parser.add_argument("--wallet_name", type=str, help="Wallet name")
    parser.add_argument("--wallet_hotkey", type=str, help="Wallet hotkey")
    parser.add_argument("--netuid", type=int, default=71, help="Network UID")
    parser.add_argument("--subtensor_network", type=str, default="finney", help="Subtensor network")
    parser.add_argument("--logging_trace", action="store_true", help="Enable trace logging")
    args = parser.parse_args()

    if args.logging_trace:
        bt.logging.set_trace(True)

    ensure_data_files()

    # Add this near the beginning of your validator startup, after imports
    from Leadpoet.validator.reward import start_epoch_monitor

    # Start the background epoch monitor when validator starts
    start_epoch_monitor()

    # Run the proper Bittensor validator
    config = bt.Config()
    config.wallet = bt.Config()
    config.wallet.name = args.wallet_name
    config.wallet.hotkey = args.wallet_hotkey
    # Only set custom wallet path if default doesn't exist
    default_wallet_path = Path.home() / ".bittensor" / "wallets" / "validator"
    if not default_wallet_path.exists():
        config.wallet.path = str(Path.cwd() / "bittensor" / "wallets") + "/"
    config.netuid = args.netuid
    config.subtensor = bt.Config()
    config.subtensor.network = args.subtensor_network

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
