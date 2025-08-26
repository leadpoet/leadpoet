import re
import time
import random
import requests, textwrap              # ← new deps
import numpy as np
import bittensor as bt
import os
import argparse
from datetime import datetime, timedelta
from Leadpoet.base.validator import BaseValidatorNeuron
from Leadpoet.protocol import LeadRequest
from validator_models.automated_checks import validate_lead_list as auto_check_leads, run_automated_checks
from Leadpoet.validator.reward import post_approval_check
from Leadpoet.base.utils.config import add_validator_args
import threading
import json
from Leadpoet.base.utils import queue as lead_queue
from Leadpoet.base.utils import pool as lead_pool
import asyncio
from typing import List, Dict
from aiohttp import web
import socket
from Leadpoet.utils.cloud_db import (
    save_leads_to_cloud,
    fetch_prospects_from_cloud,
    fetch_curation_requests,
    push_curation_result,
    push_miner_curation_request,     # ← NEW
    fetch_miner_curation_result,     # ← NEW
    push_validator_weights,
)
import uuid
import grpc
from google.cloud import firestore
from datetime import datetime, timezone
from math import isclose

# ─────────── LLM re-scoring helpers ────────────────────────────────
AVAILABLE_MODELS = [
    "deepseek/deepseek-chat-v3-0324:free",
    "deepseek/deepseek-r1:free",
    "meta-llama/llama-3.1-405b-instruct:free",
    #"mistralai/mistral-7b-instruct",
    "google/gemini-2.0-flash-exp:free",
    "moonshotai/kimi-k2:free",
    "mistralai/mistral-small-3.2-24b-instruct:free",
    #"cognitivecomputations/dolphin3.0-mistral-24b:free",
    #"openrouter/quasar-alpha",
    #"qwen/qwen-2.5-72b-instruct:free",
    #"google/gemma-3-27b-it:free",
    "meta-llama/llama-3.3-70b-instruct:free"
]

# New – models tried ONLY when a primary call fails  ────────────────
FALLBACK_MODELS = [
    "mistralai/mistral-7b-instruct",
    #"qwen/qwen3-4b:free",
]

OPENROUTER_KEY = os.getenv("OPENROUTER_API_KEY")

def _llm_score_lead(lead: dict, description: str, model: str) -> float:
    """
    Return a 0-0.5 score telling how well this lead fits the buyer
    description.  Falls back to a tiny keyword heuristic on failure.
    """
    def _heuristic() -> float:
        d  = description.lower()
        txt = (lead.get("Business","") + " " + lead.get("Industry","")).lower()
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

 # Accept either camel-case or lower-case field names coming from the miner
    prompt_user = (
        f"BUYER:\n{description}\n\n"
        f"LEAD:\n"
        f"Company:  {lead.get('Business',  lead.get('business',  ''))}\n"
        f"Industry: {lead.get('Industry', lead.get('industry', ''))}\n"
        f"Role:     {lead.get('role',     lead.get('Role', ''))}\n"
        f"Website:  {lead.get('Website',  lead.get('website',  ''))}"
    )



    # ─── debug: show what the model is judging ─────────────────────
    print("\n🛈  VALIDATOR-LLM INPUT ↓")
    print(textwrap.shorten(prompt_user, width=250, placeholder=" …"))

    def _extract(json_plus_reason: str) -> float:
        """Return score from first {...} block; raise if not parsable."""
        txt = json_plus_reason.strip()
        if txt.startswith("```"):
            txt = txt.strip("`").lstrip("json").strip()
        start, end = txt.find("{"), txt.find("}")
        if start == -1 or end == -1:
            raise ValueError("No JSON object found")
        payload = txt[start:end + 1]
        score = float(json.loads(payload).get("score", 0))
        score = max(0.0, min(score, 0.5))     # <= clamp every time
        print("🛈  VALIDATOR-LLM OUTPUT ↓")
        print(textwrap.shorten(txt, width=250, placeholder=" …"))
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

    # 1️⃣ primary
    try:
        return _try(model)
    except Exception as e:
        print(f"⚠️  Primary model failed ({model}): {e}")
        print("🛈  VALIDATOR-LLM OUTPUT ↓")
        print("<< no JSON response – using fallback >>")

    # 2️⃣ fallback picked at random
    for fb_model in random.sample(FALLBACK_MODELS, k=len(FALLBACK_MODELS)):
        try:
            print(f"🔄  Trying fallback model: {fb_model}")
            return _try(fb_model)
        except Exception as e:
            print(f"⚠️  Fallback failed ({fb_model}): {e}")
            print("<< fallback also failed – using heuristic >>")

    # Last-ditch heuristic
    return _heuristic()

import os, grpc, asyncio
os.environ.setdefault("GRPC_VERBOSITY", "DEBUG")
os.environ.setdefault("GRPC_TRACE", "handshaker,tsi,security,api,transport,http,call_error")

async def _grpc_ready_check(addr: str, timeout: float = 5.0) -> bool:
    try:
        ch = grpc.aio.insecure_channel(addr)
        await asyncio.wait_for(ch.channel_ready(), timeout=timeout)
        await ch.close()
        print(f"✅ gRPC preflight OK → {addr}")
        return True
    except Exception as e:
        print(f"⚠️ aio preflight failed for {addr}: {e}")
    # Fallback to sync probe, run in a thread so it doesn't require a Task
    def _sync_probe() -> bool:
        ch = grpc.insecure_channel(addr)
        grpc.channel_ready_future(ch).result(timeout=timeout)
        ch.close()
        return True
    try:
        ok = await asyncio.get_running_loop().run_in_executor(None, _sync_probe)
        if ok:
            print(f"✅ gRPC preflight OK (sync) → {addr}")
            return True
    except Exception as e:
        print(f"❌ gRPC preflight FAIL → {addr} | {e}")
    return False            # ← requires google-cloud-firestore already in requirements

# ──────────────────────────────────────────────────────────────────────────────
def _store_weights_in_firestore(uid: int, hotkey: str, weights: dict):
    """
    Persist the latest weight vector to Firestore.
    Collection: validator_weights
      • uid        – validator UID (int)
      • hotkey     – SS58 hotkey string
      • weights    – { miner_hotkey: share_float }
      • created_at – server timestamp (UTC)
    """
    try:
        db  = firestore.Client()
        doc = {
            "uid":        uid,
            "hotkey":     hotkey,
            "weights":    weights,
            "created_at": datetime.utcnow()
        }
        db.collection("validator_weights").add(doc)
        print("📝 Stored weights in Firestore (validator_weights)")
    except Exception as e:
        print(f"⚠️  Firestore write failed: {e}")
# ───────────────────────────────────────────────────────
class Validator(BaseValidatorNeuron):
    def __init__(self, config=None):
        super().__init__(config=config)

        # Set validator UID
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

        bt.logging.info("load_state()")
        self.load_state()

        # Add HTTP server for API requests
        self.app = web.Application()
        self.app.add_routes([web.post('/api/leads', self.handle_api_request)])
        
        self.email_regex = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
        self.sample_ratio = 0.2
        self.use_open_source_model = config.get("neuron", {}).get("use_open_source_validator_model", True)
        
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

        import time, numpy as np
        from datetime import datetime

        # Always refresh metagraph just before selecting miners so we don't use stale flags.
        try:
            self.metagraph.sync(subtensor=self.subtensor)
            print("🔄 Metagraph refreshed for miner selection.")
        except Exception as e:
            print(f"⚠️  Metagraph refresh failed (continuing with cached state): {e}")

        # 1️⃣ build the FULL list of miner axons (exclude validators)
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
            # Hairpin NAT bypass: if miner ip == my public ip, use localhost
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

            # TCP + robust gRPC preflight
            grpc_ready = {}
            for uid, ep in zip(miner_uids, endpoints):
                host, p = ep.split(":")
                try:
                    with socket.create_connection((host, int(p)), timeout=3):
                        print(f"✅ TCP preflight OK → {host}:{p} (UID {uid})")
                except Exception as e:
                    print(f"❌ TCP preflight FAIL → {host}:{p} (UID {uid}) :: {e}")
                grpc_ready[uid] = await _grpc_ready_check(ep, timeout=5.0)

        # Always start with an empty list so we never hit UnboundLocalError
        all_miner_leads: list = []

        # When dialing dendrite, wrap in a Task (prevents "Timeout context manager should be used inside a task")
        print("\n─────────  VALIDATOR ➜ DENDRITE  ─────────")
        print(f"📡  Dialing {len(axons)} miners: {[f'UID{u}' for u in miner_uids]}")
        print(f"⏱️   at {datetime.utcnow().isoformat()} UTC")

        _t0 = time.time()            # <── fix NameError
        miner_req = LeadRequest(num_leads=synapse.num_leads,
                                business_desc=synapse.business_desc)

        # bump timeout to 85 s (miner will cut off at 90 s)
        responses_task = asyncio.create_task(self.dendrite(
            axons       = axons,
            synapse     = miner_req,
            timeout     = 85,                   # <── was 60
            deserialize = False,
        ))
        responses = await responses_task
        print(f"⏲️  Dendrite completed in {(time.time() - _t0):.2f}s, analysing responses…")
        # Inspect each response with full status
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

        # 3️⃣ If no leads from Bittensor, fallback to Cloud-Run
        if not all_miner_leads:
            print("⚠️  Axon unreachable – falling back to cloud broker")
            req_id = push_miner_curation_request(
                self.wallet,
                {"num_leads": synapse.num_leads, "business_desc": synapse.business_desc}
            )
            print(f"📤 Sent curation request to Cloud-Run: {req_id}")

            # ── Wait for miner response via Cloud-Run ─────────────────────
            MAX_ATTEMPTS = 40      # 40 × 5 s  = 200 s
            SLEEP_SEC    = 5
            total_wait   = MAX_ATTEMPTS * SLEEP_SEC
            print(f"⏳ Waiting for miner response (up to {total_wait} s)…")

            for _ in range(MAX_ATTEMPTS):
                res = fetch_miner_curation_result(self.wallet)
                if res and res.get("leads"):
                    all_miner_leads = res["leads"]
                    print(f"✅ Received {len(all_miner_leads)} leads from Cloud-Run")
                    break
                time.sleep(SLEEP_SEC)

        # 4️⃣ Rank leads using LLM scoring (TWO rounds as intended)
        if all_miner_leads:
            print(f"🔍 Ranking {len(all_miner_leads)} leads with LLM...")
            scored_leads = []
            
            # Initialize aggregation dictionary for each lead
            aggregated = {id(lead): 0.0 for lead in all_miner_leads}
            
            # ROUND 1: First LLM scoring
            print(f"🔄 LLM round 1/2 (model: deepseek/deepseek-chat-v3-0324:free)")
            for lead in all_miner_leads:
                score = _llm_score_lead(lead, synapse.business_desc, "deepseek/deepseek-chat-v3-0324:free")
                aggregated[id(lead)] += score
            
            # ROUND 2: Second LLM scoring (random model selection)
            second_model = random.choice(AVAILABLE_MODELS)
            print(f"🔄 LLM round 2/2 (model: {second_model})")
            for lead in all_miner_leads:
                score = _llm_score_lead(lead, synapse.business_desc, second_model)
                aggregated[id(lead)] += score
            
            # Apply aggregated scores to leads
            for lead in all_miner_leads:
                lead["intent_score"] = round(aggregated[id(lead)], 3)  # Total of both rounds (0.0 - 1.0)
                scored_leads.append(lead)

            # Sort by aggregated intent_score and take top N
            scored_leads.sort(key=lambda x: x["intent_score"], reverse=True)
            top_leads = scored_leads[:synapse.num_leads]

            print(f" Top {len(top_leads)} leads selected:")
            for i, lead in enumerate(top_leads, 1):
                business = lead.get('Business', lead.get('business', 'Unknown'))
                score = lead.get('intent_score', 0)
                print(f"  {i}. {business} (score={score:.3f})")

            # Add c_validator_hotkey to leads being sent to client via Bittensor
            for lead in top_leads:
                lead["c_validator_hotkey"] = self.wallet.hotkey.ss58_address

            synapse.leads = top_leads

            # ✅ V2: After Final Curated List is frozen, call reward calculation
            if top_leads:
                try:
                    from Leadpoet.validator.reward import calculate_weights, record_event

                    # Record events for each lead in the Final Curated List
                    for lead in top_leads:
                        if lead.get("source") and lead.get("curated_by"):
                          
                            record_event(lead)

                    # Calculate V2 weights and emissions
                    rewards = calculate_weights(100.0)  # 100 Alpha total emission

                    # Log final weights and emissions
                    print(f"\n🎯 V2 REWARD CALCULATION COMPLETE:")
                    print(f"   Final Curated List: {len(top_leads)} prospects")
                    print(f"   K weights (Legitimacy): {rewards['K']}")
                    print(f"   S weights (Sourcing): {rewards['S']}")
                    print(f"   C weights (Curating): {rewards['C']}")
                    print(f"   Final weights (W): {rewards['W']}")
                    print(f"   Emissions: {rewards['E']}")

                    weights_dict = rewards["W"]                   # miner-hotkey ➜ share
                    # ─── NEW: publish weights on-chain ─────────────────────────────────
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

                    # SINGLE remaining Firestore write
                    push_validator_weights(self.wallet, self.uid, weights_dict)
                    # ───────────────────────────────────────────────────────────────────

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

                    # ✅ V2: Record events for reward calculation before adding to pool
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
        """Handle API requests from clients and broadcast to miners."""
        try:
            data = await request.json()
            num_leads     = data.get("num_leads", 1)
            business_desc = data.get("business_desc", "")
            print(f"\n RECEIVED API QUERY from client: {num_leads} leads | desc='{business_desc[:50]}…'")
            bt.logging.info(f" RECEIVED API QUERY from client: {num_leads} leads | desc='{business_desc[:50]}…'")

            # Get available miners
            from Leadpoet.api.get_query_axons import get_query_api_axons
            axons = await get_query_api_axons(self.wallet, self.metagraph, n=0.1, timeout=5)

            if not axons:
                print("❌ No available miners to query.")
                bt.logging.error("No available miners to query.")
                return web.json_response({
                    "leads": [],
                    "status_code": 503,
                    "status_message": "No active miners available",
                    "process_time": "0"
                }, status=503)

            print(f" SENDING QUERY to {len(axons)} available miners...")
            bt.logging.info(f" BROADCASTING QUERY to {len(axons)} miners")

            # Prepare request
            synapse = LeadRequest(num_leads=num_leads,
                                   business_desc=business_desc)

            # Send request to all miners
            # give miners enough time to curate
            responses = await self.dendrite(
                axons=axons,
                synapse=synapse,
                timeout=60,           # ← was 30 s
                deserialize=True
            )

            print(f" RECEIVED responses from {len(responses)} miners")

            # Process responses and collect all leads
            all_leads = []
            for response in responses:
                if isinstance(response, LeadRequest) and response.dendrite.status_code == 200 and response.leads:
                    all_leads.extend(response.leads)

            # miner may still be finishing – wait once more up to 90 s
            if not all_leads:
                print("⏳ No leads yet – waiting up to 60 s for miners to finish")
                try:
                    more_task = asyncio.create_task(
                        self.dendrite(axons=axons,
                                      synapse=synapse,
                                      timeout=60,
                                      deserialize=True),
                    )
                    more = await asyncio.wait_for(more_task, timeout=60)
                    for r in more:
                        if isinstance(r, LeadRequest) and r.dendrite.status_code == 200 and r.leads:
                            all_leads.extend(r.leads)
                except asyncio.TimeoutError:
                    pass

            if not all_leads:
                print("❌ Still no leads – giving up")
                bt.logging.warning("No valid leads received from miners")
                return web.json_response(
                    {"leads": [],
                     "status_code": 504,
                     "status_message":"Timeout waiting for miners",
                     "process_time":"0"},
                    status=504)

            print(f"📊 RECEIVED {len(all_leads)} leads – running TWO LLM re-scoring rounds…")

            aggregated = {id(ld): 0.0 for ld in all_leads}
            for rnd in range(2):
                mdl = random.choice(AVAILABLE_MODELS)
                print(f"\n🔄  LLM round {rnd+1}/2  (model: {mdl})")
                for ld in all_leads:
                    aggregated[id(ld)] += _llm_score_lead(ld, business_desc, mdl)

            for ld in all_leads:
                ld["intent_score"] = round(aggregated[id(ld)], 3)

            # Rank primarily by LLM aggregate, tie-break by miner's score.
            #  -> always default to 0 so we never compare None with float.
            all_leads.sort(
                key=lambda x: (
                    x.get("intent_score", 0.0),
                    x.get("miner_intent_score", 0.0)
                ),
                reverse=True,
            )

            top_leads = all_leads[:num_leads]

            print(f" TOP {len(top_leads)} RANKED LEADS:")
            for i, lead in enumerate(top_leads, 1):
                business    = lead.get('Business', lead.get('business', 'Unknown'))
                score_llm   = lead.get('intent_score', 0)
                score_miner = lead.get('miner_intent_score', 0)
                source      = lead.get('source', 'Unknown')
                curator     = lead.get('curated_by', 'Unknown')
                print(
                    f"  {i}. {business} "
                    f"(LLM {score_llm:.3f} | Miner {score_miner:.3f}, "
                    f"Source: {source}, Curator: {curator})"
                )

            # Add c_validator_hotkey to leads being sent to client
            for lead in top_leads:
                lead["c_validator_hotkey"] = self.wallet.hotkey.ss58_address

            print(f"📤 SENDING top {len(top_leads)} leads to client")
            bt.logging.info(f" RETURNING top {len(top_leads)} leads to client")

            return web.json_response({
                "leads": top_leads,
                "status_code": 200,
                "status_message": "OK",
                "process_time": "0"
            })

        except Exception as e:
            print(f"❌ Error handling API request: {e}")
            bt.logging.error(f"Error handling API request: {e}")
            return web.json_response({
                "leads": [],
                "status_code": 500,
                "status_message": f"Error: {str(e)}",
                "process_time": "0"
            }, status=500)

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

        # Enable low-level gRPC logs (printed directly to console)
        os.environ.setdefault("GRPC_VERBOSITY", "DEBUG")
        os.environ.setdefault(
            "GRPC_TRACE",
            "handshaker,tsi,security,api,transport,http,call_error"
        )
        print(f"🧪 gRPC env → VERBOSITY={os.getenv('GRPC_VERBOSITY')} "
              f"TRACE={os.getenv('GRPC_TRACE')}")

        # NOW build the axon with the **correct** port
        self.axon = bt.axon(
            wallet=self.wallet,
            ip      = "0.0.0.0",
            port    = self.config.axon.port,
            external_ip   = self.config.axon.external_ip,
            external_port = self.config.axon.external_port,
        )
        # expose buyer-query endpoint (LeadRequest → LeadRequest)
        self.axon.attach(self.forward)           # ← NEW
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

        print(f"Validator starting at block: {self.block}")
        print("✅ Validator is now serving on the Bittensor network")
        print("   Processing sourced leads and waiting for client requests...")

        # Show available miners
        self.show_available_miners()

        try:
            # Keep the validator running and continuously process leads
            while not self.should_exit:
                # Process any new leads that need validation (continuous)
                self.process_sourced_leads_continuous()
                self.process_curation_requests_continuous()

                self.sync()
        except KeyboardInterrupt:
            self.axon.stop()
            bt.logging.success("Validator killed by keyboard interrupt.")
            exit()
        except Exception as e:
            bt.logging.error(f"Error in validator: {e}")
            import traceback
            bt.logging.error(traceback.format_exc())

    def show_available_miners(self):
        """Show all available miners on the network"""
        try:
            print("\n🔍 Discovering available miners on subnet 401...")
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
        Continuously pull un-validated prospects from Firestore and process them.
        """
        try:
            prospects_batch = fetch_prospects_from_cloud(self.wallet, limit=250)

            if not prospects_batch:
                time.sleep(1)
                return

            print(f"🛎️  Pulled {len(prospects_batch)} prospect batch(es) from cloud")

            for entry in prospects_batch:
                # new format = list-of-leads under 'prospects'
                # fallback = entry IS the lead (old one-row format)
                miner_hotkey = entry.get("miner_hotkey", "unknown")[:10]
                prospects    = entry.get("prospects") or [entry]
                print(f"\n🟣 Processing sourced batch of {len(prospects)} prospects "
                      f"from miner {miner_hotkey}...")

                for lead in prospects:
                    try:
                        print(f"\n  Validating: {lead.get('business', lead.get('website',''))}")
                        print(f"    Email: {lead.get('owner_email','?')}")

                        # Run async validate_lead in sync context
                        result = asyncio.run(self.validate_lead(lead))
                        if result["is_legitimate"]:
                            # ✅ V2: Set current K miner after automated checks pass
                            from Leadpoet.validator.reward import set_current_K_miner
                            set_current_K_miner(miner_hotkey)
                            print(f"    ✅ Passed automated checks - K miner updated to {miner_hotkey}")
                            print(f"    🔑 K miner allocation: {miner_hotkey} now has Kₘ = 1, all others have Kₘ = 0")

                            # Add prospect to pool after K miner update
                            self.move_to_validated_leads(lead, result["score"])
                        else:
                            print(f"    ❌ Rejected: {result['reason']}")
                    except Exception as e:
                        print(f"    ❌ Error validating lead: {e}")

        except Exception as e:
            bt.logging.error(f"process_sourced_leads_continuous failure: {e}")
            time.sleep(1)

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

    def move_to_validated_leads(self, lead, score):
        """Write validated lead to Firestore; skip if it already exists."""
        # REMOVED: validation_score - this should only be added during curation
        lead["validator_hotkey"] = self.wallet.hotkey.ss58_address
        # Use ISO format timestamp instead of Unix timestamp
        lead["validated_at"] = datetime.now(timezone.utc).isoformat()

        try:
            stored = save_leads_to_cloud(self.wallet, [lead])   # True ↔ actually written
            email  = lead.get("owner_email", lead.get("email", "?"))
            biz    = lead.get("business", lead.get("website", ""))

            if stored:
                print(f"✅ Added 1 verified lead to main DB → {biz} ({email})")
            else:
                print(f"⚠️  Duplicate lead skipped → {biz} ({email})")
        except Exception as e:
            bt.logging.error(f"Cloud save failed: {e}")

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
            # 1️⃣ Check for required email field first
            email = lead.get('owner_email', lead.get('email', ''))
            if not email:
                return {'is_legitimate': False,
                        'reason': 'Missing email',
                        'score': 0.0}
            
            # 2️⃣ Map your field names to what automated_checks expects
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
            
            # 3️⃣ Use automated_checks for comprehensive validation
            passed, reason = await run_automated_checks(mapped_lead)
            
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
        except:
            return {'website_score': 0.0, 'industry_score': 0.0, 'region_score': 0.0}

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

            # 1️⃣ append once
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
    http_port = await validator.start_http_server()

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

            # ───────────────────────── curated list ─────────────────────────
            if request_type == "curated":
                print(f"🔍 Processing curated leads from {miner_hotkey[:20]}...")
                # Set the curator hotkey for all prospects in this batch
                for prospect in prospects:
                    prospect["curated_by"] = miner_hotkey

                # score with your open-source conversion model
                report  = await validate_lead_list(prospects, industry="Unknown")
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

            # ───────────────────────── sourced list ─────────────────────────
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

    # ✅ V2: Record events for reward calculation when leads are added to pool
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
    parser.add_argument("--netuid", type=int, default=401, help="Network UID")
    parser.add_argument("--subtensor_network", type=str, default="test", help="Subtensor network")
    parser.add_argument("--logging_trace", action="store_true", help="Enable trace logging")
    args = parser.parse_args()

    if args.logging_trace:
        bt.logging.set_trace(True)

    ensure_data_files()

    # Run the proper Bittensor validator
    config = bt.Config()
    config.wallet = bt.Config()
    config.wallet.name = args.wallet_name
    config.wallet.hotkey = args.wallet_hotkey
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

if __name__ == "__main__":
    main()
