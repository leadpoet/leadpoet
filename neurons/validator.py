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
from Leadpoet.validator.forward import forward
from Leadpoet.protocol import LeadRequest
from validator_models.os_validator_model import validate_lead_list
from validator_models.automated_checks import validate_lead_list as auto_check_leads
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
    fetch_prospects_from_cloud,          # NEW
)

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
        
        # Otherwise proceed with normal validation
        if self.use_open_source_model:
            report = await validate_lead_list(leads, industry or "Unknown")
            O_v = report["score"] / 100.0
            if not await self.run_automated_checks(leads):
                O_v = 0.0
            return {"score": report["score"], "O_v": O_v}
        else:
            sample_size = max(1, int(len(leads) * self.sample_ratio))
            sample_leads = random.sample(leads, min(sample_size, len(leads)))
            valid_emails = sum(1 for lead in sample_leads if self.validate_email(lead.get('Owner(s) Email', '')))
            O_v = random.uniform(0.8, 1.0) * (valid_emails / sample_size)
            if not await self.run_automated_checks(leads):
                O_v = 0.0
            return {"score": O_v * 100, "O_v": O_v}

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

    async def forward(self):
        await forward(self, post_process=self._post_process_with_checks, num_leads=10)

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

        # Record rewards for ALL delivered leads in this query
        from Leadpoet.base.utils.pool import record_delivery_rewards
        record_delivery_rewards(all_delivered_leads)

        # Reset all_delivered_leads after recording rewards to prevent accumulation across API calls
        all_delivered_leads = []

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
                timeout=90,           # ← was 30 s
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
                print("⏳ No leads yet – waiting up to 90 s for miners to finish")
                try:
                    more = await asyncio.wait_for(
                        self.dendrite(axons=axons,
                                      synapse=synapse,
                                      timeout=90,
                                      deserialize=True),
                        timeout=90
                    )
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
                ld["validator_intent_score"] = round(aggregated[id(ld)], 3)

             # Rank primarily by LLM aggregate, tie-break by miner’s score.
            #  -> always default to 0 so we never compare None with float.
            all_leads.sort(
                key=lambda x: (
                    x.get("validator_intent_score", 0.0),
                    x.get("miner_intent_score",     0.0)
                ),
                reverse=True,
            )

            top_leads = all_leads[:num_leads]
            
            print(f" TOP {len(top_leads)} RANKED LEADS:")
            for i, lead in enumerate(top_leads, 1):
                business    = lead.get('Business', lead.get('business', 'Unknown'))
                score_llm   = lead.get('validator_intent_score', 0)
                score_miner = lead.get('miner_intent_score', 0)
                source      = lead.get('source', 'Unknown')
                curator     = lead.get('curated_by', 'Unknown')
                print(
                    f"  {i}. {business} "
                    f"(LLM {score_llm:.3f} | Miner {score_miner:.3f}, "
                    f"Source: {source}, Curator: {curator})"
                )

            
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
        
        self.axon.serve(netuid=self.config.netuid, subtensor=self.subtensor)
        self.axon.start()

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

                        result = self.validate_lead(lead)
                        if result["is_legitimate"]:
                            self.move_to_validated_leads(lead, result["score"])
                            print("    ✅ Passed basic checks")
                        else:
                            print(f"    ❌ Rejected: {result['reason']}")
                    except Exception as e:
                        print(f"    ❌ Error validating lead: {e}")

        except Exception as e:
            bt.logging.error(f"process_sourced_leads_continuous failure: {e}")
            time.sleep(1)

    def move_to_validated_leads(self, lead, score):
        """Write validated lead to Firestore; no local fallback."""
        lead["validation_score"] = score
        lead["validator_hotkey"] = self.wallet.hotkey.ss58_address
        lead["validated_at"]     = time.time()

        try:
            save_leads_to_cloud(self.wallet, [lead])
            bt.logging.info(f"✅ Lead saved to Firestore: {lead.get('email','?')}")
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

    def validate_lead(self, lead):
        """Validate a single lead (sourcing-side). No scoring – just pass / fail."""
        try:
            email  = lead.get('owner_email', '')
            domain = lead.get('business', '')

            # 1️⃣  Required fields present
            if not email or not domain:
                return {'is_legitimate': False,
                        'reason': 'Missing email or domain',
                        'score': 0.0}

            # 2️⃣  Disposable / throw-away e-mail check
            if self.is_disposable_email(email):
                return {'is_legitimate': False,
                        'reason': 'Disposable email detected',
                        'score': 0.0}

            # 3️⃣  Domain syntax validity
            if not self.check_domain_legitimacy(domain):
                return {'is_legitimate': False,
                        'reason': 'Invalid domain',
                        'score': 0.0}

            # 4️⃣  Duplicate detection handled elsewhere; if we reach here we pass
            return {'is_legitimate': True,
                    'reason': 'Passed validation',
                    'score': 1.0}

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
    
    lead_pool.add_to_pool(mapped_leads)


def main():
    parser = argparse.ArgumentParser(description="LeadPoet Validator")
    add_validator_args(None, parser)
    parser.add_argument("--wallet_name", type=str, help="Wallet name")
    parser.add_argument("--wallet_hotkey", type=str, help="Wallet hotkey")
    parser.add_argument("--netuid", type=int, default=343, help="Network UID")
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