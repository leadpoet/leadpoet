# neurons/validator.py

import re
import time
import random
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

class Validator(BaseValidatorNeuron):
    def __init__(self, config=None):
        super(Validator, self).__init__(config=config)
        bt.logging.info("load_state()")
        self.load_state()
        
        self.email_regex = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
        self.sample_ratio = 0.2
        self.use_open_source_model = config.get("use_open_source_validator_model", True) if config else True
        
      
        self.precision = 15.0 
        self.consistency = 1.0  
        self.collusion_flag = 1  # F_v: 1 (no collusion) or 0 (collusion)
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
        if self.use_open_source_model:
            report = await validate_lead_list(leads, industry or "Unknown")
            O_v = report["score"] / 100.0
            if await self.run_automated_checks(leads) == False:
                O_v = 0.0  # Set O_v = 0 if automated checks fail
            return {"score": report["score"], "O_v": O_v}
        else:
            sample_size = max(1, int(len(leads) * self.sample_ratio))
            sample_leads = random.sample(leads, min(sample_size, len(leads)))
            valid_emails = sum(1 for lead in sample_leads if self.validate_email(lead.get('Owner(s) Email', '')))
            O_v = random.uniform(0.8, 1.0) * (valid_emails / sample_size)
            if await self.run_automated_checks(leads) == False:
                O_v = 0.0
            return {"score": O_v * 100, "O_v": O_v}

    async def run_automated_checks(self, leads: list) -> bool:
        report = await auto_check_leads(leads)
        valid_count = sum(1 for entry in report if entry["status"] == "Valid")
        return valid_count / len(leads) >= 0.9 if leads else False

    async def reputation_challenge(self):
        """Simulate dummy lead list audit."""
        dummy_leads = [
            {"Business": f"Mock Business {i}", "Owner(s) Email": f"owner{i}@mock.com", "Website": f"https://business{i}.com", "Industry": "Tech & AI"}
            for i in range(10)
        ]
        known_score = random.uniform(0.8, 1.0)  # Simulate known quality score
        validation = await self.validate_leads(dummy_leads)
        O_v = validation["O_v"]
        if abs(O_v - known_score) <= 0.1:
            bt.logging.info("Passed reputation challenge")
        else:
            self.precision = max(0, self.precision - 10)
            bt.logging.warning(f"Failed reputation challenge, P_v reduced to {self.precision}")
        self.update_reputation()

    def update_consistency(self):
        """Calculate C_v based on J_v over 14, 30, and 90 days."""
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
        """Calculate R_v = P_v * C_v * F_v."""
        self.reputation = self.precision * self.consistency * self.collusion_flag
        # Check trusted validator status
        registration_duration = (datetime.now() - self.registration_time).days
        self.trusted_validator = self.reputation > 85 and registration_duration >= 30
        bt.logging.debug(f"Updated R_v: {self.reputation}, Trusted: {self.trusted_validator}")

    async def handle_buyer_feedback(self, leads: list, feedback_score: float):
        """Adjust P_v and F based on buyer feedback (B)."""
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
                # Update validation history with new F
                for validation in self.validation_history:
                    if validation["leads"] == leads:
                        validation["F"] = f_new
                bt.logging.info(f"Applied buyer feedback B={feedback_score}: P_v={self.precision}, F={f_new}")
                break
        self.update_reputation()

    async def submit_appeal(self):
        """Submit a collusion flag appeal."""
        if self.collusion_flag == 1:
            bt.logging.info("No collusion flag to appeal")
            return
        self.appeal_status = {"votes": [], "start_time": datetime.now()}
        bt.logging.info("Collusion flag appeal submitted")

    async def vote_on_appeal(self, validator_hotkey: str, vote: int):
        """Vote on an appeal (E_v = 0 or 1)."""
        if self.appeal_status is None or self.appeal_status != "pending":
            bt.logging.warning("No active appeal to vote on")
            return
        weight = {90: 5, 80: 3, 70: 2, 0: 1}.get(next(k for k in [90, 80, 70, 0] if self.precision > k), 1)
        self.appeal_status["votes"].append({"hotkey": validator_hotkey, "E_v": vote, "H_v": weight})
        bt.logging.debug(f"Vote submitted: E_v={vote}, H_v={weight}")

    async def resolve_appeal(self):
        """Resolve appeal after 7 days."""
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
        await forward(self, post_process=self._post_process_with_checks, num_leads=100)

    async def _post_process_with_checks(self, rewards: np.ndarray, miner_uids: list, responses: list):
        """Calculate F, update P_v, and handle trusted validator logic."""
        validators = [self]  # In a real subnet, query other validators
        validator_scores = []
        trusted_validators = [v for v in validators if v.trusted_validator]
        
        for i, response in enumerate(responses):
            if not response.leads:
                continue
            validation = await self.validate_leads(response.leads, response.industry)
            O_v = validation["O_v"]
            validator_scores.append({"O_v": O_v, "R_v": self.reputation, "leads": response.leads})
        
        # Check trusted validator conditions
        trusted_low_scores = sum(1 for v in trusted_validators for s in validator_scores if v == self and s["O_v"] < 0.8)
        trusted_rejections = sum(1 for v in trusted_validators for s in validator_scores if v == self and s["O_v"] == 0)
        use_trusted = trusted_low_scores / len(trusted_validators) > 0.67 if trusted_validators else False
        reject = trusted_rejections / len(trusted_validators) > 0.5 if trusted_validators else False
        
        if reject:
            bt.logging.info("Submission rejected by >50% trusted validators")
            return
        
        # Calculate F
        Rs_total = sum(s["R_v"] for s in validator_scores if s["R_v"] > 15)
        F = sum(s["O_v"] * (s["R_v"] / Rs_total) for s in validator_scores if s["R_v"] > 15) if Rs_total > 0 else 0
        if use_trusted:
            trusted_scores = [s for s in validator_scores if any(v == self and v.trusted_validator for v in validators)]
            Rs_total_trusted = sum(s["R_v"] for s in trusted_scores if s["R_v"] > 15)
            F = sum(s["O_v"] * (s["R_v"] / Rs_total_trusted) for s in trusted_scores if s["R_v"] > 15) if Rs_total_trusted > 0 else 0
        
        # Update P_v and history
        for s in validator_scores:
            if abs(s["O_v"] - F) <= 0.1:
                self.precision = min(100, self.precision + 10)
            elif s["O_v"] > 0 and not await self.run_automated_checks(s["leads"]):
                self.precision = max(0, self.precision - 15)
            self.validation_history.append({"O_v": s["O_v"], "F": F, "timestamp": datetime.now(), "leads": s["leads"]})
        
        self.update_consistency()
        self.update_reputation()
        
        # Add to pool if approved
        for i, (reward, response) in enumerate(zip(rewards, responses)):
            if reward >= 0.9 and response.leads:
                if await self.run_automated_checks(response.leads):
                    from Leadpoet.base.utils.pool import add_to_pool
                    add_to_pool(response.leads)
                    bt.logging.info(f"Added {len(response.leads)} leads from UID {miner_uids[i]} to pool")
                else:
                    self.precision = max(0, self.precision - 15)
                    bt.logging.warning(f"Post-approval check failed for UID {miner_uids[i]}, P_v reduced: {self.precision}")
        
        # Periodically run reputation challenge
        if random.random() < 0.1:  # 10% chance per forward pass
            await self.reputation_challenge()

    def save_state(self):
        """Saves validator state."""
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
        """Loads validator state with defaults if not found."""
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
        """Initializes default state values."""
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

def main():
    parser = argparse.ArgumentParser(description="LeadPoet Validator")
    parser.add_argument("--wallet_name", type=str, help="Wallet name")
    parser.add_argument("--wallet_hotkey", type=str, help="Wallet hotkey")
    parser.add_argument("--netuid", type=int, default=343, help="Network UID")
    parser.add_argument("--subtensor_network", type=str, default="test", help="Subtensor network")
    parser.add_argument("--mock", action="store_true", help="Run in mock mode")
    parser.add_argument("--logging_trace", action="store_true", help="Enable trace logging")
    BaseValidatorNeuron.add_args(parser)
    args = parser.parse_args()

    if args.logging_trace:
        bt.logging.set_trace(True)

    config = bt.Config()
    config.wallet = bt.Config()
    config.wallet.name = args.wallet_name
    config.wallet.hotkey = args.wallet_hotkey
    config.netuid = args.netuid
    config.subtensor = bt.Config()
    config.subtensor.network = args.subtensor_network
    config.mock = args.mock
    config.neuron = bt.Config()
    config.neuron.sample_size = getattr(args, 'neuron_sample_size', 10)
    config.neuron.moving_average_alpha = getattr(args, 'neuron_moving_average_alpha', 0.1)
    config.neuron.use_open_source_validator_model = getattr(args, 'use_open_source_validator_model', True)
    config.neuron.num_concurrent_forwards = 1

    with Validator(config=config) as validator:
        while True:
            bt.logging.info(f"Validator running... {time.time()}")
            time.sleep(5)

if __name__ == "__main__":
    main()