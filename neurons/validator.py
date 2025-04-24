import re
import time
import random
import numpy as np
import bittensor as bt
import os
import argparse
from Leadpoet.base.validator import BaseValidatorNeuron
from Leadpoet.validator.forward import forward
from Leadpoet.protocol import LeadRequest
from validator_models.os_validator_model import validate_lead_list
from Leadpoet.validator.reward import post_approval_check

class Validator(BaseValidatorNeuron):
    """
    Validator neuron class for the LeadPoet subnet. Queries miners for lead batches,
    validates lead quality using os_validator_model.py, performs automated checks with
    automated_checks.py, and scores miners.
    """
    def __init__(self, config=None):
        super(Validator, self).__init__(config=config)
        bt.logging.info("load_state()")
        self.load_state()
        
        self.email_regex = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
        self.sample_ratio = 0.2
        self.reputation_score = 0
        self.consistency_streak = 0
        self.use_open_source_model = config.get("use_open_source_validator_model", True) if config else True
        from Leadpoet.base.utils.pool import initialize_pool
        initialize_pool()  # Initialize the pool file

    def validate_email(self, email: str) -> bool:
        """Validates email format using regex."""
        return bool(self.email_regex.match(email))

    def check_duplicates(self, leads: list) -> set:
        """Identifies duplicate emails in the batch."""
        emails = [lead.get('Owner(s) Email', '') for lead in leads]
        seen = set()
        duplicates = set(email for email in emails if email in seen or seen.add(email))
        return duplicates

    async def validate_leads(self, leads: list, industry: str = None) -> float:
        """Validates leads using os_validator_model.py."""
        if not leads:
            return 0.0
        if self.use_open_source_model:
            report = await validate_lead_list(leads, industry or "Unknown")
            return report["score"] / 100
        else:
            sample_size = max(1, int(len(leads) * self.sample_ratio))
            sample_leads = random.sample(leads, min(sample_size, len(leads)))
            valid_emails = sum(1 for lead in sample_leads if self.validate_email(lead.get('Owner(s) Email', '')))
            return random.uniform(0.8, 1.0) * (valid_emails / sample_size)

    async def run_automated_checks(self, leads: list) -> bool:
        """Runs post-approval checks using automated_checks.py."""
        result = await post_approval_check(self, leads)
        if not result:
            self.reputation_score -= 20
            bt.logging.warning(f"Post-approval check failed, reputation reduced: {self.reputation_score}")
        return result

    async def forward(self):
        await forward(self, post_process=lambda rewards, uids, responses: self._post_process_with_checks(rewards, uids, responses), num_leads=100)

    async def _post_process_with_checks(self, rewards: np.ndarray, miner_uids: list, responses: list):
        self._update_reputation_and_consistency(rewards, miner_uids)
        for i, (reward, response) in enumerate(zip(rewards, responses)):
            if reward >= 0.9 and response.leads:
                leads = response.leads  # Use the initial batch
                if await self.run_automated_checks(leads):
                    from Leadpoet.base.utils.pool import add_to_pool
                    add_to_pool(leads)
                    bt.logging.info(f"Added {len(leads)} leads from UID {miner_uids[i]} to pool")
                else:
                    self.reputation_score -= 20
                    bt.logging.warning(f"Post-approval check failed for UID {miner_uids[i]}, reputation reduced: {self.reputation_score}")

    def _update_reputation_and_consistency(self, rewards: np.ndarray, miner_uids: list):
        """Updates reputation and consistency per documentation."""
        avg_reward = np.mean(rewards) if rewards.size > 0 else 0
        if avg_reward >= 0.9:
            self.reputation_score += 5
            self.consistency_streak += 1
        else:
            self.reputation_score -= 10
            self.consistency_streak = 0
        self.consistency_factor = min(1 + 0.025 * self.consistency_streak, 2.0)
        bt.logging.debug(f"Reputation: {self.reputation_score}, Consistency Factor: {self.consistency_factor}")

    def save_state(self):
        """Saves validator state."""
        bt.logging.info("Saving validator state.")
        state_path = os.path.join(self.config.neuron.full_path or os.getcwd(), "validator_state.npz")
        np.savez(
            state_path,
            step=self.step,
            scores=self.scores,
            hotkeys=self.hotkeys,
            reputation_score=self.reputation_score,
            consistency_streak=self.consistency_streak,
            consistency_factor=self.consistency_factor
        )

    def load_state(self):
        """Loads validator state with defaults if not found."""
        state_path = os.path.join(self.config.neuron.full_path or os.getcwd(), "validator_state.npz")
        if os.path.exists(state_path):
            bt.logging.info("Loading validator state.")
            try:
                state = np.load(state_path)
                self.step = state["step"]
                self.scores = state["scores"]
                self.hotkeys = state["hotkeys"]
                self.reputation_score = state["reputation_score"]
                self.consistency_streak = state["consistency_streak"]
                self.consistency_factor = state["consistency_factor"]
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
        self.reputation_score = 0
        self.consistency_streak = 0
        self.consistency_factor = 1.0

def main():
    """Entry point for the leadpoet-validate command."""
    parser = argparse.ArgumentParser(description="LeadPoet Validator")
    # Add common arguments
    parser.add_argument("--wallet_name", type=str, help="Wallet name")
    parser.add_argument("--wallet_hotkey", type=str, help="Wallet hotkey")
    parser.add_argument("--netuid", type=int, default=343, help="Network UID")
    parser.add_argument("--subtensor_network", type=str, default="test", help="Subtensor network")
    parser.add_argument("--mock", action="store_true", help="Run in mock mode")
    parser.add_argument("--logging_trace", action="store_true", help="Enable trace logging")
    # Add validator-specific arguments from BaseValidatorNeuron and utils
    BaseValidatorNeuron.add_args(parser)  # This adds neuron.sample_size, etc.
    args = parser.parse_args()

    if args.logging_trace:
        bt.logging.set_trace(True)

    # Manually create config object
    config = bt.Config()
    config.wallet = bt.Config()
    config.wallet.name = args.wallet_name
    config.wallet.hotkey = args.wallet_hotkey
    config.netuid = args.netuid
    config.subtensor = bt.Config()
    config.subtensor.network = args.subtensor_network
    config.mock = args.mock
    config.neuron = bt.Config()
    config.neuron.sample_size = getattr(args, 'neuron_sample_size', 10)  # Default to 10 if not provided
    config.neuron.moving_average_alpha = getattr(args, 'neuron_moving_average_alpha', 0.1)  # Default to 0.1
    config.neuron.use_open_source_validator_model = getattr(args, 'use_open_source_validator_model', True)  # Default to True
    config.neuron.num_concurrent_forwards = 1  # Default to 1 forward pass

    with Validator(config=config) as validator:
        while True:
            bt.logging.info(f"Validator running... {time.time()}")
            time.sleep(5)

if __name__ == "__main__":
    main()