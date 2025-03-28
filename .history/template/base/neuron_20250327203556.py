# The MIT License (MIT)
# Copyright © 2025 Yuma Rao

import re
import random
import numpy as np
import bittensor as bt
import os
from template.base.validator import BaseValidatorNeuron
from template.validator.forward import forward

class Validator(BaseValidatorNeuron):
    """
    Validator neuron class for the LeadPoet subnet. This validator queries miners for lead batches,
    performs automated checks, simulates manual reviews, and scores the quality of the leads.
    Uses the forward method from templates/validator/forward.py.
    """

    def __init__(self, config=None):
        super(Validator, self).__init__(config=config)
        bt.logging.info("load_state()")
        self.load_state()
        # Email regex for format validation per documentation
        self.email_regex = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
        # Sample size for manual review (20% of batch as per documentation: 20/100)
        self.sample_ratio = 0.2
        # Track validator reputation and consistency for scoring
        self.reputation_score = 0  # Starting reputation per documentation
        self.consistency_streak = 0  # For consistency factor

    def validate_email(self, email: str) -> bool:
        """Validates email format using regex from documentation."""
        return bool(self.email_regex.match(email))

    def check_duplicates(self, leads: list) -> set:
        """Identifies duplicate emails in the batch; duplicates count as incorrect."""
        emails = [lead.get('Owner(s) Email', '') for lead in leads]
        seen = set()
        duplicates = set(email for email in emails if email in seen or seen.add(email))
        return duplicates

    def simulate_manual_review(self, leads: list, sample_size: int) -> float:
        """
        Simulates manual review of a sample. In a real scenario, this would check accuracy,
        relevance, and compliance. Here, we simulate with a random approval rate adjusted
        by basic checks.
        """
        if not leads:
            return 0.0
        # Basic simulation: assume 80-100% approval, adjusted by format validity
        valid_emails = sum(1 for lead in leads[:sample_size] if self.validate_email(lead.get('Owner(s) Email', '')))
        base_rate = random.uniform(0.8, 1.0) * (valid_emails / sample_size)
        return base_rate

    async def forward(self):
        """
        Calls the forward method from templates/validator/forward.py with a post-process callback
        to update validator reputation and consistency.
        """
        await forward(self, post_process=self._update_reputation_and_consistency)

    def _update_reputation_and_consistency(self, rewards: np.ndarray, miner_uids: list):
        """Updates validator reputation and consistency based on validation accuracy."""
        avg_reward = np.mean(rewards) if rewards.size > 0 else 0
        if avg_reward >= 0.9:  # High accuracy threshold per documentation
            self.reputation_score += 5
            self.consistency_streak += 1
        else:
            self.reputation_score -= 10
            self.consistency_streak = 0
        self.consistency_factor = min(1 + 0.025 * self.consistency_streak, 2.0)  # Alpha=0.025 per documentation
        bt.logging.debug(f"Reputation: {self.reputation_score}, Consistency Factor: {self.consistency_factor}")

    def save_state(self):
        """Saves the validator's state to a file for LeadPoet persistence."""
        bt.logging.info("Saving validator state.")
        state_path = os.path.join(self.config.neuron.full_path, "validator_state.npz")
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
        """Loads the validator's state from a file, with defaults if not found."""
        state_path = os.path.join(self.config.neuron.full_path, "validator_state.npz")
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
        else:
            bt.logging.info("No state file found. Initializing with defaults.")
            self.step = 0
            self.scores = np.zeros(self.metagraph.n, dtype=np.float32)
            self.hotkeys = self.metagraph.hotkeys.copy()
            self.reputation_score = 0
            self.consistency_streak = 0
            self.consistency_factor = 1.0

if __name__ == "__main__":
    with Validator() as validator:
        while True:
            bt.logging.info(f"Validator running... {time.time()}")
            time.sleep(5)