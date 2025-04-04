# The MIT License (MIT)
# Copyright © 2025 Yuma Rao
# Leadpoet
# Copyright © 2025 Leadpoet


import re
import random
import numpy as np
import bittensor as bt
import os
from Leadpoet.base.validator import BaseValidatorNeuron
from Leadpoet.validator.forward import forward
from Leadpoet.protocol import LeadRequest
from validator_models.os_validator_model import validate_lead_list
from Leadpoet.validator.reward import post_approval_check

class Validator(BaseValidatorNeuron):
    """
    Validator neuron class for the LeadPoet subnet. This validator queries miners for lead batches,
    performs automated checks, validates lead quality, and scores miners. Uses the forward method
    from templates/validator/forward.py and integrates open-source validation models.
    """

    def __init__(self, config=None):
        super(Validator, self).__init__(config=config)
        bt.logging.info("load_state()")
        self.load_state()
       
        self.email_regex = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
        # 20% of batch
        self.sample_ratio = 0.2
        # Track validator reputation and consistency for scoring
        self.reputation_score = 0  
        self.consistency_streak = 0  # For consistency factor
        # Configurable flag to use the open-source validator model
        self.use_open_source_model = config.get("use_open_source_validator_model", False) if config else False

    def validate_email(self, email: str) -> bool:
        """Validates email format using regex from documentation."""
        return bool(self.email_regex.match(email))

    def check_duplicates(self, leads: list) -> set:
        """Identifies duplicate emails in the batch; duplicates count as incorrect."""
        emails = [lead.get('Owner(s) Email', '') for lead in leads]
        seen = set()
        duplicates = set(email for email in emails if email in seen or seen.add(email))
        return duplicates

    async def validate_leads(self, leads: list, industry: str = None) -> float:
        """
        Validates a batch of leads using either the open-source model or a simulated review.
        
        Args:
            leads (list): List of lead dictionaries.
            industry (str, optional): Industry filter for validation.

        Returns:
            float: Accuracy score between 0.0 and 1.0.
        """
        if not leads:
            return 0.0
        if self.use_open_source_model:
            report = await validate_lead_list(leads, industry or "Unknown")
            return report["score"] / 100  # Return accuracy as 0.0-1.0
        else:
            sample_size = max(1, int(len(leads) * self.sample_ratio))
            sample_leads = random.sample(leads, min(sample_size, len(leads)))
            valid_emails = sum(1 for lead in sample_leads if self.validate_email(lead.get('Owner(s) Email', '')))
            return random.uniform(0.8, 1.0) * (valid_emails / sample_size)

    async def run_automated_checks(self, leads: list) -> bool:
        """Run post-approval automated checks and adjust reputation."""
        result = await post_approval_check(self, leads)
        if not result:
            self.reputation_score -= 20
            bt.logging.warning(f"Post-approval check failed, reputation reduced: {self.reputation_score}")
        return result

    async def forward(self):
        """Override forward to include post-approval checks."""
        await forward(self, post_process=lambda rewards, uids: self._post_process_with_checks(rewards, uids))

    async def _post_process_with_checks(self, rewards: np.ndarray, miner_uids: list):
        """Post-process rewards and run automated checks on approved batches."""
        self._update_reputation_and_consistency(rewards, miner_uids)
        # Run automated checks on approved batches (reward >= 0.8)
        for i, reward in enumerate(rewards):
            if reward >= 0.8:
                leads = (await self.dendrite(
                    [self.metagraph.axons[miner_uids[i]]],
                    LeadRequest(num_leads=100),
                    deserialize=True,
                    timeout=120
                ))[0].leads
                await self.run_automated_checks(leads)

    def _update_reputation_and_consistency(self, rewards: np.ndarray, miner_uids: list):
        """Updates validator reputation and consistency based on validation accuracy."""
        avg_reward = np.mean(rewards) if rewards.size > 0 else 0
        if avg_reward >= 0.8:  # High accuracy threshold
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

if __name__ == "__main__":
    with Validator() as validator:
        while True:
            bt.logging.info(f"Validator running... {time.time()}")
            time.sleep(5)