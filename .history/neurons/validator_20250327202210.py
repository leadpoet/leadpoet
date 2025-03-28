# The MIT License (MIT)
# Copyright © 2025 Yuma Rao
# Leadpoet
# Copyright © 2025 Leadpoet

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import re
import random
import numpy as np
import bittensor as bt
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

if __name__ == "__main__":
    with Validator() as validator:
        while True:
            bt.logging.info(f"Validator running... {time.time()}")
            time.sleep(5)