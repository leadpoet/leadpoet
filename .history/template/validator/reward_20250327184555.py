# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# Leadpoet
# Copyright © 2023 Leadpoet

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
import numpy as np
from typing import List

def get_rewards(self, responses: List[List[dict]]) -> np.ndarray:
    """
    Calculates rewards for miners based on lead batch quality per LeadPoet documentation.

    Args:
        responses (List[List[dict]]): List of lead batches from miners.

    Returns:
        np.ndarray: Array of rewards (0.0 to 1.0) for each miner.
    """
    rewards = []
    for leads in responses:
        if not leads or len(leads) == 0:
            rewards.append(0.0)
            continue
        
        # Automated Pre-Checks
        duplicates = self.check_duplicates(leads)
        invalid_emails = [lead for lead in leads if not self.validate_email(lead.get('Owner(s) Email', ''))]
        total_issues = len(duplicates) + len(invalid_emails)
        
        # Simulated Manual Review (20% sample)
        sample_size = max(1, int(len(leads) * self.sample_ratio))
        sample_leads = random.sample(leads, min(sample_size, len(leads)))
        manual_approval_rate = self.simulate_manual_review(sample_leads, sample_size)
        
        # Accuracy Calculation
        accuracy = 1.0 - (total_issues / len(leads))
        final_accuracy = accuracy * manual_approval_rate
        
        # Reward per documentation: >=90% for approval, else 0
        reward = final_accuracy if final_accuracy >= 0.9 else 0.0
        rewards.append(reward)
    
    return np.array(rewards)