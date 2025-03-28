# The MIT License (MIT)
# Copyright © 2025 Yuma Rao
# Leadpoet
# Copyright © 2025 Leadpoet

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
        
        # Using a 20% sample
        sample_size = max(1, int(len(leads) * self.sample_ratio))
        sample_leads = random.sample(leads, min(sample_size, len(leads)))
        manual_approval_rate = self.simulate_manual_review(sample_leads, sample_size)
        
        # Accuracy Calculation
        accuracy = 1.0 - (total_issues / len(leads))
        final_accuracy = accuracy * manual_approval_rate
        
        # Reward >=90% for approval, else 0
        reward = final_accuracy if final_accuracy >= 0.9 else 0.0
        rewards.append(reward)
    
    return np.array(rewards)