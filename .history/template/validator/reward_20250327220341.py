# The MIT License (MIT)
# Copyright © 2025 Yuma Rao
# Leadpoet
# Copyright © 2025 Leadpoet

import numpy as np
from typing import List
from validator_models.automated_checks import validate_lead_list as auto_check_leads

async def get_rewards(self, responses: List[List[dict]]) -> np.ndarray:
    rewards = []
    for leads in responses:
        if not leads or len(leads) == 0:
            rewards.append(0.0)
            continue
        
        # Automated Pre-Checks
        duplicates = self.check_duplicates(leads)
        invalid_emails = [lead for lead in leads if not self.validate_email(lead.get('Owner(s) Email', ''))]
        total_issues = len(duplicates) + len(invalid_emails)
        
        # Validation using open-source model or simulation
        accuracy = await self.validate_leads(leads, industry=None)
        pre_check_accuracy = 1.0 - (total_issues / len(leads))
        final_accuracy = pre_check_accuracy * accuracy
        
        reward = final_accuracy if final_accuracy >= 0.8 else 0.0
        rewards.append(reward)
    
    return np.array(rewards)

async def post_approval_check(self, leads: List[dict]) -> bool:
    """Run automated checks post-approval."""
    report = await auto_check_leads(leads)
    valid_count = sum(1 for entry in report if entry["status"] == "Valid")
    return valid_count / len(leads) >= 0.9 if leads else False