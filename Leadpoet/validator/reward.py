# Leadpoet/validator/reward.py

import numpy as np
from typing import List
from validator_models.automated_checks import validate_lead_list as auto_check_leads

async def get_rewards(self, responses: List[List[dict]]) -> np.ndarray:
    """Compute miner rewards based on validator O_v scores."""
    rewards = []
    for leads in responses:
        if not leads or len(leads) == 0:
            rewards.append(0.0)
            continue
        validation = await self.validate_leads(leads)
        rewards.append(validation["O_v"])
    return np.array(rewards)

async def post_approval_check(self, leads: List[dict]) -> bool:
    """Run automated checks post-approval."""
    report = await auto_check_leads(leads)
    valid_count = sum(1 for entry in report if entry["status"] == "Valid")
    return valid_count / len(leads) >= 0.9 if leads else False

def calculate_emissions(self, total_emissions: float, validators: list) -> dict:
    """Calculate validator emissions (V_v) based on R_v."""
    Rv_total = sum(v.reputation for v in validators if v.reputation > 15)
    emissions = {}
    for v in validators:
        if v.reputation > 15:
            V_v = total_emissions * (v.reputation / Rv_total) if Rv_total > 0 else 0
            emissions[v.wallet.hotkey.ss58_address] = V_v
        else:
            emissions[v.wallet.hotkey.ss58_address] = 0
    return emissions