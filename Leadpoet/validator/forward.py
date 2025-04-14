import time
import asyncio
import numpy as np
import bittensor as bt
from typing import Tuple
from Leadpoet.utils.uids import get_random_uids
from Leadpoet.protocol import LeadRequest
from validator_models.os_validator_model import validate_lead_list


async def forward(
    self,
    post_process=None,
    num_leads: int = 10,  # Reduced for faster testing
    industry: str = None,
    region: str = None,
) -> Tuple[np.ndarray, list]:
    """
    Forward pass for validator: query miners, validate responses, assign rewards.
    """
    bt.logging.debug("Starting forward pass")
    start_time = time.time()

    # Get random miner UIDs
    miner_uids = get_random_uids(self, k=self.config.neuron.sample_size)
    bt.logging.debug(f"Selected miner UIDs: {miner_uids.tolist()}")

    if len(miner_uids) == 0:
        bt.logging.warning("No available miners to query.")
        return np.array([]), []

    # Get axons for selected UIDs
    axons = [self.metagraph.axons[uid] for uid in miner_uids]
    bt.logging.debug(f"Querying {len(axons)} axons")

    # Create LeadRequest synapse
    request = LeadRequest(num_leads=num_leads, industry=industry, region=region)
    bt.logging.debug(f"LeadRequest: num_leads={num_leads}, industry={industry}, region={region}")

    # Query miners
    try:
        responses = await self.dendrite(
            axons=axons,
            synapse=request,
            timeout=30,  # Reduced timeout for faster testing
            deserialize=True
        )
        bt.logging.info(f"Received {len(responses)} lead batch responses from miners")
        for i, resp in enumerate(responses):
            bt.logging.debug(f"Response {i}: status={resp.dendrite.status_code}, leads={len(resp.leads)}")
    except Exception as e:
        bt.logging.error(f"Error querying miners: {e}")
        return np.array([]), miner_uids

    # Validate responses and assign rewards
    rewards = np.zeros(len(responses), dtype=np.float32)
    for idx, response in enumerate(responses):
        bt.logging.debug(f"Processing response {idx} from UID {miner_uids[idx]}")
        if response.dendrite.status_code != 200 or not response.leads:
            bt.logging.warning(f"Invalid response from UID {miner_uids[idx]}: {response.dendrite.status_message}")
            rewards[idx] = 0.0
            continue

        try:
            # Validate leads using open-source model
            validation = await validate_lead_list(response.leads, industry=response.industry)
            bt.logging.trace(f"validation_report: {validation['validation_report']}")
            score = validation["score"] / 100.0  # Convert percentage to [0,1]
            rewards[idx] = score
            bt.logging.debug(f"UID {miner_uids[idx]} score: {score}")
        except Exception as e:
            bt.logging.error(f"Validation failed for UID {miner_uids[idx]}: {e}")
            rewards[idx] = 0.0

    # Apply post-processing (e.g., reputation updates)
    if post_process is not None:
        bt.logging.debug("Applying post-processing")
        post_process(rewards, miner_uids)

    # Update scores
    self.update_scores(rewards, miner_uids)
    bt.logging.info(f"Scored responses: {rewards.tolist()}")

    elapsed_time = time.time() - start_time
    bt.logging.trace(f"Forward pass completed in {elapsed_time:.2f}s")
    return rewards, miner_uids