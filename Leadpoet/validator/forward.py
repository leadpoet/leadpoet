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
    num_leads: int = 100,  # Match batch size from documentation
    industry: str = None,
    region: str = None,
) -> Tuple[np.ndarray, list]:
    bt.logging.debug("Starting forward pass")
    start_time = time.time()

    miner_uids = get_random_uids(self, k=self.config.neuron.sample_size)
    bt.logging.debug(f"Selected miner UIDs: {miner_uids.tolist()}")

    if len(miner_uids) == 0:
        bt.logging.warning("No available miners to query.")
        return np.array([]), []

    axons = [self.metagraph.axons[uid] for uid in miner_uids]
    request = LeadRequest(num_leads=num_leads, industry=industry, region=region)
    responses = await self.dendrite(
        axons=axons,
        synapse=request,
        timeout=30,
        deserialize=True
    )
    bt.logging.info(f"Received {len(responses)} lead batch responses from miners")

    rewards = np.zeros(len(responses), dtype=np.float32)
    for idx, response in enumerate(responses):
        if not isinstance(response, LeadRequest):
            bt.logging.warning(f"Invalid response from UID {miner_uids[idx]}: Not a LeadRequest")
            rewards[idx] = 0.0
            continue
        if response.dendrite.status_code != 200:
            bt.logging.warning(f"Invalid response from UID {miner_uids[idx]}: Status {response.dendrite.status_code} - {response.dendrite.status_message}")
            rewards[idx] = 0.0
            continue
        if not response.leads:
            bt.logging.warning(f"Invalid response from UID {miner_uids[idx]}: No leads provided")
            rewards[idx] = 0.0
            continue
        validation = await validate_lead_list(response.leads, industry=response.industry)
        score = validation["score"] / 100.0
        rewards[idx] = score
        bt.logging.debug(f"UID {miner_uids[idx]} score: {score}")

    if post_process is not None:
        await post_process(rewards, miner_uids, responses)  # Await async post_process

    self.update_scores(rewards, miner_uids)
    bt.logging.info(f"Scored responses: {rewards.tolist()}")

    elapsed_time = time.time() - start_time
    bt.logging.trace(f"Forward pass completed in {elapsed_time:.2f}s")
    return rewards, miner_uids