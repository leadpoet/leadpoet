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
    num_leads: int = 10,
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
        timeout=60,
        deserialize=True
    )
    bt.logging.info(f"Received {len(responses)} lead batch responses from miners")

    rewards = np.zeros(len(miner_uids), dtype=np.float32)
    valid_responses = []

    for idx, response in enumerate(responses):
        if not isinstance(response, LeadRequest):
            bt.logging.warning(f"Invalid response from UID {miner_uids[idx]}: Not a LeadRequest")
            rewards[idx] = 0.0
            continue

        if response.dendrite.status_code != 200:
            bt.logging.warning(f"Invalid response from UID {miner_uids[idx]}: Status {response.dendrite.status_code}")
            rewards[idx] = 0.0
            continue
        if not response.leads:
            bt.logging.warning(f"Invalid response from UID {miner_uids[idx]}: No leads provided")
            rewards[idx] = 0.0
            continue
        validation = await self.validate_leads(response.leads, industry=response.industry)
        rewards[idx] = validation["O_v"]
        valid_responses.append(response)

    if post_process is not None:
        await post_process(rewards, miner_uids, valid_responses)

    self.update_scores(rewards, miner_uids)
    bt.logging.info(f"Scored responses: {rewards.tolist()}")

    elapsed_time = time.time() - start_time
    bt.logging.trace(f"Forward pass completed in {elapsed_time:.2f}s")
    return rewards, miner_uids