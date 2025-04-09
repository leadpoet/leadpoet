# The MIT License (MIT)
# Copyright © 2025 Yuma Rao
# Leadpoet
# Copyright © 2025 Leadpoet

import time
import bittensor as bt
from typing import Callable, Optional

from Leadpoet.protocol import LeadRequest
from Leadpoet.validator.reward import get_rewards
from Leadpoet.utils.uids import get_random_uids

async def forward(self, post_process: Optional[Callable] = None):
    """
    The forward function is called by the validator every time step for the LeadPoet subnet.
    It queries miners for lead batches and scores their responses within a 2-minute constraint.

    Args:
        self (:obj:`bittensor.neuron.Neuron`): The validator neuron object with necessary state.
        post_process (Optional[Callable]): Optional callback to process results after scoring (e.g., update reputation).
    """
    start_time = time.time()

    # Select miners to query (sample_size configurable, e.g., 10 miners)
    miner_uids = get_random_uids(self, k=self.config.neuron.sample_size)

    # Query miners with a LeadRequest for 100 leads (default batch size per documentation)
    synapse = LeadRequest(num_leads=100, industry=None, region=None)
    responses = await self.dendrite(
        axons=[self.metagraph.axons[uid] for uid in miner_uids],
        synapse=synapse,
        deserialize=True,
        timeout=120  # 2-minute timeout per LeadPoet documentation
    )

    # Log the number of responses received
    bt.logging.info(f"Received {len(responses)} lead batch responses from miners")

    # Calculate rewards based on lead quality
    rewards = get_rewards(self, responses=responses)
    bt.logging.info(f"Scored responses: {rewards}")

    # Update miner scores using the base class method
    self.update_scores(rewards, miner_uids)

    # Call post-process callback if provided (e.g., for reputation updates)
    if post_process:
        post_process(rewards, miner_uids)

    # Ensure the forward pass stays within timing constraints
    elapsed = time.time() - start_time
    if elapsed > 120:
        bt.logging.warning(f"Forward pass exceeded 2 minutes: {elapsed:.2f}s")
    else:
        bt.logging.debug(f"Forward pass completed in {elapsed:.2f}s")

    # Sleep to maintain a consistent step interval (e.g., 5 seconds total)
    time.sleep(max(0, 5 - elapsed))