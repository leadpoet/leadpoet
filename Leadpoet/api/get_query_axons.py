# Leadpoet/api/get_query_axons.py

import numpy as np
import random
import bittensor as bt
from Leadpoet.protocol import LeadRequest
from Leadpoet.mock import MockDendrite

async def ping_uids(dendrite, metagraph, uids, timeout=5, mock=False):
    # In mock mode - assume all UIDs are available
    if mock or getattr(bt.config(), 'mock', False) or isinstance(dendrite, MockDendrite):
        bt.logging.debug(f"Mock mode or MockDendrite: Assuming all UIDs {uids} are available")
        return uids, []
    
    axons = [metagraph.axons[uid] for uid in uids]
    try:
        synapse = LeadRequest(num_leads=1, industry=None, region=None)
        responses = await dendrite(
            axons,
            synapse,
            deserialize=False,
            timeout=timeout,
        )
        successful_uids = [
            uid
            for uid, response in zip(uids, responses)
            if response.dendrite.status_code == 200
        ]
        failed_uids = [
            uid
            for uid, response in zip(uids, responses)
            if response.dendrite.status_code != 200
        ]
    except Exception as e:
        bt.logging.error(f"Dendrite ping failed: {e}")
        successful_uids = []
        failed_uids = uids
    bt.logging.debug(f"ping() successful uids: {successful_uids}")
    bt.logging.debug(f"ping() failed uids    : {failed_uids}")
    return successful_uids, failed_uids

async def get_query_api_nodes(dendrite, metagraph, n=0.1, timeout=5, mock=False):
    bt.logging.debug(f"Fetching available miner nodes for subnet {metagraph.netuid}")
    # Select top nodes by stake, with fallback if stake array is empty
    if metagraph.S.size == 0 or np.isnan(metagraph.S).any():
        bt.logging.warning("Stake array is empty or contains NaN. Selecting all active UIDs.")
        top_uids = [i for i in range(len(metagraph.axons)) if metagraph.axons[i].is_serving]
    else:
        top_uids = np.where(metagraph.S > np.quantile(metagraph.S, 1 - n))[0].tolist()
    bt.logging.debug(f"Top UIDs by stake: {top_uids}")
    
    # Bypass ping_uids in mock mode
    if mock:
        bt.logging.debug(f"Mock mode: Skipping ping_uids, returning all UIDs {top_uids}")
        query_uids = top_uids
    else:
        query_uids, _ = await ping_uids(dendrite, metagraph, top_uids, timeout=timeout, mock=mock)
    
    bt.logging.debug(f"Available miner UIDs for subnet {metagraph.netuid}: {query_uids}")
    if len(query_uids) > 5:
        query_uids = random.sample(query_uids, 5)
    return query_uids

async def get_query_api_axons(wallet, metagraph=None, n=0.1, timeout=5, uids=None, mock=False):
    # Use MockDendrite in mock mode
    if mock:
        dendrite = MockDendrite(wallet=wallet, use_open_source=True)
        bt.logging.debug("Using MockDendrite for get_query_api_axons in mock mode")
    else:
        dendrite = bt.dendrite(wallet=wallet)
    bt.logging.debug(f"Dendrite type: {dendrite.__class__.__name__}, mock: {mock}, bt.config().mock: {getattr(bt.config(), 'mock', False)}")
    
    if metagraph is None:
        metagraph = bt.metagraph(netuid=343)  # Updated to match LeadPoet netuid
    if uids is not None:
        query_uids = [uids] if isinstance(uids, int) else uids
    else:
        query_uids = await get_query_api_nodes(dendrite, metagraph, n=n, timeout=timeout, mock=mock)
    return [metagraph.axons[uid] for uid in query_uids]