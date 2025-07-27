import numpy as np
import random
import bittensor as bt
from Leadpoet.protocol import LeadRequest

async def ping_uids(dendrite, metagraph, uids, timeout=5):
    axons = [metagraph.axons[uid] for uid in uids]
    successful_uids = []
    failed_uids = []
    try:
        synapse = LeadRequest(num_leads=1, industry=None, region=None)
        responses = await dendrite(
            axons,
            synapse,
            deserialize=False,
            timeout=timeout,
        )
        for uid, response in zip(uids, responses):
            if response.dendrite.status_code == 200:
                successful_uids.append(uid)
                bt.logging.debug(f"UID {uid} responded with status 200")
            else:
                failed_uids.append(uid)
                bt.logging.debug(f"UID {uid} failed with status {response.dendrite.status_code}: {response.dendrite.status_message}")
    except Exception as e:
        bt.logging.error(f"Dendrite ping failed: {e}")
        failed_uids = uids
    bt.logging.debug(f"ping() successful uids: {successful_uids}")
    bt.logging.debug(f"ping() failed uids    : {failed_uids}")
    return successful_uids, failed_uids

async def get_query_api_nodes(dendrite, metagraph, n=0.1, timeout=5):
    bt.logging.debug(f"Fetching available miner nodes for subnet {metagraph.netuid}")
    metagraph.sync(subtensor=metagraph.subtensor)
    if metagraph.S.size == 0 or np.isnan(metagraph.S).any():
        bt.logging.warning("Stake array is empty or contains NaN. Selecting all active UIDs.")
        top_uids = [i for i in range(len(metagraph.axons)) if metagraph.axons[i].is_serving]
    else:
        top_uids = np.where(metagraph.S > np.quantile(metagraph.S, 1 - n))[0].tolist()
        if not top_uids:
            bt.logging.warning("No UIDs with sufficient stake. Falling back to all active UIDs.")
            top_uids = [i for i in range(len(metagraph.axons)) if metagraph.axons[i].is_serving]
    bt.logging.debug(f"Top UIDs by stake: {top_uids}")
    
    query_uids, failed_uids = await ping_uids(dendrite, metagraph, top_uids, timeout=timeout)
    if not query_uids:
        bt.logging.warning("No responsive UIDs after ping. Falling back to all active UIDs.")
        query_uids = [i for i in range(len(metagraph.axons)) if metagraph.axons[i].is_serving]
    
    bt.logging.debug(f"Available miner UIDs for subnet {metagraph.netuid}: {query_uids}")
    return query_uids

async def get_query_api_axons(wallet, metagraph=None, n=0.1, timeout=5, uids=None):
        dendrite = bt.dendrite(wallet=wallet)
        bt.logging.debug(f"Dendrite type: {dendrite.__class__.__name__}")
        
        if metagraph is None:
            metagraph = bt.metagraph(netuid=343)
        if uids is not None:
            query_uids = [uids] if isinstance(uids, int) else uids
        else:
            query_uids = await get_query_api_nodes(dendrite, metagraph, n=n, timeout=timeout)
        return [metagraph.axons[uid] for uid in query_uids]