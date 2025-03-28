# The MIT License (MIT)
# Copyright © 2025 Yuma Rao
# Copyright © 2025 Opentensor Foundation
# Copyright © 2025 Opentensor Technologies Inc

import numpy as np
import random
import bittensor as bt
from template.protocol import LeadRequest

async def ping_uids(dendrite, metagraph, uids, timeout=5):
    """
    Pings a list of UIDs to check their availability on the Bittensor network using a LeadRequest.

    Args:
        dendrite (bittensor.dendrite): The dendrite instance to use for pinging nodes.
        metagraph (bittensor.metagraph): The metagraph instance containing network information.
        uids (list): A list of UIDs (unique identifiers) to ping.
        timeout (int, optional): The timeout in seconds for each ping. Defaults to 5.

    Returns:
        tuple: A tuple containing two lists:
            - The first list contains UIDs that were successfully pinged.
            - The second list contains UIDs that failed to respond.
    """
    axons = [metagraph.axons[uid] for uid in uids]
    try:
        # Use a small LeadRequest to test miner compatibility
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

async def get_query_api_nodes(dendrite, metagraph, n=0.1, timeout=5):
    """
    Fetches available miner nodes to query for the LeadPoet subnet.

    Args:
        dendrite (bittensor.dendrite): The dendrite instance to use for querying nodes.
        metagraph (bittensor.metagraph): The metagraph instance containing network information.
        n (float, optional): The fraction of top nodes to consider based on stake. Defaults to 0.1.
        timeout (int, optional): The timeout in seconds for pinging nodes. Defaults to 5.

    Returns:
        list: A list of UIDs representing the available miner nodes.
    """
    bt.logging.debug(f"Fetching available miner nodes for subnet {metagraph.netuid}")
    # Select top nodes by stake, no validator trust filter for miners
    top_uids = np.where(metagraph.S > np.quantile(metagraph.S, 1 - n))[0].tolist()
    query_uids, _ = await ping_uids(dendrite, metagraph, top_uids, timeout=timeout)
    bt.logging.debug(f"Available miner UIDs for subnet {metagraph.netuid}: {query_uids}")
    if len(query_uids) > 5:  # Increased cap to 5 for LeadPoet scalability
        query_uids = random.sample(query_uids, 5)
    return query_uids

async def get_query_api_axons(wallet, metagraph=None, n=0.1, timeout=5, uids=None):
    """
    Retrieves the axons of query API nodes (miners) based on their availability and stake.

    Args:
        wallet (bittensor.wallet): The wallet instance to use for querying nodes.
        metagraph (bittensor.metagraph, optional): The metagraph instance containing network information.
        n (float, optional): The fraction of top nodes to consider based on stake. Defaults to 0.1.
        timeout (int, optional): The timeout in seconds for pinging nodes. Defaults to 5.
        uids (Union[List[int], int], optional): The specific UID(s) of the miner node(s) to query. Defaults to None.

    Returns:
        list: A list of axon objects for the available miner nodes.
    """
    dendrite = bt.dendrite(wallet=wallet)

    if metagraph is None:
        metagraph = bt.metagraph(netuid=21)  # Default netuid, overridden in LeadPoetAPI

    if uids is not None:
        query_uids = [uids] if isinstance(uids, int) else uids
    else:
        query_uids = await get_query_api_nodes(dendrite, metagraph, n=n, timeout=timeout)
    return [metagraph.axons[uid] for uid in query_uids]