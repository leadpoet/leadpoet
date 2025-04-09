# The MIT License (MIT)
# Copyright © 2025 Yuma Rao
# Copyright © 2025 Opentensor Foundation
# Copyright © 2025 Opentensor Technologies Inc

import bittensor as bt
from typing import List, Optional, Union, Any, Dict
from Leadpoet.protocol import LeadRequest
from Leadpoet.api.get_query_axons import get_query_api_axons

class LeadPoetAPI:
    """API for customers to request leads from the LeadPoet subnet."""

    def __init__(self, wallet: "bt.wallet", netuid: int = None):
        """
        Initializes the LeadPoetAPI with a wallet and optional netuid.

        Args:
            wallet (bt.wallet): The customer's wallet for network interaction.
            netuid (int, optional): The network UID of the Leadpoet subnet. Defaults to 33.
        """
        self.wallet = wallet
        self.netuid = netuid if netuid is not None else 33  # Default netuid, override as needed
        self.name = "leadpoet"
        bt.logging.info(f"Initialized LeadPoetAPI with netuid: {self.netuid}")

    def prepare_synapse(self, num_leads: int, industry: Optional[str] = None, region: Optional[str] = None) -> LeadRequest:
        """
        Prepares a LeadRequest synapse for querying miners.

        Args:
            num_leads (int): Number of leads requested (1-100).
            industry (Optional[str]): Industry filter for leads.
            region (Optional[str]): Region filter for leads.

        Returns:
            LeadRequest: The prepared synapse object.
        """
        if not 1 <= num_leads <= 100:
            raise ValueError("num_leads must be between 1 and 100")
        synapse = LeadRequest(num_leads=num_leads, industry=industry, region=region)
        bt.logging.debug(f"Prepared LeadRequest: num_leads={num_leads}, industry={industry}, region={region}")
        return synapse

    def process_responses(self, responses: List[Union["bt.Synapse", Any]]) -> List[Dict]:
        """
        Processes miner responses, extracting valid lead batches.

        Args:
            responses (List[Union[bt.Synapse, Any]]): List of LeadRequest responses from miners.

        Returns:
            List[Dict]: List of lead dictionaries from successful responses.
        """
        leads = []
        for response in responses:
            if not isinstance(response, LeadRequest) or response.dendrite.status_code != 200:
                bt.logging.warning(f"Invalid response: {response.dendrite.status_message if hasattr(response, 'dendrite') else 'No dendrite'}")
                continue
            if response.leads:
                leads.extend(response.leads)
            else:
                bt.logging.debug("Response contained no leads")
        bt.logging.info(f"Processed {len(leads)} leads from {len(responses)} responses")
        return leads

    async def get_leads(self, num_leads: int, industry: Optional[str] = None, region: Optional[str] = None) -> List[Dict]:
        """
        Queries miners for a batch of leads based on customer request.

        Args:
            num_leads (int): Number of leads to request (1-100).
            industry (Optional[str]): Industry filter.
            region (Optional[str]): Region filter.

        Returns:
            List[Dict]: List of lead dictionaries.
        """
        # Prepare the synapse
        synapse = self.prepare_synapse(num_leads=num_leads, industry=industry, region=region)

        # Get available miner axons
        dendrite = bt.dendrite(wallet=self.wallet)
        metagraph = bt.metagraph(netuid=self.netuid, subtensor=bt.subtensor(config=self.wallet.config))
        axons = await get_query_api_axons(self.wallet, metagraph=metagraph, n=0.1, timeout=3)

        if not axons:
            bt.logging.warning("No available miners to query")
            return []

        # Calculate dynamic timeout: 60 seconds base + 2 seconds per lead
        timeout = 60 + 2 * num_leads
        bt.logging.info(f"Querying {len(axons)} miners for {num_leads} leads with timeout {timeout}s")

        # Query miners
        responses = await dendrite(
            axons=axons,
            synapse=synapse,
            deserialize=True,
            timeout=timeout
        )

        # Process responses
        leads = self.process_responses(responses)
        return leads[:num_leads]  # Truncate to requested number if miners return more