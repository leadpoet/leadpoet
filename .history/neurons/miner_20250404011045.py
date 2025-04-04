# The MIT License (MIT)
# Copyright © 2025 Yuma Rao
# Leadpoet
# Copyright © 2025 Leadpoet

import time
import typing
import bittensor as bt
import Leadpoet
from Leadpoet.base.miner import BaseMinerNeuron
from miner_models.get_leads import get_leads  # Optional import

class Miner(BaseMinerNeuron):
    """
    Miner neuron class for the LeadPoet subnet. This miner responds to validator queries by providing batches
    of leads, which are contact details for potential business opportunities. Validators then review these
    leads for quality.

    Inherits from BaseMinerNeuron, handling wallet, subtensor, metagraph, and other boilerplate tasks.
    """

    def __init__(self, config=None):
        super(Miner, self).__init__(config=config)
        # Initialize a counter for generating unique dummy leads
        self.lead_counter = 0
        # Configurable flag to use the open-source lead generation model
        self.use_open_source_model = config.get("use_open_source_lead_model", False) if config else False

    def get_dummy_lead(self) -> dict:
        """Generates a single dummy lead with incremental uniqueness."""
        self.lead_counter += 1
        return {
            "Business": f"Business {self.lead_counter}",
            "Owner Full name": f"Owner {self.lead_counter}",
            "First": f"First {self.lead_counter}",
            "Last": f"Last {self.lead_counter}",
            "Owner(s) Email": f"owner{self.lead_counter}@example.com",
            "LinkedIn": f"https://linkedin.com/in/owner{self.lead_counter}",
            "Website": f"https://business{self.lead_counter}.com",
            "Industry": "Dummy Industry",
            "Region": "Dummy Region"
        }

    async def forward(self, synapse: 'template.protocol.LeadRequest') -> 'template.protocol.LeadRequest':
        """
        Processes a LeadRequest synapse by generating a batch of leads based on the requested number.
        Optionally uses the open-source get_leads model if enabled, falling back to dummy leads on failure.

        Args:
            synapse (template.protocol.LeadRequest): The incoming request with num_leads, industry, and region.

        Returns:
            template.protocol.LeadRequest: The synapse with the leads field populated.
        """
        start_time = time.time()
        if self.use_open_source_model:
            try:
                leads = await get_leads(synapse.num_leads, synapse.industry, synapse.region)
            except Exception as e:
                bt.logging.error(f"Open-source lead generation failed: {e}")
                leads = [self.get_dummy_lead() for _ in range(synapse.num_leads)]
        else:
            leads = [self.get_dummy_lead() for _ in range(synapse.num_leads)]
        synapse.leads = leads
        elapsed = time.time() - start_time
        if elapsed > 122:
            bt.logging.warning(f"Lead generation exceeded 122 seconds: {elapsed:.2f}s")
        return synapse

    async def blacklist(self, synapse: 'template.protocol.LeadRequest') -> typing.Tuple[bool, str]:
        """
        Determines whether an incoming LeadRequest should be blacklisted. Only allows requests from
        registered hotkeys, with an optional check for validator status.

        Args:
            synapse (template.protocol.LeadRequest): The incoming request from a validator or API.

        Returns:
            Tuple[bool, str]: (True, reason) if blacklisted, (False, reason) if allowed.
        """
        if synapse.dendrite is None or synapse.dendrite.hotkey is None:
            bt.logging.warning("Received a request without a dendrite or hotkey.")
            return True, "Missing dendrite or hotkey"

        # Check if the hotkey is registered in the metagraph
        if (
            not self.config.blacklist.allow_non_registered
            and synapse.dendrite.hotkey not in self.metagraph.hotkeys
        ):
            bt.logging.trace(f"Blacklisting un-registered hotkey {synapse.dendrite.hotkey}")
            return True, "Unrecognized hotkey"

        # Optionally enforce that only validators can query
        if self.config.blacklist.force_validator_permit:
            uid = self.metagraph.hotkeys.index(synapse.dendrite.hotkey)
            if not self.metagraph.validator_permit[uid]:
                bt.logging.warning(f"Blacklisting non-validator hotkey {synapse.dendrite.hotkey}")
                return True, "Non-validator hotkey"

        bt.logging.trace(f"Not blacklisting recognized hotkey {synapse.dendrite.hotkey}")
        return False, "Hotkey recognized!"

    async def priority(self, synapse: 'template.protocol.LeadRequest') -> float:
        """
        Assigns priority to incoming LeadRequests based on the stake of the requesting entity.
        Higher stake results in higher priority.

        Args:
            synapse (template.protocol.LeadRequest): The incoming request.

        Returns:
            float: Priority score based on stake.
        """
        if synapse.dendrite is None or synapse.dendrite.hotkey is None:
            bt.logging.warning("Received a request without a dendrite or hotkey.")
            return 0.0

        caller_uid = self.metagraph.hotkeys.index(synapse.dendrite.hotkey)
        priority = float(self.metagraph.S[caller_uid])
        bt.logging.trace(f"Prioritizing {synapse.dendrite.hotkey} with value: {priority}")
        return priority

if __name__ == "__main__":
    with Miner() as miner:
        while True:
            bt.logging.info(f"Miner running... {time.time()}")
            time.sleep(5)