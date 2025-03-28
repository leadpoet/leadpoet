# The MIT License (MIT)
# Copyright © 2025 Yuma Rao
# Leadpoet
# Copyright © 2025 Leadpoet

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import time
import typing
import bittensor as bt
import template
from template.base.miner import BaseMinerNeuron

class Miner(BaseMinerNeuron):
    """
    A miner neuron for the Leadpoet subnet, responsible for generating and submitting batches of leads.
    """
    BATCH_SIZE = 100  

    def __init__(self, config=None):
        super(Miner, self).__init__(config=config)
        # Initialization can be expanded later for lead sources if needed

    async def forward(self, synapse: template.protocol.LeadBatch) -> template.protocol.LeadBatch:
        """
        Processes incoming LeadBatch synapse by generating a fixed batch of leads.
        
        Args:
            synapse (template.protocol.LeadBatch): The synapse object from the validator.
        
        Returns:
            template.protocol.LeadBatch: The synapse with the leads field populated.
        """
        leads = self.generate_leads(self.BATCH_SIZE)
        synapse.leads = leads
        return synapse

    def generate_leads(self, batch_size: int) -> typing.List[typing.Dict[str, str]]:
        """
        Generates a batch of dummy leads for testing purposes.
        
        Args:
            batch_size (int): Number of leads to generate (fixed at 100 for Leadpoet).
        
        Returns:
            typing.List[typing.Dict[str, str]]: A list of lead dictionaries.
        """
        leads = []
        for i in range(batch_size):
            lead = {
                "Business": f"Business{i}",
                "Owner Full name": f"Owner{i}",
                "First": f"First{i}",
                "Last": f"Last{i}",
                "Owner(s) Email": f"owner{i}@example.com",
                "LinkedIn": f"https://linkedin.com/in/owner{i}",
                "Website": f"https://business{i}.com"
            }
            leads.append(lead)
        return leads

    async def blacklist(self, synapse: template.protocol.LeadBatch) -> typing.Tuple[bool, str]:
        """
        Determines whether an incoming request should be blacklisted.
        
        Args:
            synapse (template.protocol.LeadBatch): The synapse object from the requester.
        
        Returns:
            typing.Tuple[bool, str]: (blacklist flag, reason).
        """
        if synapse.dendrite is None or synapse.dendrite.hotkey is None:
            bt.logging.warning("Received a request without a dendrite or hotkey.")
            return True, "Missing dendrite or hotkey"

        uid = self.metagraph.hotkeys.index(synapse.dendrite.hotkey)
        if not self.config.blacklist.allow_non_registered and synapse.dendrite.hotkey not in self.metagraph.hotkeys:
            bt.logging.trace(f"Blacklisting un-registered hotkey {synapse.dendrite.hotkey}")
            return True, "Unrecognized hotkey"

        if self.config.blacklist.force_validator_permit:
            if not self.metagraph.validator_permit[uid]:
                bt.logging.warning(f"Blacklisting a request from non-validator hotkey {synapse.dendrite.hotkey}")
                return True, "Non-validator hotkey"

        bt.logging.trace(f"Not blacklisting recognized hotkey {synapse.dendrite.hotkey}")
        return False, "Hotkey recognized!"

    async def priority(self, synapse: template.protocol.LeadBatch) -> float:
        """
        Assigns priority to incoming requests based on requester stake.
        
        Args:
            synapse (template.protocol.LeadBatch): The synapse object from the requester.
        
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