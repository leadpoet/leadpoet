# The MIT License (MIT)
# Copyright © 2025 Yuma Rao
# Leadpoet
# Copyright © 2025 Leadpoet

import typing
import bittensor as bt

class LeadRequest(bt.Synapse):
   
    num_leads: int
    industry: typing.Optional[str] = None
    region: typing.Optional[str] = None
    leads: typing.Optional[typing.List[dict]] = None

    def deserialize(self) -> typing.List[dict]:
        """
        Deserializes the leads field for the validator to process the miner's response.

        Returns:
            List[dict]: The list of leads, or an empty list if none provided.
        """
        return self.leads if self.leads is not None else []