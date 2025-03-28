# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# TODO(developer): Set your name
# Copyright © 2023 <your name>

import typing
import bittensor as bt

class LeadRequest(bt.Synapse):
    """
    Protocol for the LeadPoet subnet. Validators send a LeadRequest to miners specifying the number
    of leads needed and optional filters (industry, region). Miners respond by filling the leads field.

    Attributes:
        num_leads (int): Number of leads requested by the validator (1-100).
        industry (Optional[str]): Optional filter for the industry of the leads.
        region (Optional[str]): Optional filter for the region of the leads.
        leads (Optional[List[dict]]): List of lead dictionaries filled by the miner.
    """
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