# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# TODO(developer): Set your name
# Copyright © 2023 <your name>

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

import typing
import bittensor as bt

# TODO(developer): Rewrite with your protocol definition.

# This is the protocol for the dummy miner and validator.
# It is a simple request-response protocol where the validator sends a request
# to the miner, and the miner responds with a dummy response.

# ---- miner ----
# Example usage:
#   def dummy( synapse: Dummy ) -> Dummy:
#       synapse.dummy_output = synapse.dummy_input + 1
#       return synapse
#   axon = bt.axon().attach( dummy ).serve(netuid=...).start()

# ---- validator ---
# Example usage:
#   dendrite = bt.dendrite()
#   dummy_output = dendrite.query( Dummy( dummy_input = 1 ) )
#   assert dummy_output == 2


import typing
import bittensor as bt

class LeadBatch(bt.Synapse):
    """
    A protocol for the LeadPoet subnet where validators request lead batches from miners,
    and miners respond with a list of leads.

    Attributes:
    - batch_size: An integer specifying the number of leads requested by the validator.
    - leads: An optional list of dictionaries containing lead details (email, industry, region),
             filled by the miner in response.
    """
    # Request field, set by the validator
    batch_size: int

    # Response field, set by the miner
    leads: typing.Optional[typing.List[dict]] = None

    def deserialize(self) -> typing.List[dict]:
        """
        Deserialize the leads response from the miner.

        Returns:
        - typing.List[dict]: The list of leads provided by the miner.
        """
        return self.leads