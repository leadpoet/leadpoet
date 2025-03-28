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

class LeadBatch(bt.Synapse):
    """
    A protocol for handling lead batch requests and responses in the Leadpoet subnet.
    
    Attributes:
    - leads: An optional list of dictionaries, each representing a lead with fields such as
      Business, Owner Full name, First, Last, Owner(s) Email, LinkedIn, and Website.
    """
    leads: typing.Optional[typing.List[typing.Dict[str, str]]] = None

    def deserialize(self) -> typing.List[typing.Dict[str, str]]:
        """
        Deserialize the lead batch output for validator processing.
        
        Returns:
        - typing.List[typing.Dict[str, str]]: The list of leads, or an empty list if None.
        """
        return self.leads if self.leads is not None else []