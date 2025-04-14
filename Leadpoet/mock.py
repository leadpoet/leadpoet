import time
import asyncio
import random
import bittensor as bt
from typing import List, Dict, Optional
from Leadpoet.protocol import LeadRequest

# Import LeadPoet-specific models (assuming they'll be moved to miner_models and validator_models)
try:
    from miner_models.get_leads import get_leads
    from validator_models.automated_checks import validate_lead_list as auto_check_leads
except ImportError:
    # Fallback for when files aren't in place yet
    get_leads = None
    auto_check_leads = None

class MockSubtensor(bt.MockSubtensor):
    """Mock Bittensor subtensor for LeadPoet subnet simulation."""
    def __init__(self, netuid: int = 33, n: int = 16, wallet: Optional[bt.wallet] = None, network: str = "mock"):
        super().__init__(network=network)
        self.netuid = netuid  # Default to LeadPoet netuid (adjustable)

        if not self.subnet_exists(netuid):
            self.create_subnet(netuid)

        # Register validator (uid=0)
        if wallet is not None:
            self.force_register_neuron(
                netuid=netuid,
                hotkey=wallet.hotkey.ss58_address,
                coldkey=wallet.coldkey.ss58_address,
                balance=100000,
                stake=100000,
                validator_permit=True  # Mark as validator
            )

        # Register n mock miners
        for i in range(1, n + 1):
            self.force_register_neuron(
                netuid=netuid,
                hotkey=f"miner-hotkey-{i}",
                coldkey=f"mock-coldkey-{i}",
                balance=100000,
                stake=100000,
                validator_permit=False  # Miners, not validators
            )
        bt.logging.info(f"MockSubtensor initialized with {n} miners and 1 validator on netuid {netuid}")

class MockMetagraph(bt.metagraph):
    """Mock metagraph for LeadPoet subnet simulation."""
    def __init__(self, netuid: int = 33, network: str = "mock", subtensor: Optional[bt.MockSubtensor] = None):
        super().__init__(netuid=netuid, network=network, sync=False)
        if subtensor is not None:
            self.subtensor = subtensor
        self.sync(subtensor=subtensor)

        # Set mock IP and port for axons
        for axon in self.axons:
            axon.ip = "127.0.0.1"
            axon.port = 8091 + self.axons.index(axon)  # Unique ports for each axon

        bt.logging.info(f"MockMetagraph: {self}")
        bt.logging.debug(f"Axons: {self.axons}")

class MockDendrite(bt.dendrite):
    """Mock dendrite for LeadPoet subnet, simulating miner responses with lead batches."""
    def __init__(self, wallet: bt.wallet, use_open_source: bool = False):
        super().__init__(wallet)
        self._wallet = wallet  # Explicitly store wallet
        self.use_open_source = use_open_source  # Toggle for using get_leads.py
        self.email_counter = 0  # For generating unique dummy emails
        bt.logging.debug(f"MockDendrite initialized with wallet: {self._wallet}")

    def generate_dummy_lead(self) -> Dict:
        """Generate a single dummy lead matching LeadPoet format."""
        self.email_counter += 1
        lead = {
            "Business": f"Mock Business {self.email_counter}",
            "Owner Full name": f"Owner {self.email_counter}",
            "First": f"First {self.email_counter}",
            "Last": f"Last {self.email_counter}",
            "Owner(s) Email": f"owner{self.email_counter}@mockleadpoet.com",
            "LinkedIn": f"https://linkedin.com/in/owner{self.email_counter}",
            "Website": f"https://business{self.email_counter}.com",
            "Industry": random.choice(["Tech & AI", "Finance & Fintech", "Health & Wellness", "Media & Education", "Energy & Industry"]),
            "Region": random.choice(["US", "EU", "Asia", "Global"])
        }
        bt.logging.debug(f"Generated dummy lead: {lead}")
        return lead

    def preprocess_synapse_for_request(self, axon: bt.AxonInfo, synapse: bt.Synapse, timeout: float) -> bt.Synapse:
        """Override preprocessing to use mock wallet's hotkey."""
        bt.logging.debug(f"Preprocessing synapse for axon: {axon.hotkey}")
        synapse.dendrite = bt.TerminalInfo(
            hotkey=self._wallet.hotkey.ss58_address,
            version=600,  # Static version for mock mode
            ip="0.0.0.0",
            port=0,
            process_time=0.0,
            status_code=200,
            status_message="OK"
        )
        synapse.axon = bt.TerminalInfo(
            hotkey=axon.hotkey,
            version=600,  # Static version for mock mode
            ip=axon.ip,
            port=axon.port,
            process_time=0.0,
            status_code=200,
            status_message="OK"
        )
        bt.logging.debug("Synapse preprocessing complete")
        return synapse

    async def forward(
        self,
        axons: List[bt.AxonInfo],
        synapse: bt.Synapse = LeadRequest(num_leads=1),
        timeout: float = 30,  # Reduced timeout
        deserialize: bool = True,
        run_async: bool = True,
        streaming: bool = False,
    ) -> List[LeadRequest]:
        """Simulate miner responses to LeadRequest queries."""
        bt.logging.debug(f"Starting forward with {len(axons)} axons, num_leads={synapse.num_leads}")
        if streaming:
            raise NotImplementedError("Streaming not implemented yet.")
        if not isinstance(synapse, LeadRequest):
            raise ValueError("Synapse must be a LeadRequest for LeadPoet subnet")

        async def query_axon(axon: bt.AxonInfo) -> LeadRequest:
            """Simulate a single miner's response."""
            bt.logging.debug(f"Querying axon: {axon.hotkey}")
            start_time = time.time()
            response = synapse.copy()
            try:
                response = self.preprocess_synapse_for_request(axon, response, timeout)

                # Simulate processing time (capped at 5s for testing)
                process_time = random.uniform(0.5, min(5, timeout))
                bt.logging.debug(f"Simulating process time: {process_time:.2f}s")
                await asyncio.sleep(process_time)  # Simulate network delay

                if process_time >= timeout:
                    bt.logging.warning(f"Timeout for axon {axon.hotkey}")
                    response.leads = []
                    response.dendrite.status_code = 408
                    response.dendrite.status_message = "Timeout"
                    response.dendrite.process_time = str(timeout)
                else:
                    # Generate leads
                    if self.use_open_source and get_leads is not None:
                        bt.logging.debug("Attempting open-source lead generation")
                        try:
                            leads = await get_leads(synapse.num_leads, synapse.industry, synapse.region)
                            # Normalize lead format to match expected keys
                            normalized_leads = [
                                {
                                    "Business": lead.get("business", "Unknown"),
                                    "Owner Full name": lead.get("owner_full_name", "Unknown"),
                                    "First": lead.get("first_name", "Unknown"),
                                    "Last": lead.get("last_name", "Unknown"),
                                    "Owner(s) Email": lead["emails"][0] if lead.get("emails") else "unknown@mock.com",
                                    "LinkedIn": lead.get("linkedin", ""),
                                    "Website": lead.get("website", ""),
                                    "Industry": lead.get("industry", "Unknown"),
                                    "Region": synapse.region or "Global"
                                } for lead in leads
                            ]
                            bt.logging.debug(f"Generated {len(normalized_leads)} open-source leads")
                        except Exception as e:
                            bt.logging.warning(f"Open-source lead generation failed: {e}. Falling back to dummy leads.")
                            normalized_leads = [self.generate_dummy_lead() for _ in range(synapse.num_leads)]
                    else:
                        normalized_leads = []
                        for _ in range(synapse.num_leads):
                            normalized_leads.append(self.generate_dummy_lead())
                        bt.logging.debug(f"Generated {len(normalized_leads)} dummy leads")

                    response.leads = normalized_leads
                    response.dendrite.status_code = 200
                    response.dendrite.status_message = "OK"
                    response.dendrite.process_time = str(process_time)

                if deserialize:
                    bt.logging.debug("Deserializing response leads")
                    response.leads = response.deserialize()
            except Exception as e:
                bt.logging.error(f"Error in query_axon for {axon.hotkey}: {e}")
                response.leads = []
                response.dendrite.status_code = 500
                response.dendrite.status_message = f"Error: {str(e)}"
                response.dendrite.process_time = str(time.time() - start_time)

            bt.logging.debug(f"Query complete for axon {axon.hotkey}")
            return response

        try:
            responses = await asyncio.gather(*(query_axon(axon) for axon in axons), return_exceptions=True)
            bt.logging.debug(f"Received {len(responses)} responses")
            # Filter out exceptions
            valid_responses = []
            for i, resp in enumerate(responses):
                if isinstance(resp, Exception):
                    bt.logging.error(f"Exception in response from axon {axons[i].hotkey}: {resp}")
                    continue
                valid_responses.append(resp)
            return valid_responses
        except Exception as e:
            bt.logging.error(f"Error in forward: {e}")
            return []

    def __str__(self) -> str:
        return f"MockDendrite({self._wallet.hotkey.ss58_address}, use_open_source={self.use_open_source})"

# Example usage for testing
async def test_mock_leadpoet():
    """Test the mock environment for LeadPoet subnet."""
    wallet = bt.wallet()  # Mock wallet
    subtensor = MockSubtensor(netuid=33, n=5, wallet=wallet)
    metagraph = MockMetagraph(netuid=33, subtensor=subtensor)
    dendrite = MockDendrite(wallet=wallet, use_open_source=True)

    # Simulate a validator querying miners
    axons = metagraph.axons[1:]  # Exclude validator (uid=0)
    request = LeadRequest(num_leads=10, industry="Tech & AI", region="US")
    responses = await dendrite.forward(axons, request, timeout=30, deserialize=True)

    bt.logging.info(f"Received {len(responses)} responses:")
    for i, response in enumerate(responses):
        bt.logging.info(f"Miner {i+1}: Status {response.dendrite.status_code}, {len(response.leads)} leads")
        if response.leads:
            bt.logging.debug(f"Sample lead: {response.leads[0]}")

    # Simulate post-approval check (assuming automated_checks.py is available)
    if auto_check_leads and responses[0].leads:
        report = await auto_check_leads(responses[0].leads)
        valid_count = sum(1 for entry in report if entry["status"] == "Valid")
        bt.logging.info(f"Post-approval check: {valid_count}/{len(responses[0].leads)} valid")

if __name__ == "__main__":
    bt.logging.set_trace(True)
    asyncio.run(test_mock_leadpoet())