import time
import asyncio
import threading
import argparse
import traceback
import bittensor as bt
import socket
from Leadpoet.base.miner import BaseMinerNeuron
from Leadpoet.protocol import LeadRequest
from miner_models.get_leads import get_leads
from typing import Union, Tuple

class Miner(BaseMinerNeuron):
    """
    Miner neuron class for the LeadPoet subnet. Generates leads using get_leads.py
    and responds to validator queries.
    """
    def __init__(self, config=None):
        super().__init__(config=config)
        self.use_open_source_lead_model = config.get("use_open_source_lead_model", True) if config else True
        bt.logging.info(f"Using open-source lead model: {self.use_open_source_lead_model}")

    async def forward(self, synapse: LeadRequest) -> LeadRequest:
        """Handles LeadRequest queries by generating leads."""
        bt.logging.debug(f"Received LeadRequest: num_leads={synapse.num_leads}, industry={synapse.industry}, region={synapse.region}")
        start_time = time.time()

        try:
            if self.use_open_source_lead_model:
                leads = await get_leads(synapse.num_leads, synapse.industry, synapse.region)
                bt.logging.debug(f"Generated {len(leads)} leads using open-source model")
            else:
                leads = [
                    {
                        "Business": f"Mock Business {i}",
                        "Owner Full name": f"Owner {i}",
                        "First": f"First {i}",
                        "Last": f"Last {i}",
                        "Owner(s) Email": f"owner{i}@mockleadpoet.com",
                        "LinkedIn": f"https://linkedin.com/in/owner{i}",
                        "Website": f"https://business{i}.com",
                        "Industry": synapse.industry or "Tech & AI",
                        "Region": synapse.region or "US"
                    } for i in range(synapse.num_leads)
                ]
                bt.logging.debug(f"Generated {len(leads)} dummy leads")

            synapse.leads = leads
            synapse.dendrite.status_code = 200
            synapse.dendrite.status_message = "OK"
            synapse.dendrite.process_time = str(time.time() - start_time)
        except Exception as e:
            bt.logging.error(f"Error generating leads: {e}")
            synapse.leads = []
            synapse.dendrite.status_code = 500
            synapse.dendrite.status_message = f"Error: {str(e)}"
            synapse.dendrite.process_time = str(time.time() - start_time)

        return synapse

    def blacklist(self, synapse: LeadRequest) -> Tuple[bool, str]:
        """Blacklist logic for incoming requests. Returns (is_blacklisted, reason)."""
        if self.config.blacklist_force_validator_permit and not self.metagraph.axons[self.uid].is_serving:
            bt.logging.debug(f"Blacklisting non-validator request from {synapse.dendrite.hotkey}")
            return True, f"Non-validator request from {synapse.dendrite.hotkey}"
        if not self.config.blacklist_allow_non_registered and synapse.dendrite.hotkey not in self.metagraph.hotkeys:
            bt.logging.debug(f"Blacklisting non-registered hotkey {synapse.dendrite.hotkey}")
            return True, f"Non-registered hotkey {synapse.dendrite.hotkey}"
        return False, ""

    def priority(self, synapse: LeadRequest) -> float:
        """Assigns priority to requests (default: equal priority)."""
        return 1.0

    def check_port_availability(self, port: int) -> bool:
        """Check if a port is available."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(('0.0.0.0', port))
                return True
            except socket.error:
                return False

    def find_available_port(self, start_port: int, max_attempts: int = 10) -> int:
        """Find an available port starting from start_port."""
        port = start_port
        for _ in range(max_attempts):
            if self.check_port_availability(port):
                return port
            port += 1
        raise RuntimeError(f"No available ports found between {start_port} and {start_port + max_attempts - 1}")

def main():
    """Entry point for the leadpoet command."""
    parser = argparse.ArgumentParser(description="LeadPoet Miner")
    # Add miner-specific arguments from BaseMinerNeuron
    BaseMinerNeuron.add_args(parser)
    # Add additional arguments
    parser.add_argument("--axon_port", type=int, default=8092, help="Port for axon")
    args = parser.parse_args()

    if args.logging_trace:
        bt.logging.set_trace(True)

    # Manually create config object
    config = bt.Config()
    config.wallet = bt.Config()
    config.wallet.name = args.wallet_name
    config.wallet.hotkey = args.wallet_hotkey
    config.netuid = args.netuid
    config.subtensor = bt.Config()
    config.subtensor.network = args.subtensor_network
    config.mock = args.mock
    config.axon = bt.Config()
    config.axon.port = args.axon_port
    config.blacklist = bt.Config()
    config.blacklist.force_validator_permit = args.blacklist_force_validator_permit
    config.blacklist.allow_non_registered = args.blacklist_allow_non_registered
    config.neuron = bt.Config()
    config.neuron.epoch_length = args.neuron_epoch_length
    config.use_open_source_lead_model = args.use_open_source_lead_model

    miner = Miner(config=config)
    try:
        # Find an available port
        config.axon.port = miner.find_available_port(config.axon.port)
        bt.logging.info(f"Using axon port: {config.axon.port}")
    except RuntimeError as e:
        bt.logging.error(str(e))
        return

    with miner:
        while True:
            bt.logging.info(f"Miner running... {time.time()}")
            time.sleep(5)

if __name__ == "__main__":
    main()