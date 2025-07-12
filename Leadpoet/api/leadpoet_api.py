import bittensor as bt
from typing import List, Optional, Union, Any, Dict
import asyncio
import argparse
from Leadpoet.protocol import LeadRequest
from Leadpoet.api.get_query_axons import get_query_api_axons
from Leadpoet.mock import MockDendrite, MockWallet, MockSubtensor
from validator_models.os_validator_model import validate_lead_list
from validator_models.automated_checks import validate_lead_list as auto_check_leads
from miner_models.get_leads import VALID_INDUSTRIES
import logging as _py_logging
import aiohttp
import socket

class LeadPoetAPI:
    def __init__(self, wallet: "bt.wallet", netuid: int = 343, subtensor_network: str = "test", mock: bool = False):
        self.wallet = wallet
        self.netuid = netuid
        self.subtensor_network = subtensor_network
        self.mock = mock
        self.name = "leadpoet"
        if mock:
            bt.config().mock = True
            bt.logging.debug(f"Explicitly set bt.config().mock to {bt.config().mock}")
        self.dendrite = MockDendrite(wallet=wallet, use_open_source=True) if mock else bt.dendrite(wallet=wallet)
        bt.logging.debug(f"Initialized dendrite: {self.dendrite.__class__.__name__}")
        if mock:
            bt.logging.debug("Initializing MockSubtensor for mock mode")
            subtensor = MockSubtensor(netuid=netuid, wallet=wallet, network="mock")
            self.metagraph = bt.metagraph(netuid=netuid, network="mock", subtensor=subtensor)
        else:
            bt.logging.debug(f"Initializing Subtensor with network: {subtensor_network}")
            config = bt.config()
            config.subtensor.network = subtensor_network
            subtensor = bt.subtensor(network=subtensor_network, config=config)
            self.metagraph = bt.metagraph(netuid=netuid, network=subtensor_network, subtensor=subtensor)
        bt.logging.info(f"Initialized LeadPoetAPI with netuid: {self.netuid}, subtensor_network: {self.subtensor_network}, mock: {self.mock}")

    def find_validator_port(self) -> Optional[int]:
        """Find the validator port by trying common ports."""
        common_ports = [8093, 8101, 8094, 8095, 8096, 8097, 8098, 8099, 8100, 8102]
        
        for port in common_ports:
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.settimeout(1)
                    result = s.connect_ex(('localhost', port))
                    if result == 0:
                        bt.logging.info(f"Found validator on port {port}")
                        return port
            except Exception:
                continue
        
        bt.logging.warning("Could not find validator port automatically")
        return None

    def prepare_synapse(self, num_leads: int, industry: Optional[str] = None, region: Optional[str] = None) -> LeadRequest:
        if not 1 <= num_leads <= 100:
            raise ValueError("num_leads must be between 1 and 100")
        synapse = LeadRequest(num_leads=num_leads, industry=industry, region=region)
        bt.logging.debug(f"Prepared LeadRequest: num_leads={num_leads}, industry={industry}, region={region}")
        return synapse

    def process_responses(self, responses: List[Union["bt.Synapse", Any]]) -> List[Dict]:
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

    async def get_leads(self, num_leads: int, business_desc: str) -> List[Dict]:
        bt.logging.info(f"Requesting {num_leads} leads, desc='{business_desc[:40]}…'")
        max_retries = 3
        attempt = 1

        while attempt <= max_retries:
            bt.logging.info(f"Attempt {attempt}/{max_retries} to fetch leads")
            try:
                # Find validator port
                validator_port = self.find_validator_port()
                if not validator_port:
                    bt.logging.error("Could not find validator port")
                    attempt += 1
                    continue
                
                # Call validator API
                validator_url = f"http://localhost:{validator_port}/api/leads"
                request_data = {"num_leads": num_leads,
                                "business_desc": business_desc}
                
                bt.logging.info(f"Calling validator API at {validator_url}")
                
                async with aiohttp.ClientSession() as session:
                    async with session.post(validator_url, json=request_data, timeout=30) as response:
                        if response.status == 200:
                            data = await response.json()
                            leads = data.get("leads", [])
                            bt.logging.info(f"Received {len(leads)} leads from validator")
                            return leads
                        else:
                            error_text = await response.text()
                            bt.logging.error(f"Validator API error: {response.status} - {error_text}")
                            attempt += 1
                            continue

            except aiohttp.ClientConnectorError as e:
                bt.logging.error(f"Connection error to validator: {e}")
                attempt += 1
                continue
            except Exception as e:
                bt.logging.error(f"Error calling validator API: {e}")
                attempt += 1
                if attempt > max_retries:
                    bt.logging.error(f"Max retries ({max_retries}) reached, no valid leads found.")
                    return []
                continue

        return []

    async def submit_feedback(self, leads: List[Dict], feedback_score: float):
        from neurons.validator import Validator
        validator = Validator(config=bt.config())
        await validator.handle_buyer_feedback(leads, feedback_score)
        bt.logging.info(f"Submitted feedback score {feedback_score} for leads")

def main():
    parser = argparse.ArgumentParser(description="LeadPoet API Client")
    parser.add_argument("--wallet_name", type=str, required=True, help="Name of the wallet")
    parser.add_argument("--wallet_hotkey", type=str, required=True, help="Hotkey of the wallet")
    parser.add_argument("--netuid", type=int, default=343, help="Network UID")
    parser.add_argument("--subtensor_network", type=str, default="test", help="Subtensor network")
    parser.add_argument("--mock", action="store_true", help="Run in mock mode")
    parser.add_argument("--logging_trace", action="store_true", help="Enable trace-level logging")
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # QUIET-MODE: unless the user explicitly supplied --logging_trace,
    # drop the global logging level to WARNING.  This suppresses all the
    # huge DEBUG / INFO payload printed by bittensor + mock objects.
    # ------------------------------------------------------------------
    if not args.logging_trace:
        _py_logging.getLogger().setLevel(_py_logging.WARNING)
        # If bittensor's helper is present, drop its trace flag as well.
        if getattr(bt.logging, "set_trace", None):
            bt.logging.set_trace(False)

    if args.logging_trace:
        bt.logging.set_trace(True)      # unchanged – trace beats quiet mode

    if args.mock:
        bt.config().mock = True
        bt.logging.debug(f"Set bt.config().mock to {bt.config().mock}")

    wallet = MockWallet(name=args.wallet_name, hotkey=args.wallet_hotkey) if args.mock else bt.wallet(name=args.wallet_name, hotkey=args.wallet_hotkey)

    api = LeadPoetAPI(
        wallet=wallet,
        netuid=args.netuid,
        subtensor_network=args.subtensor_network,
        mock=args.mock
    )

    print("Welcome to LeadPoet API")
    num_leads     = int(input("Enter number of leads (1-100): "))
    business_desc = input("Describe your business & ideal customer: ").strip()

    # build new request
    payload = {"num_leads": num_leads,
               "business_desc": business_desc}

    async def fetch_leads():
        leads = await api.get_leads(num_leads, business_desc)
        print(f"Retrieved {len(leads)} leads:")
        for i, lead in enumerate(leads, 1):
            print(f"Lead {i}: {lead}")
        
        # Handle empty feedback input
        while True:
            feedback_input = input("Enter feedback score (0-10): ").strip()
            if feedback_input == "":
                print("Please enter a valid feedback score.")
                continue
            try:
                feedback = float(feedback_input)
                if 0 <= feedback <= 10:
                    break
                else:
                    print("Feedback score must be between 0 and 10.")
            except ValueError:
                print("Please enter a valid number.")
        
        await api.submit_feedback(leads, feedback)

    asyncio.run(fetch_leads())

if __name__ == "__main__":
    main()
