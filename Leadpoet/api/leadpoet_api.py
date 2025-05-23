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

    async def get_leads(self, num_leads: int, industry: Optional[str] = None, region: Optional[str] = None) -> List[Dict]:
        bt.logging.info(f"Requesting {num_leads} leads, industry={industry}, region={region}")
        max_retries = 3
        attempt = 1

        while attempt <= max_retries:
            bt.logging.info(f"Attempt {attempt}/{max_retries} to fetch leads")
            try:
                # Get available miners
                axons = await get_query_api_axons(self.wallet, self.metagraph, n=0.1, timeout=5, mock=self.mock)
                if not axons:
                    bt.logging.error("No available miners to query.")
                    raise RuntimeError("No active miners available")

                # Prepare request
                synapse = self.prepare_synapse(num_leads, industry, region)
                
                # Send request to miners
                responses = await self.dendrite(
                    axons=axons,
                    synapse=synapse,
                    timeout=30,
                    deserialize=True
                )
                
                # Process responses
                leads = self.process_responses(responses)
                
                if not leads:
                    bt.logging.error("No valid leads received from miners.")
                    attempt += 1
                    if attempt > max_retries:
                        bt.logging.error(f"Max retries ({max_retries}) reached, no valid leads found.")
                        return []
                    continue

                # Send leads to validator for scoring
                from neurons.validator import Validator
                validator = Validator(config=bt.config())
                validation = await validator.validate_leads(leads, industry=industry)
                score = validation["score"] / 100.0
                bt.logging.info(f"Lead validation score: {score}")

                if score < 0.7:  # Threshold is 0.7 as per your code
                    bt.logging.warning(f"Lead batch rejected: score {score} below threshold (0.7)")
                    attempt += 1
                    if attempt > max_retries:
                        bt.logging.error(f"Max retries ({max_retries}) reached, no valid lead batch found.")
                        return []
                    continue

                # Add valid leads to pool
                from Leadpoet.base.utils.pool import add_to_pool
                add_to_pool(leads)
                bt.logging.info(f"Added {len(leads)} approved leads to pool")
                
                # Filter leads by industry and region
                filtered_leads = leads
                if industry:
                    filtered_leads = [lead for lead in filtered_leads if lead.get("Industry") == industry]
                if region:
                    filtered_leads = [lead for lead in filtered_leads if lead.get("Region") == region]
                filtered_leads = filtered_leads[:num_leads]
                
                bt.logging.info(f"Served {len(filtered_leads)} leads to client")
                return filtered_leads

            except Exception as e:
                bt.logging.error(f"Error in get_leads: {e}")
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

    if args.logging_trace:
        bt.logging.set_trace(True)

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
    num_leads = int(input("Enter number of leads (1-100): "))
    print(f"Available industries: {', '.join(VALID_INDUSTRIES)}")
    while True:
        industry = input("Enter industry: ").strip()
        if not industry or industry in VALID_INDUSTRIES:  # Allow empty industry
            break
        print(f"Invalid industry. Please choose from: {', '.join(VALID_INDUSTRIES)}")
    region = input("Enter region (e.g., 'US'): ").strip() or None  # Allow empty region

    async def fetch_leads():
        leads = await api.get_leads(num_leads, industry, region)
        print(f"Retrieved {len(leads)} leads:")
        for i, lead in enumerate(leads, 1):
            print(f"Lead {i}: {lead}")
        feedback = float(input("Enter feedback score (0-10): "))
        await api.submit_feedback(leads, feedback)

    asyncio.run(fetch_leads())

if __name__ == "__main__":
    main()