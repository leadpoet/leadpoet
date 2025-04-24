# Leadpoet/api/leadpoet_api.py

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
    """API for customers to request leads from the LeadPoet subnet."""
    def __init__(self, wallet: "bt.wallet", netuid: int = 343, subtensor_network: str = "test", mock: bool = False):
        """
        Initializes the LeadPoetAPI with a wallet, netuid, subtensor network, and mock mode.
        Args:
            wallet (bt.wallet): The customer's wallet.
            netuid (int, optional): The network UID. Defaults to 343.
            subtensor_network (str, optional): Subtensor network name. Defaults to "test".
            mock (bool, optional): Run in mock mode. Defaults to False.
        """
        self.wallet = wallet
        self.netuid = netuid
        self.subtensor_network = subtensor_network
        self.mock = mock
        self.name = "leadpoet"
        # Explicitly set mock config
        if mock:
            bt.config().mock = True
            bt.logging.debug(f"Explicitly set bt.config().mock to {bt.config().mock}")
        self.dendrite = MockDendrite(wallet=wallet, use_open_source=True) if mock else bt.dendrite(wallet=wallet)
        bt.logging.debug(f"Initialized dendrite: {self.dendrite.__class__.__name__}")
        if mock:
            subtensor = MockSubtensor(netuid=netuid, wallet=wallet)
            self.metagraph = bt.metagraph(netuid=netuid, network="mock", subtensor=subtensor)
        else:
            self.metagraph = bt.metagraph(netuid=netuid)
        bt.logging.info(f"Initialized LeadPoetAPI with netuid: {self.netuid}, subtensor_network: {self.subtensor_network}, mock: {self.mock}, bt.config().mock: {bt.config().mock}")

    def prepare_synapse(self, num_leads: int, industry: Optional[str] = None, region: Optional[str] = None) -> LeadRequest:
        """Prepares a LeadRequest synapse."""
        if not 1 <= num_leads <= 100:
            raise ValueError("num_leads must be between 1 and 100")
        synapse = LeadRequest(num_leads=num_leads, industry=industry, region=region)
        bt.logging.debug(f"Prepared LeadRequest: num_leads={num_leads}, industry={industry}, region={region}")
        return synapse

    def process_responses(self, responses: List[Union["bt.Synapse", Any]]) -> List[Dict]:
        """Processes miner responses, extracting valid leads."""
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
        """Retrieve leads by querying miners, validating, and checking them."""
        bt.logging.info(f"Requesting {num_leads} leads, industry={industry}, region={region}")
        max_retries = 3
        attempt = 1

        while attempt <= max_retries:
            bt.logging.info(f"Attempt {attempt}/{max_retries} to fetch leads")
            
            # Query miners, passing mock flag explicitly
            axons = await get_query_api_axons(self.wallet, self.metagraph, n=0.1, timeout=5, mock=self.mock)
            if not axons:
                bt.logging.error("No available miners to query. Ensure miners are running.")
                raise RuntimeError("No active miners available")

            synapse = self.prepare_synapse(num_leads, industry, region)
            responses = await self.dendrite(
                axons=axons,
                synapse=synapse,
                timeout=30,
                deserialize=True
            )
            leads = self.process_responses(responses)
            
            if not leads:
                bt.logging.error("No valid leads received from miners.")
                raise RuntimeError("No valid leads returned by miners")

            # Validate leads using os_validator_model
            validation = await validate_lead_list(leads, industry=industry)
            score = validation["score"] / 100.0
            bt.logging.info(f"Lead validation score: {score}")

            if score < 0.9:
                bt.logging.warning(f"Lead batch rejected: score {score} below threshold (0.9)")
                attempt += 1
                if attempt > max_retries:
                    bt.logging.error(f"Max retries ({max_retries}) reached, no valid lead batch found.")
                    return []
                continue

            # Run automated checks
            check_report = await auto_check_leads(leads)
            valid_count = sum(1 for entry in check_report if entry["status"] == "Valid")
            check_score = valid_count / len(leads) if leads else 0
            bt.logging.info(f"Automated check score: {check_score}")

            if check_score < 0.9:
                bt.logging.warning(f"Lead batch failed automated checks: score {check_score} below threshold (0.9)")
                attempt += 1
                if attempt > max_retries:
                    bt.logging.error(f"Max retries ({max_retries}) reached, no valid lead batch found.")
                    return []
                continue

            # Add to pool and return leads
            from Leadpoet.base.utils.pool import add_to_pool
            add_to_pool(leads)
            bt.logging.info(f"Added {len(leads)} approved leads to pool")
            
            # Filter leads by industry and region (if specified) and trim to num_leads
            filtered_leads = leads
            if industry:
                filtered_leads = [lead for lead in filtered_leads if lead.get("Industry") == industry]
            if region:
                filtered_leads = [lead for lead in filtered_leads if lead.get("Region") == region]
            filtered_leads = filtered_leads[:num_leads]
            
            bt.logging.info(f"Served {len(filtered_leads)} leads to client")
            return filtered_leads

        return []

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

    # Explicitly set mock config based on args
    if args.mock:
        bt.config().mock = True
        bt.logging.debug(f"Set bt.config().mock to {bt.config().mock} based on --mock flag")

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
        industry = input("Enter industry (or press Enter to skip): ").strip() or None
        if industry is None or industry in VALID_INDUSTRIES:
            break
        print(f"Invalid industry. Please choose from: {', '.join(VALID_INDUSTRIES)}")
    region = input("Enter region (e.g., 'US') or press Enter to skip: ").strip() or None

    async def fetch_leads():
        leads = await api.get_leads(num_leads, industry, region)
        print(f"Retrieved {len(leads)} leads:")
        for i, lead in enumerate(leads, 1):
            print(f"Lead {i}: {lead}")

    asyncio.run(fetch_leads())

if __name__ == "__main__":
    main()
