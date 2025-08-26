import bittensor as bt
from typing import List, Optional, Union, Any, Dict
import asyncio
import argparse
from Leadpoet.protocol import LeadRequest
from Leadpoet.api.get_query_axons import get_query_api_axons
from validator_models.automated_checks import validate_lead_list as auto_check_leads
from miner_models.get_leads import VALID_INDUSTRIES
import logging as _py_logging
import aiohttp
import socket
import requests
import base64
import json
import time
import os
from Leadpoet.utils.cloud_db import (
     push_curation_request, fetch_curation_result)
import uuid

# Cloud API configuration
CLOUD_API_URL = os.getenv("LEAD_API", "https://leadpoet-api-511161415764.us-central1.run.app")
SUBNET_UID = 401  # Your subnet ID

class CloudDatabase:
    """Centralized database operations for LeadPoet subnet"""
    
    def __init__(self, wallet: bt.wallet = None):
        self.wallet = wallet
        self.api_url = CLOUD_API_URL
        
    def _verify_miner_registration(self, hotkey: str) -> bool:
        """Verify if a hotkey is registered as a miner on the subnet"""
        try:
            subtensor = bt.subtensor(network="test")
            metagraph = subtensor.metagraph(netuid=SUBNET_UID)
            
            # Check if hotkey exists in metagraph
            if hotkey in metagraph.hotkeys:
                uid = metagraph.hotkeys.index(hotkey)
                # For miners, we just check they're registered (not validator permit)
                return uid < metagraph.n
            return False
        except Exception as e:
            bt.logging.error(f"Error verifying miner registration: {e}")
            return False
    
    def _verify_validator_registration(self, hotkey: str) -> bool:
        """Verify if a hotkey is registered as a validator on the subnet"""
        try:
            subtensor = bt.subtensor(network="test")
            metagraph = subtensor.metagraph(netuid=SUBNET_UID)
            
            if hotkey in metagraph.hotkeys:
                uid = metagraph.hotkeys.index(hotkey)
                # Check if they have validator permit
                return metagraph.validator_permit[uid].item()
            return False
        except Exception as e:
            bt.logging.error(f"Error verifying validator registration: {e}")
            return False
    
    def read_leads(self, limit: int = 100) -> List[Dict]:
        """Read leads from cloud database (for miners)"""
        if not self.wallet:
            raise ValueError("Wallet required for authenticated reads")
            
        # Verify miner registration
        if not self._verify_miner_registration(self.wallet.hotkey.ss58_address):
            raise ValueError("Wallet not registered as miner on subnet")
        
        try:
            response = requests.get(
                f"{self.api_url}/leads",
                params={"limit": limit},
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            bt.logging.error(f"Failed to read leads from cloud: {e}")
            return []
    
    def write_leads(self, leads: List[Dict]) -> bool:
        """Write validated leads to cloud database (for validators only)"""
        if not self.wallet:
            raise ValueError("Wallet required for authenticated writes")
            
        # Verify validator registration
        if not self._verify_validator_registration(self.wallet.hotkey.ss58_address):
            raise ValueError("Wallet not registered as validator on subnet")
        
        if not leads:
            return True
            
        try:
            # Create signed payload
            timestamp = str(int(time.time()) // 300)  # 5-min window
            payload = (timestamp + json.dumps(leads, sort_keys=True)).encode()
            signature = base64.b64encode(self.wallet.sign(payload)).decode()
            
            # Prepare request
            body = {
                "wallet": self.wallet.hotkey.ss58_address,
                "signature": signature,
                "leads": leads
            }
            
            # Send to cloud API
            response = requests.post(
                f"{self.api_url}/leads",
                json=body,
                timeout=30
            )
            response.raise_for_status()
            
            result = response.json()
            bt.logging.info(f"Successfully wrote {result.get('stored', 0)} leads to cloud database")
            return True
            
        except requests.exceptions.RequestException as e:
            bt.logging.error(f"Failed to write leads to cloud: {e}")
            return False
        except Exception as e:
            bt.logging.error(f"Error in write_leads: {e}")
            return False

# Convenience functions for easy integration
def get_cloud_leads(wallet: bt.wallet, limit: int = 100) -> List[Dict]:
    """Get leads from cloud database"""
    db = CloudDatabase(wallet)
    return db.read_leads(limit)

def save_leads_to_cloud(wallet: bt.wallet, leads: List[Dict]) -> bool:
    """Save leads to cloud database"""
    db = CloudDatabase(wallet)
    return db.write_leads(leads)

class LeadPoetAPI:
    def __init__(
        self,
        wallet: "bt.wallet",
        netuid: int = 401,
        subtensor_network: str = "test",
        validator_host: Optional[str] = None,
        validator_port: Optional[int] = None,
    ):
        self.wallet = wallet
        self.netuid = netuid
        self.subtensor_network = subtensor_network
        self.name = "leadpoet"
        self.validator_host = validator_host   # may be None
        self.validator_port = validator_port   # may be None
        self.dendrite = bt.dendrite(wallet=wallet)
        bt.logging.debug(f"Initialized dendrite: {self.dendrite.__class__.__name__}")
        bt.logging.debug(f"Initializing Subtensor with network: {subtensor_network}")
        config = bt.config()
        # `bt.config()` may not contain a subtensor section yet.
        if not hasattr(config, "subtensor") or config.subtensor is None:
            config.subtensor = bt.Config()
        config.subtensor.network = subtensor_network
        self.subtensor = bt.subtensor(network=subtensor_network, config=config)
        self.metagraph = bt.metagraph(netuid=netuid, network=subtensor_network, subtensor=self.subtensor)
        bt.logging.info(f"Initialized LeadPoetAPI with netuid: {self.netuid}, subtensor_network: {self.subtensor_network}")

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
        """
        Submit the curation request to Cloud-Run and poll for the
        validator’s response.  The Bittensor axon path is bypassed.
        """
        bt.logging.info(f"Requesting {num_leads} leads, desc='{business_desc[:40]}…'")

        # ── 1️⃣  push request to Cloud-Run
        req_id = push_curation_request(
            {"num_leads": num_leads, "business_desc": business_desc}
        )
        bt.logging.info(f"Sent curation request to Cloud-Run: {req_id}")

        # ── 2️⃣  poll for validator result
        MAX_ATTEMPTS = 80      # 80 × 5 s = 400 s
        SLEEP_SEC    = 5
        total_wait   = MAX_ATTEMPTS * SLEEP_SEC
        print(f"⏳ Waiting for validator result (up to {total_wait} s)…")

        for _ in range(MAX_ATTEMPTS):
            res = fetch_curation_result(req_id)
            if res.get("leads"):
                return res["leads"]
            time.sleep(SLEEP_SEC)

        bt.logging.error("Timed-out waiting for validator result")
        return []

    async def _get_leads_via_http(self, num_leads: int, business_desc: str) -> List[Dict]:
        """Legacy HTTP server approach for mock mode"""
        max_retries = 3
        attempt = 1

        while attempt <= max_retries:
            bt.logging.info(f"Attempt {attempt}/{max_retries} to fetch leads")
            try:
                validator_port = self.find_validator_port()
                if not validator_port:
                    bt.logging.error("Could not find validator port")
                    attempt += 1
                    continue
                
                validator_url = f"http://localhost:{validator_port}/api/leads"
                request_data = {"num_leads": num_leads, "business_desc": business_desc}
                
                bt.logging.info(f"Calling validator API at {validator_url}")
                
                async with aiohttp.ClientSession() as session:
                    async with session.post(validator_url, json=request_data, timeout=90) as response:
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

    async def _get_leads_via_bittensor(self, num_leads: int, business_desc: str) -> List[Dict]:
        """Use the actual Bittensor network to get leads"""
        try:
            # Create synapse for the request
            synapse = LeadRequest(
                num_leads=num_leads,
                business_desc=business_desc
            )
            
            # Get available validators from the metagraph
            self.metagraph.sync(subtensor=self.subtensor)
            
            # Debug: Print metagraph info
            print(f" Metagraph info:")
            print(f"   Total axons: {len(self.metagraph.axons)}")
            print(f"   Validator permits: {self.metagraph.validator_permit}")
            print(f"   Axon serving status: {[axon.is_serving for axon in self.metagraph.axons]}")
            
            validator_uids = [uid for uid in range(len(self.metagraph.axons)) 
                            if self.metagraph.validator_permit[uid] and self.metagraph.axons[uid].is_serving]
            
            print(f"   Found validator UIDs: {validator_uids}")
            
            if not validator_uids:
                bt.logging.error("No active validators found on the network")
                return []
            
            # Query the first available validator
            validator_axon = self.metagraph.axons[validator_uids[0]]
            bt.logging.info(f"Querying validator {validator_uids[0]} at {validator_axon.ip}:{validator_axon.port}")
            
            # Send request through Bittensor network  (auto-decode LeadRequest)
            responses = await self.dendrite(
                axons=[validator_axon],
                synapse=synapse,
                timeout=45,
                deserialize=True,
            )

            if responses:
                first = responses[0]
                # bittensor returns a “responses per axon” structure
                if isinstance(first, list):
                    if not first or first[0] is None:       # validator sent nothing
                        bt.logging.error("Validator sent no data")
                        return []
                    response = first[0]
                else:
                    response = first

                if response and response.dendrite.status_code == 200 and response.leads:
                    bt.logging.info(f"Received {len(response.leads)} leads from validator")
                    return response.leads
                else:
                    bt.logging.error(f"Validator response error: {response.dendrite.status_message}")
            else:
                bt.logging.error("No response from validator")
                
        except Exception as e:
            bt.logging.error(f"Error querying Bittensor network: {e}")

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
    parser.add_argument("--validator_host", type=str,
                        help="Public IP or DNS name of the validator (omit for localhost dev)")
    parser.add_argument("--validator_port", type=int,
                        help="Validator HTTP port (defaults to 8093 or auto-scan)")
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

    wallet = bt.wallet(name=args.wallet_name, hotkey=args.wallet_hotkey)

    api = LeadPoetAPI(
        wallet=wallet,
        netuid=args.netuid,
        subtensor_network=args.subtensor_network,
        validator_host=args.validator_host,
        validator_port=args.validator_port,
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
