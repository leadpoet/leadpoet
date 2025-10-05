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

    def find_validator_port(self, check_http=False):
        """
        Find which port the validator HTTP server is running on.
        NOW OPTIONAL - only needed for legacy direct queries, not for broadcast flow.
        
        Args:
            check_http: If True, verify HTTP server is responding
        """
        if not check_http:
            # For broadcast flow, we don't need validator HTTP endpoints
            return None
            
        # Common ports to check (prioritize higher ports to avoid gRPC)
        common_ports = [8094, 8095, 8096, 8097, 8098, 8099, 8100, 8093]
        
        for port in common_ports:
            if self.is_port_running_http(port):
                bt.logging.info(f"✅ Found validator HTTP server on port {port}")
                return port
        
        bt.logging.warning("⚠️  Could not find validator HTTP server port")
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
        Submit API request using broadcast mechanism - DIRECTLY to Firestore.
        
        Flow:
        1. Write request directly to Firestore (broadcasts to ALL validators and miners simultaneously)
        2. Wait for validators to rank leads
        3. Fetch validator rankings from Firestore
        4. Calculate consensus client-side
        5. Return top N leads
        """
        import time
        import uuid
        from datetime import datetime, timezone
        
        # Generate unique request ID
        request_id = str(uuid.uuid4())[:16]
        
        print(f"🔴 Broadcasting request {request_id} to Firestore...")
        print(f"   num_leads: {num_leads}")
        print(f"   business_desc: {business_desc}")
        
        # Broadcast to ALL validators and miners via Firestore
        from Leadpoet.utils.cloud_db import broadcast_api_request
        success = broadcast_api_request(
            wallet=self.wallet,
            request_id=request_id,
            num_leads=num_leads,
            business_desc=business_desc,
            client_id=self.wallet.hotkey.ss58_address
        )
        
        if not success:
            bt.logging.error("❌ Failed to broadcast API request to Firestore!")
            return []
        
        print(f"✅ Request {request_id} written to Firestore successfully!")
        print(f"⏳ Waiting for validators to rank leads (up to 400s)...")

        try:
            # ══════════════════════════════════════════════════════════
            # STEP 2: Poll Firestore for validator rankings
            # ══════════════════════════════════════════════════════════
            from Leadpoet.utils.cloud_db import fetch_validator_rankings, get_broadcast_status
            
            MAX_ATTEMPTS = 80
            SLEEP_SEC = 5
            total_wait = MAX_ATTEMPTS * SLEEP_SEC
            
            print(f"📊 Multiple validators are independently ranking leads...")
            
            for attempt in range(1, MAX_ATTEMPTS + 1):
                try:
                    # Fetch validator rankings directly from Firestore
                    validator_rankings = fetch_validator_rankings(request_id, timeout_sec=2)
                    
                    validators_submitted = len(validator_rankings)
                    
                    # Calculate elapsed time
                    status_data = get_broadcast_status(request_id)
                    
                    request_time = status_data.get("created_at", "")
                    timeout_reached = False
                    elapsed = 0
                    
                    if request_time:
                        try:
                            req_dt = datetime.fromisoformat(request_time.replace('Z', '+00:00'))
                            elapsed = (datetime.now(timezone.utc) - req_dt).total_seconds()
                            timeout_reached = elapsed > 90
                        except:
                            pass
                    
                    # Check if we have enough validators to calculate consensus
                    if validators_submitted > 0 and (timeout_reached or validators_submitted >= 2 or elapsed > 60):
                        # ═══════════════════════════════════════════════════════
                        # STEP 3: CONSENSUS CALCULATION (CLIENT-SIDE)
                        # ═══════════════════════════════════════════════════════
                        from Leadpoet.validator.consensus import calculate_consensus_ranking
                        
                        print(f"\n🔮 CALCULATING CONSENSUS from {validators_submitted} validator(s)...")
                        
                        final_leads, metadata = calculate_consensus_ranking(
                            validator_rankings=validator_rankings,
                            num_leads_requested=num_leads,
                            min_validators=1
                        )
                        
                        if final_leads:
                            print(f"✅ CONSENSUS COMPLETE!")
                            print(f"   📊 {metadata['num_validators']} validator(s) participated")
                            print(f"   ⚖️  Total validator trust: {metadata['total_trust']:.4f}")
                            print(f"   🎯 Received {len(final_leads)} consensus-ranked lead(s)")
                            
                            return final_leads
                    
                    elif timeout_reached and validators_submitted == 0:
                        bt.logging.error("Request timed out with no validator responses")
                        return []
                    
                    else:
                        # Still waiting
                        if attempt % 6 == 0:
                            print(f"⏳ Still processing... ({validators_submitted} validators submitted, {elapsed:.0f}s elapsed)")
                
                except Exception as e:
                    # Print full error with traceback
                    import traceback
                    error_details = traceback.format_exc()
                    bt.logging.warning(f"Error polling status: {e}\n{error_details}")
                    print(f"❌ Polling error: {e}")
                
                await asyncio.sleep(SLEEP_SEC)
            
            bt.logging.error(f"Timed out waiting for validators after {total_wait}s")
            return []
            
        except Exception as e:
            bt.logging.error(f"Error in broadcast request: {e}")
            import traceback
            bt.logging.error(traceback.format_exc())
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
        try:
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
        except Exception as e:
            print(f"❌ Error occurred: {e}")
            print("This could be due to network connectivity issues or API unavailability.")
            print("Please check your connection and try again.")
            return  # Return gracefully instead of crashing

    # Add a retry loop for the main API interaction
    while True:
        try:
            asyncio.run(fetch_leads())
            break  # Exit if successful
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"❌ Connection failed: {e}")
            retry = input("Would you like to retry? (y/n): ").strip().lower()
            if retry != 'y':
                break
            print("Retrying...")

if __name__ == "__main__":
    main()
