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
from typing import Union, Tuple, List, Dict, Optional
from aiohttp import web
import os
import re
import html
import uuid
from datetime import datetime
from Leadpoet.base.utils import queue as lead_queue
from Leadpoet.base.utils import pool as lead_pool
import json
from Leadpoet.base.utils.pool import get_leads_from_pool

class Miner(BaseMinerNeuron):
    def __init__(self, config=None):
        super().__init__(config=config)
        self.use_open_source_lead_model = config.get("use_open_source_lead_model", True) if config else True
        bt.logging.info(f"Using open-source lead model: {self.use_open_source_lead_model}")
        self.app = web.Application()
        self.app.add_routes([web.post('/lead_request', self.handle_lead_request)])
        self.sourcing_mode = True  # Start in sourcing mode
        self.sourcing_lock = asyncio.Lock()  # Lock for switching modes

    async def forward(self, synapse: LeadRequest) -> LeadRequest:
        bt.logging.debug(f"Received lead request: {synapse}")
        start_time = time.time()

        try:
            # Temporarily disable sourcing while curating
            async with self.sourcing_lock:
                self.sourcing_mode = False
                try:
                    # Get leads from pool first
                    curated_leads = get_leads_from_pool(
                        synapse.num_leads,
                        industry=synapse.industry,
                        region=synapse.region
                    )
                    
                    if not curated_leads:
                        bt.logging.info("No leads found in pool, generating new leads")
                        new_leads = await get_leads(synapse.num_leads, synapse.industry, synapse.region)
                        sanitized = [sanitize_prospect(p, self.wallet.hotkey.ss58_address) for p in new_leads]
                        curated_leads = sanitized
                    
                    # Map the fields to match the API format and ensure all required fields are present
                    mapped_leads = []
                    for lead in curated_leads:
                        # Map the fields correctly - note the field name changes
                        mapped_lead = {
                            "email": lead.get("owner_email", ""),
                            "Business": lead.get("business", ""),
                            "Owner Full name": lead.get("owner_full_name", ""),
                            "First": lead.get("first", ""),
                            "Last": lead.get("last", ""),
                            "LinkedIn": lead.get("linkedin", ""),
                            "Website": lead.get("website", ""),
                            "Industry": lead.get("industry", ""),
                            "Region": lead.get("region", ""),
                            "conversion_score": lead.get("conversion_score", 1.0)
                        }
                        # Only include leads that have all required fields
                        if all(mapped_lead.get(field) for field in ["email", "Business", "Owner Full name"]):
                            mapped_leads.append(mapped_lead)
                    
                    if not mapped_leads:
                        bt.logging.warning("No valid leads found in pool after mapping")
                        synapse.leads = []
                        synapse.dendrite.status_code = 404
                        synapse.dendrite.status_message = "No valid leads found matching criteria"
                        synapse.dendrite.process_time = str(time.time() - start_time)
                        return synapse
                    
                    bt.logging.info(f"Returning {len(mapped_leads)} leads to request")
                    synapse.leads = mapped_leads
                    synapse.dendrite.status_code = 200
                    synapse.dendrite.status_message = "OK"
                    synapse.dendrite.process_time = str(time.time() - start_time)
                    
                finally:
                    # Re-enable sourcing after curation
                    self.sourcing_mode = True
                
        except Exception as e:
            bt.logging.error(f"Error curating leads: {e}")
            synapse.leads = []
            synapse.dendrite.status_code = 500
            synapse.dendrite.status_message = f"Error: {str(e)}"
            synapse.dendrite.process_time = str(time.time() - start_time)
            # Ensure sourcing is re-enabled even on error
            self.sourcing_mode = True

        return synapse

    async def handle_lead_request(self, request):
        bt.logging.info(f"Received HTTP lead request: {await request.text()}")
        try:
            data = await request.json()
            num_leads = data.get("num_leads", 1)
            industry = data.get("industry")
            region = data.get("region")
            
            # Get leads from pool first
            curated_leads = get_leads_from_pool(
                num_leads,
                industry=industry,
                region=region
            )
            
            if not curated_leads:
                bt.logging.info("No leads found in pool, generating new leads")
                new_leads = await get_leads(num_leads, industry, region)
                sanitized = [sanitize_prospect(p, self.wallet.hotkey.ss58_address) for p in new_leads]
                curated_leads = sanitized
            
            # Map the fields - FIXED VERSION
            mapped_leads = []
            for lead in curated_leads:
                # Map the fields correctly using the same keys as stored in pool
                mapped_lead = {
                    "email": lead.get("owner_email", ""),
                    "Business": lead.get("business", ""),
                    "Owner Full name": lead.get("owner_full_name", ""),
                    "First": lead.get("first", ""),
                    "Last": lead.get("last", ""),
                    "LinkedIn": lead.get("linkedin", ""),
                    "Website": lead.get("website", ""),
                    "Industry": lead.get("industry", ""),
                    "Region": lead.get("region", ""),
                    "conversion_score": lead.get("conversion_score", 1.0)
                }
                
                # Debug log to see what's happening
                bt.logging.debug(f"Original lead: {lead}")
                bt.logging.debug(f"Mapped lead: {mapped_lead}")
                
                # Only include leads that have all required fields
                if all(mapped_lead.get(field) for field in ["email", "Business", "Owner Full name"]):
                    mapped_leads.append(mapped_lead)
                else:
                    bt.logging.warning(f"Lead missing required fields: {mapped_lead}")
            
            if not mapped_leads:
                bt.logging.warning("No valid leads found in pool after mapping")
                return web.json_response({
                    "leads": [],
                    "status_code": 404,
                    "status_message": "No valid leads found matching criteria",
                    "process_time": "0"
                }, status=404)
            
            bt.logging.info(f"Returning {len(mapped_leads)} leads to HTTP request")
            return web.json_response({
                "leads": mapped_leads,
                "status_code": 200,
                "status_message": "OK",
                "process_time": "0"
            })
        except Exception as e:
            bt.logging.error(f"Error in HTTP lead request: {e}")
            return web.json_response({
                "leads": [],
                "status_code": 500,
                "status_message": f"Error: {str(e)}",
                "process_time": "0"
            }, status=500)

    def blacklist(self, synapse: LeadRequest) -> Tuple[bool, str]:
        if self.config.blacklist_force_validator_permit and not self.metagraph.axons[self.uid].is_serving:
            bt.logging.debug(f"Blacklisting non-validator request from {synapse.dendrite.hotkey}")
            return True, f"Non-validator request from {synapse.dendrite.hotkey}"
        if not self.config.blacklist_allow_non_registered and synapse.dendrite.hotkey not in self.metagraph.hotkeys:
            bt.logging.debug(f"Blacklisting non-registered hotkey {synapse.dendrite.hotkey}")
            return True, f"Non-registered hotkey {synapse.dendrite.hotkey}"
        return False, ""

    def priority(self, synapse: LeadRequest) -> float:
        return 1.0

    def check_port_availability(self, port: int) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(('0.0.0.0', port))
                return True
            except socket.error:
                return False

    def find_available_port(self, start_port: int, max_attempts: int = 10) -> int:
        port = start_port
        for _ in range(max_attempts):
            if self.check_port_availability(port):
                return port
            port += 1
        raise RuntimeError(f"No available ports found between {start_port} and {start_port + max_attempts - 1}")

    async def start_http_server(self):
        runner = web.AppRunner(self.app)
        await runner.setup()
        site = web.TCPSite(runner, '0.0.0.0', self.config.axon.port)
        await site.start()
        bt.logging.info(f"HTTP server started on port {self.config.axon.port}")

DATA_DIR = "data"
SOURCING_LOG = os.path.join(DATA_DIR, "sourcing_logs.json")
MINERS_LOG = os.path.join(DATA_DIR, "miners.json")
LEADS_FILE = os.path.join(DATA_DIR, "leads.json")

def ensure_data_files():
    """Ensure data directory and required JSON files exist."""
    os.makedirs(DATA_DIR, exist_ok=True)
    for file in [SOURCING_LOG, MINERS_LOG, LEADS_FILE]:
        if not os.path.exists(file):
            with open(file, "w") as f:
                json.dump([], f)

def sanitize_prospect(prospect, miner_hotkey=None):
    """Sanitize and validate prospect fields."""
    def strip_html(s):
        return re.sub('<.*?>', '', html.unescape(str(s))) if isinstance(s, str) else s
    def valid_url(url):
        return bool(re.match(r"^https?://[^\s]+$", url))
    
    # Special handling for email field
    email = prospect.get("Owner(s) Email", "")
    sanitized = {
        "business": strip_html(prospect.get("Business", "")),
        "owner_full_name": strip_html(prospect.get("Owner Full name", "")),
        "first": strip_html(prospect.get("First", "")),
        "last": strip_html(prospect.get("Last", "")),
        "owner_email": strip_html(email),  # Use consistent field name
        "linkedin": strip_html(prospect.get("LinkedIn", "")),
        "website": strip_html(prospect.get("Website", "")),
        "industry": strip_html(prospect.get("Industry", "")),
        "region": strip_html(prospect.get("Region", "")),
        "source": miner_hotkey  # Add source field
    }
    
    if not valid_url(sanitized["linkedin"]):
        sanitized["linkedin"] = ""
    if not valid_url(sanitized["website"]):
        sanitized["website"] = ""
    
    sanitized["id"] = str(uuid.uuid4())
    now = datetime.utcnow().isoformat()
    sanitized["created_at"] = now
    sanitized["updated_at"] = now
    sanitized["conversion_score"] = 0.0
    return sanitized

def log_sourcing(hotkey, num_prospects):
    entry = {"timestamp": datetime.utcnow().isoformat(), "hotkey": hotkey, "num_prospects": num_prospects}
    with threading.Lock():
        if not os.path.exists(SOURCING_LOG):
            logs = []
        else:
            with open(SOURCING_LOG, "r") as f:
                try:
                    logs = json.load(f)
                except Exception:
                    logs = []
        logs.append(entry)
        with open(SOURCING_LOG, "w") as f:
            json.dump(logs, f, indent=2)

def update_miner_stats(hotkey, valid_count):
    with threading.Lock():
        if not os.path.exists(MINERS_LOG):
            miners = []
        else:
            with open(MINERS_LOG, "r") as f:
                try:
                    miners = json.load(f)
                except Exception:
                    miners = []
        found = False
        for miner in miners:
            if miner["hotkey"] == hotkey:
                miner["valid_prospects_count"] += valid_count
                miner["last_updated"] = datetime.utcnow().isoformat()
                found = True
                break
        if not found:
            miners.append({
                "hotkey": hotkey,
                "valid_prospects_count": valid_count,
                "last_updated": datetime.utcnow().isoformat()
            })
        with open(MINERS_LOG, "w") as f:
            json.dump(miners, f, indent=2)

async def run_miner(miner, miner_hotkey=None, interval=60, queue_maxsize=1000):
    # Start HTTP server
    await miner.start_http_server()
    
    async def sourcing_loop():
        while True:
            try:
                if not miner.sourcing_mode:
                    await asyncio.sleep(1)
                    continue
                    
                # Generate leads
                num_prospects = 10  # Default batch size
                bt.logging.debug(f"Generating {num_prospects} leads, industry=None, region=None")
                prospects = await get_leads(num_prospects)
                bt.logging.debug(f"Generated {len(prospects)} leads")
                
                # Sanitize prospects with miner_hotkey
                sanitized = [sanitize_prospect(p, miner_hotkey) for p in prospects]
                
                # Add to queue for validation
                lead_queue.enqueue_prospects(sanitized, miner_hotkey)
                bt.logging.info(f"Sourced and enqueued {len(sanitized)} prospects at {datetime.utcnow().isoformat()}")
                
                # Log sourcing activity
                log_sourcing(miner_hotkey, len(sanitized))
                
            except Exception as e:
                bt.logging.error(f"Error in sourcing loop: {e}")
            
            await asyncio.sleep(interval)
    
    # Start sourcing loop
    asyncio.create_task(sourcing_loop())
    
    # Keep the process alive
    while True:
        await asyncio.sleep(1)

def main():
    parser = argparse.ArgumentParser(description="LeadPoet Miner")
    BaseMinerNeuron.add_args(parser)
    parser.add_argument("--axon_port", type=int, default=8091, help="Port for axon and HTTP server")
    args = parser.parse_args()

    if args.logging_trace:
        bt.logging.set_trace(True)

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

    ensure_data_files()
    miner = Miner(config=config)
    try:
        config.axon.port = miner.find_available_port(config.axon.port)
        bt.logging.info(f"Using axon port: {config.axon.port}")
        # Ensure axon uses the same port
        miner.config.axon.port = config.axon.port
        miner.axon = bt.axon(
            port=config.axon.port,
            ip='0.0.0.0',
            wallet=miner.wallet,
            external_ip='0.0.0.0'
        )
        miner.axon.attach(forward_fn=miner.forward, blacklist_fn=miner.blacklist, priority_fn=miner.priority)
        bt.logging.info(f"Reattached axon on port {config.axon.port}")
    except RuntimeError as e:
        bt.logging.error(str(e))
        return

    miner_hotkey = None
    interval = 60
    queue_maxsize = 1000
    if config.mock:
        try:
            from Leadpoet.mock import MockWallet
            miner_hotkey = MockWallet().hotkey_ss58_address
        except Exception:
            miner_hotkey = "5MockHotkeyAddress123456789"
        from Leadpoet.base.utils.config import add_validator_args
        parser = argparse.ArgumentParser()
        add_validator_args(None, parser)
        args, _ = parser.parse_known_args()
        interval = getattr(args, "sourcing_interval", 60)
        queue_maxsize = getattr(args, "queue_maxsize", 1000)
    asyncio.run(run_miner(miner, miner_hotkey, interval, queue_maxsize))

if __name__ == "__main__":
    main()