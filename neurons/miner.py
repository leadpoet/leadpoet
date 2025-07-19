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
from datetime import datetime, timezone
from Leadpoet.base.utils import queue as lead_queue
from Leadpoet.base.utils import pool as lead_pool
import json
from Leadpoet.base.utils.pool import get_leads_from_pool
from validator_models.os_validator_model import validate_lead_list
from miner_models.intent_model import (
    rank_leads,
    classify_industry,
    classify_roles,
    _role_match,
)
from Leadpoet.api.leadpoet_api import get_query_api_axons
from Leadpoet.mock import MockWallet
from collections import OrderedDict


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
            print(f"\n🟡 RECEIVED QUERY from validator: {synapse.num_leads} leads, industry={synapse.industry}, region={synapse.region}")
            print("⏸️  Stopping sourcing, switching to curation mode...")
            
            # Temporarily disable sourcing while curating
            async with self.sourcing_lock:
                self.sourcing_mode = False
                try:
                    # ── derive target industry from buyer description ───────────
                    target_ind = classify_industry(synapse.business_desc) or synapse.industry
                    print(f"🔍 Target industry inferred: {target_ind or 'any'}")

                    # 1️⃣ detect role keywords ONCE
                    desired_roles = classify_roles(synapse.business_desc)
                    if desired_roles:
                        print(f"🛈  Role filter active → {desired_roles}")

                    # 2️⃣ pull a LARGE slice of the pool for this industry
                    pool_slice = get_leads_from_pool(
                        1000,                           # big number = “all we have”
                        industry=target_ind,
                        region=synapse.region
                    )

                    # 3️⃣ role-filter first, then random-sample down
                    if desired_roles:
                        pool_slice = [
                            ld for ld in pool_slice
                            if _role_match(ld.get("role", ""), desired_roles)
                        ] or pool_slice  # fall back if nothing matched

                    # finally down-sample to N×3 for ranking
                    import random
                    curated_leads = random.sample(
                        pool_slice,
                        min(len(pool_slice), synapse.num_leads * 3)
                    )
                    
                    if not curated_leads:
                        print("📝 No leads found in pool, generating new leads...")
                        bt.logging.info("No leads found in pool, generating new leads")
                        new_leads = await get_leads(synapse.num_leads * 2, target_ind, synapse.region)
                        sanitized = [sanitize_prospect(p, self.wallet.hotkey.ss58_address) for p in new_leads]
                        curated_leads = sanitized
                    else:
                        print(f" Curated {len(curated_leads)} leads in pool")
                    
                    # Map the fields to match the API format and ensure all required fields are present
                    mapped_leads = []
                    for lead in curated_leads:
                        mapped_lead = {
                            "email": lead.get("owner_email", ""),
                            "Business": lead.get("business", ""),
                            "Owner Full name": lead.get("owner_full_name", ""),
                            "First": lead.get("first", ""),
                            "Last": lead.get("last", ""),
                            "LinkedIn": lead.get("linkedin", ""),
                            "Website": lead.get("website", ""),
                            "Industry": lead.get("industry", ""),
                            "sub_industry": lead.get("sub_industry", ""),
                            "Region": lead.get("region", ""),
                            "role": lead.get("role", ""),
                            "source":       lead.get("source", ""),
                            "curated_by":   self.wallet.hotkey.ss58_address,
                        }
                        # Only include leads that have all required fields
                        if all(mapped_lead.get(field) for field in ["email", "Business"]):
                            mapped_leads.append(mapped_lead)
                    
                 
                    # run the open-source validator model to attach conversion_score
                    print(" Scoring leads with conversion model...")
                    val          = await validate_lead_list(mapped_leads, target_ind)
                    scored_copy  = val.get("scored_leads", [])
                    # bring scores back onto our rich objects (keeps Business/source/curated_by)
                    for orig, sc in zip(mapped_leads, scored_copy):
                        orig["conversion_score"] = sc.get("conversion_score", 0.0)

                    # ── NEW: apply business-intent ranking ──────────────────────────
                    ranked = await rank_leads(mapped_leads, description=synapse.business_desc)
                    # ────────────────────────────────────────────────────────────────
                    top_leads = ranked[: synapse.num_leads]
                    
                    if not top_leads:
                        print("❌ No valid leads found in pool after mapping")
                        bt.logging.warning("No valid leads found in pool after mapping")
                        synapse.leads = []
                        synapse.dendrite.status_code = 404
                        synapse.dendrite.status_message = "No valid leads found matching criteria"
                        synapse.dendrite.process_time = str(time.time() - start_time)
                        return synapse
                    
                    print(f"📤 SENDING {len(top_leads)} curated leads to validator:")
                    for i, lead in enumerate(top_leads, 1):
                        business = lead.get('Business', 'Unknown')
                        score = lead.get('miner_intent_score', 0)
                        print(f"  {i}. {business} (intent={score:.3f})")
                    
                    bt.logging.info(f"Returning {len(top_leads)} scored leads")
                    synapse.leads = top_leads
                    synapse.dendrite.status_code = 200
                    synapse.dendrite.status_message = "OK"
                    synapse.dendrite.process_time = str(time.time() - start_time)
                            
                finally:
                    # Re-enable sourcing after curation
                    print("▶️  Resuming sourcing mode...")
                    self.sourcing_mode = True
            
        except Exception as e:
            print(f"❌ Error curating leads: {e}")
            bt.logging.error(f"Error curating leads: {e}")
            synapse.leads = []
            synapse.dendrite.status_code = 500
            synapse.dendrite.status_message = f"Error: {str(e)}"
            synapse.dendrite.process_time = str(time.time() - start_time)
            # Ensure sourcing is re-enabled even on error
            self.sourcing_mode = True

        return synapse

    async def handle_lead_request(self, request):
        print(f"\n🟡 RECEIVED QUERY from validator: {await request.text()}")
        bt.logging.info(f"Received HTTP lead request: {await request.text()}")
        try:
            data = await request.json()
            num_leads = data.get("num_leads", 1)
            industry = data.get("industry")      # legacy field – may be empty
            region = data.get("region")
            business_desc = data.get("business_desc", "")
            
            print(f"⏸️  Stopping sourcing, switching to curation mode...")
            
            # Get leads from pool first
            target_ind = classify_industry(business_desc) or industry
            print(f"🔍 Target industry inferred: {target_ind or 'any'}")

            # 1️⃣ detect role keywords ONCE
            desired_roles = classify_roles(business_desc)
            if desired_roles:
                print(f"🛈  Role filter active → {desired_roles}")

            # 2️⃣ pull a LARGE slice of the pool for this industry
            pool_slice = get_leads_from_pool(
                1000,                           # big number = “all we have”
                industry=target_ind,
                region=region
            )

            # 3️⃣ role-filter first, then random-sample down
            if desired_roles:
                pool_slice = [
                    ld for ld in pool_slice
                    if _role_match(ld.get("role", ""), desired_roles)
                ] or pool_slice  # fall back if nothing matched

            # finally down-sample to N×3 for ranking
            import random
            curated_leads = random.sample(
                pool_slice,
                min(len(pool_slice), num_leads * 3)
            )
            
            if not curated_leads:
                print("📝 No leads found in pool, generating new leads...")
                bt.logging.info("No leads found in pool, generating new leads")
                new_leads = await get_leads(num_leads * 2, target_ind, region)
                sanitized = [sanitize_prospect(p, self.wallet.hotkey.ss58_address) for p in new_leads]
                curated_leads = sanitized
            else:
                print(f" Found {len(curated_leads)} leads in pool")
            
            # Map the fields - FIXED VERSION
            mapped_leads = []
            for lead in curated_leads:
                # Map the fields correctly using the same keys as stored in pool
                mapped_lead = {
                    "email":       lead.get("owner_email", ""),
                    "owner_email": lead.get("owner_email", ""),
                    "Business": lead.get("business", ""),
                    "Owner Full name": lead.get("owner_full_name", ""),
                    "First": lead.get("first", ""),
                    "Last": lead.get("last", ""),
                    "LinkedIn": lead.get("linkedin", ""),
                    "Website": lead.get("website", ""),
                    "Industry": lead.get("industry", ""),
                    "sub_industry": lead.get("sub_industry", ""),
                    "role": lead.get("role", ""),
                    "Region": lead.get("region", ""),
                    "source":       lead.get("source", ""),
                    "curated_by":   self.wallet.hotkey.ss58_address,
                }
                
                # Debug log to see what's happening
                bt.logging.debug(f"Original lead: {lead}")
                bt.logging.debug(f"Mapped lead: {mapped_lead}")
                
                # Only include leads that have all required fields
                if all(mapped_lead.get(field) for field in ["email", "Business"]):
                    mapped_leads.append(mapped_lead)
                else:
                    bt.logging.warning(f"Lead missing required fields: {mapped_lead}")
            
            if not mapped_leads:
                print("❌ No valid leads found in pool after mapping")
                bt.logging.warning("No valid leads found in pool after mapping")
                return web.json_response({
                    "leads": [],
                    "status_code": 404,
                    "status_message": "No valid leads found matching criteria",
                    "process_time": "0"
                }, status=404)
            
            # ── score + intent-rank ────────────────────────────────────────────
            print(" Scoring leads with conversion model and intent ranking...")
            val         = await validate_lead_list(mapped_leads, target_ind)
            scored_copy = val.get("scored_leads", [])
            for orig, sc in zip(mapped_leads, scored_copy):
                orig["conversion_score"]     = sc.get("conversion_score", 0.0)

            ranked    = await rank_leads(mapped_leads, description=business_desc)
            top_leads = ranked[:num_leads]

            print(f"📤 SENDING {len(top_leads)} curated leads to validator:")
            for i, lead in enumerate(top_leads, 1):
                business = lead.get('Business', 'Unknown')
                score    = lead.get('miner_intent_score', 0)
                print(f"  {i}. {business}  (intent={score:.3f})")

            print("▶️  Resuming sourcing mode...")

            bt.logging.info(f"Returning {len(top_leads)} leads to HTTP request")
            lead_queue.enqueue_prospects(
                top_leads,
                self.wallet.hotkey.ss58_address,
                request_type="curated",
                requested=num_leads
            )
            return web.json_response({
                "leads": top_leads,
                "status_code": 200,
                "status_message": "OK",
                "process_time": "0"
            })
        except Exception as e:
            print(f"❌ Error curating leads: {e}")
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
        # axon already owns self.config.axon.port – pick the next free one
        http_port = self.find_available_port(self.config.axon.port + 100)
        site = web.TCPSite(runner, '0.0.0.0', http_port)
        await site.start()
        bt.logging.info(f"HTTP server started on port {http_port}")

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
        "role": strip_html(prospect.get("role", "") or prospect.get("Title","")),
        # accept either spelling, but store lower-case
        "sub_industry": strip_html(
            prospect.get("sub_industry") or prospect.get("Sub Industry", "")
        ),
        "region": strip_html(prospect.get("Region", "")),
        "source": miner_hotkey  # Add source field
    }
    
    if not valid_url(sanitized["linkedin"]):
        sanitized["linkedin"] = ""
    if not valid_url(sanitized["website"]):
        sanitized["website"] = ""
    
    sanitized["id"] = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()
    sanitized["created_at"] = now
    sanitized["updated_at"] = now
    return sanitized

def log_sourcing(hotkey, num_prospects):
    """Log sourcing activity to sourcing_logs.json."""
    entry = {"timestamp": datetime.now(timezone.utc).isoformat(), "hotkey": hotkey, "num_prospects": num_prospects}
    
    with open(SOURCING_LOG, "r+") as f:
        try:
            logs = json.load(f)
        except Exception:
            logs = []
        logs.append(entry)
        f.seek(0)
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
                miner["last_updated"] = datetime.now(timezone.utc).isoformat()
                found = True
                break
        if not found:
            miners.append({
                "hotkey": hotkey,
                "valid_prospects_count": valid_count,
                "last_updated": datetime.now(timezone.utc).isoformat()
            })
        with open(MINERS_LOG, "w") as f:
            json.dump(miners, f, indent=2)

async def run_miner(miner, miner_hotkey=None, interval=60, queue_maxsize=1000):
    # Start HTTP server
    await miner.start_http_server()
    
    # Suppress Bittensor verbose logs but keep block emissions
    import logging
    logging.getLogger('bittensor.subtensor').setLevel(logging.WARNING)
    logging.getLogger('bittensor.axon').setLevel(logging.WARNING)
    # Keep block emission logs by not suppressing them
    
    async def sourcing_loop():
        print(f"🔄 Starting continuous sourcing loop (interval: {interval}s)")
        while True:
            try:
                if miner.sourcing_mode:
                    print(f"\n🔄 Sourcing new leads...")
                    # fetch ONE lead at a time so the validator sees it sooner
                    new_leads = await get_leads(1, industry=None, region=None)
                    sanitized = [sanitize_prospect(p, miner_hotkey) for p in new_leads]
                    
                    # Print sourced leads
                    print(f"🔄 Sourced {len(sanitized)} new leads:")
                    for i, lead in enumerate(sanitized, 1):
                        business = lead.get('business', 'Unknown')
                        owner = lead.get('owner_full_name', 'Unknown')
                        email = lead.get('owner_email', 'No email')
                        print(f"  {i}. {business} - {owner} ({email})")
                    
                    # Add to queue for validation (NOT directly to pool)
                    lead_queue.enqueue_prospects(
                        sanitized,
                        miner_hotkey,
                        request_type="sourced"
                    )
                    
                    # Log sourcing activity
                    log_sourcing(miner_hotkey, len(sanitized))
                    
                    print(f"📤 Queued {len(sanitized)} leads for validation at {datetime.now(timezone.utc).strftime('%H:%M:%S')}")
                    
                else:
                    print("⏸️  Sourcing paused (in curation mode)")
                
                await asyncio.sleep(interval)
            except Exception as e:
                print(f"❌ Error in sourcing loop: {e}")
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
    if config.mock:
        miner_wallet = MockWallet(name=config.wallet.name, hotkey=config.wallet.hotkey)
    else:
        miner_wallet = bt.wallet(name=config.wallet.name, hotkey=config.wallet.hotkey)

    miner = Miner(config=config)
    miner.wallet = miner_wallet
    try:
        config.axon.port = miner.find_available_port(config.axon.port)
        bt.logging.info(f"Using axon port: {config.axon.port}")
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

    # Use the actual miner wallet's hotkey instead of creating a new one
    miner_hotkey = miner_wallet.hotkey_ss58_address
    interval = 60
    queue_maxsize = 1000
    if config.mock:
        from Leadpoet.base.utils.config import add_validator_args
        parser = argparse.ArgumentParser()
        add_validator_args(None, parser)
        args, _ = parser.parse_known_args()
        interval = getattr(args, "sourcing_interval", 60)
        queue_maxsize = getattr(args, "queue_maxsize", 1000)
    asyncio.run(run_miner(miner, miner_hotkey, interval, queue_maxsize))

if __name__ == "__main__":
    main()