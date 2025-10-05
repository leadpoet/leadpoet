import time
import asyncio
import threading
import argparse
import traceback
import bittensor as bt
import socket
from Leadpoet.base.miner import BaseMinerNeuron
from Leadpoet.protocol import LeadRequest
from miner_models.lead_sorcerer_main.main_leads import get_leads
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

from miner_models.intent_model import (
    rank_leads,
    classify_industry,
    classify_roles,
    _role_match,
)
from Leadpoet.api.leadpoet_api import get_query_api_axons
from collections import OrderedDict
from Leadpoet.utils.cloud_db import get_cloud_leads
from Leadpoet.utils.cloud_db import push_prospects_to_cloud         # NEW
from Leadpoet.utils.cloud_db import fetch_prospects_from_cloud     # NEW
from Leadpoet.utils.cloud_db import (
    get_cloud_leads,
    push_prospects_to_cloud,
    fetch_prospects_from_cloud,
    fetch_miner_curation_request,      # NEW
    push_miner_curation_result,        # NEW
    fetch_broadcast_requests,          # NEW
    mark_broadcast_processing,         # NEW
)
import logging
import random
import socket, struct     # already have socket; add struct
import grpc  # add near other imports

# Remove this if you don't want to silence noisy "InvalidRequestNameError … Improperly formatted request" lines ──
class _SilenceInvalidRequest(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        # Drop only those specific ERROR messages – let everything else through.
        if record.levelno >= logging.ERROR and "InvalidRequestNameError" in record.getMessage():
            return False
        return True

root_logger       = logging.getLogger()            # root
bittensor_logger  = logging.getLogger("bittensor") # axon middleware logs here
root_logger.addFilter(_SilenceInvalidRequest())
bittensor_logger.addFilter(_SilenceInvalidRequest())
# ────────────────────────────────────────────────────────────────────────────────

class Miner(BaseMinerNeuron):
    def __init__(self, config=None):
        super().__init__(config=config)
        self.use_open_source_lead_model = config.get("use_open_source_lead_model", True) if config else True
        bt.logging.info(f"Using open-source lead model: {self.use_open_source_lead_model}")
        self.app = web.Application()
        self.app.add_routes([web.post('/lead_request', self.handle_lead_request)])
        self.sourcing_mode = True
        self.sourcing_lock = threading.Lock()   # thread-safe
        # background loop orchestration
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self.sourcing_task: Optional[asyncio.Task] = None
        self.cloud_task: Optional[asyncio.Task] = None
        self._bg_interval: int = 60
        self._miner_hotkey: Optional[str] = None

    def pause_sourcing(self):
        print("⏸️ Pausing sourcing (cancel background task)…")
        self.sourcing_mode = False
        if self._loop and self.sourcing_task and not self.sourcing_task.done():
            try:
                self._loop.call_soon_threadsafe(self.sourcing_task.cancel)
            except Exception as e:
                print(f"⚠️ pause_sourcing error: {e}")

    def resume_sourcing(self):
        if not self._loop or not self._miner_hotkey:
            return
        def _restart():
            if self.sourcing_task and not self.sourcing_task.done():
                return
            print("▶️ Resuming sourcing (restart background task)…")
            self.sourcing_mode = True
            self.sourcing_task = asyncio.create_task(
                self.sourcing_loop(self._bg_interval, self._miner_hotkey),
                name="sourcing_loop"
            )
        try:
            self._loop.call_soon_threadsafe(_restart)
        except Exception as e:
            print(f"⚠️ resume_sourcing error: {e}")

    async def sourcing_loop(self, interval: int, miner_hotkey: str):
        print(f"🔄 Starting continuous sourcing loop (interval: {interval}s)")
        while True:
            try:
                # cooperative pause: don’t hold the lock during network I/O
                if not self.sourcing_mode:
                    await asyncio.sleep(1)
                    continue
                with self.sourcing_lock:
                    if not self.sourcing_mode:
                        continue
                    print(f"\n🔄 Sourcing new leads...")
                # do network I/O OUTSIDE the lock so pause can cancel immediately
                new_leads = await get_leads(1, industry=None, region=None)
                sanitized = [sanitize_prospect(p, miner_hotkey) for p in new_leads]
                print(f"🔄 Sourced {len(sanitized)} new leads:")
                for i, lead in enumerate(sanitized, 1):
                    business = lead.get('business', 'Unknown')
                    owner = lead.get('owner_full_name', 'Unknown')
                    email = lead.get('owner_email', 'No email')
                    print(f"  {i}. {business} - {owner} ({email})")
                try:
                    push_prospects_to_cloud(self.wallet, sanitized)
                    print(f"✅ Pushed {len(sanitized)} prospects to cloud queue "
                          f"at {datetime.now(timezone.utc).strftime('%H:%M:%S')}")
                except Exception as e:
                    print(f"❌ Cloud push failed: {e}")
                await asyncio.sleep(interval)
            except asyncio.CancelledError:
                print("🛑 Sourcing task cancelled")
                break
            except Exception as e:
                print(f"❌ Error in sourcing loop: {e}")
                await asyncio.sleep(interval)

    async def cloud_curation_loop(self, miner_hotkey: str):
        print("🔄 Polling Cloud-Run for curation jobs")
        while True:
            try:
                req = fetch_miner_curation_request(self.wallet)
                if req:
                    # stop sourcing immediately
                    self.pause_sourcing()
                    with self.sourcing_lock:
                        print(f"🟢 Curation request pulled from cloud: "
                              f"{req.get('business_desc','')[:40]}…")
                        n = int(req.get("num_leads", 1))
                        target_ind = classify_industry(req.get("business_desc", ""))
                        print(f"🔍 Target industry inferred: {target_ind or 'any'}")
                    # rest of curation OUTSIDE lock
                    desired_roles = classify_roles(req.get("business_desc", ""))
                    if desired_roles:
                        print(f"🛈  Role filter active → {desired_roles}")
                    pool_slice = get_leads_from_pool(
                        1000, industry=target_ind, region=None, wallet=self.wallet
                    )
                    if desired_roles:
                        pool_slice = [
                            ld for ld in pool_slice
                            if _role_match(ld.get("role", ""), desired_roles)
                        ] or pool_slice
                    curated_leads = random.sample(pool_slice, min(len(pool_slice), n * 3))
                    if not curated_leads:
                        print("📝 No leads found in pool, generating new leads...")
                        new_leads = await get_leads(n * 2, target_ind, None)
                        curated_leads = [sanitize_prospect(p, miner_hotkey) for p in new_leads]
                    else:
                        print(f" Curated {len(curated_leads)} leads in pool")
                    mapped_leads = []
                    for lead in curated_leads:
                        m = {
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
                            "source": lead.get("source", ""),
                            "curated_by": self.wallet.hotkey.ss58_address,
                            "curated_at": datetime.now(timezone.utc).isoformat(),
                        }
                        if all(m.get(f) for f in ["email", "Business"]):
                            mapped_leads.append(m)
                    print(" Ranking leads by intent...")
                    ranked = await rank_leads(mapped_leads, description=req.get("business_desc",""))
                    top_leads = ranked[:n]
                    
                    # Add curated_at timestamp to each lead
                    for lead in top_leads:
                        lead["curated_at"] = datetime.now(timezone.utc).isoformat()
                    
                    print(f"📤 SENDING {len(top_leads)} curated leads to validator:")
                    for i, lead in enumerate(top_leads, 1):
                        print(f"  {i}. {lead.get('Business','?')} (intent={lead.get('miner_intent_score',0):.3f})")
                    push_miner_curation_result(
                        self.wallet,
                        {"miner_request_id": req["miner_request_id"], "leads": top_leads},
                    )
                    print(f"✅ Returned {len(top_leads)} leads to cloud broker")
                    # resume sourcing after job
                    self.resume_sourcing()
                await asyncio.sleep(5)
            except asyncio.CancelledError:
                print("🛑 Cloud-curation task cancelled")
                break
            except Exception as e:
                print(f"❌ Cloud-curation loop error: {e}")
                await asyncio.sleep(10)

    async def broadcast_curation_loop(self, miner_hotkey: str):
        """
        Poll Firestore for broadcast API requests and process them.
        """
        print("🟢 Miner broadcast polling loop initialized!")
        print("📡 Polling for broadcast API requests... (will notify when requests are found)")
        
        # Local tracking to prevent re-processing
        processed_requests = set()
        
        poll_count = 0
        while True:
            try:
                poll_count += 1
                
                # Fetch broadcast API requests from Firestore
                from Leadpoet.utils.cloud_db import fetch_broadcast_requests
                requests = fetch_broadcast_requests(self.wallet, role="miner")
                
                # fetch_broadcast_requests() will print when requests are found
                # No need to log anything here when empty
                
                if requests:
                    print(f"🔔 Miner found {len(requests)} broadcast request(s) to process")
                
                for req in requests:
                    request_id = req.get("request_id")
                    
                    # Skip if already processed locally
                    if request_id in processed_requests:
                        print(f"⏭️  Skipping locally processed request {request_id[:8]}...")
                        continue
                    
                    print(f"🔍 Checking request {request_id[:8]}... (status={req.get('status')})")
                    
                    # Try to mark as processing (atomic operation in Firestore)
                    from Leadpoet.utils.cloud_db import mark_broadcast_processing
                    success = mark_broadcast_processing(self.wallet, request_id)
                    
                    if not success:
                        # Another miner already claimed it - mark as processed locally
                        print(f"⏭️  Request {request_id[:8]}... already claimed by another miner")
                        processed_requests.add(request_id)  # ← ADD THIS LINE
                        continue
                    
                    # Mark as processed locally
                    processed_requests.add(request_id)
                    
                    num_leads = req.get("num_leads", 1)
                    business_desc = req.get("business_desc", "")
                    
                    print(f"\n📨 Broadcast API request received {request_id[:8]}...")
                    print(f"   Requested: {num_leads} leads")
                    print(f"   Description: {business_desc[:50]}...")
                    
                    # Pause sourcing
                    self.pause_sourcing()
                    print("🟢 Processing broadcast request: {}…".format(business_desc[:20]))
                    
                    with self.sourcing_lock:
                        print(f"🟢 Processing broadcast request: {business_desc[:40]}…")
                        target_ind = classify_industry(business_desc)
                        print(f"🔍 Target industry inferred: {target_ind or 'any'}")
                    
                    # Curation logic (same as cloud_curation_loop)
                    desired_roles = classify_roles(business_desc)
                    if desired_roles:
                        print(f"🛈  Role filter active → {desired_roles}")
                    
                    pool_slice = get_leads_from_pool(
                        1000, industry=target_ind, region=None, wallet=self.wallet
                    )
                    
                    if desired_roles:
                        pool_slice = [
                            ld for ld in pool_slice
                            if _role_match(ld.get("role", ""), desired_roles)
                        ] or pool_slice
                    
                    curated_leads = random.sample(pool_slice, min(len(pool_slice), num_leads * 3))
                    
                    if not curated_leads:
                        print("📝 No leads found in pool, generating new leads...")
                        new_leads = await get_leads(num_leads * 2, target_ind, None)
                        curated_leads = [sanitize_prospect(p, miner_hotkey) for p in new_leads]
                    else:
                        print(f"📊 Curated {len(curated_leads)} leads from pool")
                    
                    # Map leads to proper format
                    mapped_leads = []
                    for lead in curated_leads:
                        m = {
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
                            "source": lead.get("source", ""),
                            "curated_by": self.wallet.hotkey.ss58_address,
                            "curated_at": datetime.now(timezone.utc).isoformat(),
                        }
                        if all(m.get(f) for f in ["email", "Business"]):
                            mapped_leads.append(m)
                    
                    print("🔄 Ranking leads by intent...")
                    ranked = await rank_leads(mapped_leads, description=business_desc)
                    top_leads = ranked[:num_leads]
                    
                    # Add request_id to track which broadcast this is for
                    for lead in top_leads:
                        lead["curated_at"] = datetime.now(timezone.utc).isoformat()
                        lead["broadcast_request_id"] = request_id
                    
                    print(f"📤 SENDING {len(top_leads)} curated leads for broadcast:")
                    for i, lead in enumerate(top_leads, 1):
                        print(f"  {i}. {lead.get('Business','?')} (intent={lead.get('miner_intent_score',0):.3f})")
                    
                    # NEW: Send leads to Firestore (not Cloud Run API)
                    from Leadpoet.utils.cloud_db import push_miner_curated_leads
                    success = push_miner_curated_leads(
                        self.wallet,
                        request_id,
                        top_leads
                    )
                    
                    if success:
                        print(f"✅ Sent {len(top_leads)} leads to Firestore for request {request_id[:8]}...")
                    else:
                        print(f"❌ Failed to send leads to Firestore for request {request_id[:8]}...")
                    
                    # Resume sourcing
                    self.resume_sourcing()
                
            except asyncio.CancelledError:
                print("🛑 Broadcast-curation task cancelled")
                break
            except Exception as e:
                print(f"❌ Broadcast-curation loop error: {e}")
                print(f"Broadcast-curation loop error: {e}")
                import traceback
                print(traceback.format_exc())
                await asyncio.sleep(5)  # Wait before retrying on error
            
            # Poll every 1 second for instant response
            await asyncio.sleep(1)  # ← REDUCED from 5 to 1 second

    async def _forward_async(self, synapse: LeadRequest) -> LeadRequest:
        # ─────────────────────────────────────────────────────────────
        import time as _t
        _t0 = _t.time()
        print("\n─────────  AXON ➜ MINER  ─────────")
        print(f"⚡  AXON call received  | leads={synapse.num_leads}"
              f" industry={synapse.industry or '∅'} region={synapse.region or '∅'}")
        print(f"⏱️   at {datetime.utcnow().isoformat()} UTC")
        # ─────────────────────────────────────────────────────────────
        bt.logging.info(f" AXON CALL RECEIVED: {synapse}")

        start_time = time.time()

        try:
            print(f"\n🟡 RECEIVED QUERY from validator: {synapse.num_leads} leads, industry={synapse.industry}, region={synapse.region}")
            print("⏸️  Stopping sourcing, switching to curation mode...")
            
            # Take the global lock so sourcing stays paused
            with self.sourcing_lock:
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
                        region=synapse.region,
                        wallet=self.wallet            # ensures cloud read
                    )

                    # 3️⃣ role-filter first, then random-sample down
                    if desired_roles:
                        pool_slice = [
                            ld for ld in pool_slice
                            if _role_match(ld.get("role", ""), desired_roles)
                        ] or pool_slice  # fall back if nothing matched

                    # finally down-sample to N×3 for ranking
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
                            "curated_at":   datetime.now(timezone.utc).isoformat(),  # NEW: ISO timestamp
                        }
                        # Only include leads that have all required fields
                        if all(mapped_lead.get(field) for field in ["email", "Business"]):
                            mapped_leads.append(mapped_lead)
                    
                    # REMOVED: conversion_score calculation - no longer needed
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
                    
                    print("🚚 Returning leads over AXON")
                    print(f"✅  Prepared {len(top_leads)} leads in"
                          f" {(_t.time()-_t0):.2f}s – sending back to validator")
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
            print(f"❌ AXON FORWARD ERROR: {e}")
            bt.logging.error(f"AXON FORWARD ERROR: {e}")
            # Return empty response so validator gets something
            synapse.leads = []
            synapse.dendrite.status_code = 500
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
                region=region,
                wallet=self.wallet               # <-- passes hotkey for auth
            )

            # 3️⃣ role-filter first, then random-sample down
            if desired_roles:
                pool_slice = [
                    ld for ld in pool_slice
                    if _role_match(ld.get("role", ""), desired_roles)
                ] or pool_slice  # fall back if nothing matched

            # finally down-sample to N×3 for ranking
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
            
            # ── intent-rank ────────────────────────────────────────────
            print(" Ranking leads by intent...")
            ranked = await rank_leads(mapped_leads, description=business_desc)
            top_leads = ranked[:num_leads]

            print(f"📤 SENDING {len(top_leads)} curated leads to validator:")
            for i, lead in enumerate(top_leads, 1):
                business = lead.get('Business', 'Unknown')
                score    = lead.get('miner_intent_score', 0)
                print(f"  {i}. {business}  (intent={score:.3f})")

            print("▶️  Resuming sourcing mode...")

            bt.logging.info(f"Returning {len(top_leads)} leads to HTTP request")
            # 🔄 send prospects to Firestore queue
            push_prospects_to_cloud(self.wallet, top_leads)
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

    # Pause sourcing at the earliest possible moment when any axon call arrives
    def blacklist(self, synapse: LeadRequest) -> Tuple[bool, str]:
        # Ignore random HTTP scanners that trigger InvalidRequestNameError
        if getattr(synapse, "dendrite", None) is None:
            return True, "Malformed request"
        try:
            self.pause_sourcing()
        except Exception as _e:
            print(f"⚠️ pause_sourcing in blacklist failed: {_e}")
        caller_hk = getattr(synapse.dendrite, "hotkey", None)
        caller_uid = None
        if caller_hk in self.metagraph.hotkeys:
            caller_uid = self.metagraph.hotkeys.index(caller_hk)
        if getattr(self.config.blacklist, "force_validator_permit", False):
            is_validator = (caller_uid is not None and bool(self.metagraph.validator_permit[caller_uid]))
            if not is_validator:
                print(f"🛑 Blacklist: rejecting {caller_hk} (not a validator)")
                return True, "Caller is not a validator"
        if not getattr(self.config.blacklist, "allow_non_registered", True):
            if caller_uid is None:
                print(f"🛑 Blacklist: rejecting {caller_hk} (not registered)")
                return True, "Caller not registered"
        print(f"✅ Blacklist: allowing {caller_hk} (uid={caller_uid})")
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

    # -------------------------------------------------------------------
    #  Wrapper the axon actually calls (sync)
    # -------------------------------------------------------------------
    def forward(self, synapse: LeadRequest) -> LeadRequest:
        # 🔔 this fires only when the request arrives via AXON
        print(f"🔔 AXON QUERY from {getattr(synapse.dendrite, 'hotkey', 'unknown')} | "
              f"{synapse.num_leads} leads | desc='{(synapse.business_desc or '')[:40]}…'")
        # stop sourcing immediately
        self.pause_sourcing()
        result_holder = {}
        error_holder = {}
        def _runner():
            try:
                result_holder["res"] = asyncio.run(self._forward_async(synapse))
            except Exception as e:
                error_holder["err"] = e
        t = threading.Thread(target=_runner, daemon=True)
        t.start()
        t.join(timeout=120)                          # <── was 90
        if t.is_alive():
            print("⏳ AXON forward timed out after 95 s")
            synapse.leads = []
            synapse.dendrite.status_code = 504
            synapse.dendrite.status_message = "Miner forward timeout"
            self.resume_sourcing()
            return synapse
        if "err" in error_holder:
            print(f"❌ AXON FORWARD ERROR: {error_holder['err']}")
            synapse.leads = []
            synapse.dendrite.status_code = 500
            synapse.dendrite.status_message = f"Error: {error_holder['err']}"
            self.resume_sourcing()
            return synapse
        res = result_holder["res"]
        self.resume_sourcing()
        return res

    def stop(self):
        try:
            if getattr(self, "axon", None):
                print("🛑 Stopping axon gRPC server…")
                self.axon.stop()
                print("✅ Axon stopped")
        except Exception as e:
            print(f"⚠️ Error stopping axon: {e}")
        try:
            self.resume_sourcing()  # ensure background is not left paused
        except Exception:
            pass

    def run(self):
        # Delegate to the base class run loop; avoids calling non-callable step() and missing stop().
        bt.logging.info(f"Miner starting at block: {self.block}")
        try:
            return super().run()
        except KeyboardInterrupt:
            self.stop()
            bt.logging.success("Miner killed by keyboard interrupt.")
        except Exception as e:
            print(f"❌ Error in miner.run(): {e}")
            bt.logging.error(traceback.format_exc())
            self.stop()

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
    
    # REMOVED: id, created_at, updated_at - these should only be added during curation
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
    logging.getLogger('bittensor.subtensor').setLevel(logging.WARNING)
    logging.getLogger('bittensor.axon').setLevel(logging.WARNING)
    miner._loop = asyncio.get_running_loop()
    miner._bg_interval = interval
    miner._miner_hotkey = miner_hotkey
    
    # Start all background tasks
    miner.sourcing_task = asyncio.create_task(
        miner.sourcing_loop(interval, miner_hotkey), name="sourcing_loop"
    )
    miner.cloud_task = asyncio.create_task(
        miner.cloud_curation_loop(miner_hotkey), name="cloud_curation_loop"
    )
    # NEW: Start broadcast curation task
    miner.broadcast_task = asyncio.create_task(
        miner.broadcast_curation_loop(miner_hotkey), name="broadcast_curation_loop"
    )
    
    print("✅ Started 3 background tasks:")
    print("   1. sourcing_loop - Continuous lead sourcing")
    print("   2. cloud_curation_loop - Cloud-Run curation requests")
    print("   3. broadcast_curation_loop - Broadcast API requests")
    
    # Keep alive
    while True:
        await asyncio.sleep(1)

async def _grpc_ready_check(addr: str, timeout: float = 5.0) -> bool:
    try:
        ch = grpc.aio.insecure_channel(addr)
        await asyncio.wait_for(ch.channel_ready(), timeout=timeout)
        await ch.close()
        print(f"✅ gRPC preflight OK → {addr}")
        return True
    except Exception as e:
        print(f"⚠️ aio preflight failed for {addr}: {e}")
    # Fallback to sync probe, run in a thread so it doesn't require a Task
    def _sync_probe() -> bool:
        ch = grpc.insecure_channel(addr)
        grpc.channel_ready_future(ch).result(timeout=timeout)
        ch.close()
        return True
    try:
        ok = await asyncio.get_running_loop().run_in_executor(None, _sync_probe)
        if ok:
            print(f"✅ gRPC preflight OK (sync) → {addr}")
            return True
    except Exception as e:
        print(f"❌ gRPC preflight FAIL → {addr} | {e}")
    return False


def main():
    parser = argparse.ArgumentParser(description="LeadPoet Miner")
    BaseMinerNeuron.add_args(parser)
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
    config.blacklist = bt.Config()
    config.blacklist.force_validator_permit = args.blacklist_force_validator_permit
    config.blacklist.allow_non_registered = args.blacklist_allow_non_registered
    config.neuron = bt.Config()
    config.neuron.epoch_length = args.neuron_epoch_length or 1000
    config.use_open_source_lead_model = args.use_open_source_lead_model

    # ──────────────────────── AXON NETWORKING ──────────────────────────
    # Bind locally on 0.0.0.0 but advertise the user-supplied external
    # IP/port on-chain so validators can connect over the Internet.
    config.axon = bt.Config()
    config.axon.ip   = "0.0.0.0"                 # listen on all interfaces
    config.axon.port = args.axon_port or 8091    # internal bind port
    if args.axon_ip:
        config.axon.external_ip = args.axon_ip   # public address
    if args.axon_port:
        config.axon.external_port = args.axon_port
        config.axon.port = args.axon_port

    ensure_data_files()
    
    # Create miner and run it properly on the Bittensor network
    miner = Miner(config=config)
    
    # Check if miner is properly registered
    print(f"🔍 Checking miner registration...")
    print(f"   Wallet: {miner.wallet.hotkey.ss58_address}")
    print(f"   NetUID: {config.netuid}")
    print(f"   UID: {miner.uid}")
    
    if miner.uid is None:
        print("❌ Miner is not registered on the network!")
        print("   Please register your wallet on subnet 401 first.")
        return

    print(f"✅ Miner registered with UID: {miner.uid}")
    
    # Start the Bittensor miner in background thread (this will start the axon and connect to testnet)
    import threading
    def run_miner_safe():
        try:
            print(" Starting Bittensor miner axon...")
            print("   Syncing metagraph...")
            miner.sync()
            print(f"   Current block: {miner.block}")
            print(f"   Metagraph has {len(miner.metagraph.axons)} axons")
            print(f"   My axon should be at index {miner.uid}")
            
            miner.run()
        except Exception as e:
            print(f"❌ Error in miner.run(): {e}")
            import traceback
            traceback.print_exc()
    
    miner_thread = threading.Thread(target=run_miner_safe, daemon=True)
    miner_thread.start()
    
    # Give the miner a moment to start up
    import time
    time.sleep(3)
    
    # Run the sourcing loop in the main thread
    async def run_sourcing():
        miner_hotkey = miner.wallet.hotkey.ss58_address
        interval = 60
        queue_maxsize = 1000
        await run_miner(miner, miner_hotkey, interval, queue_maxsize)
    
    # Run the sourcing loop
    asyncio.run(run_sourcing())

if __name__ == "__main__":
    main()
