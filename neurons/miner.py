import time
import asyncio
import threading
import argparse
import traceback
import getpass
import bittensor as bt
import socket
from Leadpoet.base.miner import BaseMinerNeuron
from Leadpoet.protocol import LeadRequest
from miner_models.lead_sorcerer_main.main_leads import get_leads
from typing import Tuple, List, Dict, Optional
from aiohttp import web
import os
import re
import html
from datetime import datetime, timezone
import json
from Leadpoet.base.utils.pool import get_leads_from_pool

from miner_models.intent_model import (
    rank_leads,
    classify_industry,
    classify_roles,
    _role_match,
)

from Leadpoet.utils.cloud_db import (
    push_prospects_to_cloud,
    fetch_miner_curation_request,
    push_miner_curation_result,
    check_linkedin_combo_duplicate,
    gateway_poll_fulfillment_requests,
    gateway_submit_fulfillment_commit,
    gateway_reveal_fulfillment,
)
from Leadpoet.utils.hashing import hash_lead, HASH_SCHEMA_VERSION
import logging
import httpx
import requests
import random
import grpc
from pathlib import Path
import hashlib

from gateway.research_lab.config import DEFAULT_LOOP_START_FEE_USD as RESEARCH_LAB_DEFAULT_LOOP_START_FEE_USD


class _SilenceInvalidRequest(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if record.levelno >= logging.ERROR and "InvalidRequestNameError" in record.getMessage():
            return False
        return True


root_logger = logging.getLogger()
bittensor_logger = logging.getLogger("bittensor")
root_logger.addFilter(_SilenceInvalidRequest())
bittensor_logger.addFilter(_SilenceInvalidRequest())

for logger_name in ['orchestrator', 'domain', 'crawl', 'enrich']:
    logging.getLogger(logger_name).setLevel(logging.WARNING)


class Miner(BaseMinerNeuron):

    def __init__(self, config=None):
        super().__init__(config=config)
        self.use_open_source_lead_model = config.get(
            "use_open_source_lead_model", True) if config else True
        bt.logging.info(
            f"Using open-source lead model: {self.use_open_source_lead_model}")
        self.app = web.Application()
        self.app.add_routes(
            [web.post('/lead_request', self.handle_lead_request)])
        self.sourcing_mode = True
        self.sourcing_lock = threading.Lock()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self.sourcing_task: Optional[asyncio.Task] = None
        self.cloud_task: Optional[asyncio.Task] = None
        self.fulfillment_task: Optional[asyncio.Task] = None
        self._bg_interval: int = 60
        self._miner_hotkey: Optional[str] = None
        self._pending_fulfillment: Dict[str, dict] = {}  # request_id -> state
        self._sourcing_active: bool = False
        self._fulfillment_semaphore = asyncio.Semaphore(
            int(os.environ.get("FULFILLMENT_MAX_CONCURRENT_SOURCES", "2")))
        
        bt.logging.info(f"✅ Miner initialized (using trustless gateway - no JWT tokens)")

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
            self.sourcing_task = asyncio.create_task(self.sourcing_loop(
                self._bg_interval, self._miner_hotkey),
                                                     name="sourcing_loop")

        try:
            self._loop.call_soon_threadsafe(_restart)
        except Exception as e:
            print(f"⚠️ resume_sourcing error: {e}")

    async def process_generated_leads(self, leads: list) -> list:
        """
        Process and enrich leads with source provenance BEFORE sanitization.
        
        This function validates and enriches leads at the protocol level to ensure
        compliance with regulatory requirements. It cannot be bypassed by miners.
        
        Steps:
        1. Extract Website field from each lead
        2. Validate source URL against regulatory requirements
        3. Filter out invalid leads
        4. Determine source type (public_registry, company_site, etc.)
        5. Enrich lead with source_url and source_type
        
        Args:
            leads: Raw leads from lead generation model
            
        Returns:
            List of validated and enriched leads
        """
        from Leadpoet.utils.source_provenance import (
            validate_source_url,
            determine_source_type
        )
        
        validated_leads = []
        
        for lead in leads:
            # Extract website field (try multiple common field names)
            source_url = (
                lead.get("Website") or 
                lead.get("website") or 
                lead.get("Website URL") or
                lead.get("Company Website") or
                ""
            )
            
            if not source_url:
                bt.logging.warning(
                    f"Lead missing source URL, skipping: "
                    f"{lead.get('Business', lead.get('business', 'Unknown'))}"
                )
                continue
            
            # Determine source type FIRST (needed for validation)
            source_type = determine_source_type(source_url, lead)
            
            # Validate source URL against regulatory requirements
            try:
                is_valid, reason = await validate_source_url(source_url, source_type)
                if not is_valid:
                    bt.logging.warning(f"Invalid source URL: {source_url} - {reason}")
                    continue
            except Exception as e:
                bt.logging.error(f"Error validating source URL {source_url}: {e}")
                continue
            
            # Enrich lead with provenance metadata
            lead["source_url"] = source_url
            lead["source_type"] = source_type
            
            validated_leads.append(lead)
        
        if validated_leads:
            bt.logging.info(
                f"✅ Source provenance: {len(validated_leads)}/{len(leads)} leads validated"
            )
        else:
            bt.logging.warning("⚠️ No leads passed source provenance validation")
        
        return validated_leads

    async def sourcing_loop(self, interval: int, miner_hotkey: str):
        print(f"🔄 Starting continuous sourcing loop (interval: {interval}s)")
        while True:
            try:
                if not self.sourcing_mode:
                    await asyncio.sleep(1)
                    continue
                with self.sourcing_lock:
                    if not self.sourcing_mode:
                        continue
                    print("\n🔄 Sourcing new leads...")
                self._sourcing_active = True
                new_leads = await get_leads(1, industry=None, region=None)
                
                # Process leads through source provenance validation (protocol level)
                validated_leads = await self.process_generated_leads(new_leads)
                
                # Sanitize validated leads
                sanitized = [
                    sanitize_prospect(p, miner_hotkey) for p in validated_leads
                ]
                print(f"🔄 Sourced {len(sanitized)} new leads:")
                for i, lead in enumerate(sanitized, 1):
                    business = lead.get('business', 'Unknown')
                    owner = lead.get('full_name', 'Unknown')
                    email = lead.get('email', 'No email')
                    print(f"  {i}. {business} - {owner} ({email})")
                
                # Submit leads via gateway (Passage 1 workflow)
                try:
                    from Leadpoet.utils.cloud_db import (
                        check_email_duplicate,
                        gateway_get_presigned_url,
                        gateway_upload_lead,
                        gateway_verify_submission
                    )
                    
                    submitted_count = 0
                    verified_count = 0
                    duplicate_count = 0
                    
                    for lead in sanitized:
                        business_name = lead.get('business', 'Unknown')
                        email = lead.get('email', '')
                        linkedin_url = lead.get('linkedin', '')
                        company_linkedin_url = lead.get('company_linkedin', '')
                        
                        # Step 0: Check for duplicates BEFORE calling presign (saves time & rate limit)
                        # Check both email AND linkedin combo (person+company)
                        
                        # Check email duplicate (approved or processing = skip, rejected = allow)
                        if check_email_duplicate(email):
                            print(f"⏭️  Skipping duplicate email: {business_name} ({email})")
                            duplicate_count += 1
                            continue
                        
                        # Check linkedin combo duplicate (same logic: approved/processing = skip, rejected = allow)
                        if linkedin_url and company_linkedin_url:
                            if check_linkedin_combo_duplicate(linkedin_url, company_linkedin_url):
                                print(f"⏭️  Skipping duplicate person+company: {business_name}")
                                print(f"      LinkedIn: {linkedin_url[:50]}...")
                                print(f"      Company: {company_linkedin_url[:50]}...")
                            duplicate_count += 1
                            continue
                        
                        # Step 1: Get presigned URLs (gateway logs SUBMISSION_REQUEST with committed hash)
                        presign_result = gateway_get_presigned_url(self.wallet, lead)
                        if not presign_result:
                            print(f"⚠️  Failed to get presigned URL for {business_name}")
                            continue
                        
                        # Step 2: Upload to S3 (gateway will mirror to MinIO automatically)
                        s3_uploaded = gateway_upload_lead(presign_result['s3_url'], lead)
                        if not s3_uploaded:
                            print(f"⚠️  Failed to upload to S3: {business_name}")
                            continue
                        
                        print(f"✅ Lead uploaded to S3 (gateway will mirror to MinIO)")
                        submitted_count += 1
                        
                        # Step 4: Trigger gateway verification (BRD Section 4.1, Steps 5-6)
                        # Gateway will:
                        # - Fetch uploaded blobs from S3/MinIO
                        # - Verify hashes match committed lead_blob_hash
                        # - Log STORAGE_PROOF events (one per mirror)
                        # - Store lead in leads_private table
                        # - Log SUBMISSION event
                        verification_result = gateway_verify_submission(
                            self.wallet,
                            presign_result['lead_id']
                        )
                        
                        if verification_result:
                            verified_count += 1
                            print(f"✅ Verified: {business_name} (backends: {verification_result['storage_backends']})")
                        else:
                            print(f"⚠️  Verification failed: {business_name}")
                    
                    if verified_count > 0:
                        print(
                            f"✅ Successfully submitted and verified {verified_count}/{len(sanitized)} leads "
                            f"at {datetime.now(timezone.utc).strftime('%H:%M:%S')}"
                        )
                        if duplicate_count > 0:
                            print(f"   ⏭️  Skipped {duplicate_count} duplicate(s)")
                    elif submitted_count > 0:
                        print(f"⚠️  {submitted_count} lead(s) rejected by gateway (see error details above)")
                    elif duplicate_count > 0:
                        print(f"⏭️  All {duplicate_count} lead(s) were duplicates (already submitted)")
                    else:
                        print("⚠️  Failed to submit any leads via gateway")
                except Exception as e:
                    print(f"❌ Gateway submission exception: {e}")
                self._sourcing_active = False
                await asyncio.sleep(interval)
            except asyncio.CancelledError:
                self._sourcing_active = False
                print("🛑 Sourcing task cancelled")
                break
            except Exception as e:
                self._sourcing_active = False
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
                        target_ind = classify_industry(
                            req.get("business_desc", ""))
                        print(
                            f"🔍 Target industry inferred: {target_ind or 'any'}"
                        )
                    desired_roles = classify_roles(req.get(
                        "business_desc", ""))
                    if desired_roles:
                        print(f"🛈  Role filter active → {desired_roles}")
                    pool_slice = get_leads_from_pool(1000,
                                                     industry=target_ind,
                                                     region=None,
                                                     wallet=self.wallet)
                    if desired_roles:
                        pool_slice = [
                            ld for ld in pool_slice
                            if _role_match(ld.get("role", ""), desired_roles)
                        ] or pool_slice
                    curated_leads = random.sample(pool_slice,
                                                  min(len(pool_slice), n * 3))
                    if not curated_leads:
                        print(
                            "📝 No leads found in pool, generating new leads..."
                        )
                        new_leads = await get_leads(n * 2, target_ind, None)
                        
                        # Process leads through source provenance validation (protocol level)
                        validated_leads = await self.process_generated_leads(new_leads)
                        
                        # Sanitize validated leads
                        curated_leads = [
                            sanitize_prospect(p, miner_hotkey)
                            for p in validated_leads
                        ]
                    else:
                        print(f" Curated {len(curated_leads)} leads in pool")
                    mapped_leads = []
                    for lead in curated_leads:
                        m = {
                            "email": lead.get("email", ""),
                            "business": lead.get("business", ""),
                            "full_name": lead.get("full_name", ""),
                            "first": lead.get("first", ""),
                            "last": lead.get("last", ""),
                            "linkedin": lead.get("linkedin", ""),
                            "website": lead.get("website", ""),
                            "industry": lead.get("industry", ""),
                            "sub_industry": lead.get("sub_industry", ""),
                            "country": lead.get("country", ""),
                            "state": lead.get("state", ""),
                            "city": lead.get("city", ""),
                            "region": lead.get("region", ""),
                            "role": lead.get("role", ""),
                            "description": lead.get("description", ""),
                            "company_linkedin": lead.get("company_linkedin", ""),
                            "employee_count": lead.get("employee_count", ""),
                            "source": lead.get("source", ""),
                            "curated_by": self.wallet.hotkey.ss58_address,
                            "curated_at":
                            datetime.now(timezone.utc).isoformat(),
                        }
                        if all(m.get(f) for f in ["email", "business"]):
                            mapped_leads.append(m)
                    print(" Ranking leads by intent...")
                    ranked = await rank_leads(mapped_leads,
                                              description=req.get(
                                                  "business_desc", ""))
                    top_leads = ranked[:n]

                    # Add curated_at timestamp to each lead
                    for lead in top_leads:
                        lead["curated_at"] = datetime.now(
                            timezone.utc).isoformat()

                    print(
                        f"📤 SENDING {len(top_leads)} curated leads to validator:"
                    )
                    for i, lead in enumerate(top_leads, 1):
                        print(
                            f"  {i}. {lead.get('business','?')} (intent={lead.get('miner_intent_score',0):.3f})"
                        )
                    push_miner_curation_result(
                        self.wallet,
                        {
                            "miner_request_id": req["miner_request_id"],
                            "leads": top_leads
                        },
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
        print(
            "📡 Polling for broadcast API requests... (will notify when requests are found)"
        )

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
                    print(
                        f"🔔 Miner found {len(requests)} broadcast request(s) to process"
                    )

                for req in requests:
                    request_id = req.get("request_id")

                    # Skip if already processed locally
                    if request_id in processed_requests:
                        print(
                            f"⏭️  Skipping locally processed request {request_id[:8]}..."
                        )
                        continue

                    print(
                        f"🔍 Checking request {request_id[:8]}... (status={req.get('status')})"
                    )

                    # Try to mark as processing (atomic operation in Firestore)
                    from Leadpoet.utils.cloud_db import mark_broadcast_processing
                    success = mark_broadcast_processing(
                        self.wallet, request_id)

                    if not success:
                        # Another miner already claimed it - mark as processed locally
                        print(
                            f"⏭️  Request {request_id[:8]}... already claimed by another miner"
                        )
                        processed_requests.add(request_id)
                        continue

                    # Mark as processed locally
                    processed_requests.add(request_id)

                    num_leads = req.get("num_leads", 1)
                    business_desc = req.get("business_desc", "")

                    print(
                        f"\n📨 Broadcast API request received {request_id[:8]}..."
                    )
                    print(f"   Requested: {num_leads} leads")
                    print(f"   Description: {business_desc[:50]}...")

                    # Pause sourcing
                    self.pause_sourcing()
                    print("🟢 Processing broadcast request: {}…".format(
                        business_desc[:20]))

                    with self.sourcing_lock:
                        print(
                            f"🟢 Processing broadcast request: {business_desc[:40]}…"
                        )
                        target_ind = classify_industry(business_desc)
                        print(
                            f"🔍 Target industry inferred: {target_ind or 'any'}"
                        )

                    # Curation logic (same as cloud_curation_loop)
                    desired_roles = classify_roles(business_desc)
                    if desired_roles:
                        print(f"🛈  Role filter active → {desired_roles}")

                    pool_slice = get_leads_from_pool(1000,
                                                     industry=target_ind,
                                                     region=None,
                                                     wallet=self.wallet)

                    if desired_roles:
                        pool_slice = [
                            ld for ld in pool_slice
                            if _role_match(ld.get("role", ""), desired_roles)
                        ] or pool_slice

                    curated_leads = random.sample(
                        pool_slice, min(len(pool_slice), num_leads * 3))

                    if not curated_leads:
                        print(
                            "📝 No leads found in pool, generating new leads..."
                        )
                        new_leads = await get_leads(num_leads * 2, target_ind,
                                                    None)
                        
                        # Process leads through source provenance validation (protocol level)
                        validated_leads = await self.process_generated_leads(new_leads)
                        
                        # Sanitize validated leads
                        curated_leads = [
                            sanitize_prospect(p, miner_hotkey)
                            for p in validated_leads
                        ]
                    else:
                        print(
                            f"📊 Curated {len(curated_leads)} leads from pool")

                    # Map leads to proper format
                    mapped_leads = []
                    for lead in curated_leads:
                        m = {
                            "email": lead.get("email", ""),
                            "business": lead.get("business", ""),
                            "full_name": lead.get("full_name", ""),
                            "first": lead.get("first", ""),
                            "last": lead.get("last", ""),
                            "linkedin": lead.get("linkedin", ""),
                            "website": lead.get("website", ""),
                            "industry": lead.get("industry", ""),
                            "sub_industry": lead.get("sub_industry", ""),
                            "country": lead.get("country", ""),
                            "state": lead.get("state", ""),
                            "city": lead.get("city", ""),
                            "region": lead.get("region", ""),
                            "role": lead.get("role", ""),
                            "description": lead.get("description", ""),
                            "company_linkedin": lead.get("company_linkedin", ""),
                            "employee_count": lead.get("employee_count", ""),
                            "source": lead.get("source", ""),
                            "curated_by": self.wallet.hotkey.ss58_address,
                            "curated_at":
                            datetime.now(timezone.utc).isoformat(),
                        }
                        if all(m.get(f) for f in ["email", "business"]):
                            mapped_leads.append(m)

                    print("🔄 Ranking leads by intent...")
                    ranked = await rank_leads(mapped_leads,
                                              description=business_desc)
                    top_leads = ranked[:num_leads]

                    # Add request_id to track which broadcast this is for
                    for lead in top_leads:
                        lead["curated_at"] = datetime.now(
                            timezone.utc).isoformat()
                        lead["broadcast_request_id"] = request_id

                    print(
                        f"📤 SENDING {len(top_leads)} curated leads for broadcast:"
                    )
                    for i, lead in enumerate(top_leads, 1):
                        print(
                            f"  {i}. {lead.get('business','?')} (intent={lead.get('miner_intent_score',0):.3f})"
                        )

                    from Leadpoet.utils.cloud_db import push_miner_curated_leads
                    success = push_miner_curated_leads(self.wallet, request_id,
                                                       top_leads)

                    if success:
                        print(
                            f"✅ Sent {len(top_leads)} leads to Firestore for request {request_id[:8]}..."
                        )
                    else:
                        print(
                            f"❌ Failed to send leads to Firestore for request {request_id[:8]}..."
                        )

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
            await asyncio.sleep(1)

    # ---------------------------------------------------------------
    #  Fulfillment persistence helpers (crash recovery)
    # ---------------------------------------------------------------

    _FULFILLMENT_DIR = "fulfillment_pending"

    def _save_pending_fulfillment(self, request_id: str, state: dict) -> None:
        """Persist a pending commit to disk so reveals survive crashes.
        Uses write-to-tmp + rename for atomicity."""
        try:
            os.makedirs(self._FULFILLMENT_DIR, exist_ok=True)
            path = os.path.join(self._FULFILLMENT_DIR, f"{request_id}.json")
            tmp_path = path + ".tmp"
            with open(tmp_path, "w") as f:
                json.dump(state, f)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, path)
        except Exception as e:
            bt.logging.warning(f"Failed to persist fulfillment state for {request_id[:8]}: {e}")

    def _remove_pending_fulfillment(self, request_id: str) -> None:
        """Remove on-disk state after a successful reveal (or expiry)."""
        self._pending_fulfillment.pop(request_id, None)
        try:
            path = os.path.join(self._FULFILLMENT_DIR, f"{request_id}.json")
            if os.path.exists(path):
                os.remove(path)
        except Exception as e:
            bt.logging.warning(f"Failed to remove fulfillment state file for {request_id[:8]}: {e}")

    def _load_pending_fulfillment(self) -> None:
        """Recover pending commits from disk on startup."""
        if not os.path.isdir(self._FULFILLMENT_DIR):
            return
        count = 0
        for fname in os.listdir(self._FULFILLMENT_DIR):
            if not fname.endswith(".json"):
                continue
            request_id = fname[:-5]
            if request_id in self._pending_fulfillment:
                continue
            try:
                path = os.path.join(self._FULFILLMENT_DIR, fname)
                with open(path, "r") as f:
                    state = json.load(f)
                self._pending_fulfillment[request_id] = state
                count += 1
            except Exception as e:
                bt.logging.warning(f"Failed to load fulfillment state {fname}: {e}")
        if count:
            bt.logging.info(f"Recovered {count} pending fulfillment commit(s) from disk")

    # ---------------------------------------------------------------
    #  Fulfillment loop — poll gateway, source leads, commit/reveal
    # ---------------------------------------------------------------

    async def fulfillment_loop(self, miner_hotkey: str):
        """Background loop: poll for ICP fulfillment requests, source leads,
        commit hashes, then reveal after the commit window closes."""
        print("🔄 Fulfillment loop started (polling every 30s)")
        poll_interval = int(os.environ.get("FULFILLMENT_POLL_INTERVAL", "30"))

        self._load_pending_fulfillment()

        while True:
            try:
                # Phase 1: poll for active requests (skip sourcing when
                # AXON is not yielding or the main sourcing_loop is active)
                skip_sourcing = False
                if not self.sourcing_mode:
                    bt.logging.debug("Fulfillment: AXON not yielding, skipping new sourcing")
                    skip_sourcing = True
                if self._sourcing_active:
                    bt.logging.debug("Fulfillment: sourcing_loop active, deferring new sourcing")
                    skip_sourcing = True

                if not skip_sourcing:
                    data = gateway_poll_fulfillment_requests(self.wallet)
                    active_requests = data.get("requests", []) if isinstance(data, dict) else []
                else:
                    active_requests = []

                for req in active_requests:
                    request_id = req.get("request_id", "")
                    if not request_id or request_id in self._pending_fulfillment:
                        continue

                    icp = req.get("icp", {})
                    num_leads = req.get("num_leads", 10)
                    industry = icp.get("industry", None)

                    print(f"\n🎯 Fulfillment: sourcing {num_leads} leads for {request_id[:8]}... (industry={industry})")

                    try:
                        async with self._fulfillment_semaphore:
                            leads = await self._source_fulfillment_leads(
                                icp, num_leads, miner_hotkey,
                            )
                    except Exception as e:
                        print(f"   ❌ Fulfillment sourcing failed for {request_id[:8]}: {e}")
                        continue

                    if not leads:
                        print(f"   ⚠️ No leads sourced for {request_id[:8]}")
                        continue

                    # Hash each lead and build commit entries
                    lead_dicts = [ld.model_dump(mode="json") for ld in leads]
                    hashes = [
                        {"hash": hash_lead(ld, HASH_SCHEMA_VERSION)}
                        for ld in lead_dicts
                    ]

                    try:
                        result = gateway_submit_fulfillment_commit(
                            self.wallet,
                            request_id,
                            hashes,
                            schema_version=HASH_SCHEMA_VERSION,
                        )
                        submission_id = result.get("submission_id", "")
                        if not submission_id:
                            print(f"   ⚠️ Commit rejected for {request_id[:8]}: {result}")
                            continue

                        state = {
                            "submission_id": submission_id,
                            "leads": lead_dicts,
                            "reveal_after": req.get("window_end", ""),
                        }
                        self._pending_fulfillment[request_id] = state
                        self._save_pending_fulfillment(request_id, state)
                        print(f"   ✅ Committed {len(hashes)} lead hashes for {request_id[:8]}... (reveal after {req.get('window_end', '?')})")
                    except Exception as e:
                        print(f"   ❌ Commit failed for {request_id[:8]}: {e}")

                # Phase 2: reveal any commits whose window has closed
                now = datetime.now(timezone.utc)
                revealed = []
                for rid, state in list(self._pending_fulfillment.items()):
                    reveal_after = state.get("reveal_after", "")
                    if not reveal_after:
                        continue
                    try:
                        deadline = datetime.fromisoformat(reveal_after.replace("Z", "+00:00"))
                    except (ValueError, TypeError):
                        continue

                    if now < deadline:
                        continue

                    print(f"\n📤 Revealing {len(state['leads'])} leads for {rid[:8]}...")
                    try:
                        gateway_reveal_fulfillment(
                            self.wallet,
                            rid,
                            state["submission_id"],
                            state["leads"],
                        )
                        print(f"   ✅ Reveal successful for {rid[:8]}!")
                        revealed.append(rid)
                    except Exception as e:
                        print(f"   ❌ Reveal failed for {rid[:8]}: {e}")
                        revealed.append(rid)

                for rid in revealed:
                    self._remove_pending_fulfillment(rid)

                await asyncio.sleep(poll_interval)

            except asyncio.CancelledError:
                bt.logging.info("Fulfillment loop cancelled")
                break
            except Exception as e:
                bt.logging.error(f"Fulfillment loop error: {e}")
                await asyncio.sleep(poll_interval)

    async def _source_fulfillment_leads(
        self,
        icp: dict,
        num_leads: int,
        miner_hotkey: str,
    ) -> list:
        """Source leads matching a fulfillment ICP using the fulfillment sourcer.

        Uses ScrapingDog + OpenRouter to discover real companies, find real
        decision-makers, and mine verifiable intent signals from the web.
        Falls back to the legacy get_leads pipeline if the sourcer fails.
        """
        from gateway.fulfillment.models import FulfillmentLead, IntentSignal

        try:
            from miner_models.Main_fulfillment_model.discovery import source_fulfillment_leads
            raw_leads = await source_fulfillment_leads(icp, num_leads=num_leads)
        except Exception as e:
            bt.logging.error(f"Fulfillment sourcer failed: {e}")
            raw_leads = []

        if not raw_leads:
            bt.logging.warning(f"No leads sourced for {icp.get('industry', '?')}")
            return []

        fulfillment_leads = []
        for lead_dict in raw_leads:
            try:
                signals = []
                for sig in lead_dict.get("intent_signals", []):
                    signals.append(IntentSignal(
                        source=sig.get("source", "other"),
                        description=sig.get("description", ""),
                        url=sig.get("url", ""),
                        date=sig.get("date"),
                        snippet=sig.get("snippet", "")[:1000],
                        # REQUIRED on submission: miner must declare which
                        # client-listed intent signal this evidence proves.
                        # Pass through whatever the upstream miner model
                        # produced; -1 default in the schema means "not set"
                        # and is rejected at Tier 3 scoring time.
                        matched_icp_signal=int(sig.get("matched_icp_signal", -1)),
                    ))
                if not signals:
                    bt.logging.warning(f"Skipping lead {lead_dict.get('full_name')} — no intent signals")
                    continue

                fl = FulfillmentLead(
                    full_name=lead_dict.get("full_name", ""),
                    email=lead_dict.get("email", ""),
                    linkedin_url=lead_dict.get("linkedin_url", ""),
                    phone=lead_dict.get("phone", ""),
                    business=lead_dict.get("business", ""),
                    company_linkedin=lead_dict.get("company_linkedin", ""),
                    company_website=lead_dict.get("company_website", ""),
                    employee_count=lead_dict.get("employee_count", ""),
                    company_hq_country=lead_dict.get("company_hq_country", ""),
                    company_hq_state=lead_dict.get("company_hq_state", ""),
                    company_hq_city=lead_dict.get("company_hq_city", ""),
                    industry=lead_dict.get("industry", ""),
                    sub_industry=lead_dict.get("sub_industry", ""),
                    country=lead_dict.get("country", ""),
                    city=lead_dict.get("city", ""),
                    state=lead_dict.get("state", ""),
                    role=lead_dict.get("role", ""),
                    role_type=lead_dict.get("role_type", "Sales"),
                    seniority=lead_dict.get("seniority", "VP"),
                    intent_signals=signals,
                )
                fulfillment_leads.append(fl)
                bt.logging.info(f"✅ Built FulfillmentLead: {fl.full_name} @ {fl.business}")
            except Exception as e:
                bt.logging.warning(f"Skipping lead — validation error: {e}")
                continue

        return fulfillment_leads

    async def _forward_async(self, synapse: LeadRequest) -> LeadRequest:
        import time as _t
        _t0 = _t.time()
        print("\n─────────  AXON ➜ MINER  ─────────")
        print(
            f"⚡  AXON call received  | leads={synapse.num_leads}"
            f" industry={synapse.industry or '∅'} region={synapse.region or '∅'}"
        )
        print(f"⏱️   at {datetime.utcnow().isoformat()} UTC")
        bt.logging.info(f" AXON CALL RECEIVED: {synapse}")

        start_time = time.time()

        try:
            print(
                f"\n🟡 RECEIVED QUERY from validator: {synapse.num_leads} leads, industry={synapse.industry}, region={synapse.region}"
            )
            print("⏸️  Stopping sourcing, switching to curation mode...")

            # Take the global lock so sourcing stays paused
            with self.sourcing_lock:
                self.sourcing_mode = False
                try:
                    target_ind = classify_industry(
                        synapse.business_desc) or synapse.industry
                    print(f"🔍 Target industry inferred: {target_ind or 'any'}")

                    # detect role keywords ONCE
                    desired_roles = classify_roles(synapse.business_desc)
                    if desired_roles:
                        print(f"🛈  Role filter active → {desired_roles}")

                    # pull a LARGE slice of the pool for this industry
                    pool_slice = get_leads_from_pool(
                        1000,  # big number = "all we have"
                        industry=target_ind,
                        region=synapse.region,
                        wallet=self.wallet  # ensures cloud read
                    )

                    # role-filter first, then random-sample down
                    if desired_roles:
                        pool_slice = [
                            ld for ld in pool_slice
                            if _role_match(ld.get("role", ""), desired_roles)
                        ] or pool_slice  # fall back if nothing matched

                    # finally down-sample to N×3 for ranking
                    curated_leads = random.sample(
                        pool_slice, min(len(pool_slice),
                                        synapse.num_leads * 3))

                    if not curated_leads:
                        print(
                            "📝 No leads found in pool, generating new leads..."
                        )
                        bt.logging.info(
                            "No leads found in pool, generating new leads")
                        new_leads = await get_leads(synapse.num_leads * 2,
                                                    target_ind, synapse.region)
                        
                        # Process leads through source provenance validation (protocol level)
                        validated_leads = await self.process_generated_leads(new_leads)
                        
                        # Sanitize validated leads
                        sanitized = [
                            sanitize_prospect(p,
                                              self.wallet.hotkey.ss58_address)
                            for p in validated_leads
                        ]
                        curated_leads = sanitized
                    else:
                        print(f" Curated {len(curated_leads)} leads in pool")

                    # Map the fields to match the API format and ensure all required fields are present
                    mapped_leads = []
                    for lead in curated_leads:
                        mapped_lead = {
                            "email": lead.get("email", ""),
                            "business": lead.get("business", ""),
                            "full_name": lead.get("full_name", ""),
                            "first": lead.get("first", ""),
                            "last": lead.get("last", ""),
                            "linkedin": lead.get("linkedin", ""),
                            "website": lead.get("website", ""),
                            "industry": lead.get("industry", ""),
                            "sub_industry": lead.get("sub_industry", ""),
                            "country": lead.get("country", ""),
                            "state": lead.get("state", ""),
                            "city": lead.get("city", ""),
                            "region": lead.get("region", ""),
                            "role": lead.get("role", ""),
                            "description": lead.get("description", ""),
                            "company_linkedin": lead.get("company_linkedin", ""),
                            "employee_count": lead.get("employee_count", ""),
                            "source": lead.get("source", ""),
                            "curated_by": self.wallet.hotkey.ss58_address,
                            "curated_at": datetime.now(timezone.utc).isoformat(),
                        }
                        # Only include leads that have all required fields
                        if all(
                                mapped_lead.get(field)
                                for field in ["email", "business"]):
                            mapped_leads.append(mapped_lead)

                    # apply business-intent ranking
                    ranked = await rank_leads(
                        mapped_leads, description=synapse.business_desc)
                    top_leads = ranked[:synapse.num_leads]

                    if not top_leads:
                        print("❌ No valid leads found in pool after mapping")
                        bt.logging.warning(
                            "No valid leads found in pool after mapping")
                        synapse.leads = []
                        synapse.dendrite.status_code = 404
                        synapse.dendrite.status_message = "No valid leads found matching criteria"
                        synapse.dendrite.process_time = str(time.time() -
                                                            start_time)
                        return synapse

                    print(
                        f"📤 SENDING {len(top_leads)} curated leads to validator:"
                    )
                    for i, lead in enumerate(top_leads, 1):
                        business = lead.get('business', 'Unknown')
                        score = lead.get('miner_intent_score', 0)
                        print(f"  {i}. {business} (intent={score:.3f})")

                    print("🚚 Returning leads over AXON")
                    print(
                        f"✅  Prepared {len(top_leads)} leads in"
                        f" {(_t.time()-_t0):.2f}s – sending back to validator")
                    bt.logging.info(f"Returning {len(top_leads)} scored leads")
                    synapse.leads = top_leads
                    synapse.dendrite.status_code = 200
                    synapse.dendrite.status_message = "OK"
                    synapse.dendrite.process_time = str(time.time() -
                                                        start_time)

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
            industry = data.get("industry")  # legacy field – may be empty
            region = data.get("region")
            business_desc = data.get("business_desc", "")

            print("⏸️  Stopping sourcing, switching to curation mode...")

            # Get leads from pool first
            target_ind = classify_industry(business_desc) or industry
            print(f"🔍 Target industry inferred: {target_ind or 'any'}")

            # detect role keywords ONCE
            desired_roles = classify_roles(business_desc)
            if desired_roles:
                print(f"🛈  Role filter active → {desired_roles}")

            # pull a LARGE slice of the pool for this industry
            pool_slice = get_leads_from_pool(
                1000,  # big number = "all we have"
                industry=target_ind,
                region=region,
                wallet=self.wallet  # <-- passes hotkey for auth
            )

            # role-filter first, then random-sample down
            if desired_roles:
                pool_slice = [
                    ld for ld in pool_slice
                    if _role_match(ld.get("role", ""), desired_roles)
                ] or pool_slice  # fall back if nothing matched

            # finally down-sample to N×3 for ranking
            curated_leads = random.sample(pool_slice,
                                          min(len(pool_slice), num_leads * 3))

            if not curated_leads:
                print("📝 No leads found in pool, generating new leads...")
                bt.logging.info("No leads found in pool, generating new leads")
                new_leads = await get_leads(num_leads * 2, target_ind, region)
                
                # Process leads through source provenance validation (protocol level)
                validated_leads = await self.process_generated_leads(new_leads)
                
                # Sanitize validated leads
                sanitized = [
                    sanitize_prospect(p, self.wallet.hotkey.ss58_address)
                    for p in validated_leads
                ]
                curated_leads = sanitized
            else:
                print(f" Found {len(curated_leads)} leads in pool")

            # Map the fields - FIXED VERSION
            mapped_leads = []
            for lead in curated_leads:
                # Map the fields correctly using the same keys as stored in pool
                mapped_lead = {
                    "email": lead.get("email", ""),
                    "business": lead.get("business", ""),
                    "full_name": lead.get("full_name", ""),
                    "first": lead.get("first", ""),
                    "last": lead.get("last", ""),
                    "linkedin": lead.get("linkedin", ""),
                    "website": lead.get("website", ""),
                    "industry": lead.get("industry", ""),
                    "sub_industry": lead.get("sub_industry", ""),
                    "role": lead.get("role", ""),
                    "country": lead.get("country", ""),
                    "state": lead.get("state", ""),
                    "city": lead.get("city", ""),
                    "region": lead.get("region", ""),
                    "description": lead.get("description", ""),
                    "company_linkedin": lead.get("company_linkedin", ""),
                    "employee_count": lead.get("employee_count", ""),
                    "source": lead.get("source", ""),
                    "curated_by": self.wallet.hotkey.ss58_address,
                }

                # Debug log to see what's happening
                bt.logging.debug(f"Original lead: {lead}")
                bt.logging.debug(f"Mapped lead: {mapped_lead}")

                # Only include leads that have all required fields
                if all(
                        mapped_lead.get(field)
                        for field in ["email", "business"]):
                    mapped_leads.append(mapped_lead)
                else:
                    bt.logging.warning(
                        f"Lead missing required fields: {mapped_lead}")

            if not mapped_leads:
                print("❌ No valid leads found in pool after mapping")
                bt.logging.warning(
                    "No valid leads found in pool after mapping")
                return web.json_response(
                    {
                        "leads": [],
                        "status_code": 404,
                        "status_message":
                        "No valid leads found matching criteria",
                        "process_time": "0"
                    },
                    status=404)

            # intent-rank
            print(" Ranking leads by intent...")
            ranked = await rank_leads(mapped_leads, description=business_desc)
            top_leads = ranked[:num_leads]

            print(f"📤 SENDING {len(top_leads)} curated leads to validator:")
            for i, lead in enumerate(top_leads, 1):
                business = lead.get('business', 'Unknown')
                score = lead.get('miner_intent_score', 0)
                print(f"  {i}. {business}  (intent={score:.3f})")

            print("▶️  Resuming sourcing mode...")

            bt.logging.info(
                f"Returning {len(top_leads)} leads to HTTP request")
            # send prospects to Firestore queue
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
            return web.json_response(
                {
                    "leads": [],
                    "status_code": 500,
                    "status_message": f"Error: {str(e)}",
                    "process_time": "0"
                },
                status=500)

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
            is_validator = (caller_uid is not None and bool(
                self.metagraph.validator_permit[caller_uid]))
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

    def find_available_port(self,
                            start_port: int,
                            max_attempts: int = 10) -> int:
        port = start_port
        for _ in range(max_attempts):
            if self.check_port_availability(port):
                return port
            port += 1
        raise RuntimeError(
            f"No available ports found between {start_port} and {start_port + max_attempts - 1}"
        )

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
        # this fires only when the request arrives via AXON
        print(
            f"🔔 AXON QUERY from {getattr(synapse.dendrite, 'hotkey', 'unknown')} | "
            f"{synapse.num_leads} leads | desc='{(synapse.business_desc or '')[:40]}…'"
        )
        # stop sourcing immediately
        self.pause_sourcing()
        result_holder = {}
        error_holder = {}

        def _runner():
            try:
                result_holder["res"] = asyncio.run(
                    self._forward_async(synapse))
            except Exception as e:
                error_holder["err"] = e

        t = threading.Thread(target=_runner, daemon=True)
        t.start()
        t.join(timeout=120)
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
        """
        Start the miner and run until interrupted.
        
        The miner uses wallet signature-based authentication via the trustless gateway.
        No JWT tokens or server-issued credentials are used (BRD Section 3.5).
        """
        bt.logging.info("Starting miner...")
        
        try:
            while True:
                # Sync metagraph and check miner status
                time.sleep(12)
                
        except KeyboardInterrupt:
            bt.logging.success("Miner killed by keyboard interrupt.")
            exit()
        except Exception as e:
            bt.logging.error(f"Miner error: {e}")
            bt.logging.error(traceback.format_exc())


# =============================================================================
# QUALIFICATION MODEL SUBMISSION
# =============================================================================

QUALIFICATION_GATEWAY_URL = os.environ.get("GATEWAY_URL", "http://52.91.135.79:8000")
QUALIFICATION_SUBMISSION_COST_USD = float(os.environ.get("QUALIFICATION_SUBMISSION_COST_USD", "10.0"))  # $10 submission cost

# =============================================================================
# LeadPoet Payment Wallet - COLDKEY addresses
# =============================================================================
# These are the coldkeys that receive qualification submission payments.
# The gateway verifies on-chain that payments go to the correct address.
#
# MAINNET (netuid 71):
#   Coldkey: 5ExoWGyajvzucCqS5GxZSpuzzXEzG1oNFcDqdW3sXeTujoD7
#
# TESTNET (netuid 401):
#   Validator Hotkey: 5CJyMxw6YJJvLhPf58gSpMB7mvSKSCMx9RXhXJum6cNfqMEz
#   Validator Coldkey: 5Gh5kw7rV1x7FDDd5E3Uc7YYMoeQtm4gn93c7VYeL5oUyoAD
#
LEADPOET_COLDKEY_MAINNET = "5ExoWGyajvzucCqS5GxZSpuzzXEzG1oNFcDqdW3sXeTujoD7"
LEADPOET_COLDKEY_TESTNET = "5Gh5kw7rV1x7FDDd5E3Uc7YYMoeQtm4gn93c7VYeL5oUyoAD"

def get_leadpoet_coldkey(netuid: int) -> str:
    """Get the correct LeadPoet coldkey based on netuid."""
    if netuid == 71:
        return LEADPOET_COLDKEY_MAINNET
    elif netuid == 401:
        return LEADPOET_COLDKEY_TESTNET
    else:
        # Unknown netuid - use testnet for safety
        print(f"⚠️  Unknown netuid {netuid}, using testnet coldkey")
        return LEADPOET_COLDKEY_TESTNET


def get_tao_price_sync() -> float:
    """
    Get current TAO price from CoinGecko (sync version for CLI).
    
    Raises:
        Exception: If CoinGecko API fails or is rate-limited
    """
    import requests
    import time
    
    # Try multiple times with delays between attempts
    # Retry schedule: 0s, wait 30s, wait 45s (total ~75s)
    retry_delays = [0, 30, 45]
    last_error = None
    
    for attempt in range(3):
        try:
            response = requests.get(
                "https://api.coingecko.com/api/v3/simple/price",
                params={"ids": "bittensor", "vs_currencies": "usd"},
                timeout=15 + (attempt * 5)  # 15s, 20s, 25s
            )
            response.raise_for_status()
            data = response.json()
            price = data.get("bittensor", {}).get("usd")
            if price:
                return float(price)
        except Exception as e:
            last_error = e
            if attempt < 2:
                delay = retry_delays[attempt + 1]
                print(f"   ⚠️  CoinGecko error: {e}")
                print(f"   ⏳ Waiting {delay}s before retry {attempt + 2}/3...")
                time.sleep(delay)
            else:
                print(f"❌ Could not fetch TAO price after 3 attempts: {e}")
    
    # No fallback - raise exception
    raise Exception(f"CoinGecko API failed: {last_error}")


def calculate_tao_required(usd_amount: float) -> float:
    """
    Calculate how much TAO is needed for a given USD amount.
    
    Returns:
        float: Amount of TAO required
        
    Raises:
        Exception: If TAO price cannot be fetched from CoinGecko
    """
    try:
        tao_price = get_tao_price_sync()
        tao_required = usd_amount / tao_price
        return round(tao_required, 6)  # Round to 6 decimals (nano precision)
    except Exception as e:
        # Re-raise with helpful message
        raise Exception(
            f"\n❌ Cannot calculate TAO amount: {e}\n\n"
            "CoinGecko is rate-limiting API requests. Please:\n"
            "   1. Wait 60 seconds and try again\n"
            "   2. OR check https://www.coingecko.com/en/coins/bittensor for current price\n"
            "   3. OR try again in a few minutes\n"
        )


def transfer_tao(wallet, dest_coldkey: str, amount_tao: float, subtensor) -> tuple:
    """
    Transfer TAO to destination coldkey using direct substrate calls.
    
    Uses substrate-interface directly to get the actual block hash
    and extrinsic index of the transfer (not just chain head).
    
    Returns:
        tuple: (success: bool, block_hash: str or None, extrinsic_index: int or None, error: str or None)
    """
    print(f"\n💸 Initiating TAO transfer...")
    print(f"   To: {dest_coldkey}")
    print(f"   Amount: {amount_tao:.6f} TAO")
    print("")
    
    # Retry logic for websocket errors (testnet can be flaky)
    max_retries = 3
    last_error = None
    
    for attempt in range(max_retries):
        try:
            if attempt > 0:
                print(f"   Retry {attempt + 1}/{max_retries}...")
                import time
                time.sleep(2)  # Wait before retry
            
            # Use direct substrate extrinsic for better control
            # This gives us the actual block hash containing our transfer
            
            # Create the transfer extrinsic
            call = subtensor.substrate.compose_call(
                call_module="Balances",
                call_function="transfer_keep_alive",
                call_params={
                    "dest": dest_coldkey,
                    "value": int(amount_tao * 1e9)  # Convert to rao
                }
            )
            
            # Create and sign the extrinsic
            extrinsic = subtensor.substrate.create_signed_extrinsic(
                call=call,
                keypair=wallet.coldkey
            )
            
            # Submit and wait for inclusion
            print(f"   Submitting transfer...")
            receipt = subtensor.substrate.submit_extrinsic(
                extrinsic,
                wait_for_inclusion=True,
                wait_for_finalization=False
            )
            
            if receipt.is_success:
                block_hash = receipt.block_hash
                extrinsic_index = receipt.extrinsic_idx
                print(f"✅ Transfer successful!")
                print(f"   Block hash: {block_hash}")
                print(f"   Extrinsic index: {extrinsic_index}")
                return True, block_hash, extrinsic_index, None
            else:
                error_msg = f"Transfer failed: {receipt.error_message}"
                print(f"❌ {error_msg}")
                return False, None, None, error_msg
                
        except Exception as e:
            last_error = str(e)
            # Check if it's a retryable websocket error
            if "close frame" in last_error.lower() or "websocket" in last_error.lower() or "connection" in last_error.lower():
                print(f"   ⚠️  Connection error: {last_error}")
                continue  # Retry
            else:
                # Non-retryable error
                print(f"❌ Transfer failed: {last_error}")
                return False, None, None, last_error
    
    # All retries exhausted
    print(f"❌ Transfer failed after {max_retries} attempts: {last_error}")
    return False, None, None, last_error


# =============================================================================
# Qualification Model Submission (Direct S3 Upload - Frontrunning Protected)
# =============================================================================

def create_model_tarball(model_path: str) -> tuple:
    """
    Create a tarball from the model directory and compute its hash.
    
    Args:
        model_path: Path to the model directory
    
    Returns:
        Tuple of (tarball_path, code_hash)
    """
    import tarfile
    import hashlib
    import tempfile
    import uuid
    
    # Expand user path (~) and make absolute
    model_path = os.path.abspath(os.path.expanduser(model_path))
    
    if not os.path.isdir(model_path):
        raise ValueError(f"Model path does not exist or is not a directory: {model_path}")
    
    # Create tarball in temp directory
    tarball_name = f"model_{uuid.uuid4().hex[:8]}.tar.gz"
    tarball_path = os.path.join(tempfile.gettempdir(), tarball_name)
    
    print(f"   Creating tarball from: {model_path}")
    
    # Create gzipped tarball
    with tarfile.open(tarball_path, "w:gz") as tar:
        tar.add(model_path, arcname="model")
    
    # Compute SHA256 hash
    sha256 = hashlib.sha256()
    with open(tarball_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    
    code_hash = sha256.hexdigest()
    file_size = os.path.getsize(tarball_path)
    
    print(f"   ✅ Tarball created: {tarball_path}")
    print(f"   ✅ Size: {file_size / (1024*1024):.2f} MB")
    print(f"   ✅ Code hash: {code_hash[:16]}...")
    
    return tarball_path, code_hash


def get_presigned_upload_url(wallet) -> dict:
    """
    Get a presigned URL from the gateway for direct S3 upload.
    
    Also checks rate limits - gateway will reject if limit reached.
    This happens BEFORE payment so miner doesn't waste TAO.
    
    Args:
        wallet: Bittensor wallet for signing
    
    Returns:
        dict with upload_url, s3_key, expires_in_seconds, rate_limit_info
        or dict with "error" and "rate_limit_exceeded" keys on failure
    """
    import requests
    import time
    
    hotkey = wallet.hotkey.ss58_address
    timestamp = int(time.time())
    
    # Create and sign the presign request
    presign_data = {
        "miner_hotkey": hotkey,
        "timestamp": timestamp,
    }
    message = json.dumps(presign_data, sort_keys=True)
    signature = wallet.hotkey.sign(message.encode()).hex()
    presign_data["signature"] = signature
    
    print(f"\n📡 Getting presigned upload URL from gateway...")
    print(f"   (This also checks your submission rate limit)")
    
    try:
        response = requests.post(
            f"{QUALIFICATION_GATEWAY_URL}/qualification/model/presign",
            json=presign_data,
            timeout=120
        )
        
        if response.status_code == 200:
            result = response.json()
            # Display rate limit info from response
            daily_used = result.get('daily_submissions_used', 0)
            daily_max = result.get('daily_submissions_max', 2)
            credits = result.get('submission_credits', 0)
            print(f"   📊 Submissions today: {daily_used}/{daily_max}")
            if credits > 0:
                print(f"   💳 Retry credits available: {credits}")
            print(f"   ✅ Got presigned URL (expires in {result.get('expires_in_seconds')}s)")
            return result
            
        elif response.status_code == 429:
            # Rate limit exceeded - display friendly message
            error_data = response.json() if response.headers.get('content-type', '').startswith('application/json') else {"detail": response.text}
            error_msg = error_data.get('detail', 'Rate limit exceeded')
            print(f"\n❌ RATE LIMIT EXCEEDED")
            print(f"   {error_msg}")
            print(f"\n   ⚠️  You cannot submit until the limit resets at midnight UTC.")
            print(f"   💡 TIP: Wait until tomorrow or use retry credits if available.")
            return {"error": error_msg, "rate_limit_exceeded": True}
            
        else:
            error = response.json() if response.headers.get('content-type', '').startswith('application/json') else response.text
            print(f"   ❌ Failed to get presigned URL: {response.status_code}")
            print(f"      Error: {error}")
            return {"error": error}
            
    except Exception as e:
        print(f"   ❌ Error getting presigned URL: {e}")
        return {"error": str(e)}


def upload_to_s3_presigned(tarball_path: str, upload_url: str) -> bool:
    """
    Upload the tarball to S3 using the presigned URL.
    Includes retry logic for transient network/SSL errors.
    
    Args:
        tarball_path: Path to the local tarball file
        upload_url: Presigned S3 URL for PUT
    
    Returns:
        True on success, False on failure
    """
    import requests
    import time
    
    file_size = os.path.getsize(tarball_path)
    print(f"\n📤 Uploading model to S3 ({file_size / (1024*1024):.2f} MB)...")
    
    # Retry logic: attempt 1, wait 10s + attempt 2, wait 20s + attempt 3
    retry_delays = [0, 10, 20]  # First attempt immediately, then 10s, then 20s
    
    for attempt, delay in enumerate(retry_delays, 1):
        if delay > 0:
            print(f"   ⏳ Waiting {delay}s before retry {attempt}/3...")
            time.sleep(delay)
        
        try:
            print(f"   🔄 Upload attempt {attempt}/3...")
            with open(tarball_path, 'rb') as f:
                response = requests.put(
                    upload_url,
                    data=f,
                    headers={
                        'Content-Type': 'application/gzip',
                        'Content-Length': str(file_size)
                    },
                    timeout=300  # 5 minutes for large files
                )
            
            if response.status_code in [200, 201, 204]:
                print(f"   ✅ Upload successful!")
                return True
            else:
                print(f"   ⚠️  Attempt {attempt}/3 failed: HTTP {response.status_code}")
                print(f"      Response: {response.text[:200]}")
                if attempt == len(retry_delays):
                    print(f"   ❌ All upload attempts failed.")
                    return False
                    
        except Exception as e:
            print(f"   ⚠️  Attempt {attempt}/3 error: {e}")
            if attempt == len(retry_delays):
                print(f"   ❌ All upload attempts failed.")
                return False
    
    return False


def submit_qualification_model(wallet, s3_key: str, code_hash: str,
                                block_hash: str, extrinsic_index: int,
                                model_name: str = None) -> dict:
    """
    Submit a qualification model to the gateway (after S3 upload).
    
    NEW FLOW (Direct S3 Upload - Frontrunning Protected):
    This is called AFTER the model has been uploaded to S3.
    The gateway will verify the S3 object exists and hash matches.
    
    Args:
        wallet: Bittensor wallet
        s3_key: S3 object key from presign response
        code_hash: SHA256 hash of the tarball (computed locally)
        block_hash: Block hash containing the payment
        extrinsic_index: Extrinsic index within the block
        model_name: Optional human-readable name for the model
    
    Returns:
        dict with model_id on success, or error on failure
    """
    import requests
    
    # Prepare submission data
    import time
    hotkey = wallet.hotkey.ss58_address
    timestamp = int(time.time())
    
    submission_data = {
        "miner_hotkey": hotkey,
        "s3_key": s3_key,
        "code_hash": code_hash,
        "payment_block_hash": block_hash,
        "payment_extrinsic_index": extrinsic_index,
        "timestamp": timestamp,  # Required for signature verification
    }
    
    if model_name:
        submission_data["model_name"] = model_name
    
    # Sign the submission (exclude signature field)
    message = json.dumps(submission_data, sort_keys=True)
    signature = wallet.hotkey.sign(message.encode()).hex()
    submission_data["signature"] = signature
    
    print(f"\n📤 Submitting model to gateway...")
    print(f"   S3 Key: {s3_key}")
    print(f"   Code Hash: {code_hash[:16]}...")
    if model_name:
        print(f"   Model Name: {model_name}")
    
    try:
        response = requests.post(
            f"{QUALIFICATION_GATEWAY_URL}/qualification/model/submit",
            json=submission_data,
            timeout=600  # 10 minutes - payment verification + block propagation + event loop contention
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"   ✅ Model submitted successfully!")
            print(f"   Model ID: {result.get('model_id')}")
            print(f"   Status: {result.get('status')}")
            
            # Display rate limit info
            daily_remaining = result.get('daily_submissions_remaining')
            credits_remaining = result.get('submission_credits_remaining')
            if daily_remaining is not None or credits_remaining is not None:
                print(f"\n   📊 Rate Limit Status:")
                if daily_remaining is not None:
                    print(f"      Daily submissions remaining: {daily_remaining}/2")
                if credits_remaining is not None:
                    print(f"      Retry credits available: {credits_remaining}")
            
            return result
        else:
            error = response.json() if response.headers.get('content-type', '').startswith('application/json') else response.text
            print(f"   ❌ Submission failed: {response.status_code}")
            print(f"      Error: {error}")
            return {"error": error, "status_code": response.status_code}
            
    except Exception as e:
        print(f"   ❌ Submission error: {e}")
        return {"error": str(e)}


def run_qualification_submission_flow(wallet, config, netuid: int):
    """
    Run the interactive qualification model submission flow.
    
    NEW FLOW (Direct S3 Upload - Frontrunning Protected):
    1. Ask for model directory path (local folder with qualify.py)
    2. Create tarball and compute hash locally
    3. Get presigned S3 URL from gateway
    4. Upload tarball to S3
    5. Calculate required TAO
    6. Connect to chain (AFTER user input to avoid WebSocket timeout)
    7. Confirm and execute transfer
    8. Submit to gateway with s3_key + code_hash
    
    This prevents frontrunning because:
    - Code is uploaded to private S3 before any public disclosure
    - Hash is computed locally and verified by gateway
    
    Args:
        wallet: Bittensor wallet
        config: Bittensor config (used to create subtensor when needed)
        netuid: Network UID (71 for mainnet, 401 for testnet)
    """
    # Determine correct coldkey based on netuid
    dest_coldkey = get_leadpoet_coldkey(netuid)
    is_testnet = netuid == 401
    
    print("\n" + "="*80)
    print(" 🏆 QUALIFICATION MODEL SUBMISSION")
    if is_testnet:
        print(" 🧪 TESTNET MODE (netuid 401)")
    print("="*80)
    print("")
    print("Submit your lead qualification model to compete for 10% of subnet emissions!")
    print(f"Submission cost: ${QUALIFICATION_SUBMISSION_COST_USD:.2f} USD (paid in TAO)")
    print("")
    
    # Step 1: Get model directory path
    print("📦 Step 1: Enter the path to your model directory")
    print("   This should be a folder containing your qualify.py and any dependencies.")
    print("   Example: ~/my-lead-qualifier/  or  /Users/you/projects/qualifier")
    print("")
    print("   Required file structure:")
    print("   └── your-model/")
    print("       ├── qualify.py      # REQUIRED: Must define `qualify(icp)` (takes ICP dict, returns one CompanyOutput dict)")
    print("       ├── requirements.txt  # Optional: Dependencies")
    print("       └── ...             # Any other files your model needs")
    print("")
    print("   The model competition is single-path company-mode: surface a")
    print("   company from the open web that matches the ICP and has at least one")
    print("   verifiable intent signal.  Return a CompanyOutput dict (company +")
    print("   intent signals, NO contact / role / email / seniority fields —")
    print("   those are fulfillment-side concerns and are scored separately).")
    print("   Schema lives in gateway/qualification/models.py.  Extra/unknown")
    print("   fields cause instant score-0 validation failure (gaming prevention).")
    print("")
    
    model_path = input("   Model directory path: ").strip()
    
    if not model_path:
        print("❌ No path provided.")
        return False
    
    # Expand ~ and make absolute
    model_path = os.path.abspath(os.path.expanduser(model_path))
    
    if not os.path.isdir(model_path):
        print(f"❌ Directory does not exist: {model_path}")
        return False
    
    # Check for qualify.py
    qualify_file = os.path.join(model_path, "qualify.py")
    if not os.path.isfile(qualify_file):
        print(f"❌ Missing required file: qualify.py")
        print(f"   Expected at: {qualify_file}")
        return False
    
    print(f"   ✅ Found model directory: {model_path}")
    print(f"   ✅ Found qualify.py")
    
    # Step 2: Create tarball and compute hash
    print("\n📦 Step 2: Creating tarball and computing hash...")
    
    try:
        tarball_path, code_hash = create_model_tarball(model_path)
    except Exception as e:
        print(f"❌ Failed to create tarball: {e}")
        return False
    
    # Required: Get model name
    print("\n📝 Step 3: Model name (REQUIRED)")
    while True:
        model_name = input("   Enter a name for your model: ").strip()
        if model_name:
            break
        print("   ⚠️  Model name is required. Please enter a name.")
    
    # Step 4: Get presigned URL from gateway
    print("\n📡 Step 4: Getting upload URL from gateway...")
    
    presign_result = get_presigned_upload_url(wallet)
    
    if "error" in presign_result:
        if presign_result.get("rate_limit_exceeded"):
            # Rate limit message already displayed by get_presigned_upload_url
            pass
        else:
            print(f"❌ Failed to get upload URL: {presign_result['error']}")
        # Clean up tarball
        try:
            os.remove(tarball_path)
        except:
            pass
        return False
    
    upload_url = presign_result["upload_url"]
    s3_key = presign_result["s3_key"]
    
    # Step 5: Upload to S3
    print("\n📤 Step 5: Uploading model to S3...")
    
    upload_success = upload_to_s3_presigned(tarball_path, upload_url)
    
    # Clean up local tarball
    try:
        os.remove(tarball_path)
        print(f"   ✅ Cleaned up local tarball")
    except:
        pass
    
    if not upload_success:
        print("❌ Failed to upload model to S3.")
        return False
    
    # Step 5b: Check if miner already has credit (from previous failed submission)
    print("\n🔍 Checking submission credit status...")
    has_existing_credit = False
    try:
        credit_response = requests.get(
            f"{QUALIFICATION_GATEWAY_URL}/qualification/model/rate-limit/{wallet.hotkey.ss58_address}",
            timeout=10
        )
        if credit_response.status_code == 200:
            credit_status = credit_response.json()
            existing_credits = credit_status.get("submission_credits", 0)
            daily_used = credit_status.get("daily_submissions_used", 0)
            daily_max = credit_status.get("daily_submissions_max", 2)
            
            print(f"   📊 Daily submissions: {daily_used}/{daily_max}")
            print(f"   💳 Unused credits: {existing_credits}")
            
            if daily_used >= daily_max:
                print(f"\n❌ Daily submission limit reached ({daily_max}/day).")
                print(f"   Please wait until midnight UTC to submit again.")
                if existing_credits > 0:
                    print(f"   You have {existing_credits} credit(s) that will be available tomorrow.")
                return False
            
            if existing_credits > 0:
                has_existing_credit = True
                print(f"   ✅ You have {existing_credits} credit(s) from a previous payment!")
                print(f"   ⏭️  Skipping payment - will use existing credit.")
    except Exception as e:
        print(f"   ⚠️  Could not check credit status: {e}")
        print(f"   Proceeding with payment flow...")
    
    # Initialize payment variables
    block_hash = None
    extrinsic_index = None
    
    # Step 6-9: Payment (skip if has existing credit)
    if not has_existing_credit:
        # Step 6: Calculate TAO required
        print("\n💰 Step 6: Calculating payment...")
        try:
            tao_price = get_tao_price_sync()
            tao_required = calculate_tao_required(QUALIFICATION_SUBMISSION_COST_USD)
        except Exception as e:
            print(f"\n{e}")
            print("\n⚠️  Submission cancelled due to CoinGecko rate limiting.")
            print("   Your model is uploaded to S3 but not yet submitted.")
            print("   Please try again in 1-2 minutes.")
            return False
        
        print(f"   Current TAO price: ${tao_price:.2f}")
        print(f"   Submission cost: ${QUALIFICATION_SUBMISSION_COST_USD:.2f}")
        print(f"   TAO required: {tao_required:.6f} TAO")
        
        # Add 1% buffer for price fluctuation
        tao_with_buffer = round(tao_required * 1.01, 6)
        print(f"   TAO to send (with 1% buffer): {tao_with_buffer:.6f} TAO")
        
        # Step 7: Confirm transfer
        print(f"\n📝 Step 7: Confirm payment")
        print(f"   From wallet: {wallet.name}")
        print(f"   To: {dest_coldkey}")
        if is_testnet:
            print(f"   Network: TESTNET (subnet 401)")
        else:
            print(f"   Network: MAINNET (subnet 71)")
        print(f"   Amount: {tao_with_buffer:.6f} TAO (~${QUALIFICATION_SUBMISSION_COST_USD:.2f})")
        print("")
        print(f"   Model hash: {code_hash[:16]}...")
        print(f"   S3 location: {s3_key}")
        if model_name:
            print(f"   Model name: {model_name}")
        
        confirm = input("\n   Proceed with transfer? (Y/N): ").strip().upper()
        
        if confirm != "Y":
            print("\n❌ Transfer cancelled.")
            print("   Note: Your model is still uploaded to S3 but submission is not finalized.")
            return False
        
        # Step 8: Connect to chain NOW (after all user input to avoid WebSocket timeout)
        # Retry logic: attempt 1, wait 10s + attempt 2, wait 20s + attempt 3
        print("\n🔗 Connecting to chain...")
        subtensor = None
        retry_delays = [0, 10, 20]  # First attempt immediately, then 10s, then 20s
        
        for attempt, delay in enumerate(retry_delays, 1):
            if delay > 0:
                print(f"   ⏳ Waiting {delay}s before retry {attempt}/3...")
                import time
                time.sleep(delay)
            
            try:
                print(f"   🔄 Connection attempt {attempt}/3...")
                subtensor = bt.subtensor(config=config)
                print(f"   ✅ Connected to: {subtensor.chain_endpoint}")
                break  # Success - exit retry loop
            except Exception as e:
                print(f"   ⚠️  Attempt {attempt}/3 failed: {e}")
                if attempt == len(retry_delays):
                    print("   ❌ All connection attempts failed.")
                    print("   Your model is uploaded but payment could not be processed.")
                    print("   You can retry the submission later.")
                    return False
        
        if subtensor is None:
            print("   ❌ Failed to connect to chain after all retries.")
            print("   Your model is uploaded but payment could not be processed.")
            return False
        
        # Step 9: Execute transfer
        success, block_hash, extrinsic_index, error = transfer_tao(
            wallet=wallet,
            dest_coldkey=dest_coldkey,
            amount_tao=tao_with_buffer,
            subtensor=subtensor
        )
        
        if not success:
            print(f"\n❌ Payment failed: {error}")
            print("   Your model was NOT submitted (payment required).")
            return False
        
        print("\n✅ Payment confirmed! Finalizing submission with gateway...")
    else:
        # Using existing credit - no payment needed
        print("\n✅ Using existing credit! Finalizing submission with gateway...")
    
    # Step 10: Submit to gateway
    
    result = submit_qualification_model(
        wallet=wallet,
        s3_key=s3_key,
        code_hash=code_hash,
        block_hash=block_hash,
        extrinsic_index=extrinsic_index,
        model_name=model_name
    )
    
    if "error" in result:
        print(f"\n❌ Model submission failed: {result['error']}")
        print("   A submission credit has been preserved for retry.")
        return False
    
    print("\n" + "="*80)
    print(" 🎉 SUBMISSION COMPLETE!")
    print("="*80)
    print(f"   Model ID: {result.get('model_id')}")
    print(f"   Status: {result.get('status')}")
    print(f"   Code Hash: {code_hash[:24]}...")
    if model_name:
        print(f"   Model Name: {model_name}")
    
    # Show rate limit status
    daily_remaining = result.get('daily_submissions_remaining')
    credits_remaining = result.get('submission_credits_remaining')
    if daily_remaining is not None or credits_remaining is not None:
        print("")
        print("   📊 Daily Rate Limit:")
        if daily_remaining is not None:
            print(f"      Submissions remaining today: {daily_remaining}/2")
        if credits_remaining is not None:
            print(f"      Retry credits available: {credits_remaining}")
    
    print("")
    print("   Your model will be evaluated by validators against 20 ICPs (up to 5 companies per ICP)")
    print("   (one per industry, company-only).")
    print("   Check your model status at:")
    print(f"   {QUALIFICATION_GATEWAY_URL}/qualification/model/{result.get('model_id')}/status")
    print("")
    print("   If your model scores higher than the current champion by >5%,")
    print("   you'll become the new champion and receive 10% of subnet emissions!")
    print("="*80)
    
    return True


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
    """
    Sanitize and validate prospect fields + add regulatory attestations.
    
    Task 1.2: Appends attestation metadata from data/regulatory/miner_attestation.json
    to ensure every lead submission includes regulatory compliance information.
    """

    def strip_html(s):
        return re.sub('<.*?>', '', html.unescape(str(s))) if isinstance(
            s, str) else s

    def valid_url(url):
        return bool(re.match(r"^https?://[^\s]+$", url))

    # Get email and full_name with fallback to legacy names for backward compatibility
    email = prospect.get("email", prospect.get("Owner(s) Email", ""))
    full_name = prospect.get("full_name", prospect.get("Owner Full name", ""))
    
    sanitized = {
        "business":
        strip_html(prospect.get("business", prospect.get("Business", ""))),
        "full_name":
        strip_html(full_name),
        "first":
        strip_html(prospect.get("first", prospect.get("First", ""))),
        "last":
        strip_html(prospect.get("last", prospect.get("Last", ""))),
        "email":
        strip_html(email),  # Use consistent field name
        "linkedin":
        strip_html(prospect.get("linkedin", prospect.get("LinkedIn", ""))),
        "website":
        strip_html(prospect.get("website", prospect.get("Website", ""))),
        "industry":
        strip_html(prospect.get("industry", prospect.get("Industry", ""))),
        "role":
        strip_html(prospect.get("role", prospect.get("Title", ""))),
        "sub_industry":
        strip_html(
            prospect.get("sub_industry", prospect.get("Sub Industry", ""))),
        "country":
        strip_html(prospect.get("country", prospect.get("Country", ""))),
        "state":
        strip_html(prospect.get("state", prospect.get("State", ""))),
        "city":
        strip_html(prospect.get("city", prospect.get("City", ""))),
        "region":
        strip_html(prospect.get("region", prospect.get("Region", ""))),
        "description":
        strip_html(prospect.get("description", "")),
        "company_linkedin":
        strip_html(prospect.get("company_linkedin", prospect.get("Company LinkedIn", ""))),
        "phone_numbers":
        prospect.get("phone_numbers", []),
        "founded_year":
        prospect.get("founded_year", prospect.get("Founded Year", "")),
        "ownership_type":
        strip_html(prospect.get("ownership_type", prospect.get("Ownership Type", ""))),
        "company_type":
        strip_html(prospect.get("company_type", prospect.get("Company Type", ""))),
        "number_of_locations":
        prospect.get("number_of_locations", prospect.get("Number of Locations", "")),
        "employee_count":
        strip_html(prospect.get("employee_count", prospect.get("Employee Count", ""))),
        "socials":
        prospect.get("socials", {}),
        "source":
        miner_hotkey  # Add source field
    }

    if not valid_url(sanitized["linkedin"]):
        sanitized["linkedin"] = ""
    if not valid_url(sanitized["website"]):
        sanitized["website"] = ""

    # Load miner's attestation from subnet-level regulatory directory
    attestation_file = Path("data/regulatory/miner_attestation.json")
    if attestation_file.exists():
        try:
            with open(attestation_file, 'r') as f:
                attestation = json.load(f)
            terms_hash = attestation.get("terms_version_hash")
            wallet_ss58 = attestation.get("wallet_ss58")
        except Exception as e:
            bt.logging.warning(f"Failed to load attestation file: {e}")
            terms_hash = "NOT_ATTESTED"
            wallet_ss58 = miner_hotkey or "UNKNOWN"
    else:
        # Should never happen if TASK 1.1 is working, but handle gracefully
        bt.logging.warning("No attestation file found - miner should have accepted terms at startup")
        terms_hash = "NOT_ATTESTED"
        wallet_ss58 = miner_hotkey or "UNKNOWN"
    
    # Add regulatory attestation fields (per-submission metadata)
    sanitized.update({
        # Miner identity & attestation
        "wallet_ss58": wallet_ss58,
        "submission_timestamp": datetime.now(timezone.utc).isoformat(),
        "terms_version_hash": terms_hash,
        
        # Boolean attestations (implicit from terms acceptance)
        "lawful_collection": True,
        "no_restricted_sources": True,
        "license_granted": True,
        
        # Source provenance (Task 1.3 - may be added later)
        # These fields will be populated by process_generated_leads() in Task 1.3
        "source_url": prospect.get("source_url", ""),
        "source_type": prospect.get("source_type", ""),
        
        # Optional: Licensed resale fields (Task 1.4)
        "license_doc_hash": prospect.get("license_doc_hash", ""),
        "license_doc_url": prospect.get("license_doc_url", ""),
    })

    return sanitized


def log_sourcing(hotkey, num_prospects):
    """Log sourcing activity to sourcing_logs.json."""
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "hotkey": hotkey,
        "num_prospects": num_prospects
    }

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
                "hotkey":
                hotkey,
                "valid_prospects_count":
                valid_count,
                "last_updated":
                datetime.now(timezone.utc).isoformat()
            })
        with open(MINERS_LOG, "w") as f:
            json.dump(miners, f, indent=2)


async def run_miner(miner, miner_hotkey=None, interval=60, queue_maxsize=1000, mode="sourcing"):
    logging.getLogger('bittensor.subtensor').setLevel(logging.WARNING)
    logging.getLogger('bittensor.axon').setLevel(logging.WARNING)
    miner._loop = asyncio.get_running_loop()
    miner._bg_interval = interval
    miner._miner_hotkey = miner_hotkey

    tasks = []

    if mode == "sourcing":
        miner.sourcing_task = asyncio.create_task(
            miner.sourcing_loop(interval, miner_hotkey), name="sourcing_loop")
        tasks.append("sourcing_loop - Continuous lead sourcing via trustless gateway")

        if os.environ.get("ENABLE_FULFILLMENT", "false").lower() == "true":
            miner.fulfillment_task = asyncio.create_task(
                miner.fulfillment_loop(miner_hotkey), name="fulfillment_loop")
            tasks.append("fulfillment_loop - Lead fulfillment commit-reveal system")

    elif mode == "fulfillment":
        miner.fulfillment_task = asyncio.create_task(
            miner.fulfillment_loop(miner_hotkey), name="fulfillment_loop")
        tasks.append("fulfillment_loop - Lead fulfillment commit-reveal system (ONLY)")

    for i, t in enumerate(tasks, 1):
        print(f"   {i}. {t}")
    print(f"✅ Started {len(tasks)} background task(s) in {mode.upper()} mode")

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
        ok = await asyncio.get_running_loop().run_in_executor(
            None, _sync_probe)
        if ok:
            print(f"✅ gRPC preflight OK (sync) → {addr}")
            return True
    except Exception as e:
        print(f"❌ gRPC preflight FAIL → {addr} | {e}")
    return False


def _looks_like_raw_research_lab_secret(value: str) -> bool:
    lowered = (value or "").lower()
    return any(marker in lowered for marker in ("sk-or-", "openrouter_api_key", "raw_openrouter", "raw_secret", "service_role"))


def _research_lab_signed_payload(wallet, payload: dict) -> dict:
    message = json.dumps(payload, sort_keys=True)
    signature = wallet.hotkey.sign(message.encode()).hex()
    return {**payload, "signature": signature}


def _post_research_lab_json(path: str, payload: dict, *, timeout: int = 60) -> dict:
    response = requests.post(f"{QUALIFICATION_GATEWAY_URL.rstrip('/')}{path}", json=payload, timeout=timeout)
    if response.status_code >= 400:
        try:
            detail = response.json()
        except Exception:
            detail = response.text
        return {"error": detail, "status_code": response.status_code}
    return response.json()


def _register_research_lab_openrouter_key(wallet, status: dict) -> tuple[str, str] | None:
    if status.get("openrouter_key_registration_enabled"):
        raw_key = getpass.getpass("   OpenRouter API key (hidden; encrypted by gateway): ").strip()
        if not raw_key:
            print("❌ OpenRouter API key is required for paid auto-research loops.")
            return None
        if not raw_key.startswith("sk-or-v1-"):
            print("❌ OpenRouter API key must start with sk-or-v1-.")
            return None
        import time
        payload = _research_lab_signed_payload(
            wallet,
            {
                "miner_hotkey": wallet.hotkey.ss58_address,
                "timestamp": int(time.time()),
                "idempotency_key": f"research-openrouter-key:{wallet.hotkey.ss58_address}:{int(time.time())}",
                "openrouter_api_key": raw_key,
                "key_label": "research-lab-miner-key",
            },
        )
        result = _post_research_lab_json("/research-lab/openrouter-keys", payload, timeout=120)
        if "error" in result:
            print(f"❌ OpenRouter key verification failed: HTTP {result.get('status_code')}")
            print(f"   {result['error']}")
            return None
        print("✅ OpenRouter key verified and encrypted by gateway")
        return str(result["key_ref"]), "encrypted_ref"

    print("❌ This gateway does not have encrypted OpenRouter key registration enabled.")
    print("   Ask the operator to set RESEARCH_LAB_OPENROUTER_KEY_KMS_KEY_ID.")
    return None


def _brief_sanitized_ref_from_input(value: str) -> str:
    value = value.strip()
    if value.startswith("brief_sanitized:"):
        if _looks_like_raw_research_lab_secret(value):
            raise ValueError("brief_sanitized_ref cannot contain raw secret material")
        return value
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
    return f"brief_sanitized:sha256:{digest}"


def _brief_public_summary_from_input(value: str) -> str | None:
    value = value.strip()
    if value.startswith("brief_sanitized:"):
        return None
    return value[:2000]


def _research_lab_default_model_tier(default_tier: str, approved_tiers: dict) -> str:
    if default_tier:
        return default_tier
    if approved_tiers:
        return sorted(approved_tiers)[0]
    return "default"


def _research_lab_runtime_label(min_seconds: int, max_seconds: int) -> str:
    if min_seconds <= 0 and max_seconds <= 0:
        return ""

    def _minutes_label(seconds: int) -> str:
        minutes = max(1, round(seconds / 60))
        return f"{minutes} min"

    if min_seconds > 0 and max_seconds > 0 and min_seconds != max_seconds:
        return f"{_minutes_label(min_seconds)}-{_minutes_label(max_seconds)}"
    return _minutes_label(max(min_seconds, max_seconds))


def _research_lab_prompt_model_tier(default_tier: str, approved_tiers: dict) -> str:
    tiers = sorted(str(tier) for tier in approved_tiers) if approved_tiers else [default_tier or "default"]
    if len(tiers) <= 1:
        return tiers[0]
    print("   Available model tiers:")
    for index, tier in enumerate(tiers, start=1):
        print(f"     {index}. {tier}")
    default_choice = default_tier if default_tier in tiers else tiers[0]
    value = input(f"   Model tier [{default_choice}]: ").strip()
    if not value:
        return default_choice
    if value.isdigit():
        selected_index = int(value)
        if 1 <= selected_index <= len(tiers):
            return tiers[selected_index - 1]
    if value in tiers:
        return value
    print(f"❌ Unknown model tier. Using default: {default_choice}")
    return default_choice


def _research_lab_prompt_budget(label: str, *, default: float, minimum: float, maximum: float) -> float:
    while True:
        value = input(f"{label} [{default:.2f}]: ").strip()
        if not value:
            return float(default)
        try:
            parsed = float(value)
        except ValueError:
            print("❌ Enter a number.")
            continue
        if parsed < minimum or parsed > maximum:
            print(f"❌ Budget must be between ${minimum:.2f} and ${maximum:.2f}.")
            continue
        return parsed


def _research_lab_prompt_int(label: str, *, default: int, minimum: int, maximum: int) -> int:
    while True:
        value = input(f"{label} [{default}]: ").strip()
        if not value:
            return int(default)
        try:
            parsed = int(value)
        except ValueError:
            print("❌ Enter an integer.")
            continue
        if parsed < minimum or parsed > maximum:
            print(f"❌ Value must be between {minimum} and {maximum}.")
            continue
        return parsed


RESEARCH_LAB_RESEARCH_AREA_CHOICES: tuple[tuple[str, str, str], ...] = (
    ("1", "generalist", "Broad improvements across all lead types"),
    ("2", "healthcare", "Healthcare and life sciences leads"),
    ("3", "fintech", "Finance and fintech leads"),
    ("4", "industrial", "Industrial and manufacturing leads"),
    ("5", "crypto_infra", "Crypto and blockchain infrastructure leads"),
    ("6", "real_estate", "Real estate and property leads"),
)


def _research_lab_prompt_research_area(default: str = "generalist") -> str:
    labels_by_value = {
        value: label for _number, value, label in RESEARCH_LAB_RESEARCH_AREA_CHOICES
    }
    values_by_number = {
        number: value for number, value, _label in RESEARCH_LAB_RESEARCH_AREA_CHOICES
    }
    values_by_name = {
        value.replace("_", "-"): value for _number, value, _label in RESEARCH_LAB_RESEARCH_AREA_CHOICES
    }
    values_by_name.update(
        {value: value for _number, value, _label in RESEARCH_LAB_RESEARCH_AREA_CHOICES}
    )

    print("   Choose where this run should focus:")
    for number, value, label in RESEARCH_LAB_RESEARCH_AREA_CHOICES:
        default_marker = " (default)" if value == default else ""
        print(f"     {number}. {label}{default_marker}")
    while True:
        choice = input(
            f"   Research area [{labels_by_value.get(default, default)}]: "
        ).strip().lower()
        if not choice:
            return default
        if choice in values_by_number:
            return values_by_number[choice]
        normalized_choice = choice.replace(" ", "_")
        if normalized_choice in values_by_name:
            return values_by_name[normalized_choice]
        print("   Choose a number from the list, or press Enter for the default.")


def _research_lab_research_area_label(value: str) -> str:
    for _number, area_value, label in RESEARCH_LAB_RESEARCH_AREA_CHOICES:
        if area_value == value:
            return label
    return value.replace("_", " ")


def _execute_research_lab_payment(
    *,
    wallet,
    config,
    netuid: int,
    usd_amount: float,
    payment_label: str,
    confirm_transfer: bool = True,
) -> tuple[str, int] | None:
    """Submit a TAO payment and return the on-chain proof for gateway verification."""
    dest_coldkey = get_leadpoet_coldkey(netuid)
    print("")
    print(f"{payment_label}: ${usd_amount:.2f} USD-equivalent in TAO")
    print(f"Payment destination coldkey for netuid {netuid}: {dest_coldkey}")

    try:
        tao_price = get_tao_price_sync()
        tao_required = calculate_tao_required(usd_amount)
    except Exception as exc:
        print(f"\n{exc}")
        print("\nPayment cancelled because TAO pricing could not be fetched.")
        return None

    tao_with_buffer = round(tao_required * 1.01, 6)
    print(f"   Current TAO price: ${tao_price:.2f}")
    print(f"   TAO required: {tao_required:.6f}")
    print(f"   TAO to send with 1% buffer: {tao_with_buffer:.6f}")

    if confirm_transfer:
        confirm = input("   Proceed with TAO transfer? [Y/n]: ").strip().lower()
        if confirm in ("n", "no"):
            print("Payment cancelled.")
            return None

    print("\nConnecting to chain...")
    subtensor = None
    retry_delays = [0, 10, 20]
    for attempt, delay in enumerate(retry_delays, start=1):
        if delay > 0:
            import time

            print(f"   Waiting {delay}s before retry {attempt}/3...")
            time.sleep(delay)
        try:
            print(f"   Connection attempt {attempt}/3...")
            subtensor = bt.subtensor(config=config)
            print(f"   Connected to: {subtensor.chain_endpoint}")
            break
        except Exception as exc:
            print(f"   Attempt {attempt}/3 failed: {exc}")

    if subtensor is None:
        print("Could not connect to chain. Payment was not sent.")
        return None

    success, block_hash, extrinsic_index, error = transfer_tao(
        wallet=wallet,
        dest_coldkey=dest_coldkey,
        amount_tao=tao_with_buffer,
        subtensor=subtensor,
    )
    if not success or not block_hash or extrinsic_index is None:
        print(f"\nPayment failed: {error or 'missing payment proof'}")
        return None

    print("\nPayment submitted. Gateway will now verify it on-chain.")
    return str(block_hash), int(extrinsic_index)


def run_research_lab_auto_research_flow(wallet, config, netuid: int) -> None:
    """Run the miner-facing Research Lab auto-research entrypoint."""
    gateway_url = QUALIFICATION_GATEWAY_URL.rstrip("/")
    print("\n" + "="*80)
    print(" LEADPOET AUTO-RESEARCH LOOPS")
    print("="*80)
    print("")
    print(f"Miner hotkey: {wallet.hotkey.ss58_address}")
    print(f"Gateway: {gateway_url}")
    print("")
    print("Auto-research loops are the current miner research workflow.")
    print("Miners fund hosted model-improvement loops. Leadpoet runs the private")
    print("model, sealed benchmark, and server-side data providers.")
    print("")

    try:
        response = requests.get(f"{gateway_url}/research-lab/status", timeout=10)
        if response.status_code != 200:
            print(f"❌ Research Lab status unavailable: HTTP {response.status_code}")
            print(f"   {response.text[:300]}")
            return
        status = response.json()
    except Exception as exc:
        print(f"❌ Could not reach Research Lab gateway status: {exc}")
        return

    worker_status = status.get("hosted_worker") if isinstance(status.get("hosted_worker"), dict) else {}
    approved_tiers = worker_status.get("approved_model_tiers") if isinstance(worker_status.get("approved_model_tiers"), dict) else {}
    default_tier = str(worker_status.get("default_model_tier") or "default")
    default_budget = float(worker_status.get("default_compute_budget_usd") or 5.0)
    min_budget = float(worker_status.get("min_compute_budget_usd") or 1.0)
    max_budget = float(worker_status.get("max_compute_budget_usd") or 100.0)
    loop_fee = float(status.get("loop_start_fee_usd") or RESEARCH_LAB_DEFAULT_LOOP_START_FEE_USD)
    min_runtime_seconds = int(worker_status.get("auto_research_min_seconds") or 0)
    max_runtime_seconds = int(worker_status.get("auto_research_max_seconds") or 0)
    runtime_label = _research_lab_runtime_label(min_runtime_seconds, max_runtime_seconds)
    reimbursement_status = status.get("reimbursement") if isinstance(status.get("reimbursement"), dict) else {}

    print("Research Lab status:")
    ready = bool(status.get("api_enabled") and status.get("paid_loops_enabled") and status.get("hosted_runs_enabled"))
    print(f"  hosted auto-research: {'ready' if ready else 'not ready'}")
    print(f"  loop-start fee: ${loop_fee:.2f} USD-equivalent in TAO")
    print(f"  default compute budget: ${default_budget:.2f}")
    if runtime_label:
        print(f"  expected research runtime: {runtime_label}")
    if reimbursement_status:
        estimate = float(reimbursement_status.get("default_rebate_rate_estimate") or 0.0)
        epochs = int(reimbursement_status.get("reimbursement_epochs") or 0)
        included = "yes" if reimbursement_status.get("loop_start_fee_included") else "no"
        print(f"  estimated default reimbursement: {estimate * 100:.1f}% over {epochs} epochs")
        print(f"  loop-start fee reimbursed: {included}")

    if not status.get("api_enabled"):
        print("")
        print("Auto-research APIs are not enabled on this gateway yet.")
        return

    print("")
    run_choice = input("❓ Start new loop or top up an existing promising loop? [new/topup] ").strip().lower() or "new"
    if run_choice in ("topup", "top-up", "t"):
        _run_research_lab_topup_flow(
            wallet=wallet,
            config=config,
            netuid=netuid,
            status=status,
            default_tier=default_tier,
            default_budget=default_budget,
            min_budget=min_budget,
            max_budget=max_budget,
        )
        return
    if run_choice not in ("new", "n", "start"):
        print("❌ Choose either 'new' or 'topup'.")
        return

    print("")
    print("Research focus (optional).")
    print("Use this to suggest a broad model-improvement area; evaluation remains benchmark-wide.")
    print("Leave blank to let the system choose the highest-priority general improvement path.")
    print("Examples: evidence freshness scoring; reducing overbroad matches; source-routing quality.")
    brief_input = input("   Research focus [benchmark-wide model improvement]: ").strip()
    if not brief_input:
        brief_input = "default: benchmark-wide Leadpoet sourcing-model improvement"
    try:
        brief_sanitized_ref = _brief_sanitized_ref_from_input(brief_input)
        brief_public_summary = _brief_public_summary_from_input(brief_input)
    except ValueError as exc:
        print(f"❌ {exc}")
        return

    island = "generalist"
    requested_loop_count = 1
    research_model_tier = _research_lab_default_model_tier(default_tier, approved_tiers)
    requested_compute_budget_usd = default_budget
    max_compute_budget_usd = default_budget
    advanced = input("❓ Adjust advanced settings? [y/N]: ").strip().lower()
    if advanced in ("y", "yes"):
        island = _research_lab_prompt_research_area()
        requested_loop_count = _research_lab_prompt_int("   Requested loop count", default=1, minimum=1, maximum=100)
        research_model_tier = _research_lab_prompt_model_tier(default_tier, approved_tiers)
        requested_compute_budget_usd = _research_lab_prompt_budget(
            "   Compute budget USD",
            default=default_budget,
            minimum=min_budget,
            maximum=max_budget,
        )
        max_compute_budget_usd = requested_compute_budget_usd
    print(
        f"   Using: research area={_research_lab_research_area_label(island)}, "
        f"budget=${requested_compute_budget_usd:.2f}, "
        f"model_tier={research_model_tier}, loops={requested_loop_count}"
        + (f", expected_runtime={runtime_label}" if runtime_label else "")
    )

    key_result = _register_research_lab_openrouter_key(wallet, status)
    if key_result is None:
        return
    key_ref, key_handling = key_result

    import time
    ticket_payload = _research_lab_signed_payload(
        wallet,
        {
            "miner_hotkey": wallet.hotkey.ss58_address,
            "timestamp": int(time.time()),
            "idempotency_key": f"research-ticket:{wallet.hotkey.ss58_address}:{int(time.time())}",
            "island": island,
            "brief_sanitized_ref": brief_sanitized_ref,
            "brief_public_summary": brief_public_summary,
            "requested_loop_count": requested_loop_count,
            "loop_start_fee_required_usd": float(
                status.get("loop_start_fee_usd") or RESEARCH_LAB_DEFAULT_LOOP_START_FEE_USD
            ),
            "research_model_tier": research_model_tier,
            "requested_compute_budget_usd": requested_compute_budget_usd,
            "max_compute_budget_usd": max_compute_budget_usd,
            "miner_openrouter_key_ref": key_ref,
            "miner_openrouter_key_handling": key_handling,
        },
    )
    ticket_result = _post_research_lab_json("/research-lab/tickets", ticket_payload)
    if "error" in ticket_result:
        print(f"❌ Ticket creation failed: HTTP {ticket_result.get('status_code')}")
        print(f"   {ticket_result['error']}")
        return

    ticket_id = ticket_result["ticket_id"]
    print("✅ Research Lab ticket opened")
    print(f"   Ticket ID: {ticket_id}")
    print(f"   Event seq: {ticket_result.get('event_seq')}")

    if not status.get("paid_loops_enabled") or not status.get("hosted_runs_enabled"):
        print("")
        print("Paid hosted loop starts are not enabled on this gateway yet.")
        print("The ticket is open, but this miner process will not submit a paid loop-start request.")
        return

    start_now = input(
        f"❓ Pay the TAO loop-start fee of ${loop_fee:.2f} USD-equivalent and queue this run now? [Y/n]: "
    ).strip().lower()
    if start_now in ("n", "no"):
        return

    payment_result = _execute_research_lab_payment(
        wallet=wallet,
        config=config,
        netuid=netuid,
        usd_amount=loop_fee,
        payment_label="Loop-start fee",
        confirm_transfer=False,
    )
    if payment_result is None:
        return
    payment_block_hash, payment_extrinsic_index = payment_result

    loop_payload = _research_lab_signed_payload(
        wallet,
        {
            "miner_hotkey": wallet.hotkey.ss58_address,
            "timestamp": int(time.time()),
            "idempotency_key": f"research-loop-start:{ticket_id}:{int(time.time())}",
            "ticket_id": ticket_id,
            "payment_block_hash": payment_block_hash,
            "payment_extrinsic_index": payment_extrinsic_index,
            "miner_openrouter_key_ref": key_ref,
            "miner_openrouter_key_handling": key_handling,
            "miner_openrouter_preflight_status": "passed",
            "requested_loop_count": requested_loop_count,
            "research_model_tier": research_model_tier,
            "requested_compute_budget_usd": requested_compute_budget_usd,
            "max_compute_budget_usd": max_compute_budget_usd,
        },
    )
    loop_result = _post_research_lab_json("/research-lab/loop-start", loop_payload, timeout=600)
    if "error" in loop_result:
        print(f"❌ Loop-start failed: HTTP {loop_result.get('status_code')}")
        print(f"   {loop_result['error']}")
        return

    print("✅ Research Lab loop-start accepted")
    print(f"   Run ID: {loop_result.get('run_id')}")
    print(f"   Status: {loop_result.get('status')}")
    print(f"   Payment ref: {loop_result.get('payment_ref')}")


def _run_research_lab_topup_flow(
    *,
    wallet,
    config,
    netuid: int,
    status: dict,
    default_tier: str,
    default_budget: float,
    min_budget: float,
    max_budget: float,
) -> None:
    import time

    ticket_id = input("   Existing ticket ID: ").strip()
    continue_from_run_id = input("   Continue from run ID (optional): ").strip()
    research_model_tier = _research_lab_default_model_tier(default_tier, {})
    additional_compute_budget_usd = _research_lab_prompt_budget(
        "   Additional compute budget USD",
        default=default_budget,
        minimum=min_budget,
        maximum=max_budget,
    )
    advanced = input("❓ Adjust advanced top-up settings? [y/N]: ").strip().lower()
    if advanced in ("y", "yes"):
        worker_status = status.get("hosted_worker") if isinstance(status.get("hosted_worker"), dict) else {}
        approved_tiers = worker_status.get("approved_model_tiers") if isinstance(worker_status.get("approved_model_tiers"), dict) else {}
        research_model_tier = _research_lab_prompt_model_tier(default_tier, approved_tiers)
    key_result = _register_research_lab_openrouter_key(wallet, status)
    if key_result is None:
        return
    key_ref, key_handling = key_result

    if not ticket_id:
        print("❌ Ticket ID is required.")
        return
    payment_result = _execute_research_lab_payment(
        wallet=wallet,
        config=config,
        netuid=netuid,
        usd_amount=additional_compute_budget_usd,
        payment_label="Top-up payment",
    )
    if payment_result is None:
        return
    payment_block_hash, payment_extrinsic_index = payment_result

    topup_payload = _research_lab_signed_payload(
        wallet,
        {
            "miner_hotkey": wallet.hotkey.ss58_address,
            "timestamp": int(time.time()),
            "idempotency_key": f"research-loop-topup:{ticket_id}:{int(time.time())}",
            "ticket_id": ticket_id,
            "continue_from_run_id": continue_from_run_id or None,
            "payment_block_hash": payment_block_hash,
            "payment_extrinsic_index": payment_extrinsic_index,
            "additional_compute_budget_usd": additional_compute_budget_usd,
            "research_model_tier": research_model_tier,
            "topup_reason": "promising_needs_topup",
            "miner_openrouter_key_ref": key_ref,
            "miner_openrouter_key_handling": key_handling,
            "miner_openrouter_preflight_status": "passed",
        },
    )
    topup_result = _post_research_lab_json("/research-lab/loop-topups", topup_payload, timeout=600)
    if "error" in topup_result:
        print(f"❌ Top-up failed: HTTP {topup_result.get('status_code')}")
        print(f"   {topup_result['error']}")
        return
    print("✅ Research Lab top-up accepted")
    print(f"   New run ID: {topup_result.get('run_id')}")
    print(f"   Status: {topup_result.get('status')}")
    print(f"   Payment ref: {topup_result.get('payment_ref')}")

def main():
    parser = argparse.ArgumentParser(description="LeadPoet Miner")
    BaseMinerNeuron.add_args(parser)
    args = parser.parse_args()

    if args.logging_trace:
        bt.logging.set_trace(True)

    # Build config from args (using dot notation like validator.py)
    config = bt.Config()
    config.wallet = bt.Config()
    config.wallet.name = args.wallet_name
    config.wallet.hotkey = args.wallet_hotkey
    config.wallet.path = str(Path(args.wallet_path).expanduser()) if args.wallet_path else str(Path.home() / ".bittensor" / "wallets")
    config.netuid = args.netuid
    config.subtensor = bt.Config()
    config.subtensor.network = args.subtensor_network
    config.blacklist = bt.Config()
    config.blacklist.force_validator_permit = args.blacklist_force_validator_permit
    config.blacklist.allow_non_registered = args.blacklist_allow_non_registered
    config.neuron = bt.Config()
    config.neuron.epoch_length = args.neuron_epoch_length or 1000
    config.use_open_source_lead_model = args.use_open_source_lead_model

    # AXON NETWORKING
    # Bind locally on 0.0.0.0 but advertise the user-supplied external
    # IP/port on-chain so validators can connect over the Internet.
    config.axon = bt.Config()
    config.axon.ip = "0.0.0.0"  # listen on all interfaces
    config.axon.port = args.axon_port or 8091  # internal bind port
    if args.axon_ip:
        config.axon.external_ip = args.axon_ip  # public address
    if args.axon_port:
        config.axon.external_port = args.axon_port
        config.axon.port = args.axon_port

    ensure_data_files()

    from Leadpoet.utils.contributor_terms import (
        display_terms_prompt,
        verify_attestation,
        create_attestation_record,
        save_attestation,
        TERMS_VERSION_HASH
    )
    
    # Attestation stored locally (trustless gateway verifies from lead metadata)
    # BRD Section 5.1: "✅ No JWT tokens or server-issued credentials"
    attestation_file = Path("data/regulatory/miner_attestation.json")
    
    # Check if attestation exists
    if not attestation_file.exists():
        # First-time run - show full terms
        print("\n" + "="*80)
        print(" FIRST TIME SETUP: CONTRIBUTOR TERMS ACCEPTANCE REQUIRED")
        print("="*80)
        display_terms_prompt()
        
        response = input("\n❓ Do you accept these terms? (Y/N): ").strip().upper()
        
        if response != "Y":
            print("\n❌ Terms not accepted. Miner disabled.")
            print("   You must accept the Contributor Terms to participate in the Leadpoet network.")
            print("   Please review the terms at: https://leadpoet.com/contributor-terms\n")
            raise SystemExit(0)
        
        # Record attestation LOCALLY (gateway verifies via lead metadata)
        # Load wallet to get SS58 address
        try:
            temp_wallet = bt.wallet(config=config)
            wallet_address = temp_wallet.hotkey.ss58_address
        except Exception as e:
            bt.logging.error(f"❌ Could not load wallet for attestation: {e}")
            print("\n❌ Failed to load wallet. Cannot proceed without valid wallet.")
            raise SystemExit(1)
        
        attestation = create_attestation_record(wallet_address, TERMS_VERSION_HASH)
        
        # Store locally at subnet level
        save_attestation(attestation, attestation_file)
        print(f"\n✅ Terms accepted and recorded locally.")
        print(f"   Local: {attestation_file}")
        print(f"   Attestation metadata will be included in each lead submission.")
        print(f"   Gateway will verify attestations via wallet signatures (no JWT tokens).\n")
        
    else:
        # Verify existing attestation hash matches current version
        is_valid, message = verify_attestation(attestation_file, TERMS_VERSION_HASH)
        
        if not is_valid:
            print("\n" + "="*80)
            print(" ⚠️  TERMS HAVE BEEN UPDATED - RE-ACCEPTANCE REQUIRED")
            print("="*80)
            print(f"   Reason: {message}\n")
            
            display_terms_prompt()
            
            response = input("\n❓ Do you accept the updated terms? (Y/N): ").strip().upper()
            
            if response != "Y":
                print("\n❌ Updated terms not accepted. Miner disabled.")
                print("   You must accept the updated Contributor Terms to continue mining.\n")
                raise SystemExit(0)
            
            # Update attestation
            # Load wallet to get SS58 address
            try:
                temp_wallet = bt.wallet(config=config)
                wallet_address = temp_wallet.hotkey.ss58_address
            except Exception as e:
                bt.logging.error(f"❌ Could not load wallet for attestation: {e}")
                print("\n❌ Failed to load wallet. Cannot proceed without valid wallet.")
                raise SystemExit(1)
            
            attestation = create_attestation_record(wallet_address, TERMS_VERSION_HASH)
            attestation["updated_at"] = datetime.now(timezone.utc).isoformat()
            
            save_attestation(attestation, attestation_file)
            print(f"\n✅ Updated terms accepted and recorded locally.")
            print(f"   Local: {attestation_file}\n")
        else:
            bt.logging.info(f"✅ Contributor terms attestation valid (hash: {TERMS_VERSION_HASH[:16]}...)")
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # MINER MODE SELECTION
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    print("\n" + "="*80)
    print(" LEADPOET MINER — SELECT MODE")
    print("="*80)
    print("")
    print("  1. Auto Research  — Check hosted auto-research loop availability (default)")
    print("  2. Fulfillment    — Poll for client ICP requests and fulfill them")
    print("")
    print("  You can run multiple active modes simultaneously in separate terminals.")
    print("")

    mode_input = input("❓ Select mode (1/2) [default: 1]: ").strip()
    if mode_input not in ("1", "2"):
        mode_input = "1"

    miner_mode = {
        "1": "research_lab",
        "2": "fulfillment",
    }[mode_input]
    print(f"\n✅ Selected mode: {miner_mode.upper()}")

    if miner_mode == "research_lab":
        try:
            temp_wallet = bt.wallet(config=config)
            print(f"\n✅ Wallet loaded: {temp_wallet.hotkey.ss58_address}")
            run_research_lab_auto_research_flow(temp_wallet, config, config.netuid)
        except Exception as e:
            bt.logging.error(f"❌ Error during Research Lab mode: {e}")
            import traceback
            traceback.print_exc()
        print("\n👋 Done. Run the miner again to select another mode.")
        raise SystemExit(0)

    print("")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    # Create miner and run it properly on the Bittensor network
    miner = Miner(config=config)

    # Check if miner is properly registered
    print("🔍 Checking miner registration...")
    print(f"   Wallet: {miner.wallet.hotkey.ss58_address}")
    print(f"   NetUID: {config.netuid}")
    print(f"   UID: {miner.uid}")

    if miner.uid is None:
        print("❌ Miner is not registered on the network!")
        print("   Please register your wallet on subnet 71 first.")
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

    async def run_selected_mode():
        miner_hotkey = miner.wallet.hotkey.ss58_address
        interval = 60
        queue_maxsize = 1000
        await run_miner(miner, miner_hotkey, interval, queue_maxsize, mode=miner_mode)

    asyncio.run(run_selected_mode())


if __name__ == "__main__":
    main()
