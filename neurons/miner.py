import time
import asyncio
import threading
import argparse
import subprocess
import sys
import traceback

# Opt-in, fail-closed error monitoring (docs/sentry_error_monitoring.md).
# Complete no-op unless the LEADPOET_SENTRY_* environment gate is satisfied.
try:
    try:
        from leadpoet_observability import init_sentry as _init_sentry
    except ImportError:
        import os as _os
        import sys as _sys

        _sys.path.append(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
        from leadpoet_observability import init_sentry as _init_sentry

    _init_sentry(component="miner")
except Exception as _sentry_exc:  # error monitoring must never break the miner
    print(
        "leadpoet_sentry_wiring_skipped error=%s" % type(_sentry_exc).__name__,
        flush=True,
    )

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
from urllib.parse import urlparse

from research_lab.source_add_miner import (
    SOURCE_ADD_AUTH_TYPES,
    SOURCE_ADD_SOURCE_KIND_DESCRIPTIONS,
    SOURCE_ADD_SOURCE_KINDS,
    build_source_add_submission_docs,
    source_add_text_contains_credential_material,
    source_add_submission_ready,
)
from research_lab.source_add import source_add_contains_credential_material


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


GATEWAY_URL = os.environ.get("GATEWAY_URL", "https://gateway.subnet71.com")

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
    return any(
        marker in lowered
        for marker in (
            "sk-or-",
            "openrouter_api_key",
            "openrouter_management_key",
            "raw_openrouter",
            "raw_secret",
            "service_role",
        )
    )


def _research_lab_signed_payload(wallet, payload: dict) -> dict:
    message = json.dumps(payload, sort_keys=True)
    signature = wallet.hotkey.sign(message.encode()).hex()
    return {**payload, "signature": signature}


def _research_lab_source_add_signed_payload(wallet, payload: dict) -> dict:
    """Sign the complete credential-free SOURCE_ADD intake payload."""

    if payload.get("adapter_credential") or payload.get("adapter_credential_v2"):
        raise ValueError("miners must not submit SOURCE_ADD API credentials")
    sign_payload = {key: value for key, value in payload.items() if key != "signature"}
    if source_add_contains_credential_material(sign_payload):
        raise ValueError(
            "SOURCE_ADD submission appears to contain credential material"
        )
    message = json.dumps(sign_payload, sort_keys=True)
    signature = wallet.hotkey.sign(message.encode()).hex()
    return {**sign_payload, "signature": signature}


def _research_lab_insecure_gateway_allowed(url: str) -> bool:
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    if parsed.scheme == "https":
        return True
    if host in {"localhost", "127.0.0.1", "::1"}:
        return True
    return os.getenv("RESEARCH_LAB_ALLOW_INSECURE_GATEWAY", "").strip().lower() in {"1", "true", "yes"}


def _post_research_lab_json(path: str, payload: dict, *, timeout: int = 60) -> dict:
    if path.startswith("/research-lab/") and not _research_lab_insecure_gateway_allowed(GATEWAY_URL):
        return {
            "error": (
                "Refusing to send Research Lab signed payload over insecure gateway URL. "
                "Set GATEWAY_URL to an https:// gateway, or set "
                "RESEARCH_LAB_ALLOW_INSECURE_GATEWAY=true for local/dev testing only."
            ),
            "status_code": 0,
        }
    response = requests.post(f"{GATEWAY_URL.rstrip('/')}{path}", json=payload, timeout=timeout)
    if response.status_code >= 400:
        try:
            detail = response.json()
        except Exception:
            detail = response.text
        return {"error": detail, "status_code": response.status_code}
    return response.json()


def _source_add_submission_error_message(result: dict) -> str:
    """Return one safe, human-readable SOURCE_ADD gateway error."""

    error = result.get("error")
    if isinstance(error, dict):
        detail = error.get("detail")
        if isinstance(detail, dict):
            message = detail.get("message")
            if isinstance(message, str) and message.strip():
                return message.strip()
        if isinstance(detail, str) and detail.strip():
            return detail.strip()
    if isinstance(error, str) and error.strip():
        return error.strip()
    return "The gateway did not accept this submission."


def _get_research_lab_status(gateway_url: str) -> Optional[dict]:
    try:
        response = requests.get(f"{gateway_url.rstrip('/')}/research-lab/status", timeout=10)
        if response.status_code != 200:
            print(f"❌ Research Lab status unavailable: HTTP {response.status_code}")
            print(f"   {response.text[:300]}")
            return None
        return response.json()
    except Exception as exc:
        print(f"❌ Could not reach Research Lab gateway status: {exc}")
        return None


def _research_lab_prompt_required_text(label: str, *, max_length: int = 1000) -> str:
    while True:
        value = input(label).strip()
        if not value:
            print("❌ This field is required.")
            continue
        if len(value) > max_length:
            print(f"❌ Value must be at most {max_length} characters.")
            continue
        if _looks_like_raw_research_lab_secret(
            value
        ) or source_add_text_contains_credential_material(value):
            print("❌ This field appears to contain credential material. Remove secrets and retry.")
            continue
        return value


def _research_lab_prompt_optional_text(label: str, *, max_length: int = 1000) -> str:
    value = input(label).strip()
    if not value:
        return ""
    if len(value) > max_length:
        print(f"   Truncating to {max_length} characters.")
        value = value[:max_length]
    if _looks_like_raw_research_lab_secret(
        value
    ) or source_add_text_contains_credential_material(value):
        raise ValueError("optional text appears to contain credential material")
    return value


def _research_lab_prompt_source_add_auth_type() -> str:
    print("   Auth type:")
    for index, auth_type in enumerate(SOURCE_ADD_AUTH_TYPES, start=1):
        print(f"     {index}. {auth_type}")
    while True:
        value = input("   Choose auth type [none]: ").strip().lower()
        if not value:
            return "none"
        if value.isdigit() and 1 <= int(value) <= len(SOURCE_ADD_AUTH_TYPES):
            return SOURCE_ADD_AUTH_TYPES[int(value) - 1]
        if value in SOURCE_ADD_AUTH_TYPES:
            return value
        print("❌ Choose an auth type from the list.")


def _research_lab_prompt_source_add_endpoint_examples() -> list[dict[str, str]]:
    examples: list[dict[str, str]] = []
    print("")
    print("Endpoint examples help reviewers and the integration loop understand the API.")
    while True:
        print(f"   Endpoint example #{len(examples) + 1}")
        method = input("     Method [GET]: ").strip().upper() or "GET"
        path = _research_lab_prompt_required_text("     Path (example: /v1/search): ", max_length=160)
        purpose = _research_lab_prompt_required_text("     Purpose: ", max_length=300)
        example_query = _research_lab_prompt_required_text(
            "     Example query/body without secrets: ",
            max_length=500,
        )
        examples.append(
            {
                "method": method,
                "path": path,
                "purpose": purpose,
                "example_query": example_query,
            }
        )
        if len(examples) >= 12:
            print("   Maximum endpoint examples reached.")
            break
        another = input("   Add another endpoint example? [y/N]: ").strip().lower()
        if another not in {"y", "yes"}:
            break
    return examples


def _research_lab_prompt_source_add_third_party_refs() -> list[str]:
    raw = input("   Optional third-party reference URLs, comma-separated: ").strip()
    if not raw:
        return []
    refs = []
    for item in re.split(r"[\s,]+", raw):
        cleaned = item.strip()
        if cleaned:
            refs.append(cleaned)
    return refs[:8]


def run_research_lab_source_add_flow(
    wallet,
    config,
    netuid: int,
) -> Optional[bool]:
    """Miner-facing SOURCE_ADD API submission entrypoint.

    The gateway endpoint is launch-gated by RESEARCH_LAB_SOURCE_ADD_ENABLED.
    Until the operator enables it, this flow exits before collecting source
    details. API credentials are always operator-managed after submission.

    True means the gateway returned a complete durable-admission receipt.
    False means nothing was confirmed as saved. None means the miner cancelled.
    """

    gateway_url = GATEWAY_URL.rstrip("/")
    print("\n" + "=" * 80)
    print(" RESEARCH LAB — SUBMIT API SOURCE")
    print("=" * 80)
    print("")
    print(f"Miner hotkey: {wallet.hotkey.ss58_address}")
    print(f"Gateway: {gateway_url}")
    print("")

    status = _get_research_lab_status(gateway_url)
    if status is None:
        return False
    if not source_add_submission_ready(status):
        print("SOURCE_ADD submissions are not accepting new submissions on this gateway.")
        print("No source details were collected or sent.")
        return False

    print("This submits a structured source/API candidate for operator review.")
    print("Do not paste API keys into docs, endpoint examples, rate limits, or provenance notes.")
    print("Any required API credential is added later by an operator; miners never submit keys.")
    print("")

    source_name = _research_lab_prompt_required_text("   Source/API name: ", max_length=160)
    print("   Source kind:")
    for index, kind in enumerate(SOURCE_ADD_SOURCE_KINDS, start=1):
        print(f"     {index}. {kind} — {SOURCE_ADD_SOURCE_KIND_DESCRIPTIONS[kind]}")
    kind_input = input("   Choose source kind [web]: ").strip().lower()
    if not kind_input:
        source_kind = "web"
    elif kind_input.isdigit() and 1 <= int(kind_input) <= len(SOURCE_ADD_SOURCE_KINDS):
        source_kind = SOURCE_ADD_SOURCE_KINDS[int(kind_input) - 1]
    else:
        source_kind = kind_input

    api_base_url = _research_lab_prompt_required_text("   API base URL: ", max_length=500)
    documentation_url = _research_lab_prompt_required_text("   Documentation URL: ", max_length=500)
    auth_type = _research_lab_prompt_source_add_auth_type()
    rate_limit_notes = _research_lab_prompt_required_text("   Rate-limit notes: ", max_length=1000)
    endpoint_examples = _research_lab_prompt_source_add_endpoint_examples()
    try:
        data_provenance_notes = _research_lab_prompt_optional_text(
            "   Optional data provenance notes: ",
            max_length=1000,
        )
        third_party_refs = _research_lab_prompt_source_add_third_party_refs()
    except ValueError as exc:
        print(f"❌ Invalid source submission: {exc}")
        return False

    try:
        manifest, source_brief, idempotency_key, source_metadata = build_source_add_submission_docs(
            miner_hotkey=wallet.hotkey.ss58_address,
            source_name=source_name,
            source_kind=source_kind,
            api_base_url=api_base_url,
            documentation_url=documentation_url,
            auth_type=auth_type,
            endpoint_examples=endpoint_examples,
            rate_limit_notes=rate_limit_notes,
            data_provenance_notes=data_provenance_notes,
            third_party_refs=third_party_refs,
            credential_supplied=False,
        )
    except ValueError as exc:
        print(f"❌ Invalid source submission: {exc}")
        return False

    print("")
    print("Submission preview:")
    print(f"   Source: {source_name}")
    print(f"   Kind: {manifest.get('source_kind')}")
    print(f"   API base URL: {source_metadata.get('api_base_url')}")
    print(f"   Documentation URL: {source_metadata.get('documentation_url')}")
    print(f"   Auth type: {source_metadata.get('auth_type')}")
    print(f"   Endpoint examples: {len(source_metadata.get('endpoint_examples') or [])}")
    print(f"   Domains: {', '.join(manifest.get('declared_base_domains') or [])}")
    print("   API credentials: operator-managed")
    confirm = input("   Submit for SOURCE_ADD review? [y/N]: ").strip().lower()
    if confirm not in {"y", "yes"}:
        print("Cancelled. Nothing was sent or saved.")
        return None

    import time

    payload = {
        "miner_hotkey": wallet.hotkey.ss58_address,
        "timestamp": int(time.time()),
        "idempotency_key": idempotency_key,
        "manifest": manifest,
        "source_brief": source_brief,
        "source_metadata": source_metadata,
    }
    signed_payload = _research_lab_source_add_signed_payload(wallet, payload)
    result = _post_research_lab_json("/research-lab/source-adapters", signed_payload, timeout=180)
    if not isinstance(result, dict):
        print("❌ SOURCE_ADD submission failed: invalid gateway response")
        return False
    if "error" in result:
        print(f"❌ SOURCE_ADD submission failed: HTTP {result.get('status_code')}")
        print(f"   {_source_add_submission_error_message(result)}")
        return False

    submission_id = result.get("submission_id")
    adapter_id = result.get("adapter_id")
    stage = result.get("stage")
    expected_adapter_id = manifest.get("adapter_id")
    if (
        not isinstance(submission_id, str)
        or re.fullmatch(r"source_add_submission:[0-9a-f]{16}", submission_id)
        is None
        or not isinstance(adapter_id, str)
        or not adapter_id
        or adapter_id != expected_adapter_id
        or stage != "provenance_queued"
    ):
        print("❌ SOURCE_ADD submission failed: invalid admission receipt")
        return False

    print("✅ SOURCE_ADD submission received")
    print(f"   Submission ID: {submission_id}")
    print(f"   Adapter ID: {adapter_id}")
    print(f"   Stage: {stage}")
    if result.get("precheck_status"):
        print(f"   Precheck: {result.get('precheck_status')}")
    for reason in (result.get("precheck_reasons") or [])[:8]:
        print(f"     - {reason}")
    print(
        "   Run the miner again, select Submit SOURCE_ADD, then check your submissions."
    )
    return True


def run_research_lab_source_add_status_flow(wallet, config, netuid: int) -> None:
    """Show the signing miner's private, sanitized SOURCE_ADD decisions."""

    print("\n" + "=" * 80)
    print(" RESEARCH LAB — MY API SOURCE SUBMISSIONS")
    print("=" * 80)
    print("")
    print(f"Miner hotkey: {wallet.hotkey.ss58_address}")
    print("")

    cursor = None
    seen_cursors: set[str] = set()
    while True:
        now = int(time.time())
        payload = {
            "miner_hotkey": wallet.hotkey.ss58_address,
            "timestamp": now,
            "idempotency_key": (
                f"source-add-status:{wallet.hotkey.ss58_address}:{now}:"
                f"{cursor or 'latest'}"
            ),
            "request_kind": "source_add_status_v1",
            "limit": 20,
        }
        if cursor:
            payload["cursor"] = cursor
        result = _post_research_lab_json(
            "/research-lab/source-adapters/status",
            _research_lab_signed_payload(wallet, payload),
            timeout=60,
        )
        if "error" in result:
            print(f"❌ SOURCE_ADD status failed: HTTP {result.get('status_code')}")
            print("   Submission status is temporarily unavailable.")
            return

        submissions = result.get("submissions")
        if not isinstance(submissions, list):
            print("❌ SOURCE_ADD status returned an invalid response.")
            return
        if not submissions and cursor is None:
            print("No SOURCE_ADD submissions were found for this hotkey.")
            return
        if not submissions:
            print("No older SOURCE_ADD submissions were found.")
            return

        for item in submissions:
            if not isinstance(item, dict):
                print("❌ SOURCE_ADD status returned an invalid response.")
                return
            decision = str(item.get("decision_status") or "pending").upper()
            print(f"{decision}: {item.get('source_name') or 'API source'}")
            print(f"   Submission ID: {item.get('submission_id') or 'unavailable'}")
            print(f"   Submitted: {item.get('submitted_at') or 'unavailable'}")
            print(f"   Reason: {item.get('decision_reason') or 'Status unavailable.'}")
            reward_status = str(item.get("reward_status") or "not_decided")
            alpha_percent = item.get("alpha_percent")
            reward_epochs = item.get("reward_epochs")
            start_epoch = item.get("start_epoch")
            end_epoch = item.get("end_epoch")
            if all(
                value is not None
                for value in (
                    alpha_percent,
                    reward_epochs,
                    start_epoch,
                    end_epoch,
                )
            ):
                print(
                    "   Reward: "
                    f"{float(alpha_percent):g}% per epoch for "
                    f"{int(reward_epochs)} epochs "
                    f"({int(start_epoch)}–{int(end_epoch)}; {reward_status})"
                )
            else:
                print(f"   Reward: {reward_status.replace('_', ' ')}")
            print("")

        next_cursor = result.get("next_cursor")
        if not isinstance(next_cursor, str) or not next_cursor:
            return
        if next_cursor in seen_cursors:
            print("❌ SOURCE_ADD status pagination did not advance.")
            return
        if input("Show older submissions? [y/N]: ").strip().lower() not in {
            "y",
            "yes",
        }:
            return
        seen_cursors.add(next_cursor)
        cursor = next_cursor


def _run_research_lab_source_add_mode(config) -> int:
    """Run SOURCE_ADD once and make the process status match persistence."""

    try:
        temp_wallet = bt.Wallet(config=config)
        print(f"\n✅ Wallet loaded: {temp_wallet.hotkey.ss58_address}")
        outcome = run_research_lab_source_add_flow(
            temp_wallet,
            config,
            config.netuid,
        )
    except Exception as exc:
        bt.logging.error(f"❌ Error during source-add mode: {exc}")
        traceback.print_exc()
        outcome = False

    if outcome is True:
        print("\n👋 Done. Run the miner again to select another mode.")
        return 0
    if outcome is None:
        return 0

    print("\n❌ SOURCE_ADD submission NOT SAVED.")
    print("   Only a displayed Submission ID confirms that the gateway accepted it.")
    return 1


def _choose_primary_miner_mode(input_fn=input, output_fn=print) -> str:
    """Return one of the two primary submission modes or SOURCE_ADD status."""

    output_fn("")
    output_fn("=" * 80)
    output_fn(" LEADPOET MINER — SELECT ACTION")
    output_fn("=" * 80)
    output_fn("")
    output_fn("  1. Submit SOURCE_ADD — Submit or check an API/source candidate")
    output_fn("  2. Submit Model — Submit model source and run credentials (default)")
    output_fn("")
    selection = input_fn("Select action (1/2) [default: 2]: ").strip()
    if selection != "1":
        return "agent_competition"

    output_fn("")
    output_fn(" SOURCE_ADD")
    output_fn("  1. Submit a new API/source candidate (default)")
    output_fn("  2. Check my submissions, decisions, and rewards")
    output_fn("")
    source_add_selection = input_fn(
        "Select SOURCE_ADD action (1/2) [default: 1]: "
    ).strip()
    if source_add_selection == "2":
        return "research_lab_source_add_status"
    return "research_lab_source_add"


def main():
    parser = argparse.ArgumentParser(description="LeadPoet Miner")
    BaseMinerNeuron.add_args(parser)
    parser.add_argument(
        "--mode",
        choices=("menu", "fulfillment"),
        default="menu",
        help="use fulfillment only for the legacy background worker",
    )
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
            temp_wallet = bt.Wallet(config=config)
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
                temp_wallet = bt.Wallet(config=config)
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
    
    # Fulfillment remains callable for existing operators, but it is not a
    # primary submission action in the miner menu.
    miner_mode = (
        "fulfillment"
        if args.mode == "fulfillment"
        else _choose_primary_miner_mode()
    )
    print(f"\n✅ Selected mode: {miner_mode.upper()}")

    if miner_mode == "agent_competition":
        try:
            temp_wallet = bt.Wallet(config=config)
            print(f"\n✅ Wallet loaded: {temp_wallet.hotkey.ss58_address}")
            helper = Path(__file__).resolve().parents[1] / "scripts" / "lab_arena_miner.py"
            result = subprocess.run(
                [
                    sys.executable,
                    str(helper),
                    "interactive",
                    "--api-base-url",
                    os.environ.get("LAB_ARENA_API_BASE_URL", GATEWAY_URL),
                    "--wallet-name",
                    str(config.wallet.name),
                    "--hotkey-name",
                    str(config.wallet.hotkey),
                    "--wallet-path",
                    str(config.wallet.path),
                ],
                check=False,
            )
            submitted = result.returncode == 0
        except Exception as e:
            bt.logging.error(f"❌ Error during Agent Competition mode: {e}")
            import traceback
            traceback.print_exc()
            submitted = False
        print("\n👋 Done. Run the miner again to select another mode.")
        raise SystemExit(0 if submitted else 1)

    if miner_mode == "research_lab_source_add":
        raise SystemExit(_run_research_lab_source_add_mode(config))

    if miner_mode == "research_lab_source_add_status":
        try:
            temp_wallet = bt.Wallet(config=config)
            print(f"\n✅ Wallet loaded: {temp_wallet.hotkey.ss58_address}")
            run_research_lab_source_add_status_flow(
                temp_wallet,
                config,
                config.netuid,
            )
        except Exception as e:
            bt.logging.error(f"❌ Error during source-add status mode: {e}")
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
