"""
Database helper for the LeadPoet subnet with Supabase integration.
Miners = write-only to prospect_queue, Validators = read prospect_queue, write to leads.
"""
import os
import json
import time
import base64
import requests
import bittensor as bt
from typing import List, Dict
from datetime import datetime, timezone
from dotenv import load_dotenv
from Leadpoet.utils.misc import generate_timestamp
from Leadpoet.utils.utils_lead_extraction import get_email, get_field

load_dotenv()

# Gateway URL (TEE-based trustless gateway on AWS EC2)
GATEWAY_URL = os.getenv("GATEWAY_URL", "http://54.226.209.164:8000")
API_URL   = os.getenv("LEAD_API", "https://leadpoet-api-511161415764.us-central1.run.app")

# Network defaults - can be overridden via environment variables
SUBNET_ID = int(os.getenv("NETUID", "71"))  # Default to mainnet subnet 71
NETWORK   = os.getenv("SUBTENSOR_NETWORK", "finney")  # Default to mainnet

SUPABASE_URL = "https://qplwoislplkcegvdmbim.supabase.co"
SUPABASE_JWT = os.getenv("SUPABASE_JWT")

# Supabase anon key for API routing (public, safe to commit)
SUPABASE_ANON_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InFwbHdvaXNscGxrY2VndmRtYmltIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDQ4NDcwMDUsImV4cCI6MjA2MDQyMzAwNX0.5E0WjAthYDXaCWY6qjzXm2k20EhadWfigak9hleKZk8"

# Create a response object similar to what supabase-py returns
class RPCResponse:
    def __init__(self, data):
        self.data = data

    def execute(self):
        return self

class CustomSupabaseClient:
    """
    Custom Supabase client that uses direct HTTP requests to Postgrest API.
    This ensures our custom JWT reaches the database for RLS policy evaluation.
    """
    def __init__(self, url: str, jwt: str, anon_key: str):
        self.url = url
        self.jwt = jwt
        self.anon_key = anon_key
        self.postgrest_url = f"{url}/rest/v1"
        
    def table(self, table_name: str):
        """Return a table query builder."""
        return CustomTableQuery(self.postgrest_url, table_name, self.jwt, self.anon_key)
    
    def rpc(self, function_name: str, params: dict = None):
        """Call a PostgreSQL function via PostgREST RPC."""
        import requests
        url = f"{self.postgrest_url}/rpc/{function_name}"
        headers = {
            "Authorization": f"Bearer {self.jwt}",
            "apikey": self.anon_key,
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        
        try:
            response = requests.post(url, json=params or {}, headers=headers)
            response.raise_for_status()
            
            return RPCResponse(response.json() if response.text else [])
        except requests.exceptions.HTTPError as e:
            bt.logging.error(f"RPC call failed: {e}")
            if e.response is not None:
                bt.logging.error(f"Response status: {e.response.status_code}")
                bt.logging.error(f"Response body: {e.response.text}")
                bt.logging.error(f"Response headers: {dict(e.response.headers)}")
            # Return empty response on error
            return RPCResponse([])

class CustomTableQuery:
    """Query builder for table operations using direct HTTP requests."""
    def __init__(self, postgrest_url: str, table_name: str, jwt: str, anon_key: str):
        self.postgrest_url = postgrest_url
        self.table_name = table_name
        self.jwt = jwt
        self.anon_key = anon_key
        self._select_cols = "*"
        self._filters = []
        self._order = None
        self._limit_val = None
        
    def select(self, cols: str = "*"):
        """Set columns to select."""
        self._select_cols = cols
        return self
    
    def eq(self, column: str, value):
        """Add equality filter."""
        self._filters.append(f"{column}=eq.{value}")
        return self
    
    def in_(self, column: str, values: list):
        """Add IN filter."""
        vals_str = ",".join(str(v) for v in values)
        self._filters.append(f"{column}=in.({vals_str})")
        return self
    
    def gte(self, column: str, value):
        """Add greater than or equal filter."""
        self._filters.append(f"{column}=gte.{value}")
        return self
    
    def lt(self, column: str, value):
        """Add less than filter."""
        self._filters.append(f"{column}=lt.{value}")
        return self
    
    def not_(self):
        """Add NOT modifier - returns a NotFilter wrapper."""
        return NotFilter(self)
    
    def order(self, column: str, desc: bool = False):
        """Set order."""
        self._order = f"{column}.{'desc' if desc else 'asc'}"
        return self
    
    def limit(self, n: int):
        """Set limit."""
        self._limit_val = n
        return self
    
    def insert(self, data):
        """Execute INSERT with custom JWT."""
        url = f"{self.postgrest_url}/{self.table_name}"
        headers = {
            "Authorization": f"Bearer {self.jwt}",
            "apikey": self.anon_key,
            "Content-Type": "application/json",
            "Prefer": "return=minimal"
        }
        response = requests.post(url, json=data, headers=headers, timeout=30)
        
        return CustomResponse(response)
    
    def upsert(self, data, on_conflict=None):
        """
        Execute UPSERT (INSERT with conflict resolution) with custom JWT.
        
        Args:
            data: Data to upsert (dict or list of dicts)
            on_conflict: Comma-separated list of column names for conflict resolution
        
        Returns:
            CustomResponse with upserted data
        """
        url = f"{self.postgrest_url}/{self.table_name}"
        
        headers = {
            "Authorization": f"Bearer {self.jwt}",
            "apikey": self.anon_key,
            "Content-Type": "application/json",
            "Prefer": "return=representation,resolution=merge-duplicates"
        }
        
        # If on_conflict specified, add it as query parameter
        if on_conflict:
            url += f"?on_conflict={on_conflict}"
        
        response = requests.post(url, json=data, headers=headers, timeout=30)
        
        return CustomResponse(response)
    
    def update(self, data):
        """Execute UPDATE with custom JWT."""
        url = f"{self.postgrest_url}/{self.table_name}"
        if self._filters:
            url += "?" + "&".join(self._filters)
        
        headers = {
            "Authorization": f"Bearer {self.jwt}",
            "apikey": self.anon_key,
            "Content-Type": "application/json",
            "Prefer": "return=representation"
        }
        
        response = requests.patch(url, json=data, headers=headers, timeout=30)
        
        return CustomResponse(response)
    
    def execute(self):
        """Execute SELECT query."""
        # Build query parameters
        params = {"select": self._select_cols}
        
        # Add filters (they're already in the right format, e.g. "status=eq.pending")
        if self._filters:
            for filter_str in self._filters:
                # Parse filter string "column=op.value" into param
                parts = filter_str.split("=", 1)
                if len(parts) == 2:
                    params[parts[0]] = parts[1]
        
        if self._order:
            params["order"] = self._order
        if self._limit_val:
            params["limit"] = str(self._limit_val)
        
        url = f"{self.postgrest_url}/{self.table_name}"
        
        headers = {
            "Authorization": f"Bearer {self.jwt}",
            "apikey": self.anon_key,
            "Content-Type": "application/json"
        }
        
        # Use params instead of building URL manually
        response = requests.get(url, headers=headers, params=params, timeout=30)
        
        return CustomResponse(response)

class CustomResponse:
    """Response wrapper to match supabase-py API."""
    def __init__(self, response: requests.Response):
        self.response = response
        if response.status_code >= 400:
            # Parse error response
            try:
                error_data = response.json()
                raise Exception(error_data)
            except Exception:
                response.raise_for_status()
        
        # Parse success response
        try:
            self.data = response.json() if response.text else []
        except Exception:
            self.data = []

class NotFilter:
    """Wrapper for NOT filters in PostgREST."""
    def __init__(self, parent_query):
        self.parent_query = parent_query
    
    def contains(self, column: str, values: list):
        """Add NOT contains filter (for array columns)."""
        # PostgREST syntax for NOT array contains
        vals_str = ",".join(f'"{v}"' if isinstance(v, str) else str(v) for v in values)
        self.parent_query._filters.append(f"{column}=not.cs.{{{vals_str}}}")
        return self.parent_query

def get_supabase_client():
    """
    Get custom Supabase client with JWT-based RLS support.
    Uses direct HTTP requests to ensure JWT reaches database.
    
    Returns:
        CustomSupabaseClient instance or None if not configured
    """
    try:
        jwt = os.getenv("SUPABASE_JWT")
        if not jwt:
            bt.logging.debug("No SUPABASE_JWT found - Supabase client not available")
            return None
        
        # Decode JWT to log role (minimal logging)
        try:
            import jwt as pyjwt
            decoded = pyjwt.decode(jwt, options={"verify_signature": False})
            role = decoded.get('app_role', 'unknown')
            bt.logging.debug(f"Supabase client created - role: {role}")
        except Exception:
            pass
        
        # Create custom client that uses direct HTTP requests
        client = CustomSupabaseClient(SUPABASE_URL, jwt, SUPABASE_ANON_KEY)
        
        bt.logging.debug("✅ Custom Supabase client created with direct HTTP + JWT")
        return client
        
    except Exception as e:
        bt.logging.error(f"Error creating Supabase client: {e}")
        import traceback
        traceback.print_exc()
        return None

class _Verifier:
    """
    Lightweight on-chain permission checks.
    
    Supports both sync and async metagraph queries:
    - Sync methods: Create new subtensor instance (for backward compatibility)
    - Async methods: Use injected async subtensor (no memory leaks)
    """
    def __init__(self):
        self._network = NETWORK
        self._netuid = SUBNET_ID
        
        # Async subtensor instance (injected from validator)
        self._async_subtensor = None
    
    def inject_async_subtensor(self, async_subtensor):
        """
        Inject async subtensor instance from validator.
        
        Called from neurons/validator.py after initializing async subtensor.
        Allows verifier to use shared instance (no memory leaks).
        
        Args:
            async_subtensor: AsyncSubtensor instance from validator
        
        Example:
            # In neurons/validator.py run_async():
            from Leadpoet.utils import cloud_db
            cloud_db._VERIFY.inject_async_subtensor(self.async_subtensor)
        """
        self._async_subtensor = async_subtensor
        bt.logging.info(f"✅ AsyncSubtensor injected into _Verifier (network: {async_subtensor.network})")
    
    async def _get_fresh_metagraph_async(self, network=None, netuid=None):
        """
        Get metagraph using injected async subtensor (ASYNC VERSION).
        
        Use this from async contexts to avoid memory leaks.
        
        Args:
            network: Network name (ignored - uses injected instance's network)
            netuid: Subnet ID (default: self._netuid)
        
        Returns:
            Metagraph object
        
        Raises:
            Exception: If async_subtensor not injected
        """
        if self._async_subtensor is None:
            raise Exception(
                "AsyncSubtensor not injected - call inject_async_subtensor() first. "
                "This should be done in neurons/validator.py run_async()."
            )
        
        nid = netuid or self._netuid
        
        # Use injected async subtensor (NO new instance!)
        return await self._async_subtensor.metagraph(netuid=nid)

    def _get_fresh_metagraph(self, network=None, netuid=None):
        """
        Get metagraph (SYNC VERSION - creates new instance).
        
        DEPRECATED: Use _get_fresh_metagraph_async() from async contexts.
        This creates a new subtensor instance - use sparingly.
        
        Args:
            network: Network name
            netuid: Subnet ID
        
        Returns:
            Metagraph object
        """
        net = network or self._network
        nid = netuid or self._netuid
        subtensor = bt.subtensor(network=net)
        return subtensor.metagraph(netuid=nid)
    
    async def is_miner_async(self, ss58: str, network=None, netuid=None) -> bool:
        """
        Check if hotkey is registered as miner (ASYNC VERSION).
        
        Use this from async contexts to avoid memory leaks.
        
        Args:
            ss58: Hotkey SS58 address
            network: Network (ignored - uses injected instance)
            netuid: Subnet ID
        
        Returns:
            True if registered
        """
        try:
            mg = await self._get_fresh_metagraph_async(network, netuid)
            return ss58 in mg.hotkeys
        except Exception as e:
            bt.logging.warning(f"Failed to verify miner registration: {e}")
            return False

    def is_miner(self, ss58: str, network=None, netuid=None) -> bool:
        """
        Check if hotkey is registered as miner (SYNC VERSION).
        
        DEPRECATED: Use is_miner_async() from async contexts.
        
        Args:
            ss58: Hotkey SS58 address
            network: Network
            netuid: Subnet ID
        
        Returns:
            True if registered
        """
        try:
            mg = self._get_fresh_metagraph(network, netuid)
            return ss58 in mg.hotkeys
        except Exception as e:
            bt.logging.warning(f"Failed to verify miner registration: {e}")
            return False
    
    async def is_validator_async(self, ss58: str, network=None, netuid=None) -> bool:
        """
        Check if hotkey is registered as validator (ASYNC VERSION).
        
        Use this from async contexts to avoid memory leaks.
        
        Args:
            ss58: Hotkey SS58 address
            network: Network (ignored - uses injected instance)
            netuid: Subnet ID
        
        Returns:
            True if has validator permit
        """
        try:
            mg = await self._get_fresh_metagraph_async(network, netuid)
            uid = mg.hotkeys.index(ss58)
            return mg.validator_permit[uid].item()
        except ValueError:
            return False
        except Exception as e:
            bt.logging.warning(f"Failed to verify validator registration: {e}")
            return False

    def is_validator(self, ss58: str, network=None, netuid=None) -> bool:
        """
        Check if hotkey is registered as validator (SYNC VERSION).
        
        DEPRECATED: Use is_validator_async() from async contexts.
        
        Args:
            ss58: Hotkey SS58 address
            network: Network
            netuid: Subnet ID
        
        Returns:
            True if has validator permit
        """
        try:
            mg = self._get_fresh_metagraph(network, netuid)
            uid = mg.hotkeys.index(ss58)
            return mg.validator_permit[uid].item()
        except ValueError:
            return False
        except Exception as e:
            bt.logging.warning(f"Failed to verify validator registration: {e}")
            return False

_VERIFY = _Verifier()                       # singleton

def _has_firestore_credentials() -> bool:
    """Check if Google Cloud Firestore credentials are available."""
    return os.path.exists("service_account_key.json") or bool(os.getenv("GOOGLE_APPLICATION_CREDENTIALS"))

# ─────────────────────────────── READ ────────────────────────────────
def get_cloud_leads(wallet: bt.wallet, limit: int = 100) -> List[Dict]:
    if not _VERIFY.is_miner(wallet.hotkey.ss58_address):
        raise PermissionError("Hotkey not registered as miner on subnet")

    r = requests.get(f"{API_URL}/leads", params={"limit": limit}, timeout=10)
    r.raise_for_status()
    return r.json()

# ─────────────────────────────── WRITE ───────────────────────────────
def save_leads_to_cloud(wallet: bt.wallet, leads: List[Dict]) -> bool:
    if not leads:
        return True

    if not _VERIFY.is_validator(wallet.hotkey.ss58_address):
        bt.logging.warning(           
            f"Hotkey {wallet.hotkey.ss58_address[:10]}… is NOT a registered "
            "validator – storing leads anyway (DEV mode)"
        )
        # continue – do NOT raise

    ts      = str(int(time.time()) // 300)
    payload = (ts + json.dumps(leads, sort_keys=True)).encode()
    sig_b64 = base64.b64encode(wallet.hotkey.sign(payload)).decode()

    body = {
        "wallet":    wallet.hotkey.ss58_address,
        "signature": sig_b64,
        "leads":     leads,
    }
    r = requests.post(f"{API_URL}/leads", json=body, timeout=30)
    r.raise_for_status()
    res = r.json()
    stored  = res.get("stored", 0)
    dupes   = res.get("duplicates", 0)
    print(f"✅ Cloud save: {stored} new / {dupes} duplicate")
    return stored > 0

# ───────────────────────────── Queued prospects ─────────────────────────────
def push_prospects_to_cloud(
    wallet: bt.wallet, 
    prospects: List[Dict],
    network: str = None,
    netuid: int = None
) -> bool:
    """
    Miners call this to enqueue prospects for validation in Supabase prospect_queue.
    
    Args:
        wallet: Miner's Bittensor wallet
        prospects: List of prospect dictionaries to push
        network: Subtensor network (e.g., "test", "finney"). If None, uses NETWORK env var.
        netuid: Subnet ID. If None, uses NETUID env var.
    """
    if not prospects:
        return True
    
    # Use provided network/netuid or fall back to environment variables
    check_network = network or NETWORK
    check_netuid = netuid or SUBNET_ID
    
    if not _VERIFY.is_miner(wallet.hotkey.ss58_address, network=check_network, netuid=check_netuid):
        raise PermissionError(
            f"Hotkey not registered as miner on subnet (network={check_network}, netuid={check_netuid})"
        )
    
    try:
        # Get Supabase client with miner's JWT token
        supabase = get_supabase_client()
        if not supabase:
            bt.logging.error("❌ Supabase client not available")
            return False
        
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # REGULATORY: Verify attestation fields are present (Task 1.2)
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        required_fields = ["wallet_ss58", "terms_version_hash", "lawful_collection"]
        skipped_prospects = []
        
        for i, prospect in enumerate(prospects):
            missing = [f for f in required_fields if f not in prospect]
            
            if missing:
                bt.logging.error(f"❌ Prospect {i+1} missing regulatory fields: {missing}")
                skipped_prospects.append(i)
        
        if skipped_prospects:
            bt.logging.error(f"❌ Skipping {len(skipped_prospects)} prospects with missing attestation fields")
            bt.logging.error("   Ensure sanitize_prospect() is adding regulatory metadata (Task 1.2)")
            # Filter out invalid prospects
            prospects = [p for i, p in enumerate(prospects) if i not in skipped_prospects]
            
            if not prospects:
                bt.logging.error("❌ No valid prospects to push after regulatory validation")
                return False
        
        # Insert each prospect into the queue
        # Extract regulatory fields as top-level columns (Task 1.2)
        records = []
        for prospect in prospects:
            record = {
                "miner_hotkey": wallet.hotkey.ss58_address,
                "prospect": prospect,
                "status": "pending",
                
                # Regulatory attestation fields (extracted to top-level for SQL queries)
                "wallet_ss58": prospect.get("wallet_ss58"),
                "terms_version_hash": prospect.get("terms_version_hash"),
                "lawful_collection": prospect.get("lawful_collection", False),
                "no_restricted_sources": prospect.get("no_restricted_sources", False),
                "license_granted": prospect.get("license_granted", False),
                
                # Source provenance fields (Task 1.3)
                "source_url": prospect.get("source_url"),
                "source_type": prospect.get("source_type"),
                "license_doc_hash": prospect.get("license_doc_hash"),
                "license_doc_url": prospect.get("license_doc_url"),
                
                # Submission timestamp
                "submission_timestamp": prospect.get("submission_timestamp"),
            }
            records.append(record)
        
        # Minimal logging
        bt.logging.debug(f"Pushing {len(records)} prospects to Supabase queue")
        
        # Batch insert (CustomResponse already executes, no .execute() needed)
        try:
            response = supabase.table("prospect_queue").insert(records)
            # Check if response indicates an error
            if hasattr(response, 'error') and response.error:
                bt.logging.error(f"❌ Supabase insert error: {response.error}")
                return False
        except Exception as insert_error:
            bt.logging.error(f"❌ Failed to insert prospects: {insert_error}")
            # Try to get more details from the error
            if hasattr(insert_error, 'response'):
                try:
                    error_detail = insert_error.response.json()
                    bt.logging.error(f"   Error details: {error_detail}")
                except:
                    bt.logging.error(f"   Raw error: {insert_error}")
            return False
        
        bt.logging.info(f"✅ Pushed {len(prospects)} prospects to Supabase queue")
        print(f"✅ Supabase queue ACK: {len(prospects)} prospect(s)")
        
        # Log submission to audit trail (Task 2.5)
        from Leadpoet.utils.audit_log import log_submission_audit
        
        audit_count = 0
        for prospect in prospects:
            if log_submission_audit(
                lead=prospect,
                wallet=wallet.hotkey.ss58_address,
                event_type="submission"
            ):
                audit_count += 1
        
        if audit_count == len(prospects):
            print(f"✅ Audit trail logged ({audit_count} submission(s))")
        
        return True
        
    except Exception as e:
        error_str = str(e)

        # ─────────────────────────────────────────────────────────────────────
        # 1. COOLDOWN ERROR (50 rejections reached - Error Code P0001)
        # ─────────────────────────────────────────────────────────────────────
        if "Rate limit exceeded" in error_str and ("50 rejected leads" in error_str or "Cooldown active" in error_str):
            bt.logging.error("🚨 RATE LIMIT: COOLDOWN ACTIVE")
            bt.logging.error("   You have reached 50 rejected leads today")
            bt.logging.error("   Your account is temporarily suspended until 12:00 AM ET")
            
            print(f"\n{'='*70}")
            print("🚨  DAILY REJECTION LIMIT REACHED")
            print(f"{'='*70}")
            print("\n⛔ Your mining account has been placed on cooldown")
            print("\nReason: 50 consensus-rejected leads in the past 24 hours")
            print("\nWhat happened:")
            print("  • All your pending leads have been removed from the queue")
            print("  • You cannot submit new leads until the daily reset")
            
            # Try to extract oldest removed lead info from error message
            try:
                if "Oldest removed lead:" in error_str:
                    # Extract email/ID from error message
                    import re
                    email_match = re.search(r'Oldest removed lead: ([^\s]+)', error_str)
                    id_match = re.search(r'\(ID: ([^\)]+)\)', error_str)
                    if email_match or id_match:
                        print("\nOldest removed lead:")
                        if email_match:
                            print(f"  • Email: {email_match.group(1)}")
                        if id_match:
                            print(f"  • Lead ID: {id_match.group(1)}")
            except Exception:
                pass
            
            print("\nNext steps:")
            print("  1. Review your lead quality and sourcing methods")
            print("  2. Check rejection feedback: query rejection_feedback table")
            print("  3. Wait until 12:00 AM ET for automatic cooldown reset")
            print("  4. Improve lead quality before resuming submissions")
            print(f"\n{'='*70}\n")
            return False
        
        # ─────────────────────────────────────────────────────────────────────
        # 2. DAILY SUBMISSION LIMIT (1000 submissions reached - Error Code P0002)
        # ─────────────────────────────────────────────────────────────────────
        elif "Rate limit exceeded" in error_str and ("Maximum" in error_str and "submissions per day" in error_str):
            # Extract current count if present
            import re
            count_match = re.search(r'Current count: (\d+)', error_str)
            current_count = count_match.group(1) if count_match else "1000"
            
            bt.logging.error("🚨 RATE LIMIT: DAILY SUBMISSION LIMIT REACHED")
            bt.logging.error(f"   You have submitted {current_count}/1000 leads today")
            bt.logging.error("   Cannot submit more until 12:00 AM ET")
            
            print(f"\n{'='*70}")
            print("🚨  DAILY SUBMISSION LIMIT REACHED")
            print(f"{'='*70}")
            print(f"\n⛔ You have reached the maximum daily submission limit")
            print(f"\nSubmissions today: {current_count}/1000")
            print("   (This includes all attempts, even duplicates)")
            print("\nNext steps:")
            print("  1. Wait until 12:00 AM ET for automatic reset")
            print("  2. You will be able to submit 1000 new leads tomorrow")
            print("  3. Consider spacing out submissions throughout the day")
            print(f"\n{'='*70}\n")
            return False
        
        # ─────────────────────────────────────────────────────────────────────
        # 3. HOTKEY MISMATCH ERROR (security validation - Error Code P0005)
        # ─────────────────────────────────────────────────────────────────────
        elif "Security violation" in error_str and "does not match" in error_str:
            bt.logging.error("🚨 SECURITY ERROR: Hotkey mismatch detected")
            bt.logging.error("   The hotkey in your submission does not match your JWT token")
            
            print(f"\n{'='*70}")
            print("🚨  AUTHENTICATION ERROR")
            print(f"{'='*70}")
            print("\n⛔ Hotkey verification failed")
            print("\nThis usually means:")
            print("  • Your local code has been modified incorrectly")
            print("  • JWT token is stale or corrupted")
            print("\nNext steps:")
            print("  1. Restart your miner to refresh JWT token")
            print("  2. Ensure you're using the official subnet code")
            print("  3. Do not manually modify hotkey fields in submissions")
            print(f"\n{'='*70}\n")
            return False
        
        # ─────────────────────────────────────────────────────────────────────
        # 4. REQUIRED FIELDS MISSING ERROR
        # ─────────────────────────────────────────────────────────────────────
        elif "Required fields missing" in error_str or "Required column not found" in error_str:
            # Extract field names if present
            import re
            fields_match = re.search(r'Required fields missing: ([^\n]+)', error_str)
            missing_fields = fields_match.group(1) if fields_match else "unknown"
            
            bt.logging.error("❌ VALIDATION ERROR: Required fields missing")
            bt.logging.error(f"   Missing fields: {missing_fields}")
            
            print(f"\n{'='*70}")
            print("❌  REQUIRED FIELDS MISSING")
            print(f"{'='*70}")
            print(f"\n⛔ Your lead is missing required fields: {missing_fields}")
            print("\nAll leads must include:")
            print("  • Email address")
            print("  • Company name")
            print("  • Company website")
            print("  • Contact name (full_name or first + last)")
            print("  • Industry")
            print("  • Sub-industry")
            print("  • Role/title")
            print("  • Location/region")
            print("  • Source type")
            print("  • Source URL")
            print("\nNext steps:")
            print("  1. Update your lead extraction to include all required fields")
            print("  2. Verify your data source provides complete information")
            print(f"\n{'='*70}\n")
            return False
        
        # ─────────────────────────────────────────────────────────────────────
        # 5. DUPLICATE LEAD ERROR (existing logic, preserved)
        # ─────────────────────────────────────────────────────────────────────
        elif "409" in error_str or "Conflict" in error_str or "Duplicate lead" in error_str or "already exists" in error_str:
            # Try to extract email from error message or from prospects
            duplicate_emails = []
            
            # First try to extract from error message
            if "Email " in error_str and " already exists" in error_str:
                try:
                    email_start = error_str.find("Email ") + 6
                    email_end = error_str.find(" already exists")
                    duplicate_email = error_str[email_start:email_end] if email_start > 6 and email_end > email_start else None
                    if duplicate_email:
                        duplicate_emails.append(duplicate_email)
                except Exception:
                    pass
            
            # If no email found in error, get from prospects
            if not duplicate_emails:
                for prospect in prospects:
                    email = get_email(prospect)
                    if email:
                        duplicate_emails.append(email)
            
            # Display clear duplicate message
            if duplicate_emails:
                bt.logging.warning(f"⚠️ Duplicate lead(s) rejected: {', '.join(duplicate_emails)}")
                print(f"\n{'='*60}")
                print("⚠️  DUPLICATE LEAD DETECTED")
                print(f"{'='*60}")
                print("The following lead(s) have already been validated:")
                for email in duplicate_emails:
                    print(f"  • {email}")
                print("\nPlease submit unique leads that haven't been validated yet.")
                print(f"{'='*60}\n")
            else:
                bt.logging.warning("⚠️ Duplicate lead rejected (409 Conflict)")
                print("\n⚠️  DUPLICATE LEAD - This lead has already been validated.")
                print("   Please submit unique leads.\n")
            return False
        
        # ─────────────────────────────────────────────────────────────────────
        # 6. RLS POLICY VIOLATIONS (preserved)
        # ─────────────────────────────────────────────────────────────────────
        elif "row-level security policy" in error_str.lower() or "policy" in error_str.lower():
            bt.logging.error("❌ Access denied: Row-level security policy violation")
            bt.logging.error("   Your JWT role may not have permission to insert prospects")
            return False
        
        # ─────────────────────────────────────────────────────────────────────
        # 7. GENERIC ERROR (fallback)
        # ─────────────────────────────────────────────────────────────────────
        else:
            bt.logging.error(f"❌ Failed to push prospects to Supabase: {e}")
            bt.logging.error(f"   Error details: {error_str}")
            return False


def fetch_prospects_from_cloud(
    wallet: bt.wallet, 
    limit: int = 100,
    network: str = None,
    netuid: int = None
) -> List[Dict]:
    """
    CONSENSUS VERSION: First-come-first-served prospect fetching.
    Validators fetch prospects they haven't pulled yet, where pull_count < 3.
    Each prospect can be pulled by up to 3 validators for consensus.
    Returns a list of prospect data with their IDs for tracking.
    
    Args:
        wallet: Validator's Bittensor wallet
        limit: Maximum number of prospects to fetch
        network: Subtensor network (e.g., "test", "finney"). If None, uses NETWORK env var.
        netuid: Subnet ID. If None, uses NETUID env var.
    """
    check_network = network or NETWORK
    check_netuid = netuid or SUBNET_ID
    
    if not _VERIFY.is_validator(wallet.hotkey.ss58_address, network=check_network, netuid=check_netuid):
        bt.logging.warning(
            f"Hotkey {wallet.hotkey.ss58_address[:10]}… is NOT a registered validator "
            f"(network={check_network}, netuid={check_netuid})"
        )
        return []
    
    try:
        # Get Supabase client with validator's JWT token
        supabase = get_supabase_client()
        if not supabase:
            bt.logging.warning("⚠️ Supabase client not available")
            return []
        
        result = supabase.rpc('pull_prospects_for_validator').execute()
        
        if not result.data:
            bt.logging.debug("No prospects available for this validator")
            return []
        
        # SQL function succeeded, format the response
        prospects_with_ids = []
        for row in result.data:
            # Include miner_hotkey in the prospect data
            prospect_data = row.get('prospect', {})
            if prospect_data and isinstance(prospect_data, dict):
                prospect_data = prospect_data.copy()
                # Add miner_hotkey if available
                if row.get('miner_hotkey'):
                    prospect_data['miner_hotkey'] = row['miner_hotkey']
            prospects_with_ids.append({
                'prospect_id': get_field(row, 'prospect_id', 'id'),
                'data': prospect_data
            })
        
        bt.logging.info(f"✅ Atomically pulled {len(prospects_with_ids)} prospects via server-side function")
        return prospects_with_ids
        
    except Exception as e:
        bt.logging.error(f"❌ Failed to fetch prospects from Supabase: {e}")
        return []

# ---- Consensus Validation Functions ---------------------------------
def submit_validation_assessment(
    wallet: bt.wallet,
    prospect_id: str,
    lead_id: str,
    lead_data: Dict,
    is_valid: bool,
    rejection_reason: Dict = None,
    network: str = None,
    netuid: int = None
) -> bool:
    """
    Submit validator's assessment to the validation tracking system.
    This is called after a validator has evaluated a lead.
    ALL validations go to validation_tracking table (both accepted and rejected).
    
    Args:
        wallet: Validator's wallet
        prospect_id: UUID of the prospect from prospect_queue
        lead_id: UUID generated for this lead
        lead_data: The full lead data dictionary
        is_valid: Boolean indicating if validator considers lead valid
        rejection_reason: Structured rejection reason dict (required if is_valid=False, None if is_valid=True)
                         Format: {"stage": ..., "check_name": ..., "message": ..., "failed_fields": [...]}
        network: Subtensor network (e.g., "test", "finney"). If None, uses NETWORK env var.
        netuid: Subnet ID. If None, uses NETUID env var.
    
    Returns:
        bool: True if submission successful, False otherwise
    """
    try:
        check_network = network or NETWORK
        check_netuid = netuid or SUBNET_ID
        
        # Verify this is a validator
        if not _VERIFY.is_validator(wallet.hotkey.ss58_address, network=check_network, netuid=check_netuid):
            bt.logging.warning(
                f"Hotkey {wallet.hotkey.ss58_address[:10]}… is NOT a registered validator "
                f"(network={check_network}, netuid={check_netuid})"
            )
            return False
        
        # Get Supabase client
        supabase = get_supabase_client()
        if not supabase:
            bt.logging.error("Supabase client not available")
            return False
        
        # Get current epoch number
        try:
            from Leadpoet.validator.reward import _calculate_epoch_number, _get_current_block
            current_block = _get_current_block()  # Function takes no arguments
            epoch_number = _calculate_epoch_number(current_block)
        except Exception as e:
            bt.logging.warning(f"Could not get epoch number: {e}, using 0")
            epoch_number = 0
        
        # Prepare validation data
        validation_data = {
            "lead_id": lead_id,
            "prospect_id": prospect_id,
            "validator_hotkey": wallet.hotkey.ss58_address,
            "is_valid": bool(is_valid),
            "rejection_reason": rejection_reason if not is_valid else None,  # Structured rejection dict
            "epoch_number": epoch_number,
            "prospect": lead_data  # Include the full lead data for database triggers
        }
        
        # Validation: Ensure rejection_reason is not null when is_valid=False
        if not is_valid and not rejection_reason:
            bt.logging.warning("⚠️ rejection_reason is None for invalid lead - this should not happen")
            # Create a fallback rejection reason to ensure data integrity
            rejection_reason = {
                "stage": "Unknown",
                "check_name": "unknown",
                "message": "No rejection reason provided",
                "failed_fields": []
            }
            validation_data["rejection_reason"] = rejection_reason
        
        # Debug: Log what we're trying to insert
        bt.logging.info("🔍 DEBUG: Attempting to insert validation data:")
        bt.logging.info(f"   - validator_hotkey: {wallet.hotkey.ss58_address}")
        bt.logging.info(f"   - prospect_id: {prospect_id}")
        bt.logging.info(f"   - lead_id: {lead_id}")
        bt.logging.info(f"   - is_valid: {validation_data['is_valid']}")
        bt.logging.info(f"   - epoch_number: {validation_data['epoch_number']}")
        
        # Debug: Check what JWT we're using
        if hasattr(supabase, 'jwt'):
            import jwt as pyjwt
            try:
                decoded = pyjwt.decode(supabase.jwt, options={"verify_signature": False})
                bt.logging.info("🔑 JWT Claims:")
                bt.logging.info(f"   - role: {decoded.get('role', 'MISSING')}")
                bt.logging.info(f"   - app_role: {decoded.get('app_role', 'MISSING')}")
                bt.logging.info(f"   - hotkey: {decoded.get('hotkey', 'MISSING')}")
                bt.logging.info(f"   - iss: {decoded.get('iss', 'MISSING')}")
                bt.logging.info(f"   - aud: {decoded.get('aud', 'MISSING')}")
            except Exception as e:
                bt.logging.error(f"Failed to decode JWT: {e}")
        
        # Debug: Log the exact request being made
        bt.logging.info(f"📤 Making INSERT request to: {supabase.postgrest_url}/validation_tracking")
        
        # Submit to validation_tracking table
        try:
            result = supabase.table("validation_tracking").insert([validation_data])
            bt.logging.info("✅ INSERT response received")
        except Exception as insert_error:
            bt.logging.error(f"❌ INSERT failed with error: {insert_error}")
            bt.logging.error(f"   Error type: {type(insert_error)}")
            if hasattr(insert_error, 'response'):
                bt.logging.error(f"   Response status: {insert_error.response.status_code}")
                bt.logging.error(f"   Response body: {insert_error.response.text}")
            raise
        
        # Check if insert was successful (status 201 or data present)
        if result.response.status_code == 201 or result.data:
            bt.logging.info(f"✅ Submitted validation for lead {lead_id[:8]}... (valid: {is_valid})")
            
            # ══════════════════════════════════════════════════════════════════════════
            # CONSENSUS IS NOW HANDLED SERVER-SIDE BY DATABASE TRIGGERS + EDGE FUNCTIONS
            # Validators no longer need to check consensus locally - it's automatic!
            # ══════════════════════════════════════════════════════════════════════════
            bt.logging.debug("Consensus will be processed server-side by database triggers")
            
            return True
        else:
            bt.logging.error(f"Failed to insert validation assessment - Status: {result.response.status_code}")
            return False
            
    except Exception as e:
        bt.logging.error(f"❌ Failed to submit validation assessment: {e}")
        import traceback
        bt.logging.debug(traceback.format_exc())
        return False


def get_rejection_feedback(
    wallet: bt.wallet,
    limit: int = 50,
    network: str = None,
    netuid: int = None
) -> List[Dict]:
    """
    Query rejection feedback for miner's rejected leads.
    
    This function allows miners to retrieve detailed feedback about leads that were
    rejected by consensus (2+ validators rejected). The feedback includes:
    - Which checks failed (stage, check name, message)
    - How many validators rejected the lead
    - A snapshot of the lead data at time of rejection
    - Consensus timestamp and epoch information
    
    Security: RLS policies ensure miners can only see their own rejection feedback.
    The query is filtered server-side by the miner's hotkey from their JWT token.
    
    Args:
        wallet: Miner's Bittensor wallet (for hotkey identification)
        limit: Maximum number of rejection records to return (default: 50)
        network: Subtensor network (e.g., "test", "finney"). If None, uses NETWORK env var.
        netuid: Subnet ID. If None, uses NETUID env var.
    
    Returns:
        List[Dict]: List of rejection feedback records, ordered by most recent first.
                    Each record contains:
                    - id: UUID of the feedback record
                    - prospect_id: UUID of the rejected prospect
                    - miner_hotkey: Miner's hotkey (always matches caller)
                    - rejection_summary: JSONB with detailed rejection reasons
                    - validator_count: Number of validators who assessed the lead
                    - consensus_timestamp: When consensus was reached
                    - epoch_number: Epoch in which rejection occurred
                    - lead_snapshot: JSONB snapshot of the lead data
                    - created_at: Timestamp when feedback was created
                    
                    Returns empty list if:
                    - No rejection feedback exists for this miner
                    - Supabase client unavailable
                    - Error occurs during query
    
    Example:
        >>> feedback = get_rejection_feedback(wallet, limit=10)
        >>> if feedback:
        ...     for record in feedback:
        ...         print(f"Lead rejected at {record['consensus_timestamp']}")
        ...         summary = record['rejection_summary']
        ...         print(f"  Rejected by {summary['rejected_by']}/{summary['total_validators']} validators")
        ...         for failure in summary['common_failures']:
        ...             print(f"  - {failure['check_name']}: {failure['message']}")
    """
    try:
        check_network = network or NETWORK
        check_netuid = netuid or SUBNET_ID
        
        # Verify this is a miner (optional - RLS will enforce this anyway)
        if not _VERIFY.is_miner(wallet.hotkey.ss58_address, network=check_network, netuid=check_netuid):
            bt.logging.warning(
                f"Hotkey {wallet.hotkey.ss58_address[:10]}… is NOT a registered miner "
                f"(network={check_network}, netuid={check_netuid})"
            )
            return []
        
        # Get Supabase client with miner's JWT token
        supabase = get_supabase_client()
        if not supabase:
            bt.logging.error("Supabase client not available - cannot fetch rejection feedback")
            return []
        
        bt.logging.debug(f"Querying rejection feedback for miner {wallet.hotkey.ss58_address[:10]}...")
        
        # Query rejection_feedback table
        # RLS policy ensures we only get feedback for this miner's hotkey
        result = supabase.table("rejection_feedback") \
            .select("*") \
            .eq("miner_hotkey", wallet.hotkey.ss58_address) \
            .order("created_at", desc=True) \
            .limit(limit) \
            .execute()
        
        # Extract data from response
        feedback_records = result.data if result.data else []
        
        if feedback_records:
            bt.logging.info(f"✅ Retrieved {len(feedback_records)} rejection feedback record(s)")
            
            # Log summary for first record (most recent)
            if len(feedback_records) > 0:
                latest = feedback_records[0]
                summary = latest.get('rejection_summary', {})
                rejected_by = summary.get('rejected_by', 0)
                total_validators = summary.get('total_validators', 0)
                bt.logging.debug(
                    f"   Most recent: {rejected_by}/{total_validators} validators rejected "
                    f"(epoch {latest.get('epoch_number', 'unknown')})"
                )
        else:
            bt.logging.debug("No rejection feedback found for this miner")
        
        return feedback_records
        
    except Exception as e:
        bt.logging.error(f"❌ Failed to fetch rejection feedback: {e}")
        import traceback
        bt.logging.debug(traceback.format_exc())
        return []


# ---- Curations -------------------------------------------------------
def push_curation_request(payload: dict) -> str:
    try:
        r = requests.post(f"{API_URL}/curate", json=payload, timeout=10)
        r.raise_for_status()
        return r.json()["request_id"]
    except requests.exceptions.RequestException as e:
        bt.logging.error(f"push_curation_request failed: {e}")
        raise  # Re-raise to be handled by caller

def fetch_curation_requests() -> dict:
    try:
        r = requests.post(f"{API_URL}/curate/fetch", timeout=10)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.RequestException as e:
        bt.logging.warning(f"fetch_curation_requests failed: {e}")
        return None  # Return None instead of raising

def push_curation_result(result: dict):
    try:
        requests.post(f"{API_URL}/curate/result", json=result, timeout=30).raise_for_status()
    except requests.exceptions.RequestException as e:
        bt.logging.error(f"push_curation_result failed: {e}")
        # Don't raise - just log and continue

def fetch_curation_result(request_id: str) -> dict:
    try:
        r = requests.get(f"{API_URL}/curate/result/{request_id}", timeout=30)
        r.raise_for_status()
    except requests.exceptions.Timeout:
        return None        # let the caller loop again
    return r.json()

def _signed_body(wallet: bt.wallet, extra: dict) -> dict:
    payload  = generate_timestamp(json.dumps(extra, sort_keys=True))
    sig_b64  = base64.b64encode(wallet.hotkey.sign(payload)).decode()
    return {"wallet": wallet.hotkey.ss58_address,
            "signature": sig_b64, **extra}

# ───────── validator → miner ------------------------------------------------
def push_miner_curation_request(wallet: bt.wallet, payload: dict) -> str:
    body = _signed_body(wallet, payload)
    r    = requests.post(f"{API_URL}/curate/miner_request", json=body, timeout=10)
    r.raise_for_status()
    return r.json()["miner_request_id"]

def fetch_miner_curation_request(wallet: bt.wallet) -> dict:
    body = _signed_body(wallet, {})
    r    = requests.post(f"{API_URL}/curate/miner_request/fetch", json=body, timeout=10)
    r.raise_for_status()
    return r.json()

# ───────── miner → validator -----------------------------------------------
def push_miner_curation_result(wallet: bt.wallet, result: dict):
    body = _signed_body(wallet, result)
    requests.post(f"{API_URL}/curate/miner_result", json=body, timeout=30).raise_for_status()

def fetch_miner_curation_result(wallet: bt.wallet) -> dict:
    body = _signed_body(wallet, {})
    r    = requests.post(f"{API_URL}/curate/miner_result/fetch", json=body, timeout=10)
    r.raise_for_status()
    return r.json()

def push_validator_weights(wallet: bt.wallet, uid: int, weights: dict):
    body   = _signed_body(wallet, {"uid": uid, "weights": weights})
    r      = requests.post(f"{API_URL}/validator_weights", json=body, timeout=10)
    r.raise_for_status()
    print("📝 Stored weights in Firestore via Cloud-Run")

# ──────────────────── BROADCAST API REQUESTS ─────────────────────────────

def broadcast_api_request(wallet: bt.wallet, num_leads: int, business_desc: str, client_id: str = None) -> str:
    """
    Broadcast an API request to Supabase for ALL validators and miners to process.

    Args:
        wallet: Client's wallet
        num_leads: Number of leads requested
        business_desc: Business description
        client_id: Optional client identifier

    Returns:
        str: request_id if successful, None otherwise
    """
    try:
        from datetime import datetime
        import uuid

        supabase = get_supabase_client()
        if not supabase:
            bt.logging.error("Supabase client not available")
            return None

        # Generate unique request ID
        request_id = str(uuid.uuid4())

        # Insert to Supabase api_requests table
        data = {
            "request_id": request_id,
            "client_hotkey": wallet.hotkey.ss58_address,
            "client_id": client_id or "unknown",
            "num_leads": num_leads,
            "business_desc": business_desc,
            "status": "pending",
            "created_at": datetime.utcnow().isoformat() + "Z",
            "updated_at": datetime.utcnow().isoformat() + "Z",
        }
        
        supabase.table("api_requests").insert(data)

        bt.logging.info(f"📡 Broadcast API request {request_id[:8]}... to Supabase")
        return request_id

    except Exception as e:
        bt.logging.error(f"Failed to broadcast API request: {e}")
        return None


def fetch_broadcast_requests(wallet: bt.wallet, role: str = "validator") -> List[Dict]:
    """
    Fetch pending broadcast API requests from Supabase.
    Returns list of pending requests that need processing.
    
    NOTE: This function is deprecated. Validators should use TEE gateway
    instead for lead fetching (tasks6.md). Returning empty list.

    Args:
        wallet: Bittensor wallet
        role: "validator" or "miner" - determines which requests to fetch
    """
    # Broadcast API feature deprecated - validators use TEE gateway now
    # Return empty list to prevent JWT authentication errors
    bt.logging.debug(f"fetch_broadcast_requests() called but deprecated - returning empty list")
    return []
    
    # DEPRECATED CODE BELOW (kept for reference):
    # try:
    #     supabase = get_supabase_client()
    #     if not supabase:
    #         return []
    #
    #     # Fetch pending requests
    #     result = supabase.table("api_requests") \
    #         .select("*") \
    #         .eq("status", "pending") \
    #         .order("created_at", desc=False) \
    #         .limit(10) \
    #         .execute()
    #
    #     requests_list = result.data if result.data else []
    #
    #     # Only log when requests are found
    #     if requests_list:
    #         bt.logging.info(f"🔔 [{role.upper()}] Found {len(requests_list)} NEW broadcast request(s)!")
    #
    #     return requests_list
    #
    # except Exception as e:
    #     bt.logging.error(f"fetch_broadcast_requests ({role}) failed: {e}")
    #     return []


def mark_broadcast_processing(wallet: bt.wallet, request_id: str) -> bool:
    """
    Mark a broadcast request as being processed to prevent duplicates.
    Uses conditional UPDATE for atomic operation.
    Only ONE miner will successfully mark it.
    """
    try:
        supabase = get_supabase_client()
        if not supabase:
            return False

        # Try to update ONLY if status is still "pending" (atomic)
        # This prevents race conditions - only one miner succeeds
        result = supabase.table("api_requests") \
            .eq("request_id", request_id) \
            .eq("status", "pending") \
            .update({
                "status": "processing",
                "processing_by": wallet.hotkey.ss58_address,
                "updated_at": datetime.now(timezone.utc).isoformat()
            })

        # If result.data is empty, another miner already claimed it
        success = result.data and len(result.data) > 0

        if success:
            bt.logging.info(f"✅ Marked request {request_id[:8]}... as processing")
        else:
            bt.logging.debug(f"Request {request_id[:8]}... already being processed")

        return success

    except Exception as e:
        bt.logging.error(f"Failed to mark request as processing: {e}")
        return False


def get_broadcast_status(request_id: str) -> Dict:
    """
    Get the status of a broadcast API request from Supabase.
    Used by validators and clients to check request status.
    """
    try:
        supabase = get_supabase_client()
        if not supabase:
            return {"status": "error", "leads": [], "error": "Supabase client not available"}

        # Fetch request by ID
        result = supabase.table("api_requests") \
            .select("*") \
            .eq("request_id", request_id) \
            .execute()

        if not result.data or len(result.data) == 0:
            return {"status": "not_found", "leads": [], "request_id": request_id}

        return result.data[0]

    except Exception as e:
        bt.logging.error(f"Failed to get status for request {request_id[:8]}...: {e}")
        return {"status": "error", "leads": [], "error": str(e)}

def push_validator_ranking(wallet: bt.wallet, request_id: str, ranked_leads: List[Dict], validator_trust: float) -> bool:
    """
    Submit validator's ranking for a broadcast API request to Supabase.

    Args:
        wallet: Validator's wallet
        request_id: Broadcast request ID
        ranked_leads: List of leads with scores and ranks
        validator_trust: Validator's trust value from metagraph

    Returns:
        bool: Success status
    """
    # Get validator UID from metagraph
    try:
        mg = _VERIFY._get_fresh_metagraph()
        validator_uid = mg.hotkeys.index(wallet.hotkey.ss58_address)
    except ValueError:
        validator_uid = -1  # Unknown UID

    try:
        supabase = get_supabase_client()
        if not supabase:
            bt.logging.warning("Supabase client not available, cannot push ranking")
            return False

        # Insert/upsert ranking to Supabase
        data = {
            "request_id": request_id,
            "validator_hotkey": wallet.hotkey.ss58_address,
            "validator_uid": validator_uid,
            "validator_trust": validator_trust,
            "ranked_leads": ranked_leads,
            "num_leads_ranked": len(ranked_leads),
            "submitted_at": datetime.now(timezone.utc).isoformat(),
        }
        
        supabase.table("validator_rankings").insert(data)

        bt.logging.info(f"📊 Submitted ranking for request {request_id[:8]}... ({len(ranked_leads)} leads)")
        return True

    except Exception as e:
        bt.logging.error(f"Failed to submit validator ranking: {e}")
        return False


def fetch_validator_rankings(request_id: str, timeout_sec: int = 5) -> List[Dict]:
    """
    Fetch all validator rankings for a broadcast request from Supabase.

    Args:
        request_id: Broadcast request ID
        timeout_sec: Not used (kept for API compatibility)

    Returns:
        List of validator ranking submissions
    """
    try:
        supabase = get_supabase_client()
        if not supabase:
            return []

        # Query all validator rankings for this request
        result = supabase.table("validator_rankings") \
            .select("*") \
            .eq("request_id", request_id) \
            .execute()

        rankings = result.data if result.data else []

        if rankings:
            bt.logging.debug(f"📊 Fetched {len(rankings)} validator ranking(s) for request {request_id[:8]}...")

        return rankings

    except Exception as e:
        bt.logging.debug(f"Failed to fetch validator rankings: {e}")
        return []


def mark_consensus_complete(request_id: str, final_leads: List[Dict]) -> bool:
    """
    Mark a broadcast request as complete with final consensus leads.

    Args:
        request_id: Broadcast request ID
        final_leads: Final ranked leads after consensus

    Returns:
        bool: Success status
    """
    body = {
        "request_id": request_id,
        "status": "completed",
        "leads": final_leads,
        "completed_at": time.time(),
    }

    try:
        r = requests.post(f"{API_URL}/api_requests/complete", json=body, timeout=10)
        r.raise_for_status()
        bt.logging.info(f"✅ Marked request {request_id[:8]}... as completed with {len(final_leads)} leads")
        return True
    except requests.exceptions.RequestException as e:
        bt.logging.error(f"Failed to mark consensus complete: {e}")
        return False

def log_consensus_metrics(
    request_id: str,
    num_validators_participated: int,
    num_validators_expected: int,
    trust_distribution: Dict[str, float],
    total_trust: float,
    average_response_time: float,
    top_leads_summary: List[Dict],
    calculation_time: float
) -> bool:
    """
    Log consensus metrics to Firestore for monitoring and analytics.

    Args:
        request_id: The broadcast request ID
        num_validators_participated: Number of validators who submitted rankings
        num_validators_expected: Total active validators at time of request
        trust_distribution: Dict mapping validator_hotkey -> trust value
        total_trust: Sum of all participating validator trust values
        average_response_time: Average time for validators to respond (seconds)
        top_leads_summary: List of top lead summaries with scores
        calculation_time: Time taken to calculate consensus (seconds)

    Returns:
        bool: Success status
    """
    from datetime import datetime

    body = {
        "request_id": request_id,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "metrics": {
            "validators": {
                "participated": num_validators_participated,
                "expected": num_validators_expected,
                "participation_rate": round(num_validators_participated / max(num_validators_expected, 1), 3),
            },
            "trust": {
                "total": round(total_trust, 4),
                "average": round(total_trust / max(num_validators_participated, 1), 4),
                "distribution": {hk[:10]: round(t, 4) for hk, t in trust_distribution.items()},
            },
            "timing": {
                "average_response_time_sec": round(average_response_time, 2),
                "consensus_calculation_sec": round(calculation_time, 3),
            },
            "leads": {
                "total_selected": len(top_leads_summary),
                "top_leads": top_leads_summary,
            }
        }
    }

    try:
        r = requests.post(f"{API_URL}/consensus_metrics/log", json=body, timeout=10)
        r.raise_for_status()
        bt.logging.info(f"📊 Logged consensus metrics for request {request_id[:8]}...")
        return True
    except requests.exceptions.RequestException as e:
        bt.logging.warning(f"Failed to log consensus metrics (non-critical): {e}")
        return False

def push_miner_curated_leads(wallet: bt.wallet, request_id: str, leads: List[Dict]) -> bool:
    """
    Push miner's curated leads to Supabase for validators to pick up.

    Args:
        wallet: Miner's wallet
        request_id: Broadcast request ID
        leads: Curated leads from miner

    Returns:
        bool: Success status
    """
    try:
        supabase = get_supabase_client()
        if not supabase:
            bt.logging.warning("Supabase client not available, cannot push miner leads")
            return False

        # Insert to Supabase
        data = {
            "request_id": request_id,
            "miner_hotkey": wallet.hotkey.ss58_address,
            "leads": leads,
            "num_leads": len(leads),
            "submitted_at": datetime.now(timezone.utc).isoformat(),
        }

        supabase.table("miner_submissions").insert(data)

        bt.logging.info(f"📤 Pushed {len(leads)} curated lead(s) to Supabase for request {request_id[:8]}...")
        return True

    except Exception as e:
        bt.logging.error(f"Failed to push miner leads: {e}")
        return False


def fetch_miner_leads_for_request(request_id: str) -> List[Dict]:
    """
    Fetch all miner submissions for a broadcast request from Supabase.

    Args:
        request_id: Broadcast request ID

    Returns:
        List of miner submission dicts
    """
    try:
        supabase = get_supabase_client()
        if not supabase:
            return []

        # Query all miner submissions for this request
        result = supabase.table("miner_submissions") \
            .select("*") \
            .eq("request_id", request_id) \
            .execute()

        return result.data if result.data else []

    except Exception as e:
        bt.logging.debug(f"Failed to fetch miner leads: {e}")
        return []


# ───────────────────────────── Metagraph Sync ─────────────────────────────
# This function should not be used by validators - metagraph sync should be done server-side
# Validators do NOT need service role keys
def sync_metagraph_to_supabase(metagraph, netuid: int) -> bool:
    """
    Sync the current metagraph to Supabase for JWT verification.
    This allows the Edge Function to verify hotkeys without direct RPC access.
    
    CRITICAL: Uses service role key (not JWT) to avoid chicken-and-egg problem.
    Should be called by validators BEFORE requesting JWT.
    """
    try:
        from supabase import create_client
        
        # CRITICAL: Use service role key directly (not JWT)
        # This must work BEFORE the validator has a JWT token
        service_role_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
        if not service_role_key:
            bt.logging.error("❌ SUPABASE_SERVICE_ROLE_KEY not found in environment")
            return False
        
        supabase = create_client(SUPABASE_URL, service_role_key)
        
        # Prepare records for all neurons in the metagraph
        records = []
        for uid in range(len(metagraph.hotkeys)):
            records.append({
                'netuid': netuid,
                'uid': uid,
                'hotkey': metagraph.hotkeys[uid],
                'validator_permit': bool(metagraph.validator_permit[uid].item()),
                'active': bool(metagraph.active[uid].item()),  # CRITICAL: Check if actively validating
            })
        
        bt.logging.info(f"📊 Syncing {len(records)} neurons to metagraph cache...")
        
        # Upsert all records (insert or update if exists)
        # Note: This uses service_role client (real supabase-py), so .execute() IS needed
        for record in records:
            supabase.table("metagraph_cache").upsert(record, on_conflict='netuid,hotkey').execute()
        
        bt.logging.info(f"✅ Synced {len(records)} neurons to metagraph cache")
        return True
        
    except Exception as e:
        bt.logging.error(f"❌ Failed to sync metagraph to Supabase: {e}")
        import traceback
        bt.logging.error(traceback.format_exc())
        return False


# ═══════════════════════════════════════════════════════════════════
#  GATEWAY INTEGRATION (Passages 1 & 2 Workflow)
# ═══════════════════════════════════════════════════════════════════

def check_email_duplicate(email: str) -> bool:
    """
    Check if an email has already been submitted by querying the public transparency_log.
    
    This allows miners to detect duplicates BEFORE wasting time on presign/upload.
    The transparency_log is PUBLIC (read-only via SUPABASE_ANON_KEY).
    
    Args:
        email: Email address to check (will be normalized: lowercase, trimmed)
        
    Returns:
        True if email is a duplicate (already in transparency_log)
        False if email is unique (safe to submit)
    """
    try:
        import hashlib
        from supabase import create_client
        
        # Normalize email (same as gateway does)
        normalized_email = email.strip().lower()
        
        # Compute email hash (same as gateway does)
        email_hash = hashlib.sha256(normalized_email.encode()).hexdigest()
        
        # Use ANON key for public read-only access to transparency_log
        supabase = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
        
        # Query transparency_log for this email_hash
        # CRITICAL: Only check SUBMISSION events (successful submissions)
        # NOT SUBMISSION_REQUEST events (which exist even if validation failed)
        result = supabase.table("transparency_log") \
            .select("id, actor_hotkey, ts, event_type") \
            .eq("email_hash", email_hash) \
            .eq("event_type", "SUBMISSION") \
            .limit(1) \
            .execute()
        
        if result.data and len(result.data) > 0:
            # Duplicate found!
            existing = result.data[0]
            bt.logging.warning(f"⚠️  DUPLICATE DETECTED: Email already submitted")
            bt.logging.warning(f"   Original submission: {existing.get('ts', 'unknown time')}")
            bt.logging.warning(f"   Original miner: {existing.get('actor_hotkey', 'unknown')[:20]}...")
            return True
        
        # No duplicate found
        print(f"✅ No duplicate found - lead is unique")
        return False
        
    except Exception as e:
        bt.logging.warning(f"Failed to check duplicate (assuming not duplicate): {e}")
        # If check fails, assume not duplicate (don't block submission)
        return False


def gateway_get_presigned_url(wallet: bt.wallet, lead_data: Dict) -> Dict:
    """
    Get presigned URL from gateway for S3/MinIO upload.
    
    Retries up to 3 times with fresh nonce/signature on each attempt.
    
    Args:
        wallet: Miner's wallet
        lead_data: Lead data (used to generate lead_id)
        
    Returns:
        Dict with: lead_id, presigned_url, storage_backend
    """
    import hashlib
    import uuid
    
    # Compute lead_id and hashes ONCE (reused across retries)
    lead_id = str(uuid.uuid4())
    lead_blob = json.dumps(lead_data, sort_keys=True)
    lead_blob_hash = hashlib.sha256(lead_blob.encode()).hexdigest()
    email = lead_data.get("email", "").strip().lower()
    email_hash = hashlib.sha256(email.encode()).hexdigest()
    
    print(f"📤 Sending lead_blob_hash: {lead_blob_hash[:16]}...")
    print(f"📤 Sending email_hash: {email_hash[:16]}... (for duplicate detection)")
    
    # Create payload (reused across retries)
    payload = {
        "lead_id": lead_id,
        "lead_blob_hash": lead_blob_hash,
        "email_hash": email_hash
    }
    payload_json = json.dumps(payload, sort_keys=True, separators=(',', ':'))
    payload_hash = hashlib.sha256(payload_json.encode()).hexdigest()
    build_id = os.getenv("BUILD_ID", "miner-client")
    
    # Retry loop: Up to 3 attempts
    for attempt in range(1, 4):
        try:
            # Generate FRESH nonce and timestamp for each attempt
            nonce = str(uuid.uuid4())
            ts = datetime.now(timezone.utc).isoformat()
            
            # Construct message to sign with FRESH nonce/timestamp
            message = f"SUBMISSION_REQUEST:{wallet.hotkey.ss58_address}:{nonce}:{ts}:{payload_hash}:{build_id}"
            
            if attempt == 1:
                print(f"🔐 Signing message to prove wallet ownership...")
            else:
                print(f"🔐 Retry {attempt}/3: Signing with fresh nonce/timestamp...")
            
            # Sign the message
            signature = wallet.hotkey.sign(message.encode()).hex()
            
            # Create full event object
            event = {
                "event_type": "SUBMISSION_REQUEST",
                "actor_hotkey": wallet.hotkey.ss58_address,
                "nonce": nonce,
                "ts": ts,
                "payload_hash": payload_hash,
                "build_id": build_id,
                "signature": signature,
                "payload": payload
            }
            
            # Request presigned URL
            response = requests.post(
                f"{GATEWAY_URL}/presign",
                json=event,
                timeout=300  # 5 minutes timeout (allows for international network latency and gateway processing)
            )
            response.raise_for_status()
            
            result = response.json()
            if attempt > 1:
                print(f"✅ Retry {attempt}/3 succeeded!")
            print(f"✅ Received presigned URLs for lead {result['lead_id'][:8]}...")
            return result
            
        except requests.HTTPError as e:
            # HTTP error (4xx, 5xx) - try to parse detailed message from gateway
            error_msg = str(e)
            if e.response is not None:
                try:
                    # Gateway returns detailed error in response body
                    response_data = e.response.json()
                    if isinstance(response_data, dict) and "detail" in response_data:
                        error_detail = response_data["detail"]
                        # Handle both string and dict detail formats
                        if isinstance(error_detail, dict):
                            error_msg = error_detail.get("message", str(e))
                        else:
                            error_msg = error_detail
                except:
                    # If parsing fails, fall back to generic error
                    pass
            
            if attempt < 3:
                bt.logging.warning(f"⚠️  Attempt {attempt}/3 failed: {error_msg}")
                bt.logging.warning(f"   Retrying with fresh nonce/signature...")
                continue  # Try again
            else:
                # All attempts exhausted
                bt.logging.error(f"❌ All 3 attempts failed. Last error: {error_msg}")
                return None
            
        except Exception as e:
            # Non-HTTP errors (network timeout, connection error, etc.)
            if attempt < 3:
                bt.logging.warning(f"⚠️  Attempt {attempt}/3 failed: {e}")
                bt.logging.warning(f"   Retrying with fresh nonce/signature...")
                continue  # Try again
            else:
                # All attempts exhausted
                bt.logging.error(f"❌ All 3 attempts failed. Last error: {e}")
                return None


def gateway_upload_lead(presigned_url: str, lead_data: Dict) -> bool:
    """
    Upload lead blob to S3/MinIO using presigned URL.
    
    Args:
        presigned_url: Presigned URL from gateway
        lead_data: Complete lead data
        
    Returns:
        bool: Success status
    """
    try:
        # Determine storage backend from URL
        backend = "S3" if "s3.amazonaws.com" in presigned_url else "MinIO"
        
        print(f"📤 Uploading lead blob to {backend}...")
        
        # Upload lead JSON to storage (MUST use sort_keys=True to match hash computation)
        response = requests.put(
            presigned_url,
            data=json.dumps(lead_data, sort_keys=True),
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        response.raise_for_status()
        
        print(f"✅ Lead blob uploaded successfully to {backend}")
        return True
        
    except Exception as e:
        bt.logging.error(f"Failed to upload lead: {e}")
        return False


def gateway_verify_submission(wallet: bt.wallet, lead_id: str) -> Dict:
    """
    Trigger gateway verification of uploaded lead (BRD Section 4.1, Step 5-6).
    
    Called after miner uploads lead to both S3 and MinIO via presigned URLs.
    Gateway will:
    1. Fetch uploaded blobs from both mirrors
    2. Verify SHA256 hashes match committed lead_blob_hash
    3. Log STORAGE_PROOF events (one per mirror)
    4. Store lead in leads_private table
    5. Log SUBMISSION event
    
    This prevents blob substitution attacks.
    
    Args:
        wallet: Miner's wallet
        lead_id: UUID of the lead
        
    Returns:
        Dict with: {status, lead_id, storage_backends, merkle_proof, submission_ts}
        None if verification failed
    """
    try:
        import uuid
        import hashlib
        
        print(f"🔐 Requesting gateway to verify uploaded lead...")
        
        # Generate UUID v4 nonce
        nonce = str(uuid.uuid4())
        
        # Create ISO timestamp
        ts = datetime.now(timezone.utc).isoformat()
        
        # Create payload
        payload = {
            "lead_id": lead_id
        }
        
        # Compute payload hash (deterministic JSON)
        payload_json = json.dumps(payload, sort_keys=True, separators=(',', ':'))
        payload_hash = hashlib.sha256(payload_json.encode()).hexdigest()
        
        # Build ID
        build_id = os.getenv("BUILD_ID", "miner-client")
        
        # Construct message to sign (format: {event_type}:{actor_hotkey}:{nonce}:{ts}:{payload_hash}:{build_id})
        message = f"SUBMIT_LEAD:{wallet.hotkey.ss58_address}:{nonce}:{ts}:{payload_hash}:{build_id}"
        
        # Sign the message
        signature = wallet.hotkey.sign(message.encode()).hex()
        
        # Create full event object
        event = {
            "event_type": "SUBMIT_LEAD",
            "actor_hotkey": wallet.hotkey.ss58_address,
            "nonce": nonce,
            "ts": ts,
            "payload_hash": payload_hash,
            "build_id": build_id,
            "signature": signature,
            "payload": payload
        }
        
        # Request verification
        response = requests.post(
            f"{GATEWAY_URL}/submit",
            json=event,
            timeout=300  # 5 minutes timeout (allows for international network latency + gateway verification steps: S3, MinIO, DB, TEE)
        )
        response.raise_for_status()
        
        result = response.json()
        print(f"✅ Gateway verified lead: {result['lead_id'][:8]}...")
        print(f"   Storage backends: {result['storage_backends']}")
        print(f"   Submission time: {result['submission_timestamp']}")
        
        # Display rate limit stats
        if "rate_limit_stats" in result:
            stats = result["rate_limit_stats"]
            print(f"   📊 Rate limits: {stats['submissions']}/{stats['max_submissions']} submissions, {stats['rejections']}/{stats['max_rejections']} rejections")
        
        return result
        
    except requests.HTTPError as e:
        bt.logging.error(f"Failed to verify submission: {e}")
        
        # Try to extract detailed error info from response
        try:
            error_details = e.response.json()
            if isinstance(error_details, dict):
                if "detail" in error_details and isinstance(error_details["detail"], dict):
                    detail = error_details["detail"]
                    error_msg = detail.get("error", "unknown_error")
                    message = detail.get("message", str(e))
                    
                    # Check if it's a rate limit error (HTTP 429)
                    if e.response.status_code == 429:
                        print(f"\n{'='*70}")
                        print(f"🚫 RATE LIMIT EXCEEDED")
                        print(f"{'='*70}")
                        print(f"{message}")
                        
                        if "stats" in detail:
                            stats = detail["stats"]
                            limit_type = stats.get("limit_type", "unknown")
                            if limit_type == "submissions":
                                print(f"\n📊 Daily submission limit: {stats.get('submissions', 'N/A')}/{stats.get('max_submissions', '?')} reached")
                            elif limit_type == "rejections":
                                print(f"\n📊 Daily rejection limit: {stats.get('rejections', 'N/A')}/{stats.get('max_rejections', '?')} reached")
                            print(f"🕐 Resets at: {stats.get('reset_at', 'unknown')}")
                        print(f"{'='*70}\n")
                    else:
                        print(f"\n{'='*70}")
                        print(f"❌ GATEWAY REJECTION: {error_msg}")
                        print(f"{'='*70}")
                        print(f"Reason: {message}")
                        
                        # Show missing fields if present
                        if "missing_fields" in detail:
                            print(f"\n⚠️  Missing required fields ({len(detail['missing_fields'])}):")
                            for field in detail['missing_fields']:
                                print(f"   • {field}")
                            
                            if "required_fields" in detail:
                                print(f"\n📋 All required fields:")
                                for field in detail['required_fields']:
                                    print(f"   • {field}")
                        
                        # Show rate limit stats for ALL errors (success or failure)
                        if "rate_limit_stats" in detail:
                            stats = detail["rate_limit_stats"]
                            print(f"\n📊 Rate limits: {stats['submissions']}/{stats['max_submissions']} submissions, {stats['rejections']}/{stats['max_rejections']} rejections")
                        
                        print(f"{'='*70}\n")
                else:
                    print(f"❌ Gateway error: {error_details}")
            else:
                print(f"❌ Gateway error: {error_details}")
        except Exception:
            # If we can't parse error details, just show the exception
            pass
        
        return None
        
    except Exception as e:
        bt.logging.error(f"Failed to verify submission: {e}")
        return None


def gateway_get_epoch_leads(wallet: bt.wallet, epoch_id: int) -> List[Dict]:
    """
    Get assigned leads for current epoch (validator only).
    
    Args:
        wallet: Validator's wallet
        epoch_id: Current epoch ID
        
    Returns:
        List of 50 lead dicts with full data
    """
    try:
        # Generate signature for authentication
        # Gateway expects message format: "GET_EPOCH_LEADS:{epoch_id}:{validator_hotkey}"
        message = f"GET_EPOCH_LEADS:{epoch_id}:{wallet.hotkey.ss58_address}"
        signature = wallet.hotkey.sign(message.encode()).hex()
        
        # Request epoch leads
        response = requests.get(
            f"{GATEWAY_URL}/epoch/{epoch_id}/leads",
            params={
                "validator_hotkey": wallet.hotkey.ss58_address,
                "signature": signature
            },
            timeout=180  # Increased to 180s (3 minutes) - gateway may need time to query Supabase and build lead_blob data
        )
        response.raise_for_status()
        
        result = response.json()
        leads = result.get("leads", [])
        
        # Check if gateway returned a message (e.g., "already submitted")
        message = result.get("message", "")
        
        if not leads and message:
            # Gateway explicitly said why there are no leads
            bt.logging.info(f"ℹ️  Gateway: {message}")
            # Return special marker: None means "already processed, don't retry"
            return None
        
        bt.logging.info(f"✅ Fetched {len(leads)} leads for epoch {epoch_id}")
        return leads
        
    except requests.exceptions.Timeout as e:
        # Timeout is common during epoch transitions (gateway processing epoch lifecycle)
        # This is NOT a fatal error - validator will retry automatically
        bt.logging.warning(f"⏳ Gateway timeout fetching leads for epoch {epoch_id} - this is normal during epoch transitions. Validator will retry automatically.")
        return []  # Return empty list (not None) to indicate "retry later"
    except Exception as e:
        bt.logging.error(f"Failed to get epoch leads: {e}")
        return []  # Return empty list (not None) to indicate "retry later"


def gateway_submit_validation(wallet: bt.wallet, epoch_id: int, validation_results: List[Dict]) -> bool:
    """
    Submit hashed validation results for all leads in an epoch.
    
    Efficient submission:
    - 1 HTTP request (not N individual requests)
    - 1 signature verification on gateway
    - Atomic operation (all succeed or all fail)
    - Works dynamically with any MAX_LEADS_PER_EPOCH (10, 20, 50, etc.)
    - Retries up to 3 times with fresh nonce/timestamp on gateway timeout
    
    Args:
        wallet: Validator's wallet
        epoch_id: Current epoch ID
        validation_results: List of dicts with lead_id, decision_hash, rep_score_hash, rejection_reason_hash, evidence_hash, evidence_blob
        
    Returns:
        bool: Success status
    """
    import uuid
    import hashlib
    
    # Build ID (constant for all attempts)
    build_id = os.getenv("BUILD_ID", "validator-client")
    
    # Format validations for batch submission (constant for all attempts)
    validations = []
    for v in validation_results:
        validations.append({
            "lead_id": v["lead_id"],
            "decision_hash": v["decision_hash"],
            "rep_score_hash": v["rep_score_hash"],
            "rejection_reason_hash": v["rejection_reason_hash"],
            "evidence_hash": v["evidence_hash"],
            "evidence_blob": v.get("evidence_blob", {})
        })
    
    # Create payload (constant for all attempts)
    payload = {
        "epoch_id": epoch_id,
        "validations": validations
    }
    
    # Compute payload hash (deterministic JSON, constant for all attempts)
    payload_json = json.dumps(payload, sort_keys=True, separators=(',', ':'))
    payload_hash = hashlib.sha256(payload_json.encode()).hexdigest()
    
    # Retry loop: Up to 3 attempts with fresh nonce/timestamp
    for attempt in range(1, 4):
        try:
            # Generate FRESH nonce and timestamp for each attempt
            nonce = str(uuid.uuid4())
            ts = datetime.now(timezone.utc).isoformat()
            
            # Construct message to sign with FRESH nonce/timestamp
            message = f"VALIDATION_RESULT_BATCH:{wallet.hotkey.ss58_address}:{nonce}:{ts}:{payload_hash}:{build_id}"
            
            if attempt == 1:
                bt.logging.info(f"📤 Submitting {len(validations)} hashed validations to gateway...")
            else:
                bt.logging.warning(f"🔄 Retry {attempt}/3: Submitting with fresh nonce/timestamp...")
            
            # Sign the message
            signature = wallet.hotkey.sign(message.encode()).hex()
            
            # Create full event object
            event = {
                "event_type": "VALIDATION_RESULT_BATCH",
                "actor_hotkey": wallet.hotkey.ss58_address,
                "nonce": nonce,
                "ts": ts,
                "payload_hash": payload_hash,
                "build_id": build_id,
                "signature": signature,
                "payload": payload
            }
            
            # Submit validation (single request for all leads)
            response = requests.post(
                f"{GATEWAY_URL}/validate",
                json=event,
                timeout=600  # 10 minutes timeout (gateway needs time for validation evidence storage + consensus + database operations)
            )
            response.raise_for_status()
            
            result = response.json()
            if attempt > 1:
                bt.logging.info(f"✅ Retry {attempt}/3 succeeded!")
            bt.logging.info(f"✅ Validation submitted successfully: {result.get('validation_count', len(validations))} validations")
            return True
            
        except Exception as e:
            if attempt < 3:
                bt.logging.warning(f"⚠️  Attempt {attempt}/3 failed: {e}")
                if hasattr(e, 'response') and e.response is not None:
                    try:
                        error_detail = e.response.json()
                        bt.logging.warning(f"   Error details: {error_detail}")
                    except:
                        pass
                bt.logging.warning(f"   Retrying with fresh nonce/signature...")
                time.sleep(2)  # Brief delay before retry
                continue  # Try again
            else:
                # All attempts exhausted
                bt.logging.error(f"❌ All 3 attempts failed. Last error: {e}")
                if hasattr(e, 'response') and e.response is not None:
                    try:
                        error_detail = e.response.json()
                        bt.logging.error(f"   Error details: {error_detail}")
                    except:
                        bt.logging.error(f"   Response text: {e.response.text}")
                return False


def gateway_submit_reveal(wallet: bt.wallet, epoch_id: int, reveal_results: List[Dict]) -> bool:
    """
    Submit revealed validation results after epoch closes (POST /reveal).
    Retries up to 3 times with fresh nonce/signature on gateway timeout.
    
    Args:
        wallet: Validator's wallet
        epoch_id: Epoch ID to reveal
        reveal_results: List of dicts with lead_id, decision, rep_score, rejection_reason, salt
        
    Returns:
        bool: Success status
    """
    # Retry loop: Up to 3 attempts with fresh nonce/signature
    for attempt in range(1, 4):
        try:
            # Generate FRESH nonce and signature for each attempt
            nonce = str(int(time.time() * 1000))  # Millisecond precision to ensure uniqueness
            message = f"reveal:{wallet.hotkey.ss58_address}:{epoch_id}:{nonce}"
            signature = wallet.hotkey.sign(message.encode()).hex()
            
            if attempt == 1:
                bt.logging.info(f"📤 Submitting {len(reveal_results)} reveals to gateway...")
            else:
                bt.logging.warning(f"🔄 Retry {attempt}/3: Submitting with fresh nonce/signature...")
            
            # Submit reveal batch
            response = requests.post(
                f"{GATEWAY_URL}/reveal/batch",
                json={
                    "validator_hotkey": wallet.hotkey.ss58_address,
                    "epoch_id": epoch_id,
                    "signature": signature,
                    "nonce": nonce,
                    "reveals": reveal_results
                },
                timeout=600  # 10 minutes timeout (gateway needs time for consensus re-calculation + database operations)
            )
            response.raise_for_status()
            
            if attempt > 1:
                bt.logging.info(f"✅ Retry {attempt}/3 succeeded!")
            bt.logging.info(f"✅ Submitted reveal for {len(reveal_results)} leads")
            return True
            
        except Exception as e:
            if attempt < 3:
                bt.logging.warning(f"⚠️  Attempt {attempt}/3 failed: {e}")
                bt.logging.warning(f"   Retrying with fresh nonce/signature...")
                time.sleep(2)  # Brief delay before retry
                continue  # Try again
            else:
                # All attempts exhausted
                bt.logging.error(f"❌ All 3 attempts failed. Last error: {e}")
                bt.logging.error(f"Failed to reveal validation for epoch {epoch_id}")
                return False