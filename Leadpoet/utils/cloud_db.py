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

load_dotenv()

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
            bt.logging.error(f"RPC call failed: {e.response.text if e.response else str(e)}")
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
            bt.logging.warning("No SUPABASE_JWT found - Supabase client not available")
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
    """Lightweight on-chain permission checks."""
    def __init__(self):
        self._network = NETWORK
        self._netuid = SUBNET_ID

    def _get_fresh_metagraph(self, network=None, netuid=None):
        """
        Always get a fresh metagraph.
        If network/netuid not provided, use defaults from environment/config.
        """
        net = network or self._network
        nid = netuid or self._netuid
        subtensor = bt.subtensor(network=net)
        return subtensor.metagraph(netuid=nid)

    def is_miner(self, ss58: str, network=None, netuid=None) -> bool:
        try:
            mg = self._get_fresh_metagraph(network, netuid)
            return ss58 in mg.hotkeys
        except Exception as e:
            bt.logging.warning(f"Failed to verify miner registration: {e}")
            return False

    def is_validator(self, ss58: str, network=None, netuid=None) -> bool:
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
        bt.logging.warning(           # ← NEW: soft-fail instead of raise
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
        
        # Insert each prospect into the queue
        records = []
        for prospect in prospects:
            records.append({
                "miner_hotkey": wallet.hotkey.ss58_address,
                "prospect": prospect,
                "status": "pending"
            })
        
        # Minimal logging
        bt.logging.debug(f"Pushing {len(records)} prospects to Supabase queue")
        
        # Batch insert (CustomResponse already executes, no .execute() needed)
        supabase.table("prospect_queue").insert(records)
        
        bt.logging.info(f"✅ Pushed {len(prospects)} prospects to Supabase queue")
        print(f"✅ Supabase queue ACK: {len(prospects)} prospect(s)")
        return True
        
    except Exception as e:
        error_str = str(e)
        
        # Check if this is a duplicate lead error (409 Conflict or explicit message)
        if "409" in error_str or "Conflict" in error_str or "Duplicate lead" in error_str or "already exists" in error_str:
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
                    email = prospect.get('owner_email', prospect.get('email', ''))
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
        
        # Check for RLS policy violations
        elif "row-level security policy" in error_str.lower() or "policy" in error_str.lower():
            bt.logging.error("❌ Access denied: Row-level security policy violation")
            bt.logging.error("   Your JWT role may not have permission to insert prospects")
            return False
        
        # Generic error
        else:
            bt.logging.error(f"❌ Failed to push prospects to Supabase: {e}")
            return False


def fetch_prospects_from_cloud(wallet: bt.wallet, limit: int = 100) -> List[Dict]:
    """
    CONSENSUS VERSION: First-come-first-served prospect fetching.
    Validators fetch prospects they haven't pulled yet, where pull_count < 3.
    Each prospect can be pulled by up to 3 validators for consensus.
    Returns a list of prospect data with their IDs for tracking.
    """
    if not _VERIFY.is_validator(wallet.hotkey.ss58_address):
        bt.logging.warning(
            f"Hotkey {wallet.hotkey.ss58_address[:10]}… is NOT a registered validator"
        )
        return []
    
    try:
        # Get Supabase client with validator's JWT token
        supabase = get_supabase_client()
        if not supabase:
            bt.logging.warning("⚠️ Supabase client not available")
            return []
        
        validator_hotkey = wallet.hotkey.ss58_address
        
        # Use the SQL function for atomic pull operation to prevent race conditions
        # This ensures true first-come-first-served behavior
        result = supabase.rpc('pull_prospects_for_validator', {
            'p_validator_hotkey': validator_hotkey,
            'p_limit': limit
        }).execute()
        
        if not result.data:
            # Fallback to Python-based approach if SQL function doesn't exist
            bt.logging.debug("SQL function not available, using Python-based approach")
            
            # Fetch prospects where:
            # 1. Status is pending
            # 2. Pull count is less than 3
            # 3. This validator hasn't pulled it yet
            # 4. Consensus status is pending
        result = supabase.table("prospect_queue") \
            .select("*") \
            .eq("status", "pending") \
                .lt("pull_count", 3) \
                .eq("consensus_status", "pending") \
            .order("created_at", desc=False) \
                .limit(limit * 2).execute()  # Get more to filter in Python
        
        if not result.data:
            return []
        
            # Filter out prospects this validator has already pulled
            available_prospects = []
            for row in result.data:
                validators_pulled = row.get('validators_pulled', [])
                if validator_hotkey not in validators_pulled:
                    available_prospects.append(row)
                    if len(available_prospects) >= limit:
                        break
            
            if not available_prospects:
                return []
            
            # Update each prospect to mark this validator has pulled it
            prospects_with_ids = []
            for prospect_row in available_prospects:
                prospect_id = prospect_row['id']
                current_validators = prospect_row.get('validators_pulled', [])
                current_pull_count = prospect_row.get('pull_count', 0)
                
                # Update the prospect to add this validator
                update_result = supabase.table("prospect_queue") \
            .update({
                        "validators_pulled": current_validators + [validator_hotkey],
                        "pull_count": current_pull_count + 1
            }) \
                    .eq("id", prospect_id) \
            .execute()
        
                if update_result.data:
                    # Return prospect data with ID for tracking
                    # Include miner_hotkey from the prospect_queue row
                    prospect_data = prospect_row.get('prospect', {})
                    if prospect_data and isinstance(prospect_data, dict):
                        prospect_data = prospect_data.copy()
                        # Add miner_hotkey if available
                        if prospect_row.get('miner_hotkey'):
                            prospect_data['miner_hotkey'] = prospect_row['miner_hotkey']
                    prospects_with_ids.append({
                        'prospect_id': prospect_id,
                        'data': prospect_data
                    })
            
            bt.logging.info(f"✅ Pulled {len(prospects_with_ids)} prospects (first-come-first-served)")
            return prospects_with_ids
        
        else:
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
                    'prospect_id': row.get('prospect_id', row.get('id')),
                    'data': prospect_data
                })
            
            bt.logging.info(f"✅ Atomically pulled {len(prospects_with_ids)} prospects (first-come-first-served)")
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
    score: float,
    is_valid: bool
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
        score: Validation score (0.0 to 1.0)
        is_valid: Boolean indicating if validator considers lead valid
    
    Returns:
        bool: True if submission successful, False otherwise
    """
    try:
        # Verify this is a validator
        if not _VERIFY.is_validator(wallet.hotkey.ss58_address):
            bt.logging.warning(f"Hotkey {wallet.hotkey.ss58_address[:10]}… is NOT a registered validator")
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
            "score": round(float(score), 2),  # Ensure it's a float with 2 decimal places
            "is_valid": bool(is_valid),
            "epoch_number": epoch_number
        }
        
        # Debug: Log what we're trying to insert
        bt.logging.info("🔍 DEBUG: Attempting to insert validation data:")
        bt.logging.info(f"   - validator_hotkey: {wallet.hotkey.ss58_address}")
        bt.logging.info(f"   - prospect_id: {prospect_id}")
        bt.logging.info(f"   - lead_id: {lead_id}")
        bt.logging.info(f"   - score: {validation_data['score']}")
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
            bt.logging.info(f"✅ Submitted validation for lead {lead_id[:8]}... (score: {score:.2f}, valid: {is_valid})")
            
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

# DEPRECATED: This function is no longer used - consensus is handled by database triggers
# Keeping for reference only
def check_and_process_consensus(
    prospect_id: str, 
    lead_id: str, 
    lead_data: Dict,
    wallet: bt.wallet = None
) -> bool:
    """
    Check if consensus has been reached for a lead and process accordingly.
    IMPORTANT: 
    - All validations go to validation_tracking table
    - Only ACCEPTED leads (2/3 valid) go to the main leads table
    - Rejected leads stay only in validation_tracking
    
    Args:
        prospect_id: UUID of the prospect from prospect_queue
        lead_id: UUID of the lead being validated
        lead_data: The full lead data to insert if accepted
        wallet: Optional validator wallet for logging
    
    Returns:
        bool: True if consensus was reached (3 validations), False otherwise
    """
    try:
        # Get Supabase client with service role for reading validation_tracking
        import os
        from supabase import create_client
        
        SUPABASE_URL = "https://qplwoislplkcegvdmbim.supabase.co"
        service_role_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
        
        if not service_role_key:
            bt.logging.error("Service role key not available for consensus checking")
            return False
            
        # Use service role client to read validation_tracking
        service_supabase = create_client(SUPABASE_URL, service_role_key)
        
        # Get all validations for this PROSPECT (not lead_id, since each validator generates different lead_id)
        validations = service_supabase.table("validation_tracking") \
            .select("*") \
            .eq("prospect_id", prospect_id) \
            .execute()
        
        if not validations.data:
            bt.logging.debug(f"No validations found for lead {lead_id[:8]}...")
            return False
        
        validation_count = len(validations.data)
        bt.logging.debug(f"Lead {lead_id[:8]}... has {validation_count}/3 validations")
        
        # Check if we have all 3 validations
        if validation_count >= 3:
            # Count valid and invalid votes
            valid_count = sum(1 for v in validations.data if v['is_valid'])
            invalid_count = validation_count - valid_count
            
            # Calculate average score
            avg_score = sum(v['score'] for v in validations.data) / validation_count
            
            # Get list of validators who participated
            validators = [v['validator_hotkey'] for v in validations.data]
            
            bt.logging.info(f"📊 Consensus check for lead {lead_id[:8]}...")
            bt.logging.info(f"   Valid votes: {valid_count}/3")
            bt.logging.info(f"   Invalid votes: {invalid_count}/3")
            bt.logging.info(f"   Average score: {avg_score:.2f}")
            
            # Determine consensus decision
            if valid_count >= 2:  # ACCEPTED - 2 or more validators said VALID
                bt.logging.info(f"✅ Lead {lead_id[:8]}... ACCEPTED by consensus ({valid_count}/3 valid)")
                
                # Prepare lead data for insertion into main leads table
                lead_data_for_insert = lead_data.copy()
                
                # Map owner_email to email (required field)
                if 'owner_email' in lead_data_for_insert and 'email' not in lead_data_for_insert:
                    lead_data_for_insert['email'] = lead_data_for_insert['owner_email']
                
                # Add consensus metadata
                lead_data_for_insert['lead_id'] = lead_id
                lead_data_for_insert['prospect_id'] = prospect_id
                lead_data_for_insert['consensus_score'] = round(avg_score, 2)
                lead_data_for_insert['validator_count'] = validation_count
                lead_data_for_insert['consensus_validators'] = validators
                lead_data_for_insert['consensus_status'] = 'accepted'
                lead_data_for_insert['validated_at'] = datetime.now(timezone.utc).isoformat()
                
                # Insert into MAIN leads table using SERVICE ROLE (only service role has access)
                try:
                    # Need to use service role client for leads table access
                    import os
                    from supabase import create_client
                    
                    SUPABASE_URL = "https://qplwoislplkcegvdmbim.supabase.co"
                    service_role_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
                    
                    if service_role_key:
                        service_supabase = create_client(SUPABASE_URL, service_role_key)
                        leads_result = service_supabase.table("leads").insert([lead_data_for_insert]).execute()
                    else:
                        bt.logging.error("Service role key not available for leads table insert")
                        leads_result = None
                    
                    if leads_result and leads_result.data:
                        bt.logging.info(f"✅ Lead {lead_id[:8]}... added to main leads database")
                        bt.logging.info(f"   Email: {lead_data_for_insert.get('email', 'unknown')}")
                        bt.logging.info(f"   Business: {lead_data_for_insert.get('business', 'unknown')}")
                        bt.logging.info(f"   Consensus: {valid_count}/3 validators approved")
                    else:
                        bt.logging.error("Failed to insert accepted lead into leads table")
                except Exception as e:
                    bt.logging.error(f"Error inserting lead into main database: {e}")
                
                # Update prospect_queue status to accepted
                try:
                    queue_update = service_supabase.table("prospect_queue") \
                        .update({
                            "consensus_status": "accepted",
                            "consensus_timestamp": datetime.now(timezone.utc).isoformat()
                        }) \
                        .eq("id", prospect_id) \
                        .execute()
                    
                    if queue_update.data:
                        bt.logging.debug("Updated prospect_queue status to accepted")
                except Exception as e:
                    bt.logging.error(f"Error updating prospect_queue status: {e}")
                
            else:  # REJECTED - 2 or more validators said INVALID
                bt.logging.info(f"❌ Lead {lead_id[:8]}... REJECTED by consensus ({invalid_count}/3 invalid)")
                
                # DO NOT insert into main leads table
                # Only update prospect_queue status to rejected
                try:
                    queue_update = service_supabase.table("prospect_queue") \
                        .update({
                            "consensus_status": "rejected",
                            "consensus_timestamp": datetime.now(timezone.utc).isoformat()
                        }) \
                        .eq("id", prospect_id) \
                        .execute()
                    
                    if queue_update.data:
                        bt.logging.debug("Updated prospect_queue status to rejected")
                except Exception as e:
                    bt.logging.error(f"Error updating prospect_queue status: {e}")
            
            # Consensus was reached (whether accepted or rejected)
            return True
            
        else:
            # Not enough validations yet
            bt.logging.debug(f"Waiting for more validations ({validation_count}/3)")
            return False
            
    except Exception as e:
        bt.logging.error(f"❌ Failed to check consensus: {e}")
        import traceback
        bt.logging.debug(traceback.format_exc())
        return False

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

    Args:
        wallet: Bittensor wallet
        role: "validator" or "miner" - determines which requests to fetch
    """
    try:
        supabase = get_supabase_client()
        if not supabase:
            return []

        # Fetch pending requests
        result = supabase.table("api_requests") \
            .select("*") \
            .eq("status", "pending") \
            .order("created_at", desc=False) \
            .limit(10) \
            .execute()

        requests_list = result.data if result.data else []

        # Only log when requests are found
        if requests_list:
            bt.logging.info(f"🔔 [{role.upper()}] Found {len(requests_list)} NEW broadcast request(s)!")

        return requests_list

    except Exception as e:
        bt.logging.error(f"fetch_broadcast_requests ({role}) failed: {e}")
        return []


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
# DEPRECATED: This function should not be used by validators - metagraph sync should be done server-side
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