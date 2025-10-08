"""
Centralised Firestore helper for the LeadPoet subnet.
Miners = read-only, Validators = write-enabled (wallet-signed).
"""
import os, json, time, base64, requests, bittensor as bt
from typing import List, Dict
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

API_URL   = os.getenv("LEAD_API", "https://leadpoet-api-511161415764.us-central1.run.app")
SUBNET_ID = 401          # NetUID of your subnet
NETWORK   = "test"       # Bittensor network (finney-test)

class _Verifier:
    """Lightweight on-chain permission checks."""
    def __init__(self):
        self._network = NETWORK
        self._netuid = SUBNET_ID

    def _get_fresh_metagraph(self):
        """Always get a fresh metagraph"""
        subtensor = bt.subtensor(network=self._network)
        return subtensor.metagraph(netuid=self._netuid)

    def is_miner(self, ss58: str) -> bool:
        try:
            mg = self._get_fresh_metagraph()
            return ss58 in mg.hotkeys
        except:
            return False

    def is_validator(self, ss58: str) -> bool:
        try:
            mg = self._get_fresh_metagraph()
            uid = mg.hotkeys.index(ss58)
            return mg.validator_permit[uid].item()
        except ValueError:
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
def push_prospects_to_cloud(wallet: bt.wallet, prospects: List[Dict]) -> bool:
    """
    Miners call this to enqueue prospects for validation.
    """
    if not prospects:
        return True
    if not _VERIFY.is_miner(wallet.hotkey.ss58_address):
        raise PermissionError("Hotkey not registered as miner on subnet")

    ts      = str(int(time.time()) // 300)
    payload = (ts + json.dumps(prospects, sort_keys=True)).encode()
    sig     = base64.b64encode(wallet.hotkey.sign(payload)).decode()

    body = {"wallet": wallet.hotkey.ss58_address,
            "signature": sig,
            "prospects": prospects}
    r = requests.post(f"{API_URL}/prospects", json=body, timeout=30)
    r.raise_for_status()
    bt.logging.info(f"✅ Pushed {len(prospects)} prospects to cloud queue")
    print(f"✅ Cloud-queue ACK: {len(prospects)} prospect(s)")
    return True


def fetch_prospects_from_cloud(wallet: bt.wallet, limit: int = 100) -> List[Dict]:
    """
    Validators call this to atomically fetch + delete a batch of prospects.
    Returns an empty list when nothing is available or the endpoint is missing.
    """
    if not _VERIFY.is_validator(wallet.hotkey.ss58_address):
        bt.logging.warning(
            f"Hotkey {wallet.hotkey.ss58_address[:10]}… is NOT a registered "
            "validator – pulling prospects anyway (DEV mode)"
        )

    ts      = str(int(time.time()) // 300)
    payload = (ts + str(limit)).encode()
    sig     = base64.b64encode(wallet.hotkey.sign(payload)).decode()

    body = {
        "wallet":    wallet.hotkey.ss58_address,
        "signature": sig,
        "limit":     limit,
    }

    try:
        r = requests.post(f"{API_URL}/prospects/fetch", json=body, timeout=30)
        if r.status_code == 404:                     # ← graceful fallback
            bt.logging.error(
                "Cloud API route /prospects/fetch not found -- did you "
                "deploy the latest server?  Returning empty prospect list."
            )
            return []
        r.raise_for_status()
        return r.json()
    except requests.exceptions.RequestException as e:
        bt.logging.error(f"fetch_prospects_from_cloud: {e}")
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
    ts       = str(int(time.time()) // 300)
    payload  = (ts + json.dumps(extra, sort_keys=True)).encode()
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

def broadcast_api_request(wallet: bt.wallet, request_id: str, num_leads: int, business_desc: str, client_id: str = None) -> bool:
    """
    Broadcast an API request to Firestore for ALL validators and miners to process.

    Args:
        wallet: Client's wallet
        request_id: Unique request ID
        num_leads: Number of leads requested
        business_desc: Business description
        client_id: Optional client identifier

    Returns:
        bool: True if broadcast successful, False otherwise
    """
    try:
        from datetime import datetime
        from google.cloud import firestore

        if not _has_firestore_credentials():
            bt.logging.warning("Firestore credentials not available, cannot broadcast request")
            return False

        db = firestore.Client()

        # Create broadcast request document
        doc_ref = db.collection("api_requests").document(request_id)

        doc_ref.set({
            "request_id": request_id,
            "client_hotkey": wallet.hotkey.ss58_address,
            "client_id": client_id or "unknown",
            "num_leads": num_leads,
            "business_desc": business_desc,
            "status": "pending",
            "created_at": datetime.utcnow().isoformat() + "Z",
            "updated_at": datetime.utcnow().isoformat() + "Z",
        })

        bt.logging.info(f"📡 Broadcast API request {request_id[:8]}... to Firestore")
        return True

    except Exception as e:
        bt.logging.error(f"Failed to broadcast API request: {e}")
        import traceback
        bt.logging.error(traceback.format_exc())
        return False


def fetch_broadcast_requests(wallet: bt.wallet, role: str = "validator") -> List[Dict]:
    """
    Fetch pending broadcast API requests for this validator/miner from Firestore.
    Returns list of pending requests that need processing.

    Args:
        wallet: Bittensor wallet
        role: "validator" or "miner" - determines which requests to fetch
    """
    try:
        from google.cloud import firestore
        import warnings

        # Suppress Firestore positional argument warnings
        warnings.filterwarnings("ignore", message=".*positional arguments.*")

        # Check if Google Cloud credentials exist
        if not _has_firestore_credentials():
            # Silently return empty list if no credentials (development mode)
            return []

        db = firestore.Client()

        # BOTH validators and miners should ONLY fetch "pending" requests
        # Local tracking (processed_requests set) prevents re-processing
        query = db.collection("api_requests").where("status", "==", "pending").limit(10)

        docs = query.stream()
        requests_list = []

        for doc in docs:
            data = doc.to_dict()
            data["request_id"] = doc.id  # Ensure request_id is set
            requests_list.append(data)

        # Only log when requests are found
        if requests_list:
            print(f"\n🔔 [{role.upper()}] Found {len(requests_list)} NEW broadcast request(s)!")
            for req in requests_list:
                print(f"   📨 Request {req['request_id'][:8]}... - {req.get('business_desc', '')[:30]}")

        return requests_list

    except Exception as e:
        print(f"❌ fetch_broadcast_requests ({role}) FAILED: {e}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return []


def mark_broadcast_processing(wallet: bt.wallet, request_id: str) -> bool:
    """
    Mark a broadcast request as being processed to prevent duplicates.
    Uses Firestore transaction for atomic update.
    Only ONE miner will successfully mark it.
    """
    try:
        from google.cloud import firestore

        if not _has_firestore_credentials():
            return False

        db = firestore.Client()

        doc_ref = db.collection("api_requests").document(request_id)

        # Use a transaction for atomic read-modify-write
        transaction = db.transaction()

        @firestore.transactional
        def update_in_transaction(transaction, doc_ref):
            snapshot = doc_ref.get(transaction=transaction)

            if not snapshot.exists:
                return False

            data = snapshot.to_dict()
            current_status = data.get("status")

            # Only allow processing if status is "pending"
            if current_status != "pending":
                return False

            # Atomic update: only one miner wins
            transaction.update(doc_ref, {
                "status": "processing",
                "processing_by": wallet.hotkey.ss58_address,
            })
            return True

        success = update_in_transaction(transaction, doc_ref)

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
    Get the status of a broadcast API request from Firestore.
    Used by validators and clients to check request status.
    """
    try:
        from google.cloud import firestore

        if not _has_firestore_credentials():
            return {"status": "error", "leads": [], "error": "Firestore credentials not available"}

        db = firestore.Client()

        doc_ref = db.collection("api_requests").document(request_id)
        doc = doc_ref.get()

        if not doc.exists:
            return {"status": "not_found", "leads": [], "request_id": request_id}

        data = doc.to_dict()
        return data

    except Exception as e:
        bt.logging.error(f"Failed to get status for request {request_id[:8]}...: {e}")
        return {"status": "error", "leads": [], "error": str(e)}

def push_validator_ranking(wallet: bt.wallet, request_id: str, ranked_leads: List[Dict], validator_trust: float) -> bool:
    """
    Submit validator's ranking for a broadcast API request directly to Firestore.

    Args:
        wallet: Validator's wallet
        request_id: Broadcast request ID
        ranked_leads: List of leads with scores and ranks
        validator_trust: Validator's trust value from metagraph

    Returns:
        bool: Success status
    """
    from datetime import datetime
    from google.cloud import firestore

    # Get validator UID from metagraph
    try:
        mg = _VERIFY._get_fresh_metagraph()
        validator_uid = mg.hotkeys.index(wallet.hotkey.ss58_address)
    except ValueError:
        validator_uid = -1  # Unknown UID

    try:
        if not _has_firestore_credentials():
            bt.logging.warning("Firestore credentials not available, cannot push ranking")
            return False

        db = firestore.Client()

        # Document ID: {request_id}_{validator_hotkey}
        doc_id = f"{request_id}_{wallet.hotkey.ss58_address}"
        doc_ref = db.collection("validator_rankings").document(doc_id)

        # Write ranking to Firestore
        doc_ref.set({
            "request_id": request_id,
            "validator_hotkey": wallet.hotkey.ss58_address,
            "validator_uid": validator_uid,
            "validator_trust": validator_trust,
            "ranked_leads": ranked_leads,
            "num_leads_ranked": len(ranked_leads),
            "submitted_at": datetime.utcnow().isoformat() + "Z",
        })

        bt.logging.info(f"📊 Submitted ranking for request {request_id[:8]}... ({len(ranked_leads)} leads)")
        return True

    except Exception as e:
        bt.logging.error(f"Failed to submit validator ranking: {e}")
        return False


def fetch_validator_rankings(request_id: str, timeout_sec: int = 5) -> List[Dict]:
    """
    Fetch all validator rankings for a broadcast request from Firestore.

    Args:
        request_id: Broadcast request ID
        timeout_sec: Not used (kept for API compatibility)

    Returns:
        List of validator ranking submissions
    """
    try:
        from google.cloud import firestore
        import warnings

        # Suppress warnings
        warnings.filterwarnings("ignore", message=".*positional arguments.*")

        if not _has_firestore_credentials():
            return []

        db = firestore.Client()

        # Query all validator rankings for this request
        query = db.collection("validator_rankings").where("request_id", "==", request_id)

        docs = query.stream()
        rankings = []

        for doc in docs:
            data = doc.to_dict()
            rankings.append(data)

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
    Push miner's curated leads to Firestore for validators to pick up.

    Args:
        wallet: Miner's wallet
        request_id: Broadcast request ID
        leads: Curated leads from miner

    Returns:
        bool: Success status
    """
    try:
        from google.cloud import firestore
        from datetime import datetime

        if not _has_firestore_credentials():
            bt.logging.warning("Firestore credentials not available, cannot push miner leads")
            return False

        db = firestore.Client()

        # Document ID: {request_id}_{miner_hotkey}
        doc_id = f"{request_id}_{wallet.hotkey.ss58_address}"
        doc_ref = db.collection("miner_submissions").document(doc_id)

        doc_ref.set({
            "request_id": request_id,
            "miner_hotkey": wallet.hotkey.ss58_address,
            "leads": leads,
            "num_leads": len(leads),
            "submitted_at": datetime.utcnow().isoformat() + "Z",
        })

        bt.logging.info(f"📤 Pushed {len(leads)} curated lead(s) to Firestore for request {request_id[:8]}...")
        return True

    except Exception as e:
        bt.logging.error(f"Failed to push miner leads: {e}")
        return False


def fetch_miner_leads_for_request(request_id: str) -> List[Dict]:
    """
    Fetch all miner submissions for a broadcast request.

    Args:
        request_id: Broadcast request ID

    Returns:
        List of miner submission dicts
    """
    try:
        from google.cloud import firestore
        import warnings

        # Suppress warnings
        warnings.filterwarnings("ignore", message=".*positional arguments.*")

        if not _has_firestore_credentials():
            return []

        db = firestore.Client()

        # Query all miner submissions for this request
        query = db.collection("miner_submissions").where("request_id", "==", request_id)

        docs = query.stream()
        submissions = []

        for doc in docs:
            data = doc.to_dict()
            submissions.append(data)

        return submissions

    except Exception as e:
        bt.logging.debug(f"Failed to fetch miner leads: {e}")
        return []