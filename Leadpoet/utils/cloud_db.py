"""
Centralised Firestore helper for the LeadPoet subnet.
Miners = read-only, Validators = write-enabled (wallet-signed).
"""
import os, json, time, base64, requests, bittensor as bt
from typing import List, Dict

API_URL   = os.getenv("LEAD_API", "https://leadpoet-api-511161415764.us-central1.run.app")
SUBNET_ID = 401          # NetUID of your subnet
NETWORK   = "test"       # Bittensor network (finney-test)

class _Verifier:
    """Lightweight on-chain permission checks."""
    def __init__(self):
        self._subtensor  = bt.subtensor(network=NETWORK)
        self._metagraph  = self._subtensor.metagraph(netuid=SUBNET_ID)

    def is_miner(self, ss58: str) -> bool:
        try:
            return ss58 in self._metagraph.hotkeys
        except:                             # noqa
            return False

    def is_validator(self, ss58: str) -> bool:
        try:
            uid = self._metagraph.hotkeys.index(ss58)
            return self._metagraph.validator_permit[uid].item()
        except ValueError:
            return False

_VERIFY = _Verifier()                       # singleton

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
    r = requests.post(f"{API_URL}/curate", json=payload, timeout=10)
    r.raise_for_status(); return r.json()["request_id"]

def fetch_curation_requests() -> dict:
    r = requests.post(f"{API_URL}/curate/fetch", timeout=10); r.raise_for_status()
    return r.json()

def push_curation_result(result: dict):
    requests.post(f"{API_URL}/curate/result", json=result, timeout=30).raise_for_status()

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