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
        raise PermissionError("Hotkey not registered as validator on subnet")

    ts       = str(int(time.time()) // 300)                 # 5-min window
    payload  = (ts + json.dumps(leads, sort_keys=True)).encode()
    sig_b64  = base64.b64encode(wallet.sign(payload)).decode()

    body = {
        "wallet":    wallet.hotkey.ss58_address,
        "signature": sig_b64,
        "leads":     leads,
    }
    r = requests.post(f"{API_URL}/leads", json=body, timeout=30)
    r.raise_for_status()
    return True