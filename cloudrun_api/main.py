from fastapi import FastAPI, HTTPException, Request, status
from google.cloud import firestore
import bittensor as bt, time, base64, json

app = FastAPI()
db  = firestore.Client()          # auto picks project from env

COLLECTION = "leads"              # change for prospect_queue etc.
MAX_BATCH  = 500                  # Firestore batch write limit

# ---------- Helper: verify validator ------------
def is_validator(wallet_ss58: str) -> bool:
    # Light-weight on-chain check via Subtensor.
    # Uses default finney endpoint; tweak if needed.
    subtensor = bt.subtensor(network="test")
    metagraph = subtensor.metagraph(netuid=401)
    try:
        uid = metagraph.hotkeys.index(wallet_ss58)
        return metagraph.validator_permit[uid].item()   # True / False
    except ValueError:
        return False
# ---------- Helper: verify signature ------------
def verify_sig(wallet_ss58: str, payload: bytes, signature_b64: str) -> bool:
    sig_bytes = base64.b64decode(signature_b64)
    wallet    = bt.wallet(ss58_address=wallet_ss58)
    return wallet.verify_sig(payload, sig_bytes)

# ---------- Public read -------------------------
@app.get("/leads")
async def get_leads(limit: int = 100):
    try:
        docs = db.collection(COLLECTION).limit(limit).stream()
        return [doc.to_dict() | {"id": doc.id} for doc in docs]
    except Exception as e:
        import traceback, sys; traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

# ---------- Authenticated write -----------------
@app.post("/leads")
async def add_leads(request: Request):
    body = await request.json()
    wallet = body.get("wallet")            # ss58 string
    sig    = body.get("signature")         # base64 string
    leads  = body.get("leads")             # list[dict]

    # 1. basic validation
    if not (wallet and sig and isinstance(leads, list)):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                            detail="wallet, signature, leads required")

    # 2. freshness: include timestamp in signed payload
    timestamp = str(int(time.time()) // 300)            # 5-min windows
    payload   = (timestamp + json.dumps(leads, sort_keys=True)).encode()

    if not verify_sig(wallet, payload, sig):
        raise HTTPException(status_code=401, detail="Bad signature")

    if not is_validator(wallet):
        raise HTTPException(status_code=403, detail="Not a registered validator")

    # 3. commit to Firestore in batches
    batch, count = db.batch(), 0
    for lead in leads:
        doc_ref = db.collection(COLLECTION).document()
        batch.set(doc_ref, lead)
        count += 1
        if count % MAX_BATCH == 0:
            batch.commit(); batch = db.batch()
    batch.commit()
    return {"stored": count}