from fastapi import FastAPI, HTTPException, Request, status
from google.cloud import firestore
import bittensor as bt, time, base64, json, traceback

app = FastAPI()
db  = firestore.Client()          # auto picks project from env

# ------------ Firestore collections -------------
COLLECTION_LEADS      = "leads"       # verified + client-visible
COLLECTION_PROSPECTS  = "prospects"   # awaiting validator review
MAX_BATCH  = 500                     # Firestore batch-write limit

# ---------- Helper: role checks -----------------
def is_miner(wallet_ss58: str) -> bool:
    subtensor = bt.subtensor(network="test")
    metagraph = subtensor.metagraph(netuid=401)
    return wallet_ss58 in metagraph.hotkeys
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
    """
    Best-effort Ed25519 verification.
    Never raises – returns False on any failure so callers can decide
    whether to continue in DEV mode.
    """
    try:
        sig_bytes = base64.b64decode(signature_b64)
        w         = bt.wallet(ss58_address=wallet_ss58)
        return w.verify_sig(payload, sig_bytes)
    except Exception as e:
        print(f"[WARN] verify_sig failed: {e}")
        traceback.print_exc()
        return False

# ---------- Public READ (unchanged, but use new constant) -----------
@app.get("/leads")
async def get_leads(limit: int = 100):
    try:
        docs = db.collection(COLLECTION_LEADS).limit(limit).stream()
        return [doc.to_dict() | {"id": doc.id} for doc in docs]
    except Exception as e:
        import traceback; traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# ---------- VALIDATED write (unchanged, but use new constant) -------
@app.post("/leads")
async def add_leads(request: Request):
    body    = await request.json()
    wallet  = body.get("wallet")
    sig     = body.get("signature")
    leads   = body.get("leads")

    # 1. basic validation
    if not (wallet and sig and isinstance(leads, list)):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                            detail="wallet, signature, leads required")

    # 2. freshness: include timestamp in signed payload
    timestamp = str(int(time.time()) // 300)            # 5-min windows
    payload   = (timestamp + json.dumps(leads, sort_keys=True)).encode()

    if not verify_sig(wallet, payload, sig):
        print("[WARN] bad signature – continuing (DEV mode)")

    if not is_validator(wallet):
        print("[WARN] hotkey lacks validator-permit – continuing (DEV mode)")

    # 3. commit to Firestore
    batch, count = db.batch(), 0
    for lead in leads:
        doc_ref = db.collection(COLLECTION_LEADS).document()
        batch.set(doc_ref, lead)
        count += 1
        if count % MAX_BATCH == 0:
            batch.commit(); batch = db.batch()
    batch.commit()
    return {"stored": count}


# ===================================================================
# NEW:  miners → `/prospects` (enqueue)  |  validators ← `/prospects/fetch`
# ===================================================================

@app.post("/prospects")                     # miners push here
async def enqueue_prospects(request: Request):
    body      = await request.json()
    wallet_ss = body.get("wallet")
    sig       = body.get("signature")
    prospects = body.get("prospects")

    if not (wallet_ss and sig and isinstance(prospects, list)):
        raise HTTPException(status_code=400, detail="wallet, signature, prospects required")

    # time-boxed payload identical to miners’ signing scheme
    timestamp = str(int(time.time()) // 300)
    payload   = (timestamp + json.dumps(prospects, sort_keys=True)).encode()

    # soft-fail on bad sig / permit
    if not verify_sig(wallet_ss, payload, sig):
        print("[WARN] bad signature – continuing (DEV mode)")
    if not is_miner(wallet_ss):
        print("[WARN] hotkey lacks miner-permit – continuing (DEV mode)")

    # single document per miner-push  → keeps miner_hotkey & list intact
    doc = {
        "miner_hotkey": wallet_ss,
        "prospects":    prospects,
        "created_at":   firestore.SERVER_TIMESTAMP,
    }
    db.collection(COLLECTION_PROSPECTS).add(doc)
    return {"queued": len(prospects)}


@app.post("/prospects/fetch")               # validators pull + delete
async def fetch_prospects(request: Request):
    body      = await request.json()
    wallet_ss = body.get("wallet")
    sig       = body.get("signature")
    limit     = int(body.get("limit", 100))

    timestamp = str(int(time.time()) // 300)
    payload   = (timestamp + str(limit)).encode()

    # soft-fail on bad sig / permit
    if not verify_sig(wallet_ss, payload, sig):
        print("[WARN] bad signature – continuing (DEV mode)")
    if not is_validator(wallet_ss):
        print("[WARN] hotkey lacks validator-permit – continuing (DEV mode)")

    try:
        docs   = list(db.collection(COLLECTION_PROSPECTS).limit(limit).stream())
        leads  = [d.to_dict() | {"id": d.id} for d in docs]

        batch = db.batch()
        for d in docs:
            batch.delete(d.reference)
        batch.commit()

        return leads
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))