from fastapi import FastAPI, HTTPException, Request, status
import bittensor as bt, time, base64, json, traceback
import os
import signal

app = FastAPI()

DEMO_MODE = os.getenv("DEMO_MODE", "true").lower() == "true"

db = None
SERVER_TIMESTAMP = None  # Will be set if Firestore is available

if not DEMO_MODE:
    from google.cloud import firestore
    try:
        db = firestore.Client()
        SERVER_TIMESTAMP = firestore.SERVER_TIMESTAMP
        print("[INFO] Firestore client initialized successfully")
    except Exception as e:
        print(f"[ERROR] Database initialization failed in production mode: {e}")
        raise
else:
    print("[INFO] Running in DEMO_MODE - Firestore not loaded, database operations will return 503")

@app.get("/")
async def root():
    return {
        "name": "LeadPoet API",
        "version": "1.0.0",
        "status": "running",
        "database": "connected" if db else "demo_mode",
        "mode": "demo" if DEMO_MODE else "production"
    }

# ------------ Firestore collections -------------
COLLECTION_LEADS      = "leads"       # verified + client-visible
COLLECTION_PROSPECTS  = "prospects"   # awaiting validator review
MAX_BATCH  = 500                     # Firestore batch-write limit

# ───────── new collections ─────────────────
COLL_CURATE_REQ  = "curation_requests"
COLL_CURATE_RES  = "curation_results"

# NEW: collections for validator↔miner fallback ----------------------
COLL_MINER_REQ   = "curation_miner_requests"
COLL_MINER_RES   = "curation_miner_results"

# ---------- Helper: check database connection -------
def check_db():
    if db is None:
        raise HTTPException(
            status_code=503,
            detail="Database not configured. This instance is running in demo mode."
        )

# ---------- Helper: role checks -----------------
def is_miner(wallet_ss58: str) -> bool:
    try:
        subtensor = bt.subtensor(network="test")
        metagraph = subtensor.metagraph(netuid=401)
        return wallet_ss58 in metagraph.hotkeys
    except Exception as e:
        print(f"[WARN] is_miner check failed: {e}")
        if DEMO_MODE:
            print("[WARN] DEMO_MODE - allowing request")
            return True
        return False
        
# ---------- Helper: verify validator ------------
def is_validator(wallet_ss58: str) -> bool:
    # Light-weight on-chain check via Subtensor.
    # Uses default finney endpoint; tweak if needed.
    try:
        subtensor = bt.subtensor(network="test")
        metagraph = subtensor.metagraph(netuid=401)
        uid = metagraph.hotkeys.index(wallet_ss58)
        return metagraph.validator_permit[uid].item()   # True / False
    except Exception as e:
        print(f"[WARN] is_validator check failed: {e}")
        if DEMO_MODE:
            print("[WARN] DEMO_MODE - allowing request")
            return True
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
    check_db()
    try:
        docs = db.collection(COLLECTION_LEADS).limit(limit).stream()
        return [doc.to_dict() | {"id": doc.id} for doc in docs]
    except Exception as e:
        import traceback; traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# ---------- VALIDATED write (unchanged, but use new constant) -------
@app.post("/leads")
async def add_leads(request: Request):
    check_db()
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

    # 3. commit, but skip duplicates (same e-mail already stored)
    batch, stored, dupes = db.batch(), 0, 0
    for lead in leads:
        email = (
            lead.get("email")
            or lead.get("owner_email")
            or lead.get("Owner(s) Email")
            or ""
        ).lower()

        if email:
            col = db.collection(COLLECTION_LEADS)
            dup = any(col.where("email",        "==", email).limit(1).stream()) \
                or any(col.where("owner_email", "==", email).limit(1).stream())
            if dup:
                print(f"[INFO] duplicate e-mail skipped: {email}")
                dupes += 1
                continue

        # always store both aliases to keep future checks simple
        doc_ref = db.collection(COLLECTION_LEADS).document()
        batch.set(
            doc_ref,
            lead
            | {"email": email,
               "owner_email": email}
        )
        stored += 1
    # flush
    batch.commit()
    return {"stored": stored, "duplicates": dupes}


# ===================================================================
# NEW:  miners → `/prospects` (enqueue)  |  validators ← `/prospects/fetch`
# ===================================================================

@app.post("/prospects")                     # miners push here
async def enqueue_prospects(request: Request):
    check_db()
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
        "created_at":   SERVER_TIMESTAMP,
    }
    db.collection(COLLECTION_PROSPECTS).add(doc)
    return {"queued": len(prospects)}


@app.post("/prospects/fetch")               # validators pull + delete
async def fetch_prospects(request: Request):
    check_db()
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

# ───────── buyer submits request ───────────
@app.post("/curate")
async def create_curation(request: Request):
    check_db()
    body = await request.json()
    doc   = db.collection(COLL_CURATE_REQ).document()
    doc.set({
        "created_at": SERVER_TIMESTAMP,
        **body               # num_leads, business_desc, maybe wallet
    })
    return {"request_id": doc.id}

# ───────── validator pulls next request ────
@app.post("/curate/fetch")
async def fetch_curation(request: Request):
    check_db()
    docs = list(db.collection(COLL_CURATE_REQ).order_by("created_at")
                                   .limit(1).stream())
    if not docs:
        return {}
    d = docs[0]
    db.collection(COLL_CURATE_REQ).document(d.id).delete()
    return d.to_dict() | {"request_id": d.id}

# ───────── validator pushes result ─────────
@app.post("/curate/result")
async def push_curation_result(request: Request):
    check_db()
    body = await request.json()          # {request_id, leads:[…]}
    req_id = body.get("request_id")
    if not req_id:
        raise HTTPException(status_code=400, detail="request_id missing")
    db.collection(COLL_CURATE_RES).document(req_id).set(body)
    return {"stored": True}

# ───────── buyer polls result ──────────────
@app.get("/curate/result/{request_id}")
async def get_curation_result(request_id: str):
    check_db()
    doc = db.collection(COLL_CURATE_RES).document(request_id).get()
    if not doc.exists:
        return {}
    return doc.to_dict()

# ───────── validator → miner ------------------------------------------------
@app.post("/curate/miner_request")
async def create_miner_request(request: Request):
    check_db()
    body = await request.json()                           # num_leads , business_desc , …
    doc  = db.collection(COLL_MINER_REQ).document()
    doc.set({"created_at": SERVER_TIMESTAMP, **body})
    return {"miner_request_id": doc.id}

# ───────── miner pulls next queued query ───────────────────────────
@app.post("/curate/miner_request/fetch")
async def fetch_miner_request(_: Request):
    check_db()
    docs = list(db.collection(COLL_MINER_REQ).order_by("created_at").limit(1).stream())
    if not docs:
        return {}
    d = docs[0]
    db.collection(COLL_MINER_REQ).document(d.id).delete()
    return d.to_dict() | {"miner_request_id": d.id}

# ───────── miner pushes curated leads back ─────────────────────────
@app.post("/curate/miner_result")
async def push_miner_result(request: Request):
    check_db()
    body = await request.json()               # miner_request_id , leads
    rid  = body.get("miner_request_id")
    if not rid:
        raise HTTPException(status_code=400, detail="miner_request_id missing")
    db.collection(COLL_MINER_RES).document(rid).set(
        {"created_at": SERVER_TIMESTAMP, **body}
    )
    return {"stored": True}

# ───────── validator polls curated leads ───────────────────────────
@app.post("/curate/miner_result/fetch")
async def fetch_miner_result(_: Request):
    check_db()
    docs = list(db.collection(COLL_MINER_RES).order_by("created_at").limit(1).stream())
    if not docs:
        return {}
    d = docs[0]
    db.collection(COLL_MINER_RES).document(d.id).delete()
    return d.to_dict()

@app.post("/validator_weights")
async def validator_weights(request: Request):
    check_db()
    body = await request.json()
    wallet_ss = body.get("wallet")
    sig       = body.get("signature")
    weights   = body.get("weights")

    if not (wallet_ss and sig and isinstance(weights, dict)):
        raise HTTPException(status_code=400, detail="wallet, signature, weights required")

    # time-boxed payload identical to miners’ signing scheme
    timestamp = str(int(time.time()) // 300)
    payload   = (timestamp + json.dumps(weights, sort_keys=True)).encode()

    # soft-fail on bad sig / permit
    if not verify_sig(wallet_ss, payload, sig):
        print("[WARN] bad signature – continuing (DEV mode)")
    if not is_validator(wallet_ss):
        print("[WARN] hotkey lacks validator-permit – continuing (DEV mode)")

    doc = {
        "hotkey":     wallet_ss,
        "weights":    weights,
        "created_at": SERVER_TIMESTAMP,
    }
    db.collection("validator_weights").add(doc)
    return {"stored": True}