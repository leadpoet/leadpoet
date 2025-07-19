"""
Leadpoet.com →  local proxy  →  running validator
-------------------------------------------------
• Listens on http://localhost:5050/lead  (POST JSON)
• Accepts: {"num_leads": int, "business_desc": "..."}
• Returns: identical JSON payload the validator sends today.
"""

import asyncio, os, socket, json
from typing import Optional, List, Dict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import aiohttp

# ───────────────────────────────────────── CONFIG

PROXY_PORT            = 5050          # do NOT change per user request
VALIDATOR_COMMON_PORT = 8093          # default used by validator.py
VALIDATOR_TIMEOUT_S   = 75            # website waits ≤ 75 s before retry

# ───────────────────────────────────────── MODELS

class LeadRequestBody(BaseModel):
    num_leads: int  = Field(..., gt=0, le=100)
    business_desc: str = Field(..., min_length=5, max_length=2048)

class LeadResponseBody(BaseModel):
    leads: List[Dict]
    status_code: int
    status_message: str
    process_time: str

# ───────────────────────────────────────── HELPERS

def _find_validator_port() -> Optional[int]:
    """Scan localhost for an open validator HTTP port."""
    common = [VALIDATOR_COMMON_PORT] + list(range(8094, 8105))
    for port in common:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.5)
            if s.connect_ex(("127.0.0.1", port)) == 0:
                return port
    return None

async def _forward_to_validator(payload: dict) -> dict:
    port = _find_validator_port()
    if port is None:
        raise RuntimeError("Validator HTTP server not found on localhost")
    url = f"http://localhost:{port}/api/leads"
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload, timeout=VALIDATOR_TIMEOUT_S) as r:
            if r.status != 200:
                txt = await r.text()
                raise RuntimeError(f"Validator returned {r.status}: {txt[:200]}")
            return await r.json()

# ───────────────────────────────────────── FASTAPI APP

app = FastAPI(title="LeadPoet local proxy", docs_url="/docs")

# Allow browser requests from leadpoet.com (adjust as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://leadpoet.com", "http://localhost:3000", "*"],
    allow_methods=["POST"],
    allow_headers=["*"],
)

@app.post("/lead", response_model=LeadResponseBody)
async def get_leads(body: LeadRequestBody):
    """
    Pass the request to the local validator and return its response verbatim.
    """
    try:
        response = await _forward_to_validator(body.dict())
        return response
    except Exception as e:
        # Map any failure to a 503 so the website can retry / show error
        raise HTTPException(status_code=503, detail=str(e))

# ───────────────────────────────────────── CLI ENTRY

if __name__ == "__main__":
    import uvicorn
    print(f"🔌  LeadPoet proxy listening on http://localhost:{PROXY_PORT}/lead")
    uvicorn.run("local_proxy:app", host="0.0.0.0", port=PROXY_PORT, reload=False)

