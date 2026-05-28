"""Call the (updated) production ICP generator locally and save 20 fresh ICPs.

Does NOT push to Supabase — just writes a local JSON file we can run qual_engine
against. Lets us verify the generator changes (regional geo + verifiable intents
+ broader products) actually produce broader ICPs.
"""

from __future__ import annotations
import asyncio
import json
import os
import sys
from pathlib import Path

# Make gateway importable
sys.path.insert(0, "/Users/tasnimul/Desktop/leadpoet")

# Load .env for OPENROUTER_API_KEY / OPENROUTER_KEY
env_path = Path("/Users/tasnimul/Desktop/leadpoet/.env")
if env_path.exists():
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip())

# The generator reads OPENROUTER_API_KEY; .env may use OPENROUTER_KEY.
if not os.environ.get("OPENROUTER_API_KEY") and os.environ.get("OPENROUTER_KEY"):
    os.environ["OPENROUTER_API_KEY"] = os.environ["OPENROUTER_KEY"]

# Generator module reads OPENROUTER_API_KEY at IMPORT TIME, so we must
# import AFTER the env var is set.
from gateway.tasks.icp_generator import generate_icps_with_openrouter
import gateway.tasks.icp_generator as _gen_module
# In case the module read it as empty at import time before we set it
_gen_module.OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")

OUT_PATH = "/Users/tasnimul/Desktop/leadpoet/Model_competition/scripts/fresh_icps_20260525.json"
SET_ID = 20260525


async def main():
    print(f"Calling generator (set_id={SET_ID}) ...")
    result = await generate_icps_with_openrouter(set_id=SET_ID, total_icps=20)
    if result is None:
        print("Generator returned None — fallback path would trigger in production.")
        return
    icps, distribution, icp_hash = result
    print(f"Got {len(icps)} ICPs, hash={icp_hash[:16]}...")
    print(f"Distribution: {distribution}\n")
    with open(OUT_PATH, "w") as f:
        json.dump(icps, f, indent=2, default=str)
    print(f"Saved to: {OUT_PATH}")
    # Print summary of broadness
    print(f"\n{'#':3} {'GEO':35} {'STAGE':18} {'SIZE':12} INTENT  PRODUCT")
    for i, icp in enumerate(icps, 1):
        geo = (icp.get("geography") or "")[:35]
        stage = (icp.get("company_stage") or "")[:18]
        size = (icp.get("employee_count") or "")[:12]
        intent = ", ".join(icp.get("intent_signals", [])[:1])[:35]
        prod = (icp.get("product_service") or "")[:40]
        print(f"{i:>3} {geo:35} {stage:18} {size:12} {intent:35} {prod}")


if __name__ == "__main__":
    asyncio.run(main())
