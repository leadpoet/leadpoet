import json
import os

POOL_FILE = "lead_pool.json"

def initialize_pool():
    """Initialize the pool file if it doesn’t exist."""
    if not os.path.exists(POOL_FILE):
        with open(POOL_FILE, "w") as f:
            json.dump([], f)

def add_to_pool(leads):
    """Add approved leads to the pool."""
    with open(POOL_FILE, "r+") as f:
        pool = json.load(f)
        pool.extend(leads)
        f.seek(0)
        json.dump(pool, f)

def get_leads_from_pool(num_leads):
    """Retrieve and remove the requested number of leads from the pool."""
    with open(POOL_FILE, "r") as f:
        pool = json.load(f)
    if len(pool) >= num_leads:
        leads = pool[:num_leads]
        with open(POOL_FILE, "w") as f:
            json.dump(pool[num_leads:], f)
        return leads
    else:
        leads = pool
        with open(POOL_FILE, "w") as f:
            json.dump([], f)
        return leads