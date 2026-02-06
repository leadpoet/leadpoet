"""
Gateway API Endpoints

This package contains all FastAPI routers for the gateway:
- epoch: Epoch-related endpoints (GET /epoch/{epoch_id}/leads)
- validate: Validation result submission (POST /validate) - IMMEDIATE REVEAL MODE
- manifest: Epoch manifest submission (POST /manifest)
- weights: Weight commit endpoints (POST /weights)

NOTE (Jan 2026): reveal router REMOVED - IMMEDIATE REVEAL MODE means validators
submit both hashes AND actual values in one request to /validate. No separate
reveal phase needed, consensus runs at block 330+ of the same epoch.
"""
