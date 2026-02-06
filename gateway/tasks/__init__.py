"""
Gateway Background Tasks

This package contains async background tasks that run alongside the FastAPI app:
- epoch_lifecycle: Manages epoch transitions, consensus, and events
- checkpoints: Creates Merkle checkpoints every 10 minutes
- anchor: Anchors Merkle roots on-chain daily
- mirror_monitor: Verifies storage mirror integrity

NOTE (Jan 2026): reveal_collector REMOVED - IMMEDIATE REVEAL MODE means validators
submit both hashes AND actual values in one request. No separate reveal phase to monitor.
Consensus now runs at block 330+ of the same epoch (not next epoch).
"""

