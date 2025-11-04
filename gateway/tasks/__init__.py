"""
Gateway Background Tasks

This package contains async background tasks that run alongside the FastAPI app:
- epoch_lifecycle: Manages epoch transitions and events
- reveal_collector: Monitors and collects validator reveals
- checkpoints: Creates Merkle checkpoints every 10 minutes
- anchor: Anchors Merkle roots on-chain daily
- mirror_monitor: Verifies storage mirror integrity
"""

