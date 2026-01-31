"""
Gateway Qualification Module

This module contains the qualification API endpoints and shared utilities
that run on the gateway. The validator-specific code (sandbox, scoring, etc.)
remains in the top-level qualification/ folder.

Structure:
- api/       - FastAPI endpoints for model submission, status, work distribution
- utils/     - Shared utilities (chain interaction, helpers)
- config.py  - Configuration
- models.py  - Pydantic models
"""

# Minimal imports to avoid circular dependencies
# Import specific items only when needed in code

__all__ = [
    "CONFIG",
]
