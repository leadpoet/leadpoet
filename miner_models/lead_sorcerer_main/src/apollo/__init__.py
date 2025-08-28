"""
Apollo Package for Lead Sorcerer

This package provides comprehensive Apollo API integration including:
- Lead generation and search optimization
- Company and contact enrichment
- Adaptive search strategies
- Result scoring and data mapping
"""

# Re-export main public APIs for backward compatibility
from .client import ApolloClient, create_apollo_client_from_env
from .lead_gen import ApolloLeadGen, ApolloLeadGenError

# Version info
__version__ = "1.0.0"
__all__ = [
    "ApolloClient",
    "create_apollo_client_from_env",
    "ApolloLeadGen",
    "ApolloLeadGenError",
]
