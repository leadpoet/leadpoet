"""
Enrichment Providers Package for Lead Sorcerer.

This package contains all enrichment provider implementations including:
- Legacy providers (Coresignal, Snovio)
- New providers (CompanyEnrich, Anymail Finder, Apollo)
- Base interfaces and typed exceptions
- Provider router for fallback logic

Authoritative specifications: BRD §333-336, §410-439
"""

from .base import (
    BaseEnrichmentProvider,
    EnrichmentProviderError,
    CompanyEnrichmentError,
    ContactEnrichmentError,
    RateLimitError,
    AuthenticationError,
    ConfigurationError,
)

from .coresignal import CoresignalProvider
from .snovio import SnovioProvider
from .company_enrich import CompanyEnrichProvider
from .anymail_finder import AnymailFinderProvider
from .apollo import ApolloEnrichProvider

__all__ = [
    # Base classes and exceptions
    "BaseEnrichmentProvider",
    "EnrichmentProviderError",
    "CompanyEnrichmentError",
    "ContactEnrichmentError",
    "RateLimitError",
    "AuthenticationError",
    "ConfigurationError",
    # Legacy providers
    "CoresignalProvider",
    "SnovioProvider",
    # New primary providers
    "CompanyEnrichProvider",
    "AnymailFinderProvider",
    "ApolloEnrichProvider",
]

__version__ = "1.0.0"
