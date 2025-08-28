"""
Base classes and interfaces for enrichment providers.

This module defines the common interface that all enrichment providers must implement,
along with typed exceptions for error handling and provider-specific error types.

Authoritative specifications: BRD §333-336, §410-439
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass
from enum import Enum


class ProviderTier(Enum):
    """Provider tier for fallback logic."""

    TIER_0 = 0  # Primary providers (highest priority)
    TIER_1 = 1  # Secondary providers (fallback)
    TIER_2 = 2  # Legacy providers (last resort)


@dataclass
class EnrichmentRequest:
    """Base request structure for enrichment operations."""

    domain: str
    company_name: Optional[str] = None
    linkedin_url: Optional[str] = None
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    job_title: Optional[str] = None


@dataclass
class CompanyData:
    """Standardized company data structure."""

    domain: str
    name: Optional[str] = None
    description: Optional[str] = None
    industry: Optional[str] = None
    size: Optional[str] = None
    founded_year: Optional[int] = None
    revenue: Optional[str] = None
    website: Optional[str] = None
    linkedin_url: Optional[str] = None
    twitter_url: Optional[str] = None
    facebook_url: Optional[str] = None
    phone: Optional[str] = None
    address: Optional[Dict[str, str]] = None
    technologies: Optional[List[str]] = None
    social_media_presence: Optional[Dict[str, str]] = None
    employee_count: Optional[int] = None
    funding: Optional[Dict[str, Any]] = None
    competitors: Optional[List[str]] = None
    tags: Optional[List[str]] = None
    source: Optional[str] = None
    confidence_score: Optional[float] = None
    last_updated: Optional[str] = None


@dataclass
class ContactData:
    """Standardized contact data structure."""

    email: Optional[str] = None
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    full_name: Optional[str] = None
    job_title: Optional[str] = None
    department: Optional[str] = None
    seniority_level: Optional[str] = None
    company_domain: Optional[str] = None
    company_name: Optional[str] = None
    linkedin_url: Optional[str] = None
    phone: Optional[str] = None
    location: Optional[str] = None
    social_media: Optional[Dict[str, str]] = None
    skills: Optional[List[str]] = None
    experience_years: Optional[int] = None
    education: Optional[List[str]] = None
    source: Optional[str] = None
    confidence_score: Optional[float] = None
    last_updated: Optional[str] = None
    email_status: Optional[str] = None  # valid, risky, catch_all, invalid, unknown


@dataclass
class EnrichmentResult:
    """Standardized result structure for enrichment operations."""

    success: bool
    data: Optional[Union[CompanyData, ContactData, List[ContactData]]] = None
    error: Optional[str] = None
    error_code: Optional[str] = None
    provider: Optional[str] = None
    cost: Optional[float] = None
    latency_ms: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None


# ============================================================================
# Typed Exceptions
# ============================================================================


class EnrichmentProviderError(Exception):
    """Base exception for all enrichment provider errors."""

    def __init__(
        self,
        message: str,
        provider: Optional[str] = None,
        error_code: Optional[str] = None,
    ):
        self.message = message
        self.provider = provider
        self.error_code = error_code
        super().__init__(self.message)


class CompanyEnrichmentError(EnrichmentProviderError):
    """Exception raised when company enrichment fails."""

    pass


class ContactEnrichmentError(EnrichmentProviderError):
    """Exception raised when contact enrichment fails."""

    pass


class RateLimitError(EnrichmentProviderError):
    """Exception raised when API rate limit is exceeded."""

    def __init__(
        self,
        message: str,
        provider: Optional[str] = None,
        retry_after: Optional[int] = None,
    ):
        self.retry_after = retry_after
        super().__init__(message, provider, "RATE_LIMIT")


class AuthenticationError(EnrichmentProviderError):
    """Exception raised when API authentication fails."""

    def __init__(self, message: str, provider: Optional[str] = None):
        super().__init__(message, provider, "AUTHENTICATION")


class ConfigurationError(EnrichmentProviderError):
    """Exception raised when provider configuration is invalid."""

    def __init__(self, message: str, provider: Optional[str] = None):
        super().__init__(message, provider, "CONFIGURATION")


class DataValidationError(EnrichmentProviderError):
    """Exception raised when input data validation fails."""

    def __init__(
        self, message: str, provider: Optional[str] = None, field: Optional[str] = None
    ):
        self.field = field
        super().__init__(message, provider, "VALIDATION")


class ServiceUnavailableError(EnrichmentProviderError):
    """Exception raised when the enrichment service is unavailable."""

    def __init__(self, message: str, provider: Optional[str] = None):
        super().__init__(message, provider, "SERVICE_UNAVAILABLE")


# ============================================================================
# Base Provider Interface
# ============================================================================


class BaseEnrichmentProvider(ABC):
    """
    Abstract base class for all enrichment providers.

    All enrichment providers must implement this interface to ensure
    consistent behavior and enable the provider router to work correctly.
    """

    def __init__(self, name: str, tier: ProviderTier = ProviderTier.TIER_1):
        self.name = name
        self.tier = tier
        self.is_available = True
        self.last_error = None
        self.error_count = 0
        self.success_count = 0
        self.total_cost = 0.0
        self.total_latency = 0.0

    @abstractmethod
    async def enrich_company(self, request: EnrichmentRequest) -> EnrichmentResult:
        """
        Enrich company information.

        Args:
            request: Enrichment request containing domain and optional company name

        Returns:
            EnrichmentResult with company data or error information

        Raises:
            CompanyEnrichmentError: When company enrichment fails
        """
        pass

    @abstractmethod
    async def enrich_contacts(self, request: EnrichmentRequest) -> EnrichmentResult:
        """
        Enrich company contacts.

        Args:
            request: Enrichment request containing domain and optional filters

        Returns:
            EnrichmentResult with list of contact data or error information

        Raises:
            ContactEnrichmentError: When contact enrichment fails
        """
        pass

    @abstractmethod
    async def find_email(self, request: EnrichmentRequest) -> EnrichmentResult:
        """
        Find email address for a specific contact.

        Args:
            request: Enrichment request containing domain, first name, and last name

        Returns:
            EnrichmentResult with contact data including email or error information

        Raises:
            ContactEnrichmentError: When email discovery fails
        """
        pass

    @abstractmethod
    async def validate_email(self, email: str) -> EnrichmentResult:
        """
        Validate email address.

        Args:
            email: Email address to validate

        Returns:
            EnrichmentResult with validation status or error information

        Raises:
            ContactEnrichmentError: When email validation fails
        """
        pass

    def get_provider_info(self) -> Dict[str, Any]:
        """
        Get provider information and statistics.

        Returns:
            Dictionary containing provider metadata and performance metrics
        """
        return {
            "name": self.name,
            "tier": self.tier.value,
            "is_available": self.is_available,
            "last_error": self.last_error,
            "error_count": self.error_count,
            "success_count": self.success_count,
            "total_cost": self.total_cost,
            "total_latency": self.total_latency,
            "success_rate": (
                self.success_count / (self.success_count + self.error_count)
                if (self.success_count + self.error_count) > 0
                else 0.0
            ),
            "average_latency": (
                self.total_latency / self.success_count
                if self.success_count > 0
                else 0.0
            ),
        }

    def update_metrics(self, success: bool, cost: float = 0.0, latency_ms: float = 0.0):
        """
        Update provider performance metrics.

        Args:
            success: Whether the operation was successful
            cost: Cost of the operation in USD
            latency_ms: Operation latency in milliseconds
        """
        if success:
            self.success_count += 1
            self.total_cost += cost
            self.total_latency += latency_ms
            self.last_error = None
        else:
            self.error_count += 1

    def mark_unavailable(self, error: str):
        """
        Mark provider as temporarily unavailable.

        Args:
            error: Error message describing why the provider is unavailable
        """
        self.is_available = False
        self.last_error = error

    def mark_available(self):
        """Mark provider as available again."""
        self.is_available = True
        self.last_error = None

    def reset_metrics(self):
        """Reset all performance metrics."""
        self.error_count = 0
        self.success_count = 0
        self.total_cost = 0.0
        self.total_latency = 0.0
        self.last_error = None

    def __str__(self) -> str:
        return f"{self.name} (Tier {self.tier.value})"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', tier={self.tier})"
