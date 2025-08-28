"""
Enrichment Provider Router for Lead Sorcerer.

This module implements intelligent routing and fallback logic between enrichment providers
based on their tier assignments and availability. The router ensures high availability
and optimal cost management by automatically falling back to lower-tier providers
when higher-tier ones fail.

Authoritative specifications: BRD §333-336, §410-439
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional

from .enrich_providers.base import (
    BaseEnrichmentProvider,
    EnrichmentRequest,
    EnrichmentResult,
    ProviderTier,
)
from .enrich_providers import (
    CompanyEnrichProvider,
    AnymailFinderProvider,
    CoresignalProvider,
    SnovioProvider,
    ApolloEnrichProvider,
)


class EnrichmentProviderRouter:
    """
    Intelligent router for enrichment providers with automatic fallback logic.

    The router manages provider selection based on:
    - Provider tiers (TIER_0 = primary, TIER_1 = secondary, TIER_2 = legacy)
    - Provider availability and health status
    - Cost optimization and rate limiting
    - Automatic fallback on failures
    """

    def __init__(self, icp_config: Optional[Dict[str, Any]] = None):
        """
        Initialize the enrichment provider router.

        Args:
            icp_config: ICP-specific configuration for provider tiers and preferences
                        (No ICP hard-coding - loaded from config files)
        """
        self.icp_config = icp_config or {}
        self.providers: Dict[ProviderTier, List[BaseEnrichmentProvider]] = {
            ProviderTier.TIER_0: [],
            ProviderTier.TIER_1: [],
            ProviderTier.TIER_2: [],
        }

        # Initialize providers based on configuration
        self._initialize_providers()

        # Track provider health and performance
        self.provider_health: Dict[str, Dict[str, Any]] = {}
        self._update_health_tracking()

        logging.info("Enrichment provider router initialized")

    def _initialize_providers(self):
        """Initialize and categorize enrichment providers by tier."""
        # TIER_0: Primary providers (new, high-quality)
        # Note: Anymail Finder is initialized first for contact/email discovery priority
        if self._should_use_provider("anymail_finder", ProviderTier.TIER_0):
            try:
                self.providers[ProviderTier.TIER_0].append(AnymailFinderProvider())
                logging.info("✅ AnymailFinder provider initialized successfully")
            except Exception as exc:
                logging.warning(f"⚠️ Failed to initialize AnymailFinder provider: {exc}")

        if self._should_use_provider("company_enrich", ProviderTier.TIER_0):
            try:
                self.providers[ProviderTier.TIER_0].append(CompanyEnrichProvider())
                logging.info("✅ CompanyEnrich provider initialized successfully")
            except Exception as exc:
                logging.warning(f"⚠️ Failed to initialize CompanyEnrich provider: {exc}")

        # Apollo provider - only initialize if Apollo mode is enabled
        if self._should_use_apollo_provider():
            try:
                apollo_config = self.icp_config.get("apollo", {}).get("enrich", {})
                self.providers[ProviderTier.TIER_0].append(
                    ApolloEnrichProvider(apollo_config)
                )
                logging.info("✅ Apollo enrichment provider initialized successfully")
            except Exception as exc:
                logging.warning(
                    f"⚠️ Failed to initialize Apollo enrichment provider: {exc}"
                )

        # TIER_1: Secondary providers (legacy, reliable)
        if self._should_use_provider("coresignal", ProviderTier.TIER_1):
            try:
                self.providers[ProviderTier.TIER_1].append(CoresignalProvider())
                logging.info("✅ Coresignal provider initialized successfully")
            except Exception as exc:
                logging.warning(f"⚠️ Failed to initialize Coresignal provider: {exc}")

        if self._should_use_provider("snovio", ProviderTier.TIER_1):
            try:
                self.providers[ProviderTier.TIER_1].append(SnovioProvider())
                logging.info("✅ Snov.io provider initialized successfully")
            except Exception as exc:
                logging.warning(f"⚠️ Failed to initialize Snov.io provider: {exc}")

        # TIER_2: Legacy providers (last resort)
        # Currently no TIER_2 providers, but structure is ready for future use

        logging.info(
            f"Router initialized with {len(self.providers[ProviderTier.TIER_0])} TIER_0, "
            f"{len(self.providers[ProviderTier.TIER_1])} TIER_1 providers"
        )

        # Ensure we have at least one provider available
        if not any(self.providers.values()):
            logging.warning("No providers available, falling back to Coresignal")
            try:
                self.providers[ProviderTier.TIER_1].append(CoresignalProvider())
                logging.info("✅ Coresignal fallback provider initialized successfully")
            except Exception as exc:
                logging.error(
                    f"❌ Failed to initialize Coresignal fallback provider: {exc}"
                )
                logging.error(
                    "❌ No enrichment providers available - enrichment will fail"
                )

    def _should_use_provider(self, provider_name: str, tier: ProviderTier) -> bool:
        """
        Determine if a provider should be used based on ICP configuration.

        Args:
            provider_name: Name of the provider to check
            tier: Tier level for the provider

        Returns:
            True if provider should be used, False otherwise
        """
        # Check ICP-specific provider preferences
        provider_config = self.icp_config.get("enrichment_providers", {})
        provider_settings = provider_config.get(provider_name, {})

        # Default to enabled unless explicitly disabled
        if "enabled" in provider_settings:
            return provider_settings["enabled"]

        # Check tier-specific settings
        tier_config = provider_config.get(f"tier_{tier.value}", {})
        if "enabled" in tier_config:
            return tier_config["enabled"]

        # Default behavior based on tier
        if tier == ProviderTier.TIER_0:
            return True  # Primary providers enabled by default
        elif tier == ProviderTier.TIER_1:
            return True  # Secondary providers enabled by default
        else:
            return False  # Legacy providers disabled by default

    def _should_use_apollo_provider(self) -> bool:
        """
        Determine if Apollo enrichment provider should be used.

        Apollo provider is only enabled when:
        1. Apollo mode is enabled in ICP config
        2. Apollo enrichment is enabled in Apollo config

        Returns:
            True if Apollo provider should be used, False otherwise
        """
        # Check if Apollo mode is enabled
        lead_generation_mode = self.icp_config.get(
            "lead_generation_mode", "traditional"
        )
        if lead_generation_mode != "apollo":
            return False

        # Check if Apollo config exists and enrichment is enabled
        apollo_config = self.icp_config.get("apollo", {})
        if not apollo_config:
            return False

        # Check if Apollo enrichment is enabled
        enrich_config = apollo_config.get("enrich", {})
        return enrich_config.get("enabled", False)

    def _update_health_tracking(self):
        """Initialize health tracking for all providers."""
        for tier in self.providers:
            for provider in self.providers[tier]:
                self.provider_health[provider.name] = {
                    "is_available": True,
                    "error_count": 0,
                    "success_count": 0,
                    "last_error": None,
                    "last_success": None,
                    "average_latency": 0.0,
                    "total_cost": 0.0,
                }

    async def enrich_company(self, request: EnrichmentRequest) -> EnrichmentResult:
        """
        Enrich company information using provider fallback logic.

        Args:
            request: Enrichment request with domain

        Returns:
            Enrichment result with company data from best available provider
        """
        return await self._execute_with_fallback(
            "enrich_company", request, "company enrichment"
        )

    async def enrich_contacts(self, request: EnrichmentRequest) -> EnrichmentResult:
        """
        Enrich company contacts using provider fallback logic.

        Args:
            request: Enrichment request with domain

        Returns:
            Enrichment result with contact data from best available provider
        """
        return await self._execute_with_fallback(
            "enrich_contacts", request, "contact enrichment"
        )

    async def find_email(self, request: EnrichmentRequest) -> EnrichmentResult:
        """
        Find email address using provider fallback logic.

        Args:
            request: Enrichment request with domain, first_name, last_name

        Returns:
            Enrichment result with email data from best available provider
        """
        return await self._execute_with_fallback(
            "find_email", request, "email discovery"
        )

    async def get_all_domain_prospects(
        self, request: EnrichmentRequest
    ) -> EnrichmentResult:
        """
        Get all domain prospects using provider fallback logic.

        Args:
            request: Enrichment request with domain

        Returns:
            Enrichment result with prospects data from best available provider
        """
        return await self._execute_with_fallback(
            "get_all_domain_prospects", request, "prospect discovery"
        )

    async def search_prospect_email(
        self, request: EnrichmentRequest
    ) -> EnrichmentResult:
        """
        Search for prospect email using provider fallback logic.

        Args:
            request: Enrichment request with prospect_hash

        Returns:
            Enrichment result with email data from best available provider
        """
        return await self._execute_with_fallback(
            "search_prospect_email", request, "prospect email search"
        )

    async def _execute_with_fallback(
        self, method_name: str, request: EnrichmentRequest, operation_description: str
    ) -> EnrichmentResult:
        """
        Execute enrichment operation with automatic provider fallback.

        Args:
            method_name: Name of the method to call on providers
            request: Enrichment request
            operation_description: Human-readable description of the operation

        Returns:
            Enrichment result from the first successful provider
        """
        errors = []

        # Try TIER_0 providers first (primary)
        for provider in self.providers[ProviderTier.TIER_0]:
            if not self._is_provider_healthy(provider):
                continue

            try:
                result = await getattr(provider, method_name)(request)
                if result.success:
                    self._record_success(provider, result)
                    return result
                else:
                    self._record_error(provider, result.error)
                    errors.append(f"{provider.name}: {result.error}")
            except Exception as exc:
                self._record_error(provider, str(exc))
                errors.append(f"{provider.name}: {exc}")

        # Try TIER_1 providers (secondary/legacy)
        for provider in self.providers[ProviderTier.TIER_1]:
            if not self._is_provider_healthy(provider):
                continue

            try:
                result = await getattr(provider, method_name)(request)
                if result.success:
                    self._record_success(provider, result)
                    return result
                else:
                    self._record_error(provider, result.error)
                    errors.append(f"{provider.name}: {result.error}")
            except Exception as exc:
                self._record_error(provider, str(exc))
                errors.append(f"{provider.name}: {exc}")

        # Try TIER_2 providers (last resort)
        for provider in self.providers[ProviderTier.TIER_2]:
            if not self._is_provider_healthy(provider):
                continue

            try:
                result = await getattr(provider, method_name)(request)
                if result.success:
                    self._record_success(provider, result)
                    return result
                else:
                    self._record_error(provider, result.error)
                    errors.append(f"{provider.name}: {result.error}")
            except Exception as exc:
                self._record_error(provider, str(exc))
                errors.append(f"{provider.name}: {exc}")

        # All providers failed
        error_msg = f"All enrichment providers failed for {operation_description}: {'; '.join(errors)}"
        logging.error(error_msg)

        return EnrichmentResult(
            success=False, error=error_msg, provider="router", cost=0.0
        )

    def _is_provider_healthy(self, provider: BaseEnrichmentProvider) -> bool:
        """
        Check if a provider is healthy and available.

        Args:
            provider: Provider to check

        Returns:
            True if provider is healthy, False otherwise
        """
        health = self.provider_health.get(provider.name, {})

        # Check if provider is marked as unavailable
        if not health.get("is_available", True):
            return False

        # Check error threshold (mark as unavailable if too many errors)
        error_count = health.get("error_count", 0)
        success_count = health.get("success_count", 0)
        total_attempts = error_count + success_count

        if total_attempts > 10 and error_count / total_attempts > 0.8:
            self._mark_provider_unavailable(provider, "High error rate")
            return False

        return True

    def _record_success(
        self, provider: BaseEnrichmentProvider, result: EnrichmentResult
    ):
        """Record successful operation for a provider."""
        if provider.name not in self.provider_health:
            self.provider_health[provider.name] = {}

        health = self.provider_health[provider.name]
        health["success_count"] = health.get("success_count", 0) + 1
        health["last_success"] = asyncio.get_event_loop().time()
        health["is_available"] = True

        # Update cost tracking
        if result.cost:
            health["total_cost"] = health.get("total_cost", 0.0) + result.cost

    def _record_error(self, provider: BaseEnrichmentProvider, error: str):
        """Record failed operation for a provider."""
        if provider.name not in self.provider_health:
            self.provider_health[provider.name] = {}

        health = self.provider_health[provider.name]
        health["error_count"] = health.get("error_count", 0) + 1
        health["last_error"] = error

    def _mark_provider_unavailable(self, provider: BaseEnrichmentProvider, reason: str):
        """Mark a provider as temporarily unavailable."""
        if provider.name not in self.provider_health:
            self.provider_health[provider.name] = {}

        health = self.provider_health[provider.name]
        health["is_available"] = False
        health["last_error"] = f"Marked unavailable: {reason}"

        logging.warning(f"Provider {provider.name} marked as unavailable: {reason}")

    def get_provider_status(self) -> Dict[str, Any]:
        """
        Get current status of all providers.

        Returns:
            Dictionary with provider health and performance metrics
        """
        status = {
            "tier_0_providers": [],
            "tier_1_providers": [],
            "tier_2_providers": [],
            "overall_health": "healthy",
        }

        # Collect provider status by tier
        for tier in [ProviderTier.TIER_0, ProviderTier.TIER_1, ProviderTier.TIER_2]:
            tier_name = f"tier_{tier.value}_providers"
            for provider in self.providers[tier]:
                provider_status = {
                    "name": provider.name,
                    "tier": tier.value,
                    "is_available": self.provider_health.get(provider.name, {}).get(
                        "is_available", True
                    ),
                    "success_count": self.provider_health.get(provider.name, {}).get(
                        "success_count", 0
                    ),
                    "error_count": self.provider_health.get(provider.name, {}).get(
                        "error_count", 0
                    ),
                    "total_cost": self.provider_health.get(provider.name, {}).get(
                        "total_cost", 0.0
                    ),
                    "last_error": self.provider_health.get(provider.name, {}).get(
                        "last_error"
                    ),
                }
                status[tier_name].append(provider_status)

        # Determine overall health
        total_providers = sum(len(providers) for providers in self.providers.values())
        available_providers = sum(
            len([p for p in providers if self._is_provider_healthy(p)])
            for providers in self.providers.values()
        )

        if available_providers == 0:
            status["overall_health"] = "critical"
        elif available_providers < total_providers * 0.5:
            status["overall_health"] = "degraded"
        elif available_providers < total_providers:
            status["overall_health"] = "warning"
        else:
            status["overall_health"] = "healthy"

        return status

    def reset_provider_health(self, provider_name: Optional[str] = None):
        """
        Reset health tracking for providers.

        Args:
            provider_name: Specific provider to reset, or None for all providers
        """
        if provider_name:
            if provider_name in self.provider_health:
                self.provider_health[provider_name] = {
                    "is_available": True,
                    "error_count": 0,
                    "success_count": 0,
                    "last_error": None,
                    "last_success": None,
                    "average_latency": 0.0,
                    "total_cost": 0.0,
                }
                logging.info(f"Reset health tracking for provider: {provider_name}")
        else:
            self._update_health_tracking()
            logging.info("Reset health tracking for all providers")


# Factory function for creating router instances
def create_enrichment_router(
    icp_config: Optional[Dict[str, Any]] = None,
) -> EnrichmentProviderRouter:
    """
    Factory function to create enrichment provider router instances.

    Args:
        icp_config: ICP-specific configuration for provider tiers and preferences

    Returns:
        Configured EnrichmentProviderRouter instance
    """
    return EnrichmentProviderRouter(icp_config)
