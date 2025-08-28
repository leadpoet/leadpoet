"""
Snovio enrichment provider implementation.

This module implements the Snovio API client as an enrichment provider
following the BaseEnrichmentProvider interface.

Authoritative specifications: BRD §333-336, §410-439
"""

import asyncio
import logging
import os
import time
from typing import Any, Dict, List, Optional

import httpx

from .base import (
    BaseEnrichmentProvider,
    EnrichmentRequest,
    EnrichmentResult,
    ContactData,
    ProviderTier,
    RateLimitError,
    AuthenticationError,
    ServiceUnavailableError,
)


class SnovioProvider(BaseEnrichmentProvider):
    """Snovio API enrichment provider."""

    def __init__(
        self, client_id: Optional[str] = None, client_secret: Optional[str] = None
    ):
        """
        Initialize Snovio provider.

        Args:
            client_id: Snovio client ID. If not provided, will try to load from environment.
            client_secret: Snovio client secret. If not provided, will try to load from environment.
        """
        super().__init__(name="Snovio", tier=ProviderTier.TIER_2)

        # Load credentials from environment if not provided
        self.client_id = client_id or os.getenv("SNOVIO_CLIENT_ID")
        self.client_secret = client_secret or os.getenv("SNOVIO_CLIENT_SECRET")

        if not self.client_id or not self.client_secret:
            raise ValueError(
                "SNOVIO_CLIENT_ID and SNOVIO_CLIENT_SECRET environment variables are required"
            )

        self.base_url = "https://api.snov.io/v2"
        self.connect_timeout = 3.0
        self.read_timeout = 10.0
        self.max_wall_clock = 45.0
        self._access_token = None
        self._token_expires_at = None
        self._last_request_time = 0.0
        self._min_request_interval = (
            1.0  # Minimum 1 second between requests (60 RPM limit)
        )

        logging.info(
            f"🔑 Snovio provider initialized with credentials status: {'SET' if self.client_id and self.client_secret else 'NOT SET'}"
        )

    async def enrich_company(self, request: EnrichmentRequest) -> EnrichmentResult:
        """
        Enrich company information using Snovio.

        Note: Snovio doesn't provide company enrichment service.
        This method raises NotImplementedError.

        Args:
            request: Enrichment request containing domain

        Returns:
            EnrichmentResult with company data or error information

        Raises:
            NotImplementedError: Snovio doesn't support company enrichment
        """
        raise NotImplementedError(
            "Snovio doesn't provide company enrichment service. "
            "Use a dedicated company enrichment provider instead."
        )

    async def enrich_contacts(self, request: EnrichmentRequest) -> EnrichmentResult:
        """
        Enrich company contacts using Snovio.

        Args:
            request: Enrichment request containing domain and optional job title

        Returns:
            EnrichmentResult with list of contact data or error information
        """
        start_time = time.time()

        try:
            token = await self._get_access_token()
            url = f"{self.base_url}/domain-search/prospects/start"

            # Prepare request data
            data = {
                "access_token": token,
                "domain": request.domain,
            }

            if request.job_title:
                # Use 'position' (singular) as the v2 API expects this format
                data["position"] = request.job_title

            async with httpx.AsyncClient(
                timeout=httpx.Timeout(
                    timeout=self.max_wall_clock,
                    connect=self.connect_timeout,
                    read=self.read_timeout,
                )
            ) as client:
                await self._rate_limit()
                response = await client.post(url, data=data)
                await self._handle_snovio_error(response, "contact enrichment")

                data = response.json()
                contacts_data = self._parse_contacts_response(data, request.domain)

                latency_ms = (time.time() - start_time) * 1000
                self.update_metrics(True, cost=0.0, latency_ms=latency_ms)

                return EnrichmentResult(
                    success=True,
                    data=contacts_data,
                    provider=self.name,
                    cost=0.0,
                    latency_ms=latency_ms,
                    metadata={"raw_response": data},
                )

        except Exception as exc:
            await self._handle_generic_error(exc, "contact enrichment")
        raise
        latency_ms = (time.time() - start_time) * 1000
        self.update_metrics(False, cost=0.0, latency_ms=latency_ms)

        return EnrichmentResult(
            success=False,
            error="Contact enrichment failed",
            error_code="ENRICHMENT_FAILED",
            provider=self.name,
            cost=0.0,
            latency_ms=latency_ms,
        )

    async def find_email(self, request: EnrichmentRequest) -> EnrichmentResult:
        """
        Find email address for a specific contact using Snovio.

        Args:
            request: Enrichment request containing domain, first name, and last name

        Returns:
            EnrichmentResult with contact data including email or error information
        """
        start_time = time.time()

        try:
            token = await self._get_access_token()
            url = f"{self.base_url}/search-profile"
            params = {
                "access_token": token,
                "domain": request.domain,
                "firstName": request.first_name,
                "lastName": request.last_name,
            }

            async with httpx.AsyncClient(
                timeout=httpx.Timeout(
                    timeout=self.max_wall_clock,
                    connect=self.connect_timeout,
                    read=self.read_timeout,
                )
            ) as client:
                await self._rate_limit()
                response = await client.get(url, params=params)
                await self._handle_snovio_error(response, "finding email")

                data = response.json()
                contact_data = self._parse_email_response(data, request)

                latency_ms = (time.time() - start_time) * 1000
                self.update_metrics(True, cost=0.0, latency_ms=latency_ms)

                return EnrichmentResult(
                    success=True,
                    data=contact_data,
                    provider=self.name,
                    cost=0.0,
                    latency_ms=latency_ms,
                    metadata={"raw_response": data},
                )

        except Exception as exc:
            await self._handle_generic_error(exc, "finding email")
            raise
        latency_ms = (time.time() - start_time) * 1000
        self.update_metrics(False, cost=0.0, latency_ms=latency_ms)

        return EnrichmentResult(
            success=False,
            error="Email discovery failed",
            error_code="ENRICHMENT_FAILED",
            provider=self.name,
            cost=0.0,
            latency_ms=latency_ms,
        )

    async def validate_email(self, email: str) -> EnrichmentResult:
        """
        Validate email address using Snovio.

        Args:
            email: Email address to validate

        Returns:
            EnrichmentResult with validation status or error information
        """
        start_time = time.time()

        try:
            token = await self._get_access_token()
            url = f"{self.base_url}/verify-email"
            params = {
                "access_token": token,
                "email": email,
            }

            async with httpx.AsyncClient(
                timeout=httpx.Timeout(
                    timeout=self.max_wall_clock,
                    connect=self.connect_timeout,
                    read=self.read_timeout,
                )
            ) as client:
                await self._rate_limit()
                response = await client.get(url, params=params)
                await self._handle_snovio_error(response, "email validation")

                data = response.json()
                validation_result = self._parse_validation_response(data, email)

                latency_ms = (time.time() - start_time) * 1000
                self.update_metrics(True, cost=0.0, latency_ms=latency_ms)

                return EnrichmentResult(
                    success=True,
                    data=validation_result,
                    provider=self.name,
                    cost=0.0,
                    latency_ms=latency_ms,
                    metadata={"raw_response": data},
                )

        except Exception as exc:
            await self._handle_generic_error(exc, "email validation")

        raise
        latency_ms = (time.time() - start_time) * 1000
        self.update_metrics(False, cost=0.0, latency_ms=latency_ms)

        return EnrichmentResult(
            success=False,
            error="Email validation failed",
            error_code="ENRICHMENT_FAILED",
            provider=self.name,
            cost=0.0,
            latency_ms=latency_ms,
        )

    async def _rate_limit(self):
        """Ensure we don't exceed Snovio's 60 RPM rate limit."""
        current_time = time.time()
        time_since_last = current_time - self._last_request_time
        if time_since_last < self._min_request_interval:
            sleep_time = self._min_request_interval - time_since_last
            await asyncio.sleep(sleep_time)
        self._last_request_time = time.time()

    async def _get_access_token(self) -> str:
        """Get access token for Snovio API."""
        if (
            self._access_token
            and self._token_expires_at
            and time.time() < self._token_expires_at
        ):
            return self._access_token

        url = "https://api.snov.io/v1/oauth/access_token"
        data = {
            "grant_type": "client_credentials",
            "client_id": self.client_id,
            "client_secret": self.client_secret,
        }

        async with httpx.AsyncClient(
            timeout=httpx.Timeout(
                timeout=self.max_wall_clock,
                connect=self.connect_timeout,
                read=self.read_timeout,
            )
        ) as client:
            await self._rate_limit()
            response = await client.post(url, data=data)
            await self._handle_snovio_error(response, "getting access token")

            token_data = response.json()
            self._access_token = token_data["access_token"]
            # Snovio tokens typically expire in 1 hour, refresh 5 minutes early
            self._token_expires_at = (
                time.time() + token_data.get("expires_in", 3600) - 300
            )
            return self._access_token

    async def _handle_snovio_error(self, response: httpx.Response, operation: str):
        """Handle Snovio specific error responses."""
        if response.status_code == 200:
            return

        if response.status_code == 402:
            error_msg = (
                f"402 Payment Required - Snovio API credits exhausted for {operation}"
            )
            logging.error(error_msg)
            self.mark_unavailable(error_msg)
            raise ServiceUnavailableError(error_msg, self.name)
        elif response.status_code == 429:
            retry_after = response.headers.get("retry-after", "60")
            error_msg = f"429 Too Many Requests - Rate limit exceeded for {operation}. Retry after {retry_after} seconds"
            logging.error(error_msg)
            self.mark_unavailable(error_msg)
            raise RateLimitError(error_msg, self.name, retry_after=int(retry_after))
        elif response.status_code == 401:
            error_msg = f"401 Unauthorized - Invalid or expired Snovio API token for {operation}"
            logging.error(error_msg)
            self.mark_unavailable(error_msg)
            raise AuthenticationError(error_msg, self.name)
        elif response.status_code == 403:
            error_msg = f"403 Forbidden - Feature not available on current Snovio plan for {operation}"
            logging.error(error_msg)
            self.mark_unavailable(error_msg)
            raise ServiceUnavailableError(error_msg, self.name)
        else:
            response.raise_for_status()

    def _parse_contacts_response(
        self, data: Dict[str, Any], domain: str
    ) -> List[ContactData]:
        """Parse Snovio contacts response into standardized ContactData list."""
        contacts = []
        try:
            prospects = data.get("prospects", [])
            for prospect in prospects:
                contact = ContactData(
                    email=prospect.get("email"),
                    first_name=prospect.get("firstName"),
                    last_name=prospect.get("lastName"),
                    full_name=prospect.get("fullName"),
                    job_title=prospect.get("position"),
                    company_domain=domain,
                    company_name=prospect.get("companyName"),
                    linkedin_url=prospect.get("linkedinUrl"),
                    phone=prospect.get("phone"),
                    location=prospect.get("location"),
                    source=self.name,
                    confidence_score=0.8,  # Default confidence for Snovio
                    last_updated=data.get("updatedAt"),
                )
                contacts.append(contact)
        except Exception as exc:
            logging.warning(f"Failed to parse Snovio contacts response: {exc}")

        return contacts

    def _parse_email_response(
        self, data: Dict[str, Any], request: EnrichmentRequest
    ) -> ContactData:
        """Parse Snovio email search response into standardized ContactData."""
        try:
            profile = data.get("profile", {})
            return ContactData(
                email=profile.get("email"),
                first_name=request.first_name,
                last_name=request.last_name,
                full_name=f"{request.first_name} {request.last_name}".strip(),
                job_title=profile.get("position"),
                company_domain=request.domain,
                company_name=profile.get("companyName"),
                linkedin_url=profile.get("linkedinUrl"),
                phone=profile.get("phone"),
                location=profile.get("location"),
                source=self.name,
                confidence_score=0.8,  # Default confidence for Snovio
                last_updated=data.get("updatedAt"),
            )
        except Exception as exc:
            logging.warning(f"Failed to parse Snovio email response: {exc}")
            return ContactData(
                first_name=request.first_name,
                last_name=request.last_name,
                company_domain=request.domain,
                source=self.name,
            )

    def _parse_validation_response(
        self, data: Dict[str, Any], email: str
    ) -> ContactData:
        """Parse Snovio email validation response into standardized ContactData."""
        try:
            validation = data.get("validation", {})
            return ContactData(
                email=email,
                email_status=validation.get(
                    "status"
                ),  # valid, risky, catch_all, invalid, unknown
                source=self.name,
                confidence_score=validation.get("confidence", 0.8),
                last_updated=data.get("updatedAt"),
            )
        except Exception as exc:
            logging.warning(f"Failed to parse Snovio validation response: {exc}")
            return ContactData(email=email, source=self.name)

    async def _handle_generic_error(self, exc: Exception, operation: str):
        """Handle generic exceptions."""
        error_msg = f"Snovio exception for {operation}: {exc}"
        logging.error(error_msg)

        # Re-raise unexpected exceptions that indicate configuration issues
        if "connection" in str(exc).lower() or "timeout" in str(exc).lower():
            self.mark_unavailable(error_msg)
            raise ServiceUnavailableError(error_msg, self.name)

        self.mark_unavailable(error_msg)
