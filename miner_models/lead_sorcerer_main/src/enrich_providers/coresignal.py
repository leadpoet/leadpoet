"""
Coresignal enrichment provider implementation.

This module implements the Coresignal API client as an enrichment provider
following the BaseEnrichmentProvider interface.

Authoritative specifications: BRD §333-336, §410-439
"""

import logging
import os
import time
from typing import Any, Dict, List, Optional

import httpx

from .base import (
    BaseEnrichmentProvider,
    EnrichmentRequest,
    EnrichmentResult,
    CompanyData,
    ContactData,
    ProviderTier,
    RateLimitError,
    AuthenticationError,
    ServiceUnavailableError,
)


class CoresignalProvider(BaseEnrichmentProvider):
    """Coresignal API enrichment provider."""

    def __init__(self, api_token: Optional[str] = None):
        """
        Initialize Coresignal provider.

        Args:
            api_token: Coresignal API token. If not provided, will try to load from environment.
        """
        super().__init__(name="Coresignal", tier=ProviderTier.TIER_1)

        # Load API token from environment if not provided
        self.api_token = api_token or os.getenv("CORESIGNAL_API_TOKEN")
        if not self.api_token:
            raise ValueError("CORESIGNAL_API_TOKEN environment variable is required")

        self.base_url = "https://api.coresignal.com"
        self.connect_timeout = 3.0
        self.read_timeout = 10.0
        self.max_wall_clock = 45.0

        logging.info(
            f"🔑 Coresignal provider initialized with token status: {'SET' if self.api_token else 'NOT SET'}"
        )

    async def enrich_company(self, request: EnrichmentRequest) -> EnrichmentResult:
        """
        Enrich company information using Coresignal.

        Args:
            request: Enrichment request containing domain

        Returns:
            EnrichmentResult with company data or error information
        """
        start_time = time.time()

        try:
            # Prepare website parameter
            website_param = request.domain.strip()
            if not website_param.startswith(("http://", "https://")):
                website_param = f"https://{website_param}"

            url = f"{self.base_url}/cdapi/v2/company_clean/enrich"
            params = {"website": website_param}
            headers = {
                "accept": "application/json",
                "apikey": self.api_token,
            }

            async with httpx.AsyncClient(timeout=30.0) as client:
                logging.debug(
                    f"Coresignal: Calling API: {url} with website={website_param}"
                )
                response = await client.get(url, params=params, headers=headers)

                if response.status_code == 200:
                    data = response.json()
                    company_data = self._parse_company_response(data, request.domain)

                    latency_ms = (time.time() - start_time) * 1000
                    self.update_metrics(True, cost=0.0, latency_ms=latency_ms)

                    return EnrichmentResult(
                        success=True,
                        data=company_data,
                        provider=self.name,
                        cost=0.0,
                        latency_ms=latency_ms,
                        metadata={"raw_response": data},
                    )
                else:
                    await self._handle_error_response(response, "company enrichment")

        except httpx.HTTPStatusError as exc:
            await self._handle_http_error(exc, "company enrichment")
        except Exception as exc:
            await self._handle_generic_error(exc, "company enrichment")

        latency_ms = (time.time() - start_time) * 1000
        self.update_metrics(False, cost=0.0, latency_ms=latency_ms)

        return EnrichmentResult(
            success=False,
            error="Company enrichment failed",
            error_code="ENRICHMENT_FAILED",
            provider=self.name,
            cost=0.0,
            latency_ms=latency_ms,
        )

    async def enrich_contacts(self, request: EnrichmentRequest) -> EnrichmentResult:
        """
        Enrich company contacts using Coresignal.

        Args:
            request: Enrichment request containing domain

        Returns:
            EnrichmentResult with list of contact data or error information
        """
        start_time = time.time()

        try:
            url = f"{self.base_url}/v2/people/by-company-domain/{request.domain}"
            headers = {"apikey": self.api_token}

            async with httpx.AsyncClient(
                timeout=httpx.Timeout(
                    timeout=self.max_wall_clock,
                    connect=self.connect_timeout,
                    read=self.read_timeout,
                )
            ) as client:
                response = await client.get(url, headers=headers)

                if response.status_code == 200:
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
                else:
                    await self._handle_error_response(response, "contact enrichment")

        except httpx.HTTPStatusError as exc:
            await self._handle_http_error(exc, "contact enrichment")
        except Exception as exc:
            await self._handle_generic_error(exc, "contact enrichment")

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
        Find email address for a specific contact.

        Note: Coresignal doesn't support direct LinkedIn URL enrichment.
        This method raises NotImplementedError for backward compatibility.

        Args:
            request: Enrichment request containing domain, first name, and last name

        Returns:
            EnrichmentResult with contact data including email or error information

        Raises:
            NotImplementedError: Coresignal doesn't support this operation
        """
        raise NotImplementedError(
            "Coresignal doesn't support direct LinkedIn URL enrichment. "
            "Use enrich_contacts() method instead."
        )

    async def validate_email(self, email: str) -> EnrichmentResult:
        """
        Validate email address.

        Note: Coresignal doesn't provide email validation service.
        This method raises NotImplementedError.

        Args:
            email: Email address to validate

        Returns:
            EnrichmentResult with validation status or error information

        Raises:
            NotImplementedError: Coresignal doesn't support email validation
        """
        raise NotImplementedError(
            "Coresignal doesn't provide email validation service. "
            "Use a dedicated email validation provider instead."
        )

    def _parse_company_response(self, data: Dict[str, Any], domain: str) -> CompanyData:
        """Parse Coresignal company response into standardized CompanyData."""
        try:
            return CompanyData(
                domain=domain,
                name=data.get("name"),
                description=data.get("description"),
                industry=data.get("industry"),
                size=data.get("size"),
                founded_year=data.get("founded_year"),
                revenue=data.get("revenue"),
                website=data.get("website"),
                linkedin_url=data.get("linkedin_url"),
                phone=data.get("phone"),
                address={
                    "street": data.get("address", {}).get("street"),
                    "city": data.get("address", {}).get("city"),
                    "state": data.get("address", {}).get("state"),
                    "country": data.get("address", {}).get("country"),
                    "zip": data.get("address", {}).get("zip"),
                }
                if data.get("address")
                else None,
                employee_count=data.get("employee_count"),
                source=self.name,
                confidence_score=data.get("confidence_score", 0.8),
                last_updated=data.get("updated_at"),
            )
        except Exception as exc:
            logging.warning(f"Failed to parse Coresignal company response: {exc}")
            return CompanyData(domain=domain, source=self.name)

    def _parse_contacts_response(
        self, data: Dict[str, Any], domain: str
    ) -> List[ContactData]:
        """Parse Coresignal contacts response into standardized ContactData list."""
        contacts = []
        try:
            people = data.get("people", [])
            for person in people:
                contact = ContactData(
                    email=person.get("email"),
                    first_name=person.get("first_name"),
                    last_name=person.get("last_name"),
                    full_name=person.get("full_name"),
                    job_title=person.get("job_title"),
                    department=person.get("department"),
                    seniority_level=person.get("seniority_level"),
                    company_domain=domain,
                    company_name=person.get("company_name"),
                    linkedin_url=person.get("linkedin_url"),
                    phone=person.get("phone"),
                    location=person.get("location"),
                    source=self.name,
                    confidence_score=person.get("confidence_score", 0.8),
                    last_updated=person.get("updated_at"),
                )
                contacts.append(contact)
        except Exception as exc:
            logging.warning(f"Failed to parse Coresignal contacts response: {exc}")

        return contacts

    async def _handle_error_response(self, response: httpx.Response, operation: str):
        """Handle HTTP error responses from Coresignal API."""
        if response.status_code == 404:
            logging.warning(
                f"Coresignal: {operation} not found (404) - this is normal for smaller companies"
            )
        elif response.status_code == 401:
            error_msg = f"Coresignal API authentication failed for {operation}"
            logging.error(error_msg)
            self.mark_unavailable(error_msg)
            raise AuthenticationError(error_msg, self.name)
        elif response.status_code == 429:
            error_msg = f"Coresignal API rate limit exceeded for {operation}"
            logging.error(error_msg)
            self.mark_unavailable(error_msg)
            raise RateLimitError(error_msg, self.name)
        else:
            error_msg = f"Coresignal API error for {operation}: {response.status_code} - {response.text}"
            logging.error(error_msg)
            self.mark_unavailable(error_msg)

    async def _handle_http_error(self, exc: httpx.HTTPStatusError, operation: str):
        """Handle httpx HTTPStatusError exceptions."""
        error_msg = f"Coresignal HTTP error for {operation}: {exc.response.status_code}"
        logging.error(error_msg)
        self.mark_unavailable(error_msg)
        raise ServiceUnavailableError(error_msg, self.name)

    async def _handle_generic_error(self, exc: Exception, operation: str):
        """Handle generic exceptions."""
        error_msg = f"Coresignal exception for {operation}: {exc}"
        logging.error(error_msg)

        # Re-raise unexpected exceptions that indicate configuration issues
        if "connection" in str(exc).lower() or "timeout" in str(exc).lower():
            self.mark_unavailable(error_msg)
            raise ServiceUnavailableError(error_msg, self.name)

        self.mark_unavailable(error_msg)
