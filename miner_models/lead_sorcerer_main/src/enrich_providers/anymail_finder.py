"""
Anymail Finder enrichment provider implementation.

This module implements the Anymail Finder API client as an enrichment provider
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
    CompanyData,
    ContactData,
    ProviderTier,
    CompanyEnrichmentError,
    RateLimitError,
    AuthenticationError,
    ServiceUnavailableError,
)


class AnymailFinderProvider(BaseEnrichmentProvider):
    """Anymail Finder API enrichment provider."""

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Anymail Finder provider.

        Args:
            api_key: Anymail Finder API key (defaults to env var ANYMAIL_FINDER_API_KEY)
        """
        super().__init__(name="AnymailFinder", tier=ProviderTier.TIER_0)
        self.api_key = api_key or os.getenv("ANYMAIL_FINDER_API_KEY")
        if not self.api_key:
            raise AuthenticationError("Anymail Finder API key not provided")

        self.base_url = "https://api.anymailfinder.com/v5.1"
        self._last_request_time = 0.0
        self._min_request_interval = 0.2  # 200ms between requests (5 RPS limit)

        logging.info("Anymail Finder provider initialized")

    async def _rate_limit(self):
        """Ensure we don't exceed Anymail Finder's rate limits."""
        current_time = time.time()
        time_since_last = current_time - self._last_request_time
        if time_since_last < self._min_request_interval:
            sleep_time = self._min_request_interval - time_since_last
            await asyncio.sleep(sleep_time)
        self._last_request_time = time.time()

    async def _handle_response(
        self, response: httpx.Response, operation: str
    ) -> Dict[str, Any]:
        """
        Handle API response and extract data.

        Args:
            response: HTTP response from Anymail Finder API
            operation: Description of the operation being performed

        Returns:
            Extracted data from response

        Raises:
            RateLimitError: If rate limit exceeded
            AuthenticationError: If authentication failed
            ServiceUnavailableError: If service is unavailable
            CompanyEnrichmentError: For other API errors
        """
        if response.status_code == 200:
            try:
                data = response.json()
                # Ensure we always return a dictionary
                if not isinstance(data, dict):
                    logging.warning(
                        f"Anymail Finder API returned non-dict response for {operation}: {type(data)}"
                    )
                    return {}
                return data
            except Exception as exc:
                logging.error(
                    f"Anymail Finder API response parsing failed for {operation}: {exc}"
                )
                return {}
        elif response.status_code == 401:
            logging.error(
                f"Anymail Finder 401 response for {operation}: {response.text}"
            )
            raise AuthenticationError(
                f"Anymail Finder API authentication failed for {operation}"
            )
        elif response.status_code == 429:
            retry_after = response.headers.get("retry-after", "60")
            raise RateLimitError(
                f"Anymail Finder API rate limit exceeded for {operation}. Retry after {retry_after} seconds"
            )
        elif response.status_code == 503:
            raise ServiceUnavailableError(
                f"Anymail Finder API service unavailable for {operation}"
            )
        else:
            logging.error(
                f"Anymail Finder {response.status_code} response for {operation}: {response.text}"
            )
            error_msg = f"Anymail Finder API error {response.status_code} for {operation}: {response.text}"
            raise CompanyEnrichmentError(error_msg)

    async def enrich_company(self, request: EnrichmentRequest) -> EnrichmentResult:
        """
        Enrich company information using Anymail Finder.

        Args:
            request: Enrichment request with domain

        Returns:
            Enrichment result with company data
        """
        domain = request.domain
        try:
            await self._rate_limit()

            # Anymail Finder doesn't have a company search endpoint
            # We'll use the decision-maker endpoint to find company contacts
            url = f"{self.base_url}/find-email/decision-maker"
            payload = {"domain": domain, "decision_maker_category": "ceo"}
            headers = {
                "Authorization": self.api_key,
                "Content-Type": "application/json",
            }

            logging.info(
                f"Anymail Finder company enrichment request: {url} with payload: {payload}"
            )

            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(url, json=payload, headers=headers)
                data = await self._handle_response(response, "company enrichment")

                # Map Anymail Finder response to CompanyData
                company_data = self._map_company_response(data, domain)

                return EnrichmentResult(
                    success=True,
                    data=company_data,
                    provider="anymail_finder",
                    cost=0.02,  # Anymail Finder decision-maker endpoint costs 2 credits
                )

        except Exception as exc:
            logging.error(
                f"Anymail Finder company enrichment failed for {domain}: {exc}"
            )
            return EnrichmentResult(
                success=False, error=str(exc), provider="anymail_finder"
            )

    async def enrich_contacts(self, request: EnrichmentRequest) -> EnrichmentResult:
        """
        Enrich company contacts using Anymail Finder.

        Args:
            request: Enrichment request with domain

        Returns:
            Enrichment result with contact data
        """
        domain = request.domain
        try:
            await self._rate_limit()

            # Anymail Finder doesn't have a company search endpoint
            # We'll use the decision-maker endpoint to find company contacts
            url = f"{self.base_url}/find-email/decision-maker"
            payload = {"domain": domain, "decision_maker_category": "ceo"}
            headers = {
                "Authorization": self.api_key,
                "Content-Type": "application/json",
            }

            logging.info(
                f"Anymail Finder contact enrichment request: {url} with payload: {payload}"
            )

            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(url, json=payload, headers=headers)
                data = await self._handle_response(response, "contact enrichment")

                # Map Anymail Finder response to ContactData list
                contacts_data = self._map_contacts_response(data)

                return EnrichmentResult(
                    success=True,
                    data=contacts_data,
                    provider="anymail_finder",
                    cost=0.02,  # Anymail Finder decision-maker endpoint costs 2 credits
                )

        except Exception as exc:
            logging.error(
                f"Anymail Finder contact enrichment failed for {domain}: {exc}"
            )
            return EnrichmentResult(
                success=False, error=str(exc), provider="anymail_finder"
            )

    async def find_email(self, request: EnrichmentRequest) -> EnrichmentResult:
        """
        Find email address using Anymail Finder.

        Args:
            request: Enrichment request with domain, first_name, last_name

        Returns:
            Enrichment result with email data
        """
        domain = request.domain
        try:
            await self._rate_limit()

            url = f"{self.base_url}/find-email/person"
            payload = {
                "domain": domain,
                "first_name": request.first_name,
                "last_name": request.last_name,
            }
            headers = {
                "Authorization": self.api_key,
                "Content-Type": "application/json",
            }

            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(url, json=payload, headers=headers)
                data = await self._handle_response(response, "email discovery")

                # Map Anymail Finder response to ContactData
                contact_data = self._map_email_response(data)

                return EnrichmentResult(
                    success=True,
                    data=contact_data,
                    provider="anymail_finder",
                    cost=0.012,  # Anymail Finder cost per email lookup
                )

        except Exception as exc:
            logging.error(f"Anymail Finder email discovery failed for {domain}: {exc}")
            return EnrichmentResult(
                success=False, error=str(exc), provider="anymail_finder"
            )

    def _map_company_response(
        self, response_data: Dict[str, Any], domain: str
    ) -> CompanyData:
        """
        Map Anymail Finder company response to CompanyData.

        Args:
            response_data: Raw response from Anymail Finder API
            domain: Domain for the company

        Returns:
            Mapped company data
        """
        company = CompanyData(domain=domain)

        # Anymail Finder doesn't provide company data, so we create basic info
        company.name = domain.split(".")[0].title()  # Extract company name from domain

        # Try to get company info from the response if available
        if response_data.get("person_company_name"):
            company.name = response_data["person_company_name"]

        return company

    def _map_contacts_response(
        self, response_data: Dict[str, Any]
    ) -> List[ContactData]:
        """
        Map Anymail Finder contacts response to ContactData list.

        Args:
            response_data: Raw response from Anymail Finder API

        Returns:
            List of mapped contact data
        """
        contacts = []

        # Anymail Finder returns a single contact from decision-maker endpoint
        if response_data.get("email"):
            contact = ContactData()

            # Map core contact fields from the actual API response
            if response_data.get("person_full_name"):
                contact.full_name = response_data["person_full_name"]
                # Try to split full name into first/last
                name_parts = response_data["person_full_name"].split()
                if len(name_parts) >= 2:
                    contact.first_name = name_parts[0]
                    contact.last_name = " ".join(name_parts[1:])
                else:
                    contact.first_name = response_data["person_full_name"]

            if response_data.get("email"):
                contact.email = response_data["email"]
                contact.email_status = response_data.get("email_status", "unknown")

            if response_data.get("person_job_title"):
                contact.job_title = response_data["person_job_title"]

            if response_data.get("person_linkedin_url"):
                linkedin_url = response_data["person_linkedin_url"]
                # Filter out fake/placeholder LinkedIn URLs
                if self._is_valid_linkedin_url(linkedin_url):
                    contact.linkedin_url = linkedin_url

            contacts.append(contact)

        return contacts

    def _map_email_response(self, response_data: Dict[str, Any]) -> ContactData:
        """
        Map Anymail Finder email response to ContactData.

        Args:
            response_data: Raw response from Anymail Finder API

        Returns:
            Mapped contact data with email
        """
        contact = ContactData()

        # Map email-specific fields from the actual API response
        if response_data.get("email"):
            contact.email = response_data["email"]
            contact.email_status = response_data.get("email_status", "unknown")

        if response_data.get("person_full_name"):
            contact.full_name = response_data["person_full_name"]
            # Try to split full name into first/last
            name_parts = response_data["person_full_name"].split()
            if len(name_parts) >= 2:
                contact.first_name = name_parts[0]
                contact.last_name = " ".join(name_parts[1:])
            else:
                contact.first_name = response_data["person_full_name"]

        if response_data.get("person_job_title"):
            contact.job_title = response_data["person_job_title"]

        if response_data.get("person_linkedin_url"):
            linkedin_url = response_data["person_linkedin_url"]
            # Filter out fake/placeholder LinkedIn URLs
            if self._is_valid_linkedin_url(linkedin_url):
                contact.linkedin_url = linkedin_url

        return contact

    async def get_all_domain_prospects(
        self, request: EnrichmentRequest
    ) -> EnrichmentResult:
        """
        Get all domain prospects using Anymail Finder.

        Args:
            request: Enrichment request with domain

        Returns:
            Enrichment result with prospects data
        """
        # Anymail Finder doesn't have a prospects endpoint, so we'll use contacts instead
        return await self.enrich_contacts(request)

    async def search_prospect_email(
        self, request: EnrichmentRequest
    ) -> EnrichmentResult:
        """
        Search for prospect email using Anymail Finder.

        Args:
            request: Enrichment request with prospect_hash

        Returns:
            Enrichment result with email data
        """
        # Anymail Finder doesn't support prospect hash lookups
        # Return error indicating this method is not supported
        return EnrichmentResult(
            success=False,
            error="Anymail Finder doesn't support prospect hash email searches",
            provider="anymail_finder",
        )

    async def validate_email(self, email: str) -> EnrichmentResult:
        """
        Validate email address.

        This method is not supported by Anymail Finder API.
        """
        return EnrichmentResult(
            success=False,
            error="Email validation not supported by Anymail Finder API",
            provider="anymail_finder",
        )

    def _is_valid_linkedin_url(self, linkedin_url: str) -> bool:
        """
        Check if a LinkedIn URL is valid and not a placeholder/fake URL.

        Args:
            linkedin_url: LinkedIn URL to validate

        Returns:
            True if URL appears to be valid, False if it's a placeholder
        """
        if not linkedin_url or not isinstance(linkedin_url, str):
            return False

        # Check for obvious placeholder patterns
        placeholder_patterns = [
            "xxxxxx",  # Common placeholder pattern
            "placeholder",  # Explicit placeholder text
            "test",  # Test data
            "example",  # Example data
            "dummy",  # Dummy data
            "fake",  # Fake data
            "mock",  # Mock data
            "sample",  # Sample data
        ]

        linkedin_lower = linkedin_url.lower()
        for pattern in placeholder_patterns:
            if pattern in linkedin_lower:
                return False

        # Check for valid LinkedIn URL structure
        if not linkedin_url.startswith("https://www.linkedin.com/in/"):
            return False

        # Check if the profile identifier is reasonable (not just placeholder text)
        profile_id = linkedin_url.replace("https://www.linkedin.com/in/", "")
        if (
            len(profile_id) < 3
            or profile_id.isdigit()
            or profile_id in ["profile", "id"]
        ):
            return False

        return True
