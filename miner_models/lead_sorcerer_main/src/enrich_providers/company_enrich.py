"""
CompanyEnrich enrichment provider implementation.

This module implements the CompanyEnrich API client as an enrichment provider
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


class CompanyEnrichProvider(BaseEnrichmentProvider):
    """CompanyEnrich API enrichment provider."""

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize CompanyEnrich provider.

        Args:
            api_key: CompanyEnrich API key (defaults to env var COMPANY_ENRICH_API_KEY)
        """
        super().__init__(name="CompanyEnrich", tier=ProviderTier.TIER_0)
        self.api_key = api_key or os.getenv("COMPANY_ENRICH_API_KEY")
        if not self.api_key:
            raise AuthenticationError("CompanyEnrich API key not provided")

        self.base_url = "https://api.companyenrich.com"
        self._last_request_time = 0.0
        self._min_request_interval = (
            0.2  # 200ms between requests (5 RPS limit, conservative for 300/min)
        )

        logging.info("CompanyEnrich provider initialized")

    async def _rate_limit(self):
        """Ensure we don't exceed CompanyEnrich's rate limits."""
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
            response: HTTP response from CompanyEnrich API
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
                        f"CompanyEnrich API returned non-dict response for {operation}: {type(data)}"
                    )
                    return {}
                return data
            except Exception as exc:
                logging.error(
                    f"CompanyEnrich API response parsing failed for {operation}: {exc}"
                )
                return {}
        elif response.status_code == 401:
            raise AuthenticationError(
                f"CompanyEnrich API authentication failed for {operation}"
            )
        elif response.status_code == 429:
            retry_after = response.headers.get("retry-after", "60")
            raise RateLimitError(
                f"CompanyEnrich API rate limit exceeded for {operation}. Retry after {retry_after} seconds"
            )
        elif response.status_code == 503:
            raise ServiceUnavailableError(
                f"CompanyEnrich API service unavailable for {operation}"
            )
        else:
            error_msg = f"CompanyEnrich API error {response.status_code} for {operation}: {response.text}"
            raise CompanyEnrichmentError(error_msg)

    async def enrich_company(self, request: EnrichmentRequest) -> EnrichmentResult:
        """
        Enrich company information using CompanyEnrich.

        Args:
            request: Enrichment request with domain

        Returns:
            Enrichment result with company data
        """
        domain = request.domain
        try:
            await self._rate_limit()

            url = f"{self.base_url}/companies/enrich"
            params = {"domain": domain}
            headers = {"Authorization": f"Basic {self.api_key}"}

            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(url, params=params, headers=headers)
                data = await self._handle_response(response, "company enrichment")

                # Map CompanyEnrich response to CompanyData
                company_data = self._map_company_response(data, domain)

                return EnrichmentResult(
                    success=True,
                    data=company_data,
                    provider="company_enrich",
                    cost=0.01,  # CompanyEnrich cost per company lookup
                )

        except Exception as exc:
            logging.error(
                f"CompanyEnrich company enrichment failed for {domain}: {exc}"
            )
            return EnrichmentResult(
                success=False, error=str(exc), provider="company_enrich"
            )

    async def enrich_contacts(self, request: EnrichmentRequest) -> EnrichmentResult:
        """
        Enrich company contacts using CompanyEnrich.

        Note: CompanyEnrich doesn't have a separate contacts endpoint.
        We use the main company enrichment endpoint and extract any available contact info.

        Args:
            request: Enrichment request with domain

        Returns:
            Enrichment result with contact data
        """
        domain = request.domain
        try:
            await self._rate_limit()

            url = f"{self.base_url}/companies/enrich"
            params = {"domain": domain}
            headers = {"Authorization": f"Basic {self.api_key}"}

            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(url, params=params, headers=headers)
                data = await self._handle_response(response, "contact enrichment")

                # CompanyEnrich doesn't provide detailed contact lists, so we return basic company info
                # as a contact placeholder
                company_name = data.get("name", domain)
                contact_data = ContactData(
                    first_name=company_name.split()[0] if company_name else "",
                    last_name=" ".join(company_name.split()[1:])
                    if company_name and len(company_name.split()) > 1
                    else "",
                    full_name=company_name,
                    company_name=company_name,
                    company_domain=domain,
                )

                return EnrichmentResult(
                    success=True,
                    data=[contact_data],
                    provider="company_enrich",
                    cost=0.01,  # CompanyEnrich cost per company lookup
                )

        except Exception as exc:
            logging.error(
                f"CompanyEnrich contact enrichment failed for {domain}: {exc}"
            )
            return EnrichmentResult(
                success=False, error=str(exc), provider="company_enrich"
            )

    async def find_email(self, request: EnrichmentRequest) -> EnrichmentResult:
        """
        Find email address using CompanyEnrich.

        Note: CompanyEnrich doesn't support individual email lookups.
        We use the company enrichment endpoint and return basic company info.

        Args:
            request: Enrichment request with domain, first_name, last_name

        Returns:
            Enrichment result with company data (no email available)
        """
        domain = request.domain
        try:
            await self._rate_limit()

            url = f"{self.base_url}/companies/enrich"
            params = {"domain": domain}
            headers = {"Authorization": f"Basic {self.api_key}"}

            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(url, params=params, headers=headers)
                data = await self._handle_response(response, "company enrichment")

                # CompanyEnrich doesn't provide email lookups, so we return company info
                company_name = data.get("name", domain)
                contact_data = ContactData(
                    first_name=request.first_name or company_name.split()[0]
                    if company_name
                    else "",
                    last_name=request.last_name or " ".join(company_name.split()[1:])
                    if company_name and len(company_name.split()) > 1
                    else "",
                    full_name=f"{request.first_name or ''} {request.last_name or ''}".strip()
                    or company_name,
                    company_name=company_name,
                    company_domain=domain,
                    email="",  # No email available from CompanyEnrich
                )

                return EnrichmentResult(
                    success=True,
                    data=contact_data,
                    provider="company_enrich",
                    cost=0.01,  # CompanyEnrich cost per company lookup
                )

        except Exception as exc:
            logging.error(f"CompanyEnrich email discovery failed for {domain}: {exc}")
            return EnrichmentResult(
                success=False, error=str(exc), provider="company_enrich"
            )

    def _map_company_response(
        self, response_data: Dict[str, Any], domain: str
    ) -> CompanyData:
        """
        Map CompanyEnrich company response to CompanyData.

        Args:
            response_data: Raw response from CompanyEnrich API
            domain: Domain for the company

        Returns:
            Mapped company data
        """
        company = CompanyData(domain=domain)

        # Map core company fields based on official API response
        if response_data.get("name"):
            company.name = response_data["name"]

        if response_data.get("description"):
            company.description = response_data["description"]

        if response_data.get("industry"):
            company.industry = response_data["industry"]

        if response_data.get("size"):
            company.size = response_data["size"]

        if response_data.get("location"):
            location_data = response_data["location"]
            if isinstance(location_data, dict):
                company.address = {
                    "city": location_data.get("city"),
                    "state": location_data.get("state"),
                    "country": location_data.get("country"),
                }
            else:
                company.address = {"city": str(location_data)}

        if response_data.get("foundedYear"):
            company.founded_year = response_data["foundedYear"]

        if response_data.get("domain"):
            company.website = f"https://{response_data['domain']}"

        if response_data.get("social", {}).get("linkedinUrl"):
            company.linkedin_url = response_data["social"]["linkedinUrl"]

        if response_data.get("techStack"):
            company.technologies = response_data["techStack"]

        return company

    def _map_contacts_response(
        self, response_data: Dict[str, Any]
    ) -> List[ContactData]:
        """
        Map CompanyEnrich contacts response to ContactData list.

        Args:
            response_data: Raw response from CompanyEnrich API

        Returns:
            List of mapped contact data
        """
        contacts = []

        # Extract contacts from response
        contacts_list = response_data.get("contacts", [])
        if not isinstance(contacts_list, list):
            return contacts

        for contact_data in contacts_list:
            contact = ContactData()

            # Map core contact fields
            if contact_data.get("first_name"):
                contact.first_name = contact_data["first_name"]

            if contact_data.get("last_name"):
                contact.last_name = contact_data["last_name"]

            if contact_data.get("full_name"):
                contact.full_name = contact_data["full_name"]

            if contact_data.get("email"):
                contact.email = contact_data["email"]
                contact.email_source = ["provider"]
                contact.email_status = ["unknown"]

            if contact_data.get("phone"):
                contact.phone = contact_data["phone"]

            if contact_data.get("title"):
                contact.job_title = contact_data["title"]

            if contact_data.get("linkedin_url"):
                linkedin_url = contact_data["linkedin_url"]
                # Filter out fake/placeholder LinkedIn URLs
                if self._is_valid_linkedin_url(linkedin_url):
                    contact.linkedin_url = linkedin_url

            if contact_data.get("department"):
                contact.department = contact_data["department"]

            if contact_data.get("seniority"):
                contact.seniority = contact_data["seniority"]

            contacts.append(contact)

        return contacts

    def _map_email_response(self, response_data: Dict[str, Any]) -> ContactData:
        """
        Map CompanyEnrich email response to ContactData.

        Args:
            response_data: Raw response from CompanyEnrich API

        Returns:
            Mapped contact data with email
        """
        contact = ContactData()

        # Map email-specific fields
        if response_data.get("email"):
            contact.email = response_data["email"]
            contact.email_source = ["provider"]
            contact.email_status = ["unknown"]

        if response_data.get("confidence"):
            contact.email_confidence = response_data["confidence"]

        if response_data.get("first_name"):
            contact.first_name = response_data["first_name"]

        if response_data.get("last_name"):
            contact.last_name = response_data["last_name"]

        if response_data.get("full_name"):
            contact.full_name = response_data["full_name"]

        return contact

    async def get_all_domain_prospects(
        self, request: EnrichmentRequest
    ) -> EnrichmentResult:
        """
        Get all domain prospects using CompanyEnrich.

        Args:
            request: Enrichment request with domain

        Returns:
            Enrichment result with prospects data
        """
        # CompanyEnrich doesn't have a prospects endpoint, so we'll use contacts instead
        return await self.enrich_contacts(request)

    async def search_prospect_email(
        self, request: EnrichmentRequest
    ) -> EnrichmentResult:
        """
        Search for prospect email using CompanyEnrich.

        Args:
            request: Enrichment request with prospect_hash

        Returns:
            Enrichment result with email data
        """
        # CompanyEnrich doesn't support prospect hash lookups
        # Return error indicating this method is not supported
        return EnrichmentResult(
            success=False,
            error="CompanyEnrich doesn't support prospect hash email searches",
            provider="company_enrich",
        )

    async def validate_email(self, email: str) -> EnrichmentResult:
        """
        Validate email address.

        This method is not supported by CompanyEnrich API.
        """
        return EnrichmentResult(
            success=False,
            error="Email validation not supported by CompanyEnrich API",
            provider="company_enrich",
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
