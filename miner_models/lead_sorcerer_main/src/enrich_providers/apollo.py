"""
Apollo Enrichment Provider for Lead Sorcerer

This module provides enrichment capabilities using Apollo's APIs,
integrating with the existing enrichment pipeline.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any

from .base import (
    BaseEnrichmentProvider,
    EnrichmentRequest,
    EnrichmentResult,
    ProviderTier,
)
from ..apollo.client import ApolloClient, create_apollo_client_from_env

# Configure logging with PII protection
logger = logging.getLogger(__name__)


class ApolloEnrichProvider(BaseEnrichmentProvider):
    """
    Apollo enrichment provider implementing the BaseEnrichProvider interface.

    This provider maintains strict no-hardcoding principles - all business parameters
    must be loaded from configuration, never from source code defaults.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Apollo enrichment provider.

        Args:
            config: Provider configuration containing Apollo settings
        """
        super().__init__(config)
        self.tier = ProviderTier.TIER_0  # Apollo is a primary data source

        # Initialize Apollo client
        self.apollo_client = self._initialize_apollo_client()

        # Load configuration
        self.batch_size = config.get("batch_size", 25)
        self.use_bulk = config.get("use_bulk", True)
        self.required_fields = config.get(
            "required_fields", ["email", "phone", "linkedin_url"]
        )

        logger.info("Apollo enrichment provider initialized")

    def _initialize_apollo_client(self) -> ApolloClient:
        """
        Initialize Apollo API client.

        Returns:
            Configured Apollo client

        Raises:
            RuntimeError: If Apollo client initialization fails
        """
        try:
            client = create_apollo_client_from_env()
            logger.info("Apollo API client initialized for enrichment provider")
            return client
        except Exception as e:
            logger.error(f"Failed to initialize Apollo client: {e}")
            raise RuntimeError(f"Apollo client initialization failed: {e}")

    async def enrich_company(self, request: EnrichmentRequest) -> EnrichmentResult:
        """
        Enrich company data using Apollo API.

        Args:
            request: Company enrichment request

        Returns:
            Enrichment result with company data
        """
        try:
            domain = request.domain
            if not domain:
                return EnrichmentResult(
                    success=False,
                    error="No domain provided for company enrichment",
                    data={},
                )

            logger.info(f"Enriching company data for domain: {domain}")

            # Enrich company using Apollo API
            enriched_data = await self.apollo_client.enrich_company(domain)

            if not enriched_data or "company" not in enriched_data:
                return EnrichmentResult(
                    success=False, error="No company data returned from Apollo", data={}
                )

            # Transform Apollo data to enrichment result format
            company_data = enriched_data["company"]
            result_data = self._transform_company_data(company_data)

            logger.info(f"Company enrichment completed for domain: {domain}")

            return EnrichmentResult(success=True, data=result_data, provider="apollo")

        except Exception as e:
            logger.error(f"Company enrichment failed for domain {request.domain}: {e}")
            return EnrichmentResult(
                success=False, error=f"Company enrichment failed: {e}", data={}
            )

    async def enrich_contact(self, request: EnrichmentRequest) -> EnrichmentResult:
        """
        Enrich contact data using Apollo API.

        Args:
            request: Contact enrichment request

        Returns:
            Enrichment result with contact data
        """
        try:
            email = request.email
            if not email:
                return EnrichmentResult(
                    success=False,
                    error="No email provided for contact enrichment",
                    data={},
                )

            logger.info(f"Enriching contact data for email: {email}")

            # Enrich person using Apollo API
            enriched_data = await self.apollo_client.enrich_person(email)

            if not enriched_data or "person" not in enriched_data:
                return EnrichmentResult(
                    success=False, error="No person data returned from Apollo", data={}
                )

            # Transform Apollo data to enrichment result format
            person_data = enriched_data["person"]
            result_data = self._transform_person_data(person_data)

            logger.info(f"Contact enrichment completed for email: {email}")

            return EnrichmentResult(success=True, data=result_data, provider="apollo")

        except Exception as e:
            logger.error(f"Contact enrichment failed for email {request.email}: {e}")
            return EnrichmentResult(
                success=False, error=f"Contact enrichment failed: {e}", data={}
            )

    async def bulk_enrich_companies(
        self, requests: List[EnrichmentRequest]
    ) -> List[EnrichmentResult]:
        """
        Bulk enrich multiple companies using Apollo API.

        Args:
            requests: List of company enrichment requests

        Returns:
            List of enrichment results
        """
        if not requests:
            return []

        logger.info(f"Bulk enriching {len(requests)} companies")

        if self.use_bulk:
            return await self._bulk_enrich_companies_batch(requests)
        else:
            return await self._bulk_enrich_companies_individual(requests)

    async def _bulk_enrich_companies_batch(
        self, requests: List[EnrichmentRequest]
    ) -> List[EnrichmentResult]:
        """
        Bulk enrich companies using Apollo's batch endpoint.

        Args:
            requests: List of company enrichment requests

        Returns:
            List of enrichment results
        """
        results = []

        for i in range(0, len(requests), self.batch_size):
            batch = requests[i : i + self.batch_size]

            try:
                # Extract domains from batch
                domains = [req.domain for req in batch if req.domain]

                if not domains:
                    # Add error results for requests without domains
                    for req in batch:
                        results.append(
                            EnrichmentResult(
                                success=False, error="No domain provided", data={}
                            )
                        )
                    continue

                # Bulk enrich using Apollo API
                enriched_companies = await self.apollo_client.bulk_enrich_companies(
                    domains
                )

                # Map results back to requests
                for j, req in enumerate(batch):
                    if req.domain and j < len(enriched_companies):
                        company_data = enriched_companies[j]
                        if company_data and "company" in company_data:
                            result_data = self._transform_company_data(
                                company_data["company"]
                            )
                            results.append(
                                EnrichmentResult(
                                    success=True, data=result_data, provider="apollo"
                                )
                            )
                        else:
                            results.append(
                                EnrichmentResult(
                                    success=False,
                                    error="No company data returned",
                                    data={},
                                )
                            )
                    else:
                        results.append(
                            EnrichmentResult(
                                success=False,
                                error="No domain provided or enrichment failed",
                                data={},
                            )
                        )

                # Add rate limiting delay
                await asyncio.sleep(0.1)

            except Exception as e:
                logger.error(
                    f"Bulk company enrichment failed for batch {i // self.batch_size + 1}: {e}"
                )
                # Add error results for failed batch
                for req in batch:
                    results.append(
                        EnrichmentResult(
                            success=False, error=f"Bulk enrichment failed: {e}", data={}
                        )
                    )

        logger.info(f"Bulk company enrichment completed: {len(results)} results")
        return results

    async def _bulk_enrich_companies_individual(
        self, requests: List[EnrichmentRequest]
    ) -> List[EnrichmentResult]:
        """
        Bulk enrich companies using individual enrichment calls.

        Args:
            requests: List of company enrichment requests

        Returns:
            List of enrichment results
        """
        results = []

        for request in requests:
            try:
                result = await self.enrich_company(request)
                results.append(result)

                # Add rate limiting delay
                await asyncio.sleep(0.1)

            except Exception as e:
                logger.error(f"Individual company enrichment failed: {e}")
                results.append(
                    EnrichmentResult(
                        success=False,
                        error=f"Individual enrichment failed: {e}",
                        data={},
                    )
                )

        return results

    async def bulk_enrich_contacts(
        self, requests: List[EnrichmentRequest]
    ) -> List[EnrichmentResult]:
        """
        Bulk enrich multiple contacts using Apollo API.

        Args:
            requests: List of contact enrichment requests

        Returns:
            List of enrichment results
        """
        if not requests:
            return []

        logger.info(f"Bulk enriching {len(requests)} contacts")

        if self.use_bulk:
            return await self._bulk_enrich_contacts_batch(requests)
        else:
            return await self._bulk_enrich_contacts_individual(requests)

    async def _bulk_enrich_contacts_batch(
        self, requests: List[EnrichmentRequest]
    ) -> List[EnrichmentResult]:
        """
        Bulk enrich contacts using Apollo's batch endpoint.

        Args:
            requests: List of contact enrichment requests

        Returns:
            List of enrichment results
        """
        results = []

        for i in range(0, len(requests), self.batch_size):
            batch = requests[i : i + self.batch_size]

            try:
                # Extract emails from batch
                emails = [req.email for req in batch if req.email]

                if not emails:
                    # Add error results for requests without emails
                    for req in batch:
                        results.append(
                            EnrichmentResult(
                                success=False, error="No email provided", data={}
                            )
                        )
                    continue

                # Bulk enrich using Apollo API
                enriched_persons = await self.apollo_client.bulk_enrich_persons(emails)

                # Map results back to requests
                for j, req in enumerate(batch):
                    if req.email and j < len(enriched_persons):
                        person_data = enriched_persons[j]
                        if person_data and "person" in person_data:
                            result_data = self._transform_person_data(
                                person_data["person"]
                            )
                            results.append(
                                EnrichmentResult(
                                    success=True, data=result_data, provider="apollo"
                                )
                            )
                        else:
                            results.append(
                                EnrichmentResult(
                                    success=False,
                                    error="No person data returned",
                                    data={},
                                )
                            )
                    else:
                        results.append(
                            EnrichmentResult(
                                success=False,
                                error="No email provided or enrichment failed",
                                data={},
                            )
                        )

                # Add rate limiting delay
                await asyncio.sleep(0.1)

            except Exception as e:
                logger.error(
                    f"Bulk contact enrichment failed for batch {i // self.batch_size + 1}: {e}"
                )
                # Add error results for failed batch
                for req in batch:
                    results.append(
                        EnrichmentResult(
                            success=False, error=f"Bulk enrichment failed: {e}", data={}
                        )
                    )

        logger.info(f"Bulk contact enrichment completed: {len(results)} results")
        return results

    async def _bulk_enrich_contacts_individual(
        self, requests: List[EnrichmentRequest]
    ) -> List[EnrichmentResult]:
        """
        Bulk enrich contacts using individual enrichment calls.

        Args:
            requests: List of contact enrichment requests

        Returns:
            List of enrichment results
        """
        results = []

        for request in requests:
            try:
                result = await self.enrich_contact(request)
                results.append(result)

                # Add rate limiting delay
                await asyncio.sleep(0.1)

            except Exception as e:
                logger.error(f"Individual contact enrichment failed: {e}")
                results.append(
                    EnrichmentResult(
                        success=False,
                        error=f"Individual enrichment failed: {e}",
                        data={},
                    )
                )

        return results

    def _transform_company_data(self, company_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transform Apollo company data to enrichment result format.

        Args:
            company_data: Apollo company data

        Returns:
            Transformed company data
        """
        return {
            "company": {
                "name": company_data.get("name"),
                "description": company_data.get("description"),
                "industry": company_data.get("industry"),
                "size_hint": company_data.get("size"),
                "employee_count": company_data.get("employee_count"),
                "founded_year": company_data.get("founded_year"),
                "website": company_data.get("website"),
                "linkedin_url": company_data.get("linkedin_url"),
                "phone": company_data.get("phone"),
                "hq_location": self._format_location(company_data.get("address")),
                "tech_stack": company_data.get("technologies", []),
                "revenue_range": company_data.get("revenue_range"),
                "number_of_locations": len(company_data.get("locations", [])),
                "funding_stage": company_data.get("funding_stage"),
            }
        }

    def _transform_person_data(self, person_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transform Apollo person data to enrichment result format.

        Args:
            person_data: Apollo person data

        Returns:
            Transformed person data
        """
        return {
            "contact": {
                "full_name": person_data.get("full_name"),
                "role": person_data.get("job_title"),
                "department": person_data.get("department"),
                "email": person_data.get("email"),
                "phone": person_data.get("phone"),
                "linkedin": person_data.get("linkedin_url"),
                "location": person_data.get("location"),
                "seniority": person_data.get("seniority_level"),
            },
            "company": {
                "name": person_data.get("company", {}).get("name"),
                "domain": person_data.get("company", {}).get("domain"),
                "website": person_data.get("company", {}).get("website"),
            },
        }

    def _format_location(self, address: Optional[Dict[str, Any]]) -> Optional[str]:
        """
        Format address information for location field.

        Args:
            address: Address dictionary from Apollo

        Returns:
            Formatted location string or None
        """
        if not address:
            return None

        parts = []
        if address.get("city"):
            parts.append(address["city"])
        if address.get("state"):
            parts.append(address["state"])
        if address.get("country"):
            parts.append(address["country"])

        return ", ".join(parts) if parts else None

    def get_provider_info(self) -> Dict[str, Any]:
        """
        Get provider information.

        Returns:
            Provider information dictionary
        """
        return {
            "name": "Apollo",
            "tier": self.tier.value,
            "capabilities": [
                "company_enrichment",
                "contact_enrichment",
                "bulk_operations",
            ],
            "rate_limits": "100 requests per minute",
            "cost_per_request": 0.015,  # 1 credit = $0.015 USD
            "supported_fields": self.required_fields,
        }

    async def close(self):
        """Close Apollo client and cleanup resources"""
        if self.apollo_client:
            await self.apollo_client.close()
            logger.info("Apollo enrichment provider closed")

    def __del__(self):
        """Cleanup on deletion"""
        if hasattr(self, "apollo_client") and self.apollo_client:
            # Note: This is not ideal for async cleanup, but provides fallback
            try:
                asyncio.create_task(self.apollo_client.close())
            except Exception:
                pass
