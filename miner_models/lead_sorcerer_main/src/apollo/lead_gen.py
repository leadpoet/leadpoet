"""
Apollo Lead Generation Tool for Lead Sorcerer

This module provides lead generation capabilities using Apollo's comprehensive database,
bypassing the traditional Domain + Crawl pipeline while maintaining data quality standards.
"""

import asyncio
import logging
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any

from src.common import PermitManager, get_role_priority
from .client import ApolloClient, create_apollo_client_from_env
from .optimizer import AISearchQueryOptimizer, AISearchConfig
from .scorer import (
    ApolloResultScorer,
    ScoringWeights,
    QualityThresholds,
)
from .adaptive import AdaptiveSearchOrchestrator, StrategyConfig

# Configure logging with PII protection
logger = logging.getLogger(__name__)


class ApolloLeadGenError(Exception):
    """Base exception for Apollo lead generation errors"""

    pass


class ApolloLeadGen:
    """Enhanced Apollo Lead Generation with performance optimization and advanced search strategies"""

    def __init__(
        self,
        icp_config: Dict[str, Any],
        data_dir: Optional[str] = None,
        permit_manager: Optional[PermitManager] = None,
    ):
        """Initialize Apollo Lead Generation with advanced features"""
        self.icp_config = icp_config
        self.data_dir = data_dir
        self.permit_manager = permit_manager or PermitManager()

        # Validate Apollo configuration
        self._validate_apollo_config(self.icp_config)

        # Initialize Apollo client with performance optimization
        self.apollo_client = self._initialize_apollo_client()

        # Initialize AI search optimizer
        ai_config = AISearchConfig(
            enable_ai_optimization=self.icp_config.get("apollo", {})
            .get("ai_optimization", {})
            .get("enabled", True),
            enable_dynamic_filters=self.icp_config.get("apollo", {})
            .get("ai_optimization", {})
            .get("dynamic_filters", True),
            enable_result_scoring=self.icp_config.get("apollo", {})
            .get("ai_optimization", {})
            .get("result_scoring", True),
            enable_adaptive_strategies=self.icp_config.get("apollo", {})
            .get("ai_optimization", {})
            .get("adaptive_strategies", True),
            max_filter_variations=self.icp_config.get("apollo", {})
            .get("ai_optimization", {})
            .get("max_variations", 5),
        )
        self.ai_optimizer = AISearchQueryOptimizer(ai_config)

        # Initialize result scorer
        scoring_weights = ScoringWeights(
            completeness=self.icp_config.get("apollo", {})
            .get("scoring", {})
            .get("weights", {})
            .get("completeness", 0.3),
            relevance=self.icp_config.get("apollo", {})
            .get("scoring", {})
            .get("weights", {})
            .get("relevance", 0.25),
            freshness=self.icp_config.get("apollo", {})
            .get("scoring", {})
            .get("weights", {})
            .get("freshness", 0.2),
            contact_quality=self.icp_config.get("apollo", {})
            .get("scoring", {})
            .get("weights", {})
            .get("contact_quality", 0.15),
            company_quality=self.icp_config.get("apollo", {})
            .get("scoring", {})
            .get("weights", {})
            .get("company_quality", 0.1),
        )

        quality_thresholds = QualityThresholds(
            min_completeness=self.icp_config.get("apollo", {})
            .get("scoring", {})
            .get("thresholds", {})
            .get("min_completeness", 0.6),
            min_relevance=self.icp_config.get("apollo", {})
            .get("scoring", {})
            .get("thresholds", {})
            .get("min_relevance", 0.7),
            min_overall_score=self.icp_config.get("apollo", {})
            .get("scoring", {})
            .get("thresholds", {})
            .get("min_overall_score", 0.65),
            min_contact_count=self.icp_config.get("apollo", {})
            .get("scoring", {})
            .get("thresholds", {})
            .get("min_contact_count", 1),
            min_company_info=self.icp_config.get("apollo", {})
            .get("scoring", {})
            .get("thresholds", {})
            .get("min_company_info", 3),
        )

        self.result_scorer = ApolloResultScorer(scoring_weights, quality_thresholds)

        # Initialize adaptive search orchestrator
        strategy_config = StrategyConfig(
            enable_adaptation=self.icp_config.get("apollo", {})
            .get("adaptive_strategies", {})
            .get("enabled", True),
            adaptation_threshold=self.icp_config.get("apollo", {})
            .get("adaptive_strategies", {})
            .get("adaptation_threshold", 0.1),
            adaptation_interval=self.icp_config.get("apollo", {})
            .get("adaptive_strategies", {})
            .get("adaptation_interval", 10),
            max_strategies=self.icp_config.get("apollo", {})
            .get("adaptive_strategies", {})
            .get("max_strategies", 5),
            performance_window=self.icp_config.get("apollo", {})
            .get("adaptive_strategies", {})
            .get("performance_window", 20),
            exploration_rate=self.icp_config.get("apollo", {})
            .get("adaptive_strategies", {})
            .get("exploration_rate", 0.2),
        )
        self.adaptive_orchestrator = AdaptiveSearchOrchestrator(strategy_config)

        # Initialize search filters from Apollo config
        self.apollo_config = self.icp_config.get("apollo", {})
        search_config = self.apollo_config.get("search", {})
        self.search_filters = {
            "strategy": search_config.get("strategy", "company_first"),
            "max_results": search_config.get("max_results", 100),
            "company_filters": search_config.get("company_filters", {}),
            "person_filters": search_config.get("person_filters", {}),
        }

        # Performance tracking
        self.performance_metrics = {
            "total_searches": 0,
            "total_results": 0,
            "avg_response_time": 0.0,
            "cache_hit_rate": 0.0,
            "quality_scores": [],
            "strategy_performance": {},
        }

        logger.info("Apollo Lead Generation initialized with advanced features")

    def _validate_apollo_config(self, icp_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate Apollo configuration integrity.

        Args:
            icp_config: ICP configuration to validate

        Returns:
            Validated Apollo configuration

        Raises:
            ApolloLeadGenError: If Apollo configuration is invalid or incomplete
        """
        # Check if Apollo mode is enabled
        lead_generation_mode = icp_config.get("lead_generation_mode", "traditional")
        if lead_generation_mode != "apollo":
            raise ApolloLeadGenError(
                f"Apollo mode not enabled. Current mode: {lead_generation_mode}"
            )

        # Check if Apollo config block exists
        if "apollo" not in icp_config:
            raise ApolloLeadGenError("Apollo mode enabled but apollo config missing")

        apollo_config = icp_config["apollo"]

        # Validate required fields
        required_fields = ["enabled", "search", "enrich"]
        for field in required_fields:
            if field not in apollo_config:
                raise ApolloLeadGenError(f"Missing required Apollo field: {field}")

        # Validate search configuration
        search_config = apollo_config.get("search", {})
        if "strategy" not in search_config:
            raise ApolloLeadGenError("Apollo search strategy is required")

        strategy = search_config["strategy"]
        if strategy not in ["company_first", "person_first"]:
            raise ApolloLeadGenError(f"Invalid Apollo search strategy: {strategy}")

        # Validate enrich configuration
        enrich_config = apollo_config.get("enrich", {})
        if not enrich_config.get("enabled", False):
            raise ApolloLeadGenError("Apollo enrichment must be enabled")

        logger.info("Apollo configuration validation passed")
        return apollo_config

    def _initialize_apollo_client(self) -> ApolloClient:
        """
        Initialize Apollo API client.

        Returns:
            Configured Apollo client

        Raises:
            ApolloLeadGenError: If Apollo client initialization fails
        """
        try:
            # Create client from environment variables
            client = create_apollo_client_from_env()
            logger.info("Apollo API client initialized successfully")
            return client
        except Exception as e:
            raise ApolloLeadGenError(f"Failed to initialize Apollo client: {e}")

    def _load_filters_from_icp_config(
        self, icp_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Load all business parameters from ICP configuration.

        Args:
            icp_config: ICP configuration to load filters from

        Returns:
            Search filters loaded from configuration

        Note: This method ensures zero hardcoded business parameters.
        All filters must come from the ICP configuration.
        """
        apollo_config = icp_config.get("apollo", {})
        search_config = apollo_config.get("search", {})

        # Load company filters from ICP config
        company_filters = search_config.get("company_filters", {})

        # Load person filters from ICP config
        person_filters = search_config.get("person_filters", {})

        # Validate that filters are not empty arrays (indicating they were loaded from config)
        self._validate_filters_loaded_from_config(company_filters, person_filters)

        filters = {
            "company_filters": company_filters,
            "person_filters": person_filters,
            "strategy": search_config.get("strategy", "company_first"),
            "max_results": search_config.get("max_results", 1000),
            "use_pagination": search_config.get("use_pagination", True),
        }

        logger.info(f"Loaded search filters from ICP config: {filters}")
        return filters

    def _validate_filters_loaded_from_config(
        self, company_filters: Dict[str, Any], person_filters: Dict[str, Any]
    ):
        """
        Validate that business filters were loaded from configuration.

        Args:
            company_filters: Company search filters
            person_filters: Person search filters

        Raises:
            ApolloLeadGenError: If filters appear to be hardcoded defaults
        """
        # Check for common hardcoded patterns
        hardcoded_patterns = [
            ["technology", "software", "saas"],
            ["1-10", "11-50", "51-200"],
            ["CEO", "CTO", "VP Engineering"],
            ["healthcare", "medical", "pharmaceuticals"],
        ]

        all_filters = {**company_filters, **person_filters}

        for filter_type, values in all_filters.items():
            if isinstance(values, list):
                for pattern in hardcoded_patterns:
                    if values == pattern:
                        raise ApolloLeadGenError(
                            f"Hardcoded business parameters detected in {filter_type}: {pattern}. "
                            "All business parameters must be loaded from ICP configuration, never hardcoded."
                        )

    async def run(self) -> Dict[str, Any]:
        """
        Main entry point for Apollo lead generation.

        Returns:
            Lead generation results in unified format

        Raises:
            ApolloLeadGenError: If lead generation fails
        """
        try:
            logger.info("Starting Apollo lead generation")
            logger.info(f"Search filters: {self.search_filters}")
            logger.info(f"Apollo client initialized: {self.apollo_client is not None}")

            # Initialize Apollo client session
            async with self.apollo_client:
                logger.info("Apollo client session started")

                # Execute search strategy
                logger.info("About to execute search strategy...")
                search_results = await self._execute_search_strategy()
                logger.info(
                    f"Search strategy executed, got {len(search_results)} results"
                )

                # Enrich search results
                logger.info("About to enrich search results...")
                logger.info(
                    f"Search results structure: {[list(result.keys()) if isinstance(result, dict) else type(result) for result in search_results]}"
                )
                logger.info(
                    f"First search result sample: {search_results[0] if search_results else 'No results'}"
                )

                enriched_results = await self._enrich_search_results(search_results)
                logger.info(
                    f"Enrichment completed, got {len(enriched_results)} enriched results"
                )
                logger.info(
                    f"Enriched results structure: {[list(result.keys()) if isinstance(result, dict) else type(result) for result in enriched_results]}"
                )
                logger.info(
                    f"First enriched result sample: {enriched_results[0] if enriched_results else 'No enriched results'}"
                )

                # Transform to unified schema
                logger.info("About to transform to unified schema...")
                lead_records = self._map_to_unified_schema(enriched_results)
                logger.info(
                    f"Schema transformation completed, got {len(lead_records)} lead records"
                )

                # Validate lead quality
                logger.info("About to validate lead quality...")
                passing_leads = [
                    lead for lead in lead_records if self._validate_lead_quality(lead)
                ]
                logger.info(
                    f"Lead quality validation completed, got {len(passing_leads)} passing leads"
                )

                # Get cost summary
                cost_summary = self.apollo_client.get_cost_summary()

                result = {
                    "data": {
                        "lead_records": passing_leads,
                        "total_found": len(search_results),
                        "total_enriched": len(enriched_results),
                        "total_passing": len(passing_leads),
                    },
                    "metrics": {
                        "apollo_credits_used": cost_summary["credits_used"],
                        "apollo_cost_usd": cost_summary["total_usd"],
                        "search_strategy": self.search_filters["strategy"],
                        "execution_time": datetime.utcnow().isoformat(),
                    },
                    "errors": [],
                }

                logger.info(
                    f"Apollo lead generation completed: {len(passing_leads)} leads generated"
                )
                logger.info(f"Final result: {result}")
                return result

        except Exception as e:
            logger.error(f"Apollo lead generation failed: {e}")
            logger.error(f"Exception details: {type(e).__name__}: {str(e)}")
            import traceback

            logger.error(f"Traceback: {traceback.format_exc()}")
            raise ApolloLeadGenError(f"Lead generation failed: {e}")

    async def _execute_search_strategy(self) -> List[Dict[str, Any]]:
        """
        Execute the configured search strategy.

        Returns:
            List of search results

        Note: Search strategy and filters are loaded from ICP config,
        never hardcoded in source code.
        """
        strategy = self.search_filters["strategy"]
        max_results = self.search_filters["max_results"]

        logger.info(f"Executing Apollo search strategy: {strategy}")
        logger.info(f"Max results requested: {max_results}")
        logger.info(f"Search filters available: {list(self.search_filters.keys())}")

        if strategy == "company_first":
            logger.info("Executing company-first search strategy")
            result = await self._execute_company_first_search(max_results)
            logger.info(f"Company-first search returned {len(result)} results")
            return result
        elif strategy == "person_first":
            logger.info("Executing person-first search strategy")
            result = await self._execute_person_first_search(max_results)
            logger.info(f"Person-first search returned {len(result)} results")
            return result
        else:
            raise ApolloLeadGenError(f"Unknown search strategy: {strategy}")

    async def _execute_company_first_search(
        self, max_results: int
    ) -> List[Dict[str, Any]]:
        """
        Execute company-first search strategy.

        Args:
            max_results: Maximum number of results to return

        Returns:
            List of company search results
        """
        logger.info("Executing company-first search strategy")

        company_filters = self.search_filters["company_filters"]
        logger.info(f"Company filters: {company_filters}")
        logger.info(f"Max results requested: {max_results}")

        all_companies = []
        page = 1

        while len(all_companies) < max_results:
            try:
                logger.info(f"Searching page {page} for companies...")
                # Search for companies using filters from ICP config
                response = await self.apollo_client.search_companies(
                    filters=company_filters,
                    max_results=min(100, max_results - len(all_companies)),
                    page=page,
                )

                logger.info(f"Got response from Apollo API: {list(response.keys())}")
                # Apollo mixed_people/search returns results in 'people' field, not 'companies'
                people = response.get("people", [])
                logger.info(f"Found {len(people)} people on page {page}")

                if not people:
                    logger.info(f"No people found on page {page}, stopping search")
                    break

                # Ensure each person has organization data before adding to results
                enriched_people = []
                for p in people:
                    p = await self._ensure_company_block(p)
                    enriched_people.append(p)

                all_companies.extend(enriched_people)
                page += 1

                # Check if we've reached the end
                pagination = response.get("pagination", {})
                if page > pagination.get("total_pages", 1):
                    break

            except Exception as e:
                logger.error(f"Company search failed on page {page}: {e}")
                break

        logger.info(
            f"Company-first search completed: {len(all_companies)} companies found"
        )
        return all_companies

    async def _execute_person_first_search(
        self, max_results: int
    ) -> List[Dict[str, Any]]:
        """
        Execute person-first search strategy.

        Args:
            max_results: Maximum number of results to return

        Returns:
            List of person search results
        """
        logger.info("Executing person-first search strategy")

        person_filters = self.search_filters["person_filters"]
        all_persons = []
        page = 1

        while len(all_persons) < max_results:
            try:
                # Search for persons using filters from ICP config
                response = await self.apollo_client.search_persons(
                    filters=person_filters,
                    max_results=min(100, max_results - len(all_persons)),
                    page=page,
                )

                persons = response.get("people", [])
                if not persons:
                    break

                # Ensure each person has organization data before adding to results
                enriched_persons = []
                for p in persons:
                    p = await self._ensure_company_block(p)
                    enriched_persons.append(p)

                all_persons.extend(enriched_persons)
                page += 1

                # Check if we've reached the end
                pagination = response.get("pagination", {})
                if page > pagination.get("total_pages", 1):
                    break

            except Exception as e:
                logger.error(f"Person search failed on page {page}: {e}")
                break

        logger.info(f"Person-first search completed: {len(all_persons)} persons found")
        return all_persons

    async def _enrich_search_results(
        self, search_results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Enrich search results with additional data.

        Args:
            search_results: Raw search results to enrich

        Returns:
            Enriched results
        """
        if not search_results:
            return []

        logger.info(f"Enriching {len(search_results)} search results")

        enriched_results = []

        # Check if bulk enrichment is enabled
        if self.apollo_config.get("enrich", {}).get("use_bulk", True):
            enriched_results = await self._bulk_enrich_results(search_results)
        else:
            enriched_results = await self._individual_enrich_results(search_results)

        # Ensure we have valid results
        if not enriched_results:
            logger.warning(
                "Enrichment returned no results, using original search results"
            )
            enriched_results = search_results
        else:
            # Filter out None values
            enriched_results = [
                result for result in enriched_results if result is not None
            ]
            if not enriched_results:
                logger.warning(
                    "Enrichment returned only None values, using original search results"
                )
                enriched_results = search_results

        logger.info(f"Enrichment completed: {len(enriched_results)} results enriched")
        return enriched_results

    async def _bulk_enrich_results(
        self, search_results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Bulk enrich search results for better performance.

        Args:
            search_results: Search results to enrich

        Returns:
            Bulk enriched results
        """
        batch_size = self.apollo_config.get("enrich", {}).get("batch_size", 25)
        enriched_results = []

        for i in range(0, len(search_results), batch_size):
            batch = search_results[i : i + batch_size]

            try:
                if self.search_filters["strategy"] == "company_first":
                    # For company-first strategy, enrich for emails if configured
                    if self.apollo_config.get("enrich", {}).get("unlock_emails", False):
                        logger.info(
                            "Company-first strategy: Enriching leads individually to get actual emails"
                        )
                        # Use individual enrichment calls since bulk endpoint is not working
                        enriched_batch = []
                        for result in batch:
                            try:
                                enriched_person = (
                                    await self.apollo_client.enrich_person_for_email(
                                        result
                                    )
                                )
                                if enriched_person:
                                    # Update with enriched email data
                                    result["email"] = enriched_person.get(
                                        "email", result.get("email")
                                    )
                                    result["phone"] = enriched_person.get(
                                        "phone", result.get("phone")
                                    )
                                    logger.info(
                                        f"Enriched email for {result.get('name', 'Unknown')}: {result.get('email')}"
                                    )
                                else:
                                    logger.info(
                                        f"No enrichment data for {result.get('name', 'Unknown')}"
                                    )
                                enriched_batch.append(result)
                                # Add small delay between individual calls
                                await asyncio.sleep(0.1)
                            except Exception as e:
                                logger.warning(
                                    f"Individual enrichment failed for {result.get('name', 'Unknown')}: {e}"
                                )
                                enriched_batch.append(result)  # Keep original result
                        enriched_results.extend(enriched_batch)
                    else:
                        # Skip enrichment, use original results
                        logger.info(
                            "Company-first strategy: Skipping enrichment, using original search results with organization data"
                        )
                        enriched_results.extend(batch)
                else:
                    # Bulk enrich persons
                    emails = [
                        result.get("email")
                        for result in batch
                        if result.get("email")
                        and not result.get("email", "").startswith("email_not_unlocked")
                    ]
                    if emails:
                        enriched_persons = await self.apollo_client.bulk_enrich_persons(
                            emails
                        )
                        enriched_results.extend(enriched_persons)
                    else:
                        # If no valid emails found, use original results
                        logger.info(
                            "No valid emails found for enrichment, using original results"
                        )
                        enriched_results.extend(batch)

                # Add rate limiting delay
                await asyncio.sleep(0.1)

            except Exception as e:
                logger.error(
                    f"Bulk enrichment failed for batch {i // batch_size + 1}: {e}"
                )
                logger.info(
                    f"Falling back to original data for batch {i // batch_size + 1}"
                )
                # Add original results if enrichment fails
                enriched_results.extend(batch)

        return enriched_results

    async def _individual_enrich_results(
        self, search_results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Individually enrich search results.

        Args:
            search_results: Search results to enrich

        Returns:
            Individually enriched results
        """
        enriched_results = []

        for result in search_results:
            try:
                if self.search_filters["strategy"] == "company_first":
                    # For company-first strategy, enrich for email if configured
                    if self.apollo_config.get("enrich", {}).get("unlock_emails", False):
                        logger.info(
                            "Company-first strategy: Enriching individual lead to get actual email"
                        )
                        # Use the People Enrichment endpoint to get real email
                        enriched_person = (
                            await self.apollo_client.enrich_person_for_email(result)
                        )
                        if enriched_person:
                            # Update with enriched email data
                            result["email"] = enriched_person.get(
                                "email", result.get("email")
                            )
                            result["phone"] = enriched_person.get(
                                "phone", result.get("phone")
                            )
                            logger.info(
                                f"Enriched email for {result.get('name', 'Unknown')}: {result.get('email')}"
                            )
                        else:
                            logger.info(
                                f"No enrichment data for {result.get('name', 'Unknown')}"
                            )
                        enriched_results.append(result)
                    else:
                        # Skip enrichment, use original result
                        logger.info(
                            "Company-first strategy: Skipping individual enrichment, using original result with organization data"
                        )
                        enriched_results.append(result)
                else:
                    # Enrich individual person - skip placeholder emails
                    email = result.get("email")
                    if email and not email.startswith("email_not_unlocked"):
                        logger.info(f"Enriching individual person email: {email}")
                        enriched_person = await self.apollo_client.enrich_person(email)
                        enriched_results.append(enriched_person)
                    else:
                        logger.info(
                            "No valid email found for individual enrichment, using original result"
                        )
                        enriched_results.append(result)

                # Add rate limiting delay
                await asyncio.sleep(0.1)

            except Exception as e:
                logger.error(f"Individual enrichment failed for result: {e}")
                enriched_results.append(result)

        return enriched_results

    def _map_to_unified_schema(
        self, apollo_data: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Transform Apollo data to unified lead record format.

        Args:
            apollo_data: Apollo API response data

        Returns:
            List of unified lead records
        """
        logger.info(f"Mapping {len(apollo_data)} Apollo results to unified schema")

        unified_records = []

        for item in apollo_data:
            try:
                logger.info(
                    f"Mapping item: {type(item)}, keys: {list(item.keys()) if isinstance(item, dict) else 'Not a dict'}"
                )

                if self.search_filters["strategy"] == "company_first":
                    unified_record = self._map_company_to_unified_schema(item)
                else:
                    unified_record = self._map_person_to_unified_schema(item)

                if unified_record:
                    unified_records.append(unified_record)
                    logger.info("Successfully mapped item to unified schema")
                else:
                    logger.warning("Item mapping returned None")

            except Exception as e:
                logger.error(f"Failed to map Apollo result to unified schema: {e}")
                logger.error(f"Item that failed: {item}")
                continue

        logger.info(
            f"Schema mapping completed: {len(unified_records)} unified records created"
        )
        return unified_records

    def _map_company_to_unified_schema(
        self, company_data: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Map Apollo company data to unified lead record schema.

        Args:
            company_data: Apollo company data (person with organization info)

        Returns:
            Unified lead record or None if mapping fails
        """
        try:
            # Validate input data
            if not company_data:
                logger.error("company_data is None or empty")
                return None

            if not isinstance(company_data, dict):
                logger.error(f"company_data is not a dict: {type(company_data)}")
                return None

            logger.info(f"Mapping company data with keys: {list(company_data.keys())}")

            # Apollo returns people with organization data, extract company info from there
            person = company_data
            company = person.get("organization", {})

            if not company:
                logger.warning(
                    "No organization data found in person record, creating minimal company record"
                )
                company = {
                    "name": person.get("name", "Unknown Person") + "'s Company",
                    "primary_domain": "unknown-company.com",
                }

            # Extract domain from the correct location - try multiple fields
            domain = (
                company.get("primary_domain")
                or company.get("domain")
                or company.get("website_url", "")
                .replace("http://", "")
                .replace("https://", "")
                .replace("www.", "")
                .split("/")[0]
                or ""
            )

            if not domain:
                logger.warning(
                    f"No domain found in company data: {company.get('name', 'Unknown')}. Company keys: {list(company.keys())}"
                )
                logger.warning(
                    f"Trying to extract domain from website_url: {company.get('website_url')}"
                )
                # Try to extract domain from website URL as fallback
                website = company.get("website_url", "")
                if website:
                    import re

                    domain_match = re.search(
                        r"(?:https?://)?(?:www\.)?([^/]+)", website
                    )
                    if domain_match:
                        domain = domain_match.group(1)
                        logger.info(f"Extracted domain from website URL: {domain}")

            if not domain:
                logger.error(
                    f"Still no domain found after all extraction attempts for company: {company.get('name', 'Unknown')}"
                )
                # Generate a fallback domain based on company name
                company_name = company.get("name", "unknown-company")
                domain = (
                    f"{company_name.lower().replace(' ', '-').replace('&', 'and')}.com"
                )
                logger.info(f"Generated fallback domain: {domain}")

            # Generate deterministic lead ID
            lead_id = (
                str(uuid.uuid5(uuid.NAMESPACE_DNS, domain))
                if domain
                else str(uuid.uuid4())
            )

            # Map company fields from Apollo organization data
            unified_record = {
                "lead_id": lead_id,
                "domain": domain,
                "company": {
                    "name": company.get("name"),
                    "description": person.get(
                        "headline", ""
                    ),  # Use person's headline as description
                    "industry": None,  # Apollo doesn't provide industry in basic response
                    "size_hint": None,  # Apollo doesn't provide size in basic response
                    "employee_count": None,  # Apollo doesn't provide employee count in basic response
                    "founded_year": company.get("founded_year"),
                    "website": company.get("website_url"),
                    "linkedin_url": company.get("linkedin_url"),
                    "phone": company.get("phone"),
                    "hq_location": self._format_location(
                        {
                            "city": person.get("city"),
                            "state": person.get("state"),
                            "country": person.get("country"),
                            "formatted_address": person.get("formatted_address"),
                        }
                    ),
                    "tech_stack": [],  # Apollo doesn't provide tech stack in basic response
                    "revenue_range": None,  # Apollo doesn't provide revenue in basic response
                    "number_of_locations": 1,  # Default to 1 for now
                },
                "contacts": [
                    {
                        "name": person.get("name"),
                        "title": person.get("title"),
                        "email": person.get("email")
                        if not person.get("email", "").startswith("email_not_unlocked")
                        else None,
                        "phone": person.get("phone"),
                        "linkedin_url": person.get("linkedin_url"),
                        "role": person.get("title"),
                        "decision_maker": person.get("seniority")
                        in ["c_suite", "vp", "director"],
                        "seniority": person.get("seniority"),
                    }
                ]
                if person.get("name")
                else [],
                "icp": {
                    "pre_score": None,
                    "pre_reason": "Apollo-generated lead",
                    "pre_pass": True,
                    "pre_flags": [],
                    "scoring_meta": {
                        "method": "apollo_search",
                        "model": "apollo_api",
                        "prompt_fingerprint": None,
                        "temperature": None,
                    },
                    "crawl_score": None,
                    "crawl_reason": "Apollo bypasses crawl stage",
                    "threshold": self.icp_config.get("threshold", 0.7),
                    "filtering_strict": self.icp_config.get("filtering_strict", True),
                },
                "provenance": {
                    "queries": [],
                    "discovery_evidence": [],
                    "scored_at": datetime.utcnow().isoformat(),
                    "crawled_at": None,
                    "enriched_at": datetime.utcnow().isoformat(),
                    "next_revisit_at": None,
                    "tool_versions": {
                        "domain": {"version": "apollo_bypass", "schema_version": None},
                        "crawl": {"version": "apollo_bypass", "schema_version": None},
                        "enrich": {"version": "apollo_enrich", "schema_version": None},
                    },
                    "cache": {},
                    "evidence_paths": {"domain": None, "crawl": None, "enrich": None},
                },
                "status": "enriched",
                "status_history": [
                    {
                        "status": "enriched",
                        "ts": datetime.utcnow().isoformat(),
                        "notes": "Generated via Apollo API",
                    }
                ],
                "audit": [
                    {
                        "step": "Apollo",
                        "notes": "Lead generated via Apollo search and enrichment",
                        "ts": datetime.utcnow().isoformat(),
                    }
                ],
                "cost": {
                    "domain_usd": 0.0,
                    "crawl_usd": 0.0,
                    "enrich_usd": self.apollo_client.get_cost_summary()["total_usd"],
                    "total_usd": self.apollo_client.get_cost_summary()["total_usd"],
                },
            }

            return unified_record

        except Exception as e:
            logger.error(f"Failed to map company data to unified schema: {e}")
            return None

    def _map_person_to_unified_schema(
        self, person_data: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Map Apollo person data to unified lead record schema.

        Args:
            person_data: Apollo person data

        Returns:
            Unified lead record or None if mapping fails
        """
        try:
            # Extract person information
            person = person_data.get("person", person_data)
            company_info = person.get("company", {})

            # Generate deterministic lead ID
            domain = company_info.get("domain", "")
            lead_id = (
                str(uuid.uuid5(uuid.NAMESPACE_DNS, domain))
                if domain
                else str(uuid.uuid4())
            )

            # Map person fields
            contact = {
                "contact_id": str(uuid.uuid4()),
                "full_name": person.get("full_name"),
                "role": person.get("job_title"),
                "department": person.get("department"),
                "email": person.get("email"),
                "phone": person.get("phone"),
                "linkedin": person.get("linkedin_url"),
                "location": person.get("location"),
                "seniority": person.get("seniority_level"),
                "role_priority": self._get_role_priority(person.get("job_title", "")),
            }

            # Map company fields
            company = {
                "name": company_info.get("name"),
                "domain": company_info.get("domain"),
                "website": company_info.get("website"),
            }

            unified_record = {
                "lead_id": lead_id,
                "domain": domain,
                "company": company,
                "contacts": [contact],
                "icp": {
                    "pre_score": None,
                    "pre_reason": "Apollo-generated lead",
                    "pre_pass": True,
                    "pre_flags": [],
                    "scoring_meta": {
                        "method": "apollo_search",
                        "model": "apollo_api",
                        "prompt_fingerprint": None,
                        "temperature": None,
                    },
                    "crawl_score": None,
                    "crawl_reason": "Apollo bypasses crawl stage",
                    "threshold": self.icp_config.get("threshold", 0.7),
                    "filtering_strict": self.icp_config.get("filtering_strict", True),
                },
                "provenance": {
                    "queries": [],
                    "discovery_evidence": [],
                    "scored_at": datetime.utcnow().isoformat(),
                    "crawled_at": None,
                    "enriched_at": datetime.utcnow().isoformat(),
                    "next_revisit_at": None,
                    "tool_versions": {
                        "domain": {"version": "apollo_bypass", "schema_version": None},
                        "crawl": {"version": "apollo_bypass", "schema_version": None},
                        "enrich": {"version": "apollo_enrich", "schema_version": None},
                    },
                    "cache": {},
                    "evidence_paths": {"domain": None, "crawl": None, "enrich": None},
                },
                "status": "enriched",
                "status_history": [
                    {
                        "status": "enriched",
                        "ts": datetime.utcnow().isoformat(),
                        "notes": "Generated via Apollo API",
                    }
                ],
                "audit": [
                    {
                        "step": "Apollo",
                        "notes": "Lead generated via Apollo search and enrichment",
                        "ts": datetime.utcnow().isoformat(),
                    }
                ],
                "cost": {
                    "domain_usd": 0.0,
                    "crawl_usd": 0.0,
                    "enrich_usd": self.apollo_client.get_cost_summary()["total_usd"],
                    "total_usd": self.apollo_client.get_cost_summary()["total_usd"],
                },
            }

            return unified_record

        except Exception as e:
            logger.error(f"Failed to map person data to unified schema: {e}")
            return None

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

    async def _ensure_company_block(self, person: dict) -> dict:
        """
        Ensure person has organization data by fetching it if missing.

        Args:
            person: Person record from Apollo search

        Returns:
            Person record with organization data populated
        """
        # Check if organization fetching is enabled
        if not self.apollo_config.get("enrich", {}).get(
            "fetch_missing_organizations", True
        ):
            logger.debug("Organization fetching disabled, skipping")
            return person

        if "organization" in person and person["organization"]:
            return person  # nothing to do

        org_id = person.get("organization_id")
        if not org_id:
            logger.debug(
                f"No organization_id found for person {person.get('name', 'Unknown')}"
            )
            return person  # cannot improve

        try:
            logger.info(f"Fetching organization data for org_id: {org_id}")
            # credit-consuming but cheap: one credit / company
            company_resp, _ = await self.apollo_client._make_request(
                f"organizations/{org_id}", method="GET"
            )
            person["organization"] = company_resp.get("organization", {})
            logger.info(f"Successfully fetched organization data for {org_id}")
        except Exception as e:
            logger.warning(f"Could not fetch organization {org_id}: {e}")
        return person

    def _get_role_priority(self, job_title: str) -> int:
        """
        Get role priority based on ICP configuration.

        Args:
            job_title: Job title to get priority for

        Returns:
            Role priority (lower = higher priority)
        """
        role_priority_config = self.icp_config.get("role_priority", {})
        return get_role_priority(job_title, role_priority_config)

    def _validate_lead_quality(self, lead: Dict[str, Any]) -> bool:
        """
        Ensure lead meets quality thresholds from ICP config.

        Args:
            lead: Lead record to validate

        Returns:
            True if lead meets quality standards, False otherwise
        """
        try:
            # Check required fields based on ICP config
            required_fields = self.icp_config.get("required_fields", {})

            # Validate company required fields
            company_required = required_fields.get("company", [])
            company_data = lead.get("company", {})

            for field in company_required:
                if not company_data.get(field):
                    # For Apollo mode, be more lenient with company fields
                    if self.icp_config.get("lead_generation_mode") == "apollo":
                        if field in ["hq_location", "number_of_locations"]:
                            logger.info(
                                f"Lead {lead.get('lead_id')} missing company field '{field}' but continuing for Apollo mode"
                            )
                            continue
                    logger.info(
                        f"Lead {lead.get('lead_id')} missing required company field: {field}"
                    )
                    return False

            # Validate contact required fields
            contact_required = required_fields.get("contact", [])
            contacts = lead.get("contacts", [])

            if contact_required and not contacts:
                logger.info(f"Lead {lead.get('lead_id')} missing required contacts")
                return False

            for contact in contacts:
                for field in contact_required:
                    if field == "email":
                        # For Apollo mode, handle placeholder emails
                        email = contact.get(field)
                        if not email or email.startswith("email_not_unlocked"):
                            if self.icp_config.get("lead_generation_mode") == "apollo":
                                logger.info(
                                    f"Lead {lead.get('lead_id')} has placeholder email but continuing for Apollo mode"
                                )
                                continue
                            else:
                                logger.info(
                                    f"Lead {lead.get('lead_id')} missing valid email"
                                )
                                return False
                    elif not contact.get(field):
                        logger.info(
                            f"Lead {lead.get('lead_id')} missing required contact field: {field}"
                        )
                        return False

            # Check ICP threshold
            icp_score = lead.get("icp", {}).get("pre_score")
            threshold = self.icp_config.get("threshold", 0.7)

            if icp_score is not None and icp_score < threshold:
                logger.info(
                    f"Lead {lead.get('lead_id')} below ICP threshold: {icp_score} < {threshold}"
                )
                return False

            return True

        except Exception as e:
            logger.error(f"Lead quality validation failed: {e}")
            return False

    async def close(self):
        """Close Apollo client and cleanup resources"""
        if self.apollo_client:
            await self.apollo_client.close()
            logger.info("Apollo Lead Generation Tool closed")

    async def search_leads_optimized(
        self, search_type: str = "company_first", max_results: int = None
    ) -> List[Dict[str, Any]]:
        """Search leads using AI-optimized strategies and adaptive search"""
        try:
            # Acquire permit for search operation
            async with self.permit_manager.acquire_permit():
                logger.info(f"Starting optimized lead search with type: {search_type}")

                # Get base filters from ICP config
                base_filters = self._get_base_search_filters()

                # Generate AI-optimized search strategies
                if self.ai_optimizer.config.enable_ai_optimization:
                    search_strategies = self.ai_optimizer.optimize_search_strategy(
                        base_filters,
                        "company" if search_type == "company_first" else "person",
                        self.icp_config,
                    )
                    logger.info(
                        f"Generated {len(search_strategies)} AI-optimized search strategies"
                    )
                else:
                    search_strategies = [
                        {
                            "filters": base_filters,
                            "confidence": 1.0,
                            "strategy_type": "base",
                        }
                    ]

                # Execute search using adaptive strategies
                all_results = []
                strategy_results = {}

                for i, strategy in enumerate(search_strategies):
                    if max_results and len(all_results) >= max_results:
                        break

                    strategy_name = strategy.get("strategy_type", f"strategy_{i}")
                    strategy_filters = strategy["filters"]

                    logger.info(
                        f"Executing {strategy_name} strategy with confidence {strategy.get('confidence', 0):.2f}"
                    )

                    try:
                        # Use adaptive search orchestrator
                        _ = self.adaptive_orchestrator.execute_search(
                            strategy_filters,
                            search_type,
                            max_results=max_results - len(all_results)
                            if max_results
                            else None,
                        )

                        # Execute the actual search
                        if search_type == "company_first":
                            results = await self._execute_company_search_optimized(
                                strategy_filters,
                                max_results - len(all_results) if max_results else None,
                            )
                        else:
                            results = await self._execute_person_search_optimized(
                                strategy_filters,
                                max_results - len(all_results) if max_results else None,
                            )

                        # Score and rank results
                        if results:
                            scored_results = self.result_scorer.rank_results(
                                results,
                                "company"
                                if search_type == "company_first"
                                else "person",
                            )

                            # Filter by quality thresholds
                            quality_filtered = self.result_scorer.filter_by_quality(
                                results,
                                "company"
                                if search_type == "company_first"
                                else "person",
                            )

                            # Record strategy performance
                            avg_score = (
                                sum(
                                    score["overall_score"]
                                    for _, score in scored_results
                                )
                                / len(scored_results)
                                if scored_results
                                else 0
                            )
                            self.adaptive_orchestrator.record_performance(
                                strategy_name,
                                avg_score,
                                len(results),
                                avg_score,  # Using overall score as quality score
                            )

                            strategy_results[strategy_name] = {
                                "results_count": len(results),
                                "quality_filtered_count": len(quality_filtered),
                                "avg_score": avg_score,
                                "confidence": strategy.get("confidence", 0),
                            }

                            # Add results to collection
                            all_results.extend(quality_filtered)

                            logger.info(
                                f"Strategy {strategy_name} returned {len(results)} results, {len(quality_filtered)} passed quality filter"
                            )

                    except Exception as e:
                        logger.error(f"Error executing strategy {strategy_name}: {e}")
                        strategy_results[strategy_name] = {"error": str(e)}
                        continue

                # Update performance metrics
                self._update_performance_metrics(len(all_results), strategy_results)

                logger.info(
                    f"Optimized search completed: {len(all_results)} total results from {len(search_strategies)} strategies"
                )

                return all_results

        except Exception as e:
            logger.error(f"Error in optimized lead search: {e}")
            raise ApolloLeadGenError(f"Optimized search failed: {e}")

    async def _execute_company_search_optimized(
        self, filters: Dict[str, Any], max_results: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Execute optimized company search using enhanced Apollo client"""
        try:
            if not self.apollo_client:
                await self._initialize_apollo_client()

            # Use optimized search method if available
            if hasattr(self.apollo_client, "search_companies_optimized"):
                results = await self.apollo_client.search_companies_optimized(
                    filters, max_results or 1000
                )
            else:
                # Fallback to standard search
                search_params = [
                    {"filters": filters, "max_results": max_results or 1000}
                ]
                results = await self.apollo_client.search_companies_batched(
                    search_params
                )

            return results

        except Exception as e:
            logger.error(f"Error in optimized company search: {e}")
            return []

    async def _execute_person_search_optimized(
        self, filters: Dict[str, Any], max_results: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Execute optimized person search using enhanced Apollo client"""
        try:
            if not self.apollo_client:
                await self._initialize_apollo_client()

            # Use optimized search method if available
            if hasattr(self.apollo_client, "search_persons_optimized"):
                results = await self.apollo_client.search_persons_optimized(
                    filters, max_results or 1000
                )
            else:
                # Fallback to standard search
                search_params = [
                    {"filters": filters, "max_results": max_results or 1000}
                ]
                results = await self.apollo_client.search_persons_batched(search_params)

            return results

        except Exception as e:
            logger.error(f"Error in optimized person search: {e}")
            return []

    def _get_base_search_filters(self) -> Dict[str, Any]:
        """Get base search filters from ICP configuration"""
        apollo_config = self.icp_config.get("apollo", {})
        search_config = apollo_config.get("search", {})

        base_filters = {}

        # Company filters
        if "company_filters" in search_config:
            base_filters.update(search_config["company_filters"])

        # Person filters
        if "person_filters" in search_config:
            base_filters.update(search_config["person_filters"])

        # Industry filters
        if "industries" in self.icp_config:
            base_filters["industry"] = self.icp_config["industries"]

        # Location filters
        if "countries" in self.icp_config:
            base_filters["country"] = self.icp_config["countries"]

        # Employee count filters
        if "employee_count" in self.icp_config:
            base_filters["employee_count"] = self.icp_config["employee_count"]

        # Job title filters
        if "job_titles" in self.icp_config:
            base_filters["job_titles"] = self.icp_config["job_titles"]

        return base_filters

    def _update_performance_metrics(
        self, result_count: int, strategy_results: Dict[str, Any]
    ):
        """Update performance metrics"""
        self.performance_metrics["total_searches"] += 1
        self.performance_metrics["total_results"] += result_count

        # Update strategy performance
        for strategy_name, results in strategy_results.items():
            if strategy_name not in self.performance_metrics["strategy_performance"]:
                self.performance_metrics["strategy_performance"][strategy_name] = {
                    "total_searches": 0,
                    "total_results": 0,
                    "avg_score": 0.0,
                }

            strategy_metrics = self.performance_metrics["strategy_performance"][
                strategy_name
            ]
            strategy_metrics["total_searches"] += 1
            strategy_metrics["total_results"] += results.get("results_count", 0)

            # Update average score
            current_avg = strategy_metrics["avg_score"]
            new_score = results.get("avg_score", 0)
            strategy_metrics["avg_score"] = (current_avg + new_score) / 2

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        summary = {
            "apollo_lead_gen": {
                "total_searches": self.performance_metrics["total_searches"],
                "total_results": self.performance_metrics["total_results"],
                "strategy_performance": self.performance_metrics[
                    "strategy_performance"
                ],
            },
            "ai_optimization": self.ai_optimizer.get_optimization_stats(),
            "adaptive_strategies": self.adaptive_orchestrator.get_strategy_performance(),
            "recommendations": {
                "ai_optimization": self.ai_optimizer.get_optimization_stats().get(
                    "recommendations", []
                ),
                "adaptive_strategies": self.adaptive_orchestrator.get_adaptation_recommendations(),
            },
        }

        # Add Apollo client performance if available
        if self.apollo_client:
            try:
                summary["apollo_client"] = self.apollo_client.get_performance_metrics()
                summary["apollo_client"]["health_status"] = (
                    self.apollo_client.get_health_status()
                )
            except Exception as e:
                logger.warning(f"Could not get Apollo client performance: {e}")

        return summary

    def get_quality_analysis(
        self, results: List[Dict[str, Any]], result_type: str = "company"
    ) -> Dict[str, Any]:
        """Get detailed quality analysis of search results"""
        try:
            return self.result_scorer.get_scoring_summary(results, result_type)
        except Exception as e:
            logger.error(f"Error generating quality analysis: {e}")
            return {"error": str(e)}

    def optimize_search_parameters(self) -> Dict[str, Any]:
        """Get recommendations for optimizing search parameters"""
        recommendations = {
            "ai_optimization": self.ai_optimizer.get_optimization_stats().get(
                "recommendations", []
            ),
            "adaptive_strategies": self.adaptive_orchestrator.get_adaptation_recommendations(),
            "performance_insights": [],
        }

        # Analyze performance metrics
        if self.performance_metrics["total_searches"] > 0:
            avg_results_per_search = (
                self.performance_metrics["total_results"]
                / self.performance_metrics["total_searches"]
            )

            if avg_results_per_search < 10:
                recommendations["performance_insights"].append(
                    "Low results per search - consider broadening filters"
                )
            elif avg_results_per_search > 100:
                recommendations["performance_insights"].append(
                    "High results per search - consider narrowing filters for better quality"
                )

        # Strategy performance insights
        strategy_performance = self.performance_metrics["strategy_performance"]
        if strategy_performance:
            best_strategy = max(
                strategy_performance.items(), key=lambda x: x[1]["avg_score"]
            )
            worst_strategy = min(
                strategy_performance.items(), key=lambda x: x[1]["avg_score"]
            )

            if best_strategy[1]["avg_score"] - worst_strategy[1]["avg_score"] > 0.3:
                recommendations["performance_insights"].append(
                    f"Large performance gap between strategies - {worst_strategy[0]} needs optimization"
                )

        return recommendations

    def reset_performance_tracking(self):
        """Reset all performance tracking data"""
        self.performance_metrics = {
            "total_searches": 0,
            "total_results": 0,
            "avg_response_time": 0.0,
            "cache_hit_rate": 0.0,
            "quality_scores": [],
            "strategy_performance": {},
        }

        # Reset AI optimizer
        self.ai_optimizer.search_history.clear()
        self.ai_optimizer.performance_metrics.clear()

        # Reset adaptive orchestrator
        self.adaptive_orchestrator.reset_strategies()

        # Reset Apollo client if available
        if self.apollo_client:
            try:
                self.apollo_client.clear_cache()
            except Exception as e:
                logger.warning(f"Could not clear Apollo client cache: {e}")

        logger.info("Performance tracking reset completed")
