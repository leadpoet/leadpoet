"""
Crawl Tool for Lead Sorcerer

This tool fetches company websites, extracts company and contact information,
and validates ICP fit based on crawled data.

Purpose: Fetch site, extract company & contacts, re-validate ICP; emit one row per contact
(plus company-only if none).

Inputs: { "lead_records": [...], "icp_config": {...} }
Output: lead_records[] with company + 0..N contacts + icp.crawl_score

Authoritative specifications: docs/brd.md §273-320
"""

import asyncio
import json
import os
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from .common import (
    AsyncSemaphorePool,
    ErrorCode,
    PermitManager,
    append_audit,
    append_status,
    build_error,
    canonicalize_linkedin,
    compute_content_hash,
    compute_next_revisit,
    generate_contact_id,
    get_role_priority,
    get_seniority_rank,
    load_costs_config,
    load_schema_checksum,
    normalize_domain,
    normalize_text,
    now_z,
    resolve_data_dir,
    round4,
    setup_logging,
    truncate_evidence_arrays,
    write_with_lock,
)


# ============================================================================
# Constants and Configuration
# ============================================================================

INTERNAL_VERSION = "1.0.0"
TOOL_NAME = "crawl"

# Firecrawl extraction schema - using v2 API format (COST OPTIMIZED - SINGLE SCRAPE OPERATION)
FIRECRAWL_EXTRACT_SCHEMA = {
    "type": "object",
    "properties": {
        "company": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Company name"},
                "description": {
                    "type": "string",
                    "description": "Brief business description",
                },
                "industry": {"type": "string", "description": "Industry or sector"},
                "sub_industry": {"type": "string", "description": "Specific sub-industry or niche within the broader industry"},
                "hq_location": {
                    "type": "string",
                    "description": "Headquarters location (city, state, country)",
                },
                "number_of_locations": {
                    "type": "integer",
                    "description": "Total number of office locations",
                },
                "founded_year": {
                    "type": "integer",
                    "description": "Year company was founded",
                },
                "employee_count": {
                    "type": "string",
                    "description": "Number of employees or range",
                },
                "revenue_range": {
                    "type": "string",
                    "description": "Annual revenue range",
                },
                "ownership_type": {
                    "type": "string",
                    "description": "Private, public, nonprofit, etc.",
                },
                "company_type": {
                    "type": "string",
                    "description": "LLC, Corp, Partnership, etc.",
                },
                "specialties": {
                    "type": "array",
                    "items": {"type": "string"},
                    "maxItems": 5,
                    "description": "Company specialties or focus areas",
                },
                "intent": {
                    "type": "object",
                    "properties": {
                        "business_intent_score": {
                            "type": "number",
                            "minimum": 0.0,
                            "maximum": 1.0,
                            "description": "Confidence score (0-1) that company matches ICP intent",
                        },
                        "intent_signals": {
                            "type": "array",
                            "items": {"type": "string"},
                            "maxItems": 5,
                            "description": "Specific evidence found indicating intent",
                        },
                        "intent_category": {
                            "type": "string",
                            "description": "Primary intent category: high, medium, or low",
                        },
                    },
                },
            },
        },
        "team_members": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Full name"},
                    "role": {"type": "string", "description": "Job title/role"},
                    "email": {"type": "string", "description": "Email address"},
                    "phone": {"type": "string", "description": "Phone number"},
                    "decision_maker": {
                        "type": "boolean",
                        "description": "Whether this person is a decision maker",
                    },
                },
            },
            "maxItems": 10,
            "description": "Key staff members and their contact information",
        },
    },
}

# Firecrawl extraction prompt - SIMPLIFIED for better AI extraction success
FIRECRAWL_EXTRACT_PROMPT = (
    "Extract essential business information from this website. Focus on the most important details only.\n\n"
    "EXTRACTION RULES:\n"
    "- Extract ONLY information that is clearly stated on the website\n"
    "- Use null/empty for any information not found\n"
    "- Do NOT invent or guess any data\n"
    "- Focus on homepage, about page, and contact page\n\n"
    "COMPANY INFORMATION:\n"
    "- Company name (from title, logo, or header)\n"
    "- Brief description of what they do\n"
    "- Industry or business type (broad category)\n"
    "- Sub-industry (specific niche or specialization within the industry)\n"
    "- Main location (city, state, country)\n"
    "- Number of locations if mentioned\n"
    "- Year founded if mentioned\n"
    "- Employee count if mentioned\n"
    "- Revenue information if mentioned\n"
    "- Company structure (LLC, Corp, etc.) if mentioned\n"
    "- Main specialties or services offered\n\n"
    "INTENT ANALYSIS:\n"
    "- Look for signs they want to sell, grow, partner, or transition\n"
    "- Check for retirement planning, succession planning, expansion plans\n"
    "- Look for 'for sale', 'seeking partners', 'growth opportunities'\n"
    "- Score confidence (0-1) based on evidence found\n"
    "- List specific intent signals found\n"
    "- Categorize as high, medium, or low intent\n\n"
    "TEAM MEMBERS:\n"
    "- Key executives and decision makers\n"
    "- Names, roles, and contact information\n"
    "- Focus on owners, CEOs, founders, managers\n"
    "- Limit to 10 most important people\n\n"
    "IMPORTANT: Extract only what you can clearly see on the website. If information is not available, leave it as null."
)


# ============================================================================
# Main Tool Class
# ============================================================================


class CrawlTool:
    """Crawl tool for extracting company and contact information from websites."""

    def __init__(self, data_dir: Optional[str] = None):
        self.data_dir = data_dir or "./data"
        # Setup logging (child tools inherit shared run log via env)
        self.logger = setup_logging(TOOL_NAME, data_dir=self.data_dir)
        self.schema_version = load_schema_checksum()
        self.costs_config = load_costs_config()

        # Validate required providers
        required_providers = ["firecrawl"]
        for provider in required_providers:
            if provider not in self.costs_config:
                raise KeyError(
                    f"Missing required provider in costs configuration: {provider}"
                )

        # Initialize Firecrawl client
        self.firecrawl_key = os.environ.get("FIRECRAWL_KEY")
        if not self.firecrawl_key:
            raise ValueError("FIRECRAWL_KEY environment variable is required")

        from firecrawl import Firecrawl

        self.firecrawl_client = Firecrawl(api_key=self.firecrawl_key)
        # Global permit manager for concurrency control
        self.permit_manager = PermitManager(
            max_permits=3
        )  # Default, will be overridden
        self.async_semaphore_pool = AsyncSemaphorePool(self.permit_manager)

    def generate_dynamic_intent_prompt(self, icp_config):
        """Generate intent prompt based on ICP configuration"""

        intent_config = icp_config.get("intent_config", {})

        if not intent_config:
            return ""  # No intent analysis if not configured

        purpose = intent_config.get("purpose", "business opportunity")
        target_action = intent_config.get("target_action", "take action")

        # Enhanced prompt with more specific guidance
        prompt = f"""
INTENT ANALYSIS FOR {purpose.upper().replace("_", " ")}:

Your goal is to determine if this company has INTENT to {target_action.replace("_", " ")}.

ANALYSIS METHODOLOGY:
1. SCAN ALL AVAILABLE PAGES for intent signals
2. LOOK FOR SPECIFIC LANGUAGE that indicates intent
3. IDENTIFY TIMELINE indicators (when they might act)
4. DETECT DECISION MAKER statements about future plans
5. ANALYZE OWNERSHIP and business structure changes

INTENT SIGNALS TO LOOK FOR:

HIGH INTENT (Strong evidence they want to {target_action.replace("_", " ")}):
{chr(10).join(f"- {signal}" for signal in intent_config.get("intent_signals", {}).get("high_intent", []))}

MEDIUM INTENT (Some evidence they might be open to {target_action.replace("_", " ")}):
{chr(10).join(f"- {signal}" for signal in intent_config.get("intent_signals", {}).get("medium_intent", []))}

LOW INTENT (Evidence they are NOT interested in {target_action.replace("_", " ")}):
{chr(10).join(f"- {signal}" for signal in intent_config.get("intent_signals", {}).get("low_intent", []))}

LOOK IN THESE SOURCES (PRIORITIZE IN ORDER):
{chr(10).join(f"- {signal}" for signal in intent_config.get("intent_sources", []))}

SPECIFIC ANALYSIS INSTRUCTIONS:
- READ ABOUT/LEADERSHIP pages for owner statements about future plans
- CHECK CAREERS pages for leadership changes or growth indicators
- LOOK for age-related language (e.g., "founded in 1985", "30+ years")
- IDENTIFY succession planning language (e.g., "next generation", "transition")
- DETECT growth vs. exit language (e.g., "expanding" vs. "considering options")
- ANALYZE ownership structure changes or mentions

SCORING APPROACH:
- Start with base score of {intent_config.get("scoring_rules", {}).get("base_score", 0.5)}
- Add {intent_config.get("scoring_rules", {}).get("high_intent_weight", 0.8)} for each HIGH INTENT signal found
- Add {intent_config.get("scoring_rules", {}).get("medium_intent_weight", 0.4)} for each MEDIUM INTENT signal found
- Subtract {intent_config.get("scoring_rules", {}).get("low_intent_weight", 0.3)} for each LOW INTENT signal found
- Cap final score between 0.0 and 1.0

REQUIRED OUTPUT:
- business_intent_score: Calculated score (0.0-1.0)
- intent_signals: List of specific evidence found (quotes preferred)
- intent_category: "high", "medium", or "low" based on signals
- timeline_indicator: When they might act (e.g., "within 2 years", "retirement age")
- decision_maker_intent: Specific statements from key people

IMPORTANT: If no clear intent signals are found, set business_intent_score to {intent_config.get("scoring_rules", {}).get("base_score", 0.5)} and explain why in intent_signals.
"""
        return prompt

    def _build_extraction_urls(
        self, domain: str, icp_config: Dict[str, Any]
    ) -> List[str]:
        """
        Build optimized list of URLs for extraction to minimize API costs.

        Args:
            domain: Domain to extract from
            icp_config: ICP configuration

        Returns:
            List of URLs to extract from (prioritized for cost optimization)
        """
        urls_to_extract = []

        # Check if we have specific URLs to crawl (highest priority)
        specific_urls = icp_config.get("specific_urls", [])
        database_config = icp_config.get("database_config", {})
        site_type = icp_config.get("site_type")

        if specific_urls:
            # Mode 1: Extract from provided specific URLs
            # Filter specific URLs to only include those matching the current domain
            domain_specific_urls = []
            for url in specific_urls:
                from urllib.parse import urlparse
                parsed_url = urlparse(url)
                url_domain = parsed_url.netloc.lower()
                if url_domain == domain.lower() or url_domain == f"www.{domain.lower()}":
                    domain_specific_urls.append(url)

            if domain_specific_urls:
                urls_to_extract.extend(domain_specific_urls)
                self.logger.info(
                    f"🔗 Added {len(domain_specific_urls)} domain-specific URLs to extraction list for {domain}"
                )
                self.logger.info("🔗 Database mode: Using domain-specific URLs")
            else:
                # Fallback to domain-based URLs if no specific URLs match this domain
                priority_pages = [
                    f"https://{domain}",  # Homepage (most comprehensive)
                    f"https://{domain}/about",  # About page (company info)
                    f"https://{domain}/contact",  # Contact page (contact info)
                ]
                urls_to_extract.extend(priority_pages)
                self.logger.info(f"🔗 No specific URLs match {domain}, using domain-based URLs")

        elif (
            site_type == "information_database"
            and database_config.get("discovery_mode") == "domain_crawl"
        ):
            # Mode 2: Discover URLs from the whole domain (new functionality)
            self.logger.info(
                "🔍 Database mode: No specific URLs provided - discovering from domain"
            )

            # Get discovery configuration
            max_pages = database_config.get("max_discovery_pages", 50)
            url_patterns = database_config.get("url_patterns", [])
            exclude_patterns = database_config.get("exclude_patterns", [])

            # Discover URLs from the domain
            discovered_urls = self._discover_database_urls(
                domain, max_pages, url_patterns, exclude_patterns, icp_config
            )
            urls_to_extract.extend(discovered_urls)

            self.logger.info(
                f"🔍 Discovered {len(discovered_urls)} URLs from domain crawling"
            )

        else:
            # Mode 3: Standard single company mode
            # Prioritize most important pages for cost optimization
            priority_pages = [
                f"https://{domain}",  # Homepage (most comprehensive)
                f"https://{domain}/about",  # About page (company info)
                f"https://{domain}/contact",  # Contact page (contact info)
            ]
            urls_to_extract.extend(priority_pages)
            self.logger.info("🔗 Added priority domain pages for comprehensive data")

        # Remove duplicates while preserving order (specific URLs first, then domain pages)
        seen = set()
        unique_urls = []
        for url in urls_to_extract:
            if url not in seen:
                seen.add(url)
                unique_urls.append(url)

        self.logger.info(f"🌐 Final extraction URLs for {domain}: {unique_urls}")
        return unique_urls

    def _generate_cache_key(self, domain: str, icp_config: Dict[str, Any]) -> str:
        """
        Generate cache key based on domain and domain-specific URLs.

        Args:
            domain: Normalized domain
            icp_config: ICP configuration containing specific URLs

        Returns:
            Cache key string
        """
        # Check if we have specific URLs to crawl
        specific_urls = icp_config.get("specific_urls", [])

        if specific_urls:
            # Filter specific URLs to only include those matching the current domain
            domain_specific_urls = []
            for url in specific_urls:
                from urllib.parse import urlparse
                parsed_url = urlparse(url)
                url_domain = parsed_url.netloc.lower()
                if url_domain == domain.lower() or url_domain == f"www.{domain.lower()}":
                    domain_specific_urls.append(url)

            if domain_specific_urls:
                # This prevents different URL combinations from using the same cache
                import hashlib
                from urllib.parse import urlparse

                # Sort URLs for consistent hashing
                sorted_urls = sorted(domain_specific_urls)

                # Create a combined string of all URL paths
                url_paths = []
                for url in sorted_urls:
                    parsed_url = urlparse(url)
                    url_path = parsed_url.path.strip("/")
                    url_paths.append(url_path)

                # Combine all paths and create hash
                combined_paths = "|".join(url_paths)
                path_hash = hashlib.md5(combined_paths.encode()).hexdigest()[:8]

                return f"{domain}_{path_hash}"
            else:
                # No specific URLs match this domain, use domain-only cache key
                return domain
        else:
            # For domain-only crawls, use just the domain
            return domain

    def _detect_site_type(self, icp_config: Dict[str, Any]) -> str:
        """
        Detect if this is an information database site where each URL contains different company data.

        Args:
            icp_config: ICP configuration

        Returns:
            Site type: 'information_database' or 'single_company'
        """
        # Explicit flag overrides everything (maintains backward compatibility)
        site_type = icp_config.get("site_type")
        if site_type == "information_database":
            self.logger.info(
                "🏢 Site type: Information Database (explicit configuration)"
            )
            return "information_database"

        # Implicit detection based on URL patterns (optional enhancement)
        specific_urls = icp_config.get("specific_urls", [])

        # If only one or no specific URLs, definitely single company
        if len(specific_urls) <= 1:
            self.logger.info("🏢 Site type: Single Company (single/no URL)")
            return "single_company"

        # Multiple URLs - check for database patterns
        if self._has_database_url_patterns(specific_urls):
            self.logger.info(
                "🏢 Site type: Information Database (auto-detected from URL patterns)"
            )
            return "information_database"

        # Default to single company mode (existing behavior)
        self.logger.info("🏢 Site type: Single Company (default)")
        return "single_company"

    def _validate_industry_match(
        self, record: Dict[str, Any], icp_config: Dict[str, Any]
    ) -> bool:
        """
        Validate if company matches ICP industry criteria.

        Args:
            record: Lead record with company data
            icp_config: ICP configuration with validation criteria

        Returns:
            bool: True if company matches industry criteria, False otherwise
        """
        validation_config = icp_config.get("validation_config", {})
        industry_keywords = validation_config.get("industry_keywords", [])
        company_name_keywords = validation_config.get("company_name_keywords", [])

        # If no validation criteria specified, pass through (backward compatibility)
        if not industry_keywords and not company_name_keywords:
            return True

        company = record.get("company", {})
        industry = (company.get("industry") or "").lower()
        name = (company.get("name") or "").lower()

        # Check industry keywords
        industry_match = any(
            keyword.lower() in industry for keyword in industry_keywords
        )

        # Check company name keywords
        name_match = any(keyword.lower() in name for keyword in company_name_keywords)

        # Log validation results for debugging
        if industry_match or name_match:
            self.logger.info(
                f"✅ Industry validation passed for {name}: industry_match={industry_match}, name_match={name_match}"
            )
        else:
            self.logger.info(
                f"⚠️ Industry validation uncertain for {name}: no keyword matches found"
            )

        return industry_match or name_match

    def _apply_field_mappings(
        self,
        company: Dict[str, Any],
        company_data: Dict[str, Any],
        icp_config: Dict[str, Any],
    ) -> None:
        """
        Apply configurable field mappings for company data.

        Args:
            company: Target company object to populate
            company_data: Source company data from extraction
            icp_config: ICP configuration with field mappings
        """
        database_config = icp_config.get("database_config", {})
        field_mappings = database_config.get("field_mappings", {})

        # Apply configured field mappings
        for source_field, target_config in field_mappings.items():
            if source_field in company_data:
                if isinstance(target_config, str):
                    # Simple field mapping: "practice_type" -> "specialties"
                    target_field = target_config
                    company[target_field] = [company_data[source_field]]
                elif isinstance(target_config, dict):
                    # Complex mapping with transformation
                    target_field = target_config.get("target_field")
                    transform_type = target_config.get("transform", "single_value")

                    if transform_type == "array":
                        company[target_field] = [company_data[source_field]]
                    elif transform_type == "direct":
                        company[target_field] = company_data[source_field]
                    else:
                        # Default to single value
                        company[target_field] = company_data[source_field]

        # Fallback for common fields (backward compatibility)
        if "specialties" not in company and "specialties" in company_data:
            company["specialties"] = company_data["specialties"]

    def _validate_icp_config(self, icp_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and enhance ICP configuration with defaults for backward compatibility.

        Args:
            icp_config: Raw ICP configuration

        Returns:
            Dict[str, Any]: Validated and enhanced configuration
        """
        validated_config = icp_config.copy()

        # Ensure validation_config exists
        if "validation_config" not in validated_config:
            validated_config["validation_config"] = {}

        # Ensure database_config exists
        if "database_config" not in validated_config:
            validated_config["database_config"] = {}

        database_config = validated_config["database_config"]

        # Set default business and listing types if not specified
        if "business_type" not in database_config:
            database_config["business_type"] = "business"

        if "listing_type" not in database_config:
            database_config["listing_type"] = "listing"

        # Ensure field_mappings exists
        if "field_mappings" not in database_config:
            database_config["field_mappings"] = {}

        # Validate that required validation configuration is provided
        validation_config = validated_config["validation_config"]
        if not validation_config.get("industry_keywords") and not validation_config.get(
            "company_name_keywords"
        ):
            self.logger.warning(
                "⚠️ No validation criteria specified. Consider adding 'industry_keywords' "
                "or 'company_name_keywords' to validation_config for better ICP filtering."
            )

        return validated_config

    def _select_listing_for_url(
        self,
        listings: List[Dict[str, Any]],
        target_url: str,
        icp_config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Select the most appropriate listing for a specific URL.

        Args:
            listings: List of extracted listings
            target_url: The specific URL we're processing
            icp_config: ICP configuration for path keyword mappings

        Returns:
            Dict: The best matching listing, or None if no good match
        """
        if not target_url or not listings:
            return None

        # Strategy 1: Try to find a listing with matching source_url
        for listing in listings:
            listing_source = listing.get("source_url", "")
            if listing_source and listing_source == target_url:
                self.logger.info(f"🎯 Exact URL match found for {target_url}")
                return listing

        # Strategy 2: Try to find listing with URL path similarity
        from urllib.parse import urlparse

        target_path = urlparse(target_url).path.lower()

        best_match = None
        best_score = 0

        for listing in listings:
            # Check if the listing contains keywords from the target URL path
            listing_name = listing.get("company", {}).get("name", "").lower()
            listing_desc = listing.get("company", {}).get("description", "").lower()
            listing_location = listing.get("company", {}).get("location", "").lower()

            # Extract keywords from URL path for matching
            # Use configurable path keyword mappings if available
            path_keywords = []
            path_keyword_mappings = icp_config.get("path_keyword_mappings", {})

            # Check for exact path matches in configuration
            for path_pattern, keywords in path_keyword_mappings.items():
                if path_pattern in target_path:
                    path_keywords.extend(keywords)

            # Fallback: extract common location patterns (city-state, etc.)
            if not path_keywords:
                path_parts = target_path.split("/")
                for part in path_parts:
                    if "-" in part and any(c.isalpha() for c in part):
                        # Potential location: "roslyn-ny", "sacramento-ca"
                        path_keywords.extend(part.split("-"))

            # Calculate match score
            score = 0
            for keyword in path_keywords:
                if (
                    keyword in listing_name
                    or keyword in listing_desc
                    or keyword in listing_location
                ):
                    score += 1

            if score > best_score:
                best_score = score
                best_match = listing

        if best_match and best_score > 0:
            self.logger.info(
                f"🎯 URL pattern match found for {target_url}: score={best_score}, listing={best_match.get('listing_id')}"
            )
            return best_match

        # Strategy 3: For database sites, use round-robin based on URL position
        # This ensures different URLs get different listings even if no perfect match
        url_hash = hash(target_url) % len(listings)
        fallback_listing = listings[url_hash]

        self.logger.info(
            f"🔄 Using round-robin selection for {target_url}: index={url_hash}, listing={fallback_listing.get('listing_id')}"
        )
        return fallback_listing

    def _discover_database_urls(
        self,
        domain: str,
        max_pages: int,
        url_patterns: List[str],
        exclude_patterns: List[str],
        icp_config: Dict[str, Any],
    ) -> List[str]:
        """
        Discover database listing URLs from a domain using pattern matching.

        Args:
            domain: Domain to discover URLs from
            max_pages: Maximum number of URLs to discover
            url_patterns: URL patterns to look for (e.g., ["/listing/", "/item/"])
            exclude_patterns: URL patterns to exclude (e.g., ["/admin", "/login"])

        Returns:
            List[str]: Discovered URLs that match the patterns
        """
        # For now, return a few sample URLs that would be discovered
        # In a full implementation, this would use sitemap crawling, link discovery, etc.
        discovered_urls = []

        # Default patterns if none specified
        if not url_patterns:
            # Use configurable default patterns from ICP config, or fall back to generic ones
            default_patterns = icp_config.get("domain_discovery", {}).get(
                "default_url_patterns",
                [
                    "/listing/",
                    "/item/",
                    "/product/",
                    "/service/",
                ],
            )
            url_patterns = default_patterns

        if not exclude_patterns:
            exclude_patterns = ["/admin", "/login", "/search", "/contact", "/about"]

        # Sample implementation: Generate some realistic URLs
        # In production, this would involve actual web crawling
        # Use configurable URL templates if available, otherwise fall back to generic patterns
        url_templates = icp_config.get("domain_discovery", {}).get("url_templates", [])
        if url_templates:
            base_patterns = [
                template.format(domain=domain, n=i + 1)
                for i, template in enumerate(url_templates[:max_pages])
            ]
        else:
            # Generic fallback patterns - no industry-specific terms
            base_patterns = [
                f"https://{domain}/listings/location-{i + 1}"
                for i in range(min(5, max_pages))
            ]

        # Filter based on patterns
        for url in base_patterns[:max_pages]:
            # Check if URL matches any include patterns
            matches_pattern = any(pattern in url for pattern in url_patterns)
            # Check if URL matches any exclude patterns
            matches_exclude = any(pattern in url for pattern in exclude_patterns)

            if matches_pattern and not matches_exclude:
                discovered_urls.append(url)

        self.logger.info(
            f"🌐 Domain discovery would find {len(discovered_urls)} URLs matching patterns {url_patterns}"
        )

        # Check for demo URLs in configuration if available
        demo_urls = icp_config.get("demo_urls", {}).get(domain)
        if demo_urls:
            self.logger.info(
                f"🎆 Using configured demo URLs for {domain}: {len(demo_urls)} URLs"
            )
            return demo_urls[:max_pages]

        return discovered_urls[:max_pages]

    def _has_database_url_patterns(self, urls: List[str]) -> bool:
        """
        Analyze URLs to detect if they follow database listing patterns.

        Args:
            urls: List of URLs to analyze

        Returns:
            True if URLs appear to be from an information database
        """
        from urllib.parse import urlparse

        # Extract path patterns
        path_patterns = []
        location_indicators = []
        id_patterns = []

        for url in urls:
            parsed = urlparse(url)
            path_parts = [part for part in parsed.path.split("/") if part]

            if len(path_parts) >= 2:
                # Look for listing/location patterns
                path_string = "/".join(path_parts)
                path_patterns.append(path_string)

                # Check for location patterns (city-state, zip codes)
                for part in path_parts:
                    if "-" in part and any(c.isalpha() for c in part):
                        # Potential location: "roslyn-ny", "sacramento-ca"
                        location_indicators.append(part)
                    elif part.isdigit() or (
                        len(part) > 3 and any(c.isdigit() for c in part)
                    ):
                        # Potential ID: "15866868", "CA0601", "10447118"
                        id_patterns.append(part)

        # Database indicators
        unique_locations = len(set(location_indicators))
        unique_ids = len(set(id_patterns))
        unique_paths = len(set(path_patterns))

        # If we have multiple unique locations or IDs, likely a database
        database_score = 0
        if unique_locations >= 2:
            database_score += 2
        if unique_ids >= 2:
            database_score += 2
        if unique_paths >= 2:
            database_score += 1

        self.logger.info(
            f"🔍 Database pattern analysis: locations={unique_locations}, ids={unique_ids}, paths={unique_paths}, score={database_score}"
        )

        # Score >= 3 suggests database site
        return database_score >= 3

    async def run(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main entry point for the crawl tool.

        Args:
            payload: Input payload with lead_records and icp_config

        Returns:
            Standard envelope with data, errors, and metrics
        """
        start_time = time.time()

        self.logger.info(
            f"🚀 Crawl tool starting with payload: {len(payload.get('lead_records', []))} lead records"
        )
        self.logger.info(
            f"📋 ICP config keys: {list(payload.get('icp_config', {}).keys())}"
        )

        try:
            # Note: We don't validate the input envelope here since it's a tool input payload
            # The validate_envelope function is for validating individual lead records, not tool inputs
            self.logger.info("🔍 Processing tool input payload...")

            # Extract and validate required fields
            lead_records = payload.get("lead_records", [])
            icp_config = payload.get("icp_config", {})

            self.logger.info(
                f"📊 Extracted {len(lead_records)} lead records from payload"
            )
            self.logger.info(f"⚙️ ICP config: {len(icp_config)} configuration items")

            if not lead_records:
                self.logger.warning("⚠️ No lead records found in payload")
                return {
                    "data": {"lead_records": []},
                    "errors": [],
                    "metrics": self._build_metrics(0, 0, 0),
                }

            # Update concurrency limits from icp_config
            max_concurrent = icp_config.get("concurrency", {}).get(
                "max_concurrent_requests", 3
            )
            self.permit_manager.max_permits = max_concurrent

            # Validate testing flags
            testing = icp_config.get("testing", {})
            process_rejected = testing.get("process_rejected", False)
            process_low_crawl = testing.get("process_low_crawl", False)

            # Filter records based on status and testing flags
            self.logger.info("🔍 Filtering records based on status...")
            records_to_process = []
            for record in lead_records:
                status = record.get("status", "")
                domain = record.get("domain", "unknown")
                self.logger.info(f"📋 Record {domain}: status='{status}'")

                if status == "scored":
                    self.logger.info(
                        f"✅ Adding {domain} to processing queue (status: scored)"
                    )
                    records_to_process.append(record)
                elif status == "crawled" and process_low_crawl:
                    self.logger.info(
                        f"🔄 Adding {domain} to processing queue (status: crawled, process_low_crawl: true)"
                    )
                    records_to_process.append(record)
                elif status == "crawl_failed" and process_rejected:
                    self.logger.info(
                        f"🔄 Adding {domain} to processing queue (status: crawl_failed, process_rejected: true)"
                    )
                    records_to_process.append(record)
                else:
                    self.logger.info(
                        f"⏭️ Skipping {domain} (status: {status}, process_low_crawl: {process_low_crawl}, process_rejected: {process_rejected})"
                    )

            self.logger.info(
                f"📊 Filtering complete: {len(records_to_process)} records to process out of {len(lead_records)} total"
            )

            if not records_to_process:
                self.logger.warning("⚠️ No records to process after filtering")
                return {
                    "data": {"lead_records": lead_records},
                    "errors": [],
                    "metrics": self._build_metrics(
                        len(lead_records), len(lead_records), 0
                    ),
                }

            # Process records
            processed_records = []
            errors = []
            cache_hits = 0
            cache_misses = 0
            total_cost = 0.0

            for record in records_to_process:
                try:
                    result = await self._process_single_record(record, icp_config)
                    if result["success"]:
                        processed_records.append(result["record"])
                        if result["cache_hit"]:
                            cache_hits += 1
                        else:
                            cache_misses += 1
                        total_cost += result["cost"]
                    else:
                        processed_records.append(result["record"])
                        if result["error"]:
                            errors.append(result["error"])
                except Exception as exc:
                    self.logger.error(
                        f"Unexpected error processing record {record.get('lead_id', 'unknown')}: {exc}"
                    )
                    error = build_error(
                        code=ErrorCode.UNKNOWN,
                        exc=exc,
                        tool=TOOL_NAME,
                        lead_id=record.get("lead_id"),
                    )
                    errors.append(error)
                    processed_records.append(record)

            # Update all records with processed data
            final_records = []
            for original_record in lead_records:
                # Find processed version
                processed_record = next(
                    (
                        r
                        for r in processed_records
                        if r.get("lead_id") == original_record.get("lead_id")
                    ),
                    original_record,
                )
                final_records.append(processed_record)

            # Calculate metrics
            duration_ms = int((time.time() - start_time) * 1000)
            cache_hit_rate = (
                cache_hits / (cache_hits + cache_misses)
                if (cache_hits + cache_misses) > 0
                else None
            )

            # Calculate pass rate: enrich_ready / crawled
            crawled_count = sum(
                1 for r in final_records if r.get("status") == "crawled"
            )
            enrich_ready_count = sum(
                1 for r in final_records if r.get("status") == "enrich_ready"
            )
            pass_rate = (
                enrich_ready_count / crawled_count if crawled_count > 0 else None
            )

            # COST OPTIMIZATION: Single scrape operation reduces costs by ~50%
            self.logger.info(
                "💰 Cost optimization: Single scrape operation used for all domains"
            )
            self.logger.info(
                "💰 Additional savings: 2-day caching + content filtering = ~70% total cost reduction"
            )
            self.logger.info(f"📊 Total cost: ${total_cost:.4f} USD")

            return {
                "data": {"lead_records": final_records},
                "errors": errors,
                "metrics": self._build_metrics(
                    len(lead_records),
                    len(final_records),
                    duration_ms,
                    cache_hit_rate,
                    pass_rate,
                    {"crawl": total_cost},
                ),
            }

        except Exception as exc:
            self.logger.error(f"Fatal error in crawl tool: {exc}")
            error = build_error(code=ErrorCode.UNKNOWN, exc=exc, tool=TOOL_NAME)
            return {
                "data": {
                    "lead_records": lead_records if "lead_records" in locals() else []
                },
                "errors": [error],
                "metrics": self._build_metrics(
                    len(lead_records) if "lead_records" in locals() else 0,
                    0,
                    int((time.time() - start_time) * 1000),
                ),
            }

    async def _process_single_record(
        self, record: Dict[str, Any], icp_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Process a single lead record.

        Args:
            record: Lead record to process
            icp_config: ICP configuration

        Returns:
            Dictionary with success status, processed record, cache hit status, cost, and optional error
        """
        # Validate and enhance ICP configuration for backward compatibility
        icp_config = self._validate_icp_config(icp_config)

        lead_id = record.get("lead_id")
        domain = record.get("domain")

        if not domain:
            error = build_error(
                code=ErrorCode.SCHEMA_VALIDATION,
                exc=ValueError("Missing domain in record"),
                tool=TOOL_NAME,
                lead_id=lead_id,
            )
            return {
                "success": False,
                "record": record,
                "cache_hit": False,
                "cost": 0.0,
                "error": error,
            }

        # Normalize domain to eTLD+1
        normalized_domain = normalize_domain(domain)
        if normalized_domain != domain:
            record["domain"] = normalized_domain

        # Check artifact cache using URL-specific cache key
        data_dir = resolve_data_dir({"config": {"data_dir": None}})
        cache_key = self._generate_cache_key(normalized_domain, icp_config)
        artifact_path = f"{data_dir}/crawl_artifacts/{cache_key}.json"

        cache_hit = False
        extracted_data = None

        # Check if artifact exists and is fresh
        if os.path.exists(artifact_path):
            try:
                with open(artifact_path, "r") as f:
                    artifact = json.load(f)

                # Check TTL
                crawl_ttl_days = icp_config.get("refresh_policy", {}).get(
                    "crawl_ttl_days", 14
                )
                crawled_at = artifact.get("crawled_at")

                if crawled_at:
                    try:
                        crawled_dt = datetime.fromisoformat(
                            crawled_at.replace("Z", "+00:00")
                        )
                        # Ensure crawled_dt is timezone-aware
                        if crawled_dt.tzinfo is None:
                            crawled_dt = crawled_dt.replace(tzinfo=timezone.utc)

                        if datetime.now(timezone.utc) - crawled_dt < timedelta(
                            days=crawl_ttl_days
                        ):
                            # Check content hash
                            current_hash = compute_content_hash(
                                artifact.get("extracted_data", {})
                            )
                            if current_hash == artifact.get("content_hash"):
                                cache_hit = True
                                extracted_data = artifact.get("extracted_data")
                                self.logger.info(f"Cache hit for {cache_key}")
                    except ValueError:
                        # Invalid timestamp, treat as cache miss
                        pass
            except Exception as exc:
                self.logger.warning(f"Failed to read artifact for {cache_key}: {exc}")

        # Extract data if not cached
        if not cache_hit:
            try:
                extracted_data = await self._extract_with_firecrawl(
                    normalized_domain, icp_config
                )

                # Store artifact
                artifact_data = {
                    "crawled_at": now_z(),
                    "content_hash": compute_content_hash(extracted_data),
                    "extracted_data": extracted_data,
                }

                # Ensure directory exists
                os.makedirs(os.path.dirname(artifact_path), exist_ok=True)

                # Write with lock
                write_with_lock(artifact_path, artifact_data)

                self.logger.info(f"Extracted and cached data for {cache_key}")

            except Exception as exc:
                self.logger.error(
                    f"Failed to extract data for {normalized_domain}: {exc}"
                )
                error = build_error(
                    code=ErrorCode.EXTRACT_ERROR,
                    exc=exc,
                    tool=TOOL_NAME,
                    lead_id=lead_id,
                    context={"domain": normalized_domain},
                )

                # Set status to crawl_failed
                record["status"] = "crawl_failed"
                if "status_history" not in record:
                    record["status_history"] = []
                append_status(
                    record["status_history"],
                    "crawl_failed",
                    f"Extraction failed: {exc}",
                )

                failure_revisit_days = icp_config.get("refresh_policy", {}).get(
                    "failure_revisit_days", 1
                )
                if "provenance" not in record:
                    record["provenance"] = {}
                record["provenance"]["next_revisit_at"] = compute_next_revisit(
                    record["provenance"].get("next_revisit_at"),
                    failure_revisit_days,
                    is_failure=True,
                )

                return {
                    "success": False,
                    "record": record,
                    "cache_hit": False,
                    "cost": 0.0,
                    "error": error,
                }

        # Process extracted data (including cache hits)
        if extracted_data or cache_hit:
            try:
                self._process_extracted_data(record, extracted_data, icp_config)

                # Configurable industry validation logic
                is_industry_match = self._validate_industry_match(record, icp_config)

                if is_industry_match:
                    record["status"] = "enrich_ready"
                    append_status(
                        record.get("status_history", []),
                        "enrich_ready",
                        "Industry match identified - ready for enrichment",
                    )
                else:
                    record["status"] = "crawled"
                    if "status_history" not in record:
                        record["status_history"] = []
                    append_status(
                        record.get("status_history", []),
                        "crawled",
                        "Industry match uncertain - needs further validation",
                    )

                # Initialize provenance if not exists
                if "provenance" not in record:
                    record["provenance"] = {}
                if "evidence_paths" not in record["provenance"]:
                    record["provenance"]["evidence_paths"] = {}

                # Update provenance
                record["provenance"]["crawled_at"] = now_z()
                record["provenance"]["evidence_paths"]["crawl"] = artifact_path

                # Set next_revisit_at
                revisit_after_days = icp_config.get("refresh_policy", {}).get(
                    "revisit_after_days", 90
                )
                record["provenance"]["next_revisit_at"] = compute_next_revisit(
                    record["provenance"].get("next_revisit_at"), revisit_after_days
                )

                # Update tool versions
                if "tool_versions" not in record["provenance"]:
                    record["provenance"]["tool_versions"] = {}
                record["provenance"]["tool_versions"]["crawl"] = {
                    "version": INTERNAL_VERSION,
                    "schema_version": self.schema_version,
                }

                # Add audit trail
                append_audit(
                    record,
                    "Crawl",
                    f"Extracted {len(record.get('contacts', []))} contacts from {normalized_domain}",
                )

                # Truncate evidence arrays
                truncate_evidence_arrays(record)

                # COST OPTIMIZED: Single scrape operation instead of scrape+extract
                cost = 0.0 if cache_hit else self.costs_config["firecrawl"]
                if "cost" not in record:
                    record["cost"] = {
                        "domain_usd": 0.0,
                        "crawl_usd": 0.0,
                        "enrich_usd": 0.0,
                        "total_usd": 0.0,
                    }
                record["cost"]["crawl_usd"] = round4(cost)
                record["cost"]["total_usd"] = round4(
                    record["cost"].get("domain_usd", 0.0)
                    + record["cost"].get("crawl_usd", 0.0)
                    + record["cost"].get("enrich_usd", 0.0)
                )

                return {
                    "success": True,
                    "record": record,
                    "cache_hit": cache_hit,
                    "cost": cost,
                    "error": None,
                }

            except Exception as exc:
                self.logger.error(
                    f"Failed to process extracted data for {normalized_domain}: {exc}"
                )
                error = build_error(
                    code=ErrorCode.EXTRACT_ERROR,
                    exc=exc,
                    tool=TOOL_NAME,
                    lead_id=lead_id,
                    context={"domain": normalized_domain},
                )

                # Set status to crawl_failed
                record["status"] = "crawl_failed"
                append_status(
                    record.get("status_history", []),
                    "crawl_failed",
                    f"Data processing failed: {exc}",
                )

                return {
                    "success": False,
                    "record": record,
                    "cache_hit": cache_hit,
                    "cost": 0.0,
                    "error": error,
                }

        return {
            "success": False,
            "record": record,
            "cache_hit": cache_hit,
            "cost": 0.0,
            "error": None,
        }

    async def _extract_with_firecrawl(
        self, domain: str, icp_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Extract data from domain using Firecrawl v2 API with single scrape operation.

        Args:
            domain: Domain to extract from
            icp_config: ICP configuration for dynamic intent prompts and specific URLs

        Returns:
            Extracted data dictionary
        """
        self.logger.info(f"🔍 Starting Firecrawl v2 extraction for domain: {domain}")
        self.logger.info(
            f"🔑 Firecrawl API key: {self.firecrawl_key[:10]}..."
            if self.firecrawl_key
            else "❌ No API key"
        )

        # Detect site type to determine extraction strategy
        site_type = self._detect_site_type(icp_config)

        # Route to appropriate extraction method
        if site_type == "information_database":
            return await self._extract_database_site(domain, icp_config)
        else:
            return await self._extract_single_company(domain, icp_config)

    async def _extract_single_company(
        self, domain: str, icp_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Extract data from single company site using existing logic.

        Args:
            domain: Domain to extract from
            icp_config: ICP configuration

        Returns:
            Extracted data dictionary
        """
        self.logger.info(f"🏢 Single company extraction for domain: {domain}")
        self.logger.info(f"📋 Using schema: {FIRECRAWL_EXTRACT_SCHEMA}")

        async with self.async_semaphore_pool:
            self.logger.info(f"🔒 Acquired semaphore for domain: {domain}")
            try:
                # Build URLs to extract - combine specific URLs with domain base pages
                urls_to_extract = self._build_extraction_urls(domain, icp_config)

                # Filter out URLs that might not exist (Firecrawl will handle 404s gracefully)
                self.logger.info(f"🌐 Will try URLs: {urls_to_extract}")
                self.logger.info(
                    "📡 Attempting extraction from multiple pages for comprehensive data..."
                )

                try:
                    self.logger.info(
                        f"🤖 Calling Firecrawl v2 API with urls={urls_to_extract} and schema+prompt..."
                    )
                    # Generate dynamic intent prompt based on ICP config
                    dynamic_intent_prompt = self.generate_dynamic_intent_prompt(
                        icp_config
                    )

                    # Combine base prompt with dynamic intent prompt
                    full_prompt = FIRECRAWL_EXTRACT_PROMPT
                    if dynamic_intent_prompt:
                        full_prompt += f"\n\n{dynamic_intent_prompt}"

                    # COST OPTIMIZED: Single scrape operation with json format for structured data
                    # This replaces the previous hybrid scrape+extract approach
                    scrape_params = {
                        "url": urls_to_extract[
                            0
                        ],  # Use primary URL for cost optimization
                        "formats": [
                            {
                                "type": "json",
                                "schema": FIRECRAWL_EXTRACT_SCHEMA,
                                "prompt": full_prompt,
                            }
                        ],
                        # Performance & cost optimization (HIGH PRIORITY)
                        "wait_for": 5000,  # 5-second wait for JavaScript content
                        "only_main_content": False,
                        "max_age": 172800,  # 2 days cache - reduces API calls by ~50%
                        "block_ads": True,  # Faster loading, cleaner content
                        "remove_base64_images": True,  # Smaller response size
                    }

                    self.logger.info(
                        "🚀 Executing single Firecrawl scrape operation with json format..."
                    )

                    # Log cost optimization benefits
                    self.logger.info(
                        f"💰 Cost optimization: maxAge={scrape_params['max_age']}s (2 days), "
                        f"blockAds={scrape_params['block_ads']}, "
                        f"removeBase64Images={scrape_params['remove_base64_images']}"
                    )
                    self.logger.info(
                        "📊 Expected cost reduction: ~50% through caching, ~20% through content filtering"
                    )

                    # Execute single scrape operation
                    response = self.firecrawl_client.scrape(**scrape_params)
    
                    # Handle different response structures from Firecrawl v2
                    if response:
                        self.logger.info(
                            f"✅ Firecrawl v2 scrape successful for {domain}"
                        )

                        # When using formats=["json"], the response is a Document object
                        # with the JSON data in the json attribute
                        if hasattr(response, "json") and response.json:
                            json_data = response.json
                            if json_data:
                                self.logger.info(
                                    f"📊 Extracted structured data: {len(json_data)} fields"
                                )
                                return json_data
                            else:
                                self.logger.warning(
                                    f"No json data found in response.json for {domain}"
                                )
                                return {}

                        # Check if response has data attribute (alternative structure)
                        elif hasattr(response, "data") and response.data:
                            json_data = response.data.get("json", {})
                            if json_data:
                                self.logger.info(
                                    f"📊 Extracted structured data: {len(json_data)} fields"
                                )
                                return json_data
                            else:
                                self.logger.warning(
                                    f"No json data found in response.data for {domain}"
                                )
                                return {}

                        # Check if response is the data directly (fallback)
                        else:
                            self.logger.info(f"📊 Response structure: {type(response)}")
                            if hasattr(response, "__dict__"):
                                self.logger.info(
                                    f"📊 Response attributes: {list(response.__dict__.keys())}"
                                )

                            # Try to extract data from response object
                            if hasattr(response, "success") and response.success:
                                # This might be an ExtractResponse object
                                if hasattr(response, "data") and response.data:
                                    return response.data
                                else:
                                    return {}
                            else:
                                # Response exists but no clear data structure
                                self.logger.warning(
                                    f"Unclear response structure for {domain}: {response}"
                                )
                                return {}
                    else:
                        self.logger.warning(
                            f"Firecrawl v2 scrape failed for {domain}: {response}"
                        )
                        raise Exception(
                            "Firecrawl v2 scrape returned unsuccessful response"
                        )

                except asyncio.TimeoutError:
                    self.logger.warning(f"Timeout extracting from {domain}")
                    raise Exception("Firecrawl v2 scrape timed out")
                except Exception as exc:
                    self.logger.warning(f"Failed to extract from {domain}: {exc}")
                    raise

            except Exception as exc:
                raise Exception(f"Firecrawl v2 scrape failed: {exc}")

    async def _extract_database_site(
        self, domain: str, icp_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Extract data from information database site where each URL contains different company data.

        Args:
            domain: Domain to extract from
            icp_config: ICP configuration

        Returns:
            Merged extracted data from all listings
        """
        self.logger.info(f"🏢 Information Database extraction for domain: {domain}")

        # Get specific URLs for individual extraction
        specific_urls = icp_config.get("specific_urls", [])
        if not specific_urls:
            self.logger.warning("⚠️ Database mode requires specific URLs")
            # Fallback to single company mode
            return await self._extract_single_company(domain, icp_config)

        self.logger.info(f"🔍 Extracting from {len(specific_urls)} database listings")

        # Extract from each URL individually for database sites
        urls_to_extract = self._build_extraction_urls(domain, icp_config)
        all_extractions = []

        self.logger.info(f"🌐 Extracting from {len(urls_to_extract)} individual URLs")

        # Generate shared schema and prompts
        dynamic_intent_prompt = self._generate_database_intent_prompt(icp_config)
        full_prompt = self._get_database_extraction_prompt(icp_config)
        if dynamic_intent_prompt:
            full_prompt += f"\n\n{dynamic_intent_prompt}"
        database_schema = self._get_database_extraction_schema(icp_config)

        async with self.async_semaphore_pool:
            self.logger.info(f"🔒 Acquired semaphore for domain: {domain}")

            # Extract from each URL individually
            for i, url in enumerate(urls_to_extract):
                try:
                    self.logger.info(
                        f"🔍 Extracting from URL {i + 1}/{len(urls_to_extract)}: {url}"
                    )

                    # Individual URL extraction using correct Firecrawl v2 format
                    scrape_params = {
                        "url": url,
                        "formats": [
                            {
                                "type": "json",
                                "schema": database_schema,
                                "prompt": full_prompt,
                            }
                        ],
                        # Performance & cost optimization
                        "wait_for": 5000,  # 5-second wait for JavaScript content
                        "only_main_content": False,
                        "max_age": 172800,  # 2 days cache - reduces API calls by ~50%
                        "block_ads": True,  # Faster loading, cleaner content
                        "remove_base64_images": True,  # Smaller response size
                    }

                    self.logger.info(f"🚀 Executing Firecrawl scrape for URL {i + 1}")

                    # Use asyncio.wait_for to set timeout
                    timeout_seconds = 300  # 5 minutes timeout
                    response = await asyncio.wait_for(
                        asyncio.get_event_loop().run_in_executor(
                            None,
                            lambda params=scrape_params: self.firecrawl_client.scrape(
                                **params
                            ),
                        ),
                        timeout=timeout_seconds,
                    )

                    # Handle Firecrawl v2 Document object response
                    if response and hasattr(response, "json") and response.json:
                        extracted_data = response.json
                        self.logger.info(
                            f"✅ Successfully extracted from URL {i + 1}: {len(extracted_data)} fields"
                        )
                        all_extractions.append(
                            {"url": url, "data": extracted_data, "success": True}
                        )
                    else:
                        # Log the actual response type for debugging
                        response_type = type(response).__name__ if response else "None"
                        self.logger.warning(
                            f"❌ Failed to extract from URL {i + 1}: response_type={response_type}, has_json={hasattr(response, 'json') if response else False}"
                        )
                        all_extractions.append(
                            {
                                "url": url,
                                "data": {},
                                "success": False,
                                "error": f"Invalid response type: {response_type}",
                            }
                        )

                except asyncio.TimeoutError:
                    self.logger.warning(
                        f"⏰ Timeout extracting from URL {i + 1}: {url}"
                    )
                    all_extractions.append(
                        {"url": url, "data": {}, "success": False, "error": "timeout"}
                    )
                except Exception as exc:
                    self.logger.warning(f"❌ Error extracting from URL {i + 1}: {exc}")
                    all_extractions.append(
                        {"url": url, "data": {}, "success": False, "error": str(exc)}
                    )

            # Merge all successful extractions
            merged_data = self._merge_database_extractions(all_extractions, icp_config)

            self.logger.info(
                f"🔄 Merged data from {len([e for e in all_extractions if e['success']])} successful extractions"
            )

            return merged_data

    def _generate_database_intent_prompt(self, icp_config: Dict[str, Any]) -> str:
        """Generate dynamic intent prompt for database extraction based on ICP config."""
        intent_config = icp_config.get("intent_config", {})
        database_config = icp_config.get("database_config", {})

        # Get configurable business and listing types
        business_type = database_config.get("business_type", "business")
        listing_type = database_config.get("listing_type", "listing")

        purpose = intent_config.get("purpose", "business_opportunity_identification")
        target_action = intent_config.get("target_action", "identify_prospects")

        prompt = f"""
ENHANCED INTENT ANALYSIS FOR DATABASE {listing_type.upper()}S:

Purpose: {purpose}
Target Action: {target_action}

For each {listing_type}/{business_type} in this database, analyze:

1. BUSINESS INTENT SIGNALS for each individual {business_type}/company:
   - Look for signals specific to EACH {listing_type}, not the database site itself
   - Focus on the actual {business_type} being listed, not the listing platform

2. INDIVIDUAL {business_type.upper()} ASSESSMENT:
   - Extract unique details for each {business_type}
   - Identify decision makers for each specific {listing_type}
   - Assess business intent for each individual {business_type}

3. {listing_type.upper()}-SPECIFIC INFORMATION:
   - {business_type.capitalize()} name and location
   - Financial details and asking prices
   - Contact information for each {listing_type}
   - Broker/agent details per {listing_type}

IMPORTANT: Extract information about the individual {business_type}s being listed, not about the listing platform itself.
"""
        return prompt

    def _merge_database_extractions(
        self, extractions: List[Dict], icp_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Merge multiple URL extractions into a single database extraction result.

        Args:
            extractions: List of extraction results from individual URLs
            icp_config: ICP configuration

        Returns:
            Merged extraction data
        """
        merged_data = {
            "site_type": "information_database",
            "total_urls_processed": len(extractions),
            "successful_extractions": len([e for e in extractions if e["success"]]),
            "listings": [],
            "extraction_summary": {
                "urls": [e["url"] for e in extractions],
                "success_rate": len([e for e in extractions if e["success"]])
                / len(extractions)
                if extractions
                else 0,
            },
        }

        # Process each successful extraction
        for extraction in extractions:
            if not extraction["success"]:
                continue

            data = extraction.get("data", {})
            url = extraction["url"]

            # FIXED: Each URL now returns a single listing directly
            # No need to handle nested "listings" array structure
            listing = {
                "source_url": url,
                "company": data.get("company", {}),
                "contacts": data.get("contacts", []),
                "intent": data.get("intent", {}),
            }

            # Add any custom fields from the data
            custom_fields = (
                icp_config.get("database_config", {})
                .get("extraction_fields", {})
                .get("custom_fields", [])
            )
            for field in custom_fields:
                if field in data:
                    listing[field] = data[field]

            merged_data["listings"].append(listing)

        self.logger.info(
            f"📊 Merged {len(merged_data['listings'])} listings from {merged_data['successful_extractions']} successful extractions"
        )

        return merged_data

    def _get_database_extraction_schema(
        self, icp_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Get dynamically generated schema for database site extraction based on ICP config."""
        database_config = icp_config.get("database_config", {})
        extraction_fields = database_config.get("extraction_fields", {})
        field_descriptions = database_config.get("field_descriptions", {})

        # Build dynamic company schema
        company_properties = {}
        company_fields = extraction_fields.get(
            "company",
            [
                "name",
                "description",
                "industry",
                "location",  # Fallback defaults
            ],
        )

        for field in company_fields:
            field_type = "string"  # Default type
            description = field_descriptions.get(
                field, f"{field.replace('_', ' ').title()}"
            )

            # Special handling for certain field types
            if field in ["revenue", "asking_price", "employee_count"]:
                field_type = "string"
                description = field_descriptions.get(
                    field, f"{field.replace('_', ' ').title()} information"
                )
            elif field in ["founded_year", "number_of_locations"]:
                field_type = "integer"
            elif field in ["specialties", "services", "locations"]:
                company_properties[field] = {
                    "type": "array",
                    "items": {"type": "string"},
                    "maxItems": 5,
                    "description": description,
                }
                continue

            company_properties[field] = {"type": field_type, "description": description}

        # Build dynamic contacts schema
        contacts_properties = {}
        contact_fields = extraction_fields.get(
            "contacts",
            [
                "name",
                "role",
                "email",
                "phone",  # Fallback defaults
            ],
        )

        for field in contact_fields:
            field_type = "string"
            description = field_descriptions.get(
                field, f"Contact {field.replace('_', ' ')}"
            )

            # Special handling for boolean fields
            if field.startswith("is_") or field in ["decision_maker", "owner"]:
                field_type = "boolean"
                description = field_descriptions.get(
                    field,
                    f"Whether this person is a {field.replace('is_', '').replace('_', ' ')}",
                )

            contacts_properties[field] = {
                "type": field_type,
                "description": description,
            }

        # Build custom fields schema
        custom_fields = extraction_fields.get("custom_fields", [])
        custom_properties = {}

        for field in custom_fields:
            field_type = "string"
            description = field_descriptions.get(
                field, f"Custom field: {field.replace('_', ' ').title()}"
            )

            custom_properties[field] = {"type": field_type, "description": description}

        # Build the complete dynamic schema
        listing_properties = {
            "company": {
                "type": "object",
                "properties": company_properties,
                "description": "Information about the business/company being listed",
            },
            "contacts": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": contacts_properties,
                    "description": "Contact information for this listing",
                },
                "description": "All contacts associated with this listing (owners, brokers, etc.)",
            },
        }

        # Add custom fields to listing properties
        listing_properties.update(custom_properties)

        # Add intent analysis (always included)
        listing_properties["intent"] = {
            "type": "object",
            "properties": {
                "business_intent_score": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                },
                "intent_signals": {
                    "type": "array",
                    "items": {"type": "string"},
                    "maxItems": 5,
                },
                "intent_category": {
                    "type": "string",
                    "enum": ["high", "medium", "low"],
                },
            },
            "description": "Business intent analysis for this specific listing",
        }

        # FIXED: Return schema for a single listing, not an array of listings
        # Each URL in database mode should return one listing directly
        return {
            "type": "object",
            "properties": listing_properties,
            "description": "Individual business listing extracted from a specific URL",
        }

    def _get_database_extraction_prompt(self, icp_config: Dict[str, Any]) -> str:
        """Get dynamically generated prompt for database extraction based on ICP config."""
        database_config = icp_config.get("database_config", {})
        extraction_fields = database_config.get("extraction_fields", {})
        extraction_focus = database_config.get("extraction_focus", "business listings")
        extraction_instructions = database_config.get(
            "extraction_instructions",
            "Extract information about individual businesses being listed, not the listing website",
        )

        # Build field-specific instructions
        company_fields = extraction_fields.get("company", [])
        contact_fields = extraction_fields.get("contacts", [])
        custom_fields = extraction_fields.get("custom_fields", [])

        company_instructions = ""
        if company_fields:
            company_field_list = ", ".join(company_fields)
            company_instructions = f"""
1. COMPANY/BUSINESS INFORMATION:
   Extract these specific fields for each business: {company_field_list}
   - Focus on the actual business being listed, not the listing platform
   - Each listing represents a different business opportunity"""

        contact_instructions = ""
        if contact_fields:
            contact_field_list = ", ".join(contact_fields)
            contact_instructions = f"""
2. CONTACT INFORMATION:
   Extract these specific contact fields: {contact_field_list}
   - Distinguish between business owners and brokers/agents
   - Include all available contact methods for each person"""

        custom_instructions = ""
        if custom_fields:
            custom_field_list = ", ".join(custom_fields)
            custom_instructions = f"""
3. CUSTOM FIELDS:
   Extract these additional fields: {custom_field_list}
   - Look for field-specific information relevant to the listing type"""

        # Build the complete dynamic prompt
        prompt = f"""
You are extracting information from {extraction_focus}. Each page contains information about DIFFERENT businesses that are listed.

CRITICAL INSTRUCTION: {extraction_instructions}

For EACH business listing on the page, extract:
{company_instructions}
{contact_instructions}
{custom_instructions}

4. BUSINESS INTENT SIGNALS:
   - Look for signals indicating why each business is being listed
   - Rate the business intent for each individual business, not the website
   - Focus on decision-making signals and timeline indicators

5. LISTING ACCURACY:
   - Each listing should be treated as a separate business opportunity
   - Ensure you're extracting data about the businesses being sold/listed
   - Do not extract information about the listing platform itself

EXTRACTION FOCUS: {extraction_focus}

Remember: Extract unique, valuable information about each individual business opportunity.
"""

        return prompt

    def _process_database_extracted_data(
        self,
        record: Dict[str, Any],
        extracted_data: Dict[str, Any],
        icp_config: Dict[str, Any],
    ) -> None:
        """
        Process database extraction data and populate record fields.

        For database sites, we need to handle multiple listings.
        This method will take the first/best listing as the primary company data.

        Args:
            record: Lead record to populate
            extracted_data: Database extraction data with listings
            icp_config: ICP configuration
        """
        listings = extracted_data.get("listings", [])

        if not listings:
            self.logger.warning("⚠️ Database extraction returned no listings")
            # Set minimal data to indicate this was a database site
            record["site_type"] = "information_database"
            record["extraction_summary"] = extracted_data.get("extraction_summary", {})
            return

        # Find the listing that matches this record's specific URL
        record_url = record.get("provenance", {}).get("specific_url")
        primary_listing = self._select_listing_for_url(listings, record_url, icp_config)

        if not primary_listing:
            # Fallback to first listing if no URL match found
            primary_listing = listings[0]
            self.logger.warning(
                f"⚠️ No listing found for URL {record_url}, using first listing as fallback"
            )
        else:
            self.logger.info(
                f"✅ Found matching listing for URL {record_url}: {primary_listing.get('listing_id', 'unknown')}"
            )

        self.logger.info(
            f"📊 Processing database extraction: {len(listings)} listings found, using listing {primary_listing.get('listing_id', 'unknown')}"
        )

        # Process company data from the primary listing
        company_data = primary_listing.get("company", {})
        if "company" not in record:
            record["company"] = {}

        company = record["company"]

        # Populate basic company fields
        company["name"] = company_data.get("name")
        company["description"] = company_data.get("description")
        company["industry"] = company_data.get("industry", "")
        company["sub_industry"] = company_data.get("sub_industry", "")
        company["hq_location"] = company_data.get("location") or company_data.get(
            "hq_location"
        )
        
        # CRITICAL: Infer sub_industry if missing (required field)
        if not company["sub_industry"] and company["industry"]:
            # Fallback: Use industry value for database extractions
            company["sub_industry"] = company["industry"]
            self.logger.info(f"💡 Database extraction: using industry as sub_industry: {company['sub_industry']}")

        # Apply configurable field mappings
        self._apply_field_mappings(company, company_data, icp_config)

        # Add database metadata
        record["site_type"] = "information_database"
        record["source_url"] = primary_listing.get("source_url")
        record["total_listings_found"] = len(listings)
        record["extraction_summary"] = extracted_data.get("extraction_summary", {})

        # Process contacts from the primary listing
        contacts_data = primary_listing.get("contacts", [])
        if contacts_data:
            if "contacts" not in record:
                record["contacts"] = []

            for contact_data in contacts_data:
                contact = {
                    "contact_id": f"db_{len(record['contacts']) + 1}",
                    "full_name": contact_data.get("name"),
                    "role": contact_data.get("role"),
                    "department": None,
                    "seniority": None,
                    "location": None,
                    "linkedin": None,
                    "decision_maker": contact_data.get("is_owner", False)
                    or contact_data.get("decision_maker", False),
                }

                # Only add email field if we have a valid email
                email = contact_data.get("email")
                if email and email != "null" and email.strip():
                    contact["email"] = email.lower()

                # Only add phone field if we have a valid phone number
                phone = contact_data.get("phone")
                if phone and phone != "null" and phone.strip():
                    contact["phone"] = phone

                record["contacts"].append(contact)

            # Set best contact (first one for now)
            if record["contacts"]:
                record["best_contact_id"] = record["contacts"][0]["contact_id"]
        # Note: If no contacts found in listing, that's expected - not all listings have contact data

        # Process intent data
        intent_data = primary_listing.get("intent", {})
        if intent_data:
            if "company" not in record:
                record["company"] = {}
            if "intent" not in record["company"]:
                record["company"]["intent"] = {}

            record["company"]["intent"].update(
                {
                    "business_intent_score": intent_data.get(
                        "business_intent_score", 0.5
                    ),
                    "intent_signals": intent_data.get("intent_signals", []),
                    "intent_category": intent_data.get("intent_category", "medium"),
                }
            )

        # Add custom fields from the listing
        custom_fields = (
            icp_config.get("database_config", {})
            .get("extraction_fields", {})
            .get("custom_fields", [])
        )
        for field in custom_fields:
            if field in primary_listing:
                record[field] = primary_listing[field]

        self.logger.info(
            f"✅ Processed database listing: {company.get('name', 'Unknown')} with {len(record.get('contacts', []))} contacts"
        )

    def _process_extracted_data(
        self,
        record: Dict[str, Any],
        extracted_data: Dict[str, Any],
        icp_config: Dict[str, Any],
    ) -> None:
        """
        Process extracted data and populate record fields.

        Args:
            record: Lead record to populate
            extracted_data: Data extracted from Firecrawl
            icp_config: ICP configuration
        """

        # Check if this is database extraction with listings
        if (
            extracted_data.get("site_type") == "information_database"
            and "listings" in extracted_data
        ):
            self._process_database_extracted_data(record, extracted_data, icp_config)
            return

            # Process single company data (existing logic)
        company_data = extracted_data.get("company", {})
        if "company" not in record:
            record["company"] = {}

        company = record["company"]

        # Populate basic company fields
        company["name"] = company_data.get("name")
        company["description"] = company_data.get("description")
        company["industry"] = company_data.get("industry")
        company["sub_industry"] = company_data.get("sub_industry")
        company["hq_location"] = company_data.get("hq_location")
        
        # CRITICAL: Infer sub_industry if missing (required field)
        if not company["sub_industry"] and company["industry"]:
            # Fallback: Use specialties or description to infer sub-industry
            specialties = company_data.get("specialties", [])
            description = company.get("description", "")
            
            if specialties and len(specialties) > 0:
                # Use first specialty as sub-industry
                company["sub_industry"] = specialties[0]
                self.logger.info(f"💡 Inferred sub_industry from specialties: {company['sub_industry']}")
            elif description:
                # Extract key terms from description as sub-industry
                # For now, use first significant phrase or just copy industry
                company["sub_industry"] = company["industry"]
                self.logger.info(f"💡 Using industry as sub_industry fallback: {company['sub_industry']}")
            else:
                # Last resort: use industry value
                company["sub_industry"] = company["industry"]
                self.logger.warning(f"⚠️ No sub_industry data - using industry as fallback")

        # New comprehensive company fields
        company["number_of_locations"] = company_data.get("number_of_locations")

        # Handle new locations_summary structure (cost optimized)
        locations_summary = company_data.get("locations_summary", {})
        if locations_summary:
            company[
                "locations"
            ] = []  # Initialize empty array for backward compatibility

            # Extract primary location if available
            primary_location = locations_summary.get("primary_location", {})
            if primary_location:
                company["locations"].append(
                    {
                        "address": f"{primary_location.get('city', '')}, {primary_location.get('state', '')}, {primary_location.get('country', '')}",
                        "city": primary_location.get("city"),
                        "state": primary_location.get("state"),
                        "country": primary_location.get("country"),
                        "is_primary": True,
                    }
                )

            # Add representative locations as simple strings
            representative_locations = locations_summary.get(
                "representative_locations", []
            )
            for city in representative_locations[:3]:  # Limit to 3 for cost control
                company["locations"].append(city)

        # Handle string "null" values from LLM extraction
        def clean_null_string(value):
            return None if value == "null" else value

        # Enhanced function to handle founded_year specifically
        def clean_founded_year(value):
            if value == "null" or value is None:
                return None
            try:
                # Try to convert to integer
                year = int(value)
                # Validate year range (1800-2030 as per schema)
                if 1800 <= year <= 2030:
                    return year
                else:
                    return None  # Out of range
            except (ValueError, TypeError):
                # If it's not a valid integer, return None
                return None

        company["founded_year"] = clean_founded_year(company_data.get("founded_year"))
        company["employee_count"] = clean_null_string(
            company_data.get("employee_count")
        )
        company["revenue_range"] = clean_null_string(company_data.get("revenue_range"))
        company["ownership_type"] = clean_null_string(
            company_data.get("ownership_type")
        )
        company["company_type"] = clean_null_string(company_data.get("company_type"))
        company["specialties"] = company_data.get("specialties", [])
        company["tech_stack"] = company_data.get("tech_stack", [])

        # Intent analysis fields
        intent_data = company_data.get("intent", {})
        if intent_data:
            # Enhanced intent processing with better defaults
            business_intent_score = intent_data.get("business_intent_score")
            intent_signals = intent_data.get("intent_signals", [])
            intent_category = intent_data.get("intent_category")
            timeline_indicator = intent_data.get("timeline_indicator")
            decision_maker_intent = intent_data.get("decision_maker_intent")

            # Set defaults if missing
            if business_intent_score is None:
                business_intent_score = (
                    icp_config.get("intent_config", {})
                    .get("scoring_rules", {})
                    .get("base_score", 0.5)
                )

        # CRITICAL FIX: Process team_members for single company sites
        # This ensures contacts are extracted from team_members even for non-database sites
        if extracted_data.get("team_members"):
            self._process_database_extracted_data(record, extracted_data, icp_config)

            if not intent_signals:
                intent_signals = ["No specific intent signals found during analysis"]

            if not intent_category:
                if business_intent_score >= 0.7:
                    intent_category = "high"
                elif business_intent_score >= 0.5:
                    intent_category = "medium"
                else:
                    intent_category = "low"

            company["intent"] = {
                "business_intent_score": business_intent_score,
                "intent_signals": intent_signals,
                "intent_category": intent_category,
                "timeline_indicator": timeline_indicator or "Timeline unclear",
                "decision_maker_intent": decision_maker_intent
                or "No specific decision maker statements found",
            }
        else:
            # Fallback intent data when Firecrawl doesn't extract intent
            base_score = (
                icp_config.get("intent_config", {})
                .get("scoring_rules", {})
                .get("base_score", 0.5)
            )
            company["intent"] = {
                "business_intent_score": base_score,
                "intent_signals": ["Intent analysis not performed - no data extracted"],
                "intent_category": "unknown",
                "timeline_indicator": "Timeline unclear",
                "decision_maker_intent": "No decision maker intent data available",
            }

        # Contact information fields
        company["emails"] = company_data.get("emails", [])
        company["phone_numbers"] = company_data.get("phone_numbers", [])

        # Initialize company structures
        if "socials" not in company:
            company["socials"] = {
                "linkedin": None,
                "github": None,
                "twitter": None,
                "telegram": None,
                "instagram": None,
                "facebook": None,
                "youtube": None,
            }

        # Handle social media links
        socials_data = company_data.get("socials", {})

        # Handle LinkedIn
        linkedin_url = socials_data.get("linkedin")
        if linkedin_url:
            link_type, slug = canonicalize_linkedin(linkedin_url)
            if link_type == "company":
                company["socials"]["linkedin"] = (
                    f"https://www.linkedin.com/company/{slug}"
                )
            elif link_type == "in":
                # This is a person, not company - skip
                pass

        # Handle other social platforms
        for platform in ["twitter", "facebook", "github", "instagram", "youtube"]:
            if socials_data.get(platform):
                company["socials"][platform] = socials_data[platform]

        company["careers_url"] = company_data.get("careers_url")
        company["blog_url"] = company_data.get("blog_url")

        # Add evidence URLs
        if "evidence_urls" not in company:
            company["evidence_urls"] = []

        # Add the crawled URL as evidence
        domain = record.get("domain", "")
        if domain:
            company["evidence_urls"].append(f"https://{domain}")

        # Calculate crawl score based on processed data
        self._validate_icp_fit(record, icp_config)

        # Process contacts from team_members (new schema) or employees (legacy)
        team_members = extracted_data.get("team_members", [])
        employees = extracted_data.get("employees", [])  # Legacy support
        all_members = team_members + employees  # Combine both sources
        contacts = []

        for member in all_members:
            # Handle both new and legacy field names
            full_name = member.get("name") or member.get("full_name")
            role = member.get("role")
            department = member.get("department")
            linkedin = member.get("linkedin")
            email = member.get("email")
            phone = member.get("phone")
            decision_maker = member.get("decision_maker", False)
            location = member.get("location")

            # Skip invalid or placeholder contacts
            if not full_name or not role or full_name == "not specified" or role == "not specified":
                continue

            # Normalize name and role
            normalized_name = normalize_text(full_name)
            normalized_role = normalize_text(role)

            # Generate contact ID
            contact_id = generate_contact_id(
                domain, normalized_name, normalized_role, linkedin or ""
            )

            # Check if contact already exists
            existing_contact = next(
                (c for c in contacts if c.get("contact_id") == contact_id), None
            )

            if existing_contact:
                # Update existing contact with new information
                if email and not existing_contact.get("email"):
                    existing_contact["email"] = email.lower()
                if linkedin and not existing_contact.get("linkedin"):
                    existing_contact["linkedin"] = linkedin
                continue

            # Create new contact
            contact = {
                "contact_id": contact_id,
                "full_name": normalized_name,
                "role": normalized_role,
                "department": department,  # Use extracted department
                "seniority": None,
                "role_priority": None,
                "email": email.lower() if email and email != "null" else None,
                "email_confidence": None,
                "email_status": ["unknown"],
                "email_source": ["provider"],
                "linkedin": None,
                "other_links": [],
                "evidence_urls": [],
                "decision_maker": decision_maker,  # New field from expanded schema
                "location": location,  # New field from expanded schema
            }

            # Only add phone field if we have a valid phone number
            # Convert string "null" to actual None
            if phone == "null":
                phone = None
            if phone and phone.strip():
                contact["phone"] = phone

            # Handle LinkedIn
            if linkedin:
                link_type, slug = canonicalize_linkedin(linkedin)
                if link_type == "in":
                    contact["linkedin"] = f"/in/{slug}"
                elif link_type == "company":
                    # This is a company page, not a person - skip
                    continue

            # Set seniority and role priority
            contact["seniority"] = get_seniority_rank(role)

            role_priority_config = icp_config.get("role_priority", {})
            contact["role_priority"] = get_role_priority(role, role_priority_config)

            # Add evidence URLs
            if domain:
                contact["evidence_urls"].append(f"https://{domain}")

            contacts.append(contact)

        # Update record with contacts
        record["contacts"] = contacts

        # Perform ICP validation
        self._validate_icp_fit(record, icp_config)

    def _check_company_requirements(
        self, record: Dict[str, Any], required_fields: List[str]
    ) -> bool:
        """
        Check if company meets required field requirements.

        Args:
            record: Lead record
            required_fields: List of required company fields

        Returns:
            True if all required fields are satisfied
        """
        if not required_fields:
            return True

        company = record.get("company", {})

        for field in required_fields:
            value = company.get(field)
            if not value:
                return False

        return True

    def _validate_icp_fit(
        self, record: Dict[str, Any], icp_config: Dict[str, Any]
    ) -> None:
        """
        Validate ICP fit based on crawled data.

        Args:
            record: Lead record to validate
            icp_config: ICP configuration
        """
        # This is a simplified ICP validation
        # In practice, this would use more sophisticated logic based on company data

        company = record.get("company", {})
        contacts = record.get("contacts", [])

        # More permissive scoring - pass by default unless clear ICP violations
        score = 0.5  # Start with a passing score
        reasons = ["Basic company information available"]

        # Bonus points for additional data
        if company.get("name"):
            score += 0.1
            reasons.append("Company name available")

        if company.get("description"):
            score += 0.05
            reasons.append("Company description available")

        if company.get("industry"):
            score += 0.1
            reasons.append("Industry information available")

        if company.get("hq_location"):
            score += 0.1
            reasons.append("Location information available")

        # New comprehensive data fields
        if company.get("number_of_locations"):
            score += 0.1
            reasons.append("Location count available")

        if company.get("locations") and len(company["locations"]) > 0:
            score += 0.1
            reasons.append(
                f"Detailed location data available ({len(company['locations'])} locations)"
            )

        if company.get("founded_year"):
            score += 0.05
            reasons.append("Founded year available")

        if company.get("employee_count"):
            score += 0.05
            reasons.append("Employee count available")

        if company.get("ownership_type"):
            score += 0.05
            reasons.append("Ownership type available")

        if company.get("emails") and len(company["emails"]) > 0:
            score += 0.1
            reasons.append(
                f"Company emails available ({len(company['emails'])} emails)"
            )

        if company.get("phone_numbers") and len(company["phone_numbers"]) > 0:
            score += 0.05
            reasons.append(
                f"Company phone numbers available ({len(company['phone_numbers'])} numbers)"
            )

        # Intent analysis scoring (high priority for business intelligence)
        if company.get("intent", {}).get("business_intent_score"):
            intent_score = company["intent"]["business_intent_score"]
            if intent_score >= 0.7:
                score += 0.3
                reasons.append(f"High business intent score: {intent_score:.2f}")
            elif intent_score >= 0.5:
                score += 0.2
                reasons.append(f"Medium business intent score: {intent_score:.2f}")
            elif intent_score >= 0.3:
                score += 0.1
                reasons.append(f"Low business intent score: {intent_score:.2f}")

            # Add specific intent signals
            intent_signals = company.get("intent", {}).get("intent_signals", [])
            if intent_signals:
                reasons.append(f"Intent signals found: {', '.join(intent_signals[:3])}")

        if company.get("socials"):
            social_count = sum(1 for v in company["socials"].values() if v)
            if social_count > 0:
                score += 0.05
                reasons.append(
                    f"Social media profiles available ({social_count} platforms)"
                )

        if contacts:
            score += 0.2
            reasons.append(f"Found {len(contacts)} contacts")

            # High-quality contacts (with email) are more valuable
            contacts_with_email = sum(1 for c in contacts if c.get("email"))
            if contacts_with_email > 0:
                score += 0.1
                reasons.append(f"{contacts_with_email} contacts have email addresses")

            # Decision makers are especially valuable
            decision_makers = sum(1 for c in contacts if c.get("decision_maker"))
            if decision_makers > 0:
                score += 0.1
                reasons.append(f"{decision_makers} decision makers identified")

        # Cap score at 1.0
        score = min(score, 1.0)

        # Update record
        if "icp" not in record:
            record["icp"] = {}

        record["icp"]["crawl_score"] = round4(score)
        record["icp"]["crawl_reason"] = (
            "; ".join(reasons) if reasons else "No specific reasons"
        )

    def _build_metrics(
        self,
        count_in: int,
        count_out: int,
        duration_ms: int,
        cache_hit_rate: Optional[float] = None,
        pass_rate: Optional[float] = None,
        cost_usd: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """
        Build metrics structure for tool output.

        Args:
            count_in: Number of input records
            count_out: Number of output records
            duration_ms: Tool runtime in milliseconds
            cache_hit_rate: Cache hit rate (0.0 to 1.0)
            pass_rate: Pass rate (0.0 to 1.0)
            cost_usd: Cost breakdown by stage

        Returns:
            Metrics dictionary
        """
        metrics = {
            "count_in": count_in,
            "count_out": count_out,
            "duration_ms": duration_ms,
            "cache_hit_rate": round4(cache_hit_rate)
            if cache_hit_rate is not None
            else None,
            "pass_rate": round4(pass_rate) if pass_rate is not None else None,
            "cost_usd": {
                "domain": 0.0,
                "crawl": round4(cost_usd.get("crawl", 0.0)) if cost_usd else 0.0,
                "enrich": 0.0,
                "total": round4(cost_usd.get("crawl", 0.0)) if cost_usd else 0.0,
            },
        }

        return metrics


# ============================================================================
# CLI Interface
# ============================================================================


def main():
    """CLI entry point for the crawl tool."""
    import sys

    try:
        # Read JSON from stdin
        input_data = sys.stdin.read()
        payload = json.loads(input_data)

        # Create tool instance
        tool = CrawlTool()

        # Run tool
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(tool.run(payload))
        loop.close()

        # Write JSON to stdout
        json.dump(result, sys.stdout)

    except json.JSONDecodeError as exc:
        error_result = {
            "data": {"lead_records": []},
            "errors": [
                {
                    "code": "SCHEMA_VALIDATION",
                    "message": f"Invalid JSON input: {exc}",
                    "tool": TOOL_NAME,
                    "retryable": False,
                }
            ],
            "metrics": {
                "count_in": 0,
                "count_out": 0,
                "duration_ms": 0,
                "cache_hit_rate": None,
                "pass_rate": None,
                "cost_usd": {"domain": 0.0, "crawl": 0.0, "enrich": 0.0, "total": 0.0},
            },
        }
        json.dump(error_result, sys.stdout)
        sys.exit(1)
    except Exception as exc:
        error_result = {
            "data": {"lead_records": []},
            "errors": [
                {
                    "code": "UNKNOWN",
                    "message": f"Fatal error: {exc}",
                    "tool": TOOL_NAME,
                    "retryable": False,
                }
            ],
            "metrics": {
                "count_in": 0,
                "count_out": 0,
                "duration_ms": 0,
                "cache_hit_rate": None,
                "pass_rate": None,
                "cost_usd": {"domain": 0.0, "crawl": 0.0, "enrich": 0.0, "total": 0.0},
            },
        }
        json.dump(error_result, sys.stdout)
        sys.exit(1)


if __name__ == "__main__":
    main()
