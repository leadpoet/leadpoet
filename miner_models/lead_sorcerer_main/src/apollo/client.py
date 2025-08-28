"""
Apollo API Client for Lead Sorcerer

This module provides a comprehensive client for interacting with Apollo's APIs,
including company and person search, enrichment, and bulk operations.
"""

import asyncio
import logging
import os
import time
import json
import hashlib
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta

import httpx
from pydantic import BaseModel, Field

# Configure logging with PII protection
logger = logging.getLogger(__name__)


class ApolloConfig(BaseModel):
    """Configuration for Apollo API client"""

    api_key: str = Field(..., description="Apollo API key")
    refresh_token: Optional[str] = Field(
        None, description="Apollo refresh token for OAuth"
    )
    base_url: str = Field(
        default="https://api.apollo.io/api/v1", description="Apollo API base URL"
    )
    timeout_seconds: int = Field(default=30, description="Request timeout in seconds")
    max_retries: int = Field(default=3, description="Maximum number of retry attempts")
    backoff_factor: float = Field(default=2.0, description="Exponential backoff factor")
    rate_limit_delay_ms: int = Field(
        default=100, description="Delay between requests in milliseconds"
    )


class ApolloError(Exception):
    """Base exception for Apollo API errors"""

    pass


class ApolloRateLimitError(ApolloError):
    """Exception raised when Apollo API rate limit is exceeded"""

    def __init__(self, retry_after: int, message: str = "Rate limit exceeded"):
        self.retry_after = retry_after
        super().__init__(f"{message}. Retry after {retry_after} seconds")


class ApolloAuthenticationError(ApolloError):
    """Exception raised when Apollo API authentication fails"""

    pass


@dataclass
class BatchConfig:
    """Configuration for intelligent batching"""

    max_batch_size: int = 50
    min_batch_size: int = 10
    target_response_time: float = 2.0  # seconds
    max_concurrent_batches: int = 3
    adaptive_batch_sizing: bool = True
    rate_limit_buffer: float = 0.1  # 10% buffer for rate limits


class BatchProcessor:
    """Handles intelligent batching of API calls"""

    def __init__(self, config: BatchConfig):
        self.config = config
        self.performance_history: List[float] = []
        self.rate_limit_history: List[float] = []
        self.current_batch_size = config.max_batch_size

    def calculate_optimal_batch_size(self, recent_performance: List[float]) -> int:
        """Calculate optimal batch size based on performance history"""
        if not recent_performance or len(recent_performance) < 3:
            return self.current_batch_size

        avg_response_time = sum(recent_performance) / len(recent_performance)

        if avg_response_time < self.config.target_response_time * 0.8:
            # Performance is good, can increase batch size
            new_size = min(self.current_batch_size + 10, self.config.max_batch_size)
        elif avg_response_time > self.config.target_response_time * 1.2:
            # Performance is poor, decrease batch size
            new_size = max(self.current_batch_size - 10, self.config.min_batch_size)
        else:
            new_size = self.current_batch_size

        self.current_batch_size = new_size
        return new_size

    def should_adapt_batch_size(self) -> bool:
        """Determine if batch size should be adapted"""
        if not self.config.adaptive_batch_sizing:
            return False

        # Adapt every 10 API calls
        return len(self.performance_history) % 10 == 0

    def record_performance(
        self, response_time: float, rate_limit_remaining: Optional[int] = None
    ):
        """Record performance metrics for batch size optimization"""
        self.performance_history.append(response_time)

        if rate_limit_remaining is not None:
            self.rate_limit_history.append(rate_limit_remaining)

        # Keep only recent history
        if len(self.performance_history) > 50:
            self.performance_history = self.performance_history[-50:]
        if len(self.rate_limit_history) > 50:
            self.rate_limit_history = self.rate_limit_history[-50:]


@dataclass
class CacheConfig:
    """Configuration for caching layer"""

    max_cache_size: int = 1000
    default_ttl: int = 3600  # 1 hour in seconds
    company_ttl: int = 7200  # 2 hours for company data
    person_ttl: int = 1800  # 30 minutes for person data
    enable_compression: bool = True
    cache_stats: bool = True


class CacheEntry:
    """Single cache entry with metadata"""

    def __init__(self, data: Any, ttl: int):
        self.data = data
        self.created_at = datetime.now()
        self.ttl = ttl
        self.access_count = 0
        self.last_accessed = datetime.now()

    def is_expired(self) -> bool:
        """Check if cache entry has expired"""
        return datetime.now() > self.created_at + timedelta(seconds=self.ttl)

    def access(self):
        """Record access to cache entry"""
        self.access_count += 1
        self.last_accessed = datetime.now()

    def get_age(self) -> float:
        """Get age of cache entry in seconds"""
        return (datetime.now() - self.created_at).total_seconds()


class ApolloCache:
    """Intelligent caching layer for Apollo API responses"""

    def __init__(self, config: CacheConfig):
        self.config = config
        self.cache: Dict[str, CacheEntry] = {}
        self.stats = {"hits": 0, "misses": 0, "evictions": 0, "total_requests": 0}

    def _generate_cache_key(self, endpoint: str, params: Dict[str, Any]) -> str:
        """Generate unique cache key for request"""
        # Sort parameters for consistent key generation
        sorted_params = json.dumps(params, sort_keys=True)
        key_data = f"{endpoint}:{sorted_params}"

        if self.config.enable_compression:
            # Use hash for shorter keys
            return hashlib.md5(key_data.encode()).hexdigest()
        else:
            return key_data

    def get(
        self, endpoint: str, params: Dict[str, Any]
    ) -> Optional[Tuple[Dict[str, Any], float]]:
        """Get cached response if available and not expired"""
        cache_key = self._generate_cache_key(endpoint, params)
        self.stats["total_requests"] += 1

        if cache_key in self.cache:
            entry = self.cache[cache_key]

            if entry.is_expired():
                # Remove expired entry
                del self.cache[cache_key]
                self.stats["evictions"] += 1
                self.stats["misses"] += 1
                return None

            # Cache hit
            entry.access()
            self.stats["hits"] += 1
            logger.debug(f"Cache hit for {endpoint}")
            return (entry.data, 0.0)  # Return cached data and 0 response time

        self.stats["misses"] += 1
        return None

    def set(
        self,
        endpoint: str,
        params: Dict[str, Any],
        data: Any,
        ttl: Optional[int] = None,
    ):
        """Cache API response with TTL"""
        if len(self.cache) >= self.config.max_cache_size:
            self._evict_oldest()

        cache_key = self._generate_cache_key(endpoint, params)

        # Determine TTL based on endpoint type
        if ttl is None:
            if "companies" in endpoint:
                ttl = self.config.company_ttl
            elif "people" in endpoint:
                ttl = self.config.person_ttl
            else:
                ttl = self.config.default_ttl

        self.cache[cache_key] = CacheEntry(data, ttl)
        logger.debug(f"Cached response for {endpoint} with TTL {ttl}s")

    def _evict_oldest(self):
        """Evict oldest cache entries when cache is full"""
        if not self.cache:
            return

        # Find entries to evict (oldest and least accessed)
        entries = list(self.cache.items())
        entries.sort(key=lambda x: (x[1].access_count, x[1].get_age()), reverse=True)

        # Evict 10% of cache
        evict_count = max(1, len(entries) // 10)

        for i in range(evict_count):
            if entries:
                key, _ = entries.pop()
                del self.cache[key]
                self.stats["evictions"] += 1

        logger.info(f"Evicted {evict_count} cache entries")

    def invalidate_pattern(self, pattern: str):
        """Invalidate cache entries matching pattern"""
        keys_to_remove = [key for key in self.cache.keys() if pattern in key]

        for key in keys_to_remove:
            del self.cache[key]
            self.stats["evictions"] += 1

        if keys_to_remove:
            logger.info(
                f"Invalidated {len(keys_to_remove)} cache entries matching '{pattern}'"
            )

    def clear(self):
        """Clear all cache entries"""
        count = len(self.cache)
        self.cache.clear()
        self.stats["evictions"] += count
        logger.info(f"Cleared {count} cache entries")

    def get_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics"""
        if not self.stats["total_requests"]:
            hit_rate = 0.0
        else:
            hit_rate = (self.stats["hits"] / self.stats["total_requests"]) * 100

        return {
            **self.stats,
            "hit_rate_percent": round(hit_rate, 2),
            "cache_size": len(self.cache),
            "cache_utilization": (len(self.cache) / self.config.max_cache_size) * 100,
        }

    def get_cache_info(self) -> Dict[str, Any]:
        """Get detailed cache information"""
        if not self.cache:
            return {"cache_size": 0, "entries": []}

        entries_info = []
        for key, entry in list(self.cache.items())[:10]:  # Show first 10 entries
            entries_info.append(
                {
                    "key": key[:20] + "..." if len(key) > 20 else key,
                    "age_seconds": round(entry.get_age(), 1),
                    "access_count": entry.access_count,
                    "expires_in": round(entry.ttl - entry.get_age(), 1),
                }
            )

        return {
            "cache_size": len(self.cache),
            "max_size": self.config.max_cache_size,
            "sample_entries": entries_info,
        }


@dataclass
class SearchOptimizationConfig:
    """Configuration for search query optimization"""

    enable_query_analysis: bool = True
    max_parallel_searches: int = 5
    query_timeout: float = 30.0
    result_quality_threshold: float = 0.7
    enable_fallback_strategies: bool = True
    max_retry_variations: int = 3


class SearchQueryOptimizer:
    """Optimizes search queries for better results and performance"""

    def __init__(self, config: SearchOptimizationConfig):
        self.config = config
        self.query_performance_history: Dict[str, List[float]] = {}
        self.successful_patterns: List[str] = []

    def optimize_company_search(
        self, base_filters: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate optimized company search variations"""
        variations = []

        # Base search
        variations.append(base_filters.copy())

        if not self.config.enable_query_analysis:
            return variations

        # Industry-based variations
        if "industry" in base_filters:
            industry = base_filters["industry"]
            if isinstance(industry, list) and len(industry) > 1:
                # Split large industry lists into smaller chunks
                for i in range(0, len(industry), 3):
                    chunk_filters = base_filters.copy()
                    chunk_filters["industry"] = industry[i : i + 3]
                    variations.append(chunk_filters)

        # Location-based variations
        if "country" in base_filters:
            country = base_filters["country"]
            if isinstance(country, list) and len(country) > 2:
                # Split countries into smaller groups
                for i in range(0, len(country), 2):
                    chunk_filters = base_filters.copy()
                    chunk_filters["country"] = country[i : i + 2]
                    variations.append(chunk_filters)

        # Employee count variations
        if "employee_count" in base_filters:
            emp_range = base_filters["employee_count"]
            if (
                isinstance(emp_range, dict)
                and "min" in emp_range
                and "max" in emp_range
            ):
                # Create overlapping ranges for better coverage
                min_emp = emp_range["min"]
                max_emp = emp_range["max"]
                mid_point = (min_emp + max_emp) // 2

                variations.append(
                    {
                        **base_filters,
                        "employee_count": {"min": min_emp, "max": mid_point},
                    }
                )
                variations.append(
                    {
                        **base_filters,
                        "employee_count": {"min": mid_point, "max": max_emp},
                    }
                )

        return variations[: self.config.max_retry_variations]

    def optimize_person_search(
        self, base_filters: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate optimized person search variations"""
        variations = []

        # Base search
        variations.append(base_filters.copy())

        if not self.config.enable_query_analysis:
            return variations

        # Job title variations
        if "job_titles" in base_filters:
            job_titles = base_filters["job_titles"]
            if isinstance(job_titles, list) and len(job_titles) > 2:
                # Split job titles into smaller groups
                for i in range(0, len(job_titles), 2):
                    chunk_filters = base_filters.copy()
                    chunk_filters["job_titles"] = job_titles[i : i + 2]
                    variations.append(chunk_filters)

        # Seniority level variations
        if "seniority" in base_filters:
            seniority = base_filters["seniority"]
            if isinstance(seniority, list) and len(seniority) > 1:
                # Create individual seniority searches
                for level in seniority:
                    level_filters = base_filters.copy()
                    level_filters["seniority"] = [level]
                    variations.append(level_filters)

        return variations[: self.config.max_retry_variations]

    def record_query_performance(
        self, query_hash: str, response_time: float, result_count: int
    ):
        """Record performance metrics for query optimization"""
        if query_hash not in self.query_performance_history:
            self.query_performance_history[query_hash] = []

        self.query_performance_history[query_hash].append(response_time)

        # Keep only recent history
        if len(self.query_performance_history[query_hash]) > 10:
            self.query_performance_history[query_hash] = self.query_performance_history[
                query_hash
            ][-10:]

    def get_optimal_queries(
        self, base_filters: Dict[str, Any], search_type: str = "company"
    ) -> List[Dict[str, Any]]:
        """Get optimized search queries based on performance history"""
        if search_type == "company":
            variations = self.optimize_company_search(base_filters)
        else:
            variations = self.optimize_person_search(base_filters)

        # Sort by historical performance if available
        def sort_key(filters):
            query_hash = str(hash(frozenset(sorted(filters.items()))))
            if query_hash in self.query_performance_history:
                avg_time = sum(self.query_performance_history[query_hash]) / len(
                    self.query_performance_history[query_hash]
                )
                return avg_time
            else:
                return float("inf")  # Unknown queries go last

        variations.sort(key=sort_key)
        return variations


class ApolloClient:
    """Enhanced Apollo API client with intelligent batching, caching, and search optimization"""

    def __init__(self, config: ApolloConfig):
        self.config = config
        self.api_key = config.api_key
        self.base_url = config.base_url
        self.session = None
        self.batch_processor = BatchProcessor(BatchConfig())
        self.cache = ApolloCache(CacheConfig())
        self.search_optimizer = SearchQueryOptimizer(SearchOptimizationConfig())
        self.semaphore = asyncio.Semaphore(3)  # Limit concurrent requests
        self.credits_used = 0  # Track API credits used
        self.total_requests = 0  # Track total API requests made
        self.total_errors = 0  # Track total API errors

    async def __aenter__(self):
        logger.info(f"Initializing Apollo session with API key: {self.api_key[:10]}...")
        logger.info(f"Base URL: {self.base_url}")

        self.session = httpx.AsyncClient(
            headers={
                "X-Api-Key": self.api_key,
                "Content-Type": "application/json",
                "Cache-Control": "no-cache",
                "Accept": "application/json",
            },
            timeout=30.0,
        )
        logger.info(f"Session headers: {self.session.headers}")
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.aclose()

    async def _make_request(
        self, endpoint: str, params: Dict[str, Any] = None, method: str = "POST"
    ) -> Tuple[Dict[str, Any], float]:
        """Make a single API request with timing and caching"""
        start_time = time.time()

        # Default to empty dict if no params provided
        if params is None:
            params = {}

        # Check cache first
        cached_response = self.cache.get(endpoint, params)
        if cached_response:
            logger.debug(f"Cache hit for {endpoint}")
            return cached_response, 0.0  # 0 response time for cache hits

        async with self.semaphore:
            try:
                # Debug logging to see what we're sending
                logger.debug(
                    f"Making Apollo API {method} request to: {self.base_url}/{endpoint}"
                )
                logger.debug(f"Headers: {self.session.headers}")
                if method == "POST":
                    logger.debug(f"Request body: {params}")

                # Apollo API uses POST for search endpoints, GET for organization details
                if method == "GET":
                    response = await self.session.get(f"{self.base_url}/{endpoint}")
                else:
                    response = await self.session.post(
                        f"{self.base_url}/{endpoint}", json=params
                    )

                response.raise_for_status()

                response_time = time.time() - start_time
                rate_limit_remaining = response.headers.get("X-RateLimit-Remaining")

                # Parse response
                response_data = response.json()

                # Cache successful response
                self.cache.set(endpoint, params, response_data)

                # Track credit usage (1 credit per API call)
                self.credits_used += 1
                self.total_requests += 1

                # Record performance for batch optimization
                self.batch_processor.record_performance(
                    response_time,
                    int(rate_limit_remaining) if rate_limit_remaining else None,
                )

                return response_data, response_time

            except Exception as e:
                response_time = time.time() - start_time
                self.total_errors += 1
                self.batch_processor.record_performance(response_time)
                raise e

    async def search_companies(
        self,
        filters: Dict[str, Any] = None,
        max_results: int = 25,
        page: int = 1,
        **kwargs,
    ) -> Dict[str, Any]:
        """Search companies using Apollo API - Apollo doesn't have dedicated company search, use mixed_people/search"""
        # Map our generic filters to Apollo's expected API parameters
        apollo_params = self._map_company_filters_to_apollo(filters or {})

        params = {
            "page": page,
            "per_page": min(max_results, 100),  # Apollo max per page is 100
            **apollo_params,
            **kwargs,
        }

        logger.info(f"Searching companies with params: {params}")
        logger.info(f"Apollo search filters: {filters}")
        logger.info(f"Mapped Apollo params: {apollo_params}")

        # Apollo only has mixed_people/search endpoint - no dedicated companies/search
        response, response_time = await self._make_request(
            "mixed_people/search", params
        )

        # Log response summary
        total_entries = response.get("pagination", {}).get("total_entries", 0)
        people_count = len(response.get("people", []))
        logger.info(
            f"Apollo search response: {total_entries} total entries, {people_count} people returned"
        )

        return response

    def _map_company_filters_to_apollo(self, filters: Dict[str, Any]) -> Dict[str, Any]:
        """Map our generic company filters to Apollo's API parameters"""
        apollo_params = {}

        # Map industry filters
        if "industry" in filters:
            apollo_params["q_organization_industries_list[]"] = filters["industry"]

        # Map size filters (employee count)
        if "size" in filters:
            size_mapping = {
                "1-10": {"min_employees": 1, "max_employees": 10},
                "11-50": {"min_employees": 11, "max_employees": 50},
                "51-200": {"min_employees": 51, "max_employees": 200},
                "201-500": {"min_employees": 201, "max_employees": 500},
            }
            for size in filters["size"]:
                if size in size_mapping:
                    apollo_params.update(size_mapping[size])

        # Map location filters
        if "location" in filters:
            apollo_params["q_organization_locations_list[]"] = filters["location"]

        # Map technology filters
        if "technologies" in filters:
            apollo_params["q_organization_technologies_list[]"] = filters[
                "technologies"
            ]

        # Map funding stage
        if "funding_stage" in filters:
            apollo_params["q_organization_funding_stages_list[]"] = filters[
                "funding_stage"
            ]

        return apollo_params

    async def search_persons(
        self,
        filters: Dict[str, Any] = None,
        max_results: int = 25,
        page: int = 1,
        **kwargs,
    ) -> Dict[str, Any]:
        """Search persons using Apollo API"""
        # Map our generic filters to Apollo's expected API parameters
        apollo_params = self._map_person_filters_to_apollo(filters or {})

        params = {
            "page": page,
            "per_page": min(max_results, 100),  # Apollo max per page is 100
            **apollo_params,
        }
        # Apollo uses mixed_people/search for both company and person searches
        response, response_time = await self._make_request(
            "mixed_people/search", params
        )
        return response

    def _map_person_filters_to_apollo(self, filters: Dict[str, Any]) -> Dict[str, Any]:
        """Map our generic person filters to Apollo's API parameters"""
        apollo_params = {}

        # Map job title filters
        if "job_titles" in filters:
            apollo_params["person_titles[]"] = filters["job_titles"]

        # Map seniority filters
        if "seniority" in filters:
            apollo_params["person_seniorities[]"] = filters["seniority"]

        # Map department filters
        if "departments" in filters:
            apollo_params["person_departments[]"] = filters["departments"]

        return apollo_params

    async def bulk_enrich_companies(self, domains: List[str]) -> List[Dict[str, Any]]:
        """Bulk enrich companies using Apollo API"""
        if not domains:
            return []

        results = []
        for domain in domains:
            try:
                # Use company search with domain filter
                params = {"q_organization_domains_list[]": [domain], "per_page": 1}
                response, _ = await self._make_request("mixed_people/search", params)
                companies = response.get("companies", [])
                if companies:
                    results.append({"company": companies[0]})
                else:
                    results.append(None)
            except Exception as e:
                logger.error(f"Company enrichment failed for {domain}: {e}")
                results.append(None)

        return results

    async def enrich_person(
        self,
        email: str = None,
        first_name: str = None,
        last_name: str = None,
        organization_domain: str = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Enrich a single person using Apollo API people/match endpoint"""
        params = {}

        if email:
            params["email"] = email
        elif first_name and last_name and organization_domain:
            params["first_name"] = first_name
            params["last_name"] = last_name
            params["organization_domain"] = organization_domain
        else:
            raise ValueError(
                "Must provide either email OR first_name + last_name + organization_domain"
            )

        # Add optional enrichment parameters
        if kwargs.get("reveal_personal_emails"):
            params["reveal_personal_emails"] = kwargs["reveal_personal_emails"]
        if kwargs.get("reveal_phone_number"):
            params["reveal_phone_number"] = kwargs["reveal_phone_number"]
        if kwargs.get("webhook_url"):
            params["webhook_url"] = kwargs["webhook_url"]

        response, _ = await self._make_request("people/match", params)
        return response

    async def bulk_enrich_persons(
        self, person_data: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Bulk enrich persons using Apollo API people/bulk_match endpoint (max 10 per call)"""
        if not person_data or len(person_data) > 10:
            raise ValueError("Apollo bulk enrichment supports max 10 persons per call")

        # Transform person data to Apollo's expected format
        bulk_params = []
        for person in person_data:
            person_param = {}
            if person.get("email"):
                person_param["email"] = person["email"]
            elif (
                person.get("first_name")
                and person.get("last_name")
                and person.get("organization_domain")
            ):
                person_param["first_name"] = person["first_name"]
                person_param["last_name"] = person["last_name"]
                person_param["organization_domain"] = person["organization_domain"]
            else:
                continue  # Skip invalid person data

            # Add optional parameters
            if person.get("reveal_personal_emails"):
                person_param["reveal_personal_emails"] = person[
                    "reveal_personal_emails"
                ]
            if person.get("reveal_phone_number"):
                person_param["reveal_phone_number"] = person["reveal_phone_number"]
            if person.get("webhook_url"):
                person_param["webhook_url"] = person["webhook_url"]

            bulk_params.append(person_param)

        if not bulk_params:
            return []

        response, _ = await self._make_request(
            "people/bulk_match", {"people": bulk_params}
        )
        return response.get("people", [])

    async def enrich_person_for_email(
        self, person_data: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Enrich a person to get their actual email using Apollo People Enrichment API.

        Args:
            person_data: Person data from search (must have email or name+domain)

        Returns:
            Enriched person data with email or None if failed
        """
        try:
            # Build enrichment parameters based on available data
            params = {}

            if person_data.get("email") and not person_data.get("email", "").startswith(
                "email_not_unlocked"
            ):
                # If we already have a real email, use it for enrichment
                params["email"] = person_data["email"]
            elif (
                person_data.get("first_name")
                and person_data.get("last_name")
                and person_data.get("organization", {}).get("primary_domain")
            ):
                # Use name + domain + company for enrichment - these are the exact parameter names Apollo expects
                params["first_name"] = person_data["first_name"]
                params["last_name"] = person_data["last_name"]
                params["organization_domain"] = person_data["organization"][
                    "primary_domain"
                ]

                # Adding organization_name is CRITICAL for successful email matching
                if person_data.get("organization", {}).get("name"):
                    params["organization_name"] = person_data["organization"]["name"]
            else:
                logger.warning(
                    f"Cannot enrich person {person_data.get('name', 'Unknown')}: missing email or name+domain"
                )
                return None

            # Add required parameters to reveal emails and phone numbers
            # These are the exact parameter names from Apollo documentation
            params["reveal_personal_emails"] = True

            # Note: reveal_phone_number requires webhook_url according to Apollo API
            # For now, we'll focus on email enrichment only
            # params["reveal_phone_number"] = True

            # Debug logging to see what we're sending
            logger.info(f"Enriching person with params: {params}")

            # Use People Enrichment endpoint
            response, response_time = await self._make_request("people/match", params)

            # Log response to understand structure
            logger.info(
                f"People enrichment response keys: {list(response.keys()) if response else 'None'}"
            )

            return response.get("person") if response else None
        except Exception as e:
            logger.error(
                f"Person enrichment failed for {person_data.get('name', 'Unknown')}: {e}"
            )
            return None

    async def bulk_enrich_people_for_emails(
        self, people_data: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Bulk enrich people to get their actual emails using Apollo API.

        Args:
            people_data: List of person data from search

        Returns:
            List of enriched person data with emails
        """
        if not people_data:
            return []

        # Filter people who can be enriched
        enrichable_people = []
        for person in people_data:
            if (
                person.get("email")
                and not person.get("email", "").startswith("email_not_unlocked")
            ) or (
                person.get("first_name")
                and person.get("last_name")
                and person.get("organization", {}).get("primary_domain")
            ):
                enrichable_people.append(person)

        if not enrichable_people:
            logger.info("No people found that can be enriched for emails")
            return []

        # Apollo bulk enrichment supports max 10 per call
        batch_size = min(10, len(enrichable_people))
        enriched_people = []

        for i in range(0, len(enrichable_people), batch_size):
            batch = enrichable_people[i : i + batch_size]
            try:
                # Build bulk enrichment parameters
                bulk_params = []
                for person in batch:
                    person_param = {}

                    if person.get("email") and not person.get("email", "").startswith(
                        "email_not_unlocked"
                    ):
                        person_param["email"] = person["email"]
                    elif (
                        person.get("first_name")
                        and person.get("last_name")
                        and person.get("organization", {}).get("primary_domain")
                    ):
                        person_param["first_name"] = person["first_name"]
                        person_param["last_name"] = person["last_name"]
                        person_param["organization_domain"] = person["organization"][
                            "primary_domain"
                        ]

                    # Add optional parameters for better results
                    person_param["reveal_personal_emails"] = "true"
                    person_param["reveal_phone_number"] = "true"

                    bulk_params.append(person_param)

                if bulk_params:
                    # For bulk enrichment, send data in request body as JSON
                    # Apollo bulk_match expects: {"people": [{"first_name": "...", "last_name": "...", "organization_domain": "..."}]}
                    request_body = {"people": bulk_params}

                    # Debug logging to see what we're sending
                    logger.info(f"Bulk enrichment request body: {request_body}")

                    response, response_time = await self._make_request(
                        "people/bulk_match", request_body
                    )
                    enriched_people.extend(response.get("people", []))

                    # Add rate limiting delay
                    await asyncio.sleep(0.1)

            except Exception as e:
                logger.error(
                    f"Bulk person enrichment failed for batch {i // batch_size + 1}: {e}"
                )
                # Continue with next batch
                continue

        return enriched_people

    async def enrich_company(self, domain: str) -> Dict[str, Any]:
        """Enrich a company using Apollo API - since there's no dedicated company enrichment, use mixed_people/search"""
        params = {"q_organization_domains_list[]": [domain], "per_page": 1}
        response, _ = await self._make_request("mixed_people/search", params)
        return response

    async def search_companies_batched(
        self, search_params: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Search companies using intelligent batching"""
        if not search_params:
            return []

        # Calculate optimal batch size
        if self.batch_processor.should_adapt_batch_size():
            optimal_size = self.batch_processor.calculate_optimal_batch_size(
                self.batch_processor.performance_history
            )
            logger.info(f"Adapted batch size to: {optimal_size}")

        batch_size = self.batch_processor.current_batch_size
        results = []

        # Process in batches
        for i in range(0, len(search_params), batch_size):
            batch = search_params[i : i + batch_size]
            batch_results = await self._process_company_batch(batch)
            results.extend(batch_results)

            # Small delay between batches to respect rate limits
            if i + batch_size < len(search_params):
                await asyncio.sleep(0.1)

        return results

    async def _process_company_batch(
        self, batch_params: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Process a single batch of company search parameters"""
        tasks = []

        for params in batch_params:
            task = self._make_request("mixed_people/search", params)
            tasks.append(task)

        # Execute batch concurrently
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out exceptions and extract results
        valid_results = []
        for result in batch_results:
            if isinstance(result, Exception):
                logger.error(f"Batch request failed: {result}")
                continue
            data, _ = result
            if "companies" in data:
                valid_results.extend(data["companies"])

        return valid_results

    async def search_persons_batched(
        self, search_params: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Search persons using intelligent batching"""
        if not search_params:
            return []

        batch_size = self.batch_processor.current_batch_size
        results = []

        # Process in batches
        for i in range(0, len(search_params), batch_size):
            batch = search_params[i : i + batch_size]
            batch_results = await self._process_person_batch(batch)
            results.extend(batch_results)

            # Small delay between batches
            if i + batch_size < len(search_params):
                await asyncio.sleep(0.1)

        return results

    async def _process_person_batch(
        self, batch_params: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Process a single batch of person search parameters"""
        tasks = []

        for params in batch_params:
            task = self._make_request("mixed_people/search", params)
            tasks.append(task)

        # Execute batch concurrently
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out exceptions and extract results
        valid_results = []
        for result in batch_results:
            if isinstance(result, Exception):
                logger.error(f"Batch request failed: {result}")
                continue
            data, _ = result
            if "people" in data:
                valid_results.extend(data["people"])

        return valid_results

    async def enrich_companies_batched(
        self, company_ids: List[str]
    ) -> List[Dict[str, Any]]:
        """Enrich companies using intelligent batching"""
        if not company_ids:
            return []

        batch_size = self.batch_processor.current_batch_size
        results = []

        # Process in batches
        for i in range(0, len(company_ids), batch_size):
            batch_ids = company_ids[i : i + batch_size]
            batch_results = await self._process_enrichment_batch(batch_ids, "companies")
            results.extend(batch_results)

            # Small delay between batches
            if i + batch_size < len(company_ids):
                await asyncio.sleep(0.1)

        return results

    async def enrich_persons_batched(
        self, person_ids: List[str]
    ) -> List[Dict[str, Any]]:
        """Enrich persons using intelligent batching"""
        if not person_ids:
            return []

        batch_size = self.batch_processor.current_batch_size
        results = []

        # Process in batches
        for i in range(0, len(person_ids), batch_size):
            batch_ids = person_ids[i : i + batch_size]
            batch_results = await self._process_enrichment_batch(batch_ids, "people")
            results.extend(batch_results)

            # Small delay between batches
            if i + batch_size < len(person_ids):
                await asyncio.sleep(0.1)

        return results

    async def _process_enrichment_batch(
        self, entity_ids: List[str], entity_type: str
    ) -> List[Dict[str, Any]]:
        """Process a single batch of enrichment requests"""
        tasks = []

        for entity_id in entity_ids:
            params = {"id": entity_id}
            task = self._make_request(f"{entity_type}/enrich", params)
            tasks.append(task)

        # Execute batch concurrently
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out exceptions and extract results
        valid_results = []
        for result in batch_results:
            if isinstance(result, Exception):
                logger.error(f"Enrichment batch request failed: {result}")
                continue
            data, _ = result
            if entity_type in data:
                valid_results.append(data[entity_type])

        return valid_results

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics for monitoring"""
        batch_metrics = {
            "current_batch_size": self.batch_processor.current_batch_size,
            "performance_history": self.batch_processor.performance_history[-10:],
            "rate_limit_history": self.batch_processor.rate_limit_history[-10:],
            "avg_response_time": (
                sum(self.batch_processor.performance_history)
                / len(self.batch_processor.performance_history)
                if self.batch_processor.performance_history
                else 0
            ),
        }

        cache_metrics = self.cache.get_stats()

        # Add comprehensive performance monitoring
        performance_summary = {
            "batch_processing": batch_metrics,
            "caching": cache_metrics,
            "concurrent_requests": self.semaphore._value,
            "search_optimization": self.get_search_optimization_stats(),
            "overall_performance": {
                "total_requests": batch_metrics.get("performance_history", []),
                "cache_hit_rate": cache_metrics.get("hit_rate_percent", 0),
                "avg_batch_size": batch_metrics.get("current_batch_size", 0),
                "rate_limit_utilization": self._calculate_rate_limit_utilization(),
            },
        }

        # Log performance metrics for monitoring
        logger.info(f"Performance metrics: {performance_summary}")

        return performance_summary

    def _calculate_rate_limit_utilization(self) -> float:
        """Calculate current rate limit utilization percentage"""
        if not self.batch_processor.rate_limit_history:
            return 0.0

        recent_limits = self.batch_processor.rate_limit_history[-5:]
        if not recent_limits:
            return 0.0

        # Calculate average remaining rate limit
        avg_remaining = sum(recent_limits) / len(recent_limits)

        # Assume Apollo has a default rate limit of 100 requests per minute
        # This is a conservative estimate - actual limits may vary
        default_limit = 100
        utilization = ((default_limit - avg_remaining) / default_limit) * 100

        return min(utilization, 100.0)  # Cap at 100%

    def log_performance_summary(self):
        """Log a comprehensive performance summary"""
        metrics = self.get_performance_metrics()

        logger.info("=== Apollo Client Performance Summary ===")
        logger.info(
            f"Batch Processing: {metrics['batch_processing']['current_batch_size']} current batch size"
        )
        logger.info(f"Cache Hit Rate: {metrics['caching']['hit_rate_percent']}%")
        logger.info(
            f"Average Response Time: {metrics['batch_processing']['avg_response_time']:.2f}s"
        )
        logger.info(
            f"Rate Limit Utilization: {metrics['overall_performance']['rate_limit_utilization']:.1f}%"
        )
        logger.info(f"Concurrent Requests: {metrics['concurrent_requests']}")
        logger.info("=========================================")

    def get_health_status(self) -> Dict[str, Any]:
        """Get overall health status of the Apollo client"""
        try:
            metrics = self.get_performance_metrics()

            # Define health thresholds
            response_time_threshold = 5.0  # seconds
            cache_hit_threshold = 20.0  # percentage
            rate_limit_threshold = 80.0  # percentage

            # Assess health based on metrics
            avg_response_time = metrics["batch_processing"]["avg_response_time"]
            cache_hit_rate = metrics["caching"]["hit_rate_percent"]
            rate_limit_util = metrics["overall_performance"]["rate_limit_utilization"]

            health_score = 100

            # Response time penalty
            if avg_response_time > response_time_threshold:
                health_score -= min(
                    30, (avg_response_time - response_time_threshold) * 10
                )

            # Cache hit rate penalty
            if cache_hit_rate < cache_hit_threshold:
                health_score -= min(20, (cache_hit_threshold - cache_hit_rate) * 2)

            # Rate limit penalty
            if rate_limit_util > rate_limit_threshold:
                health_score -= min(25, (rate_limit_util - rate_limit_threshold) * 2.5)

            health_score = max(0, health_score)

            # Determine status
            if health_score >= 80:
                status = "healthy"
            elif health_score >= 60:
                status = "warning"
            else:
                status = "critical"

            return {
                "status": status,
                "health_score": health_score,
                "metrics": metrics,
                "thresholds": {
                    "response_time_threshold": response_time_threshold,
                    "cache_hit_threshold": cache_hit_threshold,
                    "rate_limit_threshold": rate_limit_threshold,
                },
                "recommendations": self._generate_health_recommendations(
                    health_score, metrics
                ),
            }

        except Exception as e:
            logger.error(f"Error calculating health status: {e}")
            return {"status": "error", "health_score": 0, "error": str(e)}

    def _generate_health_recommendations(
        self, health_score: int, metrics: Dict[str, Any]
    ) -> List[str]:
        """Generate health improvement recommendations"""
        recommendations = []

        avg_response_time = metrics["batch_processing"]["avg_response_time"]
        cache_hit_rate = metrics["caching"]["hit_rate_percent"]
        rate_limit_util = metrics["overall_performance"]["rate_limit_utilization"]

        if avg_response_time > 3.0:
            recommendations.append(
                "Consider reducing batch size to improve response times"
            )

        if cache_hit_rate < 30:
            recommendations.append(
                "Cache hit rate is low - consider increasing TTL or cache size"
            )

        if rate_limit_util > 70:
            recommendations.append(
                "Rate limit utilization is high - consider implementing backoff strategies"
            )

        if health_score < 60:
            recommendations.append(
                "Overall health is poor - review configuration and network connectivity"
            )

        if not recommendations:
            recommendations.append(
                "Performance is optimal - no immediate actions required"
            )

        return recommendations

    async def get_credits_remaining(self) -> int:
        """
        Get remaining Apollo credits.

        Returns:
            Number of credits remaining
        """
        logger.info("Checking remaining Apollo credits")

        response = await self._make_request(method="GET", endpoint="/credits")

        credits_remaining = response.get("credits_remaining", 0)
        logger.info(f"Apollo credits remaining: {credits_remaining}")
        return credits_remaining

    async def get_usage_metrics(self) -> Dict[str, Any]:
        """
        Get Apollo API usage metrics.

        Returns:
            Usage metrics including total requests, enrichments, and credits used
        """
        logger.info("Retrieving Apollo usage metrics")

        response = await self._make_request(method="GET", endpoint="/usage")

        # Add client-side metrics
        metrics = {
            **response,
            "client_credits_used": self.credits_used,
            "client_total_requests": self.total_requests,
            "client_total_errors": self.total_errors,
        }

        logger.info(f"Apollo usage metrics retrieved: {metrics}")
        return metrics

    def _validate_filters_not_hardcoded(self, filters: Dict[str, Any]):
        """
        Validate that business filters are not hardcoded.

        Args:
            filters: Filters to validate

        Raises:
            ValueError: If hardcoded business parameters are detected
        """
        # Common hardcoded patterns to check for
        hardcoded_patterns = [
            ["technology", "software", "saas"],
            ["1-10", "11-50", "51-200"],
            ["CEO", "CTO", "VP Engineering"],
            ["healthcare", "medical", "pharmaceuticals"],
            ["series_a", "series_b", "series_c"],
        ]

        for filter_type, values in filters.items():
            if isinstance(values, list):
                for pattern in hardcoded_patterns:
                    if values == pattern:
                        raise ValueError(
                            f"Hardcoded business parameters detected in {filter_type}: {pattern}. "
                            "All business parameters must be loaded from configuration, never hardcoded."
                        )

    def get_cost_summary(self) -> Dict[str, Any]:
        """
        Get cost summary for Apollo API usage.

        Returns:
            Cost summary including credits used and estimated USD cost
        """
        # Apollo pricing: 1 credit = $0.015 USD
        usd_per_credit = 0.015
        total_usd = self.credits_used * usd_per_credit

        return {
            "credits_used": self.credits_used,
            "usd_per_credit": usd_per_credit,
            "total_usd": total_usd,
            "total_requests": self.total_requests,
            "total_errors": self.total_errors,
        }

    def reset_metrics(self):
        """Reset client metrics (useful for testing)"""
        self.credits_used = 0
        self.total_requests = 0
        self.total_errors = 0
        logger.info("Apollo client metrics reset")

    def get_cache_info(self) -> Dict[str, Any]:
        """Get detailed cache information"""
        return self.cache.get_cache_info()

    def clear_cache(self):
        """Clear all cached data"""
        self.cache.clear()
        logger.info("Cache cleared")

    def invalidate_cache_pattern(self, pattern: str):
        """Invalidate cache entries matching pattern"""
        self.cache.invalidate_pattern(pattern)

    async def search_companies_optimized(
        self, base_filters: Dict[str, Any], max_results: int = 1000
    ) -> List[Dict[str, Any]]:
        """Search companies using optimized query strategies"""
        # Generate optimized search variations
        search_variations = self.search_optimizer.get_optimal_queries(
            base_filters, "company"
        )

        all_results = []
        seen_companies = set()

        for variation in search_variations:
            if len(all_results) >= max_results:
                break

            try:
                # Execute search with timeout
                search_task = self._execute_company_search(
                    variation, max_results - len(all_results)
                )
                results = await asyncio.wait_for(
                    search_task, timeout=self.search_optimizer.config.query_timeout
                )

                # Deduplicate results
                for company in results:
                    company_id = company.get("id") or company.get("domain")
                    if company_id and company_id not in seen_companies:
                        all_results.append(company)
                        seen_companies.add(company_id)

                # Record performance for optimization
                query_hash = str(hash(frozenset(sorted(variation.items()))))
                self.search_optimizer.record_query_performance(
                    query_hash,
                    0.0,  # Will be updated by actual request timing
                    len(results),
                )

            except asyncio.TimeoutError:
                logger.warning(f"Search variation timed out: {variation}")
                continue
            except Exception as e:
                logger.error(f"Search variation failed: {variation}, error: {e}")
                continue

        return all_results[:max_results]

    async def search_persons_optimized(
        self, base_filters: Dict[str, Any], max_results: int = 1000
    ) -> List[Dict[str, Any]]:
        """Search persons using optimized query strategies"""
        # Generate optimized search variations
        search_variations = self.search_optimizer.get_optimal_queries(
            base_filters, "person"
        )

        all_results = []
        seen_persons = set()

        for variation in search_variations:
            if len(all_results) >= max_results:
                break

            try:
                # Execute search with timeout
                search_task = self._execute_person_search(
                    variation, max_results - len(all_results)
                )
                results = await asyncio.wait_for(
                    search_task, timeout=self.search_optimizer.config.query_timeout
                )

                # Deduplicate results
                for person in results:
                    person_id = person.get("id") or person.get("email")
                    if person_id and person_id not in seen_persons:
                        all_results.append(person)
                        seen_persons.add(person_id)

                # Record performance for optimization
                query_hash = str(hash(frozenset(sorted(variation.items()))))
                self.search_optimizer.record_query_performance(
                    query_hash,
                    0.0,  # Will be updated by actual request timing
                    len(results),
                )

            except asyncio.TimeoutError:
                logger.warning(f"Search variation timed out: {variation}")
                continue
            except Exception as e:
                logger.error(f"Search variation failed: {variation}, error: {e}")
                continue

        return all_results[:max_results]

    async def _execute_company_search(
        self, filters: Dict[str, Any], max_results: int
    ) -> List[Dict[str, Any]]:
        """Execute a single company search"""
        # Use the original search method for single searches
        try:
            response, _ = await self._make_request("companies/search", filters)
            return response.get("companies", [])[:max_results]
        except Exception as e:
            logger.error(f"Company search failed: {e}")
            return []

    async def _execute_person_search(
        self, filters: Dict[str, Any], max_results: int
    ) -> List[Dict[str, Any]]:
        """Execute a single person search"""
        # Use the original search method for single searches
        try:
            response, _ = await self._make_request("people/search", filters)
            return response.get("people", [])[:max_results]
        except Exception as e:
            logger.error(f"Person search failed: {e}")
            return []

    async def parallel_bulk_enrich(
        self, companies: List[str] = None, persons: List[str] = None
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Perform parallel bulk enrichment of companies and persons"""
        tasks = []

        if companies:
            companies_task = self.enrich_companies_batched(companies)
            tasks.append(("companies", companies_task))

        if persons:
            persons_task = self.enrich_persons_batched(persons)
            tasks.append(("persons", persons_task))

        if not tasks:
            return {"companies": [], "persons": []}

        # Execute all enrichment tasks in parallel
        results = await asyncio.gather(
            *[task for _, task in tasks], return_exceptions=True
        )

        # Process results
        final_results = {"companies": [], "persons": []}

        for i, (entity_type, result) in enumerate(
            zip([task[0] for task in tasks], results)
        ):
            if isinstance(result, Exception):
                logger.error(f"Bulk enrichment failed for {entity_type}: {result}")
                continue

            if entity_type == "companies":
                final_results["companies"] = result
            elif entity_type == "persons":
                final_results["persons"] = result

        return final_results

    def get_search_optimization_stats(self) -> Dict[str, Any]:
        """Get search optimization performance statistics"""
        return {
            "query_performance_history": {
                k: {
                    "avg_response_time": sum(v) / len(v) if v else 0,
                    "total_queries": len(v),
                    "recent_performance": v[-5:] if v else [],
                }
                for k, v in self.search_optimizer.query_performance_history.items()
            },
            "successful_patterns": self.search_optimizer.successful_patterns,
            "optimization_config": {
                "enable_query_analysis": self.search_optimizer.config.enable_query_analysis,
                "max_parallel_searches": self.search_optimizer.config.max_parallel_searches,
                "max_retry_variations": self.search_optimizer.config.max_retry_variations,
            },
        }

    # Note: Duplicate methods removed - enrich_company and bulk_enrich_companies are defined earlier in the file


# Factory function for creating Apollo client from environment
def create_apollo_client_from_env() -> ApolloClient:
    """
    Create Apollo client from environment variables.

    Returns:
        Configured Apollo client

    Raises:
        ValueError: If required environment variables are missing
    """
    api_key = os.getenv("APOLLO_API_KEY")
    if not api_key:
        raise ValueError("APOLLO_API_KEY environment variable is required")

    refresh_token = os.getenv("APOLLO_REFRESH_TOKEN")

    config = ApolloConfig(api_key=api_key, refresh_token=refresh_token)

    return ApolloClient(config)
