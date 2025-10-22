"""
Domain tool for Lead Sorcerer.

This tool generates qualified domains from ICP queries using cheap pre-crawl scoring.
It calls Google Programmable Search (GSE) and uses LLM scoring to filter domains.

Authoritative specifications: BRD §226-280
"""

import asyncio
import hashlib
import json
import logging
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx
from openai import AsyncOpenAI

from .common import (
    AsyncSemaphorePool,
    ErrorCode,
    PermitManager,
    append_audit,
    append_status,
    build_error,
    build_metrics,
    compute_next_revisit,
    generate_lead_id,
    load_costs_config,
    load_schema_checksum,
    normalize_domain,
    now_z,
    round4,
    recompute_total_cost,
    setup_logging,
    validate_provider_config,
    validate_template_placeholders,
    write_with_lock,
    append_jsonl_with_lock,
)

# ============================================================================
# Constants and Configuration
# ============================================================================

INTERNAL_VERSION = "1.0.0"
TOOL_NAME = "domain"

# Default values from BRD
DEFAULT_REVISIT_AFTER_DAYS = 90
DEFAULT_FAILURE_REVISIT_DAYS = 1
DEFAULT_DOMAIN_TTL_DAYS = 180
DEFAULT_DOMAIN_SERP_TTL_HOURS = 24
DEFAULT_MAX_PAGES = 5
DEFAULT_MODE = "thorough"

# ============================================================================
# Google Search Engine (GSE) Integration
# ============================================================================


class GSESearchClient:
    """Client for Google Programmable Search Engine (GSE)."""

    def __init__(self, api_key: str, cx: str, semaphore_pool: AsyncSemaphorePool):
        self.api_key = api_key
        self.cx = cx
        self.semaphore_pool = semaphore_pool
        self.base_url = "https://www.googleapis.com/customsearch/v1"

    async def search(self, query: str, page: int = 1) -> Dict[str, Any]:
        """
        Perform Google search with pagination.

        Args:
            query: Search query
            page: Page number (1-based)

        Returns:
            Search results dictionary
        """
        async with self.semaphore_pool:
            params = {
                "key": self.api_key,
                "cx": self.cx,
                "q": query,
                "start": (page - 1) * 10 + 1,  # GSE uses 1-based indexing
                "num": 10,
            }

            async with httpx.AsyncClient(timeout=(3.0, 10.0)) as client:
                response = await client.get(self.base_url, params=params)
                response.raise_for_status()
                return response.json()


# ============================================================================
# LLM Scoring
# ============================================================================


class LLMScorer:
    """LLM-based domain scoring using OpenRouter."""

    def __init__(self, api_key: str, semaphore_pool: AsyncSemaphorePool):
        self.client = AsyncOpenAI(
            api_key=api_key, base_url="https://openrouter.ai/api/v1"
        )
        self.semaphore_pool = semaphore_pool
        self.primary_model = "gpt-4o-mini"
        self.fallback_model = "gpt-3.5-turbo"

    def _build_scoring_prompt(
        self, title: str, snippet: str, query: str, icp_text: str
    ) -> str:
        """
        Build scoring prompt for LLM.

        Args:
            title: SERP result title
            snippet: SERP result snippet
            query: Search query used
            icp_text: ICP description

        Returns:
            Formatted prompt string
        """
        return f"""You are evaluating whether a company matches an Ideal Customer Profile (ICP).

ICP Description: {icp_text}

Search Query: {query}

Company Information:
Title: {title}
Description: {snippet}

Rate this company from 0.0 to 1.0 based on how well it matches the ICP.
Consider factors like:
- Industry relevance
- Company size and stage
- Geographic location
- Technology focus
- Business model alignment

Return only a JSON object with this exact format:
{{
    "score": <float between 0.0 and 1.0>,
    "reason": "<brief explanation of score>",
    "flags": ["<any disqualifying flags like geo_mismatch, size_mismatch, etc.>"]
}}

Example flags: geo_mismatch, size_mismatch, industry_mismatch, stage_mismatch, technology_mismatch"""

    async def score_domain(
        self, title: str, snippet: str, query: str, icp_text: str
    ) -> Tuple[float, str, List[str], str, str]:
        """
        Score a domain using LLM.

        Args:
            title: SERP result title
            snippet: SERP result snippet
            query: Search query used
            icp_text: ICP description

        Returns:
            Tuple of (score, reason, flags, model_used, prompt_fingerprint)
        """
        async with self.semaphore_pool:
            prompt = self._build_scoring_prompt(title, snippet, query, icp_text)
            prompt_fingerprint = hashlib.sha256(prompt.encode()).hexdigest()

            try:
                # Try primary model first
                response = await self.client.chat.completions.create(
                    model=self.primary_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=500,
                )
                model_used = self.primary_model
            except Exception:
                # Fallback to secondary model
                response = await self.client.chat.completions.create(
                    model=self.fallback_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=500,
                )
                model_used = self.fallback_model

            content = response.choices[0].message.content.strip()

            # Parse JSON response
            try:
                result = json.loads(content)
                score = float(result.get("score", 0.0))
                reason = result.get("reason", "No reason provided")
                flags = result.get("flags", [])

                # Validate score range
                score = max(0.0, min(1.0, score))

                return score, reason, flags, model_used, prompt_fingerprint
            except (json.JSONDecodeError, ValueError, KeyError) as e:
                # Fallback to default values if parsing fails
                logging.warning(f"Failed to parse LLM response: {e}")
                return (
                    0.0,
                    "Failed to parse response",
                    [],
                    model_used,
                    prompt_fingerprint,
                )


# ============================================================================
# Domain History Management
# ============================================================================


class DomainHistoryManager:
    """Manages domain history for cross-run deduplication and TTL enforcement."""

    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.history_file = self.data_dir / "domain_history.jsonl"
        self.history_cache: Dict[str, Dict[str, Any]] = {}
        self._load_history()

    def _load_history(self) -> None:
        """Load existing domain history from disk."""
        if not self.history_file.exists():
            return

        try:
            with open(self.history_file, "r") as f:
                for line in f:
                    if line.strip():
                        record = json.loads(line)
                        domain = record.get("domain")
                        if domain:
                            self.history_cache[domain] = record
        except Exception as e:
            logging.warning(f"Failed to load domain history: {e}")

    def _cleanup_expired_entries(self, ttl_days: int) -> None:
        """Remove expired domain history entries."""
        cutoff_time = datetime.utcnow() - timedelta(days=ttl_days)
        expired_domains = []

        for domain, record in self.history_cache.items():
            scored_at = record.get("provenance", {}).get("scored_at")
            if scored_at:
                try:
                    record_time = datetime.fromisoformat(
                        scored_at.replace("Z", "+00:00")
                    )
                    # Make both timezone-aware for comparison
                    cutoff_time_aware = cutoff_time.replace(tzinfo=record_time.tzinfo)
                    if record_time < cutoff_time_aware:
                        expired_domains.append(domain)
                except ValueError:
                    # Invalid timestamp, remove
                    expired_domains.append(domain)

        for domain in expired_domains:
            del self.history_cache[domain]

    def get_domain_record(self, domain: str) -> Optional[Dict[str, Any]]:
        """
        Get existing domain record if it exists and is not expired.

        Args:
            domain: Normalized domain

        Returns:
            Domain record or None if not found/expired
        """
        return self.history_cache.get(domain)

    def add_domain_record(self, record: Dict[str, Any]) -> None:
        """
        Add or update domain record in history.

        Args:
            record: Domain record to add/update
        """
        domain = record.get("domain")
        if domain:
            self.history_cache[domain] = record

    def save_history(self, ttl_days: int) -> None:
        """
        Save domain history to disk and cleanup expired entries.

        Args:
            ttl_days: TTL for domain history entries
        """
        self._cleanup_expired_entries(ttl_days)

        # Write updated history in JSONL format
        records = list(self.history_cache.values())
        if records:
            # Remove existing file if present to ensure clean JSONL format

            # Write to a temporary file first for atomic replacement
            temp_file = self.history_file.with_suffix(".tmp")
            append_jsonl_with_lock(str(temp_file), records)
            # Atomically replace the old file
            temp_file.replace(self.history_file)


# ============================================================================
# Search Cache Management
# ============================================================================


class SearchCache:
    """Manages SERP result caching with TTL."""

    def __init__(self):
        self.cache: Dict[str, Dict[str, Any]] = {}

    def _get_cache_key(self, query: str, page: int) -> str:
        """Generate cache key for query and page."""
        return f"{query}|{page}|google"

    def get_cached_results(
        self, query: str, page: int, ttl_hours: int
    ) -> Optional[Dict[str, Any]]:
        """
        Get cached search results if not expired.

        Args:
            query: Search query
            page: Page number
            ttl_hours: TTL in hours

        Returns:
            Cached results or None if expired/not found
        """
        cache_key = self._get_cache_key(query, page)
        cached = self.cache.get(cache_key)

        if not cached:
            return None

        # Check TTL
        fetched_at = cached.get("fetched_at")
        if fetched_at:
            try:
                fetch_time = datetime.fromisoformat(fetched_at.replace("Z", "+00:00"))
                if datetime.utcnow() - fetch_time.replace(tzinfo=None) < timedelta(
                    hours=ttl_hours
                ):
                    return cached
            except ValueError:
                pass

        # Expired, remove from cache
        del self.cache[cache_key]
        return None

    def cache_results(self, query: str, page: int, results: Dict[str, Any]) -> None:
        """
        Cache search results.

        Args:
            query: Search query
            page: Page number
            results: Search results
        """
        cache_key = self._get_cache_key(query, page)
        self.cache[cache_key] = {"results": results, "fetched_at": now_z()}


# ============================================================================
# Main Domain Tool
# ============================================================================


class DomainTool:
    """Main Domain tool implementation."""

    def __init__(self, icp_config: Dict[str, Any], data_dir: str):
        self.icp_config = icp_config
        self.data_dir = Path(data_dir)
        self.evidence_dir = self.data_dir / "evidence" / "domain"
        self.evidence_dir.mkdir(parents=True, exist_ok=True)

        # Load costs configuration
        self.costs_config = load_costs_config()
        validate_provider_config(["gse", "openrouter"], self.costs_config)

        # Initialize components
        self.permit_manager = PermitManager(
            max_permits=icp_config.get("concurrency", {}).get(
                "max_concurrent_requests", 3
            )
        )
        self.semaphore_pool = AsyncSemaphorePool(self.permit_manager)

        self.gse_client = GSESearchClient(
            api_key=os.environ["GSE_API_KEY"],
            cx=os.environ["GSE_CX"],
            semaphore_pool=self.semaphore_pool,
        )

        self.llm_scorer = LLMScorer(
            api_key=os.environ["OPENROUTER_KEY"], semaphore_pool=self.semaphore_pool
        )

        self.domain_history = DomainHistoryManager(data_dir)
        self.search_cache = SearchCache()

        # Load schema checksum
        self.schema_checksum = load_schema_checksum()

        # Setup logging (child tools inherit shared run log via env)
        self.logger = setup_logging(TOOL_NAME, data_dir=self.data_dir)

    def _validate_icp_config(self) -> List[Dict[str, Any]]:
        """
        Validate ICP configuration.

        Returns:
            List of validation errors
        """
        errors = []

        # Check required fields
        required_fields = ["name", "icp_text", "queries"]
        for field in required_fields:
            if field not in self.icp_config:
                errors.append(
                    build_error(
                        ErrorCode.SCHEMA_VALIDATION,
                        context={"missing_field": field, "stage": TOOL_NAME},
                    )
                )

        # Validate template placeholders
        if "queries" in self.icp_config:
            for query in self.icp_config["queries"]:
                missing_placeholders = validate_template_placeholders(query, {})
                if missing_placeholders:
                    errors.append(
                        build_error(
                            ErrorCode.SCHEMA_VALIDATION,
                            context={
                                "missing_placeholders": missing_placeholders,
                                "query": query,
                                "stage": TOOL_NAME,
                            },
                        )
                    )

        return errors

    def _get_search_parameters(self) -> Tuple[int, int, str]:
        """
        Get search parameters from ICP config.

        Returns:
            Tuple of (max_pages, ttl_hours, mode)
        """
        search_config = self.icp_config.get("search", {})
        max_pages = search_config.get("max_pages", DEFAULT_MAX_PAGES)

        refresh_config = self.icp_config.get("refresh_policy", {})
        ttl_hours = refresh_config.get(
            "domain_serp_ttl_hours", DEFAULT_DOMAIN_SERP_TTL_HOURS
        )

        mode = self.icp_config.get("mode", DEFAULT_MODE)
        if mode == "fast":
            max_pages = min(1, max_pages)

        return max_pages, ttl_hours, mode

    def _get_refresh_parameters(self) -> Tuple[int, int, int]:
        """
        Get refresh policy parameters.

        Returns:
            Tuple of (revisit_after_days, failure_revisit_days, domain_ttl_days)
        """
        refresh_config = self.icp_config.get("refresh_policy", {})
        revisit_after_days = refresh_config.get(
            "revisit_after_days", DEFAULT_REVISIT_AFTER_DAYS
        )
        failure_revisit_days = refresh_config.get(
            "failure_revisit_days", DEFAULT_FAILURE_REVISIT_DAYS
        )
        domain_ttl_days = refresh_config.get("domain_ttl_days", DEFAULT_DOMAIN_TTL_DAYS)

        return revisit_after_days, failure_revisit_days, domain_ttl_days

    def _create_lead_record(self, domain: str, query: str) -> Dict[str, Any]:
        """
        Create base lead record structure.

        Args:
            domain: Normalized domain
            query: Search query

        Returns:
            Base lead record
        """
        lead_id = generate_lead_id(domain)

        return {
            "lead_id": lead_id,
            "domain": domain,
            "company": {
                "name": None,
                "description": None,
                "industry": None,
                "naics_keywords": [],
                "hq_location": None,
                "locations": [],
                "size_hint": None,
                "socials": {
                    "linkedin": None,
                    "github": None,
                    "twitter": None,
                    "telegram": None,
                    "instagram": None,
                },
                "careers_url": None,
                "blog_url": None,
                "tech_stack": [],
                "evidence_urls": [],
            },
            "contacts": [],
            "icp": {
                "pre_score": None,
                "pre_reason": None,
                "pre_pass": None,
                "pre_flags": [],
                "scoring_meta": {
                    "method": None,
                    "model": None,
                    "prompt_fingerprint": None,
                    "temperature": None,
                },
                "crawl_score": None,
                "crawl_reason": None,
                "threshold": self.icp_config.get("threshold", 0.7),
                "filtering_strict": self.icp_config.get("filtering_strict", True),
            },
            "provenance": {
                "queries": [query],
                "discovery_evidence": [],
                "scored_at": None,
                "crawled_at": None,
                "enriched_at": None,
                "next_revisit_at": None,
                "tool_versions": {
                    "domain": {
                        "version": INTERNAL_VERSION,
                        "schema_version": self.schema_checksum,
                    },
                    "crawl": {"version": None, "schema_version": None},
                    "enrich": {"version": None, "schema_version": None},
                },
                "cache": {"domain_cache_hit": None},
                "evidence_paths": {"domain": None, "crawl": None, "enrich": None},
            },
            "best_contact_id": None,
            "status": "scored",
            "status_history": [],
            "audit": [],
            "cost": {
                "domain_usd": 0.0,
                "crawl_usd": 0.0,
                "enrich_usd": 0.0,
                "total_usd": 0.0,
            },
        }

    def _normalize_serp_result(
        self, item: Dict[str, Any], query: str, page_index: int
    ) -> Dict[str, Any]:
        """
        Normalize SERP result item.

        Args:
            item: Raw SERP result
            query: Search query
            page_index: Page index

        Returns:
            Normalized result
        """
        # Extract domain from URL
        url = item.get("link", "")
        domain = normalize_domain(url)

        return {
            "url": url,
            "title": item.get("title", ""),
            "snippet": item.get("snippet", ""),
            "query": query,
            "page_index": page_index,
            "rank_on_page": item.get("rank", 0),
            "fetched_at": now_z(),
            "domain": domain,
        }

    async def _search_query(
        self, query: str, max_pages: int, ttl_hours: int
    ) -> Tuple[List[Dict[str, Any]], int, int]:
        """
        Search for domains using a single query.

        Args:
            query: Search query
            max_pages: Maximum pages to search
            ttl_hours: Cache TTL in hours

        Returns:
            Tuple of (results, cache_hits, cache_misses)
        """
        self.logger.info(f"🔍 Searching query: {query} (max_pages={max_pages})")

        normalized_results: List[Dict[str, Any]] = []
        cache_hits = 0
        cache_misses = 0

        for page in range(1, max_pages + 1):
            # Try cache first
            cached = self.search_cache.get_cached_results(query, page, ttl_hours)
            if cached:
                cache_hits += 1
                for item in cached["results"].get("items", []):
                    normalized_results.append(
                        self._normalize_serp_result(item, query, page)
                    )
                continue

            # Fetch fresh
            try:
                results = await self.gse_client.search(query, page)
                cache_misses += 1
                # Cache
                self.search_cache.cache_results(query, page, results)
                # Normalize
                for item in results.get("items", []):
                    normalized_results.append(
                        self._normalize_serp_result(item, query, page)
                    )
            except Exception as e:
                self.logger.error(
                    f"❌ Search failed for query '{query}' page {page}: {e}"
                )
                continue

        return normalized_results, cache_hits, cache_misses

    async def _search_and_score_domains(
        self, queries: List[str], max_pages: int, ttl_hours: int
    ) -> Tuple[List[Dict[str, Any]], int, int]:
        """
        Search and score domains for given queries.

        Args:
            queries: List of search queries
            max_pages: Maximum pages to fetch
            ttl_hours: Cache TTL in hours

        Returns:
            Tuple of (scored_domains, cache_hits, cache_misses)
        """
        self.logger.info(
            f"🔍 _search_and_score_domains: {len(queries)} queries, {max_pages} pages, {ttl_hours}h TTL"
        )
        all_scored_domains: List[Dict[str, Any]] = []
        total_cache_hits = 0
        total_cache_misses = 0

        for query_idx, query in enumerate(queries, 1):
            self.logger.info(
                f"🔍 Processing query {query_idx}/{len(queries)}: '{query}'"
            )
            # Search for domains using the current query
            (
                query_results,
                query_cache_hits,
                query_cache_misses,
            ) = await self._search_query(query, max_pages, ttl_hours)
            total_cache_hits += query_cache_hits
            total_cache_misses += query_cache_misses
            # Score domains for this query
            self.logger.info(
                f"🤖 Starting LLM scoring for {len(query_results)} domains..."
            )
            scored_domains = await self._score_domains(query_results)
            self.logger.info(
                f"✅ LLM scoring completed: {len(scored_domains)} scored domains"
            )
            all_scored_domains.extend(scored_domains)

        self.logger.info(
            f"📊 Search completed: {len(all_scored_domains)} total results, {total_cache_hits} cache hits, {total_cache_misses} cache misses"
        )
        return all_scored_domains, total_cache_hits, total_cache_misses

    async def _score_domains(
        self, serp_results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Score domains using LLM.

        Args:
            serp_results: List of normalized SERP results

        Returns:
            List of scored domain records
        """
        self.logger.info(
            f"🤖 _score_domains: Processing {len(serp_results)} SERP results"
        )
        scored_domains = []
        domain_map = {}  # Track unique domains

        for idx, result in enumerate(serp_results, 1):
            domain = result["domain"]
            self.logger.info(
                f"🤖 Processing domain {idx}/{len(serp_results)}: {domain}"
            )

            # Skip if we already processed this domain
            if domain in domain_map:
                self.logger.info(
                    f"🔄 Domain {domain} already processed, adding to evidence"
                )
                # Add to existing record's discovery evidence
                existing_record = domain_map[domain]
                if len(existing_record["provenance"]["discovery_evidence"]) < 3:
                    existing_record["provenance"]["discovery_evidence"].append(result)
                continue

            # Create new lead record
            self.logger.info(f"📝 Creating new lead record for domain: {domain}")
            record = self._create_lead_record(domain, result["query"])

            # Score the domain
            self.logger.info(f"🤖 Scoring domain {domain} with LLM...")
            try:
                (
                    score,
                    reason,
                    flags,
                    model_used,
                    prompt_fingerprint,
                ) = await self.llm_scorer.score_domain(
                    result["title"],
                    result["snippet"],
                    result["query"],
                    self.icp_config["icp_text"],
                )

                # Update record with scoring results
                record["icp"]["pre_score"] = round4(score)
                record["icp"]["pre_reason"] = reason
                record["icp"]["pre_flags"] = flags
                record["icp"]["scoring_meta"] = {
                    "method": "llm",
                    "model": model_used,
                    "prompt_fingerprint": prompt_fingerprint,
                    "temperature": 0.0,
                }

                # Set pre_pass based on threshold
                threshold = record["icp"]["threshold"]
                record["icp"]["pre_pass"] = score >= threshold

                # Add discovery evidence
                record["provenance"]["discovery_evidence"].append(result)

                # Set scored_at and next_revisit_at
                record["provenance"]["scored_at"] = now_z()
                revisit_after_days, failure_revisit_days, _ = (
                    self._get_refresh_parameters()
                )
                record["provenance"]["next_revisit_at"] = compute_next_revisit(
                    None, revisit_after_days, failure_revisit_days, False
                )

                # Add status history entry
                append_status(
                    record["status_history"], "scored", f"Scored with {model_used}"
                )

                # Add audit entry
                append_audit(
                    record, "Domain", f"Scored domain {domain} with score {score}"
                )

                # Calculate costs
                gse_cost = self.costs_config["gse"]
                openrouter_cost = self.costs_config["openrouter"]

                # Estimate token usage (rough approximation)
                prompt_tokens = (
                    len(
                        result["title"]
                        + result["snippet"]
                        + result["query"]
                        + self.icp_config["icp_text"]
                    )
                    // 4
                )
                response_tokens = 100  # Estimated response length
                total_tokens = prompt_tokens + response_tokens
                token_cost = (total_tokens / 1000) * openrouter_cost

                record["cost"]["domain_usd"] = round4(gse_cost + token_cost)
                recompute_total_cost(record)

            except Exception as e:
                self.logger.error(f"LLM scoring failed for domain {domain}: {e}")
                # Set default values on scoring failure
                record["icp"]["pre_score"] = 0.0
                record["icp"]["pre_reason"] = f"Scoring failed: {str(e)}"
                record["icp"]["pre_pass"] = False
                record["icp"]["pre_flags"] = ["scoring_error"]

                # Set failure revisit time
                _, failure_revisit_days, _ = self._get_refresh_parameters()
                record["provenance"]["next_revisit_at"] = compute_next_revisit(
                    None, 0, failure_revisit_days, True
                )

                # Add status history entry
                append_status(record["status_history"], "scored", "Scoring failed")

                # Add audit entry
                append_audit(
                    record, "Domain", f"Scoring failed for domain {domain}: {e}"
                )

                # Set minimal cost
                record["cost"]["domain_usd"] = round4(self.costs_config["gse"])
                recompute_total_cost(record)

            # Store in domain map
            domain_map[domain] = record
            scored_domains.append(record)

        return scored_domains

    def _save_evidence(self, record: Dict[str, Any]) -> None:
        """
        Save evidence to disk and update record pointer.

        Args:
            record: Lead record
        """
        domain = record["domain"]
        evidence_file = self.evidence_dir / f"{domain}.json"

        evidence_data = {
            "domain": domain,
            "queries": record["provenance"]["queries"],
            "fetched_at": now_z(),
            "serp_results": record["provenance"]["discovery_evidence"],
            "scoring_prompt_fingerprint": record["icp"]["scoring_meta"][
                "prompt_fingerprint"
            ],
        }

        write_with_lock(str(evidence_file), evidence_data)
        record["provenance"]["evidence_paths"]["domain"] = str(evidence_file)

    async def run(self, icp_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Main entry point for Domain tool.

        Args:
            icp_config: ICP configuration (overrides instance config if provided)

        Returns:
            Tool output envelope
        """
        start_time = time.time()
        self.logger.info(f"🚀 Domain tool run() started at {start_time}")

        try:
            # Use provided config or instance config
            config = icp_config or self.icp_config
            self.logger.info(f"📋 Using config: {config.get('name', 'Unknown')}")

            # Validate configuration
            self.logger.info("🔍 Validating ICP configuration...")
            validation_errors = self._validate_icp_config()
            if validation_errors:
                self.logger.error(
                    f"❌ Configuration validation failed: {validation_errors}"
                )
                return {
                    "data": {"lead_records": []},
                    "errors": validation_errors,
                    "metrics": build_metrics(0, 0, 0),
                }
            self.logger.info("✅ Configuration validation passed")

            # Get search and refresh parameters
            self.logger.info("⚙️ Getting search and refresh parameters...")
            max_pages, ttl_hours, mode = self._get_search_parameters()
            revisit_after_days, failure_revisit_days, domain_ttl_days = (
                self._get_refresh_parameters()
            )
            self.logger.info(
                f"📊 Search params: max_pages={max_pages}, ttl_hours={ttl_hours}, mode={mode}"
            )

            # Get queries and validate placeholders
            queries = config.get("queries", [])
            self.logger.info(f"🔍 Found {len(queries)} queries to process")
            if not queries:
                self.logger.error("❌ No queries provided")
                return {
                    "data": {"lead_records": []},
                    "errors": [
                        build_error(
                            ErrorCode.SCHEMA_VALIDATION,
                            context={
                                "error": "No queries provided",
                                "stage": TOOL_NAME,
                            },
                        )
                    ],
                    "metrics": build_metrics(0, 0, 0),
                }

            # Search and score domains
            self.logger.info("🔍 Starting search and score domains...")
            (
                scored_domains,
                cache_hits,
                cache_misses,
            ) = await self._search_and_score_domains(queries, max_pages, ttl_hours)

            # Save evidence for each domain
            for record in scored_domains:
                self._save_evidence(record)

                # Update domain history
                self.domain_history.add_domain_record(record)

            # Save domain history
            self.domain_history.save_history(domain_ttl_days)

            # Calculate metrics
            count_in = len(queries)
            count_out = len(scored_domains)
            duration_ms = int((time.time() - start_time) * 1000)

            cache_hit_rate = None
            if cache_hits + cache_misses > 0:
                cache_hit_rate = cache_hits / (cache_hits + cache_misses)

            pass_rate = None
            if count_out > 0:
                pass_count = sum(1 for r in scored_domains if r["icp"]["pre_pass"])
                pass_rate = pass_count / count_out

            total_cost = sum(r["cost"]["domain_usd"] for r in scored_domains)

            metrics = build_metrics(
                count_in=count_in,
                count_out=count_out,
                duration_ms=duration_ms,
                cache_hit_rate=cache_hit_rate,
                pass_rate=pass_rate,
                cost_usd={
                    "domain": total_cost,
                    "crawl": 0.0,
                    "enrich": 0.0,
                    "total": total_cost,
                },
            )

            return {
                "data": {"lead_records": scored_domains},
                "errors": [],
                "metrics": metrics,
            }

        except Exception as e:
            self.logger.error(f"Domain tool failed: {e}")
            return {
                "data": {"lead_records": []},
                "errors": [build_error(ErrorCode.UNKNOWN, exc=e, tool=TOOL_NAME)],
                "metrics": build_metrics(0, 0, int((time.time() - start_time) * 1000)),
            }

    def truncate_evidence_arrays(self, record: Dict[str, Any]) -> None:
        """
        Enforce evidence array truncation limits.

        Args:
            record: Lead record to truncate
        """
        # Company evidence_urls ≤ 10
        if "company" in record and "evidence_urls" in record["company"]:
            if len(record["company"]["evidence_urls"]) > 10:
                record["company"]["evidence_urls"] = record["company"]["evidence_urls"][
                    :10
                ]

        # Contacts evidence_urls ≤ 5
        if "contacts" in record:
            for contact in record["contacts"]:
                if "evidence_urls" in contact and len(contact["evidence_urls"]) > 5:
                    contact["evidence_urls"] = contact["evidence_urls"][:5]


# ============================================================================
# CLI Interface
# ============================================================================


def main():
    """CLI entry point for Domain tool."""
    import sys

    try:
        # Read JSON from stdin
        input_data = json.load(sys.stdin)

        # Extract icp_config
        icp_config = input_data.get("icp_config", {})
        data_dir = input_data.get("config", {}).get("data_dir", "./data")

        # Create tool instance
        tool = DomainTool(icp_config, data_dir)

        # Run tool
        result = asyncio.run(tool.run(icp_config))

        # Write JSON to stdout
        json.dump(result, sys.stdout, indent=2)

    except json.JSONDecodeError as e:
        print(
            json.dumps(
                {
                    "data": {"lead_records": []},
                    "errors": [
                        build_error(ErrorCode.SCHEMA_VALIDATION, exc=e, tool=TOOL_NAME)
                    ],
                    "metrics": build_metrics(0, 0, 0),
                }
            ),
            file=sys.stderr,
        )
        sys.exit(1)
    except Exception as e:
        print(
            json.dumps(
                {
                    "data": {"lead_records": []},
                    "errors": [build_error(ErrorCode.UNKNOWN, exc=e, tool=TOOL_NAME)],
                    "metrics": build_metrics(0, 0, 0),
                }
            ),
            file=sys.stderr,
        )
        sys.exit(1)
