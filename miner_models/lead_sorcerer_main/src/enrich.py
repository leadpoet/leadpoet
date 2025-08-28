"""
Enrich tool for Lead Sorcerer.

This tool enriches company and contact data using external APIs like Coresignal,
Snov.io, and Mailgun. It handles email validation, contact discovery, and
company data enrichment.

Authoritative specifications: BRD §333-336, §410-439
"""

import asyncio
import json
import logging
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
from dotenv import load_dotenv

import httpx
from httpx import HTTPStatusError, TimeoutException
from openai import AsyncOpenAI
# import portalocker  # Removed due to Windows compatibility issues

from src.common import (
    ErrorCode,
    append_audit,
    append_status,
    build_error,
    build_metrics,
    get_email_status_rank,
    get_role_priority,
    get_seniority_rank,
    load_costs_config,
    load_schema_checksum,
    normalize_domain,
    now_z,
    parse_phone_number,
    recompute_total_cost,
    resolve_data_dir,
    setup_logging,
    truncate_evidence_arrays,
    validate_envelope,
    validate_testing_flags,
)

from src.enrich_provider_router import create_enrichment_router
from src.enrich_providers.base import EnrichmentRequest

# Load environment variables
load_dotenv()

# Debug: Show environment variable status without exposing sensitive data
logging.info("🔑 Environment variables will be checked during router initialization")
# ============================================================================
# Constants and Configuration
# ============================================================================

INTERNAL_VERSION = "1.0.0"
TOOL_NAME = "enrich"

# Email status ranking for best contact selection
EMAIL_STATUS_RANKS = {
    "valid": 0,
    "risky": 1,
    "catch_all": 2,
    "unknown": 3,
    "invalid": 4,
}

# Default timeout values
CONNECT_TIMEOUT = 3.0
READ_TIMEOUT = 10.0
MAX_WALL_CLOCK = 45.0

# LLM Configuration for contact selection
LLM_MODEL = "gpt-4o-mini"
LLM_FALLBACK_MODEL = "gpt-3.5-turbo"
LLM_TEMPERATURE = 0.0
LLM_MAX_TOKENS = 500
LLM_TIMEOUT = 30.0
LLM_BASE_URL = "https://openrouter.ai/api/v1"

# LLM Configuration per ICP type (can be overridden in ICP config)
LLM_ICP_CONFIGS = {
    "default": {
        "model": LLM_MODEL,
        "fallback_model": LLM_FALLBACK_MODEL,
        "temperature": LLM_TEMPERATURE,
        "max_tokens": LLM_MAX_TOKENS,
        "timeout": LLM_TIMEOUT,
        "prompt_template": "base_contact_selection.txt",
    },
    "investment": {
        "model": "gpt-4o-mini",
        "fallback_model": "gpt-3.5-turbo",
        "temperature": 0.0,
        "max_tokens": 600,
        "timeout": 30.0,
        "prompt_template": "investment_contact_selection.txt",
    },
    "healthcare": {
        "model": "gpt-4o-mini",
        "fallback_model": "gpt-3.5-turbo",
        "temperature": 0.0,
        "max_tokens": 500,
        "timeout": 30.0,
        "prompt_template": "healthcare_contact_selection.txt",
    },
    "dental": {
        "model": "gpt-4o-mini",
        "fallback_model": "gpt-3.5-turbo",
        "temperature": 0.0,
        "max_tokens": 500,
        "timeout": 30.0,
        "prompt_template": "healthcare_contact_selection.txt",
    },
    "technology": {
        "model": "gpt-4o-mini",
        "fallback_model": "gpt-3.5-turbo",
        "temperature": 0.1,
        "max_tokens": 700,
        "timeout": 30.0,
        "prompt_template": "technology_contact_selection.txt",
    },
    "enterprise": {
        "model": "gpt-4o-mini",
        "fallback_model": "gpt-3.5-turbo",
        "temperature": 0.0,
        "max_tokens": 800,
        "timeout": 45.0,
        "prompt_template": "enterprise_contact_selection.txt",
    },
}

# Retry configuration
LLM_MAX_RETRIES = 3
LLM_RETRY_DELAY = 1.0  # seconds

# Cache configuration
LLM_CACHE_TTL = 3600  # 1 hour in seconds
LLM_CACHE_MAX_SIZE = 1000  # Maximum number of cached responses
LLM_CACHE_MAX_CONTACT_LIST_SIZE = 20  # Maximum contact list size to cache

# Fallback monitoring configuration
FALLBACK_METRICS = {
    "llm_failures": 0,
    "llm_none_responses": 0,
    "invalid_contact_ids": 0,
    "total_fallbacks": 0,
}

# LLM performance monitoring
LLM_PERFORMANCE_METRICS = {
    "total_calls": 0,
    "successful_calls": 0,
    "failed_calls": 0,
    "timeout_errors": 0,
    "rate_limit_errors": 0,
    "server_errors": 0,
    "invalid_responses": 0,
    "fallback_to_rule_based": 0,
    "cache_hits": 0,
    "cache_misses": 0,
    "total_response_time_ms": 0,
    "average_response_time_ms": 0,
}


# ============================================================================
# Provider Clients
# ============================================================================


class LLMContactSelector:
    """LLM-based contact selection using OpenRouter."""

    def __init__(
        self,
        api_key: str,
        icp_type: str = "default",
        icp_config: Optional[Dict[str, Any]] = None,
    ):
        self.client = AsyncOpenAI(api_key=api_key, base_url=LLM_BASE_URL)

        # Get configuration for this ICP type with potential overrides
        if icp_config:
            config = get_icp_llm_config(icp_type, icp_config)
        else:
            config = LLM_ICP_CONFIGS.get(icp_type, LLM_ICP_CONFIGS["default"])

        self.primary_model = config["model"]
        self.fallback_model = config["fallback_model"]
        self.temperature = config["temperature"]
        self.max_tokens = config["max_tokens"]
        self.timeout = config["timeout"]
        self.icp_type = icp_type
        self.prompt_template = config.get(
            "prompt_template", "base_contact_selection.txt"
        )

        # Initialize cache
        self._cache = {}
        self._cache_timestamps = {}

        # Initialize cost tracking
        self.total_cost = 0.0
        self.cost_details = []

    async def select_contact(
        self,
        contacts: List[Dict[str, Any]],
        icp_config: Dict[str, Any],
        role_priority_config: Optional[Dict[str, int]] = None,
    ) -> Optional[str]:
        """
        Select best contact using LLM.

        Args:
            contacts: List of contacts to choose from
            icp_config: ICP configuration with business context

        Returns:
            Contact ID of best contact or None if LLM fails
        """
        # Check cache first
        cache_key = self._generate_cache_key(contacts, icp_config)
        cached_result = self._get_cached_result(cache_key)
        if cached_result is not None:
            logging.info("💾 Using cached LLM result for contact selection")
            track_llm_performance_metric("cache_hits")
            return cached_result

        track_llm_performance_metric("cache_misses")

        prompt = self._build_contact_selection_prompt(
            contacts, icp_config, role_priority_config
        )

        # Try with retries
        for attempt in range(LLM_MAX_RETRIES):
            try:
                start_time = time.time()
                track_llm_performance_metric("total_calls")

                # Try primary model first
                try:
                    response = await self.client.chat.completions.create(
                        model=self.primary_model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                        timeout=self.timeout,
                    )
                    model_used = self.primary_model
                except Exception:
                    # Fallback to secondary model
                    response = await self.client.chat.completions.create(
                        model=self.fallback_model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                        timeout=self.timeout,
                    )
                    model_used = self.fallback_model

                content = response.choices[0].message.content.strip()

                # Track response time
                response_time_ms = int((time.time() - start_time) * 1000)
                track_llm_performance_metric("total_response_time_ms", response_time_ms)

                # Track LLM costs
                cost_info = self._track_llm_costs(response, model_used)
                self.total_cost += cost_info["cost_usd"]
                self.cost_details.append(cost_info)

                # Parse and validate response
                contact_id = extract_contact_id(content, contacts)
                if contact_id:
                    # Track successful call
                    track_llm_performance_metric("successful_calls")

                    # Log successful LLM call
                    self._log_llm_call(prompt, content, model_used, True)
                    logging.info(f"✅ LLM ({model_used}) selected contact {contact_id}")

                    # Cache the successful result
                    self._cache_result(cache_key, contact_id)

                    return contact_id
                else:
                    # Track invalid response
                    track_llm_performance_metric("invalid_responses")

                    # Log failed LLM call
                    self._log_llm_call(prompt, content, model_used, False)
                    logging.warning(
                        f"⚠️ LLM ({model_used}) returned invalid response, falling back to rule-based"
                    )
                    return None

            except TimeoutException:
                track_llm_performance_metric("timeout_errors")
                logging.warning(
                    f"⏰ LLM request timed out (attempt {attempt + 1}/{LLM_MAX_RETRIES})"
                )
                if attempt < LLM_MAX_RETRIES - 1:
                    await asyncio.sleep(
                        LLM_RETRY_DELAY * (2**attempt)
                    )  # Exponential backoff
                    continue
                else:
                    logging.error("⏰ LLM request timed out after all retries")
                    return None

            except HTTPStatusError as e:
                if e.response.status_code == 429:
                    track_llm_performance_metric("rate_limit_errors")
                    logging.warning(
                        f"🚫 LLM rate limit exceeded (attempt {attempt + 1}/{LLM_MAX_RETRIES})"
                    )
                    if attempt < LLM_MAX_RETRIES - 1:
                        await asyncio.sleep(
                            LLM_RETRY_DELAY * (2**attempt)
                        )  # Exponential backoff
                        continue
                    else:
                        logging.error("🚫 LLM rate limit exceeded after all retries")
                        return None
                elif e.response.status_code >= 500:
                    track_llm_performance_metric("server_errors")
                    logging.warning(
                        f"🔧 LLM server error: {e.response.status_code} (attempt {attempt + 1}/{LLM_MAX_RETRIES})"
                    )
                    if attempt < LLM_MAX_RETRIES - 1:
                        await asyncio.sleep(
                            LLM_RETRY_DELAY * (2**attempt)
                        )  # Exponential backoff
                        continue
                    else:
                        logging.error(
                            f"🔧 LLM server error after all retries: {e.response.status_code}"
                        )
                        return None
                else:
                    track_llm_performance_metric("failed_calls")
                    logging.error(f"❌ LLM HTTP error: {e.response.status_code}")
                    return None

            except Exception as e:
                track_llm_performance_metric("failed_calls")
                logging.warning(
                    f"❌ LLM contact selection failed (attempt {attempt + 1}/{LLM_MAX_RETRIES}): {str(e)}"
                )
                if attempt < LLM_MAX_RETRIES - 1:
                    await asyncio.sleep(
                        LLM_RETRY_DELAY * (2**attempt)
                    )  # Exponential backoff
                    continue
                else:
                    logging.error(
                        f"❌ LLM contact selection failed after all retries: {str(e)}"
                    )
                    return None

        return None

    def _track_llm_costs(self, response, model_used: str) -> Dict[str, Any]:
        """
        Track LLM costs for monitoring and billing.

        Args:
            response: OpenAI API response
            model_used: Model that was used

        Returns:
            Dictionary with cost and usage information
        """
        try:
            usage = response.usage
            prompt_tokens = usage.prompt_tokens
            completion_tokens = usage.completion_tokens
            total_tokens = usage.total_tokens

            # Calculate cost based on OpenRouter pricing (configured in costs.yaml)
            # Standard rate: $0.002 per 1k tokens
            cost_per_1k_tokens = (
                0.002  # This should match costs.yaml openrouter.usd_per_unit
            )
            cost_usd = (total_tokens / 1000.0) * cost_per_1k_tokens

            # Build cost tracking information
            cost_info = {
                "model": model_used,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "cost_usd": cost_usd,
            }

            # Log cost information
            logging.info(
                f"💰 LLM costs - Model: {model_used}, "
                f"Prompt: {prompt_tokens}, Completion: {completion_tokens}, "
                f"Total: {total_tokens} tokens, Cost: ${cost_usd:.4f}"
            )

            return cost_info

        except Exception as e:
            logging.warning(f"⚠️ Failed to track LLM costs: {str(e)}")
            return {
                "model": model_used,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "cost_usd": 0.0,
            }

    def _log_llm_call(self, prompt: str, response: str, model_used: str, success: bool):
        """
        Log LLM call details for monitoring and debugging.

        Args:
            prompt: The prompt sent to LLM
            response: The response from LLM
            model_used: Model that was used
            success: Whether the call was successful
        """
        try:
            # Redact PII from prompt and response for logging
            redacted_prompt = self._redact_pii(prompt)
            redacted_response = self._redact_pii(response)

            # Analyze response quality for monitoring
            quality_metrics = analyze_llm_response_quality(response)

            log_data = {
                "model": model_used,
                "success": success,
                "prompt_length": len(prompt),
                "response_length": len(response),
                "quality_score": quality_metrics["format_compliance"],
                "quality_issues": quality_metrics["issues"],
                "timestamp": time.time(),
            }

            if success:
                logging.info(f"📝 LLM call successful - {log_data}")
                logging.debug(f"🔍 LLM prompt: {redacted_prompt}")
                logging.debug(f"🔍 LLM response: {redacted_response}")

                # Log quality metrics for successful calls
                if quality_metrics["format_compliance"] < 1.0:
                    logging.warning(
                        f"⚠️ LLM response quality issues: {quality_metrics['issues']}"
                    )
                else:
                    logging.info(
                        f"✅ LLM response quality: {quality_metrics['format_compliance']:.2f}"
                    )
            else:
                logging.warning(f"📝 LLM call failed - {log_data}")
                logging.debug(f"🔍 LLM prompt: {redacted_prompt}")

                # Log quality metrics for failed calls
                logging.warning(
                    f"❌ LLM response quality issues: {quality_metrics['issues']}"
                )

        except Exception as e:
            logging.warning(f"⚠️ Failed to log LLM call: {str(e)}")

    def _generate_cache_key(
        self, contacts: List[Dict[str, Any]], icp_config: Dict[str, Any]
    ) -> str:
        """
        Generate cache key for contacts and ICP config.
        Guards against very large contact lists by not caching them.

        Args:
            contacts: List of contacts
            icp_config: ICP configuration

        Returns:
            Cache key string or empty string if list is too large
        """
        try:
            # Guard against very large contact lists
            if len(contacts) > LLM_CACHE_MAX_CONTACT_LIST_SIZE:
                logging.info(
                    f"📏 Contact list too large for caching ({len(contacts)} > {LLM_CACHE_MAX_CONTACT_LIST_SIZE}), skipping cache"
                )
                return ""

            # Create a hash of the contacts and ICP config
            import hashlib

            # Extract key fields for caching (limit to first 10 contacts)
            contact_data = []
            for contact in contacts[:10]:
                contact_data.append(
                    {
                        "contact_id": contact.get("contact_id"),
                        "role": contact.get("role") or contact.get("job_title"),
                        "seniority": contact.get("seniority"),
                        "company": contact.get("company"),
                    }
                )

            # Create cache key from contacts and ICP config
            cache_data = {
                "contacts": sorted(contact_data, key=lambda x: x.get("contact_id", "")),
                "icp_type": icp_config.get("icp_type", "default"),
                "icp_text_hash": hashlib.md5(
                    icp_config.get("icp_text", "").encode()
                ).hexdigest()[:8],
            }

            return hashlib.sha256(
                json.dumps(cache_data, sort_keys=True).encode()
            ).hexdigest()

        except Exception as e:
            logging.warning(f"⚠️ Failed to generate cache key: {str(e)}")
            return ""

    def _get_cached_result(self, cache_key: str) -> Optional[str]:
        """
        Get cached result if available and not expired.

        Args:
            cache_key: Cache key

        Returns:
            Cached contact ID or None
        """
        if not cache_key:
            return None

        try:
            if cache_key in self._cache:
                timestamp = self._cache_timestamps.get(cache_key, 0)
                if time.time() - timestamp < LLM_CACHE_TTL:
                    return self._cache[cache_key]
                else:
                    # Remove expired cache entry
                    del self._cache[cache_key]
                    del self._cache_timestamps[cache_key]

            return None

        except Exception as e:
            logging.warning(f"⚠️ Failed to get cached result: {str(e)}")
            return None

    def _cache_result(self, cache_key: str, contact_id: str):
        """
        Cache the result for future use.

        Args:
            cache_key: Cache key
            contact_id: Contact ID to cache
        """
        if not cache_key:
            return

        try:
            self._cache[cache_key] = contact_id
            self._cache_timestamps[cache_key] = time.time()

            # Clean up old cache entries
            self._cleanup_cache()

        except Exception as e:
            logging.warning(f"⚠️ Failed to cache result: {str(e)}")

    def _cleanup_cache(self):
        """Clean up expired cache entries and enforce size limits."""
        try:
            current_time = time.time()

            # Remove expired entries
            expired_keys = [
                key
                for key, timestamp in self._cache_timestamps.items()
                if current_time - timestamp > LLM_CACHE_TTL
            ]

            for key in expired_keys:
                del self._cache[key]
                del self._cache_timestamps[key]

            # Enforce cache size limit (remove oldest entries if needed)
            if len(self._cache) > LLM_CACHE_MAX_SIZE:
                # Sort by timestamp and remove oldest entries
                sorted_items = sorted(
                    self._cache_timestamps.items(),
                    key=lambda x: x[1],  # Sort by timestamp
                )

                # Remove oldest entries until we're under the limit
                entries_to_remove = len(self._cache) - LLM_CACHE_MAX_SIZE
                for i in range(entries_to_remove):
                    key_to_remove = sorted_items[i][0]
                    del self._cache[key_to_remove]
                    del self._cache_timestamps[key_to_remove]

                logging.info(
                    f"🧹 Cache size limit enforced, removed {entries_to_remove} oldest entries"
                )

        except Exception as e:
            logging.warning(f"⚠️ Failed to cleanup cache: {str(e)}")

    def get_cost_info(self) -> Dict[str, Any]:
        """
        Get cost information for this LLM session.

        Returns:
            Dictionary with total cost and detailed breakdown
        """
        return {
            "total_cost_usd": self.total_cost,
            "cost_details": self.cost_details.copy(),
            "total_calls": len(self.cost_details),
            "total_tokens": sum(detail["total_tokens"] for detail in self.cost_details),
        }

    def _redact_pii(self, text: str) -> str:
        """
        Redact PII from text for safe logging.

        Args:
            text: Text to redact

        Returns:
            Redacted text
        """
        # Simple PII redaction - replace emails and phone numbers

        # Redact email addresses
        text = re.sub(
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "[EMAIL]", text
        )

        # Redact phone numbers
        text = re.sub(r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b", "[PHONE]", text)

        # Redact names (simple heuristic)
        text = re.sub(r"\b[A-Z][a-z]+ [A-Z][a-z]+\b", "[NAME]", text)

        return text

    def _build_contact_selection_prompt(
        self,
        contacts: List[Dict[str, Any]],
        icp_config: Dict[str, Any],
        role_priority_config: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Build prompt for contact selection with role priorities and business context.

        Args:
            contacts: List of contacts
            icp_config: ICP configuration
            role_priority_config: Optional role priority configuration

        Returns:
            Formatted prompt string
        """
        icp_text = icp_config.get("icp_text", "General business contacts")
        icp_type = icp_config.get("icp_type", "default")

        # Format contacts for LLM consumption with role priorities
        formatted_contacts = format_contacts_for_llm(contacts, role_priority_config)

        # Add role priority context to the prompt if available
        role_context = ""
        if role_priority_config:
            high_priority_roles = [
                role for role, priority in role_priority_config.items() if priority <= 2
            ]
            if high_priority_roles:
                role_context = (
                    f"\nHigh Priority Roles: {', '.join(high_priority_roles[:5])}"
                )

        # Load and use prompt template from file
        try:
            template = load_prompt_template(self.prompt_template)
            prompt = template.replace("{icp_text}", icp_text)
            prompt = prompt.replace("{contacts}", formatted_contacts)
            prompt = prompt.replace("{role_context}", role_context)
            return prompt
        except Exception as e:
            logging.warning(
                f"⚠️ Failed to load prompt template, using fallback: {str(e)}"
            )
            return self._get_fallback_prompt(
                icp_text, formatted_contacts, icp_type, role_context
            )

    def _get_fallback_prompt(
        self,
        icp_text: str,
        formatted_contacts: str,
        icp_type: str,
        role_context: str = "",
    ) -> str:
        """
        Get fallback prompt when template loading fails.

        Args:
            icp_text: ICP description
            formatted_contacts: Formatted contact list
            icp_type: Type of ICP
            role_context: Role priority context

        Returns:
            Fallback prompt string
        """
        # Fallback to hardcoded prompts if template files are not available
        if icp_type == "investment":
            return f"""ICP: {icp_text}

Select the BEST contact who can actually allocate capital or make investment decisions. Prioritize:
- Family office decision makers
- Investment fund founders/managers
- Capital allocators with crypto exposure
- HNW individuals with investment capacity
- Portfolio managers and analysts

AVOID: Marketing executives, consultants, or people without investment decision-making power.
{role_context}

Contacts:
{formatted_contacts}

Return ONLY a JSON response in this exact format:
{{ "contact_id": "123" }}"""

        elif icp_type == "healthcare" or icp_type == "dental":
            return f"""ICP: {icp_text}

Select the BEST contact who can make healthcare business decisions or allocate resources. Prioritize:
- Practice owners and administrators
- Medical directors and CMOs
- Healthcare executives and managers
- Decision makers in healthcare organizations

AVOID: Marketing executives, consultants, or people without healthcare decision-making power.
{role_context}

Contacts:
{formatted_contacts}

Return ONLY a JSON response in this exact format:
{{ "contact_id": "123" }}"""

        else:
            # Default prompt for other ICP types
            return f"""ICP: {icp_text}

Select the BEST contact who can actually make business decisions or allocate resources. Prioritize:
- Decision makers and executives
- People with budget authority
- Senior managers and directors
- Business owners and founders

AVOID: Marketing executives, consultants, or people without decision-making power.
{role_context}

Contacts:
{formatted_contacts}

Return ONLY a JSON response in this exact format:
{{ "contact_id": "123" }}"""


class MailgunClient:
    """Client for Mailgun SMTP validation."""

    def __init__(self, smtp_login: str, smtp_password: str):
        self.smtp_login = smtp_login
        self.smtp_password = smtp_password
        self.base_url = "https://api.mailgun.net/v3"
        self.domain = smtp_login.split("@")[1] if "@" in smtp_login else None

    async def validate_email(self, email: str) -> Dict[str, Any]:
        """
        Validate email address using Mailgun.

        Args:
            email: Email address to validate

        Returns:
            Validation result with status and confidence
        """
        url = f"{self.base_url}/address/validate"
        params = {"address": email, "api_key": self.smtp_password}

        async with httpx.AsyncClient(
            timeout=httpx.Timeout(
                connect=CONNECT_TIMEOUT, read=READ_TIMEOUT, write=None, pool=None
            )
        ) as client:
            response = await client.get(url, params=params)
            response.raise_for_status()
            return response.json()


# ============================================================================
# Field Mapping Functions
# ============================================================================


# Field mapping now handled by router and providers


# Contact field mapping now handled by router and providers


def parse_contact_names(contacts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Parse full names into first/last names for email discovery.

    Args:
        contacts: List of contacts with potential full_name fields

    Returns:
        List of contacts with parsed first_name and last_name fields
    """
    for contact in contacts:
        if not isinstance(contact, dict):
            continue

        # If contact already has first_name and last_name, skip
        if contact.get("first_name") and contact.get("last_name"):
            continue

        # Check for full_name field
        full_name = contact.get("full_name") or contact.get("name")
        if not full_name:
            continue

        # Parse the full name
        first_name, last_name = parse_full_name(full_name)

        # Update the contact with parsed names
        contact["first_name"] = first_name
        contact["last_name"] = last_name

        logging.info(
            f"✅ Parsed name: '{full_name}' -> first: '{first_name}', last: '{last_name}'"
        )

    return contacts


def parse_full_name(full_name: str) -> Tuple[str, str]:
    """
    Parse a full name into first and last names, handling prefixes and suffixes.

    Args:
        full_name: Full name string (e.g., "mr. david sable", "ms. barbara guggenheim")

    Returns:
        Tuple of (first_name, last_name)
    """
    if not full_name:
        return "", ""

    # Common name prefixes and suffixes to remove
    prefixes = [
        "mr.",
        "ms.",
        "mrs.",
        "dr.",
        "prof.",
        "professor",
        "sir",
        "madam",
        "captain",
        "general",
    ]
    suffixes = ["jr.", "sr.", "ii", "iii", "iv", "phd", "mba", "esq", "esquire"]

    # Convert to lowercase for comparison
    name_lower = full_name.lower().strip()

    # Remove prefixes
    for prefix in prefixes:
        if name_lower.startswith(prefix):
            full_name = full_name[len(prefix) :].strip()
            break

    # Remove suffixes
    for suffix in suffixes:
        if name_lower.endswith(suffix):
            full_name = full_name[: -len(suffix)].strip()
            break

    # Split remaining name into parts
    name_parts = full_name.strip().split()

    if len(name_parts) == 0:
        return "", ""
    elif len(name_parts) == 1:
        return name_parts[0], ""
    elif len(name_parts) == 2:
        return name_parts[0], name_parts[1]
    else:
        # For names with 3+ parts, first is first name, rest is last name
        return name_parts[0], " ".join(name_parts[1:])


# ============================================================================
# Core Enrichment Logic
# ============================================================================


def validate_llm_response(llm_response: str) -> bool:
    """
    Validate LLM response format before parsing.
    Handles various response formats gracefully.

    Args:
        llm_response: Raw LLM response string

    Returns:
        True if response format is valid, False otherwise
    """
    if not llm_response or not isinstance(llm_response, str):
        return False

    # Clean up the response - remove markdown formatting if present
    cleaned_response = llm_response.strip()

    # Handle various markdown formats
    if "```" in cleaned_response:
        # Find the start of the markdown block
        start_marker = cleaned_response.find("```")
        if start_marker != -1:
            # Find the end of the opening markdown block
            first_newline = cleaned_response.find("\n", start_marker)
            if first_newline != -1:
                # Remove everything before the content
                cleaned_response = cleaned_response[first_newline + 1 :]

        # Find the end of the markdown block
        end_marker = cleaned_response.rfind("```")
        if end_marker != -1:
            # Remove the closing ```
            cleaned_response = cleaned_response[:end_marker]

    cleaned_response = cleaned_response.strip()

    try:
        # Try to parse JSON response
        response_data = json.loads(cleaned_response)

        # Check if it's a dictionary
        if not isinstance(response_data, dict):
            return False

        # Check if contact_id key exists
        if "contact_id" not in response_data:
            return False

        # Check if contact_id is not empty
        contact_id = response_data["contact_id"]
        if not contact_id or not isinstance(contact_id, str):
            return False

        return True

    except (json.JSONDecodeError, KeyError, TypeError):
        return False


def extract_contact_id(
    llm_response: str, contacts: List[Dict[str, Any]]
) -> Optional[str]:
    """
    Extract contact ID from LLM response with validation.
    Handles various response formats gracefully.

    Args:
        llm_response: Raw LLM response string
        contacts: List of contacts for validation

    Returns:
        Valid contact ID or None if invalid
    """
    # First validate the response format
    if not validate_llm_response(llm_response):
        logging.warning(f"⚠️ Invalid LLM response format: {llm_response[:100]}...")
        return None

    try:
        # Clean up the response - remove markdown formatting if present
        cleaned_response = llm_response.strip()

        # Handle various markdown formats
        if "```" in cleaned_response:
            # Find the start of the markdown block
            start_marker = cleaned_response.find("```")
            if start_marker != -1:
                # Find the end of the opening markdown block
                first_newline = cleaned_response.find("\n", start_marker)
                if first_newline != -1:
                    # Remove everything before the content
                    cleaned_response = cleaned_response[first_newline + 1 :]

            # Find the end of the markdown block
            end_marker = cleaned_response.rfind("```")
            if end_marker != -1:
                # Remove the closing ```
                cleaned_response = cleaned_response[:end_marker]

        cleaned_response = cleaned_response.strip()

        # Parse JSON response
        response_data = json.loads(cleaned_response)
        contact_id = response_data["contact_id"]

        # Validate that contact_id exists in contacts list
        contact_ids = [c.get("contact_id") for c in contacts if c.get("contact_id")]
        if contact_id in contact_ids:
            logging.info(f"✅ Valid contact_id extracted: {contact_id}")
            return contact_id
        else:
            logging.warning(
                f"⚠️ LLM returned contact_id {contact_id} not found in contacts list"
            )
            # Track invalid contact_id fallback
            track_fallback_metric("invalid_contact_ids")
            return None

    except Exception as e:
        logging.error(f"❌ Error extracting contact_id: {str(e)}")
        return None


def format_contacts_for_llm(
    contacts: List[Dict[str, Any]],
    role_priority_config: Optional[Dict[str, int]] = None,
) -> str:
    """
    Format contacts for LLM consumption with role priorities and business context.

    Args:
        contacts: List of contacts
        role_priority_config: Optional role priority configuration for context

    Returns:
        Formatted string representation
    """
    if not contacts:
        return "No contacts available."

    formatted = []
    for i, contact in enumerate(
        contacts[:10]
    ):  # Cap at 10 contacts to avoid token limits
        contact_id = contact.get("contact_id", f"contact_{i}")
        name = contact.get("name", "Unknown")
        role = contact.get("role") or contact.get("job_title", "Unknown")
        company = contact.get("company", "Unknown")
        seniority = contact.get("seniority", "Unknown")

        # Handle decision maker status
        decision_maker = contact.get("decision_maker", False)
        decision_status = "Decision Maker" if decision_maker else "Non-Decision Maker"

        # Handle email status for contact quality
        email_status = contact.get("email_status", "unknown")
        if isinstance(email_status, list):
            email_status = email_status[0] if email_status else "unknown"

        # Handle LinkedIn presence
        linkedin = contact.get("linkedin") or contact.get("linkedin_url", "")
        linkedin_status = "Has LinkedIn" if linkedin else "No LinkedIn"

        # Get role priority if available
        role_priority = ""
        if role_priority_config:
            priority = get_role_priority(role, role_priority_config)
            role_priority = f" | Priority: {priority}"

        # Build comprehensive contact description
        contact_description = (
            f"{i + 1}. ID: {contact_id} | "
            f"Name: {name} | "
            f"Role: {role} | "
            f"Company: {company} | "
            f"Seniority: {seniority} | "
            f"Status: {decision_status} | "
            f"Email: {email_status} | "
            f"LinkedIn: {linkedin_status}"
            f"{role_priority}"
        )

        formatted.append(contact_description)

    if len(contacts) > 10:
        formatted.append(
            f"... and {len(contacts) - 10} more contacts (showing first 10)"
        )

    return "\n".join(formatted)


async def llm_with_fallback(fn_llm, fn_rule_based, *args, **kwargs) -> Any:
    """
    Execute LLM function with fallback to rule-based function.

    Args:
        fn_llm: LLM-based function to try first
        fn_rule_based: Rule-based fallback function
        *args: Arguments to pass to both functions
        **kwargs: Keyword arguments to pass to both functions

    Returns:
        Result from LLM function or fallback function
    """
    try:
        result = await fn_llm(*args, **kwargs)
        if result is not None:
            logging.info("✅ LLM function succeeded")
            return result
        else:
            logging.info("🔄 LLM returned None, falling back to rule-based")
            # Track fallback metric
            track_fallback_metric("llm_none_responses")
            return fn_rule_based(*args, **kwargs)
    except Exception as e:
        logging.warning(f"⚠️ LLM function failed: {str(e)}, falling back to rule-based")
        # Track fallback metric
        track_fallback_metric("llm_failures")
        return fn_rule_based(*args, **kwargs)


async def select_best_contact_llm(
    contacts: List[Dict[str, Any]],
    role_priority_config: Dict[str, int],
    icp_config: Optional[Dict[str, Any]] = None,
) -> Tuple[Optional[str], Dict[str, Any]]:
    """
    Select best contact using LLM with fallback to rule-based.

    Args:
        contacts: List of contacts
        role_priority_config: Role priority configuration
        icp_config: ICP configuration for business context

    Returns:
        Tuple of (Contact ID of best contact, Cost information)
    """
    if not contacts:
        return None, {
            "total_cost_usd": 0.0,
            "cost_details": [],
            "total_calls": 0,
            "total_tokens": 0,
        }

    if not icp_config:
        logging.info("ℹ️ No ICP config provided, using rule-based selection")
        return select_best_contact(contacts, role_priority_config), {
            "total_cost_usd": 0.0,
            "cost_details": [],
            "total_calls": 0,
            "total_tokens": 0,
        }

    # Get OpenRouter API key from environment
    api_key = os.getenv("OPENROUTER_KEY")
    if not api_key:
        logging.warning("⚠️ No OPENROUTER_KEY found, using rule-based selection")
        return select_best_contact(contacts, role_priority_config), {
            "total_cost_usd": 0.0,
            "cost_details": [],
            "total_calls": 0,
            "total_tokens": 0,
        }

    # Get ICP type for configuration
    icp_type = icp_config.get("icp_type", "default")

    # Create LLM client with ICP-specific configuration
    llm_selector = LLMContactSelector(api_key, icp_type, icp_config)

    # Create wrapper functions for proper parameter passing
    async def llm_function(contacts_param, icp_param):
        result = await llm_selector.select_contact(
            contacts_param, icp_param, role_priority_config
        )
        return result, llm_selector.get_cost_info()

    def fallback_function(contacts_param, _):
        track_llm_performance_metric("fallback_to_rule_based")
        return select_best_contact(contacts_param, role_priority_config), {
            "total_cost_usd": 0.0,
            "cost_details": [],
            "total_calls": 0,
            "total_tokens": 0,
        }

    # Try LLM selection with fallback
    try:
        result = await llm_function(contacts, icp_config)
        if result[0] is not None:
            return result
        else:
            logging.info("🔄 LLM returned None, falling back to rule-based")
            track_fallback_metric("llm_none_responses")
            return fallback_function(contacts, icp_config)
    except Exception as e:
        logging.warning(f"⚠️ LLM function failed: {str(e)}, falling back to rule-based")
        track_fallback_metric("llm_failures")
        return fallback_function(contacts, icp_config)


def track_fallback_metric(metric_type: str):
    """
    Track fallback metrics for monitoring and analysis.

    Args:
        metric_type: Type of fallback metric to track
    """
    global FALLBACK_METRICS

    if metric_type in FALLBACK_METRICS:
        FALLBACK_METRICS[metric_type] += 1
        FALLBACK_METRICS["total_fallbacks"] += 1

        # Log the metric for monitoring
        logging.info(
            f"📊 FALLBACK_METRIC: {metric_type} - Total: {FALLBACK_METRICS[metric_type]}"
        )

    # Log current fallback statistics
    logging.info(f"📊 FALLBACK_STATS: {FALLBACK_METRICS}")


def get_fallback_metrics() -> Dict[str, int]:
    """
    Get current fallback metrics for monitoring.

    Returns:
        Dictionary of fallback metrics
    """
    return FALLBACK_METRICS.copy()


def track_llm_performance_metric(metric_type: str, value: int = 1):
    """
    Track LLM performance metrics for monitoring.

    Args:
        metric_type: Type of performance metric to track
        value: Value to add (default: 1)
    """
    global LLM_PERFORMANCE_METRICS
    if metric_type in LLM_PERFORMANCE_METRICS:
        LLM_PERFORMANCE_METRICS[metric_type] += value
        logging.debug(
            f"📊 LLM_PERFORMANCE: {metric_type} - Total: {LLM_PERFORMANCE_METRICS[metric_type]}"
        )
    else:
        logging.warning(f"⚠️ Unknown LLM performance metric type: {metric_type}")


def get_llm_performance_metrics() -> Dict[str, Any]:
    """
    Get current LLM performance metrics.

    Returns:
        Dictionary of LLM performance metrics with calculated averages
    """
    metrics = LLM_PERFORMANCE_METRICS.copy()
    # Calculate average response time
    if metrics["total_calls"] > 0:
        metrics["average_response_time_ms"] = (
            metrics["total_response_time_ms"] / metrics["total_calls"]
        )
    return metrics


def reset_llm_performance_metrics():
    """Reset LLM performance metrics (useful for testing)."""
    global LLM_PERFORMANCE_METRICS
    for key in LLM_PERFORMANCE_METRICS:
        LLM_PERFORMANCE_METRICS[key] = 0


def analyze_llm_response_quality(
    response: str, expected_format: str = "json"
) -> Dict[str, Any]:
    """
    Analyze the quality of LLM responses for monitoring and improvement.

    Args:
        response: The LLM response to analyze
        expected_format: Expected format (default: "json")

    Returns:
        Dictionary with quality metrics
    """
    quality_metrics = {
        "length": len(response),
        "is_valid_json": False,
        "has_contact_id": False,
        "format_compliance": 0.0,
        "issues": [],
    }

    try:
        # Check if response is valid JSON
        if expected_format == "json":
            parsed = json.loads(response)
            quality_metrics["is_valid_json"] = True

            # Check for required fields
            if "contact_id" in parsed:
                quality_metrics["has_contact_id"] = True
                quality_metrics["format_compliance"] += 0.5
            else:
                quality_metrics["issues"].append("missing_contact_id")

            # Check if contact_id is not empty
            if parsed.get("contact_id"):
                quality_metrics["format_compliance"] += 0.5
            else:
                quality_metrics["issues"].append("empty_contact_id")

        else:
            quality_metrics["issues"].append("unexpected_format")

    except json.JSONDecodeError:
        quality_metrics["issues"].append("invalid_json")
    except Exception as e:
        quality_metrics["issues"].append(f"parsing_error: {str(e)}")

    # Log quality metrics for monitoring
    if quality_metrics["format_compliance"] < 1.0:
        logging.warning(f"📊 LLM response quality issues: {quality_metrics['issues']}")
    else:
        logging.debug(
            f"📊 LLM response quality: {quality_metrics['format_compliance']:.2f}"
        )

    return quality_metrics


def get_llm_monitoring_summary() -> Dict[str, Any]:
    """
    Get comprehensive LLM monitoring summary.

    Returns:
        Dictionary with fallback and performance metrics
    """
    fallback_metrics = get_fallback_metrics()
    performance_metrics = get_llm_performance_metrics()

    # Calculate success rate
    total_calls = performance_metrics.get("total_calls", 0)
    successful_calls = performance_metrics.get("successful_calls", 0)
    success_rate = (successful_calls / total_calls * 100) if total_calls > 0 else 0

    # Calculate cache hit rate
    cache_hits = performance_metrics.get("cache_hits", 0)
    cache_misses = performance_metrics.get("cache_misses", 0)
    cache_hit_rate = (
        (cache_hits / (cache_hits + cache_misses) * 100)
        if (cache_hits + cache_misses) > 0
        else 0
    )

    return {
        "fallback_metrics": fallback_metrics,
        "performance_metrics": performance_metrics,
        "calculated_metrics": {
            "success_rate_percent": round(success_rate, 2),
            "cache_hit_rate_percent": round(cache_hit_rate, 2),
            "fallback_rate_percent": round(
                (fallback_metrics.get("total_fallbacks", 0) / total_calls * 100)
                if total_calls > 0
                else 0,
                2,
            ),
        },
    }


def load_prompt_template(template_name: str) -> str:
    """
    Load prompt template from file.

    Args:
        template_name: Name of the template file

    Returns:
        Template content as string
    """
    try:
        template_path = (
            Path(__file__).parent.parent / "config" / "prompts" / template_name
        )
        with open(template_path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        logging.warning(f"⚠️ Prompt template {template_name} not found, using default")
        # Return a basic fallback template to avoid recursion
        return """ICP: {icp_text}

Select the BEST contact who can actually make business decisions or allocate resources. Prioritize:
- Decision makers and executives
- People with budget authority
- Senior managers and directors
- Business owners and founders

AVOID: Marketing executives, consultants, or people without decision-making power.
{role_context}

Contacts:
{contacts}

Return ONLY a JSON response in this exact format:
{ "contact_id": "123" }"""
    except Exception as e:
        logging.error(f"❌ Failed to load prompt template {template_name}: {str(e)}")
        # Return a basic fallback template to avoid recursion
        return """ICP: {icp_text}

Select the BEST contact who can actually make business decisions or allocate resources. Prioritize:
- Decision makers and executives
- People with budget authority
- Senior managers and directors
- Business owners and founders

AVOID: Marketing executives, consultants, or people without decision-making power.
{role_context}

Contacts:
{contacts}

Return ONLY a JSON response in this exact format:
{ "contact_id": "123" }"""


def get_icp_llm_config(icp_type: str, icp_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get LLM configuration for specific ICP, with overrides from ICP config.

    Args:
        icp_type: ICP type identifier
        icp_config: Full ICP configuration

    Returns:
        Merged LLM configuration
    """
    # Get base config for ICP type
    base_config = LLM_ICP_CONFIGS.get(icp_type, LLM_ICP_CONFIGS["default"]).copy()

    # Apply ICP-specific overrides if present
    if "llm_config" in icp_config:
        llm_overrides = icp_config["llm_config"]
        base_config.update(llm_overrides)

    # Ensure all required fields are present
    required_fields = [
        "model",
        "fallback_model",
        "temperature",
        "max_tokens",
        "timeout",
        "prompt_template",
    ]
    for field in required_fields:
        if field not in base_config:
            base_config[field] = LLM_ICP_CONFIGS["default"][field]

    return base_config


def test_llm_response_handling():
    """
    Test function to validate LLM response handling with various formats.
    This function tests the response validation and extraction logic.
    """
    # Test contacts
    test_contacts = [
        {"contact_id": "123", "name": "John Doe", "role": "CEO"},
        {"contact_id": "456", "name": "Jane Smith", "role": "CTO"},
    ]

    # Test various response formats
    test_responses = [
        '{"contact_id": "123"}',  # Valid JSON
        '```json\n{"contact_id": "456"}\n```',  # Markdown JSON
        '{"contact_id": "789"}',  # Invalid contact_id
        '{"wrong_key": "123"}',  # Missing contact_id
        "invalid json",  # Invalid JSON
        '{"contact_id": ""}',  # Empty contact_id
        '{"contact_id": null}',  # Null contact_id
    ]

    print("🧪 Testing LLM response handling...")

    for i, response in enumerate(test_responses):
        is_valid = validate_llm_response(response)
        contact_id = extract_contact_id(response, test_contacts)

        print(f"Test {i + 1}: {response[:30]}...")
        print(f"  Valid: {is_valid}")
        print(f"  Contact ID: {contact_id}")
        print()


def test_prompt_variations():
    """
    Test different prompt variations to ensure LLM understands business context.
    This function tests prompt engineering and business context understanding.
    """
    print("🧪 Testing Prompt Variations and Business Context...")

    # Test different ICP types
    icp_types = ["investment", "healthcare", "technology", "enterprise", "dental"]

    for icp_type in icp_types:
        print(f"\n📋 Testing {icp_type.upper()} ICP type:")

        # Get configuration for this ICP type
        config = LLM_ICP_CONFIGS.get(icp_type, LLM_ICP_CONFIGS["default"])

        print(f"  Model: {config['model']}")
        print(f"  Temperature: {config['temperature']}")
        print(f"  Max Tokens: {config['max_tokens']}")
        print(f"  Timeout: {config['timeout']}")
        print(f"  Prompt Template: {config['prompt_template']}")

        # Test prompt template loading
        try:
            template = load_prompt_template(config["prompt_template"])
            print(f"  ✅ Template loaded successfully ({len(template)} chars)")
        except Exception as e:
            print(f"  ❌ Template loading failed: {str(e)}")

    print("\n✅ Prompt variation testing completed!")


def test_icp_configuration_overrides():
    """
    Test ICP configuration overrides for LLM settings.
    This function tests the configurable LLM parameters per ICP.
    """
    print("🧪 Testing ICP Configuration Overrides...")

    # Test ICP config with LLM overrides
    test_icp_config = {
        "icp_type": "investment",
        "icp_text": "Test investment ICP",
        "llm_config": {"temperature": 0.2, "max_tokens": 800, "timeout": 60.0},
    }

    # Get merged configuration
    merged_config = get_icp_llm_config("investment", test_icp_config)

    print("📋 Base investment config:")
    print(f"  Temperature: {LLM_ICP_CONFIGS['investment']['temperature']}")
    print(f"  Max Tokens: {LLM_ICP_CONFIGS['investment']['max_tokens']}")
    print(f"  Timeout: {LLM_ICP_CONFIGS['investment']['timeout']}")

    print("\n📋 Merged config with overrides:")
    print(f"  Temperature: {merged_config['temperature']}")
    print(f"  Max Tokens: {merged_config['max_tokens']}")
    print(f"  Timeout: {merged_config['timeout']}")

    # Verify overrides are applied
    assert merged_config["temperature"] == 0.2, "Temperature override not applied"
    assert merged_config["max_tokens"] == 800, "Max tokens override not applied"
    assert merged_config["timeout"] == 60.0, "Timeout override not applied"

    print("✅ Configuration overrides working correctly!")


def select_best_contact(
    contacts: List[Dict[str, Any]], role_priority_config: Dict[str, int]
) -> Optional[str]:
    """
    Select best contact using deterministic ranking.

    Args:
        contacts: List of contacts
        role_priority_config: Role priority configuration

    Returns:
        Contact ID of best contact or None
    """
    if not contacts:
        return None

    # Score each contact
    scored_contacts = []
    for contact in contacts:
        contact_id = contact.get("contact_id", "")
        # Handle both old and new field names
        role = contact.get("role") or contact.get("job_title", "")
        seniority = contact.get("seniority", 99)
        email_status = (
            contact.get("email_status", ["unknown"])[0]
            if isinstance(contact.get("email_status"), list)
            else contact.get("email_status", "unknown")
        )
        # Handle both old and new field names
        linkedin = contact.get("linkedin") or contact.get("linkedin_url", "")

        # Get role priority
        role_priority = get_role_priority(role, role_priority_config)

        # Get seniority rank
        seniority_rank = get_seniority_rank(role) if seniority is None else seniority

        # Get email status rank
        email_status_rank = get_email_status_rank(email_status)

        # Check if has LinkedIn
        has_linkedin = 0 if linkedin else 1

        # Create sort tuple: (role_priority, seniority_rank, email_status_rank, has_linkedin, contact_id)
        sort_tuple = (
            role_priority,
            seniority_rank,
            email_status_rank,
            has_linkedin,
            contact_id,
        )
        scored_contacts.append((sort_tuple, contact_id))

    # Sort by tuple (ascending) and pick first
    scored_contacts.sort(key=lambda x: x[0])
    return scored_contacts[0][1] if scored_contacts else None


# Company enrichment now handled by router


# Contact enrichment now handled by router


# Contact discovery now handled by router


def write_evidence_file(
    domain: str, evidence_data: Dict[str, Any], data_dir: str
) -> str:
    """
    Write evidence to file and return path.

    Args:
        domain: Company domain
        evidence_data: Evidence data to write
        data_dir: Data directory

    Returns:
        Evidence file path
    """
    # Normalize domain for file path
    normalized_domain = normalize_domain(domain)

    # Create evidence directory
    evidence_dir = os.path.join(data_dir, "evidence", "enrich")
    os.makedirs(evidence_dir, exist_ok=True)

    # Write evidence file
    evidence_path = os.path.join(evidence_dir, f"{normalized_domain}.json")

    try:
        # Write directly without locking (Windows compatibility)
        with open(evidence_path, "w") as f:
            json.dump(evidence_data, f, indent=2)
    except Exception as e:
        raise RuntimeError(f"Failed to write evidence file {evidence_path}: {e}")

    return evidence_path


# ============================================================================
# Main Tool Logic
# ============================================================================


async def run_enrich(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main enrichment function.

    Args:
        payload: Input payload with lead_records, icp_config, and caps

    Returns:
        Enrichment results with data, errors, and metrics
    """
    print("🚀🚀🚀 RUN_ENRICH FUNCTION STARTED 🚀🚀🚀", flush=True)
    start_time = time.time()

    # Resolve data directory first for logging setup. Child tool will use shared
    # run log path from env if orchestrator provided one.
    data_dir = resolve_data_dir(payload)
    setup_logging(TOOL_NAME, data_dir=data_dir)

    try:
        # Validate input
        if not isinstance(payload, dict):
            return {
                "data": {"lead_records": []},
                "errors": [build_error(ErrorCode.SCHEMA_VALIDATION, tool=TOOL_NAME)],
                "metrics": build_metrics(0, 0, 0),
            }

        lead_records = payload.get("lead_records", [])
        icp_config = payload.get("icp_config", {})
        caps = payload.get("caps", {})

        if not lead_records or not icp_config:
            return {
                "data": {"lead_records": []},
                "errors": [build_error(ErrorCode.SCHEMA_VALIDATION, tool=TOOL_NAME)],
                "metrics": build_metrics(0, 0, 0),
            }

        # Validate testing flags
        try:
            logging.info("🔍 Validating testing flags...")
            logging.info(f"🔍 ICP config keys: {list(icp_config.keys())}")
            logging.info(
                f"🔍 Testing section: {icp_config.get('testing', 'NOT_FOUND')}"
            )
            logging.info("🔍 About to call validate_testing_flags...")
            validate_testing_flags(icp_config)
            logging.info("🔍 validate_testing_flags returned successfully")
            logging.info("✅ Testing flags validation passed")
        except ValueError as exc:
            return {
                "data": {"lead_records": []},
                "errors": [
                    build_error(ErrorCode.SCHEMA_VALIDATION, exc=exc, tool=TOOL_NAME)
                ],
                "metrics": build_metrics(0, 0, 0),
            }

        # Load costs configuration
        try:
            logging.info("🔍 Loading costs configuration...")
            costs_config = load_costs_config()
            logging.info(f"✅ Costs config loaded: {list(costs_config.keys())}")

            # Provider validation now handled by router
            logging.info("✅ Provider configuration will be validated by router")
        except (FileNotFoundError, KeyError) as exc:
            return {
                "data": {"lead_records": []},
                "errors": [
                    build_error(ErrorCode.PROVIDER_ERROR, exc=exc, tool=TOOL_NAME)
                ],
                "metrics": build_metrics(0, 0, 0),
            }

        # Load schema checksum
        try:
            logging.info("🔍 Loading schema checksum...")
            schema_checksum = load_schema_checksum()
            logging.info(f"✅ Schema checksum loaded: {schema_checksum[:8]}...")
        except FileNotFoundError as exc:
            return {
                "data": {"lead_records": []},
                "errors": [
                    build_error(ErrorCode.SCHEMA_VALIDATION, exc=exc, tool=TOOL_NAME)
                ],
                "metrics": build_metrics(0, 0, 0),
            }

        # Data directory already resolved for logging setup above
        logging.info(f"✅ Using data directory: {data_dir}")

        # Note: Removed permit manager and semaphore pools since we no longer use file locking
        logging.info("🔍 Skipping concurrency controls (no file locking needed)")

        # Initialize provider clients
        logging.info("🔍 Checking environment variables...")
        mailgun_login = os.environ.get("MAILGUN_SMTP_LOGIN")
        mailgun_password = os.environ.get("MAILGUN_SMTP_PW")

        logging.info(
            f"🔑 Environment check - Mailgun: {'✅' if mailgun_login else '❌'}"
        )

        # Check which specific environment variables are missing
        missing_vars = []
        if not mailgun_login:
            missing_vars.append("MAILGUN_SMTP_LOGIN")
        if not mailgun_password:
            missing_vars.append("MAILGUN_SMTP_PW")

        if missing_vars:
            return {
                "data": {"lead_records": []},
                "errors": [
                    build_error(
                        ErrorCode.PROVIDER_ERROR,
                        tool=TOOL_NAME,
                        context={
                            "error": f"Missing required environment variables: {', '.join(missing_vars)}",
                            "missing_variables": missing_vars,
                            "expected_variables": [
                                "MAILGUN_SMTP_LOGIN",
                                "MAILGUN_SMTP_PW",
                            ],
                        },
                    )
                ],
                "metrics": build_metrics(0, 0, 0),
            }

        # Initialize enrichment provider router
        enrichment_router = create_enrichment_router(icp_config)
        mailgun_client = MailgunClient(mailgun_login, mailgun_password)

        # Get configuration values
        role_priority_config = icp_config.get("role_priority", {})
        # Ensure role_priority_config is a dictionary
        if not isinstance(role_priority_config, dict):
            logging.warning(
                f"⚠️ role_priority_config is not a dictionary: {type(role_priority_config)}, resetting to empty dict"
            )
            role_priority_config = {}

        default_country = icp_config.get("default_country", "US")
        run_usd_max = caps.get("run_usd_max", float("inf"))
        company_usd_cap = caps.get("company_usd", 0.02)
        contact_usd_cap = caps.get("contact_usd", 0.03)

        # Process each lead record
        processed_records = []
        errors = []
        total_cost = 0.0
        enriched_count = 0
        partial_count = 0

        print(
            f"📋📋📋 ABOUT TO PROCESS {len(lead_records)} LEAD RECORDS 📋📋📋",
            flush=True,
        )
        for record in lead_records:
            try:
                # EARLY DEBUG: Log that we're starting to process a record
                logging.info(
                    f"🚀 STARTING RECORD PROCESSING FOR: {record.get('domain', 'UNKNOWN')}"
                )

                # Debug logging to see what record contains
                logging.info(
                    f"🔍 Processing record: type={type(record)}, content={record}"
                )

                # Check if we should process this record
                status = record.get("status", "")
                if status not in ["enrich_ready", "crawled"]:
                    # Skip records that aren't ready for enrichment
                    processed_records.append(record)
                    continue

                # Check if processing this lead would exceed run_usd_max
                if total_cost >= run_usd_max:
                    errors.append(
                        build_error(
                            ErrorCode.BUDGET_EXCEEDED,
                            tool=TOOL_NAME,
                            lead_id=record.get("lead_id"),
                            context={
                                "error": "Run budget exceeded",
                                "total_cost": total_cost,
                                "run_usd_max": run_usd_max,
                            },
                        )
                    )
                    # Skip this lead entirely
                    processed_records.append(record)
                    continue

                domain = record.get("domain", "")
                company = record.get("company", {})
                contacts = record.get("contacts", [])

                # Debug logging for contacts
                logging.info(f"🔍 Contacts field type for {domain}: {type(contacts)}")
                logging.info(f"🔍 Contacts field content for {domain}: {contacts}")

                # Ensure contacts is always a list
                if not isinstance(contacts, list):
                    logging.warning(
                        f"⚠️ Contacts field is not a list for {domain}, resetting to empty list"
                    )
                    contacts = []
                    record["contacts"] = contacts

                # Parse full names into first/last names for email discovery
                contacts = parse_contact_names(contacts)

                # Normalize domain
                normalized_domain = normalize_domain(domain)
                record["domain"] = normalized_domain

                # Enrich company data first (cheaper)
                company_cost = 0.0
                if company_usd_cap > 0:
                    try:
                        logging.info(
                            f"🔍 About to call enrich_company_data for {normalized_domain}"
                        )
                        # Use router for company enrichment
                        request = EnrichmentRequest(domain=normalized_domain)
                        result = await enrichment_router.enrich_company(request)

                        if result.success and result.data:
                            # Map the enriched data to company fields
                            enriched_data = result.data
                            fields_updated = 0

                            # Handle both dataclass objects and dictionaries
                            if hasattr(enriched_data, "__dataclass_fields__"):
                                # It's a dataclass object, access attributes directly
                                if enriched_data.name:
                                    company["name"] = enriched_data.name
                                    fields_updated += 1
                                if enriched_data.description:
                                    company["description"] = enriched_data.description
                                    fields_updated += 1
                                if enriched_data.industry:
                                    company["industry"] = enriched_data.industry
                                    fields_updated += 1
                                if enriched_data.size:
                                    company["size"] = enriched_data.size
                                    fields_updated += 1
                                if enriched_data.address:
                                    company["address"] = enriched_data.address
                                    fields_updated += 1
                                if enriched_data.technologies:
                                    company["technologies"] = enriched_data.technologies
                                    fields_updated += 1
                            else:
                                # It's a dictionary, use .get() method
                                if enriched_data.get("name"):
                                    company["name"] = enriched_data["name"]
                                    fields_updated += 1
                                if enriched_data.get("description"):
                                    company["description"] = enriched_data[
                                        "description"
                                    ]
                                    fields_updated += 1
                                if enriched_data.get("industry"):
                                    company["industry"] = enriched_data["industry"]
                                    fields_updated += 1
                                if enriched_data.get("size"):
                                    company["size"] = enriched_data["size"]
                                    fields_updated += 1
                                if enriched_data.get("address"):
                                    company["address"] = enriched_data["address"]
                                    fields_updated += 1
                                if enriched_data.get("technologies"):
                                    company["technologies"] = enriched_data[
                                        "technologies"
                                    ]
                                    fields_updated += 1

                            logging.info(
                                f"✅ Company enrichment completed for {normalized_domain}: {fields_updated} fields updated"
                            )
                            company_cost = result.cost
                        else:
                            logging.warning(
                                f"Company enrichment failed for {normalized_domain}: {result.error}"
                            )
                            company_cost = 0.0

                        enriched_company = company
                        logging.info(
                            f"🔍 enrich_company_data returned: type={type(enriched_company)}, content={enriched_company}"
                        )
                        company_cost = min(company_cost, company_usd_cap)
                        # Update the record's company data
                        record["company"] = enriched_company
                        company = enriched_company  # Update local company variable
                    except Exception as e:
                        logging.warning(
                            f"⚠️ Company enrichment failed for {normalized_domain}: {e}"
                        )
                        # Continue with contact discovery even if company enrichment fails
                        company_cost = 0.0

                # Ensure company is always a dictionary for safety
                if not isinstance(company, dict):
                    logging.warning(
                        f"⚠️ Company data is not a dictionary for {normalized_domain}, resetting to empty dict"
                    )
                    company = {}
                    record["company"] = company

                # Debug logging to see what company contains
                logging.info(
                    f"🔍 Company data type for {normalized_domain}: {type(company)}"
                )
                logging.info(
                    f"🔍 Company data content for {normalized_domain}: {company}"
                )
                logging.info(
                    f"🔍 Record company field type: {type(record.get('company'))}"
                )
                logging.info(
                    f"🔍 Record company field content: {record.get('company')}"
                )

                # Normalize contact data from crawling to ensure consistent field names
                if contacts and len(contacts) > 0:
                    logging.info(
                        f"🔍 Normalizing {len(contacts)} contacts from crawling for {normalized_domain}"
                    )

                    normalized_contacts = []
                    for contact in contacts:
                        normalized_contact = contact.copy()

                        # Handle different field names from crawling
                        if "name" in normalized_contact and not ("first_name" in normalized_contact or "last_name" in normalized_contact):
                            # Split full name into first and last name
                            full_name = normalized_contact.pop("name", "")
                            if full_name and full_name != "null":
                                name_parts = full_name.split(" ", 1)
                                if len(name_parts) >= 2:
                                    normalized_contact["first_name"] = name_parts[0]
                                    normalized_contact["last_name"] = name_parts[1]
                                else:
                                    normalized_contact["first_name"] = full_name
                                    normalized_contact["last_name"] = ""

                        # Handle role field mapping
                        if "role" in normalized_contact and "job_title" not in normalized_contact:
                            normalized_contact["job_title"] = normalized_contact.pop("role", "")

                        # Handle phone field mapping
                        if "phone" in normalized_contact:
                            phone = normalized_contact["phone"]
                            if phone and phone != "null":
                                normalized_contact["phone"] = phone

                        # Handle email field mapping (clean up "null" strings)
                        if "email" in normalized_contact:
                            email = normalized_contact["email"]
                            if email == "null" or not email:
                                normalized_contact["email"] = None

                        normalized_contacts.append(normalized_contact)

                    # Use normalized contacts for all subsequent processing
                    contacts = normalized_contacts
                    record["contacts"] = contacts
                    logging.info(
                        f"✅ Contact normalization completed for {normalized_domain}"
                    )

                # Select best contact using LLM-based selection with fallback to rule-based
                logging.info(
                    f"🔍 Contacts type before contact selection: {type(contacts)}"
                )
                logging.info(
                    f"🔍 Contacts content before contact selection: {contacts}"
                )
                logging.info(f"🔍 Role priority config: {role_priority_config}")
                logging.info(
                    f"🔍 ICP config available for LLM selection: {bool(icp_config)}"
                )

                # Use LLM-based contact selection with fallback to rule-based
                best_contact_id, llm_cost_info = await select_best_contact_llm(
                    contacts, role_priority_config, icp_config
                )
                record["best_contact_id"] = best_contact_id

                # Track LLM costs for contact selection
                llm_selection_cost = llm_cost_info.get("total_cost_usd", 0.0)

                # Enrich contacts (only if within budget)
                contact_cost = 0.0

                # Log the current state before contact discovery
                logging.info(
                    f"🔍 About to start contact discovery logic for {normalized_domain}"
                )
                logging.info(f"🔍 Current contacts list: {contacts}")
                logging.info(f"🔍 Contact USD cap: {contact_usd_cap}")
                logging.info(
                    f"🔍 Force contact discovery flag: {icp_config.get('testing', {}).get('force_contact_discovery', False)}"
                )

                # If no contacts exist, try to discover them using Snov.io
                if (
                    not contacts
                    or icp_config.get("testing", {}).get(
                        "force_contact_discovery", False
                    )
                ) and contact_usd_cap > 0:
                    # Debug logging to see what's happening
                    logging.info(f"🔍 DEBUG: contacts list length: {len(contacts)}")
                    logging.info(
                        f"🔍 DEBUG: force_contact_discovery flag: {icp_config.get('testing', {}).get('force_contact_discovery', False)}"
                    )
                    logging.info(f"🔍 DEBUG: contact_usd_cap: {contact_usd_cap}")

                    # If force_contact_discovery is enabled, clear existing contacts to force fresh discovery
                    if icp_config.get("testing", {}).get(
                        "force_contact_discovery", False
                    ):
                        logging.info(
                            f"🔍 Force contact discovery enabled for {normalized_domain}, clearing existing contacts and starting fresh..."
                        )
                        contacts = []
                        record["contacts"] = contacts
                        logging.info(
                            f"🔍 DEBUG: After clearing, contacts list length: {len(contacts)}"
                        )

                    logging.info(
                        f"🔍 {'No contacts found' if not contacts else 'Force contact discovery enabled'} for {normalized_domain}, attempting contact discovery..."
                    )
                    try:
                        # Use router for contact discovery
                        try:
                            request = EnrichmentRequest(domain=normalized_domain)
                            result = await enrichment_router.get_all_domain_prospects(
                                request
                            )

                            if result.success and result.data:
                                prospects_data = result.data
                                discovered_contacts = []

                                # Process prospects data to extract contacts
                                if (
                                    isinstance(prospects_data, dict)
                                    and "data" in prospects_data
                                ):
                                    prospects = prospects_data["data"]
                                elif isinstance(prospects_data, list):
                                    prospects = prospects_data
                                else:
                                    prospects = []

                                # Build decision-maker titles from role priorities
                                role_priority = icp_config.get("role_priority", {})
                                decision_maker_titles = []

                                for role, priority in role_priority.items():
                                    if priority <= 5:  # Focus on high-priority roles
                                        decision_maker_titles.append(role.title())
                                        # Add common variations
                                        if role.lower() == "ceo":
                                            decision_maker_titles.extend(
                                                [
                                                    "Chief Executive Officer",
                                                    "Chief Executive",
                                                ]
                                            )
                                        elif role.lower() == "owner":
                                            decision_maker_titles.extend(
                                                ["Founder", "Proprietor"]
                                            )
                                        elif role.lower() == "president":
                                            decision_maker_titles.extend(
                                                [
                                                    "Chief Executive",
                                                    "Executive Director",
                                                ]
                                            )
                                        elif role.lower() == "vp":
                                            decision_maker_titles.extend(
                                                ["Vice President", "Vice-President"]
                                            )
                                        elif role.lower() == "director":
                                            decision_maker_titles.extend(
                                                [
                                                    "Managing Director",
                                                    "Executive Director",
                                                    "General Manager",
                                                ]
                                            )
                                        elif role.lower() == "head":
                                            decision_maker_titles.extend(
                                                [
                                                    "Head of",
                                                    "Department Head",
                                                    "Division Head",
                                                ]
                                            )
                                        elif role.lower() == "manager":
                                            # Use configurable manager titles from ICP config, or fall back to generic ones
                                            manager_titles = icp_config.get(
                                                "role_priority", {}
                                            ).get(
                                                "manager_titles",
                                                [
                                                    "Operations Manager",
                                                    "General Manager",
                                                    "Department Manager",
                                                ],
                                            )
                                            decision_maker_titles.extend(manager_titles)

                                decision_set = {
                                    t.lower() for t in decision_maker_titles
                                }

                                # Process prospects and build contacts
                                for profile in prospects[:5]:  # Limit to 5 contacts
                                    # Handle both ContactData objects and dictionaries
                                    if hasattr(profile, "__dataclass_fields__"):
                                        # It's a ContactData object, access attributes directly
                                        title_value = profile.job_title or ""
                                        first_name = profile.first_name or ""
                                        last_name = profile.last_name or ""
                                        company_name = profile.company_name or ""
                                        linkedin_url = profile.linkedin_url or ""
                                        location = profile.location or ""
                                    else:
                                        # It's a dictionary, use .get() method
                                        title_value = (
                                            profile.get("jobTitle")
                                            or profile.get("position")
                                            or profile.get("title")
                                            or ""
                                        )
                                        first_name = profile.get("firstName", "")
                                        last_name = profile.get("lastName", "")
                                        company_name = profile.get("company", "")
                                        linkedin_url = profile.get("linkedinUrl", "")
                                        location = profile.get("location", "")

                                    # Check if title matches any decision-maker role
                                    is_decision_maker = False
                                    if title_value:
                                        title_lower = title_value.lower()
                                        # Check for exact matches first
                                        if title_lower in decision_set:
                                            is_decision_maker = True
                                        else:
                                            # Check for partial matches (e.g., "Founder & CEO" contains "founder")
                                            for decision_title in decision_set:
                                                if (
                                                    decision_title in title_lower
                                                    or title_lower in decision_title
                                                ):
                                                    is_decision_maker = True
                                                    break

                                    if not is_decision_maker:
                                        logging.info(
                                            f"🔍 Skipping contact with non-decision-maker title: '{title_value}'"
                                        )
                                        continue

                                    contact = {
                                        "contact_id": f"discovered_{len(discovered_contacts)}",
                                        "first_name": first_name,
                                        "last_name": last_name,
                                        "email": "",
                                        "job_title": title_value,
                                        "company": company_name,
                                        "linkedin_url": linkedin_url,
                                        "location": location,
                                        "discovery_source": "router",
                                        "discovery_method": "domain_prospects",
                                    }

                                    # Try to discover email for this prospect
                                    if hasattr(profile, "__dataclass_fields__"):
                                        # ContactData object - check if email is already available
                                        if profile.email:
                                            contact["email"] = profile.email
                                            contact["email_source"] = ["provider"]
                                            contact["email_status"] = [
                                                profile.email_status or "unknown"
                                            ]
                                            logging.info(
                                                f"✅ Found email from ContactData: {contact['email']} for {contact['first_name']} {contact['last_name']}"
                                            )
                                    else:
                                        # Dictionary - check for prospect hash
                                        if (
                                            profile.get("prospectHash")
                                            or profile.get("hash")
                                            or profile.get("id")
                                        ):
                                            try:
                                                prospect_request = EnrichmentRequest(
                                                    prospect_hash=profile.get(
                                                        "prospectHash"
                                                    )
                                                    or profile.get("hash")
                                                    or profile.get("id")
                                                )
                                                email_result = await enrichment_router.search_prospect_email(
                                                    prospect_request
                                                )

                                                if (
                                                    email_result.success
                                                    and email_result.data
                                                ):
                                                    email_data = email_result.data
                                                    if (
                                                        isinstance(email_data, list)
                                                        and len(email_data) > 0
                                                    ):
                                                        first_email = email_data[0]
                                                        if isinstance(
                                                            first_email, dict
                                                        ) and first_email.get("email"):
                                                            contact["email"] = (
                                                                first_email["email"]
                                                            )
                                                            logging.info(
                                                                f"✅ Found email for {contact['first_name']} {contact['last_name']}: {contact['email']}"
                                                            )
                                            except Exception as exc:
                                                logging.warning(
                                                    f"Email discovery failed for prospect: {exc}"
                                                )

                                    # Only add if we have identifying info
                                    has_identifying_info = (
                                        (contact["first_name"] and contact["last_name"])
                                        or contact["email"]
                                        or contact["job_title"]
                                    )

                                    if has_identifying_info:
                                        # Check for duplicates
                                        is_duplicate = any(
                                            c.get("email") == contact["email"]
                                            and contact["email"]
                                            or (
                                                c.get("first_name")
                                                == contact["first_name"]
                                                and c.get("last_name")
                                                == contact["last_name"]
                                                and contact["first_name"]
                                                and contact["last_name"]
                                            )
                                            for c in discovered_contacts
                                        )

                                        if not is_duplicate:
                                            discovered_contacts.append(contact)
                                            logging.info(
                                                f"✅ Discovered contact: {contact['first_name']} {contact['last_name']} - {contact['job_title']} - {contact['email']}"
                                            )

                                if discovered_contacts:
                                    logging.info(
                                        f"🎯 Successfully discovered {len(discovered_contacts)} contacts for {normalized_domain}"
                                    )
                                else:
                                    logging.warning(
                                        f"❌ No contacts discovered for {normalized_domain}"
                                    )
                            else:
                                discovered_contacts = []
                                logging.warning(
                                    f"Contact discovery failed for {normalized_domain}: {result.error}"
                                )
                        except Exception as exc:
                            discovered_contacts = []
                            logging.warning(
                                f"Contact discovery failed for {normalized_domain}: {exc}"
                            )

                        if discovered_contacts:
                            contacts = discovered_contacts
                            record["contacts"] = contacts
                            logging.info(
                                f"✅ Discovered {len(contacts)} contacts for {normalized_domain}"
                            )
                        else:
                            logging.warning(
                                f"⚠️ No contacts discovered for {normalized_domain}"
                            )
                    except Exception as contact_discovery_error:
                        # Check if it's a credit exhaustion error
                        if "402 Payment Required" in str(contact_discovery_error):
                            logging.warning(
                                f"⚠️ Snov.io API credits exhausted for {normalized_domain} - skipping contact discovery"
                            )
                        else:
                            logging.warning(
                                f"⚠️ Contact discovery failed for {normalized_domain}: {contact_discovery_error}"
                            )
                        # Continue without contacts rather than failing the entire enrichment

                # NEW: Enrich existing contacts with emails (when contacts already exist from crawling)
                if contacts and len(contacts) > 0 and contact_usd_cap > 0:
                    logging.info(
                        f"🔍 Enriching {len(contacts)} existing contacts for {normalized_domain} with email discovery"
                    )

                    # Find the best contact for ICP (prioritize by role priority and decision maker status)
                    best_contact = None
                    best_contact_score = -1

                    for contact in contacts:
                        contact_score = 0

                        # Check if contact is a decision maker
                        if contact.get("decision_maker"):
                            contact_score += 100

                        # Check role priority
                        job_title = (contact.get("job_title") or "").lower()
                        role_priority = icp_config.get("role_priority", {})

                        for role, priority in role_priority.items():
                            if role.lower() in job_title:
                                contact_score += (10 - priority) * 10  # Higher priority = higher score
                                break

                        # Check if contact has a name
                        if contact.get("first_name") and contact.get("last_name"):
                            contact_score += 10

                        # Check if contact already has email
                        if contact.get("email"):
                            contact_score -= 50  # Prefer contacts without emails

                        if contact_score > best_contact_score:
                            best_contact_score = contact_score
                            best_contact = contact

                    if best_contact and not best_contact.get("email"):
                        first_name = best_contact.get("first_name", "")
                        last_name = best_contact.get("last_name", "")
                        job_title = best_contact.get("job_title", "")

                        logging.info(
                            f"🎯 Enriching best contact for ICP: {first_name} {last_name} ({job_title}) at {normalized_domain}"
                        )

                        try:
                            # Try to discover email for the best contact
                            email_request = EnrichmentRequest(
                                domain=normalized_domain,
                                first_name=first_name,
                                last_name=last_name,
                            )
                            email_result = await enrichment_router.find_email(
                                email_request
                            )

                            if email_result.success and email_result.data:
                                # Update the existing contact with the discovered email
                                email_data = email_result.data
                                if hasattr(email_data, "__dataclass_fields__"):
                                    # It's a ContactData object
                                    discovered_email = email_data.email
                                else:
                                    # It's a dictionary
                                    discovered_email = email_data.get("email", "")

                                if discovered_email:
                                    best_contact["email"] = discovered_email
                                    best_contact["email_source"] = ["provider"]
                                    best_contact["email_status"] = ["unknown"]
                                    best_contact["email_confidence"] = 0.0

                                    logging.info(
                                        f"✅ Email enrichment successful for {first_name} {last_name}: {discovered_email}"
                                    )

                                    # Add to cost tracking
                                    contact_cost += email_result.cost

                                    # Update the record with enriched contact
                                    record["contacts"] = contacts
                                else:
                                    logging.warning(
                                        f"⚠️ Email discovery returned no email for {first_name} {last_name}"
                                    )
                            else:
                                logging.warning(
                                    f"⚠️ Email discovery failed for {first_name} {last_name}: {email_result.error}"
                                )

                        except Exception as exc:
                            logging.warning(
                                f"⚠️ Email enrichment failed for {first_name} {last_name}: {exc}"
                            )
                    elif best_contact and best_contact.get("email"):
                        logging.info(
                            f"✅ Best contact {best_contact.get('first_name')} {best_contact.get('last_name')} already has email: {best_contact.get('email')}"
                        )
                    else:
                        logging.info(
                            f"⚠️ No suitable contact found for email enrichment in {normalized_domain}"
                        )

                # Add standalone email discovery step for the best contact
                if contacts and len(contacts) > 0:
                    # Find the best contact for email discovery
                    best_contact = None
                    for contact in contacts:
                        if contact.get("first_name") and contact.get("last_name"):
                            best_contact = contact
                            break

                    if best_contact:
                        first_name = best_contact.get("first_name", "")
                        last_name = best_contact.get("last_name", "")
                        logging.info(
                            f"🔍 Attempting email discovery for best contact: {first_name} {last_name} at {normalized_domain}..."
                        )

                        try:
                            # Try to discover email for the best contact
                            email_request = EnrichmentRequest(
                                domain=normalized_domain,
                                first_name=first_name,
                                last_name=last_name,
                            )
                            email_result = await enrichment_router.find_email(
                                email_request
                            )

                            if email_result.success and email_result.data:
                                # Create a contact entry for the discovered email
                                email_data = email_result.data
                                if hasattr(email_data, "__dataclass_fields__"):
                                    # It's a ContactData object
                                    discovered_email = email_data.email
                                    email_first_name = email_data.first_name or ""
                                    email_last_name = email_data.last_name or ""
                                    job_title = email_data.job_title or ""
                                else:
                                    # It's a dictionary
                                    discovered_email = email_data.get("email", "")
                                    email_first_name = email_data.get("first_name", "")
                                    email_last_name = email_data.get("last_name", "")
                                    job_title = email_data.get("job_title", "")

                                if discovered_email:
                                    # Create a new contact entry
                                    email_contact = {
                                        "contact_id": f"email_discovery_{len(contacts)}",
                                        "first_name": email_first_name,
                                        "last_name": email_last_name,
                                        "email": discovered_email,
                                        "job_title": job_title,
                                        "company": company.get("name", ""),
                                        "discovery_source": "email_discovery",
                                        "discovery_method": "domain_email_search",
                                        "email_source": ["provider"],
                                        "email_status": ["unknown"],
                                    }

                                    # Add to contacts list
                                    contacts.append(email_contact)
                                    logging.info(
                                        f"✅ Email discovery found: {discovered_email} for {normalized_domain}"
                                    )

                                    # Add to cost tracking
                                    single_contact_cost = email_result.cost
                                    contact_cost += single_contact_cost

                        except Exception as exc:
                            logging.warning(
                                f"Standalone email discovery failed for {normalized_domain}: {exc}"
                            )

                # Enrich existing or discovered contacts
                if contacts and contact_usd_cap > 0:
                    logging.info(
                        f"🔍 About to enrich {len(contacts)} contacts for {normalized_domain}"
                    )
                    for i, contact in enumerate(contacts):
                        logging.info(
                            f"🔍 Contact {i}: type={type(contact)}, content={contact}"
                        )

                        # Check if contact is a dictionary (defensive programming)
                        if not isinstance(contact, dict):
                            logging.warning(
                                f"⚠️ Contact {i} is not a dictionary (type: {type(contact)}), skipping: {contact}"
                            )
                            continue

                        if contact_cost >= contact_usd_cap:
                            break

                        # Add company country and default country for phone parsing
                        contact["company_country"] = company.get("hq_location")
                        contact["default_country"] = default_country

                        # Use router for contact enrichment
                        single_contact_cost = 0.0

                        # Try to enrich contact via router if no email
                        if not contact.get("email"):
                            try:
                                request = EnrichmentRequest(domain=normalized_domain)
                                result = await enrichment_router.enrich_contacts(
                                    request
                                )

                                if result.success and result.data:
                                    # Update contact with enriched data
                                    enriched_contact_data = result.data

                                    # Handle both dataclass objects and dictionaries
                                    if hasattr(
                                        enriched_contact_data, "__dataclass_fields__"
                                    ):
                                        # It's a ContactData object, access attributes directly
                                        if (
                                            enriched_contact_data.email
                                            and not contact.get("email")
                                        ):
                                            contact["email"] = (
                                                enriched_contact_data.email
                                            )
                                            contact["email_source"] = ["provider"]
                                            contact["email_status"] = ["unknown"]
                                        if (
                                            enriched_contact_data.phone
                                            and not contact.get("phone")
                                        ):
                                            contact["phone"] = (
                                                enriched_contact_data.phone
                                            )
                                        if (
                                            enriched_contact_data.job_title
                                            and not contact.get("job_title")
                                        ):
                                            contact["job_title"] = (
                                                enriched_contact_data.job_title
                                            )
                                    else:
                                        # It's a dictionary, use .get() method
                                        if enriched_contact_data.get(
                                            "email"
                                        ) and not contact.get("email"):
                                            contact["email"] = enriched_contact_data[
                                                "email"
                                            ]
                                            contact["email_source"] = ["provider"]
                                            contact["email_status"] = ["unknown"]
                                        if enriched_contact_data.get(
                                            "phone"
                                        ) and not contact.get("phone"):
                                            contact["phone"] = enriched_contact_data[
                                                "phone"
                                            ]
                                        if enriched_contact_data.get(
                                            "role"
                                        ) and not contact.get("role"):
                                            contact["role"] = enriched_contact_data[
                                                "role"
                                            ]

                                    single_contact_cost += result.cost
                                    logging.info(
                                        f"✅ Contact enrichment completed for {normalized_domain}"
                                    )
                            except Exception as exc:
                                logging.warning(
                                    f"Contact enrichment failed for {normalized_domain}: {exc}"
                                )

                        # Try to find email if still missing
                        if (
                            not contact.get("email")
                            and contact.get("first_name")
                            and contact.get("last_name")
                        ):
                            try:
                                request = EnrichmentRequest(
                                    domain=normalized_domain,
                                    first_name=contact["first_name"],
                                    last_name=contact["last_name"],
                                )
                                result = await enrichment_router.find_email(request)

                                if result.success and result.data and result.data.email:
                                    contact["email"] = result.data.email
                                    contact["email_source"] = ["provider"]
                                    contact["email_status"] = ["unknown"]
                                    single_contact_cost += result.cost
                                    logging.info(
                                        f"✅ Email discovery completed for {normalized_domain}"
                                    )
                            except Exception as exc:
                                logging.warning(
                                    f"Email discovery failed for {normalized_domain}: {exc}"
                                )

                        # Validate email if present
                        if contact.get("email"):
                            try:
                                validation_result = await mailgun_client.validate_email(
                                    contact["email"]
                                )

                                # Map Mailgun status to our email_status
                                mailgun_status = validation_result.get(
                                    "result", "unknown"
                                )
                                if mailgun_status == "valid":
                                    contact["email_status"] = ["valid"]
                                    contact["email_confidence"] = validation_result.get(
                                        "score", 0.0
                                    )
                                elif mailgun_status == "catch_all":
                                    contact["email_status"] = ["catch_all"]
                                    contact["email_confidence"] = 0.5
                                elif mailgun_status == "invalid":
                                    contact["email_status"] = ["invalid"]
                                    contact["email_confidence"] = 0.0
                                else:
                                    contact["email_status"] = ["unknown"]
                                    contact["email_confidence"] = 0.5

                                single_contact_cost += costs_config.get("mailgun", 0.0)

                            except Exception as exc:
                                logging.warning(f"Failed to validate email: {exc}")
                                contact["email_status"] = ["unknown"]
                                contact["email_confidence"] = 0.0

                        # Normalize phone number if present
                        if contact.get("phone"):
                            try:
                                normalized_phone = parse_phone_number(
                                    contact["phone"],
                                    country_code=contact.get("company_country"),
                                    default_country=contact.get(
                                        "default_country", "US"
                                    ),
                                )
                                contact["phone"] = normalized_phone
                            except ValueError:
                                # Keep original phone if parsing fails
                                pass

                        # Cap individual contact cost
                        single_contact_cost = min(
                            single_contact_cost, contact_usd_cap - contact_cost
                        )
                        contact_cost += single_contact_cost

                # Update costs including LLM costs
                if "cost" not in record:
                    record["cost"] = {
                        "domain_usd": 0.0,
                        "crawl_usd": 0.0,
                        "enrich_usd": 0.0,
                        "llm_usd": 0.0,
                        "total_usd": 0.0,
                    }
                record["cost"]["enrich_usd"] = company_cost + contact_cost
                record["cost"]["llm_usd"] = llm_selection_cost
                recompute_total_cost(record)  # Update total cost
                total_cost += company_cost + contact_cost + llm_selection_cost

                # CRITICAL FIX: Update record with final merged contacts array
                # This ensures the CSV export can find the best contact
                record["contacts"] = contacts
                logging.info(f"✅ Final contacts array updated for {normalized_domain}: {len(contacts)} contacts")

                # CRITICAL FIX: Ensure best_contact_id matches an actual contact in the contacts array
                if best_contact_id and contacts:
                    # Check if best_contact_id exists in contacts array
                    contact_exists = any(c.get("contact_id") == best_contact_id for c in contacts)
                    if not contact_exists:
                        # If best_contact_id doesn't match any contact, use the first contact's ID
                        if contacts:
                            record["best_contact_id"] = contacts[0]["contact_id"]
                            logging.info(f"🔄 Updated best_contact_id to match actual contact: {contacts[0]['contact_id']} for {normalized_domain}")
                        else:
                            record["best_contact_id"] = None
                            logging.info(f"🔄 Set best_contact_id to None (no contacts) for {normalized_domain}")
                elif not best_contact_id and contacts:
                    # If no best_contact_id was set but we have contacts, use the first one
                    record["best_contact_id"] = contacts[0]["contact_id"]
                    logging.info(f"🔄 Set best_contact_id to first contact: {contacts[0]['contact_id']} for {normalized_domain}")
                elif not contacts:
                    # If no contacts, ensure best_contact_id is None
                    record["best_contact_id"] = None
                    logging.info(f"🔄 Set best_contact_id to None (no contacts) for {normalized_domain}")

                # Update provenance
                if "provenance" not in record:
                    record["provenance"] = {}

                record["provenance"]["enriched_at"] = now_z()
                record["provenance"]["tool_versions"] = record["provenance"].get(
                    "tool_versions", {}
                )
                record["provenance"]["tool_versions"]["enrich"] = {
                    "version": INTERNAL_VERSION,
                    "schema_version": schema_checksum,
                }

                # Set evidence path
                evidence_data = {
                    "domain": normalized_domain,
                    "enriched_at": now_z(),
                    "company_enrichment": company,
                    "contacts_enrichment": contacts,
                    "llm_contact_selection": {
                        "best_contact_id": best_contact_id,
                        "llm_cost_info": llm_cost_info,
                        "selection_method": "llm"
                        if llm_cost_info.get("total_calls", 0) > 0
                        else "rule_based",
                    },
                    "costs": {
                        "company_usd": company_cost,
                        "contact_usd": contact_cost,
                        "llm_usd": llm_selection_cost,
                        "total_usd": company_cost + contact_cost + llm_selection_cost,
                    },
                }

                evidence_path = write_evidence_file(
                    normalized_domain, evidence_data, data_dir
                )
                record["provenance"]["evidence_paths"] = record["provenance"].get(
                    "evidence_paths", {}
                )
                record["provenance"]["evidence_paths"]["enrich"] = evidence_path

                # Update status

                # Check if all required fields are filled
                required_fields = icp_config.get("required_fields", {})

                # Handle both old format (list) and new format (dict)
                if isinstance(required_fields, list):
                    # Old format: required_fields is a list, assume all are company fields
                    required_company_fields = required_fields
                    required_contact_fields = []
                elif isinstance(required_fields, dict):
                    # New format: required_fields is a dict with company/contact keys
                    required_company_fields = required_fields.get("company", [])
                    required_contact_fields = required_fields.get("contact", [])
                else:
                    # Fallback
                    required_company_fields = []
                    required_contact_fields = []

                company_complete = all(
                    company.get(field) for field in required_company_fields
                )

                contact_complete = True
                if required_contact_fields and contacts:
                    best_contact = next(
                        (c for c in contacts if c.get("contact_id") == best_contact_id),
                        None,
                    )
                    if best_contact:
                        contact_complete = all(
                            best_contact.get(field) for field in required_contact_fields
                        )
                    else:
                        contact_complete = False

                if company_complete and contact_complete:
                    new_status = "enriched"
                    enriched_count += 1
                else:
                    new_status = "enrich_partial"
                    partial_count += 1

                # Update status and append to history
                record["status"] = new_status
                append_status(
                    record["status_history"],
                    new_status,
                    f"Enrichment completed. Company: {company_complete}, Contact: {contact_complete}",
                )

                # Append audit entry
                append_audit(
                    record,
                    "Enrich",
                    f"Enriched company and contacts. Cost: ${company_cost + contact_cost:.4f}",
                )

                # Truncate evidence arrays
                truncate_evidence_arrays(record)

                # Validate schema
                validation_errors = validate_envelope(record, TOOL_NAME)
                if validation_errors:
                    errors.extend(validation_errors)

                processed_records.append(record)

            except Exception as exc:
                # Handle individual record errors

                # Create a more specific error message based on the actual error
                if "'list' object has no attribute 'get'" in str(exc):
                    if "402" in str(exc) or "credits" in str(exc).lower():
                        error_msg = "API credits exhausted - some enrichment features unavailable"
                    else:
                        error_msg = "API response format issue - contact discovery may be unavailable"
                elif "402" in str(exc) or "Payment Required" in str(exc):
                    if "Snov.io" in str(exc) or "contact" in str(exc).lower():
                        error_msg = (
                            "Snov.io API credits exhausted - contact discovery skipped"
                        )
                    else:
                        error_msg = "CoreSignal API credits exhausted - company enrichment limited"
                else:
                    error_msg = f"Enrichment failed: {str(exc)}"

                # Ensure record is a dictionary
                if not isinstance(record, dict):
                    logging.error(
                        f"⚠️ Record is not a dictionary: {type(record)}, content: {record}"
                    )
                    # Create a minimal error without trying to access record fields
                    error = build_error(
                        ErrorCode.UNKNOWN,
                        exc=exc,
                        tool=TOOL_NAME,
                        lead_id="unknown",
                        context={
                            "error": "Record is not a dictionary",
                            "record_type": str(type(record)),
                        },
                    )
                else:
                    error = build_error(
                        ErrorCode.UNKNOWN,
                        exc=exc,
                        tool=TOOL_NAME,
                        lead_id=record.get("lead_id"),
                        context={"domain": record.get("domain", "")},
                    )
                errors.append(error)

                # Keep the record but mark as failed
                record["status"] = "enrich_partial"

                # Ensure status_history exists and is a list
                if "status_history" not in record:
                    record["status_history"] = []
                elif not isinstance(record["status_history"], list):
                    logging.warning(
                        f"⚠️ status_history is not a list for {record.get('domain', 'unknown')}, resetting to empty list"
                    )
                    record["status_history"] = []

                try:
                    append_status(
                        record["status_history"],
                        "enrich_partial",
                        error_msg,
                    )
                except Exception as append_error:
                    logging.error(
                        f"⚠️ Failed to append status for {record.get('domain', 'unknown')}: {append_error}"
                    )
                    # Create a minimal status entry manually
                    record["status_history"].append(
                        {
                            "status": "enrich_partial",
                            "ts": now_z(),
                            "notes": f"Enrichment failed: {str(exc)} (status append also failed: {append_error})",
                        }
                    )

                processed_records.append(record)

        # Calculate metrics
        duration_ms = int((time.time() - start_time) * 1000)
        pass_rate = enriched_count / len(lead_records) if lead_records else 0.0

        metrics = build_metrics(
            count_in=len(lead_records),
            count_out=len(processed_records),
            duration_ms=duration_ms,
            cache_hit_rate=None,  # Enrich doesn't have cache hits
            pass_rate=pass_rate,
            cost_usd={
                "domain": 0.0,
                "crawl": 0.0,
                "enrich": total_cost,
                "llm": sum(
                    record.get("cost", {}).get("llm_usd", 0.0)
                    for record in processed_records
                ),
                "total": total_cost,
            },
        )

        return {
            "data": {"lead_records": processed_records},
            "errors": errors,
            "metrics": metrics,
        }

    except Exception as exc:
        # Handle tool-level errors
        error = build_error(ErrorCode.UNKNOWN, exc=exc, tool=TOOL_NAME)
        duration_ms = int((time.time() - start_time) * 1000)

        return {
            "data": {"lead_records": []},
            "errors": [error],
            "metrics": build_metrics(0, 0, duration_ms),
        }


# ============================================================================
# CLI Interface
# ============================================================================


def main():
    """CLI entry point for the Enrich tool."""
    try:
        # Read JSON from stdin
        input_data = input()
        payload = json.loads(input_data)

        # Run enrichment
        result = asyncio.run(run_enrich(payload))

        # Write JSON to stdout
        print(json.dumps(result))

    except json.JSONDecodeError as exc:
        # Handle JSON parsing errors
        error_result = {
            "data": {"lead_records": []},
            "errors": [
                build_error(ErrorCode.SCHEMA_VALIDATION, exc=exc, tool=TOOL_NAME)
            ],
            "metrics": build_metrics(0, 0, 0),
        }
        print(json.dumps(error_result))
        exit(1)

    except Exception as exc:
        # Handle other errors
        error_result = {
            "data": {"lead_records": []},
            "errors": [build_error(ErrorCode.UNKNOWN, exc=exc, tool=TOOL_NAME)],
            "metrics": build_metrics(0, 0, 0),
        }
        print(json.dumps(error_result))
        exit(1)


if __name__ == "__main__":
    main()
