"""
Common utilities for Lead Sorcerer tools.

This module provides cross-cutting functionality that is shared across all tools
while maintaining single-file isolation requirements. Each tool imports only
what it needs from this module.
"""

import asyncio
import hashlib
import json
import logging
import os
import re
import threading
import time
import unicodedata
from datetime import datetime, timedelta
from enum import Enum
from functools import wraps
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import phonenumbers

# portalocker removed - using direct file I/O for Windows compatibility
import publicsuffix2
from jsonschema import Draft202012Validator, ValidationError


# ============================================================================
# Constants and Configuration
# ============================================================================


class ErrorCode(Enum):
    """Authoritative error codes for all tools."""

    SCHEMA_VALIDATION = "SCHEMA_VALIDATION"
    HTTP_429 = "HTTP_429"
    HTTP_TIMEOUT = "HTTP_TIMEOUT"
    SMTP_TIMEOUT = "SMTP_TIMEOUT"
    DNS_FAILURE = "DNS_FAILURE"
    PROVIDER_ERROR = "PROVIDER_ERROR"
    EXTRACT_ERROR = "EXTRACT_ERROR"
    BUDGET_EXCEEDED = "BUDGET_EXCEEDED"
    PARSE_ERROR = "PARSE_ERROR"
    UNKNOWN = "UNKNOWN"
    NO_RESULTS = "NO_RESULTS"


# ============================================================================
# Data & File Conventions
# ============================================================================


def normalize_domain(domain: str) -> str:
    """
    Normalize domain to eTLD+1 using Public Suffix List.

    Args:
        domain: Raw domain string (e.g., "sub.foo.co.uk")

    Returns:
        Normalized eTLD+1 domain (e.g., "foo.co.uk")
    """
    if not domain:
        return domain

    # Remove protocol if present
    if domain.startswith(("http://", "https://")):
        domain = urlparse(domain).netloc

    # Remove port if present
    if ":" in domain:
        domain = domain.split(":")[0]

    # Normalize to lowercase
    domain = domain.lower().strip()

    # Get eTLD+1 using publicsuffix2
    try:
        ps = publicsuffix2.PublicSuffixList()
        return ps.get_public_suffix(domain)
    except Exception:
        # Fallback: return as-is if parsing fails
        return domain


def normalize_text(text: str) -> str:
    """
    Normalize text using Unicode NFKD + single-space normalization.

    Args:
        text: Raw text string

    Returns:
        Normalized text: trimmed, single-space, lowercase, NFKD folded
    """
    if text is None:
        return ""

    if not text:
        return ""

    # Convert to string if not already
    text = str(text)

    # Unicode NFKD normalization (strip accents)
    text = unicodedata.normalize("NFKD", text)

    # Convert to lowercase
    text = text.lower()

    # Trim and collapse internal whitespace to single spaces
    text = " ".join(text.split())

    # Remove combining characters (accents) after NFKD
    text = "".join(c for c in text if not unicodedata.combining(c))

    return text


def generate_lead_id(domain: str) -> str:
    """
    Generate deterministic lead_id using uuid5 with NAMESPACE_DNS.

    Args:
        domain: Normalized eTLD+1 domain

    Returns:
        Deterministic UUID5 string
    """
    import uuid

    normalized_domain = normalize_domain(domain)
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, normalized_domain))


def generate_contact_id(
    domain: str, full_name: str, role: str, linkedin: str = ""
) -> str:
    """
    Generate deterministic contact_id using uuid5 with NAMESPACE_URL.

    Args:
        domain: Normalized eTLD+1 domain
        full_name: Contact's full name
        role: Contact's role/title
        linkedin: LinkedIn URL or path (optional)

    Returns:
        Deterministic UUID5 string
    """
    import uuid

    normalized_domain = normalize_domain(domain)
    normalized_name = normalize_text(full_name)
    normalized_role = normalize_text(role)
    normalized_linkedin = linkedin or ""

    # Create deterministic string for UUID5
    contact_string = (
        f"{normalized_domain}|{normalized_name}|{normalized_role}|{normalized_linkedin}"
    )
    return str(uuid.uuid5(uuid.NAMESPACE_URL, contact_string))


def now_z() -> str:
    """
    Get current timestamp in ISO-8601 UTC format with Z suffix.

    Returns:
        ISO-8601 UTC timestamp string
    """
    return datetime.utcnow().replace(tzinfo=None).isoformat() + "Z"


def mask_email(email: str) -> str:
    """
    Mask email address for PII protection.

    Args:
        email: Raw email address

    Returns:
        Masked email (e.g., "j***@domain.com")
    """
    if not email or "@" not in email:
        return email

    local_part, domain = email.split("@", 1)
    if len(local_part) <= 1:
        return f"{local_part}***@{domain}"

    return f"{local_part[0]}***@{domain}"


def mask_phone(phone: str) -> str:
    """
    Mask phone number for PII protection.

    Args:
        phone: Raw phone number

    Returns:
        Masked phone (e.g., "+XXXXXXXX7890")
    """
    if not phone:
        return phone

    # Remove all non-digit characters except +
    digits_only = re.sub(r"[^\d+]", "", phone)

    if len(digits_only) <= 4:
        return phone

    # Keep + and last 4 digits
    if digits_only.startswith("+"):
        return f"+XXXXXXXX{digits_only[-4:]}"
    else:
        return f"XXXXXXXX{digits_only[-4:]}"


def mask_linkedin(linkedin_path: str) -> str:
    """
    Mask LinkedIn path for PII protection.

    Args:
        linkedin_path: LinkedIn URL or path

    Returns:
        Masked LinkedIn path (e.g., "/in/***" or "/company/***")
    """
    if not linkedin_path:
        return linkedin_path

    # Extract path from full URL if needed
    if linkedin_path.startswith("http"):
        parsed = urlparse(linkedin_path)
        path = parsed.path
    else:
        path = linkedin_path

    # Mask the slug part
    if path.startswith("/in/"):
        return "/in/***"
    elif path.startswith("/company/"):
        return "/company/***"
    else:
        return path


def canonicalize_linkedin(linkedin_url: str) -> Tuple[str, str]:
    """
    Canonicalize LinkedIn URL and extract type and slug.

    Args:
        linkedin_url: Raw LinkedIn URL

    Returns:
        Tuple of (type, slug) where type is "in" or "company"
    """
    if not linkedin_url:
        return "", ""

    # LinkedIn canonicalizer regex from BRD
    pattern = r"^(?:https?://)?(?:[a-z]{2,3}\.)?linkedin\.com/(?P<type>in|company)/(?P<slug>[^/?#]+)"
    match = re.match(pattern, linkedin_url)

    if match:
        link_type = match.group("type")
        slug = match.group("slug")
        return link_type, slug

    # Handle relative paths
    if linkedin_url.startswith("/in/") or linkedin_url.startswith("/company/"):
        parts = linkedin_url.split("/")
        if len(parts) >= 3:
            return parts[1], parts[2]

    return "", ""


def parse_phone_number(
    phone: str,
    country_code: Optional[str] = None,
    company_country: Optional[str] = None,
    default_country: str = "US",
) -> Optional[str]:
    """
    Parse and normalize phone number to E.164 format.

    Args:
        phone: Raw phone number string
        country_code: Explicit country code
        company_country: Country from company.hq_location
        default_country: Default country from icp_config

    Returns:
        E.164 formatted phone number or None if parsing fails

    Raises:
        ValueError: If phone number cannot be parsed
    """
    if not phone:
        return None

    # Try explicit country code first
    if country_code:
        try:
            parsed = phonenumbers.parse(phone, country_code)
            if phonenumbers.is_valid_number(parsed):
                return phonenumbers.format_number(
                    parsed, phonenumbers.PhoneNumberFormat.E164
                )
        except phonenumbers.NumberParseException:
            pass

    # Try company country
    if company_country:
        try:
            parsed = phonenumbers.parse(phone, company_country)
            if phonenumbers.is_valid_number(parsed):
                return phonenumbers.format_number(
                    parsed, phonenumbers.PhoneNumberFormat.E164
                )
        except phonenumbers.NumberParseException:
            pass

    # Try default country
    try:
        parsed = phonenumbers.parse(phone, default_country)
        if phonenumbers.is_valid_number(parsed):
            return phonenumbers.format_number(
                parsed, phonenumbers.PhoneNumberFormat.E164
            )
    except phonenumbers.NumberParseException:
        pass

    # If all attempts fail, raise error
    raise ValueError(f"Could not parse phone number: {phone}")


def resolve_data_dir(payload: Dict[str, Any]) -> str:
    """
    Resolve data directory with precedence: payload.config.data_dir → env LEADPOET_DATA_DIR → ./data.

    Args:
        payload: Tool input payload

    Returns:
        Resolved data directory path
    """
    # Check payload.config.data_dir first
    if payload.get("config", {}).get("data_dir"):
        return payload["config"]["data_dir"]

    # Check environment variable
    env_data_dir = os.environ.get("LEADPOET_DATA_DIR")
    if env_data_dir:
        return env_data_dir

    # Default to ./data
    return "./data"


def compute_next_revisit(
    existing_next_revisit: Optional[str],
    revisit_after_days: int,
    failure_revisit_days: Optional[int] = None,
    is_failure: bool = False,
) -> str:
    """
    Compute next_revisit_at using max-rule.

    Args:
        existing_next_revisit: Existing next_revisit_at timestamp or None
        revisit_after_days: Days to add for successful operations
        failure_revisit_days: Days to add for failures (defaults to 1)
        is_failure: Whether this is a failure case

    Returns:
        ISO-8601 UTC timestamp string
    """
    if failure_revisit_days is None:
        failure_revisit_days = 1

    # Determine which delta to use
    delta_days = failure_revisit_days if is_failure else revisit_after_days

    # Calculate new revisit time
    new_revisit = datetime.utcnow() + timedelta(days=delta_days)
    new_revisit_str = new_revisit.isoformat() + "Z"

    # Apply max-rule: max(existing_next_revisit_at or 0, now + delta_days)
    if existing_next_revisit:
        try:
            existing_dt = datetime.fromisoformat(
                existing_next_revisit.replace("Z", "+00:00")
            )
            # Make both timezone-aware for comparison
            new_revisit_aware = new_revisit.replace(tzinfo=existing_dt.tzinfo)
            if existing_dt > new_revisit_aware:
                return existing_next_revisit
        except ValueError:
            # If existing timestamp is invalid, use new one
            pass

    return new_revisit_str


def append_status(
    history: List[Dict[str, Any]], new_status: str, notes: Optional[str] = None
) -> None:
    """
    Append status entry to status_history array.

    Args:
        history: Existing status_history array
        new_status: New status value
        notes: Optional notes about the status change
    """
    status_entry = {"status": new_status, "ts": now_z(), "notes": notes}
    history.append(status_entry)


def append_audit(record: Dict[str, Any], step: str, notes: str) -> None:
    """
    Append audit entry to audit array.

    Args:
        record: Lead record to update
        step: Tool step (Domain|Crawl|Enrich)
        notes: Audit notes
    """
    if "audit" not in record:
        record["audit"] = []

    audit_entry = {"step": step, "notes": notes, "ts": now_z()}
    record["audit"].append(audit_entry)


def recompute_total_cost(record: Dict[str, Any]) -> None:
    """
    Recompute cost.total_usd as sum of stage costs including LLM costs.

    Args:
        record: Lead record to update
    """
    if "cost" not in record:
        record["cost"] = {}

    domain_cost = record["cost"].get("domain_usd", 0.0)
    crawl_cost = record["cost"].get("crawl_usd", 0.0)
    enrich_cost = record["cost"].get("enrich_usd", 0.0)
    llm_cost = record["cost"].get("llm_usd", 0.0)

    total = domain_cost + crawl_cost + enrich_cost + llm_cost
    record["cost"]["total_usd"] = round4(total)


def round4(f: float) -> float:
    """
    Round float to 4 decimal places.

    Args:
        f: Float to round

    Returns:
        Rounded float
    """
    return round(f, 4)


def enforce_field_ownership(
    record_before: Dict[str, Any], record_after: Dict[str, Any], stage: str
) -> None:
    """
    Validate that only authorized fields were modified.

    Args:
        record_before: Record before modification
        record_after: Record after modification
        stage: Tool stage (Domain|Crawl|Enrich)

    Raises:
        ValueError: If unauthorized fields were modified
    """
    # This is a placeholder for field ownership validation
    # Implementation will be added based on BRD field ownership matrix
    pass


def truncate_evidence_arrays(record: Dict[str, Any]) -> None:
    """
    Enforce evidence array truncation limits.

    Args:
        record: Lead record to truncate
    """
    # Company evidence_urls ≤ 10
    if "company" in record and "evidence_urls" in record["company"]:
        if len(record["company"]["evidence_urls"]) > 10:
            record["company"]["evidence_urls"] = record["company"]["evidence_urls"][:10]

    # Contacts evidence_urls ≤ 5
    if "contacts" in record:
        for contact in record["contacts"]:
            if "evidence_urls" in contact and len(contact["evidence_urls"]) > 5:
                contact["evidence_urls"] = contact["evidence_urls"][:5]


# ============================================================================
# Concurrency Guard
# ============================================================================


class PermitManager:
    """Global permit manager for controlling concurrent requests across all tools."""

    def __init__(self, max_permits: int = 3):
        self.max_permits = max_permits
        self._available_permits = max_permits
        self._active_permits = 0
        self._lock = threading.Lock()
        self._async_lock = asyncio.Lock()

    def blocking_acquire(self, timeout: Optional[float] = None) -> bool:
        """
        Acquire a permit for blocking operations (thread-safe).

        Args:
            timeout: Maximum time to wait for permit

        Returns:
            True if permit acquired, False if timeout
        """
        start_time = time.time()
        while True:
            with self._lock:
                if self._available_permits > 0:
                    self._available_permits -= 1
                    self._active_permits += 1
                    return True

            if timeout and (time.time() - start_time) > timeout:
                return False

            time.sleep(0.01)  # Small delay before retry

    def blocking_release(self) -> None:
        """Release a permit for blocking operations."""
        with self._lock:
            if self._active_permits > 0:
                self._active_permits -= 1
                self._available_permits += 1

    async def async_acquire(self) -> None:
        """Acquire a permit for async operations."""
        async with self._async_lock:
            while self._available_permits <= 0:
                await asyncio.sleep(0.01)

            with self._lock:
                self._available_permits -= 1
                self._active_permits += 1

    async def async_release(self) -> None:
        """Release a permit for async operations."""
        with self._lock:
            if self._active_permits > 0:
                self._active_permits -= 1
                self._available_permits += 1

    @property
    def active_count(self) -> int:
        """Get current active permit count."""
        with self._lock:
            return self._active_permits


class AsyncSemaphorePool:
    """Async semaphore pool using global PermitManager."""

    def __init__(self, permit_manager: PermitManager):
        self.permit_manager = permit_manager

    async def acquire(self) -> None:
        """Acquire permit for async operation."""
        await self.permit_manager.async_acquire()

    async def release(self) -> None:
        """Release permit for async operation."""
        await self.permit_manager.async_release()

    async def __aenter__(self):
        await self.acquire()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.release()


class SyncSemaphorePool:
    """Sync semaphore pool using global PermitManager."""

    def __init__(self, permit_manager: PermitManager):
        self.permit_manager = permit_manager

    def acquire(self, timeout: Optional[float] = None) -> bool:
        """Acquire permit for blocking operation."""
        return self.permit_manager.blocking_acquire(timeout)

    def release(self) -> None:
        """Release permit for blocking operation."""
        self.permit_manager.blocking_release()

    def __enter__(self):
        if not self.acquire():
            raise RuntimeError("Failed to acquire permit")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()


# ============================================================================
# Error Handling & Retries
# ============================================================================


def build_error(
    code: ErrorCode,
    exc: Optional[Exception] = None,
    tool: str = "",
    lead_id: Optional[str] = None,
    context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Build error object with PII masking and secret redaction.

    Args:
        code: Error code from ErrorCode enum
        exc: Exception that caused the error
        tool: Tool name
        lead_id: Lead ID if available
        context: Additional context

    Returns:
        Error object conforming to schema
    """
    error = {
        "code": code.value,
        "tool": tool,
        "retryable": code
        in [
            ErrorCode.HTTP_429,
            ErrorCode.HTTP_TIMEOUT,
            ErrorCode.SMTP_TIMEOUT,
            ErrorCode.PROVIDER_ERROR,
        ],
    }

    if lead_id:
        error["lead_id"] = lead_id

    if exc:
        # Truncate exception message to prevent overly long error messages
        message = str(exc)
        if len(message) > 200:
            message = message[:197] + "..."
        error["message"] = message
    else:
        error["message"] = f"Error: {code.value}"

    if context:
        # Mask any PII in context
        masked_context = {}
        for key, value in context.items():
            if isinstance(value, str):
                if "email" in key.lower():
                    masked_context[key] = mask_email(value)
                elif "phone" in key.lower():
                    masked_context[key] = mask_phone(value)
                elif "linkedin" in key.lower():
                    masked_context[key] = mask_linkedin(value)
                else:
                    masked_context[key] = value
            else:
                masked_context[key] = value
        error["context"] = masked_context

    return error


def wrap_unknown_errors(tool_name=None):
    """
    Decorator to wrap uncaught exceptions as UNKNOWN errors.

    Args:
        tool_name: Optional tool name, defaults to function module

    Returns:
        Decorator function
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as exc:
                # Create UNKNOWN error
                error = build_error(
                    code=ErrorCode.UNKNOWN,
                    exc=exc,
                    tool=tool_name or func.__module__ or "unknown",
                )

                # Return partial results with error
                return {
                    "data": {"lead_records": []},
                    "errors": [error],
                    "metrics": {
                        "count_in": 0,
                        "count_out": 0,
                        "duration_ms": 0,
                        "cache_hit_rate": None,
                        "pass_rate": None,
                        "cost_usd": {
                            "domain": 0.0,
                            "crawl": 0.0,
                            "enrich": 0.0,
                            "total": 0.0,
                        },
                    },
                }

        return wrapper

    return decorator


def exponential_backoff_with_timeout(
    max_attempts: int = 5, base_delay: float = 1.0, max_wall_clock: float = 45.0
):
    """
    Decorator for exponential backoff with wall-clock timeout.

    Args:
        max_attempts: Maximum number of attempts
        base_delay: Base delay in seconds
        max_wall_clock: Maximum total time in seconds

    Returns:
        Decorated function
    """

    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            attempt = 0

            while attempt < max_attempts:
                try:
                    return await func(*args, **kwargs)
                except Exception as exc:
                    attempt += 1

                    # Check wall-clock timeout
                    if time.time() - start_time > max_wall_clock:
                        raise TimeoutError(
                            f"Wall-clock timeout after {max_wall_clock}s"
                        )

                    # Don't sleep on last attempt
                    if attempt < max_attempts:
                        delay = min(base_delay * (2 ** (attempt - 1)), 16.0)
                        await asyncio.sleep(delay)

                    # Re-raise on last attempt
                    if attempt >= max_attempts:
                        raise exc

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            attempt = 0

            while attempt < max_attempts:
                try:
                    return func(*args, **kwargs)
                except Exception as exc:
                    attempt += 1

                    # Check wall-clock timeout
                    if time.time() - start_time > max_wall_clock:
                        raise TimeoutError(
                            f"Wall-clock timeout after {max_wall_clock}s"
                        )

                    # Don't sleep on last attempt
                    if attempt < max_attempts:
                        delay = min(base_delay * (2 ** (attempt - 1)), 16.0)
                        time.sleep(delay)

                    # Re-raise on last attempt
                    if attempt >= max_attempts:
                        raise exc

        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


# ============================================================================
# Schema Validation & Versioning
# ============================================================================


def load_schema_checksum() -> str:
    """
    Load canonical schema and compute SHA256 checksum.

    Returns:
        SHA256 hex digest of schema file
    """
    schema_path = Path("schemas/unified_lead_record.json")
    if not schema_path.exists():
        raise FileNotFoundError(f"Schema file not found: {schema_path}")

    with open(schema_path, "rb") as f:
        schema_content = f.read()

    return hashlib.sha256(schema_content).hexdigest()


def validate_envelope(io_data: Dict[str, Any], stage: str) -> List[Dict[str, Any]]:
    """
    Validate I/O data against canonical schema.

    Args:
        io_data: Data to validate
        stage: Tool stage for error reporting

    Returns:
        List of validation errors (empty if valid)
    """
    errors = []

    try:
        # Load schema
        schema_path = Path("schemas/unified_lead_record.json")
        with open(schema_path, "r") as f:
            schema = json.load(f)

        # Validate against schema
        validator = Draft202012Validator(schema)
        validation_errors = list(validator.iter_errors(io_data))

        for error in validation_errors:
            errors.append(
                build_error(
                    code=ErrorCode.SCHEMA_VALIDATION,
                    exc=ValidationError(f"Schema validation failed: {error.message}"),
                    tool=stage,
                    context={
                        "path": list(error.path),
                        "schema_path": str(error.schema_path),
                    },
                )
            )

    except Exception as exc:
        errors.append(
            build_error(
                code=ErrorCode.SCHEMA_VALIDATION,
                exc=exc,
                tool=stage,
                context={"error": "Failed to load or validate schema"},
            )
        )

    return errors


def validate_template_placeholders(template: str, context: Dict[str, Any]) -> List[str]:
    """
    Validate that all template placeholders are satisfied.

    Args:
        template: Template string with ${placeholder} syntax
        context: Context dictionary for substitution

    Returns:
        List of missing placeholder names
    """
    import string

    # Extract all placeholders from template
    string.Template(template)
    placeholders = re.findall(r"\$\{([^}]+)\}", template)

    # Check which placeholders are missing from context
    missing = []
    for placeholder in placeholders:
        if placeholder not in context:
            missing.append(placeholder)

    return missing


# ============================================================================
# Metrics
# ============================================================================


def build_metrics(
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
            "domain": round4(cost_usd.get("domain", 0.0)) if cost_usd else 0.0,
            "crawl": round4(cost_usd.get("crawl", 0.0)) if cost_usd else 0.0,
            "enrich": round4(cost_usd.get("enrich", 0.0)) if cost_usd else 0.0,
            "llm": round4(cost_usd.get("llm", 0.0)) if cost_usd else 0.0,
            "total": round4(cost_usd.get("total", 0.0)) if cost_usd else 0.0,
        },
    }

    return metrics


def load_costs_config(path: str = "config/costs.yaml") -> Dict[str, float]:
    """
    Load costs configuration from YAML file.

    Args:
        path: Path to costs.yaml file

    Returns:
        Dictionary mapping provider names to USD per unit

    Raises:
        FileNotFoundError: If costs file not found
        KeyError: If required provider is missing
    """
    try:
        import yaml

        with open(path, "r") as f:
            costs = yaml.safe_load(f)

        # Extract usd_per_unit for each provider
        result = {}
        for provider, config in costs.items():
            if "usd_per_unit" in config:
                result[provider] = float(config["usd_per_unit"])

        return result

    except ImportError:
        raise ImportError("PyYAML is required to load costs configuration")
    except FileNotFoundError:
        raise FileNotFoundError(f"Costs configuration file not found: {path}")
    except Exception as exc:
        raise RuntimeError(f"Failed to load costs configuration: {exc}")


# ============================================================================
# Logging & Secret Redaction
# ============================================================================


class PIIMaskingFormatter(logging.Formatter):
    """Log formatter that masks PII and secrets."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.secret_keys = {"key", "token", "secret", "password", "pw"}

    def format(self, record):
        # Get the original message
        msg = record.getMessage()

        # Try to parse as JSON for structured logging
        try:
            log_data = json.loads(msg)
            masked_data = self._mask_json(log_data)
            record.msg = json.dumps(masked_data)
        except (json.JSONDecodeError, TypeError):
            # Not JSON, mask as plain text
            record.msg = self._mask_text(msg)

        return super().format(record)

    def _mask_json(self, data):
        """Mask PII and secrets in JSON data."""
        if isinstance(data, dict):
            masked = {}
            for key, value in data.items():
                if isinstance(value, dict):
                    masked[key] = self._mask_json(value)
                elif isinstance(value, list):
                    masked[key] = [self._mask_json(item) for item in value]
                elif isinstance(value, str):
                    masked[key] = self._mask_value(key, value)
                else:
                    masked[key] = value
            return masked
        elif isinstance(data, list):
            return [self._mask_json(item) for item in data]
        else:
            return data

    def _mask_value(self, key: str, value: str) -> str:
        """Mask individual values based on key and content."""
        key_lower = key.lower()

        # Mask secrets
        if any(secret_key in key_lower for secret_key in self.secret_keys):
            return "***REDACTED***"

        # Mask emails
        if "email" in key_lower and "@" in value:
            return mask_email(value)

        # Mask phones
        if "phone" in key_lower and any(c.isdigit() for c in value):
            return mask_phone(value)

        # Mask LinkedIn URLs
        if "linkedin" in key_lower and "linkedin.com" in value:
            return mask_linkedin(value)

        # Mask large payloads
        if len(value) > 2048:
            return f"{value[:100]}...***TRUNCATED***"

        # Mask provider payloads with headers/body
        if key in ["headers", "body"] or "payload" in key_lower:
            return "***PROVIDER_PAYLOAD_REDACTED***"

        return value

    def _mask_text(self, text: str) -> str:
        """Mask PII in plain text."""
        # This is a simplified text masker
        # In practice, you might want more sophisticated pattern matching
        return text


def setup_logging(
    tool_name: str,
    level: int = logging.INFO,
    data_dir: Optional[str] = None,
    log_file_path: Optional[str] = None,
) -> logging.Logger:
    """
    Setup logging with PII masking and JSON formatting.

    Creates both console and file handlers. File logs are stored in data/logs folder.

    Args:
        tool_name: Name of the tool
        level: Logging level
        data_dir: Optional data directory path for log files

    Returns:
        Configured logger
    """
    logger = logging.getLogger(tool_name)
    logger.setLevel(level)

    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Create console handler (stderr)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)

    # Create formatter
    formatter = PIIMaskingFormatter(
        '{"ts": "%(asctime)s", "level": "%(levelname)s", "tool": "%(name)s", "msg": "%(message)s"}'
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Determine shared run log file from explicit arg or env
    if not log_file_path:
        log_file_path = os.environ.get("LEADSORCERER_LOG_FILE")

    logs_dir = None

    # Create file handler if either a log file was provided or a data_dir exists
    if log_file_path or data_dir:
        try:
            if not log_file_path:
                # Fallback to per-tool file naming if shared path not provided
                logs_dir = os.path.join(data_dir, "logs") if data_dir else None
                if logs_dir:
                    os.makedirs(logs_dir, exist_ok=True)
                    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M")
                    log_file_path = os.path.join(
                        logs_dir, f"{tool_name}_{timestamp}.log"
                    )
            else:
                # Ensure parent directory exists
                logs_dir = os.path.dirname(log_file_path) or "."
                os.makedirs(logs_dir, exist_ok=True)

            # Create file handler
            file_handler = logging.FileHandler(log_file_path, encoding="utf-8")
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

            # Log that file logging is enabled
            logger.info(f"File logging enabled: {log_file_path}")

        except Exception as e:
            # If file logging fails, just log to console and continue
            logger.warning(f"Failed to setup file logging: {e}")

    logger.propagate = False

    # Clean up old log files if file logging is enabled
    if logs_dir:
        cleanup_old_logs(logs_dir, max_log_files=10)

    return logger


def cleanup_old_logs(logs_dir: str, max_log_files: int = 10) -> None:
    """
    Clean up old log files, keeping only the most recent ones.

    Args:
        logs_dir: Directory containing log files
        max_log_files: Maximum number of log files to keep
    """
    try:
        if not os.path.exists(logs_dir):
            return

        # Get all log files with their modification times
        log_files = []
        for filename in os.listdir(logs_dir):
            if filename.endswith(".log"):
                filepath = os.path.join(logs_dir, filename)
                mtime = os.path.getmtime(filepath)
                log_files.append((filepath, mtime))

        # Sort by modification time (newest first)
        log_files.sort(key=lambda x: x[1], reverse=True)

        # Remove old files beyond the limit
        for filepath, _ in log_files[max_log_files:]:
            try:
                os.remove(filepath)
            except OSError:
                pass  # Ignore errors when removing files

    except Exception:
        pass  # Ignore cleanup errors


# ============================================================================
# File Operations with Portallocker
# ============================================================================


def write_with_lock(file_path: str, data: Any, mode: str = "w") -> None:
    """
    Write data to file with simple error handling.

    Args:
        file_path: Path to file
        data: Data to write
        mode: File write mode
    """
    try:
        # Ensure directory exists
        import os

        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, mode) as f:
            if isinstance(data, str):
                f.write(data)
            else:
                json.dump(data, f, indent=2)
    except Exception as e:
        raise RuntimeError(f"Failed to write to {file_path}: {e}")


def append_jsonl_with_lock(file_path: str, records: List[Dict[str, Any]]) -> None:
    """
    Append records to JSONL file with simple error handling.

    Args:
        file_path: Path to JSONL file
        records: List of records to append
    """
    try:
        # Ensure directory exists
        import os

        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, "a") as f:
            for record in records:
                f.write(json.dumps(record) + "\n")
    except Exception as e:
        raise RuntimeError(f"Failed to append to {file_path}: {e}")


# ============================================================================
# Content Hash Deduplication
# ============================================================================


def compute_content_hash(data: Dict[str, Any]) -> str:
    """
    Compute content hash for deduplication.

    Args:
        data: Data to hash

    Returns:
        SHA256 hex digest
    """
    # Sort keys for deterministic hashing
    sorted_json = json.dumps(data, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(sorted_json.encode()).hexdigest()


# ============================================================================
# Progressive SERP Pagination
# ============================================================================


def should_fetch_next_page(
    current_page: int, max_pages: int, cache_hit: bool, mode: str = "thorough"
) -> bool:
    """
    Determine if next SERP page should be fetched.

    Args:
        current_page: Current page number (1-based)
        max_pages: Maximum pages to fetch
        cache_hit: Whether current results were from cache
        mode: Search mode (fast/thorough)

    Returns:
        True if next page should be fetched
    """
    if mode == "fast":
        return False  # Fast mode only fetches first page

    if current_page >= max_pages:
        return False

    if cache_hit:
        return False  # Don't fetch next page if current was cached

    return True


# ============================================================================
# Provider Configuration Validation
# ============================================================================


def validate_provider_config(
    required_providers: List[str], costs_config: Dict[str, float]
) -> None:
    """
    Validate that required providers are present in costs configuration.

    Args:
        required_providers: List of required provider names
        costs_config: Loaded costs configuration

    Raises:
        KeyError: If any required provider is missing
    """
    missing_providers = []
    for provider in required_providers:
        if provider not in costs_config:
            missing_providers.append(provider)

    if missing_providers:
        raise KeyError(
            f"Missing required providers in costs configuration: {missing_providers}"
        )


# ============================================================================
# Testing Flags & Configuration
# ============================================================================


def validate_testing_flags(icp_config: Dict[str, Any]) -> None:
    """
    Validate testing flags in ICP configuration.

    Args:
        icp_config: ICP configuration dictionary

    Raises:
        ValueError: If testing flags are invalid
    """
    testing = icp_config.get("testing", {})

    # Validate process_rejected flag
    if "process_rejected" in testing and not isinstance(
        testing["process_rejected"], bool
    ):
        raise ValueError("testing.process_rejected must be boolean")

    # Validate process_low_crawl flag
    if "process_low_crawl" in testing and not isinstance(
        testing["process_low_crawl"], bool
    ):
        raise ValueError("testing.process_low_crawl must be boolean")


# ============================================================================
# Role Priority & Seniority
# ============================================================================


def normalize_role_priority(role_priority_config: Dict[str, Any]) -> Dict[str, int]:
    """
    Normalize role_priority configuration by converting keys to lowercase and values to integers.

    Args:
        role_priority_config: Raw role priority configuration from config file

    Returns:
        Normalized role priority configuration with lowercase keys and integer values
    """
    if not isinstance(role_priority_config, dict):
        return {}

    normalized = {}
    for key, value in role_priority_config.items():
        # Normalize key: strip whitespace and convert to lowercase
        normalized_key = str(key).strip().lower()

        # Normalize value: convert to integer
        try:
            normalized_value = int(value)
        except (ValueError, TypeError):
            # If conversion fails, use default priority
            normalized_value = 99

        normalized[normalized_key] = normalized_value

    return normalized


def get_role_priority(role: str, role_priority_config: Dict[str, int]) -> int:
    """
    Get role priority from configuration.

    Args:
        role: Role/title string
        role_priority_config: Role priority configuration from icp_config

    Returns:
        Priority value (lower = higher priority)
    """
    if not role:
        return role_priority_config.get("default", 99)

    role_lower = normalize_text(role)

    # Check exact matches first
    if role_lower in role_priority_config:
        return role_priority_config[role_lower]

    # Check partial matches
    for config_role, priority in role_priority_config.items():
        if config_role != "default" and config_role in role_lower:
            return priority

    return role_priority_config.get("default", 99)


def get_seniority_rank(title: str) -> int:
    """
    Get seniority rank from title.

    Args:
        title: Job title string

    Returns:
        Seniority rank (lower = higher seniority)
    """
    if not title:
        return 99

    title_lower = normalize_text(title)

    # Seniority ranking mapping from BRD
    seniority_map = {
        "c-level": 1,
        "ceo": 1,
        "cto": 1,
        "cfo": 1,
        "coo": 1,
        "vp": 2,
        "head": 3,
        "director": 3,
        "manager": 4,
        "lead": 5,
        "ic": 6,
    }

    # Check for matches
    for pattern, rank in seniority_map.items():
        if pattern in title_lower:
            return rank

    return 99  # other/unknown


# ============================================================================
# Email Status Ranking
# ============================================================================


def get_email_status_rank(email_status: str) -> int:
    """
    Get email status rank for best contact selection.

    Args:
        email_status: Email status string

    Returns:
        Status rank (lower = higher priority)
    """
    status_ranks = {"valid": 0, "risky": 1, "catch_all": 2, "unknown": 3, "invalid": 4}

    return status_ranks.get(email_status, 4)
