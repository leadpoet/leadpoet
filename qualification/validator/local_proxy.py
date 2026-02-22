"""
Local Validator Proxy for Qualification Model Evaluation

This proxy runs in a SEPARATE THREAD within the validator process.
It provides the same API key injection as the gateway proxy, but:
1. Runs locally (no network hop to gateway)
2. Keys stored in closure variables (not os.environ)
3. Model's thread cannot access proxy thread's local variables

SECURITY MODEL:
- Proxy thread starts BEFORE sandbox activates
- Proxy stores API keys in thread-local closure
- Sandbox sanitizes os.environ (removes keys)
- Model runs (no keys in os.environ, no access to proxy thread's stack)
- gc module is blocked (can't scan memory for key strings)
- inspect.stack() only shows current thread (can't see proxy thread)

COST TRACKING:
- Proxy intercepts ALL API responses
- For OpenRouter: extracts actual cost from response (usage.total_cost or tokens)
- Accumulates costs per provider
- Validator queries /cost endpoint to get total after evaluation

This is IDENTICAL security to the gateway proxy approach.
"""

import os
import re
import json
import logging
import threading
import socket
from typing import Optional, Dict, Any, List, Tuple
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
from contextlib import contextmanager
from functools import partial
from dataclasses import dataclass, field

import httpx

logger = logging.getLogger(__name__)


# =============================================================================
# Cost Tracking (Thread-Safe)
# =============================================================================

@dataclass
class CostTracker:
    """
    Thread-safe cost accumulator for API calls.
    
    Tracks:
    - Total cost in USD
    - Call counts per provider
    - Token usage (for LLMs)
    """
    _lock: threading.Lock = field(default_factory=threading.Lock)
    total_cost_usd: float = 0.0
    call_counts: Dict[str, int] = field(default_factory=dict)
    token_counts: Dict[str, Dict[str, int]] = field(default_factory=dict)  # {provider: {input: X, output: Y}}
    
    def add_cost(self, provider: str, cost_usd: float, input_tokens: int = 0, output_tokens: int = 0):
        """Add cost from an API call (thread-safe)."""
        with self._lock:
            self.total_cost_usd += cost_usd
            self.call_counts[provider] = self.call_counts.get(provider, 0) + 1
            
            if input_tokens or output_tokens:
                if provider not in self.token_counts:
                    self.token_counts[provider] = {"input": 0, "output": 0}
                self.token_counts[provider]["input"] += input_tokens
                self.token_counts[provider]["output"] += output_tokens
    
    def get_summary(self) -> Dict[str, Any]:
        """Get cost summary (thread-safe)."""
        with self._lock:
            return {
                "total_cost_usd": round(self.total_cost_usd, 6),
                "call_counts": dict(self.call_counts),
                "token_counts": dict(self.token_counts),
            }
    
    def reset(self):
        """Reset all tracking (thread-safe)."""
        with self._lock:
            self.total_cost_usd = 0.0
            self.call_counts.clear()
            self.token_counts.clear()


def extract_openrouter_cost(usage: Dict[str, Any]) -> float:
    """
    Extract cost from OpenRouter usage object.
    
    OpenRouter ALWAYS returns cost in the response body under usage.cost.
    Example response:
    {
        "usage": {
            "prompt_tokens": 15,
            "completion_tokens": 2,
            "total_tokens": 17,
            "cost": 3.45e-06,  ← THIS IS THE ACTUAL COST
            "cost_details": {
                "upstream_inference_cost": 3.45e-06,
                "upstream_inference_prompt_cost": 2.25e-06,
                "upstream_inference_completions_cost": 1.2e-06
            }
        }
    }
    
    Args:
        usage: The usage object from OpenRouter response
    
    Returns:
        Cost in USD (always returns actual cost, never estimates)
    
    Raises:
        ValueError: If cost not found in response (should never happen with OpenRouter)
    """
    # OpenRouter returns cost in usage.cost (NOT usage.total_cost)
    cost = usage.get("cost")
    
    if cost is not None:
        return float(cost)
    
    # Fallback: check cost_details.upstream_inference_cost
    cost_details = usage.get("cost_details", {})
    upstream_cost = cost_details.get("upstream_inference_cost")
    if upstream_cost is not None:
        return float(upstream_cost)
    
    # This should NEVER happen - OpenRouter always returns cost
    raise ValueError(
        f"OpenRouter response missing cost field. Usage: {usage}. "
        "This is unexpected - OpenRouter should always include usage.cost"
    )


# =============================================================================
# ScrapingDog Credit-Based Cost Calculation
# =============================================================================
# Credit costs per endpoint (from ScrapingDog documentation)
# See: https://docs.scrapingdog.com/
#
# The cost in USD is calculated as:
#   cost_usd = credits_used × (plan_cost_usd / plan_credits_total)
#
# Default plan: $2,000/month for 42,000,000 credits
# Cost per credit: $0.0000476
# =============================================================================

# Base credit costs by endpoint
SCRAPINGDOG_ENDPOINT_CREDITS = {
    # Web Scraping API (/scrape) - costs vary by parameters
    "scrape": {
        "base": 1,                    # GWS with rotating proxy
        "premium": 10,                # GWS with premium proxy
        "dynamic": 5,                 # GWS with JS rendering + rotating
        "dynamic_premium": 25,        # GWS with JS rendering + premium
        "country": 10,                # GWS with country
        "country_premium": 10,        # GWS with country + premium
    },
    
    # Google APIs
    "google": {"base": 5, "advanced": 10},           # Google Search (light=5, advanced=10)
    "google_ai_mode": {"base": 10},                  # Google AI Mode API
    "google_ai_overview": {"base": 10},              # Google AI Overview API
    "google_maps": {"base": 5},                      # Google Maps API
    "google_trends": {"base": 5},                    # Google Trends API
    "google_images": {"base": 10},                   # Google Images API
    "google_news": {"base": 5},                      # Google News API
    "google_shopping": {"base": 10},                 # Google Shopping API
    "google_product": {"base": 5},                   # Google Product API
    "google_immersive_product": {"base": 5},         # Google Immersive Product API
    "google_scholar": {"base": 5},                   # Google Scholar API
    "google_finance": {"base": 5},                   # Google Finance API
    "google_jobs": {"base": 5},                      # Google Jobs API
    "google_hotels": {"base": 5},                    # Google Hotels API
    "google_patents": {"base": 5},                   # Google Patents API
    "google_videos": {"base": 5},                    # Google Videos API
    "google_shorts": {"base": 5},                    # Google Shorts API
    "google_autocomplete": {"base": 5},              # Google Autocomplete API
    "google_lens": {"base": 5},                      # Google Lens API
    "google_local": {"base": 5},                     # Google Local API
    "google_serp": {"base": 5, "advanced": 10},      # Google SERP API
    
    # LinkedIn APIs
    "linkedin": {"base": 50, "private": 100},        # LinkedIn Profile (50 normal, 100 private)
    "linkedin_jobs": {"base": 5},                    # LinkedIn Jobs API
    
    # Amazon APIs
    "amazon": {"base": 1},                           # Amazon Search API
    "amazon_product": {"base": 1},                   # Amazon Product API
    "amazon_reviews": {"base": 100},                 # Amazon Reviews API
    
    # YouTube APIs
    "youtube": {"base": 5},                          # YouTube Scraper API
    "youtube_transcript": {"base": 1},               # YouTube Transcript API
    "youtube_channel": {"base": 5},                  # YouTube Channel API
    "youtube_comment": {"base": 5},                  # YouTube Comment API
    
    # Other Search Engines
    "bing": {"base": 5},                             # Bing Scraper API
    "duckduckgo": {"base": 5},                       # DuckDuckGo Scraper API
    "baidu": {"base": 5},                            # Baidu Scraper API
    "universal": {"base": 20},                       # Universal Search API
    
    # Job Sites
    "indeed": {"base": 1},                           # Indeed Scraper API
    
    # Social Media
    "twitter": {"base": 5},                          # Twitter/X Scraper API
    "x": {"base": 5},                                # X Scraper API (alias)
    
    # E-commerce
    "walmart": {"base": 5},                          # Walmart Scraper API
    "ebay": {"base": 5},                             # eBay Scraper API
    "flipkart": {"base": 5},                         # Flipkart Scraper API
    "myntra": {"base": 5},                           # Myntra Scraper API
    
    # Real Estate
    "zillow": {"base": 5},                           # Zillow Scraper API
    "yelp": {"base": 1},                             # Yelp Scraper API
    
    # Utilities
    "screenshot": {"base": 5},                       # Screenshot API
}

# Default plan values (can be overridden from config)
DEFAULT_SCRAPINGDOG_PLAN_COST_USD = 2000.00
DEFAULT_SCRAPINGDOG_PLAN_CREDITS = 42_000_000


def calculate_scrapingdog_credits(
    endpoint: str,
    query_params: Dict[str, str],
    method: str = "GET"
) -> Tuple[int, str]:
    """
    Calculate credits used for a ScrapingDog API call.
    
    Args:
        endpoint: The API endpoint (e.g., "scrape", "google", "linkedin")
        query_params: Query string parameters (e.g., {"dynamic": "true", "premium": "true"})
        method: HTTP method (GET or POST)
    
    Returns:
        Tuple of (credits_used, description)
    
    Example:
        >>> calculate_scrapingdog_credits("scrape", {"dynamic": "true"})
        (5, "scrape (JS rendering)")
        
        >>> calculate_scrapingdog_credits("linkedin", {"private": "true"})
        (100, "linkedin (private)")
    """
    # Normalize endpoint (remove leading slashes, lowercase)
    endpoint = endpoint.strip("/").lower().split("/")[0] if endpoint else "scrape"
    
    # Handle endpoint aliases
    endpoint_aliases = {
        "jobs": "google_jobs" if "google" in endpoint else "linkedin_jobs",
        "search": "google",
        "profile": "linkedin",
    }
    
    # Get credits config for endpoint
    if endpoint in SCRAPINGDOG_ENDPOINT_CREDITS:
        credits_config = SCRAPINGDOG_ENDPOINT_CREDITS[endpoint]
    else:
        # Try to match google_* or similar prefixed endpoints
        for known_endpoint in SCRAPINGDOG_ENDPOINT_CREDITS:
            if endpoint.startswith(known_endpoint) or known_endpoint.startswith(endpoint):
                credits_config = SCRAPINGDOG_ENDPOINT_CREDITS[known_endpoint]
                break
        else:
            # Default: assume 5 credits for unknown endpoints
            return (5, f"{endpoint} (unknown, default)")
    
    # Get base credits
    base_credits = credits_config.get("base", 5)
    description = endpoint
    
    # Check for modifiers that increase credits
    if endpoint == "scrape":
        # Web scraping API has complex pricing based on parameters
        # Note: ScrapingDog defaults dynamic=true, but we count credits based on explicit param
        dynamic = query_params.get("dynamic", "false").lower() == "true"
        premium = query_params.get("premium", "false").lower() == "true"
        country = query_params.get("country", "")
        
        if dynamic and premium:
            return (25, "scrape (JS + premium)")
        elif premium:
            return (10, "scrape (premium)")
        elif dynamic:
            return (5, "scrape (JS rendering)")
        elif country:
            return (10, "scrape (with country)")
        else:
            return (1, "scrape (basic)")
    
    elif endpoint == "google":
        # Google Search: 5 for light, 10 for advanced
        advanced = query_params.get("advance_search", "false").lower() == "true"
        mobile = query_params.get("mob_search", "false").lower() == "true"
        if advanced or mobile:
            return (10, "google (advanced)")
        return (5, "google (light)")
    
    elif endpoint == "linkedin":
        # LinkedIn: 50 normal, 100 for private profiles
        private = query_params.get("private", "false").lower() == "true"
        if private:
            return (100, "linkedin (private)")
        return (50, "linkedin")
    
    elif endpoint == "google_serp":
        # SERP API: 5 or 10 based on features
        advanced = query_params.get("advance_search", "false").lower() == "true"
        if advanced:
            return (10, "google_serp (advanced)")
        return (5, "google_serp")
    
    # Return base credits for other endpoints
    return (base_credits, description)


def calculate_scrapingdog_cost_usd(
    credits: int,
    plan_cost_usd: float = DEFAULT_SCRAPINGDOG_PLAN_COST_USD,
    plan_credits: int = DEFAULT_SCRAPINGDOG_PLAN_CREDITS
) -> float:
    """
    Convert ScrapingDog credits to USD cost.
    
    Args:
        credits: Number of credits used
        plan_cost_usd: Monthly plan cost in USD (default: $2000)
        plan_credits: Total credits in plan (default: 42,000,000)
    
    Returns:
        Cost in USD
    
    Example:
        >>> calculate_scrapingdog_cost_usd(50)  # LinkedIn profile
        0.00238  # ~$0.002 per profile scrape
    """
    if plan_credits <= 0:
        return 0.0
    cost_per_credit = plan_cost_usd / plan_credits
    return credits * cost_per_credit


# =============================================================================
# Provider Configuration (Same as gateway/qualification/proxy/routes.py)
# =============================================================================

PAID_PROVIDERS: Dict[str, str] = {
    "openrouter": "https://openrouter.ai/api/v1",
    "scrapingdog": "https://api.scrapingdog.com",
    "builtwith": "https://api.builtwith.com",
    "crunchbase": "https://api.crunchbase.com",
    "desearch": "https://api.desearch.ai",
    "datauniverse": "https://datauniverse.macrocosmos.ai",
    "googlenews": "https://newsapi.org/v2",
    "jobdata": "https://api.theirstack.com",
}

FREE_PROVIDERS: Dict[str, str] = {
    "github": "https://api.github.com",
    "stackoverflow": "https://api.stackexchange.com",
}

ALL_PROVIDERS: Dict[str, str] = {**PAID_PROVIDERS, **FREE_PROVIDERS}

PROXY_TIMEOUT_SECONDS = 30.0

# =============================================================================
# GitHub API Endpoint Allowlist (Security)
# =============================================================================
# STRICT ALLOWLIST: Only these read-only endpoints are permitted.
# Everything else is blocked with 403. This prevents:
#   - Data exfiltration (POST /gists)
#   - Code download + exec (GET /repos/.../contents/, /repos/.../git/)
#   - Account info leakage (GET /user)
#   - Any write operations (POST/PUT/PATCH/DELETE)
#
# Patterns match the endpoint path AFTER the provider prefix is stripped.
# e.g. proxy_url/github/search/repositories -> endpoint = "search/repositories"
# =============================================================================

GITHUB_ALLOWED_ENDPOINTS: List[str] = [
    # Search APIs (high value for lead qualification)
    r"^search/repositories$",
    r"^search/users$",
    r"^search/code$",
    r"^search/issues$",
    # Organization endpoints
    r"^orgs/[^/]+$",
    r"^orgs/[^/]+/repos$",
    # User public profiles (NOT /user which is the authenticated user)
    r"^users/[^/]+$",
    r"^users/[^/]+/repos$",
    r"^users/[^/]+/orgs$",
    # Repository metadata (read-only, explicitly excludes /contents/ and /git/)
    r"^repos/[^/]+/[^/]+$",
    r"^repos/[^/]+/[^/]+/topics$",
    r"^repos/[^/]+/[^/]+/languages$",
    r"^repos/[^/]+/[^/]+/contributors$",
    r"^repos/[^/]+/[^/]+/commits$",
    r"^repos/[^/]+/[^/]+/releases$",
    r"^repos/[^/]+/[^/]+/issues$",
    r"^repos/[^/]+/[^/]+/pulls$",
    r"^repos/[^/]+/[^/]+/events$",
]

_GITHUB_COMPILED_PATTERNS = [re.compile(p) for p in GITHUB_ALLOWED_ENDPOINTS]


# =============================================================================
# Request Handler (runs in proxy thread)
# =============================================================================

class LocalProxyHandler(BaseHTTPRequestHandler):
    """
    HTTP request handler for local proxy.
    
    CRITICAL: This runs in a SEPARATE THREAD from the model.
    The api_keys dict is stored in a closure (self.api_keys) that 
    the model's thread CANNOT access via inspect.stack().
    """
    
    # Set by the factory function - these are closure variables
    api_keys: Dict[str, str] = {}
    cost_tracker: Optional[CostTracker] = None
    
    def log_message(self, format: str, *args) -> None:
        """Suppress default logging to prevent noise."""
        logger.debug(f"LocalProxy: {format % args}")
    
    def do_GET(self) -> None:
        """Handle GET requests."""
        self._handle_request("GET")
    
    def do_POST(self) -> None:
        """Handle POST requests."""
        self._handle_request("POST")
    
    def do_PUT(self) -> None:
        """Handle PUT requests (blocked for restricted providers)."""
        self._handle_request("PUT")
    
    def do_PATCH(self) -> None:
        """Handle PATCH requests (blocked for restricted providers)."""
        self._handle_request("PATCH")
    
    def do_DELETE(self) -> None:
        """Handle DELETE requests (blocked for restricted providers)."""
        self._handle_request("DELETE")
    
    def _handle_request(self, method: str) -> None:
        """
        Handle proxy request.
        
        Expected path format: /{provider}/{endpoint}
        Example: /openrouter/chat/completions
        
        Special endpoints:
        - /cost - Returns accumulated API costs (for validator)
        - /cost/reset - Resets cost tracking
        """
        try:
            # Parse URL path
            parsed = urlparse(self.path)
            path_parts = parsed.path.strip("/").split("/", 1)
            
            if len(path_parts) < 1:
                self._send_error(400, "Missing provider in path")
                return
            
            provider = path_parts[0]
            endpoint = path_parts[1] if len(path_parts) > 1 else ""
            
            # ================================================================
            # Special endpoint: /cost - Returns cost summary
            # ================================================================
            if provider == "cost":
                if endpoint == "reset":
                    # Reset cost tracking
                    if self.cost_tracker:
                        self.cost_tracker.reset()
                    self._send_json({"status": "reset", "total_cost_usd": 0.0})
                else:
                    # Return cost summary
                    if self.cost_tracker:
                        self._send_json(self.cost_tracker.get_summary())
                    else:
                        self._send_json({"total_cost_usd": 0.0, "call_counts": {}, "token_counts": {}})
                return
            
            # Check provider allowlist
            if provider not in ALL_PROVIDERS:
                self._send_error(
                    403, 
                    f"Provider '{provider}' not in allowlist. "
                    f"Allowed: {list(ALL_PROVIDERS.keys())}"
                )
                return
            
            # ================================================================
            # SECURITY: GitHub endpoint allowlist (GET-only, strict list)
            # ================================================================
            if provider == "github":
                if method != "GET":
                    # Drain request body to avoid connection reset
                    try:
                        cl = int(self.headers.get("Content-Length", 0))
                        if cl > 0:
                            self.rfile.read(cl)
                    except Exception:
                        pass
                    self._send_error(
                        403,
                        f"GitHub: only GET requests allowed (got {method})"
                    )
                    return
                if not self._is_github_endpoint_allowed(endpoint):
                    self._send_error(
                        403,
                        f"GitHub endpoint not in allowlist: /{endpoint}"
                    )
                    return
            
            # Get API key for provider (stored in proxy thread's closure)
            api_key = self.api_keys.get(provider)
            if provider in PAID_PROVIDERS and not api_key:
                self._send_error(500, f"No API key configured for: {provider}")
                return
            
            # Build upstream URL
            base_url = ALL_PROVIDERS[provider]
            upstream_url = f"{base_url}/{endpoint}"
            
            # Include query string
            if parsed.query:
                upstream_url = f"{upstream_url}?{parsed.query}"
            
            # ================================================================
            # SPECIAL: Providers that use query-parameter auth, not headers
            # ================================================================
            if provider == "scrapingdog" and api_key:
                separator = "&" if "?" in upstream_url else "?"
                upstream_url = f"{upstream_url}{separator}api_key={api_key}"
            
            elif provider == "builtwith" and api_key:
                separator = "&" if "?" in upstream_url else "?"
                upstream_url = f"{upstream_url}{separator}KEY={api_key}"
            
            # Get request body for POST (also extract model for cost calculation)
            body = None
            request_model = None
            if method == "POST":
                content_length = int(self.headers.get("Content-Length", 0))
                if content_length > 0:
                    body = self.rfile.read(content_length)
                    # Try to extract model from request for cost calculation
                    try:
                        body_json = json.loads(body)
                        request_model = body_json.get("model")
                    except:
                        pass
            
            # Build headers with auth
            headers = self._get_auth_headers(provider, api_key)
            
            # Forward content-type
            if "Content-Type" in self.headers:
                headers["Content-Type"] = self.headers["Content-Type"]
            
            # Make upstream request
            response = self._make_upstream_request(
                method, upstream_url, headers, body
            )
            
            if response is None:
                return  # Error already sent
            
            # ================================================================
            # COST TRACKING: Extract cost from response
            # ================================================================
            if response.status_code == 200 and self.cost_tracker:
                # Parse query params for ScrapingDog credit calculation
                query_params = {}
                if parsed.query:
                    query_params = {k: v[0] for k, v in parse_qs(parsed.query).items()}
                
                self._track_cost(
                    provider=provider, 
                    response=response, 
                    request_model=request_model,
                    endpoint=endpoint,
                    query_params=query_params
                )
            
            # Send response
            self.send_response(response.status_code)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(response.content)
            
        except Exception as e:
            logger.error(f"LocalProxy error: {e}")
            self._send_error(500, str(e))
    
    def _track_cost(
        self, 
        provider: str, 
        response: httpx.Response, 
        request_model: Optional[str] = None,
        endpoint: str = "",
        query_params: Optional[Dict[str, str]] = None
    ) -> None:
        """
        Extract and track cost from API response.
        
        For OpenRouter: Extracts actual cost from usage.cost (ALWAYS present).
        For ScrapingDog: Calculates based on endpoint and credits used.
        For other providers: Uses fixed per-call cost.
        
        Args:
            provider: API provider name (e.g., "openrouter", "scrapingdog")
            response: HTTP response from upstream
            request_model: Model name for OpenRouter calls
            endpoint: API endpoint for ScrapingDog (e.g., "scrape", "google", "linkedin")
            query_params: Query parameters for ScrapingDog pricing calculation
        """
        try:
            if provider == "openrouter":
                # Parse OpenRouter response for cost info
                data = response.json()
                usage = data.get("usage", {})
                
                if not usage:
                    logger.warning(f"LocalProxy: OpenRouter response missing usage object")
                    return
                
                input_tokens = usage.get("prompt_tokens", 0)
                output_tokens = usage.get("completion_tokens", 0)
                
                # Extract ACTUAL cost from OpenRouter response
                # OpenRouter ALWAYS returns this - no estimation needed
                try:
                    cost = extract_openrouter_cost(usage)
                except ValueError as e:
                    # This should never happen, but log loudly if it does
                    logger.error(f"LocalProxy: {e}")
                    return
                
                self.cost_tracker.add_cost(
                    provider="openrouter",
                    cost_usd=cost,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens
                )
                
                model = request_model or data.get("model", "unknown")
                logger.info(f"💰 OpenRouter [{model}]: ${cost:.6f} ({input_tokens}+{output_tokens} tokens)")
            
            elif provider == "scrapingdog":
                # ScrapingDog: Calculate cost based on endpoint and credits
                query_params = query_params or {}
                credits, description = calculate_scrapingdog_credits(
                    endpoint=endpoint,
                    query_params=query_params
                )
                
                # Load plan config from CONFIG if available
                try:
                    from gateway.qualification.config import CONFIG
                    plan_cost = CONFIG.SCRAPINGDOG_PLAN_COST_USD
                    plan_credits = CONFIG.SCRAPINGDOG_PLAN_CREDITS
                except ImportError:
                    plan_cost = DEFAULT_SCRAPINGDOG_PLAN_COST_USD
                    plan_credits = DEFAULT_SCRAPINGDOG_PLAN_CREDITS
                
                cost = calculate_scrapingdog_cost_usd(
                    credits=credits,
                    plan_cost_usd=plan_cost,
                    plan_credits=plan_credits
                )
                
                self.cost_tracker.add_cost(provider="scrapingdog", cost_usd=cost)
                logger.info(f"💰 ScrapingDog [{description}]: ${cost:.6f} ({credits} credits)")
            
            elif provider == "desearch":
                # Desearch has different pricing per endpoint type
                # - X (Twitter) API: $0.30 / 1000 posts = $0.0003 per result
                # - Web API: $0.25 / 100 searches = $0.0025 per search
                # - AI Search API: $0.40 / 100 searches = $0.004 per search
                cost, description = self._calculate_desearch_cost(endpoint, response)
                if cost > 0:
                    self.cost_tracker.add_cost(provider="desearch", cost_usd=cost)
                    logger.info(f"💰 Desearch [{description}]: ${cost:.6f}")
            
            elif provider == "builtwith":
                cost, description = self._calculate_builtwith_cost(endpoint, response)
                if cost >= 0:
                    self.cost_tracker.add_cost(provider="builtwith", cost_usd=cost)
                    logger.info(f"💰 BuiltWith [{description}]: ${cost:.6f}")
            
            elif provider == "github":
                # GitHub free tier: $0 per call, but track call count
                self.cost_tracker.add_cost(provider="github", cost_usd=0.0)
                logger.info(f"💰 GitHub [{endpoint.split('/')[0]}]: $0 (free)")
            
            else:
                # For other paid providers, use fixed per-call cost
                per_call_costs = {
                    "crunchbase": 0.01,      # $0.01 per call
                    "datauniverse": 0.00005, # $0.00005 per call
                    "googlenews": 0.001,     # $0.001 per call
                    "jobdata": 0.01,         # $0.01 per call
                }
                
                cost = per_call_costs.get(provider, 0.0)
                if cost > 0:
                    self.cost_tracker.add_cost(provider=provider, cost_usd=cost)
                    logger.info(f"💰 {provider}: ${cost:.6f} (per-call)")
        
        except json.JSONDecodeError:
            logger.warning(f"LocalProxy: Response not valid JSON for {provider}")
        except Exception as e:
            logger.warning(f"LocalProxy: Failed to track cost for {provider}: {e}")
    
    def _calculate_desearch_cost(
        self,
        endpoint: str,
        response: httpx.Response
    ) -> Tuple[float, str]:
        """
        Calculate cost for a Desearch API call.
        
        Desearch API Pricing (from https://desearch.ai/pricing):
        - X (Twitter) API: $0.30 / 1000 posts = $0.0003 per result
        - Web API: $0.25 / 100 searches = $0.0025 per search
        - AI Search API: $0.40 / 100 searches = $0.004 per search
        
        Actual Endpoint Patterns (from https://desearch.ai/docs/api-reference):
        
        AI Search ($0.004/call):
        - POST /desearch/ai/search - AI Contextual Search
        - POST /desearch/ai/search/links/web - AI Web Search
        - POST /desearch/ai/search/links/twitter - AI X posts search
        
        X/Twitter ($0.0003/result):
        - GET /twitter?query=... - X search API
        - GET /twitter/urls?urls=... - Fetch Posts by URL
        - GET /twitter/post - Retrieve post by ID
        - GET /twitter/post/user?user=... - Search X posts by user
        - GET /twitter/user/posts?username=... - Get X posts by username
        - GET /twitter/replies?user=... - Fetch Users Tweets and replies
        - GET /twitter/replies/post?post_id=... - Retrieve Replies for a post
        
        Web ($0.0025/call):
        - GET /web?query=... - Serp Web Search API
        - GET /web/crawl?url=... - Crawl API
        
        Args:
            endpoint: The API endpoint called
            response: The HTTP response to count results
        
        Returns:
            Tuple of (cost_usd, description)
        """
        try:
            # Load pricing from config if available
            try:
                from gateway.qualification.config import CONFIG
                x_cost_per_1000 = CONFIG.DESEARCH_X_SEARCH_COST_PER_1000
                web_cost_per_100 = CONFIG.DESEARCH_WEB_SEARCH_COST_PER_100
                ai_cost_per_100 = CONFIG.DESEARCH_AI_SEARCH_COST_PER_100
            except ImportError:
                x_cost_per_1000 = 0.30
                web_cost_per_100 = 0.25
                ai_cost_per_100 = 0.40
            
            endpoint_lower = endpoint.lower()
            
            # AI Search APIs - charged per search ($0.004)
            # Endpoints: /desearch/ai/search, /desearch/ai/search/links/web, /desearch/ai/search/links/twitter
            if "/desearch/ai/" in endpoint_lower:
                cost = ai_cost_per_100 / 100  # $0.004
                if "links/twitter" in endpoint_lower:
                    return (cost, "AI X search")
                elif "links/web" in endpoint_lower:
                    return (cost, "AI Web search")
                else:
                    return (cost, "AI Contextual search")
            
            # X (Twitter) APIs - charged per result ($0.0003)
            # Endpoints: /twitter, /twitter/urls, /twitter/post, /twitter/post/user, 
            # /twitter/user/posts, /twitter/replies, /twitter/replies/post
            elif "/twitter" in endpoint_lower:
                # Try to count results in response
                try:
                    data = response.json()
                    if isinstance(data, list):
                        num_results = len(data)
                    elif isinstance(data, dict):
                        # Check various response formats
                        if "tweets" in data:
                            num_results = len(data["tweets"])
                        elif "miner_tweets" in data:
                            num_results = len(data["miner_tweets"])
                        elif "data" in data:
                            num_results = len(data["data"])
                        else:
                            num_results = 1  # Single post/user response
                    else:
                        num_results = 10  # Default estimate
                except:
                    num_results = 10
                
                cost_per_result = x_cost_per_1000 / 1000  # $0.0003
                cost = cost_per_result * max(num_results, 1)
                
                # Identify endpoint type for logging
                if "/replies" in endpoint_lower:
                    desc = f"X replies ({num_results} results)"
                elif "/urls" in endpoint_lower:
                    desc = f"X posts by URL ({num_results} results)"
                elif "/user" in endpoint_lower:
                    desc = f"X user posts ({num_results} results)"
                elif "/post" in endpoint_lower:
                    desc = f"X post ({num_results} results)"
                else:
                    desc = f"X search ({num_results} results)"
                
                return (cost, desc)
            
            # Web APIs - charged per search ($0.0025)
            # Endpoints: /web, /web/crawl
            elif "/web" in endpoint_lower:
                cost = web_cost_per_100 / 100  # $0.0025
                if "/crawl" in endpoint_lower:
                    return (cost, "Web crawl")
                else:
                    return (cost, "Web search")
            
            # Default - assume web search cost
            else:
                cost = web_cost_per_100 / 100
                return (cost, f"Unknown ({endpoint[:30]})")
        
        except Exception as e:
            logger.warning(f"Failed to calculate Desearch cost: {e}")
            # Default to web search cost
            return (0.0025, "unknown")
    
    # BuiltWith free-plan endpoints and their cost structure
    BUILTWITH_FREE_ENDPOINTS = {
        "free1":  0.0,    # Free API - technology groups/categories
        "lists12": 0.0,   # Lists API - sites using a technology
        "trends": 0.0,    # Trends API - technology trend data
    }
    BUILTWITH_PAID_ENDPOINTS = {
        "v22":        0.05,  # Domain API
        "rv4":        0.05,  # Relationships API
        "ctu3":       0.05,  # Company to URL API
        "tag1":       0.05,  # Tags API
        "rec1":       0.05,  # Recommendations API
        "redirect1":  0.05,  # Redirects API
        "kw2":        0.05,  # Keywords API
        "productv1":  0.05,  # Product API
        "trustv1":    0.05,  # Trust API
        "social1":    0.05,  # Social API
        "financial1": 0.05,  # Financial API
    }

    def _calculate_builtwith_cost(
        self,
        endpoint: str,
        response: httpx.Response
    ) -> Tuple[float, str]:
        """
        Calculate cost for a BuiltWith API call.

        Free plan endpoints (free1, lists12, trends) cost $0.
        Paid endpoints cost ~$0.05 per call (but will return errors on free plan).
        """
        # Extract the API prefix from the endpoint path
        # e.g. "free1/api.json" -> "free1", "trends/v6/api.json" -> "trends"
        prefix = endpoint.split("/")[0] if endpoint else ""

        if prefix in self.BUILTWITH_FREE_ENDPOINTS:
            return (0.0, f"builtwith/{prefix} (free)")

        if prefix in self.BUILTWITH_PAID_ENDPOINTS:
            cost = self.BUILTWITH_PAID_ENDPOINTS[prefix]
            return (cost, f"builtwith/{prefix} (paid)")

        return (0.0, f"builtwith/{prefix} (unknown)")

    @staticmethod
    def _is_github_endpoint_allowed(endpoint: str) -> bool:
        """
        Check if a GitHub API endpoint is on the strict allowlist.

        Only read-only metadata/search endpoints are permitted.
        Blocks: gists, contents, git objects, user account, all writes.
        """
        # Strip query string if present (patterns match path only)
        clean = endpoint.split("?")[0].rstrip("/")
        return any(p.match(clean) for p in _GITHUB_COMPILED_PATTERNS)

    def _send_json(self, data: Dict[str, Any]) -> None:
        """Send JSON response."""
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())
    
    def _get_auth_headers(
        self, provider: str, api_key: Optional[str]
    ) -> Dict[str, str]:
        """Get authentication headers for provider."""
        headers: Dict[str, str] = {}
        
        if provider in FREE_PROVIDERS:
            if provider == "github" and api_key:
                headers["Authorization"] = f"token {api_key}"
                headers["Accept"] = "application/vnd.github.v3+json"
            return headers
        
        if not api_key:
            return headers
        
        if provider == "openrouter":
            headers["Authorization"] = f"Bearer {api_key}"
            headers["HTTP-Referer"] = "https://leadpoet.ai"
            headers["X-Title"] = "LeadPoet Qualification"
        
        elif provider == "scrapingdog":
            # ScrapingDog uses api_key as query parameter, handled in do_GET/do_POST
            pass
        
        elif provider == "builtwith":
            # BuiltWith uses KEY= query parameter, handled in do_GET/do_POST
            pass
        
        elif provider == "crunchbase":
            headers["X-Cb-User-Key"] = api_key
        
        elif provider == "desearch":
            # Desearch uses plain API key, NOT Bearer token
            # Docs: https://desearch.ai/docs/api-reference/post-desearch-ai-search
            headers["Authorization"] = api_key
        
        elif provider == "datauniverse":
            headers["Authorization"] = f"Bearer {api_key}"
        
        elif provider == "googlenews":
            headers["X-Api-Key"] = api_key
        
        elif provider == "jobdata":
            headers["Authorization"] = f"Bearer {api_key}"
        
        return headers
    
    def _make_upstream_request(
        self, 
        method: str, 
        url: str, 
        headers: Dict[str, str], 
        body: Optional[bytes]
    ) -> Optional[httpx.Response]:
        """Make synchronous upstream request."""
        try:
            with httpx.Client(timeout=PROXY_TIMEOUT_SECONDS) as client:
                if method == "GET":
                    return client.get(url, headers=headers)
                elif method == "POST":
                    json_body = None
                    if body:
                        try:
                            json_body = json.loads(body)
                        except json.JSONDecodeError:
                            pass
                    
                    if json_body:
                        return client.post(url, json=json_body, headers=headers)
                    else:
                        return client.post(url, content=body, headers=headers)
        
        except httpx.TimeoutException:
            self._send_error(504, f"Upstream timeout ({PROXY_TIMEOUT_SECONDS}s)")
            return None
        except httpx.RequestError as e:
            self._send_error(502, f"Upstream error: {e}")
            return None
    
    def _send_error(self, code: int, message: str) -> None:
        """Send JSON error response."""
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        error_body = json.dumps({"error": message}).encode()
        self.wfile.write(error_body)


# =============================================================================
# Local Proxy Server
# =============================================================================

class LocalProxyServer:
    """
    Manages the local proxy server running in a separate thread.
    
    Usage:
        proxy = LocalProxyServer()
        proxy.start()
        # ... run model with proxy.url ...
        cost = proxy.get_total_cost()  # Get accumulated API costs
        proxy.stop()
    
    Or use context manager:
        with LocalProxyServer() as proxy:
            # ... run model with proxy.url ...
            cost = proxy.get_total_cost()
    
    COST TRACKING:
        - Automatically tracks costs for all API calls
        - For OpenRouter: extracts actual cost from response (or estimates from tokens)
        - For other providers: uses fixed per-call costs
        - Query via: GET {proxy.url}/cost
        - Reset via: GET {proxy.url}/cost/reset
    """
    
    def __init__(self, port: int = 0):
        """
        Initialize proxy server.
        
        Args:
            port: Port to listen on (0 = auto-assign)
        """
        self._port = port
        self._server: Optional[ThreadingHTTPServer] = None
        self._thread: Optional[threading.Thread] = None
        self._api_keys: Dict[str, str] = {}
        self._cost_tracker: CostTracker = CostTracker()
    
    def _load_api_keys(self) -> Dict[str, str]:
        """
        Load API keys from environment variables.
        
        SECURITY: This is called BEFORE the sandbox sanitizes os.environ.
        The keys are stored in a closure variable within the proxy thread.
        After sandbox activation, os.environ won't have these keys anymore,
        but the proxy thread still has them in its closure.
        """
        keys = {}
        
        # Map provider names to environment variable names
        # 
        # SECURITY: Qualification uses SEPARATE API keys with limited funds.
        # If a malicious miner somehow extracts keys, they only get the
        # qualification keys (limited budget), not the main sourcing keys.
        #
        env_var_map = {
            # TODO: After beta release, change back to "OPENROUTER_API_KEY" (shared with sourcing)
            "openrouter": "QUALIFICATION_OPENROUTER_API_KEY",
            # TODO: After beta release, change back to "SCRAPINGDOG_API_KEY" (shared with sourcing)
            "scrapingdog": "QUALIFICATION_SCRAPINGDOG_API_KEY",
            "builtwith": "BUILTWITH_API_KEY",
            "crunchbase": "CRUNCHBASE_API_KEY",
            # Desearch stays as-is (sourcing doesn't use it)
            "desearch": "DESEARCH_API_KEY",
            "datauniverse": "DATAUNIVERSE_API_KEY",
            "googlenews": "NEWS_API_KEY",
            "jobdata": "JOBDATA_API_KEY",
            # SECURITY: GitHub removed from proxy - models could use gists to exfiltrate data
            # Validator's intent_verification.py still uses GITHUB_TOKEN for READ-only verification
        }
        
        for provider, env_var in env_var_map.items():
            key = os.environ.get(env_var)
            if key:
                keys[provider] = key
                logger.debug(f"LocalProxy: Loaded key for {provider}")
        
        return keys
    
    def start(self) -> str:
        """
        Start the proxy server in a background thread.
        
        Returns:
            The proxy URL (e.g., "http://localhost:12345")
        
        SECURITY: Keys are loaded HERE, before sandbox activates.
        They're stored in the handler's closure, not os.environ.
        """
        # Load keys NOW (before sandbox sanitizes os.environ)
        self._api_keys = self._load_api_keys()
        
        # Reset cost tracker for fresh evaluation
        self._cost_tracker.reset()
        
        # Find available port
        if self._port == 0:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.bind(('localhost', 0))
            self._port = sock.getsockname()[1]
            sock.close()
        
        # Create custom handler class with API keys and cost tracker in closure
        api_keys = self._api_keys  # Capture in closure
        cost_tracker = self._cost_tracker  # Capture in closure
        
        class ProxyHandlerWithKeys(LocalProxyHandler):
            pass
        
        ProxyHandlerWithKeys.api_keys = api_keys
        ProxyHandlerWithKeys.cost_tracker = cost_tracker
        
        # Create THREADED server - handles each request in its own thread
        # This prevents blocking when multiple requests come in
        self._server = ThreadingHTTPServer(('localhost', self._port), ProxyHandlerWithKeys)
        self._server.daemon_threads = True  # Threads die when main thread dies
        
        # Start server in background thread using serve_forever()
        # serve_forever() is more efficient than manual handle_request() loop
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        
        url = f"http://localhost:{self._port}"
        logger.info(f"LocalProxy started at {url} (cost tracking enabled)")
        return url
    
    def stop(self) -> None:
        """Stop the proxy server."""
        if self._server:
            logger.info(f"LocalProxy stopping...")
            server = self._server
            self._server = None
            server.shutdown()
            
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None
        
        # Clear keys from memory
        self._api_keys.clear()
        logger.info("LocalProxy stopped")
    
    @property
    def url(self) -> str:
        """Get the proxy URL."""
        return f"http://localhost:{self._port}"
    
    @property
    def port(self) -> int:
        """Get the proxy port."""
        return self._port
    
    def get_total_cost(self) -> float:
        """
        Get total accumulated API cost in USD.
        
        This is the source of truth for model evaluation cost.
        Returns actual costs from OpenRouter (when available) or estimates.
        """
        return self._cost_tracker.total_cost_usd
    
    def get_cost_summary(self) -> Dict[str, Any]:
        """
        Get detailed cost breakdown.
        
        Returns:
            {
                "total_cost_usd": 0.001234,
                "call_counts": {"openrouter": 5, "scrapingdog": 2},
                "token_counts": {"openrouter": {"input": 500, "output": 200}}
            }
        """
        return self._cost_tracker.get_summary()
    
    def reset_cost(self) -> None:
        """Reset cost tracking (e.g., between ICP evaluations)."""
        self._cost_tracker.reset()
    
    def __enter__(self) -> "LocalProxyServer":
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.stop()


# =============================================================================
# Convenience Function
# =============================================================================

@contextmanager
def local_proxy():
    """
    Context manager for running local proxy.
    
    Usage:
        with local_proxy() as proxy_url:
            # Run model with proxy_url in icp["_config"]["PROXY_URL"]
            pass
    
    Yields:
        proxy_url: The URL to use for API calls (e.g., "http://localhost:12345")
    """
    server = LocalProxyServer()
    try:
        url = server.start()
        yield url
    finally:
        server.stop()
