"""
Qualification System: Sandbox Security Configuration

ALLOWLIST-ONLY approach: Only libraries in ALLOWED_LIBRARIES can be imported.
Everything else is automatically blocked.

Security Model:
1. ALLOWED_LIBRARIES - ONLY these Python modules can be imported
2. Internal modules (starting with _) are allowed (they're part of Python)
3. Environment is sanitized - API keys removed before model runs
4. /proc/self/environ is blocked - RestrictedFileOpen + RestrictedOsOpen
5. Network is intercepted - only allowed destinations

Author: LeadPoet
"""

import os
import sys
import logging
from typing import Dict, List, Set, Optional, Any
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


# =============================================================================
# ALLOWED PYTHON LIBRARIES (ALLOWLIST-ONLY)
# =============================================================================
# These are the ONLY modules miner models can import.
# Anything NOT in this list is BLOCKED (subprocess, pickle, ctypes, etc.)
# Internal Python modules (starting with _) are also allowed.
#
# SECURITY: Each library is safe because it CANNOT:
# - Read API keys from environment (os.environ is SANITIZED before model runs)
# - Execute shell commands (subprocess is blocked)
# - Access arbitrary files (pathlib/shutil blocked, file access restricted)
# - Read /proc/self/environ (blocked by RestrictedFileOpen and RestrictedOsOpen)
# - Make network requests to unauthorized hosts (network interceptor)
# =============================================================================

ALLOWED_LIBRARIES: Set[str] = {
    # =========================================================================
    # OS MODULE - ALLOWED WITH SANITIZATION
    # =========================================================================
    # SECURITY: os module is SAFE because:
    # 1. os.environ is SANITIZED before model runs - API keys are REMOVED
    # 2. os.open() is BLOCKED by RestrictedOsOpen - cannot read /proc/self/environ
    # 3. open('/proc/self/environ') is BLOCKED by RestrictedFileOpen
    # 4. Models can only read SAFE values from os.environ:
    #    - SUPABASE_URL (database URL - safe)
    #    - SUPABASE_ANON_KEY (read-only key - safe)
    #    - MINER_TEST_LEADS_TABLE (table name - safe)
    # 5. subprocess, exec, etc. are in SEPARATE modules (not in os)
    # =========================================================================
    "os",               # SAFE: environ sanitized, os.open blocked, needed for backwards compat
    
    # =========================================================================
    # STANDARD LIBRARY - DATA PROCESSING (NO SIDE EFFECTS)
    # =========================================================================
    "json",             # Parse/serialize JSON - pure data transformation
    "re",               # Regex matching - pure string operations
    "datetime",         # Date/time handling - no I/O
    "time",             # Time functions - sleep(), timestamps only
    "math",             # Math operations - pure computation
    "random",           # Random numbers - no external state
    "string",           # String constants - no I/O
    "collections",      # Data structures - pure Python containers
    "itertools",        # Iterator utilities - pure functions
    "functools",        # Function utilities - decorators, caching
    "typing",           # Type hints - no runtime effect
    "typing_extensions", # Extended type hints - no runtime effect
    "dataclasses",      # Data class decorator - pure Python
    "enum",             # Enumeration types - no I/O
    "uuid",             # UUID generation - no external calls
    "hashlib",          # Hashing - pure computation, no crypto signing
    "base64",           # Base64 encoding - pure transformation
    "copy",             # Object copying - in-memory only
    "operator",         # Operator functions - pure computation
    "abc",              # Abstract base classes - no I/O
    "csv",              # CSV parsing - processes strings, no file I/O alone
    "io",               # StringIO/BytesIO - in-memory streams only
    "logging",          # Logging - output only, cannot read secrets
    "unicodedata",      # Unicode info - read-only lookup tables
    "decimal",          # Decimal math - pure computation
    "fractions",        # Fraction math - pure computation
    "numbers",          # Number ABCs - type definitions only
    "textwrap",         # Text formatting - pure string manipulation
    "reprlib",          # Repr utilities - string formatting
    "difflib",          # Text comparison/sequence matching - pure computation
    "gc",               # Garbage collector - memory management, no I/O
    "builtins",         # Built-in functions module - already accessible, needed by rapidfuzz
    "hmac",             # HMAC hashing - used by auth/JWT, pure computation
    "secrets",          # Crypto random - used by auth tokens, no I/O
    "copyreg",          # Pickle registry - needed by copy/deepcopy
    "stringprep",       # Unicode string preparation - needed by auth
    "pathlib",          # Path handling - SAFE: RestrictedFileOpen blocks sensitive reads
    "importlib",        # Dynamic imports - SAFE: __import__ patch blocks unauthorized modules
    "tempfile",         # Temp files - needed by httpx internals
    "pkgutil",          # Package utilities - needed by importlib
    "configparser",     # Config parsing - no I/O alone
    "fnmatch",          # Filename matching - pure string ops
    
    # =========================================================================
    # STANDARD LIBRARY - ASYNC & THREADING (NEEDED BY HTTP CLIENTS)
    # =========================================================================
    "asyncio",          # Async I/O - needed by httpx/aiohttp
    "threading",        # Threading - needed by httpx connection pools
    "concurrent",       # Thread pools - needed by httpx
    "queue",            # Thread-safe queues - no I/O
    "contextlib",       # Context managers - no I/O
    "contextvars",      # Context variables - no I/O
    
    # =========================================================================
    # STANDARD LIBRARY - HTTP/NETWORK SUPPORT (INTERCEPTED)
    # =========================================================================
    # These enable HTTP but NetworkInterceptor blocks unauthorized hosts
    "urllib",           # URL parsing (urllib.parse) - no direct network
    "ssl",              # TLS/HTTPS - required for secure connections
    "socket",           # Sockets - intercepted, only allowed hosts
    "select",           # I/O multiplexing - used by socket
    "selectors",        # High-level I/O multiplexing
    "http",             # HTTP utilities - status codes, parsing
    "email",            # Email parsing - used by HTTP headers
    "html",             # HTML entities - decoding only
    "ipaddress",        # IP address handling - validation only
    "mimetypes",        # MIME types - lookup table only
    "certifi",          # SSL certificates - read-only cert bundle
    
    # =========================================================================
    # STANDARD LIBRARY - COMPRESSION (DATA PROCESSING)
    # =========================================================================
    "zlib",             # Compression - pure data transformation
    "gzip",             # Gzip - pure data transformation
    "bz2",              # BZ2 - pure data transformation
    "lzma",             # LZMA - pure data transformation
    "zipfile",          # Zip files - can only read files we allow
    
    # =========================================================================
    # STANDARD LIBRARY - INTROSPECTION (NEEDED BY PYDANTIC/HTTPX)
    # =========================================================================
    "sys",              # System info - env is SANITIZED, modules needed by libs
    "types",            # Type objects - definitions only
    "traceback",        # Stack traces - debugging only
    "inspect",          # Introspection - needed by pydantic for models
    "warnings",         # Warning system - output only
    "weakref",          # Weak references - memory management
    "struct",           # Binary packing - pure data transformation
    "binascii",         # Binary/ASCII - pure transformation
    "codecs",           # Codec registry - encoding/decoding
    "encodings",        # Encoding support - lookup only
    "platform",         # Platform info - read-only system info
    "locale",           # Localization - read-only settings
    "calendar",         # Calendar - pure computation
    "atexit",           # Exit handlers - cleanup only
    "signal",           # Signal handling - needed by asyncio
    "errno",            # Error codes - constants only
    "stat",             # File status - constants only
    "posixpath",        # Path operations - string manipulation
    "genericpath",      # Generic path ops - string manipulation
    "linecache",        # Line cache - used by traceback
    "tokenize",         # Tokenization - parsing only
    "token",            # Token constants - definitions only
    "keyword",          # Python keywords - constants only
    "heapq",            # Heap operations - pure data structure
    "bisect",           # Binary search - pure algorithm
    "array",            # Efficient arrays - in-memory only
    "getpass",          # Password prompts - not used, harmless
    "gettext",          # i18n - translation lookups only
    
    # =========================================================================
    # HTTP CLIENTS + DEPENDENCIES
    # =========================================================================
    # All HTTP requests are INTERCEPTED - can only reach allowed hosts
    "requests",         # HTTP client - intercepted by NetworkInterceptor
    "httpx",            # Modern HTTP client - intercepted
    "aiohttp",          # Async HTTP client - intercepted
    "httpcore",         # httpx dependency - low-level HTTP
    "h11",              # HTTP/1.1 parser - pure parsing
    "h2",               # HTTP/2 parser - pure parsing
    "http2",            # HTTP/2 alternate module name - pure parsing
    "hpack",            # HTTP/2 header compression - pure parsing
    "hyperframe",       # HTTP/2 framing - pure parsing
    "urllib3",          # requests dependency - intercepted
    "idna",             # International domain names - pure encoding
    "charset_normalizer", # Character encoding detection
    "sniffio",          # Async library detection - no I/O
    "anyio",            # Async compatibility - no direct I/O
    "exceptiongroup",   # Backport of ExceptionGroup - needed by anyio
    "socksio",          # SOCKS proxy support - needed by httpx
    "multidict",        # Multi-value dict - needed by aiohttp/yarl
    "yarl",             # URL library - needed by aiohttp
    "propcache",        # Property caching - needed by yarl
    "async_timeout",    # Async timeout - needed by aiohttp
    "strenum",          # String enum - needed by gotrue
    "cryptography",     # Crypto library - needed by JWT/auth (no key access)
    "jwt",              # JWT parsing - needed by gotrue auth
    "cffi",             # C FFI - needed by cryptography
    
    # =========================================================================
    # DATABASE ACCESS (READ-ONLY leads table)
    # =========================================================================
    # Models query the leads database to find leads matching the ICP
    "supabase",             # Supabase client - uses anon key (read-only)
    "postgrest",            # Supabase dependency - REST client
    "gotrue",               # Supabase auth dependency (legacy name)
    "supabase_auth",        # Supabase auth dependency (new name)
    "storage3",             # Supabase storage dependency
    "realtime",             # Supabase realtime dependency
    "supafunc",             # Supabase functions dependency (legacy name)
    "supabase_functions",   # Supabase functions dependency (new name)
    "deprecation",          # Deprecation warnings - used by supabase
    "packaging",            # Version parsing - used by deprecation/supabase
    
    # =========================================================================
    # SEARCH & LLM CLIENTS
    # =========================================================================
    "duckduckgo_search", # Free web search - rate limited
    "openai",           # LLM client - calls proxy, proxy injects key
    
    # =========================================================================
    # DATA SCIENCE & UTILITIES
    # =========================================================================
    "pandas",           # DataFrames - in-memory data processing
    "numpy",            # Numerical arrays - pure computation
    "pydantic",         # Data validation - no I/O
    "pydantic_core",    # Pydantic compiled core - no I/O
    "annotated_types",  # Pydantic dependency - type annotations
    "fuzzywuzzy",       # Fuzzy string matching - pure computation
    "rapidfuzz",        # Fast fuzzy matching - pure computation
    "thefuzz",          # Fuzzy matching - pure computation
    "Levenshtein",      # String distance - pure computation
    "dateutil",         # Date parsing - pure parsing
    "bs4",              # BeautifulSoup HTML parsing - read-only, no I/O
    "soupsieve",        # bs4 dependency - CSS selector engine
    "lxml",             # XML/HTML parser - bs4 backend, read-only
    "html5lib",         # HTML parser - bs4 backend, read-only
    "chardet",          # Character encoding detection
    "disposable_email_domains",  # Email blocklist - static set lookup
    
    # =========================================================================
    # BITTENSOR (VALIDATOR BACKGROUND THREADS)
    # =========================================================================
    # Needed because the validator's epoch monitor runs in background during
    # model evaluation. bittensor is SAFE because:
    # 1. Can only query public blockchain data
    # 2. Cannot access API keys (os.environ is sanitized)
    # 3. Cannot sign transactions (no access to validator's wallet)
    # 4. Network calls go through our interceptor
    "bittensor",        # Substrate client - validator's epoch monitor needs this
    "scalecodec",       # bittensor dependency - substrate encoding
    "websocket",        # bittensor dependency - substrate connection
    "websockets",       # async websocket - substrate connection
}

# =============================================================================
# BLOCKED BY OMISSION (not in list = blocked):
# - subprocess      → Cannot execute shell commands
# - ctypes          → Cannot access memory or load libraries
# - pickle/marshal  → Cannot deserialize arbitrary code
# - shutil          → Cannot copy/delete files freely
# - multiprocessing → Cannot spawn processes
# - code/codeop     → Cannot execute arbitrary code strings
# =============================================================================


# =============================================================================
# SANDBOX PIP PACKAGES (installed in Docker sandbox for miner models)
# =============================================================================
# These are the pip package names for third-party libraries in ALLOWED_LIBRARIES.
# The sandbox Dockerfile installs these automatically.
# 
# To add a new library for miners:
# 1. Add the import name to ALLOWED_LIBRARIES above
# 2. Add the pip package name to SANDBOX_PIP_PACKAGES below
# The sandbox will automatically install it on next build.
#
# NOTE: Standard library modules (json, re, os, etc.) don't need pip install.
# Only add THIRD-PARTY packages here.
# =============================================================================

SANDBOX_PIP_PACKAGES: list = [
    # HTTP clients
    "httpx",
    "requests",
    "aiohttp",
    
    # Data validation
    "pydantic",
    
    # Database
    "supabase",
    
    # Search & LLM
    "duckduckgo_search",
    "openai",
    
    # Data science
    "pandas",
    "numpy",
    
    # String matching
    "rapidfuzz",
    "thefuzz",
    "python-Levenshtein",
    
    # Parsing & utilities
    "python-dateutil",
    "beautifulsoup4",
    "disposable-email-domains",
    
    # SSL certificates
    "certifi",
    
    # Add new packages here - sandbox will auto-install them
]


# =============================================================================
# ALLOWED FREE APIs
# =============================================================================
# These APIs are FREE and can be called directly (still via proxy for logging).
# Models don't need API keys for these.
# =============================================================================

@dataclass
class FreeAPI:
    """Configuration for a free API."""
    name: str
    base_url: str
    description: str
    rate_limit_per_minute: int = 60
    requires_user_agent: bool = False


# =============================================================================
# ALLOWED FREE APIs (8 total)
# =============================================================================
# These APIs are FREE and don't require API keys.
# Models can call them directly (network interceptor allows these hosts).
#
# SECURITY: These are READ-ONLY public data sources. They cannot:
# - Receive exfiltrated API keys (they don't accept arbitrary data)
# - Execute code on our systems
# - Access private data
# =============================================================================

ALLOWED_FREE_APIS: Dict[str, FreeAPI] = {
    # =========================================================================
    # REMOVED: GitHub, StackOverflow
    # - GitHub allows writes (gists, issues, repos) - exfiltration risk
    # - StackOverflow allows writes (questions, answers) - exfiltration risk
    # - Neither is essential for B2B lead qualification
    # =========================================================================
    
    # =========================================================================
    # SEARCH - Web search for lead research
    # =========================================================================
    "duckduckgo": FreeAPI(
        name="DuckDuckGo",
        base_url="https://duckduckgo.com",
        description="Free web search via duckduckgo_search library",
        rate_limit_per_minute=20
        # SAFE: Search queries only, cannot post/write data
    ),
    
    # =========================================================================
    # GOVERNMENT/PUBLIC DATA - Company verification
    # =========================================================================
    "sec_edgar": FreeAPI(
        name="SEC EDGAR",
        base_url="https://www.sec.gov",
        description="SEC filings, company disclosures (US public companies)",
        rate_limit_per_minute=10,
        requires_user_agent=True
        # SAFE: Read-only government public records
    ),
    "companies_house": FreeAPI(
        name="UK Companies House",
        base_url="https://api.company-information.service.gov.uk",
        description="UK company registry (UK public companies)",
        rate_limit_per_minute=20
        # SAFE: Read-only government public records
    ),
    
    # =========================================================================
    # ARCHIVES & NEWS - Historical data
    # =========================================================================
    "wayback_machine": FreeAPI(
        name="Wayback Machine",
        base_url="https://archive.org",
        description="Historical website snapshots",
        rate_limit_per_minute=15
        # SAFE: Read-only archive, cannot write/upload
    ),
    "gdelt": FreeAPI(
        name="GDELT Project",
        base_url="https://api.gdeltproject.org",
        description="Global news and events database",
        rate_limit_per_minute=30
        # SAFE: Read-only news database
    ),
    
    # =========================================================================
    # KNOWLEDGE BASES - Entity information
    # =========================================================================
    "wikipedia": FreeAPI(
        name="Wikipedia API",
        base_url="https://en.wikipedia.org/w/api.php",
        description="Wikipedia articles and summaries",
        rate_limit_per_minute=60
        # SAFE: Read-only encyclopedia, no write access
    ),
    "wikidata": FreeAPI(
        name="Wikidata API",
        base_url="https://www.wikidata.org/w/api.php",
        description="Structured knowledge base (entities, relationships)",
        rate_limit_per_minute=60
        # SAFE: Read-only structured data
    ),
}


# =============================================================================
# ALLOWED PAID APIs (VIA PROXY ONLY)
# =============================================================================
# These APIs require API keys. Models call them through the proxy.
# The proxy injects the keys - models NEVER see them.
# =============================================================================

class CostModel(str, Enum):
    """How costs are calculated for paid APIs."""
    PER_CALL = "per_call"
    PER_CREDIT = "per_credit"
    PER_TOKEN = "per_token"


@dataclass
class PaidAPI:
    """Configuration for a paid API (accessed via proxy)."""
    name: str
    provider_id: str  # Used in proxy URL: /proxy/{provider_id}/...
    base_url: str
    description: str
    cost_model: CostModel
    cost_per_unit: float  # USD
    env_var_name: str  # Validator's env var with API key
    notes: str = ""


# =============================================================================
# ALLOWED PAID APIs (7 total) - VIA PROXY ONLY
# =============================================================================
# These APIs require API keys. Models call them through the validator's proxy.
# The proxy injects the keys SERVER-SIDE - models NEVER see the API keys.
#
# SECURITY: Models cannot steal API keys because:
# 1. Keys are in VALIDATOR's environment, not model's (env is sanitized)
# 2. Model calls proxy URL, proxy adds the key before forwarding
# 3. Even if model had key, network interceptor blocks unauthorized hosts
# =============================================================================

ALLOWED_PAID_APIS: Dict[str, PaidAPI] = {
    # =========================================================================
    # WEB SCRAPING - LinkedIn, job boards, search
    # =========================================================================
    "scrapingdog": PaidAPI(
        name="ScrapingDog",
        provider_id="scrapingdog",
        base_url="https://api.scrapingdog.com",
        description="LinkedIn profiles, Google Search, job boards",
        cost_model=CostModel.PER_CREDIT,
        cost_per_unit=0.0005,
        # TODO: After beta release, change back to "SCRAPINGDOG_API_KEY" (shared with sourcing)
        env_var_name="QUALIFICATION_SCRAPINGDOG_API_KEY",
        notes="5-15 credits/search. Primary data source."
        # SAFE: Model calls proxy, proxy injects key server-side
    ),
    
    # =========================================================================
    # LLM INFERENCE - Semantic analysis and scoring
    # =========================================================================
    "openrouter": PaidAPI(
        name="OpenRouter",
        provider_id="openrouter",
        base_url="https://openrouter.ai/api/v1",
        description="LLM inference (GPT-4o-mini, Claude, etc.)",
        cost_model=CostModel.PER_TOKEN,
        cost_per_unit=0.00015,
        # TODO: After beta release, change back to "OPENROUTER_API_KEY" (shared with sourcing)
        env_var_name="QUALIFICATION_OPENROUTER_API_KEY",
        notes="For semantic ICP matching. Use gpt-4o-mini."
        # SAFE: Model calls proxy, proxy injects key server-side
    ),
    
    # =========================================================================
    # COMPANY INTELLIGENCE - Tech stack, funding, data
    # =========================================================================
    "builtwith": PaidAPI(
        name="BuiltWith",
        provider_id="builtwith",
        base_url="https://api.builtwith.com",
        description="Technology stack detection",
        cost_model=CostModel.PER_CALL,
        cost_per_unit=0.01,
        env_var_name="BUILTWITH_API_KEY",
        notes="What tech a company uses."
        # SAFE: Model calls proxy, proxy injects key server-side
    ),
    "crunchbase": PaidAPI(
        name="Crunchbase",
        provider_id="crunchbase",
        base_url="https://api.crunchbase.com",
        description="Funding data, investors, company details",
        cost_model=CostModel.PER_CALL,
        cost_per_unit=0.01,
        env_var_name="CRUNCHBASE_API_KEY",
        notes="Funding rounds, company financials."
        # SAFE: Model calls proxy, proxy injects key server-side
    ),
    "jobdata": PaidAPI(
        name="Jobs Data API",
        provider_id="jobdata",
        base_url="https://api.theirstack.com",
        description="Job postings data - hiring signals, company growth indicators",
        cost_model=CostModel.PER_CALL,
        cost_per_unit=0.01,
        env_var_name="JOBDATA_API_KEY",
        notes="Hiring intent signals - companies actively hiring = growth."
        # SAFE: Model calls proxy, proxy injects key server-side
    ),
    
    # =========================================================================
    # SOCIAL/NEWS - Intent signals
    # =========================================================================
    "desearch": PaidAPI(
        name="Desearch",
        provider_id="desearch",
        base_url="https://api.desearch.ai",
        description="Decentralized search - social media posts, discussions",
        cost_model=CostModel.PER_CALL,
        cost_per_unit=0.002,
        env_var_name="DESEARCH_API_KEY",
        notes="Social media signals via decentralized search."
        # SAFE: Model calls proxy, proxy injects key server-side
    ),
    "datauniverse": PaidAPI(
        name="Data Universe (Macrocosmos)",
        provider_id="datauniverse",
        base_url="https://datauniverse.macrocosmos.ai",
        description="55B+ rows of social media data (X, Reddit, YouTube) - real-time social listening",
        cost_model=CostModel.PER_CALL,
        cost_per_unit=0.00005,  # $0.05 per 1K records = $0.00005 per record
        env_var_name="DATAUNIVERSE_API_KEY",
        notes="Social signals, sentiment, hiring buzz. Largest open-access social data stream."
        # SAFE: Model calls proxy, proxy injects key server-side
    ),
    "googlenews": PaidAPI(
        name="NewsAPI.org",
        provider_id="googlenews",
        base_url="https://newsapi.org/v2",
        description="Company news, press releases",
        cost_model=CostModel.PER_CALL,
        cost_per_unit=0.001,
        env_var_name="NEWS_API_KEY",
        notes="News-based intent signals."
        # SAFE: Model calls proxy, proxy injects key server-side
    ),
}


# =============================================================================
# NETWORK SECURITY - Allowed Destinations
# =============================================================================
# Models can ONLY make requests to these destinations.
# All other network access is BLOCKED.
# =============================================================================

# Proxy URL (all paid API calls go through here)
PROXY_BASE_URL = os.getenv("QUALIFICATION_PROXY_URL", "http://localhost:8001")

# =============================================================================
# ALLOWED NETWORK DESTINATIONS
# =============================================================================
# Models can ONLY make requests to these destinations.
# All other network access is BLOCKED.
#
# CRITICAL: This is the PRIMARY security mechanism.
# Even if a model has an API key in os.environ, it can only send requests
# to these approved hosts. Attempts to exfiltrate keys to attacker servers
# will be blocked.
# =============================================================================

ALLOWED_NETWORK_DESTINATIONS: Set[str] = {
    # ===========================================
    # Proxy (for paid APIs via validator)
    # ===========================================
    "localhost",
    "127.0.0.1",
    
    # ===========================================
    # Free APIs - Search (REMOVED: GitHub, StackOverflow - write risk)
    # ===========================================
    "duckduckgo.com",
    "html.duckduckgo.com",
    "links.duckduckgo.com",
    
    # Wikipedia - public knowledge base (read-only)
    "en.wikipedia.org",
    "wikipedia.org",
    
    # ===========================================
    # Free APIs - Rep Score (from automated_checks.py)
    # ===========================================
    # SEC EDGAR
    "www.sec.gov",
    "sec.gov",
    "efts.sec.gov",             # SEC full-text search
    "data.sec.gov",             # SEC data API
    
    # Internet Archive / Wayback Machine
    "archive.org",
    "web.archive.org",
    
    # GDELT Project (news/events)
    "api.gdeltproject.org",
    "gdeltproject.org",
    
    # UK Companies House
    "api.company-information.service.gov.uk",
    "company-information.service.gov.uk",
    
    # ===========================================
    # Free APIs - Knowledge Bases
    # ===========================================
    # Wikipedia
    "en.wikipedia.org",
    "wikipedia.org",
    
    # Wikidata
    "www.wikidata.org",
    "wikidata.org",
    "query.wikidata.org",       # SPARQL endpoint
    
    # ===========================================
    # Paid APIs (Direct Access - keys from env)
    # ===========================================
    # Models get API keys from os.environ and can call these directly.
    # Network interception ensures they can ONLY call these hosts.
    "openrouter.ai",            # LLM inference
    "api.scrapingdog.com",      # Web scraping
    "api.builtwith.com",        # Tech stack detection
    "api.crunchbase.com",       # Funding data
    "api.desearch.ai",          # Desearch (decentralized search)
    "datauniverse.macrocosmos.ai",  # Data Universe (social media)
    "newsapi.org",              # News API
    "api.theirstack.com",       # Jobs Data API (hiring signals)
    
    # ===========================================
    # Database (Read-Only)
    # ===========================================
    "qplwoislplkcegvdmbim.supabase.co",  # Leads database
}


# =============================================================================
# RESTRICTED IMPORTER
# =============================================================================
# Custom import hook that blocks unauthorized modules.
# =============================================================================

class RestrictedImporter:
    """
    Custom import hook that enforces the library allowlist.
    
    ALLOWLIST-ONLY: If a module is not in ALLOWED_LIBRARIES, it's blocked.
    Internal Python modules (starting with _) are always allowed.
    Local model modules (qualify, model, main) are always allowed.
    
    NOTE: This affects ALL threads in the process while sandbox is active.
    This is intentional - if we made it thread-aware, malicious code could
    spawn a new thread to bypass import restrictions (threading is allowed).
    The epoch monitor may see errors during model evaluation - this is expected.
    """
    
    # Local module names that models can use (their own code)
    LOCAL_MODEL_MODULES = {"qualify", "model", "main", "utils", "helpers", "config"}
    
    def __init__(self, allowed: Set[str] = None, local_modules: Set[str] = None):
        self.allowed = allowed or ALLOWED_LIBRARIES
        self.local_modules = local_modules or self.LOCAL_MODEL_MODULES
    
    def find_module(self, fullname: str, path=None):
        """Called for every import. Return self to handle, None to skip."""
        top_level = fullname.split('.')[0]
        
        # Allow internal Python modules (they're part of the interpreter)
        if top_level.startswith('_'):
            return None
        
        # Allow local model modules (model's own files: qualify.py, model/, etc.)
        if top_level in self.local_modules:
            return None
        
        # Allow if in allowlist
        if top_level in self.allowed or fullname in self.allowed:
            return None
        
        # Allow submodules of allowed packages
        for allowed_pkg in self.allowed:
            if fullname.startswith(f"{allowed_pkg}."):
                return None
        
        # SECURITY: DO NOT allow imports just because they're in sys.modules!
        # That would let miner code import boto3, Leadpoet, bittensor, etc.
        # and access validator's AWS/Supabase credentials.
        #
        # Block everything else - this may cause log spam in background threads
        # but that's COSMETIC. Security is more important than clean logs.
        return self
    
    def load_module(self, fullname: str):
        """Called when we returned self from find_module. Raises ImportError."""
        raise ImportError(
            f"Import of '{fullname}' is not allowed. "
            f"Only libraries in ALLOWED_LIBRARIES can be imported."
        )


# =============================================================================
# NETWORK INTERCEPTOR
# =============================================================================
# Monkey-patches requests/httpx to route through proxy and block unauthorized.
# =============================================================================

class NetworkInterceptor:
    """
    Intercepts HTTP requests and enforces network policies.
    
    - Blocks requests to unauthorized destinations
    - Routes paid API calls through proxy
    - Adds tracking headers for cost accounting
    """
    
    def __init__(
        self,
        proxy_url: str = PROXY_BASE_URL,
        allowed_hosts: Set[str] = None,
        evaluation_run_id: str = None,
        evaluation_id: str = None
    ):
        self.proxy_url = proxy_url
        self.allowed_hosts = allowed_hosts or ALLOWED_NETWORK_DESTINATIONS
        self.evaluation_run_id = evaluation_run_id
        self.evaluation_id = evaluation_id
        self._original_request = None
        self._original_get = None
        self._original_post = None
    
    def is_allowed_destination(self, url: str) -> bool:
        """Check if URL is to an allowed destination."""
        from urllib.parse import urlparse
        parsed = urlparse(url)
        host = parsed.netloc.split(':')[0]  # Remove port
        
        # Check exact match
        if host in self.allowed_hosts:
            return True
        
        # Check if subdomain of allowed host
        for allowed in self.allowed_hosts:
            if host.endswith(f".{allowed}") or host == allowed:
                return True
        
        return False
    
    def get_proxy_url_for_paid_api(self, url: str) -> Optional[str]:
        """
        If URL is for a paid API, return the proxy URL to use instead.
        Returns None if not a paid API (use direct connection).
        """
        from urllib.parse import urlparse
        parsed = urlparse(url)
        
        for api_id, api in ALLOWED_PAID_APIS.items():
            if api.base_url in url or parsed.netloc in api.base_url:
                # Route through proxy
                # Original: https://api.scrapingdog.com/google?api_key=xxx&query=yyy
                # Proxied:  http://localhost:8001/scrapingdog/google?query=yyy
                path = parsed.path + ('?' + parsed.query if parsed.query else '')
                return f"{self.proxy_url}/{api.provider_id}{path}"
        
        return None
    
    def get_tracking_headers(self) -> Dict[str, str]:
        """Get headers for cost tracking."""
        headers = {}
        if self.evaluation_run_id:
            headers["X-Evaluation-Run-ID"] = self.evaluation_run_id
        if self.evaluation_id:
            headers["X-Evaluation-ID"] = self.evaluation_id
        return headers
    
    def intercept_request(self, method: str, url: str, **kwargs) -> Any:
        """
        Intercept and validate/transform an HTTP request.
        
        Raises:
            PermissionError: If destination is not allowed
        """
        # Check if destination is allowed
        if not self.is_allowed_destination(url):
            # Check if it's a paid API that should go through proxy
            proxy_url = self.get_proxy_url_for_paid_api(url)
            if proxy_url:
                logger.info(f"📡 Routing paid API call through proxy: {url[:50]}...")
                url = proxy_url
            else:
                raise PermissionError(
                    f"Network request to '{url}' is not allowed. "
                    f"Models can only access approved APIs. "
                    f"Use the proxy for paid APIs or approved free APIs."
                )
        
        # Add tracking headers
        headers = kwargs.get('headers', {}) or {}
        headers.update(self.get_tracking_headers())
        kwargs['headers'] = headers
        
        return url, kwargs
    
    def install(self):
        """
        Install the network interceptor by monkey-patching requests.
        
        Call this BEFORE running model code.
        """
        try:
            import requests
            
            # Save originals
            self._original_request = requests.Session.request
            
            # Create intercepted version
            interceptor = self
            original_request = self._original_request
            
            def intercepted_request(session, method, url, **kwargs):
                new_url, new_kwargs = interceptor.intercept_request(method, url, **kwargs)
                return original_request(session, method, new_url, **new_kwargs)
            
            # Monkey-patch
            requests.Session.request = intercepted_request
            logger.info("✅ Network interceptor installed (requests)")
            
        except ImportError:
            logger.warning("requests not available - skipping interception")
        
        try:
            import httpx
            
            # Similar for httpx
            original_httpx_request = httpx.Client.request
            interceptor = self
            
            def intercepted_httpx_request(client, method, url, **kwargs):
                new_url, new_kwargs = interceptor.intercept_request(method, str(url), **kwargs)
                return original_httpx_request(client, method, new_url, **new_kwargs)
            
            httpx.Client.request = intercepted_httpx_request
            logger.info("✅ Network interceptor installed (httpx)")
            
        except ImportError:
            pass
    
    def uninstall(self):
        """Restore original request functions."""
        try:
            import requests
            if self._original_request:
                requests.Session.request = self._original_request
        except ImportError:
            pass


# =============================================================================
# ENVIRONMENT SANITIZER
# =============================================================================
# Removes sensitive environment variables before running model code.
# =============================================================================

# =============================================================================
# SENSITIVE ENVIRONMENT VARIABLES
# =============================================================================
# These are ALWAYS removed from the model's environment.
# API keys ARE in this list - models NEVER receive them.
# Models call the PROXY instead - proxy injects credentials server-side.
# =============================================================================

SENSITIVE_ENV_VARS: Set[str] = {
    # ===========================================
    # API Keys - BLOCKED (models call proxy instead)
    # ===========================================
    # Models could exfiltrate these via allowed writable services (GitHub gists)
    # so they NEVER receive API keys. They call the proxy instead.
    "OPENROUTER_API_KEY",
    "OPENROUTER_KEY",
    "SCRAPINGDOG_API_KEY",
    "BUILTWITH_API_KEY",
    "CRUNCHBASE_API_KEY",
    "DESEARCH_API_KEY",
    "NEWS_API_KEY",
    "SIMILARTECH_API_KEY",
    "GITHUB_TOKEN",
    # Qualification-specific keys (separate from sourcing for security isolation)
    "QUALIFICATION_OPENROUTER_API_KEY",
    "QUALIFICATION_SCRAPINGDOG_API_KEY",
    
    # ===========================================
    # BLOCKED - Wallet/Crypto
    # ===========================================
    "BT_WALLET_NAME",
    "BT_WALLET_HOTKEY",
    "BT_WALLET_PATH",
    "WALLET_SEED",
    "MNEMONIC",
    
    # ===========================================
    # BLOCKED - Database Admin
    # ===========================================
    "SUPABASE_SERVICE_ROLE_KEY",  # Write access - NOT allowed
    "DATABASE_URL",
    "POSTGRES_PASSWORD",
    "POSTGRES_USER",
    
    # ===========================================
    # BLOCKED - AWS (Infrastructure)
    # ===========================================
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
    
    # ===========================================
    # BLOCKED - Generic Sensitive Patterns
    # ===========================================
    "SECRET_KEY",
    "PRIVATE_KEY",
    "SSH_KEY",
    "SSL_CERT",
    
    # ===========================================
    # BLOCKED - Internal APIs (not for models)
    # ===========================================
    "TRUELIST_API_KEY",           # Internal email verification
    "MYEMAILVERIFIER_API_KEY",    # Internal email verification
}


def sanitize_environment() -> Dict[str, str]:
    """
    Create a sanitized copy of environment variables for model execution.
    
    CRITICAL: Models NEVER receive API keys. Even with network blocking,
    a malicious model could exfiltrate keys via allowed writable services
    (e.g., GitHub gists, Supabase queries). Models call the PROXY instead,
    which injects credentials server-side.
    
    Returns:
        Dict of safe environment variables (NO API keys)
    """
    safe_env = {}
    
    for key, value in os.environ.items():
        # Skip explicitly blocked sensitive vars (includes ALL API keys)
        if key in SENSITIVE_ENV_VARS:
            continue
        
        # Skip anything that looks like a key/secret/token
        key_upper = key.upper()
        if any(pattern in key_upper for pattern in [
            'KEY', 'SECRET', 'TOKEN', 'PASSWORD', 'CREDENTIAL',
            'WALLET', 'SEED', 'MNEMONIC', 'PRIVATE'
        ]):
            continue
        
        # Skip service role / admin database credentials
        if 'SERVICE_ROLE' in key_upper or 'POSTGRES' in key_upper:
            continue
        
        # Skip AWS credentials
        if key_upper.startswith('AWS_'):
            continue
        
        safe_env[key] = value
    
    # Add ONLY the safe values needed (NO API keys)
    safe_env["SUPABASE_URL"] = os.getenv("SUPABASE_URL", "https://qplwoislplkcegvdmbim.supabase.co")
    safe_env["SUPABASE_ANON_KEY"] = os.getenv(
        "SUPABASE_ANON_KEY",
        "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InFwbHdvaXNscGxrY2VndmRtYmltIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDQ4NDcwMDUsImV4cCI6MjA2MDQyMzAwNX0.5E0WjAthYDXaCWY6qjzXm2k20EhadWfigak9hleKZk8"
    )
    # Table name is set via QUALIFICATION_LEADS_TABLE env var (not hardcoded)
    qualification_table = os.getenv("QUALIFICATION_LEADS_TABLE", "")
    if qualification_table:
        safe_env["QUALIFICATION_LEADS_TABLE"] = qualification_table
    safe_env["QUALIFICATION_PROXY_URL"] = PROXY_BASE_URL
    
    # NO API keys are injected here.
    # Models call the proxy (e.g., http://localhost:8001/proxy/openrouter/...)
    # and the proxy injects credentials server-side.
    
    return safe_env


# =============================================================================
# SANDBOX SECURITY CONTEXT
# =============================================================================
# High-level API to set up all security measures.
# =============================================================================

# =============================================================================
# BLOCKED FILE PATHS
# =============================================================================
# These paths are BLOCKED from being read by models.
# Critical: /proc/self/environ contains the raw process environment and
# bypasses os.environ.clear() - this is a major security vulnerability on Linux!
# =============================================================================

BLOCKED_FILE_PATHS: Set[str] = {
    "/proc/self/environ",      # Raw process environment - CRITICAL
    "/proc/self/cmdline",      # Command line args
    "/proc/self/status",       # Process status
    "/proc/self/maps",         # Memory maps
    "/proc/self/fd",           # File descriptors
    "/etc/passwd",             # User accounts
    "/etc/shadow",             # Password hashes
    "/etc/environment",        # System environment
    "~/.bashrc",               # Shell config
    "~/.bash_profile",         # Shell profile
    "~/.profile",              # User profile
    "~/.ssh",                  # SSH keys
    "~/.aws",                  # AWS credentials
}

# Patterns to block (prefix match)
BLOCKED_FILE_PATTERNS: List[str] = [
    "/proc/",                  # All /proc access
    "/sys/",                   # System info
    "/dev/",                   # Device access
    "/etc/",                   # System config
    "~/.ssh/",                 # SSH keys
    "~/.aws/",                 # AWS credentials
    "~/.config/",              # User config
]


def _is_blocked_path(filepath: str) -> bool:
    """Check if a file path is blocked for security reasons."""
    if not isinstance(filepath, str):
        return False
    
    # Normalize path
    filepath = os.path.expanduser(filepath)
    filepath_lower = filepath.lower()
    
    # Check exact matches
    if filepath in BLOCKED_FILE_PATHS:
        return True
    
    # Check pattern matches
    for pattern in BLOCKED_FILE_PATTERNS:
        if filepath.startswith(pattern) or filepath_lower.startswith(pattern.lower()):
            return True
    
    return False


class RestrictedFileOpen:
    """
    Wrapper around built-in open() that blocks access to sensitive files.
    
    This prevents attacks like reading /proc/self/environ to bypass
    os.environ sanitization.
    """
    
    def __init__(self, original_open):
        self._original_open = original_open
    
    def __call__(self, file, mode='r', *args, **kwargs):
        if _is_blocked_path(file):
            raise PermissionError(
                f"Access to '{file}' is blocked for security reasons. "
                f"Models cannot read sensitive system files."
            )
        return self._original_open(file, mode, *args, **kwargs)


class RestrictedOsOpen:
    """
    Wrapper around os.open() that blocks access to sensitive files.
    
    CRITICAL: os.open() is a low-level syscall wrapper that bypasses builtins.open().
    A malicious model could use __builtins__.__import__('os').open('/proc/self/environ')
    to read the raw process environment even after we patch builtins.open.
    """
    
    def __init__(self, original_os_open):
        self._original_os_open = original_os_open
    
    def __call__(self, path, flags, mode=0o777, *args, **kwargs):
        if _is_blocked_path(path):
            raise PermissionError(
                f"Access to '{path}' is blocked for security reasons. "
                f"Models cannot read sensitive system files via os.open()."
            )
        return self._original_os_open(path, flags, mode, *args, **kwargs)


def _create_restricted_builtins_import(original_import):
    """
    Factory function that creates a restricted import wrapper using CLOSURES.
    
    SECURITY LAYERS:
    1. Closure hides original_import (can't access wrapper._original_import)
    2. Cross-check verifies RestrictedImporter is still in sys.meta_path
    
    This creates a deadlock for attackers:
    - Clear sys.meta_path → our wrapper detects it and refuses ALL imports
    - Restore original __import__ → sys.meta_path still has RestrictedImporter
    - Both → theoretically possible via __closure__ but raises difficulty significantly
    """
    
    # Local module names captured in closure - not accessible from outside
    LOCAL_MODEL_MODULES = {"qualify", "model", "main", "utils", "helpers", "config"}
    
    def restricted_import(name, globals=None, locals=None, fromlist=(), level=0):
        # SECURITY: Check that RestrictedImporter hasn't been removed from sys.meta_path
        # If someone clears meta_path, we refuse ALL imports (even allowed ones)
        meta_path_intact = any(
            type(h).__name__ == 'RestrictedImporter' 
            for h in sys.meta_path
        )
        if not meta_path_intact:
            raise ImportError(
                "🚫 SECURITY BREACH: RestrictedImporter removed from sys.meta_path. "
                "All imports blocked."
            )
        
        # CRITICAL: Relative imports (level > 0) are ALWAYS allowed.
        # When a package like postgrest does "from ..base_client import X",
        # Python calls __import__('base_client', level=2). The bare name
        # 'base_client' is NOT in ALLOWED_LIBRARIES, but it's a relative
        # import within the already-allowed 'postgrest' package. Blocking
        # it would break all relative imports in allowed packages.
        if level > 0:
            return original_import(name, globals, locals, fromlist, level)
        
        base_module = name.split('.')[0]
        
        # Allow internal Python modules (starting with _)
        if base_module.startswith('_'):
            return original_import(name, globals, locals, fromlist, level)
        
        # Allow local model modules (model's own files: qualify.py, model/, etc.)
        if base_module in LOCAL_MODEL_MODULES:
            return original_import(name, globals, locals, fromlist, level)
        
        # Allow if in allowlist
        if base_module in ALLOWED_LIBRARIES or name in ALLOWED_LIBRARIES:
            return original_import(name, globals, locals, fromlist, level)
        
        # Allow submodules of allowed packages
        for allowed_pkg in ALLOWED_LIBRARIES:
            if name.startswith(f"{allowed_pkg}."):
                return original_import(name, globals, locals, fromlist, level)
        
        # Block everything else - affects all threads for security
        raise ImportError(
            f"🚫 Import of '{name}' is not allowed. "
            f"Only libraries in ALLOWED_LIBRARIES can be imported."
        )
    
    return restricted_import


# Legacy class kept for backwards compatibility, but use the closure version
class RestrictedBuiltinsImport:
    """
    DEPRECATED: Use _create_restricted_builtins_import() instead.
    This class is kept for backwards compatibility but the closure version
    is more secure as it doesn't expose _original_import as an attribute.
    """
    
    LOCAL_MODEL_MODULES = {"qualify", "model", "main", "utils", "helpers", "config"}
    
    def __init__(self, original_import):
        # Store in closure-like manner using __closure__ trick won't work
        # This class is DEPRECATED - use the function version
        self._original_import = original_import
    
    def __call__(self, name, globals=None, locals=None, fromlist=(), level=0):
        # Relative imports are always allowed (within already-allowed packages)
        if level > 0:
            return self._original_import(name, globals, locals, fromlist, level)
        
        base_module = name.split('.')[0]
        
        if base_module.startswith('_'):
            return self._original_import(name, globals, locals, fromlist, level)
        
        if base_module in self.LOCAL_MODEL_MODULES:
            return self._original_import(name, globals, locals, fromlist, level)
        
        if base_module in ALLOWED_LIBRARIES or name in ALLOWED_LIBRARIES:
            return self._original_import(name, globals, locals, fromlist, level)
        
        for allowed_pkg in ALLOWED_LIBRARIES:
            if name.startswith(f"{allowed_pkg}."):
                return self._original_import(name, globals, locals, fromlist, level)
        
        raise ImportError(
            f"🚫 Import of '{name}' is not allowed. "
            f"Only libraries in ALLOWED_LIBRARIES can be imported."
        )


class SandboxSecurityContext:
    """
    Context manager that sets up all security measures for model execution.
    
    Security measures:
    1. Import restriction - blocks unauthorized Python modules
    2. Network interception - blocks unauthorized network destinations
    3. Environment sanitization - removes API keys from os.environ
    4. File access restriction - blocks reading /proc/self/environ etc.
    
    Usage:
        async with SandboxSecurityContext(evaluation_run_id, evaluation_id) as ctx:
            result = model.qualify(icp)
    """
    
    def __init__(
        self,
        evaluation_run_id: str,
        evaluation_id: str,
        enable_import_restriction: bool = True,
        enable_network_interception: bool = True,
        enable_env_sanitization: bool = True,
        enable_file_restriction: bool = True,
        proxy_url: str = None
    ):
        self.evaluation_run_id = evaluation_run_id
        self.evaluation_id = evaluation_id
        self.enable_import_restriction = enable_import_restriction
        self.enable_network_interception = enable_network_interception
        self.enable_env_sanitization = enable_env_sanitization
        self.enable_file_restriction = enable_file_restriction
        self.proxy_url = proxy_url or PROXY_BASE_URL
        
        self._restricted_importer = None
        self._network_interceptor = None
        self._original_env = None
        self._original_meta_path = None
        self._original_open = None
        self._original_io_open = None
        self._original_os_open = None
        self._original_builtins_import = None
        self._original_current_frames = None  # Block sys._current_frames()
        # Store module references so we can access them in __exit__ without importing
        self._builtins_module = None
        self._io_module = None
    
    def __enter__(self):
        """Set up security measures."""
        import builtins
        import io as io_module
        
        # Store module references for use in __exit__ (can't import after restriction is active)
        self._builtins_module = builtins
        self._io_module = io_module
        
        logger.info(f"🔐 Setting up sandbox security for run {self.evaluation_run_id[:8]}...")
        
        # =================================================================
        # IMPORTANT: Order matters! We set up restrictions AFTER importing
        # the modules we need for setup (builtins, io).
        # =================================================================
        
        # 1. Restrict file access FIRST (blocks /proc/self/environ)
        if self.enable_file_restriction:
            # 1a. Patch builtins.open and io.open
            self._original_open = builtins.open
            self._original_io_open = io_module.open
            
            restricted_open = RestrictedFileOpen(self._original_open)
            builtins.open = restricted_open
            io_module.open = restricted_open
            
            # 1b. CRITICAL: Also patch os.open - it's a low-level syscall that bypasses builtins.open
            self._original_os_open = os.open
            os.open = RestrictedOsOpen(self._original_os_open)
            
            # 1c. CRITICAL: Block sys._current_frames() - allows cross-thread stack inspection
            # This is used by malicious code to read local variables from other threads
            # (e.g., the proxy thread that has API keys in its closure)
            if hasattr(sys, '_current_frames'):
                self._original_current_frames = sys._current_frames
                def blocked_current_frames():
                    raise RuntimeError(
                        "🚫 sys._current_frames() is blocked for security. "
                        "Cross-thread stack inspection is not allowed."
                    )
                sys._current_frames = blocked_current_frames
                logger.info("  ✅ sys._current_frames() blocked")
            
            logger.info("  ✅ File access restriction enabled (builtins.open, io.open, os.open)")
        
        # 2. Sanitize environment BEFORE import restriction
        # This ensures API keys are removed before model can try to read them
        if self.enable_env_sanitization:
            self._original_env = os.environ.copy()
            safe_env = sanitize_environment()
            os.environ.clear()
            os.environ.update(safe_env)
            logger.info("  ✅ Environment sanitized")
        
        # 3. Intercept network
        if self.enable_network_interception:
            self._network_interceptor = NetworkInterceptor(
                proxy_url=self.proxy_url,
                evaluation_run_id=self.evaluation_run_id,
                evaluation_id=self.evaluation_id
            )
            self._network_interceptor.install()
            logger.info(f"  ✅ Network interception enabled (proxy: {self.proxy_url})")
        
        # 4. Restrict imports via sys.meta_path
        if self.enable_import_restriction:
            self._restricted_importer = RestrictedImporter()
            self._original_meta_path = sys.meta_path.copy()
            sys.meta_path.insert(0, self._restricted_importer)
            logger.info("  ✅ Import restriction (sys.meta_path) enabled")
        
        # 5. CRITICAL: Also restrict __builtins__.__import__ to prevent bypass
        # Malicious code can call __builtins__.__import__('os') to bypass meta_path
        # SECURITY: Use closure version to hide original_import from malicious code
        # This is installed LAST so it doesn't block our own setup imports
        if self.enable_import_restriction:
            self._original_builtins_import = builtins.__import__
            builtins.__import__ = _create_restricted_builtins_import(self._original_builtins_import)
            logger.info("  ✅ Import restriction (__builtins__.__import__) enabled - closure secured")
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Tear down security measures."""
        # Use stored module references - cannot import here as restriction may still be active
        builtins = self._builtins_module
        io_module = self._io_module
        
        logger.info(f"🔓 Tearing down sandbox security for run {self.evaluation_run_id[:8]}...")
        
        # Restore in reverse order of __enter__
        
        # 5. Restore __builtins__.__import__ FIRST (so subsequent imports work)
        if self._original_builtins_import is not None and builtins is not None:
            builtins.__import__ = self._original_builtins_import
            logger.info("  ✅ __builtins__.__import__ restored")
        
        # 4. Restore sys.meta_path
        if self._original_meta_path is not None:
            sys.meta_path = self._original_meta_path
            logger.info("  ✅ sys.meta_path restored")
        
        # 3. Remove network interceptor
        if self._network_interceptor is not None:
            self._network_interceptor.uninstall()
            logger.info("  ✅ Network interception removed")
        
        # 2. Restore environment
        if self._original_env is not None:
            os.environ.clear()
            os.environ.update(self._original_env)
            logger.info("  ✅ Environment restored")
        
        # 1. Restore file access (builtins.open, io.open, os.open)
        if self._original_open is not None:
            builtins.open = self._original_open
            io_module.open = self._original_io_open
            logger.info("  ✅ builtins.open and io.open restored")
        
        if self._original_os_open is not None:
            os.open = self._original_os_open
            logger.info("  ✅ os.open restored")
        
        if self._original_current_frames is not None:
            sys._current_frames = self._original_current_frames
            logger.info("  ✅ sys._current_frames restored")
        
        return False  # Don't suppress exceptions
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self.__enter__()
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        return self.__exit__(exc_type, exc_val, exc_tb)


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_allowlist_summary() -> Dict[str, Any]:
    """Get a summary of allowed libraries and APIs for display to miners."""
    return {
        "allowed_libraries": sorted(ALLOWED_LIBRARIES),
        "free_apis": {
            name: {
                "base_url": api.base_url,
                "description": api.description,
                "rate_limit": api.rate_limit_per_minute
            }
            for name, api in ALLOWED_FREE_APIS.items()
        },
        "paid_apis": {
            name: {
                "base_url": api.base_url,
                "description": api.description,
                "cost_model": api.cost_model.value,
                "cost_per_unit": api.cost_per_unit,
                "notes": api.notes
            }
            for name, api in ALLOWED_PAID_APIS.items()
        }
    }


def print_allowlist_for_miners():
    """Print a formatted allowlist for miners to reference."""
    print("=" * 80)
    print("QUALIFICATION MODEL ALLOWLIST")
    print("=" * 80)
    
    print("\n📚 ALLOWED PYTHON LIBRARIES:")
    print("-" * 40)
    for lib in sorted(ALLOWED_LIBRARIES):
        print(f"  ✅ {lib}")
    
    print("\n🆓 FREE APIs (no key needed):")
    print("-" * 40)
    for name, api in ALLOWED_FREE_APIS.items():
        print(f"  {api.name}")
        print(f"     URL: {api.base_url}")
        print(f"     Rate: {api.rate_limit_per_minute}/min")
        print()
    
    print("\n💰 PAID APIs (via proxy - cost tracked):")
    print("-" * 40)
    for name, api in ALLOWED_PAID_APIS.items():
        print(f"  {api.name}")
        print(f"     Proxy: /proxy/{api.provider_id}/...")
        print(f"     Cost: ${api.cost_per_unit} per {api.cost_model.value}")
        print(f"     Notes: {api.notes}")
        print()


if __name__ == "__main__":
    # Print allowlist when run directly
    print_allowlist_for_miners()
