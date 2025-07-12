import asyncio
import csv
import json
import os
import re
import time
import argparse
import gzip
import hashlib
import signal
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Set
from datetime import datetime, timedelta
import aiohttp
from aiohttp import web
import aiodns
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from prometheus_client import Counter, Histogram, start_http_server, Gauge
import functools
import dns.asyncresolver
import structlog
import redis.asyncio as redis
import itertools
from dotenv import load_dotenv
import logging
import random
from collections import OrderedDict       # NEW
import requests, json, textwrap

class EmailValidator:
    """Email validation class with disposable domain checking."""
    
    def __init__(self, disposable_domains_file: str = "shared/disposable_email_blocklist.conf"):
        """Initialize EmailValidator with disposable domains file."""
        # Always resolve to absolute path relative to this script
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.disposable_domains_file = os.path.abspath(os.path.join(base_dir, disposable_domains_file))
        self._disposable_domains: Optional[Set[str]] = None
        self._email_regex = re.compile(
            r"^[a-zA-Z0-9.!#$%&'*+/=?^_`{|}~-]+@[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?(?:\.[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?)*$"
        )
        # Async cache for MX lookups
        self._mx_cache = {}
        self._mx_cache_maxsize = 10_000
    
    def load_disposable_domains(self) -> Set[str]:
        """Load disposable domains from the configuration file."""
        if self._disposable_domains is not None:
            return self._disposable_domains
            
        try:
            with open(self.disposable_domains_file, "r", encoding="utf-8") as f:
                domains = {line.strip().lower() for line in f if line.strip() and not line.startswith('#')}
                self._disposable_domains = domains
                return domains
        except FileNotFoundError:
            print(f"Warning: {self.disposable_domains_file} not found, using empty set")
            self._disposable_domains = set()
            return set()
        except Exception as e:
            print(f"Error loading disposable domains: {e}")
            self._disposable_domains = set()
            return set()
    
    def is_valid_format(self, email: str) -> bool:
        """Check if email has valid format."""
        if not email or not isinstance(email, str):
            return False
        email = email.lower().strip()
        return bool(self._email_regex.match(email))
    
    def is_disposable_domain(self, email: str) -> bool:
        """Check if email domain is in disposable domains list."""
        if not self.is_valid_format(email):
            return False
        
        domain = email.split("@")[-1]
        disposable_domains = self.load_disposable_domains()
        return domain in disposable_domains
    
    async def mx_exists(self, domain: str) -> bool:
        """Check if domain has MX records with async caching."""
        # Check cache first
        if domain in self._mx_cache:
            return self._mx_cache[domain]

        def _evict_oldest_if_needed():
            """Helper to evict oldest cache entry if needed."""
            if len(self._mx_cache) >= self._mx_cache_maxsize:
                oldest_key = next(iter(self._mx_cache))
                del self._mx_cache[oldest_key]

        # Perform MX lookup
        try:
            resolver = dns.asyncresolver.Resolver()
            answers = await resolver.resolve(domain, 'MX')
            result = bool(answers)

            _evict_oldest_if_needed()
            self._mx_cache[domain] = result
            return result
        except (dns.resolver.NXDOMAIN, dns.resolver.NoAnswer):
            # Cache definitive negative responses
            _evict_oldest_if_needed()
            self._mx_cache[domain] = False
            return False
        except Exception:
            # Don't cache temporary failures (network issues, timeouts, etc.)
            result = False
            return result
    
    async def is_legitimate(self, email: str) -> bool:
        """Check if email is legitimate (valid format, not disposable, has MX records)."""
        if not self.is_valid_format(email):
            return False
        
        if self.is_disposable_domain(email):
            return False
        
        domain = email.split("@")[-1]
        has_mx = await self.mx_exists(domain)
        return has_mx
    
    def get_disposable_domains(self) -> Set[str]:
        """Get the set of disposable domains."""
        return self.load_disposable_domains()
    
    def reload_disposable_domains(self) -> None:
        """Reload disposable domains from file (clears cache)."""
        self._disposable_domains = None
        self.load_disposable_domains()
    
    def clear_mx_cache(self) -> None:
        """Clear the MX lookup cache."""
        self._mx_cache.clear()

# Global instance for use across the application
email_validator = EmailValidator()

# Load .env file
load_dotenv()

# Ensure logs directory exists
Path("logs").mkdir(exist_ok=True)

# Configure structlog with comprehensive processors for both console and file output
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

# Configure standard library logging to use structlog
logging.basicConfig(
    level=logging.WARNING,  # Changed from INFO to WARNING to suppress verbose logs
    format="%(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/pipeline.log", mode="a", encoding="utf-8")
    ]
)

logger = structlog.get_logger()

# Cross-platform file locking
try:
    import fcntl
    def lock_file(file_obj, exclusive=True):
        """Lock file for exclusive or shared access (Unix)."""
        if exclusive:
            fcntl.flock(file_obj.fileno(), fcntl.LOCK_EX)
        else:
            fcntl.flock(file_obj.fileno(), fcntl.LOCK_SH)
    
    def unlock_file(file_obj):
        """Unlock file (Unix)."""
        fcntl.flock(file_obj.fileno(), fcntl.LOCK_UN)
        
except ImportError:
    # Windows fallback - no file locking, but log warning
    def lock_file(file_obj, exclusive=True):
        """No-op lock for Windows (not supported)."""
        logger.warning("file_locking_not_available_windows", note="Multiple processes may cause race conditions")
        pass
    
    def unlock_file(file_obj):
        """No-op unlock for Windows."""
        pass

# ------------------------------------------------------------------
#  Guarantee that the domains CSV exists (header: domain,last_scraped)
# ------------------------------------------------------------------
def ensure_domains_file(path: str):
    csv_path = Path(path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if not csv_path.exists():
        csv_path.write_text("domain,last_scraped\n", encoding="utf-8")
        logger.info("domains_csv_created", file=str(csv_path))

# ---------------- CLI ----------------
def parse_args(argv: list | None = None):
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Firecrawl Lead Sourcing Pipeline")
    parser.add_argument("--domains", default="data/domains.csv", help="Path to domains CSV file")
    parser.add_argument("--max-pages-per-domain", type=int, default=50, help="Maximum pages to crawl per domain")
    parser.add_argument("--batch-size", type=int, default=10, help="Batch size for enrichment")
    parser.add_argument("--flush-sec", type=int, default=60, help="Flush timeout in seconds")
    parser.add_argument("--redis-url", default="redis://localhost:6379", help="Redis URL")
    parser.add_argument("--web-port", type=int, default=8080, help="Web server port")
    parser.add_argument("--metrics-port", type=int, default=9090, help="Metrics server port")
    parser.add_argument("--firecrawl-key", default=os.getenv("FIRECRAWL_API_KEY"),
                       help="Firecrawl API key (defaults to FIRECRAWL_API_KEY env var)")
    parser.add_argument("--hunter-key", default=os.getenv("HUNTER_API_KEY"),
                       help="Hunter API key (defaults to HUNTER_API_KEY env var)")
    parser.add_argument("--openrouter_key",
                        default=os.getenv("OPENROUTER_API_KEY"),
                        help="OpenRouter API key (defaults to OPENROUTER_API_KEY env var)")
    
    args = parser.parse_args(argv)
    
    # Only log and check API keys if we're running as main (not being imported)
    if __name__ == "__main__":
        logger.info("api_configuration_status",
                    required_keys_present=all([
                        args.firecrawl_key and args.firecrawl_key.strip(),
                        args.hunter_key and args.hunter_key.strip(),
                        args.openrouter_key and args.openrouter_key.strip()
                    ]))

        if not (args.firecrawl_key and args.firecrawl_key.strip()):
            logger.error("missing_firecrawl_key", note="FIRECRAWL_API_KEY environment variable is required")
            sys.exit(1)
            
        if not (args.hunter_key and args.hunter_key.strip()):
            logger.error("missing_hunter_key", note="HUNTER_API_KEY environment variable is required")
            sys.exit(1)
            
        if not (args.openrouter_key and args.openrouter_key.strip()):
            logger.error("missing_openrouter_key", note="OPENROUTER_API_KEY environment variable is required")
            sys.exit(1)
    
    return args

# ---------------- Global State ----------------
if __name__ == "__main__":
    ARGS = parse_args()          # honour real CLI flags
else:
    # When imported as a module, don't check API keys immediately
    # Only check them when functions are actually called
    ARGS = parse_args(argv=[])   # imported: ignore sys.argv

#  Make sure the CSV is present BEFORE anything tries to read/write it
ensure_domains_file(ARGS.domains)

# Only create headers if we have a valid API key
if ARGS.firecrawl_key and ARGS.firecrawl_key.strip():
    HEADERS_FC = {"Authorization": f"Bearer {ARGS.firecrawl_key}"}
else:
    HEADERS_FC = None

# Global state for graceful shutdown
shutdown_requested = False
current_batch = []
outbox_file = None

def signal_handler(signum, frame):
    """Handle SIGTERM for graceful shutdown."""
    global shutdown_requested
    logger.info("shutdown_signal_received", signal=signum)
    shutdown_requested = True

# Register signal handler
signal.signal(signal.SIGTERM, signal_handler)

# ---------------- Prometheus Metrics ----------------
LEADS_LEGIT_TOTAL = Counter('leads_legit_total', 'Total legitimate leads processed')
ENRICH_CALLS_TOTAL = Counter('enrich_calls_total', 'Total enrichment API calls')
LEAD_LATENCY_SECONDS = Histogram('lead_latency_seconds', 'Time from extract to enriched')
CRAWL_CREDITS_SPENT_TOTAL = Counter('crawl_credits_spent_total', 'Total crawl credits spent')
LEGITIMACY_CHECKED_TOTAL = Counter('legitimacy_checked_total', 'Total emails checked for legitimacy')
LEGITIMACY_PASS_TOTAL = Counter('legitimacy_pass_total', 'Total emails passing legitimacy check')
BATCH_FLUSH_TOTAL = Counter('batch_flush_total', 'Total batch flushes', ['reason'])
BATCH_SIZE_GAUGE = Gauge('batch_size', 'Configured batch size for enrichment')
FLUSH_SEC_GAUGE = Gauge('flush_sec', 'Configured flush timeout in seconds')
ENRICH_RECORDS_TOTAL = Counter('enrich_records_total', 'Total records enriched')
ENRICH_CREDITS_TOTAL = Counter('enrich_credits_total', 'Total enrichment credits spent')
FIRECRAWL_ENRICH_RECORDS_TOTAL = Counter('firecrawl_enrich_records_total', 'Total records enriched via Firecrawl')

# ---------------- Legitimacy helpers ----------------
async def is_legit(email: str, mx_check=None) -> bool:
    """Check if email is legitimate using EmailValidator."""
    LEGITIMACY_CHECKED_TOTAL.inc()
    
    # Use EmailValidator for all validation
    if not email_validator.is_valid_format(email):
        return False
    
    if email_validator.is_disposable_domain(email):
        return False
    
    # Use provided mx_check or EmailValidator's mx_exists
    if mx_check is None:
        mx_check = email_validator.mx_exists
    
    domain = email.split("@")[-1]
    has_mx = await mx_check(domain)
    if has_mx:
        LEGITIMACY_PASS_TOTAL.inc()
    return has_mx

# ---------------- Crawl & extract ----------------
EXTRACT_SCHEMA = {
    "email": "",
    "first_name": "",
    "last_name": "",
    "title": "",
    "company": "",
    "url": "",
    "source_domain": "",
    "timestamp": "",
    "enrichment": {}
}

# URL patterns for contact extraction
INCLUDE_PATTERNS = ["/contact", "/about", "/team", "/legal", "/privacy", "/impressum"]
EXCLUDE_PATTERNS = ["/admin", "/login", "/api", "/assets", "/static"]

def should_include_url(url: str) -> bool:
    """Check if URL should be included for contact extraction."""
    for pattern in INCLUDE_PATTERNS:
        if re.search(pattern, url, re.I):
            return True
    return False

def should_exclude_url(url: str) -> bool:
    """Check if URL should be excluded."""
    for pattern in EXCLUDE_PATTERNS:
        if re.search(pattern, url, re.I):
            return True
    return False

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
async def extract_contacts(session: aiohttp.ClientSession, url: str, source_domain: str) -> List[Dict[str, Any]]:
    """Extract contacts from a single URL."""
    try:
        async with session.post(
            "https://api.firecrawl.dev/v1/extract",
            headers=HEADERS_FC,
            json={"urls": [url]},
            timeout=aiohttp.ClientTimeout(total=30)
        ) as response:
            response.raise_for_status()
            result = await response.json()
            
            contacts = []
            
            # Create basic contact record for contact pages
            if "/contact" in url:
                domain_name = source_domain.replace('https://', '').replace('http://', '')
                contact_data = {
                    "email": f"contact@{domain_name}",
                    "first_name": "Contact",
                    "last_name": "Team",
                    "title": "Contact",
                    "company": domain_name,
                    "url": url,
                    "source_domain": source_domain,
                    "timestamp": datetime.utcnow().isoformat(),
                    "enrichment": {}
                }
                contacts.append(contact_data)
            
            # If API returns contact data, extract it
            if "data" in result and "contacts" in result["data"]:
                for contact in result["data"]["contacts"]:
                    full_name = contact.get("name", "")
                    first_name = ""
                    last_name = ""
                    if full_name:
                        name_parts = full_name.strip().split()
                        if len(name_parts) >= 2:
                            first_name = name_parts[0]
                            last_name = " ".join(name_parts[1:])
                        else:
                            first_name = full_name
                    
                    contact_data = {
                        "email": contact.get("email", ""),
                        "first_name": first_name,
                        "last_name": last_name,
                        "title": contact.get("title", ""),
                        "company": contact.get("company", ""),
                        "url": url,
                        "source_domain": source_domain,
                        "timestamp": datetime.utcnow().isoformat(),
                        "enrichment": {}
                    }
                    contacts.append(contact_data)
            
            return contacts
            
    except Exception as e:
        print(f"Error extracting contacts from {url}: {e}")
        return []

async def crawl_domain(domain: str) -> List[Dict[str, Any]]:
    """Crawl a domain and extract contact information."""
    start_time = time.time()
    
    try:
        # Check if domain was recently scraped
        if is_recently_scraped(domain):
            return []
        
        # Crawl domain
        crawl_data = await firecrawl_crawl(domain)
        if not crawl_data:
            # Mark domain as invalid since crawl failed
            update_last_scraped(domain, "invalid")
            return []
        
        # Extract eligible URLs from crawl data
        eligible_urls = []
        for page in crawl_data:
            url = page.get("metadata", {}).get("sourceURL")
            if url and should_include_url(url):
                eligible_urls.append(url)
        
        if not eligible_urls:
            # Mark domain as valid but no leads found
            update_last_scraped(domain, "valid")
            return []
        
        # Extract contacts from eligible URLs
        contacts = await extract_contacts_from_urls(eligible_urls, domain)
        
        # Filter for legitimate emails
        legitimate_contacts = []
        for contact in contacts:
            email = contact.get("email", "")
            if email and await is_legit(email):
                legitimate_contacts.append(contact)
        
        # Update domain status based on results
        if legitimate_contacts:
            update_last_scraped(domain, "valid")
        else:
            update_last_scraped(domain, "valid")  # Still valid, just no leads
        
        crawl_time = time.time() - start_time
        
        return legitimate_contacts
        
    except Exception as e:
        # Mark domain as invalid due to error
        update_last_scraped(domain, "invalid")
        return []

def load_domains(domains_file: str) -> List[str]:
    """Load domains from CSV file, filtering out deactivated/invalid domains."""
    ensure_domains_file(domains_file)
    try:
        domains = []
        with open(domains_file, 'r', encoding='utf-8') as f:
            # Read all lines and extract domains (remove trailing comma)
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):  # Skip empty lines and comments
                    parts = line.split(',')
                    if len(parts) >= 1:
                        domain = parts[0].rstrip(',')
                        
                        # Check if domain is marked as deactivated/invalid
                        if len(parts) >= 3 and "deactivated/invalid" in parts[2]:
                            logger.debug("domain_skipped_deactivated", domain=domain)
                            continue
                        
                        domains.append(domain)
        
        # Only log the count, not the verbose JSON
        print(f"📋 Loaded {len(domains)} valid domains from {domains_file}")
        return domains
    except Exception as e:
        print(f"❌ Error loading domains: {e}")
        return []

def get_random_domains(domains_file: str, sample_size: int = 1) -> List[str]:
    """Get a random sample of domains from the CSV file."""
    all_domains = load_domains(domains_file)
    if not all_domains:
        return []
    
    # Sample without replacement
    sample_size = min(sample_size, len(all_domains))
    return random.sample(all_domains, sample_size)

def update_last_scraped(domain: str, status: str = "valid"):
    """Update the last_scraped timestamp and status for a domain with file locking."""
    try:
        domains_file = ARGS.domains
        ensure_domains_file(domains_file)
        
        # Open file for reading and writing
        with open(domains_file, 'r+', encoding='utf-8') as f:
            # Acquire exclusive lock
            lock_file(f, True)
            
            try:
                # Read all domains
                f.seek(0)
                lines = f.readlines()
                
                # Update the specific domain
                updated = False
                for i, line in enumerate(lines):
                    line = line.strip()
                    if line and not line.startswith('#'):
                        # Extract domain (remove trailing comma)
                        current_domain = line.rstrip(',')
                        
                        if current_domain == domain:
                            # Update the line with timestamp and status
                            timestamp = datetime.utcnow().isoformat() + 'Z'
                            if status == "valid":
                                new_line = f"{domain},{timestamp},domain valid\n"
                            else:
                                new_line = f"{domain},{timestamp},domain deactivated/invalid\n"
                            lines[i] = new_line
                            updated = True
                            break
                
                if not updated:
                    logger.warning("domain_not_found_for_update", domain=domain)
                    return
                
                # Write back to file
                f.seek(0)
                f.truncate()  # Clear file content
                f.writelines(lines)
                
                logger.debug("domain_timestamp_updated", domain=domain, status=status)
                
            finally:
                # Release lock
                unlock_file(f)
            
    except Exception as e:
        logger.error("update_last_scraped_failed", domain=domain, error=str(e))
        raise

def is_recently_scraped(domain: str) -> bool:
    """Check if domain was recently scraped (within 24 hours)."""
    try:
        domains_file = ARGS.domains
        ensure_domains_file(domains_file)
        
        with open(domains_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    parts = line.split(',')
                    if len(parts) >= 2:
                        current_domain = parts[0]
                        if current_domain == domain:
                            # Check if there's a timestamp
                            if len(parts) >= 2 and parts[1]:
                                try:
                                    # Parse the timestamp
                                    timestamp_str = parts[1]
                                    if timestamp_str.endswith('Z'):
                                        timestamp_str = timestamp_str[:-1]  # Remove Z
                                    last_scraped = datetime.fromisoformat(timestamp_str)
                                    now = datetime.utcnow()
                                    
                                    # Check if it's been less than 24 hours
                                    time_diff = now - last_scraped
                                    if time_diff.total_seconds() < 24 * 3600:  # 24 hours in seconds
                                        logger.debug("domain_recently_scraped", domain=domain, hours_ago=time_diff.total_seconds() / 3600)
                                        return True
                                except Exception as e:
                                    logger.warning("timestamp_parse_failed", domain=domain, timestamp=parts[1], error=str(e))
                                    return False
                            break
        
        return False
        
    except Exception as e:
        logger.error("recent_scrape_check_failed", domain=domain, error=str(e))
        return False

# ---------------- Enrichment ----------------
# Initialize Redis client if URL is provided
redis_client = None
if os.getenv('REDIS_URL'):
    try:
        redis_client = redis.from_url(os.getenv('REDIS_URL'))
        logger.info("redis_client_initialized", url=os.getenv('REDIS_URL'))
    except Exception as e:
        logger.error("redis_client_init_failed", error=str(e))
        redis_client = None

def compress_payload(data: Dict[str, Any]) -> bytes:
    """Compress JSON payload with gzip."""
    json_str = json.dumps(data)
    return gzip.compress(json_str.encode('utf-8'))

def get_email_hash(email: str) -> str:
    """Generate SHA256 hash of email for Redis key."""
    return hashlib.sha256(email.lower().encode()).hexdigest()

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=32), retry=retry_if_exception_type(aiohttp.ClientError))
async def enrich_with_pdl(session: aiohttp.ClientSession, email: str) -> Optional[Dict[str, Any]]:
    """Enrich email with PDL API using retry logic and gzip compression."""
    try:
        payload = {"email": email}
        compressed_data = compress_payload(payload)
        
        async with session.post(
            "https://api.peopledatalabs.com/v5/person/enrich",
            params={"email": email},
            headers={
                "X-Api-Key": ARGS.pdl_key,
                "Content-Encoding": "gzip",
                "Content-Type": "application/json"
            },
            data=compressed_data,
            timeout=aiohttp.ClientTimeout(total=30)
        ) as response:
            if response.status == 200:
                data = await response.json()
                ENRICH_CREDITS_TOTAL.inc()  # 1 credit per PDL call
                return data
            else:
                logger.warning("pdl_api_error", email=email, status=response.status)
                return None
    except Exception as e:
        logger.error("pdl_enrichment_failed", email=email, error=str(e))
        raise

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=32), retry=retry_if_exception_type(aiohttp.ClientError))
async def enrich_with_hunter(session: aiohttp.ClientSession, email: str) -> Optional[Dict[str, Any]]:
    """Enrich email with Hunter API using retry logic."""
    try:
        async with session.get(
            "https://api.hunter.io/v2/email-verifier",
            params={"email": email, "api_key": ARGS.hunter_key},
            timeout=aiohttp.ClientTimeout(total=30)
        ) as response:
            if response.status == 200:
                data = await response.json()
                return data
            else:
                logger.warning("hunter_api_error", email=email, status=response.status)
                return None
    except Exception as e:
        logger.error("hunter_enrichment_failed", email=email, error=str(e))
        raise

async def check_redis_duplicate(email: str) -> bool:
    """Check if email is a duplicate using Redis."""
    if not redis_client:
        return False
    
    try:
        email_hash = get_email_hash(email)
        result = redis_client.set(f"email:{email_hash}", "1", ex=86400, nx=True)
        return not result  # True if key already existed (duplicate)
    except Exception as e:
        logger.error("redis_check_failed", email=email, error=str(e))
        return False

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
async def enrich_batch(session: aiohttp.ClientSession, batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Enrich batch of leads with PDL and Hunter APIs."""
    if not batch:
        return batch
    
    logger.info("enriching_batch", batch_size=len(batch))
    ENRICH_CALLS_TOTAL.inc()
    
    # Check for duplicates using Redis
    leads_to_enrich = []
    for lead in batch:
        email = lead.get("email", "").lower().strip()
        if not email:
            continue
            
        is_duplicate = await check_redis_duplicate(email)
        if is_duplicate:
            logger.info("skipping_duplicate_email", email=email)
            continue
        
        leads_to_enrich.append(lead)
    
    if not leads_to_enrich:
        logger.info("no_leads_to_enrich_after_duplicate_check")
        return batch
    
    # Enrich each lead
    for lead in leads_to_enrich:
        email = lead.get("email", "").lower().strip()
        
        # Enrich with PDL first
        try:
            pdl_data = await enrich_with_pdl(session, email)
            if pdl_data:
                lead["enrichment"]["pdl"] = pdl_data
        except Exception as e:
            logger.error("pdl_enrichment_failed", email=email, error=str(e))
        
        # Enrich with Hunter as fallback
        try:
            hunter_data = await enrich_with_hunter(session, email)
            if hunter_data:
                lead["enrichment"]["hunter"] = hunter_data
        except Exception as e:
            logger.error("hunter_enrichment_failed", email=email, error=str(e))
        
        # Record latency
        start_time = time.time()
        await asyncio.sleep(0.1)
    
    logger.info("batch_enriched", enriched_count=len(leads_to_enrich))
    return batch

# ---------------- Validator submission ----------------
OUTBOX = "outbox/leads.jsonl"
PENDING_ENRICH_DIR = "outbox/pending_enrich"

async def submit_pre_enrich(batch: List[Dict[str, Any]]):
    """Submit pre-enrich leads to pending_enrich directory."""
    if not batch:
        return
    
    try:
        # Create pending_enrich directory if it doesn't exist
        Path(PENDING_ENRICH_DIR).mkdir(parents=True, exist_ok=True)
        
        # Get current date for filename
        date_str = datetime.utcnow().strftime("%Y%m%d")
        pending_file = Path(PENDING_ENRICH_DIR) / f"{date_str}.jsonl"
        
        # Log pre-enrich storage start
        logger.info("pre_enrich_storage_started", 
                   batch_size=len(batch), 
                   file=str(pending_file),
                   emails=[lead.get("email", "") for lead in batch])
        
        # Write leads to pending file
        with open(pending_file, "a", encoding="utf-8") as f:
            for lead in batch:
                json_line = json.dumps(lead, ensure_ascii=False) + "\n"
                f.write(json_line)
        
        logger.info("leads_submitted_pre_enrich", 
                   count=len(batch), 
                   file=str(pending_file),
                   file_size=Path(pending_file).stat().st_size if Path(pending_file).exists() else 0)
        
    except Exception as e:
        logger.error("pre_enrich_submission_failed", error=str(e), batch_size=len(batch))
        raise

async def submit_enriched(batch: List[Dict[str, Any]]):
    """Submit enriched leads to final outbox JSONL file."""
    if not batch:
        return
    
    try:
        # Log final storage start
        logger.info("final_storage_started", 
                   batch_size=len(batch),
                   emails=[lead.get("email", "") for lead in batch])
        
        for lead in batch:
            json_line = json.dumps(lead, ensure_ascii=False) + "\n"
            outbox_file.write(json_line)
            outbox_file.flush()  # Ensure immediate write
        
        logger.info("leads_submitted_final", 
                   count=len(batch),
                   outbox_file=OUTBOX,
                   enrichment_summary={
                       "with_pdl": len([l for l in batch if l.get("enrichment", {}).get("pdl")]),
                       "with_hunter": len([l for l in batch if l.get("enrichment", {}).get("hunter")]),
                       "no_enrichment": len([l for l in batch if not l.get("enrichment", {})])
                   })
        
    except Exception as e:
        logger.error("final_submission_failed", error=str(e), batch_size=len(batch))
        raise

# ---------------- Orchestrator ----------------
async def flush_batch(batch: List[Dict[str, Any]]):
    """Flush a batch of leads with pre-enrich storage, enrichment, and final submission."""
    if not batch:
        return
    
    logger.info("batch_flush_started", 
               batch_size=len(batch),
               emails=[lead.get("email", "") for lead in batch])
    
    # Step 1: Store leads in pre-enrich state
    logger.info("batch_pre_enrich_started", batch_size=len(batch))
    await submit_pre_enrich(batch)
    logger.info("batch_pre_enrich_completed", batch_size=len(batch))
    
    # Step 2: Enrich the leads
    logger.info("batch_enrichment_started", batch_size=len(batch))
    async with aiohttp.ClientSession() as session:
        enriched_batch = await enrich_batch(session, batch)
    logger.info("batch_enrichment_completed", 
               original_size=len(batch),
               enriched_size=len(enriched_batch))
    
    # Step 3: Submit enriched leads to final destination
    logger.info("batch_final_storage_started", batch_size=len(enriched_batch))
    await submit_enriched(enriched_batch)
    logger.info("batch_final_storage_completed", batch_size=len(enriched_batch))
    
    logger.info("batch_flush_completed", 
               original_size=len(batch),
               final_size=len(enriched_batch))

async def process_domain(domain: str):
    """Process a single domain with the new flow: extract → pre-enrich → enrich → final storage."""
    global current_batch
    
    logger.info("processing_domain_started", domain=domain)
    
    # Check if recently scraped
    if is_recently_scraped(domain):
        logger.info("domain_skipped_recent", domain=domain)
        return
    
    # Update timestamp
    update_last_scraped(domain)
    logger.info("domain_timestamp_updated", domain=domain)
    
    # Crawl domain
    logger.info("crawl_started", domain=domain)
    crawl_data = await crawl_domain(domain)
    if not crawl_data:
        logger.warning("crawl_failed", domain=domain)
        return
    
    logger.info("crawl_completed", domain=domain, pages_crawled=len(crawl_data))
    
    # Extract eligible URLs
    eligible_urls = []
    for page in crawl_data:
        if "metadata" in page and "sourceURL" in page["metadata"]:
            url = page["metadata"]["sourceURL"]
            if should_include_url(url) and not should_exclude_url(url):
                eligible_urls.append(url)
    
    logger.info("eligible_urls_found", domain=domain, count=len(eligible_urls), urls=eligible_urls)
    
    if not eligible_urls:
        logger.info("no_eligible_urls", domain=domain)
        return
    
    # Extract contacts from URLs
    logger.info("extraction_started", domain=domain, url_count=len(eligible_urls))
    contacts = await extract_contacts_from_urls(eligible_urls, domain)
    logger.info("extraction_completed", 
               domain=domain, 
               contacts_found=len(contacts),
               emails=[contact.get("email", "") for contact in contacts])
    
    # Filter legitimate emails
    legitimate_contacts = []
    for contact in contacts:
        email = contact.get("email", "")
        if email and await is_legit(email):
            legitimate_contacts.append(contact)
    
    logger.info("legitimacy_check_completed", 
               domain=domain, 
               total_contacts=len(contacts),
               legitimate_contacts=len(legitimate_contacts),
               legitimate_emails=[contact.get("email", "") for contact in legitimate_contacts])
    
    if not legitimate_contacts:
        logger.info("no_legitimate_contacts", domain=domain)
        return
    
    # Store leads in pre-enrich stage
    logger.info("pre_enrich_stage_started", domain=domain, lead_count=len(legitimate_contacts))
    await submit_pre_enrich(legitimate_contacts)
    logger.info("pre_enrich_stage_completed", domain=domain)
    
    # Enrich leads
    logger.info("enrichment_started", domain=domain, lead_count=len(legitimate_contacts))
    async with aiohttp.ClientSession() as session:
        enriched_contacts = await enrich_batch(session, legitimate_contacts)
    logger.info("enrichment_completed", 
               domain=domain, 
               enriched_count=len(enriched_contacts),
               enrichment_stats={
                   "pdl_success": len([c for c in enriched_contacts if c.get("enrichment", {}).get("pdl")]),
                   "hunter_success": len([c for c in enriched_contacts if c.get("enrichment", {}).get("hunter")]),
                   "no_enrichment": len([c for c in enriched_contacts if not c.get("enrichment", {})])
               })
    
    # Store enriched leads in final outbox
    logger.info("final_storage_stage_started", domain=domain, lead_count=len(enriched_contacts))
    await submit_enriched(enriched_contacts)
    logger.info("final_storage_stage_completed", domain=domain)
    
    logger.info("domain_processing_completed", 
               domain=domain,
               total_leads_processed=len(enriched_contacts))

async def miner():
    """Main mining function with graceful shutdown."""
    global shutdown_requested, current_batch, outbox_file
    
    print("DEBUG: Miner function started")  # Debug output
    
    try:
        print("DEBUG: Starting web server")  # Debug output
        # Start web server
        web_runner = await start_web_server()
        
        print("DEBUG: Web server started, initializing metrics")  # Debug output
        # Initialize metrics
        BATCH_SIZE_GAUGE.set(ARGS.batch_size)
        FLUSH_SEC_GAUGE.set(ARGS.flush_sec)
        
        # Check file locking availability
        try:
            import fcntl
            logger.info("file_locking_available")
        except ImportError:
            logger.warning("file_locking_not_available_windows", 
                          note="Multiple processes may cause race conditions")
        
        print("DEBUG: Loading domains")  # Debug output
        logger.info("miner_started", 
                   domains_file=ARGS.domains,
                   outbox_file=OUTBOX,
                   pending_enrich_dir=PENDING_ENRICH_DIR)
        
        # Load domains
        domains = load_domains(ARGS.domains)
        print(f"DEBUG: Loaded {len(domains)} domains: {domains}")  # Debug output
        
        # Initialize outbox
        outbox_file = open(OUTBOX, "a", encoding="utf-8")
        
        print("DEBUG: Processing domains")  # Debug output
        # Process domains
        for domain in domains:
            if shutdown_requested:
                logger.info("shutdown_requested_processing_domain", domain=domain)
                break
                
            print(f"DEBUG: Processing domain: {domain}")  # Debug output
            try:
                await process_domain(domain)
            except Exception as e:
                logger.error("domain_processing_failed", domain=domain, error=str(e))
                continue
        
        # Graceful shutdown: flush any remaining batch
        if current_batch and not shutdown_requested:
            logger.info("flushing_final_batch", batch_size=len(current_batch))
            await flush_batch(current_batch)
        
        print("DEBUG: Miner completed successfully")  # Debug output
        
    except Exception as e:
        print(f"DEBUG: Miner failed with error: {e}")  # Debug output
        logger.error("miner_failed", error=str(e))
        raise
    finally:
        # Cleanup
        if outbox_file:
            outbox_file.close()
            logger.info("outbox_file_closed")
        
        if 'web_runner' in locals():
            await web_runner.cleanup()
            logger.info("web_server_stopped")
        
        if shutdown_requested:
            logger.info("miner_shutdown_complete")
        else:
            logger.info("miner_completed")

# ---------------- Web Server ----------------
async def health_handler(request):
    """Health check endpoint."""
    return web.Response(text="OK", status=200)

async def metrics_handler(request):
    """Prometheus metrics endpoint."""
    from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
    return web.Response(
        body=generate_latest(),
        content_type='text/plain',
        charset='utf-8'
    )

async def start_web_server():
    """Start aiohttp web server for health and metrics."""
    app = web.Application()
    app.router.add_get('/healthz', health_handler)
    app.router.add_get('/metrics', metrics_handler)
    
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, '0.0.0.0', 9100)
    await site.start()
    
    logger.info("web_server_started", port=9100)
    return runner

# ---------------- Domain Processing ----------------
def is_recently_scraped(domain: str) -> bool:
    """Check if domain was recently scraped (within 24 hours)."""
    try:
        domains_file = ARGS.domains
        ensure_domains_file(domains_file)
        
        with open(domains_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    parts = line.split(',')
                    if len(parts) >= 2:
                        current_domain = parts[0]
                        if current_domain == domain:
                            # Check if there's a timestamp
                            if len(parts) >= 2 and parts[1]:
                                try:
                                    # Parse the timestamp
                                    timestamp_str = parts[1]
                                    if timestamp_str.endswith('Z'):
                                        timestamp_str = timestamp_str[:-1]  # Remove Z
                                    last_scraped = datetime.fromisoformat(timestamp_str)
                                    now = datetime.utcnow()
                                    
                                    # Check if it's been less than 24 hours
                                    time_diff = now - last_scraped
                                    if time_diff.total_seconds() < 24 * 3600:  # 24 hours in seconds
                                        logger.debug("domain_recently_scraped", domain=domain, hours_ago=time_diff.total_seconds() / 3600)
                                        return True
                                except Exception as e:
                                    logger.warning("timestamp_parse_failed", domain=domain, timestamp=parts[1], error=str(e))
                                    return False
                            break
        
        return False
        
    except Exception as e:
        logger.error("recent_scrape_check_failed", domain=domain, error=str(e))
        return False

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
async def firecrawl_crawl(domain: str) -> Optional[List[Dict[str, Any]]]:
    """Crawl domain using Firecrawl API with retry logic."""
    try:
        async with aiohttp.ClientSession() as session:
            # Start crawl job
            async with session.post(
                "https://api.firecrawl.dev/v1/crawl",
                headers=HEADERS_FC,
                json={
                    "url": domain if domain.startswith(('http://', 'https://')) else f"https://{domain}",
                    "maxDepth": 6,
                    "limit": ARGS.max_pages_per_domain,
                    "allowBackwardLinks": True,
                    "includePaths": INCLUDE_PATTERNS,
                    "excludePaths": EXCLUDE_PATTERNS
                },
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error("firecrawl_error", 
                               domain=domain,
                               status=response.status,
                               error=error_text,
                               headers=dict(response.headers))
                    response.raise_for_status()
                    
                job = await response.json()
                job_id = job["id"]
                logger.info("crawl_job_started", domain=domain, job_id=job_id)
            
            # Poll for completion with early-stop safeguard and 429 back-off
            crawl_start_time = time.time()
            retry_count = 0
            max_retries = 3
        
            max_crawl_time = 300
            while True:
                await asyncio.sleep(5)  # Poll interval
                
                # Early-stop safeguard: check if elapsed time > max_crawl_time
                elapsed_time = time.time() - crawl_start_time
                if elapsed_time > max_crawl_time:
                    logger.warning("crawl_timeout_exceeded", domain=domain, job_id=job_id, elapsed_time=elapsed_time)
                    break
                
                try:
                    async with session.get(
                        f"https://api.firecrawl.dev/v1/crawl/{job_id}",
                        headers=HEADERS_FC,
                        timeout=aiohttp.ClientTimeout(total=30)
                    ) as response:
                        if response.status != 200:
                            error_text = await response.text()
                            logger.error("firecrawl_status_error", 
                                       domain=domain,
                                       job_id=job_id,
                                       status=response.status,
                                       error=error_text,
                                       headers=dict(response.headers))
                            response.raise_for_status()
                            
                        status = await response.json()
                        
                        pages = status.get("data", [])
                        if pages:
                            urls = [p["metadata"]["sourceURL"] for p in pages if "metadata" in p and "sourceURL" in p["metadata"]]
                            logger.info("crawl_progress", domain=domain, job_id=job_id, pages_found=len(urls))
                            
                            # Update Prometheus metric for crawl credits spent
                            CRAWL_CREDITS_SPENT_TOTAL.inc(len(urls))
                        
                        if status.get("status") == "completed":
                            logger.info("crawl_completed", domain=domain, job_id=job_id, pages_found=len(pages))
                            return pages
                        
                        logger.info("crawl_status", domain=domain, job_id=job_id, status=status.get('status'))
                        
                except aiohttp.ClientResponseError as e:
                    if e.status == 429:
                        retry_count += 1
                        if retry_count > max_retries:
                            logger.error("crawl_rate_limit_exceeded", domain=domain, job_id=job_id)
                            return None
                        
                        logger.warning("crawl_rate_limited", domain=domain, job_id=job_id, retry_count=retry_count)
                        await asyncio.sleep(30)
                        continue
                    else:
                        logger.error("firecrawl_status_exception", 
                                   domain=domain,
                                   job_id=job_id,
                                   error=str(e),
                                   error_type=type(e).__name__)
                        raise
            
            return None
                    
    except Exception as e:
        logger.error("firecrawl_exception", 
                     domain=domain,
                     error=str(e),
                     error_type=type(e).__name__)
        raise

def batched(iterable, n):
    """Split iterable into batches of size n."""
    iterator = iter(iterable)
    while True:
        batch = list(itertools.islice(iterator, n))
        if not batch:
            break
        yield batch

async def firecrawl_extract_contacts(session, url):
    """Call Firecrawl /v1/extract for a single URL and return contacts list."""
    try:
        logger.info("extract_api_call_started", url=url)
        
        # Step 1: Start the extract job
        async with session.post(
            "https://api.firecrawl.dev/v1/extract",
            headers=HEADERS_FC,
            json={"urls": [url]},
            timeout=aiohttp.ClientTimeout(total=30)
        ) as response:
            logger.info("extract_api_response_received", url=url, status=response.status)
            response.raise_for_status()
            result = await response.json()
            logger.info("extract_api_response_parsed", url=url, result_keys=list(result.keys()) if isinstance(result, dict) else "Not a dict")
            logger.debug("extract_api_full_response", url=url, response=result)
            
            if not result.get("success") or "id" not in result:
                logger.error("extract_job_failed", url=url, result=result)
                return []
            
            job_id = result["id"]
            logger.info("extract_job_started", url=url, job_id=job_id)
            
        # Step 2: Poll the extract job until completion
        max_wait_time = 300  # 5 minutes
        poll_interval = 5
        elapsed_time = 0
        
        while elapsed_time < max_wait_time:
            await asyncio.sleep(poll_interval)
            elapsed_time += poll_interval
            
            logger.info("extract_job_polling", url=url, job_id=job_id, elapsed=elapsed_time)
            
            async with session.get(
                f"https://api.firecrawl.dev/v1/extract/{job_id}",
                headers=HEADERS_FC,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as poll_response:
                poll_response.raise_for_status()
                poll_result = await poll_response.json()
                logger.debug("extract_job_poll_response", url=url, job_id=job_id, result=poll_result)
                
                status = poll_result.get("status", "unknown")
                logger.info("extract_job_status", url=url, job_id=job_id, status=status)
                
                if status == "completed":
                    # Step 3: Parse the completed results
                    logger.info("extract_job_completed", url=url, job_id=job_id)
                    return await parse_extract_results(session, url, poll_result)
                elif status == "failed":
                    logger.error("extract_job_failed", url=url, job_id=job_id, result=poll_result)
                    break
                elif status in ["pending", "processing"]:
                    continue
                else:
                    logger.warning("extract_job_unknown_status", url=url, job_id=job_id, status=status)
                    break
        
        logger.error("extract_job_timeout", url=url, job_id=job_id, max_wait_time=max_wait_time)
        return await fallback_heuristic_contact(url)
        
    except Exception as e:
        logger.error("extract_api_error", url=url, error=str(e), error_type=type(e).__name__)
        return await fallback_heuristic_contact(url)

async def parse_extract_results(session, url, poll_result):
    """Parse contacts from the completed extract job results."""
    contacts = []
    
    try:
        # Check if we have data with contacts
        if "data" in poll_result and "contacts" in poll_result["data"]:
            logger.info("contacts_found_in_api_response", url=url, contact_count=len(poll_result['data']['contacts']))
            for contact in poll_result["data"]["contacts"]:
                logger.debug("processing_contact", url=url, contact=contact)
                full_name = contact.get("name", "")
                first_name = ""
                last_name = ""
                if full_name:
                    name_parts = full_name.strip().split()
                    if len(name_parts) >= 2:
                        first_name = name_parts[0]
                        last_name = " ".join(name_parts[1:])
                    else:
                        first_name = full_name
                email = contact.get("email", "").strip()
                if email and "@" not in email:
                    logger.warning("invalid_email_format", url=url, email=email)
                    email = ""
                contact_data = {
                    "email": email,
                    "first_name": first_name,
                    "last_name": last_name,
                    "title": contact.get("title", "").strip(),
                    "company": contact.get("company", "").strip(),
                    "url": url,
                    "source_domain": url.split("/")[2] if url.startswith("http") else url,
                    "timestamp": datetime.utcnow().isoformat() + 'Z',
                    "enrichment": {}
                }
                if email or first_name or last_name:
                    contacts.append(contact_data)
                    logger.info("contact_added", url=url, contact_data=contact_data)
                else:
                    logger.warning("contact_skipped_no_data", url=url, contact_data=contact_data)
        else:
            logger.info("no_contacts_in_api_response", url=url, data_keys=list(poll_result.get('data', {}).keys()) if isinstance(poll_result.get('data'), dict) else "No data field")
        
        # Check if we should use heuristic fallback
        if not contacts and "/contact" in url:
            logger.info("using_heuristic_fallback", url=url)
            heuristic_contact = await create_heuristic_contact(url)
            contacts.append(heuristic_contact)
            logger.info("heuristic_contact_created", url=url, contact_data=heuristic_contact)
        elif not contacts:
            logger.info("no_contacts_and_no_heuristic", url=url)
        
        logger.info("extract_completed", url=url, contacts_found=len(contacts))
        return contacts
        
    except Exception as e:
        logger.error("parse_extract_results_error", url=url, error=str(e))
        return await fallback_heuristic_contact(url)

async def create_heuristic_contact(url):
    """Create a heuristic contact for /contact pages."""
    domain_name = url.replace('https://', '').replace('http://', '').split('/')[0]
    return {
        "email": f"contact@{domain_name}",
        "first_name": "",
        "last_name": "",
        "title": "Contact Page",
        "company": domain_name,
        "url": url,
        "source_domain": domain_name,
        "timestamp": datetime.utcnow().isoformat() + 'Z',
        "enrichment": {}
    }

async def fallback_heuristic_contact(url):
    """Fallback to heuristic contact for /contact pages on any error."""
    contacts = []
    if "/contact" in url:
        logger.info("using_heuristic_fallback_on_error", url=url)
        heuristic_contact = await create_heuristic_contact(url)
        contacts.append(heuristic_contact)
        logger.info("heuristic_contact_created_on_error", url=url, contact_data=heuristic_contact)
    return contacts

async def extract_contacts_from_urls(urls, source_domain):
    """Extract contacts from all eligible URLs using Firecrawl extract API."""
    contacts = []
    async with aiohttp.ClientSession() as session:
        for url in urls:
            url_contacts = await firecrawl_extract_contacts(session, url)
            contacts.extend(url_contacts)
            await asyncio.sleep(1)  # avoid rate limiting
    return contacts

# ------------------------------------------------------------------
#  Public helper so the miner can ask "give me N leads please"
# ------------------------------------------------------------------
async def get_firecrawl_leads(num_leads: int,
                              industry: Optional[str] = None,
                              region:   Optional[str] = None) -> List[Dict]:
    """
    Get leads using Firecrawl API with domain filtering and deduplication.
    """
    leads = []
    attempts = 0
    max_attempts = 20  # Increased to handle more domain attempts

    while len(leads) < num_leads and attempts < max_attempts:
        attempts += 1
        
        # Get a random sample of domains (try 1 domain at a time for efficiency)
        random_domains = get_random_domains(ARGS.domains, sample_size=1)
        
        if not random_domains:
            print("⚠️  No domains available")
            break
        
        domain = random_domains[0]
        print(f"🔄 Attempt {attempts}: Trying domain {domain}")
        
        try:
            # Try to crawl this domain
            contacts = await crawl_domain(domain)
            
            for c in contacts:
                email = c.get("email", "")
                
                # Skip if email already exists in pool
                if email and check_existing_contact(email):
                    print(f"⏭️  Skipping duplicate contact: {email}")
                    continue
                
                # ---- LLM classification ------------------------------------
                parts = [
                    domain,
                    c.get("company", ""),
                    c.get("title", ""),
                    c.get("url") or f"https://{domain}"
                ]
                # drop duplicate root-domain if company == domain
                seen = set()
                meta_text = " ".join(
                    p for p in parts if p and (p not in seen and not seen.add(p))
                )
                industry, sub_ind = _classify_industry(meta_text)

                # ---- map the Firecrawl contact object to legacy keys ----
                full_name = f"{c.get('first_name','')} {c.get('last_name','')}".strip()
                website   = c.get("url") or f"https://{domain}"
                mapped = {
                    "Business":        c.get("company") or domain,
                    "Owner Full name": full_name,
                    "First":           c.get("first_name", ""),
                    "Last":            c.get("last_name", ""),
                    "Owner(s) Email":  email,
                    "LinkedIn":        c.get("enrichment", {}).get("pdl", {}).get("linkedin", ""),
                    "Website":         website,
                    "Industry":       industry,
                    "sub_industry":   sub_ind,      # <- lower-case, consistent
                    "Region":          region or "Global",
                }
                # basic filtering so we don't return empty rows
                if mapped["Owner(s) Email"]:
                    leads.append(mapped)

                if len(leads) >= num_leads:
                    print(f"✅ Found {len(leads)} leads after {attempts} domain attempts")
                    return leads
            
            # Show results for this domain
            if contacts:
                print(f"✅ Domain {domain}: Found {len(contacts)} leads")
            else:
                print(f"❌ Domain {domain}: No leads found")
            
        except Exception as e:
            print(f"⚠️  Error processing domain {domain}: {e}")
            # Continue to next domain instead of failing completely
            continue

    print(f" Final result: {len(leads)} leads found after {attempts} domain attempts")
    return leads[:num_leads]     # may be fewer if Firecrawl found less

def check_existing_contact(email: str) -> bool:
    """Check if a contact with this email already exists in the leads pool."""
    try:
        from Leadpoet.base.utils.pool import get_leads_from_pool
        
        # Get all leads from pool to check for duplicates
        all_leads = get_leads_from_pool(1000000)  # Get a large number to check all
        
        existing_emails = {lead.get("owner_email", "").lower() for lead in all_leads}
        return email.lower() in existing_emails
        
    except Exception as e:
        print(f"❌ Error checking existing contact: {e}")
        return False

# ─────────────────────────────────────────────────────────────────────
# CONFIG

LEADS_PER_DOMAIN = 1          # emit every lead immediately

# keyword → (sub-industry, industry)
KEYWORD_MAP = {
    # Tech & AI  (25)
    "saas":               ("SaaS", "Tech & AI"),
    "software":           ("Software Development", "Tech & AI"),
    "cloud":              ("Cloud Computing", "Tech & AI"),
    "devops":             ("DevOps", "Tech & AI"),
    "cybersecurity":      ("Cybersecurity", "Tech & AI"),
    "security":           ("Cybersecurity", "Tech & AI"),
    "ai":                 ("Artificial Intelligence", "Tech & AI"),
    "machine learning":   ("Artificial Intelligence", "Tech & AI"),
    "deep learning":      ("Artificial Intelligence", "Tech & AI"),
    "nlp":                ("Artificial Intelligence", "Tech & AI"),
    "data analytics":     ("Data Analytics", "Tech & AI"),
    "big data":           ("Data Analytics", "Tech & AI"),
    "iot":                ("Internet of Things", "Tech & AI"),
    "internet of things": ("Internet of Things", "Tech & AI"),
    "robotics":           ("Robotics", "Tech & AI"),
    "semiconductor":      ("Semiconductors", "Tech & AI"),
    "hardware":           ("Hardware", "Tech & AI"),
    "ar":                 ("Augmented Reality / VR", "Tech & AI"),
    "vr":                 ("Augmented Reality / VR", "Tech & AI"),
    "virtual reality":    ("Augmented Reality / VR", "Tech & AI"),
    "augmented reality":  ("Augmented Reality / VR", "Tech & AI"),
    "gaming":             ("Gaming", "Tech & AI"),
    "esports":            ("Gaming", "Tech & AI"),
    "web development":    ("Web Development", "Tech & AI"),
    "mobile app":         ("Mobile Development", "Tech & AI"),

    # Finance & Fintech (21)
    "fintech":            ("FinTech", "Finance & Fintech"),
    "payments":           ("Payments", "Finance & Fintech"),
    "payment":            ("Payments", "Finance & Fintech"),
    "crypto":             ("Cryptocurrency", "Finance & Fintech"),
    "cryptocurrency":     ("Cryptocurrency", "Finance & Fintech"),
    "blockchain":         ("Blockchain", "Finance & Fintech"),
    "defi":               ("DeFi", "Finance & Fintech"),
    "lending":            ("Lending", "Finance & Fintech"),
    "insurtech":          ("InsurTech", "Finance & Fintech"),
    "insurance":          ("InsurTech", "Finance & Fintech"),
    "bank":               ("Banking", "Finance & Fintech"),
    "banking":            ("Banking", "Finance & Fintech"),
    "wealth":             ("Wealth Management", "Finance & Fintech"),
    "investment":         ("Wealth Management", "Finance & Fintech"),
    "trading":            ("Trading", "Finance & Fintech"),
    "brokerage":          ("Trading", "Finance & Fintech"),
    "regtech":            ("RegTech", "Finance & Fintech"),
    "accounting":         ("Accounting", "Finance & Fintech"),
    "payments gateway":   ("Payments", "Finance & Fintech"),
    "pos":                ("Payments", "Finance & Fintech"),
    "personal finance":   ("Personal Finance", "Finance & Fintech"),

    # Health & Wellness (21)
    "health":             ("Healthcare", "Health & Wellness"),
    "healthcare":         ("Healthcare", "Health & Wellness"),
    "clinic":             ("Healthcare Provider", "Health & Wellness"),
    "hospital":           ("Healthcare Provider", "Health & Wellness"),
    "pharma":             ("Pharmaceuticals", "Health & Wellness"),
    "pharmaceutical":     ("Pharmaceuticals", "Health & Wellness"),
    "biotech":            ("Biotech", "Health & Wellness"),
    "life science":       ("Biotech", "Health & Wellness"),
    "medical device":     ("Medical Devices", "Health & Wellness"),
    "medtech":            ("Medical Devices", "Health & Wellness"),
    "telemedicine":       ("Telemedicine", "Health & Wellness"),
    "telehealth":         ("Telemedicine", "Health & Wellness"),
    "fitness":            ("Fitness", "Health & Wellness"),
    "wellness":           ("Fitness", "Health & Wellness"),
    "nutrition":          ("Nutrition", "Health & Wellness"),
    "mental health":      ("Mental Health", "Health & Wellness"),
    "dental":             ("Healthcare Provider", "Health & Wellness"),
    "veterinary":         ("Healthcare Provider", "Health & Wellness"),
    "elder care":         ("Elder Care", "Health & Wellness"),
    "digital health":     ("Digital Health", "Health & Wellness"),
    "sports medicine":    ("Sports Medicine", "Health & Wellness"),

    # Media & Education (23)
    "edtech":             ("EdTech", "Media & Education"),
    "education":          ("Education Services", "Media & Education"),
    "e-learning":         ("EdTech", "Media & Education"),
    "online course":      ("EdTech", "Media & Education"),
    "school":             ("Education Services", "Media & Education"),
    "university":         ("Education Services", "Media & Education"),
    "training":           ("Training & Coaching", "Media & Education"),
    "coaching":           ("Training & Coaching", "Media & Education"),
    "publishing":         ("Publishing", "Media & Education"),
    "media":              ("Media & Entertainment", "Media & Education"),
    "news":               ("Media & Entertainment", "Media & Education"),
    "content":            ("Media & Entertainment", "Media & Education"),
    "streaming":          ("Streaming", "Media & Education"),
    "video":              ("Streaming", "Media & Education"),
    "audio":              ("Streaming", "Media & Education"),
    "podcast":            ("Streaming", "Media & Education"),
    "marketing":          ("Marketing Services", "Media & Education"),
    "advertising":        ("Marketing Services", "Media & Education"),
    "pr agency":          ("Marketing Services", "Media & Education"),
    "game studio":        ("Game Studio", "Media & Education"),
    "animation":          ("Media Production", "Media & Education"),
    "film":               ("Media Production", "Media & Education"),
    "music":              ("Media Production", "Media & Education"),

    # Energy & Industry (20)
    "energy":             ("Energy", "Energy & Industry"),
    "oil":                ("Oil & Gas", "Energy & Industry"),
    "gas":                ("Oil & Gas", "Energy & Industry"),
    "lng":                ("Oil & Gas", "Energy & Industry"),
    "renewable":          ("Renewable Energy", "Energy & Industry"),
    "solar":              ("Renewable Energy", "Energy & Industry"),
    "wind":               ("Renewable Energy", "Energy & Industry"),
    "geothermal":         ("Renewable Energy", "Energy & Industry"),
    "battery":            ("Energy Storage", "Energy & Industry"),
    "storage":            ("Energy Storage", "Energy & Industry"),
    "mining":             ("Mining", "Energy & Industry"),
    "manufacturing":      ("Manufacturing", "Energy & Industry"),
    "automotive":         ("Automotive", "Energy & Industry"),
    "industrial":         ("Industrial Automation", "Energy & Industry"),
    "automation":         ("Industrial Automation", "Energy & Industry"),
    "chemicals":          ("Chemicals", "Energy & Industry"),
    "aerospace":          ("Aerospace", "Energy & Industry"),
    "construction":       ("Construction", "Energy & Industry"),
    "logistics":          ("Logistics & Supply Chain", "Energy & Industry"),
    "supply chain":       ("Logistics & Supply Chain", "Energy & Industry"),
}

def _classify(text: str) -> tuple[str, str]:
    """
    Return (sub_industry, industry) using the first keyword match.
    Falls back to ('General Tech', 'Tech & AI').
    """
    txt = text.lower()
    for kw, (sub, ind) in KEYWORD_MAP.items():
        if kw in txt:
            return sub, ind
    return "General Tech", "Tech & AI"
# ─────────────────────────────────────────────────────────────────────

# ─────────────────────────── NEW CONFIG ────────────────────────────
LEADS_PER_DOMAIN = 1          # send every lead immediately
# very small seed list – extend freely
INDUSTRY_KEYWORDS = {
    "saas":          "Tech & AI",
    "software":      "Tech & AI",
    "ai":            "Tech & AI",
    "machine learning": "Tech & AI",
    "blockchain":    "Finance & Fintech",
    "crypto":        "Finance & Fintech",
    "bank":          "Finance & Fintech",
    "clinic":        "Health & Wellness",
    "hospital":      "Health & Wellness",
    "fitness":       "Health & Wellness",
    "solar":         "Energy & Industry",
    "oil":           "Energy & Industry",
    "manufacturing": "Energy & Industry",
    # …add more here…
}

def _infer_industry(text: str) -> str:
    """
    Very fast rule-based classifier using the Firecrawl title/description.
    Returns first match or 'Unknown'.
    """
    txt = text.lower()
    for kw, ind in INDUSTRY_KEYWORDS.items():
        if kw in txt:
            return ind
    return "Unknown"

# ──────────────────────────── TAXONOMY ────────────────────────────
# Ordered so the first hit "wins" when multiple keywords appear
SUB_INDUSTRY_KEYWORDS: "OrderedDict[str, tuple[str,list[str]]]" = OrderedDict([
    # Tech & AI  (20)
    ("SaaS",             ("Tech & AI", ["saas", "subscription software"])),
    ("Cloud Computing",  ("Tech & AI", ["cloud", "aws", "azure", "gcp"])),
    ("Cyber Security",   ("Tech & AI", ["cybersecurity", "firewall", "antivirus", "infosec"])),
    ("E-Commerce Tech",  ("Tech & AI", ["e-commerce platform", "shopify", "woocommerce"])),
    ("Blockchain",       ("Tech & AI", ["blockchain", "web3", "ethereum", "bitcoin"])),
    ("FinTech Stack",    ("Tech & AI", ["payment gateway", "stripe api", "banking-as-a-service"])),
    ("EdTech",           ("Tech & AI", ["edtech", "learning management system", "lms"])),
    ("MarTech",          ("Tech & AI", ["marketing automation", "crm", "hubspot"])),
    ("HR Tech",          ("Tech & AI", ["hr software", "talent platform", "ats"])),
    ("PropTech",         ("Tech & AI", ["proptech", "real-estate tech"])),
    ("Health IT",        ("Tech & AI", ["ehr", "emr", "telehealth", "medtech software"])),
    ("AI LLM",           ("Tech & AI", ["ai", "machine learning", "deep learning", "llm"])),
    ("Robotics",         ("Tech & AI", ["robotics", "autonomous", "drone"])),
    ("IoT",              ("Tech & AI", ["iot device", "internet of things"])),
    ("AR / VR",          ("Tech & AI", ["augmented reality", "virtual reality", "metaverse"])),
    ("Analytics",        ("Tech & AI", ["data analytics", "business intelligence", "bi tool"])),
    ("DevOps",           ("Tech & AI", ["ci/cd", "kubernetes", "devops"])),
    ("Gaming",           ("Tech & AI", ["game studio", "game development", "gaming platform"])),
    ("AdTech",           ("Tech & AI", ["ad network", "programmatic ads"])),
    ("General Software", ("Tech & AI", ["software development", "custom software"])),
    # Finance & Fintech (20)
    ("Neo Bank",         ("Finance & Fintech", ["neobank", "digital bank"])),
    ("Payment Gateway",  ("Finance & Fintech", ["payment gateway", "merchant services"])),
    ("Lending Platform", ("Finance & Fintech", ["p2p lending", "loan platform", "mortgage tech"])),
    ("Wealth Tech",      ("Finance & Fintech", ["wealth management", "robo-advisor"])),
    ("InsurTech",        ("Finance & Fintech", ["insurtech", "insurance platform"])),
    ("Accounting SaaS",  ("Finance & Fintech", ["accounting software", "bookkeeping platform"])),
    ("Crypto Exchange",  ("Finance & Fintech", ["crypto exchange", "defi", "dex"])),
    ("RegTech",          ("Finance & Fintech", ["regtech", "kyc", "aml"])),
    ("Payroll Tech",     ("Finance & Fintech", ["payroll software", "salary management"])),
    ("Point-of-Sale",    ("Finance & Fintech", ["pos system", "point of sale"])),
    ("Trading Platform", ("Finance & Fintech", ["trading platform", "brokerage"])),
    ("Tax Software",     ("Finance & Fintech", ["tax software", "tax filing"])),
    ("Expense Mgmt",     ("Finance & Fintech", ["expense management", "spend control"])),
    ("Invoice Factoring",("Finance & Fintech", ["invoice factoring", "receivables finance"])),
    ("Crowdfunding",     ("Finance & Fintech", ["crowdfunding", "kickstarter-like"])),
    ("Billing API",      ("Finance & Fintech", ["recurring billing", "stripe"])),
    ("Financial Data API",("Finance & Fintech",["open banking api", "plaid"])),
    ("Treasury Mgmt",    ("Finance & Fintech", ["treasury management", "cash management"])),
    ("ATM / Kiosk",      ("Finance & Fintech", ["atm network", "bank kiosk"])),
    ("Angel Network",    ("Finance & Fintech", ["angel network", "venture funding"])),
    # Health & Wellness (20)
    ("Hospital",         ("Health & Wellness", ["hospital", "clinic", "medical center"])),
    ("Telemedicine",     ("Health & Wellness", ["telemedicine", "telehealth"])),
    ("Mental Health App",("Health & Wellness", ["mental health app", "therapy app"])),
    ("Fitness Studio",   ("Health & Wellness", ["gym", "fitness center", "crossfit"])),
    ("Yoga Studio",      ("Health & Wellness", ["yoga studio", "pilates studio"])),
    ("Wellness Product", ("Health & Wellness", ["supplements", "nutraceutical"])),
    ("Pharma Company",   ("Health & Wellness", ["pharmaceutical", "drug discovery"])),
    ("Biotech",          ("Health & Wellness", ["biotech", "gene therapy"])),
    ("Medical Device",   ("Health & Wellness", ["medical device", "diagnostic device"])),
    ("Dental Clinic",    ("Health & Wellness", ["dental clinic", "dentist"])),
    ("EHR Vendor",       ("Health & Wellness", ["ehr", "emr platform"])),
    ("Health Insurance", ("Health & Wellness", ["health insurance", "payer"])),
    ("Nutrition Coaching",("Health & Wellness",["nutrition coach", "dietician"])),
    ("Spa / Wellness",   ("Health & Wellness", ["spa", "wellness retreat"])),
    ("Life Sciences",    ("Health & Wellness", ["life sciences", "clinical trial"])),
    ("Child Care",       ("Health & Wellness", ["childcare center", "daycare"])),
    ("Senior Care",      ("Health & Wellness", ["senior care", "assisted living"])),
    ("Sports Medicine",  ("Health & Wellness", ["sports medicine", "physiotherapy"])),
    ("Pet Health",       ("Health & Wellness", ["veterinary", "pet clinic"])),
    ("Healthcare Consulting",("Health & Wellness",["healthcare consulting"])),
    # Media & Education (20)
    ("University",       ("Media & Education", ["university", "college", "campus"])),
    ("K-12 School",      ("Media & Education", ["k-12", "elementary school", "high school"])),
    ("Online Course",    ("Media & Education", ["online course", "mooc", "udemy"])),
    ("Publishing House", ("Media & Education", ["publisher", "publishing house"])),
    ("News Outlet",      ("Media & Education", ["news", "newspaper", "magazine"])),
    ("Streaming Media",  ("Media & Education", ["streaming", "ott platform", "vod"])),
    ("Podcast Network",  ("Media & Education", ["podcast network", "podcasting"])),
    ("Marketing Agency", ("Media & Education", ["marketing agency", "advertising agency"])),
    ("Social Network",   ("Media & Education", ["social network", "social media platform"])),
    ("Gaming Studio",    ("Media & Education", ["game studio", "game publisher"])),
    ("Bookstore",        ("Media & Education", ["bookstore", "library"])),
    ("Language Learning",("Media & Education", ["language learning", "duolingo"])),
    ("Ed-Publishing",    ("Media & Education", ["educational publishing"])),
    ("E-Learning SaaS",  ("Media & Education", ["learning management system", "lms"])),
    ("PR Agency",        ("Media & Education", ["public relations", "pr agency"])),
    ("Photo Stock",      ("Media & Education", ["stock photos", "photo marketplace"])),
    ("Video Production", ("Media & Education", ["video production", "film studio"])),
    ("Music Label",      ("Media & Education", ["music label", "record label"])),
    ("Ticketing Platform",("Media & Education",["ticketing platform", "event tickets"])),
    ("e-Sports Org",     ("Media & Education", ["esports", "competitive gaming"])),
    # Energy & Industry (20)
    ("Solar Installer",  ("Energy & Industry", ["solar installer", "pv system"])),
    ("Wind Energy",      ("Energy & Industry", ["wind farm", "wind turbine"])),
    ("Oil & Gas",        ("Energy & Industry", ["oil field", "drilling", "refinery"])),
    ("Battery Tech",     ("Energy & Industry", ["battery technology", "lithium-ion"])),
    ("EV Charging",      ("Energy & Industry", ["ev charging", "charging station"])),
    ("Nuclear Energy",   ("Energy & Industry", ["nuclear power", "reactor"])),
    ("Water Utility",    ("Energy & Industry", ["water utility", "water treatment"])),
    ("Waste Management", ("Energy & Industry", ["waste management", "recycling"])),
    ("Manufacturing",    ("Energy & Industry", ["manufacturing", "factory"])),
    ("Logistics",        ("Energy & Industry", ["logistics", "supply chain"])),
    ("Aerospace",        ("Energy & Industry", ["aerospace", "space systems"])),
    ("Automotive",       ("Energy & Industry", ["automotive", "car manufacturer"])),
    ("Agritech",         ("Energy & Industry", ["agritech", "precision agriculture"])),
    ("Chemical Plant",   ("Energy & Industry", ["chemical plant", "chemicals"])),
    ("Mining",           ("Energy & Industry", ["mining", "mineral extraction"])),
    ("Construction",     ("Energy & Industry", ["construction", "civil engineering"])),
    ("HVAC Services",    ("Energy & Industry", ["hvac", "heating cooling"])),
    ("Packaging",        ("Energy & Industry", ["packaging", "labeling"])),
    ("Textiles",         ("Energy & Industry", ["textile mill", "fabric manufacturer"])),
    ("3D Printing",      ("Energy & Industry", ["3d printing", "additive manufacturing"])),
])

def _classify_industry(meta_txt: str) -> tuple[str, str]:
    # ---- single source of truth → LLM ----
    print("\n🛈  LLM-CLASSIFY  INPUT ↓")
    print(meta_txt[:800])                         #  debug

    result = _llm_classify(meta_txt)
    if result:
        industry, sub = result
        print("🛈  LLM-CLASSIFY  OUTPUT ↓")
        print({"industry": industry, "sub_industry": sub})   #  debug
        return industry, sub

    # fallback (should be rare – only if API/key fails)
    return "Media & Education", "General Media"

# ──────────────────────── CORE CRAWLER ──────────────────────────────
async def _crawl_domain(session, domain: str) -> list[dict]:
    """
    Crawl a domain, extract contacts & meta, return at most LEADS_PER_DOMAIN leads.
    """
    api_url = "https://api.firecrawl.dev/v1/crawl"
    payload = {"url": f"https://{domain}", "depth": 0, "parseContacts": True}
    async with session.post(api_url, json=payload, timeout=30) as response:
        if response.status != 200:
            raise RuntimeError(f"Firecrawl {response.status}")
        data = await response.json()

    contacts = data.get("contacts", []) or []
    meta_text = " ".join(filter(None, [
        data.get("title", ""),
        data.get("description", ""),
        " ".join(data.get("keywords", []))
    ]))

    industry, sub_ind = _classify_industry(meta_text)

    leads = []
    for contact in contacts:
        leads.append({
            "business": domain,
            "owner_full_name": contact.get("name", ""),
            "first": contact.get("first", ""),
            "last": contact.get("last", ""),
            "owner_email": contact["email"],
            "linkedin": contact.get("linkedin", ""),
            "website": contact.get("website") or f"https://{domain}",
            "industry": industry,
            "sub_industry": sub_ind,
            "region": "Global",
        })
        if len(leads) >= LEADS_PER_DOMAIN:
            break
    return leads

OPENROUTER_KEY = ARGS.openrouter_key

def _llm_classify(text: str) -> tuple[str, str] | None:
    """
    Ask Mistral-7B (openrouter.ai) to classify the company.
    Returns (industry, sub_industry) or None on any failure.
    """
    if not OPENROUTER_KEY:
        return None
    prompt_system = (
        "You are an industry classifier. "
        "Given short snippets (domain, page title / description, contact URL) "
        "return JSON ONLY: {\"industry\":\"<one of: Tech & AI | Finance & Fintech | "
        "Health & Wellness | Media & Education | Energy & Industry>\","
        " \"sub_industry\":\"<your best guess sub-industry>\"}"
    )
    prompt_user = textwrap.shorten(text, width=400, placeholder=" …")
    try:
        r = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "mistralai/mistral-7b-instruct",
                "messages": [
                    {"role": "system", "content": prompt_system},
                    {"role": "user", "content": prompt_user}
                ],
                "temperature": 0.2
            },
            timeout=20
        )
        r.raise_for_status()
        content = r.json()["choices"][0]["message"]["content"]
        j = json.loads(content.strip())
        ind  = j.get("industry")
        sub  = j.get("sub_industry") or j.get("subIndustry")
        if ind and sub:
            return ind, sub
    except Exception as e:
        logger.warning("llm_classify_failed", error=str(e)[:200])
    return None

if __name__ == "__main__":
    asyncio.run(miner()) 