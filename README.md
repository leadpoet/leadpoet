# Leadpoet | AI Sales Agents Powered by Bittensor

Leadpoet is Subnet 71, the decentralized AI sales agent subnet built on Bittensor. Leadpoet's vision is streamlining the top of sales funnel, starting with high-quality lead generation today and evolving into a fully automated sales engine where meetings with your ideal customers seamlessly appear on your calendar.

## Overview

Leadpoet transforms lead generation by creating a decentralized marketplace where:
- **Miners** source high-quality prospects using web scraping and AI
- **Validators** ensure quality through consensus-based validation
- **Buyers** access curated prospects optimized for their Ideal Customer Profile (ICP)

Unlike traditional lead databases, Leadpoet requires **consensus from multiple validators** before a lead is approved:
- Each prospect is validated by three independent validators
- Prevents gaming and ensures the lead pool limited to **verified, highest quality** leads

---

## Prerequisites

### Hardware Requirements
- **Miners/Validators**: 16GB RAM, 8-core CPU, 100GB SSD
- **Network**: Stable internet connection with open ports for axon communication

### Software Requirements
- Python 3.9 - 3.12       
- Bittensor CLI: `pip install bittensor>=9.10`
- Bittensor Wallet: `btcli wallet create`

## Required Credentials

### For Miners

```bash
# Required for Dynamic Lead Generation
export FIRECRAWL_API_KEY="your_firecrawl_key"        # Web scraping
export OPENROUTER_KEY="your_openrouter_key"          # AI classification
export SCRAPINGDOG_API_KEY="your_scrapingdog_key"    # Google Search (via ScrapingDog)

```

### For Validators

**TIP**: Copy `env.example` to `.env` and fill in your API keys for easier configuration.

```bash

# Email Validation API (REQUIRED)
# Truelist - Unlimited email validation: https://truelist.io/
export TRUELIST_API_KEY="your_truelist_key"

# LinkedIn Validation (REQUIRED)
# Uses ScrapingDog API for Google Search Engine results
# Get your API key at: https://www.scrapingdog.com/
export SCRAPINGDOG_API_KEY="your_scrapingdog_key"   # ScrapingDog API (for GSE searches)
export OPENROUTER_KEY="your_openrouter_key"          # openrouter.ai (for LLM verification)

# Reputation Score APIs (OPTIONAL - soft checks use mostly free public APIs)
# Note: Most reputation checks use free public APIs (Wayback, SEC, GDELT)
# UK Companies House API Key Setup:
# 1. Go to https://developer.company-information.service.gov.uk/get-started
# 2. Click "register a user account" -> "create sign in details" if you don't have an account
# 3. Either create a GOV.UK One Login or create sign in details without using GOV.UK One Login
# 4. Create your account
# 5. Once created, go to https://developer.company-information.service.gov.uk/manage-applications
# 6. Add an application with:
#    - Application name: "API Key"
#    - Description: "Requesting the Companies House API to verify eligibility of companies for <your company name>"
#    - Environment: "live"
export COMPANIES_HOUSE_API_KEY="your_companies_house_key"

```

See [`env.example`](env.example) for complete configuration template.

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/leadpoet/Leadpoet.git
cd Leadpoet

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate 

# 3. Install the packages

pip install --upgrade pip
pip install -e .

```

## For Miners

### Getting Started

1. **Register on subnet** (netuid 71):
```bash
btcli subnet register \
    --netuid 71 \
    --subtensor.network finney \
    --wallet.name miner \
    --wallet.hotkey default
```

2. **Publish your IP** (one-time setup):
```bash
python scripts/post_ip.py \
    --netuid 71 \
    --subtensor_network finney \
    --wallet_name miner \
    --wallet_hotkey default \
    --external_ip YOUR_PUBLIC_IP \
    --external_port 18091
```

3. **Run the miner**:
```bash
python neurons/miner.py \
    --wallet_name miner \
    --wallet_hotkey default \
    --wallet_path <your_wallet_path> \  # Optional: custom wallet directory (default: ~/.bittensor/wallets)
    --netuid 71 \
    --subtensor_network finney
```

### How Miners Work

1. **Continuous Sourcing**: Actively search for new prospects
2. **Secure Submission**: Get pre-signed S3 URL, hash lead data, sign with private key, and upload
3. **Consensus Validation**: Prospects validated by multiple validators using commit/reveal protocol
4. **Approved Leads**: Only consensus-approved leads enter the main lead pool

### Lead JSON Structure

Miners must submit prospects with the following structure:


```json
{
  "business": "Microsoft",                 # REQUIRED
  "full_name": "Satya Nadella",            # REQUIRED
  "first": "Satya",                        # REQUIRED
  "last": "Nadella",                       # REQUIRED
  "email": "satya@microsoft.com",          # REQUIRED
  "role": "CEO",                           # REQUIRED
  "website": "https://microsoft.com",      # REQUIRED
  "industry": "Technology",                # REQUIRED - must be from industry_taxonomy.py
  "sub_industry": "Software",              # REQUIRED - must be from industry_taxonomy.py
  "country": "United States",              # REQUIRED - see Country Format below
  "state": "Washington",                   # REQUIRED for US leads only
  "city": "Redmond",                       # REQUIRED for all leads
  "linkedin": "https://linkedin.com/in/satyanadella", # REQUIRED
  "company_linkedin": "https://linkedin.com/company/microsoft", # REQUIRED
  "source_url": "https://microsoft.com/about", # REQUIRED (URL where lead was found, OR "proprietary_database")
  "description": "Technology company developing software, cloud services, and AI solutions", # REQUIRED
  "employee_count": "10,001+",             # REQUIRED - valid ranges: "0-1", "2-10", "11-50", "51-200", "201-500", "501-1,000", "1,001-5,000", "5,001-10,000", "10,001+"
  "source_type": "company_site",
  "phone_numbers": ["+1-425-882-8080"],
  "founded_year": 1975,
  "ownership_type": "Public",
  "company_type": "Corporation",
  "number_of_locations": 100,
  "socials": {"twitter": "Microsoft"}
}
```

**Source URL:** Provide the actual URL where the lead was found. For proprietary databases, set both `source_url` and `source_type` to `"proprietary_database"`. LinkedIn URLs in `source_url` are blocked.

**Industry & Sub-Industry:** Must be exact values from `validator_models/industry_taxonomy.py`. The `sub_industry` key maps to valid parent `industries`.

**Country Format:**
- **US leads:** Require `country`, `state`, AND `city` (e.g., "United States", "California", "San Francisco")
- **Non-US leads:** Require `country` and `city` only (`state` is optional)
- **Accepted country names:** Use standard names like "United States", "United Kingdom", "Germany", etc. Common aliases are also accepted: "USA", "US", "UK", "UAE", etc.
- **199 countries supported** - see `gateway/api/submit.py` for the full list

### Lead Requirements

**Email Quality:**
- **Only "Valid" emails accepted** - Catch-all, invalid, and unknown emails will be rejected
- **No general purpose emails** - Addresses like hello@, info@, team@, support@, contact@ are not accepted
- **Proper email format required** - Must follow standard `name@domain.com` structure

**Name-Email Matching:**

Contact's first or last name must appear in the email address. We accept 26 common patterns plus partial matches to ensure quality while capturing the majority of legitimate business emails:

**Starting with first name:**
```
johndoe, john.doe, john_doe, john-doe
johnd, john.d, john_d, john-d
jdoe, j.doe, j_doe, j-doe
```

**Starting with last name:**
```
doejohn, doe.john, doe_john, doe-john
doej, doe.j, doe_j, doe-j
djohn, d.john, d_john, d-john
```

**Single tokens:**
```
john, doe
```

These strict requirements at initial go-live demonstrate our dedication to quality leads, while still capturing majority of good emails.

### Reward System

Miners earn rewards based on the **quality and validity** of leads they submit, with rewards weighted entirely by a rolling 30-epoch history to incentivize consistent long-term quality:

**How It Works:**
1. Each epoch, validators receive leads to validate
2. Validators run automated checks on all leads (email verification, domain checks, LinkedIn validation, reputation scoring)
3. Each validator calculates weights proportionally: miners who submitted **VALID** (approved) leads receive rewards
4. Rewards are weighted by each lead's reputation score (0-48 points: domain history, regulatory filings, and press coverage)
5. Formula: `miner_reward ∝ Σ(rep_score for all approved leads from that miner)`

**Example:** If Miner A submitted 3 valid leads (scores: 10, 15, 12) and Miner B submitted 2 valid leads (scores: 8, 20), then:
- Miner A total: 37 points
- Miner B total: 28 points
- Weights distributed proportionally: 57% to Miner A, 43% to Miner B


### Rejection Feedback

If your lead is rejected by validator consensus, you're able to access the reject reason explaining why. This helps you improve lead quality and increase approval rates.

**Query Your Rejections:**

```python
python3 - <<EOF
from Leadpoet.utils.cloud_db import get_rejection_feedback
import bittensor as bt

wallet = bt.wallet(name="miner", hotkey="default")
feedback = get_rejection_feedback(wallet, limit=10, network="finney", netuid=71)

print(f"\nFound {len(feedback)} rejection(s)\n")
for idx, record in enumerate(feedback, 1):
    summary = record['rejection_summary']
    print(f"[{idx}] Epoch {record['epoch_number']} - Rejected by {summary['rejected_by']}/{summary['total_validators']} validators")
    for failure in summary['common_failures']:
        print(f"    • {failure.get('check_name')}: {failure.get('message')}")
    print()
EOF
```

**Common Rejection Reasons & Fixes:**

| Issue | Fix |
|-------|-----|
| Invalid email format | Verify email follows `name@domain.com` format |
| Email from disposable provider | Use business emails only (no tempmail, 10minutemail, etc.) |
| Domain too new (< 7 days) | Wait for domain to age |
| Email marked invalid | Check for typos, verify email exists |
| Website not accessible | Verify website is online and accessible |
| Domain blacklisted | Avoid domains flagged for spam/abuse |

### Rate Limits & Cooldown

To maintain lead quality and prevent spam, we enforce daily submission limits server-side. Think of it as guardrails to keep the lead pool high-quality.

**Daily Limits (Reset at 12:00 AM UTC):**
- **1000 submission attempts per day** - Counts all submission attempts (including duplicates/invalid)
- **200 rejections per day** - Includes:
  - Duplicate submissions
  - Missing required fields
  - **Validator consensus rejections** - When validator consensus rejects your lead based on quality checks

**What Happens at Rate Limit:**

When you hit the rejection limit, all subsequent submissions are blocked until the daily reset at midnight UTC. All rate limit events are logged to the TEE buffer and permanently stored on Arweave for transparency.

---

## Qualification Model System (Lead Curation)

In addition to sourcing leads, miners can submit **qualification models** - AI/ML models that curate leads from the approved lead pool based on Ideal Customer Profiles (ICPs).

### How It Works

1. **Miner develops a model** that queries a leads database and finds leads matching given ICPs
2. **Miner submits the model** to the gateway (as a tarball) with a TAO payment
3. **Validators evaluate the model** by running it against 100 ICPs
4. **Model is scored** based on how well it finds matching leads
5. **Champion model** earns rewards for its curation ability

### Qualification Model Requirements

Your model must follow these **strict requirements**:

#### 1. Function Signature

Your model must expose a function named `find_leads` (or `qualify` for backwards compatibility):

```python
def find_leads(icp: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Find the best lead from the database matching the given ICP.
    
    CRITICAL: The 'prompt' field contains a NATURAL LANGUAGE description
    that your model must INTERPRET to find matching leads.
    
    Args:
        icp: Dict containing:
            # PRIMARY - Parse and interpret this!
            - prompt: str  
              Example: "VP Sales and Heads of Revenue at Series A-C SaaS 
                       companies in the US. Showing signals: researching 
                       outbound tools, hiring SDRs, or evaluating competitors."
            
            # Structured fields (for reference/validation)
            - industry: str (e.g., "Software")
            - sub_industry: str (e.g., "Enterprise Software")  
            - target_roles: List[str] (e.g., ["VP of Sales", "Head of Revenue"])
            - company_size: str (e.g., "100-500")
            - company_stage: str (e.g., "Series A")
            - geography: str (e.g., "United States, California")
            - country: str (e.g., "United States")
            - intent_signals: List[str] (e.g., ["hiring SDRs", "evaluating CRM"])
    
    Returns:
        Lead dict matching LeadOutput schema, or None if no match found
    """
```

**Important:** The `prompt` field is what real customers would type (e.g., "Who should we find for you?"). Your model should **parse and interpret** this natural language prompt to understand what's being asked, not just do direct database field lookups.

#### 2. Database Connection (DO NOT HARDCODE)

Your model receives database credentials via environment variables injected by the validator:

```python
# Config is injected into icp["_config"] at runtime
config = icp.get("_config", {})
supabase_url = config.get("SUPABASE_URL")
supabase_key = config.get("SUPABASE_ANON_KEY")
table_name = config.get("QUALIFICATION_LEADS_TABLE")

# Create client
from supabase import create_client
client = create_client(supabase_url, supabase_key)

# Query leads
result = client.table(table_name).select("*").eq("industry", "Technology").execute()
```

**CRITICAL:** Never hardcode database URLs or API keys. The validator injects these at runtime.

#### 3. Return Schema (LeadOutput) - STRICT

Your model must return a dict with **EXACTLY** these 14 fields - no more, no less:

> ⚠️ **CRITICAL:** Any extra fields = instant score 0. Models cannot fabricate person-level data (email, name, phone, etc.)

```python
{
    # Company info (ALL REQUIRED)
    "business": "Stripe",
    "company_linkedin": "https://linkedin.com/company/stripe",
    "company_website": "https://stripe.com",
    "employee_count": "1001-5000",  # Must match database format
    
    # Industry info (ALL REQUIRED)
    "industry": "Financial Services",
    "sub_industry": "Payment Processing",
    
    # Location (ALL REQUIRED - NOT a combined "geography" field)
    "country": "United States",
    "city": "San Francisco", 
    "state": "California",
    
    # Role info (ALL REQUIRED)
    "role": "VP of Engineering",
    "role_type": "Engineer/Technical",  # e.g., "C-Level Executive", "Sales", "Marketing"
    "seniority": "VP",  # Must be: "C-Suite", "VP", "Director", "Manager", "Individual Contributor"
    
    # Intent signal (REQUIRED)
    "intent_signal": {
        "source": "linkedin",  # One of: linkedin, job_board, social_media, news, github, review_site, company_website, other
        "description": "Hiring backend engineers for payments infrastructure",
        "url": "https://linkedin.com/jobs/123456",
        "date": "2026-01-15",  # ISO format YYYY-MM-DD
        "snippet": "Looking for senior engineers..."  # Optional
    }
}
```

**Required Fields Summary (14 total):**

| Field | Type | Example |
|-------|------|---------|
| `business` | str | "Stripe" |
| `company_linkedin` | str | "https://linkedin.com/company/stripe" |
| `company_website` | str | "https://stripe.com" |
| `employee_count` | str | "1001-5000" |
| `industry` | str | "Financial Services" |
| `sub_industry` | str | "Payment Processing" |
| `country` | str | "United States" |
| `city` | str | "San Francisco" |
| `state` | str | "California" |
| `role` | str | "VP of Engineering" |
| `role_type` | str | "Engineer/Technical" |
| `seniority` | enum | "VP" |
| `intent_signal` | object | See above |

**NOT ALLOWED (instant score 0 if included):**
- `email`, `full_name`, `first_name`, `last_name`, `phone`, `linkedin_url` (person-level PII)
- `geography` (use `country`/`city`/`state` instead)
- `company_size` (use `employee_count` instead)
- **ANY other field not listed above**

#### 4. Time & Cost Limits

- **8 seconds** maximum per ICP evaluation
- **$5.00 total** maximum for all 100 ICP evaluations
- Models exceeding limits receive score penalties or failures

### Test Database for Local Development

To develop and test your qualification model locally, we provide a **public test database** with 50,000 sample leads. This allows you to build and debug your model before submitting it for evaluation.

> **Note:** The test database intentionally includes some leads with bad data quality to test your model's robustness. A good model should filter out or handle these gracefully.

**Test Database Connection:**
```python
SUPABASE_URL = "https://qplwoislplkcegvdmbim.supabase.co"
SUPABASE_ANON_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InFwbHdvaXNscGxrY2VndmRtYmltIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDQ4NDcwMDUsImV4cCI6MjA2MDQyMzAwNX0.5E0WjAthYDXaCWY6qjzXm2k20EhadWfigak9hleKZk8"
TABLE_NAME = "test_leads_for_miners"
```

**Test Database Schema:**

| Column | Type | Description |
|--------|------|-------------|
| `id` | integer | Row identifier |
| `business` | text | Company name |
| `website` | text | Company website |
| `employee_count` | text | Employee count range |
| `role` | text | Contact's role/title |
| `role_type` | text | Role classification (e.g., "Senior Professional") |
| `industry` | text | Primary industry |
| `sub_industry` | text | Sub-industry |
| `city` | text | City |
| `state` | text | State/Province |
| `country` | text | Country |
| `company_linkedin` | text | Company LinkedIn URL |
| `description` | text | Company description |
| `last_refreshed_at` | timestamp | When the row was last updated |

**Note:** Personal information (email, name, personal LinkedIn) is NOT available - models find company+role matches, and can enrich contacts using external APIs.

### Sample Model

See `miner_qualification_models/sample_model/qualify.py` for a **foundation example** that demonstrates the basic patterns - miners should build upon this with more sophisticated ICP parsing, database querying, and intent signal discovery to achieve competitive scores.

The sample model:
1. **Parses the natural language prompt** to extract search criteria
2. **Queries the test database** using injected credentials
3. **Intelligently filters** by industry, sub-industry, and country
4. **Scores candidates** based on role match and prompt keywords
5. **Returns the best match** in LeadOutput format

The sample model demonstrates:
- How to parse prompts like "VP Sales at early-stage SaaS companies in the US"
- How to map prompt keywords to database queries
- How to score leads based on prompt relevance

### Model Requirements (Quick Reference)

**File Structure:**
```
your_model/
├── qualify.py          # Required: must contain find_leads() or qualify()
└── requirements.txt    # Optional: additional dependencies
```

**Size Limit:** Model tarball must be under **200KB**. Submissions exceeding this limit will be rejected.

**Required Function:**
```python
def find_leads(icp: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Find a lead matching the ICP's natural language prompt.
    
    Config is injected in icp["_config"]:
        - SUPABASE_URL, SUPABASE_ANON_KEY - Database credentials
        - QUALIFICATION_LEADS_TABLE - Table name (use "test_leads_for_miners" for local testing)
        - PROXY_URL - For paid API calls (e.g., "http://localhost:8001")
    
    Returns: Dict matching LeadOutput schema, or None
    """
    config = icp.get("_config", {})
    supabase_url = config.get("SUPABASE_URL")
    table_name = config.get("QUALIFICATION_LEADS_TABLE")
    proxy_url = config.get("PROXY_URL")
    # ... your logic ...
```

**Paid API Calls (via Proxy):**
```python
# DON'T call APIs directly - use the proxy (no API key needed)
# proxy_url is already "http://localhost:8000/qualification/proxy"
response = httpx.post(
    f"{proxy_url}/openrouter/chat/completions",  # NOT /proxy/openrouter/
    json={"model": "openai/gpt-4o-mini", "messages": [...]}
)  # Proxy injects API key server-side
```

**Allowed Libraries:** `os`, `sys`, `json`, `re`, `datetime`, `time`, `math`, `random`, `collections`, `typing`, `pandas`, `numpy`, `pydantic`, `requests`, `httpx`, `aiohttp`, `supabase`, `postgrest`, `duckduckgo_search`, `openai`, `fuzzywuzzy`, `rapidfuzz`, `dateutil`

**Blocked Libraries:** `subprocess`, `ctypes`, `pickle`, `marshal`, `socket`, `multiprocessing`, `shutil`, `pathlib`, `glob`, `tempfile`

**Blocked Patterns:** `eval()`, `exec()`, `__import__()`, `os.system()`, `os.popen()`, accessing `.bittensor`, `.ssh`, `/proc/self/environ`

> **Security:** Models are scanned on upload (gateway) AND at runtime (validator sandbox). Obfuscation attempts are caught by the runtime sandbox.

**Allowed APIs:**
| Type | APIs |
|------|------|
| Free | GitHub, StackOverflow, DuckDuckGo, SEC EDGAR, Wayback Machine, GDELT, UK Companies House, Wikipedia, Wikidata |
| Paid (via proxy) | OpenRouter, ScrapingDog, BuiltWith, Crunchbase, Desearch, NewsAPI, SimilarTech |

### Submitting Your Model

```bash
# Package your model
cd your_model_directory
tar -czvf my_model.tar.gz .

# Submit via miner
python neurons/miner.py submit-model \
    --model_path my_model.tar.gz \
    --wallet_name miner \
    --wallet_hotkey default
```

---

## For Validators

### Getting Started

1. **Stake Alpha / TAO** (meet base Bittensor validator requirements):
```bash
btcli stake add \
    --amount <amount> \
    --subtensor.network finney \
    --wallet.name validator \
    --wallet.hotkey default
```

2. **Register on subnet**:
```bash
btcli subnet register \
    --netuid 71 \
    --subtensor.network finney \
    --wallet.name validator \
    --wallet.hotkey default
```

3. **Run the validator**:
```bash
python neurons/validator.py \
    --wallet_name validator \
    --wallet_hotkey default \
    --wallet_path <your_wallet_path> \  # Optional: custom wallet directory (default: ~/.bittensor/wallets)
    --netuid 71 \
    --subtensor_network finney
```

Note: Validators are configured to auto-update from GitHub on a 5-minute interval.

### Consensus Validation System

Validators receive batches of ~50 leads per epoch. Each validator independently validates leads using a commit/reveal protocol (submit hashed decisions, then reveal actual decisions). Majority agreement is required for consensus. Approved leads move to the main database, rejected leads are discarded.

**Eligibility for Rewards:**
- Must participate in consensus validation epochs consistently and remain in consensus.

**Validators perform multi-stage quality checks:**
1. **Email validation**: Format, domain, disposable check, deliverability check
2. **Company & Contact verification**: Website, LinkedIn, Google search

Validators must label leads with valid emails as "Valid" or "valid".

### Auditor Validator

For validators who want to run a lightweight alternative that copies TEE-verified weights from the primary validator:

```bash
python neurons/auditor_validator.py \
    --netuid 71 \
    --subtensor.network finney \
    --wallet.name validator \
    --wallet.hotkey default
```

**How it works:**
- Fetches weight bundles from the gateway (signed by primary validator's TEE)
- Verifies Ed25519 signature and recomputes hash (doesn't trust claimed hash)
- Verifies AWS Nitro attestation (proves weights came from real enclave)
- Submits verified weights to chain
- Auto-updates from GitHub on each restart

**Trust Model:**
- AWS certificate chain verified (proves REAL Nitro enclave)
- COSE signature verified (proves authentic attestation)  
- Ed25519 signature verified (proves weights from enclave)
- Epoch binding verified (replay protection)
- Soft anti-equivocation check (retroactively verifies bundle weights match on-chain weights)

This is for validators who want to participate in consensus without running the full validation logic and paying the costs associated with it.

## 🔐 Gateway Verification & Transparency

**Verify Gateway Integrity**: Run `python scripts/verify_attestation.py` to verify the gateway is running canonical code (see [`scripts/VERIFICATION_GUIDE.md`](scripts/VERIFICATION_GUIDE.md) for details).

**Query Immutable Logs**: Run `python scripts/decompress_arweave_checkpoint.py` to view complete event logs from Arweave's permanent, immutable storage.

## Reward Distribution

### Consensus-Based Rewards

1. Validators participate in epoch-based consensus validation using commit/reveal protocol
2. Miner weights calculated based on approved leads sourced
3. Validators compute and commit weights on-chain proportional to leads sourced

### Security Features

- **TEE Gateway**: All events logged through hardware-protected Trusted Execution Environment
- **Immutable transparency**: Events permanently stored on Arweave with cryptographic proofs
- **Commit/Reveal protocol**: Prevents validators from copying each other's decisions
- **Consensus requirement**: Majority validator agreement, weighted by stake and v_trust, is required for lead approval

## Data Flow

```
Miner Sources Leads → Submit to TEE Gateway (S3 Upload) → 
Epoch Assignment → Validators Validate (Commit/Reveal) 
```

## Troubleshooting

Common Errors:

**Validator not receiving epoch assignments**
- Ensure validator is registered on subnet with active stake
- Check that validator is running latest code version (auto-updates every 5 minutes)
- Verify axon is accessible and ports are open

**Lead submission rejected**
- Check lead meets all requirements (valid email, name-email matching, required fields)
- Verify you haven't hit daily rate limits (1000 submissions, 200 rejections per day)
- Check gateway logs on Arweave for specific rejection reasons

**Consensus results not appearing**
- Wait for current epoch to complete (~72 minutes / 360 blocks)
- Check transparency log on Arweave for CONSENSUS_RESULT events
- Run `python scripts/decompress_arweave_checkpoint.py` to view recent results

## Support

For support and discussion:
- **Leadpoet FAQ**: Check out our FAQ at www.leadpoet.com/faq to learn more about Leadpoet!
- **Bittensor Discord**: Join the Leadpoet SN71 channel and message us!
- **Email**: hello@leadpoet.com

## License

MIT License - See LICENSE file for details


