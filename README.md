# Leadpoet | AI Sales Agents Powered by Bittensor

Leadpoet is Subnet 71, the decentralized AI sales agent subnet built on Bittensor. Leadpoet's vision is streamlining the top of the sales funnel, starting with high-quality lead generation today and evolving into a fully automated sales engine where meetings with your ideal customers seamlessly appear on your calendar.

## Overview

Leadpoet runs two complementary miner tracks on a single subnet:

- **Model Competition** — Miners submit AI/ML models that surface in-market **companies** from the open web that match a buyer's ICP and show genuine intent signals. The best-scoring model becomes the **champion** and earns rewards while it holds the crown.
- **Fulfillment** — Miners compete head-to-head on real, paid **client requests** for fully enriched leads (contact, company, intent evidence). Top-scoring leads per request earn rewards over a 100-epoch runway.

Both tracks are validated by **independent validators** running the same scoring pipeline (ICP fit + data accuracy + intent evidence), so quality is measured the same way across the subnet.

---

## Prerequisites

### Hardware Requirements
- **Validators**: 64GB RAM, 8-core CPU, 100GB SSD, AWS Nitro Enclaves enabled instance
- **Miners**: Variable depending on your model — no strict minimum
- **Network**: Stable internet connection

### Software Requirements
- Python 3.9 - 3.12
- Bittensor CLI: `pip install bittensor>=9.10`
- Bittensor Wallet: `btcli wallet create`

## Required Credentials

### For Miners

Miners choose their own tools and APIs for sourcing companies and enriching leads. Common examples include web scraping APIs (ScrapingDog, Firecrawl, Apify), LLMs (OpenRouter), and search APIs — but miners are free to use any approach (that is in compliance with our ToS).

For **qualification models** (Model Competition), paid API calls (LLM, ScrapingDog, etc.) go through the validator's proxy which injects keys server-side. Your model never needs API keys directly.

For **Fulfillment**, miners run their own infrastructure end-to-end (sourcing, enrichment, intent evidence collection) and pay for their own API calls.

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
git clone https://github.com/leadpoet/leadpoet.git
cd leadpoet

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

2. **Run the miner**:
```bash
python neurons/miner.py \
    --wallet_name miner \
    --wallet_hotkey default \
    --wallet_path <your_wallet_path> \  # Optional: custom wallet directory (default: ~/.bittensor/wallets)
    --netuid 71 \
    --subtensor_network finney
```

The miner participates in both Model Competition (if you've submitted a model) and Fulfillment (if you respond to client requests). The two tracks are independent and you can run either or both.

---

## Model Competition (Company Discovery)

Miners submit **qualification models** — AI/ML models that surface in-market companies from the open web matching a buyer's ICP. Models compete continuously: the highest-scoring model becomes the **champion** and earns rewards until a new model beats it by the published threshold.

### How It Works

1. **Miner builds a model** that takes an ICP and returns a single best-matching company plus verifiable intent evidence
2. **Miner submits the model** to the gateway (as a tarball) with a TAO payment
3. **Validators evaluate the model** by running it against 100 fresh ICPs
4. **Model is scored** on ICP fit, intent-signal quality, cost, and runtime
5. **Champion model** holds the crown and earns rewards until dethroned

### Model Requirements

Your model must follow these **strict requirements**:

#### 1. Function Signature

Your model must expose a function named `find_leads` (or `qualify` for backwards compatibility):

```python
def find_leads(icp: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Find the best-fit company from the open web for the given ICP.

    CRITICAL: The 'prompt' field contains a NATURAL LANGUAGE description
    that your model must INTERPRET to find matching companies.

    Args:
        icp: Dict containing:
            # PRIMARY - Parse and interpret this!
            - prompt: str
              Example: "Series A-C SaaS companies in the US showing
                       signals of hiring SDRs or evaluating outbound
                       sales tools."

            # Structured fields (for reference/validation)
            - industry: str (e.g., "Software")
            - sub_industry: str (e.g., "Enterprise Software")
            - employee_count: str (e.g., "51-200")
            - company_stage: str (e.g., "Series A")
            - country: str (e.g., "United States")
            - intent_signals: List[str] (e.g., ["hiring SDRs", "evaluating CRM"])

    Returns:
        Company dict matching the CompanyOutput schema, or None if no match found
    """
```

Your model should **parse and interpret** the natural-language `prompt`, not just do direct field lookups. Contact enrichment (person, email, phone) is **out of scope** for Model Competition — fulfillment miners layer that on top.

#### 2. Return Schema (CompanyOutput) - STRICT

Your model must return a dict with **EXACTLY** these fields — no more, no less:

> ⚠️ **CRITICAL:** Any extra fields = instant score 0. Person-level fields (email, name, phone, linkedin_url) are not allowed in Model Competition.

```python
{
    # Company info (ALL REQUIRED)
    "business": "Stripe",
    "company_linkedin": "https://linkedin.com/company/stripe",
    "company_website": "https://stripe.com",
    "employee_count": "1001-5000",

    # Industry info (ALL REQUIRED)
    "industry": "Financial Services",
    "sub_industry": "Payment Processing",

    # Company HQ Location (ALL REQUIRED — NOT a combined "geography" field)
    "country": "United States",
    "city": "San Francisco",
    "state": "California",

    # Intent signals (REQUIRED - list of one or more signals)
    # IMPORTANT: `source` must match the URL's domain.
    # Tagging a PRNewswire/TechCrunch/TheBlock article as "company_website"
    # auto-rejects it as fabricated.
    "intent_signals": [
        {
            "source": "linkedin",  # One of: linkedin, job_board, social_media, news, github, review_site, company_website, wikipedia, other
            "description": "Hiring backend engineers for payments infrastructure",
            "url": "https://linkedin.com/jobs/123456",
            "date": "2026-01-15",  # ISO format YYYY-MM-DD, or null if no verifiable date
            "snippet": "Looking for senior engineers to scale our payments platform..."  # REQUIRED
        }
    ]
}
```

Each intent signal object has 5 **required** fields: `source`, `description`, `url`, `date`, `snippet`. You can provide multiple intent signals per company — each is scored independently and the best one is used.

**NOT ALLOWED (instant score 0 if included):**
- `email`, `full_name`, `first_name`, `last_name`, `phone`, `linkedin_url` (person-level PII)
- `role`, `role_type`, `seniority` (person-level role data)
- `lead_id` (Model Competition surfaces companies from the open web, not rows from a database)
- `geography` (use `country`/`city`/`state` instead)
- `company_size` (use `employee_count` instead)
- `intent_signal` (singular — use `intent_signals` list instead)
- **ANY other field not listed above**

#### 3. Time & Cost Limits

- **8 seconds** maximum per ICP evaluation
- **$5.00 total** maximum for all 100 ICP evaluations
- Models exceeding limits receive score penalties or failures

### Expired ICP Sets (Debug Your Model)

Once an ICP set expires (after its 24-hour evaluation window), it becomes publicly available via the `qualification_expired_icp_sets` view. Use this to replay past evaluations locally and debug your model's scoring.

```python
import requests

url = "https://qplwoislplkcegvdmbim.supabase.co/rest/v1/qualification_expired_icp_sets"
headers = {"apikey": SUPABASE_ANON_KEY}

# Most recent expired set
resp = requests.get(url, headers=headers, params={"select": "*", "limit": "1"})
icp_set = resp.json()[0]

# Get a specific day's ICPs
resp = requests.get(url, headers=headers, params={"select": "*", "set_id": "eq.20260513"})
```

Each row contains `set_id`, `active_from`, `active_until`, and the full `icps` array (100 ICP prompts with industry, geography, intent signals, etc.). Active ICP sets are never exposed.

### Quick-Start Model Template

Here's a minimal working model to get you started. Create a `qualify.py` file:

```python
import os
import httpx

def find_leads(icp):
    config = icp.get("_config", {})
    proxy_url = config.get("PROXY_URL", "http://localhost:8001")

    # Use the validator's proxy for paid APIs (no key needed — proxy injects)
    response = httpx.post(
        f"{proxy_url}/openrouter/chat/completions",
        json={
            "model": "openai/gpt-4o-mini",
            "messages": [{
                "role": "user",
                "content": f"Find one real US-based company that matches: {icp.get('prompt', '')}",
            }],
        },
        timeout=8.0,
    )
    # ... parse response, scrape candidate company, verify intent evidence ...

    return {
        "business": "ExampleCo",
        "company_linkedin": "https://linkedin.com/company/exampleco",
        "company_website": "https://exampleco.com",
        "employee_count": "51-200",
        "industry": icp.get("industry", ""),
        "sub_industry": icp.get("sub_industry", ""),
        "country": "United States",
        "city": "San Francisco",
        "state": "California",
        "intent_signals": [{
            "source": "company_website",
            "description": "ExampleCo announced Series B funding",
            "url": "https://exampleco.com/blog/series-b",
            "date": "2026-04-12",
            "snippet": "Today we're excited to announce our $30M Series B led by ..."
        }]
    }
```

This is a **starting point** — competitive models should have sophisticated ICP parsing, multi-source intent discovery, and intelligent candidate ranking.

### Model Requirements (Quick Reference)

**File Structure:**
```
your_model/
├── qualify.py          # Required: must contain find_leads() or qualify()
└── requirements.txt    # Optional: additional dependencies
```

**Size Limit:** Model tarball must be under **200KB**. Submissions exceeding this limit will be rejected.

**Paid API Calls (via Proxy):**
```python
# DON'T call APIs directly - use the proxy (no API key needed)
proxy_url = config.get("PROXY_URL", "http://localhost:8001")
response = httpx.post(
    f"{proxy_url}/openrouter/chat/completions",
    json={"model": "openai/gpt-4o-mini", "messages": [...]}
)  # Proxy injects API key server-side
```

**Allowed Libraries (key ones):** `os`, `sys`, `json`, `re`, `datetime`, `time`, `math`, `random`, `string`, `collections`, `itertools`, `functools`, `typing`, `dataclasses`, `enum`, `uuid`, `hashlib`, `base64`, `copy`, `csv`, `io`, `logging`, `difflib`, `pathlib`, `asyncio`, `threading`, `concurrent`, `urllib`, `ssl`, `http`, `html`, `requests`, `httpx`, `aiohttp`, `duckduckgo_search`, `openai`, `pandas`, `numpy`, `pydantic`, `fuzzywuzzy`, `rapidfuzz`, `thefuzz`, `Levenshtein`, `dateutil`, `bs4`, `lxml`, `html5lib`, `soupsieve`, `certifi`, `cryptography`, `jwt`

Full allowlist: [`qualification/validator/sandbox_security.py`](qualification/validator/sandbox_security.py) `ALLOWED_LIBRARIES`

**Blocked Libraries:** `subprocess`, `ctypes`, `cffi`, `pickle`, `marshal`, `multiprocessing`, `shutil`, `glob`, `importlib.machinery`

**Blocked Patterns:** `eval()`, `exec()`, `__import__()`, `os.system()`, `os.popen()`, accessing `.bittensor`, `.ssh`, `/proc/self/environ`

> **Security:** Models are scanned on upload (gateway) AND at runtime (validator sandbox). Models that call APIs not on the allowlist are terminated after 10 blocked attempts. Obfuscation attempts are caught by the runtime sandbox. Hardcoded/gaming models are detected by LLM analysis before execution.

#### Prohibited Practices (Instant Ban)

Models that manipulate quality signals will be **banned** and the hotkey blacklisted. Specifically:

| Violation | Example | Why It's Banned |
|-----------|---------|-----------------|
| **Stripping negative LLM assessments** | Using `re.sub` to delete phrases like "no specific evidence" from descriptions | Hides honest verification failures to make bad evidence look good |
| **Fabricating dates** | Assigning `date.today()` when no date exists in the evidence | Games the recency score — stale content appears fresh |
| **Injecting fake intent signals** | Defaulting to `"hiring, funding, expansion"` when the ICP doesn't specify any | Searches for evidence the buyer never asked for, then presents it as relevant |
| **Fabricating evidence text** | Using `f"{company} hiring {title}"` instead of verbatim scraped text | Constructs fake descriptions not found in the source URL |
| **Bypassing LLM verification** | Fallback layers that skip verification and accept any 50+ chars of website text | Submits unverified content as "evidence" |
| **Defaulting verification to pass** | `claim_supported = parsed.get("claim_supported", True)` | When verification fails/is ambiguous, assumes it passed |

**What good models do instead:**
- Return `None` when no genuine intent evidence exists for an ICP
- Use only verbatim text extracted from real sources as descriptions
- Set the date field to `null` if no verifiable date is found (the field is optional)
- Respect LLM verification results — if the LLM says "no evidence," don't submit that company
- Only search for intent signals that the ICP actually requested

**Allowed APIs:**
| Type | APIs |
|------|------|
| Free (direct) | DuckDuckGo, SEC EDGAR, Wayback Machine, GDELT, UK Companies House, Wikipedia, Wikidata |
| Paid (via proxy) | OpenRouter, ScrapingDog, BuiltWith, Crunchbase, Desearch, Data Universe (Macrocosmos), NewsAPI, Jobs Data API (TheirStack), Apify |

### Submitting Your Model

```bash
# Package your model
cd your_model_directory
tar -czvf my_model.tar.gz .
```

Model submission is handled through the gateway API. See the miner code for the submission flow.

---

## Fulfillment (Direct Client Requests)

Fulfillment is the second incentive mechanism: miners compete directly on real, paid client requests. Instead of surfacing companies, fulfillment miners deliver **fully enriched leads** — company, contact, role, verified email, and intent evidence — for a specific ICP.

### How It Works

1. **Request Published** — A client submits a fulfillment request with a specific ICP (industry, roles, seniority, geography, intent signals). The request is published globally for all miners.

2. **Commit Window (1 epoch, ~72 minutes)** — Miners source leads matching the ICP, then submit **hashed lead data** (commit). This prevents other miners from copying your leads.

3. **Reveal Window (15 minutes after commit closes)** — Miners reveal the actual lead data corresponding to their committed hashes. Hashes must match or the submission is rejected.

4. **Validator Scoring** — Validators score every revealed lead through a three-tier pipeline:
   - **Tier 1 (ICP Fit)** — Industry, sub-industry, role, seniority, employee count, country must match the request
   - **Tier 2 (Data Accuracy)** — Email verification (TrueList), LinkedIn/person verification (ScrapingDog), company verification, reputation score
   - **Tier 3 (Intent Scoring)** — Each intent signal URL is fetched and verified. An LLM evaluates relevance. Signals are scored and aggregated with time decay. Minimum threshold: 5.0

5. **Winner Selection & Rewards** — Leads are ranked by score, deduplicated by company. The top `num_leads` (as requested by the client) are selected as winners. Each winning lead earns **0.05% of emission per epoch for 100 epochs** (about 5 days of payout per lead). Ties on the same company split the reward. The longer runway is intentional: a single winning lead keeps a miner earning emission for ~5 days, protecting active miners from de-registration during low-volume windows.

### Fulfillment Request Schema (What Miners See)

When a client submits a request, miners receive an ICP with this structure:

```json
{
  "prompt": "VP of Sales and Heads of Revenue at Series A-C SaaS companies in the US showing signals of evaluating outbound sales tools, hiring SDRs, or researching competitors.",
  "industry": "Software",
  "sub_industry": "SaaS",
  "target_role_types": ["Sales", "Business Development"],
  "target_roles": ["VP of Sales", "Head of Revenue", "Director of Sales"],
  "target_seniority": "VP",
  "employee_count": "50-500",
  "company_stage": "Series A",
  "geography": "United States",
  "country": "United States",
  "product_service": "outbound sales automation platform",
  "intent_signals": [
    "hiring SDRs",
    {"text": "evaluating sales tools", "required": true, "is_scored": true},
    {"text": "ships to Asia", "required": true, "is_scored": false}
  ],
  "num_leads": 2
}
```

- `prompt` — Natural language description of the ideal lead. Your model should interpret this.
- `target_roles` — Exact role titles the client wants. Your lead's `role` must match one of these (fuzzy matching is applied, e.g. "VP, Corporate Sales" matches "VP of Sales").
- `target_seniority` — Required seniority level.
- `intent_signals` — The types of buying signals the client cares about. Each entry can be a plain string (default: optional, scored) or a structured object `{"text", "required", "is_scored"}`:
  - `required=true` — the lead **must** produce verified evidence for this signal or it fails with `missing_required_intent_signal`.
  - `is_scored=false` — binary pass/fail; verified evidence is required if the signal is also `required`, but the signal does not contribute to the numeric intent score.
- `num_leads` — How many winning leads the client wants. Only the top N by score earn rewards.

### Fulfillment Lead Schema

Miners must submit leads with this exact structure via the commit-reveal endpoints:

```json
{
  "full_name": "Jane Smith",
  "email": "jsmith@company.com",
  "linkedin_url": "https://linkedin.com/in/janesmith",
  "phone": "",

  "business": "Company Inc",
  "company_linkedin": "https://linkedin.com/company/company-inc",
  "company_website": "https://company.com",
  "employee_count": "501-1,000",

  "company_hq_country": "United States",
  "company_hq_state": "California",
  "company_hq_city": "San Francisco",

  "industry": "Software",
  "sub_industry": "SaaS",

  "description": "Company Inc is a cloud-based SaaS platform that helps mid-market sales teams automate outbound prospecting with AI-driven email sequencing, deal tracking, and conversation intelligence.",

  "country": "United States",
  "city": "Austin",
  "state": "Texas",

  "role": "VP of Sales",
  "role_type": "Sales",
  "seniority": "VP",

  "intent_signals": [
    {
      "source": "job_board",
      "description": "Company Inc hiring Sales Development Representatives",
      "url": "https://jobs.lever.co/company/abc123",
      "date": null,
      "snippet": "Sales Development Representative - Full Time. We are looking for driven SDRs to join our growing sales team."
    }
  ]
}
```

**Key fields:**
- `city`/`state`/`country` — The **contact's** location (from their LinkedIn profile), not the company HQ
- `company_hq_city`/`company_hq_state`/`company_hq_country` — The **company's** headquarters location
- `industry`/`sub_industry` — Must match values from `validator_models/industry_taxonomy.py`
- `description` — **Required**, min 30 characters. A free-form company description written by the miner. Fed to the validator's Stage 5 3-stage classification pipeline (`validator_models/stage5_verification.py::classify_company_industry`): an LLM compares it against the scraped website/LinkedIn content; if the LLM decides the two don't describe the same business, the lead is rejected with `stage1_invalid_description` before intent scoring runs.
- `role_type` — One of: `C-Level Executive`, `VP`, `Director`, `Manager`, `Sales`, `Marketing`, `Engineering`, `Product`, `Operations`, `Finance`, `HR`, `Legal`, `IT`, `Customer Success`, `Business Development`, `Data & Analytics`, `Design`, `Research`, `Supply Chain`, `Consulting`, `Other`
- `seniority` — One of: `C-Suite`, `VP`, `Director`, `Manager`, `Individual Contributor`
- `intent_signals` — At least one signal required. Each signal needs `source`, `description`, `url`, `date` (ISO format or null), and `snippet` (verbatim text from the URL)

#### Picking the right `source` for each URL

The validator enforces that the URL's domain must match the `source` you claim. Mis-tagging a real signal with the wrong source gets the lead **auto-rejected as fabricated** (confidence=0, intent score=0). Use this table:

| If the URL is on... | Use `source` |
|---------------------|--------------|
| The lead's own company site (same domain as `company_website`, e.g. `acme.com/blog/...`) | `company_website` |
| A jobs page — Lever, Greenhouse, Indeed, the company's own `/careers` page | `job_board` |
| A press release or news article — PRNewswire, GlobeNewswire, TechCrunch, TheBlock, Bizjournals, local news outlets | `news` |
| LinkedIn (any post, company page, or job listing) | `linkedin` |
| Twitter / X / Threads / Instagram / Facebook / TikTok | `social_media` |
| GitHub (repo, release, or organization page) | `github` |
| G2, Capterra, TrustRadius, Glassdoor, Trustpilot | `review_site` |
| Wikipedia | `wikipedia` |
| Anything else with a verifiable dated event | `other` |

**The most common mistake**: tagging a third-party news article (PRNewswire, TechCrunch, TheBlock, etc.) as `company_website`. The validator checks that the signal URL's domain matches the lead's `company_website` field. If it doesn't, the signal is auto-rejected as fabricated. **For third-party news, always use `source: news`.**

**Examples:**

```jsonc
// CORRECT: company's own blog post
{ "source": "company_website", "url": "https://acme.com/blog/series-b-announcement", ... }

// WRONG: third-party press release tagged as company_website -> auto-fabricated
{ "source": "company_website", "url": "https://www.prnewswire.com/news/acme-raises-50m", ... }

// CORRECT: same press release with the right source
{ "source": "news",            "url": "https://www.prnewswire.com/news/acme-raises-50m", ... }

// CORRECT: LinkedIn job posting
{ "source": "linkedin",        "url": "https://linkedin.com/jobs/view/12345", ... }

// WRONG: LinkedIn job tagged as company_website -> auto-fabricated
{ "source": "company_website", "url": "https://linkedin.com/jobs/view/12345", ... }
```

What also matters: the `snippet` must be **verbatim text from the URL** (a literal substring of the page body), and the `description` must be grounded in that snippet. Generic boilerplate like "Acme is a leading provider of..." pulled from a homepage About page won't pass — the verifier looks for a specific dated event (funding, hire, expansion, product launch, etc.).

### Commit-Reveal Flow

```
POST /fulfillment/commit
  request_id, miner_hotkey, lead_hashes[], signature, nonce, timestamp

POST /fulfillment/reveal
  request_id, submission_id, miner_hotkey, leads[], signature, nonce, timestamp
```

The commit hash is computed from the lead JSON using the schema defined in `Leadpoet/utils/hashing.py`. Leads must be revealed within the reveal window or they are discarded.

### Scoring Details

| Stage | What's Checked | Cost |
|-------|---------------|------|
| Tier 1 | Industry, sub-industry, role, seniority, country, employee count, duplicate company | Free |
| Tier 2 | Email format, name-in-email, domain age, MX/SPF/DMARC, DNSBL, TrueList verification, LinkedIn person verification, company verification, reputation score | API calls |
| Tier 3 | Each intent signal URL scraped, snippet overlap verified, LLM evaluates relevance, time decay applied, peak-weighted aggregation, required-signal gate enforced | LLM + scraping |

### Foundation Model

A reference implementation is available at `miner_models/Main_fulfillment_model/`. This is a **foundation to build off of** — it demonstrates the full pipeline (company discovery, contact search, email verification, intent signal mining, self-correction) but is not production-ready. Competitive miners should build their own sourcing strategies.

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

### Validation Pipeline

Validators run the same multi-stage scoring pipeline across both miner tracks:

1. **Email validation**: Format, domain, disposable check, deliverability via TrueList (fulfillment leads only)
2. **Company & Contact verification**: Website, LinkedIn, Google search via ScrapingDog
3. **Intent verification**: URL fetch + LLM relevance scoring + snippet overlap check + time decay
4. **Reputation scoring**: Wayback Machine, SEC EDGAR, GDELT, WHOIS/DNSBL, Companies House

Model Competition runs the same pipeline against company-only outputs (no person/email verification). Fulfillment runs the full pipeline against fully enriched leads.

**Eligibility for Rewards:**
- Must participate in consensus validation epochs consistently and remain in consensus.

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

### Model Competition
- Champion model earns rewards while it holds the crown.
- A new model must beat the current champion's score by the published threshold to take over.

### Fulfillment
- Each winning lead earns **0.05% of emission per epoch for 100 epochs** (~5 days of payout per winning lead).
- Top `num_leads` per request are selected; ties on the same company split the reward.

### Security Features

- **TEE Gateway**: All events logged through hardware-protected Trusted Execution Environment
- **Immutable transparency**: Events permanently stored on Arweave with cryptographic proofs
- **Commit/Reveal protocol**: Prevents miners from copying each other's fulfillment leads
- **Validator consensus**: Majority validator agreement, weighted by stake and v_trust, is required for fulfillment winner selection

## Data Flow

```
Model Competition:
  Miner submits model → Gateway sandboxes & scans → Validators evaluate
  against 100 ICPs → Champion crowned / dethroned

Fulfillment:
  Client publishes request → Miners commit (hashed) → Miners reveal →
  Validators score (Tier 1 / Tier 2 / Tier 3) → Top N leads win
```

## Troubleshooting

Common Errors:

**Validator not receiving epoch assignments**
- Ensure validator is registered on subnet with active stake
- Check that validator is running latest code version (auto-updates every 5 minutes)

**Fulfillment lead rejected**
- Check the rejection reason in the public dashboard or `fulfillment_score_consensus` table
- Common causes: `tier1_role_mismatch`, `email_not_valid`, `missing_required_intent_signal`, `insufficient_intent`, `stage1_invalid_description`

**Model evaluation failed**
- Check the model is under 200KB, only uses allowed libraries, and doesn't call APIs outside the allowlist
- Inspect the `qualification_models` row for `status` and any error fields

**Consensus results not appearing**
- Wait for the current epoch to complete (~72 minutes / 360 blocks)
- Check the transparency log on Arweave for CONSENSUS_RESULT events
- Run `python scripts/decompress_arweave_checkpoint.py` to view recent results

## Support

For support and discussion:
- **Leadpoet FAQ**: Check out our FAQ at www.leadpoet.com/faq to learn more about Leadpoet!
- **Bittensor Discord**: Join the Leadpoet SN71 channel and message us!
- **Email**: hello@leadpoet.com

## License

MIT License - See LICENSE file for details
