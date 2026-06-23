# Leadpoet | AI Sales Agents Powered by Bittensor

Leadpoet is Subnet 71, the decentralized AI sales agent subnet built on Bittensor. Leadpoet's vision is streamlining the top of the sales funnel, starting with high-quality lead generation today and evolving into a fully automated sales engine where meetings with your ideal customers seamlessly appear on your calendar

## Overview

Leadpoet's active miner track is **Fulfillment**:

- **Fulfillment** — Miners compete head-to-head on real, paid **client requests** for fully enriched leads (contact, company, intent evidence). Top-scoring leads per request earn rewards over a 100-epoch runway.

Fulfillment is validated by **independent validators** running a shared scoring pipeline (ICP fit + data accuracy + intent evidence), so quality is measured the same way across the subnet.

> **Note:** The Model Competition track (miner-submitted company-discovery models) is currently **inactive** — model submissions are closed.

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

The miner participates in **Fulfillment** by responding to real client requests.

---

## Fulfillment (Direct Client Requests)

Fulfillment is the second incentive mechanism: miners compete directly on real, paid client requests. Instead of surfacing companies, fulfillment miners deliver **fully enriched leads** — company, contact, role, verified email, and intent evidence — for a specific ICP.

### How It Works

1. **Request Published** — A client submits a fulfillment request with a specific ICP (industry, roles, seniority, geography, intent signals). The request is published globally for all miners.

2. **Commit Window (1 epoch, ~72 minutes)** — Miners source leads matching the ICP, then submit **hashed lead data** (commit). This prevents other miners from copying your leads.

3. **Reveal Window (15 minutes after commit closes)** — Miners reveal the actual lead data corresponding to their committed hashes. Hashes must match or the submission is rejected.

4. **Validator Scoring** — Validators score every revealed lead through a multi-tier pipeline:
   - **Tier 1 (ICP Fit)** — Industry, sub-industry, role, seniority, employee count, company/contact country must match the request
   - **Tier 2 (Data Accuracy)** — Email verification (TrueList + ZeroBounce fallback), LinkedIn/person verification (ScrapingDog), company verification, reputation score
   - **Tier 2c (Required Attributes)** — Each `required_attributes` entry verified via Sonar web search (optionally grounded in miner-supplied `attribute_evidence` URLs)
   - **Tier 3 (Intent Scoring)** — Each intent signal URL is fetched and verified. An LLM evaluates relevance and that each signal maps to the ICP entry the miner claims (`matched_icp_signal`). Signals are scored and aggregated with time decay. Minimum threshold: 5.0

5. **Winner Selection & Rewards** — Leads are ranked by score, deduplicated by company. The top `num_leads` (as requested by the client) are selected as winners. Each winning lead earns **0.05% of emission per epoch for 100 epochs** (about 5 days of payout per lead). Ties on the same company split the reward. The longer runway is intentional: a single winning lead keeps a miner earning emission for ~5 days, protecting active miners from de-registration during low-volume windows.

### Fulfillment Request Schema (What Miners See)

When a client submits a request, miners receive an ICP with this structure:

```json
{
  "prompt": "VP of Sales and Heads of Revenue at Series A-C SaaS companies in the US showing signals of evaluating outbound sales tools, hiring SDRs, or researching competitors.",
  "industry": ["Software"],
  "sub_industry": ["SaaS", "Sales Automation"],
  "target_role_types": ["Sales", "Business Development"],
  "target_roles": ["VP of Sales", "Head of Revenue", "Director of Sales"],
  "target_seniority": "VP",
  "employee_count": ["51-200", "201-500"],
  "company_stage": "Series A",
  "company_country": ["United States"],
  "company_region": "",
  "contact_country": ["United States"],
  "contact_region": "California or Texas",
  "product_service": "outbound sales automation platform",
  "intent_signals": [
    "hiring SDRs",
    {"text": "evaluating sales tools", "required": true},
    {"text": "recent Series B or later", "required": false}
  ],
  "required_attributes": {
    "company": ["Has an active outbound sales motion"],
    "contact": ["Is a W-2 employee with hiring authority"]
  },
  "excluded_companies": ["AcmeCorp", "Competitor Inc"],
  "num_leads": 2
}
```

- `prompt` — Natural language description of the ideal lead. Your model should interpret this.
- `industry` / `sub_industry` / `employee_count` — Multi-valued. A lead matching ANY listed value passes Tier 1. Use canonical buckets for `employee_count` (`0-1`, `2-10`, `11-50`, `51-200`, `201-500`, `501-1,000`, `1,001-5,000`, `5,001-10,000`, `10,001+`).
- `company_country` / `company_region` — Filters on the lead's HQ. `contact_country` / `contact_region` — filters on the contact's own location (LinkedIn profile). Either side empty = not checked. Legacy `country` / `geography` keys still accepted (treated as company-side).
- `target_roles` — Exact role titles the client wants. Your lead's `role` must match one of these (fuzzy matching is applied, e.g. "VP, Corporate Sales" matches "VP of Sales").
- `target_seniority` — Required seniority level.
- `intent_signals` — Buying signals the client cares about. Each entry can be a plain string (default: optional) or a structured object `{"text", "required"}`. All credited signals contribute to the intent score; when `required=true`, the lead **must** produce verified evidence for that signal or it fails with `missing_required_intent_signal`.
- `required_attributes` — Buyer-side gate-keeping statements verified at Tier 2c. Two scopes: `company` (verified via Sonar web search) and `contact` (verified against the Apify LinkedIn data). A lead must satisfy every listed attribute for its scope or it fails with `required_attribute_failed`.
- `excluded_companies` — Companies the client doesn't want re-pitched. Hard-rejected at Tier 1 (case-insensitive `business` match). Pre-filter your search to avoid wasted work.
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
      "snippet": "Sales Development Representative - Full Time. We are looking for driven SDRs to join our growing sales team.",
      "matched_icp_signal": 0
    }
  ],

  "attribute_evidence": [
    {
      "scope": "company",
      "index": 0,
      "url": "https://company.com/blog/scaling-outbound",
      "snippet": "we've grown our SDR team from 4 to 14 reps over the past 6 months"
    }
  ]
}
```

**Key fields:**
- `city`/`state`/`country` — The **contact's** location (from their LinkedIn profile), not the company HQ
- `company_hq_city`/`company_hq_state`/`company_hq_country` — The **company's** headquarters location
- `industry`/`sub_industry` — Must match values from `gateway/utils/industry_taxonomy.py` (canonical) — `validator_models/industry_taxonomy.py` is a mirrored fallback
- `description` — **Required**, min 30 characters. A free-form company description written by the miner. Fed to the validator's Stage 5 3-stage classification pipeline (`validator_models/stage5_verification.py::classify_company_industry`): an LLM compares it against the scraped website/LinkedIn content; if the LLM decides the two don't describe the same business, the lead is rejected with `stage1_invalid_description` before intent scoring runs.
- `role_type` — One of: `C-Level Executive`, `VP`, `Director`, `Manager`, `Sales`, `Marketing`, `Engineering`, `Product`, `Operations`, `Finance`, `HR`, `Legal`, `IT`, `Customer Success`, `Business Development`, `Data & Analytics`, `Design`, `Research`, `Supply Chain`, `Consulting`, `Other`
- `seniority` — One of: `C-Suite`, `VP`, `Director`, `Manager`, `Individual Contributor`
- `intent_signals` — At least one signal required. Each signal needs `source`, `description`, `url`, `date` (ISO format or null), `snippet` (verbatim text from the URL), and **`matched_icp_signal`** (REQUIRED: zero-based integer index into the request's `icp_details.intent_signals` list of the client-listed signal this evidence proves; signals with `-1` or out-of-range values are rejected at Tier 3 scoring)
- `attribute_evidence` — OPTIONAL. Each entry pins a `(scope, index)` pair from the request's `required_attributes` to a `url` (and optional verbatim `snippet`). Supplying these lets the Tier 2c verifier read the URL directly via ScrapingDog instead of falling back to free-form Sonar search — strongly recommended for anti-bot URLs (LinkedIn jobs, paywalled news). Snippets that don't appear in the fetched page are flagged as fabricated and force a NO verdict.

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
| Tier 1 | Industry, sub-industry, role, seniority, company/contact country, employee count, duplicate company, `excluded_companies` | Free |
| Tier 2 | Email format, name-in-email, domain age, MX/SPF/DMARC, DNSBL, TrueList + ZeroBounce verification, LinkedIn person verification, company verification, reputation score | API calls |
| Tier 2c | Each `required_attributes` statement verified via Sonar web search, optionally grounded in miner-supplied `attribute_evidence` URLs | LLM + scraping |
| Tier 3 | Each intent signal URL scraped, snippet overlap verified, LLM evaluates relevance + `matched_icp_signal` mapping, time decay applied, peak-weighted aggregation, required-signal gate enforced | LLM + scraping |

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

The validation pipeline runs against fully enriched fulfillment leads.

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

### Fulfillment
- Each winning lead earns **0.4% of emission per epoch for 100 epochs** (~5 days of payout per winning lead), capped so the per-epoch pool never exceeds 90.5%.
- Top `num_leads` per request are selected; ties on the same company split the reward.

### Weekly Leaderboard
- A separate **9.5%** of miner emission funds a weekly leaderboard for total fulfillment wins.
- **#1 → 5.0%**, **#2 → 3.0%**, **#3 → 1.5%** of total emission.
- Rolling 140-epoch window (~7 days). Empty leaderboard slots burn to the treasury.

### Emission Split (Current)
- 0% sourcing · **90.5% fulfillment per-epoch pool** · **9.5% weekly leaderboard** (= 100%; the model-competition champion share was retired and folded into the fulfillment pool).

### Security Features

- **TEE Gateway**: All events logged through hardware-protected Trusted Execution Environment
- **Immutable transparency**: Events permanently stored on Arweave with cryptographic proofs
- **Commit/Reveal protocol**: Prevents miners from copying each other's fulfillment leads
- **Validator consensus**: Majority validator agreement, weighted by stake and v_trust, is required for fulfillment winner selection

## Data Flow

```
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
- Common causes: `tier1_role_mismatch`, `email_not_valid`, `company_geography_mismatch`, `contact_geography_mismatch`, `required_attribute_failed`, `missing_required_intent_signal`, `insufficient_intent`, `stage1_invalid_description`

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
