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
  "business": "SpaceX",                    # REQUIRED
  "full_name": "Elon Musk",                # REQUIRED
  "first": "Elon",                         # REQUIRED
  "last": "Musk",                          # REQUIRED
  "email": "elon@spacex.com",              # REQUIRED
  "role": "CEO",                           # REQUIRED
  "website": "https://spacex.com",         # REQUIRED
  "industry": "Science and Engineering",   # REQUIRED - must be from industry_taxonomy.py
  "sub_industry": "Aerospace",             # REQUIRED - must be from industry_taxonomy.py
  "country": "United States",              # REQUIRED - see Country Format below
  "state": "California",                   # REQUIRED for US leads only
  "city": "Hawthorne",                     # REQUIRED for all leads
  "linkedin": "https://linkedin.com/in/elonmusk", # REQUIRED
  "company_linkedin": "https://linkedin.com/company/spacex", # REQUIRED
  "source_url": "https://spacex.com/careers", # REQUIRED (URL where lead was found, OR "proprietary_database")
  "description": "Aerospace manufacturer focused on reducing space transportation costs", # REQUIRED
  "employee_count": "1,001-5,000",         # REQUIRED - valid ranges: "0-1", "2-10", "11-50", "51-200", "201-500", "501-1,000", "1,001-5,000", "5,001-10,000", "10,001+"
  "source_type": "company_site",
  "phone_numbers": ["+1-310-363-6000"],
  "founded_year": 2002,
  "ownership_type": "Private",
  "company_type": "Corporation",
  "number_of_locations": 5,
  "socials": {"twitter": "spacex"}
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

**Daily Limits (Reset at 12:00 AM EST):**
- **10 submission attempts per day** - Counts all submission attempts (including duplicates/invalid)
- **8 rejections per day** - Includes:
  - Duplicate submissions
  - Missing required fields
  - **Validator consensus rejections** - When validator consensus rejects your lead based on quality checks

**What Happens at Rate Limit:**
```
When you hit the rejection limit, all subsequent submissions are blocked until the daily reset at midnight EST. All rate limit events are logged to the TEE buffer and permanently stored on Arweave for transparency.

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

### Community Audit Tool

The `leadpoet-audit` CLI allows anyone to verify validation outcomes by querying public transparency logs:

```bash
# Install
pip install -e .

# Generate audit report for epoch
leadpoet-audit report 19000

# Save report to JSON
leadpoet-audit report 19000 --output report.json

# Query transparency logs by date, hours, or lead UUID (outputs ALL database fields)
leadpoet-audit logs --date 2025-11-14 --output report.json
leadpoet-audit logs --hours 4 --output report.json
leadpoet-audit logs --lead-id 8183c849-c017-4f4c-b9fe-7f407873a799 --output report.json
```

The audit tool queries **public data only** (transparency log) and shows consensus results, rejection reasons, and miner performance statistics.

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

## Roadmap

### Month 1: Launch & Foundation
- Codebase goes live on SN71
- Refine sourcing; gatekeep low-quality leads from the DB
- Ensure stable miner and validator operations
- Monitor and optimize consensus validation

### Month 2: Curation & Beta Users
- Miners begin curating leads from the lead pool based on Ideal Customer Profiles (ICPs)
- Implement curation rewards into the incentive mechanism
- Onboard initial beta customers for feedback
- Refine models for lead ranking and scoring

### Month 3: Product Launch & Growth
- Product launch with marketing and sales campaigns
- Open Leadpoet platform to paying customers
- Scale miner curation and sourcing capabilities
- Introduce weekly ICP themes to incentivize sourcing leads in specific industries
- Optimize end-to-end lead generation pipeline

## Troubleshooting

Common Errors:

**Validator not receiving epoch assignments**
- Ensure validator is registered on subnet with active stake
- Check that validator is running latest code version (auto-updates every 5 minutes)
- Verify axon is accessible and ports are open

**Lead submission rejected**
- Check lead meets all requirements (valid email, name-email matching, required fields)
- Verify you haven't hit daily rate limits (10 submissions, 8 rejections per day)
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


