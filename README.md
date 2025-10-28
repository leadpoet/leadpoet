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

## Prerequisites

### Hardware Requirements
- **Miners/Validators**: 16GB RAM, 8-core CPU, 100GB SSD
- **Network**: Stable internet connection with open ports for axon communication

### Software Requirements
- Python 3.8+
- Bittensor CLI: `pip install bittensor>=6.9.3`
- Bittensor Wallet: `btcli wallet create`

## Required Credentials

### For Miners

```bash
# Required for Dynamic Lead Generation
export FIRECRAWL_API_KEY="your_firecrawl_key"        # Web scraping
export OPENROUTER_KEY="your_openrouter_key"          # AI classification
export GSE_API_KEY="your_google_api_key"             # Google Search
export GSE_CX="your_search_engine_id"                # Custom Search ID

```

### For Validators

```bash
# Email Validation API
export MYEMAILVERIFIER_API_KEY="your_mev_key"        # Email validation

```

## Installation

```bash
# Clone the repository
git clone https://github.com/leadpoet/Leadpoet.git
cd Leadpoet

# Install dependencies
pip install -e .

# Verify installation
python -c "import Leadpoet; print('Leadpoet installed successfully')"
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
    --netuid 71 \
    --subtensor_network finney
```

### How Miners Work

1. **Continuous Sourcing**: Actively search for new prospects
2. **Smart Submission**: Send prospects to Supabase `prospect_queue`
3. **Consensus Validation**: Prospects validated by validators
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
  "industry": "Aerospace Manufacturing",   # REQUIRED
  "sub_industry": "Space Transportation",  # REQUIRED
  "region": "Hawthorne, CA",               # REQUIRED
  "linkedin": "https://linkedin.com/in/elonmusk", # REQUIRED
  "description": "Aerospace manufacturer and space transportation company focused on reducing space transportation costs",
  "phone_numbers": ["+1-310-363-6000"],
  "founded_year": 2002,
  "ownership_type": "Private",
  "company_type": "Corporation",
  "number_of_locations": 5,
  "socials": {"twitter": "spacex"}
}
```

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

Miners earn rewards based on a **reputation-weighted formula** combining recent activity and historical performance:

**Weight Calculation (20% Recent + 80% Historical):**
- **20% from last 72 minutes**: Encourages consistent mining activity
- **80% from last 3 days**: Rewards sustained quality over time


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

**Daily Limits (Reset at 12:00 AM ET):**
- **10 submission attempts per day** - Counts all submission attempts
- **4 consensus rejections per day** - Only rejections from 2+ validators, or low reputation scores count toward this limit

**What Happens at 50 Rejections:**

```
50th Rejection → Cooldown Triggered → Your Pending Leads Deleted → Blocked Until 12 AM ET
                                    
```

When you hit 50 rejected leads in a day, the system automatically removes all your pending leads from the validation queue and blocks new submissions until the daily reset at midnight. You'll get a notification showing which lead was oldest in the queue when this happened, so you know where validation stopped.

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
    --netuid 71 \
    --subtensor_network finney
```

Note: Validators are configured to auto-update from GitHub on a 5-minute interval.

### Consensus Validation System

Validators pull prospects from the queue (first-come, first-served). With two - three validators participating, majority agreement is required for consensus. Approved leads move to the main database, rejected leads are discarded.

**Eligibility for Rewards:**
- Must participate in at least 5% of consensus decisions for last 24 hours
- Verified server-side via Edge Function 
- If eligible, validators receive miner weights to commit on-chain

**Validators perform multi-stage quality checks:**
1. **Email validation**: Format, domain, disposable check, deliverability check
2. **Company & Contact verification**: Website, LinkedIn, Google search

Validators must label leads with valid emails as "Valid" or "valid".

## Reward Distribution

### Consensus-Based Rewards

1. Validators check eligibility (≥ 5% consensus participation requirement for last 24 hours)
2. Miner weights calculated based on sourced approved leads
3. Weights set on-chain proportional to leads sourced

### Security Features

- **Edge Function enforcement**: Eligibility checked server-side
- **Server-side weight calculations**: Validators can't manipulate weights
- **JWT-based auth**: Validators only have limited database access
- **Consensus requirement**: No single validator can approve/reject leads

## Data Flow

```
Miner Sources Leads → Prospect Queue → Validators Run Quality Checks → 
Validator Consensus → Lead Pool → Curation for Buyer Requests (Month 2)
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

**"No JWT token available"**
- Validators need to wait for token generation on first run
- Token auto-refreshes every hour

**"Not eligible - less than 5% consensus"**
- Validator needs to validate more prospects to meet eligibility threshold (5% of leads in last 24 hours)

**"Prospect already in queue"**
- The prospect attempted to be submitted is already in the prospect queue

**Consensus not reached after 15 seconds**
- Insufficient number of validators ran quality checks on the lead to reach consensus

## Support

For support and discussion:
- **Leadpoet FAQ**: Check out our FAQ at www.leadpoet.com/faq to learn more about Leadpoet!
- **Bittensor Discord**: Join the Leadpoet SN71 channel and message us!
- **Email**: hello@leadpoet.com

## License

MIT License - See LICENSE file for details


