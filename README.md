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
# Email and LLM Validation Tools
export OPENROUTER_KEY="your_openrouter_key"          # Required for lead validation
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
  "linkedin": "https://linkedin.com/in/elonmusk",
  "description": "Aerospace manufacturer and space transportation company focused on reducing space transportation costs",
  "phone_numbers": ["+1-310-363-6000"],
  "founded_year": 2002,
  "ownership_type": "Private",
  "company_type": "Corporation",
  "number_of_locations": 5,
  "ids": {"crunchbase": "spacex"},
  "socials": {"twitter": "spacex"}
}
```

### Reward System

Miners earn rewards **proportional to approved leads** they source:
- If a miner sources 60% of approved leads in an epoch, they receive 60% of miner emissions for the following epoch
- Simple, transparent, and directly tied to value creation

### Rejection Feedback

When 2+ validators reject your lead, you'll receive detailed feedback explaining why. This helps you improve lead quality and increase approval rates.

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

**Common Rejections & Fixes:**

| Issue | Fix |
|-------|-----|
| Invalid email format | Verify email follows `name@domain.com` format |
| Email from disposable provider | Use business emails only (no tempmail, 10minutemail, etc.) |
| Domain too new (< 7 days) | Wait for domain to age or verify legitimacy |
| Email marked invalid | Check for typos, verify email exists |
| Website not accessible | Verify website is online and accessible |
| Domain blacklisted | Avoid domains flagged for spam/abuse |

**Validation Pipeline:** Leads are validated in 6 stages by validators - Terms Attestation → Source Provenance → Required Fields → DNS/Domain → Reputation → Email Deliverability. Validation stops at first failure.

**Security:** You can only see your own rejections (RLS enforced). Feedback only created when 2+ validators reject, preventing single bad-actor manipulation.

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

Validators pull prospects from the queue (first-come, first-served) and have a 15-second window to validate. With three validators participating, majority agreement is required for consensus. Approved leads move to the main database, rejected leads are discarded.

**Eligibility for Rewards:**
- Must participate in at least 10% of consensus decisions per epoch
- Verified server-side via Edge Function 
- If eligible, validators receive miner weights at end of epoch to commit on-chain

**Validators perform multi-stage quality checks:**
1. **Email validation**: Format, domain, disposable check, deliverability check
2. **Company & Contact verification**: Website, LinkedIn, Google search
3. **Reputation Score**: Coming soon

## Reward Distribution

### Epoch-Based Rewards

1. Validators check eligibility (> 10% consensus participation requirement)
2. Miner weights calculated based on sourced approved leads
3. Weights set on-chain proportional to leads sourced

### Security Features

- **Edge Function enforcement**: Eligibility checked server-side, cannot be bypassed
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

**"Not eligible - less than 10% consensus"**
- Validator needs to validate more prospects to meet eligibility threshold

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
